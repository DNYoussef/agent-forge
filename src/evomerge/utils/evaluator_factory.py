"""
EvaluatorFactory - Consolidated factory for fitness evaluator creation.

Eliminates duplicate evaluator creation patterns found in:
- COA-004 violation: 3x duplicate evaluator creation patterns
- real_fitness_evaluator.py: create_classification_evaluator, create_language_model_evaluator, create_efficiency_evaluator

This factory standardizes evaluator creation across the EvoMerge system.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Union, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EvaluatorType(Enum):
    """Types of fitness evaluators available."""
    CLASSIFICATION = "classification"
    LANGUAGE_MODEL = "language_model"
    EFFICIENCY = "efficiency"
    MULTI_OBJECTIVE = "multi_objective"
    CUSTOM = "custom"
    COMPOSITE = "composite"


class MetricType(Enum):
    """Types of metrics for evaluation."""
    ACCURACY = "accuracy"
    PERPLEXITY = "perplexity"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    BLEU = "bleu"
    ROUGE = "rouge"
    LOSS = "loss"
    EFFICIENCY = "efficiency"
    INFERENCE_TIME = "inference_time"
    MEMORY_USAGE = "memory_usage"
    PARAMETER_COUNT = "parameter_count"


@dataclass
class EvaluatorConfig:
    """Configuration for evaluator creation."""
    evaluator_type: EvaluatorType
    metric_types: List[MetricType]
    weights: Optional[Dict[MetricType, float]] = None
    batch_size: int = 32
    device: str = "auto"
    timeout_seconds: float = 300.0
    enable_caching: bool = True
    custom_metrics: Optional[Dict[str, Callable]] = None


class BaseEvaluator(ABC):
    """
    Base class for all fitness evaluators.

    Provides standardized interface and common functionality for
    all evaluator implementations in the EvoMerge system.
    """

    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self.device = self._resolve_device(config.device)
        self.metrics_cache: Dict[str, Any] = {}
        self._evaluation_history: List[Dict[str, Any]] = []

        logger.info(f"Initialized {self.__class__.__name__} with {len(config.metric_types)} metrics")

    def _resolve_device(self, device_spec: str) -> torch.device:
        """Resolve device specification to torch.device."""
        if device_spec == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(device_spec)

    @abstractmethod
    def evaluate(self, model: nn.Module, data: Any) -> Dict[MetricType, float]:
        """
        Evaluate model performance.

        Args:
            model: Model to evaluate
            data: Evaluation data

        Returns:
            Dictionary mapping metric types to scores
        """
        pass

    def batch_evaluate(self, models: List[nn.Module], data: Any) -> List[Dict[MetricType, float]]:
        """
        Evaluate multiple models efficiently.

        Consolidates duplicate batch evaluation patterns found across operators.
        """
        results = []
        for model in models:
            try:
                result = self.evaluate(model, data)
                results.append(result)
                self._evaluation_history.append({
                    "model_id": id(model),
                    "metrics": result,
                    "timestamp": torch.tensor(0).item()  # Simple timestamp
                })
            except Exception as e:
                logger.error(f"Evaluation failed for model {id(model)}: {e}")
                results.append({metric: 0.0 for metric in self.config.metric_types})

        return results

    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get history of evaluations performed."""
        return self._evaluation_history.copy()

    def clear_cache(self):
        """Clear evaluation cache."""
        self.metrics_cache.clear()
        logger.debug("Evaluation cache cleared")


class ClassificationEvaluator(BaseEvaluator):
    """
    Evaluator for classification models.

    Consolidates create_classification_evaluator pattern from
    real_fitness_evaluator.py lines 499-513.
    """

    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        self.num_classes = config.custom_metrics.get("num_classes", 10) if config.custom_metrics else 10

    def evaluate(self, model: nn.Module, data: Any) -> Dict[MetricType, float]:
        """Evaluate classification model."""
        model.eval()
        model = model.to(self.device)

        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        # Support for different data formats
        if isinstance(data, torch.utils.data.DataLoader):
            dataloader = data
        else:
            # Assume it's a tensor pair (inputs, targets)
            inputs, targets = data
            dataloader = [(inputs, targets)]

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch_inputs, batch_targets in dataloader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)

                _, predicted = torch.max(outputs.data, 1)
                total_samples += batch_targets.size(0)
                total_correct += (predicted == batch_targets).sum().item()
                total_loss += loss.item()

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')

        return {
            MetricType.ACCURACY: accuracy,
            MetricType.LOSS: avg_loss
        }


class LanguageModelEvaluator(BaseEvaluator):
    """
    Evaluator for language models.

    Consolidates create_language_model_evaluator pattern from
    real_fitness_evaluator.py lines 515-529.
    """

    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        self.max_sequence_length = config.custom_metrics.get("max_seq_len", 512) if config.custom_metrics else 512

    def evaluate(self, model: nn.Module, data: Any) -> Dict[MetricType, float]:
        """Evaluate language model."""
        model.eval()
        model = model.to(self.device)

        total_loss = 0.0
        total_tokens = 0

        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        with torch.no_grad():
            if isinstance(data, torch.utils.data.DataLoader):
                for batch in data:
                    if len(batch) == 2:
                        input_ids, labels = batch
                    else:
                        input_ids = batch["input_ids"]
                        labels = batch["labels"]

                    input_ids = input_ids.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(input_ids)

                    # Handle different output formats
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs

                    # Reshape for loss calculation
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                    total_loss += loss.item()
                    total_tokens += (shift_labels != -100).sum().item()

        avg_loss = total_loss / len(data) if len(data) > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            MetricType.LOSS: avg_loss,
            MetricType.PERPLEXITY: perplexity
        }


class EfficiencyEvaluator(BaseEvaluator):
    """
    Evaluator for model efficiency metrics.

    Consolidates create_efficiency_evaluator pattern from
    real_fitness_evaluator.py lines 531-543.
    """

    def __init__(self, config: EvaluatorConfig):
        super().__init__(config)
        self.warmup_runs = config.custom_metrics.get("warmup_runs", 5) if config.custom_metrics else 5
        self.timing_runs = config.custom_metrics.get("timing_runs", 20) if config.custom_metrics else 20

    def evaluate(self, model: nn.Module, data: Any) -> Dict[MetricType, float]:
        """Evaluate model efficiency."""
        model.eval()
        model = model.to(self.device)

        # Parameter count
        param_count = sum(p.numel() for p in model.parameters())

        # Memory usage estimation
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        memory_mb = param_memory / (1024 * 1024)

        # Inference time measurement
        if isinstance(data, torch.Tensor):
            sample_input = data[:1].to(self.device)  # Use first sample
        else:
            # Create dummy input based on expected shape
            sample_input = torch.randn(1, 3, 224, 224).to(self.device)  # Default image-like input

        # Warmup runs
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(sample_input)

        # Timing runs
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        import time
        start_time = time.time()

        with torch.no_grad():
            for _ in range(self.timing_runs):
                _ = model(sample_input)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        avg_inference_time = (end_time - start_time) / self.timing_runs * 1000  # ms

        return {
            MetricType.PARAMETER_COUNT: float(param_count),
            MetricType.MEMORY_USAGE: memory_mb,
            MetricType.INFERENCE_TIME: avg_inference_time
        }


class CompositeEvaluator(BaseEvaluator):
    """
    Composite evaluator that combines multiple evaluation metrics.

    Enables multi-objective optimization for EvoMerge Phase 2.
    """

    def __init__(self, config: EvaluatorConfig, evaluators: List[BaseEvaluator]):
        super().__init__(config)
        self.evaluators = evaluators
        self.weights = config.weights or {}

    def evaluate(self, model: nn.Module, data: Any) -> Dict[MetricType, float]:
        """Evaluate using all component evaluators."""
        combined_results = {}

        for evaluator in self.evaluators:
            try:
                results = evaluator.evaluate(model, data)
                combined_results.update(results)
            except Exception as e:
                logger.error(f"Evaluator {evaluator.__class__.__name__} failed: {e}")

        # Apply weights if specified
        if self.weights:
            weighted_results = {}
            for metric, value in combined_results.items():
                weight = self.weights.get(metric, 1.0)
                weighted_results[metric] = value * weight
            return weighted_results

        return combined_results

    def get_composite_score(self, metrics: Dict[MetricType, float]) -> float:
        """Calculate single composite fitness score."""
        if not metrics:
            return 0.0

        # Simple weighted sum with normalization
        total_score = 0.0
        total_weight = 0.0

        for metric, value in metrics.items():
            weight = self.weights.get(metric, 1.0)

            # Normalize different metric types to [0, 1] range
            if metric in [MetricType.ACCURACY, MetricType.F1_SCORE, MetricType.PRECISION, MetricType.RECALL]:
                normalized_value = max(0.0, min(1.0, value))
            elif metric in [MetricType.LOSS, MetricType.PERPLEXITY]:
                # Lower is better, so invert
                normalized_value = 1.0 / (1.0 + value)
            elif metric in [MetricType.INFERENCE_TIME, MetricType.MEMORY_USAGE]:
                # Lower is better, normalize by typical ranges
                max_time = self.config.custom_metrics.get("max_inference_time", 1000.0) if self.config.custom_metrics else 1000.0
                normalized_value = max(0.0, 1.0 - (value / max_time))
            else:
                normalized_value = value

            total_score += normalized_value * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0


class EvaluatorFactory:
    """
    Factory for creating standardized fitness evaluators.

    Eliminates all duplicate evaluator creation patterns found in the
    Phase 2 duplication analysis (COA-004 violation).
    """

    _registry: Dict[EvaluatorType, Type[BaseEvaluator]] = {
        EvaluatorType.CLASSIFICATION: ClassificationEvaluator,
        EvaluatorType.LANGUAGE_MODEL: LanguageModelEvaluator,
        EvaluatorType.EFFICIENCY: EfficiencyEvaluator,
        EvaluatorType.MULTI_OBJECTIVE: CompositeEvaluator,  # Use composite for multi-objective
        EvaluatorType.COMPOSITE: CompositeEvaluator,
        EvaluatorType.CUSTOM: EfficiencyEvaluator  # Default fallback
    }

    @classmethod
    def register_evaluator(cls, evaluator_type: EvaluatorType, evaluator_class: Type[BaseEvaluator]):
        """Register custom evaluator type."""
        cls._registry[evaluator_type] = evaluator_class
        logger.info(f"Registered evaluator: {evaluator_type.value} -> {evaluator_class.__name__}")

    @classmethod
    def create_evaluator(cls, config: EvaluatorConfig, **kwargs) -> BaseEvaluator:
        """
        Create evaluator instance based on configuration.

        This method consolidates all duplicate evaluator creation patterns
        found in the Phase 2 duplication report.

        Args:
            config: Evaluator configuration
            **kwargs: Additional arguments for specific evaluator types

        Returns:
            Configured evaluator instance

        Raises:
            ValueError: If evaluator type is not supported
        """
        if config.evaluator_type not in cls._registry:
            raise ValueError(f"Unsupported evaluator type: {config.evaluator_type}")

        evaluator_class = cls._registry[config.evaluator_type]

        try:
            if config.evaluator_type == EvaluatorType.COMPOSITE:
                # Special handling for composite evaluator
                component_evaluators = kwargs.get("evaluators", [])
                return evaluator_class(config, component_evaluators)
            else:
                return evaluator_class(config)

        except Exception as e:
            logger.error(f"Failed to create evaluator {config.evaluator_type.value}: {e}")
            raise

    @classmethod
    def create_classification_evaluator(cls, num_classes: int = 10, **kwargs) -> ClassificationEvaluator:
        """
        Create classification evaluator.

        Replaces duplicate create_classification_evaluator implementations.
        """
        config = EvaluatorConfig(
            evaluator_type=EvaluatorType.CLASSIFICATION,
            metric_types=[MetricType.ACCURACY, MetricType.LOSS],
            custom_metrics={"num_classes": num_classes},
            **kwargs
        )
        return cls.create_evaluator(config)

    @classmethod
    def create_language_model_evaluator(cls, max_seq_len: int = 512, **kwargs) -> LanguageModelEvaluator:
        """
        Create language model evaluator.

        Replaces duplicate create_language_model_evaluator implementations.
        """
        config = EvaluatorConfig(
            evaluator_type=EvaluatorType.LANGUAGE_MODEL,
            metric_types=[MetricType.LOSS, MetricType.PERPLEXITY],
            custom_metrics={"max_seq_len": max_seq_len},
            **kwargs
        )
        return cls.create_evaluator(config)

    @classmethod
    def create_efficiency_evaluator(cls, warmup_runs: int = 5, timing_runs: int = 20, **kwargs) -> EfficiencyEvaluator:
        """
        Create efficiency evaluator.

        Replaces duplicate create_efficiency_evaluator implementations.
        """
        config = EvaluatorConfig(
            evaluator_type=EvaluatorType.EFFICIENCY,
            metric_types=[MetricType.PARAMETER_COUNT, MetricType.MEMORY_USAGE, MetricType.INFERENCE_TIME],
            custom_metrics={"warmup_runs": warmup_runs, "timing_runs": timing_runs},
            **kwargs
        )
        return cls.create_evaluator(config)

    @classmethod
    def create_multi_objective_evaluator(cls,
                                       evaluator_configs: List[EvaluatorConfig],
                                       weights: Optional[Dict[MetricType, float]] = None) -> CompositeEvaluator:
        """
        Create multi-objective evaluator for Phase 2 EvoMerge.

        Combines multiple evaluation criteria for comprehensive model assessment.
        """
        # Create component evaluators
        component_evaluators = []
        for comp_config in evaluator_configs:
            evaluator = cls.create_evaluator(comp_config)
            component_evaluators.append(evaluator)

        # Create composite configuration
        all_metrics = []
        for config in evaluator_configs:
            all_metrics.extend(config.metric_types)

        composite_config = EvaluatorConfig(
            evaluator_type=EvaluatorType.COMPOSITE,
            metric_types=list(set(all_metrics)),  # Remove duplicates
            weights=weights
        )

        return cls.create_evaluator(composite_config, evaluators=component_evaluators)

    @classmethod
    def get_supported_types(cls) -> List[EvaluatorType]:
        """Get list of supported evaluator types."""
        return list(cls._registry.keys())


# Convenience functions for backward compatibility
def create_classification_evaluator(**kwargs) -> ClassificationEvaluator:
    """Convenience function for classification evaluator creation."""
    return EvaluatorFactory.create_classification_evaluator(**kwargs)


def create_language_model_evaluator(**kwargs) -> LanguageModelEvaluator:
    """Convenience function for language model evaluator creation."""
    return EvaluatorFactory.create_language_model_evaluator(**kwargs)


def create_efficiency_evaluator(**kwargs) -> EfficiencyEvaluator:
    """Convenience function for efficiency evaluator creation."""
    return EvaluatorFactory.create_efficiency_evaluator(**kwargs)