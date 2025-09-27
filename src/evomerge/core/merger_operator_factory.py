"""
MergerOperatorFactory - 3→8 Model Creation Pipeline for Phase 2 EvoMerge.

Implements the user's specification:
- Takes 3 models from Phase 1 or current generation
- Creates 8 diverse models using different merger techniques
- Supports SLERP, TIES, DARE, and custom weighted combinations
- Provides lineage tracking for 3D visualization
- Thread-safe operations for concurrent processing

Factory Pattern Design:
- Centralized merger orchestration
- Configurable technique selection
- Extensible for new merger types
- Automatic diversity optimization
"""

import torch
import torch.nn as nn
import numpy as np
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
import logging
import random
from abc import ABC, abstractmethod

from ..operators.slerp_operator import SLERPOperator, create_slerp_operator
from ..operators.ties_operator import TIESOperator, create_ties_operator, TaskConfig
from ..operators.dare_operator import DAREOperator, create_dare_operator
from ..utils.model_operations import get_model_operations, clone_model
from ..utils.evaluator_factory import EvaluatorFactory, EvaluatorConfig, EvaluatorType, MetricType

logger = logging.getLogger(__name__)


class MergerTechnique(Enum):
    """Available merger techniques."""
    SLERP = "slerp"
    TIES = "ties"
    DARE = "dare"
    WEIGHTED_AVERAGE = "weighted_average"
    ARITHMETIC_MEAN = "arithmetic_mean"
    GEOMETRIC_MEAN = "geometric_mean"
    LERP = "lerp"
    CUSTOM = "custom"


class DiversityStrategy(Enum):
    """Strategies for ensuring model diversity."""
    RANDOM_PARAMS = "random_params"
    TECHNIQUE_ROTATION = "technique_rotation"
    PARAMETER_SCALING = "parameter_scaling"
    NOISE_INJECTION = "noise_injection"
    HYBRID_COMBINATION = "hybrid_combination"


@dataclass
class MergerConfig:
    """Configuration for a single merger operation."""
    technique: MergerTechnique
    parameters: Dict[str, Any]
    weight_models: Optional[List[float]] = None
    diversity_factor: float = 1.0
    enable_validation: bool = True


@dataclass
class MergerResult:
    """Result of merger operation."""
    merged_model: nn.Module
    technique: MergerTechnique
    config: MergerConfig
    source_models: List[str]  # Model IDs
    diversity_score: float
    quality_metrics: Dict[str, float]
    lineage_info: Dict[str, Any]


class MergerOperatorFactory:
    """
    Factory for creating 8 diverse models from 3 input models.

    Implements Phase 2 EvoMerge specification by orchestrating multiple
    merger techniques to create maximum diversity while maintaining quality.
    """

    def __init__(self,
                 diversity_strategy: DiversityStrategy = DiversityStrategy.TECHNIQUE_ROTATION,
                 enable_quality_validation: bool = True,
                 max_concurrent_operations: int = 4):
        """
        Initialize MergerOperatorFactory.

        Args:
            diversity_strategy: Strategy for ensuring model diversity
            enable_quality_validation: Whether to validate merged models
            max_concurrent_operations: Max concurrent merger operations
        """
        self.diversity_strategy = diversity_strategy
        self.enable_quality_validation = enable_quality_validation
        self.max_concurrent_operations = max_concurrent_operations

        # Initialize operators
        self.operators = {
            MergerTechnique.SLERP: None,  # Lazy initialization
            MergerTechnique.TIES: None,
            MergerTechnique.DARE: None
        }

        # Model operations
        self.model_ops = get_model_operations()

        # Thread safety
        self._operation_semaphore = threading.Semaphore(max_concurrent_operations)
        self._operator_lock = threading.Lock()

        # Diversity tracking
        self._diversity_cache: Dict[str, float] = {}

        # Quality evaluator (optional)
        self.quality_evaluator = None
        if enable_quality_validation:
            try:
                evaluator_config = EvaluatorConfig(
                    evaluator_type=EvaluatorType.EFFICIENCY,
                    metric_types=[MetricType.PARAMETER_COUNT, MetricType.MEMORY_USAGE]
                )
                self.quality_evaluator = EvaluatorFactory.create_evaluator(evaluator_config)
            except Exception as e:
                logger.warning(f"Failed to initialize quality evaluator: {e}")

        logger.info(f"MergerOperatorFactory initialized with {diversity_strategy.value} diversity strategy")

    def _get_operator(self, technique: MergerTechnique):
        """Get operator instance for technique (thread-safe lazy initialization)."""
        with self._operator_lock:
            if technique not in self.operators or self.operators[technique] is None:
                if technique == MergerTechnique.SLERP:
                    self.operators[technique] = create_slerp_operator()
                elif technique == MergerTechnique.TIES:
                    # Create with default task configs
                    default_tasks = [
                        TaskConfig("default", 1.0)
                    ]
                    self.operators[technique] = create_ties_operator(default_tasks)
                elif technique == MergerTechnique.DARE:
                    self.operators[technique] = create_dare_operator()
                else:
                    raise ValueError(f"Unsupported technique: {technique}")

            return self.operators[technique]

    def create_8_from_3_models(self,
                              input_models: List[nn.Module],
                              generation: int = 1,
                              parent_lineages: Optional[List[str]] = None) -> List[MergerResult]:
        """
        Create 8 diverse models from 3 input models.

        This is the core Phase 2 EvoMerge functionality that takes 3 models
        (from Phase 1 or previous generation) and creates 8 diverse variants
        using different merger techniques.

        Args:
            input_models: 3 input models
            generation: Generation number for tracking
            parent_lineages: Parent lineage colors for 3D visualization

        Returns:
            List of 8 MergerResult objects

        Raises:
            ValueError: If input doesn't contain exactly 3 models
        """
        if len(input_models) != 3:
            raise ValueError(f"Expected exactly 3 input models, got {len(input_models)}")

        logger.info(f"Creating 8 models from 3 inputs for generation {generation}")

        # Define 8 diverse merger configurations
        merger_configs = self._generate_diverse_configs(input_models, generation)

        # Create models using different techniques
        results = []

        # Use threading for concurrent operations (but limit to avoid memory issues)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent_operations) as executor:
            futures = []

            for i, config in enumerate(merger_configs):
                future = executor.submit(
                    self._create_single_merged_model,
                    input_models,
                    config,
                    i,
                    generation,
                    parent_lineages
                )
                futures.append(future)

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Merger operation failed: {e}")

        # Ensure we have exactly 8 results
        if len(results) < 8:
            logger.warning(f"Only created {len(results)} models, expected 8. Creating fallbacks...")
            results.extend(self._create_fallback_models(input_models, 8 - len(results), generation))

        return results[:8]  # Ensure exactly 8 models

    def _generate_diverse_configs(self, input_models: List[nn.Module], generation: int) -> List[MergerConfig]:
        """Generate 8 diverse merger configurations."""
        configs = []

        if self.diversity_strategy == DiversityStrategy.TECHNIQUE_ROTATION:
            # Strategy 1: Rotate through techniques with different parameters
            base_configs = [
                # SLERP variations
                MergerConfig(
                    technique=MergerTechnique.SLERP,
                    parameters={"t": 0.3},
                    weight_models=[0.7, 0.3, 0.0]
                ),
                MergerConfig(
                    technique=MergerTechnique.SLERP,
                    parameters={"t": 0.7},
                    weight_models=[0.3, 0.7, 0.0]
                ),
                # TIES variations
                MergerConfig(
                    technique=MergerTechnique.TIES,
                    parameters={"threshold": 0.2, "conflict_resolution": "majority_vote"},
                    weight_models=[0.4, 0.4, 0.2]
                ),
                MergerConfig(
                    technique=MergerTechnique.TIES,
                    parameters={"threshold": 0.5, "conflict_resolution": "weighted_average"},
                    weight_models=[0.3, 0.3, 0.4]
                ),
                # DARE variations
                MergerConfig(
                    technique=MergerTechnique.DARE,
                    parameters={"dropout_rate": 0.1, "rescale_method": "magnitude_preserving"},
                    weight_models=[0.5, 0.3, 0.2]
                ),
                MergerConfig(
                    technique=MergerTechnique.DARE,
                    parameters={"dropout_rate": 0.3, "rescale_method": "variance_preserving"},
                    weight_models=[0.2, 0.5, 0.3]
                ),
                # Weighted average variations
                MergerConfig(
                    technique=MergerTechnique.WEIGHTED_AVERAGE,
                    parameters={},
                    weight_models=[0.5, 0.3, 0.2]
                ),
                MergerConfig(
                    technique=MergerTechnique.WEIGHTED_AVERAGE,
                    parameters={},
                    weight_models=[0.2, 0.6, 0.2]
                )
            ]

        elif self.diversity_strategy == DiversityStrategy.RANDOM_PARAMS:
            # Strategy 2: Random parameter variations
            techniques = [MergerTechnique.SLERP, MergerTechnique.TIES, MergerTechnique.DARE, MergerTechnique.WEIGHTED_AVERAGE]
            base_configs = []

            for i in range(8):
                technique = techniques[i % len(techniques)]

                if technique == MergerTechnique.SLERP:
                    config = MergerConfig(
                        technique=technique,
                        parameters={"t": random.uniform(0.2, 0.8)},
                        weight_models=self._random_weights(3)
                    )
                elif technique == MergerTechnique.TIES:
                    config = MergerConfig(
                        technique=technique,
                        parameters={
                            "threshold": random.uniform(0.1, 0.6),
                            "conflict_resolution": random.choice(["majority_vote", "weighted_average"])
                        },
                        weight_models=self._random_weights(3)
                    )
                elif technique == MergerTechnique.DARE:
                    config = MergerConfig(
                        technique=technique,
                        parameters={
                            "dropout_rate": random.uniform(0.05, 0.4),
                            "rescale_method": random.choice(["magnitude_preserving", "variance_preserving"])
                        },
                        weight_models=self._random_weights(3)
                    )
                else:  # WEIGHTED_AVERAGE
                    config = MergerConfig(
                        technique=technique,
                        parameters={},
                        weight_models=self._random_weights(3)
                    )

                base_configs.append(config)

        else:
            # Default: balanced technique distribution
            base_configs = [
                MergerConfig(MergerTechnique.SLERP, {"t": 0.3}, [0.6, 0.4, 0.0]),
                MergerConfig(MergerTechnique.SLERP, {"t": 0.7}, [0.4, 0.6, 0.0]),
                MergerConfig(MergerTechnique.TIES, {"threshold": 0.3}, [0.5, 0.3, 0.2]),
                MergerConfig(MergerTechnique.TIES, {"threshold": 0.4}, [0.3, 0.5, 0.2]),
                MergerConfig(MergerTechnique.DARE, {"dropout_rate": 0.15}, [0.4, 0.3, 0.3]),
                MergerConfig(MergerTechnique.DARE, {"dropout_rate": 0.25}, [0.3, 0.4, 0.3]),
                MergerConfig(MergerTechnique.WEIGHTED_AVERAGE, {}, [0.5, 0.3, 0.2]),
                MergerConfig(MergerTechnique.WEIGHTED_AVERAGE, {}, [0.2, 0.3, 0.5])
            ]

        # Add diversity factors
        for i, config in enumerate(base_configs):
            config.diversity_factor = 1.0 + (i * 0.1)  # Slight diversity boost
            configs.append(config)

        return configs

    def _random_weights(self, count: int) -> List[float]:
        """Generate random normalized weights."""
        weights = [random.random() for _ in range(count)]
        total = sum(weights)
        return [w / total for w in weights]

    def _create_single_merged_model(self,
                                   input_models: List[nn.Module],
                                   config: MergerConfig,
                                   model_index: int,
                                   generation: int,
                                   parent_lineages: Optional[List[str]] = None) -> Optional[MergerResult]:
        """Create a single merged model using the specified configuration."""
        with self._operation_semaphore:
            try:
                logger.debug(f"Creating model {model_index} using {config.technique.value}")

                # Select and execute merger technique
                if config.technique in [MergerTechnique.SLERP, MergerTechnique.TIES, MergerTechnique.DARE]:
                    merged_model = self._use_operator_technique(input_models, config)
                elif config.technique == MergerTechnique.WEIGHTED_AVERAGE:
                    merged_model = self._weighted_average_merge(input_models, config.weight_models or [1/3, 1/3, 1/3])
                elif config.technique == MergerTechnique.ARITHMETIC_MEAN:
                    merged_model = self._arithmetic_mean_merge(input_models)
                else:
                    raise ValueError(f"Unsupported technique: {config.technique}")

                # Calculate diversity score
                diversity_score = self._calculate_diversity_score(merged_model, input_models)

                # Quality validation
                quality_metrics = {}
                if self.quality_evaluator:
                    try:
                        dummy_data = torch.randn(1, 3, 224, 224)  # Placeholder
                        quality_metrics = self.quality_evaluator.evaluate(merged_model, dummy_data)
                    except Exception as e:
                        logger.warning(f"Quality evaluation failed: {e}")

                # Lineage tracking
                lineage_info = {
                    "generation": generation,
                    "model_index": model_index,
                    "parent_count": len(input_models),
                    "technique": config.technique.value,
                    "parameters": config.parameters,
                    "parent_lineages": parent_lineages or []
                }

                return MergerResult(
                    merged_model=merged_model,
                    technique=config.technique,
                    config=config,
                    source_models=[f"model_{i}" for i in range(len(input_models))],
                    diversity_score=diversity_score,
                    quality_metrics=quality_metrics,
                    lineage_info=lineage_info
                )

            except Exception as e:
                logger.error(f"Failed to create model {model_index}: {e}")
                return None

    def _use_operator_technique(self, input_models: List[nn.Module], config: MergerConfig) -> nn.Module:
        """Use specific operator technique to merge models."""
        operator = self._get_operator(config.technique)

        if config.technique == MergerTechnique.SLERP:
            # SLERP between first two models, then with third
            t = config.parameters.get("t", 0.5)
            intermediate = operator.interpolate(input_models[0], input_models[1], t)

            if len(input_models) > 2:
                # Incorporate third model
                weights = config.weight_models or [0.5, 0.5]
                if len(weights) > 2:
                    t2 = weights[2]
                    return operator.interpolate(intermediate, input_models[2], t2)

            return intermediate

        elif config.technique == MergerTechnique.TIES:
            # TIES merge all models
            return operator.merge(input_models)

        elif config.technique == MergerTechnique.DARE:
            # DARE merge all models
            return operator.merge_models(input_models)

        else:
            raise ValueError(f"Operator technique not implemented: {config.technique}")

    def _weighted_average_merge(self, models: List[nn.Module], weights: List[float]) -> nn.Module:
        """Perform weighted average merge of models."""
        if len(weights) != len(models):
            raise ValueError(f"Weight count {len(weights)} doesn't match model count {len(models)}")

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Clone first model as base
        merged_model = clone_model(models[0])

        # Weighted combination of parameters
        merged_params = dict(merged_model.named_parameters())
        model_params = [dict(model.named_parameters()) for model in models]

        for name in merged_params:
            if all(name in params for params in model_params):
                # Weighted combination
                weighted_param = torch.zeros_like(merged_params[name])
                for i, (weight, params) in enumerate(zip(weights, model_params)):
                    weighted_param += weight * params[name].data

                merged_params[name].data.copy_(weighted_param)

        return merged_model

    def _arithmetic_mean_merge(self, models: List[nn.Module]) -> nn.Module:
        """Perform simple arithmetic mean merge."""
        weights = [1.0 / len(models)] * len(models)
        return self._weighted_average_merge(models, weights)

    def _calculate_diversity_score(self, merged_model: nn.Module, input_models: List[nn.Module]) -> float:
        """Calculate diversity score of merged model relative to inputs."""
        try:
            total_distance = 0.0
            for input_model in input_models:
                distance = self.model_ops.calculate_model_distance(merged_model, input_model)
                total_distance += distance

            return total_distance / len(input_models)

        except Exception as e:
            logger.warning(f"Failed to calculate diversity score: {e}")
            return 0.0

    def _create_fallback_models(self, input_models: List[nn.Module], count: int, generation: int) -> List[MergerResult]:
        """Create fallback models if primary creation fails."""
        fallback_results = []

        for i in range(count):
            try:
                # Simple weighted average with random weights
                weights = self._random_weights(len(input_models))
                merged_model = self._weighted_average_merge(input_models, weights)

                result = MergerResult(
                    merged_model=merged_model,
                    technique=MergerTechnique.WEIGHTED_AVERAGE,
                    config=MergerConfig(MergerTechnique.WEIGHTED_AVERAGE, {}, weights),
                    source_models=[f"fallback_{i}"],
                    diversity_score=0.5,  # Default score
                    quality_metrics={},
                    lineage_info={"generation": generation, "fallback": True}
                )
                fallback_results.append(result)

            except Exception as e:
                logger.error(f"Failed to create fallback model {i}: {e}")

        return fallback_results

    def create_merged_model(self,
                           models: List[nn.Module],
                           config: Dict[str, Any]) -> nn.Module:
        """
        Create a single merged model with specified configuration.

        This method provides compatibility with the GenerationManager interface.
        """
        # Convert config dict to MergerConfig
        technique = MergerTechnique(config.get("strategy", "weighted_average"))
        parameters = {k: v for k, v in config.items() if k not in ["strategy", "weights"]}
        weights = config.get("weights")

        merger_config = MergerConfig(
            technique=technique,
            parameters=parameters,
            weight_models=weights
        )

        # Use appropriate merger method
        if technique == MergerTechnique.WEIGHTED_AVERAGE:
            return self._weighted_average_merge(models, weights or [1/len(models)] * len(models))
        elif technique in [MergerTechnique.SLERP, MergerTechnique.TIES, MergerTechnique.DARE]:
            return self._use_operator_technique(models, merger_config)
        else:
            # Default to weighted average
            return self._weighted_average_merge(models, weights or [1/len(models)] * len(models))

    def get_supported_techniques(self) -> List[MergerTechnique]:
        """Get list of supported merger techniques."""
        return list(MergerTechnique)

    def get_diversity_statistics(self, results: List[MergerResult]) -> Dict[str, Any]:
        """Calculate diversity statistics for a set of merger results."""
        if not results:
            return {}

        diversity_scores = [r.diversity_score for r in results]
        techniques_used = [r.technique.value for r in results]

        return {
            "avg_diversity": np.mean(diversity_scores),
            "min_diversity": np.min(diversity_scores),
            "max_diversity": np.max(diversity_scores),
            "diversity_std": np.std(diversity_scores),
            "techniques_distribution": {tech: techniques_used.count(tech) for tech in set(techniques_used)},
            "total_models": len(results)
        }


# Factory function for backward compatibility
def create_merger_factory(**kwargs) -> MergerOperatorFactory:
    """Create MergerOperatorFactory instance."""
    return MergerOperatorFactory(**kwargs)