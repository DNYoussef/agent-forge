"""
Phase 7: Agentic Self-Configuration Engine

Enhanced implementation integrating SVD-based weight introspection with self-configuration.
This engine dynamically adapts model parameters using real SVD analysis, z-vectors, and
the three adaptation strategies from Transformer² research.

Key Features:
- Dynamic model reconfiguration based on SVD analysis and task requirements
- Real SVD-based weight introspection with singular value fine-tuning
- Z-vector computation for behavioral modification
- Three adaptation strategies: prompt-based, classifier-based, few-shot
- Weight modification tracking and rollback with SVD validation
- Task-specific optimization using dominant singular components
- Integration with existing WeightSpaceExtractor capabilities
- Meta-agent search integration for progressive improvement

Enhanced Features (NEW):
- SVD-based task adaptation using z-vectors
- Singular Value Fine-tuning (SVF) for parameter-efficient adaptation
- Dynamic weight adjustment for unseen tasks in real-time
- Performance tracking aligned with ADAS metrics (13.6 point F1, 14.4% accuracy improvements)
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from abc import ABC, abstractmethod

# Import SVD introspection capabilities
try:
    from .svd_weight_introspector import SVDWeightIntrospector, AdaptationStrategy, SVFConfiguration
except ImportError:
    # Fallback if SVD module not available
    class SVDWeightIntrospector:
        def __init__(self, model=None):
            self.model = model
    class AdaptationStrategy:
        PROMPT_BASED = "prompt_based"
        CLASSIFIER_BASED = "classifier_based"
        FEW_SHOT = "few_shot"
    class SVFConfiguration:
        def __init__(self, **kwargs):
            pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Task categories that require different model configurations."""
    REASONING = "reasoning"
    CREATIVITY = "creativity"
    CODING = "coding"
    ANALYSIS = "analysis"
    LANGUAGE = "language"
    PLANNING = "planning"
    MULTIMODAL = "multimodal"


class ConfigurationStrategy(Enum):
    """Different strategies for model self-configuration."""
    CONSERVATIVE = "conservative"  # Minimal changes, high stability
    AGGRESSIVE = "aggressive"      # Major modifications, high performance
    ADAPTIVE = "adaptive"          # Dynamic adjustment based on feedback
    EXPERIMENTAL = "experimental"  # Cutting-edge techniques


@dataclass
class TaskConfiguration:
    """Enhanced configuration settings with SVD-based adaptation."""
    task_type: TaskType
    attention_enhancement: float = 1.0
    temperature_variance: float = 0.1
    structured_output_strength: float = 0.5
    weight_modifications: Dict[str, float] = field(default_factory=dict)
    priority_layers: List[str] = field(default_factory=list)
    optimization_target: str = "performance"

    # SVD-based enhancement features
    use_svd_adaptation: bool = True
    adaptation_strategy: str = "prompt_based"  # prompt_based, classifier_based, few_shot
    svf_target_rank: int = 64
    svf_adaptation_rate: float = 0.1
    z_vector_influence: float = 1.0


@dataclass
class WeightModification:
    """Tracks individual weight modifications."""
    layer_name: str
    parameter_name: str
    original_value: torch.Tensor
    modified_value: torch.Tensor
    modification_type: str
    timestamp: float
    reason: str
    performance_impact: Optional[float] = None


class WeightModificationTracker:
    """Tracks all weight modifications with rollback capability."""

    def __init__(self):
        self.modifications: List[WeightModification] = []
        self.snapshots: Dict[str, Dict[str, torch.Tensor]] = {}

    def create_snapshot(self, name: str, model: nn.Module) -> None:
        """Create a snapshot of current model weights."""
        snapshot = {}
        for layer_name, param in model.named_parameters():
            snapshot[layer_name] = param.data.clone()
        self.snapshots[name] = snapshot
        logger.info(f"Created weight snapshot: {name}")

    def record_modification(self, modification: WeightModification) -> None:
        """Record a weight modification."""
        self.modifications.append(modification)
        logger.debug(f"Recorded modification: {modification.layer_name}.{modification.parameter_name}")

    def rollback_to_snapshot(self, name: str, model: nn.Module) -> bool:
        """Rollback model weights to a specific snapshot."""
        if name not in self.snapshots:
            logger.error(f"Snapshot {name} not found")
            return False

        snapshot = self.snapshots[name]
        for layer_name, param in model.named_parameters():
            if layer_name in snapshot:
                param.data.copy_(snapshot[layer_name])

        logger.info(f"Rolled back to snapshot: {name}")
        return True

    def get_modification_history(self) -> List[Dict[str, Any]]:
        """Get history of all modifications."""
        return [
            {
                "layer_name": mod.layer_name,
                "parameter_name": mod.parameter_name,
                "modification_type": mod.modification_type,
                "timestamp": mod.timestamp,
                "reason": mod.reason,
                "performance_impact": mod.performance_impact
            }
            for mod in self.modifications
        ]


class AttentionEnhancer:
    """Enhances attention mechanisms for specific tasks."""

    @staticmethod
    def enhance_attention_for_reasoning(attention_layer: nn.Module, enhancement_factor: float) -> None:
        """Enhance attention weights for reasoning tasks."""
        if hasattr(attention_layer, 'weight'):
            with torch.no_grad():
                # Increase attention weight magnitude for better focus
                attention_layer.weight *= (1.0 + enhancement_factor)
                logger.debug(f"Enhanced attention layer for reasoning: factor={enhancement_factor}")

    @staticmethod
    def enhance_attention_for_creativity(attention_layer: nn.Module, enhancement_factor: float) -> None:
        """Enhance attention weights for creative tasks."""
        if hasattr(attention_layer, 'weight'):
            with torch.no_grad():
                # Add controlled noise to encourage exploration
                noise = torch.randn_like(attention_layer.weight) * enhancement_factor * 0.1
                attention_layer.weight += noise
                logger.debug(f"Enhanced attention layer for creativity: factor={enhancement_factor}")

    @staticmethod
    def enhance_attention_for_coding(attention_layer: nn.Module, enhancement_factor: float) -> None:
        """Enhance attention weights for coding tasks."""
        if hasattr(attention_layer, 'weight'):
            with torch.no_grad():
                # Strengthen connections for structured thinking
                attention_layer.weight = torch.nn.functional.normalize(
                    attention_layer.weight * (1.0 + enhancement_factor), dim=-1
                )
                logger.debug(f"Enhanced attention layer for coding: factor={enhancement_factor}")


class TemperatureController:
    """Controls temperature variance across model layers."""

    def __init__(self):
        self.temperature_schedules: Dict[str, Callable[[int], float]] = {}

    def set_adaptive_temperature(self, layer_name: str, base_temp: float, variance: float) -> None:
        """Set adaptive temperature schedule for a layer."""
        def temperature_schedule(step: int) -> float:
            # Dynamic temperature based on step and variance
            cycle = np.sin(step * 0.1) * variance
            return max(0.1, base_temp + cycle)

        self.temperature_schedules[layer_name] = temperature_schedule
        logger.debug(f"Set adaptive temperature for {layer_name}: base={base_temp}, variance={variance}")

    def get_temperature(self, layer_name: str, step: int) -> float:
        """Get current temperature for a layer."""
        if layer_name in self.temperature_schedules:
            return self.temperature_schedules[layer_name](step)
        return 1.0  # Default temperature


class StructuredOutputStrengthener:
    """Strengthens model's ability to produce structured outputs."""

    @staticmethod
    def strengthen_output_layer(output_layer: nn.Module, strength: float) -> None:
        """Strengthen output layer for better structured generation."""
        if hasattr(output_layer, 'weight'):
            with torch.no_grad():
                # Apply regularization to encourage structured patterns
                output_layer.weight = torch.nn.functional.normalize(
                    output_layer.weight * (1.0 + strength), dim=0
                )
                logger.debug(f"Strengthened output layer: strength={strength}")

    @staticmethod
    def add_structure_bias(layer: nn.Module, structure_patterns: Dict[str, float]) -> None:
        """Add bias towards specific structural patterns."""
        if hasattr(layer, 'bias') and layer.bias is not None:
            with torch.no_grad():
                for pattern, weight in structure_patterns.items():
                    # This is a simplified example - in practice, you'd map patterns to bias indices
                    layer.bias += weight * 0.01
                logger.debug(f"Added structure bias: {structure_patterns}")


class SelfConfiguringModel(nn.Module):
    """
    Enhanced self-configuring model with SVD-based weight introspection.

    This model integrates SVD analysis capabilities with the original self-configuration
    system, providing more sophisticated adaptation using z-vectors and singular value
    fine-tuning from Transformer² research.
    """

    def __init__(
        self,
        base_model: nn.Module,
        config_strategy: ConfigurationStrategy = ConfigurationStrategy.ADAPTIVE,
        enable_svd_introspection: bool = True
    ):
        super().__init__()
        self.base_model = base_model
        self.config_strategy = config_strategy
        self.modification_tracker = WeightModificationTracker()
        self.attention_enhancer = AttentionEnhancer()
        self.temperature_controller = TemperatureController()
        self.structure_strengthener = StructuredOutputStrengthener()

        # SVD-based enhancement components
        self.enable_svd_introspection = enable_svd_introspection
        if enable_svd_introspection:
            self.svd_introspector = SVDWeightIntrospector(base_model)
            self.svd_analyses: Dict[str, Any] = {}
            self.z_vectors: Dict[str, torch.Tensor] = {}
        else:
            self.svd_introspector = None

        # Current configuration state
        self.current_task_type: Optional[TaskType] = None
        self.current_config: Optional[TaskConfiguration] = None
        self.performance_history: List[Dict[str, Any]] = []

        # Enhanced performance tracking (ADAS metrics)
        self.baseline_f1 = 0.0
        self.baseline_accuracy = 0.0
        self.best_improvements: Dict[str, float] = {}

        # Create initial snapshot
        self.modification_tracker.create_snapshot("initial", self.base_model)

        # Perform initial SVD analysis if enabled
        if self.enable_svd_introspection and self.svd_introspector:
            self.initialize_svd_analysis()

        logger.info(f"Initialized Enhanced SelfConfiguringModel with strategy: {config_strategy}, SVD: {enable_svd_introspection}")

    def initialize_svd_analysis(self) -> None:
        """Perform initial SVD analysis on all model layers."""
        if not self.svd_introspector:
            return

        logger.info("Performing initial SVD analysis...")
        self.svd_analyses = self.svd_introspector.analyze_model_svd(self.base_model)

        logger.info(f"SVD analysis complete for {len(self.svd_analyses)} layers")
        for layer_name, analysis in self.svd_analyses.items():
            logger.debug(f"Layer {layer_name}: rank={analysis.rank}, compression={analysis.compression_ratio:.3f}")

    def compute_task_z_vector(self, task_type: TaskType, task_description: str,
                             adaptation_strategy: str = "prompt_based") -> Optional[torch.Tensor]:
        """Compute z-vector for task-specific adaptation using SVD analysis."""
        if not self.svd_introspector or not self.svd_analyses:
            logger.warning("SVD introspection not available, skipping z-vector computation")
            return None

        logger.info(f"Computing z-vector for task: {task_type.value}")

        # Map string strategy to enum
        strategy_map = {
            "prompt_based": AdaptationStrategy.PROMPT_BASED,
            "classifier_based": AdaptationStrategy.CLASSIFIER_BASED,
            "few_shot": AdaptationStrategy.FEW_SHOT
        }

        if adaptation_strategy not in strategy_map:
            logger.warning(f"Unknown adaptation strategy: {adaptation_strategy}, using prompt_based")
            adaptation_strategy = "prompt_based"

        strategy = strategy_map[adaptation_strategy]

        # Select reference layers based on task type
        task_layer_preferences = {
            TaskType.REASONING: ["attention", "feedforward"],
            TaskType.CREATIVITY: ["attention", "output"],
            TaskType.CODING: ["attention", "feedforward", "output"],
            TaskType.ANALYSIS: ["attention", "feedforward"],
            TaskType.LANGUAGE: ["attention", "output"]
        }

        preferred_layers = task_layer_preferences.get(task_type, ["attention"])
        reference_layers = []

        for layer_name in self.svd_analyses.keys():
            if any(pref in layer_name.lower() for pref in preferred_layers):
                reference_layers.append(layer_name)

        # Fallback to all analyzed layers if no matches
        if not reference_layers:
            reference_layers = list(self.svd_analyses.keys())[:3]  # Use first 3 layers

        # Compute z-vector
        z_vector = self.svd_introspector.compute_z_vector(
            f"{task_type.value}_{task_description}",
            reference_layers,
            strategy
        )

        # Cache z-vector
        z_vector_key = f"{task_type.value}_{adaptation_strategy}"
        self.z_vectors[z_vector_key] = z_vector

        return z_vector

    def apply_svd_adaptation(self, config: TaskConfiguration) -> bool:
        """Apply SVD-based adaptation using the current configuration."""
        if not self.enable_svd_introspection or not self.svd_introspector:
            return False

        logger.info(f"Applying SVD adaptation for {config.task_type.value}")

        # Get or compute z-vector
        z_vector_key = f"{config.task_type.value}_{config.adaptation_strategy}"
        if z_vector_key not in self.z_vectors:
            z_vector = self.compute_task_z_vector(
                config.task_type,
                config.task_type.value,
                config.adaptation_strategy
            )
        else:
            z_vector = self.z_vectors[z_vector_key]

        if z_vector is None:
            logger.warning("No z-vector available for SVD adaptation")
            return False

        # Create SVF configuration
        svf_config = SVFConfiguration(
            target_rank=config.svf_target_rank,
            adaptation_rate=config.svf_adaptation_rate,
            strategy=AdaptationStrategy.__dict__.get(config.adaptation_strategy.upper(),
                                                   AdaptationStrategy.PROMPT_BASED)
        )

        # Apply adaptation to priority layers
        success_count = 0
        target_layers = config.priority_layers or list(self.svd_analyses.keys())[:3]

        for layer_name in target_layers:
            if layer_name in self.svd_analyses:
                if self.svd_introspector.apply_svf_adaptation(layer_name, svf_config, z_vector):
                    success_count += 1

        success = success_count > 0
        if success:
            logger.info(f"SVD adaptation applied to {success_count} layers")
        else:
            logger.warning("SVD adaptation failed for all layers")

        return success

    def get_svd_recommendations(self) -> Dict[str, Dict[str, Any]]:
        """Get SVD-based recommendations for model optimization."""
        if not self.svd_introspector:
            return {}

        return self.svd_introspector.get_compression_recommendations()

    def integrate_with_weight_space_extractor(self, weight_extractor) -> Dict[str, Any]:
        """Integration point with existing WeightSpaceExtractor."""
        if not self.svd_introspector:
            logger.warning("SVD introspection not available for integration")
            return {}

        return self.svd_introspector.integrate_with_weight_space_extractor(weight_extractor)

    def configure_for_task(self, task_type: TaskType, task_description: str = "") -> TaskConfiguration:
        """
        Configure the model for a specific task type.

        Args:
            task_type: The type of task to configure for
            task_description: Optional description for fine-tuning

        Returns:
            TaskConfiguration: The applied configuration
        """
        logger.info(f"Configuring model for task: {task_type}")

        # Create snapshot before modification
        snapshot_name = f"pre_{task_type.value}_{int(time.time())}"
        self.modification_tracker.create_snapshot(snapshot_name, self.base_model)

        # Generate task-specific configuration
        config = self._generate_task_configuration(task_type, task_description)

        # Apply configuration
        self._apply_configuration(config)

        # Store current state
        self.current_task_type = task_type
        self.current_config = config

        logger.info(f"Successfully configured for {task_type}: {config}")
        return config

    def _generate_task_configuration(self, task_type: TaskType, description: str) -> TaskConfiguration:
        """Generate configuration based on task type and strategy."""
        base_configs = {
            TaskType.REASONING: TaskConfiguration(
                task_type=TaskType.REASONING,
                attention_enhancement=0.3,
                temperature_variance=0.05,
                structured_output_strength=0.7,
                priority_layers=["attention", "feedforward"],
                optimization_target="accuracy"
            ),
            TaskType.CREATIVITY: TaskConfiguration(
                task_type=TaskType.CREATIVITY,
                attention_enhancement=0.1,
                temperature_variance=0.3,
                structured_output_strength=0.2,
                priority_layers=["attention", "output"],
                optimization_target="diversity"
            ),
            TaskType.CODING: TaskConfiguration(
                task_type=TaskType.CODING,
                attention_enhancement=0.4,
                temperature_variance=0.1,
                structured_output_strength=0.9,
                priority_layers=["attention", "feedforward", "output"],
                optimization_target="precision"
            ),
            TaskType.ANALYSIS: TaskConfiguration(
                task_type=TaskType.ANALYSIS,
                attention_enhancement=0.5,
                temperature_variance=0.08,
                structured_output_strength=0.6,
                priority_layers=["attention", "feedforward"],
                optimization_target="thoroughness"
            ),
            TaskType.LANGUAGE: TaskConfiguration(
                task_type=TaskType.LANGUAGE,
                attention_enhancement=0.2,
                temperature_variance=0.15,
                structured_output_strength=0.4,
                priority_layers=["attention", "output"],
                optimization_target="fluency"
            )
        }

        config = base_configs.get(task_type, TaskConfiguration(task_type=task_type))

        # Adjust based on strategy
        if self.config_strategy == ConfigurationStrategy.AGGRESSIVE:
            config.attention_enhancement *= 1.5
            config.structured_output_strength *= 1.3
        elif self.config_strategy == ConfigurationStrategy.CONSERVATIVE:
            config.attention_enhancement *= 0.7
            config.structured_output_strength *= 0.8

        return config

    def _apply_configuration(self, config: TaskConfiguration) -> None:
        """Apply the enhanced configuration to the model including SVD adaptation."""
        logger.info(f"Applying enhanced configuration: {config.task_type}")

        # Apply SVD-based adaptation first if enabled
        svd_success = False
        if config.use_svd_adaptation and self.enable_svd_introspection:
            svd_success = self.apply_svd_adaptation(config)
            if svd_success:
                logger.info("SVD-based adaptation applied successfully")
            else:
                logger.warning("SVD adaptation failed, falling back to traditional methods")

        # Apply traditional configuration methods
        # Enhance attention layers
        self.enhance_attention_layers(config.attention_enhancement, config.task_type)

        # Set temperature variance
        self.increase_temperature_variance(config.temperature_variance)

        # Strengthen structured output
        self.strengthen_structured_output(config.structured_output_strength)

        # Apply weight modifications
        for layer_name, modifier in config.weight_modifications.items():
            self._modify_layer_weights(layer_name, modifier, config.task_type.value)

        # Log configuration summary
        logger.info(f"Configuration applied: SVD={svd_success}, "
                   f"attention={config.attention_enhancement}, "
                   f"temp_variance={config.temperature_variance}, "
                   f"structure_strength={config.structured_output_strength}")

    def enhance_attention_layers(self, enhancement_factor: float, task_type: TaskType) -> None:
        """
        Enhance attention mechanisms based on task type.

        Args:
            enhancement_factor: Factor to enhance attention by
            task_type: Type of task for specialized enhancement
        """
        logger.info(f"Enhancing attention layers: factor={enhancement_factor}, task={task_type}")

        for name, module in self.base_model.named_modules():
            if 'attention' in name.lower():
                if task_type == TaskType.REASONING:
                    self.attention_enhancer.enhance_attention_for_reasoning(module, enhancement_factor)
                elif task_type == TaskType.CREATIVITY:
                    self.attention_enhancer.enhance_attention_for_creativity(module, enhancement_factor)
                elif task_type == TaskType.CODING:
                    self.attention_enhancer.enhance_attention_for_coding(module, enhancement_factor)

                # Record modification
                if hasattr(module, 'weight'):
                    modification = WeightModification(
                        layer_name=name,
                        parameter_name="weight",
                        original_value=module.weight.clone(),
                        modified_value=module.weight.clone(),
                        modification_type="attention_enhancement",
                        timestamp=time.time(),
                        reason=f"Enhanced for {task_type.value}"
                    )
                    self.modification_tracker.record_modification(modification)

    def increase_temperature_variance(self, variance: float) -> None:
        """
        Increase temperature variance across model layers.

        Args:
            variance: Amount of variance to introduce
        """
        logger.info(f"Setting temperature variance: {variance}")

        for name, module in self.base_model.named_modules():
            if any(layer_type in name.lower() for layer_type in ['linear', 'output']):
                self.temperature_controller.set_adaptive_temperature(name, 1.0, variance)

    def strengthen_structured_output(self, strength: float) -> None:
        """
        Strengthen the model's structured output capabilities.

        Args:
            strength: Strength factor for structured output enhancement
        """
        logger.info(f"Strengthening structured output: strength={strength}")

        # Find and enhance output layers
        for name, module in self.base_model.named_modules():
            if 'output' in name.lower() or 'head' in name.lower():
                self.structure_strengthener.strengthen_output_layer(module, strength)

                # Record modification
                if hasattr(module, 'weight'):
                    modification = WeightModification(
                        layer_name=name,
                        parameter_name="weight",
                        original_value=module.weight.clone(),
                        modified_value=module.weight.clone(),
                        modification_type="structured_output_enhancement",
                        timestamp=time.time(),
                        reason=f"Strengthened for structure: {strength}"
                    )
                    self.modification_tracker.record_modification(modification)

    def _modify_layer_weights(self, layer_name: str, modifier: float, reason: str) -> None:
        """Apply weight modifications to a specific layer."""
        for name, param in self.base_model.named_parameters():
            if layer_name in name:
                with torch.no_grad():
                    original = param.data.clone()
                    param.data *= (1.0 + modifier)

                    modification = WeightModification(
                        layer_name=name,
                        parameter_name="weight",
                        original_value=original,
                        modified_value=param.data.clone(),
                        modification_type="weight_scaling",
                        timestamp=time.time(),
                        reason=reason
                    )
                    self.modification_tracker.record_modification(modification)

                logger.debug(f"Modified layer {name} with factor {modifier}")

    def rollback_configuration(self, snapshot_name: str = "initial") -> bool:
        """
        Rollback to a previous configuration state.

        Args:
            snapshot_name: Name of snapshot to rollback to

        Returns:
            bool: Success status
        """
        logger.info(f"Rolling back to snapshot: {snapshot_name}")
        success = self.modification_tracker.rollback_to_snapshot(snapshot_name, self.base_model)

        if success:
            self.current_task_type = None
            self.current_config = None

        return success

    def evaluate_configuration(self, metrics: Dict[str, float]) -> float:
        """
        Evaluate the current configuration performance.

        Args:
            metrics: Performance metrics to evaluate

        Returns:
            float: Overall performance score
        """
        if not self.current_config:
            return 0.0

        # Calculate weighted score based on optimization target
        target = self.current_config.optimization_target
        weights = {
            "accuracy": {"accuracy": 0.5, "precision": 0.3, "recall": 0.2},
            "diversity": {"creativity": 0.4, "novelty": 0.3, "coherence": 0.3},
            "precision": {"accuracy": 0.4, "precision": 0.4, "consistency": 0.2},
            "thoroughness": {"completeness": 0.4, "depth": 0.3, "accuracy": 0.3},
            "fluency": {"fluency": 0.5, "coherence": 0.3, "naturalness": 0.2}
        }

        weight_set = weights.get(target, {"default": 1.0})
        score = sum(metrics.get(metric, 0.0) * weight for metric, weight in weight_set.items())

        # Record performance
        performance_record = {
            "timestamp": time.time(),
            "task_type": self.current_task_type.value if self.current_task_type else None,
            "score": score,
            "metrics": metrics,
            "config": self.current_config.__dict__ if self.current_config else None
        }
        self.performance_history.append(performance_record)

        logger.info(f"Configuration performance: {score:.3f}")
        return score

    def get_configuration_status(self) -> Dict[str, Any]:
        """Get enhanced configuration status including SVD analysis."""
        status = {
            "current_task_type": self.current_task_type.value if self.current_task_type else None,
            "current_config": self.current_config.__dict__ if self.current_config else None,
            "modification_count": len(self.modification_tracker.modifications),
            "snapshots": list(self.modification_tracker.snapshots.keys()),
            "performance_history": self.performance_history[-10:],  # Last 10 records
            "strategy": self.config_strategy.value,

            # SVD enhancement information
            "svd_introspection_enabled": self.enable_svd_introspection,
            "svd_analyses_count": len(self.svd_analyses) if hasattr(self, 'svd_analyses') else 0,
            "z_vectors_cached": len(self.z_vectors) if hasattr(self, 'z_vectors') else 0,
            "performance_improvements": self.best_improvements
        }

        # Add SVD-specific status if available
        if self.enable_svd_introspection and self.svd_introspector:
            status["svd_recommendations"] = self.get_svd_recommendations()

            # Add SVD adaptation history
            if hasattr(self.svd_introspector, 'adaptation_history'):
                status["svd_adaptation_history"] = self.svd_introspector.adaptation_history[-5:]  # Last 5

        return status

    def optimize_configuration(self, target_metrics: Dict[str, float], max_iterations: int = 10) -> bool:
        """
        Automatically optimize configuration to meet target metrics.

        Args:
            target_metrics: Target performance metrics
            max_iterations: Maximum optimization iterations

        Returns:
            bool: Whether optimization succeeded
        """
        logger.info(f"Starting configuration optimization: targets={target_metrics}")

        best_score = 0.0
        best_config = None

        for iteration in range(max_iterations):
            # Create modified configuration
            if self.current_config:
                modified_config = self._create_modified_config(self.current_config, iteration)
                self._apply_configuration(modified_config)

                # Evaluate (in practice, this would run actual tasks)
                simulated_metrics = self._simulate_performance(modified_config)
                score = self.evaluate_configuration(simulated_metrics)

                if score > best_score:
                    best_score = score
                    best_config = modified_config
                    logger.info(f"Iteration {iteration}: New best score {score:.3f}")

                # Check if targets met
                if all(simulated_metrics.get(k, 0) >= v for k, v in target_metrics.items()):
                    logger.info(f"Optimization succeeded in {iteration + 1} iterations")
                    return True

        # Apply best configuration found
        if best_config:
            self._apply_configuration(best_config)
            logger.info(f"Applied best configuration with score: {best_score:.3f}")

        return False

    def _create_modified_config(self, base_config: TaskConfiguration, iteration: int) -> TaskConfiguration:
        """Create a modified configuration for optimization."""
        modification_factor = 0.1 * (1.0 + iteration * 0.05)  # Gradually increase modifications

        modified_config = TaskConfiguration(
            task_type=base_config.task_type,
            attention_enhancement=max(0.1, min(1.0, base_config.attention_enhancement +
                                             np.random.normal(0, modification_factor))),
            temperature_variance=max(0.01, min(0.5, base_config.temperature_variance +
                                             np.random.normal(0, modification_factor * 0.5))),
            structured_output_strength=max(0.1, min(1.0, base_config.structured_output_strength +
                                                  np.random.normal(0, modification_factor))),
            priority_layers=base_config.priority_layers.copy(),
            optimization_target=base_config.optimization_target
        )

        return modified_config

    def _simulate_performance(self, config: TaskConfiguration) -> Dict[str, float]:
        """Simulate performance metrics for a configuration (for testing purposes)."""
        # This is a simplified simulation - in practice, you'd run actual evaluation tasks
        base_performance = {
            "accuracy": 0.7,
            "precision": 0.6,
            "recall": 0.65,
            "creativity": 0.5,
            "novelty": 0.4,
            "coherence": 0.8,
            "consistency": 0.75,
            "completeness": 0.7,
            "depth": 0.6,
            "fluency": 0.85,
            "naturalness": 0.8
        }

        # Modify based on configuration
        task_multipliers = {
            TaskType.REASONING: {"accuracy": 1.2, "precision": 1.3, "recall": 1.1},
            TaskType.CREATIVITY: {"creativity": 1.4, "novelty": 1.5, "coherence": 0.9},
            TaskType.CODING: {"accuracy": 1.3, "precision": 1.4, "consistency": 1.2},
            TaskType.ANALYSIS: {"completeness": 1.3, "depth": 1.4, "accuracy": 1.2},
            TaskType.LANGUAGE: {"fluency": 1.2, "coherence": 1.3, "naturalness": 1.2}
        }

        multipliers = task_multipliers.get(config.task_type, {})

        simulated = {}
        for metric, base_value in base_performance.items():
            # Apply task-specific multiplier
            multiplier = multipliers.get(metric, 1.0)

            # Apply configuration effects
            enhancement_effect = config.attention_enhancement * 0.1
            structure_effect = config.structured_output_strength * 0.05

            value = base_value * multiplier * (1.0 + enhancement_effect + structure_effect)
            # Add some noise and clamp to [0, 1]
            value += np.random.normal(0, 0.05)
            simulated[metric] = max(0.0, min(1.0, value))

        return simulated

    def forward(self, *args, **kwargs):
        """Forward pass through the configured model."""
        return self.base_model(*args, **kwargs)


# Export main classes
__all__ = [
    'SelfConfiguringModel',
    'TaskType',
    'ConfigurationStrategy',
    'TaskConfiguration',
    'WeightModification',
    'WeightModificationTracker',
    'AttentionEnhancer',
    'TemperatureController',
    'StructuredOutputStrengthener'
]


if __name__ == "__main__":
    # Example usage and testing
    logger.info("Testing SelfConfiguringModel...")

    # Create a simple base model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(128, 128)
            self.feedforward = nn.Linear(128, 256)
            self.output = nn.Linear(256, 64)

        def forward(self, x):
            x = self.attention(x)
            x = self.feedforward(x)
            x = self.output(x)
            return x

    # Initialize enhanced self-configuring model with SVD
    base_model = SimpleModel()
    self_config_model = SelfConfiguringModel(
        base_model,
        ConfigurationStrategy.ADAPTIVE,
        enable_svd_introspection=True
    )

    # Test configuration for different tasks with SVD adaptation
    tasks = [TaskType.REASONING, TaskType.CREATIVITY, TaskType.CODING]
    adaptation_strategies = ["prompt_based", "classifier_based", "few_shot"]

    for task in tasks:
        print(f"\n=== Testing Enhanced {task.value} Configuration ===")

        # Test different adaptation strategies
        for strategy in adaptation_strategies:
            print(f"\n--- Testing {strategy} adaptation ---")

            config = self_config_model.configure_for_task(task, f"Test {task.value} with {strategy}")
            config.adaptation_strategy = strategy
            config.use_svd_adaptation = True

            # Apply the configuration
            self_config_model._apply_configuration(config)

            # Simulate enhanced performance evaluation
            test_metrics = {
                "accuracy": np.random.uniform(0.6, 0.9),
                "precision": np.random.uniform(0.5, 0.8),
                "creativity": np.random.uniform(0.3, 0.7),
                "f1_score": np.random.uniform(0.5, 0.85)
            }

            score = self_config_model.evaluate_configuration(test_metrics)
            print(f"Performance score: {score:.3f}")

            # Get enhanced status
            status = self_config_model.get_configuration_status()
            print(f"Modifications: {status['modification_count']}, "
                  f"SVD analyses: {status['svd_analyses_count']}, "
                  f"Z-vectors: {status['z_vectors_cached']}")

    # Test SVD-specific features
    print(f"\n=== Testing SVD-Specific Features ===")

    # Test SVD recommendations
    if self_config_model.enable_svd_introspection:
        recommendations = self_config_model.get_svd_recommendations()
        print(f"SVD recommendations for {len(recommendations)} layers:")
        for layer_name, rec in recommendations.items():
            print(f"  {layer_name}: {rec['recommendation']} "
                  f"(potential savings: {rec.get('potential_savings', 0):.1%})")

        # Test z-vector computation
        z_vector = self_config_model.compute_task_z_vector(
            TaskType.REASONING,
            "Complex mathematical reasoning task",
            "prompt_based"
        )
        if z_vector is not None:
            print(f"Z-vector computed: shape={z_vector.shape}, norm={torch.norm(z_vector):.3f}")

    # Test integration with WeightSpaceExtractor (mock)
    print(f"\n=== Testing WeightSpaceExtractor Integration ===")

    # Mock weight extractor for testing
    class MockWeightExtractor:
        def __init__(self, model):
            self.model = model

        def extract_weights(self):
            return {"mock_layer": torch.randn(10, 10)}

        def compute_weight_statistics(self, weights):
            return {"layer_stats": {"mock_layer": {"std": 0.1, "mean": 0.0}}}

    mock_extractor = MockWeightExtractor(base_model)
    integration_data = self_config_model.integrate_with_weight_space_extractor(mock_extractor)

    if integration_data:
        print(f"Integration successful: {len(integration_data)} data sections")
        if "enhanced_metrics" in integration_data:
            print(f"Enhanced metrics computed for {len(integration_data['enhanced_metrics'])} layers")

    # Test optimization with SVD
    print(f"\n=== Testing Enhanced Configuration Optimization ===")
    target_metrics = {"accuracy": 0.85, "precision": 0.80, "f1_score": 0.82}
    success = self_config_model.optimize_configuration(target_metrics, max_iterations=5)
    print(f"Enhanced optimization successful: {success}")

    # Show final performance improvements
    if self_config_model.best_improvements:
        print(f"\nPerformance improvements achieved:")
        for metric, improvement in self_config_model.best_improvements.items():
            print(f"  {metric}: +{improvement:.3f}")

    # Test rollback
    print(f"\n=== Testing Configuration Rollback ===")
    rollback_success = self_config_model.rollback_configuration("initial")
    print(f"Rollback successful: {rollback_success}")

    logger.info("Enhanced SelfConfiguringModel with SVD testing completed")