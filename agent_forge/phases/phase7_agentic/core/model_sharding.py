"""
Phase 7: Model Sharding System

Implementation of a sophisticated model sharding system that divides model capabilities
into specialized shards (reasoning, creativity, coding, analysis, language) with
dynamic routing and weight masking based on task requirements.

Key Features:
- Specialized model shards for different cognitive tasks
- Dynamic shard routing based on task analysis
- Weight masking for shard isolation
- Load balancing across shards
- Performance monitoring per shard
- Adaptive shard combination strategies
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShardType(Enum):
    """Types of specialized model shards."""
    REASONING = "reasoning"
    CREATIVITY = "creativity"
    CODING = "coding"
    ANALYSIS = "analysis"
    LANGUAGE = "language"
    MULTIMODAL = "multimodal"
    PLANNING = "planning"


class RoutingStrategy(Enum):
    """Strategies for routing tasks to shards."""
    SINGLE_BEST = "single_best"           # Route to single best shard
    WEIGHTED_ENSEMBLE = "weighted_ensemble"  # Combine multiple shards
    SEQUENTIAL = "sequential"             # Process through multiple shards
    ADAPTIVE = "adaptive"                 # Learn optimal routing over time


@dataclass
class ShardCapability:
    """Defines the capability profile of a shard."""
    shard_type: ShardType
    primary_tasks: List[str]
    secondary_tasks: List[str] = field(default_factory=list)
    performance_profile: Dict[str, float] = field(default_factory=dict)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    compatible_shards: List[ShardType] = field(default_factory=list)


@dataclass
class RoutingDecision:
    """Represents a routing decision for a task."""
    task_description: str
    selected_shards: List[ShardType]
    routing_strategy: RoutingStrategy
    confidence_scores: Dict[ShardType, float]
    reasoning: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ShardPerformanceMetrics:
    """Performance metrics for individual shards."""
    shard_type: ShardType
    total_requests: int = 0
    successful_completions: int = 0
    average_latency: float = 0.0
    average_quality_score: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    last_updated: float = field(default_factory=time.time)


class WeightMask:
    """Manages weight masking for shard isolation."""

    def __init__(self, shard_type: ShardType, mask_ratio: float = 0.7):
        self.shard_type = shard_type
        self.mask_ratio = mask_ratio  # Fraction of weights to keep active
        self.masks: Dict[str, torch.Tensor] = {}
        self.masked_layers: List[str] = []

    def create_mask(self, layer_name: str, weight_shape: Tuple[int, ...]) -> torch.Tensor:
        """Create a binary mask for a layer based on shard specialization."""
        # Different masking strategies for different shards
        if self.shard_type == ShardType.REASONING:
            # Focus on central processing units
            mask = self._create_center_focused_mask(weight_shape)
        elif self.shard_type == ShardType.CREATIVITY:
            # Random sparse mask for exploration
            mask = self._create_random_sparse_mask(weight_shape)
        elif self.shard_type == ShardType.CODING:
            # Structured mask for systematic processing
            mask = self._create_structured_mask(weight_shape)
        elif self.shard_type == ShardType.ANALYSIS:
            # Dense mask for thorough processing
            mask = self._create_dense_mask(weight_shape)
        elif self.shard_type == ShardType.LANGUAGE:
            # Distributed mask for linguistic patterns
            mask = self._create_distributed_mask(weight_shape)
        else:
            # Default uniform random mask
            mask = self._create_random_sparse_mask(weight_shape)

        self.masks[layer_name] = mask
        self.masked_layers.append(layer_name)
        return mask

    def _create_center_focused_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Create mask focused on central regions (for reasoning)."""
        mask = torch.zeros(shape)
        if len(shape) == 2:
            h, w = shape
            center_h, center_w = h // 2, w // 2
            radius = int(min(h, w) * self.mask_ratio * 0.5)

            for i in range(max(0, center_h - radius), min(h, center_h + radius)):
                for j in range(max(0, center_w - radius), min(w, center_w + radius)):
                    mask[i, j] = 1.0
        else:
            # For higher dimensional tensors, flatten and apply center focusing
            flat_mask = torch.zeros(torch.prod(torch.tensor(shape)))
            center = len(flat_mask) // 2
            radius = int(len(flat_mask) * self.mask_ratio * 0.5)
            flat_mask[max(0, center - radius):min(len(flat_mask), center + radius)] = 1.0
            mask = flat_mask.reshape(shape)

        return mask

    def _create_random_sparse_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Create random sparse mask (for creativity)."""
        mask = torch.zeros(shape)
        num_active = int(torch.prod(torch.tensor(shape)) * self.mask_ratio)
        flat_mask = mask.flatten()
        indices = torch.randperm(len(flat_mask))[:num_active]
        flat_mask[indices] = 1.0
        return flat_mask.reshape(shape)

    def _create_structured_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Create structured mask with regular patterns (for coding)."""
        mask = torch.zeros(shape)
        if len(shape) == 2:
            h, w = shape
            # Create checkerboard-like pattern with varying density
            for i in range(h):
                for j in range(w):
                    if (i + j) % int(1.0 / self.mask_ratio) == 0:
                        mask[i, j] = 1.0
        else:
            # Apply structured pattern to flattened tensor
            flat_mask = mask.flatten()
            step = int(1.0 / self.mask_ratio)
            for i in range(0, len(flat_mask), step):
                flat_mask[i] = 1.0
            mask = flat_mask.reshape(shape)

        return mask

    def _create_dense_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Create dense mask with high connectivity (for analysis)."""
        # Use higher mask ratio for analysis tasks
        dense_ratio = min(0.9, self.mask_ratio + 0.2)
        mask = torch.bernoulli(torch.full(shape, dense_ratio))
        return mask

    def _create_distributed_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Create distributed mask for linguistic processing."""
        mask = torch.zeros(shape)
        if len(shape) == 2:
            h, w = shape
            # Create wave-like pattern for distributed processing
            for i in range(h):
                for j in range(w):
                    wave_val = np.sin(i * 0.1) * np.cos(j * 0.1)
                    if wave_val > (1.0 - self.mask_ratio * 2):
                        mask[i, j] = 1.0
        else:
            flat_mask = mask.flatten()
            for i in range(len(flat_mask)):
                if np.sin(i * 0.01) > (1.0 - self.mask_ratio * 2):
                    flat_mask[i] = 1.0
            mask = flat_mask.reshape(shape)

        return mask

    def apply_mask(self, layer: nn.Module) -> None:
        """Apply the mask to a specific layer."""
        if hasattr(layer, 'weight') and layer.weight is not None:
            layer_name = id(layer)  # Use object id as unique identifier
            if str(layer_name) not in self.masks:
                mask = self.create_mask(str(layer_name), layer.weight.shape)
            else:
                mask = self.masks[str(layer_name)]

            with torch.no_grad():
                layer.weight.data *= mask.to(layer.weight.device)

    def get_active_weight_ratio(self, layer_name: str) -> float:
        """Get the ratio of active weights in a layer."""
        if layer_name in self.masks:
            return float(self.masks[layer_name].sum() / self.masks[layer_name].numel())
        return 0.0


class ShardRouter:
    """Routes tasks to appropriate shards based on content analysis."""

    def __init__(self, routing_strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE):
        self.routing_strategy = routing_strategy
        self.routing_history: List[RoutingDecision] = []
        self.performance_feedback: Dict[Tuple[str, ShardType], float] = {}

        # Task classification patterns
        self.task_patterns = {
            ShardType.REASONING: [
                "analyze", "deduce", "infer", "logical", "proof", "theorem", "conclusion",
                "hypothesis", "evidence", "reasoning", "syllogism", "argument"
            ],
            ShardType.CREATIVITY: [
                "create", "imagine", "brainstorm", "innovative", "original", "artistic",
                "design", "story", "poem", "creative", "novel", "inventive"
            ],
            ShardType.CODING: [
                "code", "program", "function", "algorithm", "debug", "implementation",
                "software", "API", "class", "method", "variable", "syntax"
            ],
            ShardType.ANALYSIS: [
                "evaluate", "assess", "compare", "review", "examine", "investigate",
                "study", "research", "analysis", "breakdown", "decompose"
            ],
            ShardType.LANGUAGE: [
                "translate", "grammar", "linguistic", "language", "text", "writing",
                "communication", "expression", "vocabulary", "style", "tone"
            ]
        }

    def route_task(self, task_description: str, context: Dict[str, Any] = None) -> RoutingDecision:
        """Route a task to appropriate shards."""
        logger.debug(f"Routing task: {task_description[:50]}...")

        # Analyze task content
        shard_scores = self._analyze_task_content(task_description)

        # Apply routing strategy
        if self.routing_strategy == RoutingStrategy.SINGLE_BEST:
            selected_shards = [max(shard_scores, key=shard_scores.get)]
        elif self.routing_strategy == RoutingStrategy.WEIGHTED_ENSEMBLE:
            # Select top shards with score > threshold
            threshold = 0.3
            selected_shards = [shard for shard, score in shard_scores.items() if score > threshold]
        elif self.routing_strategy == RoutingStrategy.SEQUENTIAL:
            # Select shards in order of relevance
            sorted_shards = sorted(shard_scores.items(), key=lambda x: x[1], reverse=True)
            selected_shards = [shard for shard, score in sorted_shards[:3]]
        else:  # ADAPTIVE
            selected_shards = self._adaptive_routing(task_description, shard_scores)

        # Create routing decision
        decision = RoutingDecision(
            task_description=task_description,
            selected_shards=selected_shards,
            routing_strategy=self.routing_strategy,
            confidence_scores=shard_scores,
            reasoning=self._generate_routing_reasoning(task_description, selected_shards, shard_scores)
        )

        self.routing_history.append(decision)
        logger.info(f"Routed to shards: {[s.value for s in selected_shards]}")

        return decision

    def _analyze_task_content(self, task_description: str) -> Dict[ShardType, float]:
        """Analyze task content to determine shard relevance scores."""
        task_lower = task_description.lower()
        shard_scores = {}

        for shard_type, patterns in self.task_patterns.items():
            score = 0.0
            pattern_matches = 0

            for pattern in patterns:
                if pattern in task_lower:
                    score += 1.0
                    pattern_matches += 1

            # Normalize by pattern count
            if patterns:
                score /= len(patterns)

            # Add bonus for multiple pattern matches
            if pattern_matches > 1:
                score *= (1.0 + 0.1 * pattern_matches)

            # Consider historical performance
            historical_performance = self._get_historical_performance(task_description, shard_type)
            score = 0.7 * score + 0.3 * historical_performance

            shard_scores[shard_type] = min(1.0, score)

        # Ensure at least one shard has reasonable score
        if max(shard_scores.values()) < 0.1:
            shard_scores[ShardType.ANALYSIS] = 0.5  # Default fallback

        return shard_scores

    def _adaptive_routing(self, task_description: str, shard_scores: Dict[ShardType, float]) -> List[ShardType]:
        """Implement adaptive routing based on historical performance."""
        # Weight scores by historical success
        adjusted_scores = {}
        for shard, score in shard_scores.items():
            historical_perf = self._get_historical_performance(task_description, shard)
            adjusted_scores[shard] = score * (0.5 + 0.5 * historical_perf)

        # Select top scoring shards
        sorted_shards = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)

        # Dynamic selection based on confidence
        selected = []
        for shard, score in sorted_shards:
            if score > 0.2:  # Minimum confidence threshold
                selected.append(shard)
            if len(selected) >= 3:  # Maximum number of shards
                break

        if not selected:
            selected = [sorted_shards[0][0]]  # At least select the best one

        return selected

    def _get_historical_performance(self, task_description: str, shard_type: ShardType) -> float:
        """Get historical performance for similar tasks."""
        # Simple similarity-based lookup
        task_key = self._generate_task_key(task_description)

        matching_performance = []
        for (stored_key, stored_shard), performance in self.performance_feedback.items():
            if stored_shard == shard_type and self._calculate_similarity(task_key, stored_key) > 0.5:
                matching_performance.append(performance)

        if matching_performance:
            return np.mean(matching_performance)
        return 0.5  # Default neutral performance

    def _generate_task_key(self, task_description: str) -> str:
        """Generate a key for task similarity matching."""
        # Extract key terms for matching
        words = task_description.lower().split()
        key_words = [word for word in words if len(word) > 3]
        return " ".join(sorted(key_words)[:5])  # Top 5 key words

    def _calculate_similarity(self, key1: str, key2: str) -> float:
        """Calculate similarity between task keys."""
        words1 = set(key1.split())
        words2 = set(key2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _generate_routing_reasoning(self, task: str, shards: List[ShardType], scores: Dict[ShardType, float]) -> str:
        """Generate human-readable reasoning for routing decision."""
        reasoning_parts = []

        for shard in shards:
            score = scores.get(shard, 0.0)
            reasoning_parts.append(f"{shard.value} (score: {score:.2f})")

        return f"Selected {len(shards)} shards based on content analysis: {', '.join(reasoning_parts)}"

    def update_performance_feedback(self, decision: RoutingDecision, performance_score: float) -> None:
        """Update performance feedback for routing decisions."""
        task_key = self._generate_task_key(decision.task_description)

        for shard in decision.selected_shards:
            self.performance_feedback[(task_key, shard)] = performance_score

        logger.debug(f"Updated performance feedback: {performance_score:.2f}")


class ModelShard(nn.Module):
    """Individual model shard with specialized capabilities."""

    def __init__(
        self,
        base_model: nn.Module,
        shard_type: ShardType,
        capability: ShardCapability,
        mask_ratio: float = 0.7
    ):
        super().__init__()
        self.base_model = base_model
        self.shard_type = shard_type
        self.capability = capability
        self.weight_mask = WeightMask(shard_type, mask_ratio)
        self.performance_metrics = ShardPerformanceMetrics(shard_type)
        self.is_active = True

        # Apply initial masking
        self._apply_shard_masking()

        logger.info(f"Initialized {shard_type.value} shard with {mask_ratio:.1%} weight masking")

    def _apply_shard_masking(self) -> None:
        """Apply weight masking to specialize the shard."""
        for name, module in self.base_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                self.weight_mask.apply_mask(module)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the shard."""
        if not self.is_active:
            raise RuntimeError(f"Shard {self.shard_type.value} is not active")

        start_time = time.time()

        try:
            output = self.base_model(*args, **kwargs)

            # Update performance metrics
            latency = time.time() - start_time
            self.performance_metrics.total_requests += 1
            self.performance_metrics.successful_completions += 1
            self.performance_metrics.average_latency = (
                self.performance_metrics.average_latency *
                (self.performance_metrics.total_requests - 1) + latency
            ) / self.performance_metrics.total_requests

            return output

        except Exception as e:
            self.performance_metrics.error_count += 1
            logger.error(f"Error in {self.shard_type.value} shard: {e}")
            raise

    def get_specialization_score(self, task_description: str) -> float:
        """Calculate how well this shard matches a task."""
        task_lower = task_description.lower()

        # Score based on primary tasks
        primary_score = sum(1.0 for task in self.capability.primary_tasks if task.lower() in task_lower)
        primary_score /= max(len(self.capability.primary_tasks), 1)

        # Score based on secondary tasks (lower weight)
        secondary_score = sum(0.5 for task in self.capability.secondary_tasks if task.lower() in task_lower)
        secondary_score /= max(len(self.capability.secondary_tasks), 1)

        return min(1.0, primary_score + secondary_score)

    def update_performance_metrics(self, quality_score: float) -> None:
        """Update quality performance metrics."""
        total = self.performance_metrics.successful_completions
        if total == 0:
            self.performance_metrics.average_quality_score = quality_score
        else:
            self.performance_metrics.average_quality_score = (
                self.performance_metrics.average_quality_score * (total - 1) + quality_score
            ) / total

        self.performance_metrics.last_updated = time.time()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for this shard."""
        return {
            "shard_type": self.shard_type.value,
            "total_requests": self.performance_metrics.total_requests,
            "success_rate": (
                self.performance_metrics.successful_completions /
                max(self.performance_metrics.total_requests, 1)
            ),
            "average_latency": self.performance_metrics.average_latency,
            "average_quality": self.performance_metrics.average_quality_score,
            "error_rate": (
                self.performance_metrics.error_count /
                max(self.performance_metrics.total_requests, 1)
            ),
            "is_active": self.is_active,
            "weight_mask_ratio": self.weight_mask.mask_ratio
        }


class ModelShardingSystem:
    """
    Main model sharding system that manages specialized shards and routing.

    This system implements advanced model sharding with dynamic routing,
    load balancing, and performance monitoring for optimal task execution.
    """

    def __init__(
        self,
        base_model: nn.Module,
        shard_types: List[ShardType] = None,
        routing_strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE
    ):
        self.base_model = base_model
        self.shard_types = shard_types or list(ShardType)
        self.routing_strategy = routing_strategy

        # Core components
        self.shards: Dict[ShardType, ModelShard] = {}
        self.router = ShardRouter(routing_strategy)
        self.load_balancer = LoadBalancer()

        # System metrics
        self.system_metrics = {
            "total_requests": 0,
            "routing_decisions": 0,
            "ensemble_operations": 0,
            "cache_hits": 0,
            "average_response_time": 0.0
        }

        # Initialize shards
        self._initialize_shards()

        logger.info(f"Initialized ModelShardingSystem with {len(self.shards)} shards")

    def _initialize_shards(self) -> None:
        """Initialize all model shards with their capabilities."""
        # Define capabilities for each shard type
        capabilities = {
            ShardType.REASONING: ShardCapability(
                shard_type=ShardType.REASONING,
                primary_tasks=["logical reasoning", "problem solving", "inference", "deduction"],
                secondary_tasks=["analysis", "evaluation"],
                performance_profile={"accuracy": 0.9, "speed": 0.7, "creativity": 0.3},
                compatible_shards=[ShardType.ANALYSIS, ShardType.LANGUAGE]
            ),
            ShardType.CREATIVITY: ShardCapability(
                shard_type=ShardType.CREATIVITY,
                primary_tasks=["creative writing", "brainstorming", "artistic generation", "innovation"],
                secondary_tasks=["storytelling", "design"],
                performance_profile={"accuracy": 0.6, "speed": 0.8, "creativity": 0.95},
                compatible_shards=[ShardType.LANGUAGE, ShardType.MULTIMODAL]
            ),
            ShardType.CODING: ShardCapability(
                shard_type=ShardType.CODING,
                primary_tasks=["programming", "code generation", "debugging", "software development"],
                secondary_tasks=["technical documentation", "algorithm design"],
                performance_profile={"accuracy": 0.85, "speed": 0.75, "creativity": 0.4},
                compatible_shards=[ShardType.REASONING, ShardType.ANALYSIS]
            ),
            ShardType.ANALYSIS: ShardCapability(
                shard_type=ShardType.ANALYSIS,
                primary_tasks=["data analysis", "research", "evaluation", "comparison"],
                secondary_tasks=["summarization", "extraction"],
                performance_profile={"accuracy": 0.9, "speed": 0.6, "creativity": 0.3},
                compatible_shards=[ShardType.REASONING, ShardType.CODING]
            ),
            ShardType.LANGUAGE: ShardCapability(
                shard_type=ShardType.LANGUAGE,
                primary_tasks=["translation", "grammar", "linguistic analysis", "communication"],
                secondary_tasks=["writing assistance", "style adaptation"],
                performance_profile={"accuracy": 0.8, "speed": 0.85, "creativity": 0.6},
                compatible_shards=[ShardType.CREATIVITY, ShardType.ANALYSIS]
            )
        }

        # Create shards
        for shard_type in self.shard_types:
            if shard_type in capabilities:
                # Create a copy of the base model for each shard
                shard_model = self._create_shard_model()
                capability = capabilities[shard_type]

                self.shards[shard_type] = ModelShard(
                    base_model=shard_model,
                    shard_type=shard_type,
                    capability=capability,
                    mask_ratio=0.7
                )

    def _create_shard_model(self) -> nn.Module:
        """Create a copy of the base model for a shard."""
        # This is a simplified approach - in practice, you'd use proper model copying
        # or implement shared parameters with different views
        return self.base_model  # For now, share the model (masking handles specialization)

    def process_task(
        self,
        task_description: str,
        input_data: Any,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a task through appropriate shards.

        Args:
            task_description: Description of the task
            input_data: Input data for processing
            context: Additional context information

        Returns:
            Dict containing results, metadata, and performance info
        """
        start_time = time.time()
        self.system_metrics["total_requests"] += 1

        logger.info(f"Processing task: {task_description[:50]}...")

        try:
            # Route task to appropriate shards
            routing_decision = self.router.route_task(task_description, context)
            self.system_metrics["routing_decisions"] += 1

            # Execute task on selected shards
            if routing_decision.routing_strategy == RoutingStrategy.SINGLE_BEST:
                result = self._execute_single_shard(
                    routing_decision.selected_shards[0], input_data, task_description
                )
            elif routing_decision.routing_strategy == RoutingStrategy.WEIGHTED_ENSEMBLE:
                result = self._execute_ensemble(
                    routing_decision.selected_shards, input_data, task_description, routing_decision.confidence_scores
                )
            elif routing_decision.routing_strategy == RoutingStrategy.SEQUENTIAL:
                result = self._execute_sequential(
                    routing_decision.selected_shards, input_data, task_description
                )
            else:  # ADAPTIVE
                result = self._execute_adaptive(
                    routing_decision, input_data, task_description
                )

            # Calculate metrics
            processing_time = time.time() - start_time
            self._update_system_metrics(processing_time)

            # Prepare response
            response = {
                "result": result,
                "routing_decision": {
                    "selected_shards": [s.value for s in routing_decision.selected_shards],
                    "strategy": routing_decision.routing_strategy.value,
                    "confidence_scores": {s.value: score for s, score in routing_decision.confidence_scores.items()},
                    "reasoning": routing_decision.reasoning
                },
                "performance": {
                    "processing_time": processing_time,
                    "shards_used": len(routing_decision.selected_shards),
                    "strategy": routing_decision.routing_strategy.value
                },
                "metadata": {
                    "timestamp": time.time(),
                    "task_hash": hash(task_description)
                }
            }

            logger.info(f"Task completed in {processing_time:.3f}s using {len(routing_decision.selected_shards)} shards")
            return response

        except Exception as e:
            logger.error(f"Error processing task: {e}")
            return {
                "error": str(e),
                "routing_decision": None,
                "performance": {"processing_time": time.time() - start_time},
                "metadata": {"timestamp": time.time(), "error": True}
            }

    def _execute_single_shard(self, shard_type: ShardType, input_data: Any, task_description: str) -> Any:
        """Execute task on a single shard."""
        shard = self.shards[shard_type]
        logger.debug(f"Executing on single shard: {shard_type.value}")

        # In a real implementation, this would process the actual input
        # For now, we'll simulate the processing
        result = self._simulate_shard_processing(shard, input_data, task_description)

        return result

    def _execute_ensemble(
        self,
        shard_types: List[ShardType],
        input_data: Any,
        task_description: str,
        confidence_scores: Dict[ShardType, float]
    ) -> Any:
        """Execute task on multiple shards and combine results."""
        self.system_metrics["ensemble_operations"] += 1
        logger.debug(f"Executing ensemble with {len(shard_types)} shards")

        shard_results = []
        shard_weights = []

        for shard_type in shard_types:
            if shard_type in self.shards:
                shard = self.shards[shard_type]
                result = self._simulate_shard_processing(shard, input_data, task_description)
                weight = confidence_scores.get(shard_type, 0.1)

                shard_results.append(result)
                shard_weights.append(weight)

        # Combine results (weighted average for numerical results, voting for categorical)
        combined_result = self._combine_shard_results(shard_results, shard_weights)

        return combined_result

    def _execute_sequential(self, shard_types: List[ShardType], input_data: Any, task_description: str) -> Any:
        """Execute task sequentially through multiple shards."""
        logger.debug(f"Executing sequential processing through {len(shard_types)} shards")

        current_input = input_data
        results = []

        for shard_type in shard_types:
            if shard_type in self.shards:
                shard = self.shards[shard_type]
                result = self._simulate_shard_processing(shard, current_input, task_description)
                results.append(result)
                current_input = result  # Use output as input for next shard

        return results[-1] if results else None  # Return final result

    def _execute_adaptive(self, routing_decision: RoutingDecision, input_data: Any, task_description: str) -> Any:
        """Execute task using adaptive strategy based on context."""
        # Adaptive execution can switch between strategies based on performance
        shard_types = routing_decision.selected_shards

        if len(shard_types) == 1:
            return self._execute_single_shard(shard_types[0], input_data, task_description)
        elif len(shard_types) <= 3:
            return self._execute_ensemble(shard_types, input_data, task_description, routing_decision.confidence_scores)
        else:
            return self._execute_sequential(shard_types, input_data, task_description)

    def _simulate_shard_processing(self, shard: ModelShard, input_data: Any, task_description: str) -> Any:
        """Simulate processing on a shard (placeholder for actual model inference)."""
        # In a real implementation, this would call shard.forward(input_data)
        # For now, we'll simulate different types of outputs based on shard type

        specialization_score = shard.get_specialization_score(task_description)

        # Simulate different output types based on shard
        if shard.shard_type == ShardType.REASONING:
            result = {
                "type": "reasoning_result",
                "conclusion": f"Logical analysis of '{task_description[:30]}...'",
                "confidence": 0.7 + 0.2 * specialization_score,
                "steps": ["premise", "inference", "conclusion"]
            }
        elif shard.shard_type == ShardType.CREATIVITY:
            result = {
                "type": "creative_result",
                "content": f"Creative interpretation of '{task_description[:30]}...'",
                "novelty": 0.6 + 0.3 * specialization_score,
                "alternatives": ["option1", "option2", "option3"]
            }
        elif shard.shard_type == ShardType.CODING:
            result = {
                "type": "code_result",
                "code": f"# Implementation for: {task_description[:30]}...\ndef solution():\n    pass",
                "quality": 0.8 + 0.1 * specialization_score,
                "complexity": "medium"
            }
        elif shard.shard_type == ShardType.ANALYSIS:
            result = {
                "type": "analysis_result",
                "findings": f"Analysis of '{task_description[:30]}...'",
                "depth": 0.7 + 0.2 * specialization_score,
                "categories": ["category1", "category2"]
            }
        else:  # LANGUAGE
            result = {
                "type": "language_result",
                "text": f"Linguistic processing of '{task_description[:30]}...'",
                "fluency": 0.8 + 0.1 * specialization_score,
                "style": "formal"
            }

        # Update shard performance metrics
        quality_score = specialization_score * 0.8 + 0.2  # Base quality with specialization bonus
        shard.update_performance_metrics(quality_score)

        return result

    def _combine_shard_results(self, results: List[Any], weights: List[float]) -> Any:
        """Combine results from multiple shards."""
        if not results:
            return None

        if len(results) == 1:
            return results[0]

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)

        # For now, return the result from the highest weighted shard
        max_weight_idx = weights.index(max(weights))
        primary_result = results[max_weight_idx]

        # Add ensemble metadata
        if isinstance(primary_result, dict):
            primary_result["ensemble_info"] = {
                "num_shards": len(results),
                "weights": weights,
                "combined": True
            }

        return primary_result

    def _update_system_metrics(self, processing_time: float) -> None:
        """Update system-level performance metrics."""
        total_requests = self.system_metrics["total_requests"]
        current_avg = self.system_metrics["average_response_time"]

        # Update rolling average
        self.system_metrics["average_response_time"] = (
            current_avg * (total_requests - 1) + processing_time
        ) / total_requests

    def get_shard_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all shards."""
        performance = {}
        for shard_type, shard in self.shards.items():
            performance[shard_type.value] = shard.get_performance_summary()
        return performance

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and metrics."""
        return {
            "total_shards": len(self.shards),
            "active_shards": sum(1 for shard in self.shards.values() if shard.is_active),
            "routing_strategy": self.routing_strategy.value,
            "system_metrics": self.system_metrics,
            "shard_performance": self.get_shard_performance(),
            "routing_history_size": len(self.router.routing_history)
        }

    def optimize_sharding(self, performance_threshold: float = 0.8) -> Dict[str, Any]:
        """Optimize shard configuration based on performance data."""
        logger.info("Starting shard optimization...")

        optimization_results = {
            "actions_taken": [],
            "performance_improvements": {},
            "recommendations": []
        }

        # Analyze shard performance
        for shard_type, shard in self.shards.items():
            perf_summary = shard.get_performance_summary()

            # Check if shard is underperforming
            if perf_summary["average_quality"] < performance_threshold:
                # Adjust mask ratio to increase capacity
                new_mask_ratio = min(0.9, shard.weight_mask.mask_ratio + 0.1)
                shard.weight_mask.mask_ratio = new_mask_ratio
                shard._apply_shard_masking()  # Re-apply masking

                optimization_results["actions_taken"].append(
                    f"Increased {shard_type.value} shard capacity to {new_mask_ratio:.1%}"
                )

        # Analyze routing efficiency
        routing_stats = self._analyze_routing_efficiency()
        if routing_stats["efficiency"] < 0.7:
            optimization_results["recommendations"].append(
                "Consider adjusting routing strategy or task classification patterns"
            )

        logger.info(f"Optimization completed: {len(optimization_results['actions_taken'])} actions taken")
        return optimization_results

    def _analyze_routing_efficiency(self) -> Dict[str, float]:
        """Analyze routing decision efficiency."""
        if not self.router.routing_history:
            return {"efficiency": 1.0, "accuracy": 1.0}

        # Simple efficiency metrics based on routing history
        total_decisions = len(self.router.routing_history)
        single_shard_decisions = sum(
            1 for decision in self.router.routing_history
            if len(decision.selected_shards) == 1
        )

        efficiency = single_shard_decisions / total_decisions if total_decisions > 0 else 1.0

        return {
            "efficiency": efficiency,
            "accuracy": 0.85,  # Placeholder - would be calculated from feedback
            "total_decisions": total_decisions
        }


class LoadBalancer:
    """Simple load balancer for shard utilization."""

    def __init__(self):
        self.request_counts: Dict[ShardType, int] = {}

    def get_least_utilized_shard(self, available_shards: List[ShardType]) -> ShardType:
        """Get the least utilized shard from available options."""
        if not available_shards:
            return available_shards[0] if available_shards else None

        # Return shard with lowest request count
        min_requests = float('inf')
        selected_shard = available_shards[0]

        for shard_type in available_shards:
            request_count = self.request_counts.get(shard_type, 0)
            if request_count < min_requests:
                min_requests = request_count
                selected_shard = shard_type

        # Update count
        self.request_counts[selected_shard] = self.request_counts.get(selected_shard, 0) + 1

        return selected_shard


# Export main classes
__all__ = [
    'ModelShardingSystem',
    'ModelShard',
    'ShardRouter',
    'WeightMask',
    'ShardType',
    'RoutingStrategy',
    'ShardCapability',
    'RoutingDecision',
    'LoadBalancer'
]


if __name__ == "__main__":
    # Example usage and testing
    logger.info("Testing ModelShardingSystem...")

    # Create a simple base model for testing
    class SimpleTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(128, 128)
            self.feedforward = nn.Linear(128, 256)
            self.output = nn.Linear(256, 64)

        def forward(self, x):
            x = F.relu(self.attention(x))
            x = F.relu(self.feedforward(x))
            x = self.output(x)
            return x

    # Initialize sharding system
    base_model = SimpleTransformer()
    sharding_system = ModelShardingSystem(
        base_model=base_model,
        shard_types=[ShardType.REASONING, ShardType.CREATIVITY, ShardType.CODING, ShardType.ANALYSIS],
        routing_strategy=RoutingStrategy.ADAPTIVE
    )

    # Test different types of tasks
    test_tasks = [
        ("Solve this logical puzzle step by step", "reasoning task"),
        ("Write a creative story about space exploration", "creativity task"),
        ("Implement a binary search algorithm in Python", "coding task"),
        ("Analyze the performance implications of this data structure", "analysis task"),
        ("Translate this text and improve its style", "language task")
    ]

    print("\n=== Testing Task Processing ===")
    for task_desc, task_type in test_tasks:
        print(f"\nProcessing: {task_desc}")
        result = sharding_system.process_task(
            task_description=task_desc,
            input_data={"text": task_desc},
            context={"domain": task_type}
        )

        print(f"Selected shards: {result['routing_decision']['selected_shards']}")
        print(f"Strategy: {result['routing_decision']['strategy']}")
        print(f"Processing time: {result['performance']['processing_time']:.3f}s")
        if "error" not in result:
            print(f"Result type: {result['result']['type']}")

    # Test system status and optimization
    print(f"\n=== System Status ===")
    status = sharding_system.get_system_status()
    print(f"Total requests: {status['system_metrics']['total_requests']}")
    print(f"Active shards: {status['active_shards']}/{status['total_shards']}")
    print(f"Average response time: {status['system_metrics']['average_response_time']:.3f}s")

    # Test optimization
    print(f"\n=== Optimization Test ===")
    optimization_results = sharding_system.optimize_sharding(performance_threshold=0.7)
    print(f"Actions taken: {optimization_results['actions_taken']}")
    print(f"Recommendations: {optimization_results['recommendations']}")

    logger.info("ModelShardingSystem testing completed")