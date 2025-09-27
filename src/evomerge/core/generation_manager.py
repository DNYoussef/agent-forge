"""
GenerationManager - FSM-based generation lifecycle management for Phase 2 EvoMerge.

Implements the user's specification:
- 50 generation iterations
- Maximum 16 models stored at any time (current + previous generation)
- Automatic N-2 generation deletion
- FSM-first design with explicit states and transitions
- Thread-safe operations for concurrent evaluation

States: IDLE -> INITIALIZING -> RUNNING -> BENCHMARKING -> SELECTING -> CLEANING -> COMPLETE
"""

import torch
import torch.nn as nn
import threading
import time
import json
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum, auto
from pathlib import Path
import logging
from abc import ABC, abstractmethod

from ..utils.model_operations import ModelOperations, get_model_operations
from ..utils.evaluator_factory import EvaluatorFactory, EvaluatorConfig, EvaluatorType, MetricType

logger = logging.getLogger(__name__)


class GenerationState(Enum):
    """FSM states for generation lifecycle."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    BENCHMARKING = "benchmarking"
    SELECTING = "selecting"
    CLEANING = "cleaning"
    COMPLETE = "complete"
    ERROR = "error"


class GenerationEvent(Enum):
    """FSM events for generation transitions."""
    START = "start"
    MODELS_CREATED = "models_created"
    BENCHMARK_COMPLETE = "benchmark_complete"
    SELECTION_COMPLETE = "selection_complete"
    CLEANUP_COMPLETE = "cleanup_complete"
    ITERATION_COMPLETE = "iteration_complete"
    ERROR_OCCURRED = "error_occurred"
    RESET = "reset"


@dataclass
class ModelInfo:
    """Information about a model in the generation."""
    model: nn.Module
    fitness_score: float
    generation: int
    parent_ids: List[str]
    creation_method: str  # "merge", "mutate", "initial"
    lineage_color: str   # For 3D visualization
    metadata: Dict[str, Any]
    model_id: str


@dataclass
class GenerationMetrics:
    """Metrics for a generation."""
    generation_number: int
    model_count: int
    avg_fitness: float
    best_fitness: float
    worst_fitness: float
    diversity_score: float
    winner_count: int
    loser_count: int
    processing_time_ms: float
    memory_usage_mb: float


@dataclass
class GenerationResult:
    """Result of generation processing."""
    generation: int
    models: List[ModelInfo]
    metrics: GenerationMetrics
    winners: List[ModelInfo]
    losers: List[ModelInfo]
    state: GenerationState
    error_message: Optional[str] = None


class GenerationStateError(Exception):
    """Exception for invalid state transitions."""
    pass


class GenerationManager:
    """
    FSM-based generation manager for Phase 2 EvoMerge.

    Manages the complete 50-generation lifecycle with:
    - Maximum 16 models constraint enforcement
    - Winner/Loser selection logic
    - Automatic cleanup of old generations
    - Thread-safe operations
    - 3D visualization data export
    """

    # FSM transition table
    VALID_TRANSITIONS = {
        GenerationState.IDLE: [GenerationState.INITIALIZING, GenerationState.ERROR],
        GenerationState.INITIALIZING: [GenerationState.RUNNING, GenerationState.ERROR],
        GenerationState.RUNNING: [GenerationState.BENCHMARKING, GenerationState.ERROR],
        GenerationState.BENCHMARKING: [GenerationState.SELECTING, GenerationState.ERROR],
        GenerationState.SELECTING: [GenerationState.CLEANING, GenerationState.ERROR],
        GenerationState.CLEANING: [GenerationState.RUNNING, GenerationState.COMPLETE, GenerationState.ERROR],
        GenerationState.COMPLETE: [GenerationState.IDLE],
        GenerationState.ERROR: [GenerationState.IDLE]
    }

    def __init__(self,
                 max_generations: int = 50,
                 max_models_per_generation: int = 8,
                 max_total_models: int = 16,
                 evaluator_config: Optional[EvaluatorConfig] = None,
                 enable_3d_export: bool = True):
        """
        Initialize GenerationManager.

        Args:
            max_generations: Maximum number of generations (50 for Phase 2)
            max_models_per_generation: Models per generation (8 for Phase 2)
            max_total_models: Maximum total models stored (16 for Phase 2)
            evaluator_config: Configuration for fitness evaluation
            enable_3d_export: Whether to export data for 3D visualization
        """
        self.max_generations = max_generations
        self.max_models_per_generation = max_models_per_generation
        self.max_total_models = max_total_models
        self.enable_3d_export = enable_3d_export

        # FSM state management
        self._state = GenerationState.IDLE
        self._state_lock = threading.Lock()

        # Generation tracking
        self.current_generation = 0
        self.generations: Dict[int, List[ModelInfo]] = {}
        self.generation_metrics: Dict[int, GenerationMetrics] = {}
        self.generation_history: List[GenerationResult] = []

        # Model operations
        self.model_ops = get_model_operations()

        # Fitness evaluation
        if evaluator_config is None:
            evaluator_config = EvaluatorConfig(
                evaluator_type=EvaluatorType.EFFICIENCY,
                metric_types=[MetricType.ACCURACY, MetricType.EFFICIENCY, MetricType.INFERENCE_TIME]
            )
        self.evaluator = EvaluatorFactory.create_evaluator(evaluator_config)

        # 3D visualization data
        self.visualization_data = {
            "generations": [],
            "lineages": {},
            "merges": [],
            "mutations": []
        }

        # Event handlers
        self._event_handlers: Dict[GenerationEvent, List[Callable]] = {}

        logger.info(f"GenerationManager initialized: {max_generations} generations, {max_total_models} max models")

    def get_state(self) -> GenerationState:
        """Get current FSM state (thread-safe)."""
        with self._state_lock:
            return self._state

    def _transition_to(self, new_state: GenerationState, event: GenerationEvent):
        """
        Perform FSM state transition with validation.

        Args:
            new_state: Target state
            event: Triggering event

        Raises:
            GenerationStateError: If transition is invalid
        """
        with self._state_lock:
            current_state = self._state

            if new_state not in self.VALID_TRANSITIONS.get(current_state, []):
                raise GenerationStateError(
                    f"Invalid transition: {current_state.value} -> {new_state.value} on event {event.value}"
                )

            logger.info(f"State transition: {current_state.value} -> {new_state.value} (event: {event.value})")
            self._state = new_state

            # Trigger event handlers
            self._trigger_event_handlers(event, current_state, new_state)

    def _trigger_event_handlers(self, event: GenerationEvent, old_state: GenerationState, new_state: GenerationState):
        """Trigger registered event handlers."""
        handlers = self._event_handlers.get(event, [])
        for handler in handlers:
            try:
                handler(event, old_state, new_state)
            except Exception as e:
                logger.error(f"Event handler failed for {event.value}: {e}")

    def register_event_handler(self, event: GenerationEvent, handler: Callable):
        """Register event handler for FSM transitions."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def initialize_from_phase1(self, phase1_models: List[nn.Module]) -> bool:
        """
        Initialize generation 0 from Phase 1 output.

        Args:
            phase1_models: 3 models from Phase 1 Cognate training

        Returns:
            True if initialization successful
        """
        try:
            self._transition_to(GenerationState.INITIALIZING, GenerationEvent.START)

            if len(phase1_models) != 3:
                raise ValueError(f"Expected 3 Phase 1 models, got {len(phase1_models)}")

            # Create initial ModelInfo for Phase 1 models
            initial_models = []
            lineage_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]  # Red, Teal, Blue

            for i, model in enumerate(phase1_models):
                model_info = ModelInfo(
                    model=self.model_ops.clone_model(model),
                    fitness_score=0.0,  # Will be evaluated
                    generation=0,
                    parent_ids=[],
                    creation_method="initial",
                    lineage_color=lineage_colors[i],
                    metadata={"phase1_model_index": i},
                    model_id=f"gen0_model_{i}"
                )
                initial_models.append(model_info)

            self.generations[0] = initial_models
            self.current_generation = 0

            # Export for 3D visualization
            if self.enable_3d_export:
                self._export_generation_for_3d(0, initial_models, "initialization")

            self._transition_to(GenerationState.RUNNING, GenerationEvent.MODELS_CREATED)
            logger.info(f"Initialized generation 0 with {len(initial_models)} models")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self._transition_to(GenerationState.ERROR, GenerationEvent.ERROR_OCCURRED)
            return False

    def run_generation_cycle(self, merger_factory, selection_strategy: str = "winner_loser") -> GenerationResult:
        """
        Run a complete generation cycle: create -> benchmark -> select -> clean.

        Args:
            merger_factory: Factory for creating new models (3 -> 8)
            selection_strategy: Strategy for model selection

        Returns:
            GenerationResult with metrics and new models
        """
        start_time = time.time()

        try:
            current_state = self.get_state()
            if current_state not in [GenerationState.RUNNING, GenerationState.CLEANING]:
                raise GenerationStateError(f"Cannot run cycle from state: {current_state.value}")

            # Step 1: Create 8 models from current generation
            current_models = self.generations[self.current_generation]
            if len(current_models) == 3:
                # First iteration: create 8 from 3
                new_models = self._create_8_from_3(current_models, merger_factory)
            else:
                # Subsequent iterations: already have 8 models
                new_models = current_models

            # Step 2: Benchmark all models
            self._transition_to(GenerationState.BENCHMARKING, GenerationEvent.MODELS_CREATED)
            benchmarked_models = self._benchmark_models(new_models)
            self._transition_to(GenerationState.SELECTING, GenerationEvent.BENCHMARK_COMPLETE)

            # Step 3: Selection (winner/loser logic)
            winners, losers = self._select_winners_and_losers(benchmarked_models)
            next_generation_models = self._create_next_generation(winners, losers, merger_factory)
            self._transition_to(GenerationState.CLEANING, GenerationEvent.SELECTION_COMPLETE)

            # Step 4: Store new generation and cleanup
            self.current_generation += 1
            self.generations[self.current_generation] = next_generation_models
            self._cleanup_old_generations()

            # Calculate metrics
            metrics = self._calculate_generation_metrics(self.current_generation)
            self.generation_metrics[self.current_generation] = metrics

            # Create result
            result = GenerationResult(
                generation=self.current_generation,
                models=next_generation_models,
                metrics=metrics,
                winners=winners,
                losers=losers,
                state=self.get_state()
            )

            # Export for 3D visualization
            if self.enable_3d_export:
                self._export_generation_for_3d(self.current_generation, next_generation_models, "generation")

            # Check completion
            if self.current_generation >= self.max_generations:
                self._transition_to(GenerationState.COMPLETE, GenerationEvent.ITERATION_COMPLETE)
            else:
                self._transition_to(GenerationState.RUNNING, GenerationEvent.CLEANUP_COMPLETE)

            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Generation {self.current_generation} completed in {processing_time:.2f}ms")

            return result

        except Exception as e:
            logger.error(f"Generation cycle failed: {e}")
            self._transition_to(GenerationState.ERROR, GenerationEvent.ERROR_OCCURRED)
            return GenerationResult(
                generation=self.current_generation,
                models=[],
                metrics=None,
                winners=[],
                losers=[],
                state=GenerationState.ERROR,
                error_message=str(e)
            )

    def _create_8_from_3(self, models: List[ModelInfo], merger_factory) -> List[ModelInfo]:
        """Create 8 diverse models from 3 input models using merger techniques."""
        if len(models) != 3:
            raise ValueError(f"Expected 3 models, got {len(models)}")

        new_models = []

        # Use different merger techniques to create diversity
        merger_configs = [
            {"strategy": "slerp", "weight": 0.3},
            {"strategy": "slerp", "weight": 0.7},
            {"strategy": "ties", "threshold": 0.5},
            {"strategy": "ties", "threshold": 0.2},
            {"strategy": "dare", "dropout_rate": 0.1},
            {"strategy": "dare", "dropout_rate": 0.3},
            {"strategy": "weighted_average", "weights": [0.5, 0.3, 0.2]},
            {"strategy": "weighted_average", "weights": [0.2, 0.6, 0.2]}
        ]

        for i, config in enumerate(merger_configs):
            try:
                # Use merger factory to create new model
                merged_model = merger_factory.create_merged_model(
                    [m.model for m in models],
                    config
                )

                # Determine lineage color (blend of parent colors)
                parent_colors = [m.lineage_color for m in models]
                blended_color = self._blend_colors(parent_colors)

                model_info = ModelInfo(
                    model=merged_model,
                    fitness_score=0.0,  # Will be evaluated
                    generation=self.current_generation + 1,
                    parent_ids=[m.model_id for m in models],
                    creation_method=f"merge_{config['strategy']}",
                    lineage_color=blended_color,
                    metadata={"merge_config": config},
                    model_id=f"gen{self.current_generation + 1}_model_{i}"
                )
                new_models.append(model_info)

            except Exception as e:
                logger.error(f"Failed to create model {i} with config {config}: {e}")

        return new_models

    def _benchmark_models(self, models: List[ModelInfo]) -> List[ModelInfo]:
        """Benchmark all models and update fitness scores."""
        # TODO: Replace with actual data from Phase 2 pipeline
        dummy_data = torch.randn(32, 3, 224, 224)  # Placeholder

        for model_info in models:
            try:
                # Use evaluator to get metrics
                metrics = self.evaluator.evaluate(model_info.model, dummy_data)

                # Calculate composite fitness score
                if hasattr(self.evaluator, 'get_composite_score'):
                    fitness = self.evaluator.get_composite_score(metrics)
                else:
                    # Simple average for basic evaluators
                    fitness = sum(metrics.values()) / len(metrics)

                model_info.fitness_score = fitness
                model_info.metadata["evaluation_metrics"] = metrics

            except Exception as e:
                logger.error(f"Benchmarking failed for model {model_info.model_id}: {e}")
                model_info.fitness_score = 0.0

        return models

    def _select_winners_and_losers(self, models: List[ModelInfo]) -> Tuple[List[ModelInfo], List[ModelInfo]]:
        """
        Implement Phase 2 winner/loser selection logic.

        Top 2 performers -> winners (will be mutated into 6 models)
        Bottom 6 -> losers (will be merged into 2 models)
        """
        if len(models) != 8:
            raise ValueError(f"Expected 8 models for selection, got {len(models)}")

        # Sort by fitness (descending)
        sorted_models = sorted(models, key=lambda m: m.fitness_score, reverse=True)

        winners = sorted_models[:2]  # Top 2
        losers = sorted_models[2:]   # Bottom 6

        logger.info(f"Selected {len(winners)} winners and {len(losers)} losers")
        return winners, losers

    def _create_next_generation(self, winners: List[ModelInfo], losers: List[ModelInfo], merger_factory) -> List[ModelInfo]:
        """
        Create next generation: 6 winner children + 2 loser children = 8 models.

        Winners (2) -> mutate into 6 models
        Losers (6) -> merge into 2 models
        """
        next_gen_models = []

        # Create 6 models from 2 winners through mutation
        for i in range(6):
            winner_idx = i % 2  # Alternate between the 2 winners
            winner = winners[winner_idx]

            try:
                # Clone and mutate
                mutated_model = self.model_ops.clone_model(winner.model)
                self._mutate_model(mutated_model, mutation_rate=0.1)

                model_info = ModelInfo(
                    model=mutated_model,
                    fitness_score=0.0,
                    generation=self.current_generation + 1,
                    parent_ids=[winner.model_id],
                    creation_method="mutate",
                    lineage_color=winner.lineage_color,  # Inherit parent color
                    metadata={"parent_fitness": winner.fitness_score, "mutation_rate": 0.1},
                    model_id=f"gen{self.current_generation + 1}_winner_child_{i}"
                )
                next_gen_models.append(model_info)

            except Exception as e:
                logger.error(f"Failed to create winner child {i}: {e}")

        # Create 2 models from 6 losers through merging
        loser_groups = [losers[:3], losers[3:]]  # Split into two groups of 3

        for i, group in enumerate(loser_groups):
            try:
                # Merge group of losers
                merged_model = merger_factory.create_merged_model(
                    [m.model for m in group],
                    {"strategy": "weighted_average", "weights": [1/3, 1/3, 1/3]}
                )

                # Blend colors from loser group
                parent_colors = [m.lineage_color for m in group]
                blended_color = self._blend_colors(parent_colors)

                model_info = ModelInfo(
                    model=merged_model,
                    fitness_score=0.0,
                    generation=self.current_generation + 1,
                    parent_ids=[m.model_id for m in group],
                    creation_method="merge_losers",
                    lineage_color=blended_color,
                    metadata={"parent_count": len(group)},
                    model_id=f"gen{self.current_generation + 1}_loser_child_{i}"
                )
                next_gen_models.append(model_info)

            except Exception as e:
                logger.error(f"Failed to create loser child {i}: {e}")

        return next_gen_models

    def _mutate_model(self, model: nn.Module, mutation_rate: float = 0.1):
        """Apply Gaussian mutation to model parameters."""
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * mutation_rate
                    param.add_(noise)

    def _cleanup_old_generations(self):
        """Enforce max_total_models constraint by deleting old generations."""
        # Keep current and previous generation only (N-2 deletion)
        generations_to_keep = [self.current_generation, self.current_generation - 1]
        generations_to_delete = []

        for gen_num in list(self.generations.keys()):
            if gen_num not in generations_to_keep and gen_num >= 0:
                generations_to_delete.append(gen_num)

        for gen_num in generations_to_delete:
            del self.generations[gen_num]
            if gen_num in self.generation_metrics:
                del self.generation_metrics[gen_num]

        total_models = sum(len(models) for models in self.generations.values())
        logger.info(f"Cleanup: {len(generations_to_delete)} generations deleted, {total_models} models remaining")

        if total_models > self.max_total_models:
            logger.warning(f"Model count {total_models} exceeds limit {self.max_total_models}")

    def _calculate_generation_metrics(self, generation: int) -> GenerationMetrics:
        """Calculate comprehensive metrics for a generation."""
        models = self.generations[generation]
        fitness_scores = [m.fitness_score for m in models]

        if not fitness_scores:
            return GenerationMetrics(
                generation_number=generation,
                model_count=0,
                avg_fitness=0.0,
                best_fitness=0.0,
                worst_fitness=0.0,
                diversity_score=0.0,
                winner_count=0,
                loser_count=0,
                processing_time_ms=0.0,
                memory_usage_mb=0.0
            )

        # Calculate diversity as average pairwise distance
        diversity_score = 0.0
        if len(models) > 1:
            total_distance = 0.0
            pair_count = 0
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    distance = self.model_ops.calculate_model_distance(
                        models[i].model, models[j].model
                    )
                    total_distance += distance
                    pair_count += 1
            diversity_score = total_distance / pair_count if pair_count > 0 else 0.0

        return GenerationMetrics(
            generation_number=generation,
            model_count=len(models),
            avg_fitness=sum(fitness_scores) / len(fitness_scores),
            best_fitness=max(fitness_scores),
            worst_fitness=min(fitness_scores),
            diversity_score=diversity_score,
            winner_count=2 if generation > 0 else 0,
            loser_count=6 if generation > 0 else 0,
            processing_time_ms=0.0,  # Will be set by caller
            memory_usage_mb=self._estimate_memory_usage()
        )

    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB."""
        total_memory = 0.0
        for models in self.generations.values():
            for model_info in models:
                summary = self.model_ops.get_model_summary(model_info.model)
                total_memory += summary["memory_usage_mb"]
        return total_memory

    def _blend_colors(self, colors: List[str]) -> str:
        """Blend multiple hex colors for lineage visualization."""
        # Simple color blending by averaging RGB values
        if not colors:
            return "#808080"  # Gray default

        total_r, total_g, total_b = 0, 0, 0
        for color in colors:
            if color.startswith("#"):
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                total_r += r
                total_g += g
                total_b += b

        count = len(colors)
        avg_r = min(255, total_r // count)
        avg_g = min(255, total_g // count)
        avg_b = min(255, total_b // count)

        return f"#{avg_r:02x}{avg_g:02x}{avg_b:02x}"

    def _export_generation_for_3d(self, generation: int, models: List[ModelInfo], event_type: str):
        """Export generation data for 3D tree visualization."""
        if not self.enable_3d_export:
            return

        generation_data = {
            "generation": generation,
            "timestamp": time.time(),
            "event_type": event_type,
            "models": []
        }

        for model in models:
            model_data = {
                "id": model.model_id,
                "generation": model.generation,
                "fitness": model.fitness_score,
                "parents": model.parent_ids,
                "creation_method": model.creation_method,
                "lineage_color": model.lineage_color,
                "metadata": model.metadata
            }
            generation_data["models"].append(model_data)

        self.visualization_data["generations"].append(generation_data)

    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data formatted for 3D tree visualization."""
        return self.visualization_data.copy()

    def export_results(self, output_path: str):
        """Export complete generation results to JSON."""
        results = {
            "config": {
                "max_generations": self.max_generations,
                "max_models_per_generation": self.max_models_per_generation,
                "max_total_models": self.max_total_models
            },
            "final_generation": self.current_generation,
            "state": self._state.value,
            "metrics": {gen: asdict(metrics) for gen, metrics in self.generation_metrics.items()},
            "visualization_data": self.visualization_data
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results exported to {output_path}")

    def get_best_model(self) -> Optional[ModelInfo]:
        """Get the best model from all generations."""
        best_model = None
        best_fitness = -float('inf')

        for models in self.generations.values():
            for model in models:
                if model.fitness_score > best_fitness:
                    best_fitness = model.fitness_score
                    best_model = model

        return best_model

    def reset(self):
        """Reset manager to initial state."""
        with self._state_lock:
            self._state = GenerationState.IDLE
            self.current_generation = 0
            self.generations.clear()
            self.generation_metrics.clear()
            self.generation_history.clear()
            self.visualization_data = {
                "generations": [],
                "lineages": {},
                "merges": [],
                "mutations": []
            }

        logger.info("GenerationManager reset to initial state")