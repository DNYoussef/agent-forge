"""
Phase 2 Orchestrator - Complete EvoMerge Workflow Integration

Integrates all Phase 2 components:
- GenerationManager (FSM-based lifecycle, 16-model constraint)
- MergerOperatorFactory (3→8 model creation pipeline)
- ModelOperations (consolidated clone/distance operations)
- EvaluatorFactory (standardized fitness evaluation)
- 3D Visualization data export

Implements the complete user specification:
- 3 models → 8 variants → benchmark → select → 50 generations
- Winner/Loser logic: top 2 mutate to 6, bottom 6 merge to 2
- Max 16 models storage, automatic N-2 cleanup
- Real-time 3D tree visualization
- Thread-safe concurrent operations
"""

import torch
import torch.nn as nn
import asyncio
import threading
import time
import json
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from enum import Enum

from .core.generation_manager import GenerationManager, GenerationState, GenerationEvent, ModelInfo, GenerationResult
from .core.merger_operator_factory import MergerOperatorFactory, MergerResult, DiversityStrategy
from .utils.model_operations import get_model_operations, ModelOperations
from .utils.evaluator_factory import EvaluatorFactory, EvaluatorConfig, EvaluatorType, MetricType

logger = logging.getLogger(__name__)


class Phase2State(Enum):
    """Phase 2 orchestrator states."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class Phase2Config:
    """Configuration for Phase 2 EvoMerge."""
    max_generations: int = 50
    max_models_per_generation: int = 8
    max_total_models: int = 16
    diversity_strategy: DiversityStrategy = DiversityStrategy.TECHNIQUE_ROTATION
    enable_3d_visualization: bool = True
    enable_quality_validation: bool = True
    max_concurrent_operations: int = 4
    output_directory: str = "./phase2_results"


@dataclass
class Phase2Progress:
    """Progress tracking for Phase 2."""
    current_generation: int
    total_generations: int
    models_created: int
    best_fitness: float
    avg_fitness: float
    diversity_score: float
    processing_time_seconds: float
    estimated_remaining_seconds: float
    state: Phase2State


@dataclass
class Phase2Result:
    """Final result of Phase 2 EvoMerge."""
    best_model: nn.Module
    best_fitness: float
    total_generations: int
    total_models_created: int
    total_processing_time: float
    generation_metrics: Dict[int, Dict[str, Any]]
    visualization_data: Dict[str, Any]
    config: Phase2Config
    success: bool
    error_message: Optional[str] = None


class Phase2Orchestrator:
    """
    Complete Phase 2 EvoMerge orchestrator.

    Coordinates all Phase 2 components to implement the user's specification:
    - Takes 3 models from Phase 1
    - Runs 50 generations of evolutionary model merging
    - Maintains max 16 models constraint
    - Exports 3D visualization data
    - Returns best evolved model for Phase 3
    """

    def __init__(self, config: Phase2Config):
        """
        Initialize Phase 2 orchestrator.

        Args:
            config: Phase 2 configuration
        """
        self.config = config
        self.state = Phase2State.IDLE

        # Initialize components
        self.generation_manager = GenerationManager(
            max_generations=config.max_generations,
            max_models_per_generation=config.max_models_per_generation,
            max_total_models=config.max_total_models,
            enable_3d_export=config.enable_3d_visualization
        )

        self.merger_factory = MergerOperatorFactory(
            diversity_strategy=config.diversity_strategy,
            enable_quality_validation=config.enable_quality_validation,
            max_concurrent_operations=config.max_concurrent_operations
        )

        self.model_ops = get_model_operations()

        # Progress tracking
        self.start_time: Optional[float] = None
        self.progress_callbacks: List[Callable[[Phase2Progress], None]] = []
        self.generation_times: List[float] = []

        # Results storage
        self.generation_results: List[GenerationResult] = []
        self.best_model_history: List[Tuple[int, float, nn.Module]] = []

        # Thread safety
        self._state_lock = threading.Lock()

        # Output directory
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Phase2Orchestrator initialized: {config.max_generations} generations, {config.max_total_models} max models")

    def register_progress_callback(self, callback: Callable[[Phase2Progress], None]):
        """Register callback for progress updates."""
        self.progress_callbacks.append(callback)

    def _emit_progress(self, progress: Phase2Progress):
        """Emit progress update to all registered callbacks."""
        for callback in self.progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")

    def _update_state(self, new_state: Phase2State):
        """Update orchestrator state (thread-safe)."""
        with self._state_lock:
            logger.info(f"Phase 2 state: {self.state.value} -> {new_state.value}")
            self.state = new_state

    async def execute_phase2(self, phase1_models: List[nn.Module]) -> Phase2Result:
        """
        Execute complete Phase 2 EvoMerge workflow.

        Args:
            phase1_models: 3 models from Phase 1 Cognate training

        Returns:
            Phase2Result with best model and complete metrics

        Raises:
            ValueError: If input models are invalid
            RuntimeError: If execution fails
        """
        try:
            self._update_state(Phase2State.INITIALIZING)
            self.start_time = time.time()

            # Validate inputs
            if len(phase1_models) != 3:
                raise ValueError(f"Expected 3 Phase 1 models, got {len(phase1_models)}")

            # Initialize generation manager with Phase 1 models
            if not self.generation_manager.initialize_from_phase1(phase1_models):
                raise RuntimeError("Failed to initialize generation manager")

            self._update_state(Phase2State.RUNNING)

            # Run 50 generation evolutionary loop
            for generation in range(1, self.config.max_generations + 1):
                generation_start = time.time()

                logger.info(f"Starting generation {generation}/{self.config.max_generations}")

                # Run generation cycle
                result = self.generation_manager.run_generation_cycle(
                    merger_factory=self.merger_factory,
                    selection_strategy="winner_loser"
                )

                if result.state == GenerationState.ERROR:
                    raise RuntimeError(f"Generation {generation} failed: {result.error_message}")

                # Track results
                self.generation_results.append(result)
                generation_time = time.time() - generation_start
                self.generation_times.append(generation_time)

                # Update best model tracking
                if result.metrics:
                    best_in_generation = max(result.models, key=lambda m: m.fitness_score)
                    self.best_model_history.append((generation, best_in_generation.fitness_score, best_in_generation.model))

                # Emit progress update
                progress = self._calculate_progress(generation)
                self._emit_progress(progress)

                # Save intermediate results
                if generation % 10 == 0:  # Save every 10 generations
                    await self._save_intermediate_results(generation)

                # Check for early termination (optional)
                if self._should_terminate_early(result):
                    logger.info(f"Early termination at generation {generation}")
                    break

            self._update_state(Phase2State.COMPLETED)

            # Create final result
            final_result = await self._create_final_result()

            # Save complete results
            await self._save_final_results(final_result)

            logger.info(f"Phase 2 completed: {final_result.total_generations} generations, best fitness: {final_result.best_fitness:.4f}")
            return final_result

        except Exception as e:
            self._update_state(Phase2State.ERROR)
            logger.error(f"Phase 2 execution failed: {e}")

            # Create error result
            return Phase2Result(
                best_model=phase1_models[0],  # Fallback to first input model
                best_fitness=0.0,
                total_generations=0,
                total_models_created=0,
                total_processing_time=time.time() - (self.start_time or time.time()),
                generation_metrics={},
                visualization_data={},
                config=self.config,
                success=False,
                error_message=str(e)
            )

    def _calculate_progress(self, current_generation: int) -> Phase2Progress:
        """Calculate current progress metrics."""
        total_processing_time = time.time() - (self.start_time or time.time())

        # Calculate metrics from recent results
        if self.generation_results:
            recent_results = self.generation_results[-5:]  # Last 5 generations
            avg_fitness = sum(r.metrics.avg_fitness for r in recent_results if r.metrics) / len(recent_results)
            best_fitness = max(r.metrics.best_fitness for r in recent_results if r.metrics)
            diversity_score = sum(r.metrics.diversity_score for r in recent_results if r.metrics) / len(recent_results)
        else:
            avg_fitness = 0.0
            best_fitness = 0.0
            diversity_score = 0.0

        # Estimate remaining time
        if self.generation_times:
            avg_generation_time = sum(self.generation_times) / len(self.generation_times)
            remaining_generations = self.config.max_generations - current_generation
            estimated_remaining = avg_generation_time * remaining_generations
        else:
            estimated_remaining = 0.0

        # Count total models created
        total_models_created = sum(len(r.models) for r in self.generation_results)

        return Phase2Progress(
            current_generation=current_generation,
            total_generations=self.config.max_generations,
            models_created=total_models_created,
            best_fitness=best_fitness,
            avg_fitness=avg_fitness,
            diversity_score=diversity_score,
            processing_time_seconds=total_processing_time,
            estimated_remaining_seconds=estimated_remaining,
            state=self.state
        )

    def _should_terminate_early(self, result: GenerationResult) -> bool:
        """Check if evolution should terminate early."""
        # Terminate if fitness has converged
        if len(self.best_model_history) >= 10:
            recent_fitness = [fitness for _, fitness, _ in self.best_model_history[-10:]]
            fitness_variance = np.var(recent_fitness) if len(recent_fitness) > 1 else 1.0

            if fitness_variance < 0.001:  # Very low variance
                logger.info("Early termination: fitness converged")
                return True

        # Terminate if diversity is too low
        if result.metrics and result.metrics.diversity_score < 0.01:
            logger.info("Early termination: diversity too low")
            return True

        return False

    async def _create_final_result(self) -> Phase2Result:
        """Create final Phase 2 result."""
        total_processing_time = time.time() - (self.start_time or time.time())

        # Find best model across all generations
        if self.best_model_history:
            best_generation, best_fitness, best_model = max(self.best_model_history, key=lambda x: x[1])
        else:
            # Fallback: get current best model
            best_model_info = self.generation_manager.get_best_model()
            if best_model_info:
                best_model = best_model_info.model
                best_fitness = best_model_info.fitness_score
                best_generation = best_model_info.generation
            else:
                best_model = nn.Module()  # Empty fallback
                best_fitness = 0.0
                best_generation = 0

        # Collect generation metrics
        generation_metrics = {}
        for result in self.generation_results:
            if result.metrics:
                generation_metrics[result.generation] = asdict(result.metrics)

        # Get visualization data
        visualization_data = self.generation_manager.get_visualization_data()

        # Count total models created
        total_models_created = sum(len(r.models) for r in self.generation_results)

        return Phase2Result(
            best_model=best_model,
            best_fitness=best_fitness,
            total_generations=len(self.generation_results),
            total_models_created=total_models_created,
            total_processing_time=total_processing_time,
            generation_metrics=generation_metrics,
            visualization_data=visualization_data,
            config=self.config,
            success=True
        )

    async def _save_intermediate_results(self, generation: int):
        """Save intermediate results during execution."""
        try:
            intermediate_file = self.output_dir / f"generation_{generation:03d}.json"

            progress = self._calculate_progress(generation)
            intermediate_data = {
                "generation": generation,
                "progress": asdict(progress),
                "best_model_history": [(gen, fitness, "model_saved_separately") for gen, fitness, _ in self.best_model_history],
                "visualization_data": self.generation_manager.get_visualization_data()
            }

            with open(intermediate_file, 'w') as f:
                json.dump(intermediate_data, f, indent=2)

            logger.debug(f"Saved intermediate results: {intermediate_file}")

        except Exception as e:
            logger.warning(f"Failed to save intermediate results: {e}")

    async def _save_final_results(self, result: Phase2Result):
        """Save complete final results."""
        try:
            # Save JSON results
            result_file = self.output_dir / "phase2_complete_results.json"

            # Convert result to serializable format
            result_data = asdict(result)
            result_data["best_model"] = "saved_separately"  # Don't serialize model

            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2)

            # Save best model
            if result.success and result.best_model:
                model_file = self.output_dir / "phase2_best_model.pth"
                torch.save({
                    'model_state_dict': result.best_model.state_dict(),
                    'model_class': result.best_model.__class__.__name__,
                    'fitness': result.best_fitness,
                    'generation': result.total_generations
                }, model_file)

            # Save visualization data
            viz_file = self.output_dir / "phase2_visualization.json"
            with open(viz_file, 'w') as f:
                json.dump(result.visualization_data, f, indent=2)

            # Export generation manager results
            self.generation_manager.export_results(str(self.output_dir / "generation_manager_results.json"))

            logger.info(f"Saved final results to {self.output_dir}")

        except Exception as e:
            logger.error(f"Failed to save final results: {e}")

    def get_current_progress(self) -> Phase2Progress:
        """Get current execution progress."""
        current_gen = self.generation_manager.current_generation
        return self._calculate_progress(current_gen)

    def get_visualization_data(self) -> Dict[str, Any]:
        """Get current visualization data for 3D tree."""
        return self.generation_manager.get_visualization_data()

    def stop_execution(self):
        """Stop execution gracefully."""
        # Implementation would set a stop flag that the main loop checks
        logger.info("Stop execution requested")
        # For now, just log - would need more sophisticated stopping mechanism

    def reset(self):
        """Reset orchestrator to initial state."""
        self._update_state(Phase2State.IDLE)
        self.generation_manager.reset()
        self.generation_results.clear()
        self.best_model_history.clear()
        self.generation_times.clear()
        self.start_time = None


# Factory function for easy instantiation
def create_phase2_orchestrator(config: Optional[Phase2Config] = None) -> Phase2Orchestrator:
    """
    Create Phase2Orchestrator with default or custom configuration.

    Args:
        config: Optional custom configuration

    Returns:
        Configured Phase2Orchestrator instance
    """
    if config is None:
        config = Phase2Config()

    return Phase2Orchestrator(config)


# Example usage function
async def run_phase2_example():
    """Example of running Phase 2 EvoMerge."""

    # Create dummy Phase 1 models (replace with actual models)
    dummy_models = [
        nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1)),
        nn.Sequential(nn.Linear(10, 5), nn.Tanh(), nn.Linear(5, 1)),
        nn.Sequential(nn.Linear(10, 5), nn.Sigmoid(), nn.Linear(5, 1))
    ]

    # Configure Phase 2
    config = Phase2Config(
        max_generations=10,  # Reduced for testing
        enable_3d_visualization=True,
        output_directory="./test_phase2_results"
    )

    # Create orchestrator
    orchestrator = create_phase2_orchestrator(config)

    # Register progress callback
    def progress_callback(progress: Phase2Progress):
        print(f"Generation {progress.current_generation}/{progress.total_generations}: "
              f"Best fitness: {progress.best_fitness:.4f}, Avg: {progress.avg_fitness:.4f}")

    orchestrator.register_progress_callback(progress_callback)

    # Execute Phase 2
    result = await orchestrator.execute_phase2(dummy_models)

    if result.success:
        print(f"Phase 2 completed successfully!")
        print(f"Best fitness: {result.best_fitness:.4f}")
        print(f"Total generations: {result.total_generations}")
        print(f"Processing time: {result.total_processing_time:.2f}s")
    else:
        print(f"Phase 2 failed: {result.error_message}")

    return result


if __name__ == "__main__":
    # Run example
    import asyncio
    asyncio.run(run_phase2_example())