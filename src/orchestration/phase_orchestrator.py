"""
Phase Orchestrator for Agent Forge 8-Phase Pipeline

Manages phase state, transitions, dependencies, and data flow between all phases.
Provides checkpoint management, error recovery, and parallel execution capabilities.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class PhaseState(Enum):
    """Phase execution states."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class PhaseType(Enum):
    """Phase types for dependency management."""
    CREATION = "creation"  # Phase 1: Cognate
    EVOLUTION = "evolution"  # Phase 2: EvoMerge
    REASONING = "reasoning"  # Phase 3: QuietSTaR
    COMPRESSION = "compression"  # Phase 4: BitNet
    TRAINING = "training"  # Phase 5: Forge Training
    SPECIALIZATION = "specialization"  # Phase 6: Tool/Persona Baking
    ARCHITECTURE = "architecture"  # Phase 7: ADAS
    FINALIZATION = "finalization"  # Phase 8: Final Compression


@dataclass
class PhaseMetrics:
    """Metrics for phase execution tracking."""
    phase_name: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    model_parameters: int = 0
    model_size_mb: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    loss_metrics: Dict[str, float] = field(default_factory=dict)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoint_size_mb: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0


@dataclass
class PhaseResult:
    """Result of phase execution."""
    success: bool
    phase_name: str
    phase_type: PhaseType
    model: Optional[nn.Module] = None
    model_path: Optional[str] = None
    output_data: Dict[str, Any] = field(default_factory=dict)
    metrics: PhaseMetrics = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    dependencies_met: List[str] = field(default_factory=list)
    next_phase_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseDependency:
    """Phase dependency specification."""
    source_phase: str
    target_phase: str
    dependency_type: str  # "model", "data", "config", "validation"
    required_outputs: List[str]
    optional_outputs: List[str] = field(default_factory=list)
    validation_function: Optional[callable] = None


@dataclass
class PhaseCheckpoint:
    """Phase checkpoint for recovery."""
    phase_name: str
    checkpoint_id: str
    timestamp: datetime
    model_state: Dict[str, Any]
    phase_state: PhaseState
    metrics: PhaseMetrics
    recovery_data: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[str] = None


class PhaseOrchestrator:
    """
    Advanced orchestrator for 8-phase Agent Forge pipeline.

    Features:
    - State management and transitions
    - Dependency resolution
    - Parallel execution where possible
    - Checkpoint management and recovery
    - Progress monitoring and reporting
    - Error handling and rollback
    """

    def __init__(self, output_dir: str = "./orchestration_output"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Phase management
        self.phases: Dict[str, PhaseState] = {}
        self.phase_results: Dict[str, PhaseResult] = {}
        self.phase_metrics: Dict[str, PhaseMetrics] = {}
        self.phase_dependencies: List[PhaseDependency] = []
        self.checkpoints: Dict[str, PhaseCheckpoint] = {}

        # Execution state
        self.current_phase: Optional[str] = None
        self.execution_started: bool = False
        self.execution_completed: bool = False
        self.execution_failed: bool = False

        # Progress tracking
        self.progress_callbacks: List[callable] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Initialize phase dependencies
        self._initialize_phase_dependencies()

    def _initialize_phase_dependencies(self):
        """Initialize the standard 8-phase dependencies."""
        dependencies = [
            # Phase 1 -> 2: Cognate creates base model for EvoMerge
            PhaseDependency(
                source_phase="CognatePhase",
                target_phase="EvoMergePhase",
                dependency_type="model",
                required_outputs=["model", "model_path"],
                validation_function=self._validate_model_output
            ),

            # Phase 2 -> 3: EvoMerge evolved model for QuietSTaR
            PhaseDependency(
                source_phase="EvoMergePhase",
                target_phase="QuietSTaRPhase",
                dependency_type="model",
                required_outputs=["best_model", "model_path"],
                validation_function=self._validate_evolved_model
            ),

            # Phase 3 -> 4: QuietSTaR reasoning-enhanced model for BitNet
            PhaseDependency(
                source_phase="QuietSTaRPhase",
                target_phase="BitNetCompressionPhase",
                dependency_type="model",
                required_outputs=["enhanced_model", "model_path"],
                validation_function=self._validate_reasoning_model
            ),

            # Phase 4 -> 5: BitNet compressed model for training
            PhaseDependency(
                source_phase="BitNetCompressionPhase",
                target_phase="ForgeTrainingPhase",
                dependency_type="model",
                required_outputs=["compressed_model", "model_path"],
                validation_function=self._validate_compressed_model
            ),

            # Phase 5 -> 6: Trained model for tool/persona baking
            PhaseDependency(
                source_phase="ForgeTrainingPhase",
                target_phase="ToolPersonaBakingPhase",
                dependency_type="model",
                required_outputs=["trained_model", "model_path"],
                validation_function=self._validate_trained_model
            ),

            # Phase 6 -> 7: Specialized model for ADAS
            PhaseDependency(
                source_phase="ToolPersonaBakingPhase",
                target_phase="ADASPhase",
                dependency_type="model",
                required_outputs=["specialized_model", "model_path"],
                validation_function=self._validate_specialized_model
            ),

            # Phase 7 -> 8: ADAS-optimized model for final compression
            PhaseDependency(
                source_phase="ADASPhase",
                target_phase="FinalCompressionPhase",
                dependency_type="model",
                required_outputs=["optimized_model", "model_path"],
                validation_function=self._validate_optimized_model
            )
        ]

        self.phase_dependencies = dependencies
        self.logger.info(f"Initialized {len(dependencies)} phase dependencies")

    def register_phase(self, phase_name: str, phase_type: PhaseType = None):
        """Register a phase for orchestration."""
        self.phases[phase_name] = PhaseState.PENDING
        self.phase_metrics[phase_name] = PhaseMetrics(phase_name=phase_name)
        self.logger.info(f"Registered phase: {phase_name}")

    def add_progress_callback(self, callback: callable):
        """Add a callback for progress updates."""
        self.progress_callbacks.append(callback)

    async def execute_phase_pipeline(self, phases: List[Tuple[str, Any]],
                                   initial_model: Optional[nn.Module] = None,
                                   resume_from: Optional[str] = None) -> PhaseResult:
        """
        Execute the complete 8-phase pipeline with dependency management.

        Args:
            phases: List of (phase_name, phase_controller) tuples
            initial_model: Starting model (None for Cognate to create)
            resume_from: Phase name to resume from

        Returns:
            Final phase result
        """
        self.start_time = datetime.now()
        self.execution_started = True

        try:
            # Register all phases
            for phase_name, _ in phases:
                self.register_phase(phase_name)

            # Determine execution sequence
            execution_sequence = self._determine_execution_sequence(phases, resume_from)

            self.logger.info(f"Starting pipeline execution with {len(execution_sequence)} phases")
            self._notify_progress("pipeline_started", {"total_phases": len(execution_sequence)})

            # Execute phases in sequence
            current_model = initial_model

            for i, (phase_name, phase_controller) in enumerate(execution_sequence):
                self.current_phase = phase_name

                # Check dependencies
                if not self._check_phase_dependencies(phase_name):
                    error_msg = f"Dependencies not met for phase {phase_name}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)

                # Update phase state
                self.phases[phase_name] = PhaseState.READY
                self._notify_progress("phase_ready", {"phase": phase_name, "index": i})

                # Execute phase
                phase_result = await self._execute_single_phase(
                    phase_name, phase_controller, current_model
                )

                # Store result
                self.phase_results[phase_name] = phase_result

                # Handle phase result
                if not phase_result.success:
                    self.execution_failed = True
                    self.phases[phase_name] = PhaseState.FAILED
                    error_msg = f"Phase {phase_name} failed: {phase_result.error}"
                    self.logger.error(error_msg)

                    # Attempt recovery if possible
                    if await self._attempt_phase_recovery(phase_name, phase_controller):
                        phase_result = self.phase_results[phase_name]
                        if phase_result.success:
                            self.logger.info(f"Phase {phase_name} recovered successfully")
                        else:
                            raise RuntimeError(error_msg)
                    else:
                        raise RuntimeError(error_msg)

                # Update current model for next phase
                if phase_result.model is not None:
                    current_model = phase_result.model

                # Create checkpoint
                await self._create_checkpoint(phase_name, phase_result)

                # Update state
                self.phases[phase_name] = PhaseState.COMPLETED
                self._notify_progress("phase_completed", {
                    "phase": phase_name,
                    "index": i,
                    "success": phase_result.success
                })

            # Pipeline completed successfully
            self.execution_completed = True
            self.end_time = datetime.now()

            # Generate final result
            final_result = self._generate_final_result(execution_sequence[-1][0])

            self.logger.info("Pipeline execution completed successfully")
            self._notify_progress("pipeline_completed", {
                "total_duration": (self.end_time - self.start_time).total_seconds(),
                "phases_completed": len([p for p in self.phases.values() if p == PhaseState.COMPLETED])
            })

            return final_result

        except Exception as e:
            self.execution_failed = True
            self.end_time = datetime.now()
            error_msg = f"Pipeline execution failed: {str(e)}"
            self.logger.error(error_msg)

            # Create failure result
            failure_result = PhaseResult(
                success=False,
                phase_name="Pipeline",
                phase_type=PhaseType.FINALIZATION,
                error=error_msg,
                metrics=self._aggregate_pipeline_metrics()
            )

            self._notify_progress("pipeline_failed", {"error": error_msg})
            return failure_result

    async def _execute_single_phase(self, phase_name: str, phase_controller: Any,
                                   input_model: Optional[nn.Module]) -> PhaseResult:
        """Execute a single phase with full monitoring."""
        self.logger.info(f"Executing phase: {phase_name}")
        start_time = datetime.now()

        # Update phase state
        self.phases[phase_name] = PhaseState.RUNNING
        self.phase_metrics[phase_name].start_time = start_time

        try:
            # Prepare input data based on dependencies
            input_data = self._prepare_phase_input_data(phase_name, input_model)

            # Execute phase
            if hasattr(phase_controller, 'execute_async'):
                result = await phase_controller.execute_async(input_data)
            elif hasattr(phase_controller, 'execute'):
                result = phase_controller.execute(input_data)
            else:
                raise NotImplementedError(f"Phase {phase_name} has no execute method")

            # Update metrics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            self.phase_metrics[phase_name].end_time = end_time
            self.phase_metrics[phase_name].duration_seconds = duration

            # Create successful result
            if isinstance(result, PhaseResult):
                phase_result = result
            else:
                # Wrap result in PhaseResult
                phase_result = PhaseResult(
                    success=True,
                    phase_name=phase_name,
                    phase_type=self._get_phase_type(phase_name),
                    model=getattr(result, 'model', None),
                    output_data=result if isinstance(result, dict) else {"result": result},
                    metrics=self.phase_metrics[phase_name]
                )

            # Monitor resource usage
            await self._update_resource_metrics(phase_name)

            self.logger.info(f"Phase {phase_name} completed in {duration:.2f}s")
            return phase_result

        except Exception as e:
            # Handle phase failure
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            self.phase_metrics[phase_name].end_time = end_time
            self.phase_metrics[phase_name].duration_seconds = duration
            self.phase_metrics[phase_name].error_count += 1

            error_msg = f"Phase {phase_name} failed: {str(e)}"
            self.logger.error(error_msg)

            return PhaseResult(
                success=False,
                phase_name=phase_name,
                phase_type=self._get_phase_type(phase_name),
                error=error_msg,
                metrics=self.phase_metrics[phase_name]
            )

    def _determine_execution_sequence(self, phases: List[Tuple[str, Any]],
                                    resume_from: Optional[str] = None) -> List[Tuple[str, Any]]:
        """Determine the execution sequence based on dependencies."""
        if resume_from:
            # Find resume point and return remaining phases
            phase_names = [name for name, _ in phases]
            if resume_from in phase_names:
                resume_index = phase_names.index(resume_from)
                return phases[resume_index:]
            else:
                self.logger.warning(f"Resume point {resume_from} not found, starting from beginning")

        # For now, return phases in order (they should already be dependency-sorted)
        return phases

    def _check_phase_dependencies(self, phase_name: str) -> bool:
        """Check if all dependencies for a phase are satisfied."""
        for dependency in self.phase_dependencies:
            if dependency.target_phase == phase_name:
                source_phase = dependency.source_phase

                # Check if source phase completed successfully
                if source_phase not in self.phase_results:
                    self.logger.warning(f"Dependency {source_phase} not executed for {phase_name}")
                    return False

                source_result = self.phase_results[source_phase]
                if not source_result.success:
                    self.logger.warning(f"Dependency {source_phase} failed for {phase_name}")
                    return False

                # Check required outputs
                for required_output in dependency.required_outputs:
                    if required_output not in source_result.output_data and not hasattr(source_result, required_output):
                        self.logger.warning(f"Required output {required_output} missing from {source_phase}")
                        return False

                # Run validation function if provided
                if dependency.validation_function:
                    if not dependency.validation_function(source_result):
                        self.logger.warning(f"Validation failed for dependency {source_phase} -> {phase_name}")
                        return False

        return True

    def _prepare_phase_input_data(self, phase_name: str, input_model: Optional[nn.Module]) -> Dict[str, Any]:
        """Prepare input data for a phase based on dependencies."""
        input_data = {"model": input_model}

        # Collect data from dependency sources
        for dependency in self.phase_dependencies:
            if dependency.target_phase == phase_name:
                source_phase = dependency.source_phase
                if source_phase in self.phase_results:
                    source_result = self.phase_results[source_phase]

                    # Add required outputs
                    for output_name in dependency.required_outputs:
                        if output_name in source_result.output_data:
                            input_data[output_name] = source_result.output_data[output_name]
                        elif hasattr(source_result, output_name):
                            input_data[output_name] = getattr(source_result, output_name)

                    # Add optional outputs
                    for output_name in dependency.optional_outputs:
                        if output_name in source_result.output_data:
                            input_data[output_name] = source_result.output_data[output_name]
                        elif hasattr(source_result, output_name):
                            input_data[output_name] = getattr(source_result, output_name)

        return input_data

    async def _create_checkpoint(self, phase_name: str, phase_result: PhaseResult):
        """Create a checkpoint for phase recovery."""
        checkpoint_id = f"{phase_name}_{int(time.time())}"
        checkpoint_path = self.output_dir / "checkpoints" / f"{checkpoint_id}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = PhaseCheckpoint(
            phase_name=phase_name,
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(),
            model_state=phase_result.model.state_dict() if phase_result.model else {},
            phase_state=self.phases[phase_name],
            metrics=self.phase_metrics[phase_name],
            file_path=str(checkpoint_path)
        )

        # Save checkpoint
        torch.save({
            "checkpoint": checkpoint,
            "phase_result": phase_result,
            "phase_state": self.phases[phase_name]
        }, checkpoint_path)

        self.checkpoints[checkpoint_id] = checkpoint
        self.logger.info(f"Created checkpoint: {checkpoint_id}")

    async def _attempt_phase_recovery(self, phase_name: str, phase_controller: Any) -> bool:
        """Attempt to recover a failed phase."""
        self.logger.info(f"Attempting recovery for phase: {phase_name}")

        # Look for the most recent successful checkpoint
        recent_checkpoint = self._find_recent_checkpoint(phase_name)
        if not recent_checkpoint:
            self.logger.warning(f"No checkpoint found for recovery of {phase_name}")
            return False

        try:
            # Load checkpoint
            checkpoint_data = torch.load(recent_checkpoint.file_path)

            # Reset phase state
            self.phases[phase_name] = PhaseState.READY

            # Retry execution with checkpoint data
            recovery_result = await self._execute_single_phase(
                phase_name, phase_controller, None  # Model will be loaded from checkpoint
            )

            if recovery_result.success:
                self.phase_results[phase_name] = recovery_result
                return True

        except Exception as e:
            self.logger.error(f"Recovery failed for {phase_name}: {str(e)}")

        return False

    def _find_recent_checkpoint(self, phase_name: str) -> Optional[PhaseCheckpoint]:
        """Find the most recent checkpoint for a phase."""
        phase_checkpoints = [cp for cp in self.checkpoints.values() if cp.phase_name == phase_name]
        if not phase_checkpoints:
            return None

        return max(phase_checkpoints, key=lambda cp: cp.timestamp)

    async def _update_resource_metrics(self, phase_name: str):
        """Update resource usage metrics for a phase."""
        metrics = self.phase_metrics[phase_name]

        # Update memory usage
        import psutil
        process = psutil.Process()
        metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024

        # Update GPU memory if available
        if torch.cuda.is_available():
            metrics.gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024

    def _get_phase_type(self, phase_name: str) -> PhaseType:
        """Get the phase type for a phase name."""
        phase_type_mapping = {
            "CognatePhase": PhaseType.CREATION,
            "EvoMergePhase": PhaseType.EVOLUTION,
            "QuietSTaRPhase": PhaseType.REASONING,
            "BitNetCompressionPhase": PhaseType.COMPRESSION,
            "ForgeTrainingPhase": PhaseType.TRAINING,
            "ToolPersonaBakingPhase": PhaseType.SPECIALIZATION,
            "ADASPhase": PhaseType.ARCHITECTURE,
            "FinalCompressionPhase": PhaseType.FINALIZATION
        }
        return phase_type_mapping.get(phase_name, PhaseType.FINALIZATION)

    def _aggregate_pipeline_metrics(self) -> PhaseMetrics:
        """Aggregate metrics from all phases."""
        total_duration = sum(m.duration_seconds for m in self.phase_metrics.values())
        total_memory = max(m.memory_usage_mb for m in self.phase_metrics.values() if m.memory_usage_mb > 0) or 0
        total_errors = sum(m.error_count for m in self.phase_metrics.values())

        return PhaseMetrics(
            phase_name="Pipeline",
            duration_seconds=total_duration,
            memory_usage_mb=total_memory,
            error_count=total_errors,
            custom_metrics={
                "phases_completed": len([p for p in self.phases.values() if p == PhaseState.COMPLETED]),
                "phases_failed": len([p for p in self.phases.values() if p == PhaseState.FAILED]),
                "total_phases": len(self.phases)
            }
        )

    def _generate_final_result(self, final_phase_name: str) -> PhaseResult:
        """Generate the final pipeline result."""
        final_phase_result = self.phase_results[final_phase_name]

        return PhaseResult(
            success=True,
            phase_name="Pipeline",
            phase_type=PhaseType.FINALIZATION,
            model=final_phase_result.model,
            output_data={
                "final_model": final_phase_result.model,
                "phase_results": {name: result.output_data for name, result in self.phase_results.items()},
                "pipeline_metrics": self._aggregate_pipeline_metrics().__dict__
            },
            metrics=self._aggregate_pipeline_metrics(),
            artifacts={
                "checkpoints": list(self.checkpoints.keys()),
                "phase_sequence": list(self.phases.keys())
            }
        )

    def _notify_progress(self, event: str, data: Dict[str, Any]):
        """Notify progress callbacks of events."""
        for callback in self.progress_callbacks:
            try:
                callback(event, data)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {str(e)}")

    # Validation functions for dependencies
    def _validate_model_output(self, result: PhaseResult) -> bool:
        """Validate that a phase produces a valid model."""
        return result.model is not None or result.model_path is not None

    def _validate_evolved_model(self, result: PhaseResult) -> bool:
        """Validate that EvoMerge produces an evolved model."""
        return self._validate_model_output(result) and "fitness_score" in result.output_data

    def _validate_reasoning_model(self, result: PhaseResult) -> bool:
        """Validate that QuietSTaR produces a reasoning-enhanced model."""
        return self._validate_model_output(result) and "reasoning_capability" in result.output_data

    def _validate_compressed_model(self, result: PhaseResult) -> bool:
        """Validate that BitNet produces a compressed model."""
        return self._validate_model_output(result) and "compression_ratio" in result.output_data

    def _validate_trained_model(self, result: PhaseResult) -> bool:
        """Validate that training produces a trained model."""
        return self._validate_model_output(result) and "training_loss" in result.output_data

    def _validate_specialized_model(self, result: PhaseResult) -> bool:
        """Validate that baking produces a specialized model."""
        return self._validate_model_output(result) and "specialization_score" in result.output_data

    def _validate_optimized_model(self, result: PhaseResult) -> bool:
        """Validate that ADAS produces an optimized model."""
        return self._validate_model_output(result) and "architecture_score" in result.output_data


# Export main classes
__all__ = [
    "PhaseOrchestrator",
    "PhaseResult",
    "PhaseMetrics",
    "PhaseState",
    "PhaseType",
    "PhaseDependency",
    "PhaseCheckpoint"
]