"""
Orchestration module for Agent Forge 8-Phase Pipeline

This module provides comprehensive orchestration capabilities for the complete
Agent Forge pipeline including phase management, validation, and control.
"""

from .phase_orchestrator import (
    PhaseOrchestrator,
    PhaseResult,
    PhaseMetrics,
    PhaseState,
    PhaseType,
    PhaseDependency,
    PhaseCheckpoint
)

from .pipeline_controller import (
    PipelineController,
    PipelineConfig,
    ResourceConstraints
)

from .phase_validators import (
    PhaseValidationSuite,
    ValidationResult,
    QualityGate,
    BaseValidator,
    ModelIntegrityValidator,
    DataFormatValidator,
    PerformanceValidator,
    ResourceValidator,
    QualityGateValidator
)

__version__ = "1.0.0"

__all__ = [
    # Phase Orchestrator
    "PhaseOrchestrator",
    "PhaseResult",
    "PhaseMetrics",
    "PhaseState",
    "PhaseType",
    "PhaseDependency",
    "PhaseCheckpoint",

    # Pipeline Controller
    "PipelineController",
    "PipelineConfig",
    "ResourceConstraints",

    # Validators
    "PhaseValidationSuite",
    "ValidationResult",
    "QualityGate",
    "BaseValidator",
    "ModelIntegrityValidator",
    "DataFormatValidator",
    "PerformanceValidator",
    "ResourceValidator",
    "QualityGateValidator"
]