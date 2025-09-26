"""
Phase Validators for Agent Forge 8-Phase Pipeline

Comprehensive validation system for phase outputs, data compatibility,
resource requirements, and quality gates.
"""

import json
import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

from .phase_orchestrator import PhaseResult, PhaseMetrics, PhaseType


@dataclass
class ValidationResult:
    """Result of validation check."""
    valid: bool
    validator_name: str
    phase_name: str
    validation_type: str
    score: float = 0.0  # 0.0 to 1.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QualityGate:
    """Quality gate specification."""
    name: str
    description: str
    min_score: float
    max_errors: int = 0
    max_warnings: int = 5
    required: bool = True
    validator_type: str = "generic"


class BaseValidator(ABC):
    """Base class for all phase validators."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    async def validate(self, phase_result: PhaseResult, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate a phase result."""
        pass

    def _create_result(self, valid: bool, phase_name: str, validation_type: str,
                      score: float = 0.0, errors: List[str] = None,
                      warnings: List[str] = None, details: Dict[str, Any] = None) -> ValidationResult:
        """Helper to create validation result."""
        return ValidationResult(
            valid=valid,
            validator_name=self.name,
            phase_name=phase_name,
            validation_type=validation_type,
            score=score,
            errors=errors or [],
            warnings=warnings or [],
            details=details or {}
        )


class ModelIntegrityValidator(BaseValidator):
    """Validates model integrity and structure."""

    def __init__(self):
        super().__init__("ModelIntegrityValidator")

    async def validate(self, phase_result: PhaseResult, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate model integrity."""
        errors = []
        warnings = []
        details = {}
        score = 0.0

        try:
            model = phase_result.model
            if model is None:
                errors.append("No model found in phase result")
                return self._create_result(False, phase_result.phase_name, "model_integrity",
                                         score=0.0, errors=errors)

            # Check model structure
            if not isinstance(model, nn.Module):
                errors.append(f"Model is not a PyTorch nn.Module: {type(model)}")

            # Check parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            details["total_parameters"] = total_params
            details["trainable_parameters"] = trainable_params
            details["parameter_ratio"] = trainable_params / total_params if total_params > 0 else 0

            if total_params == 0:
                errors.append("Model has no parameters")
            elif total_params < 1000:
                warnings.append(f"Model has very few parameters: {total_params}")

            # Check model state
            model_state = model.state_dict()
            details["state_dict_keys"] = len(model_state)

            # Check for NaN or infinite values
            nan_params = 0
            inf_params = 0
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    nan_params += 1
                if torch.isinf(param).any():
                    inf_params += 1

            if nan_params > 0:
                errors.append(f"Found NaN values in {nan_params} parameters")
            if inf_params > 0:
                errors.append(f"Found infinite values in {inf_params} parameters")

            details["nan_parameters"] = nan_params
            details["inf_parameters"] = inf_params

            # Check model configuration
            if hasattr(model, 'config'):
                config = model.config
                details["has_config"] = True
                details["config_attributes"] = len(dir(config)) if config else 0
            else:
                warnings.append("Model has no config attribute")
                details["has_config"] = False

            # Calculate score
            score = 1.0
            if errors:
                score = 0.0
            elif warnings:
                score = 0.8
            elif nan_params > 0 or inf_params > 0:
                score = 0.3
            elif total_params < 1000:
                score = 0.6

            return self._create_result(
                valid=len(errors) == 0,
                phase_name=phase_result.phase_name,
                validation_type="model_integrity",
                score=score,
                errors=errors,
                warnings=warnings,
                details=details
            )

        except Exception as e:
            errors.append(f"Validation failed with exception: {str(e)}")
            return self._create_result(False, phase_result.phase_name, "model_integrity",
                                     score=0.0, errors=errors)


class DataFormatValidator(BaseValidator):
    """Validates data format compatibility between phases."""

    def __init__(self):
        super().__init__("DataFormatValidator")

    async def validate(self, phase_result: PhaseResult, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate data format compatibility."""
        errors = []
        warnings = []
        details = {}

        try:
            # Check output data structure
            output_data = phase_result.output_data
            details["output_data_keys"] = list(output_data.keys()) if output_data else []

            # Phase-specific format validation
            if phase_result.phase_type == PhaseType.CREATION:
                # Cognate phase should produce model and metadata
                required_keys = ["model_path", "architecture_info"]
                missing_keys = [key for key in required_keys if key not in output_data]
                if missing_keys:
                    errors.extend([f"Missing required output: {key}" for key in missing_keys])

            elif phase_result.phase_type == PhaseType.EVOLUTION:
                # EvoMerge should produce fitness scores and evolved model
                required_keys = ["fitness_score", "generation_info"]
                missing_keys = [key for key in required_keys if key not in output_data]
                if missing_keys:
                    warnings.extend([f"Missing recommended output: {key}" for key in missing_keys])

                if "fitness_score" in output_data:
                    score = output_data["fitness_score"]
                    if not isinstance(score, (int, float)) or score < 0:
                        errors.append(f"Invalid fitness score: {score}")

            elif phase_result.phase_type == PhaseType.REASONING:
                # QuietSTaR should produce reasoning metrics
                required_keys = ["reasoning_capability", "thought_generation_stats"]
                missing_keys = [key for key in required_keys if key not in output_data]
                if missing_keys:
                    warnings.extend([f"Missing recommended output: {key}" for key in missing_keys])

            elif phase_result.phase_type == PhaseType.COMPRESSION:
                # BitNet should produce compression metrics
                required_keys = ["compression_ratio", "quantization_info"]
                missing_keys = [key for key in required_keys if key not in output_data]
                if missing_keys:
                    warnings.extend([f"Missing recommended output: {key}" for key in missing_keys])

                if "compression_ratio" in output_data:
                    ratio = output_data["compression_ratio"]
                    if not isinstance(ratio, (int, float)) or ratio <= 0:
                        errors.append(f"Invalid compression ratio: {ratio}")

            elif phase_result.phase_type == PhaseType.TRAINING:
                # Training should produce loss metrics
                required_keys = ["training_loss", "validation_metrics"]
                missing_keys = [key for key in required_keys if key not in output_data]
                if missing_keys:
                    warnings.extend([f"Missing recommended output: {key}" for key in missing_keys])

            elif phase_result.phase_type == PhaseType.SPECIALIZATION:
                # Tool baking should produce specialization metrics
                required_keys = ["specialization_score", "baked_capabilities"]
                missing_keys = [key for key in required_keys if key not in output_data]
                if missing_keys:
                    warnings.extend([f"Missing recommended output: {key}" for key in missing_keys])

            elif phase_result.phase_type == PhaseType.ARCHITECTURE:
                # ADAS should produce architecture metrics
                required_keys = ["architecture_score", "optimization_info"]
                missing_keys = [key for key in required_keys if key not in output_data]
                if missing_keys:
                    warnings.extend([f"Missing recommended output: {key}" for key in missing_keys])

            elif phase_result.phase_type == PhaseType.FINALIZATION:
                # Final compression should produce final metrics
                required_keys = ["final_size", "compression_summary"]
                missing_keys = [key for key in required_keys if key not in output_data]
                if missing_keys:
                    warnings.extend([f"Missing recommended output: {key}" for key in missing_keys])

            # Check model path validity
            if phase_result.model_path:
                if not os.path.exists(phase_result.model_path):
                    errors.append(f"Model path does not exist: {phase_result.model_path}")
                else:
                    details["model_file_size"] = os.path.getsize(phase_result.model_path)

            # Calculate score
            score = 1.0
            if errors:
                score = 0.0
            elif warnings:
                score = 0.7

            return self._create_result(
                valid=len(errors) == 0,
                phase_name=phase_result.phase_name,
                validation_type="data_format",
                score=score,
                errors=errors,
                warnings=warnings,
                details=details
            )

        except Exception as e:
            errors.append(f"Data format validation failed: {str(e)}")
            return self._create_result(False, phase_result.phase_name, "data_format",
                                     score=0.0, errors=errors)


class PerformanceValidator(BaseValidator):
    """Validates performance metrics and benchmarks."""

    def __init__(self):
        super().__init__("PerformanceValidator")

    async def validate(self, phase_result: PhaseResult, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate performance metrics."""
        errors = []
        warnings = []
        details = {}

        try:
            metrics = phase_result.metrics
            if not metrics:
                errors.append("No metrics found in phase result")
                return self._create_result(False, phase_result.phase_name, "performance",
                                         score=0.0, errors=errors)

            # Check duration
            duration = metrics.duration_seconds
            details["duration_seconds"] = duration

            if duration <= 0:
                errors.append(f"Invalid duration: {duration}")
            elif duration > 86400:  # 24 hours
                warnings.append(f"Very long execution time: {duration/3600:.1f} hours")

            # Check memory usage
            memory_mb = metrics.memory_usage_mb
            details["memory_usage_mb"] = memory_mb

            if memory_mb > 32000:  # 32GB
                warnings.append(f"High memory usage: {memory_mb:.1f} MB")

            # Check GPU memory
            gpu_memory_mb = metrics.gpu_memory_mb
            details["gpu_memory_mb"] = gpu_memory_mb

            if gpu_memory_mb > 24000:  # 24GB
                warnings.append(f"High GPU memory usage: {gpu_memory_mb:.1f} MB")

            # Check error count
            error_count = metrics.error_count
            details["error_count"] = error_count

            if error_count > 0:
                warnings.append(f"Phase had {error_count} errors")

            # Phase-specific performance checks
            if phase_result.phase_type == PhaseType.EVOLUTION:
                # EvoMerge should have reasonable convergence
                if "fitness_score" in phase_result.output_data:
                    fitness = phase_result.output_data["fitness_score"]
                    if fitness < 0.5:
                        warnings.append(f"Low fitness score: {fitness}")

            elif phase_result.phase_type == PhaseType.TRAINING:
                # Training should show loss improvement
                if "training_loss" in phase_result.output_data:
                    loss = phase_result.output_data["training_loss"]
                    if isinstance(loss, list) and len(loss) > 1:
                        initial_loss = loss[0]
                        final_loss = loss[-1]
                        if final_loss >= initial_loss:
                            warnings.append("No training loss improvement observed")

            # Calculate performance score
            score = 1.0

            # Penalize for errors
            if error_count > 0:
                score -= 0.2

            # Penalize for excessive duration (relative to expected)
            expected_durations = {
                PhaseType.CREATION: 300,      # 5 minutes
                PhaseType.EVOLUTION: 3600,    # 1 hour
                PhaseType.REASONING: 1800,    # 30 minutes
                PhaseType.COMPRESSION: 600,   # 10 minutes
                PhaseType.TRAINING: 7200,     # 2 hours
                PhaseType.SPECIALIZATION: 1800,  # 30 minutes
                PhaseType.ARCHITECTURE: 3600,    # 1 hour
                PhaseType.FINALIZATION: 600      # 10 minutes
            }

            expected_duration = expected_durations.get(phase_result.phase_type, 1800)
            if duration > expected_duration * 2:
                score -= 0.3

            # Penalize for excessive memory usage
            if memory_mb > 16000:  # 16GB
                score -= 0.2

            score = max(0.0, score)

            return self._create_result(
                valid=len(errors) == 0,
                phase_name=phase_result.phase_name,
                validation_type="performance",
                score=score,
                errors=errors,
                warnings=warnings,
                details=details
            )

        except Exception as e:
            errors.append(f"Performance validation failed: {str(e)}")
            return self._create_result(False, phase_result.phase_name, "performance",
                                     score=0.0, errors=errors)


class ResourceValidator(BaseValidator):
    """Validates resource requirements and constraints."""

    def __init__(self):
        super().__init__("ResourceValidator")

    async def validate(self, phase_result: PhaseResult, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate resource requirements."""
        errors = []
        warnings = []
        details = {}

        try:
            # Check model size if available
            if phase_result.model:
                model_size = self._calculate_model_size(phase_result.model)
                details["model_size_mb"] = model_size

                # Check if model size is reasonable
                max_sizes = {
                    PhaseType.CREATION: 500,      # 500MB
                    PhaseType.EVOLUTION: 1000,    # 1GB
                    PhaseType.REASONING: 1500,    # 1.5GB
                    PhaseType.COMPRESSION: 200,   # 200MB (compressed)
                    PhaseType.TRAINING: 2000,     # 2GB
                    PhaseType.SPECIALIZATION: 2500,  # 2.5GB
                    PhaseType.ARCHITECTURE: 3000,    # 3GB
                    PhaseType.FINALIZATION: 100      # 100MB (final compressed)
                }

                max_size = max_sizes.get(phase_result.phase_type, 1000)
                if model_size > max_size:
                    warnings.append(f"Model size {model_size:.1f}MB exceeds expected {max_size}MB")

            # Check disk usage if model was saved
            if phase_result.model_path and os.path.exists(phase_result.model_path):
                file_size = os.path.getsize(phase_result.model_path) / (1024 * 1024)  # MB
                details["saved_model_size_mb"] = file_size

            # Check artifacts size
            artifacts_size = 0
            if phase_result.artifacts:
                for key, value in phase_result.artifacts.items():
                    if isinstance(value, str) and os.path.exists(value):
                        artifacts_size += os.path.getsize(value) / (1024 * 1024)

            details["artifacts_size_mb"] = artifacts_size

            if artifacts_size > 1000:  # 1GB
                warnings.append(f"Large artifacts size: {artifacts_size:.1f}MB")

            # Check resource efficiency
            if phase_result.metrics:
                duration = phase_result.metrics.duration_seconds
                memory_usage = phase_result.metrics.memory_usage_mb

                if duration > 0 and memory_usage > 0:
                    # Simple efficiency metric (inverse of memory * time)
                    efficiency = 1000000 / (duration * memory_usage)
                    details["resource_efficiency"] = efficiency

                    if efficiency < 1.0:
                        warnings.append("Low resource efficiency")

            # Calculate score
            score = 1.0
            if warnings:
                score = 0.8

            return self._create_result(
                valid=len(errors) == 0,
                phase_name=phase_result.phase_name,
                validation_type="resource",
                score=score,
                errors=errors,
                warnings=warnings,
                details=details
            )

        except Exception as e:
            errors.append(f"Resource validation failed: {str(e)}")
            return self._create_result(False, phase_result.phase_name, "resource",
                                     score=0.0, errors=errors)

    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        total_size = param_size + buffer_size
        return total_size / (1024 * 1024)  # Convert to MB


class QualityGateValidator(BaseValidator):
    """Validates quality gates and requirements."""

    def __init__(self, quality_gates: List[QualityGate]):
        super().__init__("QualityGateValidator")
        self.quality_gates = quality_gates

    async def validate(self, phase_result: PhaseResult, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate quality gates."""
        errors = []
        warnings = []
        details = {}

        try:
            phase_gates = [gate for gate in self.quality_gates
                          if gate.validator_type == "all" or
                          gate.validator_type == phase_result.phase_type.value]

            gate_results = {}

            for gate in phase_gates:
                gate_passed = True
                gate_details = {}

                # Check score requirements
                if hasattr(phase_result, 'score') and phase_result.score is not None:
                    if phase_result.score < gate.min_score:
                        gate_passed = False
                        errors.append(f"Quality gate '{gate.name}' failed: score {phase_result.score} < {gate.min_score}")

                # Check error count
                if phase_result.metrics and phase_result.metrics.error_count > gate.max_errors:
                    gate_passed = False
                    errors.append(f"Quality gate '{gate.name}' failed: {phase_result.metrics.error_count} errors > {gate.max_errors}")

                # Check success requirement
                if gate.required and not phase_result.success:
                    gate_passed = False
                    errors.append(f"Required quality gate '{gate.name}' failed: phase was not successful")

                gate_results[gate.name] = {
                    "passed": gate_passed,
                    "required": gate.required,
                    "details": gate_details
                }

            details["quality_gates"] = gate_results

            # Calculate overall score
            total_gates = len(phase_gates)
            passed_gates = sum(1 for result in gate_results.values() if result["passed"])

            score = passed_gates / total_gates if total_gates > 0 else 1.0

            # Check if any required gates failed
            failed_required = any(
                not result["passed"] and result["required"]
                for result in gate_results.values()
            )

            return self._create_result(
                valid=not failed_required and len(errors) == 0,
                phase_name=phase_result.phase_name,
                validation_type="quality_gates",
                score=score,
                errors=errors,
                warnings=warnings,
                details=details
            )

        except Exception as e:
            errors.append(f"Quality gate validation failed: {str(e)}")
            return self._create_result(False, phase_result.phase_name, "quality_gates",
                                     score=0.0, errors=errors)


class PhaseValidationSuite:
    """Complete validation suite for all phases."""

    def __init__(self, output_dir: str = "./validation_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Initialize validators
        self.validators = {
            "model_integrity": ModelIntegrityValidator(),
            "data_format": DataFormatValidator(),
            "performance": PerformanceValidator(),
            "resource": ResourceValidator()
        }

        # Default quality gates
        self.quality_gates = self._create_default_quality_gates()
        self.validators["quality_gates"] = QualityGateValidator(self.quality_gates)

        # Validation history
        self.validation_history: List[Dict[str, ValidationResult]] = []

    def _create_default_quality_gates(self) -> List[QualityGate]:
        """Create default quality gates for all phases."""
        return [
            QualityGate(
                name="BasicSuccess",
                description="Phase must complete successfully",
                min_score=0.0,
                max_errors=0,
                required=True,
                validator_type="all"
            ),
            QualityGate(
                name="ModelIntegrity",
                description="Model must be valid and well-formed",
                min_score=0.8,
                max_errors=0,
                required=True,
                validator_type="all"
            ),
            QualityGate(
                name="Performance",
                description="Performance must be acceptable",
                min_score=0.6,
                max_errors=5,
                required=False,
                validator_type="all"
            ),
            QualityGate(
                name="EvolutionQuality",
                description="Evolution must show improvement",
                min_score=0.7,
                max_errors=0,
                required=True,
                validator_type="evolution"
            ),
            QualityGate(
                name="CompressionEfficiency",
                description="Compression must be effective",
                min_score=0.8,
                max_errors=0,
                required=True,
                validator_type="compression"
            )
        ]

    async def validate_phase(self, phase_result: PhaseResult,
                           context: Dict[str, Any] = None) -> Dict[str, ValidationResult]:
        """Validate a phase result using all validators."""
        self.logger.info(f"Validating phase: {phase_result.phase_name}")

        validation_results = {}

        for validator_name, validator in self.validators.items():
            try:
                result = await validator.validate(phase_result, context)
                validation_results[validator_name] = result

                if not result.valid:
                    self.logger.warning(f"Validation failed for {validator_name}: {result.errors}")
                elif result.warnings:
                    self.logger.info(f"Validation warnings for {validator_name}: {result.warnings}")

            except Exception as e:
                self.logger.error(f"Validator {validator_name} failed: {str(e)}")
                validation_results[validator_name] = ValidationResult(
                    valid=False,
                    validator_name=validator_name,
                    phase_name=phase_result.phase_name,
                    validation_type="error",
                    errors=[f"Validator failed: {str(e)}"]
                )

        # Store validation history
        self.validation_history.append(validation_results)

        # Save validation report
        await self._save_validation_report(phase_result.phase_name, validation_results)

        return validation_results

    async def validate_pipeline(self, phase_results: List[PhaseResult]) -> Dict[str, Dict[str, ValidationResult]]:
        """Validate complete pipeline results."""
        self.logger.info("Validating complete pipeline")

        pipeline_validation = {}

        for phase_result in phase_results:
            phase_validation = await self.validate_phase(phase_result)
            pipeline_validation[phase_result.phase_name] = phase_validation

        # Generate pipeline validation summary
        await self._generate_pipeline_validation_summary(pipeline_validation)

        return pipeline_validation

    async def _save_validation_report(self, phase_name: str, validation_results: Dict[str, ValidationResult]):
        """Save validation report for a phase."""
        report = {
            "phase_name": phase_name,
            "timestamp": datetime.now().isoformat(),
            "validation_results": {
                name: {
                    "valid": result.valid,
                    "score": result.score,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "details": result.details
                }
                for name, result in validation_results.items()
            }
        }

        report_file = self.output_dir / f"{phase_name}_validation.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

    async def _generate_pipeline_validation_summary(self, pipeline_validation: Dict[str, Dict[str, ValidationResult]]):
        """Generate comprehensive pipeline validation summary."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_phases": len(pipeline_validation),
            "phase_summaries": {},
            "overall_statistics": {
                "total_validations": 0,
                "passed_validations": 0,
                "failed_validations": 0,
                "total_errors": 0,
                "total_warnings": 0
            }
        }

        total_validations = 0
        passed_validations = 0
        failed_validations = 0
        total_errors = 0
        total_warnings = 0

        for phase_name, validation_results in pipeline_validation.items():
            phase_summary = {
                "total_validators": len(validation_results),
                "passed_validators": sum(1 for r in validation_results.values() if r.valid),
                "failed_validators": sum(1 for r in validation_results.values() if not r.valid),
                "total_errors": sum(len(r.errors) for r in validation_results.values()),
                "total_warnings": sum(len(r.warnings) for r in validation_results.values()),
                "average_score": sum(r.score for r in validation_results.values()) / len(validation_results)
            }

            summary["phase_summaries"][phase_name] = phase_summary

            total_validations += phase_summary["total_validators"]
            passed_validations += phase_summary["passed_validators"]
            failed_validations += phase_summary["failed_validators"]
            total_errors += phase_summary["total_errors"]
            total_warnings += phase_summary["total_warnings"]

        summary["overall_statistics"] = {
            "total_validations": total_validations,
            "passed_validations": passed_validations,
            "failed_validations": failed_validations,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "success_rate": passed_validations / total_validations if total_validations > 0 else 0
        }

        # Save summary
        summary_file = self.output_dir / "pipeline_validation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Pipeline validation summary saved to: {summary_file}")
        self.logger.info(f"Overall success rate: {summary['overall_statistics']['success_rate']:.2%}")


# Export main classes
__all__ = [
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