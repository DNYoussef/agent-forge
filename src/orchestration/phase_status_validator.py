"""
Phase Status Validator and Dependency Mapper

Validates the status of all 8 phases and creates comprehensive dependency mapping
for the Agent Forge pipeline.
"""

import json
import logging
import os
import sys
import importlib.util
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent.parent))


@dataclass
class PhaseInfo:
    """Information about a phase implementation."""
    name: str
    file_path: str
    exists: bool = False
    importable: bool = False
    has_execute_method: bool = False
    has_config_class: bool = False
    dependencies: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    status: str = "UNKNOWN"  # OPERATIONAL, NEEDS_FIXES, MISSING, BROKEN
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DependencyMapping:
    """Dependency mapping between phases."""
    source_phase: str
    target_phase: str
    dependency_type: str  # "model", "data", "config", "validation"
    required_outputs: List[str]
    optional_outputs: List[str] = field(default_factory=list)
    validation_rules: List[str] = field(default_factory=list)


class PhaseStatusValidator:
    """Validates status of all 8 phases and creates dependency mapping."""

    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.phases_dir = self.base_dir / "phases"
        self.logger = logging.getLogger(__name__)

        # Define the 8 phases
        self.phase_definitions = {
            "CognatePhase": {
                "file": "cognate.py",
                "class": "CognatePhase",
                "config_class": "CognateConfig",
                "description": "Model creation and initialization",
                "expected_outputs": ["model", "model_path", "architecture_info"]
            },
            "EvoMergePhase": {
                "file": "evomerge.py",
                "class": "EvoMergePhase",
                "config_class": "EvoMergeConfig",
                "description": "Evolutionary model optimization",
                "expected_outputs": ["best_model", "model_path", "fitness_score", "generation_info"]
            },
            "QuietSTaRPhase": {
                "file": "quietstar.py",
                "class": "QuietSTaRPhase",
                "config_class": "QuietSTaRConfig",
                "description": "Reasoning enhancement baking",
                "expected_outputs": ["enhanced_model", "model_path", "reasoning_capability", "thought_generation_stats"]
            },
            "BitNetCompressionPhase": {
                "file": "bitnet_compression.py",
                "class": "BitNetCompressionPhase",
                "config_class": "BitNetCompressionConfig",
                "description": "Initial compression with BitNet 1.58",
                "expected_outputs": ["compressed_model", "model_path", "compression_ratio", "quantization_info"]
            },
            "ForgeTrainingPhase": {
                "file": "forge_training.py",
                "class": "ForgeTrainingPhase",
                "config_class": "ForgeTrainingConfig",
                "description": "Main training loop with Grokfast",
                "expected_outputs": ["trained_model", "model_path", "training_loss", "validation_metrics"]
            },
            "ToolPersonaBakingPhase": {
                "file": "tool_persona_baking.py",
                "class": "ToolPersonaBakingPhase",
                "config_class": "ToolPersonaBakingConfig",
                "description": "Identity and capability baking",
                "expected_outputs": ["specialized_model", "model_path", "specialization_score", "baked_capabilities"]
            },
            "ADASPhase": {
                "file": "adas.py",
                "class": "ADASPhase",
                "config_class": "ADASConfig",
                "description": "Architecture search with vector composition",
                "expected_outputs": ["optimized_model", "model_path", "architecture_score", "optimization_info"]
            },
            "FinalCompressionPhase": {
                "file": "final_compression.py",
                "class": "FinalCompressionPhase",
                "config_class": "FinalCompressionConfig",
                "description": "SeedLM + VPTQ + Hypercompression",
                "expected_outputs": ["final_model", "model_path", "final_size", "compression_summary"]
            }
        }

        # Initialize phase info
        self.phase_info: Dict[str, PhaseInfo] = {}
        self.dependency_mappings: List[DependencyMapping] = []

    def validate_all_phases(self) -> Dict[str, PhaseInfo]:
        """Validate all 8 phases and return status information."""
        self.logger.info("Validating all 8 Agent Forge phases...")

        for phase_name, definition in self.phase_definitions.items():
            self.logger.info(f"Validating phase: {phase_name}")
            phase_info = self._validate_single_phase(phase_name, definition)
            self.phase_info[phase_name] = phase_info

        # Create dependency mappings
        self._create_dependency_mappings()

        # Generate validation report
        self._generate_validation_report()

        self.logger.info("Phase validation completed")
        return self.phase_info

    def _validate_single_phase(self, phase_name: str, definition: Dict[str, Any]) -> PhaseInfo:
        """Validate a single phase implementation."""
        file_path = self.phases_dir / definition["file"]

        phase_info = PhaseInfo(
            name=phase_name,
            file_path=str(file_path),
            outputs=definition["expected_outputs"]
        )

        # Check if file exists
        if file_path.exists():
            phase_info.exists = True
            self.logger.debug(f"Phase file exists: {file_path}")
        else:
            phase_info.exists = False
            phase_info.status = "MISSING"
            phase_info.issues.append(f"Phase file not found: {file_path}")
            phase_info.recommendations.append(f"Create {definition['file']} with {definition['class']} implementation")
            return phase_info

        # Try to import the phase
        try:
            spec = importlib.util.spec_from_file_location(phase_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            phase_info.importable = True
            self.logger.debug(f"Phase module imported successfully: {phase_name}")

            # Check for required classes
            phase_class = getattr(module, definition["class"], None)
            config_class = getattr(module, definition["config_class"], None)

            if phase_class:
                phase_info.has_execute_method = hasattr(phase_class, 'execute') or hasattr(phase_class, 'execute_async')
                if not phase_info.has_execute_method:
                    phase_info.issues.append(f"Phase class {definition['class']} missing execute method")
                    phase_info.recommendations.append("Add execute() or execute_async() method to phase class")

            if config_class:
                phase_info.has_config_class = True
            else:
                phase_info.issues.append(f"Config class {definition['config_class']} not found")
                phase_info.recommendations.append(f"Create {definition['config_class']} configuration class")

            # Determine status
            if phase_info.has_execute_method and phase_info.has_config_class:
                phase_info.status = "OPERATIONAL"
            elif phase_info.has_execute_method or phase_info.has_config_class:
                phase_info.status = "NEEDS_FIXES"
            else:
                phase_info.status = "BROKEN"

            # Additional validation based on phase content
            self._validate_phase_content(phase_info, module, definition)

        except Exception as e:
            phase_info.importable = False
            phase_info.status = "BROKEN"
            phase_info.issues.append(f"Import failed: {str(e)}")
            phase_info.recommendations.append("Fix import errors and syntax issues")
            self.logger.error(f"Failed to import {phase_name}: {str(e)}")

        return phase_info

    def _validate_phase_content(self, phase_info: PhaseInfo, module: Any, definition: Dict[str, Any]):
        """Validate phase content and implementation details."""
        try:
            # Check for required methods and attributes
            phase_class = getattr(module, definition["class"], None)
            if phase_class:
                # Check constructor
                init_method = getattr(phase_class, '__init__', None)
                if init_method:
                    # Check if it accepts config parameter
                    import inspect
                    sig = inspect.signature(init_method)
                    if 'config' not in sig.parameters:
                        phase_info.issues.append("Constructor should accept 'config' parameter")

                # Check for async support
                if hasattr(phase_class, 'execute_async'):
                    phase_info.recommendations.append("Async execution supported")

                # Check for checkpoint support
                if hasattr(phase_class, 'save_checkpoint') and hasattr(phase_class, 'load_checkpoint'):
                    phase_info.recommendations.append("Checkpoint support available")
                else:
                    phase_info.recommendations.append("Consider adding checkpoint support")

            # Check for dependencies
            if hasattr(module, '__dependencies__'):
                phase_info.dependencies = getattr(module, '__dependencies__')

        except Exception as e:
            phase_info.issues.append(f"Content validation failed: {str(e)}")

    def _create_dependency_mappings(self):
        """Create comprehensive dependency mappings between phases."""
        self.logger.info("Creating dependency mappings...")

        # Define the standard 8-phase dependencies
        dependency_specs = [
            {
                "source": "CognatePhase",
                "target": "EvoMergePhase",
                "type": "model",
                "required": ["model", "model_path"],
                "optional": ["architecture_info"],
                "validation": ["model_integrity", "parameter_count"]
            },
            {
                "source": "EvoMergePhase",
                "target": "QuietSTaRPhase",
                "type": "model",
                "required": ["best_model", "model_path"],
                "optional": ["fitness_score", "generation_info"],
                "validation": ["model_integrity", "fitness_threshold"]
            },
            {
                "source": "QuietSTaRPhase",
                "target": "BitNetCompressionPhase",
                "type": "model",
                "required": ["enhanced_model", "model_path"],
                "optional": ["reasoning_capability"],
                "validation": ["model_integrity", "reasoning_capability_check"]
            },
            {
                "source": "BitNetCompressionPhase",
                "target": "ForgeTrainingPhase",
                "type": "model",
                "required": ["compressed_model", "model_path"],
                "optional": ["compression_ratio"],
                "validation": ["model_integrity", "compression_quality"]
            },
            {
                "source": "ForgeTrainingPhase",
                "target": "ToolPersonaBakingPhase",
                "type": "model",
                "required": ["trained_model", "model_path"],
                "optional": ["training_loss", "validation_metrics"],
                "validation": ["model_integrity", "training_convergence"]
            },
            {
                "source": "ToolPersonaBakingPhase",
                "target": "ADASPhase",
                "type": "model",
                "required": ["specialized_model", "model_path"],
                "optional": ["specialization_score"],
                "validation": ["model_integrity", "specialization_quality"]
            },
            {
                "source": "ADASPhase",
                "target": "FinalCompressionPhase",
                "type": "model",
                "required": ["optimized_model", "model_path"],
                "optional": ["architecture_score"],
                "validation": ["model_integrity", "architecture_optimization"]
            }
        ]

        for spec in dependency_specs:
            mapping = DependencyMapping(
                source_phase=spec["source"],
                target_phase=spec["target"],
                dependency_type=spec["type"],
                required_outputs=spec["required"],
                optional_outputs=spec["optional"],
                validation_rules=spec["validation"]
            )
            self.dependency_mappings.append(mapping)

        self.logger.info(f"Created {len(self.dependency_mappings)} dependency mappings")

    def _generate_validation_report(self):
        """Generate comprehensive validation report."""
        timestamp = datetime.now().isoformat()

        # Phase status summary
        status_counts = {}
        for phase_info in self.phase_info.values():
            status = phase_info.status
            status_counts[status] = status_counts.get(status, 0) + 1

        # Overall system status
        operational_count = status_counts.get("OPERATIONAL", 0)
        total_phases = len(self.phase_info)
        system_health = operational_count / total_phases

        report = {
            "validation_info": {
                "timestamp": timestamp,
                "validator_version": "1.0.0",
                "total_phases": total_phases,
                "base_directory": str(self.base_dir)
            },
            "system_status": {
                "health_score": system_health,
                "operational_phases": operational_count,
                "total_phases": total_phases,
                "status_distribution": status_counts,
                "ready_for_production": operational_count >= 6  # At least 75% operational
            },
            "phase_details": {
                name: {
                    "status": info.status,
                    "file_path": info.file_path,
                    "exists": info.exists,
                    "importable": info.importable,
                    "has_execute_method": info.has_execute_method,
                    "has_config_class": info.has_config_class,
                    "expected_outputs": info.outputs,
                    "issues": info.issues,
                    "recommendations": info.recommendations
                }
                for name, info in self.phase_info.items()
            },
            "dependency_mappings": [
                {
                    "source_phase": mapping.source_phase,
                    "target_phase": mapping.target_phase,
                    "dependency_type": mapping.dependency_type,
                    "required_outputs": mapping.required_outputs,
                    "optional_outputs": mapping.optional_outputs,
                    "validation_rules": mapping.validation_rules
                }
                for mapping in self.dependency_mappings
            ],
            "recommendations": self._generate_system_recommendations()
        }

        # Save report
        report_file = self.base_dir / "phase_validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Validation report saved: {report_file}")

        # Log summary
        self._log_validation_summary(report)

        return report

    def _generate_system_recommendations(self) -> List[str]:
        """Generate system-level recommendations."""
        recommendations = []

        # Count issues by type
        missing_count = sum(1 for info in self.phase_info.values() if info.status == "MISSING")
        broken_count = sum(1 for info in self.phase_info.values() if info.status == "BROKEN")
        needs_fixes_count = sum(1 for info in self.phase_info.values() if info.status == "NEEDS_FIXES")

        if missing_count > 0:
            recommendations.append(f"Implement {missing_count} missing phase(s)")

        if broken_count > 0:
            recommendations.append(f"Fix {broken_count} broken phase(s)")

        if needs_fixes_count > 0:
            recommendations.append(f"Address issues in {needs_fixes_count} phase(s) that need fixes")

        # System-level recommendations
        operational_count = sum(1 for info in self.phase_info.values() if info.status == "OPERATIONAL")

        if operational_count < 4:
            recommendations.append("CRITICAL: Less than 50% of phases are operational")
        elif operational_count < 6:
            recommendations.append("WARNING: Less than 75% of phases are operational")
        elif operational_count == 8:
            recommendations.append("EXCELLENT: All phases are operational")

        # Specific technical recommendations
        if any("execute method" in issue for info in self.phase_info.values() for issue in info.issues):
            recommendations.append("Standardize execute() method signatures across all phases")

        if any("config class" in issue for info in self.phase_info.values() for issue in info.issues):
            recommendations.append("Create missing configuration classes")

        recommendations.append("Run integration tests to validate phase interactions")
        recommendations.append("Consider adding comprehensive error handling and logging")

        return recommendations

    def _log_validation_summary(self, report: Dict[str, Any]):
        """Log validation summary."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PHASE VALIDATION SUMMARY")
        self.logger.info("=" * 80)

        system_status = report["system_status"]
        self.logger.info(f"System Health Score: {system_status['health_score']:.1%}")
        self.logger.info(f"Operational Phases: {system_status['operational_phases']}/{system_status['total_phases']}")
        self.logger.info(f"Production Ready: {'YES' if system_status['ready_for_production'] else 'NO'}")

        self.logger.info("\nPhase Status:")
        for phase_name, details in report["phase_details"].items():
            status = details["status"]
            issues_count = len(details["issues"])
            self.logger.info(f"  {phase_name}: {status} ({issues_count} issues)")

        self.logger.info(f"\nDependency Mappings: {len(report['dependency_mappings'])} configured")

        if report["recommendations"]:
            self.logger.info("\nKey Recommendations:")
            for rec in report["recommendations"][:5]:  # Show top 5
                self.logger.info(f"  - {rec}")

        self.logger.info("=" * 80)

    def get_phase_execution_order(self) -> List[str]:
        """Get the correct execution order based on dependencies."""
        # Use topological sort based on dependencies
        order = [
            "CognatePhase",
            "EvoMergePhase",
            "QuietSTaRPhase",
            "BitNetCompressionPhase",
            "ForgeTrainingPhase",
            "ToolPersonaBakingPhase",
            "ADASPhase",
            "FinalCompressionPhase"
        ]

        # Filter to only include operational phases
        operational_phases = [
            name for name, info in self.phase_info.items()
            if info.status == "OPERATIONAL"
        ]

        return [phase for phase in order if phase in operational_phases]

    def get_ready_phases(self) -> List[str]:
        """Get list of phases ready for execution."""
        return [
            name for name, info in self.phase_info.items()
            if info.status in ["OPERATIONAL", "NEEDS_FIXES"]
        ]

    def get_blocking_issues(self) -> Dict[str, List[str]]:
        """Get issues that would block pipeline execution."""
        blocking_issues = {}

        for name, info in self.phase_info.items():
            if info.status in ["MISSING", "BROKEN"]:
                blocking_issues[name] = info.issues

        return blocking_issues


def main():
    """Main function for standalone execution."""
    logging.basicConfig(level=logging.INFO)

    validator = PhaseStatusValidator()
    phase_info = validator.validate_all_phases()

    print("\nPhase Validation Complete!")
    print(f"Report saved to: {validator.base_dir / 'phase_validation_report.json'}")

    # Show quick summary
    operational = [name for name, info in phase_info.items() if info.status == "OPERATIONAL"]
    print(f"\nOperational Phases ({len(operational)}/8): {', '.join(operational)}")

    blocking_issues = validator.get_blocking_issues()
    if blocking_issues:
        print(f"\nBlocking Issues Found in {len(blocking_issues)} phases:")
        for phase, issues in blocking_issues.items():
            print(f"  {phase}: {len(issues)} issues")


if __name__ == "__main__":
    main()