"""
Enhanced Main Pipeline for Agent Forge 8-Phase Integration

Complete orchestration system with comprehensive monitoring, validation,
and error recovery for the full 8-phase Agent Forge pipeline.
"""

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import orchestration components
from orchestration import (
    PipelineController,
    PipelineConfig,
    ResourceConstraints,
    PhaseValidationSuite,
    PhaseOrchestrator,
    PhaseResult,
    PhaseMetrics,
    PhaseType
)

# Import unified pipeline configuration
from unified_pipeline import UnifiedConfig, UnifiedPipeline


@dataclass
class EnhancedPipelineConfig:
    """Enhanced configuration that combines unified and orchestration configs."""

    # Unified pipeline settings
    unified_config: UnifiedConfig = field(default_factory=UnifiedConfig)

    # Orchestration settings
    orchestration_config: PipelineConfig = field(default_factory=lambda: PipelineConfig(
        output_dir="./enhanced_pipeline_output",
        checkpoint_dir="./enhanced_checkpoints",
        logs_dir="./enhanced_logs",
        artifacts_dir="./enhanced_artifacts",
        resource_constraints=ResourceConstraints(
            max_memory_gb=32.0,
            max_gpu_memory_gb=24.0,
            max_execution_time_hours=12.0,
            min_disk_space_gb=50.0
        ),
        enable_performance_monitoring=True,
        enable_progress_reporting=True,
        enable_health_checks=True,
        enable_auto_recovery=True,
        enable_integration_tests=True
    ))

    # Validation settings
    enable_comprehensive_validation: bool = True
    enable_quality_gates: bool = True
    validation_output_dir: str = "./validation_output"

    # Monitoring and reporting
    enable_wandb: bool = False
    wandb_project: str = "agent-forge-enhanced"
    enable_tensorboard: bool = True
    tensorboard_log_dir: str = "./tensorboard_logs"

    # Recovery and resilience
    enable_auto_checkpointing: bool = True
    checkpoint_interval_phases: int = 1  # Checkpoint after every phase
    enable_phase_rollback: bool = True
    max_retry_attempts: int = 3


class EnhancedAgentForgePipeline:
    """
    Enhanced Agent Forge Pipeline with comprehensive orchestration.

    Combines the UnifiedPipeline with advanced orchestration capabilities
    including monitoring, validation, error recovery, and quality gates.
    """

    def __init__(self, config: EnhancedPipelineConfig):
        self.config = config
        self.logger = self._setup_logging()

        # Initialize core components
        self.unified_pipeline = UnifiedPipeline(config.unified_config)
        self.pipeline_controller = PipelineController(config.orchestration_config)

        # Initialize validation suite
        if config.enable_comprehensive_validation:
            self.validation_suite = PhaseValidationSuite(config.validation_output_dir)
        else:
            self.validation_suite = None

        # Pipeline state
        self.execution_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.phase_results: List[PhaseResult] = []
        self.validation_results: Dict[str, Any] = {}

        # Progress tracking
        self.progress_callbacks: List[callable] = []
        self.setup_progress_tracking()

        self.logger.info("Enhanced Agent Forge Pipeline initialized")
        self.logger.info(f"Execution ID: {self.execution_id}")

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger(f"{__name__}.EnhancedPipeline")
        logger.setLevel(logging.INFO)

        # Create logs directory
        logs_dir = Path(self.config.orchestration_config.logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)

        # File handler
        log_file = logs_dir / f"enhanced_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def setup_progress_tracking(self):
        """Setup progress tracking and callbacks."""
        def log_progress(event: str, data: Dict[str, Any]):
            if event == "pipeline_started":
                self.logger.info(f"Pipeline started with {data.get('total_phases', 0)} phases")
            elif event == "phase_ready":
                self.logger.info(f"Phase {data.get('phase')} ready for execution")
            elif event == "phase_completed":
                phase = data.get('phase')
                success = data.get('success', False)
                status = "SUCCESS" if success else "FAILED"
                self.logger.info(f"Phase {phase} completed: {status}")
            elif event == "pipeline_completed":
                duration = data.get('total_duration', 0)
                completed = data.get('phases_completed', 0)
                self.logger.info(f"Pipeline completed successfully in {duration:.2f}s ({completed} phases)")
            elif event == "pipeline_failed":
                error = data.get('error', 'Unknown error')
                self.logger.error(f"Pipeline failed: {error}")
            elif event == "progress_report":
                percent = data.get('progress_percent', 0)
                current_phase = data.get('current_phase', 'Unknown')
                self.logger.info(f"Progress: {percent:.1f}% (Current: {current_phase})")

        # Add built-in progress logger
        self.progress_callbacks.append(log_progress)
        self.pipeline_controller.add_progress_callback(log_progress)

    def add_progress_callback(self, callback: callable):
        """Add a custom progress callback."""
        self.progress_callbacks.append(callback)
        self.pipeline_controller.add_progress_callback(callback)

    async def execute_complete_pipeline(self,
                                      resume_from: Optional[str] = None,
                                      skip_validation: bool = False) -> PhaseResult:
        """
        Execute the complete enhanced 8-phase pipeline.

        Args:
            resume_from: Optional phase name to resume from
            skip_validation: Skip comprehensive validation (faster execution)

        Returns:
            Final pipeline result with comprehensive metrics
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING ENHANCED AGENT FORGE 8-PHASE PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Execution ID: {self.execution_id}")

        self.start_time = datetime.now()

        try:
            # Pre-execution setup
            await self._pre_execution_setup()

            # Get phases from unified pipeline
            phases = self.unified_pipeline.phases

            # Execute through enhanced controller
            self.logger.info(f"Executing {len(phases)} phases through enhanced controller")
            pipeline_result = await self.pipeline_controller.execute_full_pipeline(
                phases=phases,
                initial_model=None,  # Let Cognate create the model
                resume_from=resume_from
            )

            # Collect phase results
            self.phase_results = list(self.pipeline_controller.orchestrator.phase_results.values())

            # Post-execution validation
            if not skip_validation and self.validation_suite:
                await self._comprehensive_validation()

            # Post-execution processing
            await self._post_execution_processing(pipeline_result)

            self.end_time = datetime.now()
            total_duration = (self.end_time - self.start_time).total_seconds()

            self.logger.info("=" * 80)
            if pipeline_result.success:
                self.logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_duration:.2f} SECONDS")
            else:
                self.logger.error(f"PIPELINE FAILED AFTER {total_duration:.2f} SECONDS")
            self.logger.info("=" * 80)

            return pipeline_result

        except Exception as e:
            self.end_time = datetime.now()
            error_msg = f"Enhanced pipeline execution failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            # Create failure result
            failure_result = PhaseResult(
                success=False,
                phase_name="EnhancedPipeline",
                phase_type=PhaseType.FINALIZATION,
                error=error_msg,
                metrics=self._get_current_metrics()
            )

            await self._handle_pipeline_failure(failure_result)
            return failure_result

    async def _pre_execution_setup(self):
        """Pre-execution setup and validation."""
        self.logger.info("Performing pre-execution setup...")

        # Create output directories
        directories = [
            self.config.orchestration_config.output_dir,
            self.config.orchestration_config.checkpoint_dir,
            self.config.orchestration_config.artifacts_dir,
            self.config.validation_output_dir
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

        # Setup monitoring
        if self.config.enable_tensorboard:
            await self._setup_tensorboard()

        if self.config.enable_wandb:
            await self._setup_wandb()

        # System health check
        await self._system_health_check()

        self.logger.info("Pre-execution setup completed")

    async def _setup_tensorboard(self):
        """Setup TensorBoard logging."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = Path(self.config.tensorboard_log_dir) / self.execution_id
            tb_dir.mkdir(parents=True, exist_ok=True)

            self.tensorboard_writer = SummaryWriter(tb_dir)
            self.logger.info(f"TensorBoard logging setup: {tb_dir}")
        except ImportError:
            self.logger.warning("TensorBoard not available, skipping")

    async def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        try:
            import wandb

            wandb.init(
                project=self.config.wandb_project,
                name=self.execution_id,
                config={
                    "unified_config": self.config.unified_config.__dict__,
                    "orchestration_config": self.config.orchestration_config.__dict__
                },
                tags=["agent-forge", "enhanced-pipeline"]
            )
            self.logger.info("W&B logging initialized")
        except ImportError:
            self.logger.warning("W&B not available, skipping")

    async def _system_health_check(self):
        """Perform system health check."""
        import psutil

        # Check available resources
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')

        available_memory_gb = memory.available / (1024**3)
        available_disk_gb = disk.free / (1024**3)

        constraints = self.config.orchestration_config.resource_constraints

        health_report = {
            "memory_available_gb": available_memory_gb,
            "memory_required_gb": constraints.max_memory_gb,
            "disk_available_gb": available_disk_gb,
            "disk_required_gb": constraints.min_disk_space_gb,
            "cpu_cores": psutil.cpu_count()
        }

        # Check requirements
        if available_memory_gb < constraints.max_memory_gb:
            self.logger.warning(f"Limited memory: {available_memory_gb:.1f}GB < {constraints.max_memory_gb}GB")

        if available_disk_gb < constraints.min_disk_space_gb:
            raise RuntimeError(f"Insufficient disk space: {available_disk_gb:.1f}GB < {constraints.min_disk_space_gb}GB")

        # GPU check
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            health_report["gpu_count"] = gpu_count
            health_report["gpu_memory_gb"] = gpu_memory
            self.logger.info(f"GPU available: {gpu_count} devices, {gpu_memory:.1f}GB memory")

        self.logger.info(f"System health check completed: {health_report}")

    async def _comprehensive_validation(self):
        """Perform comprehensive validation of all phases."""
        self.logger.info("Performing comprehensive validation...")

        if not self.phase_results:
            self.logger.warning("No phase results to validate")
            return

        # Validate each phase
        phase_validations = {}
        for phase_result in self.phase_results:
            validation_result = await self.validation_suite.validate_phase(phase_result)
            phase_validations[phase_result.phase_name] = validation_result

        # Validate complete pipeline
        pipeline_validation = await self.validation_suite.validate_pipeline(self.phase_results)

        self.validation_results = {
            "individual_phases": phase_validations,
            "pipeline_validation": pipeline_validation,
            "validation_summary": self._create_validation_summary(pipeline_validation)
        }

        self.logger.info("Comprehensive validation completed")

    def _create_validation_summary(self, pipeline_validation: Dict) -> Dict[str, Any]:
        """Create validation summary statistics."""
        total_validations = 0
        passed_validations = 0
        total_errors = 0
        total_warnings = 0

        for phase_validation in pipeline_validation.values():
            for validation_result in phase_validation.values():
                total_validations += 1
                if validation_result.valid:
                    passed_validations += 1
                total_errors += len(validation_result.errors)
                total_warnings += len(validation_result.warnings)

        return {
            "total_validations": total_validations,
            "passed_validations": passed_validations,
            "failed_validations": total_validations - passed_validations,
            "success_rate": passed_validations / total_validations if total_validations > 0 else 0,
            "total_errors": total_errors,
            "total_warnings": total_warnings
        }

    async def _post_execution_processing(self, pipeline_result: PhaseResult):
        """Post-execution processing and reporting."""
        self.logger.info("Performing post-execution processing...")

        # Generate comprehensive report
        await self._generate_comprehensive_report(pipeline_result)

        # Save final model if successful
        if pipeline_result.success and pipeline_result.model:
            await self._save_final_model(pipeline_result.model)

        # Performance analysis
        await self._generate_performance_analysis()

        # Cleanup if requested
        await self._cleanup_temporary_files()

        self.logger.info("Post-execution processing completed")

    async def _generate_comprehensive_report(self, pipeline_result: PhaseResult):
        """Generate comprehensive execution report."""
        report = {
            "execution_info": {
                "execution_id": self.execution_id,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "total_duration_seconds": (
                    (self.end_time - self.start_time).total_seconds()
                    if self.start_time and self.end_time else None
                ),
                "success": pipeline_result.success,
                "error": pipeline_result.error
            },
            "configuration": {
                "unified_config": self.config.unified_config.__dict__,
                "orchestration_config": self.config.orchestration_config.__dict__
            },
            "phase_results": {
                result.phase_name: {
                    "success": result.success,
                    "duration_seconds": result.metrics.duration_seconds if result.metrics else 0,
                    "error": result.error,
                    "output_keys": list(result.output_data.keys()) if result.output_data else []
                }
                for result in self.phase_results
            },
            "validation_results": self.validation_results,
            "performance_metrics": {
                "controller_metrics": self.pipeline_controller.performance_metrics,
                "resource_usage": self.pipeline_controller._get_resource_usage_summary()
            }
        }

        # Save report
        artifacts_dir = Path(self.config.orchestration_config.artifacts_dir)
        report_file = artifacts_dir / f"comprehensive_report_{self.execution_id}.json"

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Comprehensive report saved: {report_file}")

        # Log summary
        self._log_execution_summary(report)

    def _log_execution_summary(self, report: Dict[str, Any]):
        """Log execution summary."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXECUTION SUMMARY")
        self.logger.info("=" * 80)

        # Basic info
        exec_info = report["execution_info"]
        self.logger.info(f"Execution ID: {exec_info['execution_id']}")
        self.logger.info(f"Success: {exec_info['success']}")
        self.logger.info(f"Duration: {exec_info['total_duration_seconds']:.2f} seconds")

        # Phase results
        self.logger.info("\nPhase Results:")
        for phase_name, result in report["phase_results"].items():
            status = "SUCCESS" if result["success"] else "FAILED"
            duration = result["duration_seconds"]
            self.logger.info(f"  {phase_name}: {status} ({duration:.2f}s)")

        # Validation summary
        if "validation_results" in report and report["validation_results"]:
            validation_summary = report["validation_results"].get("validation_summary", {})
            if validation_summary:
                self.logger.info(f"\nValidation Summary:")
                self.logger.info(f"  Success Rate: {validation_summary.get('success_rate', 0):.1%}")
                self.logger.info(f"  Total Errors: {validation_summary.get('total_errors', 0)}")
                self.logger.info(f"  Total Warnings: {validation_summary.get('total_warnings', 0)}")

        self.logger.info("=" * 80)

    async def _save_final_model(self, model: nn.Module):
        """Save the final trained model."""
        model_dir = Path(self.config.orchestration_config.output_dir) / "final_models"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"final_model_{self.execution_id}.pt"

        torch.save({
            "model_state_dict": model.state_dict(),
            "execution_id": self.execution_id,
            "timestamp": datetime.now().isoformat(),
            "config": self.config.unified_config.__dict__
        }, model_path)

        self.logger.info(f"Final model saved: {model_path}")

    async def _generate_performance_analysis(self):
        """Generate detailed performance analysis."""
        if not self.phase_results:
            return

        analysis = {
            "phase_performance": {},
            "bottleneck_analysis": {},
            "resource_efficiency": {},
            "recommendations": []
        }

        # Analyze each phase
        total_duration = 0
        for result in self.phase_results:
            if result.metrics:
                phase_name = result.phase_name
                duration = result.metrics.duration_seconds
                memory = result.metrics.memory_usage_mb

                analysis["phase_performance"][phase_name] = {
                    "duration_seconds": duration,
                    "memory_usage_mb": memory,
                    "parameters": result.metrics.model_parameters,
                    "efficiency_score": self._calculate_phase_efficiency(result.metrics)
                }

                total_duration += duration

        # Identify bottlenecks
        if analysis["phase_performance"]:
            slowest_phase = max(analysis["phase_performance"].items(),
                              key=lambda x: x[1]["duration_seconds"])
            analysis["bottleneck_analysis"]["slowest_phase"] = {
                "name": slowest_phase[0],
                "duration": slowest_phase[1]["duration_seconds"],
                "percentage_of_total": (slowest_phase[1]["duration_seconds"] / total_duration) * 100
            }

        # Save analysis
        artifacts_dir = Path(self.config.orchestration_config.artifacts_dir)
        analysis_file = artifacts_dir / f"performance_analysis_{self.execution_id}.json"

        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        self.logger.info(f"Performance analysis saved: {analysis_file}")

    def _calculate_phase_efficiency(self, metrics: PhaseMetrics) -> float:
        """Calculate efficiency score for a phase."""
        # Simple efficiency metric based on duration and memory usage
        if metrics.duration_seconds > 0 and metrics.memory_usage_mb > 0:
            # Inverse relationship - lower time and memory = higher efficiency
            return 1.0 / (metrics.duration_seconds * metrics.memory_usage_mb / 1000)
        return 0.0

    async def _cleanup_temporary_files(self):
        """Clean up temporary files if configured."""
        # This could clean up intermediate model files, temporary data, etc.
        pass

    async def _handle_pipeline_failure(self, failure_result: PhaseResult):
        """Handle pipeline failure with comprehensive error reporting."""
        self.logger.error("Handling pipeline failure...")

        # Generate failure report
        failure_report = {
            "execution_id": self.execution_id,
            "failure_time": datetime.now().isoformat(),
            "error": failure_result.error,
            "completed_phases": [r.phase_name for r in self.phase_results if r.success],
            "failed_phase": failure_result.phase_name,
            "partial_results": len(self.phase_results),
            "recovery_suggestions": self._generate_recovery_suggestions()
        }

        # Save failure report
        artifacts_dir = Path(self.config.orchestration_config.artifacts_dir)
        failure_file = artifacts_dir / f"failure_report_{self.execution_id}.json"

        with open(failure_file, 'w') as f:
            json.dump(failure_report, f, indent=2)

        self.logger.error(f"Failure report saved: {failure_file}")

    def _generate_recovery_suggestions(self) -> List[str]:
        """Generate recovery suggestions based on failure analysis."""
        suggestions = [
            "Check system resources (memory, disk space, GPU)",
            "Verify all required dependencies are installed",
            "Review phase-specific error logs for detailed diagnostics",
            "Consider resuming from the last successful checkpoint",
            "Adjust resource constraints if needed"
        ]

        if self.phase_results:
            last_phase = self.phase_results[-1].phase_name
            suggestions.append(f"Resume execution from phase: {last_phase}")

        return suggestions

    def _get_current_metrics(self) -> PhaseMetrics:
        """Get current pipeline metrics."""
        return PhaseMetrics(
            phase_name="EnhancedPipeline",
            start_time=self.start_time,
            end_time=self.end_time or datetime.now(),
            duration_seconds=(
                (self.end_time or datetime.now()) - self.start_time
            ).total_seconds() if self.start_time else 0,
            custom_metrics={
                "execution_id": self.execution_id,
                "completed_phases": len([r for r in self.phase_results if r.success]),
                "total_phases": len(self.phase_results)
            }
        )


# CLI and utility functions
async def run_enhanced_pipeline(config_path: Optional[str] = None, **kwargs) -> PhaseResult:
    """
    Run the enhanced pipeline with optional configuration file.

    Args:
        config_path: Optional path to configuration file
        **kwargs: Override configuration parameters

    Returns:
        Final pipeline result
    """
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config_dict = json.load(f)
        config_dict.update(kwargs)

        # Split configuration
        unified_config = UnifiedConfig(**config_dict.get("unified_config", {}))
        orchestration_config = PipelineConfig(**config_dict.get("orchestration_config", {}))

        config = EnhancedPipelineConfig(
            unified_config=unified_config,
            orchestration_config=orchestration_config
        )
    else:
        config = EnhancedPipelineConfig()
        # Apply any kwargs overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Create and run pipeline
    pipeline = EnhancedAgentForgePipeline(config)
    return await pipeline.execute_complete_pipeline()


def create_sample_config() -> Dict[str, Any]:
    """Create a sample configuration file."""
    return {
        "unified_config": {
            "base_models": [
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                "nvidia/Nemotron-Research-Reasoning-Qwen-1.5B"
            ],
            "enable_cognate": True,
            "enable_evomerge": True,
            "enable_quietstar": True,
            "enable_initial_compression": True,
            "enable_training": True,
            "enable_tool_baking": True,
            "enable_adas": True,
            "enable_final_compression": True,
            "evomerge_generations": 25,
            "training_steps": 50000,
            "grokfast_enabled": True
        },
        "orchestration_config": {
            "output_dir": "./enhanced_output",
            "enable_performance_monitoring": True,
            "enable_progress_reporting": True,
            "enable_auto_recovery": True,
            "resource_constraints": {
                "max_memory_gb": 16.0,
                "max_gpu_memory_gb": 12.0,
                "max_execution_time_hours": 8.0
            }
        },
        "enable_comprehensive_validation": True,
        "enable_quality_gates": True,
        "enable_wandb": False,
        "enable_tensorboard": True
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Agent Forge Pipeline")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--resume-from", type=str, help="Phase to resume from")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation")
    parser.add_argument("--create-config", action="store_true", help="Create sample config")

    args = parser.parse_args()

    if args.create_config:
        config = create_sample_config()
        with open("enhanced_pipeline_config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("Sample configuration saved to enhanced_pipeline_config.json")
        sys.exit(0)

    # Run pipeline
    async def main():
        try:
            result = await run_enhanced_pipeline(
                config_path=args.config,
                resume_from=args.resume_from,
                skip_validation=args.skip_validation
            )
            if result.success:
                print("Pipeline completed successfully!")
                sys.exit(0)
            else:
                print(f"Pipeline failed: {result.error}")
                sys.exit(1)
        except Exception as e:
            print(f"Pipeline execution failed: {str(e)}")
            sys.exit(1)

    asyncio.run(main())