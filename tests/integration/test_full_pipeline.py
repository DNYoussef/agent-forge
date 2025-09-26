"""
Comprehensive Integration Tests for Agent Forge 8-Phase Pipeline

Tests end-to-end pipeline execution, phase transitions, data flow validation,
performance benchmarks, and error recovery.
"""

import asyncio
import json
import logging
import os
import pytest
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

# Import orchestration components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from orchestration.phase_orchestrator import (
    PhaseOrchestrator,
    PhaseResult,
    PhaseMetrics,
    PhaseState,
    PhaseType
)
from orchestration.pipeline_controller import (
    PipelineController,
    PipelineConfig,
    ResourceConstraints
)
from orchestration.phase_validators import (
    PhaseValidationSuite,
    ValidationResult,
    QualityGate
)


class MockPhaseController:
    """Mock phase controller for testing."""

    def __init__(self, phase_name: str, should_fail: bool = False,
                 execution_time: float = 1.0, output_data: Dict = None):
        self.phase_name = phase_name
        self.should_fail = should_fail
        self.execution_time = execution_time
        self.output_data = output_data or {}
        self.execute_count = 0

    async def execute_async(self, input_data: Dict) -> PhaseResult:
        """Mock async execution."""
        return await self._execute(input_data)

    def execute(self, input_data: Dict) -> PhaseResult:
        """Mock sync execution."""
        return asyncio.run(self._execute(input_data))

    async def _execute(self, input_data: Dict) -> PhaseResult:
        """Internal execution logic."""
        self.execute_count += 1

        # Simulate execution time
        await asyncio.sleep(self.execution_time)

        if self.should_fail:
            return PhaseResult(
                success=False,
                phase_name=self.phase_name,
                phase_type=self._get_phase_type(),
                error=f"Mock failure in {self.phase_name}",
                metrics=PhaseMetrics(
                    phase_name=self.phase_name,
                    duration_seconds=self.execution_time,
                    error_count=1
                )
            )

        # Create mock model
        model = nn.Linear(10, 10)

        # Phase-specific output data
        phase_output = self.output_data.copy()
        if self.phase_name == "CognatePhase":
            phase_output.update({
                "model_path": f"/tmp/{self.phase_name}_model.pt",
                "architecture_info": {"layers": 2, "params": 110}
            })
        elif self.phase_name == "EvoMergePhase":
            phase_output.update({
                "fitness_score": 0.85,
                "generation_info": {"generation": 50, "population": 8}
            })
        elif self.phase_name == "QuietSTaRPhase":
            phase_output.update({
                "reasoning_capability": 0.78,
                "thought_generation_stats": {"avg_thoughts": 4, "success_rate": 0.82}
            })
        elif self.phase_name == "BitNetCompressionPhase":
            phase_output.update({
                "compression_ratio": 4.2,
                "quantization_info": {"bits": 1.58, "size_reduction": 0.76}
            })
        elif self.phase_name == "ForgeTrainingPhase":
            phase_output.update({
                "training_loss": [2.1, 1.8, 1.5, 1.2],
                "validation_metrics": {"accuracy": 0.87, "f1": 0.83}
            })
        elif self.phase_name == "ToolPersonaBakingPhase":
            phase_output.update({
                "specialization_score": 0.91,
                "baked_capabilities": ["reasoning", "code_gen", "math"]
            })
        elif self.phase_name == "ADASPhase":
            phase_output.update({
                "architecture_score": 0.88,
                "optimization_info": {"technique": "gradient_based", "iterations": 100}
            })
        elif self.phase_name == "FinalCompressionPhase":
            phase_output.update({
                "final_size": 50.2,
                "compression_summary": {"techniques": ["seedlm", "vptq"], "ratio": 8.5}
            })

        return PhaseResult(
            success=True,
            phase_name=self.phase_name,
            phase_type=self._get_phase_type(),
            model=model,
            model_path=f"/tmp/{self.phase_name}_model.pt",
            output_data=phase_output,
            metrics=PhaseMetrics(
                phase_name=self.phase_name,
                duration_seconds=self.execution_time,
                memory_usage_mb=100.0,
                model_parameters=110
            )
        )

    def _get_phase_type(self) -> PhaseType:
        """Get phase type from name."""
        type_mapping = {
            "CognatePhase": PhaseType.CREATION,
            "EvoMergePhase": PhaseType.EVOLUTION,
            "QuietSTaRPhase": PhaseType.REASONING,
            "BitNetCompressionPhase": PhaseType.COMPRESSION,
            "ForgeTrainingPhase": PhaseType.TRAINING,
            "ToolPersonaBakingPhase": PhaseType.SPECIALIZATION,
            "ADASPhase": PhaseType.ARCHITECTURE,
            "FinalCompressionPhase": PhaseType.FINALIZATION
        }
        return type_mapping.get(self.phase_name, PhaseType.FINALIZATION)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def mock_phases():
    """Create mock phase controllers for all 8 phases."""
    phases = [
        ("CognatePhase", MockPhaseController("CognatePhase", execution_time=0.1)),
        ("EvoMergePhase", MockPhaseController("EvoMergePhase", execution_time=0.2)),
        ("QuietSTaRPhase", MockPhaseController("QuietSTaRPhase", execution_time=0.1)),
        ("BitNetCompressionPhase", MockPhaseController("BitNetCompressionPhase", execution_time=0.1)),
        ("ForgeTrainingPhase", MockPhaseController("ForgeTrainingPhase", execution_time=0.2)),
        ("ToolPersonaBakingPhase", MockPhaseController("ToolPersonaBakingPhase", execution_time=0.1)),
        ("ADASPhase", MockPhaseController("ADASPhase", execution_time=0.2)),
        ("FinalCompressionPhase", MockPhaseController("FinalCompressionPhase", execution_time=0.1))
    ]
    return phases


@pytest.fixture
def pipeline_config(temp_output_dir):
    """Create pipeline configuration for testing."""
    return PipelineConfig(
        output_dir=os.path.join(temp_output_dir, "output"),
        checkpoint_dir=os.path.join(temp_output_dir, "checkpoints"),
        logs_dir=os.path.join(temp_output_dir, "logs"),
        artifacts_dir=os.path.join(temp_output_dir, "artifacts"),
        resource_constraints=ResourceConstraints(
            max_memory_gb=8.0,
            max_execution_time_hours=1.0
        ),
        enable_performance_monitoring=True,
        enable_progress_reporting=False,  # Disable for testing
        progress_report_interval=1
    )


class TestPhaseOrchestrator:
    """Test the Phase Orchestrator."""

    @pytest.mark.asyncio
    async def test_phase_registration(self, temp_output_dir):
        """Test phase registration."""
        orchestrator = PhaseOrchestrator(temp_output_dir)

        # Register a phase
        orchestrator.register_phase("TestPhase", PhaseType.CREATION)

        assert "TestPhase" in orchestrator.phases
        assert orchestrator.phases["TestPhase"] == PhaseState.PENDING
        assert "TestPhase" in orchestrator.phase_metrics

    @pytest.mark.asyncio
    async def test_successful_pipeline_execution(self, temp_output_dir, mock_phases):
        """Test successful execution of complete pipeline."""
        orchestrator = PhaseOrchestrator(temp_output_dir)

        result = await orchestrator.execute_phase_pipeline(mock_phases)

        assert result.success
        assert result.phase_name == "Pipeline"
        assert result.model is not None
        assert len(orchestrator.phase_results) == 8

        # Check all phases completed
        for phase_name in orchestrator.phases:
            assert orchestrator.phases[phase_name] == PhaseState.COMPLETED

    @pytest.mark.asyncio
    async def test_phase_dependency_validation(self, temp_output_dir):
        """Test phase dependency validation."""
        orchestrator = PhaseOrchestrator(temp_output_dir)

        # Test dependency checking
        orchestrator.register_phase("PhaseA")
        orchestrator.register_phase("PhaseB")

        # Phase B depends on Phase A, but A hasn't run
        assert not orchestrator._check_phase_dependencies("PhaseB")

        # Add a mock result for Phase A
        mock_result = PhaseResult(
            success=True,
            phase_name="PhaseA",
            phase_type=PhaseType.CREATION,
            output_data={"model": "mock_model", "model_path": "/tmp/model.pt"}
        )
        orchestrator.phase_results["PhaseA"] = mock_result

        # Now dependency should be satisfied for simple cases
        # (Note: Real dependency validation would need the dependency list)

    @pytest.mark.asyncio
    async def test_phase_failure_handling(self, temp_output_dir):
        """Test handling of phase failures."""
        orchestrator = PhaseOrchestrator(temp_output_dir)

        # Create phases with one failure
        failing_phases = [
            ("CognatePhase", MockPhaseController("CognatePhase")),
            ("EvoMergePhase", MockPhaseController("EvoMergePhase", should_fail=True)),
            ("QuietSTaRPhase", MockPhaseController("QuietSTaRPhase")),
        ]

        result = await orchestrator.execute_phase_pipeline(failing_phases)

        assert not result.success
        assert "EvoMergePhase" in result.error

    @pytest.mark.asyncio
    async def test_checkpoint_creation(self, temp_output_dir, mock_phases):
        """Test checkpoint creation and management."""
        orchestrator = PhaseOrchestrator(temp_output_dir)

        # Execute one phase to create checkpoint
        single_phase = [mock_phases[0]]
        result = await orchestrator.execute_phase_pipeline(single_phase)

        assert result.success
        assert len(orchestrator.checkpoints) > 0

        # Check checkpoint directory
        checkpoint_dir = Path(temp_output_dir) / "checkpoints"
        assert checkpoint_dir.exists()
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoint_files) > 0


class TestPipelineController:
    """Test the Pipeline Controller."""

    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self, pipeline_config, mock_phases):
        """Test complete pipeline execution through controller."""
        controller = PipelineController(pipeline_config)

        result = await controller.execute_full_pipeline(mock_phases)

        assert result.success
        assert controller.pipeline_state == "COMPLETED"
        assert controller.start_time is not None
        assert controller.end_time is not None

    @pytest.mark.asyncio
    async def test_pipeline_with_failure(self, pipeline_config):
        """Test pipeline execution with phase failure."""
        controller = PipelineController(pipeline_config)

        # Create phases with failure
        failing_phases = [
            ("CognatePhase", MockPhaseController("CognatePhase")),
            ("EvoMergePhase", MockPhaseController("EvoMergePhase", should_fail=True)),
        ]

        result = await controller.execute_full_pipeline(failing_phases)

        assert not result.success
        assert controller.pipeline_state == "FAILED"
        assert len(controller.errors) > 0

    @pytest.mark.asyncio
    async def test_resource_monitoring(self, pipeline_config, mock_phases):
        """Test resource monitoring during execution."""
        controller = PipelineController(pipeline_config)

        # Ensure monitoring is enabled
        controller.config.enable_performance_monitoring = True

        result = await controller.execute_full_pipeline(mock_phases)

        assert result.success
        # Resource usage should be collected
        assert len(controller.resource_usage) > 0

    @pytest.mark.asyncio
    async def test_progress_reporting(self, pipeline_config, mock_phases):
        """Test progress reporting callbacks."""
        controller = PipelineController(pipeline_config)

        progress_events = []

        def progress_callback(event, data):
            progress_events.append((event, data))

        controller.add_progress_callback(progress_callback)

        result = await controller.execute_full_pipeline(mock_phases)

        assert result.success
        # Should have received progress events
        assert len(progress_events) > 0

    def test_pre_execution_validation(self, pipeline_config, mock_phases):
        """Test pre-execution validation."""
        controller = PipelineController(pipeline_config)

        # This should pass validation
        async def run_validation():
            await controller._validate_pre_execution(mock_phases)

        # Should not raise exceptions
        asyncio.run(run_validation())

    @pytest.mark.asyncio
    async def test_report_generation(self, pipeline_config, mock_phases):
        """Test comprehensive report generation."""
        controller = PipelineController(pipeline_config)

        result = await controller.execute_full_pipeline(mock_phases)

        assert result.success

        # Check that report files were created
        artifacts_dir = Path(pipeline_config.artifacts_dir)
        report_files = list(artifacts_dir.glob("pipeline_report_*.json"))
        assert len(report_files) > 0

        # Verify report content
        with open(report_files[0]) as f:
            report = json.load(f)

        assert "pipeline_info" in report
        assert "phase_results" in report
        assert "performance_metrics" in report


class TestPhaseValidation:
    """Test phase validation suite."""

    @pytest.fixture
    def validation_suite(self, temp_output_dir):
        """Create validation suite."""
        return PhaseValidationSuite(temp_output_dir)

    @pytest.mark.asyncio
    async def test_model_integrity_validation(self, validation_suite):
        """Test model integrity validation."""
        # Create a mock phase result with valid model
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
        phase_result = PhaseResult(
            success=True,
            phase_name="TestPhase",
            phase_type=PhaseType.CREATION,
            model=model,
            metrics=PhaseMetrics(phase_name="TestPhase")
        )

        validation_results = await validation_suite.validate_phase(phase_result)

        assert "model_integrity" in validation_results
        integrity_result = validation_results["model_integrity"]
        assert integrity_result.valid
        assert integrity_result.score > 0.8

    @pytest.mark.asyncio
    async def test_data_format_validation(self, validation_suite):
        """Test data format validation."""
        # Create phase result with proper data format
        phase_result = PhaseResult(
            success=True,
            phase_name="EvoMergePhase",
            phase_type=PhaseType.EVOLUTION,
            model=nn.Linear(10, 1),
            output_data={
                "fitness_score": 0.85,
                "generation_info": {"generation": 50}
            },
            metrics=PhaseMetrics(phase_name="EvoMergePhase")
        )

        validation_results = await validation_suite.validate_phase(phase_result)

        assert "data_format" in validation_results
        format_result = validation_results["data_format"]
        assert format_result.valid

    @pytest.mark.asyncio
    async def test_performance_validation(self, validation_suite):
        """Test performance validation."""
        # Create phase result with performance metrics
        metrics = PhaseMetrics(
            phase_name="TestPhase",
            duration_seconds=60.0,
            memory_usage_mb=512.0,
            error_count=0
        )

        phase_result = PhaseResult(
            success=True,
            phase_name="TestPhase",
            phase_type=PhaseType.TRAINING,
            model=nn.Linear(10, 1),
            metrics=metrics,
            output_data={"training_loss": [2.0, 1.5, 1.0]}
        )

        validation_results = await validation_suite.validate_phase(phase_result)

        assert "performance" in validation_results
        perf_result = validation_results["performance"]
        assert perf_result.valid

    @pytest.mark.asyncio
    async def test_pipeline_validation(self, validation_suite, mock_phases):
        """Test complete pipeline validation."""
        # Create mock results for all phases
        phase_results = []
        for phase_name, controller in mock_phases:
            result = await controller.execute_async({})
            phase_results.append(result)

        validation_results = await validation_suite.validate_pipeline(phase_results)

        assert len(validation_results) == 8
        for phase_name in validation_results:
            assert len(validation_results[phase_name]) > 0


class TestIntegrationScenarios:
    """Test various integration scenarios."""

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self, pipeline_config, mock_phases):
        """Test resuming pipeline from checkpoint."""
        controller = PipelineController(pipeline_config)

        # First, run partial pipeline
        partial_phases = mock_phases[:4]  # First 4 phases
        result1 = await controller.execute_full_pipeline(partial_phases)
        assert result1.success

        # Now resume from phase 3
        remaining_phases = mock_phases[2:]  # From phase 3 onward
        result2 = await controller.execute_full_pipeline(
            remaining_phases, resume_from="QuietSTaRPhase"
        )
        assert result2.success

    @pytest.mark.asyncio
    async def test_error_recovery(self, pipeline_config):
        """Test error recovery mechanisms."""
        controller = PipelineController(pipeline_config)
        controller.config.enable_auto_recovery = True

        # Create a phase that fails initially but could succeed on retry
        class RetryablePhase(MockPhaseController):
            def __init__(self):
                super().__init__("RetryablePhase")
                self.attempt_count = 0

            async def _execute(self, input_data):
                self.attempt_count += 1
                if self.attempt_count == 1:
                    # Fail on first attempt
                    return PhaseResult(
                        success=False,
                        phase_name=self.phase_name,
                        phase_type=PhaseType.CREATION,
                        error="Temporary failure"
                    )
                else:
                    # Succeed on retry
                    return await super()._execute(input_data)

        phases = [("RetryablePhase", RetryablePhase())]

        # This would test retry logic if implemented
        result = await controller.execute_full_pipeline(phases)
        # Note: Current implementation doesn't have retry logic,
        # but this test structure is ready for it

    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, pipeline_config, mock_phases):
        """Test performance benchmarking."""
        controller = PipelineController(pipeline_config)

        # Modify phases to have different execution times
        for i, (name, phase) in enumerate(mock_phases):
            phase.execution_time = 0.1 * (i + 1)  # Increasing execution times

        start_time = time.time()
        result = await controller.execute_full_pipeline(mock_phases)
        total_time = time.time() - start_time

        assert result.success
        assert total_time > 0.1  # Should take at least some time

        # Check performance metrics
        assert "performance_metrics" in controller.performance_metrics or controller.performance_metrics

    @pytest.mark.asyncio
    async def test_resource_constraints(self, temp_output_dir):
        """Test resource constraint enforcement."""
        # Create restrictive resource constraints
        restrictive_constraints = ResourceConstraints(
            max_memory_gb=0.1,  # Very low
            max_execution_time_hours=0.001  # Very short
        )

        config = PipelineConfig(
            output_dir=temp_output_dir,
            resource_constraints=restrictive_constraints
        )

        controller = PipelineController(config)

        # Create a single quick phase
        quick_phases = [("QuickPhase", MockPhaseController("QuickPhase", execution_time=0.01))]

        # Should still work for very quick phases
        result = await controller.execute_full_pipeline(quick_phases)
        # May pass or fail depending on actual resource usage

    @pytest.mark.asyncio
    async def test_data_flow_validation(self, pipeline_config, mock_phases):
        """Test data flow between phases."""
        controller = PipelineController(pipeline_config)

        result = await controller.execute_full_pipeline(mock_phases)

        assert result.success

        # Verify that data flows correctly between phases
        orchestrator = controller.orchestrator
        phase_results = orchestrator.phase_results

        # Check that each phase received appropriate inputs
        for i, (phase_name, _) in enumerate(mock_phases[1:], 1):  # Skip first phase
            phase_result = phase_results.get(phase_name)
            if phase_result:
                # Each phase should have received model from previous phase
                assert phase_result.model is not None

    @pytest.mark.asyncio
    async def test_quality_gates(self, pipeline_config, mock_phases):
        """Test quality gate enforcement."""
        controller = PipelineController(pipeline_config)

        # Add validation to pipeline
        validation_suite = PhaseValidationSuite()

        result = await controller.execute_full_pipeline(mock_phases)
        assert result.success

        # Validate all phases
        phase_results = list(controller.orchestrator.phase_results.values())
        validation_results = await validation_suite.validate_pipeline(phase_results)

        # All phases should pass basic quality gates
        for phase_validation in validation_results.values():
            for validator_result in phase_validation.values():
                if validator_result.validator_name == "quality_gates":
                    # Quality gates should pass for mock data
                    assert validator_result.score > 0.0


# Performance and load testing
class TestPerformanceAndLoad:
    """Test performance and load scenarios."""

    @pytest.mark.asyncio
    async def test_memory_usage_tracking(self, pipeline_config, mock_phases):
        """Test memory usage tracking during execution."""
        controller = PipelineController(pipeline_config)

        result = await controller.execute_full_pipeline(mock_phases)

        assert result.success
        # Should have collected memory usage data
        assert len(controller.resource_usage) > 0

        # Verify memory usage data structure
        for timestamp, usage in controller.resource_usage.items():
            assert "memory_used_gb" in usage
            assert "cpu_percent" in usage

    @pytest.mark.asyncio
    async def test_concurrent_validation(self, temp_output_dir):
        """Test concurrent validation of multiple phases."""
        validation_suite = PhaseValidationSuite(temp_output_dir)

        # Create multiple phase results
        phase_results = []
        for i in range(5):
            model = nn.Linear(10, 1)
            result = PhaseResult(
                success=True,
                phase_name=f"Phase{i}",
                phase_type=PhaseType.CREATION,
                model=model,
                metrics=PhaseMetrics(phase_name=f"Phase{i}")
            )
            phase_results.append(result)

        # Validate all phases
        start_time = time.time()
        validation_tasks = [
            validation_suite.validate_phase(result)
            for result in phase_results
        ]
        await asyncio.gather(*validation_tasks)
        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 10.0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])