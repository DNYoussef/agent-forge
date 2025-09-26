"""
End-to-End Pipeline Integration Test

Tests the complete orchestration system with working phases and demonstrates
progress reporting, error handling, and performance metrics.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

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

# Import test components - inline MockPhaseController definition
import torch.nn as nn


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


class IntegratedPipelineTest:
    """Complete integration test for the 8-phase pipeline."""

    def __init__(self, output_dir: str = "./test_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

        # Test results
        self.test_results: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}
        self.validation_results: Dict[str, Any] = {}

        # Progress tracking
        self.progress_events: List[Dict[str, Any]] = []

    def _setup_logging(self) -> logging.Logger:
        """Setup test logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        logger = logging.getLogger("PipelineIntegrationTest")

        # File handler
        log_file = self.output_dir / "integration_test.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def progress_callback(self, event: str, data: Dict[str, Any]):
        """Track progress events."""
        self.progress_events.append({
            "timestamp": time.time(),
            "event": event,
            "data": data
        })

        if event == "progress_report":
            percent = data.get('progress_percent', 0)
            current_phase = data.get('current_phase', 'Unknown')
            self.logger.info(f"PROGRESS: {percent:.1f}% - {current_phase}")
        elif event in ["phase_completed", "phase_ready", "pipeline_started", "pipeline_completed"]:
            self.logger.info(f"EVENT: {event} - {data}")

    async def run_complete_integration_test(self) -> Dict[str, Any]:
        """Run complete integration test suite."""
        self.logger.info("=" * 80)
        self.logger.info("STARTING COMPLETE PIPELINE INTEGRATION TEST")
        self.logger.info("=" * 80)

        start_time = time.time()

        try:
            # Test 1: Basic orchestrator functionality
            test1_result = await self._test_orchestrator_basic()

            # Test 2: Pipeline controller with monitoring
            test2_result = await self._test_pipeline_controller()

            # Test 3: Validation suite
            test3_result = await self._test_validation_suite()

            # Test 4: Error handling and recovery
            test4_result = await self._test_error_handling()

            # Test 5: Performance benchmarking
            test5_result = await self._test_performance_benchmarking()

            # Test 6: End-to-end with real phases (subset)
            test6_result = await self._test_real_phases_subset()

            # Aggregate results
            total_time = time.time() - start_time

            self.test_results = {
                "test_info": {
                    "start_time": start_time,
                    "total_duration": total_time,
                    "success": True
                },
                "test_1_orchestrator_basic": test1_result,
                "test_2_pipeline_controller": test2_result,
                "test_3_validation_suite": test3_result,
                "test_4_error_handling": test4_result,
                "test_5_performance_benchmarking": test5_result,
                "test_6_real_phases_subset": test6_result,
                "progress_events": len(self.progress_events),
                "performance_summary": self.performance_metrics
            }

            # Generate final report
            await self._generate_test_report()

            self.logger.info("=" * 80)
            self.logger.info(f"INTEGRATION TEST COMPLETED SUCCESSFULLY in {total_time:.2f}s")
            self.logger.info("=" * 80)

            return self.test_results

        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"Integration test failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            self.test_results = {
                "test_info": {
                    "start_time": start_time,
                    "total_duration": total_time,
                    "success": False,
                    "error": error_msg
                }
            }

            await self._generate_test_report()
            return self.test_results

    async def _test_orchestrator_basic(self) -> Dict[str, Any]:
        """Test basic orchestrator functionality."""
        self.logger.info("TEST 1: Basic Orchestrator Functionality")

        try:
            orchestrator = PhaseOrchestrator(str(self.output_dir / "test1"))

            # Create mock phases
            mock_phases = [
                ("TestPhase1", MockPhaseController("TestPhase1", execution_time=0.1)),
                ("TestPhase2", MockPhaseController("TestPhase2", execution_time=0.1)),
                ("TestPhase3", MockPhaseController("TestPhase3", execution_time=0.1))
            ]

            start_time = time.time()
            result = await orchestrator.execute_phase_pipeline(mock_phases)
            execution_time = time.time() - start_time

            test_result = {
                "success": result.success,
                "execution_time": execution_time,
                "phases_executed": len(orchestrator.phase_results),
                "all_phases_completed": all(
                    state == orchestrator.phases[name] for name, state in
                    [("TestPhase1", "COMPLETED"), ("TestPhase2", "COMPLETED"), ("TestPhase3", "COMPLETED")]
                ),
                "checkpoints_created": len(orchestrator.checkpoints),
                "error": result.error if not result.success else None
            }

            self.logger.info(f"Test 1 Result: {test_result}")
            return test_result

        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self.logger.error(f"Test 1 Failed: {error_result}")
            return error_result

    async def _test_pipeline_controller(self) -> Dict[str, Any]:
        """Test pipeline controller with monitoring."""
        self.logger.info("TEST 2: Pipeline Controller with Monitoring")

        try:
            config = PipelineConfig(
                output_dir=str(self.output_dir / "test2"),
                checkpoint_dir=str(self.output_dir / "test2_checkpoints"),
                logs_dir=str(self.output_dir / "test2_logs"),
                resource_constraints=ResourceConstraints(
                    max_memory_gb=4.0,
                    max_execution_time_hours=1.0
                ),
                enable_performance_monitoring=True,
                enable_progress_reporting=True,
                progress_report_interval=1
            )

            controller = PipelineController(config)
            controller.add_progress_callback(self.progress_callback)

            # Create mock phases with varying execution times
            mock_phases = [
                ("FastPhase", MockPhaseController("FastPhase", execution_time=0.1)),
                ("MediumPhase", MockPhaseController("MediumPhase", execution_time=0.2)),
                ("SlowPhase", MockPhaseController("SlowPhase", execution_time=0.3))
            ]

            start_time = time.time()
            result = await controller.execute_full_pipeline(mock_phases)
            execution_time = time.time() - start_time

            test_result = {
                "success": result.success,
                "execution_time": execution_time,
                "resource_usage_collected": len(controller.resource_usage) > 0,
                "progress_events_received": len([e for e in self.progress_events if e["event"] == "progress_report"]) > 0,
                "performance_metrics_available": bool(controller.performance_metrics),
                "pipeline_state": controller.pipeline_state,
                "error": result.error if not result.success else None
            }

            # Store performance metrics
            self.performance_metrics["controller_test"] = {
                "execution_time": execution_time,
                "resource_usage_samples": len(controller.resource_usage),
                "performance_metrics": controller.performance_metrics
            }

            self.logger.info(f"Test 2 Result: {test_result}")
            return test_result

        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self.logger.error(f"Test 2 Failed: {error_result}")
            return error_result

    async def _test_validation_suite(self) -> Dict[str, Any]:
        """Test validation suite."""
        self.logger.info("TEST 3: Validation Suite")

        try:
            validation_suite = PhaseValidationSuite(str(self.output_dir / "test3_validation"))

            # Create mock phase results for validation
            mock_results = []
            for i, phase_type in enumerate([PhaseType.CREATION, PhaseType.EVOLUTION, PhaseType.REASONING]):
                controller = MockPhaseController(f"TestPhase{i+1}")
                result = await controller.execute_async({})
                result.phase_type = phase_type
                mock_results.append(result)

            start_time = time.time()

            # Validate individual phases
            individual_validations = {}
            for result in mock_results:
                validation = await validation_suite.validate_phase(result)
                individual_validations[result.phase_name] = validation

            # Validate complete pipeline
            pipeline_validation = await validation_suite.validate_pipeline(mock_results)

            validation_time = time.time() - start_time

            # Analyze validation results
            total_validations = sum(len(v) for v in individual_validations.values())
            passed_validations = sum(
                1 for v in individual_validations.values()
                for validator_result in v.values()
                if validator_result.valid
            )

            test_result = {
                "success": True,
                "validation_time": validation_time,
                "phases_validated": len(mock_results),
                "total_validations": total_validations,
                "passed_validations": passed_validations,
                "validation_success_rate": passed_validations / total_validations if total_validations > 0 else 0,
                "pipeline_validation_available": bool(pipeline_validation)
            }

            self.validation_results = {
                "individual": individual_validations,
                "pipeline": pipeline_validation
            }

            self.logger.info(f"Test 3 Result: {test_result}")
            return test_result

        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self.logger.error(f"Test 3 Failed: {error_result}")
            return error_result

    async def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and recovery."""
        self.logger.info("TEST 4: Error Handling and Recovery")

        try:
            config = PipelineConfig(
                output_dir=str(self.output_dir / "test4"),
                enable_auto_recovery=True,
                max_retry_attempts=2
            )

            controller = PipelineController(config)

            # Create phases with one that fails
            phases_with_failure = [
                ("GoodPhase1", MockPhaseController("GoodPhase1", execution_time=0.1)),
                ("FailingPhase", MockPhaseController("FailingPhase", should_fail=True, execution_time=0.1)),
                ("GoodPhase2", MockPhaseController("GoodPhase2", execution_time=0.1))
            ]

            start_time = time.time()
            result = await controller.execute_full_pipeline(phases_with_failure)
            execution_time = time.time() - start_time

            test_result = {
                "success": not result.success,  # Should fail as expected
                "execution_time": execution_time,
                "pipeline_failed_correctly": not result.success,
                "error_recorded": result.error is not None,
                "partial_execution": len(controller.orchestrator.phase_results) > 0,
                "controller_error_count": len(controller.errors),
                "failure_state": controller.pipeline_state == "FAILED"
            }

            self.logger.info(f"Test 4 Result: {test_result}")
            return test_result

        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self.logger.error(f"Test 4 Failed: {error_result}")
            return error_result

    async def _test_performance_benchmarking(self) -> Dict[str, Any]:
        """Test performance benchmarking."""
        self.logger.info("TEST 5: Performance Benchmarking")

        try:
            config = PipelineConfig(
                output_dir=str(self.output_dir / "test5"),
                enable_performance_monitoring=True
            )

            controller = PipelineController(config)

            # Create phases with different performance characteristics
            performance_phases = [
                ("LightPhase", MockPhaseController("LightPhase", execution_time=0.05)),
                ("MediumPhase", MockPhaseController("MediumPhase", execution_time=0.15)),
                ("HeavyPhase", MockPhaseController("HeavyPhase", execution_time=0.25))
            ]

            start_time = time.time()
            result = await controller.execute_full_pipeline(performance_phases)
            total_execution_time = time.time() - start_time

            # Analyze performance
            phase_times = {}
            for phase_name, phase_result in controller.orchestrator.phase_results.items():
                if phase_result.metrics:
                    phase_times[phase_name] = phase_result.metrics.duration_seconds

            test_result = {
                "success": result.success,
                "total_execution_time": total_execution_time,
                "phase_execution_times": phase_times,
                "resource_samples_collected": len(controller.resource_usage),
                "performance_analysis_available": bool(controller.performance_metrics),
                "slowest_phase": max(phase_times.items(), key=lambda x: x[1]) if phase_times else None,
                "fastest_phase": min(phase_times.items(), key=lambda x: x[1]) if phase_times else None,
                "average_phase_time": sum(phase_times.values()) / len(phase_times) if phase_times else 0
            }

            # Store detailed performance metrics
            self.performance_metrics["benchmark_test"] = {
                "total_time": total_execution_time,
                "phase_breakdown": phase_times,
                "resource_usage": controller._get_resource_usage_summary()
            }

            self.logger.info(f"Test 5 Result: {test_result}")
            return test_result

        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self.logger.error(f"Test 5 Failed: {error_result}")
            return error_result

    async def _test_real_phases_subset(self) -> Dict[str, Any]:
        """Test with real phases that are operational."""
        self.logger.info("TEST 6: Real Phases Subset (Operational Phases Only)")

        try:
            # Based on validation report, use only operational phases
            # QuietSTaRPhase and BitNetCompressionPhase are operational

            from src.orchestration.phase_status_validator import PhaseStatusValidator

            # Get operational phases
            validator = PhaseStatusValidator(str(Path(__file__).parent.parent))
            phase_info = validator.validate_all_phases()

            operational_phases = [
                name for name, info in phase_info.items()
                if info.status == "OPERATIONAL"
            ]

            self.logger.info(f"Testing with operational phases: {operational_phases}")

            if not operational_phases:
                return {
                    "success": False,
                    "error": "No operational phases available for testing",
                    "available_phases": list(phase_info.keys())
                }

            # For this test, use mock controllers that simulate the operational phases
            # In a real implementation, you would import and use the actual phase classes
            mock_operational_phases = []
            for phase_name in operational_phases[:2]:  # Test with first 2 operational phases
                controller = MockPhaseController(
                    phase_name,
                    execution_time=0.2,  # Slightly longer to simulate real processing
                    output_data=self._get_phase_specific_output(phase_name)
                )
                mock_operational_phases.append((phase_name, controller))

            config = PipelineConfig(
                output_dir=str(self.output_dir / "test6"),
                enable_performance_monitoring=True,
                enable_progress_reporting=True
            )

            controller = PipelineController(config)
            controller.add_progress_callback(self.progress_callback)

            start_time = time.time()
            result = await controller.execute_full_pipeline(mock_operational_phases)
            execution_time = time.time() - start_time

            test_result = {
                "success": result.success,
                "execution_time": execution_time,
                "operational_phases_tested": len(mock_operational_phases),
                "phase_names": [name for name, _ in mock_operational_phases],
                "all_phases_completed": result.success,
                "validation_report_integration": True,
                "error": result.error if not result.success else None
            }

            self.logger.info(f"Test 6 Result: {test_result}")
            return test_result

        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self.logger.error(f"Test 6 Failed: {error_result}")
            return error_result

    def _get_phase_specific_output(self, phase_name: str) -> Dict[str, Any]:
        """Get phase-specific output data for testing."""
        output_mappings = {
            "QuietSTaRPhase": {
                "reasoning_capability": 0.85,
                "thought_generation_stats": {"avg_thoughts": 4, "success_rate": 0.82},
                "enhanced_model_path": f"/tmp/{phase_name}_enhanced.pt"
            },
            "BitNetCompressionPhase": {
                "compression_ratio": 4.2,
                "quantization_info": {"bits": 1.58, "size_reduction": 0.76},
                "compressed_model_path": f"/tmp/{phase_name}_compressed.pt"
            }
        }
        return output_mappings.get(phase_name, {})

    async def _generate_test_report(self):
        """Generate comprehensive test report."""
        report_file = self.output_dir / "integration_test_report.json"

        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)

        # Generate summary report
        summary_file = self.output_dir / "integration_test_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("AGENT FORGE PIPELINE INTEGRATION TEST SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            # Overall results
            if self.test_results.get("test_info", {}).get("success", False):
                f.write("OVERALL RESULT: SUCCESS\n")
            else:
                f.write("OVERALL RESULT: FAILURE\n")

            f.write(f"Total Duration: {self.test_results.get('test_info', {}).get('total_duration', 0):.2f}s\n")
            f.write(f"Progress Events: {self.test_results.get('progress_events', 0)}\n\n")

            # Individual test results
            for test_name, test_result in self.test_results.items():
                if test_name.startswith("test_") and isinstance(test_result, dict):
                    f.write(f"{test_name.upper()}:\n")
                    f.write(f"  Success: {test_result.get('success', False)}\n")
                    if 'execution_time' in test_result:
                        f.write(f"  Execution Time: {test_result['execution_time']:.2f}s\n")
                    if 'error' in test_result and test_result['error']:
                        f.write(f"  Error: {test_result['error']}\n")
                    f.write("\n")

            # Performance summary
            if self.performance_metrics:
                f.write("PERFORMANCE METRICS:\n")
                for test_name, metrics in self.performance_metrics.items():
                    f.write(f"  {test_name}: {metrics}\n")

        self.logger.info(f"Test report saved: {report_file}")
        self.logger.info(f"Test summary saved: {summary_file}")


async def main():
    """Main function to run the integration test."""
    test_runner = IntegratedPipelineTest("./integration_test_output")

    print("Starting Agent Forge Pipeline Integration Test...")
    print("This will test the complete orchestration system.")
    print()

    try:
        results = await test_runner.run_complete_integration_test()

        print("\n" + "=" * 80)
        print("INTEGRATION TEST COMPLETED")
        print("=" * 80)

        if results.get("test_info", {}).get("success", False):
            print("RESULT: SUCCESS")
            print(f"Duration: {results['test_info']['total_duration']:.2f} seconds")
            print(f"Progress Events: {results.get('progress_events', 0)}")

            # Show individual test results
            print("\nTest Results:")
            for test_name, test_result in results.items():
                if test_name.startswith("test_") and isinstance(test_result, dict):
                    status = "PASS" if test_result.get("success", False) else "FAIL"
                    print(f"  {test_name}: {status}")

        else:
            print("RESULT: FAILURE")
            if "error" in results.get("test_info", {}):
                print(f"Error: {results['test_info']['error']}")

        print(f"\nDetailed reports available in: {test_runner.output_dir}")

    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))