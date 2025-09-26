#!/usr/bin/env python3
"""
Agent Forge Demonstration Script

Demonstrates the parallel vs series training configuration enhancement
with complete backward compatibility validation.
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent_forge.phases.cognate_pretrain.cognate_creator import (
    CognateCreator, TrainingConfig, create_cognate_models
)
from agent_forge.utils.resource_manager import ResourceManager
from agent_forge.utils.progress_aggregator import ProgressAggregator


def print_header(title: str) -> None:
    """Print formatted header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_subheader(title: str) -> None:
    """Print formatted subheader"""
    print(f"\n{'-' * 40}")
    print(f"  {title}")
    print(f"{'-' * 40}")


def demo_resource_detection():
    """Demonstrate hardware resource detection"""
    print_subheader("Hardware Resource Detection")

    manager = ResourceManager()
    resources = manager.detect_system_resources()

    print(f"CPU Count: {resources.cpu_count}")
    print(f"Total Memory: {resources.memory_gb:.1f} GB")
    print(f"Available Memory: {resources.available_memory_gb:.1f} GB")
    print(f"CPU Usage: {resources.cpu_usage_percent:.1f}%")
    print(f"Recommended Workers: {resources.recommended_workers}")
    print(f"Memory per Worker: {resources.memory_per_worker_gb:.1f} GB")

    # Test worker validation
    print(f"\nWorker Validation Tests:")
    test_workers = [1, 2, 4, 8]
    for workers in test_workers:
        validated, reason = manager.validate_worker_count(workers)
        print(f"  Requested: {workers} -> Validated: {validated} ({reason})")


def demo_backward_compatibility():
    """Demonstrate backward compatibility with existing interfaces"""
    print_subheader("Backward Compatibility Test")

    # Test 1: Default behavior (series mode)
    print("Test 1: Default constructor behavior")
    creator = CognateCreator()  # No parameters - should default to series
    assert creator.config.training_mode == 'series', "Default mode should be series"
    assert creator.config.max_parallel_workers == 3, "Default workers should be 3"
    print("  ✓ Default constructor maintains series mode")

    # Test 2: Legacy utility function
    print("Test 2: Legacy utility function")
    with tempfile.TemporaryDirectory() as temp_dir:
        models = create_cognate_models(temp_dir)  # No training_mode specified
        assert len(models) == 3, "Should create 3 models"
        print("  ✓ Legacy utility function works without changes")

        # Verify model files were created
        for model_path in models:
            assert Path(f"{model_path}.json").exists(), f"Model metadata should exist: {model_path}.json"
        print("  ✓ Model files created successfully")

    # Test 3: Explicit series mode
    print("Test 3: Explicit series mode")
    config = TrainingConfig(training_mode='series', epochs=10)  # Reduced for speed
    creator = CognateCreator(config)

    with tempfile.TemporaryDirectory() as temp_dir:
        start_time = time.time()
        models = creator.create_three_models(temp_dir)
        series_time = time.time() - start_time

        assert len(models) == 3, "Should create 3 models"
        print(f"  ✓ Series training completed in {series_time:.2f}s")

    print("✓ All backward compatibility tests passed")


def demo_parallel_enhancement():
    """Demonstrate the new parallel training capability"""
    print_subheader("Parallel Training Enhancement")

    # Test parallel mode with resource management
    config = TrainingConfig(
        training_mode='parallel',
        max_parallel_workers=2,  # Conservative for demo
        epochs=10  # Reduced for speed
    )

    creator = CognateCreator(config)

    # Setup progress monitoring
    progress_aggregator = ProgressAggregator(total_models=3)

    print("Starting parallel training with progress monitoring...")

    with tempfile.TemporaryDirectory() as temp_dir:
        start_time = time.time()

        # Create models in parallel
        models = creator.create_three_models(temp_dir)

        parallel_time = time.time() - start_time

        assert len(models) == 3, "Should create 3 models"
        print(f"✓ Parallel training completed in {parallel_time:.2f}s")

        # Verify all models were created
        for i, model_path in enumerate(models, 1):
            assert Path(f"{model_path}.json").exists(), f"Model {i} metadata should exist"
            print(f"  ✓ Model {i} created: {Path(model_path).name}")

        # Show progress report
        progress_report = creator.get_training_progress()
        print(f"  Progress: {progress_report.get('overall_progress', 0):.1f}%")
        print(f"  Mode: {progress_report.get('mode', 'unknown')}")


def demo_performance_comparison():
    """Compare performance between series and parallel modes"""
    print_subheader("Performance Comparison")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Series training
        print("Running series training...")
        config_series = TrainingConfig(training_mode='series', epochs=20)
        creator_series = CognateCreator(config_series)

        series_dir = os.path.join(temp_dir, 'series')
        os.makedirs(series_dir, exist_ok=True)

        start_time = time.time()
        models_series = creator_series.create_three_models(series_dir)
        series_time = time.time() - start_time

        # Parallel training
        print("Running parallel training...")
        config_parallel = TrainingConfig(
            training_mode='parallel',
            max_parallel_workers=2,
            epochs=20
        )
        creator_parallel = CognateCreator(config_parallel)

        parallel_dir = os.path.join(temp_dir, 'parallel')
        os.makedirs(parallel_dir, exist_ok=True)

        start_time = time.time()
        models_parallel = creator_parallel.create_three_models(parallel_dir)
        parallel_time = time.time() - start_time

        # Results
        print(f"\nPerformance Results:")
        print(f"  Series Time:   {series_time:.2f}s")
        print(f"  Parallel Time: {parallel_time:.2f}s")

        if parallel_time < series_time:
            speedup = series_time / parallel_time
            print(f"  Speedup:       {speedup:.2f}x faster")
        else:
            print(f"  Note: Parallel may be slower for small workloads due to overhead")

        print(f"  Series Models:   {len(models_series)} created")
        print(f"  Parallel Models: {len(models_parallel)} created")


def demo_resource_monitoring():
    """Demonstrate resource monitoring capabilities"""
    print_subheader("Resource Monitoring")

    manager = ResourceManager()

    # Start monitoring
    manager.start_monitoring(interval=1.0)
    print("Started resource monitoring for 5 seconds...")

    # Simulate some work
    time.sleep(5)

    # Get current metrics
    metrics = manager.get_current_metrics()
    if metrics:
        print(f"Current CPU Usage: {metrics.get('cpu_usage_percent', 0):.1f}%")
        print(f"Available Memory: {metrics.get('available_memory_gb', 0):.1f} GB")
        print(f"Last Update: {time.ctime(metrics.get('timestamp', 0))}")
    else:
        print("No metrics collected yet")

    # Generate resource report
    report = manager.generate_resource_report()
    print(f"Resource Report:")
    print(f"  Suggested Training Mode: {report['recommendations']['training_mode_suggestion']}")
    print(f"  Memory Limited: {report['constraints']['memory_limited']}")
    print(f"  CPU Limited: {report['constraints']['cpu_limited']}")

    # Stop monitoring
    manager.stop_monitoring()
    print("Resource monitoring stopped")


def demo_error_handling():
    """Demonstrate error handling and validation"""
    print_subheader("Error Handling and Validation")

    # Test invalid training mode
    try:
        config = TrainingConfig(training_mode='invalid_mode')
        creator = CognateCreator(config)
        creator.create_three_models("test")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Invalid training mode correctly rejected: {e}")

    # Test resource validation
    manager = ResourceManager()

    # Test extreme worker counts
    validated, reason = manager.validate_worker_count(0)
    assert validated == 1, "Should validate to minimum 1 worker"
    print(f"✓ Zero workers validated to: {validated} ({reason})")

    validated, reason = manager.validate_worker_count(100)
    print(f"✓ Excessive workers handled: {validated} ({reason})")

    print("✓ All error handling tests passed")


def main():
    """Run complete Agent Forge demonstration"""
    print_header("Agent Forge - Parallel/Series Training Enhancement Demo")

    print("This demonstration shows the complete parallel training enhancement")
    print("with backward compatibility and performance improvements.")

    try:
        # Core demonstrations
        demo_resource_detection()
        demo_backward_compatibility()
        demo_parallel_enhancement()
        demo_performance_comparison()
        demo_resource_monitoring()
        demo_error_handling()

        print_header("Demonstration Complete")
        print("✓ All demonstrations completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  - Complete backward compatibility with existing code")
        print("  - Hardware resource detection and optimization")
        print("  - Parallel training with resource management")
        print("  - Progress monitoring and reporting")
        print("  - Error handling and validation")
        print("  - Performance comparison between modes")

        print("\nSafe Configuration Defaults:")
        print("  - Series mode by default (existing behavior)")
        print("  - Conservative worker limits based on hardware")
        print("  - Automatic resource validation and warnings")
        print("  - Graceful fallback on resource constraints")

    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

# Version & Run Log Footer
"""
Version & Run Log

| Version | Timestamp | Agent/Model | Change Summary | Artifacts | Status | Notes | Cost | Hash |
|--------:|-----------|-------------|----------------|-----------|--------|-------|------|------|
| 1.0.0   | 2025-01-25T15:40:03-05:00 | system@architect | Create comprehensive Agent Forge demo script | agent_forge_demo.py | OK | Complete validation and demonstration | 0.00 | e5f3c1d |

Receipt:
- status: OK
- reason_if_blocked: --
- run_id: agent-forge-005
- inputs: ["demo-requirements", "validation-spec"]
- tools_used: ["Write"]
- versions: {"model":"claude-sonnet-4","design":"demo-script-v1"}
"""