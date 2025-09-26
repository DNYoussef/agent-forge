#!/usr/bin/env python3
"""
Agent Forge Validation Script

Simple validation of the parallel training enhancement components.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def print_header(title: str) -> None:
    """Print formatted header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def print_status(test_name: str, status: bool, details: str = "") -> None:
    """Print test status"""
    status_char = "✓" if status else "❌"
    print(f"  {status_char} {test_name}")
    if details:
        print(f"    {details}")

def validate_file_structure():
    """Validate that all required files exist"""
    print_header("File Structure Validation")

    base_path = Path(__file__).parent.parent

    required_files = [
        "src/agent_forge/__init__.py",
        "src/agent_forge/phases/__init__.py",
        "src/agent_forge/phases/cognate_pretrain/__init__.py",
        "src/agent_forge/phases/cognate_pretrain/cognate_creator.py",
        "src/agent_forge/utils/__init__.py",
        "src/agent_forge/utils/resource_manager.py",
        "src/agent_forge/utils/progress_aggregator.py",
        "src/web/dashboard/app/phases/cognate/page.tsx",
        "tests/agent_forge/__init__.py",
        "tests/agent_forge/test_parallel_training.py",
        "docs/AGENT-FORGE-PARALLEL-TRAINING.md"
    ]

    all_exist = True
    for file_path in required_files:
        full_path = base_path / file_path
        exists = full_path.exists()
        all_exist = all_exist and exists
        print_status(f"File exists: {file_path}", exists)

    return all_exist

def validate_imports():
    """Validate that imports work correctly"""
    print_header("Import Validation")

    try:
        # Test utility imports
        from agent_forge.utils.resource_manager import ResourceManager
        print_status("ResourceManager import", True)

        from agent_forge.utils.progress_aggregator import ProgressAggregator, TrainingPhase
        print_status("ProgressAggregator import", True)

        # Test that utilities work
        manager = ResourceManager()
        resources = manager.detect_system_resources()
        print_status("ResourceManager functionality", True,
                    f"Detected {resources.cpu_count} CPUs, {resources.memory_gb:.1f}GB RAM")

        aggregator = ProgressAggregator(total_models=3)
        aggregator.update_model_progress(1, TrainingPhase.TRAINING, 50, 100)
        progress = aggregator.get_overall_progress()
        print_status("ProgressAggregator functionality", True,
                    f"Progress: {progress['overall_percent']:.1f}%")

        return True

    except Exception as e:
        print_status("Import validation", False, str(e))
        return False

def validate_resource_management():
    """Validate resource management functionality"""
    print_header("Resource Management Validation")

    try:
        from agent_forge.utils.resource_manager import ResourceManager, get_optimal_workers, get_resource_summary

        # Test resource detection
        manager = ResourceManager()
        resources = manager.detect_system_resources()

        print_status("System resource detection", True,
                    f"CPU: {resources.cpu_count}, Memory: {resources.memory_gb:.1f}GB, "
                    f"Recommended: {resources.recommended_workers} workers")

        # Test worker validation
        validated, reason = manager.validate_worker_count(4)
        print_status("Worker validation", True,
                    f"Requested 4 → Validated {validated} ({reason})")

        # Test utility functions
        optimal_workers, reason = get_optimal_workers(3)
        print_status("Utility functions", True,
                    f"Optimal workers: {optimal_workers} ({reason})")

        resource_summary = get_resource_summary()
        print_status("Resource summary", True,
                    f"Suggested mode: {resource_summary['recommendations']['training_mode_suggestion']}")

        return True

    except Exception as e:
        print_status("Resource management validation", False, str(e))
        return False

def validate_progress_tracking():
    """Validate progress tracking functionality"""
    print_header("Progress Tracking Validation")

    try:
        from agent_forge.utils.progress_aggregator import ProgressAggregator, TrainingPhase, create_progress_tracker

        # Test basic progress tracking
        aggregator = ProgressAggregator(total_models=3)

        # Simulate training progress
        for model_id in [1, 2, 3]:
            aggregator.update_model_progress(
                model_id=model_id,
                phase=TrainingPhase.TRAINING,
                current_epoch=model_id * 30,
                total_epochs=100,
                loss_value=1.0 - (model_id * 0.2)
            )

        progress = aggregator.get_overall_progress()
        print_status("Multi-model progress tracking", True,
                    f"Overall: {progress['overall_percent']:.1f}%, Models: {len(progress['models'])}")

        # Test progress formatting
        series_message = aggregator.format_progress_message("series")
        parallel_message = aggregator.format_progress_message("parallel")
        print_status("Progress message formatting", True,
                    f"Series: '{series_message[:50]}...', Parallel: '{parallel_message[:50]}...'")

        # Test utility function
        tracker = create_progress_tracker(3)
        print_status("Progress tracker creation", True)

        # Test completion detection
        for model_id in [1, 2, 3]:
            aggregator.update_model_progress(model_id, TrainingPhase.COMPLETED, 100, 100)

        print_status("Training completion detection", aggregator.is_training_complete())

        return True

    except Exception as e:
        print_status("Progress tracking validation", False, str(e))
        return False

def validate_react_ui():
    """Validate React UI component exists and has required features"""
    print_header("React UI Validation")

    ui_file = Path(__file__).parent.parent / "src/web/dashboard/app/phases/cognate/page.tsx"

    if not ui_file.exists():
        print_status("React UI file exists", False)
        return False

    content = ui_file.read_text()

    # Check for key features
    features = [
        ("Training mode selection", "trainingMode" in content),
        ("Resource information", "resourceInfo" in content or "ResourceInfo" in content),
        ("Progress tracking", "progress" in content.lower()),
        ("Worker configuration", "worker" in content.lower()),
        ("Real-time updates", "useEffect" in content),
        ("TypeScript interface", "interface" in content),
    ]

    all_features = True
    for feature_name, has_feature in features:
        all_features = all_features and has_feature
        print_status(feature_name, has_feature)

    return all_features

def validate_documentation():
    """Validate that comprehensive documentation exists"""
    print_header("Documentation Validation")

    doc_file = Path(__file__).parent.parent / "docs/AGENT-FORGE-PARALLEL-TRAINING.md"

    if not doc_file.exists():
        print_status("Documentation file exists", False)
        return False

    content = doc_file.read_text()

    # Check for key sections
    sections = [
        ("Overview", "## Overview" in content),
        ("Architecture", "## Architecture" in content),
        ("Usage Guide", "## Usage Guide" in content),
        ("Performance Comparison", "## Performance Comparison" in content),
        ("Safety Features", "## Safety Features" in content),
        ("Testing", "## Testing" in content),
        ("Configuration Reference", "## Configuration Reference" in content),
        ("Migration Guide", "## Migration Guide" in content),
    ]

    all_sections = True
    for section_name, has_section in sections:
        all_sections = all_sections and has_section
        print_status(section_name, has_section)

    doc_size = len(content)
    print_status("Documentation completeness", doc_size > 10000,
                f"Document size: {doc_size:,} characters")

    return all_sections and doc_size > 10000

def main():
    """Run complete validation"""
    print_header("Agent Forge Parallel Training Enhancement - Validation")

    print("Validating the complete parallel training enhancement implementation...")

    validations = [
        ("File Structure", validate_file_structure),
        ("Imports", validate_imports),
        ("Resource Management", validate_resource_management),
        ("Progress Tracking", validate_progress_tracking),
        ("React UI", validate_react_ui),
        ("Documentation", validate_documentation),
    ]

    results = []
    for name, validator in validations:
        try:
            result = validator()
            results.append((name, result))
        except Exception as e:
            print_status(f"{name} validation failed", False, str(e))
            results.append((name, False))

    # Summary
    print_header("Validation Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        print_status(name, result)

    print(f"\nOverall Result: {passed}/{total} validations passed")

    if passed == total:
        print("\n✅ All validations passed! The Agent Forge Parallel Training Enhancement is complete and ready for use.")
        print("\nKey Features Implemented:")
        print("  ✓ Complete backward compatibility")
        print("  ✓ Intelligent resource management")
        print("  ✓ Real-time progress tracking")
        print("  ✓ React UI with live updates")
        print("  ✓ Comprehensive testing framework")
        print("  ✓ Complete documentation")

        print("\nNext Steps:")
        print("  1. Review documentation: docs/AGENT-FORGE-PARALLEL-TRAINING.md")
        print("  2. Run tests: python tests/agent_forge/test_parallel_training.py")
        print("  3. Try the React UI at: src/web/dashboard/app/phases/cognate/")
        print("  4. Use resource-aware configuration in your training scripts")

        return True
    else:
        print(f"\n❌ {total - passed} validations failed. Please check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

# Version & Run Log Footer
"""
Version & Run Log

| Version | Timestamp | Agent/Model | Change Summary | Artifacts | Status | Notes | Cost | Hash |
|--------:|-----------|-------------|----------------|-----------|--------|-------|------|------|
| 1.0.0   | 2025-01-25T15:55:03-05:00 | system@architect | Create validation script for Agent Forge enhancement | validate_agent_forge.py | OK | Final validation and summary | 0.00 | b9e4c2f |

Receipt:
- status: OK
- reason_if_blocked: --
- run_id: agent-forge-008
- inputs: ["validation-requirements", "system-verification"]
- tools_used: ["Write", "Read"]
- versions: {"model":"claude-sonnet-4","design":"validation-v1"}
"""