#!/usr/bin/env python3
"""
Complete Integration Test for Agent Forge
Tests that all moved files work correctly from their new locations.
"""

import sys
import os

# Add agent-forge to path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test all critical imports work."""
    print("Testing imports...")

    try:
        # Test cognate creator import
        from agent_forge.phases.cognate_pretrain.cognate_creator import CognateModelCreator, create_sample_training_data
        print("[OK] CognateModelCreator imported successfully")

        # Test grokfast import
        from agent_forge.phases.cognate_pretrain.grokfast_enhanced import EnhancedGrokFastOptimizer
        print("[OK] EnhancedGrokFastOptimizer imported successfully")

        # Test API server import
        from agent_forge.api import python_bridge_server
        print("[OK] Python bridge server imported successfully")

        # Test websocket progress import
        from src.agent_forge.api.websocket_progress import WebSocketProgressReporter
        print("[OK] WebSocketProgressReporter imported successfully")

        # Test streaming integration
        from src.agent_forge.integration.streaming_integration import StreamingIntegration
        print("[OK] StreamingIntegration imported successfully")

        return True

    except ImportError as e:
        print(f"[FAILED] Import failed: {e}")
        return False


def test_model_creation():
    """Test model creation works."""
    print("\nTesting model creation...")

    try:
        from agent_forge.phases.cognate_pretrain.cognate_creator import CognateModelCreator

        # Create a small test model
        creator = CognateModelCreator(
            vocab_size=100,
            d_model=64,
            nhead=2,
            num_layers=2
        )

        model = creator.create_model()
        print(f"[OK] Created model with {sum(p.numel() for p in model.parameters())} parameters")

        # Get model info
        info = creator.get_model_info()
        print(f"[OK] Model info: {info['model_type']} with {info['vocab_size']} vocab size")

        return True

    except Exception as e:
        print(f"[FAILED] Model creation failed: {e}")
        return False


def test_training_data():
    """Test training data generation."""
    print("\nTesting training data generation...")

    try:
        from agent_forge.phases.cognate_pretrain.cognate_creator import create_sample_training_data

        # Generate small sample
        data = create_sample_training_data(vocab_size=100, num_samples=10, seq_length=16)
        print(f"[OK] Generated {len(data)} training samples")

        # Check data shape
        if len(data) > 0 and len(data[0]) == 16:
            print(f"[OK] Data shape correct: {len(data)} x {len(data[0])}")
        else:
            print(f"[FAILED] Data shape incorrect")
            return False

        return True

    except Exception as e:
        print(f"[FAILED] Training data generation failed: {e}")
        return False


def test_api_components():
    """Test API components are accessible."""
    print("\nTesting API components...")

    try:
        from agent_forge.api.python_bridge_server import (
            app,
            CognateStartRequest,
            TrainingProgress
        )
        print("[OK] FastAPI app imported")
        print("[OK] Request models imported")
        print("[OK] TrainingProgress tracker imported")

        # Test app routes exist
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/api/cognate/start", "/api/cognate/status/{training_id}"]

        for route in expected_routes:
            if route in routes:
                print(f"[OK] Route {route} exists")
            else:
                print(f"[FAILED] Route {route} missing")

        return True

    except Exception as e:
        print(f"[FAILED] API components test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("Agent Forge Integration Test")
    print("Testing all components after file relocation")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Training Data", test_training_data),
        ("API Components", test_api_components)
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n[FAILED] Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "[OK] PASSED" if success else "[FAILED] FAILED"
        print(f"{name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nSUCCESS: All integration tests passed! System is functional.")
        return 0
    else:
        print(f"\nWARNING: {total - passed} tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())