#!/usr/bin/env python3
"""
Simple Agent Forge Test
Quick validation that the core components are working.
"""

import sys
import os
sys.path.insert(0, 'agent_forge')

def test_imports():
    """Test that all components can be imported."""
    print("Testing imports...")
    try:
        from agent_forge.phases.cognate_pretrain.grokfast_enhanced import EnhancedGrokFastOptimizer
        print("[OK] Grokfast optimizer imported successfully")

        from agent_forge.phases.cognate_pretrain.cognate_creator import CognateModelCreator
        print("[OK] Cognate creator imported successfully")

        from agent_forge.api.python_bridge_server import app
        print("[OK] Bridge server imported successfully")

        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def test_grokfast():
    """Test Grokfast optimizer."""
    print("\nTesting Grokfast optimizer...")
    try:
        import torch
        from agent_forge.phases.cognate_pretrain.grokfast_enhanced import EnhancedGrokFastOptimizer

        # Create simple model
        model = torch.nn.Linear(10, 5)
        base_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        grokfast_optimizer = EnhancedGrokFastOptimizer(base_optimizer)

        print("[OK] Grokfast optimizer created successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Grokfast test failed: {e}")
        return False

def test_cognate_creator():
    """Test Cognate model creator."""
    print("\nTesting Cognate model creator...")
    try:
        from agent_forge.phases.cognate_pretrain.cognate_creator import CognateModelCreator

        creator = CognateModelCreator(vocab_size=100, d_model=32, nhead=2, num_layers=1)
        model = creator.create_model()
        model_info = creator.get_model_info()

        print(f"[OK] Model created with {model_info['total_parameters']} parameters")
        return True
    except Exception as e:
        print(f"[FAIL] Cognate creator test failed: {e}")
        return False

def test_api_server():
    """Test API server."""
    print("\nTesting API server...")
    try:
        from agent_forge.api.python_bridge_server import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/")

        if response.status_code == 200:
            print("[OK] API server responding correctly")
            return True
        else:
            print(f"[FAIL] API server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] API server test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Agent Forge Simple Validation Test")
    print("=" * 40)

    tests = [
        test_imports,
        test_grokfast,
        test_cognate_creator,
        test_api_server
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("[SUCCESS] All tests passed! Agent Forge is working correctly.")
        return True
    else:
        print("[WARNING] Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)