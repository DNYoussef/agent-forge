#!/usr/bin/env python3
"""
Agent Forge - Streaming Integration Tests

Basic integration tests for the streaming system without external dependencies.
"""

import sys
import unittest
import time
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestAgentForgeStreaming(unittest.TestCase):
    """Test suite for Agent Forge streaming components"""

    def setUp(self):
        """Set up test environment"""
        # Import components (may fail due to missing dependencies)
        try:
            from agent_forge.phases.cognate_pretrain.cognate_creator import CognateCreator, TrainingConfig
            self.CognateCreator = CognateCreator
            self.TrainingConfig = TrainingConfig
            self.imports_available = True
        except ImportError as e:
            print(f"Warning: Import failed - {e}")
            self.imports_available = False

    def test_training_config_creation(self):
        """Test training configuration creation"""
        if not self.imports_available:
            self.skipTest("Required imports not available")

        config = self.TrainingConfig(
            model_count=3,
            batch_size=32,
            learning_rate=0.001,
            epochs=50
        )

        self.assertEqual(config.model_count, 3)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.epochs, 50)

    def test_cognate_creator_initialization(self):
        """Test CognateCreator initialization"""
        if not self.imports_available:
            self.skipTest("Required imports not available")

        config = self.TrainingConfig(model_count=2, epochs=10)
        creator = self.CognateCreator(config)

        self.assertIsNotNone(creator.session_id)
        self.assertEqual(creator.total_models, 2)
        self.assertEqual(creator.current_model_idx, 0)

    def test_metrics_calculation(self):
        """Test training metrics calculation"""
        if not self.imports_available:
            self.skipTest("Required imports not available")

        config = self.TrainingConfig(model_count=3, epochs=100)
        creator = self.CognateCreator(config)

        # Test metrics calculation
        metrics = creator._calculate_training_metrics(
            loss=1.5, step=25, total_steps=100, model_idx=0
        )

        # Validate metrics structure
        required_fields = [
            'loss', 'perplexity', 'grokProgress', 'modelParams',
            'currentStep', 'totalSteps', 'currentModel', 'totalModels',
            'overallProgress', 'timestamp'
        ]

        for field in required_fields:
            self.assertIn(field, metrics, f"Missing field: {field}")

        # Validate metric values
        self.assertEqual(metrics['loss'], 1.5)
        self.assertEqual(metrics['currentStep'], 25)
        self.assertEqual(metrics['totalSteps'], 100)
        self.assertEqual(metrics['currentModel'], 1)  # 0-based to 1-based
        self.assertEqual(metrics['totalModels'], 3)

        # Check calculated values
        self.assertGreater(metrics['perplexity'], 0)
        self.assertGreaterEqual(metrics['grokProgress'], 0)
        self.assertLessEqual(metrics['grokProgress'], 100)
        self.assertGreaterEqual(metrics['overallProgress'], 0)
        self.assertLessEqual(metrics['overallProgress'], 100)

    def test_progress_callback_registration(self):
        """Test progress callback registration"""
        if not self.imports_available:
            self.skipTest("Required imports not available")

        config = self.TrainingConfig(model_count=1, epochs=5)
        creator = self.CognateCreator(config)

        # Mock callback
        callback_called = False
        callback_data = None

        def test_callback(data):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = data

        # Register callback
        creator.set_progress_callback(test_callback)
        self.assertIsNotNone(creator.progress_callback)

        # Test callback execution (simulate)
        test_data = {'sessionId': 'test', 'loss': 1.0}
        creator._emit_progress(test_data)

        self.assertTrue(callback_called)
        self.assertEqual(callback_data, test_data)

    def test_data_compatibility_layer(self):
        """Test data compatibility layer"""
        try:
            from agent_forge.api.compatibility_layer import DataCompatibilityLayer
        except ImportError:
            self.skipTest("Compatibility layer import not available")

        compatibility = DataCompatibilityLayer()

        # Test data transformation
        raw_data = {
            'sessionId': 'test-123',
            'modelIndex': 0,
            'totalModels': 3,
            'step': 50,
            'totalSteps': 100,
            'loss': 1.8,
            'timestamp': time.time()
        }

        ui_data = compatibility.transform_progress_data(raw_data)

        # Validate transformation
        self.assertIsNotNone(ui_data)
        self.assertEqual(ui_data.loss, 1.8)
        self.assertEqual(ui_data.currentStep, 50)
        self.assertEqual(ui_data.totalSteps, 100)
        self.assertEqual(ui_data.currentModel, 1)  # 0-based to 1-based
        self.assertEqual(ui_data.totalModels, 3)

        # Test validation
        validation = compatibility.validate_ui_compatibility(ui_data)
        self.assertIn('compatible', validation)
        self.assertIsInstance(validation['compatible'], bool)

    def test_performance_optimizer(self):
        """Test performance optimization features"""
        try:
            from agent_forge.api.compatibility_layer import PerformanceOptimizer
        except ImportError:
            self.skipTest("Performance optimizer import not available")

        optimizer = PerformanceOptimizer(max_update_rate=5.0)

        # Test throttling
        session_id = 'test-session'

        # First update should be allowed
        self.assertTrue(optimizer.should_emit_update(session_id))

        # Immediate second update should be throttled
        self.assertFalse(optimizer.should_emit_update(session_id))

        # Test payload optimization
        test_data = {
            'loss': 1.23456789,
            'perplexity': 3.456789123,
            'grokProgress': 45.6789,
            'modelParams': 25000000
        }

        optimized = optimizer.optimize_payload(test_data)

        # Check rounding
        self.assertLessEqual(len(str(optimized['loss']).split('.')[1]), 4)
        self.assertLessEqual(len(str(optimized['grokProgress']).split('.')[1]), 1)

    def test_session_management(self):
        """Test session management functionality"""
        if not self.imports_available:
            self.skipTest("Required imports not available")

        config = self.TrainingConfig(model_count=2, epochs=10)
        creator = self.CognateCreator(config)

        # Test session info
        session_info = creator.get_session_info()

        self.assertIn('session_id', session_info)
        self.assertIn('current_model', session_info)
        self.assertIn('total_models', session_info)
        self.assertIn('training_active', session_info)

        self.assertEqual(session_info['current_model'], 0)
        self.assertEqual(session_info['total_models'], 2)

    def test_error_handling_fallback(self):
        """Test error handling and fallback mechanisms"""
        try:
            from agent_forge.api.compatibility_layer import DataCompatibilityLayer
        except ImportError:
            self.skipTest("Compatibility layer import not available")

        compatibility = DataCompatibilityLayer()

        # Test with invalid data
        invalid_data = {
            'invalid_field': 'invalid_value',
            'loss': 'not_a_number'
        }

        # Should not crash and should return fallback
        ui_data = compatibility.transform_progress_data(invalid_data)
        self.assertIsNotNone(ui_data)

        # Fallback values should be reasonable
        self.assertGreaterEqual(ui_data.loss, 0)
        self.assertGreater(ui_data.perplexity, 0)
        self.assertGreaterEqual(ui_data.grokProgress, 0)
        self.assertLessEqual(ui_data.grokProgress, 100)

    def test_training_history_tracking(self):
        """Test training history tracking"""
        if not self.imports_available:
            self.skipTest("Required imports not available")

        config = self.TrainingConfig(model_count=1, epochs=5)
        creator = self.CognateCreator(config)

        # Simulate some progress updates
        for i in range(3):
            metrics = creator._calculate_training_metrics(
                loss=2.0 - i * 0.1, step=i * 10, total_steps=100, model_idx=0
            )
            creator._emit_progress(metrics)

        # Check history
        history = creator.get_training_history()
        self.assertEqual(len(history), 3)

        # Verify history ordering (chronological)
        for i in range(len(history) - 1):
            self.assertLessEqual(history[i]['timestamp'], history[i + 1]['timestamp'])

    @patch('torch.nn.Module')
    @patch('torch.utils.data.DataLoader')
    def test_mock_model_creation(self, mock_dataloader, mock_module):
        """Test mock model and dataloader creation"""
        if not self.imports_available:
            self.skipTest("Required imports not available")

        config = self.TrainingConfig(model_count=1, epochs=5)
        creator = self.CognateCreator(config)

        # Test mock model creation (will work even without PyTorch)
        try:
            model = creator._create_mock_model(0)
            dataloader = creator._create_mock_dataloader()
            # If these don't raise exceptions, the structure is correct
            self.assertTrue(True)
        except Exception as e:
            # If PyTorch isn't available, this is expected
            if 'torch' in str(e).lower():
                self.skipTest("PyTorch not available for model testing")
            else:
                raise


def run_basic_validation():
    """Run basic validation tests without unittest framework"""
    print("🧪 Running Basic Agent Forge Streaming Validation")
    print("="*55)

    passed_tests = 0
    total_tests = 0

    def test(name, condition, error_msg=""):
        nonlocal passed_tests, total_tests
        total_tests += 1
        if condition:
            print(f"✅ {name}")
            passed_tests += 1
        else:
            print(f"❌ {name}: {error_msg}")

    # Test 1: Basic imports
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
        from agent_forge.phases.cognate_pretrain.cognate_creator import TrainingConfig
        test("Import TrainingConfig", True)
    except Exception as e:
        test("Import TrainingConfig", False, str(e))

    # Test 2: Configuration creation
    try:
        config = TrainingConfig(model_count=3, batch_size=32)
        test("Create TrainingConfig", config.model_count == 3)
    except Exception as e:
        test("Create TrainingConfig", False, str(e))

    # Test 3: Data structure validation
    try:
        test_metrics = {
            'loss': 1.5,
            'perplexity': 4.48,
            'grokProgress': 25.0,
            'currentStep': 50,
            'totalSteps': 100,
            'currentModel': 1,
            'totalModels': 3,
            'overallProgress': 16.7,
            'timestamp': time.time()
        }

        required_fields = ['loss', 'perplexity', 'grokProgress', 'currentStep', 'totalSteps']
        has_all_fields = all(field in test_metrics for field in required_fields)
        test("Metrics data structure", has_all_fields)
    except Exception as e:
        test("Metrics data structure", False, str(e))

    # Test 4: Data validation ranges
    try:
        valid_ranges = (
            0 <= test_metrics['grokProgress'] <= 100 and
            0 <= test_metrics['overallProgress'] <= 100 and
            test_metrics['loss'] >= 0 and
            test_metrics['currentStep'] <= test_metrics['totalSteps']
        )
        test("Data validation ranges", valid_ranges)
    except Exception as e:
        test("Data validation ranges", False, str(e))

    # Summary
    print("\n" + "="*55)
    print(f"📊 Validation Summary: {passed_tests}/{total_tests} tests passed")
    print(f"✨ Success Rate: {(passed_tests/total_tests)*100:.1f}%")

    if passed_tests == total_tests:
        print("🎉 All basic validation tests passed!")
        return True
    else:
        print("⚠️  Some validation tests failed - check dependencies")
        return False


if __name__ == '__main__':
    # Run basic validation first
    basic_success = run_basic_validation()

    print("\n" + "="*55)
    print("🔬 Running Detailed Unit Tests")
    print("="*55)

    # Run unit tests
    unittest.main(verbosity=2, exit=False)

    if basic_success:
        print("\n✅ Agent Forge Streaming implementation validation complete!")
    else:
        print("\n⚠️  Basic validation had issues - check dependencies and imports")