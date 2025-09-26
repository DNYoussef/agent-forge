"""
Comprehensive Testing Procedures for Agent Forge Error Recovery System
Tests checkpoint management, error handling, and recovery capabilities
"""

import pytest
import tempfile
import shutil
import os
import torch
import torch.nn as nn
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Import our modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.agent_forge.utils.checkpoint_manager import CheckpointManager, TrainingErrorHandler
from src.agent_forge.utils.system_health_monitor import SystemHealthMonitor
from src.agent_forge.phases.cognate_pretrain.cognate_creator import CognateCreator, TrainingConfig


class TestCheckpointManager:
    """Test suite for CheckpointManager functionality"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def checkpoint_manager(self, temp_dir):
        """Create CheckpointManager instance for testing"""
        return CheckpointManager("test_session", save_dir=temp_dir)

    @pytest.fixture
    def mock_model(self):
        """Create mock PyTorch model for testing"""
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )

    def test_checkpoint_manager_initialization(self, checkpoint_manager):
        """Test CheckpointManager initializes correctly"""
        assert checkpoint_manager.session_id == "test_session"
        assert checkpoint_manager.save_dir.exists()
        assert checkpoint_manager.auto_save_interval == 100
        assert checkpoint_manager.max_checkpoints_per_model == 3

    def test_save_training_checkpoint(self, checkpoint_manager, mock_model):
        """Test saving training checkpoints with complete state"""
        # Prepare checkpoint data
        model_state = {'state_dict': mock_model.state_dict()}
        optimizer_state = {'param_groups': []}
        training_metrics = {'loss': 0.5, 'accuracy': 0.8}

        # Save checkpoint
        checkpoint_path = checkpoint_manager.save_training_checkpoint(
            model_idx=0,
            step=50,
            model_state=model_state,
            optimizer_state=optimizer_state,
            training_metrics=training_metrics
        )

        # Verify checkpoint was saved
        assert Path(checkpoint_path).exists()
        assert (Path(checkpoint_path) / "model_state.pt").exists()
        assert (Path(checkpoint_path) / "optimizer_state.pt").exists()
        assert (Path(checkpoint_path) / "training_metrics.json").exists()
        assert (Path(checkpoint_path) / "metadata.json").exists()
        assert (Path(checkpoint_path) / "random_state.pkl").exists()

    def test_load_training_checkpoint(self, checkpoint_manager, mock_model):
        """Test loading training checkpoints with state restoration"""
        # First, save a checkpoint
        model_state = {'state_dict': mock_model.state_dict()}
        optimizer_state = {'param_groups': []}
        training_metrics = {'loss': 0.3, 'accuracy': 0.9}

        checkpoint_manager.save_training_checkpoint(
            model_idx=1,
            step=75,
            model_state=model_state,
            optimizer_state=optimizer_state,
            training_metrics=training_metrics
        )

        # Load the checkpoint
        loaded_data = checkpoint_manager.load_training_checkpoint(model_idx=1, step=75)

        # Verify loaded data
        assert loaded_data is not None
        assert 'model_state' in loaded_data
        assert 'optimizer_state' in loaded_data
        assert 'training_metrics' in loaded_data
        assert loaded_data['training_metrics']['loss'] == 0.3
        assert loaded_data['training_metrics']['accuracy'] == 0.9

    def test_checkpoint_integrity_verification(self, checkpoint_manager, mock_model):
        """Test checkpoint integrity verification"""
        # Save a checkpoint
        model_state = {'state_dict': mock_model.state_dict()}
        optimizer_state = {}
        training_metrics = {'loss': 0.2}

        checkpoint_path = checkpoint_manager.save_training_checkpoint(
            model_idx=2,
            step=100,
            model_state=model_state,
            optimizer_state=optimizer_state,
            training_metrics=training_metrics
        )

        # Test integrity verification
        assert checkpoint_manager._verify_checkpoint_integrity(Path(checkpoint_path))

        # Corrupt a file and test integrity failure
        (Path(checkpoint_path) / "model_state.pt").unlink()
        assert not checkpoint_manager._verify_checkpoint_integrity(Path(checkpoint_path))

    def test_checkpoint_cleanup(self, checkpoint_manager, mock_model):
        """Test automatic cleanup of old checkpoints"""
        # Save multiple checkpoints for same model
        model_state = {'state_dict': mock_model.state_dict()}
        optimizer_state = {}
        training_metrics = {'loss': 0.1}

        checkpoints = []
        for step in [10, 20, 30, 40, 50]:
            checkpoint_path = checkpoint_manager.save_training_checkpoint(
                model_idx=3,
                step=step,
                model_state=model_state,
                optimizer_state=optimizer_state,
                training_metrics=training_metrics
            )
            checkpoints.append(Path(checkpoint_path))

        # Verify all checkpoints exist
        for checkpoint in checkpoints:
            assert checkpoint.exists()

        # Trigger cleanup (keep last 3)
        checkpoint_manager.cleanup_old_checkpoints(model_idx=3, keep_last=3)

        # Verify only last 3 checkpoints remain
        remaining_checkpoints = list(checkpoint_manager.save_dir.glob("model_3/step_*"))
        assert len(remaining_checkpoints) == 3

    def test_recovery_options_listing(self, checkpoint_manager, mock_model):
        """Test listing available recovery options"""
        # Save some checkpoints
        model_state = {'state_dict': mock_model.state_dict()}
        optimizer_state = {}
        training_metrics = {'loss': 0.4}

        checkpoint_manager.save_training_checkpoint(
            model_idx=0, step=25, model_state=model_state,
            optimizer_state=optimizer_state, training_metrics=training_metrics
        )

        checkpoint_manager.save_phase_checkpoint("training_phase", {"status": "completed"})

        # Get recovery options
        options = checkpoint_manager.get_recovery_options()

        # Verify options include both model and phase checkpoints
        model_options = [opt for opt in options if opt.startswith('model:')]
        phase_options = [opt for opt in options if opt.startswith('phase:')]

        assert len(model_options) > 0
        assert len(phase_options) > 0


class TestTrainingErrorHandler:
    """Test suite for TrainingErrorHandler functionality"""

    @pytest.fixture
    def checkpoint_manager(self):
        """Mock checkpoint manager for testing"""
        return Mock(spec=CheckpointManager)

    @pytest.fixture
    def error_handler(self, checkpoint_manager):
        """Create TrainingErrorHandler instance for testing"""
        return TrainingErrorHandler(checkpoint_manager)

    def test_error_classification_cuda_oom(self, error_handler):
        """Test CUDA out of memory error classification"""
        cuda_error = RuntimeError("CUDA out of memory: tried to allocate 2.00 GiB")
        error_type = error_handler.classify_error(cuda_error)
        assert error_type == 'cuda_oom'

    def test_error_classification_network_failure(self, error_handler):
        """Test network failure error classification"""
        network_error = ConnectionError("Connection timeout occurred")
        error_type = error_handler.classify_error(network_error)
        assert error_type == 'network_failure'

    def test_error_classification_data_corruption(self, error_handler):
        """Test data corruption error classification"""
        data_error = ValueError("Corrupted tensor data detected")
        error_type = error_handler.classify_error(data_error)
        assert error_type == 'data_corruption'

    def test_error_classification_convergence_failure(self, error_handler):
        """Test convergence failure error classification"""
        convergence_error = RuntimeError("Loss became NaN during training")
        error_type = error_handler.classify_error(convergence_error)
        assert error_type == 'convergence_failure'

    def test_recovery_from_memory_error(self, error_handler):
        """Test recovery from CUDA memory errors"""
        context = {'batch_size': 32}
        success = error_handler._recover_from_memory_error(context)

        assert success
        assert 'recovery_suggestions' in context
        suggestions = context['recovery_suggestions']
        assert any('batch size' in suggestion.lower() for suggestion in suggestions)

    def test_recovery_from_network_error(self, error_handler):
        """Test recovery from network errors"""
        context = {}
        success = error_handler._recover_from_network_error(context)

        assert success
        assert 'recovery_suggestions' in context
        suggestions = context['recovery_suggestions']
        assert any('retry' in suggestion.lower() for suggestion in suggestions)

    def test_recovery_strategy_selection(self, error_handler):
        """Test that appropriate recovery strategies are selected"""
        # Test CUDA OOM recovery
        cuda_error = RuntimeError("CUDA out of memory")
        context = {'batch_size': 64}
        success = error_handler.recover_from_error('cuda_oom', context)
        assert success

        # Test unknown error handling
        unknown_context = {}
        success = error_handler.recover_from_error('unknown_error', unknown_context)
        assert not success  # Should fail for unknown errors


class TestSystemHealthMonitor:
    """Test suite for SystemHealthMonitor functionality"""

    @pytest.fixture
    def health_monitor(self):
        """Create SystemHealthMonitor instance for testing"""
        return SystemHealthMonitor()

    def test_system_health_check(self, health_monitor):
        """Test comprehensive system health check"""
        health_status = health_monitor.check_system_health()

        # Verify required fields are present
        assert 'timestamp' in health_status
        assert 'gpu_memory' in health_status
        assert 'disk_space' in health_status
        assert 'cpu_usage' in health_status
        assert 'memory_usage' in health_status
        assert 'prediction_score' in health_status

        # Verify data types
        assert isinstance(health_status['prediction_score'], float)
        assert 0.0 <= health_status['prediction_score'] <= 1.0

    @patch('psutil.disk_usage')
    def test_disk_space_monitoring(self, mock_disk_usage, health_monitor):
        """Test disk space monitoring with mocked data"""
        # Mock disk usage data
        mock_usage = Mock()
        mock_usage.total = 1000 * (1024**3)  # 1TB
        mock_usage.used = 800 * (1024**3)    # 800GB
        mock_usage.free = 200 * (1024**3)    # 200GB
        mock_disk_usage.return_value = mock_usage

        disk_info = health_monitor._check_disk_space()

        assert disk_info['total_gb'] == 1000
        assert disk_info['used_gb'] == 800
        assert disk_info['available_gb'] == 200
        assert disk_info['utilization_percent'] == 80.0

    @patch('psutil.cpu_percent')
    def test_cpu_usage_monitoring(self, mock_cpu_percent, health_monitor):
        """Test CPU usage monitoring with mocked data"""
        mock_cpu_percent.return_value = 75.5

        cpu_info = health_monitor._check_cpu_usage()

        assert cpu_info['current_percent'] == 75.5

    def test_failure_risk_prediction(self, health_monitor):
        """Test failure risk prediction algorithm"""
        # Test low-risk scenario
        low_risk_metrics = {
            'gpu_memory': {'utilization_percent': 50},
            'disk_space': {'utilization_percent': 60},
            'cpu_usage': {'current_percent': 40},
            'memory_usage': {'utilization_percent': 45},
            'temperature': {'cpu_temp_c': 65, 'gpu_temp_c': 70},
            'network_connectivity': {'internet_available': True}
        }

        low_risk = health_monitor.predict_failure_risk(low_risk_metrics)
        assert low_risk < 0.5

        # Test high-risk scenario
        high_risk_metrics = {
            'gpu_memory': {'utilization_percent': 98},
            'disk_space': {'utilization_percent': 97},
            'cpu_usage': {'current_percent': 95},
            'memory_usage': {'utilization_percent': 92},
            'temperature': {'cpu_temp_c': 90, 'gpu_temp_c': 88},
            'network_connectivity': {'internet_available': False}
        }

        high_risk = health_monitor.predict_failure_risk(high_risk_metrics)
        assert high_risk > 0.7

    def test_action_recommendations(self, health_monitor):
        """Test action recommendation system"""
        # Test high GPU usage recommendations
        high_gpu_status = {
            'gpu_memory': {'utilization_percent': 95},
            'disk_space': {'utilization_percent': 50},
            'cpu_usage': {'current_percent': 30},
            'memory_usage': {'utilization_percent': 40}
        }

        recommendations = health_monitor.recommend_actions(high_gpu_status)
        gpu_recommendations = [r for r in recommendations if 'batch size' in r.lower() or 'gpu' in r.lower()]
        assert len(gpu_recommendations) > 0

    def test_health_trend_tracking(self, health_monitor):
        """Test health trend tracking over time"""
        # Simulate multiple health checks
        for i in range(5):
            health_monitor.check_system_health()
            time.sleep(0.1)  # Small delay

        trends = health_monitor.get_health_trend(hours=1)

        assert len(trends['timestamps']) >= 5
        assert len(trends['risk_score']) >= 5
        assert all(isinstance(score, float) for score in trends['risk_score'])


class TestCognateCreator:
    """Test suite for enhanced CognateCreator functionality"""

    @pytest.fixture
    def training_config(self):
        """Create training configuration for testing"""
        return TrainingConfig(
            model_count=2,
            batch_size=16,
            learning_rate=0.001,
            epochs=10,
            checkpoint_interval=50,
            enable_recovery=True
        )

    @pytest.fixture
    def cognate_creator(self, training_config):
        """Create CognateCreator instance for testing"""
        return CognateCreator(training_config)

    def test_cognate_creator_initialization(self, cognate_creator):
        """Test CognateCreator initializes with recovery systems"""
        assert cognate_creator.session_id is not None
        assert cognate_creator.config.enable_recovery
        assert cognate_creator.recovery_count == 0
        assert not cognate_creator.recovery_active

    @patch('src.agent_forge.phases.cognate_pretrain.cognate_creator.CheckpointManager')
    @patch('src.agent_forge.phases.cognate_pretrain.cognate_creator.TrainingErrorHandler')
    def test_model_creation_with_recovery(self, mock_error_handler, mock_checkpoint_manager, cognate_creator):
        """Test model creation with recovery systems enabled"""
        # Mock the recovery systems
        mock_checkpoint_manager.return_value.get_recovery_options.return_value = []
        mock_error_handler.return_value.classify_error.return_value = 'unknown_error'

        # Test model creation
        model_paths = cognate_creator.create_three_models(save_dir="test_models")

        # Verify models were created (or attempted)
        assert isinstance(model_paths, list)
        # Note: Due to mocking, this tests the recovery flow structure

    def test_recovery_status_tracking(self, cognate_creator):
        """Test recovery status tracking and reporting"""
        status = cognate_creator.get_recovery_status()

        assert 'session_id' in status
        assert 'recovery_enabled' in status
        assert 'recovery_active' in status
        assert 'recovery_count' in status
        assert status['session_id'] == cognate_creator.session_id

    def test_error_simulation_and_handling(self, cognate_creator):
        """Test error simulation and recovery handling"""
        # This tests the error simulation logic in training
        with patch.object(cognate_creator, '_train_epoch_with_recovery') as mock_train:
            # Simulate CUDA OOM error
            mock_train.side_effect = RuntimeError("CUDA out of memory")

            # Test that error handling is triggered
            try:
                cognate_creator._create_single_model_with_recovery(0, "test_dir")
            except Exception:
                pass  # Expected to fail in test environment

            # Verify error handler would be called
            assert mock_train.called

    def test_configuration_adaptation(self, cognate_creator):
        """Test configuration adaptation for recovery"""
        original_batch_size = cognate_creator.config.batch_size
        original_lr = cognate_creator.config.learning_rate

        # Test batch size reduction suggestion
        suggestions = ["Reduce batch size from 32 to 16"]
        cognate_creator._apply_recovery_suggestions(suggestions)

        # Verify batch size was reduced
        assert cognate_creator.config.batch_size == 16

        # Test learning rate reduction suggestion
        suggestions = ["Reduce learning rate for stability"]
        cognate_creator._apply_recovery_suggestions(suggestions)

        # Verify learning rate was reduced
        assert cognate_creator.config.learning_rate < original_lr


class TestIntegrationScenarios:
    """Integration tests for complete error recovery workflows"""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for integration tests"""
        workspace = tempfile.mkdtemp()
        yield workspace
        shutil.rmtree(workspace)

    def test_complete_recovery_workflow(self, temp_workspace):
        """Test complete error recovery workflow from start to finish"""
        # Initialize all components
        config = TrainingConfig(
            model_count=2,
            batch_size=8,
            epochs=5,
            enable_recovery=True,
            save_dir=temp_workspace
        )

        creator = CognateCreator(config)

        # Mock recovery systems for controlled testing
        with patch.object(creator, 'checkpoint_manager') as mock_checkpoint, \
             patch.object(creator, 'error_handler') as mock_error_handler:

            # Configure mocks
            mock_checkpoint.get_recovery_options.return_value = [
                'model:0:step:10', 'phase:training_start'
            ]
            mock_error_handler.classify_error.return_value = 'cuda_oom'
            mock_error_handler.recover_from_error.return_value = True

            # Simulate recovery scenario
            try:
                model_paths = creator.create_three_models()
                # Test passes if no unhandled exceptions occur
                assert True
            except Exception as e:
                # If exceptions occur, they should be handled gracefully
                assert creator.recovery_count > 0 or creator.recovery_active

    def test_checkpoint_corruption_recovery(self, temp_workspace):
        """Test recovery from checkpoint corruption scenarios"""
        checkpoint_manager = CheckpointManager("corruption_test", save_dir=temp_workspace)

        # Create a valid checkpoint first
        mock_model = nn.Linear(10, 5)
        model_state = {'state_dict': mock_model.state_dict()}
        optimizer_state = {}
        training_metrics = {'loss': 0.5}

        checkpoint_path = checkpoint_manager.save_training_checkpoint(
            model_idx=0, step=50,
            model_state=model_state,
            optimizer_state=optimizer_state,
            training_metrics=training_metrics
        )

        # Corrupt the checkpoint by removing metadata
        metadata_path = Path(checkpoint_path) / "metadata.json"
        metadata_path.unlink()

        # Test that corruption is detected
        assert not checkpoint_manager._verify_checkpoint_integrity(Path(checkpoint_path))

        # Test that load fails gracefully
        loaded_data = checkpoint_manager.load_training_checkpoint(model_idx=0, step=50)
        assert loaded_data is None

    def test_system_resource_exhaustion_scenario(self):
        """Test system behavior under resource exhaustion"""
        health_monitor = SystemHealthMonitor()

        # Simulate resource exhaustion
        exhaustion_metrics = {
            'gpu_memory': {'utilization_percent': 99},
            'disk_space': {'utilization_percent': 98},
            'cpu_usage': {'current_percent': 96},
            'memory_usage': {'utilization_percent': 94}
        }

        risk_score = health_monitor.predict_failure_risk(exhaustion_metrics)
        recommendations = health_monitor.recommend_actions(exhaustion_metrics)

        # Verify high risk is detected
        assert risk_score > 0.8

        # Verify appropriate recommendations are provided
        assert len(recommendations) > 3
        assert any('batch size' in rec.lower() for rec in recommendations)
        assert any('checkpoint' in rec.lower() for rec in recommendations)


class TestAPIErrorHandling:
    """Test API error handling and response formatting"""

    def test_error_analysis_structure(self):
        """Test error analysis data structure"""
        from src.web.dashboard.app.api.phases.cognate.route import analyzeTrainingError

        # Test CUDA error analysis
        cuda_error = RuntimeError("CUDA out of memory: tried to allocate 2.00 GiB")
        analysis = analyzeTrainingError(cuda_error)

        assert analysis['errorType'] == 'cuda_oom'
        assert analysis['severity'] == 'high'
        assert analysis['category'] == 'system'
        assert len(analysis['possibleCauses']) > 0
        assert len(analysis['recommendations']) > 0
        assert analysis['autoRecoverable'] is True

    def test_recovery_options_generation(self):
        """Test recovery options generation"""
        from src.web.dashboard.app.api.phases.cognate.route import getRecoveryOptions

        options = getRecoveryOptions("test_session")

        assert len(options) >= 3
        assert all('id' in option for option in options)
        assert all('name' in option for option in options)
        assert all('successRate' in option for option in options)
        assert all(0 <= option['successRate'] <= 1 for option in options)

    def test_configuration_validation(self):
        """Test configuration validation logic"""
        from src.web.dashboard.app.api.phases.cognate.route import validateCognateConfig

        # Test valid configuration
        valid_config = {
            'model_count': 3,
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 100,
            'checkpoint_interval': 50
        }

        result = validateCognateConfig(valid_config)
        assert result['valid'] is True
        assert len(result['errors']) == 0

        # Test invalid configuration
        invalid_config = {
            'model_count': 15,  # Too many
            'batch_size': 0,    # Invalid
            'learning_rate': 2.0,  # Too high
            'epochs': -5        # Invalid
        }

        result = validateCognateConfig(invalid_config)
        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert len(result['suggestions']) > 0


# Performance and reliability tests
class TestSystemReliability:
    """Test system reliability under various conditions"""

    def test_checkpoint_manager_performance(self):
        """Test CheckpointManager performance with multiple operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager("perf_test", save_dir=temp_dir)

            # Measure checkpoint save/load performance
            start_time = time.time()

            for i in range(10):
                mock_model = nn.Linear(100, 50)
                model_state = {'state_dict': mock_model.state_dict()}

                manager.save_training_checkpoint(
                    model_idx=0,
                    step=i * 10,
                    model_state=model_state,
                    optimizer_state={},
                    training_metrics={'loss': 0.1 * i}
                )

            save_time = time.time() - start_time

            # Load checkpoints
            start_time = time.time()
            for i in range(10):
                manager.load_training_checkpoint(model_idx=0, step=i * 10)

            load_time = time.time() - start_time

            # Verify reasonable performance (adjust thresholds as needed)
            assert save_time < 30.0  # Should save 10 checkpoints in under 30 seconds
            assert load_time < 15.0  # Should load 10 checkpoints in under 15 seconds

    def test_error_handler_stress(self):
        """Test error handler under rapid error classification"""
        manager = Mock()
        handler = TrainingErrorHandler(manager)

        error_types = [
            RuntimeError("CUDA out of memory"),
            ConnectionError("Network timeout"),
            ValueError("Corrupted data"),
            RuntimeError("Loss became NaN"),
            TypeError("Invalid configuration")
        ]

        start_time = time.time()

        # Classify many errors rapidly
        for _ in range(100):
            for error in error_types:
                error_type = handler.classify_error(error)
                assert error_type in handler.error_patterns.keys() or error_type == 'unknown'

        classification_time = time.time() - start_time

        # Should handle 500 classifications quickly
        assert classification_time < 5.0

    def test_memory_usage_stability(self):
        """Test memory usage doesn't grow excessively"""
        import psutil
        import gc

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create and destroy many objects
        for _ in range(100):
            with tempfile.TemporaryDirectory() as temp_dir:
                manager = CheckpointManager(f"mem_test_{_}", save_dir=temp_dir)

                for i in range(10):
                    mock_model = nn.Linear(50, 25)
                    manager.save_training_checkpoint(
                        model_idx=0,
                        step=i,
                        model_state={'state_dict': mock_model.state_dict()},
                        optimizer_state={},
                        training_metrics={'loss': 0.1}
                    )

            # Force garbage collection
            gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable (less than 100MB for this test)
        assert memory_growth < 100.0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "--color=yes"])