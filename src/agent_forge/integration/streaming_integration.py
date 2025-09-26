"""
Agent Forge - Streaming Integration and Performance Testing

Complete integration system that connects training process, WebSocket streaming,
and UI components with comprehensive performance validation and testing.
"""

import asyncio
import time
import threading
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import unittest
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our components
from ..phases.cognate_pretrain.cognate_creator import CognateCreator, TrainingConfig
from ..api.websocket_progress import TrainingProgressEmitter
from ..api.compatibility_layer import DataCompatibilityLayer, PerformanceOptimizer

logger = logging.getLogger(__name__)


@dataclass
class IntegrationTestResults:
    """Results from integration testing"""
    test_name: str
    success: bool
    duration: float
    metrics_received: int
    errors: List[str]
    performance_stats: Dict[str, Any]


class StreamingIntegration:
    """
    Complete integration system for real-time training progress streaming.

    Connects all components and provides comprehensive testing and validation.
    """

    def __init__(self):
        self.compatibility_layer = DataCompatibilityLayer()
        self.performance_optimizer = PerformanceOptimizer(max_update_rate=10.0)
        self.emitter: Optional[TrainingProgressEmitter] = None
        self.creator: Optional[CognateCreator] = None

        # Integration state
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.integration_stats = {
            'sessions_created': 0,
            'total_metrics_processed': 0,
            'transformations_successful': 0,
            'websocket_messages_sent': 0,
            'errors_encountered': 0
        }

    def initialize_components(self, websocket_port: int = 5001) -> bool:
        """Initialize all integration components"""
        try:
            # Initialize WebSocket emitter
            self.emitter = TrainingProgressEmitter()
            logger.info(f"WebSocket emitter initialized on port {websocket_port}")

            # Setup HTTP fallback routes
            from ..api.websocket_progress import setup_http_fallback_routes
            setup_http_fallback_routes(self.emitter)

            return True

        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            return False

    def create_training_session(self, session_config: Dict[str, Any]) -> Optional[str]:
        """Create and register a new training session"""
        try:
            # Create training configuration
            config = TrainingConfig(
                model_count=session_config.get('model_count', 3),
                batch_size=session_config.get('batch_size', 32),
                learning_rate=session_config.get('learning_rate', 0.001),
                epochs=session_config.get('epochs', 100),
                progress_update_interval=session_config.get('progress_update_interval', 10),
                session_id=session_config.get('session_id')
            )

            # Create cognate creator
            self.creator = CognateCreator(config)
            session_id = self.creator.session_id

            # Register with WebSocket emitter
            if self.emitter:
                success = self.emitter.register_training_session(session_id, session_config)
                if not success:
                    logger.error("Failed to register session with WebSocket emitter")
                    return None

            # Setup progress callback with integration layer
            def integrated_progress_callback(progress_data: Dict[str, Any]):
                self._handle_progress_update(progress_data)

            self.creator.set_progress_callback(integrated_progress_callback)

            # Track session
            self.active_sessions[session_id] = {
                'config': session_config,
                'creator': self.creator,
                'start_time': time.time(),
                'status': 'initialized',
                'metrics_count': 0
            }

            self.integration_stats['sessions_created'] += 1
            logger.info(f"Training session created: {session_id}")

            return session_id

        except Exception as e:
            logger.error(f"Failed to create training session: {e}")
            return None

    def _handle_progress_update(self, raw_progress: Dict[str, Any]):
        """Handle progress update through integration layer"""
        try:
            session_id = raw_progress.get('sessionId')
            if not session_id or session_id not in self.active_sessions:
                logger.warning(f"Progress update for unknown session: {session_id}")
                return

            # Performance throttling
            if not self.performance_optimizer.should_emit_update(session_id):
                self.performance_optimizer.buffer_update(session_id, raw_progress)
                return

            # Transform data for UI compatibility
            ui_compatible_data = self.compatibility_layer.transform_progress_data(raw_progress)

            # Validate compatibility
            validation = self.compatibility_layer.validate_ui_compatibility(ui_compatible_data)
            if not validation['compatible']:
                logger.error(f"Data compatibility validation failed: {validation['errors']}")
                self.integration_stats['errors_encountered'] += 1
                return

            # Optimize payload
            optimized_data = self.performance_optimizer.optimize_payload(ui_compatible_data.to_dict())

            # Emit through WebSocket
            if self.emitter:
                self.emitter.emit_progress({
                    **optimized_data,
                    'sessionId': session_id
                })
                self.integration_stats['websocket_messages_sent'] += 1

            # Update session tracking
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['metrics_count'] += 1
                self.active_sessions[session_id]['last_update'] = time.time()

            self.integration_stats['total_metrics_processed'] += 1
            self.integration_stats['transformations_successful'] += 1

        except Exception as e:
            logger.error(f"Progress update handling failed: {e}")
            self.integration_stats['errors_encountered'] += 1

    def start_training_session(self, session_id: str) -> bool:
        """Start training for a registered session"""
        if session_id not in self.active_sessions:
            logger.error(f"Session not found: {session_id}")
            return False

        try:
            session_data = self.active_sessions[session_id]
            creator = session_data['creator']

            # Start training in background thread
            def training_thread():
                try:
                    session_data['status'] = 'running'

                    # Create mock models and data loaders for training
                    models = [creator._create_mock_model(i) for i in range(creator.config.model_count)]
                    train_loaders = [creator._create_mock_dataloader() for _ in range(creator.config.model_count)]

                    # Start training with progress streaming
                    trained_models = creator.train_all_models(models, train_loaders)

                    session_data['status'] = 'completed'
                    session_data['trained_models'] = trained_models

                    logger.info(f"Training completed for session: {session_id}")

                except Exception as e:
                    logger.error(f"Training failed for session {session_id}: {e}")
                    session_data['status'] = 'failed'
                    session_data['error'] = str(e)

            training_thread_obj = threading.Thread(target=training_thread)
            training_thread_obj.daemon = True
            training_thread_obj.start()

            return True

        except Exception as e:
            logger.error(f"Failed to start training session {session_id}: {e}")
            return False

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive session status"""
        if session_id not in self.active_sessions:
            return None

        session_data = self.active_sessions[session_id]
        return {
            'session_id': session_id,
            'status': session_data.get('status', 'unknown'),
            'start_time': session_data.get('start_time'),
            'metrics_received': session_data.get('metrics_count', 0),
            'last_update': session_data.get('last_update'),
            'config': session_data.get('config', {}),
            'error': session_data.get('error')
        }

    def run_server(self, host: str = '0.0.0.0', port: int = 5001):
        """Run the integrated server"""
        if not self.emitter:
            raise RuntimeError("Components not initialized. Call initialize_components() first.")

        logger.info(f"Starting integrated streaming server on {host}:{port}")
        self.emitter.run(host=host, port=port, debug=False)

    def cleanup_session(self, session_id: str):
        """Clean up a training session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

        if self.emitter:
            self.emitter.cleanup_session(session_id)

        self.compatibility_layer.clear_cache(session_id)

    def get_integration_stats(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics"""
        return {
            **self.integration_stats,
            'active_sessions': len(self.active_sessions),
            'compatibility_stats': self.compatibility_layer.get_compatibility_stats(),
            'performance_stats': {
                'throttling_enabled': self.performance_optimizer.throttle_enabled,
                'max_update_rate': self.performance_optimizer.max_update_rate,
                'buffered_sessions': len(self.performance_optimizer.update_buffer)
            }
        }


class IntegrationTester:
    """Comprehensive testing suite for integration validation"""

    def __init__(self):
        self.integration = StreamingIntegration()
        self.test_results: List[IntegrationTestResults] = []

    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive integration tests"""
        logger.info("Starting comprehensive integration tests...")

        # Test suite
        tests = [
            self.test_component_initialization,
            self.test_session_creation,
            self.test_data_transformation,
            self.test_websocket_streaming,
            self.test_performance_optimization,
            self.test_error_handling,
            self.test_end_to_end_integration
        ]

        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
                logger.info(f"Test {result.test_name}: {'PASSED' if result.success else 'FAILED'}")
            except Exception as e:
                logger.error(f"Test failed with exception: {e}")
                results.append(IntegrationTestResults(
                    test_name=test.__name__,
                    success=False,
                    duration=0,
                    metrics_received=0,
                    errors=[str(e)],
                    performance_stats={}
                ))

        self.test_results = results

        # Generate summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)

        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            'total_duration': sum(r.duration for r in results),
            'results': results
        }

        logger.info(f"Integration tests completed: {passed_tests}/{total_tests} passed "
                   f"({summary['success_rate']:.1f}% success rate)")

        return summary

    def test_component_initialization(self) -> IntegrationTestResults:
        """Test component initialization"""
        start_time = time.time()
        errors = []

        try:
            success = self.integration.initialize_components()

            if not success:
                errors.append("Component initialization returned False")

            if not self.integration.emitter:
                errors.append("WebSocket emitter not initialized")

        except Exception as e:
            errors.append(f"Initialization exception: {e}")
            success = False

        return IntegrationTestResults(
            test_name="component_initialization",
            success=len(errors) == 0,
            duration=time.time() - start_time,
            metrics_received=0,
            errors=errors,
            performance_stats={}
        )

    def test_session_creation(self) -> IntegrationTestResults:
        """Test training session creation"""
        start_time = time.time()
        errors = []

        try:
            # Initialize components first
            self.integration.initialize_components()

            # Create test session
            session_config = {
                'model_count': 2,
                'batch_size': 16,
                'epochs': 10,
                'progress_update_interval': 5
            }

            session_id = self.integration.create_training_session(session_config)

            if not session_id:
                errors.append("Session creation returned None")
            else:
                # Verify session exists
                status = self.integration.get_session_status(session_id)
                if not status:
                    errors.append("Created session not found")

                # Clean up
                self.integration.cleanup_session(session_id)

        except Exception as e:
            errors.append(f"Session creation exception: {e}")

        return IntegrationTestResults(
            test_name="session_creation",
            success=len(errors) == 0,
            duration=time.time() - start_time,
            metrics_received=0,
            errors=errors,
            performance_stats={}
        )

    def test_data_transformation(self) -> IntegrationTestResults:
        """Test data transformation and compatibility"""
        start_time = time.time()
        errors = []

        try:
            # Test raw data transformation
            raw_data = {
                'sessionId': 'test-123',
                'modelIndex': 0,
                'totalModels': 3,
                'step': 25,
                'totalSteps': 100,
                'loss': 1.8,
                'timestamp': time.time()
            }

            ui_data = self.integration.compatibility_layer.transform_progress_data(raw_data)

            # Validate transformation
            if not ui_data:
                errors.append("Data transformation returned None")

            # Validate compatibility
            validation = self.integration.compatibility_layer.validate_ui_compatibility(ui_data)
            if not validation['compatible']:
                errors.extend(validation['errors'])

        except Exception as e:
            errors.append(f"Data transformation exception: {e}")

        return IntegrationTestResults(
            test_name="data_transformation",
            success=len(errors) == 0,
            duration=time.time() - start_time,
            metrics_received=1,
            errors=errors,
            performance_stats={}
        )

    def test_websocket_streaming(self) -> IntegrationTestResults:
        """Test WebSocket streaming functionality"""
        start_time = time.time()
        errors = []
        metrics_received = 0

        try:
            # This is a simplified test since full WebSocket testing requires client connection
            # In practice, you would use a WebSocket client to verify connectivity

            # Initialize components
            self.integration.initialize_components()

            # Test emitter functionality
            if not self.integration.emitter:
                errors.append("WebSocket emitter not available")
            else:
                # Test session registration
                success = self.integration.emitter.register_training_session(
                    'test-ws-123',
                    {'model_count': 2}
                )

                if not success:
                    errors.append("WebSocket session registration failed")

                # Test progress emission (without actual client)
                test_progress = {
                    'sessionId': 'test-ws-123',
                    'loss': 1.5,
                    'perplexity': 4.48,
                    'grokProgress': 25.0,
                    'currentModel': 1,
                    'totalModels': 2
                }

                self.integration.emitter.emit_progress(test_progress)
                metrics_received = 1

        except Exception as e:
            errors.append(f"WebSocket streaming exception: {e}")

        return IntegrationTestResults(
            test_name="websocket_streaming",
            success=len(errors) == 0,
            duration=time.time() - start_time,
            metrics_received=metrics_received,
            errors=errors,
            performance_stats={}
        )

    def test_performance_optimization(self) -> IntegrationTestResults:
        """Test performance optimization features"""
        start_time = time.time()
        errors = []

        try:
            # Test throttling
            session_id = 'perf-test-123'
            optimizer = self.integration.performance_optimizer

            # Test rapid updates (should be throttled)
            updates_allowed = 0
            for i in range(50):  # Try 50 rapid updates
                if optimizer.should_emit_update(session_id):
                    updates_allowed += 1
                time.sleep(0.01)  # 10ms between updates

            # Should allow significantly fewer than 50 updates
            if updates_allowed > 15:  # Should be around 10 for 10 Hz limit
                errors.append(f"Throttling not working: {updates_allowed} updates allowed")

            # Test payload optimization
            test_data = {
                'loss': 1.23456789,
                'perplexity': 3.456789123,
                'grokProgress': 45.6789,
                'modelParams': 25000000
            }

            optimized = optimizer.optimize_payload(test_data)

            # Check rounding
            if len(str(optimized['loss']).split('.')[1]) > 4:
                errors.append("Loss not properly rounded")

            if len(str(optimized['grokProgress']).split('.')[1]) > 1:
                errors.append("Progress not properly rounded")

        except Exception as e:
            errors.append(f"Performance optimization exception: {e}")

        return IntegrationTestResults(
            test_name="performance_optimization",
            success=len(errors) == 0,
            duration=time.time() - start_time,
            metrics_received=0,
            errors=errors,
            performance_stats={'updates_allowed': updates_allowed if 'updates_allowed' in locals() else 0}
        )

    def test_error_handling(self) -> IntegrationTestResults:
        """Test error handling and recovery"""
        start_time = time.time()
        errors = []

        try:
            # Test invalid data transformation
            invalid_data = {
                'invalid_field': 'invalid_value',
                'loss': 'not_a_number'
            }

            # Should not crash and should return fallback
            ui_data = self.integration.compatibility_layer.transform_progress_data(invalid_data)
            if not ui_data:
                errors.append("No fallback data returned for invalid input")

            # Test validation of incompatible data
            validation = self.integration.compatibility_layer.validate_ui_compatibility(ui_data)
            # Validation should detect issues but not crash

        except Exception as e:
            errors.append(f"Error handling exception: {e}")

        return IntegrationTestResults(
            test_name="error_handling",
            success=len(errors) == 0,
            duration=time.time() - start_time,
            metrics_received=0,
            errors=errors,
            performance_stats={}
        )

    def test_end_to_end_integration(self) -> IntegrationTestResults:
        """Test complete end-to-end integration"""
        start_time = time.time()
        errors = []
        metrics_received = 0

        try:
            # Initialize full integration
            self.integration.initialize_components()

            # Create training session
            session_config = {
                'model_count': 2,
                'batch_size': 8,
                'epochs': 5,  # Short for testing
                'progress_update_interval': 1
            }

            session_id = self.integration.create_training_session(session_config)
            if not session_id:
                errors.append("Failed to create session for e2e test")
                return IntegrationTestResults(
                    test_name="end_to_end_integration",
                    success=False,
                    duration=time.time() - start_time,
                    metrics_received=0,
                    errors=errors,
                    performance_stats={}
                )

            # Monitor metrics for a short period
            test_duration = 3.0  # seconds
            test_end_time = time.time() + test_duration

            initial_metrics_count = self.integration.integration_stats['total_metrics_processed']

            # Start training (non-blocking)
            self.integration.start_training_session(session_id)

            # Wait and monitor
            time.sleep(test_duration)

            final_metrics_count = self.integration.integration_stats['total_metrics_processed']
            metrics_received = final_metrics_count - initial_metrics_count

            # Verify session status
            status = self.integration.get_session_status(session_id)
            if not status:
                errors.append("Session status not available")

            # Clean up
            self.integration.cleanup_session(session_id)

        except Exception as e:
            errors.append(f"End-to-end integration exception: {e}")

        return IntegrationTestResults(
            test_name="end_to_end_integration",
            success=len(errors) == 0,
            duration=time.time() - start_time,
            metrics_received=metrics_received,
            errors=errors,
            performance_stats={'test_duration': test_duration}
        )


def run_performance_benchmark():
    """Run performance benchmark for streaming system"""
    logger.info("Running performance benchmark...")

    integration = StreamingIntegration()
    integration.initialize_components()

    # Create multiple concurrent sessions
    num_sessions = 5
    sessions = []

    start_time = time.time()

    for i in range(num_sessions):
        session_config = {
            'session_id': f'benchmark-{i}',
            'model_count': 3,
            'batch_size': 16,
            'epochs': 20,
            'progress_update_interval': 5
        }

        session_id = integration.create_training_session(session_config)
        if session_id:
            sessions.append(session_id)
            integration.start_training_session(session_id)

    # Monitor for benchmark duration
    benchmark_duration = 10.0  # seconds
    time.sleep(benchmark_duration)

    end_time = time.time()

    # Collect results
    stats = integration.get_integration_stats()

    benchmark_results = {
        'duration': end_time - start_time,
        'sessions_created': len(sessions),
        'total_metrics_processed': stats['total_metrics_processed'],
        'metrics_per_second': stats['total_metrics_processed'] / (end_time - start_time),
        'websocket_messages_sent': stats['websocket_messages_sent'],
        'transformations_successful': stats['transformations_successful'],
        'errors_encountered': stats['errors_encountered'],
        'error_rate': stats['errors_encountered'] / max(1, stats['total_metrics_processed']) * 100
    }

    # Clean up
    for session_id in sessions:
        integration.cleanup_session(session_id)

    logger.info(f"Benchmark completed: {benchmark_results['metrics_per_second']:.1f} metrics/sec, "
               f"{benchmark_results['error_rate']:.1f}% error rate")

    return benchmark_results


if __name__ == '__main__':
    # Run integration tests
    tester = IntegrationTester()
    test_summary = tester.run_all_tests()

    print(f"\nIntegration Test Summary:")
    print(f"Tests passed: {test_summary['passed_tests']}/{test_summary['total_tests']}")
    print(f"Success rate: {test_summary['success_rate']:.1f}%")
    print(f"Total duration: {test_summary['total_duration']:.2f}s")

    # Run performance benchmark
    benchmark_results = run_performance_benchmark()
    print(f"\nPerformance Benchmark:")
    print(f"Metrics per second: {benchmark_results['metrics_per_second']:.1f}")
    print(f"Error rate: {benchmark_results['error_rate']:.1f}%")