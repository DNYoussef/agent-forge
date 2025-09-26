#!/usr/bin/env python3
"""
Agent Forge - Real-time Progress Streaming Demo

Complete demonstration of the real-time training progress streaming system.
Shows integration between training process, WebSocket streaming, and UI compatibility.

Usage:
    python examples/agent_forge_streaming_demo.py [--mode demo|test|benchmark]
"""

import sys
import time
import asyncio
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from agent_forge.integration.streaming_integration import StreamingIntegration, IntegrationTester
from agent_forge.phases.cognate_pretrain.cognate_creator import TrainingConfig
from agent_forge.api.websocket_progress import simulate_training_progress

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StreamingDemo:
    """Comprehensive demonstration of the streaming system"""

    def __init__(self):
        self.integration = StreamingIntegration()

    def run_demo(self):
        """Run interactive demonstration"""
        print("\n" + "="*60)
        print("🚀 Agent Forge - Real-time Training Progress Streaming Demo")
        print("="*60)

        # Initialize components
        print("\n📦 Initializing components...")
        if not self.integration.initialize_components():
            print("❌ Component initialization failed!")
            return False

        print("✅ Components initialized successfully!")

        # Create demo training session
        print("\n🧠 Creating training session...")
        session_config = {
            'model_count': 3,
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 50,
            'progress_update_interval': 5
        }

        session_id = self.integration.create_training_session(session_config)
        if not session_id:
            print("❌ Failed to create training session!")
            return False

        print(f"✅ Training session created: {session_id}")

        # Display session information
        self._display_session_info(session_id, session_config)

        # Start training
        print("\n🚂 Starting training session...")
        if not self.integration.start_training_session(session_id):
            print("❌ Failed to start training!")
            return False

        print("✅ Training started! Monitoring progress...")

        # Monitor progress
        self._monitor_progress(session_id)

        # Cleanup
        print("\n🧹 Cleaning up session...")
        self.integration.cleanup_session(session_id)
        print("✅ Demo completed successfully!")

        return True

    def _display_session_info(self, session_id: str, config: dict):
        """Display session information"""
        print(f"""
📋 Session Configuration:
   Session ID: {session_id}
   Models: {config['model_count']}
   Batch Size: {config['batch_size']}
   Learning Rate: {config['learning_rate']}
   Epochs: {config['epochs']}
   Update Interval: {config['progress_update_interval']} steps
        """)

    def _monitor_progress(self, session_id: str, duration: float = 15.0):
        """Monitor training progress for specified duration"""
        start_time = time.time()
        last_status_time = 0

        print(f"⏱️  Monitoring for {duration} seconds...")
        print("\n📈 Progress Updates:")

        while time.time() - start_time < duration:
            current_time = time.time()

            # Get session status every 2 seconds
            if current_time - last_status_time >= 2.0:
                status = self.integration.get_session_status(session_id)
                if status:
                    metrics_count = status.get('metrics_received', 0)
                    session_status = status.get('status', 'unknown')

                    print(f"   📊 Status: {session_status} | Metrics received: {metrics_count}")

                    if session_status == 'completed':
                        print("🎉 Training completed!")
                        break

                last_status_time = current_time

            time.sleep(0.5)

        # Final statistics
        final_stats = self.integration.get_integration_stats()
        print(f"\n📈 Final Statistics:")
        print(f"   Total metrics processed: {final_stats['total_metrics_processed']}")
        print(f"   WebSocket messages sent: {final_stats['websocket_messages_sent']}")
        print(f"   Transformations successful: {final_stats['transformations_successful']}")
        print(f"   Errors encountered: {final_stats['errors_encountered']}")

    def run_server_mode(self):
        """Run in server mode for UI testing"""
        print("\n🌐 Starting server mode for UI testing...")
        print("Server will run on http://localhost:5001")
        print("WebSocket endpoint: ws://localhost:5001")
        print("\nPress Ctrl+C to stop the server")

        try:
            self.integration.run_server(host='0.0.0.0', port=5001)
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")

    def create_test_session(self):
        """Create a test session and return session ID for UI testing"""
        print("\n🧪 Creating test session for UI...")

        # Initialize components
        if not self.integration.initialize_components():
            print("❌ Component initialization failed!")
            return None

        # Create session
        session_config = {
            'model_count': 3,
            'batch_size': 16,
            'learning_rate': 0.0005,
            'epochs': 100,
            'progress_update_interval': 10
        }

        session_id = self.integration.create_training_session(session_config)
        if session_id:
            print(f"✅ Test session created: {session_id}")

            # Start training
            self.integration.start_training_session(session_id)
            print("✅ Training started!")

            return session_id

        return None


def run_integration_tests():
    """Run comprehensive integration tests"""
    print("\n🧪 Running Integration Tests...")
    print("="*50)

    tester = IntegrationTester()
    results = tester.run_all_tests()

    # Display results
    print(f"\n📊 Test Results Summary:")
    print(f"   Total tests: {results['total_tests']}")
    print(f"   Passed: {results['passed_tests']}")
    print(f"   Failed: {results['failed_tests']}")
    print(f"   Success rate: {results['success_rate']:.1f}%")
    print(f"   Total duration: {results['total_duration']:.2f}s")

    # Detail failed tests
    failed_tests = [r for r in results['results'] if not r.success]
    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for test in failed_tests:
            print(f"   - {test.test_name}")
            for error in test.errors:
                print(f"     • {error}")

    return results['success_rate'] > 80  # 80% pass rate required


def run_performance_benchmark():
    """Run performance benchmark"""
    print("\n⚡ Running Performance Benchmark...")
    print("="*50)

    from agent_forge.integration.streaming_integration import run_performance_benchmark

    benchmark_results = run_performance_benchmark()

    print(f"\n📈 Benchmark Results:")
    print(f"   Duration: {benchmark_results['duration']:.2f}s")
    print(f"   Sessions created: {benchmark_results['sessions_created']}")
    print(f"   Metrics processed: {benchmark_results['total_metrics_processed']}")
    print(f"   Metrics per second: {benchmark_results['metrics_per_second']:.1f}")
    print(f"   WebSocket messages: {benchmark_results['websocket_messages_sent']}")
    print(f"   Error rate: {benchmark_results['error_rate']:.2f}%")

    # Performance thresholds
    performance_good = (
        benchmark_results['metrics_per_second'] > 5.0 and  # At least 5 metrics/sec
        benchmark_results['error_rate'] < 5.0  # Less than 5% error rate
    )

    print(f"\n{'✅' if performance_good else '⚠️'} Performance: {'Good' if performance_good else 'Needs improvement'}")

    return benchmark_results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Agent Forge Streaming Demo')
    parser.add_argument('--mode', choices=['demo', 'test', 'benchmark', 'server'],
                       default='demo', help='Mode to run')
    parser.add_argument('--duration', type=float, default=15.0,
                       help='Duration for demo monitoring (seconds)')

    args = parser.parse_args()

    print("🤖 Agent Forge - Real-time Progress Streaming System")
    print("="*60)

    if args.mode == 'demo':
        demo = StreamingDemo()
        success = demo.run_demo()
        sys.exit(0 if success else 1)

    elif args.mode == 'test':
        success = run_integration_tests()
        sys.exit(0 if success else 1)

    elif args.mode == 'benchmark':
        results = run_performance_benchmark()
        # Exit code based on performance
        performance_good = (results['metrics_per_second'] > 5.0 and
                          results['error_rate'] < 5.0)
        sys.exit(0 if performance_good else 1)

    elif args.mode == 'server':
        demo = StreamingDemo()

        # Initialize and create test session
        if not demo.integration.initialize_components():
            print("❌ Failed to initialize components")
            sys.exit(1)

        test_session = demo.create_test_session()
        if test_session:
            print(f"\n📡 Test session running: {test_session}")
            print("You can now connect your UI to ws://localhost:5001")

        # Run server
        demo.run_server_mode()


if __name__ == '__main__':
    main()