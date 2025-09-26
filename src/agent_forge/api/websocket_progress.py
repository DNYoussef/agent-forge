"""
Agent Forge - WebSocket Progress Emitter

Real-time training progress streaming using WebSocket with fallback to HTTP polling.
Provides layered enhancement approach preserving existing interfaces.
"""

import json
import time
import asyncio
import logging
from typing import Dict, Any, Set, Optional, Callable
from dataclasses import dataclass, asdict
from flask import Flask
from flask_socketio import SocketIO, emit, join_room, leave_room
import threading
from queue import Queue, Empty
import uuid

logger = logging.getLogger(__name__)


@dataclass
class TrainingSession:
    """Training session metadata"""
    session_id: str
    config: Dict[str, Any]
    start_time: float
    status: str = 'starting'
    models_completed: int = 0
    current_model: int = 1
    total_models: int = 3
    last_update: float = 0


class TrainingProgressEmitter:
    """
    WebSocket-based real-time training progress emitter.

    Provides real-time streaming of training metrics with automatic fallbacks
    and compatibility with existing HTTP endpoints.
    """

    def __init__(self, app: Optional[Flask] = None):
        self.app = app or Flask(__name__)
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode='threading',
            logger=False,
            engineio_logger=False
        )

        # Session management
        self.active_sessions: Dict[str, TrainingSession] = {}
        self.client_sessions: Dict[str, Set[str]] = {}  # client_id -> session_ids
        self.progress_queue = Queue()

        # Metrics storage for HTTP fallback
        self.latest_metrics: Dict[str, Dict[str, Any]] = {}
        self.metrics_history: Dict[str, list] = {}

        # Background processing
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        self._setup_socket_handlers()
        logger.info("TrainingProgressEmitter initialized")

    def _setup_socket_handlers(self):
        """Setup WebSocket event handlers"""

        @self.socketio.on('connect')
        def handle_connect():
            client_id = self.socketio.request.sid
            logger.info(f"Client connected: {client_id}")
            emit('connection_established', {'client_id': client_id})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            client_id = self.socketio.request.sid
            self._cleanup_client(client_id)
            logger.info(f"Client disconnected: {client_id}")

        @self.socketio.on('subscribe_session')
        def handle_subscribe(data):
            client_id = self.socketio.request.sid
            session_id = data.get('sessionId')

            if not session_id:
                emit('error', {'message': 'sessionId required'})
                return

            self._subscribe_client_to_session(client_id, session_id)
            join_room(session_id)

            # Send current session state if available
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                emit('session_status', asdict(session))

            # Send latest metrics if available
            if session_id in self.latest_metrics:
                emit('progress_update', {
                    'metrics': self.latest_metrics[session_id],
                    'status': 'running'
                })

            logger.info(f"Client {client_id} subscribed to session {session_id}")

        @self.socketio.on('unsubscribe_session')
        def handle_unsubscribe(data):
            client_id = self.socketio.request.sid
            session_id = data.get('sessionId')

            if session_id:
                self._unsubscribe_client_from_session(client_id, session_id)
                leave_room(session_id)
                logger.info(f"Client {client_id} unsubscribed from session {session_id}")

        @self.socketio.on('get_session_history')
        def handle_get_history(data):
            session_id = data.get('sessionId')
            limit = data.get('limit', 100)

            if session_id in self.metrics_history:
                history = self.metrics_history[session_id][-limit:]
                emit('session_history', {'sessionId': session_id, 'history': history})
            else:
                emit('session_history', {'sessionId': session_id, 'history': []})

    def _subscribe_client_to_session(self, client_id: str, session_id: str):
        """Subscribe client to session updates"""
        if client_id not in self.client_sessions:
            self.client_sessions[client_id] = set()
        self.client_sessions[client_id].add(session_id)

    def _unsubscribe_client_from_session(self, client_id: str, session_id: str):
        """Unsubscribe client from session updates"""
        if client_id in self.client_sessions:
            self.client_sessions[client_id].discard(session_id)

    def _cleanup_client(self, client_id: str):
        """Clean up client subscriptions"""
        if client_id in self.client_sessions:
            del self.client_sessions[client_id]

    def register_training_session(self, session_id: str, config: Dict[str, Any]) -> bool:
        """
        Register new training session for progress tracking.

        Args:
            session_id: Unique session identifier
            config: Training configuration

        Returns:
            True if registration successful
        """
        try:
            session = TrainingSession(
                session_id=session_id,
                config=config,
                start_time=time.time(),
                status='starting',
                models_completed=0,
                total_models=config.get('model_count', 3)
            )

            self.active_sessions[session_id] = session
            self.metrics_history[session_id] = []

            # Emit training started event
            self.socketio.emit('training_started', {
                'sessionId': session_id,
                'config': config,
                'timestamp': time.time()
            }, room=session_id)

            logger.info(f"Training session registered: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to register training session {session_id}: {e}")
            return False

    def emit_progress(self, progress_data: Dict[str, Any]):
        """
        Emit progress update to specific session.

        Args:
            progress_data: Training progress metrics
        """
        session_id = progress_data.get('sessionId')
        if not session_id:
            logger.warning("Progress data missing sessionId")
            return

        try:
            # Update session state
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.current_model = progress_data.get('currentModel', session.current_model)
                session.last_update = time.time()

                # Check for model completion
                if progress_data.get('modelCompleted'):
                    session.models_completed = progress_data.get('modelIndex', 0) + 1

                # Check for training completion
                if progress_data.get('event') == 'training_completed':
                    session.status = 'completed'

            # Store latest metrics for HTTP fallback
            self.latest_metrics[session_id] = progress_data.copy()

            # Store in history (keep last 1000 entries per session)
            if session_id not in self.metrics_history:
                self.metrics_history[session_id] = []

            self.metrics_history[session_id].append({
                **progress_data,
                'timestamp': time.time()
            })

            if len(self.metrics_history[session_id]) > 1000:
                self.metrics_history[session_id] = self.metrics_history[session_id][-1000:]

            # Emit to WebSocket clients
            self.socketio.emit('progress_update', {
                'metrics': progress_data,
                'status': self.active_sessions.get(session_id, TrainingSession('', {}, 0)).status if session_id in self.active_sessions else 'unknown'
            }, room=session_id)

            # Special handling for model completion
            if progress_data.get('modelCompleted'):
                self.socketio.emit('model_completed', {
                    'sessionId': session_id,
                    'modelIndex': progress_data.get('modelIndex', 0),
                    'currentModel': progress_data.get('currentModel', 1),
                    'totalModels': progress_data.get('totalModels', 3)
                }, room=session_id)

        except Exception as e:
            logger.error(f"Failed to emit progress for session {session_id}: {e}")

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session status"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            return {
                **asdict(session),
                'latest_metrics': self.latest_metrics.get(session_id)
            }
        return None

    def get_latest_metrics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get latest metrics for HTTP fallback"""
        return self.latest_metrics.get(session_id)

    def get_metrics_history(self, session_id: str, limit: int = 100) -> list:
        """Get metrics history for HTTP fallback"""
        if session_id in self.metrics_history:
            return self.metrics_history[session_id][-limit:]
        return []

    def cleanup_session(self, session_id: str):
        """Clean up completed or abandoned session"""
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

            # Keep metrics for a while for potential HTTP requests
            # They will be cleaned up by background process

            self.socketio.emit('session_cleanup', {'sessionId': session_id}, room=session_id)
            logger.info(f"Session cleaned up: {session_id}")

        except Exception as e:
            logger.error(f"Failed to cleanup session {session_id}: {e}")

    def start_background_processing(self):
        """Start background processing thread"""
        if not self._running:
            self._running = True
            self._worker_thread = threading.Thread(target=self._background_worker)
            self._worker_thread.daemon = True
            self._worker_thread.start()
            logger.info("Background processing started")

    def stop_background_processing(self):
        """Stop background processing thread"""
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        logger.info("Background processing stopped")

    def _background_worker(self):
        """Background worker for cleanup and maintenance"""
        while self._running:
            try:
                self._cleanup_old_sessions()
                self._cleanup_old_metrics()
                time.sleep(60)  # Run every minute
            except Exception as e:
                logger.error(f"Background worker error: {e}")

    def _cleanup_old_sessions(self):
        """Clean up old inactive sessions"""
        current_time = time.time()
        expired_sessions = []

        for session_id, session in self.active_sessions.items():
            # Clean up sessions older than 4 hours or completed sessions older than 1 hour
            if (current_time - session.start_time > 14400 or  # 4 hours
                (session.status == 'completed' and current_time - session.last_update > 3600)):  # 1 hour
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            self.cleanup_session(session_id)

    def _cleanup_old_metrics(self):
        """Clean up old metrics history"""
        current_time = time.time()
        sessions_to_clean = []

        for session_id, history in self.metrics_history.items():
            if history and current_time - history[-1]['timestamp'] > 86400:  # 24 hours
                sessions_to_clean.append(session_id)

        for session_id in sessions_to_clean:
            del self.metrics_history[session_id]
            if session_id in self.latest_metrics:
                del self.latest_metrics[session_id]

    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Run the WebSocket server"""
        self.start_background_processing()
        logger.info(f"Starting WebSocket server on {host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=debug)


# Flask routes for HTTP fallback
def setup_http_fallback_routes(emitter: TrainingProgressEmitter):
    """Setup HTTP routes for polling fallback"""

    @emitter.app.route('/api/training/status/<session_id>')
    def get_training_status(session_id):
        """HTTP endpoint for training status"""
        status = emitter.get_session_status(session_id)
        if status:
            return {'success': True, 'data': status}
        return {'success': False, 'error': 'Session not found'}, 404

    @emitter.app.route('/api/training/metrics/<session_id>')
    def get_training_metrics(session_id):
        """HTTP endpoint for latest metrics"""
        metrics = emitter.get_latest_metrics(session_id)
        if metrics:
            return {'success': True, 'data': metrics}
        return {'success': False, 'error': 'No metrics found'}, 404

    @emitter.app.route('/api/training/history/<session_id>')
    def get_training_history(session_id):
        """HTTP endpoint for metrics history"""
        from flask import request
        limit = request.args.get('limit', 100, type=int)
        history = emitter.get_metrics_history(session_id, limit)
        return {'success': True, 'data': history}


# Example usage and testing functions
def create_test_emitter() -> TrainingProgressEmitter:
    """Create test emitter instance"""
    emitter = TrainingProgressEmitter()
    setup_http_fallback_routes(emitter)
    return emitter


def simulate_training_progress(emitter: TrainingProgressEmitter, session_id: str):
    """Simulate training progress for testing"""
    import threading
    import random

    def simulate():
        # Register session
        config = {
            'model_count': 3,
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 100
        }
        emitter.register_training_session(session_id, config)

        # Simulate training for 3 models
        for model_idx in range(3):
            for step in range(0, 101, 10):  # Steps 0-100, every 10
                loss = 3.0 * math.exp(-step / 50) + random.uniform(-0.1, 0.1)  # Decreasing loss with noise

                metrics = {
                    'sessionId': session_id,
                    'modelIndex': model_idx,
                    'totalModels': 3,
                    'step': step,
                    'totalSteps': 100,
                    'loss': round(loss, 4),
                    'perplexity': round(math.exp(min(loss, 10)), 2),
                    'grokProgress': min(100, max(0, (2.0 - loss) / 2.0 * 100)),
                    'modelParams': 25000000,
                    'currentModel': model_idx + 1,
                    'overallProgress': round(((model_idx * 100 + step) / 3), 1),
                    'timestamp': time.time()
                }

                if step == 100:  # Model completed
                    metrics['modelCompleted'] = True

                emitter.emit_progress(metrics)
                time.sleep(0.5)  # Simulate time between updates

        # Training completed
        emitter.emit_progress({
            'sessionId': session_id,
            'event': 'training_completed',
            'overallProgress': 100.0,
            'timestamp': time.time()
        })

    thread = threading.Thread(target=simulate)
    thread.daemon = True
    thread.start()
    return thread


if __name__ == '__main__':
    # Example usage
    emitter = create_test_emitter()

    # Start simulation in background
    simulate_training_progress(emitter, 'test-session-123')

    # Run server
    emitter.run(debug=True)