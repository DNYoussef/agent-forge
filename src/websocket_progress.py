#!/usr/bin/env python3
"""
WebSocket Progress Server for Agent Forge
========================================

Real-time progress streaming with Socket.IO for browser compatibility.
Integrates with existing training loops via callback hooks.
Provides fallback support for HTTP polling.

Features:
- Room-based sessions (sessionId as room)
- Progress events: training_started, step_update, model_completed, phase_completed
- Error events: training_error, connection_lost
- Automatic reconnection handling
- Backward compatibility with HTTP polling
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict

import socketio
from aiohttp import web
from aiohttp.web import Application

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProgressMetrics:
    """Progress metrics matching existing polling data format."""
    loss: float = 0.0
    perplexity: float = 0.0
    grokProgress: int = 0
    currentStep: int = 0
    totalSteps: int = 1000
    currentModel: int = 1
    totalModels: int = 3
    learningRate: float = 0.001
    accuracy: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProgressEvent:
    """WebSocket event structure."""
    sessionId: str
    eventType: str  # training_started, step_update, model_completed, phase_completed, error
    metrics: ProgressMetrics
    status: str = "running"  # running, completed, error, paused
    timestamp: str = None
    message: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sessionId': self.sessionId,
            'eventType': self.eventType,
            'metrics': self.metrics.to_dict(),
            'status': self.status,
            'timestamp': self.timestamp,
            'message': self.message
        }


class WebSocketProgressServer:
    """WebSocket server for real-time progress streaming."""
    
    def __init__(self, port: int = 8001):
        self.port = port
        self.sio = socketio.AsyncServer(
            cors_allowed_origins="*",
            logger=logger,
            engineio_logger=logger
        )
        self.app = web.Application()
        self.sio.attach(self.app)
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_callbacks: Dict[str, Callable] = {}
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # HTTP fallback endpoints (for compatibility)
        self._setup_http_routes()
        
    def _setup_event_handlers(self):
        """Setup WebSocket event handlers."""
        
        @self.sio.event
        async def connect(sid, environ):
            logger.info(f"Client connected: {sid}")
            await self.sio.emit('connection_established', {
                'status': 'connected',
                'timestamp': datetime.now().isoformat(),
                'fallback_available': True
            }, room=sid)
        
        @self.sio.event
        async def disconnect(sid):
            logger.info(f"Client disconnected: {sid}")
            # Clean up any session associations
            for session_id, session_data in self.active_sessions.items():
                if session_data.get('socket_id') == sid:
                    session_data['socket_connected'] = False
                    break
        
        @self.sio.event
        async def join_room(sid, session_id):
            """Join a training session room."""
            await self.sio.enter_room(sid, session_id)
            logger.info(f"Client {sid} joined session room: {session_id}")
            
            # Track session
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = {
                    'created_at': datetime.now().isoformat(),
                    'last_update': datetime.now().isoformat(),
                    'socket_connected': True,
                    'socket_id': sid,
                    'metrics': ProgressMetrics().to_dict()
                }
            else:
                self.active_sessions[session_id].update({
                    'socket_connected': True,
                    'socket_id': sid,
                    'last_update': datetime.now().isoformat()
                })
            
            # Send current session state if exists
            await self.sio.emit('session_joined', {
                'sessionId': session_id,
                'status': 'joined',
                'currentState': self.active_sessions[session_id]['metrics'],
                'timestamp': datetime.now().isoformat()
            }, room=session_id)
        
        @self.sio.event
        async def leave_room(sid, session_id):
            """Leave a training session room."""
            await self.sio.leave_room(sid, session_id)
            logger.info(f"Client {sid} left session room: {session_id}")
            
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['socket_connected'] = False
        
        @self.sio.event
        async def get_session_status(sid, session_id):
            """Get current session status."""
            if session_id in self.active_sessions:
                await self.sio.emit('session_status', {
                    'sessionId': session_id,
                    'exists': True,
                    'data': self.active_sessions[session_id]
                }, room=sid)
            else:
                await self.sio.emit('session_status', {
                    'sessionId': session_id,
                    'exists': False
                }, room=sid)
    
    def _setup_http_routes(self):
        """Setup HTTP fallback routes for compatibility."""
        
        async def health_check(request):
            return web.json_response({
                'status': 'healthy',
                'websocket_enabled': True,
                'active_sessions': len(self.active_sessions),
                'timestamp': datetime.now().isoformat()
            })
        
        async def get_session_progress(request):
            """HTTP fallback for progress polling."""
            session_id = request.match_info.get('session_id')
            
            if session_id and session_id in self.active_sessions:
                return web.json_response({
                    'sessionId': session_id,
                    'metrics': self.active_sessions[session_id]['metrics'],
                    'status': 'running',
                    'websocket_available': self.active_sessions[session_id]['socket_connected'],
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return web.json_response({
                    'error': 'Session not found',
                    'sessionId': session_id
                }, status=404)
        
        # Add routes
        self.app.router.add_get('/health', health_check)
        self.app.router.add_get('/api/progress/{session_id}', get_session_progress)
    
    async def emit_progress(self, session_id: str, event_type: str, metrics: ProgressMetrics, 
                           status: str = "running", message: str = None):
        """Emit progress update to WebSocket clients."""
        
        # Update session data
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                'created_at': datetime.now().isoformat(),
                'socket_connected': False,
                'socket_id': None
            }
        
        self.active_sessions[session_id].update({
            'metrics': metrics.to_dict(),
            'last_update': datetime.now().isoformat(),
            'status': status
        })
        
        # Create progress event
        event = ProgressEvent(
            sessionId=session_id,
            eventType=event_type,
            metrics=metrics,
            status=status,
            message=message
        )
        
        # Emit to WebSocket clients in the session room
        await self.sio.emit('progress_update', event.to_dict(), room=session_id)
        
        # Log progress
        logger.info(f"Progress update for {session_id}: {event_type} - Step {metrics.currentStep}/{metrics.totalSteps}")
    
    async def emit_error(self, session_id: str, error_message: str, error_type: str = "training_error"):
        """Emit error event to WebSocket clients."""
        
        error_event = {
            'sessionId': session_id,
            'eventType': error_type,
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'status': 'error'
        }
        
        # Update session status
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['status'] = 'error'
            self.active_sessions[session_id]['error'] = error_message
        
        await self.sio.emit('training_error', error_event, room=session_id)
        logger.error(f"Error in session {session_id}: {error_message}")
    
    def create_progress_emitter(self, session_id: str):
        """Create a progress emitter function for integration with training loops."""
        
        class ProgressEmitter:
            def __init__(self, server, session_id):
                self.server = server
                self.session_id = session_id
            
            async def emit_step_update(self, step: int, loss: float, model_idx: int = 0, **kwargs):
                """Emit training step update."""
                metrics = ProgressMetrics(
                    currentStep=step,
                    loss=loss,
                    currentModel=model_idx + 1,
                    **kwargs
                )
                await self.server.emit_progress(
                    self.session_id, 
                    "step_update", 
                    metrics, 
                    "running"
                )
            
            async def emit_model_completed(self, model_idx: int, final_loss: float, **kwargs):
                """Emit model completion."""
                metrics = ProgressMetrics(
                    currentModel=model_idx + 1,
                    loss=final_loss,
                    **kwargs
                )
                await self.server.emit_progress(
                    self.session_id,
                    "model_completed",
                    metrics,
                    "running",
                    f"Model {model_idx + 1} completed"
                )
            
            async def emit_phase_completed(self, phase_name: str, **kwargs):
                """Emit phase completion."""
                metrics = ProgressMetrics(**kwargs)
                await self.server.emit_progress(
                    self.session_id,
                    "phase_completed",
                    metrics,
                    "completed",
                    f"Phase {phase_name} completed"
                )
            
            async def emit_training_started(self, total_steps: int, total_models: int, **kwargs):
                """Emit training start."""
                metrics = ProgressMetrics(
                    totalSteps=total_steps,
                    totalModels=total_models,
                    **kwargs
                )
                await self.server.emit_progress(
                    self.session_id,
                    "training_started",
                    metrics,
                    "running",
                    "Training started"
                )
            
            async def emit_error(self, error_message: str):
                """Emit training error."""
                await self.server.emit_error(self.session_id, error_message)
        
        return ProgressEmitter(self, session_id)
    
    async def start_server(self):
        """Start the WebSocket server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        logger.info(f"WebSocket Progress Server started on port {self.port}")
        logger.info(f"Health check: http://localhost:{self.port}/health")
        logger.info(f"WebSocket endpoint: ws://localhost:{self.port}/socket.io/")
    
    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session data for HTTP fallback."""
        return self.active_sessions.get(session_id)


# Global server instance
websocket_server: Optional[WebSocketProgressServer] = None


async def initialize_websocket_server(port: int = 8001) -> WebSocketProgressServer:
    """Initialize and start the WebSocket server."""
    global websocket_server
    
    if websocket_server is None:
        websocket_server = WebSocketProgressServer(port)
        await websocket_server.start_server()
    
    return websocket_server


def get_websocket_server() -> Optional[WebSocketProgressServer]:
    """Get the global WebSocket server instance."""
    return websocket_server


if __name__ == '__main__':
    async def main():
        server = await initialize_websocket_server(8001)
        
        # Keep server running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down WebSocket server...")
    
    # Run the server
    asyncio.run(main())
