"""
WebSocket API Routes

Real-time event streaming endpoints for pipeline monitoring.
"""

import asyncio
import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse

from ..websocket.connection_manager import ConnectionManager
from ..models.pipeline_models import (
    WebSocketEvent,
    PipelineProgressEvent,
    AgentUpdateEvent,
    PhaseCompletionEvent,
    MetricsStreamEvent,
)

router = APIRouter(tags=["websocket"])

# Initialize connection manager
connection_manager = ConnectionManager()


@router.websocket("/ws/agents")
async def websocket_agents(websocket: WebSocket, session_id: Optional[str] = None):
    """
    WebSocket endpoint for agent status updates.

    Streams:
        - Agent state changes
        - Task assignments
        - Resource utilization
    """
    await connection_manager.connect(websocket, "agents", session_id)

    try:
        while True:
            # Keep connection alive and listen for client messages
            data = await websocket.receive_text()

            # Handle client commands (optional)
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


@router.websocket("/ws/tasks")
async def websocket_tasks(websocket: WebSocket, session_id: Optional[str] = None):
    """
    WebSocket endpoint for task execution updates.

    Streams:
        - Task progress
        - Phase transitions
        - Completion events
    """
    await connection_manager.connect(websocket, "tasks", session_id)

    try:
        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


@router.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket, session_id: Optional[str] = None):
    """
    WebSocket endpoint for performance metrics streaming.

    Streams:
        - CPU/GPU utilization
        - Memory usage
        - Throughput metrics
        - Custom performance indicators
    """
    await connection_manager.connect(websocket, "metrics", session_id)

    try:
        # Start metrics streaming
        while True:
            # Simulate metrics (replace with actual metrics collection)
            metrics_event = MetricsStreamEvent(
                session_id=session_id,
                data={
                    "metrics": {
                        "cpu_percent": 45.2,
                        "memory_mb": 2048.5,
                        "gpu_utilization": 78.3,
                        "throughput_ops": 1250.0,
                    }
                },
            )

            await websocket.send_json(metrics_event.dict())
            await asyncio.sleep(1)  # Update every second

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


@router.websocket("/ws/pipeline")
async def websocket_pipeline(websocket: WebSocket, session_id: Optional[str] = None):
    """
    WebSocket endpoint for pipeline progress updates.

    Streams:
        - Phase progress
        - Overall pipeline status
        - Quality gate results
    """
    await connection_manager.connect(websocket, "pipeline", session_id)

    try:
        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


@router.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket, session_id: Optional[str] = None):
    """
    WebSocket endpoint for combined dashboard data.

    Streams:
        - Pipeline status
        - Agent status
        - Real-time metrics
        - Quality indicators
    """
    await connection_manager.connect(websocket, "dashboard", session_id)

    try:
        while True:
            # Send combined dashboard update
            dashboard_data = {
                "event_type": "dashboard_update",
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": session_id,
                "data": {
                    "pipeline_status": "running",
                    "current_phase": "training",
                    "progress_percent": 65.5,
                    "active_agents": 12,
                    "metrics": {
                        "cpu_percent": 45.2,
                        "memory_mb": 2048.5,
                        "gpu_utilization": 78.3,
                    },
                },
            }

            await websocket.send_json(dashboard_data)
            await asyncio.sleep(2)  # Update every 2 seconds

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


@router.get("/ws/stats")
async def get_websocket_stats():
    """
    Get WebSocket connection statistics.
    """
    return connection_manager.get_stats()


@router.get("/ws/client", response_class=HTMLResponse)
async def websocket_test_client():
    """
    Simple WebSocket test client for development.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Agent Forge WebSocket Client</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                background: white;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
            }
            h2 {
                color: #666;
                margin-top: 0;
            }
            .controls {
                display: flex;
                gap: 10px;
                margin-bottom: 10px;
            }
            button {
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
            }
            .connect {
                background: #4CAF50;
                color: white;
            }
            .disconnect {
                background: #f44336;
                color: white;
            }
            .status {
                padding: 10px;
                border-radius: 4px;
                margin-bottom: 10px;
            }
            .status.connected {
                background: #d4edda;
                color: #155724;
            }
            .status.disconnected {
                background: #f8d7da;
                color: #721c24;
            }
            #messages {
                height: 400px;
                overflow-y: auto;
                border: 1px solid #ddd;
                padding: 10px;
                background: #fafafa;
                font-family: monospace;
                font-size: 12px;
            }
            .message {
                padding: 5px;
                margin-bottom: 5px;
                border-left: 3px solid #2196F3;
                background: white;
            }
            select, input {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
        <h1>Agent Forge WebSocket Test Client</h1>

        <div class="container">
            <h2>Connection</h2>
            <div class="controls">
                <select id="channel">
                    <option value="agents">Agents</option>
                    <option value="tasks">Tasks</option>
                    <option value="metrics">Metrics</option>
                    <option value="pipeline">Pipeline</option>
                    <option value="dashboard">Dashboard</option>
                </select>
                <input type="text" id="session_id" placeholder="Session ID (optional)">
                <button class="connect" onclick="connect()">Connect</button>
                <button class="disconnect" onclick="disconnect()">Disconnect</button>
            </div>
            <div id="status" class="status disconnected">Disconnected</div>
        </div>

        <div class="container">
            <h2>Messages</h2>
            <div id="messages"></div>
        </div>

        <script>
            let ws = null;

            function connect() {
                const channel = document.getElementById('channel').value;
                const sessionId = document.getElementById('session_id').value;

                let url = `ws://${window.location.host}/ws/${channel}`;
                if (sessionId) {
                    url += `?session_id=${sessionId}`;
                }

                ws = new WebSocket(url);

                ws.onopen = function() {
                    document.getElementById('status').className = 'status connected';
                    document.getElementById('status').textContent = `Connected to ${channel}`;
                    addMessage('Connected to ' + url, 'info');
                };

                ws.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        addMessage(JSON.stringify(data, null, 2), 'data');
                    } catch (e) {
                        addMessage(event.data, 'raw');
                    }
                };

                ws.onclose = function() {
                    document.getElementById('status').className = 'status disconnected';
                    document.getElementById('status').textContent = 'Disconnected';
                    addMessage('Connection closed', 'info');
                };

                ws.onerror = function(error) {
                    addMessage('Error: ' + error, 'error');
                };
            }

            function disconnect() {
                if (ws) {
                    ws.close();
                }
            }

            function addMessage(text, type) {
                const messagesDiv = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message';
                messageDiv.textContent = new Date().toISOString() + ' - ' + text;
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            // Ping every 30 seconds to keep connection alive
            setInterval(() => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'ping'}));
                }
            }, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# Helper function to broadcast events from service layer
async def broadcast_pipeline_event(event: WebSocketEvent, channels: list = None):
    """
    Helper function to broadcast events from service layer.

    Usage:
        from .routes.websocket_routes import broadcast_pipeline_event
        await broadcast_pipeline_event(PipelineProgressEvent(...))
    """
    await connection_manager.broadcast_event(event, channels)