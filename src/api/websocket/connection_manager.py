"""
WebSocket Connection Manager

Manages WebSocket connections and real-time event broadcasting.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set

from fastapi import WebSocket, WebSocketDisconnect

from ..models.pipeline_models import WebSocketEvent

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and event broadcasting."""

    def __init__(self):
        # Active connections by channel
        self.active_connections: Dict[str, List[WebSocket]] = {
            "agents": [],
            "tasks": [],
            "metrics": [],
            "pipeline": [],
            "dashboard": [],
        }

        # Session-specific connections
        self.session_connections: Dict[str, Set[WebSocket]] = {}

        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict] = {}

        self.logger = logging.getLogger(__name__)

    async def connect(
        self, websocket: WebSocket, channel: str, session_id: str = None
    ):
        """Accept new WebSocket connection."""

        await websocket.accept()

        # Add to channel
        if channel in self.active_connections:
            self.active_connections[channel].append(websocket)
        else:
            self.active_connections[channel] = [websocket]

        # Add to session tracking
        if session_id:
            if session_id not in self.session_connections:
                self.session_connections[session_id] = set()
            self.session_connections[session_id].add(websocket)

        # Store metadata
        self.connection_metadata[websocket] = {
            "channel": channel,
            "session_id": session_id,
            "connected_at": datetime.utcnow(),
        }

        self.logger.info(
            f"WebSocket connected: channel={channel}, session={session_id}"
        )

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""

        metadata = self.connection_metadata.get(websocket)
        if not metadata:
            return

        channel = metadata["channel"]
        session_id = metadata.get("session_id")

        # Remove from channel
        if channel in self.active_connections:
            try:
                self.active_connections[channel].remove(websocket)
            except ValueError:
                pass

        # Remove from session tracking
        if session_id and session_id in self.session_connections:
            self.session_connections[session_id].discard(websocket)
            if not self.session_connections[session_id]:
                del self.session_connections[session_id]

        # Remove metadata
        del self.connection_metadata[websocket]

        self.logger.info(
            f"WebSocket disconnected: channel={channel}, session={session_id}"
        )

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific connection."""

        try:
            await websocket.send_text(message)
        except Exception as e:
            self.logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast_to_channel(self, channel: str, message: str):
        """Broadcast message to all connections in a channel."""

        connections = self.active_connections.get(channel, [])
        disconnected = []

        for connection in connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                self.logger.error(f"Error broadcasting to {channel}: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_to_session(self, session_id: str, message: str):
        """Broadcast message to all connections for a specific session."""

        connections = self.session_connections.get(session_id, set())
        disconnected = []

        for connection in connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                self.logger.error(f"Error broadcasting to session {session_id}: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_event(self, event: WebSocketEvent, channels: List[str] = None):
        """Broadcast event to specified channels."""

        event_json = json.dumps(event.dict(), default=str)

        target_channels = channels or list(self.active_connections.keys())

        for channel in target_channels:
            await self.broadcast_to_channel(channel, event_json)

        # Also broadcast to session if specified
        if event.session_id:
            await self.broadcast_to_session(event.session_id, event_json)

    def get_connection_count(self, channel: str = None) -> int:
        """Get number of active connections."""

        if channel:
            return len(self.active_connections.get(channel, []))
        return sum(len(conns) for conns in self.active_connections.values())

    def get_session_count(self) -> int:
        """Get number of sessions with active connections."""

        return len(self.session_connections)

    def get_stats(self) -> Dict:
        """Get connection statistics."""

        return {
            "total_connections": self.get_connection_count(),
            "active_sessions": self.get_session_count(),
            "channels": {
                channel: len(conns)
                for channel, conns in self.active_connections.items()
            },
        }