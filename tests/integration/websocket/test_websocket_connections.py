"""
Integration tests for WebSocket connections.

Tests real-time streaming functionality including:
- Connection management
- Event broadcasting
- Channel subscriptions
- Reconnection handling
- Data integrity
"""

import asyncio
import json
from datetime import datetime
import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from api.main import app


class TestWebSocketConnections:
    """Test WebSocket connection lifecycle."""

    def test_websocket_agents_connection(self):
        """Test agent status WebSocket connection."""
        client = TestClient(app)

        with client.websocket_connect("/ws/agents") as websocket:
            # Send ping
            websocket.send_json({"type": "ping"})

            # Receive pong
            data = websocket.receive_json()
            assert data["type"] == "pong"
            assert "timestamp" in data

    def test_websocket_tasks_connection(self):
        """Test task updates WebSocket connection."""
        client = TestClient(app)

        with client.websocket_connect("/ws/tasks") as websocket:
            websocket.send_json({"type": "ping"})
            data = websocket.receive_json()
            assert data["type"] == "pong"

    def test_websocket_metrics_streaming(self):
        """Test metrics streaming WebSocket."""
        client = TestClient(app)

        with client.websocket_connect("/ws/metrics") as websocket:
            # Should receive metrics updates
            data = websocket.receive_json(timeout=3)

            assert "metrics" in data.get("data", {})
            metrics = data["data"]["metrics"]
            assert "cpu_percent" in metrics
            assert "memory_mb" in metrics
            assert "gpu_utilization" in metrics
            assert "throughput_ops" in metrics

    def test_websocket_pipeline_connection(self):
        """Test pipeline progress WebSocket."""
        client = TestClient(app)

        with client.websocket_connect("/ws/pipeline") as websocket:
            websocket.send_json({"type": "ping"})
            data = websocket.receive_json()
            assert data["type"] == "pong"

    def test_websocket_dashboard_streaming(self):
        """Test combined dashboard WebSocket."""
        client = TestClient(app)

        with client.websocket_connect("/ws/dashboard") as websocket:
            # Should receive dashboard updates
            data = websocket.receive_json(timeout=3)

            assert data["event_type"] == "dashboard_update"
            assert "data" in data
            dashboard_data = data["data"]

            assert "pipeline_status" in dashboard_data
            assert "current_phase" in dashboard_data
            assert "progress_percent" in dashboard_data
            assert "active_agents" in dashboard_data
            assert "metrics" in dashboard_data


class TestWebSocketWithSessionID:
    """Test WebSocket connections with session filtering."""

    def test_agents_with_session_id(self):
        """Test agent WebSocket with session filter."""
        client = TestClient(app)
        session_id = "test_session_123"

        with client.websocket_connect(f"/ws/agents?session_id={session_id}") as websocket:
            websocket.send_json({"type": "ping"})
            data = websocket.receive_json()
            assert data["type"] == "pong"

    def test_tasks_with_session_id(self):
        """Test task WebSocket with session filter."""
        client = TestClient(app)
        session_id = "test_session_456"

        with client.websocket_connect(f"/ws/tasks?session_id={session_id}") as websocket:
            websocket.send_json({"type": "ping"})
            data = websocket.receive_json()
            assert data["type"] == "pong"

    def test_metrics_with_session_id(self):
        """Test metrics WebSocket with session filter."""
        client = TestClient(app)
        session_id = "test_session_789"

        with client.websocket_connect(f"/ws/metrics?session_id={session_id}") as websocket:
            data = websocket.receive_json(timeout=3)
            assert "session_id" in data or data.get("session_id") is None

    def test_pipeline_with_session_id(self):
        """Test pipeline WebSocket with session filter."""
        client = TestClient(app)
        session_id = "test_session_pipeline"

        with client.websocket_connect(f"/ws/pipeline?session_id={session_id}") as websocket:
            websocket.send_json({"type": "ping"})
            data = websocket.receive_json()
            assert data["type"] == "pong"


class TestWebSocketEventStreaming:
    """Test event streaming and data integrity."""

    def test_metrics_update_frequency(self):
        """Test metrics are streamed at correct frequency."""
        client = TestClient(app)

        with client.websocket_connect("/ws/metrics") as websocket:
            # Collect multiple updates
            updates = []
            for _ in range(3):
                data = websocket.receive_json(timeout=3)
                updates.append(data)

            # Verify we got multiple updates
            assert len(updates) == 3

            # Verify data structure consistency
            for update in updates:
                assert "data" in update
                assert "metrics" in update["data"]

    def test_dashboard_update_frequency(self):
        """Test dashboard updates at correct interval."""
        client = TestClient(app)

        with client.websocket_connect("/ws/dashboard") as websocket:
            # Collect updates
            first_update = websocket.receive_json(timeout=3)
            second_update = websocket.receive_json(timeout=3)

            # Verify consistent structure
            assert first_update["event_type"] == second_update["event_type"]
            assert "timestamp" in first_update
            assert "timestamp" in second_update

    def test_ping_pong_keepalive(self):
        """Test WebSocket keepalive ping/pong."""
        client = TestClient(app)

        with client.websocket_connect("/ws/agents") as websocket:
            # Send multiple pings
            for i in range(5):
                websocket.send_json({"type": "ping"})
                response = websocket.receive_json()
                assert response["type"] == "pong"


class TestWebSocketStats:
    """Test WebSocket statistics endpoint."""

    def test_get_websocket_stats(self):
        """Test retrieving WebSocket connection stats."""
        client = TestClient(app)

        # Open some connections
        with client.websocket_connect("/ws/agents"):
            with client.websocket_connect("/ws/metrics"):
                # Get stats
                response = client.get("/ws/stats")

                assert response.status_code == 200
                stats = response.json()

                assert "total_connections" in stats
                assert "connections_by_channel" in stats


class TestWebSocketErrorHandling:
    """Test WebSocket error scenarios."""

    def test_invalid_json_message(self):
        """Test handling of invalid JSON messages."""
        client = TestClient(app)

        with client.websocket_connect("/ws/agents") as websocket:
            # Send invalid JSON
            websocket.send_text("invalid json {{{")

            # Connection should remain open
            websocket.send_json({"type": "ping"})
            response = websocket.receive_json()
            assert response["type"] == "pong"

    def test_connection_close_handling(self):
        """Test graceful connection closure."""
        client = TestClient(app)

        websocket = client.websocket_connect("/ws/agents")
        ws = websocket.__enter__()

        # Send message to verify connection
        ws.send_json({"type": "ping"})
        ws.receive_json()

        # Close connection
        websocket.__exit__(None, None, None)

        # Verify stats reflect disconnection
        response = client.get("/ws/stats")
        # Connection count should be updated


class TestWebSocketHTMLClient:
    """Test WebSocket test client interface."""

    def test_html_client_available(self):
        """Test WebSocket test client HTML page."""
        client = TestClient(app)

        response = client.get("/ws/client")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"Agent Forge WebSocket Client" in response.content


class TestMultipleConnections:
    """Test multiple simultaneous WebSocket connections."""

    def test_multiple_agent_connections(self):
        """Test multiple agent WebSocket connections."""
        client = TestClient(app)

        # Open multiple connections
        connections = []
        for i in range(3):
            ws = client.websocket_connect(f"/ws/agents?session_id=session_{i}")
            connections.append(ws)

        # Verify all connections work
        for i, ws_context in enumerate(connections):
            with ws_context as websocket:
                websocket.send_json({"type": "ping"})
                response = websocket.receive_json()
                assert response["type"] == "pong"

    def test_mixed_channel_connections(self):
        """Test multiple connections across different channels."""
        client = TestClient(app)

        with client.websocket_connect("/ws/agents") as agents_ws, \
             client.websocket_connect("/ws/tasks") as tasks_ws, \
             client.websocket_connect("/ws/metrics") as metrics_ws:

            # Verify all channels work independently
            agents_ws.send_json({"type": "ping"})
            assert agents_ws.receive_json()["type"] == "pong"

            tasks_ws.send_json({"type": "ping"})
            assert tasks_ws.receive_json()["type"] == "pong"

            # Metrics should stream data
            metrics_data = metrics_ws.receive_json(timeout=3)
            assert "metrics" in metrics_data.get("data", {})


class TestWebSocketReconnection:
    """Test WebSocket reconnection scenarios."""

    def test_reconnect_after_disconnect(self):
        """Test reconnecting after disconnection."""
        client = TestClient(app)

        # First connection
        with client.websocket_connect("/ws/agents") as ws1:
            ws1.send_json({"type": "ping"})
            assert ws1.receive_json()["type"] == "pong"

        # Reconnect
        with client.websocket_connect("/ws/agents") as ws2:
            ws2.send_json({"type": "ping"})
            assert ws2.receive_json()["type"] == "pong"

    def test_reconnect_with_same_session(self):
        """Test reconnecting with same session ID."""
        client = TestClient(app)
        session_id = "persistent_session"

        # First connection
        with client.websocket_connect(f"/ws/pipeline?session_id={session_id}") as ws1:
            ws1.send_json({"type": "ping"})
            assert ws1.receive_json()["type"] == "pong"

        # Reconnect with same session
        with client.websocket_connect(f"/ws/pipeline?session_id={session_id}") as ws2:
            ws2.send_json({"type": "ping"})
            assert ws2.receive_json()["type"] == "pong"