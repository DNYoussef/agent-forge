"""
Test utilities and helper functions for integration testing.

Provides common functionality for:
- API client setup
- WebSocket testing
- Mock data management
- Assertion helpers
- Test fixtures
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Callable
from contextlib import asynccontextmanager
from datetime import datetime
import websockets
from httpx import AsyncClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from api.main import app


class APITestClient:
    """Helper class for API testing."""

    def __init__(self, base_url: str = "http://test"):
        self.base_url = base_url
        self.client: Optional[AsyncClient] = None

    @asynccontextmanager
    async def session(self):
        """Create async client session."""
        async with AsyncClient(app=app, base_url=self.base_url) as client:
            self.client = client
            yield client

    async def start_pipeline(
        self,
        phases: List[str],
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Start a pipeline and return session info."""
        async with self.session() as client:
            response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": phases,
                    "config": config or {},
                    **kwargs
                }
            )
            response.raise_for_status()
            return response.json()

    async def get_status(self, session_id: str) -> Dict[str, Any]:
        """Get pipeline status."""
        async with self.session() as client:
            response = await client.get(f"/api/v1/pipeline/status/{session_id}")
            response.raise_for_status()
            return response.json()

    async def control_pipeline(
        self,
        session_id: str,
        action: str,
        force: bool = False
    ) -> Dict[str, Any]:
        """Control pipeline execution."""
        async with self.session() as client:
            response = await client.post(
                "/api/v1/pipeline/control",
                json={
                    "session_id": session_id,
                    "action": action,
                    "force": force
                }
            )
            response.raise_for_status()
            return response.json()

    async def validate_quality_gates(
        self,
        session_id: str,
        phase: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate quality gates."""
        async with self.session() as client:
            params = {"phase": phase} if phase else {}
            response = await client.post(
                f"/api/v1/pipeline/quality-gates/{session_id}",
                params=params
            )
            response.raise_for_status()
            return response.json()

    async def save_checkpoint(
        self,
        session_id: str,
        checkpoint_name: Optional[str] = None,
        include_model: bool = True,
        include_swarm: bool = True
    ) -> Dict[str, Any]:
        """Save pipeline checkpoint."""
        async with self.session() as client:
            response = await client.post(
                "/api/v1/pipeline/checkpoint/save",
                json={
                    "session_id": session_id,
                    "checkpoint_name": checkpoint_name,
                    "include_model_state": include_model,
                    "include_swarm_state": include_swarm
                }
            )
            response.raise_for_status()
            return response.json()

    async def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Load pipeline checkpoint."""
        async with self.session() as client:
            response = await client.post(
                f"/api/v1/pipeline/checkpoint/load/{checkpoint_id}"
            )
            response.raise_for_status()
            return response.json()


class WebSocketTestClient:
    """Helper class for WebSocket testing."""

    def __init__(self, base_url: str = "ws://localhost:8000"):
        self.base_url = base_url
        self.messages: List[Dict[str, Any]] = []

    @asynccontextmanager
    async def connect(
        self,
        channel: str,
        session_id: Optional[str] = None
    ):
        """Connect to WebSocket channel."""
        url = f"{self.base_url}/ws/{channel}"
        if session_id:
            url += f"?session_id={session_id}"

        async with websockets.connect(url) as websocket:
            self.websocket = websocket
            yield websocket

    async def send_json(self, data: Dict[str, Any]):
        """Send JSON message."""
        await self.websocket.send(json.dumps(data))

    async def receive_json(self, timeout: float = 5.0) -> Dict[str, Any]:
        """Receive JSON message with timeout."""
        try:
            message = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=timeout
            )
            data = json.loads(message)
            self.messages.append(data)
            return data
        except asyncio.TimeoutError:
            raise TimeoutError(f"No message received within {timeout} seconds")

    async def collect_messages(
        self,
        duration: float = 5.0,
        count: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Collect messages for a duration or until count reached."""
        messages = []
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time

            if count and len(messages) >= count:
                break
            if elapsed >= duration:
                break

            try:
                msg = await self.receive_json(timeout=1.0)
                messages.append(msg)
            except TimeoutError:
                continue

        return messages

    async def wait_for_event(
        self,
        event_type: str,
        timeout: float = 10.0,
        predicate: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> Dict[str, Any]:
        """Wait for specific event type."""
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                msg = await self.receive_json(timeout=1.0)

                if msg.get("event_type") == event_type:
                    if predicate is None or predicate(msg):
                        return msg

            except TimeoutError:
                continue

        raise TimeoutError(f"Event '{event_type}' not received within {timeout} seconds")


class AssertionHelpers:
    """Common assertion helpers for integration tests."""

    @staticmethod
    def assert_pipeline_status(
        status: Dict[str, Any],
        expected_status: Optional[str] = None,
        expected_phase: Optional[str] = None
    ):
        """Assert pipeline status structure and values."""
        assert "session_id" in status
        assert "status" in status
        assert "current_phase" in status
        assert "progress_percent" in status
        assert "metrics" in status

        if expected_status:
            assert status["status"] == expected_status

        if expected_phase:
            assert status["current_phase"] == expected_phase

        assert 0 <= status["progress_percent"] <= 100

    @staticmethod
    def assert_swarm_status(
        status: Dict[str, Any],
        expected_topology: Optional[str] = None,
        min_agents: int = 0
    ):
        """Assert swarm status structure and values."""
        assert "session_id" in status
        assert "topology" in status
        assert "active_agents" in status
        assert "agent_distribution" in status

        if expected_topology:
            assert status["topology"] == expected_topology

        assert status["active_agents"] >= min_agents

    @staticmethod
    def assert_quality_gate_result(
        result: Dict[str, Any],
        should_pass: Optional[bool] = None
    ):
        """Assert quality gate result structure."""
        assert "session_id" in result
        assert "overall_passed" in result
        assert "results" in result
        assert isinstance(result["results"], list)

        if should_pass is not None:
            assert result["overall_passed"] == should_pass

        for gate in result["results"]:
            assert "phase" in gate
            assert "passed" in gate
            assert "metrics" in gate

    @staticmethod
    def assert_checkpoint(
        checkpoint: Dict[str, Any],
        should_include_model: Optional[bool] = None,
        should_include_swarm: Optional[bool] = None
    ):
        """Assert checkpoint structure."""
        assert "checkpoint_id" in checkpoint
        assert "session_id" in checkpoint
        assert "timestamp" in checkpoint
        assert "model_included" in checkpoint
        assert "swarm_included" in checkpoint

        if should_include_model is not None:
            assert checkpoint["model_included"] == should_include_model

        if should_include_swarm is not None:
            assert checkpoint["swarm_included"] == should_include_swarm

    @staticmethod
    def assert_websocket_event(
        event: Dict[str, Any],
        expected_type: Optional[str] = None,
        expected_session: Optional[str] = None
    ):
        """Assert WebSocket event structure."""
        assert "event_type" in event or "type" in event
        assert "timestamp" in event or "data" in event

        if expected_type:
            event_type = event.get("event_type") or event.get("type")
            assert event_type == expected_type

        if expected_session:
            assert event.get("session_id") == expected_session

    @staticmethod
    def assert_metrics(metrics: Dict[str, Any]):
        """Assert metrics structure and reasonable values."""
        required_fields = ["cpu_percent", "memory_mb"]

        for field in required_fields:
            if field in metrics:
                assert isinstance(metrics[field], (int, float))
                assert metrics[field] >= 0

        if "cpu_percent" in metrics:
            assert 0 <= metrics["cpu_percent"] <= 100

        if "gpu_utilization" in metrics:
            assert 0 <= metrics["gpu_utilization"] <= 100


class TestScenarios:
    """Pre-built test scenarios for common workflows."""

    @staticmethod
    async def run_single_phase_pipeline(
        client: APITestClient,
        phase: str,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Run a single-phase pipeline and return session ID."""
        result = await client.start_pipeline(
            phases=[phase],
            config={phase: config} if config else None
        )
        return result["session_id"]

    @staticmethod
    async def run_multi_phase_pipeline(
        client: APITestClient,
        phases: List[str],
        config: Optional[Dict[str, Any]] = None,
        enable_checkpoints: bool = True
    ) -> str:
        """Run multi-phase pipeline with checkpoints."""
        result = await client.start_pipeline(
            phases=phases,
            config=config,
            enable_checkpoints=enable_checkpoints
        )
        return result["session_id"]

    @staticmethod
    async def test_pause_resume_cycle(
        client: APITestClient,
        session_id: str
    ):
        """Test pause-resume cycle."""
        # Pause
        pause_result = await client.control_pipeline(session_id, "pause")
        assert pause_result["action"] == "pause"

        # Verify paused status
        status = await client.get_status(session_id)
        # Status should reflect pause (implementation-dependent)

        # Resume
        resume_result = await client.control_pipeline(session_id, "resume")
        assert resume_result["action"] == "resume"

    @staticmethod
    async def test_checkpoint_recovery(
        client: APITestClient,
        session_id: str
    ) -> str:
        """Test checkpoint save and load."""
        # Save checkpoint
        checkpoint = await client.save_checkpoint(
            session_id,
            checkpoint_name="test_recovery",
            include_model=True,
            include_swarm=True
        )

        checkpoint_id = checkpoint["checkpoint_id"]

        # Stop pipeline
        await client.control_pipeline(session_id, "stop")

        # Load checkpoint
        load_result = await client.load_checkpoint(checkpoint_id)
        assert load_result["status"] == "loaded"

        return checkpoint_id

    @staticmethod
    async def monitor_websocket_updates(
        ws_client: WebSocketTestClient,
        channel: str,
        session_id: Optional[str] = None,
        duration: float = 5.0
    ) -> List[Dict[str, Any]]:
        """Monitor WebSocket updates for duration."""
        async with ws_client.connect(channel, session_id):
            messages = await ws_client.collect_messages(duration=duration)
            return messages


class MockDataBuilder:
    """Builder for creating mock test data."""

    def __init__(self):
        self.data = {}

    def with_phases(self, phases: List[str]) -> 'MockDataBuilder':
        """Add phases to config."""
        self.data["phases"] = phases
        return self

    def with_swarm_topology(self, topology: str) -> 'MockDataBuilder':
        """Set swarm topology."""
        self.data["swarm_topology"] = topology
        return self

    def with_max_agents(self, count: int) -> 'MockDataBuilder':
        """Set max agents."""
        self.data["max_agents"] = count
        return self

    def with_phase_config(
        self,
        phase: str,
        config: Dict[str, Any]
    ) -> 'MockDataBuilder':
        """Add phase-specific config."""
        if "config" not in self.data:
            self.data["config"] = {}
        self.data["config"][phase] = config
        return self

    def with_monitoring(self, enabled: bool = True) -> 'MockDataBuilder':
        """Enable/disable monitoring."""
        self.data["enable_monitoring"] = enabled
        return self

    def with_checkpoints(self, enabled: bool = True) -> 'MockDataBuilder':
        """Enable/disable checkpoints."""
        self.data["enable_checkpoints"] = enabled
        return self

    def build(self) -> Dict[str, Any]:
        """Build final config."""
        return self.data


# Convenience functions for common operations

async def quick_start_pipeline(phases: List[str], **kwargs) -> str:
    """Quick helper to start a pipeline."""
    client = APITestClient()
    result = await client.start_pipeline(phases, **kwargs)
    return result["session_id"]


async def wait_for_completion(
    client: APITestClient,
    session_id: str,
    timeout: float = 30.0,
    poll_interval: float = 1.0
) -> Dict[str, Any]:
    """Wait for pipeline completion."""
    start_time = asyncio.get_event_loop().time()

    while asyncio.get_event_loop().time() - start_time < timeout:
        status = await client.get_status(session_id)

        if status["status"] in ["completed", "failed", "cancelled"]:
            return status

        await asyncio.sleep(poll_interval)

    raise TimeoutError(f"Pipeline did not complete within {timeout} seconds")