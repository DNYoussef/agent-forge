"""
Integration tests for Pipeline API endpoints.

Tests the complete REST API surface including:
- Pipeline lifecycle (start/stop/pause/resume)
- Configuration management
- Quality gates
- Checkpoints
- Execution history
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any
import pytest
from httpx import AsyncClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from api.main import app
from api.models.pipeline_models import PipelinePhase, SwarmTopology


class TestPipelineLifecycle:
    """Test pipeline start, control, and monitoring."""

    @pytest.mark.asyncio
    async def test_start_pipeline_success(self):
        """Test successful pipeline initialization."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": ["cognate", "evomerge", "quietstar"],
                    "config": {
                        "cognate": {"base_models": ["model1", "model2"]},
                        "evomerge": {"population_size": 10, "generations": 5}
                    },
                    "enable_monitoring": True,
                    "enable_checkpoints": True,
                    "swarm_topology": "hierarchical",
                    "max_agents": 50
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data
            assert data["status"] == "started"
            assert len(data["session_id"]) > 0

    @pytest.mark.asyncio
    async def test_start_pipeline_invalid_phases(self):
        """Test pipeline start with invalid phase names."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": ["invalid_phase"],
                    "swarm_topology": "hierarchical"
                }
            )

            assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_pipeline_pause_resume(self):
        """Test pipeline pause and resume functionality."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Start pipeline
            start_response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": ["cognate"],
                    "swarm_topology": "mesh"
                }
            )
            session_id = start_response.json()["session_id"]

            # Pause pipeline
            pause_response = await client.post(
                "/api/v1/pipeline/control",
                json={
                    "session_id": session_id,
                    "action": "pause",
                    "force": False
                }
            )

            assert pause_response.status_code == 200
            assert pause_response.json()["action"] == "pause"

            # Resume pipeline
            resume_response = await client.post(
                "/api/v1/pipeline/control",
                json={
                    "session_id": session_id,
                    "action": "resume",
                    "force": False
                }
            )

            assert resume_response.status_code == 200
            assert resume_response.json()["action"] == "resume"

    @pytest.mark.asyncio
    async def test_pipeline_stop_graceful(self):
        """Test graceful pipeline termination."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Start pipeline
            start_response = await client.post(
                "/api/v1/pipeline/start",
                json={"phases": ["cognate"]}
            )
            session_id = start_response.json()["session_id"]

            # Stop pipeline
            stop_response = await client.post(
                "/api/v1/pipeline/control",
                json={
                    "session_id": session_id,
                    "action": "stop",
                    "force": False
                }
            )

            assert stop_response.status_code == 200
            assert stop_response.json()["status"] == "success"

    @pytest.mark.asyncio
    async def test_pipeline_cancel_force(self):
        """Test force cancel of pipeline execution."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Start pipeline
            start_response = await client.post(
                "/api/v1/pipeline/start",
                json={"phases": ["training"]}
            )
            session_id = start_response.json()["session_id"]

            # Force cancel
            cancel_response = await client.post(
                "/api/v1/pipeline/control",
                json={
                    "session_id": session_id,
                    "action": "cancel",
                    "force": True
                }
            )

            assert cancel_response.status_code == 200


class TestPipelineStatus:
    """Test pipeline status and monitoring endpoints."""

    @pytest.mark.asyncio
    async def test_get_pipeline_status(self):
        """Test retrieving pipeline status."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Start pipeline
            start_response = await client.post(
                "/api/v1/pipeline/start",
                json={"phases": ["cognate", "evomerge"]}
            )
            session_id = start_response.json()["session_id"]

            # Get status
            status_response = await client.get(
                f"/api/v1/pipeline/status/{session_id}"
            )

            assert status_response.status_code == 200
            status_data = status_response.json()

            # Verify status structure
            assert "session_id" in status_data
            assert "status" in status_data
            assert "current_phase" in status_data
            assert "progress_percent" in status_data
            assert "phases" in status_data
            assert "metrics" in status_data

    @pytest.mark.asyncio
    async def test_get_swarm_status(self):
        """Test swarm coordination status retrieval."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Start pipeline with swarm
            start_response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": ["cognate"],
                    "swarm_topology": "mesh",
                    "max_agents": 25
                }
            )
            session_id = start_response.json()["session_id"]

            # Get swarm status
            swarm_response = await client.get(
                f"/api/v1/pipeline/swarm/{session_id}"
            )

            assert swarm_response.status_code == 200
            swarm_data = swarm_response.json()

            assert "session_id" in swarm_data
            assert "topology" in swarm_data
            assert "active_agents" in swarm_data
            assert "agent_distribution" in swarm_data

    @pytest.mark.asyncio
    async def test_status_not_found(self):
        """Test status request for non-existent session."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/pipeline/status/invalid_session_id"
            )

            assert response.status_code == 404


class TestQualityGates:
    """Test quality gate validation."""

    @pytest.mark.asyncio
    async def test_validate_quality_gates_all_phases(self):
        """Test quality gate validation for all phases."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Start pipeline
            start_response = await client.post(
                "/api/v1/pipeline/start",
                json={"phases": ["cognate", "evomerge"]}
            )
            session_id = start_response.json()["session_id"]

            # Validate quality gates
            gate_response = await client.post(
                f"/api/v1/pipeline/quality-gates/{session_id}"
            )

            assert gate_response.status_code == 200
            gate_data = gate_response.json()

            assert "session_id" in gate_data
            assert "results" in gate_data
            assert "overall_passed" in gate_data
            assert isinstance(gate_data["results"], list)

    @pytest.mark.asyncio
    async def test_validate_quality_gates_specific_phase(self):
        """Test quality gate validation for specific phase."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Start pipeline
            start_response = await client.post(
                "/api/v1/pipeline/start",
                json={"phases": ["training"]}
            )
            session_id = start_response.json()["session_id"]

            # Validate specific phase
            gate_response = await client.post(
                f"/api/v1/pipeline/quality-gates/{session_id}",
                params={"phase": "training"}
            )

            assert gate_response.status_code == 200
            gate_data = gate_response.json()

            # Verify phase-specific validation
            assert gate_data["results"][0]["phase"] == "training"


class TestCheckpoints:
    """Test checkpoint save and restore functionality."""

    @pytest.mark.asyncio
    async def test_save_checkpoint(self):
        """Test checkpoint creation."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Start pipeline
            start_response = await client.post(
                "/api/v1/pipeline/start",
                json={"phases": ["cognate", "evomerge"]}
            )
            session_id = start_response.json()["session_id"]

            # Save checkpoint
            checkpoint_response = await client.post(
                "/api/v1/pipeline/checkpoint/save",
                json={
                    "session_id": session_id,
                    "checkpoint_name": "test_checkpoint",
                    "include_model_state": True,
                    "include_swarm_state": True
                }
            )

            assert checkpoint_response.status_code == 200
            checkpoint_data = checkpoint_response.json()

            assert "checkpoint_id" in checkpoint_data
            assert checkpoint_data["model_included"] is True
            assert checkpoint_data["swarm_included"] is True

    @pytest.mark.asyncio
    async def test_load_checkpoint(self):
        """Test checkpoint restoration."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Start and save checkpoint
            start_response = await client.post(
                "/api/v1/pipeline/start",
                json={"phases": ["cognate"]}
            )
            session_id = start_response.json()["session_id"]

            checkpoint_response = await client.post(
                "/api/v1/pipeline/checkpoint/save",
                json={
                    "session_id": session_id,
                    "include_model_state": True,
                    "include_swarm_state": True
                }
            )
            checkpoint_id = checkpoint_response.json()["checkpoint_id"]

            # Load checkpoint
            load_response = await client.post(
                f"/api/v1/pipeline/checkpoint/load/{checkpoint_id}"
            )

            assert load_response.status_code == 200
            assert load_response.json()["status"] == "loaded"


class TestConfigurationPresets:
    """Test configuration preset management."""

    @pytest.mark.asyncio
    async def test_save_preset(self):
        """Test saving configuration preset."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            preset_config = {
                "phases": ["cognate", "evomerge", "quietstar"],
                "swarm_topology": "hierarchical",
                "max_agents": 50,
                "phase_config": {
                    "cognate": {"base_models": ["model1", "model2"]},
                    "evomerge": {"population_size": 20}
                }
            }

            response = await client.post(
                "/api/v1/pipeline/preset/save",
                json={
                    "preset_name": "test_preset",
                    "config": preset_config
                }
            )

            assert response.status_code == 200
            preset_data = response.json()
            assert preset_data["preset_name"] == "test_preset"

    @pytest.mark.asyncio
    async def test_load_preset(self):
        """Test loading configuration preset."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Save preset first
            preset_config = {"phases": ["cognate"]}
            await client.post(
                "/api/v1/pipeline/preset/save",
                json={
                    "preset_name": "load_test_preset",
                    "config": preset_config
                }
            )

            # Load preset
            load_response = await client.get(
                "/api/v1/pipeline/preset/load_test_preset"
            )

            assert load_response.status_code == 200
            loaded_data = load_response.json()
            assert loaded_data["config"]["phases"] == ["cognate"]

    @pytest.mark.asyncio
    async def test_list_presets(self):
        """Test listing all presets."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/pipeline/presets")

            assert response.status_code == 200
            assert isinstance(response.json(), list)


class TestExecutionHistory:
    """Test execution history tracking."""

    @pytest.mark.asyncio
    async def test_get_execution_history(self):
        """Test retrieving execution history."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Run a few pipelines
            for _ in range(3):
                await client.post(
                    "/api/v1/pipeline/start",
                    json={"phases": ["cognate"]}
                )

            # Get history
            history_response = await client.get(
                "/api/v1/pipeline/history",
                params={"limit": 10}
            )

            assert history_response.status_code == 200
            history = history_response.json()
            assert isinstance(history, list)
            assert len(history) > 0

    @pytest.mark.asyncio
    async def test_filter_execution_history_by_status(self):
        """Test filtering history by execution status."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            history_response = await client.get(
                "/api/v1/pipeline/history",
                params={"limit": 10, "status": "completed"}
            )

            assert history_response.status_code == 200
            history = history_response.json()

            # Verify all results match filter
            for entry in history:
                assert entry["status"] == "completed"


class TestHealthCheck:
    """Test API health monitoring."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check endpoint."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/pipeline/health")

            assert response.status_code == 200
            health_data = response.json()

            assert health_data["status"] == "healthy"
            assert "version" in health_data
            assert "active_sessions" in health_data
            assert "total_memory_mb" in health_data
            assert "available_memory_mb" in health_data


class TestAPIInfo:
    """Test API information endpoint."""

    @pytest.mark.asyncio
    async def test_api_info(self):
        """Test API info endpoint."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/info")

            assert response.status_code == 200
            info = response.json()

            assert info["name"] == "Agent Forge API"
            assert "version" in info
            assert "features" in info
            assert "endpoints" in info
            assert isinstance(info["features"], list)


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()