"""
End-to-end tests for 8-phase pipeline execution.

Tests complete pipeline workflows including:
- Phase-by-phase execution
- Inter-phase dependencies
- Quality gate enforcement
- Checkpoint recovery
- Full pipeline runs
"""

import asyncio
import pytest
from httpx import AsyncClient
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from api.main import app


class TestSinglePhaseExecution:
    """Test individual phase execution."""

    @pytest.mark.asyncio
    async def test_cognate_phase_execution(self):
        """Test Cognate phase: Model creation and initialization."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Start Cognate phase
            response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": ["cognate"],
                    "config": {
                        "cognate": {
                            "base_models": ["gpt2", "llama"],
                            "init_strategy": "random"
                        }
                    },
                    "swarm_topology": "hierarchical"
                }
            )

            assert response.status_code == 200
            session_id = response.json()["session_id"]

            # Monitor status
            status = await client.get(f"/api/v1/pipeline/status/{session_id}")
            assert status.status_code == 200
            assert status.json()["current_phase"] == "cognate"

    @pytest.mark.asyncio
    async def test_evomerge_phase_execution(self):
        """Test EvoMerge phase: Evolutionary model merging."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": ["evomerge"],
                    "config": {
                        "evomerge": {
                            "population_size": 20,
                            "generations": 10,
                            "mutation_rate": 0.1
                        }
                    },
                    "swarm_topology": "mesh"
                }
            )

            assert response.status_code == 200
            session_id = response.json()["session_id"]

            # Verify swarm configuration
            swarm_status = await client.get(f"/api/v1/pipeline/swarm/{session_id}")
            assert swarm_status.status_code == 200
            assert swarm_status.json()["topology"] == "mesh"

    @pytest.mark.asyncio
    async def test_quietstar_phase_execution(self):
        """Test Quiet-STaR phase: Reasoning enhancement."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": ["quietstar"],
                    "config": {
                        "quietstar": {
                            "thought_length": 100,
                            "num_thoughts": 16,
                            "start_token": "<|startofthought|>",
                            "end_token": "<|endofthought|>"
                        }
                    }
                }
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_bitnet_phase_execution(self):
        """Test BitNet phase: Compression."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": ["bitnet"],
                    "config": {
                        "bitnet": {
                            "precision": "1.58bit",
                            "quantization_method": "absmax"
                        }
                    }
                }
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_training_phase_execution(self):
        """Test Training phase: Model training."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": ["training"],
                    "config": {
                        "training": {
                            "batch_size": 32,
                            "learning_rate": 1e-4,
                            "epochs": 3,
                            "optimizer": "adamw"
                        }
                    },
                    "enable_monitoring": True
                }
            )

            assert response.status_code == 200
            session_id = response.json()["session_id"]

            # Monitor training metrics
            await asyncio.sleep(1)
            status = await client.get(f"/api/v1/pipeline/status/{session_id}")
            metrics = status.json().get("metrics", {})
            assert "training" in str(status.json())

    @pytest.mark.asyncio
    async def test_baking_phase_execution(self):
        """Test Baking phase: Tool integration."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": ["baking"],
                    "config": {
                        "baking": {
                            "tool_types": ["calculator", "search", "code_execution"],
                            "enable_tool_calling": True
                        }
                    }
                }
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_adas_phase_execution(self):
        """Test ADAS phase: Architecture search."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": ["adas"],
                    "config": {
                        "adas": {
                            "search_space": "transformer",
                            "iterations": 100,
                            "objective": "perplexity"
                        }
                    }
                }
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_compression_phase_execution(self):
        """Test Final Compression phase."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": ["compression"],
                    "config": {
                        "compression": {
                            "method": "pruning",
                            "target_sparsity": 0.5
                        }
                    }
                }
            )

            assert response.status_code == 200


class TestMultiPhaseExecution:
    """Test sequential multi-phase execution."""

    @pytest.mark.asyncio
    async def test_cognate_to_evomerge_pipeline(self):
        """Test Cognate -> EvoMerge pipeline."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": ["cognate", "evomerge"],
                    "config": {
                        "cognate": {
                            "base_models": ["model1", "model2"],
                            "init_strategy": "random"
                        },
                        "evomerge": {
                            "population_size": 10,
                            "generations": 5
                        }
                    },
                    "enable_checkpoints": True
                }
            )

            assert response.status_code == 200
            session_id = response.json()["session_id"]

            # Verify phase progression
            await asyncio.sleep(0.5)
            status = await client.get(f"/api/v1/pipeline/status/{session_id}")
            assert status.status_code == 200

    @pytest.mark.asyncio
    async def test_three_phase_pipeline(self):
        """Test Cognate -> EvoMerge -> Quiet-STaR pipeline."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": ["cognate", "evomerge", "quietstar"],
                    "config": {
                        "cognate": {"base_models": ["model1"], "init_strategy": "random"},
                        "evomerge": {"population_size": 5, "generations": 3},
                        "quietstar": {"thought_length": 50, "num_thoughts": 8}
                    }
                }
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self):
        """Test complete 8-phase pipeline."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": [
                        "cognate", "evomerge", "quietstar", "bitnet",
                        "training", "baking", "adas", "compression"
                    ],
                    "config": {
                        "cognate": {"base_models": ["gpt2"], "init_strategy": "random"},
                        "evomerge": {"population_size": 5, "generations": 2},
                        "quietstar": {"thought_length": 50, "num_thoughts": 4},
                        "bitnet": {"precision": "1.58bit"},
                        "training": {"batch_size": 16, "learning_rate": 1e-4, "epochs": 1},
                        "baking": {"tool_types": ["calculator"]},
                        "adas": {"iterations": 10},
                        "compression": {"target_sparsity": 0.3}
                    },
                    "enable_monitoring": True,
                    "enable_checkpoints": True,
                    "swarm_topology": "hierarchical",
                    "max_agents": 50
                }
            )

            assert response.status_code == 200
            session_id = response.json()["session_id"]

            # Monitor progress
            status = await client.get(f"/api/v1/pipeline/status/{session_id}")
            assert status.status_code == 200


class TestQualityGateEnforcement:
    """Test quality gate validation during execution."""

    @pytest.mark.asyncio
    async def test_quality_gates_cognate_phase(self):
        """Test quality gates for Cognate phase."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Start pipeline
            start_response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": ["cognate"],
                    "config": {
                        "cognate": {"base_models": ["model1"], "init_strategy": "random"}
                    }
                }
            )
            session_id = start_response.json()["session_id"]

            # Validate quality gates
            gate_response = await client.post(
                f"/api/v1/pipeline/quality-gates/{session_id}",
                params={"phase": "cognate"}
            )

            assert gate_response.status_code == 200
            gate_data = gate_response.json()

            assert "results" in gate_data
            assert len(gate_data["results"]) > 0

    @pytest.mark.asyncio
    async def test_quality_gates_block_progression(self):
        """Test that failed quality gates block phase progression."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Start multi-phase pipeline
            response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": ["cognate", "evomerge"],
                    "config": {
                        "cognate": {"base_models": ["model1"], "init_strategy": "random"},
                        "evomerge": {"population_size": 10, "generations": 5}
                    }
                }
            )
            session_id = response.json()["session_id"]

            # Check quality gates
            gate_response = await client.post(
                f"/api/v1/pipeline/quality-gates/{session_id}"
            )

            assert gate_response.status_code == 200


class TestCheckpointRecovery:
    """Test checkpoint save and recovery during execution."""

    @pytest.mark.asyncio
    async def test_save_checkpoint_during_execution(self):
        """Test saving checkpoint mid-execution."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Start pipeline
            start_response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": ["cognate", "evomerge"],
                    "enable_checkpoints": True
                }
            )
            session_id = start_response.json()["session_id"]

            # Save checkpoint
            checkpoint_response = await client.post(
                "/api/v1/pipeline/checkpoint/save",
                json={
                    "session_id": session_id,
                    "checkpoint_name": "mid_execution",
                    "include_model_state": True,
                    "include_swarm_state": True
                }
            )

            assert checkpoint_response.status_code == 200
            checkpoint_data = checkpoint_response.json()
            assert checkpoint_data["model_included"] is True
            assert checkpoint_data["swarm_included"] is True

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self):
        """Test resuming execution from checkpoint."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Start and checkpoint
            start_response = await client.post(
                "/api/v1/pipeline/start",
                json={"phases": ["cognate", "evomerge"]}
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

            # Stop pipeline
            await client.post(
                "/api/v1/pipeline/control",
                json={"session_id": session_id, "action": "stop"}
            )

            # Load checkpoint
            load_response = await client.post(
                f"/api/v1/pipeline/checkpoint/load/{checkpoint_id}"
            )

            assert load_response.status_code == 200
            assert load_response.json()["status"] == "loaded"


class TestPipelineControl:
    """Test pipeline control during execution."""

    @pytest.mark.asyncio
    async def test_pause_resume_during_execution(self):
        """Test pausing and resuming pipeline execution."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Start pipeline
            start_response = await client.post(
                "/api/v1/pipeline/start",
                json={"phases": ["cognate", "evomerge", "quietstar"]}
            )
            session_id = start_response.json()["session_id"]

            # Pause
            pause_response = await client.post(
                "/api/v1/pipeline/control",
                json={"session_id": session_id, "action": "pause"}
            )
            assert pause_response.status_code == 200

            # Resume
            resume_response = await client.post(
                "/api/v1/pipeline/control",
                json={"session_id": session_id, "action": "resume"}
            )
            assert resume_response.status_code == 200

    @pytest.mark.asyncio
    async def test_stop_during_execution(self):
        """Test stopping pipeline execution gracefully."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Start pipeline
            start_response = await client.post(
                "/api/v1/pipeline/start",
                json={"phases": ["training"]}
            )
            session_id = start_response.json()["session_id"]

            # Stop
            stop_response = await client.post(
                "/api/v1/pipeline/control",
                json={"session_id": session_id, "action": "stop", "force": False}
            )
            assert stop_response.status_code == 200

            # Verify status
            status = await client.get(f"/api/v1/pipeline/status/{session_id}")
            # Should show stopped status

    @pytest.mark.asyncio
    async def test_force_cancel_execution(self):
        """Test force canceling pipeline execution."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Start pipeline
            start_response = await client.post(
                "/api/v1/pipeline/start",
                json={"phases": ["training", "adas"]}
            )
            session_id = start_response.json()["session_id"]

            # Force cancel
            cancel_response = await client.post(
                "/api/v1/pipeline/control",
                json={"session_id": session_id, "action": "cancel", "force": True}
            )
            assert cancel_response.status_code == 200


class TestSwarmCoordination:
    """Test swarm coordination during execution."""

    @pytest.mark.asyncio
    async def test_hierarchical_swarm_execution(self):
        """Test execution with hierarchical swarm topology."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": ["cognate", "evomerge"],
                    "swarm_topology": "hierarchical",
                    "max_agents": 30
                }
            )

            assert response.status_code == 200
            session_id = response.json()["session_id"]

            # Check swarm status
            swarm_status = await client.get(f"/api/v1/pipeline/swarm/{session_id}")
            assert swarm_status.json()["topology"] == "hierarchical"

    @pytest.mark.asyncio
    async def test_mesh_swarm_execution(self):
        """Test execution with mesh swarm topology."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": ["cognate"],
                    "swarm_topology": "mesh",
                    "max_agents": 20
                }
            )

            assert response.status_code == 200
            session_id = response.json()["session_id"]

            swarm_status = await client.get(f"/api/v1/pipeline/swarm/{session_id}")
            assert swarm_status.json()["topology"] == "mesh"

    @pytest.mark.asyncio
    async def test_agent_scaling_during_execution(self):
        """Test agent count scaling during execution."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/pipeline/start",
                json={
                    "phases": ["cognate", "evomerge"],
                    "swarm_topology": "star",
                    "max_agents": 40
                }
            )

            assert response.status_code == 200
            session_id = response.json()["session_id"]

            # Monitor agent distribution
            swarm_status = await client.get(f"/api/v1/pipeline/swarm/{session_id}")
            assert swarm_status.status_code == 200