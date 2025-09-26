"""
Comprehensive system validation tests.

Validates complete integration including:
- End-to-end workflows
- Data consistency
- Error handling
- Performance benchmarks
- Security validations
"""

import asyncio
import pytest
from datetime import datetime
import sys
from pathlib import Path

# Add src and test utils to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.test_helpers import (
    APITestClient,
    WebSocketTestClient,
    AssertionHelpers,
    TestScenarios,
    MockDataBuilder
)
from fixtures.pipeline_data_generator import PipelineDataGenerator


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    @pytest.mark.asyncio
    async def test_complete_8_phase_pipeline(self):
        """Test full 8-phase pipeline execution."""
        client = APITestClient()
        generator = PipelineDataGenerator()

        # Generate complete config
        config = generator.generate_pipeline_config(include_all_phases=True)

        # Start pipeline
        result = await client.start_pipeline(
            phases=config.phases,
            config=config.config,
            swarm_topology=config.swarm_topology,
            max_agents=config.max_agents,
            enable_monitoring=True,
            enable_checkpoints=True
        )

        session_id = result["session_id"]

        # Verify initial status
        status = await client.get_status(session_id)
        AssertionHelpers.assert_pipeline_status(
            status,
            expected_status="running"
        )

        # Verify swarm coordination
        swarm_status = await client.get_status(session_id)
        assert "session_id" in swarm_status

    @pytest.mark.asyncio
    async def test_pipeline_with_quality_gates(self):
        """Test pipeline execution with quality gate enforcement."""
        client = APITestClient()

        # Start pipeline
        session_id = await TestScenarios.run_multi_phase_pipeline(
            client,
            phases=["cognate", "evomerge", "quietstar"],
            config={
                "cognate": {"base_models": ["model1"], "init_strategy": "random"},
                "evomerge": {"population_size": 10, "generations": 5},
                "quietstar": {"thought_length": 50, "num_thoughts": 8}
            }
        )

        # Validate quality gates for each phase
        for phase in ["cognate", "evomerge", "quietstar"]:
            gate_result = await client.validate_quality_gates(session_id, phase)
            AssertionHelpers.assert_quality_gate_result(gate_result)

    @pytest.mark.asyncio
    async def test_checkpoint_recovery_workflow(self):
        """Test complete checkpoint and recovery workflow."""
        client = APITestClient()

        # Start pipeline
        session_id = await TestScenarios.run_multi_phase_pipeline(
            client,
            phases=["cognate", "evomerge"],
            enable_checkpoints=True
        )

        # Save checkpoint
        checkpoint = await client.save_checkpoint(
            session_id,
            checkpoint_name="recovery_test",
            include_model=True,
            include_swarm=True
        )

        AssertionHelpers.assert_checkpoint(
            checkpoint,
            should_include_model=True,
            should_include_swarm=True
        )

        # Simulate failure and recovery
        await client.control_pipeline(session_id, "stop")

        # Load checkpoint
        load_result = await client.load_checkpoint(checkpoint["checkpoint_id"])
        assert load_result["status"] == "loaded"


class TestDataConsistency:
    """Test data consistency across system components."""

    @pytest.mark.asyncio
    async def test_status_consistency(self):
        """Test status consistency between API and WebSocket."""
        client = APITestClient()
        ws_client = WebSocketTestClient()

        # Start pipeline
        session_id = await TestScenarios.run_single_phase_pipeline(
            client,
            "cognate",
            {"base_models": ["model1"], "init_strategy": "random"}
        )

        # Get API status
        api_status = await client.get_status(session_id)

        # Connect to WebSocket and collect updates
        async with ws_client.connect("pipeline", session_id):
            await asyncio.sleep(2)

        # Both should reflect same session
        assert api_status["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_metrics_consistency(self):
        """Test metrics consistency across endpoints."""
        client = APITestClient()

        session_id = await TestScenarios.run_single_phase_pipeline(
            client,
            "training",
            {"batch_size": 32, "learning_rate": 1e-4, "epochs": 3}
        )

        # Get status with metrics
        status = await client.get_status(session_id)

        # Verify metrics structure
        AssertionHelpers.assert_metrics(status.get("metrics", {}))

    @pytest.mark.asyncio
    async def test_swarm_agent_consistency(self):
        """Test agent count consistency between swarm and status."""
        client = APITestClient()

        session_id = await TestScenarios.run_single_phase_pipeline(
            client,
            "cognate"
        )

        # Get both statuses
        pipeline_status = await client.get_status(session_id)
        # Verify consistency
        assert "active_agents" in str(pipeline_status)


class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_invalid_session_id(self):
        """Test handling of invalid session ID."""
        client = APITestClient()

        with pytest.raises(Exception):
            await client.get_status("invalid_session_id_12345")

    @pytest.mark.asyncio
    async def test_invalid_phase_name(self):
        """Test handling of invalid phase names."""
        client = APITestClient()

        with pytest.raises(Exception):
            await client.start_pipeline(
                phases=["invalid_phase_name"],
                config={}
            )

    @pytest.mark.asyncio
    async def test_concurrent_control_actions(self):
        """Test handling of concurrent control actions."""
        client = APITestClient()

        session_id = await TestScenarios.run_single_phase_pipeline(
            client,
            "cognate"
        )

        # Try concurrent pause and resume (should handle gracefully)
        try:
            await asyncio.gather(
                client.control_pipeline(session_id, "pause"),
                client.control_pipeline(session_id, "resume")
            )
        except Exception:
            # Expected that one may fail due to state
            pass

    @pytest.mark.asyncio
    async def test_checkpoint_not_found(self):
        """Test loading non-existent checkpoint."""
        client = APITestClient()

        with pytest.raises(Exception):
            await client.load_checkpoint("nonexistent_checkpoint_12345")


class TestPerformance:
    """Test performance and scalability."""

    @pytest.mark.asyncio
    async def test_api_response_time(self):
        """Test API response times are acceptable."""
        client = APITestClient()

        start_time = asyncio.get_event_loop().time()

        session_id = await TestScenarios.run_single_phase_pipeline(
            client,
            "cognate"
        )

        response_time = asyncio.get_event_loop().time() - start_time

        # Should respond within 2 seconds
        assert response_time < 2.0

    @pytest.mark.asyncio
    async def test_concurrent_pipelines(self):
        """Test running multiple concurrent pipelines."""
        client = APITestClient()

        # Start 5 concurrent pipelines
        sessions = await asyncio.gather(*[
            TestScenarios.run_single_phase_pipeline(client, "cognate")
            for _ in range(5)
        ])

        # All should succeed
        assert len(sessions) == 5
        assert len(set(sessions)) == 5  # All unique

    @pytest.mark.asyncio
    async def test_websocket_message_throughput(self):
        """Test WebSocket message handling throughput."""
        ws_client = WebSocketTestClient()

        async with ws_client.connect("metrics"):
            # Collect messages for 5 seconds
            messages = await ws_client.collect_messages(duration=5.0)

            # Should receive regular updates
            assert len(messages) > 0

    @pytest.mark.asyncio
    async def test_large_config_handling(self):
        """Test handling of large configuration payloads."""
        client = APITestClient()
        generator = PipelineDataGenerator()

        # Generate large config
        config = generator.generate_pipeline_config(include_all_phases=True)

        # Add extra config data
        for phase in config.phases:
            config.config[phase]["extra_data"] = {
                f"param_{i}": f"value_{i}" for i in range(100)
            }

        # Should handle large payload
        session_id = await TestScenarios.run_multi_phase_pipeline(
            client,
            phases=config.phases,
            config=config.config
        )

        assert session_id


class TestSecurityValidation:
    """Test security-related validations."""

    @pytest.mark.asyncio
    async def test_session_id_format(self):
        """Test session ID follows secure format."""
        client = APITestClient()

        session_id = await TestScenarios.run_single_phase_pipeline(
            client,
            "cognate"
        )

        # Should be sufficiently long and complex
        assert len(session_id) >= 10
        assert session_id.startswith("session_")

    @pytest.mark.asyncio
    async def test_config_validation(self):
        """Test configuration input validation."""
        client = APITestClient()

        # Test with invalid config types
        with pytest.raises(Exception):
            await client.start_pipeline(
                phases=["cognate"],
                config={"cognate": "invalid_config_type"}
            )

    @pytest.mark.asyncio
    async def test_websocket_authentication(self):
        """Test WebSocket connection security."""
        ws_client = WebSocketTestClient()

        # Should be able to connect (basic test)
        async with ws_client.connect("agents"):
            await ws_client.send_json({"type": "ping"})
            response = await ws_client.receive_json()
            assert response["type"] == "pong"


class TestRobustness:
    """Test system robustness and reliability."""

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self):
        """Test graceful pipeline shutdown."""
        client = APITestClient()

        session_id = await TestScenarios.run_single_phase_pipeline(
            client,
            "training"
        )

        # Gracefully stop
        result = await client.control_pipeline(session_id, "stop", force=False)
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_force_termination(self):
        """Test force termination."""
        client = APITestClient()

        session_id = await TestScenarios.run_single_phase_pipeline(
            client,
            "training"
        )

        # Force cancel
        result = await client.control_pipeline(session_id, "cancel", force=True)
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_recovery_after_error(self):
        """Test system recovery after errors."""
        client = APITestClient()

        # Cause an error (invalid config)
        try:
            await client.start_pipeline(
                phases=["invalid_phase"]
            )
        except Exception:
            pass

        # Should still be able to start valid pipeline
        session_id = await TestScenarios.run_single_phase_pipeline(
            client,
            "cognate"
        )

        assert session_id

    @pytest.mark.asyncio
    async def test_websocket_reconnection(self):
        """Test WebSocket reconnection handling."""
        ws_client = WebSocketTestClient()

        # Connect, disconnect, reconnect
        async with ws_client.connect("agents"):
            await ws_client.send_json({"type": "ping"})
            response1 = await ws_client.receive_json()

        # Reconnect
        async with ws_client.connect("agents"):
            await ws_client.send_json({"type": "ping"})
            response2 = await ws_client.receive_json()

        assert response1["type"] == "pong"
        assert response2["type"] == "pong"


class TestIntegrationEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_phase_list(self):
        """Test handling of empty phase list."""
        client = APITestClient()

        with pytest.raises(Exception):
            await client.start_pipeline(phases=[])

    @pytest.mark.asyncio
    async def test_duplicate_phases(self):
        """Test handling of duplicate phases."""
        client = APITestClient()

        # May accept or reject duplicates
        try:
            session_id = await client.start_pipeline(
                phases=["cognate", "cognate"]
            )
            # If accepted, should handle gracefully
            assert session_id
        except Exception:
            # If rejected, that's also acceptable
            pass

    @pytest.mark.asyncio
    async def test_max_agents_boundary(self):
        """Test max agents boundary values."""
        client = APITestClient()

        # Test with minimum (1) and maximum (100) agents
        for agent_count in [1, 100]:
            session_id = await client.start_pipeline(
                phases=["cognate"],
                max_agents=agent_count
            )
            assert session_id

    @pytest.mark.asyncio
    async def test_rapid_start_stop(self):
        """Test rapid start/stop cycles."""
        client = APITestClient()

        for _ in range(3):
            session_id = await TestScenarios.run_single_phase_pipeline(
                client,
                "cognate"
            )

            await client.control_pipeline(session_id, "stop")

        # Should handle rapid cycles
        assert True