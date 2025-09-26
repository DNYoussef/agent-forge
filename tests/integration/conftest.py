"""
Pytest configuration and fixtures for integration tests.

Provides shared fixtures and configuration for all integration tests.
"""

import asyncio
import pytest
from typing import Generator, AsyncGenerator
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_pipeline_config() -> dict:
    """Provide sample pipeline configuration."""
    return {
        "phases": ["cognate", "evomerge"],
        "config": {
            "cognate": {
                "base_models": ["gpt2", "llama"],
                "init_strategy": "random"
            },
            "evomerge": {
                "population_size": 10,
                "generations": 5,
                "mutation_rate": 0.1
            }
        },
        "swarm_topology": "hierarchical",
        "max_agents": 25,
        "enable_monitoring": True,
        "enable_checkpoints": True
    }


@pytest.fixture
def sample_phase_configs() -> dict:
    """Provide sample configurations for all phases."""
    return {
        "cognate": {
            "base_models": ["model1", "model2"],
            "init_strategy": "random"
        },
        "evomerge": {
            "population_size": 20,
            "generations": 10,
            "mutation_rate": 0.1,
            "crossover_rate": 0.7
        },
        "quietstar": {
            "thought_length": 100,
            "num_thoughts": 16,
            "start_token": "<|startofthought|>",
            "end_token": "<|endofthought|>"
        },
        "bitnet": {
            "precision": "1.58bit",
            "quantization_method": "absmax"
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "epochs": 3,
            "optimizer": "adamw"
        },
        "baking": {
            "tool_types": ["calculator", "search", "code_execution"],
            "enable_tool_calling": True
        },
        "adas": {
            "search_space": "transformer",
            "iterations": 100,
            "objective": "perplexity"
        },
        "compression": {
            "method": "pruning",
            "target_sparsity": 0.5
        }
    }


@pytest.fixture
async def api_client():
    """Provide async API client for testing."""
    from utils.test_helpers import APITestClient

    client = APITestClient()
    yield client


@pytest.fixture
async def websocket_client():
    """Provide WebSocket client for testing."""
    from utils.test_helpers import WebSocketTestClient

    client = WebSocketTestClient()
    yield client


@pytest.fixture
def data_generator():
    """Provide pipeline data generator."""
    from fixtures.pipeline_data_generator import PipelineDataGenerator

    return PipelineDataGenerator()


@pytest.fixture
def mock_simulator():
    """Provide mock pipeline simulator."""
    from fixtures.pipeline_data_generator import MockPipelineSimulator

    return MockPipelineSimulator()


@pytest.fixture
def assertion_helpers():
    """Provide assertion helper utilities."""
    from utils.test_helpers import AssertionHelpers

    return AssertionHelpers()


@pytest.fixture
def test_scenarios():
    """Provide test scenario utilities."""
    from utils.test_helpers import TestScenarios

    return TestScenarios()


@pytest.fixture
def mock_data_builder():
    """Provide mock data builder."""
    from utils.test_helpers import MockDataBuilder

    return MockDataBuilder()


# Pytest hooks

def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Auto-mark all tests in integration directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark slow tests
        if "slow" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)

        # Mark by test type
        if "/api/" in str(item.fspath):
            item.add_marker(pytest.mark.api)
        elif "/websocket/" in str(item.fspath):
            item.add_marker(pytest.mark.websocket)
        elif "/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "/validation/" in str(item.fspath):
            item.add_marker(pytest.mark.validation)


def pytest_report_header(config):
    """Add custom header to pytest output."""
    return [
        "Agent Forge Integration Test Suite",
        "Testing: FastAPI Backend + Next.js Frontend + 8-Phase Pipeline",
        "=" * 80
    ]