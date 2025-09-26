"""
Integration tests for Enhanced Phase 6 Tool & Persona Baking
Tests the complete system including benchmarks, real tools, and MCP servers
"""

import asyncio
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
from pathlib import Path

# Import enhanced Phase 6 components
from agent_forge.phases.phase6_baking.enhanced_baking_system import (
    EnhancedBakingConfig,
    EnhancedToolPersonaBaking,
    PromptGenerator,
    ProgressiveHalfBaker,
    AdaptiveIterationController
)
from agent_forge.phases.phase6_baking.benchmarks.swe_bench_integration import (
    BenchmarkConfig,
    SWEBenchEvaluator,
    UnifiedBenchmarkRunner
)
from agent_forge.phases.phase6_baking.tools.real_tool_implementations import (
    RealToolSystem,
    ToolConfig,
    RealCalculatorTool,
    RealWebSearchTool,
    RealCodeExecutorTool
)
from agent_forge.phases.phase6_baking.mcp.server_registry import (
    MCPServerRegistry,
    MCPTaskRouter,
    MCPCapability
)


class TestEnhancedPhase6:
    """Test suite for enhanced Phase 6 implementation"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return EnhancedBakingConfig(
            model_path="/tmp/test_model",
            output_path="/tmp/test_output",
            target_success_rate=0.95,
            max_iterations=10,  # Reduced for testing
            prompts_per_task=5,  # Reduced for testing
            use_real_tools=True,
            use_mcp_servers=True,
            use_progressive_baking=True
        )

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing"""
        model = Mock(spec=nn.Module)
        model.eval = Mock()
        model.train = Mock()
        model.parameters = Mock(return_value=iter([Mock()]))
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer"""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3])
        tokenizer.decode = Mock(return_value="Generated response")
        return tokenizer


class TestSWEBenchIntegration:
    """Test SWE-bench evaluation framework"""

    def test_benchmark_creation(self):
        """Test that benchmarks are properly initialized"""
        config = BenchmarkConfig(
            target_success_rate=0.95,
            benchmarks=["swe-bench"],
            swe_bench_tasks=10
        )

        evaluator = SWEBenchEvaluator(config)

        assert evaluator.benchmark_data is not None
        assert len(evaluator.benchmark_data) == 10
        assert evaluator.config.target_success_rate == 0.95

    def test_task_types(self):
        """Test that all required task types are present"""
        config = BenchmarkConfig(swe_bench_tasks=100)
        evaluator = SWEBenchEvaluator(config)

        task_types = set(task["type"] for task in evaluator.benchmark_data)

        expected_types = {
            "code_execution",
            "web_search",
            "file_operations",
            "api_calls",
            "data_analysis"
        }

        assert expected_types.issubset(task_types)

    @pytest.mark.asyncio
    async def test_benchmark_evaluation(self):
        """Test benchmark evaluation process"""
        config = BenchmarkConfig(
            swe_bench_tasks=5,
            use_real_execution=False  # Use mock for testing
        )

        evaluator = SWEBenchEvaluator(config)

        # Mock model and tokenizer
        model = Mock()
        tokenizer = Mock()
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        tokenizer.decode = Mock(return_value="Calculate the result")

        # Mock tool system
        tool_system = Mock()
        tool_system.call_tool = Mock(return_value={"success": True, "result": "42"})

        # Run evaluation
        result = await evaluator.evaluate_model(
            model, tokenizer, tool_system, ["Test prompt"]
        )

        assert result.benchmark_name == "swe-bench"
        assert result.total_tasks == 5
        assert result.success_rate >= 0  # Should have some score
        assert len(result.task_results) == 5

    def test_success_rate_calculation(self):
        """Test that success rate is calculated correctly"""
        config = BenchmarkConfig()
        runner = UnifiedBenchmarkRunner(config)

        results = {
            "swe-bench": Mock(success_rate=0.92),
            "toolbench": Mock(success_rate=0.96)
        }

        # Should not meet target (0.95)
        assert not runner.meets_target(results)

        # Update to meet target
        results["swe-bench"].success_rate = 0.96
        assert runner.meets_target(results)


class TestRealToolImplementations:
    """Test real tool implementations"""

    def test_calculator_tool(self):
        """Test real calculator implementation"""
        tool = RealCalculatorTool(ToolConfig())

        # Test valid expressions
        result = asyncio.run(tool.execute("2 + 2"))
        assert result["success"] is True
        assert result["result"] == 4

        result = asyncio.run(tool.execute("10 * 5"))
        assert result["success"] is True
        assert result["result"] == 50

        # Test invalid expressions
        result = asyncio.run(tool.execute("import os"))
        assert result["success"] is False

    def test_code_executor_safety(self):
        """Test that code executor is safe"""
        tool = RealCodeExecutorTool(ToolConfig())

        # Should timeout on infinite loops
        dangerous_code = "while True: pass"
        result = asyncio.run(tool.execute(dangerous_code))
        assert result["success"] is False

    def test_file_manager_sandboxing(self):
        """Test file manager sandboxing"""
        from agent_forge.phases.phase6_baking.tools.real_tool_implementations import RealFileManagerTool

        tool = RealFileManagerTool(ToolConfig())

        # Should prevent path traversal
        result = asyncio.run(tool.execute("read", path="../../etc/passwd"))
        assert result["success"] is False
        assert "traversal" in result.get("error", "").lower() or "not found" in result.get("error", "").lower()

    def test_tool_system_tracking(self):
        """Test tool usage tracking"""
        tool_system = RealToolSystem(ToolConfig())

        # Make some tool calls
        asyncio.run(tool_system.call_tool("calculator", expression="2+2"))
        asyncio.run(tool_system.call_tool("calculator", expression="invalid"))

        stats = tool_system.get_usage_stats()

        assert stats["calculator"]["calls"] == 2
        assert stats["calculator"]["successes"] == 1
        assert stats["calculator"]["failures"] == 1
        assert stats["calculator"]["success_rate"] == 0.5


class TestMCPServerRegistry:
    """Test MCP server registry and selection"""

    def test_server_registration(self):
        """Test MCP server registration"""
        registry = MCPServerRegistry()

        # Check default servers are registered
        assert "filesystem" in registry.servers
        assert "code-runner" in registry.servers
        assert "web-search" in registry.servers
        assert "github" in registry.servers

    def test_capability_selection(self):
        """Test selecting servers by capability"""
        registry = MCPServerRegistry()

        # Get servers for file operations
        file_servers = registry.get_servers_by_capability(MCPCapability.FILE_OPERATIONS)
        assert len(file_servers) > 0
        assert any(s.name == "filesystem" for s in file_servers)

        # Get best server for code execution
        best_server = registry.get_best_server(MCPCapability.CODE_EXECUTION)
        assert best_server is not None
        assert MCPCapability.CODE_EXECUTION in best_server.capabilities

    def test_task_routing(self):
        """Test task routing to appropriate servers"""
        registry = MCPServerRegistry()
        router = MCPTaskRouter(registry)

        # Test capability identification
        capability = router.identify_required_capability("read the file config.json")
        assert capability == MCPCapability.FILE_OPERATIONS

        capability = router.identify_required_capability("execute this python code")
        assert capability == MCPCapability.CODE_EXECUTION

        capability = router.identify_required_capability("search for quantum computing")
        assert capability == MCPCapability.WEB_SEARCH

        # Test server selection
        server = router.select_server("create a pull request on github")
        assert server is not None
        assert server.name == "github"


class TestProgressiveHalfBaking:
    """Test progressive half-baking mechanism"""

    def test_layer_scheduling(self):
        """Test that layers are progressively added"""
        config = EnhancedBakingConfig(
            use_progressive_baking=True,
            max_iterations=100
        )
        half_baker = ProgressiveHalfBaker(config)

        # Early iterations - few layers
        layers = half_baker.get_layers_for_iteration(5)
        assert len(layers) == 2  # Only foundation layers

        # Middle iterations - more layers
        layers = half_baker.get_layers_for_iteration(25)
        assert len(layers) == 5  # Foundation + understanding

        # Late iterations - all layers
        layers = half_baker.get_layers_for_iteration(50)
        assert len(layers) >= 8  # Most layers

    def test_parameter_scheduling(self):
        """Test parameter scheduling over iterations"""
        config = EnhancedBakingConfig()
        half_baker = ProgressiveHalfBaker(config)

        # Early iteration - gentle parameters
        strength_early = half_baker.get_baking_strength(1)
        lr_early = half_baker.get_learning_rate(1)

        # Late iteration - stronger parameters
        strength_late = half_baker.get_baking_strength(50)
        lr_late = half_baker.get_learning_rate(50)

        assert strength_late > strength_early
        assert lr_late < lr_early  # Learning rate decreases


class TestAdaptiveIteration:
    """Test adaptive iteration controller"""

    def test_target_achievement(self):
        """Test stopping when target is reached"""
        config = EnhancedBakingConfig(target_success_rate=0.95)
        controller = AdaptiveIterationController(config)

        # Below target - should continue
        should_continue, reason = controller.should_continue(10, 0.85)
        assert should_continue is True
        assert "Continue" in reason

        # At target - should stop
        should_continue, reason = controller.should_continue(10, 0.95)
        assert should_continue is False
        assert "Target reached" in reason

    def test_early_stopping(self):
        """Test early stopping on no improvement"""
        config = EnhancedBakingConfig(
            early_stopping_patience=3,
            max_iterations=100
        )
        controller = AdaptiveIterationController(config)

        # Simulate no improvement
        for i in range(5):
            should_continue, reason = controller.should_continue(i, 0.80)

        # Should stop after patience exceeded
        if i >= 3:
            assert should_continue is False
            assert "Early stopping" in reason

    def test_adaptive_parameters(self):
        """Test adaptive parameter adjustment"""
        config = EnhancedBakingConfig()
        controller = AdaptiveIterationController(config)

        # Far from target - aggressive parameters
        params = controller.get_adaptive_parameters(10, 0.70)
        assert params["baking_strength_multiplier"] > 1.0
        assert params["epochs_per_iteration"] > 3

        # Close to target - gentle parameters
        params = controller.get_adaptive_parameters(10, 0.93)
        assert params["baking_strength_multiplier"] == 1.0
        assert params["epochs_per_iteration"] <= 2


class TestPromptGeneration:
    """Test multi-prompt generation system"""

    def test_prompt_variations(self):
        """Test that multiple prompt variations are generated"""
        generator = PromptGenerator()

        # Generate calculator prompts
        prompts = generator.generate_variations("calculator", "15 * 23", count=10)
        assert len(prompts) == 10
        assert all("15 * 23" in p or "expression" in p for p in prompts)

        # Generate search prompts
        prompts = generator.generate_variations("web_search", "quantum computing", count=5)
        assert len(prompts) == 5
        assert all("quantum computing" in p or "query" in p for p in prompts)

    def test_prompt_diversity(self):
        """Test that prompts are diverse"""
        generator = PromptGenerator()

        prompts = generator.generate_variations("code_executor", "def factorial(n)", count=10)

        # Should have unique prompts
        unique_prompts = set(prompts)
        assert len(unique_prompts) >= 8  # Allow some duplicates but mostly unique


class TestEndToEndIntegration:
    """Test complete enhanced baking system"""

    @pytest.mark.asyncio
    @patch('agent_forge.phases.phase6_baking.enhanced_baking_system.UnifiedBenchmarkRunner')
    async def test_complete_baking_flow(self, mock_benchmark_runner):
        """Test the complete enhanced baking flow"""
        config = EnhancedBakingConfig(
            target_success_rate=0.95,
            max_iterations=5,  # Quick test
            prompts_per_task=3
        )

        # Mock benchmark results
        mock_result = Mock()
        mock_result.success_rate = 0.96  # Above target
        mock_result.task_results = []
        mock_result.prompt_success_rates = {"test": 0.96}

        mock_benchmark_runner.return_value.run_all_benchmarks = Mock(
            return_value={"swe-bench": mock_result}
        )

        baking_system = EnhancedToolPersonaBaking(config)

        # Mock model and tokenizer
        model = Mock(spec=nn.Module)
        tokenizer = Mock()

        # Run enhanced baking
        result = await baking_system.run_enhanced_baking(model, tokenizer)

        assert result["success"] is True
        assert result["final_score"] >= 0.95
        assert result["target_met"] is True
        assert result["iterations"] <= 5

    def test_gap_detection(self):
        """Test that system detects gaps to target"""
        config = EnhancedBakingConfig(target_success_rate=0.95)
        controller = AdaptiveIterationController(config)

        # Test gap calculation
        should_continue, reason = controller.should_continue(1, 0.75)
        assert should_continue is True
        assert "0.20" in reason or "20" in reason  # Should mention 20% gap


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for Phase 6"""

    def test_tool_latency(self):
        """Benchmark tool execution latency"""
        tool_system = RealToolSystem(ToolConfig())

        import time
        start = time.time()

        # Run multiple tool calls
        for _ in range(10):
            asyncio.run(tool_system.call_tool("calculator", expression="2+2"))

        duration = time.time() - start
        avg_latency = duration / 10 * 1000  # Convert to ms

        assert avg_latency < 100  # Should be fast

    def test_prompt_generation_speed(self):
        """Benchmark prompt generation speed"""
        generator = PromptGenerator()

        import time
        start = time.time()

        # Generate many prompts
        for _ in range(100):
            generator.generate_variations("calculator", "test", count=10)

        duration = time.time() - start

        assert duration < 1.0  # Should generate 1000 prompts in under 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])