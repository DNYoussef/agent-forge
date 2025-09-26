"""
Enhanced Tool & Persona Baking System for Phase 6
Integrates SWE-bench, real tools, MCP servers, and adaptive iteration
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Import new components
from .benchmarks.swe_bench_integration import (
    BenchmarkConfig, UnifiedBenchmarkRunner, BenchmarkResult
)
from .tools.real_tool_implementations import (
    RealToolSystem, ToolConfig
)
from .mcp.server_registry import (
    MCPServerRegistry, MCPTaskRouter, MCPServerOptimizer
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedBakingConfig:
    """Enhanced configuration for Phase 6 baking with all improvements"""

    # Model configuration
    model_path: str = ""
    output_path: str = ""

    # Target metrics (your requirement)
    target_success_rate: float = 0.95  # 95% target
    max_iterations: int = 100  # Continue until target met or max reached
    early_stopping_patience: int = 5  # Stop if no improvement for N iterations

    # Multi-prompt testing (your requirement)
    prompts_per_task: int = 10  # Try 10 different prompts per task
    prompt_selection_top_k: int = 3  # Select top 3 performing prompts

    # Progressive half-baking (your requirement)
    use_progressive_baking: bool = True
    baking_layers_schedule: List[List[int]] = field(default_factory=lambda: [
        [0, 1],       # Iteration 1-10: Bake early layers
        [2, 3, 4],    # Iteration 11-20: Add middle layers
        [5, 6, 7],    # Iteration 21-30: Add more layers
        [8, 9, 10],   # Iteration 31+: Add final layers
    ])

    # Benchmarks to use
    benchmarks: List[str] = field(default_factory=lambda: [
        "swe-bench", "toolbench", "humaneval"
    ])

    # Tool configuration
    use_real_tools: bool = True
    use_mcp_servers: bool = True

    # Baking parameters
    baking_strength_schedule: List[float] = field(default_factory=lambda: [
        0.05,  # Start gentle
        0.10,
        0.15,
        0.20,  # Increase gradually
        0.25
    ])

    learning_rate_schedule: List[float] = field(default_factory=lambda: [
        1e-4,  # Start higher
        5e-5,
        1e-5,  # Decrease over time
        5e-6,
        1e-6
    ])

    # Grokfast configuration
    enable_grokfast: bool = True
    grokfast_ema_alpha: float = 0.98
    grokfast_lambda_schedule: List[float] = field(default_factory=lambda: [
        0.01, 0.05, 0.10, 0.15, 0.20
    ])


class PromptGenerator:
    """Generates multiple prompt variations for testing"""

    def __init__(self):
        self.templates = {
            "calculator": [
                "Calculate: {expression}",
                "Compute the result of {expression}",
                "What is {expression}?",
                "Solve: {expression}",
                "Evaluate {expression} mathematically",
                "Use the calculator tool to find {expression}",
                "I need to calculate {expression}",
                "Help me compute {expression}",
                "Mathematical calculation needed: {expression}",
                "Please calculate {expression} for me"
            ],
            "web_search": [
                "Search for: {query}",
                "Find information about {query}",
                "Look up {query} on the web",
                "I need to know about {query}",
                "Research {query} online",
                "Use web search to find {query}",
                "Google {query} for me",
                "What can you find about {query}?",
                "Search the internet for {query}",
                "Get me information on {query}"
            ],
            "code_executor": [
                "Run this code: {code}",
                "Execute: {code}",
                "Please run: {code}",
                "I need to execute this: {code}",
                "Use the code executor: {code}",
                "Evaluate this code: {code}",
                "Test this code: {code}",
                "What's the output of: {code}",
                "Run and show results: {code}",
                "Execute this Python code: {code}"
            ]
        }

    def generate_variations(
        self,
        task_type: str,
        base_content: str,
        count: int = 10
    ) -> List[str]:
        """Generate multiple prompt variations for a task"""

        if task_type not in self.templates:
            # Generic variations
            variations = [
                f"Please {base_content}",
                f"I need you to {base_content}",
                f"Can you {base_content}?",
                f"Help me {base_content}",
                f"{base_content}",
                f"Task: {base_content}",
                f"Objective: {base_content}",
                f"Goal: {base_content}",
                f"Request: {base_content}",
                f"Action needed: {base_content}"
            ]
        else:
            templates = self.templates[task_type]
            variations = [
                template.format(**{task_type.split('_')[0]: base_content})
                for template in templates
            ]

        # Add some randomization
        random.shuffle(variations)

        return variations[:count]


class ProgressiveHalfBaker:
    """Implements progressive half-baking with layer scheduling"""

    def __init__(self, config: EnhancedBakingConfig):
        self.config = config
        self.current_stage = 0
        self.baked_layers = set()

    def get_layers_for_iteration(self, iteration: int) -> List[int]:
        """Get which layers to bake based on iteration"""

        if not self.config.use_progressive_baking:
            # Bake all layers if not using progressive
            return list(range(12))

        # Determine stage based on iteration
        stage_size = self.config.max_iterations // len(self.config.baking_layers_schedule)

        new_stage = min(
            iteration // stage_size,
            len(self.config.baking_layers_schedule) - 1
        )

        # Get layers for current stage
        layers = []
        for i in range(new_stage + 1):
            layers.extend(self.config.baking_layers_schedule[i])

        # Track what's new
        new_layers = set(layers) - self.baked_layers
        if new_layers:
            logger.info(f"Adding layers {new_layers} to baking at iteration {iteration}")
            self.baked_layers.update(new_layers)

        return layers

    def get_baking_strength(self, iteration: int) -> float:
        """Get baking strength based on iteration"""

        schedule_index = min(
            iteration // 10,
            len(self.config.baking_strength_schedule) - 1
        )

        return self.config.baking_strength_schedule[schedule_index]

    def get_learning_rate(self, iteration: int) -> float:
        """Get learning rate based on iteration"""

        schedule_index = min(
            iteration // 10,
            len(self.config.learning_rate_schedule) - 1
        )

        return self.config.learning_rate_schedule[schedule_index]

    def get_grokfast_lambda(self, iteration: int) -> float:
        """Get Grokfast lambda based on iteration"""

        schedule_index = min(
            iteration // 10,
            len(self.config.grokfast_lambda_schedule) - 1
        )

        return self.config.grokfast_lambda_schedule[schedule_index]


class AdaptiveIterationController:
    """Controls iteration based on performance towards 95% target"""

    def __init__(self, config: EnhancedBakingConfig):
        self.config = config
        self.iteration_history = []
        self.best_score = 0.0
        self.iterations_without_improvement = 0

    def should_continue(
        self,
        iteration: int,
        current_score: float
    ) -> Tuple[bool, str]:
        """Determine if baking should continue"""

        # Record history
        self.iteration_history.append({
            "iteration": iteration,
            "score": current_score,
            "timestamp": time.time()
        })

        # Check if target reached
        if current_score >= self.config.target_success_rate:
            return False, f"Target reached: {current_score:.2%} >= {self.config.target_success_rate:.2%}"

        # Check max iterations
        if iteration >= self.config.max_iterations:
            return False, f"Max iterations reached: {iteration}"

        # Check improvement
        if current_score > self.best_score:
            self.best_score = current_score
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1

        # Check early stopping
        if self.iterations_without_improvement >= self.config.early_stopping_patience:
            return False, f"Early stopping: No improvement for {self.iterations_without_improvement} iterations"

        # Continue
        gap = self.config.target_success_rate - current_score
        return True, f"Continue: Current {current_score:.2%}, need {gap:.2%} more"

    def get_adaptive_parameters(
        self,
        iteration: int,
        current_score: float
    ) -> Dict[str, Any]:
        """Get adaptive parameters based on performance"""

        gap = self.config.target_success_rate - current_score

        # Adjust aggressiveness based on gap
        if gap > 0.2:  # Far from target
            return {
                "baking_strength_multiplier": 1.5,
                "learning_rate_multiplier": 2.0,
                "epochs_per_iteration": 5
            }
        elif gap > 0.1:  # Getting closer
            return {
                "baking_strength_multiplier": 1.2,
                "learning_rate_multiplier": 1.5,
                "epochs_per_iteration": 3
            }
        else:  # Close to target
            return {
                "baking_strength_multiplier": 1.0,
                "learning_rate_multiplier": 1.0,
                "epochs_per_iteration": 2
            }


class EnhancedToolPersonaBaking:
    """Main enhanced baking system with all improvements"""

    def __init__(self, config: EnhancedBakingConfig):
        self.config = config

        # Initialize components
        self.benchmark_config = BenchmarkConfig(
            target_success_rate=config.target_success_rate,
            benchmarks=config.benchmarks
        )
        self.benchmark_runner = UnifiedBenchmarkRunner(self.benchmark_config)

        # Initialize real tools
        if config.use_real_tools:
            self.tool_system = RealToolSystem(ToolConfig())
        else:
            self.tool_system = None  # Use mock

        # Initialize MCP servers
        if config.use_mcp_servers:
            self.mcp_registry = MCPServerRegistry()
            self.mcp_router = MCPTaskRouter(self.mcp_registry)
            self.mcp_optimizer = MCPServerOptimizer(self.mcp_registry)
        else:
            self.mcp_registry = None
            self.mcp_router = None
            self.mcp_optimizer = None

        # Initialize other components
        self.prompt_generator = PromptGenerator()
        self.half_baker = ProgressiveHalfBaker(config)
        self.iteration_controller = AdaptiveIterationController(config)

        # Tracking
        self.best_prompts = {}
        self.benchmark_history = []

    async def run_enhanced_baking(
        self,
        model: nn.Module,
        tokenizer
    ) -> Dict[str, Any]:
        """Run the enhanced baking process"""

        logger.info("Starting Enhanced Tool & Persona Baking")
        start_time = time.time()

        iteration = 0
        current_score = 0.0

        while True:
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration}")
            logger.info(f"{'='*60}")

            # Get adaptive parameters
            adaptive_params = self.iteration_controller.get_adaptive_parameters(
                iteration, current_score
            )

            # Progressive baking - determine layers
            layers_to_bake = self.half_baker.get_layers_for_iteration(iteration)
            baking_strength = self.half_baker.get_baking_strength(iteration)
            learning_rate = self.half_baker.get_learning_rate(iteration)

            # Apply adaptive multipliers
            baking_strength *= adaptive_params["baking_strength_multiplier"]
            learning_rate *= adaptive_params["learning_rate_multiplier"]

            logger.info(f"Baking layers: {layers_to_bake}")
            logger.info(f"Baking strength: {baking_strength:.4f}")
            logger.info(f"Learning rate: {learning_rate:.6f}")

            # Generate prompt variations for this iteration
            prompt_variations = await self._generate_iteration_prompts()

            # Bake with current parameters
            model = await self._bake_iteration(
                model,
                tokenizer,
                layers_to_bake,
                baking_strength,
                learning_rate,
                adaptive_params["epochs_per_iteration"],
                prompt_variations
            )

            # Evaluate on benchmarks
            benchmark_results = await self.benchmark_runner.run_all_benchmarks(
                model,
                tokenizer,
                self.tool_system,
                prompt_variations
            )

            # Track results
            self.benchmark_history.append({
                "iteration": iteration,
                "results": benchmark_results,
                "timestamp": time.time()
            })

            # Calculate overall score
            scores = [r.success_rate for r in benchmark_results.values()]
            current_score = np.mean(scores) if scores else 0.0

            logger.info(f"Current success rate: {current_score:.2%}")

            # Select best prompts from results
            await self._update_best_prompts(benchmark_results)

            # Check if we should continue
            should_continue, reason = self.iteration_controller.should_continue(
                iteration, current_score
            )

            logger.info(f"Decision: {reason}")

            if not should_continue:
                break

            # Update MCP server metrics if enabled
            if self.config.use_mcp_servers:
                await self._update_mcp_metrics(benchmark_results)

        # Compile final results
        duration = time.time() - start_time

        return {
            "success": current_score >= self.config.target_success_rate,
            "final_score": current_score,
            "iterations": iteration,
            "duration_seconds": duration,
            "best_prompts": self.best_prompts,
            "benchmark_history": self.benchmark_history,
            "target_met": current_score >= self.config.target_success_rate
        }

    async def _generate_iteration_prompts(self) -> List[str]:
        """Generate prompt variations for current iteration"""

        prompts = []

        # Generate variations for different task types
        for task_type in ["calculator", "web_search", "code_executor"]:
            base_content = f"test_{task_type}"
            variations = self.prompt_generator.generate_variations(
                task_type,
                base_content,
                self.config.prompts_per_task
            )
            prompts.extend(variations)

        return prompts

    async def _bake_iteration(
        self,
        model: nn.Module,
        tokenizer,
        layers: List[int],
        strength: float,
        lr: float,
        epochs: int,
        prompts: List[str]
    ) -> nn.Module:
        """Perform one baking iteration"""

        # TODO: Implement actual baking logic
        # This would involve:
        # 1. Creating training data from prompts
        # 2. Fine-tuning specific layers
        # 3. Using Grokfast optimization

        await asyncio.sleep(0.1)  # Simulate baking

        return model

    async def _update_best_prompts(
        self,
        results: Dict[str, BenchmarkResult]
    ):
        """Update best performing prompts"""

        for benchmark_name, result in results.items():
            # Get top performing prompts
            prompt_scores = result.prompt_success_rates

            if prompt_scores:
                # Sort by score
                sorted_prompts = sorted(
                    prompt_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                # Keep top k
                top_prompts = sorted_prompts[:self.config.prompt_selection_top_k]

                self.best_prompts[benchmark_name] = [
                    prompt for prompt, score in top_prompts
                ]

                logger.info(
                    f"Best prompts for {benchmark_name}: "
                    f"{len(top_prompts)} with avg score "
                    f"{np.mean([s for _, s in top_prompts]):.2%}"
                )

    async def _update_mcp_metrics(
        self,
        results: Dict[str, BenchmarkResult]
    ):
        """Update MCP server performance metrics"""

        for result in results.values():
            for task_result in result.task_results:
                server_name = task_result.get("server")
                if server_name:
                    self.mcp_optimizer.update_metrics(
                        server_name,
                        task_result["success"],
                        task_result.get("latency_ms", 0)
                    )