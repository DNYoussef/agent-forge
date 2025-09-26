"""
Test data generators for pipeline execution testing.

Provides realistic mock data for:
- Pipeline configurations
- Phase execution states
- Agent status
- Metrics and performance data
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import uuid


@dataclass
class MockPipelineConfig:
    """Mock pipeline configuration."""
    phases: List[str]
    swarm_topology: str = "hierarchical"
    max_agents: int = 50
    enable_monitoring: bool = True
    enable_checkpoints: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


class PipelineDataGenerator:
    """Generate realistic test data for pipeline execution."""

    PHASES = [
        "cognate", "evomerge", "quietstar", "bitnet",
        "training", "baking", "adas", "compression"
    ]

    SWARM_TOPOLOGIES = ["hierarchical", "mesh", "star", "ring"]

    AGENT_ROLES = [
        "coordinator", "worker", "validator", "optimizer",
        "monitor", "data_loader", "trainer", "evaluator"
    ]

    @staticmethod
    def generate_session_id() -> str:
        """Generate unique session ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"session_{timestamp}_{unique_id}"

    @staticmethod
    def generate_pipeline_config(
        num_phases: Optional[int] = None,
        include_all_phases: bool = False
    ) -> MockPipelineConfig:
        """Generate pipeline configuration."""
        if include_all_phases:
            phases = PipelineDataGenerator.PHASES.copy()
        elif num_phases:
            phases = random.sample(PipelineDataGenerator.PHASES, min(num_phases, len(PipelineDataGenerator.PHASES)))
        else:
            phases = random.sample(PipelineDataGenerator.PHASES, random.randint(1, 4))

        topology = random.choice(PipelineDataGenerator.SWARM_TOPOLOGIES)
        max_agents = random.choice([10, 25, 50, 75, 100])

        # Generate phase-specific configs
        config = {}
        for phase in phases:
            config[phase] = PipelineDataGenerator._generate_phase_config(phase)

        return MockPipelineConfig(
            phases=phases,
            swarm_topology=topology,
            max_agents=max_agents,
            enable_monitoring=random.choice([True, False]),
            enable_checkpoints=random.choice([True, False]),
            config=config
        )

    @staticmethod
    def _generate_phase_config(phase: str) -> Dict[str, Any]:
        """Generate phase-specific configuration."""
        configs = {
            "cognate": {
                "base_models": random.sample(["gpt2", "llama", "mistral", "phi"], random.randint(1, 3)),
                "init_strategy": random.choice(["random", "pretrained", "custom"])
            },
            "evomerge": {
                "population_size": random.choice([10, 20, 50, 100]),
                "generations": random.randint(5, 50),
                "mutation_rate": round(random.uniform(0.01, 0.3), 2),
                "crossover_rate": round(random.uniform(0.5, 0.9), 2)
            },
            "quietstar": {
                "thought_length": random.choice([50, 100, 200]),
                "num_thoughts": random.choice([4, 8, 16, 32]),
                "start_token": "<|startofthought|>",
                "end_token": "<|endofthought|>",
                "mix_heads": random.choice([True, False])
            },
            "bitnet": {
                "precision": random.choice(["1.58bit", "2bit", "4bit"]),
                "quantization_method": random.choice(["absmax", "minmax", "percentile"])
            },
            "training": {
                "batch_size": random.choice([8, 16, 32, 64]),
                "learning_rate": random.choice([1e-5, 5e-5, 1e-4, 5e-4]),
                "epochs": random.randint(1, 10),
                "optimizer": random.choice(["adamw", "sgd", "adafactor"]),
                "warmup_steps": random.randint(100, 1000)
            },
            "baking": {
                "tool_types": random.sample(
                    ["calculator", "search", "code_execution", "web_browser", "file_system"],
                    random.randint(1, 3)
                ),
                "enable_tool_calling": random.choice([True, False])
            },
            "adas": {
                "search_space": random.choice(["transformer", "mamba", "hybrid"]),
                "iterations": random.choice([50, 100, 200]),
                "objective": random.choice(["perplexity", "accuracy", "f1_score"])
            },
            "compression": {
                "method": random.choice(["pruning", "distillation", "quantization"]),
                "target_sparsity": round(random.uniform(0.3, 0.7), 2),
                "preserve_accuracy": random.choice([True, False])
            }
        }

        return configs.get(phase, {})

    @staticmethod
    def generate_pipeline_status(
        session_id: str,
        current_phase: Optional[str] = None,
        status: str = "running"
    ) -> Dict[str, Any]:
        """Generate pipeline status response."""
        if not current_phase:
            current_phase = random.choice(PipelineDataGenerator.PHASES)

        phase_index = PipelineDataGenerator.PHASES.index(current_phase)
        completed_phases = PipelineDataGenerator.PHASES[:phase_index]
        remaining_phases = PipelineDataGenerator.PHASES[phase_index + 1:]

        progress_percent = round((len(completed_phases) / len(PipelineDataGenerator.PHASES)) * 100, 1)

        return {
            "session_id": session_id,
            "status": status,
            "current_phase": current_phase,
            "progress_percent": progress_percent,
            "phases": {
                "completed": completed_phases,
                "current": current_phase,
                "remaining": remaining_phases
            },
            "metrics": PipelineDataGenerator.generate_metrics(),
            "start_time": (datetime.utcnow() - timedelta(hours=random.randint(0, 5))).isoformat(),
            "estimated_completion": (datetime.utcnow() + timedelta(hours=random.randint(1, 10))).isoformat(),
            "active_agents": random.randint(5, 50),
            "total_tasks": random.randint(100, 1000),
            "completed_tasks": random.randint(50, 500)
        }

    @staticmethod
    def generate_metrics() -> Dict[str, Any]:
        """Generate performance metrics."""
        return {
            "cpu_percent": round(random.uniform(20, 95), 1),
            "memory_mb": round(random.uniform(1024, 8192), 1),
            "gpu_utilization": round(random.uniform(0, 100), 1),
            "throughput_ops": round(random.uniform(100, 5000), 1),
            "latency_ms": round(random.uniform(10, 500), 1),
            "tokens_per_second": round(random.uniform(100, 10000), 1),
            "loss": round(random.uniform(0.1, 5.0), 3),
            "accuracy": round(random.uniform(0.5, 0.99), 3)
        }

    @staticmethod
    def generate_agent_status(agent_id: str, phase: str) -> Dict[str, Any]:
        """Generate agent status data."""
        return {
            "agent_id": agent_id,
            "role": random.choice(PipelineDataGenerator.AGENT_ROLES),
            "phase": phase,
            "state": random.choice(["idle", "busy", "waiting", "completed"]),
            "current_task": f"task_{random.randint(1, 1000)}",
            "memory_usage_mb": round(random.uniform(100, 2048), 1),
            "cpu_usage_percent": round(random.uniform(0, 100), 1),
            "uptime_seconds": random.randint(60, 36000),
            "tasks_completed": random.randint(0, 500),
            "tasks_failed": random.randint(0, 10),
            "last_heartbeat": datetime.utcnow().isoformat()
        }

    @staticmethod
    def generate_swarm_status(
        session_id: str,
        topology: str = "hierarchical",
        num_agents: int = 50
    ) -> Dict[str, Any]:
        """Generate swarm coordination status."""
        agent_distribution = {}
        remaining = num_agents

        for phase in random.sample(PipelineDataGenerator.PHASES, random.randint(2, 5)):
            count = random.randint(1, min(remaining, 20))
            agent_distribution[phase] = count
            remaining -= count

        return {
            "session_id": session_id,
            "topology": topology,
            "active_agents": num_agents,
            "agent_distribution": agent_distribution,
            "coordination_metrics": {
                "message_latency_ms": round(random.uniform(1, 50), 2),
                "throughput_msg_per_sec": round(random.uniform(100, 10000), 1),
                "failed_communications": random.randint(0, 10)
            },
            "resource_utilization": {
                "total_cpu_percent": round(random.uniform(30, 90), 1),
                "total_memory_gb": round(random.uniform(10, 100), 1),
                "total_gpu_percent": round(random.uniform(0, 100), 1)
            }
        }

    @staticmethod
    def generate_quality_gate_results(
        session_id: str,
        phases: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate quality gate validation results."""
        if not phases:
            phases = random.sample(PipelineDataGenerator.PHASES, random.randint(1, 4))

        results = []
        for phase in phases:
            passed = random.choice([True, True, True, False])  # 75% pass rate
            results.append({
                "phase": phase,
                "passed": passed,
                "metrics": {
                    "accuracy": round(random.uniform(0.7, 0.99), 3),
                    "loss": round(random.uniform(0.1, 3.0), 3),
                    "perplexity": round(random.uniform(1.5, 15.0), 2)
                },
                "thresholds": {
                    "min_accuracy": 0.75,
                    "max_loss": 2.0,
                    "max_perplexity": 10.0
                },
                "details": f"Quality gate for {phase}: {'PASSED' if passed else 'FAILED'}"
            })

        overall_passed = all(r["passed"] for r in results)

        return {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "overall_passed": overall_passed,
            "results": results,
            "summary": f"{sum(r['passed'] for r in results)}/{len(results)} gates passed"
        }

    @staticmethod
    def generate_checkpoint_data(
        session_id: str,
        include_model: bool = True,
        include_swarm: bool = True
    ) -> Dict[str, Any]:
        """Generate checkpoint metadata."""
        checkpoint_id = f"checkpoint_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        return {
            "checkpoint_id": checkpoint_id,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "size_mb": round(random.uniform(100, 5000), 1),
            "model_included": include_model,
            "swarm_included": include_swarm,
            "path": f"/checkpoints/{checkpoint_id}",
            "metadata": {
                "current_phase": random.choice(PipelineDataGenerator.PHASES),
                "progress_percent": round(random.uniform(0, 100), 1),
                "model_parameters": random.randint(1_000_000, 10_000_000_000),
                "optimizer_state_size_mb": round(random.uniform(50, 1000), 1)
            }
        }

    @staticmethod
    def generate_execution_history(count: int = 10) -> List[Dict[str, Any]]:
        """Generate execution history entries."""
        history = []

        for _ in range(count):
            session_id = PipelineDataGenerator.generate_session_id()
            start_time = datetime.utcnow() - timedelta(days=random.randint(0, 30))
            duration = random.randint(300, 36000)
            end_time = start_time + timedelta(seconds=duration)

            status = random.choice(["completed", "failed", "cancelled", "completed", "completed"])
            success = status == "completed"

            phases = random.sample(PipelineDataGenerator.PHASES, random.randint(1, 8))

            history.append({
                "session_id": session_id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat() if status != "running" else None,
                "status": status,
                "phases": phases,
                "success": success,
                "duration_seconds": duration if status != "running" else None,
                "final_metrics": PipelineDataGenerator.generate_metrics() if success else None
            })

        return sorted(history, key=lambda x: x["start_time"], reverse=True)

    @staticmethod
    def generate_websocket_event(
        event_type: str,
        session_id: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate WebSocket event message."""
        if data is None:
            data = {}

        events = {
            "pipeline_progress": {
                "current_phase": random.choice(PipelineDataGenerator.PHASES),
                "progress_percent": round(random.uniform(0, 100), 1),
                "status": random.choice(["running", "paused", "completed"])
            },
            "agent_update": {
                "agent_id": f"agent_{random.randint(1, 100)}",
                "status": random.choice(["spawned", "active", "idle", "terminated"]),
                "task": f"task_{random.randint(1, 1000)}"
            },
            "metrics_stream": {
                "metrics": PipelineDataGenerator.generate_metrics()
            },
            "phase_completion": {
                "phase": random.choice(PipelineDataGenerator.PHASES),
                "success": random.choice([True, True, True, False]),
                "duration_seconds": random.randint(60, 7200)
            }
        }

        event_data = events.get(event_type, data)

        return {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "data": event_data
        }


class MockPipelineSimulator:
    """Simulate realistic pipeline execution scenarios."""

    def __init__(self):
        self.generator = PipelineDataGenerator()
        self.current_phase_index = 0
        self.phases = []

    def start_simulation(self, phases: List[str]) -> str:
        """Start a simulated pipeline run."""
        self.phases = phases
        self.current_phase_index = 0
        return self.generator.generate_session_id()

    def advance_phase(self) -> Optional[str]:
        """Advance to next phase."""
        if self.current_phase_index < len(self.phases) - 1:
            self.current_phase_index += 1
            return self.phases[self.current_phase_index]
        return None

    def get_current_status(self, session_id: str) -> Dict[str, Any]:
        """Get current simulation status."""
        if self.current_phase_index < len(self.phases):
            current_phase = self.phases[self.current_phase_index]
            status = "running"
        else:
            current_phase = self.phases[-1] if self.phases else "completed"
            status = "completed"

        return self.generator.generate_pipeline_status(
            session_id=session_id,
            current_phase=current_phase,
            status=status
        )

    def simulate_failure(self, session_id: str) -> Dict[str, Any]:
        """Simulate pipeline failure."""
        return {
            "session_id": session_id,
            "status": "failed",
            "error": "Simulated failure in phase: " + self.phases[self.current_phase_index],
            "phase": self.phases[self.current_phase_index],
            "timestamp": datetime.utcnow().isoformat()
        }