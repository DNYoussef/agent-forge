"""
Pipeline Service Layer

Business logic for pipeline execution, coordination, and state management.
Integrates with SwarmCoordinator and UnifiedPipeline.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...swarm_coordinator import SwarmCoordinator, SwarmConfig, SwarmTopology as SwarmTopologyEnum
from ...swarm_execution import SwarmExecutionManager
from ...swarm_monitor import create_swarm_monitor
from ...unified_pipeline import UnifiedPipeline, UnifiedConfig

from ..models.pipeline_models import (
    PipelinePhase,
    PipelineStatus,
    SwarmTopology,
    AgentStatus,
    PhaseMetrics,
    PipelineStatusResponse,
    PipelineResultResponse,
    SwarmStatusResponse,
    CheckpointResponse,
    ExecutionHistoryResponse,
    QualityGateResponse,
    QualityGateResult,
    TheaterDetectionResult,
)

logger = logging.getLogger(__name__)


class PipelineSession:
    """Represents an active pipeline execution session."""

    def __init__(self, session_id: str, config: Dict[str, Any]):
        self.session_id = session_id
        self.config = config
        self.status = PipelineStatus.INITIALIZING
        self.current_phase: Optional[PipelinePhase] = None
        self.phases_completed: List[PipelinePhase] = []
        self.phases_failed: List[PipelinePhase] = []
        self.start_time = datetime.utcnow()
        self.end_time: Optional[datetime] = None
        self.coordinator: Optional[SwarmCoordinator] = None
        self.execution_manager: Optional[SwarmExecutionManager] = None
        self.monitor = None
        self.phase_results = []
        self.checkpoints = {}
        self.metrics_history = []


class PipelineService:
    """Service layer for pipeline operations."""

    def __init__(self):
        self.sessions: Dict[str, PipelineSession] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.presets: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)

        # Load default presets
        self._load_default_presets()

    async def start_pipeline(
        self,
        phases: List[PipelinePhase],
        config: Optional[Dict[str, Any]] = None,
        enable_monitoring: bool = True,
        enable_checkpoints: bool = True,
        swarm_topology: SwarmTopology = SwarmTopology.HIERARCHICAL,
        max_agents: int = 50,
    ) -> str:
        """Start a new pipeline execution session."""

        session_id = str(uuid.uuid4())
        self.logger.info(f"Starting pipeline session {session_id}")

        # Create session
        session = PipelineSession(
            session_id=session_id,
            config={
                "phases": [p.value for p in phases],
                "enable_monitoring": enable_monitoring,
                "enable_checkpoints": enable_checkpoints,
                "swarm_topology": swarm_topology.value,
                "max_agents": max_agents,
                "custom_config": config or {},
            },
        )
        self.sessions[session_id] = session

        # Initialize swarm coordinator
        swarm_config = SwarmConfig(
            topology=SwarmTopologyEnum(swarm_topology.value),
            max_agents=max_agents,
        )

        session.coordinator = SwarmCoordinator(swarm_config)
        await session.coordinator.initialize_swarm()

        # Initialize execution manager
        session.execution_manager = SwarmExecutionManager(session.coordinator)

        # Initialize monitoring if enabled
        if enable_monitoring:
            session.monitor = create_swarm_monitor(session.coordinator)
            await session.monitor.start_monitoring()

        session.status = PipelineStatus.RUNNING

        # Start execution in background
        asyncio.create_task(self._execute_pipeline(session_id, phases, config))

        return session_id

    async def _execute_pipeline(
        self,
        session_id: str,
        phases: List[PipelinePhase],
        config: Optional[Dict[str, Any]],
    ):
        """Execute pipeline phases (runs in background)."""

        session = self.sessions.get(session_id)
        if not session:
            self.logger.error(f"Session {session_id} not found")
            return

        try:
            current_data = {"model": None}

            for phase in phases:
                if session.status not in [PipelineStatus.RUNNING, PipelineStatus.PAUSED]:
                    self.logger.info(f"Pipeline {session_id} stopped")
                    break

                # Wait if paused
                while session.status == PipelineStatus.PAUSED:
                    await asyncio.sleep(0.5)

                session.current_phase = phase
                self.logger.info(f"Executing phase: {phase.value}")

                # Map phase enum to integer (1-8)
                phase_map = {
                    PipelinePhase.COGNATE: 1,
                    PipelinePhase.EVOMERGE: 2,
                    PipelinePhase.QUIETSTAR: 3,
                    PipelinePhase.BITNET: 4,
                    PipelinePhase.TRAINING: 5,
                    PipelinePhase.BAKING: 6,
                    PipelinePhase.ADAS: 7,
                    PipelinePhase.COMPRESSION: 8,
                }

                phase_num = phase_map[phase]

                # Execute phase
                result = await session.execution_manager.execute_pipeline_phase(
                    phase_num, current_data
                )

                session.phase_results.append(result)

                if result.success:
                    session.phases_completed.append(phase)
                    current_data = {
                        "model": result.model,
                        "previous_phase_result": result,
                        "pipeline_state": session.coordinator.memory,
                    }
                else:
                    session.phases_failed.append(phase)
                    self.logger.error(f"Phase {phase.value} failed: {result.error}")
                    session.status = PipelineStatus.FAILED
                    break

            # Mark as completed if all phases succeeded
            if session.status == PipelineStatus.RUNNING:
                session.status = PipelineStatus.COMPLETED

            session.end_time = datetime.utcnow()

            # Stop monitoring
            if session.monitor:
                await session.monitor.stop_monitoring()

            # Save to history
            self._save_to_history(session)

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            session.status = PipelineStatus.FAILED
            session.end_time = datetime.utcnow()

    async def get_pipeline_status(self, session_id: str) -> PipelineStatusResponse:
        """Get current pipeline status."""

        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Calculate progress
        total_phases = len(session.config["phases"])
        completed = len(session.phases_completed)
        total_progress = (completed / total_phases * 100) if total_phases > 0 else 0

        # Get elapsed time
        elapsed = (
            (session.end_time or datetime.utcnow()) - session.start_time
        ).total_seconds()

        # Estimate remaining time
        estimated_remaining = None
        if completed > 0 and session.status == PipelineStatus.RUNNING:
            avg_phase_time = elapsed / completed
            remaining_phases = total_phases - completed
            estimated_remaining = avg_phase_time * remaining_phases

        # Get agent status
        active_agents = []
        if session.coordinator:
            for agent_id, agent in session.coordinator.agents.items():
                active_agents.append(
                    AgentStatus(
                        agent_id=agent_id,
                        role=agent.config.role.value,
                        phase=agent.config.phase,
                        state=agent.state,
                        current_task=agent.current_task,
                        memory_usage_mb=agent.config.max_memory,
                        cpu_usage_percent=0.0,  # TODO: Get actual CPU usage
                        uptime_seconds=(
                            (datetime.utcnow() - agent.start_time).total_seconds()
                            if agent.start_time
                            else 0
                        ),
                    )
                )

        # Build phase metrics
        phase_metrics = []
        for result in session.phase_results:
            phase_name = list(PipelinePhase)[result.phase - 1].value
            phase_metrics.append(
                PhaseMetrics(
                    phase=phase_name,
                    status=PipelineStatus.COMPLETED if result.success else PipelineStatus.FAILED,
                    progress_percent=100.0 if result.success else 0.0,
                    duration_seconds=result.duration_seconds,
                    memory_usage_mb=0.0,  # TODO: Get actual memory
                    custom_metrics=result.metrics or {},
                    error_message=result.error if not result.success else None,
                )
            )

        # Get remaining phases
        all_phases = [PipelinePhase(p) for p in session.config["phases"]]
        phases_remaining = [
            p for p in all_phases if p not in session.phases_completed
        ]

        return PipelineStatusResponse(
            session_id=session_id,
            status=session.status,
            current_phase=session.current_phase,
            phases_completed=session.phases_completed,
            phases_remaining=phases_remaining,
            total_progress_percent=total_progress,
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=estimated_remaining,
            phase_metrics=phase_metrics,
            agent_count=len(active_agents),
            active_agents=active_agents,
            last_checkpoint=session.checkpoints.get("latest"),
        )

    async def control_pipeline(
        self, session_id: str, action: str, force: bool = False
    ) -> bool:
        """Control pipeline execution (pause, resume, stop, cancel)."""

        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if action == "pause":
            if session.status == PipelineStatus.RUNNING:
                session.status = PipelineStatus.PAUSED
                self.logger.info(f"Pipeline {session_id} paused")
                return True
            return False

        elif action == "resume":
            if session.status == PipelineStatus.PAUSED:
                session.status = PipelineStatus.RUNNING
                self.logger.info(f"Pipeline {session_id} resumed")
                return True
            return False

        elif action == "stop":
            if session.status in [PipelineStatus.RUNNING, PipelineStatus.PAUSED]:
                session.status = PipelineStatus.CANCELLED
                session.end_time = datetime.utcnow()
                self.logger.info(f"Pipeline {session_id} stopped")
                return True
            return False

        elif action == "cancel":
            if force or session.status in [PipelineStatus.RUNNING, PipelineStatus.PAUSED]:
                session.status = PipelineStatus.CANCELLED
                session.end_time = datetime.utcnow()
                if session.monitor:
                    await session.monitor.stop_monitoring()
                self.logger.info(f"Pipeline {session_id} cancelled")
                return True
            return False

        return False

    async def get_swarm_status(self, session_id: str) -> SwarmStatusResponse:
        """Get swarm coordination status."""

        session = self.sessions.get(session_id)
        if not session or not session.coordinator:
            raise ValueError(f"Session {session_id} not found or not initialized")

        coordinator_status = await session.coordinator.get_swarm_status()

        agents = []
        for agent_id, agent in session.coordinator.agents.items():
            agents.append(
                AgentStatus(
                    agent_id=agent_id,
                    role=agent.config.role.value,
                    phase=agent.config.phase,
                    state=agent.state,
                    current_task=agent.current_task,
                    memory_usage_mb=agent.config.max_memory,
                    cpu_usage_percent=0.0,
                    uptime_seconds=(
                        (datetime.utcnow() - agent.start_time).total_seconds()
                        if agent.start_time
                        else 0
                    ),
                )
            )

        active_count = len([a for a in agents if a.state == "active"])
        idle_count = len([a for a in agents if a.state == "idle"])

        return SwarmStatusResponse(
            topology=SwarmTopology(session.config["swarm_topology"]),
            total_agents=len(agents),
            active_agents=active_count,
            idle_agents=idle_count,
            agents=agents,
            memory_usage_mb=sum(a.memory_usage_mb for a in agents),
            cpu_usage_percent=0.0,
            communication_latency_ms=0.0,  # TODO: Calculate actual latency
        )

    async def validate_quality_gates(
        self, session_id: str, phase: Optional[PipelinePhase] = None
    ) -> QualityGateResponse:
        """Run quality gate validation."""

        session = self.sessions.get(session_id)
        if not session or not session.monitor:
            raise ValueError(f"Session {session_id} not found or monitoring not enabled")

        target_phase = phase or session.current_phase
        if not target_phase:
            raise ValueError("No phase specified and no current phase")

        # Map phase to number
        phase_map = {
            PipelinePhase.COGNATE: 1,
            PipelinePhase.EVOMERGE: 2,
            PipelinePhase.QUIETSTAR: 3,
            PipelinePhase.BITNET: 4,
            PipelinePhase.TRAINING: 5,
            PipelinePhase.BAKING: 6,
            PipelinePhase.ADAS: 7,
            PipelinePhase.COMPRESSION: 8,
        }

        phase_num = phase_map[target_phase]

        # Get phase data
        phase_data = session.coordinator.memory.get("phase_states", {}).get(
            phase_num, {}
        )

        # Run quality gates
        result = await session.monitor.validate_quality_gates(phase_num, phase_data)

        # Convert to response model
        gate_results = []
        for gate_name, gate_result in result.get("gate_results", {}).items():
            gate_results.append(
                QualityGateResult(
                    gate_name=gate_name,
                    passed=gate_result.get("passed", False),
                    score=gate_result.get("score", 0.0),
                    threshold=gate_result.get("threshold", 0.0),
                    details=gate_result.get("details", {}),
                    recommendations=gate_result.get("recommendations", []),
                )
            )

        # Theater detection
        theater_result = None
        if "theater_detection" in result:
            td = result["theater_detection"]
            theater_result = TheaterDetectionResult(
                theater_detected=td.get("theater_detected", False),
                confidence=td.get("confidence", 0.0),
                theater_score=td.get("theater_score", 0.0),
                indicators=td.get("indicators", {}),
                recommendations=td.get("recommendations", []),
            )

        return QualityGateResponse(
            phase=target_phase,
            all_gates_passed=result.get("all_gates_passed", False),
            gates_passed=result.get("gates_passed", 0),
            gates_failed=result.get("gates_failed", 0),
            gate_results=gate_results,
            theater_detection=theater_result,
            blocking_failures=result.get("blocking_failures", []),
            can_proceed=result.get("can_proceed", True),
        )

    def _save_to_history(self, session: PipelineSession):
        """Save session to execution history."""

        history_entry = {
            "session_id": session.session_id,
            "start_time": session.start_time,
            "end_time": session.end_time,
            "status": session.status.value,
            "phases": session.config["phases"],
            "success": session.status == PipelineStatus.COMPLETED,
            "duration_seconds": (
                (session.end_time - session.start_time).total_seconds()
                if session.end_time
                else None
            ),
        }

        self.execution_history.append(history_entry)

    def _load_default_presets(self):
        """Load default configuration presets."""

        self.presets = {
            "quick_test": {
                "phases": ["cognate", "evomerge"],
                "max_agents": 20,
                "enable_monitoring": True,
            },
            "full_pipeline": {
                "phases": [
                    "cognate",
                    "evomerge",
                    "quietstar",
                    "bitnet",
                    "training",
                    "baking",
                    "adas",
                    "compression",
                ],
                "max_agents": 50,
                "enable_monitoring": True,
            },
            "compression_only": {
                "phases": ["bitnet", "compression"],
                "max_agents": 30,
                "enable_monitoring": True,
            },
        }