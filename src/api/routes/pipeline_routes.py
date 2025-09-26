"""
Pipeline API Routes

RESTful endpoints for pipeline control and monitoring.
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from ..models.pipeline_models import (
    PipelineStartRequest,
    PipelineControlRequest,
    PhaseConfigRequest,
    CheckpointRequest,
    PresetRequest,
    PipelineStatusResponse,
    PipelineResultResponse,
    SwarmStatusResponse,
    CheckpointResponse,
    ConfigPresetResponse,
    ExecutionHistoryResponse,
    QualityGateResponse,
    HealthCheckResponse,
    ErrorResponse,
    PipelinePhase,
)
from ..services.pipeline_service import PipelineService

router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])

# Initialize service
pipeline_service = PipelineService()


@router.post("/start", response_model=dict)
async def start_pipeline(request: PipelineStartRequest):
    """
    Start a new pipeline execution.

    Returns:
        session_id: Unique identifier for tracking the pipeline execution
    """
    try:
        session_id = await pipeline_service.start_pipeline(
            phases=request.phases,
            config=request.config,
            enable_monitoring=request.enable_monitoring,
            enable_checkpoints=request.enable_checkpoints,
            swarm_topology=request.swarm_topology,
            max_agents=request.max_agents,
        )

        return {"session_id": session_id, "status": "started"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/control", response_model=dict)
async def control_pipeline(request: PipelineControlRequest):
    """
    Control pipeline execution (pause, resume, stop, cancel).

    Actions:
        - pause: Temporarily pause execution
        - resume: Resume from pause
        - stop: Gracefully stop execution
        - cancel: Force terminate execution
    """
    try:
        success = await pipeline_service.control_pipeline(
            session_id=request.session_id, action=request.action, force=request.force
        )

        if success:
            return {"status": "success", "action": request.action}
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot perform action '{request.action}' in current state",
            )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{session_id}", response_model=PipelineStatusResponse)
async def get_pipeline_status(session_id: str):
    """
    Get current pipeline status and progress.

    Returns detailed information about:
        - Execution status
        - Phase progress
        - Active agents
        - Performance metrics
        - Estimated completion time
    """
    try:
        status = await pipeline_service.get_pipeline_status(session_id)
        return status

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/swarm/{session_id}", response_model=SwarmStatusResponse)
async def get_swarm_status(session_id: str):
    """
    Get swarm coordination status.

    Returns information about:
        - Swarm topology
        - Agent distribution
        - Resource utilization
        - Communication latency
    """
    try:
        status = await pipeline_service.get_swarm_status(session_id)
        return status

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quality-gates/{session_id}", response_model=QualityGateResponse)
async def validate_quality_gates(
    session_id: str, phase: Optional[PipelinePhase] = None
):
    """
    Run quality gate validation for a phase.

    Validates:
        - Performance metrics
        - Theater detection
        - Compliance requirements
        - Security checks
    """
    try:
        result = await pipeline_service.validate_quality_gates(session_id, phase)
        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/checkpoint/save", response_model=CheckpointResponse)
async def save_checkpoint(request: CheckpointRequest):
    """
    Save pipeline checkpoint for recovery.

    Saves:
        - Model state
        - Swarm configuration
        - Execution state
        - Metrics history
    """
    try:
        checkpoint_id = f"checkpoint_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # TODO: Implement actual checkpoint saving
        return CheckpointResponse(
            checkpoint_id=checkpoint_id,
            session_id=request.session_id,
            timestamp=datetime.utcnow(),
            size_mb=0.0,
            model_included=request.include_model_state,
            swarm_included=request.include_swarm_state,
            path=f"/checkpoints/{checkpoint_id}",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/checkpoint/load/{checkpoint_id}", response_model=dict)
async def load_checkpoint(checkpoint_id: str):
    """
    Load pipeline from checkpoint.

    Restores:
        - Model state
        - Swarm configuration
        - Execution state
    """
    try:
        # TODO: Implement actual checkpoint loading
        return {"status": "loaded", "checkpoint_id": checkpoint_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=List[ExecutionHistoryResponse])
async def get_execution_history(
    limit: int = Query(10, ge=1, le=100),
    status: Optional[str] = None,
):
    """
    Get pipeline execution history.

    Filters:
        - limit: Maximum number of entries
        - status: Filter by execution status
    """
    try:
        history = pipeline_service.execution_history[-limit:]

        if status:
            history = [h for h in history if h["status"] == status]

        return [
            ExecutionHistoryResponse(
                session_id=h["session_id"],
                start_time=h["start_time"],
                end_time=h.get("end_time"),
                status=h["status"],
                phases=[PipelinePhase(p) for p in h["phases"]],
                success=h["success"],
                duration_seconds=h.get("duration_seconds"),
                final_metrics=h.get("final_metrics"),
            )
            for h in history
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preset/save", response_model=ConfigPresetResponse)
async def save_preset(request: PresetRequest):
    """
    Save configuration preset for reuse.
    """
    try:
        pipeline_service.presets[request.preset_name] = request.config

        return ConfigPresetResponse(
            preset_name=request.preset_name,
            config=request.config,
            created_at=datetime.utcnow(),
            usage_count=0,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preset/{preset_name}", response_model=ConfigPresetResponse)
async def load_preset(preset_name: str):
    """
    Load configuration preset.
    """
    try:
        if preset_name not in pipeline_service.presets:
            raise HTTPException(status_code=404, detail="Preset not found")

        config = pipeline_service.presets[preset_name]

        return ConfigPresetResponse(
            preset_name=preset_name,
            config=config,
            created_at=datetime.utcnow(),
            usage_count=0,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/presets", response_model=List[str])
async def list_presets():
    """
    List available configuration presets.
    """
    return list(pipeline_service.presets.keys())


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    API health check endpoint.
    """
    import psutil

    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=0.0,  # TODO: Track actual uptime
        active_sessions=len(pipeline_service.sessions),
        total_memory_mb=psutil.virtual_memory().total / 1024 / 1024,
        available_memory_mb=psutil.virtual_memory().available / 1024 / 1024,
    )