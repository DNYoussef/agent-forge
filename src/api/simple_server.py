#!/usr/bin/env python3
"""
Simple Agent Forge API Server
Provides REST and WebSocket endpoints for the UI
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import asyncio
import json
import uvicorn
from typing import Dict, List, Optional
import random

app = FastAPI(title="Agent Forge API", version="1.0.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for demo
pipeline_sessions: Dict = {}
phase_sessions: Dict = {}
connected_clients: List[WebSocket] = []

@app.get("/")
async def root():
    return {"message": "Agent Forge API Server Running", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Pipeline Management
@app.post("/api/pipeline")
async def pipeline_control(request: dict):
    action = request.get("action")

    if action == "start":
        session_id = f"pipeline-{int(datetime.now().timestamp())}"
        pipeline_sessions[session_id] = {
            "id": session_id,
            "status": "running",
            "phases": request.get("phases", [1, 2, 3, 4, 5, 6, 7, 8]),
            "currentPhase": 1,
            "progress": 0,
            "startTime": datetime.now().isoformat(),
            "config": request.get("config", {})
        }

        # Start background progress simulation
        asyncio.create_task(simulate_pipeline_progress(session_id))

        return {"success": True, "sessionId": session_id}

    elif action == "stop":
        session_id = request.get("sessionId")
        if session_id in pipeline_sessions:
            pipeline_sessions[session_id]["status"] = "stopped"
        return {"success": True}

    elif action == "pause":
        session_id = request.get("sessionId")
        if session_id in pipeline_sessions:
            pipeline_sessions[session_id]["status"] = "paused"
        return {"success": True}

    elif action == "resume":
        session_id = request.get("sessionId")
        if session_id in pipeline_sessions:
            pipeline_sessions[session_id]["status"] = "running"
        return {"success": True}

    return {"error": "Invalid action"}

@app.get("/api/pipeline")
async def get_pipeline_status(sessionId: Optional[str] = None):
    if sessionId and sessionId in pipeline_sessions:
        return pipeline_sessions[sessionId]
    return list(pipeline_sessions.values())

# Stats endpoint
@app.get("/api/stats")
async def get_stats():
    return {
        "totalAgents": random.randint(100, 500),
        "successRate": random.randint(85, 98),
        "activePipelines": len([p for p in pipeline_sessions.values() if p["status"] == "running"]),
        "avgPipelineTime": random.randint(15, 45)
    }

# Phase-specific endpoints
@app.post("/api/phases/cognate")
async def cognate_control(request: dict):
    action = request.get("action")

    if action == "start":
        session_id = f"cognate-{int(datetime.now().timestamp())}"
        phase_sessions[session_id] = {
            "id": session_id,
            "phase": "cognate",
            "status": "running",
            "config": request.get("config", {}),
            "startTime": datetime.now().isoformat(),
            "metrics": {
                "loss": 2.5,
                "perplexity": 12.18,
                "grokProgress": 0,
                "modelParams": 25000000
            }
        }

        asyncio.create_task(simulate_phase_progress(session_id, "cognate"))
        return {"success": True, "sessionId": session_id}

    elif action == "stop":
        session_id = request.get("sessionId")
        if session_id in phase_sessions:
            phase_sessions[session_id]["status"] = "stopped"
        return {"success": True}

    return {"error": "Invalid action"}

@app.get("/api/phases/cognate")
async def get_cognate_status(sessionId: str):
    if sessionId in phase_sessions:
        return phase_sessions[sessionId]
    return {"error": "Session not found"}

@app.post("/api/phases/evomerge")
async def evomerge_control(request: dict):
    action = request.get("action")

    if action == "start":
        session_id = f"evomerge-{int(datetime.now().timestamp())}"
        phase_sessions[session_id] = {
            "id": session_id,
            "phase": "evomerge",
            "status": "running",
            "config": request.get("config", {}),
            "startTime": datetime.now().isoformat(),
            "currentGeneration": 0,
            "metrics": {
                "bestFitness": 0,
                "avgFitness": 0,
                "diversity": 1.0,
                "paretoFront": []
            }
        }

        asyncio.create_task(simulate_phase_progress(session_id, "evomerge"))
        return {"success": True, "sessionId": session_id}

    return {"error": "Invalid action"}

# WebSocket endpoint
@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)

    try:
        while True:
            # Send periodic updates
            await asyncio.sleep(1)

            # Send pipeline updates
            for session_id, session in pipeline_sessions.items():
                if session["status"] == "running":
                    await websocket.send_json({
                        "type": "pipeline.status",
                        "data": session
                    })

            # Send phase updates
            for session_id, session in phase_sessions.items():
                if session["status"] == "running":
                    await websocket.send_json({
                        "type": f"phase.{session['phase']}.metrics",
                        "data": session["metrics"]
                    })

    except WebSocketDisconnect:
        connected_clients.remove(websocket)

# Background tasks
async def simulate_pipeline_progress(session_id: str):
    """Simulate pipeline progress"""
    while session_id in pipeline_sessions:
        session = pipeline_sessions[session_id]

        if session["status"] != "running":
            await asyncio.sleep(1)
            continue

        # Update progress
        session["progress"] = min(100, session["progress"] + random.uniform(0.5, 2))

        # Update current phase
        phases = session["phases"]
        phase_progress = session["progress"] / (100 / len(phases))
        session["currentPhase"] = min(int(phase_progress) + 1, len(phases))

        # Check completion
        if session["progress"] >= 100:
            session["status"] = "completed"
            session["endTime"] = datetime.now().isoformat()
            break

        await asyncio.sleep(0.5)

async def simulate_phase_progress(session_id: str, phase_type: str):
    """Simulate individual phase progress"""
    while session_id in phase_sessions:
        session = phase_sessions[session_id]

        if session["status"] != "running":
            await asyncio.sleep(1)
            continue

        if phase_type == "cognate":
            # Update Cognate metrics
            metrics = session["metrics"]
            metrics["loss"] = max(0.1, metrics["loss"] - random.uniform(0.01, 0.05))
            metrics["perplexity"] = max(1.1, metrics["perplexity"] - random.uniform(0.1, 0.3))
            metrics["grokProgress"] = min(100, metrics["grokProgress"] + random.uniform(1, 3))

            if metrics["grokProgress"] >= 100:
                session["status"] = "completed"
                break

        elif phase_type == "evomerge":
            # Update EvoMerge metrics
            metrics = session["metrics"]
            session["currentGeneration"] = min(50, session["currentGeneration"] + 1)
            metrics["bestFitness"] = min(0.99, metrics["bestFitness"] + random.uniform(0.01, 0.03))
            metrics["avgFitness"] = metrics["bestFitness"] * 0.8
            metrics["diversity"] = max(0.3, 1.0 - (session["currentGeneration"] / 50) * 0.5)

            if session["currentGeneration"] >= 50:
                session["status"] = "completed"
                break

        await asyncio.sleep(1)

if __name__ == "__main__":
    print("=" * 70)
    print("  Agent Forge API Server")
    print("=" * 70)
    print()
    print("Starting server on http://localhost:8000")
    print()
    print("Endpoints:")
    print("  REST API: http://localhost:8000")
    print("  WebSocket: ws://localhost:8000/ws/dashboard")
    print("  Health: http://localhost:8000/health")
    print()
    print("Frontend should connect to http://localhost:3000")
    print("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=8000)