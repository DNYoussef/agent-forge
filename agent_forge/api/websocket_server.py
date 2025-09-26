"""
WebSocket Server for Agent Forge Real-time Updates
Provides real-time progress tracking for all phases, especially Cognate training.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Set, Any
import asyncio
import json
import logging
from datetime import datetime
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agent Forge WebSocket Server",
    description="Real-time updates for training phases",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    """Manages WebSocket connections and broadcasts."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.channel_subscriptions: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove disconnected WebSocket."""
        self.active_connections.remove(websocket)
        # Remove from all channel subscriptions
        for channel in self.channel_subscriptions:
            if websocket in self.channel_subscriptions[channel]:
                self.channel_subscriptions[channel].remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def subscribe(self, websocket: WebSocket, channel: str):
        """Subscribe a WebSocket to a specific channel."""
        if channel not in self.channel_subscriptions:
            self.channel_subscriptions[channel] = set()
        self.channel_subscriptions[channel].add(websocket)
        logger.info(f"WebSocket subscribed to channel: {channel}")

    async def unsubscribe(self, websocket: WebSocket, channel: str):
        """Unsubscribe a WebSocket from a channel."""
        if channel in self.channel_subscriptions:
            self.channel_subscriptions[channel].discard(websocket)
            logger.info(f"WebSocket unsubscribed from channel: {channel}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket."""
        await websocket.send_text(message)

    async def broadcast(self, message: str, channel: str = None):
        """Broadcast message to all connections or specific channel."""
        if channel and channel in self.channel_subscriptions:
            # Send to channel subscribers
            for connection in self.channel_subscriptions[channel]:
                try:
                    await connection.send_text(message)
                except:
                    # Connection might be closed
                    pass
        else:
            # Send to all connections
            for connection in self.active_connections:
                try:
                    await connection.send_text(message)
                except:
                    # Connection might be closed
                    pass


manager = ConnectionManager()


# Store training progress for each phase
training_progress = {
    "cognate": {
        "status": "idle",
        "models": {
            "model-1": {"progress": 0, "loss": 0, "status": "pending"},
            "model-2": {"progress": 0, "loss": 0, "status": "pending"},
            "model-3": {"progress": 0, "loss": 0, "status": "pending"}
        },
        "overall_progress": 0,
        "start_time": None,
        "eta": None
    }
}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)

    # Send initial connection message
    await websocket.send_text(json.dumps({
        "type": "connection",
        "status": "connected",
        "timestamp": datetime.now().isoformat()
    }))

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle different message types
            if message["type"] == "subscribe":
                channel = message.get("channel", "general")
                await manager.subscribe(websocket, channel)
                await websocket.send_text(json.dumps({
                    "type": "subscribed",
                    "channel": channel
                }))

            elif message["type"] == "unsubscribe":
                channel = message.get("channel", "general")
                await manager.unsubscribe(websocket, channel)
                await websocket.send_text(json.dumps({
                    "type": "unsubscribed",
                    "channel": channel
                }))

            elif message["type"] == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))

            elif message["type"] == "get_status":
                phase = message.get("phase", "cognate")
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "phase": phase,
                    "data": training_progress.get(phase, {})
                }))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")


@app.post("/broadcast/{channel}")
async def broadcast_message(channel: str, message: dict):
    """
    HTTP endpoint to broadcast messages to WebSocket clients.
    Used by training processes to send progress updates.
    """
    try:
        # Format message with timestamp
        broadcast_data = {
            "type": "broadcast",
            "channel": channel,
            "timestamp": datetime.now().isoformat(),
            **message
        }

        # Special handling for Cognate phase updates
        if channel == "cognate" and message.get("type") == "progress":
            # Update stored progress
            model_id = message.get("model_id")
            if model_id in training_progress["cognate"]["models"]:
                training_progress["cognate"]["models"][model_id].update({
                    "progress": message.get("progress", 0),
                    "loss": message.get("loss", 0),
                    "status": message.get("status", "training")
                })

                # Calculate overall progress
                total_progress = sum(
                    m["progress"] for m in training_progress["cognate"]["models"].values()
                )
                training_progress["cognate"]["overall_progress"] = total_progress / 3

                # Add overall progress to broadcast
                broadcast_data["overall_progress"] = training_progress["cognate"]["overall_progress"]

        # Send to WebSocket clients
        await manager.broadcast(json.dumps(broadcast_data), channel)

        return {"status": "broadcast sent", "channel": channel}

    except Exception as e:
        logger.error(f"Broadcast error: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/cognate/start")
async def start_cognate_training():
    """Endpoint to signal Cognate training start."""
    training_progress["cognate"]["status"] = "training"
    training_progress["cognate"]["start_time"] = datetime.now().isoformat()

    # Reset model progress
    for model_id in training_progress["cognate"]["models"]:
        training_progress["cognate"]["models"][model_id] = {
            "progress": 0,
            "loss": 0,
            "status": "pending"
        }

    # Broadcast start event
    await manager.broadcast(json.dumps({
        "type": "cognate:start",
        "timestamp": datetime.now().isoformat(),
        "models": ["model-1", "model-2", "model-3"]
    }), "cognate")

    return {"status": "training started"}


@app.post("/cognate/update")
async def update_cognate_progress(update: dict):
    """
    Update Cognate training progress.

    Expected format:
    {
        "model_id": "model-1",
        "step": 100,
        "total_steps": 1000,
        "loss": 0.45,
        "perplexity": 12.3,
        "status": "training"
    }
    """
    model_id = update.get("model_id")
    step = update.get("step", 0)
    total_steps = update.get("total_steps", 1)
    progress = (step / total_steps) * 100 if total_steps > 0 else 0

    # Update progress
    if model_id in training_progress["cognate"]["models"]:
        training_progress["cognate"]["models"][model_id].update({
            "progress": progress,
            "loss": update.get("loss", 0),
            "status": update.get("status", "training"),
            "step": step,
            "total_steps": total_steps
        })

    # Calculate overall progress
    total_progress = sum(
        m["progress"] for m in training_progress["cognate"]["models"].values()
    )
    training_progress["cognate"]["overall_progress"] = total_progress / 3

    # Broadcast update
    await manager.broadcast(json.dumps({
        "type": "cognate:progress",
        "model_id": model_id,
        "progress": progress,
        "overall_progress": training_progress["cognate"]["overall_progress"],
        "loss": update.get("loss", 0),
        "perplexity": update.get("perplexity", 0),
        "step": step,
        "total_steps": total_steps,
        "timestamp": datetime.now().isoformat()
    }), "cognate")

    # Check if model completed
    if progress >= 100:
        await manager.broadcast(json.dumps({
            "type": "cognate:model_complete",
            "model_id": model_id,
            "timestamp": datetime.now().isoformat()
        }), "cognate")

        # Check if all models completed
        all_complete = all(
            m["progress"] >= 100
            for m in training_progress["cognate"]["models"].values()
        )
        if all_complete:
            training_progress["cognate"]["status"] = "completed"
            await manager.broadcast(json.dumps({
                "type": "cognate:complete",
                "timestamp": datetime.now().isoformat()
            }), "cognate")

    return {"status": "updated", "overall_progress": training_progress["cognate"]["overall_progress"]}


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Agent Forge WebSocket Server",
        "status": "running",
        "active_connections": len(manager.active_connections),
        "channels": list(manager.channel_subscriptions.keys()),
        "training_status": {
            phase: data["status"]
            for phase, data in training_progress.items()
        }
    }


@app.get("/cognate/status")
async def get_cognate_status():
    """Get current Cognate training status."""
    return training_progress["cognate"]


def run_server(host: str = "127.0.0.1", port: int = 8085):
    """Run the WebSocket server."""
    logger.info(f"Starting WebSocket server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent Forge WebSocket Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8085, help="Port to bind to")

    args = parser.parse_args()
    run_server(host=args.host, port=args.port)