"""
Agent Forge FastAPI Application

Main FastAPI application with RESTful and WebSocket endpoints.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .routes import pipeline_routes, websocket_routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("Starting Agent Forge API")
    yield
    # Shutdown
    logger.info("Shutting down Agent Forge API")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title="Agent Forge API",
        description="RESTful and WebSocket API for Agent Forge 8-Phase Pipeline",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(pipeline_routes.router)
    app.include_router(websocket_routes.router)

    @app.get("/")
    async def root():
        """Redirect to API documentation."""
        return RedirectResponse(url="/docs")

    @app.get("/api/v1/info")
    async def api_info():
        """Get API information."""
        return {
            "name": "Agent Forge API",
            "version": "1.0.0",
            "description": "8-Phase Pipeline Orchestration",
            "features": [
                "Pipeline Control (start/stop/pause/resume)",
                "Phase Configuration",
                "Real-time WebSocket Streaming",
                "Quality Gate Validation",
                "Checkpoint Management",
                "Execution History",
                "Configuration Presets",
            ],
            "endpoints": {
                "rest": "/api/v1/pipeline/*",
                "websocket": "/ws/*",
                "docs": "/docs",
                "health": "/api/v1/pipeline/health",
            },
        }

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )