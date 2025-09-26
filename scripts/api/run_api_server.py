#!/usr/bin/env python
"""
Agent Forge API Server

Production-ready FastAPI server with WebSocket support for pipeline orchestration.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Start the API server."""
    try:
        import uvicorn
        from src.api.main import app

        print("=" * 80)
        print("  Agent Forge Pipeline API Server")
        print("=" * 80)
        print()
        print("Starting server...")
        print()
        print("API Documentation:")
        print("  http://localhost:8000/docs       - Swagger UI")
        print("  http://localhost:8000/redoc      - ReDoc")
        print()
        print("RESTful Endpoints:")
        print("  POST   /api/v1/pipeline/start               - Start pipeline")
        print("  POST   /api/v1/pipeline/control             - Control execution")
        print("  GET    /api/v1/pipeline/status/{id}         - Get status")
        print("  GET    /api/v1/pipeline/swarm/{id}          - Swarm status")
        print("  POST   /api/v1/pipeline/quality-gates/{id}  - Validate gates")
        print("  GET    /api/v1/pipeline/history             - Execution history")
        print("  GET    /api/v1/pipeline/health              - Health check")
        print()
        print("WebSocket Endpoints:")
        print("  ws://localhost:8000/ws/agents     - Agent updates")
        print("  ws://localhost:8000/ws/tasks      - Task progress")
        print("  ws://localhost:8000/ws/metrics    - Performance metrics")
        print("  ws://localhost:8000/ws/pipeline   - Pipeline progress")
        print("  ws://localhost:8000/ws/dashboard  - Combined dashboard")
        print()
        print("WebSocket Test Client:")
        print("  http://localhost:8000/ws/client")
        print()
        print("WebSocket Statistics:")
        print("  http://localhost:8000/ws/stats")
        print()
        print("Configuration Presets:")
        print("  GET    /api/v1/pipeline/presets            - List presets")
        print("  GET    /api/v1/pipeline/preset/{name}      - Get preset")
        print("  POST   /api/v1/pipeline/preset/save        - Save preset")
        print()
        print("Checkpoint Management:")
        print("  POST   /api/v1/pipeline/checkpoint/save    - Save checkpoint")
        print("  POST   /api/v1/pipeline/checkpoint/load/{id} - Load checkpoint")
        print()
        print("=" * 80)
        print()

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True,
        )

    except ImportError as e:
        print(f"ERROR: Missing dependencies: {e}")
        print()
        print("Please install required packages:")
        print("  pip install -r requirements_api.txt")
        print()
        print("Or install individually:")
        print("  pip install fastapi uvicorn websockets pydantic psutil")
        sys.exit(1)

    except Exception as e:
        print(f"ERROR: Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()