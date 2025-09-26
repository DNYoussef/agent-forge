#!/usr/bin/env python
"""
Quick start script for Agent Forge API with WebSocket support.

This script starts the enhanced API with real-time WebSocket endpoints.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Start the WebSocket-enabled API server."""
    try:
        import uvicorn
        from src.api.api_enhanced import create_enhanced_app

        print("=" * 70)
        print("  Agent Forge API with WebSocket Support")
        print("=" * 70)
        print()
        print("Starting server...")
        print()
        print("📚 API Documentation:")
        print("   http://localhost:8000/docs")
        print("   http://localhost:8000/redoc")
        print()
        print("🔌 WebSocket Endpoints:")
        print("   ws://localhost:8000/ws/agents     - Agent status updates")
        print("   ws://localhost:8000/ws/tasks      - Task execution progress")
        print("   ws://localhost:8000/ws/knowledge  - Knowledge graph changes")
        print("   ws://localhost:8000/ws/metrics    - Performance metrics stream")
        print("   ws://localhost:8000/ws/dashboard  - Combined dashboard data")
        print()
        print("🧪 Test Client:")
        print("   http://localhost:8000/ws/client")
        print()
        print("📊 WebSocket Stats:")
        print("   http://localhost:8000/ws/stats")
        print()
        print("=" * 70)
        print()

        app = create_enhanced_app()
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print()
        print("Please install required dependencies:")
        print("  pip install fastapi uvicorn websockets")
        sys.exit(1)

    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()