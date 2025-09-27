#!/usr/bin/env python3
"""
Start all servers for the Agent Forge Cognate System
with ACT Titans Architecture (3x 25M parameter models)
"""

import subprocess
import time
import sys
import os
import webbrowser
from pathlib import Path

def start_server(name, command, cwd=None):
    """Start a server in a separate process."""
    print(f"Starting {name}...")
    try:
        if sys.platform == "win32":
            # Windows: Use CREATE_NEW_CONSOLE flag
            subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            # Unix/Mac: Use gnome-terminal, xterm, or Terminal.app
            if sys.platform == "darwin":
                # macOS
                apple_script = f'''
                tell application "Terminal"
                    do script "cd {cwd or os.getcwd()} && {command}"
                end tell
                '''
                subprocess.Popen(["osascript", "-e", apple_script])
            else:
                # Linux
                subprocess.Popen(
                    f"gnome-terminal -- bash -c 'cd {cwd or os.getcwd()} && {command}; exec bash'",
                    shell=True
                )
        print(f"✓ {name} started")
    except Exception as e:
        print(f"✗ Failed to start {name}: {e}")
        return False
    return True

def main():
    """Start all servers for the Cognate system."""
    print("=" * 60)
    print("Starting Agent Forge Cognate System with ACT Titans")
    print("3x 25M Parameter Models with Real-time WebSocket Updates")
    print("=" * 60)
    print()

    project_root = Path(__file__).parent
    dashboard_path = project_root / "src" / "web" / "dashboard"

    # Start WebSocket server
    if not start_server(
        "WebSocket Server (Port 8085)",
        "python agent_forge/api/websocket_server.py",
        cwd=project_root
    ):
        print("Failed to start WebSocket server. Exiting.")
        return

    time.sleep(2)

    # Start Python Bridge API
    if not start_server(
        "Python Bridge API (Port 8001)",
        "python agent_forge/api/python_bridge_server.py",
        cwd=project_root
    ):
        print("Failed to start Python Bridge API. Exiting.")
        return

    time.sleep(2)

    # Start Dashboard
    if not start_server(
        "Dashboard (Port 3000)",
        "npm run dev",
        cwd=dashboard_path
    ):
        print("Failed to start Dashboard. Exiting.")
        return

    time.sleep(5)

    print()
    print("=" * 60)
    print("All servers started successfully!")
    print()
    print("Services:")
    print("- WebSocket Server:  http://localhost:8085")
    print("- Python Bridge API: http://localhost:8001")
    print("- Dashboard:         http://localhost:3000/phases/cognate")
    print()
    print("To test the system:")
    print("1. Open http://localhost:3000/phases/cognate")
    print("2. Click 'Start Pretraining'")
    print("3. Watch the 3 orbs ripple and change color")
    print("4. Progress bars update in real-time")
    print("5. Each model trains to 25M parameters")
    print()
    print("Opening dashboard in browser...")

    # Open browser
    time.sleep(2)
    webbrowser.open("http://localhost:3000/phases/cognate")

    print()
    print("Press Ctrl+C to stop all servers")

    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        # In a production environment, you'd properly terminate the subprocesses here

if __name__ == "__main__":
    main()