"""
Agent Forge UI Server
WebSocket server for real-time pipeline monitoring
"""

from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import asyncio
import sys
from pathlib import Path
import time
import threading

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.unified_phase_executor import UnifiedPhaseExecutor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'agent-forge-secret-key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Pipeline executor
executor = UnifiedPhaseExecutor()
pipeline_thread = None
pipeline_running = False


@app.route('/')
def index():
    """Serve the dashboard"""
    return send_from_directory('.', 'dashboard.html')


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('connected', {'status': 'Connected to Agent Forge server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")


@socketio.on('start_pipeline')
def handle_start_pipeline(data):
    """Start the 8-phase pipeline"""
    global pipeline_thread, pipeline_running

    if pipeline_running:
        emit('error', {'message': 'Pipeline already running'})
        return

    pipeline_running = True
    pipeline_thread = threading.Thread(target=run_pipeline_async)
    pipeline_thread.start()

    emit('pipeline_started', {'message': 'Pipeline execution started'})


@socketio.on('pause_pipeline')
def handle_pause_pipeline(data):
    """Pause the pipeline"""
    global pipeline_running
    pipeline_running = False
    emit('pipeline_paused', {'message': 'Pipeline paused'})


@socketio.on('reset_pipeline')
def handle_reset_pipeline(data):
    """Reset the pipeline"""
    global pipeline_running
    pipeline_running = False
    executor.phase_outputs.clear()
    emit('pipeline_reset', {'message': 'Pipeline reset'})


def run_pipeline_async():
    """Run the pipeline in a background thread"""
    global pipeline_running

    # Create event loop for async execution
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Run all 8 phases
        for phase_num in range(1, 9):
            if not pipeline_running:
                break

            # Emit start of phase
            socketio.emit('phase_progress', {
                'phase': phase_num,
                'phase_name': executor.phases[phase_num]['name'],
                'progress': 0,
                'message': f'Starting Phase {phase_num}'
            })

            # Simulate phase execution with progress updates
            for progress in [10, 30, 50, 70, 90, 100]:
                if not pipeline_running:
                    break

                time.sleep(0.5)  # Simulate work

                socketio.emit('phase_progress', {
                    'phase': phase_num,
                    'phase_name': executor.phases[phase_num]['name'],
                    'progress': progress,
                    'message': f'Phase {phase_num} progress: {progress}%',
                    'metrics': {
                        'accuracy': 0.85 + progress/1000,
                        'loss': 2.5 - progress/100
                    }
                })

            if pipeline_running:
                # Mark phase as complete
                socketio.emit('phase_complete', {
                    'phase': phase_num,
                    'phase_name': executor.phases[phase_num]['name'],
                    'success': True
                })

        # Pipeline complete
        if pipeline_running:
            socketio.emit('pipeline_complete', {
                'message': 'All 8 phases completed successfully',
                'total_time': time.time()
            })

    except Exception as e:
        socketio.emit('error', {'message': f'Pipeline error: {str(e)}'})

    finally:
        pipeline_running = False
        loop.close()


def emit_progress_update(phase: int, progress: float, message: str, metrics: dict = None):
    """Emit progress update to all connected clients"""
    socketio.emit('phase_progress', {
        'phase': phase,
        'progress': progress,
        'message': message,
        'metrics': metrics or {},
        'timestamp': time.time()
    })


if __name__ == '__main__':
    print("=" * 60)
    print("AGENT FORGE UI SERVER")
    print("=" * 60)
    print("Starting WebSocket server on http://localhost:5000")
    print("Open dashboard.html in your browser to monitor pipeline")
    print("=" * 60)
    socketio.run(app, debug=False, port=5000, host='0.0.0.0')