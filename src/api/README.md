# Agent Forge Pipeline API

Production-ready FastAPI backend for the Agent Forge 8-Phase Pipeline orchestration system.

## Features

- **RESTful API**: Complete pipeline control with type-safe Pydantic models
- **WebSocket Streaming**: Real-time updates for agents, tasks, and metrics
- **Swarm Coordination**: Integration with SwarmCoordinator and UnifiedPipeline
- **Quality Gates**: Automated validation with theater detection
- **State Management**: Session tracking, checkpoints, and configuration presets
- **Error Handling**: Comprehensive error responses with proper HTTP status codes

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements_api.txt
```

### Start Server

```bash
# Using the run script
python run_api_server.py

# Or using uvicorn directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Access Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- WebSocket Test Client: http://localhost:8000/ws/client

## Architecture

```
src/api/
├── main.py                 # FastAPI application entry point
├── models/
│   ├── __init__.py
│   └── pipeline_models.py  # Pydantic models for requests/responses
├── routes/
│   ├── __init__.py
│   ├── pipeline_routes.py  # RESTful endpoints
│   └── websocket_routes.py # WebSocket endpoints
├── services/
│   ├── __init__.py
│   └── pipeline_service.py # Business logic layer
└── websocket/
    ├── __init__.py
    └── connection_manager.py # WebSocket connection management
```

## API Endpoints

### Pipeline Control

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/pipeline/start` | Start pipeline execution |
| POST | `/api/v1/pipeline/control` | Control execution (pause/resume/stop) |
| GET | `/api/v1/pipeline/status/{id}` | Get pipeline status |
| GET | `/api/v1/pipeline/swarm/{id}` | Get swarm coordination status |

### Quality & Validation

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/pipeline/quality-gates/{id}` | Validate quality gates |

### State Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/pipeline/checkpoint/save` | Save checkpoint |
| POST | `/api/v1/pipeline/checkpoint/load/{id}` | Load checkpoint |
| GET | `/api/v1/pipeline/presets` | List configuration presets |
| GET | `/api/v1/pipeline/preset/{name}` | Get preset |
| POST | `/api/v1/pipeline/preset/save` | Save preset |

### History & Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/pipeline/history` | Get execution history |
| GET | `/api/v1/pipeline/health` | Health check |

## WebSocket Channels

All WebSocket endpoints support optional `session_id` query parameter.

| Endpoint | Description |
|----------|-------------|
| `/ws/agents` | Agent status updates |
| `/ws/tasks` | Task execution progress |
| `/ws/metrics` | Performance metrics stream |
| `/ws/pipeline` | Pipeline progress updates |
| `/ws/dashboard` | Combined dashboard data |

### WebSocket Stats

| Endpoint | Description |
|----------|-------------|
| `/ws/stats` | Connection statistics |
| `/ws/client` | Interactive test client |

## Usage Examples

### Start Pipeline

```python
import requests

response = requests.post('http://localhost:8000/api/v1/pipeline/start', json={
    "phases": ["cognate", "evomerge", "training"],
    "config": {
        "cognate": {
            "base_models": ["model1", "model2"]
        }
    },
    "enable_monitoring": True,
    "swarm_topology": "hierarchical",
    "max_agents": 50
})

session_id = response.json()['session_id']
```

### Monitor Progress

```python
# REST API
status = requests.get(f'http://localhost:8000/api/v1/pipeline/status/{session_id}')
print(f"Progress: {status.json()['total_progress_percent']}%")

# WebSocket
import asyncio
import websockets

async def monitor():
    uri = f"ws://localhost:8000/ws/pipeline?session_id={session_id}"
    async with websockets.connect(uri) as ws:
        async for message in ws:
            print(message)

asyncio.run(monitor())
```

### Control Execution

```python
# Pause
requests.post('http://localhost:8000/api/v1/pipeline/control', json={
    "action": "pause",
    "session_id": session_id
})

# Resume
requests.post('http://localhost:8000/api/v1/pipeline/control', json={
    "action": "resume",
    "session_id": session_id
})

# Stop
requests.post('http://localhost:8000/api/v1/pipeline/control', json={
    "action": "stop",
    "session_id": session_id
})
```

### Validate Quality Gates

```python
gates = requests.post(
    f'http://localhost:8000/api/v1/pipeline/quality-gates/{session_id}',
    params={"phase": "training"}
)

result = gates.json()
if result['all_gates_passed']:
    print("All quality gates passed!")
    if result['theater_detection']['theater_detected']:
        print("Warning: Performance theater detected")
```

## Data Models

### Request Models

- `PipelineStartRequest`: Start pipeline with configuration
- `PipelineControlRequest`: Control pipeline execution
- `PhaseConfigRequest`: Configure specific phase
- `CheckpointRequest`: Save/load checkpoint
- `PresetRequest`: Manage configuration presets

### Response Models

- `PipelineStatusResponse`: Current pipeline status
- `SwarmStatusResponse`: Swarm coordination status
- `QualityGateResponse`: Quality gate validation results
- `CheckpointResponse`: Checkpoint information
- `ExecutionHistoryResponse`: Historical execution data

### WebSocket Events

- `PipelineProgressEvent`: Progress updates
- `AgentUpdateEvent`: Agent state changes
- `PhaseCompletionEvent`: Phase completion
- `MetricsStreamEvent`: Real-time metrics
- `ErrorEvent`: Error notifications

## Integration with SwarmCoordinator

The API service layer integrates seamlessly with the existing SwarmCoordinator:

```python
# In pipeline_service.py
swarm_config = SwarmConfig(
    topology=SwarmTopologyEnum(topology.value),
    max_agents=max_agents,
)

session.coordinator = SwarmCoordinator(swarm_config)
await session.coordinator.initialize_swarm()

session.execution_manager = SwarmExecutionManager(session.coordinator)

if enable_monitoring:
    session.monitor = create_swarm_monitor(session.coordinator)
    await session.monitor.start_monitoring()
```

## WebSocket Connection Management

The ConnectionManager handles:

- Multi-channel subscriptions
- Session-specific broadcasts
- Automatic disconnection cleanup
- Connection statistics

```python
# Broadcasting events
from .routes.websocket_routes import broadcast_pipeline_event
from .models.pipeline_models import PipelineProgressEvent

event = PipelineProgressEvent(
    session_id=session_id,
    phase=PipelinePhase.TRAINING,
    progress_percent=75.5,
    metrics={"loss": 0.125}
)

await broadcast_pipeline_event(event, channels=["pipeline", "dashboard"])
```

## Error Handling

All endpoints return standardized error responses:

```json
{
  "error": "Session not found",
  "error_code": "SESSION_NOT_FOUND",
  "details": {"session_id": "..."},
  "timestamp": "2024-01-15T10:30:00Z"
}
```

HTTP Status Codes:
- `200`: Success
- `400`: Bad Request
- `404`: Not Found
- `500`: Internal Server Error

## Production Deployment

### Environment Configuration

```bash
# .env file
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
LOG_LEVEL=info
CORS_ORIGINS=["https://your-frontend.com"]
```

### Running with Gunicorn

```bash
gunicorn src.api.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install -r requirements_api.txt

COPY . .

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/api/
```

Example test:

```python
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_start_pipeline():
    response = client.post("/api/v1/pipeline/start", json={
        "phases": ["cognate", "evomerge"],
        "max_agents": 20
    })
    assert response.status_code == 200
    assert "session_id" in response.json()
```

## Development

### Adding New Endpoints

1. Define Pydantic models in `models/pipeline_models.py`
2. Add route in `routes/pipeline_routes.py`
3. Implement logic in `services/pipeline_service.py`
4. Update documentation

### Adding WebSocket Channels

1. Create endpoint in `routes/websocket_routes.py`
2. Use ConnectionManager for broadcasting
3. Define event models in `models/pipeline_models.py`

## Performance Considerations

- **Concurrent Sessions**: Supports multiple simultaneous pipeline executions
- **WebSocket Scaling**: Connection manager handles thousands of connections
- **State Management**: In-memory state with optional Redis backend
- **Background Tasks**: Async execution doesn't block API responses

## Security

- Input validation with Pydantic
- CORS middleware configuration
- Optional JWT authentication (implement in middleware)
- Rate limiting (recommended: slowapi)

## Monitoring

- Built-in health check endpoint
- WebSocket statistics
- Execution history tracking
- Integration with logging frameworks

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit pull request

## License

[Your License Here]

## Support

For questions and issues:
- GitHub Issues: [link]
- Documentation: [link]
- API Reference: http://localhost:8000/docs