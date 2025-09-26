# Agent Forge API - Quick Start Guide

Get the Agent Forge Pipeline API running in 5 minutes.

## Prerequisites

- Python 3.11+
- pip

## Installation

```bash
# 1. Install dependencies
pip install -r requirements_api.txt

# Or install individually
pip install fastapi uvicorn websockets pydantic psutil
```

## Start the Server

```bash
# Method 1: Using the run script (recommended)
python run_api_server.py

# Method 2: Using uvicorn directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
================================================================================
  Agent Forge Pipeline API Server
================================================================================

Starting server...

API Documentation:
  http://localhost:8000/docs       - Swagger UI
  http://localhost:8000/redoc      - ReDoc

RESTful Endpoints:
  POST   /api/v1/pipeline/start               - Start pipeline
  POST   /api/v1/pipeline/control             - Control execution
  GET    /api/v1/pipeline/status/{id}         - Get status
  ...

WebSocket Endpoints:
  ws://localhost:8000/ws/agents     - Agent updates
  ws://localhost:8000/ws/pipeline   - Pipeline progress
  ...
```

## Your First API Call

### 1. Start a Pipeline

```bash
curl -X POST "http://localhost:8000/api/v1/pipeline/start" \
  -H "Content-Type: application/json" \
  -d '{
    "phases": ["cognate", "evomerge"],
    "enable_monitoring": true,
    "max_agents": 20
  }'
```

Response:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "started"
}
```

### 2. Check Status

```bash
curl "http://localhost:8000/api/v1/pipeline/status/{session_id}"
```

### 3. WebSocket Monitoring (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/pipeline?session_id=...');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Progress:', data.data.progress_percent + '%');
};
```

## Interactive Testing

### Swagger UI
Visit http://localhost:8000/docs to:
- Browse all endpoints
- Try API calls interactively
- See request/response schemas

### WebSocket Test Client
Visit http://localhost:8000/ws/client to:
- Connect to WebSocket channels
- View live updates
- Test with different session IDs

## Quick Examples

### Python Client

```python
import requests

# Start pipeline
response = requests.post('http://localhost:8000/api/v1/pipeline/start', json={
    "phases": ["cognate", "evomerge", "training"],
    "enable_monitoring": True,
    "max_agents": 30
})

session_id = response.json()['session_id']
print(f"Started session: {session_id}")

# Get status
status = requests.get(f'http://localhost:8000/api/v1/pipeline/status/{session_id}')
print(f"Progress: {status.json()['total_progress_percent']}%")

# Pause execution
requests.post('http://localhost:8000/api/v1/pipeline/control', json={
    "action": "pause",
    "session_id": session_id
})
```

### JavaScript/TypeScript Client

```typescript
// Start pipeline
const response = await fetch('http://localhost:8000/api/v1/pipeline/start', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    phases: ['cognate', 'evomerge'],
    enable_monitoring: true,
    max_agents: 20
  })
});

const { session_id } = await response.json();

// WebSocket monitoring
const ws = new WebSocket(`ws://localhost:8000/ws/dashboard?session_id=${session_id}`);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Pipeline:', data.data.pipeline_status);
  console.log('Phase:', data.data.current_phase);
  console.log('Progress:', data.data.progress_percent);
};
```

## Available Presets

```bash
# List presets
curl "http://localhost:8000/api/v1/pipeline/presets"

# Use preset
curl -X POST "http://localhost:8000/api/v1/pipeline/start" \
  -H "Content-Type: application/json" \
  -d @presets/full_pipeline.json
```

Built-in presets:
- `quick_test` - Fast test (cognate + evomerge)
- `full_pipeline` - Complete 8-phase pipeline
- `compression_only` - BitNet + final compression

## Common Operations

### Pause/Resume
```bash
# Pause
curl -X POST "http://localhost:8000/api/v1/pipeline/control" \
  -H "Content-Type: application/json" \
  -d '{"action": "pause", "session_id": "..."}'

# Resume
curl -X POST "http://localhost:8000/api/v1/pipeline/control" \
  -H "Content-Type: application/json" \
  -d '{"action": "resume", "session_id": "..."}'
```

### Save Checkpoint
```bash
curl -X POST "http://localhost:8000/api/v1/pipeline/checkpoint/save" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "...",
    "checkpoint_name": "phase3_complete",
    "include_model_state": true,
    "include_swarm_state": true
  }'
```

### Quality Gates
```bash
curl -X POST "http://localhost:8000/api/v1/pipeline/quality-gates/{session_id}?phase=training"
```

### Execution History
```bash
# Get last 10 runs
curl "http://localhost:8000/api/v1/pipeline/history?limit=10"

# Filter by status
curl "http://localhost:8000/api/v1/pipeline/history?status=completed"
```

## Health Check

```bash
curl "http://localhost:8000/api/v1/pipeline/health"
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "active_sessions": 2,
  "total_memory_mb": 16384,
  "available_memory_mb": 8192
}
```

## WebSocket Channels

All channels support optional `session_id` query parameter:

```javascript
// Agent updates
new WebSocket('ws://localhost:8000/ws/agents?session_id=...');

// Task progress
new WebSocket('ws://localhost:8000/ws/tasks?session_id=...');

// Performance metrics (updates every 1s)
new WebSocket('ws://localhost:8000/ws/metrics?session_id=...');

// Pipeline progress
new WebSocket('ws://localhost:8000/ws/pipeline?session_id=...');

// Combined dashboard (updates every 2s)
new WebSocket('ws://localhost:8000/ws/dashboard?session_id=...');
```

## Troubleshooting

### Port already in use
```bash
# Change port
uvicorn src.api.main:app --port 8001
```

### Import errors
```bash
# Ensure you're in project root
cd /path/to/agent-forge

# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### WebSocket connection fails
- Ensure server is running
- Check firewall settings
- Use `ws://` not `wss://` for localhost

### Missing dependencies
```bash
pip install -r requirements_api.txt --upgrade
```

## Next Steps

1. **Explore API Docs**: http://localhost:8000/docs
2. **Read Full Documentation**: `docs/API_DOCUMENTATION.md`
3. **Developer Guide**: `src/api/README.md`
4. **Integration Examples**: `docs/API_DOCUMENTATION.md#examples`

## Production Deployment

### Using Gunicorn

```bash
gunicorn src.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile -
```

### Using Docker

```bash
docker build -t agent-forge-api .
docker run -p 8000:8000 agent-forge-api
```

### Environment Variables

```bash
export API_HOST=0.0.0.0
export API_PORT=8000
export LOG_LEVEL=info
export CORS_ORIGINS='["https://your-frontend.com"]'
```

## Support

- **API Documentation**: http://localhost:8000/docs
- **Developer Guide**: `src/api/README.md`
- **Full API Reference**: `docs/API_DOCUMENTATION.md`
- **Backend Summary**: `docs/BACKEND_API_SUMMARY.md`

---

Happy coding! 🚀