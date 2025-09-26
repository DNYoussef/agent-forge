# Agent Forge API Integration Guide

Complete guide for replacing simulation layers with real Python backend calls while maintaining exact UI compatibility.

## 🎯 Overview

This system provides seamless integration between Next.js frontend APIs and Python backend services with automatic fallback to simulation when the backend is unavailable.

### Key Features

- **Exact Compatibility**: Maintains identical response formats for UI compatibility
- **Automatic Fallback**: Graceful degradation to simulation when backend fails
- **Session Management**: Unified session handling between frontend and backend
- **Comprehensive Testing**: Full compatibility validation and performance testing
- **Error Recovery**: Robust error handling with meaningful user feedback
- **Real-time Metrics**: Live progress tracking with realistic data

## 📁 Project Structure

```
src/api/
├── python-bridge-server.py          # Main Python backend server
├── types/
│   └── phase-interfaces.ts           # TypeScript interfaces
├── utils/
│   ├── api-client.ts                 # HTTP client utilities
│   └── session-manager.ts            # Session management
├── simulation/
│   ├── cognate-simulation.ts         # Cognate fallback simulation
│   └── evomerge-simulation.ts        # EvoMerge fallback simulation
└── testing/
    ├── compatibility-validator.ts     # Compatibility tests
    └── test-runner.ts                # Comprehensive test runner

src/web/dashboard/app/api/phases/
├── cognate/
│   └── route.ts                      # Cognate phase API route
└── evomerge/
    └── route.ts                      # EvoMerge phase API route

scripts/
├── setup-api-integration.sh          # Setup script
├── start-api-integration.sh          # Start services
├── stop-api-integration.sh           # Stop services
└── test-api-integration.sh           # Run tests
```

## 🚀 Quick Start

### 1. Setup

```bash
# Run the setup script
./scripts/setup-api-integration.sh

# Or manually install dependencies
npm install uuid @types/uuid
pip3 install fastapi uvicorn python-multipart
```

### 2. Start the System

```bash
# Start Python bridge and services
./scripts/start-api-integration.sh

# Or start manually
python3 src/api/python-bridge-server.py &
```

### 3. Test Integration

```bash
# Run comprehensive tests
./scripts/test-api-integration.sh

# Or run specific tests
npx tsx src/api/testing/test-runner.ts --report
```

### 4. Stop the System

```bash
# Stop all services
./scripts/stop-api-integration.sh
```

## 🔧 Configuration

### Environment Variables

Create `.env.api-integration`:

```bash
PYTHON_BRIDGE_PORT=8001
PYTHON_BRIDGE_HOST=127.0.0.1
API_TIMEOUT=5000
API_RETRY_ATTEMPTS=3
API_RETRY_DELAY=1000
ENABLE_FALLBACK=true
FALLBACK_DELAY=500
DEBUG_MODE=false
```

### API Client Configuration

```typescript
import { ApiClient } from './src/api/utils/api-client';

const apiClient = new ApiClient({
  baseUrl: 'http://localhost:8001',
  timeout: 5000,
  retryAttempts: 3,
  retryDelay: 1000,
  enableFallback: true,
  fallbackDelay: 500,
});
```

## 📡 API Endpoints

### Python Bridge Server (Port 8001)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/cognate/start` | Start Cognate phase |
| GET | `/api/cognate/status/{sessionId}` | Get Cognate status |
| POST | `/api/cognate/stop/{sessionId}` | Stop Cognate phase |
| GET | `/api/evomerge/metrics/{sessionId}` | Get EvoMerge metrics |
| GET | `/api/sessions` | List all sessions |
| DELETE | `/api/sessions/{sessionId}` | Delete session |

### Next.js API Routes (Port 3000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/phases/cognate` | Start Cognate with fallback |
| GET | `/api/phases/cognate?sessionId=xxx` | Get Cognate status |
| DELETE | `/api/phases/cognate?sessionId=xxx` | Stop Cognate phase |
| GET | `/api/phases/evomerge?sessionId=xxx` | Get EvoMerge metrics |
| POST | `/api/phases/evomerge` | Initialize EvoMerge |
| PUT | `/api/phases/evomerge?sessionId=xxx` | Reset EvoMerge |
| DELETE | `/api/phases/evomerge?sessionId=xxx` | Stop EvoMerge |

## 💻 Usage Examples

### Starting a Cognate Phase

```typescript
import { apiUtils } from './src/api/utils/api-client';
import { startCognatePhase } from './src/api/simulation/cognate-simulation';

const config = {
  sessionId: 'my-session-123',
  maxIterations: 10,
  convergenceThreshold: 0.95,
  parallelAgents: 3,
};

// This will try Python backend first, fallback to simulation
const result = await apiUtils.startCognatePhase(config, async () => {
  return await startCognatePhase(config.sessionId, config);
});

console.log('Phase started:', result);
```

### Getting Real-time Status

```typescript
// Get status with automatic fallback
const status = await apiUtils.getCognateStatus('my-session-123', async () => {
  return await getCognateStatus('my-session-123');
});

console.log('Current status:', status);
console.log('Convergence score:', status.metrics?.convergenceScore);
```

### Frontend Integration

```typescript
// In your Next.js component
const startPhase = async () => {
  try {
    const response = await fetch('/api/phases/cognate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sessionId: 'frontend-session-123',
        maxIterations: 15,
      }),
    });

    const result = await response.json();
    if (result.success) {
      console.log('Phase started successfully');
      // Start polling for status updates
      pollForStatus(result.sessionId);
    }
  } catch (error) {
    console.error('Failed to start phase:', error);
  }
};
```

## 🔄 Fallback Mechanism

The system provides seamless fallback from real backend to simulation:

### Automatic Fallback Flow

1. **Try Real Backend**: Attempt HTTP request to Python bridge
2. **Network Error Detection**: Catch connection failures, timeouts
3. **Fallback Trigger**: Automatically switch to simulation
4. **Transparent Response**: Return simulation data with same format
5. **User Notification**: Log fallback for debugging (optional UI notification)

### Fallback Configuration

```typescript
// Configure fallback behavior
const fallbackOptions = {
  enableFallback: true,        // Enable automatic fallback
  fallbackDelay: 500,          // Delay before fallback (ms)
  maxRetries: 3,               // Max backend retry attempts
  simulationConfig: {
    enableLogging: true,       // Log simulation activities
    simulateLatency: true,     // Add realistic delays
    errorRate: 0.05,          // 5% simulated error rate
    maxLatency: 2000,         // Max simulated delay (ms)
  }
};
```

### Manual Fallback Control

```typescript
// Force fallback to simulation
const result = await apiClient.request('/api/cognate/start', {
  method: 'POST',
  body: JSON.stringify(config),
}, {
  enableFallback: false  // Disable automatic fallback
});

// Or handle fallback manually
try {
  const result = await apiClient.post('/api/cognate/start', config);
} catch (error) {
  if (error instanceof NetworkError) {
    console.log('Backend unavailable, using simulation');
    const fallbackResult = await startCognatePhase(sessionId, config);
    return fallbackResult;
  }
  throw error;
}
```

## 🧪 Testing

### Compatibility Testing

```bash
# Run full compatibility test suite
npx tsx src/api/testing/test-runner.ts

# Generate detailed report
npx tsx src/api/testing/test-runner.ts --report
```

### Manual Testing

```bash
# Test Python bridge directly
curl http://localhost:8001/health

# Test Cognate phase start
curl -X POST http://localhost:8001/api/cognate/start \
  -H "Content-Type: application/json" \
  -d '{"sessionId":"test-123","maxIterations":5}'

# Test Next.js API with fallback
curl -X POST http://localhost:3000/api/phases/cognate \
  -H "Content-Type: application/json" \
  -d '{"sessionId":"test-456","maxIterations":5}'
```

### Performance Testing

```typescript
// Test concurrent requests
const sessionIds = Array.from({length: 10}, (_, i) => `perf-test-${i}`);
const promises = sessionIds.map(sessionId =>
  fetch('/api/phases/cognate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionId }),
  })
);

const results = await Promise.all(promises);
console.log('All requests completed:', results.length);
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Python Bridge Not Starting

```bash
# Check if port is available
lsof -i :8001

# Check Python dependencies
pip3 list | grep fastapi

# Start with debug mode
DEBUG_MODE=true python3 src/api/python-bridge-server.py
```

#### 2. Connection Refused Errors

```bash
# Verify bridge is running
curl http://localhost:8001/health

# Check firewall settings
sudo ufw status

# Test with different host/port
PYTHON_BRIDGE_HOST=0.0.0.0 python3 src/api/python-bridge-server.py
```

#### 3. TypeScript Compilation Errors

```bash
# Install missing types
npm install @types/node @types/uuid

# Check TypeScript configuration
npx tsc --noEmit --skipLibCheck src/api/**/*.ts
```

#### 4. Fallback Not Working

```typescript
// Verify fallback is enabled
const config = apiClient.getConfig();
console.log('Fallback enabled:', config.enableFallback);

// Test fallback manually
apiClient.updateConfig({ baseUrl: 'http://localhost:9999' }); // Force failure
const result = await apiUtils.getCognateStatus('test', fallbackFunction);
```

### Debug Mode

Enable detailed logging:

```bash
# Environment variable
export DEBUG_MODE=true

# Or in code
apiClient.updateConfig({
  enableFallback: true,
  fallbackDelay: 0  // Immediate fallback for testing
});
```

## 🚀 Production Deployment

### Docker Setup

```dockerfile
# Python Bridge Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/api/python-bridge-server.py .
EXPOSE 8001

CMD ["python", "python-bridge-server.py"]
```

### Systemd Service

```ini
[Unit]
Description=Agent Forge Python Bridge
After=network.target

[Service]
Type=simple
User=agent-forge
WorkingDirectory=/opt/agent-forge
ExecStart=/usr/bin/python3 src/api/python-bridge-server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Load Balancer Configuration

```nginx
# Nginx configuration
upstream python_bridge {
    server 127.0.0.1:8001;
    server 127.0.0.1:8002 backup;
}

location /api/python/ {
    proxy_pass http://python_bridge/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_connect_timeout 5s;
    proxy_read_timeout 30s;

    # Fallback to simulation on bridge failure
    error_page 502 503 504 = @fallback;
}

location @fallback {
    return 307 /api/simulation$request_uri;
}
```

## 📊 Monitoring

### Health Checks

```bash
# Basic health check
curl http://localhost:8001/health

# Detailed status
curl http://localhost:8001/api/sessions

# Performance metrics
curl http://localhost:8001/api/metrics
```

### Logging

```typescript
// Enable detailed logging
import { apiClient } from './src/api/utils/api-client';

apiClient.on('api_request', (event) => {
  console.log('API Request:', {
    endpoint: event.request.path,
    duration: event.duration,
    fallback: event.usedFallback
  });
});

apiClient.on('api_error', (event) => {
  console.error('API Error:', event.error);
});
```

## 🔒 Security Considerations

### API Key Authentication

```typescript
// Configure API keys for production
const apiClient = new ApiClient({
  baseUrl: 'https://api.example.com',
  headers: {
    'Authorization': 'Bearer your-api-key',
    'X-API-Version': '1.0'
  }
});
```

### Rate Limiting

```python
# In Python bridge server
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/api/cognate/status/{session_id}")
@limiter.limit("30/minute")
async def get_cognate_status(request: Request, session_id: str):
    # Implementation
    pass
```

### Input Validation

```typescript
// Validate session IDs
import { sessionUtils } from './src/api/utils/session-manager';

const sessionId = sessionUtils.validateSessionId(req.query.sessionId);
if (!sessionId) {
  return res.status(400).json({ error: 'Invalid session ID' });
}
```

## 📈 Performance Optimization

### Connection Pooling

```typescript
// Configure connection pooling
const apiClient = new ApiClient({
  baseUrl: 'http://localhost:8001',
  timeout: 5000,
  keepAlive: true,
  maxSockets: 50
});
```

### Response Caching

```typescript
// Cache frequently accessed data
import { LRUCache } from 'lru-cache';

const statusCache = new LRUCache({
  max: 1000,
  maxAge: 5000 // 5 seconds
});

const getCachedStatus = async (sessionId: string) => {
  const cached = statusCache.get(sessionId);
  if (cached) return cached;

  const status = await apiUtils.getCognateStatus(sessionId, fallback);
  statusCache.set(sessionId, status);
  return status;
};
```

## 📚 API Reference

### TypeScript Interfaces

```typescript
interface CognateConfig {
  sessionId: string;
  maxIterations?: number;
  convergenceThreshold?: number;
  parallelAgents?: number;
  timeout?: number;
  enableDebugging?: boolean;
  customParams?: Record<string, any>;
}

interface CognateMetrics {
  iterationsCompleted: number;
  convergenceScore: number;
  activeAgents: number;
  averageResponseTime: number;
  errorCount: number;
  successRate: number;
  lastUpdated: string;
  estimatedCompletion: string;
  throughput: number;
  memoryUsage: number;
}

interface EvoMergeMetrics {
  generationsCompleted: number;
  populationSize: number;
  fitnessScore: number;
  mutationRate: number;
  crossoverRate: number;
  eliteCount: number;
  averageFitness: number;
  bestFitness: number;
  diversityIndex: number;
  stagnationCount: number;
  lastUpdated: string;
  estimatedCompletion: string;
}
```

### Error Handling

```typescript
import { ApiError, NetworkError, TimeoutError } from './src/api/types/phase-interfaces';

try {
  const result = await apiClient.post('/api/cognate/start', config);
} catch (error) {
  if (error instanceof NetworkError) {
    console.log('Network issue, falling back to simulation');
  } else if (error instanceof TimeoutError) {
    console.log('Request timed out, try again');
  } else if (error instanceof ApiError) {
    console.log('API error:', error.status, error.message);
  } else {
    console.log('Unexpected error:', error);
  }
}
```

## 🤝 Contributing

### Development Setup

```bash
# Clone and setup
git clone <repository>
cd agent-forge
./scripts/setup-api-integration.sh

# Make changes
# ... edit files ...

# Run tests
./scripts/test-api-integration.sh

# Submit PR
```

### Adding New Endpoints

1. **Python Bridge**: Add endpoint to `python-bridge-server.py`
2. **TypeScript Interface**: Update `phase-interfaces.ts`
3. **API Client**: Add utility function to `api-client.ts`
4. **Next.js Route**: Create route in `src/web/dashboard/app/api/`
5. **Simulation**: Add fallback to `simulation/` directory
6. **Tests**: Add validation to `compatibility-validator.ts`

## 📋 Checklist

### Pre-deployment

- [ ] All tests pass (`./scripts/test-api-integration.sh`)
- [ ] Python bridge starts without errors
- [ ] Next.js API routes respond correctly
- [ ] Fallback mechanism works when bridge is down
- [ ] Session management handles edge cases
- [ ] Performance is acceptable under load
- [ ] Error messages are user-friendly
- [ ] Documentation is updated

### Production

- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Rate limiting enabled
- [ ] Monitoring and alerting setup
- [ ] Backup and recovery procedures
- [ ] Load balancer configured
- [ ] Security audit completed

---

**Success Criteria**: ✅ UI receives identical data structure from both real backend and simulation, with seamless fallback when backend is unavailable.

For support, check the troubleshooting section or create an issue in the project repository.