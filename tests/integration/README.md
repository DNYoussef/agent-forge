# Agent Forge Integration Test Suite

Comprehensive integration testing suite for the Agent Forge 8-phase AI agent creation pipeline.

## Overview

This test suite validates the complete integration between:
- **FastAPI Backend** (`src/api/`) - REST API and WebSocket endpoints
- **Next.js Frontend** (`src/web/dashboard/`) - UI components and real-time updates
- **Pipeline Execution** - 8-phase agent creation workflow
- **Swarm Coordination** - Multi-agent orchestration

## Test Structure

```
tests/integration/
├── api/                          # REST API endpoint tests
│   └── test_pipeline_api.py     # Pipeline lifecycle, config, quality gates
├── websocket/                    # WebSocket connection tests
│   └── test_websocket_connections.py  # Real-time streaming, events
├── e2e/                          # End-to-end workflow tests
│   ├── test_phase_execution.py  # Phase-by-phase execution
│   └── test_ui_integration.spec.ts  # Playwright UI tests
├── fixtures/                     # Test data and mocks
│   └── pipeline_data_generator.py  # Realistic mock data generators
├── utils/                        # Test utilities
│   └── test_helpers.py          # API clients, assertions, scenarios
├── validation/                   # System validation
│   └── test_system_validation.py  # Comprehensive validation suite
└── mocks/                        # Mock implementations
```

## Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install pytest pytest-asyncio httpx websockets

# Install Node.js dependencies
npm install

# Install Playwright browsers
npx playwright install
```

### Running Tests

#### All Integration Tests
```bash
# Python tests
pytest tests/integration/ -v

# Playwright tests
npx playwright test tests/integration/e2e/

# Run with coverage
pytest tests/integration/ --cov=src/api --cov-report=html
```

#### Specific Test Categories
```bash
# API tests only
pytest tests/integration/api/ -v

# WebSocket tests only
pytest tests/integration/websocket/ -v

# E2E Python tests
pytest tests/integration/e2e/test_phase_execution.py -v

# UI tests (Playwright)
npx playwright test tests/integration/e2e/test_ui_integration.spec.ts

# Validation suite
pytest tests/integration/validation/ -v
```

#### Test Options
```bash
# Run in parallel (faster)
pytest tests/integration/ -n auto

# Stop on first failure
pytest tests/integration/ -x

# Run specific test
pytest tests/integration/api/test_pipeline_api.py::TestPipelineLifecycle::test_start_pipeline_success -v

# Run with markers
pytest tests/integration/ -m "not slow" -v
```

## Test Categories

### 1. API Integration Tests (`api/`)

Tests REST API endpoints for:
- ✅ Pipeline lifecycle (start, stop, pause, resume, cancel)
- ✅ Status monitoring and progress tracking
- ✅ Swarm coordination and agent management
- ✅ Quality gate validation
- ✅ Checkpoint save/restore
- ✅ Configuration presets
- ✅ Execution history
- ✅ Health checks

**Example:**
```python
@pytest.mark.asyncio
async def test_start_pipeline_success():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/v1/pipeline/start", json={
            "phases": ["cognate", "evomerge"],
            "swarm_topology": "hierarchical",
            "max_agents": 50
        })
        assert response.status_code == 200
        assert "session_id" in response.json()
```

### 2. WebSocket Integration Tests (`websocket/`)

Tests real-time streaming for:
- ✅ Agent status updates
- ✅ Task execution progress
- ✅ Performance metrics streaming
- ✅ Pipeline progress events
- ✅ Dashboard data aggregation
- ✅ Connection management
- ✅ Reconnection handling

**Example:**
```python
def test_websocket_metrics_streaming():
    client = TestClient(app)
    with client.websocket_connect("/ws/metrics") as websocket:
        data = websocket.receive_json(timeout=3)
        assert "metrics" in data["data"]
        assert "cpu_percent" in data["data"]["metrics"]
```

### 3. End-to-End Tests (`e2e/`)

#### Python E2E (`test_phase_execution.py`)
Tests complete pipeline workflows:
- ✅ Single phase execution (all 8 phases)
- ✅ Multi-phase pipelines
- ✅ Full 8-phase pipeline
- ✅ Quality gate enforcement
- ✅ Checkpoint recovery
- ✅ Pipeline control during execution
- ✅ Swarm coordination

**Example:**
```python
@pytest.mark.asyncio
async def test_full_pipeline_execution():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/v1/pipeline/start", json={
            "phases": [
                "cognate", "evomerge", "quietstar", "bitnet",
                "training", "baking", "adas", "compression"
            ],
            "enable_monitoring": True,
            "enable_checkpoints": True
        })
        assert response.status_code == 200
```

#### UI E2E (`test_ui_integration.spec.ts`)
Tests frontend integration:
- ✅ Dashboard loading and stats display
- ✅ Phase navigation
- ✅ Pipeline control from UI
- ✅ Real-time WebSocket updates
- ✅ Progress visualization
- ✅ Responsive design
- ✅ API integration

**Example:**
```typescript
test('should start pipeline from UI', async ({ page }) => {
  await page.goto('/phases/cognate');
  await page.click('button:has-text("Start")');
  await expect(page.locator('text=/running|started/i')).toBeVisible();
});
```

### 4. Test Data & Fixtures (`fixtures/`)

Realistic mock data generators:
- ✅ Pipeline configurations
- ✅ Phase execution states
- ✅ Agent status
- ✅ Performance metrics
- ✅ Quality gate results
- ✅ Checkpoint metadata
- ✅ Execution history
- ✅ WebSocket events

**Example:**
```python
from fixtures.pipeline_data_generator import PipelineDataGenerator

generator = PipelineDataGenerator()
config = generator.generate_pipeline_config(include_all_phases=True)
status = generator.generate_pipeline_status(session_id, current_phase="training")
metrics = generator.generate_metrics()
```

### 5. Test Utilities (`utils/`)

Helper classes and functions:
- ✅ `APITestClient` - Simplified API testing
- ✅ `WebSocketTestClient` - WebSocket testing utilities
- ✅ `AssertionHelpers` - Common assertions
- ✅ `TestScenarios` - Pre-built test workflows
- ✅ `MockDataBuilder` - Fluent config builder

**Example:**
```python
from utils.test_helpers import APITestClient, TestScenarios

client = APITestClient()
session_id = await TestScenarios.run_single_phase_pipeline(
    client, "cognate", {"base_models": ["gpt2"]}
)
await TestScenarios.test_pause_resume_cycle(client, session_id)
```

### 6. System Validation (`validation/`)

Comprehensive validation tests:
- ✅ End-to-end workflows
- ✅ Data consistency across components
- ✅ Error handling and recovery
- ✅ Performance benchmarks
- ✅ Security validations
- ✅ Robustness testing
- ✅ Edge cases

**Example:**
```python
@pytest.mark.asyncio
async def test_complete_8_phase_pipeline():
    """Validate full pipeline execution with all quality gates"""
    client = APITestClient()
    session_id = await TestScenarios.run_multi_phase_pipeline(
        client,
        phases=PipelineDataGenerator.PHASES,
        enable_checkpoints=True
    )
    # Validate each phase's quality gates
    for phase in phases:
        gate_result = await client.validate_quality_gates(session_id, phase)
        AssertionHelpers.assert_quality_gate_result(gate_result)
```

## Test Coverage

Current test coverage areas:

### API Coverage
- ✅ Pipeline endpoints (100%)
- ✅ WebSocket endpoints (100%)
- ✅ Control operations (100%)
- ✅ Quality gates (100%)
- ✅ Checkpoint management (100%)
- ✅ Configuration presets (100%)

### Phase Coverage
- ✅ Cognate (model initialization)
- ✅ EvoMerge (evolutionary merging)
- ✅ Quiet-STaR (reasoning)
- ✅ BitNet (compression)
- ✅ Training (model training)
- ✅ Baking (tool integration)
- ✅ ADAS (architecture search)
- ✅ Compression (final optimization)

### Integration Coverage
- ✅ Backend ↔ Frontend
- ✅ REST API ↔ WebSocket
- ✅ Pipeline ↔ Swarm
- ✅ UI ↔ Real-time updates

## Environment Setup

### Backend (FastAPI)
```bash
# Start API server
cd src/api
uvicorn main:app --reload --port 8000
```

### Frontend (Next.js)
```bash
# Start development server
cd src/web/dashboard
npm run dev
```

### Environment Variables
```bash
# .env.test
API_BASE_URL=http://localhost:8000
UI_BASE_URL=http://localhost:3000
WEBSOCKET_URL=ws://localhost:8000
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio httpx websockets

      - name: Run API tests
        run: pytest tests/integration/api/ -v

      - name: Run WebSocket tests
        run: pytest tests/integration/websocket/ -v

      - name: Run E2E tests
        run: pytest tests/integration/e2e/test_phase_execution.py -v

      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install Playwright
        run: |
          npm install
          npx playwright install --with-deps

      - name: Run UI tests
        run: npx playwright test tests/integration/e2e/test_ui_integration.spec.ts
```

## Best Practices

### 1. Test Isolation
- Each test is independent
- No shared state between tests
- Clean up resources after tests

### 2. Realistic Data
- Use `PipelineDataGenerator` for realistic mocks
- Test with various configurations
- Include edge cases

### 3. Async Testing
- Use `pytest.mark.asyncio` for async tests
- Properly await all async operations
- Handle timeouts gracefully

### 4. Error Handling
- Test both success and failure paths
- Verify error messages
- Test recovery mechanisms

### 5. Performance
- Keep tests fast (< 5s per test)
- Use parallel execution when possible
- Mock expensive operations

## Troubleshooting

### Common Issues

**1. WebSocket connection failures**
```bash
# Ensure API server is running
uvicorn src.api.main:app --reload
```

**2. Import errors**
```bash
# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

**3. Playwright browser issues**
```bash
# Reinstall browsers
npx playwright install --force
```

**4. Async test failures**
```python
# Ensure proper event loop handling
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
```

## Contributing

### Adding New Tests

1. **API Tests**: Add to `tests/integration/api/`
2. **WebSocket Tests**: Add to `tests/integration/websocket/`
3. **E2E Tests**: Add to `tests/integration/e2e/`
4. **Use helpers**: Leverage `test_helpers.py` utilities

### Test Naming Convention
- `test_<feature>_<scenario>.py` for files
- `test_<action>_<expected_result>` for functions
- Use descriptive names

### Documentation
- Add docstrings to test classes and methods
- Update this README for new test categories
- Document any new fixtures or utilities

## Metrics & Reporting

### Generate Coverage Report
```bash
pytest tests/integration/ --cov=src/api --cov-report=html
open htmlcov/index.html
```

### Playwright Report
```bash
npx playwright test --reporter=html
npx playwright show-report
```

### Test Timing
```bash
pytest tests/integration/ --durations=10
```

## Future Enhancements

- [ ] Load testing for concurrent pipelines
- [ ] Stress testing for WebSocket connections
- [ ] Security penetration testing
- [ ] Performance regression testing
- [ ] Visual regression testing (UI)
- [ ] Contract testing for API
- [ ] Mutation testing for coverage

## Support

For issues or questions:
1. Check troubleshooting section
2. Review test examples
3. Open an issue with test logs
4. Include environment details

---

**Last Updated**: 2024-01-20
**Test Suite Version**: 1.0.0
**Maintained By**: Agent Forge Team