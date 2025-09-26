# Agent Forge Integration Tests - Quick Start Guide

## 🚀 Quick Start (< 5 minutes)

### 1. Prerequisites
```bash
# Install Python dependencies
pip install pytest pytest-asyncio httpx websockets

# Install Playwright
npm install
npx playwright install
```

### 2. Start Services
```bash
# Terminal 1: Start API
cd src/api
uvicorn main:app --reload --port 8000

# Terminal 2: Start Frontend
cd src/web/dashboard
npm run dev
```

### 3. Run Tests
```bash
# All tests
./tests/integration/run_tests.sh

# Specific category
./tests/integration/run_tests.sh api
```

## 📁 Test Structure

```
tests/integration/
├── api/                  # REST API tests
├── websocket/           # WebSocket tests
├── e2e/                 # End-to-end tests
├── fixtures/            # Mock data
├── utils/               # Helpers
└── validation/          # System validation
```

## 🧪 Common Test Commands

### Run Specific Tests
```bash
# API tests
pytest tests/integration/api/ -v

# WebSocket tests
pytest tests/integration/websocket/ -v

# E2E Python tests
pytest tests/integration/e2e/test_phase_execution.py -v

# UI tests (Playwright)
npx playwright test tests/integration/e2e/test_ui_integration.spec.ts

# Validation suite
pytest tests/integration/validation/ -v
```

### Run with Options
```bash
# With coverage
pytest tests/integration/ --cov=src/api --cov-report=html

# Parallel execution
pytest tests/integration/ -n auto

# Stop on first failure
pytest tests/integration/ -x

# Verbose output
pytest tests/integration/ -vv

# Specific test
pytest tests/integration/api/test_pipeline_api.py::TestPipelineLifecycle::test_start_pipeline_success -v
```

### Using the Test Runner
```bash
# All tests
./tests/integration/run_tests.sh

# API tests only
./tests/integration/run_tests.sh api

# WebSocket tests only
./tests/integration/run_tests.sh websocket

# E2E tests only
./tests/integration/run_tests.sh e2e

# UI tests only
./tests/integration/run_tests.sh ui

# With coverage
./tests/integration/run_tests.sh coverage

# In parallel
./tests/integration/run_tests.sh parallel

# Check services
./tests/integration/run_tests.sh check

# Clean artifacts
./tests/integration/run_tests.sh clean

# Generate reports
./tests/integration/run_tests.sh report
```

## 💡 Quick Examples

### Example 1: Test Pipeline Start
```python
import pytest
from httpx import AsyncClient
from src.api.main import app

@pytest.mark.asyncio
async def test_start_pipeline():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/v1/pipeline/start", json={
            "phases": ["cognate", "evomerge"],
            "swarm_topology": "hierarchical"
        })
        assert response.status_code == 200
        assert "session_id" in response.json()
```

### Example 2: Test WebSocket
```python
from fastapi.testclient import TestClient
from src.api.main import app

def test_websocket_connection():
    client = TestClient(app)
    with client.websocket_connect("/ws/agents") as websocket:
        websocket.send_json({"type": "ping"})
        data = websocket.receive_json()
        assert data["type"] == "pong"
```

### Example 3: Use Test Helpers
```python
from tests.integration.utils.test_helpers import APITestClient, TestScenarios

@pytest.mark.asyncio
async def test_with_helpers():
    client = APITestClient()
    session_id = await TestScenarios.run_single_phase_pipeline(
        client, "cognate"
    )
    await TestScenarios.test_pause_resume_cycle(client, session_id)
```

### Example 4: Generate Mock Data
```python
from tests.integration.fixtures.pipeline_data_generator import PipelineDataGenerator

generator = PipelineDataGenerator()

# Generate config
config = generator.generate_pipeline_config(include_all_phases=True)

# Generate status
status = generator.generate_pipeline_status(
    session_id="test_123",
    current_phase="training"
)
```

### Example 5: UI Test (Playwright)
```typescript
test('should start pipeline from UI', async ({ page }) => {
  await page.goto('http://localhost:3000/phases/cognate');
  await page.click('button:has-text("Start")');
  await expect(page.locator('text=/running|started/i')).toBeVisible();
});
```

## 📊 Test Coverage

### What's Tested
- ✅ All 8 phases (individual + combined)
- ✅ Pipeline lifecycle (start/stop/pause/resume)
- ✅ WebSocket real-time updates (5 channels)
- ✅ Quality gates validation
- ✅ Checkpoint save/restore
- ✅ UI controls and interactions
- ✅ Error handling
- ✅ Performance validation

### Test Statistics
- **Total Tests**: 500+
- **API Tests**: 300+
- **WebSocket Tests**: 50+
- **E2E Tests**: 100+
- **UI Tests**: 40+
- **Coverage**: 95%+

## 🔧 Troubleshooting

### API Server Not Running
```bash
# Check if running
curl http://localhost:8000/api/v1/info

# Start server
cd src/api
uvicorn main:app --reload --port 8000
```

### Frontend Not Running
```bash
# Check if running
curl http://localhost:3000

# Start frontend
cd src/web/dashboard
npm run dev
```

### Import Errors
```bash
# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

# Or use the test runner
./tests/integration/run_tests.sh
```

### Playwright Issues
```bash
# Reinstall browsers
npx playwright install --force

# Run in headed mode
npx playwright test --headed

# Debug mode
npx playwright test --debug
```

## 📚 Resources

### Documentation
- `README.md` - Complete test suite guide
- `INTEGRATION_TEST_SUMMARY.md` - Detailed summary
- `QUICK_START.md` - This guide

### Test Files
- `api/test_pipeline_api.py` - API tests
- `websocket/test_websocket_connections.py` - WebSocket tests
- `e2e/test_phase_execution.py` - E2E Python tests
- `e2e/test_ui_integration.spec.ts` - UI tests
- `validation/test_system_validation.py` - Validation suite

### Utilities
- `utils/test_helpers.py` - Helper classes
- `fixtures/pipeline_data_generator.py` - Mock data

## 🎯 Quick Tips

1. **Run tests incrementally** - Start with API, then WebSocket, then E2E
2. **Use the test runner** - `run_tests.sh` handles setup automatically
3. **Check services first** - `./run_tests.sh check` before testing
4. **Use helpers** - Reduce boilerplate with `test_helpers.py`
5. **Generate mock data** - Use `pipeline_data_generator.py` for realistic data
6. **Run in parallel** - Use `-n auto` for faster execution
7. **Check coverage** - Use `--cov` to ensure complete coverage
8. **Clean artifacts** - Run `./run_tests.sh clean` periodically

## 🚦 Test Workflow

```
1. Start Services
   ↓
2. Check Services (./run_tests.sh check)
   ↓
3. Run Tests (./run_tests.sh)
   ↓
4. Review Reports (htmlcov/index.html)
   ↓
5. Clean Artifacts (./run_tests.sh clean)
```

## ✅ Success Checklist

- [ ] Services running (API + Frontend)
- [ ] Dependencies installed
- [ ] All tests passing
- [ ] Coverage > 90%
- [ ] No errors in logs
- [ ] Reports generated

---

**Quick Help**: `./run_tests.sh help`
**Full Docs**: See `README.md`
**Summary**: See `INTEGRATION_TEST_SUMMARY.md`