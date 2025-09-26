# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agent Forge is an 8-phase AI agent creation pipeline that builds models from scratch through evolutionary optimization, reasoning enhancement, compression, and specialized training. The project implements custom architectures (Cognate, Cogment, HRRM) with multi-agent swarm coordination.

## Build & Development Commands

### Python Environment
```bash
# Install dependencies
pip install -r agent_forge/requirements.txt
pip install -e .  # Install project in editable mode

# Run the main pipeline
python agent_forge/core/unified_pipeline.py

# Start the API server
python agent_forge/api/pipeline_server_fixed.py
```

### Web Dashboard
```bash
# Install and run the dashboard
cd src/web/dashboard
npm install
npm run dev  # Runs on http://localhost:3000
```

### Testing
```bash
# Python tests
pytest tests/                    # Run all tests
pytest tests/unit/               # Run unit tests only
pytest -m integration            # Run integration tests
pytest -v                        # Verbose output

# E2E tests (Playwright)
npm run test:e2e                # Run all E2E tests
npm run test:e2e:ui             # Run with UI mode
npm run test:e2e:chromium       # Test Chrome only
```

### Code Quality
```bash
# Linting and formatting
black agent_forge/ --line-length 100
isort agent_forge/
flake8 agent_forge/
mypy agent_forge/ --strict

# Type checking
mypy agent_forge/core/
```

## Architecture Overview

### Core Pipeline Structure
The system orchestrates 8 phases through `agent_forge/core/unified_pipeline.py`:

1. **Phase 1: Cognate** - Model creation from scratch (25M-1.5B parameters)
2. **Phase 2: EvoMerge** - Evolutionary optimization with 50+ generations
3. **Phase 3: Quiet-STaR** - Reasoning enhancement through thought tokens
4. **Phase 4: BitNet** - 1.58-bit quantization compression
5. **Phase 5: Forge Training** - Main training with Grokfast acceleration
6. **Phase 6: Tool & Persona** - Capability and identity baking
7. **Phase 7: ADAS** - Architecture discovery and optimization
8. **Phase 8: Final Compression** - Production optimization with SeedLM/VPTQ

### Key Components

**Phase Controllers** (`agent_forge/core/phase_controller.py`):
- `PhaseController`: Base class for individual phases
- `PhaseOrchestrator`: Manages phase execution and dependencies
- `PhaseResult`: Standardized phase output format

**Model Architectures** (`agent_forge/models/`):
- `cognate/`: Custom Cognate architecture implementation
- `cogment/`: Cogment model with distributed training
- `hrrm/`: Hierarchical Recursive Reasoning Model

**Phase Implementations** (`agent_forge/phases/`):
- Each phase has its own directory with core logic
- Standardized interface through PhaseController
- Configuration via YAML files in `configs/`

### API Architecture
- FastAPI server at `agent_forge/api/pipeline_server_fixed.py`
- WebSocket support for real-time updates
- RESTful endpoints for phase control and monitoring

## Phase-Specific Commands

```python
# Run individual phases
from agent_forge.phases.phase2_evomerge import EvoMergePhase
phase = EvoMergePhase()
result = await phase.run(config)

# Run complete pipeline
from agent_forge.core.unified_pipeline import UnifiedPipeline
pipeline = UnifiedPipeline()
result = await pipeline.run_complete_pipeline(
    base_models=["model_name"],
    output_dir="./output"
)
```

## File Organization

- `/agent_forge/` - Main Python package
  - `/core/` - Pipeline orchestration
  - `/phases/` - Individual phase implementations
  - `/models/` - Model architectures
  - `/api/` - FastAPI server and endpoints
  - `/benchmarks/` - Performance benchmarking
- `/src/web/dashboard/` - Next.js dashboard application
- `/tests/` - Test suites
- `/docs/` - Documentation

## Critical Implementation Details

### Theater Detection System
The system includes theater detection to identify fake performance improvements. Located in phases, it validates genuine model improvements vs. superficial changes.

### Grokfast Acceleration
Integrated throughout training phases for faster convergence. Key parameters:
- Amplification factor: Controls acceleration strength
- Window size: Gradient history for momentum
- Lambda: Regularization weight

### Multi-Agent Coordination
45+ specialized agents work in parallel. Coordination through:
- Swarm topology selection (mesh, hierarchical, adaptive)
- Consensus protocols for decision-making
- Shared memory and state management

### Compression Pipeline
Multi-stage compression achieving 8-16x size reduction:
1. BitNet 1.58-bit quantization (Phase 4)
2. SeedLM vocabulary optimization (Phase 8)
3. VPTQ vector quantization (Phase 8)
4. Hypercompression final pass (Phase 8)

## Environment Variables

```bash
# API Configuration
PIPELINE_API_HOST=0.0.0.0
PIPELINE_API_PORT=8000

# Model Paths
MODEL_CACHE_DIR=./models
OUTPUT_DIR=./output

# Performance
TORCH_NUM_THREADS=8
OMP_NUM_THREADS=8
```

## Common Development Tasks

### Adding a New Phase
1. Create directory in `agent_forge/phases/phase{N}_{name}/`
2. Implement PhaseController interface
3. Register in `unified_pipeline.py`
4. Add UI component in dashboard

### Debugging Pipeline Issues
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use phase-specific debugging
phase.debug_mode = True
phase.save_checkpoints = True
```

### Performance Monitoring
- Dashboard provides real-time metrics at http://localhost:3000
- API metrics endpoint: http://localhost:8000/metrics
- Phase-specific logs in `./output/{phase_name}/logs/`