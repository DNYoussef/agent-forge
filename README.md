# Agent Forge - 8-Phase AI Agent Creation Pipeline

> Complete pipeline for building AI agents from the model level up through evolutionary optimization and compression

## Overview

Agent Forge is an advanced 8-phase pipeline system that creates AI agents from scratch, starting at the model level and progressing through evolutionary optimization, reasoning enhancement, compression, and specialized training.

## The 8 Phases

### Phase 1: Cognate - Model Creation & Initialization
- Creates foundation models from scratch (25M-1.5B parameters)
- Implements custom architectures (Cognate, Cogment, HRRM)
- Initializes with specialized tokenizers and embeddings

### Phase 2: EvoMerge - Evolutionary Model Optimization  
- Evolutionary merging of multiple model candidates
- 50+ generation evolution with fitness selection
- Advanced merge operators (SLERP, TIES, DARE, etc.)

### Phase 3: Quiet-STaR - Reasoning Enhancement
- Bakes reasoning capabilities into models
- Implements thought tokens and reasoning chains
- Theater detection to prevent fake improvements

### Phase 4: BitNet Compression - Initial Quantization
- 1.58-bit quantization for extreme compression
- Maintains accuracy while reducing model size by 8-16x
- Performance optimization for inference

### Phase 5: Forge Training - Main Training Loop
- Grokfast acceleration for faster convergence
- Edge learning and self-modeling
- Dream cycle integration for continuous improvement

### Phase 6: Tool & Persona Baking - Specialization
- Bakes specific tools and capabilities into models
- Persona optimization for specialized behaviors
- Identity crystallization through targeted training

### Phase 7: ADAS - Architecture Discovery & Search
- Automated architecture optimization
- Vector composition with Transformers Squared
- Multi-objective optimization (speed, accuracy, size)

### Phase 8: Final Compression - Production Optimization
- SeedLM compression for vocabulary optimization
- VPTQ (Vector Post-Training Quantization)
- Hypercompression for maximum efficiency

## Key Features

- **Multi-Agent Swarm Coordination**: 45+ specialized agents orchestrate the pipeline
- **Theater Detection System**: Identifies and eliminates performance theater
- **Production Ready**: NASA POT10 compliant quality gates
- **Web-Based Dashboard**: Real-time monitoring and control of all 8 phases

## User Interface

Agent Forge includes a comprehensive web-based dashboard for monitoring and controlling the entire 8-phase pipeline. Each phase has its own dedicated interface with real-time metrics, configuration controls, and visual feedback.

### Dashboard Overview

![Agent Forge Dashboard](docs/screenshots/dashboard.png)

The main dashboard provides:
- Pipeline control center with phase selection
- Real-time statistics (total agents, success rate, active pipelines)
- Individual phase cards for quick navigation
- Live progress tracking

### Phase Interfaces

Each phase has a dedicated UI with functional controls that connect directly to the pipeline:

#### Phase 1: Cognate (Model Creation)

![Phase 1: Cognate](docs/screenshots/phase1-cognate.png)

- Model type selection (Planner, Reasoner, Memory)
- Parameter configuration (vocab size, learning rate, batch size)
- Grokfast acceleration settings
- Real-time training metrics (loss, perplexity, grokking progress)
- Model architecture visualization

#### Phase 2: EvoMerge (Evolution)

![Phase 2: EvoMerge](docs/screenshots/phase2-evomerge.png)

- Evolution parameters (generations, population size, mutation rate)
- Merge technique selection (Linear, SLERP, TIES, DARE, Frankenmerge, DFS)
- Genetic operation controls (elite size, tournament size, crossover rate)
- Live fitness tracking and diversity metrics
- Population distribution visualization

#### Phase 3: Quiet-STaR (Reasoning)

![Phase 3: Quiet-STaR](docs/screenshots/phase3-quietstar.png)

- Thinking tokens configuration
- Mixing head weight control
- Reward function selection (correctness, coherence, efficiency, hybrid)
- Thought length and rollout settings
- Real-time reasoning performance metrics

#### Phase 4: BitNet (Compression)

![Phase 4: BitNet](docs/screenshots/phase4-bitnet.png)

- Quantization bit selection (1.58-bit)
- Optimization profile (development, production, inference)
- Compression ratio tracking
- Memory reduction metrics
- Performance retention monitoring

#### Phase 5: Forge Training

![Phase 5: Forge Training](docs/screenshots/phase5-forge.png)

- Training configuration (epochs, batch size, learning rate)
- Optimizer selection (Adam, SGD, RMSprop)
- Scheduler settings (cosine, linear, Grokfast)
- Live training metrics (loss, accuracy, learning rate)
- Training progress visualization

#### Phase 6: Tool & Persona Baking

![Phase 6: Tool & Persona Baking](docs/screenshots/phase6-baking.png)

- Tool selection and configuration
- Persona type selection
- Baking iteration controls
- A/B testing configuration
- Tool accuracy and persona coherence metrics

#### Phase 7: ADAS (Architecture Search)

![Phase 7: ADAS](docs/screenshots/phase7-adas.png)

- Search space configuration
- Search strategy selection (random, evolutionary, Bayesian)
- Hardware target optimization (CPU, GPU, Edge)
- Multi-objective optimization tracking
- Architecture diversity metrics

#### Phase 8: Final Compression

![Phase 8: Final Compression](docs/screenshots/phase8-final.png)

- SeedLM configuration
- VPTQ quantization settings
- Deployment target selection (cloud, edge, mobile)
- Compression ratio tracking
- Performance retention metrics

### Running the UI

```bash
# Start the backend API server
cd agent-forge
python src/api/pipeline_server_fixed.py

# Start the Next.js dashboard (in a new terminal)
cd src/web/dashboard
npm install
npm run dev
```

Access the dashboard at `http://localhost:3000`

## Quick Start

```python
from agent_forge.core.unified_pipeline import UnifiedPipeline

# Initialize pipeline
pipeline = UnifiedPipeline()

# Run full 8-phase pipeline
result = await pipeline.run_complete_pipeline(
    base_models=["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"],
    output_dir="./agent_output"
)
```

## Documentation

See [README_SWARM.md](./README_SWARM.md) for detailed swarm coordination documentation.

## License

MIT License
