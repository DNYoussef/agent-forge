# Agent Forge Phase Control Interface Design

## Overview

This document provides detailed control panel specifications for all 8 phases of the Agent Forge ML pipeline. Each phase has been analyzed to identify key configuration parameters, metrics, visualizations, and control requirements.

---

## Phase 1: Cognate Pretrain

### Purpose
Create three foundational models (Planner, Reasoner, Memory) with specialized architectures optimized for different cognitive tasks.

### Key Configuration Parameters

#### Model Architecture
- **Model Types** (Multi-select): `planner | reasoner | memory`
  - Default: All three selected
  - Validation: At least one must be selected

- **Base Architecture** (Dropdown): `transformer | llama | gpt`
  - Default: `transformer`
  - Dependency: Determines available layer configurations

- **Model Size** (Radio):
  - `nano` (12M params) | `small` (25M) | `medium` (50M) | `large` (100M)
  - Default: `small`
  - Validation: Must fit in available GPU memory

#### Training Parameters
- **Learning Rate** (Slider): `1e-6` to `1e-3`
  - Default: `1e-4`
  - Log scale: true
  - Validation: Must be positive

- **Batch Size** (Number): `1` to `256`
  - Default: `32`
  - Validation: Must be power of 2 for optimal GPU utilization

- **Max Epochs** (Number): `1` to `100`
  - Default: `10`
  - Early stopping: Enabled if convergence detected

- **Warmup Steps** (Number): `0` to `10000`
  - Default: `1000`
  - Recommendation: ~10% of total training steps

#### Grokfast Acceleration
- **Enable Grokfast** (Toggle): `on | off`
  - Default: `on`
  - Impact: 50x faster grokking

- **EMA Alpha** (Slider): `0.9` to `0.999`
  - Default: `0.98`
  - Tooltip: "Controls gradient smoothing (higher = smoother)"

- **Lambda Factor** (Slider): `0.01` to `0.5`
  - Default: `0.05`
  - Tooltip: "Amplification strength for slow gradients"

### Metrics & Visualizations

#### Real-time Metrics
- **Training Loss** (Line chart)
  - Update frequency: Every 10 steps
  - Y-axis: Log scale

- **Validation Perplexity** (Line chart)
  - Per model type (3 lines)
  - Update frequency: Every 100 steps

- **Grokking Progress** (Gauge)
  - 0-100% scale
  - Color: Red < 50%, Yellow 50-75%, Green > 75%

- **Memory Usage** (Bar chart)
  - GPU vs CPU
  - Update frequency: Every second

#### Phase Completion Metrics
- Model Quality Score: 0-100
- Parameter Count per model
- Training Time (hours)
- Token throughput (tokens/sec)

### Controls

#### Start/Stop/Pause
- **Start Training** (Button)
  - Validation: Check GPU availability, dataset loaded
  - State transition: Idle → Running

- **Pause/Resume** (Toggle button)
  - Saves current state
  - Can resume from checkpoint

- **Stop Training** (Button)
  - Confirmation dialog if progress > 10%
  - Saves best checkpoint automatically

#### Advanced Controls
- **Load Checkpoint** (File picker)
  - Formats: `.pt`, `.safetensors`
  - Validates model compatibility

- **Export Models** (Button)
  - Formats: HuggingFace, ONNX, TorchScript
  - Destination: Local or S3

### Validation & Dependencies
- **GPU Memory Check**: Required before start
- **Dataset Availability**: Download if missing
- **Model Compatibility**: Validate architecture params
- **Dependencies**: Phase 2 requires Phase 1 models

---

## Phase 2: EvoMerge

### Purpose
Evolutionary optimization to merge multiple base models using various merging techniques, creating a strong foundation model.

### Key Configuration Parameters

#### Base Models
- **Model Sources** (Multi-select + Upload):
  - HuggingFace models
  - Local model paths
  - Phase 1 outputs (auto-detected)
  - Default: Use Phase 1 outputs if available

- **Prefer Seed Models** (Toggle):
  - Default: `on` if Phase 1 completed
  - Benefit: Faster iteration with smaller models

#### Evolution Settings
- **Generations** (Slider): `10` to `100`
  - Default: `50`
  - Validation: More generations = better results but longer time

- **Population Size** (Number): `4` to `16`
  - Default: `8`
  - Validation: Must be even number
  - Resource impact: Higher = more GPU memory

- **Mutation Rate** (Slider): `0.0` to `0.5`
  - Default: `0.1`
  - Tooltip: "Probability of random variation"

- **Crossover Rate** (Slider): `0.0` to `1.0`
  - Default: `0.7`
  - Tooltip: "Probability of combining two parents"

#### Merge Techniques (Multi-select)
- **Enabled Techniques**:
  - ☑ Linear (weighted average)
  - ☑ SLERP (spherical interpolation)
  - ☑ TIES (trim, interpolate, elect, sign)
  - ☑ DARE (drop and rescale)
  - ☑ Frankenmerge (layer mixing)
  - ☑ DFS (hierarchical merging)
  - ☐ Task Arithmetic (optional)

  Default: All except Task Arithmetic
  Validation: At least 2 must be selected

#### Evaluation Domains
- **Domain Weights** (Sliders, must sum to 1.0):
  - Code: 0-100% (default: 25%)
  - Math: 0-100% (default: 25%)
  - Multilingual: 0-100% (default: 25%)
  - Structured Data: 0-100% (default: 25%)

### Metrics & Visualizations

#### Evolution Progress
- **Fitness Over Generations** (Multi-line chart)
  - Best, Average, Worst fitness
  - Update: Per generation

- **Population Diversity** (Line chart)
  - Measures exploration vs exploitation
  - Warning if < 0.1 (premature convergence)

- **Pareto Front** (Scatter plot)
  - X-axis: Performance
  - Y-axis: Efficiency
  - Points: Current population

#### Domain Performance (Radar chart)
- 4 axes for each evaluation domain
- Shows current best model vs baseline
- Update: Per generation

#### Merge Technique Analysis (Stacked bar)
- Success rate per technique
- Contribution to best models

### Controls

#### Breeding Strategy (Radio buttons)
- **Tournament Selection** (default)
  - Top 2 → 6 children
  - Bottom 6 → 2 children (groups of 3)

- **NSGA-II**
  - Multi-objective optimization
  - Pareto-optimal selection

- **Random**
  - Pure random breeding
  - For baseline comparison

#### Convergence Controls
- **Plateau Patience** (Number): `3` to `10`
  - Default: `5`
  - Stops if no improvement for N generations

- **Convergence Threshold** (Slider): `0.0001` to `0.01`
  - Default: `0.001`
  - Fitness variance threshold

- **Auto-cleanup** (Toggle)
  - Default: `on`
  - Removes old generation models to save space

### Validation & Dependencies
- **Model Compatibility**: Validates architectures can be merged
- **Benchmark Datasets**: HumanEval, GSM8K, HellaSwag, ARC
- **Memory Requirements**: 8GB+ GPU for 1.5B models
- **Dependencies**: Requires base models or Phase 1 output

---

## Phase 3: Quiet-STaR (Baking)

### Purpose
Bake reasoning tokens into model weights through iterative training until thoughts "stick."

### Key Configuration Parameters

#### Thought Tokens
- **Special Tokens** (Text inputs):
  - Start Thought: `<|startofthought|>` (customizable)
  - End Thought: `<|endofthought|>`
  - No Thought: `<|nothought|>`

  Validation: Must not conflict with model vocab

- **Max Thought Length** (Slider): `16` to `256`
  - Default: `64`
  - Impact: Longer thoughts = more expressiveness but slower

- **Thought Probability** (Slider): `0.0` to `1.0`
  - Default: `0.5`
  - Tooltip: "Probability of inserting thought tokens"

#### Baking Configuration
- **Max Iterations** (Number): `1` to `20`
  - Default: `5`
  - Early stopping when convergence reached

- **Convergence Threshold** (Slider): `0.80` to `0.99`
  - Default: `0.95`
  - Tooltip: "Stop when thoughts stick at this rate"

#### Cognitive Strategies (Multi-select)
- ☑ Systems Thinking
- ☑ First Principles
- ☑ Cross-Domain Analysis
- ☑ Probabilistic Thinking
- ☑ Rapid Iteration
- ☑ Paradox Resolution

Default: All selected
Impact: More strategies = richer reasoning but longer training

#### A/B Testing
- **Test Rounds** (Number): `1` to `10`
  - Default: `3`
  - Compares baked vs unbaked model

- **Significance Threshold** (Slider): `0.01` to `0.10`
  - Default: `0.05` (p-value)

- **Min Improvement** (Slider): `0.0` to `0.1`
  - Default: `0.02` (2%)
  - Required improvement to accept baking

#### Loss Weights
- **Task Loss Weight** (Slider): `0.0` to `2.0`
  - Default: `1.0`

- **Reflection Loss Weight** (Slider): `0.0` to `2.0`
  - Default: `0.3`

- **Leak Prevention Weight** (Slider): `0.0` to `20.0`
  - Default: `10.0`
  - Prevents thought tokens leaking into outputs

### Metrics & Visualizations

#### Baking Progress
- **Thought Stickiness** (Line chart per iteration)
  - % of thoughts that stick in weights
  - Target line at convergence threshold

- **Reasoning Quality** (Grouped bar chart)
  - Per cognitive strategy
  - Before vs After baking

- **Loss Components** (Stacked area chart)
  - Task, Reflection, Leak losses over time

#### A/B Test Results (Table)
- Model variant
- Performance metric
- P-value
- Statistical significance indicator

#### ThoughtMixingHead Analysis
- **Attention Patterns** (Heatmap)
  - Shows which layers attend to thoughts

- **Gate Activation** (Histogram)
  - Distribution of thought gate values

### Controls

#### Baking Mode (Radio)
- **Iterative** (default)
  - Multiple passes until convergence

- **Single Pass**
  - One-shot baking (faster, less thorough)

- **Adaptive**
  - Adjusts iteration count based on progress

#### Dataset Selection
- **Evaluation Dataset** (Dropdown)
  - GSM8K (math reasoning)
  - MMLU (general knowledge)
  - Custom dataset (upload)

- **Eval Samples** (Number): `50` to `1000`
  - Default: `100`
  - More samples = reliable metrics but slower

### Validation & Dependencies
- **Tokenizer Compatibility**: Validates special tokens
- **Model Architecture**: Must support mixing head
- **Dependencies**: Requires Phase 2 merged model

---

## Phase 4: BitNet Compression

### Purpose
Apply 1-bit quantization techniques to drastically reduce model size while maintaining performance.

### Key Configuration Parameters

#### Quantization Settings
- **Bit Width** (Radio):
  - 1-bit (ternary: -1, 0, 1)
  - 1.58-bit (quaternary: -1, -0.5, 0.5, 1)
  - 2-bit (extended range)

  Default: 1.58-bit
  Impact: Lower bits = smaller but may hurt quality

- **Quantization Scope** (Checkboxes):
  - ☑ Weights
  - ☑ Activations
  - ☐ Gradients (training only)

  Default: Weights + Activations

#### Layer Selection
- **Quantize Layers** (Multi-select list):
  - All layers (default)
  - Embedding layers only
  - Attention layers only
  - FFN layers only
  - Custom selection

  Recommendation: Start with FFN, then Attention

#### Calibration
- **Calibration Dataset** (Dropdown):
  - C4 subset (default)
  - WikiText
  - Custom dataset

- **Calibration Samples** (Slider): `128` to `2048`
  - Default: `512`
  - More samples = better calibration but slower

- **Scaling Method** (Radio):
  - Per-tensor (faster)
  - Per-channel (better quality)
  - Learned scales (best quality, slowest)

  Default: Per-channel

### Metrics & Visualizations

#### Compression Results
- **Model Size Reduction** (Gauge)
  - Before/After size in MB/GB
  - Compression ratio (e.g., "7.3x smaller")

- **Parameter Distribution** (Histogram)
  - Before: Full precision
  - After: Quantized values
  - Shows value clustering

- **Layer-wise Impact** (Bar chart)
  - Y-axis: Layers
  - X-axis: Performance change
  - Color: Red (degradation) to Green (maintained)

#### Performance Validation
- **Perplexity Comparison** (Table)
  - Full precision baseline
  - Quantized model
  - Delta (%)

- **Task Performance** (Radar chart)
  - Multiple evaluation tasks
  - Full vs Quantized

### Controls

#### Quantization Strategy (Tabs)
- **Post-Training** (default)
  - No training required
  - Fast deployment

- **Quantization-Aware Training**
  - Fine-tune after quantization
  - Better quality, longer time

- **Mixed Precision**
  - Keep critical layers in higher precision
  - Auto-identify based on sensitivity

#### Export Options
- **Export Format** (Checkboxes):
  - ☑ ONNX
  - ☑ TensorRT
  - ☐ CoreML
  - ☐ OpenVINO

- **Target Hardware** (Radio):
  - CPU (optimize for AVX2/AVX512)
  - GPU (CUDA kernels)
  - Mobile (ARM optimization)
  - Edge (minimal dependencies)

### Validation & Dependencies
- **Accuracy Threshold**: Warn if >5% perplexity increase
- **Hardware Compatibility**: Check target hardware support
- **Dependencies**: Requires Phase 3 or Phase 5 model

---

## Phase 5: Forge Training

### Purpose
Main training loop with edge-of-chaos optimization, self-modeling, dream cycles, and Grokfast acceleration.

### Key Configuration Parameters

#### Training Configuration
- **Max Steps** (Number): `1000` to `100000`
  - Default: `50000`
  - Validation: Must have enough data

- **Learning Rate Schedule** (Dropdown + params):
  - Cosine (warmup + decay)
  - Linear warmup + constant
  - Inverse sqrt
  - Custom

  Default: Cosine with warmup

- **Gradient Accumulation** (Number): `1` to `16`
  - Default: `4`
  - Enables larger effective batch sizes

#### Edge-of-Chaos Controller
- **Enable Edge Control** (Toggle): `on | off`
  - Default: `on`

- **Target Success Range** (Range slider): `0.0` to `1.0`
  - Default: `0.55` to `0.75`
  - Tooltip: "Optimal learning zone"

- **Window Size** (Number): `50` to `500`
  - Default: `100`
  - Rolling window for success rate calculation

- **Exploration Rate** (Slider): `0.0` to `0.5`
  - Default: `0.1`
  - Chance of random difficulty adjustment

#### Self-Modeling
- **Enable Self-Model** (Toggle): `on | off`
  - Default: `on`

- **Self-Model Weight** (Slider): `0.0` to `1.0`
  - Default: `0.1`
  - Contribution to total loss

- **Monitor Layers** (Multi-select):
  - Layer indices to monitor
  - Default: `[4, 8, 12]` (every 4th layer)

#### Dream Cycles
- **Enable Dreams** (Toggle): `on | off`
  - Default: `on`

- **Dream Interval** (Number): `100` to `5000`
  - Default: `1000` steps

- **Dream Duration** (Number): `10` to `200`
  - Default: `50` steps

- **Buffer Capacity** (Number): `1000` to `50000`
  - Default: `10000` experiences

- **Augmentation Strength** (Slider): `0.0` to `1.0`
  - Default: `0.2`

#### Temperature Curriculum
- **Enable Curriculum** (Toggle): `on | off`
  - Default: `on`

- **Initial Temperature** (Slider): `0.1` to `2.0`
  - Default: `1.0`

- **Final Temperature** (Slider): `0.01` to `1.0`
  - Default: `0.1`

- **Curriculum Schedule** (Dropdown):
  - Linear
  - Exponential
  - Step-wise

  Default: Linear

### Metrics & Visualizations

#### Training Dashboard
- **Training Loss** (Multi-line chart)
  - Total loss
  - Task loss
  - Self-modeling loss
  - Update: Every 10 steps

- **Edge-of-Chaos Status** (Real-time gauge)
  - Current success rate
  - Target range overlay
  - Color: Green (in range), Yellow (near edge), Red (off)

- **Grokking Progress** (Line chart)
  - Training accuracy
  - Validation accuracy
  - Shows grokking moment if occurs

- **Dream Cycle Impact** (Scatter plot)
  - X: Regular training steps
  - Y: Performance
  - Highlighted: Dream cycle intervals

#### Self-Modeling Analysis
- **Efficiency Predictions** (Line chart)
  - Predicted vs Actual compute efficiency
  - Per monitored layer

- **Geometry Probe** (3D scatter)
  - Hidden state manifold
  - Color by training progress
  - Interactive rotation

#### Resource Utilization
- **GPU Utilization** (Area chart)
  - Memory usage
  - Compute usage
  - Temperature (optional)

- **Token Throughput** (Line chart)
  - Tokens/second over time
  - Average throughput indicator

### Controls

#### Training Tasks (Tabs)
- **Language Modeling**
  - Next token prediction
  - Default dataset

- **Arithmetic**
  - Math problem solving
  - Difficulty progression

- **Pattern Matching**
  - Sequence completion
  - Abstract reasoning

**Task Switching**:
- Automatic (interval-based)
- Manual (user-triggered)
- Adaptive (based on performance)

#### Checkpointing
- **Save Interval** (Number): `100` to `5000`
  - Default: `1000` steps

- **Keep Best N** (Number): `1` to `10`
  - Default: `3`
  - Based on validation metric

- **Save Optimizer State** (Toggle):
  - Default: `on`
  - Disable to save space

#### Emergency Controls
- **Emergency Stop** (Button)
  - Immediate halt
  - Saves current state

- **Reduce Batch Size** (Button)
  - If OOM errors occur
  - Halves current batch size

### Validation & Dependencies
- **Memory Check**: Estimate memory before start
- **Dataset Validation**: Check all task datasets loaded
- **Checkpoint Space**: Verify disk space available
- **Dependencies**: Can use output from Phase 3 or Phase 4

---

## Phase 6: Tool Persona Baking

### Purpose
Bake tool usage patterns and persona characteristics into the model weights.

### Key Configuration Parameters

#### Persona Configuration
- **Persona Templates** (Multi-select + Custom):
  - Analytical (data-focused)
  - Creative (generative)
  - Helpful (assistance-focused)
  - Professional (formal)
  - Friendly (casual)
  - Custom (JSON upload)

  Default: Analytical + Helpful

- **Persona Strength** (Slider): `0.0` to `1.0`
  - Default: `0.5`
  - Lower = subtle, Higher = strong personality

#### Tool Configuration
- **Available Tools** (Checklist):
  - ☑ Calculator
  - ☑ Code Interpreter
  - ☑ Web Search
  - ☑ File Operations
  - ☐ Database Query
  - Custom tools (upload spec)

- **Tool Usage Frequency** (Sliders per tool):
  - 0-100% (proportion in training data)
  - Auto-balance: Equal distribution
  - Realistic: Matches real-world usage

#### Baking Strategy
- **Baking Method** (Radio):
  - Iterative refinement (default)
  - Single-pass injection
  - Gradual fade-in

- **Convergence Metric** (Dropdown):
  - Tool call accuracy
  - Persona consistency
  - Combined score

  Default: Combined score

- **Max Baking Rounds** (Number): `1` to `10`
  - Default: `3`

### Metrics & Visualizations

#### Persona Analysis
- **Trait Distribution** (Radar chart)
  - 5+ personality dimensions
  - Before vs After baking

- **Response Style Examples** (Text comparison)
  - Same prompt, different baking levels
  - Shows personality emergence

#### Tool Usage Metrics
- **Tool Call Accuracy** (Bar chart)
  - Per tool type
  - Correct vs Incorrect invocations

- **Tool Selection Patterns** (Sankey diagram)
  - Query type → Tool chosen → Outcome
  - Shows routing effectiveness

- **Baking Progress** (Multi-line chart)
  - Persona consistency
  - Tool accuracy
  - Overall convergence

### Controls

#### Persona Customization
- **Upload Persona Spec** (File picker)
  - JSON format
  - Validates schema

- **Live Persona Test** (Interactive panel)
  - Enter prompt
  - See response with current baking level
  - Adjust parameters in real-time

#### Tool Fine-tuning
- **Tool Examples Dataset** (Upload):
  - CSV/JSON with tool usage examples
  - Auto-validates format

- **Error Correction Mode** (Toggle):
  - Emphasizes fixing tool call mistakes
  - Useful if high error rate detected

### Validation & Dependencies
- **Tool Spec Validation**: All tools must have valid schemas
- **Persona Coherence**: Check for conflicting traits
- **Dependencies**: Requires Phase 5 trained model

---

## Phase 7: ADAS (Architecture Discovery)

### Purpose
Search for optimal model architectures using vector composition and evolutionary optimization.

### Key Configuration Parameters

#### Search Space
- **Architecture Parameters** (Ranges):
  - Num Layers: `4` to `48` (default: 12)
  - Hidden Size: `256` to `4096` (default: 768)
  - Attention Heads: `4` to `64` (default: 12)
  - Intermediate Size: `512` to `16384` (default: 3072)

- **Architecture Features** (Checkboxes):
  - ☑ Rotary Position Embeddings
  - ☑ Flash Attention
  - ☐ Gradient Checkpointing
  - ☐ MoE (Mixture of Experts)
  - ☐ Sparse Attention

#### Vector Composition
- **Composition Scale** (Slider): `0.0` to `1.0`
  - Default: `0.1`
  - Controls magnitude of architectural changes

- **Composition Distribution** (Radio):
  - Gaussian (default)
  - Uniform
  - Beta

- **Vector Dimension** (Number): `4` to `64`
  - Default: `16`
  - Higher = finer control, slower search

#### Search Strategy
- **Search Algorithm** (Radio):
  - NSGA-II (multi-objective, default)
  - Random search
  - Bayesian optimization
  - Evolutionary algorithm

- **Search Iterations** (Number): `10` to `200`
  - Default: `50`

- **Population Size** (Number): `4` to `32`
  - Default: `8`

#### Optimization Objectives
- **Primary Objectives** (Multi-select, weights sum to 1.0):
  - Performance (0-100%, default: 40%)
  - Memory efficiency (0-100%, default: 30%)
  - Inference speed (0-100%, default: 30%)

- **Constraints** (Number inputs):
  - Max parameters: Default 100M
  - Max memory (GB): Default 8
  - Min accuracy: Default 0.8

### Metrics & Visualizations

#### Search Progress
- **Pareto Front Evolution** (Animated scatter)
  - X: Performance
  - Y: Efficiency
  - Color: Generation number
  - Shows non-dominated solutions

- **Architecture Space Exploration** (Heatmap)
  - Rows: Architecture params
  - Columns: Search iterations
  - Color: Parameter values

- **Objective Convergence** (Multi-line chart)
  - One line per objective
  - Shows improvement over iterations

#### Architecture Analysis
- **Best Architectures Table**:
  - Rank | Config | Performance | Memory | Speed
  - Sortable by any column
  - Export to JSON/CSV

- **Layer Configuration Vizualizer** (Interactive diagram)
  - Shows layer types and connections
  - Hover for detailed params

- **Composition History** (Tree diagram)
  - Shows how architectures were combined
  - Traces lineage of best models

### Controls

#### Search Control
- **Start/Pause/Resume** (Standard controls)
  - Auto-save search state

- **Manual Architecture** (JSON editor)
  - Inject custom architecture into population
  - Validates before adding

#### Evaluation Settings
- **Timeout per Architecture** (Number): `30` to `600` sec
  - Default: `60`
  - Kills runaway evaluations

- **Memory Limit** (Number): `512` to `16384` MB
  - Default: `1024`
  - Prevents OOM crashes

#### Export & Deployment
- **Export Best N** (Number): `1` to `10`
  - Default: `3`
  - Exports top architectures

- **Deploy Selected** (Button)
  - Proceeds to training with chosen architecture

### Validation & Dependencies
- **Parameter Compatibility**: Validates architecture constraints
- **Resource Estimation**: Predicts memory/compute needs
- **Dependencies**: Can work standalone or after Phase 6

---

## Phase 8: Final Compression

### Purpose
Apply advanced compression stack (SeedLM + VPTQ + Hypercompression) for maximum size reduction.

### Key Configuration Parameters

#### SeedLM Settings
- **Bits per Weight** (Radio):
  - 2-bit (max compression)
  - 3-bit (balanced)
  - 4-bit (minimal quality loss, default)

- **Block Size** (Auto-calculated from bits):
  - 2-bit: C=16, P=2
  - 3-bit: C=12, P=4
  - 4-bit: C=8, P=3
  - Display only (not editable)

- **Max Seed Candidates** (Slider): `4` to `32`
  - Default: `16`
  - More candidates = better compression, slower

#### VPTQ Settings
- **Codebook Bits** (Radio):
  - 1-bit (2 entries)
  - 2-bit (4 entries, default)
  - 3-bit (8 entries)
  - 4-bit (16 entries)

- **Vector Dimension** (Number): `2` to `16`
  - Default: `4`
  - Higher = better quality, larger codebook

- **K-means Iterations** (Slider): `5` to `50`
  - Default: `10`
  - More iterations = better codebook, slower

#### Hypercompression
- **Enable Hypercompression** (Toggle): `on | off`
  - Default: `off`
  - Warning: Experimental, may reduce quality

- **Trajectory Sampling** (Slider): `100` to `10000`
  - Default: `1000`
  - Samples for ergodic trajectory

- **Compression Depth** (Number): `1` to `5`
  - Default: `2`
  - Recursive compression levels

#### Compression Pipeline
- **Technique Order** (Drag-and-drop list):
  - SeedLM → VPTQ → Hypercompression (default)
  - Reorder to customize pipeline

- **Layer Selection** (Multi-select):
  - All layers (default)
  - Embeddings only
  - Attention only
  - FFN only
  - Custom selection

### Metrics & Visualizations

#### Compression Results
- **Size Reduction Cascade** (Waterfall chart)
  - Original size
  - After SeedLM (-X%)
  - After VPTQ (-Y%)
  - After Hypercompression (-Z%)
  - Final size

- **Compression Quality** (Multi-metric table):
  | Metric | Original | Compressed | Delta |
  |--------|----------|------------|-------|
  | Perplexity | X | Y | +Z% |
  | Accuracy | A | B | -C% |
  | Inference Speed | M | N | +P% |

- **Technique Contribution** (Pie chart)
  - % size reduction per technique
  - Shows which method contributed most

#### Quality Analysis
- **Layer-wise Impact** (Heatmap)
  - Rows: Layers
  - Columns: Metrics (perplexity, accuracy, etc.)
  - Color: Degradation level

- **Decompression Fidelity** (Histogram)
  - Weight reconstruction error distribution
  - Per technique overlay

#### Performance Benchmarks
- **Inference Latency** (Bar chart)
  - Original vs Compressed
  - Per batch size

- **Memory Footprint** (Grouped bar)
  - Disk size
  - RAM usage
  - VRAM usage

### Controls

#### Compression Strategy (Tabs)
- **Aggressive** (Max compression)
  - SeedLM 2-bit + VPTQ 1-bit + Hyper
  - Targets <5x original size

- **Balanced** (Default)
  - SeedLM 4-bit + VPTQ 2-bit
  - Targets 3-5x reduction

- **Conservative** (Min quality loss)
  - SeedLM 4-bit + VPTQ 3-bit
  - Targets 2-3x reduction

- **Custom** (User-defined)
  - Manual parameter selection

#### Validation & Rollback
- **Quality Gate** (Threshold inputs):
  - Max Perplexity Increase: `±5%` (default)
  - Max Accuracy Drop: `±2%` (default)
  - Auto-rollback if exceeded

- **A/B Testing** (Toggle + params):
  - Compare compressed vs original
  - Statistical significance test
  - Required p-value: 0.05 (default)

#### Export & Deployment
- **Export Format** (Multi-select):
  - ☑ Compressed checkpoint (.pt)
  - ☑ Deployment-ready (ONNX + weights)
  - ☐ Decompression code
  - ☐ Benchmark report (JSON/PDF)

- **Target Platform** (Radio):
  - Server (full precision fallback)
  - Edge device (minimal dependencies)
  - Mobile (ARM-optimized)

### Validation & Dependencies
- **Decompression Test**: Validates compressed model can be loaded
- **Quality Verification**: Runs benchmark suite automatically
- **Memory Estimation**: Predicts deployment footprint
- **Dependencies**: Requires any trained model (Phase 5, 6, or 7)

---

## Cross-Phase Dependencies

### Dependency Graph
```
Phase 1 (Cognate) → Phase 2 (EvoMerge) → Phase 3 (Quiet-STaR)
                                              ↓
                                         Phase 5 (Training) → Phase 6 (Persona)
                                              ↓                     ↓
Phase 4 (BitNet) ←---------------------------+                     ↓
                                                                    ↓
Phase 7 (ADAS) ←---------------------------------------------------|
                                                                    ↓
Phase 8 (Compression) ←--------------------------------------------|
```

### Optional Paths
- Can skip Phase 1-2 and start with external model at Phase 3
- Can skip Phase 3 and go Phase 2 → Phase 5
- Can apply Phase 4 (compression) at multiple stages
- Phase 7 (ADAS) can be run standalone for architecture search
- Phase 8 can compress output from Phase 5, 6, or 7

### State Management
- Each phase saves checkpoint with metadata
- Checkpoints include phase config for reproducibility
- Automatic version detection for compatibility
- Rollback capability to any previous phase

---

## Global Controls

### Session Management
- **Save Session** (Button)
  - Saves entire pipeline state
  - Named sessions with timestamps

- **Load Session** (Dropdown + button)
  - Resume any saved session
  - Shows compatible phases

- **Export Pipeline Config** (Button)
  - YAML/JSON export of all settings
  - Reproducibility manifest

### Resource Management
- **GPU Selection** (Dropdown)
  - Lists available GPUs
  - Shows memory and utilization

- **CPU Threads** (Slider): `1` to `max_cores`
  - Default: `max_cores - 2`

- **Memory Limit** (Slider)
  - RAM limit for data processing
  - Auto-adjusts batch sizes if exceeded

### Monitoring Dashboard
- **Real-time System Metrics** (Always visible):
  - GPU temp, usage, memory
  - CPU usage, RAM
  - Disk I/O
  - Network (if distributed)

- **Cost Tracking** (Optional):
  - Compute cost estimation
  - Cloud resource usage
  - Budget alerts

### Alert System
- **Alert Categories**:
  - 🔴 Critical: OOM, crashes, NaN losses
  - 🟡 Warning: High resource usage, slow progress
  - 🟢 Info: Checkpoints saved, phase completed

- **Notification Channels**:
  - In-app toasts
  - Email (configurable)
  - Slack/Discord webhook (optional)

---

## Implementation Recommendations

### Technology Stack
- **Frontend Framework**: React or Vue.js
  - Component library: Material-UI or Ant Design
  - Charting: Plotly.js or D3.js for visualizations

- **Backend API**: FastAPI (Python)
  - WebSocket for real-time metrics
  - REST API for control commands

- **State Management**: Redux or Vuex
  - Persist state to localStorage
  - Sync with backend via WebSocket

### Default Value Strategy
- Conservative defaults for safety
- "Quick start" presets for common workflows
- Advanced mode reveals all parameters
- Beginner mode shows only essentials

### Validation Approach
- Client-side validation for immediate feedback
- Server-side validation before execution
- Dependency checks before phase start
- Resource estimation with warnings

### User Experience
- Wizard mode for beginners (guided steps)
- Expert mode with all controls visible
- Contextual help (tooltips, docs links)
- Keyboard shortcuts for power users
- Dark/Light theme support

---

## Conclusion

This control interface design provides comprehensive access to all 8 phases of the Agent Forge pipeline while maintaining usability through:

1. **Sensible defaults** - Start quickly without configuration
2. **Progressive disclosure** - Show complexity only when needed
3. **Real-time feedback** - Metrics and visualizations at every step
4. **Safety mechanisms** - Validation, alerts, rollback capabilities
5. **Flexibility** - Support for both guided and expert workflows

The design balances power and simplicity, enabling both ML researchers to fine-tune every parameter and practitioners to run end-to-end pipelines with minimal configuration.