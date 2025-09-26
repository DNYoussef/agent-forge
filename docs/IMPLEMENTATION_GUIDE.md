# Agent Forge Implementation Guide

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install torch transformers datasets numpy tqdm
pip install fastapi uvicorn websockets
npm install # For web dashboard
```

### Basic Pipeline Execution

```python
from agent_forge.core.unified_pipeline import UnifiedPipeline

# Initialize pipeline
pipeline = UnifiedPipeline()

# Run full 8-phase pipeline
result = await pipeline.run_complete_pipeline(
    base_models=["microsoft/DialoGPT-medium"],
    output_dir="./agent_output"
)
```

## Phase-by-Phase Implementation

### Phase 2: EvoMerge - Evolutionary Model Optimization

#### Basic Usage

```python
from phases.phase2_evomerge.evomerge import EvoMergePhase
from phases.phase2_evomerge.config import EvoMergeConfig

# Configure evolution parameters
config = EvoMergeConfig(
    population_size=10,
    generations=50,
    mutation_rate=0.1,
    crossover_rate=0.8,
    elite_size=2
)

# Create phase
evomerge_phase = EvoMergePhase(config)

# Execute evolution
result = await evomerge_phase.execute("./input_model_path")
```

#### Advanced Configuration

```python
# Real evolutionary engine configuration
from src.evomerge.core.EvolutionaryEngine import EvolutionaryEngine, SelectionStrategy

config = EvoMergeConfig(
    # Population parameters
    population_size=20,
    generations=100,

    # Genetic operations
    selection_strategy=SelectionStrategy.TOURNAMENT,
    tournament_size=3,
    crossover_rate=0.8,
    mutation_rate=0.1,
    elite_size=2,

    # Merge operators
    merge_techniques=['slerp', 'ties', 'dare'],
    merge_weights=[0.4, 0.4, 0.2],

    # Fitness evaluation
    fitness_functions=['accuracy', 'efficiency', 'diversity'],
    fitness_weights=[0.5, 0.3, 0.2],

    # Convergence criteria
    max_generations=100,
    convergence_threshold=0.001,
    patience=10
)
```

#### Merge Operator Usage

```python
# SLERP (Spherical Linear Interpolation)
from src.evomerge.operators.slerp_operator import SLERPOperator

slerp = SLERPOperator()
merged_model = slerp.merge([model1, model2], weights=[0.6, 0.4])

# TIES (Task-wise Internal Ensemble Selection)
from src.evomerge.operators.ties_operator import TIESOperator, TaskConfig

ties = TIESOperator()
task_config = TaskConfig(
    tasks=['language_modeling', 'question_answering'],
    task_weights=[0.7, 0.3]
)
merged_model = ties.merge([model1, model2], task_config=task_config)

# DARE (Drop And REscale)
from src.evomerge.operators.dare_operator import DAREOperator

dare = DAREOperator(drop_rate=0.1, rescale_factor=1.1)
merged_model = dare.merge([model1, model2])
```

### Phase 3: Quiet-STaR - Reasoning Enhancement

#### Basic Implementation

```python
from phases.phase3_quietstar.quietstar import QuietSTaR, ThoughtConfig

# Configure thought generation
config = ThoughtConfig(
    num_thoughts=4,              # Parallel thoughts
    thought_length=32,           # Tokens per thought
    coherence_threshold=0.6,     # Minimum coherence score
    temperature=0.8,             # Generation temperature
    top_p=0.9                    # Top-p sampling
)

# Initialize Quiet-STaR
quietstar = QuietSTaR(
    model_name="microsoft/DialoGPT-medium",
    config=config
)

# Process input
result = quietstar("Explain the concept of neural networks step by step.")
```

#### Advanced Thought Processing

```python
# Custom thought validation
config = ThoughtConfig(
    num_thoughts=6,
    thought_length=48,
    coherence_threshold=0.7,
    special_tokens={
        'start_thought': '<|startofthought|>',
        'end_thought': '<|endofthought|>',
        'thought_sep': '<|thoughtsep|>'
    }
)

# Process multiple inputs asynchronously
inputs = [
    "What are the key principles of machine learning?",
    "How do neural networks learn from data?",
    "Explain backpropagation in simple terms."
]

results = await quietstar.async_forward(inputs, return_thoughts=True)

# Analyze results
for i, result in enumerate(results):
    print(f"Input {i+1}:")
    print(f"  Processing time: {result['processing_time']:.3f}s")
    print(f"  Valid thoughts: {result['valid_thoughts_mask'].sum()}")
    print(f"  Coherence score: {result['coherence_scores'].mean():.3f}")

    # Extract thought text
    if 'raw_thoughts' in result:
        for j, thought in enumerate(result['raw_thoughts']):
            thought_text = quietstar.get_thought_text(thought)
            print(f"  Thought {j+1}: {thought_text}")
```

#### Component-Level Usage

```python
from phases.phase3_quietstar.quietstar import (
    ThoughtGenerator, ThoughtInjectionSystem, CoherenceValidator
)

# Initialize components separately
thought_generator = ThoughtGenerator(model, tokenizer, config)
injection_system = ThoughtInjectionSystem(model, config)
coherence_validator = CoherenceValidator(model, tokenizer, config)

# Generate thoughts
thought_output = thought_generator(input_ids, attention_mask)

# Validate coherence
validation_results = coherence_validator.validate_thoughts(
    input_ids, thought_output['thoughts'], attention_mask
)

# Inject valid thoughts
if validation_results['filtered_thoughts']:
    injection_results = injection_system.process_batch(
        input_ids, validation_results['filtered_thoughts'], attention_mask
    )
```

### Phase 4: BitNet Compression

#### Basic Compression

```python
from phases.bitnet_compression import BitNetCompressionPhase, BitNetCompressionConfig

# Configure compression
config = BitNetCompressionConfig(
    model_path="./input_model",
    output_path="./compressed_model",
    quantization_bits=1.58,
    target_compression_ratio=8.0,
    calibration_samples=1000,
    enable_fine_tuning=True
)

# Create and run compression phase
compression_phase = BitNetCompressionPhase(config)
result = await compression_phase.execute("./input_model")

print(f"Compression ratio: {result.metrics['compression_ratio']:.2f}x")
print(f"Accuracy preserved: {result.metrics['accuracy_preserved']}")
```

#### Advanced Compression Configuration

```python
config = BitNetCompressionConfig(
    # Model configuration
    model_path="./input_model",
    output_path="./compressed_model",
    tokenizer_path=None,  # Auto-detect

    # BitNet quantization
    quantization_bits=1.58,
    preserve_embedding_precision=True,
    preserve_output_precision=True,
    sparsity_threshold=0.1,

    # Calibration
    calibration_samples=2000,
    calibration_dataset="openwebtext",  # or "c4", "wikitext"
    calibration_batch_size=4,
    calibration_sequence_length=512,

    # Fine-tuning
    enable_fine_tuning=True,
    fine_tune_epochs=3,
    fine_tune_lr=1e-5,
    warmup_steps=100,
    weight_decay=0.01,

    # Grokfast optimization
    enable_grokfast=True,
    grokfast_ema_alpha=0.98,
    grokfast_lambda=2.0,

    # Quality targets
    target_compression_ratio=8.0,
    max_accuracy_drop=0.05,

    # Hardware
    device="auto",
    mixed_precision=True,
    seed=42
)
```

#### Custom Quantization

```python
from phases.bitnet_compression import BitNetQuantizer, BitNetCompressedModel

# Create quantizer
quantizer = BitNetQuantizer(config)

# Quantize specific tensors
tensor = torch.randn(1024, 768)
quantized_data = quantizer.quantize_tensor(tensor, preserve_precision=False)

print(f"Original shape: {tensor.shape}")
print(f"Quantized: {quantized_data['is_quantized']}")
print(f"Sparsity: {quantized_data['sparsity']:.3f}")

# Dequantize
dequantized = quantizer.dequantize_tensor(quantized_data)
print(f"Reconstruction error: {torch.norm(tensor - dequantized).item():.6f}")

# Create compressed model wrapper
compressed_model = BitNetCompressedModel(original_model, quantizer)
stats = compressed_model.get_compression_stats()

print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
print(f"Original size: {stats['original_size_mb']:.2f} MB")
print(f"Compressed size: {stats['compressed_size_mb']:.2f} MB")
```

## Phase Orchestration

### Sequential Execution

```python
from agent_forge.core.phase_controller import PhaseOrchestrator

# Initialize orchestrator
orchestrator = PhaseOrchestrator()

# Define phase sequence
phases = [
    ("EvoMergePhase", evomerge_phase),
    ("QuietSTaRPhase", quietstar_phase),
    ("BitNetCompressionPhase", compression_phase)
]

# Run sequence with validation
results = await orchestrator.run_phase_sequence(phases, initial_model)

# Analyze results
for i, result in enumerate(results):
    phase_name = phases[i][0]
    print(f"{phase_name}: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
    if result.success:
        print(f"  Duration: {result.duration_seconds:.2f}s")
        print(f"  Metrics: {result.metrics}")
    else:
        print(f"  Error: {result.error}")
```

### Custom Phase Implementation

```python
from agent_forge.core.phase_controller import PhaseController, PhaseResult
import time

class CustomPhase(PhaseController):
    def __init__(self, config):
        super().__init__(config)
        self.phase_name = "Custom Processing"

    async def run(self, model: nn.Module) -> PhaseResult:
        # Validate input
        if not self.validate_input_model(model):
            return self.create_failure_result(model, "Input validation failed")

        start_time = time.time()

        try:
            # Custom processing logic
            processed_model = self.process_model(model)

            # Calculate metrics
            metrics = {
                "processing_quality": 0.95,
                "improvements": ["stability", "accuracy"]
            }

            duration = time.time() - start_time

            return self.create_success_result(
                model=processed_model,
                metrics=metrics,
                artifacts={"logs": "processing_complete"},
                duration=duration
            )

        except Exception as e:
            duration = time.time() - start_time
            return self.create_failure_result(model, str(e), duration)

    def process_model(self, model):
        # Implement custom processing
        # This is where your specific logic goes
        return model
```

## Performance Optimization

### Memory Management

```python
import torch

# Enable memory efficient attention
torch.backends.cuda.enable_flash_sdp(True)

# Use gradient checkpointing for large models
model.gradient_checkpointing_enable()

# Clear cache between phases
torch.cuda.empty_cache()

# Monitor memory usage
def log_memory_usage(phase_name):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{phase_name}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

### Parallel Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def parallel_phase_execution(phases, models):
    """Execute multiple phases in parallel on different models"""

    async def run_phase(phase, model):
        return await phase.run(model)

    # Create tasks for parallel execution
    tasks = []
    for phase, model in zip(phases, models):
        task = asyncio.create_task(run_phase(phase, model))
        tasks.append(task)

    # Wait for all phases to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return results
```

### Batch Processing

```python
def batch_process_models(phase, models, batch_size=4):
    """Process multiple models in batches"""
    results = []

    for i in range(0, len(models), batch_size):
        batch = models[i:i + batch_size]
        batch_results = []

        for model in batch:
            result = await phase.run(model)
            batch_results.append(result)

        results.extend(batch_results)

        # Memory cleanup between batches
        torch.cuda.empty_cache()

    return results
```

## Configuration Management

### Environment Configuration

```python
import os
from pathlib import Path

# Set up directories
base_dir = Path("./agent_forge_workspace")
base_dir.mkdir(exist_ok=True)

config_dirs = {
    "models": base_dir / "models",
    "outputs": base_dir / "outputs",
    "cache": base_dir / "cache",
    "logs": base_dir / "logs"
}

for dir_path in config_dirs.values():
    dir_path.mkdir(exist_ok=True)

# Environment variables
os.environ["TRANSFORMERS_CACHE"] = str(config_dirs["cache"])
os.environ["HF_HOME"] = str(config_dirs["cache"])
```

### Configuration Files

```yaml
# config/pipeline_config.yaml
pipeline:
  name: "agent_forge_v1"
  description: "8-phase AI agent creation pipeline"

phases:
  evomerge:
    enabled: true
    population_size: 10
    generations: 50

  quietstar:
    enabled: true
    num_thoughts: 4
    coherence_threshold: 0.6

  bitnet:
    enabled: true
    compression_ratio: 8.0
    calibration_samples: 1000

hardware:
  device: "auto"
  mixed_precision: true
  memory_efficient: true

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Testing and Validation

### Unit Testing

```python
import unittest
import torch

class TestBitNetCompression(unittest.TestCase):
    def setUp(self):
        self.config = BitNetCompressionConfig(
            model_path="test_model",
            output_path="test_output",
            calibration_samples=100
        )
        self.quantizer = BitNetQuantizer(self.config)

    def test_quantization(self):
        tensor = torch.randn(64, 128)
        quantized = self.quantizer.quantize_tensor(tensor)

        self.assertTrue(quantized['is_quantized'])
        self.assertEqual(quantized['quantization_type'], 'bitnet_1.58')

        # Test dequantization
        dequantized = self.quantizer.dequantize_tensor(quantized)
        self.assertEqual(dequantized.shape, tensor.shape)

    def test_compression_ratio(self):
        model = torch.nn.Linear(128, 64)
        compressed = BitNetCompressedModel(model, self.quantizer)
        stats = compressed.get_compression_stats()

        self.assertGreater(stats['compression_ratio'], 1.0)

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

```python
async def test_full_pipeline():
    """Test complete pipeline execution"""

    # Create test model
    test_model = torch.nn.Sequential(
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32)
    )

    # Configure phases
    evomerge_config = EvoMergeConfig(population_size=4, generations=5)
    quietstar_config = ThoughtConfig(num_thoughts=2, thought_length=16)
    bitnet_config = BitNetCompressionConfig(calibration_samples=50)

    # Create phases
    phases = [
        ("EvoMerge", EvoMergePhase(evomerge_config)),
        ("QuietSTaR", QuietSTaRPhase(quietstar_config)),
        ("BitNet", BitNetCompressionPhase(bitnet_config))
    ]

    # Run pipeline
    orchestrator = PhaseOrchestrator()
    results = await orchestrator.run_phase_sequence(phases, test_model)

    # Validate results
    assert all(result.success for result in results), "Pipeline execution failed"

    print("✅ Full pipeline test passed!")

# Run test
asyncio.run(test_full_pipeline())
```

## Monitoring and Debugging

### Logging Configuration

```python
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_forge.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Phase-specific loggers
evomerge_logger = logging.getLogger('evomerge')
quietstar_logger = logging.getLogger('quietstar')
bitnet_logger = logging.getLogger('bitnet')
```

### Performance Monitoring

```python
import time
import psutil
import torch

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}

    def start_phase(self, phase_name):
        self.metrics[phase_name] = {
            'start_time': time.time(),
            'start_memory': psutil.virtual_memory().used / 1024**3,
            'start_gpu_memory': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        }

    def end_phase(self, phase_name):
        if phase_name in self.metrics:
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / 1024**3
            end_gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

            phase_metrics = self.metrics[phase_name]
            phase_metrics.update({
                'duration': end_time - phase_metrics['start_time'],
                'memory_delta': end_memory - phase_metrics['start_memory'],
                'gpu_memory_delta': end_gpu_memory - phase_metrics['start_gpu_memory']
            })

    def get_report(self):
        return self.metrics

# Usage
monitor = PerformanceMonitor()

# In phase execution
monitor.start_phase("EvoMerge")
# ... phase execution ...
monitor.end_phase("EvoMerge")

# Get performance report
report = monitor.get_report()
print(f"EvoMerge duration: {report['EvoMerge']['duration']:.2f}s")
```

This implementation guide provides comprehensive examples for using Agent Forge's real implementations, covering all operational phases with practical code examples and best practices.