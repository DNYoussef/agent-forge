# Agent Forge API Reference

## Core APIs

### Phase Controller

#### PhaseController (Abstract Base Class)

**Location**: `agent_forge/core/phase_controller.py`

```python
class PhaseController(ABC):
    """Abstract base class for Agent Forge phase controllers"""

    def __init__(self, config: Any)
    async def run(self, model: nn.Module) -> PhaseResult
    def validate_input_model(self, model: nn.Module) -> bool
    def create_success_result(self, model, metrics, artifacts, duration) -> PhaseResult
    def create_failure_result(self, model, error, duration) -> PhaseResult
```

**Methods**:

- **`__init__(self, config: Any)`**
  - Initialize phase controller with configuration
  - Sets up logging for the phase

- **`async run(self, model: nn.Module) -> PhaseResult`** *(Abstract)*
  - Execute the phase processing
  - **Parameters**: `model` - Input model from previous phase
  - **Returns**: PhaseResult with processed model and metrics

- **`validate_input_model(self, model: nn.Module) -> bool`**
  - Validate input model before processing
  - **Parameters**: `model` - Model to validate
  - **Returns**: True if model is valid for this phase

- **`create_success_result(...)`**
  - Create a successful phase result
  - **Parameters**:
    - `model` - Processed model
    - `metrics` - Performance metrics dictionary
    - `artifacts` - Additional artifacts (optional)
    - `duration` - Processing duration in seconds
  - **Returns**: PhaseResult object

#### PhaseResult (Data Class)

```python
@dataclass
class PhaseResult:
    success: bool
    model: nn.Module
    phase_name: str | None = None
    metrics: dict[str, Any] | None = None
    duration_seconds: float = 0.0
    artifacts: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    error: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
```

#### PhaseOrchestrator

```python
class PhaseOrchestrator:
    """Orchestrates execution of multiple phases with model passing validation"""

    def __init__(self)
    async def run_phase_sequence(self, phases: list[tuple[str, PhaseController]],
                                initial_model: nn.Module) -> list[PhaseResult]
    def validate_phase_compatibility(self, phases: list[tuple[str, PhaseController]]) -> bool
```

**Methods**:

- **`run_phase_sequence(phases, initial_model)`**
  - Run a sequence of phases with validation
  - **Parameters**:
    - `phases` - List of (phase_name, controller) tuples
    - `initial_model` - Initial model to start pipeline
  - **Returns**: List of PhaseResult objects

- **`validate_phase_compatibility(phases)`**
  - Validate phase sequence compatibility
  - **Parameters**: `phases` - List of (phase_name, controller) tuples
  - **Returns**: True if phases are compatible

### Model Passing Validator

```python
class ModelPassingValidator:
    """Validates model compatibility between phases"""

    @staticmethod
    def validate_model_transition(source_phase: str, target_phase: str,
                                model: nn.Module) -> tuple[bool, str]
```

**Supported Transitions**:
- EvoMergePhase → QuietSTaRPhase
- QuietSTaRPhase → BitNetCompressionPhase
- BitNetCompressionPhase → ForgeTrainingPhase
- ForgeTrainingPhase → ToolPersonaBakingPhase
- ToolPersonaBakingPhase → ADASPhase
- ADASPhase → FinalCompressionPhase

## Phase 2: EvoMerge APIs

### EvoMergePhase

**Location**: `phases/phase2_evomerge/evomerge.py`

```python
class EvoMergePhase(PhaseController):
    def __init__(self, config: EvoMergeConfig)
    async def run(self, model: nn.Module) -> PhaseResult
    async def execute(self, input_model_path: str) -> PhaseResult
```

### EvolutionaryEngine

**Location**: `src/evomerge/core/EvolutionaryEngine.py`

```python
class EvolutionaryEngine:
    def __init__(self, config: EvolutionConfig)
    def evolve_population(self, initial_population: List[Individual]) -> EvolutionResult
    def evaluate_fitness(self, individual: Individual) -> float
    def select_parents(self, population: List[Individual]) -> List[Individual]
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual
    def mutate(self, individual: Individual) -> Individual
    def check_convergence(self) -> bool
```

### Merge Operators

#### SLERPOperator

**Location**: `src/evomerge/operators/slerp_operator.py`

```python
class SLERPOperator:
    def __init__(self, config: SLERPConfig = None)
    def merge(self, models: List[nn.Module], weights: List[float] = None) -> nn.Module
    def interpolate_weights(self, weight1: torch.Tensor, weight2: torch.Tensor,
                          t: float) -> torch.Tensor
```

**Methods**:

- **`merge(models, weights)`**
  - Perform spherical linear interpolation between models
  - **Parameters**:
    - `models` - List of models to merge
    - `weights` - Interpolation weights (optional, defaults to equal)
  - **Returns**: Merged model

#### TIESOperator

**Location**: `src/evomerge/operators/ties_operator.py`

```python
class TIESOperator:
    def __init__(self, config: TIESConfig = None)
    def merge(self, models: List[nn.Module], task_config: TaskConfig) -> nn.Module
    def compute_task_vectors(self, models: List[nn.Module]) -> List[torch.Tensor]
    def resolve_conflicts(self, task_vectors: List[torch.Tensor]) -> torch.Tensor
```

#### DAREOperator

**Location**: `src/evomerge/operators/dare_operator.py`

```python
class DAREOperator:
    def __init__(self, drop_rate: float = 0.1, rescale_factor: float = 1.0)
    def merge(self, models: List[nn.Module]) -> nn.Module
    def apply_dropout_rescale(self, tensor: torch.Tensor) -> torch.Tensor
```

### Configuration Classes

#### EvoMergeConfig

```python
@dataclass
class EvoMergeConfig:
    population_size: int = 10
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 2
    selection_strategy: SelectionStrategy = SelectionStrategy.TOURNAMENT
    tournament_size: int = 3
    merge_techniques: List[str] = field(default_factory=lambda: ['slerp', 'ties', 'dare'])
    fitness_functions: List[str] = field(default_factory=lambda: ['accuracy', 'efficiency'])
    convergence_threshold: float = 0.001
    max_generations: int = 100
```

## Phase 3: Quiet-STaR APIs

### QuietSTaR (Main Class)

**Location**: `phases/phase3_quietstar/quietstar.py`

```python
class QuietSTaR(nn.Module):
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium",
                 config: ThoughtConfig = None)
    def forward(self, input_text: str, return_thoughts: bool = True) -> Dict[str, Any]
    async def async_forward(self, input_texts: List[str],
                           return_thoughts: bool = True) -> List[Dict[str, Any]]
    def get_thought_text(self, thought_ids: torch.Tensor) -> str
    def reset_metrics(self)
    def get_statistics(self) -> Dict[str, float]
```

**Methods**:

- **`forward(input_text, return_thoughts)`**
  - Main processing with thought generation
  - **Parameters**:
    - `input_text` - Input text to process
    - `return_thoughts` - Whether to return generated thoughts
  - **Returns**: Dictionary with enhanced outputs and metadata

- **`async_forward(input_texts, return_thoughts)`**
  - Asynchronous processing for multiple inputs
  - **Parameters**:
    - `input_texts` - List of input texts
    - `return_thoughts` - Whether to return thoughts
  - **Returns**: List of result dictionaries

### ThoughtGenerator

```python
class ThoughtGenerator(nn.Module):
    def __init__(self, model, tokenizer, config: ThoughtConfig)
    def forward(self, input_ids: torch.Tensor,
               attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]
```

**Returns**:
```python
{
    'thoughts': List[torch.Tensor],           # Generated thoughts
    'thought_scores': torch.Tensor,           # Quality scores
    'thought_features': torch.Tensor,         # Hidden representations
    'input_ids': torch.Tensor,                # Original input
    'attention_mask': torch.Tensor            # Attention mask
}
```

### ThoughtInjectionSystem

```python
class ThoughtInjectionSystem(nn.Module):
    def __init__(self, model, config: ThoughtConfig)
    def inject_thoughts(self, base_hidden: torch.Tensor,
                       thought_hidden: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor
    def process_batch(self, input_ids: torch.Tensor,
                     thoughts: List[torch.Tensor],
                     attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]
```

### CoherenceValidator

```python
class CoherenceValidator:
    def __init__(self, model, tokenizer, config: ThoughtConfig)
    def validate_thoughts(self, input_ids: torch.Tensor,
                         thoughts: List[torch.Tensor],
                         attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]
```

**Returns**:
```python
{
    'coherence_scores': torch.Tensor,         # Overall coherence scores
    'metric_scores': Dict[str, torch.Tensor], # Individual metric scores
    'valid_thoughts': torch.Tensor,           # Boolean validity mask
    'filtered_thoughts': List[torch.Tensor]   # Thoughts passing threshold
}
```

### Configuration Classes

#### ThoughtConfig

```python
@dataclass
class ThoughtConfig:
    num_thoughts: int = 4                     # Number of parallel thoughts
    thought_length: int = 32                  # Tokens per thought
    coherence_threshold: float = 0.6          # Minimum coherence score
    temperature: float = 0.8                  # Generation temperature
    top_p: float = 0.9                       # Top-p sampling
    special_tokens: Dict[str, str] = None     # Special token definitions
```

## Phase 4: BitNet Compression APIs

### BitNetCompressionPhase

**Location**: `phases/bitnet_compression.py`

```python
class BitNetCompressionPhase(PhaseController):
    def __init__(self, config: BitNetCompressionConfig)
    async def run(self, model: nn.Module) -> PhaseResult
    async def execute(self, input_model_path: str) -> PhaseResult
```

### BitNetQuantizer

```python
class BitNetQuantizer:
    def __init__(self, config: BitNetCompressionConfig)
    def quantize_tensor(self, tensor: torch.Tensor,
                       preserve_precision: bool = False) -> Dict[str, Any]
    def dequantize_tensor(self, quantized_data: Dict[str, Any]) -> torch.Tensor
```

**Quantization Output**:
```python
{
    'weights': np.ndarray,                    # Quantized weights
    'scale': np.ndarray,                      # Scaling factors
    'quantization_type': str,                 # 'bitnet_1.58' or 'none'
    'is_quantized': bool,                     # Quantization status
    'shape': tuple,                           # Original tensor shape
    'dtype': str,                             # Original data type
    'sparsity': float                         # Sparsity ratio (if quantized)
}
```

### BitNetCompressedModel

```python
class BitNetCompressedModel(nn.Module):
    def __init__(self, original_model: nn.Module, quantizer: BitNetQuantizer)
    def forward(self, *args, **kwargs)
    def get_compression_stats(self) -> Dict[str, Any]
```

**Compression Stats**:
```python
{
    'original_size_mb': float,                # Original model size in MB
    'compressed_size_mb': float,              # Compressed size in MB
    'compression_ratio': float,               # Compression ratio
    'layers_compressed': int,                 # Number of compressed layers
    'quantization_stats': Dict[str, Any]      # Detailed quantization statistics
}
```

### BitNetCompressionPipeline

```python
class BitNetCompressionPipeline:
    def __init__(self, config: BitNetCompressionConfig)
    async def compress_model(self, model_path: str) -> Dict[str, Any]
```

**Compression Results**:
```python
{
    'success': bool,                          # Compression success status
    'model_path': str,                        # Output model path
    'compression_ratio': float,               # Achieved compression ratio
    'original_size_mb': float,                # Original model size
    'compressed_size_mb': float,              # Compressed model size
    'layers_compressed': int,                 # Number of layers compressed
    'quantization_stats': Dict[str, Any],     # Quantization statistics
    'pre_compression_metrics': Dict[str, float],   # Pre-compression performance
    'post_compression_metrics': Dict[str, float],  # Post-compression performance
    'final_metrics': Dict[str, float],        # Final performance after fine-tuning
    'accuracy_preserved': bool                # Whether accuracy was preserved
}
```

### Configuration Classes

#### BitNetCompressionConfig

```python
@dataclass
class BitNetCompressionConfig:
    # Model configuration
    model_path: str = ""
    output_path: str = ""
    tokenizer_path: str | None = None

    # BitNet quantization settings
    quantization_bits: float = 1.58
    preserve_embedding_precision: bool = True
    preserve_output_precision: bool = True
    sparsity_threshold: float = 0.1

    # Calibration settings
    calibration_samples: int = 1000
    calibration_dataset: str = "openwebtext"
    calibration_batch_size: int = 4
    calibration_sequence_length: int = 512

    # Fine-tuning configuration
    enable_fine_tuning: bool = True
    fine_tune_epochs: int = 2
    fine_tune_lr: float = 1e-5
    warmup_steps: int = 50
    weight_decay: float = 0.01

    # Grokfast integration
    enable_grokfast: bool = True
    grokfast_ema_alpha: float = 0.98
    grokfast_lambda: float = 2.0

    # Quality targets
    target_compression_ratio: float = 8.0
    max_accuracy_drop: float = 0.05

    # System configuration
    device: str = "auto"
    mixed_precision: bool = True
    seed: int = 42
```

## Utility APIs

### CalibrationDataset

```python
class CalibrationDataset(Dataset):
    def __init__(self, dataset_name: str, num_samples: int,
                 tokenizer, max_length: int = 512)
    def __len__(self) -> int
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]
```

**Supported Datasets**:
- `"openwebtext"` - OpenWebText corpus
- `"c4"` - C4 dataset
- `"wikitext"` - WikiText-2-raw-v1

### Factory Functions

#### create_bitnet_compression_phase

```python
def create_bitnet_compression_phase(
    model_path: str = "",
    output_path: str = "",
    target_compression_ratio: float = 8.0,
    calibration_samples: int = 1000,
    enable_fine_tuning: bool = True,
    enable_grokfast: bool = True,
    device: str = "auto",
    **kwargs
) -> BitNetCompressionPhase
```

## Error Handling

### Common Exceptions

- **`ValidationError`**: Raised when model validation fails
- **`PhaseExecutionError`**: Raised when phase execution encounters errors
- **`ModelCompatibilityError`**: Raised when models are incompatible between phases
- **`CompressionError`**: Raised when compression fails to meet quality targets
- **`CoherenceValidationError`**: Raised when thought coherence validation fails

### Error Response Format

```python
{
    'success': False,
    'error': str,                             # Error message
    'error_type': str,                        # Error type/category
    'phase_name': str,                        # Phase where error occurred
    'timestamp': datetime,                    # Error timestamp
    'debug_info': Dict[str, Any]              # Additional debug information
}
```

## Performance Monitoring

### Metrics Collection

All phases collect standard performance metrics:

```python
{
    'duration_seconds': float,                # Phase execution time
    'memory_usage_mb': float,                 # Peak memory usage
    'gpu_memory_usage_mb': float,             # GPU memory usage (if available)
    'cpu_utilization': float,                 # CPU utilization percentage
    'throughput': float,                      # Processing throughput
    'quality_metrics': Dict[str, float]       # Phase-specific quality metrics
}
```

### Logging Integration

All APIs integrate with Python's logging framework:

```python
import logging

# Configure logging for Agent Forge
logging.getLogger('agent_forge').setLevel(logging.INFO)
logging.getLogger('evomerge').setLevel(logging.DEBUG)
logging.getLogger('quietstar').setLevel(logging.INFO)
logging.getLogger('bitnet').setLevel(logging.INFO)
```

This API reference provides comprehensive documentation for all operational components of Agent Forge, enabling developers to integrate and extend the system effectively.