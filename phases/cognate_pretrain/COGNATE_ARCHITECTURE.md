# Cognate Architecture: Tiny Titans + HRM Training

## What Cognate Actually Is

**Cognate = Tiny Titans (25M) with Combined HRM + Titans Training Process**

### Architecture (From Titans Paper)
- **Base**: Titans neural memory architecture
- **Size**: 25M parameters (tiny version)
- **Memory**: Surprise-based neural memory module
- **Core Components**:
  - Neural memory with surprise gating
  - Memory states M_t and surprise states S_t
  - Gate network computing (α, η, θ)
  - 1D depthwise separable convolution

### Training Process (Combined HRM + Titans)

#### From HRM Paper:
- **No intermediate supervision** - train only on final output
- **Two-timescale processing**:
  - Slow abstract planning (high-level)
  - Fast detailed computation (low-level)
- **Minimal data requirement** - works with 1000 samples
- **Deep recurrent computation** without explicit CoT

#### From Titans Paper:
- **Surprise-based memory updates**:
  - S_t = η_t * S_{t-1} - θ_t * ∇ℓ(M_{t-1}; x_t)
- **Adaptive forgetting**:
  - M_t = (1 - α_t) * M_{t-1} + S_t
- **Test-time memorization** - learns to memorize during inference
- **Gradient-based surprise** - uses loss gradients as surprise signal

### Implementation Details

```python
# Our Cognate is effectively:
class Cognate:
    def __init__(self):
        # Titans architecture (25M params)
        self.model = TinyTitansModel(
            dim=384,
            depth=12,
            heads=6,
            memory_capacity=2048
        )

    def train_step(self, batch):
        # HRM: No intermediate supervision
        loss = cross_entropy(output, target)

        # Titans: Compute surprise from gradient
        loss.backward(retain_graph=True)
        surprise = compute_gradient_magnitude(loss)

        # Titans: Update memory based on surprise
        if surprise > threshold:
            self.model.update_memory(surprise)

        # HRM: Two-timescale processing
        # (implemented through architecture depth)

        return loss
```

### Key Insight

**We're not building a Cognate architecture from scratch.**
**We're using Titans architecture as our Cognate.**
**The innovation is combining HRM and Titans training processes.**

### Why This Combination?

1. **Titans gives us**: Adaptive memory that learns what to remember
2. **HRM gives us**: Efficient training without intermediate supervision
3. **Together**: A model that learns reasoning patterns with minimal data and remembers important information adaptively

### Datasets

Using datasets from both papers:
- **HRM datasets**: ARC, Sudoku (reasoning tasks, small data)
- **Titans datasets**: WikiText, PIQA, HellaSwag (language/memory tasks)
- **Original Cognate**: GSM8K, HotpotQA (mixed reasoning/memory)

### Summary

Cognate Phase 1 = Tiny Titans (25M) + HRM training process + Titans memory updates

This is elegant because:
- We leverage proven architectures (Titans)
- We combine complementary training approaches (HRM + Titans)
- We get benefits of both: efficient reasoning (HRM) + adaptive memory (Titans)