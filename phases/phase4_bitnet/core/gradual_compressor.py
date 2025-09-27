#!/usr/bin/env python3
"""
Gradual BitNet Compression with HuggingFace Methodology
=======================================================

Implements HuggingFace's gradual compression approach for BitNet 1.58-bit quantization.
Uses progressive quantization-aware training (QAT) with sigmoid scheduling for smooth
transition from full precision to ternary weights.

Key Features:
- 3-stage gradual compression (16-bit → transition → 1.58-bit)
- Sigmoid lambda scheduler for smooth quantization strength
- Shadow weights system with straight-through estimators (STE)
- Temperature annealing for soft quantization
- Integration with Phase 3 self-generated text
- Real-time compression monitoring

Based on research:
- BitNet b1.58 paper (Microsoft Research)
- HuggingFace BitNet implementation
- Fast Quiet-STaR curriculum learning patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
import logging
from dataclasses import dataclass
from tqdm import tqdm

from .bitnet_base import BitNetComponent, BitNetConfig, BitNetQuantizationMixin


@dataclass
class CompressionStage:
    """Definition of a compression stage"""
    name: str
    start_step: int
    end_step: int
    target_bits: float
    learning_rate: float
    description: str


class StraightThroughEstimator(Function):
    """
    Straight-through estimator for gradient flow through quantization.
    Forward: quantize
    Backward: pass gradients through
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, scale: float) -> torch.Tensor:
        """Quantize to {-1, 0, +1} in forward pass"""
        # Store scale for backward pass
        ctx.save_for_backward(input)
        ctx.scale = scale

        # Quantize using absmean method
        if scale == 0:
            return torch.zeros_like(input)

        quantized = torch.round(input / scale).clamp(-1, 1)
        return quantized * scale  # Return scaled quantized values

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Pass gradients through with optional clipping"""
        input, = ctx.saved_tensors

        # Apply gradient clipping for stability
        grad_input = F.hardtanh(grad_output, min_val=-1.0, max_val=1.0)

        # Scale preservation: maintain gradient magnitude
        # This helps prevent gradient vanishing during quantization
        grad_scale = grad_output.abs().mean() / (grad_input.abs().mean() + 1e-8)
        grad_input = grad_input * grad_scale

        return grad_input, None


class GradualBitNetCompressor(BitNetComponent, BitNetQuantizationMixin):
    """
    Gradual compression system for BitNet with HuggingFace methodology.
    Progressively transitions from full precision to 1.58-bit quantization.
    """

    def __init__(
        self,
        config: Optional[BitNetConfig] = None,
        model: Optional[nn.Module] = None,
        **kwargs
    ):
        """
        Initialize gradual compressor.

        Args:
            config: BitNet configuration
            model: Model to compress
            **kwargs: Additional configuration
        """
        super().__init__(config, model, **kwargs)

        # Initialize compression stages
        self.stages = self._define_compression_stages()
        self.current_stage_idx = 0

        # Shadow weights for gradient flow
        self.shadow_weights = {}
        self.weight_scales = {}

        # Lambda scheduling
        self.lambda_value = 0.0  # Quantization strength (0=full precision, 1=fully quantized)
        self.scheduler_func = self._create_scheduler()

        # Temperature for soft quantization
        self.temperature = 1.0
        self.temperature_schedule = lambda step: max(0.1, 1.0 - step / 10000)

        # Phase 3 text buffer for self-supervised training
        self.phase3_text_buffer = []
        self.phase3_reasoning_tokens = []

        # Monitoring
        self.compression_history = {
            'steps': [],
            'lambda_values': [],
            'compression_ratios': [],
            'performance_scores': [],
            'weight_distributions': [],
        }

        # STE function for gradient flow
        self.ste = StraightThroughEstimator.apply

        # Initialize shadow weights if model provided
        if self.base_model:
            self._initialize_shadow_weights(self.base_model)

    def _define_compression_stages(self) -> List[CompressionStage]:
        """
        Define compression stages based on configuration.

        Returns:
            List[CompressionStage]: Compression stage definitions
        """
        stages = [
            CompressionStage(
                name="full_precision",
                start_step=0,
                end_step=self.config.warmup_steps,
                target_bits=16.0,
                learning_rate=self.config.learning_rate,
                description="Full precision training (16-bit) for foundation"
            ),
            CompressionStage(
                name="gradual_transition",
                start_step=self.config.warmup_steps,
                end_step=self.config.warmup_steps + self.config.transition_steps,
                target_bits=4.0,  # Intermediate target
                learning_rate=self.config.learning_rate * 0.5,
                description="Gradual transition with sigmoid scheduling"
            ),
            CompressionStage(
                name="full_quantization",
                start_step=self.config.warmup_steps + self.config.transition_steps,
                end_step=100000,  # Continue until convergence
                target_bits=1.58,
                learning_rate=self.config.learning_rate * 0.1,
                description="Full 1.58-bit quantization with fine-tuning"
            ),
        ]

        return stages

    def _create_scheduler(self) -> Callable[[int], float]:
        """
        Create lambda scheduler function based on configuration.

        Returns:
            Callable: Scheduler function (step -> lambda_value)
        """
        if self.config.scheduler_type == "sigmoid":
            def sigmoid_scheduler(step: int) -> float:
                """Sigmoid scheduler for smooth transition"""
                if step < self.config.warmup_steps:
                    return 0.0

                # Normalize step within transition period
                transition_start = self.config.warmup_steps
                transition_end = transition_start + self.config.transition_steps

                if step >= transition_end:
                    return 1.0

                normalized = (step - transition_start) / self.config.transition_steps
                k = self.config.scheduler_k
                return 1.0 / (1.0 + np.exp(-k * (normalized - 0.5)))

            return sigmoid_scheduler

        elif self.config.scheduler_type == "linear":
            def linear_scheduler(step: int) -> float:
                """Linear scheduler for gradual transition"""
                if step < self.config.warmup_steps:
                    return 0.0

                transition_start = self.config.warmup_steps
                transition_end = transition_start + self.config.transition_steps

                if step >= transition_end:
                    return 1.0

                return (step - transition_start) / self.config.transition_steps

            return linear_scheduler

        elif self.config.scheduler_type == "exponential":
            def exponential_scheduler(step: int) -> float:
                """Exponential scheduler for rapid transition"""
                if step < self.config.warmup_steps:
                    return 0.0

                transition_start = self.config.warmup_steps
                transition_end = transition_start + self.config.transition_steps

                if step >= transition_end:
                    return 1.0

                normalized = (step - transition_start) / self.config.transition_steps
                return 1.0 - np.exp(-self.config.scheduler_k * normalized)

            return exponential_scheduler

        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")

    def _initialize_shadow_weights(self, model: nn.Module) -> None:
        """
        Initialize shadow weights for gradient flow.

        Args:
            model: Model to extract weights from
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Create shadow (full precision) copy of weights
                self.shadow_weights[name] = nn.Parameter(
                    module.weight.data.clone().to(self.device)
                )
                # Initialize scale factors
                self.weight_scales[name] = 1.0

                self.logger.info(f"Initialized shadow weights for {name}")

    def apply_gradual_quantization(
        self,
        model: nn.Module,
        step: int,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Apply gradual quantization to model weights.

        Args:
            model: Model to quantize
            step: Current training step
            training: Whether in training mode

        Returns:
            Dict[str, torch.Tensor]: Quantized weights
        """
        # Update lambda value based on scheduler
        self.lambda_value = self.scheduler_func(step)
        self.compression_step = step

        # Update temperature for soft quantization
        self.temperature = self.temperature_schedule(step)

        quantized_weights = {}

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in self.shadow_weights:
                # Get shadow weights
                shadow = self.shadow_weights[name]

                if training:
                    # Mixed precision: interpolate between full and quantized
                    # Full precision component
                    full_precision = shadow * (1 - self.lambda_value)

                    # Quantized component with STE
                    scale = shadow.abs().mean()
                    quantized = self.ste(shadow, scale)
                    quantized_component = quantized * self.lambda_value

                    # Combine
                    mixed_weights = full_precision + quantized_component

                    # Apply temperature-based soft quantization
                    if self.temperature < 1.0:
                        mixed_weights = self._apply_soft_quantization(mixed_weights)

                    # Update module weights
                    module.weight.data = mixed_weights
                    quantized_weights[name] = mixed_weights

                    # Store scale for later use
                    self.weight_scales[name] = scale.item()
                else:
                    # Inference: use fully quantized weights if past transition
                    if self.lambda_value >= 1.0:
                        quantized, scale = self.quantize_weights_absmean(shadow)
                        module.weight.data = quantized * scale
                        quantized_weights[name] = module.weight.data
                    else:
                        module.weight.data = shadow
                        quantized_weights[name] = shadow

        # Update metrics
        self._update_compression_metrics(model, step)

        return quantized_weights

    def _apply_soft_quantization(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature-based soft quantization.

        Args:
            weights: Weights to soft-quantize

        Returns:
            torch.Tensor: Soft-quantized weights
        """
        # Soft ternary quantization with temperature
        scale = weights.abs().mean()
        if scale == 0:
            return weights

        normalized = weights / scale

        # Soft assignment to {-1, 0, 1} using tanh
        soft_quantized = torch.tanh(normalized / self.temperature) * scale

        return soft_quantized

    def update_shadow_weights(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """
        Update shadow weights after optimizer step.

        Args:
            model: Model with updated weights
            optimizer: Optimizer that performed the step
        """
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and name in self.shadow_weights:
                    # Update shadow weights with gradient-updated values
                    # Shadow weights maintain full precision for gradient flow
                    if module.weight.grad is not None:
                        # Apply gradient to shadow weights
                        self.shadow_weights[name].data -= (
                            optimizer.param_groups[0]['lr'] * module.weight.grad
                        )

    def integrate_phase3_reasoning(
        self,
        reasoning_tokens: List[torch.Tensor],
        generated_text: List[str]
    ) -> None:
        """
        Integrate Phase 3 Quiet Star reasoning tokens and self-generated text.

        Args:
            reasoning_tokens: Reasoning tokens from Phase 3
            generated_text: Self-generated text from Phase 3
        """
        self.phase3_reasoning_tokens.extend(reasoning_tokens)
        self.phase3_text_buffer.extend(generated_text)

        # Update reasoning preservation stats
        self.reasoning_stats['tokens_preserved'] = len(self.phase3_reasoning_tokens)
        self.reasoning_stats['self_generated_samples'] = len(self.phase3_text_buffer)

        self.logger.info(
            f"Integrated {len(reasoning_tokens)} reasoning tokens and "
            f"{len(generated_text)} text samples from Phase 3"
        )

    def get_self_generated_batch(self, batch_size: int) -> List[str]:
        """
        Get batch of self-generated text from Phase 3 for training.

        Args:
            batch_size: Size of batch to retrieve

        Returns:
            List[str]: Batch of self-generated text
        """
        if not self.phase3_text_buffer:
            return []

        # Sample from buffer with replacement if needed
        if len(self.phase3_text_buffer) >= batch_size:
            indices = np.random.choice(len(self.phase3_text_buffer), batch_size, replace=False)
        else:
            indices = np.random.choice(len(self.phase3_text_buffer), batch_size, replace=True)

        return [self.phase3_text_buffer[i] for i in indices]

    def _update_compression_metrics(self, model: nn.Module, step: int) -> None:
        """
        Update compression metrics and history.

        Args:
            model: Model being compressed
            step: Current step
        """
        # Calculate weight distribution
        weight_dist = self._calculate_weight_distribution(model)

        # Calculate effective bits
        if self.lambda_value >= 1.0:
            effective_bits = 1.58
        else:
            effective_bits = 16 * (1 - self.lambda_value) + 1.58 * self.lambda_value

        # Update current bits
        self.current_bits = effective_bits

        # Calculate compression ratio
        compression_ratio = 32.0 / effective_bits

        # Update history
        self.compression_history['steps'].append(step)
        self.compression_history['lambda_values'].append(self.lambda_value)
        self.compression_history['compression_ratios'].append(compression_ratio)
        self.compression_history['weight_distributions'].append(weight_dist)

        # Update base metrics
        self.update_metrics(
            compression_ratio=compression_ratio,
            quantization_loss=1.0 - self.lambda_value  # Simplified metric
        )

    def _calculate_weight_distribution(self, model: nn.Module) -> Dict[str, float]:
        """
        Calculate distribution of weights in {-1, 0, +1}.

        Args:
            model: Model to analyze

        Returns:
            Dict[str, float]: Distribution percentages
        """
        total_weights = 0
        negative_count = 0
        zero_count = 0
        positive_count = 0

        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    weights = module.weight.data
                    quantized, _ = self.quantize_weights_absmean(weights)

                    negative_count += (quantized == -1).sum().item()
                    zero_count += (quantized == 0).sum().item()
                    positive_count += (quantized == 1).sum().item()
                    total_weights += quantized.numel()

        if total_weights == 0:
            return {'negative': 0.0, 'zero': 0.0, 'positive': 0.0}

        return {
            'negative': negative_count / total_weights,
            'zero': zero_count / total_weights,
            'positive': positive_count / total_weights,
        }

    def get_current_stage(self) -> CompressionStage:
        """
        Get current compression stage based on step.

        Returns:
            CompressionStage: Current stage
        """
        for stage in self.stages:
            if stage.start_step <= self.compression_step < stage.end_step:
                return stage
        return self.stages[-1]  # Return final stage if beyond defined stages

    def export_compression_report(self) -> Dict[str, Any]:
        """
        Export comprehensive compression report.

        Returns:
            Dict[str, Any]: Compression report
        """
        current_stage = self.get_current_stage()
        weight_dist = self.compression_history['weight_distributions'][-1] if self.compression_history['weight_distributions'] else {}

        report = {
            'configuration': self.config.to_dict(),
            'current_state': {
                'step': self.compression_step,
                'stage': current_stage.name,
                'lambda_value': self.lambda_value,
                'temperature': self.temperature,
                'effective_bits': self.current_bits,
                'is_fully_quantized': self.is_quantized,
            },
            'performance_metrics': self.performance_metrics,
            'weight_distribution': weight_dist,
            'phase3_integration': self.reasoning_stats,
            'quality_gates': self.validate_quality_gates(),
            'memory_footprint': self.get_memory_footprint(),
            'compression_history': {
                'total_steps': len(self.compression_history['steps']),
                'final_compression_ratio': self.compression_history['compression_ratios'][-1] if self.compression_history['compression_ratios'] else 1.0,
                'lambda_progression': self.compression_history['lambda_values'][-10:] if self.compression_history['lambda_values'] else [],
            }
        }

        return report

    def train_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        step: int
    ) -> Dict[str, float]:
        """
        Single training step with gradual compression.

        Args:
            model: Model to train
            batch: Training batch
            optimizer: Optimizer
            criterion: Loss criterion
            step: Current step

        Returns:
            Dict[str, float]: Training metrics
        """
        model.train()

        # Apply gradual quantization
        self.apply_gradual_quantization(model, step, training=True)

        # Forward pass
        outputs = model(**batch)
        loss = criterion(outputs.logits, batch['labels'])

        # Add quantization regularization
        if self.lambda_value > 0:
            # Encourage ternary distribution
            reg_loss = self._compute_quantization_regularization(model)
            loss = loss + 0.01 * reg_loss * self.lambda_value

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update shadow weights
        self.update_shadow_weights(model, optimizer)

        # Optimizer step
        optimizer.step()

        # Calculate metrics
        metrics = {
            'loss': loss.item(),
            'lambda': self.lambda_value,
            'temperature': self.temperature,
            'compression_ratio': self.calculate_compression_ratio(),
            'stage': self.get_current_stage().name,
        }

        return metrics

    def _compute_quantization_regularization(self, model: nn.Module) -> torch.Tensor:
        """
        Compute regularization loss to encourage ternary weights.

        Args:
            model: Model with weights

        Returns:
            torch.Tensor: Regularization loss
        """
        reg_loss = 0.0

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weights = module.weight
                scale = weights.abs().mean()

                if scale > 0:
                    normalized = weights / scale
                    # Encourage weights to be close to {-1, 0, 1}
                    distance_to_ternary = torch.min(
                        torch.min(
                            (normalized - 1).abs(),
                            (normalized + 1).abs()
                        ),
                        normalized.abs()
                    )
                    reg_loss = reg_loss + distance_to_ternary.mean()

        return reg_loss


# Export main class
__all__ = ['GradualBitNetCompressor', 'CompressionStage', 'StraightThroughEstimator']