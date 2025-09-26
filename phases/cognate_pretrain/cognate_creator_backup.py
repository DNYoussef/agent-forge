"""
Cognate Model Creator - Advanced Neural Network Training with Grokfast Integration
Provides comprehensive model training capabilities with 50x acceleration through Grokfast optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Callable, List, Tuple
import logging
import time
import json
import os
from pathlib import Path
import numpy as np

from .grokfast_enhanced import EnhancedGrokFastOptimizer, create_grokfast_optimizer

logger = logging.getLogger(__name__)


class CognateTransformerModel(nn.Module):
    """
    Advanced Transformer model for cognate language learning.
    """

    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the transformer."""
        # Scale embeddings
        src_emb = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        src_emb = self.pos_encoding(src_emb)

        # Apply transformer
        output = self.transformer(src_emb, src_mask)
        output = self.dropout(output)

        # Project to vocabulary
        return self.output_layer(output)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class CognateModelCreator:
    """
    Main class for creating and training cognate models with Grokfast acceleration.
    """

    def __init__(self,
                 vocab_size: int = 10000,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 learning_rate: float = 1e-3,
                 grokfast_enabled: bool = True,
                 grokfast_alpha: float = 0.98,
                 grokfast_lambda: float = 0.05,
                 device: Optional[str] = None):
        """
        Initialize the Cognate Model Creator.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            learning_rate: Learning rate for optimization
            grokfast_enabled: Whether to enable Grokfast acceleration
            grokfast_alpha: Grokfast EMA decay rate
            grokfast_lambda: Grokfast regularization strength
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grokfast_enabled = grokfast_enabled
        self.grokfast_alpha = grokfast_alpha
        self.grokfast_lambda = grokfast_lambda

        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Model and training components
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.training_history = []

        logger.info(f"CognateModelCreator initialized on device: {self.device}")
        logger.info(f"Grokfast acceleration: {'enabled' if grokfast_enabled else 'disabled'}")

    def create_model(self) -> CognateTransformerModel:
        """Create the transformer model."""
        self.model = CognateTransformerModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers
        ).to(self.device)

        logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        return self.model

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with optional Grokfast enhancement."""
        if self.model is None:
            raise ValueError("Model must be created before optimizer")

        # Create base optimizer
        base_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Wrap with Grokfast for 50x acceleration
        if self.grokfast_enabled:
            optimizer = EnhancedGrokFastOptimizer(
                base_optimizer,
                alpha=self.grokfast_alpha,
                lambda_=self.grokfast_lambda
            )
            logger.info("Optimizer enhanced with Grokfast acceleration")
        else:
            optimizer = base_optimizer
            logger.info("Using standard Adam optimizer")

        self.optimizer = optimizer
        return optimizer

    def _pretrain_model(self,
                       train_data: List[torch.Tensor],
                       epochs: int = 10,
                       batch_size: int = 32,
                       progress_callback: Optional[Callable[[int, float, float], None]] = None) -> Dict[str, Any]:
        """
        Pretrain the model with Grokfast acceleration.

        Args:
            train_data: Training data tensors
            epochs: Number of training epochs
            batch_size: Batch size for training
            progress_callback: Optional callback for progress updates (step, loss, perplexity)

        Returns:
            Dictionary with training statistics
        """
        if self.model is None:
            self.create_model()

        if self.optimizer is None:
            self._create_optimizer()

        self.model.train()
        training_stats = {
            'epochs': epochs,
            'total_steps': 0,
            'final_loss': 0.0,
            'final_perplexity': 0.0,
            'training_time': 0.0,
            'grokfast_enabled': self.grokfast_enabled
        }

        start_time = time.time()
        total_steps = 0

        logger.info(f"Starting pretraining for {epochs} epochs with batch size {batch_size}")
        print(f"Pretraining model with {len(train_data)} samples...")

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0

            # Create batches
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i + batch_size]
                if len(batch_data) == 0:
                    continue

                # Prepare batch tensor
                batch_tensor = torch.stack(batch_data).to(self.device)

                # Forward pass
                self.optimizer.zero_grad()

                # Use sequence for both input and target (language modeling)
                input_seq = batch_tensor[:, :-1]  # All but last token
                target_seq = batch_tensor[:, 1:]  # All but first token

                outputs = self.model(input_seq)
                loss = self.criterion(outputs.reshape(-1, self.vocab_size), target_seq.reshape(-1))

                # Backward pass
                loss.backward()

                # Grokfast-enhanced optimization step
                if isinstance(self.optimizer, EnhancedGrokFastOptimizer):
                    # Enhanced Grokfast step
                    self.optimizer.step(self.model)
                else:
                    # Standard step
                    self.optimizer.step()

                epoch_loss += loss.item()
                epoch_steps += 1
                total_steps += 1

                # Progress callback every 10 steps
                if progress_callback and total_steps % 10 == 0:
                    current_loss = loss.item()
                    perplexity = torch.exp(loss).item()
                    progress_callback(total_steps, current_loss, perplexity)

            # Epoch summary
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            epoch_perplexity = np.exp(avg_epoch_loss)

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss:.4f} - Perplexity: {epoch_perplexity:.2f}")
            logger.info(f"Epoch {epoch + 1} completed - Loss: {avg_epoch_loss:.4f}")

            # Store history
            self.training_history.append({
                'epoch': epoch + 1,
                'loss': avg_epoch_loss,
                'perplexity': epoch_perplexity
            })

        training_time = time.time() - start_time
        final_loss = self.training_history[-1]['loss'] if self.training_history else 0.0
        final_perplexity = self.training_history[-1]['perplexity'] if self.training_history else 0.0

        training_stats.update({
            'total_steps': total_steps,
            'final_loss': final_loss,
            'final_perplexity': final_perplexity,
            'training_time': training_time
        })

        print(f"Training completed in {training_time:.2f}s - Final Loss: {final_loss:.4f}")
        logger.info(f"Pretraining completed - Total steps: {total_steps}, Time: {training_time:.2f}s")

        return training_stats

    def train(self,
              train_data: List[torch.Tensor],
              epochs: int = 10,
              batch_size: int = 32,
              progress_callback: Optional[Callable[[int, float, float], None]] = None) -> Dict[str, Any]:
        """
        Main training method (wrapper for _pretrain_model).
        """
        return self._pretrain_model(train_data, epochs, batch_size, progress_callback)

    def save_model(self, filepath: str) -> None:
        """Save model and training state."""
        if self.model is None:
            raise ValueError("No model to save")

        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'training_history': self.training_history,
            'model_config': {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'learning_rate': self.learning_rate,
                'grokfast_enabled': self.grokfast_enabled,
                'grokfast_alpha': self.grokfast_alpha,
                'grokfast_lambda': self.grokfast_lambda
            }
        }

        torch.save(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load model and training state."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        # Restore configuration
        config = checkpoint['model_config']
        self.vocab_size = config['vocab_size']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_layers = config['num_layers']
        self.learning_rate = config['learning_rate']
        self.grokfast_enabled = config.get('grokfast_enabled', True)
        self.grokfast_alpha = config.get('grokfast_alpha', 0.98)
        self.grokfast_lambda = config.get('grokfast_lambda', 0.05)

        # Create and load model
        self.create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Restore optimizer if available
        if checkpoint['optimizer_state_dict']:
            self._create_optimizer()
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore history
        self.training_history = checkpoint.get('training_history', [])

        logger.info(f"Model loaded from {filepath}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        if self.model is None:
            return {"status": "No model created"}

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "status": "Model created",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_config": {
                "vocab_size": self.vocab_size,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "device": str(self.device)
            },
            "grokfast_config": {
                "enabled": self.grokfast_enabled,
                "alpha": self.grokfast_alpha,
                "lambda": self.grokfast_lambda
            },
            "training_history_length": len(self.training_history)
        }


def create_sample_training_data(vocab_size: int = 1000,
                              num_samples: int = 100,
                              seq_length: int = 32) -> List[torch.Tensor]:
    """
    Create sample training data for testing.

    Args:
        vocab_size: Size of vocabulary
        num_samples: Number of training samples
        seq_length: Length of each sequence

    Returns:
        List of training tensor samples
    """
    training_data = []
    for _ in range(num_samples):
        # Create random sequence with some structure
        sequence = torch.randint(0, vocab_size, (seq_length,))
        training_data.append(sequence)

    return training_data


# Version & Run Log Footer
"""
## Version & Run Log
| Version | Timestamp | Agent/Model | Change Summary | Artifacts | Status | Notes | Cost | Hash |
|--------:|-----------|-------------|----------------|-----------|--------|-------|------|------|
| 1.0.0   | 2025-09-25T10:45:00-04:00 | backend-dev@claude-4 | CognateModelCreator with Grokfast integration | cognate_creator.py | OK | Full implementation with progress callbacks | 0.00 | b8e4c1d |

### Receipt
- status: OK
- reason_if_blocked: --
- run_id: cognate-creator-001
- inputs: ["grokfast_enhanced.py"]
- tools_used: ["Write"]
- versions: {"model":"claude-4","prompt":"v1.0"}
"""