"""
Weight Space Introspection System using Transformers Squared
Enables models to examine and understand their own weight space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class WeightToken:
    """Represents a tokenized chunk of model weights"""
    layer_name: str
    position: Tuple[int, int]
    values: torch.Tensor
    importance_score: float
    gradient_norm: float = 0.0


class TransformerWeightEncoder(nn.Module):
    """
    Transformer encoder that processes weight tokens
    Based on Transformers Squared paper approach
    """

    def __init__(self, d_model: int = 768, n_heads: int = 12, n_layers: int = 6):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Weight embedding layers
        self.weight_embedding = nn.Linear(64, d_model)  # 64 = chunk size
        self.position_embedding = nn.Embedding(10000, d_model)
        self.layer_embedding = nn.Embedding(100, d_model)  # Max 100 layers

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, weight_tokens: List[WeightToken]) -> torch.Tensor:
        """Process weight tokens through transformer"""
        batch_size = 1  # Single model introspection
        seq_len = len(weight_tokens)

        # Convert tokens to tensors
        weight_values = torch.stack([token.values for token in weight_tokens])
        weight_values = weight_values.view(batch_size, seq_len, -1)

        # Embed weights
        weight_embeddings = self.weight_embedding(weight_values)

        # Add positional embeddings
        positions = torch.arange(seq_len).unsqueeze(0)
        pos_embeddings = self.position_embedding(positions)

        # Combine embeddings
        embeddings = weight_embeddings + pos_embeddings

        # Apply transformer
        encoded = self.transformer(embeddings)

        # Project output
        output = self.output_projection(encoded)

        return output


class TransformerWeightDecoder(nn.Module):
    """
    Transformer decoder that generates weight modifications
    """

    def __init__(self, d_model: int = 768, n_heads: int = 12, n_layers: int = 6):
        super().__init__()
        self.d_model = d_model

        # Task embedding
        self.task_embedding = nn.Embedding(10, d_model)  # 10 task types

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output layers for weight modifications
        self.modification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 64),  # Output modification vector
            nn.Tanh()  # Bounded modifications
        )

        self.importance_head = nn.Linear(d_model, 1)  # Importance score for modification

    def forward(self, encoded_weights: torch.Tensor, task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate weight modifications for given task"""
        batch_size, seq_len, _ = encoded_weights.shape

        # Get task embedding
        task_emb = self.task_embedding(torch.tensor([task_id]))
        task_emb = task_emb.unsqueeze(1).expand(batch_size, seq_len, -1)

        # Decode with task context
        decoded = self.transformer(task_emb, encoded_weights)

        # Generate modifications
        modifications = self.modification_head(decoded)
        importance_scores = torch.sigmoid(self.importance_head(decoded))

        return modifications, importance_scores


class WeightSpaceIntrospector:
    """
    Main class for weight space introspection
    Allows models to examine and understand their own weights
    """

    def __init__(self, target_model: nn.Module, chunk_size: int = 64, device: str = 'cuda'):
        self.target_model = target_model
        self.chunk_size = chunk_size
        self.device = device

        # Initialize transformer components
        self.weight_encoder = TransformerWeightEncoder().to(device)
        self.weight_decoder = TransformerWeightDecoder().to(device)

        # Task mappings
        self.task_map = {
            "reasoning": 0,
            "creativity": 1,
            "coding": 2,
            "analysis": 3,
            "language": 4,
            "math": 5,
            "vision": 6,
            "audio": 7,
            "multimodal": 8,
            "general": 9
        }

        # Cache for introspection results
        self.introspection_cache = {}
        self.weight_importance_map = {}

        logger.info(f"Initialized WeightSpaceIntrospector for model with {self.count_parameters()} parameters")

    def count_parameters(self) -> int:
        """Count total parameters in target model"""
        return sum(p.numel() for p in self.target_model.parameters())

    def tokenize_weights(self) -> List[WeightToken]:
        """
        Convert model weights into tokens for self-attention processing
        Each token represents a chunk of weights
        """
        weight_tokens = []

        for layer_name, param in self.target_model.named_parameters():
            if not param.requires_grad:
                continue

            # Flatten parameter tensor
            flat_weights = param.data.flatten()

            # Calculate importance based on magnitude and gradient
            importance = torch.abs(flat_weights).mean().item()
            grad_norm = 0.0
            if param.grad is not None:
                grad_norm = param.grad.norm().item()

            # Chunk weights into tokens
            num_chunks = math.ceil(len(flat_weights) / self.chunk_size)

            for i in range(num_chunks):
                start_idx = i * self.chunk_size
                end_idx = min((i + 1) * self.chunk_size, len(flat_weights))

                # Pad if necessary
                chunk = flat_weights[start_idx:end_idx]
                if len(chunk) < self.chunk_size:
                    chunk = F.pad(chunk, (0, self.chunk_size - len(chunk)))

                # Create weight token
                token = WeightToken(
                    layer_name=layer_name,
                    position=(start_idx, end_idx),
                    values=chunk.to(self.device),
                    importance_score=importance,
                    gradient_norm=grad_norm
                )

                weight_tokens.append(token)

        logger.info(f"Tokenized weights into {len(weight_tokens)} tokens")
        return weight_tokens

    def introspect(self, task_description: str) -> Dict[str, Any]:
        """
        Model examines its own weights for a specific task
        Returns understanding and proposed modifications
        """
        logger.info(f"Starting introspection for task: {task_description}")

        # Map task to ID
        task_id = self.task_map.get(task_description.lower(), 9)  # Default to 'general'

        # Tokenize current weights
        weight_tokens = self.tokenize_weights()

        # Encode weights through transformer
        with torch.no_grad():
            encoded_weights = self.weight_encoder(weight_tokens)

            # Generate modifications for task
            modifications, importance_scores = self.weight_decoder(encoded_weights, task_id)

        # Analyze weight patterns
        weight_analysis = self.analyze_weight_patterns(weight_tokens, encoded_weights)

        # Identify critical weights for task
        critical_weights = self.identify_critical_weights(
            weight_tokens,
            importance_scores.squeeze(-1)
        )

        # Generate introspection report
        introspection_result = {
            "task": task_description,
            "task_id": task_id,
            "total_weights": self.count_parameters(),
            "tokens_analyzed": len(weight_tokens),
            "weight_analysis": weight_analysis,
            "critical_weights": critical_weights,
            "proposed_modifications": {
                "modification_tensors": modifications.cpu().numpy(),
                "importance_scores": importance_scores.cpu().numpy(),
                "num_modifications": (importance_scores > 0.5).sum().item()
            },
            "introspection_depth": self.calculate_introspection_depth(encoded_weights)
        }

        # Cache result
        self.introspection_cache[task_description] = introspection_result

        logger.info(f"Introspection complete. Identified {len(critical_weights)} critical weight regions")

        return introspection_result

    def analyze_weight_patterns(self, tokens: List[WeightToken], encoded: torch.Tensor) -> Dict[str, Any]:
        """Analyze patterns in weight space"""
        # Layer-wise statistics
        layer_stats = {}
        for token in tokens:
            layer = token.layer_name.split('.')[0]
            if layer not in layer_stats:
                layer_stats[layer] = {
                    "mean_importance": [],
                    "gradient_norms": [],
                    "weight_magnitudes": []
                }

            layer_stats[layer]["mean_importance"].append(token.importance_score)
            layer_stats[layer]["gradient_norms"].append(token.gradient_norm)
            layer_stats[layer]["weight_magnitudes"].append(torch.abs(token.values).mean().item())

        # Aggregate statistics
        for layer in layer_stats:
            layer_stats[layer]["mean_importance"] = np.mean(layer_stats[layer]["mean_importance"])
            layer_stats[layer]["gradient_norms"] = np.mean(layer_stats[layer]["gradient_norms"])
            layer_stats[layer]["weight_magnitudes"] = np.mean(layer_stats[layer]["weight_magnitudes"])

        # Attention pattern analysis from encoded weights
        attention_patterns = self.analyze_attention_patterns(encoded)

        return {
            "layer_statistics": layer_stats,
            "attention_patterns": attention_patterns,
            "weight_sparsity": self.calculate_sparsity(tokens),
            "weight_distribution": self.analyze_distribution(tokens)
        }

    def analyze_attention_patterns(self, encoded: torch.Tensor) -> Dict[str, float]:
        """Analyze self-attention patterns in encoded weights"""
        # Calculate attention scores between weight tokens
        similarity_matrix = torch.matmul(encoded, encoded.transpose(-2, -1))
        similarity_matrix = F.softmax(similarity_matrix / math.sqrt(encoded.shape[-1]), dim=-1)

        # Extract pattern metrics
        patterns = {
            "mean_attention": similarity_matrix.mean().item(),
            "max_attention": similarity_matrix.max().item(),
            "attention_entropy": -torch.sum(similarity_matrix * torch.log(similarity_matrix + 1e-10)).item(),
            "attention_sparsity": (similarity_matrix < 0.01).float().mean().item()
        }

        return patterns

    def identify_critical_weights(self, tokens: List[WeightToken], importance_scores: torch.Tensor) -> List[Dict]:
        """Identify the most critical weights for the task"""
        critical_weights = []

        # Sort by importance
        importance_values = importance_scores.cpu().numpy()
        top_indices = np.argsort(importance_values)[-20:]  # Top 20 critical regions

        for idx in top_indices:
            if idx < len(tokens):
                token = tokens[idx]
                critical_weights.append({
                    "layer": token.layer_name,
                    "position": token.position,
                    "importance": float(importance_values[idx]),
                    "magnitude": torch.abs(token.values).mean().item(),
                    "gradient_norm": token.gradient_norm
                })

        return critical_weights

    def calculate_sparsity(self, tokens: List[WeightToken]) -> float:
        """Calculate overall weight sparsity"""
        total_zeros = 0
        total_weights = 0

        for token in tokens:
            total_zeros += (torch.abs(token.values) < 1e-6).sum().item()
            total_weights += len(token.values)

        return total_zeros / total_weights if total_weights > 0 else 0.0

    def analyze_distribution(self, tokens: List[WeightToken]) -> Dict[str, float]:
        """Analyze weight value distribution"""
        all_weights = torch.cat([token.values for token in tokens])

        return {
            "mean": all_weights.mean().item(),
            "std": all_weights.std().item(),
            "min": all_weights.min().item(),
            "max": all_weights.max().item(),
            "median": all_weights.median().item(),
            "skewness": self.calculate_skewness(all_weights),
            "kurtosis": self.calculate_kurtosis(all_weights)
        }

    def calculate_skewness(self, tensor: torch.Tensor) -> float:
        """Calculate skewness of distribution"""
        mean = tensor.mean()
        std = tensor.std()
        skew = ((tensor - mean) ** 3).mean() / (std ** 3)
        return skew.item()

    def calculate_kurtosis(self, tensor: torch.Tensor) -> float:
        """Calculate kurtosis of distribution"""
        mean = tensor.mean()
        std = tensor.std()
        kurt = ((tensor - mean) ** 4).mean() / (std ** 4) - 3
        return kurt.item()

    def calculate_introspection_depth(self, encoded: torch.Tensor) -> float:
        """Calculate how deeply the model introspected its weights"""
        # Measure information content in encoded representation
        singular_values = torch.linalg.svdvals(encoded.squeeze(0))

        # Normalized entropy of singular values
        singular_values = singular_values / singular_values.sum()
        entropy = -torch.sum(singular_values * torch.log(singular_values + 1e-10))

        # Normalize to 0-1 range
        max_entropy = math.log(len(singular_values))
        introspection_depth = (entropy / max_entropy).item() if max_entropy > 0 else 0.0

        return introspection_depth

    def apply_modifications(self, modifications: Dict[str, Any]) -> None:
        """Apply weight modifications to the target model"""
        logger.info("Applying weight modifications to model")

        modification_tensors = modifications["modification_tensors"]
        importance_scores = modifications["importance_scores"]

        # Apply modifications to high-importance weights
        token_idx = 0
        for name, param in self.target_model.named_parameters():
            if not param.requires_grad:
                continue

            flat_weights = param.data.flatten()
            num_chunks = math.ceil(len(flat_weights) / self.chunk_size)

            for i in range(num_chunks):
                if token_idx >= len(modification_tensors):
                    break

                if importance_scores[token_idx] > 0.5:  # Only apply important modifications
                    start_idx = i * self.chunk_size
                    end_idx = min((i + 1) * self.chunk_size, len(flat_weights))

                    # Apply scaled modification
                    modification = torch.tensor(modification_tensors[token_idx][:end_idx-start_idx])
                    modification = modification.to(param.device)

                    # Scale by importance
                    scale = float(importance_scores[token_idx]) * 0.1  # Max 10% change
                    flat_weights[start_idx:end_idx] += modification * scale

                token_idx += 1

            # Reshape and update parameter
            param.data = flat_weights.reshape(param.shape)

        logger.info(f"Applied modifications to {(importance_scores > 0.5).sum()} weight regions")

    def get_weight_space_snapshot(self) -> Dict[str, Any]:
        """Get current snapshot of weight space for visualization"""
        snapshot = {
            "timestamp": torch.cuda.Event.elapsed_time() if torch.cuda.is_available() else 0,
            "layers": {},
            "global_stats": {}
        }

        for name, param in self.target_model.named_parameters():
            layer_name = name.split('.')[0]

            if layer_name not in snapshot["layers"]:
                snapshot["layers"][layer_name] = {
                    "num_parameters": 0,
                    "mean_magnitude": 0,
                    "sparsity": 0,
                    "gradient_norm": 0
                }

            snapshot["layers"][layer_name]["num_parameters"] += param.numel()
            snapshot["layers"][layer_name]["mean_magnitude"] += torch.abs(param.data).mean().item()
            snapshot["layers"][layer_name]["sparsity"] += (torch.abs(param.data) < 1e-6).float().mean().item()

            if param.grad is not None:
                snapshot["layers"][layer_name]["gradient_norm"] += param.grad.norm().item()

        # Global statistics
        total_params = self.count_parameters()
        snapshot["global_stats"] = {
            "total_parameters": total_params,
            "introspection_depth": len(self.introspection_cache),
            "cached_tasks": list(self.introspection_cache.keys())
        }

        return snapshot