"""
ModelOperations - Unified utility class for all model operations.

Consolidates duplicate model operations found across:
- EvolutionaryEngine.py: _clone_model (lines 610-613)
- dare_operator.py: _clone_model (lines 646-649)
- slerp_operator.py: _clone_model (lines 385-388)
- ties_operator.py: _clone_model (lines 535-538)

This eliminates COA-003 critical violation: 4x identical _clone_model functions.
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import threading
from typing import Dict, Any, Optional, List, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DeviceStrategy(Enum):
    """Device management strategies for model operations."""
    AUTO = "auto"  # Automatically detect and use best device
    CPU = "cpu"    # Force CPU operations
    GPU = "gpu"    # Force GPU operations if available
    ORIGINAL = "original"  # Maintain original model device


class ModelOperationError(Exception):
    """Custom exception for model operation failures."""
    pass


@dataclass
class ModelMetadata:
    """Metadata for model operations tracking."""
    device: torch.device
    dtype: torch.dtype
    parameter_count: int
    memory_usage_mb: float
    operation_history: List[str]


class ModelOperations(ABC):
    """
    Abstract base class for all EvoMerge model operations.

    Provides thread-safe, device-aware model operations with caching
    and comprehensive error handling. Eliminates all duplicate model
    manipulation code across the EvoMerge system.
    """

    def __init__(self,
                 device_strategy: DeviceStrategy = DeviceStrategy.AUTO,
                 enable_caching: bool = True,
                 cache_size: int = 16):
        """
        Initialize ModelOperations with configuration.

        Args:
            device_strategy: Strategy for device management
            enable_caching: Whether to cache cloned models
            cache_size: Maximum number of cached models (matches 16-model constraint)
        """
        self.device_strategy = device_strategy
        self.enable_caching = enable_caching
        self.cache_size = cache_size

        # Thread-safe caching
        self._model_cache: Dict[str, nn.Module] = {}
        self._metadata_cache: Dict[str, ModelMetadata] = {}
        self._cache_lock = threading.Lock()

        # Device detection
        self._default_device = self._detect_optimal_device()

        logger.info(f"ModelOperations initialized: device={self._default_device}, caching={enable_caching}")

    def _detect_optimal_device(self) -> torch.device:
        """Detect optimal device based on strategy."""
        if self.device_strategy == DeviceStrategy.CPU:
            return torch.device("cpu")
        elif self.device_strategy == DeviceStrategy.GPU:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:  # AUTO
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_model_hash(self, model: nn.Module) -> str:
        """Generate hash for model caching."""
        # Use model structure and first few parameters for hashing
        param_signature = []
        for name, param in list(model.named_parameters())[:5]:  # First 5 params for speed
            if param.numel() > 0:
                param_signature.append(f"{name}:{param.shape}:{param.data.flatten()[:3].sum().item():.6f}")

        return hash("|".join(param_signature)).__str__()

    def _get_model_metadata(self, model: nn.Module) -> ModelMetadata:
        """Extract metadata from model."""
        total_params = sum(p.numel() for p in model.parameters())

        # Estimate memory usage (rough approximation)
        memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        dtype = next(model.parameters()).dtype if list(model.parameters()) else torch.float32

        return ModelMetadata(
            device=device,
            dtype=dtype,
            parameter_count=total_params,
            memory_usage_mb=memory_mb,
            operation_history=[]
        )

    def clone_model(self,
                   model: nn.Module,
                   target_device: Optional[torch.device] = None,
                   preserve_training_state: bool = True) -> nn.Module:
        """
        Create a deep copy of a model with device and caching management.

        This method consolidates all 4 duplicate _clone_model implementations:
        - EvolutionaryEngine._clone_model()
        - DAREOperator._clone_model()
        - SLERPOperator._clone_model()
        - TIESOperator._clone_model()

        Args:
            model: Source model to clone
            target_device: Target device for cloned model
            preserve_training_state: Whether to preserve training/eval mode

        Returns:
            Deep copy of the model

        Raises:
            ModelOperationError: If cloning fails
        """
        try:
            model_hash = self._get_model_hash(model)

            # Check cache first
            if self.enable_caching:
                with self._cache_lock:
                    if model_hash in self._model_cache:
                        cached_model = self._model_cache[model_hash]
                        logger.debug(f"Retrieved model from cache: {model_hash[:8]}")

                        # Clone the cached model to avoid mutation
                        cloned = copy.deepcopy(cached_model)
                        if target_device:
                            cloned = cloned.to(target_device)
                        return cloned

            # Perform deep copy (core consolidation of duplicate code)
            original_training = model.training
            cloned_model = copy.deepcopy(model)

            # Preserve training state if requested
            if preserve_training_state:
                cloned_model.train(original_training)

            # Handle device placement
            if target_device:
                cloned_model = cloned_model.to(target_device)
            elif self.device_strategy != DeviceStrategy.ORIGINAL:
                cloned_model = cloned_model.to(self._default_device)

            # Cache the result
            if self.enable_caching:
                with self._cache_lock:
                    # Implement LRU cache with size limit
                    if len(self._model_cache) >= self.cache_size:
                        # Remove oldest entry
                        oldest_key = next(iter(self._model_cache))
                        del self._model_cache[oldest_key]
                        del self._metadata_cache[oldest_key]

                    self._model_cache[model_hash] = copy.deepcopy(cloned_model)
                    self._metadata_cache[model_hash] = self._get_model_metadata(cloned_model)

            logger.debug(f"Successfully cloned model: {self._get_model_metadata(cloned_model).parameter_count:,} parameters")
            return cloned_model

        except Exception as e:
            raise ModelOperationError(f"Failed to clone model: {str(e)}") from e

    def calculate_model_distance(self,
                               model1: nn.Module,
                               model2: nn.Module,
                               distance_type: str = "euclidean") -> float:
        """
        Calculate distance between two models.

        Consolidates duplicate distance calculation algorithms found in:
        - COA-002 violation: 3x duplicate distance calculations

        Args:
            model1: First model
            model2: Second model
            distance_type: Type of distance ("euclidean", "cosine", "geodesic")

        Returns:
            Distance between models
        """
        try:
            # Get flattened parameters
            params1 = torch.cat([p.flatten() for p in model1.parameters()])
            params2 = torch.cat([p.flatten() for p in model2.parameters()])

            if params1.shape != params2.shape:
                raise ModelOperationError(f"Model parameter shapes don't match: {params1.shape} vs {params2.shape}")

            if distance_type == "euclidean":
                return torch.norm(params1 - params2).item()
            elif distance_type == "cosine":
                cos_sim = torch.nn.functional.cosine_similarity(params1.unsqueeze(0), params2.unsqueeze(0))
                return (1 - cos_sim).item()
            elif distance_type == "geodesic":
                # Geodesic distance on parameter manifold
                norm1 = torch.norm(params1)
                norm2 = torch.norm(params2)
                dot_product = torch.dot(params1, params2)
                cos_angle = dot_product / (norm1 * norm2 + 1e-8)
                cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
                return torch.acos(cos_angle).item()
            else:
                raise ModelOperationError(f"Unknown distance type: {distance_type}")

        except Exception as e:
            raise ModelOperationError(f"Failed to calculate model distance: {str(e)}") from e

    def get_model_summary(self, model: nn.Module) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        metadata = self._get_model_metadata(model)

        return {
            "parameter_count": metadata.parameter_count,
            "memory_usage_mb": metadata.memory_usage_mb,
            "device": str(metadata.device),
            "dtype": str(metadata.dtype),
            "layer_count": len(list(model.modules())),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_params": sum(p.numel() for p in model.parameters() if not p.requires_grad)
        }

    def clear_cache(self):
        """Clear the model cache."""
        with self._cache_lock:
            self._model_cache.clear()
            self._metadata_cache.clear()
        logger.info("Model cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            total_memory = sum(meta.memory_usage_mb for meta in self._metadata_cache.values())
            return {
                "cached_models": len(self._model_cache),
                "cache_size_limit": self.cache_size,
                "total_memory_mb": total_memory,
                "cache_hit_ratio": getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1)
            }


class ConcurrentModelOperations(ModelOperations):
    """
    Thread-safe model operations for concurrent evolutionary processing.

    Extends ModelOperations with concurrent access patterns needed for
    the Phase 2 EvoMerge system with parallel model evaluation.
    """

    def __init__(self, max_workers: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.max_workers = max_workers
        self._operation_semaphore = threading.Semaphore(max_workers)

    def batch_clone_models(self,
                          models: List[nn.Module],
                          target_device: Optional[torch.device] = None) -> List[nn.Module]:
        """
        Clone multiple models concurrently.

        Used for Phase 2 EvoMerge when creating 8 models from 3 inputs.
        """
        results = []

        def clone_single(model):
            with self._operation_semaphore:
                return self.clone_model(model, target_device)

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(clone_single, model) for model in models]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        return results

    def batch_calculate_distances(self,
                                model_pairs: List[Tuple[nn.Module, nn.Module]],
                                distance_type: str = "euclidean") -> List[float]:
        """Calculate distances for multiple model pairs concurrently."""
        def calc_single(pair):
            with self._operation_semaphore:
                return self.calculate_model_distance(pair[0], pair[1], distance_type)

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(calc_single, pair) for pair in model_pairs]
            return [future.result() for future in concurrent.futures.as_completed(futures)]


# Factory function for backward compatibility
def create_model_operations(concurrent: bool = False, **kwargs) -> ModelOperations:
    """
    Factory function to create ModelOperations instance.

    Args:
        concurrent: Whether to create concurrent-safe operations
        **kwargs: Additional arguments for ModelOperations

    Returns:
        ModelOperations instance
    """
    if concurrent:
        return ConcurrentModelOperations(**kwargs)
    else:
        return ModelOperations(**kwargs)


# Singleton instance for global access
_global_model_ops = None
_ops_lock = threading.Lock()


def get_model_operations() -> ModelOperations:
    """Get singleton ModelOperations instance."""
    global _global_model_ops
    if _global_model_ops is None:
        with _ops_lock:
            if _global_model_ops is None:
                _global_model_ops = ConcurrentModelOperations(
                    device_strategy=DeviceStrategy.AUTO,
                    enable_caching=True,
                    cache_size=16  # Matches Phase 2 constraint
                )
    return _global_model_ops


# Convenience functions for direct usage (replaces all _clone_model calls)
def clone_model(model: nn.Module, **kwargs) -> nn.Module:
    """Convenience function for model cloning."""
    return get_model_operations().clone_model(model, **kwargs)


def calculate_model_distance(model1: nn.Module, model2: nn.Module, **kwargs) -> float:
    """Convenience function for distance calculation."""
    return get_model_operations().calculate_model_distance(model1, model2, **kwargs)