"""
SLERP (Spherical Linear Interpolation) Operator for Model Merging.

Implements mathematically correct spherical linear interpolation for:
- Neural network parameter interpolation
- Weight space geodesic interpolation
- Quaternion-like interpolation for parameter manifolds
- Smooth transitions between model states

NO MOCK CODE - Real mathematical implementation using PyTorch.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import math
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class SLERPOperator:
    """
    Spherical Linear Interpolation operator for neural network models.
    
    SLERP provides smooth interpolation along the surface of a hypersphere,
    which is particularly useful for model merging where we want to preserve
    the relative magnitudes of parameter vectors while interpolating their directions.
    
    Mathematical foundation:
    SLERP(q1, q2, t) = (sin((1-t)θ) * q1 + sin(t*θ) * q2) / sin(θ)
    where θ is the angle between vectors q1 and q2.
    """
    
    def __init__(self, 
                 epsilon: float = 1e-8,
                 normalize_before: bool = True,
                 normalize_after: bool = True,
                 fallback_to_linear: bool = True,
                 device: str = "cpu"):
        """
        Initialize SLERP operator.
        
        Args:
            epsilon: Small value to prevent division by zero
            normalize_before: Whether to normalize vectors before SLERP
            normalize_after: Whether to normalize result after SLERP
            fallback_to_linear: Use linear interpolation when vectors are nearly parallel
            device: Device for computations
        """
        self.epsilon = epsilon
        self.normalize_before = normalize_before
        self.normalize_after = normalize_after
        self.fallback_to_linear = fallback_to_linear
        self.device = device
        
        logger.info(f"Initialized SLERP operator with epsilon={epsilon}")
    
    def interpolate(self, 
                   model1: nn.Module, 
                   model2: nn.Module, 
                   t: float) -> nn.Module:
        """
        Perform SLERP interpolation between two models.
        
        Args:
            model1: First model (t=0)
            model2: Second model (t=1)
            t: Interpolation parameter [0, 1]
            
        Returns:
            Interpolated model
        """
        if not 0 <= t <= 1:
            raise ValueError(f"Interpolation parameter t must be in [0, 1], got {t}")
            
        # Clone first model as base
        result_model = self._clone_model(model1)
        
        # Get parameter dictionaries
        params1 = dict(model1.named_parameters())
        params2 = dict(model2.named_parameters())
        result_params = dict(result_model.named_parameters())
        
        # Perform SLERP on each parameter tensor
        for name in params1:
            if name in params2:
                try:
                    interpolated = self._slerp_tensors(
                        params1[name].data,
                        params2[name].data,
                        t
                    )
                    result_params[name].data = interpolated
                except Exception as e:
                    logger.warning(f"SLERP failed for parameter {name}: {e}. Using linear interpolation.")
                    # Fallback to linear interpolation
                    result_params[name].data = (1 - t) * params1[name].data + t * params2[name].data
            else:
                logger.warning(f"Parameter {name} not found in second model")
                
        return result_model
    
    def interpolate_batch(self, 
                         models: List[nn.Module], 
                         weights: List[float]) -> nn.Module:
        """
        Perform weighted SLERP interpolation across multiple models.
        
        Uses progressive SLERP: interpolate first two models, then interpolate
        result with third model, etc.
        
        Args:
            models: List of models to interpolate
            weights: Interpolation weights (must sum to 1.0)
            
        Returns:
            Interpolated model
        """
        if len(models) != len(weights):
            raise ValueError("Number of models must equal number of weights")
            
        if abs(sum(weights) - 1.0) > self.epsilon:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")
            
        if len(models) == 0:
            raise ValueError("Must provide at least one model")
            
        if len(models) == 1:
            return self._clone_model(models[0])
            
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Progressive SLERP
        result = self._clone_model(models[0])
        cumulative_weight = weights[0]
        
        for i in range(1, len(models)):
            # Calculate relative weight for this interpolation
            total_weight = cumulative_weight + weights[i]
            if total_weight > self.epsilon:
                t = weights[i] / total_weight
                result = self.interpolate(result, models[i], t)
                cumulative_weight = total_weight
                
        return result
    
    def _slerp_tensors(self, tensor1: torch.Tensor, tensor2: torch.Tensor, t: float) -> torch.Tensor:
        """
        Perform SLERP on two tensors.
        
        Args:
            tensor1: First tensor
            tensor2: Second tensor
            t: Interpolation parameter
            
        Returns:
            Interpolated tensor
        """
        if tensor1.shape != tensor2.shape:
            raise ValueError(f"Tensor shapes must match: {tensor1.shape} vs {tensor2.shape}")
            
        # Handle edge cases
        if t == 0:
            return tensor1.clone()
        if t == 1:
            return tensor2.clone()
            
        # Flatten tensors for computation
        original_shape = tensor1.shape
        flat1 = tensor1.flatten()
        flat2 = tensor2.flatten()
        
        # Normalize if requested
        if self.normalize_before:
            norm1 = torch.norm(flat1)
            norm2 = torch.norm(flat2)
            
            if norm1 > self.epsilon:
                flat1 = flat1 / norm1
            if norm2 > self.epsilon:
                flat2 = flat2 / norm2
        
        # Calculate dot product (cosine of angle)
        dot_product = torch.dot(flat1, flat2).clamp(-1 + self.epsilon, 1 - self.epsilon)
        
        # Calculate angle between vectors
        theta = torch.acos(torch.abs(dot_product))
        
        # Check if vectors are nearly parallel
        if theta < self.epsilon:
            if self.fallback_to_linear:
                logger.debug("Vectors nearly parallel, using linear interpolation")
                result = (1 - t) * flat1 + t * flat2
            else:
                # Use the first vector
                result = flat1.clone()
        else:
            # Perform SLERP
            sin_theta = torch.sin(theta)
            
            # Handle sign of dot product
            if dot_product < 0:
                flat2 = -flat2
                
            # SLERP formula
            coeff1 = torch.sin((1 - t) * theta) / sin_theta
            coeff2 = torch.sin(t * theta) / sin_theta
            
            result = coeff1 * flat1 + coeff2 * flat2
        
        # Normalize result if requested
        if self.normalize_after:
            result_norm = torch.norm(result)
            if result_norm > self.epsilon:
                result = result / result_norm
                
                # Restore original magnitude (average of input magnitudes)
                if self.normalize_before:
                    target_norm = (norm1 + norm2) / 2
                    result = result * target_norm
        
        # Reshape back to original shape
        return result.reshape(original_shape)
    
    def slerp_with_multiple_points(self, 
                                  models: List[nn.Module], 
                                  t_values: List[float]) -> List[nn.Module]:
        """
        Perform SLERP interpolation at multiple points along the path.
        
        Useful for creating smooth transitions or animation sequences.
        
        Args:
            models: List of models defining the path (minimum 2)
            t_values: List of interpolation points in [0, 1]
            
        Returns:
            List of interpolated models at each t value
        """
        if len(models) < 2:
            raise ValueError("Need at least 2 models for interpolation")
            
        results = []
        
        for t in t_values:
            if not 0 <= t <= 1:
                raise ValueError(f"t value must be in [0, 1], got {t}")
                
            if len(models) == 2:
                # Simple case: interpolate between two models
                result = self.interpolate(models[0], models[1], t)
            else:
                # Multiple models: find appropriate segment and interpolate
                segment_length = 1.0 / (len(models) - 1)
                segment_index = min(int(t / segment_length), len(models) - 2)
                local_t = (t - segment_index * segment_length) / segment_length
                
                result = self.interpolate(models[segment_index], models[segment_index + 1], local_t)
                
            results.append(result)
            
        return results
    
    def calculate_geodesic_distance(self, model1: nn.Module, model2: nn.Module) -> float:
        """
        Calculate geodesic distance between two models on the parameter manifold.
        
        This provides a measure of how "far apart" two models are in the
        spherical parameter space.
        
        Args:
            model1: First model
            model2: Second model
            
        Returns:
            Geodesic distance
        """
        total_distance = 0.0
        total_params = 0
        
        params1 = dict(model1.named_parameters())
        params2 = dict(model2.named_parameters())
        
        for name in params1:
            if name in params2:
                # Calculate angle between parameter vectors
                flat1 = params1[name].data.flatten()
                flat2 = params2[name].data.flatten()
                
                # Normalize
                norm1 = torch.norm(flat1)
                norm2 = torch.norm(flat2)
                
                if norm1 > self.epsilon and norm2 > self.epsilon:
                    flat1_norm = flat1 / norm1
                    flat2_norm = flat2 / norm2
                    
                    # Calculate angle
                    dot_product = torch.dot(flat1_norm, flat2_norm).clamp(-1 + self.epsilon, 1 - self.epsilon)
                    angle = torch.acos(torch.abs(dot_product))
                    
                    total_distance += angle.item()
                    total_params += 1
                    
        return total_distance / total_params if total_params > 0 else 0.0
    
    def find_optimal_interpolation_path(self, 
                                      start_model: nn.Module, 
                                      end_model: nn.Module, 
                                      num_waypoints: int = 5) -> List[nn.Module]:
        """
        Find optimal interpolation path between two models using SLERP.
        
        Creates intermediate waypoints that minimize the total path length
        on the parameter manifold.
        
        Args:
            start_model: Starting model
            end_model: Target model
            num_waypoints: Number of intermediate waypoints
            
        Returns:
            List of models forming optimal path
        """
        if num_waypoints < 1:
            raise ValueError("Number of waypoints must be at least 1")
            
        # Create evenly spaced t values
        t_values = np.linspace(0, 1, num_waypoints + 2)  # +2 for start and end
        
        # Generate waypoints using SLERP
        waypoints = []
        for t in t_values:
            waypoint = self.interpolate(start_model, end_model, t)
            waypoints.append(waypoint)
            
        return waypoints
    
    def adaptive_slerp(self, 
                      model1: nn.Module, 
                      model2: nn.Module, 
                      target_distance: float,
                      max_steps: int = 10) -> nn.Module:
        """
        Perform adaptive SLERP to achieve a target geodesic distance.
        
        Iteratively adjusts the interpolation parameter to achieve a model
        that is approximately the target distance from model1.
        
        Args:
            model1: Starting model
            model2: Target direction model
            target_distance: Desired geodesic distance from model1
            max_steps: Maximum optimization steps
            
        Returns:
            Model at approximately target distance
        """
        if target_distance <= 0:
            return self._clone_model(model1)
            
        # Binary search for optimal t
        t_low, t_high = 0.0, 1.0
        best_t = 0.5
        
        for step in range(max_steps):
            current_model = self.interpolate(model1, model2, best_t)
            current_distance = self.calculate_geodesic_distance(model1, current_model)
            
            if abs(current_distance - target_distance) < self.epsilon:
                break
                
            if current_distance < target_distance:
                t_low = best_t
            else:
                t_high = best_t
                
            best_t = (t_low + t_high) / 2
            
        return self.interpolate(model1, model2, best_t)
    
    def _clone_model(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of a model."""
        import copy
        return copy.deepcopy(model)
    
    def get_interpolation_statistics(self, 
                                   models: List[nn.Module]) -> Dict[str, Any]:
        """
        Calculate statistics about interpolation quality.
        
        Args:
            models: List of models to analyze
            
        Returns:
            Dictionary with interpolation statistics
        """
        if len(models) < 2:
            return {}
            
        # Calculate pairwise distances
        distances = []
        for i in range(len(models) - 1):
            dist = self.calculate_geodesic_distance(models[i], models[i + 1])
            distances.append(dist)
            
        # Calculate path smoothness (variance in step sizes)
        distance_variance = np.var(distances) if len(distances) > 1 else 0.0
        
        # Calculate total path length
        total_length = sum(distances)
        
        # Calculate direct distance
        direct_distance = self.calculate_geodesic_distance(models[0], models[-1])
        
        # Path efficiency (direct distance / total path length)
        efficiency = direct_distance / total_length if total_length > 0 else 1.0
        
        return {
            'num_models': len(models),
            'pairwise_distances': distances,
            'average_step_size': np.mean(distances) if distances else 0.0,
            'step_size_variance': distance_variance,
            'total_path_length': total_length,
            'direct_distance': direct_distance,
            'path_efficiency': efficiency,
            'smoothness_score': 1.0 / (1.0 + distance_variance)  # Higher is smoother
        }

# Factory function for easy creation
def create_slerp_operator(config: Optional[Dict[str, Any]] = None) -> SLERPOperator:
    """
    Factory function to create SLERP operator with configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured SLERP operator
    """
    if config is None:
        config = {}
        
    return SLERPOperator(
        epsilon=config.get('epsilon', 1e-8),
        normalize_before=config.get('normalize_before', True),
        normalize_after=config.get('normalize_after', True),
        fallback_to_linear=config.get('fallback_to_linear', True),
        device=config.get('device', 'cpu')
    )
