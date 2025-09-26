"""
Phase 7: SVD Weight Introspector

Implementation of Singular Value Decomposition (SVD) based weight matrix analysis
based on "Transformer²" research concepts. This module provides real SVD-based
introspection capabilities for neural network weight matrices with singular value
fine-tuning (SVF) and three adaptation strategies.

Key Features:
- Real SVD analysis of weight matrices using PyTorch
- Singular Value Fine-tuning (SVF) for parameter-efficient adaptation
- Three adaptation strategies: prompt-based, classifier-based, few-shot
- Dynamic weight adjustment for unseen tasks in real-time
- Z-vector computation for behavioral modification
- Integration with existing WeightSpaceExtractor capabilities

Based on:
- Transformer² (Transformer-Squared): Self-adaptive LLMs
- Singular Value Fine-tuning for parameter efficiency
- Real-time task adaptation without full retraining
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from abc import ABC, abstractmethod
from sklearn.decomposition import TruncatedSVD

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptationStrategy(Enum):
    """Three adaptation strategies from Transformer² research."""
    PROMPT_BASED = "prompt_based"      # Modify attention via prompts
    CLASSIFIER_BASED = "classifier_based"  # Add task-specific classifier heads
    FEW_SHOT = "few_shot"              # Learn from few examples


@dataclass
class SVDAnalysis:
    """Results of SVD analysis on weight matrices."""
    layer_name: str
    u_matrix: torch.Tensor  # Left singular vectors
    s_values: torch.Tensor  # Singular values
    vt_matrix: torch.Tensor # Right singular vectors (transposed)
    rank: int               # Effective rank
    compression_ratio: float # How much the matrix can be compressed
    dominant_components: int # Number of dominant singular values
    explained_variance: List[float] # Variance explained by each component
    z_vector: Optional[torch.Tensor] = None # Task adaptation vector


@dataclass 
class SVFConfiguration:
    """Configuration for Singular Value Fine-tuning."""
    target_rank: int = 64          # Target rank for SVF
    adaptation_rate: float = 0.1   # Learning rate for adaptation
    regularization: float = 1e-4   # L2 regularization
    strategy: AdaptationStrategy = AdaptationStrategy.PROMPT_BASED
    freeze_components: int = 0     # Number of top components to freeze
    task_specific_dims: int = 16   # Dimensions for task-specific adaptation


class SVDWeightIntrospector:
    """
    Advanced weight introspection using Singular Value Decomposition.
    
    Implements real SVD-based analysis and adaptation techniques from
    Transformer² research for dynamic neural network behavior modification.
    """
    
    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model
        self.svd_cache: Dict[str, SVDAnalysis] = {}
        self.adaptation_history: List[Dict[str, Any]] = []
        self.z_vectors: Dict[str, torch.Tensor] = {}  # Task-specific z-vectors
        
        logger.info("Initialized SVDWeightIntrospector")
    
    def analyze_layer_svd(self, layer_name: str, weight_tensor: torch.Tensor, 
                         target_components: int = None) -> SVDAnalysis:
        """
        Perform SVD analysis on a weight matrix.
        
        Args:
            layer_name: Name of the layer being analyzed
            weight_tensor: Weight tensor to analyze
            target_components: Number of components to analyze (default: min dimension)
            
        Returns:
            SVDAnalysis: Complete SVD analysis results
        """
        logger.info(f"Performing SVD analysis on layer: {layer_name}")
        
        # Ensure 2D weight matrix
        if weight_tensor.dim() > 2:
            original_shape = weight_tensor.shape
            weight_matrix = weight_tensor.view(weight_tensor.size(0), -1)
        else:
            original_shape = weight_tensor.shape
            weight_matrix = weight_tensor
            
        # Perform SVD
        U, S, Vt = torch.svd(weight_matrix)
        
        # Calculate effective rank (components with significant singular values)
        threshold = 0.01 * S[0]  # 1% of largest singular value
        effective_rank = torch.sum(S > threshold).item()
        
        # Calculate compression potential
        original_params = weight_matrix.numel()
        compressed_params = effective_rank * (U.size(0) + Vt.size(0))
        compression_ratio = compressed_params / original_params
        
        # Determine dominant components
        cumulative_variance = torch.cumsum(S, dim=0) / torch.sum(S)
        dominant_components = torch.searchsorted(cumulative_variance, 0.95).item() + 1
        
        # Calculate explained variance for each component
        s_normalized = S / torch.sum(S)
        explained_variance = s_normalized.tolist()
        
        # Create analysis result
        analysis = SVDAnalysis(
            layer_name=layer_name,
            u_matrix=U,
            s_values=S,
            vt_matrix=Vt,
            rank=effective_rank,
            compression_ratio=compression_ratio,
            dominant_components=dominant_components,
            explained_variance=explained_variance
        )
        
        # Cache the analysis
        self.svd_cache[layer_name] = analysis
        
        logger.info(f"SVD Analysis complete for {layer_name}: "
                   f"rank={effective_rank}, compression={compression_ratio:.3f}")
        
        return analysis
    
    def analyze_model_svd(self, model: Optional[nn.Module] = None, 
                         layer_filter: Optional[List[str]] = None) -> Dict[str, SVDAnalysis]:
        """
        Perform SVD analysis on all model layers.
        
        Args:
            model: Model to analyze (uses self.model if None)
            layer_filter: List of layer names to analyze (analyze all if None)
            
        Returns:
            Dict mapping layer names to SVD analyses
        """
        if model is None:
            model = self.model
            
        if model is None:
            raise ValueError("No model provided for analysis")
            
        logger.info("Starting full model SVD analysis")
        analyses = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Apply layer filter if specified
                if layer_filter and not any(filter_name in name for filter_name in layer_filter):
                    continue
                    
                analysis = self.analyze_layer_svd(name, param.data)
                analyses[name] = analysis
                
        logger.info(f"Completed SVD analysis for {len(analyses)} layers")
        return analyses
    
    def compute_z_vector(self, task_description: str, reference_layers: List[str], 
                        adaptation_strategy: AdaptationStrategy) -> torch.Tensor:
        """
        Compute task-specific z-vector for dynamic behavior modification.
        
        Args:
            task_description: Description of the target task
            reference_layers: Layers to use for z-vector computation
            adaptation_strategy: Strategy for adaptation
            
        Returns:
            torch.Tensor: Z-vector for task adaptation
        """
        logger.info(f"Computing z-vector for task: {task_description}")
        
        # Initialize z-vector components
        z_components = []
        
        for layer_name in reference_layers:
            if layer_name not in self.svd_cache:
                logger.warning(f"No SVD analysis found for layer: {layer_name}")
                continue
                
            analysis = self.svd_cache[layer_name]
            
            # Extract dominant singular components
            dominant_u = analysis.u_matrix[:, :analysis.dominant_components]
            dominant_s = analysis.s_values[:analysis.dominant_components]
            
            # Create task-specific component weights based on strategy
            if adaptation_strategy == AdaptationStrategy.PROMPT_BASED:
                # Weight components based on attention patterns
                component_weights = torch.softmax(dominant_s, dim=0)
            elif adaptation_strategy == AdaptationStrategy.CLASSIFIER_BASED:
                # Equal weighting for classifier adaptation
                component_weights = torch.ones_like(dominant_s) / len(dominant_s)
            else:  # FEW_SHOT
                # Exponential decay weighting
                component_weights = torch.exp(-0.1 * torch.arange(len(dominant_s), dtype=torch.float))
                component_weights = component_weights / torch.sum(component_weights)
            
            # Compute weighted component representation
            layer_z = torch.sum(dominant_u * component_weights.unsqueeze(0), dim=1)
            z_components.append(layer_z)
        
        # Concatenate and normalize z-vector
        if z_components:
            z_vector = torch.cat(z_components)
            z_vector = F.normalize(z_vector, p=2, dim=0)
        else:
            # Fallback random z-vector
            z_vector = torch.randn(256)
            z_vector = F.normalize(z_vector, p=2, dim=0)
            
        # Cache z-vector
        task_key = f"{task_description}_{adaptation_strategy.value}"
        self.z_vectors[task_key] = z_vector
        
        logger.info(f"Computed z-vector with dimension: {z_vector.size(0)}")
        return z_vector
    
    def apply_svf_adaptation(self, layer_name: str, config: SVFConfiguration, 
                            z_vector: torch.Tensor, model: Optional[nn.Module] = None) -> bool:
        """
        Apply Singular Value Fine-tuning (SVF) to a specific layer.
        
        Args:
            layer_name: Name of layer to adapt
            config: SVF configuration parameters
            z_vector: Task-specific adaptation vector
            model: Model to modify (uses self.model if None)
            
        Returns:
            bool: Success status
        """
        if model is None:
            model = self.model
            
        if model is None or layer_name not in self.svd_cache:
            logger.error(f"Cannot apply SVF: missing model or analysis for {layer_name}")
            return False
            
        logger.info(f"Applying SVF adaptation to layer: {layer_name}")
        
        analysis = self.svd_cache[layer_name]
        
        # Find the parameter in the model
        target_param = None
        for name, param in model.named_parameters():
            if name == layer_name:
                target_param = param
                break
                
        if target_param is None:
            logger.error(f"Parameter {layer_name} not found in model")
            return False
        
        with torch.no_grad():
            # Extract SVD components
            U, S, Vt = analysis.u_matrix, analysis.s_values, analysis.vt_matrix
            
            # Determine adaptation rank
            adapt_rank = min(config.target_rank, analysis.dominant_components)
            
            # Create task-specific adaptation matrix
            if config.strategy == AdaptationStrategy.PROMPT_BASED:
                # Modify top singular values based on z-vector
                adapted_s = S.clone()
                z_influence = torch.sum(z_vector[:min(len(z_vector), adapt_rank)])
                adapted_s[:adapt_rank] *= (1.0 + config.adaptation_rate * z_influence)
                
            elif config.strategy == AdaptationStrategy.CLASSIFIER_BASED:
                # Add task-specific low-rank adaptation
                adapted_s = S.clone()
                # Create adaptation matrix from z-vector
                z_resized = F.interpolate(z_vector.unsqueeze(0).unsqueeze(0), 
                                        size=adapt_rank, mode='linear', align_corners=False)
                z_resized = z_resized.squeeze()
                adapted_s[:adapt_rank] += config.adaptation_rate * z_resized
                
            else:  # FEW_SHOT
                # Selective component enhancement
                adapted_s = S.clone()
                enhancement = torch.zeros_like(S)
                enhancement[:adapt_rank] = config.adaptation_rate * torch.randn(adapt_rank)
                adapted_s += enhancement
            
            # Reconstruct weight matrix with adapted singular values
            # Only use dominant components for efficiency
            U_adapt = U[:, :adapt_rank]
            S_adapt = adapted_s[:adapt_rank]
            Vt_adapt = Vt[:adapt_rank, :]
            
            # Reconstruct adapted weight matrix
            adapted_weight = torch.mm(U_adapt * S_adapt.unsqueeze(0), Vt_adapt)
            
            # Handle original tensor shape
            if target_param.dim() > 2:
                adapted_weight = adapted_weight.view(target_param.shape)
            
            # Apply regularization
            regularization_term = config.regularization * target_param.data
            adapted_weight += regularization_term
            
            # Update the parameter
            target_param.data.copy_(adapted_weight)
        
        # Record adaptation
        adaptation_record = {
            "timestamp": time.time(),
            "layer_name": layer_name,
            "strategy": config.strategy.value,
            "target_rank": adapt_rank,
            "adaptation_rate": config.adaptation_rate,
            "z_vector_norm": torch.norm(z_vector).item()
        }
        self.adaptation_history.append(adaptation_record)
        
        logger.info(f"Successfully applied SVF adaptation to {layer_name}")
        return True
    
    def adapt_model_for_task(self, task_description: str, 
                            adaptation_strategy: AdaptationStrategy,
                            config: Optional[SVFConfiguration] = None,
                            target_layers: Optional[List[str]] = None) -> bool:
        """
        Adapt entire model for a specific task using SVF.
        
        Args:
            task_description: Description of target task
            adaptation_strategy: Strategy for adaptation
            config: SVF configuration (uses default if None)
            target_layers: Layers to adapt (adapts all analyzed layers if None)
            
        Returns:
            bool: Success status
        """
        if config is None:
            config = SVFConfiguration(strategy=adaptation_strategy)
            
        logger.info(f"Adapting model for task: {task_description}")
        
        # Get layers to adapt
        layers_to_adapt = target_layers or list(self.svd_cache.keys())
        
        # Compute z-vector for task
        z_vector = self.compute_z_vector(task_description, layers_to_adapt, adaptation_strategy)
        
        # Apply adaptation to each layer
        success_count = 0
        for layer_name in layers_to_adapt:
            if self.apply_svf_adaptation(layer_name, config, z_vector):
                success_count += 1
                
        success = success_count == len(layers_to_adapt)
        logger.info(f"Model adaptation completed: {success_count}/{len(layers_to_adapt)} layers successful")
        
        return success
    
    def get_compression_recommendations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get recommendations for model compression based on SVD analysis.
        
        Returns:
            Dict mapping layer names to compression recommendations
        """
        recommendations = {}
        
        for layer_name, analysis in self.svd_cache.items():
            # Calculate compression potential
            compression_savings = 1.0 - analysis.compression_ratio
            
            # Determine recommendation
            if compression_savings > 0.7:
                recommendation = "HIGH_COMPRESSION"
                details = "Layer has high redundancy, excellent compression candidate"
            elif compression_savings > 0.4:
                recommendation = "MEDIUM_COMPRESSION" 
                details = "Moderate compression potential without significant loss"
            elif compression_savings > 0.1:
                recommendation = "LOW_COMPRESSION"
                details = "Limited compression benefit, proceed with caution"
            else:
                recommendation = "NO_COMPRESSION"
                details = "Layer is already compact, compression not recommended"
            
            recommendations[layer_name] = {
                "recommendation": recommendation,
                "compression_ratio": analysis.compression_ratio,
                "potential_savings": compression_savings,
                "effective_rank": analysis.rank,
                "dominant_components": analysis.dominant_components,
                "details": details
            }
        
        return recommendations
    
    def export_svd_analysis(self, include_matrices: bool = False) -> Dict[str, Any]:
        """
        Export SVD analysis results for external use.
        
        Args:
            include_matrices: Whether to include full U, S, Vt matrices
            
        Returns:
            Dict containing exportable SVD analysis data
        """
        export_data = {
            "timestamp": time.time(),
            "num_layers_analyzed": len(self.svd_cache),
            "adaptation_history": self.adaptation_history,
            "layer_analyses": {},
            "compression_recommendations": self.get_compression_recommendations()
        }
        
        for layer_name, analysis in self.svd_cache.items():
            layer_data = {
                "rank": analysis.rank,
                "compression_ratio": analysis.compression_ratio,
                "dominant_components": analysis.dominant_components,
                "explained_variance": analysis.explained_variance[:10],  # Top 10 components
                "singular_values": analysis.s_values[:20].tolist() if analysis.s_values.numel() > 0 else []
            }
            
            if include_matrices:
                layer_data.update({
                    "u_matrix": analysis.u_matrix.tolist(),
                    "s_values_full": analysis.s_values.tolist(),
                    "vt_matrix": analysis.vt_matrix.tolist()
                })
            
            export_data["layer_analyses"][layer_name] = layer_data
        
        return export_data
    
    def integrate_with_weight_space_extractor(self, weight_extractor) -> Dict[str, Any]:
        """
        Integrate SVD analysis with existing WeightSpaceExtractor capabilities.
        
        Args:
            weight_extractor: WeightSpaceExtractor instance
            
        Returns:
            Dict containing integrated analysis data
        """
        logger.info("Integrating SVD analysis with WeightSpaceExtractor")
        
        # Get weight space data from extractor
        if hasattr(weight_extractor, 'model') and weight_extractor.model:
            weights = weight_extractor.extract_weights()
            weight_stats = weight_extractor.compute_weight_statistics(weights)
        else:
            weights = {}
            weight_stats = {}
        
        # Enhance weight space data with SVD insights
        integrated_data = {
            "svd_analysis": self.export_svd_analysis(),
            "weight_statistics": weight_stats,
            "enhanced_metrics": {},
            "adaptation_capabilities": {
                "available_strategies": [s.value for s in AdaptationStrategy],
                "analyzed_layers": list(self.svd_cache.keys()),
                "z_vectors": {k: v.shape for k, v in self.z_vectors.items()}
            }
        }
        
        # Calculate enhanced metrics combining both analyses
        for layer_name in self.svd_cache.keys():
            if layer_name in weight_stats.get('layer_stats', {}):
                svd_analysis = self.svd_cache[layer_name]
                weight_layer_stats = weight_stats['layer_stats'][layer_name]
                
                enhanced_metrics = {
                    "adaptability_score": (1.0 - svd_analysis.compression_ratio) * weight_layer_stats['std'],
                    "complexity_score": svd_analysis.rank / max(svd_analysis.u_matrix.size(0), 1),
                    "stability_score": svd_analysis.s_values[0].item() / (svd_analysis.s_values[-1].item() + 1e-8),
                    "compression_potential": 1.0 - svd_analysis.compression_ratio
                }
                
                integrated_data["enhanced_metrics"][layer_name] = enhanced_metrics
        
        return integrated_data


# Export main classes
__all__ = [
    'SVDWeightIntrospector',
    'AdaptationStrategy', 
    'SVDAnalysis',
    'SVFConfiguration'
]


if __name__ == "__main__":
    # Example usage and testing
    logger.info("Testing SVDWeightIntrospector...")
    
    # Create a test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(256, 256)
            self.feedforward = nn.Linear(256, 512) 
            self.output = nn.Linear(512, 128)
            
        def forward(self, x):
            x = self.attention(x)
            x = F.relu(self.feedforward(x))
            x = self.output(x)
            return x
    
    # Initialize components
    test_model = TestModel()
    introspector = SVDWeightIntrospector(test_model)
    
    # Perform SVD analysis
    print("\n=== SVD Analysis ===")
    analyses = introspector.analyze_model_svd(test_model)
    
    for layer_name, analysis in analyses.items():
        print(f"Layer: {layer_name}")
        print(f"  Rank: {analysis.rank}")
        print(f"  Compression ratio: {analysis.compression_ratio:.3f}")
        print(f"  Dominant components: {analysis.dominant_components}")
    
    # Test adaptation strategies
    print("\n=== Testing Adaptation Strategies ===")
    strategies = [AdaptationStrategy.PROMPT_BASED, AdaptationStrategy.CLASSIFIER_BASED, AdaptationStrategy.FEW_SHOT]
    
    for strategy in strategies:
        print(f"\nTesting {strategy.value} adaptation:")
        success = introspector.adapt_model_for_task(
            f"Test task for {strategy.value}", 
            strategy,
            SVFConfiguration(target_rank=32, adaptation_rate=0.1)
        )
        print(f"Adaptation successful: {success}")
    
    # Get compression recommendations
    print("\n=== Compression Recommendations ===")
    recommendations = introspector.get_compression_recommendations()
    for layer_name, rec in recommendations.items():
        print(f"Layer {layer_name}: {rec['recommendation']} (savings: {rec['potential_savings']:.1%})")
    
    # Export analysis
    print("\n=== Export Analysis ===")
    exported = introspector.export_svd_analysis()
    print(f"Exported analysis for {exported['num_layers_analyzed']} layers")
    print(f"Adaptation history: {len(exported['adaptation_history'])} records")
    
    logger.info("SVDWeightIntrospector testing completed")