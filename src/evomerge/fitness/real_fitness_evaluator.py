"""
Real Fitness Evaluator for Evolutionary Model Merging.

Implements genuine fitness evaluation metrics:
- Model performance on validation tasks
- Parameter efficiency and sparsity
- Inference speed and memory usage
- Task-specific accuracy metrics
- Multi-objective optimization support

NO MOCK CODE - Real evaluation with measurable metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
import logging
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
import psutil
import gc

logger = logging.getLogger(__name__)

@dataclass
class FitnessMetrics:
    """Container for fitness evaluation metrics."""
    accuracy: float = 0.0
    loss: float = float('inf')
    perplexity: float = float('inf')
    inference_speed: float = 0.0  # samples per second
    memory_usage: float = 0.0  # MB
    parameter_efficiency: float = 0.0
    sparsity_ratio: float = 0.0
    task_specific_scores: Dict[str, float] = None
    composite_fitness: float = 0.0
    
    def __post_init__(self):
        if self.task_specific_scores is None:
            self.task_specific_scores = {}

class FitnessObjective(ABC):
    """Abstract base class for fitness objectives."""
    
    @abstractmethod
    def evaluate(self, model: nn.Module, data_loader: Optional[torch.utils.data.DataLoader] = None) -> float:
        """Evaluate the objective for a given model."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the objective."""
        pass
    
    @property
    def weight(self) -> float:
        """Weight of this objective in multi-objective optimization."""
        return 1.0

class AccuracyObjective(FitnessObjective):
    """Accuracy-based fitness objective."""
    
    def __init__(self, data_loader: torch.utils.data.DataLoader, device: str = "cpu"):
        self.data_loader = data_loader
        self.device = device
    
    def evaluate(self, model: nn.Module, data_loader: Optional[torch.utils.data.DataLoader] = None) -> float:
        """Evaluate classification accuracy."""
        loader = data_loader or self.data_loader
        if loader is None:
            return 0.0
            
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                if batch_idx >= 10:  # Limit evaluation for speed
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                
                try:
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
                except Exception as e:
                    logger.warning(f"Accuracy evaluation failed: {e}")
                    return 0.0
                    
        return correct / total if total > 0 else 0.0
    
    @property
    def name(self) -> str:
        return "accuracy"

class PerplexityObjective(FitnessObjective):
    """Perplexity-based fitness objective for language models."""
    
    def __init__(self, data_loader: torch.utils.data.DataLoader, device: str = "cpu"):
        self.data_loader = data_loader
        self.device = device
    
    def evaluate(self, model: nn.Module, data_loader: Optional[torch.utils.data.DataLoader] = None) -> float:
        """Evaluate perplexity (lower is better, so return negative)."""
        loader = data_loader or self.data_loader
        if loader is None:
            return -100.0  # Penalty for no data
            
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                if batch_idx >= 10:  # Limit evaluation for speed
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                
                try:
                    output = model(data)
                    if output.dim() == 3:  # Sequence model
                        output = output.view(-1, output.size(-1))
                        target = target.view(-1)
                        
                    loss = F.cross_entropy(output, target)
                    total_loss += loss.item() * target.size(0)
                    total_tokens += target.size(0)
                except Exception as e:
                    logger.warning(f"Perplexity evaluation failed: {e}")
                    return -100.0
                    
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 100.0
        perplexity = math.exp(avg_loss)
        
        # Return negative log perplexity (higher is better)
        return -math.log(perplexity + 1e-8)
    
    @property
    def name(self) -> str:
        return "perplexity"

class EfficiencyObjective(FitnessObjective):
    """Parameter efficiency objective."""
    
    def __init__(self, reference_param_count: int):
        self.reference_param_count = reference_param_count
    
    def evaluate(self, model: nn.Module, data_loader: Optional[torch.utils.data.DataLoader] = None) -> float:
        """Evaluate parameter efficiency (fewer parameters is better)."""
        total_params = sum(p.numel() for p in model.parameters())
        nonzero_params = sum(torch.count_nonzero(p).item() for p in model.parameters())
        
        # Efficiency = (nonzero params / total params) * (reference / total)
        sparsity_bonus = nonzero_params / total_params if total_params > 0 else 0.0
        size_bonus = self.reference_param_count / total_params if total_params > 0 else 0.0
        
        return sparsity_bonus * size_bonus
    
    @property
    def name(self) -> str:
        return "efficiency"

class InferenceSpeedObjective(FitnessObjective):
    """Inference speed objective."""
    
    def __init__(self, sample_input: torch.Tensor, device: str = "cpu", num_trials: int = 5):
        self.sample_input = sample_input.to(device)
        self.device = device
        self.num_trials = num_trials
    
    def evaluate(self, model: nn.Module, data_loader: Optional[torch.utils.data.DataLoader] = None) -> float:
        """Evaluate inference speed (samples per second)."""
        model.eval()
        model = model.to(self.device)
        
        # Warmup
        with torch.no_grad():
            try:
                for _ in range(2):
                    _ = model(self.sample_input)
            except Exception:
                return 0.0
        
        # Timing
        times = []
        batch_size = self.sample_input.size(0)
        
        with torch.no_grad():
            for _ in range(self.num_trials):
                start_time = time.time()
                try:
                    _ = model(self.sample_input)
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    end_time = time.time()
                    times.append(end_time - start_time)
                except Exception:
                    return 0.0
        
        if not times:
            return 0.0
            
        avg_time = np.mean(times)
        samples_per_second = batch_size / avg_time if avg_time > 0 else 0.0
        
        # Normalize to [0, 1] range (assuming max 1000 samples/sec)
        return min(samples_per_second / 1000.0, 1.0)
    
    @property
    def name(self) -> str:
        return "inference_speed"

class MemoryUsageObjective(FitnessObjective):
    """Memory usage objective (lower is better)."""
    
    def __init__(self, sample_input: torch.Tensor, device: str = "cpu"):
        self.sample_input = sample_input.to(device)
        self.device = device
    
    def evaluate(self, model: nn.Module, data_loader: Optional[torch.utils.data.DataLoader] = None) -> float:
        """Evaluate memory usage during inference."""
        model.eval()
        model = model.to(self.device)
        
        # Clear cache
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        else:
            initial_memory = psutil.Process().memory_info().rss
        
        # Forward pass
        with torch.no_grad():
            try:
                _ = model(self.sample_input)
                if self.device == "cuda":
                    peak_memory = torch.cuda.max_memory_allocated()
                    memory_used = (peak_memory - initial_memory) / 1024**2  # MB
                else:
                    current_memory = psutil.Process().memory_info().rss
                    memory_used = (current_memory - initial_memory) / 1024**2  # MB
            except Exception:
                return 0.0
        
        # Return inverse of memory usage (higher is better)
        return 1.0 / (1.0 + memory_used / 100.0)  # Normalize assuming 100MB baseline
    
    @property
    def name(self) -> str:
        return "memory_efficiency"

class RealFitnessEvaluator:
    """
    Real fitness evaluator for evolutionary model merging.
    
    Combines multiple objectives into a single fitness score:
    - Task performance (accuracy, perplexity)
    - Efficiency metrics (parameters, speed, memory)
    - Robustness measures
    - Custom objectives
    """
    
    def __init__(self,
                 objectives: List[FitnessObjective],
                 objective_weights: Optional[List[float]] = None,
                 aggregation_method: str = "weighted_sum",
                 cache_evaluations: bool = True,
                 device: str = "cpu"):
        """
        Initialize fitness evaluator.
        
        Args:
            objectives: List of fitness objectives to evaluate
            objective_weights: Weights for each objective (normalized automatically)
            aggregation_method: Method to combine objectives ("weighted_sum", "product", "min")
            cache_evaluations: Whether to cache evaluation results
            device: Device for computations
        """
        self.objectives = objectives
        self.device = device
        self.aggregation_method = aggregation_method
        self.cache_evaluations = cache_evaluations
        
        # Setup objective weights
        if objective_weights is None:
            self.objective_weights = [obj.weight for obj in objectives]
        else:
            if len(objective_weights) != len(objectives):
                raise ValueError("Number of weights must match number of objectives")
            self.objective_weights = objective_weights
            
        # Normalize weights
        total_weight = sum(self.objective_weights)
        if total_weight > 0:
            self.objective_weights = [w / total_weight for w in self.objective_weights]
        else:
            self.objective_weights = [1.0 / len(objectives)] * len(objectives)
            
        # Evaluation cache
        self.evaluation_cache = {} if cache_evaluations else None
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"Initialized RealFitnessEvaluator with {len(objectives)} objectives")
        logger.info(f"Objectives: {[obj.name for obj in objectives]}")
        logger.info(f"Weights: {self.objective_weights}")
    
    def evaluate(self, model: nn.Module, data_loader: Optional[torch.utils.data.DataLoader] = None) -> FitnessMetrics:
        """
        Evaluate fitness of a model.
        
        Args:
            model: Model to evaluate
            data_loader: Optional data loader for evaluation
            
        Returns:
            FitnessMetrics with detailed evaluation results
        """
        # Generate cache key if caching enabled
        cache_key = None
        if self.cache_evaluations:
            cache_key = self._generate_cache_key(model)
            if cache_key in self.evaluation_cache:
                self.cache_hits += 1
                return self.evaluation_cache[cache_key]
            self.cache_misses += 1
        
        # Evaluate each objective
        objective_scores = []
        task_specific_scores = {}
        
        for obj in self.objectives:
            try:
                score = obj.evaluate(model, data_loader)
                objective_scores.append(score)
                task_specific_scores[obj.name] = score
            except Exception as e:
                logger.warning(f"Objective {obj.name} evaluation failed: {e}")
                objective_scores.append(0.0)
                task_specific_scores[obj.name] = 0.0
        
        # Aggregate objectives
        composite_fitness = self._aggregate_objectives(objective_scores)
        
        # Calculate additional metrics
        total_params = sum(p.numel() for p in model.parameters())
        nonzero_params = sum(torch.count_nonzero(p).item() for p in model.parameters())
        sparsity_ratio = 1.0 - (nonzero_params / total_params) if total_params > 0 else 0.0
        
        # Create metrics object
        metrics = FitnessMetrics(
            accuracy=task_specific_scores.get('accuracy', 0.0),
            loss=-task_specific_scores.get('perplexity', 0.0),  # Convert back from negative log
            perplexity=math.exp(-task_specific_scores.get('perplexity', 0.0)) if 'perplexity' in task_specific_scores else float('inf'),
            inference_speed=task_specific_scores.get('inference_speed', 0.0),
            memory_usage=1.0 / task_specific_scores.get('memory_efficiency', 1e-6) if 'memory_efficiency' in task_specific_scores else 0.0,
            parameter_efficiency=task_specific_scores.get('efficiency', 0.0),
            sparsity_ratio=sparsity_ratio,
            task_specific_scores=task_specific_scores,
            composite_fitness=composite_fitness
        )
        
        # Cache result
        if self.cache_evaluations and cache_key:
            self.evaluation_cache[cache_key] = metrics
            
        return metrics
    
    def batch_evaluate(self, models: List[nn.Module], data_loader: Optional[torch.utils.data.DataLoader] = None) -> List[FitnessMetrics]:
        """
        Evaluate fitness for multiple models.
        
        Args:
            models: List of models to evaluate
            data_loader: Optional data loader for evaluation
            
        Returns:
            List of FitnessMetrics for each model
        """
        results = []
        
        for i, model in enumerate(models):
            logger.debug(f"Evaluating model {i+1}/{len(models)}")
            metrics = self.evaluate(model, data_loader)
            results.append(metrics)
            
        return results
    
    def _aggregate_objectives(self, scores: List[float]) -> float:
        """Aggregate multiple objective scores into single fitness."""
        if not scores:
            return 0.0
            
        if self.aggregation_method == "weighted_sum":
            return sum(w * s for w, s in zip(self.objective_weights, scores))
        elif self.aggregation_method == "product":
            # Geometric mean with weights
            product = 1.0
            for w, s in zip(self.objective_weights, scores):
                if s > 0:
                    product *= s ** w
                else:
                    return 0.0  # Any zero score makes product zero
            return product
        elif self.aggregation_method == "min":
            # Weighted minimum
            weighted_scores = [w * s for w, s in zip(self.objective_weights, scores)]
            return min(weighted_scores)
        else:
            # Default to weighted sum
            return sum(w * s for w, s in zip(self.objective_weights, scores))
    
    def _generate_cache_key(self, model: nn.Module) -> str:
        """Generate cache key for model."""
        # Use hash of model parameters
        param_hash = 0
        for param in model.parameters():
            if param.requires_grad:
                param_hash ^= hash(param.data.cpu().numpy().tobytes())
        return str(param_hash)
    
    def add_objective(self, objective: FitnessObjective, weight: float = 1.0):
        """Add new objective to evaluator."""
        self.objectives.append(objective)
        self.objective_weights.append(weight)
        
        # Renormalize weights
        total_weight = sum(self.objective_weights)
        if total_weight > 0:
            self.objective_weights = [w / total_weight for w in self.objective_weights]
            
        logger.info(f"Added objective: {objective.name} with weight {weight}")
    
    def remove_objective(self, objective_name: str):
        """Remove objective by name."""
        for i, obj in enumerate(self.objectives):
            if obj.name == objective_name:
                del self.objectives[i]
                del self.objective_weights[i]
                
                # Renormalize weights
                total_weight = sum(self.objective_weights)
                if total_weight > 0:
                    self.objective_weights = [w / total_weight for w in self.objective_weights]
                    
                logger.info(f"Removed objective: {objective_name}")
                return
                
        logger.warning(f"Objective {objective_name} not found")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_enabled': self.cache_evaluations,
            'cache_size': len(self.evaluation_cache) if self.evaluation_cache else 0,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }
    
    def clear_cache(self):
        """Clear evaluation cache."""
        if self.evaluation_cache:
            self.evaluation_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            logger.info("Cleared fitness evaluation cache")
    
    def get_objective_weights(self) -> Dict[str, float]:
        """Get current objective weights."""
        return {obj.name: weight for obj, weight in zip(self.objectives, self.objective_weights)}
    
    def set_objective_weights(self, weights: Dict[str, float]):
        """Set objective weights by name."""
        for i, obj in enumerate(self.objectives):
            if obj.name in weights:
                self.objective_weights[i] = weights[obj.name]
                
        # Renormalize
        total_weight = sum(self.objective_weights)
        if total_weight > 0:
            self.objective_weights = [w / total_weight for w in self.objective_weights]
            
        logger.info(f"Updated objective weights: {self.get_objective_weights()}")

# Factory functions
def create_classification_evaluator(data_loader: torch.utils.data.DataLoader,
                                   sample_input: torch.Tensor,
                                   reference_param_count: int,
                                   device: str = "cpu") -> RealFitnessEvaluator:
    """Create fitness evaluator for classification tasks."""
    objectives = [
        AccuracyObjective(data_loader, device),
        EfficiencyObjective(reference_param_count),
        InferenceSpeedObjective(sample_input, device),
        MemoryUsageObjective(sample_input, device)
    ]
    
    weights = [0.5, 0.2, 0.15, 0.15]  # Prioritize accuracy
    
    return RealFitnessEvaluator(objectives, weights, device=device)

def create_language_model_evaluator(data_loader: torch.utils.data.DataLoader,
                                   sample_input: torch.Tensor,
                                   reference_param_count: int,
                                   device: str = "cpu") -> RealFitnessEvaluator:
    """Create fitness evaluator for language models."""
    objectives = [
        PerplexityObjective(data_loader, device),
        EfficiencyObjective(reference_param_count),
        InferenceSpeedObjective(sample_input, device),
        MemoryUsageObjective(sample_input, device)
    ]
    
    weights = [0.6, 0.2, 0.1, 0.1]  # Prioritize perplexity
    
    return RealFitnessEvaluator(objectives, weights, device=device)

def create_efficiency_evaluator(sample_input: torch.Tensor,
                               reference_param_count: int,
                               device: str = "cpu") -> RealFitnessEvaluator:
    """Create fitness evaluator focused on efficiency."""
    objectives = [
        EfficiencyObjective(reference_param_count),
        InferenceSpeedObjective(sample_input, device),
        MemoryUsageObjective(sample_input, device)
    ]
    
    weights = [0.5, 0.3, 0.2]  # Balanced efficiency focus
    
    return RealFitnessEvaluator(objectives, weights, device=device)
