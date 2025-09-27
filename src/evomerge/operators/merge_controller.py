"""
Merge Controller - Orchestrates all evolutionary model merging operations.

Coordinates:
- SLERP (Spherical Linear Interpolation) operations
- TIES (Task-wise Internal Ensemble Selection) merging
- DARE (Drop And REscale) operations
- Evolutionary optimization algorithms
- Fitness evaluation and convergence detection

NO MOCK CODE - Real orchestration with mathematical correctness.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json

from .slerp_operator import SLERPOperator, create_slerp_operator
from .ties_operator import TIESOperator, create_ties_operator, TaskConfig
from .dare_operator import DAREOperator, create_dare_operator
from ..core.EvolutionaryEngine import EvolutionaryEngine, FitnessFunction, Individual, SelectionStrategy, CrossoverType, MutationType
from ..utils.model_operations import calculate_model_distance

logger = logging.getLogger(__name__)

class MergeStrategy(Enum):
    """Available merging strategies."""
    SLERP = "slerp"
    TIES = "ties"
    DARE = "dare"
    EVOLUTIONARY = "evolutionary"
    HYBRID = "hybrid"
    SEQUENTIAL = "sequential"
    PARALLEL_ENSEMBLE = "parallel_ensemble"

@dataclass
class MergeConfig:
    """Configuration for merge operations."""
    strategy: MergeStrategy
    num_generations: int = 50
    population_size: int = 20
    convergence_threshold: float = 1e-6
    convergence_patience: int = 10
    enable_checkpointing: bool = True
    checkpoint_interval: int = 5
    output_dir: Optional[str] = None
    
    # SLERP specific
    slerp_config: Optional[Dict[str, Any]] = None
    
    # TIES specific
    ties_config: Optional[Dict[str, Any]] = None
    task_configs: Optional[List[TaskConfig]] = None
    
    # DARE specific
    dare_config: Optional[Dict[str, Any]] = None
    
    # Evolutionary specific
    evolution_config: Optional[Dict[str, Any]] = None
    
    # Hybrid/Sequential specific
    sequence_strategies: Optional[List[MergeStrategy]] = None
    ensemble_weights: Optional[List[float]] = None

@dataclass
class MergeResult:
    """Result of merge operation."""
    merged_model: nn.Module
    strategy_used: MergeStrategy
    fitness_score: float
    convergence_generation: Optional[int]
    total_time: float
    statistics: Dict[str, Any]
    intermediate_results: Optional[List[nn.Module]] = None
    evolution_history: Optional[List[Dict[str, Any]]] = None

class CompositeFitnessFunction(FitnessFunction):
    """Composite fitness function combining multiple metrics."""
    
    def __init__(self, 
                 fitness_functions: List[Tuple[Callable[[nn.Module], float], float]],
                 device: str = "cpu"):
        """
        Initialize composite fitness function.
        
        Args:
            fitness_functions: List of (function, weight) tuples
            device: Device for computations
        """
        self.fitness_functions = fitness_functions
        self.device = device
        
        # Normalize weights
        total_weight = sum(weight for _, weight in fitness_functions)
        if total_weight > 0:
            self.fitness_functions = [(func, weight / total_weight) 
                                    for func, weight in fitness_functions]
    
    def evaluate(self, model: nn.Module) -> float:
        """Evaluate composite fitness."""
        total_score = 0.0
        
        for fitness_func, weight in self.fitness_functions:
            try:
                score = fitness_func(model)
                total_score += weight * score
            except Exception as e:
                logger.warning(f"Fitness function failed: {e}")
                # Use penalty for failed evaluation
                total_score += weight * (-1000.0)
                
        return total_score
    
    def batch_evaluate(self, models: List[nn.Module]) -> List[float]:
        """Batch evaluate multiple models."""
        return [self.evaluate(model) for model in models]

class MergeController:
    """
    Controller that orchestrates all model merging operations.
    
    Provides unified interface for:
    - SLERP interpolation
    - TIES merging
    - DARE dropout and rescaling
    - Evolutionary optimization
    - Hybrid approaches
    - Performance evaluation
    """
    
    def __init__(self, 
                 device: str = "cpu",
                 enable_profiling: bool = True,
                 cache_intermediate: bool = True):
        """
        Initialize merge controller.
        
        Args:
            device: Device for computations
            enable_profiling: Whether to enable performance profiling
            cache_intermediate: Whether to cache intermediate results
        """
        self.device = device
        self.enable_profiling = enable_profiling
        self.cache_intermediate = cache_intermediate
        
        # Initialize operators
        self.slerp_operator = None
        self.ties_operator = None
        self.dare_operator = None
        self.evolutionary_engine = None
        
        # Performance tracking
        self.profiling_data = {}
        self.merge_history = []
        
        # Caching
        self.intermediate_cache = {}
        
        logger.info(f"Initialized MergeController on device: {device}")
    
    def merge_models(self, 
                    models: List[nn.Module], 
                    config: MergeConfig,
                    fitness_function: Optional[FitnessFunction] = None,
                    model_weights: Optional[List[float]] = None) -> MergeResult:
        """
        Merge models using specified strategy.
        
        Args:
            models: List of models to merge
            config: Merge configuration
            fitness_function: Function to evaluate model quality
            model_weights: Weights for each model
            
        Returns:
            MergeResult with merged model and statistics
        """
        if len(models) < 2:
            raise ValueError("Need at least 2 models for merging")
            
        start_time = time.time()
        
        logger.info(f"Starting merge with strategy: {config.strategy.value}")
        logger.info(f"Merging {len(models)} models")
        
        # Setup default fitness function if not provided
        if fitness_function is None:
            fitness_function = self._create_default_fitness_function(models[0])
            
        # Dispatch to appropriate merge method
        if config.strategy == MergeStrategy.SLERP:
            result = self._merge_with_slerp(models, config, model_weights)
        elif config.strategy == MergeStrategy.TIES:
            result = self._merge_with_ties(models, config, model_weights)
        elif config.strategy == MergeStrategy.DARE:
            result = self._merge_with_dare(models, config, model_weights)
        elif config.strategy == MergeStrategy.EVOLUTIONARY:
            result = self._merge_with_evolution(models, config, fitness_function, model_weights)
        elif config.strategy == MergeStrategy.HYBRID:
            result = self._merge_with_hybrid(models, config, fitness_function, model_weights)
        elif config.strategy == MergeStrategy.SEQUENTIAL:
            result = self._merge_with_sequential(models, config, fitness_function, model_weights)
        elif config.strategy == MergeStrategy.PARALLEL_ENSEMBLE:
            result = self._merge_with_parallel_ensemble(models, config, fitness_function, model_weights)
        else:
            raise ValueError(f"Unknown merge strategy: {config.strategy}")
            
        # Calculate final fitness
        final_fitness = fitness_function.evaluate(result.merged_model)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Create final result
        final_result = MergeResult(
            merged_model=result.merged_model,
            strategy_used=config.strategy,
            fitness_score=final_fitness,
            convergence_generation=result.convergence_generation,
            total_time=total_time,
            statistics=result.statistics,
            intermediate_results=result.intermediate_results,
            evolution_history=result.evolution_history
        )
        
        # Store in history
        self.merge_history.append({
            'timestamp': time.time(),
            'strategy': config.strategy.value,
            'num_models': len(models),
            'fitness_score': final_fitness,
            'total_time': total_time
        })
        
        logger.info(f"Merge completed in {total_time:.2f}s with fitness: {final_fitness:.6f}")
        
        return final_result
    
    def _merge_with_slerp(self, 
                         models: List[nn.Module], 
                         config: MergeConfig,
                         model_weights: Optional[List[float]]) -> MergeResult:
        """Merge models using SLERP interpolation."""
        if self.slerp_operator is None:
            self.slerp_operator = create_slerp_operator(config.slerp_config)
            
        if len(models) == 2:
            # Simple pairwise SLERP
            weight = 0.5 if model_weights is None else model_weights[1] / sum(model_weights)
            merged_model = self.slerp_operator.interpolate(models[0], models[1], weight)
        else:
            # Multi-model SLERP
            if model_weights is None:
                model_weights = [1.0] * len(models)
            merged_model = self.slerp_operator.interpolate_batch(models, model_weights)
            
        # Calculate interpolation statistics
        stats = self.slerp_operator.get_interpolation_statistics(models + [merged_model])
        
        return MergeResult(
            merged_model=merged_model,
            strategy_used=MergeStrategy.SLERP,
            fitness_score=0.0,  # Will be calculated by caller
            convergence_generation=None,
            total_time=0.0,  # Will be calculated by caller
            statistics={'slerp_stats': stats}
        )
    
    def _merge_with_ties(self, 
                        models: List[nn.Module], 
                        config: MergeConfig,
                        model_weights: Optional[List[float]]) -> MergeResult:
        """Merge models using TIES algorithm."""
        if self.ties_operator is None:
            self.ties_operator = create_ties_operator(config.ties_config)
            
        # Use base model as first model if not specified
        base_model = models[0]
        
        ties_result = self.ties_operator.merge_models(
            models=models,
            base_model=base_model,
            task_configs=config.task_configs,
            model_weights=model_weights
        )
        
        return MergeResult(
            merged_model=ties_result.merged_model,
            strategy_used=MergeStrategy.TIES,
            fitness_score=ties_result.merge_quality_score,
            convergence_generation=None,
            total_time=0.0,
            statistics={
                'ties_stats': {
                    'selected_parameters': len(ties_result.selected_parameters),
                    'conflict_resolution': ties_result.conflict_resolution,
                    'task_contributions': ties_result.task_contributions,
                    'magnitude_statistics': ties_result.magnitude_statistics
                }
            }
        )
    
    def _merge_with_dare(self, 
                        models: List[nn.Module], 
                        config: MergeConfig,
                        model_weights: Optional[List[float]]) -> MergeResult:
        """Merge models using DARE algorithm."""
        if self.dare_operator is None:
            self.dare_operator = create_dare_operator(config.dare_config)
            
        dare_result = self.dare_operator.merge_models(
            models=models,
            model_weights=model_weights
        )
        
        return MergeResult(
            merged_model=dare_result.merged_model,
            strategy_used=MergeStrategy.DARE,
            fitness_score=0.0,  # Will be calculated by caller
            convergence_generation=None,
            total_time=0.0,
            statistics={
                'dare_stats': {
                    'sparsity_ratio': dare_result.sparsity_ratio,
                    'effective_dropout_rates': dare_result.effective_dropout_rates,
                    'rescale_factors': dare_result.rescale_factors,
                    'merge_statistics': dare_result.merge_statistics
                }
            }
        )
    
    def _merge_with_evolution(self, 
                             models: List[nn.Module], 
                             config: MergeConfig,
                             fitness_function: FitnessFunction,
                             model_weights: Optional[List[float]]) -> MergeResult:
        """Merge models using evolutionary optimization."""
        # Initialize evolutionary engine
        evolution_config = config.evolution_config or {}
        
        self.evolutionary_engine = EvolutionaryEngine(
            population_size=config.population_size,
            elite_size=evolution_config.get('elite_size', 5),
            tournament_size=evolution_config.get('tournament_size', 3),
            mutation_rate=evolution_config.get('mutation_rate', 0.01),
            crossover_rate=evolution_config.get('crossover_rate', 0.8),
            selection_strategy=SelectionStrategy(evolution_config.get('selection_strategy', 'tournament')),
            crossover_type=CrossoverType(evolution_config.get('crossover_type', 'uniform')),
            mutation_type=MutationType(evolution_config.get('mutation_type', 'gaussian')),
            convergence_patience=config.convergence_patience,
            convergence_threshold=config.convergence_threshold,
            device=self.device
        )
        
        # Initialize population from input models
        population = self.evolutionary_engine.initialize_population(models)
        
        # Evolution loop
        evolution_history = []
        
        for generation in range(config.num_generations):
            stats = self.evolutionary_engine.evolve_generation(fitness_function)
            
            evolution_history.append({
                'generation': generation,
                'best_fitness': stats.best_fitness,
                'average_fitness': stats.average_fitness,
                'diversity': stats.diversity
            })
            
            logger.info(f"Generation {generation}: Best={stats.best_fitness:.6f}, Avg={stats.average_fitness:.6f}")
            
            # Check convergence
            if self.evolutionary_engine.has_converged():
                logger.info(f"Converged at generation {generation}")
                break
                
            # Checkpoint if enabled
            if config.enable_checkpointing and generation % config.checkpoint_interval == 0:
                if config.output_dir:
                    checkpoint_path = Path(config.output_dir) / f"evolution_checkpoint_gen_{generation}.pt"
                    self.evolutionary_engine.save_checkpoint(str(checkpoint_path))
                    
        # Get best individual
        best_individual = self.evolutionary_engine.get_best_individual()
        
        return MergeResult(
            merged_model=best_individual.model,
            strategy_used=MergeStrategy.EVOLUTIONARY,
            fitness_score=best_individual.fitness,
            convergence_generation=self.evolutionary_engine.generation,
            total_time=0.0,
            statistics={
                'evolution_stats': {
                    'final_generation': self.evolutionary_engine.generation,
                    'population_diversity': self.evolutionary_engine.get_population_diversity(),
                    'convergence_achieved': self.evolutionary_engine.has_converged()
                }
            },
            evolution_history=evolution_history
        )
    
    def _merge_with_hybrid(self, 
                          models: List[nn.Module], 
                          config: MergeConfig,
                          fitness_function: FitnessFunction,
                          model_weights: Optional[List[float]]) -> MergeResult:
        """Merge models using hybrid approach (combine multiple strategies)."""
        # Default hybrid: DARE -> TIES -> Evolution refinement
        strategies = config.sequence_strategies or [MergeStrategy.DARE, MergeStrategy.TIES, MergeStrategy.EVOLUTIONARY]
        
        current_models = models
        intermediate_results = []
        combined_stats = {}
        
        for i, strategy in enumerate(strategies):
            logger.info(f"Hybrid step {i+1}: {strategy.value}")
            
            # Create sub-config for this strategy
            sub_config = MergeConfig(
                strategy=strategy,
                num_generations=max(10, config.num_generations // len(strategies)),
                population_size=config.population_size,
                convergence_threshold=config.convergence_threshold,
                convergence_patience=config.convergence_patience // len(strategies),
                slerp_config=config.slerp_config,
                ties_config=config.ties_config,
                dare_config=config.dare_config,
                evolution_config=config.evolution_config
            )
            
            # Apply strategy
            result = self.merge_models(
                current_models, 
                sub_config, 
                fitness_function, 
                model_weights
            )
            
            intermediate_results.append(result.merged_model)
            combined_stats[f'step_{i+1}_{strategy.value}'] = result.statistics
            
            # Use result as input for next step
            current_models = [result.merged_model] + models[1:]  # Keep original models for diversity
            
        final_model = intermediate_results[-1]
        
        return MergeResult(
            merged_model=final_model,
            strategy_used=MergeStrategy.HYBRID,
            fitness_score=0.0,  # Will be calculated by caller
            convergence_generation=None,
            total_time=0.0,
            statistics={'hybrid_stats': combined_stats},
            intermediate_results=intermediate_results
        )
    
    def _merge_with_sequential(self, 
                              models: List[nn.Module], 
                              config: MergeConfig,
                              fitness_function: FitnessFunction,
                              model_weights: Optional[List[float]]) -> MergeResult:
        """Merge models sequentially (pairwise merging)."""
        if len(models) < 2:
            raise ValueError("Need at least 2 models for sequential merging")
            
        # Default to SLERP for pairwise merging
        merge_strategy = config.sequence_strategies[0] if config.sequence_strategies else MergeStrategy.SLERP
        
        current_model = models[0]
        intermediate_results = [current_model]
        
        for i in range(1, len(models)):
            logger.info(f"Sequential merge step {i}: merging with model {i}")
            
            # Create pairwise config
            pair_config = MergeConfig(
                strategy=merge_strategy,
                slerp_config=config.slerp_config,
                ties_config=config.ties_config,
                dare_config=config.dare_config
            )
            
            # Merge current result with next model
            result = self.merge_models(
                [current_model, models[i]], 
                pair_config, 
                fitness_function,
                [1.0, model_weights[i] if model_weights else 1.0]
            )
            
            current_model = result.merged_model
            intermediate_results.append(current_model)
            
        return MergeResult(
            merged_model=current_model,
            strategy_used=MergeStrategy.SEQUENTIAL,
            fitness_score=0.0,  # Will be calculated by caller
            convergence_generation=None,
            total_time=0.0,
            statistics={'sequential_stats': {'num_steps': len(models) - 1}},
            intermediate_results=intermediate_results
        )
    
    def _merge_with_parallel_ensemble(self, 
                                     models: List[nn.Module], 
                                     config: MergeConfig,
                                     fitness_function: FitnessFunction,
                                     model_weights: Optional[List[float]]) -> MergeResult:
        """Merge models using parallel ensemble approach."""
        # Apply different strategies in parallel and ensemble the results
        strategies = config.sequence_strategies or [MergeStrategy.SLERP, MergeStrategy.TIES, MergeStrategy.DARE]
        
        ensemble_models = []
        ensemble_weights = config.ensemble_weights or [1.0] * len(strategies)
        combined_stats = {}
        
        for i, strategy in enumerate(strategies):
            logger.info(f"Parallel ensemble branch {i+1}: {strategy.value}")
            
            # Create config for this strategy
            branch_config = MergeConfig(
                strategy=strategy,
                num_generations=config.num_generations,
                population_size=config.population_size,
                slerp_config=config.slerp_config,
                ties_config=config.ties_config,
                dare_config=config.dare_config,
                evolution_config=config.evolution_config
            )
            
            # Apply strategy
            result = self.merge_models(
                models, 
                branch_config, 
                fitness_function, 
                model_weights
            )
            
            ensemble_models.append(result.merged_model)
            combined_stats[f'branch_{i+1}_{strategy.value}'] = result.statistics
            
        # Ensemble the results using SLERP
        if self.slerp_operator is None:
            self.slerp_operator = create_slerp_operator()
            
        final_model = self.slerp_operator.interpolate_batch(ensemble_models, ensemble_weights)
        
        return MergeResult(
            merged_model=final_model,
            strategy_used=MergeStrategy.PARALLEL_ENSEMBLE,
            fitness_score=0.0,  # Will be calculated by caller
            convergence_generation=None,
            total_time=0.0,
            statistics={'parallel_ensemble_stats': combined_stats},
            intermediate_results=ensemble_models
        )
    
    def _create_default_fitness_function(self, reference_model: nn.Module) -> FitnessFunction:
        """Create default fitness function based on model characteristics."""
        # Create simple fitness function that evaluates parameter magnitude and diversity
        def magnitude_fitness(model: nn.Module) -> float:
            total_magnitude = 0.0
            param_count = 0
            
            for param in model.parameters():
                if param.requires_grad:
                    total_magnitude += torch.norm(param.data).item()
                    param_count += 1
                    
            return total_magnitude / param_count if param_count > 0 else 0.0
        
        def diversity_fitness(model: nn.Module) -> float:
            # Measure parameter diversity (variance)
            all_params = []
            for param in model.parameters():
                if param.requires_grad:
                    all_params.extend(param.data.flatten().tolist())
                    
            return float(np.var(all_params)) if all_params else 0.0
        
        # Combine fitness functions
        fitness_functions = [
            (magnitude_fitness, 0.7),
            (diversity_fitness, 0.3)
        ]
        
        return CompositeFitnessFunction(fitness_functions, self.device)
    
    def evaluate_merge_quality(self, 
                              original_models: List[nn.Module], 
                              merged_model: nn.Module,
                              fitness_function: Optional[FitnessFunction] = None) -> Dict[str, float]:
        """
        Evaluate quality of merge operation.
        
        Args:
            original_models: Original input models
            merged_model: Merged result model
            fitness_function: Optional fitness function for evaluation
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        # Basic metrics
        merged_param_count = sum(p.numel() for p in merged_model.parameters())
        avg_original_param_count = np.mean([sum(p.numel() for p in model.parameters()) for model in original_models])
        
        metrics['parameter_preservation'] = merged_param_count / avg_original_param_count
        
        # Fitness evaluation
        if fitness_function:
            merged_fitness = fitness_function.evaluate(merged_model)
            original_fitness = [fitness_function.evaluate(model) for model in original_models]
            
            metrics['fitness_score'] = merged_fitness
            metrics['fitness_improvement'] = merged_fitness / np.mean(original_fitness) if original_fitness else 1.0
            
        # Diversity metrics
        if len(original_models) >= 2:
            # Calculate distances between original models
            original_distances = []
            for i in range(len(original_models)):
                for j in range(i + 1, len(original_models)):
                    dist = self._calculate_model_distance(original_models[i], original_models[j])
                    original_distances.append(dist)
                    
            avg_original_distance = np.mean(original_distances)
            
            # Calculate distance from merged model to originals
            merged_distances = [self._calculate_model_distance(merged_model, model) for model in original_models]
            avg_merged_distance = np.mean(merged_distances)
            
            metrics['diversity_preservation'] = avg_merged_distance / avg_original_distance if avg_original_distance > 0 else 1.0
            
        return metrics
    
    def _calculate_model_distance(self, model1: nn.Module, model2: nn.Module) -> float:
        """Calculate Euclidean distance between two models using consolidated ModelOperations."""
        return calculate_model_distance(model1, model2, distance_type="euclidean")
    
    def get_merge_history(self) -> List[Dict[str, Any]]:
        """Get history of all merge operations."""
        return self.merge_history
    
    def clear_cache(self):
        """Clear intermediate result cache."""
        self.intermediate_cache.clear()
        logger.info("Cleared merge controller cache")
    
    def get_profiling_data(self) -> Dict[str, Any]:
        """Get performance profiling data."""
        return self.profiling_data

# Factory function
def create_merge_controller(device: str = "cpu", **kwargs) -> MergeController:
    """
    Factory function to create merge controller.
    
    Args:
        device: Device for computations
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured merge controller
    """
    return MergeController(
        device=device,
        enable_profiling=kwargs.get('enable_profiling', True),
        cache_intermediate=kwargs.get('cache_intermediate', True)
    )
