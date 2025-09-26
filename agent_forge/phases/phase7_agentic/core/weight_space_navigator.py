"""
Phase 7: Weight Space Navigator

Implementation of a sophisticated weight space navigation system that explores
and optimizes neural network weight configurations using gradient-free optimization
techniques inspired by "Transformers Squared" and meta-learning approaches.

Key Features:
- Multi-dimensional weight space exploration
- Gradient-free optimization algorithms
- Weight evolution tracking and analysis
- Optimal configuration discovery
- Historical weight trajectory analysis
- Dynamic search strategy adaptation
- Performance-guided navigation
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import copy
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationAlgorithm(Enum):
    """Available optimization algorithms for weight space navigation."""
    RANDOM_SEARCH = "random_search"
    EVOLUTIONARY = "evolutionary"
    SIMULATED_ANNEALING = "simulated_annealing"
    PARTICLE_SWARM = "particle_swarm"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    CMA_ES = "cma_es"  # Covariance Matrix Adaptation Evolution Strategy


class SearchStrategy(Enum):
    """Search strategies for exploration."""
    EXPLOITATION = "exploitation"  # Focus on promising regions
    EXPLORATION = "exploration"    # Broad search for diversity
    BALANCED = "balanced"          # Mix of both
    ADAPTIVE = "adaptive"          # Dynamic adjustment


@dataclass
class WeightConfiguration:
    """Represents a specific weight configuration in weight space."""
    config_id: str
    weights: Dict[str, torch.Tensor]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)
    search_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NavigationStep:
    """Represents a single step in weight space navigation."""
    step_id: int
    from_config: str
    to_config: str
    algorithm: OptimizationAlgorithm
    step_size: float
    direction: Optional[torch.Tensor] = None
    performance_change: float = 0.0
    exploration_reward: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrajectoryPoint:
    """A point in the weight space trajectory."""
    config_id: str
    coordinates: torch.Tensor  # Compressed representation of weights
    performance: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceEvaluator(ABC):
    """Abstract base class for performance evaluation."""

    @abstractmethod
    def evaluate(self, model: nn.Module, config: WeightConfiguration) -> Dict[str, float]:
        """Evaluate model performance with given weight configuration."""
        pass


class SimplePerformanceEvaluator(PerformanceEvaluator):
    """Simple performance evaluator for testing purposes."""

    def __init__(self, test_data: Optional[Any] = None):
        self.test_data = test_data
        self.evaluation_count = 0

    def evaluate(self, model: nn.Module, config: WeightConfiguration) -> Dict[str, float]:
        """Simulate performance evaluation."""
        self.evaluation_count += 1

        # Simulate different performance metrics
        # In practice, this would run actual model evaluation
        base_performance = {
            "accuracy": 0.75,
            "loss": 0.25,
            "efficiency": 0.80,
            "robustness": 0.70,
            "generalization": 0.65
        }

        # Add some variation based on weight characteristics
        weight_complexity = self._calculate_weight_complexity(config.weights)
        weight_stability = self._calculate_weight_stability(config.weights)

        # Simulate performance changes based on weight properties
        performance = {}
        for metric, base_value in base_performance.items():
            if metric == "accuracy":
                # Higher complexity might improve accuracy up to a point
                performance[metric] = base_value + 0.1 * weight_complexity - 0.05 * (weight_complexity ** 2)
            elif metric == "loss":
                # More stable weights generally mean lower loss
                performance[metric] = base_value - 0.1 * weight_stability
            elif metric == "efficiency":
                # Less complex weights are more efficient
                performance[metric] = base_value - 0.05 * weight_complexity
            elif metric == "robustness":
                # Stability improves robustness
                performance[metric] = base_value + 0.1 * weight_stability
            else:  # generalization
                # Balance between complexity and stability
                performance[metric] = base_value + 0.05 * (weight_stability - weight_complexity)

            # Add some noise and clamp to valid range
            performance[metric] += np.random.normal(0, 0.02)
            performance[metric] = max(0.0, min(1.0, performance[metric]))

        return performance

    def _calculate_weight_complexity(self, weights: Dict[str, torch.Tensor]) -> float:
        """Calculate a complexity measure for the weight configuration."""
        total_variance = 0.0
        total_params = 0

        for weight_tensor in weights.values():
            if weight_tensor.numel() > 0:
                total_variance += float(weight_tensor.var())
                total_params += weight_tensor.numel()

        return total_variance / max(total_params, 1) if total_params > 0 else 0.0

    def _calculate_weight_stability(self, weights: Dict[str, torch.Tensor]) -> float:
        """Calculate a stability measure for the weight configuration."""
        total_mean_abs = 0.0
        total_params = 0

        for weight_tensor in weights.values():
            if weight_tensor.numel() > 0:
                total_mean_abs += float(weight_tensor.abs().mean())
                total_params += 1

        return 1.0 / (1.0 + total_mean_abs / max(total_params, 1)) if total_params > 0 else 1.0


class WeightSpaceProjector:
    """Projects high-dimensional weight spaces to lower dimensions for analysis."""

    def __init__(self, target_dimension: int = 50):
        self.target_dimension = target_dimension
        self.projection_matrix: Optional[torch.Tensor] = None
        self.is_fitted = False

    def fit(self, weight_configs: List[WeightConfiguration]) -> None:
        """Fit the projector to a set of weight configurations."""
        if not weight_configs:
            return

        # Flatten and concatenate all weights
        weight_vectors = []
        for config in weight_configs:
            flattened = self._flatten_weights(config.weights)
            weight_vectors.append(flattened)

        if not weight_vectors:
            return

        # Stack weight vectors
        weight_matrix = torch.stack(weight_vectors)

        # Simple random projection for now
        # In practice, you might use PCA or other dimensionality reduction
        full_dimension = weight_matrix.shape[1]
        self.projection_matrix = torch.randn(self.target_dimension, full_dimension)
        self.projection_matrix = nn.functional.normalize(self.projection_matrix, dim=1)

        self.is_fitted = True
        logger.info(f"Fitted projector: {full_dimension} -> {self.target_dimension} dimensions")

    def project(self, weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Project weights to lower dimensional space."""
        if not self.is_fitted:
            raise RuntimeError("Projector must be fitted before use")

        flattened = self._flatten_weights(weights)
        projected = torch.matmul(self.projection_matrix, flattened)
        return projected

    def _flatten_weights(self, weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten weight dictionary to a single tensor."""
        flattened_parts = []
        for weight_tensor in weights.values():
            flattened_parts.append(weight_tensor.flatten())

        return torch.cat(flattened_parts) if flattened_parts else torch.empty(0)


class BaseOptimizer(ABC):
    """Base class for weight space optimization algorithms."""

    def __init__(self, name: str):
        self.name = name
        self.history: List[WeightConfiguration] = []

    @abstractmethod
    def initialize_population(self, base_config: WeightConfiguration, population_size: int) -> List[WeightConfiguration]:
        """Initialize the optimization population."""
        pass

    @abstractmethod
    def step(self, population: List[WeightConfiguration], evaluator: PerformanceEvaluator) -> List[WeightConfiguration]:
        """Perform one optimization step."""
        pass

    def get_best_configuration(self, population: List[WeightConfiguration], metric: str = "accuracy") -> Optional[WeightConfiguration]:
        """Get the best configuration from the population."""
        if not population:
            return None

        best_config = None
        best_score = float('-inf')

        for config in population:
            if metric in config.performance_metrics:
                score = config.performance_metrics[metric]
                if score > best_score:
                    best_score = score
                    best_config = config

        return best_config


class RandomSearchOptimizer(BaseOptimizer):
    """Random search optimization algorithm."""

    def __init__(self, mutation_std: float = 0.1):
        super().__init__("RandomSearch")
        self.mutation_std = mutation_std

    def initialize_population(self, base_config: WeightConfiguration, population_size: int) -> List[WeightConfiguration]:
        """Initialize population with random mutations of base configuration."""
        population = [base_config]  # Include the base configuration

        for i in range(population_size - 1):
            mutated_config = self._mutate_configuration(base_config, f"random_{i}")
            population.append(mutated_config)

        return population

    def step(self, population: List[WeightConfiguration], evaluator: PerformanceEvaluator) -> List[WeightConfiguration]:
        """Generate new random candidates."""
        # Keep best half of population and generate new random candidates
        population.sort(key=lambda x: x.performance_metrics.get("accuracy", 0.0), reverse=True)

        keep_count = len(population) // 2
        new_population = population[:keep_count]

        # Generate new random candidates
        for i in range(len(population) - keep_count):
            base_config = np.random.choice(new_population)
            mutated_config = self._mutate_configuration(base_config, f"random_gen_{i}")
            new_population.append(mutated_config)

        return new_population

    def _mutate_configuration(self, base_config: WeightConfiguration, new_id: str) -> WeightConfiguration:
        """Create a mutated version of a configuration."""
        mutated_weights = {}

        for layer_name, weight_tensor in base_config.weights.items():
            # Add Gaussian noise to weights
            noise = torch.randn_like(weight_tensor) * self.mutation_std
            mutated_weights[layer_name] = weight_tensor + noise

        return WeightConfiguration(
            config_id=new_id,
            weights=mutated_weights,
            generation=base_config.generation + 1,
            parent_ids=[base_config.config_id],
            mutation_history=base_config.mutation_history + ["random_mutation"]
        )


class EvolutionaryOptimizer(BaseOptimizer):
    """Evolutionary optimization algorithm."""

    def __init__(self, mutation_rate: float = 0.1, crossover_rate: float = 0.7):
        super().__init__("Evolutionary")
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def initialize_population(self, base_config: WeightConfiguration, population_size: int) -> List[WeightConfiguration]:
        """Initialize population with diverse mutations."""
        population = [base_config]

        for i in range(population_size - 1):
            mutated_config = self._mutate_configuration(base_config, f"evo_init_{i}")
            population.append(mutated_config)

        return population

    def step(self, population: List[WeightConfiguration], evaluator: PerformanceEvaluator) -> List[WeightConfiguration]:
        """Perform evolutionary step with selection, crossover, and mutation."""
        # Selection: tournament selection
        parents = self._tournament_selection(population, len(population))

        # Crossover and mutation
        new_population = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[min(i + 1, len(parents) - 1)]

            if np.random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2, f"evo_child_{i}_0", f"evo_child_{i}_1")
            else:
                child1, child2 = parent1, parent2

            # Mutation
            if np.random.random() < self.mutation_rate:
                child1 = self._mutate_configuration(child1, f"evo_mut_{i}_0")
            if np.random.random() < self.mutation_rate:
                child2 = self._mutate_configuration(child2, f"evo_mut_{i}_1")

            new_population.extend([child1, child2])

        return new_population[:len(population)]

    def _tournament_selection(self, population: List[WeightConfiguration], num_parents: int) -> List[WeightConfiguration]:
        """Tournament selection for choosing parents."""
        parents = []
        tournament_size = 3

        for _ in range(num_parents):
            tournament = np.random.choice(population, min(tournament_size, len(population)), replace=False)
            winner = max(tournament, key=lambda x: x.performance_metrics.get("accuracy", 0.0))
            parents.append(winner)

        return parents

    def _crossover(self, parent1: WeightConfiguration, parent2: WeightConfiguration,
                  child1_id: str, child2_id: str) -> Tuple[WeightConfiguration, WeightConfiguration]:
        """Crossover between two parent configurations."""
        child1_weights = {}
        child2_weights = {}

        for layer_name in parent1.weights.keys():
            if layer_name in parent2.weights:
                # Uniform crossover
                mask = torch.rand_like(parent1.weights[layer_name]) > 0.5

                child1_weights[layer_name] = torch.where(
                    mask, parent1.weights[layer_name], parent2.weights[layer_name]
                )
                child2_weights[layer_name] = torch.where(
                    mask, parent2.weights[layer_name], parent1.weights[layer_name]
                )
            else:
                # If layer doesn't exist in both parents, copy from parent1
                child1_weights[layer_name] = parent1.weights[layer_name].clone()
                child2_weights[layer_name] = parent1.weights[layer_name].clone()

        child1 = WeightConfiguration(
            config_id=child1_id,
            weights=child1_weights,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.config_id, parent2.config_id],
            mutation_history=["crossover"]
        )

        child2 = WeightConfiguration(
            config_id=child2_id,
            weights=child2_weights,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.config_id, parent2.config_id],
            mutation_history=["crossover"]
        )

        return child1, child2

    def _mutate_configuration(self, config: WeightConfiguration, new_id: str) -> WeightConfiguration:
        """Mutate a configuration."""
        mutated_weights = {}

        for layer_name, weight_tensor in config.weights.items():
            # Adaptive mutation based on layer size
            mutation_std = 0.1 / np.sqrt(weight_tensor.numel())
            noise = torch.randn_like(weight_tensor) * mutation_std
            mutated_weights[layer_name] = weight_tensor + noise

        return WeightConfiguration(
            config_id=new_id,
            weights=mutated_weights,
            generation=config.generation + 1,
            parent_ids=[config.config_id],
            mutation_history=config.mutation_history + ["mutation"]
        )


class SimulatedAnnealingOptimizer(BaseOptimizer):
    """Simulated annealing optimization algorithm."""

    def __init__(self, initial_temperature: float = 1.0, cooling_rate: float = 0.95):
        super().__init__("SimulatedAnnealing")
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.current_temperature = initial_temperature
        self.current_config: Optional[WeightConfiguration] = None

    def initialize_population(self, base_config: WeightConfiguration, population_size: int) -> List[WeightConfiguration]:
        """Initialize with the base configuration (SA works with single candidate)."""
        self.current_config = base_config
        return [base_config]

    def step(self, population: List[WeightConfiguration], evaluator: PerformanceEvaluator) -> List[WeightConfiguration]:
        """Perform simulated annealing step."""
        if not self.current_config:
            return population

        # Generate neighbor
        neighbor = self._generate_neighbor(self.current_config, "sa_neighbor")

        # Evaluate neighbor
        neighbor_performance = evaluator.evaluate(None, neighbor)
        neighbor.performance_metrics = neighbor_performance

        # Decide whether to accept neighbor
        current_score = self.current_config.performance_metrics.get("accuracy", 0.0)
        neighbor_score = neighbor_performance.get("accuracy", 0.0)

        if self._accept_neighbor(current_score, neighbor_score):
            self.current_config = neighbor

        # Cool down temperature
        self.current_temperature *= self.cooling_rate

        return [self.current_config]

    def _generate_neighbor(self, config: WeightConfiguration, new_id: str) -> WeightConfiguration:
        """Generate a neighboring configuration."""
        neighbor_weights = {}

        for layer_name, weight_tensor in config.weights.items():
            # Small random perturbation
            perturbation_std = self.current_temperature * 0.1
            noise = torch.randn_like(weight_tensor) * perturbation_std
            neighbor_weights[layer_name] = weight_tensor + noise

        return WeightConfiguration(
            config_id=new_id,
            weights=neighbor_weights,
            generation=config.generation + 1,
            parent_ids=[config.config_id],
            mutation_history=config.mutation_history + ["sa_neighbor"]
        )

    def _accept_neighbor(self, current_score: float, neighbor_score: float) -> bool:
        """Decide whether to accept a neighbor based on SA criteria."""
        if neighbor_score > current_score:
            return True

        # Accept worse solutions with probability based on temperature
        delta = current_score - neighbor_score
        probability = np.exp(-delta / max(self.current_temperature, 1e-10))
        return np.random.random() < probability


class WeightSpaceNavigator:
    """
    Main weight space navigation system that explores and optimizes neural network
    weight configurations using various gradient-free optimization algorithms.

    This navigator implements advanced meta-learning concepts to discover optimal
    weight configurations through intelligent exploration of the weight space.
    """

    def __init__(
        self,
        model: nn.Module,
        evaluator: PerformanceEvaluator,
        algorithm: OptimizationAlgorithm = OptimizationAlgorithm.EVOLUTIONARY,
        search_strategy: SearchStrategy = SearchStrategy.BALANCED
    ):
        self.model = model
        self.evaluator = evaluator
        self.algorithm = algorithm
        self.search_strategy = search_strategy

        # Navigation components
        self.optimizer = self._create_optimizer(algorithm)
        self.projector = WeightSpaceProjector()

        # Navigation state
        self.configurations: Dict[str, WeightConfiguration] = {}
        self.trajectory: List[TrajectoryPoint] = []
        self.navigation_steps: List[NavigationStep] = []
        self.current_population: List[WeightConfiguration] = []

        # Performance tracking
        self.best_configuration: Optional[WeightConfiguration] = None
        self.performance_history: List[Tuple[int, float]] = []
        self.convergence_threshold = 1e-6
        self.max_stagnation_steps = 10
        self.stagnation_counter = 0

        # Search adaptation
        self.search_radius = 1.0
        self.exploration_factor = 0.3
        self.exploitation_factor = 0.7

        logger.info(f"Initialized WeightSpaceNavigator with {algorithm.value} algorithm")

    def _create_optimizer(self, algorithm: OptimizationAlgorithm) -> BaseOptimizer:
        """Create optimizer based on selected algorithm."""
        optimizers = {
            OptimizationAlgorithm.RANDOM_SEARCH: RandomSearchOptimizer(),
            OptimizationAlgorithm.EVOLUTIONARY: EvolutionaryOptimizer(),
            OptimizationAlgorithm.SIMULATED_ANNEALING: SimulatedAnnealingOptimizer()
        }

        if algorithm in optimizers:
            return optimizers[algorithm]
        else:
            logger.warning(f"Algorithm {algorithm.value} not implemented, using evolutionary")
            return EvolutionaryOptimizer()

    def initialize_navigation(self, population_size: int = 20) -> None:
        """Initialize navigation with the current model weights."""
        logger.info(f"Initializing navigation with population size: {population_size}")

        # Create base configuration from current model weights
        base_weights = {}
        for name, param in self.model.named_parameters():
            base_weights[name] = param.data.clone()

        base_config = WeightConfiguration(
            config_id="base_config",
            weights=base_weights,
            generation=0
        )

        # Evaluate base configuration
        base_performance = self.evaluator.evaluate(self.model, base_config)
        base_config.performance_metrics = base_performance

        # Initialize population
        self.current_population = self.optimizer.initialize_population(base_config, population_size)

        # Evaluate initial population
        for config in self.current_population:
            if not config.performance_metrics:
                self._apply_configuration(config)
                performance = self.evaluator.evaluate(self.model, config)
                config.performance_metrics = performance

            self.configurations[config.config_id] = config

        # Set initial best configuration
        self.best_configuration = self.optimizer.get_best_configuration(self.current_population)

        # Initialize trajectory tracking
        if self.best_configuration:
            self.performance_history.append((0, self.best_configuration.performance_metrics.get("accuracy", 0.0)))

        logger.info(f"Navigation initialized with {len(self.current_population)} configurations")

    def navigate_steps(self, num_steps: int, target_performance: Optional[float] = None) -> Dict[str, Any]:
        """
        Navigate through weight space for a specified number of steps.

        Args:
            num_steps: Number of navigation steps to perform
            target_performance: Optional target performance to achieve

        Returns:
            Dict containing navigation results and statistics
        """
        logger.info(f"Starting navigation for {num_steps} steps")

        start_time = time.time()
        convergence_achieved = False
        target_achieved = False

        for step in range(num_steps):
            step_start_time = time.time()

            # Perform optimization step
            self.current_population = self.optimizer.step(self.current_population, self.evaluator)

            # Evaluate new configurations
            for config in self.current_population:
                if config.config_id not in self.configurations:
                    self._apply_configuration(config)
                    performance = self.evaluator.evaluate(self.model, config)
                    config.performance_metrics = performance
                    self.configurations[config.config_id] = config

            # Update best configuration
            current_best = self.optimizer.get_best_configuration(self.current_population)
            if current_best and (not self.best_configuration or
                                current_best.performance_metrics.get("accuracy", 0.0) >
                                self.best_configuration.performance_metrics.get("accuracy", 0.0)):

                prev_best_score = self.best_configuration.performance_metrics.get("accuracy", 0.0) if self.best_configuration else 0.0
                current_best_score = current_best.performance_metrics.get("accuracy", 0.0)

                self.best_configuration = current_best
                self.stagnation_counter = 0

                logger.info(f"Step {step + 1}: New best configuration found (score: {current_best_score:.4f})")
            else:
                self.stagnation_counter += 1

            # Record performance history
            best_score = self.best_configuration.performance_metrics.get("accuracy", 0.0) if self.best_configuration else 0.0
            self.performance_history.append((step + 1, best_score))

            # Update trajectory
            if self.best_configuration:
                self._update_trajectory(self.best_configuration)

            # Adapt search strategy
            self._adapt_search_strategy(step)

            # Check convergence
            if self._check_convergence():
                convergence_achieved = True
                logger.info(f"Convergence achieved at step {step + 1}")
                break

            # Check target performance
            if target_performance and best_score >= target_performance:
                target_achieved = True
                logger.info(f"Target performance {target_performance:.4f} achieved at step {step + 1}")
                break

            # Log progress
            step_time = time.time() - step_start_time
            if (step + 1) % 10 == 0:
                logger.info(f"Step {step + 1}: best_score={best_score:.4f}, step_time={step_time:.3f}s")

        total_time = time.time() - start_time

        # Prepare results
        results = {
            "steps_completed": step + 1 if 'step' in locals() else num_steps,
            "total_time": total_time,
            "convergence_achieved": convergence_achieved,
            "target_achieved": target_achieved,
            "best_configuration": self.best_configuration.config_id if self.best_configuration else None,
            "best_performance": self.best_configuration.performance_metrics if self.best_configuration else {},
            "total_configurations": len(self.configurations),
            "stagnation_steps": self.stagnation_counter,
            "performance_improvement": self._calculate_performance_improvement(),
            "navigation_efficiency": self._calculate_navigation_efficiency(),
            "trajectory_summary": self._get_trajectory_summary()
        }

        logger.info(f"Navigation completed: {results['steps_completed']} steps, best score: {results['best_performance'].get('accuracy', 0.0):.4f}")
        return results

    def _apply_configuration(self, config: WeightConfiguration) -> None:
        """Apply a weight configuration to the model."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in config.weights:
                    param.data.copy_(config.weights[name])

    def _update_trajectory(self, config: WeightConfiguration) -> None:
        """Update the trajectory with a new configuration point."""
        if not self.projector.is_fitted:
            # Fit projector if we have enough configurations
            if len(self.configurations) >= 10:
                self.projector.fit(list(self.configurations.values()))

        if self.projector.is_fitted:
            coordinates = self.projector.project(config.weights)

            trajectory_point = TrajectoryPoint(
                config_id=config.config_id,
                coordinates=coordinates,
                performance=config.performance_metrics.get("accuracy", 0.0),
                timestamp=time.time(),
                metadata={
                    "generation": config.generation,
                    "algorithm": self.algorithm.value,
                    "search_strategy": self.search_strategy.value
                }
            )

            self.trajectory.append(trajectory_point)

            # Keep trajectory manageable
            if len(self.trajectory) > 1000:
                self.trajectory = self.trajectory[-800:]  # Keep last 800 points

    def _adapt_search_strategy(self, step: int) -> None:
        """Adapt search strategy based on progress."""
        if self.search_strategy == SearchStrategy.ADAPTIVE:
            # Increase exploration if stagnating
            if self.stagnation_counter > 5:
                self.exploration_factor = min(0.8, self.exploration_factor + 0.1)
                self.exploitation_factor = 1.0 - self.exploration_factor
            else:
                # Gradually shift toward exploitation
                self.exploration_factor = max(0.1, self.exploration_factor * 0.95)
                self.exploitation_factor = 1.0 - self.exploration_factor

            # Adapt search radius
            if self.stagnation_counter > 3:
                self.search_radius *= 1.1  # Expand search
            else:
                self.search_radius *= 0.98  # Contract search

    def _check_convergence(self) -> bool:
        """Check if navigation has converged."""
        if len(self.performance_history) < 10:
            return False

        # Check for stagnation
        if self.stagnation_counter > self.max_stagnation_steps:
            return True

        # Check for small improvements
        recent_scores = [score for _, score in self.performance_history[-10:]]
        if max(recent_scores) - min(recent_scores) < self.convergence_threshold:
            return True

        return False

    def _calculate_performance_improvement(self) -> float:
        """Calculate total performance improvement during navigation."""
        if len(self.performance_history) < 2:
            return 0.0

        initial_score = self.performance_history[0][1]
        final_score = self.performance_history[-1][1]
        return final_score - initial_score

    def _calculate_navigation_efficiency(self) -> float:
        """Calculate navigation efficiency (improvement per evaluation)."""
        improvement = self._calculate_performance_improvement()
        evaluations = len(self.configurations)
        return improvement / max(evaluations, 1)

    def _get_trajectory_summary(self) -> Dict[str, Any]:
        """Get a summary of the navigation trajectory."""
        if not self.trajectory:
            return {}

        performances = [point.performance for point in self.trajectory]

        return {
            "trajectory_length": len(self.trajectory),
            "performance_range": [min(performances), max(performances)],
            "average_performance": np.mean(performances),
            "performance_std": np.std(performances),
            "trajectory_diversity": self._calculate_trajectory_diversity()
        }

    def _calculate_trajectory_diversity(self) -> float:
        """Calculate diversity of the trajectory in weight space."""
        if len(self.trajectory) < 2:
            return 0.0

        # Calculate average distance between consecutive points
        distances = []
        for i in range(1, len(self.trajectory)):
            prev_coords = self.trajectory[i-1].coordinates
            curr_coords = self.trajectory[i].coordinates
            distance = torch.norm(curr_coords - prev_coords).item()
            distances.append(distance)

        return np.mean(distances) if distances else 0.0

    def get_optimal_configuration(self) -> Optional[WeightConfiguration]:
        """Get the optimal configuration found during navigation."""
        return self.best_configuration

    def apply_optimal_configuration(self) -> bool:
        """Apply the optimal configuration to the model."""
        if not self.best_configuration:
            logger.warning("No optimal configuration found")
            return False

        self._apply_configuration(self.best_configuration)
        logger.info(f"Applied optimal configuration: {self.best_configuration.config_id}")
        return True

    def analyze_weight_evolution(self) -> Dict[str, Any]:
        """Analyze how weights evolved during navigation."""
        if not self.best_configuration or "base_config" not in self.configurations:
            return {}

        base_config = self.configurations["base_config"]
        final_config = self.best_configuration

        analysis = {}

        # Analyze layer-wise changes
        layer_changes = {}
        for layer_name in base_config.weights.keys():
            if layer_name in final_config.weights:
                base_weights = base_config.weights[layer_name]
                final_weights = final_config.weights[layer_name]

                # Calculate various metrics
                weight_change = final_weights - base_weights
                layer_changes[layer_name] = {
                    "mean_change": float(weight_change.mean()),
                    "std_change": float(weight_change.std()),
                    "max_change": float(weight_change.abs().max()),
                    "relative_change": float(weight_change.norm() / base_weights.norm())
                }

        analysis["layer_changes"] = layer_changes

        # Overall statistics
        all_changes = []
        for layer_data in layer_changes.values():
            all_changes.append(layer_data["relative_change"])

        analysis["overall_stats"] = {
            "average_relative_change": np.mean(all_changes) if all_changes else 0.0,
            "max_relative_change": max(all_changes) if all_changes else 0.0,
            "layers_analyzed": len(layer_changes),
            "performance_improvement": self._calculate_performance_improvement()
        }

        return analysis

    def visualize_navigation(self, save_path: Optional[str] = None) -> None:
        """Visualize the navigation trajectory and performance."""
        if not self.performance_history:
            logger.warning("No navigation history to visualize")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Performance over time
        steps, scores = zip(*self.performance_history)
        ax1.plot(steps, scores, 'b-', linewidth=2)
        ax1.set_xlabel('Navigation Step')
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Performance Evolution')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Performance distribution
        ax2.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Performance Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Performance Distribution')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Trajectory in 2D (if available)
        if self.trajectory and len(self.trajectory[0].coordinates) >= 2:
            x_coords = [point.coordinates[0].item() for point in self.trajectory]
            y_coords = [point.coordinates[1].item() for point in self.trajectory]
            performances = [point.performance for point in self.trajectory]

            scatter = ax3.scatter(x_coords, y_coords, c=performances, cmap='viridis', alpha=0.7)
            ax3.plot(x_coords, y_coords, 'r-', alpha=0.3, linewidth=1)
            ax3.set_xlabel('Weight Space Dimension 1')
            ax3.set_ylabel('Weight Space Dimension 2')
            ax3.set_title('Navigation Trajectory in Weight Space')
            plt.colorbar(scatter, ax=ax3, label='Performance')
        else:
            ax3.text(0.5, 0.5, 'No trajectory data available',
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Weight Space Trajectory (N/A)')

        # Plot 4: Algorithm statistics
        if hasattr(self.evaluator, 'evaluation_count'):
            eval_count = self.evaluator.evaluation_count
            improvement = self._calculate_performance_improvement()
            efficiency = improvement / max(eval_count, 1)

            stats = ['Evaluations', 'Improvement', 'Efficiency']
            values = [eval_count, improvement * 100, efficiency * 100]

            bars = ax4.bar(stats, values, color=['lightcoral', 'lightgreen', 'lightskyblue'])
            ax4.set_ylabel('Value')
            ax4.set_title('Navigation Statistics')

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Navigation visualization saved to {save_path}")
        else:
            plt.show()

    def export_navigation_data(self, export_path: str) -> None:
        """Export navigation data for analysis."""
        export_data = {
            "algorithm": self.algorithm.value,
            "search_strategy": self.search_strategy.value,
            "performance_history": self.performance_history,
            "best_configuration_id": self.best_configuration.config_id if self.best_configuration else None,
            "best_performance": self.best_configuration.performance_metrics if self.best_configuration else {},
            "total_configurations": len(self.configurations),
            "trajectory_summary": self._get_trajectory_summary(),
            "weight_evolution_analysis": self.analyze_weight_evolution(),
            "convergence_info": {
                "stagnation_counter": self.stagnation_counter,
                "convergence_threshold": self.convergence_threshold,
                "max_stagnation_steps": self.max_stagnation_steps
            }
        }

        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Navigation data exported to {export_path}")

    def get_navigation_status(self) -> Dict[str, Any]:
        """Get current navigation status."""
        return {
            "algorithm": self.algorithm.value,
            "search_strategy": self.search_strategy.value,
            "total_configurations": len(self.configurations),
            "current_population_size": len(self.current_population),
            "best_performance": self.best_configuration.performance_metrics if self.best_configuration else {},
            "stagnation_counter": self.stagnation_counter,
            "navigation_steps": len(self.navigation_steps),
            "trajectory_length": len(self.trajectory),
            "search_radius": self.search_radius,
            "exploration_factor": self.exploration_factor,
            "exploitation_factor": self.exploitation_factor
        }


# Export main classes
__all__ = [
    'WeightSpaceNavigator',
    'WeightConfiguration',
    'NavigationStep',
    'TrajectoryPoint',
    'OptimizationAlgorithm',
    'SearchStrategy',
    'PerformanceEvaluator',
    'SimplePerformanceEvaluator',
    'WeightSpaceProjector',
    'RandomSearchOptimizer',
    'EvolutionaryOptimizer',
    'SimulatedAnnealingOptimizer'
]


if __name__ == "__main__":
    # Example usage and testing
    logger.info("Testing WeightSpaceNavigator...")

    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 20)
            self.layer2 = nn.Linear(20, 10)
            self.layer3 = nn.Linear(10, 1)

        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            x = self.layer3(x)
            return x

    # Initialize model and evaluator
    model = SimpleModel()
    evaluator = SimplePerformanceEvaluator()

    # Test different optimization algorithms
    algorithms = [
        OptimizationAlgorithm.RANDOM_SEARCH,
        OptimizationAlgorithm.EVOLUTIONARY,
        OptimizationAlgorithm.SIMULATED_ANNEALING
    ]

    for algorithm in algorithms:
        print(f"\n=== Testing {algorithm.value} ===")

        navigator = WeightSpaceNavigator(
            model=model,
            evaluator=evaluator,
            algorithm=algorithm,
            search_strategy=SearchStrategy.ADAPTIVE
        )

        # Initialize navigation
        navigator.initialize_navigation(population_size=10)

        # Navigate for a few steps
        results = navigator.navigate_steps(num_steps=20, target_performance=0.85)

        print(f"Steps completed: {results['steps_completed']}")
        print(f"Best performance: {results['best_performance'].get('accuracy', 0.0):.4f}")
        print(f"Performance improvement: {results['performance_improvement']:.4f}")
        print(f"Navigation efficiency: {results['navigation_efficiency']:.6f}")
        print(f"Convergence achieved: {results['convergence_achieved']}")

        # Get navigation status
        status = navigator.get_navigation_status()
        print(f"Total configurations explored: {status['total_configurations']}")

        # Analyze weight evolution
        evolution_analysis = navigator.analyze_weight_evolution()
        if evolution_analysis:
            overall_stats = evolution_analysis.get('overall_stats', {})
            print(f"Average relative weight change: {overall_stats.get('average_relative_change', 0.0):.4f}")

    logger.info("WeightSpaceNavigator testing completed")