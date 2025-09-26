"""
Automatic Discovery of Agentic Expert Vector Configurations
Combines ADAS Meta Agent Search with Transformers² Expert Vector Discovery

Uses ADAS strategy to automatically program and discover optimal expert vector
configurations for Transformers² SVD-based adaptation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
import json
import time
import random
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ExpertVectorConfig:
    """Configuration for a discovered expert vector"""
    config_id: str
    svd_components: Dict[str, List[int]]  # Layer -> singular value indices
    expert_weights: Dict[str, torch.Tensor]  # Layer -> expert vector
    task_specialization: str
    performance_score: float = 0.0
    rl_training_iterations: int = 0
    discovery_method: str = "meta_agent_search"


class ExpertVectorArchive:
    """Archive of discovered expert vector configurations (ADAS-style)"""

    def __init__(self, max_size: int = 100):
        self.configurations: List[ExpertVectorConfig] = []
        self.max_size = max_size
        self.performance_history = {}

    def add_configuration(self, config: ExpertVectorConfig):
        """Add a new expert configuration to archive"""
        self.configurations.append(config)

        # Maintain archive size
        if len(self.configurations) > self.max_size:
            # Remove lowest performing configurations
            self.configurations.sort(key=lambda x: x.performance_score, reverse=True)
            self.configurations = self.configurations[:self.max_size]

        logger.info(f"Added expert config {config.config_id} with score {config.performance_score:.3f}")

    def get_best_configurations(self, top_k: int = 5) -> List[ExpertVectorConfig]:
        """Get top-k performing configurations"""
        sorted_configs = sorted(self.configurations, key=lambda x: x.performance_score, reverse=True)
        return sorted_configs[:top_k]

    def get_diversity_sample(self, sample_size: int = 3) -> List[ExpertVectorConfig]:
        """Get diverse sample of configurations for crossover"""
        if len(self.configurations) <= sample_size:
            return self.configurations.copy()

        # Simple diversity sampling based on different task specializations
        specializations = {}
        for config in self.configurations:
            spec = config.task_specialization
            if spec not in specializations:
                specializations[spec] = []
            specializations[spec].append(config)

        diverse_sample = []
        for spec_configs in specializations.values():
            if len(diverse_sample) < sample_size:
                # Take best from each specialization
                best_in_spec = max(spec_configs, key=lambda x: x.performance_score)
                diverse_sample.append(best_in_spec)

        return diverse_sample[:sample_size]


class MetaAgentExpertDiscovery:
    """
    Meta Agent for discovering expert vector configurations
    Implements ADAS strategy applied to Transformers² expert vectors
    """

    def __init__(self, target_model: nn.Module, device: str = 'cuda'):
        self.target_model = target_model
        self.device = device

        # ADAS-style archive
        self.archive = ExpertVectorArchive()

        # SVD analysis cache
        self.svd_cache = {}

        # Meta agent state
        self.search_iteration = 0
        self.discovery_log = []

        # Task specializations to explore
        self.task_specializations = [
            "reasoning", "creativity", "coding", "analysis",
            "math", "language", "multimodal", "general"
        ]

        # Initialize with SVD analysis of target model
        self._initialize_svd_analysis()

        logger.info("Initialized Meta Agent for Expert Vector Discovery")

    def _initialize_svd_analysis(self):
        """Initialize SVD analysis of target model weights"""
        logger.info("Performing initial SVD analysis of target model")

        for name, param in self.target_model.named_parameters():
            if len(param.shape) >= 2:  # Only analyze weight matrices
                # Perform SVD
                weight_matrix = param.data.cpu().numpy()
                U, S, Vt = np.linalg.svd(weight_matrix, full_matrices=False)

                # Store SVD components
                self.svd_cache[name] = {
                    'U': torch.from_numpy(U).to(self.device),
                    'S': torch.from_numpy(S).to(self.device),
                    'Vt': torch.from_numpy(Vt).to(self.device),
                    'effective_rank': self._calculate_effective_rank(S),
                    'shape': param.shape
                }

        logger.info(f"SVD analysis complete for {len(self.svd_cache)} layers")

    def _calculate_effective_rank(self, singular_values: np.ndarray, threshold: float = 0.01) -> int:
        """Calculate effective rank of matrix"""
        normalized_sv = singular_values / singular_values[0]
        return int(np.sum(normalized_sv > threshold))

    def discover_expert_configurations(
        self,
        num_iterations: int = 20,
        evaluation_function: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Main discovery loop - ADAS-style meta agent search for expert vectors
        """
        logger.info(f"Starting expert vector discovery for {num_iterations} iterations")

        discovery_results = {
            "iterations": [],
            "best_configurations": [],
            "performance_progression": [],
            "discovery_log": []
        }

        for iteration in range(num_iterations):
            self.search_iteration = iteration + 1
            logger.info(f"\n=== DISCOVERY ITERATION {self.search_iteration} ===")

            # Meta agent generates new expert configuration ideas
            new_configurations = self._meta_agent_generate_configurations()

            # Evaluate each configuration
            for config in new_configurations:
                performance = self._evaluate_expert_configuration(config, evaluation_function)
                config.performance_score = performance

                # Add to archive
                self.archive.add_configuration(config)

            # Track iteration results
            iteration_result = {
                "iteration": self.search_iteration,
                "new_configurations": len(new_configurations),
                "archive_size": len(self.archive.configurations),
                "best_score": max([c.performance_score for c in new_configurations]) if new_configurations else 0.0
            }

            discovery_results["iterations"].append(iteration_result)
            discovery_results["performance_progression"].append(iteration_result["best_score"])

            # Log discovery progress
            self._log_discovery_progress(new_configurations)

        # Compile final results
        discovery_results["best_configurations"] = self.archive.get_best_configurations(10)
        discovery_results["total_discovered"] = len(self.archive.configurations)

        logger.info(f"Discovery complete. Found {len(self.archive.configurations)} expert configurations")

        return discovery_results

    def _meta_agent_generate_configurations(self) -> List[ExpertVectorConfig]:
        """
        Meta agent generates new expert vector configurations
        Uses ADAS-inspired strategy with code-like configuration programming
        """
        new_configurations = []

        # Strategy 1: Random exploration (early iterations)
        if self.search_iteration <= 5:
            config = self._generate_random_configuration()
            new_configurations.append(config)

        # Strategy 2: Archive-based mutation (middle iterations)
        if len(self.archive.configurations) >= 2:
            config = self._mutate_existing_configuration()
            new_configurations.append(config)

        # Strategy 3: Archive-based crossover (later iterations)
        if len(self.archive.configurations) >= 3 and self.search_iteration > 3:
            config = self._crossover_configurations()
            new_configurations.append(config)

        # Strategy 4: Specialized generation based on task analysis
        if self.search_iteration > 2:
            config = self._generate_task_specialized_configuration()
            new_configurations.append(config)

        return new_configurations

    def _generate_random_configuration(self) -> ExpertVectorConfig:
        """Generate random expert vector configuration for exploration"""
        config_id = f"random_{self.search_iteration}_{random.randint(1000, 9999)}"
        task_spec = random.choice(self.task_specializations)

        svd_components = {}
        expert_weights = {}

        # Randomly select layers and singular value components
        selected_layers = random.sample(
            list(self.svd_cache.keys()),
            k=min(3, len(self.svd_cache))  # Select up to 3 layers
        )

        for layer_name in selected_layers:
            svd_data = self.svd_cache[layer_name]
            effective_rank = svd_data['effective_rank']

            # Randomly select singular value components
            num_components = random.randint(1, min(10, effective_rank))
            selected_indices = random.sample(range(effective_rank), k=num_components)
            svd_components[layer_name] = selected_indices

            # Generate random expert vector for this layer
            expert_vector = torch.randn(len(selected_indices), device=self.device)
            expert_vector = torch.nn.functional.normalize(expert_vector, dim=0)
            expert_weights[layer_name] = expert_vector

        return ExpertVectorConfig(
            config_id=config_id,
            svd_components=svd_components,
            expert_weights=expert_weights,
            task_specialization=task_spec,
            discovery_method="random_exploration"
        )

    def _mutate_existing_configuration(self) -> ExpertVectorConfig:
        """Mutate an existing high-performing configuration"""
        # Select a good configuration to mutate
        parent_configs = self.archive.get_best_configurations(top_k=3)
        parent = random.choice(parent_configs)

        config_id = f"mutate_{self.search_iteration}_{random.randint(1000, 9999)}"

        # Copy parent configuration
        svd_components = parent.svd_components.copy()
        expert_weights = {}

        # Mutate expert weights
        for layer_name, parent_weights in parent.expert_weights.items():
            # Add noise to expert vector
            noise_scale = 0.1
            noise = torch.randn_like(parent_weights) * noise_scale
            mutated_weights = parent_weights + noise
            mutated_weights = torch.nn.functional.normalize(mutated_weights, dim=0)
            expert_weights[layer_name] = mutated_weights

        # Possibly mutate SVD components selection
        if random.random() < 0.3:  # 30% chance to change components
            layer_to_mutate = random.choice(list(svd_components.keys()))
            svd_data = self.svd_cache[layer_to_mutate]
            effective_rank = svd_data['effective_rank']

            # Add or remove one component
            current_components = svd_components[layer_to_mutate]
            if random.random() < 0.5 and len(current_components) < effective_rank:
                # Add component
                available = set(range(effective_rank)) - set(current_components)
                if available:
                    new_component = random.choice(list(available))
                    current_components.append(new_component)
                    # Extend expert vector
                    expert_weights[layer_to_mutate] = torch.cat([
                        expert_weights[layer_to_mutate],
                        torch.randn(1, device=self.device)
                    ])
            elif len(current_components) > 1:
                # Remove component
                remove_idx = random.randint(0, len(current_components) - 1)
                current_components.pop(remove_idx)
                # Trim expert vector
                mask = torch.ones(len(expert_weights[layer_to_mutate]), dtype=torch.bool)
                mask[remove_idx] = False
                expert_weights[layer_to_mutate] = expert_weights[layer_to_mutate][mask]

            # Renormalize
            expert_weights[layer_to_mutate] = torch.nn.functional.normalize(
                expert_weights[layer_to_mutate], dim=0
            )

        return ExpertVectorConfig(
            config_id=config_id,
            svd_components=svd_components,
            expert_weights=expert_weights,
            task_specialization=parent.task_specialization,
            discovery_method="mutation"
        )

    def _crossover_configurations(self) -> ExpertVectorConfig:
        """Crossover between two high-performing configurations"""
        # Select diverse parents for crossover
        parents = self.archive.get_diversity_sample(sample_size=2)
        if len(parents) < 2:
            parents = self.archive.get_best_configurations(top_k=2)

        parent1, parent2 = parents[0], parents[1]
        config_id = f"cross_{self.search_iteration}_{random.randint(1000, 9999)}"

        # Combine SVD components from both parents
        all_layers = set(parent1.svd_components.keys()) | set(parent2.svd_components.keys())
        svd_components = {}
        expert_weights = {}

        for layer_name in all_layers:
            components1 = parent1.svd_components.get(layer_name, [])
            components2 = parent2.svd_components.get(layer_name, [])

            # Combine components (union)
            combined_components = list(set(components1 + components2))
            svd_components[layer_name] = combined_components

            # Blend expert vectors
            if layer_name in parent1.expert_weights and layer_name in parent2.expert_weights:
                # Average the expert vectors (resize if needed)
                w1 = parent1.expert_weights[layer_name]
                w2 = parent2.expert_weights[layer_name]

                min_len = min(len(w1), len(w2))
                blended_weights = 0.5 * (w1[:min_len] + w2[:min_len])

                # Extend with remaining components if needed
                if len(combined_components) > min_len:
                    additional = torch.randn(
                        len(combined_components) - min_len,
                        device=self.device
                    )
                    blended_weights = torch.cat([blended_weights, additional])

                expert_weights[layer_name] = torch.nn.functional.normalize(blended_weights, dim=0)

            elif layer_name in parent1.expert_weights:
                expert_weights[layer_name] = parent1.expert_weights[layer_name].clone()
            elif layer_name in parent2.expert_weights:
                expert_weights[layer_name] = parent2.expert_weights[layer_name].clone()

        # Inherit task specialization from better parent
        better_parent = parent1 if parent1.performance_score > parent2.performance_score else parent2

        return ExpertVectorConfig(
            config_id=config_id,
            svd_components=svd_components,
            expert_weights=expert_weights,
            task_specialization=better_parent.task_specialization,
            discovery_method="crossover"
        )

    def _generate_task_specialized_configuration(self) -> ExpertVectorConfig:
        """Generate configuration specialized for specific task type"""
        config_id = f"specialized_{self.search_iteration}_{random.randint(1000, 9999)}"
        task_spec = random.choice(self.task_specializations)

        svd_components = {}
        expert_weights = {}

        # Task-specific layer selection strategy
        layer_priorities = self._get_task_specific_layer_priorities(task_spec)
        selected_layers = list(layer_priorities.keys())[:3]  # Top 3 layers for task

        for layer_name in selected_layers:
            if layer_name not in self.svd_cache:
                continue

            svd_data = self.svd_cache[layer_name]
            effective_rank = svd_data['effective_rank']
            priority = layer_priorities[layer_name]

            # Select more components for higher priority layers
            num_components = max(1, int(priority * 15))  # Scale by priority
            num_components = min(num_components, effective_rank)

            # Select top singular value components for specialized tasks
            selected_indices = list(range(num_components))
            svd_components[layer_name] = selected_indices

            # Generate task-specialized expert vector
            expert_vector = torch.randn(num_components, device=self.device)

            # Apply task-specific transformations
            if task_spec == "reasoning":
                expert_vector[0] *= 2.0  # Boost primary component
            elif task_spec == "creativity":
                expert_vector = expert_vector * torch.rand_like(expert_vector)  # Add randomness
            elif task_spec == "math":
                expert_vector = torch.abs(expert_vector)  # Use positive values

            expert_vector = torch.nn.functional.normalize(expert_vector, dim=0)
            expert_weights[layer_name] = expert_vector

        return ExpertVectorConfig(
            config_id=config_id,
            svd_components=svd_components,
            expert_weights=expert_weights,
            task_specialization=task_spec,
            discovery_method="task_specialized"
        )

    def _get_task_specific_layer_priorities(self, task_spec: str) -> Dict[str, float]:
        """Get layer priorities based on task specialization"""
        priorities = {}

        for layer_name in self.svd_cache.keys():
            # Heuristic layer prioritization based on task
            if "attention" in layer_name.lower():
                if task_spec in ["reasoning", "analysis"]:
                    priorities[layer_name] = 0.9
                elif task_spec in ["creativity", "language"]:
                    priorities[layer_name] = 0.7
                else:
                    priorities[layer_name] = 0.5
            elif "ffn" in layer_name.lower() or "mlp" in layer_name.lower():
                if task_spec in ["math", "coding"]:
                    priorities[layer_name] = 0.8
                else:
                    priorities[layer_name] = 0.6
            else:
                priorities[layer_name] = 0.4

        return priorities

    def _evaluate_expert_configuration(
        self,
        config: ExpertVectorConfig,
        evaluation_function: Optional[Callable] = None
    ) -> float:
        """Evaluate performance of an expert vector configuration"""
        if evaluation_function:
            try:
                return evaluation_function(config, self.target_model)
            except Exception as e:
                logger.warning(f"Custom evaluation failed: {e}")

        # Default evaluation: synthetic performance based on configuration properties
        score = 0.0

        # Reward configurations that use multiple layers
        score += len(config.svd_components) * 0.1

        # Reward configurations with balanced component selection
        component_counts = [len(components) for components in config.svd_components.values()]
        if component_counts:
            avg_components = np.mean(component_counts)
            score += min(avg_components / 10.0, 0.5)  # Up to 0.5 points

        # Add some randomness to simulate task performance variation
        score += random.uniform(-0.2, 0.3)

        # Ensure score is in reasonable range
        return max(0.0, min(1.0, score))

    def _log_discovery_progress(self, new_configurations: List[ExpertVectorConfig]):
        """Log progress of discovery process"""
        if not new_configurations:
            return

        best_config = max(new_configurations, key=lambda x: x.performance_score)

        log_entry = {
            "iteration": self.search_iteration,
            "best_config_id": best_config.config_id,
            "best_score": best_config.performance_score,
            "task_specialization": best_config.task_specialization,
            "discovery_method": best_config.discovery_method,
            "layers_used": list(best_config.svd_components.keys()),
            "total_components": sum(len(comp) for comp in best_config.svd_components.values())
        }

        self.discovery_log.append(log_entry)

        logger.info(
            f"Best config: {best_config.config_id} "
            f"(score: {best_config.performance_score:.3f}, "
            f"method: {best_config.discovery_method}, "
            f"task: {best_config.task_specialization})"
        )

    def get_best_expert_configuration(self, task_specialization: Optional[str] = None) -> Optional[ExpertVectorConfig]:
        """Get the best discovered configuration, optionally filtered by task"""
        if not self.archive.configurations:
            return None

        candidates = self.archive.configurations

        if task_specialization:
            candidates = [c for c in candidates if c.task_specialization == task_specialization]

        if not candidates:
            return None

        return max(candidates, key=lambda x: x.performance_score)

    def apply_expert_configuration(
        self,
        config: ExpertVectorConfig,
        strength: float = 1.0
    ) -> Dict[str, Any]:
        """Apply discovered expert configuration to target model"""
        logger.info(f"Applying expert configuration {config.config_id} with strength {strength}")

        modifications = {}

        with torch.no_grad():
            for layer_name, expert_vector in config.expert_weights.items():
                if layer_name not in self.svd_cache:
                    continue

                # Get original SVD components
                svd_data = self.svd_cache[layer_name]
                U, S, Vt = svd_data['U'], svd_data['S'], svd_data['Vt']

                # Get selected components
                selected_components = config.svd_components[layer_name]

                # Modify singular values using expert vector
                modified_S = S.clone()
                for i, component_idx in enumerate(selected_components):
                    if i < len(expert_vector) and component_idx < len(modified_S):
                        # Apply expert vector scaling
                        expert_scaling = 1.0 + strength * expert_vector[i].item()
                        modified_S[component_idx] *= expert_scaling

                # Reconstruct weight matrix
                reconstructed_weight = torch.matmul(
                    torch.matmul(U, torch.diag(modified_S)), Vt
                )

                # Apply to model
                for name, param in self.target_model.named_parameters():
                    if name == layer_name:
                        original_shape = param.shape
                        param.data = reconstructed_weight.reshape(original_shape).to(param.device)
                        modifications[layer_name] = {
                            "components_modified": len(selected_components),
                            "expert_vector_norm": torch.norm(expert_vector).item(),
                            "singular_value_changes": (modified_S - S).abs().mean().item()
                        }
                        break

        application_result = {
            "config_applied": config.config_id,
            "layers_modified": list(modifications.keys()),
            "total_modifications": sum(mod["components_modified"] for mod in modifications.values()),
            "modification_details": modifications,
            "application_strength": strength
        }

        logger.info(f"Applied expert configuration to {len(modifications)} layers")

        return application_result

    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get summary of discovery process"""
        if not self.archive.configurations:
            return {"status": "no_configurations_discovered"}

        # Analyze discovered configurations
        task_distribution = {}
        method_distribution = {}

        for config in self.archive.configurations:
            task = config.task_specialization
            method = config.discovery_method

            task_distribution[task] = task_distribution.get(task, 0) + 1
            method_distribution[method] = method_distribution.get(method, 0) + 1

        best_configs = self.archive.get_best_configurations(5)

        return {
            "total_configurations": len(self.archive.configurations),
            "search_iterations": self.search_iteration,
            "best_score": max(c.performance_score for c in self.archive.configurations),
            "task_distribution": task_distribution,
            "discovery_method_distribution": method_distribution,
            "top_5_configurations": [
                {
                    "config_id": c.config_id,
                    "score": c.performance_score,
                    "task": c.task_specialization,
                    "method": c.discovery_method
                } for c in best_configs
            ],
            "layers_analyzed": len(self.svd_cache),
            "discovery_log": self.discovery_log
        }