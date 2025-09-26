"""
Tournament Selection Algorithm for EvoMerge Phase.
Implements specific selection rules:
- Top 2 models → 6 children (3 mutations each)
- Bottom 6 models → 2 children (grouped into 2 sets of 3, merged)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any
import copy
import random
import logging

logger = logging.getLogger(__name__)

class TournamentSelection:
    """
    Implements the specific tournament selection algorithm for EvoMerge.
    """

    def __init__(self, merge_techniques, mutation_rate: float = 0.1):
        self.merge_techniques = merge_techniques
        self.mutation_rate = mutation_rate
        self.generation_tree = []  # Track evolution history

    def create_initial_population(self, cognate_models: List[nn.Module]) -> Tuple[List[nn.Module], List[Dict]]:
        """
        Create initial 8 models from 3 Cognate models using different merge techniques.

        Args:
            cognate_models: List of 3 Cognate models (~25M params each)

        Returns:
            Tuple of (8 merged models, lineage info)
        """
        if len(cognate_models) != 3:
            raise ValueError(f"Expected exactly 3 Cognate models, got {len(cognate_models)}")

        population = []
        lineage = []

        # Create pairs: (M1,M2), (M1,M3), (M2,M3)
        pairs = [
            (cognate_models[0], cognate_models[1], "M1+M2"),
            (cognate_models[0], cognate_models[2], "M1+M3"),
            (cognate_models[1], cognate_models[2], "M2+M3")
        ]

        # Apply different merge techniques to get 8 total combinations
        techniques = ['linear', 'slerp', 'ties', 'dare', 'frankenmerge', 'dfs']

        model_id = 0
        for pair_idx, (model_a, model_b, pair_name) in enumerate(pairs):
            # Use 2-3 techniques per pair to get ~8 total
            techs_for_pair = techniques[pair_idx*2:(pair_idx+1)*2+1] if pair_idx < 2 else techniques[-2:]

            for technique in techs_for_pair[:3 if pair_idx == 0 else 2]:  # Distribute to get 8 total
                if model_id >= 8:
                    break

                # Apply merge technique
                merged_model = self._apply_merge_technique(model_a, model_b, technique)
                population.append(merged_model)

                # Track lineage
                lineage.append({
                    'id': f"G0_M{model_id}",
                    'generation': 0,
                    'parents': pair_name,
                    'technique': technique,
                    'type': 'initial_merge'
                })

                model_id += 1

        # Fill to exactly 8 if needed
        while len(population) < 8:
            # Use linear merge as default
            pair = random.choice(pairs)
            merged = self._apply_merge_technique(pair[0], pair[1], 'linear')
            population.append(merged)
            lineage.append({
                'id': f"G0_M{len(population)-1}",
                'generation': 0,
                'parents': pair[2],
                'technique': 'linear',
                'type': 'initial_merge'
            })

        logger.info(f"Created initial population of {len(population)} models")
        return population[:8], lineage[:8]

    def tournament_step(self, population: List[nn.Module], fitness_scores: List[float],
                       generation: int) -> Tuple[List[nn.Module], List[Dict]]:
        """
        Execute one tournament selection step.

        Top 2 models → 6 children (3 mutations each)
        Bottom 6 models → 2 children (grouped into 2 sets of 3, merged)

        Args:
            population: Current generation of 8 models
            fitness_scores: Fitness scores for each model
            generation: Current generation number

        Returns:
            Tuple of (new population of 8 models, lineage info)
        """
        if len(population) != 8 or len(fitness_scores) != 8:
            raise ValueError("Tournament expects exactly 8 models with fitness scores")

        # Sort models by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order
        sorted_models = [population[i] for i in sorted_indices]
        sorted_scores = [fitness_scores[i] for i in sorted_indices]

        new_population = []
        new_lineage = []

        # TOP 2 WINNERS → 6 CHILDREN (3 mutations each)
        winners = sorted_models[:2]
        winner_scores = sorted_scores[:2]

        for winner_idx, winner_model in enumerate(winners):
            for mutation_idx in range(3):
                # Create mutated child
                mutated_child = self._mutate_model(winner_model, strength=self.mutation_rate)
                new_population.append(mutated_child)

                # Track lineage
                new_lineage.append({
                    'id': f"G{generation}_W{winner_idx}_M{mutation_idx}",
                    'generation': generation,
                    'parent': f"Winner_{winner_idx}",
                    'parent_fitness': winner_scores[winner_idx],
                    'technique': 'mutation',
                    'type': 'winner_child',
                    'mutation_strength': self.mutation_rate
                })

        # BOTTOM 6 LOSERS → 2 CHILDREN (2 groups of 3, merged)
        losers = sorted_models[2:8]  # Bottom 6
        loser_scores = sorted_scores[2:8]

        # Group losers into 2 sets of 3
        loser_groups = [
            losers[:3],  # First 3 losers
            losers[3:6]  # Last 3 losers
        ]

        for group_idx, group in enumerate(loser_groups):
            # Merge the group of 3 using a random technique
            technique = random.choice(['linear', 'ties', 'dare', 'frankenmerge'])
            merged_child = self._merge_group(group, technique)
            new_population.append(merged_child)

            # Track lineage
            new_lineage.append({
                'id': f"G{generation}_L{group_idx}",
                'generation': generation,
                'parents': f"Loser_Group_{group_idx}",
                'parent_fitness': np.mean(loser_scores[group_idx*3:(group_idx+1)*3]),
                'technique': technique,
                'type': 'loser_merge',
                'chaos_preservation': True
            })

        # Store in generation tree for visualization
        self.generation_tree.append({
            'generation': generation,
            'winners': winner_scores[:2],
            'losers': loser_scores,
            'children': new_lineage
        })

        logger.info(f"Tournament generation {generation}: Created 6 winner children + 2 loser children")
        return new_population[:8], new_lineage

    def _apply_merge_technique(self, model_a: nn.Module, model_b: nn.Module, technique: str) -> nn.Module:
        """Apply specific merge technique to two models."""
        models = [model_a, model_b]

        if technique == 'linear':
            weights = [0.5, 0.5]  # Equal weights
            return self.merge_techniques.linear_merge(models, weights)
        elif technique == 'slerp':
            return self.merge_techniques.slerp_merge(models, t=0.5)
        elif technique == 'ties':
            return self.merge_techniques.ties_merge(models, threshold=0.7)
        elif technique == 'dare':
            return self.merge_techniques.dare_merge(models, drop_rate=0.5)
        elif technique == 'frankenmerge':
            return self.merge_techniques.frankenmerge(models)
        elif technique == 'dfs':
            return self.merge_techniques.dfs_merge(models)
        else:
            # Default to linear
            return self.merge_techniques.linear_merge(models, [0.5, 0.5])

    def _merge_group(self, models: List[nn.Module], technique: str) -> nn.Module:
        """Merge a group of 3 models."""
        if technique == 'linear':
            weights = [1/3, 1/3, 1/3]  # Equal weights for 3 models
            return self.merge_techniques.linear_merge(models, weights)
        elif technique == 'ties':
            return self.merge_techniques.ties_merge(models, threshold=0.6)
        elif technique == 'dare':
            return self.merge_techniques.dare_merge(models, drop_rate=0.4)
        elif technique == 'frankenmerge':
            return self.merge_techniques.frankenmerge(models)
        else:
            # Default to linear with equal weights
            return self.merge_techniques.linear_merge(models, [1/3, 1/3, 1/3])

    def _mutate_model(self, model: nn.Module, strength: float) -> nn.Module:
        """
        Mutate a model by adding controlled noise to weights.

        Args:
            model: Model to mutate
            strength: Mutation strength (0.0 to 1.0)

        Returns:
            Mutated copy of the model
        """
        mutated = copy.deepcopy(model)

        with torch.no_grad():
            for param in mutated.parameters():
                # Add Gaussian noise scaled by parameter magnitude
                noise = torch.randn_like(param) * param.abs() * strength
                param.add_(noise)

        return mutated

    def get_evolution_tree(self) -> List[Dict]:
        """Get the complete evolution tree for visualization."""
        return self.generation_tree

    def check_diminishing_returns(self, fitness_history: List[float],
                                 threshold: float = 0.001,
                                 patience: int = 3) -> bool:
        """
        Check if we have diminishing returns (3 consecutive tests with minimal improvement).

        Args:
            fitness_history: List of best fitness scores per generation
            threshold: Minimum improvement threshold
            patience: Number of generations to check (default 3)

        Returns:
            True if diminishing returns detected
        """
        if len(fitness_history) < patience + 1:
            return False

        # Check last 'patience' improvements
        for i in range(1, patience + 1):
            improvement = fitness_history[-i] - fitness_history[-i-1]
            if improvement > threshold:
                return False  # Still improving

        logger.info(f"Diminishing returns detected after {len(fitness_history)} generations")
        return True