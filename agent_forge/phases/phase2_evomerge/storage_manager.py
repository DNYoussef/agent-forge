"""
Storage management for EvoMerge with n-2 generation deletion policy.
Keeps original 3 models + last 16 models (2 generations) to save disk space.
"""

import os
import shutil
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StorageManager:
    """
    Manages model storage with automatic cleanup.
    Implements n-2 generation deletion policy.
    """

    def __init__(self,
                 base_dir: str = "./models/evomerge",
                 keep_generations: int = 2,
                 population_size: int = 8):
        """
        Initialize storage manager.

        Args:
            base_dir: Base directory for storing models
            keep_generations: Number of recent generations to keep (default: 2)
            population_size: Number of models per generation (default: 8)
        """
        self.base_dir = Path(base_dir)
        self.keep_generations = keep_generations
        self.population_size = population_size

        # Create directory structure
        self.originals_dir = self.base_dir / "originals"
        self.generations_dir = self.base_dir / "generations"
        self.best_models_dir = self.base_dir / "best_models"

        # Create directories
        self.originals_dir.mkdir(parents=True, exist_ok=True)
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        self.best_models_dir.mkdir(parents=True, exist_ok=True)

        # Track current generation
        self.current_generation = 0

        # Storage statistics
        self.stats = {
            'models_stored': 0,
            'models_deleted': 0,
            'total_size_mb': 0,
            'generations_deleted': []
        }

    def save_original_models(self, models: List[nn.Module], metadata: Optional[List[Dict]] = None):
        """
        Save the original 3 models that will be kept permanently.

        Args:
            models: List of 3 original models
            metadata: Optional metadata for each model
        """
        if len(models) != 3:
            raise ValueError(f"Expected 3 original models, got {len(models)}")

        for i, model in enumerate(models):
            model_path = self.originals_dir / f"original_model_{i}.pt"

            # Prepare checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_index': i,
                'save_time': datetime.now().isoformat(),
                'type': 'original'
            }

            # Add metadata if provided
            if metadata and i < len(metadata):
                checkpoint['metadata'] = metadata[i]

            # Save model
            torch.save(checkpoint, model_path)
            logger.info(f"Saved original model {i} to {model_path}")

        # Update stats
        self.stats['models_stored'] += 3
        self._update_storage_size()

    def save_generation(self,
                        generation: int,
                        models: List[nn.Module],
                        fitness_scores: List[float],
                        metadata: Optional[Dict] = None):
        """
        Save models from a generation with automatic cleanup of old generations.

        Args:
            generation: Generation number
            models: List of models in this generation
            fitness_scores: Fitness scores for each model
            metadata: Optional additional metadata
        """
        self.current_generation = generation

        # Create generation directory
        gen_dir = self.generations_dir / f"generation_{generation:04d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Save each model
        for i, (model, fitness) in enumerate(zip(models, fitness_scores)):
            model_path = gen_dir / f"model_{i}_fitness_{fitness:.4f}.pt"

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'generation': generation,
                'model_index': i,
                'fitness_score': fitness,
                'save_time': datetime.now().isoformat()
            }

            if metadata:
                checkpoint['metadata'] = metadata

            torch.save(checkpoint, model_path)

        # Save generation summary
        summary = {
            'generation': generation,
            'num_models': len(models),
            'fitness_scores': fitness_scores,
            'best_fitness': max(fitness_scores),
            'average_fitness': sum(fitness_scores) / len(fitness_scores),
            'save_time': datetime.now().isoformat()
        }

        with open(gen_dir / 'generation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved generation {generation} with {len(models)} models")
        self.stats['models_stored'] += len(models)

        # Perform cleanup of old generations
        self._cleanup_old_generations()

    def _cleanup_old_generations(self):
        """
        Remove generations older than n-keep_generations to save space.
        Implements the n-2 policy by default.
        """
        # Get all generation directories
        gen_dirs = sorted([
            d for d in self.generations_dir.iterdir()
            if d.is_dir() and d.name.startswith('generation_')
        ])

        # Calculate how many to delete
        num_to_delete = max(0, len(gen_dirs) - self.keep_generations)

        if num_to_delete > 0:
            dirs_to_delete = gen_dirs[:num_to_delete]

            for gen_dir in dirs_to_delete:
                # Count models before deletion
                models_in_dir = len(list(gen_dir.glob('*.pt')))
                self.stats['models_deleted'] += models_in_dir

                # Extract generation number for stats
                gen_num = int(gen_dir.name.split('_')[-1])
                self.stats['generations_deleted'].append(gen_num)

                # Delete the directory
                shutil.rmtree(gen_dir)
                logger.info(f"Deleted generation directory: {gen_dir.name}")

        # Update storage size
        self._update_storage_size()

    def save_best_model(self, model: nn.Module, generation: int, fitness: float, is_final: bool = False):
        """
        Save the best model from evolution.

        Args:
            model: The best model
            generation: Generation where this model was found
            fitness: Fitness score
            is_final: If True, this is the final best model
        """
        if is_final:
            model_path = self.best_models_dir / f"final_best_model_gen{generation}_fitness{fitness:.4f}.pt"
        else:
            model_path = self.best_models_dir / f"best_model_gen{generation}_fitness{fitness:.4f}.pt"

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'generation': generation,
            'fitness_score': fitness,
            'is_final': is_final,
            'save_time': datetime.now().isoformat()
        }

        torch.save(checkpoint, model_path)
        logger.info(f"Saved {'final' if is_final else ''} best model from generation {generation}")

    def cleanup_all_except_best(self, best_model: nn.Module, best_fitness: float):
        """
        Final cleanup - keep only the best model and originals.
        This is called at the end of evolution.

        Args:
            best_model: The final best model
            best_fitness: Its fitness score
        """
        logger.info("Performing final cleanup - keeping only best model and originals")

        # Save the final best model
        self.save_best_model(best_model, self.current_generation, best_fitness, is_final=True)

        # Delete all generation directories
        for gen_dir in self.generations_dir.iterdir():
            if gen_dir.is_dir():
                models_in_dir = len(list(gen_dir.glob('*.pt')))
                self.stats['models_deleted'] += models_in_dir
                shutil.rmtree(gen_dir)
                logger.info(f"Deleted generation directory: {gen_dir.name}")

        # Delete non-final best models
        for model_file in self.best_models_dir.glob('*.pt'):
            if 'final' not in model_file.name:
                model_file.unlink()
                self.stats['models_deleted'] += 1

        logger.info(f"Final cleanup complete. Kept {self._count_remaining_models()} models")
        self._update_storage_size()

    def load_generation(self, generation: int) -> List[Dict]:
        """
        Load models from a specific generation.

        Args:
            generation: Generation number to load

        Returns:
            List of checkpoint dictionaries
        """
        gen_dir = self.generations_dir / f"generation_{generation:04d}"

        if not gen_dir.exists():
            raise ValueError(f"Generation {generation} not found")

        checkpoints = []
        for model_file in sorted(gen_dir.glob('*.pt')):
            checkpoint = torch.load(model_file, map_location='cpu')
            checkpoint['file_path'] = str(model_file)
            checkpoints.append(checkpoint)

        return checkpoints

    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        self._update_storage_size()

        remaining_models = self._count_remaining_models()
        remaining_generations = len(list(self.generations_dir.iterdir()))

        return {
            'total_models_stored': self.stats['models_stored'],
            'total_models_deleted': self.stats['models_deleted'],
            'current_models': remaining_models,
            'current_generations': remaining_generations,
            'total_size_mb': self.stats['total_size_mb'],
            'generations_deleted': self.stats['generations_deleted'],
            'keep_policy': f"n-{self.keep_generations}",
            'originals_kept': len(list(self.originals_dir.glob('*.pt'))),
            'best_models_kept': len(list(self.best_models_dir.glob('*.pt')))
        }

    def _count_remaining_models(self) -> int:
        """Count total remaining models across all directories."""
        total = 0

        # Count originals
        total += len(list(self.originals_dir.glob('*.pt')))

        # Count in generations
        for gen_dir in self.generations_dir.iterdir():
            if gen_dir.is_dir():
                total += len(list(gen_dir.glob('*.pt')))

        # Count best models
        total += len(list(self.best_models_dir.glob('*.pt')))

        return total

    def _update_storage_size(self):
        """Update total storage size in MB."""
        total_size = 0

        for path in self.base_dir.rglob('*.pt'):
            total_size += path.stat().st_size

        self.stats['total_size_mb'] = total_size / (1024 * 1024)

    def get_model_lineage(self, generation: int) -> Dict[str, Any]:
        """
        Get the lineage/history of models up to a generation.

        Args:
            generation: Target generation

        Returns:
            Dictionary containing model lineage information
        """
        lineage = {
            'originals': [],
            'generations': {},
            'best_models': []
        }

        # Get originals
        for model_file in sorted(self.originals_dir.glob('*.pt')):
            checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
            lineage['originals'].append({
                'file': model_file.name,
                'index': checkpoint.get('model_index'),
                'metadata': checkpoint.get('metadata', {})
            })

        # Get generations up to target
        for gen in range(max(0, generation - self.keep_generations + 1), generation + 1):
            gen_dir = self.generations_dir / f"generation_{gen:04d}"
            if gen_dir.exists():
                summary_file = gen_dir / 'generation_summary.json'
                if summary_file.exists():
                    with open(summary_file, 'r') as f:
                        lineage['generations'][gen] = json.load(f)

        # Get best models
        for model_file in sorted(self.best_models_dir.glob('*.pt')):
            checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
            lineage['best_models'].append({
                'file': model_file.name,
                'generation': checkpoint.get('generation'),
                'fitness': checkpoint.get('fitness_score'),
                'is_final': checkpoint.get('is_final', False)
            })

        return lineage

    def cleanup(self):
        """Clean up all temporary files and caches."""
        # This can be called to free up space during long runs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("Cleaned up caches")