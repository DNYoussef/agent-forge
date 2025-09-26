"""
Comprehensive training checkpoint and recovery system for Agent Forge.
Provides multiple layers of error detection and recovery with state preservation.
"""

import os
import json
import hashlib
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pickle
import torch
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint validation and recovery."""
    session_id: str
    model_idx: int
    step: int
    phase: str
    timestamp: str
    checksum: str
    model_config: Dict[str, Any]
    training_metrics: Dict[str, float]
    file_paths: List[str]
    integrity_verified: bool = False


class CheckpointManager:
    """Comprehensive training checkpoint and recovery system"""

    def __init__(self, session_id: str, save_dir: str = "checkpoints"):
        self.session_id = session_id
        self.save_dir = Path(save_dir) / session_id
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Checkpoint configuration
        self.auto_save_interval = 100  # steps
        self.max_checkpoints_per_model = 3
        self.compression_enabled = True

        # Recovery state tracking
        self.recovery_log = []

        self.logger.info(f"CheckpointManager initialized for session: {session_id}")

    def _generate_checksum(self, file_paths: List[str]) -> str:
        """Generate checksum for checkpoint integrity validation."""
        hasher = hashlib.sha256()

        for file_path in sorted(file_paths):
            if Path(file_path).exists():
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)

        return hasher.hexdigest()

    def save_training_checkpoint(self, model_idx: int, step: int,
                               model_state: dict, optimizer_state: dict,
                               training_metrics: dict, phase: str = "training") -> str:
        """Save complete training state for recovery"""

        checkpoint_dir = self.save_dir / f"model_{model_idx}" / f"step_{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_path = checkpoint_dir / "model_state.pt"
        torch.save(model_state, model_path)

        # Save optimizer state
        optimizer_path = checkpoint_dir / "optimizer_state.pt"
        torch.save(optimizer_state, optimizer_path)

        # Save training metrics
        metrics_path = checkpoint_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(training_metrics, f, indent=2)

        # Save random states for reproducibility
        random_state = {
            'numpy_random': np.random.get_state(),
            'torch_random': torch.get_rng_state(),
            'torch_cuda_random': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
        random_path = checkpoint_dir / "random_state.pkl"
        with open(random_path, 'wb') as f:
            pickle.dump(random_state, f)

        # Generate and save metadata
        file_paths = [str(model_path), str(optimizer_path), str(metrics_path), str(random_path)]
        checksum = self._generate_checksum(file_paths)

        metadata = CheckpointMetadata(
            session_id=self.session_id,
            model_idx=model_idx,
            step=step,
            phase=phase,
            timestamp=datetime.now().isoformat(),
            checksum=checksum,
            model_config=model_state.get('config', {}),
            training_metrics=training_metrics,
            file_paths=file_paths,
            integrity_verified=True
        )

        metadata_path = checkpoint_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)

        self.logger.info(f"Checkpoint saved for model {model_idx} at step {step}")

        # Cleanup old checkpoints
        self.cleanup_old_checkpoints(model_idx, self.max_checkpoints_per_model)

        return str(checkpoint_dir)

    def load_training_checkpoint(self, model_idx: int, step: Optional[int] = None) -> Optional[dict]:
        """Load most recent checkpoint for model recovery"""

        model_dir = self.save_dir / f"model_{model_idx}"
        if not model_dir.exists():
            self.logger.warning(f"No checkpoints found for model {model_idx}")
            return None

        # Find target checkpoint
        if step is not None:
            checkpoint_dir = model_dir / f"step_{step}"
            if not checkpoint_dir.exists():
                self.logger.warning(f"Checkpoint not found for model {model_idx} at step {step}")
                return None
        else:
            # Find most recent checkpoint
            step_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('step_')]
            if not step_dirs:
                self.logger.warning(f"No step checkpoints found for model {model_idx}")
                return None

            step_dirs.sort(key=lambda x: int(x.name.split('_')[1]), reverse=True)
            checkpoint_dir = step_dirs[0]

        # Verify checkpoint integrity
        if not self._verify_checkpoint_integrity(checkpoint_dir):
            self.logger.error(f"Checkpoint integrity check failed: {checkpoint_dir}")
            return None

        try:
            # Load checkpoint components
            model_state = torch.load(checkpoint_dir / "model_state.pt", map_location='cpu')
            optimizer_state = torch.load(checkpoint_dir / "optimizer_state.pt", map_location='cpu')

            with open(checkpoint_dir / "training_metrics.json", 'r') as f:
                training_metrics = json.load(f)

            with open(checkpoint_dir / "random_state.pkl", 'rb') as f:
                random_state = pickle.load(f)

            with open(checkpoint_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)

            # Restore random states
            if random_state['numpy_random'] is not None:
                np.random.set_state(random_state['numpy_random'])
            if random_state['torch_random'] is not None:
                torch.set_rng_state(random_state['torch_random'])
            if random_state['torch_cuda_random'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(random_state['torch_cuda_random'])

            checkpoint_data = {
                'model_state': model_state,
                'optimizer_state': optimizer_state,
                'training_metrics': training_metrics,
                'metadata': metadata,
                'random_state': random_state
            }

            self.logger.info(f"Successfully loaded checkpoint for model {model_idx} from step {metadata['step']}")
            return checkpoint_data

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None

    def save_phase_checkpoint(self, phase: str, results: dict) -> str:
        """Save completed phase results"""

        phase_dir = self.save_dir / "phases" / phase
        phase_dir.mkdir(parents=True, exist_ok=True)

        # Save phase results
        results_path = phase_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Save timestamp
        timestamp_data = {
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id
        }

        timestamp_path = phase_dir / "timestamp.json"
        with open(timestamp_path, 'w') as f:
            json.dump(timestamp_data, f, indent=2)

        self.logger.info(f"Phase checkpoint saved for: {phase}")
        return str(phase_dir)

    def get_recovery_options(self) -> List[str]:
        """List available recovery points"""

        recovery_options = []

        # Phase checkpoints
        phase_dir = self.save_dir / "phases"
        if phase_dir.exists():
            for phase_checkpoint in phase_dir.iterdir():
                if phase_checkpoint.is_dir():
                    recovery_options.append(f"phase:{phase_checkpoint.name}")

        # Model checkpoints
        for model_dir in self.save_dir.iterdir():
            if model_dir.is_dir() and model_dir.name.startswith('model_'):
                model_idx = model_dir.name.split('_')[1]
                step_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('step_')]

                for step_dir in sorted(step_dirs, key=lambda x: int(x.name.split('_')[1]), reverse=True):
                    step = step_dir.name.split('_')[1]
                    recovery_options.append(f"model:{model_idx}:step:{step}")

        return recovery_options

    def cleanup_old_checkpoints(self, model_idx: int, keep_last: int = 3):
        """Remove old checkpoints to save space"""

        model_dir = self.save_dir / f"model_{model_idx}"
        if not model_dir.exists():
            return

        step_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('step_')]
        step_dirs.sort(key=lambda x: int(x.name.split('_')[1]), reverse=True)

        # Remove old checkpoints
        for old_dir in step_dirs[keep_last:]:
            try:
                shutil.rmtree(old_dir)
                self.logger.info(f"Cleaned up old checkpoint: {old_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up checkpoint {old_dir}: {e}")

    def _verify_checkpoint_integrity(self, checkpoint_dir: Path) -> bool:
        """Verify checkpoint integrity using checksums"""

        metadata_path = checkpoint_dir / "metadata.json"
        if not metadata_path.exists():
            return False

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Verify all files exist
            for file_path in metadata['file_paths']:
                if not Path(file_path).exists():
                    self.logger.error(f"Missing checkpoint file: {file_path}")
                    return False

            # Verify checksum
            current_checksum = self._generate_checksum(metadata['file_paths'])
            if current_checksum != metadata['checksum']:
                self.logger.error(f"Checksum mismatch for checkpoint: {checkpoint_dir}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Checkpoint integrity verification failed: {e}")
            return False

    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get comprehensive checkpoint statistics"""

        stats = {
            'session_id': self.session_id,
            'total_models': 0,
            'total_checkpoints': 0,
            'total_size_mb': 0,
            'phases_completed': 0,
            'models': {}
        }

        # Count model checkpoints
        for item in self.save_dir.iterdir():
            if item.is_dir() and item.name.startswith('model_'):
                model_idx = item.name.split('_')[1]
                stats['total_models'] += 1

                step_dirs = [d for d in item.iterdir() if d.is_dir() and d.name.startswith('step_')]
                model_checkpoints = len(step_dirs)
                stats['total_checkpoints'] += model_checkpoints

                # Calculate size
                model_size = sum(
                    f.stat().st_size for f in item.rglob('*') if f.is_file()
                ) / (1024 * 1024)  # Convert to MB

                stats['models'][model_idx] = {
                    'checkpoints': model_checkpoints,
                    'size_mb': round(model_size, 2),
                    'latest_step': max([int(d.name.split('_')[1]) for d in step_dirs]) if step_dirs else 0
                }

                stats['total_size_mb'] += model_size

        # Count phase checkpoints
        phase_dir = self.save_dir / "phases"
        if phase_dir.exists():
            stats['phases_completed'] = len([
                d for d in phase_dir.iterdir() if d.is_dir()
            ])

        stats['total_size_mb'] = round(stats['total_size_mb'], 2)

        return stats


class TrainingErrorHandler:
    """Classify and handle different types of training failures"""

    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
        self.logger = logging.getLogger(__name__)

        # Error classification patterns
        self.error_patterns = {
            'cuda_oom': ['out of memory', 'cuda error', 'gpu memory', 'cuda out of memory'],
            'data_corruption': ['corrupted', 'invalid tensor', 'bad data', 'data loading error'],
            'network_failure': ['connection', 'timeout', 'network', 'http error'],
            'resource_exhaustion': ['disk space', 'insufficient memory', 'no space left', 'resource limit'],
            'checkpoint_corruption': ['checkpoint', 'state_dict', 'cannot load', 'pickle error'],
            'convergence_failure': ['nan', 'inf', 'loss exploded', 'gradient exploded'],
            'configuration_error': ['config', 'parameter', 'invalid argument', 'type error']
        }

        # Recovery strategies
        self.recovery_strategies = {
            'cuda_oom': self._recover_from_memory_error,
            'data_corruption': self._recover_from_data_error,
            'network_failure': self._recover_from_network_error,
            'resource_exhaustion': self._recover_from_resource_error,
            'checkpoint_corruption': self._recover_from_checkpoint_error,
            'convergence_failure': self._recover_from_convergence_error,
            'configuration_error': self._recover_from_config_error
        }

    def classify_error(self, exception: Exception) -> str:
        """Classify error type for appropriate recovery strategy"""

        error_message = str(exception).lower()

        # Check each error pattern
        for error_type, patterns in self.error_patterns.items():
            if any(pattern in error_message for pattern in patterns):
                self.logger.info(f"Classified error as: {error_type}")
                return error_type

        # Default to unknown error
        self.logger.warning(f"Unknown error type: {error_message}")
        return 'unknown'

    def recover_from_error(self, error_type: str, context: dict) -> bool:
        """Attempt automatic recovery based on error type"""

        if error_type not in self.recovery_strategies:
            self.logger.error(f"No recovery strategy for error type: {error_type}")
            return False

        try:
            recovery_function = self.recovery_strategies[error_type]
            return recovery_function(context)

        except Exception as e:
            self.logger.error(f"Recovery strategy failed: {e}")
            return False

    def _recover_from_memory_error(self, context: dict) -> bool:
        """Recover from CUDA out of memory errors"""

        self.logger.info("Attempting memory error recovery...")

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Suggest smaller batch size
        current_batch_size = context.get('batch_size', 32)
        new_batch_size = max(1, current_batch_size // 2)

        context['recovery_suggestions'] = [
            f"Reduce batch size from {current_batch_size} to {new_batch_size}",
            "Enable gradient checkpointing",
            "Use mixed precision training",
            "Load checkpoint and resume with smaller memory footprint"
        ]

        return True

    def _recover_from_data_error(self, context: dict) -> bool:
        """Recover from data corruption errors"""

        self.logger.info("Attempting data error recovery...")

        # Find last good checkpoint
        recovery_options = self.checkpoint_manager.get_recovery_options()
        model_checkpoints = [opt for opt in recovery_options if opt.startswith('model:')]

        if model_checkpoints:
            latest_checkpoint = model_checkpoints[0]  # Most recent
            context['recovery_suggestions'] = [
                f"Resume from checkpoint: {latest_checkpoint}",
                "Validate data integrity",
                "Re-download corrupted data",
                "Skip corrupted batches"
            ]
            return True

        return False

    def _recover_from_network_error(self, context: dict) -> bool:
        """Recover from network failures"""

        self.logger.info("Attempting network error recovery...")

        context['recovery_suggestions'] = [
            "Retry with exponential backoff",
            "Switch to offline mode if available",
            "Use cached data",
            "Resume from last checkpoint"
        ]

        return True

    def _recover_from_resource_error(self, context: dict) -> bool:
        """Recover from resource exhaustion"""

        self.logger.info("Attempting resource error recovery...")

        # Clean up temporary files
        self.checkpoint_manager.cleanup_old_checkpoints(context.get('model_idx', 0), keep_last=1)

        context['recovery_suggestions'] = [
            "Free up disk space",
            "Clean up old checkpoints",
            "Reduce model size",
            "Use streaming data loading"
        ]

        return True

    def _recover_from_checkpoint_error(self, context: dict) -> bool:
        """Recover from checkpoint corruption"""

        self.logger.info("Attempting checkpoint error recovery...")

        # Find alternative checkpoints
        recovery_options = self.checkpoint_manager.get_recovery_options()

        context['recovery_suggestions'] = [
            "Use previous checkpoint",
            "Restart training from scratch",
            "Verify checkpoint integrity",
            f"Available recovery options: {recovery_options}"
        ]

        return len(recovery_options) > 0

    def _recover_from_convergence_error(self, context: dict) -> bool:
        """Recover from convergence failures (NaN/Inf)"""

        self.logger.info("Attempting convergence error recovery...")

        context['recovery_suggestions'] = [
            "Reduce learning rate",
            "Add gradient clipping",
            "Use more stable optimizer (Adam -> AdamW)",
            "Resume from stable checkpoint",
            "Check for numerical instability"
        ]

        return True

    def _recover_from_config_error(self, context: dict) -> bool:
        """Recover from configuration errors"""

        self.logger.info("Attempting configuration error recovery...")

        context['recovery_suggestions'] = [
            "Validate configuration parameters",
            "Reset to default configuration",
            "Check parameter types and ranges",
            "Use configuration from successful run"
        ]

        return True


# Export classes
__all__ = ['CheckpointManager', 'TrainingErrorHandler', 'CheckpointMetadata']