"""
Progress Aggregator - Parallel Training Progress Management

Aggregates and reports progress from multiple parallel training processes.
"""

import threading
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

class TrainingPhase(Enum):
    """Training phases for progress tracking"""
    INITIALIZING = "initializing"
    TRAINING = "training"
    CHECKPOINTING = "checkpointing"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ModelProgress:
    """Progress information for a single model"""
    model_id: int
    phase: TrainingPhase
    current_epoch: int
    total_epochs: int
    progress_percent: float
    loss_value: Optional[float] = None
    learning_rate: Optional[float] = None
    time_elapsed: float = 0.0
    estimated_remaining: Optional[float] = None
    last_update: float = 0.0


class ProgressAggregator:
    """
    Aggregates progress from multiple parallel training processes

    Provides real-time progress updates and consolidated reporting
    for parallel model training.
    """

    def __init__(self, total_models: int = 3):
        self.total_models = total_models
        self._model_progress: Dict[int, ModelProgress] = {}
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._completed_models = 0

    def update_model_progress(
        self,
        model_id: int,
        phase: TrainingPhase,
        current_epoch: int = 0,
        total_epochs: int = 100,
        loss_value: Optional[float] = None,
        learning_rate: Optional[float] = None
    ) -> None:
        """
        Update progress for a specific model

        Args:
            model_id: Unique model identifier
            phase: Current training phase
            current_epoch: Current epoch number
            total_epochs: Total number of epochs
            loss_value: Current loss value (optional)
            learning_rate: Current learning rate (optional)
        """
        with self._lock:
            progress_percent = (current_epoch / total_epochs * 100) if total_epochs > 0 else 0.0
            time_elapsed = time.time() - self._start_time

            # Estimate remaining time based on current progress
            estimated_remaining = None
            if progress_percent > 0:
                estimated_total = time_elapsed / (progress_percent / 100)
                estimated_remaining = max(0, estimated_total - time_elapsed)

            self._model_progress[model_id] = ModelProgress(
                model_id=model_id,
                phase=phase,
                current_epoch=current_epoch,
                total_epochs=total_epochs,
                progress_percent=progress_percent,
                loss_value=loss_value,
                learning_rate=learning_rate,
                time_elapsed=time_elapsed,
                estimated_remaining=estimated_remaining,
                last_update=time.time()
            )

            # Update completion counter
            if phase == TrainingPhase.COMPLETED:
                self._completed_models = len([p for p in self._model_progress.values()
                                           if p.phase == TrainingPhase.COMPLETED])

    def get_overall_progress(self) -> Dict[str, Any]:
        """
        Get aggregated progress across all models

        Returns:
            Dictionary with overall progress information
        """
        with self._lock:
            if not self._model_progress:
                return {
                    'overall_percent': 0.0,
                    'completed_models': 0,
                    'total_models': self.total_models,
                    'phase': 'not_started',
                    'time_elapsed': 0.0,
                    'estimated_remaining': None,
                    'models': {}
                }

            # Calculate overall progress
            total_progress = sum(p.progress_percent for p in self._model_progress.values())
            overall_percent = total_progress / self.total_models

            # Determine overall phase
            phases = [p.phase for p in self._model_progress.values()]
            if all(phase == TrainingPhase.COMPLETED for phase in phases):
                overall_phase = 'completed'
            elif any(phase == TrainingPhase.FAILED for phase in phases):
                overall_phase = 'failed'
            elif any(phase == TrainingPhase.TRAINING for phase in phases):
                overall_phase = 'training'
            else:
                overall_phase = 'initializing'

            # Estimate overall remaining time
            remaining_times = [p.estimated_remaining for p in self._model_progress.values()
                             if p.estimated_remaining is not None]
            overall_remaining = max(remaining_times) if remaining_times else None

            return {
                'overall_percent': overall_percent,
                'completed_models': self._completed_models,
                'total_models': self.total_models,
                'phase': overall_phase,
                'time_elapsed': time.time() - self._start_time,
                'estimated_remaining': overall_remaining,
                'models': {model_id: asdict(progress) for model_id, progress in self._model_progress.items()}
            }

    def get_model_progress(self, model_id: int) -> Optional[ModelProgress]:
        """Get progress for a specific model"""
        with self._lock:
            return self._model_progress.get(model_id)

    def format_progress_message(self, training_mode: str = "parallel") -> str:
        """
        Format human-readable progress message

        Args:
            training_mode: 'parallel' or 'series'

        Returns:
            Formatted progress string
        """
        progress = self.get_overall_progress()

        if training_mode == "series":
            # Series mode: show current model being trained
            if progress['models']:
                active_models = [m for m in progress['models'].values()
                               if m['phase'] == 'training']
                if active_models:
                    model = active_models[0]
                    return (f"Training model {model['model_id']}/{self.total_models}: "
                           f"Epoch {model['current_epoch']}/{model['total_epochs']} "
                           f"({model['progress_percent']:.1f}%)")

        else:
            # Parallel mode: show aggregate progress
            if progress['completed_models'] == self.total_models:
                return f"All {self.total_models} models completed successfully"

            active_count = len([m for m in progress['models'].values()
                              if m['phase'] == 'training'])

            if active_count > 0:
                avg_epoch = sum(m['current_epoch'] for m in progress['models'].values()
                              if m['phase'] == 'training') / active_count
                return (f"Training {active_count} models in parallel: "
                       f"Avg epoch {avg_epoch:.0f}, "
                       f"Overall {progress['overall_percent']:.1f}%")

        return f"Progress: {progress['overall_percent']:.1f}%"

    def generate_progress_report(self) -> Dict[str, Any]:
        """Generate comprehensive progress report"""
        progress = self.get_overall_progress()

        # Calculate statistics
        model_stats = []
        for model_data in progress['models'].values():
            model_stats.append({
                'model_id': model_data['model_id'],
                'phase': model_data['phase'],
                'progress_percent': model_data['progress_percent'],
                'current_epoch': model_data['current_epoch'],
                'loss_value': model_data['loss_value'],
                'time_elapsed': model_data['time_elapsed']
            })

        # Performance metrics
        if progress['models']:
            avg_progress = sum(m['progress_percent'] for m in progress['models'].values()) / len(progress['models'])
            fastest_model = max(progress['models'].values(), key=lambda x: x['progress_percent'])
            slowest_model = min(progress['models'].values(), key=lambda x: x['progress_percent'])
        else:
            avg_progress = 0.0
            fastest_model = None
            slowest_model = None

        return {
            'summary': {
                'overall_progress': progress['overall_percent'],
                'completed_models': progress['completed_models'],
                'total_models': self.total_models,
                'overall_phase': progress['phase'],
                'time_elapsed': progress['time_elapsed'],
                'estimated_remaining': progress['estimated_remaining']
            },
            'models': model_stats,
            'performance': {
                'average_progress': avg_progress,
                'fastest_model_id': fastest_model['model_id'] if fastest_model else None,
                'slowest_model_id': slowest_model['model_id'] if slowest_model else None,
                'progress_variance': self._calculate_progress_variance()
            },
            'timestamp': time.time()
        }

    def export_progress_log(self, file_path: str) -> None:
        """Export progress data to JSON file"""
        report = self.generate_progress_report()

        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Progress report exported to: {file_path}")

    def _calculate_progress_variance(self) -> float:
        """Calculate variance in progress across models"""
        with self._lock:
            if len(self._model_progress) < 2:
                return 0.0

            progresses = [p.progress_percent for p in self._model_progress.values()]
            mean_progress = sum(progresses) / len(progresses)
            variance = sum((p - mean_progress) ** 2 for p in progresses) / len(progresses)
            return variance

    def is_training_complete(self) -> bool:
        """Check if all models have completed training"""
        return self._completed_models == self.total_models

    def has_failed_models(self) -> bool:
        """Check if any models have failed"""
        with self._lock:
            return any(p.phase == TrainingPhase.FAILED for p in self._model_progress.values())


# Utility function for simple progress tracking
def create_progress_tracker(total_models: int = 3) -> ProgressAggregator:
    """Create a new progress tracker"""
    return ProgressAggregator(total_models)


if __name__ == "__main__":
    # Demo progress aggregation
    print("=== Agent Forge Progress Aggregator Demo ===")

    aggregator = ProgressAggregator(total_models=3)

    # Simulate parallel training progress
    print("\n1. Simulating Parallel Training Progress:")

    # Initial setup
    for model_id in [1, 2, 3]:
        aggregator.update_model_progress(
            model_id, TrainingPhase.INITIALIZING
        )

    # Training progress
    import time
    for epoch in range(1, 21):  # Simulate 20 epochs
        for model_id in [1, 2, 3]:
            # Different models progress at different rates
            progress_rate = 1.0 + (model_id * 0.2)  # Model 1: 1.2x, Model 2: 1.4x, Model 3: 1.6x
            model_epoch = min(int(epoch * progress_rate), 20)

            aggregator.update_model_progress(
                model_id=model_id,
                phase=TrainingPhase.TRAINING,
                current_epoch=model_epoch,
                total_epochs=20,
                loss_value=1.0 - (model_epoch / 20) * 0.8,  # Decreasing loss
                learning_rate=0.001 * (0.95 ** model_epoch)  # Decay
            )

        # Show progress every 5 epochs
        if epoch % 5 == 0:
            message = aggregator.format_progress_message("parallel")
            print(f"Epoch {epoch}: {message}")

        time.sleep(0.1)  # Brief pause for demo

    # Complete all models
    for model_id in [1, 2, 3]:
        aggregator.update_model_progress(
            model_id, TrainingPhase.COMPLETED, 20, 20
        )

    # Final report
    print("\n2. Final Progress Report:")
    report = aggregator.generate_progress_report()
    print(f"Overall: {report['summary']['overall_progress']:.1f}%")
    print(f"Time: {report['summary']['time_elapsed']:.1f}s")
    print(f"Fastest: Model {report['performance']['fastest_model_id']}")

    print("\n=== Demo Complete ===")

# Version & Run Log Footer
"""
Version & Run Log

| Version | Timestamp | Agent/Model | Change Summary | Artifacts | Status | Notes | Cost | Hash |
|--------:|-----------|-------------|----------------|-----------|--------|-------|------|------|
| 1.0.0   | 2025-01-25T15:25:03-05:00 | system@architect | Create ProgressAggregator for parallel training tracking | progress_aggregator.py | OK | Real-time progress aggregation with reporting | 0.00 | c3e1a7d |

Receipt:
- status: OK
- reason_if_blocked: --
- run_id: agent-forge-003
- inputs: ["progress-requirements", "parallel-training-spec"]
- tools_used: ["Write"]
- versions: {"model":"claude-sonnet-4","design":"progress-tracking-v1"}
"""