#!/usr/bin/env python3
"""
Phase 3 → Phase 4 Text Flow Integration
========================================

Manages the flow of self-generated text from Phase 3 Quiet Star reasoning
to Phase 4 BitNet compression. The text generated during Phase 3's reasoning
internalization becomes the training data for gradual compression.

Key Features:
- Persistent storage of Phase 3 generated text
- Reasoning token preservation through compression
- Batch loading for compression training
- Quality tracking of reasoning preservation
- Self-supervised training data pipeline
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Generator
import torch
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
from collections import deque


@dataclass
class ReasoningTextSample:
    """Single sample of reasoning text from Phase 3"""
    text: str
    reasoning_tokens: List[str]  # <|startofthought|> ... <|endofthought|>
    generation_step: int
    quality_score: float
    thought_depth: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Phase3TextCorpus:
    """Complete corpus of Phase 3 generated text"""
    samples: List[ReasoningTextSample]
    total_tokens: int
    unique_reasoning_patterns: int
    generation_stats: Dict[str, Any]
    phase3_model_checkpoint: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class Phase3TextFlowManager:
    """
    Manages the flow of text from Phase 3 to Phase 4.
    Handles storage, retrieval, and quality tracking.
    """

    def __init__(
        self,
        storage_dir: str = "/c/Users/17175/Desktop/agent-forge/phases/phase3_phase4_bridge",
        buffer_size: int = 10000,
        min_quality_threshold: float = 0.7
    ):
        """
        Initialize text flow manager.

        Args:
            storage_dir: Directory to store Phase 3 text
            buffer_size: Maximum samples in memory buffer
            min_quality_threshold: Minimum quality score for samples
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.buffer_size = buffer_size
        self.min_quality_threshold = min_quality_threshold

        # Text storage
        self.text_buffer = deque(maxlen=buffer_size)
        self.reasoning_token_buffer = deque(maxlen=buffer_size)

        # Paths for persistent storage
        self.corpus_path = self.storage_dir / "phase3_corpus.pkl"
        self.text_db_path = self.storage_dir / "generated_texts.jsonl"
        self.stats_path = self.storage_dir / "generation_stats.json"

        # Statistics tracking
        self.stats = {
            'total_samples_saved': 0,
            'total_tokens_saved': 0,
            'unique_patterns': set(),
            'quality_distribution': [],
            'thought_depth_distribution': [],
            'compression_usage_count': 0,
        }

        # Logger
        self.logger = logging.getLogger(__name__)

        # Load existing corpus if available
        self.corpus = self._load_corpus()

    def save_phase3_generation(
        self,
        text: str,
        reasoning_tokens: List[str],
        generation_step: int,
        quality_score: float = 1.0,
        thought_depth: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save a single text generation from Phase 3.

        Args:
            text: Generated text content
            reasoning_tokens: Reasoning tokens used
            generation_step: Step number in Phase 3 training
            quality_score: Quality score of generation
            thought_depth: Depth of reasoning (1-4 typical)
            metadata: Additional metadata

        Returns:
            bool: Success status
        """
        # Quality filter
        if quality_score < self.min_quality_threshold:
            self.logger.debug(f"Skipping low quality sample: {quality_score}")
            return False

        # Create sample
        sample = ReasoningTextSample(
            text=text,
            reasoning_tokens=reasoning_tokens,
            generation_step=generation_step,
            quality_score=quality_score,
            thought_depth=thought_depth,
            metadata=metadata or {}
        )

        # Add to buffer
        self.text_buffer.append(sample)
        self.reasoning_token_buffer.append(reasoning_tokens)

        # Append to persistent storage
        self._append_to_jsonl(sample)

        # Update statistics
        self.stats['total_samples_saved'] += 1
        self.stats['total_tokens_saved'] += len(text.split())
        self.stats['quality_distribution'].append(quality_score)
        self.stats['thought_depth_distribution'].append(thought_depth)

        # Extract unique reasoning patterns
        pattern = self._extract_reasoning_pattern(reasoning_tokens)
        if pattern:
            self.stats['unique_patterns'].add(pattern)

        # Auto-save stats periodically
        if self.stats['total_samples_saved'] % 100 == 0:
            self._save_stats()

        return True

    def save_phase3_batch(
        self,
        texts: List[str],
        reasoning_tokens_list: List[List[str]],
        generation_step: int,
        quality_scores: Optional[List[float]] = None,
        thought_depths: Optional[List[int]] = None
    ) -> int:
        """
        Save a batch of Phase 3 generations.

        Args:
            texts: List of generated texts
            reasoning_tokens_list: List of reasoning token lists
            generation_step: Current training step
            quality_scores: Quality scores for each text
            thought_depths: Thought depths for each text

        Returns:
            int: Number of samples saved
        """
        if quality_scores is None:
            quality_scores = [1.0] * len(texts)
        if thought_depths is None:
            thought_depths = [1] * len(texts)

        saved_count = 0
        for text, tokens, score, depth in zip(
            texts, reasoning_tokens_list, quality_scores, thought_depths
        ):
            if self.save_phase3_generation(
                text, tokens, generation_step, score, depth
            ):
                saved_count += 1

        self.logger.info(f"Saved {saved_count}/{len(texts)} Phase 3 generations")
        return saved_count

    def get_compression_training_batch(
        self,
        batch_size: int,
        use_reasoning_weights: bool = True
    ) -> Dict[str, Any]:
        """
        Get a batch of Phase 3 text for compression training.

        Args:
            batch_size: Size of batch to retrieve
            use_reasoning_weights: Weight samples by reasoning quality

        Returns:
            Dict containing texts, tokens, and metadata
        """
        # Load from buffer or disk
        available_samples = list(self.text_buffer)

        if len(available_samples) < batch_size:
            # Load more from disk
            additional_samples = self._load_samples_from_disk(
                batch_size - len(available_samples)
            )
            available_samples.extend(additional_samples)

        if not available_samples:
            self.logger.warning("No Phase 3 samples available for compression training")
            return {
                'texts': [],
                'reasoning_tokens': [],
                'quality_scores': [],
                'metadata': {}
            }

        # Sample with quality weighting if requested
        if use_reasoning_weights and available_samples:
            weights = np.array([s.quality_score for s in available_samples])
            weights = weights / weights.sum()

            indices = np.random.choice(
                len(available_samples),
                min(batch_size, len(available_samples)),
                replace=True,
                p=weights
            )
            selected_samples = [available_samples[i] for i in indices]
        else:
            # Random sampling
            indices = np.random.choice(
                len(available_samples),
                min(batch_size, len(available_samples)),
                replace=len(available_samples) < batch_size
            )
            selected_samples = [available_samples[i] for i in indices]

        # Update usage statistics
        self.stats['compression_usage_count'] += len(selected_samples)

        # Format batch
        batch = {
            'texts': [s.text for s in selected_samples],
            'reasoning_tokens': [s.reasoning_tokens for s in selected_samples],
            'quality_scores': [s.quality_score for s in selected_samples],
            'thought_depths': [s.thought_depth for s in selected_samples],
            'metadata': {
                'batch_size': len(selected_samples),
                'average_quality': np.mean([s.quality_score for s in selected_samples]),
                'source': 'phase3_quiet_star',
                'timestamp': datetime.now().isoformat(),
            }
        }

        return batch

    def create_compression_dataset(
        self,
        num_samples: Optional[int] = None,
        min_quality: float = 0.8,
        save_path: Optional[str] = None
    ) -> Phase3TextCorpus:
        """
        Create a complete dataset for compression training.

        Args:
            num_samples: Number of samples (None for all)
            min_quality: Minimum quality threshold
            save_path: Path to save corpus

        Returns:
            Phase3TextCorpus: Complete corpus for training
        """
        # Load all samples
        all_samples = self._load_all_samples()

        # Filter by quality
        filtered_samples = [
            s for s in all_samples
            if s.quality_score >= min_quality
        ]

        # Limit number if specified
        if num_samples and len(filtered_samples) > num_samples:
            # Select best samples
            filtered_samples.sort(key=lambda x: x.quality_score, reverse=True)
            filtered_samples = filtered_samples[:num_samples]

        # Calculate statistics
        total_tokens = sum(len(s.text.split()) for s in filtered_samples)
        unique_patterns = len(self.stats['unique_patterns'])

        # Create corpus
        corpus = Phase3TextCorpus(
            samples=filtered_samples,
            total_tokens=total_tokens,
            unique_reasoning_patterns=unique_patterns,
            generation_stats={
                'total_samples': len(filtered_samples),
                'quality_range': (
                    min(s.quality_score for s in filtered_samples),
                    max(s.quality_score for s in filtered_samples)
                ),
                'average_quality': np.mean([s.quality_score for s in filtered_samples]),
                'thought_depth_distribution': dict(zip(*np.unique(
                    [s.thought_depth for s in filtered_samples],
                    return_counts=True
                ))),
            },
            phase3_model_checkpoint='phase3_quietstar_final.pt'
        )

        # Save corpus
        if save_path:
            corpus_path = Path(save_path)
        else:
            corpus_path = self.corpus_path

        with open(corpus_path, 'wb') as f:
            pickle.dump(corpus, f)

        self.logger.info(
            f"Created compression corpus with {len(filtered_samples)} samples, "
            f"{total_tokens} total tokens"
        )

        self.corpus = corpus
        return corpus

    def get_reasoning_preservation_metrics(self) -> Dict[str, float]:
        """
        Calculate metrics for reasoning preservation through compression.

        Returns:
            Dict[str, float]: Preservation metrics
        """
        if not self.text_buffer:
            return {
                'reasoning_token_ratio': 0.0,
                'thought_pattern_diversity': 0.0,
                'quality_retention': 0.0,
                'compression_efficiency': 0.0,
            }

        # Calculate reasoning token ratio
        total_reasoning_tokens = sum(
            len(s.reasoning_tokens) for s in self.text_buffer
        )
        total_text_tokens = sum(
            len(s.text.split()) for s in self.text_buffer
        )
        reasoning_ratio = total_reasoning_tokens / max(1, total_text_tokens)

        # Pattern diversity
        unique_patterns = len(self.stats['unique_patterns'])
        total_samples = len(self.text_buffer)
        pattern_diversity = unique_patterns / max(1, total_samples)

        # Quality retention (compare pre/post compression)
        avg_quality = np.mean([s.quality_score for s in self.text_buffer])

        # Compression efficiency
        compression_usage = self.stats['compression_usage_count']
        total_saved = self.stats['total_samples_saved']
        efficiency = compression_usage / max(1, total_saved)

        return {
            'reasoning_token_ratio': reasoning_ratio,
            'thought_pattern_diversity': pattern_diversity,
            'quality_retention': avg_quality,
            'compression_efficiency': efficiency,
            'total_samples': total_samples,
            'unique_patterns': unique_patterns,
        }

    def _extract_reasoning_pattern(self, reasoning_tokens: List[str]) -> Optional[str]:
        """
        Extract reasoning pattern signature from tokens.

        Args:
            reasoning_tokens: List of reasoning tokens

        Returns:
            Optional[str]: Pattern signature
        """
        if not reasoning_tokens:
            return None

        # Create pattern signature (simplified)
        # In reality, this would be more sophisticated
        sot_count = reasoning_tokens.count('<|startofthought|>')
        eot_count = reasoning_tokens.count('<|endofthought|>')

        if sot_count > 0 and eot_count > 0:
            return f"pattern_{sot_count}_{eot_count}_{len(reasoning_tokens)}"
        return None

    def _append_to_jsonl(self, sample: ReasoningTextSample) -> None:
        """Append sample to JSONL file"""
        with open(self.text_db_path, 'a', encoding='utf-8') as f:
            # Convert to dict, handling non-serializable fields
            sample_dict = asdict(sample)
            sample_dict['unique_patterns'] = list(self.stats['unique_patterns']) if isinstance(
                self.stats['unique_patterns'], set) else self.stats['unique_patterns']
            json.dump(sample_dict, f)
            f.write('\n')

    def _load_samples_from_disk(self, count: int) -> List[ReasoningTextSample]:
        """Load samples from disk storage"""
        samples = []

        if not self.text_db_path.exists():
            return samples

        with open(self.text_db_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= count:
                    break
                data = json.loads(line)
                sample = ReasoningTextSample(**data)
                samples.append(sample)

        return samples

    def _load_all_samples(self) -> List[ReasoningTextSample]:
        """Load all samples from disk"""
        samples = []

        if not self.text_db_path.exists():
            return list(self.text_buffer)

        with open(self.text_db_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                sample = ReasoningTextSample(**data)
                samples.append(sample)

        # Add buffer samples
        samples.extend(list(self.text_buffer))

        return samples

    def _save_stats(self) -> None:
        """Save statistics to disk"""
        # Convert set to list for JSON serialization
        stats_copy = self.stats.copy()
        stats_copy['unique_patterns'] = list(self.stats['unique_patterns'])

        with open(self.stats_path, 'w') as f:
            json.dump(stats_copy, f, indent=2)

    def _load_corpus(self) -> Optional[Phase3TextCorpus]:
        """Load existing corpus if available"""
        if self.corpus_path.exists():
            try:
                with open(self.corpus_path, 'rb') as f:
                    corpus = pickle.load(f)
                self.logger.info(f"Loaded existing corpus with {len(corpus.samples)} samples")
                return corpus
            except Exception as e:
                self.logger.error(f"Failed to load corpus: {e}")

        return None

    def generate_compression_training_iterator(
        self,
        batch_size: int,
        epochs: int = 1,
        shuffle: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate iterator for compression training.

        Args:
            batch_size: Batch size
            epochs: Number of epochs
            shuffle: Whether to shuffle data

        Yields:
            Dict[str, Any]: Training batch
        """
        all_samples = self._load_all_samples()

        if not all_samples:
            self.logger.warning("No samples available for training")
            return

        for epoch in range(epochs):
            if shuffle:
                np.random.shuffle(all_samples)

            for i in range(0, len(all_samples), batch_size):
                batch_samples = all_samples[i:i + batch_size]

                yield {
                    'texts': [s.text for s in batch_samples],
                    'reasoning_tokens': [s.reasoning_tokens for s in batch_samples],
                    'quality_scores': [s.quality_score for s in batch_samples],
                    'thought_depths': [s.thought_depth for s in batch_samples],
                    'metadata': {
                        'epoch': epoch,
                        'batch_idx': i // batch_size,
                        'batch_size': len(batch_samples),
                    }
                }

    def export_for_phase5(self, export_path: Optional[str] = None) -> str:
        """
        Export compressed corpus for Phase 5 training.

        Args:
            export_path: Path to export to

        Returns:
            str: Path where corpus was exported
        """
        if export_path:
            path = Path(export_path)
        else:
            path = self.storage_dir / "phase4_compressed_corpus.pkl"

        # Get preservation metrics
        metrics = self.get_reasoning_preservation_metrics()

        # Create export package
        export_data = {
            'corpus': self.corpus or self.create_compression_dataset(),
            'preservation_metrics': metrics,
            'compression_stats': self.stats,
            'phase': 'phase4_bitnet',
            'exported_at': datetime.now().isoformat(),
        }

        with open(path, 'wb') as f:
            pickle.dump(export_data, f)

        self.logger.info(f"Exported Phase 4 corpus to {path}")
        return str(path)


# Export main components
__all__ = [
    'Phase3TextFlowManager',
    'ReasoningTextSample',
    'Phase3TextCorpus',
]