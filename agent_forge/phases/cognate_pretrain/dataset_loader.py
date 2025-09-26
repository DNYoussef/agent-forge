#!/usr/bin/env python3
"""
Real Dataset Loader for Cognate Pretraining

Loads actual datasets from disk (not synthetic data).
Uses environment variable for dataset path configuration.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CognateRealDataset(Dataset):
    """Real dataset loader for Cognate pretraining."""

    def __init__(
        self,
        dataset_dir: Optional[str] = None,
        tokenizer_name: str = "microsoft/phi-2",
        max_length: int = 4096,
        mix_ratio: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize real dataset loader.

        Args:
            dataset_dir: Directory containing downloaded datasets (defaults to env var or D:/cognate_datasets)
            tokenizer_name: Name of tokenizer to use
            max_length: Maximum sequence length
            mix_ratio: Mixing ratios for different dataset types
        """
        # Use environment variable or default to D: drive
        if dataset_dir is None:
            dataset_dir = os.environ.get('COGNATE_DATASET_DIR', 'D:/cognate_datasets')

        self.dataset_dir = Path(dataset_dir)
        self.max_length = max_length

        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Default mixing ratios (HRM + Titans combined curriculum)
        self.mix_ratio = mix_ratio or {
            'short_reasoning': 0.25,  # HRM-style reasoning tasks
            'long_context': 0.25,      # Titans-style long sequences
            'math': 0.20,              # Cognate math tasks
            'code': 0.15,              # Code generation
            'qa': 0.15,                # Question answering
        }

        # Load datasets
        self.samples = self._load_all_datasets()
        logger.info(f"Loaded {len(self.samples)} total samples")

    def _load_all_datasets(self) -> List[Dict]:
        """Load all available datasets from disk."""
        all_samples = []

        if not self.dataset_dir.exists():
            logger.warning(f"Dataset directory does not exist: {self.dataset_dir}")
            logger.warning("Using fallback synthetic data. Run download_datasets.py to get real data!")
            # Return empty list to trigger fallback in training
            return []

        # Try to load mixed training data first
        mixed_file = self.dataset_dir / "mixed_training_data.json"
        if mixed_file.exists():
            logger.info(f"Loading mixed training data from {mixed_file}")
            with open(mixed_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Otherwise load individual datasets
        for dataset_dir in self.dataset_dir.iterdir():
            if dataset_dir.is_dir():
                data_file = dataset_dir / "processed_data.json"
                if data_file.exists():
                    logger.info(f"Loading dataset: {dataset_dir.name}")
                    with open(data_file, 'r', encoding='utf-8') as f:
                        samples = json.load(f)
                        all_samples.extend(samples)

        return all_samples

    def __len__(self) -> int:
        """Return dataset length."""
        # If no real data, return a reasonable size for synthetic fallback
        return len(self.samples) if self.samples else 10000

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample."""
        if not self.samples:
            # Fallback to synthetic data if no real datasets available
            return self._get_synthetic_sample(idx)

        # Get real sample
        sample = self.samples[idx % len(self.samples)]
        text = sample.get('text', '')

        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': encoded['input_ids'].squeeze(0),  # For language modeling
            'metadata': sample.get('metadata', {}),
        }

    def _get_synthetic_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate synthetic sample as fallback."""
        # Use deterministic seed for reproducibility
        torch.manual_seed(idx)

        # Alternate between short and long sequences
        seq_len = 4096 if idx % 2 == 0 else 512

        return {
            'input_ids': torch.randint(0, self.tokenizer.vocab_size, (seq_len,), dtype=torch.long),
            'attention_mask': torch.ones(seq_len, dtype=torch.long),
            'labels': torch.randint(0, self.tokenizer.vocab_size, (seq_len,), dtype=torch.long),
            'metadata': {'synthetic': True},
        }


def create_real_dataloader(
    dataset_dir: Optional[str] = None,
    batch_size: int = 4,
    max_length: int = 4096,
    num_workers: int = 0,
    shuffle: bool = True,
    tokenizer_name: str = "microsoft/phi-2",
) -> Tuple[DataLoader, bool]:
    """
    Create DataLoader with real datasets.

    Returns:
        DataLoader and a boolean indicating if real data is being used.
    """
    dataset = CognateRealDataset(
        dataset_dir=dataset_dir,
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )

    # Check if we have real data
    has_real_data = bool(dataset.samples)

    if not has_real_data:
        logger.warning("=" * 80)
        logger.warning("NO REAL DATASETS FOUND - USING SYNTHETIC DATA")
        logger.warning(f"Expected dataset directory: {dataset.dataset_dir}")
        logger.warning("To use real data:")
        logger.warning("1. Run: python agent_forge/phases/cognate_pretrain/download_datasets.py")
        logger.warning("2. Or set COGNATE_DATASET_DIR environment variable to existing dataset location")
        logger.warning("=" * 80)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,  # For stable training
    )

    return dataloader, has_real_data


if __name__ == "__main__":
    # Test the dataset loader
    logger.info("Testing real dataset loader...")

    dataloader, has_real_data = create_real_dataloader(
        batch_size=2,
        max_length=512
    )

    if has_real_data:
        logger.info("✅ Using REAL datasets!")
    else:
        logger.info("⚠️ Using SYNTHETIC fallback data")

    # Test loading a batch
    for i, batch in enumerate(dataloader):
        if i >= 2:  # Just test first 2 batches
            break

        logger.info(f"Batch {i+1}:")
        logger.info(f"  Input shape: {batch['input_ids'].shape}")
        logger.info(f"  Attention mask shape: {batch['attention_mask'].shape}")
        logger.info(f"  Labels shape: {batch['labels'].shape}")

        if 'metadata' in batch and batch['metadata']:
            if isinstance(batch['metadata'], dict):
                synthetic = batch['metadata'].get('synthetic', [False])[0]
                logger.info(f"  Synthetic: {synthetic}")

    logger.info("✅ Dataset loader test complete!")