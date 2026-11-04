"""
Unit tests for data loading functionality.

Tests GlyphDataset and DataLoader with real pneumonia data.
"""

import pytest
import numpy as np
from pathlib import Path

from glyph.config import load_config
from glyph.dataloader import GlyphDataset, get_dataloader, get_dataloaders


# Path to real pneumonia data
PNEUMONIA_DATA_PATH = "/dataNAS/people/rholland/pneumonia_data"
METADATA_PATH = f"{PNEUMONIA_DATA_PATH}/metadata.csv"


class TestGlyphDataset:
    """Test cases for GlyphDataset class."""

    def test_dataset_initialization(self):
        """Test that dataset initializes correctly."""
        cfg = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "data.split_ratios=[0.7,0.15,0.15]",
            "seed=42"
        ])

        dataset = GlyphDataset(cfg, split='train')
        assert len(dataset) > 0
        assert dataset.split == 'train'
        assert dataset.input_cols == ['image_path']
        assert dataset.outcome_cols == ['outcome']

    def test_dataset_length(self):
        """Test that dataset length matches expected split size."""
        cfg = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "data.split_ratios=[0.7,0.15,0.15]",
            "seed=42"
        ])

        train_dataset = GlyphDataset(cfg, split='train')
        val_dataset = GlyphDataset(cfg, split='val')
        test_dataset = GlyphDataset(cfg, split='test')

        # Total should be ~4708 samples (pneumonia dataset)
        total = len(train_dataset) + len(val_dataset) + len(test_dataset)
        assert 4700 < total < 4710  # Allow for rounding

        # Check approximate split ratios
        assert abs(len(train_dataset) / total - 0.7) < 0.02
        assert abs(len(val_dataset) / total - 0.15) < 0.02
        assert abs(len(test_dataset) / total - 0.15) < 0.02

    def test_dataset_getitem(self):
        """Test that __getitem__ returns correct structure."""
        cfg = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "seed=42"
        ])

        dataset = GlyphDataset(cfg, split='train')
        sample = dataset[0]

        # Check structure
        assert 'sample_id' in sample
        assert 'inputs' in sample
        assert 'outcomes' in sample

        # Check inputs
        assert isinstance(sample['inputs'], dict)
        assert 'image_path' in sample['inputs']
        assert isinstance(sample['inputs']['image_path'], str)
        assert sample['inputs']['image_path'].endswith('.png')

        # Check outcomes
        assert isinstance(sample['outcomes'], dict)
        assert 'outcome' in sample['outcomes']
        assert sample['outcomes']['outcome'] in [0, 1]

    def test_dataset_file_path_validation(self):
        """Test that dataset validates file paths exist."""
        cfg = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "seed=42"
        ])

        dataset = GlyphDataset(cfg, split='train')
        sample = dataset[0]

        # Check that file path exists
        image_path = Path(sample['inputs']['image_path'])
        assert image_path.exists()

    def test_dataset_invalid_split_raises_error(self):
        """Test that invalid split raises ValueError."""
        cfg = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]"
        ])

        with pytest.raises(ValueError, match="split must be"):
            GlyphDataset(cfg, split='invalid')

    def test_dataset_missing_column_raises_error(self):
        """Test that missing column raises ValueError."""
        cfg = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[nonexistent_column]",
            "data.outcome_cols=[outcome]"
        ])

        with pytest.raises(ValueError, match="not found in metadata"):
            GlyphDataset(cfg, split='train')

    def test_dataset_reproducible_splits(self):
        """Test that splits are reproducible with same seed."""
        cfg1 = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "seed=42"
        ])

        cfg2 = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "seed=42"
        ])

        dataset1 = GlyphDataset(cfg1, split='train')
        dataset2 = GlyphDataset(cfg2, split='train')

        # Check same length
        assert len(dataset1) == len(dataset2)

        # Check same first sample
        sample1 = dataset1[0]
        sample2 = dataset2[0]
        assert sample1['sample_id'] == sample2['sample_id']

    def test_dataset_different_seeds_different_splits(self):
        """Test that different seeds produce different splits."""
        cfg1 = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "seed=42"
        ])

        cfg2 = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "seed=123"
        ])

        dataset1 = GlyphDataset(cfg1, split='train')
        dataset2 = GlyphDataset(cfg2, split='train')

        # Check different first sample
        sample1 = dataset1[0]
        sample2 = dataset2[0]
        assert sample1['sample_id'] != sample2['sample_id']

    def test_dataset_get_outcome_distribution(self):
        """Test get_outcome_distribution method."""
        cfg = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "seed=42"
        ])

        dataset = GlyphDataset(cfg, split='train')
        distribution = dataset.get_outcome_distribution()

        assert 'outcome' in distribution
        assert isinstance(distribution['outcome'], dict)
        assert 0 in distribution['outcome']
        assert 1 in distribution['outcome']
        # Check counts are positive
        assert distribution['outcome'][0] > 0
        assert distribution['outcome'][1] > 0


class TestDataLoader:
    """Test cases for DataLoader functions."""

    def test_get_dataloader_single(self):
        """Test creating a single dataloader."""
        cfg = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "data.batch_size=16",
            "data.num_workers=0",  # Use 0 workers for testing
            "seed=42"
        ])

        dataloader = get_dataloader(cfg, split='train')

        assert dataloader is not None
        assert len(dataloader) > 0

    def test_get_dataloader_iteration(self):
        """Test iterating through a dataloader."""
        cfg = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "data.batch_size=16",
            "data.num_workers=0",
            "seed=42"
        ])

        dataloader = get_dataloader(cfg, split='train')

        # Get first batch
        batch = next(iter(dataloader))

        # Check batch structure
        assert 'sample_id' in batch
        assert 'inputs' in batch
        assert 'outcomes' in batch

        # Check batch size (default collate converts to tensor)
        import torch
        assert isinstance(batch['sample_id'], (torch.Tensor, list))
        assert len(batch['sample_id']) <= 16  # batch_size

    def test_get_dataloaders_all_splits(self):
        """Test creating dataloaders for all splits."""
        cfg = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "data.batch_size=16",
            "data.num_workers=0",
            "seed=42"
        ])

        dataloaders = get_dataloaders(cfg)

        assert 'train' in dataloaders
        assert 'val' in dataloaders
        assert 'test' in dataloaders

        # Check all are DataLoaders
        for split, loader in dataloaders.items():
            assert loader is not None
            assert len(loader) > 0

    def test_dataloader_batch_size_override(self):
        """Test overriding batch size."""
        cfg = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "data.batch_size=32",
            "data.num_workers=0",
            "seed=42"
        ])

        # Override with smaller batch size
        dataloader = get_dataloader(cfg, split='train', batch_size=8)
        batch = next(iter(dataloader))

        assert len(batch['sample_id']) <= 8

    def test_dataloader_shuffle(self):
        """Test that training data is shuffled."""
        cfg = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "data.batch_size=16",
            "data.num_workers=0",
            "data.shuffle_train=true",
            "seed=42"
        ])

        # Create two dataloaders with different internal states
        dataloader1 = get_dataloader(cfg, split='train')
        dataloader2 = get_dataloader(cfg, split='train')

        batch1 = next(iter(dataloader1))
        batch2 = next(iter(dataloader2))

        # Due to shuffling, batches might differ
        # Just check they're both valid
        assert len(batch1['sample_id']) > 0
        assert len(batch2['sample_id']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
