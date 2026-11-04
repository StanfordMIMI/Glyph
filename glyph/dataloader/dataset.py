"""
Glyph Dataset class for domain-agnostic data loading.

Loads metadata CSV and provides access to samples with their inputs and outcomes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from torch.utils.data import Dataset


class GlyphDataset(Dataset):
    """
    Domain-agnostic dataset for Glyph.

    Loads metadata CSV with sample IDs, input file paths, and outcome labels.
    Supports train/val/test splits via random splitting based on config.

    Expected metadata CSV format:
        - Must contain columns specified in cfg.data.input_cols and cfg.data.outcome_cols
        - Should contain a primary key (e.g., 'sample_id')
        - Each input column should contain file paths or values

    Args:
        cfg: Hydra config with data settings
        split: 'train', 'val', or 'test'

    Example:
        >>> from glyph.config import load_config
        >>> cfg = load_config(overrides=['data.metadata=data.csv'])
        >>> dataset = GlyphDataset(cfg, split='train')
        >>> sample = dataset[0]
        >>> print(sample.keys())
        dict_keys(['sample_id', 'inputs', 'outcomes'])
    """

    def __init__(self, cfg, split: str = 'train', allow_missing_inputs=False):
        """
        Initialize dataset.

        Args:
            cfg: Hydra DictConfig with data configuration
            split: One of 'train', 'val', or 'test'
        """
        self.cfg = cfg
        self.allow_missing_inputs = allow_missing_inputs
        self.split = split

        # Validate split
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        # Load metadata
        self.metadata = self._load_metadata()

        # Split data
        if 'split' in self.metadata.columns:
            if not self.metadata['split'].isin(['train', 'val', 'test']).all():
                raise ValueError(
                    'Metadata split labels must be train, val, or test; '
                    'rename other labels explicitly before fitting so no studies are silently omitted'
                )
            self.metadata = self.metadata[self.metadata['split'] == split]
        else:
            self.metadata = self._split_data()

        # Validate file paths exist
        self._validate_paths()

        # Store column names
        self.input_cols = list(cfg.data.input_cols)
        self.outcome_cols = list(cfg.data.outcome_cols)

        # Detect outcome types (only for train split to avoid data leakage)
        if split == 'train':
            self.outcome_types = self._detect_outcome_types()
        else:
            self.outcome_types = None  # Will be set from train dataset

        # Determine sample ID column (first column if not specified)
        if 'sample_id' in self.metadata.columns:
            self.sample_id_col = 'sample_id'
        else:
            # Use first column as sample ID
            self.sample_id_col = self.metadata.columns[0]

    def _load_metadata(self) -> pd.DataFrame:
        """Load metadata CSV file."""
        metadata_path = Path(self.cfg.data.metadata)

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Load CSV (handles both with and without index column)
        df = pd.read_csv(metadata_path, dtype={"sample_id": str})

        # Drop unnamed index column if present
        if df.columns[0].startswith('Unnamed'):
            df = df.drop(columns=[df.columns[0]])

        return df

    def _split_data(self) -> pd.DataFrame:
        """Split data into train/val/test based on config."""
        # Get split ratios
        split_ratios = self.cfg.data.split_ratios

        if len(split_ratios) != 3:
            raise ValueError(f"split_ratios must have 3 values [train, val, test], got {len(split_ratios)}")

        if not np.isclose(sum(split_ratios), 1.0):
            raise ValueError(f"split_ratios must sum to 1.0, got {sum(split_ratios)}")

        # Set random seed for reproducibility
        np.random.seed(self.cfg.seed)

        # Shuffle indices
        n_samples = len(self.metadata)
        indices = np.random.permutation(n_samples)

        # Calculate split sizes
        train_size = int(n_samples * split_ratios[0])
        val_size = int(n_samples * split_ratios[1])

        # Split indices
        if self.split == 'train':
            split_indices = indices[:train_size]
        elif self.split == 'val':
            split_indices = indices[train_size:train_size + val_size]
        else:  # test
            split_indices = indices[train_size + val_size:]

        # Return subset
        return self.metadata.iloc[split_indices].reset_index(drop=True)

    def _validate_paths(self):
        """Validate that required columns exist and file paths are valid."""
        # Check input columns exist
        for col in self.cfg.data.input_cols:
            if col not in self.metadata.columns and not self.allow_missing_inputs:
                raise ValueError(f"Input column '{col}' not found in metadata. "
                               f"Available columns: {list(self.metadata.columns)}")

        # Check outcome columns exist
        for col in self.cfg.data.outcome_cols:
            if col not in self.metadata.columns:
                raise ValueError(f"Outcome column '{col}' not found in metadata. "
                               f"Available columns: {list(self.metadata.columns)}")

    def _detect_outcome_types(self) -> Dict[str, str]:
        """
        Detect the type of each outcome column.

        Returns:
            Dictionary mapping outcome column names to their types:
            - 'binary': Binary classification (only values 0 and 1)
            - 'regression': Continuous numerical values
            - 'categorical': Non-binary categorical (for future multiclass support)
        """
        outcome_types = {}

        for col in self.outcome_cols:
            # Get unique values (excluding NaN)
            values = self.metadata[col].dropna()
            unique_vals = values.unique()

            # Check if numeric
            if not pd.api.types.is_numeric_dtype(values):
                outcome_types[col] = 'categorical'
                continue

            # Exclude -1 sentinel (missing indicator) before type detection
            unique_vals = values[values != -1].unique()

            # Check for binary classification (only 0 and 1)
            if len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
                outcome_types[col] = 'binary'
            # Check for binary classification with floats (0.0 and 1.0)
            elif len(unique_vals) == 2 and set(unique_vals) == {0.0, 1.0}:
                outcome_types[col] = 'binary'
            # Otherwise treat as regression
            else:
                outcome_types[col] = 'regression'

        return outcome_types

    def __len__(self) -> int:
        """Return number of samples in this split."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with:
                - sample_id: Primary key for this sample
                - inputs: Dict mapping input column names to values/paths
                - outcomes: Dict mapping outcome column names to values
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        row = self.metadata.iloc[idx]

        # Extract sample ID
        sample_id = row[self.sample_id_col]

        # Extract inputs
        inputs = {}
        for col in self.input_cols:
            if col in row:
                inputs[col] = row[col]

        # Extract outcomes
        outcomes = {}
        for col in self.outcome_cols:
            outcomes[col] = row[col]

        return {
            'sample_id': sample_id,
            'inputs': inputs,
            'outcomes': outcomes
        }

    def get_outcome_distribution(self) -> Dict[str, Dict]:
        """
        Get distribution of outcome values for each outcome column.

        Returns:
            Dictionary mapping outcome column names to value counts
        """
        distribution = {}
        for col in self.outcome_cols:
            value_counts = self.metadata[col].value_counts().to_dict()
            distribution[col] = value_counts
        return distribution

    def __repr__(self) -> str:
        """String representation."""
        return (f"GlyphDataset(split='{self.split}', "
                f"n_samples={len(self)}, "
                f"input_cols={self.input_cols}, "
                f"outcome_cols={self.outcome_cols})")
