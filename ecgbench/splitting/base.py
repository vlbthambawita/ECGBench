"""
Abstract base class for dataset splitters and the SplitResult dataclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    from ecgbench.config import DatasetConfig


@dataclass
class SplitResult:
    """Output of any splitting operation."""

    folds: dict[int, pd.DataFrame]  # fold_number (1-indexed) -> DataFrame
    default_train_folds: list[int]  # e.g., [1, 2, 3, 4, 5, 6, 7, 8]
    default_val_folds: list[int]  # e.g., [9]
    default_test_folds: list[int]  # e.g., [10]
    stratify_column: str  # what was used for stratification
    group_column: str | None  # what was used for patient grouping (None if N/A)
    split_metadata: dict = field(default_factory=dict)

    @property
    def train(self) -> pd.DataFrame:
        """Default training set: all default train folds concatenated."""
        return pd.concat(
            [self.folds[f] for f in self.default_train_folds], ignore_index=True
        )

    @property
    def val(self) -> pd.DataFrame:
        """Default validation set."""
        return pd.concat(
            [self.folds[f] for f in self.default_val_folds], ignore_index=True
        )

    @property
    def test(self) -> pd.DataFrame:
        """Default test set."""
        return pd.concat(
            [self.folds[f] for f in self.default_test_folds], ignore_index=True
        )

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def get_fold(self, fold_number: int) -> pd.DataFrame:
        """Get a single fold by number (1-indexed)."""
        if fold_number not in self.folds:
            raise ValueError(
                f"Fold {fold_number} not found. Available: {sorted(self.folds)}"
            )
        return self.folds[fold_number]

    def get_kfold_split(
        self, val_fold: int, test_fold: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get train/val/test for a custom k-fold rotation.

        All folds except val_fold and test_fold become train.

        Args:
            val_fold: Fold number to use as validation.
            test_fold: Fold number to use as test.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        train_folds = [f for f in self.folds if f != val_fold and f != test_fold]
        train = pd.concat([self.folds[f] for f in sorted(train_folds)], ignore_index=True)
        val = self.folds[val_fold]
        test = self.folds[test_fold]
        return train, val, test


class DatasetSplitter(ABC):
    """Abstract base for dataset-specific splitting logic.

    Subclass this when a dataset has unusual structure that can't be handled
    by config alone (e.g., PTB-XL's SCP code -> superclass mapping).
    For simple datasets, use GenericSplitter which reads everything from config.
    """

    @abstractmethod
    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        """Load the dataset's metadata CSV and normalise it.

        Must return a DataFrame containing at minimum:
          - config.record_id_column
          - config.signal_path_columns values
          - config.label_column (or derived stratification column)
          - config.patient_id_column (if applicable)
        """
        ...

    @abstractmethod
    def get_stratification_labels(
        self, df: pd.DataFrame, config: DatasetConfig
    ) -> pd.Series:
        """Return a Series of categorical labels for stratification.

        Must be aligned with df's index.
        """
        ...
