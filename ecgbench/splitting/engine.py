"""
Universal splitting engine.

Dispatches to StratifiedGroupKFold (patient-aware), StratifiedKFold,
or reads predefined splits based on the dataset config.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from ecgbench.splitting.base import SplitResult

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)


def split_dataset(
    df: pd.DataFrame,
    labels: pd.Series,
    config: DatasetConfig,
    n_folds: int = 10,
    random_state: int = 42,
) -> SplitResult:
    """Universal splitting pipeline.

    Decision logic:
    1. If config.has_predefined_splits: read fold assignments from config
    2. Else if config.patient_id_column is set: StratifiedGroupKFold
    3. Else: StratifiedKFold

    Args:
        df: Metadata DataFrame (from splitter.load_metadata)
        labels: Stratification labels (from splitter.get_stratification_labels)
        config: The dataset's DatasetConfig
        n_folds: Number of folds (default 10)
        random_state: Random seed for determinism

    Returns:
        SplitResult with fold assignments (1-indexed folds)
    """
    if config.has_predefined_splits and config.predefined_splits:
        return _split_predefined(df, labels, config)
    elif config.patient_id_column and config.patient_id_column in df.columns:
        return _split_grouped(df, labels, config, n_folds, random_state)
    else:
        return _split_simple(df, labels, config, n_folds, random_state)


def _split_predefined(
    df: pd.DataFrame,
    labels: pd.Series,
    config: DatasetConfig,
) -> SplitResult:
    """Use predefined fold assignments from the dataset."""
    ps = config.predefined_splits
    fold_col = ps.column
    if fold_col not in df.columns:
        raise ValueError(
            f"Predefined split column '{fold_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    folds: dict[int, pd.DataFrame] = {}
    for fold_num in sorted(df[fold_col].unique()):
        folds[int(fold_num)] = df[df[fold_col] == fold_num].copy().reset_index(drop=True)

    # Determine default split mapping
    train_folds = ps.fold_mapping.get("train", [])
    val_folds = ps.fold_mapping.get("val", [])
    test_folds = ps.fold_mapping.get("test", [])

    logger.info(
        "Predefined splits: %d folds, train=%s, val=%s, test=%s",
        len(folds), train_folds, val_folds, test_folds,
    )

    return SplitResult(
        folds=folds,
        default_train_folds=train_folds,
        default_val_folds=val_folds,
        default_test_folds=test_folds,
        stratify_column=labels.name or config.label_column,
        group_column=config.patient_id_column,
        split_metadata={"method": "predefined", "column": fold_col},
    )


def _split_grouped(
    df: pd.DataFrame,
    labels: pd.Series,
    config: DatasetConfig,
    n_folds: int,
    random_state: int,
) -> SplitResult:
    """Patient-aware stratified splitting via StratifiedGroupKFold."""
    groups = df[config.patient_id_column]

    splitter = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    )

    folds: dict[int, pd.DataFrame] = {}
    for fold_idx, (_, test_indices) in enumerate(splitter.split(df, labels, groups)):
        fold_num = fold_idx + 1  # 1-indexed
        folds[fold_num] = df.iloc[test_indices].copy().reset_index(drop=True)

    train_folds = list(range(1, n_folds - 1))
    val_folds = [n_folds - 1]
    test_folds = [n_folds]

    logger.info(
        "StratifiedGroupKFold: %d folds, grouped by '%s', stratified on '%s'",
        n_folds, config.patient_id_column, labels.name or config.label_column,
    )

    return SplitResult(
        folds=folds,
        default_train_folds=train_folds,
        default_val_folds=val_folds,
        default_test_folds=test_folds,
        stratify_column=labels.name or config.label_column,
        group_column=config.patient_id_column,
        split_metadata={
            "method": "StratifiedGroupKFold",
            "n_folds": n_folds,
            "random_state": random_state,
        },
    )


def _split_simple(
    df: pd.DataFrame,
    labels: pd.Series,
    config: DatasetConfig,
    n_folds: int,
    random_state: int,
) -> SplitResult:
    """Simple stratified splitting via StratifiedKFold (no patient grouping)."""
    splitter = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    )

    folds: dict[int, pd.DataFrame] = {}
    for fold_idx, (_, test_indices) in enumerate(splitter.split(df, labels)):
        fold_num = fold_idx + 1  # 1-indexed
        folds[fold_num] = df.iloc[test_indices].copy().reset_index(drop=True)

    train_folds = list(range(1, n_folds - 1))
    val_folds = [n_folds - 1]
    test_folds = [n_folds]

    logger.info(
        "StratifiedKFold: %d folds, stratified on '%s'",
        n_folds, labels.name or config.label_column,
    )

    return SplitResult(
        folds=folds,
        default_train_folds=train_folds,
        default_val_folds=val_folds,
        default_test_folds=test_folds,
        stratify_column=labels.name or config.label_column,
        group_column=None,
        split_metadata={
            "method": "StratifiedKFold",
            "n_folds": n_folds,
            "random_state": random_state,
        },
    )
