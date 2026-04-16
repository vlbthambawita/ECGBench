"""
Fold CSV export for both original/ and clean/ versions.

Exported CSVs contain only the minimum columns needed to identify records
in the original dataset: record ID, patient ID (if available), signal file
paths, fold number, and default split assignment. Users join these back to
the original dataset CSV to get full metadata (age, sex, labels, etc.).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig
    from ecgbench.splitting.base import SplitResult
    from ecgbench.validation.engine import ValidationResult

logger = logging.getLogger(__name__)


def _minimal_columns(config: DatasetConfig, include_quality: bool = False) -> list[str]:
    """Return the minimal set of columns to include in exported CSVs.

    Always includes:
      - record_id_column
      - patient_id_column (if configured)
      - all signal_path_columns values
      - fold, default_split

    For original version (include_quality=True):
      - is_valid, quality_issues
    """
    cols = [config.record_id_column]
    if config.patient_id_column:
        cols.append(config.patient_id_column)
    cols.extend(config.signal_path_columns.values())
    cols.extend(["fold", "default_split"])
    if include_quality:
        cols.extend(["is_valid", "quality_issues"])
    return cols


def _select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Select only columns that exist in the DataFrame, preserving order."""
    present = [c for c in columns if c in df.columns]
    return df[present]


def _build_split_column(split_result: SplitResult) -> dict[int, str]:
    """Map fold numbers to their default split name."""
    mapping = {}
    for f in split_result.default_train_folds:
        mapping[f] = "train"
    for f in split_result.default_val_folds:
        mapping[f] = "val"
    for f in split_result.default_test_folds:
        mapping[f] = "test"
    return mapping


def _write_split_csvs(
    master_df: pd.DataFrame,
    output_dir: Path,
    split_result: SplitResult,
    config: DatasetConfig,
) -> None:
    """Write per-fold CSVs into train/val/test subdirectories."""
    fold_to_split = _build_split_column(split_result)

    for fold_num in sorted(split_result.folds.keys()):
        split_name = fold_to_split.get(fold_num, "train")
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        fold_df = master_df[master_df["fold"] == fold_num].copy()
        fold_df = fold_df.sort_values(config.record_id_column).reset_index(drop=True)

        fold_path = split_dir / f"fold_{fold_num}.csv"
        fold_df.to_csv(fold_path, index=False)
        logger.debug("Wrote %s (%d records)", fold_path, len(fold_df))


def export_splits(
    split_result: SplitResult,
    validation_result: ValidationResult,
    output_dir: Path,
    config: DatasetConfig,
) -> dict:
    """Export fold CSVs in both original/ and clean/ versions.

    Exported CSVs contain only identification columns (record ID, patient ID,
    signal file paths) plus fold/split assignment. Full metadata stays in the
    original dataset CSV — users join on record_id when they need it.

    Creates:
      output_dir/
        original/
          folds.csv
          train/fold_1.csv ... fold_N.csv
          val/fold_M.csv
          test/fold_K.csv
        clean/
          folds.csv
          train/fold_1.csv ... fold_N.csv
          val/fold_M.csv
          test/fold_K.csv
        validation_report.json

    Args:
        split_result: SplitResult from split_dataset()
        validation_result: ValidationResult from validate_dataset()
        output_dir: Root output directory
        config: DatasetConfig

    Returns:
        dict with statistics: counts per split, per version
    """
    output_dir = Path(output_dir)
    original_dir = output_dir / "original"
    clean_dir = output_dir / "clean"

    # Build the full master DataFrame by concatenating all folds with fold/split columns
    fold_to_split = _build_split_column(split_result)
    all_parts = []
    for fold_num, fold_df in sorted(split_result.folds.items()):
        part = fold_df.copy()
        part["fold"] = fold_num
        part["default_split"] = fold_to_split.get(fold_num, "train")
        all_parts.append(part)

    master_df = pd.concat(all_parts, ignore_index=True)

    # Merge validation info
    val_df = validation_result.original_df[
        [config.record_id_column, "is_valid", "quality_issues"]
    ].copy()
    val_df[config.record_id_column] = val_df[config.record_id_column].astype(
        master_df[config.record_id_column].dtype
    )

    master_df = master_df.merge(val_df, on=config.record_id_column, how="left")
    master_df["is_valid"] = master_df["is_valid"].fillna(True)
    master_df["quality_issues"] = master_df["quality_issues"].fillna("")

    # Sort by record_id for deterministic output
    master_df = master_df.sort_values(config.record_id_column).reset_index(drop=True)

    # --- Original version (minimal columns + quality flags) ---
    original_cols = _minimal_columns(config, include_quality=True)
    original_slim = _select_columns(master_df, original_cols)
    original_dir.mkdir(parents=True, exist_ok=True)
    original_slim.to_csv(original_dir / "folds.csv", index=False)
    _write_split_csvs(original_slim, original_dir, split_result, config)

    # --- Clean version (minimal columns, no quality flags) ---
    clean_rows = master_df[master_df["is_valid"]]
    clean_cols = _minimal_columns(config, include_quality=False)
    clean_slim = _select_columns(clean_rows, clean_cols).reset_index(drop=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_slim.to_csv(clean_dir / "folds.csv", index=False)
    _write_split_csvs(clean_slim, clean_dir, split_result, config)

    # --- Validation report ---
    from ecgbench.validation.report import save_report

    save_report(validation_result, config, output_dir / "validation_report.json")

    # --- Statistics ---
    stats = {
        "original": {
            "total": len(original_slim),
            "train": int((original_slim["default_split"] == "train").sum()),
            "val": int((original_slim["default_split"] == "val").sum()),
            "test": int((original_slim["default_split"] == "test").sum()),
        },
        "clean": {
            "total": len(clean_slim),
            "train": int((clean_slim["default_split"] == "train").sum()),
            "val": int((clean_slim["default_split"] == "val").sum()),
            "test": int((clean_slim["default_split"] == "test").sum()),
        },
        "excluded": validation_result.excluded_records,
    }

    logger.info(
        "Export complete: original=%d, clean=%d, excluded=%d",
        stats["original"]["total"],
        stats["clean"]["total"],
        stats["excluded"],
    )

    return stats
