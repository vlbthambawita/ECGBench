"""Tests for fold CSV export."""

import pandas as pd
import pytest

from ecgbench.splitting.base import SplitResult
from ecgbench.splitting.export import export_splits
from ecgbench.validation.engine import ValidationResult


@pytest.fixture
def mock_split_result(mock_metadata_df):
    """Create a SplitResult from mock data."""
    n = len(mock_metadata_df)
    fold_size = n // 5
    folds = {}
    for i in range(5):
        start = i * fold_size
        end = start + fold_size if i < 4 else n
        folds[i + 1] = mock_metadata_df.iloc[start:end].copy().reset_index(drop=True)

    return SplitResult(
        folds=folds,
        default_train_folds=[1, 2, 3],
        default_val_folds=[4],
        default_test_folds=[5],
        stratify_column="label",
        group_column="patient_id",
    )


@pytest.fixture
def mock_validation_result(mock_metadata_df):
    """Create a ValidationResult where 2 records are invalid."""
    original_df = mock_metadata_df.copy()
    original_df["is_valid"] = True
    original_df["quality_issues"] = ""
    # Mark first 2 records as invalid
    original_df.loc[0, "is_valid"] = False
    original_df.loc[0, "quality_issues"] = "missing_lead_5"
    original_df.loc[1, "is_valid"] = False
    original_df.loc[1, "quality_issues"] = "nan_values:3_NaN_samples"

    clean_df = original_df[original_df["is_valid"]].drop(
        columns=["is_valid", "quality_issues"]
    ).reset_index(drop=True)

    return ValidationResult(
        original_df=original_df,
        clean_df=clean_df,
        record_validations=[],
        summary={"missing_leads": 1, "nan_values": 1},
        total_records=len(mock_metadata_df),
        valid_records=len(mock_metadata_df) - 2,
        excluded_records=2,
    )


def test_export_creates_structure(
    tmp_path, sample_config, mock_split_result, mock_validation_result
):
    """Test that export creates the correct directory structure."""
    export_splits(mock_split_result, mock_validation_result, tmp_path, sample_config)

    assert (tmp_path / "original" / "folds.csv").exists()
    assert (tmp_path / "clean" / "folds.csv").exists()
    assert (tmp_path / "original" / "train").exists()
    assert (tmp_path / "original" / "val").exists()
    assert (tmp_path / "original" / "test").exists()
    assert (tmp_path / "clean" / "train").exists()
    assert (tmp_path / "clean" / "val").exists()
    assert (tmp_path / "clean" / "test").exists()
    assert (tmp_path / "validation_report.json").exists()


def test_original_has_quality_columns(
    tmp_path, sample_config, mock_split_result, mock_validation_result
):
    """Original version should have is_valid and quality_issues columns."""
    export_splits(mock_split_result, mock_validation_result, tmp_path, sample_config)
    df = pd.read_csv(tmp_path / "original" / "folds.csv")
    assert "is_valid" in df.columns
    assert "quality_issues" in df.columns


def test_clean_no_quality_columns(
    tmp_path, sample_config, mock_split_result, mock_validation_result
):
    """Clean version should NOT have is_valid or quality_issues columns."""
    export_splits(mock_split_result, mock_validation_result, tmp_path, sample_config)
    df = pd.read_csv(tmp_path / "clean" / "folds.csv")
    assert "is_valid" not in df.columns
    assert "quality_issues" not in df.columns


def test_clean_fewer_records(
    tmp_path, sample_config, mock_split_result, mock_validation_result
):
    """Clean version should have fewer records than original."""
    export_splits(mock_split_result, mock_validation_result, tmp_path, sample_config)
    original = pd.read_csv(tmp_path / "original" / "folds.csv")
    clean = pd.read_csv(tmp_path / "clean" / "folds.csv")
    assert len(clean) <= len(original)


def test_fold_csvs_match_master(
    tmp_path, sample_config, mock_split_result, mock_validation_result
):
    """Per-fold CSVs should contain the same records as the master folds.csv."""
    export_splits(mock_split_result, mock_validation_result, tmp_path, sample_config)

    master = pd.read_csv(tmp_path / "clean" / "folds.csv")
    fold_records = []
    for split_dir in ("train", "val", "test"):
        for csv_file in sorted((tmp_path / "clean" / split_dir).glob("fold_*.csv")):
            fold_records.append(pd.read_csv(csv_file))

    combined = pd.concat(fold_records, ignore_index=True)
    assert set(combined["record_id"]) == set(master["record_id"])


def test_stats_returned(
    tmp_path, sample_config, mock_split_result, mock_validation_result
):
    """Export should return statistics dict."""
    stats = export_splits(mock_split_result, mock_validation_result, tmp_path, sample_config)
    assert "original" in stats
    assert "clean" in stats
    assert "excluded" in stats
    assert stats["original"]["total"] >= stats["clean"]["total"]
