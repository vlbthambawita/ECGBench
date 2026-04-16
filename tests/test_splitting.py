"""Tests for the splitting framework."""

import pandas as pd
import pytest

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import SplitResult
from ecgbench.splitting.engine import split_dataset
from ecgbench.splitting.registry import get_splitter


class TestSplitDatasetPredefined:
    """Test splitting with predefined fold assignments."""

    def test_predefined_splits(self, ptbxl_config, mock_metadata_with_folds):
        labels = mock_metadata_with_folds["label"]
        result = split_dataset(mock_metadata_with_folds, labels, ptbxl_config)

        assert isinstance(result, SplitResult)
        assert result.n_folds == 10
        assert result.default_train_folds == [1, 2, 3, 4, 5, 6, 7, 8]
        assert result.default_val_folds == [9]
        assert result.default_test_folds == [10]

    def test_predefined_no_overlap(self, ptbxl_config, mock_metadata_with_folds):
        labels = mock_metadata_with_folds["label"]
        result = split_dataset(mock_metadata_with_folds, labels, ptbxl_config)

        all_ids = set()
        for fold_num, fold_df in result.folds.items():
            fold_ids = set(fold_df["record_id"])
            assert all_ids.isdisjoint(fold_ids), f"Overlap found in fold {fold_num}"
            all_ids.update(fold_ids)

        assert len(all_ids) == len(mock_metadata_with_folds)

    def test_predefined_covers_all(self, ptbxl_config, mock_metadata_with_folds):
        labels = mock_metadata_with_folds["label"]
        result = split_dataset(mock_metadata_with_folds, labels, ptbxl_config)

        total = sum(len(df) for df in result.folds.values())
        assert total == len(mock_metadata_with_folds)


class TestSplitDatasetGrouped:
    """Test splitting with patient-aware grouping (StratifiedGroupKFold)."""

    def test_no_patient_leakage(self, sample_config, mock_metadata_df):
        """ALL records from one patient MUST be in the same fold."""
        labels = mock_metadata_df["label"]
        result = split_dataset(mock_metadata_df, labels, sample_config, n_folds=5)

        patient_fold: dict[str, int] = {}
        for fold_num, fold_df in result.folds.items():
            for pid in fold_df["patient_id"]:
                if pid in patient_fold:
                    assert patient_fold[pid] == fold_num, (
                        f"Patient {pid} appears in folds {patient_fold[pid]} and {fold_num}"
                    )
                else:
                    patient_fold[pid] = fold_num

    def test_deterministic(self, sample_config, mock_metadata_df):
        """Same seed should produce identical results."""
        labels = mock_metadata_df["label"]
        r1 = split_dataset(mock_metadata_df, labels, sample_config, random_state=42)
        r2 = split_dataset(mock_metadata_df, labels, sample_config, random_state=42)

        for fold_num in r1.folds:
            pd.testing.assert_frame_equal(
                r1.folds[fold_num].reset_index(drop=True),
                r2.folds[fold_num].reset_index(drop=True),
            )

    def test_different_seed_different_result(self, sample_config, mock_metadata_df):
        labels = mock_metadata_df["label"]
        r1 = split_dataset(mock_metadata_df, labels, sample_config, random_state=42)
        r2 = split_dataset(mock_metadata_df, labels, sample_config, random_state=99)

        # At least one fold should differ
        any_diff = False
        for fold_num in r1.folds:
            if not r1.folds[fold_num]["record_id"].equals(r2.folds[fold_num]["record_id"]):
                any_diff = True
                break
        assert any_diff


class TestSplitDatasetSimple:
    """Test splitting without patient grouping (StratifiedKFold)."""

    def test_simple_split(self, mock_metadata_df):
        config = DatasetConfig(
            name="test", slug="test", version="1.0", url="http://x",
            metadata_csv="x.csv", record_id_column="record_id",
            label_column="label",
            patient_id_column=None,  # No patient grouping
        )
        labels = mock_metadata_df["label"]
        result = split_dataset(mock_metadata_df, labels, config, n_folds=5)

        assert result.n_folds == 5
        assert result.group_column is None
        total = sum(len(df) for df in result.folds.values())
        assert total == len(mock_metadata_df)


class TestSplitResult:
    def test_train_val_test_properties(self, ptbxl_config, mock_metadata_with_folds):
        labels = mock_metadata_with_folds["label"]
        result = split_dataset(mock_metadata_with_folds, labels, ptbxl_config)

        assert len(result.train) > 0
        assert len(result.val) > 0
        assert len(result.test) > 0
        assert len(result.train) + len(result.val) + len(result.test) == len(
            mock_metadata_with_folds
        )

    def test_get_kfold_split(self, ptbxl_config, mock_metadata_with_folds):
        labels = mock_metadata_with_folds["label"]
        result = split_dataset(mock_metadata_with_folds, labels, ptbxl_config)

        train, val, test = result.get_kfold_split(val_fold=2, test_fold=5)
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == len(mock_metadata_with_folds)

    def test_get_fold(self, ptbxl_config, mock_metadata_with_folds):
        labels = mock_metadata_with_folds["label"]
        result = split_dataset(mock_metadata_with_folds, labels, ptbxl_config)

        fold_1 = result.get_fold(1)
        assert len(fold_1) > 0

    def test_get_fold_invalid(self, ptbxl_config, mock_metadata_with_folds):
        labels = mock_metadata_with_folds["label"]
        result = split_dataset(mock_metadata_with_folds, labels, ptbxl_config)

        with pytest.raises(ValueError, match="Fold 99 not found"):
            result.get_fold(99)


class TestSplitterRegistry:
    def test_ptbxl_splitter(self):
        splitter = get_splitter("ptbxl")
        assert type(splitter).__name__ == "PTBXLSplitter"

    def test_chapman_splitter(self):
        splitter = get_splitter("chapman_shaoxing")
        assert type(splitter).__name__ == "ChapmanSplitter"

    def test_unknown_falls_back_to_generic(self):
        splitter = get_splitter("some_unknown_dataset")
        assert type(splitter).__name__ == "GenericSplitter"
