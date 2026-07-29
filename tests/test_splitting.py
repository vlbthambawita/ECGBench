"""Tests for the splitting framework."""

from pathlib import Path

import pandas as pd
import pytest

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import SplitResult
from ecgbench.splitting.engine import split_dataset
from ecgbench.splitting.registry import get_splitter
from ecgbench.splitting.strategies.ecg_arrhythmia import build_metadata


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

    def test_ecg_arrhythmia_splitter(self):
        splitter = get_splitter("ecg_arrhythmia")
        assert type(splitter).__name__ == "ECGArrhythmiaSplitter"

    def test_unknown_falls_back_to_generic(self):
        splitter = get_splitter("some_unknown_dataset")
        assert type(splitter).__name__ == "GenericSplitter"


class TestECGArrhythmiaSplitter:
    """The ecg-arrhythmia splitter builds its metadata from WFDB headers."""

    def test_build_metadata_scans_headers(
        self, ecg_arrhythmia_config, tmp_ecg_arrhythmia_data
    ):
        df = build_metadata(tmp_ecg_arrhythmia_data, ecg_arrhythmia_config)

        assert len(df) == 15
        assert list(df["record_name"]) == sorted(df["record_name"])
        for col in ("record_name", "signal_path", "age", "sex", "dx",
                    "primary_dx", "primary_dx_acronym", "dx_acronyms"):
            assert col in df.columns

    def test_signal_paths_are_relative_and_point_at_mat_files(
        self, ecg_arrhythmia_config, tmp_ecg_arrhythmia_data
    ):
        df = build_metadata(tmp_ecg_arrhythmia_data, ecg_arrhythmia_config)
        first = df.loc[df["record_name"] == "JS00001", "signal_path"].item()

        assert first == "WFDBRecords/01/010/JS00001.mat"
        # Resolving against the data root is what validation and ECGDataset do.
        assert not Path(first).is_absolute()

    def test_header_fields_parsed(self, ecg_arrhythmia_config, tmp_ecg_arrhythmia_data):
        df = build_metadata(tmp_ecg_arrhythmia_data, ecg_arrhythmia_config).set_index(
            "record_name"
        )

        assert df.loc["JS00001", "n_leads"] == 12
        assert df.loc["JS00001", "sampling_rate"] == 500
        assert df.loc["JS00001", "n_samples"] == 5000
        assert str(df.loc["JS00001", "age"]) == "60"
        assert df.loc["JS00013", "sex"] == "Female"

    def test_dx_codes_mapped_to_acronyms(
        self, ecg_arrhythmia_config, tmp_ecg_arrhythmia_data
    ):
        df = build_metadata(tmp_ecg_arrhythmia_data, ecg_arrhythmia_config).set_index(
            "record_name"
        )

        assert df.loc["JS00001", "primary_dx"] == "426177001"
        assert df.loc["JS00001", "primary_dx_acronym"] == "SB"
        assert df.loc["JS00001", "dx_acronyms"] == "SB,TWC"
        # Codes absent from ConditionNames_SNOMED-CT.csv stay as raw codes.
        assert df.loc["JS00014", "primary_dx_acronym"] == "55827005"

    def test_malformed_record_line_is_tolerated(
        self, ecg_arrhythmia_config, tmp_ecg_arrhythmia_data
    ):
        """A record ECGBench cannot parse positionally is still listed.

        The validation engine is what flags it, via corrupt_header.
        """
        df = build_metadata(tmp_ecg_arrhythmia_data, ecg_arrhythmia_config).set_index(
            "record_name"
        )

        assert "JS00015" in df.index
        assert pd.isna(df.loc["JS00015", "n_samples"])
        assert df.loc["JS00015", "primary_dx_acronym"] == "SB"

    def test_load_metadata_caches_csv(
        self, ecg_arrhythmia_config, tmp_ecg_arrhythmia_data
    ):
        """The cache must land on disk — validate_dataset re-reads it from there."""
        splitter = get_splitter("ecg_arrhythmia")
        csv_path = tmp_ecg_arrhythmia_data / ecg_arrhythmia_config.metadata_csv
        assert not csv_path.exists()

        first = splitter.load_metadata(tmp_ecg_arrhythmia_data, ecg_arrhythmia_config)
        assert csv_path.exists()

        second = splitter.load_metadata(tmp_ecg_arrhythmia_data, ecg_arrhythmia_config)
        pd.testing.assert_frame_equal(
            first[["record_name", "signal_path", "primary_dx_acronym"]],
            second[["record_name", "signal_path", "primary_dx_acronym"]],
        )

    def test_stratification_pools_rare_classes(
        self, ecg_arrhythmia_config, tmp_ecg_arrhythmia_data
    ):
        splitter = get_splitter("ecg_arrhythmia")
        df = splitter.load_metadata(tmp_ecg_arrhythmia_data, ecg_arrhythmia_config)
        labels = splitter.get_stratification_labels(df, ecg_arrhythmia_config)

        assert labels.name == "primary_diagnosis"
        assert len(labels) == len(df)
        counts = labels.value_counts()
        # 13 SB records survive as their own class; AFIB and the unmapped code
        # have one record each and are pooled.
        assert counts["SB"] == 13
        assert counts["OTHER"] == 2
        assert "AFIB" not in counts

    def test_split_is_ungrouped(self, ecg_arrhythmia_config, tmp_ecg_arrhythmia_data):
        """No patient_id_column, so StratifiedKFold — not the grouped variant."""
        splitter = get_splitter("ecg_arrhythmia")
        df = splitter.load_metadata(tmp_ecg_arrhythmia_data, ecg_arrhythmia_config)
        labels = splitter.get_stratification_labels(df, ecg_arrhythmia_config)

        result = split_dataset(df, labels, ecg_arrhythmia_config, n_folds=3)

        assert result.group_column is None
        assert result.split_metadata["method"] == "StratifiedKFold"
        assert result.n_folds == 3
        assert sum(len(f) for f in result.folds.values()) == len(df)
