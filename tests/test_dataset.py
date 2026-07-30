"""Tests for the unified ECGDataset class."""

from unittest.mock import patch

import pandas as pd
import pytest

torch = pytest.importorskip("torch")


class TestEcgCollateFunction:
    def test_collate_empty(self):
        from ecgbench.dataset import ecg_collate_fn

        result = ecg_collate_fn([])
        assert result == {}

    def test_collate_mixed_types(self):
        from ecgbench.dataset import ecg_collate_fn

        batch = [
            {
                "signal": torch.randn(12, 5000),
                "record_id": "rec_001",
                "scp_codes": {"NORM": 100.0},
                "age": torch.tensor(55.0),
            },
            {
                "signal": torch.randn(12, 5000),
                "record_id": "rec_002",
                "scp_codes": {"MI": 80.0},
                "age": torch.tensor(62.0),
            },
        ]
        result = ecg_collate_fn(batch)

        # Tensors should be stacked
        assert result["signal"].shape == (2, 12, 5000)
        assert result["age"].shape == (2,)

        # Strings should be kept as lists
        assert result["record_id"] == ["rec_001", "rec_002"]

        # Dicts should be kept as lists
        assert isinstance(result["scp_codes"], list)
        assert len(result["scp_codes"]) == 2

    def test_collate_tensors_only(self):
        from ecgbench.dataset import ecg_collate_fn

        batch = [
            {"signal": torch.randn(12, 100), "value": torch.tensor(1.0)},
            {"signal": torch.randn(12, 100), "value": torch.tensor(2.0)},
        ]
        result = ecg_collate_fn(batch)
        assert result["signal"].shape == (2, 12, 100)


class TestECGDatasetLocal:
    def test_load_from_local_folds(self, tmp_splits_dir, sample_config):
        """Test loading dataset with local fold CSVs."""
        from ecgbench.dataset import ECGDataset

        # We can't actually load signals without WFDB files, but we can test
        # that the metadata loading works correctly
        ds = ECGDataset.__new__(ECGDataset)
        ds.config = sample_config
        ds.split = "train"
        ds.version = "clean"
        ds.data_path = tmp_splits_dir
        ds.metadata_source = "local"

        # Test the local loading path
        df = ds._load_from_local(fold_numbers=None)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_specific_folds(self, tmp_splits_dir, sample_config):
        """Test loading specific fold numbers."""
        from ecgbench.dataset import ECGDataset

        ds = ECGDataset.__new__(ECGDataset)
        ds.config = sample_config
        ds.split = "train"
        ds.version = "clean"
        ds.data_path = tmp_splits_dir
        ds.metadata_source = "local"

        df = ds._load_from_local(fold_numbers=[1])
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert all(df["fold"] == 1)


class TestECGDatasetLabels:
    """labels=True aligns the label frame to the split, positionally."""

    def _dataset(self, config, data_path, metadata_df):
        from ecgbench.dataset import ECGDataset

        ds = ECGDataset.__new__(ECGDataset)
        ds.config = config
        ds.split = "train"
        ds.version = "clean"
        ds.data_path = data_path
        ds.metadata_source = "local"
        ds.metadata_df = metadata_df
        return ds

    def _config(self, sample_config, **kwargs):
        from dataclasses import replace

        from ecgbench.config import LabelConfig

        return replace(
            sample_config,
            labels=LabelConfig(source_csv="source_labels.csv", join_column="rec", **kwargs),
        )

    def test_aligned_row_for_row_with_metadata(self, sample_config, tmp_labels_data):
        # Deliberately not in source order: alignment must follow metadata_df.
        metadata = pd.DataFrame({"record_id": ["rec_2", "rec_1"], "fold": [1, 1]})
        ds = self._dataset(self._config(sample_config), tmp_labels_data, metadata)

        labels = ds._load_labels()

        assert len(labels) == 2
        assert labels.loc[0, "diagnosis"] == "AFIB"  # rec_2
        assert labels.loc[1, "diagnosis"] == "NORM"  # rec_1
        assert labels.loc[0, "age"] == 78            # rec_2, not the source's first row

    def test_records_absent_from_the_label_source_warn_not_raise(
        self, sample_config, tmp_labels_data, caplog
    ):
        metadata = pd.DataFrame({"record_id": ["rec_0", "nope"], "fold": [1, 1]})
        ds = self._dataset(self._config(sample_config), tmp_labels_data, metadata)

        with caplog.at_level("WARNING"):
            labels = ds._load_labels()

        assert len(labels) == 2
        assert pd.isna(labels.loc[1, "diagnosis"])
        assert "have no label row" in caplog.text

    def test_no_overlap_at_all_raises(self, sample_config, tmp_labels_data):
        """An all-NaN join is a config error, not something to warn about."""
        metadata = pd.DataFrame({"record_id": ["x", "y"], "fold": [1, 1]})
        ds = self._dataset(self._config(sample_config), tmp_labels_data, metadata)

        with pytest.raises(ValueError, match="No record in split"):
            ds._load_labels()

    def test_int_vs_str_record_ids_still_align(self, sample_config, tmp_path):
        """Fold CSVs and source CSVs often disagree on int vs str for the same IDs."""
        pd.DataFrame({
            "rec": [1, 2, 3],                      # ints in the source
            "diagnosis": ["AFIB", "NORM", "AFIB"],
        }).to_csv(tmp_path / "source_labels.csv", index=False)
        metadata = pd.DataFrame({"record_id": ["2", "1"], "fold": [1, 1]})  # strs here

        ds = self._dataset(self._config(sample_config), tmp_path, metadata)
        labels = ds._load_labels()

        assert labels["diagnosis"].tolist() == ["NORM", "AFIB"]

    def test_getitem_nests_labels_under_one_key(self, sample_config, tmp_labels_data):
        """Labels must not be flattened into the sample dict."""
        import numpy as np
        import torch

        metadata = pd.DataFrame({"record_id": ["rec_0"], "filename": ["x"], "fold": [1]})
        ds = self._dataset(self._config(sample_config), tmp_labels_data, metadata)
        ds.signal_col = "filename"
        ds.transform = None
        ds.labels_df = ds._load_labels()

        with patch("ecgbench.dataset._load_signal", return_value=np.zeros((12, 100),
                                                                        dtype="float32")):
            sample = ds[0]

        assert isinstance(sample["labels"], dict)
        assert sample["labels"]["diagnosis"] == "AFIB"  # rec_0
        assert "diagnosis" not in sample  # nested, not flattened
        assert torch.is_tensor(sample["signal"])
