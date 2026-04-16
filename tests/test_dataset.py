"""Tests for the unified ECGDataset class."""

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
