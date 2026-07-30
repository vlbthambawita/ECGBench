"""Tests for the unified ECGDataset class."""

from pathlib import Path
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
        # __new__ bypasses __init__, so read-time adapter state must be set here.
        ds._lead_index = None
        ds._unit_factor = 1.0
        ds.units = "mV"
        ds.lead_names = None
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


class TestLeadSelectionAndUnits:
    """leads= and units= are read-time adapters over the loaded tensor."""

    STORED = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    def _dataset(self, sample_config, **kwargs):
        from dataclasses import replace

        from ecgbench.dataset import ECGDataset

        ds = ECGDataset.__new__(ECGDataset)
        ds.config = replace(sample_config, lead_names=list(self.STORED))
        ds.split = "train"
        ds.version = "clean"
        ds.metadata_source = "local"
        ds.transform = None
        ds.labels_df = None
        ds.signal_col = "filename"
        ds.data_path = Path("/nowhere")
        ds.metadata_df = pd.DataFrame({"record_id": ["r0"], "filename": ["x"], "fold": [1]})
        ds._unit_factor = 1.0
        ds.units = "mV"
        ds.lead_names = tuple(self.STORED)
        ds._lead_index = None
        for key, value in kwargs.items():
            setattr(ds, key, value)
        return ds

    def _signal(self):
        """Row i is filled with the value i, so a reorder is unmistakable."""
        import numpy as np

        return np.arange(12, dtype="float32").repeat(8).reshape(12, 8)

    def _get(self, ds):
        with patch("ecgbench.dataset._load_signal", return_value=self._signal()):
            return ds[0]["signal"]

    def test_resolve_leads_selects_and_reorders(self, sample_config):
        from ecgbench.dataset import _resolve_leads

        idx, names = _resolve_leads(["V6", "I"], self.STORED, "test")
        assert idx == [11, 0]
        assert names == ["V6", "I"]

    def test_matching_is_case_insensitive(self, sample_config):
        """PTB-XL spells them AVR/AVL/AVF, everything else aVR/aVL/aVF."""
        from ecgbench.dataset import _resolve_leads

        idx, names = _resolve_leads(["avr"], ["I", "AVR"], "test")
        assert idx == [1]
        # The resolved name is what the dataset calls it, not what was asked for.
        assert names == ["AVR"]

    def test_unknown_lead_lists_what_is_available(self):
        from ecgbench.dataset import _resolve_leads

        with pytest.raises(ValueError, match=r"'V99' is not in 'test'.*Available"):
            _resolve_leads(["V99"], self.STORED, "test")

    def test_duplicate_lead_rejected(self):
        from ecgbench.dataset import _resolve_leads

        with pytest.raises(ValueError, match="more than once"):
            _resolve_leads(["II", "ii"], self.STORED, "test")

    def test_config_without_lead_names_is_a_clear_error(self):
        from ecgbench.dataset import _resolve_leads

        with pytest.raises(ValueError, match="does not declare lead_names"):
            _resolve_leads(["I"], None, "test")

    def test_getitem_returns_selected_rows_in_order(self, sample_config):
        ds = self._dataset(sample_config, _lead_index=[11, 0], lead_names=("V6", "I"))
        signal = self._get(ds)

        assert tuple(signal.shape) == (2, 8)
        assert signal[0][0] == 11.0  # V6
        assert signal[1][0] == 0.0  # I

    def test_too_few_leads_in_the_record_raises(self, sample_config):
        import numpy as np

        ds = self._dataset(sample_config, _lead_index=[11], lead_names=("V6",))
        with patch("ecgbench.dataset._load_signal",
                   return_value=np.zeros((2, 8), dtype="float32")):
            with pytest.raises(ValueError, match="too few for the requested"):
                ds[0]

    def test_units_scale_after_selection(self, sample_config):
        ds = self._dataset(sample_config, _lead_index=[1], lead_names=("II",),
                           _unit_factor=1000.0, units="uV")
        assert self._get(ds)[0][0] == 1000.0  # row 1 == 1.0 mV -> 1000 uV

    def test_resolve_units(self):
        from ecgbench.dataset import _resolve_units

        assert _resolve_units("mV") == 1.0
        assert _resolve_units("uV") == 1000.0
        assert _resolve_units("µV") == 1000.0
        assert _resolve_units("MV") == 1.0
        with pytest.raises(ValueError, match="units must be one of"):
            _resolve_units("volts")


class TestShippedLeadNames:
    """Every implemented dataset must declare its true lead order."""

    @pytest.mark.parametrize(
        ("slug", "limb", "chest"),
        [
            ("ptbxl", ["I", "II", "III", "AVR", "AVL", "AVF"],
             ["V1", "V2", "V3", "V4", "V5", "V6"]),
            ("ecg_arrhythmia", ["I", "II", "III", "aVR", "aVL", "aVF"],
             ["V1", "V2", "V3", "V4", "V5", "V6"]),
            ("chapman_shaoxing", ["I", "II", "III", "aVR", "aVL", "aVF"],
             ["V1", "V2", "V3", "V4", "V5", "V6"]),
            # aVF and aVL transposed — the reason leads= takes names, not indices.
            ("mimic_iv_ecg_demo", ["I", "II", "III", "aVR", "aVF", "aVL"],
             ["V1", "V2", "V3", "V4", "V5", "V6"]),
            # Lowercase in the headers, and PTBDB adds the three Frank leads.
            ("ludb", ["i", "ii", "iii", "avr", "avl", "avf"],
             ["v1", "v2", "v3", "v4", "v5", "v6"]),
            ("ptbdb", ["i", "ii", "iii", "avr", "avl", "avf"],
             ["v1", "v2", "v3", "v4", "v5", "v6", "vx", "vy", "vz"]),
        ],
    )
    def test_lead_names_match_the_files(self, slug, limb, chest):
        from ecgbench.config import load_config

        config = load_config(slug)
        names = config.lead_names
        assert names is not None, f"{slug} declares no lead_names"
        assert names[:6] == limb
        assert names[6:] == chest
        # leads is a count of what lead_names lists, not an assumption of 12.
        assert len(names) == config.leads

    def test_ptbdb_declares_fifteen_leads(self):
        """PTBDB is the one dataset that is not 12-lead: 12 + Frank vx/vy/vz."""
        from ecgbench.config import load_config

        config = load_config("ptbdb")
        assert config.leads == 15
        assert config.lead_names[-3:] == ["vx", "vy", "vz"]
        assert config.validation.expected_leads == 15
