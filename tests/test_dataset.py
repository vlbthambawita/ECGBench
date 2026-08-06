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
        ds.window = None
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
        ds.window = None
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
            # Identical across all 88,253 records of all eight source cohorts.
            ("challenge2021", ["I", "II", "III", "aVR", "aVL", "aVF"],
             ["V1", "V2", "V3", "V4", "V5", "V6"]),
            # Identical across all 43,101 records of all six source cohorts —
            # unsurprising, since every one of them is a byte-for-byte copy of a
            # challenge2021 record.
            ("challenge2020", ["I", "II", "III", "aVR", "aVL", "aVF"],
             ["V1", "V2", "V3", "V4", "V5", "V6"]),
            # Uppercase limb leads, like ptbxl — identical in all 75 records.
            ("incartdb", ["I", "II", "III", "AVR", "AVL", "AVF"],
             ["V1", "V2", "V3", "V4", "V5", "V6"]),
            # aVF before aVL, like the demo it is the full release of.
            ("mimic_iv_ecg", ["I", "II", "III", "aVR", "aVF", "aVL"],
             ["V1", "V2", "V3", "V4", "V5", "V6"]),
            # Standard order and spelling, identical in all 363 headers.
            ("brugada_huca", ["I", "II", "III", "aVR", "aVL", "aVF"],
             ["V1", "V2", "V3", "V4", "V5", "V6"]),
            # Uppercase limb leads like ptbxl, identical in all 28 headers.
            ("norwegian_athlete_ecg", ["I", "II", "III", "AVR", "AVL", "AVF"],
             ["V1", "V2", "V3", "V4", "V5", "V6"]),
            # Lowercase 'a', identical in all 5,749 raw/ headers. The derived
            # medians/ headers of the SAME records spell them AVR/AVL/AVF and add
            # VCGMAG/X/Y/Z, but medians/ is not this config's signal.
            ("ecgcipa", ["I", "II", "III", "aVR", "aVL", "aVF"],
             ["V1", "V2", "V3", "V4", "V5", "V6"]),
            # Uppercase 'A' in all 4,211 raw/ headers — the opposite spelling to
            # ecgcipa, the sibling release from the same programme. Here medians/
            # agrees with raw/, which in ecgcipa it does not.
            ("ecgdmmld", ["I", "II", "III", "AVR", "AVL", "AVF"],
             ["V1", "V2", "V3", "V4", "V5", "V6"]),
            # Uppercase 'A' in all 5,232 raw/ headers, like its sibling ecgdmmld
            # and unlike ecgcipa. The medians/ headers agree.
            ("ecgrdvq", ["I", "II", "III", "AVR", "AVL", "AVF"],
             ["V1", "V2", "V3", "V4", "V5", "V6"]),
            # The release states no lead order at all — there are no headers, just
            # a (N, 1, 2500, 12) array. Inferred from the signals: Einthoven's
            # III = II - I and the Goldberger relations hold to a residual SD of
            # 0.13-0.19 against a 0.92 signal SD, where wrong pairings give 1.06+.
            ("echonext", ["I", "II", "III", "aVR", "aVL", "aVF"],
             ["V1", "V2", "V3", "V4", "V5", "V6"]),
            # Identical in all 6,877 headers — and necessarily the same as
            # challenge2020/2021, whose cpsc_2018 cohort is a byte-for-byte copy.
            ("cpsc_2018", ["I", "II", "III", "aVR", "aVL", "aVF"],
             ["V1", "V2", "V3", "V4", "V5", "V6"]),
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

    def test_staffiii_stores_nine_leads_with_the_precordials_first(self):
        """STAFF III inverts the limb-then-chest order every other dataset uses.

        The files store V1-V6 first and then I, II, III — so it cannot go in the
        parametrised table above, which assumes ``names[:6]`` are the limb leads.
        aVR, aVL and aVF are absent entirely: they are exact linear combinations
        of I and II and the depositors did not store them, so ``leads=["aVR"]``
        must raise rather than return some other lead. Verified identical across
        all 520 headers.
        """
        from ecgbench.config import load_config

        config = load_config("staffiii")
        assert config.leads == 9
        assert config.lead_names[:6] == ["V1", "V2", "V3", "V4", "V5", "V6"]
        assert config.lead_names[6:] == ["I", "II", "III"]
        assert len(config.lead_names) == config.leads
        # The augmented leads are not stored, so nothing may resolve to them.
        assert not {"aVR", "aVL", "aVF"} & {n.upper() for n in config.lead_names}
        # signal[0] is V1 here, which is lead I in every other wfdb dataset.
        assert config.lead_names[0] != "I"

    def test_mhd_declares_the_12_lead_layout_that_only_39_of_53_records_use(self):
        """The MHD dataset ships two channel layouts, and only 0-2 are shared.

        39 records are diagnostic 12-lead (Getemed CM 3000); 14 are 3-lead
        I/II/III (MRI-conditional MIPM Tesla M3). lead_names declares the 12-lead
        layout, so leads=["I"|"II"|"III"] resolves everywhere while anything past
        III raises on the 14 three-lead records rather than returning the wrong
        physical channel.
        """
        from ecgbench.config import load_config

        config = load_config("mhd_effect_ecg_mri")
        assert config.leads == 12
        assert config.lead_names[:3] == ["I", "II", "III"]
        assert config.lead_names[3:6] == ["aVR", "aVL", "aVF"]
        assert config.lead_names[6:] == ["V1", "V2", "V3", "V4", "V5", "V6"]

    def test_wctecgdb_declares_37_channels_raw_then_filtered_then_wct(self):
        """WCT is the one dataset that ships every channel twice.

        Channels 0-17 are the raw acquisition, 18-35 the same signals after DC
        removal and a 0.05-150 Hz band-pass, and 36 is the Wilson Central Terminal
        (filtered only — there is no WCT-Raw). Names, not indices, are what keeps
        the two families apart: index 18 is filtered lead I, index 0 is raw lead I,
        and the raw copy carries a DC offset the filtered one does not. aVR, aVL and
        aVF are absent from the release, so leads=["aVR"] must raise rather than
        return something plausible.
        """
        from ecgbench.config import load_config

        config = load_config("wctecgdb")
        assert config.leads == 37
        assert len(config.lead_names) == 37
        raw = config.lead_names[:18]
        filtered = config.lead_names[18:36]
        assert all(name.endswith("-Raw") for name in raw)
        assert not any(name.endswith("-Raw") for name in filtered)
        # Same 18 channels in the same order, minus the suffix.
        assert [name[:-4] for name in raw] == filtered
        assert filtered[:9] == ["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6"]
        # The three limb electrode potentials and the six true unipolar chest leads.
        assert filtered[9:] == ["LA", "RA", "LL",
                                "UV1", "UV2", "UV3", "UV4", "UV5", "UV6"]
        assert config.lead_names[36] == "WCT"
        for absent in ("aVR", "aVL", "aVF", "WCT-Raw"):
            assert absent not in config.lead_names

    def test_leipzig_declares_the_ecg_subset_of_a_variable_channel_layout(self):
        """Leipzig is the one dataset whose lead_names is a strict SUBSET of the
        channels a record holds, and that is deliberate.

        Records carry 14, 18, 19 or 20 channels in six layouts, and only indices
        0-11 — the surface ECG — are the same channel in every record. Index 12 is
        ABL12, RVA12 or ART depending on the record, so no single list can describe
        the intracardiac channels. Declaring one would make
        ECGDataset(leads=["RVA12"]) return a different physical channel per record.
        """
        from ecgbench.config import load_config

        config = load_config("leipzig_heart_center_ecg")
        assert config.leads == 12
        assert config.lead_names == ["I", "II", "III", "aVR", "aVL", "aVF",
                                     "V1", "V2", "V3", "V4", "V5", "V6"]
        # No intracardiac channel is declared, precisely because their order varies.
        for channel in ("ABL12", "RVA12", "CS12", "ART", "ABL_uni"):
            assert channel not in config.lead_names
        # Variable length rules out the truncation check, so it must stay unset.
        assert config.validation.expected_samples == {}


    def test_ptbdb_declares_fifteen_leads(self):
        """PTBDB is the one dataset that is not 12-lead: 12 + Frank vx/vy/vz."""
        from ecgbench.config import load_config

        config = load_config("ptbdb")
        assert config.leads == 15
        assert config.lead_names[-3:] == ["vx", "vy", "vz"]
        assert config.validation.expected_leads == 15


class TestResolveWindow:
    """window= is validated once at construction, not per __getitem__."""

    def test_valid_windows(self):
        from ecgbench.dataset import _resolve_window

        assert _resolve_window(None) is None
        assert _resolve_window((0, 2500)) == (0, 2500)
        assert _resolve_window((2500, 2500)) == (2500, 2500)
        assert _resolve_window([100, 200]) == (100, 200)  # list is fine too
        assert _resolve_window((2500, None)) == (2500, None)  # to the end

    def test_bare_int_is_rejected_with_the_fix(self):
        """window=5000 is the obvious mistake; the message must say what to write."""
        from ecgbench.dataset import _resolve_window

        with pytest.raises(TypeError, match=r"\(0, 5000\)"):
            _resolve_window(5000)

    @pytest.mark.parametrize("bad", [(0,), (0, 1, 2), (-1, 100), (0, 0), (0, -5)])
    def test_invalid_windows(self, bad):
        from ecgbench.dataset import _resolve_window

        with pytest.raises((ValueError, TypeError)):
            _resolve_window(bad)


class TestWindowedLoading:
    """window= is pushed into the reader, so it must return the right samples."""

    def _ds(self, root, config, **kwargs):
        from dataclasses import replace

        from ecgbench.dataset import ECGDataset

        return ECGDataset(
            replace(config, **kwargs.pop("config_overrides", {})),
            split="train",
            version="clean",
            data_path=root,
            metadata_source="local",
            **kwargs,
        )

    # --- csv format ---

    def test_csv_window_reads_the_requested_samples(self, tmp_csv_signal_dataset, sample_config):
        config_overrides = {"signal_format": "csv"}
        full = self._ds(tmp_csv_signal_dataset, sample_config, config_overrides=config_overrides)[
            0
        ]["signal"]
        assert tuple(full.shape) == (12, 5000)

        first = self._ds(
            tmp_csv_signal_dataset,
            sample_config,
            config_overrides=config_overrides,
            window=(0, 2500),
        )[0]["signal"]
        second = self._ds(
            tmp_csv_signal_dataset,
            sample_config,
            config_overrides=config_overrides,
            window=(2500, 2500),
        )[0]["signal"]

        assert tuple(first.shape) == (12, 2500)
        assert tuple(second.shape) == (12, 2500)
        # Lead j sample i holds j*100000 + i, so the offset is checkable exactly.
        assert first[0, 0].item() == 0
        assert second[0, 0].item() == 2500
        assert second[3, 0].item() == 3 * 100_000 + 2500
        assert torch.equal(first, full[:, :2500])
        assert torch.equal(second, full[:, 2500:5000])

    def test_csv_window_to_the_end(self, tmp_csv_signal_dataset, sample_config):
        ds = self._ds(
            tmp_csv_signal_dataset,
            sample_config,
            config_overrides={"signal_format": "csv"},
            window=(4000, None),
        )
        assert tuple(ds[0]["signal"].shape) == (12, 1000)

    def test_csv_window_past_the_end_raises(self, tmp_csv_signal_dataset, sample_config):
        """loadtxt silently returns a short array, so the loader must check."""
        from ecgbench.dataset import WindowOutOfRangeError

        ds = self._ds(
            tmp_csv_signal_dataset,
            sample_config,
            config_overrides={"signal_format": "csv"},
            window=(4000, 2000),
        )
        with pytest.raises(WindowOutOfRangeError, match="5000 samples"):
            ds[0]

    # --- wfdb format ---

    def test_wfdb_window_reads_the_requested_samples(self, tmp_wfdb_signal_dataset, sample_config):
        full = self._ds(tmp_wfdb_signal_dataset, sample_config)[0]["signal"]
        assert tuple(full.shape) == (12, 5000)

        second = self._ds(tmp_wfdb_signal_dataset, sample_config, window=(2500, 2500))[0]["signal"]

        assert tuple(second.shape) == (12, 2500)
        assert torch.allclose(second, full[:, 2500:5000], atol=1e-3)
        # Values are lead + sample/10000, so sample 2500 of lead 0 is 0.25 mV.
        assert second[0, 0].item() == pytest.approx(0.25, abs=1e-3)

    def test_wfdb_window_past_the_end_raises_a_useful_error(
        self, tmp_wfdb_signal_dataset, sample_config
    ):
        """wfdb's own message names neither the record nor its length."""
        from ecgbench.dataset import WindowOutOfRangeError

        ds = self._ds(tmp_wfdb_signal_dataset, sample_config, window=(0, 999_999))
        with pytest.raises(WindowOutOfRangeError) as excinfo:
            ds[0]
        message = str(excinfo.value)
        assert "rec_" in message  # names the record
        assert "5000 samples" in message  # and its real length

    # --- composition and picklability ---

    def test_window_composes_with_leads_and_units(self, tmp_wfdb_signal_dataset, sample_config):
        from dataclasses import replace

        config = replace(sample_config, lead_names=[f"L{i}" for i in range(12)])
        ds = self._ds(
            tmp_wfdb_signal_dataset,
            config,
            window=(1000, 500),
            leads=["L2", "L5"],
            units="uV",
        )
        signal = ds[0]["signal"]

        assert tuple(signal.shape) == (2, 500)
        # L2 sample 1000 is 2.1 mV -> 2100 uV.
        assert signal[0, 0].item() == pytest.approx(2100.0, abs=1.0)

    def test_window_is_picklable_where_a_lambda_transform_is_not(
        self, tmp_wfdb_signal_dataset, sample_config
    ):
        """This is why window= exists rather than just documenting transform=.

        DataLoader(num_workers>0) pickles the dataset under the 'spawn' start
        method, the default on macOS and Windows.
        """
        import pickle

        windowed = self._ds(tmp_wfdb_signal_dataset, sample_config, window=(0, 2500))
        assert pickle.loads(pickle.dumps(windowed)).window == (0, 2500)

        lambda_cropped = self._ds(
            tmp_wfdb_signal_dataset,
            sample_config,
            transform=lambda x: x[:, :2500],
        )
        with pytest.raises((AttributeError, pickle.PicklingError)):
            pickle.dumps(lambda_cropped)

    def test_no_window_still_loads_whole_records(self, tmp_wfdb_signal_dataset, sample_config):
        """The default path must be untouched by the window plumbing."""
        ds = self._ds(tmp_wfdb_signal_dataset, sample_config)
        assert ds.window is None
        assert tuple(ds[0]["signal"].shape) == (12, 5000)


class TestFoldSelection:
    """fold_numbers scopes to a split; split=None crosses the boundary."""

    def _ds(self, root, config, **kwargs):
        from ecgbench.dataset import ECGDataset

        return ECGDataset(
            config, version="clean", data_path=root, metadata_source="local", **kwargs
        )

    def test_subsample_a_single_fold(self, tmp_wfdb_signal_dataset, sample_config):
        whole = self._ds(tmp_wfdb_signal_dataset, sample_config, split="train")
        one = self._ds(tmp_wfdb_signal_dataset, sample_config, split="train", fold_numbers=[1])

        assert len(one) < len(whole)
        assert set(one.metadata_df["fold"]) == {1}

    def test_fold_outside_the_split_names_the_fix(self, tmp_wfdb_signal_dataset, sample_config):
        """Fold 4 is the val fold, so asking for it as train cannot work."""
        with pytest.raises(FileNotFoundError) as excinfo:
            self._ds(tmp_wfdb_signal_dataset, sample_config, split="train", fold_numbers=[4])
        assert "split=None" in str(excinfo.value)

    def test_split_none_selects_across_splits(self, tmp_wfdb_signal_dataset, sample_config):
        """Custom CV: hold out fold 1, train on the rest, ignoring default splits."""
        held_out = self._ds(tmp_wfdb_signal_dataset, sample_config, split=None, fold_numbers=[1])
        rest = self._ds(
            tmp_wfdb_signal_dataset,
            sample_config,
            split=None,
            fold_numbers=[2, 3, 4, 5],
        )

        assert set(held_out.metadata_df["fold"]) == {1}
        assert set(rest.metadata_df["fold"]) == {2, 3, 4, 5}
        # rest spans what would normally be train, val and test.
        assert set(rest.metadata_df["default_split"]) == {"train", "val", "test"}
        # The two partitions are disjoint and together cover everything.
        ids_a = set(held_out.metadata_df["record_id"])
        ids_b = set(rest.metadata_df["record_id"])
        assert not (ids_a & ids_b)
        assert len(ids_a | ids_b) == 5

    def test_split_none_reports_each_record_s_own_split(
        self, tmp_wfdb_signal_dataset, sample_config
    ):
        """A single split name would be a lie when rows span several."""
        ds = self._ds(tmp_wfdb_signal_dataset, sample_config, split=None, fold_numbers=[4, 5])
        assert ds.split is None
        assert {ds[i]["split"] for i in range(len(ds))} == {"val", "test"}

    def test_split_none_requires_fold_numbers(self, tmp_wfdb_signal_dataset, sample_config):
        with pytest.raises(ValueError, match="fold_numbers is required"):
            self._ds(tmp_wfdb_signal_dataset, sample_config, split=None)

    def test_unknown_fold_lists_what_exists(self, tmp_wfdb_signal_dataset, sample_config):
        with pytest.raises(ValueError, match=r"Available: \[1, 2, 3, 4, 5\]"):
            self._ds(tmp_wfdb_signal_dataset, sample_config, split=None, fold_numbers=[99])

    def test_invalid_split_name_still_rejected(self, tmp_wfdb_signal_dataset, sample_config):
        with pytest.raises(ValueError, match="split must be"):
            self._ds(tmp_wfdb_signal_dataset, sample_config, split="training")

    def test_folds_and_window_combine(self, tmp_wfdb_signal_dataset, sample_config):
        """The two features are independent: subsample folds, then window samples."""
        ds = self._ds(
            tmp_wfdb_signal_dataset,
            sample_config,
            split=None,
            fold_numbers=[2],
            window=(2500, 2500),
        )
        assert set(ds.metadata_df["fold"]) == {2}
        assert tuple(ds[0]["signal"].shape) == (12, 2500)


class TestNpySignalFormat:
    """EchoNext stores records as rows of one shared array per split."""

    def _array(self, tmp_path, n=4, samples=50, leads=12, channel_axis=True):
        """A stand-in for EchoNext_<split>_waveforms.npy, values encoding position."""
        import numpy as np

        # value = record*10000 + sample*100 + lead, so a shifted or transposed
        # read cannot pass by accident.
        rec = np.arange(n)[:, None, None] * 10000
        smp = np.arange(samples)[None, :, None] * 100
        led = np.arange(leads)[None, None, :]
        data = (rec + smp + led).astype(np.float64)
        if channel_axis:
            data = data[:, None]  # (n, 1, samples, leads), the shipped shape
        path = tmp_path / "EchoNext_test_waveforms.npy"
        np.save(path, data)
        return path

    def test_reference_names_a_row_not_just_a_file(self, tmp_path):
        from ecgbench.dataset import _parse_npy_ref

        path, row = _parse_npy_ref("/data/EchoNext_test_waveforms.npy:417")

        assert path == "/data/EchoNext_test_waveforms.npy"
        assert row == 417

    def test_reference_without_a_row_is_rejected(self):
        """A bare path is ambiguous — every record lives in the same file."""
        from ecgbench.dataset import _parse_npy_ref

        with pytest.raises(ValueError, match="must end in ':<row>'"):
            _parse_npy_ref("/data/EchoNext_test_waveforms.npy")

    def test_reads_the_named_row_and_transposes_to_leads_first(self, tmp_path):
        from ecgbench.dataset import _load_signal

        array = self._array(tmp_path)
        signal = _load_signal(f"{array}:2", "npy")

        assert signal.shape == (12, 50)          # (leads, samples)
        # Row 2, sample 0, lead 0 -> 20000; sample 3, lead 5 -> 20305.
        assert signal[0, 0] == 20000
        assert signal[5, 3] == 20305
        # A different row is genuinely different data, not a re-read of row 0.
        assert _load_signal(f"{array}:0", "npy")[0, 0] == 0

    def test_singleton_channel_axis_is_squeezed(self, tmp_path):
        """The shipped arrays are (N, 1, samples, leads); a plain (N, s, l) works too."""
        from ecgbench.dataset import _load_signal

        with_axis = self._array(tmp_path, channel_axis=True)
        plain_dir = tmp_path / "plain"
        plain_dir.mkdir()
        plain = self._array(plain_dir, channel_axis=False)

        four_d = _load_signal(f"{with_axis}:1", "npy")
        three_d = _load_signal(f"{plain}:1", "npy")

        assert four_d.shape == (12, 50)
        assert three_d.shape == (12, 50)
        # Same record either way — the channel axis carries no information.
        assert (four_d == three_d).all()

    def test_a_record_that_is_not_2d_after_squeezing_is_rejected(self, tmp_path):
        """Better than silently returning something of the wrong rank."""
        import numpy as np

        from ecgbench.dataset import _load_signal

        path = tmp_path / "EchoNext_odd_waveforms.npy"
        np.save(path, np.zeros((3, 2, 50, 12)))  # channel axis of 2, not 1

        with pytest.raises(ValueError, match="expected .*samples, leads"):
            _load_signal(f"{path}:0", "npy")

    def test_window_is_applied_before_materialising(self, tmp_path):
        from ecgbench.dataset import _load_signal

        array = self._array(tmp_path)
        signal = _load_signal(f"{array}:1", "npy", window=(10, 5))

        assert signal.shape == (12, 5)
        # First returned sample is sample 10, not sample 0.
        assert signal[0, 0] == 10000 + 10 * 100

    def test_window_past_the_end_names_the_record(self, tmp_path):
        from ecgbench.dataset import WindowOutOfRangeError, _load_signal

        array = self._array(tmp_path, samples=50)

        with pytest.raises(WindowOutOfRangeError, match="50 samples"):
            _load_signal(f"{array}:0", "npy", window=(40, 20))

    def test_record_length_is_reported_for_error_messages(self, tmp_path):
        from ecgbench.dataset import _record_length

        array = self._array(tmp_path, samples=50)

        assert _record_length(f"{array}:0", "npy") == 50


class TestNonPhysicalUnits:
    """A source whose publisher standardised the waveforms has no millivolts."""

    def test_zscore_source_refuses_unit_conversion(self):
        from ecgbench.dataset import UnitConversionError, _resolve_units

        with pytest.raises(UnitConversionError, match="cannot be converted"):
            _resolve_units("uV", source_units="zscore")

    def test_zscore_source_passes_samples_through_unscaled(self):
        """units='mV' is the default nobody chose, so it must not claim millivolts."""
        from ecgbench.dataset import _resolve_units

        assert _resolve_units("mV", source_units="zscore") == 1.0

    def test_millivolt_sources_are_unaffected(self):
        from ecgbench.dataset import _resolve_units

        assert _resolve_units("mV", source_units="mV") == 1.0
        assert _resolve_units("uV", source_units="mV") == 1000.0
