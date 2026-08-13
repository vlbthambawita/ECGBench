"""Tests for the unified ECGDataset class."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
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
        # Read-time lead-layout state. This helper bypasses __init__, so every new
        # attribute __getitem__ touches has to be set here or it raises
        # AttributeError — see TestAlternateLeadLayouts for what these do.
        ds._requested_leads = None
        ds._declared_n_leads = len(self.STORED)
        ds._alt_lead_index = {}
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
            # The HDF5 arrays carry no lead names, so this order was derived from
            # the signals: III = II - I, aVR = -(I+II)/2, aVL = I - II/2 and
            # aVF = II - I/2 all hold to under 2% relative RMS error.
            ("sph", ["I", "II", "III", "aVR", "aVL", "aVF"],
             ["V1", "V2", "V3", "V4", "V5", "V6"]),
            # Standard, and checked rather than assumed because its own sibling
            # release is not — see test_the_two_code_releases_disagree below.
            # Derived from the arrays over 1,200 records spanning four parts.
            ("code15", ["I", "II", "III", "aVR", "aVL", "aVF"],
             ["V1", "V2", "V3", "V4", "V5", "V6"]),
            # aVL, aVF, aVR — the release's own documented order, confirmed
            # against all 827 arrays. Same cohort as code15, different order.
            ("code_test", ["I", "II", "III", "aVL", "aVF", "aVR"],
             ["V1", "V2", "V3", "V4", "V5", "V6"]),
            # Standard, and checked rather than assumed for the same reason as
            # code15 — the third TNMG release, and the other two disagree with
            # each other. Derived from the arrays over 400 records: channels 2-5
            # match II-I, -(I+II)/2, I-II/2 and II-I/2 to under 0.3% relative
            # error, where every other assignment is off by more than 33%.
            ("sami_trop", ["I", "II", "III", "aVR", "aVL", "aVF"],
             ["V1", "V2", "V3", "V4", "V5", "V6"]),
            # Uppercase limb leads like ptbxl, identical in all 12,334 twelve-lead
            # headers. The other 1,856 records store a 9-lead subset — see
            # TestAlternateLeadLayouts, which is where that is pinned.
            ("zzu_pecg", ["I", "II", "III", "AVR", "AVL", "AVF"],
             ["V1", "V2", "V3", "V4", "V5", "V6"]),
            # Simulated, so there is no header to read: the order is the one the
            # release README states, corroborated by the per-record parameter
            # files, which give electrode positions for RA/LA/RL/LL and V1-V6 and
            # nothing else — the augmented leads are computed, not placed.
            ("medalcare_xl", ["I", "II", "III", "aVR", "aVL", "aVF"],
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

    def test_the_two_code_releases_disagree_about_the_augmented_leads(self):
        """CODE-15% and CODE-test permute aVR/aVL/aVF differently.

        Same cohort, same telehealth network, same 400 Hz, same
        ``(N, 4096, 12)`` HDF5 layout — and channels 3, 4 and 5 mean different
        physical leads in each. CODE-15% is standard (aVR, aVL, aVF) and
        CODE-test is not (aVL, aVF, aVR), both verified from the arrays. Anyone
        stacking the two into one training set by index crosses all three
        augmented leads on one of them, so this asserts the disagreement rather
        than letting a later "consistency" edit quietly erase it.
        """
        from ecgbench.config import load_config

        code15 = load_config("code15").lead_names
        code_test = load_config("code_test").lead_names

        # Identical where it matters least...
        assert code15[:3] == code_test[:3] == ["I", "II", "III"]
        assert code15[6:] == code_test[6:]
        # ...and permuted where it matters most.
        assert code15[3:6] == ["aVR", "aVL", "aVF"]
        assert code_test[3:6] == ["aVL", "aVF", "aVR"]
        assert set(code15) == set(code_test)
        # The index a reader would reach for by habit is wrong in one of them.
        assert code15[3] != code_test[3]

    def test_ningbo_iva_stores_its_leads_in_alphabetical_order(self):
        """Ningbo IVA sorts the CSV columns alphabetically, so signal[0] is aVF.

        The header row reads ``aVF,aVL,aVR,I,II,III,V1..V6`` — the augmented leads
        first and the limb leads third, fourth and fifth. It cannot go in the
        parametrised table above, which assumes ``names[:6]`` are I/II/III then
        aVR/aVL/aVF. Every index a reader would reach for by habit is wrong here:
        ``signal[0]`` is aVF rather than lead I, and ``signal[4]`` is lead II
        rather than aVL.
        """
        from ecgbench.config import load_config

        config = load_config("ningbo_iva")
        assert config.leads == 12
        assert config.lead_names == ["aVF", "aVL", "aVR", "I", "II", "III",
                                     "V1", "V2", "V3", "V4", "V5", "V6"]
        assert len(config.lead_names) == config.leads
        # Alphabetical, case-insensitively, for the nine non-precordial-ordered
        # names — which is what makes the order predictable but non-standard.
        assert config.lead_names[:6] == sorted(
            config.lead_names[:6], key=lambda n: n.lower()
        )
        # The two indices most likely to be assumed.
        assert config.lead_names[0] != "I"
        assert config.lead_names[4] != "aVL"
        # All twelve standard leads are present, just permuted — so leads= can
        # recover any of them by name.
        assert {n.upper() for n in config.lead_names} == {
            "I", "II", "III", "AVR", "AVL", "AVF",
            "V1", "V2", "V3", "V4", "V5", "V6",
        }

    def test_mitdb_declares_a_predominant_layout_because_it_has_no_single_one(self):
        """MIT-BIH is the one dataset where ``lead_names`` cannot be the whole truth.

        It stores two modified chest-placed leads, not any of the standard twelve,
        so it cannot go in the parametrised table above. More importantly the
        layout is not constant: counted over the 48 headers, 40 records store
        MLII/V1, two each store MLII/V5, MLII/V2 and V5/V2, one stores MLII/V4,
        and record 114 stores V5/MLII — the same pair as the majority, reversed.
        ``lead_names`` is therefore the *predominant* layout and
        ``record_lead_layouts`` carries the rest; see TestPerRecordLeadLayouts for
        what that changes at read time.
        """
        from ecgbench.config import load_config

        config = load_config("mitdb")
        assert config.leads == 2
        assert config.lead_names == ["MLII", "V1"]
        assert len(config.lead_names) == config.leads
        # Not a standard 12-lead name among them: MLII is a modified limb lead II
        # taken from chest electrodes, and there are no augmented leads at all.
        assert not {"aVR", "aVL", "aVF", "I", "II", "III"} & set(config.lead_names)
        # signal[0] is a limb-type lead in 46 records and a chest lead in 2, which
        # is exactly why an index cannot stand in for a name here.
        assert ["V5", "V2"] in config.record_lead_layouts
        assert ["V5", "MLII"] in config.record_lead_layouts

    def test_picsdb_names_its_single_channel_three_different_ways(self):
        """One channel per record, and the ten headers do not agree what it is.

        The mitdb problem at a lead count of one: seven records call the single
        channel ``II``, infant1 and infant5 (the 250 Hz "compound" recordings) call
        it ``ECG``, and infant10 calls it ``I``. ``alternate_lead_names`` cannot
        express that — it is keyed by lead *count*, and the count is 1 throughout —
        so ``record_lead_layouts`` carries it and ``leads=["II"]`` is resolved
        against each record's own header.

        The consequence is deliberate and worth pinning: ``leads=["II"]`` RAISES
        for three of the ten records rather than handing back the one channel there
        is. The release says only "a single channel of a 3-lead ECG" and nothing
        states that the ``ECG`` channel is lead II, so returning it under that name
        would let it be stacked with real lead II from other datasets.
        """
        from ecgbench.config import load_config

        config = load_config("picsdb")
        assert config.leads == 1
        assert config.lead_names == ["II"]  # the predominant name, 7 of 10
        assert len(config.lead_names) == config.leads
        assert config.record_lead_layouts == [["II"], ["ECG"], ["I"]]
        # Every layout is one channel — this is a naming difference, not a layout
        # difference, which is precisely what a count-keyed map cannot see.
        assert {len(layout) for layout in config.record_lead_layouts} == {1}
        assert config.alternate_lead_names is None

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

    # --- hdf5 format ---

    def test_hdf5_window_reads_the_requested_samples(
        self, tmp_hdf5_signal_dataset, sample_config
    ):
        """h5py slices on read, so the window must land on the right samples."""
        config_overrides = {"signal_format": "hdf5"}
        full = self._ds(
            tmp_hdf5_signal_dataset, sample_config, config_overrides=config_overrides
        )[0]["signal"]
        assert tuple(full.shape) == (12, 5000)
        # Stored (leads, samples) already, so a stray transpose would show here.
        assert full[0, 0].item() == 0
        assert full[3, 7].item() == 3 * 100_000 + 7

        second = self._ds(
            tmp_hdf5_signal_dataset,
            sample_config,
            config_overrides=config_overrides,
            window=(2500, 2500),
        )[0]["signal"]

        assert tuple(second.shape) == (12, 2500)
        assert second[0, 0].item() == 2500
        assert second[3, 0].item() == 3 * 100_000 + 2500
        assert torch.equal(second, full[:, 2500:5000])

    def test_hdf5_window_to_the_end(self, tmp_hdf5_signal_dataset, sample_config):
        ds = self._ds(
            tmp_hdf5_signal_dataset,
            sample_config,
            config_overrides={"signal_format": "hdf5"},
            window=(4000, None),
        )
        assert tuple(ds[0]["signal"].shape) == (12, 1000)

    def test_hdf5_window_past_the_end_raises_a_useful_error(
        self, tmp_hdf5_signal_dataset, sample_config
    ):
        from ecgbench.dataset import WindowOutOfRangeError

        ds = self._ds(
            tmp_hdf5_signal_dataset,
            sample_config,
            config_overrides={"signal_format": "hdf5"},
            window=(4000, 2000),
        )
        with pytest.raises(WindowOutOfRangeError) as excinfo:
            ds[0]
        message = str(excinfo.value)
        assert "rec_" in message  # names the record
        assert "5000 samples" in message  # and its real length

    def test_hdf5_explicit_dataset_key_and_its_failure_modes(self, tmp_hdf5_signal_dataset):
        """``<file>.h5:<dataset>`` is how a multi-dataset file gets disambiguated."""
        h5py = pytest.importorskip("h5py")

        from ecgbench.dataset import _load_signal, _parse_hdf5_ref

        path = str(tmp_hdf5_signal_dataset / "records" / "rec_0.h5")
        implicit = _load_signal(path, "hdf5")
        assert np.array_equal(_load_signal(f"{path}:ecg", "hdf5"), implicit)

        # A colon inside a directory name is not a key.
        assert _parse_hdf5_ref("/a:b/c.h5") == ("/a:b/c.h5", None, None)
        assert _parse_hdf5_ref(f"{path}:ecg") == (path, "ecg", None)
        # A bare trailing number is still a key, not a row: the row form needs
        # the dataset named too, so a dataset called "3" stays reachable.
        assert _parse_hdf5_ref(f"{path}:3") == (path, "3", None)

        with pytest.raises(ValueError, match="not in the file"):
            _load_signal(f"{path}:missing", "hdf5")

        # Two root datasets and no key: refuse rather than guess.
        ambiguous = tmp_hdf5_signal_dataset / "two.h5"
        with h5py.File(ambiguous, "w") as handle:
            handle.create_dataset("ecg", data=np.zeros((12, 10), dtype=np.float32))
            handle.create_dataset("other", data=np.zeros((12, 10), dtype=np.float32))
        with pytest.raises(ValueError, match="ambiguous"):
            _load_signal(str(ambiguous), "hdf5")
        assert _load_signal(f"{ambiguous}:ecg", "hdf5").shape == (12, 10)

    def test_hdf5_row_of_a_shared_3d_array(self, tmp_path):
        """``<file>.h5:<dataset>:<row>`` — the CODE layout, many records per file.

        Note the orientation flip: a 2-D HDF5 array is ``(leads, samples)`` and a
        3-D one is ``(records, samples, leads)``. That is what the releases ship,
        so the reader transposes one and not the other.
        """
        h5py = pytest.importorskip("h5py")

        from ecgbench.dataset import (
            WindowOutOfRangeError,
            _load_signal,
            _parse_hdf5_ref,
        )

        # Values encode (row, lead, sample) so a wrong row, a transpose or a
        # shifted window cannot pass by accident.
        raw = np.arange(3 * 40 * 12, dtype=np.float32).reshape(3, 40, 12)
        shared = tmp_path / "exams.hdf5"
        with h5py.File(shared, "w") as handle:
            handle.create_dataset("tracings", data=raw)
            handle.create_dataset("exam_id", data=np.array([10, 11, 12]))

        ref = f"{shared}:tracings:1"
        assert _parse_hdf5_ref(ref) == (str(shared), "tracings", 1)

        signal = _load_signal(ref, "hdf5")
        assert signal.shape == (12, 40)  # (leads, samples), transposed
        assert np.array_equal(signal, raw[1].T)

        windowed = _load_signal(ref, "hdf5", window=(5, 10))
        assert np.array_equal(windowed, raw[1].T[:, 5:15])

        with pytest.raises(WindowOutOfRangeError):
            _load_signal(ref, "hdf5", window=(35, 10))

        # A 3-D array with no row named is ambiguous, and a row past the end is
        # an error rather than a numpy wrap-around.
        with pytest.raises(ValueError, match="must name one"):
            _load_signal(f"{shared}:tracings", "hdf5")
        with pytest.raises(ValueError, match="holds 3 records"):
            _load_signal(f"{shared}:tracings:9", "hdf5")

        # A row index against a 2-D array is a mistake, not a no-op.
        flat = tmp_path / "one.h5"
        with h5py.File(flat, "w") as handle:
            handle.create_dataset("ecg", data=np.zeros((12, 40), dtype=np.float32))
        with pytest.raises(ValueError, match="only applies to a 3-D"):
            _load_signal(f"{flat}:ecg:0", "hdf5")

        # And the validation engine's own copy must agree.
        from ecgbench.validation.engine import _load_signal as validation_load

        assert np.array_equal(validation_load(ref, "hdf5"), signal)

    def test_hdf5_validation_engine_agrees_with_the_dataset_reader(
        self, tmp_hdf5_signal_dataset
    ):
        """The two _load_signal copies must not drift — validation has its own."""
        from ecgbench.dataset import _load_signal
        from ecgbench.validation.engine import _load_signal as validation_load

        pytest.importorskip("h5py")
        path = str(tmp_hdf5_signal_dataset / "records" / "rec_0.h5")
        assert np.array_equal(validation_load(path, "hdf5"), _load_signal(path, "hdf5"))
        # And the unit scale is applied the same way on both sides.
        assert np.allclose(
            validation_load(path, "hdf5", 0.001), _load_signal(path, "hdf5", 0.001)
        )

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


class TestAlternateLeadLayouts:
    """A release whose records do not all store the same leads.

    ZZU-pECG is the only such dataset in the catalogue: 12,334 records hold 12
    leads and 1,856 hold 9, dropping V2, V4 and V6. The 9-lead layout is not a
    prefix of the 12-lead one, so stored position 7 is V2 in one and V3 in the
    other. Selecting leads by index — which is what a single ``lead_names`` list
    amounts to — silently returns V3 where V2 was asked for, for 13% of the
    release. These tests pin the behaviour that prevents it.
    """

    def _ds(self, root, config, **kwargs):
        from dataclasses import replace

        from ecgbench.dataset import ECGDataset

        from .conftest import NINE_LEAD_LAYOUT, TWELVE_LEAD_LAYOUT

        config = replace(
            config,
            signal_format="wfdb",
            leads=12,
            lead_names=list(TWELVE_LEAD_LAYOUT),
            alternate_lead_names=kwargs.pop(
                "alternate_lead_names", {9: list(NINE_LEAD_LAYOUT)}
            ),
        )
        return ECGDataset(
            config,
            split="train",
            version="clean",
            data_path=root,
            metadata_source="local",
            **kwargs,
        )

    def test_unselected_reads_return_each_record_its_own_lead_count(
        self, tmp_mixed_lead_dataset, sample_config
    ):
        """Without leads=, nothing is re-indexed and both layouts load as stored."""
        ds = self._ds(tmp_mixed_lead_dataset, sample_config)
        assert tuple(ds[0]["signal"].shape) == (12, 5000)  # rec_0, 12-lead
        assert tuple(ds[2]["signal"].shape) == (9, 5000)  # rec_2, 9-lead

    def test_a_shared_lead_resolves_to_the_same_physical_lead_in_both_layouts(
        self, tmp_mixed_lead_dataset, sample_config
    ):
        """V5 is position 10 in the 12-lead layout and position 8 in the 9-lead one.

        The fixture encodes each lead's canonical index in its samples, so this
        asserts the *physical* lead rather than merely a shape.
        """
        from .conftest import CANONICAL_LEAD_INDEX

        ds = self._ds(tmp_mixed_lead_dataset, sample_config, leads=["V5"])
        assert ds.lead_names == ("V5",)
        twelve = ds[0]["signal"]
        nine = ds[2]["signal"]
        assert tuple(twelve.shape) == (1, 5000)
        assert tuple(nine.shape) == (1, 5000)
        expected = CANONICAL_LEAD_INDEX["V5"]
        assert twelve[0, 0].item() == pytest.approx(expected, abs=1e-3)
        # The point: same value from a record that stores V5 four positions
        # earlier. An index-based selection would return V3 (8.0) here.
        assert nine[0, 0].item() == pytest.approx(expected, abs=1e-3)

    def test_a_lead_absent_from_the_reduced_layout_raises_rather_than_substituting(
        self, tmp_mixed_lead_dataset, sample_config
    ):
        """V2 does not exist in a 9-lead record, and position 7 there holds V3.

        This is the bug the mechanism exists for: before it, ``leads=["V2"]``
        returned V3's samples for these records with no error at all, because
        index 7 is a valid index into a 9-row signal.
        """
        from .conftest import CANONICAL_LEAD_INDEX

        ds = self._ds(tmp_mixed_lead_dataset, sample_config, leads=["V2"])
        # Fine for a record that has V2.
        assert ds[0]["signal"][0, 0].item() == pytest.approx(
            CANONICAL_LEAD_INDEX["V2"], abs=1e-3
        )
        with pytest.raises(ValueError, match="not in 'test_dataset'|V2"):
            ds[2]["signal"]

    def test_a_lead_count_covered_by_no_declared_layout_refuses_to_guess(
        self, tmp_mixed_lead_dataset, sample_config
    ):
        """Once a dataset declares that layout varies, an unknown count is an error.

        The map here covers 10 leads, not the 9 the fixture's reduced records
        hold, so there is no layout to resolve against and guessing is exactly
        what must not happen.
        """
        ds = self._ds(
            tmp_mixed_lead_dataset,
            sample_config,
            alternate_lead_names={10: ["I", "II", "III", "AVR", "AVL", "AVF",
                                       "V1", "V2", "V3", "V4"]},
            leads=["I"],
        )
        assert ds[0]["signal"].shape[0] == 1  # the 12-lead record is unaffected
        with pytest.raises(ValueError, match="alternate_lead_names covers only"):
            ds[2]["signal"]

    def test_declaring_no_alternates_keeps_the_original_behaviour(
        self, tmp_mixed_lead_dataset, sample_config
    ):
        """A dataset with no alternate_lead_names asserts one layout throughout.

        Every other dataset in the catalogue is in this case, so the declared
        indices must still be used unchanged — the out-of-range check is what
        guards them, exactly as before.
        """
        ds = self._ds(
            tmp_mixed_lead_dataset, sample_config, alternate_lead_names=None, leads=["I"]
        )
        # Position 0 is lead I in both layouts, so this resolves for both.
        assert ds[0]["signal"].shape[0] == 1
        assert ds[2]["signal"].shape[0] == 1
        # And a lead beyond the reduced record's rows still raises the old error.
        ds = self._ds(
            tmp_mixed_lead_dataset, sample_config, alternate_lead_names=None, leads=["V6"]
        )
        with pytest.raises(ValueError, match="too few for the requested"):
            ds[2]["signal"]

    def test_zzu_pecg_declares_the_layout_its_reduced_records_use(self):
        """The shipped config must carry the map, or the guard never engages."""
        from ecgbench.config import load_config

        config = load_config("zzu_pecg")
        assert config.alternate_lead_names is not None
        nine = config.alternate_lead_names[9]
        assert len(nine) == 9
        # The three dropped leads, and the reason position 7 differs.
        assert set(config.lead_names) - set(nine) == {"V2", "V4", "V6"}
        assert config.lead_names[7] == "V2"
        assert nine[7] == "V3"

    def test_ikem_stores_eight_leads_precordial_first_with_ii_before_i(self):
        """The most unusual lead order in the catalogue, and it is not a typo.

        IKEM keeps only the 8 independent leads and stores them V1-V6, II, I.
        Derived from the arrays: ch0-ch5 show the banded correlation structure of
        a precordial sweep with net QRS dominance running -377 counts (rS, V1) to
        +408 (dominant R, V6), no channel matches any augmented-lead identity,
        and taking ch7 as I puts the frontal QRS axis at a median +51 deg where
        the swap gives +1 deg.
        """
        from ecgbench.config import load_config

        config = load_config("ikem")
        assert config.lead_names == ["V1", "V2", "V3", "V4", "V5", "V6", "II", "I"]
        assert config.leads == 8
        # No augmented lead is stored, so nothing here can be indexed as one.
        assert not {"III", "aVR", "aVL", "aVF"} & set(config.lead_names)
        # The habit-driven index is wrong: signal[0] is V1, not lead I.
        assert config.lead_names[0] != "I"
        # And it has no alternate layout — all 98,130 records store the same 8.
        assert config.alternate_lead_names is None


class TestPerRecordLeadLayouts:
    """A release whose records store the same NUMBER of leads under different NAMES.

    ZZU-pECG varies the lead count, which a count-keyed ``alternate_lead_names``
    map can resolve. MIT-BIH does not: all 48 of its records hold exactly 2
    leads, but 40 store MLII/V1 and the other 8 store MLII/V5, MLII/V2, MLII/V4,
    V5/V2 — or, for record 114, the predominant pair *reversed*. There is no
    count to key on, so ``record_lead_layouts`` says "layout varies, read it from
    the record" and ``ECGDataset`` resolves the requested names against each
    record's own header. These tests pin that.
    """

    def _ds(self, root, config, **kwargs):
        from dataclasses import replace

        from ecgbench.dataset import ECGDataset

        from .conftest import MITDB_LAYOUTS

        config = replace(
            config,
            signal_format="wfdb",
            leads=2,
            lead_names=list(MITDB_LAYOUTS[0]),
            record_lead_layouts=kwargs.pop(
                "record_lead_layouts", [list(layout) for layout in MITDB_LAYOUTS]
            ),
        )
        return ECGDataset(
            config,
            split="train",
            version="clean",
            data_path=root,
            metadata_source="local",
            **kwargs,
        )

    def test_unselected_reads_are_untouched(
        self, tmp_varied_lead_name_dataset, sample_config
    ):
        """Without leads=, nothing is re-indexed and every record loads as stored."""
        ds = self._ds(tmp_varied_lead_name_dataset, sample_config)
        assert tuple(ds[0]["signal"].shape) == (2, 5000)
        assert tuple(ds[2]["signal"].shape) == (2, 5000)

    def test_a_reversed_record_still_returns_the_lead_that_was_asked_for(
        self, tmp_varied_lead_name_dataset, sample_config
    ):
        """Record 1 stores V5 then MLII — mitdb's record 114, whose signals are reversed.

        This is the failure the mechanism exists for. Both records store 2 leads,
        so nothing about the shape distinguishes them, and an index-based
        selection returns V5 where MLII was asked for with no error at all.
        """
        from .conftest import CANONICAL_LEAD_INDEX

        ds = self._ds(tmp_varied_lead_name_dataset, sample_config, leads=["MLII"])
        assert ds.lead_names == ("MLII",)
        normal = ds[0]["signal"]  # MLII, V1 — MLII at position 0
        reversed_ = ds[1]["signal"]  # V5, MLII — MLII at position 1
        assert tuple(normal.shape) == tuple(reversed_.shape) == (1, 5000)
        expected = CANONICAL_LEAD_INDEX["MLII"]
        assert normal[0, 0].item() == pytest.approx(expected, abs=1e-3)
        # The point: the same physical lead from a record that stores it second.
        # Taking index 0 here would return V5 (10.0).
        assert reversed_[0, 0].item() == pytest.approx(expected, abs=1e-3)
        assert reversed_[0, 0].item() != pytest.approx(
            CANONICAL_LEAD_INDEX["V5"], abs=1e-3
        )

    def test_a_lead_absent_from_a_records_layout_raises_rather_than_substituting(
        self, tmp_varied_lead_name_dataset, sample_config
    ):
        """Record 2 stores V5/V2 and has no MLII — mitdb's records 102 and 104.

        Both positions hold a valid lead, so there is nothing an index-based
        selection could complain about; it would simply return V5.
        """
        ds = self._ds(tmp_varied_lead_name_dataset, sample_config, leads=["MLII"])
        assert ds[0]["signal"].shape[0] == 1  # fine where MLII exists
        with pytest.raises(ValueError, match="more than one lead layout"):
            ds[2]["signal"]

    def test_a_lead_in_no_layout_at_all_fails_at_construction(
        self, tmp_varied_lead_name_dataset, sample_config
    ):
        """A typo should not wait until the first __getitem__ to surface.

        The request is checked against the union of the declared layouts, so a
        name that no record could ever hold fails immediately.
        """
        with pytest.raises(ValueError, match="V6"):
            self._ds(tmp_varied_lead_name_dataset, sample_config, leads=["V6"])

    def test_a_lead_in_only_some_layouts_is_accepted_at_construction(
        self, tmp_varied_lead_name_dataset, sample_config
    ):
        """V2 is in one layout of four, and asking for it is legitimate.

        Resolving against ``lead_names`` alone would reject it out of hand, even
        though a record storing V2 exists — which is why construction checks the
        union and read time checks the record.
        """
        from .conftest import CANONICAL_LEAD_INDEX

        ds = self._ds(tmp_varied_lead_name_dataset, sample_config, leads=["V2"])
        assert ds.lead_names == ("V2",)
        assert ds[2]["signal"][0, 0].item() == pytest.approx(
            CANONICAL_LEAD_INDEX["V2"], abs=1e-3
        )
        # ...and refused for a record that does not store it.
        with pytest.raises(ValueError, match="more than one lead layout"):
            ds[0]["signal"]

    def test_resolution_is_cached_per_record_path(
        self, tmp_varied_lead_name_dataset, sample_config
    ):
        """One header read per record, not one per __getitem__."""
        ds = self._ds(tmp_varied_lead_name_dataset, sample_config, leads=["MLII"])
        assert ds._path_lead_index == {}
        ds[0]["signal"]
        ds[0]["signal"]
        assert len(ds._path_lead_index) == 1
        ds[1]["signal"]
        # Different records, different resolutions — 0 for one, 1 for the other.
        assert sorted(ds._path_lead_index.values()) == [[0], [1]]

    def test_declaring_no_layouts_keeps_the_original_behaviour(
        self, tmp_varied_lead_name_dataset, sample_config
    ):
        """Every other dataset in the catalogue asserts one layout throughout.

        Without the field the declared indices are used unchanged, which is
        precisely the silent substitution — asserted here so a future edit cannot
        quietly make the field a no-op and still pass.
        """
        from .conftest import CANONICAL_LEAD_INDEX

        ds = self._ds(
            tmp_varied_lead_name_dataset,
            sample_config,
            record_lead_layouts=None,
            leads=["MLII"],
        )
        assert ds[0]["signal"][0, 0].item() == pytest.approx(
            CANONICAL_LEAD_INDEX["MLII"], abs=1e-3
        )
        # The reversed record hands back V5, with no error. That is the bug.
        assert ds[1]["signal"][0, 0].item() == pytest.approx(
            CANONICAL_LEAD_INDEX["V5"], abs=1e-3
        )

    def test_a_format_with_no_per_record_lead_names_refuses(
        self, tmp_varied_lead_name_dataset, sample_config
    ):
        """Only wfdb names its leads per record; anything else has nothing to read."""
        from dataclasses import replace

        ds = self._ds(tmp_varied_lead_name_dataset, sample_config, leads=["MLII"])
        ds.config = replace(ds.config, signal_format="csv")
        with pytest.raises(ValueError, match="stores no lead names per record"):
            ds._lead_index_for(2, "rec_0", "irrelevant")

    def test_mitdb_lead_names_are_the_predominant_layout_and_not_the_only_one(self):
        """The shipped config must carry the layouts, or the guard never engages."""
        from ecgbench.config import load_config

        config = load_config("mitdb")
        assert config.lead_names == ["MLII", "V1"]
        assert config.leads == 2
        layouts = config.record_lead_layouts
        assert config.lead_names in layouts
        # Every layout holds 2 leads, which is exactly why a count-keyed map is
        # useless here.
        assert {len(layout) for layout in layouts} == {2}
        assert config.alternate_lead_names is None
        # MLII is absent from one layout entirely — records 102 and 104.
        assert any("MLII" not in layout for layout in layouts)
        # Five distinct lead names across the release.
        assert {name for layout in layouts for name in layout} == {
            "MLII", "V1", "V2", "V4", "V5"
        }

    def test_challenge2017_declares_the_single_channel_name_its_headers_use(self):
        """The one single-lead dataset here, and its channel is called "ECG", not "I".

        The AliveCor device produces a nominal lead I (LA-RA) equivalent, so "I"
        is the tempting name — but the challenge paper states that "many of the
        ECGs were inverted (RA-LA) since the device did not require the user to
        rotate it in any particular orientation", and all 8,528 headers name the
        channel ``ECG``. Naming it "I" would let it be stacked with 12-lead data
        by name while an unknown fraction of the records carry the opposite sign.
        """
        from ecgbench.config import load_config

        config = load_config("challenge2017")
        assert config.leads == 1
        assert config.lead_names == ["ECG"]
        assert len(config.lead_names) == config.leads
        # Emphatically not a 12-lead name, and specifically not lead I.
        assert "I" not in config.lead_names
        assert not {"I", "II", "III", "aVR", "aVL", "aVF"} & set(config.lead_names)
        # And no alternate layout: one channel in every record.
        assert not config.alternate_lead_names
        assert not config.record_lead_layouts

    def test_apnea_ecg_declares_the_single_channel_name_its_headers_use(self):
        """The second single-lead dataset here, and its channel is also "ECG".

        Like challenge2017, and for the same reason: all 70 headers end their one
        signal line with the bare description ``ECG``, and the release documents
        no electrode placement anywhere — not on the landing page, not in
        ``annotations.html``, not in ``additional-information.txt``. The overnight
        Holter montage would make a modified chest lead the likely guess, but a
        guess is exactly what naming it "II" or "V2" would ship. It stays a
        channel position.
        """
        from ecgbench.config import load_config

        config = load_config("apnea_ecg")
        assert config.leads == 1
        assert config.lead_names == ["ECG"]
        assert len(config.lead_names) == config.leads
        # Must not be stackable with 12-lead data by name.
        assert not {"I", "II", "III", "aVR", "aVL", "aVF"} & set(config.lead_names)
        # One channel in every record: no per-record layout mechanism applies.
        assert not config.alternate_lead_names
        assert not config.record_lead_layouts

    def test_afdb_channels_are_positions_and_must_not_be_named_leads(self):
        """AFDB's headers call its two channels ECG1/ECG2 and name no anatomy.

        Its sibling MIT-BIH Arrhythmia Database documents MLII/V1, and the
        temptation is to carry that across. Nothing in the AFDB release supports
        it: there is no electrode placement statement anywhere, so ECG1/ECG2 are
        channel positions. This pins the config against a "helpful" edit.
        """
        from ecgbench.config import load_config

        config = load_config("afdb")
        assert config.lead_names == ["ECG1", "ECG2"]
        assert config.leads == 2
        # One layout in all 23 readable headers, so neither per-record mechanism
        # applies — and no anatomical lead name appears at all.
        assert config.record_lead_layouts is None
        assert config.alternate_lead_names is None
        assert not {"MLII", "V1", "V5", "II"} & set(config.lead_names)

    def test_ltafdb_channels_are_positions_and_the_files_do_not_even_number_them(self):
        """LTAFDB's headers call BOTH channels "ECG" — the same string twice.

        afdb at least spells its two channels ECG1 and ECG2. Here there is nothing
        to tell them apart by name, so ECGBench declares the positional names
        ECG1/ECG2 and the config says so. That is a deliberate deviation from
        "spell leads as the source spells them", taken because the alternative —
        ["ECG", "ECG"] — makes channel 1 unreachable through leads= entirely:
        _resolve_leads keys on the first occurrence of a name and rejects a
        repeated request.
        """
        from ecgbench.config import load_config
        from ecgbench.dataset import _resolve_leads

        config = load_config("ltafdb")
        assert config.lead_names == ["ECG1", "ECG2"]
        assert config.leads == 2
        # One layout in all 84 headers, so neither per-record mechanism applies.
        assert config.record_lead_layouts is None
        assert config.alternate_lead_names is None
        assert not {"MLII", "V1", "V5", "II"} & set(config.lead_names)

        # Both positions are reachable by name, which is the point of the choice.
        assert _resolve_leads(["ECG2"], config.lead_names, "ltafdb")[0] == [1]
        assert _resolve_leads(["ECG2", "ECG1"], config.lead_names, "ltafdb")[0] == [1, 0]
        # What the honest-but-unusable alternative would have cost:
        assert _resolve_leads(["ECG"], ["ECG", "ECG"], "ltafdb")[0] == [0]
        with pytest.raises(ValueError, match="requested more than once"):
            _resolve_leads(["ECG", "ECG"], ["ECG", "ECG"], "ltafdb")

    def test_nsrdb_channels_are_positions_like_afdb_and_not_named_leads(self):
        """NSRDB's headers spell ECG1/ECG2 and, like afdb, name no anatomy.

        Same institution and same Arrhythmia Laboratory as the MIT-BIH Arrhythmia
        Database, which does document MLII/V1 — and the same absence of any
        electrode placement statement as afdb. All 18 headers agree on the two
        names, so the one layout needs neither per-record mechanism.
        """
        from ecgbench.config import load_config

        config = load_config("nsrdb")
        assert config.lead_names == ["ECG1", "ECG2"]
        assert config.leads == 2
        assert config.record_lead_layouts is None
        assert config.alternate_lead_names is None
        assert not {"MLII", "V1", "V5", "II"} & set(config.lead_names)

    def test_svdb_channels_are_positions_and_the_catalogue_once_said_otherwise(self):
        """SVDB's headers spell ECG1/ECG2 and, like afdb and nsrdb, name no anatomy.

        This one is worth its own test because the claim it refutes was shipped:
        the catalogue entry described SVDB as "2-lead (MLII + V1) · 360 Hz" before
        this config existed, and both halves were wrong — the recordings are 128 Hz
        and all 78 headers name the channels ECG1/ECG2 with no electrode placement
        stated anywhere in the release. The plausible-looking values came from
        assuming mitdb's properties carry across, which is exactly what
        ``lead_names`` exists to stop.
        """
        from ecgbench.config import load_config

        config = load_config("svdb")
        assert config.lead_names == ["ECG1", "ECG2"]
        assert config.leads == 2
        assert config.sampling_rates == [128]
        assert config.record_lead_layouts is None
        assert config.alternate_lead_names is None
        assert not {"MLII", "V1", "V5", "II"} & set(config.lead_names)

    def test_szdb_is_the_only_single_channel_holter_and_its_channel_is_a_position(self):
        """One channel, described only as "ECG", with no electrode placement stated.

        The paper says "continuous single-lead ECG signals" and nothing more, so
        ``ECGDataset(leads=["ECG"])`` selects the one channel there is rather than
        a known anatomical lead — the afdb/nsrdb/svdb/chfdb situation, at half the
        channel count. Stacking it with mitdb by index crosses leads just as those
        do.

        The wrinkle here matches chfdb's exactly and is why ``lead_names`` was read
        from the current ``.hea`` files: the 7 superseded ``.hea-`` copies that
        ship beside them — and are listed in the release's own SHA256SUMS.txt —
        describe the channel as ``column 1``, so a reader pointed at those loses
        name-based selection entirely.
        """
        from ecgbench.config import load_config

        config = load_config("szdb")
        assert config.lead_names == ["ECG"]
        assert config.leads == 1
        assert config.sampling_rates == [200]
        assert config.record_lead_layouts is None
        assert config.alternate_lead_names is None
        assert not {"MLII", "V1", "V5", "II", "ECG1"} & set(config.lead_names)

    def test_chfdb_channels_are_positions_and_only_the_current_headers_name_them(self):
        """CHFDB's headers spell ECG1/ECG2 and, like afdb and nsrdb, name no anatomy.

        Same hospital as the MIT-BIH Arrhythmia Database, which does document
        MLII/V1, and the same absence of any electrode placement statement as afdb.
        All 15 headers agree on the two names, so the one layout needs neither
        per-record mechanism.

        One wrinkle specific to this release: the names exist only in the *current*
        headers. The 15 superseded ``.hea-`` copies that ship beside them — and are
        listed in the release's own SHA256SUMS.txt — carry no signal descriptions at
        all, because the 2012 revision is what added them. A reader pointed at those
        gets two unnamed channels, which is why ``lead_names`` was read from the
        ``.hea`` files.
        """
        from ecgbench.config import load_config

        config = load_config("chfdb")
        assert config.lead_names == ["ECG1", "ECG2"]
        assert config.leads == 2
        assert config.sampling_rates == [250]
        assert config.record_lead_layouts is None
        assert config.alternate_lead_names is None
        assert not {"MLII", "V1", "V5", "II"} & set(config.lead_names)

    def test_sddb_channels_are_both_called_ecg_like_ltafdb_not_numbered_like_afdb(self):
        """SDDB's headers call BOTH channels "ECG" — the ltafdb case, not the afdb one.

        Worth its own test because the two sibling situations look alike and are
        not. afdb, nsrdb, svdb and chfdb ship headers that spell ECG1 and ECG2, so
        the declared names are the source's own. Here — as in ltafdb — every signal
        line of all 23 current headers ends in the bare description ``ECG``, twice,
        so ["ECG", "ECG"] would be the literal reading and would make channel 1
        unreachable through ``leads=`` entirely. ECGBench declares the positional
        names instead.

        A wrinkle specific to this release: the 23 superseded ``.hea-`` copies that
        ship beside the current headers — and are listed in the release's own
        SHA256SUMS.txt — describe the channels as "record 30, signal 0" and "record
        30, signal 1". The 2008 revision is what replaced those with "ECG", so a
        reader pointed at the backups gets two differently-named channels. Neither
        naming states an electrode placement, in the headers or on the landing page.
        """
        from ecgbench.config import load_config
        from ecgbench.dataset import _resolve_leads

        config = load_config("sddb")
        assert config.lead_names == ["ECG1", "ECG2"]
        assert config.leads == 2
        assert config.sampling_rates == [250]
        # One layout in all 23 headers, so neither per-record mechanism applies.
        assert config.record_lead_layouts is None
        assert config.alternate_lead_names is None
        assert not {"MLII", "V1", "V5", "II"} & set(config.lead_names)

        # Both positions stay reachable by name, which is why the names are
        # positional rather than the literal duplicate the files carry.
        assert _resolve_leads(["ECG2"], config.lead_names, "sddb")[0] == [1]
        assert _resolve_leads(["ECG2", "ECG1"], config.lead_names, "sddb")[0] == [1, 0]

    def test_stdb_is_the_one_two_lead_release_where_ten_records_hold_a_single_channel(self):
        """STDB's headers say "ECG" like sddb's — but only 18 of 28 say it twice.

        Two separate facts, and the second is the one nothing documents. Like sddb
        and ltafdb, every signal line ends in the bare description ``ECG``, so
        ["ECG", "ECG"] would be the literal reading and would make channel 1
        unreachable through ``leads=``; ECGBench declares positional names instead.
        The temptation to "correct" them to MLII/V1 is strongest here of all,
        because this release shares mitdb's 360 Hz rate and its record numbering
        style — and nothing whatsoever supports it.

        On top of that, ten records (313-317 and 319-323) declare ONE signal, not
        two, which neither the landing page nor the catalogue's "2 leads" mentions.
        ``alternate_lead_names`` declares that layout so ``leads=["ECG2"]`` raises
        for those records against a named layout instead of falling into the
        generic too-few-leads path.
        """
        from ecgbench.config import load_config
        from ecgbench.dataset import _resolve_leads

        config = load_config("stdb")
        assert config.lead_names == ["ECG1", "ECG2"]
        assert config.leads == 2
        assert config.sampling_rates == [360]
        # Same count, different names is the mitdb mechanism and does not apply:
        # all 46 channels here carry the identical description.
        assert config.record_lead_layouts is None
        # Different count IS the mechanism that applies.
        assert config.alternate_lead_names == {1: ["ECG1"]}
        assert not {"MLII", "V1", "V5", "II"} & set(config.lead_names)

        # Both positions stay reachable by name on the 18 two-channel records.
        assert _resolve_leads(["ECG2"], config.lead_names, "stdb")[0] == [1]
        assert _resolve_leads(["ECG2", "ECG1"], config.lead_names, "stdb")[0] == [1, 0]
        # And ECG1 resolves against the single-channel layout too, which is what
        # makes leads=["ECG1"] the safe way to batch this dataset whole.
        assert _resolve_leads(["ECG1"], config.alternate_lead_names[1], "stdb")[0] == [0]

    def test_shdb_af_is_the_one_two_lead_holter_whose_channel_names_mean_something(self):
        """ECG1 is a modified CC5 lead and ECG2 a NASA lead, and the release says so.

        Every other two-lead Holter in this catalogue exposes ECG1/ECG2 as bare
        channel positions, because none of them states an electrode placement
        anywhere. This one does — in the Data Description section and in the shipped
        README — so ``leads=["ECG1"]`` here selects a known placement rather than
        merely position 0. That does not make them stackable with 12-lead data:
        modified CC5 and NASA are neither of them any of the standard twelve.

        The names still come from the headers rather than from the documentation,
        which is why they read ECG1/ECG2 and not CC5/NASA — spelling leads as the
        source spells them is the rule, and the release spells them this way in all
        128 headers.
        """
        from ecgbench.config import load_config

        config = load_config("shdb_af")
        assert config.lead_names == ["ECG1", "ECG2"]
        assert config.leads == 2
        # One layout in all 128 headers, so neither per-record mechanism applies.
        assert config.record_lead_layouts is None
        assert config.alternate_lead_names is None
        assert not {"MLII", "V1", "V5", "II", "CC5", "NASA"} & set(config.lead_names)

    def test_the_six_two_lead_mit_bih_holters_present_one_convention(self):
        """afdb, ltafdb, nsrdb, svdb, chfdb and sddb all expose ECG1/ECG2.

        For two different reasons, which is the point of grouping them. afdb's,
        nsrdb's, svdb's and chfdb's headers spell ECG1/ECG2 themselves; ltafdb's and
        sddb's call both channels "ECG" and ECGBench numbers them so both are
        reachable by name. None of the six states an electrode placement, so all six
        are channel positions — and stacking any of them with mitdb by index crosses
        leads.
        """
        from ecgbench.config import load_config

        slugs = ("afdb", "ltafdb", "nsrdb", "svdb", "chfdb", "sddb")
        assert {tuple(load_config(s).lead_names) for s in slugs} == {("ECG1", "ECG2")}
        assert load_config("mitdb").lead_names == ["MLII", "V1"]

    def test_butqdb_declares_the_single_channel_its_headers_name(self):
        """BUT QDB's 18 headers all name one channel, "ECG", and no anatomy.

        A chest-worn Bittium Faros 180 under free-living conditions; the release
        says "single-lead" and states no electrode placement, so the name is a
        channel position like afdb's and nsrdb's rather than a derivation. With one
        channel and one layout, ``leads=[...]`` has nothing to choose between — but
        the name still has to be declared, because ``ECGDataset(leads=...)`` is
        unusable without it.

        The 100 Hz three-axis accelerometer shipped alongside every recording is
        deliberately not a lead and not a declared rate: it is a separate WFDB
        record (``<id>_ACC``, channels ACCx/ACCy/ACCz in milli-g) and not an ECG.
        """
        from ecgbench.config import load_config

        config = load_config("butqdb")
        assert config.lead_names == ["ECG"]
        assert config.leads == 1
        assert config.record_lead_layouts is None
        assert config.alternate_lead_names is None
        assert not {"MLII", "V1", "II", "ECG1", "ECG2"} & set(config.lead_names)
        assert not {"ACCx", "ACCy", "ACCz"} & set(config.lead_names)
        assert config.sampling_rates == [1000]

    def test_edb_declares_fifteen_layouts_and_no_lead_common_to_every_record(self):
        """The most varied lead layout in the catalogue, and the weakest ``lead_names``.

        All 90 European ST-T records store two leads, and they use FIFTEEN different
        orderings of eleven different lead pairs, counted from the 90 headers. Two
        consequences the mitdb case does not have:

        - ``lead_names`` describes only 19 of the 90 records, so unlike every other
          config here it is a genuine minority. It exists to name the modal layout,
          not to be indexed against.
        - **no lead is present in all 90 records** — V5 reaches 51, MLIII 47, V4 34
          and D3 exactly one — so every name-based selection on this dataset raises
          for some records. That is the documented behaviour, not a bug, but it means
          ``leads=`` cannot be used to make edb batchable without also restricting
          which records are loaded.

        MLIII/V4 and V4/MLIII are both declared, 15 records each: the same pair in
        either order, so ``signal[0]`` is a limb lead in one half and a chest lead in
        the other with nothing in the fold CSVs to say which.
        """
        from ecgbench.config import load_config

        config = load_config("edb")
        assert config.lead_names == ["V5", "MLI"]
        assert config.leads == 2
        layouts = config.record_lead_layouts
        assert len(layouts) == 15
        assert config.lead_names in layouts
        # Every layout holds 2 leads — the reason a count-keyed map is useless here.
        assert {len(layout) for layout in layouts} == {2}
        assert config.alternate_lead_names is None
        # The same pair in both orders, which is what breaks positional indexing.
        assert ["MLIII", "V4"] in layouts
        assert ["V4", "MLIII"] in layouts
        names = {name for layout in layouts for name in layout}
        assert names == {"V5", "MLI", "MLIII", "V4", "V1", "V2", "V3", "D3"}
        # No lead is in every layout, so there is no universally selectable lead.
        assert not {n for n in names if all(n in layout for layout in layouts)}

    def test_ltstdb_varies_the_lead_count_as_well_as_the_names(self):
        """The only release here where both the count and the names vary.

        68 of the 86 Long-Term ST records store two signals and 18 store three, in
        twelve layouts. ``alternate_lead_names`` is keyed by lead count and so
        cannot express "V4/MLIII and MLIII/V4 are both two-lead layouts";
        ``record_lead_layouts`` says "read it from the record" and covers both
        kinds of variation at once, which is why this config declares that one and
        not the other.

        Two consequences, both documented in the config rather than hidden:

        - **The modal layout names nothing.** 22 records describe both signals as
          ``ECG`` because their headers state "Electrode locations were not
          recorded", making it the single largest layout at 26% of the release. So
          ``leads=["ECG"]`` resolves to signal 0 for those 22 and there is no name
          that reaches signal 1.
        - **No lead is in all 86 records** — MLIII reaches 29, V4 27, ECG 22 — and
          a batch mixing 2- and 3-channel records raises in ``default_collate``
          anyway. This dataset cannot be batched whole by any ``leads=`` value.
        """
        from ecgbench.config import load_config

        config = load_config("ltstdb")
        assert config.lead_names == ["ECG", "ECG"]
        assert config.leads == 2
        layouts = config.record_lead_layouts
        assert len(layouts) == 12
        assert config.lead_names in layouts
        assert config.alternate_lead_names is None
        # BOTH lead counts appear, which is what rules alternate_lead_names out.
        assert {len(layout) for layout in layouts} == {2, 3}
        # The same pair in both orders, which is what breaks positional indexing
        # even within the two-lead half.
        assert ["V4", "MLIII"] in layouts
        assert ["MLIII", "V4"] in layouts
        # The modal layout names neither signal, and names them the SAME.
        assert ["ECG", "ECG"] in layouts
        names = {name for layout in layouts for name in layout}
        assert names == {
            "ECG", "V4", "MLIII", "ML2", "MV2", "E-S", "A-S", "A-I",
            "V3", "V5", "V2", "V6", "II", "aVF",
        }
        # No lead is in every layout, so there is no universally selectable lead.
        assert not {n for n in names if all(n in layout for layout in layouts)}

    def test_qtdb_declares_twenty_layouts_and_its_modal_one_is_a_placeholder(self):
        """The most varied layout in the catalogue, and the only placeholder modal pair.

        105 records, TWENTY layouts. What makes qtdb different from edb and mitdb is
        that its ``lead_names`` is not a real lead pair at all: 57 of the 105 records
        — every excerpt from the Supraventricular, Normal Sinus Rhythm, ST Change,
        Long-Term and sudden-death sources — describe both channels only as ECG1 and
        ECG2, stating no electrode placement anywhere. So the modal layout is a
        majority (57 of 105, where edb's covers 19 of 90) *and* is positional.

        The other 48 records do name their channels, in 19 further layouts, and both
        orders of the same pair appear: D3/V4 in 5 records and V4/D3 in 7.
        """
        from ecgbench.config import load_config

        config = load_config("qtdb")
        assert config.lead_names == ["ECG1", "ECG2"]
        assert config.leads == 2
        layouts = config.record_lead_layouts
        assert len(layouts) == 20
        assert config.lead_names in layouts
        # Every layout holds 2 leads — a count-keyed map cannot tell them apart.
        assert {len(layout) for layout in layouts} == {2}
        assert config.alternate_lead_names is None
        # The same pair in both orders, which is what breaks positional indexing.
        assert ["D3", "V4"] in layouts
        assert ["V4", "D3"] in layouts
        names = {name for layout in layouts for name in layout}
        # No lead is in every layout, so every name-based selection raises for some
        # records — including ECG1/ECG2, which raise for the 48 that name channels.
        assert not {n for n in names if all(n in layout for layout in layouts)}

    def test_qtdb_and_edb_disagree_about_the_names_of_identical_channels(self):
        """The same signal samples, two lead vocabularies — a silent coverage change.

        30 of qtdb's 33 European ST-T excerpts are bit-identical to edb 1.0.0's
        recordings, but qtdb keeps the ESC's original bipolar-electrode nomenclature
        while edb relabelled the same channels to standard names: edb's MLIII is
        qtdb's D3 or ML5, its V5 is CM5, its V2 is CM2, V1-V2 or V2-V3. Only
        ``sele0107`` and ``sele0704`` agree.

        Nothing returns the WRONG lead — no name maps to a different physical
        channel in the two releases — but the coverage changes silently. Of the 33
        records the two datasets share, ``leads=["V5"]`` selects **14** under edb's
        names and **2** under qtdb's, over signals that are the same samples. This
        asserts the disagreement rather than letting a later "consistency" edit
        erase it.
        """
        from ecgbench.config import load_config

        qtdb = load_config("qtdb").record_lead_layouts
        edb = load_config("edb").record_lead_layouts
        qtdb_names = {n for layout in qtdb for n in layout}
        edb_names = {n for layout in edb for n in layout}

        # qtdb carries electrode names edb has none of...
        assert {"CM5", "CC5", "ML5", "CM2", "CM4", "mod.V1"} <= qtdb_names
        assert not {"CM5", "CC5", "ML5", "CM2", "CM4", "mod.V1"} & edb_names
        # ...and edb carries limb-lead names qtdb has none of.
        assert {"MLI", "MLIII"} <= edb_names
        assert not {"MLI", "MLIII"} & qtdb_names
        # V5 is the one widely-used name both vocabularies share, so it is the name
        # a cross-dataset selection would reach for.
        assert "V5" in qtdb_names and "V5" in edb_names
        # And qtdb's placeholder pair, which covers 57 of its records, is absent
        # from edb entirely — edb names the channels of all 90 of its own.
        assert ["ECG1", "ECG2"] in qtdb
        assert not {"ECG1", "ECG2"} & edb_names

    # Which configs may declare record_lead_layouts is pinned once, in
    # tests/test_config.py::test_only_mitdb_edb_and_qtdb_declare_per_record_lead_layouts.


class TestZeroPaddedRecordIds:
    """Fold CSVs must round-trip identifiers as strings, or afdb silently breaks.

    ``ECGDataset._read_csv`` is the single place every fold-CSV read goes through
    for exactly this reason. Read with pandas' default inference, afdb's record
    ``00735`` comes back as 735, ``__getitem__`` builds ``data_path / "735"``, and
    the failure surfaces as a file-not-found naming a record that does not exist.
    """

    def _folds_csv(self, path):
        path.write_text(
            "record_name,signal_path,fold,default_split\n"
            "00735,00735,5,train\n"
            "03665,03665,1,train\n"
            "04015,04015,9,val\n",
            encoding="utf-8",
        )
        return path

    def test_read_csv_keeps_leading_zeros(self, tmp_path):
        from ecgbench.config import load_config
        from ecgbench.dataset import ECGDataset

        ds = ECGDataset.__new__(ECGDataset)
        ds.config = load_config("afdb")

        df = ds._read_csv(self._folds_csv(tmp_path / "folds.csv"))
        assert list(df["record_name"]) == ["00735", "03665", "04015"]
        assert list(df["signal_path"]) == ["00735", "03665", "04015"]
        # fold stays numeric, so _filter_master's int comparison still works.
        assert df["fold"].tolist() == [5, 1, 9]

    def test_the_local_fold_reader_uses_it_too(self, tmp_path):
        """Both fold-CSV paths must agree about what a record is called."""
        from ecgbench.config import load_config
        from ecgbench.dataset import ECGDataset

        split_dir = tmp_path / "clean" / "train"
        split_dir.mkdir(parents=True)
        self._folds_csv(split_dir / "fold_5.csv")

        ds = ECGDataset.__new__(ECGDataset)
        ds.config = load_config("afdb")
        ds.split = "train"

        df = ds._read_fold_csvs(split_dir, [5])
        assert list(df["record_name"]) == ["00735", "03665", "04015"]

    def test_an_opted_in_config_protects_every_identifier_column(self):
        """Opting in must cover the paths too, not only the record id.

        Protecting the id alone would leave ``data_path / "735"`` unresolvable
        while the id looked right — the worst of both.
        """
        from ecgbench.config import load_config

        config = load_config("afdb")
        dtypes = config.identifier_dtypes()
        assert config.record_id_column in dtypes
        assert all(column in dtypes for column in config.signal_path_columns.values())
        assert all(value == "str" for value in dtypes.values())

    def test_export_refuses_a_zero_padded_id_from_a_config_that_did_not_opt_in(self):
        """The guard that makes the flag impossible to forget.

        Checked at export because after the CSV round-trip the leading zeros are
        gone and no consumer can tell they were ever there.
        """
        from dataclasses import replace

        from ecgbench.config import load_config
        from ecgbench.splitting.export import _check_zero_padded_identifiers

        config = replace(load_config("afdb"), zero_padded_identifiers=False)
        df = pd.DataFrame({
            "record_name": ["00735", "04015"],
            "signal_path": ["00735", "04015"],
        })

        with pytest.raises(ValueError, match="zero_padded_identifiers"):
            _check_zero_padded_identifiers(df, config)

        # Opted in, the same frame is fine.
        _check_zero_padded_identifiers(df, load_config("afdb"))

    def test_ltafdb_ids_lose_far_more_than_afdb_ids_do(self):
        """"00" becomes 0 — a shorter id, so an easier mistake to miss.

        afdb's "00735" at least still looks like a record name after the round
        trip. ltafdb's seven zero-prefixed records collapse to single digits that
        collide with nothing and resolve to nothing, and the export guard is what
        stops a config edit reintroducing that.
        """
        import pandas as pd

        from ecgbench.config import load_config
        from ecgbench.splitting.export import _check_zero_padded_identifiers

        config = load_config("ltafdb")
        assert config.identifier_dtypes() == {
            "record_name": "str",
            "signal_path": "str",
        }

        df = pd.DataFrame({
            "record_name": ["00", "01", "08", "122"],
            "signal_path": ["00", "01", "08", "122"],
        })
        _check_zero_padded_identifiers(df, config)  # opted in: fine

        from dataclasses import replace

        with pytest.raises(ValueError, match="zero_padded_identifiers"):
            _check_zero_padded_identifiers(
                df, replace(config, zero_padded_identifiers=False)
            )

    def test_the_guard_ignores_ids_that_merely_look_numeric(self):
        """mitdb's "100" and ptbxl's "1" are not zero-padded and must not trip it."""
        from ecgbench.config import load_config
        from ecgbench.splitting.export import _check_zero_padded_identifiers

        config = load_config("mitdb")
        df = pd.DataFrame({
            "record_name": ["100", "114", "234"],
            "patient_id": ["tape1085", "tape750", "tape1960"],
            "signal_path": ["100", "114", "234"],
        })
        _check_zero_padded_identifiers(df, config)  # must not raise


class TestECGIDDBChannels:
    """ECG-ID stores ONE lead twice, and the two artefacts disagree on purpose."""

    def test_both_channels_are_lead_i_raw_and_filtered(self):
        """All 310 headers name them "ECG I" and "ECG I filtered", raw first.

        This cannot go in ``TestShippedLeadNames``, which assumes the first six
        names are I/II/III then the augmented leads. The point here is that
        ``config.leads == 2`` is a channel count over a single electrode pair: the
        second channel is the thesis's own preprocessing of the first, not a second
        placement. A model given both rows is given Lead I twice.
        """
        from ecgbench.config import load_config

        config = load_config("ecgiddb")
        assert config.leads == 2
        assert config.lead_names == ["ECG I", "ECG I filtered"]
        assert len(config.lead_names) == config.leads
        # Same lead, two processings — so the names share a prefix rather than
        # naming different anatomy, and neither is a standard lead spelling.
        assert config.lead_names[1].startswith(config.lead_names[0])
        assert config.validation is not None
        assert config.validation.expected_leads == 2
        # No per-record variation, unlike mitdb (record_lead_layouts) or zzu_pecg
        # (alternate_lead_names).
        assert config.record_lead_layouts is None
        assert config.alternate_lead_names is None

    def test_the_catalogue_says_one_lead_and_the_config_says_two(self):
        """A deliberate disagreement, pinned so nobody "fixes" one to match.

        ``docs/_datasets/ecg-id-database.md`` sets ``leads: 1`` because ECG-ID is a
        one-lead database — Lead I, from limb clamps — and it sits in the
        ``one-lead`` catalogue category. The config sets ``leads: 2`` because that
        is the shape of the tensor ``ECGDataset`` returns. Both are true of
        different things, and the catalogue's ``format`` string is what reconciles
        them for a reader.
        """
        from ecgbench import get_dataset
        from ecgbench.config import load_config

        entry = get_dataset("ECG-ID Database")
        assert entry.leads == 1
        assert load_config("ecgiddb").leads == 2
        # The format string has to carry the reconciliation, or the "1" is a lie.
        assert "raw" in entry.format and "filtered" in entry.format


class TestOpenSignalsReader:
    """The PLUX/BITalino text export, added for tOLIet.

    Three things this format does that no other ECGBench reader does: it names its
    columns in a JSON header line, it stores channels the dataset does not want
    (a sequence number, digital I/O, and analog inputs at a different bit depth),
    and it returns **fractions of full scale rather than millivolts** — the volts
    per full scale is a property of the sensor, not the file, so it lives in
    ``signal_unit_scale``.
    """

    #: The real tOLIet preamble, with A5/A6 at 6 bits like a BITalino board, so a
    #: reader that assumed one bit depth for the whole row would be wrong.
    HEADER = (
        "# OpenSignals Text File Format. Version 1\n"
        '# {"": {"sampling rate": 1000, '
        '"resolution": [4, 1, 1, 1, 1, 10, 10, 10, 10, 6, 6], '
        '"label": ["A1", "A2", "A3", "A4", "A5", "A6"], '
        '"column": ["nSeq", "I1", "I2", "O1", "O2", "A1", "A2", "A3", "A4", '
        '"A5", "A6"]}}\n'
        "# EndOfHeader\n"
    )

    def _write(self, path, n=50):
        """One file where every column is identifiable: column j holds (j+1)*100 + i."""
        rows = []
        for i in range(n):
            rows.append("\t".join(str((j + 1) * 100 + i) for j in range(11)) + "\t\r")
        path.write_text(self.HEADER + "\n".join(rows) + "\n", encoding="utf-8")
        return path

    def test_channels_are_selected_by_name_from_the_path(self, tmp_path):
        """``<file>.txt:A1,A3`` reads those two columns, in that order."""
        from ecgbench.dataset import _load_signal

        path = self._write(tmp_path / "1.txt")
        signal = _load_signal(f"{path}:A1,A3", "opensignals")
        assert signal.shape == (2, 50)
        # A1 is column 5 (600 + i) and A3 column 7 (800 + i), each over 2**10.
        assert signal[0][0] == pytest.approx(600 / 1024 - 0.5)
        assert signal[1][0] == pytest.approx(800 / 1024 - 0.5)

    def test_requested_order_is_honoured_not_file_order(self, tmp_path):
        """read_csv returns usecols in file order; the reader must reorder."""
        from ecgbench.dataset import _load_signal

        path = self._write(tmp_path / "1.txt")
        forward = _load_signal(f"{path}:A1,A4", "opensignals")
        reversed_ = _load_signal(f"{path}:A4,A1", "opensignals")
        assert np.allclose(reversed_[0], forward[1])
        assert np.allclose(reversed_[1], forward[0])

    def test_bit_depth_is_per_column(self, tmp_path):
        """A5 is 6-bit on the same rows as the 10-bit A1, per the header."""
        from ecgbench.dataset import _load_signal

        path = self._write(tmp_path / "1.txt")
        signal = _load_signal(f"{path}:A1,A5", "opensignals")
        assert signal[0][0] == pytest.approx(600 / 1024 - 0.5)
        assert signal[1][0] == pytest.approx(1000 / 64 - 0.5)

    def test_unit_scale_is_applied_and_may_be_negative(self, tmp_path):
        """tOLIet's -3.0 both scales to millivolts and inverts the polarity."""
        from ecgbench.dataset import _load_signal

        path = self._write(tmp_path / "1.txt")
        plain = _load_signal(f"{path}:A1", "opensignals")
        scaled = _load_signal(f"{path}:A1", "opensignals", -3.0)
        assert np.allclose(scaled, plain * -3.0)
        # And it reproduces the release's own read_ecg_data.py formula exactly.
        raw = np.array([600 + i for i in range(50)], dtype=np.float64)
        assert np.allclose(scaled[0], ((1024 - raw) / 1024 - 0.5) * (33 / 11))

    def test_default_is_every_channel_the_header_labels(self, tmp_path):
        """No ':' suffix means the six analog channels, in header order."""
        from ecgbench.dataset import _load_signal

        path = self._write(tmp_path / "1.txt")
        assert _load_signal(str(path), "opensignals").shape == (6, 50)

    def test_window_is_pushed_into_the_reader(self, tmp_path):
        """(start, length) becomes skiprows/nrows, so it decodes only its samples."""
        from ecgbench.dataset import _load_signal

        path = self._write(tmp_path / "1.txt")
        whole = _load_signal(f"{path}:A1", "opensignals")
        assert np.allclose(_load_signal(f"{path}:A1", "opensignals", 1.0, (10, 5)),
                           whole[:, 10:15])
        # length=None means "to the end".
        assert np.allclose(_load_signal(f"{path}:A1", "opensignals", 1.0, (40, None)),
                           whole[:, 40:])

    def test_a_window_past_the_end_names_the_record_and_its_length(self, tmp_path):
        """numpy/pandas return a short array rather than raising; we must not."""
        from ecgbench.dataset import WindowOutOfRangeError, _load_signal, _record_length

        path = self._write(tmp_path / "1.txt")
        with pytest.raises(WindowOutOfRangeError, match="1.txt"):
            _load_signal(f"{path}:A1", "opensignals", 1.0, (48, 10))
        with pytest.raises(WindowOutOfRangeError):
            _load_signal(f"{path}:A1", "opensignals", 1.0, (60, 5))
        # The length in that message comes from counting data lines.
        assert _record_length(f"{path}:A1", "opensignals") == 50

    def test_an_unknown_channel_lists_what_the_file_has(self, tmp_path):
        from ecgbench.dataset import _load_signal

        path = self._write(tmp_path / "1.txt")
        with pytest.raises(ValueError, match=r"A9.*nSeq"):
            _load_signal(f"{path}:A9", "opensignals")

    def test_a_file_with_no_json_header_says_what_the_format_is(self, tmp_path):
        from ecgbench.dataset import _load_signal

        path = tmp_path / "1.txt"
        path.write_text("1\t2\t3\n", encoding="utf-8")
        with pytest.raises(ValueError, match="OpenSignals"):
            _load_signal(f"{path}:A1", "opensignals")

    def test_a_multi_device_export_is_refused_rather_than_half_read(self, tmp_path):
        """Two devices continue the same rows, so column indices would be wrong."""
        from ecgbench.dataset import _load_signal

        path = tmp_path / "1.txt"
        path.write_text(
            "# OpenSignals Text File Format. Version 1\n"
            '# {"AA": {"resolution": [10], "column": ["A1"], "label": ["A1"]}, '
            '"BB": {"resolution": [10], "column": ["A1"], "label": ["A1"]}}\n'
            "# EndOfHeader\n1\t2\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="2 devices"):
            _load_signal(f"{path}:A1", "opensignals")

    def test_the_validation_engine_reads_it_the_same_way(self, tmp_path):
        """engine.py keeps its own window-less copy of _load_signal; it must agree."""
        from ecgbench.dataset import _load_signal
        from ecgbench.validation.engine import _load_signal as _validation_load

        path = self._write(tmp_path / "1.txt")
        assert np.allclose(
            _validation_load(f"{path}:A1,A2", "opensignals", -3.0),
            _load_signal(f"{path}:A1,A2", "opensignals", -3.0),
        )

    def test_a_path_without_a_recognised_extension_keeps_its_colon(self, tmp_path):
        """A directory named with a colon must not be read as a channel list."""
        from ecgbench.dataset import _parse_opensignals_ref

        assert _parse_opensignals_ref("a/b.txt:A1,A2") == ("a/b.txt", ["A1", "A2"])
        assert _parse_opensignals_ref("a:b/c.txt") == ("a:b/c.txt", None)
        assert _parse_opensignals_ref("plain.txt") == ("plain.txt", None)
