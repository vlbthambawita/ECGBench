"""Tests for the config system."""

import pytest

from ecgbench.config import DatasetConfig, list_available_configs, load_config


def test_load_ptbxl_config():
    """Test loading ptbxl.yaml produces valid DatasetConfig."""
    config = load_config("ptbxl")
    assert isinstance(config, DatasetConfig)
    assert config.name == "PTB-XL"
    assert config.slug == "ptbxl"
    assert config.version == "1.0.3"
    assert config.leads == 12
    assert 500 in config.sampling_rates
    assert 100 in config.sampling_rates
    assert config.has_predefined_splits is True
    assert config.predefined_splits is not None
    assert config.predefined_splits.column == "strat_fold"
    assert config.predefined_splits.fold_mapping["train"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert config.predefined_splits.fold_mapping["val"] == [9]
    assert config.predefined_splits.fold_mapping["test"] == [10]


def test_load_chapman_config():
    """Test loading chapman_shaoxing.yaml produces valid DatasetConfig."""
    config = load_config("chapman_shaoxing")
    assert config.name == "Chapman-Shaoxing"
    assert config.slug == "chapman_shaoxing"
    assert config.has_predefined_splits is False
    # figshare release: CSV signals in microvolts, one record per patient.
    assert config.signal_format == "csv"
    assert config.signal_unit_scale == 0.001
    assert config.patient_id_column is None
    assert config.record_id_column == "FileName"
    assert config.metadata_csv == "ecgbench_metadata.csv"
    assert config.signal_path_columns == {500: "signal_path"}
    assert config.stratification is not None
    assert config.stratification.method == "direct"


def test_load_ecg_arrhythmia_config():
    """Test loading ecg_arrhythmia.yaml produces valid DatasetConfig."""
    config = load_config("ecg_arrhythmia")
    assert config.slug == "ecg_arrhythmia"
    assert config.version == "1.0.0"
    assert config.signal_format == "wfdb"
    assert config.sampling_rates == [500]
    # No metadata CSV ships with the dataset — the splitter generates this one.
    assert config.metadata_csv == "ecgbench_metadata.csv"
    assert config.record_id_column == "record_name"
    assert config.signal_path_columns == {500: "signal_path"}
    # One record per patient, so no grouping column and no predefined folds.
    assert config.patient_id_column is None
    assert config.has_predefined_splits is False
    assert config.stratification is not None
    assert config.stratification.method == "custom_function"
    assert config.validation is not None
    assert config.validation.expected_samples[500] == 5000


def test_load_mimic_iv_ecg_demo_config():
    """Test loading mimic_iv_ecg_demo.yaml produces valid DatasetConfig."""
    config = load_config("mimic_iv_ecg_demo")
    assert config.slug == "mimic_iv_ecg_demo"
    assert config.version == "0.1"
    assert config.signal_format == "wfdb"
    assert config.metadata_csv == "record_list.csv"
    assert config.record_id_column == "study_id"
    # 659 records from 92 subjects — grouping is the point of this config.
    assert config.patient_id_column == "subject_id"
    assert config.signal_path_columns == {500: "path"}
    assert config.has_predefined_splits is False
    assert config.stratification is not None
    assert config.stratification.method == "custom_function"
    assert config.validation is not None
    assert config.validation.expected_samples[500] == 5000


def test_load_ptbdb_config():
    """PTBDB is the 15-lead, variable-length, no-metadata-file dataset."""
    config = load_config("ptbdb")
    assert config.slug == "ptbdb"
    assert config.version == "1.0.0"
    assert config.signal_format == "wfdb"
    assert config.signal_unit_scale == 1.0
    assert config.leads == 15
    assert config.lead_names[-3:] == ["vx", "vy", "vz"]  # Frank leads
    assert config.sampling_rates == [1000]
    # No metadata ships: PTBDBSplitter generates this from the .hea comments.
    assert config.metadata_csv == "ecgbench_metadata.csv"
    assert config.record_id_column == "record_name"
    # 113 of 290 patients have more than one recording, so grouping is required.
    assert config.patient_id_column == "patient_id"
    assert config.signal_path_columns == {1000: "signal_path"}
    # Records are genuinely variable length; an entry here would fail most of them.
    assert config.validation.expected_samples == {}
    assert config.validation.amplitude_range_mv == (-15.0, 15.0)


def test_load_ludb_config():
    """LUDB: 200 delineation-annotated records, lowercase lead names."""
    config = load_config("ludb")
    assert config.slug == "ludb"
    assert config.version == "1.0.1"
    assert config.signal_format == "wfdb"
    assert config.signal_unit_scale == 1.0
    assert config.leads == 12
    assert config.lead_names == ["i", "ii", "iii", "avr", "avl", "avf",
                                 "v1", "v2", "v3", "v4", "v5", "v6"]
    assert config.record_id_column == "ID"
    assert config.patient_id_column is None  # one record per patient
    assert config.metadata_csv == "ecgbench_metadata.csv"
    assert config.validation.expected_samples[500] == 5000


def test_list_available_configs():
    """Test list_available_configs returns expected slugs."""
    slugs = list_available_configs()
    assert "ptbxl" in slugs
    assert "chapman_shaoxing" in slugs
    assert "ecg_arrhythmia" in slugs
    assert "mimic_iv_ecg_demo" in slugs
    assert "ptbdb" in slugs
    assert "ludb" in slugs
    # Template should not be listed (starts with _)
    assert "_template" not in slugs


def test_missing_config_raises():
    """Test that loading a nonexistent config raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Config not found"):
        load_config("nonexistent_dataset")


def test_config_validation_fields():
    """Test that validation config is parsed correctly."""
    config = load_config("ptbxl")
    assert config.validation is not None
    assert config.validation.expected_leads == 12
    assert config.validation.expected_samples[500] == 5000
    assert config.validation.expected_samples[100] == 1000
    assert "missing_leads" in config.validation.checks
    assert config.validation.amplitude_range_mv == (-10.0, 10.0)


def test_config_signal_path_columns():
    """Test signal_path_columns parsing with int keys."""
    config = load_config("ptbxl")
    assert isinstance(config.signal_path_columns, dict)
    assert 500 in config.signal_path_columns
    assert 100 in config.signal_path_columns
    assert config.signal_path_columns[500] == "filename_hr"
    assert config.signal_path_columns[100] == "filename_lr"


def test_config_creators():
    """Test creators list is parsed correctly."""
    config = load_config("ptbxl")
    assert len(config.creators) > 0
    assert config.creators[0].type == "Organization"
    assert "PTB" in config.creators[0].name


def test_load_challenge2021_config():
    """Challenge 2021: the eight-cohort meta-dataset with per-record sampling rates."""
    config = load_config("challenge2021")
    assert config.slug == "challenge2021"
    assert config.version == "1.0.3"
    assert config.signal_format == "wfdb"
    # Every header declares a gain of 1000/mV, so wfdb already yields millivolts.
    assert config.signal_unit_scale == 1.0
    assert config.leads == 12
    # Standard order and standard capitalisation in all 88,253 records.
    assert config.lead_names == ["I", "II", "III", "aVR", "aVL", "aVF",
                                "V1", "V2", "V3", "V4", "V5", "V6"]
    # All three rates are declared, but each record exists at exactly ONE of
    # them, so there is a single path column keyed on the nominal 500 Hz. Adding
    # a key per rate would emit duplicate columns in the exported fold CSVs.
    assert config.sampling_rates == [257, 500, 1000]
    assert config.default_sampling_rate == 500
    assert config.signal_path_columns == {500: "signal_path"}
    # No metadata ships: Challenge2021Splitter generates this from the headers.
    assert config.metadata_csv == "ecgbench_metadata.csv"
    assert config.record_id_column == "record_name"
    # No patient identifiers are published anywhere in this release.
    assert config.patient_id_column is None
    # Records run from 5 s to 1800 s, so any entry here would fail thousands.
    assert config.validation.expected_samples == {}
    assert config.validation.amplitude_range_mv == (-10.0, 10.0)
    assert config.has_predefined_splits is False


def test_load_incartdb_config():
    """INCART: 30-minute Holter records, per-record gains, mandatory patient grouping."""
    config = load_config("incartdb")
    assert config.slug == "incartdb"
    assert config.version == "1.0.0"
    assert config.signal_format == "wfdb"
    # Gains vary between records (240-1063); wfdb applies each header's own gain,
    # so no extra scaling reaches ECGBench.
    assert config.signal_unit_scale == 1.0
    assert config.leads == 12
    # Uppercase AVR/AVL/AVF, as PTB-XL spells them — not Chapman's aVR.
    assert config.lead_names == ["I", "II", "III", "AVR", "AVL", "AVF",
                                "V1", "V2", "V3", "V4", "V5", "V6"]
    assert config.sampling_rates == [257]
    assert config.duration_seconds == 1800
    assert config.signal_path_columns == {257: "signal_path"}
    # No metadata ships: INCARTDBSplitter generates this from headers + .atr files.
    assert config.metadata_csv == "ecgbench_metadata.csv"
    assert config.record_id_column == "record_name"
    # 30 of 32 patients contributed more than one record — grouping is mandatory.
    assert config.patient_id_column == "patient_id"
    # Unlike ptbdb/challenge2021, length is uniform here, so truncation is checked.
    assert config.validation.expected_samples == {257: 462600}
    # Widened from the usual +/-10 deliberately: at +/-10 this check would drop 29
    # of 75 records for amplitudes that are ordinary in 30-minute Holter data.
    assert config.validation.amplitude_range_mv == (-30.0, 30.0)
