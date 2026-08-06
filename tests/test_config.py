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


def test_load_norwegian_athlete_ecg_config():
    """Norwegian athletes: 28 header-labelled records, uncalibrated amplitudes."""
    config = load_config("norwegian_athlete_ecg")
    assert config.slug == "norwegian_athlete_ecg"
    assert config.version == "1.0.0"
    assert config.signal_format == "wfdb"
    # wfdb applies the header's nominal 50000/mV gain; there is no scale factor
    # that could undo the per-lead min-max normalisation, so this stays 1.0.
    assert config.signal_unit_scale == 1.0
    assert config.leads == 12
    assert config.lead_names == ["I", "II", "III", "AVR", "AVL", "AVF",
                                 "V1", "V2", "V3", "V4", "V5", "V6"]
    assert config.record_id_column == "record_name"
    assert config.patient_id_column is None  # 28 records, 28 athletes
    assert config.metadata_csv == "ecgbench_metadata.csv"  # generated from headers
    assert config.validation.expected_samples[500] == 5000
    # Tight on purpose: per-lead normalisation pins every lead to +/-0.6553 mV,
    # so the usual [-10, 10] could never fire on this dataset.
    assert config.validation.amplitude_range_mv == (-1.0, 1.0)
    # Labels live in the .hea comments, not a CSV.
    assert config.labels is not None
    assert config.labels.available is True
    assert config.labels.source_csv is None
    assert config.labels.join_column == "record_name"
    # CC BY 4.0, so the fold CSVs are publishable.
    assert config.publish_fold_csvs is True


def test_load_mhd_effect_ecg_mri_config():
    """MHD effect: mixed 12/3-lead, variable length, derived subject grouping."""
    config = load_config("mhd_effect_ecg_mri")
    assert config.slug == "mhd_effect_ecg_mri"
    assert config.version == "1.0.0"
    assert config.signal_format == "wfdb"
    # Per-channel, per-record gains and baselines; wfdb applies both.
    assert config.signal_unit_scale == 1.0
    assert config.leads == 12
    assert config.lead_names == ["I", "II", "III", "aVR", "aVL", "aVF",
                                 "V1", "V2", "V3", "V4", "V5", "V6"]
    assert config.sampling_rates == [1024]
    assert config.default_sampling_rate == 1024
    assert config.record_id_column == "record_name"
    # Derived from demographics, NOT the filename's per-scanner subject number.
    assert config.patient_id_column == "subject_key"
    assert config.metadata_csv == "ecgbench_metadata.csv"  # generated from headers
    # Length varies 24.4 s to 722.7 s, so the truncation check must stay disabled.
    assert config.validation.expected_samples == {}
    # MHD distortion reaches -31 mV; +/-10 would exclude 16 of 53 real records.
    assert config.validation.amplitude_range_mv == (-35.0, 35.0)
    assert config.labels is not None
    assert config.labels.available is True
    assert config.labels.source_csv is None  # headers + .qrs files
    assert config.labels.join_column == "record_name"
    # ODC-By, so the fold CSVs are publishable.
    assert config.publish_fold_csvs is True


def test_load_wctecgdb_config():
    """Wilson Central Terminal: 37 channels at 800 Hz, patient-grouped segments."""
    config = load_config("wctecgdb")
    assert config.slug == "wctecgdb"
    assert config.version == "1.0.1"
    assert config.signal_format == "wfdb"
    # The per-channel header gains are genuine calibration and wfdb applies them,
    # so records come back as physiologic millivolts.
    assert config.signal_unit_scale == 1.0
    # A channel count, not a 12-lead set: 18 raw + 18 filtered + WCT. aVR/aVL/aVF
    # are absent from the release entirely.
    assert config.leads == 37
    assert config.lead_names is not None
    assert len(config.lead_names) == 37
    assert config.lead_names[:3] == ["I-Raw", "II-Raw", "III-Raw"]
    assert config.lead_names[18:21] == ["I", "II", "III"]
    assert config.lead_names[-1] == "WCT"
    assert "aVR" not in config.lead_names and "AVR" not in config.lead_names
    # 800 Hz, and 8001 samples is 10.00125 s — duration_seconds is the nominal 10.
    assert config.sampling_rates == [800]
    assert config.default_sampling_rate == 800
    assert config.duration_seconds == 10
    assert config.validation.expected_samples == {800: 8001}
    assert config.validation.expected_leads == 37
    # Raw limb/unipolar channels are unreferenced potentials with a DC offset of
    # several mV, so the usual [-10, 10] would flag most records.
    assert config.validation.amplitude_range_mv == (-20.0, 20.0)
    # seg01 repeats in all 92 patient directories, so the id flattens the path.
    assert config.record_id_column == "record_name"
    assert config.patient_id_column == "patient_id"
    assert config.signal_path_columns == {800: "signal_path"}
    assert config.metadata_csv == "ecgbench_metadata.csv"  # generated from headers
    # Labels live in the .hea comments, not a CSV.
    assert config.labels is not None
    assert config.labels.available is True
    assert config.labels.source_csv is None
    assert config.labels.join_column == "record_name"
    assert config.label_column == "diagnosis_group"
    # ODC-By, so the fold CSVs are publishable.
    assert config.publish_fold_csvs is True


def test_load_ecgcipa_config():
    """CiPA: microvolt samples at 1 kHz, and treatment instead of a diagnosis."""
    config = load_config("ecgcipa")
    assert config.slug == "ecgcipa"
    assert config.version == "1.0.0"
    assert config.signal_format == "wfdb"
    # The headers declare gain 0.26595744680851063 with unit /uV, so wfdb returns
    # MICROVOLTS. Left at 1.0 every record's peak reads ~2000 "mV" and
    # amplitude_outlier fires on all 5,749.
    assert config.signal_unit_scale == 0.001
    assert config.leads == 12
    # Standard order, lowercase 'a' — the derived medians/ headers spell the same
    # three leads AVR/AVL/AVF, so the two directories disagree with each other.
    assert config.lead_names == [
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6",
    ]
    assert config.sampling_rates == [1000]
    assert config.default_sampling_rate == 1000
    assert config.duration_seconds == 10
    assert config.validation.expected_samples == {1000: 10000}
    assert config.validation.expected_leads == 12
    # The house default, and it excludes exactly 2 electrode-artefact records.
    assert config.validation.amplitude_range_mv == (-10.0, 10.0)
    # EGREFID UUIDs, unique across the release; subjects 1001-1050 and 2001-2010.
    assert config.record_id_column == "record_id"
    assert config.patient_id_column == "patient_id"
    assert config.signal_path_columns == {1000: "signal_path"}
    # No shipped record-to-file table; the splitter generates one.
    assert config.metadata_csv == "ecgbench_metadata.csv"
    # Labels are assembled from four analysis datasets by a module; source_csv
    # names the one the loader cannot work without.
    assert config.labels is not None
    assert config.labels.available is True
    assert config.labels.source_csv == "adeg.csv"
    assert config.labels.join_column == "EGREFID"
    # A pharmacology dataset: the closest thing to a class is the drug.
    assert config.label_column == "treatment"
    # ODC-By, so the fold CSVs are publishable.
    assert config.publish_fold_csvs is True


def test_load_ecgdmmld_config():
    """ECGDMMLD: millivolt samples at 1 kHz, and a crossover treatment arm."""
    config = load_config("ecgdmmld")
    assert config.slug == "ecgdmmld"
    assert config.version == "1.0.0"
    assert config.signal_format == "wfdb"
    # 1.0, NOT the sibling ecgcipa's 0.001. Every channel declares its own gain
    # against unit /mV, so wfdb already returns millivolts. Copying ecgcipa's
    # value would divide every sample by 1000 and amplitude_outlier would never
    # fire again.
    assert config.signal_unit_scale == 1.0
    assert config.leads == 12
    # Uppercase 'A', like ptbxl — and the opposite of ecgcipa's raw/ headers,
    # which spell the same three leads aVR/aVL/aVF.
    assert config.lead_names == [
        "I", "II", "III", "AVR", "AVL", "AVF",
        "V1", "V2", "V3", "V4", "V5", "V6",
    ]
    assert config.sampling_rates == [1000]
    assert config.default_sampling_rate == 1000
    assert config.duration_seconds == 10
    assert config.validation.expected_samples == {1000: 10000}
    assert config.validation.expected_leads == 12
    # The house default, and it excludes exactly 2 electrode-artefact records.
    assert config.validation.amplitude_range_mv == (-10.0, 10.0)
    # EGREFID UUIDs, unique across the release; subjects 2001-2022.
    assert config.record_id_column == "record_id"
    assert config.patient_id_column == "patient_id"
    assert config.signal_path_columns == {1000: "signal_path"}
    # The shipped clinical table has no signal-path column, so the splitter
    # generates one.
    assert config.metadata_csv == "ecgbench_metadata.csv"
    assert config.labels is not None
    assert config.labels.available is True
    assert config.labels.source_csv == "SCR-003.Clinical.Data.csv"
    assert config.labels.join_column == "EGREFID"
    # A pharmacology dataset: the closest thing to a class is the treatment arm.
    assert config.label_column == "treatment"
    # ODC-By, so the fold CSVs are publishable.
    assert config.publish_fold_csvs is True


def test_load_ecgrdvq_config():
    """ECGRDVQ: millivolt samples at 1 kHz, and a single-agent crossover treatment."""
    config = load_config("ecgrdvq")
    assert config.slug == "ecgrdvq"
    assert config.version == "1.0.0"
    assert config.signal_format == "wfdb"
    # 1.0, like the sibling ecgdmmld and NOT ecgcipa's 0.001. Every channel
    # declares its own gain against unit /mV, so wfdb already returns millivolts.
    assert config.signal_unit_scale == 1.0
    assert config.leads == 12
    # Uppercase 'A', like ptbxl and ecgdmmld — and the opposite of ecgcipa's raw/
    # headers, which spell the same three leads aVR/aVL/aVF.
    assert config.lead_names == [
        "I", "II", "III", "AVR", "AVL", "AVF",
        "V1", "V2", "V3", "V4", "V5", "V6",
    ]
    assert config.sampling_rates == [1000]
    assert config.default_sampling_rate == 1000
    assert config.duration_seconds == 10
    assert config.validation.expected_samples == {1000: 10000}
    assert config.validation.expected_leads == 12
    # The house default. Unlike ecgdmmld it excludes nothing — the release peaks at
    # 6.98 mV — and flat_line is the check that fires instead, on the 2 dead-V4
    # records.
    assert config.validation.amplitude_range_mv == (-10.0, 10.0)
    # EGREFID UUIDs, unique across the release; subjects 1001-1022.
    assert config.record_id_column == "record_id"
    assert config.patient_id_column == "patient_id"
    assert config.signal_path_columns == {1000: "signal_path"}
    # The shipped clinical table has no signal-path column, so the splitter
    # generates one.
    assert config.metadata_csv == "ecgbench_metadata.csv"
    assert config.labels is not None
    assert config.labels.available is True
    # SCR-002 — the first of the three FDA studies, ahead of ecgdmmld's SCR-003.
    assert config.labels.source_csv == "SCR-002.Clinical.Data.csv"
    assert config.labels.join_column == "EGREFID"
    # A pharmacology dataset: the closest thing to a class is the treatment.
    assert config.label_column == "treatment"
    # ODC-By, so the fold CSVs are publishable.
    assert config.publish_fold_csvs is True


def test_list_available_configs():
    """Test list_available_configs returns expected slugs."""
    slugs = list_available_configs()
    assert "ptbxl" in slugs
    assert "chapman_shaoxing" in slugs
    assert "ecg_arrhythmia" in slugs
    assert "mimic_iv_ecg_demo" in slugs
    assert "ptbdb" in slugs
    assert "ludb" in slugs
    assert "norwegian_athlete_ecg" in slugs
    assert "mhd_effect_ecg_mri" in slugs
    assert "wctecgdb" in slugs
    assert "ecgcipa" in slugs
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


def test_load_challenge2020_config():
    """Challenge 2020: the six-cohort meta-dataset that Challenge 2021 contains."""
    config = load_config("challenge2020")
    assert config.slug == "challenge2020"
    assert config.version == "1.0.2"
    assert config.signal_format == "wfdb"
    # Every header declares a gain of 1000/mV, so wfdb already yields millivolts.
    assert config.signal_unit_scale == 1.0
    assert config.leads == 12
    # Standard order and standard capitalisation in all 43,101 records.
    assert config.lead_names == ["I", "II", "III", "aVR", "aVL", "aVF",
                                "V1", "V2", "V3", "V4", "V5", "V6"]
    # All three rates are declared, but each record exists at exactly ONE of
    # them, so there is a single path column keyed on the nominal 500 Hz.
    assert config.sampling_rates == [257, 500, 1000]
    assert config.default_sampling_rate == 500
    assert config.signal_path_columns == {500: "signal_path"}
    # No metadata ships: Challenge2020Splitter generates this from the headers.
    assert config.metadata_csv == "ecgbench_metadata.csv"
    assert config.record_id_column == "record_name"
    # No patient identifiers are published anywhere in this release.
    assert config.patient_id_column is None
    # Records run from 5 s to 1800 s, so any entry here would fail thousands.
    assert config.validation.expected_samples == {}
    assert config.validation.amplitude_range_mv == (-10.0, 10.0)
    assert config.has_predefined_splits is False
    # CC BY 4.0, so the fold CSVs are published like Challenge 2021's.
    assert config.publish_fold_csvs is True


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


def test_load_leipzig_heart_center_ecg_config():
    """Leipzig: surface ECG + intracardiac channels, variable length AND variable
    channel count — the one dataset here where lead_names is a strict subset of the
    channels a record holds."""
    config = load_config("leipzig_heart_center_ecg")
    assert config.slug == "leipzig_heart_center_ecg"
    assert config.version == "1.0.0"
    assert config.signal_format == "wfdb"
    # Every channel of every record declares 2000.0(0)/mV, so wfdb yields mV.
    assert config.signal_unit_scale == 1.0
    # The 12 SURFACE ECG channels only. Records carry 14, 18, 19 or 20 channels in
    # six layouts, and only indices 0-11 are the same channel in every record —
    # index 12 is ABL12, RVA12 or ART depending on the record. Declaring an
    # intracardiac order here would make ECGDataset(leads=...) return the wrong
    # physical channel.
    assert config.leads == 12
    assert config.lead_names == ["I", "II", "III", "aVR", "aVL", "aVF",
                                 "V1", "V2", "V3", "V4", "V5", "V6"]
    assert config.sampling_rates == [977]
    assert config.default_sampling_rate == 977
    assert config.signal_path_columns == {977: "signal_path"}
    # Two per-subject CSVs ship with different columns; the splitter joins them.
    assert config.metadata_csv == "ecgbench_metadata.csv"
    assert config.record_id_column == "record_name"
    # 39 records from 39 subjects, one each: nothing to group by.
    assert config.patient_id_column is None
    # Length runs 75,873 to 8,824,019 samples, so truncation cannot be checked.
    assert config.validation.expected_samples == {}
    # NOT a physiologic bound: at +/-10 mV, 24 of the 39 records fail because their
    # intracardiac channels legitimately reach +/-51 mV. See the config comment.
    assert config.validation.amplitude_range_mv == (-52.0, 52.0)
    # Open licence, so the fold CSVs are published.
    assert config.publish_fold_csvs is True


def test_load_mimic_iv_ecg_config():
    """MIMIC-IV-ECG: the 800k-record credentialed release, aVF/aVL transposed."""
    config = load_config("mimic_iv_ecg")
    assert config.slug == "mimic_iv_ecg"
    assert config.version == "1.0"
    assert config.signal_format == "wfdb"
    # Header gain is 200/mV, so wfdb already yields millivolts.
    assert config.signal_unit_scale == 1.0
    assert config.leads == 12
    # aVF BEFORE aVL — the reason leads= takes names rather than indices.
    assert config.lead_names == ["I", "II", "III", "aVR", "aVF", "aVL",
                                "V1", "V2", "V3", "V4", "V5", "V6"]
    assert config.lead_names.index("aVF") < config.lead_names.index("aVL")
    assert config.sampling_rates == [500]
    # record_list.csv ships usable as-is; no generated metadata CSV here.
    assert config.metadata_csv == "record_list.csv"
    assert config.record_id_column == "study_id"
    # 64.5% of subjects have more than one study — grouping is mandatory.
    assert config.patient_id_column == "subject_id"
    assert config.signal_path_columns == {500: "path"}
    # Uniform 10 s at 500 Hz, so the truncation check is enabled.
    assert config.validation.expected_samples == {500: 5000}
    assert config.validation.amplitude_range_mv == (-10.0, 10.0)
    # Credentialed: no anonymous download URL.
    assert config.download_url is None


def test_mimic_iv_ecg_and_demo_are_distinct_configs():
    """The demo has no labels; the full release does. Easy configs to confuse."""
    full = load_config("mimic_iv_ecg")
    demo = load_config("mimic_iv_ecg_demo")

    assert full.slug != demo.slug
    assert full.lead_names == demo.lead_names  # same non-standard order
    # The distinguishing fact: only the full release ships machine_measurements.
    assert full.labels.available is True
    assert full.labels.source_csv == "machine_measurements.csv"
    assert demo.labels.available is False


def test_load_brugada_huca_config():
    """Brugada-HUCA: the only 100 Hz-only dataset, one record per subject."""
    config = load_config("brugada_huca")
    assert config.slug == "brugada_huca"
    assert config.version == "1.0.0"
    assert config.signal_format == "wfdb"
    # Every channel declares a gain of 1000/mV, so wfdb already yields mV.
    assert config.signal_unit_scale == 1.0
    assert config.leads == 12
    # Standard order and standard spelling, identical in all 363 headers.
    assert config.lead_names == ["I", "II", "III", "aVR", "aVL", "aVF",
                                "V1", "V2", "V3", "V4", "V5", "V6"]
    # The lowest rate in the catalogue, and this dataset's only one.
    assert config.sampling_rates == [100]
    assert config.default_sampling_rate == 100
    assert config.duration_seconds == 12
    # metadata.csv has no path column, so the splitter generates this one.
    assert config.metadata_csv == "ecgbench_metadata.csv"
    assert config.record_id_column == "patient_id"
    # One record per subject, so grouping would be a no-op.
    assert config.patient_id_column is None
    assert config.signal_path_columns == {100: "signal_path"}
    # 12.0 s x 100 Hz, uniform, so truncation is checked.
    assert config.validation.expected_samples == {100: 1200}
    assert config.validation.amplitude_range_mv == (-10.0, 10.0)
    # Labels come from the SHIPPED csv, not the generated one, so they do not
    # depend on the split pipeline having run.
    assert config.labels.source_csv == "metadata.csv"
    assert config.labels.join_column == "patient_id"
    assert config.label_column == "brugada"


def test_load_echonext_config():
    """EchoNext is the only npy-format, non-millivolt, unpublished-splits config."""
    config = load_config("echonext")
    assert config.name == "EchoNext"
    assert config.slug == "echonext"
    assert config.version == "1.1.0"
    # Records are rows of shared arrays, not files.
    assert config.signal_format == "npy"
    assert config.signal_path_columns == {250: "signal_path"}
    assert config.default_sampling_rate == 250
    assert config.sampling_rates == [250]
    # Standardised by the publisher: no scale recovers millivolts.
    assert config.signal_units == "zscore"
    assert config.signal_unit_scale == 1.0
    assert config.lead_names == [
        "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",
    ]
    # Generated by the splitter, because the release ships no per-record paths.
    assert config.metadata_csv == "ecgbench_metadata.csv"
    assert config.record_id_column == "ecg_key"
    assert config.patient_id_column == "patient_key"
    # The publisher's own three-way split; there is no fold 4 for no_split.
    assert config.has_predefined_splits is True
    assert config.predefined_splits is not None
    assert config.predefined_splits.column == "fold"
    assert config.predefined_splits.fold_mapping == {
        "train": [1], "val": [2], "test": [3]
    }
    # amplitude thresholds are millivolts, so the check must not be configured.
    assert config.validation is not None
    assert "amplitude_outlier" not in config.validation.checks
    assert config.validation.expected_samples == {250: 2500}
    # Restricted licence: fold CSVs stay off the public Hub.
    assert config.publish_fold_csvs is False
    assert "Restricted Health Data License" in config.no_publish_reason
    assert "ecgbench splits --dataset echonext" in config.no_publish_reason
