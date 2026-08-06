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


def test_load_staffiii_config():
    """STAFF III: 9 leads with the precordials FIRST, variable length, open licence."""
    config = load_config("staffiii")
    assert config.slug == "staffiii"
    assert config.version == "1.0.0"
    assert config.signal_format == "wfdb"
    # Every record declares gain 1600 adu/mV, so wfdb already returns millivolts.
    assert config.signal_unit_scale == 1.0
    # NINE leads. aVR/aVL/aVF are not stored — they are linear combinations of
    # I and II — and the precordials come first, so signal[0] is V1, not lead I.
    assert config.leads == 9
    assert config.lead_names == ["V1", "V2", "V3", "V4", "V5", "V6", "I", "II", "III"]
    assert config.sampling_rates == [1000]
    assert config.default_sampling_rate == 1000
    # No machine-readable metadata ships (the annotations are an .xlsx), so
    # STAFFIIISplitter generates this on first run.
    assert config.metadata_csv == "ecgbench_metadata.csv"
    assert config.record_id_column == "record_name"
    # 104 patients over 520 records, five each on average — grouping is mandatory.
    assert config.patient_id_column == "patient_id"
    assert config.signal_path_columns == {1000: "signal_path"}
    assert config.label_column == "recording_type"
    assert config.stratification is not None
    assert config.stratification.method == "custom_function"
    assert config.has_predefined_splits is False
    # Records run 94,514 to 960,000 samples; an entry here would fail most of them.
    assert config.validation is not None
    assert config.validation.expected_leads == 9
    assert config.validation.expected_samples == {}
    # The int16 rail sits at exactly +/-20.48 mV (32768/1600); +/-20 catches the
    # records that clipped, while the +/-10 default would drop 71 of 520.
    assert config.validation.amplitude_range_mv == (-20.0, 20.0)
    assert "corrupt_header" in config.validation.checks
    # ODC-By: the fold CSVs are publishable.
    assert config.publish_fold_csvs is True


def test_load_cpsc_2018_config():
    """CPSC 2018: multi-label, variable length, contained whole in both challenges."""
    config = load_config("cpsc_2018")
    assert config.slug == "cpsc_2018"
    assert config.version == "1.0.0"
    assert config.signal_format == "wfdb"
    # All 82,524 signal lines declare gain 1000/mV, so wfdb already yields mV.
    assert config.signal_unit_scale == 1.0
    assert config.leads == 12
    # Standard order and standard capitalisation in all 6,877 records.
    assert config.lead_names == ["I", "II", "III", "aVR", "aVL", "aVF",
                                "V1", "V2", "V3", "V4", "V5", "V6"]
    assert config.sampling_rates == [500]
    assert config.default_sampling_rate == 500
    assert config.signal_path_columns == {500: "signal_path"}
    # No metadata ships (not even REFERENCE.csv): CPSC2018Splitter generates it.
    assert config.metadata_csv == "ecgbench_metadata.csv"
    assert config.record_id_column == "record_name"
    # No patient identifiers are published.
    assert config.patient_id_column is None
    assert config.label_column == "dx"
    assert config.label_format == "comma_separated"
    assert config.stratification is not None
    assert config.stratification.method == "custom_function"
    assert config.has_predefined_splits is False
    # Records run 3,000 to 72,000 samples (6-144 s) in 1,650 distinct lengths,
    # so any entry here would fail thousands of legitimate records.
    assert config.validation is not None
    assert config.validation.expected_leads == 12
    assert config.validation.expected_samples == {}
    assert config.validation.amplitude_range_mv == (-10.0, 10.0)
    # CC BY 4.0 (the licence PhysioNet redistributes these exact files under).
    assert config.publish_fold_csvs is True


def test_load_sph_config():
    """SPH: the first hdf5 dataset, patient-grouped, multi-label AHA statements."""
    config = load_config("sph")
    assert config.slug == "sph"
    assert config.version == "1.0.0"
    # The one and only hdf5 dataset — one .h5 per record holding a single
    # (12, N) root dataset named 'ecg'. Needs the [hdf5] extra.
    assert config.signal_format == "hdf5"
    # float16 arrays already in millivolts, per the paper and confirmed against
    # the files (median per-record peak 1.74 mV).
    assert config.signal_unit_scale == 1.0
    assert config.leads == 12
    # Standard order and standard capitalisation, verified from the arrays via
    # Einthoven's and Goldberger's relations rather than from the paper.
    assert config.lead_names == ["I", "II", "III", "aVR", "aVL", "aVF",
                                "V1", "V2", "V3", "V4", "V5", "V6"]
    assert config.sampling_rates == [500]
    assert config.default_sampling_rate == 500
    assert config.signal_path_columns == {500: "signal_path"}
    # metadata.csv ships but has no signal-path column, so SPHSplitter normalises
    # it — and the validation engine re-reads the generated file from disk.
    assert config.metadata_csv == "ecgbench_metadata.csv"
    assert config.record_id_column == "ecg_id"
    # 1,066 of the 24,666 patients contributed more than one record, so grouping
    # is mandatory rather than cosmetic.
    assert config.patient_id_column == "patient_id"
    assert config.label_column == "aha_primary_codes"
    assert config.stratification is not None
    assert config.stratification.method == "custom_function"
    assert config.has_predefined_splits is False
    # 5,000 to 28,000 samples (10-56 s) in 39 distinct lengths; the per-record
    # length is in the metadata's n_samples column instead.
    assert config.validation is not None
    assert config.validation.expected_leads == 12
    assert config.validation.expected_samples == {}
    # 323 of 25,770 records fail this deliberately — the median peak is 1.74 mV,
    # so beyond 10 mV means a railed or artefact-dominated lead.
    assert config.validation.amplitude_range_mv == (-10.0, 10.0)
    # Labels need the code.csv join, so the loader is a module, not declarative.
    assert config.labels is not None
    assert config.labels.source_csv is None
    assert config.labels.join_column == "ecg_id"
    # CC BY 4.0: the fold CSVs are publishable.
    assert config.publish_fold_csvs is True


def test_load_ningbo_iva_config():
    """Ningbo IVA: alphabetical lead order and an estimated, not declared, mV scale."""
    config = load_config("ningbo_iva")
    assert config.slug == "ningbo_iva"
    assert config.version == "1.0.0"
    assert config.signal_format == "csv"
    # ESTIMATED, not declared — the release ships bare integers and states no
    # unit. 1 mV = 16384 counts, measured against sph sex-for-sex; see the config.
    assert config.signal_unit_scale == pytest.approx(6.1035e-05)
    assert config.leads == 12
    # ALPHABETICAL, straight off the CSV header row: signal[0] is aVF, and lead I
    # is signal[3]. This is why leads= takes names rather than indices.
    assert config.lead_names == ["aVF", "aVL", "aVR", "I", "II", "III",
                                "V1", "V2", "V3", "V4", "V5", "V6"]
    assert config.lead_names[0] == "aVF"
    assert config.lead_names.index("I") == 3
    # 2000 Hz, the highest rate in the catalogue — an EP-lab acquisition system,
    # not a diagnostic cart.
    assert config.sampling_rates == [2000]
    assert config.default_sampling_rate == 2000
    assert config.signal_path_columns == {2000: "signal_path"}
    # Diagnosis.xlsx has no path column, so NingboIVASplitter normalises it.
    assert config.metadata_csv == "ecgbench_metadata.csv"
    assert config.record_id_column == "hospital_id"
    # HospitalID is both the record and the patient id, one record each, so this
    # stays null rather than asserting a grouping nothing exercised.
    assert config.patient_id_column is None
    assert config.label_column == "left_right"
    assert config.label_format == "single"
    assert config.stratification is not None
    assert config.stratification.method == "direct"
    assert config.has_predefined_splits is False
    # 5,791 to 118,642 samples in 317 distinct lengths over 334 records.
    assert config.validation is not None
    assert config.validation.expected_leads == 12
    assert config.validation.expected_samples == {}
    # Nothing trips this: the largest sample anywhere is 9.45 estimated mV. Left
    # wide on purpose, so a 20% error in the estimated scale cannot start
    # excluding records.
    assert config.validation.amplitude_range_mv == (-10.0, 10.0)
    assert config.labels is not None
    assert config.labels.source_csv is None
    assert config.labels.join_column == "hospital_id"
    # CC BY 4.0: the fold CSVs are publishable.
    assert config.publish_fold_csvs is True


def test_the_hdf5_and_2000hz_facts_are_intentional():
    """Guard the configs that broke a catalogue-wide assumption.

    ``ningbo_iva`` is still the only 2000 Hz dataset. hdf5 is now the second
    commonest format: ``sph`` reads a 2-D ``(leads, samples)`` array per file,
    while ``code15``, ``code_test``, ``sami_trop`` and ``ikem`` read a **row of a
    shared 3-D** ``(records, samples, leads)`` array and get transposed. The two
    shapes go through the same ``signal_format``, so this list is the reminder
    that adding another means deciding which of the two layouts it is.

    ``ikem`` is also the only one whose 3-D array is not 12 leads wide — it
    stores 8 — so a reader that hardcoded 12 anywhere would pass on the other
    four and fail here.
    """
    by_format: dict[str, list[str]] = {}
    fastest: list[str] = []
    for slug in list_available_configs():
        config = load_config(slug)
        by_format.setdefault(config.signal_format, []).append(slug)
        if config.default_sampling_rate >= 2000:
            fastest.append(slug)

    assert sorted(by_format["hdf5"]) == [
        "code15", "code_test", "ikem", "sami_trop", "sph",
    ]
    assert fastest == ["ningbo_iva"]
    # Every hdf5 dataset but sph reads a row of a shared 3-D array, and all of
    # them declare 12 leads except ikem, which stores only the 8 independent ones.
    assert load_config("ikem").leads == 8
    assert {load_config(s).leads for s in by_format["hdf5"]} == {8, 12}


def test_load_code15_config():
    """CODE-15%: a row of a shared 3-D HDF5 array, patient-grouped, six flags."""
    config = load_config("code15")
    assert config.slug == "code15"
    assert config.version == "1.0.0"
    # Not one file per record: 18 parts each holding a (N, 4096, 12) array, so a
    # record names a row — "exams_part0.hdf5:tracings:417". Needs h5py.
    assert config.signal_format == "hdf5"
    # Already millivolts. The bundled README's "scale 1e-4V ... multiplied by
    # 1000 to obtain V" is self-contradictory; measured median lead-II R is
    # 1.75 mV, which settles it.
    assert config.signal_unit_scale == 1.0
    assert config.leads == 12
    # STANDARD — and the sibling code_test release is not. Verified from the
    # arrays via Einthoven's and Goldberger's relations.
    assert config.lead_names == ["I", "II", "III", "aVR", "aVL", "aVF",
                                 "V1", "V2", "V3", "V4", "V5", "V6"]
    assert config.sampling_rates == [400]
    assert config.default_sampling_rate == 400
    assert config.signal_path_columns == {400: "signal_path"}
    # exams.csv gives a part but not a row, and its rows are not in file order,
    # so CODE15Splitter resolves exam_id -> row and writes this file. The
    # validation engine re-reads it from disk.
    assert config.metadata_csv == "ecgbench_metadata.csv"
    assert config.record_id_column == "exam_id"
    # 66,929 of 233,770 patients contributed more than one record, up to 38.
    assert config.patient_id_column == "patient_id"
    assert config.label_column == "abnormality_codes"
    assert config.label_format == "comma_separated"
    assert config.stratification is not None
    assert config.stratification.method == "custom_function"
    assert config.has_predefined_splits is False
    assert config.validation is not None
    assert config.validation.expected_leads == 12
    # Uniform, unlike sph: every record is exactly 4,096 samples at 400 Hz.
    assert config.validation.expected_samples == {400: 4096}
    # Wider than ECGBench's usual +-10 on purpose. These are telehealth
    # recordings with a median peak of 4.27 mV, so +-10 would exclude 11.8% of
    # the release; +-20 excludes ~2%. See the config comment.
    assert config.validation.amplitude_range_mv == (-20.0, 20.0)
    # Module loader: the six flags need reducing to a list and a stratify class.
    assert config.labels is not None
    assert config.labels.source_csv is None
    assert config.labels.join_column == "exam_id"
    # CC BY 4.0: the fold CSVs are publishable.
    assert config.publish_fold_csvs is True


def test_load_code_test_config():
    """CODE-test: 827 records with no identifiers, joined by row position."""
    config = load_config("code_test")
    assert config.slug == "code_test"
    assert config.version == "1.0.3"
    assert config.signal_format == "hdf5"
    # float64 arrays, already millivolts (median lead-II R 1.48 mV).
    assert config.signal_unit_scale == 1.0
    assert config.leads == 12
    # NON-STANDARD: aVL, aVF, aVR. signal[3] is aVL here and aVR in code15.
    assert config.lead_names == ["I", "II", "III", "aVL", "aVF", "aVR",
                                 "V1", "V2", "V3", "V4", "V5", "V6"]
    assert config.sampling_rates == [400]
    assert config.default_sampling_rate == 400
    assert config.signal_path_columns == {400: "signal_path"}
    # No shipped table carries a record identifier, so there is nothing to point
    # at but a generated file.
    assert config.metadata_csv == "ecgbench_metadata.csv"
    # The row index into the tracings array, 0-826 — the only key the release's
    # own documentation defines.
    assert config.record_id_column == "record_id"
    # 827 tracings from 827 different patients, and no patient id ships.
    assert config.patient_id_column is None
    assert config.label_column == "abnormality_codes"
    assert config.label_format == "comma_separated"
    assert config.stratification is not None
    assert config.stratification.method == "custom_function"
    # The release is the test half of a division whose training half is a
    # different dataset, so there is no fold column to honour.
    assert config.has_predefined_splits is False
    assert config.validation is not None
    assert config.validation.expected_leads == 12
    assert config.validation.expected_samples == {400: 4096}
    # Same cohort and instruments as code15, so deliberately the same range —
    # the two must not be cleaned to different standards.
    assert config.validation.amplitude_range_mv == (-20.0, 20.0)
    assert config.validation.amplitude_range_mv == (
        load_config("code15").validation.amplitude_range_mv
    )
    # Module loader: eight keyless source files, all joined by position.
    assert config.labels is not None
    assert config.labels.source_csv is None
    assert config.labels.join_column == "record_id"
    # CC BY 4.0: the fold CSVs are publishable.
    assert config.publish_fold_csvs is True


def test_load_sami_trop_config():
    """SaMi-Trop: one keyless HDF5 array, one record per patient, no diagnoses."""
    config = load_config("sami_trop")
    assert config.slug == "sami_trop"
    assert config.version == "1.0.0"
    # A single (1631, 4096, 12) `tracings` array, so a record names a row —
    # "exams.hdf5:tracings:417". Needs h5py.
    assert config.signal_format == "hdf5"
    # Already millivolts: median per-record peak 4.3 mV, median precordial
    # peak-to-peak 1.7 mV.
    assert config.signal_unit_scale == 1.0
    assert config.leads == 12
    # Standard — checked from the arrays, because the sibling code_test release
    # from the same network is not standard.
    assert config.lead_names == ["I", "II", "III", "aVR", "aVL", "aVF",
                                 "V1", "V2", "V3", "V4", "V5", "V6"]
    assert config.sampling_rates == [400]
    assert config.default_sampling_rate == 400
    assert config.duration_seconds == 10.24
    assert config.signal_path_columns == {400: "signal_path"}
    # The HDF5 has no exam_id dataset, so the row reference has to be generated
    # and written to disk for the validation engine to see it.
    assert config.metadata_csv == "ecgbench_metadata.csv"
    assert config.record_id_column == "exam_id"
    # THE UNUSUAL BIT: one recording per patient — the release is each patient's
    # first exam — so there is genuinely nothing to group on. Every other
    # patient-level dataset in the catalogue sets this.
    assert config.patient_id_column is None
    # No diagnostic vocabulary ships at all; this names the stratification
    # reduction instead.
    assert config.label_column == "stratify_class"
    assert config.label_format == "single"
    assert config.stratification is not None
    assert config.stratification.method == "custom_function"
    assert config.has_predefined_splits is False
    assert config.validation is not None
    assert config.validation.expected_leads == 12
    # Uniform: the array is one rectangular block.
    assert config.validation.expected_samples == {400: 4096}
    # Same telehealth network and instruments as code15, so deliberately the
    # same range — +-10 would exclude 11.0% of the release, +-20 excludes 0.9%.
    assert config.validation.amplitude_range_mv == (-20.0, 20.0)
    assert config.validation.amplitude_range_mv == (
        load_config("code15").validation.amplitude_range_mv
    )
    # Module loader: the positional join has to be validated in one place.
    assert config.labels is not None
    assert config.labels.source_csv == "exams.csv"
    assert config.labels.join_column == "exam_id"
    # CC BY 4.0: the fold CSVs are publishable.
    assert config.publish_fold_csvs is True


def test_load_ikem_config():
    """IKEM: eight leads in an unusual order, microvolts, and not published."""
    config = load_config("ikem")
    assert config.slug == "ikem"
    assert config.version == "1.0.0"
    assert config.signal_format == "hdf5"
    # int16 MICROVOLTS. The release says "a granularity of 4.88 microvolts",
    # which would make the median lead p2p 7.7 mV — about 5x physiologic. At
    # 1 uV/count the median per-record peak is 1.93 mV, matching sph's 1.74.
    assert config.signal_unit_scale == 0.001
    # EIGHT, not twelve: only the independent leads are stored. III/aVR/aVL/aVF
    # are exact linear combinations and ECGBench does not synthesise them.
    assert config.leads == 8
    # THE MOST UNUSUAL LEAD ORDER IN THE CATALOGUE: precordial first, and II
    # before I. Derived from the arrays — the frontal QRS axis is +51 deg
    # (median) under this assignment and +1 deg under the swap.
    assert config.lead_names == ["V1", "V2", "V3", "V4", "V5", "V6", "II", "I"]
    assert len(config.lead_names) == config.leads
    assert config.sampling_rates == [500]
    assert config.default_sampling_rate == 500
    # 4,096 samples at 500 Hz. NOT the "10 seconds" the release claims, which
    # would be 5,000 samples; verified against the cart's own ventricular_rate.
    assert config.duration_seconds == 8.192
    assert config.metadata_csv == "ecgbench_metadata.csv"
    assert config.record_id_column == "exam_id"
    # 88.6% of records share a patient with another record; one patient has 96.
    assert config.patient_id_column == "patient_id"
    # No diagnoses ship, so this is a banded rate measurement.
    assert config.label_column == "stratify_class"
    assert config.stratification is not None
    assert config.stratification.method == "custom_function"
    assert config.has_predefined_splits is False
    assert config.validation is not None
    assert config.validation.expected_leads == 8
    # Storage length is uniform even though 48 records are zero-padded from
    # 2,500 real samples, because the arrays are rectangular.
    assert config.validation.expected_samples == {500: 4096}
    # Clean single-hospital data, so the ECGBench-usual range fits: +-10 mV
    # excludes 2.9%, where the two TNMG telehealth cohorts needed +-20.
    assert config.validation.amplitude_range_mv == (-10.0, 10.0)
    assert config.labels is not None
    assert config.labels.source_csv == "exams.csv"
    assert config.labels.join_column == "exam_id"


def test_ikem_fold_csvs_are_not_published():
    """CC BY-NC-ND 4.0: NoDerivatives, so the identifiers stay unpublished.

    The second dataset in the catalogue to be withheld, and the first withheld
    by its licence rather than by a data use agreement. The reason string is
    what ``ecgbench upload`` and ``ECGDataset`` quote back at a user, so it has
    to carry the regeneration command.
    """
    config = load_config("ikem")
    assert config.publish_fold_csvs is False
    assert config.no_publish_reason
    assert "CC BY-NC-ND" in config.no_publish_reason
    assert "ecgbench splits --dataset ikem" in config.no_publish_reason
    assert "verify_splits" in config.no_publish_reason
    # The licence field must say so too, not repeat Zenodo's vaguer "other-at".
    assert config.license == "CC BY-NC-ND 4.0"


def test_load_zzu_pecg_config():
    """ZZU-pECG: paediatric, variable length, and two lead layouts."""
    config = load_config("zzu_pecg")
    assert config.slug == "zzu_pecg"
    assert config.version == "1.0.0"
    assert config.signal_format == "wfdb"
    # Per-record, per-lead float gains in the headers, which wfdb applies.
    assert config.signal_unit_scale == 1.0
    assert config.leads == 12
    # Uppercase limb leads, as the headers spell them.
    assert config.lead_names == ["I", "II", "III", "AVR", "AVL", "AVF",
                                "V1", "V2", "V3", "V4", "V5", "V6"]
    # THE ONLY DATASET WITH A SECOND LAYOUT: 1,856 of 14,190 records store 9
    # leads without V2/V4/V6, so stored position 7 is V2 here and V3 there.
    assert config.alternate_lead_names == {
        9: ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V3", "V5"]
    }
    assert config.sampling_rates == [500]
    assert config.default_sampling_rate == 500
    assert config.metadata_csv == "ecgbench_metadata.csv"
    assert config.record_id_column == "ECG_ID"
    # 1,691 patients hold more than one record, up to 19.
    assert config.patient_id_column == "Patient_ID"
    assert config.label_column == "aha_codes"
    assert config.label_format == "comma_separated"
    assert config.stratification is not None
    assert config.stratification.method == "custom_function"
    assert config.has_predefined_splits is False
    assert config.validation is not None
    # DELIBERATELY EMPTY: lengths run 2,500-60,000 samples over 67 distinct
    # values, so any single expectation would fail thousands of valid records.
    # An omitted rate disables truncated_signal rather than making it fire.
    assert config.validation.expected_samples == {}
    # A hard rail at ~26.6 mV that 11.8% of records touch, and paediatric high
    # voltage is genuinely large — "left ventricular high voltage" is the second
    # commonest finding here.
    assert config.validation.amplitude_range_mv == (-20.0, 20.0)
    assert config.labels is not None
    assert config.labels.source_csv == "AttributesDictionary.csv"
    assert config.labels.join_column == "ECG_ID"
    # CC BY 4.0 on figshare: the fold CSVs are publishable.
    assert config.publish_fold_csvs is True
