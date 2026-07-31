"""Shared test fixtures: synthetic signals, mock configs, temporary directories."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecgbench.config import (
    CroissantConfig,
    DatasetConfig,
    PredefinedSplitConfig,
    StratificationConfig,
    ValidationConfig,
)


@pytest.fixture
def sample_config() -> DatasetConfig:
    """Minimal DatasetConfig for testing."""
    return DatasetConfig(
        name="Test Dataset",
        slug="test_dataset",
        version="1.0.0",
        url="https://example.com/test-dataset",
        leads=12,
        duration_seconds=10.0,
        sampling_rates=[500],
        default_sampling_rate=500,
        metadata_csv="metadata.csv",
        record_id_column="record_id",
        patient_id_column="patient_id",
        signal_path_columns={500: "filename"},
        label_column="label",
        label_format="single",
        stratification=StratificationConfig(method="direct"),
        validation=ValidationConfig(
            expected_leads=12,
            expected_samples={500: 5000},
            checks=[
                "missing_leads",
                "nan_values",
                "truncated_signal",
                "flat_line",
                "amplitude_outlier",
            ],
            amplitude_range_mv=(-10.0, 10.0),
        ),
        croissant=CroissantConfig(keywords=["ECG", "test"]),
    )


@pytest.fixture
def ptbxl_config() -> DatasetConfig:
    """PTB-XL-like config with predefined splits."""
    return DatasetConfig(
        name="PTB-XL",
        slug="ptbxl",
        version="1.0.3",
        url="https://physionet.org/content/ptb-xl/1.0.3/",
        leads=12,
        duration_seconds=10.0,
        sampling_rates=[500, 100],
        default_sampling_rate=500,
        metadata_csv="ptbxl_database.csv",
        record_id_column="ecg_id",
        patient_id_column="patient_id",
        signal_path_columns={500: "filename_hr", 100: "filename_lr"},
        label_column="scp_codes",
        label_format="dict_string",
        stratification=StratificationConfig(
            method="superclass_mapping",
            mapping_source="scp_statements.csv",
            superclass_column="diagnostic_class",
        ),
        has_predefined_splits=True,
        predefined_splits=PredefinedSplitConfig(
            column="strat_fold",
            fold_mapping={
                "train": [1, 2, 3, 4, 5, 6, 7, 8],
                "val": [9],
                "test": [10],
            },
        ),
        validation=ValidationConfig(
            expected_leads=12,
            expected_samples={500: 5000, 100: 1000},
            checks=["missing_leads", "nan_values", "truncated_signal",
                     "flat_line", "amplitude_outlier"],
            amplitude_range_mv=(-10.0, 10.0),
        ),
    )


@pytest.fixture
def synthetic_signal_good() -> np.ndarray:
    """Clean 12-lead, 5000-sample signal."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 0.5, (12, 5000)).astype(np.float32)


@pytest.fixture
def synthetic_signal_bad_nan() -> np.ndarray:
    """Signal with NaN values in lead 3."""
    rng = np.random.default_rng(42)
    signal = rng.normal(0, 0.5, (12, 5000)).astype(np.float32)
    signal[3, 100:110] = np.nan
    return signal


@pytest.fixture
def synthetic_signal_missing_lead() -> np.ndarray:
    """Signal with lead 5 all zeros."""
    rng = np.random.default_rng(42)
    signal = rng.normal(0, 0.5, (12, 5000)).astype(np.float32)
    signal[5, :] = 0.0
    return signal


@pytest.fixture
def synthetic_signal_truncated() -> np.ndarray:
    """Signal with only 3000 samples instead of 5000."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 0.5, (12, 3000)).astype(np.float32)


@pytest.fixture
def synthetic_signal_flat() -> np.ndarray:
    """Signal with lead 7 having near-zero variance."""
    rng = np.random.default_rng(42)
    signal = rng.normal(0, 0.5, (12, 5000)).astype(np.float32)
    signal[7, :] = 0.001  # Near-constant, not zero (flat but not missing)
    return signal


@pytest.fixture
def synthetic_signal_amplitude_outlier() -> np.ndarray:
    """Signal with amplitude outliers in lead 0."""
    rng = np.random.default_rng(42)
    signal = rng.normal(0, 0.5, (12, 5000)).astype(np.float32)
    signal[0, 0] = 15.0  # Outside [-10, 10] range
    signal[0, 1] = -12.0
    return signal


@pytest.fixture
def mock_metadata_df() -> pd.DataFrame:
    """DataFrame mimicking a dataset with 100 records, 30 patients."""
    rng = np.random.default_rng(42)
    n = 100
    labels = rng.choice(["NORM", "MI", "STTC", "HYP", "CD"], size=n, p=[0.4, 0.2, 0.2, 0.1, 0.1])
    return pd.DataFrame({
        "record_id": [f"rec_{i:04d}" for i in range(n)],
        "patient_id": [f"pat_{i % 30:03d}" for i in range(n)],
        "filename": [f"records/rec_{i:04d}" for i in range(n)],
        "label": labels,
        "age": rng.integers(20, 90, size=n),
        "sex": rng.choice(["M", "F"], size=n),
    })


@pytest.fixture
def mock_metadata_with_folds() -> pd.DataFrame:
    """DataFrame with predefined strat_fold column (1-10)."""
    rng = np.random.default_rng(42)
    n = 200
    labels = rng.choice(["NORM", "MI", "STTC", "HYP", "CD"], size=n, p=[0.4, 0.2, 0.2, 0.1, 0.1])
    folds = np.tile(np.arange(1, 11), n // 10)
    return pd.DataFrame({
        "record_id": [f"rec_{i:04d}" for i in range(n)],
        "patient_id": [f"pat_{i % 60:03d}" for i in range(n)],
        "filename": [f"records/rec_{i:04d}" for i in range(n)],
        "label": labels,
        "strat_fold": folds,
    })


@pytest.fixture
def ecg_arrhythmia_config() -> DatasetConfig:
    """PhysioNet ecg-arrhythmia-like config: generated metadata CSV, no grouping."""
    return DatasetConfig(
        name="PhysioNet ECG Arrhythmia (Chapman-Shaoxing + Ningbo)",
        slug="ecg_arrhythmia",
        version="1.0.0",
        url="https://physionet.org/content/ecg-arrhythmia/1.0.0/",
        leads=12,
        duration_seconds=10.0,
        sampling_rates=[500],
        default_sampling_rate=500,
        metadata_csv="ecgbench_metadata.csv",
        record_id_column="record_name",
        patient_id_column=None,
        signal_path_columns={500: "signal_path"},
        label_column="dx",
        label_format="comma_separated",
        stratification=StratificationConfig(
            method="custom_function",
            mapping_source="ConditionNames_SNOMED-CT.csv",
            superclass_column="Acronym Name",
        ),
        validation=ValidationConfig(
            expected_leads=12,
            expected_samples={500: 5000},
            checks=["missing_leads", "nan_values", "truncated_signal",
                    "flat_line", "corrupt_header", "amplitude_outlier"],
            amplitude_range_mv=(-10.0, 10.0),
        ),
    )


def _write_hea(path: Path, name: str, dx: str, age: str = "60",
               sex: str = "Male", record_line: str | None = None) -> None:
    """Write a minimal 12-lead WFDB header of the ecg-arrhythmia flavour."""
    header = record_line if record_line is not None else f"{name} 12 500 5000"
    lines = [header]
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    lines += [f"{name}.mat 16+24 1000/mV 16 0 0 0 0 {lead}" for lead in leads]
    lines += [f"#Age: {age}", f"#Sex: {sex}", f"#Dx: {dx}", "#Rx: Unknown"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def tmp_ecg_arrhythmia_data(tmp_path) -> Path:
    """A miniature ecg-arrhythmia tree: WFDB headers only, no signal files.

    Mirrors the real layout (``WFDBRecords/<2 digits>/<3 digits>/JS*.hea``) plus
    the shipped SNOMED-CT mapping, so the splitter's header scan can be tested
    without any signal I/O. Deliberately includes an unmapped SNOMED code, a
    record whose diagnosis appears only once, and one malformed record line
    (the real dataset's JS01052 has its record and signal lines merged).
    """
    records = tmp_path / "WFDBRecords" / "01" / "010"
    records.mkdir(parents=True)

    # 12 sinus-bradycardia records (the dataset's dominant class)
    for i in range(1, 13):
        name = f"JS{i:05d}"
        _write_hea(records / f"{name}.hea", name, dx="426177001,164934002")
    # One atrial-fibrillation record — a rare class at this scale
    _write_hea(records / "JS00013.hea", "JS00013", dx="164889003", sex="Female")
    # One record whose primary code is absent from ConditionNames_SNOMED-CT.csv
    _write_hea(records / "JS00014.hea", "JS00014", dx="55827005", age="")
    # One record with a malformed record line
    _write_hea(
        records / "JS00015.hea", "JS00015", dx="426177001",
        record_line="JS00015 12 500 500000/mV 16 0 15 31255 0 I",
    )

    pd.DataFrame({
        "Acronym Name": ["SB", "AFIB", "TWC"],
        "Full Name": ["Sinus Bradycardia", "Atrial Fibrillation", "T wave Change"],
        "Snomed_CT": [426177001, 164889003, 164934002],
    }).to_csv(tmp_path / "ConditionNames_SNOMED-CT.csv", index=False)

    return tmp_path


@pytest.fixture
def mimic_demo_config() -> DatasetConfig:
    """MIMIC-IV-ECG-Demo-like config: no labels, patient grouping essential."""
    return DatasetConfig(
        name="MIMIC-IV-ECG Demo",
        slug="mimic_iv_ecg_demo",
        version="0.1",
        url="https://physionet.org/content/mimic-iv-ecg-demo/0.1/",
        leads=12,
        duration_seconds=10.0,
        sampling_rates=[500],
        default_sampling_rate=500,
        metadata_csv="record_list.csv",
        record_id_column="study_id",
        patient_id_column="subject_id",
        signal_path_columns={500: "path"},
        label_column="label",
        label_format="single",
        stratification=StratificationConfig(method="custom_function"),
        validation=ValidationConfig(
            expected_leads=12,
            expected_samples={500: 5000},
            checks=["missing_leads", "nan_values", "truncated_signal",
                    "flat_line", "corrupt_header", "amplitude_outlier"],
            amplitude_range_mv=(-10.0, 10.0),
        ),
    )


@pytest.fixture
def tmp_mimic_demo_data(tmp_path) -> Path:
    """A miniature MIMIC-IV-ECG Demo tree: just the shipped record_list.csv.

    Records-per-subject is deliberately uneven (12, 4, 2, 1, 1) to mirror the
    real subset's skew. No signal files — the splitter only reads the CSV.
    """
    rows = []
    for subject_id, n_studies in ((10000032, 12), (10001217, 4), (10001725, 2),
                                  (10002428, 1), (10002495, 1)):
        for i in range(n_studies):
            study_id = subject_id * 10 + i
            rows.append({
                "subject_id": subject_id,
                "study_id": study_id,
                "file_name": study_id,
                "ecg_time": f"2180-07-{i + 1:02d} 08:44:24",
                "path": f"files/p{subject_id}/s{study_id}/{study_id}",
            })
    pd.DataFrame(rows).to_csv(tmp_path / "record_list.csv", index=False)
    return tmp_path


@pytest.fixture
def tmp_splits_dir(tmp_path) -> Path:
    """Create a temporary splits directory with sample CSVs."""
    for version in ("clean", "original"):
        for split_name, fold_nums in [("train", [1, 2, 3]), ("val", [4]), ("test", [5])]:
            split_dir = tmp_path / version / split_name
            split_dir.mkdir(parents=True)
            for fold_num in fold_nums:
                n = 20
                df = pd.DataFrame({
                    "record_id": [f"rec_{fold_num}_{i}" for i in range(n)],
                    "filename": [f"records/rec_{fold_num}_{i}" for i in range(n)],
                    "fold": fold_num,
                    "default_split": split_name,
                })
                df.to_csv(split_dir / f"fold_{fold_num}.csv", index=False)

        # Master folds.csv
        all_csvs = sorted((tmp_path / version).rglob("fold_*.csv"))
        master = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
        master.to_csv(tmp_path / version / "folds.csv", index=False)

    return tmp_path


@pytest.fixture
def tmp_labels_data(tmp_path, sample_config) -> Path:
    """A source CSV for the declarative label loader, keyed like sample_config."""
    pd.DataFrame({
        "rec": ["rec_0", "rec_1", "rec_2"],
        "diagnosis": ["AFIB", "NORM", "AFIB"],
        "age": [61, 44, 78],
        "note": ["atrial fibrillation", "normal", "af with rvr"],
    }).to_csv(tmp_path / "source_labels.csv", index=False)
    return tmp_path


@pytest.fixture
def tmp_ptbxl_label_data(tmp_path) -> Path:
    """Minimal ptbxl_database.csv + scp_statements.csv for the PTB-XL loader.

    Covers the cases the derivation has to get right: a single-superclass record,
    a multi-superclass record, a record whose only statements are non-diagnostic
    (so it must land in OTHER), and a tie between two superclasses.
    """
    pd.DataFrame({
        "ecg_id": [1, 2, 3, 4],
        "patient_id": [10, 11, 12, 13],
        "scp_codes": [
            "{'NORM': 100.0, 'SR': 0.0}",       # NORM only
            "{'IMI': 100.0, 'NDT': 50.0}",      # MI (+ STTC via NDT)
            "{'SR': 0.0, 'ABQRS': 0.0}",        # no diagnostic statement -> OTHER
            "{'LVH': 100.0, 'IMI': 100.0}",     # HYP vs MI tie
        ],
        "age": [56, 62, 71, 48],
        "sex": [1, 0, 0, 1],
        "report": ["sinusrhythmus", "infarkt", "unauffaellig", "hypertrophie"],
        "strat_fold": [1, 2, 9, 10],
        "filename_lr": ["records100/00000/00001_lr"] * 4,
        "filename_hr": ["records500/00000/00001_hr"] * 4,
    }).to_csv(tmp_path / "ptbxl_database.csv", index=False)

    pd.DataFrame({
        "index": ["NORM", "IMI", "NDT", "LVH", "SR", "ABQRS"],
        "diagnostic": [1, 1, 1, 1, 0, 0],
        "form": [0, 0, 1, 0, 0, 1],
        "rhythm": [0, 0, 0, 0, 1, 0],
        "diagnostic_class": ["NORM", "MI", "STTC", "HYP", None, None],
        "diagnostic_subclass": ["NORM", "IMI", "STTC", "LVH", None, None],
    }).set_index("index").to_csv(tmp_path / "scp_statements.csv")
    return tmp_path


def _write_fold_tree(root: Path, signal_paths: list[str], record_ids: list[str]) -> None:
    """Write a {clean,original}/{train,val,test}/fold_N.csv tree plus folds.csv.

    Folds 1-3 are train, 4 is val, 5 is test — the same layout as
    ``tmp_splits_dir``. Records are dealt round-robin across the five folds so
    every fold holds at least one.
    """
    folds = {1: "train", 2: "train", 3: "train", 4: "val", 5: "test"}
    assignment = [(1 + i % 5) for i in range(len(record_ids))]

    for version in ("clean", "original"):
        for fold_num, split_name in folds.items():
            rows = [
                {
                    "record_id": rid,
                    "filename": path,
                    "fold": fold_num,
                    "default_split": split_name,
                }
                for rid, path, f in zip(record_ids, signal_paths, assignment)
                if f == fold_num
            ]
            split_dir = root / version / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(split_dir / f"fold_{fold_num}.csv", index=False)

        master = pd.concat(
            [pd.read_csv(f) for f in sorted((root / version).rglob("fold_*.csv"))],
            ignore_index=True,
        )
        master.to_csv(root / version / "folds.csv", index=False)


@pytest.fixture
def tmp_csv_signal_dataset(tmp_path) -> Path:
    """A loadable csv-format dataset: 5 records of 12 x 5000, plus a fold tree.

    Sample i of lead j holds the value ``j * 100000 + i``, so a window's contents
    identify exactly which samples were read — a shifted or truncated read cannot
    pass by accident.
    """
    records = tmp_path / "records"
    records.mkdir()
    record_ids, paths = [], []
    for r in range(5):
        rid = f"rec_{r}"
        # rows = samples, columns = leads, with a header naming the leads.
        data = np.empty((5000, 12), dtype=np.int64)
        for lead in range(12):
            data[:, lead] = lead * 100_000 + np.arange(5000)
        path = records / f"{rid}.csv"
        header = ",".join(f"L{i}" for i in range(12))
        np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%d")
        record_ids.append(rid)
        paths.append(f"records/{rid}.csv")

    _write_fold_tree(tmp_path, paths, record_ids)
    return tmp_path


@pytest.fixture
def tmp_wfdb_signal_dataset(tmp_path) -> Path:
    """A loadable wfdb-format dataset: 5 records of 12 x 5000, plus a fold tree.

    This is the only on-disk WFDB fixture in the suite; without it nothing
    exercises the real ``wfdb.rdrecord`` path, including windowed reads.
    Values are ``lead + sample/10000`` mV so a window is identifiable.
    """
    wfdb = pytest.importorskip("wfdb")

    records = tmp_path / "records"
    records.mkdir()
    record_ids, paths = [], []
    for r in range(5):
        rid = f"rec_{r}"
        signal = np.empty((5000, 12), dtype=np.float64)
        for lead in range(12):
            signal[:, lead] = lead + np.arange(5000) / 10_000.0
        wfdb.wrsamp(
            rid,
            fs=500,
            units=["mV"] * 12,
            sig_name=[f"L{i}" for i in range(12)],
            p_signal=signal,
            fmt=["16"] * 12,
            adc_gain=[1000.0] * 12,
            baseline=[0] * 12,
            write_dir=str(records),
        )
        record_ids.append(rid)
        paths.append(f"records/{rid}")

    _write_fold_tree(tmp_path, paths, record_ids)
    return tmp_path
