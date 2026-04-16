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
def tmp_splits_dir(tmp_path) -> Path:
    """Create a temporary splits directory with sample CSVs."""
    rng = np.random.default_rng(42)

    for version in ("clean", "original"):
        for split_name, fold_nums in [("train", [1, 2, 3]), ("val", [4]), ("test", [5])]:
            split_dir = tmp_path / version / split_name
            split_dir.mkdir(parents=True)
            for fold_num in fold_nums:
                n = 20
                df = pd.DataFrame({
                    "record_id": [f"rec_{fold_num}_{i}" for i in range(n)],
                    "filename": [f"records/rec_{fold_num}_{i}" for i in range(n)],
                    "label": rng.choice(["NORM", "MI"], size=n),
                    "fold": fold_num,
                    "default_split": split_name,
                })
                df.to_csv(split_dir / f"fold_{fold_num}.csv", index=False)

        # Master folds.csv
        all_csvs = sorted((tmp_path / version).rglob("fold_*.csv"))
        master = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
        master.to_csv(tmp_path / version / "folds.csv", index=False)

    return tmp_path
