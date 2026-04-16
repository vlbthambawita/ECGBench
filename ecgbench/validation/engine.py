"""
Validation pipeline orchestrator.

Validates all ECG records in a dataset using configurable quality checks
and produces both original (all records + flags) and clean (valid only) DataFrames.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ecgbench.validation.checks import CHECK_REGISTRY

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)


@dataclass
class RecordValidation:
    """Validation result for a single ECG record."""

    record_id: str
    is_valid: bool
    issues: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Output of the validation pipeline."""

    original_df: pd.DataFrame  # all records, with 'is_valid' and 'quality_issues' columns
    clean_df: pd.DataFrame  # only valid records, quality columns dropped
    record_validations: list[RecordValidation]
    summary: dict[str, int]  # check_name -> failed_count
    total_records: int
    valid_records: int
    excluded_records: int


def _load_signal(record_path: str, signal_format: str) -> np.ndarray:
    """Load ECG signal from file. Returns shape (leads, samples)."""
    if signal_format == "wfdb":
        import wfdb

        record = wfdb.rdrecord(record_path)
        if record.p_signal is None:
            raise ValueError(f"Signal is None for record: {record_path}")
        return record.p_signal.T.astype(np.float32)
    else:
        raise NotImplementedError(
            f"Signal format '{signal_format}' not yet supported. "
            "Currently supported: wfdb"
        )


def _validate_single_record(
    record_id: str,
    record_path: str,
    signal_format: str,
    check_names: list[str],
    config_dict: dict,
    sampling_rate: int | None,
) -> RecordValidation:
    """Validate a single ECG record. Designed to run in a subprocess."""
    # Reconstruct config from dict (for pickling across processes)
    from ecgbench.config import DatasetConfig, ValidationConfig

    validation = None
    if config_dict.get("validation"):
        v = config_dict["validation"]
        validation = ValidationConfig(
            expected_leads=v["expected_leads"],
            expected_samples=v["expected_samples"],
            checks=v["checks"],
            amplitude_range_mv=tuple(v["amplitude_range_mv"]),
        )

    # Minimal config for check functions
    config = DatasetConfig(
        name=config_dict["name"],
        slug=config_dict["slug"],
        version=config_dict["version"],
        url=config_dict["url"],
        default_sampling_rate=config_dict["default_sampling_rate"],
        validation=validation,
    )

    all_issues: list[str] = []

    # Try to load the signal — corrupt_header check
    try:
        signal = _load_signal(record_path, signal_format)
    except Exception as e:
        if "corrupt_header" in check_names:
            all_issues.append(f"corrupt_header:{e}")
        return RecordValidation(
            record_id=str(record_id),
            is_valid=False,
            issues=all_issues if all_issues else [f"load_error:{e}"],
        )

    # Run each configured check
    for check_name in check_names:
        if check_name == "corrupt_header":
            continue  # Already handled above
        check_fn = CHECK_REGISTRY.get(check_name)
        if check_fn is None:
            logger.warning("Unknown check '%s', skipping", check_name)
            continue
        try:
            if check_name == "truncated_signal":
                issues = check_fn(signal, config, sampling_rate)
            else:
                issues = check_fn(signal, config)
            all_issues.extend(issues)
        except Exception as e:
            all_issues.append(f"{check_name}_error:{e}")

    return RecordValidation(
        record_id=str(record_id),
        is_valid=len(all_issues) == 0,
        issues=all_issues,
    )


def _config_to_dict(config: DatasetConfig) -> dict:
    """Serialise the config fields needed for subprocess validation."""
    result = {
        "name": config.name,
        "slug": config.slug,
        "version": config.version,
        "url": config.url,
        "default_sampling_rate": config.default_sampling_rate,
    }
    if config.validation:
        result["validation"] = {
            "expected_leads": config.validation.expected_leads,
            "expected_samples": config.validation.expected_samples,
            "checks": config.validation.checks,
            "amplitude_range_mv": list(config.validation.amplitude_range_mv),
        }
    return result


def validate_dataset(
    data_path: Path,
    config: DatasetConfig,
    sampling_rate: int | None = None,
    max_workers: int = 4,
    progress: bool = True,
) -> ValidationResult:
    """Validate all ECG records in a dataset.

    1. Read metadata CSV from data_path / config.metadata_csv
    2. For each record, load the signal file and run all checks
    3. Add 'is_valid' and 'quality_issues' columns to the DataFrame
    4. Return ValidationResult with both original and clean DataFrames

    Uses concurrent.futures.ProcessPoolExecutor for parallel validation.
    Falls back to sequential if max_workers=1 or multiprocessing fails.

    Args:
        data_path: Path to the dataset root directory
        config: DatasetConfig for this dataset
        sampling_rate: Which sampling rate to validate (default: config.default_sampling_rate)
        max_workers: Number of parallel workers
        progress: Show progress messages

    Returns:
        ValidationResult with original_df, clean_df, and statistics
    """
    rate = sampling_rate or config.default_sampling_rate
    signal_col = config.signal_path_columns.get(rate)
    if not signal_col:
        raise ValueError(
            f"No signal_path_column defined for sampling rate {rate}. "
            f"Available rates: {list(config.signal_path_columns.keys())}"
        )

    # Read metadata
    csv_path = data_path / config.metadata_csv
    df = pd.read_csv(csv_path, sep=config.metadata_csv_separator)

    check_names = config.validation.checks if config.validation else []
    config_dict = _config_to_dict(config)

    # Build list of (record_id, record_path) pairs
    records = []
    for _, row in df.iterrows():
        record_id = str(row[config.record_id_column])
        signal_path = str(row[signal_col])
        # Remove file extension for wfdb (it adds .dat/.hea itself)
        if config.signal_format == "wfdb":
            signal_path = str(Path(signal_path).with_suffix(""))
        record_path = str(data_path / signal_path)
        records.append((record_id, record_path))

    validations: list[RecordValidation] = []

    if max_workers <= 1:
        # Sequential
        for i, (record_id, record_path) in enumerate(records):
            if progress and (i + 1) % 500 == 0:
                logger.info("Validated %d / %d records", i + 1, len(records))
            v = _validate_single_record(
                record_id, record_path, config.signal_format,
                check_names, config_dict, rate,
            )
            validations.append(v)
    else:
        # Parallel
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _validate_single_record,
                        record_id, record_path, config.signal_format,
                        check_names, config_dict, rate,
                    ): record_id
                    for record_id, record_path in records
                }
                done_count = 0
                for future in as_completed(futures):
                    validations.append(future.result())
                    done_count += 1
                    if progress and done_count % 500 == 0:
                        logger.info("Validated %d / %d records", done_count, len(records))
        except Exception:
            logger.warning("Multiprocessing failed, falling back to sequential validation")
            validations = []
            for record_id, record_path in records:
                v = _validate_single_record(
                    record_id, record_path, config.signal_format,
                    check_names, config_dict, rate,
                )
                validations.append(v)

    # Build lookup: record_id -> RecordValidation
    validation_map = {v.record_id: v for v in validations}

    # Add columns to DataFrame
    df["is_valid"] = df[config.record_id_column].astype(str).map(
        lambda rid: validation_map[rid].is_valid if rid in validation_map else True
    )
    df["quality_issues"] = df[config.record_id_column].astype(str).map(
        lambda rid: ";".join(validation_map[rid].issues) if rid in validation_map else ""
    )

    # Compute summary
    summary: dict[str, int] = {}
    for v in validations:
        for issue in v.issues:
            check_name = issue.split(":")[0].split("_lead_")[0]
            summary[check_name] = summary.get(check_name, 0) + 1

    original_df = df.copy()
    clean_df = df[df["is_valid"]].drop(columns=["is_valid", "quality_issues"]).reset_index(
        drop=True
    )

    total = len(df)
    valid = int(df["is_valid"].sum())

    if progress:
        logger.info(
            "Validation complete: %d total, %d valid, %d excluded",
            total, valid, total - valid,
        )

    return ValidationResult(
        original_df=original_df,
        clean_df=clean_df,
        record_validations=validations,
        summary=summary,
        total_records=total,
        valid_records=valid,
        excluded_records=total - valid,
    )
