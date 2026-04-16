"""
Validation report generation.

Produces a JSON-serialisable report documenting the validation results,
including per-check statistics and excluded record details.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig
    from ecgbench.validation.engine import ValidationResult

# Check descriptions for the report
_CHECK_DESCRIPTIONS: dict[str, str] = {
    "missing_leads": "Lead entirely NaN or all-zero",
    "nan_values": "Any NaN values in signal",
    "truncated_signal": "Fewer samples than expected",
    "flat_line": "Lead with near-zero variance",
    "corrupt_header": "Unreadable signal file or header",
    "amplitude_outlier": "Samples outside physiological range",
    "load_error": "Failed to load signal file",
}


def generate_report(result: ValidationResult, config: DatasetConfig) -> dict:
    """Generate a JSON-serialisable validation report dict.

    Args:
        result: ValidationResult from validate_dataset()
        config: DatasetConfig for the dataset

    Returns:
        dict suitable for json.dump()
    """
    try:
        from ecgbench._version import __version__
    except ImportError:
        __version__ = "0.0.0.dev0"

    # Build per-check stats
    quality_checks = []
    for check_name, failed_count in sorted(result.summary.items()):
        # Count total issues (some checks produce multiple issues per record)
        total_issues = sum(
            len([i for i in v.issues if i.startswith(check_name)])
            for v in result.record_validations
        )
        quality_checks.append({
            "check": check_name,
            "description": _CHECK_DESCRIPTIONS.get(check_name, ""),
            "records_failed": failed_count,
            "total_issues": total_issues,
        })

    # Build excluded records list
    excluded_records = [
        {"record_id": v.record_id, "issues": v.issues}
        for v in result.record_validations
        if not v.is_valid
    ]

    return {
        "dataset": config.slug,
        "source_version": config.version,
        "ecgbench_version": __version__,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "sampling_rate_validated": config.default_sampling_rate,
        "original": {
            "total_records": result.total_records,
        },
        "clean": {
            "total_records": result.valid_records,
            "removed": result.excluded_records,
        },
        "quality_checks": quality_checks,
        "excluded_records": excluded_records,
    }


def save_report(
    result: ValidationResult,
    config: DatasetConfig,
    output_path: Path,
) -> Path:
    """Generate and save validation_report.json.

    Args:
        result: ValidationResult from validate_dataset()
        config: DatasetConfig for the dataset
        output_path: Where to write the JSON file

    Returns:
        Path to the saved report file
    """
    report = generate_report(result, config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return output_path
