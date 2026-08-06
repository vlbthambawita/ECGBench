"""ECG signal validation pipeline."""

from .checks import check_name_for_issue
from .engine import RecordValidation, ValidationResult, summarise_validations, validate_dataset
from .report import build_quality_checks, generate_report, save_report

__all__ = [
    "RecordValidation",
    "ValidationResult",
    "check_name_for_issue",
    "summarise_validations",
    "validate_dataset",
    "build_quality_checks",
    "generate_report",
    "save_report",
]
