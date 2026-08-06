"""Tests for validation_report.json generation.

Written against issue #55: `quality_checks` reported lead-level issue counts as
`records_failed`, and derived check names by string surgery that turned
`missing_lead_3` into a check called "missing" with no description.
"""

import json

from ecgbench.validation.engine import RecordValidation, ValidationResult, summarise_validations
from ecgbench.validation.report import build_quality_checks, generate_report, save_report


def _result(validations: list[RecordValidation], total: int | None = None) -> ValidationResult:
    """Build a ValidationResult from per-record validations alone.

    The DataFrames are unused by the report, so they stay None — this keeps the
    tests free of pandas fixtures and of any signal I/O.
    """
    records, issues = summarise_validations(validations)
    valid = sum(1 for v in validations if v.is_valid)
    total = total if total is not None else len(validations)
    return ValidationResult(
        original_df=None,
        clean_df=None,
        record_validations=validations,
        summary=records,
        total_records=total,
        valid_records=valid + (total - len(validations)),
        excluded_records=len(validations) - valid,
        issue_summary=issues,
    )


class TestQualityChecks:
    def test_records_failed_differs_from_total_issues(self, sample_config):
        validations = [
            RecordValidation(
                "A1",
                False,
                [f"amplitude_outlier:lead_{i}_min_-20.00_max_2.00" for i in range(3)],
            ),
            RecordValidation("A2", False, ["amplitude_outlier:lead_0_min_-20.00_max_2.00"]),
        ]
        (check,) = generate_report(_result(validations), sample_config)["quality_checks"]
        assert check["check"] == "amplitude_outlier"
        assert check["records_failed"] == 2
        assert check["total_issues"] == 4

    def test_missing_leads_is_not_reported_as_a_check_called_missing(self, sample_config):
        validations = [RecordValidation("A1", False, ["missing_lead_0", "missing_lead_7"])]
        (check,) = generate_report(_result(validations), sample_config)["quality_checks"]
        assert check["check"] == "missing_leads"
        assert check["records_failed"] == 1
        assert check["total_issues"] == 2

    def test_every_standard_check_has_a_description(self, sample_config):
        validations = [
            RecordValidation("A1", False, ["missing_lead_0"]),
            RecordValidation("A2", False, ["nan_values:10_NaN_samples"]),
            RecordValidation("A3", False, ["truncated:3000_vs_5000"]),
            RecordValidation("A4", False, ["flat_line_lead_2"]),
            RecordValidation("A5", False, ["amplitude_outlier:lead_0_min_-20.00_max_2.00"]),
            RecordValidation("A6", False, ["corrupt_header:missing .hea"]),
            RecordValidation("A7", False, ["load_error:boom"]),
        ]
        checks = generate_report(_result(validations), sample_config)["quality_checks"]
        assert len(checks) == 7
        for check in checks:
            assert check["description"], f"{check['check']} has an empty description"

    def test_a_crashing_check_gets_a_description_too(self, sample_config):
        validations = [RecordValidation("A1", False, ["flat_line_error:boom"])]
        (check,) = generate_report(_result(validations), sample_config)["quality_checks"]
        assert check["check"] == "flat_line_error"
        assert "exception" in check["description"]

    def test_checks_are_sorted_by_name(self, sample_config):
        validations = [
            RecordValidation("A1", False, ["nan_values:1_NaN_samples"]),
            RecordValidation("A2", False, ["amplitude_outlier:lead_0_min_-20.00_max_2.00"]),
            RecordValidation("A3", False, ["missing_lead_0"]),
        ]
        checks = generate_report(_result(validations), sample_config)["quality_checks"]
        assert [c["check"] for c in checks] == ["amplitude_outlier", "missing_leads", "nan_values"]

    def test_issue_summary_may_be_omitted(self):
        """The --skip-validation stub builds a result without issue_summary."""
        assert build_quality_checks({"nan_values": 3}) == [
            {
                "check": "nan_values",
                "description": "Any NaN values in signal",
                "records_failed": 3,
                "total_issues": 3,
            }
        ]


class TestReportInvariants:
    def test_records_failed_never_exceeds_removed(self, sample_config):
        """The invariant issue #55 violated: 7059 'failed' out of 2243 removed."""
        validations = [
            RecordValidation("A1", False, [f"missing_lead_{i}" for i in range(12)]),
            RecordValidation("A2", False, [f"missing_lead_{i}" for i in range(12)]),
            RecordValidation("A3", True, []),
        ]
        report = generate_report(_result(validations), sample_config)
        removed = report["clean"]["removed"]
        assert removed == 2
        for check in report["quality_checks"]:
            assert check["records_failed"] <= removed
            assert check["total_issues"] >= check["records_failed"]

    def test_excluded_records_matches_removed(self, sample_config):
        validations = [
            RecordValidation("A1", False, ["missing_lead_0"]),
            RecordValidation("A2", True, []),
        ]
        report = generate_report(_result(validations), sample_config)
        assert len(report["excluded_records"]) == report["clean"]["removed"] == 1
        assert report["excluded_records"][0] == {"record_id": "A1", "issues": ["missing_lead_0"]}


class TestSkipValidationStub:
    def test_stub_result_produces_no_quality_checks(self, sample_config, tmp_path):
        """cli/splits.py --skip-validation passes empty summaries; must not raise."""
        stub = ValidationResult(
            original_df=None,
            clean_df=None,
            record_validations=[],
            summary={},
            total_records=10,
            valid_records=10,
            excluded_records=0,
        )
        path = save_report(stub, sample_config, tmp_path / "validation_report.json")
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["quality_checks"] == []
        assert report["excluded_records"] == []
        assert report["original"]["total_records"] == 10
