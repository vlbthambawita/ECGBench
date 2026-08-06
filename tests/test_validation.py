"""Tests for the validation engine and individual checks."""

import numpy as np

from ecgbench.validation.checks import (
    CHECK_REGISTRY,
    check_amplitude_outlier,
    check_flat_line,
    check_missing_leads,
    check_name_for_issue,
    check_nan_values,
    check_truncated_signal,
)
from ecgbench.validation.engine import RecordValidation, summarise_validations


class TestCheckMissingLeads:
    def test_clean_signal(self, sample_config, synthetic_signal_good):
        issues = check_missing_leads(synthetic_signal_good, sample_config)
        assert issues == []

    def test_all_zero_lead(self, sample_config, synthetic_signal_missing_lead):
        issues = check_missing_leads(synthetic_signal_missing_lead, sample_config)
        assert "missing_lead_5" in issues

    def test_all_nan_lead(self, sample_config):
        signal = np.ones((12, 5000), dtype=np.float32)
        signal[3, :] = np.nan
        issues = check_missing_leads(signal, sample_config)
        assert "missing_lead_3" in issues

    def test_multiple_missing(self, sample_config):
        signal = np.ones((12, 5000), dtype=np.float32)
        signal[0, :] = 0.0
        signal[11, :] = np.nan
        issues = check_missing_leads(signal, sample_config)
        assert len(issues) == 2
        assert "missing_lead_0" in issues
        assert "missing_lead_11" in issues


class TestCheckNanValues:
    def test_clean_signal(self, sample_config, synthetic_signal_good):
        issues = check_nan_values(synthetic_signal_good, sample_config)
        assert issues == []

    def test_signal_with_nans(self, sample_config, synthetic_signal_bad_nan):
        issues = check_nan_values(synthetic_signal_bad_nan, sample_config)
        assert len(issues) == 1
        assert "nan_values:10_NaN_samples" in issues[0]


class TestCheckTruncatedSignal:
    def test_correct_length(self, sample_config, synthetic_signal_good):
        issues = check_truncated_signal(synthetic_signal_good, sample_config, sampling_rate=500)
        assert issues == []

    def test_truncated(self, sample_config, synthetic_signal_truncated):
        issues = check_truncated_signal(
            synthetic_signal_truncated, sample_config, sampling_rate=500
        )
        assert len(issues) == 1
        assert "truncated:3000_vs_5000" in issues[0]

    def test_no_validation_config(self, synthetic_signal_truncated):
        """No validation config means no truncation check."""
        from ecgbench.config import DatasetConfig
        config = DatasetConfig(
            name="test", slug="test", version="1.0", url="http://x",
            metadata_csv="x.csv", record_id_column="id", label_column="label",
        )
        issues = check_truncated_signal(synthetic_signal_truncated, config, sampling_rate=500)
        assert issues == []


class TestCheckFlatLine:
    def test_clean_signal(self, sample_config, synthetic_signal_good):
        issues = check_flat_line(synthetic_signal_good, sample_config)
        assert issues == []

    def test_flat_lead(self, sample_config, synthetic_signal_flat):
        issues = check_flat_line(synthetic_signal_flat, sample_config)
        assert "flat_line_lead_7" in issues

    def test_missing_lead_not_reported(self, sample_config, synthetic_signal_missing_lead):
        """All-zero leads should NOT be reported as flat_line (missing_leads catches them)."""
        issues = check_flat_line(synthetic_signal_missing_lead, sample_config)
        assert "flat_line_lead_5" not in issues


class TestCheckAmplitudeOutlier:
    def test_clean_signal(self, sample_config, synthetic_signal_good):
        issues = check_amplitude_outlier(synthetic_signal_good, sample_config)
        assert issues == []

    def test_outlier(self, sample_config, synthetic_signal_amplitude_outlier):
        issues = check_amplitude_outlier(synthetic_signal_amplitude_outlier, sample_config)
        assert len(issues) >= 1
        assert any("amplitude_outlier:lead_0" in i for i in issues)


class TestCheckNameForIssue:
    """Every issue string must resolve to the check that emitted it (issue #55)."""

    def test_per_lead_prefixes_keep_their_registry_name(self):
        assert check_name_for_issue("missing_lead_3") == "missing_leads"
        assert check_name_for_issue("flat_line_lead_11") == "flat_line"

    def test_truncated_keeps_its_registry_name(self):
        assert check_name_for_issue("truncated:4500_vs_5000") == "truncated_signal"

    def test_colon_delimited_issues(self):
        assert (
            check_name_for_issue("amplitude_outlier:lead_0_min_-8.67_max_11.85")
            == "amplitude_outlier"
        )
        assert check_name_for_issue("nan_values:10_NaN_samples") == "nan_values"

    def test_engine_generated_issues(self):
        assert check_name_for_issue("corrupt_header:No such file") == "corrupt_header"
        assert check_name_for_issue("load_error:boom") == "load_error"
        assert check_name_for_issue("flat_line_error:boom") == "flat_line_error"

    def test_every_check_in_the_registry_resolves_to_itself(self, sample_config):
        """Guards against a new check whose issue prefix is not registered."""
        signal = np.zeros((12, 100), dtype=np.float32)
        signal[0, :] = np.nan
        for name, fn in CHECK_REGISTRY.items():
            issues = (
                fn(signal, sample_config, 500)
                if name == "truncated_signal"
                else fn(signal, sample_config)
            )
            for issue in issues:
                assert check_name_for_issue(issue) == name, (
                    f"check '{name}' emits {issue!r}, which resolves to "
                    f"'{check_name_for_issue(issue)}'"
                )


class TestSummariseValidations:
    """`summary` counts records, not leads (issue #55)."""

    def test_counts_records_not_leads(self):
        validations = [
            RecordValidation("A1", False, ["missing_lead_0", "missing_lead_1", "missing_lead_2"]),
            RecordValidation("A2", True, []),
        ]
        records, issues = summarise_validations(validations)
        assert records == {"missing_leads": 1}
        assert issues == {"missing_leads": 3}

    def test_no_fabricated_check_names(self):
        """`missing_lead_N` used to summarise as a check called 'missing'."""
        records, _ = summarise_validations([RecordValidation("A1", False, ["missing_lead_0"])])
        assert "missing" not in records
        assert set(records) <= set(CHECK_REGISTRY) | {"corrupt_header", "load_error"}

    def test_a_record_failing_two_checks_counts_once_per_check(self):
        validations = [
            RecordValidation(
                "A1",
                False,
                ["missing_lead_0", "amplitude_outlier:lead_4_min_-20.00_max_2.00"],
            ),
            RecordValidation("A2", False, ["amplitude_outlier:lead_4_min_-20.00_max_2.00"]),
        ]
        records, issues = summarise_validations(validations)
        assert records == {"missing_leads": 1, "amplitude_outlier": 2}
        assert issues == {"missing_leads": 1, "amplitude_outlier": 2}

    def test_records_failed_never_exceeds_the_number_of_records(self):
        validations = [
            RecordValidation(f"A{i}", False, [f"flat_line_lead_{j}" for j in range(12)])
            for i in range(5)
        ]
        records, issues = summarise_validations(validations)
        assert records == {"flat_line": 5}
        assert issues == {"flat_line": 60}

    def test_empty_input(self):
        assert summarise_validations([]) == ({}, {})


class TestAllChecksCombined:
    def test_good_signal_passes_all(self, sample_config, synthetic_signal_good):
        """A clean signal should pass every check."""
        for check_fn in [
            check_missing_leads, check_nan_values, check_flat_line, check_amplitude_outlier,
        ]:
            assert check_fn(synthetic_signal_good, sample_config) == []
        assert check_truncated_signal(
            synthetic_signal_good, sample_config, sampling_rate=500
        ) == []
