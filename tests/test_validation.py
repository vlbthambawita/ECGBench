"""Tests for the validation engine and individual checks."""

import numpy as np

from ecgbench.validation.checks import (
    check_amplitude_outlier,
    check_flat_line,
    check_missing_leads,
    check_nan_values,
    check_truncated_signal,
)


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
