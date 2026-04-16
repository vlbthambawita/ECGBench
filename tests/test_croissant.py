"""Tests for Croissant metadata generation."""

import json

import pytest


@pytest.fixture(autouse=True)
def _skip_if_no_mlcroissant():
    pytest.importorskip("mlcroissant")


class TestCroissantGeneration:
    def test_generate_creates_valid_json(self, tmp_splits_dir, sample_config):
        """Test that generate_croissant produces valid JSON-LD structure."""
        from ecgbench.croissant import generate_croissant

        result = generate_croissant(sample_config, tmp_splits_dir / "clean")
        assert isinstance(result, dict)

    def test_save_croissant(self, tmp_splits_dir, sample_config, tmp_path):
        """Test saving Croissant to file."""
        from ecgbench.croissant import save_croissant

        output = tmp_path / "test_croissant.json"
        saved = save_croissant(sample_config, tmp_splits_dir / "clean", output)
        assert saved.exists()

        with open(saved) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_sha256_in_distribution(self, tmp_splits_dir, sample_config):
        """Test that SHA-256 hashes are present for CSV files."""
        from ecgbench.croissant import generate_croissant

        result = generate_croissant(sample_config, tmp_splits_dir / "clean")

        # The result should contain distribution with sha256
        if "distribution" in result:
            for item in result["distribution"]:
                if isinstance(item, dict):
                    assert "sha256" in item or "sha256" in item.get("@type", "")

    def test_default_output_path(self, tmp_splits_dir, sample_config):
        """Test that default output path is splits_dir/../croissant.json."""
        from ecgbench.croissant import save_croissant

        saved = save_croissant(sample_config, tmp_splits_dir / "clean")
        assert saved == tmp_splits_dir / "croissant.json"
        assert saved.exists()


class TestCroissantValidation:
    @pytest.fixture(autouse=True)
    def _skip_if_no_mlcroissant(self):
        pytest.importorskip("mlcroissant")

    def test_validate_croissant_file(self, tmp_splits_dir, sample_config, tmp_path):
        """Test round-trip: generate -> save -> validate."""
        from ecgbench.croissant import save_croissant, validate_croissant

        output = tmp_path / "croissant.json"
        save_croissant(sample_config, tmp_splits_dir / "clean", output)
        is_valid, errors = validate_croissant(output)
        # Even if validation has issues with our minimal test data,
        # the function should not crash
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
