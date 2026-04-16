"""Tests for the download module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from ecgbench.config import DatasetConfig
from ecgbench.download import _get_archive_type, resolve_data_path


class TestGetArchiveType:
    def test_zip_from_url(self):
        assert _get_archive_type("https://example.com/data.zip") == "zip"

    def test_tar_gz_from_url(self):
        assert _get_archive_type("https://example.com/data.tar.gz") == "tar.gz"

    def test_tgz_from_url(self):
        assert _get_archive_type("https://example.com/data.tgz") == "tar.gz"

    def test_content_type_zip(self):
        assert _get_archive_type("https://example.com/data", "application/zip") == "zip"

    def test_default_zip(self):
        assert _get_archive_type("https://example.com/data") == "zip"


class TestResolveDataPath:
    def test_existing_path(self, tmp_path, sample_config):
        """If data_path exists, return it."""
        result = resolve_data_path(tmp_path, sample_config)
        assert result == tmp_path

    def test_nonexistent_path_raises(self, sample_config):
        """If data_path doesn't exist, raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            resolve_data_path(Path("/nonexistent/path"), sample_config)

    def test_none_path_no_download_url(self):
        """If no path, no download_url, and no cache, raise."""
        config = DatasetConfig(
            name="test", slug="test_no_url", version="1.0",
            url="https://example.com", download_url=None,
            metadata_csv="data.csv", record_id_column="id", label_column="label",
        )
        with pytest.raises(FileNotFoundError, match="not found locally"):
            resolve_data_path(None, config, auto_download=False)

    def test_none_path_with_cache(self, tmp_path, sample_config):
        """If cache dir exists with metadata_csv, return it."""
        cache_dir = tmp_path / ".ecgbench" / "datasets" / sample_config.slug
        cache_dir.mkdir(parents=True)
        (cache_dir / sample_config.metadata_csv).touch()

        with patch("ecgbench.download._DEFAULT_CACHE_DIR", tmp_path / ".ecgbench" / "datasets"):
            result = resolve_data_path(None, sample_config, auto_download=False)
            assert result == cache_dir
