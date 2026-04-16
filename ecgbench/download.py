"""
Auto-download and dataset path resolution.

resolve_data_path() is the single entry point for locating dataset files.
If no local path is provided, downloads from the source URL defined in the YAML config.
"""

from __future__ import annotations

import logging
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".ecgbench" / "datasets"
_LOCK_FILENAME = ".ecgbench_downloading"


def _get_archive_type(url: str, content_type: str | None = None) -> str:
    """Determine archive type from URL suffix or Content-Type header."""
    url_lower = url.lower()
    if url_lower.endswith(".zip"):
        return "zip"
    if url_lower.endswith(".tar.gz") or url_lower.endswith(".tgz"):
        return "tar.gz"
    if url_lower.endswith(".tar"):
        return "tar"
    if content_type:
        if "zip" in content_type:
            return "zip"
        if "gzip" in content_type or "tar" in content_type:
            return "tar.gz"
    return "zip"  # Default assumption


def _find_metadata_csv(target_dir: Path, metadata_csv: str) -> Path | None:
    """Find the metadata CSV, walking one level deep for nested archives."""
    direct = target_dir / metadata_csv
    if direct.exists():
        return target_dir

    # Walk one level deep (common with PhysioNet archives)
    for child in target_dir.iterdir():
        if child.is_dir():
            candidate = child / metadata_csv
            if candidate.exists():
                return child

    return None


def download_dataset(
    config: DatasetConfig,
    target_dir: Path | None = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    """Download a dataset from its source URL.

    Args:
        config: DatasetConfig (must have download_url set)
        target_dir: Where to extract. Defaults to ~/.ecgbench/datasets/<slug>/
        force: Re-download even if target_dir exists
        progress: Show progress messages

    Returns:
        Path to the dataset root directory (containing config.metadata_csv)

    Raises:
        ValueError: if config.download_url is None
        PermissionError: if server returns 403
        ConnectionError: if download fails
    """
    if not config.download_url:
        raise ValueError(
            f"Dataset '{config.slug}' has no download_url configured. "
            f"Download it manually from {config.url} and pass the local path."
        )

    if target_dir is None:
        target_dir = _DEFAULT_CACHE_DIR / config.slug
    target_dir = Path(target_dir)

    # Check for existing download
    if target_dir.exists() and not force:
        dataset_root = _find_metadata_csv(target_dir, config.metadata_csv)
        if dataset_root:
            logger.info("Dataset already exists at %s", dataset_root)
            return dataset_root

    # Check for lock file
    lock_file = target_dir / _LOCK_FILENAME
    if lock_file.exists():
        raise RuntimeError(
            f"Lock file exists at {lock_file}. Another download may be in progress. "
            "Remove the lock file manually if this is stale."
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    lock_file.touch()

    try:
        if progress:
            logger.info("Downloading %s from %s", config.slug, config.download_url)

        # Download the archive
        request = urllib.request.Request(config.download_url)
        try:
            response = urllib.request.urlopen(request)  # noqa: S310
        except urllib.error.HTTPError as e:
            if e.code == 403:
                raise PermissionError(
                    f"Access denied (HTTP 403). This dataset may require PhysioNet "
                    f"credentials. Download it manually from {config.url} and pass "
                    f"the local path to data_path."
                ) from e
            raise ConnectionError(
                f"Download failed with HTTP {e.code}: {e.reason}"
            ) from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"Download failed: {e.reason}") from e

        content_type = response.headers.get("Content-Type", "")
        archive_type = _get_archive_type(config.download_url, content_type)

        archive_path = target_dir / f"download.{archive_type}"

        # Stream download
        total_size = response.headers.get("Content-Length")
        total_size = int(total_size) if total_size else None
        downloaded = 0
        chunk_size = 8192

        try:
            # Try tqdm for progress bar
            import tqdm

            pbar = tqdm.tqdm(
                total=total_size, unit="B", unit_scale=True,
                desc=config.slug, disable=not progress,
            )
        except ImportError:
            pbar = None

        with open(archive_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if pbar:
                    pbar.update(len(chunk))
                elif progress and total_size and downloaded % (1024 * 1024) < chunk_size:
                    pct = downloaded * 100 // total_size
                    logger.info("Downloaded %d%% (%d MB)", pct, downloaded // (1024 * 1024))

        if pbar:
            pbar.close()

        if progress:
            logger.info("Download complete. Extracting...")

        # Extract
        if archive_type == "zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(target_dir)
        elif archive_type in ("tar.gz", "tar"):
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(target_dir)

        # Clean up archive
        archive_path.unlink()

        # Find the dataset root
        dataset_root = _find_metadata_csv(target_dir, config.metadata_csv)
        if dataset_root is None:
            raise FileNotFoundError(
                f"After extraction, could not find '{config.metadata_csv}' in {target_dir}. "
                "The archive structure may be unexpected."
            )

        if progress:
            logger.info("Dataset extracted to %s", dataset_root)

        return dataset_root

    finally:
        # Remove lock file
        if lock_file.exists():
            lock_file.unlink()


def resolve_data_path(
    data_path: Path | str | None,
    config: DatasetConfig,
    auto_download: bool = True,
) -> Path:
    """Resolve the local path to a dataset.

    1. If data_path is provided and exists, return it as Path
    2. If data_path is None and auto_download is True, download
    3. If data_path is None and ~/.ecgbench/datasets/<slug>/ exists, return that
    4. Otherwise raise FileNotFoundError

    This is the entry point all other modules use to locate dataset files.

    Args:
        data_path: Explicit path to dataset root, or None for auto-resolution
        config: DatasetConfig for the dataset
        auto_download: Whether to attempt auto-download if path not found

    Returns:
        Path to the dataset root directory

    Raises:
        FileNotFoundError: if the dataset can't be found or downloaded
    """
    if data_path is not None:
        data_path = Path(data_path)
        if data_path.exists():
            return data_path
        raise FileNotFoundError(
            f"Dataset path does not exist: {data_path}"
        )

    # Check default cache location
    cache_dir = _DEFAULT_CACHE_DIR / config.slug
    if cache_dir.exists():
        dataset_root = _find_metadata_csv(cache_dir, config.metadata_csv)
        if dataset_root:
            return dataset_root

    # Auto-download
    if auto_download and config.download_url:
        return download_dataset(config)

    # Nothing found
    msg = f"Dataset '{config.slug}' not found locally."
    if config.download_url:
        msg += " Pass auto_download=True or run download_dataset() to download."
    else:
        msg += (
            f" No download URL configured. Download manually from {config.url} "
            f"and pass the path via data_path."
        )
    raise FileNotFoundError(msg)
