#!/usr/bin/env python3
"""
Upload original/ and clean/ fold CSVs to HuggingFace Hub.

Usage:
    python scripts/upload_to_huggingface.py --data-dir output/ --datasets ptbxl
    python scripts/upload_to_huggingface.py --data-dir output/ --datasets ptbxl chapman_shaoxing \
        --hf-repo-id vlbthambawita/ECGBench
"""

import argparse
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _get_hf_token() -> str:
    """Get HuggingFace token from environment or .env file."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token

    # Try loading from .env
    try:
        from dotenv import load_dotenv

        load_dotenv()
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    except ImportError:
        pass

    if not token:
        raise ValueError(
            "HF_TOKEN not found. Set it as an environment variable or in a .env file."
        )
    return token


def main():
    parser = argparse.ArgumentParser(
        description="Upload ECGBench fold CSVs to HuggingFace Hub"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Root output directory containing per-dataset subdirectories",
    )
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        help="Dataset slugs to upload (e.g., ptbxl chapman_shaoxing)",
    )
    parser.add_argument(
        "--hf-repo-id", default="vlbthambawita/ECGBench",
        help="HuggingFace repository ID (default: vlbthambawita/ECGBench)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files that would be uploaded without uploading",
    )
    args = parser.parse_args()

    from pathlib import Path

    from huggingface_hub import HfApi

    data_dir = Path(args.data_dir)
    token = _get_hf_token()
    api = HfApi(token=token)

    for dataset_slug in args.datasets:
        dataset_dir = data_dir / dataset_slug
        if not dataset_dir.exists():
            logger.warning("Directory not found: %s, skipping", dataset_dir)
            continue

        logger.info("Processing %s...", dataset_slug)

        # Collect all files to upload
        files_to_upload = []
        for version in ("original", "clean"):
            version_dir = dataset_dir / version
            if not version_dir.exists():
                logger.warning("  %s/ not found, skipping", version)
                continue
            for csv_file in version_dir.rglob("*.csv"):
                rel_path = csv_file.relative_to(data_dir)
                files_to_upload.append((csv_file, str(rel_path)))

        # Add validation report and croissant if present
        for extra_file in ("validation_report.json", "croissant.json"):
            extra_path = dataset_dir / extra_file
            if extra_path.exists():
                rel_path = extra_path.relative_to(data_dir)
                files_to_upload.append((extra_path, str(rel_path)))

        if not files_to_upload:
            logger.warning("  No files found to upload for %s", dataset_slug)
            continue

        logger.info("  Found %d files to upload", len(files_to_upload))

        if args.dry_run:
            for local_path, remote_path in files_to_upload:
                size_kb = local_path.stat().st_size / 1024
                logger.info("  [DRY RUN] %s (%.1f KB)", remote_path, size_kb)
            continue

        # Upload files
        for local_path, remote_path in files_to_upload:
            logger.info("  Uploading %s", remote_path)
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=args.hf_repo_id,
                repo_type="dataset",
            )

        logger.info("  Done uploading %s (%d files)", dataset_slug, len(files_to_upload))

    logger.info("Upload complete!")


if __name__ == "__main__":
    main()
