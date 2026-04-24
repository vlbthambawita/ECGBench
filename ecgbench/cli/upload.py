"""Upload fold CSVs and metadata to HuggingFace Hub."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_hf_token(token: str | None) -> str:
    """Resolve an HF token from arg, env vars, or a cwd .env file."""
    if token:
        return token

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if env_token:
        return env_token

    try:
        from dotenv import load_dotenv

        load_dotenv()  # loads .env from current working directory if present
        env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    except ImportError:
        pass

    if not env_token:
        raise ValueError(
            "HF_TOKEN not found. Pass token=..., set the HF_TOKEN environment "
            "variable, or place it in a .env file in the current directory."
        )
    return env_token


def run_upload(
    data_dir: Path | str,
    datasets: list[str],
    hf_repo_id: str = "vlbthambawita/ECGBench",
    dry_run: bool = False,
    token: str | None = None,
) -> dict[str, int]:
    """Upload per-dataset fold CSVs and metadata to HuggingFace Hub.

    Returns a mapping of dataset slug -> number of files uploaded (or that would
    have been uploaded, for ``dry_run=True``).
    """
    from huggingface_hub import HfApi

    data_root = Path(data_dir)
    resolved_token = _resolve_hf_token(token)
    api = HfApi(token=resolved_token)

    uploaded: dict[str, int] = {}

    for dataset_slug in datasets:
        dataset_dir = data_root / dataset_slug
        if not dataset_dir.exists():
            logger.warning("Directory not found: %s, skipping", dataset_dir)
            uploaded[dataset_slug] = 0
            continue

        logger.info("Processing %s...", dataset_slug)

        files_to_upload: list[tuple[Path, str]] = []
        for version in ("original", "clean"):
            version_dir = dataset_dir / version
            if not version_dir.exists():
                logger.warning("  %s/ not found, skipping", version)
                continue
            for csv_file in version_dir.rglob("*.csv"):
                rel_path = csv_file.relative_to(data_root)
                files_to_upload.append((csv_file, str(rel_path)))

        for extra_file in ("validation_report.json", "croissant.json"):
            extra_path = dataset_dir / extra_file
            if extra_path.exists():
                rel_path = extra_path.relative_to(data_root)
                files_to_upload.append((extra_path, str(rel_path)))

        if not files_to_upload:
            logger.warning("  No files found to upload for %s", dataset_slug)
            uploaded[dataset_slug] = 0
            continue

        logger.info("  Found %d files to upload", len(files_to_upload))

        if dry_run:
            for local_path, remote_path in files_to_upload:
                size_kb = local_path.stat().st_size / 1024
                logger.info("  [DRY RUN] %s (%.1f KB)", remote_path, size_kb)
            uploaded[dataset_slug] = len(files_to_upload)
            continue

        for local_path, remote_path in files_to_upload:
            logger.info("  Uploading %s", remote_path)
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=hf_repo_id,
                repo_type="dataset",
            )

        logger.info("  Done uploading %s (%d files)", dataset_slug, len(files_to_upload))
        uploaded[dataset_slug] = len(files_to_upload)

    logger.info("Upload complete!")
    return uploaded


def _cli_run(args: argparse.Namespace) -> int:
    run_upload(
        data_dir=args.data_dir,
        datasets=args.datasets,
        hf_repo_id=args.hf_repo_id,
        dry_run=args.dry_run,
    )
    return 0


def add_subparser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "upload",
        help="Upload fold CSVs and metadata for one or more datasets to HuggingFace Hub",
        description="Upload ECGBench fold CSVs to HuggingFace Hub.",
    )
    p.add_argument(
        "--data-dir",
        required=True,
        help="Root output directory containing per-dataset subdirectories",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset slugs to upload (e.g., ptbxl chapman_shaoxing)",
    )
    p.add_argument(
        "--hf-repo-id",
        default="vlbthambawita/ECGBench",
        help="HuggingFace repository ID (default: vlbthambawita/ECGBench)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be uploaded without uploading",
    )
    p.set_defaults(func=_cli_run)
    return p
