"""Standalone Croissant metadata generation subcommand."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_croissant(
    dataset: str,
    splits_dir: Path | str,
    output: Path | str | None = None,
    version: str = "clean",
    validate: bool = False,
) -> Path:
    """Generate (and optionally validate) Croissant 1.1 JSON-LD for a dataset.

    Returns the path to the saved ``croissant.json`` file.
    """
    if version not in ("clean", "original"):
        raise ValueError(f"version must be 'clean' or 'original', got {version!r}")

    from ecgbench.config import load_config
    from ecgbench.croissant import save_croissant, validate_croissant

    config = load_config(dataset)
    splits_path = Path(splits_dir)
    output_path = Path(output) if output else None

    logger.info("Generating Croissant metadata for '%s' (%s)", config.name, version)
    saved_path = save_croissant(config, splits_path, output_path, version=version)
    logger.info("Saved to %s", saved_path)

    if validate:
        logger.info("Validating...")
        is_valid, errors = validate_croissant(saved_path)
        if is_valid:
            logger.info("Validation passed")
        else:
            logger.error("Validation failed:")
            for err in errors:
                logger.error("  %s", err)
            raise RuntimeError(f"Croissant validation failed: {errors}")

    return saved_path


def _cli_run(args: argparse.Namespace) -> int:
    try:
        run_croissant(
            dataset=args.dataset,
            splits_dir=args.splits_dir,
            output=args.output,
            version=args.version,
            validate=args.validate,
        )
    except RuntimeError:
        return 1
    return 0


def add_subparser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "croissant",
        help="Generate Croissant 1.1 JSON-LD metadata for an existing splits directory",
        description="Standalone Croissant metadata generation.",
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Dataset slug (e.g., 'ptbxl')",
    )
    p.add_argument(
        "--splits-dir",
        required=True,
        help="Path to the splits version directory (e.g., output/ptbxl/clean/)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output path for croissant.json. Defaults to splits-dir/croissant.json",
    )
    p.add_argument(
        "--version",
        default="clean",
        choices=["clean", "original"],
        help="Dataset version (default: clean)",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Validate the generated Croissant file after saving",
    )
    p.set_defaults(func=_cli_run)
    return p
