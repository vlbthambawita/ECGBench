#!/usr/bin/env python3
"""
Standalone Croissant metadata generation.

Usage:
    python scripts/generate_croissant.py --dataset ptbxl --splits-dir output/ptbxl/clean/
    python scripts/generate_croissant.py --dataset ptbxl --splits-dir output/ptbxl/clean/ \
        --output output/ptbxl/croissant.json
"""

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Croissant 1.1 JSON-LD metadata for an ECG dataset"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset slug (e.g., 'ptbxl')",
    )
    parser.add_argument(
        "--splits-dir", required=True,
        help="Path to the splits version directory (e.g., output/ptbxl/clean/)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for croissant.json. Defaults to splits-dir/../croissant.json",
    )
    parser.add_argument(
        "--version", default="clean", choices=["clean", "original"],
        help="Dataset version (default: clean)",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate the generated Croissant file after saving",
    )
    args = parser.parse_args()

    from pathlib import Path

    from ecgbench.config import load_config
    from ecgbench.croissant import save_croissant, validate_croissant

    config = load_config(args.dataset)
    splits_dir = Path(args.splits_dir)
    output_path = Path(args.output) if args.output else None

    logger.info("Generating Croissant metadata for '%s' (%s)", config.name, args.version)
    saved_path = save_croissant(config, splits_dir, output_path, version=args.version)
    logger.info("Saved to %s", saved_path)

    if args.validate:
        logger.info("Validating...")
        is_valid, errors = validate_croissant(saved_path)
        if is_valid:
            logger.info("Validation passed")
        else:
            logger.error("Validation failed:")
            for err in errors:
                logger.error("  %s", err)


if __name__ == "__main__":
    main()
