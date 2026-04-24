"""Root ``ecgbench`` CLI dispatcher."""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from ecgbench.cli import croissant, splits, upload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ecgbench",
        description=(
            "ECGBench: reproducible ECG benchmark datasets with standardised "
            "splits, validation, and Croissant metadata."
        ),
    )
    try:
        from ecgbench import __version__
    except ImportError:
        __version__ = "0.0.0.dev0"
    parser.add_argument("--version", action="version", version=f"ecgbench {__version__}")

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        metavar="<command>",
        required=True,
    )

    splits.add_subparser(subparsers)
    croissant.add_subparser(subparsers)
    upload.add_subparser(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``ecgbench`` console script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
