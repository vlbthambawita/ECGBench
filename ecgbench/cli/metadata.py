"""``ecgbench metadata build [--check] [--output DIR]``.

Rebuilds the derived metadata files — ``metadata.json`` (committed) and
``metadata.sqlite`` (generated) — from the catalogue front matter and the
configs, or with ``--check`` verifies that the committed export still matches a
fresh build and exits 1 with the differing dataset ids when it does not. That
check is what CI and a pre-commit hook run; the packaging hook in
``hatch_build.py`` runs the build itself.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ecgbench.metadata.build import (
    DEFAULT_JSON_PATH,
    BuildResult,
    MetadataBuildError,
    ModelDiff,
    build_all,
    build_model,
    content_digest,
    diff_exports,
)

logger = logging.getLogger(__name__)


def run_metadata_check(json_path: Path | str | None = None) -> ModelDiff:
    """Compare the committed export against a fresh build.

    Args:
        json_path: Export to check; defaults to the bundled ``metadata.json``.

    Returns:
        A ``ModelDiff`` that is falsy when the export is current. A missing
        export counts as every dataset added.
    """
    target = Path(json_path) if json_path is not None else DEFAULT_JSON_PATH
    model = build_model()
    if not target.is_file():
        return ModelDiff(added=tuple(m.dataset_id for m in model), removed=(), changed=())
    with target.open(encoding="utf-8") as fh:
        document = json.load(fh)
    if document.get("content_digest") == content_digest(model):
        return ModelDiff((), (), ())
    return diff_exports(document, model)


def run_metadata_build(output_dir: Path | str | None = None) -> BuildResult:
    """Rebuild ``metadata.json``, ``metadata.sqlite`` and the source fingerprint.

    Args:
        output_dir: Where to write; defaults to ``ecgbench/data/``.
    """
    result = build_all(output_dir)
    logger.info(
        "metadata: %s %s, index %s (%s), digest %s",
        "wrote" if result.json_written else "unchanged",
        result.json_path,
        result.sqlite_path,
        result.fts,
        result.content_digest,
    )
    return result


def _cli_run(args: argparse.Namespace) -> int:
    try:
        if args.check:
            target = Path(args.output) / DEFAULT_JSON_PATH.name if args.output else None
            diff = run_metadata_check(target)
            if diff:
                print(f"metadata.json is stale: {diff.summary()}", file=sys.stderr)
                print("regenerate it with: ecgbench metadata build", file=sys.stderr)
                return 1
            print("metadata.json is up to date")
            return 0
        result = run_metadata_build(args.output)
        print(
            f"{'wrote' if result.json_written else 'unchanged'} {result.json_path}\n"
            f"wrote {result.sqlite_path} (fts: {result.fts})\n"
            f"content_digest {result.content_digest}"
        )
        return 0
    except MetadataBuildError as exc:
        print(f"ecgbench: {exc}", file=sys.stderr)
        return 1


def add_subparser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "metadata",
        help="Build or check the derived metadata files (metadata.json, metadata.sqlite)",
        description=(
            "Maintenance of the compiled metadata layer. The sources are the catalogue "
            "front matter and the YAML configs; everything under ecgbench/data/metadata.* "
            "is derived from them."
        ),
    )
    sub = p.add_subparsers(dest="metadata_command", metavar="<action>", required=True)
    build = sub.add_parser(
        "build",
        help="Rebuild metadata.json and metadata.sqlite from the sources",
        description="Rebuild the derived metadata files, or verify them with --check.",
    )
    build.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if the committed metadata.json differs from a fresh build",
    )
    build.add_argument(
        "--output",
        default=None,
        help="Directory to write into (default: ecgbench/data/ inside the package)",
    )
    build.set_defaults(func=_cli_run)
    return p
