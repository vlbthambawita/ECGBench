"""``ecgbench metadata build [--check] [--output DIR]`` and ``ecgbench metadata snapshot``.

``build`` rebuilds the derived metadata files — ``metadata.json`` (committed) and
``metadata.sqlite`` (generated) — from the catalogue front matter, the configs
and the committed snapshots, or with ``--check`` verifies that the committed
export still matches a fresh build and exits 1 with the differing dataset ids
when it does not. That check is what CI and a pre-commit hook run; the packaging
hook in ``hatch_build.py`` runs the build itself.

``snapshot`` derives ``ecgbench/data/snapshots/<slug>.json`` from an
``output/<slug>/`` tree that ``ecgbench splits`` wrote: the record counts, fold
digests, seed and per-check failure counts, and never a record identifier. Run
it after a split run, then ``build``.
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
from ecgbench.metadata.snapshot import REPORT_NAME, SnapshotError, write_snapshot

logger = logging.getLogger(__name__)

#: Where ``ecgbench splits`` writes by default, one directory per dataset slug.
DEFAULT_OUTPUT_ROOT = Path("output")


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


def run_metadata_snapshot(
    dataset: str | None = None,
    output_dir: Path | str | None = None,
    dest: Path | str | None = None,
    output_root: Path | str | None = None,
) -> list[Path]:
    """Write snapshots from split-run output trees.

    Args:
        dataset: Slug whose ``<output_root>/<dataset>/`` tree to snapshot.
        output_dir: The tree itself, when it is not under ``output_root``.
        dest: Target file for a single snapshot; defaults to
            ``ecgbench/data/snapshots/<slug>.json``.
        output_root: Parent of the per-dataset trees (default ``output/``). With
            neither ``dataset`` nor ``output_dir``, every subdirectory holding a
            validation report is snapshotted.

    Returns:
        The files written.

    Raises:
        SnapshotError: see :func:`ecgbench.metadata.snapshot.write_snapshot`.
        FileNotFoundError: no tree to snapshot.
    """
    root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
    if output_dir is not None:
        return [write_snapshot(output_dir, dest)]
    if dataset is not None:
        return [write_snapshot(root / dataset, dest)]
    trees = sorted(p.parent for p in root.glob(f"*/{REPORT_NAME}"))
    if not trees:
        raise FileNotFoundError(f"no */{REPORT_NAME} under {root}; nothing to snapshot")
    if dest is not None:
        raise ValueError("--dest applies to a single snapshot; omit it when sweeping a root")
    written = []
    for tree in trees:
        written.append(write_snapshot(tree))
        logger.info("snapshot %s", written[-1])
    return written


def _cli_snapshot(args: argparse.Namespace) -> int:
    if not (args.dataset or args.output_dir or args.all):
        print("ecgbench: give --dataset, --output-dir or --all", file=sys.stderr)
        return 2
    try:
        written = run_metadata_snapshot(
            dataset=args.dataset,
            output_dir=args.output_dir,
            dest=args.dest,
            output_root=args.output_root,
        )
    except (SnapshotError, FileNotFoundError, ValueError) as exc:
        print(f"ecgbench: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(f"wrote {path}")
    print("now run: ecgbench metadata build")
    return 0


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

    snapshot = sub.add_parser(
        "snapshot",
        help="Derive ecgbench/data/snapshots/<slug>.json from an output/<slug>/ tree",
        description=(
            "Summarise a split run's manifest.json and validation_report.json into a "
            "committed, id-free snapshot that the metadata build turns into facts with "
            "manifest / validation_report provenance. Run `ecgbench metadata build` after."
        ),
    )
    snapshot.add_argument("--dataset", default=None, help="Slug of the output/<slug>/ tree")
    snapshot.add_argument(
        "--output-dir", default=None, help="The tree itself, when not under --output-root"
    )
    snapshot.add_argument(
        "--all", action="store_true", help="Snapshot every tree under --output-root"
    )
    snapshot.add_argument(
        "--output-root",
        default=None,
        help=f"Parent of the per-dataset trees (default: {DEFAULT_OUTPUT_ROOT}/)",
    )
    snapshot.add_argument(
        "--dest", default=None, help="Target file (default: ecgbench/data/snapshots/<slug>.json)"
    )
    snapshot.set_defaults(func=_cli_snapshot)
    return p
