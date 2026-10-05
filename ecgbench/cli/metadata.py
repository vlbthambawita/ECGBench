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
from ecgbench.metadata.export import (
    WEBSITE_JSON_PATH,
    to_croissant_collection,
    to_schema_org,
    validate_croissant,
    website_json_current,
    write_website_json,
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


def run_metadata_export(
    croissant: Path | str | None = None,
    schema_org_dir: Path | str | None = None,
    website: Path | str | None = None,
    validate: bool = False,
) -> dict[str, list[Path]]:
    """Write the Croissant collection, per-dataset schema.org blocks, and the website JSON.

    Args:
        croissant: Path of the JSON-LD ``DataCatalog`` of Croissant datasets.
        schema_org_dir: Directory receiving one ``<catalogue-slug>.jsonld`` per dataset.
        website: Path of the website copy (default target when given as ``"-"``:
            ``docs/_data/metadata.json``).
        validate: Validate the Croissant collection with ``mlcroissant`` and raise
            on errors.

    Returns:
        ``{"croissant": [...], "schema_org": [...], "website": [...]}`` paths written.

    Raises:
        RuntimeError: ``validate`` found errors (the message lists them).
        ImportError: ``validate`` without ``mlcroissant`` installed.
    """
    from ecgbench.metadata.store import open_store

    model = tuple(open_store().all())
    written: dict[str, list[Path]] = {"croissant": [], "schema_org": [], "website": []}
    if croissant is not None:
        collection = to_croissant_collection(model)
        if validate:
            errors = validate_croissant(collection)
            if errors:
                raise RuntimeError("Croissant validation failed:\n  " + "\n  ".join(errors))
        target = Path(croissant)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(collection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        written["croissant"].append(target)
    if schema_org_dir is not None:
        directory = Path(schema_org_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for meta in model:
            target = directory / f"{meta.aliases[0]}.jsonld"
            target.write_text(
                json.dumps(to_schema_org(meta), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            written["schema_org"].append(target)
    if website is not None:
        target = WEBSITE_JSON_PATH if str(website) == "-" else Path(website)
        write_website_json(model, target)
        written["website"].append(target)
    return written


def run_metadata_export_check(website: Path | str | None = None) -> bool:
    """Whether the committed website copy matches a fresh export."""
    from ecgbench.metadata.store import open_store

    model = tuple(open_store().all())
    return website_json_current(model, Path(website) if website else WEBSITE_JSON_PATH)


def _cli_export(args: argparse.Namespace) -> int:
    if args.check:
        if run_metadata_export_check(args.website if args.website not in (None, "-") else None):
            print("docs/_data/metadata.json is up to date")
            return 0
        print("docs/_data/metadata.json is stale", file=sys.stderr)
        print("regenerate it with: ecgbench metadata export --website -", file=sys.stderr)
        return 1
    if not (args.croissant or args.schema_org or args.website):
        print(
            "ecgbench: give --croissant PATH, --schema-org DIR and/or --website PATH",
            file=sys.stderr,
        )
        return 2
    try:
        written = run_metadata_export(
            croissant=args.croissant,
            schema_org_dir=args.schema_org,
            website=args.website,
            validate=args.validate,
        )
    except (RuntimeError, ImportError) as exc:
        print(f"ecgbench: {exc}", file=sys.stderr)
        return 1
    for kind, paths in written.items():
        if kind == "schema_org" and paths:
            print(f"wrote {len(paths)} schema.org blocks under {paths[0].parent}")
        else:
            for path in paths:
                print(f"wrote {path}")
    if args.croissant and args.validate:
        print("mlcroissant: no errors")
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

    export = sub.add_parser(
        "export",
        help="Export Croissant 1.1, schema.org JSON-LD, or the website's metadata.json",
        description=(
            "Views over the metadata model: a Croissant 1.1 DataCatalog of every configured "
            "dataset (--croissant), one schema.org Dataset block per dataset (--schema-org), "
            "and the committed docs/_data/metadata.json the dataset pages embed (--website -). "
            "--check verifies the committed website copy, like `metadata build --check`."
        ),
    )
    export.add_argument("--croissant", default=None, metavar="PATH", help="JSON-LD collection")
    export.add_argument(
        "--schema-org", default=None, metavar="DIR", help="One <slug>.jsonld per dataset"
    )
    export.add_argument(
        "--website",
        default=None,
        metavar="PATH",
        help="Website copy; '-' means the committed docs/_data/metadata.json",
    )
    export.add_argument(
        "--validate", action="store_true", help="Validate --croissant output with mlcroissant"
    )
    export.add_argument(
        "--check", action="store_true", help="Exit 1 if the committed website copy is stale"
    )
    export.set_defaults(func=_cli_export)
    return p
