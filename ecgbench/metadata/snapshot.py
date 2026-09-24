"""Artefact snapshots: the recomputed numbers of a split run, without the record ids.

``ecgbench splits`` leaves two summaries in ``output/<slug>/``: ``manifest.json``
(fold count, seed, a SHA-256 per input file, record counts and a ``fold_digest``
over the whole record-to-fold mapping) and ``validation_report.json`` (record
counts before and after validation and the per-check failure counts, followed by
the list of excluded record ids). A **snapshot** is the id-free subset of both,
committed as ``ecgbench/data/snapshots/<slug>.json`` so the metadata build can
turn it into facts with ``manifest`` and ``validation_report`` provenance - the
sources that outrank the catalogue and the config in ``SOURCE_PRECEDENCE``.

Two guarantees hold for every snapshot, and :func:`check_id_free` enforces the
first at write time and in the tests:

- **No record identifiers.** The validation report's ``excluded_records`` block
  is the one part of either file that names records, and it is never copied; a
  snapshot carries counts and digests only, so it can ship for a dataset whose
  fold CSVs may not (``publish_fold_csvs: false``).
- **One run, not a mixture.** The manifest and the report must agree on the
  record counts, or :func:`build_snapshot` refuses: a tree holding a report from
  one run and a manifest from another would produce a snapshot describing
  neither.

This module is standard-library only, because the metadata build runs inside the
packaging hook with pyyaml alone; ``ecgbench.manifest`` imports pandas and is
therefore never imported here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Layout version of a snapshot file; bump on an incompatible change.
SNAPSHOT_VERSION = 1

#: Where the committed snapshots live, one ``<slug>.json`` per dataset.
SNAPSHOTS_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots"

#: The shipped reference manifests of the datasets whose fold CSVs are not published.
_MANIFESTS_DIR = Path(__file__).resolve().parent.parent / "data" / "manifests"

MANIFEST_NAME = "manifest.json"
REPORT_NAME = "validation_report.json"

#: Keys that name records rather than count them. None may appear in a snapshot.
RECORD_LIST_KEYS: frozenset[str] = frozenset(
    {"excluded_records", "record_id", "record_ids", "records_list", "folds", "fold_assignments"}
)

#: A list longer than this is treated as a record list, whatever it is called.
MAX_LIST_LENGTH = 32


class SnapshotError(ValueError):
    """A snapshot could not be built, or does not hold up to its own guarantees."""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def build_snapshot(output_dir: Path | str) -> dict[str, Any]:
    """Derive a snapshot from an ``output/<slug>/`` tree.

    ``validation_report.json`` is required; ``manifest.json`` is used when present
    (runs before ECGBench 0.20 wrote none), and without it the split half -
    ``fold_digest``, ``n_folds``, ``random_state``, ``inputs`` - is ``None`` or
    empty and ``sources`` says so.

    Args:
        output_dir: The directory ``ecgbench splits`` wrote into.

    Returns:
        The snapshot as a JSON-ready mapping, already checked to be id-free.

    Raises:
        SnapshotError: the report is missing, the two files name different
            datasets or disagree on the record counts, or the result would hold a
            record identifier.
    """
    output_dir = Path(output_dir)
    report_path = output_dir / REPORT_NAME
    if not report_path.is_file():
        raise SnapshotError(
            f"{report_path} does not exist; a snapshot needs the validation report that "
            f"`ecgbench splits` writes into the output directory."
        )
    report = _read_json(report_path)
    manifest_path = output_dir / MANIFEST_NAME
    manifest = _read_json(manifest_path) if manifest_path.is_file() else None

    dataset = report["dataset"]
    records = {
        "original": int(report["original"]["total_records"]),
        "clean": int(report["clean"]["total_records"]),
    }
    if manifest is not None:
        if manifest["dataset"] != dataset:
            raise SnapshotError(
                f"{manifest_path} is for {manifest['dataset']!r} but {report_path} is for "
                f"{dataset!r}"
            )
        manifest_records = {k: int(v) for k, v in manifest["records"].items()}
        if manifest_records != records:
            raise SnapshotError(
                f"{output_dir}: manifest.json counts {manifest_records} but "
                f"validation_report.json counts {records}; the two files are from different "
                "runs. Re-run `ecgbench splits` so they describe one partition."
            )

    split = manifest.get("split", {}) if manifest else {}
    snapshot: dict[str, Any] = {
        "snapshot_version": SNAPSHOT_VERSION,
        "dataset": dataset,
        "dataset_version": (
            manifest.get("dataset_version") if manifest else report.get("source_version")
        ),
        "ecgbench_version": (manifest or report).get("ecgbench_version"),
        "created": report.get("validated_at"),
        "sources": [MANIFEST_NAME, REPORT_NAME] if manifest else [REPORT_NAME],
        "records": records,
        "records_excluded": int(
            report["clean"].get("removed", records["original"] - records["clean"])
        ),
        "quality_checks": [
            {"check": str(q["check"]), "records_failed": int(q["records_failed"])}
            for q in report.get("quality_checks", ())
        ],
        "n_folds": split.get("n_folds"),
        "random_state": split.get("random_state"),
        "grouped_by_patient": split.get("grouped_by_patient"),
        "fold_digest": (
            {k: str(v) for k, v in manifest["fold_digest"].items()} if manifest else None
        ),
        "digest_version": manifest.get("digest_version") if manifest else None,
        "inputs": (
            {
                name: entry["sha256"]
                for name, entry in manifest.get("inputs", {}).items()
                if isinstance(entry, dict) and entry.get("sha256")
            }
            if manifest
            else {}
        ),
    }
    check_id_free(snapshot)
    return snapshot


def check_id_free(snapshot: dict[str, Any]) -> None:
    """Raise :class:`SnapshotError` if ``snapshot`` looks like it names records.

    A record list can hide under a familiar key (``excluded_records``), as any
    long list, or as a list of objects - the shape a per-record table takes in
    JSON. All three are refused, at every depth.
    """

    def walk(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                if key in RECORD_LIST_KEYS:
                    raise SnapshotError(f"snapshot holds a record list under {path}/{key}")
                walk(inner, f"{path}/{key}")
        elif isinstance(value, list):
            if len(value) > MAX_LIST_LENGTH:
                raise SnapshotError(
                    f"snapshot list {path} has {len(value)} entries; a snapshot carries counts "
                    "and digests, never one entry per record"
                )
            for i, inner in enumerate(value):
                if isinstance(inner, dict) and set(inner) - {"check", "records_failed"}:
                    raise SnapshotError(f"snapshot list {path} holds per-item objects")
                walk(inner, f"{path}[{i}]")

    walk(snapshot, "")


def reference_manifest_path(dataset: str) -> Path:
    """Path of the shipped reference manifest for ``dataset`` (which may not exist)."""
    return _MANIFESTS_DIR / f"{dataset}.json"


def write_snapshot(output_dir: Path | str, dest: Path | str | None = None) -> Path:
    """Build the snapshot for ``output_dir`` and write it as JSON.

    Args:
        output_dir: The ``output/<slug>/`` tree.
        dest: Target file; defaults to ``ecgbench/data/snapshots/<slug>.json``.

    Returns:
        The path written.

    Raises:
        SnapshotError: see :func:`build_snapshot`; also when a shipped reference
            manifest exists for the dataset and its fold digests differ from the
            tree's, because then the tree is not the canonical partition.
    """
    snapshot = build_snapshot(output_dir)
    dataset = snapshot["dataset"]
    reference = reference_manifest_path(dataset)
    if reference.is_file() and snapshot["fold_digest"] is not None:
        expected = _read_json(reference)["fold_digest"]
        if expected != snapshot["fold_digest"]:
            raise SnapshotError(
                f"{output_dir}: fold digests {snapshot['fold_digest']} differ from the shipped "
                f"reference manifest {reference.name} ({expected}); this tree is not the "
                "canonical partition - see ecgbench.manifest.verify_splits."
            )
    target = Path(dest) if dest is not None else SNAPSHOTS_DIR / f"{dataset}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("wrote snapshot for %s to %s", dataset, target)
    return target


def load_snapshot(path: Path | str) -> dict[str, Any]:
    """Read one snapshot file, checking its version and its id-free guarantee."""
    path = Path(path)
    snapshot = _read_json(path)
    version = snapshot.get("snapshot_version")
    if version != SNAPSHOT_VERSION:
        raise SnapshotError(
            f"{path}: snapshot_version {version!r}, expected {SNAPSHOT_VERSION}"
        )
    if snapshot.get("dataset") != path.stem:
        raise SnapshotError(
            f"{path}: file is named {path.stem!r} but describes {snapshot.get('dataset')!r}"
        )
    check_id_free(snapshot)
    return snapshot


def load_snapshots(directory: Path | str = SNAPSHOTS_DIR) -> dict[str, dict[str, Any]]:
    """Every snapshot under ``directory``, keyed by dataset slug; empty if none."""
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    return {path.stem: load_snapshot(path) for path in sorted(directory.glob("*.json"))}
