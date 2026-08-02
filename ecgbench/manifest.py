"""
Split manifests: how a split is reproduced when it cannot be published.

Fold CSVs are identifiers only, but for a credentialed or restricted dataset
those identifiers are still derived from data under a use agreement, and
ECGBench's HuggingFace repo is public and ungated. Such datasets set
``publish_fold_csvs: false`` in their config and are distributed as a *recipe*
instead of as data: the user regenerates the split on their own copy, and this
manifest is what proves the result is the canonical one rather than merely
plausible.

A manifest records four things:

- the **seed and fold count**, read from the split result rather than assumed,
  so the partition is a pure function of documented inputs;
- a **SHA-256 of every input file** the split was computed from (the metadata
  table, and the label source when it is a separate file). This is the part that
  matters most in practice: a split is only reproducible if the input is
  byte-identical, and local copies do get filtered. We hit exactly that with
  MIMIC-IV-ECG, where a local ``machine_measurements.csv`` had been reduced to
  789,481 of 800,035 rows — regenerating from it silently yields different folds;
- the **record counts** for both versions;
- a **fold digest** — a SHA-256 over the whole ``record_id,fold`` mapping in a
  canonical order. Two runs agree on this digest if and only if they produced the
  same partition, so a one-line comparison replaces trusting the procedure.

``ecgbench splits`` writes ``manifest.json`` into the output directory for every
dataset. For datasets that ECGBench cannot publish, a reference copy is shipped
in the package under ``ecgbench/data/manifests/`` so users can verify without
network access.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

#: Reference manifests shipped with the package, for datasets whose splits are
#: not published and therefore have to be regenerated and checked locally.
MANIFESTS_DIR = Path(__file__).parent / "data" / "manifests"

#: Bump when the digest definition changes, so an old manifest cannot be
#: compared against a new digest and quietly disagree.
DIGEST_VERSION = 1


class ManifestMismatchError(RuntimeError):
    """A locally generated split does not match the reference manifest."""


def fold_digest(df: pd.DataFrame, record_id_column: str, fold_column: str = "fold") -> str:
    """Return a SHA-256 over the record-to-fold mapping, order-independent.

    Rows are sorted by record identifier and rendered as ``id,fold`` lines, so
    the digest depends on the partition and nothing else — not on row order, not
    on which columns happen to be present, and not on CSV formatting.
    """
    missing = [c for c in (record_id_column, fold_column) if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot digest folds: missing column(s) {missing}")

    pairs = (
        df[[record_id_column, fold_column]]
        .astype({record_id_column: str, fold_column: int})
        .sort_values(record_id_column, kind="stable")
    )
    payload = "\n".join(
        f"{rid},{fold}" for rid, fold in zip(pairs[record_id_column], pairs[fold_column])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    """SHA-256 of a file, streamed so large metadata tables do not load fully."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_digests(data_path: Path, config: DatasetConfig) -> dict[str, dict[str, Any]]:
    """Checksum every input the split is computed from.

    That is the dataset's metadata table plus, when the labels live in a separate
    file, the label source. Files that do not exist are recorded as absent rather
    than skipped, so a manifest never silently omits an input.
    """
    names = [config.metadata_csv]
    if config.labels and config.labels.source_csv:
        names.append(config.labels.source_csv)

    digests: dict[str, dict[str, Any]] = {}
    for name in dict.fromkeys(n for n in names if n):
        path = Path(data_path) / name
        if not path.exists():
            digests[name] = {"present": False}
            continue
        entry: dict[str, Any] = {
            "present": True,
            "sha256": file_digest(path),
            "bytes": path.stat().st_size,
        }
        try:
            entry["rows"] = int(len(pd.read_csv(path, usecols=[0])))
        except Exception:  # a non-CSV or unreadable input still gets a checksum
            pass
        digests[name] = entry
    return digests


def build_manifest(
    config: DatasetConfig,
    data_path: Path,
    original_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    n_folds: int,
    random_state: int | None,
) -> dict[str, Any]:
    """Assemble the manifest for one `ecgbench splits` run."""
    from ecgbench import __version__ as ecgbench_version

    return {
        "dataset": config.slug,
        "dataset_version": config.version,
        "ecgbench_version": ecgbench_version,
        "digest_version": DIGEST_VERSION,
        "publish_fold_csvs": config.publish_fold_csvs,
        "split": {
            "n_folds": n_folds,
            "random_state": random_state,
            "record_id_column": config.record_id_column,
            "patient_id_column": config.patient_id_column,
            "grouped_by_patient": bool(config.patient_id_column),
        },
        "inputs": input_digests(Path(data_path), config),
        "records": {"original": int(len(original_df)), "clean": int(len(clean_df))},
        "fold_digest": {
            "original": fold_digest(original_df, config.record_id_column),
            "clean": fold_digest(clean_df, config.record_id_column),
        },
        "command": (
            f"ecgbench splits --dataset {config.slug} " f"--data-path /path/to/{config.slug}/"
        ),
    }


def load_reference_manifest(slug: str) -> dict[str, Any] | None:
    """Return the manifest shipped with the package for ``slug``, if any."""
    path = MANIFESTS_DIR / f"{slug}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def verify_splits(slug: str, output_dir: Path | str) -> dict[str, Any]:
    """Check a locally generated split against the manifest shipped for ``slug``.

    Args:
        slug: dataset slug, e.g. ``"mimic_iv_ecg"``.
        output_dir: the directory ``ecgbench splits`` wrote, i.e.
            ``output/<slug>/``.

    Returns:
        A report dict with ``ok`` plus per-version digest comparisons.

    Raises:
        FileNotFoundError: no reference manifest ships for this dataset, or the
            local run is missing.
        ManifestMismatchError: the local split differs from the canonical one.
    """
    reference = load_reference_manifest(slug)
    if reference is None:
        raise FileNotFoundError(
            f"No reference manifest ships for '{slug}'. Manifests exist for datasets "
            "whose fold CSVs cannot be published; everything else is on the "
            "HuggingFace Hub and needs no local verification."
        )

    output_dir = Path(output_dir)
    local_path = output_dir / "manifest.json"
    if not local_path.exists():
        raise FileNotFoundError(
            f"No manifest.json in {output_dir}. Run:\n  {reference.get('command', '')}"
        )
    local = json.loads(local_path.read_text(encoding="utf-8"))

    if local.get("digest_version") != reference.get("digest_version"):
        raise ManifestMismatchError(
            f"Digest version differs (local {local.get('digest_version')}, reference "
            f"{reference.get('digest_version')}); the two are not comparable. Upgrade "
            "ecgbench and regenerate."
        )

    report: dict[str, Any] = {"dataset": slug, "versions": {}, "inputs": {}, "ok": True}

    for version in ("original", "clean"):
        want = reference["fold_digest"].get(version)
        got = local.get("fold_digest", {}).get(version)
        match = want is not None and want == got
        report["versions"][version] = {"expected": want, "actual": got, "match": match}
        report["ok"] &= match

    # Input mismatches are the usual cause, so report them as the explanation.
    for name, want in reference.get("inputs", {}).items():
        got = local.get("inputs", {}).get(name, {})
        if want.get("sha256") and got.get("sha256") != want.get("sha256"):
            report["inputs"][name] = {
                "expected_sha256": want.get("sha256"),
                "actual_sha256": got.get("sha256"),
                "expected_rows": want.get("rows"),
                "actual_rows": got.get("rows"),
            }

    if not report["ok"]:
        detail = ""
        if report["inputs"]:
            names = ", ".join(report["inputs"])
            detail = (
                f" The input file(s) {names} differ from the reference, which is the "
                "usual cause: a filtered or reissued copy produces a different "
                "partition. Verify your download against the provider's checksums."
            )
        raise ManifestMismatchError(
            f"Local splits for '{slug}' do not match the reference manifest.{detail}"
        )

    logger.info("Local splits for '%s' match the reference manifest", slug)
    return report
