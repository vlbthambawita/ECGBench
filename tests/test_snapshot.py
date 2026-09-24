"""Tests for artefact snapshots (metadata layer, Phase 4).

Two guarantees are pinned. A snapshot never carries a record identifier - the
validation report's ``excluded_records`` block is the one thing it must not copy,
and ``check_id_free`` refuses any list that looks like one. And a snapshot
describes one run: a tree whose manifest and report disagree on the record counts
is refused rather than averaged.

The shipped snapshots are then checked as a set: id-free, schema-valid, matching
the four reference manifests, and feeding the build as ``manifest`` /
``validation_report`` facts that outrank the catalogue.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecgbench.cli import main
from ecgbench.cli.metadata import run_metadata_snapshot
from ecgbench.metadata import build as build_module
from ecgbench.metadata import open_store
from ecgbench.metadata.snapshot import (
    SNAPSHOTS_DIR,
    SnapshotError,
    build_snapshot,
    check_id_free,
    load_snapshot,
    load_snapshots,
    reference_manifest_path,
    write_snapshot,
)

# --------------------------------------------------------------------------- fixtures

MANIFEST = {
    "dataset": "toy",
    "dataset_version": "1.0.0",
    "ecgbench_version": "0.99.0",
    "digest_version": 1,
    "publish_fold_csvs": True,
    "split": {
        "n_folds": 10,
        "random_state": 42,
        "record_id_column": "record_name",
        "patient_id_column": "patient_id",
        "grouped_by_patient": True,
    },
    "inputs": {
        "ecgbench_metadata.csv": {"present": True, "sha256": "ab" * 32, "bytes": 10, "rows": 48},
        "missing.csv": {"present": False},
    },
    "records": {"original": 48, "clean": 45},
    "fold_digest": {"original": "0" * 64, "clean": "1" * 64},
    "command": "ecgbench splits --dataset toy --data-path /path/to/toy/",
}

REPORT = {
    "dataset": "toy",
    "source_version": "1.0.0",
    "ecgbench_version": "0.99.0",
    "validated_at": "2026-08-07T05:42:53.836211+00:00",
    "sampling_rate_validated": 360,
    "original": {"total_records": 48},
    "clean": {"total_records": 45, "removed": 3},
    "quality_checks": [
        {
            "check": "amplitude_outlier",
            "description": "Samples outside physiological range",
            "records_failed": 3,
            "total_issues": 3,
        }
    ],
    "excluded_records": [
        {"record_id": "103", "issues": ["amplitude_outlier:lead_1"]},
        {"record_id": "104", "issues": ["amplitude_outlier:lead_1"]},
        {"record_id": "105", "issues": ["amplitude_outlier:lead_1"]},
    ],
}


def _tree(tmp_path: Path, manifest: dict | None = MANIFEST, report: dict | None = REPORT) -> Path:
    out = tmp_path / "toy"
    out.mkdir()
    if manifest is not None:
        (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if report is not None:
        (out / "validation_report.json").write_text(json.dumps(report), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- building


class TestBuildSnapshot:
    def test_manifest_and_report_are_merged_without_ids(self, tmp_path):
        snapshot = build_snapshot(_tree(tmp_path))
        assert snapshot["dataset"] == "toy"
        assert snapshot["records"] == {"original": 48, "clean": 45}
        assert snapshot["records_excluded"] == 3
        assert snapshot["fold_digest"] == {"original": "0" * 64, "clean": "1" * 64}
        assert snapshot["n_folds"] == 10 and snapshot["random_state"] == 42
        assert snapshot["grouped_by_patient"] is True
        assert snapshot["inputs"] == {"ecgbench_metadata.csv": "ab" * 32}  # absent file dropped
        assert snapshot["quality_checks"] == [{"check": "amplitude_outlier", "records_failed": 3}]
        assert snapshot["created"] == REPORT["validated_at"]
        assert snapshot["sources"] == ["manifest.json", "validation_report.json"]
        assert "excluded_records" not in json.dumps(snapshot)
        assert "103" not in json.dumps(snapshot)

    def test_report_alone_gives_a_validation_only_snapshot(self, tmp_path):
        snapshot = build_snapshot(_tree(tmp_path, manifest=None))
        assert snapshot["sources"] == ["validation_report.json"]
        assert snapshot["fold_digest"] is None
        assert snapshot["n_folds"] is None and snapshot["random_state"] is None
        assert snapshot["inputs"] == {}
        assert snapshot["dataset_version"] == "1.0.0"  # from the report's source_version
        assert snapshot["records"] == {"original": 48, "clean": 45}

    def test_report_is_required(self, tmp_path):
        with pytest.raises(SnapshotError, match="validation_report.json"):
            build_snapshot(_tree(tmp_path, report=None))

    def test_disagreeing_files_are_refused(self, tmp_path):
        manifest = {**MANIFEST, "records": {"original": 48, "clean": 44}}
        with pytest.raises(SnapshotError, match="different runs"):
            build_snapshot(_tree(tmp_path, manifest=manifest))

    def test_files_for_different_datasets_are_refused(self, tmp_path):
        with pytest.raises(SnapshotError, match="is for 'other'"):
            build_snapshot(_tree(tmp_path, manifest={**MANIFEST, "dataset": "other"}))


class TestIdFree:
    def test_a_record_list_key_is_caught_at_any_depth(self):
        with pytest.raises(SnapshotError, match="record list"):
            check_id_free({"a": {"excluded_records": []}})

    def test_a_long_list_is_caught(self):
        with pytest.raises(SnapshotError, match="one entry per record"):
            check_id_free({"ids": list(range(100))})

    def test_a_list_of_objects_is_caught(self):
        with pytest.raises(SnapshotError, match="per-item objects"):
            check_id_free({"rows": [{"record_id": "1"}]})

    def test_quality_checks_pass(self):
        check_id_free({"quality_checks": [{"check": "nan_values", "records_failed": 2}]})


class TestWriteSnapshot:
    def test_writes_to_dest_and_round_trips(self, tmp_path):
        dest = tmp_path / "snapshots" / "toy.json"
        assert write_snapshot(_tree(tmp_path), dest) == dest
        loaded = load_snapshot(dest)
        assert loaded == build_snapshot(tmp_path / "toy")
        assert load_snapshots(dest.parent) == {"toy": loaded}

    def test_filename_must_match_the_dataset(self, tmp_path):
        path = tmp_path / "wrong.json"
        path.write_text(json.dumps(build_snapshot(_tree(tmp_path))), encoding="utf-8")
        with pytest.raises(SnapshotError, match="named 'wrong'"):
            load_snapshot(path)

    def test_version_is_checked(self, tmp_path):
        path = tmp_path / "toy.json"
        path.write_text(
            json.dumps({**build_snapshot(_tree(tmp_path)), "snapshot_version": 99}),
            encoding="utf-8",
        )
        with pytest.raises(SnapshotError, match="snapshot_version 99"):
            load_snapshot(path)

    def test_disagreement_with_a_reference_manifest_is_refused(self, tmp_path, monkeypatch):
        """A tree whose partition differs from the shipped reference is not canonical."""
        reference = tmp_path / "ref" / "toy.json"
        reference.parent.mkdir()
        reference.write_text(
            json.dumps({**MANIFEST, "fold_digest": {"original": "f" * 64, "clean": "1" * 64}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "ecgbench.metadata.snapshot.reference_manifest_path", lambda slug: reference
        )
        with pytest.raises(SnapshotError, match="reference manifest"):
            write_snapshot(_tree(tmp_path), tmp_path / "out.json")

    def test_missing_directory_yields_no_snapshots(self, tmp_path):
        assert load_snapshots(tmp_path / "nope") == {}


class TestCli:
    def test_single_dataset_under_a_root(self, tmp_path, capsys):
        _tree(tmp_path)
        dest = tmp_path / "s" / "toy.json"
        code = main(
            [
                "metadata", "snapshot", "--dataset", "toy",
                "--output-root", str(tmp_path), "--dest", str(dest),
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert str(dest) in out and "ecgbench metadata build" in out
        assert dest.is_file()

    def test_sweep_a_root(self, tmp_path):
        _tree(tmp_path)
        other = tmp_path / "toy2"
        other.mkdir()
        (other / "validation_report.json").write_text(
            json.dumps({**REPORT, "dataset": "toy2"}), encoding="utf-8"
        )
        (tmp_path / "not_a_tree").mkdir()
        target = tmp_path / "snapshots"
        # Sweeping writes into the package dir by default; redirect for the test.
        import ecgbench.metadata.snapshot as snapshot_module

        original = snapshot_module.SNAPSHOTS_DIR
        snapshot_module.SNAPSHOTS_DIR = target
        try:
            written = run_metadata_snapshot(output_root=tmp_path)
        finally:
            snapshot_module.SNAPSHOTS_DIR = original
        assert sorted(p.name for p in written) == ["toy.json", "toy2.json"]
        assert sorted(load_snapshots(target)) == ["toy", "toy2"]

    def test_sweep_with_dest_is_rejected(self, tmp_path):
        _tree(tmp_path)
        with pytest.raises(ValueError, match="--dest"):
            run_metadata_snapshot(output_root=tmp_path, dest=tmp_path / "x.json")

    def test_empty_root_is_an_error(self, tmp_path, capsys):
        assert main(["metadata", "snapshot", "--all", "--output-root", str(tmp_path)]) == 1
        assert "nothing to snapshot" in capsys.readouterr().err

    def test_no_selector_is_a_usage_error(self, capsys):
        assert main(["metadata", "snapshot"]) == 2
        assert "--dataset" in capsys.readouterr().err

    def test_disagreement_exits_nonzero(self, tmp_path, capsys):
        _tree(tmp_path, manifest={**MANIFEST, "records": {"original": 1, "clean": 1}})
        code = main(
            ["metadata", "snapshot", "--output-dir", str(tmp_path / "toy"),
             "--dest", str(tmp_path / "x.json")]
        )
        assert code == 1
        assert "different runs" in capsys.readouterr().err


# --------------------------------------------------------------------------- the shipped set


@pytest.fixture(scope="module")
def shipped() -> dict[str, dict]:
    snapshots = load_snapshots(SNAPSHOTS_DIR)
    assert snapshots, "no committed snapshots"
    return snapshots


class TestShippedSnapshots:
    def test_every_snapshot_is_id_free_and_named_after_its_dataset(self, shipped):
        for slug, snapshot in shipped.items():
            check_id_free(snapshot)
            assert snapshot["dataset"] == slug
            assert snapshot["records"]["clean"] <= snapshot["records"]["original"]
            assert snapshot["records_excluded"] == (
                snapshot["records"]["original"] - snapshot["records"]["clean"]
            )

    def test_every_snapshot_names_a_config(self, shipped):
        from ecgbench.config import list_available_configs

        assert set(shipped) <= set(list_available_configs())

    def test_snapshots_validate_against_the_schema(self, shipped):
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads(build_module.SCHEMA_PATH.read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(
            {"$ref": "#/$defs/snapshot", "$defs": schema["$defs"]}
        )
        for slug, snapshot in shipped.items():
            errors = sorted(validator.iter_errors(snapshot), key=str)
            assert not errors, f"{slug}: {errors[0].message}"

    def test_snapshots_agree_with_the_reference_manifests(self, shipped):
        """The four withheld datasets ship a manifest; the snapshot must be the same run."""
        checked = 0
        for slug, snapshot in shipped.items():
            reference = reference_manifest_path(slug)
            if not reference.is_file():
                continue
            manifest = json.loads(reference.read_text(encoding="utf-8"))
            assert snapshot["fold_digest"] == manifest["fold_digest"], slug
            assert snapshot["records"] == manifest["records"], slug
            checked += 1
        assert checked == 4

    def test_no_shipped_snapshot_predates_a_manifest_silently(self, shipped):
        """A validation-only snapshot says so in ``sources``; nothing else is None."""
        for slug, snapshot in shipped.items():
            if "manifest.json" in snapshot["sources"]:
                assert snapshot["fold_digest"] is not None, slug
                assert snapshot["n_folds"] is not None, slug
            else:
                assert snapshot["fold_digest"] is None, slug


class TestSnapshotsInTheModel:
    def test_facts_carry_manifest_and_report_provenance(self):
        meta = open_store().get("mitdb")
        sources = {f.provenance.source for f in meta.facts_for("records")}
        assert {"catalogue", "manifest", "validation_report"} <= sources
        winner = meta.fact("records")
        assert winner.provenance.source == "manifest"
        assert winner.provenance.observed_at  # the run's timestamp
        assert winner.provenance.source_path == "ecgbench/data/snapshots/mitdb.json"
        assert meta.fact("records_excluded").value == 3
        assert meta.fact("quality_check:amplitude_outlier").value == 3
        assert meta.fact("fold_digest_clean").provenance.source == "manifest"

    def test_recomputed_count_wins_the_top_level_field(self):
        """ptbxl: the catalogue says 21,799, the validation snapshot excludes 49."""
        meta = open_store().get("ptbxl")
        assert meta.records == 21799
        assert meta.fact("records").provenance.source == "validation_report"
        assert meta.fact("records_excluded").value == 49
        assert meta.fact("records_clean").value == 21750

    def test_artefacts_are_populated(self):
        meta = open_store().get("mitdb")
        kinds = {(a.kind, a.version) for a in meta.artefacts}
        assert kinds == {("fold_csv", "original"), ("fold_csv", "clean"),
                         ("validation_report", "original")}
        fold = next(a for a in meta.artefacts if a.version == "clean" and a.kind == "fold_csv")
        assert fold.n_records == 45 and len(fold.sha256) == 64
        assert open_store().get("ptb-xl-plus").artefacts == ()  # catalogue-only

    def test_a_dataset_without_a_snapshot_has_no_computed_facts(self):
        meta = open_store().get("ptb-xl-plus")
        assert not any(f.provenance.source in ("manifest", "validation_report") for f in meta.facts)

    def test_info_verbose_shows_the_excluded_count_next_to_the_catalogue(self, capsys):
        assert main(["info", "ptbxl", "--verbose"]) == 0
        out = capsys.readouterr().out
        assert "validation_report" in out
        assert "records_excluded" in out and "49" in out
        assert "ecgbench/data/snapshots/ptbxl.json" in out

    def test_info_records_line_shows_the_resolved_count(self, capsys):
        assert main(["info", "mitdb"]) == 0
        out = capsys.readouterr().out
        line = next(ln for ln in out.splitlines() if ln.startswith("records"))
        assert "48" in line

    def test_snapshot_naming_no_config_is_a_build_problem(self, tmp_path, monkeypatch):
        stray = tmp_path / "no_such_dataset.json"
        stray.write_text(
            json.dumps({**build_snapshot_toy(), "dataset": "no_such_dataset"}),
            encoding="utf-8",
        )
        monkeypatch.setattr(build_module, "load_snapshots", lambda: load_snapshots(tmp_path))
        with pytest.raises(build_module.MetadataBuildError, match="names no config"):
            build_module.build_model()


def build_snapshot_toy() -> dict:
    return {
        "snapshot_version": 1,
        "dataset": "toy",
        "dataset_version": None,
        "ecgbench_version": "0.99.0",
        "created": None,
        "sources": ["validation_report.json"],
        "records": {"original": 1, "clean": 1},
        "records_excluded": 0,
        "quality_checks": [],
        "n_folds": None,
        "random_state": None,
        "grouped_by_patient": None,
        "fold_digest": None,
        "digest_version": None,
        "inputs": {},
    }
