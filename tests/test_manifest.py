"""Tests for split manifests and the publish-policy guards."""

import json
from dataclasses import replace

import pandas as pd
import pytest

from ecgbench.config import LabelConfig, load_config
from ecgbench.manifest import (
    ManifestMismatchError,
    build_manifest,
    fold_digest,
    load_reference_manifest,
    verify_splits,
)


class TestFoldDigest:
    """The digest must depend on the partition and nothing else."""

    def _frame(self, ids, folds):
        return pd.DataFrame({"rec": ids, "fold": folds})

    def test_row_order_does_not_matter(self):
        a = self._frame(["r1", "r2", "r3"], [1, 2, 3])
        b = self._frame(["r3", "r1", "r2"], [3, 1, 2])

        assert fold_digest(a, "rec") == fold_digest(b, "rec")

    def test_a_changed_assignment_changes_the_digest(self):
        a = self._frame(["r1", "r2"], [1, 2])
        b = self._frame(["r1", "r2"], [2, 1])

        assert fold_digest(a, "rec") != fold_digest(b, "rec")

    def test_extra_columns_do_not_matter(self):
        a = self._frame(["r1", "r2"], [1, 2])
        b = a.assign(is_valid=[True, False], path=["x", "y"])

        assert fold_digest(a, "rec") == fold_digest(b, "rec")

    def test_int_and_str_ids_agree(self):
        """Fold CSVs round-trip ids through CSV, so 1 and "1" must digest alike."""
        a = self._frame([1, 2], [1, 2])
        b = self._frame(["1", "2"], [1, 2])

        assert fold_digest(a, "rec") == fold_digest(b, "rec")

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="missing column"):
            fold_digest(pd.DataFrame({"rec": ["r1"]}), "rec")


class TestBuildManifest:
    def _config(self, sample_config, tmp_path):
        (tmp_path / "metadata.csv").write_text("rec,label\nr1,0\nr2,1\n", encoding="utf-8")
        return replace(
            sample_config,
            slug="demo",
            metadata_csv="metadata.csv",
            record_id_column="rec",
            patient_id_column=None,
            labels=None,
        )

    def test_records_seed_inputs_and_digests(self, sample_config, tmp_path):
        config = self._config(sample_config, tmp_path)
        original = pd.DataFrame({"rec": ["r1", "r2"], "fold": [1, 2]})
        clean = original.iloc[:1]

        m = build_manifest(config, tmp_path, original, clean, n_folds=10, random_state=42)

        assert m["dataset"] == "demo"
        assert m["split"]["random_state"] == 42
        assert m["split"]["n_folds"] == 10
        assert m["records"] == {"original": 2, "clean": 1}
        assert m["inputs"]["metadata.csv"]["present"] is True
        assert len(m["inputs"]["metadata.csv"]["sha256"]) == 64
        assert m["fold_digest"]["original"] != m["fold_digest"]["clean"]

    def test_absent_input_is_recorded_not_skipped(self, sample_config, tmp_path):
        """A manifest must never silently omit an input it could not read."""
        config = replace(
            self._config(sample_config, tmp_path),
            labels=LabelConfig(source_csv="nope.csv", join_column="rec"),
        )
        df = pd.DataFrame({"rec": ["r1"], "fold": [1]})

        m = build_manifest(config, tmp_path, df, df, n_folds=2, random_state=1)

        assert m["inputs"]["nope.csv"] == {"present": False}


class TestVerifySplits:
    def _write(self, tmp_path, manifest):
        (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return tmp_path

    def test_matching_run_passes(self, tmp_path):
        reference = load_reference_manifest("mimic_iv_ecg")
        assert reference is not None, "mimic_iv_ecg reference manifest must ship"

        report = verify_splits("mimic_iv_ecg", self._write(tmp_path, reference))

        assert report["ok"] is True
        assert report["versions"]["clean"]["match"] is True

    def test_different_partition_is_caught(self, tmp_path):
        m = dict(load_reference_manifest("mimic_iv_ecg"))
        m["fold_digest"] = dict(m["fold_digest"], clean="0" * 64)

        with pytest.raises(ManifestMismatchError):
            verify_splits("mimic_iv_ecg", self._write(tmp_path, m))

    def test_a_filtered_input_is_named_as_the_cause(self, tmp_path):
        """The MIMIC failure mode: a filtered label CSV changes the folds."""
        m = json.loads(json.dumps(load_reference_manifest("mimic_iv_ecg")))
        m["fold_digest"]["clean"] = "0" * 64
        m["inputs"]["machine_measurements.csv"]["sha256"] = "f" * 64
        m["inputs"]["machine_measurements.csv"]["rows"] = 789481

        with pytest.raises(ManifestMismatchError, match="machine_measurements.csv"):
            verify_splits("mimic_iv_ecg", self._write(tmp_path, m))

    def test_digest_version_mismatch_refuses_to_compare(self, tmp_path):
        m = dict(load_reference_manifest("mimic_iv_ecg"), digest_version=999)

        with pytest.raises(ManifestMismatchError, match="not comparable"):
            verify_splits("mimic_iv_ecg", self._write(tmp_path, m))

    def test_dataset_without_a_reference_manifest_says_so(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No reference manifest"):
            verify_splits("ptbxl", tmp_path)


class TestPublishPolicy:
    """Credentialed sources must be refused in both directions."""

    def test_mimic_is_marked_unpublishable_with_a_reason(self):
        config = load_config("mimic_iv_ecg")

        assert config.publish_fold_csvs is False
        reason = config.no_publish_reason
        assert "credentialed" in reason.lower()
        # The reason is quoted back to users, so it must carry the fix.
        assert "ecgbench splits --dataset mimic_iv_ecg" in reason

    def test_open_datasets_still_publish_by_default(self):
        for slug in ("ptbxl", "brugada_huca", "incartdb", "challenge2021"):
            assert load_config(slug).publish_fold_csvs is True, slug

    def test_a_shipped_manifest_exists_for_every_unpublished_dataset(self):
        """Otherwise users have no way to check their regeneration."""
        from ecgbench.config import list_available_configs

        for slug in list_available_configs():
            if not load_config(slug).publish_fold_csvs:
                assert (
                    load_reference_manifest(slug) is not None
                ), f"{slug} is not published but ships no reference manifest"

    def test_upload_refuses_an_unpublishable_dataset(self, tmp_path, monkeypatch):
        from ecgbench.cli.upload import run_upload

        monkeypatch.setenv("HF_TOKEN", "not-used-the-guard-fires-first")
        (tmp_path / "mimic_iv_ecg").mkdir()

        with pytest.raises(PermissionError, match="publish_fold_csvs: false"):
            run_upload(data_dir=tmp_path, datasets=["mimic_iv_ecg"], dry_run=True)

    def test_loader_refuses_the_hub_with_an_actionable_error(self, tmp_path):
        pytest.importorskip("torch")
        from ecgbench import ECGDataset
        from ecgbench.dataset import SplitsNotPublishedError

        with pytest.raises(SplitsNotPublishedError) as excinfo:
            ECGDataset("mimic_iv_ecg", split="train", data_path=tmp_path)
        message = str(excinfo.value)
        assert "does not publish fold CSVs" in message
        assert "ecgbench splits --dataset mimic_iv_ecg" in message
        assert 'metadata_source="local"' in message
