"""Smoke tests for the ``ecgbench`` CLI and its Python API wrappers.

The underlying pipeline logic (validation, splitting, Croissant generation,
HuggingFace upload) is exercised by the other test modules. These tests only
verify that the CLI dispatcher wires everything up correctly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ecgbench.cli import main, run_croissant, run_upload
from ecgbench.cli._main import _build_parser


def _skip_if_no_mlcroissant():
    pytest.importorskip("mlcroissant")


# --- Parser / dispatch ---------------------------------------------------


def test_top_level_help_lists_subcommands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for sub in ("splits", "croissant", "upload"):
        assert sub in out


@pytest.mark.parametrize("sub", ["splits", "croissant", "upload"])
def test_subcommand_help_parses(sub, capsys):
    with pytest.raises(SystemExit) as exc:
        main([sub, "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--dataset" in out or "--data-dir" in out


def test_missing_subcommand_fails():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


# --- run_croissant (Python API) ------------------------------------------


def test_run_croissant_on_prepared_splits(
    tmp_splits_dir: Path, sample_config, monkeypatch
):
    _skip_if_no_mlcroissant()

    # run_croissant loads the config by slug; patch the loader so we can use
    # the in-memory sample_config instead of requiring a YAML on disk.
    import ecgbench.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "load_config", lambda slug: sample_config)

    saved = run_croissant(
        dataset=sample_config.slug,
        splits_dir=tmp_splits_dir / "clean",
        version="clean",
    )

    assert saved.exists()
    data = json.loads(saved.read_text())
    # Croissant 1.1 JSON-LD has @context and @type at the top level
    assert "@context" in data
    assert "@type" in data


# --- run_upload (Python API, dry-run) ------------------------------------


def test_run_upload_dry_run_counts_files(tmp_path: Path, monkeypatch):
    # Fake HF token so _resolve_hf_token succeeds without a real one
    monkeypatch.setenv("HF_TOKEN", "dummy-token-for-dry-run")

    dataset_slug = "demo"
    ds_dir = tmp_path / dataset_slug
    (ds_dir / "clean" / "train").mkdir(parents=True)
    (ds_dir / "original" / "train").mkdir(parents=True)

    pd.DataFrame({"record_id": ["a"], "filename": ["x"]}).to_csv(
        ds_dir / "clean" / "train" / "fold_1.csv", index=False
    )
    pd.DataFrame({"record_id": ["a"], "filename": ["x"]}).to_csv(
        ds_dir / "original" / "train" / "fold_1.csv", index=False
    )
    (ds_dir / "validation_report.json").write_text("{}")

    counts = run_upload(
        data_dir=tmp_path,
        datasets=[dataset_slug],
        dry_run=True,
    )
    # 2 CSVs + 1 validation_report.json = 3 files
    assert counts == {dataset_slug: 3}


def test_run_upload_missing_dataset_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy-token-for-dry-run")
    counts = run_upload(
        data_dir=tmp_path,
        datasets=["does-not-exist"],
        dry_run=True,
    )
    assert counts == {"does-not-exist": 0}
