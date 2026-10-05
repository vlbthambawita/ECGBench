"""Tests for the metadata exports (metadata layer, Phase 5).

Three views over the model: schema.org JSON-LD per dataset, a Croissant 1.1
collection, and the committed website copy. The shipped model is used
throughout, so these double as checks that every configured dataset exports.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecgbench.cli import main
from ecgbench.cli.metadata import run_metadata_export, run_metadata_export_check
from ecgbench.metadata import export as export_module
from ecgbench.metadata import open_store
from ecgbench.metadata.export import (
    HF_RESOLVE_URL,
    SITE_URL,
    WEBSITE_JSON_PATH,
    catalogue_slug,
    page_url,
    to_croissant,
    to_croissant_collection,
    to_schema_org,
    to_website,
    validate_croissant,
    website_json,
    website_json_current,
    write_website_json,
)


@pytest.fixture(scope="module")
def model():
    return tuple(open_store().all())


@pytest.fixture(scope="module")
def by_id(model):
    return {m.dataset_id: m for m in model}


# --------------------------------------------------------------------------- schema.org


class TestSchemaOrg:
    def test_every_dataset_exports_a_standard_block(self, model):
        allowed = {
            "@context", "@type", "@id", "identifier", "name", "url", "includedInDataCatalog",
            "alternateName", "description", "version", "sameAs", "license",
            "isAccessibleForFree", "conditionsOfAccess", "keywords", "creator", "citation",
            "isBasedOn", "variableMeasured", "distribution", "dateModified",
        }
        for meta in model:
            doc = to_schema_org(meta)
            assert doc["@context"] == "https://schema.org"
            assert doc["@type"] == "Dataset"
            assert doc["@id"] == page_url(meta)
            assert doc["@id"] == f"{SITE_URL}/datasets/{catalogue_slug(meta)}.html"
            assert set(doc) <= allowed, (meta.dataset_id, set(doc) - allowed)
            assert doc["name"] == meta.name

    def test_published_dataset_lists_the_hub_fold_csvs(self, by_id):
        doc = to_schema_org(by_id["mitdb"])
        downloads = {d["name"]: d for d in doc["distribution"]}
        clean = downloads["ECGBench clean folds"]
        assert clean["contentUrl"] == f"{HF_RESOLVE_URL}/mitdb/clean/folds.csv"
        assert clean["encodingFormat"] == "text/csv"
        assert len(clean["sha256"]) == 64
        assert doc["isAccessibleForFree"] is True
        assert doc["dateModified"] == "2026-08-07"
        assert any(v["name"] == "recorder" for v in doc["variableMeasured"])

    def test_withheld_dataset_has_no_hub_distribution(self, by_id):
        # ikem is open data withheld for its NoDerivatives licence: free, but no folds.
        doc = to_schema_org(by_id["ikem"])
        assert not any(
            "huggingface" in d.get("contentUrl", "") for d in doc.get("distribution", ())
        )
        assert doc["isAccessibleForFree"] is True
        # mimic_iv_ecg is credentialed: not free, and no folds either.
        doc = to_schema_org(by_id["mimic_iv_ecg"])
        assert doc["isAccessibleForFree"] is False
        assert doc["conditionsOfAccess"] == "credentialed"
        assert not any(
            "huggingface" in d.get("contentUrl", "") for d in doc.get("distribution", ())
        )

    def test_catalogue_only_dataset_still_exports(self, by_id):
        doc = to_schema_org(by_id["ptb-xl-plus"])
        assert doc["identifier"] == "ptb-xl-plus"
        assert "distribution" not in doc or all(
            "huggingface" not in d["contentUrl"] for d in doc["distribution"]
        )

    def test_derivation_becomes_is_based_on(self, by_id):
        doc = to_schema_org(by_id["qtdb"])
        assert f"{SITE_URL}/datasets/mitdb.html" in doc["isBasedOn"]


# --------------------------------------------------------------------------- croissant


class TestCroissant:
    def test_published_dataset_carries_distribution_and_record_sets(self, by_id):
        doc = to_croissant(by_id["mitdb"])
        assert doc["@type"] == "sc:Dataset"
        assert doc["conformsTo"] == "http://mlcommons.org/croissant/1.1"
        assert doc["name"] == "mitdb"
        ids = {d["@id"] for d in doc["distribution"]}
        assert ids == {"original-folds", "clean-folds"}
        clean = next(d for d in doc["distribution"] if d["@id"] == "clean-folds")
        assert clean["contentUrl"].endswith("/mitdb/clean/folds.csv")
        assert len(clean["sha256"]) == 64
        sets = {r["@id"]: r for r in doc["recordSet"]}
        original = {f["name"] for f in sets["original"]["field"]}
        assert original == {"record_name", "patient_id", "fold", "default_split", "is_valid",
                            "quality_issues"}
        assert {f["name"] for f in sets["clean"]["field"]} == {
            "record_name", "patient_id", "fold", "default_split"
        }
        assert sets["clean"]["key"] == {"@id": "clean/record_name"}
        assert doc["prov:wasDerivedFrom"][0]["name"] == "ecgbench_metadata.csv"
        assert "odrl:hasPolicy" not in doc

    def test_withheld_dataset_gets_a_prohibition_instead(self, by_id):
        doc = to_croissant(by_id["mimic_iv_ecg"])
        assert "distribution" not in doc and "recordSet" not in doc
        assert "ecgbench splits" in doc["conditionsOfAccess"]
        assert doc["odrl:hasPolicy"]["odrl:prohibition"] == [{"odrl:action": "odrl:distribute"}]
        # the 1.1 constructs can be switched off
        plain = to_croissant(by_id["mimic_iv_ecg"], include_1_1=False)
        assert "odrl:hasPolicy" not in plain and "prov:wasDerivedFrom" not in plain

    def test_collection_holds_every_configured_dataset_once(self, model):
        collection = to_croissant_collection(model)
        assert collection["@type"] == "sc:DataCatalog"
        names = [d["name"] for d in collection["dataset"]]
        assert len(names) == len(set(names)) == sum(m.has_config for m in model)
        assert all("@context" not in d for d in collection["dataset"])
        assert len(collection["hasPart"]) == sum(not m.has_config for m in model)

    def test_export_is_deterministic(self, model):
        assert to_croissant_collection(model) == to_croissant_collection(model)
        assert website_json(model) == website_json(model)

    def test_mlcroissant_validates_the_collection(self, model):
        pytest.importorskip("mlcroissant")
        errors = validate_croissant(to_croissant_collection(model))
        assert errors == []

    def test_validator_reports_a_broken_member(self, by_id):
        # mlcroissant 1.0.22 only *warns* about a missing name/license/version, so the
        # defect has to be structural: a field whose source names no FileObject.
        pytest.importorskip("mlcroissant")
        doc = to_croissant(by_id["mitdb"])
        doc["recordSet"][0]["field"][0]["source"]["fileObject"]["@id"] = "missing/folds.csv"
        errors = validate_croissant(doc)
        assert errors and errors[0].startswith("mitdb: ")
        assert "missing/folds.csv" in errors[0]
        # and a whole collection is checked member by member
        collection = to_croissant_collection([by_id["mitdb"], by_id["afdb"]])
        collection["dataset"][0]["recordSet"][0]["field"][0]["source"]["fileObject"]["@id"] = "x"
        assert len(validate_croissant(collection)) == 1


# --------------------------------------------------------------------------- website


class TestWebsite:
    def test_keyed_by_catalogue_slug_with_the_resolved_counts(self, model, by_id):
        site = to_website(model)
        assert set(site) == {catalogue_slug(m) for m in model}
        entry = site["mit-bih-arrhythmia-database"]
        assert entry["dataset_id"] == "mitdb"
        assert entry["records"] == 48 and entry["records_clean"] == 45
        assert entry["records_excluded"] == 3
        assert entry["quality_checks"] == {"amplitude_outlier": 3}
        assert "recorder" in entry["field_names"] and "recorder" in entry["search_text"]
        assert entry["schema_org"] == to_schema_org(by_id["mitdb"])
        assert site["ptb-xl-plus"]["records_clean"] is None

    def test_committed_website_copy_matches_a_fresh_export(self, model):
        """The drift guard: docs/_data/metadata.json is derived and must not be stale."""
        assert WEBSITE_JSON_PATH.is_file()
        assert website_json_current(model), (
            "docs/_data/metadata.json is stale; regenerate with "
            "`ecgbench metadata export --website -` (or `ecgbench metadata build`)"
        )
        assert json.loads(WEBSITE_JSON_PATH.read_text(encoding="utf-8")) == to_website(model)

    def test_write_is_idempotent(self, model, tmp_path):
        target = tmp_path / "metadata.json"
        assert write_website_json(model, target) is True
        assert write_website_json(model, target) is False
        assert website_json_current(model, target)


# --------------------------------------------------------------------------- cli


class TestCli:
    def test_all_three_outputs(self, tmp_path, capsys):
        code = main([
            "metadata", "export",
            "--croissant", str(tmp_path / "c.jsonld"),
            "--schema-org", str(tmp_path / "schema"),
            "--website", str(tmp_path / "site.json"),
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "c.jsonld" in out and "schema.org blocks" in out and "site.json" in out
        assert (tmp_path / "schema" / "mit-bih-arrhythmia-database.jsonld").is_file()
        block = json.loads(
            (tmp_path / "schema" / "mit-bih-arrhythmia-database.jsonld").read_text("utf-8")
        )
        assert block["@type"] == "Dataset"
        collection = json.loads((tmp_path / "c.jsonld").read_text("utf-8"))
        assert collection["@type"] == "sc:DataCatalog"

    def test_validate_flag(self, tmp_path, capsys):
        pytest.importorskip("mlcroissant")
        code = main(["metadata", "export", "--croissant", str(tmp_path / "c.jsonld"), "--validate"])
        assert code == 0
        assert "mlcroissant: no errors" in capsys.readouterr().out

    def test_check_passes_on_the_committed_copy(self, capsys):
        assert main(["metadata", "export", "--check"]) == 0
        assert "up to date" in capsys.readouterr().out
        assert run_metadata_export_check() is True

    def test_check_fails_on_a_stale_copy(self, tmp_path, capsys):
        stale = tmp_path / "metadata.json"
        stale.write_text("{}", encoding="utf-8")
        assert main(["metadata", "export", "--check", "--website", str(stale)]) == 1
        assert "stale" in capsys.readouterr().err

    def test_no_output_is_a_usage_error(self, capsys):
        assert main(["metadata", "export"]) == 2
        assert "--croissant" in capsys.readouterr().err

    def test_python_api_returns_the_paths(self, tmp_path):
        written = run_metadata_export(website=tmp_path / "w.json")
        assert written["website"] == [tmp_path / "w.json"]
        assert written["croissant"] == [] and written["schema_org"] == []


def test_build_refreshes_the_website_copy_in_a_source_checkout(tmp_path, monkeypatch, model):
    """`ecgbench metadata build` writes docs/_data/metadata.json alongside metadata.json."""
    from ecgbench.metadata import build as build_module

    target = tmp_path / "docs" / "_data" / "metadata.json"
    monkeypatch.setattr(export_module, "WEBSITE_JSON_PATH", target)
    monkeypatch.setattr(build_module, "is_source_checkout", lambda: True)
    monkeypatch.setattr(build_module, "DEFAULT_JSON_PATH", tmp_path / "metadata.json")
    monkeypatch.setattr(build_module, "SQLITE_PATH", tmp_path / "metadata.sqlite")
    monkeypatch.setattr(build_module, "SOURCES_PATH", tmp_path / "metadata.sources.json")
    build_module.build_all(model=model)
    assert target.is_file()
    assert Path(target).read_text(encoding="utf-8") == website_json(model)
