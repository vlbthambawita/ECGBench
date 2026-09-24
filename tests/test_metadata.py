"""Tests for the metadata layer: build, model, identity, store and the CLI views.

Two kinds of test live here. Whole-model invariants run the real build over the
shipped catalogue and configs (no network, no signal files) and pin the facts the
plan established: 64 datasets, 13 catalogue-only, every alias resolving. Unit
tests build tiny synthetic ``DatasetMeta`` records to exercise precedence,
serialisation and search without depending on the shipped data.

The committed ``ecgbench/data/metadata.json`` is a derived file; the drift test
fails when its digest no longer matches a fresh build, which is the signal to run
``python -c "from ecgbench.metadata import write_json; write_json()"``.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import stat
import warnings
from pathlib import Path

import pytest

import ecgbench
from ecgbench.cli import main
from ecgbench.config import list_available_configs
from ecgbench.labels import _custom_loaders
from ecgbench.metadata import (
    DEFAULT_JSON_PATH,
    IMPLEMENTATION_STATES,
    SCHEMA_VERSION,
    SQLITE_PATH,
    AccessMeta,
    AliasIndex,
    DatasetMeta,
    Fact,
    MetadataQueryError,
    MetadataStore,
    Provenance,
    RelationMeta,
    SearchHit,
    SignalMeta,
    SplitMeta,
    UnknownDatasetError,
    build_all,
    build_model,
    content_digest,
    diff_exports,
    load_json,
    open_store,
    parse_count,
    to_json,
    write_json,
    write_sqlite,
)
from ecgbench.metadata import build as build_module
from ecgbench.metadata import store as store_module


@pytest.fixture(scope="module")
def model():
    return build_model()


@pytest.fixture(scope="module")
def store(model):
    return MetadataStore(model)


@pytest.fixture(scope="module")
def by_id(model):
    return {m.dataset_id: m for m in model}


# --------------------------------------------------------------------------- whole model


class TestBuildOverShippedSources:
    def test_one_record_per_catalogue_entry(self, model):
        entries = ecgbench.list_datasets()
        assert len(model) == len(entries) == 64
        # every catalogue slug is the first alias of exactly one record
        assert sorted(m.aliases[0] for m in model) == sorted(e.slug for e in entries)

    def test_dataset_ids_are_unique_and_sorted(self, model):
        ids = [m.dataset_id for m in model]
        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)

    def test_dataset_id_is_the_config_slug_where_one_exists(self, model):
        configured = {m.dataset_id for m in model if m.has_config}
        assert configured == set(list_available_configs())

    def test_implementation_state_distribution(self, model):
        counts = {s: 0 for s in IMPLEMENTATION_STATES}
        for m in model:
            counts[m.implementation_state] += 1
        assert counts["catalogue_only"] == 13
        # The demo subset declares its labels unavailable; nothing else does.
        assert counts["config"] == 1
        # The four unpublished datasets of the distribution policy.
        assert counts["config_labels"] == 4
        assert counts["published"] == 64 - 13 - 1 - 4

    def test_catalogue_only_entries_are_the_expected_ones(self, model):
        catalogue_only = {
            m.dataset_id for m in model if m.implementation_state == "catalogue_only"
        }
        expected = {"ptb-xl-plus", "mimic-iv-ecg-ext-icd", "symile-mimic", "kurias-ecg"}
        assert expected <= catalogue_only
        for m in model:
            if m.implementation_state == "catalogue_only":
                assert m.signal is None and m.split is None and m.version is None

    def test_unpublished_datasets_match_the_policy(self, by_id):
        withheld = [m for m in by_id.values() if m.has_config and not m.access.publish_fold_csvs]
        assert {m.dataset_id for m in withheld} == {
            "mimic_iv_ecg",
            "echonext",
            "ikem",
            "ecg_capable_smartwatches",
        }
        for m in withheld:
            assert m.implementation_state == "config_labels"
            # the reason must carry the regeneration command
            assert "ecgbench splits" in m.access.no_publish_reason

    def test_label_module_files_match_the_loader_registry(self):
        """The build reads label availability from files, never importing pandas.

        This pins that shortcut to the truth: every registered custom loader has a
        module named after its slug, and every slug-named module that is also a
        config is registered.
        """
        registered = set(_custom_loaders())
        configured = set(list_available_configs())
        on_disk = {
            p.stem for p in build_module._LABELS_DIR.glob("*.py") if not p.stem.startswith("_")
        }
        assert registered <= on_disk
        assert (on_disk & configured) == registered

    def test_configured_datasets_have_signal_and_split_facets(self, model):
        for m in model:
            if m.has_config:
                assert m.signal is not None and m.split is not None, m.dataset_id
                assert m.signal.leads >= 1
                assert m.signal.default_sampling_rate in m.signal.sampling_rates
                assert m.split.n_folds >= 2

    def test_every_fact_has_known_provenance(self, model):
        from ecgbench.metadata import SOURCE_PRECEDENCE

        for m in model:
            for f in m.facts:
                assert f.provenance.source in SOURCE_PRECEDENCE, (m.dataset_id, f)
                if f.provenance.source in ("manifest", "validation_report"):
                    assert f.provenance.source_path.startswith("ecgbench/data/snapshots/"), f
                    assert f.provenance.observed_at, f  # the run's timestamp
                else:
                    assert f.provenance.source_path.endswith((".md", ".yaml")), f
                    assert f.provenance.observed_at is None, f

    def test_config_facts_are_present_only_with_a_config(self, model):
        for m in model:
            sources = {f.provenance.source for f in m.facts}
            assert "catalogue" in sources
            assert ("config" in sources) is m.has_config, m.dataset_id

    def test_relations_target_dataset_ids(self, model, by_id):
        for m in model:
            for r in m.relations:
                assert r.target in by_id, (m.dataset_id, r.target)
                assert r.target != m.dataset_id

    def test_known_relation_survives_the_merge(self, by_id):
        edge = next(r for r in by_id["qtdb"].relations if r.target == "mitdb")
        assert edge.relation == "derived_from"
        assert edge.shares_records is True
        assert edge.verified is True

    def test_counts_parse_where_the_catalogue_is_a_plain_integer(self, by_id):
        assert by_id["ptbxl"].records == 21799
        assert by_id["ptbxl"].patients == 18869
        assert by_id["mitdb"].records == 48 and by_id["mitdb"].patients == 47
        # A qualified catalogue string is not parsed, but a snapshot's recomputed
        # count outranks it (Phase 4): afdb keeps its display text and gains 25.
        assert by_id["afdb"].records_display == "25 (23 with signals)"
        assert by_id["afdb"].records == 25
        assert by_id["afdb"].fact("records").provenance.source == "manifest"
        # Without a snapshot the qualified string still yields None.
        unsnapshotted = [
            m for m in by_id.values()
            if not m.artefacts and m.records_display and parse_count(m.records_display) is None
        ]
        assert unsnapshotted, "expected a catalogue-only dataset with a qualified count"
        assert all(m.records is None for m in unsnapshotted)

    def test_name_is_the_catalogue_display_name_and_config_name_is_an_alias(self, by_id):
        m = by_id["chapman_shaoxing"]
        assert m.name == "Chapman-Shaoxing ECG Database (10,646 patients)"
        assert "Chapman-Shaoxing" in m.aliases
        assert "chapman-shaoxing-ecg-database-10-646-patients" in m.aliases
        # precedence keeps the config fact first, even though the field shows the catalogue name
        assert m.fact("name").provenance.source == "config"
        assert "name" in m.disagreements()

    def test_prose_is_searchable_but_not_a_fact(self, by_id):
        m = by_id["mitdb"]
        assert "record 114" in m.prose.lower() or "114" in m.prose
        assert not any(f.key == "prose" for f in m.facts)


class TestIdentityOverShippedSources:
    def test_every_alias_resolves_for_every_configured_dataset(self, store, model):
        for m in model:
            if not m.has_config:
                continue
            for alias in m.aliases:
                assert store.resolve(alias) == m.dataset_id, alias
                assert store.resolve(alias.upper()) == m.dataset_id, alias

    def test_the_two_slug_namespaces_and_the_display_name_agree(self, store):
        assert store.resolve("ptb-xl") == store.resolve("ptbxl") == store.resolve("PTB-XL")
        assert store.resolve("mit-bih-arrhythmia-database") == "mitdb"
        assert store.resolve("chapman-shaoxing-arrhythmia") == "ecg_arrhythmia"

    def test_unknown_key_names_close_matches(self, store):
        with pytest.raises(UnknownDatasetError) as exc:
            store.get("ptb_xl")
        assert "ptbxl" in exc.value.close_matches or "ptb-xl" in exc.value.close_matches
        assert "did you mean" in str(exc.value)

    def test_related_returns_edges(self, store):
        edges = store.related("mitdb")
        assert any(e.target == "qtdb" for e in edges)


class TestStoreSearchOverShippedSources:
    def test_no_filters_returns_everything(self, store):
        assert len(store.search()) == 64

    def test_query_is_a_superset_of_catalogue_search(self, store):
        for word in ("holter", "brazil", "wfdb", "physionet", "sleep"):
            old = {e.slug for e in ecgbench.search(word)}
            new = {m.aliases[0] for m in store.search(word)}
            assert old <= new, word

    def test_structured_filters_agree_with_the_configs(self, store):
        from ecgbench.config import load_config

        hits = store.search(leads=12, signal_format="wfdb", access="open")
        assert hits
        for m in hits:
            cfg = load_config(m.dataset_id)
            assert cfg.leads == 12 and cfg.signal_format == "wfdb"
            assert m.access.access == "open"
        # and nothing matching was left out
        expected = {
            slug
            for slug in list_available_configs()
            if load_config(slug).leads == 12
            and load_config(slug).signal_format == "wfdb"
            and store.get(slug).access.access == "open"
        }
        assert {m.dataset_id for m in hits} == expected

    def test_leads_filter_uses_catalogue_leads_for_catalogue_only_entries(self, store):
        hits = store.search(leads=12, state="catalogue_only")
        assert hits
        assert all(m.signal is None for m in hits)

    def test_published_and_has_labels_filters(self, store):
        published = store.search(published=True)
        assert all(m.implementation_state == "published" for m in published)
        assert {m.dataset_id for m in store.search(has_labels=True, published=False)} == {
            "mimic_iv_ecg",
            "echonext",
            "ikem",
            "ecg_capable_smartwatches",
        }

    def test_record_bounds_exclude_unparsed_counts(self, store):
        small = store.search(max_records=50)
        assert all(m.records is not None and m.records <= 50 for m in small)
        assert "mitdb" in {m.dataset_id for m in small}
        # afdb's catalogue string is qualified, but its snapshot resolves it to 25.
        assert "afdb" in {m.dataset_id for m in small}
        # A dataset whose only count is an unparsed catalogue string is excluded.
        unparsed = [
            m.dataset_id for m in store.all()
            if m.records is None and m.records_display
        ]
        assert unparsed
        assert not set(unparsed) & {m.dataset_id for m in small}

    def test_bad_state_is_rejected(self, store):
        with pytest.raises(ValueError, match="state must be one of"):
            store.search(state="done")


# --------------------------------------------------------------------------- serialisation


class TestSerialisation:
    def test_digest_is_stable_across_builds(self, model):
        assert content_digest(model) == content_digest(build_model())

    def test_to_json_round_trips(self, model, tmp_path: Path):
        path = tmp_path / "metadata.json"
        path.write_text(to_json(model, built_at="2026-01-01T00:00:00+00:00"), encoding="utf-8")
        reloaded = load_json(path)
        assert reloaded == model
        assert content_digest(reloaded) == content_digest(model)

    def test_committed_export_matches_a_fresh_build(self, model):
        """The drift guard: metadata.json is derived, and this fails when it is stale."""
        assert DEFAULT_JSON_PATH.is_file(), "run write_json() to create ecgbench/data/metadata.json"
        document = json.loads(DEFAULT_JSON_PATH.read_text(encoding="utf-8"))
        assert document["schema_version"] == SCHEMA_VERSION
        assert document["content_digest"] == content_digest(model), (
            "ecgbench/data/metadata.json is stale; regenerate it with "
            "python -c 'from ecgbench.metadata import write_json; write_json()'"
        )

    def test_committed_export_validates_against_the_schema(self):
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads(build_module.SCHEMA_PATH.read_text(encoding="utf-8"))
        document = json.loads(DEFAULT_JSON_PATH.read_text(encoding="utf-8"))
        jsonschema.validate(document, schema)

    def test_write_json_skips_an_unchanged_file(self, model, tmp_path: Path):
        path = tmp_path / "m.json"
        assert write_json(path, model) is True
        first = path.read_text(encoding="utf-8")
        assert write_json(path, model) is False
        assert path.read_text(encoding="utf-8") == first

    def test_load_json_refuses_another_schema_version(self, model, tmp_path: Path):
        path = tmp_path / "m.json"
        document = json.loads(to_json(model))
        document["schema_version"] = 99
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match="schema_version 99"):
            load_json(path)

    def test_open_store_reads_the_bundled_file(self):
        store = open_store()
        assert len(store) == 64
        assert store.get("ptbxl").dataset_id == "ptbxl"
        assert open_store() is store  # cached


# --------------------------------------------------------------------------- unit tests


def _meta(
    dataset_id: str = "toy",
    facts: tuple[Fact, ...] = (),
    aliases: tuple[str, ...] = ("toy-dataset", "toy", "Toy Dataset"),
    **overrides,
) -> DatasetMeta:
    base = dict(
        dataset_id=dataset_id,
        aliases=aliases,
        name="Toy Dataset",
        category="two-lead",
        status="not_started",
        implementation_state="published",
        version="1.0.0",
        description="A toy.",
        paper_title=None,
        paper_doi=None,
        citation="",
        origin_institution="Nowhere",
        origin_country=None,
        search_keywords="toy",
        records=100,
        patients=None,
        records_display="100",
        patients_display="—",
        signal=SignalMeta(
            format="wfdb",
            leads=2,
            lead_names=("MLII", "V1"),
            alternate_lead_names={1: ("MLII",)},
            record_lead_layouts=None,
            sampling_rates=(360,),
            default_sampling_rate=360,
            duration_seconds=10.0,
            units="mV",
            unit_scale=1.0,
            zero_padded_identifiers=False,
        ),
        access=AccessMeta(
            access="open",
            license_text="ODC-By 1.0",
            license_url=None,
            url="https://example.org/toy",
            download_url=None,
            publish_fold_csvs=True,
            no_publish_reason="",
        ),
        split=SplitMeta(
            n_folds=10, predefined_column=None, has_patient_id=True, record_id_column="id"
        ),
        relations=(),
        facts=facts,
        prose="",
    )
    base.update(overrides)
    return DatasetMeta(**base)


class TestPrecedence:
    def test_config_beats_catalogue_and_both_facts_are_kept(self):
        facts = (
            Fact("records", 100, Provenance("catalogue", "docs/_datasets/toy-dataset.md")),
            Fact("records", 101, Provenance("config", "ecgbench/data/configs/toy.yaml")),
        )
        meta = _meta(facts=facts)
        assert meta.fact("records").value == 101
        assert [f.value for f in meta.facts_for("records")] == [101, 100]
        assert set(meta.disagreements()) == {"records"}

    def test_manifest_beats_config(self):
        facts = (
            Fact("records", 101, Provenance("config", "c.yaml")),
            Fact(
                "records", 99, Provenance("manifest", "m.json", observed_at="2026-01-01T00:00:00Z")
            ),
        )
        assert _meta(facts=facts).fact("records").value == 99

    def test_agreeing_sources_are_not_a_disagreement(self):
        facts = (
            Fact("leads", 2, Provenance("catalogue", "a.md")),
            Fact("leads", 2, Provenance("config", "b.yaml")),
        )
        assert _meta(facts=facts).disagreements() == {}

    def test_missing_key_gives_none(self):
        assert _meta().fact("nothing") is None


class TestModelRoundTrip:
    def test_to_dict_from_dict_is_identity_including_int_keys(self):
        meta = _meta(
            facts=(Fact("k", [1, "a", None], Provenance("config", "c.yaml")),),
            relations=(RelationMeta("other", "contains", True, True, "n", False),),
        )
        data = json.loads(json.dumps(meta.to_dict()))
        assert data["signal"]["alternate_lead_names"] == {"1": ["MLII"]}
        assert DatasetMeta.from_dict(data) == meta

    def test_flags(self):
        assert _meta().published is True and _meta().has_labels is True
        m = _meta(implementation_state="config_labels")
        assert m.published is False and m.has_labels is True and m.has_config is True
        m = _meta(implementation_state="catalogue_only", signal=None, split=None)
        assert m.has_config is False and m.has_labels is False


class TestAliasIndex:
    def test_case_insensitive_and_collision_detection(self):
        a = _meta("a", aliases=("a-ds", "a", "Alpha"))
        b = _meta("b", aliases=("b-ds", "b", "Beta"))
        index = AliasIndex((a, b))
        assert index.resolve("ALPHA") == "a"
        assert index.resolve(" beta ") == "b"
        assert "a-ds" in index and "gamma" not in index
        with pytest.raises(ValueError, match="alias collisions"):
            AliasIndex((a, _meta("c", aliases=("c-ds", "alpha"))))

    def test_unknown_raises_with_hint(self):
        index = AliasIndex((_meta("alpha", aliases=("alpha-dataset", "alpha")),))
        with pytest.raises(UnknownDatasetError) as exc:
            index.resolve("alpah")
        assert exc.value.close_matches


class TestStoreUnit:
    def test_search_filters_on_synthetic_records(self):
        two_lead = _meta("two", aliases=("two-ds", "two"))
        twelve = _meta(
            "twelve",
            aliases=("twelve-ds", "twelve"),
            signal=SignalMeta(
                "csv", 12, None, None, None, (500, 100), 500, 10.0, "mV", 0.001, False
            ),
            access=AccessMeta("credentialed", "DUA", None, "u", None, False, "run ecgbench splits"),
            implementation_state="config_labels",
            records=None,
            records_display="~5,000",
        )
        store = MetadataStore((two_lead, twelve))
        assert [m.dataset_id for m in store.search(leads=12)] == ["twelve"]
        assert [m.dataset_id for m in store.search(fs=100)] == ["twelve"]
        assert [m.dataset_id for m in store.search(signal_format="WFDB")] == ["two"]
        assert [m.dataset_id for m in store.search(access="credentialed")] == ["twelve"]
        assert [m.dataset_id for m in store.search(license="odc")] == ["two"]
        assert [m.dataset_id for m in store.search(published=True)] == ["two"]
        assert [m.dataset_id for m in store.search(min_records=1)] == ["two"]  # unparsed excluded
        assert [m.dataset_id for m in store.search("nowhere")] == ["twelve", "two"]  # sorted
        assert store.search("zzz") == []


class TestParseCount:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("48", 48),
            ("18,869", 18869),
            (" 800,035 ", 800035),
            (21799, 21799),
            ("~1,000", None),
            ("n/a", None),
            ("—", None),
            ("", None),
            (None, None),
            ("10–20", None),
            ("5,749 segments", None),
            ("25 (23 with signals)", None),
            ("1,00", None),
            (True, None),
        ],
    )
    def test_parse_count(self, text, expected):
        assert parse_count(text) == expected


# --------------------------------------------------------------------------- CLI


class TestCatalogCli:
    def test_list_json_is_pure_json_on_stdout(self, capsys):
        assert main(["list", "--format", "json"]) == 0
        out = capsys.readouterr().out
        rows = json.loads(out)
        assert len(rows) == 64
        assert {"dataset_id", "implementation_state", "access"} <= set(rows[0])

    def test_list_state_filter_and_table(self, capsys):
        assert main(["list", "--state", "catalogue_only"]) == 0
        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if line.strip()]
        assert lines[0].startswith("dataset_id")
        assert len(lines) == 2 + 13  # header, rule, rows

    def test_list_csv(self, capsys):
        assert main(["list", "--format", "csv", "--category", "two-lead"]) == 0
        out = capsys.readouterr().out
        header, *rows = out.strip().splitlines()
        assert header.startswith("dataset_id,name,category")
        assert rows and all(",two-lead," in r for r in rows)

    def test_info_resolves_both_slugs_to_the_same_record(self, capsys):
        assert main(["info", "ptb-xl", "--format", "json"]) == 0
        first = json.loads(capsys.readouterr().out)
        assert main(["info", "ptbxl", "--format", "json"]) == 0
        second = json.loads(capsys.readouterr().out)
        assert first["dataset_id"] == second["dataset_id"] == "ptbxl"
        assert "facts" not in first  # plain info omits the provenance table

    def test_info_verbose_json_includes_facts(self, capsys):
        assert main(["info", "mitdb", "--format", "json", "--verbose"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert any(
            f["key"] == "records" and f["provenance"]["source"] == "catalogue"
            for f in data["facts"]
        )

    def test_info_table_prints_the_mitdb_record(self, capsys):
        assert main(["info", "mit-bih-arrhythmia-database"]) == 0
        out = capsys.readouterr().out
        assert "dataset_id" in out and "mitdb" in out
        assert "MIT-BIH Arrhythmia Database" in out
        assert "record_lead_layouts" in out

    def test_info_verbose_table_lists_sources(self, capsys):
        assert main(["info", "chapman_shaoxing", "--verbose"]) == 0
        out = capsys.readouterr().out
        assert "facts (most trustworthy source first)" in out
        assert "†" in out  # the name disagrees between catalogue and config

    def test_unknown_dataset_exits_nonzero_with_a_hint(self, capsys):
        assert main(["info", "ptb_xl"]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "unknown dataset 'ptb_xl'" in captured.err
        assert "ptbxl" in captured.err or "ptb-xl" in captured.err

    def test_related_table_and_json(self, capsys):
        assert main(["related", "ptb-xl"]) == 0
        out = capsys.readouterr().out
        assert "challenge2021" in out and "shares_records" in out
        assert main(["related", "mitdb", "--format", "json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["dataset_id"] == "mitdb"
        assert any(r["target"] == "qtdb" for r in data["relations"])

    def test_related_unknown_exits_nonzero(self, capsys):
        assert main(["related", "nope"]) == 1
        assert "unknown dataset" in capsys.readouterr().err

    def test_python_api_matches_cli(self):
        from ecgbench.cli import run_info, run_list, run_related

        assert run_info("PTB-XL").dataset_id == "ptbxl"
        assert len(run_list(state="catalogue_only")) == 13
        assert run_related("mitdb") == list(open_store().get("mitdb").relations)


class TestPublicApi:
    def test_top_level_exports(self):
        assert ecgbench.get_metadata("mitdb").dataset_id == "mitdb"
        assert ecgbench.search_metadata(leads=2, signal_format="wfdb")
        assert ecgbench.related_metadata("ptbxl")
        assert isinstance(ecgbench.open_store(), MetadataStore)
        assert ecgbench.run_list is not None  # lazy import path

    def test_import_ecgbench_pulls_in_no_heavy_dependency(self):
        import subprocess
        import sys

        code = (
            "import sys, ecgbench, ecgbench.metadata; "
            "print(sorted(m for m in ('pandas', 'numpy', 'torch', 'sklearn') if m in sys.modules))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "[]"


# --------------------------------------------------------------------------- Phase 2: SQLite



def _copy_export(tmp_path: Path, with_index: bool = True) -> Path:
    """Copy the bundled export (and optionally its index) into ``tmp_path``."""
    json_path = tmp_path / "metadata.json"
    shutil.copy(DEFAULT_JSON_PATH, json_path)
    if with_index:
        if not SQLITE_PATH.is_file():
            open_store()  # builds it
        shutil.copy(SQLITE_PATH, tmp_path / "metadata.sqlite")
    return json_path


class TestSqliteIndex:
    def test_write_sqlite_creates_the_schema_and_rows(self, model, tmp_path: Path):
        path = tmp_path / "m.sqlite"
        fts = write_sqlite(model, path)
        assert fts == "fts5"
        conn = sqlite3.connect(path)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"meta", "dataset", "alias", "fact", "field", "relation", "artefact",
                "dataset_fts"} <= tables
        assert conn.execute("SELECT count(*) FROM dataset").fetchone()[0] == 64
        assert conn.execute("SELECT count(*) FROM dataset_fts").fetchone()[0] == 64
        n_fields = sum(len(m.fields) for m in model)
        assert n_fields > 0  # Phase 3 declarations
        assert conn.execute("SELECT count(*) FROM field").fetchone()[0] == n_fields
        n_artefacts = sum(len(m.artefacts) for m in model)
        assert n_artefacts > 0  # Phase 4 snapshots
        assert conn.execute("SELECT count(*) FROM artefact").fetchone()[0] == n_artefacts
        n_facts = sum(len(m.facts) for m in model)
        assert conn.execute("SELECT count(*) FROM fact").fetchone()[0] == n_facts
        meta = dict(conn.execute("SELECT key, value FROM meta"))
        assert meta["content_digest"] == content_digest(model)
        assert meta["schema_version"] == str(SCHEMA_VERSION)
        assert meta["fts"] == "fts5"
        assert meta["ecgbench_version"]
        # aliases are case-insensitive keys, and the document column round-trips
        row = conn.execute(
            "SELECT d.document FROM alias a JOIN dataset d USING (dataset_id) WHERE a.alias = ?",
            ("MIT-BIH ARRHYTHMIA DATABASE",),
        ).fetchone()
        assert DatasetMeta.from_dict(json.loads(row[0])).dataset_id == "mitdb"
        assert not (tmp_path / "m.sqlite.tmp").exists()

    def test_rebuilding_gives_the_same_digest_not_the_same_bytes(self, model, tmp_path: Path):
        a, b = tmp_path / "a.sqlite", tmp_path / "b.sqlite"
        write_sqlite(model, a, built_at="2026-01-01T00:00:00+00:00")
        write_sqlite(model, b, built_at="2026-01-02T00:00:00+00:00")
        meta_a = dict(sqlite3.connect(a).execute("SELECT key, value FROM meta"))
        meta_b = dict(sqlite3.connect(b).execute("SELECT key, value FROM meta"))
        assert meta_a["content_digest"] == meta_b["content_digest"] == content_digest(model)
        # the bytes are deliberately not compared: SQLite files are not deterministic

    def test_build_all_writes_every_derived_file(self, model, tmp_path: Path):
        result = build_all(tmp_path, model)
        assert result.json_path.is_file() and result.sqlite_path.is_file()
        assert (tmp_path / "metadata.sources.json").is_file()
        assert result.content_digest == content_digest(model)
        assert result.json_written is True
        again = build_all(tmp_path, model)
        assert again.json_written is False  # unchanged digest leaves the JSON alone

    def test_without_fts5_the_index_is_still_written(self, model, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(build_module, "fts5_available", lambda: False)
        path = tmp_path / "m.sqlite"
        assert write_sqlite(model, path) == "none"
        conn = sqlite3.connect(path)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "dataset_fts" not in tables and "dataset" in tables
        assert dict(conn.execute("SELECT key, value FROM meta"))["fts"] == "none"


class TestRankedSearch:
    @pytest.fixture(scope="class")
    def fts_store(self):
        store = open_store()
        assert store.fts_enabled, store.fts_fallback_reason
        return store

    def test_ptb_ranks_ptbxl_first(self, fts_store):
        hits = fts_store.search("ptb")
        assert hits[0].dataset_id == "ptbxl"
        assert {"ptbdb", "ptb-xl-plus"} <= {m.dataset_id for m in hits[:5]}

    def test_prefix_query_includes_both_af_databases(self, fts_store):
        ids = {m.dataset_id for m in fts_store.search("atrial fib*")}
        assert {"afdb", "ltafdb"} <= ids

    def test_not_operator_excludes(self, fts_store):
        ids = {m.dataset_id for m in fts_store.search("holter NOT paediatric")}
        assert "picsdb" not in ids
        assert {"mitdb", "nsrdb"} <= ids

    def test_phrase_query(self, fts_store):
        ids = {m.dataset_id for m in fts_store.search('"sleep apnea"')}
        assert {"ucddb", "apnea_ecg"} <= ids

    def test_scores_are_ranked_and_carry_the_state_prior(self, fts_store):
        hits = fts_store.search_ranked("ptb")
        scores = [h.score for h in hits]
        assert all(isinstance(s, float) for s in scores)
        assert scores == sorted(scores)  # more negative first
        plus = next(h for h in hits if h.meta.dataset_id == "ptb-xl-plus")
        xl = next(h for h in hits if h.meta.dataset_id == "ptbxl")
        assert plus.score > xl.score

    def test_structured_filters_apply_after_ranking(self, fts_store):
        hits = fts_store.search("holter", leads=2, access="open", limit=3)
        assert 0 < len(hits) <= 3
        for m in hits:
            assert m.signal is not None and m.signal.leads == 2 and m.access.access == "open"

    def test_filter_only_query_is_unranked(self, fts_store):
        hits = fts_store.search_ranked(None, leads=12, signal_format="wfdb", access="open")
        assert hits and all(h.score is None for h in hits)
        from ecgbench.config import load_config

        for h in hits:
            cfg = load_config(h.meta.dataset_id)
            assert cfg.leads == 12 and cfg.signal_format == "wfdb"

    def test_invalid_fts5_syntax_raises_quoting_sqlite(self, fts_store):
        with pytest.raises(MetadataQueryError) as exc:
            fts_store.search("ptb-xl")
        assert "ptb-xl" in str(exc.value)
        assert "no such column" in str(exc.value) or "syntax error" in str(exc.value)
        # the quoted form is the fix the message suggests
        assert fts_store.search('"ptb-xl"')[0].dataset_id == "ptbxl"

    def test_empty_query_returns_everything(self, fts_store):
        assert len(fts_store.search("   ")) == 64
        assert len(fts_store.search(None)) == 64


class TestFallbackAndReadOnly:
    def test_without_fts5_substring_path_is_a_superset(self, tmp_path: Path, monkeypatch):
        fts_hits = {m.dataset_id for m in open_store().search("holter")}
        json_path = _copy_export(tmp_path)
        monkeypatch.setattr(build_module, "fts5_available", lambda: False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            store = open_store(json_path)
            assert store.fts_enabled is False
            assert "FTS5" in (store.fts_fallback_reason or "")
            substring_hits = {m.dataset_id for m in store.search("holter")}
            store.search("holter")  # second query must not warn again
        assert fts_hits <= substring_hits
        fallback = [w for w in caught if "ranked search unavailable" in str(w.message)]
        assert len(fallback) == 1

    def test_in_memory_store_never_warns(self, model):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            MetadataStore(model).search("holter")

    def test_read_only_location_still_opens_with_fts(self, tmp_path: Path):
        json_path = _copy_export(tmp_path, with_index=True)
        os.chmod(tmp_path, stat.S_IRUSR | stat.S_IXUSR)
        try:
            store = open_store(json_path)
            assert store.fts_enabled, store.fts_fallback_reason
            assert store.search("ptb")[0].dataset_id == "ptbxl"
        finally:
            os.chmod(tmp_path, stat.S_IRWXU)

    def test_read_only_location_without_index_falls_back(self, tmp_path: Path):
        json_path = _copy_export(tmp_path, with_index=False)
        os.chmod(tmp_path, stat.S_IRUSR | stat.S_IXUSR)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                store = open_store(json_path)
                assert store.fts_enabled is False
                assert store.get("ptb-xl").dataset_id == "ptbxl"
                assert "mitdb" in {m.dataset_id for m in store.search("holter")}
            assert any("ranked search unavailable" in str(w.message) for w in caught)
        finally:
            os.chmod(tmp_path, stat.S_IRWXU)

    def test_stale_index_is_rebuilt_beside_a_writable_export(self, model, tmp_path: Path):
        json_path = _copy_export(tmp_path, with_index=False)
        # an index built for a different model: one dataset dropped
        write_sqlite(model[:-1], tmp_path / "metadata.sqlite")
        store = open_store(json_path)
        assert store.fts_enabled
        meta = build_module.read_sqlite_meta(tmp_path / "metadata.sqlite")
        assert meta["content_digest"] == content_digest(model)


class TestStaleness:
    def test_fingerprint_covers_every_source_kind(self):
        fp = build_module.source_fingerprint()
        assert sum(k.endswith(".md") for k in fp) == 64
        assert sum(k.endswith(".yaml") for k in fp) == len(list_available_configs())
        assert any(k.endswith("labels/mitdb.py") for k in fp)
        n_snapshots = len(list(build_module.SNAPSHOTS_DIR.glob("*.json")))
        assert n_snapshots > 0
        assert sum(k.startswith("ecgbench/data/snapshots/") for k in fp) == n_snapshots
        assert all(isinstance(v, list) and len(v) == 2 for v in fp.values())

    def test_sources_changed_semantics(self, tmp_path: Path):
        path = tmp_path / "metadata.sources.json"
        assert build_module.sources_changed(path) is True  # no record yet
        build_module.write_sources(path)
        assert build_module.sources_changed(path) is False
        recorded = build_module.read_sources(path)
        first = next(iter(recorded))
        recorded[first][1] += 1
        path.write_text(json.dumps(recorded), encoding="utf-8")
        assert build_module.sources_changed(path) is True

    def test_source_checkout_is_detected_here(self):
        assert build_module.is_source_checkout() is True

    def test_refresh_rebuilds_only_when_stale(self, monkeypatch):
        calls = []
        monkeypatch.setattr(build_module, "is_source_checkout", lambda: True)
        monkeypatch.setattr(build_module, "sources_changed", lambda: False)
        monkeypatch.setattr(build_module, "build_all", lambda: calls.append("built"))
        store_module._refresh_if_stale()
        assert calls == []
        monkeypatch.setattr(build_module, "sources_changed", lambda: True)
        monkeypatch.setattr(
            build_module,
            "build_all",
            lambda: calls.append("built")
            or build_module.BuildResult(Path("j"), Path("s"), "sha256:x", False, "fts5"),
        )
        store_module._refresh_if_stale()
        assert calls == ["built"]


class TestDiff:
    def test_diff_exports_names_changed_added_removed(self, model):
        document = json.loads(to_json(model))
        assert not diff_exports(document, model)
        # change one, drop one, add one
        records = document["datasets"]
        records[0]["records_display"] = "changed"
        removed = records.pop(1)["dataset_id"]
        changed = records[0]["dataset_id"]
        diff = diff_exports(document, model)
        assert diff.changed == (changed,)
        assert diff.added == (removed,)
        assert diff.removed == ()
        assert changed in diff.summary() and removed in diff.summary()


class TestMetadataCli:
    def test_build_check_passes_on_a_clean_tree(self, capsys):
        assert main(["metadata", "build", "--check"]) == 0
        assert "up to date" in capsys.readouterr().out

    def test_build_check_fails_after_a_catalogue_change(self, capsys, monkeypatch):
        import dataclasses

        import ecgbench.catalogue as cat

        entries = tuple(
            dataclasses.replace(e, records="1") if e.slug == "ptb-xl" else e for e in cat._load()
        )
        monkeypatch.setattr(cat, "_load", lambda: entries)
        assert main(["metadata", "build", "--check"]) == 1
        err = capsys.readouterr().err
        assert "stale" in err and "changed (1): ptbxl" in err

    def test_build_into_an_output_directory(self, tmp_path: Path, capsys):
        assert main(["metadata", "build", "--output", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert (tmp_path / "metadata.json").is_file()
        assert (tmp_path / "metadata.sqlite").is_file()
        assert "fts: fts5" in out
        # and --check against that directory is clean
        assert main(["metadata", "build", "--check", "--output", str(tmp_path)]) == 0

    def test_metadata_requires_an_action(self):
        with pytest.raises(SystemExit) as exc:
            main(["metadata"])
        assert exc.value.code != 0


class TestSearchCli:
    def test_search_json_is_ranked_and_pure(self, capsys):
        assert main(["search", "atrial", "--leads", "2", "--format", "json"]) == 0
        rows = json.loads(capsys.readouterr().out)
        assert rows and rows[0]["rank"] == 1
        assert all(r["signal"]["leads"] == 2 for r in rows)
        assert "afdb" in {r["dataset_id"] for r in rows}

    def test_search_table_and_limit(self, capsys):
        assert main(["search", "holter", "--limit", "3"]) == 0
        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if line.strip()]
        assert lines[0].startswith("rank") and "score" in lines[0]
        assert len(lines) == 2 + 3

    def test_search_filters_only(self, capsys):
        assert main(["search", "--published", "--no-patient-id", "--format", "csv"]) == 0
        out = capsys.readouterr().out.strip().splitlines()
        assert out[0].startswith("rank,dataset_id")
        assert len(out) > 1

    def test_search_bad_query_exits_nonzero(self, capsys):
        assert main(["search", "ptb-xl"]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "invalid search query" in captured.err

    def test_search_no_match(self, capsys):
        assert main(["search", "zzzzqqq"]) == 0
        assert "no datasets match" in capsys.readouterr().out

    def test_python_api(self):
        from ecgbench.cli import run_search

        assert run_search("ptb")[0].dataset_id == "ptbxl"
        assert isinstance(ecgbench.metadata.search_ranked("ptb")[0], SearchHit)
