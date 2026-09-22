"""Tests for the label field inventory (metadata layer, Phase 3).

The one test that matters is the consistency check: for every dataset whose
fields are declared, ``set(FIELDS names) == set(load_labels(...).columns)`` on
a synthetic copy of the source. A declaration that names a column the loader
does not return, or misses one it does, fails here — so the docstring-turned-
data cannot rot.

Coverage is tracked explicitly. ``BUILDERS`` maps each declared dataset to a
function that writes a minimal synthetic source tree; ``PENDING`` lists the
label-bearing datasets not yet declared. A dataset must be in exactly one of
the two, and a module that grows a ``FIELDS`` must be moved into ``BUILDERS`` in
the same change (``test_every_declaration_has_a_consistency_builder``).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecgbench.cli import main
from ecgbench.config import LabelConfig, LabelFieldConfig, list_available_configs, load_config
from ecgbench.labels import _custom_loaders, load_labels
from ecgbench.labels._fields import (
    Field,
    FieldDeclarationError,
    declared_fields_from_source,
    fields_for,
    fields_from_config,
    has_label_module,
    to_frictionless,
    validate_type,
)
from ecgbench.metadata import FieldMeta, open_store
from tests.conftest import _write_hea

# --------------------------------------------------------------------------- builders


def _build_ptbxl(tmp_path: Path) -> Path:
    """Two records with every passthrough column the loader copies."""
    pd.DataFrame({
        "ecg_id": [1, 2],
        "patient_id": [10, 11],
        "scp_codes": ["{'NORM': 100.0, 'SR': 0.0}", "{'IMI': 100.0, 'NDT': 50.0}"],
        "age": [56, 62],
        "sex": [1, 0],
        "height": [170.0, None],
        "weight": [70.0, None],
        "report": ["sinusrhythmus", "infarkt"],
        "heart_axis": ["MID", "LAD"],
        "recording_date": ["1984-11-09 09:17:34", "1984-11-14 12:55:37"],
        "device": ["CS-12   E", "CS-12   E"],
        "validated_by_human": [True, True],
        "strat_fold": [1, 10],
        "filename_lr": ["records100/00000/00001_lr"] * 2,
        "filename_hr": ["records500/00000/00001_hr"] * 2,
    }).to_csv(tmp_path / "ptbxl_database.csv", index=False)
    pd.DataFrame({
        "index": ["NORM", "IMI", "NDT", "SR"],
        "diagnostic": [1, 1, 1, 0],
        "form": [0, 0, 1, 0],
        "rhythm": [0, 0, 0, 1],
        "diagnostic_class": ["NORM", "MI", "STTC", None],
        "diagnostic_subclass": ["NORM", "IMI", "STTC", None],
    }).set_index("index").to_csv(tmp_path / "scp_statements.csv")
    return tmp_path


def _build_mitdb(tmp_path: Path) -> Path:
    wfdb = pytest.importorskip("wfdb")
    (tmp_path / "RECORDS").write_text("100\n201\n", encoding="utf-8")
    (tmp_path / "100.hea").write_text(
        "100 2 360 650000\n"
        "100.dat 212 200 11 1024 995 -22131 0 MLII\n"
        "100.dat 212 200 11 1024 1011 20052 0 V5\n"
        "# 69 M 1085 1629 x1\n"
        "# Aldomet, Inderal\n",
        encoding="utf-8",
    )
    (tmp_path / "201.hea").write_text(
        "201 2 360 650000\n"
        "201.dat 212 200 11 1024 1024 -29350 0 MLII\n"
        "201.dat 212 200 11 1024 1049 -32096 0 V1\n"
        "# 68 M 1960 2851 x1\n"
        "# Digoxin\n"
        "# Same tape as 202.\n",
        encoding="utf-8",
    )
    for name in ("100", "201"):
        wfdb.wrann(
            name, "atr",
            sample=np.array([18, 77, 370, 662, 946]),
            symbol=["+", "N", "V", "+", "N"],
            aux_note=["(N", "", "", "(AFIB", ""],
            fs=360,
            write_dir=str(tmp_path),
        )
    return tmp_path


def _build_afdb(tmp_path: Path) -> Path:
    wfdb = pytest.importorskip("wfdb")
    (tmp_path / "RECORDS").write_text("04015\n", encoding="utf-8")
    (tmp_path / "04015.hea").write_text(
        "04015 2 250 9205760  9:00:00\n"
        "04015.dat 212 0 12 0 -55 -27172 0 ECG1\n"
        "04015.dat 212 0 12 0 -42 -28460 0 ECG2\n",
        encoding="utf-8",
    )
    wfdb.wrann(
        "04015", "atr",
        sample=np.array([0, 100_000, 900_000]),
        symbol=["+", "+", "+"],
        aux_note=["(N", "(AFIB", "(N"],
        fs=250,
        write_dir=str(tmp_path),
    )
    wfdb.wrann(
        "04015", "qrs",
        sample=np.arange(250, 9_000_000, 200),
        symbol=["N"] * len(range(250, 9_000_000, 200)),
        fs=250,
        write_dir=str(tmp_path),
    )
    return tmp_path


def _build_challenge(tmp_path: Path, cohorts: tuple[str, ...]) -> Path:
    for i, cohort in enumerate(cohorts):
        d = tmp_path / "training" / cohort / "g1"
        d.mkdir(parents=True)
        _write_hea(d / f"A{i:04d}.hea", f"A{i:04d}", dx="164889003,59118001", age="60")
        _write_hea(d / f"B{i:04d}.hea", f"B{i:04d}", dx="426783006", sex="Female")
    return tmp_path


def _build_challenge2020(tmp_path: Path) -> Path:
    return _build_challenge(tmp_path, ("cpsc_2018", "georgia"))


def _build_challenge2021(tmp_path: Path) -> Path:
    return _build_challenge(tmp_path, ("ptb-xl", "ningbo"))


def _build_mimic_iv_ecg(tmp_path: Path) -> Path:
    data = {
        "study_id": [1, 2, 3],
        "cart_id": [10, 10, 11],
        "ecg_time": ["2180-07-23 08:44:00"] * 3,
        "report_0": ["Sinus rhythm.", "Atrial fibrillation", "  Sinus   Rhythm  "],
        "report_1": [None, "Abnormal ECG", None],
        "rr_interval": [800, 65535, 900],
        "p_onset": [40, 29999, 42],
        "p_end": [150, 29999, 152],
        "qrs_onset": [200, 210, 205],
        "qrs_end": [290, 300, 299],
        "t_end": [610, 620, 615],
        "p_axis": [50, 32767, 55],
        "qrs_axis": [13, 20, -32768],
        "t_axis": [42, 40, 45],
        "bandwidth": ["0.5-40"] * 3,
        "filtering": ["60Hz"] * 3,
    }
    df = pd.DataFrame(data)
    for i in range(18):
        if f"report_{i}" not in df.columns:
            df[f"report_{i}"] = None
    df.to_csv(tmp_path / "machine_measurements.csv", index=False)
    return tmp_path


def _build_brugada_huca(tmp_path: Path) -> Path:
    pd.DataFrame({
        "patient_id": [188981, 251972],
        "basal_pattern": [1, 0],
        "sudden_death": [0, 1],
        "brugada": [1, 2],
    }).to_csv(tmp_path / "metadata.csv", index=False)
    return tmp_path


def _build_ecg_arrhythmia(tmp_path: Path) -> Path:
    pd.DataFrame({
        "record_name": ["JS00001", "JS00002"],
        "signal_path": ["WFDBRecords/01/010/JS00001", "WFDBRecords/01/010/JS00002"],
        "dx": ["426177001,164934002", "164889003"],
        "dx_acronyms": ["SB,TWC", "AFIB"],
        "primary_dx": ["426177001", "164889003"],
        "primary_dx_acronym": ["SB", "AFIB"],
        "age": [60, 71],
        "sex": ["Male", "Female"],
    }).to_csv(tmp_path / "ecgbench_metadata.csv", index=False)
    return tmp_path


#: Dataset -> builder writing a minimal synthetic source tree into tmp_path.
BUILDERS = {
    "ptbxl": _build_ptbxl,
    "mitdb": _build_mitdb,
    "afdb": _build_afdb,
    "challenge2020": _build_challenge2020,
    "challenge2021": _build_challenge2021,
    "mimic_iv_ecg": _build_mimic_iv_ecg,
    "brugada_huca": _build_brugada_huca,
    "ecg_arrhythmia": _build_ecg_arrhythmia,
}

#: Label-bearing datasets whose fields are not declared yet (later Phase 3 batches).
PENDING = {
    "apnea_ecg", "butqdb", "challenge2017", "chapman_shaoxing", "chfdb", "code15",
    "code_test", "cpsc_2018", "ecg_capable_smartwatches", "ecgcipa", "ecgdmmld", "ecgiddb",
    "ecgrdvq", "echonext", "edb", "edgar", "ikem", "incartdb", "leipzig_heart_center_ecg",
    "ltafdb", "ltstdb", "ludb", "medalcare_xl", "mhd_effect_ecg_mri", "ningbo_iva",
    "norwegian_athlete_ecg", "nsrdb", "picsdb", "ptbdb", "qtdb", "sami_trop", "sddb",
    "shdb_af", "sph", "staffiii", "stdb", "svdb", "szdb", "tollet", "ucddb", "wctecgdb",
    "zzu_pecg",
}


# --------------------------------------------------------------------------- coverage


def _label_bearing() -> set[str]:
    out = set()
    for slug in list_available_configs():
        cfg = load_config(slug)
        if cfg.labels is not None and not cfg.labels.available:
            continue
        out.add(slug)
    return out


class TestCoverage:
    def test_every_label_bearing_dataset_is_declared_or_pending(self):
        bearing = _label_bearing()
        assert set(BUILDERS) | PENDING == bearing
        assert not set(BUILDERS) & PENDING

    def test_every_declaration_has_a_consistency_builder(self):
        """A module that grows FIELDS must join BUILDERS in the same change."""
        declared = {
            slug for slug in list_available_configs()
            if fields_for(load_config(slug), static=True)
        }
        assert declared == set(BUILDERS)

    def test_pending_datasets_declare_nothing_yet(self):
        for slug in sorted(PENDING):
            assert fields_for(load_config(slug), static=True) == (), slug

    def test_label_module_files_match_the_registry(self):
        registered = set(_custom_loaders())
        assert {s for s in list_available_configs() if has_label_module(s)} == registered


# --------------------------------------------------------------------------- consistency


@pytest.mark.parametrize("slug", sorted(BUILDERS))
def test_declared_fields_match_the_loader_columns(slug, tmp_path):
    config = load_config(slug)
    data_path = BUILDERS[slug](tmp_path)
    df = load_labels(config, data_path=data_path)
    declared = fields_for(config)
    names = [f.name for f in declared]
    assert len(names) == len(set(names)), f"{slug}: duplicate field names"
    assert set(names) == set(df.columns), (
        f"{slug}: declared {sorted(set(names) - set(df.columns))} not returned; "
        f"returned {sorted(set(df.columns) - set(names))} not declared"
    )
    assert df.index.name == config.record_id_column


@pytest.mark.parametrize("slug", sorted(s for s in BUILDERS if has_label_module(s)))
def test_static_and_imported_declarations_agree(slug):
    config = load_config(slug)
    assert fields_for(config, static=True) == fields_for(config)
    assert declared_fields_from_source(slug) == fields_for(config)


def test_ptbxl_superclasses_carry_the_vocabulary():
    by_name = {f.name: f for f in fields_for(load_config("ptbxl"))}
    assert by_name["superclasses"].type == "array[string]"
    assert by_name["superclasses"].vocabulary == ("NORM", "MI", "STTC", "CD", "HYP")
    assert by_name["age"].unit == "year"
    assert by_name["scp_codes"].nullable is False


def test_mitdb_declares_the_recorder_and_every_beat_symbol():
    from ecgbench.labels.mitdb import BEAT_SYMBOLS, RHYTHM_NAMES

    by_name = {f.name: f for f in fields_for(load_config("mitdb"))}
    assert "Del Mar" in by_name["recorder"].description
    assert {f"beat_{s}" for s in BEAT_SYMBOLS} <= set(by_name)
    assert {f"rhythm_secs_{c}" for c in RHYTHM_NAMES} <= set(by_name)
    assert by_name["rhythm_secs_AFIB"].unit == "s"


def test_declarative_fields_come_from_the_config():
    fields = fields_for(load_config("brugada_huca"))
    assert [f.name for f in fields] == ["brugada", "basal_pattern", "sudden_death"]
    assert all(f.source == "config" for f in fields)
    assert fields[0].type == "integer" and fields[0].vocabulary == ("0", "1", "2")


# --------------------------------------------------------------------------- unit


class TestField:
    def test_type_validation(self):
        for ok in ("string", "integer", "array", "array[string]", "object", "datetime"):
            validate_type(ok)
        for bad in ("str", "array[array]", "list", "int", "array[]"):
            with pytest.raises(ValueError):
                validate_type(bad)
        with pytest.raises(ValueError):
            Field("x", "text")

    def test_array_item_type_and_vocabulary_coercion(self):
        f = Field("codes", "array[string]", vocabulary=["A", 1])
        assert f.base_type == "array" and f.item_type == "string"
        assert f.vocabulary == ("A", "1")
        assert Field("n", "integer").base_type == "integer"
        assert Field("n", "integer").item_type is None

    def test_source_is_restricted(self):
        with pytest.raises(ValueError, match="source"):
            Field("x", "string", source="yaml")


class TestStaticReader:
    def _module(self, tmp_path: Path, body: str) -> Path:
        (tmp_path / "toy.py").write_text(body, encoding="utf-8")
        return tmp_path

    def test_reads_a_literal_tuple(self, tmp_path):
        d = self._module(
            tmp_path,
            'from ecgbench.labels._fields import Field\n'
            'FIELDS = (\n'
            '    Field("a", "integer", "count", unit="ms", nullable=False),\n'
            '    Field("b", "array[string]", vocabulary=("x", "y")),\n'
            ')\n',
        )
        fields = declared_fields_from_source("toy", labels_dir=d)
        assert fields == (
            Field("a", "integer", "count", unit="ms", nullable=False),
            Field("b", "array[string]", vocabulary=("x", "y")),
        )

    def test_module_without_fields_is_none(self, tmp_path):
        d = self._module(tmp_path, "X = 1\n")
        assert declared_fields_from_source("toy", labels_dir=d) is None
        assert declared_fields_from_source("missing", labels_dir=d) is None

    def test_comprehension_is_rejected_with_a_helpful_error(self, tmp_path):
        d = self._module(
            tmp_path,
            'from ecgbench.labels._fields import Field\n'
            'NAMES = ["a", "b"]\n'
            'FIELDS = tuple(Field(n, "string") for n in NAMES)\n',
        )
        with pytest.raises(FieldDeclarationError, match="literal tuple"):
            declared_fields_from_source("toy", labels_dir=d)

    def test_name_reference_inside_a_call_is_rejected(self, tmp_path):
        d = self._module(
            tmp_path,
            'from ecgbench.labels._fields import Field\n'
            'UNIT = "ms"\n'
            'FIELDS = (Field("a", "integer", unit=UNIT),)\n',
        )
        with pytest.raises(FieldDeclarationError, match="without importing"):
            declared_fields_from_source("toy", labels_dir=d)

    def test_invalid_field_is_reported_with_the_line(self, tmp_path):
        d = self._module(
            tmp_path,
            'from ecgbench.labels._fields import Field\n'
            'FIELDS = (\n'
            '    Field("a", "text"),\n'
            ')\n',
        )
        with pytest.raises(FieldDeclarationError, match="toy.py:3"):
            declared_fields_from_source("toy", labels_dir=d)


class TestDeclarativeFields:
    def test_columns_only_are_strings(self):
        spec = LabelConfig(source_csv="x.csv", join_column="id", columns=["a", "b"])
        assert fields_from_config(spec) == (
            Field("a", "string", source="config"),
            Field("b", "string", source="config"),
        )

    def test_fields_block_refines_and_extends(self):
        spec = LabelConfig(
            source_csv="x.csv",
            join_column="id",
            columns=["a"],
            fields={
                "a": LabelFieldConfig(type="integer", description="count", unit="ms"),
                "c": LabelFieldConfig(vocabulary=["x", "y"], nullable=False),
            },
        )
        fields = fields_from_config(spec)
        assert [f.name for f in fields] == ["a", "c"]
        assert fields[0].type == "integer" and fields[0].unit == "ms"
        assert fields[1].vocabulary == ("x", "y") and fields[1].nullable is False

    def test_unavailable_labels_have_no_fields(self):
        assert fields_from_config(LabelConfig(available=False)) == ()
        assert fields_from_config(None) == ()
        assert fields_for(load_config("mimic_iv_ecg_demo")) == ()

    def test_shipped_yaml_parses_the_fields_block(self):
        spec = load_config("ecg_arrhythmia").labels
        assert spec.fields is not None
        assert spec.fields["sex"].vocabulary == ["Male", "Female"]
        assert spec.fields["age"].type == "integer" and spec.fields["age"].unit == "year"


class TestFrictionless:
    def test_table_schema_shape(self):
        fields = (
            Field("id", "integer", nullable=False),
            Field("codes", "array[string]", "diagnoses", vocabulary=("A", "B")),
            Field("age", "integer", unit="year", vocabulary=("1", "2")),
            Field("flag", "boolean", vocabulary=("true", "false")),
        )
        schema = to_frictionless(fields, primary_key="id")
        assert schema["primaryKey"] == "id"
        by_name = {f["name"]: f for f in schema["fields"]}
        assert by_name["id"]["constraints"] == {"required": True}
        assert by_name["codes"]["type"] == "array"
        assert by_name["codes"]["arrayItem"] == {"type": "string"}
        assert by_name["codes"]["constraints"]["enum"] == ["A", "B"]
        assert by_name["age"]["constraints"]["enum"] == [1, 2]
        assert by_name["age"]["unit"] == "year"
        assert by_name["flag"]["constraints"]["enum"] == [True, False]


# --------------------------------------------------------------------------- metadata + CLI


class TestFieldsInTheMetadataLayer:
    def test_model_carries_the_fields(self):
        meta = open_store().get("ptbxl")
        assert isinstance(meta.fields[0], FieldMeta)
        assert {f.name for f in meta.fields} == {f.name for f in fields_for(load_config("ptbxl"))}
        assert open_store().get("qtdb").fields == ()  # pending
        assert open_store().get("ptb-xl-plus").fields == ()  # catalogue-only

    def test_field_text_is_searchable(self):
        store = open_store()
        if not store.fts_enabled:
            pytest.skip(store.fts_fallback_reason)
        assert "mitdb" in {m.dataset_id for m in store.search("recorder")}
        hits = store.search("superclasses")
        assert hits and hits[0].dataset_id == "ptbxl"

    def test_sqlite_field_table_is_populated(self):
        import sqlite3

        from ecgbench.metadata import SQLITE_PATH

        open_store()
        conn = sqlite3.connect(SQLITE_PATH)
        rows = conn.execute(
            "SELECT name, type, unit FROM field WHERE dataset_id = 'mitdb' ORDER BY position"
        ).fetchall()
        assert rows[0][0] == "age"
        assert ("recorder", "string", None) in rows
        n_declared = sum(len(fields_for(load_config(s), static=True)) for s in BUILDERS)
        assert conn.execute("SELECT count(*) FROM field").fetchone()[0] == n_declared


class TestFieldsCli:
    def test_table_lists_superclasses_with_the_vocabulary(self, capsys):
        assert main(["fields", "ptbxl"]) == 0
        out = capsys.readouterr().out
        assert "superclasses" in out and "array[string]" in out
        assert "NORM, MI, STTC, CD, HYP" in out

    def test_json_and_csv(self, capsys):
        assert main(["fields", "mit-bih-arrhythmia-database", "--format", "json"]) == 0
        rows = json.loads(capsys.readouterr().out)
        assert {r["name"] for r in rows} >= {"recorder", "beat_V", "rhythm_secs_AFIB"}
        assert main(["fields", "afdb", "--format", "csv"]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert lines[0] == "name,type,unit,nullable,vocabulary,description"
        assert any(line.startswith("af_burden,number,") for line in lines)

    def test_frictionless_schema(self, capsys):
        assert main(["fields", "ptbxl", "--format", "frictionless"]) == 0
        schema = json.loads(capsys.readouterr().out)
        assert schema["primaryKey"] == "ecg_id"
        codes = next(f for f in schema["fields"] if f["name"] == "superclasses")
        assert codes["type"] == "array" and codes["arrayItem"] == {"type": "string"}
        assert codes["constraints"]["enum"] == ["NORM", "MI", "STTC", "CD", "HYP"]

    def test_undeclared_and_unlabelled_datasets_say_so(self, capsys):
        assert main(["fields", "qtdb"]) == 0
        assert "not yet declared" in capsys.readouterr().out
        assert main(["fields", "mimic_iv_ecg_demo"]) == 0
        assert "no labels" in capsys.readouterr().out
        assert main(["fields", "qtdb", "--format", "json"]) == 0
        assert json.loads(capsys.readouterr().out) == []

    def test_unknown_dataset_exits_nonzero(self, capsys):
        assert main(["fields", "nope"]) == 1
        assert "unknown dataset" in capsys.readouterr().err

    def test_python_api(self):
        from ecgbench.cli import run_fields

        assert {f.name for f in run_fields("PTB-XL")} >= {"superclasses", "strat_fold"}


def test_conftest_ptbxl_fixture_matches_the_declaration(ptbxl_config, tmp_ptbxl_label_data):
    """The shared PTB-XL fixture must also return every declared column."""
    config = replace(
        ptbxl_config, labels=LabelConfig(source_csv="ptbxl_database.csv", join_column="ecg_id")
    )
    df = load_labels(config, data_path=tmp_ptbxl_label_data)
    assert set(df.columns) == {f.name for f in fields_for(load_config("ptbxl"))}
