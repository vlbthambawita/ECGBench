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


def _build_apnea_ecg(tmp_path: Path) -> Path:
    wfdb = pytest.importorskip("wfdb")
    records = {"a01": "A" * 120 + "N" * 20, "c01": "N" * 100}
    for rec, sequence in records.items():
        n_minutes = len(sequence)
        n_samples = n_minutes * 6000
        (tmp_path / f"{rec}.hea").write_text(
            f"{rec} 1 100 {n_samples}\n{rec}.dat 16 200 12 0 -12 5827 0 ECG\n", encoding="utf-8"
        )
        np.zeros(n_samples, dtype=np.int16).tofile(tmp_path / f"{rec}.dat")
        wfdb.wrann(
            rec, "apn",
            sample=np.arange(n_minutes, dtype=np.int64) * 6000,
            symbol=list(sequence),
            fs=100,
            write_dir=str(tmp_path),
        )
        beats = np.arange(1, n_minutes * 60) * 100
        wfdb.wrann(
            rec, "qrs",
            sample=np.concatenate([[50], beats]),
            symbol=["|"] + ["N"] * len(beats),
            fs=100,
            write_dir=str(tmp_path),
        )
    (tmp_path / "RECORDS").write_text("a01\nc01\n", encoding="utf-8")
    (tmp_path / "additional-information.txt").write_text(
        "Additional information about the recordings\n\n"
        "Record\tLength\tnon-apn\tapnea\thours\tAI\tHI\tAHI\tAge\tSex\theight\tweight\n"
        "\tminutes\tminutes\tminutes\tw/apnea\t\t\t\t\t\t(cm)\t(kg)\n\n"
        "a01\t490\t20\t470\t9\t12.5\t57.1\t69.6\t51\tM\t175\t102\t\t\n"
        "c01\t485\t485\t0\t0\t0\t0\t0\t31\tM\t184\t74\t\t\n",
        encoding="utf-8",
    )
    return tmp_path


def _build_butqdb(tmp_path: Path) -> Path:
    n = 1000

    def ann_csv(per_annotator):
        columns = [[(str(a + 1), str(b), str(k)) for a, b, k in iv] for iv in per_annotator]
        height = max(len(c) for c in columns)
        lines = []
        for row in range(height):
            fields = []
            for column in columns:
                fields.extend(column[row] if row < len(column) else ("", "", ""))
            lines.append(",".join(fields))
        return "\n".join(lines) + "\n"

    full = [
        [(0, 700, 2), (700, 900, 2), (900, 1000, 3)],
        [(0, 700, 1), (700, 900, 2), (900, 1000, 3)],
        [(0, 700, 1), (700, 900, 2), (900, 1000, 3)],
        [(0, 700, 1), (700, 900, 2), (900, 1000, 3)],
    ]
    one_block = [[(0, 400, 2), (400, 1000, 0)]] * 4
    plan = {"200001": (full, 0.99998, 0), "201001": (one_block, 1.996, -12200)}
    for record_id, (per_annotator, gain, baseline) in plan.items():
        directory = tmp_path / record_id
        directory.mkdir()
        (directory / f"{record_id}_ECG.hea").write_text(
            f"{record_id}_ECG 1 1000 {n}\n"
            f"{record_id}_ECG.dat 16 {gain}({baseline})/uV 0 0 0 0 0 ECG\n#ECG\n"
        )
        samples = np.arange(n, dtype="<i2")
        samples[0] = samples[1] = 32767
        samples[2] = samples[3] = -32767
        (directory / f"{record_id}_ECG.dat").write_bytes(samples.tobytes())
        (directory / f"{record_id}_ANN.csv").write_text(ann_csv(per_annotator))
    (tmp_path / "RECORDS").write_text(
        "\n".join(f"{r}/{r}_{kind}" for r in plan for kind in ("ACC", "ECG")) + "\n"
    )
    (tmp_path / "subject-info.csv").write_text(
        "ID;Gender;Age;Height;Weight;Smoker\n200001;F;30;170;65;0\n201001;M;44;180;80;1\n"
    )
    return tmp_path


def _build_challenge2017(tmp_path: Path) -> Path:
    training = tmp_path / "training"
    records = {"A00/A00001": "N", "A00/A00002": "A"}
    for relative, code in records.items():
        (training / relative).parent.mkdir(parents=True, exist_ok=True)
        name = relative.rsplit("/", 1)[-1]
        (training / f"{relative}.hea").write_text(
            f"{name} 1 300 9000 05:05:15 1/05/2000 \n"
            f"{name}.mat 16+24 1000/mV 16 0 -127 0 0 ECG \n",
            encoding="utf-8",
        )
    (training / "RECORDS").write_text("".join(f"{r}\n" for r in records), encoding="utf-8")
    for version in (0, 1, 2, 3):
        (training / f"REFERENCE-v{version}.csv").write_text(
            "".join(f"{r},{c}\n" for r, c in records.items()), encoding="utf-8"
        )
    (training / "REFERENCE.csv").write_text(
        "".join(f"{r},{c}\n" for r, c in records.items()), encoding="utf-8"
    )
    (tmp_path / "validation").mkdir()
    (tmp_path / "validation" / "RECORDS").write_text("A00/A00001\n", encoding="utf-8")
    return tmp_path


def _build_chfdb(tmp_path: Path) -> Path:
    wfdb = pytest.importorskip("wfdb")
    (tmp_path / "RECORDS").write_text("chf01\n", encoding="utf-8")
    (tmp_path / "chf01.hea").write_text(
        "chf01 2 250 2500 10:00:00\n"
        "chf01.dat 212 0 12 0 127 17579 0 ECG1\n"
        "chf01.dat 212 0 12 0 -128 21162 0 ECG2\n"
        "#Age: 71  Sex: M  NYHA class: III-IV\n",
        encoding="utf-8",
    )
    wfdb.wrann(
        "chf01", "ecg",
        sample=np.array([100, 350, 600, 850, 1100, 1350, 1600, 1850]),
        symbol=["N", "r", "N", "V", "S", "+", "N", "N"],
        subtype=np.zeros(8, dtype=int),
        aux_note=["", "", "", "", "", "(AF", "", ""],
        fs=250,
        write_dir=str(tmp_path),
    )
    return tmp_path


def _build_code15(tmp_path: Path) -> Path:
    pd.DataFrame({
        "exam_id": [1, 2, 3],
        "age": [50, 61, 44],
        "is_male": [True, False, True],
        "nn_predicted_age": [51.0, 60.2, 45.1],
        "1dAVb": [False, True, False],
        "RBBB": [False, False, False],
        "LBBB": [False, False, False],
        "SB": [False, False, True],
        "ST": [False, False, False],
        "AF": [False, True, False],
        "patient_id": [100, 101, 100],
        "death": ["False", None, "True"],
        "timey": [1.0, None, 3.5],
        "normal_ecg": [True, False, False],
        "trace_file": ["exams_part0.hdf5"] * 3,
    }).to_csv(tmp_path / "exams.csv", index=False)
    return tmp_path


def _build_code_test(tmp_path: Path) -> Path:
    from ecgbench.labels.code_test import ABNORMALITIES, ANNOTATORS, N_RECORDS

    (tmp_path / "annotations").mkdir()
    (tmp_path / "attributes.csv").write_text(
        "age,sex\n" + "".join(f"{30 + i % 50},{'MF'[i % 2]}\n" for i in range(N_RECORDS)),
        encoding="utf-8",
    )
    for name in ANNOTATORS:
        header = ("," if name == "dnn" else "") + ",".join(ABNORMALITIES)
        lines = [header]
        for i in range(N_RECORDS):
            cells = ["1" if (i % 7 == j) else "0" for j in range(len(ABNORMALITIES))]
            lines.append(",".join(([str(i)] if name == "dnn" else []) + cells))
        (tmp_path / "annotations" / f"{name}.csv").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    return tmp_path


def _build_cpsc_2018(tmp_path: Path) -> Path:
    d = tmp_path / "Training_WFDB"
    d.mkdir()
    _write_hea(d / "A0001.hea", "A0001", dx="164889003,59118001", age="60")
    _write_hea(d / "A0002.hea", "A0002", dx="426783006", sex="Female", age="-1")
    return tmp_path


def _build_chapman_shaoxing(tmp_path: Path) -> Path:
    pd.DataFrame({
        "FileName": ["MUSE_A", "MUSE_B"],
        "Rhythm": ["AFIB", "SB"],
        "Beat": ["RBBB TWC", "NONE"],
        "PatientAge": [85, 59],
        "Gender": ["MALE", "FEMALE"],
        "VentricularRate": [117, 52],
        "AtrialRate": [234, 52],
        "QRSDuration": [114, 92],
        "QTInterval": [356, 432],
        "QTCorrected": [496, 402],
        "RAxis": [81, 61],
        "TAxis": [-27, 50],
        "QRSCount": [19, 9],
        "QOnset": [208, 214],
        "QOffset": [265, 260],
        "TOffset": [386, 430],
    }).to_csv(tmp_path / "Diagnostics.csv", index=False)
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
    # batch 2
    "apnea_ecg": _build_apnea_ecg,
    "butqdb": _build_butqdb,
    "challenge2017": _build_challenge2017,
    "chapman_shaoxing": _build_chapman_shaoxing,
    "chfdb": _build_chfdb,
    "code15": _build_code15,
    "code_test": _build_code_test,
    "cpsc_2018": _build_cpsc_2018,
}

#: Label-bearing datasets whose fields are not declared yet (later Phase 3 batches).
PENDING = {
    "ecg_capable_smartwatches", "ecgcipa", "ecgdmmld", "ecgiddb",
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
