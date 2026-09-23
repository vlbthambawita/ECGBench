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

import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecgbench.cli import main
from ecgbench.cli.catalog import format_fields
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
from tests.conftest import _write_hea, write_edf

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


def _build_ecg_capable_smartwatches(tmp_path: Path) -> Path:
    """One record per device, covering the three layouts the scan distinguishes.

    A 12-lead Philips reference, four single-lead watches; a 4-part
    ``<device>/<family>/<setting>/<name>`` path and the 3-part ``sqr-2hz`` one;
    Fitbit's upper-case ST directory (so ``setting_id`` is lowercased) and the
    Samsung record ending in WFDB's invalid-sample marker.
    """
    wfdb = pytest.importorskip("wfdb")
    from ecgbench.labels.ecg_capable_smartwatches import DEVICES

    twelve = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    records = {
        "philips_tc30": ("amp_test/amp1000/amp1000_0", twelve, 500, 5500, False),
        "applewatch_serie8": ("freq_test/f80/f80_0", ["II"], 512, 15360, False),
        "samsunggalaxy6": ("st-segment/st-p8/st-p8_0", ["II"], 500, 15001, True),
        "fitbitsense2": ("st-segment/ST-m1/ST-m1_5", ["II"], 250, 7500, False),
        "withingsscanwatch": ("sqr-2hz/sqr-2hz_0", ["II"], 300, 9000, False),
    }
    assert set(records) == set(DEVICES)
    index = []
    for device, (relative, leads, fs, n_samples, trailing_invalid) in records.items():
        path = tmp_path / device / relative
        path.parent.mkdir(parents=True)
        digital = np.tile(
            (np.arange(n_samples) % 200 - 100).astype(np.int16)[:, None], (1, len(leads))
        )
        if trailing_invalid:
            digital[-1, :] = -32768
        wfdb.wrsamp(
            path.name,
            fs=fs,
            units=["mV"] * len(leads),
            sig_name=leads,
            d_signal=digital,
            fmt=["16"] * len(leads),
            adc_gain=[20000.0] * len(leads),
            baseline=[0] * len(leads),
            comments=[f"{DEVICES[device]['model']} reading METRON PS-440 patient simulator"],
            write_dir=str(path.parent),
        )
        index.append(f"{device}/{relative}")
    (tmp_path / "RECORDS").write_text("\n".join(index) + "\n", encoding="utf-8")
    return tmp_path


def _build_ecgcipa(tmp_path: Path) -> Path:
    """Three records of two subjects across the four CDISC tables and RECORDS."""
    from ecgbench.labels.ecgcipa import (
        ANALYTE_COLUMNS,
        CONTEXT_COLUMNS,
        INTERVAL_COLUMNS,
        SUBJECT_COLUMNS,
        VITAL_COLUMNS,
    )

    records = {"1001": ["AAAA-0001", "AAAA-0002"], "1002": ["BBBB-0001"]}
    (tmp_path / "RECORDS").write_text(
        "".join(f"raw/{s}/{r}\nmedians/{s}/{r}\n" for s, recs in records.items() for r in recs),
        encoding="utf-8",
    )
    adeg = []
    for subject, recs in records.items():
        context = {
            "STUDYID": "CiPA", "USUBJID": subject, "TRTA": "Dofetilide", "TRTP": "Dofetilide",
            "TRTSEQA": "ABCDE", "APERIOD": 1, "APERIODC": "Period 1", "ATPT": "2 hrs",
            "ATPTN": 2, "NRRLT": 2.0, "ARRLT": 2.1, "ADTM": "2015-01-01T10:00",
            "ADY": 1, "APERDAY": 1,
        }
        assert set(context) == set(CONTEXT_COLUMNS)
        for i, rec in enumerate(recs):
            for code in INTERVAL_COLUMNS:
                adeg.append({
                    "EGREFID": rec, "PARAMCD": code, "AVAL": 100.0, "DTYPE": None,
                    "AEGBLFL": "Y" if i == 0 else None, "ECGPCFL": "Y", "EGREPNUM": i + 1,
                    **context,
                })
        adeg.append({
            "EGREFID": None, "PARAMCD": "QT", "AVAL": 400.0, "DTYPE": "AVERAGE",
            "AEGBLFL": None, "ECGPCFL": None, "EGREPNUM": None, **context,
        })
    pd.DataFrame(adeg).to_csv(tmp_path / "adeg.csv", index=False)
    pd.DataFrame([
        {
            "USUBJID": subject, "APERIOD": 1, "ATPTN": 2, "PARAMCD": code,
            "AVAL": 0.0 if code == "DOF" and subject == "1002" else 12.5,
            "LLOQFL": "Y" if code == "DOF" and subject == "1002" else None,
        }
        for subject in records
        for code in ANALYTE_COLUMNS
    ]).to_csv(tmp_path / "adpc.csv", index=False)
    subject_values = ["30", "M", " WHITE", "NOT HISPANIC", "A", "A"]
    pd.DataFrame({
        "USUBJID": list(records),
        **{k: subject_values[i] for i, k in enumerate(SUBJECT_COLUMNS)},
    }).to_csv(tmp_path / "adsl.csv", index=False)
    pd.DataFrame([
        {"USUBJID": subject, "PARAMCD": code, "AVAL": 70.0}
        for subject in records
        for code in VITAL_COLUMNS
    ]).to_csv(tmp_path / "addm.csv", index=False)
    return tmp_path


def _clinical_rows(module, extra: dict, sequence: str) -> pd.DataFrame:
    """Three rows of one subject-period for the two FDA crossover releases."""
    rows = []
    timepoints = [("R-1", -0.5, "Y"), ("R-2", 2.0, "N"), ("R-3", 6.5, "N")]
    for i, (rec, tpt, baseline) in enumerate(timepoints):
        rows.append({
            "EGREFID": rec, "BASELINE": baseline, "RANDID": 1001, "ARMCD": sequence,
            "VISIT": "PERIOD-1-DOSING", "TPT": tpt,
            "RR": 900.0, "PR": 150.0, "QRS": 90.0 if i else None, "QT": 400.0,
            "JTPEAK": 200.0, "TPEAKTEND": 80.0, "TPEAKTPEAKP": None,
            "ERD_30": 30.0, "LRD_30": 40.0,
            "Twave_amplitude": 400.0, "Twave_asymmetry": 1.2, "Twave_flatness": 0.4,
            "SEX": "M", "AGE": 30, "HGHT": 180.0, "WGHT": 75.0, "SYSBP": 120, "DIABP": 80,
            "RACE": "WHITE", "ETHNIC": "NOT HISPANIC",
            **{k: (v if i else None) for k, v in extra.items()},
        })
    df = pd.DataFrame(rows)
    expected = {
        "EGREFID", "BASELINE", *module.CONTEXT_COLUMNS, *module.INTERVAL_COLUMNS,
        *module.MORPHOLOGY_COLUMNS, *module.SUBJECT_COLUMNS, *module.ANALYTE_COLUMNS,
    }
    assert set(df.columns) == expected, sorted(set(df.columns) ^ expected)
    return df


def _build_ecgdmmld(tmp_path: Path) -> Path:
    from ecgbench.labels import ecgdmmld

    df = _clinical_rows(
        ecgdmmld,
        {
            "TRTA": "Mexiletine + Dofetilide", "DOF": None, "LIDO": None, "MEXI": 500.0,
            "MOXI": None, "MOXI.M2": None, "DILT": None,
        },
        sequence="C-A-B-D-E",
    )
    df["TRTA"] = "Mexiletine + Dofetilide"
    df.to_csv(tmp_path / ecgdmmld.CLINICAL_CSV, index=False)
    return tmp_path


def _build_ecgrdvq(tmp_path: Path) -> Path:
    from ecgbench.labels import ecgrdvq

    df = _clinical_rows(
        ecgrdvq,
        {
            "EXTRT": "Dofetilide", "EXDOSE": 500, "EXDOSU": "ug", "PCTEST": "Dofetilide",
            "PCSTRESN": 1500.0, "PCSTRESU": "pg/mL",
        },
        sequence="B,A,C,D,E",
    )
    df[["EXTRT", "EXDOSE", "EXDOSU"]] = ["Dofetilide", 500, "ug"]
    df.loc[2, "PR"] = -4294966951.0  # the 32-bit wrap the loader repairs
    df.to_csv(tmp_path / ecgrdvq.CLINICAL_CSV, index=False)
    return tmp_path


def _build_ecgiddb(tmp_path: Path) -> Path:
    """Two subjects, one with two sessions; ten N/t annotation pairs per record."""
    wfdb = pytest.importorskip("wfdb")
    records = {
        "Person_01": [("rec_1", "07.12.2004", "male"), ("rec_2", "12.05.2005", "male")],
        "Person_02": [("rec_1", "07.12.2004", "female")],
    }
    index = []
    for subject, recs in records.items():
        d = tmp_path / subject
        d.mkdir()
        for name, date, sex in recs:
            (d / f"{name}.hea").write_text(
                f"{name} 2 500 10000\n"
                f"{name}.dat 16 200 12 0 0 0 0 ECG I\n"
                f"{name}.dat 16 200 12 0 0 0 0 ECG I filtered\n"
                f"# Age: 25\n# Sex: {sex}\n# ECG date: {date}\n",
                encoding="utf-8",
            )
            beats = np.arange(10) * 400 + 300
            wfdb.wrann(
                name, "atr",
                sample=np.sort(np.concatenate([beats, beats + 150])),
                symbol=["N", "t"] * 10,
                fs=500,
                write_dir=str(d),
            )
            index.append(f"{subject}/{name}")
    (tmp_path / "RECORDS").write_text("\n".join(index) + "\n", encoding="utf-8")
    return tmp_path


def _build_echonext(tmp_path: Path) -> Path:
    from ecgbench.labels.echonext import (
        CONTINUOUS_COLUMNS,
        FLAG_COLUMNS,
        ORDINAL_LEVELS,
        SOURCE_CSV,
    )

    frame = {"ecg_key": ["e1", "e2", "e3"]}
    frame.update({flag: [0, 1, 0] for flag in FLAG_COLUMNS})
    frame.update({col: [levels[0], levels[-1], None] for col, levels in ORDINAL_LEVELS.items()})
    frame.update({col: [1.5, None, 2.5] for col in CONTINUOUS_COLUMNS})
    frame.update({
        "patient_key": ["p1", "p2", "p1"], "age_at_ecg": [60, 71, 62],
        "sex": ["Male", "Female", "Male"], "acquisition_year": [2018, 2019, 2020],
        "location_setting": ["inpatient", "outpatient", "ED"],
        "race_ethnicity": ["White", "Hispanic", "Other"], "most_recent_ecg": [0, 1, 1],
        "ventricular_rate": [70, 80, 90], "atrial_rate": [70, 80, 90],
        "pr_interval": [160, 170, 150], "qrs_duration": [90, 100, 95],
        "qt_corrected": [420, 430, 410],
        "split": ["train", "val", "no_split"],
    })
    pd.DataFrame(frame).to_csv(tmp_path / SOURCE_CSV, index=False)
    return tmp_path


def _build_edb(tmp_path: Path) -> Path:
    """Two records with identical headers (one reconstructed subject), one ST and one T episode."""
    wfdb = pytest.importorskip("wfdb")
    header = (
        "{name} 2 250 1800000\n"
        "{name}.dat 212 200 12 0 91 0 0 V4\n"
        "{name}.dat 212 200 12 0 751 0 0 MLIII\n"
        "\n"
        "#Age: 62  Sex: M\n#Mixed angina\n#1-vessel disease (RCA)\n"
        "#Medications: nitrates, diltiazem\n#Recorder type: ICR 7200\n"
    )
    for name in ("e0103", "e0104"):
        (tmp_path / f"{name}.hea").write_text(header.format(name=name), encoding="utf-8")
        wfdb.wrann(
            name, "atr",
            sample=np.array(
                [2, 300, 600, 900, 1200, 1500, 2000, 4000, 6000, 8000, 9000, 10000, 12000]
            ),
            symbol=["+", "N", "N", "V", "S", "~", "s", "s", "s", '"', "T", "T", "T"],
            subtype=np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
            aux_note=[
                "(N", "", "", "", "", "", "(ST0+", "AST0+600", "ST0+)", "BUTTON",
                "(T1-", "AT1-350", "T1-)",
            ],
            fs=250,
            write_dir=str(tmp_path),
        )
    (tmp_path / "RECORDS").write_text("e0103\ne0104\n", encoding="utf-8")
    return tmp_path


def _build_edgar(tmp_path: Path) -> Path:
    """One authoritative archive per experiment, each holding one potvals struct.

    Goes through ``extract_archives`` and ``scan_records`` for real rather than
    through the ``ecgbench_metadata.csv`` cache, which would make the test compare
    the declaration against a CSV this builder wrote.
    """
    scipy_io = pytest.importorskip("scipy.io")
    from ecgbench.labels.edgar import EXPERIMENTS

    buffer = io.BytesIO()
    scipy_io.savemat(buffer, {"ts": {"potvals": np.zeros((30, 30)), "unit": "mV", "fs": 2000.0}})
    payload = buffer.getvalue()
    for exp in EXPERIMENTS.values():
        patterns = [p for p in exp.surfaces if not any(tok in p for tok in exp.exclude)]
        pattern = max(patterns, key=len)
        member = pattern.replace("*", "x") + ("" if pattern.endswith("/") else "_") + "rec.mat"
        archive = tmp_path / exp.post / exp.archive
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr(member, payload)
    return tmp_path


def _build_ikem(tmp_path: Path) -> Path:
    """exams.csv with the -1 sentinel in every numeric column, plus one HDF5 part."""
    h5py = pytest.importorskip("h5py")
    pd.DataFrame({
        "exam_id": [1, 2, 3],
        "acquisition_date": ["03-18-2004", "07-26-2022", "01-01-2017"],
        "patient_id": ["a" * 40, "b" * 40, "a" * 40],
        "age": [56, -1, 0],
        "is_male": [1, 0, -1],
        "weight": [-1, 80, -1],
        "height": [-1, 180, -1],
        "ventricular_rate": [72, 110, 50],
        "atrial_rate": [72, -1, 50],
    }).to_csv(tmp_path / "exams.csv", index=False)
    with h5py.File(tmp_path / "exams_part_0.hdf5", "w") as handle:
        handle.create_dataset("exam_id", data=np.array([1, 2, 3]))
        handle.create_dataset("real_lengths", data=np.array([4096, 2500, 4096]))
    return tmp_path


def _build_incartdb(tmp_path: Path) -> Path:
    """Two records of one patient, one header without the <diagnoses> token."""
    wfdb = pytest.importorskip("wfdb")
    comments = {
        "I01": "#<age>: 65 <sex>: F <diagnoses> Coronary artery disease, arterial hypertension",
        "I02": "#<age>: 65 <sex>: F",
    }
    for name, line in comments.items():
        (tmp_path / f"{name}.hea").write_text(
            f"{name} 2 257 5140\n"
            f"{name}.dat 16 306 16 0 0 0 0 I\n"
            f"{name}.dat 16 306 16 0 0 0 0 II\n"
            f"{line}\n# patient 1\n# PVCs, noise\n",
            encoding="utf-8",
        )
        wfdb.wrann(
            name, "atr",
            sample=np.array([100, 400, 700, 1000, 1300]),
            symbol=["+", "N", "V", "N", "R"],
            aux_note=["(N", "", "", "", ""],
            fs=257,
            write_dir=str(tmp_path),
        )
    return tmp_path


def _build_leipzig_heart_center_ecg(tmp_path: Path) -> Path:
    """One child and one adult record; X and b need the release's custom labels."""
    wfdb = pytest.importorskip("wfdb")
    pd.DataFrame({
        "subject_id": ["001"], "file_name": ["x001"], "gender": ["M"], "age": [".14.3"],
        "diagnosis": ["AVRT-WPW"], "ap_loacation": ["right posteroseptal"],
        "ecg_duration": ["0:00:02.0"],
    }).to_csv(tmp_path / "children-subject-info.csv", index=False)
    pd.DataFrame({
        "subject_id": ["100"], "file_name": ["x100"], "gender": ["F"], "age": ["64.16"],
        "diagnosis": ["TOF with VT"], "ecg_duration": ["0:00:02.0"],
    }).to_csv(tmp_path / "adults-subject-info.csv", index=False)
    twelve = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    for record, extra in (("x001", ["ABL12", "RVA12"]), ("x100", ["CS12", "RVA12"])):
        names = twelve + extra
        n = len(names)
        wfdb.wrsamp(
            record, fs=977, units=["mV"] * n, sig_name=names,
            p_signal=np.tile(np.arange(1954, dtype=np.float64)[:, None] / 1000.0, (1, n)),
            fmt=["16"] * n, adc_gain=[2000.0] * n, baseline=[0] * n, write_dir=str(tmp_path),
        )
        wfdb.wrann(
            record, "atr",
            np.array([10, 20, 30, 40, 50, 60, 70, 80]),
            np.array(["N", "X", "X", "/", "Q", "~", "+", "b"]),
            aux_note=["N-Prex", "AVRT", "AFIB", "/V", "", "", "(N", "BI"],
            fs=977, write_dir=str(tmp_path),
            custom_labels=[(42, "X", "Tachycardias"), (43, "b", "AV-Block")],
        )
    return tmp_path


def _build_ltafdb(tmp_path: Path) -> Path:
    """Two records with comment-free headers, an AFIB episode and a .qrs detector file."""
    wfdb = pytest.importorskip("wfdb")
    n_samples = 128 * 1000
    for rec in ("00", "100"):
        (tmp_path / f"{rec}.hea").write_text(
            f"{rec} 2 128 {n_samples} 9:30:00 31/01/2003\n"
            f"{rec}.dat 16 166.945/mV 0 0 -1 -8202 0 ECG\n"
            f"{rec}.dat 16 173.01/mV 0 0 3 6311 0 ECG\n",
            encoding="utf-8",
        )
        wfdb.wrann(
            rec, "atr",
            sample=np.array([10, 200, 328, 456, 128 * 400, 128 * 401, 128 * 700, 128 * 800]),
            symbol=["+", "N", "A", "V", "+", "N", '"', "N"],
            aux_note=["(N", "", "", "", "(AFIB", "", "PSE", ""],
            fs=128, write_dir=str(tmp_path),
        )
        wfdb.wrann(
            rec, "qrs",
            sample=np.array([200, 328, 456, 600]),
            symbol=["N", "N", "|", "T"],
            fs=128, write_dir=str(tmp_path),
        )
    (tmp_path / "RECORDS").write_text("00\n100\n", encoding="utf-8")
    return tmp_path


def _build_ltstdb(tmp_path: Path) -> Path:
    """One two-signal record with the full comment tree and all four annotators."""
    wfdb = pytest.importorskip("wfdb")
    name = "s20021"
    (tmp_path / f"{name}.hea").write_text(
        "s20021 2 250 18975000 11:00:00 28/02/1984\n"
        "s20021.dat 212 200/mV 12 0 6 -10121 0 MLIII\n"
        "s20021.dat 212 200/mV 12 0 -2 -17799 0 V4\n"
        "#Age: 55  Sex: M\n"
        "#Comments:\n"
        "#  An excerpt of this recording is included in the European\n"
        "#  ST-T Database (record e0113).\n"
        "#Symptoms during Holter recording: No data\n"
        "#Diagnoses: \n"
        "#  Prinzmetal's angina\n"
        "#Treatment:\n"
        "#  Medications: \n"
        "#    Nitrates\n"
        "#    Verapamil\n"
        "#  Balloon Angioplasty: No data\n"
        "#  Coronary Artery bypass Grafting: No\n"
        "#History: \n"
        "#  Smoker, hypertriglyceridemia\n"
        "#  Hypertension: No\n"
        "#  Left ventricular hypertrophy: Septum 13 mm\n"
        "#  Previous Myocardial Infarction: Yes, unknown date\n"
        "#  Intraventricular conduction block: Right bundle branch block\n"
        "#  Previous tests:\n"
        "#    ECG stress test: Yes \n"
        "#      Date: No Data\n"
        "#      Findings: ST depression V4-6\n"
        "#    Coronary Arteriography: \n"
        "#      Left anterior descending coronary artery 75% stenosis\n"
        "#Holter Recording:\n"
        "#  Date: 28/02/1984\n"
        "#  Recorder: Oxford Medilog\n",
        encoding="utf-8",
    )
    wfdb.wrann(
        name, "atr",
        sample=np.array([50, 250, 450, 650, 850]),
        symbol=["N", "N", "V", "N", "A"],
        fs=250, write_dir=str(tmp_path),
    )
    st = [
        (1000, "(st0-120", 0), (2000, "ast0-160", 0), (3000, "st0-90)", 0),
        (4000, "(rtst1+100", 1), (5000, "artst1+130", 1), (6000, "rtst1+80)", 1),
        (7000, "sst0", 0), (7500, "sccst1", 1), (8000, "noi0+50", 0),
        (9000, "(urd1", 1), (9500, "urd1)", 1),
    ]
    for ext in ("sta", "stb", "stc"):
        wfdb.wrann(
            name, ext,
            sample=np.array([s for s, _, _ in st]),
            symbol=["s"] * len(st),
            subtype=np.zeros(len(st), dtype=int),
            chan=np.array([c for _, _, c in st]),
            aux_note=[a for _, a, _ in st],
            fs=250, write_dir=str(tmp_path),
        )
    (tmp_path / "RECORDS").write_text(f"{name}\n", encoding="utf-8")
    return tmp_path


def _build_ludb(tmp_path: Path) -> Path:
    """ludb.csv with the release's newline-joined cells and the '>89' age."""
    pd.DataFrame({
        "ID": [1, 2, 34],
        "Sex": ["M\n", "F\n", "M\n"],
        "Age": ["55\n", "62\n", ">89\n"],
        "Rhythms": ["Sinus rhythm\n", "Sinus bradycardia\n", "Sinus rhythm\n"],
        "Conduction abnormalities": ["", "Incomplete right bundle branch block\n", ""],
        "Extrasystolies": ["", "", "Atrial extrasystole: undefined\n"],
        "Hypertrophies": ["Left ventricular hypertrophy\n", "", ""],
        "Cardiac pacing": ["", "", ""],
        "Ischemia": ["", "STEMI: anterior wall\nIschemia: lateral wall\n", ""],
        "Non-specific repolarization abnormalities": ["", "", "Anterior wall\n"],
        "Other states": ["", "", "Atrial fibrillation\n"],
        "Electric axis of the heart": [
            "Electric axis of the heart: normal.\n",
            "Electric axis of the heart: left axis deviation.\n",
            "",
        ],
    }).to_csv(tmp_path / "ludb.csv", index=False)
    return tmp_path


def _build_medalcare_xl(tmp_path: Path) -> Path:
    """The generated metadata CSV, the only source this loader reads."""
    pd.DataFrame({
        "record_id": ["S65_sinus_1", "S65_mi_LAD_1.0_2", "S66_lbbb_3"],
        "record_number": ["1", "2", "3"],
        "pathology": ["sinus", "mi", "lbbb"],
        "pathology_subclass": ["sinus", "mi_LAD_1.0", "lbbb"],
        "mi_subclass": [None, "LAD_1.0", None],
        "mi_occlusion_site": [None, "LAD", None],
        "mi_transmurality": [None, 1.0, None],
        "mi_region": [None, None, None],
        "model_id": ["S65", "S65", "S66"],
        "source_split": ["train", "validation", "test"],
        "fold": [1, 2, 3],
        "signal_path": ["a/1.csv", "a/2.csv", "b/3.csv"],
        "signal_path_raw": ["a/1_raw.csv", "a/2_raw.csv", "b/3_raw.csv"],
        "signal_path_noise": ["a/1_noise.csv", "a/2_noise.csv", "b/3_noise.csv"],
        "atrial_params_path": ["a/1_AtrialParameters.txt"] * 3,
        "ventricular_params_path": ["a/1_VentricularParameters.txt"] * 3,
    }).to_csv(tmp_path / "ecgbench_metadata.csv", index=False)
    return tmp_path


def _build_mhd_effect_ecg_mri(tmp_path: Path) -> Path:
    """Three 3-channel headers: a 3T feet-first run, a head-first run and the reference."""
    wfdb = pytest.importorskip("wfdb")

    def header(record, field, b0, position):
        return (
            f"{record} 3 1024 25000\n"
            f"{record}.dat 16 12000.5(1234)/mV 0 0 100 200 0 I\n"
            f"{record}.dat 16 12000.5(1234)/mV 0 0 100 200 0 II\n"
            f"{record}.dat 16 12000.5(1234)/mV 0 0 100 200 0 III\n"
            "#*Technical parameters of the MR scanner:\n"
            f"#--Magnetic field strength:{field}\n"
            "#--MR scanner:Siemens Magnetom Skyra\n"
            f"#--Orientation of the static magnetic field (B0):{b0}\n"
            "#--ECG recorder:Getemed CM 3000, 12-lead Holter ECG\n"
            "#--ADC resolution:12bit\n"
            "#--ADC input voltage range:+/-6mV\n"
            "#--ECG lead configuration:Diagnostic 12 lead ECG\n"
            "#--Sex:Male\n#--Age:27years\n#--Weight:75kg\n#--Height:190cm\n"
            f"#--Positon in the scanner:{position}\n"
            "#--Respiration:Spontaneous respiration\n"
        )

    headers = {
        "ECGMRI3T01Ff": header("ECGMRI3T01Ff", "3T", "Horizontal", "Feet first (Ff)"),
        "ECGMRI3T01Hf": header("ECGMRI3T01Hf", "3T", "Horizontal", "Feet first (Ff)"),
        "ECGMRI3T01Out": header(
            "ECGMRI3T01Out", "Outside the scanner", "Outside the scanner",
            "Outside the scanner",
        ),
    }
    (tmp_path / "RECORDS").write_text("\n".join(headers) + "\n", encoding="utf-8")
    for record, text in headers.items():
        (tmp_path / f"{record}.hea").write_text(text, encoding="utf-8")
        wfdb.wrann(
            record, "qrs", sample=np.arange(1, 25) * 1000, symbol=["N"] * 24,
            fs=1024, write_dir=str(tmp_path),
        )
    return tmp_path


def _build_ningbo_iva(tmp_path: Path) -> Path:
    pd.DataFrame(
        [
            (1000364, "PVC", "Right", "AC", "female"),
            (991591, "VT", "Left", "LCC", "male"),
            (991592, "PVC", "Right", None, "female"),
        ],
        columns=["HospitalID", "Type", "LeftRight", "Sublocation", "Gender"],
    ).to_csv(tmp_path / "Diagnosis.csv", index=False)
    return tmp_path


def _build_norwegian_athlete_ecg(tmp_path: Path) -> Path:
    """Three headers: a borderline SL12 read, a STEMI alert, and a lowercase finding."""
    headers = {
        "ath_001": (
            "#SL12: Sinus bradycardia with marked sinus arrhythmia, Right axis"
            " deviation, Borderline ECG\n"
            "#C: Sinus arrhythmia,  Normal ECG\n"
        ),
        "ath_002": (
            "#SL12: ***Critical test result: STEMI, Sinus rhythm, ST elevation, consider"
            " early repolarization, pericarditis, or injury, ** ** ACUTE MI/STEMI** **,"
            " Abnormal EKG\n"
            "#C: Normal sinus rhythm, Borderline ECG\n"
        ),
        "ath_005": (
            "#SL12: Sinus bradycardia, Otherwise normal ECG\n"
            "#C: Sinus bradycardia, normal sinus rhythm, First degree AV block, Normal ECG\n"
        ),
    }
    (tmp_path / "RECORDS").write_text("\n".join(headers) + "\n", encoding="utf-8")
    for name, comments in headers.items():
        (tmp_path / f"{name}.hea").write_text(
            f"{name} 12 500 5000\n{name}.dat 16 50000/mV 16 0 10251 49595 0 I\n{comments}",
            encoding="utf-8",
        )
    return tmp_path


def _build_nsrdb(tmp_path: Path) -> Path:
    wfdb = pytest.importorskip("wfdb")
    for name, comment in (("16265", "# 32 M"), ("16272", "# 20 F")):
        (tmp_path / f"{name}.hea").write_text(
            f"{name} 2 128 128000  8:04:00\n"
            f"{name}.dat 212 0 12 0 -33 15756 0 ECG1\n"
            f"{name}.dat 212 0 12 0 -65 -21174 0 ECG2\n"
            f"{comment}\n",
            encoding="utf-8",
        )
        wfdb.wrann(
            name, "atr",
            sample=np.array([100, 228, 356, 484, 612, 740, 900, 1028]),
            symbol=["|", "N", "V", "N", "S", "~", "N", "~"],
            subtype=np.array([0, 0, 0, 0, 0, 1, 0, 0]),
            fs=128, write_dir=str(tmp_path),
        )
    (tmp_path / "RECORDS").write_text("16265\n16272\n", encoding="utf-8")
    return tmp_path


def _build_picsdb(tmp_path: Path) -> Path:
    """Two infants' ECG records with .dat (the loader memmaps it) and their respiration."""
    wfdb = pytest.importorskip("wfdb")

    def record(name, d_signal, fs, lead, annotations=()):
        wfdb.wrsamp(
            name, fs=fs, units=["mV"], sig_name=[lead],
            d_signal=np.asarray(d_signal, dtype=np.int16).reshape(-1, 1),
            fmt=["16"], adc_gain=[800.0], baseline=[0], write_dir=str(tmp_path),
        )
        for extension, samples, symbol in annotations:
            wfdb.wrann(
                name, extension, sample=np.asarray(samples, dtype=np.int64),
                symbol=[symbol] * len(samples), fs=fs, write_dir=str(tmp_path),
            )

    names = []
    for infant, fs, lead in ((1, 250, "ECG"), (2, 500, "II")):
        ecg, resp = f"infant{infant}_ecg", f"infant{infant}_resp"
        signal = np.arange(20_000, dtype=np.int16) % 71
        signal[:600] = -32767  # a spell at the converter rail
        peaks = np.arange(200, 20_000, 200)
        record(ecg, signal, fs, lead, [("qrsc", peaks, "N"), ("atr", [4001, 12001], "[")])
        record(resp, np.arange(1000, dtype=np.int16), 50, "RESP",
               [("resp", np.arange(5) * 100 + 10, "N")])
        names += [ecg, resp]
    (tmp_path / "RECORDS").write_text("\n".join(names) + "\n", encoding="utf-8")
    return tmp_path


#: The 48 comment lines of a PTBDB header (47 keys; Catheterization date repeats),
#: transcribed from PhysioNet's patient001/s0010_re.hea. Not verified against a
#: local copy of the release - none is on this machine.
_PTBDB_COMMENT_KEYS = (
    "age", "sex", "ECG date", "Diagnose", "Reason for admission",
    "Acute infarction (localization)", "Former infarction (localization)",
    "Additional diagnoses", "Smoker", "Number of coronary vessels involved",
    "Infarction date (acute)", "Previous infarction (1) date", "Previous infarction (2) date",
    "Hemodynamics", "Catheterization date", "Ventriculography", "Chest X-ray",
    "Peripheral blood Pressure (syst/diast)",
    "Pulmonary artery pressure (at rest) (syst/diast)",
    "Pulmonary artery pressure (at rest) (mean)",
    "Pulmonary capillary wedge pressure (at rest)", "Cardiac output (at rest)",
    "Cardiac index (at rest)", "Stroke volume index (at rest)",
    "Pulmonary artery pressure (laod) (syst/diast)", "Pulmonary artery pressure (laod) (mean)",
    "Pulmonary capillary wedge pressure (load)", "Cardiac output (load)",
    "Cardiac index (load)", "Stroke volume index (load)", "Aorta (at rest) (syst/diast)",
    "Aorta (at rest) mean", "Left ventricular enddiastolic pressure",
    "Left coronary artery stenoses (RIVA)", "Left coronary artery stenoses (RCX)",
    "Right coronary artery stenoses (RCA)", "Echocardiography", "Therapy",
    "Infarction date", "Catheterization date", "Admission date", "Medication pre admission",
    "Start lysis therapy (hh.mm)", "Lytic agent", "Dosage (lytic agent)",
    "Additional medication", "In hospital medication", "Medication after discharge",
)


def _build_ptbdb(tmp_path: Path) -> Path:
    """Two patients, three records, with the full 48-line comment block each."""
    records = {
        ("patient001", "s0010_re"): {"age": "81", "sex": "female",
                                     "Reason for admission": "Myocardial infarction"},
        ("patient002", "s0015lre"): {"age": "58", "sex": "male",
                                     "Reason for admission": "Healthy control"},
        ("patient002", "s0016lre"): {"age": "58", "sex": "", "Reason for admission": "n/a"},
    }
    for (patient, name), values in records.items():
        d = tmp_path / patient
        d.mkdir(exist_ok=True)
        lines = [f"{name} 15 1000 38400", f"{name}.dat 16 2000 16 0 -489 -19 0 i"]
        lines += [f"# {key}: {values.get(key, 'n/a')}" for key in _PTBDB_COMMENT_KEYS]
        (d / f"{name}.hea").write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    return tmp_path


def _build_qtdb(tmp_path: Path) -> Path:
    """A MIT-BIH and a European ST-T excerpt, with manual q1c beats and one .atr."""
    wfdb = pytest.importorskip("wfdb")
    headers = {
        "sel100": (
            "sel100 2 250/360 225000\n"
            "sel100.dat 212 200(0) 11 1024 945 -13873 0 MLII\n"
            "sel100.dat 212 200(0) 11 1024 955 14507 0 V5\n"
            "# 69 M 1085 1629 x1\n"
            "# Aldomet, Inderal\n"
            "#Produced by xform from record 100, beginning at 7:00.000\n"
        ),
        "sele0104": (
            "sele0104 2 250 225000\n"
            "sele0104.dat 212 200 12 0 -244 2025 0 D3\n"
            "sele0104.dat 212 200 12 0 -297 -13904 0 D4\n"
            "#Age: 47  Sex: M\n#Coronary artery disease\n#Coronary angiography\n"
            "#Myocardial infarction\n#unspecified medication\n"
            "#Recorder type: ICR model 7200\n"
            "#Produced by xform from record e0104, beginning at 1:35:00.000\n"
        ),
    }
    (tmp_path / "RECORDS").write_text("\n".join(headers) + "\n", encoding="utf-8")
    samples, symbols, nums = [], [], []
    for beat_start in (150000, 150250):
        for offset, symbol, num in (
            (0, "(", 0), (10, "p", 0), (20, ")", 0),
            (40, "(", 1), (50, "N", 1), (62, ")", 1),
            (80, "(", 2), (100, "t", 2), (144, ")", 2),
            (160, "u", 3), (170, ")", 3),
        ):
            samples.append(beat_start + offset)
            symbols.append(symbol)
            nums.append(num)
    for record, text in headers.items():
        (tmp_path / f"{record}.hea").write_text(text, encoding="utf-8")
        # wrann refuses a digit in the extension; rdann reads one happily.
        wfdb.wrann(
            record, "qxc", sample=np.array(samples), symbol=symbols, num=np.array(nums),
            fs=250, write_dir=str(tmp_path),
        )
        (tmp_path / f"{record}.qxc").rename(tmp_path / f"{record}.q1c")
        wfdb.wrann(
            record, "pu", sample=np.array([150050, 150100, 150300, 150350]),
            symbol=["N", "t", "N", "t"], num=np.array([0, 0, 0, 1]),
            fs=250, write_dir=str(tmp_path),
        )
    wfdb.wrann(
        "sel100", "atr",
        sample=np.array([10, 300, 600, 900, 1200, 100000, 100300]),
        symbol=["+", "N", "V", "A", '"', "+", "N"],
        aux_note=["(N", "", "", "", "MISSB", "(AFIB", ""],
        fs=250, write_dir=str(tmp_path),
    )
    return tmp_path


def _build_sami_trop(tmp_path: Path) -> Path:
    """exams.csv with exactly N_RECORDS rows, which the positional join enforces."""
    from ecgbench.labels.sami_trop import N_RECORDS

    n = N_RECORDS
    rng = np.random.default_rng(0)
    pd.DataFrame({
        "exam_id": np.arange(1, n + 1),
        "age": rng.integers(26, 98, n),
        "is_male": rng.integers(0, 2, n).astype(bool),
        "normal_ecg": np.arange(n) % 6 == 0,
        "death": np.arange(n) % 16 == 0,
        "timey": rng.uniform(0.07, 3.39, n).round(3),
        "nn_predicted_age": rng.uniform(22.6, 95.9, n).round(1),
    }).to_csv(tmp_path / "exams.csv", index=False)
    return tmp_path


def _build_sddb(tmp_path: Path) -> Path:
    """Records 30 (both annotators, vfon) and 42 (detector only, no vfon)."""
    wfdb = pytest.importorskip("wfdb")
    (tmp_path / "30.hea").write_text(
        "30 2 250 22099250 12:00:00\n"
        "30.dat 212 800 12 0 51 -24065 0 ECG\n"
        "30.dat 212 800 12 0 145 21051 0 ECG\n"
        "#Produced by xform_new from record 30, beginning at 26:35.000\n"
        "#vfon: 07:54:33\n",
        encoding="utf-8",
    )
    (tmp_path / "42.hea").write_text(
        "42 2 250 22622500 12:00:00\n"
        "42.dat 212 800 12 0 -1129 -2818 0 ECG\n"
        "42.dat 212 800 12 0 552 -30989 0 ECG\n"
        "#Produced by xform from record 42, beginning at 18:10.000\n",
        encoding="utf-8",
    )
    for name in ("30", "42"):
        wfdb.wrann(
            name, "ari",
            sample=np.array([100, 250, 500, 750, 800, 900, 1000, 1200, 1250, 1500]),
            symbol=["?", "N", "r", "N", "+", "s", "N", "s", "N", "E"],
            subtype=np.zeros(10, dtype=int),
            aux_note=["", "", "", "", "(AFIB", "(ST0+", "", "ST0+)", "", ""],
            fs=250, write_dir=str(tmp_path),
        )
    wfdb.wrann(
        "30", "atr",
        sample=np.array([250, 500, 750, 1000, 1250, 1500, 1750]),
        symbol=["N", "B", "/", "~", "N", "~", "|"],
        subtype=np.array([0, 0, 0, 51, 0, 0, 0]),
        fs=250, write_dir=str(tmp_path),
    )
    (tmp_path / "RECORDS").write_text("30\n42\n", encoding="utf-8")
    return tmp_path


#: The 45 columns of SHDB-AF's AdditionalData.csv, from the v1.0.1 release header.
_SHDB_AF_CLINICAL_COLUMNS = (
    "Data_ID", "Subject_ID", "Annotated", "Height", "Weight", "BMI", "Date_Holter",
    "Indication_Holter", "Age_at_Holter", "Sex", "AF_Type", "Previously_Documented_AFL",
    "Previous_AF_Ablation", "PPM_on_Holter", "PPM_after_Holter", "PPM_Indication", "PPM_Date",
    "Date_of_First_Diagnosis_of_AF_AFL", "AF_Duration_Months", "Antiarrhythmic_Drug_nonBB",
    "Antiarrhythmic_Drug_BB", "Anticoagulation", "Date_1st_AF_Ablation", "Ablation1_PVI",
    "Ablation1_CTI", "Ablation1_Others", "Date_Redo_AF_Ablation", "Redo_Detail", "Echo_Date",
    "Echo_LAD", "Echo_LVEF", "Echo_LV_Asynergy", "Moderate_or_Severe_MR",
    "Moderate_or_Severe_TR", "Moderate_or_Severe_AS", "Moderate_or_Severe_AR", "CHF", "HTN",
    "Age_75_or_Older", "DM", "Stroke", "Vascular_Diseases", "Comments", "Holter_start_time",
    "Holter_recording_length",
)


def _build_shdb_af(tmp_path: Path) -> Path:
    """Two records, one with the comment-only rhythm annotations, plus the 45-column table."""
    wfdb = pytest.importorskip("wfdb")
    rows = [
        ["001", "2043771", "True", 1.73, 63.5, 21.2, "2021-03-13",
         "AF monitoring after ablation", 65, "M", "PAF", "False", "True", "False", "False",
         None, None, None, None, "frecainide", None, "warfarin", "2012-09-15", 1.0, 1.0, None,
         "2021-07-30", "SVCI, re-PVI", "2021-03-13", 40.0, 39.0, "anteroseptal", 0.0, 0.0, 0.0,
         0.0, 0.0, "False", "False", "False", "False", "False", None, "10:10 AM", "23:54:59"],
        ["002", "4980615", "False", 1.66, 55.9, 20.3, "2021-03-15",
         "AF monitoring after ablation", 62, "F", "non-AF", "True", "True", "False", "False",
         None, None, "2025-06-02", 33.0, None, None, "edoxaban", "2020-06-16", 1.0, 1.0, None,
         None, None, "2020-04-16", 40.0, 56.0, None, 0.0, 0.0, 0.0, 0.0, 1.0, "True", "False",
         "False", "True", "False", "noisy tail", "10:55 AM", "23:59:59"],
    ]
    pd.DataFrame(rows, columns=_SHDB_AF_CLINICAL_COLUMNS).to_csv(
        tmp_path / "AdditionalData.csv", index=False
    )
    beats = np.array([200 * (i + 1) for i in range(12)])
    for rec in ("001", "002"):
        (tmp_path / f"{rec}.hea").write_text(
            f"{rec} 2 200 20000\n"
            f"{rec}.dat 16 8105.233566939608(-9270)/mV 16 0 -9267 62791 0 ECG1\n"
            f"{rec}.dat 16 11470.645879660451(543)/mV 16 0 408 55919 0 ECG2\n",
            encoding="utf-8",
        )
        wfdb.wrann(rec, "qrs", sample=beats, symbol=["N"] * 12, fs=200, write_dir=str(tmp_path))
    wfdb.wrann(
        "001", "atr", sample=beats, symbol=['"'] * 12,
        aux_note=["(AFIB", "", "", "", "(N", "", "", "(AB", "", "", "(AFIB", ""],
        fs=200, write_dir=str(tmp_path),
    )
    (tmp_path / "RECORDS.txt").write_text("001\n002\n", encoding="utf-8")
    return tmp_path


def _build_sph(tmp_path: Path) -> Path:
    (tmp_path / "code.csv").write_text(
        "Category,Code,Description\n"
        "A,1,Normal ECG\n"
        "C,22,Sinus bradycardia\n"
        "F,60,Ventricular premature complex(es)\n"
        'D,31,"Atrial premature complexes, nonconducted"\n'
        "Modifier,310,Frequent\n",
        encoding="utf-8",
    )
    rows = [("A00001", "60+310;22", "S1", 44, "M", 5000), ("A00002", "1", "S2", 61, "F", 5000),
            ("A00003", "1;1", "S2", 61, "F", 28000)]
    lines = ["ECG_ID,AHA_Code,Patient_ID,Age,Sex,N,Date"]
    lines += [
        f"{rid},{code},{pid},{age},{sex},{n},2020-01-01" for rid, code, pid, age, sex, n in rows
    ]
    (tmp_path / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def _build_staffiii(tmp_path: Path) -> Path:
    """One patient's baseline and inflation records, with the sheet's 10-row preamble."""
    wfdb = pytest.importorskip("wfdb")
    pytest.importorskip("openpyxl")
    n_columns = 29
    row = [None] * n_columns
    row[0], row[1], row[2], row[28] = 1, 52, "f", "no"
    row[3], row[6], row[7] = "1a", "1c", "dist circ"
    frame = pd.DataFrame([[None] * n_columns for _ in range(10)] + [row])
    frame.to_excel(tmp_path / "STAFF-III-Database-Annotations.xlsx", header=False, index=False)
    data = tmp_path / "data"
    data.mkdir()
    for record in ("001a", "001c"):
        (data / f"{record}.hea").write_text(
            f"{record} 9 1000 300000 20:26:00 27/09/1995\n"
            f"{record}.dat 16+512 1600 12 0 0 0 0  V1\n"
            f"{record}.dat 16+512 1600 12 0 0 0 0  V2\n"
            "# Age: 52\n# Sex: F\n",
            encoding="utf-8",
        )
    wfdb.wrann(
        "001c", "event", sample=np.array([1000, 60000, 240000]),
        symbol=['"'] * 3,
        aux_note=["contrast injection", "balloon inflation", "balloon deflation"],
        fs=1000, write_dir=str(data),
    )
    return tmp_path


def _build_stdb(tmp_path: Path) -> Path:
    """A two-channel exercise record and a one-channel long-term excerpt."""
    wfdb = pytest.importorskip("wfdb")
    (tmp_path / "300.hea").write_text(
        "300 2 360 3600\n300.dat 212 296 12 0 40 0 0 ECG\n300.dat 212 300 12 0 -5 0 0 ECG\n",
        encoding="utf-8",
    )
    (tmp_path / "323.hea").write_text(
        "323 1 360 3600\n323.dat 212 295 12 0 -74 0 0 ECG\n", encoding="utf-8"
    )
    for rec in ("300", "323"):
        wfdb.wrann(
            rec, "atr",
            sample=np.array([100, 460, 820, 1180, 1540, 1900, 2260]),
            symbol=["N", "S", "N", "V", "N", "~", "|"],
            subtype=np.array([0, 0, 0, 0, 0, 1, 0]),
            fs=360, write_dir=str(tmp_path),
        )
    (tmp_path / "RECORDS").write_text("300\n323\n", encoding="utf-8")
    return tmp_path


def _build_svdb(tmp_path: Path) -> Path:
    wfdb = pytest.importorskip("wfdb")
    (tmp_path / "800.hea").write_text(
        "800 2 128 230400\n800.dat 212 200 10 0 -101 -25183 0 ECG1\n"
        "800.dat 212 200 10 0 123 10510 0 ECG2\n",
        encoding="utf-8",
    )
    (tmp_path / "820.hea").write_text(
        "820 2 128 230400\n820.dat 212 0 10 0 -14 -6899 0 ECG1\n"
        "820.dat 212 0 10 0 -7 -19211 0 ECG2\n",
        encoding="utf-8",
    )
    for rec in ("800", "820"):
        wfdb.wrann(
            rec, "atr",
            sample=np.array([100, 228, 356, 484, 612, 740, 868, 996]),
            symbol=["|", "N", "S", "N", "V", "~", "a", "+"],
            subtype=np.array([0, 0, 0, 0, 0, 1, 0, 0]),
            aux_note=["", "", "", "", "", "", "", "(N"],
            fs=128, write_dir=str(tmp_path),
        )
    (tmp_path / "RECORDS").write_text("800\n820\n", encoding="utf-8")
    return tmp_path


def _build_szdb(tmp_path: Path) -> Path:
    """Two records of one reconstructed subject: 50 learning marks, beats, ST and AF."""
    wfdb = pytest.importorskip("wfdb")
    for rec in ("sz02", "sz03"):
        (tmp_path / f"{rec}.hea").write_text(
            f"{rec} 1 200 200000\n{rec}.dat 16 25 12 0 26 -30691 0 ECG\n", encoding="utf-8"
        )
        samples = list(np.arange(50) * 100 + 100) + list(np.arange(100) * 200 + 10_000)
        symbols = ["?"] * 50 + ["N"] * 100
        notes = [""] * 150
        for sample, symbol, note in (
            (20_000, "s", "(ST0-"), (24_000, "s", "ST0-)"), (25_000, "V", ""),
            (26_000, "r", ""), (27_000, "S", ""), (28_000, "+", "(AFIB"),
            (29_000, "+", "(N"), (30_000, "Q", ""),
        ):
            samples.append(sample)
            symbols.append(symbol)
            notes.append(note)
        order = np.argsort(np.asarray(samples), kind="stable")
        wfdb.wrann(
            rec, "ari", sample=np.asarray(samples)[order], symbol=[symbols[i] for i in order],
            subtype=np.zeros(len(samples), dtype=int), aux_note=[notes[i] for i in order],
            fs=200, write_dir=str(tmp_path),
        )
    (tmp_path / "RECORDS").write_text("sz02\nsz03\n", encoding="utf-8")
    (tmp_path / "times.seize").write_text(
        "sz02 00:14:36 00:16:12\nsz02 00:10:00 00:10:25\n", encoding="utf-8"
    )
    return tmp_path


def _build_tollet(tmp_path: Path) -> Path:
    """Two subjects' sittings in OpenSignals text, with dead and live electrode channels."""
    header = (
        "# OpenSignals Text File Format. Version 1\n"
        '# {"": {"sampling rate": 1000, '
        '"resolution": [4, 1, 1, 1, 1, 10, 10, 10, 10, 6, 6], '
        '"label": ["A1", "A2", "A3", "A4", "A5", "A6"], '
        '"column": ["nSeq", "I1", "I2", "O1", "O2", "A1", "A2", "A3", "A4", '
        '"A5", "A6"]}}\n'
        "# EndOfHeader\n"
    )
    root = tmp_path / "tollet"
    (root / "ECG_EXP").mkdir(parents=True)
    (root / "ECG_REF").mkdir()
    sittings = [
        ("1", [(300, 700), (350, 650), 0, (1020, 1021)], 40, "Male", ""),
        ("1_1", [(200, 800), 0, 0, 0], 40, "Male", "Paroxysmal AF"),
        ("2", [0, 0, 0, 0], 27, "Female", ""),
    ]
    lines = ["﻿ID;Age;Weight ;Height;Gender;Observations field;;;"]
    n = 200
    for name, codes, age, sex, note in sittings:
        rows = []
        for i in range(n):
            values = [i, 0, 0, 0, 0]
            for code in codes:
                if isinstance(code, tuple):
                    values.append(code[0] + (i * (code[1] - code[0])) // (n - 1))
                else:
                    values.append(code)
            values += [0, 0]
            rows.append("\t".join(str(v) for v in values) + "\t\r")
        (root / "ECG_EXP" / f"{name}.txt").write_text(
            header + "\n".join(rows) + "\n", encoding="utf-8"
        )
        lines.append(f"{name};{age};70;170;{sex};{note};;;")
    (root / "DataSet.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "ECG_REF" / "2.XML").write_text("<x/>", encoding="utf-8")
    return root


def _build_ucddb(tmp_path: Path) -> Path:
    """Two nights, one of them the record whose Holter is another subject's copy."""
    columns = [
        "S/No", "Study Number", "Height (cm)", "Weight (kg)", "Gender", "PSG Start Time",
        "PSG AHI", "BMI", "Age", "Epworth Sleepiness Score", "Study Duration (hr)",
        "Sleep Efficiency (%)", "No of data blocks in EDF",
    ]
    rows = []
    for index, (rec, ahi, n_events) in enumerate((("ucddb002", 4.0, 2), ("ucddb028", 46.0, 12))):
        write_edf(
            tmp_path / f"{rec}_lifecard.edf",
            [(lead * 1000 + np.arange(1280)).astype(np.int16) for lead in range(3)],
            ["chan 1", "chan 2", "chan 3"], [128] * 3,
            physical_range=(0.0, 10.0), digital_range=(0.0, 4095.0),
        )
        write_edf(
            tmp_path / f"{rec}.rec",
            [np.arange(1280, dtype=np.int16), np.arange(80, dtype=np.int16)],
            ["ECG", "SpO2"], [128, 8], starttime="23.00.00",
        )
        (tmp_path / f"{rec}_stage.txt").write_text(
            "\n".join(["0"] * 20 + ["3"] * 100 + ["1"] * 20) + "\n", encoding="utf-8"
        )
        lines = [
            "                              Respiratory Event List",
            "        Respiratory Event           Desaturation   Snore Arousal     B/T",
            " Time       Type   PB/CS  Duration  Low    %Drop                 Rate  Change",
        ]
        for i in range(n_events):
            lines.append(
                f"23:{30 + i:02d}:00  HYP-O             16"
                "       89.9    4.1     +     -      64.7   -5.7 "
            )
        lines.append("02:00:00  PB EVENT  PB      14                       -     -              ")
        (tmp_path / f"{rec}_respevt.txt").write_text(
            "\r\n".join(lines) + "\r\n\x1a", encoding="utf-8"
        )
        rows.append(dict(zip(columns, [
            index + 1, rec.upper(), 175, 90.0, "M", "23:00:00", ahi, 29.4, 50, 10, 1.0, 83, 10,
        ])))
    pd.DataFrame(rows).to_csv(tmp_path / "SubjectDetails.csv", index=False)
    return tmp_path


def _build_wctecgdb(tmp_path: Path) -> Path:
    """Two segments of one patient plus one of another, headers in cp1252 as shipped."""
    nstemi = "Non ST\xa0segment\xa0elevation myocardial infarction (NSTEMI)"
    headers = {
        "patient001/seg01": ("46", "M", nstemi, "V2, V2-raw"),
        "patient001/seg02": ("46", "M", nstemi, None),
        "patient002/seg01": ("71", "F", "not reported", None),
    }
    (tmp_path / "RECORDS").write_text("\n".join(headers) + "\n", encoding="utf-8")
    for name, (age, sex, diagnosis, reconstruct) in headers.items():
        stem = name.partition("/")[2]
        text = (
            f"{stem} 37 800 8001\n"
            f"{stem}.dat 16 36213.4604(-6137)/mV 0 0 500 -11346 0 I-Raw\n"
            f"{stem}.dat 16 145039.7107(2528)/mV 0 0 -3436 -23891 0 WCT\n\n"
            f"#Age: {age}\n#Sex: {sex}\n#Diagnosis report: {diagnosis}\n"
        )
        if reconstruct:
            text += f"#Reconstruct Precordials: {reconstruct}\n"
        path = tmp_path / f"{name}.hea"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="cp1252")
    return tmp_path


def _build_zzu_pecg(tmp_path: Path) -> Path:
    (tmp_path / "ECGCode.csv").write_text(
        "Description,AHA(Category&Code),CHN(Category&Code)\n"
        "Sinus tachycardia,C21,C13\n"
        '"Atrial premature complexes, nonconducted",D31,D22\n'
        "Left ventricular high voltage,N/A,J106\n"
        "Atrial reciprocal beats,N/A,D23\n",
        encoding="utf-8",
    )
    (tmp_path / "DiseaseCode.csv").write_text(
        "Disease Type,Disease Category,ICD-10 Code,ICD-10 Description\n"
        "Myocarditis,Acute myocarditis,I40.9,Acute myocarditis\n"
        '"Congenital \nheart disease",Ventricular septal defect,Q21.0,VSD\n'
        "Kawasaki disease,Kawasaki disease,M30.3,Kawasaki\n"
        "Other diseases(OD),Other,See attribute dictionary file,Other\n",
        encoding="utf-8",
    )
    columns = (
        "Filename,ECG_ID,Patient_ID,Age,Gender,Acquisition_date,Sampling_point,"
        "Lead,AHA_code,CHN_code,ICD-10 code,pSQI,basSQI,bSQI"
    )
    base = {
        "Filename": "P00/P00001/P00001_E01", "ECG_ID": "P00001_E01", "Patient_ID": "P00001",
        "Age": "572d", "Gender": "'Female'", "Acquisition_date": "2017-11-22 10:46:08",
        "Sampling_point": 15000, "Lead": 12, "AHA_code": "'C21'", "CHN_code": "'C13'",
        "ICD-10 code": "'Q21.0'", "pSQI": "'I':0.288;'II':0.323",
        "basSQI": "'I':0.98;'II':0.99", "bSQI": "'I':1.000;'II':1.000",
    }
    rows = [
        {},
        {"Filename": "P00/P00002/P00002_E01", "ECG_ID": "P00002_E01", "Patient_ID": "P00002",
         "Age": "4015d", "Gender": "'Male'", "Lead": 9, "Sampling_point": 5000,
         "AHA_code": "'D31';'Left ventricular high voltage'", "CHN_code": "'D22';'J106'",
         "ICD-10 code": "", "pSQI": "'I':0.5;'V2':Null"},
    ]
    lines = [columns]
    for row in rows:
        r = {**base, **row}
        lines.append(",".join(f'"{r[c]}"' for c in columns.split(",")))
    (tmp_path / "AttributesDictionary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
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
    # batch 3
    "ecg_capable_smartwatches": _build_ecg_capable_smartwatches,
    "ecgcipa": _build_ecgcipa,
    "ecgdmmld": _build_ecgdmmld,
    "ecgiddb": _build_ecgiddb,
    "ecgrdvq": _build_ecgrdvq,
    "echonext": _build_echonext,
    "edb": _build_edb,
    "edgar": _build_edgar,
    # batch 4
    "ikem": _build_ikem,
    "incartdb": _build_incartdb,
    "leipzig_heart_center_ecg": _build_leipzig_heart_center_ecg,
    "ltafdb": _build_ltafdb,
    "ltstdb": _build_ltstdb,
    "ludb": _build_ludb,
    "medalcare_xl": _build_medalcare_xl,
    "mhd_effect_ecg_mri": _build_mhd_effect_ecg_mri,
    # batch 5
    "ningbo_iva": _build_ningbo_iva,
    "norwegian_athlete_ecg": _build_norwegian_athlete_ecg,
    "nsrdb": _build_nsrdb,
    "picsdb": _build_picsdb,
    "ptbdb": _build_ptbdb,
    "qtdb": _build_qtdb,
    "sami_trop": _build_sami_trop,
    "sddb": _build_sddb,
    # batch 6
    "shdb_af": _build_shdb_af,
    "sph": _build_sph,
    "staffiii": _build_staffiii,
    "stdb": _build_stdb,
    "svdb": _build_svdb,
    "szdb": _build_szdb,
    "tollet": _build_tollet,
    "ucddb": _build_ucddb,
    "wctecgdb": _build_wctecgdb,
    "zzu_pecg": _build_zzu_pecg,
}

#: Label-bearing datasets whose fields are not declared yet. Empty since Phase 3
#: batch 6: every label-bearing dataset is in BUILDERS. A new dataset lands here
#: until its FIELDS and builder exist (test_every_label_bearing_dataset_is_declared_or_pending).
PENDING: set[str] = set()


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

    def test_phase_3_is_complete(self):
        """Every label-bearing dataset declares at least one field."""
        assert PENDING == set()
        for slug in sorted(_label_bearing()):
            assert fields_for(load_config(slug), static=True), slug

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
        assert main(["fields", "mimic_iv_ecg_demo"]) == 0
        assert "no labels" in capsys.readouterr().out
        assert main(["fields", "mimic_iv_ecg_demo", "--format", "json"]) == 0
        assert json.loads(capsys.readouterr().out) == []
        # No shipped dataset is undeclared any more, so the branch is exercised on
        # a real record with its fields removed.
        undeclared = replace(open_store().get("ptbxl"), fields=())
        assert "not yet declared" in format_fields(undeclared)
        assert json.loads(format_fields(undeclared, "json")) == []

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
