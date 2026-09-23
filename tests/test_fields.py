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
}

#: Label-bearing datasets whose fields are not declared yet (later Phase 3 batches).
PENDING = {
    "ningbo_iva", "norwegian_athlete_ecg", "nsrdb", "picsdb", "ptbdb", "qtdb", "sami_trop",
    "sddb", "shdb_af", "sph", "staffiii", "stdb", "svdb", "szdb", "tollet", "ucddb",
    "wctecgdb", "zzu_pecg",
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
