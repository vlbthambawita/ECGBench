"""
ZZU-pECG labels: paediatric ECG findings, ICD-10 diagnoses and signal-quality indices.

Everything comes from ``AttributesDictionary.csv`` plus two small dictionaries,
``ECGCode.csv`` and ``DiseaseCode.csv``. None of the three can be used as it
ships — every interesting column is a packed string — which is why this is a
module rather than a declarative column select.

**The two code columns are parallel and complementary, and neither is
sufficient alone.** ``AHA_code`` and ``CHN_code`` hold the same findings in the
same order (26,797 entries each), one per ``;``-separated element. Each column
gives the code *where its own vocabulary has one, and the plain-English
description where it does not*: ``ECGCode.csv`` leaves ``AHA`` empty for 14 of
its 105 findings (Osborn wave, left ventricular high voltage, prolonged QTc, …)
and ``CHN`` empty for 29. So 6,473 of the ``AHA_code`` entries are prose rather
than a code, and reading the column as a code vocabulary silently invents 15
extra "codes". Both are normalised here, and ``ecg_findings`` gives the
canonical description for every finding regardless of which vocabulary names it.

**Modifiers are glued onto codes, differently in each vocabulary.** AHA writes
``L145+Modifier362`` and bare qualifiers like ``Suggests208``; CHN writes
``L121+Depression``, ``F55+Frequent``, ``D21+Occasional`` and composites like
``J(111+112+113)``. ``aha_base_codes`` and ``chn_base_codes`` strip the modifier
so a user can group on the finding, while the unstripped lists keep the detail.

**Age is in days, and the cohort is young enough that it matters.** The column
reads ``572d``. Ages run 1 day to 5,474 days (15.0 years) with a median of 3,150
(8.6 years), so rounding to whole years would collapse the entire neonatal and
infant range — 546 records are under one year. Both ``age_days`` and
``age_years`` are exposed; prefer ``age_days`` for anything paediatric.

**Records have either 12 or 9 leads, and the 9-lead layout is not a prefix of
the 12-lead one.** 1,856 of the 14,190 records drop V2, V4 and V6, so stored
position 7 is V2 in a 12-lead record and V3 in a 9-lead one. ``n_leads`` is
exposed for filtering, and the config's ``alternate_lead_names`` is what makes
``ECGDataset(leads=["V2"])`` refuse those records instead of returning V3.

**Signal quality ships per lead, and absent leads are ``Null``.** ``pSQI``,
``basSQI`` and ``bSQI`` are ``'I':0.288;'II':0.323;…`` strings with ``Null`` for
the leads a 9-lead record does not have. The means over present leads are
exposed as ``psqi_mean``/``bassqi_mean``/``bsqi_mean``; the raw strings are kept
verbatim so nothing is lost. These are the release's own quality measures and
they agree with ECGBench's amplitude check — records that hit the ~26.6 mV rail
have a measurably lower ``basSQI`` (median 0.961 against 0.983, r = -0.36).

**Disease groups come from ICD-10 via DiseaseCode.csv, and the count does not
match the paper.** That file maps 19 real ICD-10 codes to four groups plus a
placeholder row (``See attribute dictionary file`` → ``Other diseases(OD)``,
which is not a code and is ignored). 3,716 records carry at least one of the 19,
covering 2,597 patients, against the 3,516 "diagnosed with cardiovascular
diseases" the data descriptor states. The derivation used here is stated exactly
so the difference is checkable: split ``ICD-10 code`` on ``;``, strip quotes, and
count a record if any element is one of the 19 keys. ICD-10 codes carry study
prefixes — ``(FO) Q21.1``, ``(OSD) Q21.1``, ``(F) I40.0`` — which are part of the
key and are matched literally.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

__all__ = [
    "ATTRIBUTES_CSV",
    "DISEASE_CODE_CSV",
    "ECG_CODE_CSV",
    "N_RECORDS",
    "NO_DISEASE",
    "SAMPLING_RATE",
    "SIGNAL_SUBDIR",
    "load_labels",
    "parse_code_list",
    "parse_sqi",
]

#: One row per record: identifiers, demographics, codes and quality indices.
ATTRIBUTES_CSV = "AttributesDictionary.csv"

#: Finding vocabulary: Description <-> AHA code <-> CHN code.
ECG_CODE_CSV = "ECGCode.csv"

#: ICD-10 code -> disease type/category for the studied conditions.
DISEASE_CODE_CSV = "DiseaseCode.csv"

#: Waveforms live under this directory, and the CSV's Filename column omits it.
SIGNAL_SUBDIR = "Child_ecg"

#: Records in the published release. Checked, not assumed.
N_RECORDS = 14190

#: Constant across the release.
SAMPLING_RATE = 500

#: Stratification class for a record with none of the 19 target ICD-10 codes.
NO_DISEASE = "NONE"

#: Separator inside the packed columns.
PACK_SEPARATOR = ";"

#: Separator for the exported code lists. Safe because every normalised token is
#: comma-free — asserted in _join_codes rather than assumed.
LIST_SEPARATOR = ","

#: Descriptions do contain commas ("Atrial premature complexes, nonconducted"),
#: so the human-readable list keeps the source's own separator.
DESCRIPTION_SEPARATOR = ";"

#: Placeholder row in DiseaseCode.csv that is not an ICD-10 code.
_NOT_A_CODE = "See attribute dictionary file"

#: `L145+Modifier362` -> `L145`, `L121+Depression` -> `L121`. The modifier must
#: start with a LETTER: `J(111+112+113)` is a composite code whose internal `+`
#: joins digits, and splitting on it would yield the truncated base `J(111+112`.
_MODIFIER_RE = re.compile(r"^([A-Z]\(?[\d+]*\)?)\+([A-Za-z].*)$")

#: `'I':0.288` or `'V2':Null`
_SQI_RE = re.compile(r"'([^']+)'\s*:\s*(Null|[-+0-9.eE]+)")


def parse_code_list(value: object) -> list[str]:
    """Split one of the ``;``-packed, optionally quoted code columns.

    ``"'Left ventricular high voltage';'L147'"`` -> two elements. ``Null`` and
    empty elements are dropped; a non-string (NaN) gives an empty list.
    """
    if not isinstance(value, str):
        return []
    out = []
    for part in value.split(PACK_SEPARATOR):
        token = part.strip().strip("'").strip()
        if token and token != "Null":
            out.append(token)
    return out


def parse_sqi(value: object) -> dict[str, float]:
    """Parse ``'I':0.288;'II':0.323;'V2':Null`` into a lead -> value dict.

    ``Null`` marks a lead the record does not have and becomes NaN, so the mean
    over present leads is just ``nanmean``.
    """
    out: dict[str, float] = {}
    if not isinstance(value, str):
        return out
    for lead, raw in _SQI_RE.findall(value):
        out[lead] = np.nan if raw == "Null" else float(raw)
    return out


def _base_code(token: str) -> str:
    """``L145+Modifier362`` -> ``L145``; a description is returned unchanged."""
    match = _MODIFIER_RE.match(token)
    return match.group(1) if match else token


def _join_codes(lists: list[list[str]], what: str) -> np.ndarray:
    """Join each list with a comma, refusing if a token contains one.

    The config declares ``label_format: comma_separated``, so a token with an
    embedded comma would split into two bogus codes downstream. Every normalised
    token in the published release is comma-free; this is what keeps that true
    rather than hoping.
    """
    offenders = {t for tokens in lists for t in tokens if LIST_SEPARATOR in t}
    if offenders:
        raise ValueError(
            f"{what} contains {LIST_SEPARATOR!r} in {sorted(offenders)[:3]}, so the "
            "comma-separated label column would split it into separate codes. Change "
            "the separator in ecgbench/labels/zzu_pecg.py if a re-release does this."
        )
    return np.array([LIST_SEPARATOR.join(tokens) for tokens in lists], dtype=object)


def _require(path: Path, what: str, url: str) -> None:
    from ecgbench.labels import LabelSourceMissingError

    if not path.exists():
        raise LabelSourceMissingError(
            f"ZZU-pECG {what} comes from {path.name}, which is not in {path.parent}. "
            "ECGBench publishes fold CSVs only — labels stay with the source dataset, "
            f"so point data_path at a full local copy (see {url})."
        )


def _read_dictionaries(root: Path, url: str) -> tuple[dict, dict, dict, dict]:
    """Read ECGCode.csv and DiseaseCode.csv into lookup dicts.

    Returns:
        ``(description -> aha, description -> chn, code -> description,
        icd10 -> disease type)``. A vocabulary that does not name a finding maps
        it to ``None``, which is what makes the description fallback detectable.
    """
    ecg_path = root / ECG_CODE_CSV
    disease_path = root / DISEASE_CODE_CSV
    _require(ecg_path, "the finding vocabulary", url)
    _require(disease_path, "the disease dictionary", url)

    # 'N/A' is a real value here meaning "this vocabulary has no code for it",
    # and pandas reads it as NaN by default. Kept as None deliberately.
    ecg = pd.read_csv(ecg_path).dropna(subset=["Description"])
    desc_to_aha: dict[str, str | None] = {}
    desc_to_chn: dict[str, str | None] = {}
    code_to_desc: dict[str, str] = {}
    for description, aha, chn in zip(
        ecg["Description"], ecg["AHA(Category&Code)"], ecg["CHN(Category&Code)"]
    ):
        description = str(description).strip()
        aha = None if pd.isna(aha) or str(aha).strip() in ("", "N/A") else str(aha).strip()
        chn = None if pd.isna(chn) or str(chn).strip() in ("", "N/A") else str(chn).strip()
        desc_to_aha[description] = aha
        desc_to_chn[description] = chn
        for code in (aha, chn):
            if code:
                code_to_desc.setdefault(code, description)

    disease = pd.read_csv(disease_path)
    icd_to_type: dict[str, str] = {}
    for code, disease_type in zip(disease["ICD-10 Code"], disease["Disease Type"]):
        code = str(code).strip()
        if code == _NOT_A_CODE:
            # Not an ICD-10 code: a pointer to the attribute dictionary for the
            # "other diseases" bucket. Matching it would label nothing.
            continue
        # The Disease Type cells carry embedded newlines ("Congenital \nheart
        # disease"), which would end up in a CSV column verbatim.
        icd_to_type[code] = re.sub(r"\s+", " ", str(disease_type)).strip()

    return desc_to_aha, desc_to_chn, code_to_desc, icd_to_type


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return ZZU-pECG labels and metadata indexed by ``ECG_ID``.

    Columns:
        ``patient_id``, ``age_days``, ``age_years``, ``sex``,
        ``acquisition_date``, ``n_leads``, ``n_samples``, ``duration_seconds``,
        ``sampling_rate``, ``signal_path``; the findings as ``aha_codes``,
        ``aha_base_codes``, ``chn_codes``, ``chn_base_codes``, ``ecg_findings``
        and ``n_findings``; the diagnoses as ``icd10_codes``, ``n_icd10_codes``,
        ``disease_groups``, ``n_disease_groups`` and ``primary_disease_group``;
        and quality as ``psqi_mean``, ``bassqi_mean``, ``bsqi_mean`` plus the raw
        ``psqi_by_lead``, ``bassqi_by_lead``, ``bsqi_by_lead``.

    Multi-label on both axes: a record carries 1-8 ECG findings (median 2) and 0
    or more ICD-10 diagnoses. ``aha_codes`` is the ECG-finding target;
    ``disease_groups`` is the diagnosis target. Never train on
    ``primary_disease_group`` — it is a rarest-wins reduction that exists to make
    the folds well defined.

    Raises:
        LabelSourceMissingError: one of the three source CSVs is absent.
        ValueError: the attribute table has the wrong shape or duplicate IDs.
    """
    root = Path(data_path)
    url = config.url

    attributes_path = root / ATTRIBUTES_CSV
    _require(attributes_path, "per-record attributes", url)
    raw = pd.read_csv(attributes_path)

    expected = {
        "Filename",
        "ECG_ID",
        "Patient_ID",
        "Age",
        "Gender",
        "Acquisition_date",
        "Sampling_point",
        "Lead",
        "AHA_code",
        "CHN_code",
        "ICD-10 code",
        "pSQI",
        "basSQI",
        "bSQI",
    }
    missing = expected - set(raw.columns)
    if missing:
        raise ValueError(f"{attributes_path} is missing column(s) {sorted(missing)}")
    if raw["ECG_ID"].duplicated().any():
        raise ValueError(
            f"{attributes_path} has duplicate ECG_ID values; the join would multiply rows."
        )
    if len(raw) != N_RECORDS:
        logger.warning(
            "%s has %d rows; the published release has %d. Proceeding, but figures will "
            "not match the dataset page.",
            ATTRIBUTES_CSV,
            len(raw),
            N_RECORDS,
        )

    desc_to_aha, desc_to_chn, code_to_desc, icd_to_type = _read_dictionaries(root, url)

    index = pd.Index(raw["ECG_ID"].astype(str).to_numpy(), name="ECG_ID")
    df = pd.DataFrame({"patient_id": raw["Patient_ID"].astype(str).to_numpy()}, index=index)

    # "572d" -> 572. Days, not years: 546 records are under a year old.
    age_days = pd.to_numeric(
        raw["Age"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce"
    )
    df["age_days"] = age_days.to_numpy()
    df["age_years"] = (age_days / 365.25).to_numpy()

    # "'Female'" -> "F". Quoted in the source.
    gender = raw["Gender"].astype(str).str.strip().str.strip("'").str.upper()
    df["sex"] = pd.array(
        np.select([gender.eq("MALE"), gender.eq("FEMALE")], ["M", "F"], default=None),
        dtype="string",
    )
    df["acquisition_date"] = pd.to_datetime(raw["Acquisition_date"], errors="coerce").to_numpy()

    df["n_leads"] = raw["Lead"].astype(int).to_numpy()
    df["n_samples"] = raw["Sampling_point"].astype(int).to_numpy()
    df["duration_seconds"] = df["n_samples"] / SAMPLING_RATE
    df["sampling_rate"] = SAMPLING_RATE
    # The CSV's Filename is relative to Child_ecg/ and carries no extension.
    df["signal_path"] = (
        SIGNAL_SUBDIR + "/" + raw["Filename"].astype(str).str.strip().str.lstrip("/")
    ).to_numpy()

    # --- ECG findings -------------------------------------------------------
    # Each column names the finding by code where its vocabulary has one and by
    # description where it does not, so normalisation is per element and needs
    # both dictionaries.
    aha_lists: list[list[str]] = []
    chn_lists: list[list[str]] = []
    description_lists: list[list[str]] = []
    for aha_raw, chn_raw in zip(raw["AHA_code"], raw["CHN_code"]):
        aha_tokens = parse_code_list(aha_raw)
        chn_tokens = parse_code_list(chn_raw)
        aha_norm, chn_norm, descriptions = [], [], []
        for position, token in enumerate(aha_tokens):
            # A token that is a known description means the AHA vocabulary has
            # no code for this finding; keep the description as the identifier.
            aha_norm.append(desc_to_aha.get(token) or token)
            descriptions.append(
                token if token in desc_to_aha else code_to_desc.get(_base_code(token), token)
            )
            del position
        for token in chn_tokens:
            chn_norm.append(desc_to_chn.get(token) or token)
        aha_lists.append(aha_norm)
        chn_lists.append(chn_norm)
        description_lists.append(descriptions)

    df["aha_codes"] = _join_codes(aha_lists, "AHA_code")
    df["aha_base_codes"] = _join_codes([[_base_code(t) for t in x] for x in aha_lists], "AHA_code")
    df["chn_codes"] = _join_codes(chn_lists, "CHN_code")
    df["chn_base_codes"] = _join_codes([[_base_code(t) for t in x] for x in chn_lists], "CHN_code")
    # Descriptions contain commas, so they keep the source's own separator.
    df["ecg_findings"] = np.array(
        [DESCRIPTION_SEPARATOR.join(x) for x in description_lists], dtype=object
    )
    df["n_findings"] = np.array([len(x) for x in aha_lists], dtype=int)

    # --- ICD-10 diagnoses and disease groups -------------------------------
    icd_lists = [parse_code_list(v) for v in raw["ICD-10 code"]]
    df["icd10_codes"] = _join_codes(icd_lists, "ICD-10 code")
    df["n_icd10_codes"] = np.array([len(x) for x in icd_lists], dtype=int)

    group_lists = [
        sorted({icd_to_type[c] for c in codes if c in icd_to_type}) for codes in icd_lists
    ]
    df["disease_groups"] = _join_codes(group_lists, "disease groups")
    df["n_disease_groups"] = np.array([len(x) for x in group_lists], dtype=int)

    # Single-label reduction for stratification ONLY. Rarest-wins, computed from
    # the data rather than hardcoded, so the four groups all survive a ten-way
    # split — Cardiomyopathy has only 147 records and Kawasaki 194.
    counts: dict[str, int] = {}
    for groups in group_lists:
        for group in groups:
            counts[group] = counts.get(group, 0) + 1
    rarest_first = sorted(counts, key=lambda g: (counts[g], g))
    df["primary_disease_group"] = np.array(
        [next((g for g in rarest_first if g in groups), NO_DISEASE) for groups in group_lists],
        dtype=object,
    )

    # --- Signal quality ----------------------------------------------------
    for column, prefix in (("pSQI", "psqi"), ("basSQI", "bassqi"), ("bSQI", "bsqi")):
        parsed = [parse_sqi(v) for v in raw[column]]
        df[f"{prefix}_mean"] = np.array(
            [
                (
                    np.nanmean(list(d.values()))
                    if d and not all(np.isnan(list(d.values())))
                    else np.nan
                )
                for d in parsed
            ],
            dtype=float,
        )
        # Kept verbatim: the per-lead detail is the point of these columns.
        df[f"{prefix}_by_lead"] = raw[column].to_numpy()

    logger.info(
        "Loaded ZZU-pECG labels: %d records, %d patients, %d with a target ICD-10 "
        "diagnosis, %d nine-lead; median age %.1f years, median %d findings",
        len(df),
        df["patient_id"].nunique(),
        int((df["n_disease_groups"] > 0).sum()),
        int((df["n_leads"] == 9).sum()),
        float(df["age_years"].median()),
        int(df["n_findings"].median()),
    )
    return df
