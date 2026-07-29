"""
PhysioNet ``ecg-arrhythmia`` (Chapman-Shaoxing + Ningbo) splitting strategy.

This dataset ships **no metadata CSV** — every record's demographics and
diagnoses live in the per-record WFDB header (``#Age``, ``#Sex``, ``#Dx``).
``load_metadata`` therefore builds one by scanning all ``WFDBRecords/**/*.hea``
files and caches it next to the data as ``config.metadata_csv``.

Writing the cache to disk is load-bearing, not a convenience:
``validate_dataset`` re-reads ``data_path / config.metadata_csv`` itself rather
than reusing the splitter's DataFrame, so an in-memory-only frame would leave
validation with no metadata at all.

Stratification uses the *first* ``#Dx`` SNOMED code, which is this dataset's
rhythm diagnosis, mapped to its acronym via ``ConditionNames_SNOMED-CT.csv``.
Classes with fewer than ``MIN_CLASS_SIZE`` records are pooled into ``OTHER`` so
10-fold stratification stays well defined.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import DatasetSplitter
from ecgbench.splitting.registry import register

logger = logging.getLogger(__name__)

#: Directory (relative to the dataset root) holding the WFDB record tree.
RECORDS_DIR = "WFDBRecords"

#: SNOMED-CT code -> acronym mapping shipped with the dataset.
SNOMED_MAP_CSV = "ConditionNames_SNOMED-CT.csv"

#: Stratification classes smaller than this are pooled into "OTHER".
#: Matches the default ``n_folds=10`` so every class can appear in every fold.
MIN_CLASS_SIZE = 10

_UNLABELLED = "UNLABELLED"
_OTHER = "OTHER"


def _maybe_int(token: str) -> int | None:
    """Parse an integer field, tolerating the malformed headers in this dataset.

    At least one record (JS01052) has its record line and first signal-spec line
    merged, so positional fields do not parse. Those records are still listed —
    the validation engine flags them via ``corrupt_header``.
    """
    try:
        return int(float(token))
    except ValueError:
        return None


def _parse_header(hea_path: Path) -> dict[str, object]:
    """Parse the fields ECGBench needs out of one WFDB header file.

    Headers here are a few hundred bytes, so a plain text parse is far cheaper
    than a ``wfdb.rdheader`` call per record.
    """
    with open(hea_path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    record: dict[str, object] = {
        "record_name": hea_path.stem,
        "n_leads": None,
        "sampling_rate": None,
        "n_samples": None,
        "age": None,
        "sex": None,
        "dx": "",
    }

    if lines:
        # Record line: <name> <n_sig> <fs> <n_samples>
        parts = lines[0].split()
        for key, index in (("n_leads", 1), ("sampling_rate", 2), ("n_samples", 3)):
            if len(parts) > index:
                record[key] = _maybe_int(parts[index])

    for line in lines:
        if not line.startswith("#"):
            continue
        key, _, value = line[1:].partition(":")
        key = key.strip().lower()
        value = value.strip()
        if value.lower() in ("unknown", "nan", ""):
            value = ""
        if key == "age":
            record["age"] = value
        elif key == "sex":
            record["sex"] = value
        elif key == "dx":
            record["dx"] = value.replace(" ", "")

    return record


def _load_snomed_map(data_path: Path) -> dict[str, str]:
    """Return {snomed_code: acronym} from the dataset's condition-name CSV.

    The shipped file has a UTF-8 BOM and contains one duplicated code
    (``698252002`` for both IDC and IVB) — first occurrence wins.
    """
    csv_path = data_path / SNOMED_MAP_CSV
    if not csv_path.exists():
        logger.warning("SNOMED mapping not found at %s — using raw codes", csv_path)
        return {}

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    code_col = next(c for c in df.columns if "snomed" in c.lower())
    acronym_col = next(c for c in df.columns if "acronym" in c.lower())

    mapping: dict[str, str] = {}
    for code, acronym in zip(df[code_col], df[acronym_col], strict=True):
        mapping.setdefault(str(code).strip(), str(acronym).strip())
    logger.info("Loaded %d SNOMED-CT code mappings", len(mapping))
    return mapping


def build_metadata(data_path: Path, config: DatasetConfig) -> pd.DataFrame:
    """Scan every WFDB header under ``WFDBRecords/`` into a metadata frame.

    Signal paths are stored relative to ``data_path`` so they resolve the same
    way for the splitter, the validation engine and ``ECGDataset``.
    """
    records_root = data_path / RECORDS_DIR
    if not records_root.is_dir():
        raise FileNotFoundError(
            f"Expected the WFDB record tree at {records_root}. Point --data-path at "
            "the dataset root (the directory holding WFDBRecords/ and RECORDS)."
        )

    hea_files = sorted(records_root.rglob("*.hea"))
    if not hea_files:
        raise FileNotFoundError(f"No .hea header files found under {records_root}")
    logger.info("Parsing %d WFDB headers under %s", len(hea_files), records_root)

    # Header reads are I/O bound and independent — threads are enough.
    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(_parse_header, hea_files))

    signal_col = config.signal_path_columns[config.default_sampling_rate]
    for row, hea_path in zip(rows, hea_files, strict=True):
        signal = hea_path.with_suffix(".mat")
        row[signal_col] = str(signal.relative_to(data_path))

    df = pd.DataFrame(rows)

    snomed_map = _load_snomed_map(data_path)
    df["primary_dx"] = df["dx"].apply(lambda dx: str(dx).split(",")[0] if dx else "")
    df["primary_dx_acronym"] = df["primary_dx"].apply(
        lambda code: snomed_map.get(code, code or _UNLABELLED)
    )
    df["dx_acronyms"] = df["dx"].apply(
        lambda dx: ",".join(
            snomed_map.get(code, code) for code in str(dx).split(",") if code
        )
    )

    df = df.sort_values("record_name").reset_index(drop=True)
    logger.info("Built metadata for %d records", len(df))
    return df


@register("ecg_arrhythmia")
class ECGArrhythmiaSplitter(DatasetSplitter):
    """PhysioNet ecg-arrhythmia (Chapman-Shaoxing + Ningbo) splitting strategy.

    - Builds (and caches) the metadata CSV from the per-record WFDB headers
    - Stratifies on the primary ``#Dx`` SNOMED code's acronym
    - No patient grouping: the dataset is one record per patient
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path, sep=config.metadata_csv_separator, dtype={"dx": str}
            )

        df = build_metadata(data_path, config)
        try:
            df.to_csv(csv_path, sep=config.metadata_csv_separator, index=False)
            logger.info("Wrote metadata CSV: %s", csv_path)
        except OSError as e:
            # validate_dataset re-reads this file, so a read-only data directory
            # means validation cannot see any metadata. Fail loudly instead.
            raise OSError(
                f"Could not write the generated metadata CSV to {csv_path}: {e}. "
                "The dataset root must be writable, because the validation engine "
                "reads the metadata CSV from disk."
            ) from e
        return df

    def get_stratification_labels(
        self, df: pd.DataFrame, config: DatasetConfig
    ) -> pd.Series:
        """Stratify on the primary diagnosis acronym, pooling rare classes."""
        labels = df["primary_dx_acronym"].fillna(_UNLABELLED).replace("", _UNLABELLED)

        counts = labels.value_counts()
        rare = set(counts[counts < MIN_CLASS_SIZE].index)
        if rare:
            logger.info(
                "Pooling %d classes with <%d records into '%s': %s",
                len(rare), MIN_CLASS_SIZE, _OTHER, sorted(rare),
            )
            labels = labels.where(~labels.isin(rare), _OTHER)

        labels = labels.rename("primary_diagnosis")
        logger.info("Primary diagnosis distribution:\n%s", labels.value_counts().to_string())
        return labels
