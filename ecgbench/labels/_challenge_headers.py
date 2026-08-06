"""Shared WFDB-header scanning for the PhysioNet/CinC challenge releases.

The 2020 and 2021 challenges ship the *same* file layout — a ``training/``
directory of source-cohort subdirectories, one ``.hea``/``.mat`` pair per record,
and a short comment block in each header carrying ``#Age``, ``#Sex``, ``#Dx`` and
three fields that are the literal string ``Unknown`` on every record (``#Rx``,
``#Hx``, ``#Sx``). Neither release ships a metadata file of any kind, so both
label loaders have to build one from the headers.

This module holds the part that is genuinely identical between them. What
differs — the code table, the scored subset, and the per-release data quirks —
stays in ``challenge2020.py`` and ``challenge2021.py``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

#: Directory (relative to the dataset root) holding the source cohorts.
RECORDS_DIR = "training"


def parse_header(hea_path: Path) -> dict[str, object]:
    """Parse the fields ECGBench needs out of one WFDB header.

    Headers here are under a kilobyte, so a text parse is far cheaper than a
    ``wfdb.rdheader`` call per record across tens of thousands of records.
    """
    with open(hea_path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    record: dict[str, object] = {
        "record_name": hea_path.stem,
        "n_leads": None,
        "sampling_rate": None,
        "n_samples": None,
        "age": "",
        "sex": "",
        "dx": "",
    }

    if lines:
        # Record line: <name> <n_sig> <fs> <n_samples>
        parts = lines[0].split()
        for key, index in (("n_leads", 1), ("sampling_rate", 2), ("n_samples", 3)):
            if len(parts) > index:
                try:
                    record[key] = int(float(parts[index]))
                except ValueError:
                    pass  # corrupt header — validation flags it via corrupt_header

    for line in lines:
        if not line.startswith("#"):
            continue
        key, _, value = line[1:].partition(":")
        key = key.strip().lower()
        value = value.strip()
        if value.lower() in ("unknown", "nan", "n/a"):
            value = ""
        if key in ("age", "sex", "dx"):
            record[key] = value.replace(" ", "") if key == "dx" else value

    return record


def scan_headers(data_path: Path | str, release: str) -> pd.DataFrame:
    """Parse every record header under ``training/`` into one frame.

    Adds ``source`` (the cohort directory) and ``signal_path`` (relative to
    ``data_path``, so it resolves identically for the splitter, the validation
    engine and ``ECGDataset``).

    ``release`` is only used to make the "point data_path at the dataset root"
    error name the right challenge year.
    """
    from ecgbench.labels import LabelSourceMissingError

    data_path = Path(data_path)
    records_root = data_path / RECORDS_DIR
    if not records_root.is_dir():
        raise LabelSourceMissingError(
            f"Expected the record tree at {records_root}. {release} labels live "
            "in the WFDB headers, so point data_path at the dataset root — the "
            "directory holding training/ and RECORDS."
        )

    hea_files = sorted(records_root.rglob("*.hea"))
    if not hea_files:
        raise LabelSourceMissingError(f"No .hea headers found under {records_root}")
    logger.info("Parsing %d WFDB headers under %s", len(hea_files), records_root)

    # Header reads are I/O bound and independent — threads are enough.
    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(parse_header, hea_files))

    for row, hea in zip(rows, hea_files, strict=True):
        row["source"] = hea.relative_to(records_root).parts[0]
        row["signal_path"] = str(hea.with_suffix(".mat").relative_to(data_path))

    df = pd.DataFrame(rows).sort_values("record_name").reset_index(drop=True)
    logger.info("Parsed %d headers from %d source cohorts", len(df), df["source"].nunique())
    return df
