"""
Catalogue of publicly available ECG datasets.

Provides functions to list, search, and filter the curated collection
of 64 ECG datasets bundled with ECGBench.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Any

_CATALOGUE: Optional[List[Dict[str, Any]]] = None
_CSV_PATH = Path(__file__).parent / "data" / "ecg_datasets.csv"


def _load() -> List[Dict[str, Any]]:
    """Load and cache the catalogue from the bundled CSV."""
    global _CATALOGUE
    if _CATALOGUE is not None:
        return _CATALOGUE

    rows = []
    with open(_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    _CATALOGUE = rows
    return _CATALOGUE


def list_datasets() -> List[Dict[str, Any]]:
    """Return all datasets in the catalogue.

    Returns:
        List of dicts, one per dataset. Keys: Category, Dataset Name,
        URL, Format, Patients, Records, Access, Origin, Paper, Paper DOI.
    """
    return [dict(r) for r in _load()]


def get_dataset(name: str) -> Optional[Dict[str, Any]]:
    """Look up a single dataset by exact name (case-insensitive).

    Args:
        name: Dataset name to look up.

    Returns:
        Dataset dict if found, None otherwise.
    """
    name_lower = name.lower()
    for row in _load():
        if row["Dataset Name"].lower() == name_lower:
            return dict(row)
    return None


def search(
    query: Optional[str] = None,
    category: Optional[str] = None,
    access: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search and filter datasets.

    All filters are combined with AND logic. Each is case-insensitive
    substring match.

    Args:
        query: Free-text search across name, origin, format, and paper.
        category: Filter by category (e.g. '12-Lead (PhysioNet)', '2-Lead').
        access: Filter by access type (e.g. 'Open', 'Credentialed', 'Restricted').

    Returns:
        List of matching dataset dicts.
    """
    results = _load()

    if category is not None:
        cat_lower = category.lower()
        results = [r for r in results if cat_lower in r["Category"].lower()]

    if access is not None:
        acc_lower = access.lower()
        results = [r for r in results if acc_lower in r["Access"].lower()]

    if query is not None:
        q = query.lower()
        results = [
            r
            for r in results
            if q in r["Dataset Name"].lower()
            or q in r.get("Origin", "").lower()
            or q in r.get("Format", "").lower()
            or q in r.get("Paper", "").lower()
        ]

    return [dict(r) for r in results]


def categories() -> List[str]:
    """Return the unique category names in catalogue order.

    Returns:
        List of category strings.
    """
    seen = set()
    result = []
    for row in _load():
        cat = row["Category"]
        if cat not in seen:
            seen.add(cat)
            result.append(cat)
    return result


def to_dataframe():
    """Return the full catalogue as a pandas DataFrame.

    Returns:
        pandas.DataFrame with all datasets.

    Raises:
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for to_dataframe(). "
            "Install it with: pip install ecgbench[pandas]"
        )
    return pd.read_csv(_CSV_PATH)
