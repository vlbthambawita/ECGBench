"""
Catalogue of publicly available ECG datasets.

Provides functions to list, search, and filter the curated collection
of 64 ECG datasets bundled with ECGBench. No heavy dependencies — always importable.
"""

import csv
import functools
from dataclasses import dataclass
from pathlib import Path

_CSV_PATH = Path(__file__).parent / "data" / "ecg_datasets.csv"


@dataclass(frozen=True)
class CatalogueEntry:
    """A single dataset in the ECGBench catalogue."""

    category: str
    dataset_name: str
    url: str
    format: str
    patients: str
    records: str
    access: str
    origin: str
    paper: str
    paper_doi: str


@functools.cache
def _load() -> tuple[CatalogueEntry, ...]:
    """Load and cache the catalogue from the bundled CSV."""
    entries = []
    with open(_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(
                CatalogueEntry(
                    category=row.get("Category", ""),
                    dataset_name=row.get("Dataset Name", ""),
                    url=row.get("URL", ""),
                    format=row.get("Format", ""),
                    patients=row.get("Patients", ""),
                    records=row.get("Records", ""),
                    access=row.get("Access", ""),
                    origin=row.get("Origin", ""),
                    paper=row.get("Paper", ""),
                    paper_doi=row.get("Paper DOI", ""),
                )
            )
    return tuple(entries)


def list_datasets() -> list[CatalogueEntry]:
    """Return all datasets in the catalogue.

    Returns:
        List of CatalogueEntry instances, one per dataset.
    """
    return list(_load())


def get_dataset(name: str) -> CatalogueEntry | None:
    """Look up a single dataset by exact name (case-insensitive).

    Args:
        name: Dataset name to look up.

    Returns:
        CatalogueEntry if found, None otherwise.
    """
    name_lower = name.lower()
    for entry in _load():
        if entry.dataset_name.lower() == name_lower:
            return entry
    return None


def search(
    query: str | None = None,
    category: str | None = None,
    access: str | None = None,
) -> list[CatalogueEntry]:
    """Search and filter datasets.

    All filters are combined with AND logic. Each is case-insensitive
    substring match.

    Args:
        query: Free-text search across name, origin, format, and paper.
        category: Filter by category (e.g. '12-Lead (PhysioNet)', '2-Lead').
        access: Filter by access type (e.g. 'Open', 'Credentialed', 'Restricted').

    Returns:
        List of matching CatalogueEntry instances.
    """
    results: list[CatalogueEntry] = list(_load())

    if category is not None:
        cat_lower = category.lower()
        results = [r for r in results if cat_lower in r.category.lower()]

    if access is not None:
        acc_lower = access.lower()
        results = [r for r in results if acc_lower in r.access.lower()]

    if query is not None:
        q = query.lower()
        results = [
            r
            for r in results
            if q in r.dataset_name.lower()
            or q in r.origin.lower()
            or q in r.format.lower()
            or q in r.paper.lower()
        ]

    return results


def categories() -> list[str]:
    """Return the unique category names in catalogue order.

    Returns:
        List of category strings.
    """
    seen: set[str] = set()
    result: list[str] = []
    for entry in _load():
        if entry.category not in seen:
            seen.add(entry.category)
            result.append(entry.category)
    return result


def get_download_url(dataset_name: str) -> str | None:
    """Look up the URL for a dataset from the catalogue.

    Args:
        dataset_name: Dataset name (case-insensitive).

    Returns:
        URL string if found, None otherwise.
    """
    entry = get_dataset(dataset_name)
    return entry.url if entry else None


def get_config(dataset_name: str):
    """Try to find a matching YAML config for a catalogue dataset.

    Fuzzy-matches the catalogue name to available config slugs by
    normalising to lowercase and removing hyphens/spaces.

    Args:
        dataset_name: Dataset name from the catalogue.

    Returns:
        DatasetConfig if a matching config exists, None otherwise.
    """
    from ecgbench.config import list_available_configs, load_config

    def _normalise(s: str) -> str:
        return s.lower().replace("-", "").replace(" ", "").replace("_", "")

    target = _normalise(dataset_name)
    for slug in list_available_configs():
        if _normalise(slug) == target:
            return load_config(slug)
    return None


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
            "Install it with: pip install ecgbench[all]"
        )
    return pd.read_csv(_CSV_PATH)
