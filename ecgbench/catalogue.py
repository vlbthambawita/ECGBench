"""Catalogue of publicly available ECG datasets.

Provides functions to list, search, and filter the curated collection of
ECG datasets bundled with ECGBench. Source of truth: one Markdown file per
dataset at ``docs/_datasets/<slug>.md`` (front matter holds the row fields).

No heavy dependencies — always importable.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*", re.DOTALL)

_CATEGORY_ORDER = [
    "12-lead-physionet",
    "12-lead-other",
    "two-lead",
    "one-lead",
    "three-lead",
    "bspm",
]


def _datasets_dir() -> Path:
    """Locate the directory holding dataset front-matter files.

    Installed wheels get the files at ``ecgbench/_datasets/`` via hatch's
    ``force-include``. Editable installs and source checkouts use
    ``docs/_datasets/`` relative to the repo root.
    """
    wheel_dir = Path(__file__).parent / "_datasets"
    if wheel_dir.is_dir():
        return wheel_dir
    repo_dir = Path(__file__).resolve().parent.parent / "docs" / "_datasets"
    if repo_dir.is_dir():
        return repo_dir
    raise RuntimeError(
        "ECGBench dataset definitions not found. Expected "
        f"{wheel_dir} or {repo_dir}."
    )


@dataclass(frozen=True)
class CatalogueEntry:
    """A single dataset in the ECGBench catalogue.

    Fields mirror the YAML front matter in ``docs/_datasets/<slug>.md``.
    """

    slug: str
    name: str
    category: str
    status: str
    url: str
    url_label: str | None
    format: str
    patients: str
    records: str
    access: str
    license: str | None
    origin_institution: str
    origin_country: str | None
    leads: int | str | None
    paper_title: str | None
    paper_doi: str | None
    order: int = 0
    search_keywords: str = ""
    raw: dict = field(default_factory=dict, compare=False, repr=False)


def _parse_front_matter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER.match(text)
    if not match:
        raise ValueError(f"No YAML front matter in {path}")
    data = yaml.safe_load(match.group(1))
    return data or {}


def _entry_from_meta(slug: str, meta: dict) -> CatalogueEntry:
    return CatalogueEntry(
        slug=meta.get("slug", slug),
        name=meta.get("name", ""),
        category=meta.get("category", ""),
        status=meta.get("status", "not_started"),
        url=meta.get("url", ""),
        url_label=meta.get("url_label"),
        format=meta.get("format", ""),
        patients=str(meta.get("patients", "")),
        records=str(meta.get("records", "")),
        access=meta.get("access", ""),
        license=meta.get("license"),
        origin_institution=meta.get("origin_institution", ""),
        origin_country=meta.get("origin_country"),
        leads=meta.get("leads"),
        paper_title=meta.get("paper_title"),
        paper_doi=meta.get("paper_doi"),
        order=int(meta.get("order", 0)),
        search_keywords=meta.get("search_keywords", ""),
        raw=meta,
    )


@functools.cache
def _load() -> tuple[CatalogueEntry, ...]:
    """Load and cache every dataset entry from the bundled .md files."""
    entries: list[CatalogueEntry] = []
    for path in sorted(_datasets_dir().glob("*.md")):
        meta = _parse_front_matter(path)
        entries.append(_entry_from_meta(path.stem, meta))

    def _sort_key(e: CatalogueEntry) -> tuple:
        try:
            cat_rank = _CATEGORY_ORDER.index(e.category)
        except ValueError:
            cat_rank = len(_CATEGORY_ORDER)
        return (cat_rank, e.order, e.name)

    entries.sort(key=_sort_key)
    return tuple(entries)


def list_datasets() -> list[CatalogueEntry]:
    """Return all datasets in the catalogue."""
    return list(_load())


def get_dataset(key: str) -> CatalogueEntry | None:
    """Look up a single dataset by slug or by exact name (case-insensitive).

    Args:
        key: Slug (e.g. ``ptb-xl``) or display name.

    Returns:
        CatalogueEntry if found, None otherwise.
    """
    key_lower = key.lower()
    for entry in _load():
        if entry.slug.lower() == key_lower or entry.name.lower() == key_lower:
            return entry
    return None


def search(
    query: str | None = None,
    category: str | None = None,
    access: str | None = None,
    status: str | None = None,
) -> list[CatalogueEntry]:
    """Search and filter datasets.

    All filters are AND-combined. Each is case-insensitive substring match.

    Args:
        query: Free-text search across name, origin, format, paper, and keywords.
        category: Filter by category slug (e.g. ``12-lead-physionet``).
        access: Filter by access type (``open`` | ``credentialed`` | ``restricted``).
        status: Filter by status key (``not_started``, ``implementing``,
            ``completed``, ``needs_review``).
    """
    results: list[CatalogueEntry] = list(_load())

    if category is not None:
        c = category.lower()
        results = [r for r in results if c in r.category.lower()]

    if access is not None:
        a = access.lower()
        results = [r for r in results if a in r.access.lower()]

    if status is not None:
        s = status.lower()
        results = [r for r in results if s == r.status.lower()]

    if query is not None:
        q = query.lower()
        results = [
            r
            for r in results
            if q in r.name.lower()
            or q in (r.origin_institution or "").lower()
            or q in (r.origin_country or "").lower()
            or q in r.format.lower()
            or q in (r.paper_title or "").lower()
            or q in (r.search_keywords or "").lower()
        ]

    return results


def categories() -> list[str]:
    """Return the unique category slugs in catalogue order."""
    seen: set[str] = set()
    result: list[str] = []
    for entry in _load():
        if entry.category and entry.category not in seen:
            seen.add(entry.category)
            result.append(entry.category)
    return result


def get_download_url(key: str) -> str | None:
    """Look up the URL for a dataset by slug or name."""
    entry = get_dataset(key)
    return entry.url if entry else None


def get_config(dataset_name: str):
    """Try to find a matching YAML config for a catalogue dataset.

    Fuzzy-matches the catalogue name/slug to available config slugs by
    normalising to lowercase and removing hyphens/spaces.

    Returns ``DatasetConfig`` if found, else ``None``.
    """
    from ecgbench.config import list_available_configs, load_config

    def _normalise(s: str) -> str:
        return s.lower().replace("-", "").replace(" ", "").replace("_", "")

    entry = get_dataset(dataset_name)
    targets = {_normalise(dataset_name)}
    if entry:
        targets.add(_normalise(entry.slug))
        targets.add(_normalise(entry.name))

    for slug in list_available_configs():
        if _normalise(slug) in targets:
            return load_config(slug)
    return None


def to_dataframe():
    """Return the catalogue as a pandas DataFrame."""
    try:
        import pandas as pd
    except ImportError as err:
        raise ImportError(
            "pandas is required for to_dataframe(). "
            "Install it with: pip install ecgbench[all]"
        ) from err

    rows = [
        {
            "slug": e.slug,
            "name": e.name,
            "category": e.category,
            "status": e.status,
            "url": e.url,
            "format": e.format,
            "patients": e.patients,
            "records": e.records,
            "access": e.access,
            "license": e.license,
            "origin_institution": e.origin_institution,
            "origin_country": e.origin_country,
            "leads": e.leads,
            "paper_title": e.paper_title,
            "paper_doi": e.paper_doi,
        }
        for e in _load()
    ]
    return pd.DataFrame(rows)
