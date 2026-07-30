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

#: Closed vocabulary for ``related[].relation``, mapped to the inverse used when
#: the reverse edge is derived. Declare a relationship once, on either side.
_RELATION_INVERSES = {
    "contains": "subset_of",
    "subset_of": "contains",
    "derived_from": "has_derivative",
    "has_derivative": "derived_from",
    "same_cohort": "same_cohort",
    "sibling_release": "sibling_release",
}

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
class RelatedLink:
    """A link from one catalogue dataset to another.

    Relationships are declared once, in the front matter of either endpoint;
    ``_load`` derives the reverse edge and marks it ``derived=True``. Both
    directions are therefore always consistent — the point of not writing them
    twice by hand.

    ``shares_records`` is the field that matters for leakage: True means the two
    datasets contain the same recordings, so training on one and evaluating on
    the other contaminates the test set. ``verified`` says whether that overlap
    was checked against the actual data files, as opposed to taken from
    documentation.
    """

    slug: str
    relation: str
    shares_records: bool | None = None
    note: str = ""
    verified: bool = False
    derived: bool = False


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
    related: tuple[RelatedLink, ...] = ()
    raw: dict = field(default_factory=dict, compare=False, repr=False)


def _parse_front_matter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER.match(text)
    if not match:
        raise ValueError(f"No YAML front matter in {path}")
    data = yaml.safe_load(match.group(1))
    return data or {}


def _parse_related(slug: str, meta: dict, problems: list[str]) -> list[RelatedLink]:
    """Parse a ``related:`` block, recording anything malformed in ``problems``."""
    raw = meta.get("related") or []
    if not isinstance(raw, list):
        problems.append(f"{slug}: 'related' must be a list, got {type(raw).__name__}")
        return []

    links: list[RelatedLink] = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("slug"):
            problems.append(f"{slug}: every 'related' entry needs a 'slug'")
            continue
        relation = item.get("relation")
        if relation not in _RELATION_INVERSES:
            problems.append(
                f"{slug}: unknown relation {relation!r} to {item['slug']!r}; "
                f"expected one of {sorted(_RELATION_INVERSES)}"
            )
            continue
        if item["slug"] == slug:
            problems.append(f"{slug}: 'related' entry points at itself")
            continue
        links.append(
            RelatedLink(
                slug=item["slug"],
                relation=relation,
                shares_records=item.get("shares_records"),
                note=item.get("note", "") or "",
                verified=bool(item.get("verified", False)),
            )
        )
    return links


def _with_reverse_edges(
    declared: dict[str, list[RelatedLink]], problems: list[str]
) -> dict[str, list[RelatedLink]]:
    """Add the inverse of every declared edge, unless it is already declared."""
    resolved = {slug: list(links) for slug, links in declared.items()}

    for source, links in declared.items():
        for link in links:
            if link.slug not in resolved:
                problems.append(
                    f"{source}: 'related' points at unknown dataset {link.slug!r}"
                )
                continue
            already = any(existing.slug == source for existing in resolved[link.slug])
            if already:
                continue
            resolved[link.slug].append(
                RelatedLink(
                    slug=source,
                    relation=_RELATION_INVERSES[link.relation],
                    shares_records=link.shares_records,
                    note=link.note,
                    verified=link.verified,
                    derived=True,
                )
            )

    return {slug: sorted(links, key=lambda x: x.slug) for slug, links in resolved.items()}


def _entry_from_meta(
    slug: str, meta: dict, related: tuple[RelatedLink, ...] = ()
) -> CatalogueEntry:
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
        related=related,
        raw=meta,
    )


@functools.cache
def _load() -> tuple[CatalogueEntry, ...]:
    """Load and cache every dataset entry from the bundled .md files."""
    metas: dict[str, dict] = {}
    for path in sorted(_datasets_dir().glob("*.md")):
        metas[path.stem] = _parse_front_matter(path)

    # Relationships are declared once and inverted here, so a malformed block is
    # a repo authoring error. Report every problem at once rather than the first.
    problems: list[str] = []
    declared = {slug: _parse_related(slug, meta, problems) for slug, meta in metas.items()}
    related = _with_reverse_edges(declared, problems)
    if problems:
        raise ValueError(
            "Invalid 'related' blocks in docs/_datasets/:\n  - "
            + "\n  - ".join(problems)
        )

    entries = [
        _entry_from_meta(slug, meta, tuple(related.get(slug, ())))
        for slug, meta in metas.items()
    ]

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
