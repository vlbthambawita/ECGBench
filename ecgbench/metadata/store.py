"""Read-side access to the compiled metadata: lookup, ranked search, relations.

``open_store()`` loads the export bundled in the wheel
(``ecgbench/data/metadata.json``) into a ``MetadataStore``, attaches the SQLite
index next to it (``metadata.sqlite``, opened read-only) for ranked full-text
search, and caches the result so a process does this once.

Two things happen on open that a caller does not see unless they go wrong:

- **Staleness.** In a source checkout, if any catalogue Markdown, config YAML or
  label module has changed since the last build (mtime + size fingerprint in
  ``metadata.sources.json``), both files are rebuilt into ``ecgbench/data/``
  and one log line says so. An installed wheel has no fingerprint file and no
  writable sources, so this branch is skipped there.
- **FTS5 fallback.** If the index is missing and cannot be written, or was
  built without FTS5, or the runtime SQLite lacks FTS5, free-text search falls
  back to the case-insensitive substring match over the same fields and warns
  once per store. Structured filters are unaffected.

Free-text queries are FTS5 syntax passed through verbatim — ``"atrial fib*"``,
``holter NOT paediatric``, ``"long term"`` — and a query SQLite rejects raises
``MetadataQueryError`` quoting its message. Results are ranked by ``bm25()``
with the name weighted highest, then keywords and aliases, description,
institution, page prose, and (from Phase 3) field names — plus a small
*implementation prior*: a catalogue-only entry is pushed 0.5 bm25 units down
and a label-less config 0.25, so that at near-equal relevance the dataset a user
can actually load comes first (``ptb`` gives PTB-XL before PTB-XL+, whose
shorter page would otherwise win on length normalisation alone).
"""

from __future__ import annotations

import functools
import logging
import sqlite3
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ecgbench.metadata.identity import AliasIndex
from ecgbench.metadata.model import IMPLEMENTATION_STATES, DatasetMeta, RelationMeta

logger = logging.getLogger(__name__)

#: ``bm25()`` weights for ``build.FTS_COLUMNS`` — name, keywords, description,
#: institution, prose, field_text.
BM25_WEIGHTS: tuple[int, ...] = (10, 5, 3, 2, 1, 1)

#: Added to the (negative, lower-is-better) bm25 score by ``implementation_state``:
#: the rank prior described in the module docstring. Anything not listed adds 0.
STATE_PENALTY: dict[str, float] = {"catalogue_only": 0.5, "config": 0.25}


class MetadataQueryError(ValueError):
    """SQLite rejected a full-text query; the message quotes its reason."""


@dataclass(frozen=True)
class SearchHit:
    """One ranked search result.

    Attributes:
        meta: The dataset.
        score: ``bm25()`` score from FTS5 plus the ``STATE_PENALTY`` prior (more
            negative is a better match), or ``None`` when the substring path or a
            filter-only query produced it.
    """

    meta: DatasetMeta
    score: float | None


class MetadataStore:
    """An in-memory model with an alias index and an optional FTS5 index.

    Construct one directly from a tuple of ``DatasetMeta`` for tests (substring
    search, no warning), or call ``open_store()`` for the bundled data with its
    index attached.
    """

    def __init__(self, model: tuple[DatasetMeta, ...], source: str = "<memory>"):
        self._model = tuple(sorted(model, key=lambda m: m.dataset_id))
        self._by_id = {m.dataset_id: m for m in self._model}
        self._aliases = AliasIndex(self._model)
        self.source = source
        self._fts: sqlite3.Connection | None = None
        self._fts_reason: str | None = None
        self._warned = False

    # -- index ---------------------------------------------------------------------

    def attach_index(self, path: Path | str, expected_digest: str | None = None) -> bool:
        """Open the SQLite index at ``path`` read-only for ranked search.

        Refuses — and remembers why, for the one-time warning — when the file is
        missing, was built for a different model (``meta.content_digest`` differs
        from ``expected_digest``), lacks the FTS5 table, or the runtime SQLite
        cannot read FTS5 tables.

        Returns:
            ``True`` when ranked search is now available.
        """
        from ecgbench.metadata.build import fts5_available, read_sqlite_meta

        target = Path(path)
        meta = read_sqlite_meta(target)
        if not meta:
            self._fts_reason = f"index {target} is missing or unreadable"
            return False
        if expected_digest is not None and meta.get("content_digest") != expected_digest:
            self._fts_reason = f"index {target} was built for a different model"
            return False
        if meta.get("fts") != "fts5":
            self._fts_reason = f"index {target} was built without FTS5"
            return False
        if not fts5_available():
            self._fts_reason = "this Python's SQLite has no FTS5"
            return False
        try:
            conn = sqlite3.connect(
                f"{target.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False
            )
            conn.execute("SELECT count(*) FROM dataset_fts").fetchone()
        except sqlite3.Error as exc:
            self._fts_reason = f"index {target} cannot be queried: {exc}"
            return False
        self._fts = conn
        self._fts_reason = None
        return True

    @property
    def fts_enabled(self) -> bool:
        """Whether free-text queries are ranked by FTS5 rather than substring-matched."""
        return self._fts is not None

    @property
    def fts_fallback_reason(self) -> str | None:
        """Why ranked search is unavailable, or ``None`` when it is available."""
        return self._fts_reason

    # -- basics --------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._model)

    def __iter__(self) -> Iterator[DatasetMeta]:
        return iter(self._model)

    def __contains__(self, key: str) -> bool:
        return key in self._aliases

    def all(self) -> list[DatasetMeta]:
        """Every dataset, sorted by ``dataset_id``."""
        return list(self._model)

    def resolve(self, key: str) -> str:
        """Map any alias to a ``dataset_id``; raises ``UnknownDatasetError``."""
        return self._aliases.resolve(key)

    def get(self, key: str) -> DatasetMeta:
        """The record for ``key`` (catalogue slug, config slug or display name).

        Raises:
            UnknownDatasetError: no dataset answers to ``key``; the message names
                close matches.
        """
        return self._by_id[self.resolve(key)]

    def related(self, key: str) -> list[RelationMeta]:
        """Edges from ``key``'s dataset, both declared and derived directions."""
        return list(self.get(key).relations)

    # -- search --------------------------------------------------------------------

    def search_ranked(
        self,
        query: str | None = None,
        *,
        limit: int | None = None,
        leads: int | None = None,
        fs: int | None = None,
        signal_format: str | None = None,
        access: str | None = None,
        license: str | None = None,
        category: str | None = None,
        state: str | None = None,
        min_records: int | None = None,
        max_records: int | None = None,
        has_labels: bool | None = None,
        has_patient_id: bool | None = None,
        published: bool | None = None,
    ) -> list[SearchHit]:
        """Like ``search`` but returns ``SearchHit`` with the ``bm25()`` score.

        With a query and an attached index the order is by relevance; otherwise
        by ``dataset_id``. Structured filters apply after ranking and keep it.
        """
        if state is not None and state not in IMPLEMENTATION_STATES:
            raise ValueError(f"state must be one of {IMPLEMENTATION_STATES}, got {state!r}")

        if query and query.strip():
            if self._fts is not None:
                hits = self._fts_query(query)
            else:
                self._warn_fallback()
                needle = query.casefold()
                hits = [SearchHit(m, None) for m in self._model if needle in _haystack(m)]
        else:
            hits = [SearchHit(m, None) for m in self._model]

        hits = [
            h
            for h in hits
            if _matches(
                h.meta,
                leads=leads,
                fs=fs,
                signal_format=signal_format,
                access=access,
                license=license,
                category=category,
                state=state,
                min_records=min_records,
                max_records=max_records,
                has_labels=has_labels,
                has_patient_id=has_patient_id,
                published=published,
            )
        ]
        return hits[:limit] if limit is not None else hits

    def search(
        self,
        query: str | None = None,
        *,
        limit: int | None = None,
        leads: int | None = None,
        fs: int | None = None,
        signal_format: str | None = None,
        access: str | None = None,
        license: str | None = None,
        category: str | None = None,
        state: str | None = None,
        min_records: int | None = None,
        max_records: int | None = None,
        has_labels: bool | None = None,
        has_patient_id: bool | None = None,
        published: bool | None = None,
    ) -> list[DatasetMeta]:
        """Filter datasets; every given criterion must hold (AND).

        Args:
            query: FTS5 query when the index is attached (``"atrial fib*"``,
                ``holter NOT paediatric``, ``"long term"``); otherwise a
                case-insensitive substring matched against name, aliases,
                keywords, description, institution, country, paper title, the
                catalogue's format string and page prose.
            limit: Keep at most this many results, after filtering.
            leads: Exact lead count. Uses the signal facet where a config exists,
                else the catalogue's ``leads`` when it is a number.
            fs: A sampling rate the release ships (config datasets only).
            signal_format: ``wfdb``, ``csv``, ``edf``, … (config datasets only).
            access: ``open`` | ``credentialed`` | ``restricted``.
            license: Substring of the licence name or URL.
            category: Exact catalogue category.
            state: Exact ``implementation_state``.
            min_records: Inclusive lower bound on the parsed record count;
                datasets whose count did not parse are excluded.
            max_records: Inclusive upper bound, same caveat.
            has_labels: Whether a label loader or declarative columns exist.
            has_patient_id: Whether folds are patient-grouped (config datasets only).
            published: Whether fold CSVs may be fetched from the Hub.

        Raises:
            ValueError: ``state`` is not one of ``IMPLEMENTATION_STATES``.
            MetadataQueryError: the index rejected ``query`` as FTS5 syntax.
        """
        hits = self.search_ranked(
            query,
            limit=limit,
            leads=leads,
            fs=fs,
            signal_format=signal_format,
            access=access,
            license=license,
            category=category,
            state=state,
            min_records=min_records,
            max_records=max_records,
            has_labels=has_labels,
            has_patient_id=has_patient_id,
            published=published,
        )
        return [h.meta for h in hits]

    def _fts_query(self, query: str) -> list[SearchHit]:
        assert self._fts is not None
        weights = ", ".join(str(w) for w in BM25_WEIGHTS)
        prior = " ".join(
            f"WHEN '{state}' THEN {penalty}" for state, penalty in STATE_PENALTY.items()
        )
        sql = (
            f"SELECT f.dataset_id, bm25(dataset_fts, {weights}) "
            f"+ CASE d.implementation_state {prior} ELSE 0 END AS score "
            "FROM dataset_fts AS f JOIN dataset AS d ON d.dataset_id = f.dataset_id "
            "WHERE dataset_fts MATCH ? ORDER BY score, f.dataset_id"
        )
        try:
            rows = self._fts.execute(sql, (query,)).fetchall()
        except sqlite3.OperationalError as exc:
            raise MetadataQueryError(
                f"invalid search query {query!r}: {exc}. The query is FTS5 syntax; "
                'quote a phrase with punctuation, e.g. "ptb-xl".'
            ) from None
        return [
            SearchHit(self._by_id[dataset_id], float(score))
            for dataset_id, score in rows
            if dataset_id in self._by_id
        ]

    def _warn_fallback(self) -> None:
        if self._warned or self._fts_reason is None:
            return
        self._warned = True
        warnings.warn(
            f"ranked search unavailable ({self._fts_reason}); "
            "falling back to case-insensitive substring matching",
            RuntimeWarning,
            stacklevel=4,
        )


# --------------------------------------------------------------------------- filters


def _matches(
    m: DatasetMeta,
    *,
    leads: int | None,
    fs: int | None,
    signal_format: str | None,
    access: str | None,
    license: str | None,
    category: str | None,
    state: str | None,
    min_records: int | None,
    max_records: int | None,
    has_labels: bool | None,
    has_patient_id: bool | None,
    published: bool | None,
) -> bool:
    if leads is not None and _leads_of(m) != leads:
        return False
    if fs is not None and (m.signal is None or fs not in m.signal.sampling_rates):
        return False
    if signal_format is not None and (
        m.signal is None or m.signal.format.casefold() != signal_format.casefold()
    ):
        return False
    if access is not None and m.access.access.casefold() != access.casefold():
        return False
    if license is not None:
        wanted = license.casefold()
        in_text = wanted in (m.access.license_text or "").casefold()
        in_url = wanted in (m.access.license_url or "").casefold()
        if not (in_text or in_url):
            return False
    if category is not None and m.category.casefold() != category.casefold():
        return False
    if state is not None and m.implementation_state != state:
        return False
    if min_records is not None and (m.records is None or m.records < min_records):
        return False
    if max_records is not None and (m.records is None or m.records > max_records):
        return False
    if has_labels is not None and m.has_labels is not has_labels:
        return False
    if has_patient_id is not None and (
        m.split is None or m.split.has_patient_id is not has_patient_id
    ):
        return False
    if published is not None and m.published is not published:
        return False
    return True


def _haystack(meta: DatasetMeta) -> str:
    fields = (
        meta.name,
        " ".join(meta.aliases),
        meta.search_keywords,
        meta.description,
        meta.origin_institution,
        meta.origin_country or "",
        meta.paper_title or "",
        str(meta.fact("format").value) if meta.fact("format") else "",
        meta.prose,
    )
    return "\n".join(fields).casefold()


def _leads_of(meta: DatasetMeta) -> int | None:
    if meta.signal is not None:
        return meta.signal.leads
    fact = meta.fact("leads")
    value = fact.value if fact is not None else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# --------------------------------------------------------------------------- opening


def _ensure_index(store: MetadataStore, sqlite_path: Path, digest: str | None) -> None:
    """Attach the index at ``sqlite_path``, (re)building it first if it does not match."""
    from ecgbench.metadata.build import read_sqlite_meta, write_sqlite

    meta = read_sqlite_meta(sqlite_path)
    if not meta or (digest is not None and meta.get("content_digest") != digest):
        try:
            write_sqlite(tuple(store.all()), sqlite_path)
            logger.info("built metadata index %s", sqlite_path)
        except (OSError, sqlite3.Error) as exc:
            logger.warning("cannot write metadata index %s: %s", sqlite_path, exc)
    store.attach_index(sqlite_path, expected_digest=digest)


def _refresh_if_stale() -> None:
    """In a source checkout, rebuild the derived files when the sources changed."""
    from ecgbench.metadata import build

    if not build.is_source_checkout():
        return
    try:
        if not build.sources_changed():
            return
        result = build.build_all()
    except (OSError, sqlite3.Error, ValueError) as exc:
        logger.warning("metadata sources changed but the index could not be rebuilt: %s", exc)
        return
    logger.info(
        "metadata sources changed; rebuilt %s%s",
        result.sqlite_path.name,
        " and " + result.json_path.name if result.json_written else "",
    )


@functools.lru_cache(maxsize=None)
def _bundled_store() -> MetadataStore:
    from ecgbench.metadata.build import (
        DEFAULT_JSON_PATH,
        SQLITE_PATH,
        build_model,
        load_json,
        read_digest,
    )

    _refresh_if_stale()
    if DEFAULT_JSON_PATH.is_file():
        store = MetadataStore(load_json(DEFAULT_JSON_PATH), source=str(DEFAULT_JSON_PATH))
        digest = read_digest(DEFAULT_JSON_PATH)
    else:
        logger.warning(
            "%s is missing; building the metadata model from the sources instead. "
            "Run `ecgbench metadata build` to create it.",
            DEFAULT_JSON_PATH,
        )
        store = MetadataStore(build_model(), source="<built from sources>")
        digest = None
    _ensure_index(store, SQLITE_PATH, digest)
    return store


def open_store(path: Path | str | None = None) -> MetadataStore:
    """Open the bundled metadata, or an export at ``path``.

    The bundled store is cached for the process. With ``path`` (a
    ``metadata.json``), the index is expected as ``metadata.sqlite`` beside it
    and is built there when missing or stale, if the directory is writable.
    """
    if path is None:
        return _bundled_store()
    from ecgbench.metadata.build import load_json, read_digest

    json_path = Path(path)
    store = MetadataStore(load_json(json_path), source=str(json_path))
    _ensure_index(store, json_path.with_suffix(".sqlite"), read_digest(json_path))
    return store
