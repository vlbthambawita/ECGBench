"""Read-side access to the compiled metadata: lookup, filtered search, relations.

``open_store()`` loads the export bundled in the wheel
(``ecgbench/data/metadata.json``) into a ``MetadataStore`` and caches it, so a
process parses the file once. If the file is missing — a source checkout that
has never been built — the store is built from the sources instead and a
warning says so.

Free-text search is a case-insensitive substring match over name, aliases,
keywords, description, institution, country and page prose: a superset of what
``catalogue.search()`` looks at. Ranked full-text search arrives with the
SQLite backend in a later phase; the signature here is the one it will keep.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Iterator
from pathlib import Path

from ecgbench.metadata.identity import AliasIndex
from ecgbench.metadata.model import IMPLEMENTATION_STATES, DatasetMeta, RelationMeta

logger = logging.getLogger(__name__)


class MetadataStore:
    """An in-memory model with an alias index over it.

    Construct one directly from a tuple of ``DatasetMeta`` for tests, or call
    ``open_store()`` for the bundled data.
    """

    def __init__(self, model: tuple[DatasetMeta, ...], source: str = "<memory>"):
        self._model = tuple(sorted(model, key=lambda m: m.dataset_id))
        self._by_id = {m.dataset_id: m for m in self._model}
        self._aliases = AliasIndex(self._model)
        self.source = source

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

    def search(
        self,
        query: str | None = None,
        *,
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
            query: Case-insensitive substring matched against name, aliases,
                keywords, description, institution, country, paper title, the
                catalogue's format string and page prose.
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
        """
        if state is not None and state not in IMPLEMENTATION_STATES:
            raise ValueError(f"state must be one of {IMPLEMENTATION_STATES}, got {state!r}")

        results = list(self._model)
        if query:
            needle = query.casefold()
            results = [m for m in results if needle in _haystack(m)]
        if leads is not None:
            results = [m for m in results if _leads_of(m) == leads]
        if fs is not None:
            results = [m for m in results if m.signal is not None and fs in m.signal.sampling_rates]
        if signal_format is not None:
            wanted = signal_format.casefold()
            results = [
                m for m in results if m.signal is not None and m.signal.format.casefold() == wanted
            ]
        if access is not None:
            wanted = access.casefold()
            results = [m for m in results if m.access.access.casefold() == wanted]
        if license is not None:
            wanted = license.casefold()
            results = [
                m
                for m in results
                if wanted in (m.access.license_text or "").casefold()
                or wanted in (m.access.license_url or "").casefold()
            ]
        if category is not None:
            wanted = category.casefold()
            results = [m for m in results if m.category.casefold() == wanted]
        if state is not None:
            results = [m for m in results if m.implementation_state == state]
        if min_records is not None:
            results = [m for m in results if m.records is not None and m.records >= min_records]
        if max_records is not None:
            results = [m for m in results if m.records is not None and m.records <= max_records]
        if has_labels is not None:
            results = [m for m in results if m.has_labels is has_labels]
        if has_patient_id is not None:
            results = [
                m
                for m in results
                if m.split is not None and m.split.has_patient_id is has_patient_id
            ]
        if published is not None:
            results = [m for m in results if m.published is published]
        return results


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


@functools.lru_cache(maxsize=None)
def _bundled_store() -> MetadataStore:
    from ecgbench.metadata.build import DEFAULT_JSON_PATH, build_model, load_json

    if DEFAULT_JSON_PATH.is_file():
        return MetadataStore(load_json(DEFAULT_JSON_PATH), source=str(DEFAULT_JSON_PATH))
    logger.warning(
        "%s is missing; building the metadata model from the sources instead. "
        "Run `ecgbench metadata build` to create it.",
        DEFAULT_JSON_PATH,
    )
    return MetadataStore(build_model(), source="<built from sources>")


def open_store(path: Path | str | None = None) -> MetadataStore:
    """Open the bundled metadata, or an export at ``path``.

    The bundled store is cached for the process; a ``path`` is read afresh.
    """
    if path is None:
        return _bundled_store()
    from ecgbench.metadata.build import load_json

    return MetadataStore(load_json(path), source=str(path))
