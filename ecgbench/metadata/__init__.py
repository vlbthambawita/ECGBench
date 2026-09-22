"""Unified, queryable metadata for every dataset ECGBench knows about.

The catalogue (``docs/_datasets/*.md``) and the configs
(``ecgbench/data/configs/*.yaml``) are merged by ``build`` into one
``DatasetMeta`` per dataset, exported to ``ecgbench/data/metadata.json`` and
read back by ``MetadataStore``. Any alias — ``ptb-xl``, ``ptbxl``, ``PTB-XL`` —
resolves to the same record.

Typical use::

    from ecgbench import metadata

    meta = metadata.get("mit-bih-arrhythmia-database")   # same as get("mitdb")
    meta.signal.leads, meta.access.license_text, meta.implementation_state

    for m in metadata.search("holter", leads=2, access="open"):   # FTS5, ranked
        print(m.dataset_id, m.records_display)
    metadata.search('"atrial fib*" NOT paediatric')             # FTS5 syntax verbatim

    metadata.related("ptbxl")   # leakage edges, both directions

Importing this package costs nothing beyond the standard library; the JSON is
parsed on the first call that needs it.
"""

from __future__ import annotations

from ecgbench.metadata.build import (
    DEFAULT_JSON_PATH,
    SQLITE_PATH,
    BuildResult,
    MetadataBuildError,
    ModelDiff,
    build_all,
    build_model,
    content_digest,
    diff_exports,
    load_json,
    parse_count,
    to_json,
    write_json,
    write_sqlite,
)
from ecgbench.metadata.identity import AliasIndex, UnknownDatasetError, resolve
from ecgbench.metadata.model import (
    IMPLEMENTATION_STATES,
    SCHEMA_VERSION,
    SOURCE_PRECEDENCE,
    AccessMeta,
    DatasetMeta,
    Fact,
    FieldMeta,
    Provenance,
    RelationMeta,
    SignalMeta,
    SplitMeta,
)
from ecgbench.metadata.store import MetadataQueryError, MetadataStore, SearchHit, open_store


def get(key: str) -> DatasetMeta:
    """``open_store().get(key)`` — the record for any alias of a dataset."""
    return open_store().get(key)


def search(query: str | None = None, **filters) -> list[DatasetMeta]:
    """``open_store().search(query, **filters)`` — see ``MetadataStore.search``."""
    return open_store().search(query, **filters)


def search_ranked(query: str | None = None, **filters) -> list[SearchHit]:
    """``open_store().search_ranked(query, **filters)`` — results with their scores."""
    return open_store().search_ranked(query, **filters)


def related(key: str) -> list[RelationMeta]:
    """``open_store().related(key)`` — edges to other datasets."""
    return open_store().related(key)


def list_all() -> list[DatasetMeta]:
    """``open_store().all()`` — every dataset, sorted by ``dataset_id``."""
    return open_store().all()


__all__ = [
    # model
    "DatasetMeta",
    "SignalMeta",
    "AccessMeta",
    "SplitMeta",
    "RelationMeta",
    "FieldMeta",
    "Fact",
    "Provenance",
    "IMPLEMENTATION_STATES",
    "SOURCE_PRECEDENCE",
    "SCHEMA_VERSION",
    # identity
    "AliasIndex",
    "UnknownDatasetError",
    "resolve",
    # store
    "MetadataStore",
    "MetadataQueryError",
    "SearchHit",
    "open_store",
    "get",
    "search",
    "search_ranked",
    "related",
    "list_all",
    # build
    "build_model",
    "build_all",
    "content_digest",
    "diff_exports",
    "to_json",
    "write_json",
    "write_sqlite",
    "load_json",
    "parse_count",
    "BuildResult",
    "ModelDiff",
    "MetadataBuildError",
    "DEFAULT_JSON_PATH",
    "SQLITE_PATH",
]
