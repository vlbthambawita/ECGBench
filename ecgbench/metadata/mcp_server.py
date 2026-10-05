"""An MCP server exposing the metadata layer to agents (``ecgbench mcp``).

Five tools, each a thin wrapper on ``MetadataStore`` returning JSON-ready
dicts: ``search_datasets`` (FTS5 query plus the structured filters of
``ecgbench search``), ``list_datasets``, ``get_dataset`` (the full record),
``list_fields`` (the declared label columns) and ``related_datasets`` (the
leakage edges). Nothing here touches records, signals or the network: the
server reads the bundled ``metadata.json`` / ``metadata.sqlite`` only, so it
can be given to any agent without a data path.

The ``mcp`` SDK (the ``mcp`` extra) is imported only in :func:`build_mcp_server`;
the tool functions themselves are plain Python so they can be called, and
tested, without it. SDK 1.x names the server class ``FastMCP``, 2.x renamed it
``MCPServer`` with the same ``tool`` decorator; both are supported.

Register it with Claude Code as::

    claude mcp add ecgbench -- ecgbench mcp

or in any MCP client config as the command ``ecgbench`` with arguments
``["mcp"]`` over stdio.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

from ecgbench.metadata.identity import UnknownDatasetError
from ecgbench.metadata.model import IMPLEMENTATION_STATES, DatasetMeta
from ecgbench.metadata.store import MetadataQueryError, MetadataStore, open_store

SERVER_NAME = "ecgbench"
TOOL_NAMES = (
    "search_datasets",
    "list_datasets",
    "get_dataset",
    "list_fields",
    "related_datasets",
)
#: ``search_datasets`` returns at most this many hits unless asked otherwise.
DEFAULT_LIMIT = 10

INSTRUCTIONS = (
    "ECGBench's catalogue of ECG datasets: 64 surveyed, 51 with a config that validates, "
    "splits and loads them. Use search_datasets for free text ('holter', '\"atrial fib*\"') "
    "with filters (leads, fs, access, state ...); get_dataset for one dataset's full record "
    "by any id, slug or display name; list_fields for the label columns ECGBench exposes; "
    "related_datasets for datasets that share recordings (training on one and evaluating "
    "on the other leaks). Ids are config slugs such as 'ptbxl' or 'mitdb'."
)


# --------------------------------------------------------------------------- shapes


def summary(meta: DatasetMeta) -> dict[str, Any]:
    """The compact row ``search_datasets`` and ``list_datasets`` return."""
    signal = meta.signal
    return {
        "dataset_id": meta.dataset_id,
        "name": meta.name,
        "category": meta.category,
        "implementation_state": meta.implementation_state,
        "records": meta.records,
        "records_display": meta.records_display,
        "patients": meta.patients,
        "leads": signal.leads if signal else _fact_value(meta, "leads"),
        "sampling_rates": list(signal.sampling_rates) if signal else None,
        "signal_format": signal.format if signal else None,
        "access": meta.access.access,
        "license": meta.access.license_text,
        "has_labels": meta.has_labels,
        "fold_csvs_published": meta.access.publish_fold_csvs,
        "description": meta.description,
    }


def _fact_value(meta: DatasetMeta, key: str) -> Any:
    fact = meta.fact(key)
    return fact.value if fact is not None else None


def _jsonable(value: Any) -> Any:
    """``to_dict`` output still holds tuples; MCP structured content wants lists."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _tool_error() -> type[Exception]:
    """The exception whose message the SDK forwards to the model.

    Anything else a tool raises reaches the client as the bare
    "Error executing tool <name>", with the text kept server-side. Without the
    SDK installed (the functions are usable on their own) it is ``ValueError``.
    """
    try:
        from mcp.server.mcpserver.exceptions import ToolError  # mcp 2.x
    except ImportError:
        try:
            from mcp.server.fastmcp.exceptions import ToolError  # mcp 1.x
        except ImportError:
            return ValueError
    return ToolError


# --------------------------------------------------------------------------- tools


class ToolFunctions:
    """The five tools over one store, as plain callables.

    Kept on a class rather than as module-level closures so a test can build
    them over an in-memory ``MetadataStore`` and so the SDK's decorator sees
    functions with complete type hints and docstrings (which become the tool
    schema and description).
    """

    def __init__(self, store: MetadataStore | None = None):
        self._store = store

    @property
    def store(self) -> MetadataStore:
        if self._store is None:
            self._store = open_store()
        return self._store

    def search_datasets(
        self,
        query: str | None = None,
        limit: int = DEFAULT_LIMIT,
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
    ) -> list[dict[str, Any]]:
        """Search the catalogue by free text and structured filters, best match first.

        Args:
            query: FTS5 query over name, keywords, description and field names:
                ``holter``, ``"atrial fib*"``, ``paediatric NOT simulated``. Quote a
                hyphenated term (``"ptb-xl"``). Omit it to filter only.
            limit: Maximum number of hits.
            leads: Exact lead count (2 for two-lead Holter sets, 12 for standard ECG).
            fs: A sampling rate the dataset ships, in Hz.
            signal_format: ``wfdb``, ``csv``, ``csv_lead_rows``, ``opensignals``, ``npy``,
                ``hdf5``, ``edf`` or ``mat``.
            access: ``open``, ``credentialed`` or ``restricted``.
            license: Substring of the licence text (``CC BY``, ``ODC``).
            category: Catalogue category id (``two-lead``, ``twelve-lead`` ...).
            state: ``catalogue_only``, ``config``, ``config_labels`` or ``published``.
            min_records: Lower bound on the record count.
            max_records: Upper bound on the record count.
            has_labels: Only datasets whose labels ECGBench can load.
            has_patient_id: Only datasets with a patient identifier for grouped splits.
            published: Only datasets whose fold CSVs are on the Hub.
        """
        if state is not None and state not in IMPLEMENTATION_STATES:
            raise _tool_error()(f"state must be one of {IMPLEMENTATION_STATES}, got {state!r}")
        try:
            hits = self.store.search_ranked(
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
        except MetadataQueryError as exc:
            raise _tool_error()(str(exc)) from exc
        return [{**summary(h.meta), "score": h.score} for h in hits]

    def list_datasets(
        self, state: str | None = None, category: str | None = None
    ) -> list[dict[str, Any]]:
        """Every dataset ECGBench knows, optionally filtered by implementation state or category.

        Args:
            state: ``catalogue_only`` (surveyed only), ``config`` (loadable, no labels),
                ``config_labels`` (labels, fold CSVs withheld) or ``published``.
            category: Catalogue category id.
        """
        if state is not None and state not in IMPLEMENTATION_STATES:
            raise _tool_error()(f"state must be one of {IMPLEMENTATION_STATES}, got {state!r}")
        rows = self.store.all()
        if state is not None:
            rows = [m for m in rows if m.implementation_state == state]
        if category is not None:
            rows = [m for m in rows if m.category == category]
        return [summary(m) for m in rows]

    def get_dataset(self, dataset: str) -> dict[str, Any]:
        """The full record for one dataset: identity, signal, access, split, relations,
        declared label fields, artefact snapshots and every sourced fact with provenance.

        Args:
            dataset: Dataset id, catalogue slug or display name, case-insensitive
                (``ptbxl``, ``ptb-xl`` and ``PTB-XL`` are the same record).
        """
        return _jsonable(self._get(dataset).to_dict())

    def list_fields(self, dataset: str) -> list[dict[str, Any]]:
        """The label columns ``load_labels()`` returns for a dataset, with type, unit,
        vocabulary and description. Empty for a dataset whose labels are unavailable.

        Args:
            dataset: Dataset id, catalogue slug or display name.
        """
        meta = self._get(dataset)
        return [_jsonable(dataclasses.asdict(f)) for f in meta.fields]

    def related_datasets(self, dataset: str) -> list[dict[str, Any]]:
        """Datasets related to this one, both directions, with ``shares_records``:
        true means the two hold the same recordings, so training on one and
        evaluating on the other contaminates the test set.

        Args:
            dataset: Dataset id, catalogue slug or display name.
        """
        key = self._get(dataset).dataset_id
        return [
            {"source": key, **_jsonable(dataclasses.asdict(r))} for r in self.store.related(key)
        ]

    def _get(self, key: str) -> DatasetMeta:
        try:
            return self.store.get(key)
        except UnknownDatasetError as exc:
            raise _tool_error()(str(exc)) from exc


# --------------------------------------------------------------------------- server


def _server_class():
    """``MCPServer`` (mcp 2.x) or ``FastMCP`` (mcp 1.x), whichever is installed."""
    try:
        from mcp.server.mcpserver import MCPServer

        return MCPServer
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP

        return FastMCP
    except ImportError as exc:
        raise ImportError(
            "the MCP server needs the mcp SDK: pip install ecgbench[mcp]"
        ) from exc


def build_mcp_server(store: MetadataStore | None = None):
    """An MCP server with the five tools registered over ``store`` (default: the bundled index).

    Raises:
        ImportError: the ``mcp`` SDK is not installed.
    """
    server_cls = _server_class()
    try:
        from ecgbench import __version__ as version
    except ImportError:  # pragma: no cover - only outside an installed package
        version = ""
    kwargs: dict[str, Any] = {"instructions": INSTRUCTIONS}
    if "version" in inspect.signature(server_cls.__init__).parameters:  # mcp 2.x
        kwargs["version"] = version
    server = server_cls(SERVER_NAME, **kwargs)
    tools = ToolFunctions(store)
    for name in TOOL_NAMES:
        server.tool(name=name)(getattr(tools, name))
    return server


def run_server(transport: str = "stdio") -> None:
    """Serve the tools until the client disconnects (``ecgbench mcp``).

    Args:
        transport: ``stdio`` (what MCP clients spawn), ``sse`` or ``streamable-http``.
    """
    build_mcp_server().run(transport=transport)
