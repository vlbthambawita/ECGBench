"""``ecgbench list``, ``search``, ``info``, ``fields`` and ``related``.

Read-only views over the metadata store. Each command has a public ``run_*``
function returning typed objects — the Python API — and a private ``_cli_*``
adapter that formats them. ``--format json`` writes one JSON document to stdout
and nothing else there (logging goes to stderr), so the output can be piped.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence

from ecgbench.metadata.identity import UnknownDatasetError
from ecgbench.metadata.model import IMPLEMENTATION_STATES, DatasetMeta, FieldMeta, RelationMeta
from ecgbench.metadata.store import MetadataQueryError, SearchHit, open_store

_FORMATS = ("table", "json", "csv")


# --------------------------------------------------------------------------- Python API


def run_list(
    state: str | None = None,
    category: str | None = None,
    **filters,
) -> list[DatasetMeta]:
    """Every dataset matching the filters, sorted by ``dataset_id``.

    Args:
        state: ``implementation_state`` to keep (``catalogue_only``, ``config``,
            ``config_labels``, ``published``).
        category: Catalogue category to keep.
        **filters: Any further keyword accepted by ``MetadataStore.search``.
    """
    return open_store().search(None, state=state, category=category, **filters)


def run_search(
    query: str | None = None, *, limit: int | None = None, **filters
) -> list[DatasetMeta]:
    """Ranked full-text search with structured filters — ``MetadataStore.search``.

    Args:
        query: FTS5 query (``"atrial fib*"``, ``holter NOT paediatric``); with the
            index unavailable, a case-insensitive substring.
        limit: Keep at most this many results.
        **filters: ``leads``, ``fs``, ``signal_format``, ``access``, ``license``,
            ``category``, ``state``, ``min_records``, ``max_records``,
            ``has_labels``, ``has_patient_id``, ``published``.

    Raises:
        MetadataQueryError: the index rejected ``query`` as FTS5 syntax.
    """
    return open_store().search(query, limit=limit, **filters)


def run_search_ranked(
    query: str | None = None, *, limit: int | None = None, **filters
) -> list[SearchHit]:
    """``run_search`` with each hit's ``bm25()`` score attached."""
    return open_store().search_ranked(query, limit=limit, **filters)


def run_info(key: str) -> DatasetMeta:
    """The record for ``key`` — a catalogue slug, config slug or display name.

    Raises:
        UnknownDatasetError: nothing answers to ``key``; the message names
            close matches.
    """
    return open_store().get(key)


def run_related(key: str) -> list[RelationMeta]:
    """Edges from ``key``'s dataset to others, declared and derived alike."""
    return open_store().related(key)


def run_fields(key: str) -> tuple[FieldMeta, ...]:
    """The declared label columns of ``key``'s dataset (empty when undeclared).

    Raises:
        UnknownDatasetError: nothing answers to ``key``.
    """
    return open_store().get(key).fields


# --------------------------------------------------------------------------- formatting


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    cells = [[_cell(v) for v in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))
    lines = [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip(),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for row in cells:
        lines.append("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)).rstrip())
    return "\n".join(lines)


def _cell(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def _list_rows(rows: list[DatasetMeta]) -> tuple[list[str], list[list[object]]]:
    headers = ["dataset_id", "name", "category", "state", "records", "leads", "format", "access"]
    table = []
    for m in rows:
        table.append(
            [
                m.dataset_id,
                m.name,
                m.category,
                m.implementation_state,
                m.records_display,
                _leads(m),
                m.signal.format if m.signal else None,
                m.access.access,
            ]
        )
    return headers, table


def _leads(meta: DatasetMeta) -> object:
    if meta.signal is not None:
        return meta.signal.leads
    fact = meta.fact("leads")
    return fact.value if fact is not None else None


def format_list(rows: list[DatasetMeta], fmt: str = "table") -> str:
    """Render ``run_list`` output as an aligned table, JSON or CSV."""
    if fmt == "json":
        return json.dumps([m.to_dict() for m in rows], indent=2, ensure_ascii=False)
    headers, table = _list_rows(rows)
    if fmt == "csv":
        import io

        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows([[_cell(v) if v is not None else "" for v in row] for row in table])
        return buffer.getvalue().rstrip("\n")
    return _table(headers, table)


def format_search(hits: list[SearchHit], fmt: str = "table") -> str:
    """Render ranked hits; the table adds a rank and, when FTS5 ranked them, a score."""
    if fmt == "json":
        return json.dumps(
            [{"rank": i + 1, "score": h.score, **h.meta.to_dict()} for i, h in enumerate(hits)],
            indent=2,
            ensure_ascii=False,
        )
    ranked = any(h.score is not None for h in hits)
    headers = ["rank", *(["score"] if ranked else []), "dataset_id", "name", "state", "records",
               "leads", "format", "access"]
    table = []
    for i, h in enumerate(hits, start=1):
        m = h.meta
        row: list[object] = [i]
        if ranked:
            row.append(f"{h.score:.3g}" if h.score is not None else None)
        row += [
            m.dataset_id,
            m.name,
            m.implementation_state,
            m.records_display,
            _leads(m),
            m.signal.format if m.signal else None,
            m.access.access,
        ]
        table.append(row)
    if fmt == "csv":
        import io

        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows([[_cell(v) if v is not None else "" for v in row] for row in table])
        return buffer.getvalue().rstrip("\n")
    if not hits:
        return "no datasets match"
    return _table(headers, table)


def _count_cell(value: int | None, display: str) -> str:
    """The winning count, with the catalogue's own wording when it says more.

    ``value`` is the resolved count (a snapshot's recomputed one when there is
    one); ``display`` is the catalogue string, which may carry a qualifier
    ("21,799 (10 s)") or disagree with the recomputation. The plain number is
    shown first and the catalogue text after it unless it is the same number.
    """
    if value is None:
        return display
    formatted = f"{value:,}"
    if display.strip() in (formatted, str(value)):
        return formatted
    return f"{formatted} (catalogue: {display})" if display.strip() else formatted


def _info_pairs(meta: DatasetMeta) -> list[tuple[str, object]]:
    disagreeing = set(meta.disagreements())

    def mark(key: str) -> str:
        return f"{key} †" if key in disagreeing else key

    pairs: list[tuple[str, object]] = [
        ("dataset_id", meta.dataset_id),
        ("aliases", meta.aliases),
        (mark("name"), meta.name),
        ("category", meta.category),
        ("status", meta.status),
        ("implementation_state", meta.implementation_state),
        ("version", meta.version),
        (mark("records"), _count_cell(meta.records, meta.records_display)),
        (mark("patients"), meta.patients_display),
        ("origin_institution", meta.origin_institution),
        ("origin_country", meta.origin_country),
        ("paper_title", meta.paper_title),
        ("paper_doi", meta.paper_doi),
        ("access", meta.access.access),
        (mark("license"), meta.access.license_text),
        ("license_url", meta.access.license_url),
        (mark("url"), meta.access.url),
        ("download_url", meta.access.download_url),
        ("publish_fold_csvs", meta.access.publish_fold_csvs),
    ]
    if meta.access.no_publish_reason:
        pairs.append(("no_publish_reason", meta.access.no_publish_reason))
    if meta.signal is not None:
        s = meta.signal
        pairs += [
            ("signal_format", s.format),
            (mark("leads"), s.leads),
            ("lead_names", s.lead_names),
            ("alternate_lead_names", s.alternate_lead_names),
            ("record_lead_layouts", [" ".join(layout) for layout in s.record_lead_layouts or ()]),
            ("sampling_rates", s.sampling_rates),
            ("default_sampling_rate", s.default_sampling_rate),
            ("duration_seconds", s.duration_seconds),
            ("units", f"{s.units} (scale {s.unit_scale})"),
            ("zero_padded_identifiers", s.zero_padded_identifiers),
        ]
    if meta.split is not None:
        pairs += [
            ("n_folds", meta.split.n_folds),
            ("predefined_column", meta.split.predefined_column),
            ("has_patient_id", meta.split.has_patient_id),
            ("record_id_column", meta.split.record_id_column),
        ]
    pairs.append(("relations", len(meta.relations)))
    return pairs


def format_info(meta: DatasetMeta, fmt: str = "table", verbose: bool = False) -> str:
    """Render one record. ``verbose`` appends every fact with its source.

    In the table form a dagger (``†``) marks a key whose sources disagree; the
    line shows the winning value and ``verbose`` shows the others.
    """
    if fmt == "json":
        data = meta.to_dict()
        if not verbose:
            data.pop("facts", None)
            data.pop("prose", None)
        return json.dumps(data, indent=2, ensure_ascii=False)
    pairs = _info_pairs(meta)
    width = max(len(k) for k, _ in pairs)
    lines = [f"{k.ljust(width)}  {_cell(v)}" for k, v in pairs if v not in (None, "", (), [])]
    if meta.description:
        lines += ["", meta.description]
    if verbose and meta.facts:
        lines += ["", "facts (most trustworthy source first):"]
        rows = []
        for key in sorted({f.key for f in meta.facts}):
            for fact in meta.facts_for(key):
                rows.append(
                    [
                        key,
                        _cell(fact.value),
                        fact.provenance.source,
                        fact.provenance.source_path,
                        fact.provenance.observed_at or "",
                    ]
                )
        lines.append(_table(["key", "value", "source", "source_path", "observed_at"], rows))
    return "\n".join(lines)


def format_fields(meta: DatasetMeta, fmt: str = "table") -> str:
    """Render a dataset's declared fields as a table, JSON, or a Frictionless Table Schema."""
    if fmt == "json":
        return json.dumps([f.__dict__ for f in meta.fields], indent=2, ensure_ascii=False)
    if fmt == "frictionless":
        from ecgbench.labels._fields import Field, to_frictionless

        fields = tuple(Field(**f.__dict__) for f in meta.fields)
        key = meta.split.record_id_column if meta.split is not None else None
        schema = to_frictionless(fields, primary_key=key)
        schema["title"] = f"{meta.name} — label table"
        return json.dumps(schema, indent=2, ensure_ascii=False)
    if not meta.fields:
        if not meta.has_labels:
            return f"{meta.dataset_id}: no labels ({meta.implementation_state})"
        return f"{meta.dataset_id}: labels available but fields not yet declared"
    headers = ["name", "type", "unit", "nullable", "vocabulary", "description"]
    rows = [
        [f.name, f.type, f.unit, f.nullable, _first_line(", ".join(f.vocabulary or ()), 40),
         _first_line(f.description, 96)]
        for f in meta.fields
    ]
    if fmt == "csv":
        import io

        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(
            [[f.name, f.type, f.unit or "", _cell(f.nullable), " ".join(f.vocabulary or ()),
              f.description] for f in meta.fields]
        )
        return buffer.getvalue().rstrip("\n")
    return _table(headers, rows)


def format_related(key: str, edges: list[RelationMeta], fmt: str = "table") -> str:
    """Render ``run_related`` output."""
    if fmt == "json":
        return json.dumps(
            {"dataset_id": key, "relations": [e.__dict__ for e in edges]},
            indent=2,
            ensure_ascii=False,
        )
    if not edges:
        return f"{key}: no declared relationships"
    headers = ["target", "relation", "shares_records", "verified", "derived", "note"]
    rows = [
        [e.target, e.relation, e.shares_records, e.verified, e.derived, _first_line(e.note)]
        for e in edges
    ]
    if fmt == "csv":
        import io

        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows([[_cell(v) for v in row] for row in rows])
        return buffer.getvalue().rstrip("\n")
    return _table(headers, rows)


def _first_line(text: str, limit: int = 90) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line if len(line) <= limit else line[: limit - 1] + "…"


# --------------------------------------------------------------------------- CLI adapters


def _fail_unknown(exc: UnknownDatasetError) -> int:
    print(f"ecgbench: {exc}", file=sys.stderr)
    return 1


def _cli_list(args: argparse.Namespace) -> int:
    rows = run_list(state=args.state, category=args.category)
    print(format_list(rows, args.format))
    return 0


def _cli_search(args: argparse.Namespace) -> int:
    try:
        hits = run_search_ranked(
            args.query,
            limit=args.limit,
            leads=args.leads,
            fs=args.fs,
            signal_format=args.signal_format,
            access=args.access,
            license=args.license,
            category=args.category,
            state=args.state,
            min_records=args.min_records,
            max_records=args.max_records,
            has_labels=args.labels,
            has_patient_id=args.patient_id,
            published=args.published,
        )
    except MetadataQueryError as exc:
        print(f"ecgbench: {exc}", file=sys.stderr)
        return 1
    print(format_search(hits, args.format))
    return 0


def _cli_info(args: argparse.Namespace) -> int:
    try:
        meta = run_info(args.dataset)
    except UnknownDatasetError as exc:
        return _fail_unknown(exc)
    print(format_info(meta, args.format, verbose=args.verbose))
    return 0


def _cli_fields(args: argparse.Namespace) -> int:
    try:
        meta = run_info(args.dataset)
    except UnknownDatasetError as exc:
        return _fail_unknown(exc)
    print(format_fields(meta, args.format))
    return 0


def _cli_related(args: argparse.Namespace) -> int:
    try:
        meta = run_info(args.dataset)
    except UnknownDatasetError as exc:
        return _fail_unknown(exc)
    print(format_related(meta.dataset_id, list(meta.relations), args.format))
    return 0


def add_subparser(subparsers) -> argparse.ArgumentParser:
    """Register ``list``, ``search``, ``info``, ``fields`` and ``related``; returns ``list``."""
    p_list = subparsers.add_parser(
        "list",
        help="List every dataset in the catalogue with its implementation state",
        description="One row per dataset, merged from the catalogue and the configs.",
    )
    p_list.add_argument(
        "--state",
        choices=IMPLEMENTATION_STATES,
        default=None,
        help="Keep only datasets in this implementation state",
    )
    p_list.add_argument("--category", default=None, help="Keep only this catalogue category")
    p_list.add_argument("--format", choices=_FORMATS, default="table", help="Output format")
    p_list.set_defaults(func=_cli_list)

    p_search = subparsers.add_parser(
        "search",
        help="Ranked full-text search over every dataset, with structured filters",
        description=(
            "QUERY is FTS5 syntax passed through verbatim: words are ANDed, `fib*` is a "
            'prefix, "..." is a phrase, NOT/OR/AND combine terms. Omit QUERY to filter '
            "only. Results are ranked by bm25 with the name weighted highest."
        ),
    )
    p_search.add_argument("query", nargs="?", default=None, help="FTS5 query (optional)")
    p_search.add_argument("--leads", type=int, default=None, help="Exact lead count")
    p_search.add_argument("--fs", type=int, default=None, help="A sampling rate the release ships")
    p_search.add_argument(
        "--signal-format", default=None, help="Signal format (wfdb, csv, edf, mat, hdf5, …)"
    )
    p_search.add_argument(
        "--access", default=None, choices=("open", "credentialed", "restricted")
    )
    p_search.add_argument("--license", default=None, help="Substring of the licence name or URL")
    p_search.add_argument("--category", default=None, help="Exact catalogue category")
    p_search.add_argument("--state", default=None, choices=IMPLEMENTATION_STATES)
    p_search.add_argument("--min-records", type=int, default=None)
    p_search.add_argument("--max-records", type=int, default=None)
    p_search.add_argument(
        "--labels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Only datasets with (or, --no-labels, without) a label loader",
    )
    p_search.add_argument(
        "--patient-id",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Only datasets whose folds are (or are not) patient-grouped",
    )
    p_search.add_argument(
        "--published",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Only datasets whose fold CSVs are (or are not) on the Hub",
    )
    p_search.add_argument("--limit", type=int, default=None, help="Keep at most N results")
    p_search.add_argument("--format", choices=_FORMATS, default="table")
    p_search.set_defaults(func=_cli_search)

    p_info = subparsers.add_parser(
        "info",
        help="Show one dataset's merged metadata",
        description=(
            "Accepts a catalogue slug (ptb-xl), a config slug (ptbxl) or the display "
            "name. A dagger marks a value on which the sources disagree; --verbose "
            "lists every fact with its source."
        ),
    )
    p_info.add_argument("dataset", help="Dataset id or any alias")
    p_info.add_argument("--verbose", action="store_true", help="Print every fact with provenance")
    p_info.add_argument("--format", choices=("table", "json"), default="table")
    p_info.set_defaults(func=_cli_info)

    p_fields = subparsers.add_parser(
        "fields",
        help="List a dataset's label columns: name, type, unit, vocabulary, description",
        description=(
            "The columns load_labels() returns for this dataset, as declared in its label "
            "module's FIELDS or the config's labels.fields block. --format frictionless emits "
            "a Frictionless Table Schema."
        ),
    )
    p_fields.add_argument("dataset", help="Dataset id or any alias")
    p_fields.add_argument(
        "--format", choices=("table", "json", "csv", "frictionless"), default="table"
    )
    p_fields.set_defaults(func=_cli_fields)

    p_related = subparsers.add_parser(
        "related",
        help="Show a dataset's relationships to other datasets (the leakage graph)",
        description="Declared and derived edges, with the shares_records leakage flag.",
    )
    p_related.add_argument("dataset", help="Dataset id or any alias")
    p_related.add_argument("--format", choices=_FORMATS, default="table")
    p_related.set_defaults(func=_cli_related)

    return p_list
