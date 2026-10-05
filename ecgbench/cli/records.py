"""``ecgbench records <dataset> [--data-path P] [--split S] [--sql Q] [--format F]``.

Prints a dataset's records — its fold table joined with its labels — or the
result of a DuckDB query over them. Fold CSVs come from the Hub by default, or
from a local ``output/<slug>/`` tree with ``--splits-dir``; labels always come
from the local source dataset under ``--data-path``. ``--hub`` queries a
published ``folds.csv`` in place on the Hub without labels.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ecgbench.cli.catalog import _table
from ecgbench.metadata.identity import UnknownDatasetError

logger = logging.getLogger(__name__)

_FORMATS = ("table", "json", "csv", "parquet")
#: Rows a table shows unless ``--limit`` says otherwise; the other formats print all.
TABLE_LIMIT = 20


def run_records(
    dataset: str,
    data_path: Path | str | None = None,
    version: str = "clean",
    split: str | None = None,
    fold_numbers: list[int] | None = None,
    source: str | None = None,
    splits_dir: Path | str | None = None,
    labels: bool = True,
    sql: str | None = None,
    hub: bool = False,
):
    """Load a dataset's records and optionally run SQL over them.

    Args:
        dataset: Dataset id or any alias.
        data_path: Local copy of the source dataset (labels live there).
        version: ``"clean"`` or ``"original"``.
        split: ``"train"``, ``"val"``, ``"test"`` or None for all.
        fold_numbers: Restrict to these folds.
        source: ``"hf"`` or ``"local"``; defaults to ``"local"`` when
            ``splits_dir`` is given, else ``"hf"``.
        splits_dir: Fold tree for ``source="local"`` (``output/<slug>/``).
        labels: Join the label table.
        sql: DuckDB SQL over the view ``records``.
        hub: Query the published master table on the Hub directly through
            DuckDB's httpfs (identifiers only, no labels); ``sql`` defaults to
            ``SELECT * FROM records``.

    Returns:
        A pandas DataFrame.

    Raises:
        See :func:`ecgbench.metadata.records.load_records` and
        :func:`ecgbench.metadata.records.query_records`.
    """
    from ecgbench.metadata import records as rec

    if hub:
        return rec.query_hub(dataset, sql or f"SELECT * FROM {rec.SQL_VIEW}", version=version)
    if source is None:
        source = "local" if splits_dir is not None else "hf"
    df = rec.load_records(
        dataset,
        data_path=data_path,
        version=version,
        split=split,
        fold_numbers=fold_numbers,
        source=source,
        splits_dir=splits_dir,
        labels=labels,
    )
    return rec.query_records(df, sql) if sql else df


def format_records(df, fmt: str = "table", limit: int | None = None) -> str:
    """Render a DataFrame as a table, JSON records, or CSV text."""
    if fmt == "json":
        return df.to_json(orient="records", indent=2, date_format="iso", force_ascii=False)
    if fmt == "csv":
        return df.to_csv(index=False).rstrip("\n")
    if fmt != "table":
        raise ValueError(f"format must be one of {_FORMATS}, got {fmt!r}")
    shown = df if limit is None or limit <= 0 else df.head(limit)
    rows = [[None if _is_missing(v) else v for v in row] for row in shown.itertuples(index=False)]
    text = _table([str(c) for c in df.columns], rows)
    if len(shown) < len(df):
        text += f"\n({len(shown)} of {len(df)} rows; --limit 0 shows all)"
    elif not len(df):
        text += "\n(no rows)"
    return text


def _is_missing(value) -> bool:
    try:
        return value != value  # NaN is the only value unequal to itself
    except (TypeError, ValueError):
        return False


def _cli_run(args: argparse.Namespace) -> int:
    from ecgbench.labels import LabelsUnavailableError
    from ecgbench.metadata.records import (
        RecordsQueryError,
        RecordsUnavailableError,
        SplitsNotPublishedError,
    )

    if args.format == "parquet" and not args.output:
        print("ecgbench: --format parquet needs --output PATH", file=sys.stderr)
        return 2
    try:
        df = run_records(
            args.dataset,
            data_path=args.data_path,
            version=args.version,
            split=args.split,
            fold_numbers=args.fold,
            source=args.source,
            splits_dir=args.splits_dir,
            labels=not args.no_labels,
            sql=args.sql,
            hub=args.hub,
        )
    except UnknownDatasetError as exc:
        print(f"ecgbench: {exc}", file=sys.stderr)
        return 1
    except LabelsUnavailableError as exc:
        print(f"ecgbench: {exc}\nPass --no-labels for the fold table alone.", file=sys.stderr)
        return 1
    except (
        RecordsUnavailableError,
        SplitsNotPublishedError,
        RecordsQueryError,
        FileNotFoundError,
        ImportError,
        ValueError,
    ) as exc:
        print(f"ecgbench: {exc}", file=sys.stderr)
        return 1

    if args.format == "parquet":
        from ecgbench.metadata.records import write_parquet

        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            write_parquet(df, target)
        except ImportError as exc:
            print(f"ecgbench: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {len(df)} rows to {target}")
        return 0

    limit = args.limit if args.limit is not None else (TABLE_LIMIT if args.format == "table" else 0)
    if args.format == "table":
        text = format_records(df, "table", limit)
    else:
        text = format_records(df.head(limit) if limit > 0 else df, args.format)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {len(df)} rows to {target}")
    else:
        print(text)
    return 0


def add_subparser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "records",
        help="Print a dataset's records (fold table + labels), or run SQL over them",
        description=(
            "One row per record: the fold CSVs ECGBench published (or a local output/<slug>/ "
            "tree) joined with the labels read from the source dataset under --data-path. "
            "--sql runs DuckDB over the view `records` with file access disabled."
        ),
        epilog=(
            "examples:\n"
            "  ecgbench records ptbxl --data-path /data/ptb-xl --split val --format csv\n"
            "  ecgbench records ptbxl --data-path /data/ptb-xl \\\n"
            '      --sql "select sex, count(*) n from records where fold = 9 group by 1"\n'
            "  ecgbench records mimic_iv_ecg --splits-dir output/mimic_iv_ecg --no-labels\n"
            '  ecgbench records ptbxl --hub --sql "select fold, count(*) from records group by 1"'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("dataset", help="Dataset id or any alias")
    p.add_argument("--data-path", default=None, help="Local copy of the source dataset")
    p.add_argument(
        "--splits-dir",
        default=None,
        help="Local fold tree (output/<slug>/); implies --source local",
    )
    p.add_argument(
        "--source",
        choices=("hf", "local"),
        default=None,
        help="Where the fold CSVs come from (default: local with --splits-dir, else hf)",
    )
    p.add_argument("--version", choices=("clean", "original"), default="clean")
    p.add_argument("--split", choices=("train", "val", "test"), default=None)
    p.add_argument(
        "--fold", type=int, nargs="+", default=None, metavar="N", help="Restrict to these folds"
    )
    p.add_argument("--no-labels", action="store_true", help="Fold table only, no label join")
    p.add_argument("--sql", default=None, metavar="QUERY", help="DuckDB SQL over `records`")
    p.add_argument(
        "--hub",
        action="store_true",
        help="Query the published folds.csv on the Hub in place (DuckDB httpfs; no labels)",
    )
    p.add_argument("--format", choices=_FORMATS, default="table", help="Output format")
    p.add_argument("--output", default=None, metavar="PATH", help="Write here instead of stdout")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=f"Rows to print (table default {TABLE_LIMIT}; 0 = all)",
    )
    p.set_defaults(func=_cli_run)
    return p
