"""Record-level access: a dataset's fold table joined with its labels, queryable with SQL.

The metadata layer describes datasets; this module is the one place that
returns *records*. ``load_records`` reads the fold CSVs ``ecgbench splits``
wrote (from the Hub or a local tree), joins ``load_labels()`` on the record id,
and returns one pandas DataFrame per request. ``query_records`` runs DuckDB SQL
over that frame with file-system access disabled, so a query can only ever see
the view it was given.

The fold-table readers here are also what ``ECGDataset`` uses: its Hub and
local paths were factored out so the two cannot disagree about the Hub layout,
the identifier dtypes or the fold-selection rules. Nothing in this module is
imported by ``ecgbench.metadata`` itself, because it needs pandas.

Hub layout, mirrored from ``output/<slug>/``::

    <slug>/<version>/folds.csv                 # every record + fold + default_split
    <slug>/<version>/<split>/fold_<N>.csv      # N is 1-indexed

Nothing per-record is bundled in the wheel or published for a dataset whose
``publish_fold_csvs`` is false; those raise ``SplitsNotPublishedError`` before
any network call and are reproduced locally with ``ecgbench splits``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

#: The public HuggingFace dataset repo holding the published fold CSVs.
HF_REPO_ID = "vlbthambawita/ECGBench"
VERSIONS = ("original", "clean")
SPLITS = ("train", "val", "test")
FOLD_SOURCES = ("hf", "local")
#: The name the fold table is registered under for ``--sql``.
SQL_VIEW = "records"


class SplitsNotPublishedError(RuntimeError):
    """The dataset's splits are deliberately not on the Hub.

    Raised instead of a bare 404 for credentialed or restricted sources, whose
    identifiers ECGBench will not republish. The message carries the command
    that regenerates the identical split locally.
    """


class RecordsUnavailableError(RuntimeError):
    """The dataset has no fold table at all: it is catalogue-only (no config)."""


class RecordsQueryError(ValueError):
    """DuckDB rejected the SQL, or it tried to reach outside the records view."""


# --------------------------------------------------------------------------- fold tables


def not_published_error(config: DatasetConfig) -> SplitsNotPublishedError:
    """The error for a withheld dataset, quoting the config's regeneration recipe."""
    return SplitsNotPublishedError(
        f"ECGBench does not publish fold CSVs for '{config.slug}'.\n"
        f"{config.no_publish_reason.strip()}\n"
        'Then load with metadata_source="local", pointing data_path at the '
        "directory holding the generated original/ and clean/ trees."
    )


def read_fold_csv(path: str | Path, config: DatasetConfig) -> pd.DataFrame:
    """Read one fold CSV, keeping identifier columns as strings.

    The one place every fold-CSV read goes through, so the Hub and local paths
    cannot disagree about a record's id. Without the dtype, pandas turns a
    zero-padded record id such as ``afdb``'s ``00735`` into 735 and a signal
    read then looks for a record named "735".
    """
    return pd.read_csv(path, dtype=config.identifier_dtypes())


def filter_fold_table(
    df: pd.DataFrame, split: str | None, fold_numbers: list[int] | None
) -> pd.DataFrame:
    """Filter a master ``folds.csv`` by split and/or fold.

    With ``split=None`` the ``default_split`` filter is skipped, which is what
    makes cross-split fold selection (custom cross-validation) possible.

    Raises:
        ValueError: a requested fold holds no record (in that split).
    """
    if split is not None:
        df = df[df["default_split"] == split]
    if fold_numbers is not None:
        known = {int(n) for n in df["fold"].unique()}
        unknown = [n for n in fold_numbers if int(n) not in known]
        if unknown:
            raise ValueError(
                f"Fold(s) {unknown} hold no records"
                + (f" in split '{split}'" if split else "")
                + f". Available: {sorted(known)}."
            )
        df = df[df["fold"].isin(fold_numbers)]
    return df.reset_index(drop=True)


def read_split_folds(
    split_dir: Path, split: str, fold_numbers: list[int] | None, config: DatasetConfig
) -> pd.DataFrame:
    """Concatenate ``fold_<N>.csv`` files from one split's directory.

    Raises:
        FileNotFoundError: a requested fold is not in this split, naming the
            folds it does hold and the ``split=None`` way across splits.
    """
    if fold_numbers is not None:
        files = [split_dir / f"fold_{n}.csv" for n in fold_numbers]
        missing = [f for f in files if not f.exists()]
        if missing:
            present = sorted(int(p.stem.split("_")[1]) for p in split_dir.glob("fold_*.csv"))
            raise FileNotFoundError(
                f"Fold(s) {[int(f.stem.split('_')[1]) for f in missing]} are not in "
                f"split '{split}' (it holds folds {present}). Each fold belongs "
                "to exactly one split, so to take folds across split boundaries — "
                "for custom cross-validation — pass split=None with fold_numbers."
            )
    else:
        files = sorted(split_dir.glob("fold_*.csv"))
        if not files:
            raise FileNotFoundError(f"No fold_*.csv files in {split_dir}")
    return pd.concat([read_fold_csv(f, config) for f in files], ignore_index=True)


def hub_fold_paths(
    config: DatasetConfig, version: str, split: str | None, fold_numbers: list[int] | None
) -> list[str]:
    """Repo-relative paths a request needs: the master table, or one file per fold.

    ``split=None`` means "by fold, ignoring the default split", which only the
    master ``folds.csv`` can answer; so does an unfiltered request.
    """
    if fold_numbers is None or split is None:
        return [f"{config.slug}/{version}/folds.csv"]
    return [f"{config.slug}/{version}/{split}/fold_{n}.csv" for n in fold_numbers]


def fetch_hub_fold_table(
    config: DatasetConfig,
    version: str = "clean",
    split: str | None = None,
    fold_numbers: list[int] | None = None,
    repo_id: str = HF_REPO_ID,
) -> pd.DataFrame:
    """Download the fold CSVs a request needs from the Hub and return them as one table.

    Raises:
        SplitsNotPublishedError: the config withholds its fold CSVs (checked
            before any network call).
        ImportError: ``huggingface_hub`` is not installed.
        ValueError: a requested fold holds no record.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for HF metadata. Install with: pip install ecgbench[hf]"
        )

    if not config.publish_fold_csvs:
        raise not_published_error(config)

    paths = hub_fold_paths(config, version, split, fold_numbers)
    frames = [
        read_fold_csv(hf_hub_download(repo_id=repo_id, filename=p, repo_type="dataset"), config)
        for p in paths
    ]
    if fold_numbers is None or split is None:
        return filter_fold_table(frames[0], split, fold_numbers)
    return pd.concat(frames, ignore_index=True)


def load_local_fold_table(
    splits_dir: Path,
    config: DatasetConfig,
    version: str = "clean",
    split: str | None = None,
    fold_numbers: list[int] | None = None,
) -> pd.DataFrame:
    """Read the fold CSVs from a local tree.

    ``splits_dir`` is probed as ``<dir>/<version>/<split>/``, ``<dir>/<split>/``,
    then ``<dir>/<version>/folds.csv`` and ``<dir>/folds.csv`` — so it may be
    ``output/<slug>/`` as ``ecgbench splits`` wrote it, or a dataset directory
    the fold tree was copied into.

    Raises:
        FileNotFoundError: no fold CSVs under ``splits_dir``.
        ValueError: a requested fold holds no record.
    """
    splits_dir = Path(splits_dir)
    # Per-split fold files are the fast path, but they cannot answer
    # split=None — fold N lives in exactly one split's directory.
    if split is not None:
        for candidate in (splits_dir / version / split, splits_dir / split):
            if candidate.exists():
                return read_split_folds(candidate, split, fold_numbers, config)
    for candidate in (splits_dir / version / "folds.csv", splits_dir / "folds.csv"):
        if candidate.exists():
            return filter_fold_table(read_fold_csv(candidate, config), split, fold_numbers)
    raise FileNotFoundError(
        f"Could not find fold CSVs for split '{split}' in {splits_dir}. "
        "Run the split pipeline first or use metadata_source='hf'."
    )


def load_fold_table(
    config: DatasetConfig,
    version: str = "clean",
    split: str | None = None,
    fold_numbers: list[int] | None = None,
    source: str = "hf",
    splits_dir: Path | str | None = None,
    repo_id: str = HF_REPO_ID,
) -> pd.DataFrame:
    """The fold table for a request, from the Hub (``source="hf"``) or ``splits_dir``.

    Raises:
        ValueError: unknown ``source`` or ``version``, or a fold holding no record.
    """
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}, got {version!r}")
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS} or None, got {split!r}")
    if source == "hf":
        return fetch_hub_fold_table(config, version, split, fold_numbers, repo_id)
    if source == "local":
        if splits_dir is None:
            raise ValueError("source='local' needs splits_dir (output/<slug>/ or the data path)")
        return load_local_fold_table(splits_dir, config, version, split, fold_numbers)
    raise ValueError(f"source must be one of {FOLD_SOURCES}, got {source!r}")


# --------------------------------------------------------------------------- records


def resolve_config(dataset: str | DatasetConfig) -> DatasetConfig:
    """The config behind any dataset id, alias or display name.

    Raises:
        UnknownDatasetError: no dataset by that name (the message names close matches).
        RecordsUnavailableError: the dataset is catalogue-only.
    """
    from ecgbench.config import DatasetConfig, load_config

    if isinstance(dataset, DatasetConfig):
        return dataset
    from ecgbench.metadata import get

    meta = get(dataset)
    if not meta.has_config:
        raise RecordsUnavailableError(
            f"'{meta.name}' ({meta.dataset_id}) is catalogue-only: it has no config, so "
            "ECGBench holds no fold table or labels for it."
        )
    return load_config(meta.dataset_id)


def join_labels(
    fold_table: pd.DataFrame, labels: pd.DataFrame, config: DatasetConfig
) -> pd.DataFrame:
    """Left-join a label table (indexed by record id) onto a fold table.

    Fold-table columns win: a label column with the same name (``patient_id``,
    say) is dropped rather than suffixed, because the fold CSV already holds
    it for every record in the partition. Row order is the fold table's.

    Raises:
        ValueError: no record matched a label row at all, which means the two
            tables key on different identifiers.
    """
    record_ids = fold_table[config.record_id_column]
    if labels.index.dtype != record_ids.dtype:
        # Fold CSVs and source CSVs can disagree on int vs str for the same
        # IDs, which would silently join to all-NaN.
        labels = labels.set_index(labels.index.astype(str))
        record_ids = record_ids.astype(str)
    overlap = [c for c in labels.columns if c in fold_table.columns]
    aligned = labels.drop(columns=overlap).reindex(record_ids)
    missing = int(aligned.isna().all(axis=1).sum()) if len(aligned.columns) else 0
    if len(aligned) and missing == len(aligned):
        spec = config.labels
        raise ValueError(
            "No record in the fold table matched a label row. Check that "
            f"'{spec.join_column if spec else '?'}' in {spec.source_csv if spec else '?'} "
            f"holds the same IDs as '{config.record_id_column}' in the fold CSVs."
        )
    if missing:
        logger.warning("%d of %d records have no label row", missing, len(aligned))
    aligned.index = fold_table.index
    return pd.concat([fold_table, aligned], axis=1)


def load_records(
    dataset: str | DatasetConfig,
    data_path: Path | str | None = None,
    version: str = "clean",
    split: str | None = None,
    fold_numbers: list[int] | None = None,
    source: str = "hf",
    splits_dir: Path | str | None = None,
    labels: bool = True,
) -> pd.DataFrame:
    """One row per record: the fold table joined with the dataset's labels.

    Args:
        dataset: Config slug, catalogue slug, display name, or a ``DatasetConfig``.
        data_path: Root of a local copy of the source dataset, where the labels
            live. Resolved through ``resolve_data_path`` (which may download)
            when omitted and labels are requested.
        version: ``"clean"`` (valid records only) or ``"original"``.
        split: ``"train"``, ``"val"``, ``"test"``, or None for every split.
        fold_numbers: Restrict to these folds. With a named ``split`` they must
            belong to it; with ``split=None`` they may cross split boundaries.
        source: ``"hf"`` downloads the fold CSVs from the Hub; ``"local"`` reads
            them from ``splits_dir``.
        splits_dir: The fold tree for ``source="local"`` — ``output/<slug>/`` or
            a data directory holding ``{clean,original}/``. Defaults to ``data_path``.
        labels: Join ``load_labels()``; ``False`` returns the fold table alone.

    Returns:
        The fold table's columns (record id, patient id, signal paths, ``fold``,
        ``default_split``, and ``is_valid``/``quality_issues`` in ``original``)
        followed by the label columns.

    Raises:
        UnknownDatasetError: no such dataset.
        RecordsUnavailableError: catalogue-only dataset.
        SplitsNotPublishedError: ``source="hf"`` for a withheld dataset.
        LabelsUnavailableError: ``labels=True`` for a dataset that ships none.
        FileNotFoundError: no fold CSVs under ``splits_dir``, or the label source
            is missing.
        ValueError: bad ``version``/``split``/``source``, a fold holding no
            record, or a label join that matches nothing.
    """
    config = resolve_config(dataset)
    if source == "local" and splits_dir is None:
        splits_dir = data_path
    table = load_fold_table(config, version, split, fold_numbers, source, splits_dir)
    if not labels:
        return table
    from ecgbench.labels import load_labels

    return join_labels(table, load_labels(config, data_path), config)


# --------------------------------------------------------------------------- SQL


def _sql_quote(text: str) -> str:
    """A single-quoted SQL string literal."""
    return "'" + text.replace("'", "''") + "'"


def _duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError(
            "duckdb is required for SQL over records: pip install ecgbench[analytics]"
        ) from exc
    return duckdb


def query_records(df: pd.DataFrame, sql: str, view: str = SQL_VIEW) -> pd.DataFrame:
    """Run DuckDB SQL over a DataFrame registered as the view ``records``.

    The connection is in-memory with external access disabled and the
    configuration locked, so the query can read the view and nothing else: no
    files, no URLs, no extension loads.

    Raises:
        ImportError: ``duckdb`` is not installed (``pip install ecgbench[analytics]``).
        RecordsQueryError: the SQL failed, including any attempt to reach a file.
    """
    duckdb = _duckdb()
    con = duckdb.connect(":memory:")
    try:
        con.execute("SET enable_external_access = false")
        con.execute("SET lock_configuration = true")
        con.register(view, df)
        try:
            return con.execute(sql).df()
        except duckdb.Error as exc:
            raise RecordsQueryError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        con.close()


def hub_fold_url(config: DatasetConfig, version: str = "clean", repo_id: str = HF_REPO_ID) -> str:
    """The ``hf://`` URL DuckDB's httpfs extension reads a published master table from."""
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}, got {version!r}")
    if not config.publish_fold_csvs:
        raise not_published_error(config)
    return f"hf://datasets/{repo_id}/{config.slug}/{version}/folds.csv"


def hub_view_sql(config: DatasetConfig, url: str, view: str = SQL_VIEW) -> str:
    """The ``CREATE VIEW`` statement over a remote fold CSV.

    Identifier columns are declared ``VARCHAR`` for a config with
    ``zero_padded_identifiers`` so afdb's ``00735`` survives, exactly as
    :func:`read_fold_csv` keeps them; every other config declares nothing, and
    DuckDB infers the rest.
    """
    options = ["header = true"]
    columns = sorted(config.identifier_dtypes())
    if columns:
        types = ", ".join(f"{_sql_quote(col)}: 'VARCHAR'" for col in columns)
        options.append(f"types = {{{types}}}")
    return (
        f"CREATE VIEW {view} AS SELECT * FROM read_csv({_sql_quote(url)}, {', '.join(options)})"
    )


def query_hub(
    dataset: str | DatasetConfig,
    sql: str = f"SELECT * FROM {SQL_VIEW}",
    version: str = "clean",
    repo_id: str = HF_REPO_ID,
) -> pd.DataFrame:
    """Run SQL directly over a published ``folds.csv`` on the Hub, without downloading it first.

    DuckDB's ``httpfs`` extension streams the CSV (identifiers only — labels
    never leave the source dataset, so none are joined here). The extension is
    installed on first use, which needs network access.

    Raises:
        SplitsNotPublishedError: the dataset's fold CSVs are withheld.
        ImportError: ``duckdb`` is not installed.
        RecordsQueryError: the SQL failed or the extension could not be loaded.
    """
    config = resolve_config(dataset)
    url = hub_fold_url(config, version, repo_id)
    duckdb = _duckdb()
    con = duckdb.connect(":memory:")
    try:
        try:
            con.execute("INSTALL httpfs")
            con.execute("LOAD httpfs")
            con.execute(hub_view_sql(config, url))
            return con.execute(sql).df()
        except duckdb.Error as exc:
            raise RecordsQueryError(f"{type(exc).__name__}: {exc}") from exc
    finally:
        con.close()


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame as Parquet through DuckDB, or pandas when DuckDB is absent.

    Raises:
        ImportError: neither ``duckdb`` nor a pandas Parquet engine is installed.
    """
    try:
        duckdb = _duckdb()
    except ImportError:
        try:
            df.to_parquet(path, index=False)
        except ImportError as exc:
            raise ImportError(
                "Parquet output needs duckdb (pip install ecgbench[analytics]) or pyarrow"
            ) from exc
        return
    con = duckdb.connect(":memory:")
    try:
        con.register("result", df)
        con.execute(f"COPY result TO {_sql_quote(str(path))} (FORMAT PARQUET)")
    finally:
        con.close()
