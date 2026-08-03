"""
PTB-XL+ — derived features and annotations **for PTB-XL's records**.

PTB-XL+ ships no raw ECGs. It is a companion release that annotates the *same*
21,799 recordings PTB-XL holds, keyed by PTB-XL's own ``ecg_id``:

- ``labels/ptbxl_statements.csv`` — PTB-XL's SCP statements with SNOMED extensions
- ``labels/12sl_statements.csv`` — statements from the Marquette 12SL algorithm
- ``labels/snomed_description.csv`` — the SNOMED vocabulary both map into
- ``features/{unig,ecgdeli,12sl}_features.csv`` — 749 / 532 / 783 measured features
  from three independent providers
- ``median_beats/{12sl,unig}/`` — derived single-beat waveforms (see the caveats
  below; ECGBench does not load these)
- ``fiducial_points/ecgdeli/`` — 283,326 per-lead WFDB annotation files

**There is deliberately no ``ptbxl_plus`` dataset config.** Because every row is a
PTB-XL record, generating a separate ten-fold split would create a second
partition over the same recordings that ``ptbxl`` already partitions, and a user
who trained on one and evaluated on the other would leak. So this module is a
*label provider*: you load PTB-XL as usual, on PTB-XL's official folds, and join
these columns onto it. :func:`load_ptbxl_plus` returns a frame indexed by
``ecg_id`` for exactly that.

Four defects in the shipped release, all confirmed against its own
``SHA256SUMS.txt`` so they are upstream rather than download damage:

1. **``12sl_features.csv`` hides its key column in the middle of the table.**
   ``ecg_id`` is column **145 of 783**, between ``QRS_Area_aVF`` and
   ``P_On_Global``, so inspecting the first or last few columns suggests the table
   has no key at all. Always locate the key by name; never by position.
2. **Neither 12SL table is sorted by ``ecg_id``.** Both run
   ``1, 21803, 21804, 21805, 21806, …`` — identical order in the two files, but not
   ascending. So joining by row position, or assuming sorted ids, attaches values
   to the wrong records. Use the key column.
3. **Every ``median_beats/12sl/*.hea`` is unreadable by ``wfdb.rdrecord``**: the
   record line carries a stale producer-side prefix
   (``ge_median_beats_wfdb/00001_medians``) and wfdb rejects a ``/`` there.
4. **unig median-beat amplitudes are about 1000x too large** — they span roughly
   −1361 to +602 against a declared ``/mV`` gain, i.e. the values are effectively
   microvolts. Coverage is also incomplete for both providers (20,914 and 21,794
   of 21,799).

Because of 3 and 4 the median beats are exposed only as *paths* via
:func:`median_beat_path`, with no decoding — ECGBench will not present a signal it
cannot state the units of.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

#: Join key throughout: PTB-XL's own record identifier.
JOIN_COLUMN = "ecg_id"

#: Per-record tables, relative to the PTB-XL+ root.
STATEMENTS = {
    "ptbxl": "labels/ptbxl_statements.csv",
    "12sl": "labels/12sl_statements.csv",
}
FEATURES = {
    "unig": "features/unig_features.csv",
    "ecgdeli": "features/ecgdeli_features.csv",
    "12sl": "features/12sl_features.csv",
}

#: Vocabulary and dictionary tables (not per record).
SNOMED_DESCRIPTION = "labels/snomed_description.csv"
FEATURE_DESCRIPTION = "features/feature_description.csv"

#: The provider whose feature table buries ``ecg_id`` mid-table (column 145 of
#: 783) rather than putting it first. The column IS present in v1.0.1, so the
#: loader uses it; the row-order fallback below exists only in case a future
#: release drops it, since these rows are aligned to the statements file.
BURIED_KEY_FEATURES = "12sl"

#: Median-beat directories, and the zero-padding each provider uses for the
#: record stem. The two differ, which is why this is a table and not a format
#: string: 12sl writes 00001_medians, unig writes 000001_medians.
MEDIAN_BEAT_PROVIDERS = {"12sl": 5, "unig": 6}

#: Columns holding Python-literal lists or tuples that are worth parsing.
_LITERAL_COLUMNS = (
    "statements",
    "statements_cat",
    "statements_ext",
    "statements_ext_snomed",
    "scp_codes",
    "scp_codes_ext",
    "scp_codes_ext_snomed",
)


def _require(path: Path, what: str) -> None:
    from ecgbench.labels import LabelSourceMissingError

    if not path.exists():
        raise LabelSourceMissingError(
            f"PTB-XL+ {what} not found at {path}. Point data_path at the PTB-XL+ root "
            "— the directory holding labels/, features/ and median_beats/ — from "
            "https://physionet.org/content/ptb-xl-plus/1.0.1/ ."
        )


def _parse_literals(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the ``"['NSR', 'NML']"`` string columns into real Python objects."""
    out = df.copy()
    for column in out.columns:
        if column not in _LITERAL_COLUMNS:
            continue
        out[column] = out[column].map(_safe_literal)
    return out


def _safe_literal(value: object) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[({":
        return value
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return value


def load_statements(data_path: Path | str, provider: str = "ptbxl") -> pd.DataFrame:
    """Load one statements table, indexed by ``ecg_id``.

    ``provider`` is ``"ptbxl"`` (SCP statements as PTB-XL assigns them, plus SNOMED
    extensions) or ``"12sl"`` (the Marquette 12SL algorithm's own statements).
    """
    if provider not in STATEMENTS:
        raise ValueError(f"provider must be one of {sorted(STATEMENTS)}, got {provider!r}")
    path = Path(data_path) / STATEMENTS[provider]
    _require(path, f"{provider} statements")

    df = pd.read_csv(path)
    df = _parse_literals(df).set_index(JOIN_COLUMN)
    logger.info("Loaded %d %s statements from %s", len(df), provider, path.name)
    return df


def load_features(data_path: Path | str, provider: str = "unig") -> pd.DataFrame:
    """Load one feature table, indexed by ``ecg_id``.

    ``provider`` is ``"unig"`` (University of Glasgow), ``"ecgdeli"`` (the KIT
    ECGdeli toolbox) or ``"12sl"`` (Marquette 12SL).

    The ``12sl`` table keeps ``ecg_id`` at **column 145 of 783**, not at the front,
    which is easy to miss when eyeballing the header — and neither 12SL table is
    sorted by ``ecg_id``. This function keys on the column by name, so both traps
    are handled. Should a future release drop the column entirely, it falls back to
    the row order of ``12sl_statements.csv`` (the two are aligned in v1.0.1) and
    refuses to guess if the row counts disagree.
    """
    if provider not in FEATURES:
        raise ValueError(f"provider must be one of {sorted(FEATURES)}, got {provider!r}")
    path = Path(data_path) / FEATURES[provider]
    _require(path, f"{provider} features")

    df = pd.read_csv(path, low_memory=False)

    if provider == BURIED_KEY_FEATURES:
        if JOIN_COLUMN in df.columns:
            # The normal path in v1.0.1: the column exists, just not first.
            position = list(df.columns).index(JOIN_COLUMN) + 1
            logger.debug(
                "%s carries %s at column %d of %d",
                path.name,
                JOIN_COLUMN,
                position,
                len(df.columns),
            )
        else:
            statements_path = Path(data_path) / STATEMENTS[BURIED_KEY_FEATURES]
            _require(statements_path, "12sl statements (needed to key 12sl features)")
            keys = pd.read_csv(statements_path, usecols=[JOIN_COLUMN])[JOIN_COLUMN]
            if len(keys) != len(df):
                raise ValueError(
                    f"{path.name} has {len(df)} rows but "
                    f"{statements_path.name} has {len(keys)}; the two are supposed to be "
                    "row-aligned, so refusing to guess the key. Check both files against "
                    "the release's SHA256SUMS.txt."
                )
            logger.info(
                "%s ships no %s column; keying it from %s in file order "
                "(that file is not sorted by id, so ascending order would be wrong)",
                path.name,
                JOIN_COLUMN,
                statements_path.name,
            )
            df[JOIN_COLUMN] = keys.values

    df = df.set_index(JOIN_COLUMN)
    logger.info("Loaded %d x %d %s features", len(df), df.shape[1], provider)
    return df


def load_snomed_description(data_path: Path | str) -> pd.DataFrame:
    """The SNOMED vocabulary both statement sets map into, indexed by ``snomed_id``."""
    path = Path(data_path) / SNOMED_DESCRIPTION
    _require(path, "SNOMED description")
    return pd.read_csv(path).set_index("snomed_id")


def load_feature_description(data_path: Path | str) -> pd.DataFrame:
    """The cross-provider feature dictionary: which columns mean the same thing."""
    path = Path(data_path) / FEATURE_DESCRIPTION
    _require(path, "feature description")
    # The shipped file has a UTF-8 BOM on its first column name.
    return pd.read_csv(path, encoding="utf-8-sig")


def median_beat_path(data_path: Path | str, ecg_id: int, provider: str = "unig") -> Path | None:
    """Return the WFDB record stem of one derived median beat, or None if absent.

    No decoding is offered, deliberately. The ``12sl`` headers are unreadable by
    ``wfdb.rdrecord`` — their record line carries a stale ``ge_median_beats_wfdb/``
    prefix and wfdb rejects the ``/`` — and the ``unig`` amplitudes are about
    1000x larger than their declared ``/mV`` gain implies, so ECGBench will not
    present them as millivolt signals. Coverage is also partial: 20,914 (12sl) and
    21,794 (unig) of PTB-XL's 21,799 records.
    """
    if provider not in MEDIAN_BEAT_PROVIDERS:
        raise ValueError(
            f"provider must be one of {sorted(MEDIAN_BEAT_PROVIDERS)}, got {provider!r}"
        )
    width = MEDIAN_BEAT_PROVIDERS[provider]
    stem = f"{int(ecg_id):0{width}d}_medians"
    group = f"{(int(ecg_id) // 1000) * 1000:05d}"
    path = Path(data_path) / "median_beats" / provider / group / stem
    return path if path.with_suffix(".hea").exists() else None


def load_ptbxl_plus(
    data_path: Path | str,
    statements: tuple[str, ...] = ("ptbxl", "12sl"),
    features: tuple[str, ...] = (),
    prefix: bool = True,
) -> pd.DataFrame:
    """Return PTB-XL+ annotations indexed by ``ecg_id``, ready to join onto PTB-XL.

    Args:
        data_path: the PTB-XL+ root.
        statements: which statement tables to include.
        features: which feature tables to include. Empty by default — the three
            together are over 2,000 columns, which is rarely what you want
            implicitly.
        prefix: prefix each column with its provider (``12sl_HR__Global``), so
            columns from different providers cannot collide. The three feature
            sets share many names, so leaving this off risks silent overwrites.

    Example:
        >>> plus = load_ptbxl_plus("/data/ptb-xl-plus/1.0.1/", features=("unig",))
        >>> ds = ECGDataset("ptbxl", split="train", data_path="/data/ptb-xl/1.0.3/")
        >>> joined = ds.labels_df.join(plus, how="left")   # PTB-XL's own folds
    """
    frames: list[pd.DataFrame] = []
    for provider in statements:
        df = load_statements(data_path, provider)
        frames.append(df.add_prefix(f"{provider}_") if prefix else df)
    for provider in features:
        df = load_features(data_path, provider)
        frames.append(df.add_prefix(f"{provider}_") if prefix else df)

    if not frames:
        raise ValueError("Nothing requested: pass at least one statements or features set")

    out = pd.concat(frames, axis=1)
    out.index.name = JOIN_COLUMN
    if out.columns.duplicated().any():
        dupes = sorted(set(out.columns[out.columns.duplicated()]))
        raise ValueError(
            f"Duplicate columns after concatenation: {dupes[:5]}. Pass prefix=True "
            "(the default) so provider columns cannot collide."
        )
    logger.info(
        "PTB-XL+ frame: %d records x %d columns (statements=%s features=%s)",
        len(out),
        out.shape[1],
        list(statements),
        list(features),
    )
    return out
