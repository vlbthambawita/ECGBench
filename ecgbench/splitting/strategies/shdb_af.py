"""
SHDB-AF splitting strategy.

``AdditionalData.csv`` is a real metadata file — 45 columns, one row per recording
— so unlike ``ltafdb`` this dataset does not have to invent one. It still needs a
custom splitter for three reasons, and the first is the one that would otherwise
bite silently.

**The clinical CSV has no signal-path column, and no column that can be turned into
one by configuration.** Signals are the bare zero-padded stem (``001``) in a flat
tree; ``Data_ID`` happens to *be* that stem, but ``signal_path_columns`` maps a
sampling rate to a column name and cannot point two roles at one column without the
exported fold CSVs losing the distinction. So ``load_metadata`` writes
``ecgbench_metadata.csv`` with an explicit ``signal_path``.

**That file has to be on disk, not just in memory.** ``validate_dataset`` re-reads
``data_path / config.metadata_csv`` itself rather than reusing this DataFrame, so an
in-memory-only frame would leave validation with no metadata at all. Chapman shipped
that bug for months and every record failed ``corrupt_header``.

**The fold label does not exist in any shipped column.** It is ``AF_Type`` crossed
with whether the recording carries a ``.atr``, which means reading the annotation
layer — so ``load_metadata`` goes through ``ecgbench.labels.shdb_af``, the same
loader users get from ``load_labels``, and reads the column it attaches. The
stratification label and the exposed label therefore cannot drift.

**Two things about the split itself.**

Folds are grouped on ``Subject_ID``. Six of the 122 subjects contributed two
recordings each (015/047, 005/020, 066/118, 052/128, 035/036, 129/133), so a
record-level split would put the same person on both sides of it. That grouping also
happens to contain this release's one real hazard: **005 and 020 are byte-identical
recordings** — same SHA-256 for both ``.dat`` and ``.qrs`` in the release's own
manifest, presented in the clinical table as two Holters three years apart. They
share ``Subject_ID`` 4899921, so grouping keeps the duplicate inside one fold. That
is luck rather than design, and ``duplicate_of`` in the labels names the pair for
anyone computing a per-record metric.

Record ids are zero-padded three-digit strings and must stay strings. ``001``
becomes ``1`` under pandas' default inference, at which point it no longer names a
record and ``data_path / "1"`` is not a file — every record would fail
``corrupt_header`` for a reason nothing in the traceback mentions. Every read here
passes ``config.identifier_dtypes()``, and ``export_splits`` refuses to write the
fold CSVs if the config ever loses ``zero_padded_identifiers``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import DatasetSplitter
from ecgbench.splitting.registry import register

logger = logging.getLogger(__name__)

#: Column the label loader attaches, used for stratification.
STRATIFY_COLUMN = "stratify_class"


@register("shdb_af")
class SHDBAFSplitter(DatasetSplitter):
    """SHDB-AF strategy: clinical CSV plus annotation layer, subject-grouped folds."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # Data_ID is "001" and signal_path likewise. Both MUST stay
                # strings: read as numbers they lose the leading zeros, and wfdb is
                # then handed a record called "1" that does not exist.
                dtype=config.identifier_dtypes(),
            )

        from ecgbench.labels.shdb_af import load_labels

        df = load_labels(data_path, config).reset_index()
        try:
            df.to_csv(csv_path, sep=config.metadata_csv_separator, index=False)
            logger.info("Wrote metadata CSV: %s", csv_path)
        except OSError as e:
            # validate_dataset re-reads this file, so a read-only data directory
            # leaves validation with no metadata at all. Fail loudly instead.
            raise OSError(
                f"Could not write the generated metadata CSV to {csv_path}: {e}. "
                "The dataset root must be writable, because the validation engine "
                "reads the metadata CSV from disk."
            ) from e
        return df

    def get_stratification_labels(self, df: pd.DataFrame, config: DatasetConfig) -> pd.Series:
        """Return the AF-type x annotation-availability label the loader attached."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("af_type_annotation_class")

        counts = labels.value_counts()
        logger.info("Fold classes:\n%s", counts.to_string())
        # StratifiedGroupKFold does not raise for a class it cannot spread over
        # every fold — it quietly leaves some folds without one. Say so here, since
        # nothing downstream will.
        if counts.min() < 10:
            logger.warning(
                "Smallest fold class holds %d records, fewer than the 10 folds "
                "ECGBench generates, so some folds will contain none of it. Widen "
                "the cross in ecgbench.labels.shdb_af.attach_stratify_class.",
                int(counts.min()),
            )
        return labels
