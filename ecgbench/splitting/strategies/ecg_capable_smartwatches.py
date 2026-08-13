"""ECG-capable smartwatches splitting strategy.

**No metadata file ships in any form.** The release is five device directories, a
``README``, a ``RECORDS`` index and 915 WFDB record pairs; every fact a split needs
— which device, which simulator setting, which repetition — exists only in the
directory names, and the facts that decide validity exist only in the samples. So
``load_metadata`` builds a metadata CSV through :mod:`ecgbench.labels.ecg_capable_smartwatches`,
the same loader users get from ``load_labels``, so the stratification label and the
exposed labels cannot drift apart.

Writing that cache to disk is load-bearing rather than a convenience:
``validate_dataset`` re-reads ``data_path / config.metadata_csv`` itself instead of
reusing this DataFrame, so an in-memory-only frame would leave validation with no
metadata at all. It also saves a fifteen-second rescan of 13.7 million samples on
every later run.

**Folds group on the simulator setting, and that is the whole design.** There is no
patient in this dataset — the subject is a METRON PS-440 patient simulator — so
``patient_id_column`` points at ``setting_id``, the 36 simulator conditions, and it
is named ``setting_id`` rather than anything patient-shaped so nothing here
pretends otherwise. Three measurements decided that, all made over the shipped
files with max-over-lag correlation of the single lead (Philips: its lead II),
resampled to a common 250 Hz where devices differ:

- **The five repetitions of a setting on one device are near-duplicates.** Median
  correlation 0.81 (Philips) to 0.994 (Withings), 0.95 across the release. An
  ungrouped split scatters them, so almost every held-out record has four
  near-copies of itself in the training set.
- **The same setting on a *different* device is nearly as similar: median 0.803**,
  against a 0.953 same-device self-control. The five instruments recorded the same
  simulator output simultaneously, so grouping on ``(device, setting)`` would still
  leak — the group has to be the setting alone, across all five devices.
- **Different settings are genuinely different**, which is what makes the grouping
  meaningful rather than merely conservative: adjacent ``freq_test`` settings
  correlate at a median of 0.070.

**The one place the grouping does not buy what it looks like.** Adjacent
``st-segment`` settings correlate at a median of 0.805 and up to 0.996, because the
ST ladder is a 100 µV step and ``st-p1`` really is almost ``st-p2``. Holding out
``st-p2`` therefore does not hold out an unseen *condition* in the way holding out
``f240`` does. That is a property of a densely sampled continuous label, not a
defect to engineer around — collapsing the family into one group would leave four
groups in total and no way to make ten folds — but anything reporting ST-level
accuracy should say which offsets were held out rather than quoting a fold number.

**Rejected: grouping on the repetition index.** Six groups (``_0``..``_5``), so
every fold would hold all 36 settings and all five devices, and a fold would be
"the third repetition of every experiment" — superficially the natural split for a
release designed around repeated measurement. It is discarded because those
repetitions are the 0.95-correlated near-duplicates above: every test record would
have a twin in training. It also caps ``n_folds`` at 6 and the groups are uneven,
since 17 settings have a sixth repetition and two have only four.

**``n_folds`` stays at the project default of 10**, and the fold count is not what
limits this dataset. A stratification class needs ``n_folds`` groups to reach every
fold; ``amp_test`` has 4 settings and ``sqr-2hz`` has 1, so ``amp_test`` reaches 4
folds and ``sqr-2hz`` reaches 1 at *any* fold count above four, and no choice fixes
both. Ten keeps fold sizes at 77-102 records and every device at 15-22 per fold.
The consequence for the default 8/1/1 mapping is stated in
:func:`ecgbench.labels.ecg_capable_smartwatches.attach_stratify_class`: fold 10 —
the default test set — holds ``freq_test`` and ``st-segment`` records only.
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


@register("ecg_capable_smartwatches")
class ECGCapableSmartwatchesSplitter(DatasetSplitter):
    """Simulator-setting-grouped folds over five simultaneously recording devices."""

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                # Pinned for the same reason tollet pins its subject id:
                # StratifiedGroupKFold orders its groups by value, so if the first
                # run grouped on strings from the loader and a later one grouped on
                # something pandas inferred differently, the two would partition
                # identical data differently and stamp different fold_digests into
                # manifest.json. None of these three is ever all-digits — ids start
                # with a device name, settings with a letter — so nothing is being
                # rescued from coercion here, only held still.
                dtype={"record_id": str, "setting_id": str, "signal_path": str},
            )

        from ecgbench.labels.ecg_capable_smartwatches import load_labels

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

    def get_stratification_labels(
        self, df: pd.DataFrame, config: DatasetConfig
    ) -> pd.Series:
        """Return the experiment family attached by the label loader."""
        if STRATIFY_COLUMN not in df.columns:
            raise ValueError(
                f"'{STRATIFY_COLUMN}' missing — call load_metadata() first, or pass a "
                "DataFrame produced by it."
            )

        labels = df[STRATIFY_COLUMN].astype(str).rename("experiment_family")

        logger.info("Fold classes (experiment family):\n%s", labels.value_counts().to_string())
        if "setting_id" in df.columns:
            settings = df.groupby(STRATIFY_COLUMN)["setting_id"].nunique()
            logger.info("Simulator settings per fold class:\n%s", settings.to_string())
            # StratifiedGroupKFold keeps groups intact, so a class with fewer
            # settings than folds cannot appear in every fold. Its own output says
            # nothing at all about this, and here it is unavoidable rather than
            # fixable — see the module docstring.
            short = settings[settings < config.n_folds]
            if not short.empty:
                logger.warning(
                    "Experiment families %s hold fewer simulator settings than the %d "
                    "folds ECGBench generates (%s), so they cannot appear in every "
                    "fold. This is forced by the release: use split=None with "
                    "fold_numbers to select folds holding a given family.",
                    list(short.index), config.n_folds, short.to_dict(),
                )
        if "device" in df.columns:
            logger.info(
                "%d records from %d devices at %d simulator settings; %d records "
                "carry a trailing invalid sample and will fail nan_values.",
                len(df),
                df["device"].nunique(),
                df["setting_id"].nunique() if "setting_id" in df else -1,
                int(df["trailing_invalid_sample"].sum())
                if "trailing_invalid_sample" in df
                else -1,
            )
        return labels
