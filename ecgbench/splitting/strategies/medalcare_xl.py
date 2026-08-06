"""MedalCare-XL (synthetic 12-lead ECGs from electrophysiological simulations).

This release ships **no metadata table of any kind** — every fact about a record
is encoded in its path::

    WP2_largeDataset_Noise/<pathology>/[<mi_subclass>/]<split>/run_<model>/<nnnnnn>_<variant>.csv

``load_metadata`` therefore builds one by walking that tree and caches it next to
the data as ``config.metadata_csv``. Writing the cache to disk is load-bearing,
not a convenience: ``validate_dataset`` re-reads ``data_path /
config.metadata_csv`` itself rather than reusing the splitter's DataFrame, so an
in-memory-only frame would leave validation with no metadata at all.

Four properties of the tree that this module has to deal with:

1. **Three files per simulated ECG.** ``<n>_raw.csv`` (noise-free simulator
   output), ``<n>_noise.csv`` (noise superimposed) and ``<n>_filtered.csv``
   (noise plus a 0.5-150 Hz order-3 Butterworth). They are one record in three
   renderings, not three records: 16,842 simulations, 50,526 signal files. The
   config points ``signal_path`` at the **filtered** variant, the closest thing
   to a recorded ECG; ``signal_path_raw`` and ``signal_path_noise`` carry the
   other two so nothing is lost.

2. **``mi/examples/`` is not data.** Six files named ``S62_<site>_<transmurality>_*.csv``
   sit outside the ``<split>/run_<model>/`` layout and are byte-distinct from every
   record in ``mi/*/test/run_S62/``. They are figure illustrations and are skipped,
   which is why this module reports 16,842 records where a naive ``rglob`` finds
   16,848. The parameter-file tree agrees: it holds exactly 16,842 atrial and
   16,842 ventricular files.

3. **Record numbering has gaps.** Files are numbered from 000001 per run directory
   but 13 of the 186 run directories are missing some — ``iab/train/run_S66`` has
   107 files numbered up to 000130. Records are therefore enumerated from the
   files that exist, never from a range.

4. **The split is predefined and this module does not re-derive it.** ``train``,
   ``validation`` and ``test`` are directory names chosen by the authors so that
   ECGs sharing a ventricular simulation model land in only one of them. See the
   config for the one way that guarantee does not hold globally.

``siginfo.csv`` (one per run directory, naming the atrial simulation that supplied
each P wave) is deliberately **not** read. Its rows carry no record number for
fam/iab/lae and its row count exceeds the file count in 13 directories, so joining
it means guessing at row order. The per-record ``*_AtrialParameters.txt`` files
carry the same anatomy keyed by record number instead, and
``ecgbench/labels/medalcare_xl.py`` reads those.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from ecgbench.config import DatasetConfig
from ecgbench.splitting.base import DatasetSplitter
from ecgbench.splitting.registry import register

logger = logging.getLogger(__name__)

#: Directory (relative to the dataset root) holding the signal tree.
SIGNALS_DIR = "WP2_largeDataset_Noise"

#: Directory (relative to the dataset root) holding the per-record simulation
#: parameter files. Not read here — see ecgbench/labels/medalcare_xl.py.
PARAMS_DIR = "WP2_largeDataset_ParameterFiles"

#: Directory names under SIGNALS_DIR that are splits rather than MI subclasses.
SPLIT_DIRS = ("train", "validation", "test")

#: Source split name -> ECGBench fold number. Mirrors predefined_splits in the
#: config, where fold 1 is train, 2 is val and 3 is test.
SPLIT_TO_FOLD = {"train": 1, "validation": 2, "test": 3}

#: MI subclass directory -> (occlusion site, transmurality, region). The region
#: is only resolved for LCX, the one site the release splits anterior/posterior.
_MI_SUBCLASS = re.compile(r"^(LAD|LCX|RCA)_(0\.3|1\.0)(?:_(ant|post))?$")


def _parse_mi_subclass(name: str) -> tuple[str, float, str | None]:
    """Split an MI subclass directory name into its three components."""
    match = _MI_SUBCLASS.match(name)
    if match is None:
        raise ValueError(
            f"Unrecognised MI subclass directory {name!r} under {SIGNALS_DIR}/mi/. "
            "Expected <LAD|LCX|RCA>_<0.3|1.0>[_<ant|post>]."
        )
    site, transmurality, region = match.groups()
    return site, float(transmurality), region


def _run_directories(signals_root: Path) -> list[tuple[Path, str, str | None, str]]:
    """Find every ``run_<model>`` directory, with the facts its path encodes.

    Returns ``(directory, pathology, mi_subclass, split)`` tuples. The MI arm
    carries one extra level — ``mi/<subclass>/<split>/run_<model>`` against
    ``<pathology>/<split>/run_<model>`` — and ``mi/examples/`` has no split level
    at all, which is how it gets excluded here rather than by name.
    """
    found: list[tuple[Path, str, str | None, str]] = []
    for pathology_dir in sorted(p for p in signals_root.iterdir() if p.is_dir()):
        pathology = pathology_dir.name
        for child in sorted(p for p in pathology_dir.iterdir() if p.is_dir()):
            if child.name in SPLIT_DIRS:
                levels = [(child.name, None)]
            else:
                # Either an MI subclass directory holding the three split
                # directories, or mi/examples/ — which holds loose illustration
                # files and no split level at all. Test for the split level first,
                # so anything that *does* look like data still has to parse as a
                # recognised subclass rather than being silently accepted.
                levels = [
                    (grandchild.name, child.name)
                    for grandchild in sorted(p for p in child.iterdir() if p.is_dir())
                    if grandchild.name in SPLIT_DIRS
                ]
                if not levels:
                    logger.info(
                        "Skipping %s — no train/validation/test level, so not records "
                        "(mi/examples/ holds figure illustrations)",
                        child,
                    )
                    continue
                _parse_mi_subclass(child.name)  # raises on anything unexpected
            for split, subclass in levels:
                split_dir = child if subclass is None else child / split
                for run_dir in sorted(
                    p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("run_")
                ):
                    found.append((run_dir, pathology, subclass, split))
    return found


def build_metadata(data_path: Path, config: DatasetConfig) -> pd.DataFrame:
    """Walk the signal tree into a metadata frame, one row per simulated ECG.

    Signal paths are stored relative to ``data_path`` so they resolve the same
    way for the splitter, the validation engine and ``ECGDataset``.
    """
    signals_root = data_path / SIGNALS_DIR
    if not signals_root.is_dir():
        raise FileNotFoundError(
            f"Expected the signal tree at {signals_root}. Point --data-path at the "
            f"directory holding {SIGNALS_DIR}/ and {PARAMS_DIR}/ — note the release "
            "extracts to a MedalCare-XL/MedalCare-XL/ nesting."
        )

    run_dirs = _run_directories(signals_root)
    if not run_dirs:
        raise FileNotFoundError(f"No run_<model> directories found under {signals_root}")
    logger.info("Scanning %d run directories under %s", len(run_dirs), signals_root)

    params_root = data_path / PARAMS_DIR
    signal_col = config.signal_path_columns[config.default_sampling_rate]

    rows: list[dict[str, object]] = []
    for run_dir, pathology, subclass, split in run_dirs:
        model = run_dir.name.removeprefix("run_")
        # Enumerate from the files that exist: numbering has gaps in 13 of the
        # 186 run directories, so a range(1, n + 1) would invent records.
        for filtered in sorted(run_dir.glob("*_filtered.csv")):
            number = filtered.name.removesuffix("_filtered.csv")
            stem = "_".join(part for part in (pathology, subclass, split, model, number) if part)
            site = transmurality = region = None
            if subclass is not None:
                site, transmurality, region = _parse_mi_subclass(subclass)
            params_dir = params_root / run_dir.relative_to(signals_root)
            rows.append(
                {
                    "record_id": stem,
                    "model_id": model,
                    "pathology": pathology,
                    "mi_subclass": subclass,
                    "mi_occlusion_site": site,
                    "mi_transmurality": transmurality,
                    "mi_region": region,
                    # 15 classes: the 7 non-MI pathologies plus the 8 MI subclasses.
                    "pathology_subclass": pathology if subclass is None else f"mi_{subclass}",
                    "source_split": split,
                    "fold": SPLIT_TO_FOLD[split],
                    "record_number": number,
                    signal_col: str(filtered.relative_to(data_path)),
                    "signal_path_raw": str((run_dir / f"{number}_raw.csv").relative_to(data_path)),
                    "signal_path_noise": str(
                        (run_dir / f"{number}_noise.csv").relative_to(data_path)
                    ),
                    "atrial_params_path": str(
                        (params_dir / f"{number}_AtrialParameters.txt").relative_to(data_path)
                    ),
                    "ventricular_params_path": str(
                        (params_dir / f"{number}_VentricularParameters.txt").relative_to(data_path)
                    ),
                }
            )

    df = pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)

    duplicated = df["record_id"].duplicated()
    if duplicated.any():
        raise ValueError(
            "Record ids are not unique — the path components they are built from "
            f"must have collided. Examples: {sorted(df.loc[duplicated, 'record_id'])[:5]}"
        )

    logger.info(
        "Built metadata for %d records across %d pathologies (folds: %s)",
        len(df),
        df["pathology"].nunique(),
        df["fold"].value_counts().sort_index().to_dict(),
    )
    return df


@register("medalcare_xl")
class MedalCareXLSplitter(DatasetSplitter):
    """MedalCare-XL splitting strategy.

    - Builds (and caches) the metadata CSV by walking the signal tree, since the
      release ships no metadata table
    - Honours the authors' own train/validation/test directories as folds 1/2/3
      rather than generating new ones
    - Stratifies on ``pathology_subclass`` (15 classes), which is the finest
      label the directory layout carries
    """

    def load_metadata(self, data_path: Path, config: DatasetConfig) -> pd.DataFrame:
        csv_path = data_path / config.metadata_csv

        if csv_path.exists():
            logger.info("Reading cached metadata: %s", csv_path)
            return pd.read_csv(
                csv_path,
                sep=config.metadata_csv_separator,
                dtype={"record_number": str, "model_id": str},
            )

        df = build_metadata(data_path, config)
        try:
            df.to_csv(csv_path, sep=config.metadata_csv_separator, index=False)
            logger.info("Wrote metadata CSV: %s", csv_path)
        except OSError as e:
            # validate_dataset re-reads this file, so a read-only data directory
            # means validation cannot see any metadata. Fail loudly instead.
            raise OSError(
                f"Could not write the generated metadata CSV to {csv_path}: {e}. "
                "The dataset root must be writable, because the validation engine "
                "reads the metadata CSV from disk."
            ) from e
        return df

    def get_stratification_labels(self, df: pd.DataFrame, config: DatasetConfig) -> pd.Series:
        return df["pathology_subclass"].astype(str)
