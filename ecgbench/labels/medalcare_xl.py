"""MedalCare-XL labels: the simulated condition, and the parameters that made it.

Nothing here is a diagnosis. Every record is the output of an electrophysiological
simulation, so the label is **the condition the simulator was configured to
produce** — exact by construction, with no reader disagreement, no ambiguity and no
comorbidity. That is the release's strength and its limitation at once: a model
that separates these classes perfectly has learned to separate simulator settings,
which is a weaker claim than separating patients.

Two label layers, coarse first:

    pathology            8 classes — sinus, avblock, lbbb, rbbb, lae, fam, iab, mi
    pathology_subclass  15 classes — the 7 non-MI pathologies plus 8 MI subclasses

The MI subclass decomposes further into the three fields the release varies:
``mi_occlusion_site`` (LAD/LCX/RCA), ``mi_transmurality`` (0.3 or 1.0) and
``mi_region`` (ant/post, resolved only for LCX — the one site the release splits).
All three are blank for the 9,042 non-MI records. ``pathology_subclass`` is what
the config stratifies on.

**Single-label, and every record carries one.** No record is unlabelled and no
record has two conditions: the simulator produced one pathology per run.

LABELS DEPEND ON PIPELINE ORDER. The release ships no metadata table at all —
pathology, subclass, split and model live in the directory path and nowhere else —
so these labels come from ``ecgbench_metadata.csv``, which
``MedalCareXLSplitter.load_metadata`` generates on the first ``ecgbench splits``
run against a writable dataset root. Before that run the file does not exist and
``LabelSourceMissingError`` says so.

THE SIMULATION PARAMETERS ARE THE REAL GROUND TRUTH, AND THEY ARE OPT-IN.
Each record ships ``<n>_AtrialParameters.txt`` (~21 keys) and
``<n>_VentricularParameters.txt`` (~105 keys) holding the ionic model, tissue
conductivities, conduction velocities, action-potential durations, stimulus sites,
ischaemic-region geometry, atrial and torso geometry and the ten electrode
positions. ``load_labels`` does **not** read them — that would be 33,684 file
opens on every ``ECGDataset(labels=True)``. Call
:func:`load_simulation_parameters` when you want them.

The key set is ragged, which is information rather than damage:

- MI records add 14 ``isch[0].*`` keys describing the ischaemic region;
- ``lbbb`` and ``rbbb`` records drop the ``stim[*]`` block (2 and 1 of the 5
  stimulus sites respectively are removed to create the block);
- ``lae`` records drop the 4 ``cv_t.*`` regional atrial conduction velocities.

So the union is 119 ventricular and 21 atrial keys, of which 70 and 17 appear in
every pathology. Absent keys come back as NaN, and a NaN there means "this
pathology does not have this parameter", not "missing data".

``siginfo.csv`` — one per run directory, naming the atrial simulation that supplied
each P wave — is deliberately not read anywhere in ECGBench. It carries no record
number for fam/iab/lae, and in 13 of the 186 run directories it has more rows than
there are records, so joining it means guessing at row order. The parameter files
carry the same anatomy (``geo.atria``, ``geo.torso``) keyed by record number, which
is why they are the source used here.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

__all__ = [
    "ATRIAL_SUFFIX",
    "PATHOLOGY_NAMES",
    "SAMPLING_RATE",
    "VENTRICULAR_SUFFIX",
    "load_labels",
    "load_simulation_parameters",
    "parse_parameter_file",
]

#: Constant across the release — 10 s at 500 Hz, 5000 samples, every record.
SAMPLING_RATE = 500

ATRIAL_SUFFIX = "_AtrialParameters.txt"
VENTRICULAR_SUFFIX = "_VentricularParameters.txt"

#: Directory-name abbreviation -> the condition it stands for. The directory names
#: are what the metadata CSV records; these are for display and documentation.
PATHOLOGY_NAMES = {
    "sinus": "normal sinus rhythm",
    "avblock": "AV block",
    "lbbb": "left bundle branch block",
    "rbbb": "right bundle branch block",
    "lae": "left atrial enlargement",
    "fam": "fibrotic atrial cardiomyopathy",
    "iab": "interatrial conduction block",
    "mi": "myocardial infarction",
}

#: Columns carried straight through from the generated metadata CSV.
_LABEL_COLUMNS = [
    "pathology",
    "pathology_subclass",
    "mi_subclass",
    "mi_occlusion_site",
    "mi_transmurality",
    "mi_region",
    "model_id",
    "source_split",
    "signal_path",
    "signal_path_raw",
    "signal_path_noise",
    "atrial_params_path",
    "ventricular_params_path",
]


def _metadata_path(data_path: Path, config: DatasetConfig) -> Path:
    from ecgbench.labels import LabelSourceMissingError

    path = data_path / config.metadata_csv
    if not path.exists():
        raise LabelSourceMissingError(
            f"MedalCare-XL labels come from {config.metadata_csv}, which is not in "
            f"{data_path}. Unlike most datasets this file does not ship — the release "
            "has no metadata table, so ECGBench generates it from the directory tree. "
            "Run it once against a writable copy of the data:\n"
            f"    ecgbench splits --dataset {config.slug} --data-path {data_path}\n"
            f"See {config.url} for the source download."
        )
    return path


def load_labels(data_path: Path | str, config: DatasetConfig) -> pd.DataFrame:
    """Return MedalCare-XL labels indexed by ``record_id``.

    Columns: ``pathology`` (8 classes), ``pathology_subclass`` (15),
    ``pathology_name`` (the spelled-out condition), ``mi_subclass``,
    ``mi_occlusion_site``, ``mi_transmurality``, ``mi_region``, ``model_id``,
    ``source_split``, ``sampling_rate``, and the three signal-variant paths plus
    the two parameter-file paths.

    Single-label — one simulated condition per record, none unlabelled. The MI
    columns are blank for every non-MI record. Simulation parameters are not
    included; see :func:`load_simulation_parameters`.
    """
    root = Path(data_path)
    df = pd.read_csv(
        _metadata_path(root, config),
        sep=config.metadata_csv_separator,
        dtype={"model_id": str, "record_number": str},
    )

    missing = [c for c in [config.record_id_column, *_LABEL_COLUMNS] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Generated metadata CSV is missing column(s) {missing}. It was probably "
            "written by an older ECGBench; delete "
            f"{root / config.metadata_csv} and re-run `ecgbench splits`."
        )

    df = df.set_index(config.record_id_column)[_LABEL_COLUMNS].sort_index()
    df.insert(1, "pathology_name", df["pathology"].map(PATHOLOGY_NAMES))
    df["sampling_rate"] = SAMPLING_RATE

    logger.info(
        "Loaded MedalCare-XL labels: %d records, %d pathologies, %s",
        len(df),
        df["pathology"].nunique(),
        df["pathology"].value_counts().to_dict(),
    )
    return df


def parse_parameter_file(path: Path | str) -> dict[str, str]:
    """Parse one ``key = value`` parameter file into a dict.

    Values keep their shipped spelling with surrounding quotes stripped, so
    ``im.name = "MitchellSchaeffer"`` and ``im.name = Courtemanche`` — the two
    files are inconsistent about quoting — both yield a bare model name. Units
    travel with the value (``cv_t.BulkTissue = 591mm/s``, ``rot.x = 0deg``) exactly
    as the release writes them; nothing is coerced to a number here, because a
    silent coercion would drop the unit.
    """
    values: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            key, sep, value = line.partition("=")
            if sep:
                values[key.strip()] = value.strip().strip('"')
    return values


def load_simulation_parameters(
    data_path: Path | str,
    config: DatasetConfig | str = "medalcare_xl",
    *,
    record_ids: list[str] | None = None,
    kind: str = "both",
    max_workers: int = 16,
) -> pd.DataFrame:
    """Return the per-record simulation parameters, indexed by ``record_id``.

    This reads one or two small text files **per record**, so the full release is
    16,842 or 33,684 file opens — tens of seconds on a warm cache and minutes on a
    cold one. Pass ``record_ids`` to restrict it to the split you care about; that
    is the intended usage and it is what the example script does.

    Args:
        data_path: Root of a local copy (the directory holding
            ``WP2_largeDataset_Noise/``).
        config: Slug or ``DatasetConfig``. Only used to find the metadata CSV.
        record_ids: Restrict to these records. ``None`` reads all of them.
        kind: ``"atrial"``, ``"ventricular"`` or ``"both"``.
        max_workers: Thread count. These reads are I/O bound.

    Returns:
        DataFrame indexed by ``record_id``, columns prefixed ``atrial.`` and
        ``ventricular.`` — the two files share key names (``G.torso``, ``im.name``)
        and an unprefixed concat would silently drop one of each. The union is
        ragged by pathology; NaN means the parameter does not apply to that
        pathology, not that a value is missing. See the module docstring.
    """
    from ecgbench.config import load_config

    if isinstance(config, str):
        config = load_config(config)
    if kind not in ("atrial", "ventricular", "both"):
        raise ValueError(f"kind must be 'atrial', 'ventricular' or 'both', got {kind!r}")

    root = Path(data_path)
    meta = pd.read_csv(_metadata_path(root, config), sep=config.metadata_csv_separator).set_index(
        config.record_id_column
    )

    if record_ids is not None:
        unknown = [r for r in record_ids if r not in meta.index]
        if unknown:
            raise KeyError(
                f"{len(unknown)} record id(s) are not in {config.metadata_csv}, "
                f"e.g. {unknown[:3]}"
            )
        meta = meta.loc[list(record_ids)]

    kinds = ("atrial", "ventricular") if kind == "both" else (kind,)
    frames = []
    for one in kinds:
        paths = [root / p for p in meta[f"{one}_params_path"]]
        logger.info("Reading %d %s parameter files under %s", len(paths), one, root)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            parsed = list(pool.map(parse_parameter_file, paths))
        frames.append(pd.DataFrame(parsed, index=meta.index).add_prefix(f"{one}."))

    out = pd.concat(frames, axis=1)
    duplicated = out.columns[out.columns.duplicated()]
    if len(duplicated):
        # Cannot happen with the two prefixes above, but a third provider added
        # later must not silently overwrite an existing column.
        raise ValueError(f"Duplicate parameter columns after prefixing: {list(duplicated)}")

    logger.info("Loaded simulation parameters: %d records x %d parameters", len(out), out.shape[1])
    return out
