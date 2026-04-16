"""
Dataset configuration system.

Every dataset is fully described by a YAML config file. This module provides
the typed DatasetConfig dataclass and a loader that parses YAML into it.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class CreatorInfo:
    """Dataset creator or contributing organisation."""

    type: str  # "Organization" or "Person"
    name: str
    url: str | None = None


@dataclass
class StratificationConfig:
    """How to derive stratification labels for splitting."""

    method: str  # "superclass_mapping", "direct", "custom_function"
    mapping_source: str | None = None
    superclass_column: str | None = None


@dataclass
class ValidationConfig:
    """Quality validation settings for a dataset."""

    expected_leads: int
    expected_samples: dict[int, int]  # sampling_rate -> expected sample count
    checks: list[str]
    amplitude_range_mv: tuple[float, float] = (-10.0, 10.0)


@dataclass
class PredefinedSplitConfig:
    """Describes a dataset's built-in fold assignments."""

    column: str  # column name containing fold numbers
    fold_mapping: dict[str, list[int]]  # {"train": [1..8], "val": [9], "test": [10]}


@dataclass
class CroissantConfig:
    """Croissant (MLCommons) metadata fields."""

    keywords: list[str] = field(default_factory=list)
    rai_data_collection: str = ""
    rai_data_biases: str = ""
    rai_personal_sensitive_info: str = ""


@dataclass
class DatasetConfig:
    """Complete typed representation of a dataset YAML config."""

    # Identity (required)
    name: str
    slug: str
    version: str
    url: str

    # Identity (optional)
    download_url: str | None = None
    license: str = ""
    description: str = ""
    citation: str = ""
    doi: str = ""
    creators: list[CreatorInfo] = field(default_factory=list)

    # Signal properties
    signal_format: str = "wfdb"
    leads: int = 12
    duration_seconds: float = 10.0
    sampling_rates: list[int] = field(default_factory=lambda: [500])
    default_sampling_rate: int = 500

    # File structure (required)
    metadata_csv: str = ""
    metadata_csv_separator: str = ","
    record_id_column: str = "ecg_id"
    patient_id_column: str | None = None
    signal_path_columns: dict[int, str] = field(default_factory=dict)

    # Labels (required)
    label_column: str = ""
    label_format: str = "single"  # dict_string, single, comma_separated, json
    stratification: StratificationConfig | None = None

    # Splits
    has_predefined_splits: bool = False
    predefined_splits: PredefinedSplitConfig | None = None

    # Validation
    validation: ValidationConfig | None = None

    # Croissant
    croissant: CroissantConfig = field(default_factory=CroissantConfig)


_CONFIGS_DIR = Path(__file__).parent / "data" / "configs"

_REQUIRED_FIELDS = ("name", "slug", "version", "url", "metadata_csv", "record_id_column",
                    "label_column")


def _parse_creators(raw: list[dict] | None) -> list[CreatorInfo]:
    if not raw:
        return []
    return [CreatorInfo(type=c["type"], name=c["name"], url=c.get("url")) for c in raw]


def _parse_stratification(raw: dict | None) -> StratificationConfig | None:
    if not raw or not raw.get("method"):
        return None
    return StratificationConfig(
        method=raw["method"],
        mapping_source=raw.get("mapping_source"),
        superclass_column=raw.get("superclass_column"),
    )


def _parse_validation(raw: dict | None) -> ValidationConfig | None:
    if not raw:
        return None
    expected_samples = {int(k): v for k, v in raw.get("expected_samples", {}).items()}
    amp = raw.get("amplitude_range_mv", [-10.0, 10.0])
    return ValidationConfig(
        expected_leads=raw["expected_leads"],
        expected_samples=expected_samples,
        checks=raw.get("checks", []),
        amplitude_range_mv=(amp[0], amp[1]),
    )


def _parse_predefined_splits(raw: dict | None) -> PredefinedSplitConfig | None:
    if not raw or not raw.get("column"):
        return None
    fold_mapping = {}
    for split_name, folds in raw.get("fold_mapping", {}).items():
        fold_mapping[split_name] = list(folds) if folds else []
    return PredefinedSplitConfig(column=raw["column"], fold_mapping=fold_mapping)


def _parse_croissant(raw: dict | None) -> CroissantConfig:
    if not raw:
        return CroissantConfig()
    return CroissantConfig(
        keywords=raw.get("keywords", []),
        rai_data_collection=raw.get("rai_data_collection", ""),
        rai_data_biases=raw.get("rai_data_biases", ""),
        rai_personal_sensitive_info=raw.get("rai_personal_sensitive_info", ""),
    )


def _parse_signal_path_columns(raw: dict | None) -> dict[int, str]:
    if not raw:
        return {}
    return {int(k): v for k, v in raw.items()}


def load_config(dataset_slug: str) -> DatasetConfig:
    """Load and validate a dataset config from YAML.

    Searches ecgbench/data/configs/{dataset_slug}.yaml.
    Parses YAML into DatasetConfig dataclass with full validation.

    Raises:
        FileNotFoundError: if config YAML doesn't exist
        ValueError: if required fields are missing or invalid
    """
    config_path = _CONFIGS_DIR / f"{dataset_slug}.yaml"
    if not config_path.exists():
        available = list_available_configs()
        raise FileNotFoundError(
            f"Config not found: {config_path}. Available configs: {available}"
        )

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw or not isinstance(raw, dict):
        raise ValueError(f"Config file is empty or not a YAML mapping: {config_path}")

    # Validate required fields
    missing = [k for k in _REQUIRED_FIELDS if not raw.get(k)]
    if missing:
        raise ValueError(
            f"Config '{dataset_slug}' missing required fields: {missing}"
        )

    return DatasetConfig(
        name=raw["name"],
        slug=raw["slug"],
        version=raw["version"],
        url=raw["url"],
        download_url=raw.get("download_url"),
        license=raw.get("license", ""),
        description=raw.get("description", ""),
        citation=raw.get("citation", ""),
        doi=raw.get("doi", ""),
        creators=_parse_creators(raw.get("creators")),
        signal_format=raw.get("signal_format", "wfdb"),
        leads=raw.get("leads", 12),
        duration_seconds=raw.get("duration_seconds", 10.0),
        sampling_rates=raw.get("sampling_rates", [500]),
        default_sampling_rate=raw.get("default_sampling_rate", 500),
        metadata_csv=raw["metadata_csv"],
        metadata_csv_separator=raw.get("metadata_csv_separator", ","),
        record_id_column=raw["record_id_column"],
        patient_id_column=raw.get("patient_id_column"),
        signal_path_columns=_parse_signal_path_columns(raw.get("signal_path_columns")),
        label_column=raw["label_column"],
        label_format=raw.get("label_format", "single"),
        stratification=_parse_stratification(raw.get("stratification")),
        has_predefined_splits=raw.get("has_predefined_splits", False),
        predefined_splits=_parse_predefined_splits(raw.get("predefined_splits")),
        validation=_parse_validation(raw.get("validation")),
        croissant=_parse_croissant(raw.get("croissant")),
    )


def list_available_configs() -> list[str]:
    """Return slugs of all available dataset configs."""
    if not _CONFIGS_DIR.exists():
        return []
    return sorted(
        p.stem for p in _CONFIGS_DIR.glob("*.yaml")
        if not p.stem.startswith("_")
    )
