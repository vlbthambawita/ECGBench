"""
Croissant (MLCommons) JSON-LD metadata generation and validation.

Generates Croissant 1.1 JSON-LD using the mlcroissant library (optional dep).
Includes SHA-256 hashes for all CSV files in the splits directory.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)


def _require_mlcroissant():
    """Lazily import mlcroissant, raising a helpful error if not installed."""
    try:
        import mlcroissant as mlc

        return mlc
    except ImportError:
        raise ImportError(
            "mlcroissant is required for Croissant metadata generation. "
            "Install with: pip install ecgbench[croissant]"
        )


def _sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _discover_csv_files(splits_dir: Path) -> list[Path]:
    """Find all CSV files in the splits directory tree."""
    return sorted(splits_dir.rglob("*.csv"))


def _infer_field_type(column_name: str, dtype_str: str):
    """Map pandas dtype to Croissant DataType."""
    mlc = _require_mlcroissant()
    if "int" in dtype_str:
        return mlc.DataType.INTEGER
    if "float" in dtype_str:
        return mlc.DataType.FLOAT
    if "date" in dtype_str.lower():
        return mlc.DataType.DATE
    return mlc.DataType.TEXT


def generate_croissant(
    config: DatasetConfig,
    splits_dir: Path,
    version: str = "clean",
) -> dict:
    """Generate Croissant 1.1 JSON-LD for a dataset version.

    Uses the mlcroissant library to build a valid Metadata object.

    Args:
        config: DatasetConfig
        splits_dir: Path to the version dir (e.g., .../ptbxl/clean/)
        version: "original" or "clean"

    Returns:
        dict — the Croissant JSON-LD
    """
    mlc = _require_mlcroissant()
    import pandas as pd

    splits_dir = Path(splits_dir)

    # Discover all CSV files
    csv_files = _discover_csv_files(splits_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {splits_dir}")

    # Build FileObjects for each CSV
    file_objects = []
    for csv_path in csv_files:
        rel_path = csv_path.relative_to(splits_dir)
        name = str(rel_path).replace("/", "-").replace("\\", "-").replace(".csv", "-csv")
        file_objects.append(
            mlc.FileObject(
                id=name,
                name=name,
                content_url=str(rel_path),
                encoding_formats=["text/csv"],
                sha256=_sha256(csv_path),
            )
        )

    # Build RecordSets for train/val/test from the split directories
    record_sets = []
    for split_name in ("train", "val", "test"):
        split_dir = splits_dir / split_name
        if not split_dir.exists():
            continue

        fold_csvs = sorted(split_dir.glob("fold_*.csv"))
        if not fold_csvs:
            continue

        sample_df = pd.read_csv(fold_csvs[0], nrows=5)

        # Find a file object ID to reference as source
        source_ref = None
        for fo in file_objects:
            if split_name in fo.id and "fold" in fo.id:
                source_ref = fo.id
                break

        fields = []
        for col in sample_df.columns:
            dtype = _infer_field_type(col, str(sample_df[col].dtype))
            field = mlc.Field(
                id=f"{split_name}-{col}",
                name=col,
                data_types=dtype,
                source=mlc.Source(
                    file_object=source_ref,
                    extract=mlc.Extract(column=col),
                ),
            )
            fields.append(field)

        record_sets.append(
            mlc.RecordSet(
                id=f"{split_name}-records",
                name=f"{split_name}-records",
                fields=fields,
            )
        )

    # Creator info
    creators = []
    for c in config.creators:
        cls = mlc.Organization if c.type == "Organization" else mlc.Person
        creators.append(cls(name=c.name, url=c.url))

    # Build the Croissant Metadata
    keywords = config.croissant.keywords if config.croissant else ["ECG"]
    try:
        metadata = mlc.Metadata(
            name=f"{config.slug}-{version}",
            url=config.url,
            description=config.description or f"{config.name} ECG dataset ({version} version)",
            sd_licence=config.license or "",
            version=config.version,
            cite_as=config.citation or "",
            date_published=date.today().isoformat(),
            conforms_to="http://mlcommons.org/croissant/1.1",
            creators=creators or None,
            keywords=keywords,
            distribution=file_objects,
            record_sets=record_sets,
        )
    except Exception as e:
        logger.warning(
            "Failed to build Croissant Metadata object: %s. "
            "Falling back to manual JSON-LD construction.",
            e,
        )
        return _build_manual_jsonld(config, splits_dir, version, csv_files)

    try:
        return metadata.to_json()
    except Exception:
        return json.loads(json.dumps(metadata.__dict__, default=str))


def _build_manual_jsonld(
    config: DatasetConfig,
    splits_dir: Path,
    version: str,
    csv_files: list[Path],
) -> dict:
    """Build Croissant JSON-LD manually as a fallback."""
    distribution = []
    for csv_path in csv_files:
        rel_path = csv_path.relative_to(splits_dir)
        distribution.append({
            "@type": "cr:FileObject",
            "name": str(rel_path).replace("/", "-").replace(".csv", "-csv"),
            "contentUrl": str(rel_path),
            "encodingFormat": "text/csv",
            "sha256": _sha256(csv_path),
        })

    creators = []
    for c in config.creators:
        creators.append({
            "@type": c.type,
            "name": c.name,
            **({"url": c.url} if c.url else {}),
        })

    return {
        "@context": {
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
        },
        "@type": "cr:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "name": f"{config.slug}-{version}",
        "url": config.url,
        "description": config.description or f"{config.name} ECG dataset ({version} version)",
        "license": config.license or "",
        "version": config.version,
        "datePublished": date.today().isoformat(),
        "keywords": config.croissant.keywords if config.croissant else ["ECG"],
        **({"creator": creators} if creators else {}),
        **({"citation": config.citation} if config.citation else {}),
        "distribution": distribution,
    }


def save_croissant(
    config: DatasetConfig,
    splits_dir: Path,
    output_path: Path | None = None,
    version: str = "clean",
) -> Path:
    """Generate and save croissant.json.

    Args:
        config: DatasetConfig
        splits_dir: Path to the version dir
        output_path: Where to write. Defaults to splits_dir/croissant.json
        version: "original" or "clean"

    Returns:
        Path to the saved file
    """
    splits_dir = Path(splits_dir)
    if output_path is None:
        output_path = splits_dir / "croissant.json"
    output_path = Path(output_path)

    croissant_data = generate_croissant(config, splits_dir, version)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(croissant_data, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Saved Croissant metadata to %s", output_path)
    return output_path


def validate_croissant(croissant_path: Path) -> tuple[bool, list[str]]:
    """Validate a Croissant JSON-LD file.

    Args:
        croissant_path: Path to the croissant.json file

    Returns:
        (is_valid, list_of_errors)
    """
    mlc = _require_mlcroissant()

    errors = []
    try:
        metadata = mlc.Metadata(jsonld=croissant_path)
        # Attempt to access properties to trigger validation
        _ = metadata.name
        _ = metadata.distribution
    except Exception as e:
        errors.append(str(e))

    return (len(errors) == 0, errors)
