"""Exports of the metadata model: Croissant 1.1, schema.org JSON-LD and the website's JSON.

Three views over ``DatasetMeta``, all built from the model alone and all
deterministic (no timestamps but the snapshot's own ``created``), so a committed
copy can be checked against a fresh export the way ``metadata.json`` is:

- :func:`to_schema_org` - one ``schema.org/Dataset`` per dataset, the block a
  dataset page embeds as ``<script type="application/ld+json">`` so search
  engines index the catalogue as datasets.
- :func:`to_croissant` / :func:`to_croissant_collection` - the same record as an
  MLCommons Croissant 1.1 ``Dataset`` (a ``DataCatalog`` of them for the
  collection), with ``distribution`` naming the published fold CSVs on the Hub
  and their SHA-256 from the snapshot, a ``RecordSet`` per version over the fold
  table's guaranteed columns, PROV-O ``wasDerivedFrom`` over the run's input
  files, and an ODRL prohibition on redistribution for a dataset whose fold CSVs
  are withheld. ``mlcroissant`` validates each dataset (:func:`validate_croissant`).
- :func:`to_website` - ``docs/_data/metadata.json``, keyed by **catalogue slug**
  (what ``page.slug`` holds in the Jekyll layout), carrying the resolved counts,
  the declared field names for the search box, and the schema.org block.

Standard-library only, like the rest of the build: ``mlcroissant`` is imported
lazily by the validator alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ecgbench.metadata.model import ArtefactMeta, DatasetMeta, FieldMeta

#: Where the catalogue is served; one page per dataset under ``datasets/``.
SITE_URL = "https://vlbthambawita.github.io/ECGBench"

#: The public Hub repo the fold CSVs are published to (``dataset.py`` hard-codes
#: the same id on the download side).
HF_REPO_ID = "vlbthambawita/ECGBench"
HF_RESOLVE_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main"

#: The committed website copy, read by Liquid as ``site.data.metadata``.
WEBSITE_JSON_PATH = Path(__file__).resolve().parents[2] / "docs" / "_data" / "metadata.json"

CROISSANT_CONFORMS_TO = "http://mlcommons.org/croissant/1.1"

#: The Croissant 1.1 JSON-LD context as ``mlcroissant`` 1.0.22 emits it, plus the
#: PROV-O and ODRL prefixes the 1.1 constructs below use.
CROISSANT_CONTEXT: dict[str, Any] = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "samplingRate": "cr:samplingRate",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
    "odrl": "http://www.w3.org/ns/odrl/2/",
    "prov": "http://www.w3.org/ns/prov#",
}

#: Frictionless field types -> Croissant ``dataType``.
_CROISSANT_TYPES = {
    "string": "sc:Text",
    "integer": "sc:Integer",
    "number": "sc:Float",
    "boolean": "sc:Boolean",
    "date": "sc:Date",
    "datetime": "sc:DateTime",
    "time": "sc:Time",
    "duration": "sc:Text",
    "any": "sc:Text",
    "object": "sc:Text",
    "array": "sc:Text",
}

#: The columns every exported fold CSV carries (``splitting/export.py``), typed.
_FOLD_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("fold", "sc:Integer", "1-indexed fold the record is assigned to"),
    ("default_split", "sc:Text", "train, val or test under the default fold mapping"),
)
_ORIGINAL_ONLY_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("is_valid", "sc:Boolean", "Whether the record passed every quality check"),
    (
        "quality_issues",
        "sc:Text",
        "Semicolon-joined check failures, empty for a valid record",
    ),
)


# --------------------------------------------------------------------------- helpers


def catalogue_slug(meta: DatasetMeta) -> str:
    """The dashed catalogue slug, which is always the first alias."""
    return meta.aliases[0]


def page_url(meta: DatasetMeta) -> str:
    """URL of the dataset's page on the catalogue site."""
    return f"{SITE_URL}/datasets/{catalogue_slug(meta)}.html"


def _keywords(meta: DatasetMeta) -> list[str]:
    seen: dict[str, None] = {}
    for word in meta.search_keywords.split():
        seen.setdefault(word.lower(), None)
    return list(seen)


def _license(meta: DatasetMeta) -> str | None:
    return meta.access.license_url or meta.access.license_text or None


def _fold_artefacts(meta: DatasetMeta) -> list[ArtefactMeta]:
    return [a for a in meta.artefacts if a.kind == "fold_csv"]


def _input_artefacts(meta: DatasetMeta) -> list[ArtefactMeta]:
    return [a for a in meta.artefacts if a.kind == "input"]


def _record_id_column(meta: DatasetMeta) -> str | None:
    return meta.split.record_id_column if meta.split else None


def _patient_id_column(meta: DatasetMeta) -> str | None:
    fact = meta.fact("patient_id_column")
    return str(fact.value) if fact and fact.value else None


def _variable_measured(fields: tuple[FieldMeta, ...]) -> list[dict[str, Any]]:
    out = []
    for f in fields:
        entry: dict[str, Any] = {"@type": "PropertyValue", "name": f.name}
        if f.description:
            entry["description"] = f.description
        if f.unit:
            entry["unitText"] = f.unit
        out.append(entry)
    return out


# --------------------------------------------------------------------------- schema.org


def to_schema_org(meta: DatasetMeta) -> dict[str, Any]:
    """The ``schema.org/Dataset`` JSON-LD block for one dataset.

    Only standard schema.org properties are used, so Google's Rich Results test
    accepts the block; the recomputed counts and the fold digests do not fit any
    of them and stay in the metadata export.
    """
    doc: dict[str, Any] = {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "@id": page_url(meta),
        "identifier": meta.dataset_id,
        "name": meta.name,
        "url": page_url(meta),
        "includedInDataCatalog": {"@type": "DataCatalog", "name": "ECGBench", "url": SITE_URL},
    }
    alternates = [a for a in meta.aliases if a != meta.name]
    if alternates:
        doc["alternateName"] = alternates
    if meta.description:
        doc["description"] = meta.description
    if meta.version:
        doc["version"] = meta.version
    same_as = [u for u in (meta.access.url, meta.paper_doi) if u]
    if same_as:
        doc["sameAs"] = same_as
    licence = _license(meta)
    if licence:
        doc["license"] = licence
    doc["isAccessibleForFree"] = meta.access.access == "open"
    if meta.access.access != "open":
        doc["conditionsOfAccess"] = meta.access.access
    keywords = _keywords(meta)
    if keywords:
        doc["keywords"] = keywords
    if meta.origin_institution:
        creator: dict[str, Any] = {"@type": "Organization", "name": meta.origin_institution}
        if meta.origin_country:
            creator["address"] = {"@type": "PostalAddress", "addressCountry": meta.origin_country}
        doc["creator"] = creator
    citation = meta.paper_doi or (meta.citation or None)
    if citation:
        doc["citation"] = citation
    based_on = sorted(
        {
            f"{SITE_URL}/datasets/{r.target}.html"
            for r in meta.relations
            if r.relation in ("derived_from", "subset_of") and not r.derived
        }
    )
    if based_on:
        doc["isBasedOn"] = based_on
    if meta.fields:
        doc["variableMeasured"] = _variable_measured(meta.fields)
    distribution: list[dict[str, Any]] = []
    if meta.access.download_url:
        distribution.append(
            {
                "@type": "DataDownload",
                "name": "source data",
                "contentUrl": meta.access.download_url,
            }
        )
    if meta.published:
        for artefact in _fold_artefacts(meta):
            entry: dict[str, Any] = {
                "@type": "DataDownload",
                "name": f"ECGBench {artefact.version} folds",
                "contentUrl": f"{HF_RESOLVE_URL}/{meta.dataset_id}/{artefact.name}",
                "encodingFormat": "text/csv",
            }
            if artefact.file_sha256:
                entry["sha256"] = artefact.file_sha256
            distribution.append(entry)
    if distribution:
        doc["distribution"] = distribution
    created = next((a.created for a in meta.artefacts if a.created), None)
    if created:
        doc["dateModified"] = created[:10]
    return doc


# --------------------------------------------------------------------------- croissant


def _fold_file_object(meta: DatasetMeta, artefact: ArtefactMeta) -> dict[str, Any]:
    obj: dict[str, Any] = {
        "@type": "cr:FileObject",
        "@id": f"{artefact.version}-folds",
        "name": f"{meta.dataset_id}/{artefact.name}",
        "description": (
            f"ECGBench {artefact.version} partition of {meta.name}: "
            f"{artefact.n_records:,} records with their fold assignment"
            if artefact.n_records is not None
            else f"ECGBench {artefact.version} partition of {meta.name}"
        ),
        "contentUrl": f"{HF_RESOLVE_URL}/{meta.dataset_id}/{artefact.name}",
        "encodingFormat": "text/csv",
    }
    if artefact.file_sha256:
        obj["sha256"] = artefact.file_sha256
    return obj


def _fold_record_set(meta: DatasetMeta, artefact: ArtefactMeta) -> dict[str, Any]:
    columns: list[tuple[str, str, str]] = []
    record_id = _record_id_column(meta)
    if record_id:
        columns.append((record_id, "sc:Text", "Record identifier, the join key to the labels"))
    patient_id = _patient_id_column(meta)
    if patient_id and patient_id != record_id:
        columns.append((patient_id, "sc:Text", "Patient identifier the folds are grouped on"))
    columns += _FOLD_COLUMNS
    if artefact.version == "original":
        columns += _ORIGINAL_ONLY_COLUMNS
    fields = [
        {
            "@type": "cr:Field",
            "@id": f"{artefact.version}/{name}",
            "name": name,
            "description": description,
            "dataType": data_type,
            "source": {
                "fileObject": {"@id": f"{artefact.version}-folds"},
                "extract": {"column": name},
            },
        }
        for name, data_type, description in columns
    ]
    record_set: dict[str, Any] = {
        "@type": "cr:RecordSet",
        "@id": artefact.version,
        "name": f"{artefact.version} folds",
        "description": (
            "Every record of the original release with its fold"
            if artefact.version == "original"
            else "The records that passed validation, with their fold"
        ),
        "field": fields,
    }
    if record_id:
        record_set["key"] = {"@id": f"{artefact.version}/{record_id}"}
    return record_set


def to_croissant(meta: DatasetMeta, include_1_1: bool = True) -> dict[str, Any]:
    """One dataset as a Croissant ``Dataset`` (JSON-LD, with the shared context).

    ``distribution`` and ``recordSet`` are emitted only for a published dataset
    with a snapshot; a withheld one gets ``conditionsOfAccess`` and, with
    ``include_1_1``, an ODRL policy prohibiting distribution instead. The
    run's inputs become ``prov:wasDerivedFrom`` file objects.
    """
    doc: dict[str, Any] = {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "@id": page_url(meta),
        "conformsTo": CROISSANT_CONFORMS_TO,
        "name": meta.dataset_id,
        "identifier": meta.dataset_id,
        "alternateName": [a for a in meta.aliases if a != meta.dataset_id],
        "description": meta.description or meta.name,
        "url": meta.access.url or page_url(meta),
    }
    if meta.version:
        doc["version"] = meta.version
    licence = _license(meta)
    if licence:
        doc["license"] = licence
    citation = meta.citation or meta.paper_doi
    if citation:
        doc["citeAs"] = citation
    keywords = _keywords(meta)
    if keywords:
        doc["keywords"] = keywords
    if meta.origin_institution:
        doc["creator"] = {"@type": "sc:Organization", "name": meta.origin_institution}
    same_as = [u for u in (page_url(meta), meta.paper_doi) if u and u != doc["url"]]
    if same_as:
        doc["sameAs"] = same_as
    doc["isAccessibleForFree"] = meta.access.access == "open"
    if meta.access.access != "open":
        doc["conditionsOfAccess"] = meta.access.access
    created = next((a.created for a in meta.artefacts if a.created), None)
    if created:
        doc["datePublished"] = created[:10]
    if meta.fields:
        doc["variableMeasured"] = _variable_measured(meta.fields)

    folds = _fold_artefacts(meta)
    if meta.published and folds:
        doc["distribution"] = [_fold_file_object(meta, a) for a in folds]
        doc["recordSet"] = [_fold_record_set(meta, a) for a in folds]
    elif not meta.access.publish_fold_csvs:
        reason = meta.access.no_publish_reason or "fold CSVs are not published"
        doc["conditionsOfAccess"] = reason
        if include_1_1:
            doc["odrl:hasPolicy"] = {
                "@type": "odrl:Policy",
                "odrl:prohibition": [{"odrl:action": "odrl:distribute"}],
                "odrl:target": page_url(meta),
            }
    if include_1_1:
        inputs = _input_artefacts(meta)
        if inputs:
            doc["prov:wasDerivedFrom"] = [
                {
                    "@type": "cr:FileObject",
                    "@id": f"input-{i}",
                    "name": a.name,
                    "description": "Input file the split run read, hashed by the manifest",
                    "contentUrl": a.name,
                    "encodingFormat": "text/csv",
                    "sha256": a.sha256,
                }
                for i, a in enumerate(inputs)
            ]
    return doc


def to_croissant_collection(
    model: tuple[DatasetMeta, ...], include_1_1: bool = True
) -> dict[str, Any]:
    """Every configured dataset as one JSON-LD ``DataCatalog`` of Croissant datasets.

    Catalogue-only entries have no config, no signals ECGBench reads and no
    partition, so they are listed by page URL under ``hasPart`` rather than as
    Croissant datasets.
    """
    datasets = [to_croissant(m, include_1_1) for m in model if m.has_config]
    for d in datasets:
        d.pop("@context")  # one context at the top of the document
    return {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:DataCatalog",
        "@id": SITE_URL,
        "name": "ECGBench",
        "description": (
            "Reproducible, pre-validated benchmark partitions of open ECG datasets, one "
            "Croissant record per configured dataset."
        ),
        "url": SITE_URL,
        "dataset": datasets,
        "hasPart": [page_url(m) for m in model if not m.has_config],
    }


def validate_croissant(doc: dict[str, Any]) -> list[str]:
    """Validate a Croissant document or collection with ``mlcroissant``.

    Returns the error messages, one per failing dataset, empty when valid.
    Warnings (recommended properties) are not errors.

    Each member is loaded as an ``mlcroissant.Dataset`` rather than a bare
    ``Metadata``: only the ``Dataset`` constructor builds the structure graph
    and raises ``ValidationError`` for a missing ``@type``, a field whose
    source names no ``FileObject``, or a broken ``RecordSet`` -- ``Metadata``
    alone collects the recommended-property warnings and nothing else.

    Raises:
        ImportError: ``mlcroissant`` is not installed (``pip install ecgbench[croissant]``).
    """
    try:
        import mlcroissant as mlc
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "mlcroissant is required to validate Croissant output: "
            "pip install ecgbench[croissant]"
        ) from exc

    if doc.get("@type") == "sc:DataCatalog":
        members = [{**d, "@context": doc["@context"]} for d in doc.get("dataset", ())]
    else:
        members = [doc]
    errors: list[str] = []
    for member in members:
        name = member.get("name", "?")
        try:
            dataset = mlc.Dataset(jsonld=member)
        except Exception as exc:  # noqa: BLE001 - ValidationError, but the library raises more
            errors.append(f"{name}: {exc}")
            continue
        errors += [f"{name}: {e}" for e in dataset.metadata.issues.errors]
    return errors


# --------------------------------------------------------------------------- website


def to_website(model: tuple[DatasetMeta, ...]) -> dict[str, dict[str, Any]]:
    """``docs/_data/metadata.json``: one compact entry per dataset, keyed by catalogue slug."""
    out: dict[str, dict[str, Any]] = {}
    for meta in model:
        checks = {
            f.key.partition(":")[2]: f.value
            for f in meta.facts
            if f.key.startswith("quality_check:") and f.provenance.source == "validation_report"
        }
        clean = meta.fact("records_clean")
        excluded = meta.fact("records_excluded")
        created = next((a.created for a in meta.artefacts if a.created), None)
        field_names = [f.name for f in meta.fields]
        out[catalogue_slug(meta)] = {
            "dataset_id": meta.dataset_id,
            "name": meta.name,
            "implementation_state": meta.implementation_state,
            "records": meta.records,
            "records_clean": clean.value if clean else None,
            "records_excluded": excluded.value if excluded else None,
            "patients": meta.patients,
            "n_fields": len(field_names),
            "field_names": field_names,
            "quality_checks": dict(sorted(checks.items())),
            "snapshot_created": created,
            "search_text": " ".join(dict.fromkeys([*meta.aliases, *field_names])),
            "schema_org": to_schema_org(meta),
        }
    return out


def website_json(model: tuple[DatasetMeta, ...]) -> str:
    """The canonical text of the website copy (sorted keys, two-space indent)."""
    return json.dumps(to_website(model), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_website_json(
    model: tuple[DatasetMeta, ...], path: Path | str = WEBSITE_JSON_PATH
) -> bool:
    """Write the website copy; returns ``True`` when the file changed."""
    target = Path(path)
    text = website_json(model)
    if target.is_file() and target.read_text(encoding="utf-8") == text:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return True


def website_json_current(
    model: tuple[DatasetMeta, ...], path: Path | str = WEBSITE_JSON_PATH
) -> bool:
    """Whether the committed website copy matches a fresh export."""
    target = Path(path)
    return target.is_file() and target.read_text(encoding="utf-8") == website_json(model)
