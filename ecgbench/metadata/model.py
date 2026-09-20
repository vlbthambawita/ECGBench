"""Typed, source-agnostic description of one ECGBench dataset.

The catalogue front matter and the YAML configs describe the same datasets from
two angles, in two slug namespaces, with overlapping and occasionally
disagreeing values. This module defines the one record both are merged into:
``DatasetMeta``, composed of independent facets (signal, access, split,
relations) so a facet can be absent — a catalogue-only dataset has no signal
facet — without breaking the others.

Every value that came from a source file is also kept as a ``Fact`` carrying a
``Provenance``, including the duplicates. The top-level fields hold the winning
value under the precedence ``manifest > validation_report > config >
catalogue``; ``facts`` keeps every candidate so a disagreement can be shown
rather than hidden.

Standard library only: this module is imported by ``ecgbench.metadata`` at
package import, so it must cost nothing.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

#: Fact sources, most trustworthy first. A manifest is computed from the files;
#: a validation report is computed from the signals; a config was written while
#: implementing the dataset; a catalogue entry may predate any of that.
SOURCE_PRECEDENCE: tuple[str, ...] = ("manifest", "validation_report", "config", "catalogue")

#: Values ``DatasetMeta.implementation_state`` can take, least to most complete.
IMPLEMENTATION_STATES: tuple[str, ...] = ("catalogue_only", "config", "config_labels", "published")

#: Version of the JSON layout written by ``ecgbench.metadata.build``; bump on an
#: incompatible change so a store can refuse a file it does not understand.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Provenance:
    """Where a fact came from.

    Attributes:
        source: One of ``SOURCE_PRECEDENCE``.
        source_path: Repository-relative path of the file the value was read from.
        observed_at: ISO-8601 timestamp for computed sources (manifests, reports);
            ``None`` for hand-edited ones.
    """

    source: str
    source_path: str
    observed_at: str | None = None


@dataclass(frozen=True)
class Fact:
    """One value for one key from one source.

    ``value`` is restricted to what JSON can carry: ``str``, ``int``, ``float``,
    ``bool``, ``None`` or a list of those.
    """

    key: str
    value: object
    provenance: Provenance


@dataclass(frozen=True)
class SignalMeta:
    """What the waveform files hold — from the config only.

    Attributes:
        format: One of the ``_load_signal`` branches (``wfdb``, ``csv``, ``edf``, …).
        leads: Number of leads in the predominant layout.
        lead_names: Lead names in file order for the predominant layout, or ``None``
            when the config does not declare them.
        alternate_lead_names: Layouts for records storing a different *number* of
            leads, keyed by that count.
        record_lead_layouts: Every layout used at the *same* lead count, for a
            release whose records name their leads differently (``mitdb``).
        sampling_rates: All rates the release ships.
        default_sampling_rate: The rate ``ECGDataset`` loads when none is asked for.
        duration_seconds: Nominal record length.
        units: Physical unit after ``unit_scale`` — ``mV`` everywhere but
            ``echonext``'s ``zscore``.
        unit_scale: Multiplier from stored sample values to ``units``.
        zero_padded_identifiers: Whether ids must be read as strings to survive.
    """

    format: str
    leads: int
    lead_names: tuple[str, ...] | None
    alternate_lead_names: dict[int, tuple[str, ...]] | None
    record_lead_layouts: tuple[tuple[str, ...], ...] | None
    sampling_rates: tuple[int, ...]
    default_sampling_rate: int
    duration_seconds: float
    units: str
    unit_scale: float
    zero_padded_identifiers: bool


@dataclass(frozen=True)
class AccessMeta:
    """How the data can be obtained and what ECGBench may republish.

    Attributes:
        access: ``open``, ``credentialed`` or ``restricted`` (catalogue vocabulary).
        license_text: Human-readable licence name, as the catalogue states it.
        license_url: A licence URL when the config gives one, else ``None``.
        url: Landing page of the source release.
        download_url: Direct archive URL when the config declares one.
        publish_fold_csvs: Whether fold CSVs go to the public Hub repo. Only
            meaningful when a config exists; ``False`` for catalogue-only entries.
        no_publish_reason: The config's explanation, which includes the command
            to regenerate the split locally; empty when publishing is allowed.
    """

    access: str
    license_text: str | None
    license_url: str | None
    url: str
    download_url: str | None
    publish_fold_csvs: bool
    no_publish_reason: str


@dataclass(frozen=True)
class SplitMeta:
    """How ECGBench partitions the dataset — from the config only.

    Attributes:
        n_folds: Fold count ``ecgbench splits`` produces by default.
        predefined_column: The source's own fold column when the release ships a
            split ECGBench adopts (``strat_fold`` for PTB-XL), else ``None``.
        has_patient_id: Whether folds are grouped by a patient column.
        record_id_column: Column naming a record in the metadata CSV.
    """

    n_folds: int
    predefined_column: str | None
    has_patient_id: bool
    record_id_column: str | None


@dataclass(frozen=True)
class RelationMeta:
    """An edge to another dataset, mirrored from the catalogue's ``related`` block.

    Attributes:
        target: ``dataset_id`` of the other dataset.
        relation: One of ``catalogue._RELATION_INVERSES``.
        shares_records: ``True`` when the two contain the same recordings, which
            is the leakage signal; ``None`` when unknown.
        verified: Whether the overlap was checked against the files.
        note: Free text, mandatory when ``shares_records`` is ``True``.
        derived: ``True`` when this direction was inverted from the other side's
            declaration rather than written in this dataset's front matter.
    """

    target: str
    relation: str
    shares_records: bool | None
    verified: bool
    note: str
    derived: bool


@dataclass(frozen=True)
class DatasetMeta:
    """The unified record for one dataset.

    Attributes:
        dataset_id: Config slug where a config exists, else the catalogue slug.
        aliases: Every name the dataset answers to — both slugs and both display
            names, first the catalogue slug.
        name: Display name from the catalogue (the curated, website-facing one;
            a differing config name is kept as an alias and a fact).
        category: Catalogue category (``12-lead-physionet``, ``two-lead``, …).
        status: Catalogue ``status:`` — not a reliable implementation signal.
        implementation_state: Derived: ``catalogue_only`` (no config), ``config``
            (config but no label loader), ``config_labels`` (labels available) or
            ``published`` (labels available and fold CSVs may be published; Hub
            presence is not checked).
        version: Release version from the config, ``None`` without one.
        description: Config description, falling back to the catalogue page's
            first overview section.
        paper_title: Catalogue citation short form.
        paper_doi: DOI URL, from the catalogue or derived from the config's DOI.
        citation: Full citation from the config, empty without one.
        origin_institution: Catalogue field.
        origin_country: Catalogue field.
        search_keywords: Catalogue keyword string.
        records: Record count parsed from the winning ``records`` fact, ``None``
            when the display string is not a plain integer.
        patients: Same for patients.
        records_display: The catalogue's own string, kept verbatim.
        patients_display: Same for patients.
        signal: Signal facet, ``None`` for catalogue-only datasets.
        access: Access facet, always present.
        split: Split facet, ``None`` for catalogue-only datasets.
        relations: Edges to other datasets, both directions materialised.
        facts: Every sourced value with provenance, duplicates included.
        prose: Concatenated page text and config prose, for free-text search only.
    """

    dataset_id: str
    aliases: tuple[str, ...]
    name: str
    category: str
    status: str
    implementation_state: str
    version: str | None
    description: str
    paper_title: str | None
    paper_doi: str | None
    citation: str
    origin_institution: str
    origin_country: str | None
    search_keywords: str
    records: int | None
    patients: int | None
    records_display: str
    patients_display: str
    signal: SignalMeta | None
    access: AccessMeta
    split: SplitMeta | None
    relations: tuple[RelationMeta, ...] = ()
    facts: tuple[Fact, ...] = ()
    prose: str = field(default="", repr=False)

    @property
    def has_config(self) -> bool:
        """Whether a YAML config implements this dataset."""
        return self.implementation_state != "catalogue_only"

    @property
    def has_labels(self) -> bool:
        """Whether ``load_labels()`` can return a label table for it."""
        return self.implementation_state in ("config_labels", "published")

    @property
    def published(self) -> bool:
        """Whether its fold CSVs may be fetched from the public Hub repo."""
        return self.implementation_state == "published"

    def facts_for(self, key: str) -> tuple[Fact, ...]:
        """Every fact recorded under ``key``, most trustworthy source first."""
        return tuple(sorted((f for f in self.facts if f.key == key), key=_fact_rank))

    def fact(self, key: str) -> Fact | None:
        """The winning fact for ``key`` by source precedence, or ``None``."""
        ranked = self.facts_for(key)
        return ranked[0] if ranked else None

    def disagreements(self) -> dict[str, tuple[Fact, ...]]:
        """Keys whose sources give more than one distinct value."""
        out: dict[str, tuple[Fact, ...]] = {}
        for key in sorted({f.key for f in self.facts}):
            ranked = self.facts_for(key)
            if len({_canonical(f.value) for f in ranked}) > 1:
                out[key] = ranked
        return out

    def to_dict(self) -> dict:
        """JSON-ready mapping; ``from_dict`` inverts it exactly."""
        data = dataclasses.asdict(self)
        if self.signal is not None and self.signal.alternate_lead_names is not None:
            data["signal"]["alternate_lead_names"] = {
                str(k): list(v) for k, v in self.signal.alternate_lead_names.items()
            }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> DatasetMeta:
        """Rebuild a record from ``to_dict()`` output (or the bundled JSON)."""
        signal = data.get("signal")
        access = data["access"]
        split = data.get("split")
        return cls(
            dataset_id=data["dataset_id"],
            aliases=tuple(data["aliases"]),
            name=data["name"],
            category=data["category"],
            status=data["status"],
            implementation_state=data["implementation_state"],
            version=data.get("version"),
            description=data.get("description", ""),
            paper_title=data.get("paper_title"),
            paper_doi=data.get("paper_doi"),
            citation=data.get("citation", ""),
            origin_institution=data.get("origin_institution", ""),
            origin_country=data.get("origin_country"),
            search_keywords=data.get("search_keywords", ""),
            records=data.get("records"),
            patients=data.get("patients"),
            records_display=data.get("records_display", ""),
            patients_display=data.get("patients_display", ""),
            signal=_signal_from_dict(signal) if signal else None,
            access=AccessMeta(**access),
            split=SplitMeta(**split) if split else None,
            relations=tuple(RelationMeta(**r) for r in data.get("relations", ())),
            facts=tuple(
                Fact(key=f["key"], value=f["value"], provenance=Provenance(**f["provenance"]))
                for f in data.get("facts", ())
            ),
            prose=data.get("prose", ""),
        )


def _signal_from_dict(data: dict) -> SignalMeta:
    alternates = data.get("alternate_lead_names")
    layouts = data.get("record_lead_layouts")
    return SignalMeta(
        format=data["format"],
        leads=data["leads"],
        lead_names=tuple(data["lead_names"]) if data.get("lead_names") is not None else None,
        alternate_lead_names=(
            {int(k): tuple(v) for k, v in alternates.items()} if alternates is not None else None
        ),
        record_lead_layouts=(
            tuple(tuple(layout) for layout in layouts) if layouts is not None else None
        ),
        sampling_rates=tuple(data["sampling_rates"]),
        default_sampling_rate=data["default_sampling_rate"],
        duration_seconds=data["duration_seconds"],
        units=data["units"],
        unit_scale=data["unit_scale"],
        zero_padded_identifiers=data["zero_padded_identifiers"],
    )


def _fact_rank(fact: Fact) -> tuple[int, str]:
    source = fact.provenance.source
    known = source in SOURCE_PRECEDENCE
    rank = SOURCE_PRECEDENCE.index(source) if known else len(SOURCE_PRECEDENCE)
    return (rank, fact.provenance.source_path)


def _canonical(value: object) -> object:
    """Hashable form of a fact value, so lists can be compared for disagreement."""
    if isinstance(value, list):
        return tuple(_canonical(v) for v in value)
    return value
