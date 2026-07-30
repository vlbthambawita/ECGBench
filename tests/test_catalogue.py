"""Tests for the dataset catalogue, focused on cross-dataset relationships.

The `related:` blocks are the one part of the catalogue that can rot silently: a
typo'd slug renders a dead link, and a hand-written reverse edge drifts from its
counterpart. These tests make both failures loud.
"""

import pytest

import ecgbench
from ecgbench.catalogue import (
    _RELATION_INVERSES,
    RelatedLink,
    _parse_related,
    _with_reverse_edges,
)


@pytest.fixture(scope="module")
def entries():
    return ecgbench.list_datasets()


@pytest.fixture(scope="module")
def by_slug(entries):
    return {e.slug: e for e in entries}


class TestCatalogueLoads:
    def test_all_entries_present(self, entries):
        assert len(entries) == 64

    def test_every_slug_is_unique(self, entries):
        slugs = [e.slug for e in entries]
        assert len(slugs) == len(set(slugs))


class TestRelatedIntegrity:
    """Whole-catalogue invariants — these guard the shipped .md files."""

    def test_every_related_slug_resolves(self, entries, by_slug):
        dangling = [
            (e.slug, link.slug)
            for e in entries
            for link in e.related
            if link.slug not in by_slug
        ]
        assert not dangling, f"related entries point at unknown datasets: {dangling}"

    def test_every_relation_is_in_the_vocabulary(self, entries):
        bad = [
            (e.slug, link.relation)
            for e in entries
            for link in e.related
            if link.relation not in _RELATION_INVERSES
        ]
        assert not bad, f"unknown relation values: {bad}"

    def test_no_self_links(self, entries):
        assert not [e.slug for e in entries for link in e.related if link.slug == e.slug]

    def test_every_edge_is_mirrored(self, entries, by_slug):
        """A→B implies B→A with the inverse relation, whoever declared it."""
        missing = []
        for entry in entries:
            for link in entry.related:
                other = by_slug[link.slug]
                back = [x for x in other.related if x.slug == entry.slug]
                if not back:
                    missing.append(f"{link.slug} has no edge back to {entry.slug}")
                elif back[0].relation != _RELATION_INVERSES[link.relation]:
                    missing.append(
                        f"{entry.slug} --{link.relation}--> {link.slug}, but back edge is "
                        f"{back[0].relation}, expected {_RELATION_INVERSES[link.relation]}"
                    )
        assert not missing, missing

    def test_relationships_are_declared_once(self, entries):
        """Both endpoints declaring the same pair would double-count on the website.

        The Liquid include recomputes reverse edges by scanning the collection, so
        a reciprocal pair in front matter shows up twice there while catalogue.py
        keeps one. Declare each relationship on one side only.
        """
        duplicates = []
        for entry in entries:
            declared = {r["slug"] for r in (entry.raw.get("related") or []) if r.get("slug")}
            for other_slug in declared:
                other = next(e for e in entries if e.slug == other_slug)
                other_declared = {
                    r["slug"] for r in (other.raw.get("related") or []) if r.get("slug")
                }
                if entry.slug in other_declared:
                    duplicates.append(f"{entry.slug} <-> {other_slug}")
        assert not duplicates, f"declared from both sides: {sorted(set(duplicates))}"

    def test_python_and_liquid_agree_on_edge_counts(self, entries):
        """catalogue.py's derived edges must match what the Liquid include counts.

        Liquid computes outgoing (front matter) + incoming (scan of every other
        entry). catalogue.py computes declared + derived inverses. Those are the
        same number unless a relationship is declared twice.
        """
        for entry in entries:
            outgoing = len(entry.raw.get("related") or [])
            incoming = sum(
                1
                for other in entries
                if other.slug != entry.slug
                for r in (other.raw.get("related") or [])
                if r.get("slug") == entry.slug
            )
            assert outgoing + incoming == len(entry.related), (
                f"{entry.slug}: website would show {outgoing + incoming} edges, "
                f"catalogue.py has {len(entry.related)}"
            )

    def test_shares_records_edges_carry_a_note(self, entries):
        """A leakage warning with no explanation is not actionable."""
        silent = [
            (e.slug, link.slug)
            for e in entries
            for link in e.related
            if link.shares_records and not link.note.strip()
        ]
        assert not silent, f"shares_records edges without a note: {silent}"


class TestKnownRelationships:
    """The specific relationships verified against real data files."""

    def test_ecg_arrhythmia_contains_chapman_shaoxing(self, by_slug):
        entry = by_slug["chapman-shaoxing-arrhythmia"]
        link = next(
            x for x in entry.related
            if x.slug == "chapman-shaoxing-ecg-database-10-646-patients"
        )
        assert link.relation == "contains"
        assert link.shares_records is True
        assert link.verified is True

    def test_mimic_demo_is_a_subset_of_the_full_release(self, by_slug):
        link = next(
            x for x in by_slug["mimic-iv-ecg-demo"].related if x.slug == "mimic-iv-ecg"
        )
        # Declared on mimic-iv-ecg as "contains"; this side is derived.
        assert link.relation == "subset_of"
        assert link.derived is True
        assert link.verified is True
        assert "study_id" in link.note  # the join caveat must survive the inversion

    def test_ext_icd_is_derived_from_the_full_release(self, by_slug):
        link = next(
            x for x in by_slug["mimic-iv-ecg-ext-icd"].related if x.slug == "mimic-iv-ecg"
        )
        assert link.relation == "derived_from"
        assert link.verified is True

    def test_unverified_claims_are_flagged(self, by_slug):
        """Relationships taken from documentation must not claim verification."""
        link = next(x for x in by_slug["ptb-xl-plus"].related if x.slug == "ptb-xl")
        assert link.relation == "derived_from"
        assert link.verified is False

    def test_challenge_leakage_is_recorded(self, by_slug):
        """The reason this feature exists: CinC training sources overlap PTB-XL."""
        link = next(
            x for x in by_slug["ptb-xl"].related
            if x.slug == "physionet-cinc-challenge-2021"
        )
        assert link.relation == "subset_of"
        assert link.shares_records is True


class TestRelatedParsing:
    """Unit tests for the parser and the inversion, without touching the .md files."""

    def test_unknown_relation_is_reported(self):
        problems = []
        links = _parse_related("a", {"related": [{"slug": "b", "relation": "cousin_of"}]},
                               problems)
        assert links == []
        assert "unknown relation" in problems[0]

    def test_missing_slug_is_reported(self):
        problems = []
        _parse_related("a", {"related": [{"relation": "contains"}]}, problems)
        assert "needs a 'slug'" in problems[0]

    def test_self_link_is_reported(self):
        problems = []
        _parse_related("a", {"related": [{"slug": "a", "relation": "contains"}]}, problems)
        assert "points at itself" in problems[0]

    def test_non_list_related_is_reported(self):
        problems = []
        _parse_related("a", {"related": "b"}, problems)
        assert "must be a list" in problems[0]

    def test_inverse_edge_is_derived_with_metadata_carried_over(self):
        declared = {
            "a": [RelatedLink("b", "contains", shares_records=True, note="n", verified=True)],
            "b": [],
        }
        problems = []
        resolved = _with_reverse_edges(declared, problems)

        assert not problems
        (back,) = resolved["b"]
        assert back.slug == "a"
        assert back.relation == "subset_of"
        assert back.shares_records is True
        assert back.note == "n"
        assert back.verified is True
        assert back.derived is True

    def test_symmetric_relations_invert_to_themselves(self):
        declared = {"a": [RelatedLink("b", "same_cohort")], "b": []}
        resolved = _with_reverse_edges(declared, [])
        assert resolved["b"][0].relation == "same_cohort"

    def test_unknown_target_is_reported(self):
        problems = []
        _with_reverse_edges({"a": [RelatedLink("ghost", "contains")]}, problems)
        assert "unknown dataset" in problems[0]

    def test_explicit_reciprocal_is_not_duplicated(self):
        declared = {
            "a": [RelatedLink("b", "contains")],
            "b": [RelatedLink("a", "subset_of")],
        }
        resolved = _with_reverse_edges(declared, [])
        assert len(resolved["a"]) == 1
        assert len(resolved["b"]) == 1


class TestChallenge2021Relationships:
    """Challenge 2021 is a meta-dataset: its overlaps are the leakage risk."""

    def test_contains_ptb_xl_verified(self, by_slug):
        link = next(
            x for x in by_slug["physionet-cinc-challenge-2021"].related
            if x.slug == "ptb-xl"
        )
        assert link.relation == "contains"
        assert link.shares_records is True
        assert link.verified is True
        # The version detail is the actionable part: this cohort predates the 38
        # duplicates PTB-XL v1.0.3 removed.
        assert "21,837" in link.note

    def test_contains_the_physionet_chapman_ningbo_release_verified(self, by_slug):
        link = next(
            x for x in by_slug["physionet-cinc-challenge-2021"].related
            if x.slug == "chapman-shaoxing-arrhythmia"
        )
        assert link.relation == "contains"
        assert link.shares_records is True
        assert link.verified is True
        # The misnamed record is the trap a name join falls into.
        assert "S23074" in link.note

    def test_cohort_overlaps_all_declare_shared_records(self, by_slug):
        """Every source cohort with a catalogue entry must be flagged as shared."""
        related = {x.slug: x for x in by_slug["physionet-cinc-challenge-2021"].related}
        for slug in (
            "ptb-xl",
            "chapman-shaoxing-arrhythmia",
            "ptb-diagnostic-ecg-database",
            "st-petersburg-incart-12-lead-arrhythmia-database",
            "cpsc-2018-china-physiological-signal-challenge-2018",
        ):
            assert slug in related, f"{slug} is a source cohort but is not declared"
            assert related[slug].relation == "contains"
            assert related[slug].shares_records is True

    def test_cpsc_overlap_is_not_claimed_as_verified(self, by_slug):
        """It rests on a count match, not on a file comparison — say so."""
        link = next(
            x for x in by_slug["physionet-cinc-challenge-2021"].related
            if x.slug == "cpsc-2018-china-physiological-signal-challenge-2018"
        )
        assert link.verified is False

    def test_source_datasets_see_the_inverse_edge(self, by_slug):
        """A user starting from PTBDB must be warned too, via the derived inverse."""
        link = next(
            x for x in by_slug["ptb-diagnostic-ecg-database"].related
            if x.slug == "physionet-cinc-challenge-2021"
        )
        assert link.relation == "subset_of"
        assert link.derived is True
        assert link.shares_records is True
