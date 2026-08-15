"""
EDGAR — Experimental Data and Geometric Analysis Repository.

EDGAR is not one dataset. It is a portal of 33 posts holding 26 distinct
electrocardiographic-imaging experiments from ten institutions, each shipping
its own archive layout, its own MATLAB variable names, its own sampling rate
and, in two cases, its own idea of what a millivolt is. This module is what
turns that into one record table.

Nothing here is inferred from a shape or a filename at load time. Every
per-experiment fact — which archive is authoritative, which surface an
electrode array sat on, which orientation ``potvals`` is stored in, what unit
the samples are in — is written down in :data:`EXPERIMENTS` with the evidence
that established it, because every one of them has at least one counterexample
inside the release.

THE FOUR THINGS THAT WILL BITE ANYONE WORKING ON THIS FILE
==========================================================

**1. The portal cross-posts whole archives, and the wrong ones.** WordPress
stores one upload per filename per month, so a post linking a generically named
``Interventions.zip`` gets whichever dataset uploaded that name first. Verified
by SHA-256 over all 12,212 ``.mat`` members: the Valencia-pat2 post's
``Interventions.zip`` is byte-identical to Charles-PSTOV-pat3's 594 BSPM
recordings; the KIT ``TMV_FEM`` post's ``Docs``/``Interventions``/``Meshes``
zips are Dalhousie's; and every data link on the ``KIT-2020-SimVentrPacings``
post resolves to the KIT-20 *clinical* dataset, leaving that post with no data
of its own. :data:`EXPERIMENTS` therefore names one **uniquely titled** archive
per experiment. Those 20 archives cover the 2,945 real time-series matrices
exactly once each, with no overlap and nothing left over — which is the check
that the curation is complete, and it is asserted in
:func:`verify_archive_coverage`.

**2. Seven posts are re-publications of another post.** The 2016 archive and
the 2025 re-issue of the same experiment both appear in the portal listing.
:data:`DUPLICATE_POSTS` maps them; they contribute no records.

**3. ``potvals`` orientation is not recoverable from the shape.** EDGAR's
standard is (leads, samples) and 25 of the 26 experiments follow it, including
KIT's 2223-lead by 225-sample simulations. Dalhousie stores the transpose:
1142 samples by 120 leads, established from its own ``bad_leads`` field (lead
indices up to 120) and its ``avg_beats_mtx``, which is (beats, 120). So "leads
are the shorter axis" is false in both directions and the orientation travels
in the signal reference instead.

**4. Two experiments declare the wrong unit.** Valencia pat1 and pat2 set
``ECG.units = 'mV'`` on samples reaching 5350 — five volts on a body surface.
Their own ``Docs/Readme.txt`` says "the units are microV", which is what this
module applies, recording the disagreement in ``unit_source``. Everywhere else
the declaration and the README agree, or the README is the only statement.

WHAT COUNTS AS A RECORD
=======================

One 2-D potential matrix. EDGAR also stores derived maps in the same
``potvals`` field — 570 activation/recovery-interval maps of shape (leads, 3)
and 570 integral maps of shape (leads, 5), the latter mislabelled
``unit = 'ms'`` — and those are not time signals. They are excluded by
:data:`MIN_FRAMES`, which separates them cleanly: no derived map has more than
5 frames and no real recording has fewer than 171.

Reconstructed potentials are excluded too. Maastricht ships ``heartpots.mat``,
which its README states are "NOT measured, but reconstructed from the
body-surface potentials (with a Tikhonov zeroth order regularization method)".
Including an inverse solution as if it were a recording would put a method's
output into a benchmark of measurements.
"""

from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ecgbench.config import DatasetConfig

logger = logging.getLogger(__name__)

__all__ = [
    "EXPERIMENTS",
    "DUPLICATE_POSTS",
    "EXTRACT_DIRNAME",
    "MIN_FRAMES",
    "RECORDING_SURFACES",
    "extract_archives",
    "scan_records",
    "load_labels",
    "verify_archive_coverage",
]

#: Directory under the dataset root that :func:`extract_archives` unpacks into.
#: Signal paths in the metadata CSV are relative to the dataset root and start
#: here, so moving it renames every record's path.
EXTRACT_DIRNAME = "ecgbench_extracted"

#: Smallest number of time frames a matrix must have to count as a recording.
#: The gap it sits in is wide and empty: EDGAR's derived maps have 3 frames
#: (activation/recovery intervals) or 5 (QRS/QRST/ST/ST80/STT integrals), and
#: its shortest real recording has 171. Nothing in the release has 6 to 170.
MIN_FRAMES = 20

#: The physical surface an electrode array sat on. ``transmembrane`` is not a
#: potential at all — it is the simulated membrane voltage of KIT's TMV source
#: models, a different quantity from everything else here, and it is named
#: separately so nobody trains on it as though it were an electrogram.
RECORDING_SURFACES = (
    "torso",
    "epicardium",
    "endocardium",
    "intramural",
    "transmembrane",
)

#: Portal post -> the post it re-publishes. Both directories hold the same
#: files; only the target contributes records.
DUPLICATE_POSTS = {
    "dalhousie-2006-01-05": "bspm-endo-mapping-dalhousie-2006-01-05",
    "valencia_pat1_11-01-2014": (
        "afib-with-endo-and-bsp-recordings-valencia_pat1_11-01-2014"
    ),
    "valencia_pat2_11-30-2014": (
        "afib-with-endo-and-bsp-recordings-valencia_pat2_11-30-2014"
    ),
    "sim-extracellular-endo-and-epicardial-potentials-kit-20-pvc_simulation-"
    "1906-10-30_ep_endoepi": "kit-20-pvc_simulation-1906-10-30_ep_endoepi",
    "sim-extracellular-pericardial-sources-kit-20-pvc_simulation-1906-10-30_ep_peri": (
        "kit-20-pvc_simulation-1906-10-30_ep_peri"
    ),
    "sim-transmembrane-endo-and-epicardial-sources-kit-20-pvc_simulation-"
    "1906-10-30_tmv_endoepi": "kit-20-pvc_simulation-1906-10-30_tmv_endoepi",
    "sim-transmembrane-sources-throughout-fem-kit-20-pvc_simulation-"
    "1906-10-30_tmv_fem": "kit-20-pvc_simulation-1906-10-30_tmv_fem",
}

#: Posts that ship no time signals at all, and why. Kept so the count of 26
#: experiments reconciles with the 20 that produce records.
SIGNAL_FREE_POSTS = {
    "bratislava_2020_p034": (
        "The portal publishes only Documentation and Geometries for this "
        "subject. Its README describes Run1 (RV apex pacing) and Run2 "
        "(spontaneous PVCs) as 131-channel recordings, but neither is offered "
        "for download."
    ),
    "kit-2020-simventrpacings-fivesourcemodels": (
        "Every data link on this post resolves to the KIT-20 clinical "
        "dataset's archives (verified byte-identical). Only Documentation-8.pdf "
        "is unique to the post, so the five simulated source models it "
        "describes are not actually downloadable."
    ),
}


@dataclass(frozen=True)
class Experiment:
    """One EDGAR experiment, and everything about it a reader cannot infer."""

    slug: str
    title: str
    post: str
    archive: str
    subject_id: str
    species: str
    setting: str
    unit: str
    unit_source: str
    #: Path prefix -> (recording_surface, electrode_array). Matched against the
    #: record's path inside the archive, longest prefix first.
    surfaces: dict[str, tuple[str, str]]
    sampling_rate_hz: float | None = None
    orientation: str = "ls"
    #: Members whose path contains any of these are deliberately not this
    #: experiment's recordings — derived maps, reconstructions, or a shared set
    #: that another experiment owns. Distinct from "no rule matched", which is
    #: an error rather than a decision.
    exclude: tuple[str, ...] = ()
    carto_file: str | None = None
    notes: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def claims(self, relative: str) -> tuple[str, str] | None:
        """Return this record's (surface, array), or None if it is not ours.

        None covers both "explicitly excluded" and "no rule matched"; callers
        that need to tell them apart check :attr:`exclude` themselves.
        """
        if any(token in relative for token in self.exclude):
            return None
        return _classify(self, relative)


def _pace_dirs(prefix: str) -> dict[str, tuple[str, str]]:
    """Charles PSTOV and KIT clinical: every intervention is 120/63-lead BSPM."""
    return {prefix: ("torso", "BSPM electrode array")}


#: The 20 experiments that ship time signals, keyed by ECGBench's short slug.
EXPERIMENTS: dict[str, Experiment] = {
    # --- Human clinical: body-surface mapping during catheter pacing ---------
    "charles_pat1": Experiment(
        slug="charles_pat1",
        title="Multiple Ventricle Pacing Sites Pat#1 (Charles_PSTOV-12-07-27)",
        post="multiple-ventricle-pacing-sites-pat1-charles_pstov-12-07-27",
        archive="Interventions_Charles_12-07-27.zip",
        subject_id="charles_pstov_pat1",
        species="human",
        setting="human_clinical",
        unit="mV",
        unit_source="declared ts.unit",
        surfaces=_pace_dirs("Interventions/"),
        sampling_rate_hz=2000.0,
        carto_file="2012-07-27Subject_CARTOPacingSites.txt",
        notes=(
            "33 endocardial pacing sites (21 LV, 12 RV) in a healthy ventricle, "
            "120-electrode BSPM at 2 kHz, Prague. The README warns that pacing-site "
            "labels changed between releases; the folder number indexes the CARTO "
            "table shipped with this post, which is the correspondence used here."
        ),
    ),
    "charles_pat2": Experiment(
        slug="charles_pat2",
        title="Multiple Ventricle Pacing Sites Pat#2 (Charles_PSTOV-12-07-28)",
        post="multiple-ventricle-pacing-sites-pat2-charles_pstov-12-07-28",
        archive="Interventions_Charles_PSTOV-12-07-28.zip",
        subject_id="charles_pstov_pat2",
        species="human",
        setting="human_clinical",
        unit="mV",
        unit_source="declared ts.unit",
        surfaces=_pace_dirs("Interventions/"),
        sampling_rate_hz=2000.0,
        carto_file="CARTOPacingSites_PSTOV-12-07-28.txt",
        notes="21 pacing sites (14 LV, 7 RV), 120-electrode BSPM at 2 kHz.",
    ),
    "charles_pat3": Experiment(
        slug="charles_pat3",
        title="Multiple Ventricle Pacing Sites Pat#3 (Charles_PSTOV-12-07-29)",
        post="multiple-ventricle-pacing-sites-pat3-charles_pstov-12-07-29",
        archive="Charles_PSTOV-12-07-29.zip",
        subject_id="charles_pstov_pat3",
        species="human",
        setting="human_clinical",
        unit="mV",
        unit_source="declared ts.unit",
        surfaces=_pace_dirs("Interventions/"),
        sampling_rate_hz=2000.0,
        carto_file="CARTOPacingSites_PSTOV-12-07-29.txt",
        notes=(
            "22 pacing sites (17 LV, 5 RV), 120-electrode BSPM at 2 kHz. These same "
            "594 files are also served, byte-identical, from the Valencia-pat2 post's "
            "Interventions.zip — an upstream filename collision, not a second dataset."
        ),
    ),
    "kit20_clinical": Experiment(
        slug="kit20_clinical",
        title="Human PVC and Paced beats (KIT-20-PVC_Clinical_1906-10-30)",
        post="human-pvc-and-paced-beats-kit-20-pvc_clinical_1906-10-30",
        archive="KIT-20-PVC_Clinical_1906-10-30.zip",
        subject_id="kit_subject20",
        species="human",
        setting="human_clinical",
        unit="mV",
        unit_source="declared bspm.unit, corroborated by the README",
        surfaces={"Interventions/": ("torso", "63-electrode BSPM array")},
        sampling_rate_hz=1000.0,
        carto_file="Subject20_CARTOPacingSites_KIT-20-PVC_Clinical_1906-10-30.txt",
        notes=(
            "Seven CARTO-localised pacing sites plus spontaneous PVCs, 63-electrode "
            "BSPM at 1 kHz. Shares subject_id with the four KIT simulations, which "
            "were computed on this subject's anatomy — see the kit20_sim_* entries."
        ),
    ),
    "dalhousie_2006": Experiment(
        slug="dalhousie_2006",
        title="BSPM Endo Mapping (dalhousie-2006-01-05)",
        post="bspm-endo-mapping-dalhousie-2006-01-05",
        archive="Dalhousie-2006-01-05.zip",
        subject_id="dalhousie_6105",
        species="human",
        setting="human_clinical",
        unit="uV",
        unit_source="declared bspm.units",
        surfaces={"Interventions/": ("torso", "120-lead Horacek BSPM array")},
        sampling_rate_hz=2000.0,
        orientation="sl",
        # Interventions/carto_data holds per-point local activation times and
        # bipolar voltages from the CARTO system — scalars on a mesh, not
        # recordings, and stored under Interventions/ rather than Meshes/.
        exclude=("carto_data",),
        notes=(
            "THE ONE TRANSPOSED EXPERIMENT. potvals is (samples, 120), established "
            "from bad_leads (indices up to 120) and avg_beats_mtx (beats, 120). "
            "Recordings are signal-averaged beats, not raw runs, and 6105_DC_hdr.csv "
            "names the rhythm of each: sinus, paced, VT, VF, and two runs the "
            "clinician marked JUNK/DELETE."
        ),
    ),
    "valencia_pat1": Experiment(
        slug="valencia_pat1",
        title="Afib with Endo- and BSP Recordings (Valencia_pat1_11-01-2014)",
        post="afib-with-endo-and-bsp-recordings-valencia_pat1_11-01-2014",
        archive="Valencia_pat1_11-01-2014.zip",
        subject_id="valencia_pat1",
        species="human",
        setting="human_clinical",
        unit="uV",
        unit_source="README (file declares 'mV', contradicted — see module docstring)",
        surfaces={
            "Interventions/AV_block/ECG": ("torso", "54-electrode BSPM array"),
            "Interventions/AV_block/EGM": (
                "endocardium",
                "Constellation basket + tetrapolar catheters",
            ),
        },
        sampling_rate_hz=2034.5,
        notes=(
            "Atrial fibrillation mapped during an adenosine-induced AV block, so the "
            "body-surface signal is atrial with ventricular activity suppressed. "
            "2034.5 Hz — the only non-integer rate in the repository."
        ),
    ),
    "valencia_pat2": Experiment(
        slug="valencia_pat2",
        title="Afib with Endo- and BSP recordings (Valencia_pat2_11-30-2014)",
        post="afib-with-endo-and-bsp-recordings-valencia_pat2_11-30-2014",
        archive="Valencia_pat2_11-30-2014.zip",
        subject_id="valencia_pat2",
        species="human",
        setting="human_clinical",
        unit="uV",
        unit_source="README (file declares 'mV', contradicted — see module docstring)",
        surfaces={
            "Interventions/AV_block/ECG": ("torso", "54-electrode BSPM array"),
            "Interventions/AV_block/EGM": (
                "endocardium",
                "Constellation basket + tetrapolar catheters",
            ),
        },
        sampling_rate_hz=2034.5,
        notes=(
            "As pat1. This post's own Interventions.zip is NOT this experiment — it "
            "holds Charles-PSTOV-pat3 — so the full-dataset archive is the only "
            "trustworthy source, and it is what is used."
        ),
    ),
    "nijmegen_2004": Experiment(
        slug="nijmegen_2004",
        title="Normal BSPM from Noisy and MEG Shielded Rooms (Nijmegen-2004-12-09)",
        post="normal-bspm-from-noisy-and-meg-shielded-rooms-nijmegen-2004-12-09",
        archive="Nijmegen-2004-12-09.zip",
        subject_id="nijmegen_ppd2",
        species="human",
        setting="human_clinical",
        unit="mV",
        unit_source="inferred from amplitude (no declaration anywhere)",
        surfaces={"Interventions/": ("torso", "65-electrode BSPM array")},
        notes=(
            "One healthy subject recorded twice, in an ordinary room and in an "
            "MEG-shielded room, with normal breathing and expiratory breath-hold "
            "runs — the repository's only explicit noise comparison. Bare `pots` "
            "arrays with no struct, so no rate or unit is declared; the README says "
            "run 6 of the noisy-room set is corrupted."
        ),
    ),
    # --- Human clinical: EP Solutions vest, five patients -------------------
    **{
        f"epsol_{n}": Experiment(
            slug=f"epsol_{n}",
            title=f"EP_SOLUTIONS_pt_{n}",
            post=f"ep_solutions_pt_{n}",
            archive=archive,
            subject_id=f"ep_solutions_pt_{n}",
            species="human",
            setting="human_clinical",
            unit="mV",
            unit_source="README",
            surfaces={"Interventions/": ("torso", "EP Solutions multi-electrode vest")},
            sampling_rate_hz=1000.0,
            notes=(
                "One LV-paced and one RV-paced interval per patient, cut by hand "
                "between the end of the pacing artefact and the start of the T wave. "
                "Docs/MI.txt records whether the patient had a myocardial infarction."
            ),
        )
        for n, archive in (
            ("24", "024.zip"),
            ("26", "026.zip"),
            ("27", "027.zip"),
            ("33", "033.zip"),
            ("36", "036-9.zip"),
        )
    },
    # --- Torso tank -----------------------------------------------------------
    "utah_2002_cage": Experiment(
        slug="utah_2002_cage",
        title="Ischemia torso tank with cardiac cage (Utah-02-05-15)",
        post="ischemia-torso-tank-with-cardiac-cage-utah-02-05-15",
        archive="Utah-2002-05-15.zip",
        subject_id="utah_dog_2002_05_15",
        species="dog",
        setting="torso_tank",
        unit="mV",
        unit_source="declared ts.unit",
        surfaces={
            "Interventions/*/Torso": ("torso", "192-electrode tank array"),
            "Interventions/*/Cage": ("epicardium", "599-electrode cardiac cage"),
        },
        sampling_rate_hz=1000.0,
        notes=(
            "A Langendorff-perfused dog heart in an electrolytic torso tank, paced "
            "and made ischaemic by three coronary occlusions. The cage is a rigid "
            "599-electrode array surrounding the heart rather than a sock on it; "
            "electrode_array records that. Docs/rsm15may02_header.txt names each run."
        ),
    ),
    "utah_2010_sock": Experiment(
        slug="utah_2010_sock",
        title="Ischemia torso tank with sock and needles (Utah-10-03-02)",
        post="ischemia-torso-tank-with-sock-and-needles-utah-10-03-02",
        archive="Utah-10-03-02.zip",
        subject_id="utah_dog_2010_03_02",
        species="dog",
        setting="torso_tank",
        unit="mV",
        unit_source="README (3 mV ischaemia threshold); no declaration in the files",
        surfaces={
            "Interventions/*/Tank_": ("torso", "192-electrode tank array"),
            "Interventions/*/Sock_": ("epicardium", "247-electrode sock"),
            "Interventions/*/Needles_": ("intramural", "480 plunge-needle electrodes"),
        },
        sampling_rate_hz=1000.0,
        # _Intervention_Specific_Meshes holds ischaemic-region meshes that live
        # under Interventions/ rather than Meshes/ — geometry, not recordings.
        exclude=("_ARI", "_ITG", "_Intervention_Specific_Meshes"),
        notes=(
            "THE LARGEST EXPERIMENT: 190 runs recorded simultaneously on all three "
            "arrays, 570 recordings, across two demand-ischaemia and two "
            "supply-ischaemia interventions plus controls. The same directories hold "
            "570 activation/recovery-interval maps and 570 integral maps in the same "
            "`potvals` field; both are excluded, by directory suffix here and by "
            "MIN_FRAMES as a second guard."
        ),
    ),
    "utah_2018_tank": Experiment(
        slug="utah_2018_tank",
        title="Utah_2018_08_09_TorsoTank",
        post="utah_2018_08_09_torsotank",
        archive="AllFiles_toso_tank-1.zip",
        subject_id="utah_canine_2018_08_09",
        species="dog",
        setting="torso_tank",
        unit="mV",
        unit_source="inferred from amplitude (no declaration)",
        surfaces={
            "signals/torsoBeat": ("torso", "192-electrode tank array"),
            "signals/cageBeat": ("epicardium", "256-electrode cardiac cage"),
        },
        sampling_rate_hz=1000.0,
        notes=(
            "Three activation sequences (sinus, anterior- and posterior-paced), each "
            "recorded on tank and cage at once, preprocessed with PFEIFER and "
            "trimmed to a single QRST. Bad leads have been Laplacian-interpolated "
            "upstream, so `badLeads` marks channels that are estimates rather than "
            "measurements. The only experiment carrying hand-marked fiducials."
        ),
    ),
    "bordeaux_2016": Experiment(
        slug="bordeaux_2016",
        title="Bordeaux-2016-06-20-exp16",
        post="bordeaux-2016-06-20-exp16",
        archive="Bordeaux_2016-06-20-exp16.zip",
        subject_id="bordeaux_pig_exp16",
        species="pig",
        setting="torso_tank",
        unit="mV",
        unit_source="inferred from amplitude (no declaration)",
        surfaces={
            "Signals/*-vest": ("torso", "128-electrode tank vest"),
            "Signals/*-sock": ("epicardium", "108-electrode sock"),
        },
        sampling_rate_hz=2048.0,
        notes=(
            "Pig heart in a torso tank, sinus rhythm and LV/RV pacing after LBBB was "
            "induced by ablation. The longest recordings in the repository: 45,056 to "
            "55,296 samples, 22 to 27 s."
        ),
    ),
    # --- In-situ animal -------------------------------------------------------
    "maastricht_2015": Experiment(
        slug="maastricht_2015",
        title="Dog Torso and Epicardial recordings w/ Pacing (Maastricht-15-09-06)",
        post="dog-torso-and-epicardial-recordings-w-pacing-maastricht-15-09-06",
        archive="Maastricht-15-09-06.zip",
        subject_id="maastricht_dog2",
        species="dog",
        setting="insitu_animal",
        unit="uV",
        unit_source="README",
        surfaces={
            "Interventions/*/bodypots": ("torso", "body-surface electrodes"),
            "Interventions/*/heartleadpotentials": (
                "epicardium",
                "implanted epicardial electrodes",
            ),
        },
        sampling_rate_hz=2048.0,
        exclude=("heartpots",),
        notes=(
            "A sinus and an LV-apex-paced beat in a closed-chest healthy dog, "
            "recorded simultaneously on the body surface and on implanted epicardial "
            "electrodes. `heartpots.mat` is EXCLUDED: its README states those are "
            "reconstructed by Tikhonov regularisation, not measured. The same README "
            "flags an unresolved gain factor on the epicardial recordings, so their "
            "amplitudes should not be trusted in absolute terms."
        ),
    ),
    "auckland_2012": Experiment(
        slug="auckland_2012",
        title="Pig Torso, Epi-, Endocardial w/ Pacing (Auckland-2012-06-05)",
        post="pig-torso-epi-endocardial-w-pacing-auckland-2012-06-05",
        archive="Auckland-2012-06-05.zip",
        subject_id="auckland_pig_2012_06_05",
        species="pig",
        setting="insitu_animal",
        unit="mV",
        unit_source="README",
        surfaces={
            "Interventions/*/Torso": ("torso", "184 BioSemi carbon strip electrodes"),
            "Interventions/*/Epicardium": ("epicardium", "UnEmap 239-wire sock"),
            "Interventions/*/Endocardium": ("endocardium", "EnSite LV catheter"),
        },
        notes=(
            "Sinus rhythm and epicardial pacing in an open-chest pig, three arrays "
            "at once. Rates differ between arrays within the experiment — 2048 Hz "
            "for the BioSemi and UnEmap systems, 1200 Hz for the EnSite catheter — "
            "so the rate is read per file rather than declared here."
        ),
    ),
    # --- Simulation -----------------------------------------------------------
    "valencia_sim": Experiment(
        slug="valencia_sim",
        title="Simulation of Atrial Rotors (Valencia_sim_08-01-2014)",
        post="simulation-of-atrial-rotors-valencia_sim_08-01-2014",
        archive="Valencia_sim_08-01-2014.zip",
        subject_id="valencia_sim_08_01_2014",
        species="simulated",
        setting="simulation",
        unit="mV",
        unit_source="README",
        surfaces={
            "Interventions/*/ECG": ("torso", "771-node torso mesh"),
            "Interventions/*/EGM": ("endocardium", "2048-node atrial mesh"),
        },
        sampling_rate_hz=500.0,
        notes=(
            "Atrial fibrillation driven by a rotor in the left or right atrium, with "
            "and without fibrosis, from a modified Courtemanche membrane model; the "
            "torso signal is the forward solution. Three conditions, each as an "
            "atrial and a torso file."
        ),
    ),
    "kit20_sim_ep_endoepi": Experiment(
        slug="kit20_sim_ep_endoepi",
        title=(
            "Sim. Extracellular Endo- and Epicardial Potentials "
            "(KIT-20-PVC_Simulation-1906-10-30_EP_EndoEpi)"
        ),
        post="kit-20-pvc_simulation-1906-10-30_ep_endoepi",
        archive="KIT-20-PVC_Simulation-1906-10-30_EP_EndoEpi.zip",
        subject_id="kit_subject20",
        species="simulated",
        setting="simulation",
        unit="mV",
        unit_source="declared unit",
        surfaces={
            "Interventions/Simulation_Runs-BSPM": ("torso", "163-electrode BSPM array"),
            "Interventions/Simulation_Runs-EP_EndoEpi": (
                "epicardium",
                "502-node endocardial+epicardial mesh",
            ),
        },
        notes=(
            "Eight simulated ventricular pacing sites on KIT subject 20's anatomy. "
            "This archive is the only one of the four KIT simulation posts that "
            "bundles the shared 8-run BSPM set, so the body-surface half of the "
            "family is attributed here; the other three posts link the identical "
            "Simulation_Runs-BSPM.zip. SUBJECT 20 IS ALSO kit20_clinical's SUBJECT, "
            "so all five share one patient group."
        ),
    ),
    "kit20_sim_ep_peri": Experiment(
        slug="kit20_sim_ep_peri",
        title=(
            "Sim. Extracellular Pericardial Sources "
            "(KIT-20-PVC_Simulation-1906-10-30_EP_Peri)"
        ),
        post="kit-20-pvc_simulation-1906-10-30_ep_peri",
        archive="KIT-20-PVC_Simulation-1906-10-30_EP_Peri.zip",
        subject_id="kit_subject20",
        species="simulated",
        setting="simulation",
        unit="mV",
        unit_source="declared unit",
        surfaces={
            "Interventions/Simulation_Runs-EP_Peri": (
                "epicardium",
                "502-node pericardial mesh",
            )
        },
        exclude=("Simulation_Runs-BSPM",),
        notes=(
            "The same eight pacing sites as a pericardial extracellular source model. "
            "This archive also bundles the family's shared BSPM set, which "
            "kit20_sim_ep_endoepi owns; it is excluded here rather than counted twice."
        ),
    ),
    "kit20_sim_tmv_endoepi": Experiment(
        slug="kit20_sim_tmv_endoepi",
        title=(
            "Sim. Transmembrane Endo- and Epicardial Sources "
            "(KIT-20-PVC_Simulation-1906-10-30_TMV_EndoEpi)"
        ),
        post="kit-20-pvc_simulation-1906-10-30_tmv_endoepi",
        archive="KIT-20-PVC_Simulation-1906-10-30_TMV_EndoEpi.zip",
        subject_id="kit_subject20",
        species="simulated",
        setting="simulation",
        unit="mV",
        unit_source="declared unit",
        surfaces={
            "Interventions/Simulation_Runs-TMV_EndoEpi": (
                "transmembrane",
                "502-node endocardial+epicardial mesh",
            )
        },
        exclude=("Simulation_Runs-BSPM", "Simulation_Runs-EP_EndoEpi"),
        notes=(
            "TRANSMEMBRANE VOLTAGES, NOT POTENTIALS. These are membrane voltages on "
            "the source mesh, a different physical quantity from every other record "
            "here, which is why recording_surface has a value of its own for them."
        ),
    ),
    "kit20_sim_tmv_fem": Experiment(
        slug="kit20_sim_tmv_fem",
        title=(
            "Sim. Transmembrane Sources Throughout FEM "
            "(KIT-20-PVC_Simulation-1906-10-30_TMV_FEM)"
        ),
        post="kit-20-pvc_simulation-1906-10-30_tmv_fem",
        archive="KIT-20-PVC_Simulation-1906-10-30_TMV_FEM.zip",
        subject_id="kit_subject20",
        species="simulated",
        setting="simulation",
        unit="mV",
        unit_source="declared unit",
        surfaces={
            "Interventions/Simulation_Runs-TMV_FEM": (
                "transmembrane",
                "2223-node ventricular volume mesh",
            )
        },
        exclude=("Simulation_Runs-BSPM",),
        notes=(
            "The same eight pacing sites throughout a finite-element volume: 2223 "
            "nodes by 200-272 frames, the one place in EDGAR where leads outnumber "
            "samples by an order of magnitude. Any shape-based orientation guess "
            "gets this experiment backwards."
        ),
    ),
}

#: Variables holding potentials as a bare array rather than an EDGAR struct.
#: Only these are accepted, because a forward-transfer matrix is also a 2-D
#: numeric array and would otherwise scan as a recording.
BARE_ARRAY_VARIABLES = {"pots", "lichaampots", "heartleadpots"}

#: Top-level directories inside an archive that hold recordings. Everything
#: else (Meshes, Registration, FwdInvTransforms, Images, Docs) is skipped
#: before any file is opened.
SIGNAL_DIRS = ("interventions", "signals")


def _iter_members(archive: Path):
    """Yield (normalised path, ZipInfo) for the real members of an archive.

    Two normalisations, both of which some archives need and some do not: a
    single shared root directory is stripped (EP Solutions ships ``024/...``,
    KIT ships ``KIT-20-PVC_Simulation-.../...``), and Apple's ``__MACOSX``
    resource forks are dropped.
    """
    with zipfile.ZipFile(archive) as handle:
        members = [
            info
            for info in handle.infolist()
            if not info.is_dir() and "__MACOSX" not in info.filename
        ]
        roots = {m.filename.split("/")[0] for m in members if "/" in m.filename}
        strip = (
            len(roots) == 1
            and all("/" in m.filename for m in members)
            and next(iter(roots)).lower() not in SIGNAL_DIRS
        )
        for info in members:
            name = info.filename.split("/", 1)[1] if strip else info.filename
            yield name, info


def _record_members(data_path: Path, exp: Experiment):
    """Yield (relative path, ZipInfo, surface, array) for one experiment's records.

    The single place that decides what belongs to an experiment, so extraction,
    the coverage check and the scan cannot disagree about it. Anything under a
    signal directory that is neither excluded nor matched by a surface rule
    raises — an unrecognised file is a curation gap, not something to skip.
    """
    from ecgbench.labels import LabelSourceMissingError

    archive = data_path / exp.post / exp.archive
    if not archive.exists():
        raise LabelSourceMissingError(
            f"EDGAR archive {archive} is missing. It is the authoritative source for "
            f"experiment '{exp.slug}' — see ecgbench.labels.edgar.EXPERIMENTS for why "
            "the other archives on that post are not interchangeable. Download it "
            "from https://edgar.sci.utah.edu/ (free registration)."
        )
    unmatched = []
    for name, info in _iter_members(archive):
        if not name.lower().endswith(".mat"):
            continue
        if name.split("/")[0].lower() not in SIGNAL_DIRS:
            continue
        claimed = exp.claims(name)
        if claimed is None:
            if not any(token in name for token in exp.exclude):
                unmatched.append(name)
            continue
        yield name, info, claimed[0], claimed[1]
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} members of {exp.archive} match no surface rule for "
            f"experiment '{exp.slug}', e.g. {unmatched[:3]}. Add a rule to its "
            "`surfaces` map, or list it in `exclude` if it is not a recording — do "
            "not let a file be filed under a surface it was not recorded on."
        )


def extract_archives(data_path: Path, experiments: dict[str, Experiment] | None = None):
    """Unpack each experiment's authoritative archive under ``EXTRACT_DIRNAME``.

    EDGAR ships everything inside zips, and ECGBench reads signals as files, so
    this has to happen once before anything else. Only ``.mat`` members under a
    signal directory are written — the CT and MRI volumes are 7,524 DICOM files
    and 10 of the repository's 11 GB, and nothing in the pipeline reads them.

    Idempotent: a member already on disk with the right size is left alone, so
    a re-run costs one pass over the zip directories.
    """
    experiments = EXPERIMENTS if experiments is None else experiments
    root = data_path / EXTRACT_DIRNAME
    written = 0
    for exp in experiments.values():
        target_root = root / exp.slug
        with zipfile.ZipFile(data_path / exp.post / exp.archive) as handle:
            for name, info, _surface, _array in _record_members(data_path, exp):
                target = target_root / name
                if target.exists() and target.stat().st_size == info.file_size:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(handle.read(info.filename))
                written += 1
        logger.info("extracted %s from %s", exp.slug, exp.archive)
    logger.info("EDGAR extraction complete: %d new files under %s", written, root)
    return root


def verify_archive_coverage(data_path: Path) -> pd.DataFrame:
    """Check that the curated archives cover every recording exactly once.

    Returns one row per (experiment, archive) with the number of ``.mat``
    members under a signal directory, and raises if any two experiments claim
    the same file content. This is the guard on :data:`EXPERIMENTS`: the portal
    serves the same archive from several posts, so a curation mistake shows up
    as an overlap rather than as an error.
    """
    import hashlib

    seen: dict[str, str] = {}
    rows = []
    for exp in EXPERIMENTS.values():
        count = 0
        with zipfile.ZipFile(data_path / exp.post / exp.archive) as handle:
            for name, info, _surface, _array in _record_members(data_path, exp):
                digest = hashlib.sha256(handle.read(info.filename)).hexdigest()
                owner = seen.setdefault(digest, exp.slug)
                if owner != exp.slug:
                    raise ValueError(
                        f"EDGAR archive curation overlaps: {name} of '{exp.slug}' is "
                        f"byte-identical to a member already claimed by '{owner}'. "
                        "Two experiments cannot own the same recording."
                    )
                count += 1
        rows.append({"experiment": exp.slug, "archive": exp.archive, "n_mat": count})
    return pd.DataFrame(rows)


def _classify(exp: Experiment, relative: str) -> tuple[str, str] | None:
    """Resolve a record's path to (recording_surface, electrode_array).

    Patterns are literal path prefixes, optionally with a single ``*`` standing
    for one path component, and are tried longest-first so a more specific rule
    wins. Returns None for a path no rule claims, which the caller reports
    rather than guessing at.
    """
    for pattern in sorted(exp.surfaces, key=len, reverse=True):
        regex = "^" + ".*?".join(re.escape(p) for p in pattern.split("*"))
        if re.match(regex, relative):
            return exp.surfaces[pattern]
    return None


def _read_carto_sites(path: Path) -> dict[tuple[str, int], tuple[float, float, float]]:
    """Parse a CARTO pacing-site table into (chamber, index) -> (x, y, z).

    The tables are hand-formatted text with an ``LV:``/``RV:`` section header, a
    column header line, and whitespace-separated rows. KIT's has no chamber
    header at all, so its sites are keyed under an empty chamber.
    """
    sites: dict[tuple[str, int], tuple[float, float, float]] = {}
    chamber = ""
    for line in path.read_text(errors="replace").splitlines():
        text = line.strip()
        if not text:
            continue
        head = text.split()[0].rstrip(":").upper()
        if head in {"LV", "RV"} and len(text.split()) == 1:
            chamber = head
            continue
        parts = text.split()
        if len(parts) == 4 and parts[0].isdigit():
            try:
                sites[(chamber, int(parts[0]))] = tuple(float(v) for v in parts[1:])
            except ValueError:
                continue
    return sites


#: Intervention directory -> (chamber, site index), for the pacing experiments.
_PACE_DIR = re.compile(r"intervention(Left|Right)VentPace(\d+)", re.I)
_KIT_PACE_DIR = re.compile(r"InterventionPace(\d+)", re.I)


def _pacing_site(exp: Experiment, relative: str) -> tuple[str, int | None]:
    """Return (chamber, site index) named by the record's intervention folder."""
    match = _PACE_DIR.search(relative)
    if match:
        return ("LV" if match.group(1).lower() == "left" else "RV", int(match.group(2)))
    match = _KIT_PACE_DIR.search(relative)
    if match:
        return ("", int(match.group(1)))
    return ("", None)


def _potential_matrices(path: Path) -> list[tuple[str, np.ndarray, dict]]:
    """Return [(variable, matrix, attributes)] for one .mat file.

    Accepts EDGAR's struct form (any variable with a ``potvals`` field) and the
    two contributors who ship a bare array, but only under the names in
    :data:`BARE_ARRAY_VARIABLES` — a forward-transfer matrix is also a 2-D
    numeric array and must not scan as a recording.
    """
    import scipy.io

    contents = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=False)
    out = []
    for name, value in contents.items():
        if name.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and value.dtype == object and value.size:
            member = value.flat[0]
            if not hasattr(member, "potvals"):
                continue
            attributes = {}
            for source, key in (
                ("samplefrequency", "fs"),
                ("sampleFreq", "fs"),
                ("sampling", "fs"),
                ("fs", "fs"),
                ("unit", "declared_unit"),
                ("units", "declared_unit"),
                ("label", "label"),
                ("text", "label"),
                ("gain", "gain"),
            ):
                if hasattr(member, source) and key not in attributes:
                    raw = np.asarray(getattr(member, source)).ravel()
                    if raw.size:
                        attributes[key] = raw[0]
            for source in ("leadinfo", "badLeads", "bad_leads"):
                if hasattr(member, source):
                    marks = np.asarray(getattr(member, source)).ravel()
                    attributes["n_bad_leads"] = int(
                        marks.size if source == "bad_leads" else np.count_nonzero(marks)
                    )
                    break
            out.append((name, np.asarray(member.potvals), attributes))
        elif name in BARE_ARRAY_VARIABLES and isinstance(value, np.ndarray):
            out.append((name, value, {}))
    return out


_RATE_TEXT = re.compile(r"([\d.]+)\s*k?hz", re.I)


def _parse_rate(value) -> float | None:
    """Read a sampling rate that may be a number or free text ('2kHz', '2048 Hz')."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    match = _RATE_TEXT.search(str(value))
    if not match:
        return None
    rate = float(match.group(1))
    return rate * 1000 if "k" in str(value).lower() else rate


def scan_records(data_path: Path) -> pd.DataFrame:
    """Walk the extracted tree and build one row per potential recording.

    Every file is opened, because nothing outside it declares its shape, and the
    derived maps that share the ``potvals`` field can only be told from
    recordings by their frame count.
    """
    from ecgbench.labels import LabelSourceMissingError

    root = data_path / EXTRACT_DIRNAME
    rows: list[dict] = []
    unclassified: list[str] = []

    for slug, exp in EXPERIMENTS.items():
        base = root / slug
        if not base.is_dir():
            raise LabelSourceMissingError(
                f"{base} does not exist — run ecgbench.labels.edgar.extract_archives() "
                "first (the splitter does this for you)."
            )
        carto = {}
        if exp.carto_file:
            carto_path = data_path / exp.post / exp.carto_file
            if carto_path.exists():
                carto = _read_carto_sites(carto_path)
            else:
                logger.warning("CARTO sites missing for %s: %s", slug, carto_path)

        for path in sorted(base.rglob("*.mat")):
            relative = path.relative_to(base).as_posix()
            if any(token in relative for token in exp.exclude):
                continue
            surface = _classify(exp, relative)
            if surface is None:
                unclassified.append(f"{slug}/{relative}")
                continue
            recording_surface, electrode_array = surface

            matrices = _potential_matrices(path)
            for variable, matrix, attributes in matrices:
                if matrix.ndim != 2:
                    continue
                n_leads, n_frames = (
                    (matrix.shape[1], matrix.shape[0])
                    if exp.orientation == "sl"
                    else matrix.shape
                )
                if n_frames < MIN_FRAMES:
                    continue  # activation/recovery or integral map, not a recording

                rate = _parse_rate(attributes.get("fs")) or exp.sampling_rate_hz
                chamber, site = _pacing_site(exp, relative)
                coordinates = carto.get((chamber, site)) if site is not None else None

                stem = relative[: -len(".mat")].replace("/", "_")
                record_id = f"{slug}__{stem}"
                if len(matrices) > 1:
                    record_id = f"{record_id}#{variable}"
                reference = (
                    f"{EXTRACT_DIRNAME}/{slug}/{relative}"
                    f":{variable}:{exp.orientation}:{exp.unit}"
                )
                rows.append(
                    {
                        "record_id": record_id,
                        "experiment": slug,
                        "experiment_title": exp.title,
                        "portal_post": exp.post,
                        "subject_id": exp.subject_id,
                        "species": exp.species,
                        "setting": exp.setting,
                        "recording_surface": recording_surface,
                        "electrode_array": electrode_array,
                        "intervention": relative.split("/")[1] if "/" in relative else "",
                        "pacing_chamber": chamber,
                        "pacing_site": site,
                        "pacing_site_x": coordinates[0] if coordinates else None,
                        "pacing_site_y": coordinates[1] if coordinates else None,
                        "pacing_site_z": coordinates[2] if coordinates else None,
                        "n_leads": int(n_leads),
                        "n_samples": int(n_frames),
                        "sampling_rate_hz": rate,
                        "duration_s": (n_frames / rate) if rate else None,
                        "unit_applied": exp.unit,
                        "declared_unit": _text(attributes.get("declared_unit")),
                        "unit_source": exp.unit_source,
                        "orientation": exp.orientation,
                        "matlab_variable": variable,
                        "n_bad_leads": attributes.get("n_bad_leads"),
                        "source_label": _text(attributes.get("label")),
                        "signal_path": reference,
                    }
                )

    if unclassified:
        raise ValueError(
            f"{len(unclassified)} extracted EDGAR files match no surface rule, e.g. "
            f"{unclassified[:3]}. Add a rule to the experiment's `surfaces` map in "
            "ecgbench.labels.edgar rather than letting a recording be filed under a "
            "surface it was not recorded on."
        )

    frame = pd.DataFrame(rows).sort_values("record_id", ignore_index=True)
    logger.info(
        "EDGAR scan: %d recordings across %d experiments and %d subjects",
        len(frame),
        frame["experiment"].nunique(),
        frame["subject_id"].nunique(),
    )
    return frame


def _text(value) -> str | None:
    """Trim a MATLAB char field, which is often NUL-padded to a fixed width."""
    if value is None:
        return None
    text = str(value).replace("\x00", "").strip()
    return text or None


def load_labels(data_path: Path, config: DatasetConfig) -> pd.DataFrame:
    """Return every EDGAR record's metadata, indexed by record ID.

    There is no label file to read: EDGAR ships one README per experiment and
    nothing tabular, so the "labels" are the experimental facts this module
    curates plus what each recording's own MATLAB struct declares. The
    ground-truth quantity users actually train on is the pacing site — for the
    four pacing experiments those coordinates come from the CARTO tables and
    appear as ``pacing_site_x/y/z``.
    """
    data_path = Path(data_path)
    cache = data_path / config.metadata_csv
    if cache.exists():
        return pd.read_csv(cache, dtype=_STRING_COLUMNS).set_index("record_id")
    extract_archives(data_path)
    return scan_records(data_path).set_index("record_id")


#: Columns pandas must not retype on the CSV round trip. `pacing_site` stays
#: numeric deliberately — it is an index into the CARTO table, not an
#: identifier — but everything used to build a path or a group is a string.
_STRING_COLUMNS = {
    "record_id": str,
    "experiment": str,
    "subject_id": str,
    "species": str,
    "setting": str,
    "recording_surface": str,
    "electrode_array": str,
    "intervention": str,
    "pacing_chamber": str,
    "matlab_variable": str,
    "orientation": str,
    "unit_applied": str,
    "signal_path": str,
}
