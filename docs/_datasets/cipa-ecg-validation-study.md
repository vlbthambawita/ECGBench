---
slug: "cipa-ecg-validation-study"
name: "CiPA ECG Validation Study"
category: "12-lead-physionet"
order: 16
status: "completed"
source_url: "https://physionet.org/content/ecgcipa/1.0.0/"
url_label: "physionet.org"
format: "12-lead · 10 s · 1000 Hz · + derived median beats"
patients: "60"
records: "5,749 segments"
access: "open"
license: "ODC Attribution"
origin_institution: "US FDA / CDER; Phase I Clinical Pharmacology Study (NCT03070470)"
origin_country: "USA"
leads: 12
paper_title: "Clin Pharmacol Ther, 2019"
paper_doi: "https://doi.org/10.1002/cpt.1303"
search_keywords: "cipa ecg validation usa pharmacology clinical trial qt jtpeak dofetilide ranolazine verapamil chloroquine lopinavir ritonavir diltiazem hERG proarrhythmia exposure response 1000 hz"

related:
  - slug: "ecg-effects-of-ranolazine-dofetilide-verapamil-quinidine-ecgrdvq"
    relation: "sibling_release"
    shares_records: false
    verified: false
    note: >
      Same research programme, same file layout (adeg/adpc/adsl/addm plus raw/ and
      medians/), and three overlapping drugs — ranolazine, dofetilide and verapamil.
      Different trial and different volunteers: ECGRDVQ is 22 subjects under a
      separate protocol, CiPA is 60 under NCT03070470, so no recording is shared.
      Unverified against the files — taken from the two PhysioNet pages and their
      ClinicalTrials.gov identifiers. **Their subject IDs collide anyway**: both
      releases number subjects from 1001 upward, so pooling them requires prefixing
      the subject ID with the dataset, or the two cohorts merge into one apparent
      set of people and patient-level grouping silently breaks.
  - slug: "ecg-effects-of-dofetilide-moxifloxacin-and-combinations-ecgdmmld"
    relation: "sibling_release"
    shares_records: false
    verified: false
    note: >
      Same programme and layout again, sharing dofetilide as the positive control.
      ECGDMMLD is 22 subjects under NCT02308748 at 500 Hz upsampled to 1 kHz; CiPA
      is 60 subjects under NCT03070470 recorded at 1 kHz. No shared recordings.
      Unverified against the files — taken from the PhysioNet pages. The same
      subject-ID collision applies: both releases start numbering at 1001, so
      prefix the subject ID before pooling them.

sections:
  - type: description
    title: "Overview"
    body: |
      5,749 ten-second 12-lead ECGs at **1 kHz** from **60 healthy volunteers** in a
      double-blind randomised Phase I trial (study SCR-004, ClinicalTrials.gov
      **NCT03070470**) designed by the US FDA's Center for Drug Evaluation and
      Research.

      The study exists to validate the **CiPA** hypothesis: that measuring QTc
      *together with* **J-Tpeakc** in an ordinary Phase I study can tell a
      **"balanced" ion-channel blocker** — one that blocks hERG but also blocks late
      sodium or L-type calcium current at comparable potency, and so carries little
      proarrhythmic risk — apart from a **"predominant hERG" blocker**, which does
      not. QTc alone cannot: both prolong it.

      - **Part 1** — 50 subjects (`1001`–`1050`), parallel design, ten per group,
        three consecutive days of **ranolazine**, **verapamil**,
        **lopinavir+ritonavir**, **chloroquine** or **placebo**.
      - **Part 2** — 10 subjects (`2001`–`2010`), two-period crossover of
        **dofetilide** against **diltiazem+dofetilide**, dosed on days 1–3 and 8–10.

      Records were extracted **in triplicate** at up to 28 nominal timepoints from
      one hour before dosing to 73 hours after, and each is published **twice**: as
      the raw 10 s segment and as a derived 16-channel median beat carrying
      semi-automatic fiducial annotations.

      **This is a pharmacology dataset, not a diagnosis dataset.** Every participant
      was screened to exclude cardiac disease. There is no rhythm, morphology or
      arrhythmia label anywhere in it, and none of the usual ECGBench habits built
      around a diagnostic class transfers — starting with stratification, which uses
      the drug in effect.

  - type: table
    title: "Treatment arms — the stratification label"
    headers: ["Treatment in effect", "Subjects", "Records", "Share", "Ion-channel profile"]
    rows:
      - ["Chloroquine", "10", "840", "14.6%", "predominant hERG"]
      - ["Dofetilide", "10", "840", "14.6%", "predominant hERG (positive control)"]
      - ["Lopinavir+Ritonavir", "10", "840", "14.6%", "predominant hERG"]
      - ["Ranolazine", "10", "840", "14.6%", "balanced (hERG + late sodium)"]
      - ["Placebo", "10", "835", "14.5%", "control"]
      - ["Verapamil", "10", "780", "13.6%", "balanced (hERG + L-type calcium)"]
      - ["Diltiazem+Dofetilide", "10", "774", "13.5%", "hERG block + added calcium block"]
      - ["**Total**", "**60**", "**5,749**", "**100%**", "7 values of `treatment` (adeg `TRTA`)"]

  - type: description
    title: "5,749 records are not 5,749 observations"
    body: |
      Two levels of clustering sit between the record count and anything you could
      call an independent sample, and both matter more here than in most datasets.

      **Triplicates.** Three ten-second segments were extracted per subject per
      nominal timepoint — **1,916 of the 1,917 timepoint groups hold exactly 3
      records** (one holds 1). They are the same person, in the same posture, at the
      same plasma concentration, seconds apart: near-duplicates, not repeats of an
      experiment. **The effective sample size is closer to 1,917 than to 5,749.**

      **Subjects.** Those 1,917 groups come from 60 people, **24 to 168 records
      each** (median 84), because subjects completed different numbers of
      timepoints — most 28 per period, but one only 8. So every per-record statistic
      is weighted by trial compliance rather than by person.

      Folds are grouped on `patient_id`, which handles both at once: verified on the
      shipped release, **no subject spans two folds and no timepoint triplicate is
      ever split**. Group your own analyses the same way, and weight by subject
      before quoting any rate.

  - type: description
    title: "Every record ships twice — raw, and a derived median beat"
    body: |
      | Directory | Contents | Channels | Samples | ECGBench column |
      |---|---|---|---|---|
      | `raw/<subject>/<uuid>` | the 10 s acquisition | 12 | 10,000 | `signal_path` — **the dataset's signal** |
      | `medians/<subject>/<uuid>` | derived representative median beat | 16 | 1,200 | `median_beat_path` (labels only) |

      The median beat adds the **vector-magnitude lead `VCGMAG`** and the Frank
      **`X`, `Y`, `Z`** components to the 12, and carries a `.atr` annotation file.
      Both directories are 1 kHz and use the same record ID, so they are two
      representations of one acquisition, not two records.

      **The medians get no fold of their own, deliberately.** Every median beat is a
      derivation of a raw record ECGBench already partitions; generating a second
      ten-fold split over the same recordings would let someone train on one and
      evaluate on the other. `ECGDataset` therefore always reads `raw/`, and the
      median beats reach you as paths plus a fiducial loader.

      Their headers *look* corrupt — `6276255.687397709(-1227133513)/uV` — and are
      not: each channel is scaled to fill the int32 range, and `wfdb.rdrecord`
      recovers physiologic microvolts. Spot-checked against the raw records, median
      lead II peak-to-peak comes out at **0.87–0.93×** the raw value, which is what
      median-beat averaging predicts.

      ```python
      from ecgbench import load_config
      from ecgbench.labels.ecgcipa import load_median_beat_fiducials

      config = load_config("ecgcipa")
      fid = load_median_beat_fiducials("/path/to/ecgcipa/1.0.0/", config)
      fid.loc["00689D31-8491-4643-B3C8-45241FBBD47C"]
      # p_onset_ms 165, qrs_onset_ms 353, qrs_peak_ms 396,
      # qrs_offset_ms 431, t_peak_ms 663, t_offset_ms 724
      ```

      **The fiducials and the interval table are not independent measurements.** The
      annotations are what the published intervals were measured *from*, and they
      reproduce `adeg.csv` exactly: across **all 5,749 records**, every one of PR,
      QRS, QT, J-Tpeak and Tpeak-Tend recomputed from the fiducials equals the
      published value **to the millisecond** (for the record above: 353−165 = PR
      188, 431−353 = QRS 78, 724−353 = QT 371, 663−431 = J-Tpeak 232, 724−663 =
      Tpeak-Tend 61). Treat agreement between them as a format check, never as
      corroboration.

  - type: table
    title: "Per-record labels (49 columns) — no class among them"
    headers: ["Group", "Columns", "Notes"]
    rows:
      - ["Identity & paths", "`patient_id`, `signal_path`, `median_beat_path`, `study_id`", "the record ID is the `EGREFID` UUID, unique across the release"]
      - ["Drug exposure", "`treatment`, `planned_treatment`, `treatment_sequence`, `planned_arm`, `actual_arm`", "`treatment` is the treatment **in effect** for that record; the 10 crossover subjects carry two"]
      - ["Timing", "`period`, `period_label`, `timepoint`, `timepoint_n`, `nominal_hours_from_period_start`, `nominal_hours_from_reference`, `actual_hours_from_reference`, `study_day`, `period_day`, `acquisition_datetime`", "**two different clocks** — see the units section"]
      - ["Provenance", "`replicate_number`, `replicate_number_inconsistent`, `used_for_baseline`, `has_matching_pk`", "replicate 1–3 within a timepoint"]
      - ["Intervals (ms; HR in bpm)", "`hr_bpm`, `rr_ms`, `pr_ms`, `qrs_ms`, `qt_ms`, `qtcf_ms`, `jtpeak_ms`, `jtpeakc_ms`, `tpeak_tend_ms`", "measured from the median beat's fiducials; **19 records are incomplete**"]
      - ["Plasma concentration", "`plasma_{ranolazine,verapamil,lopinavir,ritonavir,chloroquine,diltiazem}_ng_ml`, `plasma_dofetilide_pg_ml`, `plasma_below_lloq`, `plasma_any_below_lloq`", "**dofetilide is pg/mL, the other six ng/mL**; joins for 4,371 of 5,749 records"]
      - ["Subject", "`age_years`, `sex`, `race`, `ethnicity`, `height_cm`, `weight_kg`, `bmi_kg_m2`, `systolic_bp_mmhg`, `diastolic_bp_mmhg`", "constant within a subject, so repeated across their 24–168 records"]

  - type: table
    title: "Interval measurements over all 5,749 records"
    headers: ["Parameter", "Mean", "SD", "Min", "Max", "Missing"]
    rows:
      - ["`hr_bpm` (HR, bpm)", "67.3", "9.9", "38", "98", "0"]
      - ["`rr_ms` (RR)", "911.7", "141.4", "613.1", "1593.8", "0"]
      - ["`pr_ms` (PR)", "173.0", "23.2", "112", "324", "**10**"]
      - ["`qrs_ms` (QRS)", "87.9", "8.5", "67", "112", "0"]
      - ["`qt_ms` (QT)", "380.8", "27.2", "315", "515", "**9**"]
      - ["`qtcf_ms` (QTcF)", "394.1", "25.8", "332.4", "582.6", "**9**"]
      - ["`jtpeak_ms` (J-Tpeak)", "213.1", "26.2", "133", "309", "**9**"]
      - ["`jtpeakc_ms` (J-Tpeakc)", "226.0", "25.4", "149.7", "316.4", "**9**"]
      - ["`tpeak_tend_ms` (Tpeak-Tend)", "79.9", "18.4", "51", "196", "**9**"]

  - type: description
    title: "The trap: the study's own endpoints cannot be attached to a waveform"
    body: |
      `adeg.csv` holds **69,556 rows, and only 51,686 belong to a record**. The other
      **17,870 carry `DTYPE=AVERAGE` and a blank `EGREFID`** — they are the mean over
      each triplicate, and they are the **only** rows where the study's actual
      endpoints exist:

      | Column | On per-record rows | On AVERAGE rows |
      |---|---|---|
      | `AVAL` (absolute interval) | 51,686 | 17,870 |
      | `BASE` (baseline) | **0** | 15,989 |
      | `CHG` (change from baseline) | **0** | 15,989 |
      | `CCOMPCHG` (placebo-corrected change) | **0** | 13,847 |
      | `ABLFL` (baseline record flag) | **0** | 630 |

      The published exposure-response analysis — ΔQTcF and ΔJ-Tpeakc against
      concentration — therefore lives **one level of aggregation above the signals**,
      and no join can bring it down. ECGBench exposes the per-record **absolute**
      intervals, and the averages separately, keyed by
      `(patient_id, period, timepoint_n)`:

      ```python
      from ecgbench import load_config
      from ecgbench.labels.ecgcipa import load_triplicate_averages

      avg = load_triplicate_averages("/path/to/ecgcipa/1.0.0/", load_config("ecgcipa"))
      len(avg)                                           # 17870
      qtcf = avg[(avg["parameter"] == "QTCF") & avg["CHG"].notna()]
      len(qtcf), round(qtcf["CHG"].max(), 1)             # (1776, 171.5)
      ```

      If you want a per-record endpoint, build it yourself from `used_for_baseline`
      (627 pre-dose records) and say so — it will not equal the published figures,
      which average the triplicate first.

  - type: table
    title: "What the dataset is for: QTcF and J-Tpeakc against exposure"
    headers: ["Arm", "QTcF pre-dose", "QTcF at Cmax", "ΔQTcF", "ΔJ-Tpeakc", "Mean peak concentration"]
    rows:
      - ["Lopinavir+Ritonavir", "383.9", "432.1", "**+48.1**", "+5.6", "28,077 ng/mL"]
      - ["Chloroquine", "393.3", "427.3", "**+34.1**", "−1.3", "453 ng/mL"]
      - ["Dofetilide", "396.1", "428.5", "**+32.4**", "+3.9", "1,488 pg/mL"]
      - ["Diltiazem+Dofetilide", "396.4", "416.7", "+20.3", "−8.4", "1,187 pg/mL"]
      - ["Ranolazine", "386.0", "393.4", "+7.4", "−20.2", "4,307 ng/mL"]
      - ["Verapamil", "386.5", "377.6", "−8.8", "−41.9", "523 ng/mL"]
      - ["Placebo", "386.6", "376.9", "−9.8", "−13.0", "n/a"]

  - type: description
    title: "How that table was computed, and how to read it"
    body: |
      All values in **ms**, recomputed over all 5,749 records from
      `ecgbench.labels.ecgcipa`. "Pre-dose" is the mean over each arm's records at
      `nominal_hours_from_period_start <= 0`; "Cmax" is the mean over its **top
      decile of plasma concentration** (for placebo, which has no analyte, every
      post-dose record).

      This is a **descriptive summary, not the study's analysis.** The published
      result fits a concentration–response model to placebo-corrected change from
      baseline on the triplicate averages, for the reason given in the previous
      section, and its effect sizes differ from these.

      Even so the pattern the study was built to detect is visible: **dofetilide,
      the predominant-hERG positive control, prolongs QTcF by ~32 ms while barely
      moving J-Tpeakc; verapamil and ranolazine, the balanced blockers, do not
      prolong QTcF at all and shorten J-Tpeakc substantially.** Adding calcium block
      to dofetilide (`Diltiazem+Dofetilide`) cuts its ΔQTcF and pushes J-Tpeakc
      negative. That divergence between the two intervals **is** the CiPA biomarker.

      Read every arm against **placebo**, whose −9.8 ms sets the diurnal drift:
      absolute changes here are not drug effects until that is subtracted.

  - type: description
    title: "Units, sentinels and two clocks — four ways to get this wrong"
    body: |
      **1. The signals are microvolts.** Every one of the 5,749 headers declares gain
      `0.26595744680851063(0)/uV`, so `wfdb.rdrecord` returns **µV** and
      `signal_unit_scale` is `0.001`. Left at `1.0`, every record's peak reads as
      ~2000 "mV" and `amplitude_outlier` fires on the whole dataset. `units="uV"`
      gets the source scale back.

      **2. Dofetilide plasma concentration is pg/mL; the other six analytes are
      ng/mL.** The source says so in `PCSTRESU` and nothing rescales it, which is
      why the columns are named for their own unit (`plasma_dofetilide_pg_ml`
      against `plasma_ranolazine_ng_ml`). Pooling them numerically is a 1000× error.

      **3. A plasma concentration of 0 means "below the limit of quantification",
      not "no drug".** 263 of `adpc.csv`'s 1,934 rows carry `LLOQFL=Y` with
      `AVAL=0`; every unflagged value is ≥ 1.24, and nothing else in the file is 0.
      `plasma_below_lloq` names the censored analytes — but **filter on the boolean
      `plasma_any_below_lloq`**, because the list column is `""` for uncensored
      records and pandas reads that empty string back from a CSV as `NaN`, so
      `!= ""` matches every row of a re-read frame.

      **4. There are two time axes and they disagree by design.** `timepoint`
      ("54 hrs", numeric in `nominal_hours_from_period_start`) counts from the
      **period's first dose**; `nominal_hours_from_reference` (`NRRLT`) counts from
      **that day's** reference dose. A record on study day 3 reads **54** and **6**
      for the same instant, because 48 + 6 = 54. Neither is wrong; picking the wrong
      one stacks three dosing days onto one time axis.

      Two smaller ones, both handled and regression-tested:

      - **`race` ships with leading spaces** — `"  WHITE"`, `" ASIAN"` — so a
        `groupby` on the raw column produces categories that look identical and are
        not. ECGBench strips them.
      - **`EGREPNUM` disagrees with itself in 4 records**, where the interval
        parameters are numbered one apart from HR/RR/QRS. `ADTM` is identical within
        each, so the record is unambiguous — only its index within the triplicate
        is not. `replicate_number` is anchored on the HR row (present for all 5,749)
        and `replicate_number_inconsistent` flags those four.

  - type: table
    title: "Validation summary (1000 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "5,749", "all records, with is_valid + quality_issues"]
      - ["clean", "5,747", "99.97% pass rate"]
      - ["excluded", "2", "both `amplitude_outlier`; no NaN, no flat or all-zero lead, uniform 12 × 10,000"]
      - ["amplitude extremes", "−18.85 … +10.93 mV", "over all 5,749 × 12 × 10,000 samples; `amplitude_range_mv` is the default ±10"]

  - type: description
    title: "About those counts"
    body: |
      **All 28,752 shipped files were verified against the release's own
      `SHA256SUMS.txt` before any figure on this page was computed — all OK**, so
      everything here is an upstream property rather than download damage. The
      `adeg.csv` checksum recorded in `manifest.json` (`3ab7b3cd…`) is the
      provider's own published value.

      **Record and subject counts match the PhysioNet landing page exactly** at
      **5,749** and **60**, as does the 5,749-record median-beat set. There is
      nothing to reconcile — unusually for this catalogue.

      Every other figure is **recomputed from the release's four analysis datasets**
      (`adeg.csv`, `adpc.csv`, `adsl.csv`, `addm.csv`) and the WFDB headers. The
      release ships **no table mapping records to signal files**, so ECGBench builds
      one (`ecgbench_metadata.csv`) on the first `ecgbench splits` run; see
      `ecgbench/labels/ecgcipa.py`.

      **Cohort, recomputed over the 60 subjects:** age 19–50 (mean 31.7, SD 8.7),
      38 male / 22 female, race 32 Black or African American / 24 White / 4 Asian,
      BMI 19.0–30.0 (mean 25.8), weight 56.4–99.8 kg, height 156.5–188.0 cm.
      Recordings span **2017-03-27 to 2017-06-26**.

      **Interval measurements are incomplete for 19 records, for two distinct
      reasons.** 10 have no `pr_ms` because no P onset could be annotated; 9 have no
      `qt_ms`, `qtcf_ms`, `jtpeak_ms`, `jtpeakc_ms` or `tpeak_tend_ms` because no T
      annotation could be placed. `hr_bpm`, `rr_ms` and `qrs_ms` are complete for
      all 5,749. Both groups are `NaN`, never 0, and the counts agree exactly with
      the missing fiducials in the `.atr` files.

      **Nothing on this page is a diagnostic class.** `treatment` is a
      fold-balancing label about a randomised assignment, and it applies to a
      subject's whole period — so **a pre-dose record in the dofetilide arm is a
      drug-free ECG that `treatment` calls `"Dofetilide"`.** Filter on
      `nominal_hours_from_period_start` or on the plasma concentration before
      treating the label as exposure.

  - type: description
    title: "Only 2 records fail validation, and both are electrode artefacts"
    body: |
      The amplitude distribution is tight — the 99.9th percentile of per-record peak
      absolute amplitude is **4.15 mV**, and only 4 records exceed 5 mV — so the
      default `amplitude_range_mv` of ±10 does real work here rather than passing
      everything:

      | Record | Subject | Leads | Extreme |
      |---|---|---|---|
      | `60D0B564-186D-4027-8B32-5B9D14278EBD` | 1024 | II, III, aVF | **−18.85 mV** |
      | `25A3D6EC-303B-4092-8FC3-C828876115BC` | 1043 | V5, V6 | **+10.93 mV** |

      Both are confined to one electrode group, which is what an electrode artefact
      looks like; neither is physiology. Two further records reach 5–7 mV and pass.
      No record in the release has a NaN sample, a missing lead or a flat lead — the
      minimum per-lead variance across all 5,749 × 12 channels is
      **5.6 × 10⁻⁴ mV²** — so `amplitude_outlier` is the only check that fires at
      all.

  - type: description
    title: "Folds are grouped by subject, and stratified on treatment"
    body: |
      `StratifiedGroupKFold` on `patient_id`, stratified on `treatment`. Verified on
      the shipped release: **no subject spans two folds, and none of the 1,917
      timepoint triplicates is ever split.** The default 8/1/1 mapping gives
      **4,604 train / 588 val / 555 test** (clean; 4,606 / 588 / 555 for original).

      The arithmetic works out well but is thin in places. 60 subjects over 10 folds
      is exactly **6 subjects per fold**, and each treatment has exactly 10
      subjects — so **all 7 treatments reach all 10 folds**, at 84 records per
      treatment per fold in 61 of the 70 cells:

      | Fold | Chl | Dil+Dof | Dof | Lop+Rit | Pla | Ran | Ver | Total |
      |---|---|---|---|---|---|---|---|---|
      | 1 | 84 | 84 | 84 | 84 | 84 | 84 | 84 | 588 |
      | 2 | 84 | 84 | 84 | 84 | 84 | 84 | 84 | 588 |
      | 3 | 84 | 84 | 84 | 84 | 84 | 84 | 84 | 588 |
      | 4 | 83 | 84 | 84 | 84 | 84 | 84 | 84 | 587 |
      | 5 | 84 | 84 | 84 | 84 | 84 | 84 | **24** | 528 |
      | 6 | 84 | 84 | 84 | 84 | 84 | 84 | 84 | 588 |
      | 7 | 84 | 84 | 84 | 84 | **79** | 84 | 84 | 583 |
      | 8 | 83 | **51** | 84 | 84 | 84 | 84 | 84 | 554 |
      | 9 (val) | 84 | 84 | 84 | 84 | 84 | 84 | 84 | 588 |
      | 10 (test) | 84 | **51** | 84 | 84 | 84 | 84 | 84 | 555 |

      The thin cells — Verapamil in fold 5 (24 records) and Diltiazem+Dofetilide in
      folds 8 and 10 (51 each) — are why fold sizes run **528–588** rather than a
      uniform 575, and they come from the 24-record subject and the crossover arm's
      shorter second period. **A per-arm result read off one fold can rest on as few
      as 24 records from a single subject**, so check the row before quoting it.
      `StratifiedGroupKFold` emits none of the "least populated class" warnings
      `StratifiedKFold` does, so the silence is not evidence of balance.

      One structural limit worth stating: **no split of this dataset separates
      `Dofetilide` from `Diltiazem+Dofetilide` by subject.** The ten crossover
      subjects contribute a block of each, and grouping by subject necessarily keeps
      both in the same fold.

      For a stricter harness, rotate the held-out fold rather than using the default
      8/1/1 mapping:

      ```python
      from ecgbench import ECGDataset

      # Hold out fold 3, train on the rest. With split=None each sample's
      # ["split"] reports that record's own default split.
      train = ECGDataset("ecgcipa", split=None,
                         fold_numbers=[1, 2, 4, 5, 6, 7, 8, 9, 10],
                         data_path="/path/to/ecgcipa/1.0.0/")
      held  = ECGDataset("ecgcipa", split=None, fold_numbers=[3],
                         data_path="/path/to/ecgcipa/1.0.0/")
      ```

  - type: code
    title: "Getting the data"
    language: bash
    body: |
      # ~1.5 GB, genuinely public -- no PhysioNet credentials needed.
      wget -r -N -c -np https://physionet.org/files/ecgcipa/1.0.0/

      # Verify before trusting any figure -- all 28,752 files should report OK.
      cd physionet.org/files/ecgcipa/1.0.0 && sha256sum -c SHA256SUMS.txt

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      # Writes ecgbench_metadata.csv into the dataset root on first run: the release
      # ships no table mapping records to signal files, and the validation engine
      # re-reads that file from disk. The dataset root must be writable.
      ecgbench splits --dataset ecgcipa --data-path /path/to/ecgcipa/1.0.0/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # Fold CSVs come from the HuggingFace Hub by default; only the waveforms
      # need to be local.
      ds = ECGDataset(
          "ecgcipa",
          split="train",
          data_path="/path/to/ecgcipa/1.0.0/",
          labels=True,
      )

      len(ds)                                      # 4604
      ds[0]["signal"].shape                        # (12, 10000)  -- 10 s at 1000 Hz
      ds[0]["record_id"]                           # '000545B0-1E01-47E6-9F7D-2C3141F5FC3E'

      # Drug exposure, not diagnosis:
      ds[0]["labels"]["patient_id"]                # '1008'
      ds[0]["labels"]["treatment"]                 # 'Ranolazine'  -- the stratification label
      ds[0]["labels"]["timepoint"]                 # '50.5 hrs'    -- from the period's first dose
      ds[0]["labels"]["replicate_number"]          # 2             -- of 3 at this timepoint
      ds[0]["labels"]["plasma_ranolazine_ng_ml"]   # 1050.97
      ds[0]["labels"]["qtcf_ms"]                   # 349.761545189634
      ds[0]["labels"]["jtpeakc_ms"]                # 192.638770318456
      ds[0]["labels"]["median_beat_path"]          # 'medians/1008/000545B0-1E01-47E6-9F7D-2C3141F5FC3E'

      # 5,749 records from 60 subjects, in near-duplicate triplicates: group before
      # quoting any rate. labels_df is aligned POSITIONALLY with metadata_df and
      # carries a RangeIndex, not record IDs.
      ds.labels_df["patient_id"].nunique()                     # 48 subjects in this split
      ds.labels_df.groupby("patient_id").size().max()          # 168 records
      ds.labels_df["treatment"].value_counts().to_dict()
      # {'Ranolazine': 672, 'Dofetilide': 672, 'Lopinavir+Ritonavir': 672,
      #  'Chloroquine': 670, 'Placebo': 667, 'Diltiazem+Dofetilide': 639,
      #  'Verapamil': 612}

      # The stored samples are MICROVOLTS (header gain 0.26595744680851063/uV), so
      # signal_unit_scale converts them. units="uV" gets the source scale back.
      mv = ECGDataset("ecgcipa", split="train", data_path="/path/to/ecgcipa/1.0.0/")
      uv = ECGDataset("ecgcipa", split="train", data_path="/path/to/ecgcipa/1.0.0/",
                      units="uV")
      float(mv[0]["signal"][1].max()), float(uv[0]["signal"][1].max())   # (0.948, 947.5)

      # 10,000 samples per record is the largest 12-lead tensor in this catalogue.
      # window= is pushed into wfdb's sampfrom/sampto, so it decodes only what it
      # returns -- and unlike a cropping lambda it survives num_workers>0 under
      # spawn. Length is uniform, so any window inside [0, 10000) fits every record.
      cropped = ECGDataset("ecgcipa", split="train", window=(2000, 5000),
                           data_path="/path/to/ecgcipa/1.0.0/")
      cropped[0]["signal"].shape                   # (12, 5000)  -- 5 s starting at 2 s

      # Leads by name; matching is case-insensitive, so 'avr' and 'aVR' both work.
      two = ECGDataset("ecgcipa", split="train", leads=["II", "V5"],
                       data_path="/path/to/ecgcipa/1.0.0/")
      two[0]["signal"].shape                       # (2, 10000)

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/ecgcipa/1.0.0/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/C2967M" }
      - { label: "Paper (Clin Pharmacol Ther, 2019)", url: "https://doi.org/10.1002/cpt.1303" }
      - { label: "ClinicalTrials.gov NCT03070470", url: "https://clinicaltrials.gov/study/NCT03070470" }
      - { label: "CiPA initiative", url: "https://cipaproject.org/" }
      - { label: "Example script", url: "https://github.com/vlbthambawita/ECGBench/blob/main/examples/load_ecgcipa.py" }
---
