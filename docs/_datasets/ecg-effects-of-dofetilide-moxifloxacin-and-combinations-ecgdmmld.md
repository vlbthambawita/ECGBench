---
slug: "ecg-effects-of-dofetilide-moxifloxacin-and-combinations-ecgdmmld"
name: "ECG Effects of Dofetilide, Moxifloxacin and Combinations (ECGDMMLD)"
category: "12-lead-physionet"
order: 17
status: "completed"
source_url: "https://physionet.org/content/ecgdmmld/1.0.0/"
url_label: "physionet.org"
format: "12-lead · 10 s · 500 Hz → 1 kHz · + derived median beats"
patients: "22"
records: "4,211 segments"
access: "open"
license: "ODC Attribution"
origin_institution: "US FDA / CDER; Phase I Crossover Study (NCT02308748)"
origin_country: "USA"
leads: 12
paper_title: "Clin Pharmacol Ther, 2016"
paper_doi: "https://doi.org/10.1002/cpt.205"
search_keywords: "ecgdmmld dofetilide moxifloxacin mexiletine lidocaine diltiazem usa pharmacology clinical trial crossover qt jtpeak late sodium current hERG proarrhythmia repolarisation pharmacokinetics 1000 hz"

related:
  - slug: "ecg-effects-of-ranolazine-dofetilide-verapamil-quinidine-ecgrdvq"
    relation: "sibling_release"
    shares_records: false
    verified: true
    note: >
      The immediately preceding study in the same US FDA programme — ECGRDVQ is
      SCR-002, this is SCR-003, and the CiPA validation study is SCR-004 — with the
      same file layout (a single SCR-00N.Clinical.Data.csv beside raw/ and medians/)
      and dofetilide as the shared positive control. Different trial and different
      volunteers: 22 subjects each, but **no recording is shared**. Verified against
      the files rather than the papers: the two RECORDS files list 5,232 and 4,211
      record UUIDs with an intersection of 0, and their subject IDs do not overlap
      either (ECGRDVQ numbers 1001-1022, ECGDMMLD 2001-2022), so these two
      particular releases can be pooled without prefixing. **That is not true of
      either against CiPA**, whose subjects run 1001-1050 and 2001-2010 and so
      collide with both — see the CiPA entry.

sections:
  - type: description
    title: "Overview"
    body: |
      4,211 ten-second 12-lead ECGs at **1 kHz** from **22 healthy volunteers** in a
      randomised, double-blind, placebo-controlled **5-period crossover** Phase I
      trial (study SCR-003, ClinicalTrials.gov **NCT02308748**) run by the US FDA's
      Center for Drug Evaluation and Research.

      The question is narrower and sharper than CiPA's: **can blocking late sodium
      current undo the QT prolongation caused by a pure hERG blocker?** Dofetilide is
      the pure hERG blocker and prolongs QT on its own. Mexiletine and lidocaine
      block late sodium current. If the CiPA theory is right, adding either to
      dofetilide should pull **J-Tpeak** — the early-repolarisation half of the QT
      interval — back down, even while QTc stays prolonged.

      **Every subject received all five regimens**, in one of ten randomised
      sequences with a week of washout between periods:

      | Code | Regimen | Role |
      |---|---|---|
      | `A` | Dofetilide | pure hERG block — the positive control |
      | `B` | Lidocaine + Dofetilide | hERG block + late sodium block (IV) |
      | `C` | Mexiletine + Dofetilide | hERG block + late sodium block (oral) |
      | `D` | Moxifloxacin + Diltiazem | hERG block + L-type calcium block |
      | `E` | Placebo | control |

      Records were extracted **in triplicate** at 14 nominal timepoints from half an
      hour before dosing to 24 hours after, each paired with a plasma concentration
      draw, and each is published **twice**: as the raw 10 s segment and as a derived
      16-channel median beat carrying semi-automatic fiducial annotations.

      **This is a pharmacology dataset, not a diagnosis dataset.** Every participant
      was screened to exclude cardiac disease. There is no rhythm, morphology or
      arrhythmia label anywhere in it, and none of the usual ECGBench habits built
      around a diagnostic class transfers — starting with stratification, which uses
      the treatment arm.

  - type: description
    title: "The one thing to know before training on this: the label is not the drug"
    body: |
      **`treatment` names the period's randomised regimen. It does not say what was
      circulating when a given record was taken.**

      Within each period the agents were staged hours apart — the late-sodium or
      calcium blocker first, the hERG blocker later in the day. So the arm's *second*
      drug is simply absent from the early timepoints:

      | `treatment` | On board at 1.5–3 h | Second agent appears |
      |---|---|---|
      | `Dofetilide` | dofetilide | — |
      | `Mexiletine + Dofetilide` | **mexiletine only** | dofetilide, 6.5 h |
      | `Lidocaine + Dofetilide` | **lidocaine only** | dofetilide, 6.5 h |
      | `Moxifloxacin + Diltiazem` | **moxifloxacin only** | diltiazem, 12 h |
      | `Placebo` | — | — |

      A record labelled `Mexiletine + Dofetilide` at `timepoint_hours = 2.0` is a
      **mexiletine-only ECG**, and there are hundreds of them. Measured against the
      plasma columns, only **57%** of the dofetilide-arm records have any dofetilide
      in them at all, and only **27%** of the moxifloxacin arm has diltiazem —
      the rest are pre-dose or pre-second-agent.

      `treatment` is ECGBench's stratification label because it is the only
      patient-level categorical in a cohort with no disease. **It is not a training
      target.** Use the six `plasma_*` columns for actual exposure, or cross
      `treatment` with `timepoint_hours`.

  - type: table
    title: "Treatment arms — the stratification label"
    headers: ["Treatment arm", "Subjects", "Records", "Share", "Ion-channel profile"]
    rows:
      - ["Mexiletine + Dofetilide", "21", "882", "20.9%", "hERG + late sodium (oral mexiletine)"]
      - ["Dofetilide", "20", "840", "19.9%", "predominant hERG (positive control)"]
      - ["Placebo", "20", "840", "19.9%", "control"]
      - ["Lidocaine + Dofetilide", "20", "825", "19.6%", "hERG + late sodium (IV lidocaine)"]
      - ["Moxifloxacin + Diltiazem", "20", "824", "19.6%", "hERG + L-type calcium"]
      - ["**Total**", "**22**", "**4,211**", "**100%**", "5 values of `treatment` (`TRTA`)"]

  - type: description
    title: "Why the subject counts are all 20–21 and not 4 or 5"
    body: |
      Because this is a **complete crossover**: 19 of the 22 subjects passed through
      all five arms, so nearly every subject appears in every row of the table above.
      Two consequences that a parallel-group dataset would not have:

      - **No split of this dataset separates the arms.** Folds are grouped on
        `patient_id` (they must be — see the clustering section), and each patient
        carries all five treatments, so every fold receives all five automatically.
        Per-arm balance across folds is a property of the design, not of the
        stratifier. Treat "all 5 arms in all 10 folds" as expected rather than as
        evidence of anything.
      - **Every treatment comparison rests on the same 22 people.** There is no
        independent control group. That is a strength for the study — a within-subject
        comparison is far more sensitive — and a limit on generalisation.

      The three subjects who withdrew early are why the counts are 20–21 rather than
      22: **2015** completed 1 period, **2011** completed 2, and **2021** completed 3.
      Balancing those three across folds is the only real work stratification does
      here.

  - type: description
    title: "4,211 records are not 4,211 observations"
    body: |
      Two levels of clustering sit between the record count and anything you could
      call an independent sample.

      **Triplicates.** Three ten-second segments were extracted per subject per
      nominal timepoint — **1,403 of the 1,404 timepoint groups hold exactly 3
      records** (one holds 2). They are the same person, in the same posture, at the
      same plasma concentration, seconds apart: near-duplicates, not repeats of an
      experiment. **The effective sample size is closer to 1,404 than to 4,211.**

      **Subjects.** Those 1,404 groups come from 22 people, **42 to 210 records each**
      (median 210), because three withdrew early. So every per-record statistic is
      weighted by trial compliance rather than by person.

      Folds are grouped on `patient_id`, which handles both at once: verified on the
      shipped release, **no subject spans two folds and none of the 1,404 timepoint
      triplicates is ever split**. Group your own analyses the same way, and weight
      by subject before quoting any rate.

  - type: description
    title: "Every record ships twice — raw, and a derived median beat"
    body: |
      | Directory | Contents | Channels | Samples | ECGBench column |
      |---|---|---|---|---|
      | `raw/<subject>/<uuid>` | the 10 s acquisition | 12 | 10,000 | `signal_path` — **the dataset's signal** |
      | `medians/<subject>/<uuid>` | derived representative median beat | 16 | 1,200 | `median_beat_path` (labels only) |

      The median beat adds the **vector-magnitude lead `VCGMAG`** and the Frank
      **`vx`, `vy`, `vz`** components to the 12, and carries a `.atr` annotation file.
      Both directories are 1 kHz and use the same record ID, so they are two
      representations of one acquisition, not two records.

      **The medians get no fold of their own, deliberately.** Every median beat is a
      derivation of a raw record ECGBench already partitions; generating a second
      ten-fold split over the same recordings would let someone train on one and
      evaluate on the other. `ECGDataset` therefore always reads `raw/`, and the
      median beats reach you as paths plus a fiducial loader.

      ```python
      from ecgbench import load_config
      from ecgbench.labels.ecgdmmld import load_median_beat_fiducials

      config = load_config("ecgdmmld")
      fid = load_median_beat_fiducials("/path/to/ecgdmmld/1.0.0/", config)
      fid.loc["39BF8219-C83A-4121-926F-2BC730FBE127"]
      # p_onset_ms 193, qrs_onset_ms 359, qrs_offset_ms 431,
      # t_peak_ms 694, t_peak_secondary_ms None, t_offset_ms 779
      ```

      **The fiducials and the interval table are not independent measurements.** The
      annotations are what the published intervals were measured *from*, and they
      reproduce the clinical table exactly: across **all 4,211 records**, PR, QT and
      Tpeak-Tend recomputed from the fiducials equal the published value **to the
      millisecond**, as do QRS and J-Tpeak for the 4,202 that have a QRS offset (for
      the record above: 359−193 = PR 166, 431−359 = QRS 72, 779−359 = QT 420,
      694−431 = J-Tpeak 263, 779−694 = Tpeak-Tend 85). Treat agreement between them
      as a format check, never as corroboration.

  - type: description
    title: "Three defects in the release — all of them upstream"
    body: |
      All 21,059 files match the release's own `SHA256SUMS.txt`, so none of these is
      download damage.

      **1. `TPEAKTPEAKP` is empty in all 4,211 rows.** The column is documented —
      "interval between the two peaks of the T-wave (if secondary peak is present)" —
      and it is 100% `NA`. Consistently, **no `.atr` file in the release marks a
      second T peak**, though the PhysioNet page lists "secondary T peak (if
      present)" among the annotations. There is no secondary T-peak information in
      this dataset at all. ECGBench still exposes it as `tpeak_tpeakp_ms` so its
      absence is visible rather than something you infer.

      **2. Three median-beat headers are corrupt** and raise `IndexError` from
      `wfdb.rdrecord`. In each, one channel's `.dat` filename has digits from the
      gain field spliced into it:

      | Record | Subject | Broken channel | Header says | File is |
      |---|---|---|---|---|
      | `9D7B03F2-…-FFDFD3526628` | 2004 | `vy` | `…FFDFD3526620008.dat` | `…FFDFD3526628.dat` |
      | `DCA7A8CC-…-48F31964B73D` | 2007 | `VCGMAG` | `…48F31000964B73D.dat` | `…48F31964B73D.dat` |
      | `79B4DFED-…-5B5C9D803D62` | 2012 | `VCGMAG` | `…5B5C9D803D62000.dat` | `…5B5C9D803D62.dat` |

      The `.dat` payloads are intact (38,400 bytes, the correct size) and the `.atr`
      files parse, so only these three median **signals** are unreachable without
      repairing the header by hand. **4,208 of 4,211 load.** The corresponding
      `raw/` records are unaffected, so this costs the split nothing — filter on
      the `median_beat_readable` column.

      **3. Nine records have no QRS-offset annotation**, so `qrs_ms` and `jtpeak_ms`
      are `NA` for them and their `.atr` carries 4 marks instead of 5. `rr_ms`,
      `pr_ms`, `qt_ms` and `tpeak_tend_ms` are complete for all 4,211.

  - type: table
    title: "Per-record labels (38 columns) — no class among them"
    headers: ["Group", "Columns", "Notes"]
    rows:
      - ["Identity & paths", "`patient_id`, `signal_path`, `median_beat_path`, `median_beat_readable`", "the record ID is the `EGREFID` UUID, unique across the release; **neither path exists in the source** — both are derived"]
      - ["Drug exposure", "`treatment`, `treatment_sequence`", "`treatment` is the **period's arm**, not the drug on board — see above"]
      - ["Timing", "`period`, `period_label`, `timepoint_hours`, `is_baseline`", "one clock only, unlike CiPA: hours from the period's first dose, −0.5 to 24"]
      - ["Intervals (ms; HR in bpm)", "`hr_bpm`, `rr_ms`, `pr_ms`, `qrs_ms`, `qt_ms`, `qtcf_ms`, `jtpeak_ms`, `tpeak_tend_ms`, `erd_30_ms`, `lrd_30_ms`, `tpeak_tpeakp_ms`", "**`hr_bpm` and `qtcf_ms` are derived** — the release ships neither; `tpeak_tpeakp_ms` is always empty"]
      - ["T-wave morphology", "`twave_amplitude_uv`, `twave_asymmetry`, `twave_flatness`", "**amplitude is µV** while the waveforms are mV; the other two are dimensionless"]
      - ["Plasma concentration", "`plasma_{lidocaine,mexiletine,moxifloxacin,moxifloxacin_m2,diltiazem}_ng_ml`, `plasma_dofetilide_pg_ml`", "**dofetilide is pg/mL, the other five ng/mL**; 2,637 of 4,211 records have at least one measured value"]
      - ["Subject", "`age_years`, `sex`, `race`, `ethnicity`, `height_cm`, `weight_kg`, `systolic_bp_mmhg`, `diastolic_bp_mmhg`", "constant within a subject, so repeated across their 42–210 records"]

  - type: table
    title: "Interval and morphology measurements over all 4,211 records"
    headers: ["Parameter", "Mean", "SD", "Min", "Max", "Missing"]
    rows:
      - ["`hr_bpm` (HR, bpm) — derived", "67.1", "9.2", "45.1", "104.5", "0"]
      - ["`rr_ms` (RR)", "910.8", "121.2", "574", "1331", "0"]
      - ["`pr_ms` (PR)", "162.0", "20.5", "107", "353", "0"]
      - ["`qrs_ms` (QRS)", "86.3", "8.5", "65", "106", "**9**"]
      - ["`qt_ms` (QT)", "388.2", "24.3", "325", "475", "0"]
      - ["`qtcf_ms` (QTcF) — derived", "401.5", "22.9", "353.7", "499.1", "0"]
      - ["`jtpeak_ms` (J-Tpeak)", "212.4", "23.3", "138", "281", "**9**"]
      - ["`tpeak_tend_ms` (Tpeak-Tend)", "89.5", "14.9", "57", "206", "0"]
      - ["`erd_30_ms` (30% early repol. duration)", "54.8", "13.3", "20", "194", "0"]
      - ["`lrd_30_ms` (30% late repol. duration)", "35.3", "10.4", "7", "93", "0"]
      - ["`twave_amplitude_uv` (µV)", "501.0", "166.7", "81.7", "1259.7", "0"]
      - ["`twave_asymmetry`", "0.21", "0.11", "0", "1.34", "0"]
      - ["`twave_flatness`", "0.44", "0.06", "0.22", "0.62", "0"]
      - ["`tpeak_tpeakp_ms` (Tpeak-Tpeak′)", "—", "—", "—", "—", "**4,211**"]

  - type: description
    title: "The endpoint that CiPA cannot give you, and this one can"
    body: |
      The study's result is **change from baseline**, not the absolute interval. In
      the sibling CiPA release those change values exist only on `adeg.csv`'s
      triplicate-average rows, which carry a blank record ID and therefore cannot be
      attached to any waveform. **Here nothing is lost.**

      `BASELINE = Y` flags the three pre-dose (`timepoint_hours = −0.5`) records of
      each **(subject, period)** pair, and all **101** pairs in the release have one —
      **303 baseline records over 101 pairs**. So a baseline is just their mean, and
      the delta is a per-record quantity:

      ```python
      from ecgbench import load_config
      from ecgbench.labels.ecgdmmld import load_baseline_deltas

      df = load_baseline_deltas("/path/to/ecgdmmld/1.0.0/", load_config("ecgdmmld"))
      df.loc[~df["is_baseline"]].groupby("treatment")["delta_jtpeak_ms"].mean().round(1)
      # Dofetilide                 -4.9
      # Lidocaine + Dofetilide    -18.5
      # Mexiletine + Dofetilide   -25.2
      # Moxifloxacin + Diltiazem  -12.7
      # Placebo                   -18.4
      ```

      Note the placebo row: **J-Tpeak falls by 18 ms on placebo alone.** These raw
      deltas are dominated by diurnal drift, which is precisely why the published
      analysis is placebo-corrected and why you must be too.

      Baselines are **per period, not per subject** — each crossover period has its
      own pre-dose triplicate, and sharing one across periods would attribute a
      washout drift to the drug.

      What is *not* done for you is **placebo-correction**, which needs the placebo
      arm's mean across subjects at the same nominal timepoint. That is an analysis
      decision rather than a label, so it is left to you — and it is why the figures
      here do not equal the published ones.

  - type: table
    title: "What the dataset is for: J-Tpeak separates the arms that QTc cannot"
    headers: ["Arm", "QTcF pre-dose", "QTcF at Cmax", "ΔQTcF", "ΔJ-Tpeak", "Mean peak concentration"]
    rows:
      - ["Dofetilide", "398.4", "446.1", "**+47.7**", "**+7.1**", "1,822 pg/mL dofetilide"]
      - ["Lidocaine + Dofetilide", "398.4", "423.3", "+24.9", "−1.8", "2,232 pg/mL dofetilide"]
      - ["Mexiletine + Dofetilide", "398.4", "422.0", "+23.6", "−13.1", "2,086 pg/mL dofetilide"]
      - ["Moxifloxacin + Diltiazem", "398.7", "416.3", "+17.6", "−33.7", "10,490 ng/mL moxifloxacin"]
      - ["Placebo", "397.2", "389.8", "−7.5", "−18.4", "n/a"]

  - type: description
    title: "How that table was computed, and how to read it"
    body: |
      All values in **ms**, recomputed over all 4,211 records from
      `ecgbench.labels.ecgdmmld`. "Pre-dose" is the mean over each arm's
      `is_baseline` records; "Cmax" is the mean over its **top decile of plasma
      concentration** of the analyte named in the last column (for placebo, which has
      no analyte, every post-dose record).

      This is a **descriptive summary, not the study's analysis.** The published
      result fits a concentration–response model to placebo-corrected change from
      baseline, and its effect sizes differ from these.

      **Read every arm against placebo before concluding anything.** Placebo is not
      flat: it drifts −7.5 ms on QTcF and **−18.4 ms on J-Tpeak** over the same
      hours, so the raw J-Tpeak column above is mostly diurnal drift and the naive
      reading ("all the combination arms shorten J-Tpeak") is an artefact.
      Subtracting the placebo row is what the study does, and it is what makes the
      pattern appear:

      | Arm | ΔQTcF vs placebo | ΔJ-Tpeak vs placebo |
      |---|---|---|
      | Dofetilide | **+55.2** | **+25.5** |
      | Lidocaine + Dofetilide | +32.4 | +16.6 |
      | Mexiletine + Dofetilide | +31.1 | **+5.3** |
      | Moxifloxacin + Diltiazem | +25.1 | **−15.3** |

      Now the result is legible. **Dofetilide alone prolongs both QTcF (+55) and
      J-Tpeak (+26). Adding a late-sodium blocker leaves QTcF still clearly
      prolonged (+31 to +32, only about half undone) while collapsing J-Tpeak toward
      zero (+16.6 with lidocaine, +5.3 with mexiletine).** Replace late-sodium block
      with calcium block (moxifloxacin+diltiazem) and J-Tpeak goes negative
      altogether (−15.3).

      That is the point of the dataset: **QTcF cannot separate these arms — every one
      of them is prolonged by +25 to +55 ms — and J-Tpeak can**, spanning +26 to −15
      over the same records. This is a descriptive reconstruction, not the published
      model, but the ordering it recovers is the study's finding.

  - type: description
    title: "Units and scale — three ways to get this wrong"
    body: |
      **1. The signals are millivolts, and this is the opposite of its sibling.**
      Every channel of every record declares **its own gain** against unit `/mV` —
      per-lead *and* per-record, e.g. `330979.8(-11915)/mV` for lead I next to
      `33780.4(-16215)/mV` for lead II of the same record, each fitted to that
      channel's range. `wfdb.rdrecord` applies them, so `signal_unit_scale` is
      **`1.0`**. CiPA declares `/uV` and needs `0.001`; **copying that value here
      divides every sample by 1000 and `amplitude_outlier` never fires again.**
      `units="uV"` multiplies by 1000 if you want the microvolt scale.

      **2. The 1 kHz is interpolated.** Acquisition was at **500 Hz** with 2.5 µV
      resolution on a Mortara Surveyor; the depositors **up-sampled the extracted
      segments to 1000 Hz**. Every record is 10,000 samples and the headers say
      1000 Hz, but there are only 5,000 measurements in there. Anything sensitive to
      true bandwidth — high-frequency QRS content, derivative-based delineation —
      should treat this as 500 Hz data.

      **3. `twave_amplitude_uv` is microvolts while the waveforms are millivolts.**
      The source measures it on the median beat's vector-magnitude lead; the range is
      81.7–1259.7 µV. The unit is in the column name for the same reason dofetilide's
      is: so a rename cannot lose it.

      And on the plasma columns: **dofetilide is pg/mL, the other five analytes are
      ng/mL**, per the shipped column description. Pooling them numerically is a
      1000× error. A missing concentration is `NA` and means *either* "not dosed /
      not drawn" *or* "below the limit of quantification" — unlike CiPA, this release
      has no censoring flag and no zeros to distinguish the two.

  - type: table
    title: "Lead order — uppercase, unlike its sibling"
    headers: ["Index", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    rows:
      - ["`lead_names`", "I", "II", "III", "**AVR**", "**AVL**", "**AVF**", "V1", "V2", "V3", "V4", "V5", "V6"]

  - type: description
    title: "On the spelling"
    body: |
      Standard order, spelled with an **uppercase A** — `AVR`/`AVL`/`AVF`, like
      PTB-XL. Verified identical in all 4,211 `raw/` headers, and the `medians/`
      headers agree.

      This is the **opposite of CiPA**, whose `raw/` headers write `aVR`/`aVL`/`aVF`
      while its `medians/` headers write `AVR`/`AVL`/`AVF` — so the two sibling
      releases disagree with each other, and CiPA disagrees with itself. Matching in
      `ECGDataset(leads=...)` is case-insensitive, so `leads=["aVL"]` resolves in
      both; `lead_names` records what the files actually say. Never index leads
      positionally across datasets — `signal[4]` is aVL here and in CiPA, but aVF in
      MIMIC-IV-ECG.

  - type: table
    title: "Validation summary (1000 Hz)"
    headers: ["Metric", "Value"]
    rows:
      - ["Records validated", "4,211"]
      - ["Valid (`clean`)", "**4,209** (99.95%)"]
      - ["Excluded", "2"]
      - ["`amplitude_outlier`", "2 records, 4 lead-level issues"]
      - ["`missing_leads` / `nan_values` / `flat_line` / `truncated_signal` / `corrupt_header`", "**0** — none fires"]

  - type: description
    title: "What validation actually caught"
    body: |
      `amplitude_outlier` is the only check that fires. Scanned over all 4,211
      records × 12 leads × 10,000 samples the observed range is **−8.40 to
      +18.49 mV**, but the distribution is tight: the 99.9th percentile of per-record
      peak absolute amplitude is **3.91 mV** and only 4 records exceed 5 mV. The
      house default `[-10, 10]` mV therefore excludes exactly **two**:

      | Record | Subject | Arm | Range | Leads |
      |---|---|---|---|---|
      | `9779B087-4B95-4FA0-9C6E-D532B1366DAE` | 2016 | Placebo | −0.84 to **+18.49 mV** | V5 |
      | `1DA19FBE-DEB2-40BE-A657-AD9DCA8B9FBD` | 2007 | Moxifloxacin + Diltiazem | −1.29 to **+13.60 mV** | II, III, AVF |

      Both are electrode artefacts, not physiology. Nothing else is wrong with the
      release's signals: **no record has a NaN sample, a missing lead or a flat
      lead** (the minimum per-lead variance across the whole release is
      2.3 × 10⁻⁴ mV², against the check's 10⁻⁶ threshold), and **every one of the
      4,211 `raw/` headers parses** — the three corrupt headers are in `medians/`,
      which validation does not read.

  - type: table
    title: "Fold layout (10-fold, grouped on patient_id, stratified on treatment)"
    headers: ["Fold", "Split", "Records (original)", "Records (clean)", "Subjects", "Subject IDs"]
    rows:
      - ["1", "train", "504", "503", "3", "2004, 2007, 2011"]
      - ["2", "train", "462", "462", "3", "2012, 2015, 2020"]
      - ["3", "train", "336", "336", "2", "2005, 2021"]
      - ["4", "train", "404", "404", "2", "2003, 2018"]
      - ["5", "train", "420", "420", "2", "2001, 2019"]
      - ["6", "train", "405", "405", "2", "2009, 2013"]
      - ["7", "train", "420", "420", "2", "2014, 2017"]
      - ["8", "train", "420", "420", "2", "2002, 2010"]
      - ["9", "**val**", "420", "419", "2", "2016, 2022"]
      - ["10", "**test**", "420", "420", "2", "2006, 2008"]
      - ["**Total**", "", "**4,211**", "**4,209**", "**22**", "2001–2022"]

  - type: description
    title: "Reading the fold table"
    body: |
      **Fold sizes are uneven, and patient grouping is why.** 22 subjects over 10
      folds is 2–3 per fold while records per subject run 42–210, so folds run
      **336–504 records** rather than a uniform 421. That is the correct trade — the
      alternative leaks triplicates — but it means a per-fold metric is computed over
      a varying number of records from a *fixed and very small* number of people.
      **Fold 3 is two subjects. Fold 10, the test set, is two subjects.** Any result
      quoted on one fold of this dataset is a result about two or three individuals.

      The three early withdrawals were spread rather than clustered: **2011** (84
      records) lands in fold 1, **2015** (42) in fold 2, **2021** (126) in fold 3,
      each alongside completers. That is the one thing stratifying on `treatment`
      accomplished here.

      All five arms appear in all ten folds — **0 empty cells of 50** — with 84
      records per arm per fold in 45 of them. The exceptions are Dofetilide and
      Placebo in fold 1 (126 each) and fold 3 (42 each), Mexiletine+Dofetilide in
      fold 2 (126), Moxifloxacin+Diltiazem in fold 4 (68) and Lidocaine+Dofetilide
      in fold 6 (69). As explained above, that completeness follows from the
      crossover design and not from the stratifier.

      Fold membership is **identical** between `original/` and `clean/`; `clean/` is a
      row subset, differing only where the two excluded records sat (folds 1 and 9).

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # Fold CSVs come from the Hub; data_path points at your local signal files.
      ds = ECGDataset(
          "ecgdmmld",
          split="train",
          data_path="/path/to/ecgdmmld/1.0.0/",
          labels=True,
      )
      len(ds)                                  # 3370
      sample = ds[0]
      sample["signal"].shape                   # torch.Size([12, 10000])
      sample["labels"]["patient_id"]            # '2021'
      sample["labels"]["treatment"]             # 'Moxifloxacin + Diltiazem'
      sample["labels"]["timepoint_hours"]       # 8.0
      sample["labels"]["qtcf_ms"]               # 429.4 (derived — not in the source)

      # And here is the trap, live. This record's arm is "Moxifloxacin + Diltiazem",
      # but at 8 h only the moxifloxacin is on board — diltiazem is not dosed until
      # 12 h, so its concentration is NaN. `treatment` is the randomisation arm.
      sample["labels"]["plasma_moxifloxacin_ng_ml"]  # 6750.0
      sample["labels"]["plasma_diltiazem_ng_ml"]     # nan

      # The samples are already millivolts (signal_unit_scale=1.0, NOT CiPA's 0.001).
      # units="uV" gets the microvolt scale if you want it.
      uv = ECGDataset("ecgdmmld", split="train", data_path="...", units="uV")

      # Records are a uniform 10,000 samples, so any window inside [0, 10000) fits
      # all of them. window= is pushed into wfdb's sampfrom/sampto, so it decodes
      # only what it returns — and unlike a cropping lambda it survives
      # DataLoader(num_workers>0) under the spawn start method.
      short = ECGDataset("ecgdmmld", split="train", data_path="...",
                         window=(2000, 5000), leads=["I", "II", "V5"])
      short[0]["signal"].shape                 # torch.Size([3, 5000])

  - type: links
    title: "Links"
    items:
      - label: "PhysioNet project page"
        url: "https://physionet.org/content/ecgdmmld/1.0.0/"
      - label: "Paper — Johannesen et al., Clin Pharmacol Ther 2016;99(2):214–23"
        url: "https://doi.org/10.1002/cpt.205"
      - label: "Data DOI — 10.13026/C2D016"
        url: "https://doi.org/10.13026/C2D016"
      - label: "ClinicalTrials.gov — NCT02308748"
        url: "https://clinicaltrials.gov/study/NCT02308748"
      - label: "Example script — examples/load_ecgdmmld.py"
        url: "https://github.com/vlbthambawita/ECGBench/blob/main/examples/load_ecgdmmld.py"
---
