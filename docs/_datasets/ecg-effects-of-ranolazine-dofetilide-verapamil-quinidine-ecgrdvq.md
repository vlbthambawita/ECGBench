---
slug: "ecg-effects-of-ranolazine-dofetilide-verapamil-quinidine-ecgrdvq"
name: "ECG Effects of Ranolazine, Dofetilide, Verapamil, Quinidine (ECGRDVQ)"
category: "12-lead-physionet"
order: 18
status: "completed"
source_url: "https://physionet.org/content/ecgrdvq/1.0.0/"
url_label: "physionet.org"
format: "12-lead · 10 s · 500 Hz → 1 kHz · + derived median beats"
patients: "22"
records: "5,232 segments"
access: "open"
license: "ODC Attribution"
origin_institution: "US FDA / CDER; Phase I Crossover Study (NCT01873950)"
origin_country: "USA"
leads: 12
paper_title: "Clin Pharmacol Ther, 2014"
paper_doi: "https://doi.org/10.1038/clpt.2014.155"
search_keywords: "ecgrdvq ranolazine dofetilide verapamil quinidine usa pharmacology clinical trial crossover qt jtpeak tpeak-tend multichannel block hERG late sodium calcium proarrhythmia repolarisation pharmacokinetics 1000 hz"

# No `related:` block here on purpose. Both siblings already declare their edge to
# this dataset — see cipa-ecg-validation-study.md and
# ecg-effects-of-dofetilide-moxifloxacin-and-combinations-ecgdmmld.md — and
# catalogue.py derives the inverse, so mirroring it here would double-count on the
# website and fail tests/test_catalogue.py.

sections:
  - type: description
    title: "Overview"
    body: |
      5,232 ten-second 12-lead ECGs at **1 kHz** from **22 healthy volunteers** in a
      randomised, double-blind, placebo-controlled **5-period crossover** Phase I
      trial (study SCR-002, ClinicalTrials.gov **NCT01873950**) run by the US FDA's
      Center for Drug Evaluation and Research.

      This is the **first** of the three FDA studies in the catalogue — ECGRDVQ is
      SCR-002, ECGDMMLD is SCR-003 and the CiPA validation study is SCR-004 — and it
      is the one that established the measurement the other two were built to test.

      **The question: can J-Tpeak tell two kinds of QT prolongation apart?** A drug
      that blocks only the hERG potassium channel is proarrhythmic. A drug that
      blocks hERG *and* an inward current — late sodium, or L-type calcium — prolongs
      QTc just as much but carries far less risk. QTc alone cannot distinguish them.
      The claim under test is that **J-Tpeak**, the early-repolarisation half of the
      QT interval, can.

      **Every subject received all five treatments**, as a single oral dose per
      period with washout in between:

      | Code | Treatment | Dose | Ion-channel profile |
      |---|---|---|---|
      | `A` | Ranolazine | 1500 mg | hERG + late sodium |
      | `B` | Dofetilide | 500 **µg** | predominant hERG — the positive control |
      | `C` | Verapamil HCL | 120 mg | hERG + L-type calcium |
      | `D` | Quinidine Sulph | 400 mg | hERG + peak and late sodium |
      | `E` | Placebo | — | control |

      Records were extracted **in triplicate** at 16 nominal timepoints from half an
      hour before dosing to 24 hours after, each paired with a plasma concentration
      draw, and each is published **twice**: as the raw 10 s segment and as a derived
      16-channel median beat carrying semi-automatic fiducial annotations.

      **This is a pharmacology dataset, not a diagnosis dataset.** Every participant
      was screened to exclude cardiac disease. There is no rhythm, morphology or
      arrhythmia label anywhere in it, and none of the usual ECGBench habits built
      around a diagnostic class transfers — starting with stratification, which uses
      the treatment.

  - type: description
    title: "The good news: unlike its sibling, the label is the drug"
    body: |
      In **ECGDMMLD** the treatment arms are **staged combinations** — the second
      agent was dosed hours after the first, so a record labelled
      `Mexiletine + Dofetilide` at the 2-hour timepoint contains no dofetilide at all,
      and roughly a third of that dataset's labels do not describe what is in the
      blood.

      **Nothing like that applies here.** Each of the five periods administered a
      **single agent**, so `treatment` names both the randomisation arm and the drug.
      Measured against the plasma column, **93–94% of every active arm's records
      carry a measured concentration** of exactly the drug the label names.

      One much smaller caveat survives, and it is the same one every crossover has:
      **327 records are pre-dose.** They sit at `timepoint_hours = −0.5`, are flagged
      `is_baseline`, and carry their period's drug name while containing no drug —
      a "Dofetilide" record that is a drug-free ECG. At `timepoint_hours = 0.5`
      absorption is still incomplete. Filter on `is_baseline`, or use
      `plasma_concentration_ng_ml`, before treating `treatment` as exposure.

  - type: table
    title: "Treatments — the stratification label"
    headers: ["Treatment", "Dose", "Subjects", "Records", "Share", "Ion-channel profile"]
    rows:
      - ["Dofetilide", "500 µg", "22", "1,056", "20.2%", "predominant hERG (positive control)"]
      - ["Ranolazine", "1500 mg", "22", "1,056", "20.2%", "hERG + late sodium"]
      - ["Verapamil HCL", "120 mg", "22", "1,056", "20.2%", "hERG + L-type calcium"]
      - ["Placebo", "—", "22", "1,056", "20.2%", "control"]
      - ["Quinidine Sulph", "400 mg", "21", "1,008", "19.3%", "hERG + peak/late sodium"]
      - ["**Total**", "", "**22**", "**5,232**", "**100%**", "5 values of `treatment` (`EXTRT`)"]

  - type: description
    title: "Why the subject counts are all 21–22 and not 4 or 5"
    body: |
      Because this is a **near-complete crossover**: **21 of the 22 subjects passed
      through all five arms**, so nearly every subject appears in every row of the
      table above. Two consequences a parallel-group dataset would not have:

      - **No split of this dataset separates the arms.** Folds are grouped on
        `patient_id` (they must be — see the clustering section), and each patient
        carries four or five treatments, so every fold receives all five
        automatically. Per-arm balance across folds is a property of the design, not
        of the stratifier. Treat "all 5 arms in all 10 folds" as expected rather than
        as evidence of anything.
      - **Every treatment comparison rests on the same 22 people.** There is no
        independent control group. That is a strength for the study — a within-subject
        comparison is far more sensitive — and a limit on generalisation.

      **Subject `1002` is the single exception**: it withdrew after 4 of the 5
      periods, so it has 192 records instead of 240 and never received quinidine.
      That is why quinidine has 21 subjects and 1,008 records where every other arm
      has 22 and 1,056 — and placing that one lopsided subject is essentially the
      only work stratifying on `treatment` does here.

  - type: description
    title: "5,232 records are not 5,232 observations"
    body: |
      Two levels of clustering sit between the record count and anything you could
      call an independent sample.

      **Triplicates, and here they are exact.** Three ten-second segments were
      extracted per subject per nominal timepoint, and **all 1,744 timepoint groups
      hold precisely 3 records** — 5,232 = 1,744 × 3 with no exceptions. (ECGDMMLD
      manages 1,403 of 1,404; CiPA is raggeder still.) They are the same person, in
      the same posture, at the same plasma concentration, seconds apart:
      near-duplicates, not repeats of an experiment. **The effective sample size is
      closer to 1,744 than to 5,232.**

      **Subjects.** Those 1,744 groups come from 22 people, 240 records each except
      subject 1002's 192. So the whole dataset is 22 individuals, and any
      subject-level quantity has n = 22.

      Folds are grouped on `patient_id`, which handles both at once: verified on the
      shipped release, **no subject spans two folds and none of the 1,744 timepoint
      triplicates is ever split**. Group your own analyses the same way, and weight
      by subject before quoting any rate.

  - type: description
    title: "Every record ships twice — raw, and a derived median beat"
    body: |
      | Directory | Contents | Channels | Samples | ECGBench column |
      |---|---|---|---|---|
      | `raw/<subject>/<uuid>` | the 10 s acquisition | 12 | 10,000 (uniform) | `signal_path` — **the dataset's signal** |
      | `medians/<subject>/<uuid>` | derived representative median beat | 16 | **968–1,876 (varies)** | `median_beat_path` (labels only) |

      The median beat adds the **vector-magnitude lead `VCGMAG`** and the Frank
      **`vx`, `vy`, `vz`** components to the 12, and carries a `.atr` annotation file.
      Both directories are 1 kHz and use the same record ID, so they are two
      representations of one acquisition, not two records.

      **The median beat's length is not fixed here.** It runs 968 to 1,876 samples
      across 667 distinct values, where ECGDMMLD's is a constant 1,200 — so never
      assume a shape, and never read a fiducial as a fraction of the beat.

      **The medians get no fold of their own, deliberately.** Every median beat is a
      derivation of a raw record ECGBench already partitions; generating a second
      ten-fold split over the same recordings would let someone train on one and
      evaluate on the other. `ECGDataset` therefore always reads `raw/`, and the
      median beats reach you as paths plus a fiducial loader.

      ```python
      from ecgbench import load_config
      from ecgbench.labels.ecgrdvq import load_median_beat_fiducials

      config = load_config("ecgrdvq")
      fid = load_median_beat_fiducials("/path/to/ecgrdvq/1.0.0/", config)
      fid.loc["0003bfaf-41e9-4411-a1c2-a216330cbdf6"]
      # p_onset_ms 133, qrs_onset_ms 310, qrs_offset_ms 404,
      # t_peak_ms 607, t_peak_secondary_ms NaN, t_offset_ms 688, n_annotations 5
      ```

      **The fiducials and the interval table are not independent measurements.** The
      annotations are what the published intervals were measured *from*, and they
      reproduce the clinical table exactly: recomputed over every median beat in the
      release, **PR matches for all 5,221 un-wrapped records, QRS and J-Tpeak for all
      5,223, QT and Tpeak-Tend for the 5,219 with a T offset, and Tpeak-Tpeak′ for all
      42 with a secondary T peak — every one to the millisecond** (for the record
      above: 310−133 = PR 177, 404−310 = QRS 94, 688−310 = QT 378, 607−404 =
      J-Tpeak 203, 688−607 = Tpeak-Tend 81). Treat agreement between them as a format
      check, never as corroboration.

  - type: table
    title: "Annotation patterns across all 5,232 records"
    headers: ["`.atr` marks", "Records", "What is missing", "Consequence"]
    rows:
      - ["`( ( ) t )`", "5,175", "nothing — the usual five", "all intervals present"]
      - ["`( ( ) t t )`", "42", "nothing — plus a **secondary T peak**", "`tpeak_tpeakp_ms` populated"]
      - ["`( ( ) t`", "4", "the T offset", "`qt_ms`, `tpeak_tend_ms` are `NA`"]
      - ["`( ) t )`", "2", "the P onset", "`pr_ms` wrapped through 2³² — repaired"]
      - ["*no file*", "9", "the whole median beat", "`pr`/`qrs`/`qt`/`jtpeak`/`tpeak_tend` all `NA`"]

  - type: description
    title: "Four defects in the release — all of them upstream"
    body: |
      All **26,137** files match the release's own `SHA256SUMS.txt`, so none of these
      is download damage. Every one is a *missing* value rather than a wrong one, once
      the third is repaired.

      **1. Nine records have no median beat.** `medians/` holds 5,223 of the 5,232.
      Because **every interval in the clinical table was measured from the median
      beat**, those nine rows have no PR, QRS, QT, J-Tpeak or Tpeak-Tend. `rr_ms` is
      present for all nine (it comes from the raw rhythm strip) and — inconsistently —
      two of them still carry `erd_30_ms`/`lrd_30_ms`/`twave_amplitude_uv` and all
      nine carry `twave_asymmetry`/`twave_flatness`. So the median beat clearly *was*
      computed upstream and simply was not published. Their `raw/` records are intact,
      so this costs the split nothing; filter on `median_beat_available`.

      **2. Four records have no T-offset annotation**, so `qt_ms` and `tpeak_tend_ms`
      are `NA` and their `.atr` carries 4 marks instead of 5. All four are **subject
      1004 on quinidine at 2.5–3.5 h**, and they hold the four flattest T waves in the
      release — `twave_amplitude_uv` 69.7–78.5 µV against a median of **490.5**.
      Quinidine flattened the T wave until its end could not be marked. With defect 1,
      `qt_ms` is `NA` for **13** records.

      **3. Two records store PR as a 32-bit arithmetic wrap**, `-4294966951` and
      `-4294966972`. Both are **subject 1007 on verapamil at 1.0 h**, and both are
      exactly the two records whose `.atr` has **no P-onset mark** — the P onset fell
      *before* the start of the median-beat window, so an unsigned subtraction
      wrapped. Adding 2³² recovers **345 ms** and **324 ms**:

      | Record | Stored PR | + 2³² | Implied P onset | Corroboration |
      |---|---|---|---|---|
      | `c2017512-…-5a0950acc6a6` | −4294966951 | **345** | −34 ms (outside the beat) | third record of the same triplicate: PR 293 ms |
      | `ebd075f4-…-fb1157f8c61a` | −4294966972 | **324** | −21 ms (outside the beat) | verapamil PR peaks at this timepoint |

      It is the only physiologic residue, it is corroborated by the triplicate's third
      record (PR 293 ms — the highest un-wrapped PR in the release) and by verapamil's
      expected AV-nodal PR prolongation, and 293 → 324/345 is exactly the trend
      subject 1007's other verapamil timepoints trace. ECGBench **repairs both** and
      flags each with `pr_ms_repaired`, so the repair is auditable rather than silent.
      It is also why `pr_ms` has a maximum of 345 in the table below.

      **4. Two records have a dead V4 lead**, held at a constant −0.00625 mV for all
      10,000 samples. Both are **subject 1019 on quinidine at 14.0 h** and both are
      among the nine of defect 1 — the flat lead is presumably why no median beat was
      derived. **These two are the only records excluded from `clean/`.**

      **Not a defect, but unexplained:** `twave_asymmetry` and `twave_flatness` are
      `NA` for **129** records, spread across all five treatments and all 22 subjects
      with no pattern, and *not* concentrated on the flattest T waves (their amplitudes
      run 81.7–960.7 µV). The release does not say why.

  - type: table
    title: "Per-record labels (39 columns) — no class among them"
    headers: ["Group", "Columns", "Notes"]
    rows:
      - ["Identity & paths", "`patient_id`, `signal_path`, `median_beat_path`, `median_beat_available`", "the record ID is the `EGREFID` UUID, unique across the release; **neither path exists in the source** — both are derived"]
      - ["Drug exposure", "`treatment`, `treatment_sequence`, `dose`, `dose_unit`", "`treatment` **is** the drug here, unlike ECGDMMLD; `dose_unit` is `ug` for dofetilide and `mg` for the rest"]
      - ["Timing", "`period`, `period_label`, `timepoint_hours`, `is_baseline`", "one clock only: hours from the period's dose, −0.5 to 24, over 16 nominal timepoints"]
      - ["Intervals (ms; HR in bpm)", "`hr_bpm`, `rr_ms`, `pr_ms`, `pr_ms_repaired`, `qrs_ms`, `qt_ms`, `qtcf_ms`, `jtpeak_ms`, `tpeak_tend_ms`, `tpeak_tpeakp_ms`, `erd_30_ms`, `lrd_30_ms`", "**`hr_bpm` and `qtcf_ms` are derived** — the release ships neither; `tpeak_tpeakp_ms` exists only for the 42 records with a secondary T peak"]
      - ["T-wave morphology", "`twave_amplitude_uv`, `twave_asymmetry`, `twave_flatness`", "**amplitude is µV** while the waveforms are mV; the other two are dimensionless"]
      - ["Plasma concentration", "`plasma_analyte`, `plasma_concentration`, `plasma_concentration_unit`, `plasma_concentration_ng_ml`", "**long format** — one agent per period means one measurement per record; **dofetilide is pg/mL and the rest ng/mL**, which is what the `_ng_ml` column exists to fix"]
      - ["Subject", "`age_years`, `sex`, `race`, `ethnicity`, `height_cm`, `weight_kg`, `systolic_bp_mmhg`, `diastolic_bp_mmhg`", "constant within a subject, so repeated across their 192–240 records"]

  - type: table
    title: "Interval and morphology measurements over all 5,232 records"
    headers: ["Parameter", "Mean", "SD", "Min", "Max", "Missing"]
    rows:
      - ["`hr_bpm` (HR, bpm) — derived", "64.3", "9.5", "39.3", "97.1", "0"]
      - ["`rr_ms` (RR)", "953.9", "140.1", "618", "1,528", "0"]
      - ["`pr_ms` (PR)", "161.3", "23.4", "95", "345 †", "**9**"]
      - ["`qrs_ms` (QRS)", "96.6", "8.5", "75", "126", "**9**"]
      - ["`qt_ms` (QT)", "399.6", "33.6", "325", "579", "**13**"]
      - ["`qtcf_ms` (QTcF) — derived", "407.3", "32.3", "338.6", "563.2", "**13**"]
      - ["`jtpeak_ms` (J-Tpeak)", "220.5", "28.6", "132", "360", "**9**"]
      - ["`tpeak_tend_ms` (Tpeak-Tend)", "82.6", "20.7", "55", "265", "**13**"]
      - ["`erd_30_ms` (30% early repol. duration)", "50.4", "13.2", "5", "172", "**7**"]
      - ["`lrd_30_ms` (30% late repol. duration)", "32.3", "12.7", "0", "155", "**7**"]
      - ["`twave_amplitude_uv` (µV)", "511.3", "192.0", "66.6", "1,021.5", "**7**"]
      - ["`twave_asymmetry`", "0.22", "0.18", "0", "1.56", "**129**"]
      - ["`twave_flatness`", "0.43", "0.08", "0.23", "0.75", "**129**"]
      - ["`tpeak_tpeakp_ms` (Tpeak-Tpeak′)", "86.2", "26.2", "47", "131", "**5,190**"]

  - type: description
    title: "Reading that table"
    body: |
      Recomputed from `ecgbench.labels.ecgrdvq` over all 5,232 records of the shipped
      v1.0.0, not copied from the paper.

      **† The PR maximum of 345 ms is a repaired value** — one of the two 32-bit
      wraps described above. The highest PR the release stores un-wrapped is 293 ms.

      `tpeak_tpeakp_ms` is missing for 5,190 records because **a secondary T peak
      genuinely is not present** in them, not because the measurement failed. That is
      the opposite of ECGDMMLD, where the same column is empty in **all** 4,211 rows
      and no annotation marks a second peak anywhere in the release. A pipeline
      written against that sibling will silently drop a real measurement here.

      Every other "missing" count traces to defect 1, 2 or the unexplained 129 — see
      the defects section for which is which.

  - type: table
    title: "What the dataset is for: J-Tpeak separates the arms that QTc cannot"
    headers: ["Arm", "QTcF pre-dose", "QTcF at Cmax", "ΔQTcF", "ΔJ-Tpeak", "ΔTpeak-Tend", "Mean peak concentration"]
    rows:
      - ["Quinidine Sulph", "396.7", "483.7", "**+86.9**", "**+6.9**", "+42.6", "2,052 ng/mL quinidine"]
      - ["Dofetilide", "394.1", "465.6", "**+71.5**", "**+20.1**", "+31.8", "2,791 pg/mL dofetilide"]
      - ["Ranolazine", "396.7", "416.8", "+20.1", "−10.5", "+5.9", "3,725 ng/mL ranolazine"]
      - ["Verapamil HCL", "395.8", "405.0", "+9.2", "−25.2", "+1.1", "167 ng/mL verapamil"]
      - ["Placebo", "395.8", "388.1", "−7.6", "−16.8", "−0.8", "n/a"]

  - type: description
    title: "How that table was computed, and how to read it"
    body: |
      All values in **ms**, recomputed over all 5,232 records from
      `ecgbench.labels.ecgrdvq`. "Pre-dose" is the mean over each arm's `is_baseline`
      records; "Cmax" is the mean over its **top decile of plasma concentration** of
      the analyte named in the last column (96–102 records per arm; for placebo, which
      has no analyte, all 990 post-dose records).

      This is a **descriptive summary, not the study's analysis.** The published result
      fits a concentration–response model to placebo-corrected change from baseline,
      and its effect sizes differ from these.

      **Read every arm against placebo before concluding anything.** Placebo is not
      flat: it drifts −7.6 ms on QTcF and **−16.8 ms on J-Tpeak** over the same hours,
      so the raw J-Tpeak column above understates every drug and the naive reading
      ("ranolazine and verapamil *shorten* J-Tpeak") is partly artefact. Subtracting
      the placebo row is what the study does, and it is what makes the pattern appear:

      | Arm | ΔQTcF vs placebo | ΔJ-Tpeak vs placebo | ΔTpeak-Tend vs placebo |
      |---|---|---|---|
      | Quinidine Sulph | **+94.5** | **+23.7** | +43.4 |
      | Dofetilide | **+79.1** | **+36.9** | +32.6 |
      | Ranolazine | +27.7 | **+6.3** | +6.7 |
      | Verapamil HCL | +16.8 | **−8.4** | +1.9 |

      Now the result is legible, and it is the finding the whole CiPA programme rests
      on. **Every one of the four drugs prolongs QTcF** — by +17 to +95 ms, a range in
      which the ordering tells you nothing about risk. **J-Tpeak splits them in two:**
      the predominant-hERG blockers (dofetilide **+37**, quinidine **+24**) prolong it,
      and the multichannel blockers (ranolazine **+6**, verapamil **−8**) do not.
      Tpeak-Tend separates them the same way, and even more sharply.

      That is the point of the dataset: **QTc says all four drugs are dangerous;
      J-Tpeak says two of them are not.** This is a descriptive reconstruction rather
      than the published model, but the split it recovers is the study's finding — and
      it is the measurement ECGDMMLD (SCR-003) and CiPA (SCR-004) were then built to
      confirm.

  - type: description
    title: "The endpoint is computable per record"
    body: |
      The study's result is **change from baseline**, not the absolute interval. In
      the sibling CiPA release those change values exist only on `adeg.csv`'s
      triplicate-average rows, which carry a blank record ID and therefore cannot be
      attached to any waveform. **Here nothing is lost.**

      `BASELINE = Y` flags the three pre-dose (`timepoint_hours = −0.5`) records of
      each **(subject, period)** pair, and **all 109 pairs have one — 327 baseline
      records over 109 pairs.** So a baseline is just their mean, and the delta is a
      per-record quantity:

      ```python
      from ecgbench import load_config
      from ecgbench.labels.ecgrdvq import load_baseline_deltas

      df = load_baseline_deltas("/path/to/ecgrdvq/1.0.0/", load_config("ecgrdvq"))
      post = df[~df["is_baseline"]]
      post.groupby("treatment")["delta_jtpeak_ms"].mean().round(1)
      # Dofetilide          +2.4
      # Placebo            -17.7
      # Quinidine Sulph    -11.6
      # Ranolazine         -20.4
      # Verapamil HCL      -22.3
      ```

      Note the placebo row: **J-Tpeak falls by 18 ms on placebo alone.** These raw
      deltas are dominated by diurnal drift, which is precisely why the published
      analysis is placebo-corrected and why you must be too. (These figures average
      over *all* post-dose timepoints, so they are smaller than the Cmax-restricted
      ones in the table above; both are in the example script.)

      Baselines are **per period, not per subject** — each crossover period has its
      own pre-dose triplicate, and sharing one across periods would attribute a
      washout drift to the drug.

      What is *not* done for you is **placebo-correction**, which needs the placebo
      arm's mean across subjects at the same nominal timepoint. That is an analysis
      decision rather than a label, so it is left to you — and it is why the figures
      here do not equal the published ones.

  - type: description
    title: "Units and scale — four ways to get this wrong"
    body: |
      **1. The signals are millivolts.** Every channel of every record declares **its
      own gain** against unit `/mV` — per-lead *and* per-record, e.g.
      `149792(-19192)/mV` for lead I next to `40562.6(-22753)/mV` for lead II of the
      same record, each fitted to that channel's range. `wfdb.rdrecord` applies them,
      so `signal_unit_scale` is **`1.0`**, matching ECGDMMLD. **CiPA declares `/uV`
      and needs `0.001`; copying that value here divides every sample by 1000.**
      `units="uV"` multiplies by 1000 if you want the microvolt scale.

      **2. The 1 kHz is interpolated.** Acquisition was at **500 Hz** on a Mortara
      Surveyor; the depositors **up-sampled the extracted segments to 1000 Hz**. Every
      record is 10,000 samples and the headers say 1000 Hz, but there are only 5,000
      measurements in there. Anything sensitive to true bandwidth — high-frequency QRS
      content, derivative-based delineation — should treat this as 500 Hz data.

      **3. `twave_amplitude_uv` is microvolts while the waveforms are millivolts.**
      The source measures it on the median beat's vector-magnitude lead; the range is
      66.6–1,021.5 µV.

      **4. The plasma column mixes two units, and so does the dose.** The PK table is
      **long** rather than wide, because one agent per period means one measurement per
      record: `plasma_analyte` names it, `plasma_concentration` is the value,
      `plasma_concentration_unit` its unit. **Dofetilide is reported in pg/mL and
      quinidine, ranolazine and verapamil in ng/mL**, so
      `groupby("treatment").plasma_concentration.mean()` compares numbers 1000×
      apart in scale. `plasma_concentration_ng_ml` is the same quantity in one unit,
      provided so that mistake is avoidable rather than merely documented. `dose`
      carries the same hazard: **500 for dofetilide is µg**, the other three are mg —
      never sort or compare `dose` without `dose_unit`.

      A missing concentration is `NA` and means *either* "not dosed / not drawn" *or*
      "below the limit of quantification" — this release has no censoring flag and no
      zeros to distinguish the two. Nine records name an analyte but carry no value
      (three triplicates at the 0.5 h timepoint).

  - type: table
    title: "Lead order — uppercase, like ECGDMMLD and unlike CiPA"
    headers: ["Index", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    rows:
      - ["`lead_names`", "I", "II", "III", "**AVR**", "**AVL**", "**AVF**", "V1", "V2", "V3", "V4", "V5", "V6"]

  - type: description
    title: "On the spelling"
    body: |
      Standard order, spelled with an **uppercase A** — `AVR`/`AVL`/`AVF`, like
      PTB-XL and its sibling ECGDMMLD. Verified identical in all 5,232 `raw/` headers,
      and the `medians/` headers agree.

      This is the **opposite of CiPA**, whose `raw/` headers write `aVR`/`aVL`/`aVF`
      while its `medians/` headers write `AVR`/`AVL`/`AVF` — so CiPA disagrees with
      both siblings and with itself. Matching in `ECGDataset(leads=...)` is
      case-insensitive, so `leads=["aVL"]` resolves in all three; `lead_names` records
      what the files actually say. Never index leads positionally across datasets —
      `signal[4]` is aVL here and in both siblings, but aVF in MIMIC-IV-ECG.

  - type: table
    title: "Validation summary (1000 Hz)"
    headers: ["Metric", "Value"]
    rows:
      - ["Records validated", "5,232"]
      - ["Valid (`clean`)", "**5,230** (99.96%)"]
      - ["Excluded", "2"]
      - ["`flat_line`", "2 records, 2 lead-level issues (both lead 9 = V4)"]
      - ["`missing_leads` / `nan_values` / `truncated_signal` / `corrupt_header` / `amplitude_outlier`", "**0** — none fires"]

  - type: description
    title: "What validation actually caught"
    body: |
      `flat_line` is the only check that fires, and this is the **mirror image of
      ECGDMMLD**, where `amplitude_outlier` is the only one that does.

      **`amplitude_outlier` never fires here** because the release is unusually clean
      on amplitude: scanned over all 5,232 records × 12 leads × 10,000 samples the
      observed range is only **−4.86 to +6.98 mV**, the 99.9th percentile of per-record
      peak absolute amplitude is **3.75 mV**, and just **2** records exceed 5 mV. The
      house default `[-10, 10]` mV therefore excludes none of them.

      **`flat_line` excludes exactly two**, and they tell a coherent story:

      | Record | Subject | Arm | Timepoint | Dead lead | Held at |
      |---|---|---|---|---|---|
      | `6b53aa2e-…-df136bb51ecb` | 1019 | Quinidine Sulph | 14.0 h | **V4** | −0.00625 mV |
      | `ec9ca3bf-…-c2ded39fdcb6` | 1019 | Quinidine Sulph | 14.0 h | **V4** | −0.00625 mV |

      Both are two of the three records of a single triplicate, and both are among the
      nine records with **no published median beat** — a dead V4 is presumably why no
      median beat could be derived from them. The third record of that triplicate is
      fine and stays in `clean/`.

      Nothing else is wrong with the release's signals: **no record has a NaN sample,
      a missing lead or a wrong sample count** (all 5,232 headers declare exactly
      10,000 samples at 1000 Hz), and **every one of the 5,232 `raw/` headers parses**.
      The minimum per-lead variance across the release is 9.8 × 10⁻³¹ mV² — that is the
      dead V4 — while every other lead-record sits far above the check's 10⁻⁶ threshold.

  - type: table
    title: "Fold layout (10-fold, grouped on patient_id, stratified on treatment)"
    headers: ["Fold", "Split", "Records (original)", "Records (clean)", "Subjects", "Subject IDs"]
    rows:
      - ["1", "train", "672", "672", "3", "1002, 1018, 1020"]
      - ["2", "train", "480", "480", "2", "1001, 1013"]
      - ["3", "train", "480", "480", "2", "1010, 1014"]
      - ["4", "train", "720", "**718**", "3", "1007, 1009, 1019"]
      - ["5", "train", "480", "480", "2", "1016, 1022"]
      - ["6", "train", "480", "480", "2", "1006, 1008"]
      - ["7", "train", "480", "480", "2", "1011, 1021"]
      - ["8", "train", "480", "480", "2", "1012, 1015"]
      - ["9", "**val**", "480", "480", "2", "1004, 1017"]
      - ["10", "**test**", "480", "480", "2", "1003, 1005"]
      - ["**Total**", "", "**5,232**", "**5,230**", "**22**", "1001–1022"]

  - type: description
    title: "Reading the fold table"
    body: |
      **Fold sizes are uneven, and patient grouping is why.** 22 subjects over 10 folds
      is 2–3 per fold while records per subject are 240 (or 192 for subject 1002), so
      folds run **480–720 records** rather than a uniform 523. That is the correct
      trade — the alternative leaks triplicates — but it means a per-fold metric is
      computed over a *fixed and very small* number of people. **Eight of the ten folds
      are two subjects. Fold 10, the test set, is two subjects: 1003 and 1005.** Any
      result quoted on one fold of this dataset is a result about two or three
      individuals.

      Subject **1002**, the early withdrawal, lands in fold 1 alongside two completers,
      which is why that fold is 672 rather than 720. That placement is the one thing
      stratifying on `treatment` accomplished here.

      All five arms appear in all ten folds — **0 empty cells of 50**. Eight folds hold
      96 records per arm; folds 1 and 4, the three-subject folds, hold 144 — except
      quinidine in fold 1, which holds 96 because subject 1002 never received it. As
      explained above, that completeness follows from the crossover design and not from
      the stratifier.

      Fold membership is **identical** between `original/` and `clean/`; `clean/` is a
      row subset, differing only in fold 4 where the two dead-V4 records sat.

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # Fold CSVs come from the Hub; data_path points at your local signal files.
      ds = ECGDataset(
          "ecgrdvq",
          split="train",
          data_path="/path/to/ecgrdvq/1.0.0/",
          labels=True,
      )
      len(ds)                                  # 4270
      sample = ds[0]
      sample["signal"].shape                   # torch.Size([12, 10000])
      sample["labels"]["patient_id"]            # '1007'
      sample["labels"]["treatment"]             # 'Verapamil HCL'
      sample["labels"]["dose"]                  # 120.0  (dose_unit 'mg' — dofetilide is 'ug')
      sample["labels"]["timepoint_hours"]       # 3.5
      sample["labels"]["qtcf_ms"]               # 416.97 (derived — not in the source)
      sample["labels"]["jtpeak_ms"]             # 203.0

      # Unlike ECGDMMLD, the label IS the drug: each period dosed a single agent, so
      # the plasma analyte matches `treatment` whenever the record is post-dose.
      sample["labels"]["is_baseline"]                 # False
      sample["labels"]["plasma_analyte"]              # 'Verapamil'
      sample["labels"]["plasma_concentration"]        # 78.7
      sample["labels"]["plasma_concentration_unit"]   # 'ng/mL'
      # ...and use the normalised column across arms: dofetilide is reported in
      # pg/mL, so the raw column mixes two scales 1000x apart.
      sample["labels"]["plasma_concentration_ng_ml"]  # 78.7

      # The samples are already millivolts (signal_unit_scale=1.0, NOT CiPA's 0.001).
      # units="uV" gets the microvolt scale if you want it.
      uv = ECGDataset("ecgrdvq", split="train", data_path="...", units="uV")
      uv[0]["signal"][1].max()                 # 1346.9 uV, vs 1.347 mV by default

      # Records are a uniform 10,000 samples, so any window inside [0, 10000) fits
      # all of them. window= is pushed into wfdb's sampfrom/sampto, so it decodes
      # only what it returns — and unlike a cropping lambda it survives
      # DataLoader(num_workers>0) under the spawn start method.
      short = ECGDataset("ecgrdvq", split="train", data_path="...",
                         window=(2000, 5000), leads=["I", "II", "V5"])
      short[0]["signal"].shape                 # torch.Size([3, 5000])

  - type: links
    title: "Links"
    items:
      - label: "PhysioNet project page"
        url: "https://physionet.org/content/ecgrdvq/1.0.0/"
      - label: "Paper — Johannesen et al., Clin Pharmacol Ther 2014;96(5):549–58"
        url: "https://doi.org/10.1038/clpt.2014.155"
      - label: "Data DOI — 10.13026/C2HP45"
        url: "https://doi.org/10.13026/C2HP45"
      - label: "ClinicalTrials.gov — NCT01873950"
        url: "https://clinicaltrials.gov/study/NCT01873950"
      - label: "Example script — examples/load_ecgrdvq.py"
        url: "https://github.com/vlbthambawita/ECGBench/blob/main/examples/load_ecgrdvq.py"
---
