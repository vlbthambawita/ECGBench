---
slug: "eye-tracking-dataset-for-12-lead-ecg-interpretation"
name: "Eye Tracking Dataset for 12-Lead ECG Interpretation"
category: "12-lead-physionet"
order: 19
status: "completed"
source_url: "https://physionet.org/content/eye-tracking-ecg/1.0.0/"
url_label: "physionet.org"
format: "gaze metrics over 10 printed ECGs · no raw ECGs"
patients: "63 readers"
records: "630 reader sessions"
access: "open"
license: "ODC ODbL 1.0"
origin_institution: "Qatar Biomedical Research Institute, Hamad bin Khalifa Univ."
origin_country: "Qatar"
leads: 12
paper_title: "Understanding Cardiology Practitioners' Interpretations of Electrocardiograms: An Eye-Tracking Study"
paper_doi: "https://doi.org/10.2196/34058"
search_keywords: "eye tracking gaze aoi ecg interpretation qatar hamad bin khalifa expertise reader study images no waveforms"

sections:
  - type: description
    title: "Overview"
    body: |
      This release measures **how clinicians read an ECG**, not what an ECG
      contains. Sixty-three practitioners — from junior medical students to
      board-certified consultants — were each shown the same ten printed 12-lead
      ECGs for about 30 seconds apiece while an eye tracker recorded where they
      looked. What ships is the aggregated gaze behaviour:

      - **`Datasets/Grid_Anonymized.csv`** — 14,742 rows, one per reader x image x
        area of interest, where the areas are the individual lead boxes, the
        quarters of each rhythm strip, and the printed settings footer.
      - **`Datasets/Long_Short_Anonymized.csv`** — the same 630 sessions scored
        against a two-region split instead: `Long` (the rhythm strips) versus
        `Short` (the twelve short lead traces).
      - **`ECGs/ECG_Images/`** — the ten stimulus images (JPG/PNG).
      - **`AOI_Distributions/`** — two figures showing how the AOI grids were laid
        out over the normal-sinus-rhythm image.

      Each row carries 17 gaze and fixation measures: time to first hit, total and
      percentage dwell time, revisit counts, fixation counts and durations. The
      research use is expertise modelling and attention prediction — which lead
      draws the eye first, and how that changes between a medical student and a
      consultant.

  - type: description
    title: "How ECGBench integrates it — no config, no splits"
    body: |
      **There is deliberately no `eye_tracking_ecg` config, and no
      `ecgbench splits --dataset eye_tracking_ecg`.** Nothing in this release is an
      ECG recording in ECGBench's sense. There is no sampled signal, no sampling
      rate, no `signal_format` that could name a JPEG, and no patient behind a
      record — the ten ECGs are printed exemplars shipped without waveforms or
      demographics. The unit of observation is a *reader session*, so a ten-fold
      partition over "records" would be partitioning ten pictures.

      So ECGBench exposes it as a **table provider** instead, and leaves splitting
      to whoever defines a task on it — which is the right call anyway, since a
      reader study can be split by reader or by image and those are two different
      experiments.

      ```python
      from ecgbench.labels.eye_tracking_ecg import load_eye_tracking_ecg

      df = load_eye_tracking_ecg("/path/to/eye-tracking-ecg/1.0.0/")
      df.groupby(["Group", "aoi_lead"])["Time_spent_G_Percentage"].mean()
      ```

      This is the same treatment [PTB-XL+](ptb-xl-plus.html) gets, for a related
      reason: a dataset carrying no recordings of its own gets no fold assignment
      of its own.

  - type: table
    title: "What ships, per stimulus image"
    headers: ["ECG shown", "Image file", "AOI rows / reader", "Lead boxes", "Strip quarters", "Footer AOI"]
    rows:
      - ["Normal sinus rhythm", "`Normal_Sinus_Rhythm.jpg`", "25", "12", "12", "yes"]
      - ["Atrial fibrillation", "`Atrial_Fibrillation.png`", "16", "12", "4", "—"]
      - ["Atrial flutter", "`Atrial_Flutter.jpg`", "25", "12", "12", "yes"]
      - ["Complete heart block", "`Complete_Heart_Block.jpg`", "25", "12", "12", "yes"]
      - ["Left bundle branch block", "`Left_Bundle_Branch_Block.jpg`", "25", "12", "12", "yes"]
      - ["ST elevation MI", "`STEMI.png`", "24", "12", "12", "—"]
      - ["Ventricular paced rhythm", "`Ventricular_Paced_Rhythm.jpg`", "25", "12", "12", "yes"]
      - ["Ventricular tachycardia", "`Ventricular_Tachycardia.jpg`", "25", "12", "12*", "yes"]
      - ["Wolff-Parkinson-White", "`Wolf_Parkinson_White_Syndrome.jpg`", "24", "12", "12", "—"]
      - ["Hyperkalemia", "`Hyperkalemia.jpg`", "20", "**16**", "4", "—"]

  - type: description
    title: "Reading the AOI labels"
    body: |
      Ventricular tachycardia is starred above because its 12 strip quarters carry
      only 11 distinct names — see the naming defects further down.

      The grid AOI vocabulary is not self-explanatory, and
      `AOI_Distributions/Grid_AOIs.png` is what decodes it. On a full 25-AOI image:

      - **`1`, `2`, `3` are leads I, II and III.** They are numbered rather than
        named, so they read like indices and the three limb leads appear to be
        missing entirely.
      - `aVR`, `aVL`, `aVF`, `V1`–`V6` are the remaining lead boxes.
      - `II-1`…`II-4`, `V1-1`…`V1-4`, `V5-1`…`V5-4` are quarters of the three
        rhythm strips along the bottom.
      - `Information` is the printed footer (`25mm/s 10mm/mV 150Hz 12SL …`), which
        is not a trace at all — exclude it before averaging over leads.

      **Labels are scoped to their image**: the same lead is `V1 NSR` on one and
      `V1 AFib` on another, giving 233 distinct labels for what are really ~25
      regions. Grouping by `Label` therefore compares nothing across images.
      `load_aoi_metrics()` derives `aoi_area`, `aoi_kind` (`lead` / `rhythm_strip` /
      `information`) and `aoi_lead` (standard spelling, so `1` becomes `I` and
      `V5-3` becomes `V5`) to make cross-image grouping correct.

      Note the marked row in the table above: the **hyperkalemia image is a 16-lead
      trace**, adding `V3R`, `V4R`, `V7` and `V8`. The catalogue's `leads: 12`
      describes the other nine images.

  - type: description
    title: "About those counts"
    body: |
      All 19 shipped files were verified against the release's own
      `SHA256SUMS.txt` before anything below was computed, so the discrepancies are
      upstream rather than download damage.

      **The published demographics do not match the shipped CSVs.** The paper's
      Table 1 and the `Group` column agree on 63 participants and on eight of the
      ten categories, but two nurse categories are transposed and one participant's
      recorded gender differs:

      | Category | Table 1 | Shipped CSVs | Diff |
      |---|---|---|---|
      | Medical students, junior (`Med 1`) | 5 | 5 | — |
      | Medical students, senior (`Med 2`) | 11 | 11 | — |
      | Resident (`resident`) | 1 | 1 | — |
      | Fellows (`Fellow`) | 10 | 10 | — |
      | Technicians (`Technician`) | 10 | 10 | — |
      | Cardiac care unit nurses (`CCU Nurse`) | 5 | **3** | −2 |
      | Catheterization lab nurses (`Cathlab Nurse`) | 6 | 6 | — |
      | General nurses (`Nurse`) | 4 | **6** | +2 |
      | General doctors (`General Doctor`) | 2 | 2 | — |
      | Consultants (`Consultant`) | 9 | 9 | — |
      | **Total** | **63** | **63** | — |
      | Gender, male / female | 51 / 12 | **52 / 11** | ±1 |

      Derivation: `load_respondents()`, which takes the first `Group`/`Gender`/`Age`
      per `Respondent_Name` (all three are constant across each reader's 244 rows).
      The totals agree, so this is a categorisation difference between the
      manuscript and the released file rather than missing participants. Use the
      CSV's `Group`, since that is what the gaze rows are actually keyed to.

      **`Age` is unusable as shipped.** Table 1 reports five age bands covering all
      63 participants, but the CSV records a real age for only **9 of 63** — the
      other 54 carry `0`, an anonymisation artefact rather than a value. Because it
      is `0` and not a blank, `notna()` reports the column 100% populated and a mean
      age comes out around 4 years. `load_respondents()` and `load_aoi_metrics()`
      convert it to `NaN`.

      **Two more sentinels, in the gaze columns.** `-1` means "this never
      happened": `Hit_time_G` is `-1` for the **2,086** AOIs (14.2%) a reader never
      looked at, and `First_Fixation_Duration` / `Average_Fixations_Duration` are
      `-1` for the **3,654** with no fixation at all. Left in place they drag every
      mean down. Converted to `NaN` by default; pass `sentinels_to_nan=False` for
      the raw values. One related inconsistency is *not* auto-corrected, because
      fixing it would be inference rather than decoding: `TTFF_F` carries a
      plausible-looking time even on those 3,654 rows where there was no first
      fixation to time. Mask it on `Fixations_Count > 0` yourself.

      **Sessions are not all 30 seconds.** Duration is nominally ~30,000 ms but
      ranges **2,839–30,392 ms**, and **139 of 630** ran under 29 s — the shortest
      is a consultant spending 2.8 s on Wolff-Parkinson-White. Compare the
      `Time_spent_*_Percentage` columns rather than absolute milliseconds, since the
      percentages are relative to each session's own duration.

      **Two AOI naming defects.** `II-3 VTach` names *two different regions* with
      different metrics — ventricular tachycardia has 25 AOI rows per reader but
      only 24 distinct labels, for all 63 readers — so aggregating by name silently
      merges them, and `(reader, image, label)` is not a unique key.
      `load_aoi_metrics()` adds `aoi_occurrence` to disambiguate. Separately, three
      complete-heart-block labels carry accumulated authoring-tool suffixes
      (`II-2 CompleteHeartBlock copy`, `… copy copy`, `… copy copy copy`), and the
      normal-sinus-rhythm image labels one strip quarter `V3-3` where the sequence
      is otherwise `V1-1`, `V1-2`, `V1-4` — that image has no `V1-3` and no other V3
      strip, so it is a typo upstream. ECGBench strips the `copy` suffixes, which is
      unambiguous, and reports `V3-3` as written rather than rewriting data on a
      guess.

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench.labels.eye_tracking_ecg import (
          load_aoi_metrics,
          load_eye_tracking_ecg,
          load_respondents,
          stimulus_image_path,
      )

      root = "/path/to/eye-tracking-ecg/1.0.0/"

      # No ECGDataset here: the release ships no waveforms, so there is nothing
      # for it to decode and no fold CSVs on the Hub to fetch.
      df = load_eye_tracking_ecg(root)
      df.shape                     # (14742, 35)
      len(load_respondents(root))  # 63 readers, 10 sessions each

      # Sentinels are already NaN, so this averages only AOIs that were looked at.
      leads = df[df.aoi_kind == "lead"]
      leads.groupby("Group")["Hit_time_G"].mean().round(0)
      # Consultant        7266.0    <- ms to first gaze on a lead box
      # Technician        7861.0
      # Fellow            7942.0
      # Med 1            11305.0
      # CCU Nurse        12098.0

      # Group by aoi_lead, not Label: labels are scoped per image.
      leads.groupby("aoi_lead")["Time_spent_G_Percentage"].mean().nlargest(3).round(2)
      # aoi_lead
      # II     6.35
      # V2     5.76
      # aVF    5.33

      # The coarse two-region scoring of the same 630 sessions.
      ls = load_aoi_metrics(root, table="long_short")
      ls.groupby("aoi_area")["Time_spent_G_Percentage"].mean().round(1)
      # Long     33.1      <- the rhythm strips
      # Short    53.2      <- the twelve short lead traces

      # Stimuli come back as image paths, never as decoded signals.
      stimulus_image_path(root, "ST elevation MI").name    # 'STEMI.png'

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/eye-tracking-ecg/1.0.0/" }
      - { label: "Paper (JMIR Human Factors, 2022)", url: "https://doi.org/10.2196/34058" }
      - { label: "Example script", url: "https://github.com/vlbthambawita/ECGBench/blob/main/examples/load_eye_tracking_ecg.py" }
---
