---
slug: "mit-bih-normal-sinus-rhythm-database"
name: "MIT-BIH Normal Sinus Rhythm Database"
category: "two-lead"
order: 4
status: "completed"
source_url: "https://physionet.org/content/nsrdb/1.0.0/"
url_label: "physionet.org"
format: "2-lead (ECG1 + ECG2, unnamed) · 23.1–26.0 h · 128 Hz · WFDB"
patients: "18"
records: "18"
access: "open"
license: "ODC-By 1.0"
origin_institution: "Beth Israel Hospital"
origin_country: "USA"
leads: 2
paper_title: "No database paper — cite PhysioNet"
paper_doi: "https://doi.org/10.13026/C2NK5R"
search_keywords: "mit-bih normal sinus rhythm nsrdb healthy control negative class usa beth israel hospital holter 24h heart rate variability hrv sdnn rmssd beat annotations signal quality two-lead long-term"

sections:
  - type: description
    title: "Overview"
    body: |
      **The reference normal cohort.** 18 full-day two-lead Holter recordings of
      subjects referred to the Arrhythmia Laboratory at Boston's Beth Israel
      Hospital who, on review, were **found to have had no significant
      arrhythmias** — 5 men aged 26 to 45 and 13 women aged 20 to 50. 437.5 hours
      of signal at 128 Hz, and **1,729,629 reference beats of which 127 are not
      normal**.

      **There is no diagnostic label here, and that is the point.** The release
      ships no rhythm annotations at all, so `cohort_label` is
      `normal_sinus_rhythm` for every record. Use this database as a control
      group, a negative class, a pretraining corpus, or the baseline for normal
      heart-rate variability — not as a classification task. Folds are stratified
      on the subject's sex instead, which is the one axis PhysioNet documents
      about this cohort.

      **Beat annotation stops long before the signal does, and nothing says so.**
      It covers **79.5%** (19090) to **95.7%** (16539) of each record. The
      remaining **52.9 hours — 12.1% of the release** — carry waveform with no
      reference behind it, one to five hours at the end of every record. This is
      the largest trap in the dataset and it has its own section below.

      **Records are a full day: 10,659,840 to 11,960,320 samples, 85–96 MB of
      float32 each.** Batching needs a `window=(start, length)`, which is read at
      load time rather than cropped afterwards. Length is *not* uniform — 23.13 h
      to 25.96 h — so a window sized for one record need not fit another.

      **The two channels are not named leads.** The headers call them `ECG1` and
      `ECG2` and the release states no electrode placement anywhere. Its sibling
      MIT-BIH Arrhythmia Database, from the same laboratory, does document
      MLII/V1; this one gives you two channel positions and no anatomy. Do not
      carry the mitdb naming across.

  - type: table
    title: "The 18 records, recomputed from the files"
    headers: ["Record", "Age", "Sex", "Hours", "Beats", "Ectopic", "Mean HR", "SDNN ms", "Noisy %", "Annotated %"]
    rows:
      - ["16265", "32", "M", "25.46", "100,243", "27", "75.3", "170.9", "1.41", "87.4"]
      - ["16272", "20", "F", "25.00", "87,758", "1", "65.2", "140.5", "**9.60**", "93.8"]
      - ["16273", "28", "F", "24.64", "89,845", "5", "72.5", "146.0", "0.55", "83.8"]
      - ["16420", "38", "F", "23.98", "102,067", "6", "78.8", "101.1", "0.79", "90.1"]
      - ["16483", "42", "M", "25.96", "104,334", "4", "82.3", "88.8", "0.35", "81.4"]
      - ["16539", "35", "F", "24.58", "108,282", "17", "76.7", "150.7", "0.86", "**95.7**"]
      - ["16773", "26", "M", "23.97", "81,989", "27", "63.0", "**245.6**", "0.52", "90.5"]
      - ["16786", "32", "F", "24.49", "101,615", "10", "72.5", "115.9", "**0.23**", "95.3"]
      - ["16795", "20", "F", "23.58", "86,872", "**0**", "69.8", "212.3", "0.37", "88.0"]
      - ["17052", "45", "F", "**23.13**", "87,356", "2", "68.7", "158.6", "1.05", "91.7"]
      - ["17453", "32", "F", "24.38", "100,658", "3", "81.1", "103.2", "1.08", "84.9"]
      - ["18177", "26", "F", "**25.96**", "115,911", "3", "86.1", "116.9", "0.66", "86.6"]
      - ["18184", "34", "F", "23.75", "102,313", "**0**", "81.6", "98.1", "0.61", "88.0"]
      - ["19088", "41", "F", "23.80", "97,961", "4", "82.0", "119.5", "2.72", "84.7"]
      - ["19090", "45", "M", "24.18", "81,391", "9", "70.7", "99.9", "0.81", "**79.5**"]
      - ["19093", "34", "M", "23.23", "75,106", "6", "62.5", "133.6", "0.66", "86.3"]
      - ["19140", "38", "F", "24.17", "96,596", "**0**", "80.6", "100.9", "0.43", "82.7"]
      - ["19830", "50", "F", "23.22", "109,332", "3", "**87.1**", "131.3", "2.17", "90.8"]
      - ["**total**", "20–50", "13 F / 5 M", "**437.49**", "**1,729,629**", "**127**", "62.5–87.1", "88.8–245.6", "1.39", "87.8"]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the 18 headers and `.atr`
      files, after verifying the shipped data against the release's own
      `SHA256SUMS.txt` — **all 74 dataset files match**. (That file also lists 22
      `.hea-`, `.hea--` and `.atr-` entries, which are superseded backup copies
      PhysioNet keeps beside the current revisions. They are absent from this
      copy, and the record list comes from the shipped `RECORDS` file, so they
      could not enter the partition either way.)

      **The published cohort description reproduces exactly**, which is unusual
      enough in this catalogue to be worth saying: PhysioNet states "5 men, aged
      26 to 45, and 13 women, aged 20 to 50", and the header comments give
      precisely that — men at 26, 32, 34, 42, 45 and women at 20, 20, 26, 28, 32,
      32, 34, 35, 38, 38, 41, 45, 50. There is no changelog and no discrepancy to
      explain.

      **Ectopy exists, but only just.** 127 of 1,729,629 beats are not normal —
      **7.3 per 100,000**. Broken down: 91 supraventricular premature (`S`), 26
      ventricular premature (`V`), 8 fusion (`F`) and 2 nodal premature (`J`).
      Three records (16795, 18184, 19140) have none at all; the worst two, 16265
      and 16773, have 27 each. That is what "no significant arrhythmias" means
      operationally. There is no usable ectopy class here — `n_ectopic_beats` is
      exposed because "how clean is clean" is a real question, not because it can
      be trained on.

      **`cohort_label` is a constant, and it is not the stratification label.**
      It records PhysioNet's assertion about the cohort so that a user combining
      this database with an arrhythmia one has a record-level class to join on.
      Nothing in the files derives it. Folds are stratified on **sex** (13 F / 5
      M) — see the fold section below. Train on `mean_hr_bpm`, `sdnn_ms`,
      `rmssd_ms`, the `beat_*` counts or the signal-quality seconds; never on
      `stratify_class`.

      **The HRV figures are descriptive, not a result.** `mean_hr_bpm`,
      `sdnn_ms` and `rmssd_ms` are computed over RR intervals in [0.3 s, 2.0 s]
      from the reference beats. That filter is load-bearing: without it the
      multi-hour unannotated gaps below enter as single enormous "RR intervals"
      and every figure is meaningless. 894 intervals are rejected across the
      release. These are whole-record summaries over ~24 h of mixed activity and
      sleep, not the segmented, artefact-corrected analysis an HRV study would
      run.

  - type: description
    title: "The annotations stop before the signal does"
    body: |
      This is the thing to know before choosing a window. Beat annotation covers
      **79.5% to 95.7%** of each record and then simply stops; the `.dat` file
      keeps going for another **1.1 to 5.0 hours**, and neither the header nor
      the annotation file says anything about it.

      | | Record | Annotated | Unannotated tail |
      |---|---|---|---|
      | least covered | 19090 | 79.5% | 4.95 h |
      | | 16483 | 81.4% | 4.82 h |
      | most covered | 16539 | 95.7% | 1.06 h |
      | **release total** | | **87.8%** | **52.9 h of 437.5 h** |

      Five records (19088, 19090, 19093, 19140, 19830) additionally open with an
      unannotated **head** of 23–34 seconds. Every other record's first beat is
      within a second of the start.

      A window reaching into either region returns waveform with nothing to score
      it against. That is fine for self-supervised or unsupervised work and wrong
      for evaluating a beat detector. `annotated_secs`,
      `unannotated_head_secs`, `unannotated_tail_secs` and `annotated_fraction`
      in the labels report it per record, so a supervised window can be kept
      inside the annotated span.

  - type: description
    title: "Signal quality is annotated, per channel"
    body: |
      The shipped `ANNOTATORS` file promises "reference beat **and signal
      quality** annotations", and the second half is easy to miss. The `~`
      annotations mark quality transitions, and their WFDB `subtype` is a bitmask
      over the two channels — `0` clean, `1` ECG1 noisy, `2` ECG2 noisy, `3` both.
      Each transition opens an interval running to the next one, so ECGBench
      exposes it as **seconds per state per record**, not as a marker count: a
      one-second glitch and a three-hour noisy stretch are one marker each.

      Across the release **98.61% of the recorded time is annotated clean** —
      1.56 h of ECG1-only noise, 2.68 h of ECG2-only and 1.84 h of both. But the
      average hides the spread: **16272 is 9.60% noisy and 16786 is 0.23%**, a
      factor of 40. The `|` isolated-artifact marker varies even more, spanning
      three orders of magnitude (**52** in 16273, **30,782** in 16773), so a
      per-record metric is not comparable across records without controlling for
      it.

      The span before a record's first `~` is counted as clean. That is checked
      rather than assumed: in all 18 records the first `~` is a transition *into*
      noise (subtype 1, 2 or 3), never a return to clean, so nothing before it
      was ever marked otherwise.

  - type: table
    title: "Validation summary (128 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "18", "all records, with is_valid + quality_issues"]
      - ["clean", "18", "100% pass rate — nothing is excluded"]

  - type: description
    title: "Nothing fails validation, and two checks cannot fire"
    body: |
      All 18 records pass every check, so `original` and `clean` hold the same
      18 rows. There are no NaN samples, no flat or all-zero leads and no
      unreadable header anywhere. Two checks are worth explaining because they
      *cannot* fire:

      - **`truncated_signal` is disabled**, by leaving `expected_samples` empty.
        Records run 10,659,840 (17052) to 11,960,320 samples (16483, 18177) and
        every one is a complete full-day recording, so any single threshold would
        drop sound records as truncated. Omitting the rate disables the check
        rather than making it fire — the same escape hatch `ptbdb`, `afdb` and
        `ltafdb` use.
      - **`amplitude_range_mv` is `[-10.24, 10.235]`**, the 12-bit rail computed
        from the hardware (`adc_zero` 0 at wfdb's fallback gain of 200 puts every
        possible sample in [−2048, 2047] adu). The extreme sample anywhere in the
        release is **±5.115 mV**, half of that, so the check cannot fire on NSRDB
        1.0.0. What it guards is a mis-scaled copy — microvolts, or a re-release
        with a declared gain — which would exceed it by orders of magnitude on
        the first record.

      **One thing no threshold in that range would catch, so it is recorded
      here: record 16272 clips.** It saturates flat at ±1023 adu (±5.115 mV) for
      1,246 samples on ECG1 and 2 on ECG2; 18184 does for 3 samples on ECG2.
      1023 is the 11-bit rail — half the span the header's 12-bit `adc_res`
      declares — so the recorder's usable range was narrower than the header
      implies. It is 0.01% of one record, and 16272 is also the noisiest record
      in the release at 9.60%.

  - type: description
    title: "The amplitude is uncalibrated, by the headers' own declaration"
    body: |
      Every signal line in every header declares a gain of **`0`**, which is
      WFDB's code for "uncalibrated". `wfdb` therefore falls back to its default
      of 200 adu/mV and reports the samples as millivolts, so ECGBench's
      `signal_unit_scale` is `1.0` and nothing is rescaled.

      This is the same situation as AFDB, with one simplification: PhysioNet's
      description of AFDB states a ±10 mV range that implies a different gain
      (204.8 adu/mV) and has to be reconciled. **NSRDB's description states no
      millivolt range at all**, so the 200 that `wfdb` applies is the only
      calibration anything in or around this release supports. Waveform shape is
      unaffected either way; absolute calibration rests on wfdb's default.

  - type: description
    title: "Ten folds over 18 records, stratified on sex"
    body: |
      Two consequences of the arithmetic, stated rather than left to be
      discovered:

      - **The default split leaves one record in `val` and one in `test`.**
        ECGBench's convention is folds 1–8 → train, 9 → val, 10 → test, and 18
        records over 10 folds gives eight folds of two and two folds of one. So
        `train` holds 16 records, `val` holds 1 and `test` holds 1. For anything
        that needs a real evaluation set, use cross-validation: `split=None` with
        `fold_numbers=[...]` selects by fold from `folds.csv` and ignores the
        default layout.
      - **Half the folds contain no man.** The five men land in five different
        folds (16265, 16483, 16773, 19090, 19093 in folds 2, 5, 3, 1 and 4), so
        folds 9 and 10 — `val` and `test` — are both female. Five records cannot
        be spread over ten folds; stratification keeps them apart rather than
        letting them clump, which is the most it can do here.

      **Why sex, and not something clinical?** There is nothing clinical to use.
      `cohort_label` is one value for all 18 records, and `StratifiedKFold`
      requires at least one class holding `n_folds` members. Sex gives 13/5 and
      clears it. A median cut on age gives 10/8 and clears it by nothing.
      Anything ectopy-based fails outright — three records have no ectopic beats
      at all and the rest differ by single-digit counts out of ~100,000 beats.
      `sklearn` warns that the smallest class has 5 members; that warning is
      expected and correct.

      Folds are **ungrouped**. The header comment holds age and sex and nothing
      else — no tape number, no recorder, no subject code — and PhysioNet
      describes 18 recordings from 18 subjects, so one record per subject is the
      most that can be asserted.

  - type: description
    title: "Overlap with the other MIT-BIH Holter databases: none found"
    body: |
      NSRDB, the **MIT-BIH Arrhythmia Database** and the **MIT-BIH Atrial
      Fibrillation Database** all come from the Beth Israel Hospital Arrhythmia
      Laboratory's Holter collection, and none of them ships a subject identifier
      that would join, so the question was settled from the annotation files
      rather than assumed. RR intervals in seconds are commensurable across
      sampling rates (128 Hz here, 250 Hz for AFDB, 360 Hz for MITDB), so the
      check compares **sequences of 20 consecutive RR intervals quantised to
      8 ms**, on two half-bin-shifted grids — the same method used for LTAFDB.

      Against controls that make a null result mean something — a positive
      control re-finding each NSRDB record in its own pool at **100%**, and a
      negative control of each record against the pool of the other 17
      known-distinct subjects at a maximum of **0.0000%** — the result is:

      - **0 of 18 NSRDB records** share any sequence with the MIT-BIH Arrhythmia
        Database pool (highest 0.0000%);
      - **0 of 18** share any with the MIT-BIH AFDB pool (highest 0.0000%);
      - **0 of 18** share any with the **Long-Term AF Database** pool in
        substance: five records register a non-zero hit, but the highest is
        **0.0176%**, and repeating the check at a 30-interval signature drops it
        to one record at 0.0062%. A genuinely shared recording stays near 100% as
        the signature lengthens; these decay, which is what chance collisions
        against an 8.8-million-signature pool do.

      No `related:` edge is declared on those grounds. One limitation is worth
      stating rather than glossing: the RR signature survives *refinement* of
      annotations but not *re-detection*, so a shared recording annotated by two
      genuinely different detectors could evade it. Subject-level overlap cannot
      be checked at all, because none of these releases ships a subject
      identifier.

      Note also that PhysioNet's **MIT-BIH Normal Sinus Rhythm RR Interval
      Database** (`nsr2db`, 54 subjects) is a *different* database, not a derived
      layer over this one, and is not in this catalogue.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset nsrdb --data-path /path/to/nsrdb/1.0.0/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # Full-day records: a window is needed to batch at all, and because window=
      # is pushed into the reader it also avoids decoding the other 24 h.
      ds = ECGDataset(
          "nsrdb",
          split="train",
          data_path="/path/to/nsrdb/1.0.0/",
          window=(0, 1280),         # first 10 s at 128 Hz
          labels=True,
      )

      len(ds)                                    # 16
      ds[0]["signal"].shape                      # torch.Size([2, 1280])
      ds[0]["record_id"]                         # 16265
      ds.lead_names                              # ('ECG1', 'ECG2') — channel positions,
                                                 # not named leads
      ds[0]["labels"]["cohort_label"]            # 'normal_sinus_rhythm' — all 18 records
      ds[0]["labels"]["age"]                     # 32.0
      ds[0]["labels"]["sex"]                     # 'M'
      ds[0]["labels"]["n_beats"]                 # 100243  (27 of them not normal)
      ds[0]["labels"]["mean_hr_bpm"]             # 75.3
      ds[0]["labels"]["sdnn_ms"]                 # 170.9
      ds[0]["labels"]["annotated_fraction"]      # 0.874 — the last 3.2 h has no beats
      ds[0]["labels"]["unannotated_tail_secs"]   # 11586.0

      # There is no class to predict, so the useful targets are continuous:
      ds.labels_df["mean_hr_bpm"].describe()     # min 62.5, max 87.1 over the release

      # 17052 is the shortest record at 10,659,840 samples, so a window must fit
      # inside that rather than inside the longest record's 11,960,320, or it
      # raises WindowOutOfRangeError naming the record and its true length.

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/nsrdb/1.0.0/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/C2NK5R" }
      - { label: "PhysioBank annotation definitions", url: "https://archive.physionet.org/physiobank/annotations.shtml" }
---
