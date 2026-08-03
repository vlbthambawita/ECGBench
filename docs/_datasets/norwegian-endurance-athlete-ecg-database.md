---
slug: "norwegian-endurance-athlete-ecg-database"
name: "Norwegian Endurance Athlete ECG Database"
category: "12-lead-physionet"
order: 13
status: "completed"
source_url: "https://physionet.org/content/norwegian-athlete-ecg/1.0.0/"
url_label: "physionet.org"
format: "12-lead · 10 s · 500 Hz"
patients: "28"
records: "28"
access: "open"
license: "CC BY 4.0"
origin_institution: "University of Oslo"
origin_country: "Norway"
leads: 12
paper_title: "Singstad, PhysioNet, 2022"
paper_doi: "https://doi.org/10.13026/qpjf-gk87"
search_keywords: "norwegian athlete ecg norway oslo endurance rower kayak cyclist marquette sl12 ge cardiologist athlete's heart"

sections:
  - type: description
    title: "Overview"
    body: |
      28 twelve-lead resting ECGs, **one per elite Norwegian endurance athlete** —
      24 rowers, 2 kayakers and 2 cyclists, 19 men and 9 women aged 20–43,
      training roughly 800 hours a year — recorded at the University of Oslo in
      February and March 2020 on a GE MAC VUE 360.

      This is the **smallest dataset in ECGBench**, and size is not what it is
      for. Every record carries **two independent interpretations**, appended to
      its WFDB header as comment lines: one from the GE Marquette SL12 algorithm
      (version 23, v243) and one from a cardiologist trained in athlete ECG
      interpretation using international criteria. The dataset exists to show how
      badly general-purpose automated interpretation over-calls findings in hearts
      remodelled by endurance training, and it does: **SL12 reads 13 of the 28
      records as borderline or abnormal where the cardiologist reads them as
      normal**, and raises a critical `ACUTE MI/STEMI` alert on 4 athletes, three
      of whom the cardiologist signs off as a plain "Normal ECG".

      So SL12 is the **system under test here, not the ground truth**. Training a
      model against the `sl12_*` fields inverts the point of the release.

      No echocardiography was performed, so the authors note it is not confirmed
      whether these athletes had athletic remodelling of the heart. Each athlete
      contributes exactly one record, so there is no within-patient leakage and
      folds are stratified but not grouped.

  - type: description
    title: "The waveform amplitudes are not calibrated"
    body: |
      **Every lead of every record was independently min–max normalised to the
      full signed 16-bit range.** All 336 lead-records have their minimum at
      exactly `-32767` and their maximum within 8 counts of `+32767`, so with the
      headers' nominal `50000/mV` gain every lead spans exactly ±0.6553 mV.

      This is **undocumented upstream** — PhysioNet states only "16-bit,
      50000/mV" — and was established from the files. Reproduce it in one line:

      ```python
      import glob, numpy as np, wfdb
      print({wfdb.rdrecord(f[:-4], physical=False).d_signal.min() for f in glob.glob("*.hea")})
      # {-32767}   -- every lead of every record, not a coincidence
      ```

      Two consequences, both load-bearing:

      - **Amplitude is meaningless in absolute terms and incomparable between
        leads.** Voltage criteria — LVH thresholds, ST elevation in millimetres —
        cannot be computed from this release, which is a real loss given that
        athlete ECG criteria are largely voltage criteria. Morphology and timing
        are unaffected.
      - **There is no scale factor that fixes it.** The normalisation was
        per-lead, so the information needed to invert it is not in the release.
        `signal_unit_scale` is `1.0` and `units=` cannot help; ECGBench hands back
        what the headers declare and says so rather than implying calibration.

      A knock-on effect: the `missing_leads` and `flat_line` checks **cannot fire**
      on this dataset. A disconnected or railed lead would be rescaled to full
      amplitude like any other, so the two checks that exist to catch dead leads
      are blind here. `amplitude_range_mv` is therefore set to a deliberately
      tight ±1.0 — not a physiological range, but one that still passes all 28
      records while flagging any future record that is not normalised this way.

  - type: table
    title: "The two interpretations disagree"
    headers: ["Overall verdict", "GE Marquette SL12", "Cardiologist"]
    rows:
      - ["Normal ECG", "4 (14.3%)", "26 (92.9%)"]
      - ["Otherwise normal ECG", "9 (32.1%)", "—"]
      - ["Borderline ECG", "7 (25.0%)", "2 (7.1%)"]
      - ["Abnormal ECG", "8 (28.6%)", "—"]
      - ["**Normal or otherwise-normal**", "**13 (46.4%)**", "**26 (92.9%)**"]
      - ["Verdict strings identical", "4 of 28", "—"]
      - ["Critical `ACUTE MI/STEMI` alert", "4 of 28", "0 of 28"]

  - type: table
    title: "Cardiologist findings (multi-label)"
    headers: ["Finding", "Records", "Share"]
    rows:
      - ["Normal sinus rhythm", "17", "60.7%"]
      - ["Sinus arrhythmia", "7", "25.0%"]
      - ["Sinus bradycardia", "5", "17.9%"]
      - ["Left ventricular hypertrophy", "3", "10.7%"]
      - ["First degree AV block", "3", "10.7%"]
      - ["Right axis deviation", "2", "7.1%"]
      - ["Incomplete right bundle branch block", "2", "7.1%"]
      - ["Left axis deviation", "1", "3.6%"]
      - ["Possible left ventricular hypertrophy", "1", "3.6%"]
      - ["Left atrial enlargement", "1", "3.6%"]
      - ["Misplaced electrodes", "1", "3.6%"]

  - type: description
    title: "About those counts"
    body: |
      All 58 shipped files were verified against the release's own
      `SHA256SUMS.txt` before any figure here was computed — all OK. The record
      and subject counts match the release description exactly at **28 and 28**,
      so there is nothing to reconcile.

      Every figure above is **recomputed from the 28 `.hea` files**, which are the
      only place the labels exist — the release ships no metadata table at all
      (just the `.dat`/`.hea` pairs, `RECORDS`, `SHA256SUMS.txt` and
      `LICENSE.txt`). ECGBench parses the `#SL12:` and `#C:` comment lines; see
      `ecgbench/labels/norwegian_athlete_ecg.py`.

      **The findings table is multi-label and does not sum to 28.** The 28
      cardiologist readings carry 43 findings between them (1–4 each); the 28 SL12
      readings carry 55 across 27 distinct statements. Every record carries at
      least one finding, so none falls into no class.

      **The findings table is case-folded; the label loader is not.** The source
      spells the same finding two ways — `normal sinus rhythm` once against
      `Normal sinus rhythm` 16 times, and `first degree AV block` once against
      `First degree AV block` twice. The table above folds case (hence 17 and 3);
      `cardiologist_findings` preserves each string verbatim, because silently
      rewriting a clinician's text is worse than reporting it. Fold case yourself
      if you are counting.

      **The stratification label is a different quantity from this table.**
      Folds are stratified on `cardiologist_primary_rhythm` — the rhythm each
      cardiologist reading opens with, one per record, 16 / 7 / 5. The
      cardiologist's *verdict* would be degenerate for that purpose: 26 of 28 are
      "Normal ECG". Use `cardiologist_findings` for training targets, not the
      stratification label.

      **There is no supervised task with a positive class on human ground truth.**
      Not one record is read as abnormal by the cardiologist. Anyone wanting a
      normal/abnormal target has only the 2 borderline records, or the SL12
      output — which is the thing being evaluated, not a label.

      **No per-record demographics ship.** Age, sex and sport are published only
      as cohort aggregates on the landing page, so they cannot be joined to a
      record and ECGBench does not expose them.

      Three parsing traps live in these header lines, all handled and all
      regression-tested:

      - **A comma does not reliably end a statement.** Three GE statements contain
        commas of their own — `ST elevation, consider early repolarization,
        pericarditis, or injury` alone splits into four fragments on a naive
        comma split, turning 2 records into 7 spurious findings.
      - **Capitalisation cannot be used to detect the continuations**, because the
        cardiologist writes genuine findings in lowercase (`Sinus bradycardia,
        normal sinus rhythm, First degree AV block`). ECGBench masks an explicit
        list of known comma-carrying statements instead of guessing.
      - **One verdict is misspelt**: `ath_010` ends `Abnormal EKG`, not "ECG". The
        normalised field folds it in; the raw string keeps it.

  - type: table
    title: "Validation summary (500 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "28", "all records, with is_valid + quality_issues"]
      - ["clean", "28", "100% pass rate — no record excluded"]
      - ["excluded", "0", "no NaN samples, uniform 12 × 5000, peak |amplitude| 0.6553 mV"]

  - type: description
    title: "Folds: use cross-validation, not the default split"
    body: |
      With 28 records over 10 folds, folds 1–8 hold 3 records each and folds 9 and
      10 hold 2 — so the default mapping gives **24 train / 2 val / 2 test**.
      Both val and test end up holding only `Normal sinus rhythm` records, and
      with 2-record folds and a 16/7/5 rhythm split no assignment avoids that.
      scikit-learn also warns that the least populated class has 5 members against
      `n_splits=10`; that is expected and does not raise.

      Treat this as a **cross-validation harness rather than a train/val/test
      split**: pass `split=None` with `fold_numbers` and rotate which folds you
      hold out. The rhythm is spread as evenly as 28 records permit across
      folds 1–8.

      ```python
      from ecgbench import ECGDataset

      # Custom CV: hold out fold 3, train on the rest. With split=None each
      # sample's ["split"] reports that record's own default split.
      train = ECGDataset("norwegian_athlete_ecg", split=None,
                         fold_numbers=[1, 2, 4, 5, 6, 7, 8, 9, 10],
                         data_path="/path/to/norwegian-athlete-ecg/1.0.0/")
      held  = ECGDataset("norwegian_athlete_ecg", split=None, fold_numbers=[3],
                         data_path="/path/to/norwegian-athlete-ecg/1.0.0/")
      len(train), len(held)               # (25, 3)
      ```

  - type: code
    title: "Getting the data"
    language: bash
    body: |
      # The 1.7 MB zip is genuinely public -- no PhysioNet credentials needed.
      wget https://physionet.org/static/published-projects/norwegian-athlete-ecg/norwegian-endurance-athlete-ecg-database-1.0.0.zip
      unzip norwegian-endurance-athlete-ecg-database-1.0.0.zip

      # Verify before trusting any figure -- all 58 files should report OK.
      cd norwegian-endurance-athlete-ecg-database-1.0.0 && sha256sum -c SHA256SUMS.txt

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      # Writes ecgbench_metadata.csv into the dataset root on first run: the
      # release ships no metadata table, and the validation engine re-reads that
      # file from disk. The dataset root must be writable.
      ecgbench splits --dataset norwegian_athlete_ecg --data-path /path/to/norwegian-athlete-ecg/1.0.0/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # Fold CSVs come from the HuggingFace Hub by default; only the waveforms
      # need to be local.
      ds = ECGDataset(
          "norwegian_athlete_ecg",
          split="train",
          data_path="/path/to/norwegian-athlete-ecg/1.0.0/",
          labels=True,
      )

      len(ds)                                         # 24
      ds[0]["signal"].shape                           # (12, 5000)  -- 10 s at 500 Hz
      ds[0]["record_id"]                              # 'ath_001'

      # Both readings of the same record, kept separate on purpose:
      ds[0]["labels"]["cardiologist_findings"]         # ['Sinus arrhythmia']
      ds[0]["labels"]["cardiologist_verdict"]          # 'Normal ECG'
      ds[0]["labels"]["sl12_findings"]                 # ['Sinus bradycardia with marked sinus arrhythmia', 'Right axis deviation']
      ds[0]["labels"]["sl12_verdict"]                  # 'Borderline ECG'
      ds[0]["labels"]["cardiologist_primary_rhythm"]   # 'Sinus arrhythmia'  -- the stratification label

      # The dataset's headline result, in two lines. labels_df is aligned
      # POSITIONALLY with metadata_df and carries a RangeIndex, not record IDs.
      ds.labels_df["sl12_overcalls"].sum()             # 12 of 24 in this split
      ds.labels_df["sl12_critical_test_result"].notna().sum()   # 3 of 24

      # Amplitudes are per-lead normalised, so every lead spans the same range.
      # This is the dataset, not a bug -- see "The waveform amplitudes are not
      # calibrated" above.
      sig = ds[0]["signal"]
      (sig.max(dim=1).values - sig.min(dim=1).values).min()   # tensor(1.3106)

      # Lead order is uppercase AVR/AVL/AVF, as PTB-XL spells them. Always select
      # by name: index 4 is AVL here but aVF in MIMIC-IV-ECG.
      two = ECGDataset("norwegian_athlete_ecg", split="train", leads=["II", "V5"],
                       data_path="/path/to/norwegian-athlete-ecg/1.0.0/")
      two[0]["signal"].shape                           # (2, 5000)

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/norwegian-athlete-ecg/1.0.0/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/qpjf-gk87" }
      - { label: "Example script", url: "https://github.com/vlbthambawita/ECGBench/blob/main/examples/load_norwegian_athlete_ecg.py" }
---
