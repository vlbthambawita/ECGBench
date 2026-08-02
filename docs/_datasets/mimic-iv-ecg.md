---
slug: "mimic-iv-ecg"
name: "MIMIC-IV-ECG"
category: "12-lead-physionet"
order: 4
status: "completed"
source_url: "https://physionet.org/content/mimic-iv-ecg/1.0/"
url_label: "physionet.org"
format: "12-lead · 10 s · 500 Hz"
patients: "161,352"
records: "800,035"
access: "credentialed"
license: "PhysioNet DUA"
origin_institution: "Beth Israel Deaconess Medical Center"
origin_country: "USA — Boston, MA"
leads: 12
paper_title: "Gow et al."
paper_doi: "https://doi.org/10.13026/4nqg-sb35"
search_keywords: "mimic iv ecg usa boston beth israel mit machine measurements report credentialed largest"

related:
  - slug: "mimic-iv-ecg-demo"
    relation: "contains"
    shares_records: true
    verified: true
    note: >
      658 of the demo's 659 records were matched here on (subject_id, minute), and
      all 92 demo subjects appear. But study_ids are renumbered into a disjoint
      range and timestamps are truncated to the minute in this release, so the two
      cannot be joined on study_id — a study_id comparison shows 0% overlap and is
      misleading.
  - slug: "mimic-iv-ecg-ext-icd"
    relation: "has_derivative"
    shares_records: true
    verified: true
    note: >
      Adds ED and hospital ICD-10 diagnoses. Covers all 800,035 studies and all
      161,352 subjects exactly, and joins cleanly on study_id.

sections:
  - type: description
    title: "Overview"
    body: |
      **800,035 twelve-lead ECGs from 161,352 patients** at Beth Israel Deaconess
      Medical Center — the largest dataset in this catalogue by two orders of
      magnitude, about 96.5 GB of waveforms. Each record is a fixed 10 s at
      500 Hz. Access is **credentialed** under the PhysioNet DUA.

      Do not confuse it with **MIMIC-IV-ECG Demo**, the open 659-record sample:
      that one ships identifiers only, while this release adds
      `machine_measurements.csv` and so is the version that actually has labels.
      They are separate ECGBench configs, `mimic_iv_ecg` and `mimic_iv_ecg_demo`.

      **The stored lead order transposes aVF and aVL** — `I, II, III, aVR, aVF,
      aVL, V1–V6`, identical in all 3,000 headers sampled. `signal[4]` is aVF
      here and aVL in every other 12-lead dataset in ECGBench, so a model trained
      across datasets by index silently crosses two leads. `leads=[...]` selects
      by name and fixes it.

      **Patient grouping is not optional at this scale.** 92.8% of all studies
      come from patients who contributed more than one, and a single patient
      contributed 260. Folds are grouped by `subject_id`.

      Labels are the ECG cart's own **free-text report** — up to 18 lines per
      study — plus nine interval and axis measurements. They are machine output,
      not an adjudicated cardiologist reading, and they come with two traps
      documented below: the first report line is not always a rhythm, and the
      numeric measurements encode "not measurable" as integer sentinels rather
      than as missing values.

  - type: table
    title: "Studies per patient"
    headers: ["Studies", "Patients", "% of patients", "Studies contributed"]
    rows:
      - ["1", "57,344", "35.5%", "57,344"]
      - ["2", "29,218", "18.1%", "58,436"]
      - ["3–4", "28,666", "17.8%", "97,249"]
      - ["5–9", "25,322", "15.7%", "164,799"]
      - ["10–19", "13,700", "8.5%", "183,141"]
      - ["20–49", "6,263", "3.9%", "179,452"]
      - ["50–259", "838", "0.5%", "59,354"]
      - ["260", "1", "0.0%", "260"]
      - ["**total**", "**161,352**", "", "**800,035**"]

  - type: table
    title: "Most frequent first report line"
    headers: ["First line of the machine report", "Records", "Share"]
    rows:
      - ["sinus rhythm", "381,884", "47.7%"]
      - ["sinus bradycardia", "71,034", "8.9%"]
      - ["sinus tachycardia", "54,665", "6.8%"]
      - ["atrial fibrillation", "41,209", "5.2%"]
      - ["sinus rhythm with borderline 1st degree a-v block", "15,428", "1.9%"]
      - ["atrial fibrillation with rapid ventricular response", "14,925", "1.9%"]
      - ["sinus rhythm with 1st degree a-v block", "14,788", "1.8%"]
      - ["sinus arrhythmia", "13,764", "1.7%"]
      - ["sinus rhythm with pac(s)", "10,539", "1.3%"]
      - ["*** consider acute st elevation mi ***", "8,516", "1.1%"]
      - ["ventricular pacing", "8,445", "1.1%"]
      - ["--- warning: data quality may affect interpretation ---", "7,079", "0.9%"]
      - ["(938 further values)", "157,758", "19.7%"]

  - type: description
    title: "About those counts"
    body: |
      Recomputed from the shipped `record_list.csv` and `machine_measurements.csv`.
      The catalogue previously carried the rounded figures "~800,000" and
      "~160,000"; the exact counts are **800,035 records** and **161,352
      patients**, and both are confirmed twice over — `record_list.csv` and
      `machine_measurements.csv` hold exactly the same 800,035 `study_id`s, and
      `SHA256SUMS.txt` accounts for exactly 800,035 `.dat` and 800,035 `.hea`
      files. The MIMIC-IV-ECG-Ext-ICD derivative independently covers the same
      800,035 studies and 161,352 subjects.

      **Verify your own copy of `machine_measurements.csv`.** It is the file every
      label comes from, and the release publishes its SHA-256 in
      `SHA256SUMS.txt`:
      `56f6b1413221bce95bd6f48b28ca1acf27ae0b073d6f2c1d12f3af7500eabbb6`, for
      800,035 rows — one per study. A filtered copy under the same name produces
      records with no labels and figures that do not match this page; ECGBench
      logs a warning naming the shortfall when the file does not cover every
      study.

      **The report table above is the *first line only*, and it is not a rhythm
      label.** It is `report_0`, normalised (lower-cased, whitespace collapsed,
      trailing period dropped — which is what turns 1,571 raw values into 950).
      That line is usually a rhythm statement, but 7,079 records lead with a
      data-quality warning, 4,980 with a note that age was not entered, and 8,516
      with a finding rather than a rhythm. Exactly **one record of 800,035 has no
      report text at all**. Train on `report_text` — the populated lines joined —
      not on `primary_report`.

      **The report uses the cart's vocabulary, not clinical prose.** Keyword
      extraction over `report_text` has to match what the machine actually wrote:
      `infarct` appears in **179,588** records while `myocardial infarction`
      appears in **211**, and `hypertrophy` in 69,282 against `left ventricular
      hypertrophy` in 50,748. Searching for the clinical phrase suggests this
      hospital population has almost no infarcts.

      **The stratification label is a pooled version of that first line.**
      Classes with fewer than 1,000 records are pooled into `OTHER`, leaving 47
      named classes covering 92.6% of records plus `OTHER` at 58,933. It exists so
      folds are balanced; it is not a clinical grouping.

      **Nine numeric measurements, all 100% populated and none of them complete.**
      Missing values are integer sentinels, not blanks:

      | Sentinel | Meaning | Records affected |
      |---|---|---|
      | `29999` | wave timing not measurable | `p_onset` 123,434 · `p_end` 230,323 |
      | `32767` / `-32768` | axis not measurable | `p_axis` 7,199 · `t_axis` 1,440 · `qrs_axis` 1 |
      | `65535` | RR interval not measurable | 5 |

      That these mean "not measurable" rather than "not recorded" is checkable:
      `p_end` is `29999` in **100.0%** of atrial-fibrillation records and
      `p_onset` in 90.7% of them, against **0%** `p_onset` sentinels among
      sinus-rhythm records — atrial fibrillation has no organised P wave to
      measure. ECGBench converts all of them to `NaN`, which is lossless because
      the source columns contain no genuine blanks. It also drops values outside a
      physiologic range, which is why `p_axis` ends up missing in 132,388 records
      rather than only the 7,199 exact sentinels. On what survives,
      `qrs_onset < qrs_end < t_end` holds in 99.98% of records and the derived QRS
      duration has a median of 94 ms.

      **No age or sex ships in this module.** Both must be joined from MIMIC-IV
      itself on `subject_id`. Timestamps are HIPAA date-shifted into the future,
      which is why `ecg_time` runs from 2097 to 2211 and why no real acquisition
      date can be recovered.

  - type: table
    title: "Validation summary (500 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "800,035", "all records, with is_valid + quality_issues"]
      - ["clean", "786,699", "98.33% pass rate"]
      - ["excluded", "13,336", "10,554 with NaN samples, 1,950 with an all-zero lead, 989 amplitude outliers"]

  - type: description
    title: "About the excluded records"
    body: |
      1.67% of records fail validation, and the reasons are worth knowing because
      they are properties of the waveforms, not of the metadata.

      **NaN samples (10,554 records, 1.32%)** — the dominant reason. These records
      have literal `NaN` values in the signal, so they cannot be trained on
      without imputation. If you have seen a MIMIC-IV-ECG pipeline that filters
      `machine_measurements.csv` down to 789,481 rows, this is why: that is
      exactly 800,035 − 10,554, and the removed studies are **precisely** this set
      (verified by set equality, not by count). ECGBench reaches the same
      conclusion from the waveforms rather than from a pre-filtered CSV.

      **All-zero leads (1,950 records)** — one or more leads recorded as exactly
      0.0. `missing_leads` catches these; they are not NaN.

      **Amplitude outliers (989 records)** — beyond ±10 mV. Rare here: the 16-bit
      rail at this gain is ±163.8 mV, so these are genuine excursions rather than
      saturation, and 15 records additionally trip `flat_line`.

      Fold membership is identical between `original/` and `clean/` — `clean/` is a
      row subset, never a re-split — so a record excluded here keeps the fold it
      would have had.

      **No subject spans a fold.** Verified after the run: 0 of 161,352 subjects
      appear in more than one of the 10 folds, and none appears in more than one
      of train/val/test. Folds are even to within one record (80,003–80,004) and
      16,127–16,145 subjects each.

  - type: description
    title: "Splits are generated locally, not downloaded"
    body: |
      **ECGBench does not publish fold CSVs for this dataset.** They contain
      identifiers only, but 800,035 `study_id`s and 161,352 `subject_id`s are
      still data derived from a credentialed source, and the ECGBench Hub
      repository is public and ungated. MIMIC-IV-ECG is governed by the PhysioNet
      Credentialed Health Data Use Agreement, so those identifiers stay with the
      people who signed it.

      Instead the split is distributed as a **recipe**: the configuration ships
      in the package, and you regenerate the identical partition on your own
      copy. Because fold assignment is a deterministic function of the input
      table and a fixed seed, this reproduces the canonical split exactly rather
      than merely a similar one. `ecgbench splits` writes a `manifest.json`
      recording the seed, the SHA-256 of every input file, the record counts and
      a **fold digest** — a hash over the whole record-to-fold mapping — and
      ECGBench ships a reference manifest to check yours against:

      ```python
      from ecgbench import verify_splits
      verify_splits("mimic_iv_ecg", "output/mimic_iv_ecg")   # raises on mismatch
      ```

      This matters more than it might appear. A split is only reproducible if the
      input is byte-identical, and local copies do get filtered: we encountered a
      `machine_measurements.csv` reduced to 789,481 of 800,035 rows, which
      silently changes the stratification and therefore the folds. The manifest's
      input checksums catch exactly that, and the expected values are the ones
      PhysioNet publishes in its own `SHA256SUMS.txt`
      (`record_list.csv` → `b6f1130…`, `machine_measurements.csv` → `56f6b14…`).

      Attempting to load this dataset from the Hub raises
      `SplitsNotPublishedError` with the command above, rather than a bare 404,
      and `ecgbench upload` refuses to publish it.

  - type: code
    title: "Building the splits (required — they are not downloadable)"
    language: bash
    body: |
      # 1. Generate the canonical split from your own credentialed copy.
      ecgbench splits --dataset mimic_iv_ecg --data-path /path/to/mimic-iv-ecg/1.0/

      # 2. Confirm it is the canonical partition, not merely a plausible one.
      python -c "from ecgbench import verify_splits; \
                 print(verify_splits('mimic_iv_ecg', 'output/mimic_iv_ecg')['ok'])"

      # 3. Make the fold tables visible to the loader.
      cp -r output/mimic_iv_ecg/clean output/mimic_iv_ecg/original \
            /path/to/mimic-iv-ecg/1.0/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      STANDARD_12 = ["I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"]

      # metadata_source="local" is required here: the fold CSVs are generated by
      # you (see above), never downloaded. 800k records, so one fold is usually
      # enough to start. leads= reorders the stored aVF/aVL back to the
      # conventional order, by name.
      ds = ECGDataset(
          "mimic_iv_ecg",
          split="train",
          data_path="/path/to/mimic-iv-ecg/1.0/",
          metadata_source="local",
          fold_numbers=[1],
          leads=STANDARD_12,
          labels=True,
      )

      len(ds)                                 # 78655  (fold 1 of the train split)
      ds[0]["signal"].shape                   # (12, 5000)
      ds[0]["labels"]["report_text"]          # 'Demand pacing | Pacemaker rhythm - no further analysis | Abnormal ECG'
      ds[0]["labels"]["primary_report"]       # 'demand pacing'  <- first line only, not a rhythm
      ds[0]["labels"]["qrs_duration"]         # 134.0 ms, derived from qrs_end - qrs_onset
      ds[0]["labels"]["p_axis"]               # nan — paced rhythm, so no P wave to measure

      # Without leads= the stored order applies, and signal[4] is aVF, not aVL:
      ds.config.lead_names   # ['I','II','III','aVR','aVF','aVL','V1',...,'V6']

      # A multi-hot target from the free text:
      import pandas as pd
      # Match the cart's wording: "infarct", not "myocardial infarction".
      FINDINGS = ["atrial fibrillation", "sinus bradycardia", "infarct"]
      text = ds.labels_df["report_text"].fillna("").str.lower()
      targets = pd.DataFrame({f: text.str.contains(f).astype(int) for f in FINDINGS})

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/mimic-iv-ecg/1.0/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/4nqg-sb35" }
      - { label: "Open demo subset (659 records, no labels)", url: "https://physionet.org/content/mimic-iv-ecg-demo/0.1/" }
      - { label: "MIMIC-IV-ECG-Ext-ICD (ICD-10 diagnoses)", url: "https://physionet.org/content/mimic-iv-ecg-ext-icd/1.0.1/" }
---
