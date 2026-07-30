---
slug: "lobachevsky-university-ecg-database-ludb"
name: "Lobachevsky University ECG Database (LUDB)"
category: "12-lead-physionet"
order: 9
status: "completed"
source_url: "https://physionet.org/content/ludb/1.0.1/"
url_label: "physionet.org"
format: "12-lead · 10 s · 500 Hz · manually annotated waves"
patients: "200"
records: "200"
access: "open"
license: "ODC-By 1.0"
origin_institution: "Nizhny Novgorod City Hospital No. 5 / Lobachevsky University"
origin_country: "Russia"
leads: 12
paper_title: "Kalyakulina et al., IEEE Access, 2020"
paper_doi: "https://doi.org/10.1109/ACCESS.2020.3029211"
search_keywords: "ludb lobachevsky russia nizhny novgorod annotated delineation p qrs t boundaries"

sections:
  - type: description
    title: "Overview"
    body: |
      200 twelve-lead ECGs from 200 patients at Nizhny Novgorod City Hospital
      No. 5, each 10 seconds at 500 Hz. One record per patient, so folds need no
      grouping.

      What sets LUDB apart is its **manual delineation**: every record has 12
      annotation files, one per lead, marking the onset, peak and offset of each
      P, QRS and T wave — 58,429 annotated waves in total. This is the reference
      dataset for *delineation* algorithms rather than classification.

      ECGBench splits and loads the signals; it does not model the annotations.
      Read them directly where they sit, for example
      `wfdb.rdann(f"{data_path}/data/1", "ii")`, whose symbols are `(` for a wave
      onset, `)` for an offset, and `p` / `N` / `t` for the P, QRS and T peaks.

      Lead names are **lowercase** in every header (`i`, `ii`, `avr`, `v1` …).
      Select leads by name and the case is handled for you.

  - type: table
    title: "Rhythm distribution (all 200 records)"
    headers: ["Rhythm", "Records"]
    rows:
      - ["Sinus rhythm", "142"]
      - ["Sinus bradycardia", "24"]
      - ["Atrial fibrillation", "14"]
      - ["Sinus arrhythmia", "7"]
      - ["Sinus tachycardia", "4"]
      - ["Atrial flutter, typical", "3"]
      - ["Irregular sinus rhythm", "2"]
      - ["four further rhythms", "1 each"]

  - type: table
    title: "Diagnosis categories — how much of the data each covers"
    headers: ["Category", "Records with a finding", "Distinct labels"]
    rows:
      - ["Rhythms", "200", "9"]
      - ["Electric axis of the heart", "190", "5"]
      - ["Hypertrophies", "142", "7"]
      - ["Conduction abnormalities", "66", "8"]
      - ["Ischemia", "51", "22"]
      - ["Non-specific repolarization abnormalities", "49", "6"]
      - ["Extrasystolies", "14", "16"]
      - ["Cardiac pacing", "10", "6"]
      - ["Other states", "9", "1"]

  - type: description
    title: "About those counts"
    body: |
      Recomputed from `ludb.csv`, the structured source. The per-record `.hea`
      comments carry the same content flattened into free text and lose the
      category grouping, so the CSV is what ECGBench reads.

      Two properties of that file matter, and both are handled for you:

      - **Every string cell has a trailing newline**, and multi-value cells join
        their values with newlines rather than a delimiter. Read raw, `Ischemia`
        looks like 40 classes with at most 4 records each; split into atomic
        labels it is 22 findings that co-occur. The counts above are atomised.
      - **One age is not a number** — record 34 is recorded as `>89`. The loader
        exposes `age_raw` verbatim alongside a numeric `age` that is NaN there.

      Only `Rhythms` covers all 200 records. Everything below `Conduction
      abnormalities` is sparse enough that a classifier has very little to learn
      from: `Other states` has a single label across 9 records, and
      `Extrasystolies` has 16 labels across 14 records — more labels than
      records. `primary_rhythm` pools rhythms under 10 records into `OTHER` so
      10-fold stratification is defined; train on `rhythms`, which is verbatim.

  - type: table
    title: "Validation summary (500 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "200", "all records, with is_valid + quality_issues"]
      - ["clean", "200", "100% pass rate — no check fired on any record"]
      - ["excluded", "0", ""]

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset ludb --data-path /path/to/ludb/1.0.1/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      ds = ECGDataset(
          "ludb",
          split="train",
          data_path="/path/to/ludb/1.0.1/",
          labels=True,
      )

      sample = ds[0]
      sample["signal"].shape              # (12, 5000), millivolts
      sample["labels"]["rhythms"]         # ['Atrial fibrillation']  — multi-label
      sample["labels"]["hypertrophies"]   # ['Left atrial hypertrophy', ...]
      sample["labels"]["electric_axis"]   # 'left axis deviation'
      sample["labels"]["age_raw"]         # '57'  (one record is '>89')

      # Lead names are lowercase here; selecting by name is case-insensitive.
      two = ECGDataset("ludb", split="train", data_path="/path/to/ludb/1.0.1/",
                       leads=["II", "V5"])
      two.lead_names                      # ('ii', 'v5')

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/ludb/1.0.1/" }
      - { label: "IEEE Access paper", url: "https://doi.org/10.1109/ACCESS.2020.3029211" }
---
