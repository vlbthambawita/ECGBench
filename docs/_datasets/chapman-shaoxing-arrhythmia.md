---
slug: "chapman-shaoxing-arrhythmia"
name: "Chapman-Shaoxing (Arrhythmia)"
category: "12-lead-physionet"
order: 7
status: "completed"
source_url: "https://physionet.org/content/ecg-arrhythmia/1.0.0/"
url_label: "physionet.org"
format: "12-lead · 10 s · 500 Hz · WFDB"
patients: "45,152"
records: "45,152"
access: "open"
license: "CC BY 4.0"
origin_institution: "Chapman Univ.; Shaoxing People's Hospital & Ningbo First Hospital"
origin_country: "China / USA"
leads: 12
paper_title: "Zheng et al., Scientific Reports, 2020"
paper_doi: "https://doi.org/10.1038/s41598-020-59821-7"
search_keywords: "chapman shaoxing arrhythmia china usa ningbo ecg-arrhythmia snomed wfdb"

related:
  - slug: "chapman-shaoxing-ecg-database-10-646-patients"
    relation: "contains"
    shares_records: true
    verified: true
    note: >
      This PhysioNet release merges the Chapman-Shaoxing and Ningbo cohorts.
      10,247 of its 45,152 records are numbered in the Chapman-Shaoxing range
      (JS00001-JS10646) against 10,646 in the figshare release, so the overlap is
      large but not complete. Formats and labels differ: WFDB with SNOMED-CT #Dx
      codes here, CSV with Rhythm/Beat annotations there. Do not train on one and
      evaluate on the other.
  - slug: "ningbo-first-hospital-ecg-database-idiopathic-ventricular-arrhythmia"
    relation: "same_cohort"
    shares_records: false
    verified: false
    note: >
      Same institution, and 34,905 of the records here are the Ningbo cohort. The
      334-record figshare Idiopathic Ventricular Arrhythmia dataset is a separate
      release; whether its records also appear here is unconfirmed.
sections:
  - type: description
    title: "Overview"
    body: |
      The PhysioNet `ecg-arrhythmia` release merges the Chapman-Shaoxing
      (JS00001–JS10646) and Ningbo First Hospital (JS10647–JS45551) cohorts into
      a single WFDB tree of 45,152 twelve-lead recordings — one per patient, each
      10 seconds at 500 Hz. Diagnoses are SNOMED-CT codes covering 60+
      conditions, and age and sex are recorded alongside them.

      Unlike most datasets here, this one ships **no metadata CSV**: every
      record's demographics and diagnoses live in its own WFDB header
      (`#Age`, `#Sex`, `#Dx`). ECGBench's splitter builds a metadata table by
      scanning all 45,152 headers and caches it in the dataset root as
      `ecgbench_metadata.csv`.

      ECGBench bundles a deterministic 10-fold split stratified on the primary
      (first-listed) `#Dx` code, which is the rhythm diagnosis. No patient
      grouping is needed — the dataset is one record per patient.

      This is a different dataset from the
      [Chapman-Shaoxing 10,646-patient figshare release](https://doi.org/10.6084/m9.figshare.c.4560497.v2),
      which ECGBench exposes under the separate `chapman_shaoxing` config.

  - type: table
    title: "Primary rhythm diagnosis breakdown"
    headers: ["Code", "Condition", "Records"]
    rows:
      - ["SB",   "Sinus bradycardia", "15,807"]
      - ["SR",   "Sinus rhythm", "7,882"]
      - ["AF",   "Atrial flutter", "5,609"]
      - ["ST",   "Sinus tachycardia", "5,383"]
      - ["AFIB", "Atrial fibrillation", "1,779"]
      - ["SA",   "Sinus irregularity", "1,294"]
      - ["APB",  "Atrial premature beats", "975"]
      - ["other", "71 further primary diagnoses", "6,423"]

  - type: table
    title: "Validation summary (500 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "45,152", "all records, with `is_valid` + `quality_issues`"]
      - ["clean", "42,909", "95.0% pass rate"]
      - ["excluded", "2,243", "mostly one or more entirely flat leads"]

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      # PhysioNet serves this project from a credentialed path, so download it
      # yourself and point --data-path at the directory holding WFDBRecords/.
      ecgbench splits \
        --dataset ecg_arrhythmia \
        --data-path /path/to/ecg-arrhythmia/1.0.0/ \
        --max-workers 32

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset, ecg_collate_fn
      from torch.utils.data import DataLoader

      # Fold CSVs come from the Hub; signals are read from your local copy
      dataset = ECGDataset(
          "ecg_arrhythmia",
          split="train",
          version="clean",
          data_path="/path/to/ecg-arrhythmia/1.0.0/",
      )

      loader = DataLoader(dataset, batch_size=32, collate_fn=ecg_collate_fn)

      for batch in loader:
          signals = batch["signal"]        # (B, 12, 5000) at 500 Hz
          record_ids = batch["record_id"]  # e.g. "JS00001"
          break

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/ecg-arrhythmia/1.0.0/" }
      - { label: "Scientific Reports paper", url: "https://doi.org/10.1038/s41598-020-59821-7" }
      - { label: "Original Chapman-Shaoxing description (Scientific Data)", url: "https://doi.org/10.1038/s41597-020-0386-x" }
---
