---
slug: "ptb-diagnostic-ecg-database"
name: "PTB Diagnostic ECG Database"
category: "12-lead-physionet"
order: 3
status: "completed"
source_url: "https://physionet.org/content/ptbdb/1.0.0/"
url_label: "physionet.org"
format: "15-lead (12 + 3 Frank) · 32-120 s · 1,000 Hz"
patients: "290"
records: "549"
access: "open"
license: "ODC-By 1.0"
origin_institution: "Univ. Clinic Benjamin Franklin"
origin_country: "Germany — Berlin"
leads: 15
paper_title: "Bousseljot et al., Biomedizinische Technik, 1995"
paper_doi: "https://doi.org/10.13026/C28C71"
search_keywords: "ptb diagnostic germany berlin benjamin franklin frank leads vcg variable length"

related:
  - slug: "ptb-xl"
    relation: "same_cohort"
    shares_records: false
    verified: false
    note: >
      Both were digitised by the Physikalisch-Technische Bundesanstalt from German
      recordings of the same era, and both are conventionally called "PTB". They are
      separate collections with different formats — 15 signals at 1000 Hz and variable
      length here, 12 leads at 500/100 Hz for a fixed 10 s in PTB-XL. Whether any
      individual patient appears in both is unconfirmed; the record identifiers share
      no scheme and no crosswalk ships with either dataset.

sections:
  - type: description
    title: "Overview"
    body: |
      549 records from 290 subjects, recorded at the Benjamin Franklin University
      Clinic in Berlin. This is the most unusual dataset in the catalogue on three
      counts, all verified against the files.

      **15 signals, not 12.** The conventional leads plus the three Frank
      vectorcardiography leads `vx`, `vy`, `vz`. Leads 1-12 live in the `.dat`
      file and 13-15 in a companion `.xyz`; wfdb reads both as one record. Names
      are lowercase in every header. `ECGDataset(leads=[...])` takes the standard
      twelve by name.

      **Records are variable length** — 11 distinct lengths between 32 s and
      120 s, at 1000 Hz. They cannot be batched as they are: torch cannot stack
      tensors of differing width. Crop with a `transform`, or use `batch_size=1`.
      For the same reason `expected_samples` is deliberately empty in the config,
      which disables the truncation check rather than failing every short record.

      **No metadata file ships at all.** Every clinical field is in the per-record
      `.hea` comment block — 47 keys covering the admission diagnosis, infarction
      localisation, a full haemodynamics panel, coronary stenosis findings and the
      therapy history. ECGBench parses them into a metadata CSV on first run.

      113 of the 290 patients contributed more than one recording (up to seven),
      so folds are grouped by patient. No patient spans a fold.

  - type: table
    title: "Admission diagnosis"
    headers: ["Diagnosis", "Records", "Patients"]
    rows:
      - ["Myocardial infarction", "368", "148"]
      - ["Healthy control", "80", "52"]
      - ["(none recorded)", "27", "22"]
      - ["Cardiomyopathy", "17", "15"]
      - ["Bundle branch block", "17", "15"]
      - ["Dysrhythmia", "16", "14"]
      - ["Hypertrophy", "7", "7"]
      - ["Valvular heart disease", "6", "6"]
      - ["Myocarditis", "4", "4"]
      - ["Stable angina", "2", "2"]
      - ["Heart failure (NYHA 2/3/4)", "1 each", "1 each"]
      - ["Palpitation", "1", "1"]
      - ["Unstable angina", "1", "1"]

  - type: description
    title: "About those counts"
    body: |
      Recomputed from the `# Reason for admission:` field of all 549 headers.
      Single-label — one diagnosis per record, and no patient has conflicting
      diagnoses across their recordings, so the record and patient columns are
      both meaningful and neither sums to a multi-label total.

      Two disagreements with the README, worth knowing:

      - The README states **54 healthy individuals**; the files give **52**. The
        80 healthy-control records match the 80 lines of the shipped `CONTROLS`
        file exactly, across 52 patient directories.
      - The README's "18 Cardiomyopathy/Heart failure" corresponds to 15
        Cardiomyopathy + 3 Heart failure patients here, which does reconcile.

      **27 records carry no diagnosis** (`n/a` in the header). ECGBench labels
      those `UNKNOWN` and, because seven classes have fewer than 10 records,
      pools those into `OTHER` for stratification. `primary_diagnosis` is that
      pooled view — train on `diagnosis`, which is verbatim.

  - type: table
    title: "Validation summary (1000 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "549", "all records, with is_valid + quality_issues"]
      - ["clean", "546", "99.5% pass rate"]
      - ["excluded", "3", "2 amplitude outliers past ±15 mV, 1 with flat leads"]

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset ptbdb --data-path /path/to/ptbdb/1.0.0/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # Variable-length records: take a fixed window to allow batching, and the
      # standard twelve leads by name out of the fifteen stored. window= is read
      # at load time, so the other 22-110 s are never decoded.
      STANDARD_12 = ["i", "ii", "iii", "avr", "avl", "avf",
                     "v1", "v2", "v3", "v4", "v5", "v6"]

      ds = ECGDataset(
          "ptbdb",
          split="train",
          data_path="/path/to/ptbdb/1.0.0/",
          leads=STANDARD_12,
          window=(0, 10_000),                  # first 10 s at 1000 Hz
          labels=True,
      )

      ds[0]["signal"].shape                 # (12, 10000)
      ds[0]["labels"]["diagnosis"]          # 'Myocardial infarction'
      ds[0]["labels"]["age"]                # 77.0

      # Without leads= you get all 15 signals including vx, vy, vz; without the
      # crop, a DataLoader raises as soon as a batch mixes two record lengths.

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/ptbdb/1.0.0/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/C28C71" }
      - { label: "PTB-XL (larger, fixed-length PTB collection)", url: "https://physionet.org/content/ptb-xl/1.0.3/" }
---
