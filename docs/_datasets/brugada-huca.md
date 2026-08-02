---
slug: "brugada-huca"
name: "Brugada-HUCA"
category: "12-lead-physionet"
order: 10
status: "completed"
source_url: "https://physionet.org/content/brugada-huca/1.0.0/"
url_label: "physionet.org"
format: "12-lead · 12 s · 100 Hz"
patients: "363"
records: "363"
access: "open"
license: "CC BY-SA 4.0"
origin_institution: "Hospital Universitario Central de Asturias (HUCA)"
origin_country: "Spain"
leads: 12
paper_title: "Costa Cortez & Garcia Iglesias, PhysioNet, 2026"
paper_doi: "https://doi.org/10.13026/0m2w-dy83"
search_keywords: "brugada huca spain asturias hospital sudden death st elevation screening 100 hz"

sections:
  - type: description
    title: "Overview"
    body: |
      363 twelve-lead resting ECGs, **one per individual**, from people
      investigated for Brugada syndrome at Hospital Universitario Central de
      Asturias in Oviedo, Spain. Brugada syndrome is a rare inherited arrhythmia
      disorder marked by coved-type ST-segment elevation in the right precordial
      leads V1–V3, often with a right bundle branch block pattern, and it carries
      a risk of sudden cardiac death.

      This is the **smallest and cleanest** dataset in ECGBench, and the only one
      sampled at **100 Hz alone** — PTB-XL offers 100 Hz as an alternative to
      500 Hz, but here it is the native and only rate. Every record is a uniform
      12 leads × 1200 samples (12.0 s), the lead order is standard, and all 363
      records pass every quality check: no NaN samples, no flat leads, and no
      sample beyond ±3.81 mV.

      **It is a screening cohort, not a case-control study.** Everyone here was
      investigated for suspected Brugada syndrome, so the `brugada = 0` class
      means "investigated and not diagnosed" rather than a general-population
      control. A classifier trained on this data estimates *diagnosis among
      referred individuals*, which is a different quantity from population
      prevalence, and the 19% positive rate reflects referral practice at one
      tertiary hospital.

      Because each subject contributes exactly one record, there is no
      within-patient leakage to guard against, so folds are stratified but not
      grouped.

  - type: table
    title: "Clinical labels"
    headers: ["Label", "Value", "Meaning", "Subjects", "Share"]
    rows:
      - ["`brugada`", "0", "healthy (investigated, not diagnosed)", "287", "79.1%"]
      - ["", "1", "confirmed Brugada syndrome", "69", "19.0%"]
      - ["", "2", "other / atypical", "7", "1.9%"]
      - ["`basal_pattern`", "1", "pathological baseline ECG", "46", "12.7%"]
      - ["`sudden_death`", "1", "subject experienced sudden death", "11", "3.0%"]

  - type: description
    title: "About those counts"
    body: |
      All figures recomputed from the shipped `metadata.csv`, which was verified
      against the release's own `SHA256SUMS.txt` along with `RECORDS`,
      `README.md`, `LICENSE.txt`, the data dictionary and a sample of signal
      files. The record and subject counts match the release description exactly
      at **363 and 363**, so there is nothing to reconcile — unusual in this
      catalogue.

      The three labels are **independent columns, not one taxonomy**, and the
      release documents `basal_pattern` as independent of the diagnosis. They are
      far from orthogonal in practice: 19 of the 46 subjects with a pathological
      baseline are also confirmed Brugada cases, and 4 of the 7 atypical cases
      have one. Do not treat a pathological baseline as a proxy for the
      diagnosis.

      **The stratification label is `brugada`, used verbatim.** There is no
      derivation and therefore nothing that can drift from what `labels=True`
      returns — unlike the reduced labels ECGBench derives for PTB-XL,
      Challenge 2021 or MIMIC-IV-ECG.

      **The rare class is deliberately not pooled.** `brugada = 2` has 7 records,
      fewer than the 10 folds, so it cannot appear in every fold and scikit-learn
      emits a warning to that effect. Pooling it would be clinically wrong:
      "other/atypical" is neither healthy nor confirmed. In the released folds
      those 7 records are spread one per fold across 7 of the 10 folds, and the
      class shares per default split are 79.0/19.2/1.7% in train against
      80.6/16.7/2.8% in test.

      Three quirks belong to the **shipped release**, not to any one download —
      each was confirmed against `SHA256SUMS.txt`:

      - **`RECORDS` lists 364 lines for 363 records**: `files/596382/596382`
        appears twice. ECGBench enumerates subjects from `metadata.csv` instead,
        which is authoritative and carries the labels.
      - **A macOS `files/.DS_Store` ships inside the release** and is itself
        checksummed, so any code globbing `files/*` must filter to directories.
      - **`metadata_dictionary.csv` documents a `diagnosis` variable that
        `metadata.csv` does not contain.** The four real columns are
        `patient_id`, `basal_pattern`, `sudden_death` and `brugada`.

      No age, sex or ancestry is published, so no demographic balance check is
      possible.

  - type: table
    title: "Validation summary (100 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "363", "all records, with is_valid + quality_issues"]
      - ["clean", "363", "100% pass rate — no record excluded"]
      - ["excluded", "0", "no NaN samples, no flat leads, peak |amplitude| 3.81 mV"]

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset brugada_huca --data-path /path/to/brugada-huca/1.0.0/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # Fold CSVs come from the HuggingFace Hub by default; only the waveforms
      # need to be local.
      ds = ECGDataset(
          "brugada_huca",
          split="train",
          data_path="/path/to/brugada-huca/1.0.0/",
          labels=True,
      )

      len(ds)                             # 291
      ds[0]["signal"].shape               # (12, 1200)  -- 12 s at 100 Hz
      ds[0]["record_id"]                  # 188981
      ds[0]["labels"]["brugada"]          # 1  -> confirmed Brugada syndrome
      ds[0]["labels"]["basal_pattern"]    # 1  -> pathological baseline ECG
      ds[0]["labels"]["sudden_death"]     # 0

      # The codes have no string form in the CSV; the meanings are in the
      # release README and are re-exported for convenience:
      from ecgbench.splitting.strategies.brugada_huca import BRUGADA_CLASSES
      BRUGADA_CLASSES   # {0: 'healthy', 1: 'confirmed Brugada syndrome', 2: 'other/atypical'}

      # A binary target folds the atypical class in with the undiagnosed --
      # a modelling choice, not something the dataset states:
      (ds.labels_df["brugada"] == 1).mean()    # 0.192 over the train split

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/brugada-huca/1.0.0/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/0m2w-dy83" }
---
