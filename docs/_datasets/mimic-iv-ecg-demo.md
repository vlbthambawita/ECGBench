---
slug: "mimic-iv-ecg-demo"
name: "MIMIC-IV-ECG Demo"
category: "12-lead-physionet"
order: 5
status: "completed"
source_url: "https://physionet.org/content/mimic-iv-ecg-demo/0.1/"
url_label: "physionet.org"
format: "12-lead · 10 s · 500 Hz · WFDB"
patients: "92"
records: "659"
access: "open"
license: "ODbL 1.0"
origin_institution: "Beth Israel Deaconess Medical Center"
origin_country: "USA — Boston, MA"
leads: 12
paper_title: "Dataset DOI"
paper_doi: "https://doi.org/10.13026/4eqn-kt76"
search_keywords: "mimic iv ecg demo usa boston beth israel wfdb subject_id study_id"

sections:
  - type: description
    title: "Overview"
    body: |
      The open demo subset of MIMIC-IV-ECG: 659 twelve-lead diagnostic ECGs from
      92 patients at Beth Israel Deaconess Medical Center, in WFDB format at
      500 Hz for 10 seconds. Records link to the rest of MIMIC-IV through
      `subject_id` and `study_id`, and each carries an acquisition timestamp.

      Two things shape how ECGBench splits it. First, the demo ships
      **no labels** — `record_list.csv` holds identifiers and timestamps only,
      and `machine_measurements.csv` (the report text and measurements) belongs
      to the full release, not this one. There is nothing to stratify on, so the
      split is purely patient-grouped. Second, records per patient are very
      uneven — 1 to 52, median 5 — so grouping by `subject_id` is essential, and
      fold sizes vary as a consequence (60 to 87 records across 10 folds).

      For the ~800,000-record credentialed release, see
      [MIMIC-IV-ECG]({{ '/datasets/mimic-iv-ecg.html' | relative_url }}).

  - type: table
    title: "Watch out for"
    headers: ["Property", "Value"]
    rows:
      - ["Lead order in every header", "I, II, III, aVR, aVF, aVL, V1-V6 — aVF and aVL are transposed relative to the usual convention"]
      - ["Labels", "none in this subset"]
      - ["Age / sex", "not included"]
      - ["Records per patient", "1 to 52 (median 5)"]
      - ["Timestamps", "date-shifted into the future by de-identification"]

  - type: table
    title: "Validation summary (500 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "659", "all records, with is_valid + quality_issues"]
      - ["clean", "645", "97.9% pass rate"]
      - ["excluded", "14", "12 with NaN samples, 13 with one or more flat leads (overlapping)"]

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits \
        --dataset mimic_iv_ecg_demo \
        --data-path /path/to/mimic-iv-ecg-demo/0.1/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset, ecg_collate_fn
      from torch.utils.data import DataLoader

      dataset = ECGDataset(
          "mimic_iv_ecg_demo",
          split="train",
          version="clean",
          data_path="/path/to/mimic-iv-ecg-demo/0.1/",
      )

      loader = DataLoader(dataset, batch_size=16, collate_fn=ecg_collate_fn)

      for batch in loader:
          signals = batch["signal"]        # (B, 12, 5000) at 500 Hz
          record_ids = batch["record_id"]  # study_id values
          break

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/mimic-iv-ecg-demo/0.1/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/4eqn-kt76" }
      - { label: "Full MIMIC-IV-ECG release (credentialed)", url: "https://physionet.org/content/mimic-iv-ecg/1.0/" }
---
