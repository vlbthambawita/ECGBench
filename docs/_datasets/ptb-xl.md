---
slug: "ptb-xl"
name: "PTB-XL"
category: "12-lead-physionet"
order: 1
status: "completed"
url: "https://physionet.org/content/ptb-xl/1.0.3/"
url_label: "physionet.org"
format: "12-lead · 10 s · 500 Hz (also 100 Hz)"
patients: "18,869"
records: "21,799"
access: "open"
license: "CC BY 4.0"
origin_institution: "Physikalisch-Technische Bundesanstalt"
origin_country: "Germany"
leads: 12
paper_title: "PTB-XL: A Large Publicly Available ECG Dataset"
paper_doi: "https://doi.org/10.1038/s41597-020-0495-6"
search_keywords: "ptb-xl germany ptb physikalisch-technische bundesanstalt"

sections:
  - type: description
    title: "Overview"
    body: |
      PTB-XL is a large, publicly available 12-lead ECG dataset with 21,799
      clinical records from 18,869 patients, collected at the
      Physikalisch-Technische Bundesanstalt (PTB) between October 1989 and
      June 1996. Each record is 10 seconds long and provided at both 500 Hz
      and 100 Hz. Records are annotated with up to 71 SCP-ECG statements
      grouped into 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP).

      ECGBench bundles a deterministic 10-fold stratified patient-level split
      derived from the SCP superclass labels, ready to consume via the
      `ECGDataset` class.

  - type: table
    title: "Diagnostic superclass breakdown"
    headers: ["Superclass", "Description", "Records"]
    rows:
      - ["NORM", "Normal ECG", "9,514"]
      - ["MI",   "Myocardial Infarction", "5,486"]
      - ["STTC", "ST/T changes", "5,250"]
      - ["CD",   "Conduction disturbance", "4,907"]
      - ["HYP",  "Hypertrophy", "2,655"]

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset, ecg_collate_fn
      from torch.utils.data import DataLoader

      # Load the training split (folds 1-8) at 100 Hz
      dataset = ECGDataset(
          physionet_path="/path/to/ptb-xl/1.0.3/",
          dataset_name="ptbxl",
          split="train",
          frequency="100",
      )

      loader = DataLoader(dataset, batch_size=32, collate_fn=ecg_collate_fn)

      for batch in loader:
          signals = batch["signal"]     # (B, 12, 1000) at 100 Hz
          ecg_ids = batch["ecg_id"]     # list of record IDs
          labels  = batch["scp_codes"]  # list of per-record SCP dicts
          break

  - type: code
    title: "Inspecting the catalogue entry"
    language: python
    body: |
      from ecgbench import get_dataset

      entry = get_dataset("ptb-xl")
      print(entry.patients, entry.records, entry.access)
      # -> 18,869 21,799 open

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/ptb-xl/1.0.3/" }
      - { label: "Nature Scientific Data paper", url: "https://doi.org/10.1038/s41597-020-0495-6" }
      - { label: "PTB-XL+ (derived feature dataset)", url: "https://physionet.org/content/ptb-xl-plus/1.0.1/" }
---
