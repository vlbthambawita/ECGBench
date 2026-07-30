---
slug: "ptb-xl"
name: "PTB-XL"
category: "12-lead-physionet"
order: 1
status: "completed"
source_url: "https://physionet.org/content/ptb-xl/1.0.3/"
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
      `ECGDataset` class. Note that the published fold CSVs carry identifiers,
      signal paths and fold assignments only — no labels. Join them back to
      `ptbxl_database.csv` on `ecg_id` for ground truth, as shown below.

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

      # Load the training split (folds 1-8) at 100 Hz.
      # Fold CSVs come from the Hub; signals are read from your local copy.
      dataset = ECGDataset(
          "ptbxl",
          split="train",
          version="clean",
          data_path="/path/to/ptb-xl/1.0.3/",
          sampling_rate=100,
      )

      loader = DataLoader(dataset, batch_size=32, collate_fn=ecg_collate_fn)

      for batch in loader:
          signals = batch["signal"]     # (B, 12, 1000) float32, at 100 Hz
          ecg_ids = batch["record_id"]  # tensor of ecg_id values
          folds   = batch["fold"]       # tensor of fold numbers (1-10)
          break

  - type: code
    title: "Getting the labels"
    language: python
    body: |
      # ECGBench's fold CSVs are identification-only — record ID, patient ID,
      # signal paths, fold and split. Labels stay in the source metadata, so
      # join them on ecg_id when you need ground truth.
      import ast
      import pandas as pd

      PTBXL = "/path/to/ptb-xl/1.0.3/"
      src = pd.read_csv(PTBXL + "ptbxl_database.csv")

      labelled = dataset.metadata_df.merge(
          src[["ecg_id", "scp_codes", "age", "sex", "report"]],
          on="ecg_id", how="left", validate="one_to_one",
      )

      # scp_codes is a dict-string of SCP code -> confidence. Map it to the five
      # diagnostic superclasses via the statement table shipped with PTB-XL.
      stmt = pd.read_csv(PTBXL + "scp_statements.csv", index_col=0)
      code2class = stmt.loc[stmt.diagnostic == 1, "diagnostic_class"].to_dict()

      labelled["superclasses"] = labelled.scp_codes.map(
          lambda s: sorted({code2class[c] for c in ast.literal_eval(s) if c in code2class})
      )
      # -> ecg_id 1: ['NORM']; ecg_id 39: ['MI', 'STTC'] (fold 9, so it is in val)
      # 5,144 of the 21,799 records carry more than one superclass, so treat this
      # as multi-label. The single superclass used for stratification is a lossy view.

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
