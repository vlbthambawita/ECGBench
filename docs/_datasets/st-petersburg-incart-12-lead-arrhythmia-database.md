---
slug: "st-petersburg-incart-12-lead-arrhythmia-database"
name: "St Petersburg INCART 12-Lead Arrhythmia Database"
category: "12-lead-physionet"
order: 8
status: "completed"
source_url: "https://physionet.org/content/incartdb/1.0.0/"
url_label: "physionet.org"
format: "12-lead · 30 min · 257 Hz"
patients: "32"
records: "75"
access: "open"
license: "ODC-By 1.0"
origin_institution: "St. Petersburg Institute of Cardiological Technics (INCART)"
origin_country: "Russia"
leads: 12
paper_title: "Dataset DOI"
paper_doi: "https://doi.org/10.13026/C2V88N"
search_keywords: "incart st petersburg russia arrhythmia holter beat annotations pvc ventricular ectopy patient grouping"

related:
  - slug: "physionet-cinc-challenge-2020"
    relation: "subset_of"
    shares_records: true
    verified: false
    note: >
      INCART is one of the six source cohorts of the 2020 challenge training set, as it
      is of the 2021 one. The 2021 overlap is verified against the files (see that
      relationship, declared there); this one follows from the challenge descriptions
      plus the arithmetic that the 2021 release minus its Chapman and Ningbo cohorts
      leaves exactly the 43,101 records of the 2020 training set. Not checked against a
      2020 download, hence unverified. Do not evaluate on INCART after training on
      either challenge year.

sections:
  - type: description
    title: "Overview"
    body: |
      75 half-hour recordings extracted from **32 Holter records**, collected at the
      St Petersburg Institute of Cardiological Technics from patients being tested
      for coronary artery disease. Each record is 12 standard leads, 257 Hz, exactly
      1800 s. The dataset's primary product is not a record-level diagnosis but
      **175,907 reference beat annotations** across ten beat types, placed by an
      algorithm and then corrected manually.

      **Patient grouping is the thing that matters here, and ECGBench enforces it.**
      Only 2 of the 32 patients contributed a single record; 18 contributed two, 11
      contributed three and one contributed four. The concrete danger is easy to
      show: **3,166 of the 3,174 right-bundle-branch-block beats come from one
      patient** (`patient08`, records I16 and I17, where essentially every beat is
      RBBB). Split those two records across train and test and an RBBB classifier
      scores well by memorising one person. Folds here are grouped by patient, and
      no patient spans a fold.

      **Records are long — 1800 s, about 44 MB each in memory.** They cannot be
      batched as they are; crop with a `transform`, or use `batch_size=1`.

      **Gain varies between records** (240 to 1063 adu/mV, uniform across the 12
      leads within a record). wfdb divides by the per-record gain, so signals reach
      ECGBench in millivolts and nothing downstream needs to care.

      Lead names are **uppercase** `AVR`/`AVL`/`AVF`, the PTB-XL spelling. Lead
      matching is case-insensitive, so `leads=["aVL"]` still works.

  - type: table
    title: "Reference beat annotations"
    headers: ["Beat type", "Symbol", "Count", "Share", "Records containing it"]
    rows:
      - ["normal", "N", "150,410", "85.51%", "73"]
      - ["premature ventricular contraction", "V", "20,013", "11.38%", "70"]
      - ["right bundle branch block beat", "R", "3,174", "1.80%", "3"]
      - ["atrial premature contraction", "A", "1,944", "1.11%", "33"]
      - ["fusion of ventricular and normal", "F", "219", "0.12%", "22"]
      - ["nodal (junctional) escape", "j", "92", "0.05%", "2"]
      - ["supraventricular escape", "n", "32", "0.02%", "2"]
      - ["premature or ectopic supraventricular", "S", "16", "0.01%", "6"]
      - ["unclassifiable", "Q", "6", "0.00%", "5"]
      - ["left or right bundle branch block", "B", "1", "0.00%", "1"]
      - ["**total beats**", "", "**175,907**", "", "75"]

  - type: table
    title: "Patient-level diagnoses"
    headers: ["Diagnosis (from the header <diagnoses> field)", "Patients", "Records"]
    rows:
      - ["(none recorded)", "14", "34"]
      - ["Earlier MI", "4", "9"]
      - ["Coronary artery disease, arterial hypertension, left ventricular hypertrophy", "4", "11"]
      - ["Transient ischemic attack", "4", "7"]
      - ["Coronary artery disease, arterial hypertension", "3", "6"]
      - ["Acute MI", "2", "6"]
      - ["Sinus node dysfunction", "1", "2"]
      - ["**total**", "**32**", "**75**"]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the 75 headers and 75 `.atr`
      files. Two disagreements with the shipped README, both worth knowing:

      - **Demographics.** The README states "17 men and 15 women, aged 18-80; mean
        age: 58". The headers give **18 men and 14 women**, ages 18–80, mean
        **56.2** (computed per patient, not per record — records are unevenly
        distributed across patients, so a per-record mean is a different number).
      - **Gain range.** The README says gains vary "from 250 to 1100"; the headers
        actually run **240 to 1063**.

      The beat total does reconcile: the README's "over 175,000 beat annotations"
      against a recomputed **175,907**. The `.atr` files hold 175,919 annotations
      in total — the extra 12 are `+` rhythm-change markers, which are not beats
      and are exposed separately as `n_rhythm_changes`.

      **The README's diagnosis table cannot be reproduced from the `<diagnoses>`
      field alone, and its categories sum past 32 patients.** It mixes two sources.
      WPW (2 patients), atrial fibrillation, AV nodal block and bundle branch block
      appear **only** in the per-record free-text findings, never in `<diagnoses>`,
      which is why the table above looks so different. Use `record_features`
      alongside `diagnosis`; neither alone is the full label.

      **14 of the 32 patients (34 of 75 records) carry no diagnosis at all** — the
      `<diagnoses>` token is absent from those headers entirely, rather than present
      and empty. ECGBench labels them `UNKNOWN`.

      **The stratification label is not a clinical grouping.** With 32 patients over
      10 folds, and pooling any class holding fewer than 10 *patients* (patients, not
      records, because folds are grouped by patient), `stratify_class` collapses to
      two values: `UNKNOWN` (34 records) and `OTHER` (41). It exists so the folds are
      defined. Train on `diagnosis`, `record_features` and the beat counts.

      Two shipped text files, `files-patients-diagnoses.txt` and
      `record-descriptions.txt`, restate the header content. Both were checked
      against all 75 headers and agree on **every** record — no patient-id,
      diagnosis or findings disagreement — so ECGBench reads the headers only.

  - type: table
    title: "Validation summary (257 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "75", "all records, with is_valid + quality_issues"]
      - ["clean", "70", "93.3% pass rate"]
      - ["excluded", "5", "4 with a lead pinned at the 16-bit rail, plus one 43 mV excursion"]

  - type: description
    title: "About the excluded records, and the amplitude range"
    body: |
      `amplitude_range_mv` is `[-30, 30]` here rather than ECGBench's usual
      `[-10, 10]`, and that is a measured decision, not a convenience. In 30-minute
      ambulatory Holter data, amplitudes past 10 mV are ordinary: **29 of the 75
      records** contain at least one sample beyond ±10 mV, and one (I31) runs
      12–20 mV across all twelve leads for the full half hour with entirely normal
      per-lead medians. At ±10 the check would discard 39% of a 75-record dataset.

      At ±30 it does the job it is there for — catching **four records with a lead
      pinned at the 16-bit rail** for most or all of the recording, which
      `missing_leads` cannot see because those leads are railed rather than zero:

      | Record | Dead lead | Samples at the rail | Peak |
      |---|---|---|---|
      | I03 | V3 | 100% | ±31.48 mV |
      | I58 | V4 | 96.6% | ±31.48 mV |
      | I02 | V6 | 98.1% | ±107.08 mV |
      | I57 | V4 | 34.6% | ±31.48 mV |

      I69 is excluded too, on a ±43.24 mV excursion with no railed lead. No single
      threshold separates it from the four above, because its peak sits *higher*
      than their ±31.5 mV rail — the rail depends on each record's own gain.

      There are **no NaN samples and no all-zero leads anywhere** in this release.

  - type: description
    title: "This is the better copy of these records"
    body: |
      74 of these 75 records also sit inside the PhysioNet/CinC Challenge 2021
      release (renamed `I0001`–`I0075`, with `I36` absent), and inside the 2020 one.
      **Prefer this original.** The challenge re-quantised the signals to a gain of
      1000 adu/mV, which cannot represent this dataset's amplitude range — records
      here reach ±107 mV — so the challenge copy carries roughly 467,000 NaN samples
      across those 74 records, where the original has none. The challenge copies also
      drop the `.atr` beat annotations and the patient numbers, which is what makes
      grouped folds possible here.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset incartdb --data-path /path/to/incartdb/1.0.0/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # 1800 s records (~44 MB each): crop to a fixed window to allow batching.
      ds = ECGDataset(
          "incartdb",
          split="train",
          data_path="/path/to/incartdb/1.0.0/",
          metadata_source="local",
          transform=lambda x: x[:, :2570],   # first 10 s at 257 Hz
          labels=True,
      )

      len(ds)                                  # 55
      ds[0]["signal"].shape                    # (12, 2570)
      ds[0]["record_id"]                       # 'I01'
      ds[0]["labels"]["patient_id"]            # 'patient01' — folds are grouped by this
      ds[0]["labels"]["diagnosis"]             # 'Coronary artery disease, arterial hypertension'
      ds[0]["labels"]["record_features"]       # 'PVCs, noise'
      ds[0]["labels"]["n_beats"]               # 2757
      ds[0]["labels"]["beat_V"]                # 344 premature ventricular contractions
      ds[0]["labels"]["pvc_fraction"]          # 0.1248

      # Uppercase AVR/AVL/AVF in the files, but matching is case-insensitive:
      ds.config.lead_names   # ['I','II','III','AVR','AVL','AVF','V1',...,'V6']

      # PVC burden per record, straight off the reference annotations:
      ds.labels_df["pvc_fraction"].describe()   # max 0.507 (I43, ventricular bigeminy)

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/incartdb/1.0.0/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/C2V88N" }
      - { label: "PhysioBank beat annotation definitions", url: "https://archive.physionet.org/physiobank/annotations.shtml" }
---
