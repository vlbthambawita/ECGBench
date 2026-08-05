---
slug: "symile-mimic"
name: "Symile-MIMIC"
category: "12-lead-physionet"
order: 21
status: "completed"
source_url: "https://physionet.org/content/symile-mimic/1.0.0/"
url_label: "physionet.org"
format: "multimodal cohort over MIMIC-IV-ECG · ECG + CXR + labs"
patients: "9,573"
records: "11,622"
access: "credentialed"
license: "PhysioNet Credentialed Health Data License 1.5.0"
origin_institution: "New York University"
origin_country: "USA"
leads: 12
paper_title: "Contrasting with Symile: Simple Model-Agnostic Representation Learning for Unlimited Modalities"
paper_doi: "https://doi.org/10.52202/079017-1814"
search_keywords: "symile mimic multimodal ecg cxr chest x-ray blood labs retrieval nyu neurips derived cohort credentialed"

related:
  - slug: "mimic-iv-ecg"
    relation: "subset_of"
    shares_records: true
    verified: true
    note: >
      Every ECG here is a MIMIC-IV-ECG recording. Verified from the files: all
      11,610 distinct ecg_study_id values are MIMIC-IV-ECG study_ids with none
      extra, all 9,573 subject_ids are MIMIC-IV-ECG subjects, and ecg_path,
      ecg_time and subject_id agree with record_list.csv in 100% of rows. The
      shipped data_npy ECG tensors are those same waveforms min-max normalised,
      not new recordings. So ECGBench publishes no separate fold assignment for
      Symile-MIMIC — you use the mimic_iv_ecg folds and join its columns onto
      them.
      Symile-MIMIC does carry the authors' own train/val/test split, which is
      independent of ECGBench's ten folds: 75.6% of its 464 test studies sit in
      ECGBench's train split, and 349 of that split's 461 patients appear there.
      Pick one partition and stay inside it.
  - slug: "mimic-iv-ecg-demo"
    relation: "same_cohort"
    shares_records: true
    verified: true
    note: >
      Two different subsets of MIMIC-IV-ECG that overlap in 34 recordings across
      26 patients, matched on (subject_id, ecg_time to the minute). A study_id
      comparison reports 0% overlap and is misleading, because the demo renumbers
      study_id into a disjoint range — none of the 34 matched pairs share an id.
      The overlap is small but real: do not train on Symile-MIMIC and evaluate on
      the demo without excluding those 34 recordings.

sections:
  - type: description
    title: "Overview"
    body: |
      Symile-MIMIC is a **multimodal cohort over MIMIC-IV-ECG's recordings, not a
      standalone ECG dataset**. It pairs each of 11,622 hospital admissions with
      three modalities drawn from MIMIC-IV:

      - a **12-lead ECG** taken within 24 h of admission, from
        [MIMIC-IV-ECG]({{ site.baseurl }}/datasets/mimic-iv-ecg.html) — 10 s at
        500 Hz, 5,000 samples per lead;
      - a **chest X-ray** taken in the 24–72 h window post-admission, from
        MIMIC-CXR-JPG, with the 14 CheXpert findings for it;
      - up to **50 blood labs** drawn within 24 h of admission, from MIMIC-IV hosp.

      It was built for the NeurIPS 2024 Symile paper, whose contrastive objective
      needs three or more modalities at once, and its evaluation task is **CXR
      retrieval**: given a query's ECG and labs, pick that query's chest X-ray out
      of 10 candidates. Only admissions with all three modalities present — and a
      PA or AP view, and at least one of the 50 labs — qualified, which is why
      11,622 admissions come out of MIMIC-IV-ECG's 800,035 studies.

      Access is **credentialed** under the PhysioNet Credentialed Health Data Use
      Agreement — the same agreement as MIMIC-IV-ECG — so ECGBench redistributes
      none of it.

  - type: description
    title: "How ECGBench integrates it — no separate splits"
    body: |
      **There is deliberately no `symile_mimic` config, and no
      `ecgbench splits --dataset symile_mimic`.** Every ECG here is a MIMIC-IV-ECG
      record, so generating a ten-fold partition would create a *second*
      ECGBench-blessed split over recordings that `mimic_iv_ecg` already
      partitions. A user who trained on one and evaluated on the other would be
      testing on training data, with both partitions carrying ECGBench's
      imprimatur. Rather than create that trap and then warn about it, we do not
      create it.

      Instead Symile-MIMIC is a **cohort and label provider**: load MIMIC-IV-ECG on
      ECGBench's folds and join these columns onto it.

      ```python
      from ecgbench import ECGDataset
      from ecgbench.labels.symile_mimic import by_study_id, load_cohort

      ds = ECGDataset("mimic_iv_ecg", split="train", fold_numbers=[1],
                      data_path="/path/to/mimic-iv-ecg/1.0/",
                      metadata_source="local")
      # prefix= because MIMIC-IV-ECG's own label frame also carries subject_id
      # and ecg_time.
      cohort = load_cohort("/path/to/symile-mimic/1.0.0/", prefix="sym_")
      joined = by_study_id(cohort, prefix="sym_").reindex(
          ds.metadata_df["study_id"].values)
      ```

      You need **both** downloads: Symile-MIMIC has no ECGBench folds, and
      MIMIC-IV-ECG has none of these columns. MIMIC-IV-ECG's fold CSVs are
      themselves not published — see that dataset's page for the
      generate-and-verify recipe — so `metadata_source="local"` is required here
      too.

      **The join is deliberately partial.** Symile-MIMIC covers 11,610 of
      MIMIC-IV-ECG's 800,035 studies, so a single ECGBench fold of 78,655 records
      matches 1,135 of them (1.44%). That is the cohort being a cohort, not a
      broken join — unlike
      [MIMIC-IV-ECG-Ext-ICD]({{ site.baseurl }}/datasets/mimic-iv-ecg-ext-icd.html),
      which covers all 800,035 records and should match 100%.

      **The release's own train/val/test split is not ECGBench's.** Symile's splits
      are patient-disjoint on their own terms — 0 subjects are shared between any
      pair — but they are statistically independent of ECGBench's folds: 75.6% of
      the 464 test studies land in ECGBench's *train* split, and 349 of the 461
      test subjects appear there. `load_split()` gives you the release's partition
      for reproducing published numbers; `ECGDataset(split=...)` gives you
      ECGBench's. Never mix them.

  - type: table
    title: "What each modality contributes"
    headers: ["Modality", "Source", "Count", "Notes"]
    rows:
      - ["ECG", "MIMIC-IV-ECG", "11,610 distinct studies", "12-lead, 10 s, 500 Hz. 12 studies serve two admissions each"]
      - ["Chest X-ray", "MIMIC-CXR-JPG", "11,609 distinct DICOMs", "9,762 AP and 1,860 PA views, 24–72 h post-admission"]
      - ["CXR findings", "MIMIC-CXR-JPG `chexpert.csv`", "14 labels", "CheXpert four-state encoding — see the counts note"]
      - ["Blood labs", "MIMIC-IV hosp `labevents`", "50 itemids", "mean 35.0 measured per admission, min 1, max 50"]
      - ["Lab percentiles", "derived by the release", "50 columns", "train-set NaN-aware ECDF; **split CSVs only**"]
      - ["Demographics", "MIMIC-IV hosp `patients` / `admissions`", "11,622 rows", "age, sex, race, admission type, date of death"]
      - ["Preprocessed tensors", "derived by the release", "33 GB `data_npy/`", "CXR, ECG and labs as `.npy`; the ECG is unitless"]

  - type: table
    title: "The shipped files, and which splits they describe"
    headers: ["File", "Rows", "Covers", "Notes"]
    rows:
      - ["`symile_mimic_data.csv`", "11,622", "the full cohort", "94 columns — the only file with demographics, `ecg_study_id`, `ecg_time` and all 14 CheXpert labels"]
      - ["`train.csv`", "10,000", "10,000 admissions", "110 columns: 50 raw labs, 50 percentiles, 6 CheXpert labels, both paths"]
      - ["`val.csv`", "750", "750 admissions", "carries constant `label` / `label_hadm_id` columns — see the counts note"]
      - ["`val_retrieval.csv`", "7,500", "the same 750 admissions", "750 queries × 10 retrieval candidates"]
      - ["`test.csv`", "4,640", "464 admissions", "464 queries × 10 retrieval candidates"]
      - ["`labs_means.json`", "50 keys", "train-set mean percentiles", "substituted for an unmeasured lab before the model sees it"]
      - ["`data_npy/<split>/`", "—", "one tensor per modality", "row-aligned to `hadm_id_<split>.npy`, which equals the split CSV order"]
      - ["`symile_mimic_model.ckpt`", "—", "737 MB", "the paper's trained checkpoint; ECGBench does not use it"]

  - type: description
    title: "About those counts"
    body: |
      Recomputed from the shipped files. **All 40 files verify against the
      release's own `SHA256SUMS.txt`**, so what follows is the data as published
      rather than a damaged or filtered local copy.

      **11,622 is admissions; the distinct ECG count is 11,610.** The row unit here
      is the hospital admission (`hadm_id`, unique across all 11,622 rows), not the
      recording. 12 ECG studies each serve **two** admissions — always two
      admissions of the same patient hours or a day apart, where the same ECG fell
      within 24 h of both. The `records` figure above is the release's own headline
      count; if you are counting *recordings*, the number is 11,610. The CXR side is
      ragged the same way, at 11,609 distinct DICOMs. And 1,476 of the 9,573
      patients contribute more than one admission, up to 13.

      **The column literally named `study_id` is the CXR's, not the ECG's.** It is
      byte-identical to `cxr_study_id` in 100% of rows, while MIMIC-IV-ECG's key
      lives in `ecg_study_id`. Joining MIMIC-IV-ECG on the shipped `study_id`
      matches nothing at all, which looks like a broken dataset rather than a wrong
      column name. ECGBench's loader drops the duplicate so the trap is gone. The
      split CSVs drop `ecg_study_id` as well, so it is recovered there from the last
      segment of `ecg_path` — verified equal to `ecg_study_id` in 100% of rows, with
      `ecg_path` itself equal to `record_list.csv`'s `path` in 100%.

      **CheXpert labels have four states, and two of them are not "no".** 1.0
      positive, 0.0 negative, **−1.0 uncertain**, and NaN **"the report does not
      mention this finding"** — where NaN is the majority state for most findings.
      Reading −1.0 as negative, or NaN as negative, are modelling choices with
      different results, so ECGBench's `chexpert_targets()` asks you to pick rather
      than defaulting silently.

      | Finding | 1.0 | 0.0 | −1.0 | NaN |
      |---|---|---|---|---|
      | Support Devices | 4,669 | 291 | 16 | 6,646 |
      | Pleural Effusion | 4,219 | 2,001 | 426 | 4,976 |
      | Cardiomegaly | 3,906 | 1,050 | 520 | 6,146 |
      | Lung Opacity | 3,582 | 206 | 257 | 7,577 |
      | Atelectasis | 3,350 | 132 | 680 | 7,460 |
      | Edema | 2,389 | 2,070 | 968 | 6,195 |
      | No Finding | 1,606 | 0 | 0 | 10,016 |
      | Pneumonia | 1,205 | 1,367 | 1,004 | 8,046 |
      | Consolidation | 861 | 385 | 302 | 10,074 |
      | Enlarged Cardiomediastinum | 586 | 307 | 770 | 9,959 |
      | Pneumothorax | 502 | 3,757 | 75 | 7,288 |
      | Lung Lesion | 328 | 26 | 62 | 11,206 |
      | Fracture | 207 | 20 | 12 | 11,383 |
      | Pleural Other | 101 | 6 | 28 | 11,487 |

      This is multi-label data, so the columns do not sum to 11,622. `No Finding` is
      never 0.0 or −1.0, making it 1,606 positives against 10,016 unmentioned.

      **The split CSVs are not a partition of the full table, and they drop
      columns.** 10,000 + 750 + 464 = **11,214** of the 11,622 admissions appear in
      a split; the other **408** were discarded by the release's
      patient-disjointness filter, which drops any val or test candidate whose
      patient already appears in an earlier split. The split CSVs also carry only
      **6 of the 14** CheXpert labels — `Atelectasis`, `Cardiomegaly`, `Edema`,
      `Lung Opacity`, `No Finding`, `Pleural Effusion` — and no demographics,
      `ecg_study_id`, `ecg_time` or CXR metadata. Those exist only in
      `symile_mimic_data.csv`.

      **`val.csv`'s `label` and `label_hadm_id` columns carry no information.** They
      are constant (`label == 1`, `label_hadm_id == hadm_id` for all 750 rows), a
      side effect of the release's `create_dataset_splits.py` mutating the
      validation frame in place while building `val_retrieval.csv` from it. `val` is
      a plain validation set, and its `data_npy/` directory correctly ships no label
      tensor.

      **Lab coverage is uneven, but there are no sentinel rails.** 16 of the 50 labs
      are present for ≥99% of admissions, 18 for 50–90%, and 16 for under half; the
      three itemids MIMIC-IV names `H`, `L` and `I` (50934, 51678, 50947) bring up
      the rear at 10.5%. Every admission has at least one lab, which is what the
      release's `labs_all_nan` column records — it is 0 for all 11,622 rows. Unlike
      MIMIC-IV-ECG's `machine_measurements.csv`, which encodes "unmeasurable" as
      integer rails, missing labs here are genuine NaNs, so `notna()` is the right
      test. A handful of values are nonetheless physiologically impossible rather
      than missing — 10 of 5,254 Base Excess values fall below −30, the minimum
      being −413 — and are MIMIC-IV data-entry errors passed through unchanged.

      **The `data_npy` ECG tensors are not millivolts, and cannot be converted
      back.** Each record was min-max normalised to [−1, 1] over all 12 leads at
      once — `2 * (x − x.min()) / (x.max() − x.min()) − 1`, verified to float32
      precision against `wfdb.rdrecord` on the corresponding MIMIC-IV-ECG record —
      and the per-record min and max are not shipped, so the transform is not
      invertible. They are also stored channel-last with a leading singleton,
      `(n, 1, 5000, 12)`, rather than ECGBench's `(12, 5000)`. For real millivolts,
      read MIMIC-IV-ECG through `ECGDataset`. Note also that the release's README
      describes these files as `.pt`, because its script saves `torch.save` output;
      PhysioNet ships `.npy`.

      **Lead order is MIMIC-IV-ECG's, so aVF comes before aVL.** `signal[4]` is aVF
      here, not aVL as in PTB-XL or Chapman. Pass `leads=` to `ECGDataset` to select
      by name.

      **Timestamps and ages are de-identification artefacts.** MIMIC-IV shifts dates
      into the future — `ecg_time` spans 2110 to 2208 — and caps `anchor_age` at 91
      for every patient over 89, which shows up as 580 rows at exactly 91, the file
      maximum. `age` is `anchor_age` plus years elapsed and so reaches 100 (median
      69). Do not read an age above 89 as a real age.

      **47 of the 11,610 ECG studies are excluded from ECGBench's `clean/`
      version.** The release's own pipeline already removed ECGs with NaN values or
      an all-zero signal, but ECGBench's validation additionally flags 36
      `amplitude_outlier` records — leads railing near ±20 mV — and 11 with a
      `missing_lead`, with no record failing both. They remain in `original/`,
      flagged.

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset
      from ecgbench.labels.symile_mimic import (
          by_study_id, chexpert_targets, labs_frame, load_cohort,
          load_split, load_split_tensors, retrieval_queries,
      )

      STANDARD_12 = ["I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"]

      # Waveforms and folds from MIMIC-IV-ECG; the cohort from Symile-MIMIC. Both
      # are credentialed, and MIMIC-IV-ECG's fold CSVs are generated locally, never
      # downloaded — hence metadata_source="local". leads= puts the stored
      # aVF/aVL back into the conventional order, by name.
      ds = ECGDataset("mimic_iv_ecg", split="train", fold_numbers=[1],
                      data_path="/path/to/mimic-iv-ecg/1.0/",
                      metadata_source="local", leads=STANDARD_12)
      len(ds)                                  # 78655  (fold 1 of the train split)

      # prefix= keeps subject_id and ecg_time from colliding with MIMIC-IV-ECG's
      # own label frame; every helper below takes the same prefix=.
      cohort = load_cohort("/path/to/symile-mimic/1.0.0/", prefix="sym_")
      cohort.shape                             # (11622, 92) -- indexed by hadm_id
      "sym_study_id" in cohort.columns         # False: that column was the CXR's id

      # 12 ECG studies serve two admissions each, so a study_id index needs a
      # policy. The default keeps the earliest admittime; 'raise' refuses instead.
      keyed = by_study_id(cohort, prefix="sym_")
      keyed.shape                              # (11610, 92) -- indexed by study_id

      joined = keyed.reindex(ds.metadata_df["study_id"].values)
      matched = joined.notna().any(axis=1)
      int(matched.sum())                       # 1135 of 78655 (1.44%) -- a cohort,
                                               # not a label layer over all of MIMIC

      rows = joined[matched]
      labs = labs_frame(rows, "value", names=True, prefix="sym_")
      labs.shape                               # (1135, 50)
      labs.notna().sum(axis=1).mean()          # 35.2 labs measured per admission

      # CheXpert's four states, resolved explicitly. The defaults are
      # uncertain="nan" (ignore the label) and not_mentioned="negative".
      targets = chexpert_targets(rows, prefix="sym_")
      targets.shape                            # (1135, 14)
      targets["Atelectasis"].value_counts(dropna=False).to_dict()
                                               # {0.0: 778, 1.0: 298, nan: 59}

      row = rows.iloc[0]
      rows.index[0]                            # 40001630 -- MIMIC-IV-ECG's study_id
      int(row["sym_hadm_id"])                  # 23875064
      row["sym_gender"], int(row["sym_age"])   # ('M', 54)
      row["sym_cxr_ViewPosition"]              # 'AP'
      ds[0]["signal"].shape                    # (12, 5000), in mV, leads by name

      # The release's OWN partition -- for reproducing published numbers only.
      test = load_split("/path/to/symile-mimic/1.0.0/", "test")
      test.shape                               # (4640, 111) = 464 queries x 10
      len(retrieval_queries(test))             # 464 real admissions

      # The preprocessed tensors, if you want the paper's exact model inputs.
      ecg, hadm = load_split_tensors("/path/to/symile-mimic/1.0.0/", "test", "ecg")
      ecg.shape                                # (4640, 1, 5000, 12) -- channel-last
      float(ecg[0].min()), float(ecg[0].max()) # (-1.0, 1.0) -- unitless, NOT mV

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/symile-mimic/1.0.0/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/3vvj-s428" }
      - { label: "Paper (NeurIPS 2024)", url: "https://papers.nips.cc/paper_files/paper/2024/hash/6828259348d99d5e8994028bfdf15d09-Abstract-Conference.html" }
      - { label: "Paper (arXiv:2411.01053)", url: "https://arxiv.org/abs/2411.01053" }
      - { label: "Symile reference implementation", url: "https://github.com/rajesh-lab/symile" }
      - { label: "MIMIC-IV-ECG — the recordings this uses", url: "https://physionet.org/content/mimic-iv-ecg/1.0/" }
      - { label: "MIMIC-CXR-JPG — the chest X-rays", url: "https://physionet.org/content/mimic-cxr-jpg/2.0.0/" }
---
