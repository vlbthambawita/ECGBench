---
slug: "mimic-iv-ecg-ext-icd"
name: "MIMIC-IV-ECG-Ext-ICD"
category: "12-lead-physionet"
order: 6
status: "completed"
source_url: "https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/"
url_label: "physionet.org"
format: "ICD-10-CM diagnoses for MIMIC-IV-ECG · no raw ECGs"
patients: "161,352"
records: "800,035"
access: "credentialed"
license: "PhysioNet Credentialed Health Data License 1.5.0"
origin_institution: "University of Oldenburg + Charité Berlin"
origin_country: "Germany"
leads: 12
paper_title: "Prospects for AI-enhanced ECG as a unified screening tool for cardiac and non-cardiac conditions"
paper_doi: "https://doi.org/10.1093/ehjdh/ztae039"
search_keywords: "mimic iv ecg ext icd icd-10-cm discharge diagnoses oldenburg charite germany derived labels emergency department"

sections:
  - type: description
    title: "Overview"
    body: |
      MIMIC-IV-ECG-Ext-ICD is a **label layer, not a standalone dataset**: it ships
      no waveforms. It is one 323 MB table, `records_w_diag_icd10.csv`, whose
      800,035 rows are exactly the 800,035 studies of
      [MIMIC-IV-ECG]({{ site.baseurl }}/datasets/mimic-iv-ecg.html), keyed by that
      dataset's own `study_id`.

      What it adds is **clinical ground truth**, which MIMIC-IV-ECG itself lacks:
      ICD-10-CM discharge diagnoses linked from the MIMIC-IV emergency-department
      and hospital modules. MIMIC-IV-ECG's own labels are the ECG cart's automated
      report — the machine's opinion of the trace. These are what the patient was
      discharged with, which makes it possible to train an ECG model against
      conditions no ECG algorithm reports on.

      Five diagnosis columns are shipped, differing in where the codes came from:

      - **`ed_diag_ed`** — ED discharge diagnoses, from the MIMIC-IV-ED module.
      - **`ed_diag_hosp`** — hospital discharge diagnoses, reached via `ed_hadm_id`.
      - **`hosp_diag_hosp`** — hospital discharge diagnoses, reached via `hosp_hadm_id`.
      - **`all_diag_hosp`** — the previous two, de-duplicated.
      - **`all_diag_all`** — `all_diag_hosp` where it exists, otherwise `ed_diag_ed`.
        **This is the one to train on**, and the one the published benchmark uses.

      Alongside them come patient demographics, date of death, the linkage ids that
      produced the join, and the release's own 20-fold patient-grouped split.

      Access is **credentialed** under the PhysioNet DUA — the same agreement as
      MIMIC-IV-ECG — so ECGBench never redistributes these codes.

  - type: description
    title: "How ECGBench integrates it — no separate splits"
    body: |
      **There is deliberately no `mimic_iv_ecg_ext_icd` config, and no
      `ecgbench splits --dataset mimic_iv_ecg_ext_icd`.** Every row here is a
      MIMIC-IV-ECG record, so generating a ten-fold partition for it would create a
      *second* ECGBench-blessed split over recordings that `mimic_iv_ecg` already
      partitions. A user who trained on one and evaluated on the other would be
      testing on training data, with both partitions carrying ECGBench's
      imprimatur. Rather than create that trap and then warn about it, we do not
      create it.

      Instead Ext-ICD is a **label provider**: load MIMIC-IV-ECG on ECGBench's
      folds and join these columns onto it.

      ```python
      from ecgbench import ECGDataset
      from ecgbench.labels.mimic_iv_ecg_ext_icd import load_ext_icd

      ds = ECGDataset("mimic_iv_ecg", split="train", fold_numbers=[1],
                      data_path="/path/to/mimic-iv-ecg/1.0/",
                      metadata_source="local")
      # prefix= because MIMIC-IV-ECG's own label frame also carries ecg_time.
      icd = load_ext_icd("/path/to/mimic-iv-ecg-ext-icd-labels/1.0.1/", prefix="icd_")
      joined = icd.reindex(ds.metadata_df["study_id"].values)   # 100% match
      ```

      You need **both** downloads: Ext-ICD has no waveforms, and MIMIC-IV-ECG has
      no ICD codes. MIMIC-IV-ECG's fold CSVs are themselves not published — see
      that dataset's page for the generate-and-verify recipe — so
      `metadata_source="local"` is required here too.

      **The release's own 20 folds are not ECGBench's 10.** `fold` and `strat_fold`
      run 0–19 and are patient-grouped, and the upstream benchmark uses folds 0–17
      for training, 18 for validation and 19 for test. Use
      `upstream_fold_split(icd, "test", prefix="icd_")` to reproduce published
      numbers, or ECGBench's `split=` to stay on ECGBench's folds — never both. The
      two partitions are statistically independent, so the upstream test fold's
      39,569 records land 79.3% inside ECGBench's train split, and 6,449 of its
      8,067 patients appear there.

  - type: table
    title: "What ships, and how much of MIMIC-IV-ECG it covers"
    headers: ["Column", "Records with ≥1 code", "Share", "Distinct codes", "Source"]
    rows:
      - ["`ed_diag_ed`", "184,228", "23.0%", "5,483", "MIMIC-IV-ED discharge diagnoses"]
      - ["`ed_diag_hosp`", "125,277", "15.7%", "12,494", "hospital, via `ed_hadm_id`"]
      - ["`hosp_diag_hosp`", "298,150", "37.3%", "13,641", "hospital, via `hosp_hadm_id`"]
      - ["`all_diag_hosp`", "408,620", "51.1%", "14,964", "the two hospital sets, de-duplicated"]
      - ["**`all_diag_all`**", "**468,005**", "**58.5%**", "**15,197**", "`all_diag_hosp` else `ed_diag_ed` — train on this"]
      - ["*any of the five*", "468,005", "58.5%", "15,437", "332,030 records carry none"]

  - type: table
    title: "The other columns"
    headers: ["Column", "Populated", "Notes"]
    rows:
      - ["`study_id`, `subject_id`", "800,035 / 161,352 unique", "MIMIC-IV-ECG's own keys; join on `study_id`"]
      - ["`file_name`", "800,035", "`record_list.csv`'s `path` under a `mimic-iv-ecg-…-1.0/` prefix"]
      - ["`ecg_time`", "800,035", "identical to MIMIC-IV-ECG's; date-shifted, spans 2097–2211"]
      - ["`ed_stay_id`", "184,720 (23.1%)", "149,021 distinct ED stays"]
      - ["`ed_hadm_id`", "125,314 (15.7%)", "98,061 distinct admissions"]
      - ["`hosp_hadm_id`", "298,258 (37.3%)", "145,176 distinct admissions"]
      - ["`gender`", "795,546", "4,489 hold the **string** `\"missing\"`, not a null"]
      - ["`age` / `anchor_age` / `anchor_year`", "795,546 (99.4%)", "age 12–101, median 66 — but see the counts note"]
      - ["`dod`", "218,648 (27.3%)", "date of death, where MIMIC-IV records one"]
      - ["`ecg_no_within_stay`", "800,035", "0–105, and **−1** for the 331,907 records in no ED or hospital stay"]
      - ["`ecg_taken_in_ed` / `_hosp` / `_ed_or_hosp`", "800,035", "184,720 / 298,258 / 468,128 True — the benchmark's ED, HOSP and ALL subsets"]
      - ["`fold`", "800,035", "the release's own 20 random patient-grouped folds, 0–19"]
      - ["`strat_fold`", "800,035", "20 multi-label stratified folds; fixed in v1.0.1, unused in the paper"]

  - type: description
    title: "About those counts"
    body: |
      Recomputed from the shipped file, verified against the release's own
      `SHA256SUMS.txt` (`records_w_diag_icd10.csv` → `834586ff…`), so what follows
      is the data as published rather than a damaged local copy.

      **The record and patient counts are the full release, not a subset.** All
      800,035 `study_id`s and all 161,352 `subject_id`s of MIMIC-IV-ECG are present,
      with none missing and none extra in either direction. `file_name` equals
      `record_list.csv`'s `path` under a
      `mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/` prefix in 100%
      of rows, and `ecg_time` agrees in 100% of rows. So the join is exact — which
      makes the *coverage* figure below the one that matters.

      **Only 468,005 of the 800,035 records carry any diagnosis — 58.5%.** The other
      332,030 were not part of an ED or hospital stay that MIMIC-IV holds a discharge
      diagnosis for. Their five diagnosis columns hold **empty lists, not nulls**, so
      an empty list is a real value here ("no linked discharge diagnosis") and
      filtering on `notna()` will not find them. Over the labelled records
      `all_diag_all` carries a mean of 13.9 codes and a maximum of 40 — this is
      firmly multi-label data, and the per-column counts above do not sum to
      anything meaningful.

      Note the 123-record gap between the 468,128 records where
      `ecg_taken_in_ed_or_hosp` is True and the 468,005 that actually carry a code:
      being inside a stay does not guarantee a coded discharge diagnosis.

      **The published 1,076-code label set reproduces exactly.** The paper's label
      set comes from `all_diag_all` by truncating each code to five characters,
      stripping trailing ICD-10 placeholder `X`s, propagating every superclass of
      three characters or more, and keeping codes that appear in at least 2,000
      records. Running that over the shipped table gives **1,076** codes — 361 of
      three characters, 466 of four, 249 of five — against the paper's 1,076.

      The X-stripping step is the one worth knowing about: `W19XXXA` (unspecified
      fall) truncates to `W19XX`, and only after stripping does it become the `W19`
      category the benchmark counts. Skip it and the set comes out **1,089** codes,
      which looks close enough to be mistaken for a rounding difference and is not.
      `label_set()` implements the published construction; the raw 15,197 distinct
      codes in `all_diag_all` are what you get before any of it.

      **The label set is not a class breakdown.** Because superclasses are
      propagated, a record coded `I2510` counts toward `I25`, `I251` and `I2510`
      alike — which is why `E78` (183,139 records) outranks `I10` (181,341) after
      propagation while `I10` leads on the raw codes. Every count here is
      record-level: a code appearing twice in one record counts once.

      **Two de-identification artefacts in the demographics.** `gender` marks
      missing values with the literal string `"missing"` (4,489 records, the same
      rows whose `age`, `anchor_age` and `anchor_year` are null); ECGBench's loader
      converts it to `NaN`, which is lossless because the column has no genuine
      nulls. And MIMIC-IV sets `anchor_age` to 91 for every patient older than 89,
      per its own documentation — which shows up here as 26,267 records at exactly
      91, also the file's maximum `anchor_age`. `age` is `anchor_age` plus years
      elapsed and so reaches 101. **Do not read an age above 89 as a real age.**
      Timestamps are shifted into the future for the same reason.

      **Both fold columns are leakage-free on their own terms.** 0 of 161,352
      subjects span more than one `fold`, and 0 span more than one `strat_fold`.
      `fold` is random and gives 39,074–40,835 records per fold; `strat_fold` is
      multi-label stratified and much more even at 39,964–40,498. v1.0.1's release
      note says only that it "fixed issues with stratified folds (not used in the
      original benchmark)", so `fold` is what reproduces published numbers.

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset
      from ecgbench.labels.mimic_iv_ecg_ext_icd import (
          label_set, load_ext_icd, multi_hot, upstream_fold_split,
      )

      STANDARD_12 = ["I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"]

      # Waveforms and folds from MIMIC-IV-ECG; diagnoses from Ext-ICD. Both are
      # credentialed, and MIMIC-IV-ECG's fold CSVs are generated locally, never
      # downloaded — hence metadata_source="local". leads= puts the stored
      # aVF/aVL back into the conventional order, by name.
      ds = ECGDataset("mimic_iv_ecg", split="train", fold_numbers=[1],
                      data_path="/path/to/mimic-iv-ecg/1.0/",
                      metadata_source="local", leads=STANDARD_12)
      len(ds)                                  # 78655  (fold 1 of the train split)

      # prefix= keeps ecg_time from colliding with MIMIC-IV-ECG's own label frame,
      # and every helper below takes the same prefix= to find its columns.
      icd = load_ext_icd("/path/to/mimic-iv-ecg-ext-icd-labels/1.0.1/", prefix="icd_")
      icd.shape                                # (800035, 22)

      joined = icd.reindex(ds.metadata_df["study_id"].values)
      joined.notna().any(axis=1).sum()         # 78655 of 78655 -- a complete join

      # 58.5% of records carry a diagnosis; the rest are empty LISTS, not nulls.
      joined["icd_all_diag_all"].map(bool).sum()            # 46213 of 78655

      # The published 1,076-code label set, and multi-hot targets on this fold.
      codes = label_set(icd, prefix="icd_")
      len(codes), codes[:5]     # 1076, ['E78', 'I10', 'E785', 'I25', 'Z79']
      targets = multi_hot(joined, codes, prefix="icd_")
      targets.shape                            # (78655, 1076)
      (targets.sum(axis=1) > 0).sum()          # 45998 -- 215 fewer than carry codes,
                                               # whose codes all fell below the cutoff

      # ds[0] happens to be a record with no linked discharge diagnosis, which is
      # what 41.5% of the release looks like:
      joined.iloc[0].name                      # 40000162
      joined.iloc[0]["icd_all_diag_all"]       # []
      joined.iloc[0]["icd_gender"]             # 'M'
      ds[0]["signal"].shape                    # (12, 5000)

      # To reproduce the paper instead, use the release's OWN folds -- and then do
      # not touch ECGBench's split=, because the two partitions are independent.
      len(upstream_fold_split(icd, "test", prefix="icd_"))   # 39569

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/ypt5-9d58" }
      - { label: "Paper (Eur Heart J Digital Health, 2024)", url: "https://doi.org/10.1093/ehjdh/ztae039" }
      - { label: "Authors' benchmark code (ECG-MIMIC)", url: "https://github.com/AI4HealthUOL/ECG-MIMIC" }
      - { label: "MIMIC-IV-ECG — the recordings this labels", url: "https://physionet.org/content/mimic-iv-ecg/1.0/" }
---
