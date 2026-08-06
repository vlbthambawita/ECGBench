---
slug: "physionet-cinc-challenge-2021"
name: "PhysioNet/CinC Challenge 2021"
category: "12-lead-physionet"
order: 23
status: "completed"
source_url: "https://physionet.org/content/challenge-2021/1.0.3/"
url_label: "physionet.org"
format: "12-lead · 5–1,800 s · 257/500/1,000 Hz"
patients: "—"
records: "88,253"
access: "open"
license: "CC BY 4.0"
origin_institution: "CPSC, INCART, PTB, PTB-XL, Georgia, Chapman-Shaoxing, Ningbo"
origin_country: "Multi-national (China, Russia, Germany, USA)"
leads: 12
paper_title: "Reyna et al., Computing in Cardiology, 2021"
paper_doi: "https://doi.org/10.23919/CinC53138.2021.9662687"
search_keywords: "cinc challenge 2021 physionet china russia germany usa georgia chapman ningbo cpsc incart snomed multi-label meta-dataset"
patients_class: "count-na"

related:
  - slug: "physionet-cinc-challenge-2020"
    relation: "contains"
    shares_records: true
    verified: true
    note: >
      The 2021 challenge reuses the **whole** 2020 training set and adds the
      Chapman-Shaoxing and Ningbo cohorts on top (88,253 − 10,247 − 34,905 = 43,101).
      Verified from the files, not from the arithmetic: all 43,101 of the 2020 record
      names appear here and none is unique to 2020, and all 43,101 `.mat` waveform
      files are bit-identical, compared via the two releases' own published
      `SHA256SUMS.txt`. Only 705 of the 43,101 headers differ, all in the comment
      block — 631 `#Dx` lists that 2021 deduplicated and numerically sorted, and 74
      `st_petersburg_incart` `#Sex` fields that 2021 spelled out. Never use one year's
      training set to evaluate a model trained on the other; they are the same
      recordings.

  - slug: "ptb-xl"
    relation: "contains"
    shares_records: true
    verified: true
    note: >
      All 21,837 records of the `ptb-xl` cohort are PTB-XL's 500 Hz recordings,
      renamed `HR00001`-`HR21837` where the number is PTB-XL's own `ecg_id`.
      Verified from the files: HR00001, HR00500, HR10000 and HR21799 are
      bit-identical to `records500/…/{00001,00500,10000,21799}_hr` (max abs
      difference 0.0000 mV over all 12 leads and 5,000 samples). Note the count —
      21,837 is the **v1.0.1** total, so this cohort still holds the 38
      duplicate/triplicate records that PTB-XL v1.0.3 removed; exactly 38 of the
      HR numbers have no v1.0.3 `ecg_id`. Evaluating a challenge-trained model on
      PTB-XL is testing on training data.

  - slug: "chapman-shaoxing-arrhythmia"
    relation: "contains"
    shares_records: true
    verified: true
    note: >
      The `chapman_shaoxing` (10,247) and `ningbo` (34,905) cohorts together are the
      entire 45,152-record PhysioNet ecg-arrhythmia release, under the same `JS#####`
      names. Verified from the files: 45,151 of 45,152 names match exactly and 25 of 25
      randomly sampled shared records are bit-identical. The single exception is a
      typo, not a different recording — this release names one record `S23074` where
      ecg-arrhythmia has `JS23074`, so a naive name join silently loses it and reports
      45,151 instead of 45,152. Overlap is complete: do not evaluate on ecg-arrhythmia
      (or its Chapman/Ningbo subsets) after training here.

  - slug: "ptb-diagnostic-ecg-database"
    relation: "contains"
    shares_records: true
    verified: true
    note: >
      The 516 records of the `ptb` cohort are PTBDB recordings, renamed
      `S0001`-`S0549` (516 of PTBDB's 549). Verified from the files: 20 of 20 sampled
      records match a PTBDB record at correlation ≥ 0.99999 and identical length —
      e.g. `S0138` ≡ `patient044/s0146lre`, `S0541` ≡ `patient288/s0549_re`. The
      numbering is **not** a crosswalk (S0138 is not s0138), so the mapping is only
      recoverable by signal matching. That matters beyond leakage: PTBDB's patient
      directories are the only way to know which of these records share a patient,
      and that grouping is lost here.

  - slug: "st-petersburg-incart-12-lead-arrhythmia-database"
    relation: "contains"
    shares_records: true
    verified: true
    note: >
      The 74 records of the `st_petersburg_incart` cohort are 74 of INCART's 75,
      renamed `I0001`-`I0075` where the number is INCART's own (`I0004` ≡ `I04`);
      `I36` is the one absent. Verified from the files: 20 of 20 sampled records match
      at correlation ≥ 0.99999 over all 462,600 samples. INCART's ~32 patients
      contributed several recordings each and that grouping is not published here.

  - slug: "cpsc-2018-china-physiological-signal-challenge-2018"
    relation: "contains"
    shares_records: true
    verified: true
    note: >
      The `cpsc_2018` cohort is the entire CPSC-2018 public training set — 6,877
      records under CPSC's own `A0001`-`A6877` names, **not** renamed as the PTB-XL,
      PTB and INCART cohorts were. Verified from the files: all 6,877 names match and
      all 6,877 `.mat` waveforms are **byte-identical** to the CPSC-2018 release
      ECGBench catalogues, with 0 records unique to either side. `cpsc_2018_extra`
      adds 3,453 `Q####` records that CPSC-2018 did not release publicly and that
      ECGBench does not catalogue separately. Do not evaluate on CPSC-2018 after
      training here; it is the same recordings.

sections:
  - type: description
    title: "Overview"
    body: |
      The public training set of the 2021 PhysioNet/Computing in Cardiology
      Challenge, *Will Two Do? Varying Dimensions in Electrocardiography*:
      **88,253 twelve-lead records** pooled from **eight source cohorts** across
      China, Russia, Germany and the USA, redistributed in one uniform WFDB tree.

      **This is a meta-dataset, and that is the single most important thing to
      know about it.** Five of the eight cohorts are datasets ECGBench catalogues
      separately, and the records were renamed, so the overlap is invisible from
      the identifiers. Four of those overlaps are verified against the files in
      the relationships above — including PTB-XL, which is present in full and
      bit-identical. Training here and evaluating on PTB-XL, PTBDB, INCART,
      CPSC-2018, Chapman-Shaoxing, Ningbo or PhysioNet ecg-arrhythmia is testing
      on training data.

      **Only the training set was ever published.** The challenge's hidden
      validation and test sets are not in this release, which is why the record
      count is 88,253 and not the ~131,000 the challenge described. There is
      consequently no reproducible official split, and ECGBench generates its own
      10 folds.

      **Sampling rate and record length are per-record properties here**, unlike
      every other dataset in this catalogue. Rate is 500 Hz for 87,663 records,
      1000 Hz for the 516 `ptb` ones and 257 Hz for the 74 `st_petersburg_incart`
      ones; length runs from 5 s to 1800 s, with 1,650 distinct lengths in the
      `cpsc_2018` cohort alone. Both are exposed as label columns. Records cannot
      be batched as they are — take a fixed `window=(start, length)`, or use
      `batch_size=1`.

      Labels are **multi-label SNOMED-CT codes** from the `#Dx` header field: 133
      distinct codes, 2.06 per record on average and up to 12, with no unlabelled
      record. No code table ships with the data, so ECGBench packages the
      challenge's own.

  - type: table
    title: "Source cohorts"
    headers: ["Cohort", "Records", "Rate", "Length", "Also catalogued as"]
    rows:
      - ["ningbo", "34,905", "500 Hz", "10 s", "part of PhysioNet ecg-arrhythmia"]
      - ["ptb-xl", "21,837", "500 Hz", "10 s", "PTB-XL (v1.0.1, in full)"]
      - ["georgia", "10,344", "500 Hz", "5–10 s", "— (no standalone release)"]
      - ["chapman_shaoxing", "10,247", "500 Hz", "10 s", "part of PhysioNet ecg-arrhythmia"]
      - ["cpsc_2018", "6,877", "500 Hz", "6–144 s", "CPSC-2018 training set"]
      - ["cpsc_2018_extra", "3,453", "500 Hz", "8–98 s", "— (unreleased CPSC extra set)"]
      - ["ptb", "516", "1,000 Hz", "32–120 s", "PTB Diagnostic ECG Database (516 of 549)"]
      - ["st_petersburg_incart", "74", "257 Hz", "1,800 s", "St Petersburg INCART (74 of 75)"]
      - ["**total**", "**88,253**", "", "", ""]

  - type: table
    title: "Most frequent diagnoses"
    headers: ["Diagnosis", "Abbr.", "SNOMED-CT", "Records", "Scored"]
    rows:
      - ["sinus rhythm", "NSR", "426783006", "28,971", "yes"]
      - ["sinus bradycardia", "SB", "426177001", "18,918", "yes"]
      - ["t wave abnormal", "TAb", "164934002", "11,716", "yes"]
      - ["sinus tachycardia", "STach", "427084000", "9,657", "yes"]
      - ["atrial flutter", "AFL", "164890007", "8,374", "yes"]
      - ["left axis deviation", "LAD", "39732003", "7,631", "yes"]
      - ["myocardial infarction", "MI", "164865005", "6,144", "no"]
      - ["left ventricular high voltage", "LVHV", "55827005", "5,401", "no"]
      - ["atrial fibrillation", "AF", "164889003", "5,255", "yes"]
      - ["s t changes", "STC", "55930002", "5,009", "no"]
      - ["nonspecific st t abnormality", "NSSTTA", "428750005", "4,712", "no"]
      - ["left ventricular hypertrophy", "LVH", "164873001", "4,406", "no"]
      - ["t wave inversion", "TInv", "59931005", "3,989", "yes"]
      - ["sinus arrhythmia", "SA", "427393009", "3,790", "yes"]
      - ["st depression", "STD", "429622005", "3,645", "no"]
      - ["(128 further codes)", "", "", "", ""]

  - type: description
    title: "About those counts"
    body: |
      **Records: 88,253, not ~131,000.** The challenge description quoted the full corpus
      including the hidden validation and test sets; PhysioNet only ever published
      the training set. Recomputed as the number of `.hea` files under `training/`.
      An earlier version of this catalogue page said "~130,862" — that figure
      belongs to the challenge, not to this download.

      **Patients: not published.** No patient identifier ships with any cohort and
      records were renamed, so patient counts are unrecoverable from these files.
      This is exact for six cohorts (one record per patient), but **not** for
      `ptb` and `st_petersburg_incart`, whose source datasets have several
      recordings per patient — 113 of PTBDB's 290 patients and roughly 32 patients
      across INCART's 75 records. Those 590 records (0.67%) can therefore place one
      patient in more than one fold. Every other dataset in ECGBench with repeated
      patients groups folds by patient; this one cannot.

      **The diagnosis table is multi-label and does not sum to the record total.**
      It counts records carrying each code, over 181,817 code instances across
      88,253 records. Derived from the `#Dx` field of all 88,253 headers, joined to
      the challenge's code table. The `Scored` column marks the 30 classes the
      challenge metric evaluated (as 26 classes, four pairs being scored as
      equivalent); 81,966 records carry at least one scored code and **6,287 carry
      none**. Every one of the 133 codes present in the data has a table entry, and
      the per-code totals agree with the official `dx_mapping_*.csv` `Total` column
      on all 133 codes.

      **The stratification label is not the diagnosis.** Because `#Dx` ordering
      carries no clinical meaning in this release — PTB-XL's lists are numerically
      sorted, Chapman's are not — there is no primary diagnosis to read off.
      ECGBench derives `stratify_dx` as the globally rarest code each record
      carries, ties broken on the lowest numeric code, purely so stratified folds
      are well defined. It keeps all 133 classes representable, where taking the
      first listed code collapses to 123. **Train on `dx`**, never on
      `stratify_dx`.

      **The code table is packaged with ECGBench, not with the data.** No mapping
      ships in the download, so `ecgbench/data/challenge2021_dx_mapping.csv` holds
      the concatenation of the challenge's own `dx_mapping_scored.csv` (30) and
      `dx_mapping_unscored.csv` (103) from the official scoring code
      ([evaluation-2021](https://github.com/physionetchallenges/evaluation-2021),
      BSD-2-Clause). It covers exactly the 133 codes present, with no duplicate
      code or abbreviation.

      **One record is misnamed.** The `ningbo` cohort contains `S23074` where every
      other record in that cohort is `JS#####` (it is `JS23074` in the
      ecg-arrhythmia release). Record names are still globally unique, so nothing
      collides, but a join on names against ecg-arrhythmia silently drops it.

  - type: table
    title: "Validation summary"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "88,253", "all records, with is_valid + quality_issues"]
      - ["clean", "85,726", "97.1% pass rate"]
      - ["excluded", "2,527", "1,531 with one or more all-zero leads, 1,021 amplitude outliers"]

  - type: description
    title: "About the excluded records"
    body: |
      Two genuine defects, both concentrated in specific cohorts rather than spread
      evenly — another reason not to treat the eight sources as interchangeable.

      **All-zero leads (1,531 records).** Entire leads recorded as exactly 0.0,
      most often the precordials: 976 records have all six of V1–V6 flat.
      Overwhelmingly `ningbo` (1,480 of the 1,531). Verified on disk rather than
      inferred — `A0718`'s lead 11 is all zeros in the file.

      **Amplitude outliers (1,021 records).** Excursions past ±10 mV. Many sit
      exactly at ±32.767 mV, the 16-bit rail, so they are saturation rather than
      physiology. `st_petersburg_incart` is worst hit at 28 of its 74 records
      (37.8%). 105 records also contain NaN samples, and 892 records are excluded
      for an amplitude outlier alone.

      Per-cohort exclusion rates: `st_petersburg_incart` 37.8%, `ningbo` 6.3%,
      `cpsc_2018_extra` 1.8%, `cpsc_2018` 1.7%, `chapman_shaoxing` 0.4%, `ptb`
      0.4%, `georgia` 0.3%, `ptb-xl` 0.2%. `clean/` drops them; `original/` keeps
      them with the reason in `quality_issues`.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset challenge2021 --data-path /path/to/challenge-2021/1.0.3/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # Records vary in length (5 s to 1800 s) and in sampling rate, so take a
      # fixed window before batching — a DataLoader raises as soon as one batch
      # mixes two lengths. 2500 samples is the shortest record, so it always fits.
      #
      # Fold CSVs come from the HuggingFace Hub by default; only the waveforms
      # need to be local.
      ds = ECGDataset(
          "challenge2021",
          split="train",
          data_path="/path/to/challenge-2021/1.0.3/",
          window=(0, 2500),        # first 5 s at the nominal 500 Hz
          labels=True,
      )

      len(ds)                                    # 68573
      ds[0]["signal"].shape                      # (12, 2500)
      ds[0]["record_id"]                         # 'A0001'
      ds[0]["labels"]["dx"]                      # '59118001' — SNOMED-CT, multi-label
      ds[0]["labels"]["dx_abbreviations"]        # 'RBBB'
      ds[0]["labels"]["source"]                  # 'cpsc_2018' — which cohort it came from
      ds[0]["labels"]["sampling_rate"]           # 500 — per record, not dataset-wide
      ds[0]["labels"]["n_samples"]               # 7500 — i.e. 15 s before windowing

      # Lead order is the standard one, so leads= selects by name directly.
      ds.config.lead_names   # ['I','II','III','aVR','aVL','aVF','V1',...,'V6']

      # Multi-hot target over the 30 scored classes:
      import pandas as pd
      from ecgbench.labels.challenge2021 import load_dx_mapping

      mapping = load_dx_mapping()
      scored = list(mapping.index[mapping["scored"]])            # 30 SNOMED codes
      codes = ds.labels_df["dx"].fillna("").astype(str).str.split(",")
      targets = pd.DataFrame(
          {c: codes.apply(lambda lst, c=c: int(c in lst)) for c in scored}
      )
      targets.shape                          # (68573, 30)
      (targets.sum(axis=1) == 0).sum()       # 4882 records carry no scored code

      # Rate is a per-record property: filter, don't pass sampling_rate=.
      ds.labels_df["sampling_rate"].value_counts()   # 500: 68120, 1000: 414, 257: 39

      # Excluding a cohort you plan to evaluate on (see the leakage warnings above):
      keep = ds.labels_df["source"] != "ptb-xl"      # 51114 of 68573 records remain

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/challenge-2021/1.0.3/" }
      - { label: "Challenge website", url: "https://physionetchallenges.org/2021/" }
      - { label: "Challenge paper", url: "https://doi.org/10.23919/CinC53138.2021.9662687" }
      - { label: "Official scoring code and code tables", url: "https://github.com/physionetchallenges/evaluation-2021" }
---
