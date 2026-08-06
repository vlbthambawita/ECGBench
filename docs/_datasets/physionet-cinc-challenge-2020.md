---
slug: "physionet-cinc-challenge-2020"
name: "PhysioNet/CinC Challenge 2020"
category: "12-lead-physionet"
order: 22
status: "completed"
source_url: "https://physionet.org/content/challenge-2020/1.0.2/"
url_label: "physionet.org"
format: "12-lead · 5–1,800 s · 257/500/1,000 Hz"
patients: "—"
records: "43,101"
access: "open"
license: "CC BY 4.0"
origin_institution: "CPSC, INCART, PTB, PTB-XL, Georgia"
origin_country: "Multi-national (China, Russia, Germany, USA)"
leads: 12
paper_title: "Alday et al., Physiological Measurement, 2020"
paper_doi: "https://doi.org/10.1088/1361-6579/abc960"
search_keywords: "cinc challenge 2020 physionet china russia germany usa cpsc georgia incart ptb snomed multi-label meta-dataset"
patients_class: "count-na"

related:
  - slug: "ptb-xl"
    relation: "contains"
    shares_records: true
    verified: true
    note: >
      All 21,837 records of the `ptb-xl` cohort are PTB-XL's 500 Hz recordings,
      renamed `HR00001`-`HR21837` where the number is PTB-XL's own `ecg_id`. Verified
      from the files: every one of these `.mat` files is bit-identical to its
      Challenge 2021 counterpart, which was in turn checked against PTB-XL's
      `records500/` signals (max abs difference 0.0000 mV over all 12 leads and 5,000
      samples). Note the count — 21,837 is the **v1.0.1** total, so this cohort still
      holds the 38 duplicate/triplicate records that PTB-XL v1.0.3 removed. Evaluating
      a challenge-trained model on PTB-XL is testing on training data.

  - slug: "ptb-diagnostic-ecg-database"
    relation: "contains"
    shares_records: true
    verified: true
    note: >
      The 516 records of the `ptb` cohort are PTBDB recordings, renamed
      `S0001`-`S0549` (516 of PTBDB's 549), and are bit-identical to the same-named
      records in Challenge 2021, where 20 of 20 sampled records were matched against
      PTBDB at correlation ≥ 0.99999 — e.g. `S0138` ≡ `patient044/s0146lre`. The
      numbering is **not** a crosswalk (S0138 is not s0138), so the mapping is only
      recoverable by signal matching. That matters beyond leakage: PTBDB's patient
      directories are the only way to know which of these records share a patient,
      and that grouping is lost here.

  # The INCART overlap is declared on the INCART page, not here — catalogue.py
  # derives the inverse, and declaring it twice fails tests/test_catalogue.py.

  - slug: "cpsc-2018-china-physiological-signal-challenge-2018"
    relation: "contains"
    shares_records: true
    verified: false
    note: >
      The `cpsc_2018` cohort holds 6,877 records, exactly the size of the CPSC-2018
      public training set, renamed `A0001`-`A6877`; `cpsc_2018_extra` adds 3,453
      `Q####` records that CPSC-2018 did not release publicly. Assume the 6,877
      overlap completely and do not evaluate on CPSC-2018 after training here. The
      count agreement is strong evidence but was not checked against CPSC-2018 files —
      unlike the other three cohort overlaps on this page.

sections:
  - type: description
    title: "Overview"
    body: |
      The public training set of the 2020 PhysioNet/Computing in Cardiology
      Challenge, *Classification of 12-lead ECGs*: **43,101 twelve-lead records**
      pooled from **six source cohorts** across China, Russia, Germany and the
      USA, redistributed in one uniform WFDB tree.

      **This release is contained entirely in
      [Challenge 2021]({{ site.baseurl }}/datasets/physionet-cinc-challenge-2021.html).**
      Not "overlaps with" — contained. All 43,101 record names appear there, and
      all 43,101 waveform files are **bit-identical**, checked by comparing the two
      releases' own published `SHA256SUMS.txt`. The 2021 challenge simply added the
      Chapman-Shaoxing and Ningbo cohorts on top (88,253 − 45,152 = 43,101). Only
      705 of the 43,101 headers differ at all, and those differences are label
      housekeeping described below. **Never train on one year and evaluate on the
      other.**

      **It is also a meta-dataset**, which is the second thing to know. Four of the
      six cohorts are datasets ECGBench catalogues separately, and the records were
      renamed, so the overlap is invisible from the identifiers. Training here and
      evaluating on PTB-XL, PTBDB, INCART or CPSC-2018 is testing on training data.

      **Only the training set was ever published.** The challenge's hidden
      validation and test sets are not in this release, which is why the record
      count is 43,101 and not the ~52,500 the challenge described. There is
      consequently no reproducible official split, and ECGBench generates its own
      10 folds.

      **Sampling rate and record length are per-record properties here.** Rate is
      500 Hz for 42,511 records, 1000 Hz for the 516 `ptb` ones and 257 Hz for the
      74 `st_petersburg_incart` ones; length runs from 5 s to 1800 s, with 1,650
      distinct lengths in the `cpsc_2018` cohort alone. Both are exposed as label
      columns. Records cannot be batched as they are — take a fixed
      `window=(start, length)`, or use `batch_size=1`.

      Labels are **multi-label SNOMED-CT codes** from the `#Dx` header field: 111
      distinct codes, 2.18 per record on average and up to 10, with no unlabelled
      record. No code table ships with the data, so ECGBench packages the
      challenge's own.

  - type: table
    title: "Source cohorts"
    headers: ["Cohort", "Records", "Rate", "Length", "Also catalogued as"]
    rows:
      - ["ptb-xl", "21,837", "500 Hz", "10 s", "PTB-XL (v1.0.1, in full)"]
      - ["georgia", "10,344", "500 Hz", "5–10 s", "— (no standalone release)"]
      - ["cpsc_2018", "6,877", "500 Hz", "6–144 s", "CPSC-2018 training set"]
      - ["cpsc_2018_extra", "3,453", "500 Hz", "8–98 s", "— (unreleased CPSC extra set)"]
      - ["ptb", "516", "1,000 Hz", "32–120 s", "PTB Diagnostic ECG Database (516 of 549)"]
      - ["st_petersburg_incart", "74", "257 Hz", "1,800 s", "St Petersburg INCART (74 of 75)"]
      - ["**total**", "**43,101**", "", "", ""]

  - type: table
    title: "Most frequent diagnoses"
    headers: ["Diagnosis", "Abbr.", "SNOMED-CT", "Records", "Scored"]
    rows:
      - ["sinus rhythm", "NSR", "426783006", "20,846", "yes"]
      - ["left axis deviation", "LAD", "39732003", "6,086", "yes"]
      - ["myocardial infarction", "MI", "164865005", "6,021", "no"]
      - ["t wave abnormal", "TAb", "164934002", "4,673", "yes"]
      - ["left ventricular hypertrophy", "LVH", "164873001", "3,759", "no"]
      - ["nonspecific st t abnormality", "NSSTTA", "428750005", "3,554", "no"]
      - ["atrial fibrillation", "AF", "164889003", "3,475", "yes"]
      - ["abnormal QRS", "abQRS", "164951009", "3,389", "no"]
      - ["myocardial ischemia", "MIs", "164861001", "2,559", "no"]
      - ["right bundle branch block", "RBBB", "59118001", "2,402", "yes"]
      - ["sinus tachycardia", "STach", "427084000", "2,402", "yes"]
      - ["1st degree av block", "IAVB", "270492004", "2,394", "yes"]
      - ["sinus bradycardia", "SB", "426177001", "2,359", "yes"]
      - ["st depression", "STD", "429622005", "1,977", "no"]
      - ["ventricular ectopics", "VEB", "164884008", "1,944", "no"]
      - ["(96 further codes)", "", "", "", ""]

  - type: description
    title: "About those counts"
    body: |
      **Records: 43,101, not ~52,500.** The challenge description quoted the full
      corpus including the hidden validation and test sets; PhysioNet only ever
      published the training set. Recomputed as the number of `.hea` files under
      `training/`. All 86,301 shipped data files were verified against the
      release's own `SHA256SUMS.txt` before any figure on this page was computed.

      **Patients: not published.** No patient identifier ships with any cohort and
      records were renamed, so patient counts are unrecoverable from these files.
      This is exact for four cohorts (one record per patient), but **not** for
      `ptb` and `st_petersburg_incart`, whose source datasets have several
      recordings per patient — 113 of PTBDB's 290 patients and roughly 32 patients
      across INCART's 75 records. Those 590 records (1.37%) can therefore place one
      patient in more than one fold. Every other dataset in ECGBench with repeated
      patients groups folds by patient; this one cannot.

      **631 records repeat a code inside their own `#Dx` list, and ECGBench
      deduplicates them.** 596 records list `284470004` (PAC) twice, 30 list
      `17338001` (VPB) twice, five others repeat one code and one lists a code three
      times — 628 in `georgia` and 3 in `cpsc_2018_extra`. This is not cosmetic:
      counting raw list entries inflates Georgia's PAC total from 639 records to
      1,236 and makes the shipped v1.0.2 data look as though it disagrees with the
      official code table. **After deduplication all 111 codes and all six
      per-cohort columns of the official `dx_mapping_*.csv` reproduce exactly**
      (93,843 code-record pairs). The Challenge 2021 re-release of the same records
      had already deduplicated and numerically sorted these lists, which is why the
      2021 page does not mention this.

      **The diagnosis table is multi-label and does not sum to the record total.**
      It counts records carrying each code, over 93,843 code instances across
      43,101 records. Derived from the `#Dx` field of all 43,101 headers, joined to
      the challenge's code table. The `Scored` column marks the 27 classes the 2020
      challenge metric evaluated (as 24 classes, three pairs — CRBBB/RBBB,
      PAC/SVPB, PVC/VPB — being scored as equivalent); 37,749 records carry at
      least one scored code and **5,352 carry none**.

      **The scored subset is not Challenge 2021's.** 2021 scored 30 classes: these
      27 plus PRWP (`365413008`) and CLBBB (`733534002`), neither of which occurs
      anywhere in this release, and BBB (`6374002`), which does occur here (137
      records) but was unscored in 2020. Nothing scored in 2020 was dropped in 2021.
      Use the table packaged for the year you are reporting against.

      **The stratification label is not the diagnosis.** Because `#Dx` ordering
      carries no clinical meaning in this release — it varies by cohort and, in
      Georgia, is not internally consistent — there is no primary diagnosis to read
      off. ECGBench derives `stratify_dx` as the globally rarest code each record
      carries, ties broken on the lowest numeric code, purely so stratified folds
      are well defined. It keeps all 111 classes representable, where taking the
      first listed code collapses to 102. **Train on `dx`**, never on
      `stratify_dx`.

      **The code table is packaged with ECGBench, not with the data.** No mapping
      ships in the download, so `ecgbench/data/challenge2020_dx_mapping.csv` holds
      the concatenation of the challenge's own `dx_mapping_scored.csv` (27) and
      `dx_mapping_unscored.csv` (84) from the official scoring code
      ([evaluation-2020](https://github.com/physionetchallenges/evaluation-2020),
      BSD-2-Clause). It covers exactly the 111 codes present, with no duplicate
      code or abbreviation.

      **Age carries two sentinel values.** 181 records have no age at all; beyond
      that, 204 `ptb-xl` records record `300` (PTB-XL's own convention for a
      patient older than 89) and 6 CPSC records record `-1`. Genuine ages run 1–92.
      ECGBench leaves all three states in the `age` column rather than collapsing
      them, and exposes the sentinels as
      `ecgbench.labels.challenge2020.AGE_SENTINELS`. **Sex is normalised**: the 74
      `st_petersburg_incart` records spell it `M`/`F` where the other five cohorts
      spell it `Male`/`Female`, and the loader maps them onto the long form. One
      record has no sex.

  - type: description
    title: "What changed between this release and Challenge 2021"
    body: |
      Worth stating precisely, because it is the only difference between two
      datasets that hold the same recordings.

      All 43,101 `.mat` waveform files are **bit-identical** across the two
      releases — compared via both releases' published `SHA256SUMS.txt`, 43,101 of
      43,101 hashes equal. Of the 43,101 headers, **42,396 are identical and 705
      differ**, and every one of those 705 differences is in the comment block, not
      the signal specification:

      - **631 records** — the `#Dx` list. 2021 deduplicated the repeated codes
        described above and sorted the codes numerically; 2020 ships them as
        written. Once ECGBench deduplicates, the *sets* of codes agree on all
        43,101 records.
      - **74 records** — `#Sex` in the `st_petersburg_incart` cohort, `M`/`F` in
        2020 and `Male`/`Female` in 2021. ECGBench normalises 2020 to match.

      Age, sampling rate, sample count, lead count and cohort assignment are
      identical on all 43,101 records. So the two datasets differ in their label
      *encoding* and in the 45,152 extra records 2021 adds — not in any recording.

  - type: table
    title: "Validation summary"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "43,101", "all records, with is_valid + quality_issues"]
      - ["clean", "42,815", "99.3% pass rate"]
      - ["excluded", "286", "250 amplitude outliers (8 of them also with NaN samples), 36 with one or more all-zero leads"]

  - type: description
    title: "About the excluded records"
    body: |
      Two genuine defects, both concentrated in specific cohorts rather than spread
      evenly — another reason not to treat the six sources as interchangeable.

      **Amplitude outliers (250 records).** Excursions past ±10 mV. Many sit exactly
      at ±32.767 mV, the 16-bit rail, so they are saturation rather than physiology.
      `st_petersburg_incart` is worst hit at 28 of its 74 records (37.8%). Eight of
      these records also contain NaN samples.

      **All-zero leads (36 records).** Entire leads recorded as exactly 0.0, 85 leads
      in total; one record has all twelve flat.

      Per-cohort exclusion rates: `st_petersburg_incart` 37.8%, `cpsc_2018_extra`
      1.8%, `cpsc_2018` 1.7%, `ptb` 0.4%, `georgia` 0.3%, `ptb-xl` 0.2%. These match
      Challenge 2021's rates for the same six cohorts exactly, which they must —
      they are the same files. `clean/` drops them; `original/` keeps them with the
      reason in `quality_issues`.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset challenge2020 --data-path /path/to/challenge-2020/1.0.2/

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
          "challenge2020",
          split="train",
          data_path="/path/to/challenge-2020/1.0.2/",
          window=(0, 2500),        # first 5 s at the nominal 500 Hz
          labels=True,
      )

      len(ds)                                    # 34258
      ds[0]["signal"].shape                      # (12, 2500)
      ds[0]["record_id"]                         # 'A0001'
      ds[0]["labels"]["dx"]                      # '59118001' — SNOMED-CT, multi-label
      ds[0]["labels"]["dx_abbreviations"]        # 'RBBB'
      ds[0]["labels"]["source"]                  # 'cpsc_2018' — which cohort it came from
      ds[0]["labels"]["sampling_rate"]           # 500 — per record, not dataset-wide
      ds[0]["labels"]["n_samples"]               # 7500 — i.e. 15 s before windowing

      # Lead order is the standard one, so leads= selects by name directly.
      ds.config.lead_names   # ['I','II','III','aVR','aVL','aVF','V1',...,'V6']

      # Multi-hot target over the 27 classes the 2020 metric scored. Note this is
      # NOT the 2021 subset — import the mapping for the year you report against.
      import pandas as pd
      from ecgbench.labels.challenge2020 import load_dx_mapping

      mapping = load_dx_mapping()
      scored = list(mapping.index[mapping["scored"]])            # 27 SNOMED codes
      codes = ds.labels_df["dx"].fillna("").astype(str).str.split(",")
      targets = pd.DataFrame(
          {c: codes.apply(lambda lst, c=c: int(c in lst)) for c in scored}
      )
      targets.shape                          # (34258, 27)
      (targets.sum(axis=1) == 0).sum()       # 4221 records carry no scored code

      # Rate is a per-record property: filter, don't pass sampling_rate=.
      ds.labels_df["sampling_rate"].value_counts()   # 500: 33802, 1000: 417, 257: 39

      # Excluding a cohort you plan to evaluate on (see the leakage warnings above):
      keep = ds.labels_df["source"] != "ptb-xl"      # 16869 of 34258 records remain

      # Age ships with sentinels — 300 means "over 89", -1 means nothing.
      from ecgbench.labels.challenge2020 import AGE_SENTINELS
      real_age = pd.to_numeric(
          ds.labels_df["age"].where(~ds.labels_df["age"].astype(str).isin(AGE_SENTINELS)),
          errors="coerce",
      )
      real_age.min(), real_age.max(), round(real_age.mean(), 1)   # (1.0, 92.0, 60.1)

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/challenge-2020/1.0.2/" }
      - { label: "Challenge website", url: "https://physionetchallenges.org/2020/" }
      - { label: "Challenge paper", url: "https://doi.org/10.1088/1361-6579/abc960" }
      - { label: "Official scoring code and code tables", url: "https://github.com/physionetchallenges/evaluation-2020" }
---
