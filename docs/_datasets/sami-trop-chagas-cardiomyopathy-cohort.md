---
slug: "sami-trop-chagas-cardiomyopathy-cohort"
name: "SaMi-Trop (Chagas Cardiomyopathy Cohort)"
category: "12-lead-other"
order: 8
status: "completed"
source_url: "https://doi.org/10.5281/zenodo.4905618"
url_label: "zenodo.org"
format: "12-lead · 10.24 s · 400 Hz · HDF5"
patients: "1,631"
records: "1,631"
access: "open"
license: "CC BY 4.0"
origin_institution: "UFMG; Uppsala University; EPFL"
origin_country: "Brazil / Sweden / Switzerland"
leads: 12
paper_title: "Lima et al., medRxiv, 2021"
paper_doi: "https://doi.org/10.1101/2021.02.19.21251232"
search_keywords: "sami trop chagas cardiomyopathy brazil minas gerais zenodo hdf5 mortality age survival telehealth tnmg one record per patient"

related:
  - slug: "code-15-pct-telehealth-network-of-minas-gerais-15-pct-subset"
    relation: "sibling_release"
    shares_records: false
    verified: true
    note: >
      Different cohorts recorded by the same Brazilian telehealth network, in the
      same file format, by an overlapping group of authors — but not the same
      recordings. The `exam_id` spaces are the same numbering (SaMi-Trop runs
      3,629-873,765, CODE-15% 13-4,416,614) and the two sets are exactly
      disjoint: 0 of SaMi-Trop's 1,631 ids appear among CODE-15%'s 345,779. If
      SaMi-Trop's ECGs were inside the full CODE cohort, roughly 245 of them
      would be expected in an unbiased 15% sample, so this is evidence of
      absence rather than of sampling. The caveat worth knowing: CODE-15% is 15%
      of CODE-full, and CODE-full is not public, so nothing here rules out these
      recordings sitting in the other 85%. Anyone with CODE-full access should
      check the ids before using both.

sections:
  - type: description
    title: "Overview"
    body: |
      1,631 twelve-lead ECGs from **SaMi-Trop**, an NIH-funded prospective cohort
      of patients with chronic Chagas cardiomyopathy recruited in northern Minas
      Gerais, Brazil, and followed for mortality. It is the third release in this
      catalogue from the Telehealth Network of Minas Gerais after
      [CODE-15%]({{ site.baseurl }}/datasets/code-15-pct-telehealth-network-of-minas-gerais-15-pct-subset.html)
      and [CODE-test]({{ site.baseurl }}/datasets/code-test-827-record-hold-out-test-set.html),
      and the only one of the three that is **not a classification dataset**.

      **A record is a row, not a file.** One HDF5 file holds a single
      `(1631, 4096, 12)` array, so a signal reference reads
      `exams.hdf5:tracings:417`. Loading needs `pip install ecgbench[hdf5]`.

      **There is no diagnostic vocabulary at all.** Unlike its two siblings,
      which share this exact file format, SaMi-Trop ships no abnormality flags.
      The only ECG label is a binary `normal_ecg`. What it ships instead is the
      thing the dataset exists for: **complete mortality follow-up** for every
      one of the 1,631 patients, with 104 deaths over a median 2.07 years.

      **`normal_ecg` is not a healthy control.** Every patient here already has
      chronic Chagas cardiomyopathy, so a normal tracing means a normal tracing
      in a diseased patient. The 286 normal records are not usable as healthy
      controls — the mistake to avoid when pooling this release with others. And
      a record that is *not* normal carries no statement of what is wrong with
      it.

      **One recording per patient**, because the release is each patient's first
      exam. So `patient_id_column` is `null` and the folds are a plain stratified
      split — the rare case in this catalogue where that is genuinely safe rather
      than merely unchecked. There is no patient identifier in the release at
      all, which is sound given one record each.

      Its lead order is the standard `I, II, III, aVR, aVL, aVF`, **checked
      against the arrays rather than assumed**, because the sibling CODE-test
      release from the same network is not standard.

  - type: table
    title: "What ships as a label"
    headers: ["Field", "Value", "Note"]
    rows:
      - ["`normal_ecg`", "286 of 1,631 (17.5%)", "the **only** ECG label"]
      - ["not normal", "1,345 (82.5%)", "no statement of *what* is abnormal"]
      - ["`death`", "104 (6.4%)", "complete — no missing outcomes"]
      - ["`followup_years`", "median 2.07 (0.07-3.39)", "time to death or censoring"]
      - ["`age`", "26-98, median 59", "whole years"]
      - ["`is_male`", "534 M / 1,097 F", "67.3% female"]
      - ["`nn_predicted_age`", "22.6-95.9", "**a model output**, not an observation"]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the shipped `exams.csv`,
      whose md5 matches Zenodo's published value
      (`6c9007a0427f7c3d9e1b6fb091231a67`) exactly, as does `exams.zip`
      (`a7b65f115b0222ad2ecbc6a422496fdd`). `exams.hdf5` has no published
      checksum of its own — it ships inside `exams.zip`, which does.

      **The headline figures agree with the release**: 1,631 records, 1,631
      patients. Note that the *cohort* is 1,959 patients; this release is the
      first ECG of the 1,631 who have one, and the release does not say how the
      328 absent patients differ.

      **The CSV is joined to the waveforms by row position, and nothing else
      could be.** `exams.hdf5` contains only a `tracings` array — no `exam_id`
      dataset, unlike CODE-15%'s parts. That is exactly the situation in which
      CODE-15% turned out **not** to be in file order, so the alignment was
      tested rather than assumed: QRS amplitude is reliably larger in men, and
      splitting the precordial peak-to-peak by the CSV's own `is_male` in row
      order gives a Welch *t* of 4.98, where the largest |t| over 2,000 random
      permutations of the same rows is 3.44. The aligned ordering beats every
      permutation. ECGBench asserts the 1,631-row count on every run, because a
      positional join against a file of the wrong length produces 1,631
      confidently wrong rows rather than an error.

      **The stratification label is not a diagnosis.** `stratify_class` is
      mortality-first — `DEATH` (104), `NORMAL` (283), `ABNORMAL_ALIVE` (1,244) —
      so the rare outcome is what the folds balance. It is three classes and not
      the `death` × `normal_ecg` cross because only 3 records are both dead and
      normal, and a 3-member class cannot be spread over ten folds at all. Never
      train on it.

      **Amplitudes are noisy, like its siblings'.** The median per-record peak is
      4.3 mV, so the validation range is ±20 mV rather than ECGBench's usual ±10,
      matching CODE-15% because these are the same telehealth instruments. ±10
      would exclude 180 records (11.0%); ±20 excludes 14 (0.9%).

  - type: table
    title: "ECGBench splits"
    headers: ["Version", "Records", "Excluded", "Why"]
    rows:
      - ["`original`", "1,631", "—", "all records, with `is_valid` and `quality_issues`"]
      - ["`clean`", "1,615", "16", "14 `amplitude_outlier` (a lead beyond ±20 mV), 2 `missing_leads` (a lead recorded as exactly zero)"]

  - type: code
    title: "Loading with ECGBench"
    language: "python"
    body: |
      from ecgbench import ECGDataset

      # Fold CSVs come from the Hub; signals and labels from your local copy.
      ds = ECGDataset(
          "sami_trop",
          split="train",
          data_path="/path/to/SaMi-Trop/",
          labels=True,
          # The shortest real signal in the release is 1,568 samples, so a
          # wider window risks reading only zero padding on those records.
          window=(0, 1568),
      )
      print(len(ds))                      # 1290
      s = ds[0]
      print(tuple(s["signal"].shape))     # (12, 1568)
      print(s["record_id"])               # 3629
      print(s["labels"]["normal_ecg"])    # False
      print(s["labels"]["death"])         # False
      print(s["labels"]["followup_years"])  # 1.93...
      print(s["labels"]["stratify_class"])  # ABNORMAL_ALIVE

      # A survival target, which is what this release supports.
      df = ds.labels_df
      print(int(df["death"].sum()), "deaths of", len(df))   # 81 deaths of 1290

      # Lead order is standard here and NOT in the sibling CODE-test release,
      # so select by name whenever you use both.
      three = ECGDataset("sami_trop", split="train",
                         data_path="/path/to/SaMi-Trop/",
                         window=(0, 1568), leads=["aVR", "aVL", "aVF"])
      print(tuple(three[0]["signal"].shape))   # (3, 1568)

  - type: links
    title: "Links"
    items:
      - label: "Zenodo record"
        url: "https://doi.org/10.5281/zenodo.4905618"
      - label: "Paper (medRxiv)"
        url: "https://doi.org/10.1101/2021.02.19.21251232"
      - label: "Companion code (ecg-age-prediction)"
        url: "https://github.com/antonior92/ecg-age-prediction"
      - label: "Example script"
        url: "https://github.com/vlbthambawita/ECGBench/blob/main/examples/load_sami_trop.py"
---
