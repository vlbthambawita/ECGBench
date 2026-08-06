---
slug: "ikem-dataset-institute-for-clinical-and-experimental-medicine-prague"
name: "IKEM Dataset (Institute for Clinical and Experimental Medicine, Prague)"
category: "12-lead-other"
order: 9
status: "completed"
source_url: "https://doi.org/10.5281/zenodo.8393007"
url_label: "zenodo.org"
format: "8 reduced leads · 8.192 s · 500 Hz · HDF5"
patients: "30,290"
records: "98,130"
access: "open"
license: "CC BY-NC-ND 4.0"
origin_institution: "IKEM (Institute for Clinical and Experimental Medicine)"
origin_country: "Czech Republic — Prague"
leads: 8
paper_title: "Seják et al., Knowledge-Based Systems, 2023"
paper_doi: "https://doi.org/10.1016/j.knosys.2023.111014"
search_keywords: "ikem prague czech republic institute clinical experimental medicine zenodo hdf5 cardiology 8 lead reduced no diagnoses age estimation not published noderivatives"

sections:
  - type: description
    title: "Overview"
    body: |
      98,130 ECGs from 30,290 patients recorded in routine care at **IKEM** in
      Prague, a national tertiary referral centre for cardiology,
      transplantation and diabetology. Acquisition runs 2004-2022 but is
      overwhelmingly 2017 (57,179 records) and 2018 (33,355).

      **Four facts about this release differ from what it says about itself**,
      and each was measured from the files rather than taken from the
      description. They are the reason this page is long.

      **1. Eight leads, not twelve — and the order is the most unusual in the
      catalogue.** Only the independent leads are stored, as
      `V1, V2, V3, V4, V5, V6, II, I`: precordial first, and **II before I**. So
      `signal[0]` is V1, not lead I. III, aVR, aVL and aVF are exact linear
      combinations of II and I and are simply absent; ECGBench returns the 8
      stored leads and does not synthesise the others. Deriving them is exact
      arithmetic — the example script shows it.

      **2. It is 8.192 s, not 10.** The description says "sampled at a rate of
      500 Hz for a duration of 10 seconds each", which would be 5,000 samples.
      The arrays hold 4,096. The rate really is 500 Hz — see below for how that
      was settled — so 4,096 / 500 = 8.192 s.

      **3. The samples are microvolts, not the 4.88 µV counts stated.** Applying
      the declared granularity makes the median per-record peak 9.5 mV, roughly
      five times physiologic.

      **4. No diagnoses ship.** `exams.csv` carries age, sex, weight, height and
      two cart-measured rates, and nothing else. The paper trains on IKEM
      diagnostic labels that are **not part of this release**. This dataset looks
      like a 98,130-record classification corpus and is not one: what it supports
      is age and sex estimation, rate regression and self-supervised pretraining.

      **Patients repeat heavily.** 19,078 of the 30,290 patients contributed more
      than one recording and one contributed 96 — 86,918 records (88.6%) belong
      to a multi-record patient. ECGBench's folds are grouped on `patient_id`, a
      40-character SHA-1 surrogate. Unusually, there is nothing else hiding:
      every one of the 98,130 waveform SHA-1s the release ships is distinct, so
      unlike CODE-15% there are no byte-identical duplicate recordings behind
      different patient ids.

  - type: description
    title: "ECGBench does not publish this dataset's fold CSVs"
    body: |
      The release ships a `LICENSE` file that is the verbatim text of
      **CC BY-NC-ND 4.0**. The NoDerivatives term makes republishing a derived
      fold table — 98,130 `exam_id`s and 30,290 patient hashes — on a public,
      ungated, commercially operated repository legally unclear, and publication
      is effectively irreversible. So ECGBench withholds it. This is the second
      such dataset after
      [MIMIC-IV-ECG]({{ site.baseurl }}/datasets/mimic-iv-ecg.html) and the first
      withheld by **licence** rather than by a data use agreement.

      (Zenodo's own metadata field says only `other-at` — "Other, Attribution" —
      which is both less specific and less restrictive than the licence text
      travelling with the data. The licence text governs. If the depositors
      confirm that a fold table is acceptable, this decision should be revisited.)

      The split is distributed as a **recipe** instead. Fold assignment is a pure
      function of `exams.csv` and a fixed seed (`random_state=42`), so anyone can
      regenerate the identical partition:

      1. `ecgbench splits --dataset ikem --data-path /path/to/IKEM_dataset_v1.0.0/`
      2. `python -c "from ecgbench import verify_splits; verify_splits('ikem', 'output/ikem')"`
      3. `cp -r output/ikem/clean output/ikem/original /path/to/IKEM_dataset_v1.0.0/`

      Step 2 compares your run against the reference manifest shipped in the
      package (`ecgbench/data/manifests/ikem.json` — 728 bytes of seed,
      checksums, counts and one fold digest, no identifiers) and raises on
      mismatch, naming the differing input file. Step 3 is what lets
      `metadata_source="local"` find the fold CSVs: in local mode `data_path`
      serves as both the signal root and the splits root.

      `ECGDataset("ikem")` with the default `metadata_source="hf"` raises
      `SplitsNotPublishedError` quoting these steps, rather than a bare 404.

  - type: table
    title: "How the corrections were measured"
    headers: ["Claim in the release", "What the files show", "Method"]
    rows:
      - ["10 s per record", "**8.192 s** (4,096 samples at 500 Hz)", "R-peak detection at 500 Hz reproduces the cart's own `ventricular_rate` with median bias **0.0 bpm**, median absolute error 0.5 bpm and 88.7% of records within 5 bpm; assuming 400 Hz gives bias −14.3 bpm"]
      - ["granularity 4.88 µV", "**1 µV per count**", "at 4.88 µV the median lead p2p is 7.7 mV and the median per-record peak 9.5 mV; at 1 µV they are 1.58 and 1.93 mV, matching the other hospital cohort here (SPH, 1.74 mV). The stored integers also have a GCD of 1, so the resolution is genuinely 1 count"]
      - ["\"8 reduced leads\" (unnamed)", "**V1-V6, II, I**", "ch0-ch5 have the banded within-record correlation structure of a precordial sweep (adjacent 0.59-0.70, ch0 vs ch5 −0.18) with net QRS dominance running −377 counts at ch0 (rS = V1) to +408 at ch5 (dominant R = V6); no channel matches any augmented-lead identity (best relative error 0.49, where a true match elsewhere is under 0.003); and taking ch7 as I puts the frontal QRS axis at a median **+51°** (IQR 14-91) with aVF positive in 82.1%, where the swap gives +1° and 50.4%"]
      - ["licence CC BY 4.0 (as this catalogue previously said)", "**CC BY-NC-ND 4.0**", "the shipped `LICENSE` file is that licence verbatim; Zenodo's metadata says the vaguer `other-at`"]

  - type: description
    title: "About those counts"
    body: |
      All five published files match the md5s Zenodo lists and the shipped
      `zenodo_md5sums.txt` (verified with `md5sum -c`). The headline figures —
      **98,130 records, 30,290 patients** — agree with the release, and the three
      HDF5 parts hold exactly 48,264 + 48,683 + 1,183 = 98,130 rows whose
      `exam_id` sets match `exams.csv` exactly, with no padding row (CODE-15%
      has one per part).

      **`-1` is a missing-value sentinel in every numeric column**, and the
      missingness is severe rather than incidental: weight 89.6%, height 89.3%,
      age 9.0%, atrial rate 0.4%, **sex** 0.4%, ventricular rate 0.006%. Because
      the sentinel is a value rather than a blank, `notna()` on the raw CSV
      reports all 98,130 records complete and every summary statistic comes out
      wrong — read literally, the cohort's mean weight is about **−76 kg**.
      ECGBench converts every `-1` to NaN, exposes `is_male` as a nullable
      boolean so its 376 unknowns survive, and adds `has_age`/`has_weight`/
      `has_height`. That is lossless: the source has no genuine blanks and no
      genuine `-1` measurement.

      Values that survive the filter still deserve suspicion — 546 records give
      age 0, 21 give 100 or more, and one gives a ventricular rate of 0. Those
      are left as they are, since they are not sentinels and guessing which are
      real is not a loader's job.

      **`acquisition_date` is `MM-DD-YYYY`**, which sorts wrongly as a string:
      taking string min/max suggests a 2018-2021 range where the true range is
      2004-03-18 to 2022-07-26. Day-first parsing would silently swap month and
      day for the first 12 days of every month.

      **48 records are shorter than they look.** The arrays are rectangular at
      4,096 samples, but the parts carry a `real_lengths` dataset and 48 records
      hold only 2,500 real samples (5.0 s), zero-padded into the rest. That
      column is exposed as `real_length_samples`.

      **The stratification label is a measurement, not a diagnosis.** With no
      diagnoses to stratify on, folds use the cart's ventricular rate banded into
      `BRADY` (12,387), `NORMAL` (76,426) and `TACHY` (9,317). A rate of 75 bpm
      is as compatible with atrial fibrillation as with sinus rhythm. Never train
      on it.

  - type: table
    title: "ECGBench splits"
    headers: ["Version", "Records", "Excluded", "Why"]
    rows:
      - ["`original`", "98,130", "—", "all records, with `is_valid` and `quality_issues`"]
      - ["`clean`", "96,096", "2,034 (2.07%)", "1,879 `amplitude_outlier` (a lead beyond ±10 mV — the worst peaks land exactly on 32.767 mV, which is int16 full scale, i.e. a railed ADC), 190 `missing_leads`"]

  - type: code
    title: "Loading with ECGBench"
    language: "python"
    body: |
      from ecgbench import ECGDataset

      # metadata_source="local" is required: ECGBench publishes no IKEM fold
      # CSVs (CC BY-NC-ND). Run the three-step recipe above first, so the fold
      # tree sits inside the dataset directory.
      ds = ECGDataset(
          "ikem",
          split="train",
          data_path="/path/to/IKEM_dataset_v1.0.0/",
          metadata_source="local",
          labels=True,
          # 48 records hold only 2,500 real samples; sized to them so no record
          # returns pure zero padding.
          window=(0, 2500),
      )
      print(len(ds))                      # 76891
      s = ds[0]
      print(tuple(s["signal"].shape))     # (8, 2500)   <- EIGHT leads
      print(s["record_id"])               # 19
      print(s["labels"]["ventricular_rate"])  # 98.0
      print(s["labels"]["weight"])        # nan   <- -1 in the source
      print(s["labels"]["stratify_class"])    # NORMAL

      # signal[0] is V1, NOT lead I. Always select by name.
      leads = ECGDataset("ikem", split="train",
                         data_path="/path/to/IKEM_dataset_v1.0.0/",
                         metadata_source="local", window=(0, 2500),
                         leads=["I", "II", "V1"])
      print(tuple(leads[0]["signal"].shape))   # (3, 2500)

      # The four leads the release drops are exact linear combinations:
      import numpy as np
      sig = ds[0]["signal"].numpy()
      i, ii = sig[7], sig[6]          # I is row 7, II is row 6
      iii, avr = ii - i, -(i + ii) / 2
      avl, avf = i - ii / 2, ii - i / 2

  - type: links
    title: "Links"
    items:
      - label: "Zenodo record"
        url: "https://doi.org/10.5281/zenodo.8393007"
      - label: "Paper (Knowledge-Based Systems)"
        url: "https://doi.org/10.1016/j.knosys.2023.111014"
      - label: "Example script"
        url: "https://github.com/vlbthambawita/ECGBench/blob/main/examples/load_ikem.py"
---
