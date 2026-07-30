---
slug: "chapman-shaoxing-ecg-database-10-646-patients"
name: "Chapman-Shaoxing ECG Database (10,646 patients)"
category: "12-lead-other"
order: 2
status: "completed"
source_url: "https://figshare.com/collections/ChapmanECG/4560497/2"
url_label: "figshare.com"
format: "12-lead · 10 s · 500 Hz · CSV (µV)"
patients: "10,646"
records: "10,646"
access: "open"
license: "CC0 1.0"
origin_institution: "Chapman University; Shaoxing People's Hospital"
origin_country: "China / USA"
leads: 12
paper_title: "Zheng et al., Scientific Data, 2020"
paper_doi: "https://doi.org/10.1038/s41597-020-0386-x"
search_keywords: "chapman shaoxing figshare csv microvolts rhythm beat muse denoised"

sections:
  - type: description
    title: "Overview"
    body: |
      The original figshare release of the Chapman-Shaoxing database: 10,646
      twelve-lead ECGs, one per patient, each 10 seconds at 500 Hz. Signals are
      **CSV rather than WFDB** — one file per record, 5,000 rows × 12 lead columns
      with a header, in the standard lead order — and are stored in
      **microvolts**, which ECGBench converts to millivolts via
      `signal_unit_scale: 0.001`.

      Labels are unusually rich for an open ECG dataset: a single-label `Rhythm`
      class (11 values), a space-separated multi-label `Beat` annotation (57
      codes), age and sex, and eleven automated measurements — ventricular and
      atrial rate, QRS duration, QT and QTc, R and T axes, QRS count, and the
      QRS/T fiducial offsets. A denoised copy of every signal ships alongside as
      `ECGDataDenoised/`.

      The collection is six separate files and its metadata is `.xlsx`, so
      acquisition needs `examples/download_chapman_figshare.py` rather than
      ECGBench's single-archive auto-download.

  - type: table
    title: "Rhythm distribution (all 10,646 records)"
    headers: ["Code", "Rhythm", "Records"]
    rows:
      - ["SB",    "Sinus bradycardia", "3,889"]
      - ["SR",    "Sinus rhythm", "1,826"]
      - ["AFIB",  "Atrial fibrillation", "1,780"]
      - ["ST",    "Sinus tachycardia", "1,568"]
      - ["SVT",   "Supraventricular tachycardia", "587"]
      - ["AF",    "Atrial flutter", "445"]
      - ["SA",    "Sinus irregularity", "399"]
      - ["AT",    "Atrial tachycardia", "121"]
      - ["AVNRT", "AV node reentrant tachycardia", "16"]
      - ["AVRT",  "AV reentrant tachycardia", "8"]
      - ["SAAWR", "Sinus atrium to atrial wandering rhythm", "7"]

  - type: description
    title: "About those counts"
    body: |
      These are recomputed from the shipped `Diagnostics.xlsx`, and they match the
      paper's figures — 10,646 records over 11 rhythm classes. `Rhythm` is
      single-label, so the counts sum exactly to the record total.

      `Beat` is a different quantity: space-separated and multi-label, averaging
      0.84 codes per record with 5,419 records carrying `NONE`. Do not add the two
      tables together.

      Note the class imbalance: the rarest rhythm (SAAWR) has **7** records, fewer
      than the 10 folds, so `StratifiedKFold` warns and those records cannot appear
      in every fold. That is a property of the data, not a splitting fault.

  - type: table
    title: "Validation summary (500 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "10,646", "all records, with is_valid + quality_issues"]
      - ["clean", "10,607", "99.6% pass rate"]
      - ["excluded", "39", "38 amplitude outliers, 49 flat-lead issues (overlapping)"]

  - type: code
    title: "Fetching and building the splits"
    language: bash
    body: |
      # Six files, 2.8 GB. md5-verified, resumable, converts the .xlsx metadata.
      python examples/download_chapman_figshare.py --dest /path/to/chapman-figshare/

      ecgbench splits --dataset chapman_shaoxing --data-path /path/to/chapman-figshare/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      ds = ECGDataset(
          "chapman_shaoxing",
          split="train",
          version="clean",
          data_path="/path/to/chapman-figshare/",
          labels=True,
      )

      sample = ds[0]
      sample["signal"].shape          # (12, 5000), float32, millivolts
      sample["labels"]["Rhythm"]      # 'AFIB'
      sample["labels"]["Beat"]        # 'LVHV'  (space-separated, multi-label)
      sample["labels"]["QTCorrected"] # 454

  - type: links
    title: "References"
    items:
      - { label: "figshare collection", url: "https://figshare.com/collections/ChapmanECG/4560497/2" }
      - { label: "Scientific Data paper", url: "https://doi.org/10.1038/s41597-020-0386-x" }
      - { label: "PhysioNet ecg-arrhythmia (same cohort, WFDB)", url: "https://physionet.org/content/ecg-arrhythmia/1.0.0/" }
---
