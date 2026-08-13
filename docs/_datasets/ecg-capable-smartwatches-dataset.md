---
slug: "ecg-capable-smartwatches-dataset"
name: "ECG-Capable Smartwatches Dataset"
category: "one-lead"
order: 10
status: "completed"
source_url: "https://physionet.org/content/ecg-capable-smartwatches/1.0.0/"
url_label: "physionet.org"
format: "1-lead watches + 12-lead reference · 11–30 s · 250/300/500/512 Hz · WFDB"
patients: "0 (synthetic — a patient simulator)"
records: "915 (736 in the clean partition)"
access: "restricted"
license: "PhysioNet Restricted Health Data License 1.5.0"
origin_institution: "Complutense University of Madrid / Hospital Universitario Ramón y Cajal"
origin_country: "Spain — Madrid"
leads: 1
paper_title: "Recas et al., PhysioNet, 2025 (companion paper pending revision)"
paper_doi: "https://doi.org/10.13026/7018-y383"
search_keywords: "smartwatch ecg apple watch series 9 samsung galaxy watch 6 fitbit sense 2 withings scanwatch philips tc30 metron ps-440 patient simulator synthetic iec 60601 st-segment heart rate r-wave amplitude spain madrid restricted"
patients_class: "count-na"

sections:
  - type: description
    title: "Overview"
    body: |
      A bench test, not a cohort. A **METRON PS-440 patient simulator** was
      stepped through 36 calibrated settings and its output recorded
      *simultaneously* by a Philips TC30 hospital electrocardiograph — the
      clinical reference — and by four ECG-capable consumer smartwatches: Apple
      Watch Series 9, Samsung Galaxy Watch 6, Fitbit Sense 2 and Withings
      ScanWatch. The protocol is IEC 60601-2-25:2011, the accuracy standard for
      electrocardiographic devices.

      This is the only release in ECGBench with **no human subject at all**. Its
      Ethics section is one line — "Data collected from synthetic sources" — and
      the consequence is that the ground truth is not a clinical label inferred by
      an expert but *the knob the simulator was set to*, known exactly. That makes
      it the natural place to measure what a wrist-worn device does to amplitude,
      rate and ST morphology, and a poor place to train anything meant to see a
      real heart.

      Each watch sat on an adjustable stand with the simulator's right-arm output
      wired to the crown and its left-arm output to the caseback, giving an
      arm-to-arm **lead I**; the electrocardiograph was connected with standard
      12-lead cables and captured all twelve derivations.

  - type: table
    title: "The five devices, as the files actually are"
    headers: ["Directory", "Device", "Records", "Leads", "Rate", "Length", "In `clean`"]
    rows:
      - ["`philips_tc30`", "Philips TC30 *(reference)*", "195", "**12**", "500 Hz", "11.0 s", "195"]
      - ["`applewatch_serie8`", "Apple Watch **Series 9**", "180", "1", "512 Hz", "27.3–30.0 s", "180"]
      - ["`samsunggalaxy6`", "Samsung Galaxy Watch 6", "179", "1", "500 Hz", "30.002 s", "**0**"]
      - ["`fitbitsense2`", "Fitbit Sense 2", "181", "1", "250 Hz", "30.0 s", "181"]
      - ["`withingsscanwatch`", "Withings ScanWatch", "180", "1", "300 Hz", "30.0 s", "180"]
      - ["**total**", "", "**915**", "", "", "", "**736**"]

  - type: table
    title: "The 36 simulator settings"
    headers: ["Family", "Settings", "Nominal range", "Records", "In `clean`", "Folds it reaches"]
    rows:
      - ["`freq_test`", "15", "30 – 300 bpm", "391", "316", "10 of 10"]
      - ["`st-segment`", "16", "−800 – +800 µV, 100 µV steps", "399", "320", "10 of 10"]
      - ["`amp_test`", "4", "500 / 1000 / 1500 / 2000 µV", "100", "80", "**4 of 10**"]
      - ["`sqr-2hz`", "1", "2 Hz square wave", "25", "20", "**1 of 10**"]

  - type: description
    title: "Nine things to know before using it"
    body: |
      All 1,833 shipped files were verified against the release's own
      `SHA256SUMS.txt` before any figure on this page was computed, so everything
      below is an upstream property of the release rather than download damage.

      **1. The smartwatch records are lead I and their headers say `II`.** The
      Methods state the wiring (right arm → crown, left arm → caseback) and that
      the watches' own exports are "formatted as single-lead (Lead I)
      electrocardiograms". LA − RA *is* lead I. All 720 smartwatch headers
      nonetheless name the channel `II`. ECGBench's `lead_names` follows the
      files, because that is what `leads=` resolves against — so
      `ECGDataset(leads=["II"])` hands back the Philips' genuine lead II for 195
      records and an arm-to-arm lead I for the other 720, with no error. **Filter
      on the labels' `derivation` or on `device` before comparing morphology
      across devices.** This is the one thing here that can silently corrupt a
      cross-device comparison.

      **2. The reference device is 12-lead; only the watches are single-lead.**
      195 records store 12 channels, 720 store one, so the config declares the
      single-lead layout and puts the 12-lead order in `alternate_lead_names`.
      `leads=` is re-resolved against whichever layout the loaded record holds:
      `leads=["V4"]` returns the true V4 of a Philips record and **raises** for a
      smartwatch record rather than returning its only channel. Batching the
      dataset without `leads=` fails in `default_collate`, because a batch cannot
      mix 1- and 12-lead tensors.

      **3. Every Samsung record ends in an invalid sample, and it costs the whole
      device its place in `clean`.** All 179 are 15,001 samples where 15,000 is
      30.000 s at 500 Hz, and that extra final sample is digital `−32768` —
      WFDB's invalid-sample marker for format 16 — which `wfdb` returns as NaN.
      `check_nan_values` has no threshold, so all 179 fail it and **`clean`
      contains no Samsung Galaxy Watch 6 record at all**. Nothing else in the
      release holds a NaN. The signal before that sample is intact:
      `window=(0, 15000)` reads a Samsung record with none. Use `version="original"`
      and filter on `trailing_invalid_sample` if you want the device back.

      **4. "All experiments in quintuplicate" is false for 20 records.** 36
      settings × 5 devices × 5 repetitions is 900; the release ships 915.
      Seventeen settings carry a **sixth** repetition — every Philips `freq_test`
      setting, plus Fitbit's `f80` and `ST-m6` — and two carry only **four**:
      Samsung's `st-p8` and Fitbit's `ST-p8`. No two records in the release hold
      an identical signal, so the sixth repetitions are genuine extra
      acquisitions and are kept; `is_extra_replicate` flags them.

      **5. Fitbit spells its ST directories in uppercase, and every Fitbit header
      names the wrong device.** Fitbit stores `st-segment/ST-m1/ST-m1_0` where
      the other four devices store `st-segment/st-m1/st-m1_0`, so a setting key
      taken verbatim describes 32 ST conditions instead of 16 — and would put one
      simulator condition in two different folds. `setting_id` is lowercased for
      exactly that reason. Separately, all 181 Fitbit headers carry the comment
      "Withings Scanwatch reading METRON PS-440 patient simulator": a copy-paste
      error, since those records are 250 Hz against Withings' 300 Hz and share no
      signal with them.

      **6. `RECORDS` names 75 files that do not exist.** The shipped index lists
      Withings' `freq_test` records under `WithingsScanwatch/` while the directory
      is `withingsscanwatch/`, so 75 of its 915 lines resolve to nothing on a
      case-sensitive filesystem, and building paths from it yields 75 records that
      all fail `corrupt_header`. ECGBench enumerates the headers from disk and
      reports the case-only mismatches as such, rather than as missing data.

      **7. The Apple directory says Series 8 and the release says Series 9.** The
      abstract and the Data Description both name the Apple Watch Series 9, and
      the latter maps `applewatch_serie8` to it explicitly; the directory and all
      180 header comments say "Serie 8". `device` keeps the directory name so
      paths stay traceable, and `device_model` carries the release's own prose
      answer.

      **8. Every record was rescaled to fill int16, so the header gain is a
      per-record quantity.** 914 of 915 reach digital `+32767` and the same number
      reach `−32767` or `−32768`; gains run 15,082 to 207,386 adu/mV and differ
      record by record. Millivolts come back correctly from `wfdb` — no unit scale
      is applied — but a record sitting at the rail is **not** clipping: that is
      where its own extremes were mapped. The widest samples in the release are
      −2.8280 mV and +2.5850 mV.

      **9. Length and rate both vary by device, so a window in samples is not a
      window in time.** `expected_samples` is therefore unset, and
      `window=(0, 5500)` — 11.0 s of Philips, 22.0 s of Fitbit — is the largest
      window that fits every record.

  - type: description
    title: "Folds group on the simulator setting, not on a patient"
    body: |
      There is no patient to group on, so `patient_id_column` points at
      `setting_id`: the 36 simulator conditions. Three measurements decided that,
      all made over the shipped files with max-over-lag correlation of the single
      lead (for the Philips, its lead II), resampled to a common 250 Hz where the
      devices differ:

      | Pair | Median correlation |
      |---|---|
      | Same device, same setting, different repetition | 0.95 *(self-control)* |
      | **Different device**, same setting | **0.803** |
      | Same device, adjacent `st-segment` settings | 0.805 |
      | Same device, adjacent `freq_test` settings | 0.070 |

      The five repetitions of a setting are near-duplicates, and — because the
      five instruments recorded the same simulator output at the same instant —
      **so is the same setting on a different device**. Grouping on
      `(device, setting)` would therefore still leak; the group has to be the
      setting across all five devices. That it works at all is shown by the last
      row: different rate settings really are different signals.

      Two consequences worth stating plainly.

      **The ST ladder is a 100 µV step, so holding out `st-p2` is not holding out
      an unseen condition** the way holding out `f240` is. That is a property of a
      densely sampled continuous label rather than a defect to engineer around —
      collapsing the family into one group would leave four groups in total and no
      way to build ten folds — but anything reporting ST accuracy should say which
      offsets were held out rather than quoting a fold number.

      **`amp_test` reaches 4 of the 10 folds and `sqr-2hz` reaches 1.** A
      stratification class needs `n_folds` groups to appear in every fold, and
      those families have 4 settings and 1. No fold count above four fixes it. So
      the default `test` split (fold 10) holds `freq_test` and `st-segment`
      records only — 62 clean records at three settings. For amplitude or
      square-wave work, pass `split=None` with `fold_numbers=[...]` and pick the
      folds that hold the family, or hold out settings by hand.

      Fold sizes are 77–102 records, every device gets 15–22 records per fold, and
      **no setting spans a fold boundary** — verified from the exported fold CSVs.

  - type: table
    title: "The ECGBench partition"
    headers: ["Split", "Folds", "`original`", "`clean`", "Notes"]
    rows:
      - ["train", "1–8", "736", "592", "—"]
      - ["val", "9", "102", "82", "—"]
      - ["test", "10", "77", "62", "`f80`, `st-m2`, `st-m4` only"]
      - ["**total**", "", "**915**", "**736**", "179 excluded, all Samsung"]

  - type: description
    title: "What the reference device shows that the watches do not"
    body: |
      One figure worth looking at before choosing a task. Mean peak-to-peak span
      of the recorded signal, in millivolts, against the nominal R-wave amplitude
      the simulator was set to — computed from the labels' `span_mv`, which is the
      record's own range and **not** an R-wave amplitude measurement:

      | Device | 500 µV | 1000 µV | 1500 µV | 2000 µV | Monotonic? |
      |---|---|---|---|---|---|
      | Philips TC30 *(reference)* | 1.642 | 2.569 | 3.846 | 5.128 | yes |
      | Apple Watch Series 9 | 0.364 | 0.738 | 1.095 | 1.474 | yes |
      | Samsung Galaxy Watch 6 | 0.623 | 1.014 | 1.349 | 1.714 | yes |
      | Fitbit Sense 2 | 1.209 | 1.041 | 1.502 | 1.818 | **no** |
      | Withings ScanWatch | 0.331 | 0.650 | 1.613 | 1.274 | **no** |

      No device reproduces the nominal amplitude on any absolute scale — every
      record was independently rescaled to fill int16 (point 8), so absolute
      millivolts are not comparable across records in the first place. What *is*
      recoverable is the ordering, and two of the four watches lose even that on a
      four-point sweep. Amplitude fidelity is sampled at exactly four settings and
      the square-wave response at one, so treat both as illustrations rather than
      as measurements; rate and ST are the two axes this release samples densely.

  - type: description
    title: "Splits are generated, not downloaded"
    body: |
      The release is restricted-access under the **PhysioNet Restricted Health
      Data License 1.5.0**, whose clause 3 reads "The LICENSEE will not share
      access to PhysioNet restricted data with anyone else." Fold CSVs carry
      identifiers only — and here those identifiers are simulator settings rather
      than people, since no human was ever recorded — but the licence travelling
      with the data governs whatever the data turns out to contain, which is the
      same rule ECGBench applied to IKEM. So `publish_fold_csvs` is `false`:
      `ecgbench upload` refuses the dataset before any network call, and
      `ECGDataset` raises `SplitsNotPublishedError` quoting the regeneration
      command rather than a bare 404.

      The partition is distributed as a **recipe** instead. It is reproducible
      because fold assignment is a pure function of the input table and a fixed
      seed (`random_state=42`), and because the input table is itself regenerated
      deterministically from the signal files — regenerating it from scratch
      reproduces the same bytes and the same `fold_digest`.

      ```bash
      # 0. Check your copy first — the release ships its own checksums
      cd /path/to/ecg-capable-smartwatches/1.0.0/ && sha256sum -c SHA256SUMS.txt

      # 1. Generate — about 12 s, most of it decoding all 915 records once
      ecgbench splits --dataset ecg_capable_smartwatches \
                      --data-path /path/to/ecg-capable-smartwatches/1.0.0/

      # 2. Verify it is the canonical partition, not merely a plausible one
      python -c "from ecgbench import verify_splits; \
                 print(verify_splits('ecg_capable_smartwatches', \
                                     'output/ecg_capable_smartwatches')['ok'])"

      # 3. Copy the fold tree next to the signals so metadata_source='local' finds it
      cp -r output/ecg_capable_smartwatches/clean \
            output/ecg_capable_smartwatches/original \
            /path/to/ecg-capable-smartwatches/1.0.0/
      ```

      Step 1 writes `ecgbench_metadata.csv` into the dataset directory, which must
      therefore be writable — the release ships no metadata file of any kind, so
      ECGBench builds one from the directory names and the headers. Step 2 compares
      against `ecgbench/data/manifests/ecg_capable_smartwatches.json`, which ships
      with the package: seed, record counts, input checksum and a `fold_digest`
      over the whole record-to-fold mapping. That input checksum is of the
      *generated* table rather than a provider-published file, which is why step 0
      is not optional: `sha256sum -c SHA256SUMS.txt` is what establishes that the
      signals themselves are the canonical ones.

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset, load_labels

      # metadata_source="local" is required: the splits are not on the Hub.
      ds = ECGDataset("ecg_capable_smartwatches", split="test", version="clean",
                      data_path="/path/to/ecg-capable-smartwatches/1.0.0/",
                      metadata_source="local", labels=True)

      len(ds)                              # 62  -- fold 10; three settings only
      ds[0]["record_id"]                   # 'applewatch_serie8_f80_0'
      ds[0]["signal"].shape                # torch.Size([1, 15360])  -- 30 s @ 512 Hz
      ds.units                             # 'mV'
      ds.lead_names                         # ('II',)  -- but see below

      # The label is the simulator setting, not a diagnosis.
      ds[0]["labels"]["device_model"]       # 'Apple Watch Series 9'
      ds[0]["labels"]["setting_id"]         # 'f80'
      ds[0]["labels"]["nominal_rate_bpm"]   # 80.0
      ds[0]["labels"]["sampling_rate"]      # 512

      # THE HEADERS SAY "II" AND THE WATCHES RECORD LEAD I. Check `derivation`
      # before comparing morphology across devices.
      ds[0]["labels"]["derivation"]         # 'lead I (LA-RA)'
      ds.labels_df["derivation"].value_counts().to_dict()
      # {'lead I (LA-RA)': 46, 'standard 12-lead': 16}

      # Rates and lead counts differ per device, so batching needs BOTH leads= and
      # window=. 5500 samples is the longest window that fits every record.
      batched = ECGDataset("ecg_capable_smartwatches", split="test", version="clean",
                           data_path="/path/to/ecg-capable-smartwatches/1.0.0/",
                           metadata_source="local", leads=["II"], window=(0, 5500))
      batched[0]["signal"].shape           # torch.Size([1, 5500])

      # leads= is resolved against each record's own layout: a chest lead works for
      # the 12-lead reference and REFUSES for a single-lead watch record.
      ref = ECGDataset("ecg_capable_smartwatches", split="test", version="clean",
                       data_path="/path/to/ecg-capable-smartwatches/1.0.0/",
                       metadata_source="local", leads=["V4"], window=(0, 5500))
      # ValueError: Record 'applewatch_serie8_f80_0' stores 1 lead(s) (['II']), and
      # this dataset uses more than one lead layout. Lead 'V4' is not in ...

      # Samsung's trailing invalid sample, and the window that steps over it.
      orig = ECGDataset("ecg_capable_smartwatches", split="test", version="original",
                        data_path="/path/to/ecg-capable-smartwatches/1.0.0/")
      # orig[i]["signal"] for a Samsung record holds exactly 1 NaN, at the last sample
      # window=(0, 15000) holds none

      # The full label table, all 915 records regardless of split or validity.
      labels = load_labels("ecg_capable_smartwatches",
                           "/path/to/ecg-capable-smartwatches/1.0.0/")   # (915, 26)
      labels["trailing_invalid_sample"].sum()            # 179  -- every Samsung record
      labels.groupby("device_model")["sampling_rate"].max().to_dict()
      # {'Apple Watch Series 9': 512, 'Fitbit Sense 2': 250, 'Philips TC30': 500,
      #  'Samsung Galaxy Watch 6': 500, 'Withings ScanWatch': 300}

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page (restricted access)", url: "https://physionet.org/content/ecg-capable-smartwatches/1.0.0/" }
      - { label: "DOI", url: "https://doi.org/10.13026/7018-y383" }
      - { label: "IEC 60601-2-25:2011", url: "https://webstore.iec.ch/en/publication/16851" }
      - { label: "Example script", url: "https://github.com/vlbthambawita/ECGBench/blob/main/examples/load_ecg_capable_smartwatches.py" }
---
