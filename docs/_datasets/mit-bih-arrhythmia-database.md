---
slug: "mit-bih-arrhythmia-database"
name: "MIT-BIH Arrhythmia Database"
category: "two-lead"
order: 1
status: "completed"
source_url: "https://physionet.org/content/mitdb/1.0.0/"
url_label: "physionet.org"
format: "2-lead (usually MLII + V1) · 30 min · 360 Hz · WFDB"
patients: "47"
records: "48"
access: "open"
license: "ODC-By 1.0"
origin_institution: "Beth Israel Hospital / MIT"
origin_country: "USA"
leads: 2
paper_title: "Moody & Mark, IEEE EMBS 2001"
paper_doi: "https://doi.org/10.13026/C2F305"
search_keywords: "mit-bih arrhythmia usa beth israel hospital mit mlii v1 holter ambulatory benchmark annotations beat rhythm two-lead"

sections:
  - type: description
    title: "Overview"
    body: |
      The reference arrhythmia benchmark. 48 half-hour excerpts of two-channel
      ambulatory ECG, digitised at 360 Hz from Holter tapes recorded at the Beth
      Israel Hospital Arrhythmia Laboratory between 1975 and 1979, carrying
      **109,494 beat annotations** across fifteen types plus rhythm episodes,
      signal-quality changes and artefact markers. Each record was annotated
      independently by two or more cardiologists and the disagreements resolved.

      **The database is two deliberate halves, and ECGBench stratifies on that.**
      Records 100–124 (23 of them) were chosen at random from a pool of over 4000
      recordings, to be representative of routine clinical material. Records
      200–234 (25) were selected from the same pool for rare but clinically
      important phenomena. With only 48 records over 10 folds, a fold drawn
      without regard to that split can easily be all one half — the difference
      between a test set with ventricular flutter in it and one without.

      **The two leads are not the same two in every record.** This is the one
      dataset in the catalogue where records store the same *number* of leads
      under different *names*, so nothing about a signal's shape reveals which
      lead is which. See "Lead layout" below — it changes how you must call
      `leads=`.

      **Records are long — 650,000 samples, about 5 MB each.** Take a fixed
      `window=(start, length)`, which is read at load time rather than cropped
      afterwards, or use `batch_size=1`.

      **Folds are grouped by analog tape.** 48 records come from 47 subjects:
      records 201 and 202 were cut from the same tape (1960), which the shipped
      directory states and the tape number in the header confirms. One subject
      out of 47 — but it is the only one there is, and ungrouped those two land
      in different folds most of the time.

  - type: description
    title: "Lead layout — why you must select leads by name here"
    body: |
      In most records the upper signal is a modified limb lead II (MLII), taken
      from chest electrodes, and the lower a modified V1. Not always. Counted from
      all 48 headers:

      | Layout | Records | Which |
      |---|---|---|
      | `MLII, V1` | 40 | the rest |
      | `MLII, V5` | 2 | 100, 123 |
      | `MLII, V2` | 2 | 103, 117 |
      | `V5, V2` | 2 | 102, 104 |
      | `MLII, V4` | 1 | 124 |
      | `V5, MLII` | 1 | **114 — the two signals are reversed** |

      Records 102 and 104 have **no MLII at all**: surgical dressings made a
      modified II impossible, so V5 was used for the upper signal. Record 114 has
      the predominant pair the wrong way round, which the source documents as
      something that "happens occasionally in clinical practice" and which
      arrhythmia detectors should cope with.

      Every one of the 48 records stores exactly **2** leads, so
      `alternate_lead_names` — which maps a lead *count* to a layout — cannot
      express any of this. The config declares `record_lead_layouts` instead, and
      `ECGDataset` then resolves the requested lead **names** against each
      record's own header:

      ```python
      ds = ECGDataset("mitdb", split="train", data_path=..., leads=["MLII"])
      ds[0]["signal"]    # record 100: MLII, read from position 0
      # record 114: MLII, read from position 1 — an index would have returned V5
      # record 102: raises ValueError; it stores V5/V2 and has no MLII to return
      ```

      Without this, `signal[0]` is a limb-type lead in 46 records and a chest lead
      in 2, and `leads=["MLII"]` returns V5 for three of them with no error.

  - type: table
    title: "Reference beat annotations"
    headers: ["Beat type", "Symbol", "Count", "Share", "Records containing it"]
    rows:
      - ["normal beat", "N", "75,052", "68.54%", "40"]
      - ["left bundle branch block beat", "L", "8,075", "7.37%", "4"]
      - ["right bundle branch block beat", "R", "7,259", "6.63%", "6"]
      - ["premature ventricular contraction", "V", "7,130", "6.51%", "37"]
      - ["paced beat", "/", "7,028", "6.42%", "4"]
      - ["atrial premature beat", "A", "2,546", "2.33%", "27"]
      - ["fusion of paced and normal beat", "f", "982", "0.90%", "3"]
      - ["fusion of ventricular and normal beat", "F", "803", "0.73%", "17"]
      - ["nodal (junctional) escape beat", "j", "229", "0.21%", "5"]
      - ["aberrated atrial premature beat", "a", "150", "0.14%", "7"]
      - ["ventricular escape beat", "E", "106", "0.10%", "2"]
      - ["nodal (junctional) premature beat", "J", "83", "0.08%", "5"]
      - ["unclassifiable beat", "Q", "33", "0.03%", "6"]
      - ["atrial escape beat", "e", "16", "0.01%", "1"]
      - ["supraventricular premature beat", "S", "2", "0.00%", "1"]
      - ["**total beats**", "", "**109,494**", "", "48"]

  - type: table
    title: "Time spent in each annotated rhythm"
    headers: ["Rhythm", "Code", "Minutes", "Share", "Records containing it"]
    rows:
      - ["normal sinus rhythm", "N", "1,055.4", "73.08%", "42"]
      - ["atrial fibrillation", "AFIB", "132.6", "9.18%", "8"]
      - ["paced rhythm", "P", "110.2", "7.63%", "4"]
      - ["ventricular bigeminy", "B", "41.8", "2.90%", "12"]
      - ["sinus bradycardia", "SBR", "30.1", "2.08%", "1"]
      - ["ventricular trigeminy", "T", "19.0", "1.32%", "12"]
      - ["atrial flutter", "AFL", "13.1", "0.91%", "3"]
      - ["pre-excitation (WPW)", "PREX", "12.3", "0.85%", "1"]
      - ["second degree heart block", "BII", "11.7", "0.81%", "1"]
      - ["nodal (AV junctional) rhythm", "NOD", "4.7", "0.32%", "3"]
      - ["ventricular tachycardia", "VT", "3.6", "0.25%", "13"]
      - ["supraventricular tachyarrhythmia", "SVTA", "3.5", "0.24%", "7"]
      - ["idioventricular rhythm", "IVR", "2.4", "0.17%", "2"]
      - ["ventricular flutter", "VFL", "2.4", "0.17%", "1"]
      - ["atrial bigeminy", "AB", "1.5", "0.10%", "1"]
      - ["**total annotated**", "", "**1,444.3**", "", "48"]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the 48 headers and 48 `.atr`
      files, after verifying all 226 signal and annotation files against the
      release's own `SHA256SUMS.txt`. Most of them reconcile with the published
      record exactly, which is unusual in this catalogue and worth stating:

      - **109,494 beats** is the figure the database has published since 1980, and
        it is what the fifteen beat symbols sum to here. The `.atr` files hold
        **112,647** annotations in total; the other **3,153** are markers that are
        not beats — 1,291 rhythm changes, 616 signal-quality changes, 472
        ventricular-flutter waves, 437 comment annotations, 193 non-conducted P
        waves, 132 isolated artefacts and 12 flutter-episode delimiters. ECGBench
        counts them in their own columns and keeps them out of `n_beats`; adding
        any of them is how a wrong total gets quoted.
      - **47 subjects, 25 men aged 32–89 and 22 women aged 23–89**, exactly as the
        shipped directory states. Age is recorded as `-1` on records 103 and 219,
        which ECGBench returns as NaN rather than as a number — the mean age of
        63.6 is therefore over 45 known subjects, not 47.
      - **The subject identity had to be derived.** No header names a subject; the
        third field of the first comment line is the analog tape, and it is
        distinct for every record except 201 and 202, which share `1960` — exactly
        the pair the directory names. The fourth field is the Del Mar Avionics
        recorder: grouping by it reproduces the directory's recorder table to the
        record, including record 208 being the one whose recorder was never
        traced, and the one whose field reads `N/A`. Both are exposed
        (`patient_id`, `recorder`); the second is a real confounder, since one
        recorder accounts for 13 of the 48 records.

      **The rhythm table measures time, not markers.** A rhythm annotation opens
      an episode that runs until the next one, so a record with two AFIB markers
      and two sinus markers may be 99% sinus. The 1,444.3 minutes annotated is 9
      seconds short of the 1,444.5 minutes the 48 records contain, because in a
      few records the first rhythm annotation comes a beat or two after the start.

      **Beat classes are not a record-level label, and they are wildly
      imbalanced.** Almost every record carries several beat types, so there is no
      single beat class per record; and several classes live in one or two records
      (all 472 ventricular-flutter waves are in record 207, 427 of the 428
      missed-beat markers are in record 231, and the 7,028 paced beats come from 4
      records). The `stratify_class` used for folds is neither — it is the
      random/selected halves, and it is for fold construction only.

  - type: table
    title: "Validation summary (360 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "48", "all records, with is_valid + quality_issues"]
      - ["clean", "45", "93.8% pass rate"]
      - ["excluded", "3", "103, 116, 223 — each saturates the 11-bit ADC"]

  - type: description
    title: "About the excluded records, and the amplitude range"
    body: |
      `amplitude_range_mv` is `[-5.11, 5.11]` here rather than ECGBench's usual
      `[-10, 10]`, and it is set from the hardware rather than from physiology.
      The recordings are format 212 at 11 bits with a gain of 200 adu/mV and
      `adc_zero` 1024, so the full scale is exactly **−5.120 to +5.115 mV** — no
      sample in this database can lie outside it, and **any range wider than that
      is a check that provably never fires**.

      At the rail it does something real. Three records reach it, i.e. clip:

      | Record | Lead | Samples at the rail | Peak |
      |---|---|---|---|
      | 116 | MLII | 0.14% | ±5.12 mV |
      | 223 | V1 | 0.11% | ±5.12 mV |
      | 103 | V2 | 0.04% | ±5.12 mV |

      The next-highest peak anywhere in the release is 4.94 mV (record 200), so
      the threshold separates the saturating records from every other one cleanly.
      The clipping is brief — well under a second in total per record — so for
      beat detection these records are perfectly usable; take the `original`
      version if you want them, where they are present and flagged.

      There are **no NaN samples and no flat or all-zero leads anywhere** in this
      release, and all 48 records are exactly 650,000 samples, so the truncation
      check is enabled rather than skipped.

  - type: description
    title: "What is not in this dataset"
    body: |
      The download contains three things ECGBench deliberately ignores, all
      checked against the files rather than assumed:

      - **`x_mitdb/` (23 records).** Each is the first 600 s of a record already in
        the release, produced by `xform` with the baseline removed — correlation
        1.0 against its parent over all 216,000 samples. Including them would put
        the same recording into the partition twice. ECGBench takes its record
        list from the shipped `RECORDS` file, which names the 48 and none of these.
      - **`102-0.atr`.** 2,192 annotations with the same symbol counts as
        `102.atr`, and no header of its own.
      - **`108.at_`.** A superseded copy of `108.atr`.

      The shipped `ANNOTATORS` file lists `atr` and nothing else, which is what
      ECGBench reads.

      Note also that **the two channels are not synchronous**: the directory
      records skew of up to 40 ms between them, part fixed per recorder and part
      variable from tape wobble. Anything comparing the channels sample-for-sample
      has to allow for it.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset mitdb --data-path /path/to/mitdb/1.0.0/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # 650,000-sample records: a window is needed to batch at all, and because
      # window= is pushed into the reader it also avoids decoding the other 1795 s.
      ds = ECGDataset(
          "mitdb",
          split="train",
          data_path="/path/to/mitdb/1.0.0/",
          window=(0, 3600),        # first 10 s at 360 Hz
          labels=True,
      )

      len(ds)                                  # 37
      ds[0]["signal"].shape                    # torch.Size([2, 3600])
      ds[0]["record_id"]                       # 100
      ds[0]["labels"]["lead_names"]            # 'MLII|V5' — this record's own layout
      ds[0]["labels"]["patient_id"]            # 'tape1085' — folds are grouped by this
      ds[0]["labels"]["dominant_rhythm"]       # 'N'
      ds[0]["labels"]["n_beats"]               # 2273
      ds[0]["labels"]["pvc_fraction"]          # 0.00044

      # Select by NAME. Records 102 and 104 store V5/V2 and raise rather than
      # returning V5 where MLII was asked for; record 114 returns MLII from
      # position 1, where an index-based selection would have returned V5.
      mlii = ECGDataset("mitdb", split="train", data_path="...",
                        window=(0, 3600), leads=["MLII"])
      mlii[0]["signal"].shape                  # torch.Size([1, 3600])

      # Rhythm burden per record, straight off the reference annotations:
      ds.labels_df["rhythm_secs_AFIB"].max()   # 1769.92 -- record 210, AFIB for
                                               # all but 36 s of its 1805.6 s

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/mitdb/1.0.0/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/C2F305" }
      - { label: "MIT-BIH Arrhythmia Database Directory", url: "https://physionet.org/content/mitdb/1.0.0/mitdbdir/intro.htm" }
      - { label: "PhysioBank beat annotation definitions", url: "https://archive.physionet.org/physiobank/annotations.shtml" }
---
