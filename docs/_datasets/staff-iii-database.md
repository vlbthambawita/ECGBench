---
slug: "staff-iii-database"
name: "STAFF III Database"
category: "12-lead-physionet"
order: 24
status: "completed"
source_url: "https://physionet.org/content/staffiii/1.0.0/"
url_label: "physionet.org"
format: "9-lead (12-lead derivable) · 94.5–960 s · 1,000 Hz · 0.625 µV resolution · WFDB"
patients: "104"
records: "520"
access: "open"
license: "ODC Attribution"
origin_institution: "Charleston Area Medical Center; Lund University"
origin_country: "USA / Sweden"
leads: 9
paper_title: "Martínez et al., CinC 2017"
paper_doi: "https://doi.org/10.22489/CinC.2017.266-133"
search_keywords: "staff iii staffiii ptca balloon inflation occlusion ischemia ischaemia coronary angioplasty st changes transient charleston usa sweden lund 9-lead"

sections:
  - type: description
    title: "Overview"
    body: |
      520 recordings from **104 patients** undergoing prolonged elective
      percutaneous transluminal coronary angioplasty (PTCA) at Charleston Area
      Medical Center, West Virginia, in 1995–96. Almost everything else in this
      catalogue labels a recording with a diagnosis; STAFF III labels it with a
      **position in a procedure**, and that is what makes it valuable: the
      ischaemia is deliberate, timed, and each patient serves as their own control.

      Each patient contributed a short series around one angioplasty:

      | Code | Phase | Records | Patients | Duration (min – median – max) |
      |---|---|---|---|---|
      | `BR` | baseline, hospital room | 73 | 73 | 300 – 300 – 300 s |
      | `BC` | baseline, catheterisation lab | 114 | 103 | 100.8 – 300 – 648.9 s |
      | `BI` | **balloon inflated** | 142 | 104 | 94.5 – 518.4 – 960 s |
      | `PC` | post-inflation, cathlab | 95 | 93 | 101.7 – 300 – 670.3 s |
      | `PR` | post-inflation, room | 96 | 94 | 300 – 300 – 300 s |
      | | **total** | **520** | **104** | 52.4 hours |

      The 142 inflation records carry **152 balloon inflations** — nine records
      hold two or three — annotated in WFDB `.event` files with sample-accurate
      inflation, deflation and contrast-injection instants. Occlusions ran **28 s
      to 595 s (median 289 s)**, 10.4 hours of controlled coronary occlusion in
      total, and the occluded vessel is recorded for every one.

      **There are 9 signals per record, not 12, and the precordials come first.**
      Files store `V1 V2 V3 V4 V5 V6 I II III`. aVR, aVL and aVF are exact linear
      combinations of I and II and were not stored, so the montage is a standard
      12-lead one in the clinical sense but `signal[0]` is **V1, not lead I**. The
      declared order was verified against all 520 headers, and against the signals
      themselves: Einthoven's III = II − I holds to 2–6 quantisation steps
      (LSB = 0.625 µV), so the three limb channels are what they claim to be.

      **Folds are grouped by patient, and that is not optional here.** 104 patients
      contributed 520 records — a mean of 5 each, minimum 2, maximum 7 — and a
      patient's baseline, occlusion and recovery are the same heart under the same
      electrodes minutes apart. Split those across train and test and the score
      measures patient identity. No patient spans a fold.

      **Record length leaks the label.** Inflation records have a median duration
      of 518 s against 300 s for every other phase, so a model handed raw lengths
      can read the phase straight off. Window to a fixed length before training.
      The shortest record is 94.514 s, so `window=(0, 90000)` is the largest round
      window that loads every record.

  - type: table
    title: "Occluded vessel"
    headers: ["Territory", "Inflations", "Patients (primary)", "Raw values in the spreadsheet"]
    rows:
      - ["LAD", "58", "34", "prox LAD, mid LAD, prox mid LAD, LAD diag"]
      - ["RCA", "58", "47", "prox RCA, mid RCA, dist RCA, prox mid RCA"]
      - ["LCx", "33", "21", "prox circ, mid circ, dist circ"]
      - ["Left main", "3", "2", "left main"]
      - ["**total**", "**152**", "**104**", ""]

  - type: table
    title: "Cohort"
    headers: ["Attribute", "Value"]
    rows:
      - ["Patients", "104 (65 male, 39 female)"]
      - ["Age", "32–100, mean 60.8 (102 patients; 14 and 15 have none recorded)"]
      - ["No prior MI", "69 patients"]
      - ["Prior MI — inferior", "18"]
      - ["Prior MI — anterior", "9"]
      - ["Prior MI — inferior + anterior", "5"]
      - ["Prior MI — lateral / posterior / septal", "1 each"]
      - ["Contrast injections annotated", "210"]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the 520 headers, the 142
      `.event` files and the shipped annotation spreadsheet, after verifying all
      **1,189 files against the release's own `SHA256SUMS.txt`** (all match).
      Points where the shipped data differs from what is commonly quoted:

      - **104 patients, not 108.** File numbers run to 108 but 28, 67, 78 and 103
        are unused. The spreadsheet says so itself: *"the database contains 104
        patients (not 108 as mentioned in some publications since 4 file numbers
        are unused)"*.
      - **520 records.** This page previously recorded "152 inflations", which is
        the inflation count, not the record count — 152 inflations occur across 142
        of the 520 records. Both figures are given above.
      - **9 stored leads, not 12.** Descriptions of STAFF III as "standard 12-lead"
        are clinically correct — the missing three are derivable — but a program
        reading the files gets 9 channels. This page's `leads` field now says 9.
      - **The spreadsheet's `D2` field is unreliable and ECGBench ignores it.**
        `D0;D1;D2` is time-to-inflation, inflation duration, and time from
        deflation to end of file. D0 and D1 agree with the `.event` markers on all
        152 inflations to within a second, but D2 disagrees with the actual record
        length on **30 of 142 records**, by up to 575 s. All timings exposed by
        ECGBench come from the `.event` files, and record length from the header.
      - **Age and sex agree perfectly** between the headers and the spreadsheet on
        all 520 records, so neither source needed to be preferred.

      **`recording_type` is the label; `stratify_class` is not.** Folds are
      stratified on the patient's primary occluded territory, not on the protocol
      phase, and that is deliberate. Because folds are patient-grouped, only
      patient-level attributes can actually be balanced — and every patient
      contributes roughly the same mix of phases, so the phase distribution comes
      out balanced whatever the split does, while the occluded vessel does not.
      The 2 left-main patients are pooled into `OTHER` (any class under 10
      patients is), which is why `stratify_class` has four values and the
      territory table above has four rows that do not match it. Train on
      `recording_type`; `primary_artery_territory` keeps the unpooled value.

  - type: table
    title: "Validation summary (1,000 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "520", "all records, with is_valid + quality_issues"]
      - ["clean", "476", "91.5% pass rate"]
      - ["excluded", "44", "42 clipped at the ADC rail, 1 unreadable, 1 with a dead lead"]

  - type: description
    title: "About the excluded records, and the amplitude range"
    body: |
      `amplitude_range_mv` is `[-20, 20]` here rather than ECGBench's usual
      `[-10, 10]`, and that is measured, not conventional. This is a working
      catheterisation laboratory: catheter and electrode movement produce
      transient excursions well past 10 mV that are ordinary artefact, and **71 of
      520 records** contain at least one sample beyond ±10 mV. At ±10 the check
      would discard 13.7% of the dataset for noise.

      The line worth drawing is **saturation**, where samples were genuinely lost.
      The ADC rail sits at exactly ±20.48 mV (±32768 at a gain of 1600 adu/mV), and
      37 records touch it. ±20 catches those plus four that came within 0.5 mV,
      while passing the 30 records with large but unclipped excursions.

      The clipping is **transient, not a dead lead**: the worst-affected record
      (`001c`) has its worst lead clipped for 3.8% of samples, and no record has a
      lead pinned for any sustained stretch — unlike INCART, where the analogous
      check catches leads railed for most of a recording.

      A quirk of WFDB explains why 27 records are flagged for NaN rather than
      amplitude: **format 16 reserves −32768 as a missing-sample marker**, so
      samples clipped at the *negative* rail arrive from `wfdb.rdrecord` as NaN,
      not as −20.48 mV. The two checks are catching the same physical event from
      opposite rails, which is why their record sets overlap in 26 of 27 cases.

      | Reason | Records |
      |---|---|
      | `amplitude_outlier` (positive rail) | 40 |
      | `nan_values` (negative rail, −32768) | 27 |
      | `missing_leads` + `flat_line` (`016f`, lead V6 all zeros) | 1 |
      | `corrupt_header` (`089d`) | 1 |
      | **union** | **44** |

      **`089d` is broken in the published release, not in transit.** Its header
      declares 468,554 samples and its `.dat` holds 300,000, so `wfdb.rdrecord`
      refuses it outright. Both files match the shipped checksums, so this is
      upstream. The truncation falls *after* the balloon deflation — the inflation
      runs 0–278 s and the file covers 0–300 s — so the ischaemic episode itself
      is intact; read it with `sampto=300000` if you want it. `089e` has the
      mirror-image defect harmlessly: its header declares 300,000 samples and the
      `.dat` holds ~366,667, and wfdb simply ignores the tail.

      **Patients 1, 4, 5, 6 and 89 are flagged by the depositors** for possible
      lead or sign reversal. The spreadsheet does not say which leads, so ECGBench
      flags all 23 of their records with `suspect_leads` rather than guessing.
      These records are *not* excluded from `clean/` — the signals are valid, the
      lead identities are in doubt.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset staffiii --data-path /path/to/staffiii/1.0.0/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # Records run 94.5-960 s, so a fixed window must fit the SHORTEST one.
      # window= is pushed into the wfdb reader, so the 16-minute records decode
      # 90 s rather than all of it.
      ds = ECGDataset(
          "staffiii",
          split="train",
          data_path="/path/to/staffiii/1.0.0/",
          window=(0, 90_000),      # first 90 s at 1000 Hz
          labels=True,
      )

      len(ds)                                   # 371
      ds[0]["signal"].shape                     # (9, 90000)
      ds[0]["record_id"]                        # '001a'
      ds[0]["labels"]["patient_id"]             # 'patient001' — folds group on this
      ds[0]["labels"]["recording_type"]         # 'BR'  ('baseline room')
      ds[0]["labels"]["duration_seconds"]       # 300.0
      ds[0]["labels"]["age"], ds[0]["labels"]["sex"]   # '52', 'F'
      ds[0]["labels"]["prior_mi_location"]      # 'no'

      # 9 leads, precordials FIRST — so select by NAME, never by index.
      ds.config.lead_names   # ['V1','V2','V3','V4','V5','V6','I','II','III']
      ECGDataset("staffiii", split="train", data_path="...",
                 window=(0, 90_000), leads=["I", "II", "III"])[0]["signal"].shape
      # (3, 90000)   — by index these would be signal[6:9], not signal[0:3]

      # The event annotations are sample-accurate, so you can window straight
      # onto the occluded interval instead of guessing. labels_df is positional
      # and row-aligned with metadata_df, not indexed by record name:
      bi = ds.labels_df.query("recording_type == 'BI'")
      len(bi)                                   # 103 of 371 — the canonical task,
                                                # ischaemic (BI) vs everything else
      row = int(bi.index[0])                    # 5
      ds.metadata_df["record_name"].iloc[row]   # '002d'
      bi.iloc[0]["occluded_artery"]             # 'prox mid LAD'
      bi.iloc[0]["inflation_start_s"]           # '0.001'
      bi.iloc[0]["inflation_duration_s"]        # '124.999'

      ECGDataset("staffiii", split="train", data_path="...",
                 window=(1, 60_000))[row]["signal"].shape       # (9, 60000)

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/staffiii/1.0.0/" }
      - { label: "Martínez et al., CinC 2017", url: "https://doi.org/10.22489/CinC.2017.266-133" }
      - { label: "STAFF studies bibliography (ships with the release)", url: "https://physionet.org/content/staffiii/1.0.0/STAFF-Studies-bibliography-2016.pdf" }
---
