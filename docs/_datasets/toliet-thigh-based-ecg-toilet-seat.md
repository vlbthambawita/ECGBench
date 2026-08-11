---
slug: "toliet-thigh-based-ecg-toilet-seat"
name: "tOLIet (Thigh-based ECG, toilet seat)"
category: "one-lead"
order: 6
status: "completed"
source_url: "https://physionet.org/content/tollet/1.0.1/"
url_label: "physionet.org"
format: "1-lead (thigh, dry polymer electrodes) · 4 electrode textures per sitting · 14–197 s · 1,000 Hz · OpenSignals text"
patients: "86"
records: "580"
access: "open"
license: "CC BY 4.0"
origin_institution: "Centro Hospitalar Universitário de Lisboa Central (CHULC)"
origin_country: "Portugal — Lisbon"
leads: 1
paper_title: "Silva et al., Scientific Data 2026"
paper_doi: "https://doi.org/10.1038/s41597-026-06713-6"
search_keywords: "toilet tollet thigh ecg portugal lisbon dry electrode polymeric texture flat sinusoidal pyramidal trapezoidal wearable unobtrusive smart toilet seat bmi opensignals bitalino one-lead single-lead physionet"

sections:
  - type: description
    title: "Overview"
    body: |
      **An ECG taken through the backs of your thighs while you sit down.** The
      electrodes are dry polymer pads moulded into a toilet seat: no gel, no skin
      preparation, no operator placing anything, and no cooperation from the
      subject beyond sitting. 145 sittings by 86 volunteers at the Centro
      Hospitalar Universitário de Lisboa Central, published on PhysioNet in
      February 2026.

      **The release is a controlled comparison of electrode texture.** Four
      electrode pairs are embedded in the same seat, differing only in the surface
      moulded into them — flat, sinusoidal, pyramidal, trapezoidal — and all four
      record the same thigh-to-thigh derivation at the same instant. That is the
      scientific variable, and the answer it gives is blunt: see the texture table
      below.

      **One ECGBench record is one electrode channel, not one sitting.** 145 files
      × 4 electrodes = **580 records**, each a single-lead `(1, samples)` tensor
      named `<sitting>_<channel>` — `15_1_A2` is the sinusoidal channel of subject
      15's second sitting. The reason is in the next section, and it is not
      cosmetic.

      **238 of those 580 channels are electrodes that never made contact**, and
      they are exactly what separates `original` from `clean`. The default
      `version="clean"` gives you the 342 channels that recorded something.

      **Nothing here is a diagnosis dataset.** Four of the 145 sittings carry a
      free-text paroxysmal atrial fibrillation note and the other 141 carry
      nothing. The stated uses are electrode-texture comparison and biometric
      identification; for the latter the label is the subject, which is also the
      grouping column, so ECGBench's folds cannot serve it — same caveat as
      `ecg-id-database`.

  - type: description
    title: "Why one record per electrode, and not four leads per sitting"
    body: |
      The files look like four-lead records: each `.txt` holds columns `A1`–`A4`
      sampled together. Representing them that way is what ECGBench does
      everywhere else, and here it produces a dataset nobody can use.

      An electrode pair that made no contact reads a **constant ADC code** for the
      whole sitting. Kept as a four-lead record, that is a flat lead inside an
      otherwise good record — and `flat_line` rejects the whole record when any
      lead fails. Only **5 of the 145 sittings** have all four electrodes live, so
      a four-lead record set gives a `clean` version of 5 records and two empty
      folds. Split per channel and `clean` is 342 real single-lead ECGs.

      | Record model | `original` | `clean` | Usable? |
      |---|---|---|---|
      | one record per sitting, 4 leads | 145 | **5** | no — val and test folds are empty |
      | **one record per electrode, 1 lead** | **580** | **342** | yes |

      The cost is that **580 records are not 580 independent observations.** The
      four channels of a sitting are the same beats seen by four sensors, so the
      independent unit is the sitting (145) or the subject (86). Folds group by
      subject, so nothing leaks across a fold boundary — but a model reported as
      having been evaluated on 580 samples is overstating its evidence. Group on
      `source_record` before counting.

  - type: table
    title: "The four electrode textures, and how often each one worked"
    headers: ["Channel", "Texture", "Records", "Active", "Subjects with it active", "Median clipped fraction (active only)"]
    rows:
      - ["`A1`", "flat", "145", "**140** (97%)", "83", "0.0000"]
      - ["`A2`", "sinusoidal", "145", "**127** (88%)", "83", "0.0000"]
      - ["`A4`", "trapezoidal", "145", "**68** (47%)", "58", "0.0009"]
      - ["`A3`", "pyramidal", "145", "**7** (5%)", "7", "0.0396"]
      - ["**total**", "", "**580**", "**342** (59%)", "**86**", ""]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the shipped files, after
      verifying the local copy against the release's own `SHA256SUMS.txt` — **all
      174 listed files match**.

      **The release says 149 recordings and ships 145.** `DataSet.csv` lists 149
      IDs; `ECG_EXP/` holds 145 `.txt` files. `12_1`, `13_1`, `14_1` and `41_1` are
      tabulated and absent from the download, with no changelog in the release to
      explain it. ECGBench drops those four rows with a warning rather than
      emitting records that would all fail `corrupt_header`. The subject count is
      unaffected, because each of the four is a *later* sitting of a subject whose
      first sitting is present.

      | Figure | Published | Shipped | Cause |
      |---|---|---|---|
      | recordings | 149 | **145** | `12_1`, `13_1`, `14_1`, `41_1` are tabulated but absent |
      | subjects | 86 | **86** | — |
      | female / male | 50 / 36 | **50 / 36** | — |
      | mean age | 31.73 ± 13.11 | **31.73 ± 13.11** | — |
      | mean weight | 66.89 ± 10.70 kg | **66.89 ± 10.70 kg** | — |
      | mean height | 166.82 ± 6.07 cm | **166.83 ± 6.07 cm** | rounding only |
      | duration | "up to 5 minutes" | **14.4 – 197.2 s** | longest file is 3 min 17 s |

      **The published demographic means are per subject, not per recording** — that
      is how they reproduce to the second decimal. The per-sitting age mean is
      29.99 ± 10.59, so quoting "mean age 31.7" alongside a record count mixes two
      denominators.

      **`records: 580` on this page is ECGBench's record count, not the release's.**
      The release has 145 recordings; ECGBench exposes each of their four electrode
      channels separately, for the reason in the section above.

  - type: table
    title: "The cohort, recomputed"
    headers: ["", "Subjects", "Sittings", "Records", "Active records", "Age range", "Median age", "Median BMI"]
    rows:
      - ["female", "50", "85", "340", "192", "18 – 83", "27", "23.9"]
      - ["male", "36", "60", "240", "150", "19 – 82", "30", "24.1"]
      - ["**total**", "**86**", "**145**", "**580**", "**342**", "**18 – 83**", "**28**", "**23.9**"]

  - type: table
    title: "Sittings per subject"
    headers: ["Sittings for the subject", "Subjects", "Sittings", "Records"]
    rows:
      - ["1", "33", "33", "132"]
      - ["2", "47", "94", "376"]
      - ["3", "6", "18", "72"]
      - ["**total**", "**86**", "**145**", "**580**"]

  - type: description
    title: "`signal_active` is a floor, not a guarantee — always check `clipped_fraction`"
    body: |
      This is the trap on this dataset, and it survives validation.

      The front end is **±1.5 mV full scale into a 10-bit converter**, so every
      sample is inside the configured `amplitude_range_mv` **by construction** and
      `amplitude_outlier` cannot fire on a single record. What actually goes wrong
      is saturation *at* the rail — poor contact drives the amplifier into one end
      and keeps it there.

      A channel pinned at one rail has a tiny variance and `flat_line` catches it.
      A channel **oscillating between both rails** has a *large* variance, passes
      `flat_line`, and is not an ECG:

      | Record | Texture | Clipped fraction | Variance (mV²) | `signal_active` |
      |---|---|---|---|---|
      | `15_1_A4` | trapezoidal | **0.9997** | 0.000021 | ✓ passes |
      | `58_1_A4` | trapezoidal | **0.9968** | 0.028033 | ✓ passes |
      | `80_A4` | trapezoidal | **0.9962** | 0.002091 | ✓ passes |
      | `19_1_A4` | trapezoidal | **0.9898** | 0.032018 | ✓ passes |
      | `16_1_A4` | trapezoidal | **0.9586** | 0.093822 | ✓ passes |

      **12 of the 342 active records are at a rail for more than half their
      samples, and 130 touch one at all** — over 66 of the 145 sittings. Ten of the
      twelve worst are the trapezoidal electrode. No check in `CHECK_REGISTRY`
      measures clipping, so ECGBench does not exclude them; `clipped_fraction`,
      `min_mv` and `max_mv` are in the labels so you can:

      ```python
      usable = ds.labels_df[ds.labels_df["clipped_fraction"] < 0.01]
      ```

      The highest ADC code occurring anywhere in the release is 1022, not 1023.

  - type: description
    title: "Length varies fourteenfold, so pick a window that fits the shortest record"
    body: |
      Records run **14,400 to 197,250 samples** at 1 kHz — 14.4 s to 3 min 17 s,
      median 126.3 s — against the landing page's "up to 5 minutes per session".
      `expected_samples` is therefore deliberately empty in the config, which is
      the documented escape hatch for genuinely variable-length data.

      | Duration | Sittings |
      |---|---|
      | ≤ 30 s | 2 |
      | 30 – 60 s | 3 |
      | 60 – 120 s | 16 |
      | 120 – 180 s | 122 |
      | > 180 s | 2 |

      `window=(0, 14400)` is the largest fixed window that fits every record; one
      sample more raises `WindowOutOfRangeError` on record `79`. The window is
      pushed into the reader's `skiprows`/`nrows`, so on a 197-second record it
      decodes 14 seconds rather than decoding everything and slicing.

  - type: description
    title: "A new signal format, and a scale factor that is negative on purpose"
    body: |
      The signals ship as **PLUX/BITalino OpenSignals text exports**: three `#`
      preamble lines, the second a JSON blob naming all eleven columns and their
      bit depths, then tab-separated integers. ECGBench gained a `opensignals`
      reader for this release. The signal path names the column it wants:

      ```
      ECG_EXP/15_1.txt:A2
      ```

      because the same rows also carry a sequence number, four digital I/O columns
      and two 6-bit analog channels that are zero throughout.

      **The reader returns fractions of full scale, not millivolts**, and the
      config supplies `signal_unit_scale: -3.0`. That is not a unit conversion in
      the usual sense — it is the amplifier's full-scale span in millivolts, and it
      is **negative because the seat's differential front end inverts**. Together
      they reproduce the release's own `Script/read_ecg_data.py`

      ```python
      ((1024 - raw) / 1024 - 0.5) * (33 / 11)
      ```

      bit for bit; that equality was checked sample-for-sample against a raw
      `np.loadtxt` of the files rather than assumed. Code `0` is exactly `+1.5 mV`,
      which is why an unconnected electrode reads a flat **+1.5 mV** line rather
      than a zero line — and why `missing_leads` never sees one but `flat_line`
      does.

  - type: description
    title: "The 23 clinical reference ECGs are shipped but not loaded"
    body: |
      `ECG_REF/` holds 23 Sapphire-format `.XML` files — 10-second 12-lead resting
      ECGs from a hospital cardiograph, in microvolts — recorded alongside the seat
      recordings. They cover 23 of the 86 subjects (58, 59, 60, 67, and 68–86), and
      every one names a sitting that exists in `ECG_EXP/`.

      ECGBench does **not** load them and gives them no records. A 10-second
      clinical 12-lead ECG is a different modality from a two-minute thigh
      recording; putting both in one record set would mean folds containing both,
      and a `leads=` argument that means two different things depending on the row.
      `has_reference_ecg` (true for 92 of the 580 records — the four channels of
      each of the 23 sittings) and `reference_path` point at the files, and the
      release's own `Script/read_ref_data.py` parses them.

      This is the ground truth the abstract refers to, so if you are validating
      thigh-derived morphology against a clinical lead, those 23 sittings are the
      whole of the available evidence.

  - type: table
    title: "Validation summary (1000 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "580", "all records, with is_valid + quality_issues"]
      - ["clean", "342", "59.0% pass rate — 238 records excluded"]

  - type: description
    title: "One check fires, and it is the one that defines the dataset"
    body: |
      `flat_line` is the only check that fails anything: 238 records, one issue
      each. No record has a NaN sample, an unreadable header or a truncated signal,
      and `amplitude_outlier` cannot fire at all (see above). `missing_leads` finds
      nothing because a dead electrode reads +1.5 mV rather than 0.

      ECGBench does not invent a threshold for "did this electrode record
      anything": `ecgbench.labels.tollet` runs the project's own `check_flat_line`
      per channel and exposes the verdict as `signal_active`, so the label column
      and the validation report are the same decision and `clean` is exactly the
      `signal_active` records.

  - type: description
    title: "Ten folds, grouped on subject and balanced on sex × liveness"
    body: |
      Folds are built with `StratifiedGroupKFold`, grouped on `subject_id` and
      stratified on `stratify_class` — sex crossed with `signal_active`.

      | Class | Subjects | Records |
      |---|---|---|
      | `F_active` | 50 | 192 |
      | `M_active` | 36 | 150 |
      | `F_flat` | 47 | 148 |
      | `M_flat` | 34 | 90 |

      **Why the cross.** Liveness is in it because it decides how big a fold is in
      the version most people load. Measured over the shipped files at
      `random_state=42`:

      | Stratified on | `clean` records per fold | Female fraction per fold |
      |---|---|---|
      | sex only | 28 – 39 | 0.57 – 0.60 |
      | `signal_active` only | 31 – 37 | 0.33 – 0.93 |
      | **sex × `signal_active`** | **32 – 37** | **0.57 – 0.60** |

      **Electrode texture is deliberately not in the cross and does not need to
      be**: every sitting contributes all four textures, so any partition of
      sittings splits them evenly by construction — 13 to 15 active `A1` channels
      per fold. Putting it in explicitly would fail anyway, because `A3_active` has
      7 records from 7 subjects and a class needs at least ten subjects to appear
      in ten folds.

      | Fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
      |---|---|---|---|---|---|---|---|---|---|---|
      | records (original) | 60 | 60 | 56 | 60 | 56 | 56 | 56 | 60 | 56 | 60 |
      | records (clean) | 37 | 35 | 33 | 35 | 33 | 34 | 32 | 35 | 33 | 35 |
      | sittings | 15 | 15 | 14 | 15 | 14 | 14 | 14 | 15 | 14 | 15 |
      | subjects | 9 | 8 | 8 | 9 | 8 | 8 | 8 | 10 | 9 | 9 |

      No subject and no sitting spans a fold.

      **Use the folds, not the default split.** With folds 1–8 → train, 9 → val,
      10 → test:

      | Split | Records (original) | Records (clean) | Sittings | Subjects | female / male records |
      |---|---|---|---|---|---|
      | train | 464 | 274 | 116 | 68 | 272 / 192 |
      | val | 56 | 33 | 14 | 9 | 32 / 24 |
      | test | 60 | 35 | 15 | 9 | 36 / 24 |

      Nine subjects is not an evaluation set. For a real evaluation, cross-validate:
      `split=None` with `fold_numbers=[...]` selects by fold from `folds.csv` and
      ignores the default layout.

  - type: description
    title: "Overlap with other datasets in this catalogue: none"
    body: |
      No `related:` edge is declared. This is a 2020s Lisbon cohort recorded on
      purpose-built hardware that exists nowhere else in the catalogue, and no
      other release contains a thigh derivation or a dry-electrode seat recording.
      The nearest neighbour by *purpose* is `ecg-id-database`, which shares the
      biometric-identification framing and the "the label is the subject" problem,
      but not a single recording, subject or institution.

      The overlap that matters is **inside** this release, twice over: 53 of the 86
      subjects contributed more than one sitting, and every sitting contributes
      four records of the same beats. Both are handled by grouping folds on
      `subject_id`.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset tollet --data-path /path/to/tollet/1.0.1/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      ds = ECGDataset(
          "tollet",
          split="train",
          data_path="/path/to/tollet/1.0.1/",
          labels=True,
      )

      len(ds)                                     # 274  (clean: electrodes that recorded)
      ds[0]["signal"].shape                       # torch.Size([1, 35100])
      ds[0]["record_id"]                          # '10_A1'
      ds.lead_names                               # ('ECG',)  <- the electrode is a label,
                                                  # not a lead name: it varies per record
      ds[0]["labels"]["source_record"]            # '10'          <- the sitting
      ds[0]["labels"]["channel"]                  # 'A1'
      ds[0]["labels"]["electrode_texture"]        # 'flat'
      ds[0]["labels"]["subject_id"]               # '10'
      ds[0]["labels"]["session_index"]            # 0
      ds[0]["labels"]["sex"]                      # 'male'
      ds[0]["labels"]["age"]                      # 82
      ds[0]["labels"]["duration_secs"]            # 35.1
      ds[0]["labels"]["signal_active"]            # True
      ds[0]["labels"]["clipped_fraction"]         # 0.197066  <- CHECK THIS TOO: passing
                                                  # flat_line is not the same as ECG.
                                                  # 20% of this record is at a rail
      ds[0]["labels"]["has_reference_ecg"]        # False   <- only 23 sittings have one

      # Length varies 14.4-197.2 s, so a fixed window has to fit the SHORTEST
      # record. window= is pushed into the reader, so a 197 s record decodes 14 s.
      batched = ECGDataset("tollet", split="train", window=(0, 14400),
                           data_path="/path/to/tollet/1.0.1/")
      batched[0]["signal"].shape                  # torch.Size([1, 14400])

      # The source is 10-bit codes over a +/-1.5 mV span; signal_unit_scale = -3.0
      # converts and inverts. units="uV" gives the same samples x1000. (1.5 mV is
      # the converter rail, which this clipped record reaches.)
      uv = ECGDataset("tollet", split="train", window=(0, 14400), units="uV",
                      data_path="/path/to/tollet/1.0.1/")
      uv[0]["signal"].max()                       # tensor(1500.)  vs 1.5 mV

      # All 580 records, including the 238 electrodes that made no contact:
      everything = ECGDataset("tollet", split="train", version="original",
                              data_path="/path/to/tollet/1.0.1/")
      len(everything)                             # 464

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/tollet/1.0.1/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/411k-1476" }
      - { label: "Paper (Scientific Data, 2026)", url: "https://doi.org/10.1038/s41597-026-06713-6" }
---
