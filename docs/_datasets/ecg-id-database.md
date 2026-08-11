---
slug: "ecg-id-database"
name: "ECG-ID Database"
category: "one-lead"
order: 4
status: "completed"
source_url: "https://physionet.org/content/ecgiddb/1.0.0/"
url_label: "physionet.org"
format: "1-lead (Lead I, limb clamps) · stored twice: raw + filtered · 20 s · 500 Hz · WFDB"
patients: "90"
records: "310"
access: "open"
license: "ODC-By 1.0"
origin_institution: "Electrotechnical University \"LETI\""
origin_country: "Russia — St. Petersburg"
leads: 1
paper_title: "Lugovaya, MSc thesis, 2005"
paper_doi: "https://doi.org/10.13026/C2J01F"
search_keywords: "ecg id ecgiddb biometric identification person identity russia leti saint petersburg lugovaya lead i limb clamp raw filtered noise baseline drift one-lead single-lead physionet"

sections:
  - type: description
    title: "Overview"
    body: |
      **The reference dataset for ECG biometrics.** 310 twenty-second Lead I
      recordings from 90 volunteers, collected in 2004–2005 by Tatiana Lugovaya at
      the Electrotechnical University "LETI" in Saint Petersburg for a master's
      thesis that asked whether an electrocardiogram identifies the person who
      produced it. Her answer was 96% correct identification over these 90
      individuals.

      **The label is the person.** There is no diagnosis anywhere in this release,
      no clinical assessment and no rhythm annotation — the ground truth is which
      of the 90 subjects a recording came from, and the cohort is students,
      colleagues and friends of the author rather than patients. `subject_id` is
      therefore both the label and `patient_id_column`, which has a consequence
      spelled out in its own section below: **ECGBench's folds cannot be used for
      the identification task.**

      **Every record stores the same lead twice.** The two channels are `ECG I`
      (raw) and `ECG I filtered`, so the returned tensor is `(2, 10000)` and both
      rows are Lead I — one is the author's offline preprocessing of the other. The
      thesis deliberately switched off every filter in the cardiograph software, on
      the grounds that filtering might suppress features useful for identification,
      so the raw channel carries real baseline drift, 50 Hz interference and
      high-frequency noise. `leads=["ECG I"]` is how you avoid handing a model the
      same lead twice.

      **Length is uniform, unusually for a PhysioNet release.** All 310 records
      hold exactly 10,000 samples at 500 Hz — 20.000 s, no exceptions — so any
      `window=(start, length)` inside `(0, 10000)` fits every record and
      `WindowOutOfRangeError` cannot fire here.

      **The `.atr` annotations cover only the first half of each record.** Exactly
      10 R-peaks and 10 T-peaks per record, from an automatic detector the release
      states was never audited, all inside the first 5.1–11.7 s. Section below.

      **Twenty of the 90 subjects were recorded on more than one day**, up to six
      sessions spanning 156 days, which is what makes this database usable for
      studying whether an ECG biometric persists over time. The other 70 have all
      their records from a single session.

  - type: table
    title: "The cohort, recomputed from the 310 headers"
    headers: ["", "Subjects", "Records", "Age range", "Median age"]
    rows:
      - ["female", "46", "156", "13 – 68", "23"]
      - ["male", "44", "154", "16 – 75", "24"]
      - ["**total**", "**90**", "**310**", "**13 – 75**", "**23**"]

  - type: table
    title: "Records per subject — 1 to 22, not the documented 2 to 20"
    headers: ["Records for the subject", "Subjects", "Records"]
    rows:
      - ["1", "1", "1"]
      - ["2", "48", "96"]
      - ["3", "16", "48"]
      - ["4", "5", "20"]
      - ["5", "13", "65"]
      - ["6", "2", "12"]
      - ["7", "1", "7"]
      - ["8", "1", "8"]
      - ["11", "1", "11"]
      - ["20", "1", "20"]
      - ["22", "1", "22"]
      - ["**total**", "**90**", "**310**"]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the 310 headers and 310 `.atr`
      annotation files, after verifying the shipped copy against the release's own
      `SHA256SUMS.txt` — **all 977 listed data and documentation files match**. The
      22 listed files absent from a normal download are website assets under
      `.old/` (`biometric.shtml` and 21 PNGs), not data.

      **Three of the release's own headline figures reproduce exactly** and one
      does not. The README and the PhysioNet abstract give 310 records, 90 subjects,
      44 men and 46 women, ages 13 to 75 — all four confirmed. Both then say the
      number of records per subject runs "from 2 (collected during one day) to 20
      (collected periodically during 6 months)".

      | Figure | Documented | Shipped | Cause |
      |---|---|---|---|
      | records | 310 | **310** | — |
      | subjects | 90 | **90** | — |
      | men / women | 44 / 46 | **44 / 46** | — |
      | ages | 13 – 75 | **13 – 75** | — |
      | records per subject | 2 – 20 | **1 – 22** | `Person_74` has one record; `Person_02` has 22 |

      `RECORDS` and the files on disk agree with each other and disagree with the
      prose. There is no changelog in the release to explain it, and the directory
      timestamps show the 90 subject directories were written in two batches
      (September 2011 and April 2012) while the README was last touched in April
      2012 — so the likeliest reading is that the prose describes the collection
      protocol rather than the final release. ECGBench reports the shipped figures.

      **Age is a subject attribute here, not a per-record one.** No subject's
      recorded age changes across sessions, even the two spanning 140 and 156 days:
      `Person_02` is 23 in all six of its sessions. So a per-record age is safe to
      cross with sex for fold construction.

      **There are only ten acquisition days in the whole release, and one holds 43%
      of it.** `ecg_date` is a session identifier as much as a date, which is why
      the session columns are derived from it.

      | Date | Records | Subjects |
      |---|---|---|
      | 2004-12-07 | 5 | 2 |
      | 2004-12-28 | 14 | 2 |
      | 2005-03-15 | 26 | 9 |
      | 2005-04-05 | 16 | 4 |
      | 2005-04-26 | 37 | 16 |
      | **2005-05-12** | **134** | **52** |
      | 2005-05-13 | 21 | 10 |
      | 2005-05-20 | 17 | 7 |
      | 2005-05-21 | 26 | 12 |
      | 2005-05-24 | 14 | 10 |

      Any equipment, electrode or environmental peculiarity of 12 May 2005 is
      present in 134 of the 310 records.

  - type: description
    title: "The folds cannot be used for identification, and that is not a defect"
    body: |
      This is the most important thing on this page.

      ECG-ID's ground truth is identity, so its label column *is* its patient
      column. ECGBench groups folds by patient, which means every subject's records
      sit wholly inside one fold and **no fold's model has ever seen the person it
      would be asked to recognise**. Folds hold 7–11 subjects each and no subject
      spans a fold.

      That is the right default for every other use of these recordings. A model
      trained on 21 of `Person_02`'s 22 records and evaluated on the 22nd is
      measuring nothing; leaving the grouping off would produce exactly that, for
      the 89 subjects with more than one record. It is the wrong default for the
      one task this database was built for.

      **For identification, split within subject.** The label loader exposes the
      session structure for precisely this:

      ```python
      ds  = ECGDataset("ecgiddb", split="train", labels=True, data_path=...)
      ids = ds.metadata_df["record_id"]
      multi   = ds.labels_df["is_multi_session"]          # 20 subjects across the release
      enrol   = ids[multi & (ds.labels_df["session_index"] == 1)]
      verify  = ids[multi & (ds.labels_df["session_index"] >  1)]
      ```

      Holding out a subject's *later sessions* is the protocol the thesis's
      persistence experiment used, and it is the harder and more meaningful one:
      the electrodes were reattached from scratch for every recording, so a
      same-session pair shares an electrode placement that a cross-session pair
      does not.

      | | Subjects | Records |
      |---|---|---|
      | one session only | 70 | 175 |
      | two sessions | 11 | 47 |
      | three sessions | 7 | 24 |
      | five sessions | 1 | 20 |
      | six sessions | 1 | 22 |
      | **multi-session** | **20** | **135** |

      The longest-running subjects are `Person_02` (22 records, 6 sessions, 156
      days) and `Person_01` (20 records, 5 sessions, 140 days). Together they are
      42 of the 310 records.

      **The thesis's own 195/115 train/test division is not recoverable.** It
      divided these same 310 records into 195 training and 115 test records,
      chosen for "maximum difference between records in different sets both in
      monitoring time and human physical state" — and that assignment appears in no
      file. Not in `RECORDS`, not in a header comment, not in a separate list. So
      `has_predefined_splits` is `false` for lack of the data, not for lack of a
      split, and no ECGBench figure is comparable to the thesis's 96%.

  - type: description
    title: "One lead, stored twice — raw and filtered"
    body: |
      Both channels of every record are Lead I, the left-hand minus right-hand
      potential, taken with **limb clamp electrodes** with the subject seated. The
      thesis chose Lead I because it is easy to acquire and insensitive to small
      electrode displacements, imitating "the likely scenarios of user interaction
      with a practical identification system".

      | Channel | Name in the header | Contents |
      |---|---|---|
      | 0 | `ECG I` | raw — **every cardiograph filter deliberately off** |
      | 1 | `ECG I filtered` | the thesis's own offline preprocessing of channel 0 |

      The filter chain, from the thesis: baseline drift removed by level-9
      `db8` wavelet decomposition with the final approximation subtracted, an
      adaptive bandstop at 50 Hz for power-line noise, and a 5th-order Butterworth
      lowpass (Wp 40 Hz, Ws 60 Hz, Rp 0.1 dB, Rs 30 dB).

      **It is zero-phase**, verified rather than assumed: peak cross-correlation
      between the two channels is at lag 0 in 273 of the 310 records and at ±1
      sample in the other 37. So the channels are sample-aligned and the `.atr`
      annotation indices apply to both.

      Because the residual `raw − filtered` is therefore noise rather than a phase
      difference, it is a usable per-record noise level.
      `ecgbench.labels.ecgiddb.scan_noise_levels(data_path)` computes it — it is a
      separate call rather than a label because it needs the waveforms decoded:

      | Quantity | Median | 90th pct | 99th pct | Max |
      |---|---|---|---|---|
      | `removed_rms_mv` | 0.187 | 0.562 | 2.214 | **41.948** (`Person_76/rec_2`) |

      Correlation between the two channels runs 0.103 to 0.978 (median 0.691). A
      low value means drift dominates the raw channel, not that the filter failed.

      `config.leads` is **2** because that is the shape of the returned tensor;
      this page's `leads` field says **1** because that is the number of electrode
      pairs. Both are true of different things, and a test pins the pair so neither
      gets "corrected" to match the other.

  - type: description
    title: "The annotations stop about 40% of the way in, and nobody audited them"
    body: |
      The shipped `ANNOTATORS` file is one line: *"atr — unaudited R- and T-wave
      peaks annotations from an automated detector"*. Taking it at its word:

      - **Exactly 20 annotations in every one of the 310 files** — 10 `N` R-peaks
        and 10 `t` T-peaks, strictly alternating `N t N t …`. No record has more,
        none fewer, and no other symbol appears anywhere in the release.
      - **They describe the first ten beats and nothing after them.** The last
        annotation lands at sample **2,542 to 5,869 of 10,000** (mean 4,002, median
        3,970) — 5.1 s to 11.7 s into a 20.000 s record. Only 18 records have any
        annotation past the halfway mark, and 8.3 s to 14.9 s of every record is
        unannotated.
      - `annotated_fraction` carries this per record, 25.4% to 58.7%. A window past
        `last_annotation_sample` contains beats nobody marked.

      `N` here means "the detector found a beat", **not** "normal beat": no beat in
      this database was ever classified. So `mean_hr_bpm` (50.0 – 132.7, median
      76.7), `sdnn_ms` (4.1 – 393.5) and `mean_rt_interval_ms` (169.8 – 452.2) are
      estimates over nine RR intervals from an unaudited detector. They describe the
      record; they are not an HRV result, and this is not a beat-detection
      reference — use `qtdb` or `ludb` for that. `mean_rt_interval_ms` is R-peak to
      T-**peak**, not a QT interval, because the annotation marks the T peak rather
      than its offset.

      The wide SDNN range is real and expected: the thesis deliberately did not
      restrict heart rate or physical and emotional state, so within-record rate
      variation is part of the design rather than a data-quality problem.

  - type: table
    title: "Validation summary (500 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "310", "all records, with is_valid + quality_issues"]
      - ["clean", "307", "99.0% pass rate — 3 records excluded"]

  - type: description
    title: "Three records fail on amplitude, and the 12-bit rail is not a rail here"
    body: |
      No record has a NaN sample, a flat or all-zero channel, an unreadable header
      or a truncated signal. The only check that fires is `amplitude_outlier`, on
      three records:

      | Record | Raw channel (mV) | Filtered channel (mV) |
      |---|---|---|
      | `Person_76/rec_2` | **−154.155** to 10.745 | **−163.800** to **151.375** |
      | `Person_88/rec_1` | **−153.635** to 10.545 | −48.945 to 16.245 |
      | `Person_47/rec_2` | −1.910 to **10.885** | −5.035 to 3.835 |

      **The bound is `[-10.24, 10.235]` mV, and it is a physiological bound that
      merely looks like a hardware rail.** Every one of the 620 signal lines in the
      release reads `16 200 12 0` — format 16, gain 200 adu/mV, `adc_res` 12,
      `adc_zero` 0, no baseline — which *nominally* confines every sample to
      [−2048, 2047] adu. The stored int16 samples go far past it: `Person_76/rec_2`
      reaches −30,831 adu. The 12 bits describe the cardiograph's converter, and
      the baseline drift the thesis chose not to filter rides on top of it. So
      unlike `apnea_ecg`, which uses the same two numbers as a rail its data sits
      inside, here the data is not rail-bounded at all.

      No float32 slack is needed, and that was checked rather than assumed: the
      closest *passing* record reaches 10.115 mV and −3.060 mV, nowhere near the
      bound, so the rail-rounding trap that bit `chfdb` cannot arise. Widening the
      bound would only readmit two records whose drift is fifteen times the
      physiological range.

      `Person_47/rec_2` is a different case from the other two — a genuine 10.885 mV
      excursion rather than gross drift — and it is excluded on the same rule.
      `version="original"` returns all 310 with `quality_issues` attached.

  - type: description
    title: "Ten folds over 310 records, grouped on subject and balanced on sex x age"
    body: |
      Folds are built with `StratifiedGroupKFold`, grouped on `subject_id` and
      stratified on `stratify_class` — sex crossed with a single age cut at 30
      years.

      | Class | Subjects | Records |
      |---|---|---|
      | `female_le30` | 36 | 124 |
      | `male_le30` | 27 | 99 |
      | `male_gt30` | 17 | 55 |
      | `female_gt30` | **10** | 32 |

      **Why that cross and nothing finer.** A class has to contain at least as many
      subjects as there are folds to appear in every fold, and the smallest cell
      here holds exactly 10 — the floor for 10 folds. Crossing sex with four age
      bands instead produces 5-subject cells and leaves empty cells in the fold
      table. Neither axis works alone either: stratifying on **sex** alone leaves
      four folds with no subject over 45 (age is skewed — 45 of the 90 subjects are
      21–30 and only 11 are over 45), and stratifying on **age bands** alone leaves
      one fold at 9 female / 19 male records, because group sizes run 1 to 22 and a
      single subject can swing a fold.

      Heart rate was rejected as an axis: it is not a subject attribute here, and
      it comes from an unaudited detector over ten beats. `stratify_class` is a
      fold-construction device and must not be trained on as one.

      **Use the folds, not the default split.** With folds 1–8 → train, 9 → val,
      10 → test:

      | Split | Records | Subjects | female / male | Age range | Multi-session records |
      |---|---|---|---|---|---|
      | train | 246 | 76 | 119 / 127 | 13 – 75 | 97 |
      | val | 27 | 7 | 13 / 14 | 16 – 55 | 16 |
      | test | 37 | 7 | 24 / 13 | 19 – 68 | 22 |

      Seven subjects is not an evaluation set. For a real evaluation,
      cross-validate: `split=None` with `fold_numbers=[...]` selects by fold from
      `folds.csv` and ignores the default layout.

      | Fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
      |---|---|---|---|---|---|---|---|---|---|---|
      | records (original) | 30 | 28 | 29 | 39 | 33 | 30 | 27 | 30 | 27 | 37 |
      | subjects | 9 | 9 | 9 | 9 | 11 | 10 | 9 | 10 | 7 | 7 |

      Folds 4 and 10 are the largest because they drew `Person_01` (20 records) and
      `Person_02` (22). Sex is balanced 13–24 female against 13–25 male per fold;
      those two subjects are what pushes the extremes.

  - type: description
    title: "Overlap with other datasets in this catalogue: none"
    body: |
      No `related:` edge is declared. Two other releases in this catalogue come
      from Russia — `ludb` (Lobachevsky University, Nizhny Novgorod) and `incartdb`
      (St. Petersburg Institute of Cardiological Technics) — and neither shares a
      recording, an institution or a decade with this one: ECG-ID is a 2004–2005
      LETI student cohort with one lead at 20 s, `ludb` is 200 twelve-lead
      hospital records at 10 s, `incartdb` is 75 twelve-lead 30-minute Holter
      excerpts. Nothing else in the catalogue is a biometric-identification release.

      The overlap that matters is **inside** this release — 89 of the 90 subjects
      contribute more than one record — and it is handled by the subject grouping.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset ecgiddb --data-path /path/to/ecgiddb/1.0.0/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      ds = ECGDataset(
          "ecgiddb",
          split="train",
          data_path="/path/to/ecgiddb/1.0.0/",
          labels=True,
      )

      len(ds)                                    # 243
      ds[0]["signal"].shape                      # torch.Size([2, 10000])
      ds[0]["record_id"]                         # 'Person_01_rec_1'
      ds.lead_names                              # ('ECG I', 'ECG I filtered')
                                                 # ONE lead, stored twice
      ds[0]["labels"]["subject_id"]              # 'Person_01'  <- the label
      ds[0]["labels"]["record_name"]             # 'rec_1'  <- NOT unique: 90 records
                                                 # in the release are called rec_1
      ds[0]["labels"]["age"]                     # 25.0
      ds[0]["labels"]["sex"]                     # 'male'
      ds[0]["labels"]["ecg_date"]                # '2004-12-07'
      ds[0]["labels"]["session_index"]           # 1  (of 5, spanning 140 days)
      ds[0]["labels"]["is_multi_session"]        # True
      ds[0]["labels"]["mean_hr_bpm"]             # 68.5
      ds[0]["labels"]["annotated_fraction"]      # 0.4407  <- annotations stop at
                                                 # sample 4,407 of 10,000

      # Both channels are Lead I. Select one by name so a model is not handed the
      # same lead twice:
      raw = ECGDataset("ecgiddb", split="train", leads=["ECG I"],
                       data_path="/path/to/ecgiddb/1.0.0/")
      raw[0]["signal"].shape                     # torch.Size([1, 10000])

      # Length is uniform (10,000 samples), so any window fits every record.
      # window= is pushed into wfdb's sampfrom/sampto, so the rest is never decoded.
      first5s = ECGDataset("ecgiddb", split="train", window=(0, 2500),
                           data_path="/path/to/ecgiddb/1.0.0/")
      first5s[0]["signal"].shape                 # torch.Size([2, 2500])

      # For the identification task, re-split WITHIN subject — the folds group by
      # subject, so no fold's model has seen the person it would have to recognise.

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/ecgiddb/1.0.0/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/C2J01F" }
      - { label: "Lugovaya's thesis summary (shipped with the data)", url: "https://physionet.org/content/ecgiddb/1.0.0/biometric.shtml" }
---
