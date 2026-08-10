---
slug: "qt-database-qtdb"
name: "QT Database (QTDB)"
category: "two-lead"
order: 9
status: "completed"
source_url: "https://physionet.org/content/qtdb/1.0.0/"
url_label: "physionet.org"
format: "2-lead · 20 lead layouts · 15 min · 250 Hz · WFDB"
patients: "103"
records: "105"
access: "open"
license: "ODC-By 1.0"
origin_institution: "Universidad de Zaragoza; MIT; Beth Israel Deaconess Medical Center"
origin_country: "Spain / USA"
leads: 2
paper_title: "Laguna et al., Comput Cardiol 1997"
paper_doi: "https://doi.org/10.13026/C24K53"
search_keywords: "qt database qtdb waveform boundary delineation p qrs t u wave onset peak end fiducial point annotation inter-observer variability ecgpuwave aristotle laguna moody zaragoza mit physionet benchmark qt interval qtc bazett two-lead excerpt"

related:
  # The edb <-> qtdb edge is declared on the European ST-T page (has_derivative),
  # and catalogue.py derives this side of it. Declaring it here as well would
  # double-count the overlap on the website and fail tests/test_catalogue.py.
  - slug: "mit-bih-arrhythmia-database"
    relation: "derived_from"
    shares_records: true
    verified: true
    note: >
      15 of these 105 records are 15-minute excerpts of 15 of MIT-BIH Arrhythmia's
      48 recordings, resampled from 360 Hz to 250 Hz by `xform`. Each header says so
      — `sel100` carries "Produced by xform from record 100, beginning at 7:00.000"
      — and the name encodes the parent. Verified: the excerpts correlate above 0.96
      with the source at the stated offset after resampling, the clinical header
      blocks are byte-identical to mitdb's (including the analog-tape and recorder
      fields), and the lead names agree record for record. The 15 are 100, 102, 103,
      104, 114, 116, 117, 123, 213, 221, 223, 230, 231, 232 and 233 — 3.75 h of
      mitdb's 24.1 h, and 31% of its records. **Do not train a beat classifier on
      MIT-BIH Arrhythmia and evaluate delineation on QTDB's `sel1*`/`sel2*` records,
      or the reverse.** Two further traps: `sel100`, `sel102`, `sel103` and `sel104`
      declare an explicit baseline of 0 against an `adc_zero` of 1024, so QTDB
      returns them shifted +5.12 mV relative to mitdb's copy of the same samples;
      and QTDB inherits no analog-tape field for its other 90 records, so the tape
      numbers cannot be used to check for shared subjects across source databases.
  - slug: "sudden-cardiac-death-holter-database"
    relation: "derived_from"
    shares_records: true
    verified: true
    note: >
      23 of these 105 records are 15-minute excerpts of **all 23** of the Sudden
      Cardiac Death Holter Database's recordings, one per subject — `sel30` from
      sddb record 30, through `sel52`. Both are already 250 Hz, so the samples are
      directly comparable, and 22 of the 23 reproduce sddb exactly: `sel39` and
      `sel47` are bit-identical, and the other 20 are `trunc(sddb_digital / 4)`,
      because QTDB re-declared the gain from 800 to 200 adu/mV and shifted the
      samples two bits to keep the millivolt scale — an exact relation, verified
      over 4,096 samples per record, not an approximation. It costs two bits of
      amplitude resolution and it changes the physical amplitude for `sel51` and
      `sel52`, whose gains QTDB re-estimated to 600 and 400. **`sel32` is the
      exception and is unresolved**: its opening 4,096 samples occur nowhere in
      sddb record 32, in either channel, raw or divided by four, although the paper
      and the header both place it at 20:52:20 of that record. **Do not evaluate on
      QTDB's `sel3*`–`sel5*` records after training on sddb, or the reverse** — that
      is 5.75 h of sddb's 446.6 h, but it covers every one of its subjects. Note
      also that QTDB's copies carry no `.atr`, exactly as sddb's audited layer is
      missing for 11 records, and that signal 0 was delayed by 7 samples in QTDB's
      copies, which is why they are 224,993 samples rather than 225,000.
  - slug: "mit-bih-supraventricular-arrhythmia-database"
    relation: "derived_from"
    shares_records: true
    verified: true
    note: >
      13 of these 105 records are 15-minute excerpts of 13 of the MIT-BIH
      Supraventricular Arrhythmia Database's 78 recordings, resampled from 128 Hz to
      250 Hz: 803, 808, 811, 820, 821, 840, 847, 853, 871, 872, 873, 883 and 891.
      The provenance is in every header and the excerpts correlate 0.74-0.95 with
      the source after resampling — lower than the MIT-BIH Arrhythmia group only
      because upsampling from 128 Hz cannot be inverted exactly, not because the
      match is in doubt. That is 3.25 h of svdb's 78 h and 17% of its records.
      **Do not train on one and evaluate on the other.** Six of the 13 — 820, 821,
      847, 853, 883, 891 — declare a gain of 0 in QTDB, so their amplitudes are on
      wfdb's nominal 200 adu/mV fallback rather than a measured scale; svdb's own
      copies declare 10-bit resolution where QTDB's declare 12.
  - slug: "mit-bih-normal-sinus-rhythm-database"
    relation: "derived_from"
    shares_records: true
    verified: true
    note: >
      10 of these 105 records are 15-minute excerpts of 10 of the MIT-BIH Normal
      Sinus Rhythm Database's 18 recordings, resampled from 128 Hz to 250 Hz:
      16265, 16272, 16273, 16420, 16483, 16539, 16773, 16786, 16795 and 17453. The
      excerpts correlate 0.70-0.92 with the source after resampling; the residual is
      interpolation, not disagreement. **56% of nsrdb's records appear here** — the
      highest proportion of any source — so a normal-versus-abnormal classifier
      trained on nsrdb has seen most of these signals. Three QTDB headers in this
      group name their source record `mqt2`, an intermediate file, rather than the
      nsrdb record; the record name carries the real id. `sel17152`, which the
      paper lists with this group's Holters, is **not** in nsrdb: it is one of the
      age- and gender-matched controls collected alongside the sudden-death
      recordings and is published nowhere else.
  - slug: "mit-bih-st-change-database"
    relation: "derived_from"
    shares_records: true
    verified: true
    note: >
      6 of these 105 records are 15-minute excerpts of 6 of the MIT-BIH ST Change
      Database's 28 recordings, resampled from 360 Hz to 250 Hz: 301, 302, 306, 307,
      308 and 310. Provenance is in each header and the excerpts correlate above
      0.94 with the source at the stated offset. That is 1.5 h and 21% of stdb's
      records. **Do not train on one and evaluate on the other.** Neither release
      identifies its subjects, so if two of these six came from the same person
      nothing in either dataset can tell — QTDB's patient grouping treats them as
      six.

sections:
  - type: description
    title: "Overview"
    body: |
      **The standard reference for ECG waveform delineation, and the only dataset in
      this catalogue whose ground truth is fiducial points rather than labels.** 105
      fifteen-minute two-channel excerpts at 250 Hz in which expert cardiologists
      marked, beat by beat, the **onset, peak and end of the P wave**, the **onset
      and end of the QRS complex**, the **peak and end of the T wave** and — where
      present — the **peak and end of the U wave**. **3,623 beats** were annotated
      this way, on a graphic workstation showing both channels at once. It was built
      at the Universidad de Zaragoza and MIT and published in 1999 because no such
      reference existed: QT-measurement algorithms had nothing to be scored against.

      **Every record is an excerpt of another database's recording, and six of the
      seven sources are already in this catalogue.** European ST-T 33, sudden-death
      Holters 23, MIT-BIH Arrhythmia 15, Supraventricular 13, Normal Sinus Rhythm
      10, ST Change 6, Long-Term 4, plus one matched control published nowhere else.
      **100 of the 105 records share signal samples with another ECGBench dataset**,
      verified from the waveforms rather than inferred from the names. This is the
      first thing to know and it has its own section below.

      **The ground truth is in the last five minutes and nowhere else.** Annotation
      began only after the first 10 minutes, deliberately, to leave an algorithm a
      learning period. Measured across the release, the earliest manual annotation
      sits at **600.464 s** and the latest at **896.916 s**. Read
      `window=(150000, 74993)` — exactly the annotated region, and it fits all 105
      records. A window from sample 0 contains no ground truth at all.

      **The boundaries are not in the fold CSVs and cannot be.** They are 3,623 rows
      of up to eleven sample indices each, not a record-level column. Call
      `ecgbench.labels.qtdb.load_beat_annotations(data_path)` for them; `labels=True`
      gives the per-record summary that stratifies the folds.

      **There are 20 lead layouts and 57 records decline to name their channels.**
      `ECG1`/`ECG2` are channel positions, not leads. Worse for cross-dataset work:
      the 33 European ST-T records use the ESC's original electrode nomenclature
      (`D3`, `CM5`, `CC5`, `ML5`), and only **2 of the 33** agree with the names
      `edb` gives the very same channels.

      **Amplitude is unreliable for 34 of the 105 records** — 24 sudden-death
      Holters whose gains the paper calls estimates, and 10 more that declare a gain
      of `0`, which wfdb silently replaces with 200 adu/mV. Intervals, which is what
      this database is for, are unaffected. Four records additionally carry a
      constant **+5.12 mV** pedestal.

      **All 105 records pass every validation check**, so `clean/` equals
      `original/`. No NaN anywhere, no flat channel, no amplitude outlier — the
      excerpts were selected for signal quality, which is exactly the selection bias
      described below.

  - type: table
    title: "The seven sources, recomputed from the files"
    headers: ["Source", "Recs", "Subj", "Source Hz", "Beats", "Pub. beats", "QT-measurable", "P wave", "U wave", "`.atr`", "2nd annot.", "Calibrated", "Median QT ms", "Median HR bpm", "Clinical"]
    rows:
      - ["European ST-T (`edb`)", "33", "31", "250", "1,041", "1,041", "1,041", "1,041", "332", "33", "0", "33", "412.0", "63.3", "33"]
      - ["Sudden death (`sddb`)", "23", "23", "250", "714", "714", "633", "562", "122", "**0**", "0", "**0**", "440.0", "66.1", "**0**"]
      - ["MIT-BIH Arrhythmia (`mitdb`)", "15", "15", "360", "**674**", "673", "674", "469", "186", "15", "**11**", "15", "420.0", "72.8", "15"]
      - ["Supraventricular (`svdb`)", "13", "13", "128", "517", "517", "517", "477", "30", "13", "0", "7", "400.0", "65.8", "**0**"]
      - ["Normal sinus (`nsrdb`)", "10", "10", "128", "300", "300", "300", "300", "50", "10", "0", "10", "403.0", "67.9", "**0**"]
      - ["ST Change (`stdb`)", "6", "6", "360", "206", "206", "206", "176", "43", "6", "0", "6", "392.0", "73.1", "**0**"]
      - ["Long-Term (`ltdb`, not in catalogue)", "4", "4", "128", "141", "141", "141", "139", "58", "4", "0", "**0**", "413.0", "70.2", "**0**"]
      - ["BIH matched control (unpublished)", "1", "1", "128", "30", "30", "30", "30", "0", "1", "0", "**0**", "320.0", "**109.5**", "**0**"]
      - ["**total**", "**105**", "**103**", "250 · 360 · 128", "**3,623**", "3,622", "**3,542**", "**3,194**", "**821**", "**82**", "**11**", "**71**", "412.0", "69.1", "**48**"]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the 105 headers and all nine
      annotation layers, after verifying the shipped data against the release's own
      `SHA256SUMS.txt` — **all 1,132 files match**. That total includes 57
      superseded `.hea-` headers, which describe the channels as "record N, signal
      0" instead of `ECG1`/`ECG2`, and 105 `.xws` WAVE display-settings files. The
      record list comes from the shipped `RECORDS` file, so neither can enter the
      partition.

      **3,623 annotated beats, against the paper's 3,622.** Table 2 of Laguna et al.
      gives a per-record count that sums to 3,622, and 104 of the 105 records match
      it exactly. `sel223` carries **31** beats where the table says 30. The paper's
      column ships as `ecgbench.labels.qtdb.PUBLISHED_ANNOTATED_BEATS` and
      `annotated_beats_matches_published` is the per-record check, so this is
      reproducible rather than a claim. The count is of beat annotations in `.q1c`,
      the audited second pass, which is also what `.man` holds (3,593 `N` plus 30
      `A`); the audit reclassified 65 of those `N` beats as `B`, `V` or `Q` without
      changing the total.

      **3,542 of the 3,623 beats have a measurable QT** — a QRS onset and a T end.
      The 81 that do not are almost all in `sel35` and `sel37`, the two records
      whose annotators marked **QRS boundaries only** and no T wave at all; their
      median QT is therefore NaN rather than zero. P waves are annotated in 3,194
      beats and are absent from nine records (fully absent from seven). U waves are
      annotated in 821 beats across 23 records, and a T *onset* — the mark most
      often skipped — in 1,412 beats across 44 records.

      **The waveform-pattern column reproduces for 101 of 105 records.**
      `waveform_pattern` recomputes Table 2's notation (`(p)(N)t)` and so on) from
      the annotations, counting a mark as part of the pattern when at least half the
      record's beats carry it. The four disagreements are the paper's own column
      being inconsistent, not a parse failure: `sel117` and `sel14157` are listed
      with `u)` on the strength of 11 and 8 U waves in 30 beats, `sele0704` is
      listed *without* a T onset it carries in 20 of 30, and `sel37`'s beats are
      N:24, B:20, Q:6 so its modal symbol is `N` where Table 2 writes `(Q)`.

      **103 subjects is an upper bound, not a count.** Almost every record is a
      different source recording, so the record name is a subject id for 101 of
      them. Two European ST-T subjects contributed two recordings each and both
      pairs are here: `sele0121` with `sele0122` (one 51-year-old man) and
      `sele0124` with `sele0126`. Those come from running
      `ecgbench.labels.edb.reconstruct_patient_ids` over `edb` 1.0.0 and
      intersecting its seven multi-record subjects with QTDB's 33; they are a
      literal in `EDB_SHARED_SUBJECTS` because QTDB's own header text is too coarse
      to recover the second pair. The 13 Supraventricular and 6 ST Change records
      carry no subject information in any release, so a shared person among them is
      undetectable.

      **The paper describes annotation files that do not ship.** It says each record
      has a `.ari` holding ARISTOTLE's automatic QRS annotations. There is not one
      `.ari` file in the release and the shipped `ANNOTATORS` file does not list the
      extension. What does ship, per record: `.man` (beat locations), `.qt1`/`.q1c`
      (annotator 1, unaudited and audited), `.qt2`/`.q2c` for 11 records,
      `.pu`/`.pu0`/`.pu1` (ecgpuwave over the whole record, unaudited) and `.atr`
      for 82 records — the source database's own reference annotations, inherited.

  - type: description
    title: "100 of the 105 records are in another ECGBench dataset"
    body: |
      This is the single most consequential fact about the QT Database and it is not
      a caveat about metadata: **the signal samples are the same samples.** Six of
      the seven sources are datasets ECGBench also partitions, and the fold CSVs
      here are disjoint *within* QTDB only.

      | Source | Recs | Of source's | Verification |
      |---|---|---|---|
      | European ST-T (`edb`) | 33 | 37% | 30 **bit-identical** at the stated offset; 3 identical up to a DC pedestal and a matched gain rescale (r ≥ 0.99985) |
      | Sudden death (`sddb`) | 23 | **100%** | 22 exact — 2 bit-identical, 20 as `trunc(sddb/4)`; `sel32` **not located** |
      | MIT-BIH Arrhythmia (`mitdb`) | 15 | 31% | resampled 360→250 Hz, r > 0.96; header clinical blocks byte-identical |
      | Supraventricular (`svdb`) | 13 | 17% | resampled 128→250 Hz, r 0.74–0.95 |
      | Normal sinus (`nsrdb`) | 10 | **56%** | resampled 128→250 Hz, r 0.70–0.92 |
      | ST Change (`stdb`) | 6 | 21% | resampled 360→250 Hz, r > 0.94 |
      | Long-Term (`ltdb`) | 4 | — | not in this catalogue |
      | BIH matched control | 1 | — | published nowhere else |

      Where a source is already 250 Hz the comparison is exact and the result is
      unambiguous. **30 of the 33 European ST-T excerpts are bit-identical** to `edb`
      1.0.0 at the offset their own header states; the remaining three
      (`sele0112`, `sele0116`, `sele0136`) are the same waveform with a large DC
      offset added and one channel scaled by exactly the gain ratio they re-declare
      — the fitted slope is 0.925, 0.677 and 0.602 against declared gains of 185,
      135 and 120 versus `edb`'s 200. For the sudden-death group, `sel39` and
      `sel47` are bit-identical and the other 20 satisfy
      `qtdb_digital == trunc(sddb_digital / 4)` exactly, because QTDB re-declared
      the gain from 800 to 200 adu/mV; two bits of amplitude resolution are lost and
      the physical scale is preserved to within one quantisation step of 0.00375 mV.

      **`sel32` is the one record whose stated provenance does not hold.** Both the
      paper and the header place it at 20:52:20 of `sddb` record 32, but its opening
      4,096 samples occur nowhere in that 24.3-hour record — searched in both
      channels, raw and divided by four. `source_record_verified` is `False` for it
      and `True` for the other 22 sudden-death records and all 33 European ST-T
      records; it is `NA` for the resampled sources, which cannot match sample-wise
      by construction.

      **What to do about it.** Filter on `source_database` before combining QTDB with
      anything. If you are evaluating delineation, QTDB is the reference and the
      other six should be treated as its training set, not as independent test data.
      If you are evaluating beat classification or ischaemia detection on one of the
      six, exclude the records QTDB draws from — `source_record` in the labels names
      each parent — or accept that the QT Database is contaminated with respect to
      your model.

  - type: description
    title: "The ground truth: 3,623 beats, in the last five minutes"
    body: |
      `load_labels` returns one row per record; the boundaries themselves come from
      a second call, because they do not fit a record-level table:

      ```python
      from ecgbench.labels.qtdb import load_beat_annotations

      beats = load_beat_annotations("/path/to/qtdb/1.0.0/")
      len(beats)                       # 3623
      beats.columns
      # record_name beat_index symbol morphology_group
      # p_onset p_peak p_offset  qrs_onset qrs_peak qrs_offset
      # t_onset t_peak t_offset  u_onset u_peak u_offset
      # qrs_ms p_ms pr_ms qt_ms rr_ms qtc_bazett_ms
      ```

      Sample indices are in the record's own 250 Hz frame, so a boundary at 152,000
      is at 608.0 s. `NaN` means the annotator did not mark that point, which is
      information rather than missing data — see the pattern column. Marks are
      assigned to beats by position (P marks and the QRS onset to the following
      beat, everything from the QRS offset onward to the preceding one), which is
      unambiguous for this annotation style and needs no tolerance parameter.

      `rr_ms` and therefore Bazett's `qtc_bazett_ms` are defined only where the
      preceding annotated beat is the preceding actual beat. Records were annotated
      in runs of 30 consecutive beats plus up to 20 of each non-dominant morphology,
      so runs are separated by gaps of arbitrary length; anything over 3 s is
      treated as a gap rather than as a pause.

      Recomputed over the release: median QT **412 ms** per record (304–764 across
      records), median Bazett QTc **434.7 ms**, median heart rate **69.1 bpm**
      (35.5–128.2). The low central heart rate is the selection bias the authors
      warn about: excerpts were chosen to avoid noise, and "heart rates during these
      excerpts tend to be relatively low, probably since higher rates are frequently
      associated with noisy signals that would have failed to satisfy our selection
      criteria". **A delineator validated only here has not been tested at
      tachycardia, on baseline wander, or on ectopic beats** — only beats ARISTOTLE
      called normal, with normal neighbours, were eligible for annotation.

      `.pu`, `.pu0` and `.pu1` carry `ecgpuwave`'s automatic boundaries for **every**
      beat of every record: 222,319 beats from both signals, 111,031 from signal 0
      and 111,288 from signal 1. They are unaudited and are the baseline the paper's
      own method produced, not ground truth. Their `num` field on a `t` annotation
      does carry something the manual layer lacks — T-wave morphology — which
      `dominant_t_morphology` summarises: normal in 67 records, biphasic
      negative-positive in 18, biphasic positive-negative in 11, inverted in 8 and
      only-upwards in 1.

  - type: table
    title: "The 11 records with two annotators — and why parity cannot be assumed"
    headers: ["Record", "Annotator 1 beats", "Annotator 2 beats", "Median QT ms", "Fold"]
    rows:
      - ["sel100", "30", "30", "398.0", "4"]
      - ["sel102", "**85**", "**3**", "468.0", "8"]
      - ["sel103", "30", "30", "408.0", "3"]
      - ["sel114", "50", "50", "454.0", "9 (val)"]
      - ["sel116", "50", "50", "368.0", "10 (test)"]
      - ["sel117", "30", "30", "448.0", "2"]
      - ["sel123", "30", "30", "458.0", "1"]
      - ["sel213", "71", "70", "368.0", "4"]
      - ["sel221", "30", "30", "402.0", "7"]
      - ["sel223", "**31**", "31", "480.0", "3"]
      - ["sel230", "50", "50", "372.0", "5"]
      - ["**total**", "**487**", "**404**", "412.0", "1–10"]

  - type: description
    title: "Inter-observer variability: 11 records, and one of them has three beats"
    body: |
      The paper says a second annotator repeated the procedure for 11 records "to
      permit study of inter-observer variability". All 11 come from MIT-BIH
      Arrhythmia, and the coverage is **not** matched: annotator 2 marked 404 of the
      487 beats annotator 1 marked in those records.

      The gap is concentrated. In `sel102` the audit reduced annotator 2 from 97
      first-pass annotations (`.qt2`) to **13** in the final file (`.q2c`) — **three
      beats**, against annotator 1's 85. `sel213` loses one beat, `sel223` none. So
      an unweighted inter-observer statistic over "the 11 records" is dominated by
      the nine where the two agree on which beats to mark, and `sel102`'s
      disagreement — the largest — contributes almost nothing. Weight by
      `n_annotated_beats_annotator2`, which is why the column exists.

      Note that `sel114` is in fold 9 (val) and `sel116` in fold 10 (test), so two
      of the 11 double-annotated records are outside the training folds. If your
      experiment is specifically about annotator disagreement, use `split=None` with
      `fold_numbers` and select on `has_second_annotator` rather than taking a
      standard split.

  - type: description
    title: "20 lead layouts, and the names disagree with `edb` for 31 of 33 records"
    body: |
      `config.leads` is 2 for every record; `config.lead_names` is the modal layout
      and it is the **placeholder** pair `ECG1`/`ECG2`, which 57 records use. Those
      57 — every excerpt from Supraventricular, Normal Sinus Rhythm, ST Change,
      Long-Term and the sudden-death Holters — state no electrode placement anywhere
      in the release. They are spelled `ECG1`/`ECG2` to match `afdb`, `nsrdb`,
      `chfdb`, `ltafdb`, `svdb` and `sddb`, so cross-dataset code sees one
      convention, and they mean channel 0 and channel 1.

      All 20 layouts are declared in `record_lead_layouts`, so
      `ECGDataset(leads=[...])` resolves the requested **names** against each
      record's own header and raises for a record whose layout lacks one, rather
      than returning whichever signal sits at that index:

      | Layout | Recs | Layout | Recs | Layout | Recs | Layout | Recs |
      |---|---|---|---|---|---|---|---|
      | `ECG1`/`ECG2` | 57 | `D3`/`V3` | 3 | `V5`/`MLII` | 1 | `CM5`/`CM2` | 1 |
      | `MLII`/`V1` | 8 | `CM5`/`ML5` | 3 | `D3`/`D4` | 1 | `CM2`/`ML5` | 1 |
      | `V4`/`D3` | 7 | `MLII`/`V5` | 2 | `V3`/`D3` | 1 | `CM5`/`CM4` | 1 |
      | `CM5`/`CC5` | 6 | `V5`/`V2` | 2 | `V2-V3`/`V5` | 1 | `V5`/`V1` | 1 |
      | `D3`/`V4` | 5 | `MLII`/`V2` | 2 | `CM5`/`mod.V1` | 1 | `V1-V2`/`V4-V5` | 1 |

      `D3`/`V4` and `V4`/`D3` are both present, 5 and 7 records: the same electrode
      pair stored in either order. **No name is common to all 105 records**, so
      every name-based selection raises for some of them.

      The 15 MIT-BIH Arrhythmia records name their channels exactly as `mitdb` does.
      The 33 European ST-T records do **not** match `edb` 1.0.0: QTDB keeps the
      ESC's original bipolar-electrode nomenclature and `edb` relabelled the same
      channels to standard names, so `edb`'s `MLIII` is QTDB's `D3` or `ML5`, its
      `V5` is `CM5`, its `V2` is `CM2`, `V1-V2` or `V2-V3`. Only `sele0107` and
      `sele0704` agree. The consequence is quiet rather than loud: **of the 33
      records the two datasets share, `leads=["V5"]` selects 14 under `edb`'s names
      and 2 under QTDB's**, over signals that are bit-identical. (Across whole
      datasets it is 51 of `edb`'s 90 records and 7 of QTDB's 105, the latter mostly
      MIT-BIH Arrhythmia excerpts.) No name maps to a *different* physical channel in
      the two releases, so nothing returns the wrong lead — but any code that selects
      by name silently covers a different set of records.

  - type: description
    title: "Amplitude: unreliable for 34 records, and four sit 5.12 mV high"
    body: |
      Intervals are what this database is for and they are unaffected by any of
      this. Amplitudes are not.

      **24 records have gains the paper calls estimates.** Of the sudden-death
      group it says: the Holters "are not calibrated with respect to amplitude; thus
      the signal gains recorded in the header files for these records are only
      estimates". That is the 23 `sddb` excerpts plus `sel17152`. QTDB's own
      re-estimates are visible in the numbers — `sel51` declares 600 adu/mV and
      `sel52` 400, where `sddb` declares 800 for both, so their millivolt amplitudes
      differ from `sddb`'s by 3× and 2× for the same digital samples.

      **10 more records declare a gain of `0`**, which is WFDB for "uncalibrated".
      wfdb substitutes its 200 adu/mV fallback, so `p_signal` looks like millivolts
      and is a nominal scale: all four Long-Term records (`sel14046`, `sel14157`,
      `sel14172`, `sel15814`) and six Supraventricular ones (`sel820`, `sel821`,
      `sel847`, `sel853`, `sel883`, `sel891`). `amplitude_calibrated` is `False` for
      all 34 of these records and `True` for the other 71.

      **Four records carry a constant +5.12 mV pedestal.** `sel100`, `sel102`,
      `sel103` and `sel104` are the only records declaring an explicit baseline of 0
      next to an `adc_zero` of 1024 (`200(0) 11 1024`); `mitdb`'s own copies declare
      `200 11 1024` and let wfdb use `adc_zero`. wfdb honours the explicit baseline,
      so these four come back offset by 1024/200 mV: their signals never go
      negative, and their minima sit at +4.0 to +4.6 mV. `dc_pedestal_mv` carries
      the value; subtract it if you need them comparable with the other 101.

      Three European ST-T records also re-declare a gain below 200 — `sele0112` at
      185, `sele0116` at 135, `sele0136` at 120 — which is why
      `amplitude_range_mv` has to reach ±17.058 mV, the 12-bit rail at the loosest
      gain in the release. Nothing comes near it: over all 23.6 million sample-pairs
      the extremes are **−7.800** and **+16.675 mV**, the latter being `sele0136`'s
      channel 0 riding its own DC offset to 97.7% of its own rail.

  - type: description
    title: "Clinical metadata exists for 48 records, and for 3 it contradicts `edb`"
    body: |
      57 of the 105 headers carry no clinical line at all. The other 48 carry what
      the source database published, in two different formats, and
      `clinical_source` says which.

      The 15 MIT-BIH Arrhythmia records carry mitdb's own block **byte-identical**:
      `# 69 M 1085 1629 x1` (age, sex, analog tape, recorder, playback speed), then
      medications, then a free-text description of the record's arrhythmia. `sel103`
      records its age as `-1`, mitdb's sentinel for unknown, which this loader
      returns as NaN.

      The 33 European ST-T records carry an **earlier, coarser vintage** of the same
      text `edb` 1.0.0 has. QTDB writes "Coronary artery disease" where `edb` gives
      the angina type, "Coronary angiography" where `edb` gives the vessel count and
      culprit arteries, and "unspecified medication" where `edb` lists the drugs.
      Not one of the 33 blocks matches `edb`'s.

      For three records the two releases **disagree substantively**: QTDB's
      `sele0116`, `sele0121` and `sele0122` say "Coronary artery disease", while
      `edb`'s headers for `e0116`, `e0121` and `e0122` record **normal coronary
      arteries**. The recordings are the same samples; the clinical statement is
      not. Prefer `edb` for these 33 records. `sele0166` records both age and sex as
      `-`, unknown, and `sele0405` is the one record described as a normal subject.

      Ages run 32–84 where recorded, over 46 records; sex is known for 47.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset qtdb --data-path /path/to/qtdb/1.0.0/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # window=(150000, 74993) is exactly the annotated region — samples 150,000 to
      # 224,993, i.e. 600.0 s to the end of the shortest record. Without it you get
      # 15 minutes of signal of which 98% has no ground truth. window= is pushed into
      # the reader, so it also avoids decoding the first ten minutes.
      ds = ECGDataset(
          "qtdb",
          split="train",
          data_path="/path/to/qtdb/1.0.0/",
          window=(150000, 74993),
          labels=True,
      )

      len(ds)                                        # 85
      ds[0]["signal"].shape                          # torch.Size([2, 74993])
      ds[0]["record_id"]                             # 'sel100'
      ds.lead_names                                  # ('ECG1', 'ECG2') — the modal
                                                     # layout, and a placeholder pair
      ds[0]["labels"]["lead_names"]                  # 'MLII;V5'  <- this record's real
                                                     #               channels
      ds[0]["labels"]["source_database"]             # 'mitdb'
      ds[0]["labels"]["source_record"]               # '100'
      ds[0]["labels"]["source_offset_secs"]          # 420.0
      ds[0]["labels"]["source_catalogue_slug"]       # 'mit-bih-arrhythmia-database'
                                                     #   <- the leakage partner
      ds[0]["labels"]["n_annotated_beats"]           # 30
      ds[0]["labels"]["waveform_pattern"]            # '(p)(N)t)'
      ds[0]["labels"]["median_qt_ms"]                # 398.0
      ds[0]["labels"]["median_qtc_bazett_ms"]        # 447.4
      ds[0]["labels"]["median_heart_rate_bpm"]       # 75.0
      ds[0]["labels"]["has_second_annotator"]        # True  <- 11 records
      ds[0]["labels"]["amplitude_calibrated"]        # True  <- False for 34
      ds[0]["labels"]["dc_pedestal_mv"]              # 5.12  <- subtract for sel100-104
      ds[0]["labels"]["source_record_verified"]      # None  <- NA: mitdb was resampled,
                                                     #          so no sample-wise check

      # The ground truth. It is per beat, so it is a separate call: 3,623 rows with
      # up to eleven fiducial points each, in 250 Hz samples.
      from ecgbench.labels.qtdb import load_beat_annotations

      beats = load_beat_annotations("/path/to/qtdb/1.0.0/")
      beats.query("record_name == 'sel100'")[
          ["qrs_onset", "qrs_peak", "qrs_offset", "t_offset", "qt_ms", "rr_ms"]
      ].head()

      # Boundaries are absolute samples; subtract the window start to index the tensor.
      row = beats.query("record_name == 'sel100'").iloc[0]
      onset_in_window = int(row.qrs_onset) - 150000

      # Annotator 2, for inter-observer work. 11 records, and sel102 has 3 beats.
      second = load_beat_annotations("/path/to/qtdb/1.0.0/", annotator="q2c")
      len(second)                                    # 404

      # Do NOT train on source_database — it is provenance, not pathology. It is the
      # stratification class because the release has no diagnostic label at all.
      ds.labels_df["source_database"].value_counts()
      # edb 27, sddb 18, mitdb 13, svdb 11, nsrdb 8, stdb 5, ltdb 2, bih_control 1

      # 34 records have unreliable amplitude calibration. Filter before comparing
      # millivolts across source databases; intervals are unaffected. Note that
      # labels_df is reindexed positionally to align with metadata_df, so the record
      # names come from metadata_df rather than from the labels index.
      names = list(ds.metadata_df["record_name"])
      [n for n, ok in zip(names, ds.labels_df["amplitude_calibrated"]) if not ok]
      # ['sel14172', 'sel15814', 'sel17152', 'sel30', 'sel31', 'sel32', ...]  27 in train

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/qtdb/1.0.0/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/C24K53" }
      - { label: "Laguna et al. 1997 (the database paper, with Tables 1 and 2)", url: "https://physionet.org/content/qtdb/1.0.0/doc/" }
      - { label: "ecgpuwave (the automatic delineator whose output ships)", url: "https://physionet.org/content/ecgpuwave/1.3.4/" }
      - { label: "PhysioBank annotation definitions", url: "https://archive.physionet.org/physiobank/annotations.shtml" }
---
