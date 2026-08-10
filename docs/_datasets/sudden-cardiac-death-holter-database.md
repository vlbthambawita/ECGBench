---
slug: "sudden-cardiac-death-holter-database"
name: "Sudden Cardiac Death Holter Database"
category: "two-lead"
order: 8
status: "completed"
source_url: "https://physionet.org/content/sddb/1.0.0/"
url_label: "physionet.org"
format: "2-lead (ECG1 + ECG2, both unnamed) · 3.9–25.1 h · 250 Hz · WFDB"
patients: "23"
records: "23"
access: "open"
license: "ODC Attribution"
origin_institution: "MIT · Boston-area hospitals"
origin_country: "USA — Boston, MA"
leads: 2
paper_title: "Greenwald, Development and analysis of a ventricular fibrillation detector, MS thesis, MIT 1986"
paper_doi: "https://doi.org/10.13026/C2W306"
search_keywords: "sudden cardiac death holter sddb ventricular tachycardia vt vf fibrillation cardiac arrest usa mit boston greenwald scd arrhythmia terminal event vfon pacing paced atrial fibrillation ectopy two-lead long-term unaudited audited annotations malignant ventricular ectopy vfdb"

sections:
  - type: description
    title: "Overview"
    body: |
      **23 complete Holter recordings of patients who died — or nearly died —
      during the recording.** Every subject sustained a ventricular
      tachyarrhythmia and most had an actual cardiac arrest. **446.6 hours** of
      two-channel signal at 250 Hz, collected mainly in the 1980s in Boston-area
      hospitals by Scott Greenwald at MIT and published by PhysioNet in 2004 as the
      first sudden cardiac death database. Every other long-term database in this
      catalogue describes a rhythm the subject lives with; this one describes the
      one they did not survive.

      **Use `version="original"`, not the default `clean`.** 20 of the 23 records
      contain NaN samples, so they fail `nan_values` and `clean/` holds **3
      records with empty val and test**. The NaN is not corruption — it is WFDB's
      invalid-sample marker for brief analog-tape dropouts — but it is real and it
      lands in your tensor. This has its own section below and it is the first thing
      to know.

      **The terminal event is in a header comment, not an annotation.** `#vfon:
      HH:MM:SS` gives the onset elapsed from the record start, in 20 of the 23
      headers. There is not one `[` (VFON) annotation anywhere in the release.
      Onset lands from **6.0%** (record 37) to **98.9%** (record 35) of the way
      through, leaving between 976 s and 85,007 s of signal after it — so no single
      `window=` captures the event across records. Window relative to
      `vf_onset_secs`.

      **There are two annotators and they cover different records.** `.ari` is
      unaudited detector output for all 23 records (1,888,495 beats); `.atr` is the
      audited reference for only **12** (849,831 beats), which PhysioNet itself
      calls "an incomplete set of audited annotation files". Every beat column is
      prefixed `ari_` or `atr_`, because an unprefixed `n_beats` would mean one
      thing for half the release and another for the rest.

      **The `(AFIB` markers are not an atrial fibrillation label.** They disagree
      with the release's own published rhythm column in both directions — they miss
      one of the four AF subjects entirely and flag six sinus subjects at 22–36%.
      Section below.

      **No two records are the same length**, 3,540,000 to 22,627,500 samples — the
      longest is 6.4x the shortest, where `chfdb`'s spread is 1.2%. Any window must
      fit inside 14,160 s.

      **The two channels are not named leads.** Both signal lines of every current
      header end in the bare description `ECG`, as in `ltafdb`; the release states
      no electrode placement anywhere. `ECG1`/`ECG2` are channel positions.

  - type: table
    title: "The 23 records, recomputed from the files"
    headers: ["Rec", "Sex", "Age", "Rhythm", "Hours", "VF onset h", "VF %", "`ari` beats", "`atr` beats", "Gain", "NaN", "Mean HR", "`ari` AF %", "Fold"]
    rows:
      - ["30", "M", "43", "sinus", "24.55", "7.91", "32.2", "131,323", "127,418", "800", "2,095", "90.0", "2.2", "1"]
      - ["31", "F", "72", "sinus", "13.98", "13.71", "**98.1**", "60,905", "62,505", "800", "**0**", "75.0", "6.6", "1"]
      - ["32", "**?**", "62", "sinus + pacing", "24.33", "16.75", "68.9", "121,340", "120,482", "800", "1,811", "84.1", "2.4", "5"]
      - ["33", "F", "30", "sinus", "24.55", "4.77", "19.4", "69,032", "—", "800", "**0**", "**46.9**", "34.1", "3"]
      - ["34", "M", "34", "sinus", "7.09", "6.60", "93.0", "27,208", "26,667", "800", "8,681", "64.7", "1.5", "2"]
      - ["35", "F", "72", "**afib**", "24.87", "24.58", "**98.9**", "100,297", "96,767", "800", "3,481", "67.5", "**99.4**", "10 (test)"]
      - ["36", "M", "75", "**afib**", "20.36", "18.98", "93.3", "77,126", "77,159", "800", "517", "63.2", "**72.9**", "9 (val)"]
      - ["37", "F", "**89**", "**afib**", "25.13", "1.52", "**6.0**", "63,636", "—", "800", "1,207", "**43.3**", "**0.9**", "2"]
      - ["38", "**?**", "**?**", "sinus", "18.31", "8.03", "43.9", "74,211", "—", "800", "4,119", "82.0", "25.5", "7"]
      - ["39", "M", "66", "sinus", "**5.78**", "4.63", "80.1", "31,139", "—", "**200**", "**85,560**", "88.7", "22.1", "9 (val)"]
      - ["40", "M", "79", "**paced**", "24.88", "**none**", "—", "108,852", "—", "800", "6", "73.0", "1.5", "3"]
      - ["41", "M", "**?**", "sinus + pacing", "**3.93**", "2.99", "76.0", "21,985", "17,859", "800", "10,010", "95.9", "28.1", "8"]
      - ["42", "M", "**17**", "sinus", "25.14", "**none**", "—", "115,402", "—", "800", "46,089", "76.5", "0.9", "6"]
      - ["43", "M", "35", "sinus + pacing", "23.13", "15.62", "67.5", "124,439", "—", "800", "1,901", "90.9", "10.6", "2"]
      - ["44", "M", "**?**", "sinus", "23.33", "19.65", "84.2", "118,482", "—", "800", "4,379", "85.6", "36.3", "5"]
      - ["45", "M", "68", "sinus", "24.16", "18.15", "75.2", "99,389", "97,762", "800", "8,442", "69.1", "1.4", "10 (test)"]
      - ["46", "F", "**?**", "sinus", "4.25", "3.70", "86.9", "17,153", "16,523", "800", "**0**", "74.6", "26.4", "3"]
      - ["47", "M", "34", "sinus", "23.60", "6.22", "26.3", "92,027", "—", "**200**", "794", "64.9", "0.6", "7"]
      - ["48", "M", "80", "sinus", "24.60", "2.49", "10.1", "**146,549**", "—", "800", "10,614", "**99.8**", "13.4", "4"]
      - ["49", "M", "73", "sinus + pacing", "24.87", "**none**", "—", "94,615", "82,674", "800", "9,053", "62.1", "5.4", "6"]
      - ["50", "F", "68", "**afib**", "23.13", "11.76", "50.9", "63,963", "—", "800", "488", "46.4", "**96.9**", "1"]
      - ["51", "F", "67", "sinus + pacing", "**25.14**", "22.97", "91.4", "82,020", "77,859", "800", "34", "54.9", "**0.0**", "8"]
      - ["52", "F", "82", "sinus", "7.52", "2.54", "33.8", "47,402", "46,156", "800", "2,427", "**104.8**", "17.6", "4"]
      - ["**total**", "13 M / 8 F / 2 ?", "17–89", "18 / 4 / 1", "**446.63**", "20 of 23", "6.0–98.9", "**1,888,495**", "**849,831** (12 recs)", "800 · 200", "**201,708** (20 recs)", "43.3–104.8", "—", "1–10"]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the 23 headers and both
      annotation layers, after verifying the shipped data against the release's own
      `SHA256SUMS.txt` — **all 109 files match**. That total includes 23 superseded
      `.hea-` backup headers, which predate the 2008 revision that renamed both
      signal descriptions to `ECG`. The record list comes from the shipped `RECORDS`
      file, so none of them can enter the partition.

      **The published cohort description reproduces exactly.** PhysioNet states "18
      patients with underlying sinus rhythm (4 with intermittent pacing), 1 who was
      continuously paced, and 4 with atrial fibrillation", and the landing page's
      clinical table gives precisely that: 14 plain sinus plus records 32, 43, 49
      and 51 intermittently paced makes 18; record 40 is the continuously paced one;
      35, 36, 37 and 50 are the atrial fibrillation subjects. That agreement is the
      **only** external check the clinical table admits, and `clinical_frame()`
      raises rather than quietly rebalancing folds if a later edit breaks it.

      **The landing page's duration for record 49 is stale.** Its table lists
      25:01:40 (90,100 s), but `49.hea` declares 22,380,957 samples = 89,523.8 s
      (24:52:03.8) — 576 s less. That header was revised **2017-05-02**, where every
      other header dates from 2008-04-05, and the `.dat` file size (67,142,872
      bytes at 1.5 bytes per sample pair) confirms the shipped count. The file is
      right and the page was not updated; the superseded `49.hea-` still declares
      the old 22,525,000. **Every other record's listed duration matches its header
      exactly**, so this is the one discrepancy in the release.

      **Nothing clinical is in the data files.** Sex, age, history, medication and
      underlying rhythm exist only in the "Clinical information" table on the
      landing page, transcribed into `ecgbench/labels/sddb.py` as `CLINICAL_TABLE`.
      This is weaker provenance than any sibling — `chfdb` has a header comment,
      `mitdb` a shipped directory — and it cannot be recomputed from the data.
      PhysioNet is explicit about why so much is missing: "Because of the
      retrospective nature of this collection… Patient information is limited, and
      sometimes completely unavailable, including data regarding drug regimens and
      drug dosages." Age is absent for 4 subjects, sex for 2, and history and
      medication read "Unknown" for 12 and 13 respectively. **Record 37 is 89** — a
      real recorded value at the boundary of the usual age-ceiling convention, not a
      censored one.

      **Six subjects were on digoxin and five on quinidine**, agents that alter rate
      and repolarisation. The medication column is free text and populated for only
      six records, so treat it as a hint rather than a variable.

      **`ari` beats and `atr` beats are different quantities, not two estimates of
      one.** Where both exist they disagree substantially — record 30 is 131,323
      against 127,418 — because one is a detector and the other a corrected
      reference over a shorter span. Never sum or average them.

      **`ari` AF % is detector output, not atrial fibrillation.** It has its own
      section below; the column is on this table because it describes the annotation
      file, and it is named `ari_*` for the same reason.

      **The HRV figures are descriptive, not a result.** `mean_hr_bpm`, `sdnn_ms`
      and `rmssd_ms` come from `.ari` RR intervals in [0.3 s, 2.0 s] — 17,946
      intervals rejected release-wide. They span the moment the subject's heart
      stopped working. Averaging heart rate across a terminal ventricular
      arrhythmia is not an HRV measurement of anything; segment around
      `vf_onset_secs` if you want physiology.

      **`cohort_label` is a constant and is not the stratification label.** All 23
      subjects sustained the arrhythmia, so there is no negative class here. Folds
      are stratified on the underlying rhythm — see the fold section. Train on
      `vf_onset_fraction`, the `aami_*` counts or the HRV columns; never on
      `stratify_class`.

  - type: description
    title: "`clean/` holds 3 records — use `version=\"original\"`"
    body: |
      This is the one dataset in the catalogue whose `clean` version is unusable,
      and the reason is worth understanding rather than working around.

      **WFDB's invalid-sample marker in format 212 is digital −2048**, and `wfdb`
      returns those samples as NaN. There are **201,708** of them across the
      release, in every record except 31, 33 and 46:

      | | |
      |---|---|
      | Records affected | **20 of 23** |
      | Total invalid samples | 201,708 |
      | Longest single run | **1.79 s** |
      | Median run | 4–84 ms |
      | Runs per channel | 26 to 900 |
      | Worst affected channel | **0.93%** (record 39) |

      These are short scattered dropouts from 1980s analog Holter tape, not gaps —
      which is exactly the problem. `check_nan_values` has no threshold, so all 20
      records fail it, and because the three unaffected records all land in train
      folds:

      | Version | Records | Train | Val | Test |
      |---|---|---|---|---|
      | `original` | **23** | 19 | 2 | 2 |
      | `clean` | **3** (31, 33, 46) | 3 | **0** | **0** |

      So `clean/val/fold_9.csv` and `clean/test/fold_10.csv` are header-only, and
      because `ECGDataset` defaults to `version="clean"` it fails in one of two
      ways — **neither of which names the real cause**:

      ```python
      ECGDataset("sddb", split="val", labels=True)
      # ValueError: No record in split 'val' matched a label row.   <- blames the join

      ds = ECGDataset("sddb", split="val")     # constructs fine
      len(ds)                                  # 0                 <- says nothing
      ds[0]                                    # IndexError
      ```

      **The real cause is that the split is empty.** Pass `version="original"`.

      **Why the check was kept anyway.** Dropping `nan_values` from this config
      would make `clean/` equal `original/` and leave `quality_issues` empty for
      every record — so a user would get NaN in their tensors, and a NaN loss, with
      no warning from ECGBench at all. A degenerate `clean/` fails loudly and is
      documented on this page; silent NaN does not. The trade-off is real rather
      than free, and it is recorded in the config and pinned by a test so nobody
      "fixes" it by silencing it.

      **Handling it.** `original/` carries the per-record count in `quality_issues`;
      `ecgbench.labels.sddb.scan_invalid_samples(data_path)` recomputes per-channel
      counts, fractions and longest runs from any copy. Then either
      `torch.nan_to_num(signal)`, mask the loss, or drop affected windows — a 10 s
      window usually contains none, but a whole record will.

  - type: description
    title: "The terminal event lives in a header comment"
    body: |
      The database is named for an event that has **no annotation**. There is not
      one `[` (VFON) or `]` (VFOFF) marker in the release. What there is, in 20 of
      the 23 headers, is a comment:

      ```
      30 2 250 22099250 12:00:00
      30.dat 212 800 12 0 51 -24065 0 ECG
      30.dat 212 800 12 0 145 21051 0 ECG
      #Produced by xform_new from record 30, beginning at 26:35.000
      #vfon: 07:54:33
      ```

      **`#vfon:` is elapsed from the start of the record, not a time of day.**
      Reading it as a clock time is wrong in both directions: record 30 starts at
      12:00:00 and its onset is 07:54:33, and record 35's onset is `24:34:56`,
      which is not a clock time at all. `vf_onset_secs` parses it as elapsed
      seconds.

      **Three records have no onset**, and the landing page says why: 40 "(paced,
      no VF)", 42 "(no VF)", 49 "(paced, no VF)". `has_vf_onset` distinguishes them
      from a parse failure.

      **No single `window=` can capture the event.** Onset spans 6.0% to 98.9% of
      the record, so `secs_after_vf_onset` runs from **976 s** (record 31) to
      **85,007 s** (record 37) — in some records there is barely a quarter-hour of
      signal after the terminal event, in others most of a day before it. A fixed
      window from the record start captures it for a couple of records and misses it
      for the rest. Window per record instead:

      ```python
      from ecgbench import ECGDataset
      from ecgbench.labels import load_labels

      labels = load_labels("sddb", "/path/to/sddb/1.0.0/")
      onset = int(labels.loc["30", "vf_onset_secs"] * 250)

      ds = ECGDataset("sddb", split="train", version="original",
                      data_path="/path/to/sddb/1.0.0/",
                      window=(onset - 1250, 2500))   # 5 s either side
      ```

      Because `window=` is pushed into the reader, that decodes 2,500 samples rather
      than 22 million.

  - type: description
    title: "Two annotators, covering different records, with disjoint vocabularies"
    body: |
      The shipped `ANNOTATORS` file names both, and the distinction is not cosmetic:

      ```
      ari     unaudited beat annotations
      atr     reference beat annotations
      ```

      | | `.ari` | `.atr` |
      |---|---|---|
      | Status | automated detector, uncorrected | audited reference |
      | Records | **all 23** | **12** (30, 31, 32, 34, 35, 36, 41, 45, 46, 49, 51, 52) |
      | Beats | 1,888,495 | 849,831 |
      | Rhythm markers | 1,019 `+` | **none at all** |
      | Quality markers | none | 83 `~`, 16,403 `|` |
      | Extra layers | 1,150 `?` LEARN, 3,577 `s` ST | — |

      Every other MIT-BIH-family release in this catalogue has one annotator, so the
      natural move is to reach for `.atr` and treat it as the reference. **That
      silently drops 11 records.** `has_audited_annotation` is the column to filter
      on; PhysioNet describes the audited set as incomplete and invites
      contributions to finish it.

      **The two symbol vocabularies are disjoint where it matters**, which makes
      the AAMI EC57 reduction mandatory rather than convenient:

      | Symbol | Meaning | AAMI | `.ari` | `.atr` |
      |---|---|---|---|---|
      | `N` | normal | N | 1,787,515 | 745,671 |
      | `r` | R-on-T PVC | **V** | **58,820** | **0** |
      | `B` | bundle branch block | N | **0** | **54,725** (all in record 36) |
      | `/` | paced | Q | **0** | **23,123** |
      | `f` | paced/normal fusion | Q | **0** | 412 |
      | `V` | PVC | V | 19,569 | 23,600 |
      | `S` | supraventricular | S | 22,085 | 384 |
      | `J` | junctional premature | S | 0 | 1,508 |
      | `F` | ventricular fusion | F | 0 | 309 |
      | `E` | ventricular escape | V | 417 | 16 |
      | `Q` | unclassifiable | Q | 89 | 82 |
      | `a` | aberrated atrial | S | 0 | 1 |

      So `beat_V` alone is **not** ventricular ectopy in either file: in `.ari` it
      misses the 58,820 R-on-T beats, which outnumber plain `V` three to one, and in
      `.atr` it misses nothing ventricular but sits beside 54,725 `B` and 23,123 `/`
      that a naive "normal beats" count would drop. Use `ari_aami_V` /
      `atr_aami_V`, which are the only cross-annotator-comparable counts here.

      **Two `.ari` layers are not beats at all.** Exactly **50** `?` (LEARN)
      annotations open every one of the 23 records, inside the first 30–65 s — the
      detector's start-up phase. And 3,577 `s` (STCH) markers delimit unaudited
      ST-segment episodes, with aux notes like `(ST0+` and `ST0+)` for channel and
      direction, in 22 of 23 records (record 46 has none; record 47 has 363
      episodes). Together that is 4,727 annotations a scanner assuming every symbol
      is a QRS would fold into `n_beats`. They are counted in `ari_n_learning` and
      `ari_n_st_episodes` instead — and episodes are counted from **openings**,
      because record 44's two episodes are never closed.

  - type: description
    title: "The `(AFIB` markers are not an atrial fibrillation label"
    body: |
      The `.ari` files carry 1,019 `+` rhythm markers spelling `(AFIB` and `(N`, in
      22 of the 23 records. They look exactly like `afdb`'s or `ltafdb`'s reference
      rhythm annotations. They are not: they are unaudited detector output, and held
      against the landing page's own "Underlying Cardiac Rhythm" column they are
      **wrong in both directions**.

      | Record | Published rhythm | `ari` AF % | |
      |---|---|---|---|
      | 35 | atrial fibrillation | 99.4% | agrees |
      | 50 | atrial fibrillation | 96.9% | agrees |
      | 36 | atrial fibrillation | 72.9% | agrees |
      | **37** | **atrial fibrillation** | **0.9%** | **misses it entirely** |
      | 44 | sinus | **36.3%** | false positive |
      | 33 | sinus | **34.1%** | false positive |
      | 41 | sinus + pacing | **28.1%** | false positive |
      | 46 | sinus | **26.4%** | false positive |
      | 38 | sinus | **25.5%** | false positive |
      | 39 | sinus | **22.1%** | false positive |
      | 51 | sinus + pacing | 0.0% | no markers at all |

      Three of the four published AF subjects come out at 72.9–99.4%, which is why
      the column is tempting. **Record 37 comes out at 0.9%**, and six sinus
      subjects at 22–36%. A model trained on this column learns the detector, not
      the arrhythmia.

      Use **`underlying_rhythm`** (the page's verbatim text) or **`rhythm_class`**
      (`sinus` / `afib` / `paced`) instead. The `ari_afib_*` columns are exposed
      because they describe the annotation file — and carry the `ari_` prefix so it
      is never ambiguous which they are.

      Note also that **record 51 carries no `+` marker at all**, so its 0.0% means
      "never marked", not "no atrial fibrillation". `ari_has_rhythm_annotation`
      distinguishes the two.

  - type: description
    title: "The audited annotation stops early — at exactly 24 hours in four records"
    body: |
      Where the last audited beat actually falls, for the 12 records that have a
      `.atr`:

      | Record | Hours | Last `atr` beat | Tail gap |
      |---|---|---|---|
      | 49 | 24.87 | 84,530.2 s | **4,993.7 s** |
      | 51 | 25.14 | **86,398.6 s** | **4,111.4 s** |
      | 41 | 3.93 | 11,460.0 s | 2,700.0 s |
      | 46 | 4.25 | 13,310.2 s | 1,999.8 s |
      | 30 | 24.55 | **86,398.8 s** | 1,998.2 s |
      | 35 | 24.87 | **86,399.3 s** | 3,120.7 s |
      | 32 | 24.33 | **86,399.4 s** | 1,200.6 s |
      | 52 | 7.52 | 26,046.7 s | 1,018.3 s |
      | 31 | 13.98 | 49,347.6 s | 972.4 s |
      | 34 | 7.09 | 25,163.6 s | 356.4 s |
      | 36 | 20.36 | 73,167.5 s | 112.5 s |
      | 45 | 24.16 | 85,721.1 s | 1,238.9 s |

      **Records 30, 32, 35 and 51 all stop within 1.4 s of 86,400 s** — a hard
      24-hour cutoff applied to recordings that run to 25.1 h. Record 51 is
      additionally unannotated for its **first 1,078.7 s**.

      Nothing errors if you window past the cutoff: you get signal with no reference
      behind it, and an evaluation that scores a detector against silence. Check
      `atr_unannotated_tail_secs` and `atr_unannotated_head_secs`.

      The `.ari` files are the mirror image — they start 29.7–65.2 s in, because of
      the 50 LEARN annotations, and reach within 0.2–452.7 s of the end.

  - type: description
    title: "Signal quality: the `~` subtype is 51, which is not a channel bitmask"
    body: |
      The 12 `.atr` files carry **83 `~` quality transitions** and **16,403 `|`
      isolated-artifact markers**. WFDB defines a `~` subtype as a bitmask over the
      signals — 0 clean, 1 first channel noisy, 2 second, 3 both — and `svdb` and
      `nsrdb` decode exactly that.

      **This release only ever writes 0 or 51**, strictly alternating 51, 0, 51,
      0. So 51 plainly means "noisy" and 0 "clean", but **51 is not a valid
      two-channel mask** (both channels noisy would be 3), and ECGBench does not
      reinterpret it as one: it reports `atr_noisy_secs` and `atr_clean_secs` with
      **no per-channel split**, plus `atr_quality_subtypes` so a re-release adopting
      the documented encoding is visible rather than silently misread.

      The annotated noise is slight in any case: **579.5 s** — 0.161 h — across the
      whole release, at most 0.95% of a record (41), and 3 of the 12 annotated
      records carry no `~` at all (36, 46, 49).

      **Record 35 is the one head-unasserted case.** Its single `~` sits 12,600.3 s
      in and has subtype 0 — a transition *into* clean — so the span before it was
      never asserted to be anything. It is counted as clean, which is what WFDB
      does, and reported as `atr_quality_head_unasserted_secs` so the assumption is
      visible. `svdb` makes the same choice for the same reason.

  - type: table
    title: "Validation summary (250 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "23", "all records, with is_valid + quality_issues — **use this one**"]
      - ["clean", "3", "20 records excluded for NaN samples; val and test are empty"]

  - type: description
    title: "Only `nan_values` fires, and the amplitude bound is the union of two gains"
    body: |
      Of the six checks, exactly one fires. No lead is all-NaN or all-zero, no
      channel has near-zero variance across a whole record, `wfdb` reads all 23
      headers, and no sample leaves the amplitude bound.

      | Check | Records failed |
      |---|---|
      | `nan_values` | **20** |
      | `missing_leads` / `flat_line` / `corrupt_header` / `amplitude_outlier` | 0 |
      | `truncated_signal` | disabled |

      **`truncated_signal` is disabled**, by leaving `expected_samples` empty — and
      this is the least marginal such case in the catalogue. **All 23 lengths
      differ**, from 3,540,000 to 22,627,500 samples, so the longest record is 6.4x
      the shortest (`chfdb` varies by 232 s, `nsrdb` by three hours). Every one is a
      complete recording, so any single threshold would drop most of the release.
      Omitting the rate disables the check rather than making it fire.

      **`amplitude_range_mv` is `[-10.235, 10.235]`, and it is the looser of two
      rails because the gain is per record.** `adc_zero` is 0 and no channel
      declares a baseline, so a sample is confined to [−2047, 2047] adu once −2048
      is excluded as the invalid marker:

      | Gain | Records | Rail |
      |---|---|---|
      | 800 adu/mV | 21 | ±2.55875 mV |
      | **200 adu/mV** | **39 and 47** | **±10.235 mV** |

      A single range has to accommodate the loosest record or it fires on a sound
      one. It is not idle: **records 39 and 47 reach both ends**, and the observed
      extremes over all 3.2 billion samples are exactly −10.235 and +10.235 mV. What
      it guards is a mis-scaled copy — microvolts, or a re-release with the gains
      dropped — which would exceed it by orders of magnitude at once.

      **No float32 slack is needed here, unlike `chfdb`.** ECGBench loads signals as
      float32, and `float32(10.235)` is 10.234999656677246 while `float32(-10.235)`
      is −10.234999656677246 — both round *toward* zero, so neither can trip a bound
      set at the exact rail. `chfdb`'s attained rail of 10.585 rounds the other way
      and needs the extra thousandth; check the direction before copying either
      pattern.

      One record deserves a note of its own: **record 42's channel 0 sits at the
      positive rail for 2,337,660 samples**, 10.3% of the recording. It is inside
      the bound by construction, and `missing_leads` cannot see it because the
      channel is railed rather than zero, so nothing flags it. That record also has
      no VF onset.

  - type: description
    title: "Ten folds over 23 records, stratified on the underlying rhythm"
    body: |
      Four consequences of the arithmetic, stated rather than left to be
      discovered:

      - **`original` splits 19 / 2 / 2.** ECGBench's convention is folds 1–8 →
        train, 9 → val, 10 → test, and 23 records over 10 folds gives three folds
        of three and seven of two. So val holds records 36 and 39 and test holds 35
        and 45. For anything needing a real evaluation set, use cross-validation:
        `split=None` with `fold_numbers=[...]` selects by fold from `folds.csv` and
        ignores the default layout.
      - **`clean` splits 3 / 0 / 0**, which is the section above.
      - **Half the val/test records are atrial fibrillation subjects.** 36 lands in
        fold 9 and 35 in fold 10, so two of the four AF subjects in the release are
        the two non-train records with that rhythm. That is chance, not a choice,
        and another reason to cross-validate.
      - **Record 39 is in val**, and it is both the shortest-but-one record (5.78 h)
        and by far the most affected by invalid samples (85,560, 0.7–0.9% per
        channel). A two-record validation set carrying that is not a robust
        estimate of anything.

      **Why the underlying rhythm, and not ectopy burden?** Every subject sustained
      a ventricular tachyarrhythmia, so `cohort_label` is one value and carries no
      information a fold could be balanced on. What differs is the rhythm underneath
      the terminal event, and PhysioNet describes the cohort in exactly those terms:

      | Candidate fold axis | Class sizes | Result |
      |---|---|---|
      | **Underlying rhythm** | **18 / 4 / 1** | **works** |
      | Ventricular ectopy burden | any banding | works arithmetically, but see below |
      | VF onset present | 20 / 3 | works, but splits on whether a comment was written |
      | Sex | 13 / 8 / 2 | works, but says nothing about the recording |
      | Audited annotation available | 12 / 11 | best balanced, and a fact about PhysioNet's backlog |

      `StratifiedKFold` raises only when *every* class is smaller than `n_folds`, so
      the 18-record sinus class carries the split on its own and the singleton
      `paced` class is tolerated — `sklearn` warns that the least populated class has
      one member, and that warning is expected and correct.

      **Ectopy burden — the axis `svdb` and `chfdb` use — is ruled out by
      provenance, not arithmetic.** 11 of the 23 records have no audited annotation
      at all, so a burden band would be measuring a 1980s detector in half the
      release and a cardiologist in the other half, in one column. That is a worse
      failure than an unbalanced fold.

      Folds are **ungrouped**, and for an unusual reason. Unlike `nsrdb`, `svdb`,
      `afdb` and `chfdb`, which ship no subject identifier at all, this release
      *does* identify its subjects — it identifies them **with the record name**. The
      landing page keys the clinical table "Subject Number" with values 30–52, one
      record per subject, so `patient_id_column` is null because a patient column
      would be a verbatim copy of the index and grouping on it would be a no-op.

  - type: description
    title: "Overlap: the half-hour excerpts are a separate PhysioNet database"
    body: |
      **PhysioNet states the overlap itself**: "Half-hour excerpts of these
      recordings have been (and remain) available as the MIT-BIH Malignant
      Ventricular Ectopy Database" (`vfdb`, also called the Malignant Ventricular
      Arrhythmia Database on the same page). Anyone combining the two is training
      and evaluating on the same recordings.

      `vfdb` is **not in this catalogue**, so there is no `related:` edge to declare
      — a `related:` slug has to resolve to an existing entry. If it is ever added,
      the edge belongs on one side only, with `shares_records: true` and a note
      saying that.

      **Against the other Boston Holter databases, no shared recording was found.**
      SDDB, `mitdb`, `afdb`, `nsrdb`, `svdb`, `stdb`, `ltafdb`, `edb` and `chfdb`
      have overlapping provenance and none ships a subject identifier that would
      join, so the question was settled from the annotation files. RR intervals in
      seconds are commensurable across sampling rates, so the check compares
      **sequences of 20 consecutive RR intervals quantised to 8 ms**, on two
      half-bin-shifted grids — the same method used for `chfdb`, `ltafdb` and
      `nsrdb`.

      Against controls that make a null result mean something — a positive control
      re-finding each SDDB record in its own pool at **100.0000%**, and a negative
      control of each SDDB record against the pool of the other 22 known-distinct
      subjects peaking at **0.0697%** — no database shares a recording:

      | Pool | Records | Highest SDDB hit rate | At 30 intervals |
      |---|---|---|---|
      | MIT-BIH Arrhythmia | 48 | 0.0000% | — |
      | MIT-BIH NSR | 18 | 0.0000% | — |
      | MIT-BIH SVDB | 78 | 0.0027% | — |
      | Long-Term AF | 84 | 0.0091% | 0.0000% |
      | MIT-BIH AFDB | 25 | 0.0431% | 0.0048% |
      | MIT-BIH ST Change | 28 | 0.0575% | 0.0116% |
      | European ST-T | 90 | 0.0616% | 0.0096% |
      | BIDMC CHF | 15 | **0.1915%** | 0.0048% |

      Every figure sits at or below the 0.0697% chance-collision floor except BIDMC
      CHF's 0.1915%. Lengthening the signature settles it: a genuinely shared
      recording stays near 100% as the signature grows — the positive control does
      exactly that — while this decays by a factor of 40, which is what chance
      collisions do. The `chfdb` page reports the same comparison measured from the
      other direction (0.1079% → 0.0026%) and reaches the same conclusion.

      Two limitations worth stating rather than glossing. The RR signature survives
      *refinement* of annotations but not *re-detection*, so a shared recording
      annotated by two genuinely different detectors could evade it — and this check
      used SDDB's **unaudited** `.ari` annotations, because they are the only layer
      covering all 23 records, which makes that marginally more plausible here than
      elsewhere. And subject-level overlap cannot be checked at all, because none of
      these releases ships a subject identifier that crosses databases.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset sddb --data-path /path/to/sddb/1.0.0/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # version="original", NOT the default "clean" — clean/ holds 3 records and its
      # val and test are empty. And records run 3.9-25.1 h, so a window is needed to
      # batch at all; because window= is pushed into the reader it also avoids
      # decoding the other 25 hours.
      ds = ECGDataset(
          "sddb",
          split="train",
          version="original",
          data_path="/path/to/sddb/1.0.0/",
          window=(0, 2500),         # first 10 s at 250 Hz
          labels=True,
      )

      len(ds)                                    # 19
      ds[0]["signal"].shape                      # torch.Size([2, 2500])
      ds[0]["record_id"]                         # 30
      ds.lead_names                              # ('ECG1', 'ECG2') — channel positions;
                                                 # both headers say bare 'ECG'
      ds[0]["labels"]["cohort_label"]            # 'sudden_cardiac_death'  — all 23
      ds[0]["labels"]["sex"]                     # 'M'
      ds[0]["labels"]["age"]                     # 43.0   (4 subjects have none)
      ds[0]["labels"]["underlying_rhythm"]       # 'Sinus'
      ds[0]["labels"]["rhythm_class"]            # 'sinus'  <- use THIS for an AF label
      ds[0]["labels"]["vf_onset_secs"]           # 28473.0  elapsed, not a clock time
      ds[0]["labels"]["vf_onset_fraction"]       # 0.3221...
      ds[0]["labels"]["has_audited_annotation"]  # True   <- False for 11 of 23
      ds[0]["labels"]["ari_n_beats"]             # 131323   unaudited, all records
      ds[0]["labels"]["atr_n_beats"]             # 127418   audited, 12 records
      ds[0]["labels"]["ari_afib_fraction"]       # 0.0220...  NOT an AF label
      ds[0]["labels"]["atr_unannotated_tail_secs"]  # 1998.2  <- check before windowing
                                                    #            against audited beats

      # 20 of 23 records carry NaN (WFDB's -2048 invalid-sample marker). A 10 s window
      # often has none; a whole record will not. Handle it before computing a loss:
      import torch
      signal = torch.nan_to_num(ds[0]["signal"])

      # The regression target this database exists for. It is the only label derived
      # from the files rather than from a detector or the landing page.
      ds.labels_df["vf_onset_fraction"].describe()

      # Record 41 is the shortest at 3,540,000 samples (3.93 h), so a window must fit
      # inside that rather than inside the longest record's 22,627,500, or it raises
      # WindowOutOfRangeError naming the record and its true length.

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/sddb/1.0.0/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/C2W306" }
      - { label: "Greenwald 1986 (MS thesis, MIT)", url: "https://dspace.mit.edu/handle/1721.1/92988" }
      - { label: "MIT-BIH Malignant Ventricular Ectopy DB (the half-hour excerpts)", url: "https://physionet.org/content/vfdb/1.0.0/" }
      - { label: "PhysioBank annotation definitions", url: "https://archive.physionet.org/physiobank/annotations.shtml" }
---
