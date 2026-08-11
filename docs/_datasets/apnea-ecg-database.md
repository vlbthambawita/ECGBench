---
slug: "apnea-ecg-database"
name: "Apnea-ECG Database"
category: "one-lead"
order: 3
status: "completed"
source_url: "https://physionet.org/content/apnea-ecg/1.0.0/"
url_label: "physionet.org"
format: "1-lead (ECG, unnamed) · 6.75–9.62 h · 100 Hz · WFDB · per-minute apnea labels"
patients: "30"
records: "70"
access: "open"
license: "ODC-By 1.0"
origin_institution: "Philipps-University Marburg"
origin_country: "Germany"
leads: 1
paper_title: "Penzel et al., The Apnea-ECG Database, CinC 2000"
paper_doi: "https://doi.org/10.13026/C23W2R"
search_keywords: "apnea ecg sleep germany marburg philipps overnight holter obstructive sleep apnea osa ahi apnea-hypopnea index polysomnography per-minute annotations physionet cinc challenge 2000 single-lead one-lead penzel"

sections:
  - type: description
    title: "Overview"
    body: |
      **The reference dataset for detecting sleep apnea from the ECG alone**, and
      the data of the PhysioNet/CinC Challenge 2000. 70 overnight single-lead
      recordings — **573.7 hours** at 100 Hz — with an expert apnea annotation for
      every one of their **34,313 minutes**, of which **13,064 are apnea**.

      **The ground truth is per minute, not per record.** Each `.apn` file holds
      one `A`/`N` annotation per minute of recording, assigned by an expert from
      simultaneously recorded respiration and oxygen saturation. ECGBench exposes
      the whole sequence as `apnea_sequence`, so minute *i* is labelled by
      `apnea_sequence[i]` and `window=(i * 6000, 6000)` returns exactly that
      minute. The record-level `apnea_class` is a whole-night *summary* the
      challenge used to describe records; the task is the minutes.

      **The 70 records come from 30 subjects, and the release's own learning/test
      split leaks 18 of them.** This is the most important thing on this page.
      Apnea-ECG publishes **no subject identifier anywhere** — not in the headers,
      not in `RECORDS`, not in the annotations — so nothing warns a user that 27
      of the 30 subjects contributed between two and four nights. 18 of those
      subjects, **49 of the 70 records**, have recordings on *both* sides of the
      challenge's a/b/c vs x division. ECGBench therefore does **not** adopt that
      division: `patient_id_column` is a reconstructed `subject_id` and folds are
      grouped on it. `challenge_set` survives as a label so the original 2000
      result stays reproducible, with the leak stated rather than inherited.

      **Two pairs of records are the same recording, bit for bit.** `x35` is `x22`
      shifted by 40 s and `c06` is `c05` shifted by 80 s. Both are kept — each is
      an official record with its own official annotations — and the grouping puts
      each pair in one fold. Section below.

      **Records are whole nights: 2,430,000 to 3,462,000 samples.** Batching needs
      a `window=(start, length)`, which is read at load time rather than cropped
      afterwards. Length is *not* uniform — 6.75 h (`x17`) to 9.62 h (`a12`) — so
      a window sized for one record need not fit another.

      **The single channel is not a named lead.** All 70 headers call it `ECG` and
      the release documents no electrode placement anywhere, so `leads=["ECG"]`
      selects a channel position. Do not stack it with a 12-lead dataset's lead I
      or II by name.

  - type: table
    title: "The three classes, recomputed from the annotation files"
    headers: ["Class", "Records", "Subjects", "Hours", "Minutes", "Apnea minutes", "Apnea %", "AHI range"]
    rows:
      - ["A — apnea", "40", "18", "338.77", "20,232", "12,544", "62.0", "14.0 – 93.5"]
      - ["B — borderline", "10", "6", "79.98", "4,781", "495", "10.4", "0.13 – 33.0"]
      - ["C — control", "20", "9", "154.93", "9,300", "25", "0.3", "0.00 – 0.38"]
      - ["**total**", "**70**", "**30**", "**573.68**", "**34,313**", "**13,064**", "**38.1**", "0.00 – 93.5"]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the 70 headers, the 70 `.apn`
      and `.qrs` annotation files and `additional-information.txt`, after
      verifying the shipped data against the release's own `SHA256SUMS.txt` —
      **all 383 listed data files match**. The only listed file absent from this
      copy is `annotations.shtml`, a documentation page.

      **The 35 `x*.apn` files are deliberately not in that checksum list.**
      PhysioNet withheld the test set's answers for the challenge and posted them
      on 2020-06-01, a year after the 2019 checksum file was written. They are
      genuine, and they are what makes all 70 records usable rather than 35.

      **Subjects: 30, and the database is usually described as having 32.** The
      four demographic fields published per record (age, sex, height, weight) take
      exactly **32** distinct values across the 70 records, which is the figure
      the literature quotes. Two of those 32 groups then merge, because two pairs
      of records turn out to be the same recording — see the section below —
      leaving **30**. PhysioNet's own landing page states neither number; it says
      only "70 records".

      **Class counts are derived, and they reproduce the release exactly.** The
      release encodes the class in the record *name* for the learning set only
      (a01–a20 apnea, b01–b05 borderline, c01–c10 control) and the withheld test
      records are all named `xNN`. Applying PhysioNet's stated criterion to the
      `.apn` counts — control under 5 apnea minutes, apnea 100 or more,
      borderline between — reproduces **all 35 learning-set letters with no
      exceptions**, which is what licenses applying it to the x records. It lands
      them on **20 A / 5 B / 10 C**: the learning set's exact composition, which
      is evidently how the test set was built.

      **`apnea_class` is a whole-night summary, not the stratification target and
      not the task.** It is a property of the *night*, not the subject: three
      subjects contributed nights that fall in different classes (`b02 b03 x16
      x21` spans A and B, `a17 x12` spans A and B, `b04 c08` spans B and C). Folds
      are balanced on it because it is the release's own taxonomy and 40/10/20
      survives 10 folds; the AHI severity bins (23 normal / 5 mild / 11 moderate /
      31 severe) do not, since a class of 5 cannot be spread over 10 folds.

      **The minute totals differ slightly from the published table, in two ways.**
      `additional-information.txt` scores 34,428 minutes and 13,066 apnea minutes;
      the annotation files hold 34,313 and 13,064.

      | Cause | Records | Effect |
      |---|---|---|
      | annotation convention — the first annotation is at second 0 and the final partial minute carries none | 68 | −1 minute each |
      | **the shipped signal is shorter than the table scores** | `c07`, `c08` | −25 and −22 minutes |
      | apnea count off by one | 2 | −2 apnea minutes total |

      `c07` holds 429 annotated minutes (7.14 h of signal) against the 454 scored,
      and `c08` 513 (8.57 h) against 535. The apnea counts for both still agree
      exactly (4 and 0), so nothing labelled apnea is missing — but a per-record
      minute total taken from the paper will not match the file. Both figures are
      exposed side by side (`n_annotated_minutes` and `published_minutes`) so the
      discrepancy stays visible rather than being silently resolved.

      **Cohort.** Ages 27–63, BMI 19.2–45.3, and **25 of the 30 subjects are men**
      (57 of the 70 records). Mean heart rate runs 48.6–83.9 bpm and SDNN
      43.7–227.1 ms over 2,270,442 machine-detected beats.

  - type: description
    title: "The release's own learning/test split is subject-leaky"
    body: |
      Apnea-ECG divides itself into a 35-record **learning set** (`a01`–`a20`,
      `b01`–`b05`, `c01`–`c10`) and 35 **test** records (`x01`–`x35`), and
      training on the first while evaluating on the second is the standard
      protocol in this literature. It puts the same subjects on both sides.

      There is no subject identifier in the release to notice this with. What
      `additional-information.txt` does publish per record is **age, sex, height
      and weight**, and those four fields take exactly 32 distinct values over the
      70 records — the subject count the database is described by. Records sharing
      all four are one subject; two verified duplicate recordings merge two
      further groups, giving 30.

      | | Subjects | Records |
      |---|---|---|
      | in the learning set only | 6 | 11 |
      | in the test set only | 6 | 10 |
      | **in both** | **18** | **49** |

      So **70% of the test records** belong to a subject the model has already
      seen during training. ECGBench sets `has_predefined_splits: false` and
      groups its own folds on `subject_id`, which routes the split through
      `StratifiedGroupKFold`; no subject spans a fold. `challenge_set` is kept as
      a label column for anyone reproducing a 2000-era result, and the caveat
      belongs with the comparison.

      Over-grouping was preferred to under-grouping throughout. Two genuinely
      distinct people sharing all four coarse attributes would be merged, which
      costs a little fold granularity; a missed subject leaks, and nothing
      downstream can detect it.

  - type: table
    title: "The 30 subjects, reconstructed"
    headers: ["Records", "Nights", "Challenge set", "Class", "AHI", "Age", "Sex", "BMI", "Fold"]
    rows:
      - ["a05 a10 a20 x07", "4", "**both**", "A", "21 – 41", "58", "M", "25.2", "3"]
      - ["a07 a16 x01 x30", "4", "**both**", "A", "41 – 63", "44", "M", "33.5", "4"]
      - ["a19 x05 x08 x25", "4", "**both**", "A", "34 – 48", "55", "M", "28.4", "2"]
      - ["b02 b03 x16 x21", "4", "**both**", "A B", "19 – 24", "53", "M", "27.4", "8"]
      - ["c01 x17 x22 x35", "4", "**both**", "C", "0", "31", "M", "21.9", "1"]
      - ["a08 a13 x20", "3", "**both**", "A", "42 – 43", "51", "M", "27.5", "6"]
      - ["a15 x27 x28", "3", "**both**", "A", "52 – 75", "60", "M", "36.5", "7"]
      - ["c05 c06 x33", "3", "**both**", "C", "0 – 0.25", "28", "F", "20.0", "5"]
      - ["a02 x14", "2", "**both**", "A", "69.5 – 79.5", "38", "M", "37.0", "8"]
      - ["a03 x19", "2", "**both**", "A", "39.1 – 56.2", "54", "M", "28.3", "9"]
      - ["a06 x15", "2", "**both**", "A", "15.9 – 24.7", "63", "M", "32.5", "1"]
      - ["a17 x12", "2", "**both**", "A B", "33", "40", "M", "30.0", "3"]
      - ["b01 x03", "2", "**both**", "B", "0.13 – 0.24", "44", "F", "21.8", "6"]
      - ["b05 x11", "2", "**both**", "B", "5", "52", "M", "41.7", "10"]
      - ["c03 x04", "2", "**both**", "C", "0", "39", "M", "19.2", "6"]
      - ["c04 x29", "2", "**both**", "C", "0", "41", "F", "20.1", "4"]
      - ["c07 x34", "2", "**both**", "C", "0 – 0.38", "30", "F", "19.8", "7"]
      - ["c10 x18", "2", "**both**", "C", "0", "27", "M", "21.3", "9"]
      - ["a01 a14", "2", "learning", "A", "54.7 – 69.6", "51", "M", "33.3", "5"]
      - ["a04 a12", "2", "learning", "A", "77.4 – 80.2", "52", "M", "40.4", "10"]
      - ["a09 a18", "2", "learning", "A", "31.7 – 82.4", "52", "M", "25.9", "10"]
      - ["b04 c08", "2", "learning", "B C", "0 – 0.7", "42", "M", "19.8", "2"]
      - ["c02 c09", "2", "learning", "C", "0", "37", "M", "25.6", "3"]
      - ["a11", "1", "learning", "A", "14", "58", "M", "36.5", "7"]
      - ["x06 x24", "2", "test", "C", "0", "31", "M", "22.8", "2"]
      - ["x09 x23", "2", "test", "A", "14.3 – 18.5", "43", "M", "25.5", "5"]
      - ["x13 x26", "2", "test", "A", "15.1 – 18.7", "57", "M", "33.2", "9"]
      - ["x31 x32", "2", "test", "A", "71.8 – 93.5", "29", "F", "29.9", "1"]
      - ["x02", "1", "test", "A", "37.7", "46", "M", "24.7", "8"]
      - ["x10", "1", "test", "B", "10", "39", "M", "45.3", "7"]

  - type: description
    title: "Two pairs of records are the same recording"
    body: |
      `x35` **is** `x22`, and `c06` **is** `c05` — the same night released twice
      under different names, with slightly different crop windows.

      | Duplicate | Canonical | Shift | Overlapping samples | Identical |
      |---|---|---|---|---|
      | `x35` | `x22` | 4,000 samples (40 s) | 2,883,000 | **100.000%** |
      | `c06` | `c05` | 8,000 samples (80 s) | 2,785,000 | **100.000%** |

      The comparison is on the integer ADC values, so no floating-point tolerance
      is involved. They were found by counting **exact 8-grams of RR intervals**
      shared between every pair of records in the release: these two pairs share
      4,152 and 4,006 against a background of 20–90 for ordinary pairs, two orders
      of magnitude above the noise. Confirmation came from maximum-over-lag
      cross-correlation, **0.9998** and **0.9629**, against **0.003 – 0.054** for
      every control pair — including same-subject different-night pairs, which is
      the control that makes the result mean something.

      **The demographics contradict each other across `x22`/`x35`.** The published
      table gives `x22` as 27 F, 158 cm, 53 kg and `x35` as 31 M, 184 cm, 74 kg,
      for what the waveform proves is one recording. One of those rows is wrong
      and there is no way to tell which, so the grouping unions transitively:
      `{x17, x22}` and `{c01, x35}` become one subject. `c05`/`c06` differ less
      dramatically (169 cm / 57 kg against 171 cm / 65 kg) and merge the same way.

      Both records of each pair are **kept**. Each is an official record with its
      own official annotations, and dropping one would silently diverge from the
      release. `duplicate_of` names the relationship, and because the pair shares
      a `subject_id` the split cannot place identical waveform in train and test —
      which an ungrouped split over these 70 records otherwise would.

  - type: table
    title: "Validation summary (100 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "70", "all records, with is_valid + quality_issues"]
      - ["clean", "70", "100% pass rate — nothing is excluded"]

  - type: description
    title: "Nothing fails validation, and one check needs explaining"
    body: |
      All 70 records pass every check, so `original` and `clean` hold the same 70
      rows. There are no NaN samples, no flat or all-zero channels and no
      unreadable header anywhere.

      - **`truncated_signal` is disabled**, by leaving `expected_samples` empty.
        Records run 2,430,000 (`x17`) to 3,462,000 samples (`a12`) and every one
        is a complete night, so any single threshold would drop sound records as
        truncated. Omitting the rate disables the check rather than making it fire
        — the same escape hatch `ptbdb`, `afdb` and `nsrdb` use.
      - **`amplitude_range_mv` is `[-10.24, 10.235]`**, the 12-bit rail computed
        from the hardware: baseline 0 at the declared gain of 200 adu/mV puts
        every possible sample in [−2048, 2047] adu. **Unlike NSRDB, this release
        actually reaches it** — 27 records touch the lower rail and 38 the upper,
        which is what a night of ambulatory Holter with electrode movement looks
        like.

        A bound sitting exactly on an *attained* value is the case where float32
        rounding decides whether a check fires, so the direction was checked
        rather than assumed: `float32(10.235)` is 10.234999656 and
        `float32(-10.24)` is −10.239999771, so **both rails round toward zero** and
        a railed sample compares strictly inside the bound. No slack is needed.
        (Contrast `chfdb`, whose 10.585 rail rounds *outward* to 10.585000038 and
        excluded the very record it was computed from until the bound was widened
        to 10.586.) What this check guards here is a mis-scaled copy — microvolts,
        or a re-release with a different gain — which would exceed the range by
        orders of magnitude on the first record.

  - type: description
    title: "What is in the release, and what is not in the partition"
    body: |
      `RECORDS` lists **86** names. The partition is **70**, and the other 16 are
      not additional recordings:

      | Group | Count | Contents | In the partition? |
      |---|---|---|---|
      | `a01`–`a20`, `b01`–`b05`, `c01`–`c10`, `x01`–`x35` | 70 | one channel, `ECG` | **yes** |
      | `a01r`–`a04r`, `b01r`, `c01r`–`c03r` | 8 | `Resp C`, `Resp A`, `Resp N`, `SpO2` — **no ECG at all** | no |
      | `a01er`–`a04er`, `b01er`, `c01er`–`c03er` | 8 | the plain record's own `.dat` **plus** those signals | no |

      The `*er` headers point at `a01.dat` and friends, so their ECG is not merely
      equivalent — it is the same bytes. Including them would put one recording in
      the partition twice. The `*r` records hold no ECG and could not be loaded as
      ECG at all. A record is kept only when **two independent filters agree** —
      its header declares exactly one signal named `ECG`, *and* its name matches
      `[abcx]NN` — because either alone would silently admit the wrong set if a
      re-release changed the naming or the channel layout; disagreement is an
      error rather than something resolved by preferring one filter.

      `has_respiration` flags the 8 ECG records whose companions exist, so a user
      wanting chest, abdominal and nasal respiration and SpO2 knows which to open
      directly with `wfdb`. It is derived from the companions actually present
      rather than from a constant, so a partial download says so.

      **The `.qrs` beat annotations are machine-generated and unaudited.**
      PhysioNet states they came from `sqrs125` at per-record thresholds and that
      "in no case were the annotations hand-edited". `mean_hr_bpm`, `sdnn_ms` and
      `rmssd_ms` are computed from them, over RR intervals in [0.3 s, 2.0 s], and
      describe a whole night of sleep — they are not an HRV result, and `.qrs` is
      not a beat-detection reference. The `|` markers in those files are QRS-like
      artifacts (5,346 of them) and never enter an RR interval.

  - type: description
    title: "An 'A' means apnea at the START of the minute"
    body: |
      Worth stating because the release itself got this wrong first. When the
      annotations were originally posted, PhysioNet described them as "an 'A'
      annotation indicates that apnea occurs during the following one-minute
      interval". It published a correction, and `annotations.html` now reads:

      > Each "A" annotation indicates that apnea was in progress at the
      > **beginning** of the associated minute; each "N" annotation indicates that
      > apnea was not in progress at the beginning of the associated minute.

      The first annotation sits at second 0 and describes the interval 0–59.99 s,
      the second at second 60, and so on. Some papers still quote the withdrawn
      description. Annotation positions are checked against the exact one-minute
      grid on load, because `apnea_sequence` is indexed by minute downstream and a
      gap would shift every label after it with no error raised.

  - type: description
    title: "Ten folds over 70 records, grouped on subject and balanced on class"
    body: |
      Folds are built with `StratifiedGroupKFold`, grouped on the reconstructed
      `subject_id` and stratified on `apnea_class` (40 A / 10 B / 20 C). No
      subject spans a fold, and both duplicate pairs land in one fold each.

      **Use the folds, not the default split.** ECGBench's convention is folds
      1–8 → train, 9 → val, 10 → test, and with 70 records grouped into 30
      subjects that gives `train` 58 records, `val` 6 and `test` 6:

      | Split | Records | Subjects | A / B / C | Annotated minutes | Apnea % |
      |---|---|---|---|---|---|
      | train | 58 | 24 | 32 / 8 / 18 | 28,448 | 34.8 |
      | val | 6 | 3 | 4 / 0 / 2 | 2,922 | 44.2 |
      | test | 6 | 3 | 4 / 2 / 0 | 2,943 | 63.7 |

      Six records is not an evaluation set, and the class composition of `val` and
      `test` is lumpy by arithmetic rather than by defect: 30 groups over 10 folds
      is three groups per fold, and a group can hold four records of one class. The
      minute-level positive rate varies with it, 34.8% to 63.7%. **For a real
      evaluation, cross-validate**: `split=None` with `fold_numbers=[...]` selects
      by fold from `folds.csv` and ignores the default layout. Ten-fold
      cross-validation over the whole 70 records is what this dataset's size
      supports.

      Fold sizes are 6–8 records (folds 1–3 hold 8, folds 4, 9 and 10 hold 6).

  - type: description
    title: "Overlap with other datasets in this catalogue: none"
    body: |
      No `related:` edge is declared. The only other sleep-apnea release in this
      catalogue, the **St. Vincent's / UCD Sleep Apnea Database** (`ucddb`), is a
      separate 25-subject cohort recorded in Dublin with three ECG channels at
      128 Hz in EDF — a different institution, a different decade and no shared
      recordings. There is no catalogue entry for the PhysioNet/CinC Challenge
      2000, because Apnea-ECG *is* that challenge's data rather than a bundle
      containing it.

      The overlap that does exist is **inside** this release, and it is handled by
      the subject grouping rather than by a `related:` edge: 27 of 30 subjects
      contribute multiple nights, and two pairs of records are the same recording.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset apnea_ecg --data-path /path/to/apnea-ecg/1.0.0/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # Whole-night records: a window is needed to batch at all, and because
      # window= is pushed into the reader it also avoids decoding the other 8 h.
      # 6000 samples at 100 Hz is exactly one minute — the labelled unit.
      ds = ECGDataset(
          "apnea_ecg",
          split="train",
          data_path="/path/to/apnea-ecg/1.0.0/",
          window=(0, 6000),         # minute 0
          labels=True,
      )

      len(ds)                                    # 58
      ds[0]["signal"].shape                      # torch.Size([1, 6000])
      ds[0]["record_id"]                         # 'a01'
      ds.lead_names                              # ('ECG',) — a channel position,
                                                 # not a named lead
      ds[0]["labels"]["subject_id"]              # 'subj_a01'  <- reconstructed
      ds[0]["labels"]["challenge_set"]            # 'learning'  <- NOT a split
      ds[0]["labels"]["apnea_class"]             # 'A'  (whole-night summary)
      ds[0]["labels"]["ahi"]                     # 69.6  -> 'severe'
      ds[0]["labels"]["n_apnea_minutes"]         # 470  (of 489 annotated minutes)
      ds[0]["labels"]["mean_hr_bpm"]             # 60.8
      ds[0]["labels"]["apnea_sequence"][:20]     # 'NNNNNNNNNNNNNAAAAAAA'

      # The ground truth is per minute. window=(i * 6000, 6000) returns minute i,
      # and apnea_sequence[i] labels it:
      seq = ds[0]["labels"]["apnea_sequence"]
      y0  = seq[0]  == "A"                       # False — a01 starts with 13 'N'
      y13 = seq[13] == "A"                       # True  — apnea from minute 13

      # 28,448 labelled minutes in this split, against 58 records. To train on
      # them, iterate minutes rather than records:
      #   for minute in range(n):
      #       ECGDataset("apnea_ecg", window=(minute * 6000, 6000), ...)

      # x17 is the shortest record at 2,430,000 samples (6.75 h), so a window must
      # fit inside that rather than inside a12's 3,462,000, or it raises
      # WindowOutOfRangeError naming the record and its true length.

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/apnea-ecg/1.0.0/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/C23W2R" }
      - { label: "PhysioNet/CinC Challenge 2000", url: "https://physionet.org/content/challenge-2000/1.0.0/" }
      - { label: "Annotation semantics (and the correction)", url: "https://physionet.org/content/apnea-ecg/1.0.0/annotations.html" }
---
