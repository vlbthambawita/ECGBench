---
slug: "mit-bih-atrial-fibrillation-database"
name: "MIT-BIH Atrial Fibrillation Database"
category: "two-lead"
order: 2
status: "completed"
source_url: "https://physionet.org/content/afdb/1.0.0/"
url_label: "physionet.org"
format: "2-lead (ECG1 + ECG2, unnamed) · 10 h · 250 Hz · WFDB"
patients: "25"
records: "25 (23 with signals)"
access: "open"
license: "ODC-By 1.0"
origin_institution: "Beth Israel Hospital"
origin_country: "USA"
leads: 2
paper_title: "Moody & Mark, CinC 1983"
paper_doi: "https://doi.org/10.13026/C2MW2D"
search_keywords: "mit-bih atrial fibrillation afdb af flutter burden paroxysmal usa beth israel hospital holter rhythm two-lead long-term"

sections:
  - type: description
    title: "Overview"
    body: |
      The reference dataset for **AF detection**, and for onset/offset work in
      particular. 25 long-term two-lead Holter recordings of subjects with atrial
      fibrillation, digitised at 250 Hz, each running 10 h 13.7 min, with
      **623 manually reviewed rhythm episodes** marking where AF starts and stops
      over 254.7 hours of recording.

      **Every subject here has AF, so the label that matters is burden, not
      diagnosis.** AF burden — the fraction of annotated time in atrial
      fibrillation or flutter — ranges from **0.24%** (05091, 43 s of AF in ten
      hours) to **100%** (07162 and 07859, a single episode each covering the whole
      record). That spread is the dataset: a detector can be evaluated against
      records where AF is a needle in a haystack and records where it never stops.
      There is **no non-AF control group**, so nothing here measures specificity in
      a general population.

      **Two of the 25 records ship no ECG.** 00735 and 03665 are in the release's
      own `RECORDS` file and carry real rhythm annotations, but their signal files
      were never published — `notes.txt` says "Signals unavailable". ECGBench keeps
      them, so `original` has all 25 and matches the published count, and marks
      them invalid so `clean` has the 23 that can be read. See the validation
      section.

      **The two channels are not named leads.** The headers call them `ECG1` and
      `ECG2` and state no electrode placement anywhere in the release. Unlike its
      sibling MIT-BIH Arrhythmia Database, which documents MLII/V1, this one gives
      you two channel positions and no anatomy. Do not assume they are MLII and V1.

      **Records are 10 hours long — 9,205,760 samples, about 74 MB of float32 per
      record.** Batching needs a `window=(start, length)`, which is read at load
      time rather than cropped afterwards. Length is *not* uniform: 06453 stops at
      8,325,000 samples, so a window must fit inside that to work on every record.

  - type: table
    title: "Time spent in each annotated rhythm"
    headers: ["Rhythm", "Code", "Hours", "Share", "Records containing it", "Episodes"]
    rows:
      - ["sinus or any other non-AF rhythm", "N", "152.55", "59.89%", "23", "292"]
      - ["atrial fibrillation", "AFIB", "95.13", "37.34%", "**25**", "299"]
      - ["AV junctional rhythm", "J", "5.42", "2.13%", "4", "18"]
      - ["atrial flutter", "AFL", "1.63", "0.64%", "8", "14"]
      - ["**total annotated**", "", "**254.73**", "", "25", "**623**"]

  - type: table
    title: "AF burden — how the 25 records distribute"
    headers: ["Class", "AF burden", "Records", "Which"]
    rows:
      - ["`minimal`", "under 5%", "8", "00735, 04015, 04048, 04126, 05091, 05261, 06453, 08434"]
      - ["`paroxysmal`", "5–95%", "14", "03665, 04043, 04746, 04908, 04936, 05121, 06995, 07879, 07910, 08215, 08219, 08378, 08405, 08455"]
      - ["`sustained`", "95% or more", "3", "06426, 07162, 07859"]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the 25 headers and the `.atr`
      and `.qrs` annotation files, after verifying **all 155 shipped files against
      the release's own `SHA256SUMS.txt`** — all 155 match.

      - **25 records, 23 with signals.** PhysioNet's description says 25 and that
        is what the `RECORDS` file lists; the two without a `.dat` are named in
        `notes.txt`. ECGBench reports both numbers rather than picking one.
      - **254.73 hours annotated**, of which **234.28 hours have signal behind
        them**. The gap is the two annotation-only records.
      - **1,221,559 beat annotations**, from the `.qrs` files. These are
        **unaudited automatic detections**, not reviewed ground truth: every symbol
        in every `.qrs` file is `N`, because the detector emits `N` for everything
        it finds. Manually corrected beats (`.qrsc`) exist for **two** records —
        05091, which `notes.txt` names, and 07859, whose file was added in 2014,
        long after that file was written, which is why it is not listed there.
      - **`dominant_rhythm` is not the stratification label and neither is
        `af_class`.** Dominant rhythm by duration is `N` for 14 records, `AFIB` for
        10 and `J` for 1 (03665, which spends 52% of its time in junctional
        rhythm). Folds are stratified on a *binary* cut at 20% AF burden — 14
        records above, 11 below — because `StratifiedKFold` needs at least as many
        members per class as there are folds, and with 25 records over 10 folds
        neither the 3-class `af_class` (3 `sustained`) nor `dominant_rhythm` (1 `J`)
        can be spread across them. Train on `af_burden`, `dominant_rhythm` or the
        `rhythm_secs_*` columns; `stratify_class` is for fold construction only.

      **The rhythm table measures time, not markers.** A rhythm annotation opens an
      episode that runs to the next one, so episode count and duration say
      different things — 04043 has **82** AFIB episodes covering 21.5% of its
      record, while 07162 has **one** covering 100% of its.

      **There are no demographics at all.** This release's headers carry no
      comments: no age, no sex, no medications, no clinical description, and no
      subject or tape identifier. That is why `patient_id_column` is null — one
      record per subject is the most that can be asserted, so folds are stratified
      but ungrouped. Mean heart rate, derived here from the `.qrs` RR intervals,
      runs 61.0–103.2 bpm across the 25.

  - type: description
    title: "The annotations stop before the signal does"
    body: |
      Beat annotation ends at the nominal ten hours — sample ~9,000,000 — while the
      `.dat` files hold 9,205,760 samples. So the last **823 seconds** of each
      full-length record carries signal that nobody annotated (1,692 s for 04048,
      whose detector output stops earlier still, and 100 s for the short record
      06453). A `window=` reaching into that tail returns waveform with no
      reference behind it, which is fine for unsupervised work and wrong for
      evaluation.

      `unannotated_tail_secs` in the labels reports it per record. The single
      exception is 07859's 2014 `.qrsc`, which does cover its whole record.

  - type: description
    title: "The amplitude is uncalibrated, by the headers' own declaration"
    body: |
      Every signal line in every header declares a gain of **`0`**, which is WFDB's
      code for "uncalibrated". `wfdb` therefore falls back to its default of
      200 adu/mV and reports the samples as millivolts, so ECGBench's
      `signal_unit_scale` is `1.0` and nothing is rescaled.

      PhysioNet's description says 12-bit over a ±10 mV range, which implies a true
      gain of 4096/20 = **204.8** adu/mV — 2.4% finer than the 200 that `wfdb`
      applies. ECGBench keeps the 200 for two reasons: it is what the files
      themselves lead every tool to, so published work on this database is on that
      scale and a silent 2.4% divergence would make ECGBench the odd one out; and
      204.8 comes from one line of prose, not from anything in the data. **No record
      reaches the 12-bit rail** — the extreme sample anywhere in the release is
      9.065 mV (08455, ECG1) against a full scale of 10.24 — so the files cannot
      arbitrate between the two figures.

      Waveform shape is unaffected either way. Absolute calibration carries a
      nominal 2.4% uncertainty, which matters for amplitude thresholds and not for
      rhythm.

  - type: table
    title: "Validation summary (250 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "25", "all records, with is_valid + quality_issues"]
      - ["clean", "23", "92.0% pass rate"]
      - ["excluded", "2", "00735, 03665 — signals were never released"]

  - type: description
    title: "About the excluded records, and the checks that do not fire"
    body: |
      The only two records that fail validation are the two with no signal files.
      Their headers read `00735 0 250 0` — zero signals, zero samples — and
      `wfdb.rdrecord` rejects that with `sampto must be greater than sampfrom`,
      which is what lands in `quality_issues`. It means "this header declares an
      empty record", not "this file is damaged". Both records' **labels are real**
      and available through `load_labels`; only their waveforms do not exist.

      One consequence is worth stating plainly: **the `original` version is not
      iterable for this dataset.** There is no signal to return for those two
      records, so `ds[i]` raises `WindowOutOfRangeError` ("has 0 samples") and any
      `DataLoader` over `original` fails on the batch containing them. Take
      `original` to see what was excluded and why; take `clean` for anything that
      reads waveforms. Every other ECGBench dataset's `original` version holds
      records that are *flagged* but readable — this one does not.

      Nothing else fails, and two checks are worth explaining because they *cannot*:

      - **`truncated_signal` is disabled**, by leaving `expected_samples` empty.
        06453 holds 8,325,000 samples against the other 22 records' 9,205,760, and
        `notes.txt` explains why — "Recording ends after about 9 hours, 15
        minutes". A 9,205,760 threshold would drop a sound record as truncated.
      - **`amplitude_range_mv` is `[-10.24, 10.235]`**, the 12-bit rail computed
        from the hardware (`adc_zero` 0 at a gain of 200 puts every possible sample
        in [−2048, 2047] adu). Unlike the MIT-BIH Arrhythmia Database, **nothing in
        this release reaches it** and no record clips, so the check cannot fire on
        AFDB 1.0.0. That is the intended state: any tighter threshold would just
        exclude the highest-amplitude records for having high amplitude. What it
        guards is a mis-scaled copy — microvolts, or a re-release with a declared
        gain — which would exceed it by orders of magnitude on the first record.

      There are **no NaN samples and no flat or all-zero leads** anywhere in the 23
      readable records.

  - type: description
    title: "Record ids are zero-padded, and that is load-bearing"
    body: |
      `00735` is not `735`. This is the first dataset in ECGBench whose record ids
      do not survive a naive CSV round-trip, and adding it is what put
      `DatasetConfig.identifier_dtypes()` in front of every metadata and fold-CSV
      read. Read as numbers, the ids lose their leading zeros, the label join
      misses, and `data_path / "735"` is not a file — so every record fails
      `corrupt_header` for a reason nothing in the traceback mentions. If you read
      the published fold CSVs yourself, pass `dtype={"record_name": str,
      "signal_path": str}`.

  - type: description
    title: "What is not in this dataset"
    body: |
      - **`old/` (25 `.atr` files).** The pre-2001 revisions of every rhythm
        annotation file, under the same names as the current ones. ECGBench reads
        annotations from the dataset root only, and takes its record list from the
        shipped `RECORDS` file, so these cannot leak in.
      - **`.hea-`, `.atr-`, `.qrs-` files in the root.** Superseded copies, ignored
        for the same reason. 07859's `.qrs` was itself revised in 2014, and
        `07859.qrs-` is what it replaced.

      The shipped `ANNOTATORS` file lists three annotators — `atr` (reference
      rhythm), `qrs` (unaudited beats) and `qrsc` (corrected beats) — and ECGBench
      exposes all three.

      **No recordings are shared with the MIT-BIH Arrhythmia Database**, despite
      the shared institution. Checked from the annotation files rather than assumed:
      neither release carries a subject identifier that would join, so the check
      compared RR-interval sequences, which are commensurable across the two
      sampling rates. At a signature of 20 consecutive RR intervals quantised to
      8 ms, **0 of MITDB's 48 records share a single sequence with any AFDB
      record** — against a positive control that re-finds an AFDB record in the
      AFDB pool at 100% while cross-record leakage between known-distinct AFDB
      records stays at 0.005%. Subject-level overlap cannot be checked at all,
      because AFDB ships no subject identifiers.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset afdb --data-path /path/to/afdb/1.0.0/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # 9,205,760-sample records: a window is needed to batch at all, and because
      # window= is pushed into the reader it also avoids decoding the other 10 h.
      ds = ECGDataset(
          "afdb",
          split="train",
          data_path="/path/to/afdb/1.0.0/",
          window=(0, 2500),        # first 10 s at 250 Hz
          labels=True,
      )

      len(ds)                                   # 19
      ds[0]["signal"].shape                     # torch.Size([2, 2500])
      ds[0]["record_id"]                        # '04043' — a string, zero-padded
      ds.lead_names                              # ('ECG1', 'ECG2') — channel positions,
                                                 # not named leads
      ds[0]["labels"]["af_burden"]              # 0.2154 — 21.5% of the record in AF
      ds[0]["labels"]["af_class"]               # 'paroxysmal'
      ds[0]["labels"]["dominant_rhythm"]        # 'N'
      ds[0]["labels"]["n_episodes_AFIB"]        # 82 — the most fragmented record here
      ds[0]["labels"]["n_beats"]                # 61915 (unaudited .qrs detections)

      # AF burden across the split, straight off the reference annotations:
      ds.labels_df["af_burden"].describe()      # min 0.0024, max 1.0

      # 06453 is the short record: a window must fit inside 8,325,000 samples,
      # not 9,205,760, or it raises WindowOutOfRangeError naming the record.

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/afdb/1.0.0/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/C2MW2D" }
      - { label: "Release notes (notes.txt)", url: "https://physionet.org/content/afdb/1.0.0/notes.txt" }
      - { label: "PhysioBank rhythm annotation definitions", url: "https://archive.physionet.org/physiobank/annotations.shtml" }
---
