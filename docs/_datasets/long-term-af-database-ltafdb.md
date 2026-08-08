---
slug: "long-term-af-database-ltafdb"
name: "Long-Term AF Database (LTAFDB)"
category: "two-lead"
order: 3
status: "completed"
source_url: "https://physionet.org/content/ltafdb/1.0.0/"
url_label: "physionet.org"
format: "2-lead (both named ECG) · 6.1–26.4 h · 128 Hz · WFDB"
patients: "84"
records: "84"
access: "open"
license: "ODC-By 1.0"
origin_institution: "Northwestern University; MEDICALgorithmics"
origin_country: "USA / Poland"
leads: 2
paper_title: "Petrutiu et al., Europace 2007"
paper_doi: "https://doi.org/10.13026/C2QG6Q"
search_keywords: "long-term af ltafdb atrial fibrillation paroxysmal sustained northwestern usa poland medicalgorithmics holter 24h af burden af termination reference beat annotations"

sections:
  - type: description
    title: "Overview"
    body: |
      **1,960.6 hours of annotated two-lead Holter from 84 subjects with atrial
      fibrillation** — about eight times the recorded time of the MIT-BIH Atrial
      Fibrillation Database, and the largest annotated AF recording set in this
      catalogue. Records run 6.1 to 26.4 hours, typically 24–25, digitised at
      128 Hz.

      **The annotations are what make it worth having.** The `.atr` files hold
      **8,995,973 typed beat annotations** and **53,704 rhythm episodes in nine
      codes**, produced by MEDICALgorithmics' PocketECG algorithm and then
      manually verified by their ECG technicians. They were contributed to
      PhysioNet in 2012, four years after the signals. AFDB, by contrast, ships
      623 rhythm episodes and *unaudited* beats. So AFDB is the AF-detection
      benchmark and this is what you use when you need reference beats at scale,
      or AF burden measured over a whole day rather than ten hours.

      **AF burden is the label, not a diagnosis.** 83 of the 84 subjects have
      annotated AF, and burden runs the full range from **0%** (record 30, the
      only AF-free record and also the shortest at 6.1 h) to **100%**. The
      distribution is strongly bimodal — 33 records are in AF essentially
      throughout and 18 spend under 5% of the recording there — which is why
      `af_class` splits three ways rather than two.

      **There is no atrial flutter code.** AFDB annotates `N`/`AFIB`/`AFL`/`J`;
      this release annotates nine codes and `AFL` is not among them, while
      ventricular bigeminy, trigeminy, atrial bigeminy, sinus bradycardia, SVT,
      VT and idioventricular rhythm are. `af_burden` here is AFIB alone, so the
      two databases' identically named columns are computed over different code
      sets.

      **The two channels are not named leads — and unlike AFDB they are not even
      numbered.** Every header calls both channels `ECG`, the same string twice,
      and states no electrode placement anywhere in the release. Two identically
      named channels cannot be told apart by name, so ECGBench declares the
      positional names `ECG1`/`ECG2` and says so here. Do not read them as
      MLII/V1 by analogy with the MIT-BIH Arrhythmia Database.

      **Records are a full day long**, so batching needs `window=(start, length)`,
      which is read at load time rather than cropped afterwards. Length is very
      much not uniform: **55 distinct record lengths**, from 2,826,240 samples
      (record 30) to 12,142,080 (record 70), so a window must fit inside the
      shortest.

  - type: table
    title: "Time spent in each annotated rhythm"
    headers: ["Rhythm", "Code", "Hours", "Share", "Records containing it", "Episodes"]
    rows:
      - ["atrial fibrillation", "`AFIB`", "1,030.89", "53.28%", "**83**", "7,358"]
      - ["sinus rhythm or any other unlisted rhythm", "`N`", "828.55", "42.82%", "53", "22,834"]
      - ["sinus bradycardia", "`SBR`", "55.58", "2.87%", "35", "11,326"]
      - ["atrial bigeminy", "`AB`", "9.66", "0.50%", "46", "4,472"]
      - ["ventricular bigeminy", "`B`", "4.67", "0.24%", "21", "2,696"]
      - ["supraventricular tachyarrhythmia", "`SVTA`", "3.33", "0.17%", "45", "3,268"]
      - ["ventricular trigeminy", "`T`", "1.71", "0.09%", "22", "785"]
      - ["ventricular tachycardia", "`VT`", "0.40", "0.02%", "34", "828"]
      - ["idioventricular rhythm", "`IVR`", "0.13", "0.01%", "4", "137"]
      - ["**total annotated**", "", "**1,934.92**", "", "84", "**53,704**"]

  - type: table
    title: "Reference beat annotations (.atr)"
    headers: ["Symbol", "Beat type", "Count", "Share", "Records containing it"]
    rows:
      - ["`N`", "normal beat", "8,710,873", "96.831%", "84"]
      - ["`A`", "atrial premature beat", "152,332", "1.693%", "53"]
      - ["`V`", "premature ventricular contraction", "132,679", "1.475%", "84"]
      - ["`Q`", "unclassifiable beat", "89", "0.001%", "6"]
      - ["**total**", "", "**8,995,973**", "", "84"]

  - type: table
    title: "AF burden — how the 84 records distribute"
    headers: ["Class", "AF burden", "Records", "Observed range"]
    rows:
      - ["`minimal`", "under 5%", "18", "0.0000 – 0.0490"]
      - ["`paroxysmal`", "5–95%", "33", "0.0550 – 0.9306"]
      - ["`sustained`", "95% or more", "33", "0.9511 – 1.0000"]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the 84 headers and the `.atr`
      and `.qrs` annotation files, after verifying **all 339 shipped files against
      the release's own `SHA256SUMS.txt`** — all 339 match.

      They were then checked against the release's own summary tables
      (`tables.shtml`), cell by cell: **all 336 beat counts** (84 records ×
      N/A/V/Q) and **all 756 rhythm cells** (84 × nine codes, episode count and
      duration) agree. Two differences are worth recording:

      - **Record 20's published AFIB duration is 1.1 s short.** `tables.shtml`
        gives 24:19:08 (87,548 s) where the shipped header yields 87,549.1. The
        landing page thanks Mariano Llamedo Soria "for reporting an error in the
        original version of `20.hea`, and for providing a correction incorporated
        in the current version" — the table was generated against the
        pre-correction header, which was 144 samples shorter. The current header
        is the authority, so ECGBench reports 87,549.1.
      - **Record 30's comment-annotation count.** Its `.atr` ends with the WFDB
        file terminator at sample 4,198,064, while the record holds only
        2,826,240 — 2.98 hours past the end of the data. That marker's position
        is not a claim about the signal; ECGBench excludes out-of-range
        annotations from every measurement, which is what makes record 30's count
        agree with the published 1 rather than reading 2.

      Two more things the numbers do not say on their own:

      - **`af_burden`'s denominator is annotated rhythm time, not record time.**
        The first rhythm annotation sits a little way into each record — 47.5 s
        into record 20 — and nothing classifies what precedes it. Across the
        release that lead-in is 25.7 of the 1,960.6 recorded hours, which is the
        gap between the 1,934.92 h in the rhythm table and the 1,960.6 h of
        signal. `rhythm_annotated_secs` states the denominator per record.
      - **The rhythm table measures time, not markers.** A rhythm annotation opens
        an episode that runs to the next one, so episode count and duration say
        different things — record 74 has **1,044** AFIB episodes covering 23.5 h,
        while record 12 has **one** covering 24.1 h.

      **There are no demographics at all.** This release's headers carry no
      comment lines: no age, no sex, no medications, no clinical description, and
      no subject or tape identifier. That is why `patient_id_column` is null —
      one record per subject is the most that can be asserted, so folds are
      stratified but ungrouped. Mean heart rate, derived here from the `.atr` RR
      intervals, runs 40.4–129.8 bpm across the 84.

  - type: description
    title: "Reference beats, unaudited detections, and the T that means two things"
    body: |
      The shipped `ANNOTATORS` file lists two annotators, and they are not
      interchangeable:

      - **`.atr` — reference beat and rhythm annotations.** Beats are typed
        (`N`/`A`/`V`/`Q`), rhythm changes carry one of nine codes, and `"` comment
        markers flag missed beats (651) and pauses (5,224). Algorithm output,
        manually verified. Everything on this page is derived from these.
      - **`.qrs` — unaudited `sqrs` detections.** 8,611,567 of them, every one
        labelled `N` whatever it actually is, plus 2,549 artifact markers. Useful
        as a detector baseline; **never add them to the `.atr` counts.**

      The `.qrs` files also carry **81 hand-placed `T` markers** recording
      spontaneous ends of AF episodes lasting a minute or more, inserted by Steven
      Swiryn and George Moody. The AF Termination Challenge Database's 80
      one-minute excerpts were cut around them. They exist **only in records
      00–75** — all 24 records that carry any are in that range, and the 100- and
      200-series carry none — so a model trained on terminations sees roughly half
      the database.

      **`T` means something else entirely in an `.atr` rhythm code: ventricular
      trigeminy.** Same letter, different annotator, unrelated meanings.

  - type: description
    title: "The signal outlasts the annotation, by hours in some records"
    body: |
      Both annotators stop at the same place in each record, so this is a property
      of the recording rather than of one annotator. The median record's beat
      annotations stop **4.9 s** from the end — but:

      | Beat annotation stops before the end by | Records |
      |---|---|
      | more than 10 minutes | 35 |
      | more than 1 hour | 17 |
      | **8.05 hours** (record 117, a third of the record) | 1 |

      `unannotated_tail_secs` in the labels reports it per record. A `window=`
      reaching into that tail returns waveform with no reference behind it, which
      is fine for unsupervised work and wrong for evaluation.

      Rhythm *durations* nonetheless run to the end of the signal, because the
      last episode has no annotation after it to close it and PhysioNet's own
      tables close it at the record end. ECGBench follows that convention — which
      is what makes its figures reproduce those tables — so a record like 117
      attributes its final 8 unannotated hours to whatever rhythm was running at
      15.9 h.

  - type: description
    title: "The gains are real — and one of them is anomalous"
    body: |
      Unlike AFDB, whose every header declares a gain of `0` (WFDB's
      "uncalibrated") and leans on wfdb's 200 adu/mV fallback, LTAFDB's headers
      carry **50 distinct measured gains**, sometimes different for the two
      channels of one record (record 100 is 88.968 and 131.062). wfdb applies each
      header's own gain, so samples arrive in genuine millivolts and
      `signal_unit_scale` is `1.0`. There is no calibration argument to have here.

      **With one exception, which ECGBench flags and does not correct.** Record
      62's ECG1 declares **1123.6 adu/mV**, where the other 167 signal lines run
      75.0188 to 222.222 — a 5.5× outlier with nothing in between. Its raw swing
      is entirely normal (2,756 adu peak-to-peak, against a release median of
      1,220 and a maximum of 2,808), so that gain turns the largest raw excursion
      in the release into its smallest calibrated one: **2.45 mV peak-to-peak
      against a release median of 6.65**.

      ECGBench reports what the header declares, because a silently corrected
      record would disagree with every other tool reading the same file. But
      anything comparing **absolute amplitudes across records** should exclude
      record 62's ECG1 or rescale it. Its ECG2 (202.429) is unaffected, and
      `adc_gains` in the labels carries the pair for every record.

  - type: table
    title: "Validation summary (128 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "84", "all records, with is_valid + quality_issues"]
      - ["clean", "84", "100% pass rate — every record loads"]

  - type: description
    title: "About the checks that cannot fire"
    body: |
      Nothing fails validation: all 84 records read, none has a NaN sample, and
      no lead is flat or all-zero. Two checks are worth explaining because they
      *cannot* fire on this release:

      - **`truncated_signal` is disabled**, by leaving `expected_samples` empty.
        There are 55 distinct record lengths spanning 2,826,240 to 12,142,080
        samples, so no single threshold can distinguish a truncated record from a
        short one, and any threshold would drop sound records. Every `.dat` file
        holds exactly the number of samples its header declares — checked for all
        168 channels.
      - **`amplitude_range_mv` is `[-27.3, 27.3]`**, the 12-bit rail computed
        from the hardware. PhysioNet describes 12 bits over a 20 mV range, so a
        sample is confined to ±2048 adu — and the raw files bear that out, with
        every sample of all 168 channels inside [−1497, 1398]. In millivolts that
        rail moves with the header gain, and the loosest gain (75.0188 adu/mV)
        puts it at ±27.3 mV; a single range has to accommodate the loosest record
        or it fires on a sound one. The observed extreme anywhere in the release
        is **−10.599 mV** (record 100, ECG1) to **+11.583 mV** (record 34, ECG1),
        so nothing comes close. What the check guards is a mis-scaled copy —
        microvolts, or a re-release with the gains dropped — which would exceed it
        by orders of magnitude on the first record.

  - type: description
    title: "Record ids are zero-padded, and that is load-bearing"
    body: |
      `00` is not `0`. Seven of the 84 record names begin with a zero — `00`,
      `01`, `03`, `05`, `06`, `07`, `08` — and pandas reads a column of digits as
      int64, which strips them. The id then no longer names a record, the label
      join misses, and `data_path / "0"` is not a file, so **every** record fails
      `corrupt_header` for a reason nothing in the traceback mentions.

      This is the second ECGBench dataset to need `zero_padded_identifiers: true`,
      after AFDB. If you read the published fold CSVs yourself, pass
      `dtype={"record_name": str, "signal_path": str}`.

  - type: description
    title: "Overlap with the other AF databases: none found"
    body: |
      LTAFDB and the **MIT-BIH Atrial Fibrillation Database** are both long-term
      two-lead Holter recordings of AF subjects, and neither ships a subject
      identifier that would join, so the question was settled from the annotation
      files rather than assumed. RR intervals in seconds are commensurable across
      sampling rates (128 Hz here, 250 Hz for AFDB, 360 Hz for MITDB), so the
      check compares **sequences of 20 consecutive RR intervals quantised to
      8 ms**, on two half-bin-shifted grids.

      Against controls that make a null result mean something — a positive control
      re-finding each LTAFDB record in itself at **100%**, and a negative control
      of each record against the pool of the other 83 known-distinct subjects at a
      median of **0%** and a maximum of 0.15% — the result is:

      - **0 of 84 LTAFDB records** share any sequence with the MIT-BIH AFDB pool
        (highest 0.04%);
      - **0 of 84** share any with the MIT-BIH Arrhythmia Database pool (highest
        0.000%).

      No `related:` edge is declared on those grounds. The **AF Termination
      Challenge Database** *is* a genuine derivative — its 80 one-minute excerpts
      are cut from records 00–75 of this release — but it is not in this catalogue,
      so there is nothing to link it to yet.

      One limitation is worth stating rather than glossing: the RR signature
      survives *refinement* of annotations but not *re-detection*, so a shared
      recording annotated by two genuinely different detectors could evade it.
      Subject-level overlap cannot be checked at all, because none of these
      releases ships a subject identifier.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset ltafdb --data-path /path/to/ltafdb/1.0.0/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # Day-long records: a window is needed to batch at all, and because window=
      # is pushed into the reader it also avoids decoding the other 24 hours.
      ds = ECGDataset(
          "ltafdb",
          split="train",
          data_path="/path/to/ltafdb/1.0.0/",
          window=(0, 1280),        # first 10 s at 128 Hz
          labels=True,
      )

      len(ds)                                   # 68
      ds[0]["signal"].shape                     # torch.Size([2, 1280])
      ds[0]["record_id"]                        # '01' — a string, zero-padded
      ds.lead_names                             # ('ECG1', 'ECG2') — channel positions
                                                # that ECGBench assigns; the headers
                                                # call both channels 'ECG'
      ds[0]["labels"]["af_burden"]              # 0.7891 — 78.9% of the record in AF
      ds[0]["labels"]["af_class"]               # 'paroxysmal'
      ds[0]["labels"]["dominant_rhythm"]        # 'AFIB'
      ds[0]["labels"]["n_episodes_AFIB"]        # 53
      ds[0]["labels"]["n_beats"]                # 90546 typed reference beats (.atr)
      ds[0]["labels"]["adc_gains"]              # '202.429|202.429' — measured, per channel
      ds[0]["labels"]["record_hours"]           # 20.57
      ds[0]["labels"]["unannotated_tail_secs"]  # 3.93 — this one annotates to the end

      # AF burden across the split, straight off the reference annotations:
      ds.labels_df["af_burden"].describe()      # min 0.0007, median 0.417, max 1.0
      ds.labels_df["af_class"].value_counts()   # paroxysmal 27, sustained 27, minimal 14

      # Record 30 is the shortest at 2,826,240 samples, so a window must fit
      # inside that rather than inside a median record's 11,059,200, or it raises
      # WindowOutOfRangeError naming the record and its true length. The binding
      # limit is the shortest record in YOUR split — 8,371,200 in train.
---
