---
slug: "bidmc-congestive-heart-failure-database"
name: "BIDMC Congestive Heart Failure Database"
category: "two-lead"
order: 7
status: "completed"
source_url: "https://physionet.org/content/chfdb/1.0.0/"
url_label: "physionet.org"
format: "2-lead (ECG1 + ECG2, unnamed) · 19.8–20.0 h · 250 Hz · WFDB"
patients: "15"
records: "15"
access: "open"
license: "ODC-By 1.0"
origin_institution: "Beth Israel Hospital"
origin_country: "USA — Boston, MA"
leads: 2
paper_title: "Baim et al., Survival of patients with severe congestive heart failure treated with oral milrinone, J Am Coll Cardiol 1986"
paper_doi: "https://doi.org/10.13026/C29G60"
search_keywords: "bidmc congestive heart failure chfdb nyha class iii iv severe usa boston beth israel deaconess medical center holter 20h milrinone ventricular ectopy pvc r-on-t atrial fibrillation heart rate variability hrv sdnn rmssd unaudited beat annotations two-lead long-term"

sections:
  - type: description
    title: "Overview"
    body: |
      **The severe-heart-failure cohort.** 15 twenty-hour two-lead Holter
      recordings of subjects with **NYHA class III–IV** congestive heart failure —
      11 men aged 22 to 71 and 4 women aged 54 to 63 — made at Boston's Beth
      Israel Hospital while they were on conventional therapy, before entering the
      oral milrinone trial Baim et al. reported in 1986. **298.9 hours** of signal
      at 250 Hz and **1,622,282 detected beats**. It is the natural counterpart to
      the MIT-BIH Normal Sinus Rhythm Database's control cohort: one class, and
      every record is in it.

      **The beat annotations are unaudited machine output, and this is the first
      thing to know.** PhysioNet states it plainly — "Annotation files (with the
      suffix `.ecg`) were prepared using an automated detector and **have not been
      corrected manually**" — and the shipped `ANNOTATORS` file agrees. The `.ecg`
      extension is itself the tell: every audited MIT-BIH database in this
      catalogue uses `.atr`. So unlike `mitdb`, `nsrdb`, `svdb` and `edb`, nothing
      here is a cardiologist reference standard. Every beat, ectopy and HRV figure
      on this page describes what one 1980s detector reported. It has its own
      section below.

      **Ventricular ectopy is heavy and enormously uneven — the real signal here.**
      38,524 of the 1,622,282 beats are ventricular, **2.37%**, against NSRDB's
      0.0015%. But the spread across records is three orders of magnitude:
      **chf02 is 20.52% ventricular** and **chf12 is 0.017%**. A per-record metric
      is not comparable across this database without controlling for it.

      **`r` is a ventricular beat, and it outnumbers `V` in 9 of the 15 records.**
      10,353 annotations are `r`, R-on-T premature ventricular contractions, which
      AAMI EC57 classes as ventricular. A pipeline counting only `V` therefore
      undercounts ventricular ectopy across most of this release — use the `aami_*`
      columns or `veb_fraction`. There are no fusion beats at all.

      **Records are ~20 h: 17,789,952 to 17,998,848 samples, ~142 MB of float32
      each.** Batching needs a `window=(start, length)`, which is read at load time
      rather than cropped afterwards. Length varies by only 232 s, so one window
      fits every record — but it must fit the shortest, chf06.

      **The two channels are not named leads.** The headers call them `ECG1` and
      `ECG2` and the release states no electrode placement anywhere. Its siblings
      from the same hospital — the MIT-BIH Arrhythmia Database — do document
      MLII/V1; this one gives you two channel positions and no anatomy. Do not
      carry the `mitdb` naming across.

  - type: table
    title: "The 15 records, recomputed from the files"
    headers: ["Record", "Age", "Sex", "Hours", "Beats", "Vent.", "Vent. %", "SVEB", "AF %", "Mean HR", "SDNN ms", "Fold"]
    rows:
      - ["chf01", "71", "M", "19.994", "75,546", "268", "0.35", "293", "0.23", "63.0", "87.8", "5"]
      - ["chf02", "61", "F", "19.770", "114,548", "**23,510**", "**20.52**", "3", "0.00", "99.8", "**118.3**", "4"]
      - ["chf03", "63", "M", "19.999", "81,301", "1,993", "2.45", "461", "0.00", "67.7", "85.5", "1"]
      - ["chf04", "54", "M", "19.999", "112,366", "1,312", "1.17", "280", "0.00", "93.6", "59.2", "9 (val)"]
      - ["chf05", "59", "F", "19.780", "119,153", "584", "0.49", "104", "0.00", "100.4", "52.2", "3"]
      - ["chf06", "**?**", "M", "**19.767**", "118,384", "3,407", "2.88", "**3,083**", "**80.46**", "99.8", "97.1", "10 (test)"]
      - ["chf07", "48", "M", "19.999", "92,584", "1,863", "2.01", "194", "0.00", "77.1", "84.3", "2"]
      - ["chf08", "51", "M", "19.999", "90,759", "1,021", "1.12", "128", "0.00", "75.6", "65.9", "1"]
      - ["chf09", "63", "F", "19.774", "115,052", "735", "0.64", "161", "0.00", "97.0", "38.5", "2"]
      - ["chf10", "**22**", "M", "19.995", "**147,301**", "68", "0.05", "410", "0.86", "**123.3**", "38.4", "8"]
      - ["chf11", "54", "F", "19.999", "115,639", "487", "0.42", "51", "0.00", "96.4", "85.0", "5"]
      - ["chf12", "61", "M", "19.825", "115,127", "**19**", "**0.02**", "16", "0.00", "96.8", "100.5", "4"]
      - ["chf13", "63", "M", "19.996", "115,650", "404", "0.35", "11", "0.00", "96.4", "**28.3**", "7"]
      - ["chf14", "61", "M", "19.999", "93,674", "103", "0.11", "38", "0.02", "78.1", "71.0", "3"]
      - ["chf15", "53", "M", "19.993", "115,198", "2,750", "2.39", "81", "0.00", "96.1", "78.1", "6"]
      - ["**total**", "22–71", "11 M / 4 F", "**298.89**", "**1,622,282**", "**38,524**", "**2.37**", "**5,314**", "**5.39**", "63.0–123.3", "28.3–118.3", "1–10"]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the 15 headers and `.ecg` files,
      after verifying the shipped data against the release's own `SHA256SUMS.txt` —
      **all 64 files match**. Unlike NSRDB's copy, that total *includes* the
      superseded backups PhysioNet keeps beside the current revisions: 15 `.hea-`
      files predating the 2012 revision that added the `ECG1`/`ECG2` signal
      descriptions, and 2 `.ecg-` files for chf02 and chf04, whose annotations were
      regenerated in 2003. The record list comes from the shipped `RECORDS` file, so
      none of them can enter the partition.

      **The published cohort description reproduces exactly.** PhysioNet states
      "11 men, aged 22 to 71, and 4 women, aged 54 to 63", and the header comments
      give precisely that — men at 22, 48, 51, 53, 54, 61, 61, 63, 63, 71 and one
      unstated, women at 54, 59, 61, 63. There is no changelog and no discrepancy
      to explain. **One age is genuinely missing**: chf06's header records
      `Age: ?`, which ECGBench keeps as NaN rather than 0.

      **Vent. is the AAMI ventricular class, not the `V` symbol.** It sums `V`
      (26,166 release-wide), `r` (10,353) and `E` (5). The `r` beats are R-on-T
      PVCs and are ventricular under AAMI EC57; because they outnumber plain `V` in
      9 of the 15 records, the `V` column alone would understate ventricular ectopy
      for most of the database. The full symbol breakdown is 1,578,151 `N`, 26,166
      `V`, 10,353 `r`, 5,314 `S`, 293 `Q` and 5 `E`, plus 258 `+` rhythm markers
      that are **not** beats.

      **Vent. % is a fraction of that record's beats; AF % is a fraction of its
      duration.** They are not the same denominator, and the AF column carries a
      further caveat given its own section below.

      **The HRV figures are descriptive, not a result.** `mean_hr_bpm`, `sdnn_ms`
      and `rmssd_ms` come from RR intervals in [0.3 s, 2.0 s]; 612 intervals are
      rejected across the release. Two caveats compound here in a way they do not
      for NSRDB: the beats are uncorrected machine output, and these are
      whole-record summaries over ~20 h of activity and sleep in subjects with
      severe heart failure and heavy ectopy. A real HRV analysis would segment,
      exclude ectopic couplings, and not use these labels at all.

      **`cohort_label` and `nyha_class` are constants, and neither is the
      stratification label.** They record what the release asserts of the cohort so
      that a user combining this database with a normal or mixed-arrhythmia one has
      a record-level class to join on. Folds are stratified on **sex** (11 M / 4 F)
      — see the fold section. Train on `veb_fraction`, `af_fraction`,
      `mean_hr_bpm`, `sdnn_ms` or the `aami_*` counts; never on `stratify_class`.

  - type: description
    title: "The annotations were never corrected, and that changes what they are for"
    body: |
      This is the most important difference between this database and the other
      MIT-BIH-family releases in the catalogue, and it is easy to miss because the
      annotations look identical in structure to `mitdb`'s.

      | | Annotator | Status |
      |---|---|---|
      | MIT-BIH Arrhythmia | `.atr` | cardiologist-audited reference |
      | MIT-BIH NSR | `.atr` | reference beat + quality annotations |
      | MIT-BIH SVDB | `.atr` | reference |
      | European ST-T | `.atr` | reference, cardiologist-annotated |
      | **BIDMC CHF** | **`.ecg`** | **automated detector, never manually corrected** |

      PhysioNet's wording is unambiguous, and the shipped `ANNOTATORS` file
      describes the contents as "unaudited beat annotations from an automated
      detector". Two consequences worth stating rather than leaving to be inferred:

      - **Do not train or evaluate a beat classifier on these labels.** A model
        scored against them is being scored against a 1980s detector, and the
        conditions under which such detectors mislabel most — heavy ventricular
        ectopy and atrial fibrillation — are exactly what this cohort has.
      - **The ectopy rates are not clinical facts.** They are useful for describing
        a recording, for stratifying, and as weak supervision a human would have to
        confirm. They should not be quoted as this cohort's arrhythmia burden.

      What the release *is* good for is unaffected: 298.9 hours of severe-CHF
      waveform, a severity-matched positive cohort, and a self-supervised or
      unsupervised corpus.

  - type: description
    title: "Rhythm annotation covers 4 records, and its absence is not a negative"
    body: |
      258 of the annotations are `+` rhythm markers carrying `(AF` and `(N`. They
      are very unevenly distributed:

      | Record | AF episodes | AF time | % of record |
      |---|---|---|---|
      | chf06 | 125 | 15.90 h | **80.46%** |
      | chf10 | 2 | 10.3 min | 0.86% |
      | chf01 | 1 | 2.7 min | 0.23% |
      | chf14 | 1 | 17.0 s | 0.02% |
      | the other 11 | **none at all** | — | — |

      **For those 11 records, `af_secs == 0` means the rhythm was never assessed,
      not that there was no atrial fibrillation.** Filtering on `af_secs == 0`
      would pick them up as though they were confirmed negatives, which is why the
      labels expose **`has_rhythm_annotation`** — check it before treating a zero as
      evidence.

      chf06 has one further wrinkle. Its first `+` sits 1,757.0 s into the record
      and is `(N` — a *return* to normal rhythm, which implies the preceding 29
      minutes were AF without ever saying so. Counting that span as AF would invent
      annotation; counting it as normal would assert what the annotator
      contradicted. It is reported separately as
      `rhythm_head_unasserted_secs`, the same choice `svdb` makes for its
      signal-quality head.

      Note that chf06 is also the single record in the **test** fold, so the default
      test set is one predominantly-AF recording. For anything needing a real
      evaluation set, use cross-validation — see the fold section.

  - type: description
    title: "There is no signal-quality annotation layer at all"
    body: |
      `nsrdb`, `svdb` and `mitdb` all ship `~` signal-quality transitions with a
      per-channel bitmask, and `|` isolated-artifact markers. **This release
      contains neither — zero `~` and zero `|` across all 15 files.**

      ECGBench exposes `n_quality_changes` and `n_isolated_artifacts` as 0 so that a
      re-release adding a quality layer is visible rather than silent, but
      deliberately provides **no** `clean_secs`/`noisy_secs` columns. Deriving them
      from an absent annotation layer would assert that 298.9 hours of tape are
      clean when nobody assessed them. Judge quality from the waveform.

  - type: table
    title: "Validation summary (250 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "15", "all records, with is_valid + quality_issues"]
      - ["clean", "15", "100% pass rate — nothing is excluded"]

  - type: description
    title: "Nothing fails validation, and the amplitude bound is asymmetric on purpose"
    body: |
      All 15 records pass every check, so `original` and `clean` hold the same 15
      rows. There are no NaN samples anywhere in the 540 million read, no flat or
      all-zero leads and no unreadable header. Two checks need explaining:

      - **`truncated_signal` is disabled**, by leaving `expected_samples` empty.
        Records run 17,789,952 (chf06) to 17,998,848 samples and every one is a
        complete recording, so any single threshold would drop sound records as
        truncated. Omitting the rate disables the check rather than making it fire —
        the same escape hatch `ptbdb`, `afdb`, `ltafdb` and `nsrdb` use.
      - **`amplitude_range_mv` is `[-10.24, 10.586]`, which is not symmetric.** 29
        of the 30 channels have `adc_zero` 0, which at wfdb's fallback gain of 200
        puts them in [−2048, 2047] adu = [−10.24, 10.235] mV. **chf15's ECG2 is the
        one exception**: a baseline of −70 adu shifts its range to
        [−9.89, 10.585] mV, and that channel *actually reaches* +2047 adu — for 12
        samples, 0.0001% of the record. So the bound has to be the union of the two
        rails, or the release's own top record fails a check computed to admit it.

      **And it cannot be the exact rail, which is worth knowing before setting one
      of these anywhere else.** ECGBench loads signals as float32, and float32
      cannot represent 10.585 — the nearest value is 10.585000038146973, which is
      greater than a float64 bound of 10.585. A bound set to the attained rail
      therefore excludes the very record it was derived from; that is what the extra
      thousandth of a millivolt is for. The negative bound needs no such slack,
      because float32(−10.24) rounds toward zero.

      Observed extremes across the whole release: **−10.17 mV** (chf09 ECG1) to
      **+10.585 mV** (chf15 ECG2). Unlike NSRDB, which never exceeds half its
      declared span, this release uses nearly all of it.

  - type: description
    title: "The amplitude is uncalibrated by the headers, but the description agrees"
    body: |
      Every signal line in every header declares a gain of **`0`**, which is WFDB's
      code for "uncalibrated". `wfdb` therefore falls back to its default of 200
      adu/mV and reports the samples as millivolts, so ECGBench's
      `signal_unit_scale` is `1.0` and nothing is rescaled.

      This is the same situation as AFDB, NSRDB and LTAFDB — with the difference
      that here the default is **corroborated**. PhysioNet's description states
      "250 samples per second with 12-bit resolution over a range of ±10
      millivolts", and 12 bits at 200 adu/mV is exactly [−2048, 2047] adu =
      [−10.24, 10.235] mV. The declared gain and the described range agree, which
      is not true of AFDB (whose stated range implies 204.8 adu/mV and has to be
      reconciled) and is simply unstated for NSRDB.

      One further property of the recording chain matters for modelling: the signals
      are band-limited to roughly **0.1–40 Hz**, so despite the 250 Hz sampling rate
      there is no high-frequency content, and they are not comparable in detail with
      modern digital ECG.

  - type: description
    title: "Ten folds over 15 records, stratified on sex"
    body: |
      Three consequences of the arithmetic, stated rather than left to be
      discovered:

      - **The default split leaves one record in `val` and one in `test`.**
        ECGBench's convention is folds 1–8 → train, 9 → val, 10 → test, and 15
        records over 10 folds gives five folds of two and five of one. So `train`
        holds 13 records, `val` holds 1 (chf04) and `test` holds 1 (chf06). For
        anything needing a real evaluation set, use cross-validation:
        `split=None` with `fold_numbers=[...]` selects by fold from `folds.csv` and
        ignores the default layout.
      - **The single test record is the AF-dominant one.** chf06 lands in fold 10,
        so the default test set is 80% atrial fibrillation and is also the record
        whose age is unstated. That is chance, not a choice, and it is another
        reason to cross-validate rather than trust one held-out record.
      - **Most folds contain no woman.** The four women land in four different folds
        (chf02, chf05, chf09, chf11 in folds 4, 3, 2 and 5), so folds 9 and 10 —
        `val` and `test` — are both male. Four records cannot be spread over ten
        folds; stratification keeps them apart rather than letting them clump, which
        is the most it can do here.

      **Why sex, and not the ventricular ectopy burden?** Because
      `StratifiedKFold` requires at least one class holding `n_folds` members, and
      with 15 records over 10 folds that means a 10/5 or more lopsided cut. Ectopy
      burden is the clinically interesting axis and `svdb` does stratify on exactly
      that, in bands — here every meaningful banding fails:

      | Candidate fold axis | Class sizes | Result |
      |---|---|---|
      | Sex | 11 / 4 | **works** |
      | `veb_fraction` quartiles | 4 / 4 / 3 / 4 | raises |
      | `svdb`'s burden edges (1%, 3%, 10%) | 8 / 6 / 0 / 1 | raises |
      | Burden cut at 1% | 7 / 8 | raises |
      | Burden cut at 2% | 5 / 10 | works, but the threshold is fitted to these 15 numbers |

      Sex clears the requirement with margin, is documented by PhysioNet, and does
      not rest on the unaudited detector output every ectopy figure here comes from.
      `sklearn` warns that the smallest class has 4 members; that warning is
      expected and correct.

      Folds are **ungrouped**. The header comment holds age, sex and NYHA class and
      nothing else — no tape number, no recorder, no subject code, and not even the
      trial arm the cohort is defined by — and PhysioNet describes 15 recordings
      from 15 subjects, so one record per subject is the most that can be asserted.

  - type: description
    title: "Overlap with the other Beth Israel Holter databases: none found"
    body: |
      CHFDB, the **MIT-BIH Arrhythmia**, **Atrial Fibrillation**, **Normal Sinus
      Rhythm**, **Supraventricular Arrhythmia** and **ST Change** databases and the
      **Sudden Cardiac Death Holter Database** all come out of Boston's Beth Israel
      Hospital, and none of them ships a subject identifier that would join. So the
      question was settled from the annotation files rather than assumed. RR
      intervals in seconds are commensurable across sampling rates, so the check
      compares **sequences of 20 consecutive RR intervals quantised to 8 ms**, on
      two half-bin-shifted grids — the same method used for LTAFDB and NSRDB.

      Against controls that make a null result mean something — a positive control
      re-finding each CHFDB record in its own pool at **100%**, and a negative
      control of each record against the pool of the other 14 known-distinct
      subjects peaking at **0.0869%** — no database shares a recording:

      | Pool | Records | Highest CHFDB hit rate | At 30 intervals |
      |---|---|---|---|
      | MIT-BIH Arrhythmia | 48 | 0.0000% | — |
      | MIT-BIH NSR | 18 | 0.0026% | — |
      | MIT-BIH SVDB | 78 | 0.0095% | — |
      | MIT-BIH AFDB | 25 | 0.0244% | — |
      | Long-Term AF | 84 | 0.0411% | 0.0014% |
      | MIT-BIH ST Change | 28 | 0.0624% | 0.0062% |
      | Sudden Cardiac Death | 23 | 0.1079% | 0.0026% |

      Every figure sits at or below the 0.0869% chance-collision floor the negative
      control establishes, except Sudden Cardiac Death's 0.1079%, which marginally
      exceeds it against a 1.77-million-signature pool. Repeating the check at a
      30-interval signature settles it: a genuinely shared recording stays near 100%
      as the signature lengthens, and these decay by one to two orders of magnitude,
      which is what chance collisions do.

      No `related:` edge is declared on those grounds. One limitation is worth
      stating rather than glossing: the RR signature survives *refinement* of
      annotations but not *re-detection*, so a shared recording annotated by two
      genuinely different detectors could evade it — and this release's annotator is
      an uncorrected detector, which makes that marginally more plausible here than
      elsewhere. Subject-level overlap cannot be checked at all, because none of
      these releases ships a subject identifier.

      Note also that PhysioNet's **Congestive Heart Failure RR Interval Database**
      (`chf2db`, 29 subjects) is a *different* database, not a derived layer over
      this one, and is not in this catalogue.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset chfdb --data-path /path/to/chfdb/1.0.0/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # ~20 h records: a window is needed to batch at all, and because window= is
      # pushed into the reader it also avoids decoding the other 20 h.
      ds = ECGDataset(
          "chfdb",
          split="train",
          data_path="/path/to/chfdb/1.0.0/",
          window=(0, 2500),         # first 10 s at 250 Hz
          labels=True,
      )

      len(ds)                                  # 13
      ds[0]["signal"].shape                    # torch.Size([2, 2500])
      ds[0]["record_id"]                       # 'chf01'
      ds.lead_names                            # ('ECG1', 'ECG2') — channel positions,
                                               # not named leads
      ds[0]["labels"]["cohort_label"]          # 'severe_chf'  — all 15 records
      ds[0]["labels"]["nyha_class"]            # 'III-IV'      — likewise constant
      ds[0]["labels"]["age"]                   # 71.0          (chf06's is NaN)
      ds[0]["labels"]["sex"]                   # 'M'
      ds[0]["labels"]["n_beats"]               # 75546
      ds[0]["labels"]["n_veb"]                 # 268    <- AAMI ventricular: V + r + E
      ds[0]["labels"]["veb_fraction"]          # 0.0035475074788870356
      ds[0]["labels"]["mean_hr_bpm"]           # 62.97685390951637
      ds[0]["labels"]["sdnn_ms"]               # 87.7751944309925
      ds[0]["labels"]["af_secs"]               # 162.596
      ds[0]["labels"]["has_rhythm_annotation"] # True  <- check this before reading
                                               #          af_secs == 0 as "no AF"
      ds[0]["labels"]["annotated_fraction"]    # 0.999991275107476 — the whole record

      # There is no class to predict, so the useful target is continuous. Ectopy
      # burden is the obvious one — but it comes from an uncorrected detector, so
      # treat it as weak supervision rather than ground truth.
      ds.labels_df["veb_fraction"].describe()  # min 0.000165, max 0.205241 over this split

      # chf06 is the shortest record at 17,789,952 samples, so a window must fit
      # inside that rather than inside the longest record's 17,998,848, or it raises
      # WindowOutOfRangeError naming the record and its true length.

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/chfdb/1.0.0/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/C29G60" }
      - { label: "Baim et al. 1986 (J Am Coll Cardiol)", url: "https://doi.org/10.1016/S0735-1097(86)80478-8" }
      - { label: "PhysioBank annotation definitions", url: "https://archive.physionet.org/physiobank/annotations.shtml" }
---
