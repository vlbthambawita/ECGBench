---
slug: "european-st-t-database-edb"
name: "European ST-T Database (EDB)"
category: "two-lead"
order: 6
status: "completed"
source_url: "https://physionet.org/content/edb/1.0.0/"
url_label: "physionet.org"
format: "2-lead ambulatory · 15 lead layouts · 2 h · 250 Hz · WFDB"
patients: "79"
records: "90"
access: "open"
license: "ODC-By 1.0"
origin_institution: "CNR Institute of Clinical Physiology, Pisa; European Society of Cardiology"
origin_country: "Italy"
leads: 2
paper_title: "Taddei et al., Eur Heart J 1992"
paper_doi: "https://doi.org/10.13026/C2D59Z"
search_keywords: "european st-t edb ischemia ischaemia st segment depression elevation t-wave episode annotations italy pisa cnr esc ambulatory holter angina coronary artery disease two-lead"

related:
  - slug: "qt-database-qtdb"
    relation: "has_derivative"
    shares_records: true
    verified: true
    note: >
      33 of the QT Database's 105 records are 15-minute excerpts of 33 of these 90
      recordings, and QTDB's own headers say so: each `sele0*` record carries a
      "Produced by xform from record e0104, beginning at 1:35:00.000" provenance
      line, and its name encodes the parent (`sele0104` from `e0104`). Verified
      sample-for-sample at the stated offsets — the two agree exactly on 30 of the 33
      after removing QTDB's constant per-channel baseline shift, and on the other
      three the residual is under one ADC unit (0.005 mV), i.e. rounding. That is
      8.25 h of these 180 h, covering 37% of the records: e0104, e0106, e0107, e0110,
      e0111, e0112, e0114, e0116, e0121, e0122, e0124, e0126, e0129, e0133, e0136,
      e0166, e0170, e0203, e0210, e0211, e0303, e0405, e0406, e0409, e0411, e0509,
      e0603, e0604, e0606, e0607, e0609, e0612 and e0704. **Do not train on this
      database and evaluate QT-interval measurement on QTDB's `sele0*` records, or the
      reverse** — those are the same signal samples. QTDB also *renames* the leads, so
      the overlap is not visible from the channel names either: this database's MLIII
      is QTDB's `D3` or `ML5`, its V5 is `CM5`, its V2 is `CM2`, `V1-V2` or `V2-V3`.
      QTDB's other 72 records come from the MIT-BIH databases and are unrelated to
      this one.

sections:
  - type: description
    title: "Overview"
    body: |
      **The reference benchmark for ST-segment and T-wave change analysis, and the
      only dataset in this catalogue whose ground truth is episodes rather than
      labels.** 90 two-hour ambulatory recordings from 79 subjects with diagnosed or
      suspected myocardial ischaemia, assembled between 1985 and 1990 under a
      European Society of Cardiology concerted action across centres in seven
      countries, coordinated from Pisa. Two cardiologists annotated every record beat
      by beat and marked the **onset, extremum and end** of each interval of
      significant ST or T change, working on the two signals independently, with a
      third resolving their disagreements. 180 hours, **802,909 annotations**,
      **368 ST episodes** and **401 T episodes**.

      **Lead layout varies more here than in any other dataset in this catalogue,
      and no lead is present in every record.** All 90 records store two leads — and
      they use **fifteen different orderings of eleven different lead pairs**. V5
      reaches 51 records, MLIII 47, V4 34, and D3 appears exactly once. `MLIII/V4`
      and `V4/MLIII` are *both* present, 15 records each, so `signal[0]` is a limb
      lead in one half of those 30 records and a chest lead in the other with nothing
      in the metadata to say which. Lead placement was never standardised across the
      contributing centres. **This is the single most important thing on this page**:
      `config.lead_names` is `["V5", "MLI"]` and describes only 19 of the 90 records,
      so it is a name for the modal layout and not something to index against.

      **The deviations are measured against each subject's own reference waveform,
      not against an absolute isoelectric line.** ST deviation is taken 80 ms after
      the J point (60 ms above 120 bpm) and compared with a reference complex from
      that record's own first 30 seconds; an episode requires 0.1 mV of ST deviation
      (0.2 mV for T) sustained for at least 30 s. So `peak_st_deviation_uv = 600`
      means 600 µV *above where the ST segment sat in the first 30 seconds of that
      record*, and subjects with prior infarction carry fixed elevation or depression
      underneath that these annotations do not describe. A fixed ST-level threshold
      cannot reproduce these labels. The reference complexes themselves were printed
      on transparent plastic rulers which, the release notes, no longer exist.

      **The signals carry a large uncorrected DC offset.** Gain was calibrated
      against the original analog calibration signals to a uniform 200 ADC units per
      millivolt; offset was not. **116 of the 180 signals sit more than 1 mV off
      zero** and 58 more than 3 mV, up to **+9.05 mV**, and **21 records never cross
      0 mV at all** — e0114 lives entirely between +5.635 and +9.785 mV. Peak-to-peak
      is a normal 4.02 mV median. Baseline removal is a prerequisite here, not a
      refinement, and it is consistent with the annotation semantics above: this
      database never claims an absolute ST level.

      **No subject identifier ships.** `edb.txt` states the 90 records come from 79
      subjects and nothing in the files says which. ECGBench reconstructs it; see
      "Subject identity is reconstructed" below before relying on `patient_id`.

  - type: table
    title: "ST and T episodes, recomputed from the 90 .atr files"
    headers: ["Quantity", "Signal 0", "Signal 1", "Total", "Records containing it"]
    rows:
      - ["**ST episodes**", "185", "183", "**368**", "86 of 90"]
      - ["— ST elevation", "", "", "118", ""]
      - ["— ST depression", "", "", "250", ""]
      - ["**T episodes**", "219", "182", "**401**", "70 of 90"]
      - ["— T amplitude increase", "", "", "219", ""]
      - ["— T amplitude decrease", "", "", "182", ""]
      - ["extreme-T threshold crossings", "", "", "166", "*not episodes*"]
      - ["axis-shift pseudo-episodes", "", "", "21", "*not findings* — 6 records"]
      - ["episodes with no end annotation", "", "", "12", "closed at the record end"]

  - type: table
    title: "Reference beat annotations, recomputed from the 90 .atr files"
    headers: ["Beat type", "Symbol", "Count", "Share", "Records containing it"]
    rows:
      - ["normal beat", "N", "784,633", "99.250%", "90"]
      - ["premature ventricular contraction", "V", "4,467", "0.565%", "50"]
      - ["supraventricular premature or ectopic beat", "S", "1,093", "0.138%", "64"]
      - ["fusion of ventricular and normal beat", "F", "354", "0.045%", "14"]
      - ["unclassifiable beat", "Q", "11", "0.001%", "4"]
      - ["supraventricular escape beat", "n", "5", "0.001%", "1"]
      - ["nodal (junctional) premature beat", "J", "1", "0.000%", "1"]
      - ["aberrated atrial premature beat", "a", "1", "0.000%", "1"]
      - ["**total**", "", "**790,565**", "**100%**", "**90**"]

  - type: table
    title: "The same beats under the AAMI EC57 reduction — use these to combine with mitdb, svdb and incartdb"
    headers: ["AAMI class", "Built from", "Count", "Share", "Records containing it"]
    rows:
      - ["N — normal / bundle branch block", "N, L, R, e, j, B", "784,633", "99.250%", "90"]
      - ["S — supraventricular ectopic", "A, a, J, S, n", "1,100", "0.139%", "65"]
      - ["V — ventricular ectopic", "V, E", "4,467", "0.565%", "50"]
      - ["F — fusion", "F", "354", "0.045%", "14"]
      - ["Q — unclassifiable / paced", "/, f, Q", "11", "0.001%", "4"]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the 90 headers and `.atr` files,
      after verifying the shipped data against the release's own `SHA256SUMS.txt` —
      **all 275 listed files match**. There is no changelog in the release directory.

      | Quantity | `edb.txt` (1992 paper) | Recomputed (v1.0.0) | Diff |
      |---|---|---|---|
      | Records | 90 | 90 | — |
      | Subjects | 79 | 79 *(reconstructed)* | — |
      | Men | 70, aged 30–84 | 70, aged 30–84 | — |
      | Women | 8, aged 55–71 | 8, 7 with ages 53–71 | ages differ |
      | Subjects with information missing | 1 | 1 (e0166) | — |
      | T episodes | 401 | **401** | — |
      | ST episodes | 367 | **368** | **+1** |
      | Annotations | 802,866 | **802,909** | **+43** |

      **The T-episode count matches exactly, which is what validates the counting
      convention.** An "episode" is counted once, at its onset annotation. Getting
      401 requires all three of: truncating `aux_note` at its NUL terminator,
      excluding the 166 `++`/`--` extreme-T threshold crossings, and excluding the 21
      lower-case axis-shift spans. Get any one wrong and the total is 394, 567 or 411
      instead. That the same convention gives 368 ST episodes against the paper's 367
      is therefore a real one-episode difference between the shipped annotations and
      the 1992 paper, not a counting error — and with no changelog in the release
      there is nothing to attribute it to. The same goes for the 43 extra
      annotations: they are not at the record boundaries (no annotation in the release
      falls at sample 0 or at or past sample 1,800,000) and they are not a single
      annotation type. Treat the recomputed figures as authoritative for v1.0.0 and
      the published ones as describing whatever the authors held in 1992.

      **The women's age range differs and the reason is visible.** Seven of the eight
      female subjects have an age: 53, 55, 56, 62, 64, 70 and 71. e0418's is `-`. The
      paper says "8 women aged 55 to 71", which excludes the 53-year-old (e0818). The
      men's figures — 70 subjects, 30 to 84 — reproduce exactly.

      **One derived figure is a convention, and it is the largest one.** e0409 shows
      `ischaemic_fraction` 0.930, the highest in the release, and that number exists
      because its two ST depressions have onsets (8.44 min and 17.15 min) and
      extrema (103.51 min and 99.06 min) but **no end annotations**. The episodes are
      genuinely very long — the annotators marked their peaks more than 80 minutes
      after the onsets — but where they end was never recorded, and ECGBench closes
      an unterminated episode at the end of the record. 10 of the 12 unterminated
      episodes open in the final four minutes and the convention barely matters for
      them; for e0409 it sets the figure. `n_unterminated_episodes` flags it.

      **`st_episode_secs` can exceed the recording length, and that is correct.** The
      two signals are annotated independently, so concurrent ST depression in both
      channels counts twice: e0607 reaches 131.5 min of ST episode in a 120-minute
      record. `st_secs_any_signal` is the bounded union — 29.2 h across the release
      against 40.2 h summed — and `ischaemic_fraction` is that over the 7,200 s
      record. Use the union for any fraction.

  - type: table
    title: "The fifteen lead layouts, counted from the 90 headers"
    headers: ["Layout (file order)", "Records", "Layout (file order)", "Records"]
    rows:
      - ["V5 / MLI", "19", "MLIII / V3", "3"]
      - ["V4 / MLIII", "15", "V3 / MLIII", "2"]
      - ["MLIII / V4", "15", "V3 / V5", "2"]
      - ["V5 / MLIII", "11", "V2 / V4", "2"]
      - ["V5 / V1", "7", "D3 / V4", "1"]
      - ["V2 / V5", "6", "V5 / V2", "1"]
      - ["V1 / V5", "4", "V2 / MLIII", "1"]
      - ["", "", "V5 / V4", "1"]

  - type: table
    title: "…and how many records contain each lead. None reaches 90."
    headers: ["Lead", "Records", "Share of the release"]
    rows:
      - ["V5", "51", "57%"]
      - ["MLIII", "47", "52%"]
      - ["V4", "34", "38%"]
      - ["MLI", "19", "21%"]
      - ["V1", "11", "12%"]
      - ["V2", "10", "11%"]
      - ["V3", "7", "8%"]
      - ["D3", "1", "1%"]

  - type: description
    title: "What the fifteen layouts mean for loading"
    body: |
      `config.record_lead_layouts` declares all fifteen, which makes
      `ECGDataset(leads=[...])` open each record's own header and resolve the
      requested **names** against it. `mitdb` is the only other config here that
      needs this; edb needs it far more, because mitdb's declared layout covers 40 of
      48 records and this one's covers 19 of 90.

      **Because no lead is in every record, every name-based selection here raises
      for some records.** `leads=["V5"]` loads 51 and raises
      `ValueError: Record 'e0604' stores ['V2', 'MLIII'] … Lead 'V5' is not in 'edb'`
      for the other 39. That is the intended behaviour — the alternative is silently
      returning a different physical lead — but it means **`leads=` alone does not
      make this dataset batchable**. You need a lead *and* a record filter, and the
      widest single choice available is V5 at 57% of the release.

      Selecting positionally is wrong for at least 71 of the 90 records, and the
      failure is silent. It is worth restating that `MLIII/V4` and `V4/MLIII` are
      both present in equal numbers: those 30 records are indistinguishable by shape,
      by lead count, and by anything in the exported fold CSVs.

      The per-record layout is in the labels as `lead_names`, pipe-separated, so it
      can be read without opening a header.

  - type: description
    title: "Reading the episode annotations, and three ways to get them wrong"
    body: |
      An episode is **three annotations** and the WFDB symbol does not distinguish
      them — `s` for ST and `T` for T change, onset or extremum or end alike. The
      `aux_note` text is the only source: `(ST0+` opens an episode of ST elevation in
      signal 0, `AST0+600` marks its extremum at +600 µV, and `ST0+)` closes it.
      Three things about that text cost real accuracy:

      **1. `aux_note` carries bytes past its NUL terminator.** Seven T-episode onsets
      read `'(T0+\x00\x13'`, `'(T0-\x00N'`, `'(T1-\x00\x1a'` and so on. Neither
      `.strip()` nor `.rstrip('\x00')` removes the trailing garbage, so those seven
      fall into categories of their own and the T total comes out at 394 instead of
      401. Truncate at the first NUL, which is what the format means.

      **2. `++` and `--` are not episodes.** Inside a T episode whose deviation
      exceeds 400 µV, extra annotations mark each crossing of that threshold. There
      are **166** of them; counting them as episodes inflates the T total by 41%.
      They are exposed as `n_extreme_t_markers`.

      **3. Lower-case episode text is recognised artefact, and case-folding merges it
      with the findings.** In six records — e0161, e0509, e0601, e0611, e0613, e0615
      — a positional change shifts the axis and produces something that looks exactly
      like an ST or T change. The annotators marked those **21** spans on `"` comment
      annotations spelled `(st0+`, `at1-800`, `st0-)` in lower case *precisely so they
      can be told apart* from the upper-case findings. They are `n_axis_shift_episodes`
      and are excluded from every episode count.

      A fourth, in the documentation rather than the data: **the `~` signal-quality
      subtype table in the shipped `annotations.shtml` disagrees with the files.** It
      tabulates nine values and gives `un` as `0x12`, `cu` as `0x20` and `nu` as
      `0x21`; the release contains `0x13`, `0x22` and `0x23` and **none of those
      three**. The table is internally inconsistent — its own `uc` is `0x11`, which
      already sets signal 0's noisy bit — so ECGBench reads the subtype as the bitmask
      it is (bit 0/1 noisy for signal 0/1, bit 4/5 unreadable, unreadable also setting
      noisy). That accounts for all 8,918 quality annotations with nothing left over.

  - type: table
    title: "ST-episode burden bands — the stratification label"
    headers: ["Band", "ST episodes", "Records", "ST episodes in the band", "Note"]
    rows:
      - ["none", "0", "4", "0", "e0133, e0155, e0509, e0611"]
      - ["1-2", "1 – 2", "32", "53", ""]
      - ["3-5", "3 – 5", "29", "115", ""]
      - ["6+", "6 or more", "25", "200", "54% of all ST episodes"]

  - type: description
    title: "Why the folds are stratified on ST-episode burden"
    body: |
      **Because it is the quantity this database exists to measure, and nothing else
      here works.** The header clinical text is subject background — every subject was
      selected for suspected ischaemia, so it is not a case/control axis.
      `dominant_rhythm` is sinus in all 90 records. `st_t_class` has two classes of
      two records. What genuinely differs between records is how much ST change each
      one holds, and it differs enormously: e0604 holds 20 episodes and four records
      hold none.

      **The band edges are fixed counts, not quantiles** — 1, 3 and 6 episodes, giving
      4 / 32 / 29 / 25 records. Quantile edges would balance slightly better and would
      move every record's label the next time an annotation is revised; fixed edges
      reproduce against a re-release, which matters more for a partition that gets
      published.

      Every one of the ten folds carries all three populated bands — each holds 3–4
      `1-2`, 2–3 `3-5` and 2–3 `6+` records. **The `none` band holds 4 records, fewer
      than the 10 folds**, so it cannot appear in every fold; in this partition all
      four land in `train`, which means `val` and `test` contain no ST-free record.
      It is kept as its own band regardless, because a record with no ST change at all
      is the negative control an ST detector is scored against and folding those four
      into `1-2` would make them invisible. With 90 records over 10 folds, `train`
      holds 74, `val` 8 and `test` 8.

      `st_burden_band` is a **fold-construction label** and a coarsening at that.
      Train on the episode counts, `ischaemic_fraction`, `peak_st_deviation_uv`, or —
      for episode detection, which is what this database was built to evaluate — on
      the `.atr` files directly. Never on `stratify_class`.

  - type: description
    title: "Subject identity is reconstructed, not released"
    body: |
      **The release ships no subject identifier.** `edb.txt` says the 90 records come
      from 79 subjects; nothing in the files says which are which. That matters,
      because ungrouped folds would put the same person in train and test — e0118,
      e0119, e0121 and e0122 are one 51-year-old man recorded four times.

      ECGBench therefore reconstructs it from the header: records agreeing on age,
      sex, recorder model, medications and the **set** of clinical findings are taken
      to be one subject. That gives 80 groups. The remaining merge is **e0206 and
      e0210**, which agree on everything except age — 55 against 53, two recordings of
      the same 3-vessel-disease man on the same Oxford Medilog MR-20 in the same
      V5/MLI placement, two years apart. Merging them gives **79**.

      **That the reconstruction lands on the published subject count is the check
      that it is not merely plausible**, and it reproduces the published demographics
      too: 70 men aged 30–84, against `edb.txt`'s "70 men aged 30 to 84", 8 women, and
      exactly one subject whose information is missing. The result is 72 subjects with
      one record, five with two, and two with four.

      Two things to know about the method. The findings must be compared as a **set**:
      e0126 lists exactly the same five clinical lines as e0123–e0125 but with "Aortic
      valvular regurgitation" and "1-vessel disease (LAD)" swapped, so an
      order-sensitive comparison drops it from its own subject's group. And the
      reconstruction is **conservative in one direction only** — two genuinely
      different subjects sharing age, sex, recorder, medication and findings would be
      merged, which costs a little fold flexibility and no correctness, while the
      error that matters, splitting one subject across folds, is bounded by the count
      agreement.

      **An attempt to confirm the grouping from the signals was inconclusive, and is
      not the basis for it.** Normalised median QRST complexes compared across the
      same lead pair do not separate subjects in this cohort: a different-sex control
      pair (e0203 F / e0206 M) scored 0.978 against a within-group *minimum* of 0.758,
      and the control median over 468 same-layout pairs was 0.823. Ischaemic patients
      recorded on the same two leads simply look alike at that resolution. The claim
      here rests on the header agreement and the published-count match, not on the
      waveforms.

  - type: table
    title: "Clinical background from the header comments (subject-level)"
    headers: ["Field", "Values", "Records"]
    rows:
      - ["angina pattern", "resting", "23"]
      - ["", "mixed", "22"]
      - ["", "effort", "10"]
      - ["", "unspecified (\"angina pectoris\", \"chest pain\")", "5"]
      - ["", "not stated", "30"]
      - ["myocardial infarction", "any", "30"]
      - ["", "— unspecified site", "15"]
      - ["", "— inferior", "8"]
      - ["", "— anterior", "5"]
      - ["", "— infero-lateral / non-Q", "1 / 1"]
      - ["coronary vessels diseased", "1 / 2 / 3", "24 / 11 / 17"]
      - ["", "normal coronary arteries", "8"]
      - ["", "not stated", "30"]
      - ["arterial hypertension", "yes", "12"]
      - ["coronary artery by-pass graft", "yes", "4"]

  - type: description
    title: "Signal quality, recorders, and what else the annotators marked"
    body: |
      Signal quality is annotated **per channel** by `~` transitions, each opening an
      interval that runs to the next. Across the 180 h per channel, **signal 0 is
      93.21% clean** (11.52 h noisy, 0.70 h unreadable) and **signal 1 is 94.35%
      clean** (9.11 h noisy, 1.07 h unreadable). The averages hide a wide spread:
      e0148 is 52.2% noisy or unreadable across its two channels, e0808 39.4% and
      e0817 33.1%, while **e0417 carries no `~` annotation at all** and is clean end
      to end. `usable_fraction` — the share of the record in which at least one
      channel is readable — never drops below 0.982.

      **No record has a `~` at sample 0.** The first falls at a median of 9.3 minutes,
      so the leading span was never asserted to be anything; ECGBench counts it clean,
      which is what WFDB does, and reports it as
      `quality_head_unasserted_secs` so the assumption is visible rather than buried.

      Rhythm, by contrast, **is** annotated from the start of every record: all 90 open
      with a `+` between sample 2 and 264 — `(N` in 88 of them and `(SBR` in e0303 and
      e0611 — so `rhythm_secs_*` covers the whole recording bar under a second. Sinus
      dominates every record, which is why `dominant_rhythm` is not the label column,
      but the non-sinus spans are real: 20.7 min of atrial fibrillation in one record,
      17.2 min of sinus bradycardia in three, 6.5 min of ventricular tachycardia in
      nine, plus bigeminy, trigeminy, supraventricular tachyarrhythmia, sino-atrial
      block and one third-degree block.

      Also marked: **54 patient-event button presses** in 12 records (`note_BUTTON` —
      the subject pressing a button, typically at symptom onset, which is a label of a
      kind), **one tape slippage** (`note_TS`), and **78 isolated QRS-like artefacts**
      in 15 records.

      **Ten Holter recorder models** were used, unevenly: ICR 7200 for 37 records,
      Oxford Medilog MR-20 and Del Mar Avionics 445B for 14 each, Oxford Medilog 4-24
      for 12, and six models for 1–3 records each. Recorder-specific artefact is a real
      confounder in a collection of this era, so `recorder_type` is exposed to control
      for it.

  - type: table
    title: "Validation summary (250 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "90", "all records, with is_valid + quality_issues"]
      - ["clean", "90", "100% pass rate — nothing is excluded"]

  - type: description
    title: "Nothing fails validation, and the amplitude range needs explaining"
    body: |
      All 90 records pass every check, so `original` and `clean` hold the same 90
      rows. There are no NaN samples, no flat or all-zero leads and no unreadable
      header. Records are exactly **1,800,000 samples** each, uniformly, so unlike
      `afdb`, `nsrdb` and `ptbdb` the `truncated_signal` check is switched on rather
      than disabled — and any `window=` inside 7,200 s fits all 90 records.

      **`amplitude_range_mv` is `[-10.24, 10.235]`, and here that is the only
      defensible value.** Format 212 is 12-bit two's complement, so with `adc_zero` 0
      and a gain of 200 adu/mV full scale is exactly [−2048, 2047] adu = −10.240 to
      +10.235 mV. A tighter physiologic window would be meaningless for this release,
      because the DC offset is uncorrected and large: 21 records never cross 0 mV and
      one sits entirely above +5.6 mV, so absolute amplitude here measures the offset
      rather than the ECG.

      Measured across all 90 records, samples run **−8.195 to +10.045 mV**, so no
      record saturates and the check cannot fire on EDB 1.0.0 — which is the intended
      state. What it guards is a mis-scaled copy: a read in microvolts, or a gain
      applied twice, would be 200× out on the first record.

  - type: description
    title: "Overlap with the other ST databases in this catalogue"
    body: |
      **The QT Database overlap is real, verified, and declared** — 33 of its 105
      records are 15-minute excerpts of 33 of these recordings, confirmed
      sample-for-sample against the provenance line in QTDB's own headers. See the
      related-datasets note above; the practical consequence is that this database and
      QTDB's `sele0*` records cannot be used as train and test for one another.

      **The Long-Term ST Database is not declared, because it cannot be checked.**
      LTSTDB comes from the same ST-analysis research community and shares
      contributors, and its 86 records are 21–24 h rather than 2 h, so they are
      certainly different *recordings*. Whether any of its 80 subjects also appear
      here is unanswerable from the files: its record identifiers (`s20011`, `s30661`)
      share no scheme with these, and neither release publishes a subject identifier
      that could join. Nothing is asserted either way.

      **The MIT-BIH ST Change Database is a separate collection** — 28 records from
      the Beth Israel Hospital, and QTDB's `sel3*` records come from it rather than
      from here.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset edb --data-path /path/to/edb/1.0.0/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # 1,800,000 samples per record; window= is pushed into the reader, so it
      # avoids decoding the other 1 h 59 m rather than cropping afterwards.
      ds = ECGDataset(
          "edb",
          split="train",
          data_path="/path/to/edb/1.0.0/",
          window=(0, 2500),          # first 10 s at 250 Hz
          labels=True,
      )

      len(ds)                                   # 74
      ds[0]["signal"].shape                     # torch.Size([2, 2500])
      ds[0]["record_id"]                        # 'e0104'

      ds.lead_names                             # ('V5', 'MLI')  <- the MODAL layout,
                                                #    and NOT this record's
      ds[0]["labels"]["lead_names"]             # 'MLIII|V4'     <- what e0104 stores

      ds[0]["labels"]["n_st_episodes"]          # 6   (3 elevation, 3 depression)
      ds[0]["labels"]["n_t_episodes"]           # 11
      ds[0]["labels"]["peak_st_deviation_uv"]   # 350   <- vs this subject's own
                                                #          reference, not isoelectric
      ds[0]["labels"]["ischaemic_fraction"]     # 0.0571
      ds[0]["labels"]["st_t_class"]             # 'st_and_t'
      ds[0]["labels"]["st_burden_band"]         # '6+'   <- for folds, not for training
      ds[0]["labels"]["angina_type"]            # 'mixed'
      ds[0]["labels"]["patient_id"]             # 'e0104'  <- reconstructed
      ds[0]["labels"]["n_beats"]                # 7696
      ds[0]["labels"]["dominant_rhythm"]        # 'N'  -- sinus, as in all 90 records

      # Selecting a lead BY NAME re-resolves against each record's own header, which
      # is the only correct way to read this dataset. No lead is in all 90 records,
      # so it raises for the ones that lack the one you asked for.
      v5 = ECGDataset("edb", split="train", data_path="/path/to/edb/1.0.0/",
                      window=(0, 2500), leads=["V5"])

      v5[0]
      # ValueError: Record 'e0104' stores ['MLIII', 'V4'], and this dataset uses
      # more than one lead layout. Lead 'V5' is not in 'edb'.
      # Available: ['MLIII', 'V4']
      #
      # V5 resolves for 43 of the 74 records in this split -- the widest any single
      # lead reaches. Batching edb needs a lead AND a record filter.

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/edb/1.0.0/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/C2D59Z" }
      - { label: "Taddei et al., Eur Heart J 13:1164-1172 (1992)", url: "https://doi.org/10.1093/oxfordjournals.eurheartj.a060332" }
      - { label: "Annotation definitions for this database", url: "https://physionet.org/content/edb/1.0.0/annotations.shtml" }
      - { label: "PhysioBank annotation definitions", url: "https://archive.physionet.org/physiobank/annotations.shtml" }
      - { label: "ANSI/AAMI EC57 beat classes", url: "https://webstore.ansi.org/standards/aami/ansiaamiec572012r2020" }
---
