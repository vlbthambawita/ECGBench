---
slug: post-ictal-heart-rate-oscillations-in-partial-epilepsy
name: Post-Ictal Heart Rate Oscillations in Partial Epilepsy
category: one-lead
order: 5
status: completed
source_url: https://physionet.org/content/szdb/1.0.0/
url_label: physionet.org
format: 1-lead ("ECG", unnamed) · 1.50–3.77 h · 200 Hz · 8-bit · WFDB · seizure times in a text file
patients: '5'
records: '7'
access: open
license: ODC Attribution
origin_institution: Beth Israel Deaconess Medical Center / Harvard Medical School
origin_country: USA — Boston, MA
leads: 1
paper_title: Al-Aweel et al., Post-Ictal Heart Rate Oscillations in Partial Epilepsy, Neurology 53(7):1590-1592, 1999
paper_doi: https://doi.org/10.1212/wnl.53.7.1590
search_keywords: post ictal epilepsy seizure szdb usa boston beth israel harvard heart rate oscillations partial seizure video eeg monitoring aristotle unaudited single lead
sections:
- type: description
  title: Overview
  body: |-
    Seven single-lead ECG recordings — 16.77 hours at 200 Hz — made during continuous
    EEG/ECG/video epilepsy monitoring of **five women** aged 31 to 48 with partial
    seizures of frontal or temporal origin and no clinical evidence of cardiac
    disease, at Boston's Beth Israel Deaconess Medical Center. It is the smallest
    dataset in this catalogue.

    The database exists to show one finding: transient **0.01–0.10 Hz heart-rate
    oscillations**, two to six minutes long and 15 to 41 bpm peak-to-trough,
    appearing immediately *after* a seizure and absent from the pre-ictal period of
    the same seizures. The paper reports them after 5 of 11 recorded seizures, once
    in each of the five patients. What makes that analysable is the shipped
    `times.seize` file, which gives seizure onset and offset read from the
    simultaneous EEG by an electroencephalographer blinded to the heart-rate
    analysis.

    All 39 shipped files verify against the release's own `SHA256SUMS.txt`, all 7
    records pass every ECGBench quality check, and `clean/` therefore equals
    `original/`.

    This is five patients. Use it to study post-ictal dynamics against blinded,
    EEG-derived seizure times — not to train a classifier. There is no negative
    class, no control cohort, and after grouping, no more than five independent
    people in the entire release.
- type: description
  title: Seven records, five patients, and no subject id — so ECGBench reconstructed one
  body: |-
    **This is the most consequential thing on the page.** The release ships no subject
    identifier anywhere: not in the headers, not in `RECORDS`, not in the
    annotations. The paper describes five patients. Reading "there is no column" as
    "one record per patient" — the natural reading, and what `patient_id_column:
    null` would have said — puts three recordings of the same woman on both sides of
    the split.

    **sz02, sz03 and sz04 are one woman.** The evidence, in the order it was
    established:

    1. **Beat morphology.** Median beats over a P-through-T window (−0.30 s to
       +0.50 s about each `N` annotation, baseline-removed and divided by QRS peak
       amplitude, so the T-to-QRS ratio survives) correlate at **0.9989 between sz03
       and sz04**, 0.9844 between sz02 and sz04, and 0.9806 between sz02 and sz03.
       The control is each record's own first half against its own second half — the
       ceiling a same-subject pair can reach — which is 0.9987 for sz03 and 0.9984
       for sz04. **sz03 and sz04 resemble each other more closely than either
       resembles its own other half.** No other pair comes near: the best is
       sz01–sz07 at 0.8527 against sz01's ceiling of 0.9980, and four pairs correlate
       *negatively* because their T waves are inverted relative to one another. A
       beat-level nearest-template assignment agrees — 40.8% of sz04's beats match
       sz03's template better than their own, against 3.7% for sz01 to sz07.
    2. **The subject count comes out at exactly five**, which is the figure the paper
       states, and it is the check that licenses the grouping rather than merely
       permitting it.
    3. **The seizures then fall the way the paper says they do.** It reports "Two of
       the subjects had multiple recorded seizures". Under this grouping subject 2
       has 5 and subject 4 has 2, while subjects 1, 3 and 5 have one each — exactly
       two. The arithmetic discriminates: of the groupings that give five subjects,
       only one other also gives two multi-seizure subjects ({sz05, sz06} + {sz02,
       sz03}), and morphology rejects it outright (sz05 to sz06 correlates at 0.7278
       against sz05's ceiling of 0.9970). Every other five-subject grouping implies
       three or four multi-seizure subjects and contradicts the paper.

    A weaker fourth signal agrees and is independent of the waveforms: the `.dat`
    files of sz02, sz03 and sz04 were digitised within five minutes of one another
    (26 Mar 1998, 17:35 / 17:37 / 17:40) and share a gain of 25 adu/mV.

    The values are `szdb_subj_1` … `szdb_subj_5` — deliberately named so they cannot
    be mistaken for a PhysioNet identifier — and every row carries
    `subject_id_is_reconstructed: true`. Recheck it yourself; it reads all 16.8 h, so
    it is not run at load time:

    ```python
    from ecgbench.labels.szdb import verify_subject_grouping
    verify_subject_grouping("/path/to/szdb/1.0.0").head()
    ```
- type: description
  title: Five folds, not ten
  body: |-
    Seven records from five subjects cannot make ten folds, so `szdb.yaml` sets
    **`n_folds: 5`** — the only config in ECGBench that changes it. `ecgbench splits
    --dataset szdb` picks it up with no flag, which matters because `manifest.json`
    records the fold count and hashes the partition into `fold_digest`: a user who
    had to remember `--n-folds 5` would otherwise compute a different digest and have
    no way to see why.

    Both failure modes are real, and only one of them is loud:

    | `n_folds` | What happens |
    |---|---|
    | 10 | `ValueError: Cannot have number of splits n_splits=10 greater than the number of samples: n_samples=7` — names neither the dataset nor the cause |
    | 7 | **Two folds come out empty, silently.** `StratifiedGroupKFold` keeps groups intact, and there are only 5 |
    | 5 | One subject per fold — leave-one-subject-out, which is the structure this database wants |

    With the default mapping that gives train = folds 1–3 (5 records), val = fold 4
    (sz01), test = fold 5 (sz06).

    There is nothing to stratify on, and that is measured rather than assumed. Every
    fold is one subject, and `StratifiedGroupKFold` raises when every class holds
    fewer records than there are folds — so over 7 records and 5 folds a usable axis
    needs a class of 5. Seizure count (1 vs >1) gives 3 subjects against 2 and
    raises; annotated atrial fibrillation gives 1 against 4 and raises; ST burden per
    subject is 4, 25, 5, 1, 2 episodes, so no cut reaches 5; record length gives 3
    records against 4 and does not raise, but it is not constant *within* subject 2
    (1.5 h, 3.5 h, 3.8 h), and an axis that varies inside a group cannot balance a
    grouped split. The stratification label is therefore the constant `cohort_label`,
    which reduces the split to a plain partition of the five subjects.
- type: description
  title: About those counts
  body: |-
    Every figure on this page is recomputed from the shipped files. Two disagree with
    the paper, and both matter.

    | Quantity | Paper | This release | Why |
    |---|---|---|---|
    | Patients | 5 | 5 | agrees — and is what validates the reconstructed grouping |
    | Records | not stated | 7 | the paper counts patients and seizures, never recordings |
    | Seizures | 11 | **10** | `times.seize` lists 10 intervals; one seizure has no released interval |
    | Shortest seizure | 15 s | 25 s | the missing eleventh is presumably the 15 s one |
    | Longest seizure | 110 s | 110 s | agrees |
    | ADC resolution | 12 (headers) | **8** | the samples span exactly 256 levels, [−100, +155] adu |

    **The seizure count is 10, not 11.** Any per-seizure figure derived from this
    database is a figure over 10, and nothing in the release explains the omission —
    there is no changelog of any kind here. The paper's claim that oscillations
    followed "five of 11 seizures" cannot be reproduced from the shipped files at
    all, because *which* seizures they followed was never released either.

    **The headers declare an ADC resolution of 12 and the data is 8-bit.** Values span
    exactly 256 levels, and four of the seven records (sz03, sz04, sz06, sz07) sit at
    both rails for hundreds of samples. Because the gain differs by record, the rail
    in millivolts does too: [−4.0, +6.2] mV at the 25 adu/mV of five records, and
    [−10.0, +15.5] mV at the 10 adu/mV of sz05 and sz06. ECGBench's
    `amplitude_range_mv` is the union of the two, so no record can fail validation for
    a rail its own header declares.

    Beat, ectopy and HRV figures below are **unaudited detector output** — the shipped
    `ANNOTATORS` file says "unaudited beat annotations from an automated detector",
    and the `.ari` extension names it (ARISTOTLE). Every audited MIT-BIH database in
    this catalogue uses `.atr` instead.
- type: description
  title: What the annotations actually contain
  body: |-
    73,919 annotations across the 7 records, and three of the symbols are not what a
    reader expects.

    - **`?` (350, exactly 50 per record) is the detector's learning phase, not
      unclassifiable beats.** It is always annotations 0–49, covering the first 42–51 s
      of each record. These are QRS complexes the detector *located* but had not yet
      learned to classify, so ECGBench counts them in `n_beats` and in the RR series
      and folds them into AAMI `Q` — which means `aami_Q` starts at 50 per record.
      `n_learning_beats` is exposed separately so the two can be split apart again;
      genuinely unclassifiable beats number 315 across the release, 278 of them in
      sz06.
    - **`s` (74) is ST change, not a beat.** Its `aux_note` values — `(ST0+`, `ST0+)`,
      `(ST0-`, `ST0-)` — delimit **37 ST episodes**, 31 depression and 6 elevation.
      Counting `s` as a beat would inflate every beat total and put a non-beat into
      the AAMI reduction. Burden is very uneven: sz02 holds 24 depression episodes
      over 336 s, sz06 a single one lasting **826 s**, and sz04 none at all.
    - **`+` (2) is rhythm, and it exists in one record only.** sz02 carries `(AFIB` at
      10,508.6 s and `(N` at 10,526.0 s — 17.4 s of atrial fibrillation, ending 25 s
      before that record's second seizure begins. **The other six records carry no
      rhythm marker at all**, so their `af_secs` of 0.0 means "never assessed", not
      "no atrial fibrillation"; read `has_rhythm_annotation` first.

    There is **no signal-quality annotation layer** — no `~` and no `|` anywhere —
    where `mitdb`, `svdb` and `nsrdb` all carry one. Beat annotation, by contrast,
    covers essentially the whole of every record: the first beat falls 0.1–0.9 s in
    and the last within 0.6 s of the end, in all 7.

    One unexplained anomaly, recorded so nobody rediscovers it: **sz07 alone carries a
    non-zero `num` field** on 8,768 of its 8,892 annotations, spanning the full signed
    byte range. It correlates with neither ST deviation at J+60 to J+120 ms
    (|r| ≤ 0.18) nor the preceding RR interval (r = −0.05), nothing in the release
    documents it, and the other six records leave `num` at 0 throughout. ECGBench
    ignores it.
- type: table
  title: The 7 records, recomputed from the files
  headers:
  - Rec
  - Subj
  - Hours
  - Seizures
  - Durations (s)
  - Beats
  - S
  - V
  - Q
  - ST eps
  - ST s
  - HR
  - SDNN ms
  - Gain
  - Fold
  rows:
  - - sz01
    - S1
    - '1.50'
    - '1'
    - '96'
    - 8,377
    - '33'
    - '32'
    - '3'
    - '4'
    - '25'
    - '93'
    - '103'
    - '25'
    - 4 (val)
  - - sz02
    - S2
    - '3.50'
    - '2'
    - 60, 25
    - 13,145
    - '23'
    - '19'
    - '1'
    - '24'
    - '336'
    - '63'
    - '101'
    - '25'
    - 1 (train)
  - - sz03
    - S2
    - '3.77'
    - '2'
    - 108, 110
    - 16,382
    - '34'
    - '33'
    - '0'
    - '1'
    - '3'
    - '72'
    - '92'
    - '25'
    - 1 (train)
  - - sz04
    - S2
    - '1.50'
    - '1'
    - '105'
    - 6,229
    - '22'
    - '23'
    - '0'
    - '0'
    - '0'
    - '69'
    - '92'
    - '25'
    - 1 (train)
  - - sz05
    - S3
    - '1.50'
    - '1'
    - '83'
    - 8,066
    - '40'
    - '4'
    - '32'
    - '5'
    - '28'
    - '90'
    - '56'
    - '10'
    - 3 (train)
  - - sz06
    - S4
    - '3.00'
    - '2'
    - 54, 85
    - 12,756
    - '6'
    - '65'
    - '278'
    - '1'
    - '826'
    - '71'
    - '52'
    - '10'
    - 5 (test)
  - - sz07
    - S5
    - '2.00'
    - '1'
    - '89'
    - 8,888
    - '38'
    - '7'
    - '1'
    - '2'
    - '16'
    - '74'
    - '81'
    - '25'
    - 2 (train)
  - - '**Total**'
    - '**5**'
    - '**16.77**'
    - '**10**'
    - '**815 s**'
    - '**73,843**'
    - '**196**'
    - '**183**'
    - '**315**'
    - '**37**'
    - '**1,234**'
    - —
    - —
    - —
    - '**5 folds**'
  footnote: |-
    Subj is ECGBench's reconstruction, not a shipped field — sz02, sz03 and sz04 are
    one woman. S/V/Q are AAMI EC57 classes; Q excludes the 50 learning-phase
    detections every record carries, so the Beats column is 350 higher than S+V+Q+N
    would suggest. Every beat, ectopy and HRV figure is unaudited detector output.
    Gain is adu/mV and sets the clipping rail: [−4.0, +6.2] mV at 25, [−10.0, +15.5]
    at 10.
- type: code
  title: Loading with ECGBench
  language: python
  body: |
    from ecgbench import ECGDataset

    # window= is what makes this batchable: records run 1,079,998 to 2,711,998
    # samples, so a batch of whole records is both large and ragged. It is pushed
    # into wfdb as sampfrom/sampto, so only these samples are decoded.
    ds = ECGDataset(
        "szdb",
        split="train",
        labels=True,
        window=(0, 12_000),      # 60 s; must fit sz01/sz04's 1,079,998 samples
        data_path="/path/to/szdb/1.0.0/",
    )

    len(ds)                                    # 5
    sample = ds[0]
    sample["signal"].shape                     # (1, 12000)
    sample["record_id"]                        # 'sz02'
    sample["labels"]["subject_id"]             # 'szdb_subj_2'  <- RECONSTRUCTED
    sample["labels"]["n_seizures"]             # 2
    sample["labels"]["seizure_starts_secs"]    # '3763|10551'   <- pipe-joined, in seconds
    sample["labels"]["n_beats"]                # 13145
    sample["labels"]["n_learning_beats"]       # 50             <- always 50, the warm-up
    sample["labels"]["has_rhythm_annotation"]  # True           <- the only record with any
    sample["labels"]["af_secs"]                # 17.43

    # The intended use: cut the window after a seizure ends. The oscillation the
    # database was published for is 0.01-0.10 Hz and two to six minutes long, so it
    # is invisible in any whole-record HRV summary.
    end = float(sample["labels"]["seizure_ends_secs"].split("|")[0])   # 3823.0
    post = ECGDataset(
        "szdb", split="train", labels=True,
        window=(int(end * 200), 360 * 200),    # the 6 minutes after seizure offset
        data_path="/path/to/szdb/1.0.0/",
    )
- type: code
  title: Building the splits
  language: bash
  body: |
    # No --n-folds flag: szdb.yaml carries n_folds: 5, so the canonical partition
    # (one subject per fold) is reproducible from the shipped config alone.
    ecgbench splits --dataset szdb --data-path /path/to/szdb/1.0.0/
- type: links
  title: Links
  links:
  - label: PhysioNet — szdb 1.0.0
    url: https://physionet.org/content/szdb/1.0.0/
  - label: 'Al-Aweel et al., Neurology 53(7):1590-1592 (1999)'
    url: https://doi.org/10.1212/wnl.53.7.1590
  - label: Example script — examples/load_szdb.py
    url: https://github.com/vlbthambawita/ECGBench/blob/main/examples/load_szdb.py
---
