---
slug: preterm-infant-cardio-respiratory-signals-database-picsdb
name: Preterm Infant Cardio-Respiratory Signals Database (PICSDB)
category: one-lead
order: 9
status: completed
source_url: https://physionet.org/content/picsdb/1.0.0/
url_label: physionet.org
format: 1-lead (bedside monitor) · 20.3–70.3 h · 500 Hz (8 records) / 250 Hz (2) · WFDB · bradycardia onsets + R peaks
patients: '10'
records: '10'
access: open
license: ODC Attribution
origin_institution: UMass Memorial Healthcare NICU
origin_country: USA — Worcester, MA
leads: 1
paper_title: 'Gee et al., Predicting Bradycardia in Preterm Infants Using Point Process Analysis of Heart Rate, IEEE TBME 64(9):2300-2308, 2017'
paper_doi: https://doi.org/10.1109/TBME.2016.2632746
search_keywords: picsdb preterm infant cardio respiratory usa umass worcester nicu apnea bradycardia neonatal r peaks qrsc respiration inductance band point process
sections:
- type: description
  title: Overview
  body: |-
    Ten simultaneous ECG and respiration recordings — **439.8 hours of ECG**, 20.3 h
    to 70.3 h per infant — made at the bedside from Philips IntelliVue MP70 monitors
    in the neonatal intensive care unit of the University of Massachusetts Memorial
    Healthcare. The infants were 29 3/7 to 34 2/7 weeks post-conceptional age (mean
    31 1/7) and 843 to 2,100 g at study (mean 1,468), all spontaneously breathing
    room air.

    It is the **only neonatal dataset in this catalogue**, and the youngest cohort in
    it by decades. That is not a detail: these hearts run at **130–166 bpm**, so a
    normal RR interval here is 0.36–0.46 s and a bradycardia is anything past 0.6 s.
    Adult HRV parameter choices, adult-trained detectors and adult amplitude
    expectations are all mis-specified on this data.

    **The ground truth is an event time, not a diagnosis.** Every infant is a preterm
    infant in the same unit, so the record-level class is a constant and there is
    nothing to classify per record. What the database exists for is the **622
    manually validated bradycardia onsets** in the `.atr` files and the **3,797,503
    manually verified R peaks** in the `.qrsc` files — both time series, and both
    reachable aligned to `window=` through `ecgbench.labels.picsdb`.

    All 73 shipped files verify against the release's own `SHA256SUMS.txt`, all 10
    records pass every ECGBench quality check, and `clean/` therefore equals
    `original/`. Read the next two sections before trusting that last sentence.
- type: description
  title: Two sampling rates, so a window in samples is not a window in time
  body: |-
    **This is the trap most likely to produce quietly wrong results.** `infant1` and
    `infant5` are the 250 Hz "compound" recordings the release describes; the other
    eight run at 500 Hz. `window=(0, 15_000)` is **30 s of `infant2` and 60 s of
    `infant5`**, and any code converting samples to seconds with one global rate is
    wrong for two records in ten.

    Sampling rate is a per-record **property** here, not a choice of representation,
    so `picsdb.yaml` keys its single path column on the nominal 500 Hz and
    `ECGDataset(sampling_rate=250)` raises rather than handing back a mixed-rate
    subset — the same shape as `challenge2021`. To select by rate, filter on the
    per-record `sampling_rate` label column.

    The respiration companions invert the same asymmetry: 50 Hz for nine infants and
    **500 Hz for `infant1`**.
- type: description
  title: The bradycardia onset does not land on an R peak
  body: |-
    The release defines a bradycardia as heart rate below 100 bpm — equivalently
    **RR > 0.6 s** — sustained over at least two beats (> 1.2 s), with successive
    events inside a 3-minute window aggregated. What the `.atr` file marks is the
    onset of that event, and measured against the `.qrsc` peaks it sits **one sample
    after the R peak that opens the first long interval**:

    | | Onsets |
    |---|---|
    | Within 2 samples of the opening R peak | **526** of 622 |
    | …of which exactly **one sample after** it | 493 |
    | …exactly **on** it | 32 |
    | Within 10 s of one | **622** of 622 |

    So `np.isin(onsets, rpeaks)` finds 32 matches out of 622 and looks like an
    off-by-one bug in ECGBench rather than a property of the annotation. Re-measure
    it yourself — it reads only annotation files, so it is quick:

    ```python
    from ecgbench.labels.picsdb import verify_bradycardia_onsets
    verify_bradycardia_onsets("/path/to/picsdb/1.0.0/")
    ```

    Two further points about the events. The `.atr` symbol is `[`, WFDB's
    start-of-ventricular-flutter marker being reused as a generic episode start — it
    **does not mean flutter**, and nothing should count it as a beat. And the
    shortest observed gap between consecutive onsets in a record is 369 s, well
    outside the 3-minute aggregation window, so the counts below are events rather
    than beats.
- type: description
  title: Clipping and dead signal that no quality check can see
  body: |-
    All 10 records pass every ECGBench check and `clean/` equals `original/`. That
    is arithmetically correct and substantively misleading, so here is what the
    validation report cannot tell you.

    **Every record clips at the 16-bit converter rail.** For eight of them it is 1 to
    15 samples. `infant5` sits at the negative rail for **422,773 samples (1,691 s,
    0.96% of the recording)** and `infant1` for 160,567 (642 s, 0.39%), at −40.96 mV
    — nothing an infant heart did. Because gain and baseline differ per record, the
    rail in millivolts differs too, so `amplitude_range_mv` is the union of ten
    different rails, **[−40.9604, +40.915] mV**. It is not a physiologic bound and
    must not be read as one; what it still catches is a mis-scaled copy.

    **Every record holds minutes of perfectly constant signal**, 239 s to 3,147 s of
    it, the longest single run being **1,456 s — 24 minutes — in `infant5`**.
    ECGBench's `flat_line` check tests variance over the *whole* record, and a 20–70
    hour recording passes that trivially. Read `flat_secs`, `flat_fraction` and
    `longest_flat_secs` from the labels before choosing a window.

    **R-peak annotation covers 94.0% to 99.9% of each record, not all of it.**
    `infant10` has a **7,667 s (2.13 h) unannotated tail** and 49 internal gaps over
    10 s; `infant5` opens with 1,631 s of unannotated signal and `infant2` with 409
    s. `rpeaks()` returning an empty array there means "not annotated here", not "no
    beats", and nothing in the WFDB headers says where the annotated span ends.

    One more geometry mismatch, since the release says the two signals are
    synchronised: **`infant5`'s respiration record runs 3,597 s (1.0 h) longer than
    its ECG**, and `infant1`'s 35 s longer. The other eight agree to within 4 s.
- type: description
  title: Ten records, ten infants, ten folds
  body: |-
    Folds are grouped on `subject_id`, which is 1:1 here — one ECG record per infant
    — so each fold is exactly one infant and the partition is leave-one-infant-out.
    Unlike `szdb`, nothing is reconstructed: the record name states the infant. It is
    declared rather than left `null` so the grouping stays correct if a re-release
    ever adds a second recording for an infant.

    With the default mapping that gives train = folds 1–8 (8 infants), **val = fold 9
    (`infant3`) and test = fold 10 (`infant6`)**. One infant is not an evaluation set;
    use `split=None` with `fold_numbers=[...]` for real cross-validation.

    **There is nothing to stratify on, and that is measured rather than assumed.**
    `StratifiedGroupKFold` raises unless some class holds at least `n_folds` records,
    and over ten records that admits exactly one class. Every candidate axis fails
    before it can be tried: a median cut on bradycardia rate (0.85–1.83/h) gives 5/5
    and raises `n_splits=10 cannot be greater than the number of members in each
    class`; sampling rate gives 8/2 and raises; lead name gives 7/2/1 and raises. The
    stratification label is therefore the constant `cohort_label`, which reduces the
    split to a plain partition of the ten infants. **Do not read the fold layout as
    balanced on anything.**
- type: description
  title: One channel, named three different ways
  body: |-
    Every record stores exactly **one** channel, and the ten headers disagree what to
    call it: `II` in seven, `ECG` in `infant1` and `infant5`, `I` in `infant10`. That
    is the `mitdb` problem at a lead count of one — `alternate_lead_names` is keyed by
    lead *count*, and the count never changes — so the config declares
    `record_lead_layouts: [["II"], ["ECG"], ["I"]]` and `ECGDataset` resolves `leads=`
    against each record's own header.

    The consequence is deliberate: **`leads=["II"]` returns a signal for seven
    records and raises for the other three.** The release says only "a single channel
    of a 3-lead electrocardiogram" and nothing anywhere states that the `ECG` channel
    is lead II, so handing it back under that name would let it be stacked with real
    lead II from other datasets. Omit `leads=` to take whatever channel each record
    holds.
- type: description
  title: About those counts
  body: |-
    Every figure on this page is recomputed from the shipped files. Nothing
    contradicts the release, and two things it does not state are worth recording.

    | Quantity | Release / paper | This release | Note |
    |---|---|---|---|
    | Infants | 10 | 10 | agrees |
    | ECG records | 10 | 10 | plus 10 respiration records, which get no fold |
    | Recording duration | "approximately 20–70 hours" | 20.34–70.32 h | agrees; 439.84 h in total |
    | ECG sampling rate | 500 Hz, 250 Hz for infants 1 and 5 | same | agrees |
    | Respiration rate | 50 Hz, 500 Hz for infant 1 | same | agrees |
    | Bradycardia onsets | not stated on the landing page | **622** | 28–97 per infant |
    | R peaks | not stated | **3,797,503** | manually verified |
    | Per-infant age and weight | cohort ranges only | **not shipped** | see below |

    **The event counts check out against an independent source.** The per-infant
    bradycardia counts recomputed here — 77, 72, 80, 66, 72, 56, 34, 28, 97, 40 —
    match, infant for infant, the table published in *Automated Medical Care:
    Bradycardia Detection and Cardiac Monitoring of Preterm Infants*
    ([PMC8625917](https://pmc.ncbi.nlm.nih.gov/articles/PMC8625917/)), which used the
    same database. So does its duration column. That paper additionally prints a
    per-infant post-conceptional age and birth weight, and **the PhysioNet release
    ships neither** — the headers carry no comment lines at all, no age, no sex, no
    start time. ECGBench attaches no demographics to any row, because the mapping in
    a third-party table cannot be verified against the released files.

    The stratification label is *not* a class breakdown: `cohort_label` is
    `preterm_infant` for all ten records, asserted by the release of the cohort and
    derived from nothing in the files. There is no negative class and no control
    group here.
- type: table
  title: The 10 records, recomputed from the files
  headers:
  - Rec
  - Hz
  - Lead
  - Hours
  - Brady
  - /h
  - R peaks
  - HR
  - SDNN ms
  - Cover
  - Rail s
  - Flat s
  - Fold
  rows:
  - - infant8
    - '500'
    - II
    - '24.60'
    - '28'
    - '1.14'
    - 204,532
    - '140'
    - '41'
    - 98.9%
    - '0.0'
    - '946'
    - 1 (train)
  - - infant10
    - '500'
    - I
    - '47.27'
    - '40'
    - '0.85'
    - 411,241
    - '154'
    - '39'
    - 94.0%
    - '0.4'
    - '849'
    - 2 (train)
  - - infant5
    - '250'
    - ECG
    - '48.75'
    - '72'
    - '1.48'
    - 411,149
    - '143'
    - '44'
    - 98.2%
    - '1,691'
    - '3,147'
    - 3 (train)
  - - infant1
    - '250'
    - ECG
    - '45.61'
    - '77'
    - '1.69'
    - 419,233
    - '154'
    - '29'
    - 99.5%
    - '642'
    - '642'
    - 4 (train)
  - - infant7
    - '500'
    - II
    - '20.34'
    - '34'
    - '1.67'
    - 195,072
    - '161'
    - '34'
    - 99.2%
    - '0.0'
    - '570'
    - 5 (train)
  - - infant2
    - '500'
    - II
    - '43.84'
    - '72'
    - '1.64'
    - 333,604
    - '130'
    - '51'
    - 97.8%
    - '0.0'
    - '2,474'
    - 6 (train)
  - - infant9
    - '500'
    - II
    - '70.32'
    - '97'
    - '1.38'
    - 626,628
    - '149'
    - '40'
    - 99.7%
    - '0.0'
    - '853'
    - 7 (train)
  - - infant4
    - '500'
    - II
    - '46.78'
    - '66'
    - '1.41'
    - 465,565
    - '166'
    - '26'
    - 99.9%
    - '0.0'
    - '239'
    - 8 (train)
  - - infant3
    - '500'
    - II
    - '43.71'
    - '80'
    - '1.83'
    - 335,087
    - '130'
    - '45'
    - 98.4%
    - '0.0'
    - '2,522'
    - 9 (val)
  - - infant6
    - '500'
    - II
    - '48.61'
    - '56'
    - '1.15'
    - 395,392
    - '136'
    - '27'
    - 99.5%
    - '0.0'
    - '765'
    - 10 (test)
  - - '**Total**'
    - —
    - —
    - '**439.84**'
    - '**622**'
    - —
    - '**3,797,503**'
    - —
    - —
    - '**98.5%**'
    - '**2,334**'
    - '**13,007**'
    - '**10 folds**'
  footnote: |-
    Record names carry an `_ecg` suffix (`infant8_ecg`); the `_resp` companions get no
    fold. HR and SDNN are whole-record summaries over RR intervals in [0.2 s, 3.0 s] —
    descriptive of a 20–70 hour recording, not a segmented HRV analysis. **Cover** is
    the fraction of recorded time carrying R-peak annotation, excluding the
    unannotated head, the unannotated tail and every internal gap over 10 s. **Rail
    s** is time clipped at the 16-bit converter rail and **Flat s** time in constant
    runs of a second or more; neither is visible to any ECGBench quality check, and
    all 10 records are `is_valid`.
- type: code
  title: Loading with ECGBench
  language: python
  body: |
    from ecgbench import ECGDataset
    from ecgbench.labels.picsdb import bradycardia_onsets, rpeaks

    # window= is not optional here: records are 36.6M to 253.1M samples, so a batch
    # of whole records is impossible. It is pushed into wfdb as sampfrom/sampto, so
    # only these samples are decoded.
    #
    # AND IT COUNTS SAMPLES, NOT SECONDS: 15,000 is 30 s of the eight 500 Hz
    # records and 60 s of infant1 and infant5.
    ds = ECGDataset(
        "picsdb",
        split="train",
        labels=True,
        window=(0, 15_000),
        data_path="/path/to/picsdb/1.0.0/",
    )

    len(ds)                                  # 8
    sample = ds[0]
    sample["signal"].shape                   # (1, 15000)
    sample["record_id"]                      # 'infant10_ecg'
    sample["labels"]["subject_id"]           # 'infant10'
    sample["labels"]["sampling_rate"]        # 500        <- READ THIS BEFORE CONVERTING TO SECONDS
    sample["labels"]["lead_name"]            # 'I'        <- 'II' in seven records, 'ECG' in two
    sample["labels"]["n_bradycardias"]       # 40
    sample["labels"]["n_rpeaks"]             # 411241
    sample["labels"]["mean_hr_bpm"]          # 154.3      <- an infant, not an adult
    sample["labels"]["annotated_fraction"]   # 0.9397     <- its last 2.13 h carry no R peaks
    sample["labels"]["flat_secs"]            # 849.0      <- constant signal no check can see

    # The intended use: cut a window around a real event. The onsets are seconds
    # into the record, so scale by THAT record's rate.
    fs = int(sample["labels"]["sampling_rate"])
    onset = float(sample["labels"]["bradycardia_onsets_secs"].split("|")[0])   # 813.164
    start = int((onset - 15) * fs)                                            # 399082
    event = ECGDataset("picsdb", split="train", window=(start, 30 * fs),
                       data_path="/path/to/picsdb/1.0.0/")

    # Both annotation layers re-base onto the same window, so they index the tensor.
    bradycardia_onsets("/path/to/picsdb/1.0.0/", "infant10_ecg", start, 30 * fs)  # [7500]
    rpeaks("/path/to/picsdb/1.0.0/", "infant10_ecg", start, 30 * fs)             # 59 peaks
- type: code
  title: Building the splits
  language: bash
  body: |
    # No flags: 10 records from 10 infants make ten folds of one infant each.
    # The first run reads all 1.58 billion samples once to measure converter
    # clipping and constant runs, and caches the result as ecgbench_metadata.csv
    # in the dataset root — so that root must be writable.
    ecgbench splits --dataset picsdb --data-path /path/to/picsdb/1.0.0/
- type: links
  title: Links
  links:
  - label: PhysioNet — picsdb 1.0.0
    url: https://physionet.org/content/picsdb/1.0.0/
  - label: 'Gee et al., IEEE TBME 64(9):2300-2308 (2017)'
    url: https://doi.org/10.1109/TBME.2016.2632746
  - label: Example script — examples/load_picsdb.py
    url: https://github.com/vlbthambawita/ECGBench/blob/main/examples/load_picsdb.py
---
