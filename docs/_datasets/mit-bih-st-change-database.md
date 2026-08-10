---
slug: mit-bih-st-change-database
name: MIT-BIH ST Change Database
category: two-lead
order: 11
status: completed
source_url: https://physionet.org/content/stdb/1.0.0/
url_label: physionet.org
format: 2-lead on 18 records, 1-lead on 10 (ECG1/ECG2, both unnamed) · 13.1–67.2 min · 360 Hz · WFDB · mostly exercise stress ECGs
patients: —
records: '28'
access: open
license: ODC Attribution
origin_institution: MIT
origin_country: USA
leads: 2
paper_title: Albrecht, S-T segment characterization for long-term automated ECG analysis, MS thesis, MIT 1983
paper_doi: https://doi.org/10.13026/C2ZW2H
search_keywords: mit-bih st change stdb exercise stress test st depression elevation transient ischemia usa mit physionet albrecht beat annotations single channel
patients_class: count-na
sections:
- type: description
  title: Overview
  body: |-
    28 short ECG recordings, most of them exercise stress tests, selected because
    they exhibit transient ST change — 13.49 hours at 360 Hz, digitised for Paul
    Albrecht's 1983 MIT master's thesis and published by PhysioNet in 1999. The last
    five records (323–327) are excerpts of long-term ambulatory recordings and show
    ST **elevation**; the rest show transient ST **depression**.

    It is the smallest and thinnest release in this catalogue. There is no metadata
    file, no clinical table, and no header comment line of any kind — no age, no sex,
    no diagnosis, no recording date and no subject identifier anywhere in the 142
    shipped files. Everything ECGBench exposes is derived from the signal headers and
    the reference beat annotations, except the exercise/long-term grouping, which is
    transcribed from one sentence of the landing page and flagged as such by a
    `group_source` column.

    All 142 files verify against the release's own `SHA256SUMS.txt`, all 28 records
    pass every ECGBench quality check, and `clean/` therefore equals `original/` —
    which is the exception rather than the rule among the MIT-BIH-family Holter
    datasets here.
- type: description
  title: The name promises ST annotations. There are none.
  body: |-
    **This is the single most important thing to know about the database, and its
    name works against you.** PhysioNet states it on the landing page — the annotation
    files "contain only beat labels; they do not include ST change annotations, as in
    the European ST-T Database" — and the files bear it out: 76,175 of the 76,181
    annotations in the 28 `.atr` files are beat labels, the other six are
    signal-quality markers, and there is not a single `+` rhythm marker, `s` ST
    episode or non-empty `aux_note` anywhere in the release.

    There is no ST measurement, no episode boundary, no onset, no deviation in
    millivolts and no per-record ST label of any kind. What this release offers is 28
    recordings *selected* for ST change, with reference beats.

    | Want | Use |
    |---|---|
    | Annotated ST episodes with onset, extent and peak deviation | [European ST-T Database](european-st-t-database-edb.html) — 802 episodes |
    | ST episodes over 24-hour records | [Long-Term ST Database](long-term-st-database-ltstdb.html) |
    | Recordings selected for ST change, with reference beats | this database |

    The `st_change_type` label ECGBench exposes is the landing page's own grouping of
    its records, transcribed. It is not derived from the signals and it is not an ST
    measurement — `group_source` says `landing_page` on every row so a reader of the
    CSV can tell without this page in front of them.
- type: description
  title: Ten of the 28 records hold one channel, not two
  body: |-
    Nothing on the landing page mentions it, and the "2 leads" in the table above is
    true of 18 records. **Records 313, 314, 315, 316, 317, 319, 320, 321, 322 and 323
    declare a single signal.**

    The consequence is immediate rather than theoretical: `ecg_collate_fn` stacks
    signals with torch's `default_collate`, so a batch drawing from both layouts
    raises

    ```
    RuntimeError: stack expects each tensor to be equal size,
                  but got [2, 10800] at entry 0 and [1, 10800] at entry 2
    ```

    The fix is to ask for a lead every record has:

    ```python
    ECGDataset("stdb", split="train", leads=["ECG1"], window=(0, 10_800))
    ```

    The config declares `alternate_lead_names: {1: ["ECG1"]}`, which is what makes
    `leads=["ECG2"]` **raise** for those ten records against a named layout instead of
    silently returning ECG1.

    `ECG1`/`ECG2` are channel positions, not lead names. Every signal line of every
    header ends in the bare description `ECG`, and the release states no electrode
    placement anywhere. Do **not** read them as MLII/V1 by analogy with `mitdb` —
    the temptation is strongest here because this release shares mitdb's 360 Hz
    sampling rate and its three-digit record numbering, and nothing whatsoever
    supports it.
- type: table
  title: The 28 records, recomputed from the files
  headers:
  - Rec
  - Ch
  - Min
  - Group
  - ST
  - Beats
  - S
  - V
  - HR base
  - HR peak
  - HR rise
  - Gain (adu/mV)
  - Fold
  rows:
  - - '300'
    - '2'
    - '24.9'
    - exercise
    - depression
    - 2,558
    - '0'
    - '2'
    - '93'
    - '114'
    - '21.2'
    - 296 / 300
    - '6'
  - - '301'
    - '2'
    - '32.4'
    - exercise
    - depression
    - 2,497
    - '1'
    - '11'
    - '57'
    - '127'
    - '69.9'
    - 295 / 300
    - '8'
  - - '302'
    - '2'
    - '23.6'
    - exercise
    - depression
    - 2,113
    - '0'
    - '0'
    - '62'
    - '124'
    - '61.8'
    - 295 / 213
    - '1'
  - - '303'
    - '2'
    - '33.9'
    - exercise
    - depression
    - 3,005
    - '5'
    - '16'
    - '86'
    - '101'
    - '15.1'
    - 457 / 291
    - '9'
  - - '304'
    - '2'
    - '30.3'
    - exercise
    - depression
    - 1,852
    - '0'
    - '0'
    - '54'
    - '83'
    - '29.5'
    - 292 / 320
    - '3'
  - - '305'
    - '2'
    - '13.1'
    - exercise
    - depression
    - 1,036
    - '2'
    - '265'
    - '57'
    - '115'
    - '58.0'
    - 204 / 228
    - '5'
  - - '306'
    - '2'
    - '67.2'
    - exercise
    - depression
    - 6,527
    - '0'
    - '0'
    - '65'
    - '180'
    - '114.8'
    - 204 / 229
    - '2'
  - - '307'
    - '2'
    - '36.8'
    - exercise
    - depression
    - 2,469
    - '1'
    - '0'
    - '58'
    - '102'
    - '43.7'
    - 176 / 182
    - '1'
  - - '308'
    - '2'
    - '29.4'
    - exercise
    - depression
    - 2,299
    - '79'
    - '14'
    - '57'
    - '118'
    - '61.4'
    - 296 / 275
    - '10'
  - - '309'
    - '2'
    - '41.4'
    - exercise
    - depression
    - 5,149
    - '0'
    - '1'
    - '85'
    - '176'
    - '90.4'
    - 206 / 275
    - '3'
  - - '310'
    - '2'
    - '19.0'
    - exercise
    - depression
    - 2,410
    - '0'
    - '1'
    - '94'
    - '179'
    - '84.9'
    - 296 / 161
    - '4'
  - - '311'
    - '2'
    - '30.5'
    - exercise
    - depression
    - 3,009
    - '0'
    - '0'
    - '77'
    - '158'
    - '81.0'
    - 178 / 205
    - '7'
  - - '312'
    - '2'
    - '27.9'
    - exercise
    - depression
    - 2,340
    - '11'
    - '0'
    - '61'
    - '141'
    - '80.6'
    - 298 / 182
    - '2'
  - - '313'
    - '1'
    - '23.2'
    - exercise
    - depression
    - 2,701
    - '2'
    - '0'
    - '79'
    - '182'
    - '103.4'
    - '295'
    - '1'
  - - '314'
    - '1'
    - '26.1'
    - exercise
    - depression
    - 2,121
    - '5'
    - '1'
    - '66'
    - '112'
    - '45.6'
    - '206'
    - '8'
  - - '315'
    - '1'
    - '26.2'
    - exercise
    - depression
    - 3,274
    - '0'
    - '1'
    - '80'
    - '161'
    - '80.2'
    - '231'
    - '5'
  - - '316'
    - '1'
    - '25.7'
    - exercise
    - depression
    - 3,351
    - '0'
    - '1'
    - '94'
    - '189'
    - '94.5'
    - '200'
    - '2'
  - - '317'
    - '1'
    - '27.7'
    - exercise
    - depression
    - 2,776
    - '5'
    - '0'
    - '69'
    - '161'
    - '92.1'
    - '265'
    - '6'
  - - '318'
    - '2'
    - '27.1'
    - exercise
    - depression
    - 3,531
    - '0'
    - '1'
    - '91'
    - '170'
    - '79.1'
    - 319 / 270
    - '4'
  - - '319'
    - '1'
    - '23.6'
    - exercise
    - depression
    - 2,559
    - '0'
    - '0'
    - '87'
    - '164'
    - '76.6'
    - '215'
    - '7'
  - - '320'
    - '1'
    - '32.2'
    - exercise
    - depression
    - 3,135
    - '2'
    - '0'
    - '79'
    - '155'
    - '75.8'
    - '223'
    - '3'
  - - '321'
    - '1'
    - '22.9'
    - exercise
    - depression
    - 2,115
    - '0'
    - '0'
    - '73'
    - '131'
    - '58.2'
    - '163'
    - '10'
  - - '322'
    - '1'
    - '13.2'
    - exercise
    - depression
    - 1,508
    - '3'
    - '0'
    - '92'
    - '134'
    - '41.7'
    - '366'
    - '9'
  - - '323'
    - '1'
    - '42.9'
    - '**long-term**'
    - '**elevation**'
    - 5,290
    - '0'
    - '0'
    - '84'
    - '172'
    - '88.0'
    - '362'
    - '4'
  - - '324'
    - '2'
    - '30.1'
    - '**long-term**'
    - '**elevation**'
    - 1,740
    - '363'
    - '4'
    - '67'
    - '67'
    - '0.0'
    - 300 / 300
    - '8'
  - - '325'
    - '2'
    - '21.2'
    - '**long-term**'
    - '**elevation**'
    - 1,465
    - '0'
    - '0'
    - '76'
    - '79'
    - '3.1'
    - 300 / 400
    - '7'
  - - '326'
    - '2'
    - '36.6'
    - '**long-term**'
    - '**elevation**'
    - 2,075
    - '336'
    - '4'
    - '58'
    - '66'
    - '7.9'
    - 300 / 500
    - '5'
  - - '327'
    - '2'
    - '19.9'
    - '**long-term**'
    - '**elevation**'
    - 1,270
    - '0'
    - '0'
    - '55'
    - '82'
    - '27.7'
    - 300 / 400
    - '6'
- type: description
  title: About those counts
  body: |-
    Nothing in this release publishes a table to disagree with — there is no paper
    table, no per-record listing on the landing page and no shipped metadata — so
    every figure above is computed from the files rather than reconciled against a
    published one. Three things it shows that the release does not state:

    **The heart rate contradicts the grouping for one record.** The landing page names
    the five long-term excerpts per record and leaves the other 23 to the word "most",
    so the exercise assignment is by exclusion. `hr_rise_bpm` (peak minus opening
    60-second mean) is the measurable check, and an exercise test has a characteristic
    ramp-and-recover shape. Records 324, 325 and 326 rise 0.0, 3.1 and 7.9 bpm and
    never pass 79 bpm — flat, as an ambulatory excerpt should be. **Record 323 ramps
    84 → 172 bpm and is still at 117 bpm in its final minute**, which looks nothing
    like the other four, and it is also the only single-channel record among them.
    Within the exercise group the rise runs 15.1–114.8 bpm, weakest at 303 (15.1), 300
    (21.2) and 304 (29.5) — the release's "most" showing up in the data. ECGBench
    still labels by the landing page; the column is how you check it.

    **Ectopy is almost absent, and what there is sits in three records.** 1,137 of
    76,175 beats are not normal. Record 305 holds 265 of the release's 322 `V`, and
    324 and 326 together hold 699 of its 815 `S`. Nine records have no ectopic beat at
    all and nineteen have fewer than five, so there is no arrhythmia task here and no
    usable ectopy stratification.

    **The gain is per record and per channel** — 31 distinct values from 161 to 500
    adu/mV, with record 326 declaring 300 for one channel and 500 for the other, and
    no two records sharing a full pair. `wfdb` applies each record's own, so signals
    arrive in millivolts regardless; it matters because the 12-bit rail moves with it,
    which is what `amplitude_range_mv` is computed from (±2047/161 = ±12.71 mV,
    rounded outward and never attained — the widest amplitude in the release is +6.94
    mV in record 302).

    Beat annotation is effectively complete: it starts 0.2–1.0 s into each record and
    ends 0.1–0.9 s before its end, covering 99.77–99.98% of every record. There is no
    multi-hour unannotated tail to window around, unlike `nsrdb`. Signal quality is
    annotated in **exactly one record** — 319 carries the release's only six `~`
    markers, and 86.2% of it is marked not-clean — so the absence of noise markers in
    the other 27 means nobody marked them, not that the signal is clean.
- type: description
  title: The default val and test splits hold two records each
  body: |-
    28 records over 10 folds gives eight folds of three and two of two, and the
    default 8/1/1 layout therefore makes `val` and `test` **two records each**. Both
    happen to be exercise/depression records, so **neither default evaluation split
    contains a single ST-elevation record.** That is arithmetic, not a defect: there
    are only five elevation records in the whole release.

    If you need the elevation group represented in evaluation, do not use the default
    split — pass `split=None` with `fold_numbers` to select by fold across the whole
    partition, or run your own cross-validation over `folds.csv`:

    ```python
    ECGDataset("stdb", split=None, fold_numbers=[4, 5, 6, 7, 8])
    ```

    Folds are stratified on the ST-change group crossed with the channel count
    (`depression_2ch` 14, `depression_1ch` 9, `elevation_2ch` 4, `elevation_1ch` 1),
    so every fold holds at least one two-channel record and nine of the ten hold a
    single-channel record. There is **no patient grouping**, and here that is a gap
    rather than a choice: the release identifies its subjects in no way at all, so
    whether any two of these 28 recordings came from the same person is unknowable
    from the files. A record-level split is the strongest guarantee available.
- type: code
  title: Loading with ECGBench
  language: python
  body: |
    from ecgbench import ECGDataset

    # leads=["ECG1"] is what makes this batchable: ten of the 28 records hold
    # one channel, and a batch mixing layouts raises in collation.
    ds = ECGDataset(
        "stdb",
        split="train",
        labels=True,
        leads=["ECG1"],
        window=(0, 10_800),      # 30 s; must fit record 305's 282,341 samples
        data_path="/path/to/stdb/1.0.0/",
    )

    sample = ds[0]
    sample["signal"].shape        # (1, 10800)
    sample["record_id"]           # 300
    sample["labels"]["n_channels"]      # 2
    sample["labels"]["st_change_type"]  # 'depression'  <- transcribed, not measured
    sample["labels"]["group_source"]    # 'landing_page'
    sample["labels"]["hr_rise_bpm"]     # 21.2         <- measured, checks the above
- type: code
  title: Building the splits
  language: bash
  body: |
    ecgbench splits --dataset stdb --data-path /path/to/stdb/1.0.0/
---
