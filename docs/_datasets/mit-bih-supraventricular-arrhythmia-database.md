---
slug: "mit-bih-supraventricular-arrhythmia-database"
name: "MIT-BIH Supraventricular Arrhythmia Database"
category: "two-lead"
order: 5
status: "completed"
source_url: "https://physionet.org/content/svdb/1.0.0/"
url_label: "physionet.org"
format: "2-lead (ECG1 + ECG2, unnamed) · 30 min · 128 Hz · WFDB"
patients: "—"
records: "78"
access: "open"
license: "ODC-By 1.0"
origin_institution: "Harvard-MIT HST"
origin_country: "USA"
leads: 2
paper_title: "Greenwald, PhD thesis, Harvard-MIT 1990"
paper_doi: "https://doi.org/10.13026/C2V30W"
search_keywords: "mit-bih supraventricular arrhythmia svdb usa harvard-mit hst greenwald svt pac pjc supraventricular ectopic beat sveb aami ec57 beat annotations signal quality two-lead holter ambulatory"
patients_class: "count-na"

sections:
  - type: description
    title: "Overview"
    body: |
      **78 half-hour recordings assembled to supply what the MIT-BIH Arrhythmia
      Database lacks.** That database holds 109,494 beats but only 2,781
      supraventricular ones — 2.54%. This one holds **12,198 supraventricular
      ectopic beats in 184,583**, or **6.61%**, over 39 hours at 128 Hz. It exists
      for exactly one reason and the selection shows.

      **The beat symbol for a supraventricular beat is `S` here and `A` in
      mitdb.** All 12,188 supraventricular beats in this release are annotated
      `S`; mitdb annotates its 2,546 as `A` and uses `S` exactly twice.
      Concatenating the two databases on the raw symbol trains a model on two
      disjoint vocabularies for one phenomenon. **This is the single most
      important thing on this page.** The label loader exposes
      `aami_N/S/V/F/Q` — the AAMI EC57 five-class reduction, under which `A`,
      `a`, `J` and `S` all collapse to `S` — and that is the column to combine on.

      **The headers carry no comment lines at all.** No age, no sex, no subject
      identifier, no medications, no clinical description — not withheld, simply
      never released. mitdb has all of these in its header comments and nsrdb has
      age and sex; this database has none, and PhysioNet does not even state how
      many subjects its 78 recordings represent. Folds are therefore stratified
      but **ungrouped**, and nothing here can be linked to any other database.

      **There is effectively no rhythm annotation layer.** The entire release
      contains **one** `+` annotation — `(N` at sample 112,143 of record 852 —
      against 1,291 in mitdb. There is no dominant rhythm to be had, so the
      record-level label is derived from the beat counts rather than released.

      **The two channels are not named leads.** The headers call them `ECG1` and
      `ECG2` and the release states no electrode placement anywhere. Do not carry
      mitdb's MLII/V1 naming across — this catalogue's own entry claimed
      "MLII + V1 · 360 Hz" before the config was written, and both halves of that
      were wrong.

  - type: table
    title: "Reference beat annotations, recomputed from the 78 .atr files"
    headers: ["Beat type", "Symbol", "Count", "Share", "Records containing it"]
    rows:
      - ["normal beat", "N", "162,339", "87.95%", "78"]
      - ["supraventricular premature or ectopic beat", "S", "12,188", "6.60%", "73"]
      - ["premature ventricular contraction", "V", "9,943", "5.39%", "67"]
      - ["unclassifiable beat", "Q", "79", "0.04%", "20"]
      - ["fusion of ventricular and normal beat", "F", "23", "0.01%", "6"]
      - ["nodal (junctional) premature beat", "J", "9", "0.005%", "1"]
      - ["aberrated atrial premature beat", "a", "1", "0.001%", "1"]
      - ["bundle branch block beat (unspecified)", "B", "1", "0.001%", "1"]
      - ["**total**", "", "**184,583**", "**100%**", "**78**"]

  - type: table
    title: "The same beats under the AAMI EC57 reduction — use these to combine with mitdb"
    headers: ["AAMI class", "Built from", "Count", "Share", "Records containing it"]
    rows:
      - ["N — normal / bundle branch block", "N, L, R, e, j, B", "162,340", "87.95%", "78"]
      - ["S — supraventricular ectopic", "A, a, J, S", "12,198", "6.61%", "73"]
      - ["V — ventricular ectopic", "V, E", "9,943", "5.39%", "67"]
      - ["F — fusion", "F", "23", "0.01%", "6"]
      - ["Q — unclassifiable / paced", "/, f, Q", "79", "0.04%", "20"]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the 78 `.atr` files and
      headers, after verifying the shipped data against the release's own
      `SHA256SUMS.txt` — **all 416 listed files match**, including the 78 `.hea-`
      and 24 `.atr-` files. Those are PhysioNet's superseded revisions kept beside
      the current ones (the 24 revised `.atr` files are dated 2010 against the
      originals' 1992), not extra records; the record list comes from the shipped
      `RECORDS` file, so they cannot enter the partition.

      **There is no published table to disagree with.** PhysioNet's description
      of this database is a single sentence — "78 half-hour ECG recordings chosen
      to supplement the examples of supraventricular arrhythmias in the MIT-BIH
      Arrhythmia Database" — and states no beat counts, no subject count, no
      sampling rate and no lead names. Unusually for this catalogue, then, the
      only figures that exist are the recomputed ones. The record count matches;
      everything else on this page is new.

      **This catalogue's previous entry was wrong, in two ways worth naming.** It
      described the format as "2-lead (MLII + V1) · 30 min · 360 Hz". The
      recordings are **128 Hz**, not 360, and the headers name the channels
      **`ECG1` and `ECG2`** with no electrode placement stated anywhere in the
      release. Both errors are the kind that come from assuming a sibling
      database's properties carry across; they did not.

      **The database is not uniformly supraventricular.** `sveb_fraction` runs
      from **0.000 to 0.575**. Five records — 802, 803, 804, 805, 893 — contain no
      supraventricular ectopy whatsoever, and 811 contains a single beat. At the
      other end, **record 865 is the only record in this catalogue where ectopic
      beats outnumber normal ones**: 1,818 supraventricular and 235 ventricular
      against 1,102 normal, in 3,162 beats.

      **Ventricular ectopy is here too, and is not incidental.** 9,943 beats,
      5.39% of the release, in 67 of the 78 records — and three records are more
      ventricular than supraventricular by a wide margin (860 at 30.1% VEB, 879 at
      26.3%, 892 at 20.3%). A model trained here on "ectopic vs not" learns both
      classes whether or not that was intended.

      **The rare classes cannot support evaluation.** 23 fusion beats in 6
      records, 9 nodal premature beats in **one** record (878), one aberrated
      atrial premature beat (821) and one bundle branch block beat (868). They are
      exposed for completeness, not as trainable classes.

      **Rate and variability are descriptive, not a result.** `mean_hr_bpm`,
      `sdnn_ms` and `rmssd_ms` are computed over RR intervals in [0.3 s, 2.0 s]
      from the reference beats — 407 intervals are rejected across the release.
      Mean heart rate spans **47.8 bpm (811) to 137.3 bpm (848)**, and beats per
      record 1,436 to 4,259, the same two records at either end. These are
      whole-record summaries over a rhythm that is *by construction* ectopic, so
      on a record like 865 they describe the recording and not the subject's
      sinus node.

  - type: table
    title: "SVEB burden bands — the stratification label"
    headers: ["Band", "SVEB share of beats", "Records", "SVEB beats", "Note"]
    rows:
      - ["minimal", "0 – 1%", "21", "217", "includes the 5 records with none at all"]
      - ["low", "1 – 3%", "20", "808", ""]
      - ["moderate", "3 – 10%", "23", "3,115", ""]
      - ["high", "> 10%", "14", "8,058", "66% of all SVEB in the release"]

  - type: description
    title: "Why the folds are stratified on ectopy burden"
    body: |
      **Because there is nothing else.** No demographics, no subject identifiers,
      no diagnoses, and — with one `+` annotation in the whole release — no rhythm
      labels. What differs between records, and what the database was assembled to
      vary, is how much supraventricular ectopy each one holds.

      **The band edges are fixed fractions, not quantiles.** 1%, 3% and 10% of
      beats give 21 / 20 / 23 / 14 records, so every band clears the 10 members
      `StratifiedKFold` needs and no fold can end up holding only zero-ectopy
      records. Quantile edges would balance the bands slightly better and would
      move every record's label the next time a record is added or an annotation
      revised; fixed edges reproduce against a re-release, which matters more for
      a partition that gets published. The values are conventional clinical
      granularity for ectopic burden rather than anything this release states.

      The result is that **every one of the ten folds carries all four bands** —
      each fold holds 1–2 `high`, 2 `low`, 2–3 `minimal` and 2–3 `moderate`
      records. With 78 records over 10 folds, `train` holds 64, `val` 7 and
      `test` 7.

      `sveb_burden` is a **fold-construction label**, and a coarsening of a
      continuous quantity at that. Train on `aami_S`, `aami_V`, `aami_N`, the
      `beat_*` counts, or `sveb_fraction` itself — never on `stratify_class`.

      Folds are **ungrouped**, because the release ships no subject identifier of
      any kind. mitdb can group its records by the analog tape number in the
      header comment; this database has no header comments. It follows that
      nothing rules out one person having contributed several of the 78
      recordings, and no split here can protect against that.

  - type: description
    title: "Signal quality is annotated, per channel"
    body: |
      The shipped `ANNOTATORS` file promises "reference beat **and signal
      quality** annotations". The `~` annotations mark quality transitions, and
      their WFDB `subtype` is a bitmask over the two channels — `0` clean, `1`
      ECG1 noisy, `2` ECG2 noisy, `3` both. Each opens an interval running to the
      next, so ECGBench exposes it as **seconds per state per record**, not as a
      marker count.

      Across the release **97.41% of the recorded time is annotated clean** — 0.37
      h of ECG1-only noise, 0.45 h of ECG2-only and 0.19 h of both, out of 39.00 h.
      The average hides a wide spread: **35 of the 78 records carry no quality
      annotation at all**, while record **868 is 22.4% noisy**, 887 is 19.7% and
      854 is 19.2%. The `|` isolated-artifact marker varies independently — 2,211
      across the release, 508 of them in record 865 alone, none in 32 records — so
      a per-record metric is not comparable across records without controlling for
      both.

      **The leading span is not always asserted, and here that assumption is not
      free.** In 39 of the 43 records carrying a `~`, the first one is a
      transition *into* noise, so everything before it is clean by implication. In
      the other **four it is a transition into clean**: 803, 855, 857 and 885.
      Nothing ever asserted what those spans were. For 855, 857 and 885 they are
      1.8 s, 6.6 s and 3.3 s and the point is academic. For **record 803 it is
      1,555.1 s — 86% of the record**. ECGBench counts these spans as clean, which
      is what WFDB itself does, and reports them in
      `quality_head_unasserted_secs` so the assumption is visible rather than
      buried.

  - type: description
    title: "Beat annotation covers the whole record — unlike nsrdb"
    body: |
      Worth stating because the sibling database traps people here. In
      **MIT-BIH Normal Sinus Rhythm**, beat annotation stops one to five hours
      before the signal does, silently. In this database it does not: the first
      beat falls between **0.01 s and 1.27 s** and the last between **1798.87 s
      and 1800.00 s** in every one of the 78 records, so `annotated_fraction`
      never drops below **0.9988**.

      Any window inside any record therefore has reference annotation behind it.
      `annotated_secs`, `unannotated_head_secs`, `unannotated_tail_secs` and
      `annotated_fraction` are reported anyway, so a re-release that changes this
      cannot do so silently.

  - type: table
    title: "Validation summary (128 Hz)"
    headers: ["Version", "Records", "Note"]
    rows:
      - ["original", "78", "all records, with is_valid + quality_issues"]
      - ["clean", "78", "100% pass rate — nothing is excluded"]

  - type: description
    title: "Nothing fails validation, and the amplitude range needs explaining"
    body: |
      All 78 records pass every check, so `original` and `clean` hold the same 78
      rows. There are no NaN samples, no flat or all-zero leads and no unreadable
      header anywhere. Records are exactly **230,400 samples** each, uniformly, so
      unlike `afdb` and `nsrdb` the `truncated_signal` check is switched on rather
      than disabled.

      **`amplitude_range_mv` is `[-10.24, 10.235]`, and it is deliberately not the
      declared resolution.** Every signal line declares `10` bits with `adc_zero`
      0, which would put every sample in [−512, 511] adu = ±2.56 mV. **The data
      does not respect that**: samples run to −1125 and +1022 adu, i.e. **−5.625
      to +5.11 mV**, so the declared resolution is nominal and a threshold set
      from it would fail most of the release. What actually bounds the samples is
      format 212's 12 bits — [−2048, 2047] adu at 200 adu/mV.

      So the check cannot fire on SVDB 1.0.0, which is the intended state. What it
      guards is a mis-scaled copy — microvolts, or a re-release declaring a real
      gain on the uncalibrated records — which would exceed it on the first
      record.

  - type: description
    title: "Half the headers declare an uncalibrated gain"
    body: |
      **37 of the 78 records declare a gain of `0`** on both signal lines, WFDB's
      code for "uncalibrated"; the other 41 declare `200`. `wfdb` substitutes its
      default of 200 adu/mV wherever it reads 0, so `rdrecord` reports millivolts
      for all 78 and `adc_gain` comes back as 200.0 for every one of them.
      ECGBench's `signal_unit_scale` is therefore `1.0` and nothing is rescaled.

      Waveform shape is unaffected either way; absolute calibration of those 37
      records rests on wfdb's default rather than on anything the release states.
      `header_declares_uncalibrated` in the labels records which is which, so an
      analysis that cares about absolute amplitude can exclude them.

  - type: description
    title: "Overlap with the other MIT-BIH Holter databases: none found"
    body: |
      This database, the **MIT-BIH Arrhythmia Database** and the **MIT-BIH Normal
      Sinus Rhythm Database** come from the same laboratory and era, and none of
      them ships a subject identifier that would join, so the question was settled
      from the annotation files rather than assumed. RR intervals in seconds are
      commensurable across sampling rates (128 Hz here, 360 Hz for mitdb), so the
      check compares **sequences of 20 consecutive RR intervals quantised to
      10 ms**.

      Against controls that make a null result mean something — a positive control
      re-finding each SVDB record in its own pool at **100%**, and a negative
      control between *different* SVDB records topping out at **27 shared
      sequences** — the result is:

      - **0 shared sequences** between any SVDB record and any MIT-BIH Arrhythmia
        Database record;
      - **at most 11** between any SVDB record and any NSRDB record, well under
        the 27 the negative control shows two genuinely distinct records can reach
        by chance.

      No `related:` edge is declared on those grounds, matching the other MIT-BIH
      entries in this catalogue. The limitation is the same one stated there: an
      RR signature survives *refinement* of annotations but not *re-detection*, so
      a shared recording annotated by two genuinely different detectors could
      evade it. **Subject-level** overlap cannot be checked at all — this release
      ships nothing that identifies a subject.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset svdb --data-path /path/to/svdb/1.0.0/

  - type: code
    title: "Loading with ECGBench"
    language: python
    body: |
      from ecgbench import ECGDataset

      # 230,400 samples per record; window= is pushed into the reader, so it
      # avoids decoding the other 29 minutes rather than cropping afterwards.
      ds = ECGDataset(
          "svdb",
          split="train",
          data_path="/path/to/svdb/1.0.0/",
          window=(0, 1280),          # first 10 s at 128 Hz
          labels=True,
      )

      len(ds)                                   # 64
      ds[0]["signal"].shape                     # torch.Size([2, 1280])
      ds[0]["record_id"]                        # 800
      ds.lead_names                             # ('ECG1', 'ECG2') — channel positions,
                                                # not named leads
      ds[0]["labels"]["n_beats"]                # 1883
      ds[0]["labels"]["aami_S"]                 # 30    <- combine with mitdb on this,
      ds[0]["labels"]["beat_S"]                 # 30       not on the raw symbol
      ds[0]["labels"]["sveb_fraction"]          # 0.0159
      ds[0]["labels"]["sveb_burden"]            # 'low'  <- for folds, not for training
      ds[0]["labels"]["mean_hr_bpm"]            # 62.8
      ds[0]["labels"]["noisy_fraction"]         # 0.0629

      # Every record is exactly 1800 s, so any window inside that fits all 78 —
      # this is one of the few datasets here where that is true.

  - type: links
    title: "References"
    items:
      - { label: "PhysioNet page", url: "https://physionet.org/content/svdb/1.0.0/" }
      - { label: "Dataset DOI", url: "https://doi.org/10.13026/C2V30W" }
      - { label: "PhysioBank annotation definitions", url: "https://archive.physionet.org/physiobank/annotations.shtml" }
      - { label: "ANSI/AAMI EC57 beat classes", url: "https://webstore.ansi.org/standards/aami/ansiaamiec572012r2020" }
---
