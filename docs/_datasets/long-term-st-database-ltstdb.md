---
slug: "long-term-st-database-ltstdb"
name: "Long-Term ST Database (LTSTDB)"
category: "two-lead"
order: 12
status: "completed"
source_url: "https://physionet.org/content/ltstdb/1.0.0/"
url_label: "physionet.org"
format: "2- or 3-lead ambulatory · 12 lead layouts · 17–48 h · 250 Hz · WFDB · 3 sets of ST episode annotations"
patients: "80"
records: "86"
access: "open"
license: "ODC-By 1.0"
origin_institution: "University of Ljubljana; CNR Institute of Clinical Physiology, Pisa; MIT"
origin_country: "Multi-national (Slovenia, Italy, USA)"
leads: 2
paper_title: "Jager et al., Med Biol Eng Comput 2003"
paper_doi: "https://doi.org/10.13026/C2G01T"
search_keywords: "long-term st ltstdb ischemia ischaemia st episode rate-related axis shift ambulatory holter 24h ljubljana pisa cambridge zymed semia karhunen-loeve two-lead three-lead"

related:
  - slug: "european-st-t-database-edb"
    relation: "contains"
    shares_records: true
    verified: true
    note: >
      Ten of these 86 recordings are the same analog Holter tapes that produced ten
      of the European ST-T Database's 90 records, and each one's header says so in
      words: "An excerpt of this recording is included in the European ST-T Database
      (record e0113)." The pairs are s20021→e0113, s20151→e0103, s20161→e0105,
      s20171→e0127, s20181→e0162, s20291→e0104, s20301→e0125, s20311→e0129,
      s20581→e0603 and s20591→e0604 — 11.6% of this release and 11.1% of that one.
      **Do not train on one and evaluate ST detection on the other**: EDB's two hours
      sit inside this database's 21–24, so those are the same hours of the same
      hearts on both sides of the split. The overlap is invisible from the sample
      values, because the tapes were **redigitised** for this release and rescaled —
      PhysioNet states the annotations cannot be compared directly either — so a
      byte-level or correlation check finds nothing. It was verified instead from the
      beat annotations, which survive redigitisation: high-passed heart-rate series
      at 1 Hz, twelve independent 10-minute chunks of each EDB excerpt matched
      against the whole LTSTDB record, accepting only chunk offsets that fall on one
      line. Eight of the ten pairs put 7–12 of 12 chunks on a line (s20151, s20161,
      s20171, s20181, s20291, s20301, s20581, s20591), against a ceiling of 2 of 12
      for every one of the 85 non-partner records — the RANSAC floor, i.e. no
      evidence at all. The two remaining pairs, s20021→e0113 and s20311→e0129, could
      not be confirmed by this instrument and rest on the header text alone. The
      recovered offsets also say *where* each excerpt sits: e0603 begins 131.7 min
      into s20581, e0162 111.0 min into s20181.

sections:
  - type: description
    title: "Overview"
    body: |
      **Ischaemia detection over a whole day rather than a two-hour excerpt, and the
      largest annotated recording in this catalogue by both hours and beats.** 86
      continuous ambulatory records from 80 subjects, **1,992 hours** of ECG at
      250 Hz, **8,897,780** manually corrected reference beats, and **2,311** ST
      episodes under the release's most inclusive criterion. Built between 1995 and
      2002 by a project coordinated at the University of Ljubljana with the CNR
      Institute of Clinical Physiology in Pisa and MIT — the groups behind the
      European ST-T Database and the MIT-BIH Arrhythmia Database respectively — and
      supported by Medtronic and Zymed.

      **It exists to tell ischaemic ST change apart from everything that looks like
      it.** Records were chosen to show a mixture of ischaemic episodes,
      axis-related non-ischaemic episodes, slow ST level drift and combinations of
      those, and the annotations mark which is which: **1,795 ischaemic** episodes
      against **516 rate-related** ones, plus **1,493 axis shifts** and **895
      conduction-change shifts**, which are artefacts of body position and
      conduction rather than findings about the heart. A detector scored without
      those distractors is not being scored on the task this database was built for.

      **There are three sets of ST episode annotations, not one, and they disagree by
      a factor of two.** `.sta`, `.stb` and `.stc` apply different amplitude and
      duration criteria to the same recordings, giving 2,311, 1,364 and 973 episodes.
      None is more correct than the others. **Any figure from this database is
      meaningless without its criterion** — see the table below.

      **Records hold two *or three* signals, in twelve layouts, and 22 of them do not
      name their leads at all.** 68 records store two signals and 18 store three;
      the single largest layout is the 22 records whose headers describe both signals
      only as `ECG` and state "Electrode locations were not recorded." No lead is
      present in all 86 records — the widest, MLIII, reaches 29. This is the only
      dataset in the catalogue that **cannot be batched whole** by any `leads=`
      selection.

      **Subject identity is published, inside the record name.** `sXYYYZ`: `X` is the
      signal count, `YYY` the subject and `Z` that subject's record number, so
      s20271–s20274 are one person. That is the grouping ECGBench folds on, and it
      matters: those four records hold 416 of the release's 1,795 ischaemic episodes
      between them.

  - type: table
    title: "The three ST episode annotators — always quote which one a figure uses"
    headers: ["File", "Criterion", "Ischaemic", "Rate-related", "Total", "Records with ≥1 ischaemic"]
    rows:
      - ["`.sta`", "75 µV, 30 s", "**1,795**", "516", "**2,311**", "68 of 86"]
      - ["`.stb`", "100 µV, 30 s", "1,130", "234", "1,364", "66 of 86"]
      - ["`.stc`", "100 µV, 60 s", "857", "116", "973", "64 of 86"]
      - ["*all three*", "axis shifts", "1,493", "", "", "59 of 86"]
      - ["*all three*", "conduction-change shifts", "895", "", "", "4 of 86"]
      - ["*all three*", "noise events", "31", "", "", "12 of 86"]
      - ["*all three*", "unreadable intervals", "60", "", "", "7 of 86"]

  - type: description
    title: "Why the three criteria exist, and what the shared rows mean"
    body: |
      An episode is detected on the **ST deviation function** — the measured ST level
      minus a piecewise-linear baseline the annotators placed themselves, so that
      postural drift is subtracted before anything is called an episode. It begins
      when the deviation first exceeds 50 µV, must then reach `Vmin` for at least
      `Tmin`, and ends when the deviation falls below 50 µV without exceeding it
      again within 30 s. The three files differ only in `Vmin` and `Tmin`, and the
      release ships all three because the appropriate threshold depends on the
      application. ECGBench exposes all three: the criterion-A columns are unsuffixed
      and the others carry `_b` and `_c`.

      The last four rows carry no suffix because those quantities are **identical in
      all three files**. They are marks, not threshold crossings: an axis shift is a
      shift, whatever amplitude convention is applied to episodes around it.

      **Ischaemic and rate-related episodes must not be summed into "ST episodes"
      without saying so.** Record s20011 holds 20 criterion-A episodes and every one
      is rate-related; its header explains why, and adds "It is recognized that this
      is an arbitrary decision." 16 records hold rate-related episodes and no
      ischaemic ones, 55 the reverse, and 13 both.

  - type: table
    title: "ST episodes by direction and by signal, criterion A (75 µV / 30 s)"
    headers: ["Quantity", "Count", "Note"]
    rows:
      - ["ischaemic episodes", "**1,795**", "68 records"]
      - ["rate-related episodes", "**516**", "29 records"]
      - ["— ST depression", "1,896", "of the 2,311 combined"]
      - ["— ST elevation", "415", "of the 2,311 combined"]
      - ["ischaemic, signal 0", "856", ""]
      - ["ischaemic, signal 1", "786", ""]
      - ["ischaemic, signal 2", "153", "the 18 three-signal records only"]
      - ["episodes already running at sample 0", "10", "no onset annotation; measured from 0"]
      - ["episodes still running at the last sample", "18", "no end annotation; closed at the record end"]
      - ["largest annotated deviation", "1,495 µV", "s20621 — a *rate-related* episode"]
      - ["time in ischaemic episode, union over leads", "151.8 h", "7.6% of the 1,992 recorded hours"]
      - ["time in ischaemic episode, summed over leads", "244.1 h", "leads annotated independently"]

  - type: description
    title: "About those counts"
    body: |
      Every figure on this page was recomputed from the 86 headers and their `.atr`,
      `.sta`, `.stb` and `.stc` files, after verifying the shipped data against the
      release's own `SHA256SUMS.txt`. **All 774 signal and annotation files match** —
      but only after a repair: the local copy's `s30801.dat` was a truncated download
      (13.9 MB against 64.4 MB) and its `.hea`, `.sta`, `.stb`, `.stc` and `.stf`
      were absent entirely, which would have failed one record of 86 for a reason
      the traceback would have blamed on the header.

      **The episode counts are the release's own, and reproducing them exactly is
      what validates the counting convention.** Each record ships a `.cnt` text file
      summarising its episodes, and ECGBench's parse of the binary annotations agrees
      with it in **all 258 blocks** — 86 records × 3 criteria × 6 quantities. Getting
      there requires counting an episode at its **extremum** rather than its onset:
      an episode is an onset/extremum/end triple, but 10 of them were already in
      progress when the tape started and have no onset at all, so counting onsets
      gives 1,785 ischaemic episodes instead of 1,795 and disagrees with the shipped
      summaries for 14 of the 86 records.

      **The published record length is wrong for 16 of the 86.** PhysioNet says the
      recordings are "between 21 and 24 hours in duration". Measured from the
      headers, 70 are; eight are shorter, down to **16.9 h** (s30771), and eight are
      longer, up to **47.8 h** — s20611 holds 43,050,000 samples, nearly twice the
      median, and its `.dat` file size confirms the header. So `expected_samples` is
      deliberately empty and `duration_seconds` in the config is the **median**
      (23.4 h), not a uniform length. A `window=` must fit s30771.

      | Quantity | Landing page | Recomputed (v1.0.0) | Diff |
      |---|---|---|---|
      | Records | 86 | 86 | — |
      | Subjects | 80 | 80 | — |
      | Three-signal records | 18 | 18 | — |
      | From the European ST-T collection | 10 | 10 *(named in the headers)* | — |
      | From the 1995–98 pilot collection | 11 | 11 *(named in the headers)* | — |
      | Record duration | 21–24 h | **16.9–47.8 h**, median 23.4 h | 16 records outside |

      **There is no changelog in the release directory**, so the duration
      disagreement cannot be attributed to a reissue; the shipped headers and `.dat`
      sizes agree with each other, and the landing page does not agree with them.
      Treat the recomputed figures as authoritative for v1.0.0.

      **`ischemic_secs` can exceed a record's own length, and that is correct.** The
      signals are annotated independently, so ischaemia seen in two leads at once is
      two episodes and its seconds count twice: 244.1 h summed against 151.8 h in the
      bounded union. Use `ischemic_secs_any_lead`, or `ischemic_fraction`, for
      anything expressed as a proportion.

  - type: table
    title: "The twelve lead layouts, counted from the 86 headers"
    headers: ["Layout (file order)", "Signals", "Records", "Layout (file order)", "Signals", "Records"]
    rows:
      - ["ECG / ECG", "2", "**22**", "MLIII / V3", "2", "1"]
      - ["V4 / MLIII", "2", "20", "V5 / MLIII", "2", "1"]
      - ["ML2 / MV2", "2", "16", "V5 / V2", "2", "1"]
      - ["E-S / A-S / A-I", "3", "15", "V2 / MLIII", "2", "1"]
      - ["MLIII / V4", "2", "6", "V4 / V3 / II", "3", "1"]
      - ["", "", "", "V6 / V5 / aVF", "3", "1"]
      - ["", "", "", "V6 / II / V5", "3", "1"]

  - type: table
    title: "…and how many records contain each lead. None reaches 86."
    headers: ["Lead", "Records", "Share of the release"]
    rows:
      - ["MLIII", "29", "34%"]
      - ["V4", "27", "31%"]
      - ["ECG *(unnamed)*", "22", "26%"]
      - ["ML2", "16", "19%"]
      - ["MV2", "16", "19%"]
      - ["E-S / A-S / A-I", "15 each", "17%"]
      - ["V5", "4", "5%"]
      - ["V2, V3, V6, II", "2 each", "2%"]
      - ["aVF", "1", "1%"]

  - type: description
    title: "What the twelve layouts mean for loading — read this before batching"
    body: |
      `config.record_lead_layouts` declares all twelve, which makes
      `ECGDataset(leads=[...])` open each record's own header and resolve the
      requested **names** against it rather than trusting a position. Four configs in
      this catalogue need that; this is the only one where the **lead count** varies
      too, which is why `alternate_lead_names` — keyed by count — cannot express it.

      Three consequences, in decreasing order of how likely they are to bite:

      **1. This dataset cannot be batched whole.** `ecg_collate_fn` stacks with
      torch's `default_collate`, which raises on a batch mixing 2- and 3-channel
      records, and no `leads=` value fixes that because no lead is in all 86 records.
      `leads=["MLIII"]` loads 29 and raises for the other 57. Filter first — `n_leads`
      and `lead_names` are columns of `ecgbench.labels.ltstdb` — or use
      `batch_size=1`. The example script does the filtering.

      **2. `V4/MLIII` and `MLIII/V4` are both present**, 20 records and 6. Those 26
      records are indistinguishable by shape, by lead count and by anything in the
      exported fold CSVs, and `signal[0]` is a chest lead in one group and a limb
      lead in the other.

      **3. For the 22 `ECG/ECG` records, `leads=["ECG"]` returns signal 0 and there
      is no name that reaches signal 1.** That is a property of the release rather
      than of ECGBench: the electrode positions were never recorded, so the two
      signals are genuinely indistinguishable by name. Use `leads=None` and index
      them if you need both. `leads_named` flags those records in the labels.

  - type: table
    title: "Reference beat annotations, recomputed from the 86 .atr files"
    headers: ["Beat type", "Symbol", "Count", "Share", "Records containing it"]
    rows:
      - ["normal beat", "N", "8,669,297", "97.432%", "86"]
      - ["bundle branch block beat (unspecified)", "B", "88,720", "0.997%", "2"]
      - ["premature ventricular contraction", "V", "72,852", "0.819%", "76"]
      - ["supraventricular premature or ectopic beat", "S", "57,311", "0.644%", "75"]
      - ["atrial premature beat", "A", "8,730", "0.098%", "7"]
      - ["fusion of ventricular and normal beat", "F", "597", "0.007%", "36"]
      - ["aberrated atrial premature beat", "a", "162", "0.002%", "14"]
      - ["ventricular escape beat", "E", "71", "0.001%", "2"]
      - ["atrial escape beat", "e", "30", "0.000%", "4"]
      - ["nodal (junctional) escape beat", "j", "6", "0.000%", "2"]
      - ["unclassifiable beat", "Q", "2", "0.000%", "2"]
      - ["nodal (junctional) premature beat", "J", "1", "0.000%", "1"]
      - ["paced beat", "/", "1", "0.000%", "1"]
      - ["**total**", "", "**8,897,780**", "**100%**", "**86**"]

  - type: table
    title: "The same beats under the AAMI EC57 reduction — use these to combine with mitdb, svdb, incartdb and edb"
    headers: ["AAMI class", "Built from", "Count", "Share", "Records containing it"]
    rows:
      - ["N — normal / bundle branch block", "N, L, R, e, j, B", "8,758,053", "98.430%", "86"]
      - ["S — supraventricular ectopic", "A, a, J, S, n", "66,204", "0.744%", "82"]
      - ["V — ventricular ectopic", "V, E, r", "72,923", "0.820%", "76"]
      - ["F — fusion", "F", "597", "0.007%", "36"]
      - ["Q — unclassifiable / paced", "/, f, Q", "3", "0.000%", "3"]

  - type: description
    title: "The .atr files hold beats and nothing else"
    body: |
      Unusually for a MIT-BIH-family release, there is **no rhythm (`+`) annotation,
      no signal-quality (`~`) annotation and no `aux_note` anywhere** in the 86
      `.atr` files — 8,897,780 annotations, every one a beat label. So this database
      offers no rhythm ground truth at all, and quality is annotated somewhere else:
      the 31 noise events and 60 unreadable intervals live in the ST files as `noi`
      and `(urd`/`urd)`, covering 18.5 h in 7 records.

      Beat coverage is otherwise complete. The first beat falls 0.02–9.18 s into the
      record and the last 0.13–1.96 s before its end, so **every record is annotated
      over at least 99.98% of its length** — there is no unannotated tail to window
      around, unlike `nsrdb`, which leaves 12.1% of its signal in silence.

      The `.ari` files are the same detector's **uncorrected** output and are
      deliberately not read; `.atr` is the manually corrected version. The 26 `.hea-`
      files are superseded pre-2008 headers. The `.16a`, `.stf`, `.klt.zip`,
      `.tsr.zip`, `legendre/` and `kl-single*/` products — ST measurements at eight
      points of every beat, ST level and deviation functions, and Karhunen-Loève and
      Legendre coefficient time series — are shipped and rich, and ECGBench does not
      parse them. Read them from the release directly.

  - type: table
    title: "Ischaemic-burden bands — the stratification label"
    headers: ["Band", "Ischaemic episodes (criterion A)", "Records", "Episodes in the band"]
    rows:
      - ["none", "0", "18", "0"]
      - ["1-5", "1 – 5", "14", "45"]
      - ["6-20", "6 – 20", "25", "259"]
      - ["21+", "21 or more", "29", "1,491"]

  - type: description
    title: "Why the folds are grouped by subject and stratified on ischaemic burden"
    body: |
      **The grouping is published, unlike `edb`'s.** `sXYYYZ` puts the subject number
      in the record name, the landing page states the rule, and four headers restate
      it in words ("Records s20271, s20272, s20273 and s20274 are from the same
      patient."). Six records belong to subjects with more than one: 027 has four,
      and 073, 074 and 075 have two each. Ungrouped, subject 027 alone would scatter
      four recordings holding **23% of the release's ischaemia** across several
      folds, and any of them landing in both train and test is the same day of the
      same heart on both sides of the split. `patient_id` is the zero-padded
      three-digit field of the record name — `"027"`, not `27` — which is why this
      config sets `zero_padded_identifiers`.

      **Stratification is the criterion-A ischaemic episode count**, banded at fixed
      edges of 1, 6 and 21 episodes. It is what this database exists to measure and
      it is enormously uneven: 18 records hold no ischaemic episode at all while
      s20274 holds 143, and the top band holds 83% of all the episodes. A fold drawn
      without regard to it can easily be all quiet records or all busy ones. Fixed
      edges rather than quantiles, so a re-release with one extra episode cannot
      silently relabel records that did not change.

      Nothing clinical works better. `st_class` puts 55 of the 86 records in one
      class and 2 in another; the header findings describe the subject rather than
      the recording; `diagnoses` is free text with more distinct values than there
      are folds to spread them over.

      Unlike `edb`, **every band clears the ten folds** (14 is the smallest), so none
      is forced to skip any. With 86 records over 10 folds, `train` holds 70, `val` 8
      and `test` 8, and every fold carries records from all four bands.
      `stratify_class` is a **fold-construction label**. Train on the episode counts,
      `ischemic_fraction`, `peak_st_deviation_uv`, or — for episode detection, which
      is the task — on the `.sta`/`.stb`/`.stc` files directly.

  - type: table
    title: "Cohort, recomputed from the 86 headers"
    headers: ["Quantity", "Value"]
    rows:
      - ["Subjects / records", "80 / 86"]
      - ["Subjects contributing >1 record", "4 (027 ×4; 073, 074, 075 ×2)"]
      - ["Sex (subjects)", "46 M, 29 F, 5 not recorded"]
      - ["Age", "23–87, mean 61.5, median 62; 5 records not recorded"]
      - ["Recorded hours", "1,991.8 h (median record 23.4 h)"]
      - ["Mean heart rate", "51.4 – 109.2 bpm per record"]
      - ["Holter recorder", "Remco 18, Zymed 18, ICR 10, Oxford Medilog 2, Del Mar Avionics 2, **not recorded 36**"]
      - ["Recording dates", "1984 – 2000"]
      - ["Hypertension / previous MI", "25 / 24 records; 11–12 records say 'No data'"]

  - type: description
    title: "The clinical record in the headers is unusually complete — and unusually free-text"
    body: |
      Each `.hea` carries a 28-field indented tree: age, sex, annotator commentary on
      the recording itself, symptoms during monitoring, a diagnosis list, current
      medications, angioplasty and bypass history, eleven structured cardiac-history
      fields, and the findings of previous stress tests, thallium or stress
      echocardiography, LV function studies, echocardiography, coronary arteriography
      and baseline ECG. All 28 fields appear in all 86 headers.

      **The structured fields look boolean and are not.** "Previous Myocardial
      Infarction" answers include `No`, `No data`, `Yes`, `Yes, anterior`, `Yes, in
      1996 and 1997` and `Yes, had myocardial infarction in 1995`; "Left ventricular
      hypertrophy" includes `Septum 13 mm`; "Intraventricular conduction block"
      includes `Right bundle branch block` and `Borderline`. ECGBench exposes each as
      a **nullable boolean plus the verbatim text** in a `<field>_text` column — read
      the text whenever the detail matters. `NA` means the header said "No data",
      which is 11 to 45 records depending on the field.

      Two provenance fields are parsed out of the annotator commentary.
      **`edb_record`** names the European ST-T record cut from the same tape, for the
      ten records that have one — see the overlap warning at the top of this page.
      **`pilot_record`** is the name eleven records carried in the 1995–98 pilot
      collection, which was never published; **do not read it as a record id here**,
      because one of the eleven collides — s20071's pilot name is `s20511`, and
      `s20511` is also a record in this release, belonging to a different subject.

  - type: code
    title: "Building the splits"
    language: bash
    body: |
      ecgbench splits --dataset ltstdb --data-path /path/to/ltstdb/1.0.0/

  - type: code
    title: "Loading with ECGBench"
    language: "python"
    body: |
      from ecgbench import ECGDataset

      # Records run 15,200,000 to 43,050,000 samples per channel — 17 to 48 hours —
      # so window= is not optional here. It is pushed into the reader, so it avoids
      # decoding the rest of the day rather than cropping afterwards, and it must fit
      # the SHORTEST record (s30771, 15,200,000 samples).
      ds = ECGDataset(
          "ltstdb",
          split="train",
          data_path="/path/to/ltstdb/1.0.0/",
          window=(0, 2500),          # first 10 s at 250 Hz
          labels=True,
      )

      len(ds)                                      # 70
      ds[0]["signal"].shape                        # torch.Size([2, 2500])
      ds[0]["record_id"]                           # 's20011'

      ds.lead_names                                # ('ECG', 'ECG')  <- the MODAL layout,
                                                   #    which names nothing, and is NOT
                                                   #    this record's
      ds[0]["labels"]["lead_names"]                # 'ML2|MV2'   <- what s20011 stores
      ds[0]["labels"]["n_leads"]                   # 2   (18 of the 86 records store 3)

      # Criterion A (.sta, 75 uV / 30 s) is unsuffixed; _b and _c are the stricter two.
      ds[0]["labels"]["n_ischemic_episodes"]       # 0
      ds[0]["labels"]["n_rate_related_episodes"]   # 20  <- NOT ischaemia, and the header
                                                   #        says the call was arbitrary
      ds[0]["labels"]["n_rate_related_episodes_b"] # 4   <- same record, 100 uV / 30 s
      ds[0]["labels"]["n_axis_shifts"]             # 7   <- positional artefact
      ds[0]["labels"]["peak_st_deviation_uv"]      # 243  <- vs the annotator-placed
                                                   #         baseline, not isoelectric
      ds[0]["labels"]["ischemic_fraction"]         # 0.0
      ds[0]["labels"]["st_class"]                  # 'rate_related_only'
      ds[0]["labels"]["ischemic_burden_band"]      # 'none'  <- for folds, not training
      ds[0]["labels"]["patient_id"]                # '001'   <- from the record name
      ds[0]["labels"]["duration_hours"]            # 22.8831
      ds[0]["labels"]["n_beats"]                   # 100053
      ds[0]["labels"]["age"], ds[0]["labels"]["sex"]       # 58.0, 'M'
      ds[0]["labels"]["diagnoses"]                 # 'No coronary artery disease'

      # Batching needs a RECORD FILTER, not just leads=. No lead is in all 86 records
      # and 18 store three signals, so a mixed batch raises in default_collate.
      holds_mliii = ds.labels_df["lead_names"].str.split("|").apply(lambda n: "MLIII" in n)
      int(holds_mliii.sum())                       # 23 of the 70 records in this split

      # Selecting by NAME re-resolves against each record's own header, and raises for
      # the records that lack the lead rather than returning whichever one sits there.
      mliii = ECGDataset("ltstdb", split="train", data_path="/path/to/ltstdb/1.0.0/",
                         window=(0, 2500), leads=["MLIII"])
      mliii[0]    # ValueError: Record 's20011' stores ['ML2', 'MV2'], and this dataset
                  # uses more than one lead layout. Lead 'MLIII' is not in 'ltstdb'.

      # See examples/load_ltstdb.py for the full walkthrough.

  - type: links
    title: "References"
    items:
      - label: "PhysioNet: Long Term ST Database v1.0.0"
        url: "https://physionet.org/content/ltstdb/1.0.0/"
      - label: "Jager et al. (2003), Med Biol Eng Comput 41(2):172-183 — the database paper"
        url: "https://doi.org/10.1007/BF02344885"
      - label: "PhysioNet DOI (v1.0.0)"
        url: "https://doi.org/10.13026/C2G01T"
      - label: "ST annotation codes (tables/acodes.png in the release)"
        url: "https://physionet.org/files/ltstdb/1.0.0/tables/acodes.png"
      - label: "European ST-T Database — ten of these tapes also produced ten of its records"
        url: "https://physionet.org/content/edb/1.0.0/"
---
