# ECGBench

[![PyPI](https://img.shields.io/pypi/v/ecgbench?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/ecgbench/)
[![Python](https://img.shields.io/pypi/pyversions/ecgbench?logo=python&logoColor=white)](https://pypi.org/project/ecgbench/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HF Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Fold%20splits-yellow)](https://huggingface.co/datasets/vlbthambawita/ECGBench)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Space-blue)](https://huggingface.co/spaces/vlbthambawita/ECGBench)
[![Website](https://img.shields.io/badge/Website-ECGBench-6366f1?logo=githubpages&logoColor=white)](https://vlbthambawita.github.io/ECGBench/)

Reproducible ECG benchmark datasets with standardised splits, validation, and Croissant metadata.

ECGBench provides a curated catalogue of 64 publicly available ECG datasets, a config-driven pipeline for generating validated fold splits, and a unified PyTorch `Dataset` class for loading any supported dataset.

| | |
|---|---|
| **Website** | [vlbthambawita.github.io/ECGBench](https://vlbthambawita.github.io/ECGBench/) |
| **HuggingFace Space** | [huggingface.co/spaces/vlbthambawita/ECGBench](https://huggingface.co/spaces/vlbthambawita/ECGBench) |
| **Fold splits (Hub)** | [huggingface.co/datasets/vlbthambawita/ECGBench](https://huggingface.co/datasets/vlbthambawita/ECGBench) |
| **PyPI** | [pypi.org/project/ecgbench](https://pypi.org/project/ecgbench/) |

## Installation

### Base (config, catalogue, validation, splitting)

```bash
pip install ecgbench
```

### With PyTorch support

```bash
pip install ecgbench[torch]
```

### With HDF5 datasets

`sph`, `code15` and `code_test` store their waveforms as HDF5, which needs
`h5py`. They are the only datasets that do, so the dependency is its own extra:

```bash
pip install ecgbench[hdf5]
```

### With everything

```bash
pip install ecgbench[all]
```

### From source (development)

```bash
git clone https://github.com/vlbthambawita/ECGBench.git
cd ECGBench
uv pip install -e ".[dev]"
```

## Quick Start

```python
from ecgbench import ECGDataset, ecg_collate_fn
from torch.utils.data import DataLoader

# Load PTB-XL training data (downloads fold CSVs from HuggingFace Hub)
train_ds = ECGDataset("ptbxl", split="train", data_path="/path/to/ptb-xl/1.0.3/")
loader = DataLoader(train_ds, batch_size=32, collate_fn=ecg_collate_fn)

for batch in loader:
    signals = batch["signal"]   # (B, 12, 5000) float32 tensor
    ecg_ids = batch["record_id"]
    break
```

## Dataset Catalogue

Query the curated index of 64 ECG datasets:

```python
import ecgbench

# List all datasets
datasets = ecgbench.list_datasets()
print(f"{len(datasets)} datasets available")

# Search by name, origin, format, or paper
ecgbench.search("PTB-XL")

# Filter by category and access type
ecgbench.search(category="12-Lead (PhysioNet)", access="Open")

# Look up a single dataset
ecgbench.get_dataset("MIMIC-IV-ECG")

# List categories
ecgbench.categories()

# Get as pandas DataFrame
df = ecgbench.to_dataframe()
```

## Loading ECG Data

### Standard train/val/test splits

```python
from ecgbench import ECGDataset, ecg_collate_fn
from torch.utils.data import DataLoader

train_ds = ECGDataset("ptbxl", split="train", data_path="/data/ptb-xl/1.0.3/")
val_ds = ECGDataset("ptbxl", split="val", data_path="/data/ptb-xl/1.0.3/")
test_ds = ECGDataset("ptbxl", split="test", data_path="/data/ptb-xl/1.0.3/")

loader = DataLoader(train_ds, batch_size=32, collate_fn=ecg_collate_fn)
```

### Selecting specific folds

`fold_numbers` picks individual folds out of a split. Folds are 1-indexed.

```python
ECGDataset("ptbxl", split="train", fold_numbers=[3], data_path="...")       # one fold
ECGDataset("ptbxl", split="train", fold_numbers=[1, 2, 5], data_path="...") # several
```

**Each fold belongs to exactly one split** — 1-8 under `train/`, 9 under `val/`,
10 under `test/` — so `split="train", fold_numbers=[9]` is an error. To select
folds regardless of that layout, for custom cross-validation, pass `split=None`:

```python
# Hold out fold 7 as test and fold 10 as val, train on the other eight.
test  = ECGDataset("ptbxl", split=None, fold_numbers=[7],  data_path="...")
val   = ECGDataset("ptbxl", split=None, fold_numbers=[10], data_path="...")
train = ECGDataset("ptbxl", split=None,
                  fold_numbers=[n for n in range(1, 11) if n not in (7, 10)],
                  data_path="...")
```

`split=None` requires `fold_numbers`, and each returned sample's `["split"]`
reports the record's own default split rather than one name for the whole set.
Unlike stitching per-split datasets together with `ConcatDataset`, this returns a
single `ECGDataset`, so `.metadata_df` and `.labels_df` still describe the whole
selection.

### Labels

Fold CSVs are **identification-only** by design — record ID, patient ID, signal
paths, fold, split. Ground truth stays with the source dataset, so `labels=True`
needs a local copy of it:

```python
ds = ECGDataset("ptbxl", split="train", data_path="/data/ptb-xl/1.0.3/", labels=True)

ds[0]["labels"]["superclasses"]   # ['MI', 'STTC']  — multi-label
ds[0]["labels"]["report"]         # the cardiologist's text
ds.labels_df                      # the whole split's labels, aligned to metadata_df
```

Or without a Dataset at all, for class weights and filtering:

```python
from ecgbench import load_labels

labels = load_labels("chapman_shaoxing", data_path="/data/chapman-figshare/")
labels["Rhythm"].value_counts()
```

Each dataset exposes its own fields — SCP codes plus diagnostic super/subclasses
for PTB-XL, SNOMED-CT codes for `ecg_arrhythmia`, rhythm/beat annotations and
eleven automated measurements for `chapman_shaoxing`, free-text machine reports
plus nine interval/axis measurements for `mimic_iv_ecg`, reference beat counts for
`incartdb`, protocol phase and balloon-occlusion timings for `staffiii`,
AHA/ACC/HRS statements with their modifiers for `sph`, the ablation-confirmed
arrhythmia origin for `ningbo_iva`, per-symbol beat counts plus their AAMI EC57
reduction for `svdb`, and an **ST/T episode inventory** for `edb`. A dataset
that genuinely has none (`mimic_iv_ecg_demo`) raises
`LabelsUnavailableError` naming where labels could come from, rather than
returning empty columns.

**One dataset's ground truth does not fit a record-level table at all.** The QT
Database is a *delineation* reference: cardiologists marked the onset, peak and end of
the P, QRS, T and U waves of 3,623 individually selected beats, up to eleven fiducial
points each. `labels=True` returns a per-record summary — how many beats, which waves,
the QT/QTc/RR/PR/QRS medians — and the boundaries themselves come from a second call:

```python
from ecgbench.labels.qtdb import load_beat_annotations

beats = load_beat_annotations("/data/qtdb/1.0.0/")          # 3,623 rows, one per beat
beats[["record_name", "qrs_onset", "t_offset", "qt_ms"]].head()
second = load_beat_annotations("/data/qtdb/1.0.0/", annotator="q2c")   # 404 rows
```

Three things to know before using them. Sample indices are absolute in the record's
own 250 Hz frame, and **all of them lie in the last five minutes** — the earliest mark
in the release is at 600.464 s and the latest at 896.916 s, deliberately, to leave an
algorithm ten minutes of learning data — so read `window=(150000, 74993)` and subtract
the window start before indexing the tensor. `NaN` means the annotator did not mark
that point, which is information rather than missing data: two records mark QRS
boundaries and no T wave at all. And **every `qtdb` record is a fifteen-minute excerpt
of another database's recording** — 100 of the 105 share signal samples with `edb`,
`sddb`, `mitdb`, `svdb`, `nsrdb` or `stdb`, verified from the waveforms — so
`source_database` and `source_catalogue_slug` name the leakage partner of each record
and its `label_column` is provenance rather than pathology. See
`examples/load_qtdb.py`.

**One dataset's ground truth is episodes rather than labels.** The European ST-T
Database annotates the onset, extremum and end of every interval of significant ST
or T change — 368 and 401 of them — separately in each of its two signals, so
`edb`'s loader returns counts per signal and direction, the peak deviation in
microvolts, and time-in-episode both summed over signals and as a bounded union
(`st_secs_any_signal`, `ischaemic_fraction`). Two things to know before using them:
the deviations are measured against **each subject's own reference waveform** from
their record's first 30 s, not an absolute isoelectric line, so a fixed ST threshold
cannot reproduce them; and `st_episode_secs` can legitimately exceed the 7,200 s
recording, because concurrent change in both channels counts twice. See
`examples/load_edb.py`.

**And one annotates them three times over.** The Long-Term ST Database (`ltstdb`)
applies **three different detection criteria** to the same 86 day-long recordings and
ships all three: `.sta` at 75 µV / 30 s finds 1,795 ischaemic and 516 rate-related
episodes, `.stb` at 100 µV / 30 s finds 1,130 and 234, and `.stc` at 100 µV / 60 s
finds 857 and 116. None is more correct than the others, so **no episode count from
this database means anything without its criterion** — the loader makes `.sta` the
unsuffixed default and exposes the rest under `_b` and `_c`. Two more things separate
it from `edb`: ischaemic and rate-related episodes are annotated **apart from each
other**, along with 1,493 axis shifts and 895 conduction-change shifts that mimic
ischaemia, and episodes are counted at their **extremum** rather than their onset,
because 10 of them were already running when the tape started. Counting extrema
reproduces the release's own shipped `.cnt` summaries in all 258 blocks; counting
onsets does not. See `examples/load_ltstdb.py`.

**And one dataset named for ST change annotates none of it.** The MIT-BIH **ST
Change** Database (`stdb`) is 28 recordings *selected* for transient ST change —
mostly exercise stress tests — but its annotation files hold beat labels and
nothing else: 76,175 of the 76,181 annotations across the 28 `.atr` files are
beats, the other six are signal-quality markers, and there is not a single `+`
rhythm marker, `s` ST marker or non-empty `aux_note` in the release. PhysioNet says
so on the landing page and the files agree, so there is no ST measurement, episode
boundary or deviation to load. `st_change_type` (`depression` for 23 records,
`elevation` for 323-327) is the landing page's own grouping **transcribed**, which
is why a `group_source` column carries the constant `landing_page` — and why the
loader also exposes `hr_rise_bpm`, the measured quantity that checks it. Use `edb`
or `ltstdb` when you need annotated ST episodes. See `examples/load_stdb.py`.

**Beat symbols are not comparable across the MIT-BIH databases, so `svdb` exposes
the AAMI reduction alongside them.** A supraventricular beat is annotated `S` in
`svdb` (12,188 of them) and `A` in `mitdb` (2,546, with `S` used twice), so
concatenating the two on the raw symbol trains a model on two disjoint
vocabularies for one phenomenon. The `aami_N/S/V/F/Q` columns collapse `A`, `a`,
`J` and `S` to class `S` — and `L`, `R`, `B` to `N` — and are what to join on.
`AAMI_CLASSES` covers every symbol used by the MIT-BIH-family databases here, so it
also reduces `mitdb`, whose loader exposes raw per-symbol counts only, and is what
`edb` and `chfdb` import rather than keeping a second copy.

`chfdb` is the sharper case of the same trap, in the other direction: 10,353 of its
beats are `r` — R-on-T premature ventricular contractions, which AAMI classes as
**ventricular** — and `r` **outnumbers plain `V` in 9 of its 15 records**. Counting
`beat_V` there undercounts ventricular ectopy across most of the database, so its
loader exposes `n_veb`/`veb_fraction` from `aami_V` rather than from the raw symbol.
Note also that `chfdb`'s annotations are the one set in this family that is
**unaudited** — an automated detector's uncorrected output, per PhysioNet — so its
counts describe the recording rather than establishing ground truth:

```python
from ecgbench.labels.svdb import AAMI_CLASSES

sv = load_labels("svdb",  data_path="/data/svdb/1.0.0/")
mi = load_labels("mitdb", data_path="/data/mitdb/1.0.0/")

sv["aami_S"].sum() / sv["n_beats"].sum()   # 0.0661  <- the reason svdb exists

# mitdb keeps the raw symbols, so reduce them with the same table:
sveb = [f"beat_{s}" for s, c in AAMI_CLASSES.items() if c == "S" and f"beat_{s}" in mi]
mi[sveb].to_numpy().sum() / mi["n_beats"].sum()    # 0.0254
```

### Leads and units

Select and reorder leads **by name**, and choose the output unit:

```python
ds = ECGDataset("mimic_iv_ecg_demo", split="train", data_path="...",
                leads=["I", "II", "aVL", "V5"], units="uV")

ds[0]["signal"].shape   # (4, 5000)
ds.lead_names           # ('I', 'II', 'aVL', 'V5')
ds.units                # 'uV'
```

Names, not indices, because **lead order is not consistent across datasets**:

| Dataset | Order in the files |
|---|---|
| `ptbxl` | I, II, III, **AVR, AVL, AVF**, V1-V6 (uppercase) |
| `ecg_arrhythmia` | I, II, III, aVR, aVL, aVF, V1-V6 |
| `chapman_shaoxing` | I, II, III, aVR, aVL, aVF, V1-V6 |
| `mimic_iv_ecg` | I, II, III, aVR, **aVF, aVL**, V1-V6 (transposed) |
| `mimic_iv_ecg_demo` | I, II, III, aVR, **aVF, aVL**, V1-V6 (transposed) |
| `ludb` | i, ii, iii, avr, avl, avf, v1-v6 (**lowercase**) |
| `ptbdb` | i, ii, iii, avr, avl, avf, v1-v6, **vx, vy, vz** (15 signals) |
| `challenge2021` | I, II, III, aVR, aVL, aVF, V1-V6 (identical in all eight cohorts) |
| `challenge2020` | I, II, III, aVR, aVL, aVF, V1-V6 (identical in all six cohorts) |
| `incartdb` | I, II, III, **AVR, AVL, AVF**, V1-V6 (uppercase) |
| `brugada_huca` | I, II, III, aVR, aVL, aVF, V1-V6 |
| `leipzig_heart_center_ecg` | I, II, III, aVR, aVL, aVF, V1-V6, **then 2-8 intracardiac channels in six different orders** |
| `norwegian_athlete_ecg` | I, II, III, **AVR, AVL, AVF**, V1-V6 (uppercase) |
| `mhd_effect_ecg_mri` | I, II, III, aVR, aVL, aVF, V1-V6 — but **14 of 53 records hold only I, II, III** |
| `wctecgdb` | **37 channels, no aVR/aVL/aVF**: I, II, III, V1-V6, LA, RA, LL, UV1-UV6 — each **once raw (`-Raw`) and once filtered** — then `WCT` |
| `ecgcipa` | I, II, III, aVR, aVL, aVF, V1-V6 — but the **derived median beat of the same record** spells them AVR/AVL/AVF and adds VCGMAG, X, Y, Z |
| `ecgdmmld` | I, II, III, **AVR, AVL, AVF**, V1-V6 (uppercase) — the **opposite spelling to `ecgcipa`**, its sibling release from the same programme; here the median beats agree with the raw records and add VCGMAG, vx, vy, vz |
| `ecgrdvq` | I, II, III, **AVR, AVL, AVF**, V1-V6 (uppercase) — same as `ecgdmmld` and again the opposite of `ecgcipa`; its median beats agree too, and add VCGMAG, vx, vy, vz |
| `echonext` | I, II, III, aVR, aVL, aVF, V1-V6 — **not stated anywhere in the release**; inferred from the signals, since Einthoven's `III = II − I` and the Goldberger relations hold while wrong pairings do not |
| `staffiii` | **V1-V6 FIRST, then I, II, III** — 9 signals, no aVR/aVL/aVF (derivable from I and II, so the montage is 12-lead clinically but `signal[0]` is V1) |
| `cpsc_2018` | I, II, III, aVR, aVL, aVF, V1-V6 — necessarily the same as `challenge2020`/`challenge2021`, whose `cpsc_2018` cohort is a byte-for-byte copy of these records |
| `sph` | I, II, III, aVR, aVL, aVF, V1-V6 — **not stated in the HDF5 arrays**; derived from the signals, since `III = II − I` and the Goldberger relations hold to under 2% relative RMS error |
| `ningbo_iva` | **aVF, aVL, aVR, I, II, III**, V1-V6 — the columns are sorted **alphabetically**, so `signal[0]` is aVF and lead I is `signal[3]` |
| `code15` | I, II, III, aVR, aVL, aVF, V1-V6 — standard, but **checked** rather than assumed, because its own sibling release below is not |
| `code_test` | I, II, III, **aVL, aVF, aVR**, V1-V6 — the **same cohort as `code15` at the same rate, permuted differently**. `signal[3]` is aVR in one and aVL in the other, so anything stacking the two must select by name |
| `sami_trop` | I, II, III, aVR, aVL, aVF, V1-V6 — standard, and **checked** for the same reason: it is the third release from the same telehealth network and the other two disagree with each other |
| `ikem` | **V1-V6, then II, then I** — 8 signals, no III/aVR/aVL/aVF (exact linear combinations of II and I, and simply not stored). `signal[0]` is V1 and `signal[6]` is **II**, not I. The most unusual order in the catalogue, and the release names none of it — derived from the arrays |
| `zzu_pecg` | I, II, III, **AVR, AVL, AVF**, V1-V6 (uppercase) — **but 1,856 of 14,190 records store only 9 leads**, dropping V2/V4/V6, so `signal[7]` is V2 in one layout and V3 in the other. See below |
| `medalcare_xl` | I, II, III, aVR, aVL, aVF, V1-V6 — standard, and stated by the release README rather than derived, since the records are **simulated** and there are no headers. Corroborated by the per-record parameter files, which place RA/LA/RL/LL and V1-V6 and nothing else — the augmented leads are computed, not placed |

| `mitdb` | **MLII, V1** in 40 of 48 records — and **not** in the other 8. Two modified chest-placed leads, none of the standard twelve. See below |
| `afdb` | **`ECG1`, `ECG2`** — the two channels are **not named leads at all**. The release states no electrode placement anywhere, so these are channel positions, and they must not be read as `mitdb`'s MLII/V1 by analogy with its sibling release |
| `challenge2017` | **`ECG` — one channel, and it is not called `I`.** The AliveCor device gives a nominal lead I (LA-RA) equivalent, but it does not enforce orientation, so the paper reports that **many traces are inverted (RA-LA)** and no record says which. The source's own channel name is the only honest one; naming it `I` would let it be stacked with 12-lead lead I while an unknown fraction carries the opposite sign |
| `ltafdb` | **`ECG1`, `ECG2`** — worse than `afdb`: every header calls **both** channels `ECG`, the same string twice, so there is nothing to tell them apart by. These two names are positions ECGBench assigns so `leads=` works at all. Again not MLII/V1 |
| `nsrdb` | **`ECG1`, `ECG2`** — like `afdb`, and from the same Beth Israel arrhythmia laboratory as `mitdb`: the headers spell the two names and state no electrode placement, so these are channel positions too |
| `svdb` | **`ECG1`, `ECG2`** — like `afdb` and `nsrdb`. Worth singling out because this catalogue's own entry claimed **`MLII` + `V1` at 360 Hz** before the config was written, and both halves were wrong: the recordings are **128 Hz** and all 78 headers name the channels `ECG1`/`ECG2` with no placement stated. The values came from assuming `mitdb`'s properties carry across, which is the exact failure `lead_names` exists to prevent |
| `edb` | **`V5, MLI` in only 19 of 90 records** — the most varied layout in the catalogue: **fifteen orderings of eleven lead pairs**, and **no lead present in every record** (V5 reaches 51, MLIII 47, D3 exactly 1). `MLIII/V4` and `V4/MLIII` are both present, 15 records each. See below |
| `chfdb` | **`ECG1`, `ECG2`** — like `afdb`, `nsrdb` and `svdb`, and from the same Beth Israel hospital as `mitdb`: no electrode placement is stated anywhere, so these are channel positions. Note that only the **current** `.hea` files name them — the 15 superseded `.hea-` copies shipped beside them (and listed in the release's own `SHA256SUMS.txt`) carry no signal descriptions at all, because the 2012 revision is what added them |
| `sddb` | **`ECG1`, `ECG2`** — the `ltafdb` case, not the `afdb` one: every current header calls **both** channels `ECG`, the same string twice, so these two names are positions ECGBench assigns. The 23 superseded `.hea-` copies say `record 30, signal 0` instead — the 2008 revision is what introduced `ECG`. No electrode placement is stated in the headers or on the landing page, so again not MLII/V1. This is also the one dataset whose **ADC gain varies between records**: 800 adu/mV for 21 and **200 for records 39 and 47**, which moves the 12-bit rail from ±2.55875 mV to ±10.235 mV |
| `qtdb` | **`ECG1`, `ECG2` in 57 of 105 records, and 19 further layouts in the other 48** — the most varied in the catalogue, and the only one whose *modal* layout is a placeholder. Those 57 (every excerpt from `svdb`, `nsrdb`, `stdb`, MIT-BIH Long-Term and the sudden-death Holters) state no electrode placement at all. The 15 MIT-BIH Arrhythmia excerpts match `mitdb`'s names exactly; the 33 European ST-T ones use the ESC's **original electrode nomenclature** (`D3`, `CM5`, `CC5`, `ML5`, `CM2`, `mod.V1`, `V2-V3`) and agree with `edb`'s names for the same bit-identical channels in only **2 of 33**. `D3, V4` and `V4, D3` are both present. See below |
| `stdb` | **`ECG1`, `ECG2` — but only 18 of the 28 records have both.** The `ltafdb`/`sddb` case for naming (every header describes every channel as the bare word `ECG`, so these are positions ECGBench assigns), plus a second problem those two do not have: **records 313-317 and 319-323 store a single channel**, which nothing in the release mentions. The config declares `alternate_lead_names: {1: ["ECG1"]}`, so `leads=["ECG2"]` raises for those ten rather than returning ECG1. Not MLII/V1 — and the temptation is strongest here, because this release shares `mitdb`'s 360 Hz rate and three-digit record numbering. Its **ADC gain varies by record and by channel**, over 31 values from 161 to 500 adu/mV |
| `shdb_af` | **`ECG1`, `ECG2` — and here they mean something.** The only two-lead Holter in the catalogue whose channels have a documented electrode placement: the release states `ECG1` is a **modified CC5** lead and `ECG2` a **NASA** lead, in all 128 records. The names stay as the headers spell them, so `leads=["ECG1"]` selects a known placement rather than a bare position — but neither is one of the standard twelve, so this still must not be stacked with 12-lead data |
| `apnea_ecg` | **`ECG` — one channel, and the second dataset here that is not called `I`.** All 70 headers name the single overnight channel `ECG`, and the release documents no electrode placement anywhere — not on the landing page, not in `annotations.html`, not in `additional-information.txt`. The Holter montage makes a modified chest lead the likely guess, but a guess is what naming it `II` or `V2` would ship, so it stays a channel position |
| `ltstdb` | **`ECG, ECG` in 22 of 86 records, and 11 further layouts in the other 64** — the only dataset here where the lead **count** varies as well as the names: 68 records store two signals and 18 store three. The modal layout is the 22 records whose headers say "Electrode locations were not recorded", so `leads=["ECG"]` returns signal 0 for those and **no name reaches signal 1**. No lead is in all 86 (MLIII 29, V4 27), and `V4/MLIII` and `MLIII/V4` are both present, 20 records and 6. See below |
| `ecgiddb` | **`ECG I`, `ECG I filtered` — two channels holding ONE lead.** Identical in all 310 headers, raw first. Both are Lead I from limb clamps; channel 1 is the author's own offline preprocessing of channel 0 (level-9 `db8` wavelet baseline removal, adaptive 50 Hz bandstop, 5th-order Butterworth lowpass at 40 Hz), and it is zero-phase, so the two are sample-aligned. `config.leads` is 2 because that is the tensor shape; the catalogue says 1 electrode pair. Select `leads=["ECG I"]` or a model gets the same lead twice — the same trap as `wctecgdb`, which ships every one of its 37 channels raw *and* filtered |

**Two datasets store more than one lead layout.** `zzu_pecg` holds 12 leads for
12,334 records and 9 for the other 1,856, and the reduced layout is not a prefix of
the full one — it drops V2, V4 and V6, so stored position 7 is V2 in one and V3 in the
other. A single `lead_names` list would therefore return the wrong physical lead for
13% of the release without any error. The config declares the second layout in
`alternate_lead_names`, and `ECGDataset` re-resolves the requested **names** against
whatever layout each record actually uses:

```python
# Present in both layouts -> the same physical leads for every record.
ds = ECGDataset("zzu_pecg", split="train", data_path="...",
                window=(0, 2500), leads=["I", "II", "V1", "V5"])

# Absent from the 9-lead layout -> refuses, rather than returning V3.
ds = ECGDataset("zzu_pecg", split="train", data_path="...", leads=["V2"])
ds[i]   # ValueError: Lead 'V2' is not in 'zzu_pecg'. Available: [... 'V1', 'V3', 'V5']
```

`stdb` is the second case and the simpler one: ten of its 28 records hold one
channel instead of two, and that layout **is** a prefix of the full one, so nothing
returns the wrong physical lead. Declaring `alternate_lead_names: {1: ["ECG1"]}`
buys the error message instead — `leads=["ECG2"]` refuses against a named layout for
those ten records rather than falling into the generic too-few-leads path.

A dataset that declares no `alternate_lead_names` — every other one — is asserting a
single layout, and behaves exactly as before. Note that batching either of these
needs `leads=` as well as `window=`: a batch mixing layouts cannot be stacked, and
for `stdb` that is `RuntimeError: stack expects each tensor to be equal size, but
got [2, 10800] at entry 0 and [1, 10800] at entry 2`.

**And three datasets vary the lead *names* at a constant lead count**, which a
count-keyed map cannot express at all. Every one of `mitdb`'s 48 records stores
exactly 2 leads, but only 40 store `MLII, V1`: two each store `MLII, V5`, `MLII, V2`
and `V5, V2`, one stores `MLII, V4`, and record 114 stores `V5, MLII` — the
predominant pair reversed, which the source documents as something that happens in
clinical practice. So `signal[0]` is a limb-type lead in 46 records and a chest lead
in 2, and nothing about a signal's shape says which. The config lists every layout in
`record_lead_layouts`, and `ECGDataset` then reads each record's own header to
resolve the requested names:

```python
ds = ECGDataset("mitdb", split="train", data_path="...",
                window=(0, 3600), leads=["MLII"])

ds[0]["signal"]    # record 100: MLII from position 0
                   # record 114: MLII from position 1 -- an index returns V5
                   # record 102: ValueError -- it stores V5/V2 and has no MLII
```

`edb` is the same problem several times over, and is the reason to take this
mechanism seriously rather than treat `mitdb` as a curiosity. Lead placement was never
standardised across the seven countries that contributed to the European ST-T
Database, so its 90 two-lead records use **fifteen different orderings of eleven
different lead pairs** — and, unlike `mitdb`, **no lead appears in every record**. V5
reaches 51 of 90, MLIII 47, V4 34, and `D3` exactly one. Worse, `MLIII, V4` and
`V4, MLIII` are *both* present, 15 records each, so for a third of the release
`signal[0]` is a limb lead or a chest lead with equal probability and nothing
distinguishes the two cases. The declared `lead_names` covers 19 records:

```python
ds = ECGDataset("edb", split="train", data_path="...",
                window=(0, 2500), leads=["V5"])

len(ds)            # 74 records in the split...
ds[0]              # ValueError: Record 'e0104' stores ['MLIII', 'V4'] ... 43 of the
                   # 74 resolve; the other 31 store no V5 at all
```

Because no lead is universal, **`leads=` alone does not make `edb` batchable** — you
need a lead *and* a record filter, and V5 at 57% of the release is the widest choice
available. The per-record layout is in the labels as `lead_names`.

`qtdb` is the third, and it adds a wrinkle neither of the others has: **its modal
layout is not a real lead pair.** 57 of its 105 records describe both channels only as
`ECG1`/`ECG2`, so the declared `lead_names` is a placeholder that covers a majority of
the release, and the other 48 records use 19 further layouts. Since every `qtdb` record
is a fifteen-minute excerpt of another database's recording, the consequence is a
cross-dataset one: the 33 European ST-T excerpts keep the ESC's original electrode
names while `edb` relabelled the same channels to standard ones, so `edb`'s `MLIII` is
`qtdb`'s `D3` or `ML5`, its `V5` is `CM5`, its `V2` is `CM2`, `V1-V2` or `V2-V3`.

```python
# Of the 33 records the two datasets share, this selects 14 under edb's names...
ECGDataset("edb", split="train", data_path="...", leads=["V5"])
# ...and 2 under qtdb's, over signals that are bit-identical.
ECGDataset("qtdb", split="train", data_path="...", leads=["V5"])
```

Nothing returns the wrong lead — no name maps to a different physical channel in the
two releases — but any code selecting by name silently covers a different set of
records. `leads=["MLII"]` resolves for 11 of the 85 records in `qtdb`'s train split and
refuses the other 74.

`ltstdb` is the fourth, and it is the only one that varies the lead **count** as
well — which is why it needs `record_lead_layouts` rather than the count-keyed
`alternate_lead_names` that would otherwise cover a 2-vs-3 split. 68 of its 86
records store two signals and 18 store three, in twelve layouts, and the largest
single layout is the **22 records that name nothing**: their headers describe both
channels as `ECG` and state "Electrode locations were not recorded". So for those 22
`leads=["ECG"]` returns signal 0 and *no name reaches signal 1* — not a limitation of
this mechanism but of the release, which never recorded where the electrodes were.

```python
# 29 of 86 records hold MLIII -- the widest any lead reaches here.
ds = ECGDataset("ltstdb", split="train", data_path="...",
                window=(0, 2500), leads=["MLIII"])
ds[i]   # ValueError: Record 's20011' stores ['ML2', 'MV2'], and this dataset uses
        # more than one lead layout. Lead 'MLIII' is not in 'ltstdb'.
```

This dataset therefore **cannot be batched whole by any `leads=` value**: no lead is
universal, and a batch mixing 2- and 3-signal records raises in `default_collate`
regardless. Filter on `n_leads`/`lead_names` from `ecgbench.labels.ltstdb` first, or
use `batch_size=1`. See `examples/load_ltstdb.py`.

`record_lead_layouts` is wfdb-only, because no other format names its leads per
record. Datasets that do not declare it are unaffected.

**And one dataset names no leads whatsoever.** `afdb` — the MIT-BIH Atrial
Fibrillation Database, sibling to `mitdb` from the same hospital — calls its two
channels `ECG1` and `ECG2` in every header and states no electrode placement
anywhere in the release. So `leads=["ECG1"]` selects a **channel position**, not a
known anatomical lead, and the obvious inference from `mitdb` is not supported by
anything in the data. Where `mitdb` documents which of MLII/V1/V5 each record holds,
`afdb` documents nothing, and the honest config is the one that says so.

`shdb_af` is the one exception in the whole two-lead group, and it is worth knowing
about because it makes the others' silence look like the choice it is. Its headers spell
`ECG1`/`ECG2` like `afdb`'s, but the release also says what they are — `ECG1` a modified
CC5 lead and `ECG2` a NASA lead — so those two names carry a placement rather than only
a position. The config still declares the names the files use rather than `CC5`/`NASA`,
because spelling leads as the source spells them is the rule and the headers are the
source. Both are Holter placements, not members of the standard twelve.

`ltafdb` goes one step further and does not even number them: every one of its 84
headers ends **both** signal lines with the bare description `ECG`. Two identically
named channels cannot be resolved by name — `_resolve_leads` keys on the first
occurrence and rejects a repeated request — so declaring `["ECG", "ECG"]` would make
channel 1 unreachable through `leads=` entirely. Its config therefore declares the
positional names `ECG1`/`ECG2`, matching `afdb` so cross-dataset code sees one
convention, and says plainly that they are ECGBench's names rather than the files'.

**Three datasets' record ids are zero-padded numbers**, which is a bigger deal than it
sounds. `afdb`'s records are named `00735`, `03665`, `04015`; read with pandas'
default type inference they become 735, 3665, 4015, and from there the record id
stops matching the source, the label join misses, and `data_path / "735"` is not a
file — so every record fails `corrupt_header` for a reason nothing in the traceback
mentions. Its config sets `zero_padded_identifiers: true`, which makes every
metadata and fold-CSV read keep the record-id, patient-id and signal-path columns as
strings. If you read the published fold CSVs yourself, do the same:

```python
pd.read_csv("afdb/clean/folds.csv", dtype={"record_name": str, "signal_path": str})
```

`ltafdb` is the second, and it loses more: seven of its 84 records are named
`00`, `01`, `03`, `05`, `06`, `07` and `08`, which collapse to single digits that
resolve to nothing at all.

`shdb_af` is the third, and the only one where the padding is a stated part of the
de-identification rather than an accident of numbering: every recording was given a
random three-digit id in `000`-`143` and "padded with zeros to maintain consistent
length", so **all** 128 ids are three characters and **88** of them begin with a zero.
It is also the only one of the three where the padding hides a second problem: the ids
are not sequential — 16 values in `000`-`143` are unused, including the `016` and `030`
that v1.0.1 withdrew as duplicates — so a gap in the numbering is not evidence of a
missing download.

The flag is opt-in, because forcing it on would change `ds[0]["record_id"]` from an
int to a string for the six datasets whose ids are genuinely numeric. Forgetting it
is caught rather than remembered: `export_splits` refuses to write a zero-padded
identifier from a config that has not declared one.

**One dataset has no physical units at all.** `echonext` ships waveforms its
publisher median-filtered, percentile-clipped and standardised with an unreleased
mean and SD, so no scale factor recovers millivolts. Its config declares
`signal_units: zscore`, and `units=` refuses rather than silently multiplying
dimensionless numbers by 1000:

```python
ds = ECGDataset("echonext", split="test", data_path="...", metadata_source="local")
ds.units                       # 'zscore' -- reported honestly, not 'mV'
ds[0]["signal"].min()          # -6.829

ECGDataset("echonext", units="uV", ...)
# UnitConversionError: This dataset's samples are stored as 'zscore', not a
# physical unit, so they cannot be converted to 'uV'. ...
```

Every other dataset declares `signal_units: mV` (the default) and is unaffected.
`amplitude_outlier` validation is skipped for non-mV sources, since its thresholds
are millivolts.

**One dataset's millivolt scale is an estimate rather than a declared value.**
`ningbo_iva` ships bare integers, and neither its paper nor figshare states a gain
— the paper's own figures plot the raw counts. Its `signal_unit_scale` of
`6.1035e-05` (1 mV = 2¹⁴ counts) was measured by comparing median lead-II R-peak
amplitude, sex for sex, against `sph`, whose samples are millivolts by
declaration; the two sexes bracket it at 14,029 and 17,111 counts/mV. **Waveform
shape is exact; absolute calibration is good to roughly ±20%.** Divide the
millivolt values by `6.1035e-05` to recover the shipped integers if you would
rather calibrate them yourself.

`signal[4]` is aVL in most of them and aVF in both MIMIC datasets, so slicing by index across
datasets silently crosses two leads. Matching is case-insensitive — `leads=["aVL"]`
works on the lowercase datasets too — an unknown lead lists what is available, and
a duplicate is rejected.

Three datasets are not 12-lead at all. **STAFF III** stores only **9**, and in the
opposite order to everything else: `V1-V6` first, then `I, II, III`. aVR, aVL and
aVF are exact linear combinations of I and II and were never stored, so the montage
is a standard 12-lead one clinically while `signal[0]` is V1 rather than lead I —
the single most likely way to misread this dataset. **PTBDB** stores 15 signals, the conventional
twelve plus the three Frank vectorcardiography leads; `leads=` is how you take the
standard twelve out of it. **`wctecgdb`** stores 37: I/II/III, V1-V6, the three limb
electrode potentials LA/RA/LL and the six true unipolar chest leads UV1-UV6, each
present both raw and after DC removal plus a 0.05-150 Hz band-pass, plus the Wilson
Central Terminal itself. Index 0 is raw lead I and index 18 is *filtered* lead I —
the same signal in two preprocessing states — so `leads=` by name is the only safe
way to read it, and aVR/aVL/aVF have to be derived from I and II. Its records are also **variable length**
(32 s to 120 s), so batching needs a fixed `window=` — see `examples/load_ptbdb.py`.

`leipzig_heart_center_ecg` goes further: it is the one dataset where the channel
count is **not constant**. Every record holds the 12-lead surface ECG *plus* the
intracardiac electrograms from whichever catheters were in place, giving 14, 18, 19
or 20 channels in six distinct layouts — and only channels 0-11 are the same channel
in the same position in every record (index 12 is `ABL12`, `RVA12` or `ART`
depending on the record). Its `lead_names` therefore declares the ECG and nothing
else, deliberately, so `leads=` resolves to the right physical lead everywhere. To
reach an intracardiac channel, look it up by name in that record's own header:

```python
from ecgbench.labels.leipzig_heart_center_ecg import channel_index

channel_index(labels["channel_names"], "RVA12")   # 13 in most records, 18 in x100
channel_index(labels["channel_names"], "CS12")    # None where that catheter is absent
```

Pass `leads=` if you want a homogeneous batch; without it a batch mixes 14-, 18-,
19- and 20-channel tensors. See `examples/load_leipzig_heart_center_ecg.py`.

`incartdb` is the one dataset whose primary labels are **reference beat
annotations** rather than record-level diagnoses: 175,907 manually corrected beats
over ten types, exposed as per-record counts (`beat_N`, `beat_V`, …, `pvc_fraction`)
alongside the patient diagnosis and free-text per-record findings. Its records are
1800 s (~44 MB each), so batching needs a `window=`. It is also the
clearest case for **patient-grouped folds** — 3,166 of its 3,174 RBBB beats come
from a single patient — see `examples/load_incartdb.py`.

`staffiii` is the one dataset whose label is a **position in a procedure** rather
than a diagnosis. Each of its 104 patients was recorded before, during and after an
elective coronary angioplasty, so `recording_type` (`BR`/`BC`/`BI`/`PC`/`PR`) marks
which recordings were taken while a balloon was occluding a coronary artery — 152
inflations, 28-595 s each, with sample-accurate inflation, deflation and
contrast-injection times from the shipped `.event` files. That makes it the
reference set for transient ischaemia, with each patient as their own control. Two
traps: its **9 leads start with V1**, and record length correlates strongly with the
label (inflation records have a median of 518 s against 300 s elsewhere), so window
to a fixed length before training. See `examples/load_staffiii.py`.

`brugada_huca` is the smallest and cleanest dataset here — 363 records, one per
subject, all 363 passing validation — and the only one sampled at **100 Hz alone**
(PTB-XL offers 100 Hz as an alternative to 500). Its labels are bare integers with
no string form in the CSV: `brugada` is 0 healthy / 1 confirmed / 2 other-atypical,
and `ecgbench.splitting.strategies.brugada_huca.BRUGADA_CLASSES` carries the
meanings. Treat it as a **screening cohort**: class 0 means "investigated and not
diagnosed", not a general-population control. See `examples/load_brugada_huca.py`.

`mimic_iv_ecg` is the largest dataset here — 800,035 records from 161,352
patients (~96.5 GB) — and the one where `fold_numbers=` matters most: a single fold
is a tenth of it. Two facts to know before using its labels. They are **free-text
machine reports** (up to 18 lines per study, joined into `report_text`), not codes,
and `primary_report` is only the first line, which is sometimes a data-quality
warning rather than a rhythm. And its numeric measurements encode "not measurable"
as **integer sentinels** — `29999`, `32767`, `65535` — which ECGBench converts to
NaN; read the CSV yourself and a mean P-wave axis comes out meaningless. See
`examples/load_mimic_iv_ecg.py`. Its 659-record open demo is a separate config,
`mimic_iv_ecg_demo`, which has no labels at all.

`challenge2021` and `challenge2020` are the datasets where **sampling rate varies
per record** (257/500/1000 Hz), because each concatenates several source cohorts.
Rate is therefore a label to filter on, not a `sampling_rate=` argument, and record
length spans 5 s to 1800 s so batching needs a `window=` too. `challenge2021`
**contains** PTB-XL, PTBDB, INCART, CPSC-2018, Chapman-Shaoxing and Ningbo;
`challenge2020` contains the first four. Their `source` label says which cohort each
record came from, and evaluating on any of those after training is testing on
training data. See `examples/load_challenge2021.py` and
`examples/load_challenge2020.py`.

**The two challenge years are the same recordings.** All 43,101 `challenge2020`
records are in `challenge2021`, bit-identical — verified against both releases'
published `SHA256SUMS.txt`. They are separate configs because the label encodings
differ: 2020 scored 27 classes and 2021 scored 30, and 631 of the 2020 headers list
a SNOMED code twice inside their own `#Dx` field (`ecgbench.labels.challenge2020`
deduplicates them, which is what makes the shipped data reproduce the official code
table). Never train on one year and evaluate on the other.

`cpsc_2018` is the CPSC-2018 public training set as a dataset in its own right —
6,877 records, one rate (500 Hz) but **6 s to 144 s** record length, so `window=`
is mandatory and must fit the 6 s minimum. Its nine classes are multi-label (476
records carry two or three), and its **primary diagnosis is gone**: the WFDB copy
everyone uses sorted each `#Dx` list by class index, so CPSC's original
First/Second/Third labelling is unrecoverable and `stratify_dx` is a folds-only
reduction. All 6,877 records are byte-identical to the `cpsc_2018` cohort of both
challenge years, under the same `A####` names — so this is the fourth way into the
same recordings. See `examples/load_cpsc_2018.py`.

`sph` is the largest **single-source hospital** dataset here — 25,770 records from
24,666 patients at one Chinese hospital — and stored as **HDF5** (`pip install
ecgbench[hdf5]`), one `(12, N)` float16 array per record, already in millivolts. Its
labels are AHA/ACC/HRS standardised statements rather than a bespoke vocabulary: 44
primary statements in 11 categories, each optionally qualified by one of 15
modifiers, so a record reads `60+310;147`. 14.45% of records carry more than one
statement and there is **no primary diagnosis**, so `stratify_code` is a folds-only
rarest-code reduction. 1,066 patients contributed 2-5 records, so folds are grouped
on `patient_id` — the grouping was verified on the output, not assumed. Length runs
10-56 s, and the metadata's `N` column gives it exactly per record, so nothing has
to open a signal file to learn a length. See `examples/load_sph.py`.

`ningbo_iva` is the only dataset here whose label is **invasive ground truth**: 334
12-lead ECGs recorded during catheter ablation, each labelled with the outflow tract
(RVOT 257 / LVOT 77) the ablation proved the arrhythmia came from, so the task is to
predict the origin from the surface ECG before the procedure. Three things about it
are unlike everything else: the lead order is **alphabetical** (`signal[0]` is aVF),
the sampling rate is **2000 Hz** — the highest in the catalogue, from an EP-lab
system rather than a diagnostic cart — and the samples carry **no declared unit**,
so the millivolt scale is an ECGBench estimate (see "Leads and units"). Length runs
2.9-59.3 s in 317 distinct values over 334 records. See
`examples/load_ningbo_iva.py`.

`code15` is the **largest dataset in the catalogue** — 345,779 records from 233,770
Brazilian telehealth patients — and the first where a record is a **row of a shared
array** rather than a file: 18 HDF5 parts each hold one `(N, 4096, 12)` array, so a
signal path reads `exams_part0.hdf5:tracings:417`. (2-D HDF5 arrays are
`(leads, samples)` as in `sph`; 3-D ones are `(records, samples, leads)` and get
transposed.) Its label trap is worth stating twice: six binary flags ship, 308,004
records carry none of them, and **only 134,657 of those are flagged `normal_ecg`** —
so half the release has some finding the six-class vocabulary cannot name, and a
model trained on the flags alone treats 173,347 records as confident negatives for
everything. It also carries **mortality follow-up**, missing for 112,132 records,
where missing means "not followed up" rather than "survived". Folds are grouped on
`patient_id`; 66,929 patients contributed more than one record. See
`examples/load_code15.py`.

`code_test` is its 827-record sibling — the hold-out evaluation set of the same
paper, from the same cohort — and the most heavily annotated dataset here: **seven
independent readings of every record** (two cardiologists, the gold standard
adjudicated from them, two cardiology residents, two emergency residents, two
medical students, and the paper's DNN), all exposed side by side. Two things to know.
It has **no identifiers at all** — one `(827, 4096, 12)` array and eight keyless
tables aligned by row position — so `record_id` is the row index and every source
file is refused unless it has exactly 827 rows. And its limb-lead order is
`aVL, aVF, aVR` where `code15`'s is standard, despite the shared cohort and rate, so
crossing the two by index silently swaps three leads. ECGBench folds it ten ways like
everything else, but the release is an evaluation set: use
`split=None, fold_numbers=range(1, 11)` for all 827 and train on `code15`. Verified
against the waveforms, the two share **no recordings**. See
`examples/load_code_test.py`.

`norwegian_athlete_ecg` is the smallest dataset here — 28 records, one per elite
Norwegian endurance athlete — and the only one whose **amplitudes are not
calibrated**. Every lead of every record was independently min-max normalised to the
full int16 range (all 336 lead-records bottom out at exactly `-32767`), so with the
headers' nominal `50000/mV` gain each lead spans exactly ±0.6553 mV. Absolute and
inter-lead voltages are therefore gone — no LVH or ST-elevation-in-mm criteria — and
no `signal_unit_scale` or `units=` can undo a per-lead normalisation. Morphology and
timing survive. This is undocumented upstream and was established from the files.
A knock-on effect: `missing_leads` and `flat_line` **cannot fire** on it, because a
dead lead would be rescaled to full amplitude like any other.

It is also the only dataset carrying **two independent interpretations per record**,
as WFDB header comments: the GE Marquette SL12 algorithm's and a cardiologist's,
exposed as separate `sl12_*` and `cardiologist_*` label fields. SL12 is the *system
under test*, not the ground truth — it reads 13 of 28 records as borderline or
abnormal where the cardiologist reads normal, and raises a critical `ACUTE MI/STEMI`
alert on 4 athletes, three of whom the cardiologist calls a plain "Normal ECG".
Human labels are degenerate (26 of 28 "Normal ECG", no abnormal class at all), so
folds are stratified on `cardiologist_primary_rhythm` instead, and with 2-3 records
per fold you should rotate folds via `split=None, fold_numbers=[...]` rather than use
the default 24/2/2 mapping. See `examples/load_norwegian_athlete_ecg.py`.

`mhd_effect_ecg_mri` is the one dataset where the **distortion is the point**: 53
ECGs recorded inside 1T, 3T and 7T MRI scanners, where the magnetohydrodynamic
effect (blood ions moving through the static B0 field) superimposes a voltage that
buries the P wave, ST segment and T wave. Amplitudes reach **−31 mV**, far past the
recorders' nominal ±6 mV and ±2.4 mV input ranges, so `amplitude_range_mv` is ±35 —
a conventional ±10 would exclude 16 of 53 records for being exactly what they are
meant to be. 10 records are reference ECGs taken *outside* the bore for the same
subjects (−0.88…+3.09 mV over the same window), standing in for the in-bore ground
truth that cannot be measured. There is no diagnosis to predict: all subjects were
healthy and the 14,950 manual QRS marks carry no beat classification, so the label
is the acquisition condition and the task is signal separation.

It is also the one dataset whose **patient ID had to be derived**. Filename subject
numbers are scoped per scanner — `ECGMRI1T01` and `ECGMRI3T01` are different people
— and three slots belong to subjects recorded in more than one scanner, so grouping
on the number would split one person across folds. `subject_key` is the
sex/age/weight/height tuple instead, collapsing 29 slots into 26 people; folds are
grouped on it and no subject spans a fold. Records mix 12-lead and 3-lead layouts
(only I, II, III are present in every one) and run 24 s to 12 min, so batching needs
both `leads=` and `window=(0, 25000)`. Note the shipped release has 53 records where
the README, PhysioNet page and CinC paper all say 43. See
`examples/load_mhd_effect_ecg_mri.py`.

`wctecgdb` is the one dataset that **measures the reference instead of assuming it**.
Conventional ECG treats the Wilson Central Terminal — the point V1–V6 are measured
against — as 0 V; this release brings the three limb electrodes out individually so the
WCT can be recorded, and its authors report amplitudes reaching **241% of lead II**.
Each of the 540 ten-second segments therefore holds **37 channels at 800 Hz** (8001
samples, 10.00125 s): I/II/III, V1–V6, the limb electrode potentials LA/RA/LL and the
true unipolar chest leads UV1–UV6, **each present both raw and filtered** (DC removal
plus 0.05–150 Hz), then `WCT`. Index 0 is raw lead I and index 18 is *filtered* lead I,
so `leads=` by name is the only safe way in, and aVR/aVL/aVF do not exist here at all.
`amplitude_range_mv` is ±20 because the raw unreferenced channels carry several mV of DC
offset — and 140 of the 540 records have a channel clipped at the ±9.2250 mV acquisition
rail, which validation passes deliberately rather than treating saturation as damage.

Its 540 segments come from **92 patients, 1–31 each** — five patients are 24% of the
dataset — so folds are grouped on `patient_id` and any per-record rate is weighted by
segment count. The only label is a **patient-level free-text admission diagnosis** (43
distinct strings, 10 patients with none, Windows-1252 bytes and four misspellings), which
says why the patient was admitted, not what the ten seconds show; the 8-way
`diagnosis_group` reduction exists to stratify folds, not to train on. Eight records
carry precordial channels **synthesised** as `V = UV − WCT` — flagged per record, and to
be excluded when evaluating precordial reconstruction. See
`examples/load_wctecgdb.py`.

`ecgrdvq`, `ecgdmmld` and `ecgcipa` are the three datasets here with **no diagnosis at
all** — sibling releases from one FDA programme, in order SCR-002, SCR-003 and SCR-004,
and the set to read together because almost every convention they share, they share
inverted.

`ecgcipa` is 5,749 ten-second
12-lead ECGs at **1 kHz** (10,000 samples — the largest 12-lead tensor in the
catalogue) from **60 healthy volunteers** in an FDA Phase I trial, and what varies is
the drug: ranolazine, verapamil, lopinavir+ritonavir, chloroquine, placebo or a
dofetilide/diltiazem crossover. The labels are drug, time from dose, plasma
concentration and nine interval measurements (QT, QTcF, J-Tpeak, J-Tpeakc, …), so
`treatment` is the stratification label and everything else is continuous. Samples are
**microvolts** (`signal_unit_scale: 0.001`), and `units="uV"` returns the source scale.

Three things to know before using it. **Records come in near-duplicate triplicates** —
three segments per subject per timepoint, so 5,749 records are closer to 1,917
observations; patient grouping keeps each triplicate intact. **Every record ships
twice**, as the raw segment and as a derived 16-channel median beat (`+VCGMAG/X/Y/Z`)
whose `.atr` fiducials are what the published intervals were measured from — the median
beats deliberately get no fold of their own. And **the study's own endpoints cannot be
attached to a waveform**: change from baseline lives only on `adeg.csv`'s
triplicate-average rows, which carry no record ID. See `examples/load_ecgcipa.py`.

`ecgdmmld` is the same shape and inverts three of those details. 4,211 ten-second 12-lead
ECGs at 1 kHz from **22 healthy volunteers** in a **complete 5-period crossover** — every
subject took dofetilide alone, dofetilide with mexiletine, dofetilide with lidocaine,
moxifloxacin with diltiazem, and placebo. Samples are **millivolts** (`signal_unit_scale:
1.0`, *not* ecgcipa's 0.001), the limb leads are spelled **AVR/AVL/AVF** rather than
aVR/aVL/aVF, and the 1 kHz is **up-sampled from a 500 Hz acquisition**. The study's
endpoint *is* attachable here — `is_baseline` flags each period's pre-dose triplicate, so
`load_baseline_deltas()` returns change from baseline per record, the thing ecgcipa cannot
give you.

Its own trap is the label. **`treatment` names the period's randomised regimen, not the
drug on board**: the agents were staged hours apart, so only 57% of the dofetilide-arm
records contain dofetilide and a "Mexiletine + Dofetilide" record at 2 h is a
mexiletine-only ECG. Stratify on it, train on the six `plasma_*` columns. Because the
crossover is complete, every fold gets all five arms automatically and no split can
separate them — and with 2–3 subjects per fold, a per-fold metric describes two or three
people. See `examples/load_ecgdmmld.py`.

`ecgrdvq` is the earliest of the three (SCR-002) and the one whose label you can actually
trust. 5,232 ten-second 12-lead ECGs at 1 kHz from **22 healthy volunteers** in a 5-period
crossover of **single agents** — ranolazine, dofetilide, verapamil, quinidine and placebo,
one per period — so `treatment` names the drug rather than a staged combination, and 93–94%
of each active arm's records carry a measured concentration of exactly that drug. It shares
ecgdmmld's millivolts, its uppercase **AVR/AVL/AVF** and its 500 Hz → 1 kHz up-sampling,
and it computes change from baseline the same way. Reconstructed placebo-corrected from the
shipped files, it recovers its own finding: **all four drugs prolong QTcF by +17 to +95 ms,
while J-Tpeak separates them** — +37 and +24 ms for the predominant-hERG blockers
(dofetilide, quinidine) against +6 and −8 for the multichannel ones (ranolazine,
verapamil).

Four things differ from its siblings. Triplicates are **exact** (all 1,744 groups hold 3,
so 5,232 records are ~1,744 observations). The pharmacokinetic table is **long, not wide** —
`plasma_analyte` names the one agent measured — and **dofetilide is pg/mL while the other
three are ng/mL**, so use the derived `plasma_concentration_ng_ml` across arms; `dose`
carries the same split (500 µg vs 120–1500 mg). Its median beats are **variable length**
(968–1,876 samples, against ecgdmmld's fixed 1,200) and 9 are missing entirely, which is
why 9 records have no PR/QRS/QT/J-Tpeak. And **secondary T peaks are real here** — 42
records populate `tpeak_tpeakp_ms`, where ecgdmmld's copy of that column is empty in every
row. Two PR values are stored as a 32-bit arithmetic wrap and are repaired, flagged by
`pr_ms_repaired`. See `examples/load_ecgrdvq.py`.

`medalcare_xl` is the only **synthetic** dataset here: 16,842 ECGs produced by
electrophysiological simulation rather than recorded from anyone. Three consequences.
Its label is *exact by construction* — the condition the simulator was configured to
produce — which makes it excellent for pre-training, augmentation and controlled
ablations, and misleading as a standalone benchmark, since separating these classes
means separating simulator settings. Its signals are the **only `csv_lead_rows`
files** in the catalogue: 12 rows × 5000 columns with **no header**, the transpose of
`chapman_shaoxing` and `ningbo_iva`, and reading one layout with the other's reader
returns a plausibly-shaped array of the wrong thing rather than raising. And each
record ships **three times** — raw simulator output, the same with noise, and a
0.5–150 Hz filtered version — which are one record in three renderings, not three
records; the config wires up `filtered` and the label loader carries the other two.

Two things it is worth knowing before splitting on it. It uses the **authors' own**
train/validation/test directories as folds 1/2/3 (so `--n-folds` does nothing and
there are three fold CSVs, not ten), and their stated guarantee — that a ventricular
simulation model appears in only one split — holds *within* each condition and fails
*across* them: model `S64` is test-side for sinus/avblock/lbbb/rbbb/mi and train-side
for the three atrial conditions, `S67` likewise for validation. Verified at the
parameter level, and no records are shared, so it is shared anatomy rather than
duplicate rows; `model_id` is in every fold CSV so you can regroup. Separately, its
full **simulation parameters** — ~126 keys per record covering the ionic model,
conductivities, APDs, stimulus sites, ischaemic geometry and electrode positions —
are the real ground truth and are exposed opt-in via `load_simulation_parameters`,
not by `labels=True`, which would otherwise open 33,684 files. See
`examples/load_medalcare_xl.py`.

`afdb` is the only dataset here whose **`original` version cannot be iterated**. Two
of its 25 records — 00735 and 03665 — ship reference rhythm annotations and no signal
file at all: the release never published their ECG, and their headers declare zero
signals and zero samples. ECGBench keeps them so `original` matches the published
record count and flags them invalid so `clean` holds the 23 that can be read, but
there is nothing to return for them, so `ds[i]` raises and any `DataLoader` over
`original` fails on the batch containing them. Everywhere else `original` holds
records that are *flagged* but readable. Take `original` to see what was excluded and
why, `clean` for anything that reads waveforms — and note that both records' labels
are real and available either way.

Its label is **AF burden** rather than a diagnosis, since every subject has atrial
fibrillation: the fraction of annotated time in AF runs from 0.24% to 100% across the
25 records, over 254.7 hours of two-lead Holter with 623 manually reviewed rhythm
episodes. The records are 10 h (9,205,760 samples, ~74 MB of float32), so batching
needs a `window=`; length is not uniform, so the window has to fit inside 06453's
8,325,000 samples. Folds are stratified on a **binary** 20% burden cut and not on the
3-class `af_class`, because 25 records over 10 folds leaves no room for a class of 3.
See `examples/load_afdb.py`.

`sddb` is the mirror of that case: the only dataset here whose **`clean` version is
unusable**, so it must be loaded with `version="original"`. WFDB's invalid-sample
marker in format 212 is digital −2048, which `wfdb` returns as NaN, and 20 of its 23
records carry some — 201,708 samples release-wide, as brief scattered analog-tape
dropouts at most 1.79 s long. `check_nan_values` has no threshold, so all 20 fail it,
and because the three unaffected records all land in train folds, `clean` holds **3
records with empty val and test** while `original` splits 19/2/2. Since `ECGDataset`
defaults to `version="clean"`, `ECGDataset("sddb", split="val")` raises a misleading
`No record in split 'val' matched a label row` — the split is simply empty. The check
was kept rather than dropped because removing it would leave `quality_issues` empty
for every record and hand users NaN tensors, and a NaN loss, with no warning at all;
`scan_invalid_samples` gives the per-record counts.

Three more things set it apart. Its defining event — the onset of the terminal
ventricular arrhythmia — is recorded in a **header comment** (`#vfon:`, elapsed from
the record start, in 20 of 23 records) and in no annotation file, and it lands from
6.0% to 98.9% of the way through, so no single `window=` captures it. It is the only
dataset with **two annotators covering different records**: unaudited `.ari` for all
23 (1,888,495 beats) and audited `.atr` for only 12 (849,831), with disjoint symbol
vocabularies, so every beat column is prefixed and `aami_*` is the only comparable
count. And its `.ari` **`(AFIB` markers are not an AF label** — they miss one of the
four published AF subjects and flag six sinus subjects at 22–36%; use `rhythm_class`.
See `examples/load_sddb.py`.

`challenge2017` is the first **single-lead** dataset here, and the only one whose
label vocabulary includes "unusable signal": its 8,528 handheld AliveCor recordings are
labelled normal / AF / other rhythm / **too noisy to classify**, so signal quality is
part of the task rather than something to preprocess away. Three things to know before
using it. Its one channel is called **`ECG`, not `I`** — the device gives a lead I
equivalent but does not enforce orientation, so the paper reports that many traces are
inverted and no record says which. Records run **9.05–60.95 s in 1,487 distinct
lengths**, so batching needs a `window=` sized to the shortest (2,714 samples), and
length *correlates with the label*, so a model fed whole records can learn duration
instead of rhythm. And the shipped **`validation/` directory is not a split** — its 300
`.mat` files are byte-identical to `training/` records, so it takes part in the folds
and is flagged `in_challenge_validation_subset` for exclusion instead. Labels ship in
four revisions, all exposed: 412 of 8,528 changed between the first and last, almost
entirely into the noisy class, and the shipped file numbers are one behind the paper's
V1/V2/V3. No demographics or patient identifiers exist at all, so folds are stratified
but **ungrouped**. See `examples/load_challenge2017.py`.

`apnea_ecg` is the second single-lead dataset, and the one where **the release's own
train/test split leaks subjects** — the first case in this catalogue where a provider's
division had to be rejected rather than adopted. Its 70 overnight recordings carry an
expert apnea annotation for **every one of their 34,313 minutes**, so the ground truth
is per minute (`apnea_sequence`, one `A`/`N` character each) and the record-level
`apnea_class` is only a whole-night summary. Three things to know. The 70 records come
from **30 subjects**, and Apnea-ECG ships **no subject identifier anywhere**, so nothing
warns you that 27 of them contributed two to four nights — 18 subjects, **49 of the 70
records**, have recordings on both sides of the challenge's a/b/c vs x division.
`subject_id` is reconstructed from the age/sex/height/weight published per record (32
distinct values, the quoted subject count) and folds are grouped on it;
`has_predefined_splits` is `false` and `challenge_set` survives as a label. **Two pairs
of records are the same recording**: `x35` is `x22` shifted 40 s and `c06` is `c05`
shifted 80 s, 100.000% identical over 2.8 M samples, and the demographics of `x22` and
`x35` contradict each other — both are kept, grouped into one fold. And records are
whole nights, 2,430,000–3,462,000 samples, so batching needs a `window=` sized to the
shortest; `window=(i * 6000, 6000)` returns exactly minute *i*, the labelled unit. See
`examples/load_apnea_ecg.py`.

`ecgiddb` is the catalogue's **biometrics** dataset, and the first where the label and
the patient column are the same thing. Its 310 twenty-second Lead I records from 90
volunteers exist to test whether an ECG identifies the person who produced it — there is
no diagnosis in the release at all — so `subject_id` is the ground truth *and*
`patient_id_column`. The consequence is worth stating plainly: **ECGBench's folds cannot
be used for this dataset's own task**, because grouping by subject puts each person
wholly inside one fold, so no fold's model has seen the person it would be asked to
recognise. That is right for any other use of these recordings (89 of the 90 subjects
have more than one record) and wrong for identification, which needs a *within*-subject
split — `session_index` and `is_multi_session` are exposed for exactly that, and 20 of
the subjects span 2–6 sessions up to 156 days apart. Three more things. **Every record
stores the same lead twice**, `ECG I` raw and `ECG I filtered`, so `leads=["ECG I"]`
matters. **Length is uniform** — all 310 records hold exactly 10,000 samples, so unlike
almost everything else here any `window=` fits every record. And the **`.atr`
annotations stop at 25–59% of the record**: ten unaudited machine-detected R- and
T-peaks per record and nothing after second 12, which `annotated_fraction` records. The
thesis's own 195/115 train/test division exists only in prose and is unrecoverable. See
`examples/load_ecgiddb.py`.

Both are **read-time adapters**: they shape the returned tensor only. Source files,
fold CSVs and validation are untouched — a record excluded for a flat V6 stays
excluded even if you never load V6.

### ECGDataset parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `str \| DatasetConfig` | *required* | Dataset slug or config object |
| `split` | `str \| None` | `"train"` | `"train"`, `"val"`, `"test"`, or `None` to select purely by fold |
| `version` | `str` | `"clean"` | `"clean"` or `"original"` |
| `data_path` | `Path \| str \| None` | `None` | Path to signal files; auto-downloads if None |
| `sampling_rate` | `int \| None` | `None` | Sampling rate (default: dataset's default) |
| `fold_numbers` | `list[int] \| None` | `None` | Specific folds to load; None = all folds of the split |
| `window` | `tuple[int, int \| None] \| None` | `None` | `(start, length)` in samples, e.g. `(0, 2500)`; read at load time |
| `transform` | `Callable \| None` | `None` | Transform applied to signal tensor, after `window`/`leads`/`units` |
| `metadata_source` | `str` | `"hf"` | `"hf"` (HuggingFace) or `"local"` |
| `labels` | `bool` | `False` | Attach per-record labels as `sample["labels"]`; needs local source data |
| `leads` | `list[str] \| None` | `None` | Select and reorder leads by name, e.g. `["I", "II", "V5"]` |
| `units` | `str` | `"mV"` | `"mV"` or `"uV"` — applied before `transform` |

### Sample windows

`window=(start, length)` returns a fixed slice of each record, in samples:

```python
first  = ECGDataset("ptbxl", split="train", data_path="...", window=(0, 2500))
second = ECGDataset("ptbxl", split="train", data_path="...", window=(2500, 2500))
first[0]["signal"].shape    # (12, 2500) -- samples 0-2499
second[0]["signal"].shape   # (12, 2500) -- samples 2500-4999
```

`length=None` reads to the end of the record. Prefer `window=` over a cropping
`transform` for two reasons:

- **It is pushed down into the reader**, so only those samples are decoded. On
  long records that is a large difference — `incartdb` goes from ~106 ms to ~8 ms
  per record; on 10-second records it changes nothing.
- **It is picklable.** `transform=lambda x: x[:, :2500]` fails in a
  `DataLoader(num_workers>0)` under the `spawn` start method, the default on
  macOS and Windows. `window=` works under both `fork` and `spawn`.

A window that does not fit raises `WindowOutOfRangeError`, naming the record and
its true length. Record length is not constant in every dataset — `cpsc_2018`
runs 6-144 s, `ptbdb` 32-120 s, `staffiii` 94.5-960 s, `sph` 10-56 s and
`ningbo_iva` 2.9-59.3 s — so a fixed window can fit most records and not all.

`window` combines freely with `fold_numbers`, `leads` and `units`; it is applied
first, then lead selection, then units, then `transform`.

### Derived datasets (annotations for another dataset's records)

Some releases contain no recordings of their own — they annotate, or re-cut,
someone else's. There are three: **PTB-XL+** (3 feature tables, 2 statement
tables, derived median beats and 283,326 fiducial-point files, all keyed by
PTB-XL's `ecg_id`), **MIMIC-IV-ECG-Ext-ICD** (ICD-10-CM discharge diagnoses for
all 800,035 MIMIC-IV-ECG studies, keyed by `study_id`) and **Symile-MIMIC** (a
multimodal cohort pairing 11,610 of those same MIMIC-IV-ECG studies with a chest
X-ray and 50 blood labs).

Those get **no config and no splits**, deliberately. Their records are the host
dataset's, so generating folds would create a second ECGBench partition of the
same recordings and let someone train on one and evaluate on the other. They are
label providers instead: load the host on its own folds and join.

```python
from ecgbench import ECGDataset
from ecgbench.labels.ptbxl_plus import load_ptbxl_plus

ds = ECGDataset("ptbxl", split="train", data_path="/data/ptb-xl/1.0.3/", labels=True)
plus = load_ptbxl_plus("/data/ptb-xl-plus/1.0.1/", features=("unig",))

joined = plus.reindex(ds.metadata_df["ecg_id"].values)   # 17,376 of 17,376
joined.iloc[0]["ptbxl_scp_codes"]     # [('NORM', 100.0), ('LVOLT', 100.0), ('SR', 100.0)]
joined.iloc[0]["12sl_statements"]     # ['NSR', 'NML']  -- the algorithm's opinion
joined.iloc[0]["unig_QRS_Dur_Global"] # 86.0 ms
```

You need both downloads, since PTB-XL+ has no waveforms. Feature columns are
provider-prefixed because the three providers reuse names. See
`examples/load_ptbxl_plus.py`, and the dataset page for the release's own defects
— notably that `12sl_features.csv` ships with no key column.

Ext-ICD works the same way, and adds one wrinkle worth knowing: it ships the
upstream authors' **own** 20-fold split alongside the labels, which is
independent of ECGBench's 10 folds. Reproduce published numbers on one or work on
ECGBench's folds on the other, but never cross them.

```python
from ecgbench.labels.mimic_iv_ecg_ext_icd import label_set, load_ext_icd, multi_hot

# prefix= because MIMIC-IV-ECG's own label frame also carries ecg_time.
icd = load_ext_icd("/data/mimic-iv-ecg-ext-icd-labels/1.0.1/", prefix="icd_")
codes = label_set(icd, prefix="icd_")          # 1076, the published label set
targets = multi_hot(icd.head(1000), codes, prefix="icd_")
```

Only 58.5% of its records carry a diagnosis at all, and the empty ones are empty
*lists* rather than nulls — see `examples/load_mimic_iv_ecg_ext_icd.py`.

Symile-MIMIC is the same shape with one difference: it is a **cohort**, not a
layer over the whole host. It covers 11,610 of MIMIC-IV-ECG's 800,035 studies, so
a partial join is the correct result rather than a broken one.

```python
from ecgbench.labels.symile_mimic import by_study_id, chexpert_targets, load_cohort

host = ECGDataset("mimic_iv_ecg", split="train", fold_numbers=[1],
                  data_path="/data/mimic-iv-ecg/1.0/", metadata_source="local")
cohort = load_cohort("/data/symile-mimic/1.0.0/", prefix="sym_")   # (11622, 92)
# Rows are admissions, so 12 ECG studies appear twice; the default policy keeps
# the earliest admittime, and on_duplicate="raise" refuses instead.
keyed = by_study_id(cohort, prefix="sym_")                        # (11610, 92)
joined = keyed.reindex(host.metadata_df["study_id"].values)       # 1,135 of 78,655
targets = chexpert_targets(joined, uncertain="nan", prefix="sym_")  # 14 CXR findings
```

Two traps of its own: the column literally named `study_id` is the **CXR's**, not
the ECG's (the loader drops it), and the CheXpert labels have four states — −1.0
means *uncertain* and NaN means *not mentioned*, so `chexpert_targets()` makes you
resolve both. The shipped `data_npy` ECG tensors are min-max normalised to
[−1, 1] with the scale discarded, so they are not millivolts and cannot be
converted back — read MIMIC-IV-ECG for those. See
`examples/load_symile_mimic.py`.

### Datasets with no waveforms at all

A dataset can also lack recordings without annotating anyone else's. The **Eye
Tracking Dataset for 12-Lead ECG Interpretation** ships ten *printed* ECGs and
the gaze behaviour of 63 clinicians reading them — 630 sessions, scored against
16–25 areas of interest per image. There is no sampled signal, no sampling rate,
and no patient behind a record, so it too gets **no config and no splits**: the
unit of observation is a reader session, and folds over "records" would be
partitioning ten pictures. How to split a reader study — by reader or by image —
depends on the task, so ECGBench ships tables and leaves that choice open.

```python
from ecgbench.labels.eye_tracking_ecg import load_eye_tracking_ecg

df = load_eye_tracking_ecg("/data/eye-tracking-ecg/1.0.0/")

# Group by aoi_lead, not Label: labels are scoped per image ("V1 NSR" vs "V1 AFib"),
# and 1/2/3 are leads I/II/III rather than indices.
leads = df[df.aoi_kind == "lead"]
leads.groupby("Group")["Hit_time_G"].mean().round(0)   # Consultant 7266 ms, Med 1 11305 ms
```

Its `-1` "never happened" codes and `0` ages are converted to `NaN` on load —
being sentinels rather than blanks, they make every column look fully populated.
See `examples/load_eye_tracking_ecg.py` and the dataset page.

### Restricted and credentialed datasets

Most datasets' fold CSVs are published to the [HuggingFace
Hub](https://huggingface.co/datasets/vlbthambawita/ECGBench) and download
automatically. **Some are deliberately not**, and those you generate yourself.

Fold CSVs carry identifiers only — record ID, patient ID, signal path, fold,
split. For an openly licensed source that is uncontroversial. For a
**credentialed or restricted** source those identifiers are still data derived
under a use agreement — or material a licence forbids redistributing — and the
ECGBench Hub repository is public and ungated, so ECGBench does not publish them.
Three datasets are in this category: `mimic_iv_ecg`, whose 800,035 `study_id`s and
161,352 `subject_id`s stay with the people who signed the PhysioNet DUA;
`echonext`, under the PhysioNet *Restricted* Health Data License whose clause 3
forbids sharing access to the data at all; and `ikem`, which ships a `LICENSE` file
that is verbatim **CC BY-NC-ND 4.0** — the NoDerivatives term makes republishing a
derived fold table legally unclear, so it is the first dataset here withheld by
licence rather than by an access agreement.

Such a dataset declares this in its config, and the tooling enforces it in both
directions — `ecgbench upload` refuses to publish it, and `ECGDataset` raises
`SplitsNotPublishedError` (carrying the command below) instead of a 404:

```yaml
publish_fold_csvs: false
no_publish_reason: >
  MIMIC-IV-ECG is credentialed under the PhysioNet Credentialed Health Data
  Use Agreement, so ECGBench does not republish its identifiers ...
```

The split is distributed as a **recipe** instead. Because fold assignment is a
deterministic function of the input table and a fixed seed, regenerating locally
reproduces the canonical partition exactly:

```bash
# 1. Generate — writes output/<slug>/ plus a manifest.json
ecgbench splits --dataset mimic_iv_ecg --data-path /path/to/mimic-iv-ecg/1.0/

# 2. Verify it is the canonical partition, not merely a plausible one
python -c "from ecgbench import verify_splits; \
           print(verify_splits('mimic_iv_ecg', 'output/mimic_iv_ecg')['ok'])"

# 3. Point the loader at your generated folds
cp -r output/mimic_iv_ecg/{clean,original} /path/to/mimic-iv-ecg/1.0/
```

```python
ds = ECGDataset("mimic_iv_ecg", split="train", metadata_source="local",
                data_path="/path/to/mimic-iv-ecg/1.0/", labels=True)
```

**`manifest.json` is what makes "regenerate it yourself" trustworthy.**
`ecgbench splits` writes one for *every* dataset, recording the seed, fold count,
grouping column, a SHA-256 of each input file, the record counts, and a **fold
digest** — a hash over the entire record-to-fold mapping in canonical order. Two
runs agree on that digest if and only if they produced the same partition.
`verify_splits()` compares yours against a reference manifest shipped in the
package and, on mismatch, names the input file that differs.

That last part is the common failure. A split only reproduces if the input is
byte-identical, and local copies get filtered: we found a
`machine_measurements.csv` cut to 789,481 of 800,035 rows, which silently
changes the stratification and hence the folds. Verify your download against the
provider's own checksums before generating.

### Output format

Each sample is a dict:
- `signal` -- float32 tensor `(leads, samples)`, in millivolts unless `units="uV"`
- `record_id` -- record identifier
- `split`, `fold` -- split name and fold number
- `labels` -- dict of the dataset's label and metadata fields (only with `labels=True`)
- All other CSV columns as tensors (numeric) or raw values (str/dict)

The dataset object also carries `ds.lead_names` and `ds.units`, so the tensor is
self-describing.

## Data Versions

- **`clean`** (default): only records that pass all quality checks
- **`original`**: all records with `is_valid` and `quality_issues` columns

Both versions share identical fold assignments. Use `original` when you need all records or want to filter manually; use `clean` for standard benchmarking.

## Validation

ECGBench validates every signal file before splitting:

- **missing_leads** -- lead entirely NaN or all-zero
- **nan_values** -- any NaN in signal
- **truncated_signal** -- fewer samples than expected
- **flat_line** -- lead with near-zero variance
- **corrupt_header** -- unreadable signal file
- **amplitude_outlier** -- samples outside physiological range

Results are saved in `validation_report.json` with per-record details.

## Croissant Metadata

Both `clean/` and `original/` versions include MLCommons Croissant 1.1 JSON-LD metadata (`croissant.json`) with SHA-256 hashes for reproducibility. The full pipeline generates both automatically. For standalone generation:

```bash
ecgbench croissant --dataset ptbxl --splits-dir output/ptbxl/clean/ --version clean
ecgbench croissant --dataset ptbxl --splits-dir output/ptbxl/original/ --version original
```

## Adding a New Dataset

1. Copy `ecgbench/data/configs/_template.yaml` to `<slug>.yaml`, fill in fields
2. Run `ecgbench splits --dataset <slug> --data-path /path/to/data/`
3. Check `validation_report.json` -- review excluded records
4. If custom logic needed, create `ecgbench/splitting/strategies/<slug>.py` with `@register("<slug>")`
5. Run `pytest`
6. Upload: `ecgbench upload --data-dir output/ --datasets <slug>`

## CLI

Installing `ecgbench` adds a single `ecgbench` console command with three subcommands:

```bash
ecgbench --help               # top-level help
ecgbench <command> --help     # per-subcommand flags
ecgbench --version            # package version
```

| Subcommand | Purpose |
|------------|---------|
| `splits` | Full pipeline -- validate signals, generate 10-fold splits, export CSVs, and write Croissant metadata |
| `croissant` | Generate Croissant 1.1 JSON-LD for an already-split dataset directory |
| `upload` | Upload fold CSVs and metadata to HuggingFace Hub (requires `ecgbench[hf]`) |

Every subcommand has an equivalent Python function (`run_splits`, `run_croissant`, `run_upload`) with the same arguments, so the same workflow can be driven from a notebook or downstream code.

### `ecgbench splits`

Runs the full pipeline: validate -> split -> export -> Croissant. Writes `output/<dataset>/{original,clean}/` by default.

```bash
ecgbench splits --dataset ptbxl --data-path /path/to/ptb-xl/1.0.3/
ecgbench splits --dataset ptbxl                        # auto-download
ecgbench splits --dataset chapman_shaoxing \
    --data-path /data/chapman/ \
    --output-dir /data/outputs/chapman/ \
    --n-folds 10 --max-workers 8

# PhysioNet ecg-arrhythmia (45,152 records, Chapman-Shaoxing + Ningbo).
# Ships no metadata CSV — the splitter builds ecgbench_metadata.csv from the
# per-record WFDB headers on first run, so the data directory must be writable.
ecgbench splits --dataset ecg_arrhythmia \
    --data-path /data/ecg-arrhythmia/1.0.0/ --max-workers 32
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset` | str | *required* | Dataset slug — see `list_available_configs()` (e.g. `ptbxl`, `ecg_arrhythmia`, `mimic_iv_ecg_demo`) |
| `--data-path` | path | auto-download | Path to the dataset root directory |
| `--output-dir` | path | `output/<dataset>/` | Output directory for fold CSVs + metadata |
| `--sampling-rate` | int | config default | Sampling rate to validate against |
| `--n-folds` | int | `10` | Number of cross-validation folds |
| `--max-workers` | int | `4` | Parallel workers for signal validation |
| `--skip-validation` | flag | off | Skip signal validation (faster; no quality flags) |
| `--skip-croissant` | flag | off | Skip Croissant metadata generation |

Python equivalent:

```python
import ecgbench

result = ecgbench.run_splits(
    dataset="ptbxl",
    data_path="/path/to/ptb-xl/1.0.3/",
    output_dir=None,          # -> output/ptbxl/
    sampling_rate=None,       # -> config default_sampling_rate
    n_folds=10,
    max_workers=4,
    skip_validation=False,
    skip_croissant=False,
)
# result is a dict with: dataset, dataset_name, output_dir,
# original={total,train,val,test}, clean={total,train,val,test}, excluded
```

### `ecgbench croissant`

Standalone Croissant 1.1 JSON-LD generator for an existing splits directory. Run once per version (`clean` and `original`).

```bash
ecgbench croissant --dataset ptbxl --splits-dir output/ptbxl/clean/    --version clean
ecgbench croissant --dataset ptbxl --splits-dir output/ptbxl/original/ --version original
ecgbench croissant --dataset ptbxl --splits-dir output/ptbxl/clean/ --validate
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset` | str | *required* | Dataset slug |
| `--splits-dir` | path | *required* | Version directory to scan (e.g. `output/ptbxl/clean/`) |
| `--output` | path | `<splits-dir>/croissant.json` | Where to write the JSON-LD |
| `--version` | `clean`&vert;`original` | `clean` | Version label to record in the Croissant file |
| `--validate` | flag | off | Validate the file after writing (non-zero exit if invalid) |

Python equivalent:

```python
from pathlib import Path
import ecgbench

saved_path: Path = ecgbench.run_croissant(
    dataset="ptbxl",
    splits_dir="output/ptbxl/clean/",
    output=None,              # -> splits_dir/croissant.json
    version="clean",
    validate=True,            # raises RuntimeError if the file does not validate
)
```

Requires the `croissant` extra (`pip install ecgbench[croissant]`).

### `ecgbench upload`

Uploads each dataset's `original/` and `clean/` CSV folds, plus `validation_report.json` and `croissant.json` if present, to a HuggingFace Hub dataset repository. One or more dataset slugs can be uploaded in a single call.

```bash
ecgbench upload --data-dir output/ --datasets ptbxl
ecgbench upload --data-dir output/ --datasets ptbxl chapman_shaoxing
ecgbench upload --data-dir output/ --datasets ptbxl --dry-run
ecgbench upload --data-dir output/ --datasets ptbxl \
    --hf-repo-id your-org/ECGBench
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data-dir` | path | *required* | Root directory containing per-dataset subdirectories |
| `--datasets` | list | *required* | One or more dataset slugs to upload |
| `--hf-repo-id` | str | `vlbthambawita/ECGBench` | Target HuggingFace dataset repo ID |
| `--dry-run` | flag | off | Print the files that would be uploaded, without uploading |

Authentication resolves in this order: `token=` argument (Python API only) -> `HF_TOKEN` env var -> `HUGGINGFACE_HUB_TOKEN` env var -> `.env` file in the current working directory. Run with `--dry-run` first to review the file list.

Python equivalent:

```python
import ecgbench

counts: dict[str, int] = ecgbench.run_upload(
    data_dir="output/",
    datasets=["ptbxl", "chapman_shaoxing"],
    hf_repo_id="vlbthambawita/ECGBench",
    dry_run=False,
    token=None,               # falls back to env / .env
)
# counts: {"ptbxl": 42, "chapman_shaoxing": 42}
```

Requires the `hf` extra (`pip install ecgbench[hf]`).

## API Reference

### Config
- `load_config(slug)` -- load DatasetConfig from YAML
- `list_available_configs()` -- list dataset slugs with configs

### Catalogue
- `list_datasets()` -- all 64 datasets as CatalogueEntry objects
- `search(query, category, access)` -- filter datasets
- `get_dataset(name)` -- look up by name
- `categories()` -- unique categories
- `to_dataframe()` -- as pandas DataFrame

### Dataset
- `ECGDataset(dataset, split, ...)` -- unified PyTorch Dataset
- `ecg_collate_fn(batch)` -- custom collate for DataLoader
- `WindowOutOfRangeError` -- raised when a `window=` does not fit a record

### Validation
- `validate_dataset(data_path, config)` -- run quality checks
- `generate_report(result, config)` -- generate report dict
- `save_report(result, config, path)` -- save report JSON

### Splitting
- `split_dataset(df, labels, config)` -- generate folds
- `export_splits(split_result, val_result, output_dir, config)` -- write CSVs
- `get_splitter(slug)` -- get dataset-specific splitter

### Croissant
- `generate_croissant(config, splits_dir)` -- generate JSON-LD
- `save_croissant(config, splits_dir)` -- save to file
- `validate_croissant(path)` -- validate JSON-LD

### Download
- `download_dataset(config)` -- download from source
- `resolve_data_path(path, config)` -- resolve or download

### Pipelines (CLI + Python API)
- `run_splits(dataset, ...)` -- full validate + split + Croissant pipeline (same as `ecgbench splits`)
- `run_croissant(dataset, splits_dir, ...)` -- standalone Croissant generation (same as `ecgbench croissant`)
- `run_upload(data_dir, datasets, ...)` -- HuggingFace Hub upload (same as `ecgbench upload`)

## Development

```bash
uv pip install -e ".[dev]"
ruff check ecgbench/
black ecgbench/
pytest
```

## Citation

If you use ECGBench in your research, please cite:

```bibtex
@software{ecgbench,
  author = {Thambawita, Vajira},
  title = {ECGBench: Reproducible ECG Benchmark Datasets},
  url = {https://github.com/vlbthambawita/ECGBench}
}
```

## License

MIT License -- see [LICENSE](LICENSE) for details.
