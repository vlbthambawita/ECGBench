# ECGBench

Reproducible ECG benchmark datasets with standardised splits, validation, and Croissant metadata.

ECGBench provides a curated catalogue of 64 publicly available ECG datasets, a config-driven pipeline for generating validated fold splits, and a unified PyTorch `Dataset` class for loading any supported dataset.

**Website:** [vlbthambawita.github.io/ECGBench](https://vlbthambawita.github.io/ECGBench/)

## Installation

### Base (config, catalogue, validation, splitting)

```bash
pip install ecgbench
```

### With PyTorch support

```bash
pip install ecgbench[torch]
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
`incartdb`. A dataset that genuinely has none (`mimic_iv_ecg_demo`) raises
`LabelsUnavailableError` naming where labels could come from, rather than
returning empty columns.

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
| `incartdb` | I, II, III, **AVR, AVL, AVF**, V1-V6 (uppercase) |
| `brugada_huca` | I, II, III, aVR, aVL, aVF, V1-V6 |
| `leipzig_heart_center_ecg` | I, II, III, aVR, aVL, aVF, V1-V6, **then 2-8 intracardiac channels in six different orders** |
| `norwegian_athlete_ecg` | I, II, III, **AVR, AVL, AVF**, V1-V6 (uppercase) |
| `mhd_effect_ecg_mri` | I, II, III, aVR, aVL, aVF, V1-V6 — but **14 of 53 records hold only I, II, III** |

`signal[4]` is aVL in most of them and aVF in both MIMIC datasets, so slicing by index across
datasets silently crosses two leads. Matching is case-insensitive — `leads=["aVL"]`
works on the lowercase datasets too — an unknown lead lists what is available, and
a duplicate is rejected.

PTBDB is the one dataset that is not 12-lead: it stores 15 signals, the
conventional twelve plus the three Frank vectorcardiography leads. `leads=` is how
you take the standard twelve out of it. Its records are also **variable length**
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

`challenge2021` is the one dataset where **sampling rate varies per record**
(257/500/1000 Hz), because it concatenates eight source cohorts. Rate is therefore
a label to filter on, not a `sampling_rate=` argument, and record length spans 5 s
to 1800 s so batching needs a `window=` too. It also **contains** PTB-XL,
PTBDB, INCART, CPSC-2018, Chapman-Shaoxing and Ningbo — its `source` label says
which cohort each record came from, and evaluating on any of those after training
on it is testing on training data. See `examples/load_challenge2021.py`.

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
runs 6-144 s and `ptbdb` 32-120 s — so a fixed window can fit most records and
not all.

`window` combines freely with `fold_numbers`, `leads` and `units`; it is applied
first, then lead selection, then units, then `transform`.

### Derived datasets (annotations for another dataset's records)

Some releases contain no recordings of their own — they annotate someone else's.
There are two: **PTB-XL+** (3 feature tables, 2 statement tables, derived median
beats and 283,326 fiducial-point files, all keyed by PTB-XL's `ecg_id`) and
**MIMIC-IV-ECG-Ext-ICD** (ICD-10-CM discharge diagnoses for all 800,035
MIMIC-IV-ECG studies, keyed by `study_id`).

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

### Restricted and credentialed datasets

Most datasets' fold CSVs are published to the [HuggingFace
Hub](https://huggingface.co/datasets/vlbthambawita/ECGBench) and download
automatically. **Some are deliberately not**, and those you generate yourself.

Fold CSVs carry identifiers only — record ID, patient ID, signal path, fold,
split. For an openly licensed source that is uncontroversial. For a
**credentialed or restricted** source those identifiers are still data derived
under a use agreement, and the ECGBench Hub repository is public and ungated, so
ECGBench does not publish them. `mimic_iv_ecg` is the current example: 800,035
`study_id`s and 161,352 `subject_id`s stay with the people who signed the
PhysioNet DUA.

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
