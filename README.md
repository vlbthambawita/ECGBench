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

`fold_numbers` picks individual folds out of a split:

```python
ECGDataset("ptbxl", split="train", fold_numbers=[3], data_path="...")       # one fold
ECGDataset("ptbxl", split="train", fold_numbers=[1, 2, 5], data_path="...") # several
```

**A fold only exists in the split it was exported to** — folds 1-8 under `train/`,
9 under `val/`, 10 under `test/`. Asking for fold 9 with `split="train"` is a 404,
so a rotation has to look each fold up in its own split:

```python
from torch.utils.data import ConcatDataset

SPLIT_OF_FOLD = {**{n: "train" for n in range(1, 9)}, 9: "val", 10: "test"}

def folds(slug, numbers, **kw):
    parts = [ECGDataset(slug, split=SPLIT_OF_FOLD[n], fold_numbers=[n], **kw)
             for n in numbers]
    return ConcatDataset(parts) if len(parts) > 1 else parts[0]

# fold 7 as test, fold 10 as val, the other eight as train
test  = folds("ptbxl", [7],  data_path="...")
val   = folds("ptbxl", [10], data_path="...")
train = folds("ptbxl", [n for n in range(1, 11) if n not in (7, 10)], data_path="...")
```

`ConcatDataset` yields the same sample dicts, so `ecg_collate_fn` still works — but
it has no `.metadata_df` or `.labels_df`, so combine those yourself if you need them.

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
eleven automated measurements for `chapman_shaoxing`. A dataset that genuinely has
none (`mimic_iv_ecg_demo`) raises `LabelsUnavailableError` naming where labels
could come from, rather than returning empty columns.

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
| `mimic_iv_ecg_demo` | I, II, III, aVR, **aVF, aVL**, V1-V6 (transposed) |
| `ludb` | i, ii, iii, avr, avl, avf, v1-v6 (**lowercase**) |
| `ptbdb` | i, ii, iii, avr, avl, avf, v1-v6, **vx, vy, vz** (15 signals) |
| `challenge2021` | I, II, III, aVR, aVL, aVF, V1-V6 (identical in all eight cohorts) |
| `incartdb` | I, II, III, **AVR, AVL, AVF**, V1-V6 (uppercase) |

`signal[4]` is aVL in most of them and aVF in MIMIC, so slicing by index across
datasets silently crosses two leads. Matching is case-insensitive — `leads=["aVL"]`
works on the lowercase datasets too — an unknown lead lists what is available, and
a duplicate is rejected.

PTBDB is the one dataset that is not 12-lead: it stores 15 signals, the
conventional twelve plus the three Frank vectorcardiography leads. `leads=` is how
you take the standard twelve out of it. Its records are also **variable length**
(32 s to 120 s), so batching needs a cropping `transform` — see
`examples/load_ptbdb.py`.

`incartdb` is the one dataset whose primary labels are **reference beat
annotations** rather than record-level diagnoses: 175,907 manually corrected beats
over ten types, exposed as per-record counts (`beat_N`, `beat_V`, …, `pvc_fraction`)
alongside the patient diagnosis and free-text per-record findings. Its records are
1800 s (~44 MB each), so batching needs a cropping `transform`. It is also the
clearest case for **patient-grouped folds** — 3,166 of its 3,174 RBBB beats come
from a single patient — see `examples/load_incartdb.py`.

`challenge2021` is the one dataset where **sampling rate varies per record**
(257/500/1000 Hz), because it concatenates eight source cohorts. Rate is therefore
a label to filter on, not a `sampling_rate=` argument, and record length spans 5 s
to 1800 s so batching needs a cropping `transform` too. It also **contains** PTB-XL,
PTBDB, INCART, CPSC-2018, Chapman-Shaoxing and Ningbo — its `source` label says
which cohort each record came from, and evaluating on any of those after training
on it is testing on training data. See `examples/load_challenge2021.py`.

Both are **read-time adapters**: they shape the returned tensor only. Source files,
fold CSVs and validation are untouched — a record excluded for a flat V6 stays
excluded even if you never load V6.

### ECGDataset parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `str \| DatasetConfig` | *required* | Dataset slug or config object |
| `split` | `str` | `"train"` | `"train"`, `"val"`, or `"test"` |
| `version` | `str` | `"clean"` | `"clean"` or `"original"` |
| `data_path` | `Path \| str \| None` | `None` | Path to signal files; auto-downloads if None |
| `sampling_rate` | `int \| None` | `None` | Sampling rate (default: dataset's default) |
| `fold_numbers` | `list[int] \| None` | `None` | Specific folds to load; None = all |
| `transform` | `Callable \| None` | `None` | Transform applied to signal tensor |
| `metadata_source` | `str` | `"hf"` | `"hf"` (HuggingFace) or `"local"` |
| `labels` | `bool` | `False` | Attach per-record labels as `sample["labels"]`; needs local source data |
| `leads` | `list[str] \| None` | `None` | Select and reorder leads by name, e.g. `["I", "II", "V5"]` |
| `units` | `str` | `"mV"` | `"mV"` or `"uV"` — applied before `transform` |

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
