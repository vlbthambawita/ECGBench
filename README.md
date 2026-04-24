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

### K-fold cross-validation

```python
for k in range(1, 11):
    val_ds = ECGDataset("ptbxl", split="val", fold_numbers=[k], data_path="...")
    test_fold = k % 10 + 1
    test_ds = ECGDataset("ptbxl", split="test", fold_numbers=[test_fold], data_path="...")
    train_folds = [f for f in range(1, 11) if f != k and f != test_fold]
    train_ds = ECGDataset("ptbxl", split="train", fold_numbers=train_folds, data_path="...")
```

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

### Output format

Each sample is a dict:
- `signal` -- float32 tensor `(leads, samples)`
- `record_id` -- record identifier
- `split`, `fold` -- split name and fold number
- All other CSV columns as tensors (numeric) or raw values (str/dict)

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
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset` | str | *required* | Dataset slug (e.g. `ptbxl`, `chapman_shaoxing`) |
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
