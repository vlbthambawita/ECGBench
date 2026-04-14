# ECGBench

Reproducible ECG benchmark data from open-access datasets with PyTorch integration.

ECGBench provides a curated catalogue of 64 publicly available ECG datasets and a PyTorch `Dataset` class for loading ECG signals with standardised fold-based train/val/test splits.

**Website:** [vlbthambawita.github.io/ECGBench](https://vlbthambawita.github.io/ECGBench/)

## Installation

### Using uv (Recommended)

```bash
uv pip install ecgbench
```

### Using pip

```bash
pip install ecgbench
```

### From source (development)

```bash
git clone https://github.com/vlbthambawita/ECGBench.git
cd ECGBench
uv pip install -e ".[dev]"
```

## Dataset Catalogue

Query the curated index of 64 ECG datasets directly from Python:

```python
import ecgbench

# List all 64 datasets
datasets = ecgbench.list_datasets()
print(f"{len(datasets)} datasets available")

# Search by name, origin, format, or paper
ecgbench.search("PTB-XL")

# Filter by category and access type
ecgbench.search(category="12-Lead (PhysioNet)", access="Open")

# Look up a single dataset by name
ecgbench.get_dataset("MIMIC-IV-ECG")

# List available categories
ecgbench.categories()
# ['12-Lead (PhysioNet)', '12-Lead (Other)', '1-Lead', '2-Lead', '3-Lead', 'BSPM/ECGI']

# Get as a pandas DataFrame
df = ecgbench.to_dataframe()
```

## Loading ECG Signals (PyTorch)

Load ECG benchmark data as a PyTorch Dataset with reproducible fold splits:

```python
from ecgbench import ECGDataset, ecg_collate_fn
from torch.utils.data import DataLoader

# Load PTB-XL training data (100 Hz, all folds)
dataset = ECGDataset(
    physionet_path="/path/to/physionet.org/files/ptb-xl/1.0.3/",
    dataset_name="ptbxl",
    split="train",
    frequency="100",
)

# Use custom collate to handle mixed metadata types (dicts, strings)
loader = DataLoader(dataset, batch_size=32, collate_fn=ecg_collate_fn)

for batch in loader:
    signals = batch["signal"]   # (B, channels, samples)
    ecg_ids = batch["ecg_id"]   # list of IDs
    break
```

### ECGDataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `physionet_path` | `str \| Path` | *required* | Path to PhysioNet dataset root |
| `dataset_name` | `str` | `"ptbxl"` | Dataset name |
| `split` | `str` | `"train"` | `"train"`, `"val"`, or `"test"` |
| `fold_numbers` | `int \| list \| None` | `None` | Fold(s) to load; `None` = all folds |
| `frequency` | `str` | `"100"` | Sampling frequency: `"100"` or `"500"` Hz |
| `ecgbench_root` | `str \| Path \| None` | `None` | Custom metadata CSV path (auto-detected by default) |

### Output Format

Each sample is a dict containing:
- `signal` — float tensor with shape `(channels, samples)`
- `ecg_id` — record identifier
- `patient_id`, `split`, `frequency` — metadata fields
- `scp_codes` — diagnostic codes (kept as dict)
- All other CSV columns as tensors (numeric) or raw values

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Lint
ruff check ecgbench/

# Format
black ecgbench/
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
