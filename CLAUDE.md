# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ECGBench is a PyTorch-based library for reproducible ECG benchmark datasets from open-access PhysioNet sources. It provides a `Dataset` class that loads ECG signals in WFDB format with fold-based train/val/test splits. Currently supports PTB-XL; designed to be extended with additional datasets.

## Development Setup

```bash
# Install with dev dependencies (preferred)
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

## Common Commands

```bash
# Lint
ruff check ecgbench/

# Format
black ecgbench/

# Run tests (no formal test suite yet; scripts serve as integration tests)
python scripts/test_ptbxl.py
python scripts/test_hugginface_data.py

# Preprocess PTB-XL into fold CSVs
python scripts/get_ptbxl_fold_csv.py

# Upload datasets to HuggingFace Hub (requires HF_TOKEN in .env)
python scripts/upload_to_huggingface.py
```

## Code Style

- Line length: 100 (both ruff and black)
- Target: Python 3.8+
- Ruff rules: E, F, I, N, W

## Architecture

**Core library** (`ecgbench/`):
- `ECGDataset` — PyTorch `Dataset` that reads WFDB signal files and metadata CSVs. Takes a `physionet_path` (raw data location) and resolves fold CSVs from `ecgbench/datasets/<dataset_name>/<split>/fold_*.csv` relative to the package root. Returns dicts with `signal` tensor (channels, samples) plus metadata fields.
- `ecg_collate_fn` — Custom collate function for `DataLoader` that keeps dicts and strings as lists instead of failing on heterogeneous keys (e.g., `scp_codes`).

**Data pipeline**: Raw PhysioNet CSVs are preprocessed by `scripts/get_ptbxl_fold_csv.py` into per-fold CSVs stored under `ecgbench/datasets/ptbxl/{train,val,test}/`. The fold CSVs are the metadata interface between preprocessing and the Dataset class. The actual WFDB signal files remain at the original PhysioNet path.

**HuggingFace integration**: Preprocessed fold CSVs are uploaded to `deepsynthbody/ECGBench` on HuggingFace Hub via `scripts/upload_to_huggingface.py`. Validation script `scripts/test_hugginface_data.py` verifies uploaded data matches local CSVs via MD5 checksums.

## Website

Static Jekyll site in `docs/` served via GitHub Pages. Single-file `docs/index.html` with dark theme, dataset catalogue (64 ECG datasets), interactive Plotly.js charts, search/filter controls, installation guide, and usage example. No Jekyll build step required (pure HTML/CSS/JS). `docs/_version.json` is written by CI at deploy time.

## CI/CD (GitHub Actions)

- **`.github/workflows/deploy-pages.yml`** — Deploys `docs/` to GitHub Pages on push to `main` (when `docs/` changes). Uses `actions/jekyll-build-pages` + `actions/deploy-pages`.
- **`.github/workflows/publish-pypi.yml`** — Builds and publishes the `ecgbench` pip package to PyPI on `v*` tags using Trusted Publishing (OIDC). Requires a `pypi` environment configured in GitHub repo settings.

## Environment Variables

- `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) — Required for HuggingFace Hub upload/download. Set in `.env` file (see `.env.example`).
