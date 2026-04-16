# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ECGBench is a config-driven library for reproducible ECG benchmark datasets. It has four major subsystems:

1. **Config system** — every dataset described by a single YAML file (`ecgbench/data/configs/`). Adding a dataset requires zero Python for standard cases.
2. **Validation engine** — pre-validates every ECG record, producing `original` (all records + quality flags) and `clean` (valid only) versions.
3. **Splitting framework** — strategy-pattern splitters producing deterministic 10-fold stratified splits with patient-level grouping. PTB-XL, Chapman-Shaoxing, and a generic config-driven fallback.
4. **Croissant metadata** — programmatic MLCommons Croissant 1.1 JSON-LD generation and validation via `mlcroissant`.

Plus a **catalogue** of 64 ECG datasets (pure Python, no heavy deps), a unified **PyTorch Dataset**, auto-download, and HuggingFace Hub integration.

## Development Setup

```bash
uv pip install -e ".[dev]"
```

## Common Commands

```bash
# Lint & format
ruff check ecgbench/
black ecgbench/

# Tests
pytest
pytest tests/test_config.py -v          # single module
pytest -k "test_split" -v               # by name pattern

# Full pipeline: validate + split + Croissant
python scripts/generate_splits.py --dataset ptbxl --data-path /path/to/ptb-xl/1.0.3/

# Standalone Croissant generation
python scripts/generate_croissant.py --dataset ptbxl --splits-dir output/ptbxl/clean/ --version clean

# Upload to HuggingFace Hub (requires HF_TOKEN in .env)
python scripts/upload_to_huggingface.py --data-dir output/ --datasets ptbxl
```

## Code Style

- Python 3.10+ — use modern type hints (`str | None`, `list[int]`, no `typing` imports for builtins)
- Line length: 100 (both ruff and black)
- Ruff rules: E, F, I, N, W
- Use `dataclasses` over plain dicts for structured data
- Use `pathlib.Path` everywhere, never raw string paths

## Architecture

### Config System (`ecgbench/config.py` + `ecgbench/data/configs/`)
`DatasetConfig` dataclass is the typed representation of a YAML config. All modules accept `DatasetConfig`, never raw dicts. `load_config(slug)` parses YAML and validates required fields. Nested dataclasses: `CreatorInfo`, `StratificationConfig`, `ValidationConfig`, `PredefinedSplitConfig`, `CroissantConfig`.

### Catalogue (`ecgbench/catalogue.py`)
Loads `ecgbench/data/ecg_datasets.csv` with `functools.cache`. Returns `CatalogueEntry` dataclass instances. No heavy deps — always importable.

### Validation (`ecgbench/validation/`)
- `checks.py` — individual check functions (`check_missing_leads`, `check_nan_values`, `check_truncated_signal`, `check_flat_line`, `check_corrupt_header`, `check_amplitude_outlier`) registered in `CHECK_REGISTRY`.
- `engine.py` — `validate_dataset()` runs checks in parallel via `ProcessPoolExecutor`, returns `ValidationResult` with `original_df` (all records + `is_valid`/`quality_issues` columns) and `clean_df`.
- `report.py` — generates `validation_report.json`.

### Splitting (`ecgbench/splitting/`)
- `base.py` — `DatasetSplitter` ABC + `SplitResult` dataclass (with `.train`, `.val`, `.test` properties and `get_kfold_split()`).
- `engine.py` — `split_dataset()` dispatches to `StratifiedGroupKFold` (patient-aware) or `StratifiedKFold`, or reads predefined splits. Folds are 1-indexed.
- `strategies/` — `@register("slug")` decorated splitters. `PTBXLSplitter` (SCP superclass mapping), `ChapmanSplitter`, `GenericSplitter` (config-driven fallback).
- `export.py` — writes `original/` and `clean/` fold CSVs with **minimal columns only** (record ID, patient ID, signal paths, fold, split). Full metadata stays in the original dataset CSV.
- `registry.py` — splitter lookup with `GenericSplitter` fallback.

### Croissant (`ecgbench/croissant.py`)
Generates Croissant 1.1 JSON-LD using `mlcroissant` (optional dep, lazy import). Includes SHA-256 hashes for all CSVs.

### Download (`ecgbench/download.py`)
`resolve_data_path()` is the single entry point for locating dataset files. Auto-downloads to `~/.ecgbench/datasets/<slug>/` if no local path given.

### Dataset (`ecgbench/dataset.py`)
Single `ECGDataset` class loading any dataset via config. `metadata_source="hf"` (default) downloads fold CSVs from HuggingFace Hub; `"local"` reads from disk. `ecg_collate_fn` handles heterogeneous batches.

### Public API (`ecgbench/__init__.py`)
Catalogue and config imports are eager (lightweight). Everything else (`ECGDataset`, validation, splitting, croissant, download) is lazy-imported via `__getattr__` so `import ecgbench` doesn't pull in torch/wfdb/mlcroissant.

## Adding a New Dataset

1. Copy `ecgbench/data/configs/_template.yaml` to `<slug>.yaml`, fill in fields
2. Run `python scripts/generate_splits.py --dataset <slug> --data-path /path/to/data/`
3. If custom logic needed, create `ecgbench/splitting/strategies/<slug>.py` with `@register("<slug>")`
4. Run `pytest`

## Versioning & Release

Version derived from git tags via `hatch-vcs`. Push a `v*` tag to trigger PyPI publish (Trusted Publishing) and HF Space deploy.

## CI/CD (GitHub Actions)

- `deploy-pages.yml` — GitHub Pages deploy of `docs/` on push to `main`
- `deploy-hf-space.yml` — HF Space deploy on push to `main` or `v*` tags
- `publish-pypi.yml` — PyPI publish on `v*` tags via Trusted Publishing

## Environment Variables

- `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) — for HuggingFace Hub upload/download. Set in `.env` (see `.env.example`).
