# Plan: Package CLI + Python API for ECGBench Pipelines

## Goal

Expose the three existing `scripts/` pipelines as first-class features of the
`ecgbench` pip package, usable in two ways:

1. **CLI** — via a single `ecgbench` command with subcommands (installed as a
   console entry point):
   - `ecgbench splits ...` — full pipeline (validate + split + Croissant)
   - `ecgbench croissant ...` — standalone Croissant JSON-LD generation
   - `ecgbench upload ...` — upload fold CSVs to HuggingFace Hub
2. **Python API** — via importable functions so users can run the same
   pipelines from notebooks or downstream code without shelling out.

The `scripts/` directory is removed. Functionality lives inside the installed
package (`ecgbench/`) so it ships with the wheel.

## Non-Goals

- No new features. Flags, defaults, and output paths match today's scripts.
- No click/typer. Keep stdlib `argparse` to avoid a new dependency.
- No breaking change to existing public API (`load_config`, `ECGDataset`,
  `validate_dataset`, `split_dataset`, `export_splits`, `save_croissant`,
  `validate_croissant`, etc.) — those stay as-is.

## Deliverables

### 1. New subpackage: `ecgbench/cli/`

```
ecgbench/cli/
├── __init__.py      # exposes main() dispatcher; re-exports run_* functions
├── _main.py         # argparse root parser + subcommand dispatch
├── splits.py        # run_splits_pipeline() + add_parser() + cli_main()
├── croissant.py     # run_croissant_generation() + add_parser() + cli_main()
└── upload.py        # run_upload_to_huggingface() + add_parser() + cli_main()
```

Each subcommand module follows the same shape:

- `def run_<name>(...) -> <ResultDict | Path | None>` — pure Python API.
  Accepts typed kwargs (no argparse `Namespace`), returns a useful value
  (stats dict, saved path, etc.), and raises on failure. This is what
  notebook/library users call.
- `def add_parser(subparsers) -> None` — registers the subcommand's flags on
  the root parser. Keeps flag definitions colocated with the runner.
- `def cli_main(args: argparse.Namespace) -> int` — thin adapter that
  translates the parsed `Namespace` into a `run_<name>(...)` call, handles
  logging, and returns a POSIX exit code.

`ecgbench/cli/_main.py` builds the root parser, wires in each subcommand via
its `add_parser`, parses args, and dispatches to the matching `cli_main`.
`ecgbench/cli/__init__.py` exposes `main()` (the console entry point) and
re-exports `run_splits_pipeline`, `run_croissant_generation`,
`run_upload_to_huggingface` for programmatic imports.

### 2. Python API surface

Exposed lazily from `ecgbench/__init__.py` (same pattern as the existing
`_LAZY_IMPORTS` dict so base-install users don't pull in torch/hf):

```python
from ecgbench import (
    run_splits_pipeline,         # full validate + split + Croissant
    run_croissant_generation,    # standalone Croissant save
    run_upload_to_huggingface,   # HF Hub upload
)
```

Signatures (aligned with current script flags):

```python
def run_splits_pipeline(
    dataset: str,
    data_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    sampling_rate: int | None = None,
    n_folds: int = 10,
    max_workers: int = 4,
    skip_validation: bool = False,
    skip_croissant: bool = False,
) -> dict:  # stats dict from export_splits + summary fields
    ...

def run_croissant_generation(
    dataset: str,
    splits_dir: Path | str,
    output: Path | str | None = None,
    version: str = "clean",     # "clean" | "original"
    validate: bool = False,
) -> Path:  # path to the saved croissant.json
    ...

def run_upload_to_huggingface(
    data_dir: Path | str,
    datasets: list[str],
    hf_repo_id: str = "vlbthambawita/ECGBench",
    dry_run: bool = False,
    token: str | None = None,   # falls back to env / .env like today
) -> dict:  # per-dataset upload counts
    ...
```

Naming rationale: `run_*` prefix avoids collision with the existing
`generate_croissant` lazy export (which is the low-level JSON-LD builder from
`ecgbench/croissant.py`). Keeps the two layers distinguishable.

### 3. `pyproject.toml` changes

Add a single console script entry point:

```toml
[project.scripts]
ecgbench = "ecgbench.cli:main"
```

No new runtime dependencies. The `upload` subcommand already imports
`huggingface_hub` + `python-dotenv` lazily — those stay under the existing
`[hf]` extra. Calling `ecgbench upload` without `ecgbench[hf]` installed will
raise a clear `ImportError` with an install hint (same message pattern the
scripts use today for `mlcroissant`).

### 4. Remove `scripts/`

Delete:
- `scripts/generate_splits.py`
- `scripts/generate_croissant.py`
- `scripts/upload_to_huggingface.py`

The directory goes away entirely. No deprecation shims — the user explicitly
asked not to keep them. Anyone invoking the old paths will get a clear
"file not found" and can read the updated README/CLAUDE.md.

### 5. Doc updates

- **README.md** — rewrite the "CLI Commands" section to use `ecgbench splits`,
  `ecgbench croissant`, `ecgbench upload`. Add a short Python-API snippet
  showing `run_splits_pipeline(...)` for notebook users.
- **CLAUDE.md** — update the "Common Commands" block to match, and add a
  one-line note that pipelines are also importable (`run_*` helpers).
- **Adding a New Dataset** section — swap `python scripts/generate_splits.py`
  for `ecgbench splits`.

### 6. Tests (`tests/test_cli.py`)

New file with smoke tests only — we are not re-testing the underlying logic
(`test_validation.py`, `test_splitting.py`, `test_croissant.py` already cover
it). Scope:

- `test_cli_help` — `main(["--help"])` exits 0, lists the three subcommands.
- `test_cli_splits_help` / `test_cli_croissant_help` / `test_cli_upload_help`
  — each subcommand's `--help` parses without error.
- `test_splits_pipeline_smoke` — call `run_splits_pipeline` against the
  existing tiny fixture used by `test_export.py` / `test_splitting.py` and
  assert the expected output files exist.
- `test_croissant_generation_smoke` — call `run_croissant_generation` on a
  pre-split fixture directory, assert a `croissant.json` is written.
- Upload is network-bound; cover only the parser and a `--dry-run` path.
  Gate real uploads behind an env var so CI skips them.

## Implementation Order

1. **Scaffold `ecgbench/cli/`** with the four files. `_main.py` and
   `__init__.py` first (empty subcommands), then each subcommand module.
2. **Port `generate_splits.py` → `cli/splits.py`**. Split into
   `run_splits_pipeline()` (pure) + `cli_main()` (argparse adapter). Keep
   the logging, summary print, and flag names identical.
3. **Port `generate_croissant.py` → `cli/croissant.py`** the same way.
4. **Port `upload_to_huggingface.py` → `cli/upload.py`**. Keep `_get_hf_token`
   as a private helper; accept an optional `token=` kwarg in the Python API
   so notebook users can pass one explicitly.
5. **Wire `pyproject.toml`** entry point and reinstall (`uv pip install -e .`)
   to verify `ecgbench --help` works.
6. **Add lazy exports** to `ecgbench/__init__.py` so the `run_*` names are
   importable without eager heavy deps.
7. **Delete `scripts/`** once the new entry point is verified end-to-end on
   at least one dataset (PTB-XL with the existing fixture, or a dry run).
8. **Update README.md and CLAUDE.md.**
9. **Add `tests/test_cli.py`** and run `pytest`.
10. **Run `ruff check ecgbench/` and `black ecgbench/`** before committing.

## Risks / Things to Watch

- **Name collision**: `ecgbench.generate_croissant` (existing, low-level JSON-LD
  builder) vs. the new `run_croissant_generation` (dataset-aware wrapper).
  Using the `run_*` prefix keeps them distinct; worth a one-line comment in
  `__init__.py` so readers don't conflate them.
- **Argparse exit behavior**: subcommand dispatch needs to surface non-zero
  exit codes (e.g., validation failures). `cli_main` returns `int`, and
  `_main.main` passes it to `sys.exit`.
- **Lazy imports in CLI**: the root parser must be buildable without importing
  torch / mlcroissant / huggingface_hub. Heavy imports stay inside each
  `run_*` body (as they already do in the current scripts).
- **HF token loading**: today the script calls `load_dotenv(project_root)`
  where `project_root` is derived from `__file__`. Inside an installed
  package, `__file__` points into `site-packages`, so that line becomes
  useless. Fix: fall back to `load_dotenv()` (cwd) only, and document that
  users must run from a directory containing `.env` or export `HF_TOKEN`.
- **Tests that exercise `scripts/`**: none found in current `tests/`, so
  deletion is safe. Double-check with `grep -r scripts/ tests/` before
  removing.
