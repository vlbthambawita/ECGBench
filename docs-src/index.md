# ECGBench

Reproducible ECG benchmark datasets with standardised splits, validation, and
Croissant metadata.

ECGBench is two things at once: a **catalogue** of 64 publicly available ECG
datasets, and a **config-driven pipeline** that turns 52 of them into validated,
deterministic 10-fold splits behind a single PyTorch `Dataset` class.

<div class="grid cards" markdown>

-   :material-database: **Dataset catalogue**

    ---

    Every dataset ECGBench knows about — access terms, formats, lead layouts,
    record counts — with a page each.

    [:octicons-arrow-right-24: Browse the catalogue](../)

-   :material-sitemap: **Architecture**

    ---

    End-to-end flow from a YAML config to a fold CSV, in diagrams whose every
    box names a real function.

    [:octicons-arrow-right-24: How it fits together](architecture.md)

-   :material-plus-box: **Adding a dataset**

    ---

    The authoritative per-dataset checklist, including the silent-failure traps
    that a green test run will not catch.

    [:octicons-arrow-right-24: The checklist](guides/adding-a-dataset.md)

-   :material-console: **CLI**

    ---

    `ecgbench splits`, `croissant` and `upload`, and the Python API behind each.

    [:octicons-arrow-right-24: Command reference](reference/cli.md)

</div>

## Installation

--8<-- "README.md:install"

## Quick start

```python
from ecgbench import ECGDataset, ecg_collate_fn
from torch.utils.data import DataLoader

# Fold CSVs download from the HuggingFace Hub; the waveforms come from disk
# (or download on first use).
train = ECGDataset(dataset="ptbxl", split="train", version="clean")
loader = DataLoader(train, batch_size=32, collate_fn=ecg_collate_fn)

batch = next(iter(loader))
batch["signal"].shape  # (32, 12, 5000) — leads x samples, millivolts
```

The [README](https://github.com/vlbthambawita/ECGBench/blob/main/README.md)
remains the long-form tour of the loader: label handling, lead and unit
selection, sample windows, and the per-dataset notes.

## Two slug namespaces

The single most common source of confusion, so it is worth stating up front.

| | Catalogue | Implemented dataset |
|---|---|---|
| Slug style | dashed — `ptb-xl` | underscored — `ptbxl` |
| Lives in | `docs/_datasets/<slug>.md` | `ecgbench/data/configs/<slug>.yaml` |
| Count | 64 | 52 |
| Gives you | a description | validation, splits, a loader |

They do **not** map mechanically onto one another: two catalogue entries
(`chapman-shaoxing-arrhythmia`, `chapman-shaoxing-ecg-database-10-646-patients`)
are both served by the one `chapman_shaoxing` config. A dataset's `status:` in
the catalogue is likewise not a reliable signal of whether it runs — check
`ecgbench/data/configs/` for that.
