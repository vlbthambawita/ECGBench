# Dataset

`ecgbench.dataset` — the single PyTorch `Dataset` every supported dataset loads
through, plus the collate function and the two errors it raises.

The read-time adapters (`window=`, `leads=`, `units=`, `transform=`) shape the
returned tensor only, in that order. They never touch the source files, the
exported fold CSVs, or validation — which reads whole records through its own
window-less copy of `_load_signal` in `ecgbench/validation/engine.py`.

::: ecgbench.dataset
    options:
      # __getitem__ is the class's whole interface, so pull it back past the
      # default "hide everything underscored" filter.
      filters: ["!^_", "^__getitem__$"]
      members:
        - ECGDataset
        - ecg_collate_fn
        - WindowOutOfRangeError
        - SplitsNotPublishedError
        - UnitConversionError
