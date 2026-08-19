# Labels

`load_labels()` returns a dataset's label table, dispatching through
`_custom_loaders()` to a per-dataset module in `ecgbench/labels/` where one
exists.

::: ecgbench.labels
    options:
      members:
        - load_labels
        - LabelsUnavailableError
        - LabelSourceMissingError

## Derived datasets

A release whose records belong to another dataset — a feature, annotation or
relabelling layer — gets a label loader but deliberately **no config, splitter
or fold assignment**, because generating folds for it would create a second
ECGBench partition of recordings an existing config already partitions.

PTB-XL+ is the worked case, and a test enforces that no `ptbxl_plus` config
appears. Its loader is not registered in `_custom_loaders()` either, since that
dict is keyed by config slug.

::: ecgbench.labels.ptbxl_plus
