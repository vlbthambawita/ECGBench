# Pipelines

The three high-level entry points behind the `ecgbench` CLI. Each is a plain
keyword-argument function, so everything the CLI does is importable:

```python
from ecgbench import run_splits, run_croissant, run_upload
```

See the [CLI page](../cli.md) for the equivalent commands and their flags.

## `run_splits`

::: ecgbench.cli.splits
    options:
      members:
        - run_splits

## `run_croissant`

::: ecgbench.cli.croissant
    options:
      members:
        - run_croissant

## `run_upload`

`run_upload` refuses before any network call for a dataset whose config sets
`publish_fold_csvs: False`.

::: ecgbench.cli.upload
    options:
      members:
        - run_upload
