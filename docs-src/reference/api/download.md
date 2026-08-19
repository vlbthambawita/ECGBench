# Download

`resolve_data_path()` is the single entry point for locating a dataset's files;
it falls back to downloading into `~/.ecgbench/datasets/<slug>/` when no local
path is given.

::: ecgbench.download
