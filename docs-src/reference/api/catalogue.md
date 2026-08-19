# Catalogue

`ecgbench.catalogue` — the 64 surveyed datasets, read from the Markdown front
matter in `docs/_datasets/` (shipped in the wheel as `ecgbench/_datasets/`).
Pure Python, no heavy dependencies, cached with `functools.cache`.

Remember that a catalogue entry is a *description*: it does not imply a config
exists, and `status:` is not a reliable signal of that either.

::: ecgbench.catalogue
