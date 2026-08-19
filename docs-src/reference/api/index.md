# API reference

Generated from the source by [mkdocstrings], so a signature or docstring here is
the one in `ecgbench/` at the commit the site was built from.

## Import surface

Everything below is re-exported from the top-level package:

```python
from ecgbench import ECGDataset, load_config, run_splits
```

Two things about that top level are worth knowing before you read the rest.

!!! note "The package root is half lazy, on purpose"

    `ecgbench/__init__.py` imports the catalogue and config eagerly — they are
    pure-Python and cheap — and resolves everything else through a module-level
    `__getattr__` against `_LAZY_IMPORTS`. So `import ecgbench` never pulls in
    torch, wfdb or mlcroissant; those arrive only when you touch a name that
    needs them.

    A consequence for this reference: the pages below document each name at the
    **module that defines it** (`ecgbench.dataset.ECGDataset`), because a lazy
    attribute has no static definition at the package root to document. Both
    import paths work at runtime.

    Adding a public symbol that needs a heavy dependency means adding it to
    `_LAZY_IMPORTS` *and* `__all__`, never to the imports at the top of the file.

!!! note "Config slugs, not catalogue slugs"

    Every function here that takes a `dataset` argument wants the underscored
    config slug — `ptbxl`, the name of the YAML in `ecgbench/data/configs/` —
    not the dashed catalogue slug `ptb-xl`. See
    [the two namespaces](../../index.md#two-slug-namespaces).

## Where to start

| Doing | Start at |
|---|---|
| Loading records for training | [`ECGDataset`](dataset.md) |
| Looking up what exists | [`catalogue`](catalogue.md) |
| Describing a new dataset | [`DatasetConfig`](config.md) |
| Running the whole pipeline | [`run_splits`](pipelines.md) |
| Checking a split is the canonical one | [`verify_splits`](manifest.md) |

  [mkdocstrings]: https://mkdocstrings.github.io/
