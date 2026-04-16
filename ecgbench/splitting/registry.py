"""
Splitter registry with automatic GenericSplitter fallback.
"""

from __future__ import annotations

from ecgbench.splitting.base import DatasetSplitter

_REGISTRY: dict[str, type[DatasetSplitter]] = {}


def register(slug: str):
    """Decorator to register a splitter class for a dataset slug."""

    def wrapper(cls: type[DatasetSplitter]):
        _REGISTRY[slug] = cls
        return cls

    return wrapper


def get_splitter(dataset_slug: str) -> DatasetSplitter:
    """Get the splitter for a dataset. Falls back to GenericSplitter."""
    from ecgbench.splitting.strategies.generic import GenericSplitter

    cls = _REGISTRY.get(dataset_slug, GenericSplitter)
    return cls()
