"""Resolve any name a dataset goes by to its ``dataset_id``.

A dataset has up to four names: the dashed catalogue slug (``ptb-xl``), the
underscored config slug (``ptbxl``), the catalogue display name (``PTB-XL``) and
the config's own name where it differs. ``DatasetMeta.aliases`` lists them; this
module turns that list into a case-insensitive lookup, with close-match hints on
a miss so a typo is answered with the intended id rather than a bare error.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable

from ecgbench.metadata.model import DatasetMeta


class UnknownDatasetError(KeyError):
    """Raised when a key matches no alias of any dataset.

    ``close_matches`` holds up to three aliases that look like the key, so the
    message — and a CLI exit — can name what was probably meant.
    """

    def __init__(self, key: str, close_matches: tuple[str, ...] = ()):
        self.key = key
        self.close_matches = close_matches
        hint = ""
        if close_matches:
            hint = "; did you mean " + ", ".join(repr(m) for m in close_matches) + "?"
        super().__init__(f"unknown dataset {key!r}{hint}")

    def __str__(self) -> str:  # KeyError.__str__ would wrap the message in quotes
        return self.args[0]


class AliasIndex:
    """Case-insensitive alias → ``dataset_id`` table over a model.

    Built once per model; ``resolve`` is a dict lookup. Two datasets claiming the
    same alias (ignoring case) is a build error, reported with both ids.
    """

    def __init__(self, model: Iterable[DatasetMeta]):
        self._by_alias: dict[str, str] = {}
        self._display: dict[str, str] = {}
        collisions: list[str] = []
        for meta in model:
            for alias in (meta.dataset_id, *meta.aliases):
                folded = alias.casefold()
                owner = self._by_alias.get(folded)
                if owner is not None and owner != meta.dataset_id:
                    collisions.append(f"{alias!r} claimed by {owner} and {meta.dataset_id}")
                    continue
                self._by_alias[folded] = meta.dataset_id
                self._display.setdefault(folded, alias)
        if collisions:
            raise ValueError("alias collisions: " + "; ".join(collisions))

    def resolve(self, key: str) -> str:
        """Return the ``dataset_id`` for ``key`` or raise ``UnknownDatasetError``."""
        folded = key.strip().casefold()
        try:
            return self._by_alias[folded]
        except KeyError:
            close = difflib.get_close_matches(folded, list(self._by_alias), n=3, cutoff=0.6)
            raise UnknownDatasetError(key, tuple(self._display[c] for c in close)) from None

    def __contains__(self, key: str) -> bool:
        return key.strip().casefold() in self._by_alias

    def __len__(self) -> int:
        return len(self._by_alias)


def resolve(key: str, model: Iterable[DatasetMeta]) -> str:
    """One-shot ``AliasIndex(model).resolve(key)``.

    Prefer holding an ``AliasIndex`` (or a ``MetadataStore``, which owns one)
    when resolving more than a single key.
    """
    return AliasIndex(model).resolve(key)
