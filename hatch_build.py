"""Hatchling build hook: compile the metadata index into every wheel and sdist.

Registered in ``pyproject.toml`` as ``[tool.hatch.build.hooks.custom]``. Before
the archive is assembled it runs ``ecgbench.metadata.build.build_all()``, so the
distribution carries ``ecgbench/data/metadata.json`` and ``metadata.sqlite``
matching the catalogue and configs at that commit. The only build-time
dependency this adds is ``pyyaml`` (see ``[build-system].requires``).

``metadata.sqlite`` is gitignored (its bytes are not deterministic), so it is
listed under ``[tool.hatch.build].artifacts`` to be included despite that.
"""

from __future__ import annotations

import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class MetadataBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        root = Path(self.root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from ecgbench.metadata.build import build_all

        result = build_all()
        self.app.display_info(
            f"metadata index: {result.sqlite_path.relative_to(root)} "
            f"(fts: {result.fts}, digest {result.content_digest[:19]}…)"
        )
