"""Declared label fields: the columns ``load_labels()`` returns, as data.

Every per-dataset module in ``ecgbench/labels/`` describes its output in a
docstring. That prose is excellent and unsearchable. ``FIELDS`` turns it into a
module-level tuple of ``Field`` declarations — name, Frictionless Table Schema
type, description, unit, vocabulary — that the metadata layer indexes (so
``ecgbench search recorder`` finds MIT-BIH), ``ecgbench fields <id>`` prints,
and a consistency test pins to the loader's actual columns so the declaration
cannot rot.

Two rules make the declarations usable by the metadata build, which must not
import pandas (it runs inside the packaging hook with pyyaml alone):

1. ``FIELDS`` is a **literal tuple of ``Field(...)`` calls with constant
   arguments** — strings, numbers, booleans, ``None``, and tuples or lists of
   those. No comprehensions, no references to other names. The build reads it
   with ``ast`` (``declared_fields_from_source``) without executing the module;
   the runtime path (``fields_for``) imports the module and reads the same
   attribute, and ``tests/test_fields.py`` asserts the two agree.
2. ``type`` is a Frictionless Table Schema type — ``string``, ``integer``,
   ``number``, ``boolean``, ``array``, ``object``, ``date``, ``datetime``,
   ``time``, ``duration``, ``any`` — optionally with an item type for arrays,
   ``array[string]``.

Declarative datasets (a ``labels:`` block in YAML with no module) get the same
from ``labels.columns`` plus an optional ``labels.fields:`` block carrying
types and descriptions; see ``_template.yaml``.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig, LabelConfig

#: Frictionless Table Schema field types accepted in ``Field.type``.
BASE_TYPES: tuple[str, ...] = (
    "string",
    "integer",
    "number",
    "boolean",
    "array",
    "object",
    "date",
    "datetime",
    "time",
    "duration",
    "any",
)

_ARRAY_RE = re.compile(r"^array\[([a-z]+)\]$")
_LABELS_DIR = Path(__file__).parent


class FieldDeclarationError(ValueError):
    """A ``FIELDS`` declaration is malformed or not statically readable."""


def validate_type(type_: str) -> None:
    """Raise ``ValueError`` unless ``type_`` is a Table Schema type or ``array[<type>]``."""
    if type_ in BASE_TYPES:
        return
    match = _ARRAY_RE.match(type_)
    if match and match.group(1) in BASE_TYPES and match.group(1) != "array":
        return
    raise ValueError(
        f"field type {type_!r} is not a Frictionless Table Schema type "
        f"({', '.join(BASE_TYPES)}) or array[<type>]"
    )


@dataclass(frozen=True)
class Field:
    """One column of a dataset's label table.

    Attributes:
        name: Column name exactly as ``load_labels()`` returns it.
        type: Frictionless type, or ``array[<item type>]`` for list-valued columns.
        description: What the value means, including any sentinel or encoding.
        unit: Physical unit for measurements (``ms``, ``year``, ``degree``).
        vocabulary: The closed set of values a categorical column takes, as
            strings even for integer codes (``("0", "1")``). ``None`` when open.
        nullable: ``False`` when every record has a value.
        example: One representative value, as text.
        source: ``labels`` for a loader module's output, ``config`` for a
            declarative ``labels:`` block.
    """

    name: str
    type: str
    description: str = ""
    unit: str | None = None
    vocabulary: tuple[str, ...] | None = None
    nullable: bool = True
    example: str | None = None
    source: str = "labels"

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("field name must be a non-empty string")
        validate_type(self.type)
        if self.vocabulary is not None:
            object.__setattr__(self, "vocabulary", tuple(str(v) for v in self.vocabulary))
        if self.source not in ("labels", "config"):
            raise ValueError(f"field source must be 'labels' or 'config', got {self.source!r}")

    @property
    def base_type(self) -> str:
        """``array`` for ``array[string]``, else the type itself."""
        match = _ARRAY_RE.match(self.type)
        return "array" if match else self.type

    @property
    def item_type(self) -> str | None:
        """The item type of an ``array[...]`` field, else ``None``."""
        match = _ARRAY_RE.match(self.type)
        return match.group(1) if match else None

    def to_dict(self) -> dict:
        """JSON-ready mapping (``vocabulary`` as a list)."""
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- declarative


def fields_from_config(labels: LabelConfig | None) -> tuple[Field, ...]:
    """Fields for a declarative ``labels:`` block.

    Every entry of ``labels.columns`` becomes a ``string`` field unless
    ``labels.fields`` refines it; names present only in ``labels.fields`` are
    appended, which is how a block with ``columns: null`` (every column of the
    source CSV) can still enumerate them.
    """
    if labels is None or not labels.available:
        return ()
    specs = labels.fields or {}
    names = list(labels.columns or [])
    names += [n for n in specs if n not in names]
    out = []
    for name in names:
        spec = specs.get(name)
        if spec is None:
            out.append(Field(name, "string", source="config"))
            continue
        out.append(
            Field(
                name,
                spec.type,
                spec.description,
                unit=spec.unit,
                vocabulary=tuple(spec.vocabulary) if spec.vocabulary else None,
                nullable=spec.nullable,
                example=spec.example,
                source="config",
            )
        )
    return tuple(out)


# --------------------------------------------------------------------------- static reading


def has_label_module(slug: str, labels_dir: Path | None = None) -> bool:
    """Whether ``ecgbench/labels/<slug>.py`` exists."""
    return ((labels_dir or _LABELS_DIR) / f"{slug}.py").is_file()


def declared_fields_from_source(
    slug: str, labels_dir: Path | None = None
) -> tuple[Field, ...] | None:
    """Read ``FIELDS`` out of ``ecgbench/labels/<slug>.py`` without importing it.

    Returns ``None`` when the module does not exist or declares no ``FIELDS``.

    Raises:
        FieldDeclarationError: ``FIELDS`` is not a literal tuple of ``Field(...)``
            calls with constant arguments, or a call is invalid.
    """
    path = (labels_dir or _LABELS_DIR) / f"{slug}.py"
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "FIELDS" for t in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "FIELDS"
        ):
            value = node.value
        if value is not None:
            return _eval_fields(value, path)
    return None


def _eval_fields(node: ast.expr, path: Path) -> tuple[Field, ...]:
    if not isinstance(node, (ast.Tuple, ast.List)):
        raise FieldDeclarationError(
            f"{path}:{node.lineno}: FIELDS must be a literal tuple of Field(...) calls, "
            f"found {type(node).__name__}"
        )
    return tuple(_eval_field(elt, path) for elt in node.elts)


def _eval_field(node: ast.expr, path: Path) -> Field:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
        raise FieldDeclarationError(
            f"{path}:{node.lineno}: every FIELDS entry must be a Field(...) call, "
            f"found {type(node).__name__}"
        )
    if node.func.id != "Field":
        raise FieldDeclarationError(
            f"{path}:{node.lineno}: every FIELDS entry must be a Field(...) call, "
            f"found {node.func.id}(...)"
        )
    args = [_literal(a, path) for a in node.args]
    kwargs = {k.arg: _literal(k.value, path) for k in node.keywords if k.arg is not None}
    if any(k.arg is None for k in node.keywords):
        raise FieldDeclarationError(f"{path}:{node.lineno}: **kwargs is not allowed in FIELDS")
    try:
        return Field(*args, **kwargs)
    except (TypeError, ValueError) as exc:
        raise FieldDeclarationError(f"{path}:{node.lineno}: {exc}") from None


def _literal(node: ast.expr, path: Path) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_literal(e, path) for e in node.elts)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -node.operand.value
    raise FieldDeclarationError(
        f"{path}:{node.lineno}: FIELDS arguments must be constants or tuples of constants "
        f"(found {type(node).__name__}); the metadata build reads FIELDS without importing "
        "the module, so comprehensions and name references cannot be used here"
    )


# --------------------------------------------------------------------------- resolution


def fields_for(config: DatasetConfig, *, static: bool = False) -> tuple[Field, ...]:
    """The declared fields of ``config``'s label table.

    A dataset with a module in ``ecgbench/labels/`` answers with that module's
    ``FIELDS`` (empty until one is declared); a declarative dataset answers from
    its ``labels:`` block; a dataset whose labels are unavailable answers empty.

    Args:
        config: The dataset config.
        static: Read ``FIELDS`` from the module source with ``ast`` instead of
            importing the module. This is what the metadata build uses, because
            importing a label module imports pandas.
    """
    if config.labels is not None and not config.labels.available:
        return ()
    if has_label_module(config.slug):
        if static:
            return declared_fields_from_source(config.slug) or ()
        module = importlib.import_module(f"ecgbench.labels.{config.slug}")
        return tuple(getattr(module, "FIELDS", ()))
    return fields_from_config(config.labels)


# --------------------------------------------------------------------------- export


def _cast_vocabulary(field: Field) -> list:
    if field.vocabulary is None:
        return []
    base = field.item_type or field.base_type
    if base == "integer":
        try:
            return [int(v) for v in field.vocabulary]
        except ValueError:
            return list(field.vocabulary)
    if base == "number":
        try:
            return [float(v) for v in field.vocabulary]
        except ValueError:
            return list(field.vocabulary)
    if base == "boolean":
        return [v.lower() in ("true", "1", "yes") for v in field.vocabulary]
    return list(field.vocabulary)


def to_frictionless(fields: tuple[Field, ...], primary_key: str | None = None) -> dict:
    """Render fields as a Frictionless Table Schema (v2) ``dict``.

    ``array[x]`` becomes ``type: array`` with ``arrayItem: {type: x}``; a
    vocabulary becomes ``constraints.enum`` cast to the field's type; a
    non-nullable field gets ``constraints.required``. ``unit`` and ``source`` are
    carried as custom properties, which the spec permits.
    """
    out_fields = []
    for f in fields:
        entry: dict = {"name": f.name, "type": f.base_type}
        if f.item_type is not None:
            entry["arrayItem"] = {"type": f.item_type}
        if f.description:
            entry["description"] = f.description
        if f.unit:
            entry["unit"] = f.unit
        if f.example is not None:
            entry["example"] = f.example
        constraints: dict = {}
        if not f.nullable:
            constraints["required"] = True
        if f.vocabulary is not None:
            constraints["enum"] = _cast_vocabulary(f)
        if constraints:
            entry["constraints"] = constraints
        entry["source"] = f.source
        out_fields.append(entry)
    schema: dict = {"fields": out_fields}
    if primary_key:
        schema["primaryKey"] = primary_key
    return schema
