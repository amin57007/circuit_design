"""Loading and validation of pattern directories.

A pattern is a self-contained directory describing one topology:

    patterns/<id>/
      pattern.yaml     metadata, applicability ranges, gotchas
      netlist.cir.j2   Jinja2 -> SPICE subcircuit / topology
      tb.cir.j2        Jinja2 -> testbench with sources, analyses and .meas
      design.py        solve(spec) -> dict[ref, ideal_value]  (SI base units)
      draw.py          draw(components) -> schemdraw.Drawing
      notes.md         rationale, readable by a human or an LLM

Every one of those files is required.  Loading validates all of them up front
and fails loudly naming the offending path, because a pattern that is broken
at simulation time wastes far more of the user's attention than one that is
rejected at startup.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined, TemplateSyntaxError
from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "REQUIRED_FILES",
    "Applicability",
    "Gotcha",
    "Pattern",
    "PatternError",
    "PatternMeta",
    "default_pattern_root",
    "load_all",
    "load_pattern",
]

REQUIRED_FILES: tuple[str, ...] = (
    "pattern.yaml",
    "netlist.cir.j2",
    "tb.cir.j2",
    "design.py",
    "draw.py",
    "notes.md",
)


class PatternError(RuntimeError):
    """A pattern directory is missing, incomplete or invalid."""


class Gotcha(BaseModel):
    """A known failure mode of a topology, optionally machine-checkable.

    ``check`` names a callable ``check_<id>(spec, components, context)`` exported
    by the pattern's ``design.py``; it returns ``None`` when fine or a warning
    string when the gotcha is triggered.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    check: str | None = None


class Applicability(BaseModel):
    """Ranges within which this pattern is a sensible choice.

    Every numeric entry is an inclusive ``[lo, hi]`` pair in SI base units.
    ``extra="allow"`` is deliberate: a pattern may constrain any spec param by
    name (``fc_hz``, ``iout_a``, ``order``, ...) without this class enumerating
    them.  Non-range keys such as ``gain`` are treated as string tags.
    """

    model_config = ConfigDict(extra="allow")

    def ranges(self) -> dict[str, tuple[float, float]]:
        """The inclusive numeric ranges declared here, keyed by spec param name."""
        out: dict[str, tuple[float, float]] = {}
        for key, value in self.__pydantic_extra__.items() if self.__pydantic_extra__ else []:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                try:
                    lo, hi = float(value[0]), float(value[1])
                except (TypeError, ValueError):
                    continue
                out[key] = (lo, hi) if lo <= hi else (hi, lo)
        return out

    def scalars(self) -> dict[str, float]:
        """Single-number constraints such as ``q_max: 3.0``."""
        out: dict[str, float] = {}
        for key, value in self.__pydantic_extra__.items() if self.__pydantic_extra__ else []:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key] = float(value)
        return out

    def tags(self) -> dict[str, str]:
        """String constraints such as ``gain: unity_or_noninverting``."""
        items = self.__pydantic_extra__.items() if self.__pydantic_extra__ else []
        return {k: str(v) for k, v in items if isinstance(v, str)}


class Tradeoffs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)


class PatternMeta(BaseModel):
    """Parsed ``pattern.yaml``."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    block: str = Field(min_length=1)
    requires: list[str] = Field(default_factory=list)
    applicability: Applicability = Field(default_factory=Applicability)
    tradeoffs: Tradeoffs = Field(default_factory=Tradeoffs)
    gotchas: list[Gotcha] = Field(default_factory=list)
    provides_meas: list[str] = Field(default_factory=list)
    numeric_refine: bool = False
    models: list[str] = Field(default_factory=list)
    description: str = ""

    @field_validator("id")
    @classmethod
    def _id_is_directory_safe(cls, v: str) -> str:
        v = v.strip()
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"pattern id {v!r} must contain only letters, digits, '_' and '-' "
                "because it names a directory and a CLI argument"
            )
        return v


@dataclass(frozen=True)
class Pattern:
    """A validated, loaded pattern, ready to synthesize and simulate with."""

    meta: PatternMeta
    directory: Path
    netlist_template: str
    testbench_template: str
    notes: str
    design_module: ModuleType
    draw_module: ModuleType
    _env: Environment = field(repr=False, default_factory=lambda: _jinja_env())

    @property
    def id(self) -> str:
        return self.meta.id

    @property
    def block(self) -> str:
        return self.meta.block

    def render_netlist(self, context: dict[str, Any]) -> str:
        """Render ``netlist.cir.j2``.  Missing variables are an error, not blank."""
        return self._render(self.netlist_template, context, "netlist.cir.j2")

    def render_testbench(self, context: dict[str, Any]) -> str:
        """Render ``tb.cir.j2``."""
        return self._render(self.testbench_template, context, "tb.cir.j2")

    def _render(self, source: str, context: dict[str, Any], what: str) -> str:
        try:
            return self._env.from_string(source).render(**context)
        except Exception as exc:
            raise PatternError(
                f"failed to render {self.directory / what} for pattern {self.id!r}: {exc}"
            ) from exc

    def solve(self, spec: Any) -> dict[str, float]:
        """Call the pattern's ``design.solve(spec)``; values are SI base units."""
        solve: Callable[[Any], dict[str, float]] = self.design_module.solve
        result = solve(spec)
        if not isinstance(result, dict) or not result:
            raise PatternError(
                f"{self.directory / 'design.py'}: solve(spec) must return a non-empty "
                f"dict mapping component ref -> ideal value, got {type(result).__name__}"
            )
        return {str(k): float(v) for k, v in result.items()}

    def gotcha_check(self, name: str) -> Callable[..., str | None] | None:
        """Look up a gotcha check function exported by ``design.py``."""
        fn = getattr(self.design_module, f"check_{name}", None) or getattr(
            self.design_module, name, None
        )
        return fn if callable(fn) else None


def _jinja_env() -> Environment:
    """Jinja environment tuned for SPICE netlists.

    ``StrictUndefined`` turns a typo in a template variable into an immediate
    error rather than a netlist with a blank component value, which ngspice
    would then misparse in a confusing way.
    """
    env = Environment(
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )
    env.filters["eng"] = _eng_filter
    env.filters["spice"] = lambda v: f"{float(v):.6g}"
    return env


def _eng_filter(value: float, digits: int = 3) -> str:
    """Format a number in SPICE-friendly engineering notation (display only)."""
    from cforge.render import format_eng

    return format_eng(float(value), digits=digits)


def default_pattern_root() -> Path:
    """The ``patterns/`` directory shipped with the repository."""
    return Path(__file__).resolve().parents[3] / "patterns"


def _load_python_module(path: Path, pattern_id: str, kind: str) -> ModuleType:
    """Import ``path`` as a uniquely-named module.

    Patterns are user-authored Python; a syntax error there must name the file.
    """
    module_name = f"cforge_patterns.{pattern_id}.{kind}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PatternError(f"cannot import {path} for pattern {pattern_id!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        del sys.modules[module_name]
        raise PatternError(
            f"error importing {path} for pattern {pattern_id!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return module


def load_pattern(directory: Path) -> Pattern:
    """Load and fully validate one pattern directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise PatternError(f"pattern directory does not exist: {directory}")

    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        raise PatternError(
            f"pattern {directory.name!r} at {directory} is incomplete; missing: "
            + ", ".join(missing)
            + f".  Every pattern needs all of: {', '.join(REQUIRED_FILES)}"
        )

    yaml_path = directory / "pattern.yaml"
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PatternError(f"{yaml_path} is not valid YAML: {exc}") from exc
    except OSError as exc:
        raise PatternError(f"could not read {yaml_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PatternError(
            f"{yaml_path} must contain a YAML mapping at the top level, "
            f"got {type(raw).__name__}"
        )

    try:
        meta = PatternMeta.model_validate(raw)
    except Exception as exc:
        raise PatternError(f"{yaml_path} failed validation: {exc}") from exc

    if meta.id != directory.name:
        raise PatternError(
            f"{yaml_path}: id {meta.id!r} does not match its directory name "
            f"{directory.name!r}; they must be identical so --topology can find it"
        )

    env = _jinja_env()
    templates: dict[str, str] = {}
    for name in ("netlist.cir.j2", "tb.cir.j2"):
        text = (directory / name).read_text(encoding="utf-8")
        try:
            env.parse(text, filename=str(directory / name))
        except TemplateSyntaxError as exc:
            raise PatternError(
                f"{directory / name} line {exc.lineno}: Jinja2 syntax error: {exc.message}"
            ) from exc
        templates[name] = text

    design_module = _load_python_module(directory / "design.py", meta.id, "design")
    if not hasattr(design_module, "solve"):
        raise PatternError(
            f"{directory / 'design.py'} must define solve(spec) -> dict[ref, ideal_value]"
        )
    draw_module = _load_python_module(directory / "draw.py", meta.id, "draw")
    if not hasattr(draw_module, "draw"):
        raise PatternError(
            f"{directory / 'draw.py'} must define draw(components) -> schemdraw.Drawing"
        )

    for gotcha in meta.gotchas:
        if gotcha.check and not (
            hasattr(design_module, f"check_{gotcha.check}")
            or hasattr(design_module, gotcha.check)
        ):
            raise PatternError(
                f"{yaml_path}: gotcha {gotcha.id!r} declares check {gotcha.check!r} but "
                f"{directory / 'design.py'} exports neither check_{gotcha.check}() "
                f"nor {gotcha.check}()"
            )

    return Pattern(
        meta=meta,
        directory=directory,
        netlist_template=templates["netlist.cir.j2"],
        testbench_template=templates["tb.cir.j2"],
        notes=(directory / "notes.md").read_text(encoding="utf-8"),
        design_module=design_module,
        draw_module=draw_module,
        _env=env,
    )


def _pattern_dirs(root: Path) -> Iterator[Path]:
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith((".", "_")):
            yield child


def load_all(root: Path | None = None) -> list[Pattern]:
    """Load every pattern under ``root`` (default: the shipped ``patterns/``).

    Raises :class:`PatternError` on the first invalid pattern; a silently
    skipped pattern is a debugging trap.
    """
    root = Path(root) if root is not None else default_pattern_root()
    if not root.is_dir():
        raise PatternError(
            f"pattern root {root} does not exist; expected the repository's patterns/ "
            "directory, or pass --patterns to point at your own"
        )
    patterns = [load_pattern(d) for d in _pattern_dirs(root)]
    _check_unique_ids(patterns)
    return patterns


def _check_unique_ids(patterns: Sequence[Pattern]) -> None:
    seen: dict[str, Path] = {}
    for pattern in patterns:
        if pattern.id in seen:
            raise PatternError(
                f"duplicate pattern id {pattern.id!r} in {seen[pattern.id]} "
                f"and {pattern.directory}"
            )
        seen[pattern.id] = pattern.directory
