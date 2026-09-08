"""Resolution of SPICE device models from the ``models/`` directory.

Search order for a model name:

1. every ``models/*.lib`` other than ``generic.lib``, in sorted order
2. ``models/generic.lib`` (shipped with the tool)
3. otherwise raise :class:`ModelNotFound` naming the exact file to create

A "model" here is either a ``.model`` card or a ``.subckt`` block.  ``resolve``
returns the full text of the definition so it can be pasted into a netlist,
which keeps generated netlists self-contained and runnable by hand.
"""

from __future__ import annotations

import functools
import os
import re
from collections.abc import Iterable
from pathlib import Path

__all__ = ["ModelNotFound", "available_models", "model_dirs", "resolve", "resolve_many"]


class ModelNotFound(LookupError):
    """Raised when a requested SPICE model is in no library on the search path."""


_MODEL_RE = re.compile(r"^\s*\.model\s+(?P<name>\S+)\s", re.IGNORECASE)
_SUBCKT_RE = re.compile(r"^\s*\.subckt\s+(?P<name>\S+)", re.IGNORECASE)
_ENDS_RE = re.compile(r"^\s*\.ends\b", re.IGNORECASE)


def _package_models_dir() -> Path:
    """The ``models/`` directory shipped alongside the repository."""
    return Path(__file__).resolve().parents[3] / "models"


def model_dirs() -> list[Path]:
    """Directories searched for ``.lib`` files, highest priority first.

    ``CFORGE_MODEL_PATH`` (os.pathsep-separated) is prepended when set, so a
    user can override the shipped models without editing the package.
    """
    dirs: list[Path] = []
    env = os.environ.get("CFORGE_MODEL_PATH", "")
    for part in env.split(os.pathsep):
        if part.strip():
            dirs.append(Path(part.strip()).expanduser())
    dirs.append(_package_models_dir())
    return [d for d in dirs if d.is_dir()]


def _lib_files() -> list[Path]:
    """All ``.lib`` files in search order: user libs first, generic.lib last."""
    specific: list[Path] = []
    generic: list[Path] = []
    for directory in model_dirs():
        for path in sorted(directory.glob("*.lib")):
            (generic if path.name == "generic.lib" else specific).append(path)
    return specific + generic


def _definitions_in(path: Path) -> dict[str, str]:
    """Map model/subckt name (lowercased) to its full definition text."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ModelNotFound(f"could not read SPICE model library {path}: {exc}") from exc

    out: dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        subckt = _SUBCKT_RE.match(line)
        if subckt:
            block = [line]
            i += 1
            while i < len(lines):
                block.append(lines[i])
                if _ENDS_RE.match(lines[i]):
                    break
                i += 1
            out[subckt.group("name").lower()] = "\n".join(block)
            i += 1
            continue
        model = _MODEL_RE.match(line)
        if model:
            block = [line]
            j = i + 1
            while j < len(lines) and lines[j].lstrip().startswith("+"):
                block.append(lines[j])
                j += 1
            out[model.group("name").lower()] = "\n".join(block)
            i = j
            continue
        i += 1
    return out


@functools.lru_cache(maxsize=1)
def _index() -> dict[str, tuple[Path, str]]:
    """Name -> (library file, definition text) for everything on the search path."""
    index: dict[str, tuple[Path, str]] = {}
    for path in _lib_files():
        for name, body in _definitions_in(path).items():
            index.setdefault(name, (path, body))
    return index


def invalidate_cache() -> None:
    """Forget the cached library index (used by tests that write new libs)."""
    _index.cache_clear()


def available_models() -> dict[str, Path]:
    """Every resolvable model name mapped to the library that defines it."""
    return {name: path for name, (path, _) in _index().items()}


_DEVICE_LINE_RE = re.compile(r"^\s*[A-Za-z][A-Za-z0-9_]*\s+(?P<rest>.+)$")
_DIRECTIVE_RE = re.compile(r"^\s*[.*+]")


def dependencies(definition: str, exclude: str = "") -> list[str]:
    """Model names referenced by device lines inside ``definition``.

    A macromodel such as ``OPAMP_GENERIC`` instantiates diodes with their own
    ``.model`` cards.  Emitting the subcircuit without those cards produces a
    netlist ngspice rejects with "could not find a valid modelname", so
    :func:`resolve` follows these references automatically.  Making the user
    enumerate a vendor model's internals in pattern.yaml would be a trap.
    """
    known = _index()
    found: list[str] = []
    for raw in definition.splitlines():
        if _DIRECTIVE_RE.match(raw):
            continue
        match = _DEVICE_LINE_RE.match(raw)
        if match is None:
            continue
        for token in match.group("rest").split():
            key = token.strip().lower()
            if key == exclude or key in found or "=" in key:
                continue
            if key in known:
                found.append(key)
    return found


def resolve(model_name: str, _seen: frozenset[str] = frozenset()) -> str:
    """Return the ``.model`` / ``.subckt`` text defining ``model_name``.

    Definitions the model itself instantiates (a macromodel's internal diodes,
    for instance) are prepended, so the returned text is self-contained.

    Raises :class:`ModelNotFound` with the exact file to create when the name
    is not defined in any library on the search path.
    """
    key = model_name.strip().lower()
    if not key:
        raise ModelNotFound("empty model name requested")
    entry = _index().get(key)
    if entry is None:
        searched = _lib_files()
        searched_text = (
            "\n".join(f"    {p}" for p in searched) if searched else "    (no .lib files found)"
        )
        known = ", ".join(sorted(_index())) or "(none)"
        target = _package_models_dir() / f"{key}.lib"
        raise ModelNotFound(
            f"SPICE model {model_name!r} is not defined in any library.\n"
            f"  Searched:\n{searched_text}\n"
            f"  Known models: {known}\n"
            f"  Fix: create {target} containing a '.model {model_name} ...' or "
            f"'.subckt {model_name} ...' definition, or set CFORGE_MODEL_PATH to a "
            f"directory that has one. See models/README.md."
        )
    definition = entry[1]
    blocks: list[str] = []
    for dependency in dependencies(definition, exclude=key):
        if dependency in _seen or dependency == key:
            continue
        blocks.append(resolve(dependency, _seen | {key, dependency}))
    blocks.append(definition)
    return "\n".join(blocks)


def resolve_many(model_names: Iterable[str]) -> str:
    """Concatenate the definitions for several models, de-duplicated, in order.

    De-duplication spans dependencies too, so two macromodels sharing an
    internal diode model emit that ``.model`` card only once; ngspice treats a
    repeated definition as an error.
    """
    emitted: set[str] = set()
    blocks: list[str] = []
    for name in model_names:
        key = name.strip().lower()
        if not key:
            continue
        for needed in [*dependencies(resolve(key), exclude=key), key]:
            if needed in emitted:
                continue
            emitted.add(needed)
            entry = _index().get(needed)
            blocks.append(entry[1] if entry else resolve(needed))
    return "\n".join(blocks)
