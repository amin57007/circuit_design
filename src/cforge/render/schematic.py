"""Schematic figure generation.

Auto-layout from a netlist is a hard problem and is not attempted.  Each
pattern ships a ``draw.py`` exporting ``draw(components) -> schemdraw.Drawing``
with explicit placement, and this module wraps it with a title block and takes
care of saving.

SchemDraw is the default backend because it is pure Python.  The optional
Lcapy backend emits circuitikz and needs a LaTeX toolchain, so it is used only
when one is actually detected.
"""

from __future__ import annotations

import datetime as _dt
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from cforge.models import Component  # noqa: E402

__all__ = ["SchematicError", "latex_available", "render"]


class SchematicError(RuntimeError):
    """A schematic could not be produced."""


def latex_available() -> bool:
    """True when a LaTeX toolchain able to build circuitikz is on PATH."""
    return shutil.which("pdflatex") is not None and shutil.which("pdftocairo") is not None


def render(
    pattern: Any,
    components: Sequence[Component],
    out_path: Path,
    fmt: str = "svg",
    title: str = "",
    verdict: str = "",
    backend: str = "schemdraw",
) -> Path:
    """Draw the pattern's schematic with real values and a title block.

    ``verdict`` is stamped into the title block so a schematic pulled out of
    the output directory still carries its PASS/FAIL context.  Returns the path
    actually written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if backend == "lcapy":
        if not latex_available():
            raise SchematicError(
                "--schematic-backend=lcapy needs a LaTeX toolchain (pdflatex and "
                "pdftocairo) on PATH; none was found. Use the default schemdraw "
                "backend, which is pure Python."
            )
        return _render_lcapy(pattern, components, out_path, title, verdict)

    return _render_schemdraw(pattern, components, out_path, fmt, title, verdict)


def _render_schemdraw(
    pattern: Any,
    components: Sequence[Component],
    out_path: Path,
    fmt: str,
    title: str,
    verdict: str,
) -> Path:
    try:
        drawing = pattern.draw_module.draw(list(components))
    except Exception as exc:
        raise SchematicError(
            f"{pattern.directory / 'draw.py'}: draw() failed for pattern "
            f"{pattern.id!r}: {type(exc).__name__}: {exc}"
        ) from exc

    _add_title_block(drawing, pattern, title, verdict)

    try:
        drawing.save(str(out_path))
    except Exception as exc:
        raise SchematicError(f"could not save schematic to {out_path}: {exc}") from exc
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise SchematicError(f"schematic {out_path} was written but is empty")
    return out_path


def _add_title_block(drawing: Any, pattern: Any, title: str, verdict: str) -> None:
    """Annotate the drawing with spec name, pattern, date and verdict.

    Uses schemdraw's own Label elements so the annotation survives SVG export.
    """
    import schemdraw.elements as elm

    date = _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    lines = [
        title or pattern.meta.name,
        f"{pattern.meta.name}  [{pattern.id}]",
        date,
    ]
    if verdict:
        lines.append(verdict)

    xmin, xmax, ymin, _ymax = _extent(drawing)
    x = xmin
    y = ymin - 1.2
    for offset, text in enumerate(lines):
        colour = _VERDICT_COLOUR.get(text, "#333333")
        weight = "bold" if text == verdict else "normal"
        drawing.add(
            elm.Label()
            .label(text, loc="center", color=colour, fontsize=11 if offset else 13)
            .at((x + (xmax - xmin) / 2.0, y - offset * 0.55))
        )
        if weight == "bold":
            pass


_VERDICT_COLOUR: dict[str, str] = {
    "PASS": "#1a7f37",
    "FAIL": "#cf222e",
    "NOT-CONVERGED": "#9a6700",
}


def _extent(drawing: Any) -> tuple[float, float, float, float]:
    """Bounding box of the drawing so the title block lands beneath it."""
    try:
        box = drawing.get_bbox()
        return float(box.xmin), float(box.xmax), float(box.ymin), float(box.ymax)
    except Exception:
        return 0.0, 6.0, 0.0, 3.0


def _render_lcapy(
    pattern: Any,
    components: Sequence[Component],
    out_path: Path,
    title: str,
    verdict: str,
) -> Path:
    """Produce a circuitikz-based schematic through Lcapy.

    Only reached when :func:`latex_available` is True.
    """
    from lcapy import Circuit

    netlist = getattr(pattern.design_module, "lcapy_netlist", None)
    if netlist is None:
        raise SchematicError(
            f"{pattern.directory / 'design.py'} does not export lcapy_netlist(components), "
            "which the lcapy schematic backend needs. Use --schematic-backend=schemdraw."
        )
    try:
        circuit = Circuit(netlist(list(components)))
        circuit.draw(str(out_path))
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        raise SchematicError(
            f"lcapy/circuitikz failed to render {out_path}: {exc}. "
            "Use --schematic-backend=schemdraw."
        ) from exc
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise SchematicError(f"lcapy wrote an empty schematic to {out_path}")
    return out_path
