"""HTML report generation.

The report is a single self-contained file: SVGs are inlined and PNGs are
base64-encoded, so it can be emailed, attached to a ticket or archived without
losing its figures.

:class:`~cforge.models.DesignResult` is the only input.  Nothing is recomputed
here; if a number is in the report it came from ngspice or from the
deterministic synthesis stage.
"""

from __future__ import annotations

import base64
import datetime as _dt
import html
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from cforge import __version__
from cforge.models import DesignResult, Requirement
from cforge.render import format_eng, format_quantity

__all__ = ["ReportError", "default_template_dir", "render_report"]


class ReportError(RuntimeError):
    """The report could not be generated."""


def default_template_dir() -> Path:
    """The ``templates/`` directory shipped with the repository."""
    return Path(__file__).resolve().parents[2] / "templates"


def _inline_svg(path: Path) -> str:
    """Return the SVG markup with its XML prolog stripped, ready to embed."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReportError(f"could not read schematic {path} for inlining: {exc}") from exc
    text = re.sub(r"<\?xml[^>]*\?>", "", text)
    text = re.sub(r"<!DOCTYPE[^>]*>", "", text)
    return text.strip()


def _data_uri(path: Path) -> str:
    """Base64 ``data:`` URI for a PNG so the report needs no sidecar files."""
    try:
        blob = path.read_bytes()
    except OSError as exc:
        raise ReportError(f"could not read figure {path} for embedding: {exc}") from exc
    return "data:image/png;base64," + base64.b64encode(blob).decode("ascii")


def _requirement_by_id(design: DesignResult, requirement_id: str) -> Requirement | None:
    for req in design.spec.requirements:
        if req.id == requirement_id:
            return req
    return None


def _bom_rows(design: DesignResult) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for component in design.components:
        rows.append(
            {
                "ref": component.ref,
                "ideal": format_quantity(component.ideal_value, component.unit, 4),
                "value": format_quantity(component.value, component.unit, 4),
                "eseries": component.eseries or "-",
                "tolerance": f"{component.tolerance_percent:g}%"
                if component.tolerance_percent
                else "-",
                "deviation": f"{component.deviation_percent:+.2f}%",
                "model": component.model or "-",
            }
        )
    return rows


def _traceability_rows(design: DesignResult) -> list[dict[str, Any]]:
    """One row per requirement, in spec order.  Exactly one row per requirement."""
    by_id = {r.requirement_id: r for r in design.requirement_results}
    rows: list[dict[str, Any]] = []
    for req in design.spec.requirements:
        result = by_id.get(req.id)
        value_text = "-"
        if result is not None and result.value is not None:
            value_text = f"{result.value:.6g} {req.unit}".strip()
        mc_text = "-"
        if result is not None and result.mc_pass_rate is not None:
            mc_text = f"{100.0 * result.mc_pass_rate:.1f}%"
        rows.append(
            {
                "id": req.id,
                "description": req.description or req.meas,
                "meas": req.meas,
                "limit": req.limit_text(),
                "value": value_text,
                "margin": (
                    f"{result.margin_percent:+.1f}%"
                    if result is not None and result.margin_percent is not None
                    else "-"
                ),
                "mc": mc_text,
                "mc_warn": (
                    result is not None
                    and result.mc_pass_rate is not None
                    and result.mc_pass_rate < 0.99
                ),
                "status": result.status if result is not None else "NO-DATA",
            }
        )
    return rows


def _sensitivity_rows(design: DesignResult) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in design.sensitivity:
        rows.append(
            {
                "meas": item.meas,
                "nominal": f"{item.nominal:.6g}",
                "p5": f"{item.p5:.6g}",
                "p50": f"{item.p50:.6g}",
                "p95": f"{item.p95:.6g}",
                "worst": f"{item.worst:.6g}",
                "pass_rate": f"{100.0 * item.pass_rate:.1f}%",
                "n": str(item.n_samples),
                "warn": "yes" if item.pass_rate < 0.99 else "",
            }
        )
    return rows


def _figures(design: DesignResult, out_dir: Path) -> list[dict[str, str]]:
    """Embed every PNG artifact, in a stable, readable order."""
    order = ["bode", "transient", "dc_sweep"]

    def sort_key(item: tuple[str, str]) -> tuple[int, str]:
        key = item[0]
        for index, prefix in enumerate(order):
            if key.startswith(prefix):
                return (index, key)
        return (len(order), key)

    figures: list[dict[str, str]] = []
    for key, relative in sorted(design.artifacts.items(), key=sort_key):
        if not relative.lower().endswith(".png"):
            continue
        path = (out_dir / Path(relative).name).resolve()
        if not path.is_file():
            path = Path(relative)
        if not path.is_file():
            continue
        figures.append(
            {"key": key, "title": _figure_title(key), "src": _data_uri(path)}
        )
    return figures


def _figure_title(key: str) -> str:
    if key.startswith("mc_"):
        return f"Monte-Carlo distribution: {key[3:]}"
    return {
        "bode": "Frequency response",
        "transient": "Transient response",
        "dc_sweep": "DC sweep",
    }.get(key, key.replace("_", " ").capitalize())


def render_report(
    design: DesignResult,
    out_path: Path,
    template_dir: Path | None = None,
    pattern: Any = None,
) -> Path:
    """Render ``design`` to a single self-contained HTML file at ``out_path``."""
    out_path = Path(out_path)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    directory = Path(template_dir) if template_dir is not None else default_template_dir()
    if not (directory / "report.html.j2").is_file():
        raise ReportError(
            f"report template not found at {directory / 'report.html.j2'}; "
            "pass template_dir or restore the repository's templates/ directory"
        )

    env = Environment(
        loader=FileSystemLoader(str(directory)),
        undefined=StrictUndefined,
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["eng"] = format_eng

    schematic_svg = ""
    schematic_relative = design.artifacts.get("schematic", "")
    if schematic_relative.lower().endswith(".svg"):
        candidate = out_dir / Path(schematic_relative).name
        if not candidate.is_file():
            candidate = Path(schematic_relative)
        if candidate.is_file():
            schematic_svg = _inline_svg(candidate)

    verdict = design.verdict()
    context = {
        "design": design,
        "spec": design.spec,
        "pattern": pattern,
        "verdict": verdict,
        "verdict_class": {
            "PASS": "pass",
            "FAIL": "fail",
            "NOT-CONVERGED": "unknown",
        }[verdict],
        "verdict_detail": _verdict_detail(design),
        "schematic_svg": schematic_svg,
        "traceability": _traceability_rows(design),
        "bom": _bom_rows(design),
        "sensitivity": _sensitivity_rows(design),
        "figures": _figures(design, out_dir),
        "netlist": design.netlist,
        "tool_version": design.tool_version or __version__,
        "ngspice_version": design.ngspice_version or "unknown",
        "timestamp": design.timestamp or _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "spec_hash": design.spec_hash or design.spec.hash(),
        "escape": html.escape,
    }

    try:
        text = env.get_template("report.html.j2").render(**context)
    except Exception as exc:
        raise ReportError(
            f"failed to render {directory / 'report.html.j2'}: {type(exc).__name__}: {exc}"
        ) from exc

    out_path.write_text(text, encoding="utf-8")
    return out_path


def _verdict_detail(design: DesignResult) -> str:
    """One sentence under the banner explaining the verdict."""
    total = len(design.spec.requirements)
    if not design.converged:
        return (
            f"ngspice did not converge after {design.spice_attempts} attempts, so no "
            "requirement could be evaluated. This is a simulation problem, not a design "
            "failure."
        )
    failed = [r for r in design.requirement_results if not r.passed]
    if not failed:
        marginal = [
            r
            for r in design.requirement_results
            if r.mc_pass_rate is not None and r.mc_pass_rate < 0.99
        ]
        if marginal:
            return (
                f"All {total} requirements met at nominal, but "
                f"{len(marginal)} of them fall below a 99% Monte-Carlo pass rate over the "
                "declared component tolerances."
            )
        return f"All {total} requirements met, verified by ngspice."
    names = ", ".join(r.requirement_id for r in failed)
    return f"{len(failed)} of {total} requirements failed: {names}."
