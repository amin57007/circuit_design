"""``cforge`` command-line interface.

Exit codes are part of the contract so this is usable as a CI gate:

    0   PASS - every requirement met, verified by ngspice
    1   FAIL - the design simulated cleanly but violates a requirement
    2   non-convergence, or any tool error (bad spec, missing pattern, ...)
"""

from __future__ import annotations

import datetime as _dt
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

from cforge import __version__
from cforge.catalog.loader import Pattern, PatternError, load_all
from cforge.catalog.select import NoPatternFound, rank, select
from cforge.models import (
    Component,
    DesignResult,
    MeasResult,
    RequirementResult,
    Spec,
)
from cforge.report import ReportError, render_report
from cforge.spice.runner import NgspiceNotFound, SpiceRun, ngspice_version
from cforge.spice.runner import run as spice_run
from cforge.synth.solver import build_testbench, synthesize

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Deterministic circuit synthesis: a YAML spec in, an ngspice-verified "
        "design, schematic and HTML report out. No numeric result in this tool "
        "comes from a language model."
    ),
)

out_console = Console()
err_console = Console(stderr=True)


def _fail(message: str, code: int = EXIT_ERROR) -> None:
    """Print an actionable error to stderr and exit."""
    err_console.print(f"[bold red]error[/bold red] {message}")
    raise typer.Exit(code)


def _load_spec(path: Path) -> Spec:
    if not path.is_file():
        _fail(f"spec file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        _fail(f"{path} is not valid YAML: {exc}")
    except OSError as exc:
        _fail(f"could not read {path}: {exc}")
    if not isinstance(raw, dict):
        _fail(f"{path} must contain a YAML mapping at the top level")
    try:
        return Spec.model_validate(raw)
    except Exception as exc:
        _fail(f"{path} failed validation:\n{exc}")
    raise AssertionError("unreachable")


def _load_catalog(root: Path | None) -> list[Pattern]:
    try:
        return load_all(root)
    except PatternError as exc:
        _fail(str(exc))
    raise AssertionError("unreachable")


@app.command("check-env")
def check_env() -> None:
    """Verify Python and ngspice are usable before anything else."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "check_env.py"
    if not script.is_file():
        _fail(f"scripts/check_env.py not found at {script}")
    proc = subprocess.run(  # noqa: S603 - fixed argv
        [sys.executable, str(script)], check=False
    )
    raise typer.Exit(proc.returncode)


@app.command("list-patterns")
def list_patterns(
    block: Annotated[
        str | None, typer.Option("--block", help="Only patterns implementing this block.")
    ] = None,
    patterns_root: Annotated[
        Path | None, typer.Option("--patterns", help="Alternative pattern directory.")
    ] = None,
) -> None:
    """List the topologies in the catalog."""
    catalog = _load_catalog(patterns_root)
    if block:
        catalog = [p for p in catalog if p.block == block]
        if not catalog:
            _fail(f"no pattern implements block {block!r}")

    table = Table(title="cforge pattern catalog", header_style="bold")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("block", no_wrap=True)
    table.add_column("name")
    table.add_column("measurements")
    table.add_column("requires")
    for pattern in catalog:
        table.add_row(
            pattern.id,
            pattern.block,
            pattern.meta.name,
            ", ".join(pattern.meta.provides_meas),
            ", ".join(pattern.meta.requires) or "-",
        )
    out_console.print(table)


@app.command("explain")
def explain_pattern(
    pattern_id: Annotated[str, typer.Argument(help="Pattern id, e.g. sallen_key_lp2")],
    patterns_root: Annotated[
        Path | None, typer.Option("--patterns", help="Alternative pattern directory.")
    ] = None,
) -> None:
    """Print a pattern's notes, trade-offs and gotchas."""
    catalog = _load_catalog(patterns_root)
    pattern = next((p for p in catalog if p.id == pattern_id), None)
    if pattern is None:
        _fail(
            f"unknown pattern {pattern_id!r}. Available: "
            + ", ".join(sorted(p.id for p in catalog))
        )
        raise AssertionError("unreachable")

    out_console.rule(f"[bold cyan]{pattern.id}[/bold cyan] - {pattern.meta.name}")
    out_console.print(f"block: [bold]{pattern.block}[/bold]")
    if pattern.meta.requires:
        out_console.print(f"requires: {', '.join(pattern.meta.requires)}")
    out_console.print(f"provides: {', '.join(pattern.meta.provides_meas)}")

    if pattern.meta.tradeoffs.pros or pattern.meta.tradeoffs.cons:
        out_console.print("\n[bold green]Pros[/bold green]")
        for item in pattern.meta.tradeoffs.pros:
            out_console.print(f"  + {item}")
        out_console.print("[bold red]Cons[/bold red]")
        for item in pattern.meta.tradeoffs.cons:
            out_console.print(f"  - {item}")

    if pattern.meta.gotchas:
        out_console.print("\n[bold yellow]Gotchas[/bold yellow]")
        for gotcha in pattern.meta.gotchas:
            checked = f" (checked: {gotcha.check})" if gotcha.check else " (advisory)"
            out_console.print(f"  [bold]{gotcha.id}[/bold]{checked}")
            out_console.print(f"    {gotcha.text.strip()}")

    out_console.print("\n[bold]notes.md[/bold]")
    out_console.print(pattern.notes)


@app.command("design")
def design_command(
    spec_path: Annotated[Path, typer.Argument(metavar="SPEC.yaml", help="Spec file.")],
    out_dir: Annotated[
        Path, typer.Option("-o", "--out", help="Output root; results go in <out>/<spec-name>/.")
    ] = Path("out"),
    topology: Annotated[
        str | None, typer.Option("--topology", help="Force a pattern id instead of selecting.")
    ] = None,
    mc: Annotated[int, typer.Option("--mc", help="Monte-Carlo runs over tolerances.")] = 500,
    no_mc: Annotated[bool, typer.Option("--no-mc", help="Skip Monte Carlo entirely.")] = False,
    open_report: Annotated[
        bool, typer.Option("--open", help="Open the finished report in a browser.")
    ] = False,
    timeout_s: Annotated[int, typer.Option("--timeout", help="Per-ngspice-run timeout.")] = 60,
    schematic_backend: Annotated[
        str, typer.Option("--schematic-backend", help="schemdraw (default) or lcapy.")
    ] = "schemdraw",
    patterns_root: Annotated[
        Path | None, typer.Option("--patterns", help="Alternative pattern directory.")
    ] = None,
    jobs: Annotated[
        int | None, typer.Option("--jobs", help="Worker processes for Monte Carlo.")
    ] = None,
) -> None:
    """Synthesize, simulate and verify a spec, then write a report."""
    spec = _load_spec(spec_path)
    if topology:
        spec = spec.model_copy(update={"topology": topology})

    catalog = _load_catalog(patterns_root)
    try:
        pattern = select(spec, catalog)
    except NoPatternFound as exc:
        _fail(str(exc))
        raise AssertionError("unreachable")

    out_console.print(
        f"[bold]{spec.name}[/bold]: block [cyan]{spec.block}[/cyan] "
        f"-> topology [cyan]{pattern.id}[/cyan] ({pattern.meta.name})"
    )

    work_dir = Path(out_dir) / spec.name
    work_dir.mkdir(parents=True, exist_ok=True)

    design = _run_pipeline(
        spec=spec,
        pattern=pattern,
        work_dir=work_dir,
        mc_runs=0 if no_mc else max(0, mc),
        timeout_s=timeout_s,
        schematic_backend=schematic_backend,
        jobs=jobs,
    )

    _print_summary(design)
    _write_outputs(design, pattern, work_dir, open_report)
    raise typer.Exit(design.exit_code())


@app.command("verify")
def verify_command(
    design_path: Annotated[Path, typer.Argument(metavar="DESIGN.json", help="Existing design.")],
    patterns_root: Annotated[
        Path | None, typer.Option("--patterns", help="Alternative pattern directory.")
    ] = None,
    timeout_s: Annotated[int, typer.Option("--timeout", help="Per-ngspice-run timeout.")] = 60,
) -> None:
    """Re-run ngspice against an existing design.json and re-check every requirement.

    Nothing is re-synthesized: the stored component values are simulated exactly
    as they are, so this answers "does this saved design still pass?".
    """
    if not design_path.is_file():
        _fail(f"design file not found: {design_path}")
    try:
        stored = DesignResult.from_json(design_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _fail(f"{design_path} is not a valid design.json: {exc}")
        raise AssertionError("unreachable")

    catalog = _load_catalog(patterns_root)
    pattern = next((p for p in catalog if p.id == stored.pattern_id), None)
    if pattern is None:
        _fail(
            f"{design_path} was produced with pattern {stored.pattern_id!r}, which is not in "
            f"the catalog. Available: {', '.join(sorted(p.id for p in catalog))}"
        )
        raise AssertionError("unreachable")

    deck = build_testbench(stored.spec, pattern, stored.components)
    try:
        run_result = spice_run(deck, timeout_s=timeout_s)
    except NgspiceNotFound as exc:
        _fail(str(exc))
        raise AssertionError("unreachable")

    design = _assemble(
        spec=stored.spec,
        pattern=pattern,
        components=stored.components,
        netlist=deck,
        run_result=run_result,
        warnings=list(stored.warnings),
        gotchas=list(stored.gotchas),
        iterations=stored.iterations,
    )
    _print_summary(design)
    raise typer.Exit(design.exit_code())


def _run_pipeline(
    spec: Spec,
    pattern: Pattern,
    work_dir: Path,
    mc_runs: int,
    timeout_s: int,
    schematic_backend: str,
    jobs: int | None,
) -> DesignResult:
    """Synthesize, simulate, measure, sample tolerances and render figures."""
    try:
        synthesis = synthesize(
            spec,
            pattern,
            refine=True,
            simulate=_make_simulate(timeout_s) if pattern.meta.numeric_refine else None,
        )
    except (PatternError, KeyError, ValueError) as exc:
        _fail(f"synthesis failed for pattern {pattern.id!r}: {exc}")
        raise AssertionError("unreachable")

    for component in synthesis.components:
        out_console.print(
            f"  {component.ref:<5} {component.value:>12.6g} {component.unit:<4} "
            f"(ideal {component.ideal_value:.6g}, {component.deviation_percent:+.2f}%)"
        )

    deck = build_testbench(spec, pattern, synthesis.components)
    try:
        run_result = spice_run(
            deck, timeout_s=timeout_s, workdir=work_dir / "spice", collect_plots=True
        )
    except NgspiceNotFound as exc:
        _fail(str(exc))
        raise AssertionError("unreachable")

    if run_result.attempts > 1:
        out_console.print(
            f"  [yellow]ngspice needed {run_result.attempts} attempts[/yellow] "
            f"({'; '.join(run_result.options_applied)})"
        )

    design = _assemble(
        spec=spec,
        pattern=pattern,
        components=synthesis.components,
        netlist=deck,
        run_result=run_result,
        warnings=list(synthesis.warnings),
        gotchas=list(synthesis.gotchas),
        iterations=synthesis.iterations,
    )

    if mc_runs > 0 and run_result.converged:
        design = _add_sensitivity(design, spec, pattern, mc_runs, timeout_s, jobs)
    elif mc_runs > 0:
        design.warnings.append(
            "Monte Carlo skipped: the nominal design did not converge, so sampling "
            "tolerances around it would not mean anything."
        )

    _render_artifacts(design, pattern, run_result, work_dir, schematic_backend)
    return design


def _make_simulate(timeout_s: int) -> Any:
    def simulate(deck: str) -> dict[str, float]:
        return spice_run(deck, timeout_s=timeout_s).measurements

    return simulate


def _assemble(
    spec: Spec,
    pattern: Pattern,
    components: list[Component],
    netlist: str,
    run_result: SpiceRun,
    warnings: list[str],
    gotchas: list[dict[str, str]],
    iterations: int,
) -> DesignResult:
    """Build the DesignResult from a completed simulation."""
    measurements = [
        MeasResult(name=name, value=run_result.measurements.get(name), raw="")
        for name in sorted(set(run_result.measurements) | set(run_result.failed_measurements))
    ]

    requirement_results: list[RequirementResult] = []
    for req in spec.requirements:
        value = run_result.get(req.meas)
        passed, margin = req.check(value)
        requirement_results.append(
            RequirementResult(
                requirement_id=req.id,
                meas=req.meas,
                value=value,
                passed=passed,
                margin_percent=margin,
                limit_text=req.limit_text(),
                description=req.description,
                unit=req.unit,
            )
        )

    if not run_result.converged:
        warnings.append(run_result.error or "ngspice did not converge")
    for name in run_result.failed_measurements:
        warnings.append(
            f"ngspice could not evaluate measurement {name!r} (its trigger or target was "
            f"never reached). Any requirement on {name!r} is reported as NO-DATA, not as a "
            "pass."
        )
    missing = {r.meas for r in spec.requirements} - set(run_result.all_names())
    for name in sorted(missing):
        warnings.append(
            f"no .meas named {name!r} in the ngspice output. Pattern {pattern.id!r} declares "
            f"{', '.join(pattern.meta.provides_meas)}; check the spec's requirement meas names "
            f"or add the measurement to {pattern.directory / 'tb.cir.j2'}."
        )

    return DesignResult(
        spec=spec,
        pattern_id=pattern.id,
        components=components,
        netlist=netlist,
        measurements=measurements,
        requirement_results=requirement_results,
        passed=run_result.converged and all(r.passed for r in requirement_results),
        iterations=iterations,
        warnings=warnings,
        gotchas=gotchas,
        converged=run_result.converged,
        spice_attempts=run_result.attempts,
        tool_version=__version__,
        ngspice_version=ngspice_version(),
        timestamp=_dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        spec_hash=spec.hash(),
    )


def _add_sensitivity(
    design: DesignResult,
    spec: Spec,
    pattern: Pattern,
    mc_runs: int,
    timeout_s: int,
    jobs: int | None,
) -> DesignResult:
    from cforge.synth.sensitivity import monte_carlo

    if not any(c.tolerance_percent > 0 for c in design.components):
        design.warnings.append(
            "Monte Carlo skipped: no component has a non-zero tolerance. Add a "
            "'tolerance_percent:' block to the spec, e.g. {R: 1.0, C: 5.0}."
        )
        return design

    results = monte_carlo(
        spec,
        pattern,
        design.components,
        n=mc_runs,
        timeout_s=timeout_s,
        jobs=jobs,
        console=out_console,
    )
    design.sensitivity = results

    by_meas = {r.meas: r for r in results}
    for requirement_result in design.requirement_results:
        item = by_meas.get(requirement_result.meas)
        if item is None:
            continue
        requirement_result.mc_pass_rate = item.pass_rate
        if item.pass_rate < 0.99:
            design.warnings.append(
                f"{requirement_result.requirement_id}: Monte-Carlo pass rate is "
                f"{100.0 * item.pass_rate:.1f}% over the declared tolerances (nominal "
                f"{'passes' if requirement_result.passed else 'already fails'}). "
                "Tighten component tolerances or re-centre the design."
            )
    return design


def _render_artifacts(
    design: DesignResult,
    pattern: Pattern,
    run_result: SpiceRun,
    work_dir: Path,
    schematic_backend: str,
) -> None:
    """Draw the schematic and every applicable plot, recording artifact paths."""
    from cforge.render import plots as plot_module
    from cforge.render.schematic import SchematicError
    from cforge.render.schematic import render as render_schematic

    suffix = ".svg" if schematic_backend == "schemdraw" else ".pdf"
    schematic_path = work_dir / f"schematic{suffix}"
    try:
        render_schematic(
            pattern,
            design.components,
            schematic_path,
            fmt=suffix.lstrip("."),
            title=design.spec.name,
            verdict=design.verdict(),
            backend=schematic_backend,
        )
        design.artifacts["schematic"] = str(schematic_path)
    except SchematicError as exc:
        design.warnings.append(f"schematic not generated: {exc}")

    markers = _plot_markers(design, pattern)

    ac = run_result.plot("ac")
    if ac is not None:
        out = _bode_from_plot(ac, design, work_dir, plot_module, markers)
        if out is not None:
            design.artifacts["bode"] = str(out)

    tran = run_result.plot("tran")
    if tran is not None:
        out = _transient_from_plot(tran, design, work_dir, plot_module)
        if out is not None:
            design.artifacts["transient"] = str(out)

    for item in design.sensitivity:
        if not item.samples:
            continue
        requirement = next(
            (r for r in design.spec.requirements if r.meas == item.meas), None
        )
        path = work_dir / f"mc_{item.meas}.png"
        plot_module.mc_histogram(
            item, requirement, path, unit=requirement.unit if requirement else ""
        )
        design.artifacts[f"mc_{item.meas}"] = str(path)


def _plot_markers(design: DesignResult, pattern: Pattern) -> dict[str, tuple[float, float]]:
    """Ask the pattern where on the frequency axis each spot measurement was taken."""
    hook = getattr(pattern.design_module, "plot_markers", None)
    if hook is None:
        return {}
    values = {c.ref: c.value for c in design.components}
    measurements = {m.name: m.value for m in design.measurements if m.value is not None}
    try:
        return dict(hook(design.spec, values, measurements))
    except Exception as exc:
        design.warnings.append(
            f"plot markers unavailable: {pattern.directory / 'design.py'} plot_markers() "
            f"raised {type(exc).__name__}: {exc}"
        )
        return {}


def _bode_from_plot(
    ac: Any,
    design: DesignResult,
    work_dir: Path,
    plot_module: Any,
    markers: dict[str, tuple[float, float]],
) -> Path | None:
    import numpy as np

    freq = ac.get("frequency")
    out = ac.get("v(out)")
    if freq is None or out is None or freq.size == 0:
        design.warnings.append(
            "bode plot skipped: the AC analysis produced no v(out) vector. Add "
            "'.save v(out)' to the pattern testbench."
        )
        return None
    magnitude = 20.0 * np.log10(np.maximum(np.abs(out), 1e-30))
    phase = np.degrees(np.angle(out))
    path = work_dir / "bode.png"
    plot_module.bode(
        freq.real,
        magnitude,
        phase,
        design.spec.requirements,
        path,
        fc_hz=design.measurement("fc_hz"),
        title=f"{design.spec.name}: frequency response",
        markers=markers,
    )
    return path


def _transient_from_plot(
    tran: Any, design: DesignResult, work_dir: Path, plot_module: Any
) -> Path | None:
    time = tran.get("time")
    out = tran.get("v(out)")
    if time is None or out is None or time.size == 0:
        return None
    path = work_dir / "transient.png"
    plot_module.transient(
        time.real,
        {"v(out)": out.real},
        design.spec.requirements,
        path,
        settle_time_s=design.measurement("settle_time_s"),
        overshoot_percent=design.measurement("overshoot_pct"),
        title=f"{design.spec.name}: transient response",
        ylabel="Voltage [V]",
    )
    return path


def _write_outputs(
    design: DesignResult, pattern: Pattern, work_dir: Path, open_report: bool
) -> None:
    (work_dir / "netlist.cir").write_text(design.netlist, encoding="utf-8")

    report_path = work_dir / "report.html"
    try:
        render_report(design, report_path, pattern=pattern)
        design.artifacts["report"] = str(report_path)
    except ReportError as exc:
        err_console.print(f"[bold red]error[/bold red] {exc}")

    (work_dir / "design.json").write_text(design.to_json(), encoding="utf-8")

    out_console.print(f"\nwrote [bold]{work_dir}/[/bold]")
    for name in sorted(p.name for p in work_dir.iterdir() if p.is_file()):
        out_console.print(f"  {name}")

    if open_report and report_path.is_file():
        webbrowser.open(report_path.resolve().as_uri())


def _print_summary(design: DesignResult) -> None:
    table = Table(title="Requirements traceability", header_style="bold")
    table.add_column("REQ", no_wrap=True)
    table.add_column("meas", no_wrap=True)
    table.add_column("limit", no_wrap=True)
    table.add_column("measured", justify="right")
    table.add_column("margin", justify="right")
    table.add_column("MC", justify="right")
    table.add_column("status", no_wrap=True)

    for result in design.requirement_results:
        status = result.status
        colour = {"PASS": "green", "FAIL": "red", "NO-DATA": "yellow"}[status]
        table.add_row(
            result.requirement_id,
            result.meas,
            result.limit_text,
            "-" if result.value is None else f"{result.value:.6g}",
            "-" if result.margin_percent is None else f"{result.margin_percent:+.1f}%",
            "-" if result.mc_pass_rate is None else f"{100.0 * result.mc_pass_rate:.1f}%",
            f"[{colour}]{status}[/{colour}]",
        )
    out_console.print(table)

    for warning in design.warnings:
        err_console.print(f"[yellow]warning[/yellow] {warning}")
    for gotcha in design.gotchas:
        err_console.print(
            f"[yellow]gotcha[/yellow] [bold]{gotcha.get('id', '')}[/bold] "
            f"{gotcha.get('detail') or gotcha.get('text', '')}"
        )

    verdict = design.verdict()
    style = {"PASS": "bold green", "FAIL": "bold red", "NOT-CONVERGED": "bold yellow"}[verdict]
    out_console.print(f"\n[{style}]{verdict}[/{style}]")


@app.callback()
def main_callback(
    version: Annotated[
        bool, typer.Option("--version", help="Print the cforge version and exit.")
    ] = False,
) -> None:
    if version:
        out_console.print(f"circuit-forge {__version__}")
        raise typer.Exit(EXIT_PASS)


def main() -> None:
    """Console-script entry point."""
    try:
        app()
    except NgspiceNotFound as exc:
        err_console.print(f"[bold red]error[/bold red] {exc}")
        raise SystemExit(EXIT_ERROR) from exc
    except PatternError as exc:
        err_console.print(f"[bold red]error[/bold red] {exc}")
        raise SystemExit(EXIT_ERROR) from exc


if __name__ == "__main__":
    main()
