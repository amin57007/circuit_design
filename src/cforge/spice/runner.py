"""ngspice invocation, ``.meas`` parsing and the convergence retry ladder.

This module is the only place in circuit-forge that produces verification
numbers.  Two rules govern its design:

1.  A measurement that ngspice reports as ``failed`` becomes ``None``.  It is
    recorded, never guessed at, and never crashes the run.
2.  Non-convergence is not a spec failure.  ``SpiceRun.converged`` is False in
    that case and the caller must report NOT-CONVERGED rather than FAIL.

ngspice is always invoked as ``ngspice -b -o <log> <netlist.cir>`` with an
argv list.  ``shell=True`` is never used.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "RETRY_LADDER",
    "NgspiceNotFound",
    "RawPlot",
    "SpiceRun",
    "measurements_from_log",
    "ngspice_version",
    "parse_raw",
    "run",
]


class NgspiceNotFound(RuntimeError):
    """Raised when the ngspice executable cannot be located on PATH."""


@dataclass
class RawPlot:
    """One analysis worth of data read back from an ngspice ASCII rawfile.

    ``vectors`` maps the lowercased vector name (``frequency``, ``v(out)``,
    ``i(v1)``, ...) to a complex-valued numpy array.  Real analyses (``.tran``,
    ``.dc``) still yield complex arrays with zero imaginary part so callers
    have one code path; take ``.real`` when plotting them.
    """

    name: str
    vectors: dict[str, "np.ndarray[Any, np.dtype[np.complex128]]"]

    @property
    def kind(self) -> str:
        """``ac``, ``tran``, ``dc``, ``op`` or ``other``, from the plot title."""
        lowered = self.name.lower()
        for key in ("ac", "transient", "dc", "operating"):
            if key in lowered:
                return {"transient": "tran", "operating": "op"}.get(key, key)
        return "other"

    def get(self, name: str) -> "np.ndarray[Any, np.dtype[np.complex128]] | None":
        return self.vectors.get(name.strip().lower())


@dataclass
class SpiceRun:
    """Outcome of one (possibly retried) ngspice invocation.

    ``measurements`` maps ``.meas`` names (lowercased, as ngspice prints them)
    to their values in SI base units.  Measurements that ngspice reported as
    failed appear in ``failed_measurements`` and not in ``measurements``.
    """

    ok: bool
    measurements: dict[str, float]
    stdout: str
    stderr: str
    converged: bool
    attempts: int
    failed_measurements: list[str] = field(default_factory=list)
    error: str = ""
    options_applied: list[str] = field(default_factory=list)
    netlist: str = ""
    plots: list[RawPlot] = field(default_factory=list)

    def plot(self, kind: str) -> RawPlot | None:
        """First plot of the given kind (``ac``, ``tran``, ``dc``), if any."""
        for candidate in self.plots:
            if candidate.kind == kind:
                return candidate
        return None

    def get(self, name: str) -> float | None:
        """Measurement value by name, case-insensitively; None if absent or failed."""
        return self.measurements.get(name.strip().lower())

    def all_names(self) -> list[str]:
        return sorted(set(self.measurements) | set(self.failed_measurements))


# Applied in order, each appended to the netlist on the next attempt.  Every
# rung is cumulative-free: exactly one extra .options block is added per try so
# a failure can be attributed to a specific setting.
RETRY_LADDER: tuple[tuple[str, str], ...] = (
    ("baseline", ""),
    ("rshunt", ".options rshunt=1e12"),
    ("gmin", ".options gmin=1e-10 abstol=1e-10 reltol=1e-3"),
    ("gear", ".options method=gear"),
    ("nodeset", "*<<NODESET>>"),
)

_CONVERGENCE_MARKERS: tuple[str, ...] = (
    "singular matrix",
    "no convergence",
    "timestep too small",
    "iteration limit reached",
    "transient analysis failed",
    "operating point could not be simulated",
    "run simulation(s) aborted",
)

# ngspice prints e.g.:
#   fc_hz               =  1.001234e+03 targ =  ...
#   vrip                =  4.83291e-02
#   settle_time_s       =  failed
# and, for .measure over multiple points, occasionally a trailing "at=" clause.
_MEAS_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_.\[\]]*)\s*=\s*(?P<value>\S+)(?P<rest>.*)$",
)

# ngspice's memory report ends with lines such as "Stack = 0 bytes." which are
# shaped exactly like a measurement.  A real .meas line has nothing after the
# value except further measurement clauses, so that is what we require.
_MEAS_TRAILER_RE = re.compile(r"^\s*(targ|trig|at|from|to|=)\b", re.IGNORECASE)

_NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")

# ngspice 42 reports an unsatisfiable measurement by echoing the card, e.g.
#   .meas ac settle_time_s when v(out)=1.5 failed!
# rather than printing "settle_time_s = failed".  Both forms are handled.
_MEAS_FAILED_RE = re.compile(
    r"^\s*\.meas(?:ure)?\s+\w+\s+(?P<name>[A-Za-z_][A-Za-z0-9_.]*)\b.*\bfailed",
    re.IGNORECASE,
)

# Lines that look like "name = value" but are not measurement results.
_NOT_MEAS_PREFIXES: tuple[str, ...] = (
    "warning",
    "error",
    "note",
    "using",
    "circuit",
    "doing",
    "reference value",
)

_SI_SUFFIX: dict[str, float] = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}


def ngspice_executable() -> str:
    """Absolute path to ngspice, or raise with an install hint."""
    exe = shutil.which("ngspice")
    if exe is None:
        raise NgspiceNotFound(
            "ngspice was not found on PATH. circuit-forge derives every numeric "
            "verdict from ngspice and cannot run without it.\n"
            "  macOS:        brew install ngspice\n"
            "  Ubuntu/Debian: sudo apt install ngspice\n"
            "  Windows:      use WSL2, or add the ngspice Windows build to PATH\n"
            "Then re-run: python scripts/check_env.py"
        )
    return exe


def ngspice_version() -> str:
    """Version banner line from ``ngspice -v``; ``"unknown"`` if unavailable."""
    try:
        exe = ngspice_executable()
    except NgspiceNotFound:
        return "unknown"
    try:
        proc = subprocess.run(
            [exe, "-v"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    for raw in (proc.stdout + proc.stderr).splitlines():
        line = raw.strip().lstrip("*").strip()
        if "ngspice" in line.lower():
            return line
    return "unknown"


def _parse_number(token: str) -> float | None:
    """Parse an ngspice numeric token, tolerating SI suffixes like ``4.7k``."""
    tok = token.strip().rstrip(",").lower()
    if not tok:
        return None
    if _NUMBER_RE.match(tok):
        return float(tok)
    match = re.match(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)([a-z]+)$", tok)
    if match is None:
        return None
    mantissa, suffix = match.groups()
    for key in ("meg", "t", "g", "k", "m", "u", "n", "p", "f"):
        if suffix.startswith(key):
            return float(mantissa) * _SI_SUFFIX[key]
    return None


def measurements_from_log(text: str) -> tuple[dict[str, float], list[str]]:
    """Extract ``.meas`` results from ngspice output.

    Returns ``(values, failed_names)``.  Names are lowercased.  A line whose
    value token is ``failed`` (ngspice's wording when a trigger/target is never
    met) contributes to ``failed_names`` with no numeric value.

    Parsing is intentionally forgiving: ngspice's ``.meas`` output format has
    drifted between releases, and losing one measurement must never abort a run.
    """
    values: dict[str, float] = {}
    failed: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("*", "#")):
            continue
        lowered = line.lower()
        echoed = _MEAS_FAILED_RE.match(line)
        if echoed is not None:
            name = echoed.group("name").lower()
            if name not in failed:
                failed.append(name)
            values.pop(name, None)
            continue
        if any(lowered.startswith(p) for p in _NOT_MEAS_PREFIXES):
            continue
        match = _MEAS_RE.match(line)
        if match is None:
            continue
        name = match.group("name").strip().lower()
        token = match.group("value").strip()
        if token.lower().startswith("failed"):
            if name not in failed:
                failed.append(name)
            values.pop(name, None)
            continue
        rest = match.group("rest").strip()
        if rest and not _MEAS_TRAILER_RE.match(rest):
            continue
        number = _parse_number(token)
        if number is None:
            continue
        if name in failed:
            continue
        values[name] = number
    return values, failed


def parse_raw(text: str) -> list[RawPlot]:
    """Parse an ngspice ASCII rawfile into one :class:`RawPlot` per analysis.

    The format is a sequence of blocks, each with a ``Plotname:`` header, a
    ``Variables:`` table and a ``Values:`` section.  ``Flags: complex`` marks
    AC data, whose samples are written as ``real,imag`` pairs.

    Parsing never raises on malformed data: plot data is a convenience for the
    report, and losing it must not fail a run whose measurements are fine.
    """
    plots: list[RawPlot] = []
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        if not lines[i].startswith("Plotname:"):
            i += 1
            continue
        name = lines[i].split(":", 1)[1].strip()
        complex_flag = False
        n_vars = 0
        n_points = 0
        names: list[str] = []
        i += 1
        while i < n and not lines[i].startswith("Values:"):
            line = lines[i]
            if line.startswith("Flags:"):
                complex_flag = "complex" in line.lower()
            elif line.startswith("No. Variables:"):
                n_vars = _safe_int(line.split(":", 1)[1])
            elif line.startswith("No. Points:"):
                n_points = _safe_int(line.split(":", 1)[1])
            elif line.startswith("Variables:"):
                for _ in range(n_vars):
                    i += 1
                    if i >= n:
                        break
                    parts = lines[i].split()
                    names.append(parts[1].lower() if len(parts) >= 2 else f"v{len(names)}")
            i += 1
        i += 1  # step past "Values:"

        if n_vars == 0 or not names:
            continue
        columns: list[list[complex]] = [[] for _ in names]
        read = 0
        while i < n and read < n_points:
            for var in range(n_vars):
                if i >= n:
                    break
                token = lines[i].split()[-1] if lines[i].split() else ""
                columns[var].append(_parse_complex(token, complex_flag))
                i += 1
            read += 1
        vectors = {
            names[k]: np.asarray(columns[k], dtype=np.complex128) for k in range(len(names))
        }
        plots.append(RawPlot(name=name, vectors=vectors))
    return plots


def _safe_int(token: str) -> int:
    try:
        return int(token.strip())
    except ValueError:
        return 0


def _parse_complex(token: str, complex_flag: bool) -> complex:
    """Parse one rawfile sample.  Unparseable samples become NaN, not exceptions."""
    try:
        if complex_flag and "," in token:
            real_text, imag_text = token.split(",", 1)
            return complex(float(real_text), float(imag_text))
        return complex(float(token), 0.0)
    except ValueError:
        return complex("nan")


def _has_convergence_failure(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _CONVERGENCE_MARKERS)


def _apply_rung(netlist: str, directive: str, nodeset: str | None) -> tuple[str, str]:
    """Return ``(netlist_with_directive, description)`` for one ladder rung."""
    if not directive:
        return netlist, "baseline"
    if directive == "*<<NODESET>>":
        if not nodeset:
            return netlist, "nodeset (unavailable, skipped)"
        return _insert_before_end(netlist, nodeset), "nodeset from DC operating point"
    return _insert_before_end(netlist, directive), directive


def _insert_before_end(netlist: str, extra: str) -> str:
    """Insert ``extra`` just before the final ``.end`` line of ``netlist``."""
    lines = netlist.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().lower() == ".end":
            lines[i:i] = extra.splitlines()
            return "\n".join(lines) + "\n"
    return netlist.rstrip("\n") + "\n" + extra.rstrip("\n") + "\n"


def _dc_nodeset(
    netlist: str, exe: str, workdir: Path, timeout_s: int
) -> str | None:
    """Run a DC operating point and turn the node voltages into a ``.nodeset``.

    Returns None when the operating point itself does not solve, in which case
    the ladder simply skips this rung.
    """
    body = _strip_analyses(netlist)
    probe = _insert_before_end(body, ".op\n.print all")
    outcome = _invoke(probe, exe, workdir / "nodeset", timeout_s)
    if outcome is None:
        return None
    text = outcome[0]
    voltages: dict[str, float] = {}
    for raw in text.splitlines():
        match = re.match(r"^\s*V\(([^)]+)\)\s*=\s*(\S+)", raw, flags=re.IGNORECASE)
        if match is None:
            continue
        value = _parse_number(match.group(2))
        if value is not None:
            voltages[match.group(1).strip()] = value
    if not voltages:
        return None
    terms = " ".join(f"v({node})={value:.6g}" for node, value in sorted(voltages.items()))
    return f".nodeset {terms}"


_ANALYSIS_RE = re.compile(r"^\s*\.(ac|tran|dc|noise|disto|pz|sens|four)\b", re.IGNORECASE)


def _strip_analyses(netlist: str) -> str:
    """Remove analysis and .meas cards so only the topology remains."""
    out: list[str] = []
    for line in netlist.splitlines():
        stripped = line.strip().lower()
        if _ANALYSIS_RE.match(line) or stripped.startswith((".meas", ".measure")):
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def _invoke(
    netlist: str, exe: str, workdir: Path, timeout_s: int, rawfile: bool = False
) -> tuple[str, str, int, list[RawPlot]] | None:
    """Run ngspice once.  Returns ``(combined_output, stderr, returncode, plots)``.

    ``rawfile`` selects the *plot-data pass*.  ngspice 42 refuses outright to
    evaluate ``.meas`` in batch mode when ``-r`` is given ("No .measure
    possible in batch mode (-b) with -r rawfile set!"), so measurements and
    waveform data must come from two separate invocations of the same deck.

    Returns None if the process timed out or could not be started; the caller
    turns that into ``ok=False``.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    cir = workdir / "circuit.cir"
    log = workdir / "ngspice.log"
    raw = workdir / "circuit.raw"
    cir.write_text(netlist, encoding="utf-8")
    argv: list[str] = [exe, "-b", "-o", str(log)]
    if rawfile:
        argv += ["-r", str(raw)]
    argv.append(str(cir))
    env = dict(os.environ)
    env["SPICE_ASCIIRAWFILE"] = "1"
    try:
        proc = subprocess.run(  # noqa: S603 - argv list, never shell=True
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(workdir),
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    except OSError:
        return None
    log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    combined = "\n".join(part for part in (log_text, proc.stdout, proc.stderr) if part)
    plots: list[RawPlot] = []
    if rawfile and raw.exists():
        plots = parse_raw(raw.read_text(encoding="utf-8", errors="replace"))
    return combined, proc.stderr, proc.returncode, plots


def run(
    netlist: str,
    timeout_s: int = 60,
    workdir: Path | None = None,
    max_attempts: int = len(RETRY_LADDER),
    collect_plots: bool = False,
) -> SpiceRun:
    """Simulate ``netlist`` with ngspice, climbing the convergence ladder as needed.

    ``timeout_s`` applies to each individual attempt; a timeout kills the
    subprocess and returns ``ok=False`` rather than hanging the pipeline.
    ``workdir`` receives the generated ``circuit.cir`` and ``ngspice.log`` for
    each attempt (in numbered sub-directories); when omitted a temp dir is used
    and removed afterwards.

    ``collect_plots`` adds a second invocation of the winning deck with ``-r``
    to capture waveform data for the report.  It is off by default because
    Monte Carlo needs measurements only and this doubles the simulation cost.
    """
    exe = ngspice_executable()
    tmp: tempfile.TemporaryDirectory[str] | None = None
    if workdir is None:
        tmp = tempfile.TemporaryDirectory(prefix="cforge-spice-")
        base = Path(tmp.name)
    else:
        base = Path(workdir)
        base.mkdir(parents=True, exist_ok=True)

    try:
        result = _run_ladder(netlist, exe, base, timeout_s, max_attempts)
        if collect_plots and result.converged and result.netlist:
            outcome = _invoke(result.netlist, exe, base / "plotdata", timeout_s, rawfile=True)
            if outcome is not None:
                result.plots = outcome[3]
        return result
    finally:
        if tmp is not None:
            tmp.cleanup()


def _run_ladder(
    netlist: str, exe: str, base: Path, timeout_s: int, max_attempts: int
) -> SpiceRun:
    ladder = RETRY_LADDER[: max(1, max_attempts)]
    nodeset: str | None = None
    last: SpiceRun | None = None
    applied: list[str] = []

    for attempt, (label, directive) in enumerate(ladder, start=1):
        if directive == "*<<NODESET>>" and nodeset is None:
            nodeset = _dc_nodeset(netlist, exe, base / f"attempt{attempt}", timeout_s)
        candidate, description = _apply_rung(netlist, directive, nodeset)
        applied.append(description)

        outcome = _invoke(candidate, exe, base / f"attempt{attempt}", timeout_s)
        if outcome is None:
            last = SpiceRun(
                ok=False,
                measurements={},
                stdout="",
                stderr="",
                converged=False,
                attempts=attempt,
                error=(
                    f"ngspice attempt {attempt} ({label}) exceeded the {timeout_s}s "
                    "timeout and was terminated; simplify the analysis or raise timeout_s"
                ),
                options_applied=list(applied),
                netlist=candidate,
            )
            continue

        combined, stderr, returncode, plots = outcome
        values, failed = measurements_from_log(combined)
        converged = not _has_convergence_failure(combined)
        ok = converged and returncode == 0
        last = SpiceRun(
            ok=ok,
            measurements=values,
            stdout=combined,
            stderr=stderr,
            converged=converged,
            attempts=attempt,
            failed_measurements=failed,
            error="" if ok else _describe_failure(combined, returncode, converged),
            options_applied=list(applied),
            netlist=candidate,
            plots=plots,
        )
        if converged:
            return last

    if last is None:  # pragma: no cover - ladder always has at least one rung
        raise RuntimeError("convergence ladder produced no attempts")
    return last


def _describe_failure(text: str, returncode: int, converged: bool) -> str:
    """One actionable sentence naming what went wrong."""
    lowered = text.lower()
    for marker in _CONVERGENCE_MARKERS:
        if marker in lowered:
            return (
                f"ngspice reported '{marker}'; the convergence ladder "
                "(rshunt, gmin/abstol/reltol, gear, nodeset) did not recover it. "
                "This is a simulation failure, not a requirement failure."
            )
    if not converged:
        return "ngspice did not converge (reason not recognised in the log)"
    for raw in text.splitlines():
        if raw.strip().lower().startswith("error"):
            return f"ngspice error: {raw.strip()}"
    return f"ngspice exited with code {returncode}"
