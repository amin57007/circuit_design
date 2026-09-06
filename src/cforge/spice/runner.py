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

__all__ = [
    "RETRY_LADDER",
    "NgspiceNotFound",
    "SpiceRun",
    "measurements_from_log",
    "ngspice_version",
    "run",
]


class NgspiceNotFound(RuntimeError):
    """Raised when the ngspice executable cannot be located on PATH."""


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
    r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_.\[\]]*)\s*=\s*(?P<value>\S+)",
)

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
        number = _parse_number(token)
        if number is None:
            continue
        if name in failed:
            continue
        values[name] = number
    return values, failed


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
    result = _invoke(probe, exe, workdir / "nodeset", timeout_s)
    if result is None:
        return None
    text = result[0]
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
    netlist: str, exe: str, workdir: Path, timeout_s: int
) -> tuple[str, str, int] | None:
    """Run ngspice once.  Returns ``(combined_output, stderr, returncode)``.

    Returns None if the process timed out or could not be started; the caller
    turns that into ``ok=False``.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    cir = workdir / "circuit.cir"
    log = workdir / "ngspice.log"
    cir.write_text(netlist, encoding="utf-8")
    argv: Sequence[str] = [exe, "-b", "-o", str(log), str(cir)]
    env = dict(os.environ)
    env.setdefault("SPICE_ASCIIRAWFILE", "1")
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
    return combined, proc.stderr, proc.returncode


def run(
    netlist: str,
    timeout_s: int = 60,
    workdir: Path | None = None,
    max_attempts: int = len(RETRY_LADDER),
) -> SpiceRun:
    """Simulate ``netlist`` with ngspice, climbing the convergence ladder as needed.

    ``timeout_s`` applies to each individual attempt; a timeout kills the
    subprocess and returns ``ok=False`` rather than hanging the pipeline.
    ``workdir`` receives the generated ``circuit.cir`` and ``ngspice.log`` for
    each attempt (in numbered sub-directories); when omitted a temp dir is used
    and removed afterwards.
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
        return _run_ladder(netlist, exe, base, timeout_s, max_attempts)
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

        combined, stderr, returncode = outcome
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
