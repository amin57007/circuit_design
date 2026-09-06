#!/usr/bin/env python3
"""Verify the circuit-forge runtime environment.

Checks, in order:
  1. Python >= 3.11
  2. ``ngspice -v`` runs successfully and is on PATH
  3. (advisory) the Python dependencies import cleanly

Exit codes: 0 = everything required is present, 2 = a required check failed.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import warnings

MIN_PYTHON = (3, 11)

INSTALL_HINT = {
    "Darwin": "brew install ngspice",
    "Linux": "sudo apt install ngspice   # or: dnf install ngspice",
    "Windows": "use WSL2 (sudo apt install ngspice) or the ngspice Windows build, "
    "then add its bin/ directory to PATH",
}

OPTIONAL_IMPORTS = (
    "numpy",
    "scipy",
    "sympy",
    "lcapy",
    "schemdraw",
    "matplotlib",
    "pydantic",
    "yaml",
    "jinja2",
    "typer",
    "rich",
)


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def check_python() -> bool:
    """Return True when the interpreter is new enough."""
    if sys.version_info < MIN_PYTHON:
        _err(
            f"FAIL  python: found {platform.python_version()}, "
            f"need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}.\n"
            f"      Fix: install a newer Python and recreate the venv, e.g.\n"
            f"        uv venv --python 3.11 && source .venv/bin/activate"
        )
        return False
    print(f"ok    python {platform.python_version()}")
    return True


def ngspice_version() -> str | None:
    """Return the first line of ``ngspice -v`` output, or None if unavailable."""
    exe = shutil.which("ngspice")
    if exe is None:
        return None
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, never shell=True
            [exe, "-v"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _err(f"FAIL  ngspice: found at {exe} but could not be executed: {exc}")
        return None
    if proc.returncode != 0:
        _err(
            f"FAIL  ngspice: '{exe} -v' exited {proc.returncode}\n"
            f"      stdout: {proc.stdout.strip()[:400]}\n"
            f"      stderr: {proc.stderr.strip()[:400]}"
        )
        return None
    return _first_version_line(proc.stdout or proc.stderr)


def _first_version_line(text: str) -> str:
    """Pull the human-readable version line out of ngspice's banner.

    ngspice prints a box of '*' characters around lines such as
    ``** ngspice-42 : Circuit level simulation program``.
    """
    for raw in text.splitlines():
        line = raw.strip().lstrip("*").strip()
        if "ngspice" in line.lower():
            return line
    return "ngspice (version string not recognised)"


def check_ngspice() -> bool:
    """Return True when a working ngspice is on PATH."""
    version = ngspice_version()
    if version is None:
        if shutil.which("ngspice") is None:
            hint = INSTALL_HINT.get(platform.system(), INSTALL_HINT["Linux"])
            _err(
                "FAIL  ngspice: not found on PATH.\n"
                "      circuit-forge computes every numeric verdict from ngspice, "
                "so it cannot run without it.\n"
                f"      Fix ({platform.system()}): {hint}\n"
                "      Then re-run: python scripts/check_env.py"
            )
        return False
    print(f"ok    {version}")
    return True


def check_imports() -> bool:
    """Advisory check: report which Python dependencies are missing."""
    missing: list[str] = []
    with warnings.catch_warnings():
        # lcapy emits a pile of SyntaxWarnings from its docstrings on import.
        warnings.simplefilter("ignore")
        for mod in OPTIONAL_IMPORTS:
            try:
                __import__(mod)
            except ImportError:
                missing.append(mod)
    if missing:
        _err(
            "warn  python deps missing: " + ", ".join(missing) + "\n"
            "      Fix: uv pip install -e '.[dev]'   (or: pip install -e '.[dev]')"
        )
        return False
    print(f"ok    python deps ({len(OPTIONAL_IMPORTS)} modules import cleanly)")
    return True


def main() -> int:
    print("circuit-forge environment check")
    print("-" * 40)
    required = [check_python(), check_ngspice()]
    deps_ok = check_imports()
    print("-" * 40)
    if not all(required):
        _err("environment check FAILED - fix the items above before running cforge.")
        return 2
    if not deps_ok:
        print("environment usable, but install the Python dependencies before running cforge.")
        return 0
    print("environment OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
