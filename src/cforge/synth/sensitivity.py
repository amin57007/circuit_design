"""Monte-Carlo sensitivity of the measurements to component tolerances.

Each component is sampled from a **uniform** distribution across its tolerance
band, not a normal one.  That is deliberate and it matters: manufacturers sort
batches, so a reel of 1% resistors is not a narrow Gaussian around nominal --
it is closer to flat across the band, sometimes with the middle removed
entirely because those parts were sold as 0.1%.  Assuming normality would make
every pass rate here optimistic.

Each sample is a full ngspice run.  Runs are independent, so they are farmed
out to a :class:`~concurrent.futures.ProcessPoolExecutor`, with a rich progress
bar over the completed futures.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from cforge.catalog.loader import Pattern, load_pattern
from cforge.models import Component, SensitivityResult, Spec
from cforge.spice.runner import run as spice_run
from cforge.synth.solver import build_testbench

__all__ = ["MC_PASS_RATE_WARN", "monte_carlo", "sample_components"]

# A design whose Monte-Carlo pass rate is below this is flagged even when the
# nominal design passes: it will not survive production.
MC_PASS_RATE_WARN = 0.99

_SEED = 20240501


@dataclass(frozen=True)
class _Job:
    """Everything a worker process needs; must be picklable, so no Pattern."""

    pattern_dir: str
    spec_json: str
    values: dict[str, float]
    units: dict[str, str]
    ideal: dict[str, float]
    tolerances: dict[str, float]
    eseries: dict[str, str | None]
    models: dict[str, str | None]
    timeout_s: int


def sample_components(
    components: Sequence[Component], rng: np.random.Generator
) -> dict[str, float]:
    """One Monte-Carlo draw: each component uniform over its tolerance band.

    A component with zero tolerance keeps its nominal value exactly, which is
    the right model for a value the designer pins (a supply rail, a device
    that is not a passive).
    """
    out: dict[str, float] = {}
    for component in components:
        tolerance = component.tolerance_percent / 100.0
        if tolerance <= 0.0 or component.unit not in ("ohm", "F", "H"):
            out[component.ref] = component.value
            continue
        low = component.value * (1.0 - tolerance)
        high = component.value * (1.0 + tolerance)
        out[component.ref] = float(rng.uniform(low, high))
    return out


def _simulate_sample(job: _Job) -> dict[str, float]:
    """Worker entry point: rebuild the deck for one sample and run ngspice.

    Runs in a separate process, so it re-loads the pattern from disk.  Any
    failure returns an empty dict; the parent counts those as failed runs
    rather than letting one bad sample abort a 500-run sweep.
    """
    try:
        spec = Spec.model_validate_json(job.spec_json)
        pattern = load_pattern_cached(job.pattern_dir)
        components = [
            Component(
                ref=ref,
                value=value,
                unit=job.units.get(ref, ""),
                ideal_value=job.ideal.get(ref, value),
                tolerance_percent=job.tolerances.get(ref, 0.0),
                eseries=job.eseries.get(ref),  # type: ignore[arg-type]
                model=job.models.get(ref),
            )
            for ref, value in job.values.items()
        ]
        deck = build_testbench(spec, pattern, components)
        result = spice_run(deck, timeout_s=job.timeout_s, max_attempts=3)
    except Exception:
        return {}
    if not result.converged:
        return {}
    return dict(result.measurements)


_PATTERN_CACHE: dict[str, Pattern] = {}


def load_pattern_cached(directory: str) -> Pattern:
    """Load a pattern once per worker process; re-importing per sample is slow."""
    from pathlib import Path

    cached = _PATTERN_CACHE.get(directory)
    if cached is None:
        cached = load_pattern(Path(directory))
        _PATTERN_CACHE[directory] = cached
    return cached


def monte_carlo(
    spec: Spec,
    pattern: Pattern,
    components: Sequence[Component],
    n: int = 500,
    timeout_s: int = 60,
    jobs: int | None = None,
    console: Console | None = None,
    seed: int = _SEED,
) -> list[SensitivityResult]:
    """Run ``n`` tolerance samples and summarise every measured quantity.

    Returns one :class:`SensitivityResult` per measurement that at least one
    run produced, with percentiles and the pass rate against the spec's
    requirements for that measurement.  ``worst`` is the sampled value with the
    smallest requirement margin (or the extreme furthest from nominal when no
    requirement constrains it).
    """
    if n <= 0:
        return []

    console = console or Console()
    rng = np.random.default_rng(seed)
    samples = [sample_components(components, rng) for _ in range(n)]

    units = {c.ref: c.unit for c in components}
    ideal = {c.ref: c.ideal_value for c in components}
    tolerances = {c.ref: c.tolerance_percent for c in components}
    series = {c.ref: c.eseries for c in components}
    models = {c.ref: c.model for c in components}
    spec_json = spec.model_dump_json()

    jobs_list = [
        _Job(
            pattern_dir=str(pattern.directory),
            spec_json=spec_json,
            values=values,
            units=units,
            ideal=ideal,
            tolerances=tolerances,
            eseries=series,
            models=models,
            timeout_s=timeout_s,
        )
        for values in samples
    ]

    results = _execute(jobs_list, jobs, console)
    nominal = _nominal_measurements(spec, pattern, components, timeout_s)
    return _summarise(spec, results, nominal, n)


def _execute(
    jobs_list: Sequence[_Job], jobs: int | None, console: Console
) -> list[dict[str, float]]:
    """Run every sample, in parallel when it is worth it, with a progress bar."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    workers = jobs if jobs is not None else min(len(jobs_list), (os.cpu_count() or 2))
    workers = max(1, workers)

    collected: list[dict[str, float]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Monte Carlo (ngspice)", total=len(jobs_list))
        if workers == 1:
            for job in jobs_list:
                collected.append(_simulate_sample(job))
                progress.advance(task)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_simulate_sample, job) for job in jobs_list]
                for future in as_completed(futures):
                    collected.append(future.result())
                    progress.advance(task)
    return collected


def _nominal_measurements(
    spec: Spec, pattern: Pattern, components: Sequence[Component], timeout_s: int
) -> dict[str, float]:
    deck = build_testbench(spec, pattern, list(components))
    result = spice_run(deck, timeout_s=timeout_s)
    return dict(result.measurements)


def _summarise(
    spec: Spec,
    results: Sequence[dict[str, float]],
    nominal: dict[str, float],
    requested: int,
) -> list[SensitivityResult]:
    names = sorted({name for result in results for name in result} | set(nominal))
    failed_runs = sum(1 for result in results if not result)

    out: list[SensitivityResult] = []
    for name in names:
        values = np.asarray(
            [result[name] for result in results if name in result], dtype=float
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        requirements = spec.requirements_for(name)
        passing = _pass_mask(values, requirements)
        out.append(
            SensitivityResult(
                meas=name,
                nominal=float(nominal.get(name, float(np.median(values)))),
                p5=float(np.percentile(values, 5)),
                p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)),
                worst=_worst(values, requirements, float(nominal.get(name, 0.0))),
                pass_rate=float(passing.mean()) if requirements else 1.0,
                n_samples=int(values.size),
                n_failed_runs=failed_runs + (requested - len(results)),
                samples=[float(v) for v in values],
            )
        )
    return out


def _pass_mask(values: np.ndarray, requirements: Sequence[Any]) -> np.ndarray:
    """Boolean array: does each sample satisfy every requirement on this measurement?"""
    mask = np.ones(values.shape, dtype=bool)
    for req in requirements:
        if req.min is not None:
            mask &= values >= req.min
        if req.max is not None:
            mask &= values <= req.max
    return mask


def _worst(values: np.ndarray, requirements: Sequence[Any], nominal: float) -> float:
    """The sample closest to violating a requirement (or furthest from nominal)."""
    if not requirements:
        index = int(np.argmax(np.abs(values - nominal)))
        return float(values[index])

    margins = np.full(values.shape, np.inf)
    for req in requirements:
        if req.min is not None:
            scale = abs(req.min) or 1.0
            margins = np.minimum(margins, (values - req.min) / scale)
        if req.max is not None:
            scale = abs(req.max) or 1.0
            margins = np.minimum(margins, (req.max - values) / scale)
    return float(values[int(np.argmin(margins))])
