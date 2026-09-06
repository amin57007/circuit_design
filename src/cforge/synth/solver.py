"""Spec -> ideal values -> snapped, realisable component values.

The pipeline, in order:

1. Call the pattern's ``design.solve(spec)`` for ideal values (SI base units).
2. Snap them to real parts.  Snapping is driven by a *plan* the pattern
   optionally supplies through ``design.snap_plan(spec, ideal)``: ``rc_tau``
   groups go through :func:`cforge.eseries.snap_rc_pair`, ``ratio`` groups
   through :func:`cforge.eseries.snap_ratio`, and anything unclaimed is snapped
   independently.  Joint snapping matters because a divider or an RC product
   cares about a relationship, not about each part's individual error.
3. Recompute the analytical response with the snapped values (via the pattern's
   ``design.response``) and warn when a requirement has given up more than half
   its nominal margin to snapping.
4. If the pattern declares ``numeric_refine: true``, search E-series *indices*
   with ``scipy.optimize.differential_evolution``, scoring each candidate with
   a real ngspice run.

Nothing here calls a language model.  Step 4 is the only stochastic element and
it is seeded, so runs reproduce.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from cforge import eseries
from cforge.catalog.loader import Pattern, PatternError
from cforge.models import Component, ESeriesName, Spec

__all__ = ["SynthesisResult", "build_netlist", "build_testbench", "synthesize"]

# Component ref prefix -> (unit, default E-series role).
_UNIT_BY_PREFIX: dict[str, str] = {
    "R": "ohm",
    "C": "F",
    "L": "H",
    "M": "device",
    "Q": "device",
    "D": "device",
    "X": "device",
}


@dataclass
class SynthesisResult:
    """Snapped components plus everything learned on the way there."""

    components: list[Component]
    ideal: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    gotchas: list[dict[str, str]] = field(default_factory=list)
    iterations: int = 1
    context: dict[str, Any] = field(default_factory=dict)

    def by_ref(self) -> dict[str, Component]:
        return {c.ref: c for c in self.components}

    def values(self) -> dict[str, float]:
        return {c.ref: c.value for c in self.components}


def unit_for(ref: str) -> str:
    """SI unit implied by a component ref prefix, e.g. ``R1`` -> ``ohm``."""
    return _UNIT_BY_PREFIX.get(ref[0].upper(), "") if ref else ""


def synthesize(
    spec: Spec,
    pattern: Pattern,
    refine: bool = True,
    simulate: Callable[[str], dict[str, float]] | None = None,
) -> SynthesisResult:
    """Turn ``spec`` into realisable components for ``pattern``.

    ``simulate`` is the ngspice callback used by numeric refinement; when it is
    None (or the pattern does not request refinement) steps 1-3 run alone.
    """
    ideal = pattern.solve(spec)
    units = _units_for(pattern, ideal)
    warnings: list[str] = []

    snapped = _snap_all(spec, pattern, ideal, units, warnings)
    components = _components(spec, ideal, snapped, units, pattern)

    warnings.extend(_margin_warnings(spec, pattern, ideal, snapped))
    gotchas = _run_gotchas(spec, pattern, components)

    iterations = 1
    if refine and pattern.meta.numeric_refine and simulate is not None:
        refined, iterations = _numeric_refine(
            spec, pattern, snapped, units, simulate, warnings
        )
        if refined is not None:
            components = _components(spec, ideal, refined, units, pattern)
            gotchas = _run_gotchas(spec, pattern, components)

    return SynthesisResult(
        components=components,
        ideal=ideal,
        warnings=warnings,
        gotchas=gotchas,
        iterations=iterations,
    )


def _units_for(pattern: Pattern, ideal: dict[str, float]) -> dict[str, str]:
    declared = getattr(pattern.design_module, "UNITS", {}) or {}
    return {ref: str(declared.get(ref, unit_for(ref))) for ref in ideal}


def _snap_plan(pattern: Pattern, spec: Spec, ideal: dict[str, float]) -> list[dict[str, Any]]:
    maker = getattr(pattern.design_module, "snap_plan", None)
    if maker is None:
        return []
    plan = maker(spec, ideal)
    if plan is None:
        return []
    if not isinstance(plan, list):
        raise PatternError(
            f"{pattern.directory / 'design.py'}: snap_plan() must return a list of dicts, "
            f"got {type(plan).__name__}"
        )
    return plan


def _snap_all(
    spec: Spec,
    pattern: Pattern,
    ideal: dict[str, float],
    units: dict[str, str],
    warnings: list[str],
) -> dict[str, float]:
    """Apply the pattern's snapping plan, then snap whatever it did not claim."""
    out: dict[str, float] = {}
    claimed: set[str] = set()

    for entry in _snap_plan(pattern, spec, ideal):
        kind = str(entry.get("kind", "")).lower()
        if kind == "rc_tau":
            _snap_rc(spec, entry, ideal, out, claimed, warnings)
        elif kind == "ratio":
            _snap_pair_ratio(spec, entry, ideal, out, claimed)
        elif kind == "fixed":
            ref = str(entry["ref"])
            out[ref] = float(entry.get("value", ideal[ref]))
            claimed.add(ref)
        elif kind == "single":
            ref = str(entry["ref"])
            out[ref] = eseries.snap(ideal[ref], str(entry.get("series", spec.eseries)))
            claimed.add(ref)
        else:
            raise PatternError(
                f"{pattern.directory / 'design.py'}: snap_plan() returned unknown kind "
                f"{kind!r}; expected one of rc_tau, ratio, single, fixed"
            )

    for ref, value in ideal.items():
        if ref in claimed:
            continue
        if units.get(ref) == "device" or value <= 0.0:
            out[ref] = value
            continue
        out[ref] = eseries.snap(value, _series_for(spec, ref, units))
    return out


def _series_for(spec: Spec, ref: str, units: dict[str, str]) -> str:
    """E-series to use for a ref: capacitors default coarser than resistors.

    Real capacitors are simply not available in E96, so snapping one to E96
    would produce a BOM nobody can buy.
    """
    if units.get(ref) == "F":
        coarse = str(spec.params.get("c_eseries", "E6"))
        return coarse
    return spec.eseries


def _snap_rc(
    spec: Spec,
    entry: dict[str, Any],
    ideal: dict[str, float],
    out: dict[str, float],
    claimed: set[str],
    warnings: list[str],
) -> None:
    r_ref, c_ref = str(entry["r"]), str(entry["c"])
    tau = float(entry["tau"])
    c_series = str(entry.get("c_series", "E6"))
    r_series = str(entry.get("r_series", spec.eseries))
    c_range = tuple(entry.get("c_range", (1e-12, 1e-6)))
    r_ohm, c_f = eseries.snap_rc_pair(
        tau,
        c_series=c_series,
        r_series=r_series,
        c_range=(float(c_range[0]), float(c_range[1])),
    )
    out[r_ref], out[c_ref] = r_ohm, c_f
    claimed.update({r_ref, c_ref})
    # The capacitor is *chosen* from a coarse grid rather than approximated, so
    # reporting it as a deviation from design.py's placeholder is misleading.
    # Restate the ideal pair as "this capacitor, and the resistor it demands":
    # the BOM then shows the real snapping error, which lives entirely in R.
    ideal[c_ref] = c_f
    ideal[r_ref] = tau / c_f
    error_percent = 100.0 * (r_ohm * c_f - tau) / tau
    if abs(error_percent) > 2.0:
        warnings.append(
            f"snapping {r_ref}/{c_ref} to standard values moved the time constant by "
            f"{error_percent:+.2f}% (target {tau:.4g} s). Widen c_range or use a finer "
            f"capacitor series than {c_series}."
        )


def _snap_pair_ratio(
    spec: Spec,
    entry: dict[str, Any],
    ideal: dict[str, float],
    out: dict[str, float],
    claimed: set[str],
) -> None:
    hi_ref, lo_ref = str(entry["hi"]), str(entry["lo"])
    series = str(entry.get("series", spec.eseries))
    window = int(entry.get("window", 3))
    hi, lo = eseries.snap_ratio(ideal[hi_ref], ideal[lo_ref], series, window)
    out[hi_ref], out[lo_ref] = hi, lo
    claimed.update({hi_ref, lo_ref})


def _components(
    spec: Spec,
    ideal: dict[str, float],
    snapped: dict[str, float],
    units: dict[str, str],
    pattern: Pattern,
) -> list[Component]:
    models: dict[str, str] = getattr(pattern.design_module, "MODELS", {}) or {}
    components: list[Component] = []
    for ref in sorted(ideal, key=_ref_sort_key):
        unit = units.get(ref, unit_for(ref))
        components.append(
            Component(
                ref=ref,
                value=snapped[ref],
                unit=unit,
                ideal_value=ideal[ref],
                tolerance_percent=spec.tolerance_for(ref),
                eseries=_eseries_name(spec, ref, units) if unit in ("ohm", "F", "H") else None,
                model=models.get(ref),
            )
        )
    return components


def _eseries_name(spec: Spec, ref: str, units: dict[str, str]) -> ESeriesName | None:
    name = _series_for(spec, ref, units).upper()
    return name if name in eseries.SERIES else None  # type: ignore[return-value]


def _ref_sort_key(ref: str) -> tuple[str, int, str]:
    prefix = ref[0].upper() if ref else ""
    digits = "".join(ch for ch in ref[1:] if ch.isdigit())
    return (prefix, int(digits) if digits else 0, ref)


def _margin_warnings(
    spec: Spec,
    pattern: Pattern,
    ideal: dict[str, float],
    snapped: dict[str, float],
) -> list[str]:
    """Warn when snapping ate more than half of a requirement's nominal margin."""
    response = getattr(pattern.design_module, "response", None)
    if response is None:
        return []
    try:
        before = response(spec, ideal)
        after = response(spec, snapped)
    except Exception as exc:
        return [
            f"analytical response check skipped: {pattern.directory / 'design.py'} "
            f"response() raised {type(exc).__name__}: {exc}"
        ]

    warnings: list[str] = []
    for req in spec.requirements:
        if req.meas not in before or req.meas not in after:
            continue
        _, margin_before = req.check(before[req.meas])
        _, margin_after = req.check(after[req.meas])
        if margin_before is None or margin_after is None:
            continue
        if margin_after < 0.0:
            warnings.append(
                f"{req.id}: snapping to standard values pushed the analytical "
                f"{req.meas} to {after[req.meas]:.4g} {req.unit}, outside "
                f"{req.limit_text()}. ngspice has the final say, but expect a FAIL."
            )
        elif margin_before > 0.0 and margin_after < 0.5 * margin_before:
            warnings.append(
                f"{req.id}: snapping consumed {100.0 * (1 - margin_after / margin_before):.0f}% "
                f"of the margin on {req.meas} ({margin_before:.1f}% -> {margin_after:.1f}%). "
                "Consider a finer E-series or a pinned capacitor value."
            )
    return warnings


def _run_gotchas(spec: Spec, pattern: Pattern, components: list[Component]) -> list[dict[str, str]]:
    """Evaluate the pattern's machine-checkable gotchas."""
    triggered: list[dict[str, str]] = []
    context: dict[str, Any] = {"pattern": pattern, "values": {c.ref: c.value for c in components}}
    for gotcha in pattern.meta.gotchas:
        if not gotcha.check:
            continue
        fn = pattern.gotcha_check(gotcha.check)
        if fn is None:
            continue
        try:
            message = fn(spec, components, context)
        except Exception as exc:
            triggered.append(
                {
                    "id": gotcha.id,
                    "text": gotcha.text,
                    "detail": (
                        f"check {gotcha.check!r} in {pattern.directory / 'design.py'} raised "
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )
            continue
        if message:
            triggered.append({"id": gotcha.id, "text": gotcha.text, "detail": str(message)})
    return triggered


def build_netlist(spec: Spec, pattern: Pattern, components: Sequence[Component]) -> str:
    """Render the pattern's topology with the chosen component values."""
    context = _template_context(spec, pattern, components)
    return pattern.render_netlist(context)


def build_testbench(spec: Spec, pattern: Pattern, components: Sequence[Component]) -> str:
    """Render the full simulatable deck: models, topology, sources, analyses, .meas."""
    context = _template_context(spec, pattern, components)
    context["netlist"] = pattern.render_netlist(context).rstrip("\n")
    return pattern.render_testbench(context)


def _template_context(
    spec: Spec, pattern: Pattern, components: Sequence[Component]
) -> dict[str, Any]:
    from cforge.spice import modellib

    values = {c.ref: c.value for c in components}
    context: dict[str, Any] = {
        "spec": spec,
        "params": dict(spec.params),
        "components": {c.ref: c for c in components},
        "values": values,
        "pattern": pattern.meta,
        "models": modellib.resolve_many(pattern.meta.models) if pattern.meta.models else "",
    }
    extra = getattr(pattern.design_module, "testbench_context", None)
    if extra is not None:
        context.update(extra(spec, values))
    return context


def _numeric_refine(
    spec: Spec,
    pattern: Pattern,
    start: dict[str, float],
    units: dict[str, str],
    simulate: Callable[[str], dict[str, float]],
    warnings: list[str],
) -> tuple[dict[str, float] | None, int]:
    """Search E-series indices with differential evolution, scored by ngspice.

    The search space is deliberately *discrete*: each optimisable component is
    an integer offset from its snapped index in its own E-series table, bounded
    to +/- ``_REFINE_WINDOW`` positions.  Optimising continuous values would
    produce parts nobody stocks and would then have to be re-snapped, undoing
    the optimisation.
    """
    from scipy.optimize import differential_evolution

    refs = [r for r in sorted(start) if units.get(r) in ("ohm", "F", "H")]
    if not refs:
        return None, 1

    tables = {r: eseries.SERIES[_series_for(spec, r, units).upper()] for r in refs}
    baseline = {r: start[r] for r in refs}
    cache: dict[str, float] = {}
    evaluations = 0

    def candidate_values(offsets: Sequence[float]) -> dict[str, float]:
        values = dict(start)
        for ref, raw in zip(refs, offsets, strict=True):
            values[ref] = _offset_value(baseline[ref], tables[ref], int(round(raw)))
        return values

    def objective(offsets: Sequence[float]) -> float:
        nonlocal evaluations
        values = candidate_values(offsets)
        key = _values_key(spec, pattern, values)
        if key in cache:
            return cache[key]
        evaluations += 1
        components = _components(spec, start, values, units, pattern)
        try:
            measurements = simulate(build_testbench(spec, pattern, components))
        except Exception as exc:  # a candidate that will not simulate is simply bad
            warnings.append(f"numeric refinement skipped a candidate: {exc}")
            cache[key] = _BAD_SCORE
            return _BAD_SCORE
        score = _violation(spec, measurements)
        cache[key] = score
        return score

    bounds = [(-_REFINE_WINDOW, _REFINE_WINDOW)] * len(refs)
    result = differential_evolution(
        objective,
        bounds,
        seed=_SEED,
        maxiter=_REFINE_MAXITER,
        popsize=_REFINE_POPSIZE,
        tol=1e-6,
        polish=False,
        integrality=[True] * len(refs),
    )
    if not result.success and result.fun >= _BAD_SCORE:
        warnings.append(
            "numeric refinement found no simulatable candidate; keeping the snapped design"
        )
        return None, max(1, evaluations)
    if _violation(spec, simulate(build_testbench(spec, pattern, _components(
        spec, start, baseline, units, pattern
    )))) <= result.fun:
        return None, max(1, evaluations)
    return candidate_values(result.x), max(1, evaluations)


_REFINE_WINDOW = 2
_REFINE_MAXITER = 12
_REFINE_POPSIZE = 6
_SEED = 20240501
_BAD_SCORE = 1.0e6


def _offset_value(value: float, table: Sequence[float], offset: int) -> float:
    """Move ``value`` ``offset`` positions along its E-series, crossing decades."""
    n = len(table)
    decade = math.floor(math.log10(value))
    mantissa = value / 10.0**decade
    index = min(range(n), key=lambda i: abs(math.log(table[i] / mantissa)))
    j = index + offset
    return table[j % n] * 10.0 ** (decade + math.floor(j / n))


def _violation(spec: Spec, measurements: dict[str, float]) -> float:
    """Weighted requirement violation; 0.0 when everything passes with margin."""
    total = 0.0
    for req in spec.requirements:
        value = measurements.get(req.meas)
        if value is None:
            total += 100.0
            continue
        _, margin = req.check(value)
        if margin is None:
            total += 100.0
        elif margin < 0.0:
            total += margin * margin
    return total


def _values_key(spec: Spec, pattern: Pattern, values: dict[str, float]) -> str:
    """Cache key over (spec, pattern id, candidate values)."""
    blob = json.dumps(
        {
            "spec": spec.hash(),
            "pattern": pattern.id,
            "values": {k: f"{v:.12g}" for k, v in sorted(values.items())},
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
