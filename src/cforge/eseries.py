"""IEC 60063 E-series standard values and the snapping logic built on them.

Everything here is decade-aware: a value is decomposed into ``mantissa * 10**k``
with ``1 <= mantissa < 10``, the mantissa is matched against the series table,
and the decade is reapplied.  All values are plain SI floats (ohms, farads,
henries); this module never sees engineering suffixes.

The mantissa tables are the canonical IEC values, not ``round(10**(n/N))``:
the standard deliberately deviates from the geometric ideal in several places
(E24 has 2.7/3.3/3.9, E12 has 2.7 not 2.6, and so on), and using the computed
values would silently produce parts nobody stocks.
"""

from __future__ import annotations

import math
from itertools import combinations_with_replacement

__all__ = [
    "E6",
    "E12",
    "E24",
    "E48",
    "E96",
    "E192",
    "SERIES",
    "decade_values",
    "series_combo",
    "snap",
    "snap_down",
    "snap_ratio",
    "snap_rc_pair",
    "snap_up",
]

E6: tuple[float, ...] = (1.0, 1.5, 2.2, 3.3, 4.7, 6.8)

E12: tuple[float, ...] = (1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2)

E24: tuple[float, ...] = (
    1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
    3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1,
)  # fmt: skip

E48: tuple[float, ...] = (
    1.00, 1.05, 1.10, 1.15, 1.21, 1.27, 1.33, 1.40, 1.47, 1.54, 1.62, 1.69,
    1.78, 1.87, 1.96, 2.05, 2.15, 2.26, 2.37, 2.49, 2.61, 2.74, 2.87, 3.01,
    3.16, 3.32, 3.48, 3.65, 3.83, 4.02, 4.22, 4.42, 4.64, 4.87, 5.11, 5.36,
    5.62, 5.90, 6.19, 6.49, 6.81, 7.15, 7.50, 7.87, 8.25, 8.66, 9.09, 9.53,
)  # fmt: skip

E96: tuple[float, ...] = (
    1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30,
    1.33, 1.37, 1.40, 1.43, 1.47, 1.50, 1.54, 1.58, 1.62, 1.65, 1.69, 1.74,
    1.78, 1.82, 1.87, 1.91, 1.96, 2.00, 2.05, 2.10, 2.15, 2.21, 2.26, 2.32,
    2.37, 2.43, 2.49, 2.55, 2.61, 2.67, 2.74, 2.80, 2.87, 2.94, 3.01, 3.09,
    3.16, 3.24, 3.32, 3.40, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12,
    4.22, 4.32, 4.42, 4.53, 4.64, 4.75, 4.87, 4.99, 5.11, 5.23, 5.36, 5.49,
    5.62, 5.76, 5.90, 6.04, 6.19, 6.34, 6.49, 6.65, 6.81, 6.98, 7.15, 7.32,
    7.50, 7.68, 7.87, 8.06, 8.25, 8.45, 8.66, 8.87, 9.09, 9.31, 9.53, 9.76,
)  # fmt: skip

E192: tuple[float, ...] = (
    1.00, 1.01, 1.02, 1.04, 1.05, 1.06, 1.07, 1.09, 1.10, 1.11, 1.13, 1.14,
    1.15, 1.17, 1.18, 1.20, 1.21, 1.23, 1.24, 1.26, 1.27, 1.29, 1.30, 1.32,
    1.33, 1.35, 1.37, 1.38, 1.40, 1.42, 1.43, 1.45, 1.47, 1.49, 1.50, 1.52,
    1.54, 1.56, 1.58, 1.60, 1.62, 1.64, 1.65, 1.67, 1.69, 1.72, 1.74, 1.76,
    1.78, 1.80, 1.82, 1.84, 1.87, 1.89, 1.91, 1.93, 1.96, 1.98, 2.00, 2.03,
    2.05, 2.08, 2.10, 2.13, 2.15, 2.18, 2.21, 2.23, 2.26, 2.29, 2.32, 2.34,
    2.37, 2.40, 2.43, 2.46, 2.49, 2.52, 2.55, 2.58, 2.61, 2.64, 2.67, 2.71,
    2.74, 2.77, 2.80, 2.84, 2.87, 2.91, 2.94, 2.98, 3.01, 3.05, 3.09, 3.12,
    3.16, 3.20, 3.24, 3.28, 3.32, 3.36, 3.40, 3.44, 3.48, 3.52, 3.57, 3.61,
    3.65, 3.70, 3.74, 3.79, 3.83, 3.88, 3.92, 3.97, 4.02, 4.07, 4.12, 4.17,
    4.22, 4.27, 4.32, 4.37, 4.42, 4.48, 4.53, 4.59, 4.64, 4.70, 4.75, 4.81,
    4.87, 4.93, 4.99, 5.05, 5.11, 5.17, 5.23, 5.30, 5.36, 5.42, 5.49, 5.56,
    5.62, 5.69, 5.76, 5.83, 5.90, 5.97, 6.04, 6.12, 6.19, 6.26, 6.34, 6.42,
    6.49, 6.57, 6.65, 6.73, 6.81, 6.90, 6.98, 7.06, 7.15, 7.23, 7.32, 7.41,
    7.50, 7.59, 7.68, 7.77, 7.87, 7.96, 8.06, 8.16, 8.25, 8.35, 8.45, 8.56,
    8.66, 8.76, 8.87, 8.98, 9.09, 9.20, 9.31, 9.42, 9.53, 9.65, 9.76, 9.88,
)  # fmt: skip

SERIES: dict[str, tuple[float, ...]] = {
    "E6": E6,
    "E12": E12,
    "E24": E24,
    "E48": E48,
    "E96": E96,
    "E192": E192,
}

_REL_EPS = 1e-9


def _table(series: str) -> tuple[float, ...]:
    """Look up a mantissa table, raising an actionable error on a bad name."""
    key = series.strip().upper()
    try:
        return SERIES[key]
    except KeyError:
        raise ValueError(
            f"unknown E-series {series!r}; choose one of {', '.join(SERIES)}"
        ) from None


def _check_positive(value: float, what: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{what} must be a finite positive number, got {value!r}")


def _decompose(value: float) -> tuple[float, int]:
    """Split ``value`` into ``(mantissa, decade)`` with ``1 <= mantissa < 10``.

    Guards against floating-point results like 9.999999999 for an exact 1e-8,
    which would otherwise snap to the top of the wrong decade.
    """
    decade = math.floor(math.log10(value))
    mantissa = value / 10.0**decade
    if mantissa >= 10.0 - 1e-12:
        mantissa /= 10.0
        decade += 1
    elif mantissa < 1.0 - 1e-12:
        mantissa *= 10.0
        decade -= 1
    return mantissa, decade


def decade_values(series: str, lo: float, hi: float) -> list[float]:
    """Every standard value of ``series`` in the closed range [lo, hi], ascending.

    Units are whatever the caller uses; ``lo`` and ``hi`` must be positive.
    """
    _check_positive(lo, "lo")
    _check_positive(hi, "hi")
    if lo > hi:
        raise ValueError(f"lo ({lo}) must not exceed hi ({hi})")
    table = _table(series)
    out: list[float] = []
    start = math.floor(math.log10(lo))
    stop = math.floor(math.log10(hi))
    for decade in range(start - 1, stop + 2):
        scale = 10.0**decade
        for m in table:
            v = m * scale
            if lo * (1 - _REL_EPS) <= v <= hi * (1 + _REL_EPS):
                out.append(v)
    return sorted(out)


def _candidates(value: float, series: str) -> tuple[float, float]:
    """Return the standard values immediately at-or-below and at-or-above ``value``."""
    table = _table(series)
    mantissa, decade = _decompose(value)
    scale = 10.0**decade
    below = table[-1] * scale / 10.0
    above = table[0] * scale * 10.0
    for m in table:
        if m <= mantissa * (1 + _REL_EPS):
            below = m * scale
        if m >= mantissa * (1 - _REL_EPS):
            above = m * scale
            break
    return below, above


def snap(value: float, series: str = "E96") -> float:
    """Nearest standard value of ``series``, chosen on log distance.

    Log distance (not linear) is the right metric: the series is geometric, so
    the nearest-in-ratio value is what a designer would actually pick.

    >>> snap(4870.0, "E24")
    4700.0
    """
    _check_positive(value, "value")
    below, above = _candidates(value, series)
    if math.isclose(below, above, rel_tol=_REL_EPS):
        return below
    if abs(math.log(value / below)) <= abs(math.log(above / value)):
        return below
    return above


def snap_up(value: float, series: str = "E96") -> float:
    """Smallest standard value >= ``value``."""
    _check_positive(value, "value")
    below, above = _candidates(value, series)
    if math.isclose(below, value, rel_tol=_REL_EPS):
        return below
    return above


def snap_down(value: float, series: str = "E96") -> float:
    """Largest standard value <= ``value``."""
    _check_positive(value, "value")
    below, above = _candidates(value, series)
    if math.isclose(above, value, rel_tol=_REL_EPS):
        return above
    return below


def neighbours(value: float, series: str = "E96", window: int = 3) -> list[float]:
    """Standard values within +/- ``window`` positions of ``snap(value)``.

    Used by the joint-snapping searches.  Crosses decade boundaries correctly.
    """
    _check_positive(value, "value")
    if window < 0:
        raise ValueError(f"window must be >= 0, got {window}")
    table = _table(series)
    n = len(table)
    centre = snap(value, series)
    mantissa, decade = _decompose(centre)
    idx = min(range(n), key=lambda i: abs(table[i] - mantissa))
    out: list[float] = []
    for offset in range(-window, window + 1):
        j = idx + offset
        dec = decade + math.floor(j / n)
        out.append(table[j % n] * 10.0**dec)
    return sorted(set(out))


def snap_ratio(
    r_hi: float,
    r_lo: float,
    series: str = "E96",
    window: int = 3,
) -> tuple[float, float]:
    """Jointly snap a resistor pair so that ``r_hi / r_lo`` is best preserved.

    Independent snapping optimises each resistor's absolute error, which is the
    wrong objective for a divider or a gain-setting network where only the
    ratio matters.  This searches +/- ``window`` positions around each
    independent snap and returns the pair minimising relative ratio error.

    Returns ``(r_hi_snapped, r_lo_snapped)`` in the same unit as the inputs.
    """
    _check_positive(r_hi, "r_hi")
    _check_positive(r_lo, "r_lo")
    target = r_hi / r_lo
    best: tuple[float, float] | None = None
    best_err = math.inf
    hi_options = neighbours(r_hi, series, window)
    lo_options = neighbours(r_lo, series, window)
    for a in hi_options:
        for b in lo_options:
            err = abs(math.log((a / b) / target))
            if err < best_err - 1e-15:
                best_err = err
                best = (a, b)
    if best is None:  # pragma: no cover - neighbours() is never empty
        raise RuntimeError("snap_ratio found no candidate pair")
    return best


def ratio_error(pair: tuple[float, float], target_ratio: float) -> float:
    """Relative error of ``pair[0]/pair[1]`` against ``target_ratio``, in percent."""
    return 100.0 * (pair[0] / pair[1] - target_ratio) / target_ratio


def series_combo(
    target: float,
    series: str = "E24",
    max_parts: int = 2,
    tolerance_percent: float = 1.0,
    kind: str = "resistor",
) -> list[float]:
    """Combination of standard values whose net value hits ``target``.

    ``kind`` selects how parts combine: ``"resistor"`` tries series (sum) first
    then parallel; ``"capacitor"`` tries parallel (sum) first then series.  The
    search starts at one part and grows to ``max_parts``, returning the first
    (fewest-parts, then lowest-error) combination inside ``tolerance_percent``.
    If nothing meets the tolerance, the best combination found is returned
    anyway so the caller can warn rather than fail.

    Returns the individual part values, in the same unit as ``target``.
    """
    _check_positive(target, "target")
    if max_parts < 1:
        raise ValueError(f"max_parts must be >= 1, got {max_parts}")
    if kind not in ("resistor", "capacitor"):
        raise ValueError(f"kind must be 'resistor' or 'capacitor', got {kind!r}")

    def net(parts: tuple[float, ...], mode: str) -> float:
        if mode == "sum":
            return math.fsum(parts)
        return 1.0 / math.fsum(1.0 / p for p in parts)

    # "sum" is series for resistors and parallel for capacitors.
    modes = ("sum", "recip")
    pool = decade_values(series, target / 100.0, target * 100.0)
    best: list[float] = [snap(target, series)]
    best_err = abs(100.0 * (best[0] - target) / target)
    if best_err <= tolerance_percent:
        return best

    for n in range(2, max_parts + 1):
        # A useful n-part combination always has every part within a couple of
        # decades of the target; trimming the pool keeps this tractable.
        window = [v for v in pool if target / 1000.0 <= v <= target * 1000.0]
        for mode in modes:
            for combo in combinations_with_replacement(window, n):
                value = net(combo, mode)
                err = abs(100.0 * (value - target) / target)
                if err < best_err - 1e-12:
                    best_err = err
                    best = sorted(combo)
        if best_err <= tolerance_percent:
            break
    return best


def snap_rc_pair(
    target_tau: float,
    c_series: str = "E6",
    r_series: str = "E96",
    c_range: tuple[float, float] = (1e-12, 1e-6),
    r_range: tuple[float, float] = (1e3, 1e5),
) -> tuple[float, float]:
    """Pick a real (R, C) pair realising the time constant ``target_tau`` seconds.

    This encodes what a designer actually does: capacitors come in coarse
    values and wide tolerances, resistors are cheap and available in E96, so
    fix C to a coarse standard value and solve R = tau / C against the fine
    series.  Candidate capacitors are ranked by the resulting error in tau,
    with a mild preference for resistors in a sane impedance window
    (``r_range``, ohms; default 1 k-ohm .. 100 k-ohm) so the result is neither
    noise-dominated nor a load the driving stage cannot swing.

    Returns ``(R_ohms, C_farads)``.
    """
    _check_positive(target_tau, "target_tau")
    c_lo, c_hi = c_range
    _check_positive(c_lo, "c_range[0]")
    _check_positive(c_hi, "c_range[1]")
    r_lo, r_hi = r_range
    caps = decade_values(c_series, c_lo, c_hi)
    if not caps:
        raise ValueError(
            f"no {c_series} capacitor values in range {c_lo} .. {c_hi} F; widen c_range"
        )

    best: tuple[float, float] | None = None
    best_score = math.inf
    for c in caps:
        r_ideal = target_tau / c
        r = snap(r_ideal, r_series)
        err = abs(math.log((r * c) / target_tau))
        penalty = 0.0
        if r < r_lo:
            penalty = math.log(r_lo / r)
        elif r > r_hi:
            penalty = math.log(r / r_hi)
        score = err + 0.05 * penalty
        if score < best_score - 1e-15:
            best_score = score
            best = (r, c)
    if best is None:  # pragma: no cover - caps is non-empty here
        raise RuntimeError("snap_rc_pair found no candidate pair")
    return best
