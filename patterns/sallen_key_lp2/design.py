"""Synthesis of a unity-gain Sallen-Key second-order low-pass.

All values are SI base units: ohms, farads, hertz.

Topology (unity-gain / voltage-follower form)::

    in --[R1]--+--[R2]--+-----+
               |        |     |  +
             [C1]     [C2]    +--|>--+-- out
               |        |     |  -   |
               +--------|-----|------+   (op-amp output ties back to its
                        |     |          inverting input and to C1)
                       GND

Transfer function with an ideal op-amp:

    H(s) = 1 / (1 + s*C2*(R1 + R2) + s^2 * R1*R2*C1*C2)

so, comparing with ``1 + s/(Q*w0) + s^2/w0^2``:

    w0 = 1 / sqrt(R1*R2*C1*C2)                       [rad/s]
    Q  = sqrt(R1*R2*C1*C2) / (C2 * (R1 + R2))        [-]

Two facts drive the whole design procedure:

*   Writing ``r = R1/R2`` and ``g = sqrt(R1*R2)``, ``w0`` depends only on ``g``
    and ``Q`` only on ``r`` (given the capacitors).  The two specifications are
    therefore independent knobs, which is why the resistor pair is snapped with
    :func:`cforge.eseries.snap_ratio`: preserving ``R1/R2`` preserves Q exactly.
*   Real resistors exist only if ``C1/C2 >= 4*Q^2``.  For a Butterworth
    (Q = 0.7071) that means ``C1 >= 2*C2``.  This is why the *unity-gain* form
    needs unequal capacitors: with C1 = C2 the maximum achievable Q is 0.5.
    (The equal-capacitor Sallen-Key reaches higher Q only by taking gain
    ``K = 3 - 1/Q``, which would violate a unity-gain passband requirement.)

The target polynomial comes from ``scipy.signal`` rather than a hand-typed
table, so bessel and chebyshev responses work without new equations.
"""

from __future__ import annotations

import math
from typing import Any

from cforge import eseries

TWO_PI = 2.0 * math.pi

UNITS: dict[str, str] = {"R1": "ohm", "R2": "ohm", "C1": "F", "C2": "F"}
MODELS: dict[str, str] = {"X1": "OPAMP_GENERIC"}

# Gain-bandwidth product in Hz of the op-amp macromodels in models/generic.lib.
# A spec may override this with params.gbw_hz when using a vendor model.
_MODEL_GBW_HZ: dict[str, float] = {
    "OPAMP_GENERIC": 1.0e6,
    "OPAMP_FAST": 1.0e7,
}

# Capacitors are chosen from a coarse series and must stay in a buildable range.
_C_SERIES = "E6"
_C_RANGE = (100e-12, 1e-6)
# Resistors outside this window are penalised: too low loads the source and the
# op-amp, too high makes thermal noise and stray capacitance significant.
_R_RANGE = (1e3, 3e5)
# Headroom on the C1/C2 >= 4*Q^2 existence condition. Sitting exactly on the
# boundary gives R1 == R2 and a discriminant of zero, which is numerically
# fragile once the values are snapped.
_C_RATIO_MARGIN = 1.08


def target_w0_q(spec: Any) -> tuple[float, float]:
    """Target ``(w0 [rad/s], Q [-])`` from the requested response family.

    The denominator polynomial is obtained from scipy so that adding bessel or
    chebyshev support needs no new algebra here.
    """
    import numpy as np
    from scipy import signal

    fc_hz = spec.num_param("fc_hz")
    order = int(spec.num_param("order")) if "order" in spec.params else 2
    if order != 2:
        raise ValueError(
            f"spec {spec.name!r}: sallen_key_lp2 realises exactly one second-order "
            f"section, but order={order} was requested. Cascade stages for higher orders."
        )
    response = spec.str_param("response", "butterworth")
    w_target = TWO_PI * fc_hz

    if response in ("butterworth", "butter"):
        _b, a = signal.butter(2, w_target, btype="low", analog=True)
    elif response == "bessel":
        _b, a = signal.bessel(2, w_target, btype="low", analog=True, norm="mag")
    elif response in ("chebyshev", "cheby1", "chebyshev1"):
        ripple_db = float(spec.params.get("ripple_db", 0.5))
        _b, a = signal.cheby1(2, ripple_db, w_target, btype="low", analog=True)
    else:
        raise ValueError(
            f"spec {spec.name!r}: unsupported response {response!r} for sallen_key_lp2; "
            "use butterworth, bessel or chebyshev"
        )

    a = np.asarray(a, dtype=float)
    # a = [a2, a1, a0] for a2*s^2 + a1*s + a0
    w0 = math.sqrt(a[2] / a[0])
    q = math.sqrt(a[2] * a[0]) / a[1]
    return w0, q


def resistors_for(w0: float, q: float, c1: float, c2: float) -> tuple[float, float] | None:
    """Exact ``(R1, R2)`` in ohms realising ``w0`` and ``q`` with the given caps.

    Returns None when ``C1/C2 < 4*Q^2``, i.e. no real resistor pair exists.
    """
    sum_r = 1.0 / (q * w0 * c2)
    product_r = 1.0 / (w0 * w0 * c1 * c2)
    discriminant = sum_r * sum_r - 4.0 * product_r
    if discriminant < 0.0:
        return None
    root = math.sqrt(discriminant)
    return (sum_r + root) / 2.0, (sum_r - root) / 2.0


def realised(r1: float, r2: float, c1: float, c2: float) -> tuple[float, float]:
    """``(w0 [rad/s], Q [-])`` actually produced by a set of component values."""
    w0 = 1.0 / math.sqrt(r1 * r2 * c1 * c2)
    q = math.sqrt(r1 * r2 * c1 * c2) / (c2 * (r1 + r2))
    return w0, q


def solve(spec: Any) -> dict[str, float]:
    """Choose C1, C2, R1, R2 for the requested corner and response.

    The capacitor pair is searched over the coarse standard series; for each
    candidate the exact resistors are computed and then jointly snapped with
    :func:`snap_ratio`, and the pair whose *snapped* result best matches both
    w0 and Q wins.  Evaluating candidates after snapping (rather than before)
    is the whole point: a capacitor choice that looks ideal can snap badly.
    """
    w0, q = target_w0_q(spec)
    r_series = getattr(spec, "eseries", "E96")
    c_series = str(spec.params.get("c_eseries", _C_SERIES))
    caps = eseries.decade_values(c_series, *_C_RANGE)
    min_ratio = 4.0 * q * q * _C_RATIO_MARGIN

    best: dict[str, float] | None = None
    best_score = math.inf

    for c2 in caps:
        for c1 in caps:
            if c1 / c2 < min_ratio:
                continue
            exact = resistors_for(w0, q, c1, c2)
            if exact is None:
                continue
            r1_ideal, r2_ideal = exact
            if not (_R_RANGE[0] / 10 <= r2_ideal and r1_ideal <= _R_RANGE[1] * 10):
                continue
            r1, r2 = eseries.snap_ratio(r1_ideal, r2_ideal, r_series, window=3)
            score = _score(r1, r2, c1, c2, w0, q)
            if score < best_score - 1e-12:
                best_score = score
                best = {"R1": r1_ideal, "R2": r2_ideal, "C1": c1, "C2": c2}

    if best is None:
        raise ValueError(
            f"spec {spec.name!r}: no buildable Sallen-Key found for fc={spec.num_param('fc_hz')} Hz "
            f"and Q={q:.3f}. The unity-gain form needs C1/C2 >= {4 * q * q:.2f} with both "
            f"capacitors in {_C_RANGE[0]:g}..{_C_RANGE[1]:g} F and sane resistors; widen "
            "c_eseries or move fc."
        )
    return best


def _score(r1: float, r2: float, c1: float, c2: float, w0: float, q: float) -> float:
    """Combined w0/Q error of a snapped candidate, plus practicality penalties.

    w0 is weighted more heavily than Q because corner frequency is almost
    always the tighter requirement, and Q errors of a few percent are invisible
    in a Butterworth passband.

    The spread penalty is what keeps the answer sane.  Any capacitor ratio at
    or above ``4*Q^2`` hits w0 and Q *exactly*, so error alone cannot choose
    between C1/C2 = 2.2 and C1/C2 = 21.  The latter is a much worse circuit:
    the resistors then differ by 40x, which means one of them is up near
    200 k-ohm contributing thermal noise and picking up stray capacitance
    while the other loads the source.  Keeping R1/R2 close to 1 is the
    minimum-sensitivity choice.
    """
    w0_actual, q_actual = realised(r1, r2, c1, c2)
    w0_error = abs(math.log(w0_actual / w0))
    q_error = abs(math.log(q_actual / q))
    spread = abs(math.log(r1 / r2))
    penalty = 0.0
    for r in (r1, r2):
        if r < _R_RANGE[0]:
            penalty += math.log(_R_RANGE[0] / r)
        elif r > _R_RANGE[1]:
            penalty += math.log(r / _R_RANGE[1])
    return 3.0 * w0_error + 1.0 * q_error + 0.3 * spread + 0.5 * penalty


def snap_plan(spec: Any, ideal: dict[str, float]) -> list[dict[str, Any]]:
    """Caps are already standard values; the resistor pair is ratio-snapped."""
    return [
        {"kind": "fixed", "ref": "C1", "value": ideal["C1"]},
        {"kind": "fixed", "ref": "C2", "value": ideal["C2"]},
        {
            "kind": "ratio",
            "hi": "R1",
            "lo": "R2",
            "series": getattr(spec, "eseries", "E96"),
            "window": 3,
        },
    ]


def response(spec: Any, values: dict[str, float]) -> dict[str, float]:
    """Analytical prediction of the measurements, for the pre-simulation check."""
    w0_actual, q_actual = realised(
        values["R1"], values["R2"], values["C1"], values["C2"]
    )
    f0_hz = w0_actual / TWO_PI
    target_fc = spec.num_param("fc_hz")
    return {
        "f0_hz": f0_hz,
        "fc_hz": _minus_3db_frequency(f0_hz, q_actual),
        "q_actual": q_actual,
        "gain_db_passband": 0.0,
        "peak_db": _peak_db(q_actual),
        "atten_db_at_10fc": _mag_db(10.0 * target_fc, f0_hz, q_actual),
    }


def _mag_db(f_hz: float, f0_hz: float, q: float) -> float:
    """|H| in dB of a second-order low-pass at ``f_hz``."""
    x = f_hz / f0_hz
    denominator = (1.0 - x * x) ** 2 + (x / q) ** 2
    return -10.0 * math.log10(denominator)


def _minus_3db_frequency(f0_hz: float, q: float) -> float:
    """Closed-form -3 dB corner of a second-order low-pass, in Hz."""
    a = 1.0 - 1.0 / (2.0 * q * q)
    return f0_hz * math.sqrt(a + math.sqrt(a * a + 1.0))


def _peak_db(q: float) -> float:
    """Passband peak in dB; 0 dB when Q <= 1/sqrt(2) (no peaking)."""
    if q <= 1.0 / math.sqrt(2.0):
        return 0.0
    return 20.0 * math.log10(q / math.sqrt(1.0 - 1.0 / (4.0 * q * q)))


def opamp_gbw_hz(spec: Any) -> float:
    """Gain-bandwidth product in Hz of the op-amp this design will use."""
    if "gbw_hz" in spec.params:
        return spec.num_param("gbw_hz")
    return _MODEL_GBW_HZ.get(MODELS["X1"], 1.0e6)


def testbench_context(spec: Any, values: dict[str, float]) -> dict[str, Any]:
    """Extra Jinja variables for tb.cir.j2."""
    fc_hz = spec.num_param("fc_hz")
    vsupply = float(spec.params.get("vsupply", 5.0))
    return {
        "points_per_decade": 400,
        "f_start": max(fc_hz / 1000.0, 1e-3),
        "f_stop": fc_hz * 1000.0,
        "f_passband": fc_hz / 100.0,
        "f_ten_fc": fc_hz * 10.0,
        "vpos": vsupply / 2.0,
        "vneg": -vsupply / 2.0,
        # ngspice's vp() reports phase in radians, so the -90 degree crossing
        # that locates f0 must be written as -pi/2.
        "phase_minus_90_rad": -math.pi / 2.0,
    }


def plot_markers(
    spec: Any, values: dict[str, float], measurements: dict[str, float]
) -> dict[str, tuple[float, float]]:
    """Frequency in Hz at which each spot measurement was taken."""
    fc_hz = spec.num_param("fc_hz")
    frequencies = {
        "gain_db_passband": fc_hz / 100.0,
        "atten_db_at_10fc": fc_hz * 10.0,
    }
    return {
        name: (freq, measurements[name])
        for name, freq in frequencies.items()
        if name in measurements
    }


def check_q_le_3(spec: Any, components: list[Any], context: dict[str, Any]) -> str | None:
    """GOTCHA-SK-01: sensitivity to component tolerance scales with Q."""
    _w0, q = target_w0_q(spec)
    if q <= 3.0:
        return None
    return (
        f"GOTCHA-SK-01: the requested response needs Q={q:.2f}. Above Q=3 the fractional "
        f"error in Q is roughly Q times the component tolerance, so 1% parts would give "
        f"about {q:.0f}% spread in Q. Split this into cascaded lower-Q stages."
    )


def check_gbw_margin(spec: Any, components: list[Any], context: dict[str, Any]) -> str | None:
    """GOTCHA-SK-02: the op-amp must have loop gain to spare at the corner."""
    fc_hz = spec.num_param("fc_hz")
    gbw_hz = opamp_gbw_hz(spec)
    ratio = gbw_hz / fc_hz
    if ratio >= 100.0:
        return None
    return (
        f"GOTCHA-SK-02: op-amp GBW is {gbw_hz:.3g} Hz, only {ratio:.0f}x the {fc_hz:g} Hz "
        f"corner (100x is the working rule). Loop gain runs out near fc, so both fc and Q "
        f"will shift from the designed values. Use OPAMP_FAST, a faster part, or lower fc."
    )


def check_hf_feedthrough(
    spec: Any, components: list[Any], context: dict[str, Any]
) -> str | None:
    """GOTCHA-SK-03: past the GBW, C1 feeds the input straight through."""
    fc_hz = spec.num_param("fc_hz")
    gbw_hz = opamp_gbw_hz(spec)
    for req in spec.requirements:
        if req.meas != "atten_db_at_10fc" or req.max is None:
            continue
        if 10.0 * fc_hz > gbw_hz / 10.0:
            return (
                f"GOTCHA-SK-03: {req.id} is measured at {10 * fc_hz:g} Hz, which is within a "
                f"decade of the op-amp GBW ({gbw_hz:.3g} Hz). Feedback is collapsing there and "
                "C1 couples the input to the output, so the measured attenuation will be worse "
                "than the ideal -40 dB/decade. Use a faster op-amp."
            )
    return None
