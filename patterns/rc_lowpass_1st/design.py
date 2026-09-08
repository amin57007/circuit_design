"""Closed-form design of a first-order passive RC low-pass.

All values are SI base units: ohms, farads, hertz, seconds.

    H(s) = 1 / (1 + s*R1*C1),  fc = 1 / (2*pi*R1*C1)

``solve`` returns ideal values; the snapping plan tells the synthesizer to
treat R1/C1 as an RC pair so it fixes C to a coarse standard value and solves R
against the fine series, rather than snapping each part independently.
"""

from __future__ import annotations

import math
from typing import Any

TWO_PI = 2.0 * math.pi

# Ceiling on one-decade-out rejection for any single-pole section, in dB.
FIRST_ORDER_ATTEN_AT_10FC_DB = -20.0 * math.log10(math.sqrt(1.0 + 100.0)) / math.log10(10.0)

UNITS: dict[str, str] = {"R1": "ohm", "C1": "F"}


def solve(spec: Any) -> dict[str, float]:
    """Ideal component values for the requested corner frequency.

    Reads ``params.fc_hz`` (Hz).  A preferred capacitor may be pinned with
    ``params.c_f`` (farads), in which case R follows directly from it.
    """
    fc_hz = spec.num_param("fc_hz")
    if fc_hz <= 0.0:
        raise ValueError(f"spec {spec.name!r}: fc_hz must be positive, got {fc_hz}")
    tau_s = 1.0 / (TWO_PI * fc_hz)

    if "c_f" in spec.params:
        c_f = spec.num_param("c_f")
        if c_f <= 0.0:
            raise ValueError(f"spec {spec.name!r}: c_f must be positive, got {c_f}")
        return {"R1": tau_s / c_f, "C1": c_f}

    # No capacitor pinned: pick one that lands R in a sane impedance window,
    # then let snap_rc_pair refine the choice against the real value grids.
    c_f = _preferred_capacitor(tau_s)
    return {"R1": tau_s / c_f, "C1": c_f}


def _preferred_capacitor(tau_s: float, r_target_ohm: float = 1.0e4) -> float:
    """Capacitance (F) that puts R near ``r_target_ohm`` for the given tau."""
    return tau_s / r_target_ohm


def snap_plan(spec: Any, ideal: dict[str, float]) -> list[dict[str, Any]]:
    """Tell the synthesizer to snap R1 and C1 jointly against their tau."""
    fc_hz = spec.num_param("fc_hz")
    return [
        {
            "kind": "rc_tau",
            "r": "R1",
            "c": "C1",
            "tau": 1.0 / (TWO_PI * fc_hz),
            "c_series": str(spec.params.get("c_eseries", "E6")),
        }
    ]


def response(spec: Any, values: dict[str, float]) -> dict[str, float]:
    """Analytical prediction of every measurement this pattern provides.

    Used to sanity-check the effect of snapping before ngspice is invoked; the
    verdicts still come from ngspice.
    """
    tau_s = values["R1"] * values["C1"]
    fc_hz = 1.0 / (TWO_PI * tau_s)
    target_fc = spec.num_param("fc_hz")
    ten_fc = 10.0 * target_fc
    return {
        "fc_hz": fc_hz,
        "gain_db_passband": 0.0,
        "atten_db_at_10fc": _mag_db(ten_fc, fc_hz),
    }


def _mag_db(f_hz: float, fc_hz: float) -> float:
    """|H| in dB at ``f_hz`` for a single pole at ``fc_hz``."""
    return -10.0 * math.log10(1.0 + (f_hz / fc_hz) ** 2)


def testbench_context(spec: Any, values: dict[str, float]) -> dict[str, Any]:
    """Extra Jinja variables for tb.cir.j2, derived from the spec."""
    fc_hz = spec.num_param("fc_hz")
    return {
        "points_per_decade": 200,
        "f_start": max(fc_hz / 1000.0, 1e-3),
        "f_stop": fc_hz * 1000.0,
        "f_passband": fc_hz / 100.0,
        "f_ten_fc": fc_hz * 10.0,
        "rload_ohm": float(spec.params["rload_ohm"]) if "rload_ohm" in spec.params else None,
    }


def plot_markers(
    spec: Any, values: dict[str, float], measurements: dict[str, float]
) -> dict[str, tuple[float, float]]:
    """Frequency (Hz) at which each spot measurement was taken, for the Bode plot."""
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


def check_load_impedance(spec: Any, components: list[Any], context: dict[str, Any]) -> str | None:
    """GOTCHA-RC-01: a load comparable to R1 shifts the corner."""
    if "rload_ohm" not in spec.params:
        return None
    rload = spec.num_param("rload_ohm")
    r1 = next((c.value for c in components if c.ref == "R1"), None)
    if r1 is None or rload <= 0.0:
        return None
    ratio = rload / r1
    if ratio >= 100.0:
        return None
    shift_percent = 100.0 * (1.0 / (1.0 + r1 / rload) - 1.0)
    return (
        f"GOTCHA-RC-01: rload_ohm={rload:g} is only {ratio:.1f}x R1={r1:g} ohm, so the "
        f"passband droops by {abs(shift_percent):.1f}% and fc moves up by the same "
        f"factor. Raise the load impedance or buffer the output."
    )


def check_attenuation_reachable(
    spec: Any, components: list[Any], context: dict[str, Any]
) -> str | None:
    """GOTCHA-RC-02: a single pole cannot beat ~-20 dB one decade out."""
    for req in spec.requirements:
        if req.meas != "atten_db_at_10fc" or req.max is None:
            continue
        if req.max < FIRST_ORDER_ATTEN_AT_10FC_DB:
            return (
                f"GOTCHA-RC-02: {req.id} demands atten_db_at_10fc <= {req.max:g} dB, but a "
                f"first-order section reaches only {FIRST_ORDER_ATTEN_AT_10FC_DB:.2f} dB one "
                "decade out no matter what R and C are. Use sallen_key_lp2 or cascade "
                "sections."
            )
    return None
