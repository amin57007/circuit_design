"""Rendering: schematics, plots, and the display formatting they share.

This package is the *render boundary*.  Everywhere else in circuit-forge a
value is a plain float in SI base units; this is the only place where a number
acquires a prefix, a unit symbol or a fixed number of digits.

Nothing heavyweight is imported here: ``schematic`` and ``plots`` pull in
schemdraw and matplotlib respectively, so importing this package stays cheap
for the catalog and CLI paths that only need :func:`format_eng`.
"""

from __future__ import annotations

import math

__all__ = ["format_component_value", "format_eng", "format_quantity"]

# SI prefixes from pico to giga; SPICE-compatible spellings are handled
# separately because SPICE writes "meg" for 1e6 and "m" for milli.
_PREFIXES: tuple[tuple[float, str], ...] = (
    (1e9, "G"),
    (1e6, "M"),
    (1e3, "k"),
    (1.0, ""),
    (1e-3, "m"),
    (1e-6, "u"),
    (1e-9, "n"),
    (1e-12, "p"),
    (1e-15, "f"),
)


def format_eng(value: float, digits: int = 3) -> str:
    """Format ``value`` with an SI prefix, e.g. ``15.8k`` or ``4.7n``.

    ``digits`` is the number of significant digits.  Zero, NaN and infinity are
    passed through in a readable form rather than raising, because this is a
    display path and must never take down a report.
    """
    if value is None or isinstance(value, bool):
        return str(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == 0.0:
        return "0"

    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    for scale, prefix in _PREFIXES:
        if magnitude >= scale * (1 - 1e-12):
            mantissa = magnitude / scale
            return f"{sign}{_sig(mantissa, digits)}{prefix}"
    return f"{sign}{magnitude:.{digits}g}"


def _sig(value: float, digits: int) -> str:
    """Round to ``digits`` significant figures, dropping trailing zeros."""
    if value == 0.0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    decimals = max(0, digits - 1 - exponent)
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


_UNIT_SYMBOL: dict[str, str] = {
    "ohm": "\u03a9",
    "F": "F",
    "H": "H",
    "V": "V",
    "A": "A",
    "": "",
    "device": "",
}


def format_quantity(value: float, unit: str, digits: int = 3) -> str:
    """Format ``value`` in SI base units with its symbol, e.g. ``15.8 kΩ``."""
    symbol = _UNIT_SYMBOL.get(unit, unit)
    text = format_eng(value, digits)
    return f"{text} {symbol}".strip() if symbol else text


def format_component_value(value: float, unit: str, digits: int = 3) -> str:
    """Schematic-label form of a component value: ``15.8k``, ``4.7nF``, ``10uH``."""
    text = format_eng(value, digits)
    if unit == "ohm":
        # Bare number is the schematic convention for resistors in ohms.
        return text
    if unit in ("F", "H", "V", "A"):
        return f"{text}{unit}"
    return text
