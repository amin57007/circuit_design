"""SchemDraw figure for the first-order RC low-pass.

Explicit placement only: auto-layout from a netlist is a hard problem and this
tool does not attempt it.  Labels carry the real computed values.
"""

from __future__ import annotations

from typing import Any

import schemdraw
import schemdraw.elements as elm

from cforge.render import format_component_value


def _value_of(components: list[Any], ref: str) -> str:
    for component in components:
        if component.ref == ref:
            return format_component_value(component.value, component.unit)
    return "?"


def draw(components: list[Any]) -> schemdraw.Drawing:
    """Return a drawing of the RC low-pass with real values in the labels."""
    r1 = _value_of(components, "R1")
    c1 = _value_of(components, "C1")

    d = schemdraw.Drawing(show=False)
    d.config(unit=2.6, fontsize=13)

    source = d.add(elm.SourceSin().up().label("Vin\n1 V AC", loc="left"))
    d.add(elm.Line().right().length(1.0))
    d.add(elm.Dot().label("in", loc="top", ofst=(0, 0.15)))

    d.add(elm.Resistor().right().label(f"R1\n{r1}"))
    node = d.add(elm.Dot())
    d.add(elm.Line().right().length(1.2))
    d.add(elm.Dot(open=True).label("out", loc="top", ofst=(0, 0.15)))

    d.push()
    d.here = node.center
    d.add(elm.Capacitor().down().label(f"C1\n{c1}", loc="bottom"))
    d.add(elm.Ground())
    d.pop()

    d.here = source.start
    d.add(elm.Line().down().toy(node.center[1] - 2.6))
    d.add(elm.Line().right().tox(node.center[0]))

    return d
