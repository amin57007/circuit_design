"""SchemDraw figure for the unity-gain Sallen-Key low-pass.

Explicit placement; the feedback capacitor C1 is routed above the op-amp so it
does not cross the signal path.
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
    """Return a drawing of the Sallen-Key section with real values."""
    r1 = _value_of(components, "R1")
    r2 = _value_of(components, "R2")
    c1 = _value_of(components, "C1")
    c2 = _value_of(components, "C2")

    d = schemdraw.Drawing(show=False)
    d.config(unit=2.4, fontsize=12)

    d.add(elm.Dot(open=True).label("in", loc="left"))
    d.add(elm.Resistor().right().label(f"R1\n{r1}"))
    na = d.add(elm.Dot())
    d.add(elm.Resistor().right().label(f"R2\n{r2}"))
    nb = d.add(elm.Dot())

    # C2 from the op-amp non-inverting input to ground.
    d.push()
    d.here = nb.center
    d.add(elm.Capacitor().down().label(f"C2\n{c2}", loc="bottom"))
    d.add(elm.Ground())
    d.pop()

    # Op-amp as a follower.
    d.here = nb.center
    opamp = d.add(elm.Opamp(leads=True).right().anchor("in1").label("X1", loc="center", ofst=(0.4, 0)))
    out = d.add(elm.Dot().at(opamp.out))
    d.add(elm.Line().right().length(1.4).at(out.center))
    d.add(elm.Dot(open=True).label("out", loc="right"))

    # Follower feedback: output back to the inverting input.
    d.add(elm.Line().down().at(out.center).length(1.5))
    d.add(elm.Line().left().tox(opamp.in2[0] - 0.6))
    d.add(elm.Line().up().toy(opamp.in2[1]))
    d.add(elm.Line().right().tox(opamp.in2[0]))

    # C1 feedback: output back to the R1/R2 junction, routed above the op-amp.
    d.add(elm.Line().up().at(out.center).length(2.2))
    d.add(elm.Capacitor().left().tox(na.center[0]).label(f"C1\n{c1}"))
    d.add(elm.Line().down().toy(na.center[1]))

    return d
