# SPICE device models

`cforge` resolves every active device by name through `cforge.spice.modellib.resolve()`.
This directory is the search path.

## Search order

1. Every `*.lib` in a directory listed in `CFORGE_MODEL_PATH` (`os.pathsep`-separated), sorted by filename.
2. Every `*.lib` in this directory except `generic.lib`, sorted by filename.
3. `generic.lib` (shipped with the tool).
4. Otherwise `ModelNotFound` is raised, naming the exact file you should create.

The first definition found wins, so a vendor library always overrides the generic
stand-in with the same model name.

## What ships here

`generic.lib` contains behavioural stand-ins good enough to verify topology,
tolerance sensitivity and loop stability, but **not** datasheet-accurate:

| Name             | Type                | Notes |
| ---------------- | ------------------- | ----- |
| `1N4148`         | `.model D`          | Small-signal switching diode. |
| `DCLAMP`         | `.model D`          | Stiff clamp diode used inside the op-amp macromodel. Not a real part. |
| `2N3904`         | `.model NPN`        | Gummel-Poon general-purpose NPN. |
| `2N7002`         | `.model NMOS` level 1 | Vto = 2.0 V, Kp = 0.15 A/V^2 at W/L = 1. Level 1 for portability. |
| `NMOS_LV`        | `.model NMOS` level 1 | Vto = 1.0 V variant for 3.3 V rails. |
| `OPAMP_GENERIC`  | `.subckt`           | Aol = 1e5, fp = 10 Hz (GBW = 1 MHz), Rout = 50 ohm, output clamped 1.2 V from each rail. Nodes: `inp inn vcc vee out`. |
| `OPAMP_FAST`     | `.subckt`           | Same, GBW = 10 MHz. |

## Adding a model

1. Drop the vendor file in this directory with a `.lib` extension, e.g. `models/tlv9061.lib`.
2. Make sure it contains a top-level `.model <name> ...` card or a `.subckt <name> ...` block.
   `cforge` extracts the definition text by name; continuation lines starting with `+`
   are included automatically, and `.subckt` blocks are captured through their `.ends`.
3. Reference the name from a pattern's `pattern.yaml` (`requires:`) or directly from
   the netlist template as `{{ models }}`.
4. Verify resolution:

   ```bash
   python -c "from cforge.spice.modellib import available_models; print(sorted(available_models()))"
   ```

## Op-amp subcircuit pin order

Patterns assume the five-terminal order `inp inn vcc vee out`. If a vendor
macromodel uses a different order, wrap it in a thin `.subckt` adapter in your own
`.lib` rather than editing the pattern templates.
