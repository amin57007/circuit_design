# circuit-forge (`cforge`)

Local, open-source, CLI-driven circuit synthesis. You write a YAML spec; the
tool computes component values, snaps them to real E-series parts, simulates
the result in **ngspice**, and writes a schematic plus a self-contained HTML
report.

This is a **deterministic engineering tool**, not an AI wrapper. Every numeric
result comes from closed-form equations, `scipy.optimize`, or ngspice. No
runtime step calls a language model.

## Requirements

### System

| Dependency | Why | Install |
|---|---|---|
| **Python 3.11+** | Runtime | python.org / your package manager |
| **ngspice** on `PATH` | Every pass/fail verdict is parsed from ngspice `.meas` output | see below |

```bash
# Ubuntu / Debian
sudo apt install ngspice

# macOS
brew install ngspice

# Windows
# Use WSL2 (`sudo apt install ngspice`) or the ngspice Windows build, then
# add its bin/ directory to PATH.
```

Confirm:

```bash
python3 --version    # >= 3.11
ngspice -v
python scripts/check_env.py
```

### Python packages

Pinned in [`requirements.txt`](requirements.txt) (runtime) and
[`requirements-dev.txt`](requirements-dev.txt) (pytest, ruff, mypy). They match
[`pyproject.toml`](pyproject.toml). Do **not** add torch, transformers,
langchain, any LLM SDK, or a web framework.

| Package | Used for |
|---|---|
| numpy, scipy | numerics and (optional) discrete E-series refinement |
| sympy, lcapy | symbolic transfer functions |
| schemdraw, matplotlib | schematic + Bode / MC plots |
| pydantic>=2, pyyaml | spec / result contracts and YAML I/O |
| jinja2 | netlist templates and the HTML report |
| typer, rich | CLI |

## Install (local)

```bash
git clone https://github.com/amin57007/circuit_design.git
cd circuit_design

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .

# or, with uv:
# uv venv && uv pip install -e '.[dev]'

python scripts/check_env.py        # must print "environment OK"
cforge --help
```

For tests and lint:

```bash
pip install -r requirements-dev.txt
pip install -e .
make test          # fast tests only (< 30 s)
make test-all      # includes live ngspice
```

## Docker (no local Python / ngspice)

The image is Debian slim + Python 3.12 + ngspice + the Python deps.

```bash
docker build -t cforge .

docker run --rm cforge check-env
docker run --rm cforge list-patterns

# Design: mount an output directory so report.html lands on the host.
mkdir -p out
docker run --rm \
  -v "$PWD/examples:/work/examples:ro" \
  -v "$PWD/out:/work/out" \
  cforge design examples/lp_1k.yaml -o out --no-mc
```

Or with Compose:

```bash
docker compose build
docker compose run --rm cforge check-env
docker compose run --rm cforge design examples/lp_1k.yaml -o out --no-mc
docker compose run --rm cforge design examples/sk_butterworth_2k.yaml -o out --mc 50
```

Open `out/<spec-name>/report.html` in a browser.

## Quickstart

```bash
cforge list-patterns
cforge explain rc_lowpass_1st

cforge design examples/lp_1k.yaml -o out --open
cforge design examples/sk_butterworth_2k.yaml -o out --mc 200
```

| Command | Purpose |
|---|---|
| `cforge check-env` | Python version + ngspice on PATH |
| `cforge list-patterns [--block lowpass_filter]` | Catalog |
| `cforge explain PATTERN_ID` | notes, tradeoffs, gotchas |
| `cforge design SPEC.yaml [-o out/] [--topology ID] [--mc 500] [--no-mc] [--open]` | Synthesize, simulate, report |
| `cforge verify DESIGN.json` | Re-simulate a saved design, no re-synthesis |

Exit codes: **0** PASS, **1** FAIL (simulated, a requirement missed), **2** tool/sim error. Usable as a CI gate.

Output for a spec named `lp_1k` goes to `out/lp_1k/`:

```
design.json     machine-readable result (source of truth for the report)
netlist.cir     the exact deck ngspice ran
schematic.svg   real snapped component values
bode.png        frequency response
mc_*.png        Monte-Carlo histograms (unless --no-mc)
report.html     self-contained HTML (inlined SVG, base64 PNGs)
```

## Writing a spec

Copy `examples/lp_1k.yaml` or `examples/sk_butterworth_2k.yaml`. Units inside
the file are SI base units (hertz, ohms, farads, volts). Formatting with `k` /
`nF` happens only in the report.

```yaml
name: my_filter
block: lowpass_filter          # must match a pattern's block
params:
  fc_hz: 2500
  order: 2                     # 1 = RC, 2 = Sallen-Key
  response: butterworth
  gain: 1.0
  vsupply: 5.0                 # op-amp patterns only
eseries: E96                   # E6 | E12 | E24 | E48 | E96
tolerance_percent:
  R: 1.0
  C: 5.0
requirements:
  - id: REQ-001
    meas: fc_hz                # must be a measurement the pattern provides
    min: 2375
    max: 2625
    unit: Hz
    description: "-3 dB corner within +/-5% of 2.5 kHz"
```

`meas` names are not free-form:

- `rc_lowpass_1st`: `fc_hz`, `gain_db_passband`, `atten_db_at_10fc`
- `sallen_key_lp2`: those three plus `q_actual`, `peak_db`

Force a topology with `--topology sallen_key_lp2` or `topology:` in the YAML.

## How to add a new pattern

A pattern is a directory under `patterns/`:

```
patterns/my_topology/
  pattern.yaml        # id, block, applicability ranges, gotchas
  netlist.cir.j2      # Jinja2 -> SPICE topology
  tb.cir.j2           # sources, analyses, .meas  (must include .save)
  design.py           # solve(spec) -> dict[ref, ideal SI value]
  draw.py             # draw(components) -> schemdraw.Drawing
  notes.md            # rationale
```

`pattern.yaml` `id` must equal the directory name. Applicability ranges are
how auto-selection works: the catalog filters on `block` and whether spec
params sit inside those ranges, then ranks by how centred the spec is. No
scoring model.

`design.py` may export `snap_plan(spec, ideal)` so the synthesizer jointly
snaps RC products (`kind: rc_tau`) or resistor ratios (`kind: ratio`).

Device models live in `models/*.lib`. Search order is `CFORGE_MODEL_PATH`,
then `models/*.lib`, then `models/generic.lib`. See `models/README.md`.

## Architecture (short)

```
YAML spec
  -> catalog.select  (rule-based topology pick)
  -> synth.solver    (closed-form values + E-series snap)
  -> spice.runner    (ngspice -b, .meas parse, convergence ladder)
  -> synth.sensitivity (optional Monte Carlo, uniform over tolerance bands)
  -> render + report.html
```

`DesignResult` serializes to `design.json` and is the only input to the report.

## License

MIT. See `pyproject.toml`.
