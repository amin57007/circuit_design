# First-order passive RC low-pass (`rc_lowpass_1st`)

## What it is

A series resistor into a shunt capacitor:

```
in ---[ R1 ]---+--- out
               |
             [ C1 ]
               |
              GND
```

## Transfer function

With an ideal (infinite-impedance) load and an ideal (zero-impedance) source:

    H(s) = 1 / (1 + s*R1*C1)

so the time constant is `tau = R1*C1` seconds and the -3 dB corner is

    fc = 1 / (2*pi*R1*C1)      [Hz]

Magnitude falls at -20 dB/decade above `fc`. Exactly one decade above `fc` the
attenuation is

    20*log10(1/sqrt(1 + 10^2)) = -20.04 dB

which is the hard ceiling for this topology (see GOTCHA-RC-02).

## Why the design flow picks C first

`tau` alone does not fix `R1` and `C1`; any pair with the right product works.
Capacitors are the constrained part: they come in coarse values (E6 is typical
for ceramics), have 5-20% tolerances, and the available range is narrow.
Resistors are cheap, available in E96 at 1%, and cover twelve decades. So the
design fixes `C1` to a coarse standard value and solves `R1 = tau / C1`
against the fine series. That is `eseries.snap_rc_pair`, and it is what a
person does at a bench.

The resulting `R1` is nudged toward 1 k-ohm .. 100 k-ohm. Below that the source
has to drive a heavy load; above it, resistor thermal noise and stray
capacitance start to matter, and any real load impedance shifts `fc`.

## Gotchas

- **GOTCHA-RC-01 (load impedance).** The corner assumes nothing loads `out`.
  A load `RL` in parallel with `C1` both attenuates the passband by
  `RL/(R1+RL)` and moves the corner to `1/(2*pi*(R1||RL)*C1)`. Keep
  `RL >= 100*R1` for a 1% effect. Pass `rload_ohm` in the spec params to have
  the testbench include the load and measure the real behaviour.

- **GOTCHA-RC-02 (attenuation ceiling).** No choice of R and C gets more than
  ~20 dB one decade out. A requirement on `atten_db_at_10fc` tighter than about
  -20 dB is unsatisfiable here; the synthesizer warns rather than silently
  producing a design that cannot pass. Move to `sallen_key_lp2` (-40 dB/decade)
  or cascade sections.

## Sensitivity

`fc` depends on the product `R1*C1`, so tolerances add in quadrature for a
statistical estimate and linearly for worst case. With 1% resistors and 5%
capacitors, worst-case `fc` error is about 6%, which is why the Monte Carlo
pass rate matters even when the nominal design has healthy margin.

## Measurements provided

| Measurement        | Analysis | Meaning |
| ------------------ | -------- | ------- |
| `fc_hz`            | `.ac`    | Frequency where the response first falls through -3.01 dB. |
| `gain_db_passband` | `.ac`    | Response two decades below the target corner (flat region). |
| `atten_db_at_10fc` | `.ac`    | Response one decade above the target corner. |
