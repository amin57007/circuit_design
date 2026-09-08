# Sallen-Key 2nd-order low-pass (`sallen_key_lp2`)

## What it is

A two-pole active low-pass built from one op-amp and four passives, in the
**unity-gain (voltage follower)** form:

```
in --[R1]--+--[R2]--+
           |        |
         [C1]     [C2]        C1 returns to the op-amp OUTPUT,
           |        |         C2 goes to ground
           |        +---(+)\
           |               >--+-- out
           +---------------(-)/  |
                            |    |
                            +----+   (follower: out tied to the - input)
```

## Transfer function

With an ideal op-amp:

    H(s) = 1 / (1 + s*C2*(R1 + R2) + s^2 * R1*R2*C1*C2)

    w0 = 1 / sqrt(R1*R2*C1*C2)                  [rad/s]
    Q  = sqrt(R1*R2*C1*C2) / (C2*(R1 + R2))     [-]

## Why the capacitors are unequal

Write `r = R1/R2` and `g = sqrt(R1*R2)`. Then

    w0 = 1 / (g * sqrt(C1*C2))          depends only on g
    Q  = sqrt(C1/C2) / (sqrt(r) + 1/sqrt(r))   depends only on r

The corner and the damping are therefore *independent* knobs, which is what
makes the design procedure clean. It also exposes the existence condition:
real resistors require

    C1 / C2 >= 4 * Q^2

For a Butterworth (Q = 1/sqrt(2) = 0.7071) that is `C1 >= 2*C2`. **With
C1 = C2 the highest achievable Q is 0.5**, which is not even critically
damped. The textbook "equal component" Sallen-Key gets around this by taking
gain `K = 3 - 1/Q` (K = 1.586 for Butterworth, i.e. +4 dB of passband gain).
That is a fine circuit, but it fails a unity-gain passband requirement, so
this pattern uses the unequal-capacitor unity-gain form instead.

## Design procedure

1. Get the target denominator polynomial from `scipy.signal.butter` /
   `bessel` / `cheby1` (analog, order 2) and read off `w0` and `Q`. Using
   scipy rather than a hand-typed table means new response families need no
   new algebra.
2. Search the coarse capacitor series (E6 by default) for pairs satisfying
   `C1/C2 >= 4*Q^2` with ~8% headroom. Sitting exactly on the boundary forces
   `R1 == R2` with a zero discriminant, which is numerically fragile once the
   values are snapped.
3. For each candidate pair, solve the exact resistors. `R1` and `R2` are the
   roots of `x^2 - S*x + P = 0` with `S = 1/(Q*w0*C2)` and
   `P = 1/(w0^2*C1*C2)`.
4. Snap `R1`/`R2` **jointly** with `eseries.snap_ratio`. Because Q depends
   only on the ratio, preserving the ratio preserves Q; the residual error
   lands in `w0`, where the E96 grid is dense enough to absorb it.
5. Score each candidate *after* snapping, weighting `w0` error 3x the `Q`
   error and penalising resistors outside 1 k-ohm .. 300 k-ohm. Scoring after
   snapping matters: a capacitor pair that looks ideal on paper can snap badly.

## Measuring Q in ngspice

At the pole frequency a second-order low-pass has phase exactly `-90 degrees`,
and there its magnitude equals `Q` (for unity DC gain). Both `f0` and `Q` come
straight from that one crossing:

```spice
.meas ac f0_hz    WHEN vp(out)=-1.5707963
.meas ac q_actual FIND vm(out) WHEN vp(out)=-1.5707963
```

Two traps here, both learned the hard way:

- **`vp()` in ngspice is in radians**, not degrees. Writing `-90` makes the
  measurement fail with "out of interval".
- Inverting the *peak height* to get Q is the more common textbook approach,
  but it degenerates exactly at Butterworth: `Q = 1/sqrt(2)` is maximally flat,
  so there is no peak to measure. The phase crossing works for every Q.

## Gotchas

- **GOTCHA-SK-01 (Q <= 3).** The fractional sensitivity of Q to component
  values scales with Q. Past Q = 3, 1% parts give roughly 3% spread in Q and
  the design stops being manufacturable. Cascade lower-Q stages.
- **GOTCHA-SK-02 (GBW margin).** The op-amp needs loop gain to spare at the
  corner; the working rule is `GBW >= 100 * fc`. The shipped `OPAMP_GENERIC`
  has 1 MHz, so it is good to about 10 kHz. `OPAMP_FAST` (10 MHz) extends that.
- **GOTCHA-SK-03 (HF feedthrough).** Above the GBW the feedback loop stops
  working and C1 couples the input directly to the output, so the stopband
  flattens out instead of continuing at -40 dB/decade. If the attenuation
  requirement is measured within a decade of GBW, expect to miss it.

## Sensitivity

Q is the fragile quantity. Because it depends on `sqrt(C1/C2)` and on the
resistor ratio, capacitor tolerance enters at half weight but capacitors are
the loose parts (5-20%). This is exactly what the Monte Carlo is for: with 1%
resistors and 5% capacitors the corner holds well while Q spreads by several
percent.

## Measurements provided

| Measurement        | Analysis | Meaning |
| ------------------ | -------- | ------- |
| `fc_hz`            | `.ac`    | First downward -3.01 dB crossing. |
| `gain_db_passband` | `.ac`    | Response two decades below the corner. |
| `q_actual`         | `.ac`    | Magnitude at the -90 degree phase crossing. |
| `peak_db`          | `.ac`    | Maximum passband magnitude; 0 dB when flat. |
| `atten_db_at_10fc` | `.ac`    | Response one decade above the target corner. |
