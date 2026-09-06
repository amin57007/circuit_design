"""Matplotlib figures for the report.

Every figure is saved as PNG at 150 dpi with a tight bounding box, and every
axis carries explicit units.  Requirement bands are shaded so a reader can see
compliance without cross-referencing the traceability table.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cforge.models import Requirement, SensitivityResult  # noqa: E402
from cforge.render import format_eng  # noqa: E402

__all__ = ["bode", "mc_histogram", "transient"]

DPI = 150
_PASS = "#1a7f37"
_FAIL = "#cf222e"
_BAND = "#cf222e"
_TRACE = "#0969da"
_NOMINAL = "#8250df"


def _save(fig: "plt.Figure", out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def bode(
    freq_hz: Sequence[float],
    mag_db: Sequence[float],
    phase_deg: Sequence[float],
    requirements: Sequence[Requirement],
    out_path: Path,
    fc_hz: float | None = None,
    title: str = "Frequency response",
    markers: dict[str, tuple[float, float]] | None = None,
) -> Path:
    """Magnitude and phase against frequency, with requirement limits marked.

    ``freq_hz`` is in hertz, ``mag_db`` in decibels, ``phase_deg`` in degrees.
    ``fc_hz`` marks the measured corner frequency.  ``markers`` maps a
    measurement name to the ``(frequency_hz, value_db)`` point at which it was
    taken, so a spot requirement such as ``atten_db_at_10fc`` is drawn where it
    actually applies instead of as a band across the whole sweep.
    """
    f = np.asarray(freq_hz, dtype=float)
    mag = np.asarray(mag_db, dtype=float)
    phase = np.asarray(phase_deg, dtype=float)

    fig, (ax_mag, ax_ph) = plt.subplots(
        2, 1, figsize=(8.0, 5.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax_mag.semilogx(f, mag, color=_TRACE, linewidth=1.8, label="|H(f)|")
    ax_mag.set_ylabel("Magnitude [dB]")
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.set_title(title)

    # Fix the y-range from the trace before any limit lines are added, so a
    # requirement far outside the response cannot squash the curve flat.
    if mag.size:
        finite = mag[np.isfinite(mag)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            pad = max(3.0, 0.08 * (hi - lo))
            ax_mag.set_ylim(lo - pad, hi + pad)

    _draw_db_requirements(ax_mag, requirements, markers or {})

    if fc_hz is not None and fc_hz > 0:
        ax_mag.axvline(fc_hz, color=_NOMINAL, linestyle="--", linewidth=1.2)
        ax_mag.annotate(
            f"fc = {format_eng(fc_hz)}Hz",
            xy=(fc_hz, -3.01),
            xytext=(6, 8),
            textcoords="offset points",
            color=_NOMINAL,
            fontsize=9,
        )
        ax_ph.axvline(fc_hz, color=_NOMINAL, linestyle="--", linewidth=1.2)

    ax_mag.axhline(-3.01, color="#57606a", linestyle=":", linewidth=1.0)
    ax_mag.legend(loc="lower left", fontsize=9)

    ax_ph.semilogx(f, phase, color=_TRACE, linewidth=1.6)
    ax_ph.set_ylabel("Phase [deg]")
    ax_ph.set_xlabel("Frequency [Hz]")
    ax_ph.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    return _save(fig, out_path)


def _draw_db_requirements(
    ax: "plt.Axes",
    requirements: Sequence[Requirement],
    markers: dict[str, tuple[float, float]],
) -> None:
    """Mark every dB-valued requirement where it actually applies.

    A requirement measured at one frequency (``atten_db_at_10fc``) gets a point
    marker with a short limit whisker; a requirement that constrains a whole
    band gets a dashed line across the axes.  Shading the full plot for a spot
    limit would tell the reader the passband violates it, which is nonsense.
    """
    y_lo, y_hi = ax.get_ylim()
    for req in requirements:
        if req.unit.lower() not in ("db", "dbv"):
            continue
        point = markers.get(req.meas)
        if point is not None and req.min is not None and req.max is not None:
            # A two-sided spot limit reads as one bracket, not two lines that
            # collide when the band is narrow relative to the axis range.
            freq, value = point
            span = (freq / 2.5, freq * 2.5)
            ax.fill_between(span, req.min, req.max, color=_PASS, alpha=0.18, zorder=1)
            for bound in (req.min, req.max):
                ax.plot(span, [bound, bound], color=_BAND, linewidth=1.4, alpha=0.85)
            ax.plot([freq], [value], marker="o", color=_TRACE, markersize=6, zorder=5)
            ax.annotate(
                f"{req.id} {req.min:g} .. {req.max:g} dB",
                xy=(freq, req.max),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                color=_BAND,
            )
            continue
        for bound, kind in ((req.min, "min"), (req.max, "max")):
            if bound is None:
                continue
            visible = y_lo <= bound <= y_hi
            if point is not None:
                freq, value = point
                span = (freq / 2.5, freq * 2.5)
                ax.plot(span, [bound, bound], color=_BAND, linewidth=1.6, alpha=0.85)
                ax.plot([freq], [value], marker="o", color=_TRACE, markersize=6, zorder=5)
                ax.annotate(
                    f"{req.id} {kind} {bound:g} dB",
                    xy=(freq, bound),
                    xytext=(6, -12 if kind == "max" else 6),
                    textcoords="offset points",
                    fontsize=8,
                    color=_BAND,
                )
            elif visible:
                ax.axhline(bound, color=_BAND, linestyle="--", linewidth=1.1, alpha=0.7)
                ax.annotate(
                    f"{req.id} {kind} {bound:g} dB",
                    xy=(ax.get_xlim()[0], bound),
                    xytext=(6, 3),
                    textcoords="offset points",
                    fontsize=8,
                    color=_BAND,
                )


def transient(
    t_s: Sequence[float],
    signals: dict[str, Sequence[float]],
    requirements: Sequence[Requirement],
    out_path: Path,
    settle_time_s: float | None = None,
    overshoot_percent: float | None = None,
    title: str = "Transient response",
    ylabel: str = "Value [SI]",
) -> Path:
    """Time-domain traces with settling time and overshoot marked.

    ``t_s`` is in seconds; each entry of ``signals`` is a same-length sequence
    in whatever SI base unit ``ylabel`` names.
    """
    t = np.asarray(t_s, dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))

    for name, values in signals.items():
        ax.plot(t, np.asarray(values, dtype=float), linewidth=1.6, label=name)

    first = next(iter(signals.values()), None)
    if first is not None and len(first) > 0:
        final = float(np.asarray(first, dtype=float)[-1])
        ax.axhline(final, color="#57606a", linestyle=":", linewidth=1.0)
        if overshoot_percent is not None:
            peak = final * (1.0 + overshoot_percent / 100.0)
            ax.axhline(peak, color=_BAND, linestyle="--", linewidth=1.0)
            ax.annotate(
                f"overshoot {overshoot_percent:.1f}%",
                xy=(t[len(t) // 2] if t.size else 0.0, peak),
                xytext=(0, 4),
                textcoords="offset points",
                fontsize=9,
                color=_BAND,
            )

    if settle_time_s is not None and settle_time_s > 0:
        ax.axvline(settle_time_s, color=_PASS, linestyle="--", linewidth=1.2)
        ax.annotate(
            f"settles at {format_eng(settle_time_s)}s",
            xy=(settle_time_s, ax.get_ylim()[0]),
            xytext=(6, 12),
            textcoords="offset points",
            fontsize=9,
            color=_PASS,
        )

    for req in requirements:
        if req.unit.lower() in ("s", "sec", "%", "pct"):
            continue
        for bound in (req.min, req.max):
            if bound is not None:
                ax.axhline(bound, color=_BAND, linewidth=1.0, alpha=0.6)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(signals) > 1:
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return _save(fig, out_path)


def dc_sweep(
    x: Sequence[float],
    y: Sequence[float],
    out_path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    requirements: Sequence[Requirement] = (),
) -> Path:
    """A DC sweep, used for load-compliance curves."""
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(np.asarray(x, dtype=float), np.asarray(y, dtype=float), color=_TRACE, linewidth=1.8)
    for req in requirements:
        for bound in (req.min, req.max):
            if bound is not None:
                ax.axhline(bound, color=_BAND, linewidth=1.0, alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_path)


def mc_histogram(
    result: SensitivityResult,
    requirement: Requirement | None,
    out_path: Path,
    unit: str = "",
) -> Path:
    """Monte-Carlo distribution with the nominal value and requirement limits.

    The pass region is shaded green, the nominal design marked, and the
    5th/95th percentiles drawn so the reader can judge the tail directly.
    """
    samples = np.asarray(result.samples, dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    if samples.size:
        bins = max(10, min(60, int(np.sqrt(samples.size) * 2)))
        ax.hist(samples, bins=bins, color=_TRACE, alpha=0.65, edgecolor="white")

    if requirement is not None:
        lo = requirement.min if requirement.min is not None else ax.get_xlim()[0]
        hi = requirement.max if requirement.max is not None else ax.get_xlim()[1]
        ax.axvspan(lo, hi, color=_PASS, alpha=0.10, zorder=0, label="pass region")
        for bound, name in ((requirement.min, "min"), (requirement.max, "max")):
            if bound is not None:
                ax.axvline(bound, color=_FAIL, linestyle="-", linewidth=1.6)
                ax.annotate(
                    f"{requirement.id} {name} {bound:g}",
                    xy=(bound, ax.get_ylim()[1]),
                    xytext=(4, -12),
                    textcoords="offset points",
                    fontsize=8,
                    color=_FAIL,
                    rotation=90,
                    va="top",
                )

    ax.axvline(
        result.nominal,
        color=_NOMINAL,
        linestyle="--",
        linewidth=2.0,
        label=f"nominal {format_eng(result.nominal)}",
    )
    for percentile, value in (("p5", result.p5), ("p95", result.p95)):
        ax.axvline(value, color="#57606a", linestyle=":", linewidth=1.2)
        ax.annotate(
            percentile,
            xy=(value, 0),
            xytext=(2, 4),
            textcoords="offset points",
            fontsize=8,
            color="#57606a",
        )

    unit_text = f" [{unit}]" if unit else ""
    ax.set_xlabel(f"{result.meas}{unit_text}")
    ax.set_ylabel(f"Monte-Carlo runs (n = {result.n_samples})")
    ax.set_title(f"{result.meas}: tolerance sensitivity, pass rate {100 * result.pass_rate:.1f}%")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_path)
