"""Rule-based topology selection.

Selection is a filter followed by a deterministic ranking.  There is no
scoring model and no learned weighting: given the same spec and the same
catalog, the same pattern is always chosen, and the reason is printable.

Filtering
    A candidate must (a) declare the spec's ``block``, and (b) contain every
    spec param it constrains inside its declared range.  Scalar constraints
    named ``<param>_max`` / ``<param>_min`` and string tags are checked too.

Ranking
    Candidates are ordered by how centred the spec sits inside the ranges the
    pattern declares.  "Centred" is measured in log space for quantities that
    span decades (frequency, current, resistance) and linearly otherwise, so a
    2 kHz corner is judged as comfortably central in a 1 Hz .. 1 MHz range
    rather than as sitting at 0.2% of it.  Ties break on the number of
    constraints checked (more specific patterns win), then on pattern id.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from cforge.catalog.loader import Pattern
from cforge.models import Spec

__all__ = ["NoPatternFound", "Candidate", "explain", "rank", "select"]


class NoPatternFound(LookupError):
    """No pattern in the catalog can implement the spec."""


# Params whose ranges span decades, so centredness is judged on a log scale.
_LOG_SCALED: frozenset[str] = frozenset(
    {"fc_hz", "f0_hz", "bw_hz", "iout_a", "rload_ohm", "cload_f", "gbw_hz", "slew_v_s"}
)


@dataclass(frozen=True)
class Candidate:
    """One pattern considered for a spec, with the reasoning attached."""

    pattern: Pattern
    eligible: bool
    score: float
    checks: int
    reasons: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.pattern.id

    def sort_key(self) -> tuple[float, int, str]:
        # Lower score is better (0.0 = perfectly centred); more checks is more
        # specific; id is the final deterministic tiebreak.
        return (self.score, -self.checks, self.pattern.id)


def _centredness(value: float, lo: float, hi: float, log_scaled: bool) -> float:
    """0.0 at the centre of [lo, hi], 1.0 at either edge.

    Values outside the range are not passed here; the filter rejects them first.
    """
    if log_scaled and lo > 0.0 and hi > 0.0:
        value, lo, hi = math.log10(value), math.log10(lo), math.log10(hi)
    span = hi - lo
    if span <= 0.0:
        return 0.0
    return abs(2.0 * (value - lo) / span - 1.0)


def _param_value(spec: Spec, key: str) -> float | None:
    if key not in spec.params:
        return None
    try:
        return float(spec.params[key])
    except (TypeError, ValueError):
        return None


def evaluate(spec: Spec, pattern: Pattern) -> Candidate:
    """Decide whether ``pattern`` can implement ``spec``, and how well it fits."""
    reasons: list[str] = []

    if pattern.block != spec.block:
        return Candidate(
            pattern=pattern,
            eligible=False,
            score=math.inf,
            checks=0,
            reasons=[f"block {pattern.block!r} != spec block {spec.block!r}"],
        )

    applicability = pattern.meta.applicability
    penalties: list[float] = []
    checks = 0
    eligible = True

    for key, (lo, hi) in sorted(applicability.ranges().items()):
        value = _param_value(spec, key)
        if value is None:
            continue
        checks += 1
        if not (lo <= value <= hi):
            eligible = False
            reasons.append(f"{key}={value:g} outside range [{lo:g}, {hi:g}]")
            continue
        penalties.append(_centredness(value, lo, hi, key in _LOG_SCALED))
        reasons.append(f"{key}={value:g} within [{lo:g}, {hi:g}]")

    for key, limit in sorted(applicability.scalars().items()):
        param, bound = _split_bound(key)
        if param is None:
            continue
        value = _param_value(spec, param)
        if value is None:
            continue
        checks += 1
        ok = value <= limit if bound == "max" else value >= limit
        if not ok:
            eligible = False
            reasons.append(f"{param}={value:g} violates {key}={limit:g}")
        else:
            reasons.append(f"{param}={value:g} satisfies {key}={limit:g}")

    for key, tag in sorted(applicability.tags().items()):
        value = spec.params.get(key)
        if value is None:
            continue
        checks += 1
        if _tag_matches(tag, value):
            reasons.append(f"{key}={value!r} matches tag {tag!r}")
        else:
            eligible = False
            reasons.append(f"{key}={value!r} does not match tag {tag!r}")

    score = sum(penalties) / len(penalties) if penalties else 0.5
    return Candidate(
        pattern=pattern,
        eligible=eligible,
        score=score,
        checks=checks,
        reasons=reasons,
    )


def _split_bound(key: str) -> tuple[str | None, str]:
    """``"q_max"`` -> ``("q", "max")``; a key with no bound suffix is ignored."""
    if key.endswith("_max"):
        return key[: -len("_max")], "max"
    if key.endswith("_min"):
        return key[: -len("_min")], "min"
    return None, ""


def _tag_matches(tag: str, value: object) -> bool:
    """Check a string tag such as ``unity_or_noninverting`` against a spec param.

    A tag is a set of alternatives joined by ``_or_``.  Numeric spec values are
    matched against the special alternatives ``unity`` (gain == 1) and
    ``noninverting`` (gain >= 1), which is what the analog vocabulary means.
    """
    alternatives = [alt.strip().lower() for alt in tag.lower().split("_or_")]
    if isinstance(value, str):
        return value.strip().lower() in alternatives
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    for alt in alternatives:
        if alt == "unity" and math.isclose(number, 1.0, rel_tol=1e-9):
            return True
        if alt == "noninverting" and number >= 1.0:
            return True
        if alt == "inverting" and number < 0.0:
            return True
        if alt == "any":
            return True
    return False


def rank(spec: Spec, patterns: list[Pattern]) -> list[Candidate]:
    """All candidates for ``spec``, eligible ones first, best fit first."""
    candidates = [evaluate(spec, p) for p in patterns]
    eligible = sorted((c for c in candidates if c.eligible), key=Candidate.sort_key)
    rejected = sorted(
        (c for c in candidates if not c.eligible), key=lambda c: c.pattern.id
    )
    return eligible + rejected


def select(spec: Spec, patterns: list[Pattern]) -> Pattern:
    """Choose the pattern implementing ``spec``.

    ``spec.topology`` forces a specific pattern; it must exist and must declare
    the spec's block, but its applicability ranges are advisory in that case so
    a user can deliberately push a topology outside its comfort zone.
    """
    if not patterns:
        raise NoPatternFound(
            "the pattern catalog is empty; expected directories under patterns/"
        )

    if spec.topology:
        by_id = {p.id: p for p in patterns}
        forced = by_id.get(spec.topology)
        if forced is None:
            raise NoPatternFound(
                f"spec {spec.name!r} forces topology {spec.topology!r}, which is not in "
                f"the catalog. Available: {', '.join(sorted(by_id))}"
            )
        if forced.block != spec.block:
            raise NoPatternFound(
                f"spec {spec.name!r} forces topology {spec.topology!r}, but that pattern "
                f"implements block {forced.block!r}, not {spec.block!r}"
            )
        return forced

    ordered = rank(spec, patterns)
    for candidate in ordered:
        if candidate.eligible:
            return candidate.pattern

    raise NoPatternFound(
        f"no pattern can implement spec {spec.name!r} (block {spec.block!r}).\n"
        + explain(spec, patterns)
        + "\nFix: relax the spec params, or add a pattern under patterns/ that declares "
        f"block: {spec.block}"
    )


def explain(spec: Spec, patterns: list[Pattern]) -> str:
    """Human-readable account of why each pattern was accepted or rejected."""
    lines: list[str] = []
    for candidate in rank(spec, patterns):
        verdict = f"eligible (fit {candidate.score:.3f})" if candidate.eligible else "rejected"
        lines.append(f"  {candidate.id}: {verdict}")
        lines.extend(f"      - {reason}" for reason in candidate.reasons)
    return "\n".join(lines)
