"""Pydantic v2 data contracts shared by every stage of the pipeline.

Unit policy: every numeric field is in SI base units (ohm, farad, henry, hertz,
second, volt, ampere).  The only exceptions are fields whose name ends in
``_percent`` (dimensionless, 0-100) or ``_db`` (decibels).  Formatting into
engineering notation happens at the render boundary and nowhere else.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ESeriesName = Literal["E6", "E12", "E24", "E48", "E96", "E192"]

PositiveFloat = Annotated[float, Field(gt=0.0)]
Percent = Annotated[float, Field(ge=0.0, le=100.0)]

_ID_UNSET = object()


class Requirement(BaseModel):
    """One machine-checkable requirement, traceable end to end.

    ``min``/``max`` are inclusive bounds expressed in ``unit``.  At least one of
    them must be given, otherwise the requirement asserts nothing.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    meas: str = Field(min_length=1)
    min: float | None = None
    max: float | None = None
    unit: str = ""
    description: str = ""

    @field_validator("id", "meas")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def _check_bounds(self) -> Requirement:
        if self.min is None and self.max is None:
            raise ValueError(
                f"requirement {self.id!r} on meas {self.meas!r} has neither 'min' nor 'max'; "
                "a requirement with no bound can never fail and is not traceable"
            )
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(
                f"requirement {self.id!r}: min ({self.min}) is greater than max ({self.max})"
            )
        return self

    def limit_text(self) -> str:
        """Human-readable bound, e.g. ``1900 .. 2100 Hz`` or ``<= -35 dB``."""
        unit = f" {self.unit}" if self.unit else ""
        if self.min is not None and self.max is not None:
            return f"{self.min:g} .. {self.max:g}{unit}"
        if self.min is not None:
            return f">= {self.min:g}{unit}"
        return f"<= {self.max:g}{unit}"

    def check(self, value: float | None) -> tuple[bool, float | None]:
        """Evaluate ``value`` against this requirement.

        Returns ``(passed, margin_percent)``.  ``margin_percent`` is the
        headroom to the nearest violated-first bound, as a percentage of the
        bound's magnitude; negative means the bound is violated.  A ``None``
        value (measurement failed) is a fail with unknown margin.
        """
        if value is None:
            return False, None
        margins: list[float] = []
        if self.min is not None:
            margins.append(_rel_margin(value - self.min, self.min))
        if self.max is not None:
            margins.append(_rel_margin(self.max - value, self.max))
        margin = min(margins)
        return margin >= 0.0, margin


def _rel_margin(slack: float, bound: float) -> float:
    """Slack expressed as a percentage of |bound| (or absolute if bound == 0)."""
    scale = abs(bound)
    if scale == 0.0:
        return slack * 100.0
    return 100.0 * slack / scale


class Spec(BaseModel):
    """User-facing input.  This is what the YAML files in ``examples/`` contain."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    block: str = Field(min_length=1)
    params: dict[str, float | str] = Field(default_factory=dict)
    requirements: list[Requirement] = Field(min_length=1)
    tolerance_percent: dict[str, Percent] = Field(default_factory=dict)
    eseries: ESeriesName = "E96"
    topology: str | None = None

    @field_validator("name")
    @classmethod
    def _safe_name(cls, v: str) -> str:
        v = v.strip()
        if any(c in v for c in "/\\ \t"):
            raise ValueError(
                f"spec name {v!r} must be filesystem-safe (no spaces or path separators); "
                "it is used as the output directory name"
            )
        return v

    @model_validator(mode="after")
    def _unique_requirement_ids(self) -> Spec:
        seen: set[str] = set()
        for req in self.requirements:
            if req.id in seen:
                raise ValueError(
                    f"duplicate requirement id {req.id!r} in spec {self.name!r}; "
                    "each requirement must map to exactly one report row"
                )
            seen.add(req.id)
        return self

    def num_param(self, key: str) -> float:
        """Return ``params[key]`` as a float, with an actionable error otherwise."""
        if key not in self.params:
            raise KeyError(
                f"spec {self.name!r} is missing required numeric param {key!r}; "
                f"add '{key}: <value>' under 'params:'"
            )
        raw = self.params[key]
        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"spec {self.name!r} param {key!r} = {raw!r} is not numeric"
            ) from exc

    def str_param(self, key: str, default: str | None = None) -> str:
        """Return ``params[key]`` as a lowercased string."""
        if key not in self.params:
            if default is not None:
                return default
            raise KeyError(f"spec {self.name!r} is missing required param {key!r}")
        return str(self.params[key]).strip().lower()

    def tolerance_for(self, ref: str) -> float:
        """Tolerance in percent for a component ref, keyed by its letter prefix."""
        prefix = ref[0].upper() if ref else ""
        return float(self.tolerance_percent.get(prefix, 0.0))

    def requirements_for(self, meas: str) -> list[Requirement]:
        return [r for r in self.requirements if r.meas == meas]

    def hash(self) -> str:
        """Stable sha256 of the spec content (used for caching and the report footer)."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Component(BaseModel):
    """A single placed part.  ``value``/``ideal_value`` are in SI base units."""

    model_config = ConfigDict(extra="forbid")

    ref: str = Field(min_length=1)
    value: float
    unit: str
    ideal_value: float
    tolerance_percent: Percent = 0.0
    eseries: ESeriesName | None = None
    model: str | None = None

    @field_validator("unit")
    @classmethod
    def _known_unit(cls, v: str) -> str:
        v = v.strip()
        allowed = {"ohm", "F", "H", "V", "A", "", "device"}
        if v not in allowed:
            raise ValueError(
                f"unit {v!r} is not one of {sorted(allowed)}; internal values must be SI base units"
            )
        return v

    @property
    def deviation_percent(self) -> float:
        """Signed deviation of the snapped value from the ideal value, in percent."""
        if self.ideal_value == 0.0:
            return 0.0
        return 100.0 * (self.value - self.ideal_value) / self.ideal_value

    def spice_value(self) -> str:
        """Value formatted for a SPICE netlist (plain float, no unit suffix)."""
        return f"{self.value:.6g}"


class MeasResult(BaseModel):
    """One ``.meas`` result parsed out of the ngspice log."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float | None
    raw: str = ""


class RequirementResult(BaseModel):
    """Verdict for a single requirement, derived only from ngspice measurements."""

    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    meas: str
    value: float | None
    passed: bool
    margin_percent: float | None = None
    limit_text: str = ""
    description: str = ""
    unit: str = ""
    mc_pass_rate: float | None = None

    @property
    def status(self) -> str:
        if self.value is None:
            return "NO-DATA"
        return "PASS" if self.passed else "FAIL"


class SensitivityResult(BaseModel):
    """Monte-Carlo distribution of one measurement over component tolerances."""

    model_config = ConfigDict(extra="forbid")

    meas: str
    nominal: float
    p5: float
    p50: float
    p95: float
    worst: float
    pass_rate: float = Field(ge=0.0, le=1.0)
    n_samples: int = 0
    n_failed_runs: int = 0
    samples: list[float] = Field(default_factory=list)


class DesignResult(BaseModel):
    """The single source of truth for the report.  Serialized to out/design.json."""

    model_config = ConfigDict(extra="forbid")

    spec: Spec
    pattern_id: str
    components: list[Component] = Field(default_factory=list)
    netlist: str = ""
    measurements: list[MeasResult] = Field(default_factory=list)
    requirement_results: list[RequirementResult] = Field(default_factory=list)
    sensitivity: list[SensitivityResult] = Field(default_factory=list)
    passed: bool = False
    iterations: int = 1
    warnings: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    converged: bool = True
    spice_attempts: int = 1
    tool_version: str = ""
    ngspice_version: str = ""
    timestamp: str = ""
    spec_hash: str = ""
    gotchas: list[dict[str, str]] = Field(default_factory=list)

    def measurement(self, name: str) -> float | None:
        for m in self.measurements:
            if m.name == name:
                return m.value
        return None

    def verdict(self) -> str:
        """PASS / FAIL / NOT-CONVERGED - the banner text and the exit-code source."""
        if not self.converged:
            return "NOT-CONVERGED"
        return "PASS" if self.passed else "FAIL"

    def exit_code(self) -> int:
        """0 on PASS, 1 on FAIL, 2 on non-convergence or tool error."""
        verdict = self.verdict()
        if verdict == "PASS":
            return 0
        if verdict == "FAIL":
            return 1
        return 2

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> DesignResult:
        data: Any = json.loads(text)
        return cls.model_validate(data)


__all__ = [
    "Component",
    "DesignResult",
    "ESeriesName",
    "MeasResult",
    "Requirement",
    "RequirementResult",
    "SensitivityResult",
    "Spec",
]
