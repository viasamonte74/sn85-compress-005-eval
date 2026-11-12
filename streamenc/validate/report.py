from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from streamenc.validate.official_streaming import ValidityViolation


@dataclass
class GateResult:
    name: str
    status: str  # pass | fail | skip | unverified
    detail: str = ""
    measured: float | None = None
    limit: float | None = None


@dataclass
class ValidationReport:
    reference: str
    candidate: str
    passed: bool
    gates: list[GateResult] = field(default_factory=list)
    violations: list[ValidityViolation] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    unverified: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "candidate": self.candidate,
            "passed": self.passed,
            "gates": [asdict(g) for g in self.gates],
            "violations": [asdict(v) for v in self.violations],
            "metrics": self.metrics,
            "unverified": list(self.unverified),
        }
