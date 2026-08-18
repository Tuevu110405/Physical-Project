from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LandmarkCandidate:
    candidate_id: str
    geometry_type: str
    coordinates_xy: list[list[float]]
    score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LandmarkResult:
    plan_id: str
    operand_id: str
    landmark_id: str
    parent_track_id: str | None
    frame_idx: int
    time_s: float
    geometry_type: str
    coordinates_xy: list[list[float]]
    candidate_id: str | None
    method: str
    confidence: float
    uncertainty_px: float | None
    status: str
    evidence: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
