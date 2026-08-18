from .entity_specs import TrackingRequest, build_tracking_requests
from quantiphy_baseline.measurement_plan import (
    CanonicalEntity,
    LandmarkSpec,
    MeasurementOperand,
    MeasurementPlan,
    build_measurement_plan,
    build_measurement_plans,
    canonicalize_entity,
)

__all__ = [
    "TrackingRequest",
    "build_tracking_requests",
    "CanonicalEntity",
    "LandmarkSpec",
    "MeasurementOperand",
    "MeasurementPlan",
    "build_measurement_plan",
    "build_measurement_plans",
    "canonicalize_entity",
    "GroundingDinoGrounder",
    "Sam2Tracker",
    "SegmentTrackPipeline",
    "load_bitpacked_masks",
    "load_track_masks_artifact",
    "LandmarkResolver",
    "LandmarkCandidate",
    "LandmarkResult",
    "TrackMasks",
    "OpenAICompatibleVLMSelector",
    "TwoDSolver",
]


def __getattr__(name):
    if name == "GroundingDinoGrounder":
        from .grounder import GroundingDinoGrounder
        return GroundingDinoGrounder
    if name == "Sam2Tracker":
        from .sam2_tracker import Sam2Tracker
        return Sam2Tracker
    if name in {"SegmentTrackPipeline", "load_bitpacked_masks", "load_track_masks_artifact"}:
        from .pipeline import (
            SegmentTrackPipeline,
            load_bitpacked_masks,
            load_track_masks_artifact,
        )
        return {
            "SegmentTrackPipeline": SegmentTrackPipeline,
            "load_bitpacked_masks": load_bitpacked_masks,
            "load_track_masks_artifact": load_track_masks_artifact,
        }[name]
    if name in {"LandmarkResolver", "TrackMasks"}:
        from .landmark_resolver import LandmarkResolver, TrackMasks
        return {"LandmarkResolver": LandmarkResolver, "TrackMasks": TrackMasks}[name]
    if name in {"LandmarkCandidate", "LandmarkResult"}:
        from .landmark_types import LandmarkCandidate, LandmarkResult
        return {
            "LandmarkCandidate": LandmarkCandidate,
            "LandmarkResult": LandmarkResult,
        }[name]
    if name == "OpenAICompatibleVLMSelector":
        from .vlm_selector import OpenAICompatibleVLMSelector
        return OpenAICompatibleVLMSelector
    if name == "TwoDSolver":
        from quantiphy_baseline.solver_2d import TwoDSolver
        return TwoDSolver
    raise AttributeError(name)
