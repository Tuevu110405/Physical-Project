from .entity_specs import TrackingRequest, build_tracking_requests

__all__ = [
    "TrackingRequest",
    "build_tracking_requests",
    "GroundingDinoGrounder",
    "Sam2Tracker",
    "SegmentTrackPipeline",
    "load_bitpacked_masks",
]


def __getattr__(name):
    if name == "GroundingDinoGrounder":
        from .grounder import GroundingDinoGrounder
        return GroundingDinoGrounder
    if name == "Sam2Tracker":
        from .sam2_tracker import Sam2Tracker
        return Sam2Tracker
    if name in {"SegmentTrackPipeline", "load_bitpacked_masks"}:
        from .pipeline import SegmentTrackPipeline, load_bitpacked_masks
        return {"SegmentTrackPipeline": SegmentTrackPipeline, "load_bitpacked_masks": load_bitpacked_masks}[name]
    raise AttributeError(name)
