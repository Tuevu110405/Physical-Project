from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantiphy_baseline.vision.entity_specs import (
    build_tracking_requests,
    extract_reference_entity,
    infer_visual_kind,
    prompt_candidates,
)
from quantiphy_baseline.vision.grounder import box_iou


def test_reference_from_of_phrase():
    assert extract_reference_entity("diameter of the ping pong ball", []) == "the ping pong ball"


def test_reference_matches_group_entity():
    candidates = ["the turning yellow car", "the walking pedestrian"]
    assert extract_reference_entity("walking velocity", candidates) == "the walking pedestrian"


def test_pair_entity():
    kind, count, _ = infer_visual_kind("between the two black road signs")
    assert kind == "pair" and count == 2
    assert "black road sign" in prompt_candidates("between the two black road signs")


def test_box_iou():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 5, 5], [6, 6, 10, 10]) == 0.0


def test_build_requests_target_and_reference():
    group = {
        "questions": [
            {
                "qa_id": "1",
                "target_entities": ["the orange ball"],
                "temporal": {"time_s": 1.0, "start_s": None, "end_s": None},
                "prior": {
                    "description": "billiard ball diameter",
                    "timestamp_s": None,
                },
            }
        ]
    }
    reqs = build_tracking_requests(group)
    names = {r.entity_key: r for r in reqs}
    assert "orange_ball" in names
    assert any("reference" in r.roles for r in reqs)
