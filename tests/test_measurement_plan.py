from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantiphy_baseline.measurement_plan import (  # noqa: E402
    MeasurementPlan,
    build_measurement_plan,
    canonicalize_entity,
)
from quantiphy_baseline.vision.entity_specs import build_tracking_requests  # noqa: E402


def _question(**updates):
    question = {
        "qa_id": "1095",
        "video_id": "simulation_0009",
        "video_type": "V3MC",
        "inference_type": "DS",
        "raw_question": (
            "When t=3s, what is the minimum distance from the outer end "
            "of the pier to the boat in meters?"
        ),
        "quantity_family": "distance",
        "quantity_subtype": "minimum_distance",
        "output_unit": "m",
        "component": "scalar",
        "relation_type": "from_to",
        "temporal": {"mode": "instantaneous", "time_s": 3.0},
        "target_entities": ["the outer end of the pier", "the boat"],
        "prior": {"description": "speed of the boat", "value": 1.5, "unit": "m/s"},
        "depth_observations": [{"key": "distance_pier_camera", "value": 18.92, "unit": "m"}],
    }
    question.update(updates)
    return question


def test_canonicalizes_outer_end_to_parent_pier():
    entity = canonicalize_entity("the outer end of the pier")
    assert entity.parent_key == "pier"
    assert entity.tracking_key == "pier"
    assert entity.landmark.kind == "outer_end"
    assert entity.landmark.extraction == "opposite_attachment_end_cap"
    assert entity.prompts[0] == "pier"


def test_canonicalizes_pencil_tip_to_parent_pencil():
    entity = canonicalize_entity("the pencil tip moving on the block")
    assert entity.tracking_key == "pencil"
    assert entity.landmark.kind == "tip"
    assert entity.landmark.extraction == "skeleton_endpoint"


def test_preserves_instance_selector_for_left_and_right_objects():
    left = canonicalize_entity("the left tennis ball")
    right = canonicalize_entity("the right tennis ball")
    assert left.parent_key == right.parent_key == "tennis_ball"
    assert left.tracking_key == "left_tennis_ball"
    assert right.tracking_key == "right_tennis_ball"


def test_pair_tracks_two_instances_of_one_parent():
    entity = canonicalize_entity("between the two black road signs")
    assert entity.parent_key == "road_sign"
    assert entity.tracking_key == "road_sign"
    assert entity.expected_instances == 2
    assert "black road sign" in entity.prompts


def test_minimum_distance_plan_uses_landmark_and_boat_boundary():
    plan = build_measurement_plan(_question())
    assert plan.measurement_kind == "distance"
    assert plan.reduction == "minimum"
    assert plan.temporal["time_s"] == 3.0
    assert [(x.role, x.tracking_key, x.landmark.kind) for x in plan.operands] == [
        ("from", "pier", "outer_end"),
        ("to", "boat", "whole_object"),
    ]
    assert plan.calibration["physical_prior"] == "velocity"
    assert plan.calibration["dimensionality"] == "depth_aware"
    assert plan.calibration["prior_state"] == "dynamic"
    assert plan.calibration["target_state"] == "static"


def test_edge_to_edge_plan_uses_one_parent_track_and_two_landmarks():
    plan = build_measurement_plan(
        _question(
            qa_id="1094",
            raw_question=(
                "What is the length of the pier from the shore-side edge "
                "to the water-side edge in meters?"
            ),
            quantity_family="size",
            quantity_subtype="length",
            relation_type="attribute",
            temporal={"mode": "static"},
            target_entities=["the pier from the shore-side edge to the water-side edge"],
        )
    )
    assert plan.measurement_kind == "extent"
    assert plan.relation == "between_landmarks"
    assert plan.axis == "long_axis"
    assert {x.tracking_key for x in plan.operands} == {"pier"}
    assert [x.landmark.kind for x in plan.operands] == [
        "shore_side_edge",
        "water_side_edge",
    ]


def test_height_above_plan_becomes_vertical_distance():
    plan = build_measurement_plan(
        _question(
            qa_id="1097",
            raw_question="What is the height of the pier deck above the water surface in meters?",
            quantity_family="size",
            quantity_subtype="height",
            relation_type="attribute",
            temporal={"mode": "static"},
            target_entities=["the pier deck above the water surface"],
        )
    )
    assert plan.measurement_kind == "distance"
    assert plan.relation == "vertical_distance"
    assert plan.axis == "vertical"
    assert [(x.tracking_key, x.landmark.kind) for x in plan.operands] == [
        ("water", "surface"),
        ("pier", "deck"),
    ]


def test_tracking_requests_merge_pier_aliases_but_keep_landmarks():
    group = {
        "video_id": "simulation_0009",
        "video_type": "V3MC",
        "fps": 24,
        "questions": [
            _question(),
            _question(
                qa_id="1093",
                raw_question="What is the width of the wooden pier in meters?",
                quantity_family="size",
                quantity_subtype="width",
                relation_type="attribute",
                temporal={"mode": "static"},
                target_entities=["the wooden pier"],
            ),
        ],
    }
    requests = {request.entity_key: request for request in build_tracking_requests(group)}
    assert "pier" in requests
    assert "outer_end_of_the_pier" not in requests
    assert {x["kind"] for x in requests["pier"].landmarks} == {"outer_end"}
    assert set(requests["pier"].source_qa_ids) == {"1095", "1093"}


def test_measurement_plan_round_trip_for_cross_session_artifact():
    original = build_measurement_plan(_question())
    restored = MeasurementPlan.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()
