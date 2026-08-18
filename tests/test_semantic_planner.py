from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantiphy_baseline.measurement_plan import build_measurement_plan  # noqa: E402
from quantiphy_baseline.planning import (  # noqa: E402
    LLMSemanticPlanner,
    SemanticPlan,
    SemanticPlanValidationError,
)
from quantiphy_baseline.vision.entity_specs import build_tracking_requests  # noqa: E402


def _outer_end_plan():
    return {
        "relation": "distance_between",
        "operands": [
            {
                "operand_id": "from",
                "role": "from",
                "parent": {
                    "entity_id": "pier",
                    "category": "pier",
                    "referring_expression": "the wooden pier",
                    "tracking_prompts": ["wooden pier", "pier"],
                },
                "landmark": {
                    "raw_text": "outer end of the pier",
                    "semantic_type": "outer_end",
                    "geometry_type": "boundary_segment",
                    "candidate_generator": "major_axis_end_caps",
                    "selector": {
                        "operator": "opposite_to_attachment",
                        "reference_entity_id": "shore",
                        "direction_frame": "scene",
                    },
                    "resolution_policy": "geometry_then_vlm",
                    "status": "planned",
                },
            },
            {
                "operand_id": "to",
                "role": "to",
                "parent": {
                    "entity_id": "boat",
                    "category": "boat",
                    "referring_expression": "the boat",
                    "tracking_prompts": ["boat"],
                },
                "landmark": {
                    "raw_text": "boat",
                    "semantic_type": "whole_object",
                    "geometry_type": "mask",
                    "candidate_generator": "whole_mask",
                    "selector": None,
                    "resolution_policy": "geometry_only",
                    "status": "planned",
                },
            },
        ],
        "auxiliary_entities": [
            {
                "entity_id": "shore",
                "category": "shore",
                "referring_expression": "shore",
                "tracking_prompts": ["shoreline", "shore"],
                "role": "selector_reference",
                "tracking_requirement": "optional",
            }
        ],
    }


def _question():
    return {
        "qa_id": "1095",
        "video_id": "simulation_0009",
        "video_type": "V3MC",
        "inference_type": "DS",
        "raw_question": "When t=3s, what is the minimum distance from the outer end of the pier to the boat in meters?",
        "quantity_family": "distance",
        "quantity_subtype": "minimum_distance",
        "output_unit": "m",
        "relation_type": "from_to",
        "temporal": {"mode": "instantaneous", "time_s": 3.0},
        "target_entities": ["the outer end of the pier", "the boat"],
    }


def test_schema_rejects_silent_whole_object_fallback():
    raw = _outer_end_plan()
    raw["operands"][0]["landmark"] = {
        "raw_text": "outer end of the pier",
        "semantic_type": "whole_object",
        "geometry_type": "mask",
        "candidate_generator": "whole_mask",
    }
    with pytest.raises(SemanticPlanValidationError):
        SemanticPlan.from_dict(raw)


def test_llm_planner_validates_and_caches(tmp_path):
    calls = []

    def transport(payload):
        calls.append(payload)
        return _outer_end_plan()

    planner = LLMSemanticPlanner(
        "fake-model", cache_path=tmp_path / "plans.jsonl", transport=transport
    )
    first = planner.plan(_question())
    second = planner.plan(_question())
    assert first == second
    assert len(calls) == 1


def test_semantic_plan_builds_operands_without_reparsing_landmark_text():
    semantic = SemanticPlan.from_dict(_outer_end_plan())
    plan = build_measurement_plan(_question(), semantic_plan=semantic)
    assert plan.planner_version == "llm-semantic-v1"
    assert plan.relation == "distance_between"
    assert plan.operands[0].tracking_key == "pier"
    assert plan.operands[0].landmark.candidate_generator == "major_axis_end_caps"
    assert plan.operands[0].landmark.selector["reference_entity_id"] == "shore"
    assert plan.auxiliary_entities[0]["entity_id"] == "shore"


def test_tracking_uses_semantic_parent_and_adds_auxiliary_reference():
    semantic = SemanticPlan.from_dict(_outer_end_plan())
    plan = build_measurement_plan(_question(), semantic_plan=semantic)
    group = {"video_id": "simulation_0009", "questions": [_question()]}
    requests = {item.entity_key: item for item in build_tracking_requests(group, [plan])}
    assert requests["pier"].prompts[:2] == ["wooden pier", "pier"]
    assert requests["pier"].landmarks[0]["kind"] == "outer_end"
    assert "shore" in requests
    assert "reference" in requests["shore"].roles
