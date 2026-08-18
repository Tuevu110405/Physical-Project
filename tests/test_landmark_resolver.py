from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantiphy_baseline.measurement_plan import build_measurement_plan  # noqa: E402
from quantiphy_baseline.vision.landmark_candidates import (  # noqa: E402
    generate_landmark_candidates,
)
from quantiphy_baseline.vision.landmark_resolver import (  # noqa: E402
    LandmarkResolver,
    TrackMasks,
)
from quantiphy_baseline.solver_2d import TwoDSolver  # noqa: E402


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
        "temporal": {"mode": "instantaneous", "time_s": 1.0},
        "target_entities": ["the outer end of the pier", "the boat"],
    }


def test_major_axis_end_caps_are_two_distinct_boundaries():
    mask = np.zeros((60, 100), dtype=np.uint8)
    mask[25:35, 10:90] = 1
    candidates = generate_landmark_candidates(
        mask, "major_axis_end_caps", "boundary_segment"
    )
    assert [item.candidate_id for item in candidates] == [
        "end_negative",
        "end_positive",
    ]
    centers = [np.mean(item.coordinates_xy, axis=0) for item in candidates]
    assert centers[0][0] < 20
    assert centers[1][0] > 80


def test_outer_end_uses_reference_mask_to_choose_opposite_end():
    plan = build_measurement_plan(_question())
    frames = [Image.new("RGB", (100, 60), "white") for _ in range(2)]
    pier = np.zeros((2, 60, 100), dtype=np.uint8)
    pier[:, 25:35, 10:90] = 1
    shore = np.zeros_like(pier)
    shore[:, :, :12] = 1
    boat = np.zeros_like(pier)
    boat[:, 40:48, 85:95] = 1
    tracks = {
        "pier": [TrackMasks("pier__0", pier)],
        "shore": [TrackMasks("shore__0", shore)],
        "boat": [TrackMasks("boat__0", boat)],
    }
    results = LandmarkResolver().resolve_plans(
        plans=[plan], frames=frames, fps=1.0, tracks=tracks
    )
    pier_result = next(item for item in results if item.operand_id == "from")
    assert pier_result.status == "resolved"
    assert pier_result.candidate_id == "end_positive"
    assert np.mean(pier_result.coordinates_xy, axis=0)[0] > 80


def test_solver_consumes_landmark_results_instead_of_reparsing_question():
    plan = build_measurement_plan(_question())
    frames = [Image.new("RGB", (110, 60), "white") for _ in range(2)]
    pier = np.zeros((2, 60, 110), dtype=np.uint8)
    pier[:, 25:35, 10:90] = 1
    shore = np.zeros_like(pier)
    shore[:, :, :12] = 1
    boat = np.zeros_like(pier)
    boat[:, 40:48, 96:105] = 1
    tracks = {
        "pier": [TrackMasks("pier__0", pier)],
        "shore": [TrackMasks("shore__0", shore)],
        "boat": [TrackMasks("boat__0", boat)],
    }
    landmarks = LandmarkResolver().resolve_plans(
        plans=[plan], frames=frames, fps=1.0, tracks=tracks
    )
    result = TwoDSolver().solve_plan(plan, landmarks, tracks, fps=1.0)
    assert result.status == "solved_pixel"
    assert result.value_px is not None and result.value_px > 0
    assert result.value is None
    assert "non-planar" in " ".join(result.warnings)
