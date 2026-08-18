from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from quantiphy_baseline.measurement_plan import LandmarkSpec, MeasurementOperand, MeasurementPlan

from .landmark_candidates import generate_landmark_candidates
from .landmark_types import LandmarkCandidate, LandmarkResult
from .vlm_selector import CandidateSelector


@dataclass
class TrackMasks:
    track_id: str
    masks: np.ndarray
    mask_path: str | None = None


def _candidate_center(candidate: LandmarkCandidate) -> np.ndarray:
    points = np.asarray(candidate.coordinates_xy, dtype=np.float32)
    if not len(points):
        return np.array([np.nan, np.nan], dtype=np.float32)
    return points.mean(axis=0)


def _reference_distance(candidate: LandmarkCandidate, reference_mask: np.ndarray) -> float:
    center = _candidate_center(candidate)
    if not np.isfinite(center).all():
        return float("inf")
    distance = cv2.distanceTransform(
        (~np.asarray(reference_mask).astype(bool)).astype(np.uint8), cv2.DIST_L2, 3
    )
    x = int(np.clip(round(float(center[0])), 0, distance.shape[1] - 1))
    y = int(np.clip(round(float(center[1])), 0, distance.shape[0] - 1))
    return float(distance[y, x])


def _directional_choice(
    candidates: list[LandmarkCandidate], operator: str
) -> LandmarkCandidate:
    centers = np.stack([_candidate_center(item) for item in candidates])
    if operator == "leftmost":
        index = int(np.nanargmin(centers[:, 0]))
    elif operator == "rightmost":
        index = int(np.nanargmax(centers[:, 0]))
    elif operator == "topmost":
        index = int(np.nanargmin(centers[:, 1]))
    else:
        index = int(np.nanargmax(centers[:, 1]))
    return candidates[index]


class LandmarkResolver:
    def __init__(
        self,
        vlm_selector: CandidateSelector | None = None,
        *,
        default_uncertainty_px: float = 2.0,
    ) -> None:
        self.vlm_selector = vlm_selector
        self.default_uncertainty_px = default_uncertainty_px

    @staticmethod
    def _frame_indices(plan: MeasurementPlan, num_frames: int, fps: float) -> list[int]:
        temporal = plan.temporal or {}
        values = [temporal.get("time_s")]
        if temporal.get("mode") == "interval":
            values.extend([temporal.get("start_s"), temporal.get("end_s")])
        indices = {
            int(np.clip(round(float(value) * fps), 0, num_frames - 1))
            for value in values
            if isinstance(value, (int, float))
        }
        if indices:
            if plan.measurement_kind in {"speed", "velocity"}:
                center = next(iter(indices))
                indices.update(
                    index for index in (center - 1, center + 1) if 0 <= index < num_frames
                )
            elif plan.measurement_kind == "acceleration":
                center = next(iter(indices))
                indices.update(
                    index for index in (center - 1, center + 1) if 0 <= index < num_frames
                )
            return sorted(indices)
        if temporal.get("mode") in {"whole_video_average", "whole_video_total"}:
            return list(range(num_frames))
        return [num_frames // 2]

    @staticmethod
    def _find_reference_track(
        landmark: LandmarkSpec,
        tracks: dict[str, list[TrackMasks]],
    ) -> TrackMasks | None:
        selector = landmark.selector or {}
        reference_id = selector.get("reference_entity_id")
        if not reference_id:
            return None
        variants = [str(reference_id), str(reference_id).replace(" ", "_")]
        for key in variants:
            if tracks.get(key):
                return tracks[key][0]
        return None

    def _choose_candidate(
        self,
        frame: Image.Image,
        parent_mask: np.ndarray,
        candidates: list[LandmarkCandidate],
        landmark: LandmarkSpec,
        reference_mask: np.ndarray | None,
    ) -> tuple[LandmarkCandidate | None, float, str, dict[str, Any], list[str]]:
        warnings: list[str] = []
        selector = landmark.selector or {}
        operator = selector.get("operator")
        if not candidates:
            return None, 0.0, landmark.candidate_generator, {}, [
                f"candidate generator {landmark.candidate_generator} produced no candidates"
            ]
        if landmark.candidate_generator == "whole_mask":
            return candidates[0], 1.0, "whole_mask", {}, warnings
        if operator in {"leftmost", "rightmost", "topmost", "bottommost"}:
            return _directional_choice(candidates, operator), 0.95, f"geometry:{operator}", {}, warnings
        if operator in {
            "closest_to",
            "attached_to",
            "farthest_from",
            "opposite_to_attachment",
        } and reference_mask is not None:
            distances = [_reference_distance(item, reference_mask) for item in candidates]
            if operator in {"closest_to", "attached_to"}:
                index = int(np.argmin(distances))
            else:
                index = int(np.argmax(distances))
            evidence = {"reference_distances_px": dict(zip(
                [item.candidate_id for item in candidates], distances
            ))}
            return candidates[index], 0.94, f"geometry:{operator}", evidence, warnings
        if operator in {"inside", "outside"} and reference_mask is not None:
            scores = []
            for candidate in candidates:
                center = _candidate_center(candidate)
                x = int(np.clip(round(float(center[0])), 0, reference_mask.shape[1] - 1))
                y = int(np.clip(round(float(center[1])), 0, reference_mask.shape[0] - 1))
                scores.append(bool(reference_mask[y, x]))
            desired = operator == "inside"
            matches = [index for index, value in enumerate(scores) if value == desired]
            if len(matches) == 1:
                return candidates[matches[0]], 0.9, f"geometry:{operator}", {}, warnings

        needs_semantics = operator in {
            "semantic_match",
            "facing",
            "opposite_to_attachment",
            "attached_to",
            "closest_to",
            "farthest_from",
            None,
        }
        if needs_semantics and self.vlm_selector is not None:
            prompt = f"Select the candidate that corresponds to: {landmark.raw_text or landmark.kind}."
            selected_id, confidence, evidence = self.vlm_selector.select(
                frame, parent_mask, candidates, prompt, reference_mask
            )
            selected = next(item for item in candidates if item.candidate_id == selected_id)
            return selected, confidence, "vlm_candidate_selection", evidence, warnings
        if len(candidates) == 1 and operator not in {"semantic_match", "facing"}:
            return candidates[0], 0.85, f"geometry:{landmark.candidate_generator}", {}, warnings
        if reference_mask is None and selector.get("reference_entity_id"):
            warnings.append(
                f"reference track {selector['reference_entity_id']} is unavailable"
            )
        warnings.append("semantic candidate selection requires a VLM selector")
        return None, 0.0, "unresolved", {}, warnings

    def resolve_operand(
        self,
        plan: MeasurementPlan,
        operand: MeasurementOperand,
        frames: list[Image.Image],
        fps: float,
        tracks: dict[str, list[TrackMasks]],
    ) -> list[LandmarkResult]:
        parent_tracks = tracks.get(operand.tracking_key) or []
        indices = self._frame_indices(plan, len(frames), fps)
        landmark_id = f"{operand.tracking_key}:{operand.operand_id or operand.role}:{operand.landmark.kind}"
        if not parent_tracks:
            return [
                LandmarkResult(
                    plan_id=plan.plan_id,
                    operand_id=operand.operand_id or operand.role,
                    landmark_id=landmark_id,
                    parent_track_id=None,
                    frame_idx=frame_idx,
                    time_s=frame_idx / fps,
                    geometry_type=operand.landmark.geometry_type,
                    coordinates_xy=[],
                    candidate_id=None,
                    method="unresolved",
                    confidence=0.0,
                    uncertainty_px=None,
                    status="unresolved",
                    warnings=[f"parent track {operand.tracking_key} is unavailable"],
                )
                for frame_idx in indices
            ]
        reference_track = self._find_reference_track(operand.landmark, tracks)
        results: list[LandmarkResult] = []
        selected_tracks = parent_tracks[: max(1, operand.expected_instances)]
        for instance_index, track in enumerate(selected_tracks):
            result_operand_id = operand.operand_id or operand.role
            result_landmark_id = landmark_id
            if len(selected_tracks) > 1:
                result_operand_id = f"{result_operand_id}:{instance_index}"
                result_landmark_id = f"{landmark_id}:{instance_index}"
            for frame_idx in indices:
                if frame_idx >= len(track.masks) or not np.asarray(track.masks[frame_idx]).any():
                    results.append(
                        LandmarkResult(
                            plan_id=plan.plan_id,
                            operand_id=result_operand_id,
                            landmark_id=result_landmark_id,
                            parent_track_id=track.track_id,
                            frame_idx=frame_idx,
                            time_s=frame_idx / fps,
                            geometry_type=operand.landmark.geometry_type,
                            coordinates_xy=[],
                            candidate_id=None,
                            method="unresolved",
                            confidence=0.0,
                            uncertainty_px=None,
                            status="unresolved",
                            evidence={"instance_index": instance_index},
                            warnings=["parent mask is empty at requested frame"],
                        )
                    )
                    continue
                parent_mask = np.asarray(track.masks[frame_idx]).astype(bool)
                reference_mask = None
                if reference_track is not None and frame_idx < len(reference_track.masks):
                    reference_mask = np.asarray(reference_track.masks[frame_idx]).astype(bool)
                selector_operator = (operand.landmark.selector or {}).get("operator")
                candidates = generate_landmark_candidates(
                    parent_mask,
                    operand.landmark.candidate_generator,
                    operand.landmark.geometry_type,
                    selector_operator=selector_operator,
                    reference_mask=reference_mask,
                )
                selected, confidence, method, evidence, warnings = self._choose_candidate(
                    frames[frame_idx], parent_mask, candidates, operand.landmark, reference_mask
                )
                status = "resolved" if selected is not None else "unresolved"
                result_evidence = {
                    **evidence,
                    "instance_index": instance_index,
                    "candidate_generator": operand.landmark.candidate_generator,
                    "candidates": [item.to_dict() for item in candidates],
                    "parent_mask_path": track.mask_path,
                }
                results.append(
                    LandmarkResult(
                        plan_id=plan.plan_id,
                        operand_id=result_operand_id,
                        landmark_id=result_landmark_id,
                        parent_track_id=track.track_id,
                        frame_idx=frame_idx,
                        time_s=frame_idx / fps,
                        geometry_type=operand.landmark.geometry_type,
                        coordinates_xy=selected.coordinates_xy if selected else [],
                        candidate_id=selected.candidate_id if selected else None,
                        method=method,
                        confidence=confidence,
                        uncertainty_px=self.default_uncertainty_px if selected else None,
                        status=status,
                        evidence=result_evidence,
                        warnings=warnings,
                    )
                )
        return results

    def resolve_plans(
        self,
        plans: list[MeasurementPlan],
        frames: list[Image.Image],
        fps: float,
        tracks: dict[str, list[TrackMasks]],
    ) -> list[LandmarkResult]:
        results: list[LandmarkResult] = []
        for plan in plans:
            for operand in plan.operands:
                results.extend(self.resolve_operand(plan, operand, frames, fps, tracks))
        return results
