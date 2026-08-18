from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

import cv2
import numpy as np

from quantiphy_baseline.measurement_plan import MeasurementPlan
from quantiphy_baseline.vision.landmark_resolver import TrackMasks
from quantiphy_baseline.vision.landmark_types import LandmarkResult


@dataclass
class SolverResult:
    plan_id: str
    qa_id: str
    status: str
    measurement_kind: str
    value_px: float | None
    pixel_unit: str | None
    value: float | None
    output_unit: str | None
    method: str
    frame_indices: list[int] = field(default_factory=list)
    uncertainty_px: float | None = None
    warnings: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _largest_contour(mask: np.ndarray, limit: int = 512) -> np.ndarray:
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return np.empty((0, 2), dtype=np.float32)
    points = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(points) > limit:
        points = points[np.linspace(0, len(points) - 1, limit, dtype=int)]
    return points


def _centroid(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return np.array([xs.mean(), ys.mean()], dtype=np.float64)


def _pairwise_distance(a: np.ndarray, b: np.ndarray, reduction: str = "minimum") -> float:
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    if reduction == "maximum":
        return float(distances.max())
    return float(distances.min())


def _principal_extent(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    projection = (points - center) @ axis
    return float(projection.max() - projection.min())


def _extent(points: np.ndarray, subtype: str, axis: str | None) -> float:
    if not len(points):
        raise ValueError("empty geometry")
    if subtype in {"height", "thickness"} or axis == "vertical":
        return float(np.ptp(points[:, 1]))
    if subtype == "width" or axis == "horizontal":
        return float(np.ptp(points[:, 0]))
    if subtype in {"diameter", "wingspan", "shoulder_breadth"}:
        return _pairwise_distance(points, points, reduction="maximum")
    return _principal_extent(points)


def _length_unit(unit: str | None) -> str | None:
    match = re.search(r"(?<![a-z])(mm|cm|m)(?![a-z])", str(unit or "").lower())
    return match.group(1) if match else None


LENGTH_TO_M = {"mm": 0.001, "cm": 0.01, "m": 1.0}


class TwoDSolver:
    """Deterministic pixel-geometry solver; physical calibration is planar-only."""

    @staticmethod
    def _track_index(tracks: dict[str, list[TrackMasks]]) -> dict[str, TrackMasks]:
        return {track.track_id: track for items in tracks.values() for track in items}

    @staticmethod
    def _points(
        result: LandmarkResult, track_index: dict[str, TrackMasks]
    ) -> np.ndarray:
        if result.coordinates_xy:
            return np.asarray(result.coordinates_xy, dtype=np.float32).reshape(-1, 2)
        if result.geometry_type == "mask" and result.parent_track_id in track_index:
            track = track_index[result.parent_track_id]
            if result.frame_idx < len(track.masks):
                return _largest_contour(track.masks[result.frame_idx])
        return np.empty((0, 2), dtype=np.float32)

    @staticmethod
    def _results_for_plan(
        plan: MeasurementPlan, results: list[LandmarkResult]
    ) -> list[LandmarkResult]:
        return [item for item in results if item.plan_id == plan.plan_id]

    def _pixel_measurement(
        self,
        plan: MeasurementPlan,
        results: list[LandmarkResult],
        track_index: dict[str, TrackMasks],
    ) -> tuple[float, str, list[int], float | None, dict[str, Any]]:
        resolved = [item for item in results if item.status == "resolved"]
        if not resolved:
            raise ValueError("no resolved landmark operands")
        uncertainties = [item.uncertainty_px for item in resolved if item.uncertainty_px is not None]
        uncertainty = float(np.sqrt(np.sum(np.square(uncertainties)))) if uncertainties else None

        by_frame: dict[int, list[LandmarkResult]] = {}
        for item in resolved:
            by_frame.setdefault(item.frame_idx, []).append(item)
        measurement_kind = plan.measurement_kind

        if measurement_kind in {"distance", "extent"}:
            target_frame = sorted(by_frame, key=lambda index: (-len(by_frame[index]), index))[0]
            frame_results = by_frame[target_frame]
            point_sets = [self._points(item, track_index) for item in frame_results]
            point_sets = [points for points in point_sets if len(points)]
            if len(point_sets) >= 2:
                if plan.relation == "vertical_distance" or plan.axis == "vertical":
                    value = min(
                        float(np.min(np.abs(a[:, None, 1] - b[None, :, 1])))
                        for index, a in enumerate(point_sets)
                        for b in point_sets[index + 1 :]
                    )
                    method = "vertical_geometry_distance"
                else:
                    value = min(
                        _pairwise_distance(a, b, plan.reduction)
                        for index, a in enumerate(point_sets)
                        for b in point_sets[index + 1 :]
                    )
                    method = f"{plan.reduction}_geometry_distance"
            elif len(point_sets) == 1 and measurement_kind == "extent":
                value = _extent(point_sets[0], plan.quantity_subtype, plan.axis)
                method = f"{plan.quantity_subtype}_extent"
            else:
                raise ValueError("measurement does not have enough resolved geometries")
            return value, method, [target_frame], uncertainty, {}

        # Motion quantities use centroid trajectories from the resolved parent track.
        ordered = sorted(resolved, key=lambda item: item.frame_idx)
        points = []
        times = []
        frames = []
        for item in ordered:
            geometry = self._points(item, track_index)
            if len(geometry):
                points.append(geometry.mean(axis=0))
                times.append(item.time_s)
                frames.append(item.frame_idx)
        if len(points) < 2:
            raise ValueError("motion measurement needs at least two resolved frames")
        points_array = np.asarray(points)
        times_array = np.asarray(times)
        deltas = np.linalg.norm(np.diff(points_array, axis=0), axis=1)
        dt = np.diff(times_array)
        valid = dt > 0
        if not valid.any():
            raise ValueError("motion frames have no positive time interval")
        if measurement_kind == "displacement":
            value = float(np.linalg.norm(points_array[-1] - points_array[0]))
            method = "centroid_displacement"
            pixel_unit = "px"
        elif measurement_kind == "path_length":
            value = float(deltas.sum())
            method = "centroid_path_sum"
            pixel_unit = "px"
        elif measurement_kind in {"speed", "velocity"}:
            value = float(np.mean(deltas[valid] / dt[valid]))
            method = "centroid_first_derivative"
            pixel_unit = "px/s"
            return value, method, frames, uncertainty, {"pixel_unit": pixel_unit}
        elif measurement_kind == "acceleration":
            velocities = np.diff(points_array, axis=0) / dt[:, None]
            if len(velocities) < 2:
                raise ValueError("acceleration needs at least three resolved frames")
            velocity_dt = (dt[:-1] + dt[1:]) / 2
            acceleration = np.linalg.norm(np.diff(velocities, axis=0), axis=1) / velocity_dt
            value = float(np.mean(acceleration))
            method = "centroid_second_derivative"
            pixel_unit = "px/s^2"
            return value, method, frames, uncertainty, {"pixel_unit": pixel_unit}
        else:
            raise ValueError(f"unsupported measurement kind: {measurement_kind}")
        return value, method, frames, uncertainty, {"pixel_unit": pixel_unit}

    @staticmethod
    def _reference_track(
        plan: MeasurementPlan, tracks: dict[str, list[TrackMasks]]
    ) -> TrackMasks | None:
        prior = plan.calibration.get("prior") or {}
        description_tokens = set(
            re.findall(r"[a-z0-9]+", str(prior.get("description") or "").lower())
        )
        property_tokens = {
            "size", "length", "width", "height", "diameter", "speed", "velocity",
            "acceleration", "at", "of", "the", "model",
        }
        description_tokens -= property_tokens
        best: tuple[int, TrackMasks] | None = None
        for key, items in tracks.items():
            score = len(description_tokens & set(key.lower().split("_")))
            if items and score and (best is None or score > best[0]):
                best = (score, items[0])
        return best[1] if best else None

    def _calibration_scale(
        self,
        plan: MeasurementPlan,
        tracks: dict[str, list[TrackMasks]],
        fps: float,
    ) -> tuple[float | None, list[str], dict[str, Any]]:
        warnings: list[str] = []
        if plan.calibration.get("dimensionality") != "planar":
            return None, ["2D physical calibration disabled for non-planar video"], {}
        prior = plan.calibration.get("prior") or {}
        value = prior.get("value")
        prior_unit = _length_unit(prior.get("unit"))
        output_unit = _length_unit(plan.output_unit)
        if not isinstance(value, (int, float)) or prior_unit is None or output_unit is None:
            return None, ["physical prior or compatible length unit is unavailable"], {}
        track = self._reference_track(plan, tracks)
        if track is None:
            return None, ["calibration reference track is unavailable"], {}
        centers = [_centroid(mask) for mask in track.masks]
        method = plan.calibration.get("calibration_method")
        pixel_prior: float | None = None
        if method == "reference_extent_scale":
            frame_idx = int(np.clip(
                round(float(prior.get("timestamp_s") or 0) * fps), 0, len(track.masks) - 1
            ))
            points = _largest_contour(track.masks[frame_idx])
            subtype = str(prior.get("property_subtype") or "length")
            pixel_prior = _extent(points, subtype, None) if len(points) else None
        else:
            valid_centers = [(index, center) for index, center in enumerate(centers) if center is not None]
            if len(valid_centers) >= 2:
                indices = np.asarray([item[0] for item in valid_centers], dtype=float)
                positions = np.asarray([item[1] for item in valid_centers])
                dt = np.diff(indices) / fps
                velocities = np.diff(positions, axis=0) / dt[:, None]
                if method == "reference_first_derivative_scale":
                    pixel_prior = float(np.median(np.linalg.norm(velocities, axis=1)))
                elif method == "reference_second_derivative_scale" and len(velocities) >= 2:
                    acceleration = np.diff(velocities, axis=0) / ((dt[:-1] + dt[1:]) / 2)[:, None]
                    pixel_prior = float(np.median(np.linalg.norm(acceleration, axis=1)))
        if pixel_prior is None or pixel_prior <= 1e-9:
            return None, ["reference pixel prior could not be measured"], {}
        scale_output_per_px = (
            float(value) * LENGTH_TO_M[prior_unit] / LENGTH_TO_M[output_unit] / pixel_prior
        )
        evidence = {
            "reference_track_id": track.track_id,
            "prior_value": value,
            "prior_unit": prior.get("unit"),
            "pixel_prior": pixel_prior,
            "scale_output_unit_per_px": scale_output_per_px,
        }
        return scale_output_per_px, warnings, evidence

    def solve_plan(
        self,
        plan: MeasurementPlan,
        landmark_results: list[LandmarkResult],
        tracks: dict[str, list[TrackMasks]],
        fps: float,
    ) -> SolverResult:
        plan_results = self._results_for_plan(plan, landmark_results)
        unresolved = [item for item in plan_results if item.status != "resolved"]
        expected_ids = {operand.operand_id or operand.role for operand in plan.operands}
        resolved_base_ids = {item.operand_id.split(":")[0] for item in plan_results if item.status == "resolved"}
        if not expected_ids <= resolved_base_ids:
            return SolverResult(
                plan_id=plan.plan_id,
                qa_id=plan.qa_id,
                status="unresolved",
                measurement_kind=plan.measurement_kind,
                value_px=None,
                pixel_unit=None,
                value=None,
                output_unit=plan.output_unit,
                method="unresolved_landmark",
                warnings=[
                    "not all measurement operands have a resolved LandmarkResult",
                    *[warning for item in unresolved for warning in item.warnings],
                ],
            )
        track_index = self._track_index(tracks)
        try:
            value_px, method, frames, uncertainty, pixel_evidence = self._pixel_measurement(
                plan, plan_results, track_index
            )
        except ValueError as exc:
            return SolverResult(
                plan_id=plan.plan_id,
                qa_id=plan.qa_id,
                status="unresolved",
                measurement_kind=plan.measurement_kind,
                value_px=None,
                pixel_unit=None,
                value=None,
                output_unit=plan.output_unit,
                method="unsupported_geometry",
                warnings=[str(exc)],
            )
        derivative_unit = pixel_evidence.get("pixel_unit")
        if derivative_unit is None:
            derivative_unit = "px"
        scale, calibration_warnings, calibration_evidence = self._calibration_scale(
            plan, tracks, fps
        )
        physical_value = value_px * scale if scale is not None else None
        status = "solved_physical" if physical_value is not None else "solved_pixel"
        return SolverResult(
            plan_id=plan.plan_id,
            qa_id=plan.qa_id,
            status=status,
            measurement_kind=plan.measurement_kind,
            value_px=value_px,
            pixel_unit=derivative_unit,
            value=physical_value,
            output_unit=plan.output_unit,
            method=method,
            frame_indices=frames,
            uncertainty_px=uncertainty,
            warnings=calibration_warnings,
            evidence={**pixel_evidence, **calibration_evidence},
        )

    def solve_plans(
        self,
        plans: list[MeasurementPlan],
        landmark_results: list[LandmarkResult],
        tracks: dict[str, list[TrackMasks]],
        fps: float,
    ) -> list[SolverResult]:
        return [
            self.solve_plan(plan, landmark_results, tracks, fps) for plan in plans
        ]
