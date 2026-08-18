from __future__ import annotations

import cv2
import numpy as np

from .landmark_types import LandmarkCandidate


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return np.empty((0, 2), dtype=np.float32)
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)


def _sample_points(points: np.ndarray, limit: int = 48) -> list[list[float]]:
    if len(points) > limit:
        indices = np.linspace(0, len(points) - 1, limit, dtype=int)
        points = points[indices]
    return [[float(x), float(y)] for x, y in points]


def _candidate(candidate_id: str, geometry_type: str, points: np.ndarray, **metadata):
    return LandmarkCandidate(
        candidate_id=candidate_id,
        geometry_type=geometry_type,
        coordinates_xy=_sample_points(np.asarray(points).reshape(-1, 2)),
        metadata=metadata,
    )


def _principal_axis(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    points = np.column_stack([xs, ys]).astype(np.float32)
    if len(points) < 2:
        raise ValueError("mask has fewer than two pixels")
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))].astype(np.float32)
    if axis[0] < 0 or (abs(float(axis[0])) < 1e-6 and axis[1] < 0):
        axis *= -1
    return points, center, axis


def _axis_candidates(
    mask: np.ndarray, *, caps: bool, geometry_type: str
) -> list[LandmarkCandidate]:
    points, center, axis = _principal_axis(mask)
    projections = (points - center) @ axis
    contour = _largest_contour(mask)
    if len(contour) == 0:
        return []
    contour_projection = (contour - center) @ axis
    if caps:
        low, high = np.quantile(projections, [0.08, 0.92])
        negative = contour[contour_projection <= low]
        positive = contour[contour_projection >= high]
    else:
        negative = contour[[int(np.argmin(contour_projection))]]
        positive = contour[[int(np.argmax(contour_projection))]]
    return [
        _candidate(
            "end_negative",
            geometry_type,
            negative,
            axis_xy=axis.tolist(),
            axis_sign=-1,
        ),
        _candidate(
            "end_positive",
            geometry_type,
            positive,
            axis_xy=axis.tolist(),
            axis_sign=1,
        ),
    ]


def _directional_boundary(
    contour: np.ndarray, operator: str, geometry_type: str
) -> list[LandmarkCandidate]:
    if len(contour) == 0:
        return []
    axis_index = 0 if operator in {"leftmost", "rightmost"} else 1
    use_min = operator in {"leftmost", "topmost"}
    values = contour[:, axis_index]
    extreme = float(values.min() if use_min else values.max())
    span = max(float(values.max() - values.min()), 1.0)
    tolerance = max(2.0, span * 0.05)
    keep = np.abs(values - extreme) <= tolerance
    return [_candidate(operator, geometry_type, contour[keep])]


def _skeleton_endpoints(mask: np.ndarray) -> list[LandmarkCandidate]:
    image = (np.asarray(mask).astype(bool) * 255).astype(np.uint8)
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = image.copy()
    while cv2.countNonZero(work) > 0:
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = cv2.erode(work, element)
    binary = (skeleton > 0).astype(np.uint8)
    neighbors = cv2.filter2D(binary, -1, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero((binary == 1) & (neighbors == 2))
    endpoints = np.column_stack([xs, ys]).astype(np.float32)
    if len(endpoints) < 2:
        return _axis_candidates(mask, caps=False, geometry_type="point")
    if len(endpoints) > 2:
        distances = np.sum((endpoints[:, None] - endpoints[None, :]) ** 2, axis=2)
        first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
        endpoints = endpoints[[first, second]]
    return [
        _candidate(f"skeleton_end_{index}", "point", point.reshape(1, 2))
        for index, point in enumerate(endpoints)
    ]


def _corners(mask: np.ndarray) -> list[LandmarkCandidate]:
    contour = _largest_contour(mask)
    if len(contour) == 0:
        return []
    epsilon = 0.02 * cv2.arcLength(contour.reshape(-1, 1, 2), True)
    polygon = cv2.approxPolyDP(contour.reshape(-1, 1, 2), epsilon, True).reshape(-1, 2)
    return [
        _candidate(f"corner_{index}", "point", point.reshape(1, 2))
        for index, point in enumerate(polygon[:12])
    ]


def _contact_region(
    mask: np.ndarray, reference_mask: np.ndarray | None, geometry_type: str
) -> list[LandmarkCandidate]:
    if reference_mask is None or not np.asarray(reference_mask).any():
        return []
    contour = _largest_contour(mask)
    if len(contour) == 0:
        return []
    distance = cv2.distanceTransform(
        (~np.asarray(reference_mask).astype(bool)).astype(np.uint8), cv2.DIST_L2, 3
    )
    xy = np.rint(contour).astype(int)
    values = distance[xy[:, 1], xy[:, 0]]
    threshold = float(values.min() + max(2.0, np.ptp(values) * 0.05))
    return [
        _candidate(
            "contact_region",
            geometry_type,
            contour[values <= threshold],
            min_reference_distance_px=float(values.min()),
        )
    ]


def generate_landmark_candidates(
    mask: np.ndarray,
    generator: str,
    geometry_type: str,
    *,
    selector_operator: str | None = None,
    reference_mask: np.ndarray | None = None,
) -> list[LandmarkCandidate]:
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2 or not mask.any():
        return []
    contour = _largest_contour(mask)
    if generator == "whole_mask":
        return [
            LandmarkCandidate(
                candidate_id="whole_object",
                geometry_type="mask",
                coordinates_xy=[],
                metadata={"area_px": int(mask.sum())},
            )
        ]
    if generator == "mask_contour":
        if selector_operator in {"leftmost", "rightmost", "topmost", "bottommost"}:
            return _directional_boundary(contour, selector_operator, geometry_type)
        return [_candidate("mask_contour", geometry_type, contour)]
    if generator == "extreme_points":
        if len(contour) == 0:
            return []
        definitions = {
            "leftmost": int(np.argmin(contour[:, 0])),
            "rightmost": int(np.argmax(contour[:, 0])),
            "topmost": int(np.argmin(contour[:, 1])),
            "bottommost": int(np.argmax(contour[:, 1])),
        }
        candidates = [
            _candidate(name, "point", contour[[index]])
            for name, index in definitions.items()
        ]
        if selector_operator in definitions:
            return [item for item in candidates if item.candidate_id == selector_operator]
        return candidates
    if generator == "major_axis_endpoints":
        return _axis_candidates(mask, caps=False, geometry_type=geometry_type)
    if generator == "major_axis_end_caps":
        return _axis_candidates(mask, caps=True, geometry_type=geometry_type)
    if generator == "skeleton_endpoints":
        return _skeleton_endpoints(mask)
    if generator == "polygon_corners":
        return _corners(mask)
    if generator == "contact_regions":
        return _contact_region(mask, reference_mask, geometry_type)
    # semantic_part_grounding and text_detection require a visual model.
    return []
