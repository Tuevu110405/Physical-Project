from __future__ import annotations

import cv2
import numpy as np


def mask_to_features(mask: np.ndarray, mask_logit: np.ndarray | None = None) -> dict:
    m = np.asarray(mask).astype(bool)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return {"valid": False, "area_px": 0}

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    area = int(len(xs))
    cx, cy = float(xs.mean()), float(ys.mean())

    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        (rcx, rcy), (rw, rh), angle = rect
        perimeter = float(cv2.arcLength(contour, True))
    else:
        rcx, rcy, rw, rh, angle, perimeter = cx, cy, x2 - x1, y2 - y1, 0.0, 0.0

    out = {
        "valid": True,
        "bbox_xyxy": [x1, y1, x2, y2],
        "centroid_xy": [cx, cy],
        "area_px": area,
        "bbox_width_px": float(x2 - x1),
        "bbox_height_px": float(y2 - y1),
        "oriented_rect": {
            "center_xy": [float(rcx), float(rcy)],
            "width_px": float(rw),
            "height_px": float(rh),
            "angle_deg": float(angle),
        },
        "extreme_points": {
            "left": [int(xs[np.argmin(xs)]), int(ys[np.argmin(xs)])],
            "right": [int(xs[np.argmax(xs)]), int(ys[np.argmax(xs)])],
            "top": [int(xs[np.argmin(ys)]), int(ys[np.argmin(ys)])],
            "bottom": [int(xs[np.argmax(ys)]), int(ys[np.argmax(ys)])],
        },
        "perimeter_px": perimeter,
    }
    if mask_logit is not None:
        logits = np.asarray(mask_logit)
        if logits.shape == m.shape and m.any():
            out["mean_positive_logit"] = float(logits[m].mean())
    return out


def summarize_track(frames: list[dict]) -> dict:
    valid = [f for f in frames if f.get("valid")]
    if not frames:
        return {"coverage": 0.0, "num_valid": 0, "num_frames": 0, "warnings": ["empty track"]}
    warnings: list[str] = []
    coverage = len(valid) / len(frames)
    if coverage < 0.9:
        warnings.append(f"low mask coverage: {coverage:.3f}")

    areas = np.array([f["area_px"] for f in valid], dtype=float) if valid else np.array([])
    area_cv = float(areas.std() / max(areas.mean(), 1e-9)) if len(areas) > 1 else 0.0
    if area_cv > 1.0:
        warnings.append(f"large mask-area variation: CV={area_cv:.2f}")

    jumps = []
    for a, b in zip(valid, valid[1:]):
        if b["frame_idx"] == a["frame_idx"] + 1:
            ax, ay = a["centroid_xy"]
            bx, by = b["centroid_xy"]
            jumps.append(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5)
    p95_jump = float(np.percentile(jumps, 95)) if jumps else 0.0

    return {
        "coverage": coverage,
        "num_valid": len(valid),
        "num_frames": len(frames),
        "area_cv": area_cv,
        "centroid_jump_p95_px_per_frame": p95_jump,
        "warnings": warnings,
    }
