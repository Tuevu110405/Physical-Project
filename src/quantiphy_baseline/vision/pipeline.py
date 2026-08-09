from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .entity_specs import TrackingRequest, build_tracking_requests
from .grounder import Detection, GroundingDinoGrounder
from .sam2_tracker import Sam2Tracker
from .video_io import candidate_anchor_indices, load_video_pil, resolve_video_path


def _box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def choose_anchor(
    frames: list[Image.Image],
    fps: float,
    req: TrackingRequest,
    grounder: GroundingDinoGrounder,
    uniform_samples: int = 5,
) -> tuple[int, list[Detection], list[dict]]:
    candidates = candidate_anchor_indices(
        len(frames), fps, req.preferred_times_s, uniform_samples=uniform_samples
    )
    attempts: list[dict] = []
    best_idx = -1
    best_dets: list[Detection] = []
    best_quality = -1.0
    image_area = frames[0].width * frames[0].height

    for idx in candidates:
        dets = grounder.detect(
            frames[idx], req.prompts, expected_instances=req.expected_instances, entity_text=req.display_name
        )
        # For pair tracking we require both instances. Otherwise a single detection is enough.
        if len(dets) < req.expected_instances:
            attempts.append({"frame_idx": idx, "detections": len(dets), "quality": 0.0})
            continue
        scores = [d.score for d in dets[: req.expected_instances]]
        area_frac = sum(_box_area(d.box_xyxy) for d in dets[: req.expected_instances]) / max(image_area, 1)
        # Detection score dominates; tiny area bonus prefers frames with more measurement resolution.
        quality = float(np.mean(scores) + 0.03 * np.sqrt(max(area_frac, 0.0)))
        attempts.append({
            "frame_idx": idx,
            "detections": len(dets),
            "scores": scores,
            "quality": quality,
        })
        if quality > best_quality:
            best_quality = quality
            best_idx = idx
            best_dets = dets[: req.expected_instances]

    if best_idx < 0:
        raise RuntimeError(f"Grounding failed for '{req.display_name}' with prompts={req.prompts}")
    return best_idx, best_dets, attempts


def save_bitpacked_masks(path: Path, masks: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = np.packbits(masks.astype(np.uint8), axis=2)
    np.savez_compressed(path, packed=packed, shape=np.array(masks.shape, dtype=np.int64))


def load_bitpacked_masks(path: str | Path) -> np.ndarray:
    d = np.load(path)
    shape = tuple(int(x) for x in d["shape"])
    unpacked = np.unpackbits(d["packed"], axis=2)
    return unpacked[:, :, : shape[2]].reshape(shape).astype(np.uint8)


class SegmentTrackPipeline:
    def __init__(
        self,
        grounder: GroundingDinoGrounder,
        tracker: Sam2Tracker,
        output_dir: str | Path,
        anchor_samples: int = 5,
    ) -> None:
        self.grounder = grounder
        self.tracker = tracker
        self.output_dir = Path(output_dir)
        self.anchor_samples = anchor_samples

    def process_group(self, group: dict[str, Any], video_dir: str | Path | None = None) -> dict[str, Any]:
        video_id = group["video_id"]
        video_path = resolve_video_path(video_id, video_dir, group.get("video_path"))
        frames, decoded_fps = load_video_pil(video_path)
        fps = float(group.get("fps") or decoded_fps)
        if abs(fps - decoded_fps) > 0.5:
            # Prefer explicit dataset FPS for time alignment, but retain decoded FPS for auditing.
            fps_warning = f"dataset fps={fps}, decoded fps={decoded_fps:.3f}"
        else:
            fps_warning = None

        requests = build_tracking_requests(group)
        result: dict[str, Any] = {
            "video_id": video_id,
            "video_path": str(video_path),
            "fps": fps,
            "decoded_fps": decoded_fps,
            "num_frames": len(frames),
            "frame_size": [frames[0].width, frames[0].height],
            "warnings": [fps_warning] if fps_warning else [],
            "objects": {},
        }

        mask_root = self.output_dir / "masks" / video_id
        for req_i, req in enumerate(requests):
            obj_out: dict[str, Any] = {
                "display_name": req.display_name,
                "roles": req.roles,
                "visual_kind": req.visual_kind,
                "expected_instances": req.expected_instances,
                "prompts": req.prompts,
                "preferred_times_s": req.preferred_times_s,
                "source_qa_ids": req.source_qa_ids,
                "notes": req.notes,
                "instances": [],
            }
            try:
                anchor_idx, detections, attempts = choose_anchor(
                    frames, fps, req, self.grounder, uniform_samples=self.anchor_samples
                )
                obj_out["anchor_search"] = attempts
                obj_out["anchor_frame_idx"] = anchor_idx
                obj_out["anchor_time_s"] = anchor_idx / fps

                # For pair entities, sort left-to-right to keep stable instance naming.
                detections = sorted(detections, key=lambda d: (d.box_xyxy[0] + d.box_xyxy[2]) / 2)
                for inst_i, det in enumerate(detections):
                    track = self.tracker.track(
                        frames,
                        anchor_frame_idx=anchor_idx,
                        box_xyxy=det.box_xyxy,
                        fps=fps,
                        object_id=1,
                    )
                    track_id = f"{req.entity_key}__{inst_i}"
                    mask_path = mask_root / f"{track_id}.npz"
                    save_bitpacked_masks(mask_path, track.masks)
                    obj_out["instances"].append({
                        "track_id": track_id,
                        "anchor_detection": {
                            "box_xyxy": det.box_xyxy,
                            "score": det.score,
                            "label": det.label,
                        },
                        "mask_path": str(mask_path),
                        "summary": track.summary,
                        "frames": track.frames,
                    })
            except Exception as e:
                obj_out["error"] = f"{type(e).__name__}: {e}"
            result["objects"][req.entity_key] = obj_out

        out_json = self.output_dir / "tracks" / f"{video_id}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
