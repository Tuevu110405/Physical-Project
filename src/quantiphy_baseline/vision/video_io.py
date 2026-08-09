from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def resolve_video_path(video_id: str, video_dir: str | Path | None, explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    if video_dir is None:
        raise FileNotFoundError(f"No video path for {video_id}; pass --video-dir or populate video_path.")
    root = Path(video_dir)
    for ext in VIDEO_EXTS:
        p = root / f"{video_id}{ext}"
        if p.exists():
            return p
    # Support nested source/video_id.mp4 layouts.
    for ext in VIDEO_EXTS:
        matches = list(root.rglob(f"{video_id}{ext}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find video {video_id} below {root}")


def load_video_pil(path: str | Path) -> tuple[list[Image.Image], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames: list[Image.Image] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    if not fps or fps <= 0:
        fps = 30.0
    return frames, fps


def candidate_anchor_indices(
    num_frames: int,
    fps: float,
    preferred_times_s: list[float],
    uniform_samples: int = 5,
) -> list[int]:
    ids = set()
    if num_frames <= 0:
        return []
    for t in preferred_times_s:
        ids.add(max(0, min(num_frames - 1, int(round(t * fps)))))
    if uniform_samples <= 1:
        ids.add(num_frames // 2)
    else:
        for i in range(uniform_samples):
            ids.add(int(round(i * (num_frames - 1) / (uniform_samples - 1))))
    return sorted(ids)
