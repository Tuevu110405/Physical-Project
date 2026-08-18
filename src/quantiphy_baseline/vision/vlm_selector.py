from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageDraw

from .landmark_types import LandmarkCandidate


class CandidateSelector(Protocol):
    def select(
        self,
        frame: Image.Image,
        parent_mask: np.ndarray,
        candidates: list[LandmarkCandidate],
        prompt: str,
        reference_mask: np.ndarray | None = None,
    ) -> tuple[str, float, dict[str, Any]]: ...


def render_candidate_overlay(
    frame: Image.Image,
    parent_mask: np.ndarray,
    candidates: list[LandmarkCandidate],
    reference_mask: np.ndarray | None = None,
) -> Image.Image:
    base = frame.convert("RGBA")
    overlay = np.zeros((base.height, base.width, 4), dtype=np.uint8)
    overlay[np.asarray(parent_mask).astype(bool)] = [40, 220, 80, 70]
    if reference_mask is not None:
        overlay[np.asarray(reference_mask).astype(bool)] = [40, 180, 255, 75]
    base.alpha_composite(Image.fromarray(overlay, mode="RGBA"))
    draw = ImageDraw.Draw(base)
    colors = ["red", "yellow", "magenta", "cyan", "orange", "white"]
    for index, candidate in enumerate(candidates):
        points = [tuple(point) for point in candidate.coordinates_xy]
        color = colors[index % len(colors)]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=3)
        elif len(points) > 1:
            draw.line(points + [points[0]], fill=color, width=4)
        if points:
            x, y = points[len(points) // 2]
            draw.text((x + 6, y + 4), candidate.candidate_id, fill=color, stroke_width=2, stroke_fill="black")
    return base.convert("RGB")


class OpenAICompatibleVLMSelector:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str | None = None,
        timeout_s: float = 90.0,
        debug_dir: str | Path | None = None,
        cache_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        self.timeout_s = timeout_s
        self.debug_dir = Path(debug_dir) if debug_dir is not None else None
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self._cache: dict[str, dict[str, Any]] = {}
        if self.cache_path is not None and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                    self._cache[str(item["cache_key"])] = item["result"]
                except (KeyError, TypeError, json.JSONDecodeError):
                    continue
        self._call_index = 0

    def select(
        self,
        frame: Image.Image,
        parent_mask: np.ndarray,
        candidates: list[LandmarkCandidate],
        prompt: str,
        reference_mask: np.ndarray | None = None,
    ) -> tuple[str, float, dict[str, Any]]:
        image = render_candidate_overlay(frame, parent_mask, candidates, reference_mask)
        self._call_index += 1
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            image.save(self.debug_dir / f"candidate_overlay_{self._call_index:05d}.png")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92)
        image_bytes = buffer.getvalue()
        valid_ids = [item.candidate_id for item in candidates]
        cache_key = hashlib.sha256(
            image_bytes
            + prompt.encode("utf-8")
            + self.model.encode("utf-8")
            + json.dumps(valid_ids).encode("utf-8")
        ).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return (
                str(cached["selected_candidate"]),
                float(cached["confidence"]),
                {**cached.get("evidence", {}), "cache_hit": True},
            )
        data_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")
        instruction = (
            f"{prompt}\nChoose exactly one visible candidate ID from {valid_ids}. "
            "Return JSON only: {\"selected_candidate\":\"...\",\"confidence\":0.0}."
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"VLM selector HTTP {exc.code}: {detail}") from exc
        content = raw["choices"][0]["message"]["content"]
        result = content if isinstance(content, dict) else json.loads(content)
        selected = str(result.get("selected_candidate"))
        if selected not in valid_ids:
            raise RuntimeError(f"VLM selected invalid candidate {selected!r}; valid={valid_ids}")
        confidence = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
        evidence = {"vlm_model": self.model, "valid_candidates": valid_ids, "cache_hit": False}
        cache_result = {
            "selected_candidate": selected,
            "confidence": confidence,
            "evidence": evidence,
        }
        self._cache[cache_key] = cache_result
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"cache_key": cache_key, "result": cache_result}) + "\n")
        return selected, confidence, evidence
