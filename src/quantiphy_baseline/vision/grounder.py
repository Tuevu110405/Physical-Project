from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from PIL import Image


@dataclass
class Detection:
    box_xyxy: list[float]
    score: float
    label: str
    prompt: str | None = None


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = aa + ba - inter
    return inter / den if den > 0 else 0.0


def nms_python(dets: list[Detection], iou_threshold: float = 0.65) -> list[Detection]:
    out: list[Detection] = []
    for d in sorted(dets, key=lambda x: x.score, reverse=True):
        if all(box_iou(d.box_xyxy, k.box_xyxy) < iou_threshold for k in out):
            out.append(d)
    return out


class GroundingDinoGrounder:
    """Open-vocabulary box detector used only to initialize SAM2 tracks."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str | None = None,
        threshold: float = 0.28,
        text_threshold: float = 0.22,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.text_threshold = text_threshold
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def detect(
        self,
        image: Image.Image,
        prompts: Sequence[str],
        expected_instances: int = 1,
        entity_text: str | None = None,
    ) -> list[Detection]:
        prompts = [p.strip() for p in prompts if p and p.strip()]
        if not prompts:
            return []

        # Current Transformers API accepts a nested list of text labels for one image.
        inputs = self.processor(images=image, text=[prompts], return_tensors="pt")
        inputs = {
            k: (
                v.to(self.device, dtype=self.dtype)
                if hasattr(v, "is_floating_point") and v.is_floating_point()
                else v.to(self.device)
                if hasattr(v, "to")
                else v
            )
            for k, v in inputs.items()
        }
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        dets: list[Detection] = []
        labels = results.get("text_labels", results.get("labels", []))
        for box, score, label in zip(results["boxes"], results["scores"], labels):
            label_s = label if isinstance(label, str) else str(label)
            dets.append(
                Detection(
                    box_xyxy=[float(x) for x in box.tolist()],
                    score=float(score.item()),
                    label=label_s,
                )
            )

        dets = nms_python(dets)
        dets = self._spatial_rerank(dets, entity_text or "", image.size)
        return dets[: max(expected_instances, 1)]

    @staticmethod
    def _spatial_rerank(dets: list[Detection], entity_text: str, image_size: tuple[int, int]) -> list[Detection]:
        if len(dets) <= 1:
            return dets
        s = entity_text.lower()
        w, h = image_size

        def score(d: Detection) -> float:
            x1, y1, x2, y2 = d.box_xyxy
            cx, cy = (x1 + x2) / 2 / max(w, 1), (y1 + y2) / 2 / max(h, 1)
            bonus = 0.0
            if "left" in s:
                bonus += 0.18 * (1 - cx)
            if "right" in s:
                bonus += 0.18 * cx
            if "upper" in s or "top" in s:
                bonus += 0.12 * (1 - cy)
            if "lower" in s or "bottom" in s:
                bonus += 0.12 * cy
            return d.score + bonus

        return sorted(dets, key=score, reverse=True)
