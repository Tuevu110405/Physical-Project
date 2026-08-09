from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import Sam2VideoModel, Sam2VideoProcessor

from .mask_features import mask_to_features, summarize_track


@dataclass
class Sam2TrackResult:
    frames: list[dict]
    masks: np.ndarray  # [T,H,W] uint8
    summary: dict


class Sam2Tracker:
    """Track one object from a box prompt, both forward and backward from an anchor frame.

    One-object-per-session is intentionally conservative. It avoids identity coupling between
    independently grounded objects and permits a different anchor frame for every entity.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-small",
        device: str | None = None,
        video_storage_device: str = "cpu",
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.video_storage_device = video_storage_device
        if self.device.startswith("cuda"):
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.dtype = torch.float32
        self.model = Sam2VideoModel.from_pretrained(model_id).to(self.device, dtype=self.dtype)
        self.processor = Sam2VideoProcessor.from_pretrained(model_id)
        self.model.eval()

    def _post_mask(self, output, session) -> np.ndarray:
        masks = self.processor.post_process_masks(
            [output.pred_masks],
            original_sizes=[[session.video_height, session.video_width]],
            binarize=False,
        )[0]
        # Common shape is [num_obj, 1, H, W]. Robustly squeeze singleton dims.
        arr = masks[0].detach().float().cpu().numpy()
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            arr = np.squeeze(arr)
        return arr

    @torch.inference_mode()
    def track(
        self,
        video_frames: list[Image.Image],
        anchor_frame_idx: int,
        box_xyxy: list[float],
        fps: float,
        object_id: int = 1,
        mask_threshold: float = 0.0,
    ) -> Sam2TrackResult:
        session = self.processor.init_video_session(
            video=video_frames,
            inference_device=self.device,
            inference_state_device=self.device,
            processing_device=self.device,
            video_storage_device=self.video_storage_device,
            dtype=self.dtype,
        )

        self.processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=int(anchor_frame_idx),
            obj_ids=int(object_id),
            input_boxes=[[[float(x) for x in box_xyxy]]],
        )

        outputs = self.model(inference_session=session, frame_idx=int(anchor_frame_idx))
        logits_by_frame: dict[int, np.ndarray] = {anchor_frame_idx: self._post_mask(outputs, session)}

        for out in self.model.propagate_in_video_iterator(
            session, start_frame_idx=int(anchor_frame_idx), reverse=False
        ):
            logits_by_frame[int(out.frame_idx)] = self._post_mask(out, session)

        for out in self.model.propagate_in_video_iterator(
            session, start_frame_idx=int(anchor_frame_idx), reverse=True
        ):
            logits_by_frame[int(out.frame_idx)] = self._post_mask(out, session)

        n = len(video_frames)
        h, w = video_frames[0].height, video_frames[0].width
        masks = np.zeros((n, h, w), dtype=np.uint8)
        frame_rows: list[dict] = []
        for i in range(n):
            logits = logits_by_frame.get(i)
            if logits is None:
                feat = {"valid": False, "area_px": 0}
            else:
                binary = logits > mask_threshold
                masks[i] = binary.astype(np.uint8)
                feat = mask_to_features(binary, logits)
            feat.update({"frame_idx": i, "time_s": i / fps})
            frame_rows.append(feat)

        return Sam2TrackResult(
            frames=frame_rows,
            masks=masks,
            summary=summarize_track(frame_rows),
        )
