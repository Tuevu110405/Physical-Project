from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Works from a source checkout without requiring pip install -e .
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantiphy_baseline.vision import GroundingDinoGrounder, Sam2Tracker, SegmentTrackPipeline


def parse_args():
    p = argparse.ArgumentParser(description="Ground target/reference entities and track them with SAM2.")
    p.add_argument("--groups", default="data/processed/grouped_by_video.jsonl")
    p.add_argument("--video-dir", required=True)
    p.add_argument("--output-dir", default="outputs/vision")
    p.add_argument("--grounder-model", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--sam2-model", default="facebook/sam2.1-hiera-small")
    p.add_argument("--device", default=None)
    p.add_argument("--box-threshold", type=float, default=0.28)
    p.add_argument("--text-threshold", type=float, default=0.22)
    p.add_argument("--anchor-samples", type=int, default=5)
    p.add_argument("--only-video", action="append", default=[])
    p.add_argument("--max-videos", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    groups_path = Path(args.groups)
    grounder = GroundingDinoGrounder(
        model_id=args.grounder_model,
        device=args.device,
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    tracker = Sam2Tracker(model_id=args.sam2_model, device=args.device)
    pipe = SegmentTrackPipeline(
        grounder=grounder,
        tracker=tracker,
        output_dir=args.output_dir,
        anchor_samples=args.anchor_samples,
    )

    only = set(args.only_video)
    n = 0
    with groups_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            group = json.loads(line)
            if only and group["video_id"] not in only:
                continue
            print(f"[vision] {group['video_id']}")
            result = pipe.process_group(group, video_dir=args.video_dir)
            failed = sum("error" in o for o in result["objects"].values())
            print(f"  objects={len(result['objects'])}, failed={failed}")
            n += 1
            if args.max_videos is not None and n >= args.max_videos:
                break


if __name__ == "__main__":
    main()
