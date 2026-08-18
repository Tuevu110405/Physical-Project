from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Works from a source checkout without requiring pip install -e .
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantiphy_baseline.vision import GroundingDinoGrounder, Sam2Tracker, SegmentTrackPipeline
from quantiphy_baseline.planning import LLMSemanticPlanner
from quantiphy_baseline.vision import LandmarkResolver, OpenAICompatibleVLMSelector


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
    p.add_argument("--semantic-model", default=None,
                   help="Enable LLM landmark planning with an OpenAI-compatible model.")
    p.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--llm-api-key", default=None)
    p.add_argument("--semantic-cache", default="outputs/planning/semantic_plan_cache.jsonl")
    p.add_argument("--vlm-model", default=None,
                   help="Optional OpenAI-compatible VLM used only for candidate selection.")
    p.add_argument("--vlm-base-url", default=None)
    p.add_argument("--vlm-debug-dir", default="outputs/vision/debug_overlays")
    p.add_argument("--vlm-cache", default="outputs/planning/vlm_selection_cache.jsonl")
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
    semantic_planner = None
    if args.semantic_model:
        semantic_planner = LLMSemanticPlanner(
            model=args.semantic_model,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            cache_path=args.semantic_cache,
        )
    vlm_selector = None
    if args.vlm_model:
        vlm_selector = OpenAICompatibleVLMSelector(
            model=args.vlm_model,
            base_url=args.vlm_base_url or args.llm_base_url,
            api_key=args.llm_api_key,
            debug_dir=args.vlm_debug_dir,
            cache_path=args.vlm_cache,
        )
    pipe = SegmentTrackPipeline(
        grounder=grounder,
        tracker=tracker,
        output_dir=args.output_dir,
        anchor_samples=args.anchor_samples,
        semantic_planner=semantic_planner,
        landmark_resolver=LandmarkResolver(vlm_selector=vlm_selector),
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
            resolved = sum(x["status"] == "resolved" for x in result["landmark_results"])
            solved = sum(x["status"].startswith("solved") for x in result["solver_results"])
            print(
                f"  objects={len(result['objects'])}, failed={failed}, "
                f"landmarks_resolved={resolved}/{len(result['landmark_results'])}, "
                f"plans_solved={solved}/{len(result['solver_results'])}"
            )
            n += 1
            if args.max_videos is not None and n >= args.max_videos:
                break


if __name__ == "__main__":
    main()
