from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantiphy_baseline.measurement_plan import build_measurement_plans
from quantiphy_baseline.planning import LLMSemanticPlanner
from quantiphy_baseline.vision.entity_specs import build_tracking_requests


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build solver-ready measurement plans and canonical tracking requests."
    )
    parser.add_argument(
        "--groups", default="data/processed/grouped_by_video.jsonl"
    )
    parser.add_argument(
        "--output", default="data/processed/measurement_plans_by_video.jsonl"
    )
    parser.add_argument("--semantic-model", default=None)
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--semantic-cache", default="outputs/planning/semantic_plan_cache.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    groups_path = Path(args.groups)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    semantic_planner = None
    if args.semantic_model:
        semantic_planner = LLMSemanticPlanner(
            model=args.semantic_model,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            cache_path=args.semantic_cache,
        )

    videos = 0
    plans_count = 0
    requests_count = 0
    with groups_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            group = json.loads(line)
            plans = build_measurement_plans(group, semantic_planner=semantic_planner)
            requests = build_tracking_requests(group, plans=plans)
            record = {
                "video_id": group.get("video_id"),
                "video_type": group.get("video_type"),
                "fps": group.get("fps"),
                "measurement_plans": [plan.to_dict() for plan in plans],
                "tracking_requests": [asdict(request) for request in requests],
            }
            destination.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )
            videos += 1
            plans_count += len(plans)
            requests_count += len(requests)

    print(f"Videos               : {videos}")
    print(f"Measurement plans    : {plans_count}")
    print(f"Tracking requests    : {requests_count}")
    print(f"Saved to             : {output_path}")


if __name__ == "__main__":
    main()
