from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize normalized landmark programs from measurement plan JSONL."
    )
    parser.add_argument(
        "--plans", default="data/processed/measurement_plans_by_video.jsonl"
    )
    parser.add_argument("--output", default="outputs/planning/landmark_eda.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    counts: Counter[tuple[str, ...]] = Counter()
    examples: dict[tuple[str, ...], str] = {}
    with Path(args.plans).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for plan in record.get("measurement_plans") or []:
                for operand in plan.get("operands") or []:
                    landmark = operand.get("landmark") or {}
                    selector = landmark.get("selector") or {}
                    key = (
                        str(landmark.get("kind") or "unknown"),
                        str(landmark.get("geometry_type") or "unknown"),
                        str(landmark.get("candidate_generator") or "unknown"),
                        str(selector.get("operator") or "none"),
                        str(selector.get("reference_entity_id") or "none"),
                        str(landmark.get("status") or "unknown"),
                    )
                    counts[key] += 1
                    examples.setdefault(key, str(landmark.get("raw_text") or operand.get("raw_text")))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "count",
                "semantic_type",
                "geometry_type",
                "candidate_generator",
                "selector_operator",
                "reference_entity",
                "status",
                "example",
            ]
        )
        for key, count in counts.most_common():
            writer.writerow([count, *key, examples[key]])
    print(f"Landmark program groups: {len(counts)}")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
