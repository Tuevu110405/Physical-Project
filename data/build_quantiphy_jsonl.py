#!/usr/bin/env python3
"""
build_quantiphy_jsonl.py

Sinh 2 file:
1. parsed_questions.jsonl
2. grouped_by_video.jsonl

Usage:
    python build_quantiphy_jsonl.py \
        --input validation_dataset.csv \
        --parsed-out data/processed/parsed_questions.jsonl \
        --grouped-out data/processed/grouped_by_video.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# 0. Helpers
# ============================================================

def is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return str(x).strip().lower() in {"", "nan", "none", "null"}


def first_existing(row: pd.Series, *names: str, default=None):
    for name in names:
        if name in row.index and not is_missing(row[name]):
            return row[name]
    return default


def clean_text(x: Any) -> str | None:
    if is_missing(x):
        return None
    return str(x).strip()


def to_float(x: Any) -> float | None:
    if is_missing(x):
        return None
    try:
        return float(x)
    except Exception:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(x))
        return float(m.group()) if m else None


def normalize_question(text: str) -> str:
    text = text.strip().lower()

    # Các typo thực tế / typo thường gặp.
    replacements = {
        "velolicty": "velocity",
        "diasplacement": "displacement",
        "bycicle": "bicycle",
        "balck": "black",
        "soccerball": "soccer ball",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    text = re.sub(r"\s+", " ", text)
    return text


# ============================================================
# 1. Question parser
# ============================================================

UNIT_PATTERN = (
    r"(?:"
    r"km/s(?:\^?2)?|m/s(?:\^?2)?|cm/s(?:\^?2)?|mm/s(?:\^?2)?|"
    r"km/h|mph|"
    r"km|cm|mm|µm|um|nm|m"
    r")"
)


def parse_output_unit(q: str) -> str | None:
    # Ưu tiên phần "in UNIT"
    m = re.search(rf"\bin\s+({UNIT_PATTERN})\b", q, flags=re.I)
    if m:
        return m.group(1)

    # fallback
    matches = re.findall(rf"\b({UNIT_PATTERN})\b", q, flags=re.I)
    return matches[-1] if matches else None


def parse_quantity(q: str) -> tuple[str, str, str]:
    """
    return:
        quantity_family, quantity_subtype, output_dimension
    """
    rules = [
        ("acceleration", "acceleration", "acceleration"),
        ("displacement", "displacement", "length"),
        ("velocity", "velocity", "speed"),
        ("speed", "speed", "speed"),
        ("distance", "distance", "length"),
        ("orbital diameter", "orbital_diameter", "length"),
        ("wingspan", "wingspan", "length"),
        ("diameter", "diameter", "length"),
        ("thickness", "thickness", "length"),
        ("height", "height", "length"),
        ("width", "width", "length"),
        ("length", "length", "length"),
        ("size", "size", "length"),
    ]

    for token, subtype, dimension in rules:
        if token in q:
            family = "size" if subtype in {
                "orbital_diameter", "wingspan", "diameter",
                "thickness", "height", "width", "length", "size"
            } else subtype
            return family, subtype, dimension

    return "unknown", "unknown", "unknown"


def parse_component(q: str) -> str:
    if re.search(r"\bhorizontal\b|\bx[- ]component\b", q):
        return "x"
    if re.search(r"\bvertical\b|\by[- ]component\b", q):
        return "y"
    if re.search(r"\bxy[- ]plane\b|\bplanar\b", q):
        return "xy"
    if "orbital" in q:
        return "orbital"
    return "scalar"


def parse_temporal(q: str, quantity_family: str) -> dict:
    # between/from t1 to t2
    interval_patterns = [
        r"(?:from|between)\s+(?:t\s*=\s*)?(\d+(?:\.\d+)?)\s*s?"
        r"\s+(?:to|and)\s+(?:t\s*=\s*)?(\d+(?:\.\d+)?)\s*s?",
        r"between\s+(\d+(?:\.\d+)?)\s*s?\s+and\s+(\d+(?:\.\d+)?)\s*s?",
    ]
    for pat in interval_patterns:
        m = re.search(pat, q)
        if m:
            return {
                "mode": "interval",
                "time_s": None,
                "start_s": float(m.group(1)),
                "end_s": float(m.group(2)),
            }

    # at t=1.0s / at 1.0s
    m = re.search(
        r"\bat\s+(?:t\s*=\s*)?(\d+(?:\.\d+)?)\s*s\b",
        q
    )
    if m:
        return {
            "mode": "instantaneous",
            "time_s": float(m.group(1)),
            "start_s": None,
            "end_s": None,
        }

    # initial/final
    if "initial" in q:
        return {
            "mode": "initial",
            "time_s": None,
            "start_s": None,
            "end_s": None,
        }

    if "final" in q:
        return {
            "mode": "final",
            "time_s": None,
            "start_s": None,
            "end_s": None,
        }

    # average / total
    if "average" in q or "mean" in q:
        return {
            "mode": "whole_video_average",
            "time_s": None,
            "start_s": None,
            "end_s": None,
        }

    # Các đại lượng size/distance thường static nếu không có timestamp.
    if quantity_family in {"size", "distance"}:
        mode = "static"
    else:
        mode = "unspecified"

    return {
        "mode": mode,
        "time_s": None,
        "start_s": None,
        "end_s": None,
    }


def strip_question_suffix(text: str) -> str:
    text = re.sub(r"\s+in\s+" + UNIT_PATTERN + r"\s*\??$", "", text, flags=re.I)
    text = text.rstrip(" ?.")
    return text.strip()


def parse_target_entities(q: str, quantity_family: str) -> tuple[list[str], str]:
    """
    Rule parser đơn giản nhưng deterministic.

    Examples:
      "height of the cup" -> ["the cup"]
      "distance between the two balls" -> ["the two balls"]
      "distance from the car to the sign" -> ["the car", "the sign"]
    """
    body = strip_question_suffix(q)

    # from A to B
    m = re.search(
        r"\b(?:distance|displacement)\s+from\s+(.+?)\s+to\s+(.+?)$",
        body
    )
    if m:
        return [m.group(1).strip(), m.group(2).strip()], "from_to"

    # between A and B
    m = re.search(
        r"\b(?:distance|displacement)\s+between\s+(.+?)\s+and\s+(.+?)$",
        body
    )
    if m:
        return [m.group(1).strip(), m.group(2).strip()], "between"

    # "between the two black road signs" -> một referring expression,
    # vision module sẽ tách expected_instances=2.
    m = re.search(r"\b(?:distance|displacement)\s+between\s+(.+?)$", body)
    if m:
        return [m.group(1).strip()], "between"

    # General: "X of TARGET ..."
    qtokens = (
        r"orbital diameter|acceleration|displacement|velocity|speed|"
        r"distance|wingspan|diameter|thickness|height|width|length|size"
    )
    m = re.search(rf"\b(?:{qtokens})\s+of\s+(.+?)$", body)
    if m:
        target = m.group(1).strip()

        # X of target at 1.00s -> bỏ time suffix
        target = re.sub(
            r"\s+at\s+(?:t\s*=\s*)?\d+(?:\.\d+)?\s*s$",
            "",
            target,
        ).strip()

        # "... during/from ..." suffix
        target = re.sub(
            r"\s+(?:from|between)\s+\d+(?:\.\d+)?\s*s.*$",
            "",
            target,
        ).strip()

        return [target], "attribute"

    return [], "unknown"


def parse_question(question: str) -> dict:
    nq = normalize_question(question)

    family, subtype, dimension = parse_quantity(nq)
    output_unit = parse_output_unit(nq)
    temporal = parse_temporal(nq, family)
    entities, relation_type = parse_target_entities(nq, family)

    warnings = []
    if family == "unknown":
        warnings.append("Unknown quantity.")
    if not entities:
        warnings.append("No target entity extracted.")
    if output_unit is None:
        warnings.append("Output unit missing from question.")

    confidence = 1.0
    confidence -= 0.25 if family == "unknown" else 0.0
    confidence -= 0.20 if not entities else 0.0
    confidence -= 0.05 if output_unit is None else 0.0

    return {
        "normalized_question": nq,
        "quantity_family": family,
        "quantity_subtype": subtype,
        "output_dimension": dimension,
        "output_unit": output_unit,
        "output_unit_source": "question" if output_unit else None,
        "component": parse_component(nq),
        "temporal": temporal,
        "target_entities": entities,
        "relation_type": relation_type,
        "parse_confidence": max(confidence, 0.0),
        "warnings": warnings,
    }


# ============================================================
# 2. Prior parser
# ============================================================

def parse_prior(raw_prior: Any) -> dict | None:
    raw = clean_text(raw_prior)
    if raw is None:
        return None

    text = normalize_question(raw)

    timestamp_s = None
    tm = re.search(r"\bt\s*=\s*(\d+(?:\.\d+)?)\s*s?\b", text)
    if tm:
        timestamp_s = float(tm.group(1))

    # Tìm: DESCRIPTION = VALUE UNIT
    m = re.search(
        rf"(?P<desc>[a-z0-9_ ()/\-]+?)\s*=\s*"
        rf"(?P<value>[-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*"
        rf"(?P<unit>{UNIT_PATTERN})\b",
        text,
        flags=re.I,
    )

    if not m:
        return {
            "raw": raw,
            "description": None,
            "property_family": "unknown",
            "property_subtype": "unknown",
            "value": None,
            "unit": None,
            "relation": "unknown",
            "timestamp_s": timestamp_s,
            "parse_confidence": 0.2,
            "warnings": ["Could not parse prior value/unit."],
        }

    desc = m.group("desc")
    desc = re.sub(r"^t\s*=\s*\d+(?:\.\d+)?\s*s?\s*,?\s*", "", desc)
    desc = desc.strip(" ,")

    value = float(m.group("value"))
    unit = m.group("unit")

    fam, subtype, _ = parse_quantity(desc)
    if fam == "unknown":
        # infer from unit
        if "/s^2" in unit or "/s2" in unit:
            fam = subtype = "acceleration"
        elif "/s" in unit or unit in {"km/h", "mph"}:
            fam = subtype = "speed"
        else:
            fam, subtype = "size", "length"

    return {
        "raw": raw,
        "description": desc,
        "property_family": fam,
        "property_subtype": subtype,
        "value": value,
        "unit": unit,
        "relation": "exact",
        "timestamp_s": timestamp_s,
        "parse_confidence": 0.95,
        "warnings": [],
    }


# ============================================================
# 3. Depth parser
# ============================================================

def split_depth_lines(raw: str) -> list[str]:
    # Dataset có thể dùng newline hoặc semicolon.
    parts = re.split(r"[\n;]+", raw)
    return [x.strip() for x in parts if x.strip()]


def parse_depth_info(raw_depth: Any) -> list[dict]:
    raw = clean_text(raw_depth)
    if raw is None:
        return []

    observations = []

    for line in split_depth_lines(raw):
        # Example:
        # t=0s, distance_cup_camera = 1.2100 m
        m = re.search(
            r"(?:t\s*=\s*(?P<t>\d+(?:\.\d+)?)\s*s?\s*,?\s*)?"
            r"(?P<key>[A-Za-z0-9_\- ]+?)\s*=\s*"
            r"(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*"
            rf"(?P<unit>{UNIT_PATTERN})\b",
            line,
        )
        if not m:
            continue

        observations.append(
            {
                "raw": line,
                "key": m.group("key").strip(),
                "value": float(m.group("value")),
                "unit": m.group("unit"),
                "timestamp_s": float(m.group("t")) if m.group("t") else None,
            }
        )

    return observations


# ============================================================
# 4. CSV -> parsed_questions.jsonl
# ============================================================

def parse_dataframe(df: pd.DataFrame) -> list[dict]:
    records = []

    for idx, row in df.iterrows():
        question = clean_text(
            first_existing(
                row,
                "question",
                "Question",
                "query",
                "text",
            )
        )
        if question is None:
            continue

        parsed = parse_question(question)

        qa_id = first_existing(
            row,
            "qa_id",
            "id",
            "ID",
            "index",
            "Unnamed: 0",
            default=idx,
        )

        record = {
            "qa_id": str(qa_id),
            "video_id": clean_text(
                first_existing(row, "video_id", "video", "video_name")
            ),
            "video_source": clean_text(
                first_existing(row, "video_source", "source")
            ),
            "video_type": clean_text(
                first_existing(row, "video_type", "type")
            ),
            "fps": to_float(first_existing(row, "fps", "FPS")),
            "inference_type": clean_text(
                first_existing(
                    row,
                    "inference_type",
                    "category",
                    "inference",
                )
            ),
            "raw_question": question,
            **parsed,
            "prior": parse_prior(
                first_existing(
                    row,
                    "ground_truth_prior",
                    "prior",
                    "physical_prior",
                )
            ),
            "depth_observations": parse_depth_info(
                first_existing(
                    row,
                    "depth_info",
                    "depth",
                    "depth_information",
                )
            ),
            "ground_truth": to_float(
                first_existing(
                    row,
                    "ground_truth_posterior",
                    "answer",
                    "ground_truth",
                    "label",
                )
            ),
            "parser_version": "rule-v1",
        }

        records.append(record)

    return records


def write_jsonl(records: list[dict], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


# ============================================================
# 5. Context resolution trong cùng video
# ============================================================

LENGTH_UNITS = {"nm", "um", "µm", "mm", "cm", "m", "km"}
SPEED_UNITS = {"mm/s", "cm/s", "m/s", "km/s", "km/h", "mph"}
ACC_UNITS = {"mm/s^2", "cm/s^2", "m/s^2", "km/s^2"}


def compatible_units(dimension: str) -> set[str]:
    if dimension == "length":
        return LENGTH_UNITS
    if dimension == "speed":
        return SPEED_UNITS
    if dimension == "acceleration":
        return ACC_UNITS
    return set()


def infer_missing_units_from_same_video(records: list[dict]) -> list[dict]:
    """
    Chỉ infer khi:
    - cùng video
    - cùng output_dimension
    - unit thắng majority duy nhất
    - xuất hiện >= 2 lần
    """
    by_video = defaultdict(list)
    for rec in records:
        by_video[rec["video_id"]].append(rec)

    for _, items in by_video.items():
        by_dimension = defaultdict(list)

        for rec in items:
            dim = rec.get("output_dimension")
            unit = rec.get("output_unit")
            if dim and unit:
                by_dimension[dim].append(unit)

        inferred = {}
        for dim, units in by_dimension.items():
            counts = Counter(units)
            if not counts:
                continue

            ranked = counts.most_common()
            best_unit, best_count = ranked[0]
            second_count = ranked[1][1] if len(ranked) > 1 else 0

            if best_count >= 2 and best_count > second_count:
                inferred[dim] = best_unit

        for rec in items:
            if rec.get("output_unit") is not None:
                continue

            dim = rec.get("output_dimension")
            unit = inferred.get(dim)

            if unit and unit in compatible_units(dim):
                rec["output_unit"] = unit
                rec["output_unit_source"] = "video_context"
                rec.setdefault("warnings", []).append(
                    f"Output unit inferred as '{unit}' from same-video "
                    f"{dim} questions."
                )

    return records


# ============================================================
# 6. parsed_questions.jsonl -> grouped_by_video.jsonl
# ============================================================

def unique_dicts(items: list[dict]) -> list[dict]:
    seen = set()
    out = []

    for item in items:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append(item)

    return out


def group_by_video(
    records: list[dict],
    video_dir: str | Path | None = None,
) -> list[dict]:

    groups = defaultdict(list)
    for rec in records:
        groups[rec["video_id"]].append(rec)

    video_dir = Path(video_dir) if video_dir else None

    output = []

    for video_id, questions in groups.items():
        first = questions[0]

        video_path = None
        if video_dir:
            for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
                candidate = video_dir / f"{video_id}{ext}"
                if candidate.exists():
                    video_path = str(candidate)
                    break

        priors = [
            q["prior"]
            for q in questions
            if q.get("prior") is not None
        ]

        depths = []
        for q in questions:
            depths.extend(q.get("depth_observations", []))

        group_warnings = []

        # Kiểm tra metadata consistency.
        fps_values = {
            q["fps"] for q in questions
            if q.get("fps") is not None
        }
        if len(fps_values) > 1:
            group_warnings.append(
                f"Inconsistent FPS values: {sorted(fps_values)}"
            )

        source_values = {
            q["video_source"] for q in questions
            if q.get("video_source")
        }
        if len(source_values) > 1:
            group_warnings.append(
                f"Inconsistent video_source values: {sorted(source_values)}"
            )

        group = {
            "video_id": video_id,
            "video_source": first.get("video_source"),
            "video_type": first.get("video_type"),
            "fps": first.get("fps"),
            "video_path": video_path,
            "prior_variants": unique_dicts(priors),
            "depth_observations": unique_dicts(depths),
            "questions": questions,
            "group_warnings": group_warnings,
        }

        output.append(group)

    # deterministic ordering
    output.sort(key=lambda x: str(x["video_id"]))
    return output


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="CSV validation/test của QuantiPhy",
    )
    parser.add_argument(
        "--parsed-out",
        default="data/processed/parsed_questions.jsonl",
    )
    parser.add_argument(
        "--grouped-out",
        default="data/processed/grouped_by_video.jsonl",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Optional: thư mục chứa <video_id>.mp4",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # STEP 1: Question -> JSON records
    records = parse_dataframe(df)

    # STEP 2: context inference cùng video
    records = infer_missing_units_from_same_video(records)

    # STEP 3: ghi parsed_questions.jsonl
    write_jsonl(records, args.parsed_out)

    # STEP 4: group toàn bộ QA theo video_id
    groups = group_by_video(
        records,
        video_dir=args.video_dir,
    )

    # STEP 5: ghi grouped_by_video.jsonl
    write_jsonl(groups, args.grouped_out)

    print(f"Rows in CSV              : {len(df)}")
    print(f"Parsed questions         : {len(records)}")
    print(f"Unique videos            : {len(groups)}")
    print(f"Saved parsed JSONL to    : {args.parsed_out}")
    print(f"Saved grouped JSONL to   : {args.grouped_out}")


if __name__ == "__main__":
    main()
