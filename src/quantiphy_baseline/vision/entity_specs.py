from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


PROPERTY_WORDS = {
    "acceleration", "acc", "speed", "velocity", "diameter", "height", "width",
    "length", "calibre", "caliber", "thickness", "wingspan", "size", "gravity",
    "walking",
}
STOP = {"the", "a", "an", "of", "at", "in", "on", "to", "from", "and", "model"}


@dataclass
class TrackingRequest:
    entity_key: str
    display_name: str
    prompts: list[str]
    roles: list[str] = field(default_factory=list)  # target/reference
    visual_kind: str = "object"  # object/part/pair/text/region
    expected_instances: int = 1
    preferred_times_s: list[float] = field(default_factory=list)
    source_qa_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("soccerball", "soccer ball")
    s = re.sub(r"\s+", " ", s)
    return s


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", _norm(s)) if t not in STOP}


def slugify_entity(s: str) -> str:
    s = clean_entity_name(s)
    # Canonicalize common astronomy aliases: "io model" == "model of io".
    m = re.fullmatch(r"([a-z0-9]+) model", s)
    if m:
        s = f"model of {m.group(1)}"
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "entity"


def clean_entity_name(entity: str) -> str:
    s = _norm(entity)
    s = re.sub(r"^the\s+", "", s)
    s = s.replace('"', "")
    return s.strip()


def infer_visual_kind(entity: str) -> tuple[str, int, list[str]]:
    s = clean_entity_name(entity)
    notes: list[str] = []
    if s.startswith("between the two "):
        notes.append("Pair-like spatial entity: track the two physical objects separately.")
        return "pair", 2, notes
    if " text " in f" {s} " or "lettering" in s:
        notes.append("Text region: GroundingDINO may be weak; OCR/text detector is a recommended fallback.")
        return "text", 1, notes
    if any(k in s for k in ["outer end of", "edge of", "tip", "top edge", "bottom edge"]):
        notes.append("Part/landmark entity: track the parent object, then derive the requested edge/keypoint from its mask.")
        return "part", 1, notes
    if s.startswith("between "):
        return "region", 1, notes
    return "object", 1, notes


def _singularize_simple(s: str) -> str:
    if s.endswith("cars"):
        return s[:-1]
    if s.endswith("signs"):
        return s[:-1]
    if s.endswith("balls"):
        return s[:-1]
    return s


def prompt_candidates(entity: str) -> list[str]:
    raw = clean_entity_name(entity)
    visual_kind, expected, _ = infer_visual_kind(entity)
    prompts: list[str] = []

    if visual_kind == "pair" and raw.startswith("between the two "):
        raw = raw[len("between the two "):]
        raw = _singularize_simple(raw)

    # Strongest prompt first: preserve attributes / referring expression.
    prompts.append(raw)

    # Parenthetical aliases: "person on the left (the passer)".
    paren = re.findall(r"\((.*?)\)", raw)
    base = re.sub(r"\s*\([^)]*\)", "", raw).strip()
    if base and base not in prompts:
        prompts.append(base)
    for p in paren:
        p = re.sub(r"^the\s+", "", p.strip())
        if p and p not in prompts:
            prompts.append(p)

    # Part expressions: use parent as fallback.
    m = re.search(r"(?:outer end|edge|tip|top edge|bottom edge) of (?:the )?(.+)$", raw)
    if m:
        parent = m.group(1).strip()
        if parent not in prompts:
            prompts.append(parent)

    # Text on object: fall back to the containing object.
    m = re.search(r"(?:text|lettering) on (?:the )?(.+)$", raw)
    if m:
        parent = m.group(1).strip()
        if parent not in prompts:
            prompts.append(parent)

    # A weak generic noun fallback can rescue unusual referring expressions.
    words = re.findall(r"[a-z0-9]+", base)
    generic_candidates = [
        "ball", "person", "car", "boat", "bird", "bicycle", "wheel", "pier", "bench",
        "table", "cup", "pen", "cookie", "notebook", "desk", "slope", "floor", "ruler",
        "lane", "astronaut", "shark", "bubble", "droplet", "bottle", "sculpture", "pedestal",
        "panel", "road sign", "trash bag", "trash can", "bowling pin",
    ]
    for g in generic_candidates:
        if all(w in words for w in g.split()) and g not in prompts:
            prompts.append(g)

    # GroundingDINO works best with concise phrase labels.
    out: list[str] = []
    for p in prompts:
        p = re.sub(r"\s+", " ", p).strip(" .")
        if p and p not in out:
            out.append(p)
    return out[:5]


def extract_reference_entity(prior_description: str | None, candidate_entities: Iterable[str]) -> str | None:
    if not prior_description:
        return None
    desc = _norm(prior_description)
    if not desc or "gravity" in desc:
        return None

    # "diameter of the ping pong ball", "velocity of the soccer ball at 1.5s", ...
    m = re.search(r"\bof (?:the )?(.+)$", desc)
    if m:
        ref = re.sub(r"\s+at\s+[-+]?\d*\.?\d+\s*s$", "", m.group(1).strip())
        ref_tokens = _tokens(ref)
        best, best_score = None, 0.0
        for c in candidate_entities:
            ct = _tokens(c)
            if not ct:
                continue
            inter = len(ref_tokens & ct)
            union = len(ref_tokens | ct)
            score = inter / union if union else 0.0
            if ref_tokens and ref_tokens <= ct:
                score += 0.5
            if score > best_score:
                best_score, best = score, c
        return best if best is not None and best_score > 0.15 else "the " + ref

    # "Callisto's model speed" -> "model of callisto"
    m = re.match(r"(.+?)'s\s+model\b", desc)
    if m:
        return "the model of " + m.group(1).strip()

    words = [w for w in re.findall(r"[a-z0-9]+", desc) if w not in PROPERTY_WORDS]
    rough = " ".join(words).strip()
    if rough and rough not in {"walking", "pedestrian walking"}:
        rough_tokens = _tokens(rough)
    else:
        rough_tokens = _tokens(desc)

    candidates = list(candidate_entities)
    if candidates:
        best = None
        best_score = 0.0
        for c in candidates:
            ct = _tokens(c)
            if not ct:
                continue
            inter = len(rough_tokens & ct)
            union = len(rough_tokens | ct)
            score = inter / union if union else 0.0
            # Reward exact token containment heavily.
            if rough_tokens and rough_tokens <= ct:
                score += 0.5
            if score > best_score:
                best_score, best = score, c
        if best is not None and best_score > 0.15:
            return best

    if rough:
        return "the " + rough
    return None


def _collect_times(question: dict[str, Any]) -> list[float]:
    t = question.get("temporal") or {}
    vals = [t.get("time_s"), t.get("start_s"), t.get("end_s")]
    p = question.get("prior") or {}
    vals.append(p.get("timestamp_s"))
    return sorted({float(v) for v in vals if isinstance(v, (int, float))})


def build_tracking_requests(group: dict[str, Any]) -> list[TrackingRequest]:
    all_targets: list[str] = []
    for q in group.get("questions", []):
        all_targets.extend(q.get("target_entities") or [])

    requests: dict[str, TrackingRequest] = {}

    def add(entity: str, role: str, qa_id: str | None, times: list[float]):
        visual_kind, expected_instances, notes = infer_visual_kind(entity)
        key = slugify_entity(entity)
        if key not in requests:
            requests[key] = TrackingRequest(
                entity_key=key,
                display_name=entity,
                prompts=prompt_candidates(entity),
                roles=[role],
                visual_kind=visual_kind,
                expected_instances=expected_instances,
                notes=notes,
            )
        r = requests[key]
        if role not in r.roles:
            r.roles.append(role)
        if qa_id is not None and qa_id not in r.source_qa_ids:
            r.source_qa_ids.append(str(qa_id))
        r.preferred_times_s = sorted(set(r.preferred_times_s + times))

    for q in group.get("questions", []):
        qa_id = str(q.get("qa_id")) if q.get("qa_id") is not None else None
        times = _collect_times(q)
        for e in q.get("target_entities") or []:
            add(e, "target", qa_id, times)

        prior = q.get("prior") or {}
        ref = extract_reference_entity(prior.get("description"), all_targets)
        if ref:
            ref_times = times[:]
            if isinstance(prior.get("timestamp_s"), (int, float)):
                ref_times.append(float(prior["timestamp_s"]))
            add(ref, "reference", qa_id, sorted(set(ref_times)))

    return list(requests.values())
