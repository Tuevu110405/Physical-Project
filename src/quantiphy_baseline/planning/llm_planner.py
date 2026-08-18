from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .schema import SemanticPlan, SemanticPlanValidationError


SYSTEM_PROMPT = """You are the semantic planning stage of a quantitative video reasoning system.
Read the raw question and extract only visual entities, measurement operands, landmarks, and their
relations. Numeric time, units, quantity type, and calibration are handled by deterministic code.

Return one JSON object. Never return markdown or explanation. A landmark is whole_object only when
the measurement truly uses the complete object. If the question names a part, edge, end, surface,
corner, tip, text, or relational location, represent it explicitly. Do not silently use whole_object.

Allowed geometry_type: point, point_set, boundary_segment, polyline, region, mask.
Allowed candidate_generator: whole_mask, mask_contour, extreme_points, major_axis_endpoints,
major_axis_end_caps, skeleton_endpoints, polygon_corners, contact_regions,
semantic_part_grounding, text_detection.
Allowed selector.operator: leftmost, rightmost, topmost, bottommost, closest_to, farthest_from,
facing, attached_to, opposite_to_attachment, inside, outside, semantic_match.
Allowed direction_frame: image, object, scene, world.
Allowed resolution_policy: geometry_only, geometry_then_vlm, part_grounding_then_sam, vlm_required.

Each operand has operand_id, role (target/from/to/reference), parent, and landmark. Parent contains
entity_id, category, referring_expression, tracking_prompts, instance_selector, expected_instances.
Use auxiliary_entities for scene entities needed only to select a landmark, such as shore for the
outer end of a pier. Every selector reference_entity_id must match a parent or auxiliary entity_id.
Use deterministic canonical-slug entity IDs across questions: pier, boat, left_car, right_car. Do not
add an arbitrary numeric suffix for a single unqualified object.

Example landmark for a complete boat:
{"raw_text":"boat","semantic_type":"whole_object","geometry_type":"mask",
 "candidate_generator":"whole_mask","selector":null,"resolution_policy":"geometry_only",
 "status":"planned"}

Example landmark for outer end of a pier:
{"raw_text":"outer end of the pier","semantic_type":"outer_end",
 "geometry_type":"boundary_segment","candidate_generator":"major_axis_end_caps",
 "selector":{"operator":"opposite_to_attachment","reference_entity_id":"shore_0",
 "direction_frame":"scene"},"resolution_policy":"geometry_then_vlm","status":"planned"}
"""


class SemanticPlanCache:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._items: dict[str, dict[str, Any]] = {}
        if self.path is not None and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                    self._items[str(item["cache_key"])] = item["plan"]
                except (KeyError, TypeError, json.JSONDecodeError):
                    continue

    def get(self, key: str) -> dict[str, Any] | None:
        return self._items.get(key)

    def put(self, key: str, plan: dict[str, Any]) -> None:
        if self._items.get(key) == plan:
            return
        self._items[key] = plan
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"cache_key": key, "plan": plan}) + "\n")


class LLMSemanticPlanner:
    """OpenAI-compatible semantic planner suitable for hosted APIs or a local vLLM server."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str | None = None,
        cache_path: str | Path | None = None,
        timeout_s: float = 90.0,
        prompt_version: str = "landmark-semantic-v1",
        transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        self.cache = SemanticPlanCache(cache_path)
        self.timeout_s = timeout_s
        self.prompt_version = prompt_version
        self.transport = transport

    def _cache_key(self, question: dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "prompt_version": self.prompt_version,
            "raw_question": question.get("raw_question") or question.get("question"),
            "target_entities": question.get("target_entities") or [],
            "relation_type": question.get("relation_type"),
            "quantity_family": question.get("quantity_family"),
            "quantity_subtype": question.get("quantity_subtype"),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _request_payload(self, question: dict[str, Any]) -> dict[str, Any]:
        user_payload = {
            "raw_question": question.get("raw_question") or question.get("question"),
            "existing_parse": {
                "target_entities": question.get("target_entities") or [],
                "relation_type": question.get("relation_type"),
                "quantity_family": question.get("quantity_family"),
                "quantity_subtype": question.get("quantity_subtype"),
                "component": question.get("component"),
                "temporal": question.get("temporal") or {},
            },
        }
        return {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
        }

    def _call(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.transport is not None:
            return self.transport(payload)
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
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"semantic planner HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"semantic planner connection failed: {exc.reason}") from exc

    @staticmethod
    def _extract_json(response: dict[str, Any]) -> dict[str, Any]:
        if "relation" in response and "operands" in response:
            return response
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("semantic planner returned an unsupported response") from exc
        if isinstance(content, dict):
            return content
        try:
            return json.loads(str(content))
        except json.JSONDecodeError as exc:
            raise RuntimeError("semantic planner did not return valid JSON") from exc

    def plan(self, question: dict[str, Any]) -> SemanticPlan:
        key = self._cache_key(question)
        cached = self.cache.get(key)
        if cached is not None:
            return SemanticPlan.from_dict(cached)
        raw = self._extract_json(self._call(self._request_payload(question)))
        try:
            plan = SemanticPlan.from_dict(raw)
        except SemanticPlanValidationError:
            raise
        serialized = plan.to_dict()
        self.cache.put(key, serialized)
        return plan
