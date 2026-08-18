from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any


GEOMETRY_TYPES = {
    "point",
    "point_set",
    "boundary_segment",
    "polyline",
    "region",
    "mask",
}

CANDIDATE_GENERATORS = {
    "whole_mask",
    "mask_contour",
    "extreme_points",
    "major_axis_endpoints",
    "major_axis_end_caps",
    "skeleton_endpoints",
    "polygon_corners",
    "contact_regions",
    "semantic_part_grounding",
    "text_detection",
}

SELECTOR_OPERATORS = {
    "leftmost",
    "rightmost",
    "topmost",
    "bottommost",
    "closest_to",
    "farthest_from",
    "facing",
    "attached_to",
    "opposite_to_attachment",
    "inside",
    "outside",
    "semantic_match",
}

DIRECTION_FRAMES = {"image", "object", "scene", "world"}
RESOLUTION_POLICIES = {
    "geometry_only",
    "geometry_then_vlm",
    "part_grounding_then_sam",
    "vlm_required",
}
LANDMARK_STATUSES = {"planned", "ambiguous", "unresolved"}
ENTITY_ROLES = {"measurement", "selector_reference", "calibration_reference"}
OPERAND_ROLES = {"target", "from", "to", "reference"}


class SemanticPlanValidationError(ValueError):
    pass


def _require_text(value: Any, path: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise SemanticPlanValidationError(f"{path} must be a non-empty string")
    return value


@dataclass(frozen=True)
class LandmarkSelector:
    operator: str
    reference_entity_id: str | None = None
    direction_frame: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> LandmarkSelector | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise SemanticPlanValidationError("landmark.selector must be an object or null")
        operator = _require_text(value.get("operator"), "landmark.selector.operator")
        if operator not in SELECTOR_OPERATORS:
            raise SemanticPlanValidationError(f"unsupported selector operator: {operator}")
        direction_frame = value.get("direction_frame")
        if direction_frame is not None and direction_frame not in DIRECTION_FRAMES:
            raise SemanticPlanValidationError(
                f"unsupported direction frame: {direction_frame}"
            )
        reference_entity_id = value.get("reference_entity_id")
        if reference_entity_id is not None:
            reference_entity_id = _require_text(
                reference_entity_id, "landmark.selector.reference_entity_id"
            )
        if operator in {
            "closest_to",
            "farthest_from",
            "facing",
            "attached_to",
            "opposite_to_attachment",
            "inside",
            "outside",
        } and reference_entity_id is None:
            raise SemanticPlanValidationError(
                f"selector {operator} requires reference_entity_id"
            )
        return cls(
            operator=operator,
            reference_entity_id=reference_entity_id,
            direction_frame=direction_frame,
        )


@dataclass(frozen=True)
class LandmarkProgram:
    raw_text: str
    semantic_type: str
    geometry_type: str
    candidate_generator: str
    selector: LandmarkSelector | None = None
    resolution_policy: str = "geometry_then_vlm"
    status: str = "planned"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LandmarkProgram:
        if not isinstance(value, dict):
            raise SemanticPlanValidationError("operand.landmark must be an object")
        geometry_type = _require_text(value.get("geometry_type"), "landmark.geometry_type")
        if geometry_type not in GEOMETRY_TYPES:
            raise SemanticPlanValidationError(f"unsupported geometry type: {geometry_type}")
        generator = _require_text(
            value.get("candidate_generator"), "landmark.candidate_generator"
        )
        if generator not in CANDIDATE_GENERATORS:
            raise SemanticPlanValidationError(f"unsupported candidate generator: {generator}")
        policy = str(value.get("resolution_policy") or "geometry_then_vlm")
        if policy not in RESOLUTION_POLICIES:
            raise SemanticPlanValidationError(f"unsupported resolution policy: {policy}")
        status = str(value.get("status") or "planned")
        if status not in LANDMARK_STATUSES:
            raise SemanticPlanValidationError(f"unsupported landmark status: {status}")
        semantic_type = _require_text(value.get("semantic_type"), "landmark.semantic_type")
        if semantic_type == "whole_object":
            if geometry_type != "mask" or generator != "whole_mask":
                raise SemanticPlanValidationError(
                    "whole_object must use geometry_type=mask and candidate_generator=whole_mask"
                )
        return cls(
            raw_text=_require_text(value.get("raw_text") or semantic_type, "landmark.raw_text"),
            semantic_type=semantic_type,
            geometry_type=geometry_type,
            candidate_generator=generator,
            selector=LandmarkSelector.from_dict(value.get("selector")),
            resolution_policy=policy,
            status=status,
        )


@dataclass(frozen=True)
class SemanticEntity:
    entity_id: str
    category: str
    referring_expression: str
    tracking_prompts: tuple[str, ...] = ()
    instance_selector: dict[str, Any] | None = None
    expected_instances: int = 1
    role: str = "measurement"
    tracking_requirement: str = "required"

    @classmethod
    def from_dict(cls, value: dict[str, Any], path: str = "entity") -> SemanticEntity:
        if not isinstance(value, dict):
            raise SemanticPlanValidationError(f"{path} must be an object")
        role = str(value.get("role") or "measurement")
        if role not in ENTITY_ROLES:
            raise SemanticPlanValidationError(f"unsupported entity role: {role}")
        requirement = str(value.get("tracking_requirement") or "required")
        if requirement not in {"required", "optional", "context_only"}:
            raise SemanticPlanValidationError(
                f"unsupported tracking requirement: {requirement}"
            )
        prompts = []
        for prompt in value.get("tracking_prompts") or []:
            prompt = str(prompt).strip()
            if prompt and prompt not in prompts:
                prompts.append(prompt)
        expected = int(value.get("expected_instances") or 1)
        if expected < 1:
            raise SemanticPlanValidationError(f"{path}.expected_instances must be >= 1")
        return cls(
            entity_id=_require_text(value.get("entity_id"), f"{path}.entity_id"),
            category=_require_text(value.get("category"), f"{path}.category"),
            referring_expression=_require_text(
                value.get("referring_expression") or value.get("category"),
                f"{path}.referring_expression",
            ),
            tracking_prompts=tuple(prompts[:5]),
            instance_selector=value.get("instance_selector"),
            expected_instances=expected,
            role=role,
            tracking_requirement=requirement,
        )


@dataclass(frozen=True)
class SemanticOperand:
    operand_id: str
    role: str
    parent: SemanticEntity
    landmark: LandmarkProgram

    @classmethod
    def from_dict(cls, value: dict[str, Any], index: int) -> SemanticOperand:
        if not isinstance(value, dict):
            raise SemanticPlanValidationError(f"operands[{index}] must be an object")
        role = str(value.get("role") or value.get("operand_id") or "target")
        if role not in OPERAND_ROLES:
            raise SemanticPlanValidationError(f"unsupported operand role: {role}")
        parent = SemanticEntity.from_dict(
            value.get("parent") or {}, f"operands[{index}].parent"
        )
        landmark = LandmarkProgram.from_dict(value.get("landmark") or {})
        if landmark.semantic_type == "whole_object":
            landmark_terms = {
                "edge",
                "end",
                "tip",
                "corner",
                "surface",
                "deck",
                "floor",
                "base",
                "underside",
                "beak",
                "nose",
                "bumper",
                "handle",
                "wheel",
                "wing",
                "text",
                "lettering",
                "seat",
                "opening",
            }
            raw_terms = set(re.findall(r"[a-z0-9]+", landmark.raw_text.lower()))
            category_terms = set(re.findall(r"[a-z0-9]+", parent.category.lower()))
            suspicious = (raw_terms & landmark_terms) - category_terms
            if re.search(r"\b(?:top|bottom|left|right|side) of\b", landmark.raw_text.lower()):
                suspicious.add("directional-part")
            if suspicious:
                raise SemanticPlanValidationError(
                    "whole_object cannot hide an explicit landmark term: "
                    + ", ".join(sorted(suspicious))
                )
        return cls(
            operand_id=_require_text(value.get("operand_id") or role, f"operands[{index}].operand_id"),
            role=role,
            parent=parent,
            landmark=landmark,
        )


@dataclass(frozen=True)
class SemanticPlan:
    relation: str
    operands: tuple[SemanticOperand, ...]
    auxiliary_entities: tuple[SemanticEntity, ...] = ()
    warnings: tuple[str, ...] = ()
    planner_version: str = "llm-semantic-v1"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SemanticPlan:
        if not isinstance(value, dict):
            raise SemanticPlanValidationError("semantic plan must be an object")
        operands = tuple(
            SemanticOperand.from_dict(item, index)
            for index, item in enumerate(value.get("operands") or [])
        )
        if not operands:
            raise SemanticPlanValidationError("semantic plan must contain at least one operand")
        ids = [item.operand_id for item in operands]
        if len(ids) != len(set(ids)):
            raise SemanticPlanValidationError("operand_id values must be unique")
        auxiliary = tuple(
            SemanticEntity.from_dict(item, f"auxiliary_entities[{index}]")
            for index, item in enumerate(value.get("auxiliary_entities") or [])
        )
        entity_ids = {item.parent.entity_id for item in operands} | {
            item.entity_id for item in auxiliary
        }
        for operand in operands:
            selector = operand.landmark.selector
            if selector and selector.reference_entity_id not in {None, *entity_ids}:
                raise SemanticPlanValidationError(
                    f"unknown selector reference: {selector.reference_entity_id}"
                )
        relation = _require_text(value.get("relation"), "relation")
        if relation in {"distance_between", "between_landmarks", "vertical_distance"}:
            measurement_operands = [x for x in operands if x.role in {"from", "to", "target"}]
            if len(measurement_operands) < 2:
                raise SemanticPlanValidationError(
                    f"relation {relation} requires at least two operands"
                )
        return cls(
            relation=relation,
            operands=operands,
            auxiliary_entities=auxiliary,
            warnings=tuple(str(x) for x in value.get("warnings") or []),
            planner_version=str(value.get("planner_version") or "llm-semantic-v1"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SEMANTIC_PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["relation", "operands"],
    "properties": {
        "relation": {"type": "string"},
        "operands": {"type": "array", "minItems": 1},
        "auxiliary_entities": {"type": "array"},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "planner_version": {"type": "string"},
    },
}
