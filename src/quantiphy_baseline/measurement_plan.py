from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


SIZE_SUBTYPES = {
    "size",
    "length",
    "width",
    "height",
    "diameter",
    "orbital_diameter",
    "thickness",
    "wingspan",
    "shoulder_breadth",
    "side_length",
}

COMPOUND_OBJECTS = (
    "ping pong ball",
    "tennis ball",
    "soccer ball",
    "billiard ball",
    "bowling ball",
    "bowling pin",
    "road sign",
    "trash bag",
    "trash can",
    "pencil bag",
    "note book",
    "notebook",
    "roof frame",
    "water surface",
)

GENERIC_OBJECT_HEADS = (
    "person",
    "pedestrian",
    "ball",
    "boat",
    "car",
    "bicycle",
    "wheel",
    "pier",
    "bench",
    "table",
    "cup",
    "pen",
    "pencil",
    "cookie",
    "desk",
    "slope",
    "floor",
    "ruler",
    "lane",
    "astronaut",
    "shark",
    "bubble",
    "droplet",
    "bottle",
    "sculpture",
    "pedestal",
    "panel",
    "sign",
    "stairs",
    "steps",
    "bird",
    "eagle",
    "house",
    "booth",
    "block",
    "water",
    "ground",
    "cabin",
    "frame",
)

SELECTOR_WORDS = {
    "left",
    "right",
    "upper",
    "lower",
    "top",
    "bottom",
    "first",
    "second",
    "black",
    "white",
    "orange",
    "yellow",
    "red",
    "blue",
    "green",
}


def _norm(text: str) -> str:
    text = str(text or "").lower().strip().replace('"', "")
    text = text.replace("soccerball", "soccer ball")
    text = text.replace("note-book", "note book")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .?")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _norm(text)).strip("_") or "entity"


@dataclass(frozen=True)
class LandmarkSpec:
    kind: str = "whole_object"
    raw_text: str | None = None
    axis: str | None = None
    selector: dict[str, Any] | None = None
    extraction: str | None = None
    geometry_type: str = "mask"
    candidate_generator: str = "whole_mask"
    resolution_policy: str = "geometry_only"
    status: str = "planned"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LandmarkSpec":
        return cls(
            kind=str(value.get("kind") or "whole_object"),
            raw_text=value.get("raw_text"),
            axis=value.get("axis"),
            selector=value.get("selector"),
            extraction=value.get("extraction"),
            geometry_type=str(value.get("geometry_type") or "mask"),
            candidate_generator=str(value.get("candidate_generator") or "whole_mask"),
            resolution_policy=str(value.get("resolution_policy") or "geometry_only"),
            status=str(value.get("status") or "planned"),
        )


@dataclass(frozen=True)
class CanonicalEntity:
    raw_text: str
    parent_key: str
    parent_name: str
    tracking_key: str
    tracking_name: str
    prompts: tuple[str, ...]
    landmark: LandmarkSpec = field(default_factory=LandmarkSpec)
    instance_selector: str | None = None
    expected_instances: int = 1
    visual_kind: str = "object"


@dataclass(frozen=True)
class MeasurementOperand:
    role: str
    raw_text: str
    parent_key: str
    tracking_key: str
    parent_name: str
    landmark: LandmarkSpec
    instance_selector: str | None = None
    expected_instances: int = 1
    operand_id: str | None = None
    tracking_name: str | None = None
    prompts: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MeasurementOperand":
        return cls(
            role=str(value.get("role") or "target"),
            raw_text=str(value.get("raw_text") or ""),
            parent_key=str(value.get("parent_key") or ""),
            tracking_key=str(value.get("tracking_key") or ""),
            parent_name=str(value.get("parent_name") or ""),
            landmark=LandmarkSpec.from_dict(value.get("landmark") or {}),
            instance_selector=value.get("instance_selector"),
            expected_instances=int(value.get("expected_instances") or 1),
            operand_id=value.get("operand_id"),
            tracking_name=value.get("tracking_name"),
            prompts=tuple(value.get("prompts") or ()),
        )


@dataclass
class MeasurementPlan:
    plan_id: str
    qa_id: str
    video_id: str | None
    raw_question: str
    quantity_family: str
    quantity_subtype: str
    measurement_kind: str
    output_unit: str | None
    relation: str
    reduction: str
    axis: str | None
    temporal: dict[str, Any]
    operands: list[MeasurementOperand]
    calibration: dict[str, Any]
    auxiliary_entities: list[dict[str, Any]] = field(default_factory=list)
    planner_version: str = "legacy-rule-v1"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MeasurementPlan":
        return cls(
            plan_id=str(value.get("plan_id") or "unknown"),
            qa_id=str(value.get("qa_id") or "unknown"),
            video_id=str(value["video_id"]) if value.get("video_id") is not None else None,
            raw_question=str(value.get("raw_question") or ""),
            quantity_family=str(value.get("quantity_family") or "unknown"),
            quantity_subtype=str(value.get("quantity_subtype") or "unknown"),
            measurement_kind=str(value.get("measurement_kind") or "unknown"),
            output_unit=value.get("output_unit"),
            relation=str(value.get("relation") or "attribute"),
            reduction=str(value.get("reduction") or "direct"),
            axis=value.get("axis"),
            temporal=dict(value.get("temporal") or {}),
            operands=[
                MeasurementOperand.from_dict(item) for item in value.get("operands") or []
            ],
            calibration=dict(value.get("calibration") or {}),
            auxiliary_entities=list(value.get("auxiliary_entities") or []),
            planner_version=str(value.get("planner_version") or "legacy-rule-v1"),
            warnings=list(value.get("warnings") or []),
        )


def _extract_selector(text: str) -> str | None:
    words = re.findall(r"[a-z0-9]+", _norm(text))
    selected = [word for word in words if word in SELECTOR_WORDS]
    return "_".join(selected) if selected else None


def _find_object_head(text: str) -> str:
    text = _norm(text)
    for compound in COMPOUND_OBJECTS:
        if re.search(rf"\b{re.escape(compound)}\b", text):
            return compound
    for head in GENERIC_OBJECT_HEADS:
        if re.search(rf"\b{re.escape(head)}\b", text):
            return head
    text = re.sub(r"^(?:the|a|an)\s+", "", text)
    return text


def _landmark(
    kind: str,
    raw_text: str,
    *,
    axis: str | None = None,
    selector: dict[str, Any] | None = None,
    extraction: str | None = None,
    geometry_type: str | None = None,
    candidate_generator: str | None = None,
    resolution_policy: str | None = None,
    status: str = "planned",
) -> LandmarkSpec:
    legacy_routes: dict[str, tuple[str, str, dict[str, Any] | None, str]] = {
        "opposite_attachment_end_cap": (
            "boundary_segment",
            "major_axis_end_caps",
            {"operator": "opposite_to_attachment", "reference_entity_id": "shore"},
            "geometry_then_vlm",
        ),
        "attachment_end_cap": (
            "boundary_segment",
            "major_axis_end_caps",
            {"operator": "attached_to", "reference_entity_id": "shore"},
            "geometry_then_vlm",
        ),
        "oriented_top_boundary": (
            "boundary_segment",
            "mask_contour",
            {"operator": "topmost", "direction_frame": "image"},
            "geometry_only",
        ),
        "oriented_bottom_boundary": (
            "boundary_segment",
            "mask_contour",
            {"operator": "bottommost", "direction_frame": "image"},
            "geometry_only",
        ),
        "requested_boundary": (
            "boundary_segment",
            "mask_contour",
            {"operator": "semantic_match"},
            "geometry_then_vlm",
        ),
        "skeleton_endpoint": (
            "point",
            "skeleton_endpoints",
            {"operator": "semantic_match"},
            "geometry_then_vlm",
        ),
        "ocr_region": (
            "region",
            "text_detection",
            {"operator": "semantic_match"},
            "vlm_required",
        ),
        "deck_boundary": (
            "boundary_segment",
            "semantic_part_grounding",
            {"operator": "semantic_match"},
            "part_grounding_then_sam",
        ),
        "floor_boundary": (
            "boundary_segment",
            "semantic_part_grounding",
            {"operator": "semantic_match"},
            "part_grounding_then_sam",
        ),
        "surface_boundary": (
            "boundary_segment",
            "semantic_part_grounding",
            {"operator": "semantic_match"},
            "part_grounding_then_sam",
        ),
    }
    inferred = legacy_routes.get(extraction or "")
    if inferred is not None:
        inferred_geometry, inferred_generator, inferred_selector, inferred_policy = inferred
    else:
        inferred_geometry, inferred_generator, inferred_selector, inferred_policy = (
            "mask",
            "whole_mask",
            None,
            "geometry_only",
        )
    return LandmarkSpec(
        kind=kind,
        raw_text=_norm(raw_text),
        axis=axis,
        selector=selector if selector is not None else inferred_selector,
        extraction=extraction,
        geometry_type=geometry_type or inferred_geometry,
        candidate_generator=candidate_generator or inferred_generator,
        resolution_policy=resolution_policy or inferred_policy,
        status=status,
    )


def canonicalize_entity(entity: str) -> CanonicalEntity:
    raw = _norm(entity)
    base = re.sub(r"^(?:the|a|an)\s+", "", raw)
    expected_instances = 1
    visual_kind = "object"
    instance_selector = _extract_selector(base)
    landmark = LandmarkSpec()
    parent_source = base

    pair_match = re.match(r"between\s+(?:the\s+)?two\s+(.+)$", base)
    if pair_match:
        parent_source = pair_match.group(1)
        parent_source = re.sub(r"\b(cars|signs|balls)\b", lambda m: m.group(1)[:-1], parent_source)
        expected_instances = 2
        visual_kind = "pair"
        instance_selector = "pair"

    patterns = (
        (
            r"outer end of (?:the )?(.+)$",
            "outer_end",
            "long_axis",
            "opposite_attachment_end_cap",
        ),
        (
            r"shore[- ]side edge of (?:the )?(.+)$",
            "shore_side_edge",
            "long_axis",
            "attachment_end_cap",
        ),
        (
            r"water[- ]side edge of (?:the )?(.+)$",
            "water_side_edge",
            "long_axis",
            "opposite_attachment_end_cap",
        ),
        (r"top edge of (?:the )?(.+)$", "top_edge", "vertical", "oriented_top_boundary"),
        (r"bottom edge of (?:the )?(.+)$", "bottom_edge", "vertical", "oriented_bottom_boundary"),
        (r"(?:outer )?edge of (?:the )?(.+)$", "edge", None, "requested_boundary"),
        (r"tip of (?:the )?(.+)$", "tip", "long_axis", "skeleton_endpoint"),
    )
    for pattern, kind, axis, extraction in patterns:
        match = re.search(pattern, base)
        if match:
            parent_source = match.group(1)
            landmark = _landmark(kind, base, axis=axis, extraction=extraction)
            visual_kind = "part"
            break
    else:
        tip_match = re.match(r"(.+?)\s+tip(?:\s+moving.*)?$", base)
        if tip_match:
            parent_source = tip_match.group(1)
            landmark = _landmark(
                "tip", base, axis="long_axis", extraction="skeleton_endpoint"
            )
            visual_kind = "part"
        elif " text " in f" {base} " or "lettering" in base:
            container = re.search(r"(?:text|lettering) on (?:the )?(.+)$", base)
            parent_source = container.group(1) if container else base
            landmark = _landmark("text_region", base, extraction="ocr_region")
            visual_kind = "text"
        elif re.search(r"\bpier deck\b", base):
            parent_source = "pier"
            landmark = _landmark("deck", "pier deck", axis="vertical", extraction="deck_boundary")
            visual_kind = "part"
        elif re.search(r"\bcabin floor\b", base):
            parent_source = "cabin"
            landmark = _landmark("floor", "cabin floor", axis="vertical", extraction="floor_boundary")
            visual_kind = "part"
        elif re.search(r"\bwater surface\b", base):
            parent_source = "water"
            landmark = _landmark("surface", "water surface", axis="vertical", extraction="surface_boundary")
            visual_kind = "region"

    parent_name = _find_object_head(parent_source)
    parent_key = _slug(parent_name)
    selector_for_key = None if instance_selector in {None, "pair"} else instance_selector
    tracking_key = _slug(f"{selector_for_key} {parent_name}" if selector_for_key else parent_name)
    tracking_name = f"{selector_for_key.replace('_', ' ')} {parent_name}" if selector_for_key else parent_name

    prompt_values = []
    for prompt in (tracking_name, parent_name, parent_source):
        prompt = _norm(prompt)
        if prompt and prompt not in prompt_values:
            prompt_values.append(prompt)

    return CanonicalEntity(
        raw_text=raw,
        parent_key=parent_key,
        parent_name=parent_name,
        tracking_key=tracking_key,
        tracking_name=tracking_name,
        prompts=tuple(prompt_values[:5]),
        landmark=landmark,
        instance_selector=instance_selector,
        expected_instances=expected_instances,
        visual_kind=visual_kind,
    )


def _operand(
    role: str, entity: CanonicalEntity, operand_id: str | None = None
) -> MeasurementOperand:
    return MeasurementOperand(
        role=role,
        raw_text=entity.raw_text,
        parent_key=entity.parent_key,
        tracking_key=entity.tracking_key,
        parent_name=entity.parent_name,
        landmark=entity.landmark,
        instance_selector=entity.instance_selector,
        expected_instances=entity.expected_instances,
        operand_id=operand_id or role,
        tracking_name=entity.tracking_name,
        prompts=entity.prompts,
    )


def _measurement_kind(question: dict[str, Any]) -> str:
    family = str(question.get("quantity_family") or "unknown")
    subtype = str(question.get("quantity_subtype") or family)
    relation = str(question.get("relation_type") or "attribute")
    if subtype in {"distance", "minimum_distance"} or relation in {"from_to", "between"}:
        return "distance"
    if subtype == "displacement":
        return "displacement"
    if subtype == "total_distance":
        return "path_length"
    if subtype in SIZE_SUBTYPES or family == "size":
        return "extent"
    if family in {"speed", "velocity", "acceleration"}:
        return family
    return subtype


def _reduction(question: dict[str, Any]) -> str:
    subtype = str(question.get("quantity_subtype") or "")
    mode = str((question.get("temporal") or {}).get("mode") or "")
    raw = _norm(question.get("raw_question") or "")
    if "minimum" in raw or subtype == "minimum_distance":
        return "minimum"
    if "maximum" in raw:
        return "maximum"
    if mode == "whole_video_average" or "average" in raw or "mean" in raw:
        return "time_average"
    if mode == "whole_video_total" or subtype == "total_distance":
        return "path_sum"
    if mode == "interval":
        return "interval_delta"
    return "instantaneous" if mode == "instantaneous" else "direct"


def _axis(question: dict[str, Any]) -> str | None:
    component = question.get("component")
    if component in {"x", "y", "xy", "orbital"}:
        return str(component)
    raw = _norm(question.get("raw_question") or "")
    if "vertical" in raw or " above " in f" {raw} ":
        return "vertical"
    if "horizontal" in raw:
        return "horizontal"
    return None


def _calibration_route(question: dict[str, Any]) -> dict[str, Any]:
    video_type = str(question.get("video_type") or "")
    inference_type = str(question.get("inference_type") or "")
    prior_code = video_type[0] if len(video_type) >= 1 else None
    dimension_code = video_type[1] if len(video_type) >= 2 else None
    object_code = video_type[2] if len(video_type) >= 3 else None
    background_code = video_type[3] if len(video_type) >= 4 else None
    prior_type = {"S": "size", "V": "velocity", "A": "acceleration"}.get(prior_code)
    return {
        "video_type": video_type or None,
        "inference_type": inference_type or None,
        "physical_prior": prior_type,
        "calibration_method": {
            "size": "reference_extent_scale",
            "velocity": "reference_first_derivative_scale",
            "acceleration": "reference_second_derivative_scale",
        }.get(prior_type, "unknown"),
        "dimensionality": {"2": "planar", "3": "depth_aware"}.get(dimension_code),
        "object_setting": {"S": "single", "M": "multi"}.get(object_code),
        "background": {"X": "plain", "S": "simple", "C": "complex"}.get(background_code),
        "prior_state": {"S": "static", "D": "dynamic"}.get(
            inference_type[0] if len(inference_type) >= 1 else None
        ),
        "target_state": {"S": "static", "D": "dynamic"}.get(
            inference_type[1] if len(inference_type) >= 2 else None
        ),
        "prior": question.get("prior"),
        "depth_observations": question.get("depth_observations") or [],
    }


def _replace_or_add_operand(
    operands: list[MeasurementOperand], role: str, entity: CanonicalEntity
) -> None:
    item = _operand(role, entity)
    for index, existing in enumerate(operands):
        if existing.role == role:
            operands[index] = item
            return
    operands.append(item)


def _specialize_relational_operands(
    question: dict[str, Any], operands: list[MeasurementOperand]
) -> tuple[str, list[MeasurementOperand], list[str]]:
    raw = _norm(question.get("raw_question") or "")
    relation = str(question.get("relation_type") or "attribute")
    warnings: list[str] = []

    pier_span = re.search(
        r"(?:length of )?(?:the )?(.+?) from (?:the )?shore[- ]side edge to (?:the )?water[- ]side edge",
        raw,
    )
    if pier_span:
        parent = canonicalize_entity(pier_span.group(1))
        start = CanonicalEntity(
            **{
                **parent.__dict__,
                "raw_text": f"shore-side edge of {parent.parent_name}",
                "landmark": _landmark(
                    "shore_side_edge",
                    f"shore-side edge of {parent.parent_name}",
                    axis="long_axis",
                    extraction="attachment_end_cap",
                ),
                "visual_kind": "part",
            }
        )
        end = CanonicalEntity(
            **{
                **parent.__dict__,
                "raw_text": f"water-side edge of {parent.parent_name}",
                "landmark": _landmark(
                    "water_side_edge",
                    f"water-side edge of {parent.parent_name}",
                    axis="long_axis",
                    extraction="opposite_attachment_end_cap",
                ),
                "visual_kind": "part",
            }
        )
        return "between_landmarks", [_operand("from", start), _operand("to", end)], warnings

    top_bottom = re.search(r"top edge to bottom edge", raw)
    if top_bottom and operands:
        parent = canonicalize_entity(operands[0].parent_name)
        top = CanonicalEntity(
            **{
                **parent.__dict__,
                "raw_text": f"top edge of {parent.parent_name}",
                "landmark": _landmark(
                    "top_edge", f"top edge of {parent.parent_name}", axis="vertical", extraction="oriented_top_boundary"
                ),
                "visual_kind": "part",
            }
        )
        bottom = CanonicalEntity(
            **{
                **parent.__dict__,
                "raw_text": f"bottom edge of {parent.parent_name}",
                "landmark": _landmark(
                    "bottom_edge", f"bottom edge of {parent.parent_name}", axis="vertical", extraction="oriented_bottom_boundary"
                ),
                "visual_kind": "part",
            }
        )
        return "between_landmarks", [_operand("from", top), _operand("to", bottom)], warnings

    above = re.search(
        r"(?:height|distance) of (?:the )?(.+?) above (?:the )?(.+?)(?: in [a-z0-9/^]+)?$",
        raw,
    )
    if above:
        upper = canonicalize_entity(above.group(1))
        lower = canonicalize_entity(above.group(2))
        return "vertical_distance", [_operand("from", lower), _operand("to", upper)], warnings

    if relation in {"from_to", "between"} and len(operands) < 2:
        warnings.append("Relational measurement has fewer than two resolved operands.")
    return relation, operands, warnings


def _semantic_selector_text(selector: dict[str, Any] | None) -> str | None:
    if not selector:
        return None
    values = [
        str(value) for value in selector.values() if value is not None and value != ""
    ]
    return "_".join(_slug(value) for value in values) or None


def _operands_from_semantic_plan(
    semantic_plan: Any,
) -> tuple[str, list[MeasurementOperand], list[dict[str, Any]], list[str], str]:
    """Convert a validated planning.SemanticPlan without re-parsing its text."""
    operands: list[MeasurementOperand] = []
    for semantic_operand in semantic_plan.operands:
        parent = semantic_operand.parent
        selector_text = _semantic_selector_text(parent.instance_selector)
        prompts: list[str] = []
        for value in (
            *parent.tracking_prompts,
            parent.referring_expression,
            parent.category,
        ):
            normalized = _norm(value)
            if normalized and normalized not in prompts:
                prompts.append(normalized)
        landmark_program = semantic_operand.landmark
        selector = asdict(landmark_program.selector) if landmark_program.selector else None
        landmark = LandmarkSpec(
            kind=landmark_program.semantic_type,
            raw_text=_norm(landmark_program.raw_text),
            selector=selector,
            extraction=landmark_program.candidate_generator,
            geometry_type=landmark_program.geometry_type,
            candidate_generator=landmark_program.candidate_generator,
            resolution_policy=landmark_program.resolution_policy,
            status=landmark_program.status,
        )
        entity = CanonicalEntity(
            raw_text=_norm(parent.referring_expression),
            parent_key=_slug(parent.category),
            parent_name=_norm(parent.category),
            tracking_key=_slug(parent.entity_id),
            tracking_name=_norm(parent.referring_expression),
            prompts=tuple(prompts[:5]),
            landmark=landmark,
            instance_selector=selector_text,
            expected_instances=parent.expected_instances,
            visual_kind="object" if landmark.kind == "whole_object" else "part",
        )
        operands.append(
            _operand(semantic_operand.role, entity, operand_id=semantic_operand.operand_id)
        )
    auxiliary = [asdict(entity) for entity in semantic_plan.auxiliary_entities]
    return (
        semantic_plan.relation,
        operands,
        auxiliary,
        list(semantic_plan.warnings),
        semantic_plan.planner_version,
    )


def build_measurement_plan(
    question: dict[str, Any], semantic_plan: Any | None = None
) -> MeasurementPlan:
    qa_id = str(question.get("qa_id") if question.get("qa_id") is not None else "unknown")
    raw_question = str(question.get("raw_question") or question.get("question") or "")
    auxiliary_entities: list[dict[str, Any]] = []
    planner_version = "legacy-rule-v1"
    if semantic_plan is not None:
        relation, operands, auxiliary_entities, warnings, planner_version = (
            _operands_from_semantic_plan(semantic_plan)
        )
    else:
        entities = [canonicalize_entity(value) for value in question.get("target_entities") or []]
        relation = str(question.get("relation_type") or "attribute")
        roles: Iterable[str]
        if relation in {"from_to", "between"} and len(entities) >= 2:
            roles = ["from", "to", *["target"] * max(0, len(entities) - 2)]
        else:
            roles = ["target"] * len(entities)
        operands = [_operand(role, entity) for role, entity in zip(roles, entities)]
        relation, operands, warnings = _specialize_relational_operands(question, operands)

        referenced_ids = {
            operand.landmark.selector.get("reference_entity_id")
            for operand in operands
            if operand.landmark.selector
            and operand.landmark.selector.get("reference_entity_id")
        }
        for reference_id in sorted(referenced_ids):
            auxiliary_entities.append(
                {
                    "entity_id": reference_id,
                    "category": reference_id,
                    "referring_expression": reference_id,
                    "tracking_prompts": [reference_id],
                    "instance_selector": None,
                    "expected_instances": 1,
                    "role": "selector_reference",
                    "tracking_requirement": "optional",
                }
            )

    measurement_kind = _measurement_kind(question)
    if relation in {"distance_between", "vertical_distance"}:
        measurement_kind = "distance"

    axis = _axis(question)
    if axis is None:
        operand_axes = {
            operand.landmark.axis for operand in operands if operand.landmark.axis is not None
        }
        if len(operand_axes) == 1:
            axis = next(iter(operand_axes))

    if not operands:
        warnings.append("No measurement operand was resolved from target_entities.")

    return MeasurementPlan(
        plan_id=f"{question.get('video_id') or 'video'}:{qa_id}",
        qa_id=qa_id,
        video_id=str(question.get("video_id")) if question.get("video_id") is not None else None,
        raw_question=raw_question,
        quantity_family=str(question.get("quantity_family") or "unknown"),
        quantity_subtype=str(question.get("quantity_subtype") or "unknown"),
        measurement_kind=measurement_kind,
        output_unit=question.get("output_unit"),
        relation=relation,
        reduction=_reduction(question),
        axis=axis,
        temporal=dict(question.get("temporal") or {}),
        operands=operands,
        calibration=_calibration_route(question),
        auxiliary_entities=auxiliary_entities,
        planner_version=planner_version,
        warnings=warnings,
    )


def build_measurement_plans(
    group: dict[str, Any], semantic_planner: Any | None = None
) -> list[MeasurementPlan]:
    plans = []
    for question in group.get("questions", []):
        enriched = dict(question)
        enriched.setdefault("video_id", group.get("video_id"))
        enriched.setdefault("video_type", group.get("video_type"))
        enriched.setdefault("fps", group.get("fps"))
        if semantic_planner is None:
            plans.append(build_measurement_plan(enriched))
            continue
        try:
            semantic_plan = semantic_planner.plan(enriched)
            plans.append(build_measurement_plan(enriched, semantic_plan=semantic_plan))
        except Exception as exc:
            plan = build_measurement_plan(enriched)
            plan.warnings.append(
                f"Semantic planner failed; used legacy fallback: {type(exc).__name__}: {exc}"
            )
            plan.planner_version = "legacy-rule-v1-after-semantic-failure"
            plans.append(plan)
    return plans
