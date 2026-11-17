"""Build editable, target-free DEV32 character and scene asset candidates."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping


DEV32_ASSET_SCHEMA_VERSION = "bernini_dev32_asset_candidates_v1"
DEV32_REFERENCE_SKELETON_SCHEMA_VERSION = 1
_DEV32_SCHEMA_VERSION = "bernini_continuous_planner_dev32_v1"
_CHARACTER_MARKERS = frozenset(
    {
        "baker",
        "bearded",
        "boy",
        "camper",
        "chef",
        "child",
        "cyclist",
        "dancer",
        "dog",
        "engineer",
        "fisherman",
        "florist",
        "gardener",
        "girl",
        "hiker",
        "identity",
        "librarian",
        "man",
        "mechanic",
        "musician",
        "painter",
        "potter",
        "rider",
        "sailor",
        "scientist",
        "student",
        "traveler",
        "woman",
    }
)
_SCENE_MARKERS = frozenset(
    {
        "attic",
        "backstage",
        "backyard",
        "balcony",
        "bakery",
        "beach",
        "bedroom",
        "boat",
        "campsite",
        "classroom",
        "counter",
        "deck",
        "depot",
        "desk",
        "door",
        "doorway",
        "floor",
        "gallery",
        "garage",
        "gate",
        "garden",
        "hallway",
        "home",
        "house",
        "interior",
        "kitchen",
        "laboratory",
        "library",
        "lookout",
        "path",
        "pier",
        "plaza",
        "room",
        "sea",
        "shed",
        "shelter",
        "shop",
        "spring",
        "stage",
        "stall",
        "stream",
        "studio",
        "table",
        "tap",
        "theater",
        "town",
        "tunnel",
        "wall",
        "water",
        "workshop",
        "yard",
    }
)
_FRAMING_WORDS = frozenset(
    {
        "at",
        "across",
        "along",
        "close",
        "closer",
        "closeup",
        "closed",
        "dry",
        "empty",
        "front",
        "fullbody",
        "inside",
        "into",
        "it",
        "later",
        "long",
        "medium",
        "night",
        "on",
        "open",
        "outside",
        "overhead",
        "same",
        "shot",
        "side",
        "static",
        "the",
        "through",
        "toward",
        "towards",
        "tracking",
        "view",
        "wide",
    }
)
_GENERIC_SCENE_FIXTURES = frozenset(
    {"counter", "desk", "door", "doorway", "floor", "gate", "path", "table", "wall"}
)
_SCENE_HEADS = _SCENE_MARKERS | _GENERIC_SCENE_FIXTURES
_SCENE_HEAD_PATTERN = "|".join(
    sorted((re.escape(marker) for marker in _SCENE_HEADS), key=len, reverse=True)
)
_SETTING_NOUN_PHRASE = rf"(?P<setting>(?:[a-z]+\s+){{0,3}}(?:{_SCENE_HEAD_PATTERN}))\b"
_SPATIAL_LOCATION_PHRASE = re.compile(
    rf"\b(?:across|along|at|beside|inside|into|on|onto|outside|through|"
    rf"toward|towards)\s+(?:an?\s+|the\s+|same\s+)?{_SETTING_NOUN_PHRASE}",
    flags=re.IGNORECASE,
)
_ACTION_LOCATION_PHRASE = re.compile(
    rf"\b(?:approaches?|enters?|exits?|reaches?)\s+"
    rf"(?:an?\s+|the\s+|same\s+)?{_SETTING_NOUN_PHRASE}",
    flags=re.IGNORECASE,
)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def load_dev32_asset_source(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load the current DEV32 text denominator without reading any image."""

    source = Path(path).resolve(strict=True)
    manifest = _mapping(json.loads(source.read_text(encoding="utf-8")), "manifest")
    if manifest.get("schema_version") != _DEV32_SCHEMA_VERSION:
        raise ValueError(f"manifest.schema_version must equal {_DEV32_SCHEMA_VERSION}")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest.cases must be a non-empty list")
    return source, manifest


def _words(value: str) -> set[str]:
    return set(re.findall(r"[a-z]+", value.lower()))


def _character_fact(case: Mapping[str, Any]) -> tuple[str, str]:
    continuity = _mapping(case.get("continuity"), "case.continuity")
    preserve = continuity.get("preserve")
    if not isinstance(preserve, list) or not preserve:
        raise ValueError(f"case {case.get('case_id')!r} has no preserve facts")
    candidates = [
        (index, _text(fact, f"case.continuity.preserve[{index}]"))
        for index, fact in enumerate(preserve)
        if _words(_text(fact, f"case.continuity.preserve[{index}]"))
        & _CHARACTER_MARKERS
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"case {case.get('case_id')!r} must have exactly one explicit "
            f"character preserve fact; found {len(candidates)}"
        )
    index, fact = candidates[0]
    return fact, f"continuity.preserve[{index}]"


def _clean_scene_prefix(prefix: str) -> str | None:
    normalized = prefix.lower().replace("-", "")
    tokens = re.findall(r"[a-z]+", normalized)
    kept = [token for token in tokens if token not in _FRAMING_WORDS]
    if not kept or kept[-1] not in _SCENE_HEADS:
        return None
    return " ".join(kept)


def _location_scene_anchor(prompt: str) -> tuple[str, str] | None:
    for pattern in (_SPATIAL_LOCATION_PHRASE, _ACTION_LOCATION_PHRASE):
        for match in pattern.finditer(prompt):
            phrase = match.group("setting")
            anchor = _clean_scene_prefix(phrase)
            if anchor is not None:
                return anchor, match.group(0)
    return None


def _shot_scene_anchors(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract only a setting substring from every shot, including the terminal shot."""

    anchors: list[dict[str, Any]] = []
    shots = case.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError(f"case {case.get('case_id')!r} has no shots")
    for index, raw_shot in enumerate(shots):
        shot = _mapping(raw_shot, f"case.shots[{index}]")
        shot_id = _text(shot.get("shot_id"), f"case.shots[{index}].shot_id")
        prompt = _text(shot.get("prompt"), f"case.shots[{index}].prompt")
        prefix, separator, _remainder = prompt.partition(":")
        anchor = _clean_scene_prefix(prefix) if separator else None
        if anchor is not None:
            source_text = prefix
            extraction_method = "explicit_pre_colon_setting_anchor"
        else:
            location = _location_scene_anchor(prompt)
            if location is None:
                continue
            anchor, source_text = location
            extraction_method = "explicit_location_phrase_setting_anchor"
        if anchor is None:
            continue
        anchors.append(
            {
                "anchor": anchor,
                "shot_ids": [shot_id],
                "source": f"shots[{index}].prompt setting substring",
                "source_text": source_text,
                "extraction_method": extraction_method,
            }
        )
    return anchors


def _stable_scene_anchors(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    continuity = _mapping(case.get("continuity"), "case.continuity")
    preserve = continuity.get("preserve")
    if not isinstance(preserve, list):
        raise ValueError("case.continuity.preserve must be a list")
    anchors: list[dict[str, Any]] = []
    for index, raw_fact in enumerate(preserve):
        fact = _text(raw_fact, f"case.continuity.preserve[{index}]")
        if _clean_scene_prefix(fact) is not None:
            anchors.append(
                {
                    "anchor": fact,
                    "shot_ids": [],
                    "source": f"continuity.preserve[{index}]",
                    "source_text": fact,
                    "extraction_method": "stable_preserve_scene_anchor",
                }
            )
    return anchors


def _same_scene(left: str, right: str) -> bool:
    left_words = _words(left)
    right_words = _words(right)
    if left_words <= right_words or right_words <= left_words:
        return True
    place_roots = _SCENE_MARKERS - _GENERIC_SCENE_FIXTURES
    return bool(left_words & right_words & place_roots)


def _merge_scene_anchors(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for item in anchors:
        existing = next(
            (
                candidate
                for candidate in merged
                if _same_scene(candidate["anchor"], item["anchor"])
            ),
            None,
        )
        if existing is None:
            copied = {
                **item,
                "shot_ids": list(item["shot_ids"]),
                "evidence": [
                    {
                        "source": item["source"],
                        "text": item["source_text"],
                        "extraction_method": item["extraction_method"],
                    }
                ],
            }
            for field in ("source", "source_text", "extraction_method"):
                copied.pop(field)
            merged.append(copied)
            continue
        if len(_words(item["anchor"])) > len(_words(existing["anchor"])):
            existing["anchor"] = item["anchor"]
        for shot_id in item["shot_ids"]:
            if shot_id not in existing["shot_ids"]:
                existing["shot_ids"].append(shot_id)
        existing["evidence"].append(
            {
                "source": item["source"],
                "text": item["source_text"],
                "extraction_method": item["extraction_method"],
            }
        )
    return merged


def _bind_unresolved_scene_shots(
    case: Mapping[str, Any],
    scenes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    all_shot_ids = [
        _text(shot["shot_id"], "shot.shot_id") for shot in case["shots"]
    ]
    assigned = {shot_id for scene in scenes for shot_id in scene["shot_ids"]}
    unresolved = [shot_id for shot_id in all_shot_ids if shot_id not in assigned]
    stable = [
        scene
        for scene in scenes
        if any(
            evidence["extraction_method"] == "stable_preserve_scene_anchor"
            for evidence in scene["evidence"]
        )
    ]
    change_facts = case["continuity"]["change"]
    location_changes = any(
        "location changes" in _text(fact, "continuity.change").casefold()
        for fact in change_facts
    )
    destination = None
    binding_method = None
    if len(stable) == 1:
        destination = stable[0]
        binding_method = "stable_preserve_setting_applied_to_unresolved_shot"
    elif len([scene for scene in scenes if scene["shot_ids"]]) == 1 and not location_changes:
        destination = next(scene for scene in scenes if scene["shot_ids"])
        binding_method = "single_story_setting_applied_to_unresolved_shot"
    if destination is None:
        return scenes, unresolved
    for shot_id in unresolved:
        destination["shot_ids"].append(shot_id)
        destination["evidence"].append(
            {
                "source": "continuity resolution",
                "text": shot_id,
                "extraction_method": binding_method,
            }
        )
    return scenes, []


def _scene_candidates(
    case: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    anchors = _shot_scene_anchors(case) + _stable_scene_anchors(case)
    if not anchors:
        raise ValueError(
            f"case {case.get('case_id')!r} has no explicit neutral scene anchor; "
            "manual source annotation is required"
        )
    scenes = _merge_scene_anchors(anchors)
    scenes, unresolved = _bind_unresolved_scene_shots(case, scenes)
    return [scene for scene in scenes if scene["shot_ids"]], unresolved


def _case_assets(case: Mapping[str, Any]) -> dict[str, Any]:
    story_id = _text(case.get("case_id"), "case.case_id")
    _text(case.get("story_prompt"), f"{story_id}.story_prompt")
    continuity = _mapping(case.get("continuity"), f"{story_id}.continuity")
    changes = continuity.get("change")
    if not isinstance(changes, list) or not changes:
        raise ValueError(f"case {story_id!r} must declare change facts")
    excluded_changes = [
        _text(change, f"{story_id}.continuity.change") for change in changes
    ]
    shots = case.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError(f"case {story_id!r} must declare shots")

    character_fact, character_source = _character_fact(case)
    assets: list[dict[str, Any]] = [
        {
            "asset_id": f"{story_id}__character__01",
            "role": "character",
            "prompt": (
                f"Neutral full-body character reference for {character_fact}. "
                "Plain background, neutral pose, no action, no event-specific prop, "
                "and no outcome cue."
            ),
            "evidence": [
                {
                    "source": character_source,
                    "text": character_fact,
                    "extraction_method": "explicit_character_preserve_fact",
                }
            ],
            "manual_review_required": True,
            "review_note": (
                "Candidate copied from one explicit preserve fact; verify identity and "
                "appearance before image generation."
            ),
        }
    ]

    scenes, unresolved_scene_shots = _scene_candidates(case)
    for index, scene in enumerate(scenes, start=1):
        assets.append(
            {
                "asset_id": f"{story_id}__scene__{index:02d}",
                "role": "scene",
                "shot_ids": scene["shot_ids"],
                "prompt": (
                    f"Neutral empty setting reference for {scene['anchor']}. "
                    "No people, no active event, no state-changing action, and no "
                    "outcome cue."
                ),
                "evidence": scene["evidence"],
                "manual_review_required": True,
                "review_note": (
                    "Candidate contains only a setting anchor; verify the setting and "
                    "shot bindings before image generation."
                ),
            }
        )

    all_shot_ids = [_text(shot["shot_id"], "shot.shot_id") for shot in shots]
    terminal_source = f"shots[{len(shots) - 1}].prompt setting substring"
    terminal_setting_anchor_used = any(
        evidence["source"] == terminal_source
        for scene in scenes
        for evidence in scene["evidence"]
    )
    return {
        "story_id": story_id,
        "assets": assets,
        "excluded_from_asset_prompts": {
            "change_facts": excluded_changes,
            "terminal_shot_id": all_shot_ids[-1],
            "terminal_setting_anchor_used": terminal_setting_anchor_used,
        },
        "scene_shots_without_explicit_candidate": unresolved_scene_shots,
    }


def build_dev32_asset_candidates(
    manifest: Mapping[str, Any],
    *,
    source_manifest: str,
) -> dict[str, Any]:
    """Build candidate prompts while keeping semantic acceptance manual."""

    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest.cases must be a non-empty list")
    stories = [_case_assets(_mapping(case, "manifest.cases")) for case in cases]
    scene_prompt_audit = [
        {
            "story_id": story["story_id"],
            "asset_id": asset["asset_id"],
            "shot_ids": asset["shot_ids"],
            "prompt": asset["prompt"],
            "source_substrings": [
                evidence["text"]
                for evidence in asset["evidence"]
                if evidence["source"] != "continuity resolution"
            ],
            "manual_review_required": asset["manual_review_required"],
        }
        for story in stories
        for asset in story["assets"]
        if asset["role"] == "scene"
    ]
    return {
        "schema_version": DEV32_ASSET_SCHEMA_VERSION,
        "source_manifest": source_manifest,
        "status": "candidate_prompts_manual_review_required",
        "claim_boundary": (
            "Deterministic text candidates only; no image was generated and no "
            "candidate is claimed semantically correct."
        ),
        "generation_input_boundary": (
            "References must be fixed before video generation and must not use a "
            "target video, target frame, or ground-truth target-ViT."
        ),
        "selection_policy": {
            "character": "one explicit identity or appearance preserve fact",
            "scene": (
                "pure setting substrings from every shot plus explicit stable preserve "
                "settings; single stable settings may cover unresolved shots"
            ),
            "excluded": (
                "all change facts and all terminal-shot action, state, and outcome text; "
                "a pure terminal setting substring may be used"
            ),
            "ambiguity": "raise an error or emit an uncovered-shot item for review",
        },
        "stories": stories,
        "scene_prompt_audit": scene_prompt_audit,
    }


def build_dev32_reference_skeleton(
    candidates: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a runtime-shaped manifest with deliberately unfilled image paths."""

    stories = candidates.get("stories")
    if not isinstance(stories, list):
        raise ValueError("candidates.stories must be a list")
    return {
        "schema_version": DEV32_REFERENCE_SKELETON_SCHEMA_VERSION,
        "reference_policy": "fixed_non_target",
        "stories": [
            {
                "story_id": story["story_id"],
                "references": [
                    {
                        "candidate_asset_id": asset["asset_id"],
                        "path": None,
                        "role": asset["role"],
                        **(
                            {"shot_ids": asset["shot_ids"]}
                            if "shot_ids" in asset
                            else {}
                        ),
                    }
                    for asset in story["assets"]
                ],
            }
            for story in stories
        ],
    }


__all__ = [
    "DEV32_ASSET_SCHEMA_VERSION",
    "DEV32_REFERENCE_SKELETON_SCHEMA_VERSION",
    "build_dev32_asset_candidates",
    "build_dev32_reference_skeleton",
    "load_dev32_asset_source",
]
