"""Shared multi-shot inference helpers owned by :mod:`mspc.inference`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from mspc.config import ExperimentConfig
from mspc.data.multishot import (
    MULTISHOT_CSV_SCHEMA,
    SOURCE_FRAME_INDICES_COLUMN,
    SOURCE_FRAME_INDICES_BY_SHOT_COLUMN,
    SOURCE_SHOT_SEGMENTS_COLUMN,
    SOURCE_VIDEO_PATHS_COLUMN,
    SUPPORTED_PROMPT_FORMATS,
    build_prompts_from_fields,
    caption_fields,
    is_explicit_mapping_schema,
    is_multisource_mapping_schema,
    parse_multisource_source_shot_segments,
    parse_source_frame_indices,
    parse_source_frame_indices_by_shot,
    parse_source_shot_segments,
    parse_source_video_paths,
    parse_shot_groups,
    read_video_metadata,
    vae_alignment_policy_for_schema,
)
from mspc.data.frame_planner import FRAME_POLICY_FPS_AWARE_TARGET_FPS, FpsAwareTargetFpsPlanner
from mspc.data.shot2story_raw_source import (
    Shot2StoryRawPromptSource,
    Shot2StoryRawPromptSourceConfig,
)


FRAME_POLICY_TARGET_FPS = FRAME_POLICY_FPS_AWARE_TARGET_FPS
DATASET_TYPE_MULTISHOT_CSV = "multishot_csv"
_SHOT2STORY_PROMPT_SOURCE_CACHE: dict[
    tuple[str, str, str],
    Shot2StoryRawPromptSource,
] = {}

SOURCE_CONTEXT_FIELDS = (
    "story_id",
    "generator",
    "conditioning_policy",
    "story_prompt",
    "entity_descriptions",
    "shot_ids",
    "shot_prompts",
    "source_case_id",
    "source_manifest_case_id",
    "source_dataset",
    "diagnostic_anchors",
    "consistency_axis",
    "anchor_type",
    "planned_transition",
    "persistent_anchors",
    "expected_cross_shot_links",
    "readiness_status",
    "claim_status",
)
LIST_CONTEXT_FIELDS = {
    "entity_descriptions",
    "shot_ids",
    "shot_prompts",
    "diagnostic_anchors",
    "consistency_axis",
    "planned_transition",
    "persistent_anchors",
    "expected_cross_shot_links",
}


def _resolve_dataset_for_inference(
    config: ExperimentConfig,
) -> tuple[Mapping[str, Any], str]:
    dataset = config.resolved["dataset"]
    dataset_type = dataset["type"]
    if dataset_type == DATASET_TYPE_MULTISHOT_CSV:
        return dataset, dataset_type
    raise ValueError(f"CSV inference requires multishot_csv, got {dataset_type!r}; block training uses an explicit storyboard panel")


def _load_shot2story_prompt_source(
    value: Mapping[str, Any],
) -> Shot2StoryRawPromptSource:
    config = Shot2StoryRawPromptSourceConfig.from_mapping(dict(value))
    key = (
        config.kind,
        config.annotation_json,
        config.selected_shot_indices_column,
    )
    source = _SHOT2STORY_PROMPT_SOURCE_CACHE.get(key)
    if source is None:
        source = Shot2StoryRawPromptSource.load(config)
        _SHOT2STORY_PROMPT_SOURCE_CACHE[key] = source
    return source


def _resolve_task_for_inference(config: ExperimentConfig) -> Mapping[str, Any]:
    task = config.resolved["task"]
    if task is None:
        raise ValueError("config.task is required for non-cached inference")
    return task


def build_inference_dataset_config(
    config: ExperimentConfig,
    *,
    require_case_id_column: bool,
) -> dict[str, Any]:
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    dataset, dataset_type = _resolve_dataset_for_inference(config)
    task = _resolve_task_for_inference(config)
    dataset_cfg = {
        "type": dataset_type,
        "schema": dataset["schema"],
        "frame_policy": task["frame_policy"],
        "video_column": dataset["video_column"],
        "caption_column": dataset["caption_column"],
        "shot_groups_column": dataset["shot_groups_column"],
        "max_rgb_frames": task["max_rgb_frames"],
        "target_fps": task["target_fps"],
        "vae_temporal_stride": task["vae_temporal_stride"],
    }
    if dataset_cfg["frame_policy"] != FRAME_POLICY_TARGET_FPS:
        raise ValueError(f"unsupported frame policy: {dataset_cfg['frame_policy']}")
    if dataset["prompt_format"] not in SUPPORTED_PROMPT_FORMATS:
        raise ValueError(f"unsupported prompt format: {dataset['prompt_format']}")
    dataset_cfg["prompt_format"] = dataset["prompt_format"]
    prompt_source = dataset.get("shot2story_prompt_source")
    if prompt_source is None:
        if dataset_cfg["caption_column"] is None:
            raise ValueError(
                "dataset.caption_column is required without dataset.shot2story_prompt_source"
            )
    else:
        if dataset_cfg["caption_column"] is not None:
            raise ValueError(
                "dataset.caption_column must be null with dataset.shot2story_prompt_source"
            )
        dataset_cfg["shot2story_prompt_source"] = dict(prompt_source)
    case_id_column = dataset.get("case_id_column")
    if require_case_id_column or case_id_column is not None:
        if case_id_column is None:
            raise ValueError("config dataset.case_id_column is required for this backend")
        dataset_cfg["case_id_column"] = case_id_column
    return dataset_cfg


def build_inference_frame_planner(
    dataset_cfg: dict[str, Any],
    *,
    inference_fps: int | float,
) -> FpsAwareTargetFpsPlanner:
    if inference_fps != dataset_cfg["target_fps"]:
        raise ValueError("config inference.fps must equal config dataset.target_fps")
    return FpsAwareTargetFpsPlanner(
        max_rgb_frames=dataset_cfg["max_rgb_frames"],
        target_fps=dataset_cfg["target_fps"],
        vae_temporal_stride=dataset_cfg["vae_temporal_stride"],
    )


def infer_case_id(
    row: Any,
    *,
    row_index: int,
    dataset_cfg: dict[str, Any],
    source_label: str,
) -> str:
    case_id_column = dataset_cfg.get("case_id_column")
    if case_id_column is None:
        return str(row_index)
    case_id = str(row[case_id_column]).strip()
    if not case_id:
        raise ValueError(f"{source_label}:{row_index}: empty case id")
    return case_id


def row_mapping(row: Any | None) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return row
    if hasattr(row, "to_dict"):
        mapped = row.to_dict()
        return mapped if isinstance(mapped, dict) else {}
    return {}


def checkpoint_step_from_path(path: str | Path) -> int | None:
    """Parse the training step out of a ``step-N.safetensors`` checkpoint name.

    Reuses the sidecar's own filename pattern
    (:data:`mspc.training.wan22_smoke_gate.STEP_CHECKPOINT_PATTERN`) so a
    checkpoint's step is derived the same way everywhere. Returns ``None`` for a
    path that is not in that convention -- a hand-provided pretrained-weights file
    has no training step to report.
    """
    from mspc.training.wan22_smoke_gate import STEP_CHECKPOINT_PATTERN

    match = STEP_CHECKPOINT_PATTERN.fullmatch(Path(path).name)
    return int(match.group("step")) if match else None


def is_missing_context_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(value != value)
    except Exception:
        return False


def normalize_context_value(key: str, value: Any) -> Any:
    if key not in LIST_CONTEXT_FIELDS:
        return value
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    return value


def build_multishot_case_inputs(
    row: Any,
    frame_planner: Any,
    *,
    video_column: str,
    caption_column: str | None,
    shot_groups_column: str,
    schema: str = MULTISHOT_CSV_SCHEMA,
    row_label: str,
    prompt_format: str,
    shot2story_prompt_source: Mapping[str, Any] | None = None,
    read_video_metadata_fn: Any = read_video_metadata,
) -> dict[str, Any]:
    video_path = str(row[video_column])
    shot_groups = parse_shot_groups(row[shot_groups_column], row_label)
    source_shot_segments: list[list[int]] | None = None
    source_video_paths: list[str] | None = None
    if is_explicit_mapping_schema(schema):
        video_metadata = read_video_metadata_fn(video_path)
        source_shot_segments = parse_source_shot_segments(
            row[SOURCE_SHOT_SEGMENTS_COLUMN], video_path
        )
        frame_plan = frame_planner.build_explicit_frame_plan(
            source_shot_segments=source_shot_segments,
            train_shot_groups=shot_groups,
            frame_indices=parse_source_frame_indices(
                row[SOURCE_FRAME_INDICES_COLUMN], video_path
            ),
            data_path=video_path,
            source_fps=video_metadata["source_fps"],
            source_num_frames=video_metadata["source_num_frames"],
            vae_alignment_policy=vae_alignment_policy_for_schema(schema),
        )
    elif is_multisource_mapping_schema(schema):
        if shot2story_prompt_source is not None:
            raise ValueError(
                f"{row_label}: shot2story_prompt_source is not supported with "
                "multi-source explicit mappings"
            )
        source_shot_segments = parse_multisource_source_shot_segments(
            row[SOURCE_SHOT_SEGMENTS_COLUMN], video_path
        )
        source_video_paths = parse_source_video_paths(
            row[SOURCE_VIDEO_PATHS_COLUMN], video_path
        )
        frame_indices_by_shot = parse_source_frame_indices_by_shot(
            row[SOURCE_FRAME_INDICES_BY_SHOT_COLUMN], video_path
        )
        if not (
            len(source_shot_segments)
            == len(shot_groups)
            == len(source_video_paths)
            == len(frame_indices_by_shot)
        ):
            raise ValueError(f"{row_label}: source/train shot counts do not match")
        for source_path in source_video_paths:
            path = Path(source_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size <= 0:
                raise ValueError(f"{row_label}: empty source video: {path}")
        source_metadata_by_shot = [
            read_video_metadata_fn(path) for path in source_video_paths
        ]
        frame_plan = frame_planner.build_multisource_explicit_frame_plan(
            source_video_paths=source_video_paths,
            source_shot_segments=source_shot_segments,
            train_shot_groups=shot_groups,
            frame_indices_by_shot=frame_indices_by_shot,
            data_path=video_path,
            source_fps_by_shot=[
                metadata["source_fps"] for metadata in source_metadata_by_shot
            ],
            source_num_frames_by_shot=[
                metadata["source_num_frames"] for metadata in source_metadata_by_shot
            ],
        )
    else:
        video_metadata = read_video_metadata_fn(video_path)
        frame_plan = frame_planner.build_frame_plan(
            shot_groups,
            video_path,
            source_fps=video_metadata["source_fps"],
            source_num_frames=video_metadata["source_num_frames"],
        )
    if shot2story_prompt_source is None:
        if caption_column is None:
            raise ValueError(
                f"{row_label}: caption_column is required without shot2story_prompt_source"
            )
        story_prompt, shot_prompts = caption_fields(
            str(row[caption_column]),
            len(frame_plan["shot_groups"]),
        )
        prompts = build_prompts_from_fields(shot_prompts, prompt_format=prompt_format)
    else:
        if caption_column is not None:
            raise ValueError(
                f"{row_label}: caption_column must be null with shot2story_prompt_source"
            )
        if source_shot_segments is None:
            raise ValueError(
                f"{row_label}: shot2story_prompt_source requires explicit source_shot_segments"
            )
        prompt_source = _load_shot2story_prompt_source(shot2story_prompt_source)
        annotation, selected_indices = prompt_source.validate_row_binding(
            row_mapping(row),
            video_path=video_path,
            shot_count=len(frame_plan["shot_groups"]),
            source_shot_segments=source_shot_segments,
            label=row_label,
        )
        story_prompt = annotation.whole_caption
        shot_prompts = [annotation.captions[index] for index in selected_indices]
        prompts = build_prompts_from_fields(shot_prompts, prompt_format=prompt_format)
    result = {
        "video_path": video_path,
        "prompts": prompts,
        "story_prompt": story_prompt,
        "shot_prompts": shot_prompts,
        "target_fps": frame_plan["target_fps"],
        "max_rgb_frames": frame_plan["max_rgb_frames"],
        "vae_temporal_stride": frame_plan["vae_temporal_stride"],
        "shot_groups": frame_plan["shot_groups"],
        "latent_shot_groups": frame_plan["latent_shot_groups"],
        "train_rgb_frames": frame_plan["train_rgb_frames"],
        "train_latent_frames": frame_plan["train_latent_frames"],
        "source_shot_segments": source_shot_segments,
    }
    if "frame_indices" in frame_plan:
        result["source_frame_indices"] = frame_plan["frame_indices"]
    if source_video_paths is not None:
        result["source_video_paths"] = source_video_paths
    if "frame_indices_by_shot" in frame_plan:
        result["frame_indices_by_shot"] = frame_plan["frame_indices_by_shot"]
    if "vae_alignment_policy" in frame_plan:
        result["vae_alignment_policy"] = frame_plan["vae_alignment_policy"]
    return result


def generated_eval_manifest_row(
    *,
    case_id: str,
    output_path: str | Path,
    generation_paradigm: str,
    case_inputs: dict[str, Any],
    source_row: Any | None = None,
) -> dict[str, Any]:
    output = str(output_path)
    row = {
        "case_id": case_id,
        "video_path": output,
        "shot_paths": [output],
        "shot_count": len(case_inputs["latent_shot_groups"]),
        "generation_paradigm": generation_paradigm,
        "story_prompt": case_inputs.get("story_prompt", ""),
        "shot_prompts": case_inputs.get("shot_prompts", []),
        "target_fps": case_inputs["target_fps"],
        "max_rgb_frames": case_inputs["max_rgb_frames"],
        "vae_temporal_stride": case_inputs["vae_temporal_stride"],
        "train_rgb_frames": case_inputs["train_rgb_frames"],
        "train_latent_frames": case_inputs["train_latent_frames"],
        "shot_groups": case_inputs["shot_groups"],
        "latent_shot_groups": case_inputs["latent_shot_groups"],
    }
    if "source_frame_indices" in case_inputs:
        row["source_frame_indices"] = case_inputs["source_frame_indices"]
    if "source_video_paths" in case_inputs:
        row["source_video_paths"] = case_inputs["source_video_paths"]
    if "frame_indices_by_shot" in case_inputs:
        row["frame_indices_by_shot"] = case_inputs["frame_indices_by_shot"]
    if "vae_alignment_policy" in case_inputs:
        row["vae_alignment_policy"] = case_inputs["vae_alignment_policy"]
    source_shot_segments = case_inputs.get("source_shot_segments")
    if source_shot_segments is not None:
        row["source_shot_segments"] = source_shot_segments
    source = row_mapping(source_row)
    for key in SOURCE_CONTEXT_FIELDS:
        if key not in source or is_missing_context_value(source[key]):
            continue
        row[key] = normalize_context_value(key, source[key])
    return row
