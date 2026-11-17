"""Shared multi-shot dataset and manifest utilities."""

from __future__ import annotations

import ast
from pathlib import Path
import time
from typing import Any

import pandas as pd
import torch

from mspc.data.frame_planner import (
    FRAME_POLICY_FPS_AWARE_TARGET_FPS,
    FpsAwareTargetFpsPlanner,
    VAE_ALIGNMENT_POLICY_LEGACY_FLOOR,
    VAE_ALIGNMENT_POLICY_TAIL_PAD,
    validate_shot_groups,
)
from mspc.data.metadata import require_non_empty_string
from mspc.data.shot2story_raw_source import (
    Shot2StoryRawPromptSource,
    Shot2StoryRawPromptSourceConfig,
)
from mspc.data.strict_json import load_strict_json_file

PROMPT_FORMAT_CAPTION_JSON_V1 = "caption_json_v1"
PROMPT_FORMAT_STORY_NOW_V1 = "story_now_v1"
SUPPORTED_PROMPT_FORMATS = frozenset(
    {
        PROMPT_FORMAT_CAPTION_JSON_V1,
        PROMPT_FORMAT_STORY_NOW_V1,
    }
)
#: 出片清单里记录的逐镜文字拼法（:func:`build_prompt_from_fields`）。
RENDERED_PROMPT_CONTRACT = "{shot}"
CAPTION_PATH_COLUMN = "caption_path"
# Historical temporal-contract CSV used this name for a filesystem path, not Gemini text.
HISTORICAL_CAPTION_PATH_COLUMN = "gemini_caption"
MULTISHOT_CSV_SCHEMA = "multishot_csv_v1"
MULTISHOT_CSV_EXPLICIT_MAPPING_SCHEMA = "multishot_csv_explicit_mapping_v2"
MULTISHOT_CSV_TAIL_PAD_MAPPING_SCHEMA = "multishot_csv_explicit_mapping_v3_tail_padding"
# v6：源镜头区间经 shot_boundary_revision 按检测切点修过首尾，不再固定丢镜尾一帧。
MULTISHOT_CSV_MULTISOURCE_MAPPING_SCHEMA = "multishot_csv_multisource_mapping_v6_boundary_revised"
EXPLICIT_MAPPING_SCHEMAS = frozenset(
    {
        MULTISHOT_CSV_EXPLICIT_MAPPING_SCHEMA,
        MULTISHOT_CSV_TAIL_PAD_MAPPING_SCHEMA,
    }
)
SCENE_MAPPING_SCHEMA = "scene_window_v1"
MULTISOURCE_MAPPING_SCHEMAS = frozenset({MULTISHOT_CSV_MULTISOURCE_MAPPING_SCHEMA})
SOURCE_SHOT_SEGMENTS_COLUMN = "source_shot_segments"
SOURCE_FRAME_INDICES_COLUMN = "source_frame_indices"
SOURCE_VIDEO_PATHS_COLUMN = "source_video_paths"
SOURCE_FRAME_INDICES_BY_SHOT_COLUMN = "source_frame_indices_by_shot"
VIDEO_OUTPUT_MODE_NORMALIZED_FLOAT = "normalized_float"
VIDEO_OUTPUT_MODE_UINT8 = "uint8"
SUPPORTED_VIDEO_OUTPUT_MODES = frozenset(
    {
        VIDEO_OUTPUT_MODE_NORMALIZED_FLOAT,
        VIDEO_OUTPUT_MODE_UINT8,
    }
)


def is_explicit_mapping_schema(schema: str) -> bool:
    return schema in EXPLICIT_MAPPING_SCHEMAS


def is_multisource_mapping_schema(schema: str) -> bool:
    return schema in MULTISOURCE_MAPPING_SCHEMAS


def is_deterministic_mapping_schema(schema: str) -> bool:
    return schema == SCENE_MAPPING_SCHEMA or is_explicit_mapping_schema(schema) or is_multisource_mapping_schema(schema)


def vae_alignment_policy_for_schema(schema: str) -> str:
    if schema == MULTISHOT_CSV_EXPLICIT_MAPPING_SCHEMA:
        return VAE_ALIGNMENT_POLICY_LEGACY_FLOOR
    if schema == MULTISHOT_CSV_TAIL_PAD_MAPPING_SCHEMA:
        return VAE_ALIGNMENT_POLICY_TAIL_PAD
    raise ValueError(f"{schema}: not an explicit mapping schema")


def parse_shot_groups(value: Any, data_path: str) -> list[list[int]]:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    validate_shot_groups(parsed, data_path)
    return [list(group) for group in parsed]


def parse_source_shot_segments(value: Any, data_path: str) -> list[list[int]]:
    from mspc.data.frame_planner import validate_source_shot_segments

    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    validate_source_shot_segments(parsed, data_path)
    return [list(segment) for segment in parsed]


def parse_multisource_source_shot_segments(value: Any, data_path: str) -> list[list[int]]:
    from mspc.data.frame_planner import validate_multisource_shot_segments

    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    validate_multisource_shot_segments(parsed, data_path)
    return [list(segment) for segment in parsed]


def parse_source_frame_indices(value: Any, data_path: str) -> list[int]:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or not parsed or any(type(index) is not int for index in parsed):
        raise ValueError(f"{data_path}: source_frame_indices must be a non-empty list of ints")
    if any(index < 0 for index in parsed):
        raise ValueError(f"{data_path}: source_frame_indices must be non-negative")
    return list(parsed)


def parse_source_video_paths(value: Any, data_path: str) -> list[str]:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"{data_path}: source_video_paths must be a non-empty list")
    paths: list[str] = []
    for index, path in enumerate(parsed):
        if not isinstance(path, str) or not path.strip() or path != path.strip():
            raise ValueError(f"{data_path}: invalid source_video_paths[{index}]")
        paths.append(path)
    return paths


def parse_source_frame_indices_by_shot(value: Any, data_path: str) -> list[list[int]]:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"{data_path}: source_frame_indices_by_shot must be a non-empty list")
    groups: list[list[int]] = []
    for shot_index, indices in enumerate(parsed):
        if not isinstance(indices, list) or not indices or any(type(index) is not int for index in indices):
            raise ValueError(
                f"{data_path}: source_frame_indices_by_shot[{shot_index}] must be a non-empty list of ints"
            )
        if any(index < 0 for index in indices):
            raise ValueError(
                f"{data_path}: source_frame_indices_by_shot[{shot_index}] must be non-negative"
            )
        groups.append(list(indices))
    return groups


def caption_fields(caption_path: str, shot_count: int) -> tuple[str, list[str]]:
    caption = load_strict_json_file(caption_path, label=f"caption {caption_path}")
    if not isinstance(caption, dict):
        raise ValueError(f"{caption_path}: caption JSON must be an object")
    expected_keys = {"global_caption", *(f"shot{index}" for index in range(shot_count))}
    missing_keys = sorted(expected_keys - set(caption))
    unknown_keys = sorted(set(caption) - expected_keys)
    if missing_keys:
        raise ValueError(f"{caption_path}: missing caption fields: {', '.join(missing_keys)}")
    if unknown_keys:
        raise ValueError(f"{caption_path}: unknown caption fields: {', '.join(unknown_keys)}")
    if "global_caption" not in caption:
        raise ValueError(f"{caption_path}: missing global_caption")
    global_caption = require_non_empty_string(caption["global_caption"], f"{caption_path}: global_caption")
    if global_caption != global_caption.strip():
        raise ValueError(f"{caption_path}: global_caption must not contain surrounding whitespace")
    shot_prompts: list[str] = []
    for shot_index in range(shot_count):
        shot_key = f"shot{shot_index}"
        if shot_key not in caption:
            raise ValueError(f"{caption_path}: missing {shot_key}")
        shot_prompt = require_non_empty_string(caption[shot_key], f"{caption_path}: {shot_key}")
        if shot_prompt != shot_prompt.strip():
            raise ValueError(f"{caption_path}: {shot_key} must not contain surrounding whitespace")
        shot_prompts.append(shot_prompt)
    return global_caption, shot_prompts


def validate_prompt_format(prompt_format: str) -> str:
    if prompt_format not in SUPPORTED_PROMPT_FORMATS:
        raise ValueError(
            f"unsupported prompt_format={prompt_format}; expected one of {sorted(SUPPORTED_PROMPT_FORMATS)}"
        )
    return prompt_format


def compile_selected_shot_global_caption(shot_captions: list[str]) -> str:
    """MSM ``global_caption`` from the selected shots only; never a whole-story leftover."""

    if not isinstance(shot_captions, list) or not shot_captions:
        raise ValueError("shot_captions must be a non-empty list")
    parts: list[str] = []
    for index, text in enumerate(shot_captions):
        if not isinstance(text, str) or not text or text != text.strip():
            raise ValueError(f"shot_captions[{index}] must be a non-empty trimmed string")
        parts.append(text)
    return " ".join(parts)


def msm_caption_payload(shot_captions: list[str]) -> dict[str, str]:
    """Build the MSM caption object: ``global_caption`` plus ``shot0..shotN-1``."""

    global_caption = compile_selected_shot_global_caption(shot_captions)
    return {
        "global_caption": global_caption,
        **{f"shot{index}": text for index, text in enumerate(shot_captions)},
    }


def build_prompt_from_fields(shot_prompt: str, *, prompt_format: str) -> str:
    """一镜送进渲染器的文字：就是这一镜自己的 prompt，前面不再接任何故事段。

    渲染文字去掉 ``Story:`` 段。训练 caption、各评测集的 ``story_prompt``
    全文就是主体表（``Subject 1: …; Subject 2: ….``），去掉主体表后 ``Story:`` 什么也不剩，
    所以整段不写；主体描述只经按在场过滤的设定（``story_visual_setting.compose_now_text``）
    进本镜文字。训练与推理都经这一个函数。``story_prompt`` 字段本身仍留在数据里给判官与评分读。
    ``caption_json_v1`` / ``story_now_v1`` 两个格式名只剩「caption 来源」的含义，拼法相同。
    """

    validate_prompt_format(prompt_format)
    if not isinstance(shot_prompt, str) or not shot_prompt or shot_prompt != shot_prompt.strip():
        raise ValueError("shot_prompt must be a non-empty trimmed string")
    return shot_prompt


def build_prompts_from_fields(shot_prompts: list[str], *, prompt_format: str) -> list[str]:
    return [build_prompt_from_fields(shot_prompt, prompt_format=prompt_format) for shot_prompt in shot_prompts]


def build_prompts(
    caption_path: str,
    shot_count: int,
    *,
    prompt_format: str,
) -> list[str]:
    _, shot_prompts = caption_fields(caption_path, shot_count)
    return build_prompts_from_fields(shot_prompts, prompt_format=prompt_format)


def read_video_metadata(path: str) -> dict[str, int | float]:
    import decord

    reader = decord.VideoReader(path, ctx=decord.cpu(0), num_threads=1)
    source_num_frames = len(reader)
    if source_num_frames <= 0:
        raise ValueError(f"{path}: video has no frames")
    source_fps = reader.get_avg_fps()
    if source_fps <= 0:
        raise ValueError(f"{path}: video fps must be positive")
    first_frame = reader[0]
    source_height = int(first_frame.shape[0])
    source_width = int(first_frame.shape[1])
    return {
        "source_fps": float(source_fps),
        "source_num_frames": int(source_num_frames),
        "source_height": source_height,
        "source_width": source_width,
    }


def get_read_height_width(
    *,
    video_width: int,
    video_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int]:
    scale = max(target_width / video_width, target_height / video_height)
    return round(video_height * scale), round(video_width * scale)


def cover_crop_geometry(
    *, video_width: int, video_height: int, target_width: int, target_height: int,
) -> dict[str, Any]:
    """Expose the existing decoder resize + torchvision CenterCrop edge coordinates.

    Boxes use pixel edges, not pixel centers. Resize rounding can make x/y
    scales slightly different; CenterCrop rounds half offsets to nearest even.
    This describes the current full-frame path and does not consume active_box.
    """
    if any(type(v) is not int or v <= 0 for v in
           (video_width, video_height, target_width, target_height)):
        raise ValueError("source and target sizes must be positive integers")
    read_h, read_w = get_read_height_width(
        video_width=video_width, video_height=video_height,
        target_width=target_width, target_height=target_height,
    )
    left, top = round((read_w - target_width) / 2), round((read_h - target_height) / 2)
    sx, sy = read_w / video_width, read_h / video_height
    return {
        "source_size": [video_width, video_height], "resize_size": [read_w, read_h],
        "crop_box_resized": [left, top, left + target_width, top + target_height],
        "output_size": [target_width, target_height], "coordinate_system": "pixel_edges",
        "matrix": [[sx, 0, -left], [0, sy, -top], [0, 0, 1]],
        "visible_source_box": [left / sx, top / sy,
                               (left + target_width) / sx, (top + target_height) / sy],
    }


def read_video_frames(
    *,
    path: str,
    frame_indices: list[int],
    target_height: int,
    target_width: int,
    transform_centercrop: Any,
    video_height: int | None = None,
    video_width: int | None = None,
    num_threads: int = 0,
    output_mode: str = VIDEO_OUTPUT_MODE_NORMALIZED_FLOAT,
) -> torch.Tensor:
    import decord

    if type(num_threads) is not int or num_threads < 0:
        raise ValueError("num_threads must be a non-negative int")
    if output_mode not in SUPPORTED_VIDEO_OUTPUT_MODES:
        raise ValueError(
            f"unsupported video output_mode={output_mode!r}; expected one of "
            f"{sorted(SUPPORTED_VIDEO_OUTPUT_MODES)}"
        )
    if video_height is None or video_width is None:
        first_reader = decord.VideoReader(path, ctx=decord.cpu(0), num_threads=1)
        # Inspect a frame the caller actually requested. Some edit-list sources
        # decode that span correctly but fail when decord seeks to packet zero
        # (FX3 S01E3-6: frame 2496 is valid, frame 0 raises EOF).
        first_requested = first_reader[frame_indices[0]]
        video_height = int(first_requested.shape[0])
        video_width = int(first_requested.shape[1])
    read_h, read_w = get_read_height_width(
        video_width=video_width,
        video_height=video_height,
        target_width=target_width,
        target_height=target_height,
    )
    reader = decord.VideoReader(
        path,
        ctx=decord.cpu(0),
        height=read_h,
        width=read_w,
        # Per-process ffmpeg threads. Parallelism belongs in DataLoader
        # worker processes (latent_cache) or ProcessPoolExecutor, not here:
        # extra threads oversubscribe once many workers are decoding.
        num_threads=num_threads,
    )
    frames = reader.get_batch(frame_indices)
    tensor = torch.from_numpy(frames.asnumpy()).permute(3, 0, 1, 2)
    tensor = transform_centercrop(tensor)
    if output_mode == VIDEO_OUTPUT_MODE_UINT8:
        return tensor
    return tensor / 255 * 2 - 1


def dataset_timing_fields(
    *,
    frame_plan: dict[str, Any],
    frame_policy: str,
    video_decode_seconds: float,
    caption_seconds: float,
    total_seconds: float,
) -> dict[str, Any]:
    return {
        "dataset_video_decode": video_decode_seconds,
        "dataset_caption": caption_seconds,
        "dataset_total": total_seconds,
        "source_rgb_frames": frame_plan["source_rgb_frames"],
        "train_rgb_frames": frame_plan["train_rgb_frames"],
        "train_latent_frames": frame_plan["train_latent_frames"],
        "shot_count": len(frame_plan["shot_groups"]),
        "frame_policy": frame_policy,
        "max_rgb_frames": frame_plan["max_rgb_frames"],
        "vae_temporal_stride": frame_plan["vae_temporal_stride"],
        "source_fps": frame_plan["source_fps"],
        "source_num_frames": frame_plan["source_num_frames"],
        "source_duration_sec": frame_plan["source_duration_sec"],
        "target_fps": frame_plan["target_fps"],
        "effective_train_fps": frame_plan["effective_train_fps"],
        "rgb_was_scaled": frame_plan["rgb_was_scaled"],
    }


def row_string(row: dict[str, Any], column: str, label: str) -> str:
    value = row[column]
    try:
        is_nan = bool(pd.isna(value))
    except (TypeError, ValueError):
        is_nan = False
    if is_nan:
        raise ValueError(f"{label} is NaN; required string field cannot be an empty CSV cell")
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a non-empty string")
    result = require_non_empty_string(value, label)
    if result != result.strip():
        raise ValueError(f"{label} must not contain surrounding whitespace")
    return result


class MultishotCsvDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        metadata_path: str,
        repeat: int,
        height: int,
        width: int,
        frame_policy: str,
        max_rgb_frames: int,
        target_fps: float,
        vae_temporal_stride: int,
        schema: str,
        prompt_format: str,
        video_column: str,
        caption_column: str | None,
        shot_groups_column: str,
        case_id_column: str | None,
        case_id_prefix: str,
        shot2story_prompt_source: dict[str, Any] | None = None,
        video_decode_num_threads: int = 0,
        video_output_mode: str = VIDEO_OUTPUT_MODE_NORMALIZED_FLOAT,
    ):
        if type(repeat) is not int or repeat <= 0:
            raise ValueError("repeat must be a positive int")
        if type(height) is not int or height <= 0:
            raise ValueError("height must be a positive int")
        if type(width) is not int or width <= 0:
            raise ValueError("width must be a positive int")
        if frame_policy != FRAME_POLICY_FPS_AWARE_TARGET_FPS:
            raise ValueError(
                f"unsupported frame_policy={frame_policy}; expected {FRAME_POLICY_FPS_AWARE_TARGET_FPS}"
            )
        supported_schemas = [
            MULTISHOT_CSV_SCHEMA,
            SCENE_MAPPING_SCHEMA,
            *sorted(EXPLICIT_MAPPING_SCHEMAS),
            *sorted(MULTISOURCE_MAPPING_SCHEMAS),
        ]
        if schema not in {
            MULTISHOT_CSV_SCHEMA,
            SCENE_MAPPING_SCHEMA,
            *EXPLICIT_MAPPING_SCHEMAS,
            *MULTISOURCE_MAPPING_SCHEMAS,
        }:
            raise ValueError(
                f"unsupported dataset schema={schema}; expected one of "
                f"{supported_schemas}"
            )
        validate_prompt_format(prompt_format)
        self.metadata_path = require_non_empty_string(metadata_path, "metadata_path")
        if self.metadata_path != self.metadata_path.strip():
            raise ValueError("metadata_path must not contain surrounding whitespace")
        self.repeat = repeat
        self.height = height
        self.width = width
        self.frame_policy = frame_policy
        if type(video_decode_num_threads) is not int or video_decode_num_threads < 0:
            raise ValueError("video_decode_num_threads must be a non-negative int")
        if video_output_mode not in SUPPORTED_VIDEO_OUTPUT_MODES:
            raise ValueError(
                f"unsupported video_output_mode={video_output_mode!r}; expected one of "
                f"{sorted(SUPPORTED_VIDEO_OUTPUT_MODES)}"
            )
        self.video_decode_num_threads = video_decode_num_threads
        self.video_output_mode = video_output_mode
        self.schema = schema
        self.prompt_format = prompt_format
        self.video_column = require_non_empty_string(video_column, "video_column")
        self.caption_column = (
            require_non_empty_string(caption_column, "caption_column")
            if caption_column is not None
            else None
        )
        self.shot_groups_column = require_non_empty_string(shot_groups_column, "shot_groups_column")
        self.case_id_column = (
            require_non_empty_string(case_id_column, "case_id_column")
            if case_id_column is not None
            else None
        )
        self.case_id_prefix = require_non_empty_string(case_id_prefix, "case_id_prefix")
        self.shot2story_prompt_source_config = (
            Shot2StoryRawPromptSourceConfig.from_mapping(shot2story_prompt_source)
            if shot2story_prompt_source is not None
            else None
        )
        self.shot2story_prompt_source = (
            Shot2StoryRawPromptSource.load(self.shot2story_prompt_source_config)
            if self.shot2story_prompt_source_config is not None
            else None
        )
        self.media = {"height": height, "width": width, "target_fps": target_fps,
                      "vae_temporal_stride": vae_temporal_stride, "max_rgb_frames": max_rgb_frames}
        if self.schema == SCENE_MAPPING_SCHEMA:
            if self.caption_column is not None or self.shot2story_prompt_source is not None:
                raise ValueError("scene_window_v1 has no caption source; set caption_column=null")
            from mspc.data.scene_training_manifest import require_training_media
            require_training_media(self.media)
        elif self.shot2story_prompt_source is None and self.caption_column is None:
            raise ValueError("caption_column is required without shot2story_prompt_source")
        self.frame_planner = FpsAwareTargetFpsPlanner(
            max_rgb_frames=max_rgb_frames,
            target_fps=target_fps,
            vae_temporal_stride=vae_temporal_stride,
        )
        import torchvision.transforms as transforms

        self.transform_centercrop = transforms.CenterCrop((self.height, self.width))
        self.metadata_summary: dict[str, Any] = {}
        self.data = self._load_metadata()

    def _load_metadata(self) -> list[dict[str, Any]]:
        metadata = pd.read_csv(self.metadata_path)
        required_columns = [self.video_column, self.shot_groups_column]
        if self.caption_column is not None:
            required_columns.append(self.caption_column)
        if is_explicit_mapping_schema(self.schema):
            required_columns.extend([SOURCE_SHOT_SEGMENTS_COLUMN, SOURCE_FRAME_INDICES_COLUMN])
        elif is_multisource_mapping_schema(self.schema):
            required_columns.extend(
                [
                    SOURCE_SHOT_SEGMENTS_COLUMN,
                    SOURCE_VIDEO_PATHS_COLUMN,
                    SOURCE_FRAME_INDICES_BY_SHOT_COLUMN,
                ]
            )
        if self.case_id_column is not None:
            required_columns.append(self.case_id_column)
        if self.shot2story_prompt_source_config is not None:
            required_columns.extend(self.shot2story_prompt_source_config.required_csv_columns())
        if self.schema == SCENE_MAPPING_SCHEMA:
            from mspc.data.scene_cache import SCENE_COLUMNS

            required_columns.extend(SCENE_COLUMNS)
        missing = sorted(set(required_columns) - set(metadata.columns))
        if missing:
            raise ValueError(
                f"{self.metadata_path}: missing required columns: {', '.join(missing)}"
            )
        rows = [metadata.iloc[index].to_dict() for index in range(len(metadata))]
        if not rows:
            raise ValueError(f"{self.metadata_path}: metadata contains no rows")
        required_strings = [self.video_column, self.shot_groups_column]
        if self.caption_column is not None:
            required_strings.append(self.caption_column)
        if self.case_id_column is not None:
            required_strings.append(self.case_id_column)
        if is_explicit_mapping_schema(self.schema):
            required_strings.extend([SOURCE_SHOT_SEGMENTS_COLUMN, SOURCE_FRAME_INDICES_COLUMN])
        elif is_multisource_mapping_schema(self.schema):
            required_strings.extend(
                [
                    SOURCE_SHOT_SEGMENTS_COLUMN,
                    SOURCE_VIDEO_PATHS_COLUMN,
                    SOURCE_FRAME_INDICES_BY_SHOT_COLUMN,
                ]
            )
        if "mapping_schema" in metadata.columns:
            required_strings.append("mapping_schema")
        seen_case_ids: set[str] = set()
        for index, row in enumerate(rows):
            label = f"{self.metadata_path}: row {index}"
            for column in required_strings:
                row_string(row, column, f"{label} {column}")
            case_id = (
                f"{self.case_id_prefix}_{index:06d}"
                if self.case_id_column is None
                else row_string(row, self.case_id_column, f"{label} {self.case_id_column}")
            )
            if "mapping_schema" in row:
                mapping_schema = row_string(row, "mapping_schema", f"{label} mapping_schema")
                if mapping_schema != self.schema:
                    raise ValueError(
                        f"{label}: mapping_schema={mapping_schema!r} does not match "
                        f"dataset schema={self.schema!r}"
                    )
            if case_id in seen_case_ids:
                raise ValueError(f"{self.metadata_path}: duplicate case_id={case_id}")
            seen_case_ids.add(case_id)
            row["_case_id"] = case_id
        self.metadata_summary = {
            "schema": self.schema,
            "prompt_format": self.prompt_format,
            "frame_policy": self.frame_policy,
            "target_fps": self.frame_planner.target_fps,
            "max_rgb_frames": self.frame_planner.max_rgb_frames,
            "vae_temporal_stride": self.frame_planner.vae_temporal_stride,
            "video_decode_num_threads": self.video_decode_num_threads,
            "video_output_mode": self.video_output_mode,
            "metadata_path": self.metadata_path,
            "source_rows": len(rows),
            "kept_rows": len(rows),
            "dropped_rows": 0,
            "case_id_column": self.case_id_column,
            "case_id_prefix": self.case_id_prefix,
            "shot2story_prompt_source": (
                self.shot2story_prompt_source.summary()
                if self.shot2story_prompt_source is not None
                else None
            ),
        }
        return rows

    def _prepare_row(
        self,
        index: int,
        row: dict[str, Any],
    ) -> dict[str, Any]:
        label = f"{self.metadata_path}: row {index}"
        video_path = row_string(row, self.video_column, f"{label} {self.video_column}")
        if self.schema == SCENE_MAPPING_SCHEMA:
            from mspc.data.scene_cache import prepare_scene_row

            return prepare_scene_row(row, video_path=video_path, shot_groups_column=self.shot_groups_column, media=self.media)
        if self.shot2story_prompt_source is None:
            assert self.caption_column is not None
            caption_path = row_string(row, self.caption_column, f"{label} {self.caption_column}")
        else:
            caption_path = self.shot2story_prompt_source.path
        shot_groups = parse_shot_groups(row[self.shot_groups_column], video_path)
        if is_explicit_mapping_schema(self.schema):
            source_segments = parse_source_shot_segments(
                row_string(row, SOURCE_SHOT_SEGMENTS_COLUMN, f"{label} {SOURCE_SHOT_SEGMENTS_COLUMN}"),
                video_path,
            )
            source_frame_indices = parse_source_frame_indices(
                row_string(row, SOURCE_FRAME_INDICES_COLUMN, f"{label} {SOURCE_FRAME_INDICES_COLUMN}"),
                video_path,
            )
            if len(source_segments) != len(shot_groups):
                raise ValueError(f"{label}: source/train shot counts do not match")
            row["_source_shot_segments"] = source_segments
            row["_source_frame_indices"] = source_frame_indices
        elif is_multisource_mapping_schema(self.schema):
            from mspc.data.frame_planner import validate_multisource_shot_segments

            source_segments = ast.literal_eval(
                row_string(row, SOURCE_SHOT_SEGMENTS_COLUMN, f"{label} {SOURCE_SHOT_SEGMENTS_COLUMN}")
            )
            validate_multisource_shot_segments(source_segments, video_path)
            source_video_paths = parse_source_video_paths(
                row_string(row, SOURCE_VIDEO_PATHS_COLUMN, f"{label} {SOURCE_VIDEO_PATHS_COLUMN}"),
                video_path,
            )
            frame_indices_by_shot = parse_source_frame_indices_by_shot(
                row_string(
                    row,
                    SOURCE_FRAME_INDICES_BY_SHOT_COLUMN,
                    f"{label} {SOURCE_FRAME_INDICES_BY_SHOT_COLUMN}",
                ),
                video_path,
            )
            if not (
                len(source_segments) == len(shot_groups)
                == len(source_video_paths) == len(frame_indices_by_shot)
            ):
                raise ValueError(f"{label}: source/train shot counts do not match")
            for source_path in source_video_paths:
                path = Path(source_path)
                if not path.is_file():
                    raise FileNotFoundError(path)
                if path.stat().st_size <= 0:
                    raise ValueError(f"{label}: empty source video: {path}")
            row["_source_shot_segments"] = [list(segment) for segment in source_segments]
            row["_source_video_paths"] = source_video_paths
            row["_source_frame_indices_by_shot"] = frame_indices_by_shot
        prompts = (
            self.shot2story_prompt_source.prompts_for_row(
                row,
                video_path=video_path,
                shot_count=len(shot_groups),
                source_shot_segments=row.get("_source_shot_segments", shot_groups),
                prompt_format=self.prompt_format,
                build_prompts_from_fields_fn=build_prompts_from_fields,
                label=label,
            )
            if self.shot2story_prompt_source is not None
            else build_prompts(
                caption_path,
                len(shot_groups),
                prompt_format=self.prompt_format,
            )
        )
        row["_shot_groups"] = shot_groups
        row["_prompt_source_path"] = caption_path
        row["_prompts"] = tuple(prompts)
        return row

    def _read_video(
        self,
        path: str,
        frame_indices: list[int],
        *,
        video_metadata: dict[str, int | float] | None = None,
    ) -> torch.Tensor:
        video_height = None
        video_width = None
        if video_metadata is not None:
            video_height = video_metadata.get("source_height")
            video_width = video_metadata.get("source_width")
        return read_video_frames(
            path=path,
            frame_indices=frame_indices,
            target_height=self.height,
            target_width=self.width,
            transform_centercrop=self.transform_centercrop,
            video_height=video_height if isinstance(video_height, int) else None,
            video_width=video_width if isinstance(video_width, int) else None,
            num_threads=self.video_decode_num_threads,
            output_mode=self.video_output_mode,
        )

    def _read_multisource_video(
        self,
        source_video_paths: list[str],
        frame_indices_by_shot: list[list[int]],
        *,
        source_metadata_by_shot: list[dict[str, int | float] | None] | None = None,
    ) -> torch.Tensor:
        if source_metadata_by_shot is None:
            source_metadata_by_shot = [None] * len(source_video_paths)
        decoded_shots = []
        for path, frame_indices, video_metadata in zip(
            source_video_paths,
            frame_indices_by_shot,
            source_metadata_by_shot,
            strict=True,
        ):
            decoded_shots.append(
                self._read_video(path, frame_indices, video_metadata=video_metadata)
            )
        if not decoded_shots:
            raise ValueError("multisource mapping did not provide any decodable shots")
        return torch.cat(decoded_shots, dim=1)

    def describe_media(self, data_id: int) -> dict[str, Any]:
        """实际解码计划；缓存生产/复用与读帧共用，不编码像素。"""
        from mspc.data.cache_provenance import describe_media_provenance

        index = data_id % len(self.data)
        row = self._prepare_row(index, self.data[index].copy())
        video_path = row[self.video_column]
        video_metadata = None
        shot_groups = row["_shot_groups"]
        if self.schema == SCENE_MAPPING_SCHEMA:
            video_metadata = row["_scene_metadata"]
            source_metadata_by_shot = None
            if (self.height, self.width, self.frame_planner.target_fps) == (704, 1280, 24):
                from mspc.data.scene_training_manifest import frozen_scene_frame_plan
                frame_plan = frozen_scene_frame_plan(row, target_fps=self.frame_planner.target_fps,
                    max_rgb_frames=self.frame_planner.max_rgb_frames,
                    vae_temporal_stride=self.frame_planner.vae_temporal_stride)
            else:
                frame_plan = self.frame_planner.build_multisource_explicit_frame_plan(
                    source_video_paths=[video_path] * len(shot_groups),
                    source_shot_segments=row["_source_shot_segments"],
                    train_shot_groups=shot_groups,
                    frame_indices_by_shot=row["_source_frame_indices_by_shot"],
                    data_path=video_path,
                    source_fps_by_shot=[video_metadata["source_fps"]] * len(shot_groups),
                    source_num_frames_by_shot=[video_metadata["source_num_frames"]] * len(shot_groups),
                )
                frame_plan["mapping_kind"] = "scene_ffmpeg_sequential_clean_v1"
                frame_plan["source_num_frames"] = video_metadata["source_num_frames"]
        elif is_multisource_mapping_schema(self.schema):
            source_metadata_by_shot = [
                read_video_metadata(path) for path in row["_source_video_paths"]
            ]
            frame_plan = self.frame_planner.build_multisource_explicit_frame_plan(
                source_video_paths=row["_source_video_paths"],
                source_shot_segments=row["_source_shot_segments"],
                train_shot_groups=shot_groups,
                frame_indices_by_shot=row["_source_frame_indices_by_shot"],
                data_path=video_path,
                source_fps_by_shot=[
                    metadata["source_fps"] for metadata in source_metadata_by_shot
                ],
                source_num_frames_by_shot=[
                    metadata["source_num_frames"] for metadata in source_metadata_by_shot
                ],
            )
        elif is_explicit_mapping_schema(self.schema):
            source_metadata_by_shot = None
            video_metadata = read_video_metadata(video_path)
            frame_plan = self.frame_planner.build_explicit_frame_plan(
                source_shot_segments=row["_source_shot_segments"],
                train_shot_groups=shot_groups,
                frame_indices=row["_source_frame_indices"],
                data_path=video_path,
                source_fps=video_metadata["source_fps"],
                source_num_frames=video_metadata["source_num_frames"],
                vae_alignment_policy=vae_alignment_policy_for_schema(self.schema),
            )
        else:
            source_metadata_by_shot = None
            video_metadata = read_video_metadata(video_path)
            frame_plan = self.frame_planner.build_frame_plan(
                shot_groups,
                video_path,
                source_fps=video_metadata["source_fps"],
                source_num_frames=video_metadata["source_num_frames"],
            )

        paths = row["_source_video_paths"] if is_multisource_mapping_schema(self.schema) else [video_path]
        metadata = source_metadata_by_shot if is_multisource_mapping_schema(self.schema) else [video_metadata]
        evidence = describe_media_provenance(
            paths=paths, metadata=metadata, frame_plan=frame_plan,
            source_shot_segments=(row["_source_shot_segments"] if is_deterministic_mapping_schema(self.schema) else row["_shot_groups"]),
            schema=self.schema, target_height=self.height, target_width=self.width,
            scene_row=row if self.schema == SCENE_MAPPING_SCHEMA else None,
        )
        return {"row": row, "frame_plan": frame_plan, "video_metadata": video_metadata,
                "source_metadata_by_shot": source_metadata_by_shot, "provenance": evidence}

    def _load_sample(self, data_id: int) -> dict[str, Any]:
        item_start = time.perf_counter()
        description = self.describe_media(data_id)
        row = description["row"]
        video_path = row[self.video_column]
        caption_path = row["_prompt_source_path"]
        frame_plan = description["frame_plan"]
        video_metadata = description["video_metadata"]
        source_metadata_by_shot = description["source_metadata_by_shot"]
        video_start = time.perf_counter()
        if self.schema == SCENE_MAPPING_SCHEMA:
            from mspc.data.scene_cache import read_scene_video

            video = read_scene_video(row, frame_plan["frame_indices"], output_mode=self.video_output_mode,
                                     video_decode_num_threads=self.video_decode_num_threads, media=self.media)
        elif is_multisource_mapping_schema(self.schema):
            video = self._read_multisource_video(
                row["_source_video_paths"],
                frame_plan["frame_indices_by_shot"],
                source_metadata_by_shot=source_metadata_by_shot,
            )
        else:
            video = self._read_video(
                video_path,
                frame_plan["frame_indices"],
                video_metadata=video_metadata,
            )
        video_end = time.perf_counter()
        caption_start = time.perf_counter()
        prompts = list(row["_prompts"])
        if self.schema != SCENE_MAPPING_SCHEMA and len(prompts) != len(frame_plan["latent_shot_groups"]):
            raise ValueError(f"{caption_path}: prompt count does not match latent_shot_groups")

        sample = {
            "video": video,
            "prompt": prompts,
            "shot_groups": frame_plan["shot_groups"],
            "latent_shot_groups": frame_plan["latent_shot_groups"],
            "data_path": video_path,
            "media_provenance": description["provenance"],
            "_dataset_timing": dataset_timing_fields(
                frame_plan=frame_plan,
                frame_policy=self.frame_policy,
                video_decode_seconds=video_end - video_start,
                caption_seconds=time.perf_counter() - caption_start,
                total_seconds=time.perf_counter() - item_start,
            ),
        }
        if is_multisource_mapping_schema(self.schema):
            sample["source_video_paths"] = list(row["_source_video_paths"])
            sample["source_frame_indices_by_shot"] = [
                list(indices) for indices in frame_plan["frame_indices_by_shot"]
            ]
        sample["case_id"] = row["_case_id"]
        return sample

    def __getitem__(self, data_id: int) -> dict[str, Any]:
        return self._load_sample(data_id % len(self.data))

    def __len__(self) -> int:
        return len(self.data) * self.repeat


def build_training_dataset(args: Any) -> MultishotCsvDataset:
    """Build this registered dataset from component-neutral runtime arguments."""

    return MultishotCsvDataset(**args.dataset_cfg)
