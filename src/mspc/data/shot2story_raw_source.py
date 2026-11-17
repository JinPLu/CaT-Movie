"""Raw Shot2Story annotation prompt source validation."""

from __future__ import annotations

from dataclasses import dataclass
import ast
import re
from pathlib import Path
from typing import Any, Mapping

from mspc.data.metadata import require_non_empty_string
from mspc.data.strict_json import load_strict_json_file


SHOT2STORY_RAW_ANNOTATION_PROMPT_SOURCE = "shot2story_raw_annotation_v1"
SOURCE_ANNOTATION_INDEX_COLUMN = "source_annotation_index"
SOURCE_ANNOTATION_VIDEO_COLUMN = "source_annotation_video"
SOURCE_ANNOTATION_SELECTED_VIDEO_NAMES_COLUMN = "source_annotation_selected_video_names"
SELECTED_SHOT_INDICES_COLUMN = "selected_shot_indices"
SEGMENT_PATTERN = re.compile(r"_(\d+)_(\d+)\.mp4$")


def parse_shot2story_video_name_segment(name: str, *, label: str) -> list[int]:
    match = SEGMENT_PATTERN.search(name)
    if match is None:
        raise ValueError(f"{label}: cannot parse source frame range from {name!r}")
    start, inclusive_end = map(int, match.groups())
    return [start, inclusive_end + 1]


def _require_trimmed_string(value: Any, label: str) -> str:
    text = require_non_empty_string(value, label)
    if text != text.strip():
        raise ValueError(f"{label} must not contain surrounding whitespace")
    return text


def parse_selected_shot_indices(value: Any, label: str) -> list[int]:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if (
        not isinstance(parsed, list)
        or not parsed
        or any(type(index) is not int or index < 0 for index in parsed)
    ):
        raise ValueError(f"{label} must be a non-empty list of non-negative ints")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{label} must not contain duplicates")
    return list(parsed)


def annotation_item_text(value: Any, *, field: str, label: str) -> str:
    if isinstance(value, str):
        return _require_trimmed_string(value, label)
    if isinstance(value, dict) and set(value) == {field}:
        return _require_trimmed_string(value[field], f"{label}.{field}")
    raise ValueError(f"{label} must be a string or an object containing only {field!r}")


@dataclass(frozen=True)
class Shot2StoryRawPromptSourceConfig:
    kind: str
    annotation_json: str
    selected_shot_indices_column: str

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        label: str = "shot2story_prompt_source",
    ) -> "Shot2StoryRawPromptSourceConfig":
        expected = {
            "kind",
            "annotation_json",
            "selected_shot_indices_column",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError(f"{label} keys must be exactly {sorted(expected)}")
        kind = _require_trimmed_string(value["kind"], f"{label}.kind")
        if kind != SHOT2STORY_RAW_ANNOTATION_PROMPT_SOURCE:
            raise ValueError(
                f"{label}.kind must be {SHOT2STORY_RAW_ANNOTATION_PROMPT_SOURCE!r}"
            )
        return cls(
            kind=kind,
            annotation_json=_require_trimmed_string(
                value["annotation_json"],
                f"{label}.annotation_json",
            ),
            selected_shot_indices_column=_require_trimmed_string(
                value["selected_shot_indices_column"],
                f"{label}.selected_shot_indices_column",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "annotation_json": self.annotation_json,
            "selected_shot_indices_column": self.selected_shot_indices_column,
        }

    def required_csv_columns(self) -> list[str]:
        return [self.selected_shot_indices_column]


@dataclass(frozen=True)
class _AnnotationRow:
    video: str
    video_basename: str
    video_names: tuple[str, ...]
    whole_caption: str
    captions: tuple[str, ...]


@dataclass(frozen=True)
class Shot2StoryRawPromptSource:
    config: Shot2StoryRawPromptSourceConfig
    path: str
    rows: tuple[_AnnotationRow, ...]
    rows_by_video_basename: Mapping[str, _AnnotationRow]

    @classmethod
    def load(cls, config: Shot2StoryRawPromptSourceConfig) -> "Shot2StoryRawPromptSource":
        path = Path(config.annotation_json).resolve(strict=True)
        payload = load_strict_json_file(path, label=f"Shot2Story raw annotations {path}")
        if not isinstance(payload, list):
            raise ValueError(f"{path}: annotations must be a list")
        rows: list[_AnnotationRow] = []
        rows_by_video_basename: dict[str, _AnnotationRow] = {}
        for index, annotation in enumerate(payload):
            if not isinstance(annotation, dict):
                raise ValueError(f"{path}: row {index} must be an object")
            row = _parse_annotation_row(annotation, index=index, path=path)
            previous = rows_by_video_basename.get(row.video_basename)
            if previous is not None:
                raise ValueError(
                    f"{path}: duplicate video basename binding {row.video_basename}"
                )
            rows_by_video_basename[row.video_basename] = row
            rows.append(row)
        return cls(
            config=config,
            path=str(path),
            rows=tuple(rows),
            rows_by_video_basename=rows_by_video_basename,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "kind": self.config.kind,
            "annotation_json": self.path,
            "annotation_rows": len(self.rows),
            "selected_shot_indices_column": self.config.selected_shot_indices_column,
        }

    def prompts_for_row(
        self,
        row: Mapping[str, Any],
        *,
        video_path: str,
        shot_count: int,
        source_shot_segments: list[list[int]],
        prompt_format: str,
        build_prompts_from_fields_fn: Any,
        label: str,
    ) -> list[str]:
        annotation, selected_indices = self.validate_row_binding(
            row,
            video_path=video_path,
            shot_count=shot_count,
            source_shot_segments=source_shot_segments,
            label=label,
        )
        return build_prompts_from_fields_fn(
            [annotation.captions[index] for index in selected_indices],
            prompt_format=prompt_format,
        )

    def validate_row_binding(
        self,
        row: Mapping[str, Any],
        *,
        video_path: str,
        shot_count: int,
        source_shot_segments: list[list[int]],
        label: str,
    ) -> tuple[_AnnotationRow, list[int]]:
        cfg = self.config
        video_basename = Path(video_path).name
        annotation = self.rows_by_video_basename.get(video_basename)
        if annotation is None:
            raise ValueError(
                f"{label}: no raw annotation video basename binding for {video_basename!r}"
            )
        selected_indices = parse_selected_shot_indices(
            row[cfg.selected_shot_indices_column],
            f"{label} selected_shot_indices",
        )
        if len(selected_indices) != shot_count:
            raise ValueError(f"{label}: selected_shot_indices count does not match shot_groups")
        if len(source_shot_segments) != shot_count:
            raise ValueError(f"{label}: source_shot_segments count does not match shot_groups")
        if any(index >= len(annotation.captions) for index in selected_indices):
            raise ValueError(f"{label}: selected_shot_indices out of range for captions")
        if any(index >= len(annotation.video_names) for index in selected_indices):
            raise ValueError(f"{label}: selected_shot_indices out of range for video_names")
        selected_video_names = [annotation.video_names[index] for index in selected_indices]
        expected_segments = [
            parse_shot2story_video_name_segment(name, label=label)
            for name in selected_video_names
        ]
        if source_shot_segments != expected_segments:
            raise ValueError(f"{label}: source_shot_segments do not match source annotation video_names")
        return annotation, selected_indices


def _parse_annotation_row(annotation: Mapping[str, Any], *, index: int, path: Path) -> _AnnotationRow:
    label = f"{path}: row {index}"
    video = _require_trimmed_string(annotation.get("video"), f"{label}.video")
    video_basename = Path(video).name
    if not video_basename:
        raise ValueError(f"{label}.video must have a basename")
    whole_caption = _require_trimmed_string(
        annotation.get("whole_caption"),
        f"{label}.whole_caption",
    )
    video_names_value = annotation.get("video_names")
    captions_value = annotation.get("captions")
    if not isinstance(video_names_value, list) or not video_names_value:
        raise ValueError(f"{label}.video_names must be a non-empty list")
    if not isinstance(captions_value, list) or not captions_value:
        raise ValueError(f"{label}.captions must be a non-empty list")
    video_names = tuple(
        annotation_item_text(
            value,
            field="video_name",
            label=f"{label}.video_names[{item_index}]",
        )
        for item_index, value in enumerate(video_names_value)
    )
    captions = tuple(
        annotation_item_text(
            value,
            field="caption",
            label=f"{label}.captions[{item_index}]",
        )
        for item_index, value in enumerate(captions_value)
    )
    if len(video_names) != len(captions):
        raise ValueError(f"{label}: video_names and captions must align one-to-one")
    return _AnnotationRow(
        video=video,
        video_basename=video_basename,
        video_names=video_names,
        whole_caption=whole_caption,
        captions=captions,
    )
