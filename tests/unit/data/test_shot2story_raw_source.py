from __future__ import annotations

import json
from pathlib import Path

import pytest

from mspc.data.shot2story_raw_source import (
    SHOT2STORY_RAW_ANNOTATION_PROMPT_SOURCE,
    Shot2StoryRawPromptSource,
    Shot2StoryRawPromptSourceConfig,
)


def _annotation(path: Path) -> Path:
    path.write_text(
        json.dumps(
            [
                {
                    "video": "/videos/story.mp4",
                    "whole_caption": "A short story",
                    "video_names": ["shot_0_4.mp4", "shot_5_9.mp4", "shot_10_14.mp4"],
                    "captions": ["first", "second", "third"],
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


def _config(path: Path) -> Shot2StoryRawPromptSourceConfig:
    return Shot2StoryRawPromptSourceConfig.from_mapping(
        {
            "kind": SHOT2STORY_RAW_ANNOTATION_PROMPT_SOURCE,
            "annotation_json": str(path),
            "selected_shot_indices_column": "selected_shot_indices",
        }
    )


def test_config_contains_path_and_selection_column_without_hash(tmp_path: Path) -> None:
    config = _config(_annotation(tmp_path / "annotations.json"))
    assert config.to_dict() == {
        "kind": SHOT2STORY_RAW_ANNOTATION_PROMPT_SOURCE,
        "annotation_json": str(tmp_path / "annotations.json"),
        "selected_shot_indices_column": "selected_shot_indices",
    }
    with pytest.raises(ValueError, match="keys must be exactly"):
        Shot2StoryRawPromptSourceConfig.from_mapping(
            {**config.to_dict(), "annotation_sha256": "0" * 64}
        )


def test_selected_raw_prompts_preserve_requested_shot_order(tmp_path: Path) -> None:
    source = Shot2StoryRawPromptSource.load(_config(_annotation(tmp_path / "annotations.json")))
    prompts = source.prompts_for_row(
        {"selected_shot_indices": "[2, 0]"},
        video_path="/other/story.mp4",
        shot_count=2,
        source_shot_segments=[[10, 15], [0, 5]],
        prompt_format="caption_json_v1",
        build_prompts_from_fields_fn=lambda shots, **_: [f"shot: {shot}" for shot in shots],
        label="row 0",
    )
    assert prompts == ["shot: third", "shot: first"]
    assert "annotation_sha256" not in source.summary()


def test_binding_requires_prompt_count_and_source_segments(tmp_path: Path) -> None:
    source = Shot2StoryRawPromptSource.load(_config(_annotation(tmp_path / "annotations.json")))
    with pytest.raises(ValueError, match="count does not match"):
        source.validate_row_binding(
            {"selected_shot_indices": "[0]"},
            video_path="story.mp4",
            shot_count=2,
            source_shot_segments=[[0, 5], [5, 10]],
            label="row 0",
        )
    with pytest.raises(ValueError, match="do not match"):
        source.validate_row_binding(
            {"selected_shot_indices": "[0, 1]"},
            video_path="story.mp4",
            shot_count=2,
            source_shot_segments=[[0, 5], [10, 15]],
            label="row 0",
        )


def test_duplicate_video_basename_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "annotations.json"
    row = {
        "video": "/a/story.mp4",
        "whole_caption": "story",
        "video_names": ["shot_0_4.mp4"],
        "captions": ["shot"],
    }
    path.write_text(json.dumps([row, {**row, "video": "/b/story.mp4"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate video basename"):
        Shot2StoryRawPromptSource.load(_config(path))
