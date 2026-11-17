from __future__ import annotations

from tests.support.paths import REPO_ROOT
import csv
import json
import sys
import types
from pathlib import Path

import pytest


def install_import_stubs() -> None:
    try:
        import torch  # noqa: F401
    except Exception:
        torch_stub = types.ModuleType("torch")
        utils_stub = types.ModuleType("torch.utils")
        data_stub = types.ModuleType("torch.utils.data")
        data_stub.Dataset = object
        utils_stub.data = data_stub
        torch_stub.utils = utils_stub
        sys.modules["torch"] = torch_stub
        sys.modules["torch.utils"] = utils_stub
        sys.modules["torch.utils.data"] = data_stub

    try:
        import torchvision.transforms  # noqa: F401
    except Exception:
        torchvision_stub = types.ModuleType("torchvision")
        transforms_stub = types.ModuleType("torchvision.transforms")

        class CenterCrop:
            def __init__(self, size):
                self.size = size

            def __call__(self, value):
                return value

        transforms_stub.CenterCrop = CenterCrop
        torchvision_stub.transforms = transforms_stub
        sys.modules["torchvision"] = torchvision_stub
        sys.modules["torchvision.transforms"] = transforms_stub


install_import_stubs()

sys.path.insert(0, str(REPO_ROOT))

from mspc.data.multishot import (  # noqa: E402
    PROMPT_FORMAT_CAPTION_JSON_V1,
    MultishotCsvDataset,
    build_prompts_from_fields,
    caption_fields,
)
import mspc.inference.common as inference_common  # noqa: E402
from mspc.config import parse_experiment_mapping  # noqa: E402
from mspc.training.training_args import build_training_dataset_config  # noqa: E402
from tests.unit.config.test_experiment_config import experiment_mapping  # noqa: E402


def write_dataset_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    video_path = tmp_path / "video.mp4"
    video_path.write_text("video", encoding="utf-8")
    caption_path = tmp_path / "caption.json"
    caption_path.write_text(
        json.dumps(
            {
                "global_caption": "a person prepares tea",
                "shot0": "the kettle is placed on a table",
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "video_path", "caption_path", "shot_groups"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "case_001",
                "video_path": str(video_path),
                "caption_path": str(caption_path),
                "shot_groups": "[[0, 4]]",
            }
        )
    return manifest_path, video_path, caption_path


def dataset_kwargs(manifest_path: Path, prompt_format: str) -> dict[str, object]:
    return {
        "metadata_path": str(manifest_path),
        "repeat": 1,
        "height": 16,
        "width": 16,
        "frame_policy": "fps_aware_target_fps",
        "max_rgb_frames": 4,
        "target_fps": 1,
        "vae_temporal_stride": 4,
        "schema": "multishot_csv_v1",
        "prompt_format": prompt_format,
        "video_column": "video_path",
        "caption_column": "caption_path",
        "shot_groups_column": "shot_groups",
        "case_id_column": "case_id",
        "case_id_prefix": "generated_case",
    }


def test_the_rendered_prompt_is_the_shot_caption_without_a_story_segment():
    prompts = build_prompts_from_fields(
        ["the kettle is placed on a table"],
        prompt_format=PROMPT_FORMAT_CAPTION_JSON_V1,
    )

    assert prompts == ["the kettle is placed on a table"]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"global_caption": 7, "shot0": "shot"}, "global_caption must be a non-empty string"),
        ({"global_caption": " story ", "shot0": "shot"}, "surrounding whitespace"),
        ({"global_caption": "story", "shot0": 7}, "shot0 must be a non-empty string"),
    ],
)
def test_caption_fields_rejects_coercion_and_whitespace(tmp_path: Path, payload, message):
    caption_path = tmp_path / "caption.json"
    caption_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        caption_fields(str(caption_path), 1)


def test_multishot_dataset_accepts_v1_prompt_format(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("mspc.data.cache_provenance.importlib.metadata.version", lambda _: "test")
    manifest_path, _, _ = write_dataset_fixture(tmp_path)
    monkeypatch.setattr(
        "mspc.data.multishot.read_video_metadata",
        lambda path: {"source_fps": 1.0, "source_num_frames": 4, "source_width": 16, "source_height": 16},
    )

    dataset = MultishotCsvDataset(**dataset_kwargs(manifest_path, PROMPT_FORMAT_CAPTION_JSON_V1))
    dataset._read_video = lambda path, frame_indices, **kwargs: "video_tensor"

    sample = dataset[0]

    assert sample["prompt"] == ["the kettle is placed on a table"]
    assert dataset.metadata_summary["prompt_format"] == PROMPT_FORMAT_CAPTION_JSON_V1


def test_training_dataset_config_accepts_v1_prompt_format(tmp_path: Path):
    manifest_path, _, _ = write_dataset_fixture(tmp_path)
    mapping = experiment_mapping()
    runtime_dataset = dataset_kwargs(manifest_path, PROMPT_FORMAT_CAPTION_JSON_V1)
    for field in (
        "height",
        "width",
        "frame_policy",
        "max_rgb_frames",
        "target_fps",
        "vae_temporal_stride",
    ):
        runtime_dataset.pop(field)
    mapping["dataset"].update(**runtime_dataset)
    config = parse_experiment_mapping(mapping, source_path=tmp_path / "experiment.yaml")

    dataset_cfg = build_training_dataset_config(config)

    assert dataset_cfg["prompt_format"] == PROMPT_FORMAT_CAPTION_JSON_V1


def test_inference_case_inputs_uses_shared_prompt_builder(tmp_path: Path, monkeypatch):
    _, video_path, caption_path = write_dataset_fixture(tmp_path)
    calls: dict[str, object] = {}

    class FramePlanner:
        max_rgb_frames = 4
        target_fps = 1
        vae_temporal_stride = 4

        def build_frame_plan(self, shot_groups, row_label, *, source_fps, source_num_frames):
            return {
                "shot_groups": [[0, 1]],
                "latent_shot_groups": [[0, 1]],
                "target_fps": self.target_fps,
                "max_rgb_frames": self.max_rgb_frames,
                "vae_temporal_stride": self.vae_temporal_stride,
                "train_rgb_frames": 1,
                "train_latent_frames": 1,
                "frame_indices": [0],
                "vae_alignment_policy": "legacy_floor_truncate_v1",
            }

    def fake_builder(shot_prompts: list[str], *, prompt_format: str) -> list[str]:
        calls["shot_prompts"] = shot_prompts
        calls["prompt_format"] = prompt_format
        return ["shared-builder-sentinel"]

    monkeypatch.setattr(inference_common, "build_prompts_from_fields", fake_builder)

    case_inputs = inference_common.build_multishot_case_inputs(
        {
            "video_path": str(video_path),
            "caption_path": str(caption_path),
            "shot_groups": "[[0, 4]]",
        },
        FramePlanner(),
        video_column="video_path",
        caption_column="caption_path",
        shot_groups_column="shot_groups",
        schema="multishot_csv_v1",
        row_label="case_001",
        prompt_format=PROMPT_FORMAT_CAPTION_JSON_V1,
        read_video_metadata_fn=lambda path: {"source_fps": 1.0, "source_num_frames": 4, "source_width": 16, "source_height": 16},
    )

    assert case_inputs["prompts"] == ["shared-builder-sentinel"]
    assert calls == {
        "shot_prompts": ["the kettle is placed on a table"],
        "prompt_format": PROMPT_FORMAT_CAPTION_JSON_V1,
    }


def test_inference_case_inputs_reuses_the_explicit_training_mapping(
    tmp_path: Path,
) -> None:
    _, video_path, caption_path = write_dataset_fixture(tmp_path)
    calls: dict[str, object] = {}

    class FramePlanner:
        def build_explicit_frame_plan(self, **kwargs):
            calls.update(kwargs)
            return {
                "shot_groups": [[0, 4]],
                "latent_shot_groups": [[0, 1]],
                "target_fps": 1,
                "max_rgb_frames": 4,
                "vae_temporal_stride": 4,
                "train_rgb_frames": 4,
                "train_latent_frames": 1,
                "frame_indices": [0, 1, 2, 3],
                "vae_alignment_policy": "legacy_floor_truncate_v1",
                "source_shot_segments": [[0, 4]],
            }

    case_inputs = inference_common.build_multishot_case_inputs(
        {
            "video_path": str(video_path),
            "caption_path": str(caption_path),
            "shot_groups": "[[0, 4]]",
            "source_shot_segments": "[[0, 4]]",
            "source_frame_indices": "[0, 1, 2, 3]",
        },
        FramePlanner(),
        video_column="video_path",
        caption_column="caption_path",
        shot_groups_column="shot_groups",
        schema="multishot_csv_explicit_mapping_v2",
        row_label="case_001",
        prompt_format=PROMPT_FORMAT_CAPTION_JSON_V1,
        read_video_metadata_fn=lambda path: {"source_fps": 1.0, "source_num_frames": 4, "source_width": 16, "source_height": 16},
    )

    assert case_inputs["shot_groups"] == [[0, 4]]
    assert case_inputs["source_frame_indices"] == [0, 1, 2, 3]
    assert calls == {
        "source_shot_segments": [[0, 4]],
        "train_shot_groups": [[0, 4]],
        "frame_indices": [0, 1, 2, 3],
        "data_path": str(video_path),
        "source_fps": 1.0,
        "source_num_frames": 4,
        "vae_alignment_policy": "legacy_floor_truncate_v1",
    }
