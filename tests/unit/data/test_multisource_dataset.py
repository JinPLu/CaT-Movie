from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from mspc.data.frame_planner import FpsAwareTargetFpsPlanner
from mspc.data.multishot import (
    MULTISHOT_CSV_MULTISOURCE_MAPPING_SCHEMA,
    MultishotCsvDataset,
)
from mspc.data.shot_boundary_revision import SHOT_BOUNDARY_REVISION_VERSION
from tests.support.multisource_mapping import build_unscaled_multisource_mapping


@pytest.fixture(autouse=True)
def _fake_torchvision(monkeypatch: pytest.MonkeyPatch) -> None:
    transforms = ModuleType("torchvision.transforms")
    transforms.CenterCrop = lambda size: (lambda value: value)  # type: ignore[attr-defined]
    torchvision = ModuleType("torchvision")
    torchvision.transforms = transforms  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", transforms)


def _write_multisource_csv(path: Path, *, shot_count: int, case_id: str) -> list[Path]:
    media = [path.parent / f"shot{index}.mp4" for index in range(shot_count)]
    for clip in media:
        clip.write_bytes(b"fixture")
    caption = path.parent / f"{case_id}.json"
    caption.write_text(
        json.dumps(
            {
                "global_caption": "story",
                **{f"shot{index}": f"shot-{index}" for index in range(shot_count)},
            }
        ),
        encoding="utf-8",
    )
    planner = FpsAwareTargetFpsPlanner(
        max_rgb_frames=308,
        target_fps=15,
        vae_temporal_stride=4,
    )
    source_segments = [[0, 40] for _ in range(shot_count)]
    plan = build_unscaled_multisource_mapping(
        source_video_paths=[str(item) for item in media],
        source_shot_segments=source_segments,
        source_fps_by_shot=[15.0] * shot_count,
        source_num_frames_by_shot=[40] * shot_count,
        planner=planner,
        shot_cut_evidence=[
            {"version": SHOT_BOUNDARY_REVISION_VERSION, "source_video_path": str(item),
             "source_shot_segment": [0, 40], "source_fps": 15.0, "content_diff_cuts": [], "transnet_cuts": []}
            for item in media
        ],
        label=case_id,
    )
    assert plan is not None
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "multishot_video",
                "gemini_caption",
                "shot_groups",
                "source_shot_segments",
                "source_video_paths",
                "source_frame_indices_by_shot",
                "case_id",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "multishot_video": str(media[0]),
                "gemini_caption": str(caption),
                "shot_groups": json.dumps(plan["shot_groups"], separators=(",", ":")),
                "source_shot_segments": json.dumps(
                    plan["source_shot_segments"],
                    separators=(",", ":"),
                ),
                "source_video_paths": json.dumps(
                    plan["source_video_paths"],
                    separators=(",", ":"),
                ),
                "source_frame_indices_by_shot": json.dumps(
                    plan["frame_indices_by_shot"],
                    separators=(",", ":"),
                ),
                "case_id": case_id,
            }
        )
    return media


def _dataset_kwargs(metadata_path: Path) -> dict[str, object]:
    return {
        "metadata_path": str(metadata_path),
        "repeat": 1,
        "height": 32,
        "width": 32,
        "frame_policy": "fps_aware_target_fps",
        "max_rgb_frames": 308,
        "target_fps": 15,
        "vae_temporal_stride": 4,
        "schema": MULTISHOT_CSV_MULTISOURCE_MAPPING_SCHEMA,
        "prompt_format": "caption_json_v1",
        "video_column": "multishot_video",
        "caption_column": "gemini_caption",
        "shot_groups_column": "shot_groups",
        "case_id_column": "case_id",
        "case_id_prefix": "case",
    }


@pytest.mark.parametrize("shot_count", [2, 3, 4, 5])
def test_multisource_row_reader_smoke_k2_to_k5(
    tmp_path: Path,
    shot_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("mspc.data.cache_provenance.importlib.metadata.version", lambda _: "test")
    metadata = tmp_path / f"k{shot_count}.csv"
    case_id = f"ci_vid:k{shot_count}"
    media = _write_multisource_csv(metadata, shot_count=shot_count, case_id=case_id)
    dataset = MultishotCsvDataset(**_dataset_kwargs(metadata))
    metadata_by_path = {
        str(path): {
            "source_fps": 15.0,
            "source_num_frames": 40,
            "source_height": 32,
            "source_width": 32,
        }
        for path in media
    }
    monkeypatch.setattr(
        "mspc.data.multishot.read_video_metadata",
        lambda path: metadata_by_path[path],
    )
    decode_calls: list[tuple[str, list[int]]] = []

    def fake_read_video(
        self: MultishotCsvDataset,
        path: str,
        frame_indices: list[int],
        *,
        video_metadata: dict[str, int | float] | None = None,
    ) -> torch.Tensor:
        decode_calls.append((path, list(frame_indices)))
        return torch.zeros(3, len(frame_indices), 32, 32)

    monkeypatch.setattr(MultishotCsvDataset, "_read_video", fake_read_video)

    sample = dataset[0]

    assert sample["case_id"] == case_id
    assert len(sample["prompt"]) == shot_count
    assert len(sample["latent_shot_groups"]) == shot_count
    assert sample["source_video_paths"] == [str(path) for path in media]
    assert len(sample["source_frame_indices_by_shot"]) == shot_count
    assert len(decode_calls) == shot_count
    assert {path for path, _ in decode_calls} == {str(path) for path in media}
    assert sample["video"].shape[1] == sum(end - start for start, end in sample["shot_groups"])


def test_multisource_missing_shot_file_fails_loud(tmp_path: Path) -> None:
    metadata = tmp_path / "missing.csv"
    _write_multisource_csv(metadata, shot_count=2, case_id="ci_vid:missing")
    second = tmp_path / "shot1.mp4"
    second.unlink()
    dataset = MultishotCsvDataset(**_dataset_kwargs(metadata))

    with pytest.raises(FileNotFoundError):
        dataset[0]
