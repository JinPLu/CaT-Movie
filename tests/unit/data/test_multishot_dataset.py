from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from mspc.data.multishot import MultishotCsvDataset


@pytest.fixture(autouse=True)
def _fake_torchvision(monkeypatch: pytest.MonkeyPatch) -> None:
    transforms = ModuleType("torchvision.transforms")
    transforms.CenterCrop = lambda size: (lambda value: value)  # type: ignore[attr-defined]
    torchvision = ModuleType("torchvision")
    torchvision.transforms = transforms  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", transforms)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video", "caption", "shot_groups", "case_id"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _kwargs(metadata_path: Path) -> dict[str, object]:
    return {
        "metadata_path": str(metadata_path),
        "repeat": 1,
        "height": 32,
        "width": 32,
        "frame_policy": "fps_aware_target_fps",
        "max_rgb_frames": 9,
        "target_fps": 15,
        "vae_temporal_stride": 4,
        "schema": "multishot_csv_v1",
        "prompt_format": "caption_json_v1",
        "video_column": "video",
        "caption_column": "caption",
        "shot_groups_column": "shot_groups",
        "case_id_column": "case_id",
        "case_id_prefix": "case",
    }


def test_initialization_only_loads_columns_and_unique_case_ids(tmp_path: Path) -> None:
    metadata = tmp_path / "rows.csv"
    _write_csv(
        metadata,
        [{"video": "missing.mp4", "caption": "missing.json", "shot_groups": "[[0, 5]]", "case_id": "a"}],
    )

    dataset = MultishotCsvDataset(**_kwargs(metadata))

    assert dataset.data[0]["_case_id"] == "a"
    assert "_prompts" not in dataset.data[0]
    assert dataset.metadata_summary["source_rows"] == 1
    assert "preflight_checked_rows" not in dataset.metadata_summary


def test_missing_prompt_path_fails_on_actual_read(tmp_path: Path) -> None:
    metadata = tmp_path / "rows.csv"
    _write_csv(
        metadata,
        [{"video": "missing.mp4", "caption": "missing.json", "shot_groups": "[[0, 5]]", "case_id": "a"}],
    )

    dataset = MultishotCsvDataset(**_kwargs(metadata))

    with pytest.raises(FileNotFoundError):
        dataset[0]


def test_actual_read_builds_one_prompt_per_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture")
    caption = tmp_path / "caption.json"
    caption.write_text(
        '{"global_caption":"story","shot0":"first","shot1":"second"}',
        encoding="utf-8",
    )
    metadata = tmp_path / "rows.csv"
    _write_csv(
        metadata,
        [{
            "video": str(video),
            "caption": str(caption),
            "shot_groups": "[[0, 5], [5, 10]]",
            "case_id": "a",
        }],
    )
    dataset = MultishotCsvDataset(**_kwargs(metadata))
    monkeypatch.setattr(
        "mspc.data.multishot.read_video_metadata",
        lambda _: {
            "source_fps": 15.0,
            "source_num_frames": 10,
            "source_height": 32,
            "source_width": 32,
        },
    )
    monkeypatch.setattr(
        dataset,
        "_read_video",
        lambda *args, **kwargs: torch.zeros(3, 10, 32, 32),
    )

    monkeypatch.setattr("mspc.data.cache_provenance.importlib.metadata.version", lambda _: "test")
    sample = dataset[0]

    assert sample["case_id"] == "a"
    assert sample["prompt"] == ["first", "second"]
    assert len(sample["prompt"]) == len(sample["latent_shot_groups"]) == 2
    assert "_sample_metadata" not in sample


def test_duplicate_case_id_is_rejected(tmp_path: Path) -> None:
    metadata = tmp_path / "rows.csv"
    row = {"video": "v.mp4", "caption": "c.json", "shot_groups": "[[0, 5]]", "case_id": "same"}
    _write_csv(metadata, [row, row])

    with pytest.raises(ValueError, match="duplicate case_id"):
        MultishotCsvDataset(**_kwargs(metadata))


def test_sample_failure_is_not_retried_or_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = tmp_path / "rows.csv"
    _write_csv(
        metadata,
        [{"video": "v.mp4", "caption": "c.json", "shot_groups": "[[0, 5]]", "case_id": "a"}],
    )
    dataset = MultishotCsvDataset(**_kwargs(metadata))
    calls = 0

    def fail(_: int) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise RuntimeError("decode failed")

    monkeypatch.setattr(dataset, "_load_sample", fail)

    with pytest.raises(RuntimeError, match="decode failed"):
        dataset[0]
    assert calls == 1


def test_empty_mapping_schema_cell_fails_as_nan_not_yaml_nan(tmp_path: Path) -> None:
    metadata = tmp_path / "rows.csv"
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video",
                "caption",
                "shot_groups",
                "case_id",
                "mapping_schema",
                "source_shot_segments",
                "source_video_paths",
                "source_frame_indices_by_shot",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "video": "v.mp4",
                "caption": "c.json",
                "shot_groups": "[[0, 5]]",
                "case_id": "full:cine250k:cine250k:100019851_7",
                "mapping_schema": "",
                "source_shot_segments": "",
                "source_video_paths": "",
                "source_frame_indices_by_shot": "",
            }
        )
    kwargs = _kwargs(metadata)
    kwargs["schema"] = "multishot_csv_multisource_mapping_v6_boundary_revised"
    with pytest.raises(ValueError, match="is NaN"):
        MultishotCsvDataset(**kwargs)
