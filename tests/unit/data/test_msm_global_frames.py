from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from mspc.data.msm_global_frames import (
    MAX_TOTAL_FRAMES,
    RESIZE_HEIGHT,
    RESIZE_WIDTH,
    allocate_global_frame_counts,
    encode_msm_global_jpeg,
    extract_msm_global_frames,
    extract_msm_shot_frames,
    map_concat_indices_to_sources,
    plan_msm_global_indices,
    plan_shot_sample_indices,
)


def test_short_shot_keeps_every_frame() -> None:
    assert plan_shot_sample_indices(0, 4) == [0, 1, 2, 3]


def test_interval_sample_then_pad_to_five() -> None:
    indices = plan_shot_sample_indices(0, 40)
    assert indices[0] == 0
    assert 15 in indices
    assert 30 in indices
    assert len(indices) >= 5


def test_allocation_copies_msm_leftover_behavior() -> None:
    counts = allocate_global_frame_counts([4, 96], [4, 7])
    assert counts[0] == 2
    assert counts[1] == 7
    assert sum(counts) <= MAX_TOTAL_FRAMES


def test_global_plan_caps_at_twenty() -> None:
    plan = plan_msm_global_indices([[0, 90], [90, 180], [180, 270]])
    assert plan["n_frames"] <= MAX_TOTAL_FRAMES
    assert plan["n_frames"] >= 6
    assert plan["spec"]["resize_width"] == 672
    assert plan["spec"]["aspect_ratio"] == "stretch_to_672x384"


def test_encode_records_stretch(tmp_path: Path) -> None:
    del tmp_path
    jpeg, meta = encode_msm_global_jpeg(Image.new("RGB", (100, 50), color=(12, 34, 56)))
    assert jpeg
    assert meta["source_width"] == 100
    assert meta["source_height"] == 50
    assert meta["output_width"] == RESIZE_WIDTH
    assert meta["output_height"] == RESIZE_HEIGHT
    assert meta["stretched"] is True


def test_reject_empty_shot_groups() -> None:
    with pytest.raises(ValueError, match="shot_groups"):
        plan_msm_global_indices([])


def test_map_concat_indices_onto_per_clip_files() -> None:
    mapped = map_concat_indices_to_sources(
        [0, 29, 30, 59],
        shot_groups=[(0, 30), (30, 60)],
        source_video_paths=["/a.mp4", "/b.mp4"],
        source_shot_segments=[[0, 30], [0, 30]],
    )
    assert mapped == [("/a.mp4", 0), ("/a.mp4", 29), ("/b.mp4", 0), ("/b.mp4", 29)]


def test_resampled_ci_vid_uses_source_frame_indices_by_shot() -> None:
    mapped = map_concat_indices_to_sources(
        [0, 28, 29],
        shot_groups=[(0, 29), (29, 50)],
        source_video_paths=["/a.mp4", "/b.mp4"],
        source_shot_segments=[[0, 47], [0, 40]],
        source_frame_indices_by_shot=[list(range(29)), list(range(21))],
    )
    assert mapped[0] == ("/a.mp4", 0)
    assert mapped[1][0] == "/a.mp4"
    assert mapped[2][0] == "/b.mp4"
    with pytest.raises(ValueError, match="refusing to invent"):
        map_concat_indices_to_sources(
            [0],
            shot_groups=[(0, 30), (30, 90)],
            source_video_paths=["/a.mp4", "/b.mp4"],
            source_shot_segments=[[0, 30], [0, 30]],
        )


def test_multisource_extract_reads_each_clip(tmp_path: Path) -> None:
    first = tmp_path / "shot0.mp4"
    second = tmp_path / "shot1.mp4"
    first.write_bytes(b"clip-a")
    second.write_bytes(b"clip-b")
    seen: list[tuple[str, list[int]]] = []

    def reader(path: Path, indices: list[int]) -> list[Image.Image]:
        seen.append((Path(path).name, list(indices)))
        return [Image.new("RGB", (12, 8), color=(len(indices), 0, 0)) for _ in indices]

    extracted = extract_msm_global_frames(
        video_path=first,
        shot_groups=[[0, 30], [30, 60]],
        source_video_paths=[str(first), str(second)],
        source_shot_segments=[[0, 30], [0, 30]],
        frame_reader=reader,
    )
    assert extracted["media_layout"] == "multisource"
    assert extracted["n_frames"] <= MAX_TOTAL_FRAMES
    assert {name for name, _ in seen} == {"shot0.mp4", "shot1.mp4"}
    assert all(0 <= index < 30 for _, indices in seen for index in indices)


def test_shot_extract_samples_only_that_shot(tmp_path: Path) -> None:
    first = tmp_path / "shot0.mp4"
    second = tmp_path / "shot1.mp4"
    first.write_bytes(b"clip-a")
    second.write_bytes(b"clip-b")
    seen: list[tuple[str, list[int]]] = []

    def reader(path: Path, indices: list[int]) -> list[Image.Image]:
        seen.append((Path(path).name, list(indices)))
        return [Image.new("RGB", (12, 8), color=(1, 0, 0)) for _ in indices]

    extracted = extract_msm_shot_frames(
        video_path=first,
        shot_groups=[[0, 40], [40, 80]],
        shot_index=1,
        source_video_paths=[str(first), str(second)],
        source_shot_segments=[[0, 40], [0, 40]],
        frame_reader=reader,
    )
    assert extracted["shot_index"] == 1
    assert extracted["n_frames"] >= 5
    assert all(name == "shot1.mp4" for name, _ in seen)
    assert all(index < 40 for _, indices in seen for index in indices)
    with pytest.raises(ValueError, match="shot_index"):
        extract_msm_shot_frames(
            video_path=first,
            shot_groups=[[0, 40], [40, 80]],
            shot_index=2,
            frame_reader=reader,
        )
