"""逐镜拼接出片：第 i 镜的种子是 seed + shot_seed_stride·i。"""

from __future__ import annotations

import pytest

from mspc.core.video import generate_story_by_single_shot_concat


def _shots(n: int) -> list[dict]:
    return [{"shot_groups": [[0, 1]], "shot_ids": [f"s{i:02d}"]} for i in range(n)]


def _render(seeds: list[int]):
    def generate_shot(*, shot_inputs, output_path, seed):
        seeds.append(seed)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"x")

    return generate_shot


def _concat(paths, *, output_path, ffmpeg):
    output_path.write_bytes(b"x")


@pytest.mark.parametrize(("stride", "expected"), [(0, [42, 42, 42]), (1000, [42, 1042, 2042])])
def test_each_shot_draws_its_seed_from_the_stride(tmp_path, stride, expected) -> None:
    seeds: list[int] = []
    generate_story_by_single_shot_concat(
        shot_inputs=_shots(3), output_path=tmp_path / "story.mp4", seed=42,
        shot_seed_stride=stride, generate_shot=_render(seeds), concat_clips=_concat,
    )
    assert seeds == expected


def test_a_negative_stride_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="shot_seed_stride"):
        generate_story_by_single_shot_concat(
            shot_inputs=_shots(2), output_path=tmp_path / "story.mp4", seed=42,
            shot_seed_stride=-1, generate_shot=_render([]), concat_clips=_concat,
        )
