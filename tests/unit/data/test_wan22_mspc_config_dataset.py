from __future__ import annotations

from tests.support.paths import REPO_ROOT
import json
import sys

import pytest



def load_multishot_dataset_module():
    pytest.importorskip("torch")
    sys.path.insert(0, str(REPO_ROOT))
    from mspc.data import multishot

    return multishot


def test_wan22_frame_plan_uses_target_fps_for_latent_lengths():
    dataset = load_multishot_dataset_module()
    planner_10fps = dataset.FpsAwareTargetFpsPlanner(
        max_rgb_frames=128,
        target_fps=10,
        vae_temporal_stride=4,
    )
    planner_20fps = dataset.FpsAwareTargetFpsPlanner(
        max_rgb_frames=128,
        target_fps=20,
        vae_temporal_stride=4,
    )

    plan_10fps = planner_10fps.build_frame_plan(
        [[0, 30], [30, 90]],
        "case-10fps",
        source_fps=30,
        source_num_frames=90,
    )
    plan_20fps = planner_20fps.build_frame_plan(
        [[0, 30], [30, 90]],
        "case-20fps",
        source_fps=30,
        source_num_frames=90,
    )

    assert plan_10fps["latent_lengths"] == [3, 5]
    assert plan_10fps["shot_groups"] == [[0, 9], [9, 26]]
    assert plan_10fps["train_latent_frames"] == 8
    assert plan_20fps["latent_lengths"] == [5, 10]
    assert plan_20fps["shot_groups"] == [[0, 17], [17, 54]]
    assert plan_20fps["train_latent_frames"] == 15


def test_wan22_frame_plan_scales_to_max_frame_budget():
    dataset = load_multishot_dataset_module()
    planner = dataset.FpsAwareTargetFpsPlanner(
        max_rgb_frames=34,
        target_fps=20,
        vae_temporal_stride=4,
    )

    plan = planner.build_frame_plan(
        [[0, 30], [30, 90]],
        "case-scaled",
        source_fps=30,
        source_num_frames=90,
    )

    assert plan["rgb_was_scaled"] is True
    assert plan["train_rgb_frames"] <= 34
    assert plan["latent_shot_groups"][-1][-1] == plan["train_latent_frames"]


def test_wan22_frame_plan_scales_before_vae_alignment():
    dataset = load_multishot_dataset_module()
    planner = dataset.FpsAwareTargetFpsPlanner(
        max_rgb_frames=3,
        target_fps=4,
        vae_temporal_stride=4,
    )

    plan = planner.build_frame_plan(
        [[0, 10], [10, 20], [20, 30]],
        "case-scale-before-align",
        source_fps=10,
        source_num_frames=30,
    )

    assert plan["rgb_was_scaled"] is True
    assert plan["target_rgb_lengths"] == [1, 1, 1]
    assert plan["aligned_rgb_lengths"] == [1, 1, 1]
    assert plan["train_rgb_frames"] == 3


def test_legacy_explicit_mapping_samples_the_aligned_frame_count():
    dataset = load_multishot_dataset_module()
    planner = dataset.FpsAwareTargetFpsPlanner(
        max_rgb_frames=308,
        target_fps=15,
        vae_temporal_stride=4,
    )
    source_segments = [[0, 49], [49, 285]]
    train_groups = [[0, 25], [25, 142]]
    frame_indices = [
        *planner.sample_indices_for_shot(0, 49, 25),
        *planner.sample_indices_for_shot(49, 285, 117),
    ]

    plan = planner.build_explicit_frame_plan(
        source_shot_segments=source_segments,
        train_shot_groups=train_groups,
        frame_indices=frame_indices,
        data_path="legacy-explicit-mapping",
        source_fps=29.97002997002997,
        source_num_frames=285,
    )

    assert plan["target_rgb_lengths"] == [25, 118]
    assert plan["aligned_rgb_lengths"] == [25, 117]
    assert plan["frame_indices"] == frame_indices


def test_build_prompts_requires_one_caption_per_shot(tmp_path):
    dataset = load_multishot_dataset_module()
    caption_path = tmp_path / "caption.json"
    caption_path.write_text(
        json.dumps(
            {
                "global_caption": "a person prepares tea",
                "shot0": "the kettle is placed on a table",
                "shot1": "steam rises from the cup",
            }
        ),
        encoding="utf-8",
    )

    prompts = dataset.build_prompts(str(caption_path), 2, prompt_format="caption_json_v1")

    assert prompts == [
        "the kettle is placed on a table",
        "steam rises from the cup",
    ]
