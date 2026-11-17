"""镜界修订：真实失败切片（floor 200 窗 TransNetV2 实测切点）驱动区间生产端。"""
import json
from pathlib import Path

import numpy as np
import pytest

from mspc.data.frame_planner import FpsAwareTargetFpsPlanner
from mspc.data.scene_shot_manifest import TRANSNET_THRESHOLD
from mspc.data.shot_boundary_revision import (
    SHOT_BOUNDARY_REVISION_VERSION,
    TRANSNET_ISOLATION_PAD,
    ShotBoundaryRejected,
    transnet_cuts_for_shots,
)
from tests.support.multisource_mapping import (
    build_unscaled_multisource_mapping,
    select_one_contiguous_multisource_window,
)

ROOT = Path(__file__).resolve().parents[3]
WINDOWS = json.loads((ROOT / "tests/fixtures/shot_boundary_revision/floor_windows.json").read_text())["windows"]
PLANNER = FpsAwareTargetFpsPlanner(max_rgb_frames=308, target_fps=15, vae_temporal_stride=4)


def _inputs(window_id: str, shots: slice = slice(None)) -> dict:
    chosen = WINDOWS[window_id]["shots"][shots]
    return {
        "source_video_paths": [shot["source_video_path"] for shot in chosen],
        "source_shot_segments": [shot["source_shot_segment"] for shot in chosen],
        "source_fps_by_shot": [shot["source_fps"] for shot in chosen],
        "source_num_frames_by_shot": [shot["source_num_frames"] for shot in chosen],
        "shot_cut_evidence": [
            {
                "version": SHOT_BOUNDARY_REVISION_VERSION,
                "source_video_path": shot["source_video_path"],
                "source_shot_segment": shot["source_shot_segment"],
                "source_fps": shot["source_fps"],
                "transnet_cuts": shot["transnet_cuts"],
                "transnet_threshold": TRANSNET_THRESHOLD,
            }
            for shot in chosen
        ],
    }


def _plan(window_id: str) -> dict:
    plan = build_unscaled_multisource_mapping(**_inputs(window_id), planner=PLANNER, label=window_id)
    assert plan is not None
    return plan


def test_ci_vid_frame0_residual_of_previous_shot_is_trimmed():
    plan = _plan("floor-0028")
    assert plan["source_shot_segments"] == [[0, 49], [1, 85]]
    assert plan["shot_boundary_revision"]["trimmed_head_frames"] == [0, 1]
    assert plan["frame_indices_by_shot"][1][0] == 1


def test_ci_vid_multi_frame_head_residuals_are_trimmed_per_shot():
    plan = _plan("floor-0006")
    assert plan["shot_boundary_revision"]["trimmed_head_frames"] == [4, 1, 12]
    assert [indices[0] for indices in plan["frame_indices_by_shot"]] == [4, 1, 12]


def test_cine250k_two_frame_tail_mix_is_trimmed_not_just_one_frame():
    plan = _plan("floor-0062")
    assert plan["source_shot_segments"] == [[6349, 6422], [6424, 6494], [6496, 6549], [6549, 6633]]
    assert plan["shot_boundary_revision"]["trimmed_tail_frames"] == [2, 2, 0, 0]
    assert [indices[-1] for indices in plan["frame_indices_by_shot"]] == [6421, 6493, 6548, 6632]


def test_cine250k_one_frame_tail_mix_and_clean_last_shot_keep_its_final_frame():
    plan = _plan("floor-0000")
    assert plan["source_shot_segments"] == [[50299, 50407], [50408, 50474], [50475, 50572]]
    assert [indices[-1] for indices in plan["frame_indices_by_shot"]] == [50406, 50473, 50571]
    assert all(len(set(indices)) == len(indices) for indices in plan["frame_indices_by_shot"])


def test_hand_occlusion_without_detected_cut_is_left_untouched():
    plan = _plan("floor-0022")
    assert plan["source_shot_segments"] == [[0, 193], [0, 228]]
    assert plan["shot_boundary_revision"]["trimmed_head_frames"] == [0, 0]
    assert plan["shot_boundary_revision"]["trimmed_tail_frames"] == [0, 0]
    assert [indices[-1] for indices in plan["frame_indices_by_shot"]] == [192, 227]


def test_mid_shot_dissolve_rejects_the_whole_window():
    with pytest.raises(ShotBoundaryRejected) as caught:
        build_unscaled_multisource_mapping(**_inputs("floor-0055"), planner=PLANNER, label="floor-0055")
    assert caught.value.reason == "mid_shot_cut"
    assert "shot 0" in str(caught.value)


def test_window_selection_skips_the_rejected_shot():
    windows = select_one_contiguous_multisource_window(
        **_inputs("floor-0055"), planner=PLANNER, min_shots=2, label="floor-0055"
    )
    assert [indices for indices, _plan in windows] == [[1, 2]]


def test_evidence_measured_on_another_span_is_refused():
    inputs = _inputs("floor-0028")
    inputs["shot_cut_evidence"][1] = {**inputs["shot_cut_evidence"][1], "source_shot_segment": [0, 84]}
    with pytest.raises(ValueError, match="different span"):
        build_unscaled_multisource_mapping(**inputs, planner=PLANNER, label="floor-0028")


class _PeakEngine:
    """Records the stacked input and puts a TransNet peak on every hard switch in it."""

    def predict(self, frames):
        self.frames = np.asarray(frames)
        values = self.frames.reshape(len(self.frames), -1).mean(axis=1)
        return [0.9 if i + 1 < len(values) and values[i + 1] != values[i] else 0.0 for i in range(len(values))]


def test_shots_sharing_one_transnet_call_are_isolated_by_padding():
    first = np.concatenate([np.full((1, 2, 2, 3), 10, np.uint8), np.full((5, 2, 2, 3), 200, np.uint8)])
    second = np.full((4, 2, 2, 3), 50, np.uint8)
    engine = _PeakEngine()
    cuts = transnet_cuts_for_shots([first, second], starts=[100, 0], engine=engine)
    assert cuts == [[101], []]
    assert len(engine.frames) == len(first) + len(second) + 4 * TRANSNET_ISOLATION_PAD
