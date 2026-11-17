"""逐镜潜帧分组到连续token区间的合同。"""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third_party/wan22_runtime"))
from diffsynth.pipelines.wan_video import latent_shot_groups_to_token_ranges


def test_reference_frames_and_target_shots_form_contiguous_ranges():
    actual = latent_shot_groups_to_token_ranges([(0, 1), (1, 2), (2, 5), (5, 9)], 9, 2, 3)
    assert actual == ((0, 6), (6, 12), (12, 30), (30, 54))


@pytest.mark.parametrize("groups", [[(1, 3)], [(0, 2), (1, 4)], [(0, 2), (3, 4)], [(0, 0), (0, 4)]])
def test_token_ranges_reject_gaps_overlaps_or_empty_shots(groups):
    with pytest.raises(ValueError, match="partition"):
        latent_shot_groups_to_token_ranges(groups, 4, 2, 3)


def test_token_ranges_reject_uncovered_final_frame():
    with pytest.raises(ValueError, match="patchified token shape"):
        latent_shot_groups_to_token_ranges([(0, 3)], 4, 2, 3)
