import pytest

from mspc.data import frames


def test_proxy_frames_keep_the_first_and_last_frame_of_every_shot():
    assert frames.proxy_frames([[0, 21], [21, 186]], 8)[0] == [0, 8, 16, 20]
    assert frames.proxy_frames([[0, 17]], 8) == [[0, 8, 16]]


def test_shot_grid_matches_training_frame_mapping_and_keeps_both_ends():
    grid = frames.shot_grid(100, 181, 41)
    assert grid[0] == 100 and grid[-1] == 180 and len(set(grid)) == 41
    with pytest.raises(ValueError):
        frames.shot_grid(0, 10, 11)
