import math

import numpy as np
import pytest

from mspc.data import geometry_annotation as geo


def _look_at(center, target, up=(0.0, -1.0, 0.0)):
    """OpenCV 外参：相机在 center、看向 target，世界 -y 为上。"""

    center = np.asarray(center, float)
    forward = np.asarray(target, float) - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, float) * -1.0)
    right = -right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    return np.concatenate([rotation, (-rotation @ center)[:, None]], axis=1)


K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 180.0], [0.0, 0.0, 1.0]])


def test_clip_and_vggt_positions():
    clip = geo.clip_positions(100)
    assert len(clip) == geo.CLIP_MAX_FRAMES and clip[0] == geo.CLIP_TRIM and clip[-1] == 100 - geo.CLIP_TRIM - 1
    assert geo.clip_positions(30) == list(range(2, 28))
    assert geo.clip_positions(9) == list(range(9))
    with pytest.raises(ValueError):
        geo.clip_positions(7)
    vggt = geo.vggt_positions(26)
    assert len(vggt) == geo.VGGT_FRAMES and vggt[0] == 0 and vggt[-1] == 25


def test_letterbox_box_crops_black_bars():
    frames = np.zeros((2, 100, 200, 3), dtype=np.uint8)
    frames[:, 12:88, :, :] = 120
    assert geo.letterbox_box(frames) == (0, 12, 200, 88)
    frames[:] = 0
    frames[:, :, :, :] = 0
    frames[:, 40:60, 90:110] = 200  # 内容只占中心小块：不当黑边裁
    assert geo.letterbox_box(frames) == (0, 0, 200, 100)


def test_reprojection_round_trip_is_exact():
    first = _look_at((0, 0, 0), (0, 0, 5))
    second = _look_at((1.0, 0, 0.5), (0, 0, 5))
    world = np.array([[0.3, -0.2, 5.0], [-1.0, 0.5, 6.0], [0.8, 0.1, 4.0]])
    pa, za = geo.project(world, K, first)
    pb, _ = geo.project(world, K, second)
    errors = geo.reprojection_errors(pa, za, (K, first), pb, (K, second))
    assert np.allclose(errors, 0.0, atol=1e-6)
    assert np.allclose(geo.backproject(pa, za, K, first), world)
    assert np.allclose(geo.camera_center(second), [1.0, 0, 0.5])


def test_camera_axes_follow_opencv():
    extrinsic = _look_at((0, 0, 0), (0, 0, 5))
    assert np.allclose(geo.camera_forward(extrinsic), [0, 0, 1])
    assert np.allclose(geo.camera_up(extrinsic), [0, -1, 0])


def test_axis_side_flips_when_camera_crosses_the_line():
    up = np.array([0.0, -1.0, 0.0])
    first, second = np.array([-1.0, 0, 5]), np.array([1.0, 0, 5])
    same = geo.axis_side(first, second, np.array([0.0, 0, 0]), up)
    reverse = geo.axis_side(first, second, np.array([0.5, 0, 9]), up)
    assert geo.side_label(same) == -geo.side_label(reverse)
    assert geo.side_label(geo.axis_side(first, second, np.array([5.0, 0, 5.05]), up)) is None


def test_axis_side_matches_screen_order_for_two_shots():
    """两人都在画面里时，侧别与画面左右顺序一一对应（180 度规则的几何本质）。"""

    first, second = np.array([-1.0, 0, 5]), np.array([1.0, 0, 5])
    up = np.array([0.0, -1.0, 0.0])
    for center in [(0, 0, 0), (2, 0, 1), (-2, 0, 1), (0, 0, 10), (3, 0, 9)]:
        extrinsic = _look_at(center, (0, 0, 5))
        pixels, _ = geo.project(np.stack([first, second]), K, extrinsic)
        order = geo.screen_order(pixels[0, 0], pixels[1, 0], 640)
        side = geo.side_label(geo.axis_side(first, second, np.asarray(center, float), up))
        assert order == (1 if side == -1 else -1)


def test_axis_frame_camera_reports_side_and_distance():
    first, second = np.array([-1.0, 0, 5]), np.array([1.0, 0, 5])
    extrinsic = _look_at((0, 0, 1), (0, 0, 5))
    frame = geo.axis_frame_camera(first, second, geo.camera_center(extrinsic), geo.camera_forward(extrinsic),
                                  geo.camera_up(extrinsic))
    assert frame["x"] == pytest.approx(0.0) and abs(frame["y"]) == pytest.approx(2.0)
    assert frame["distance_ratio"] == pytest.approx(2.0)
    assert abs(frame["yaw_deg"]) == pytest.approx(90.0)
    assert math.copysign(1, frame["y"]) == math.copysign(1, frame["side_sine"])


def test_rotation_helpers():
    a = _look_at((0, 0, 0), (0, 0, 5))[:, :3]
    b = _look_at((0, 0, 0), (5, 0, 5))[:, :3]
    assert geo.rotation_angle_deg(a, b) == pytest.approx(45.0)
    steady = [_look_at((0, 0, 0), (math.tan(math.radians(i)) * 5, 0, 5)) for i in range(5)]
    assert geo.rotation_jitter_deg(steady) == pytest.approx(0.0, abs=1e-3)


def test_screen_order_and_depth_order_thresholds():
    assert geo.screen_order(100, 300, 640) == 1
    assert geo.screen_order(300, 100, 640) == -1
    assert geo.screen_order(300, 310, 640) is None
    assert geo.depth_order(2.0, 4.0) == 1
    assert geo.depth_order(2.0, 2.1) is None


def test_spearman():
    a = np.arange(10, dtype=float)
    assert geo.spearman(a, a ** 2) == pytest.approx(1.0)
    assert geo.spearman(a, -a) == pytest.approx(-1.0)
