"""几何标注试点的纯函数：取帧位置、相机几何、重投影核验、轴线侧与人物连线坐标系。

相机约定沿用 VGGT 输出：
外参 ``E = [R | t]`` 把世界点变到相机坐标（OpenCV：x 向右、y 向下、z 向前），内参 ``K`` 为像素单位。
本模块只依赖 numpy，GPU 与模型调用在 ``tools/run/geometry_annotation_pilot.py``。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

#: 一镜的跟踪片段与 VGGT 取帧。
CLIP_TRIM = 2
CLIP_MAX_FRAMES = 48
VGGT_FRAMES = 8
MID_POSITION = 4
#: 判据（声明里写死）。
TRACK_MIN_POINTS = 30
TRACK_PASS_ERROR = 0.02
MOVING_DISPLACEMENT = 0.01
SCREEN_MIN_GAP = 0.05
AXIS_MIN_SINE = 0.1
REGISTER_MIN_INLIERS = 20
REGISTER_PASS_ERROR = 0.03
PERSON_PASS_RATIO = 0.25
#: 镜内指认（主体短属性词得分的一对一指派，见 :func:`mspc.data.spatial_facts.assign_one_to_one`）的得分下限。
BIND_MIN_SCORE = 0.35
DEPTH_AGREE_RHO = 0.7
DEPTH_ORDER_MIN_LOG = 0.1


def clip_positions(frame_count: int) -> list[int]:
    """一镜训练帧里取作跟踪片段的位置：去首尾各 ``CLIP_TRIM`` 帧，均匀取不超过 ``CLIP_MAX_FRAMES`` 帧。"""

    if frame_count < VGGT_FRAMES + 2 * CLIP_TRIM:
        if frame_count < VGGT_FRAMES:
            raise ValueError(f"shot has {frame_count} frames, fewer than {VGGT_FRAMES}")
        return list(range(frame_count))
    usable = list(range(CLIP_TRIM, frame_count - CLIP_TRIM))
    if len(usable) <= CLIP_MAX_FRAMES:
        return usable
    picks = np.linspace(0, len(usable) - 1, CLIP_MAX_FRAMES)
    return [usable[int(round(p))] for p in picks]


def vggt_positions(clip_length: int) -> list[int]:
    """跟踪片段里给 VGGT 与 SAM 3.1 的 8 个位置；第 0 个就是跟踪查询帧。"""

    if clip_length < VGGT_FRAMES:
        raise ValueError(f"clip has {clip_length} frames, fewer than {VGGT_FRAMES}")
    positions = [int(round(p)) for p in np.linspace(0, clip_length - 1, VGGT_FRAMES)]
    if len(set(positions)) != VGGT_FRAMES:
        raise ValueError(f"duplicate VGGT positions for clip length {clip_length}: {positions}")
    return positions


def letterbox_box(frames: np.ndarray, *, threshold: float = 20.0, max_fraction: float = 0.3) -> tuple[int, int, int, int]:
    """多帧上逐行逐列取最大亮度，四周低于阈值的行列视为黑边。返回 (x0, y0, x1, y1)，右下开区间。"""

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (T, H, W, 3), got {frames.shape}")
    luminance = frames.astype(np.float32).mean(axis=-1).max(axis=0)
    height, width = luminance.shape
    rows = np.flatnonzero(luminance.max(axis=1) > threshold)
    cols = np.flatnonzero(luminance.max(axis=0) > threshold)
    if rows.size == 0 or cols.size == 0:
        return 0, 0, width, height
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    if y0 > max_fraction * height or (height - y1) > max_fraction * height:
        y0, y1 = 0, height
    if x0 > max_fraction * width or (width - x1) > max_fraction * width:
        x0, x1 = 0, width
    return x0, y0, x1, y1


def camera_center(extrinsic: np.ndarray) -> np.ndarray:
    rotation, translation = extrinsic[:3, :3], extrinsic[:3, 3]
    return -rotation.T @ translation


def camera_up(extrinsic: np.ndarray) -> np.ndarray:
    """相机向上方向（世界坐标）：OpenCV 的 -y。"""

    return -extrinsic[:3, :3].T @ np.array([0.0, 1.0, 0.0])


def camera_forward(extrinsic: np.ndarray) -> np.ndarray:
    return extrinsic[:3, :3].T @ np.array([0.0, 0.0, 1.0])


def backproject(pixels: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """像素 (N, 2) 加深度 (N,) 到世界点 (N, 3)。"""

    homogeneous = np.concatenate([pixels, np.ones((len(pixels), 1))], axis=1)
    rays = homogeneous @ np.linalg.inv(intrinsic).T
    camera_points = rays * depth[:, None]
    rotation, translation = extrinsic[:3, :3], extrinsic[:3, 3]
    return (camera_points - translation) @ rotation


def project(points: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """世界点 (N, 3) 投到像素 (N, 2)，另返回相机坐标深度 (N,)。"""

    camera_points = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    z = camera_points[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    pixels = (camera_points @ intrinsic.T)[:, :2] / safe[:, None]
    return pixels, z


def reprojection_errors(pixels_a: np.ndarray, depth_a: np.ndarray, camera_a: tuple[np.ndarray, np.ndarray],
                        pixels_b: np.ndarray, camera_b: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """A 帧像素按 A 的深度与相机抬到世界，再投到 B，与 B 的观测像素比。返回逐点像素误差；投到 B 身后的点记 inf。"""

    intrinsic_a, extrinsic_a = camera_a
    intrinsic_b, extrinsic_b = camera_b
    world = backproject(pixels_a, depth_a, intrinsic_a, extrinsic_a)
    projected, z = project(world, intrinsic_b, extrinsic_b)
    errors = np.linalg.norm(projected - pixels_b, axis=1)
    return np.where(z > 0, errors, np.inf)


def sample_map(values: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """最近邻取图上值；越界报错。``pixels`` 为 (x, y)。"""

    height, width = values.shape[:2]
    cols = np.rint(pixels[:, 0]).astype(int)
    rows = np.rint(pixels[:, 1]).astype(int)
    if cols.size and (cols.min() < 0 or rows.min() < 0 or cols.max() >= width or rows.max() >= height):
        raise ValueError("pixel outside map")
    return values[rows, cols]


def rotation_angle_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = rotation_b @ rotation_a.T
    cosine = (np.trace(relative) - 1.0) / 2.0
    return float(math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0)))))


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    angle = math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    return axis / (2.0 * math.sin(angle)) * angle


def rotation_jitter_deg(extrinsics: Sequence[np.ndarray]) -> float:
    """相邻帧角速度的二阶差分中位数（度）：手持晃动的粗指标。"""

    rotations = [np.asarray(e)[:3, :3] for e in extrinsics]
    velocity = [rotation_vector(b @ a.T) for a, b in zip(rotations, rotations[1:])]
    if len(velocity) < 2:
        raise ValueError("need at least three cameras")
    second = [np.linalg.norm(b - a) for a, b in zip(velocity, velocity[1:])]
    return float(math.degrees(float(np.median(second))))


def screen_order(x_first: float, x_second: float, width: float) -> int | None:
    """两人画面横向顺序：+1 表示第一人在左；间距不足 ``SCREEN_MIN_GAP`` 画宽返回 None。"""

    if abs(x_second - x_first) < SCREEN_MIN_GAP * width:
        return None
    return 1 if x_first < x_second else -1


def axis_side(first: np.ndarray, second: np.ndarray, center: np.ndarray, up: np.ndarray) -> float:
    """机位在人物连线（first→second）哪一侧：返回带符号的正弦，正负即侧别，绝对值是与连线夹角的正弦。"""

    axis = second - first
    to_camera = center - first
    norm = np.linalg.norm(axis) * np.linalg.norm(to_camera)
    if norm == 0.0:
        raise ValueError("degenerate axis or camera on a subject")
    return float(np.dot(up, np.cross(axis, to_camera)) / norm)


def side_label(sine: float) -> int | None:
    if abs(sine) < AXIS_MIN_SINE:
        return None
    return 1 if sine > 0 else -1


def axis_frame_camera(first: np.ndarray, second: np.ndarray, center: np.ndarray, forward: np.ndarray,
                      up: np.ndarray) -> dict[str, float]:
    """以两人中点为原点、first→second 为 x、向上（正交化）为 z 的坐标系里写机位。

    返回：机位坐标（以两人间距为单位）、侧别正弦、视线与连线的水平夹角（度）、到中点距离与间距之比。
    """

    x_axis = second - first
    spacing = float(np.linalg.norm(x_axis))
    if spacing == 0.0:
        raise ValueError("subjects coincide")
    x_axis = x_axis / spacing
    z_axis = up - np.dot(up, x_axis) * x_axis
    if np.linalg.norm(z_axis) < 1e-6:
        raise ValueError("up is parallel to the subject axis")
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    basis = np.stack([x_axis, y_axis, z_axis])
    origin = (first + second) / 2.0
    position = basis @ (center - origin) / spacing
    look = basis @ forward
    yaw = math.degrees(math.atan2(look[1], look[0]))
    return {"x": float(position[0]), "y": float(position[1]), "z": float(position[2]),
            "side_sine": axis_side(first, second, center, up), "yaw_deg": float(yaw),
            "distance_ratio": float(np.linalg.norm(position[:2]))}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.ndim != 1 or a.size < 3:
        raise ValueError("spearman needs two equal 1-D arrays with at least 3 values")
    rank_a = np.argsort(np.argsort(a)).astype(np.float64)
    rank_b = np.argsort(np.argsort(b)).astype(np.float64)
    rank_a -= rank_a.mean()
    rank_b -= rank_b.mean()
    denominator = math.sqrt(float((rank_a ** 2).sum() * (rank_b ** 2).sum()))
    if denominator == 0.0:
        raise ValueError("constant input")
    return float((rank_a * rank_b).sum() / denominator)


def depth_order(depth_first: float, depth_second: float) -> int | None:
    """前后关系：+1 表示第一人更近；对数比不足 ``DEPTH_ORDER_MIN_LOG`` 返回 None。"""

    if depth_first <= 0 or depth_second <= 0:
        raise ValueError("depth must be positive")
    ratio = math.log(depth_second / depth_first)
    if abs(ratio) < DEPTH_ORDER_MIN_LOG:
        return None
    return 1 if ratio > 0 else -1
