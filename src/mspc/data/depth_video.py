"""逐镜深度与置信度的存法：8 位量化后写成无损灰度视频，读回逐元素比对。

深度取对数再线性量化到这一镜的最小最大值之间，置信度直接线性量化；码值 1..levels+1，``0`` 留给无效。
视频是 H.264 ``-qp 0``（无损），ffmpeg 经管道喂原始帧、经管道读回原始帧；写完立刻读回与量化后的数组比对，
不一致就报错——无损是这里自证的，不是假定的。深度的绝对尺度是相对的，跨镜不可比（由调用方写进元数据）。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def _check_levels(levels):
    if type(levels) is not int or not 1 <= levels <= 254:
        raise ValueError(f"levels must be an int in 1..254 (code 0 is reserved for invalid), got {levels!r}")


def encode_log_depth(depth, levels):
    """深度（正数、有限）→ (uint8 码值, near, far)；``near``、``far`` 是这一镜深度的最小、最大值。

    ``v = 1 + round(levels * (ln d - ln near) / (ln far - ln near))``；near == far 时全部为 1。
    """
    _check_levels(levels)
    depth = np.asarray(depth)
    if not np.isfinite(depth).all() or (depth <= 0).any():
        raise ValueError("depth must be finite and positive")
    near, far = float(depth.min()), float(depth.max())
    span = np.log(far) - np.log(near)
    if span == 0:
        return np.ones(depth.shape, dtype=np.uint8), near, far
    scaled = levels * (np.log(depth.astype(np.float64)) - np.log(near)) / span
    return (1 + np.rint(scaled)).astype(np.uint8), near, far


def decode_log_depth(codes, near, far, levels):
    """``encode_log_depth`` 的反量化，返回 float64；码值 0（无效）报错。"""
    _check_levels(levels)
    codes = np.asarray(codes)
    if codes.dtype != np.uint8 or (codes == 0).any() or (codes > levels + 1).any():
        raise ValueError("depth codes must be uint8 in 1..levels+1")
    span = np.log(far) - np.log(near)
    return np.exp(np.log(near) + (codes.astype(np.float64) - 1) / levels * span)


def encode_linear(values, levels):
    """置信度（有限）→ (uint8 码值, 最小值, 最大值)：``v = 1 + round(levels * (c - min) / (max - min))``。"""
    _check_levels(levels)
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("values must be finite")
    low, high = float(values.min()), float(values.max())
    if high == low:
        return np.ones(values.shape, dtype=np.uint8), low, high
    scaled = levels * (values.astype(np.float64) - low) / (high - low)
    return (1 + np.rint(scaled)).astype(np.uint8), low, high


def decode_linear(codes, low, high, levels):
    _check_levels(levels)
    codes = np.asarray(codes)
    if codes.dtype != np.uint8 or (codes == 0).any() or (codes > levels + 1).any():
        raise ValueError("codes must be uint8 in 1..levels+1")
    return low + (codes.astype(np.float64) - 1) / levels * (high - low)


def max_log_depth_error(near, far, levels):
    """深度量化的最大误差：ln 深度上半个台阶 ``(ln far - ln near) / levels / 2``，与相对误差 ``exp(·) - 1``。"""
    half = (np.log(far) - np.log(near)) / levels / 2
    return float(half), float(np.expm1(half))


def max_linear_error(low, high, levels):
    return float((high - low) / levels / 2)


def _fail(action, path, process):
    tail = process.stderr.decode(errors="replace")[-2000:]
    raise RuntimeError(f"ffmpeg {action} failed for {path} (exit {process.returncode}): {tail}")


def write_gray_video(ffmpeg, codes, path, *, frame_rate, codec, preset, qp, pixel_format):
    """uint8 [N, H, W] → 灰度视频，原始帧经 stdin 管道喂 ffmpeg；先写 ``.partial`` 再换名。只接受 ``qp == 0``（无损）。"""
    if qp != 0:
        raise ValueError("lossless storage requires qp 0")
    codes = np.ascontiguousarray(codes)
    if codes.dtype != np.uint8 or codes.ndim != 3 or 0 in codes.shape:
        raise ValueError("codes must be a non-empty uint8 [frames, height, width] array")
    count, height, width = codes.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    command = [str(ffmpeg), "-v", "error", "-nostdin", "-y", "-threads", "1",
               "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}", "-r", str(frame_rate), "-i", "pipe:0",
               "-an", "-c:v", codec, "-preset", preset, "-qp", str(qp), "-pix_fmt", pixel_format,
               "-fps_mode", "passthrough", "-threads", "1", "-f", "mp4", str(partial)]
    process = subprocess.run(command, input=codes.tobytes(), capture_output=True)
    if process.returncode != 0:
        _fail("encode", path, process)
    partial.replace(path)
    return path


def read_gray_video(ffmpeg, path, shape):
    """灰度视频 → uint8 [N, H, W]（``shape`` 是写入时的形状，帧数不符报错）。"""
    count, height, width = shape
    command = [str(ffmpeg), "-v", "error", "-nostdin", "-threads", "1", "-i", str(path),
               "-an", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    process = subprocess.run(command, capture_output=True)
    if process.returncode != 0:
        _fail("decode", path, process)
    if len(process.stdout) != count * height * width:
        raise ValueError(f"{path}: decoded {len(process.stdout)} bytes, expected {count * height * width} for {shape}")
    return np.frombuffer(process.stdout, dtype=np.uint8).reshape(shape)


def write_verified(ffmpeg, codes, path, **storage):
    """写视频并读回比对；与量化后的数组有任何一个元素不同就报错（并留下视频供查看）。"""
    write_gray_video(ffmpeg, codes, path, **storage)
    back = read_gray_video(ffmpeg, path, codes.shape)
    if not np.array_equal(back, codes):
        raise ValueError(f"{path}: decoded video differs from the quantized array in {int((back != codes).sum())} elements")
    return path
