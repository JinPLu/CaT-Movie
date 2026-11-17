"""一条源片的第一遍「量」：解码一遍量齐 → 按镜判合格。打分、定线、装窗、代理片在这之后。

先按镜筛、再装窗；整条前置处理只用 CPU 解码；画面差按 PySceneDetect 官方默认。

``decode_measure`` 一个 ffmpeg 进程两路输出：
- 48×27 的 RGB24（官方 TransNetV2 ``predict_video`` 的缩放）→ 常驻 GPU 的切镜网络；同一批小图去掉黑边后取平均得逐帧亮度；
- 原分辨率 RGB24 → PySceneDetect ``ContentDetector`` 的相邻帧画面差（长边缩到 256、``INTER_LINEAR``、HSV 三通道均值）；
- 第一路滤镜链里的 cropdetect 量黑边（它只从四边往里扫到第一条非黑线，不扫整幅）。
"""
from __future__ import annotations

import fcntl
import os
import subprocess
import threading
import time
from typing import Any, Mapping

import numpy as np

from mspc.data.active_box_cropdetect import CROPDETECT, parse_crop

MEASURE_VERSION = "source_measure_v2"
TINY_WIDTH, TINY_HEIGHT = 48, 27
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
# 旧测量层的两个暗场启发式会把暗景当坏镜；亮度范围已接管「过暗」，画面文字由内容筛选判。
IGNORED_SOURCE_FLAGS = ("black", "text_on_black")
# PySceneDetect scene_manager.DEFAULT_MIN_WIDTH：自动缩小后长边的目标。
CONTENT_LONG_SIDE = 256
PIPE_BYTES = 1 << 20  # /proc/sys/fs/pipe-max-size 的默认值


def content_size(width: int, height: int) -> tuple[int, int]:
    """PySceneDetect ``compute_downscale_factor(max(frame_size))`` 之后的画幅；长边不足 256 不缩。"""
    factor = max(width, height) / float(CONTENT_LONG_SIDE)
    if factor <= 1.0:
        return width, height
    return max(1, round(width / factor)), max(1, round(height / factor))


def decode_measure(video: str, *, width: int, height: int, ffmpeg: str, engine, engine_lock: threading.Lock) -> dict[str, Any]:
    """解码一遍，返回逐帧 single、content、luma 与黑边框。content[i] 是进入帧 i 的画面差，帧 0 记 0。"""
    import cv2

    started = time.monotonic()
    reader, writer = os.pipe()
    # 原分辨率画面一帧几 MB；管道默认 64 KB 时每帧要来回切换几十次，放大到系统上限。
    fcntl.fcntl(writer, fcntl.F_SETPIPE_SZ, PIPE_BYTES)
    command = [ffmpeg, "-v", "info", "-nostats", "-nostdin", "-threads", "1", "-filter_threads", "1", "-i", video,
               "-map", "0:v:0", "-vf", f"setpts=N,{CROPDETECT}", "-fps_mode", "passthrough", "-enc_time_base", "demux",
               "-s", f"{TINY_WIDTH}x{TINY_HEIGHT}", "-pix_fmt", "rgb24", "-threads", "1", "-f", "rawvideo", "pipe:1",
               "-map", "0:v:0", "-vf", "setpts=N", "-fps_mode", "passthrough", "-enc_time_base", "demux",
               "-pix_fmt", "rgb24", "-threads", "1", "-f", "rawvideo", f"pipe:{writer}"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=(writer,))
    os.close(writer)
    small_size = content_size(width, height)
    scores: list[float] = [0.0]
    failure: list[BaseException] = []
    crop_line: list[str] = []
    stderr_tail: list[str] = []

    def consume_native() -> None:
        buffer = bytearray(width * height * 3)
        view, previous = memoryview(buffer), None
        try:
            with os.fdopen(reader, "rb") as pipe:
                while True:
                    filled = 0
                    while filled < len(buffer):
                        got = pipe.readinto(view[filled:])
                        if not got:
                            break
                        filled += got
                    if not filled:
                        break
                    if filled != len(buffer):
                        raise ValueError("partial native frame")
                    frame = np.frombuffer(buffer, np.uint8).reshape(height, width, 3)
                    if small_size != (width, height):
                        frame = cv2.resize(frame, small_size, interpolation=cv2.INTER_LINEAR)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                    if previous is not None:
                        scores.append(sum(cv2.mean(cv2.absdiff(hsv, previous))[:3]) / 3.0)
                    previous = hsv
        except BaseException as error:  # 交回主线程处理
            failure.append(error)

    def consume_stderr() -> None:
        for raw in process.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            if "crop=" in line:
                crop_line[:] = [line]
            else:
                stderr_tail.append(line)
                del stderr_tail[:-6]

    threads = [threading.Thread(target=consume_native), threading.Thread(target=consume_stderr)]
    for thread in threads:
        thread.start()
    tiny = process.stdout.read()
    code = process.wait()
    for thread in threads:
        thread.join()
    if code != 0:
        raise RuntimeError(f"ffmpeg exited {code}: {' | '.join(stderr_tail)[-600:]}")
    if failure:
        raise failure[0]
    frames = np.frombuffer(tiny, np.uint8).reshape(-1, TINY_HEIGHT, TINY_WIDTH, 3)
    content = np.array(scores, np.float32)
    if len(frames) < 2 or len(content) != len(frames):
        raise ValueError(f"frame mismatch: tiny {len(frames)}, content {len(content)}")
    if not crop_line:
        raise ValueError("cropdetect printed no crop line")
    box = parse_crop(crop_line[0], width=width, height=height)
    decoded = time.monotonic()
    with engine_lock:
        locked = time.monotonic()
        single, _ = engine.predict(frames)
    scored = time.monotonic()
    single = np.array(single, np.float32)
    if len(single) != len(frames) or not np.isfinite(single).all():
        raise ValueError("invalid TransNet output")
    x0 = int(np.floor(box[0] * TINY_WIDTH / width)); x1 = int(np.ceil(box[2] * TINY_WIDTH / width))
    y0 = int(np.floor(box[1] * TINY_HEIGHT / height)); y1 = int(np.ceil(box[3] * TINY_HEIGHT / height))
    luma = (frames[:, y0:y1, x0:x1].astype(np.float32) @ LUMA_WEIGHTS).mean(axis=(1, 2)).astype(np.float32)
    return {"single": single, "content": content, "luma": luma, "active_box": box, "frame_count": int(len(frames)),
            "seconds": {"decode": decoded - started, "engine_wait": locked - decoded, "engine": scored - locked}}


def judge_scene(row: Mapping[str, Any], measure: Mapping[str, Any], *, thresholds: Mapping[str, Any],
                luma_range: tuple[float, float]) -> dict[str, Any]:
    """给场景的每个镜补上镜头关与亮度的判定；不合格的镜写进 flags，装窗时自然跳过。"""
    from mspc.data.shot_gate import gate_shot

    video = row["source_video_path"]
    scores = {"coordinate": "ffmpeg_sequential", "video": video, "frame_count": measure["frame_count"],
              "fps": row["fps"], "single": measure["single"], "content": measure["content"]}
    low, high = luma_range
    shots = []
    for shot in row["shots"]:
        flags = [flag for flag in shot["flags"] if flag not in IGNORED_SOURCE_FLAGS]
        start, end = shot["start_frame"], shot["end_frame"]
        if not 0 <= start < end <= measure["frame_count"]:
            shots.append({**shot, "flags": [*flags, "segment_outside_decoded_frames"], "gate": None,
                          "mean_luma": None, "min_luma": None, "max_luma": None})
            continue
        # 先看亮度：过暗或过亮的镜直接不合格，不再问切镜网络（极端画面上它的输出不可靠）。
        luma = measure["luma"][start:end]
        record = {"mean_luma": float(luma.mean()), "min_luma": float(luma.min()), "max_luma": float(luma.max())}
        if record["mean_luma"] < low or record["mean_luma"] > high:
            flags.append("below_min_brightness" if record["mean_luma"] < low else "above_max_brightness")
            shots.append({**shot, "flags": flags, "gate": None, **record})
            continue
        gate = gate_shot({"video": video, "segment": [start, end]}, scores, coordinate="ffmpeg_sequential", **thresholds)
        if gate["status"] != "kept":
            flags.append("shot_gate")
        shots.append({**shot, "flags": flags, **record,
                      "gate": {"status": gate["status"], "reasons": gate["reasons"],
                               "revised_segment": gate["revised_segment"], "boundary_cuts": gate["boundary_cuts"]}})
    return {**row, "active_box": list(measure["active_box"]), "active_box_measurement": CROPDETECT, "shots": shots}


def scene_windows(row: Mapping[str, Any], measure: Mapping[str, Any], *, seed: int, scene_index: int,
                  thresholds: Mapping[str, Any], cut_version: str, min_window_seconds: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """用判过的场景行装窗，并把修界与保护后的帧段写进窗。整窗（各镜相加）短于 ``min_window_seconds`` 的不要。"""
    from mspc.data.scene_windows import training_windows
    from mspc.data.shot_gate import gate_window, revised_scene_window

    video = row["source_video_path"]
    tables = {video: {"coordinate": "ffmpeg_sequential", "video": video, "frame_count": measure["frame_count"],
                      "fps": row["fps"], "single": measure["single"], "content": measure["content"]}}
    by_index = {shot["source_shot_index"]: shot for shot in row["shots"]}
    audit: dict[str, Any] = {}
    windows = []
    for window in training_windows(row, seed=seed, scene_index=scene_index, audit=audit):
        window = {**window, "window_id": f"{cut_version}:{window['window_id']}", "cut_version": cut_version,
                  "width": row["width"], "height": row["height"], "video_frame_count": row["video_frame_count"],
                  "active_box": row["active_box"], "frame_basis": "ffmpeg_sequential_zero_based"}
        gate = gate_window({"window_id": window["window_id"], "family": "scene_window_v1",
                            "shots": [{"video": video, "segment": [s["start_frame"], s["end_frame"]]} for s in window["shots"]]},
                           tables, **thresholds)
        if gate["status"] != "kept":
            raise ValueError(f"window built from gated shots was rejected: {window['window_id']}")
        window = revised_scene_window(window, gate)
        if sum(shot["seconds"] for shot in window["shots"]) < min_window_seconds:
            audit.setdefault("short_windows", []).append(window["window_id"])
            continue
        for shot in window["shots"]:
            source = by_index[shot["source_shot_index"]]
            for key in ("mean_luma", "min_luma", "max_luma"):
                shot[key] = source[key]
            for key in ("scene", "characters"):
                if key in source:
                    shot[key] = source[key]
        windows.append(window)
    return windows, audit
