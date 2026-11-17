"""块流程共用的取帧与代理片小函数：FFmpeg 顺序取帧、训练画幅裁切、2 fps 代理取帧、一段镜的代理片导出、镜头取帧网格、按下标读帧。"""
from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any, Sequence

import numpy as np


def iter_scene_frames(scene, wanted, *, ffmpeg, error_path, video_decode_num_threads: int):
    """Read FFmpeg sequential coordinates; ``wanted=None`` streams to EOF."""
    import numpy as np

    if wanted is not None and (min(wanted) < 0 or max(wanted) >= scene["video_frame_count"]):
        raise ValueError("frame outside manifest video range")
    if type(video_decode_num_threads) is not int or video_decode_num_threads < 1:
        raise ValueError("video_decode_num_threads must be a positive integer")
    selection = ["-vf", "setpts=N"]
    if wanted is not None:
        select = "+".join(f"eq(n\\,{f})" for f in wanted)
        selection = ["-vf", "select=" + select + ",setpts=N", "-frames:v", str(len(wanted))]
    # rawvideo 只消费顺序像素；选帧之后给管道帧编号作为 PTS，消除源 PTS 重复。
    # 保留解复用时间基，避免编码器按平均 fps 再量化；不改输入帧号或像素。
    command = [ffmpeg,"-v","error","-nostdin","-threads",str(video_decode_num_threads),"-filter_threads","1","-i",scene["source_video_path"],
               "-map","0:v:0",*selection,"-fps_mode","passthrough",
               "-threads","1","-enc_time_base","demux","-f","rawvideo","-pix_fmt","rgb24","pipe:1"]
    error_path=Path(error_path)
    with error_path.open("wb") as errors:
        process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=errors)
        # 一帧原分辨率画面几 MB；管道默认 64 KB 时每帧来回切换几十次，放大到系统默认上限（只影响速度）。
        import fcntl
        if hasattr(fcntl,"F_SETPIPE_SZ"):fcntl.fcntl(process.stdout.fileno(),fcntl.F_SETPIPE_SZ,1<<20)
        try:
            size = scene["width"]*scene["height"]*3
            import itertools
            for position in itertools.count() if wanted is None else wanted:
                data = process.stdout.read(size)
                if wanted is None and not data:
                    break
                if len(data) != size:
                    raise ValueError(f"truncated FFmpeg frame {position}: {len(data)}/{size}")
                yield position, np.frombuffer(data,dtype=np.uint8).reshape(scene["height"],scene["width"],3)
            if process.wait() != 0:
                raise RuntimeError(error_path.read_text()[-2000:])
        finally:
            if process.poll() is None:
                process.terminate(); process.wait()
            process.stdout.close()
    if error_path.stat().st_size:
        raise RuntimeError("FFmpeg reported decode errors: "+error_path.read_text()[-2000:])


def center_crop(image, width: int = 832, height: int = 480):
    """训练读帧的画幅：整帧缩放到盖住 832×480，再居中裁切。"""
    scale = max(width / image.width, height / image.height)
    resized = image.resize((max(width, round(image.width * scale)), max(height, round(image.height * scale))))
    left, top = (resized.width - width) // 2, (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def proxy_frames(groups, proxy_stride: int) -> list[list[int]]:
    """逐镜的 2fps 代理取帧（目标帧号）：每 ``proxy_stride`` 帧取一帧，首末都保留。"""
    return [sorted(set(range(start, end, proxy_stride)) | {end - 1}) for start, end in groups]


def export_shot_span(*, wid: str, selected: list[list[int]], frame, videos: Path, ffmpeg: Path) -> dict:
    """写一窗的代理片与首末帧索引；``frame(目标帧号)`` 给训练裁切画幅的 PIL 图。实际 proxy 镜界按抽后帧数写明。"""
    proxy, keyframes, timestamps, mapping = [], [], [], []
    for shot, picks in enumerate(selected):
        offset = len(proxy)
        for target_index in picks:
            proxy.append(np.asarray(center_crop(frame(target_index))))
            mapping.append({"shot": shot, "target_frame": target_index})
        timestamps.append({"shot": shot, "start": offset / 2, "end": len(proxy) / 2})
        for phase, position in (("before", 0), ("after", len(picks) - 1)):
            fid = f"{wid}-s{shot}-{phase}"
            keyframes.append({"id": fid, "shot": shot, "phase": phase,
                "proxy_frame_index": offset + position,
                "target_frame_index": picks[position]})
    video = videos / f"{wid}.mp4"
    subprocess.run([str(ffmpeg), "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "832x480", "-r", "2", "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-threads", "1", "-crf", "18", str(video)],
        input=np.stack(proxy).tobytes(), check=True, capture_output=True)
    return {"video": str(video), "shot_timestamps": timestamps, "keyframes": keyframes, "proxy_to_target_frames": mapping}


def shot_grid(start: int, end: int, count: int) -> list[int]:
    """源镜头半开区间 ``[start, end)`` 均匀取 ``count`` 帧，首末帧入选；与 ``FpsAwareTargetFpsPlanner.clean_indices_for_shot``
    同一映射（``count`` 由上游按目标帧率与 4n+1 定好）。"""

    if type(start) is not int or type(end) is not int or type(count) is not int:
        raise ValueError("start, end and count must be ints")
    available = end - start
    if count < 2 or available < count:
        raise ValueError(f"cannot take {count} distinct frames from [{start}, {end})")
    indices = [start + round(i * (available - 1) / (count - 1)) for i in range(count)]
    if len(set(indices)) != count:
        raise ValueError("grid repeated a source frame")
    return indices


def default_frame_reader(path: Path, indices: Sequence[int]) -> list[Any]:
    import decord
    from PIL import Image

    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
    invalid = [index for index in indices if index < 0 or index >= len(reader)]
    if invalid:
        raise ValueError(f"{path}: frame indices outside 0..{len(reader) - 1}: {invalid[:8]}")
    arrays = reader.get_batch(list(indices)).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in arrays]
