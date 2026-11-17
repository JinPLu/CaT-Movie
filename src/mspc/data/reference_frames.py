"""CPU按FFmpeg顺序帧号读取抽中的整帧参考；不写缓存，不加载模型。"""
from __future__ import annotations

from bisect import bisect_left
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from mspc.config.shared_root import reroot_source_path
from mspc.data.frames import center_crop
from mspc.data.scene_training_manifest import require_training_media, training_crop_box


def read_reference_frames(scene: Mapping[str, Any], indices: Sequence[int], *,
                          video_decode_num_threads: int, media: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, float | int]]:
    import av

    media = require_training_media(media)
    crop = training_crop_box(scene, media=media)

    if type(video_decode_num_threads) is not int or video_decode_num_threads < 1:
        raise ValueError("video_decode_num_threads must be a positive integer")
    if not indices or any(type(i) is not int or not 0 <= i < scene["video_frame_count"] for i in indices):
        raise ValueError("reference indices must be nonempty source frame numbers inside the video")
    path = reroot_source_path(scene["source_video_path"], location="reference source video")
    started = time.perf_counter()
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_count = video_decode_num_threads
        opened = time.perf_counter()
        # MovieChat edit-list预卷packet虽有PTS但标为discard，不属于FFmpeg输出帧。
        # 这里只扫描压缩packet建本次调用的帧号→PTS映射，不解码整片、不跨样本保存。
        timestamps = [packet.pts for packet in container.demux(stream)
                      if packet.size and not packet.is_discard]
        if len(timestamps) != scene["video_frame_count"] or None in timestamps:
            raise ValueError(f"{path}: packet presentation order does not cover manifest frames")
        timestamps.sort()
        indexed = time.perf_counter()
        frames = {}
        decode_seconds = crop_seconds = 0.0
        for index in sorted(set(indices)):
            # 同PTS也可以是两个真实输出帧（Cinedance 067fe4cc，495/496）。
            # 帧号由显示顺序和同PTS出现次序共同定义，不能去重。
            pts = timestamps[index]
            occurrence = index - bisect_left(timestamps, pts)
            t = time.perf_counter()
            # 从目标PTS之前的关键帧解码，确保同PTS的首个输出不会被seek跳过。
            container.seek(pts - 1, stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                if frame.pts == pts:
                    if occurrence:
                        occurrence -= 1
                        continue
                    # PyAV默认src/dst相同会跳过sws色域设置，BT.709被当作默认BT.601。
                    # 明确原帧色域/range与RGB full-range，保持既有FFmpeg像素。
                    pixels = frame.to_ndarray(format="rgb24", src_colorspace=frame.colorspace,
                                              dst_colorspace=0, src_color_range=frame.color_range,
                                              dst_color_range=av.video.reformatter.ColorRange.JPEG)
                    break
                if frame.pts is not None and frame.pts > pts:
                    raise ValueError(f"{path}: seek skipped requested frame {index} at PTS {pts}")
            else:
                raise ValueError(f"{path}: EOF before requested frame {index}")
            decode_seconds += time.perf_counter() - t
            if pixels.shape != (scene["height"], scene["width"], 3):
                raise ValueError(f"{path}: decoded dimensions differ from manifest")
            t = time.perf_counter()
            frames[index] = np.asarray(center_crop(Image.fromarray(pixels).crop(crop), width=media["width"], height=media["height"]))
            crop_seconds += time.perf_counter() - t
    images = torch.from_numpy(np.stack([frames[i] for i in indices])).permute(0, 3, 1, 2).contiguous()
    return images, {"reference_open_seconds": opened - started,
                    "reference_index_seconds": indexed - opened,
                    "reference_decode_seconds": decode_seconds,
                    "reference_crop_seconds": crop_seconds,
                    "reference_read_seconds": time.perf_counter() - started,
                    "reference_output_frames": len(frames)}
