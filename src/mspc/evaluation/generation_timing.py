"""各方法生成记录的实际输出镜长；不加载模型或调用判官。"""
from __future__ import annotations

from itertools import pairwise


def shot_frames(segments: list[dict]) -> list[int]:
    """把各方法原生分段记录展平成逐镜实际帧数（按成片帧率计）。

    HoloCine 用官方量化后的 ``shot_cut_frames``；输出帧率与生成帧率不同的入口（官方 Wan 24fps）
    用 ``output_frame_counts``；逐镜生成的关键帧路线（StoryDiffusion、IC-LoRA）每段一镜，记 ``shot``（从 1 起）；
    其余用 ``frame_counts``。每段帧数之和必须等于该段 ``num_frames``。
    """
    frames, next_shot = [], 0
    for index, segment in enumerate(segments):
        if 'shot' in segment:
            segment = {**segment, 'start_shot': segment['shot'] - 1, 'end_shot': segment['shot'],
                       'frame_counts': [segment['num_frames']]}
        if segment['start_shot'] != next_shot:
            raise ValueError(f'segment {index} starts at shot {segment["start_shot"]}, expected {next_shot}')
        count = segment['end_shot'] - segment['start_shot']
        if 'shot_cut_frames' in segment:
            edges = [0, *segment['shot_cut_frames'], segment['num_frames']]
            counts = [b - a for a, b in pairwise(edges)]
        elif 'output_frame_counts' in segment:
            counts = list(segment['output_frame_counts'])
        else:
            counts = list(segment['frame_counts'])
        if len(counts) != count or any(type(n) is not int or n <= 0 for n in counts):
            raise ValueError(f'segment {index} shot frames invalid: {counts}')
        if sum(counts) != segment['num_frames']:
            raise ValueError(f'segment {index} frames {sum(counts)} != num_frames {segment["num_frames"]}')
        frames += counts
        next_shot = segment['end_shot']
    return frames
