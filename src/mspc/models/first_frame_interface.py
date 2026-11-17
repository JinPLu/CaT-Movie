"""镜首帧接口（``self_attention_policy: first_frame_interface``）：每镜第一个潜帧是这一镜的接口，外部信息只进镜首帧，
其余潜帧只在镜内读。

设计：每镜首帧是跨镜交互（参考帧、他镜首帧）与镜内交互的主力，位置编码去掉镜相位。从原生权重起训四组之二：不带规划器 token，镜首帧读全部参考
（不按在场名单路由，与全互读那组读的参考相同）。按查询写（序列是 [参考帧][目标各镜]）：

- 镜 i 首潜帧：读本镜全部潜帧、他镜首潜帧、全部参考帧；
- 镜 i 其余潜帧：只读本镜全部潜帧（含首潜帧），不读参考、他镜；
- 参考帧：只读自己（参考帧彼此不读，与 ``memory_self_only_v1`` 相同）。

每类查询一次注意力调用，K/V 按索引拼出来，不用稠密掩码。q/k/v 已加 RoPE。位置配 ``global_time_native_rope_v1``
（整段连续时间、不加镜相位）；参考按顺序各占一个时间槽。文字交叉注意力不在这里，不变。
"""
from __future__ import annotations

from typing import Sequence

import torch

FIRST_FRAME_INTERFACE = "first_frame_interface"


def first_frame_interface_attention(attn, q, k, v, *, groups: Sequence[Sequence[int]] | None, prefix_frames: int,
                                    tokens_per_frame: int):
    """``groups`` 是块里的潜帧分组（参考帧各一组在前，随后各目标镜）。单镜且没有参考时数据不给分组（``None``）：
    整段是一镜，镜首帧与其余帧都读本镜全部，就是普通的整段注意力。"""
    if groups is None:
        if prefix_frames:
            raise ValueError("references need latent shot groups")
        return attn(q, k, v)
    if type(tokens_per_frame) is not int or tokens_per_frame <= 0:
        raise ValueError("the first-frame interface needs tokens_per_frame")
    if len(groups) <= prefix_frames or any(tuple(groups[j]) != (j, j + 1) for j in range(prefix_frames)):
        raise ValueError("the first-frame interface needs one latent group per reference, then the target shots")
    shots, end = [], prefix_frames
    for start, stop in groups[prefix_frames:]:
        if start != end or stop <= start:
            raise ValueError("target shot groups must partition the target frames")
        shots.append((start * tokens_per_frame, stop * tokens_per_frame))
        end = stop
    if q.shape[1] != end * tokens_per_frame:
        raise ValueError("shot groups do not cover the video tokens")
    device = q.device

    def span(a, b):
        return torch.arange(a, b, device=device)

    out = torch.empty_like(q)

    def run(queries, keys):
        out[:, queries] = attn(q[:, queries].contiguous(), k[:, keys].contiguous(), v[:, keys].contiguous())

    references = span(0, prefix_frames * tokens_per_frame)
    firsts = [span(a, a + tokens_per_frame) for a, _ in shots]
    for i, (a, b) in enumerate(shots):
        run(firsts[i], torch.cat([span(a, b), *firsts[:i], *firsts[i + 1:], references]))
        if b - a > tokens_per_frame:
            run(span(a + tokens_per_frame, b), span(a, b))
    for j in range(prefix_frames):
        frame = span(j * tokens_per_frame, (j + 1) * tokens_per_frame)
        run(frame, frame)
    return out
