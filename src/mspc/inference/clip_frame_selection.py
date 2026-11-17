"""按画面相似度选参考帧（CLIP 文图检索，论文表 2 选帧消融；``history_policy=clip_retrieval``）。

回答「为什么不用视觉检索」：不看分镜，只按 CLIP 文图相似度从已生成的画面里取参考。

规则（只换「谁来选」，其余与按分镜选帧相同）：

1. 候选库（每个已生成镜，生成后建一次，:func:`shot_keyframes`）：同 Comp-RAP 的 CLIP Retrieval 对照——镜内帧按序，
   与同镜上一张入库帧的 CLIP 图像余弦 < 0.95 才入库，每镜至多 6 张（原文没写超过怎么截，取前 6 张，约定）。帧号沿用
   本项目不取第 0 帧的约定（:func:`mspc.data.reference_frame_latents.frame_index`）。
2. 检索（每窗，:func:`retrieve`）：本窗每镜文字与库里每帧算 CLIP 文图余弦，一帧的分取它对本窗各镜文字的最大值；每个
   源镜取它分最高的一帧（训练每镜只给一张参考、镜内帧位均匀抽，见 ``mspc.training.block_dataset``），按分取前 n 个镜，
   n 与同一窗按分镜选帧的张数相同（与「取最近几镜」同口径）。返回按镜序。
3. CLIP 与 StoryMem 官方选帧同一份 ViT-B/32（HF 的 ``openai/clip-vit-base-patch32`` 与官方 ``ViT-B-32.pt`` 嵌入逐元素
   相差 < 4e-7，11-15 核），图像预处理照 StoryMem ``extract_keyframes.py``：先按 ``smart_resize`` 缩到至多 256×28×28 像素
   （双线性、抗锯齿），再双线性缩到 224×224、不裁中心，CLIP 均值方差归一。文字按 CLIP 标准截到 77 个 token。
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

CLIP_WEIGHTS = "OoVMetric/models_ckpt/clip-vit-base-patch32"
#: Comp-RAP 库：与同镜上一张入库帧的 CLIP 图像余弦低于它才入库；每镜至多几张。
SAME_SHOT_SIMILARITY = 0.95
FRAMES_PER_SHOT = 6
#: StoryMem ``extract_keyframes.py`` 的缩图常量（``IMAGE_FACTOR``、``VIDEO_MIN_PIXELS``、``max_pixels=256*28*28``）。
_FACTOR, _MIN_PIXELS, _MAX_PIXELS = 28, 48 * 28 * 28, 256 * 28 * 28
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)


def smart_resize(height: int, width: int) -> tuple[int, int]:
    """StoryMem ``extract_keyframes.smart_resize``（factor 28，像素数夹在 [48, 256]×28×28）。"""
    h_bar, w_bar = max(_FACTOR, round(height / _FACTOR) * _FACTOR), max(_FACTOR, round(width / _FACTOR) * _FACTOR)
    if h_bar * w_bar > _MAX_PIXELS:
        beta = math.sqrt(height * width / _MAX_PIXELS)
        h_bar, w_bar = math.floor(height / beta / _FACTOR) * _FACTOR, math.floor(width / beta / _FACTOR) * _FACTOR
    elif h_bar * w_bar < _MIN_PIXELS:
        beta = math.sqrt(_MIN_PIXELS / (height * width))
        h_bar, w_bar = math.ceil(height * beta / _FACTOR) * _FACTOR, math.ceil(width * beta / _FACTOR) * _FACTOR
    return max(h_bar, _FACTOR), max(w_bar, _FACTOR)


class ClipEmbedder:
    """归一化的 CLIP 图像与文字嵌入。"""

    def __init__(self, weights: Path, device: Any) -> None:
        from transformers import CLIPModel, CLIPTokenizer

        if not Path(weights).is_dir():
            raise ValueError(f"CLIP weights are not a directory: {weights}")
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(weights).eval().to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(weights)

    @torch.no_grad()
    def images(self, frames: Sequence[Any]) -> torch.Tensor:
        """一镜的 RGB 帧（PIL 或 HWC uint8）→ ``[N, D]``，预处理照 StoryMem。"""
        pixels = torch.from_numpy(np.stack([np.asarray(frame, dtype=np.uint8) for frame in frames]))
        if pixels.ndim != 4 or pixels.shape[-1] != 3:
            raise ValueError("CLIP frames must be HWC RGB")
        x = pixels.permute(0, 3, 1, 2).to(self.device, torch.float32)
        # StoryMem 在 uint8 帧上缩图，结果仍是 0…255 的整数，再除以 255。
        x = F.interpolate(x, list(smart_resize(*x.shape[2:])), mode="bilinear", antialias=True)
        x = F.interpolate(x.round().clamp(0, 255) / 255.0, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor(_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(_STD, device=self.device).view(1, 3, 1, 1)
        return F.normalize(self.model.get_image_features(pixel_values=(x - mean) / std).float(), dim=-1).cpu()

    @torch.no_grad()
    def texts(self, texts: Sequence[str]) -> torch.Tensor:
        if isinstance(texts, str) or not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("CLIP queries must be a nonempty sequence of nonempty strings")
        tokens = self.tokenizer(list(texts), padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        return F.normalize(self.model.get_text_features(**tokens.to(self.device)).float(), dim=-1).cpu()


_EMBEDDERS: dict[str, ClipEmbedder] = {}
_LOCK = threading.Lock()


def embedder_for(device: Any) -> ClipEmbedder:
    """每张卡每进程只加载一次（同卡几部故事并发时加锁）。权重在 ``$MSPC_SHARED_ROOT`` 下。"""
    root = os.environ.get("MSPC_SHARED_ROOT")
    if not root:
        raise ValueError("CLIP retrieval needs MSPC_SHARED_ROOT to locate the CLIP weights")
    with _LOCK:
        if str(device) not in _EMBEDDERS:
            _EMBEDDERS[str(device)] = ClipEmbedder(Path(root) / CLIP_WEIGHTS, device)
        return _EMBEDDERS[str(device)]


def shot_keyframes(embeddings: torch.Tensor) -> list[int]:
    """一镜帧 1…N−1 的图像嵌入（第 i 行是第 i+1 帧）→ 入库的帧号：按序，与上一张入库帧余弦 < 0.95 才入，至多 6 张。"""
    if embeddings.ndim != 2 or len(embeddings) < 1:
        raise ValueError("a shot needs at least one candidate frame")
    kept = [0]
    for row in range(1, len(embeddings)):
        if len(kept) == FRAMES_PER_SHOT:
            break
        if float(embeddings[row] @ embeddings[kept[-1]]) < SAME_SHOT_SIMILARITY:
            kept.append(row)
    return [row + 1 for row in kept]


def best_frames(library: Mapping[int, Mapping[int, torch.Tensor]], queries: torch.Tensor) -> dict[int, tuple[int, float]]:
    """``library``：源镜下标 → {帧号: 图像嵌入}；``queries``：本窗各镜文字嵌入 ``[T, D]``。

    返回每个源镜分最高的一帧 (帧号, 分)；一帧的分是它对本窗各镜文字余弦的最大值，同分取镜内靠前的帧。"""
    best = {}
    for shot, frames in library.items():
        if not frames:
            raise ValueError(f"generated shot {shot} has no library frame")
        scored = [(float((queries @ embedding).max()), frame) for frame, embedding in frames.items()]
        score, frame = max(scored, key=lambda entry: (entry[0], -entry[1]))
        best[shot] = (frame, score)
    return best


def retrieve(best: Mapping[int, tuple[int, float]], count: int) -> list[tuple[int, int]]:
    """分最高的 ``count`` 个源镜及其帧 (源镜, 帧号)，按镜序；同分取离本窗近（下标大）的镜。"""
    if type(count) is not int or not 0 <= count <= len(best):
        raise ValueError(f"cannot retrieve {count} of {len(best)} generated shots")
    ranked = sorted(best, key=lambda shot: (-best[shot][1], -shot))
    return sorted((shot, best[shot][0]) for shot in ranked[:count])
