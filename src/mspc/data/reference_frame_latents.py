"""参考帧库：每镜 first/middle/last 候选，CLIP 去重入库，入库帧单独编码成一张潜帧。

现行只有推理用这个库：``mspc.inference.segment_story`` 每段生成后把实际帧登记进 :class:`ReferenceLibrary`，
下一段取库里的帧作参考。训练不读它：训练的参考由 ``mspc.data.reference_frames.read_reference_frames`` 在线从源片
随机取帧，不做 CLIP 去重、不建库（见 ``mspc.training.block_dataset``）。库的规则：

- 每镜候选三帧 first / middle / last（:func:`frame_index`）；
- 用**未裁剪**画面算 CLIP 图像嵌入（L2 归一化），与库内任一帧余弦 > :data:`SIMILARITY_MAX` 的不入库
  （StoryMem 的关键帧去重做法），每镜因此至多 3 帧；
- 入库帧整帧单独编码（``mspc.inference.wan22.encode_single_frame``：单帧过因果 VAE，与原生首帧口同一种编码），
  训练端编单帧也走同一入口，不裁剪（首发不做随机裁剪；v3 的库是裁剪过的，按 schema 拒收）。

CLIP 权重是必填配置路径（transformers ``CLIPModel`` 目录），与 StoryMem 发布代码用同一份。

离线产物（:func:`write_case` 与 ``tools/data/build_reference_frame_latents.py`` 写出，现行训练不读取）：
``<out>/latents/<case_id>.safetensors``，张量 ``reference_frame_latents`` 形如 ``[1, C, N, H, W]``，
元数据 ``entries`` 按入库顺序列 ``[shot_index, position]``，与张量第 n 帧一一对应。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

TENSOR_KEY = "reference_frame_latents"
SCHEMA = "reference_frame_latents_v4"
FRAME_POSITIONS = ("first", "middle", "last")
SIMILARITY_MAX = 0.9


def frame_index(position: str, num_frames: int) -> int:
    """一镜 RGB 帧里的 first/middle/last 帧号；first 跳过第 0 帧，训练与推理共用。"""
    if type(num_frames) is not int or num_frames < 2:
        raise ValueError(f"num_frames must be an int >= 2, got {num_frames!r}")
    if position == "first":
        return 1
    if position == "middle":
        return max(1, (num_frames - 1) // 2)
    if position == "last":
        return num_frames - 1
    raise ValueError(f"position must be one of {FRAME_POSITIONS}, got {position!r}")


def shot_candidates(frames: Sequence[Any]) -> dict[str, Any]:
    """一镜全部 RGB 帧 → 三个候选帧（按 first/middle/last 顺序）。"""
    indices = {}
    for position in FRAME_POSITIONS:
        index = frame_index(position, len(frames))
        if index not in indices.values():
            indices[position] = index
    return {position: frames[index] for position, index in indices.items()}


def admitted(candidates: Sequence[Any], library: Any) -> list[int]:
    """按顺序过候选嵌入：与库（含本镜已入库者）任一帧余弦 > SIMILARITY_MAX 的不入库。返回入库下标。

    嵌入须已 L2 归一化（``[D]``），``library`` 是 ``[N, D]``（N 可为 0）。
    """
    import torch

    kept, bank = [], library
    for index, embedding in enumerate(candidates):
        if bank.shape[0] and float((bank @ embedding).max()) > SIMILARITY_MAX:
            continue
        kept.append(index)
        bank = torch.cat([bank, embedding[None]], dim=0)
    return kept


class ClipImageEmbedder:
    """同一 CLIP 的归一化图像／检索文字嵌入，权重目录必须显式给出。"""

    def __init__(self, model_dir: Path, device: Any) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise ValueError(f"CLIP weights are not a directory: {model_dir}")
        self.model_dir = model_dir
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(model_dir).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_dir)

    def __call__(self, images: Sequence[Any]) -> Any:
        import torch

        with torch.no_grad():
            pixels = self.processor(images=list(images), return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**pixels)
            if not torch.is_tensor(features):
                raise RuntimeError("CLIP did not return image embeddings")
        features = features.float().cpu()
        return features / features.norm(dim=-1, keepdim=True)

    def encode_text(self, texts: Sequence[str]) -> Any:
        """独立的短检索描述；超出真实 CLIP 文字容量直接报错，不截断分镜长文。"""
        import torch

        if isinstance(texts, str) or not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("CLIP queries must be a nonempty sequence of nonempty strings")
        tokens = self.processor(text=list(texts), padding=True, truncation=False, return_tensors="pt")
        limit = self.model.config.text_config.max_position_embeddings
        if tokens["input_ids"].shape[1] > limit:
            raise ValueError(f"CLIP query has {tokens['input_ids'].shape[1]} tokens, exceeding {limit}")
        with torch.no_grad():
            features = self.model.get_text_features(**tokens.to(self.device))
        if not torch.is_tensor(features):
            raise RuntimeError("CLIP did not return text embeddings")
        features = features.float().cpu()
        return features / features.norm(dim=-1, keepdim=True)


@dataclass(frozen=True)
class LibraryFrame:
    shot_index: int
    position: str
    embedding: Any
    latent: Any


@dataclass
class ReferenceLibrary:
    """一条故事的参考帧库，按入库顺序（镜序、镜内 first→middle→last）。"""

    frames: list[LibraryFrame] = field(default_factory=list)

    def add_shot(self, shot_index: int, candidates: Mapping[str, Any], *, embed: Callable[[Sequence[Any]], Any],
                 encode: Callable[[Any], Any]) -> list[LibraryFrame]:
        """一镜的三个候选按 CLIP 去重入库；入库帧整帧单独编码。"""
        import torch

        positions = tuple(candidates)
        if not positions or positions != tuple(p for p in FRAME_POSITIONS if p in candidates):
            raise ValueError(f"candidates must be a nonempty ordered subset of {FRAME_POSITIONS}")
        if self.frames and shot_index <= self.frames[-1].shot_index:
            raise ValueError("shots must enter the library in increasing order")
        images = [candidates[position] for position in positions]
        embeddings = embed(images)
        bank = (torch.stack([f.embedding for f in self.frames]) if self.frames
                else embeddings.new_zeros((0, embeddings.shape[1])))
        added = []
        for index in admitted(list(embeddings), bank):
            frame = LibraryFrame(shot_index, positions[index], embeddings[index], encode(images[index]))
            self.frames.append(frame)
            added.append(frame)
        return added


def decode_shot_frames(runtime: Any, input_latents: Any, latent_group: Sequence[int]) -> list[Any]:
    """缓存一条故事的 ``input_latents`` 里某一镜整镜解码（镜内是独立编码的因果 VAE 序列）。"""
    start, end = int(latent_group[0]), int(latent_group[1])
    if end <= start:
        raise ValueError(f"empty latent shot group {latent_group!r}")
    decoded = runtime.pipe.vae.decode(
        input_latents[:, :, start:end].to(device=runtime.device, dtype=runtime.dtype),
        device=runtime.device,
        tiled=runtime.pipeline_cfg["tiled"],
        tile_size=tuple(runtime.pipeline_cfg["tile_size"]),
        tile_stride=tuple(runtime.pipeline_cfg["tile_stride"]),
    )
    frames = list(runtime.pipe.vae_output_to_video(decoded))
    expected = 1 + 4 * (end - start - 1)
    if len(frames) != expected:
        raise ValueError(f"{end - start} latent frames decoded to {len(frames)} RGB frames, expected {expected}")
    return frames


def read_input_latents(latent_cache_dir: Path, record: Mapping[str, Any]) -> Any:
    """缓存 manifest 一条记录对应的 ``input_latents``（CPU 张量），时间维长度与记录核对。"""
    from safetensors import safe_open

    from mspc.data.cache_provenance import require_matching_provenance

    path = Path(latent_cache_dir) / record["cache_file"]
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
        stored = json.loads(metadata["provenance"]) if "provenance" in metadata else None
        require_matching_provenance(stored, record["provenance"])
        latents = handle.get_tensor("input_latents")
    if list(latents.shape) != list(record["input_latents_shape"]):
        raise ValueError(f"{record['case_id']}: {path} has shape {list(latents.shape)}, manifest says {record['input_latents_shape']}")
    return latents


def case_reference_frames(runtime: Any, embed: Callable[[Sequence[Any]], Any], input_latents: Any,
                          record: Mapping[str, Any]) -> ReferenceLibrary:
    """一条训练故事的参考帧库：每镜整镜解码取三候选，CLIP 去重，入库帧单独编码。"""
    from mspc.inference.wan22 import encode_single_frame

    groups = record["latent_shot_groups"] or [[0, int(record["input_latents_shape"][2])]]
    library = ReferenceLibrary()
    for shot_index, group in enumerate(groups):
        candidates = shot_candidates(decode_shot_frames(runtime, input_latents, group))
        library.add_shot(shot_index, candidates, embed=embed,
            encode=lambda image: encode_single_frame(runtime, image).cpu())
    return library


def write_case(out_dir: Path, record: Mapping[str, Any], library: ReferenceLibrary, *, clip_model: str) -> Path:
    """原子写一条故事的参考帧库（先写临时文件再改名）。"""
    import torch
    from safetensors.torch import save_file

    if not library.frames:
        raise ValueError(f"{record['case_id']}: empty reference library")
    path = Path(out_dir) / "latents" / f"{record['case_id']}.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    latents = torch.cat([frame.latent for frame in library.frames], dim=2)
    save_file(
        {TENSOR_KEY: latents.contiguous()},
        str(tmp),
        metadata={
            "schema": SCHEMA,
            "case_id": record["case_id"],
            "entries": json.dumps([[f.shot_index, f.position] for f in library.frames]),
            "latent_shot_groups": json.dumps(record["latent_shot_groups"]),
            "source_provenance": json.dumps(record["provenance"], ensure_ascii=False),
            "clip_model": clip_model,
            "similarity_max": json.dumps(SIMILARITY_MAX),
        },
    )
    tmp.replace(path)
    return path


def validate_reference_source(metadata: Mapping[str, str], record: Mapping[str, Any]) -> list[tuple[int, str]]:
    """消费者核 schema、镜组与来源；返回入库条目 ``[(shot_index, position), ...]``。

    「这份库出自这条缓存记录」由 ``source_provenance`` 与记录的 provenance 逐项相等认定（源媒体、取帧、裁切、VAE 与
    编码配置）；不比目录路径——成员目录在主服务器生成、运到训练集群训练，路径两面不同。
    """
    from mspc.data.cache_provenance import require_matching_provenance

    if metadata.get("schema") != SCHEMA or metadata.get("case_id") != record["case_id"]:
        raise ValueError(f"{record['case_id']}: reference library schema/case differs (expected {SCHEMA})")
    if json.loads(metadata["latent_shot_groups"]) != record["latent_shot_groups"]:
        raise ValueError("reference library latent shot groups differ")
    stored = json.loads(metadata["source_provenance"]) if "source_provenance" in metadata else None
    require_matching_provenance(stored, record["provenance"])
    entries = [(int(shot), str(position)) for shot, position in json.loads(metadata["entries"])]
    shots = len(record["latent_shot_groups"] or [None])
    if (not entries or len(set(entries)) != len(entries)
            or any(not 0 <= shot < shots or position not in FRAME_POSITIONS for shot, position in entries)
            or entries != sorted(entries, key=lambda e: (e[0], FRAME_POSITIONS.index(e[1])))):
        raise ValueError(f"{record['case_id']}: reference library entries must be distinct, ordered, in range")
    return entries
