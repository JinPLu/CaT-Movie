"""块级训练消费：读冻结的目标窗口表，窗口外同块真片随机帧作为参考。

块只保存成员关系；目标显式选择逐镜潜变量缓存或原视频，参考只解码抽中的RGB，VAE由训练主进程执行。
坏镜留在原位置并中断目标窗口。整块没有 308 帧上限；目标窗口是连续可用镜、总帧数不超过 228，镜数不封顶
（``BlockWindowSampler`` 冻表时抽）。
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from itertools import accumulate
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors import safe_open

from mspc.config.shared_root import reroot_source_path
from mspc.core.draws import unit_draw
from mspc.core.wan22_contract import (FRAME_ENTITY_ROW_POLICIES, MEMORY_ROW_POLICY_FIXED_ROLE_V1,
    MEMORY_ATTENTION_POLICY_ROUTED_SELF_ONLY_V1, MEMORY_ATTENTION_POLICY_WINDOW_SHARED_SELF_ONLY_V1,
    ROUTED_MEMORY_ATTENTION_POLICIES, WAN22_SELF_ATTENTION_POLICIES,
    MEMORY_ROW_POLICY_FRAME_ENTITIES_WITH_SHOT_OBJECTS_V1,
    SEGMENT_MAX_FRAMES, SEGMENT_MAX_SHOTS, frame_counts_over_segment_cap, memory_prefix_case_layout,
    require_compiler_capacity)
from mspc.data.cache_provenance import require_matching_provenance
from mspc.data.scene_training_manifest import ANNOTATION_MEDIA, require_training_media
from mspc.data.reference_frames import read_reference_frames
from mspc.data.reference_crops import ReferenceCrops, apply_crops
from mspc.data.reference_entities import ReferenceEntities
from mspc.data.reference_routing import reference_readers, reference_slot_metadata
from mspc.data.shot_targets import (latent_frames, planned_row, reconcile,
                                   record_groups, shot_key, shot_row, sampled_target_positions, member_target_frame_counts)
from mspc.data.shot_text import (UMT5_PROMPT_TOKEN_BUDGET,
                                TARGET_TEXT_OMISSIONS, load_umt5_token_counter, reference_text,
                                reference_setting, reference_text_with_entity_spans, shot_object_phrases,
                                shot_text_with_entity_spans,
                                without_target_text)
from mspc.data.target_masks import TargetMaskWeights, WEIGHTINGS, compile_member_target_mask_weights
from mspc.models.memory_prefix_options import MemoryPrefixModelOptions
from mspc.models.spatial_layout_inputs import compile_spatial_layout_inputs


def _without_layouts(annotation: Mapping[str, Any]) -> dict:
    return {key: value for key, value in annotation.items() if key != "layouts"}


def jitter_layout_entity_boxes(layouts: Sequence[Mapping[str, Any]], shot_indices: Sequence[int], *,
                               jitter: Mapping[str, float], training_seed: int,
                               sample_index: int) -> tuple[dict, ...]:
    """按实体平移与缩放框，模拟规划误差：中心各轴平移 U(-center_shift, center_shift)（画面比例），宽高同乘
    U(scale_min, scale_max)；同一实体在一镜内各锚用同一组抽签，截到画面内。文字与实体 span 不变。

    抽签键是 (training seed, sample_index, 1-based shot, entity)。"""
    shift, low, high = jitter["center_shift"], jitter["scale_min"], jitter["scale_max"]
    rows = []
    for index, row in zip(shot_indices, layouts, strict=True):
        entities = []
        for entity in row["entities"]:
            key = (training_seed, sample_index, index + 1, entity["name"])
            dx = (2 * unit_draw("box_jitter_x", *key) - 1) * shift
            dy = (2 * unit_draw("box_jitter_y", *key) - 1) * shift
            scale = low + (high - low) * unit_draw("box_jitter_scale", *key)
            anchors = []
            for anchor in entity["anchors"]:
                x0, y0, x1, y1 = anchor["box"]
                cx = min(1.0, max(0.0, (x0 + x1) / 2 + dx))
                cy = min(1.0, max(0.0, (y0 + y1) / 2 + dy))
                hw, hh = (x1 - x0) * scale / 2, (y1 - y0) * scale / 2
                box = [max(0.0, cx - hw), max(0.0, cy - hh), min(1.0, cx + hw), min(1.0, cy + hh)]
                if not (box[0] < box[2] and box[1] < box[3]):
                    raise ValueError("box jitter produced an empty box")
                anchors.append({**anchor, "box": box})
            entities.append({**entity, "anchors": anchors})
        rows.append({**row, "entities": entities})
    return tuple(rows)


def require_box_jitter(jitter: Any) -> Mapping[str, float] | None:
    if jitter is None:
        return None
    if not isinstance(jitter, Mapping) or set(jitter) != {"center_shift", "scale_min", "scale_max"}:
        raise ValueError("box_jitter requires exactly center_shift, scale_min and scale_max")
    if any(type(jitter[k]) not in (int, float) for k in jitter):
        raise ValueError("box_jitter values must be numbers")
    if not (0 <= jitter["center_shift"] < 0.5 and 0 < jitter["scale_min"] <= 1 <= jitter["scale_max"] < 4):
        raise ValueError("box_jitter needs center_shift in [0, 0.5) and scale_min <= 1 <= scale_max")
    return {k: float(v) for k, v in jitter.items()}


def pack_span(frame_counts: Sequence[int], start: int) -> tuple[int, int]:
    """从 start 连续装至 228 帧或 4 镜容量；返回半开区间。"""
    if type(start) is not int or not 0 <= start < len(frame_counts):
        raise ValueError(f"start {start} is outside the block")
    frame_counts_over_segment_cap(frame_counts)
    end, frames = start, 0
    for count in frame_counts[start:start + SEGMENT_MAX_SHOTS]:
        if frames + count > SEGMENT_MAX_FRAMES:
            break
        end, frames = end + 1, frames + count
    if end == start:
        raise ValueError(f"shot {start} exceeds the segment capacity")
    return start, end


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


@dataclass(frozen=True)
class BlockRow:
    case_id: str
    frame_counts: tuple[int, ...]
    usable: tuple[int, ...]


#: 训练参考的唯一取法：同块窗外（窗前窗后）可用镜里随机抽张数、随机帧，见 :meth:`BlockDataset.references`。
#: 推理端的选帧规则（:mod:`mspc.data.reference_selection` 等）在训练分布之内，可在同一份权重上免训练比较。
#: 训练参考的选法：``random_frame`` 共有实体优先、层内随机（不看机位）；``distinct_camera_first`` 异机位优先、同机位只作兜底、
#: 空位用无关镜补回张数；``distinct_camera_only`` 同上但不兜底、不给同机位参考；``distinct_camera_shared_only`` 只给异机位
#: 共有实体参考、不兜底也不补无关镜，无异机位候选时参考置空退化为纯窗内首潜帧互读
#: （:func:`mspc.data.reference_routing.distinct_camera_first`）；``distinct_camera_character_coverage`` 异机位优先、按在场实体
#: 逐个补齐（缺的先补异机位、再补同机位）、只补一张同地点场景镜、不补无关镜
#: （:func:`mspc.data.reference_routing.distinct_camera_character_coverage`）。抽签键与上限相同，只改选哪几镜。
REFERENCE_POLICY_RANDOM_FRAME = "random_frame"
REFERENCE_POLICY_DISTINCT_CAMERA_FIRST = "distinct_camera_first"
REFERENCE_POLICY_DISTINCT_CAMERA_ONLY = "distinct_camera_only"
REFERENCE_POLICY_DISTINCT_CAMERA_SHARED_ONLY = "distinct_camera_shared_only"
REFERENCE_POLICY_DISTINCT_CAMERA_CHARACTER_COVERAGE = "distinct_camera_character_coverage"
REFERENCE_POLICIES = frozenset({REFERENCE_POLICY_RANDOM_FRAME, REFERENCE_POLICY_DISTINCT_CAMERA_FIRST,
                                REFERENCE_POLICY_DISTINCT_CAMERA_ONLY,
                                REFERENCE_POLICY_DISTINCT_CAMERA_SHARED_ONLY,
                                REFERENCE_POLICY_DISTINCT_CAMERA_CHARACTER_COVERAGE})
#: 参考时间格上可取的帧间隔：16fps 标注格上每 4 帧（0、4、8…），SAM 标签与裁剪框也在这些帧上。
REFERENCE_FRAME_GRID = 4
#: 长单镜要求的镜内活动（块标注 ``activity``，见 ``configs/data/block_annotation_v18.json``）。
LONG_SINGLE_SHOT_ACTIVITY = "action_or_state_change"


def shows_action_or_change(shot: Mapping[str, Any]) -> bool:
    """块标注里一镜有动作或镜内变化：``activity`` 为 ``action_or_state_change``，或 ``changes`` 非空。"""
    activity, changes = shot["activity"], shot["changes"]
    if not isinstance(activity, str) or not isinstance(changes, list):
        raise ValueError("block annotation shots need a string activity and a changes list")
    return activity == LONG_SINGLE_SHOT_ACTIVITY or bool(changes)


class BlockWindowSampler:
    """连续完整目标窗的唯一抽签 owner，先抽层、再在层内全部合法窗口中均匀抽。

    现行 ``shot_count_weights`` 对应 1／2／3／4及以上镜；最后一层不封顶镜数。
    此时 ``frame_buckets`` 仅给单个源帧预算，超预算的完整单镜仍独占一窗，不能与相邻镜拼接。
    历史时长档复现显式给 ``bucket_weights``，其余窗口枚举和无状态抽签共用。
    """

    def __init__(self, rows: Sequence[BlockRow], *, frame_buckets: Sequence[int], bucket_weights: Sequence[float] | None,
                 max_shots: int | None, seed: int, shot_count_weights: Sequence[float] | None,
                 long_single_shot: tuple[float, int, frozenset[tuple[str, int]]] | None = None):
        """``long_single_shot=(份额, 最少源帧, 有动作或变化的镜)``：抽到单镜层时，按份额只在长单镜窗里均匀取，
        其余照常在全部单镜窗里取；四层权重不变，抽签键与不给这一项时相同（单镜层以外的行逐行不变）。

        长单镜的唯一定义：单镜成窗、源帧不少于最少源帧，且该镜在第三项里（``(case_id, 块内镜下标)``，由
        :func:`shows_action_or_change` 从块标注得出）。Wan2.2 定稿训练（11-13）单镜层一半取不短于 5 秒、
        有动作或状态变化的长单镜（状态建立是最弱列，连续长镜头只占目标镜曝光 0.8%）。"""
        if (not isinstance(frame_buckets, (list, tuple)) or not frame_buckets
                or any(type(bound) is not int or bound < 1 for bound in frame_buckets)
                or any(a >= b for a, b in zip(frame_buckets, frame_buckets[1:]))):
            raise ValueError("frame_buckets must be increasing positive frame bounds")
        if shot_count_weights is not None:
            if bucket_weights is not None or len(frame_buckets) != 1 or max_shots is not None:
                raise ValueError("shot_count_weights require one source frame budget, no duration weights or shot limit")
            if (not isinstance(shot_count_weights, (list, tuple)) or len(shot_count_weights) != 4
                    or any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in shot_count_weights)
                    or sum(shot_count_weights) <= 0):
                raise ValueError("shot_count_weights require four finite nonnegative weights with positive sum")
            weights = shot_count_weights
        else:
            if (not isinstance(bucket_weights, (list, tuple)) or len(bucket_weights) != len(frame_buckets)
                    or any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in bucket_weights)
                    or sum(bucket_weights) <= 0):
                raise ValueError("bucket_weights must give one finite nonnegative weight per bucket with positive sum")
            weights = bucket_weights
        if max_shots is not None and (type(max_shots) is not int or max_shots < 1):
            raise ValueError("max_shots must be a positive integer or None")
        if type(seed) is not int:
            raise TypeError("sampling seed must be an integer")
        self.seed, self.max_shots = seed, max_shots
        self.max_frames = frame_buckets[-1]
        self.frame_buckets = tuple(frame_buckets)
        self.bucket_weights = None if bucket_weights is None else tuple(bucket_weights)
        self.shot_count_weights = None if shot_count_weights is None else tuple(shot_count_weights)
        self.cumulative_weights = tuple(accumulate(weights))
        windows = [[] for _ in weights]
        for row in rows:
            if any(type(count) is not int or count <= 0 for count in row.frame_counts):
                raise ValueError("shot frame counts must be positive integers")
            if shot_count_weights is None and any(row.frame_counts[position] > self.max_frames for position in row.usable):
                raise ValueError(f"{row.case_id}: complete usable shot exceeds configured frame_buckets capacity")
            usable = set(row.usable)
            for start in row.usable:
                if shot_count_weights is not None and row.frame_counts[start] > self.max_frames:
                    windows[0].append((row, start, start + 1))
                    continue
                stop, total = start, 0
                while (stop in usable and (max_shots is None or stop - start < max_shots)
                       and total + row.frame_counts[stop] <= self.max_frames):
                    total += row.frame_counts[stop]
                    stop += 1
                    bucket = (min(stop - start, 4) - 1 if shot_count_weights is not None
                              else bisect_left(self.frame_buckets, total))
                    windows[bucket].append((row, start, stop))
        self.windows = tuple(tuple(group) for group in windows)
        self.long_single_shot = None
        if long_single_shot is not None:
            share, min_frames, action_shots = long_single_shot
            if (shot_count_weights is None or type(share) is not float or not 0.0 < share < 1.0
                    or type(min_frames) is not int or min_frames < 1 or not isinstance(action_shots, frozenset)
                    or any(not isinstance(shot, tuple) or len(shot) != 2 or type(shot[0]) is not str
                           or type(shot[1]) is not int for shot in action_shots)):
                raise ValueError("long_single_shot needs shot-count layers, a share in (0, 1), a positive source frame "
                                 "minimum and the frozenset of (case_id, shot) with action or change")
            pool = tuple((row, start, stop) for row, start, stop in self.windows[0]
                         if row.frame_counts[start] >= min_frames and (row.case_id, start) in action_shots)
            if not pool:
                raise ValueError("long_single_shot requests single-shot windows that the pool does not have")
            self.long_single_shot, self.long_single_pool = (share, min_frames), pool
        for index, (weight, group) in enumerate(zip(weights, self.windows)):
            if weight > 0 and not group:
                label = (f"shot_count_weights requests layer {index + 1}"
                         if shot_count_weights is not None else f"bucket_weights requests windows up to {frame_buckets[index]} frames")
                raise ValueError(f"{label} but the pool has none")

    def bucket(self, frames: int) -> int:
        if type(frames) is not int or not 1 <= frames <= self.max_frames:
            raise ValueError(f"window frames must be in 1..{self.max_frames}")
        return bisect_left(self.frame_buckets, frames)

    def draw(self, data_id: int):
        if type(data_id) is not int or data_id < 0:
            raise ValueError("sample index must be a nonnegative integer")
        key = (self.seed, data_id)
        draw_name = "window_shot_count" if self.shot_count_weights is not None else "window_bucket"
        bucket = bisect_right(self.cumulative_weights, unit_draw(draw_name, *key) * self.cumulative_weights[-1])
        group = self.windows[bucket]
        if (bucket == 0 and self.long_single_shot is not None
                and unit_draw("window_long_single_shot", *key) < self.long_single_shot[0]):
            group = self.long_single_pool
        row, start, stop = group[int(unit_draw("window", *key) * len(group))]
        return row, (start, stop), key


@dataclass(frozen=True)
class SampleDraw:
    row: BlockRow
    epoch: int
    span: tuple[int, int]
    references: tuple[tuple[int, int], ...]  # block shot position, frame in the explicitly configured target grid
    reference_offset: tuple[int, int, int] | None
    key: tuple


@dataclass(frozen=True)
class SampleText:
    draw: SampleDraw
    prompts: tuple[str, ...]
    metadata: dict
    compiled_targets: tuple[dict, ...]


class BlockDataset(torch.utils.data.Dataset):
    stateless_index_samples = True

    def __init__(self, *, members_path: str, target_storage: str, media: Mapping[str, Any],
                 target_index_path: str | None, target_roots: Mapping[str, str] | None,
                 sample_budget: int, tokenizer_path: str, sampling_manifest_path: str,
                 segment_capacity: Mapping[str, Any],
                 reference_row_dropout_p: float | None, reference_policy: str,
                 target_mask_weights: Mapping[str, str] | None,
                 memory_prefix_options: Mapping[str, Any] | None, self_attention_policy: str, training_seed: int,
                 video_decode_num_threads: int, box_dropout_p: float | None,
                 box_jitter: Mapping[str, float] | None,
                 minimum_layout_shot_fraction: float | None = None,
                 planner_tokens: Mapping[str, Any] | None = None,
                 reference_entities: Mapping[str, Any] | None = None,
                 reference_crops: Mapping[str, Any] | None = None,
                 workload_cap: Mapping[str, int] | None = None,
                 omit_target_text: Sequence[str] = (),
                 picture_type_dropout_p: float | None = None,
                 target_latent_geometry: Mapping[str, int] | None = None,
                 text_admission_path: str | None = None,
                 required_sample_indices: Sequence[int] | None = None) -> None:
        from mspc.training.text_admission import text_admission_contract
        self.text_admission_contract = text_admission_contract(locals())
        if type(sample_budget) is not int or sample_budget <= 0 or type(training_seed) is not int:
            raise ValueError("sample_budget must be positive int and training_seed must be int")
        if (not isinstance(segment_capacity, Mapping) or set(segment_capacity) != {"max_frames"}
                or type(segment_capacity["max_frames"]) is not int or segment_capacity["max_frames"] < 1):
            raise ValueError("segment_capacity requires an explicit positive max_frames")
        self.max_frames = segment_capacity["max_frames"]
        self.media = require_training_media(media)
        self.height, self.width = self.media["height"], self.media["width"]
        if target_storage not in {"latents", "video"}:
            raise ValueError("target_storage must explicitly select latents or video")
        self.target_storage = target_storage
        if required_sample_indices is not None and target_storage != "latents":
            raise ValueError("cache admission applies only to production latent targets")
        self.required_sample_indices = (None if required_sample_indices is None
                                        else frozenset(required_sample_indices))
        self._online_vae_geometry_bound = False
        latent_geometry = None
        if target_storage == "latents":
            if (not isinstance(target_roots, Mapping) or not target_roots
                    or any(not isinstance(name, str) or not name.strip() or name != name.strip()
                           for name in target_roots) or target_index_path is None):
                raise ValueError("latent targets require an index and explicit non-empty source names in target_roots")
            if target_latent_geometry is not None:
                raise ValueError("latent targets derive geometry from their cache records")
        else:
            if target_index_path is not None or target_roots is not None:
                raise ValueError("video targets do not accept latent cache index or roots")
            if (not isinstance(target_latent_geometry, Mapping)
                    or set(target_latent_geometry) != {"channels", "spatial_stride"}
                    or type(target_latent_geometry["channels"]) is not int
                    or target_latent_geometry["channels"] < 1):
                raise ValueError("video targets require explicit VAE channels and spatial_stride")
            stride = target_latent_geometry["spatial_stride"]
            if self.height % stride or self.width % stride:
                raise ValueError("target pixels must divide the configured VAE spatial stride")
            latent_geometry = (target_latent_geometry["channels"], (self.height // stride, self.width // stride), stride)
        self.sample_budget, self.training_seed = sample_budget, training_seed
        if self_attention_policy not in WAN22_SELF_ATTENTION_POLICIES:
            raise ValueError(f"self_attention_policy must be one of {sorted(WAN22_SELF_ATTENTION_POLICIES)}")
        # 只决定每条样本的注意力帧对（负载上限与组批），不改抽签与文字。
        self.self_attention_policy = self_attention_policy
        self.references_enabled = memory_prefix_options is not None
        if reference_policy not in REFERENCE_POLICIES:
            raise ValueError(f"reference_policy must be one of {sorted(REFERENCE_POLICIES)}")
        self.reference_policy = reference_policy
        if box_dropout_p is not None and (type(box_dropout_p) not in (int, float) or not 0 <= box_dropout_p < 1):
            raise ValueError("box_dropout_p must be in [0, 1)")
        self.box_dropout_p = box_dropout_p
        omitted = frozenset(omit_target_text)
        if len(omitted) != len(omit_target_text) or not omitted <= TARGET_TEXT_OMISSIONS:
            raise ValueError(f"omit_target_text must be distinct items of {sorted(TARGET_TEXT_OMISSIONS)}")
        if "layouts" in omitted and (box_dropout_p is not None or minimum_layout_shot_fraction is not None):
            raise ValueError("omitting layouts excludes box_dropout_p and minimum_layout_shot_fraction")
        self.omit_target_text = omitted
        if picture_type_dropout_p is not None and (type(picture_type_dropout_p) not in (int, float)
                                                   or not 0 < picture_type_dropout_p < 1 or "picture_type" in omitted):
            raise ValueError("picture_type_dropout_p must be in (0,1) and excludes omitting picture_type")
        self.picture_type_dropout_p = picture_type_dropout_p
        self.box_jitter = require_box_jitter(box_jitter)
        if self.box_jitter is not None and box_dropout_p is None:
            raise ValueError("box_jitter applies to layout boxes and requires box_dropout_p")
        if (reference_row_dropout_p is not None) != self.references_enabled:
            raise ValueError("reference text dropout and memory_prefix_options must be set together")
        self.memory_options = MemoryPrefixModelOptions.from_mapping(memory_prefix_options) if self.references_enabled else None
        frame_entities = self.memory_options is not None and self.memory_options.row_policy in FRAME_ENTITY_ROW_POLICIES
        if frame_entities != (reference_entities is not None):
            raise ValueError("frame-entity reference rows and reference_entities labels must be declared together")
        self.shot_objects_in_rows = (self.memory_options is not None and self.memory_options.row_policy
                                     == MEMORY_ROW_POLICY_FRAME_ENTITIES_WITH_SHOT_OBJECTS_V1)
        self.reference_entities = (None if reference_entities is None else
                                   ReferenceEntities(reference_entities, range(sample_budget)))
        self.reference_crops = (None if reference_crops is None else
                                ReferenceCrops(reference_crops, range(sample_budget)))
        if self.references_enabled and (type(reference_row_dropout_p) not in (int, float)
                                       or not 0 <= reference_row_dropout_p < 1):
            raise ValueError("reference_row_dropout_p must be in [0,1)")
        self.reference_row_dropout_p = reference_row_dropout_p
        self.video_decode_num_threads = video_decode_num_threads
        mask_key_sets = ({"index_path", "root", "weighting"}, {"spatial_root", "weighting"})
        if target_mask_weights is not None and (not isinstance(target_mask_weights, Mapping)
                or set(target_mask_weights) not in mask_key_sets or target_mask_weights["weighting"] not in WEIGHTINGS):
            raise ValueError("target_mask_weights must explicitly select pooled weights or original SAM RLE, or null")
        pooled_masks = target_mask_weights is not None and set(target_mask_weights) == mask_key_sets[0]
        if pooled_masks and (self.height, self.width, self.media["target_fps"]) != (480, 832, 16):
            raise ValueError("pooled SAM weight files are bound to 480x832/16fps; compile other media from original RLE")
        self.target_mask_weights = (TargetMaskWeights(**target_mask_weights)
                                   if pooled_masks else None)
        self.target_mask_sources = (dict(target_mask_weights)
                                   if target_mask_weights is not None and not pooled_masks else None)
        self.sampling_manifest = tuple(_jsonl(Path(sampling_manifest_path)))
        if len(self.sampling_manifest) != sample_budget:
            raise ValueError("sampling manifest length differs from sample_budget")
        all_required_targets = sampled_target_positions(self.sampling_manifest)
        required_targets = (all_required_targets if required_sample_indices is None else
                            sampled_target_positions(self.sampling_manifest, sample_indices=required_sample_indices))
        self.roots = {} if target_roots is None else {name: Path(root) for name, root in target_roots.items()}
        self.target_index = {}
        for entry in (() if target_storage == "video" else _jsonl(Path(target_index_path))):
            key = shot_key(entry)
            if key in self.target_index or entry["origin"] not in self.roots:
                raise ValueError(f"duplicate target key or unknown origin: {key}")
            self.target_index[key] = entry
        if {entry["origin"] for entry in self.target_index.values()} != set(self.roots):
            raise ValueError("target_roots must exactly cover the origins used by the target index")
        records = {}
        for origin in {r["origin"] for r in self.target_index.values()}:
            manifest = json.loads((self.roots[origin] / "manifest.json").read_text())
            for record in manifest["records"]:
                key = (origin, record["cache_file"])
                if key in records:
                    raise ValueError(f"duplicate target file: {key}")
                records[key] = record
        self.members, self.annotations, self.targets = {}, {}, {}
        self.reference_frame_counts = {}
        rows = []
        target_encoding = None
        for member in _jsonl(Path(members_path)):
            case, shots = member["case_id"], member["shots"]
            if member["schema_version"] != 1 or not case or case in self.members:
                raise ValueError(f"invalid member schema or duplicate case: {case}")
            annotation = member["annotation"]
            expected_ids = list(range(1, len(shots) + 1))
            if len(annotation["shots"]) != len(shots) or [s["shot"] for s in shots] != expected_ids:
                raise ValueError(f"{case}: annotation/member shot count/order differs")
            if any(type(s["usable"]) is not bool for s in shots):
                raise ValueError(f"{case}: usable must be bool")
            scene = {**member["scene"], "shots": shots}
            usable = tuple(i for i, shot in enumerate(shots) if shot["usable"])
            target_frame_counts = member_target_frame_counts(member, media=self.media)
            reference_frame_counts = (target_frame_counts if self.reference_grid_media == self.media else
                                      member_target_frame_counts(member, media=self.reference_grid_media))
            if any(reference_frame_counts[i] != shots[i]["frames"] for i in usable):
                raise ValueError(f"{case}: reference time grid differs from the original annotation frames")
            self.reference_frame_counts[case] = reference_frame_counts
            if not usable:
                raise ValueError(f"{case}: member has no usable shots")
            for position in usable:
                shot = shots[position]
                if annotation["shots"][position]["shot"] != position + 1:
                    raise ValueError(f"{case}: usable annotation shot order differs")
                if shot["frames"] != shot["render_frames"]:
                    raise ValueError(f"{case}: invalid shot usable/frames")
                if shot["frames"] > SEGMENT_MAX_FRAMES:
                    raise ValueError(f"{case}: usable shot exceeds capacity")
                latent_frames(shot["frames"])
                if shot_row(scene, position, media=self.media)["shot_groups"][0][1] != target_frame_counts[position]:
                    raise ValueError("derived target count differs from the source frame planner")
                if target_frame_counts[position] > self.max_frames:
                    raise ValueError(f"{case} shot {position + 1}: complete target shot exceeds configured capacity")
                if target_storage == "video" or position not in required_targets.get(case, ()):
                    continue
                if shot_key(shot) not in self.target_index:
                    raise ValueError(f"{case} shot {position + 1}: sampled target cache is missing")
                entry = self.target_index[shot_key(shot)]
                if self.required_sample_indices is not None and not (self.roots[entry["origin"]] / entry["file"]).is_file():
                    raise FileNotFoundError(f"{case} shot {position + 1}: admitted target cache file is missing")
                record = records[(entry["origin"], entry["file"])]
                _, groups = record_groups(record)
                interval = [entry["latent_start"], entry["latent_end"]]
                if interval not in groups:
                    raise ValueError(f"{case}: target interval is not a complete encoded shot")
                problems = reconcile(entry, record, groups.index(interval), scene, position, media=self.media)
                if problems:
                    raise ValueError(f"{case} shot {position}: target source mismatch {problems}")
                cache_media = record["provenance"]["media"]
                if (cache_media["sampling"]["target_fps"] != self.media["target_fps"]
                        or cache_media["preprocessing"]["spatial"][0]["output_size"] != [self.width, self.height]):
                    raise ValueError(f"{case}: target cache pixels/fps differ from explicit task media")
                shape = record["input_latents_shape"]
                if (len(shape) != 5 or shape[0] != 1
                        or any(type(size) is not int or size <= 0 for size in shape)):
                    raise ValueError(f"{case}: target does not use the renderer latent shape")
                height, width = shape[3:]
                if (self.height % height or self.width % width or self.height // height != self.width // width):
                    raise ValueError(f"{case}: target latent grid does not exactly divide the training pixels")
                geometry = (shape[1], (height, width), self.height // height)
                if latent_geometry is not None and geometry != latent_geometry:
                    raise ValueError(f"{case}: target latent channels/grid differ across the pool")
                latent_geometry = geometry
                encoding = {name: record["provenance"]["encoding"][name]
                            for name in ("dtype", "pixel_conversion", "tiled", "tile_size", "tile_stride")}
                # 首批与补建缓存的运行根不同，同一权重的 path 会不同；跨池沿用生产端 revision。
                # 在线参考VAE与此合同在runner启动时核对；worker不加载模型。
                contract = (record["provenance"]["models"]["vae"]["revision"], encoding)
                if target_encoding is not None and target_encoding != contract:
                    raise ValueError(f"{case}: target VAE/encoding differs across the pool")
                target_encoding = contract
                self.targets[(case, position)] = (entry, record)
                if self.target_mask_weights is not None:
                    self.target_mask_weights.record_for(member, position,
                                                       latent_spatial_shape=geometry[1])
            self.members[case], self.annotations[case] = member, annotation
            rows.append(BlockRow(case, tuple(target_frame_counts), usable))
        if not rows:
            raise ValueError("no block members")
        self.rows, self.case_ids = tuple(rows), tuple(self.members)
        self.row_by_case = {row.case_id: row for row in self.rows}
        for index, sample in enumerate(self.sampling_manifest):
            if sample["sample_index"] != index:
                raise ValueError("sampling manifest indices must be consecutive and ordered")
            row = self.row_by_case[sample["case_id"]]
            start, end = sample["span"]
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(row.frame_counts)
                    or not set(range(start, end)) <= set(row.usable)
                    or sum(row.frame_counts[start:end]) > self.max_frames):
                raise ValueError(f"sample {index}: invalid target span")
        self.planner_tokens = None if planner_tokens is None else PlannerTokenData(planner_tokens, sample_budget)
        self.minimum_layout_shot_fraction = minimum_layout_shot_fraction
        if minimum_layout_shot_fraction is not None:
            if type(minimum_layout_shot_fraction) not in (int, float) or not 0 < minimum_layout_shot_fraction <= 1:
                raise ValueError("minimum_layout_shot_fraction must be in (0,1]")
            from mspc.models.spatial_layout_inputs import layout_shot_coverage
            covered, total = layout_shot_coverage(self.annotations, self.sampling_manifest)
            fraction = covered / total if total else 0
            if fraction < minimum_layout_shot_fraction:
                raise ValueError(
                    f"layout coverage {covered}/{total}={fraction:.4f} is below "
                    f"{minimum_layout_shot_fraction}")
            missing = [sample["case_id"] for sample in self.sampling_manifest
                       if "layouts" not in self.annotations[sample["case_id"]]]
            if missing:
                raise ValueError(f"{len(set(missing))} sampled blocks have no layouts field")
        self.token_counter = load_umt5_token_counter(Path(tokenizer_path))
        self._draws = {}
        self.target_encoding = target_encoding
        self.latent_channels, self.latent_spatial_shape, self.vae_spatial_stride = latent_geometry
        self.metadata_summary = {"schema": "block_video" if target_storage == "video" else "block_latent_cache",
            "target_storage": target_storage, "media": dict(self.media), "blocks": len(rows),
            "usable_shots": sum(len(row.usable) for row in rows), "sample_budget": sample_budget,
            "sampled_target_shots": sum(map(len, all_required_targets.values())),
            "cache_ready_target_shots": sum(map(len, required_targets.values())),
            "cache_ready_sample_count": sample_budget if self.required_sample_indices is None else len(self.required_sample_indices),
            "segment_capacity": dict(segment_capacity),
            "sampling_manifest_path": sampling_manifest_path,
            "target_mask_weights": None if target_mask_weights is None else dict(target_mask_weights),
            "block_sampling": "frozen_sample_manifest",
            "latent_geometry": {"channels": self.latent_channels,
                                "spatial_shape": list(self.latent_spatial_shape),
                                "vae_spatial_stride": self.vae_spatial_stride}}
        self.metadata_summary.update(reference_policy=reference_policy, omit_target_text=sorted(self.omit_target_text),
                                     reference_temporal_grid={"target_fps": self.reference_grid_media["target_fps"],
                                                              "vae_temporal_stride": self.reference_grid_media["vae_temporal_stride"]},
                                     target_spatial_media={"height": self.height, "width": self.width},
                                     picture_type_dropout_p=picture_type_dropout_p,
                                     self_attention_policy=self_attention_policy,
                                     reference_row_policy=None if self.memory_options is None else self.memory_options.row_policy)
        if target_storage == "video":
            self.metadata_summary["latent_geometry"]["verified_from_loaded_vae"] = False
        if minimum_layout_shot_fraction is not None:
            self.metadata_summary["layout_shot_coverage"] = {
                "covered": covered, "total": total, "minimum": minimum_layout_shot_fraction}
        self.text_admission = None
        if text_admission_path is not None:
            from mspc.training.text_admission import read_text_admission
            self.text_admission = read_text_admission(text_admission_path, sample_budget=sample_budget,
                                                     contract=self.text_admission_contract)
        text_eligible = tuple(range(sample_budget) if self.text_admission is None else
                              self.text_admission["eligible_sample_indices"])
        self.metadata_summary["text_admission"] = {"path": text_admission_path,
            "token_limit": UMT5_PROMPT_TOKEN_BUDGET, "original_sample_budget": sample_budget,
            "eligible_samples": len(text_eligible), "excluded_samples": sample_budget - len(text_eligible)}
        over_cap = frozenset()
        if workload_cap is not None:
            # 抽样入口按每条样本实际负载（目标潜帧＋参考张数、注意力帧对）剔除超上限的样本，计数进 run 快照。
            loads = self.sample_workloads()
            over_cap = frozenset(index for index in text_eligible
                                 if any(loads[index][name] > workload_cap[name] for name in workload_cap))
            self.metadata_summary["workload_cap"] = {**workload_cap, "excluded_samples": len(over_cap),
                "max_latent_frames_kept": max(loads[i]["latent_frames"] for i in text_eligible if i not in over_cap),
                "max_attention_frame_pairs_kept": max(loads[i]["attention_frame_pairs"]
                                                      for i in text_eligible if i not in over_cap)}
        self.eligible_sample_indices = tuple(index for index in text_eligible if index not in over_cap)
        self.excluded_sample_indices = frozenset(set(range(sample_budget)) - set(self.eligible_sample_indices))

    def __len__(self):
        return self.sample_budget

    def bind_online_vae_geometry(self, latent_channels: int, vae_spatial_stride: int) -> None:
        """主进程加载所选 VAE 后核对实际几何；worker 不加载模型、不猜 VAE 类型。"""
        if self.target_storage != "video":
            raise ValueError("only video targets bind an online VAE")
        shape = (self.height // vae_spatial_stride, self.width // vae_spatial_stride)
        if self.height % vae_spatial_stride or self.width % vae_spatial_stride:
            raise ValueError("online VAE stride must exactly divide configured target pixels")
        if (type(latent_channels) is not int or latent_channels != self.latent_channels
                or vae_spatial_stride != self.vae_spatial_stride or shape != self.latent_spatial_shape):
            raise ValueError("online VAE latent geometry differs from the explicit video target geometry")
        self._online_vae_geometry_bound = True
        self.metadata_summary["latent_geometry"]["verified_from_loaded_vae"] = True

    def target_draw(self, data_id: int):
        """只抽成员和目标段；长度分桶不读参考文件、不调用编码器。"""
        if type(data_id) is not int or not 0 <= data_id < len(self):
            raise IndexError(data_id)
        sample = self.sampling_manifest[data_id]
        return self.row_by_case[sample["case_id"]], tuple(sample["span"]), (self.training_seed, data_id)

    @property
    def reference_grid_media(self):
        """随机参考保留旧16fps取样时刻（那里有 SAM 标签与裁剪框）；RGB空间几何仍用本次目标media。"""
        return ANNOTATION_MEDIA

    def reference_plan(self, row, span, key):
        """可选的参考帧（块内镜下标 → 参考时间格帧号）和这条样本取几张。

        先做整组参考丢弃；候选是同块窗外（窗前窗后）的全部可用镜，每镜取参考时间格 0、4、8… 帧，
        张数在 1…min(frames_max, 候选镜数) 中均匀抽。SAM 不筛 RGB。
        """
        if not self.references_enabled or unit_draw("reference_group", *key) < self.memory_options.dropout_p:
            return {}, 0
        frames = {position: list(range(0, self.reference_frame_counts[row.case_id][position], REFERENCE_FRAME_GRID))
                  for position in row.usable if not span[0] <= position < span[1]}
        count = (1 + int(unit_draw("reference_count", *key) * min(self.memory_options.frames_max, len(frames)))
                 if frames else 0)
        return frames, count

    def references(self, data_id: int, span, key) -> tuple[tuple[int, int], ...]:
        """本样本实际给模型的参考帧（块内镜下标, 参考格帧号），最终按镜序排列。

        ``random_frame``：与目标在场实体相交的镜优先、层内随机，不足时补无关镜到已抽张数；``distinct_camera_first``（兜底）与
        ``distinct_camera_only``（不兜底）：见 :func:`mspc.data.reference_routing.distinct_camera_first`。每镜在候选帧里均匀选一帧。
        抽签键名（reference_group／reference_count／reference_pick／reference_frame）与 06776a926 逐项相同。
        """
        row = self.row_by_case[self.sampling_manifest[data_id]["case_id"]]
        frames, count = self.reference_plan(row, span, key)
        shots = self.annotations[row.case_id]["shots"]
        wanted = {entry["name"] for index in range(*span) for entry in shots[index]["present"]}

        def priority(position):
            shared = wanted.intersection(entry["name"] for entry in shots[position]["present"])
            return (not shared, unit_draw("reference_pick", *key, position))

        if self.reference_policy == REFERENCE_POLICY_DISTINCT_CAMERA_CHARACTER_COVERAGE:
            from mspc.data.reference_routing import distinct_camera_character_coverage
            ordered = sorted(frames, key=lambda position: unit_draw("reference_pick", *key, position))
            chosen = distinct_camera_character_coverage(shots, candidates=ordered, target_shots=range(*span),
                                                        count=count, limit=self.memory_options.frames_max)
        elif self.reference_policy != REFERENCE_POLICY_RANDOM_FRAME:
            from mspc.data.reference_routing import distinct_camera_first
            ordered = sorted(frames, key=lambda position: unit_draw("reference_pick", *key, position))
            chosen = distinct_camera_first(
                shots, candidates=ordered, target_shots=range(*span), count=count, limit=self.memory_options.frames_max,
                same_camera_fallback=self.reference_policy == REFERENCE_POLICY_DISTINCT_CAMERA_FIRST,
                fill_unrelated=self.reference_policy != REFERENCE_POLICY_DISTINCT_CAMERA_SHARED_ONLY)
        else:
            chosen = sorted(frames, key=priority)[:count]
        return tuple((position, frames[position][int(unit_draw("reference_frame", *key, position) * len(frames[position]))])
                     for position in sorted(chosen))

    def draw(self, data_id: int) -> SampleDraw:
        if type(data_id) is not int or not 0 <= data_id < len(self):
            raise IndexError(data_id)
        if data_id in self._draws:
            return self._draws[data_id]
        row, (start, end), key = self.target_draw(data_id)
        refs, offset = self.references(data_id, (start, end), key), None
        if refs:
            offset = self.memory_options.reference_offset.draw(tuple(
                unit_draw(f"reference_offset_{axis}", *key) for axis in ("dt", "dh", "dw")))
        draw = SampleDraw(row, data_id // len(self.rows), (start, end), refs, offset, key)
        self._draws[data_id] = draw
        return draw

    def reference_source_frames(self, draw: SampleDraw) -> list[int]:
        """参考时间格→原片源帧；标签与解码共用，不把参考16fps帧号交给目标24fps格。"""
        member = self.members[draw.row.case_id]
        scene = {**member["scene"], "shots": member["shots"]}
        grids = {shot: planned_row(scene, shot, media=self.reference_grid_media)["source_frame_indices_by_shot"][0]
                 for shot, _ in draw.references}
        return [grids[shot][frame] for shot, frame in draw.references]

    def text(self, data_id: int) -> SampleText:
        draw = self.draw(data_id)
        annotation = self.annotations[draw.row.case_id]
        source_entities = (None if self.reference_entities is None else self.reference_entities.for_draw(
            data_id, draw, source_frames=self.reference_source_frames(draw)))
        references = ([reference_text_with_entity_spans(row["entities"], characters=annotation["characters"],
                          setting=reference_setting(annotation, row["shot"] - 1),
                          shot_objects=shot_object_phrases(annotation, row["shot"] - 1)
                          if self.shot_objects_in_rows else ())["text"]
                       for row in source_entities]
                      if self.reference_entities is not None else [reference_text() for _ in draw.references])
        # 四组共同保留同一完整方位文字；位置token是另一个显式研究的条件通路。
        # 方位句按样本整组丢弃的现有增强仍有效：推理时计划会缺会错。
        layout_dropped = (self.box_dropout_p is not None
                          and unit_draw("box_dropout", *draw.key) < self.box_dropout_p)
        # 画风句按样本整组丢弃：推理的给定分镜表没有画风，规划器计划可能有；两种都要见过。
        picture_type_dropped = (self.picture_type_dropout_p is not None
                                and unit_draw("picture_type_dropout", *draw.key) < self.picture_type_dropout_p)
        annotation = without_target_text(annotation, self.omit_target_text | (
            {"picture_type"} if picture_type_dropped else set()))
        target_annotation = _without_layouts(annotation) if layout_dropped else annotation
        row_dropped = bool(references) and unit_draw("reference_row", *draw.key) < self.reference_row_dropout_p
        routing = None
        if draw.references and self.memory_options.attention_policy in ROUTED_MEMORY_ATTENTION_POLICIES:
            routing, _ = reference_readers(self.memory_options.attention_policy, annotation["shots"],
                                           reference_shots=[shot for shot, _ in draw.references],
                                           target_shots=range(*draw.span))
        slots = reference_slot_metadata(annotation["shots"], target_shots=range(*draw.span),
            reference_count=len(draw.references), reference_entities=None if row_dropped else source_entities,
            routing=routing)
        compiled_targets = [shot_text_with_entity_spans(target_annotation, shot, references=refs)
                            for shot, refs in zip(range(*draw.span), slots, strict=True)]
        targets = [row["text"] for row in compiled_targets]
        token_counts = [self.token_counter(text) for text in references + targets]
        if row_dropped:
            references = ["" for _ in references]
        metadata = {"sample_index": data_id,
            "flow_draw_seed": int(unit_draw("flow_noise", self.training_seed, draw.epoch, data_id) * 2**63),
            "reference_text_dropped": row_dropped, "layout_text_dropped": layout_dropped,
            "picture_type_text_dropped": picture_type_dropped,
            "reference_group_dropped": self.references_enabled and unit_draw("reference_group", *draw.key) < self.memory_options.dropout_p,
            "actual_token_counts": [self.token_counter(t) for t in references + targets],
            "case_id": draw.row.case_id, "epoch": draw.epoch, "span": list(draw.span),
            "references": [[shot + 1, frame] for shot, frame in draw.references],
            "reference_offset": list(draw.reference_offset) if draw.reference_offset is not None else None,
            "target_reference_slots": [[dict(reference) for reference in row] for row in slots],
            "opening_conflicts": [conflict for row in compiled_targets for conflict in row["opening_conflicts"]],
            "token_counts": token_counts}
        if self.reference_entities is not None:
            metadata.update(reference_visible_ids=[[entity["name"] for entity in row["entities"]]
                                                  for row in source_entities],
                reference_point_count=(0 if row_dropped else sum(entity["point_cell"] is not None
                    for row in source_entities for entity in row["entities"])),
                reference_alignment_cells=(0 if row_dropped else sum(len(row["alignment_cells"])
                                                                    for row in source_entities)))
        return SampleText(draw, tuple(references + targets), metadata,
                          tuple({"shot_index": shot, **compiled}
                                for shot, compiled in zip(range(*draw.span), compiled_targets, strict=True)))

    def latent_frame_upper_bounds(self):
        """实际参考张数＋目标潜帧，就是模型输入的潜帧数；分桶不读图片。"""
        lengths = []
        for index in range(len(self)):
            row, span, key = self.target_draw(index)
            lengths.append(len(self.references(index, span, key))
                           + sum(latent_frames(n) for n in row.frame_counts[slice(*span)]))
        return tuple(lengths)

    def training_workloads(self, *, self_attention_policy: str) -> tuple[dict[str, int], ...]:
        """从实际目标／参考抽签计算组批负载，不读取像素、文字或编码器。

        标定对应HoloCine首潜帧互读及按镜参考路由：线性层处理T+R，自注意力帧对数为R+Σl_i(l_i+K+r_i)，
        包含本镜首潜帧的重复键。复用consumer的参考选取和路由owner，不按参考上限或完整窗平方估算。

        首帧通信组（含首帧孪生帧）沿用同一估计：它的实际注意力帧对更少（R+Σ(l_i²+K+r_i)），估计偏高只影响
        各 rank 的负载均衡；用同一估计才得到与现役组逐位相同的抽签、组批与样本顺序，消融只差窗内读法。
        """
        if self_attention_policy not in ("holocine_first_frame", "first_frame_communication"):
            raise ValueError("calibrated workload costs require holocine_first_frame or "
                             "first_frame_communication self attention")
        if (self.references_enabled and self.memory_options.attention_policy
                not in (MEMORY_ATTENTION_POLICY_ROUTED_SELF_ONLY_V1, MEMORY_ATTENTION_POLICY_WINDOW_SHARED_SELF_ONLY_V1)):
            raise ValueError("calibrated workload costs require routed_self_only_v1 or window_shared_self_only_v1 references")
        if self_attention_policy != self.self_attention_policy:
            raise ValueError(f"calibrated workload costs for {self_attention_policy!r} differ from the dataset's "
                             f"self_attention_policy {self.self_attention_policy!r}")
        return self.sample_workloads()

    def sample_workloads(self) -> tuple[dict[str, int], ...]:
        """每条样本的模型输入潜帧（目标＋参考张数 T+R）与自注意力帧对，只读元数据（负载上限与组批共用）。

        帧对按本数据集的 ``self_attention_policy``（l_i 是目标镜 i 的潜帧，K 镜数，r_i 路由给镜 i 的参考张数，
        参考只读自己各计 1）：
        - ``holocine_first_frame``／``first_frame_communication``：R + Σ l_i (l_i + K + r_i)（见 :meth:`training_workloads`）；
        - ``full``（窗内全互读，按镜路由的参考走 ``routed_shot_attention`` 的 ``all_targets`` 窗口）：每镜读窗内
          全部目标潜帧 T 与路由给它的参考，R + Σ l_i (T + r_i)；整组丢弃参考时就是 T²。
        """
        if self.self_attention_policy not in ("holocine_first_frame", "first_frame_communication", "full"):
            raise ValueError(f"sample workloads cover holocine_first_frame, first_frame_communication and full "
                             f"self-attention, not {self.self_attention_policy!r}")
        if self.references_enabled and self.memory_options.attention_policy not in ROUTED_MEMORY_ATTENTION_POLICIES:
            raise ValueError(f"sample workloads require routed references, not {self.memory_options.attention_policy!r}")
        all_targets = self.self_attention_policy == "full"
        workloads = []
        for index in range(len(self)):
            row, span, key = self.target_draw(index)
            lengths = tuple(latent_frames(n) for n in row.frame_counts[slice(*span)])
            references = self.references(index, span, key)
            if references:
                routing, _ = reference_readers(self.memory_options.attention_policy, self.annotations[row.case_id]["shots"],
                    reference_shots=[shot for shot, _ in references], target_shots=range(*span))
                visible = tuple(len(entry) for entry in routing)
            else:
                visible = (0,) * len(lengths)
            reference_count, shot_count, target_frames = len(references), len(lengths), sum(lengths)
            workloads.append({
                "latent_frames": target_frames + reference_count,
                "attention_frame_pairs": reference_count + sum(
                    length * ((target_frames if all_targets else length + shot_count) + count)
                    for length, count in zip(lengths, visible, strict=True)),
            })
        return tuple(workloads)

    def _target_latent(self, case, position):
        entry, record = self.targets[(case, position)]
        path = self.roots[entry["origin"]] / entry["file"]
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"input_latents"}:
                raise ValueError(f"{path}: invalid target tensor keys")
            require_matching_provenance(json.loads((handle.metadata() or {})["provenance"]), record["provenance"])
            tensor = handle.get_slice("input_latents")
            if tensor.get_dtype() != "BF16" or tensor.get_shape() != record["input_latents_shape"]:
                raise ValueError(f"{path}: target shape/dtype differs")
            return tensor[:, :, entry["latent_start"]:entry["latent_end"]]

    def _target_video(self, case: str, span: tuple[int, int]) -> torch.Tensor:
        """读同一冻结窗口的原像素；取帧、FFmpeg 顺序解码、裁切均调用既有 producer owner。"""
        from mspc.data.scene_cache import read_scene_video

        member = self.members[case]
        scene = {**member["scene"], "shots": member["shots"]}
        rows = [shot_row(scene, position, media=self.media) for position in range(*span)]
        frame_indices = [frame for row in rows for frame in row["source_frame_indices_by_shot"][0]]
        row = rows[0]
        video = read_scene_video({
            "video": reroot_source_path(row["video"], location="online target source video"),
            "_scene_metadata": {name: row[name] for name in
                                ("source_num_frames", "source_width", "source_height", "source_fps")},
            "_scene_boxes": {"train_crop_box": row["train_crop_box"]},
        }, frame_indices, output_mode="uint8", video_decode_num_threads=self.video_decode_num_threads, media=self.media)
        if video.dtype != torch.uint8 or tuple(video.shape) != (3, len(frame_indices), self.height, self.width):
            raise ValueError("online target reader must return the exact uint8 training pixel grid")
        return video

    def _attach_layouts(self, sample, case, draw, item, compiled_targets):
        # 声明了覆盖率下限时，缺 layouts 字段直接失败，不再把空 B 当成已监督；整组丢方位句是增强，不是缺标注。
        dropped = sample["_sample_metadata"]["layout_text_dropped"]
        annotation = without_target_text(self.annotations[case], self.omit_target_text | (
            {"picture_type"} if sample["_sample_metadata"]["picture_type_text_dropped"] else set()))
        layouts = compile_spatial_layout_inputs(
            _without_layouts(annotation) if dropped else annotation, range(*draw.span),
            allow_unannotated=dropped or self.minimum_layout_shot_fraction is None,
            compiled_rows=compiled_targets)
        if self.box_jitter is not None:
            layouts = jitter_layout_entity_boxes(layouts, range(*draw.span), jitter=self.box_jitter,
                                                 training_seed=self.training_seed, sample_index=item)
        sample["target_layouts"] = layouts

    def __getitem__(self, item):
        if item in self.excluded_sample_indices:
            raise ValueError(f"sample {item}: excluded by the 512-token text admission or the workload cap")
        if self.required_sample_indices is not None and item not in self.required_sample_indices:
            raise ValueError(f"sample {item} is outside the explicit cache stage admission set")
        if self.target_storage == "video" and not self._online_vae_geometry_bound:
            raise RuntimeError("bind the loaded online VAE geometry before reading video target samples")
        started = time.perf_counter()
        text = self.text(item)
        selection_seconds = time.perf_counter() - started
        if any(n > UMT5_PROMPT_TOKEN_BUDGET for n in text.metadata["actual_token_counts"]):
            raise ValueError(f"sample {item}: text exceeds {UMT5_PROMPT_TOKEN_BUDGET} umT5 tokens")
        draw, case = text.draw, text.draw.row.case_id
        read_start = time.perf_counter()
        target = ({"video": self._target_video(case, draw.span)} if self.target_storage == "video" else
                  {"input_latents": torch.cat([self._target_latent(case, i) for i in range(*draw.span)], dim=2)})
        target_read_seconds = time.perf_counter() - read_start
        mask_weights, mask_metadata = None, None
        mask_started = time.perf_counter()
        if self.target_mask_weights is not None or self.target_mask_sources is not None:
            pairs = (compile_member_target_mask_weights(
                member=self.members[case], positions=range(*draw.span),
                vae_spatial_stride=self.vae_spatial_stride, media=self.media, **self.target_mask_sources)
                if self.target_mask_sources is not None else
                [self.target_mask_weights.read(self.members[case], i,
                                                 latent_spatial_shape=self.latent_spatial_shape)
                     for i in range(*draw.span)])
            mask_weights = torch.cat([weights for weights, _ in pairs], dim=2)
            mask_metadata = [metadata for _, metadata in pairs]
        mask_read_seconds = time.perf_counter() - mask_started
        count = len(draw.references)
        reference_images = torch.empty((0, 3, self.height, self.width), dtype=torch.uint8)
        reference_timing = {f"reference_{stage}_seconds": 0.0
                            for stage in ("open", "index", "decode", "crop", "read")}
        routing, reasons, shifted_slots = None, None, None
        if count and self.memory_options.attention_policy in ROUTED_MEMORY_ATTENTION_POLICIES:
            routing, reasons = reference_readers(self.memory_options.attention_policy, self.annotations[case]["shots"],
                                                 reference_shots=[shot for shot, _ in draw.references],
                                                 target_shots=range(*draw.span))
        if count:
            member = self.members[case]
            source_indices = self.reference_source_frames(draw)
            reference_images, reference_timing = read_reference_frames(
                member["scene"], source_indices, video_decode_num_threads=self.video_decode_num_threads, media=self.media)
            if self.reference_crops is not None and self.reference_crops.same_camera_shift is not None:
                # 同机位参考整帧缩小并贴到一侧（不再裁剪），其余参考照旧按离线表裁剪。
                from mspc.data.reference_crops import same_camera_slots, shift_same_camera
                flags = same_camera_slots(self.annotations[case]["shots"],
                                          reference_shots=[shot for shot, _ in draw.references],
                                          target_shots=range(*draw.span), routing=routing)
                boxes = [None if flag else box for flag, box in zip(flags, self.reference_crops.for_draw(item, draw),
                                                                    strict=True)]
                reference_images = apply_crops(reference_images, boxes)
                draws = [tuple(unit_draw(f"reference_shift_{part}", *draw.key, slot)
                               for part in ("scale", "x", "y", "side")) for slot in range(count)]
                reference_images = shift_same_camera(reference_images, flags, draws,
                                                     **self.reference_crops.same_camera_shift)
                shifted_slots = [slot for slot, flag in enumerate(flags) if flag]
            elif self.reference_crops is not None:
                boxes = self.reference_crops.for_draw(item, draw)
                if self.reference_crops.only_same_camera:
                    from mspc.data.reference_crops import select_same_camera_boxes
                    boxes = select_same_camera_boxes(
                        boxes, shots=self.annotations[case]["shots"],
                        reference_shots=[shot for shot, _ in draw.references],
                        target_shots=range(*draw.span), routing=routing)
                reference_images = apply_crops(reference_images, boxes)
        rgb_lengths = draw.row.frame_counts[slice(*draw.span)]
        lengths = [latent_frames(n) for n in rgb_lengths]
        rgb_edges, latent_edges = [0, *accumulate(rgb_lengths)], [0, *accumulate(lengths)]
        prompts, groups, latent_groups = memory_prefix_case_layout(target_prompts=text.prompts[count:],
            target_shot_groups=list(zip(rgb_edges, rgb_edges[1:])),
            target_latent_shot_groups=list(zip(latent_edges, latent_edges[1:])), memory_rows=text.prompts[:count],
            row_policy=(self.memory_options.row_policy if self.references_enabled else MEMORY_ROW_POLICY_FIXED_ROLE_V1))
        sample = {**target, "target_storage": self.target_storage, "height": self.height, "width": self.width,
            "num_frames": sum(rgb_lengths) + count, "prompt": list(prompts), "case_id": case, "epoch": draw.epoch,
            "target_shot_ids": list(range(draw.span[0] + 1, draw.span[1] + 1)),
            "reference_shot_ids": [shot + 1 for shot, _ in draw.references], "_sample_metadata": text.metadata,
            "_dataset_timing": {"dataset_selection_seconds": selection_seconds,
                                "target_read_seconds": target_read_seconds,
                                "target_mask_read_seconds": mask_read_seconds,
                                **reference_timing,
                                "dataset_total": time.perf_counter() - started,
                                "train_rgb_frames": sum(rgb_lengths),
                                "train_latent_frames": sum(lengths), "shot_count": len(rgb_lengths)}}
        if mask_weights is not None:
            sample["target_loss_weights"] = mask_weights
            sample["target_mask_metadata"] = mask_metadata
            sample["_sample_metadata"].update(
                target_mask_valid_frames=sum(sum(m["valid_mask_frames"]) for m in mask_metadata),
                target_mask_unknown_frames=sum(len(m["unknown_mask_frames"]) for m in mask_metadata),
                target_mask_latent_frames=sum(lengths))
        if self.planner_tokens is None:
            self._attach_layouts(sample, case, draw, item, text.compiled_targets)
        if self.references_enabled:
            sample["memory_frame_count"] = count
            sample["reference_images"] = reference_images
        sample["flow_draw_seed"] = text.metadata["flow_draw_seed"]
        if self.reference_entities is not None:
            # 参考文字丢弃时，关联的词段路由与监督同步清除；RGB保持原样。
            sample["reference_entities"] = ([] if text.metadata["reference_text_dropped"] else
                                             self.reference_entities.for_draw(
                                                 item, draw, source_frames=self.reference_source_frames(draw)))
            sample["reference_characters"] = self.annotations[case]["characters"]
            sample["reference_shot_objects"] = [shot_object_phrases(self.annotations[case], row["shot"] - 1)
                if self.shot_objects_in_rows else () for row in sample["reference_entities"]]
            sample["reference_settings"] = [reference_setting(self.annotations[case], row["shot"] - 1)
                                            for row in sample["reference_entities"]]
        if count:
            sample["reference_offset"] = list(draw.reference_offset)
        if routing is not None:
            sample["reference_routing"] = [list(entry) for entry in routing]
            # 元数据按镜号（从 1 起）：每个目标镜读到的参考源镜，及每张参考的路由依据。
            sample["_sample_metadata"].update(
                reference_routing=[[draw.references[j][0] + 1 for j in entry] for entry in routing],
                reference_routing_reasons=list(reasons))
        if shifted_slots is not None:
            # 被缩小平移的同机位参考（参考序号，从 0 起），供训练日志按机位关系拆参考收益。
            sample["_sample_metadata"]["reference_shifted"] = shifted_slots
        if self.planner_tokens is not None:
            sample["planner_tokens"] = self.planner_tokens.tokens(
                item, draw.span, lengths, training_seed=self.training_seed, key=draw.key, case_id=case)
            sample["_sample_metadata"]["planner_tokens"] = (
                None if sample["planner_tokens"] is None else
                {"shot_tokens": len(sample["planner_tokens"]["shot_tokens"])})
        if len(groups) > 1:
            sample.update(shot_groups=[list(g) for g in groups], latent_shot_groups=[list(g) for g in latent_groups])
        return sample


#: ``content`` → 必填键：点控制加特征注入要预计算特征（口径、层）；点控制（``person_index``）只要标签。
PLANNER_TOKEN_DATA_KEYS = {
    "planner_features": ("content", "labels_path", "features_root", "feature_layer", "feature_pooling", "point_jitter",
                         "point_dropout_p", "token_dropout_p"),
    "person_index": ("content", "labels_path", "point_jitter", "point_dropout_p", "token_dropout_p"),
    "person_geometry": ("content", "labels_path", "geometry_labels_path", "max_people", "point_jitter",
                        "point_dropout_p", "token_dropout_p"),
}


class PlannerTokenData:
    """带规划器 token 的样本侧（点控制、点控制加特征注入两组）：标签每样本一行（取 ``starts``：各目标镜各在场登记角色的
    镜首点，标注掩码外接框中心，镜首没有掩码记 null），点控制加特征注入另读预计算特征（每样本一个 npz，键见
    ``tools/run/planner_token_precompute.py``；只读 ``feature_pooling``、``feature_layer`` 选中的那组）。

    - token 只放本镜一处：每个「目标镜 × 镜首有点的在场角色」一个，点每轴抖动至多 ``point_jitter``（画面比例），
      另按人以 ``point_dropout_p`` 丢点（模拟规划器漏人，丢了就没有这个 token）。
    - token 整组丢弃概率 ``token_dropout_p``。标签里有、特征里没有的条目是两边对不上，报错。
    - 点控制：每个 token 的内容是此人的编号（Character k → k），不读预计算特征。
    - 参考帧与其余各组相同，随机选帧（``BlockDataset.references``），不带 token。
    """

    def __init__(self, cfg: Mapping[str, Any], sample_budget: int):
        from mspc.inference.planner_token_features import POOLINGS
        if not isinstance(cfg, Mapping) or cfg.get("content") not in PLANNER_TOKEN_DATA_KEYS:
            raise ValueError(f"dataset planner_tokens.content must be one of {sorted(PLANNER_TOKEN_DATA_KEYS)}")
        if set(cfg) != set(PLANNER_TOKEN_DATA_KEYS[cfg["content"]]):
            raise ValueError(f"dataset planner_tokens with content {cfg['content']} requires exactly "
                             f"{sorted(PLANNER_TOKEN_DATA_KEYS[cfg['content']])}")
        if cfg["content"] == "planner_features":
            if type(cfg["feature_layer"]) is not int:
                raise ValueError("planner_tokens.feature_layer must be an int")
            if cfg["feature_pooling"] not in POOLINGS:
                raise ValueError(f"planner_tokens.feature_pooling must be one of {list(POOLINGS)}")
        for name in ("point_jitter", "point_dropout_p", "token_dropout_p"):
            if type(cfg[name]) is not float or not 0 <= cfg[name] < 1:
                raise ValueError(f"planner_tokens.{name} must be a float in [0, 1)")
        self.cfg = dict(cfg)
        if cfg["content"] == "person_geometry" and (type(cfg["max_people"]) is not int or cfg["max_people"] < 1):
            raise ValueError("planner geometry requires explicit positive max_people")
        self.features_root = Path(cfg["features_root"]) if cfg["content"] == "planner_features" else None
        self.labels = {}
        for row in _jsonl(Path(cfg["labels_path"])):
            if row["sample_index"] in self.labels:
                raise ValueError("duplicate planner label sample_index")
            self.labels[row["sample_index"]] = row
        if set(self.labels) != set(range(sample_budget)):
            raise ValueError("planner token labels must cover every sample index exactly once")
        self.geometry = None
        if cfg["content"] == "person_geometry":
            from mspc.data.target_geometry import TargetGeometry
            self.geometry = TargetGeometry(Path(cfg["geometry_labels_path"]), range(sample_budget))

    def starts(self, index: int, span, *, case_id: str) -> list[tuple[int, str, list[int] | None]]:
        labels = self.labels[index]
        if labels["case_id"] != case_id:
            raise ValueError(f"sample {index}: point label case differs from the sampled case")
        if list(labels["span"]) != list(span):
            raise ValueError(f"sample {index}: label span {labels['span']} differs from the sampled span {span}")
        return [(start["shot_index"], c["name"], c["point_2d"]) for start in labels["starts"] for c in start["characters"]]

    def tokens(self, index: int, span, latent_lengths, *, training_seed: int, key, case_id: str) -> dict | None:
        from mspc.inference.planner_token_reading import person_numbers
        from mspc.models.planner_tokens import token_placement
        if unit_draw("planner_token_group", *key) < self.cfg["token_dropout_p"]:
            return None
        starts = self.starts(index, span, case_id=case_id)
        geometry = None
        if self.geometry is not None:
            from mspc.models.planner_tokens import geometry_token_features
            rows = self.geometry.for_sample(index, case_id=case_id, span=span, training_seed=training_seed,
                                             starts=starts)
            geometry = geometry_token_features(starts, rows, max_people=self.cfg["max_people"])
        if self.cfg["content"] == "planner_features":
            suffix = f"{self.cfg['feature_pooling']}_layer{self.cfg['feature_layer']}"
            with np.load(self.features_root / f"{index:06d}.npz") as data:
                features = torch.from_numpy(data[f"shot_{suffix}"].astype(np.float32))
                shot_lookup = {(int(i), str(n)): k for k, (i, n) in enumerate(zip(data["shot_index"], data["shot_name"]))}
        else:
            features = person_numbers(starts)
            shot_lookup = {(i, n): k for k, (i, n, _) in enumerate(starts)}

        def move(shot_index, name, point):
            if unit_draw("planner_point_drop", training_seed, index, shot_index, name) < self.cfg["point_dropout_p"]:
                return None
            draws = [unit_draw(f"planner_point_{axis}", training_seed, index, shot_index, name) for axis in ("x", "y")]
            return [min(1000, max(0, round(v + (2 * u - 1) * self.cfg["point_jitter"] * 1000)))
                    for v, u in zip(point, draws)]

        placed = token_placement(starts=starts, span_start=span[0], latent_starts=[0, *accumulate(latent_lengths)],
                                 shot_lookup=shot_lookup, move_point=move)
        if placed is None:
            return None
        return {"features": features, **placed, **({} if geometry is None else {"geometry": geometry})}


def build_training_dataset(args: Any) -> BlockDataset:
    plan = getattr(args, "cache_stage_plan", None)
    return BlockDataset(**args.dataset_cfg, required_sample_indices=None if plan is None else plan["sample_indices"])


def build_validation_dataset(args: Any) -> BlockDataset:
    overrides = {"members_path": args.validation["members_path"], "sample_budget": args.validation["sample_budget"],
                 "sampling_manifest_path": args.validation["sampling_manifest_path"]}
    overrides["text_admission_path"] = args.validation["text_admission_path"]
    if args.dataset_cfg.get("planner_tokens") is not None:
        # 验证集有自己的读历史标签与预计算特征；其余 token 设置与训练相同。
        overrides["planner_tokens"] = {**args.dataset_cfg["planner_tokens"], **args.validation["planner_tokens"]}
    if args.dataset_cfg.get("reference_entities") is not None:
        overrides["reference_entities"] = args.validation["reference_entities"]
    # 裁剪只是训练增强，负载上限只管训练组批：验证量损失时都不用。
    overrides.update(reference_crops=None, workload_cap=None)
    plan = getattr(args, "cache_stage_plan", None)
    return BlockDataset(**{**args.dataset_cfg, **overrides},
                        required_sample_indices=None if plan is None else plan["validation_sample_indices"])
