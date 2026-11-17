"""真实 SAM RLE → 目标 latent 权重缓存与严格 reader；全程 CPU，不运行分割模型。"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from mspc.data.scene_training_manifest import ANNOTATION_MEDIA, require_training_media, training_geometry
from mspc.data.atomic_publish import publish_directory
from mspc.data.shot_targets import latent_frames, planned_row, shot_key


MASK_WEIGHT_SCHEMA = "sam_target_loss_weights_v1"
TEMPORAL_ALIGNMENT = "causal_chunk_endpoint"
#: SAM 只在旧标注网格（480×832/16fps 的 crop）上有 RLE；其他目标媒体经源坐标映射过来。
SPATIAL_SHAPE = (480, 832)
MASK_TIME_ALIGNMENT = "nearest_sam_source_frame"
PIXEL_MAPPING = "target_to_annotation_affine_nearest"
#: 按所选VAE的空间步幅池化真实主体并集；不缩放另一种VAE的权重图。
WEIGHTINGS = frozenset({"area_coverage", "touched_cell"})


def latent_spatial_shape(pixel_shape: Sequence[int], vae_spatial_stride: int) -> tuple[int, int]:
    if (type(vae_spatial_stride) is not int or vae_spatial_stride <= 0
            or any(size % vae_spatial_stride for size in pixel_shape)):
        raise ValueError("vae_spatial_stride must exactly divide the training pixel grid")
    return tuple(size // vae_spatial_stride for size in pixel_shape)


def _spatial_alignment(weighting: str, stride: int) -> str:
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {sorted(WEIGHTINGS)}")
    operation = "area_pool" if weighting == "area_coverage" else "any_pixel"
    return f"union_then_{operation}_{stride}x{stride}"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def _source_contract(source: Mapping[str, Any], position: int) -> dict:
    scene, shot = source["scene"], source["scene"]["shots"][position]
    planned = planned_row(scene, position, media=ANNOTATION_MEDIA)
    grid = planned["source_frame_indices_by_shot"][0]
    if (planned["train_crop_box"] != scene["crop_box"]
            or planned["shot_groups"] != [[0, shot["render_frames"]]]):
        raise ValueError("SAM source crop/frame count differs from target frame producer")
    endpoints = list(range(0, shot["render_frames"], 4))
    if len(endpoints) != latent_frames(shot["render_frames"]):
        raise ValueError("SAM endpoint count differs from target latent frames")
    return {
        "block_key": source["key"], "case_id": source["case_id"], "shot": position + 1,
        **{name: shot[name] for name in ("sid", "scene_id", "source_shot_index")},
        "train_frames": shot["render_frames"], "source_video_path": scene["source_video_path"],
        "crop_box": scene["crop_box"], "mask_training_frames": endpoints,
        "mask_source_frames": [grid[i] for i in endpoints],
    }


def _target_grid(source: Mapping[str, Any], position: int, media: Mapping[str, Any]) -> dict:
    """实际训练媒体的目标潜帧端点、crop，以及每个目标像素在旧 SAM 网格上的落点。

    空间：目标像素中心经「新目标→源→旧480」仿射（两端都是 ``training_geometry``）落到旧网格，取所在像素，
    即最近邻上采样；落在旧画幅外的像素为背景。时间：每个目标潜帧端点取源帧号最近的 SAM 端点（等距取较早者），
    即 11-08 决定的「插帧对齐」，不混合两帧 mask。480×832/16fps 媒体两者都是恒等。
    """
    media = require_training_media(media)
    scene = source["scene"]
    planned = planned_row(scene, position, media=media)
    count = planned["shot_groups"][0][1]
    if planned["shot_groups"] != [[0, count]]:
        raise ValueError("target frame producer does not give one complete shot")
    endpoints = list(range(0, count, 4))
    if len(endpoints) != latent_frames(count):
        raise ValueError("target endpoint count differs from target latent frames")
    grid = planned["source_frame_indices_by_shot"][0]
    old, new = (training_geometry(scene, media=value) for value in (ANNOTATION_MEDIA, media))
    if new["train_crop_box"] != planned["train_crop_box"]:
        raise ValueError("target crop differs between frame producer and geometry owner")
    matrix = np.asarray(old["source_to_target_matrix"], dtype=np.float64) @ np.linalg.inv(
        np.asarray(new["source_to_target_matrix"], dtype=np.float64))
    width, height = new["output_size"]
    return {"media": media, "crop_box": list(planned["train_crop_box"]), "pixel_shape": (height, width),
            "training_frames": endpoints, "source_frames": [grid[i] for i in endpoints],
            "rows": np.floor(matrix[1, 1] * (np.arange(height) + .5) + matrix[1, 2]).astype(np.int64),
            "columns": np.floor(matrix[0, 0] * (np.arange(width) + .5) + matrix[0, 2]).astype(np.int64)}


def compile_target_mask_weights(
    *, source: Mapping[str, Any], frames: Mapping[str, Any], masks: Mapping[str, Any],
    shot_index: int, entity_names: Sequence[str], weighting: str, vae_spatial_stride: int,
    media: Mapping[str, Any],
) -> tuple[torch.Tensor, dict]:
    """一镜的真实 mask 并集 → 实际训练媒体、所选VAE latent网格的权重及有效位。

    Wan 因果编码先处理 RGB0，随后每次接入4帧并保持历史缓存。现有 SAM 只标旧 16fps 网格的
    RGB0、4、8…，因此每个 latent 用本次新输入末帧（按源帧最近的 SAM 端点）的观察，不混合两帧、不声称完整
    receptive field 的主体支撑。无正 mask 的时点显式记录为未知，数值权重为1；
    整镜没有任何正 mask 时记为未知，整镜权重为 1，导出不中断。未知时点的均匀损失不是背景标注。
    其他画幅的 mask 按 ``_target_grid`` 从旧 crop 映射；``observed_entities_by_frame`` 是所用 SAM 端点的可见名单。
    """
    from pycocotools import mask as coco

    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {sorted(WEIGHTINGS)}")
    target = _target_grid(source, shot_index, media)
    spatial_shape = latent_spatial_shape(target["pixel_shape"], vae_spatial_stride)
    alignment = _spatial_alignment(weighting, vae_spatial_stride)
    contract = _source_contract(source, shot_index)
    sam_sources = contract["mask_source_frames"]
    picks = [min(range(len(sam_sources)), key=lambda j: (abs(sam_sources[j] - frame), j))
             for frame in target["source_frames"]]
    target_contract = {
        "target_media": target["media"], "target_crop_box": target["crop_box"],
        "target_training_frames": target["training_frames"], "target_source_frames": target["source_frames"],
        "mask_frame_by_target": [contract["mask_training_frames"][j] for j in picks],
        "mask_time_alignment": MASK_TIME_ALIGNMENT, "pixel_mapping": PIXEL_MAPPING,
    }
    latent_count = len(target["training_frames"])
    sid = f"s{shot_index + 1:02d}"
    if (frames["key"] != source["key"] or frames["train_fps"] != 16 or frames["frame_stride"] != 4
            or len(frames["shots"]) != len(source["scene"]["shots"])):
        raise ValueError("SAM frame producer identity/fps/stride/shot count differs")
    if (masks["schema_version"] != 2 or (masks["height"], masks["width"]) != SPATIAL_SHAPE
            or masks["frame_numbers"] != "training frame within the shot"
            or masks["rle_format"] != "COCO compressed, Fortran order"):
        raise ValueError("SAM mask schema or pixel grid differs")
    fs, ms = frames["shots"][shot_index], masks["shots"][sid]
    if (fs["id"] != sid or fs["shot"] != shot_index + 1
            or fs["render_frames"] != contract["train_frames"]
            or fs["train_frames"] != contract["mask_training_frames"]
            or fs["source_frames"] != contract["mask_source_frames"]
            or fs["frames"] != len(fs["train_frames"])
            or ms["train_frames"] != fs["train_frames"] or ms["source_frames"] != fs["source_frames"]):
        raise ValueError(f"{source['key']}/{sid}: SAM crop/time grid differs from actual target source")
    if fs["annotation_errors"] or ms["annotation_errors"]:
        raise ValueError(f"{source['key']}/{sid}: SAM shot has annotation errors")
    if (not isinstance(entity_names, (list, tuple))
            or any(not isinstance(n, str) or not n for n in entity_names)
            or len(set(entity_names)) != len(entity_names)):
        raise ValueError("entity_names must be unique registered names")
    chunks = [[0, 1], *[[1 + 4 * i, 5 + 4 * i] for i in range(latent_count - 1)]]
    if not entity_names:
        weights = torch.ones((1, 1, latent_count, *spatial_shape), dtype=torch.float32)
        valid = [False] * latent_count
        metadata = {
            "schema": MASK_WEIGHT_SCHEMA, **contract, **target_contract,
            "temporal_alignment": TEMPORAL_ALIGNMENT,
            "vae_input_chunks": chunks,
            "pixel_shape": list(target["pixel_shape"]), "latent_shape": [1, 1, latent_count, *spatial_shape],
            "selected_entities": [],
            "observed_entities_by_frame": [[] for _ in range(latent_count)],
            "valid_mask_frames": valid,
            "unknown_mask_frames": list(target["training_frames"]),
            "mask_record_count": 0,
            "spatial_alignment": alignment, "subject_weight": 2, "background_weight": 1,
            "mask_status": "unknown_uniform",
        }
        return weights, metadata
    selected = set(entity_names)
    endpoints = contract["mask_training_frames"]
    planes = np.zeros((len(endpoints), *SPATIAL_SHAPE), dtype=bool)
    observed = {frame: set() for frame in endpoints}
    tracks = {tid: track for tid, track in ms["tracks"].items() if track["name"] in selected}
    for track in tracks.values():
        for frame_text, record in track["frames"].items():
            frame = int(frame_text)
            if str(frame) != frame_text or frame not in observed:
                raise ValueError(f"{source['key']}/{sid}: mask frame is outside SAM endpoint grid")
            rle = record["rle"]
            if rle["size"] != list(SPATIAL_SHAPE) or not isinstance(rle["counts"], str):
                raise ValueError("SAM requires compressed COCO RLE on the 480x832 pixel grid")
            decoded = coco.decode(rle).astype(bool)
            if (decoded.shape != SPATIAL_SHAPE or int(decoded.sum()) != record["area"]
                    or not np.allclose(coco.toBbox(rle), record["bbox_xywh"])):
                raise ValueError(f"{source['key']}/{sid}: SAM RLE area/box/shape differs")
            if decoded.any():
                planes[frame // 4] |= decoded
                observed[frame].add(track["name"])
    # 先取主体并集、映射到目标像素，再面积池化。重复轨迹／重叠主体不会累加成 3×、4×。
    rows, columns = target["rows"], target["columns"]
    inside = (((rows >= 0) & (rows < SPATIAL_SHAPE[0]))[:, None]
              & ((columns >= 0) & (columns < SPATIAL_SHAPE[1]))[None, :])
    lookup = np.ix_(rows.clip(0, SPATIAL_SHAPE[0] - 1), columns.clip(0, SPATIAL_SHAPE[1] - 1))
    coverage = {}
    for j in dict.fromkeys(picks):
        plane = torch.from_numpy(planes[j][lookup] & inside).float()[None, None]
        coverage[j] = F.avg_pool2d(plane, kernel_size=vae_spatial_stride, stride=vae_spatial_stride)[0, 0]
    valid = [bool(coverage[j].any()) for j in picks]
    if any(valid):
        pooled = torch.stack([coverage[j] for j in picks])
        if weighting == "touched_cell":
            pooled = (pooled > 0).float()
        weights = (1 + pooled)[None, None].contiguous()
        mask_status = "observed"
    else:
        weights = torch.ones((1, 1, latent_count, *spatial_shape), dtype=torch.float32)
        mask_status = "unknown_uniform"
    metadata = {
        "schema": MASK_WEIGHT_SCHEMA, **contract, **target_contract,
        "temporal_alignment": TEMPORAL_ALIGNMENT,
        "vae_input_chunks": chunks,
        "pixel_shape": list(target["pixel_shape"]), "latent_shape": [1, 1, latent_count, *spatial_shape],
        "selected_entities": list(entity_names),
        "observed_entities_by_frame": [sorted(observed[contract["mask_training_frames"][j]]) for j in picks],
        "valid_mask_frames": valid,
        "unknown_mask_frames": [frame for frame, available in zip(target["training_frames"], valid) if not available],
        "mask_record_count": sum(len(track["frames"]) for track in tracks.values()),
        "spatial_alignment": alignment, "subject_weight": 2, "background_weight": 1,
        "mask_status": mask_status,
    }
    return weights, metadata


def _match_member_source(member: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    if (member["key"] != source["key"] or member["case_id"] != source["case_id"]
            or len(member["shots"]) != len(source["scene"]["shots"])):
        raise ValueError("SAM source block identity or shot count differs from member")
    if any(member["scene"][name] != source["scene"][name]
           for name in ("source_video_path", "crop_box")):
        raise ValueError("SAM source media/crop differs from training member")
    for member_shot, source_shot in zip(member["shots"], source["scene"]["shots"], strict=True):
        if (shot_key(member_shot) != shot_key(source_shot)
                or member_shot["frames"] != source_shot["render_frames"]):
            raise ValueError("SAM source shot identity/frame count differs from member")


def compile_member_target_mask_weights(*, member: Mapping[str, Any], positions: Sequence[int],
                                       spatial_root: str, weighting: str, vae_spatial_stride: int,
                                       media: Mapping[str, Any]) -> list[tuple[torch.Tensor, dict]]:
    """训练 worker 从原 SAM RLE 按实际训练媒体编译本次目标镜；与离线生产使用同一函数，不写权重缓存。"""
    directory = Path(spatial_root) / member["key"]
    source, frames, masks = (_read(directory / name) for name in ("source.json", "frames.json", "masks.json"))
    _match_member_source(member, source)
    annotation = member["annotation"]
    registered = {entity["name"] for entity in annotation["characters"] + annotation["objects"]}
    pairs = []
    for position in positions:
        if type(position) is not int or not 0 <= position < len(member["shots"]):
            raise ValueError("online SAM position must name a member shot")
        names = list(dict.fromkeys(entry["name"] for entry in annotation["shots"][position]["present"]))
        if set(names) - registered:
            raise ValueError("target mask entity is not in the member registry")
        weights, record = compile_target_mask_weights(
            source=source, frames=frames, masks=masks, shot_index=position, entity_names=names,
            weighting=weighting, vae_spatial_stride=vae_spatial_stride, media=media)
        expected = _source_contract({"key": member["key"], "case_id": member["case_id"],
                                    "scene": {**member["scene"], "shots": member["shots"]}}, position)
        if any(record[name] != expected[name] for name in expected):
            raise ValueError("online SAM source time/grid differs from training member")
        pairs.append((weights, record))
    return pairs


def build_target_mask_weights(*, members_path: Path, spatial_root: Path, output_dir: Path, weighting: str,
                              vae_spatial_stride: int,
                              sampling_manifest_paths: Sequence[Path] | None = None) -> dict:
    """现行成员与现有空间 producer 的唯一连接；缺镜／缺文件明确失败。

    离线权重文件只在旧标注网格（480×832/16fps）上产出，``TargetMaskWeights`` 按此读；其他媒体用原 RLE 在线编译。
    """
    spatial_shape = latent_spatial_shape(SPATIAL_SHAPE, vae_spatial_stride)
    alignment = _spatial_alignment(weighting, vae_spatial_stride)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    staging = output_dir.with_name(output_dir.name + ".partial")
    staging.mkdir(parents=True, exist_ok=False)
    wanted = None
    if sampling_manifest_paths:
        wanted = set()
        for path in sampling_manifest_paths:
            for row in _rows(Path(path)):
                start, end = row["span"]
                for shot in range(start, end):
                    wanted.add((row["case_id"], shot))
    wanted_cases = None if wanted is None else {case for case, _ in wanted}
    records, counts, seen = [], Counter(), set()
    for member in _rows(members_path):
        if member["case_id"] in seen:
            raise ValueError(f"duplicate member {member['case_id']}")
        seen.add(member["case_id"])
        if wanted_cases is not None and member["case_id"] not in wanted_cases:
            continue
        directory = spatial_root / member["key"]
        source, frames, masks = (_read(directory / name) for name in ("source.json", "frames.json", "masks.json"))
        _match_member_source(member, source)
        annotation = member["annotation"]
        registered = {entity["name"] for entity in annotation["characters"] + annotation["objects"]}
        counts["blocks"] += 1
        for position, shot in enumerate(member["shots"]):
            if not shot["usable"]:
                continue
            if wanted is not None and (member["case_id"], position) not in wanted:
                continue
            names = list(dict.fromkeys(entry["name"] for entry in annotation["shots"][position]["present"]))
            if set(names) - registered:
                raise ValueError("target mask entity is not in the member registry")
            weights, record = compile_target_mask_weights(
                source=source, frames=frames, masks=masks, shot_index=position, entity_names=names,
                weighting=weighting, vae_spatial_stride=vae_spatial_stride, media=ANNOTATION_MEDIA)
            relative = Path("weights") / member["key"] / f"s{position + 1:02d}.safetensors"
            record.update(file=str(relative), masks_path=str(directory / "masks.json"),
                          frames_path=str(directory / "frames.json"), source_path=str(directory / "source.json"))
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            save_file({"target_loss_weights": weights,
                       "valid_mask_frames": torch.tensor(record["valid_mask_frames"], dtype=torch.bool)},
                      str(destination), metadata={"target_mask_record": json.dumps(record, ensure_ascii=False)})
            records.append(record)
            counts.update(shots=1, latent_frames=len(record["valid_mask_frames"]),
                          valid_mask_frames=sum(record["valid_mask_frames"]),
                          unknown_mask_frames=len(record["unknown_mask_frames"]),
                          unknown_uniform_shots=int(record["mask_status"] == "unknown_uniform"))
    if wanted is not None:
        produced = {(record["case_id"], record["shot"] - 1) for record in records}
        missing = wanted - produced
        if missing:
            raise ValueError(f"{len(missing)} sampled shots have no SAM weight record")
    if not records:
        raise ValueError("no target mask weights produced")
    (staging / "index.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    summary = {"schema": MASK_WEIGHT_SCHEMA, "spatial_alignment": alignment,
               "vae_spatial_stride": vae_spatial_stride, "latent_spatial_shape": list(spatial_shape),
               "members_path": str(members_path),
               "spatial_root": str(spatial_root), **counts}
    (staging / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    publish_directory(staging, output_dir)
    return summary


class TargetMaskWeights:
    """训练 worker 按实际目标latent网格读权重，不引入pycocotools或模型。"""

    def __init__(self, *, index_path: str, root: str, weighting: str):
        self.root = Path(root)
        self.records = {}
        for record in _rows(Path(index_path)):
            key = (record["case_id"], record["shot"])
            if key in self.records or record["schema"] != MASK_WEIGHT_SCHEMA:
                raise ValueError(f"duplicate or invalid target mask record {key}")
            self.records[key] = record
        if not self.records:
            raise ValueError("target mask weight index is empty")
        if weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {sorted(WEIGHTINGS)}")
        shapes = {tuple(record["latent_shape"][-2:]) for record in self.records.values()}
        if len(shapes) != 1:
            raise ValueError("target mask index mixes latent spatial grids")
        self.latent_spatial_shape = next(iter(shapes))
        height, width = self.latent_spatial_shape
        if (type(height) is not int or type(width) is not int or min(height, width) <= 0
                or SPATIAL_SHAPE[0] % height or SPATIAL_SHAPE[1] % width
                or SPATIAL_SHAPE[0] // height != SPATIAL_SHAPE[1] // width):
            raise ValueError("target mask latent shape is not an exact VAE pixel grid")
        stride = SPATIAL_SHAPE[0] // height
        alignment = _spatial_alignment(weighting, stride)
        alignments = {record["spatial_alignment"] for record in self.records.values()}
        if alignments != {alignment}:
            raise ValueError(f"target mask weight index holds {sorted(alignments)}, not the declared {weighting}")
        self.weighting = weighting

    def record_for(self, member: Mapping[str, Any], position: int, *,
                   latent_spatial_shape: Sequence[int]) -> dict:
        key = (member["case_id"], position + 1)
        if key not in self.records:
            raise ValueError(f"{key}: missing real SAM target weights")
        record, shot = self.records[key], member["shots"][position]
        if (record["block_key"] != member["key"] or shot_key(record) != shot_key(shot)
                or record["train_frames"] != shot["frames"]
                or record["crop_box"] != member["scene"]["crop_box"]
                or record["source_video_path"] != member["scene"]["source_video_path"]):
            raise ValueError(f"{key}: target mask source differs from member")
        expected = _source_contract({"key": member["key"], "case_id": member["case_id"],
            "scene": {**member["scene"], "shots": member["shots"]}}, position)
        if any(record[name] != expected[name] for name in expected):
            raise ValueError(f"{key}: target mask time/source grid differs")
        if record["temporal_alignment"] != TEMPORAL_ALIGNMENT:
            raise ValueError(f"{key}: unsupported target mask time alignment")
        if record["latent_shape"] != [1, 1, latent_frames(shot["frames"]), *latent_spatial_shape]:
            raise ValueError(f"{key}: target weights differ from the selected target latent grid")
        return record

    def read(self, member: Mapping[str, Any], position: int, *,
             latent_spatial_shape: Sequence[int]) -> tuple[torch.Tensor, dict]:
        record = self.record_for(member, position, latent_spatial_shape=latent_spatial_shape)
        path = self.root / record["file"]
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"target_loss_weights", "valid_mask_frames"}:
                raise ValueError(f"{path}: invalid target mask tensor keys")
            if json.loads((handle.metadata() or {})["target_mask_record"]) != record:
                raise ValueError(f"{path}: target mask record differs from index")
            weights = handle.get_tensor("target_loss_weights")
            valid = handle.get_tensor("valid_mask_frames")
        observed = record.get("mask_status", "observed") == "observed"
        uniform_unknown = (not valid.any() and torch.equal(weights, torch.ones_like(weights))
                           and record.get("mask_status") == "unknown_uniform")
        if (weights.dtype != torch.float32 or list(weights.shape) != record["latent_shape"]
                or valid.dtype != torch.bool or valid.tolist() != record["valid_mask_frames"]
                or not torch.isfinite(weights).all()
                or not torch.all((weights >= 1) & (weights <= 2))
                or (observed and not torch.equal((weights > 1).any(dim=(0, 1, 3, 4)), valid))
                or not (observed or uniform_unknown)):
            raise ValueError(f"{path}: invalid target SAM weight/validity tensor")
        return weights, record
