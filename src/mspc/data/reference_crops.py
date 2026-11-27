"""训练参考的放大裁剪增强（只训练，推理不用）。

放大裁剪，裁出面积占原图 0.65–1.0、保持宽高比、缩回原网格；不翻转、不改色相、
不加亮度抖动。原定每张参考 50% 概率；包住全部人物的约束下实际只裁到 23.6% 的参考，所以改为每张参考
都尝试裁剪，实际约 47%（其余是人物框太大包不住的）。参考与目标同一
空间网格，裁剪打乱「同位置抄同内容」。

裁剪框必须包住这一参考帧里**全部登记人物**（块标注 ``characters``）的 SAM 框；帧里没有登记人物时只守面积范围；
物体与地点框不约束。「必要性重审」（同页）定：包住全部人物而不是被路由的人物，裁剪就与读者范围无关，一份表各组
共用，身份行列出的人也永远都还在画面里。人物框太大、面积放到整图也包不住时不裁（摘要里记为
``characters_too_large``）；这一帧没有可核的 SAM 帧或 SAM 标注有错时也不裁。

框由 CPU 工具 ``tools/data/build_reference_crops.py`` 按训练抽签离线写成表（与参考身份标签同一抽签），训练只读表、
在解码后的参考 RGB 上裁并缩回。坐标都是目标画幅上的归一化 [x0, y0, x1, y1]。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import AbstractSet, Any, Mapping, Sequence

import torch

CROP_AREA = (0.65, 1.0)
#: v3：每张参考都尝试裁剪（v2 是 50% 概率），旧表不再可读。
REFERENCE_CROP_SCHEMA = "reference_crops_v3"
#: 裁剪表摘要的计数项（没有出现的记 0）。
CROP_COUNTS = ("references", "cropped", "not_cropped", "characters_too_large",
               "references_without_characters", "reference_without_reliable_sam_frame", "kept_character_boxes")


def draw_crop(keep: Sequence[float] | None, *, area: float, x: float, y: float) -> list[float] | None:
    """一张参考的裁剪框；``keep`` 是必须包住的归一化框（全部登记人物框的并集，没有就 None），其余三个是 [0,1) 抽签。

    面积在 0.65–1.0 均匀取；包不住 ``keep`` 时把面积放大到刚好包住，超过整图就不裁。
    归一化坐标下宽高按同一比例缩，所以裁出的框与原图同宽高比。"""
    scale = math.sqrt(CROP_AREA[0] + area * (CROP_AREA[1] - CROP_AREA[0]))
    if keep is not None:
        x0, y0, x1, y1 = keep
        scale = max(scale, x1 - x0, y1 - y0)
    if scale >= 1.0:
        return None
    lo_x, hi_x = (0.0, 1.0 - scale) if keep is None else (max(0.0, keep[2] - scale), min(keep[0], 1.0 - scale))
    lo_y, hi_y = (0.0, 1.0 - scale) if keep is None else (max(0.0, keep[3] - scale), min(keep[1], 1.0 - scale))
    left, top = lo_x + x * (hi_x - lo_x), lo_y + y * (hi_y - lo_y)
    return [left, top, left + scale, top + scale]


def union_box(boxes: Sequence[Sequence[float]]) -> list[float] | None:
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def apply_crops(images: torch.Tensor, boxes: Sequence[Sequence[float] | None]) -> torch.Tensor:
    """uint8 [N,3,H,W] 参考按框裁出再双线性（抗混叠）缩回 H×W；框为 None 的原样返回。"""
    if images.dtype != torch.uint8 or images.ndim != 4 or len(boxes) != images.shape[0]:
        raise ValueError("reference crops need uint8 [N,3,H,W] images and one box (or None) per reference")
    height, width = images.shape[2:]
    out = images.clone()
    for i, box in enumerate(boxes):
        if box is None:
            continue
        x0, y0 = round(box[0] * width), round(box[1] * height)
        x1, y1 = max(x0 + 1, round(box[2] * width)), max(y0 + 1, round(box[3] * height))
        patch = images[i:i + 1, :, y0:y1, x0:x1].float()
        resized = torch.nn.functional.interpolate(patch, size=(height, width), mode="bilinear",
                                                  align_corners=False, antialias=True)
        out[i] = resized.round().clamp(0, 255).to(torch.uint8)[0]
    return out


def same_camera_slots(shots: Sequence[Mapping[str, Any]], *, reference_shots: Sequence[int],
                      target_shots: Sequence[int], routing: Sequence[Sequence[int]] | None = None) -> list[bool]:
    """每张参考是否与它的某个读者同机位（任一方机位缺失也算）；读者按 ``routing``，没有路由时是全部目标镜。"""
    from mspc.data.reference_routing import camera_setup

    targets = list(target_shots)
    if routing is not None and len(routing) != len(targets):
        raise ValueError("routing must match target_shots length")
    flags = []
    for slot, source_shot in enumerate(reference_shots):
        readers = (targets if routing is None else
                   [targets[t_idx] for t_idx, slots in enumerate(routing) if slot in slots])
        setup = camera_setup(shots[source_shot])
        flags.append(setup is None or any(camera_setup(shots[t]) in (None, setup) for t in readers))
    return flags


def shift_same_camera(images: torch.Tensor, flags: Sequence[bool], draws: Sequence[Sequence[float]], *,
                      scale: Sequence[float], edge: float) -> torch.Tensor:
    """同机位参考的打乱构图：整帧缩到 ``scale`` 区间内的比例，贴到黑底画布上靠左或靠右的一侧（离边不超过空余宽度的
    ``edge``），上下位置任意；画面里所有内容都离开原位置，包括包不住、裁不动的大特写（同位置抄的捷径是逐格对齐）。

    ``draws`` 每张参考四个 [0,1) 抽签：比例、横向位置、纵向位置、靠哪一侧。``flags`` 为假的参考原样返回。缩小用抗混叠
    双线性；不翻转（翻转后居中的脸仍在原位置，且会改变脸的不对称细节）。"""
    if images.dtype != torch.uint8 or images.ndim != 4 or len(flags) != images.shape[0] or len(draws) != len(flags):
        raise ValueError("same-camera shift needs uint8 [N,3,H,W] images and one flag and draw per reference")
    low, high = scale
    if not 0 < low <= high < 1 or not 0 <= edge <= 0.5:
        raise ValueError("same-camera shift scale must lie in (0, 1) and edge in [0, 0.5]")
    height, width = images.shape[2:]
    out = images.clone()
    for i, (flag, (u_scale, u_x, u_y, u_side)) in enumerate(zip(flags, draws, strict=True)):
        if not flag:
            continue
        factor = low + (high - low) * u_scale
        h, w = max(1, round(height * factor)), max(1, round(width * factor))
        small = torch.nn.functional.interpolate(images[i:i + 1].float(), size=(h, w), mode="bilinear",
                                                align_corners=False, antialias=True)
        free_x, free_y = width - w, height - h
        offset = round(u_x * edge * free_x)
        left = offset if u_side < 0.5 else free_x - offset
        top = round(u_y * free_y)
        canvas = torch.zeros_like(images[i])
        canvas[:, top:top + h, left:left + w] = small.round().clamp(0, 255).to(torch.uint8)[0]
        out[i] = canvas
    return out


def select_same_camera_boxes(
    boxes: Sequence[Sequence[float] | None],
    *,
    shots: Sequence[Mapping[str, Any]],
    reference_shots: Sequence[int],
    target_shots: Sequence[int],
    routing: Sequence[Sequence[int]] | None = None,
) -> list[Sequence[float] | None]:
    """条件同机位裁剪：只对与实际读者中任一目标镜同机位（或任一方机位缺失）的参考槽位保留裁剪框，
    其余纯异机位参考返回 ``None``（保持未裁剪原图）。"""
    from mspc.data.reference_routing import camera_setup

    targets = list(target_shots)
    if len(boxes) != len(reference_shots):
        raise ValueError("boxes and reference_shots must have the same length")
    if routing is not None and len(routing) != len(targets):
        raise ValueError("routing must match target_shots length")
    selected: list[Sequence[float] | None] = []
    for slot, source_shot in enumerate(reference_shots):
        box = boxes[slot]
        if box is None:
            selected.append(None)
            continue
        readers = (targets if routing is None else
                   [targets[t_idx] for t_idx, slots in enumerate(routing) if slot in slots])
        setup = camera_setup(shots[source_shot])
        has_same_camera = (setup is None or any(camera_setup(shots[t]) in (None, setup) for t in readers))
        selected.append(box if has_same_camera else None)
    return selected


class ReferenceCrops:
    """离线裁剪表的读者：逐样本、逐参考核对源镜与帧号和实际抽签一致。"""

    def __init__(self, config: Mapping[str, Any], expected_sample_indices: Sequence[int]):
        if (not isinstance(config, Mapping) or "path" not in config
                or not set(config) <= {"path", "only_same_camera", "same_camera_shift"}):
            raise ValueError("reference_crops requires path and optional only_same_camera / same_camera_shift")
        only_same_camera = config.get("only_same_camera", False)
        if type(only_same_camera) is not bool:
            raise ValueError("reference_crops.only_same_camera must be a bool")
        shift = config.get("same_camera_shift")
        if shift is not None:
            if (not isinstance(shift, Mapping) or set(shift) != {"scale", "edge"} or len(shift["scale"]) != 2
                    or not 0 < shift["scale"][0] <= shift["scale"][1] < 1 or not 0 <= shift["edge"] <= 0.5):
                raise ValueError("reference_crops.same_camera_shift needs scale [low, high] in (0, 1) and edge in [0, 0.5]")
            if only_same_camera:
                raise ValueError("same_camera_shift replaces the crop on same-camera references; "
                                 "only_same_camera would drop the crops of the other references")
        self.only_same_camera = only_same_camera
        self.same_camera_shift = None if shift is None else {"scale": tuple(shift["scale"]), "edge": shift["edge"]}
        self.records = {}
        with Path(config["path"]).open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if (set(record) != {"schema", "sample_index", "reference_draw", "boxes"}
                        or record["schema"] != REFERENCE_CROP_SCHEMA or record["sample_index"] in self.records
                        or len(record["boxes"]) != len(record["reference_draw"])):
                    raise ValueError("invalid or duplicate reference crop row")
                for box in record["boxes"]:
                    if box is not None and (len(box) != 4 or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
                        raise ValueError("reference crop box must be normalized [x0, y0, x1, y1]")
                self.records[record["sample_index"]] = record
        missing = set(expected_sample_indices) - set(self.records)
        if missing:
            raise ValueError(f"reference crops miss {len(missing)} sample(s), first {min(missing)}")

    def for_draw(self, index: int, draw) -> list:
        record = self.records[index]
        if record["reference_draw"] != [[shot + 1, frame] for shot, frame in draw.references]:
            raise ValueError("reference crop row differs from the actual reference draw")
        return record["boxes"]


def entity_box(mask, old_geometry: Mapping[str, Any], geometry: Mapping[str, Any]) -> list[float] | None:
    """一个 SAM 掩码（480×832 标注画幅）→ 目标画幅上的归一化外接框；裁出画幅外就 None。"""
    import numpy as np

    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    transform = np.asarray(geometry["source_to_target_matrix"], dtype=np.float64) @ np.linalg.inv(
        np.asarray(old_geometry["source_to_target_matrix"], dtype=np.float64))
    corners = np.array([[xs.min(), ys.min(), 1.0], [xs.max() + 1.0, ys.max() + 1.0, 1.0]]).T
    (x0, x1), (y0, y1) = (transform @ corners)[:2]
    width, height = geometry["output_size"]
    x0, x1 = max(0.0, x0 / width), min(1.0, x1 / width)
    y0, y1 = max(0.0, y0 / height), min(1.0, y1 / height)
    return [x0, y0, x1, y1] if x0 < x1 and y0 < y1 else None


def character_keep_box(reader, shot: int, train_frame: int, characters: AbstractSet[str],
                       old_geometry: Mapping[str, Any], geometry: Mapping[str, Any]) -> tuple[list[float] | None, int]:
    """参考帧里全部登记人物 SAM 框的并集（目标画幅归一化）与框数；帧里没有登记人物就 ``(None, 0)``。

    ``reader`` 是该块的 :class:`mspc.data.history_reading_labels.ShotMasks`，``train_frame`` 是 SAM 的训练帧号；
    物体与地点的轨迹不看。"""
    boxes = []
    for name in sorted(set(reader.names(shot)) & set(characters)):
        mask = reader.mask(shot, name, train_frame)
        box = None if mask is None or not mask.any() else entity_box(mask, old_geometry, geometry)
        if box is not None:
            boxes.append(box)
    return union_box(boxes), len(boxes)


_CROP_STATE = None


def _init_crop_worker(dataset, spatial_root):
    global _CROP_STATE
    _CROP_STATE = (dataset, Path(spatial_root))


def _crop_sample(index: int) -> tuple[dict, dict]:
    from collections import Counter

    from mspc.core.draws import unit_draw
    from mspc.data.history_reading_labels import ShotMasks
    from mspc.data.reference_entities import _read_reference_block, matched_mask_frame
    from mspc.data.scene_training_manifest import ANNOTATION_MEDIA, training_geometry

    dataset, spatial_root = _CROP_STATE
    draw = dataset.draw(index)
    member = dataset.members[draw.row.case_id]
    characters = {entry["name"] for entry in member["annotation"]["characters"]}
    counts, boxes = Counter(), []
    if draw.references:
        source, frames, masks = _read_reference_block(spatial_root, member["key"])
        reader = ShotMasks(masks)
        old_geometry = training_geometry(member["scene"], media=ANNOTATION_MEDIA)
        geometry = training_geometry(member["scene"], media=dataset.media)
        source_frames = dataset.reference_source_frames(draw)
        for slot, (shot, _) in enumerate(draw.references):
            key = (dataset.training_seed, index, slot)
            # 与身份标签同一核对：SAM 来源合同、镜号与训练帧格都要对上，不借邻帧。
            _, errors = matched_mask_frame(member=member, shot_index=shot, source_frame=source_frames[slot],
                                           source=source, frames=frames, masks=masks)
            sampled = frames["shots"][shot]
            matching = [f for f, s in zip(sampled["train_frames"], sampled["source_frames"], strict=True)
                        if s == source_frames[slot]]
            counts["references"] += 1
            if len(matching) != 1 or errors:
                counts["reference_without_reliable_sam_frame"] += 1
                box = None
            else:
                keep, kept = character_keep_box(reader, shot, matching[0], characters, old_geometry, geometry)
                counts["kept_character_boxes"] += kept
                counts["references_without_characters"] += keep is None
                box = draw_crop(keep, area=unit_draw("reference_crop_area", *key),
                                x=unit_draw("reference_crop_x", *key), y=unit_draw("reference_crop_y", *key))
                counts["characters_too_large"] += box is None
            counts["cropped" if box is not None else "not_cropped"] += 1
            boxes.append(box)
    return ({"schema": REFERENCE_CROP_SCHEMA, "sample_index": index,
             "reference_draw": [[shot + 1, frame] for shot, frame in draw.references], "boxes": boxes}, dict(counts))


def build_reference_crops(dataset, *, spatial_root: Path, output_path: Path, sample_indices: Sequence[int],
                          workers: int) -> dict:
    """按训练的同一抽签给每张参考写裁剪框；仅 CPU，读 SAM，不解码视频。与读者范围、参考行写法无关，各组共用。

    摘要（同名 ``.summary.json``）给出「人物框太大包不住所以不裁」占全部参考的比例。"""
    from collections import Counter
    from contextlib import ExitStack
    import multiprocessing
    import time

    if output_path.exists():
        raise FileExistsError(output_path)
    if type(workers) is not int or workers < 1 or len(set(sample_indices)) != len(sample_indices):
        raise ValueError("reference crops need a positive worker count and distinct sample indices")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started, counts = time.monotonic(), Counter(dict.fromkeys(CROP_COUNTS, 0))
    staged = output_path.with_suffix(output_path.suffix + ".partial")
    with ExitStack() as stack:
        if workers == 1:
            _init_crop_worker(dataset, spatial_root)
            records = map(_crop_sample, sample_indices)
        else:
            pool = stack.enter_context(multiprocessing.get_context("fork").Pool(
                workers, initializer=_init_crop_worker, initargs=(dataset, spatial_root)))
            records = pool.imap(_crop_sample, sample_indices, chunksize=16)
        handle = stack.enter_context(staged.open("x"))
        for record, measured in records:
            handle.write(json.dumps(record) + "\n")
            counts.update(measured)
    staged.rename(output_path)
    references = counts["references"]
    summary = {"schema": REFERENCE_CROP_SCHEMA, "output_path": str(output_path), "samples": len(sample_indices),
               "training_seed": dataset.training_seed, "media": dict(dataset.media),
               "sampling_manifest_path": dataset.metadata_summary["sampling_manifest_path"],
               "crop_area": list(CROP_AREA),
               "seconds": time.monotonic() - started, **counts,
               "cropped_fraction": counts["cropped"] / references if references else None,
               "uncroppable_fraction_of_references": (counts["characters_too_large"] / references
                                                      if references else None)}
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary
