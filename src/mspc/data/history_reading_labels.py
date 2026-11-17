"""读历史标签：从块标注与 SAM 掩码程序生成「每个所需角色选哪一刻、指在哪、目标镜各人镜首点」。

一段训练样本（块内连续目标镜 ``span``）的历史是段前的可用镜。所需角色＝目标镜里在场、且在历史镜里出现过的
登记角色（人、动物、群体都算，登记表不分）。每人先过硬条件，再择优：

- 硬条件：此人在该帧有掩码且不低于画面 ``reference_min_area``；该镜镜末状态（开镜穿着与手持物，加本镜的穿着、
  手持变化）等于此人在目标段第一次在场那一镜的开镜状态。没有一镜满足状态的，退为「只供身份」，``state_match`` 记 false。
- 择优（面积代理）：掩码面积大、不碰画面边、同帧其他已登记实体少；同分取近。避开切点：每镜首尾 ``cut_margin`` 帧不选。
  面积低于 ``weak_area`` 的记 ``weak``，只供统计。
同一时刻（同镜同帧）的多人合并成一张参考；时刻多于 ``max_references`` 时按时刻内最高择优分留，落选时刻上的人
没有参考（``dropped_by_cap`` 记名字）。规划器看的参考与渲染器拿的参考就是这同一份 ``moments``。点取掩码外接框的中心（0–1000 相对坐标），
与推理时从计划镜首框取中心同一定义（统筹 10-31：训练与推理同一定义）；
目标镜镜首（第 0 帧）没有本人掩码、或掩码小于 ``start_min_area`` 的写 null。掩码只在每镜第 0、4、8… 帧上有，
候选帧就是这些帧。null、weak、只供身份都只留在标签与规划器作答里，不进渲染器。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping, Sequence

import numpy as np

BORDER_PENALTY = 0.5
LABEL_SCHEMA = "history_reading_labels_v2"


@dataclass(frozen=True)
class LabelRules:
    """标签构造的门槛（画面面积占比与帧数）；全部显式给出。"""
    reference_min_area: float
    weak_area: float
    start_min_area: float
    cut_margin: int
    max_references: int

    def __post_init__(self):
        if not (0 < self.start_min_area < 1 and 0 < self.reference_min_area <= self.weak_area < 1):
            raise ValueError("label areas need 0 < reference_min_area <= weak_area < 1 and 0 < start_min_area < 1")
        if type(self.cut_margin) is not int or self.cut_margin < 0:
            raise ValueError("cut_margin must be a nonnegative int")
        if type(self.max_references) is not int or self.max_references < 1:
            raise ValueError("max_references must be a positive int")


def decode_rle(rle: Mapping[str, Any]) -> np.ndarray:
    """COCO 压缩 RLE → [H, W] 布尔阵（列优先，与 pycocotools 一致）。"""
    height, width = rle["size"]
    text, counts, position = rle["counts"], [], 0
    while position < len(text):
        value, shift, more = 0, 0, True
        while more:
            char = ord(text[position]) - 48
            value |= (char & 0x1F) << (5 * shift)
            more = bool(char & 0x20)
            position += 1
            shift += 1
            if not more and char & 0x10:
                value |= -1 << (5 * shift)
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
    if sum(counts) != height * width or any(c < 0 for c in counts):
        raise ValueError("RLE counts do not cover the mask")
    flat = np.zeros(height * width, dtype=bool)
    start = 0
    for index, count in enumerate(counts):
        if index % 2:
            flat[start:start + count] = True
        start += count
    return flat.reshape(width, height).T


def _registry(annotation):
    return {entry["name"]: entry for entry in annotation["characters"]}


def opening_state(annotation: Mapping[str, Any], name: str, index: int) -> dict | None:
    """第 index 镜开镜时此人的穿着与手持物；穿着缺省取登记穿着；此镜不在场返回 None。"""
    entry = next((e for e in annotation["shots"][index]["present"] if e["name"] == name), None)
    if entry is None:
        return None
    outfit = entry.get("outfit") or _registry(annotation)[name].get("outfit") or ""
    return {"outfit": outfit, "holding": sorted(entry.get("holding", []))}


def closing_state(annotation: Mapping[str, Any], name: str, index: int) -> dict | None:
    """第 index 镜镜末状态：开镜值加本镜的穿着、手持变化。"""
    state = opening_state(annotation, name, index)
    if state is None:
        return None
    for change in annotation["shots"][index]["changes"]:
        if change["name"] != name:
            continue
        if change["what"] == "holding":
            if not isinstance(change["after"], list):
                raise ValueError(f"holding change for {name} in shot {index + 1} is not a list")
            state["holding"] = sorted(change["after"])
        elif change["what"] == "outfit":
            state["outfit"] = change["after"]
    return state


def needed_characters(annotation: Mapping[str, Any], span: Sequence[int], history: Sequence[int]) -> list[str]:
    """目标镜里在场、且在历史镜里出现过的登记人物，按目标镜里首次出现的顺序。"""
    registry = _registry(annotation)
    seen = {e["name"] for index in history for e in annotation["shots"][index]["present"]}
    names = []
    for index in range(*span):
        for entry in annotation["shots"][index]["present"]:
            if entry["name"] in registry and entry["name"] in seen and entry["name"] not in names:
                names.append(entry["name"])
    return names


def box_center(mask: np.ndarray) -> list[int]:
    """掩码外接框的中心，0–1000 相对坐标（x, y）；与计划镜首框取中心（``mspc.models.planner_tokens.box_point``）同一定义。"""
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    return [min(1000, int(round((xs.min() + xs.max() + 1) / 2 * 1000 / width))),
            min(1000, int(round((ys.min() + ys.max() + 1) / 2 * 1000 / height)))]


class ShotMasks:
    """一个块的 masks.json：按镜号（从 0 起）与帧号取已登记实体的掩码，解码结果缓存。"""

    def __init__(self, masks: Mapping[str, Any]):
        self.masks, self.cache = masks, {}

    def names(self, index: int) -> list[str]:
        shot = self.masks["shots"].get(f"s{index + 1:02d}")
        return [] if shot is None else sorted({t["name"] for t in shot["tracks"].values()})

    def frames(self, index: int, name: str) -> list[int]:
        shot = self.masks["shots"].get(f"s{index + 1:02d}")
        if shot is None:
            return []
        return sorted({int(f) for t in shot["tracks"].values() if t["name"] == name for f in t["frames"]})

    def mask(self, index: int, name: str, frame: int) -> np.ndarray | None:
        key = (index, name, frame)
        if key not in self.cache:
            shot = self.masks["shots"].get(f"s{index + 1:02d}")
            rles = [] if shot is None else [t["frames"][str(frame)]["rle"] for t in shot["tracks"].values()
                                              if t["name"] == name and str(frame) in t["frames"]]
            if not rles:
                self.cache[key] = None
            else:
                union = decode_rle(rles[0])
                for rle in rles[1:]:
                    union |= decode_rle(rle)
                self.cache[key] = union
        return self.cache[key]


def frame_facts(masks: ShotMasks, index: int, name: str, frame: int, people: set[str], min_area: float) -> dict | None:
    """该帧此人的掩码面积占比、是否碰边、同帧其他已登记角色数（各自不低于 ``min_area`` 才算）；
    此人掩码缺或小于 ``min_area`` 返回 None。"""
    mask = masks.mask(index, name, frame)
    if mask is None or mask.mean() < min_area:
        return None
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    others = sum(1 for other in masks.names(index) if other != name and other in people
                 and (m := masks.mask(index, other, frame)) is not None and m.mean() >= min_area)
    return {"area": float(mask.mean()), "border": bool(xs.min() == 0 or ys.min() == 0 or xs.max() == width - 1
                                                         or ys.max() == height - 1), "others": others}


def preference(facts: Mapping[str, Any]) -> float:
    return facts["area"] * (BORDER_PENALTY if facts["border"] else 1.0) / (1 + facts["others"])


def build_labels(annotation: Mapping[str, Any], masks: Mapping[str, Any], frame_counts: Sequence[int],
                 usable: Sequence[int], span: Sequence[int], rules: LabelRules) -> dict:
    """一段样本的标签。``frame_counts`` 是块内各镜 16 fps 帧数，``usable`` 是可用镜下标。"""
    start, end = span
    if not 0 <= start < end <= len(frame_counts):
        raise ValueError(f"span {span} is outside the block")
    history = [index for index in range(start) if index in set(usable)]
    shot_masks = ShotMasks(masks)
    names = needed_characters(annotation, (start, end), history)
    picks = []
    for name in names:
        target = next(i for i in range(start, end) if any(e["name"] == name for e in annotation["shots"][i]["present"]))
        want = opening_state(annotation, name, target)
        best = None
        for index in history:
            if opening_state(annotation, name, index) is None:
                continue
            consistent = closing_state(annotation, name, index) == want
            for frame in shot_masks.frames(index, name):
                if not rules.cut_margin <= frame < frame_counts[index] - rules.cut_margin:
                    continue
                facts = frame_facts(shot_masks, index, name, frame, set(_registry(annotation)), rules.reference_min_area)
                if facts is None:
                    continue
                key = (consistent, preference(facts), index, frame)
                if best is None or key > best[0]:
                    best = (key, facts)
        if best is None:
            continue
        (consistent, score, index, frame), facts = best
        picks.append({"name": name, "shot_index": index, "frame": frame, "state_match": consistent, "preference": score,
                      "point_2d": box_center(shot_masks.mask(index, name, frame)), **facts,
                      "weak": facts["area"] < rules.weak_area})
    best_at: dict[tuple[int, int], float] = {}
    for p in picks:
        moment = (p["shot_index"], p["frame"])
        best_at[moment] = max(best_at.get(moment, 0.0), p["preference"])
    kept = set(sorted(best_at, key=lambda m: (-best_at[m], m))[:rules.max_references])
    dropped = [p["name"] for p in picks if (p["shot_index"], p["frame"]) not in kept]
    picks = [p for p in picks if (p["shot_index"], p["frame"]) in kept]
    moments = sorted(kept)
    registry = _registry(annotation)
    starts = []
    for index in range(start, end):
        present = [e["name"] for e in annotation["shots"][index]["present"] if e["name"] in registry]
        entries = []
        for name in present:
            mask = shot_masks.mask(index, name, 0)
            point = box_center(mask) if mask is not None and mask.mean() >= rules.start_min_area else None
            entries.append({"name": name, "point_2d": point})
        starts.append({"shot_index": index, "characters": entries})
    return {"schema": LABEL_SCHEMA, "span": [start, end], "history": history, "needed": names,
            "picks": picks, "moments": [list(m) for m in moments], "dropped_by_cap": dropped, "starts": starts,
            # 构造时的规则随标签存下，推理端的切点约束等从这里读同一份。
            "rules": asdict(rules)}


def _label_row(job):
    member, sample, blocks, rules = job
    shots = member["shots"]
    masks = json.loads((blocks / member["key"] / "masks.json").read_text())
    labels = build_labels(member["annotation"], masks, [s["frames"] for s in shots],
                          [i for i, s in enumerate(shots) if s["usable"]], sample["span"], rules)
    return {"sample_index": sample["sample_index"], "case_id": sample["case_id"], "key": member["key"], **labels}


def main(argv: Sequence[str] | None = None) -> int:
    """为一份采样表的每个样本生成读历史标签，一行一个样本（含没有历史的样本，picks 为空）。"""
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    parser = argparse.ArgumentParser(description=main.__doc__)
    for flag in ("--members", "--sampling", "--spatial-blocks", "--output"):
        parser.add_argument(flag, type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    for flag in ("--reference-min-area", "--weak-area", "--start-min-area"):
        parser.add_argument(flag, type=float, required=True)
    parser.add_argument("--cut-margin", type=int, required=True)
    parser.add_argument("--max-references", type=int, required=True)
    args = parser.parse_args(argv)
    rules = LabelRules(args.reference_min_area, args.weak_area, args.start_min_area, args.cut_margin,
                       args.max_references)
    members = {}
    with args.members.open() as stream:
        for line in stream:
            member = json.loads(line)
            members[member["case_id"]] = member
    with args.sampling.open() as stream:
        jobs = [(members[s["case_id"]], s, args.spatial_blocks, rules) for s in map(json.loads, stream)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool, args.output.open("x") as out:
        for row in pool.map(_label_row, jobs, chunksize=16):
            out.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
