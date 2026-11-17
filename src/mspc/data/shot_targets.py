"""块里每镜的目标潜变量：补建用的单镜清单行、「源片＋镜 → 文件位置」索引、与首批窗缓存的对拍。

镜的键是 ``(sid, scene_id, source_shot_index)``，与块清单、参考候选库逐字相同。首批按窗打包的缓存文件靠索引里的
潜帧区间按镜复用；其余的镜各成一条单镜清单行，原样交给 R6（``run/precompute.sh``），一镜一个文件。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mspc.data.scene_training_manifest import ANNOTATION_MEDIA

FIRST_BATCH_WINDOW = "first_batch_window"
SHOT = "shot"
# 清单里随窗不变的列，与逐镜的列；对拍时前者整格相等，后者取该镜在窗里的那一项。
WINDOW_CELLS = ("video", "source_fps", "source_num_frames", "source_width", "source_height", "active_box",
                "train_crop_box", "film_key", "source", "scene_id", "container_preroll_frames")
SHOT_CELLS = ("source_shot_segments", "source_frame_indices_by_shot")


def sampled_target_positions(samples: Iterable[Mapping[str, Any]], *,
                             sample_indices: Sequence[int] | None = None) -> dict[str, set[int]]:
    """冻结窗口唯一决定需要缓存的目标镜；不改变成员或参考资格。"""
    selected = None if sample_indices is None else frozenset(sample_indices)
    if selected is not None and (not selected or len(selected) != len(sample_indices)
                                or any(type(index) is not int or index < 0 for index in selected)):
        raise ValueError("cache sample_indices must be nonempty distinct nonnegative dataset indices")
    positions: dict[str, set[int]] = {}
    found = set()
    for index, sample in enumerate(samples):
        if type(sample["sample_index"]) is not int or sample["sample_index"] != index:
            raise ValueError("sampling manifest indices must be consecutive and ordered")
        case, (start, end) = sample["case_id"], sample["span"]
        if (not isinstance(case, str) or not case or type(start) is not int or type(end) is not int
                or not 0 <= start < end):
            raise ValueError(f"sample {index}: invalid target span")
        if selected is not None and index not in selected:
            continue
        found.add(index)
        positions.setdefault(case, set()).update(range(start, end))
    if selected is not None and found != selected:
        raise ValueError("cache sample_indices are outside the complete frozen sampling manifest")
    if not positions:
        raise ValueError("sampling manifest is empty")
    return positions


def member_target_frame_counts(member: Mapping[str, Any], *, media: Mapping) -> tuple[int, ...]:
    """从原镜界按显式目标fps派生完整镜长度；不覆写标注成员、不裁长镜。"""
    from mspc.data.frame_planner import FpsAwareTargetFpsPlanner
    from mspc.data.scene_training_manifest import require_training_media

    media = require_training_media(media)
    planner = FpsAwareTargetFpsPlanner(max_rgb_frames=media["max_rgb_frames"],
        target_fps=media["target_fps"], vae_temporal_stride=media["vae_temporal_stride"])
    return tuple(len(planner.clean_indices_for_shot(shot["start_frame"], shot["end_frame"], member["scene"]["fps"])[0])
                 if shot["usable"] else shot["frames"] for shot in member["shots"])


def sampled_shot_rows(members_path: Path, sampling_path: Path, *, media: Mapping,
                      sample_indices: Sequence[int] | None = None):
    """原成员顺序下冻结窗口涉及的单镜输入；沿用 shot_row 的取帧与裁切 owner。"""
    with sampling_path.open(encoding="utf-8") as stream:
        wanted = sampled_target_positions((json.loads(line) for line in stream), sample_indices=sample_indices)
    found = set()
    with members_path.open(encoding="utf-8") as stream:
        for line in stream:
            member = json.loads(line)
            case = member["case_id"]
            if case not in wanted:
                continue
            if case in found:
                raise ValueError(f"duplicate sampled member: {case}")
            found.add(case)
            scene = {**member["scene"], "shots": member["shots"]}
            for position in sorted(wanted[case]):
                if position >= len(member["shots"]) or member["shots"][position]["usable"] is not True:
                    raise ValueError(f"{case}: sampled target is missing or unusable: {position}")
                yield member, position, shot_row(scene, position, media=media)
    if found != set(wanted):
        raise ValueError(f"{len(set(wanted) - found)} sampled members are missing")


def shot_key(shot: Mapping[str, Any]) -> tuple[str, str, int]:
    return shot["sid"], shot["scene_id"], shot["source_shot_index"]


def case_id(shot: Mapping[str, Any]) -> str:
    """单镜目标缓存的相对名字，转义源片/场景中的路径分隔符。"""
    for field in ("sid", "scene_id"):
        value = shot[field]
        if not isinstance(value, str) or not value or value in {".", ".."} or "\x00" in value:
            raise ValueError(f"invalid {field}")
    number = shot["source_shot_index"]
    if type(number) is not int or number < 0:
        raise ValueError("source_shot_index must be a nonnegative int")
    source, scene = (shot[k].replace("%", "%25").replace("/", "%2F") for k in ("sid", "scene_id"))
    return f"{source}/{scene}:{number}"


def latent_frames(train_frames: int) -> int:
    if type(train_frames) is not int or train_frames < 5 or (train_frames - 1) % 4:
        raise ValueError(f"train_frames must be 4n+1 with n >= 1: {train_frames!r}")
    return (train_frames - 1) // 4 + 1


def planned_row(scene: Mapping[str, Any], position: int, *, media: Mapping) -> dict[str, Any]:
    """把块里第 ``position`` 镜当成一个单镜的窗，交给场景族清单的现行生产端（取帧规则与中心裁都在那里）。"""
    from mspc.data.multishot import SCENE_MAPPING_SCHEMA
    from mspc.data.scene_training_manifest import manifest_row

    shot, revision = scene["shots"][position], scene["boundary_revision"]
    return manifest_row({**scene, "window_id": case_id(shot), "scene_id": shot["scene_id"], "shots": [shot],
                         "boundary_revision": {**revision, "shots": [revision["shots"][position]]}},
                        mapping_schema=SCENE_MAPPING_SCHEMA, media=media)


def shot_row(scene: Mapping[str, Any], position: int, *, media: Mapping) -> dict[str, Any]:
    """块里第 ``position`` 镜的单镜清单行（R6 的输入），另带镜的两个键（``scene_id`` 生产端已带）。

    帧数与裁切框各有两个出处（块清单、R6 的取帧规则与中心裁），不等即报错：索引里的训练帧数、参考帧与标注代理片
    用的框都以块清单为准。
    """
    shot = scene["shots"][position]
    annotation = planned_row(scene, position, media=ANNOTATION_MEDIA)
    if annotation["shot_groups"] != [[0, shot["render_frames"]]]:
        raise ValueError(f"{annotation['case_id']}: annotation frame planner differs from block render_frames")
    if annotation["train_crop_box"] != scene["crop_box"]:
        raise ValueError(f"{annotation['case_id']}: annotation crop differs from block crop_box")
    row = planned_row(scene, position, media=media)
    return {**row, "sid": shot["sid"], "source_shot_index": shot["source_shot_index"]}


def write_manifest_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """R6 的 CSV 载体：数组单元格写 JSON，空的追溯项写空单元格。返回行数。"""
    count = 0
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for row in rows:
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow({name: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                             if isinstance(value, list) else value for name, value in row.items()})
            count += 1
    if not count:
        raise ValueError(f"{path}: no rows")
    tmp.replace(path)
    return count


def manifest_cells(row: Mapping[str, str]) -> dict[str, Any]:
    """清单 CSV 的一行（单元格是文本）还原成值，只取对拍要比的列。"""
    values: dict[str, Any] = {name: json.loads(row[name]) for name in (*SHOT_CELLS, "shot_groups", "active_box", "train_crop_box")}
    values.update({name: row[name] for name in ("video", "source", "scene_id")})
    values["source_fps"] = float(row["source_fps"])
    values.update({name: int(row[name]) for name in ("source_num_frames", "source_width", "source_height")})
    values["film_key"] = row["film_key"] or None
    values["container_preroll_frames"] = int(row["container_preroll_frames"]) if row["container_preroll_frames"] else None
    return values


def manifest_differences(shot: Mapping[str, Any], window: Mapping[str, Any], position: int) -> list[str]:
    """单镜行与原窗行里同一镜逐项比；返回不等的列名。两边都是 ``manifest_cells`` 还原后的值或生产端的行。"""
    start, end = window["shot_groups"][position]
    differing = [name for name in WINDOW_CELLS if shot[name] != window[name]]
    differing += [name for name in SHOT_CELLS if shot[name] != [window[name][position]]]
    if shot["shot_groups"] != [[0, end - start]]:
        differing.append("shot_groups")
    return differing


def record_groups(record: Mapping[str, Any]) -> tuple[list[list[int]], list[list[int]]]:
    """缓存记录里各镜的训练帧区间与潜帧区间。单镜记录这两栏是 null（原生单段），展开成整段。"""
    if record["shot_groups"] is None:
        return [[0, record["num_frames"]]], [[0, record["input_latents_shape"][2]]]
    return record["shot_groups"], record["latent_shot_groups"]


def index_rows(record: Mapping[str, Any], shots: Sequence[Mapping[str, Any]], origin: str) -> list[dict[str, Any]]:
    """一条缓存记录（一个窗文件或一个单镜文件）→ 它装的各镜的索引行；``shots`` 按文件里的镜序给。

    ``file`` 相对这一出处的缓存目录（即记录自己的 ``cache_file``），潜帧区间半开。
    """
    if not isinstance(origin, str) or not origin.strip() or origin != origin.strip():
        raise ValueError("target cache origin requires an explicit non-empty source name")
    frames, latents = record_groups(record)
    if len(frames) != len(shots):
        raise ValueError(f"{record['case_id']}: cache holds {len(frames)} shots, {len(shots)} given")
    rows = []
    for shot, (start, end), (latent_start, latent_end) in zip(shots, frames, latents, strict=True):
        if latent_end - latent_start != latent_frames(end - start):
            raise ValueError(f"{record['case_id']}: latent range differs from (frames-1)/4+1")
        sid, scene_id, source_shot_index = shot_key(shot)
        rows.append({"sid": sid, "scene_id": scene_id, "source_shot_index": source_shot_index, "origin": origin,
                     "file": record["cache_file"], "latent_start": latent_start, "latent_end": latent_end,
                     "train_frames": end - start})
    return rows


def reconcile(row: Mapping[str, Any], record: Mapping[str, Any], position: int,
              scene: Mapping[str, Any], block_position: int, *, media: Mapping) -> list[str]:
    """首批缓存里的一镜（记录第 ``position`` 镜）对块清单里的同一镜；返回对不上的项。

    比的是缓存记录里写下的实际读帧（源帧段、源帧号、裁切框、源片），不是当时的清单。
    """
    shot = scene["shots"][block_position]
    planned = planned_row(scene, block_position, media=media)
    expected_frames = planned["shot_groups"][0][1]
    annotation_frames = planned_row(scene, block_position, media=ANNOTATION_MEDIA)["shot_groups"][0][1]
    annotation_mismatch = shot["render_frames"] != annotation_frames
    evidence = record["provenance"]["media"]
    problems = []
    if row["train_frames"] != expected_frames or annotation_mismatch:
        problems.append("train_frames")
    if row["latent_end"] - row["latent_start"] != latent_frames(expected_frames) or annotation_mismatch:
        problems.append("latent_frames")
    if evidence["source_shot_segments"][position] != [shot["start_frame"], shot["end_frame"]]:
        problems.append("source_segment")
    if evidence["sampling"]["frame_indices_by_shot"][position] != planned["source_frame_indices_by_shot"][0]:
        problems.append("frame_indices")
    if evidence["preprocessing"]["spatial"][0]["train_crop_box"] != planned["train_crop_box"]:
        problems.append("crop_box")
    if evidence["files"][0]["path"] != scene["source_video_path"]:
        problems.append("video")
    return problems


def pick_spread(rows: Sequence[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    """各层等量、层内按训练帧数从短到长等距取（含最短与最长）；不抽签，重跑结果相同。行带 ``stratum``。"""
    strata: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        strata.setdefault(row["stratum"], []).append(row)
    if type(count) is not int or count < len(strata):
        raise ValueError(f"count must be an int covering all {len(strata)} strata")
    picked: list[Mapping[str, Any]] = []
    for number, name in enumerate(sorted(strata)):
        quota = count // len(strata) + (number < count % len(strata))
        group = sorted(strata[name], key=lambda r: (r["train_frames"], *shot_key(r)))
        if quota > len(group):
            raise ValueError(f"stratum {name} has {len(group)} shots, {quota} wanted")
        picked += group[:1] if quota == 1 else [group[round(i * (len(group) - 1) / (quota - 1))] for i in range(quota)]
    return picked


def bf16_steps(a: Any, b: Any) -> Any:
    """两个 bfloat16 张量逐元素相隔几个可表示值：0 是数值相同（±0 算相同），1 是相邻的两个 BF16 数。"""
    import torch

    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or a.shape != b.shape:
        raise ValueError(f"need two bfloat16 tensors of one shape: {a.dtype} {tuple(a.shape)} vs {b.dtype} {tuple(b.shape)}")
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("non-finite latents")

    def ordinal(x):  # 符号＋幅值的位型 → 随数值单调的整数
        bits = x.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (ordinal(a) - ordinal(b)).abs()
