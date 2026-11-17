"""长集冻结 v1 修订 4：按看帧确认的漏检硬切修镜头首尾、列出排除镜。

一次性、确定性的机械修订。输入是
``configs/benchmark/script_state_bench_long_boundary_review_r4.json``：逐镜 TransNetV2 扫出的疑似切点
（阈值 0.5 与 0.2–0.5 低置信峰）及 Claude 亲看的结论。只有 ``verdict == "cut"`` 的条目生效：

- 镜头开头或结尾 0.5 秒内的真切点：按训练池生产端同一规则
  （:func:`mspc.data.shot_boundary_revision.revise_shot_boundary`）把起点挪到开头带里最后一个切点、
  终点挪到结尾带里第一个切点；该镜帧数按 :func:`mspc.benchmark.long_story_holdout.render_frame_count`
  （16 fps，与修订 3 同一函数）重算。
- 镜头中段的真切点：该镜写进 ``excluded_shots``（原因 ``mid_shot_cut``），在旧源片校准中排除；不据此裁剪新生成视频。
- 结论表 ``whole_shot_exclusions`` 里裁剪后整镜是频道片尾卡或台标画面的镜：同样写进 ``excluded_shots``
  （原因 ``end_card``）。

窗口成员、dev/test 切分、片簇、剧本、断言、B 的改动都不动。三件产物都是新文件，修订 3 保留以便复现：
候选清单（改源区间与帧数）、A/B 配对清单（改 ``source_span``、``frame_count`` 与 ``aligned_shots``）、
冻结清单（指针改指新候选清单，记修订 4 与排除镜）。
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from mspc.benchmark.long_story_holdout import MAX_SEGMENT_FRAMES, min_segments, render_frame_count, shot_seconds
from mspc.data.shot_boundary_revision import (
    BOUNDARY_EDGE_SECONDS,
    SHOT_BOUNDARY_REVISION_VERSION,
    revise_shot_boundary,
)

REVISION = 4
REVISION_DATE = "2026-10-10"
USER_DECISION = "按看帧复核结论修订"
REVIEW_SCHEMA = "long_story_boundary_review_v1"
#: 判官备料格式随修订 4 升到 v3：备料直接读冻结清单的 ``excluded_shots``（历史备料格式；不用于新版生成视频评分）。
PREPARED_SCHEMA = "mspc_long_script_judge_prepared_v3"
EXCLUSION_REASONS = ("mid_shot_cut", "end_card")
EXCLUSION_RULE = (
    "镜头中段（距镜首、镜尾都超过 0.5 秒）有看帧确认的硬切，或裁剪后整镜为频道片尾卡、台标画面：该镜不出题，"
    "不进任何读数的分子与分母，涉及该镜的跨镜断言同样跳过；同窗其余镜照常计分；读数同时报排除镜数与跳过的断言数。"
)
TRIM_RULE = (
    f"镜首或镜尾 {BOUNDARY_EDGE_SECONDS:g} 秒内看帧确认的硬切：起点挪到开头带里最后一个切点、终点挪到结尾带里第一个切点"
    "（shot_boundary_revision.revise_shot_boundary，训练池同一规则）；该镜帧数按 16 fps 重算（修订 3 同一函数）。"
)


def confirmed_cuts(review: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    """按 (窗, 镜号 1 起) 汇总 ``verdict == "cut"`` 的条目：头尾切点与中段切点分开。"""

    if review.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError(f"review schema must be {REVIEW_SCHEMA}")
    shots: dict[tuple[str, int], dict[str, Any]] = defaultdict(lambda: {"edge": [], "mid": []})
    for item in review["items"]:
        if item["verdict"] not in {"cut", "no_cut", "uncertain"}:
            raise ValueError(f"{item['window_id']} s{item['shot']}: unknown verdict {item['verdict']!r}")
        if item["verdict"] != "cut":
            continue
        entry = shots[(item["window_id"], int(item["shot"]))]
        entry["segment"] = list(item["segment"])
        entry["fps"] = float(item["fps"])
        entry["split"] = item["split"]
        entry["mid" if item["position"] == "mid" else "edge"].append(int(item["frame"]))
    return dict(shots)


def _revised_segment(story: Mapping[str, Any], shot: Mapping[str, Any], cuts: Sequence[int]) -> list[int]:
    revised = revise_shot_boundary({
        "version": SHOT_BOUNDARY_REVISION_VERSION,
        "source_shot_segment": [shot["start_frame"], shot["end_frame"]],
        "source_fps": float(story["fps"]),
        "transnet_cuts": sorted(set(cuts)),
    })
    return revised["revised_shot_segment"]


def convert_candidate(candidate: Mapping[str, Any], cuts: Mapping[tuple[str, int], Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """候选清单：确认的头尾切点修源区间，该镜帧数重算；返回 (新清单, 逐镜修改记录)。"""

    out = copy.deepcopy(dict(candidate))
    stories = {story["story_id"]: story for story in out["stories"]}
    trims = []
    for (story_id, number), entry in sorted(cuts.items()):
        if not entry["edge"]:
            continue
        story = stories[story_id]
        shot = story["shots"][number - 1]
        if [shot["start_frame"], shot["end_frame"]] != entry["segment"]:
            raise ValueError(f"{story_id} s{number}: review segment {entry['segment']} differs from candidate")
        start, end = _revised_segment(story, shot, entry["edge"])
        old = {"start_frame": shot["start_frame"], "end_frame": shot["end_frame"], "render_frames": shot["render_frames"]}
        shot["start_frame"], shot["end_frame"] = start, end
        count = render_frame_count(shot_seconds(story, shot))
        if count > MAX_SEGMENT_FRAMES:
            raise ValueError(f"{story_id} s{number}: {count} frames exceed the {MAX_SEGMENT_FRAMES}-frame cap")
        shot["render_frames"] = count
        trims.append({"window_id": story_id, "shot_id": f"s{number:02d}", "source_shot_index": shot["source_shot_index"],
                      "cuts": sorted(entry["edge"]), "before": old,
                      "after": {"start_frame": start, "end_frame": end, "render_frames": count}})
    for story_id in {trim["window_id"] for trim in trims}:
        story = stories[story_id]
        frames = [shot["render_frames"] for shot in story["shots"]]
        story["render_frames"] = sum(frames)
        story["min_segments"] = min_segments(frames)
    out["summary"]["min_segments"] = {
        str(key): value for key, value in sorted(Counter(story["min_segments"] for story in out["stories"]).items())}
    out["boundary_revision"] = {
        "revision": REVISION, "trim_rule": TRIM_RULE,
        "changed_fields": ["stories[].shots[].start_frame", "stories[].shots[].end_frame", "stories[].shots[].render_frames",
                           "stories[].render_frames", "stories[].min_segments", "summary.min_segments"],
        "trims": trims,
    }
    return out, trims


def excluded_shots(candidate_r4: Mapping[str, Any], cuts: Mapping[tuple[str, int], Mapping[str, Any]],
                   whole_shot: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """排除镜：中段真切的镜（``mid_shot_cut``）与结论表点名的整镜片尾卡（``end_card``）；源区间取修订 4。"""

    stories = {story["story_id"]: story for story in candidate_r4["stories"]}
    rows = {}
    for (story_id, number), entry in cuts.items():
        if entry["mid"]:
            rows[(story_id, number)] = {"split": entry["split"], "reason": "mid_shot_cut", "cut_frames": sorted(entry["mid"])}
    for item in whole_shot:
        key = (item["window_id"], int(item["shot"]))
        if item["reason"] not in EXCLUSION_REASONS or item["reason"] == "mid_shot_cut":
            raise ValueError(f"{key}: whole-shot exclusion reason must be end_card, got {item['reason']!r}")
        if key in rows:
            raise ValueError(f"{key}: shot is already excluded for {rows[key]['reason']}")
        rows[key] = {"split": item["split"], "reason": item["reason"], "cut_frames": []}
    out = []
    for (story_id, number), row in sorted(rows.items()):
        shot = stories[story_id]["shots"][number - 1]
        out.append({"window_id": story_id, "split": row["split"], "shot_id": f"s{number:02d}",
                    "source_shot_index": shot["source_shot_index"], "reason": row["reason"],
                    "cut_frames": row["cut_frames"], "source_span": [shot["start_frame"], shot["end_frame"]]})
    return out


def convert_pairs(payload: Mapping[str, Any], candidate_r3: Mapping[str, Any], candidate_r4: Mapping[str, Any]) -> dict[str, Any]:
    """A/B 配对清单：修过的镜改 ``source_span``、``frame_count``，``aligned_shots`` 按新区间重排。"""

    old = {story["story_id"]: story for story in candidate_r3["stories"]}
    new = {story["story_id"]: story for story in candidate_r4["stories"]}
    out = copy.deepcopy(dict(payload))
    for pair in out["pairs"]:
        before, after = old[pair["story_id"]], new[pair["story_id"]]
        for side in ("context_a", "context_b"):
            shots = pair[side]["shots"]
            if len(shots) != len(after["shots"]):
                raise ValueError(f"{pair['story_id']}/{side}: {len(shots)} shots, candidate has {len(after['shots'])}")
            for shot, src_old, src_new in zip(shots, before["shots"], after["shots"], strict=True):
                if shot["source_span"] != [src_old["start_frame"], src_old["end_frame"]]:
                    raise ValueError(f"{pair['story_id']}/{side}/{shot['shot_id']}: source_span differs from revision 3")
                shot["source_span"] = [src_new["start_frame"], src_new["end_frame"]]
                shot["frame_count"] = src_new["render_frames"]
        aligned, cursor = [], 0
        for row, src in zip(pair["aligned_shots"], after["shots"], strict=True):
            count = src["end_frame"] - src["start_frame"]
            aligned.append({**row, "start_frame": cursor, "end_frame": cursor + count, "source_start_frame": src["start_frame"]})
            cursor += count
        pair["aligned_shots"] = aligned
    out["boundary_revision"] = {"revision": REVISION, "trim_rule": TRIM_RULE,
                                "changed_fields": ["pairs[].context_a.shots[].source_span", "pairs[].context_a.shots[].frame_count",
                                                   "pairs[].context_b.shots[].source_span", "pairs[].context_b.shots[].frame_count",
                                                   "pairs[].aligned_shots"]}
    return out


def convert_freeze(freeze: Mapping[str, Any], *, predecessor_path: str, candidate_path_r4: str, review_path: str,
                   trims: Sequence[Mapping[str, Any]], excluded: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """冻结清单：窗口指针改指修订 4 候选清单，记修订 4、修首尾清单与排除镜。成员、切分、协议字段不动。"""

    out = copy.deepcopy(dict(freeze))
    if out["revision"] != 3:
        raise ValueError(f"expected revision 3, got {out['revision']}")
    previous = out["source_manifest"]
    out["source_manifest"] = candidate_path_r4
    splits = {window["window_id"]: window["split"] for window in out["windows"]}
    for window in out["windows"]:
        if window["source_reference"]["path"] != previous:
            raise ValueError(f"{window['window_id']}: source_reference.path is not {previous}")
        window["source_reference"]["path"] = candidate_path_r4
    out["excluded_shots"] = list(excluded)
    out["protocol"]["prepared_schema"] = PREPARED_SCHEMA
    out["revision"] = REVISION
    out["revision_date"] = REVISION_DATE
    out["revision_4"] = {
        "user_decision": USER_DECISION,
        "predecessor": predecessor_path,
        "predecessor_source_manifest": previous,
        "review": review_path,
        "changed": "看帧确认的镜头首尾硬切按训练池同一规则修源区间并重算帧数；镜头中段硬切的镜与裁后整镜是片尾卡的镜列入 excluded_shots。",
        "trim_rule": TRIM_RULE,
        "exclusion_rule": EXCLUSION_RULE,
        "unchanged": "窗口成员、dev/test、片簇、排除并集、剧本、断言、B 的改动、判官配置、取帧版本、16 fps 帧数规则。",
        "protocol_change": f"protocol.prepared_schema 升到 {PREPARED_SCHEMA}：判官备料直接读 excluded_shots，排除镜不出题、不取证。",
        "trims": [dict(trim) for trim in trims],
        "trimmed_shots_by_split": dict(Counter(splits[trim["window_id"]] for trim in trims)),
        "excluded_shots_by_split": dict(Counter(row["split"] for row in excluded)),
        "excluded_shots_by_reason": dict(Counter(row["reason"] for row in excluded)),
    }
    return out


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")



def freeze_shot_exclusions(freeze):
    """冻结清单的排除镜（修订 4 起）：返回 (冻结窗口集合, {窗: {镜号: 行}})。"""
    if "excluded_shots" not in freeze:
        raise ValueError("freeze manifest must carry excluded_shots (revision 4 or later)")
    windows = {window["window_id"] for window in freeze["windows"]}
    table = {}
    for row in freeze["excluded_shots"]:
        if row["window_id"] not in windows:
            raise ValueError(f"excluded shot {row['window_id']}/{row['shot_id']} is not a frozen window")
        table.setdefault(row["window_id"], {})[row["shot_id"]] = row
    return windows, table


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("review", "candidate-r3", "freeze-r3", "pairs-r3", "candidate-out", "freeze-out", "pairs-out", "repo-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    review = _read(args.review)
    cuts = confirmed_cuts(review)
    candidate_r3 = _read(args.candidate_r3)
    candidate_r4, trims = convert_candidate(candidate_r3, cuts)
    excluded = excluded_shots(candidate_r4, cuts, review["whole_shot_exclusions"])
    pairs_r4 = convert_pairs(_read(args.pairs_r3), candidate_r3, candidate_r4)
    freeze_r4 = convert_freeze(
        _read(args.freeze_r3), predecessor_path=str(args.freeze_r3.resolve().relative_to(root)),
        candidate_path_r4=str(args.candidate_out.resolve().relative_to(root)),
        review_path=str(args.review.resolve().relative_to(root)), trims=trims, excluded=excluded)
    _write_new(args.candidate_out, candidate_r4)
    _write_new(args.pairs_out, pairs_r4)
    _write_new(args.freeze_out, freeze_r4)
    print(json.dumps({"trims": len(trims), "excluded": len(excluded),
                      "trimmed_by_split": freeze_r4["revision_4"]["trimmed_shots_by_split"],
                      "excluded_by_split": freeze_r4["revision_4"]["excluded_shots_by_split"],
                      "excluded_by_reason": freeze_r4["revision_4"]["excluded_shots_by_reason"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
