#!/usr/bin/env python3
"""空间尺子两项检查的薄入口。计量规则在
:mod:`mspc.evaluation.spatial_layout_probe`。

子命令：

* ``build-items``（本机）：从测试集 v2 编第一项的探针题面、第二项的 40 条切片，各写一份新文件，拒绝覆盖。
* ``measure``（GPU）：读一条臂的成片清单（``[{case_id, video_path}]``，出片的 ``generated_eval_manifest.json``
  即是），逐镜在方位帧、景别帧与布局关键帧上跑 Grounding DINO（本故事全部主体描述一次检测）；布局关键帧
  再用同一批框的 SAM 掩码抠掉主体取 SIFT，算每对相邻镜的 RANSAC 内点数。
* ``make-arms``（CPU，要 ``mspc.inference.wan22.save_video``，用 wan22 的环境）：从地板逐镜片段造三条合成
  坏例臂。每条故事只换靶镜那一段：镜像、照抄重新编码（与出片同一个写盘函数与参数），换背景直接用供体的
  片段；其余片段原样流拷贝拼接，所以非靶镜解码出来与地板逐像素相同。另写地板与照抄臂的判官输入。
* ``summarize-probe`` / ``summarize-ruler``（CPU）：按预注册判据出读数。
* ``measure-heads``（GPU）：一条臂每镜中间帧的头框，给构图题相邻一档的工具读（10-05 改法）。
* ``calibrate-real-layout``（GPU）：地标布局尺子在 300 个真实镜界（镜界类型标注器的校准集，带看帧打的参考
  标签）上读各层「相容」率与跨视频假阳性率。
* ``sheet``（CPU）：把检测框画到量过的帧上拼图，供看帧。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for _root in (SOURCE_ROOT, REPOSITORY_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from mspc.evaluation import spatial_layout_probe as rules
from mspc.evaluation.spatial_layout_probe import (
    ARM_COPY,
    ARM_MIRROR,
    ARM_SWAP,
    ARMS,
    PROBE_SCHEMA,
    PROBE_SIDE,
    PROBE_SIZE,
    background_mask,
    copy_target,
    height_ratio,
    layout_verdicts,
    mirror_frames,
    mirror_target,
    pair_inliers,
    resample_frames,
    shot_side,
    shot_size_gap,
    side_offsets,
    side_probe_cases,
    sift_features,
    size_step_holds,
    subject_head,
    size_probe_cases,
    slice_case_ids,
    subject_boxes,
    subject_phrases,
    swap_donor,
    swap_target,
    wilson_interval,
)

MEASURE_SCHEMA = "mspc_spatial_probe_measure_v1"
TARGETS_SCHEMA = "mspc_spatial_probe_targets_v1"
TARGET_RULES = {ARM_MIRROR: mirror_target, ARM_COPY: copy_target, ARM_SWAP: swap_target}


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_new(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _rate(num: int, den: int) -> dict[str, Any]:
    interval = wilson_interval(num, den)
    return {"num": num, "den": den, "rate": num / den if den else None,
            "wilson95": None if interval is None else [round(interval[0], 4), round(interval[1], 4)]}


# ---------------------------------------------------------------------------
# build-items
# ---------------------------------------------------------------------------


def run_build_items(args: argparse.Namespace) -> int:
    bench = _read(args.bench)
    if bench["schema_version"] != "script_state_bench_v2":
        raise ValueError(f"{args.bench}: expected the script_state_bench_v2 draft")
    for path in (args.probe_out, args.slice_out):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    probe = {
        "schema_version": PROBE_SCHEMA,
        "benchmark_id": PROBE_SCHEMA,
        "evaluation_only": True,
        "fps": bench["fps"],
        "generation_seeds": bench["generation_seeds"],
        "construction": {
            "source_benchmark": str(args.bench),
            "side": {"size": args.side_size, "seed": args.side_seed},
            "size": {"size": args.size_size, "seed": args.size_seed},
        },
        "cases": [
            *side_probe_cases(bench, size=args.side_size, seed=args.side_seed),
            *size_probe_cases(bench, size=args.size_size, seed=args.size_seed),
        ],
    }
    picked = set(slice_case_ids(bench, per_shot_count=args.slice_per_shot_count, seed=args.slice_seed))
    sliced = {
        **bench,
        "benchmark_id": args.slice_benchmark_id,
        "cases": [case for case in bench["cases"] if case["case_id"] in picked],
    }
    _write_new(args.probe_out, probe)
    _write_new(args.slice_out, sliced)
    print(json.dumps({"probe_cases": len(probe["cases"]), "slice_cases": len(sliced["cases"]),
                      "slice_shots": sum(case["shot_count"] for case in sliced["cases"])}))
    return 0


# ---------------------------------------------------------------------------
# measure (GPU)
# ---------------------------------------------------------------------------


def _story_subjects(case: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """检测用的主体顺序与短语：探针读题面自带的 ``detector_phrase``，切片读 v2 镜头 prompt 的括号短描述。"""

    ids = [subject["subject_id"] for subject in case["subjects"]]
    if "probe" in case:
        return ids, [subject["detector_phrase"] for subject in case["subjects"]]
    phrases = subject_phrases(case)
    return ids, [phrases[subject_id] for subject_id in ids]


def _shot_roles(case: Mapping[str, Any], frame_count: int) -> dict[str, list[int]]:
    """一镜要量哪些帧：探针方位三帧或景别中间帧；切片方位三帧＋布局首中末三帧。"""

    if case.get("probe") == PROBE_SIDE:
        return {"side": list(side_offsets(frame_count))}
    if case.get("probe") == PROBE_SIZE:
        return {"size": [(frame_count - 1) // 2]}
    return {"side": list(side_offsets(frame_count)), "layout": [0, (frame_count - 1) // 2, frame_count - 1]}


def measure_story(engine: Any, frames: Sequence[Any], case: Mapping[str, Any], *, profile: Any,
                  detector_frame_batch_size: int, dino_crop_batch_size: int) -> dict[str, Any]:
    subject_ids, phrases = _story_subjects(case)
    height, width = (int(value) for value in frames[0].shape[:2])
    start = 0
    shots: dict[str, Any] = {}
    features: dict[str, list[Any]] = {}
    for shot in case["shots"]:
        count = shot["frame_count"]
        roles = _shot_roles(case, count)
        offsets = sorted({offset for values in roles.values() for offset in values})
        picked = [frames[start + offset] for offset in offsets]
        # 一镜的帧一批检测：同一镜在地板与坏例臂上的输入逐像素相同，批也相同。
        regions = engine.grounded_regions(
            picked, phrases, profile,
            detector_frame_batch_size=detector_frame_batch_size, dino_crop_batch_size=dino_crop_batch_size,
        )
        rows = []
        for offset, frame, frame_regions in zip(offsets, picked, regions, strict=True):
            row: dict[str, Any] = {
                "offset": offset,
                "regions": [{key: region[key] for key in ("box", "score", "phrase_scores", "mask_area_ratio")}
                            for region in frame_regions],
            }
            if offset in roles.get("layout", ()):
                # 与 grounded_regions 同一批框：SAM 掩码走引擎缓存，不重算。
                masks = engine._masks(frame, [region["box"] for region in frame_regions])
                mask = background_mask(masks, (height, width))
                feature = sift_features(frame, mask)
                features.setdefault(shot["shot_id"], []).append(feature)
                row["background_fraction"] = round(float((mask > 0).mean()), 4)
                row["sift_keypoints"] = len(feature[0])
            rows.append(row)
        shots[shot["shot_id"]] = {"frame_count": count, "roles": roles, "frames": rows}
        start += count
    pairs = []
    if features:
        for previous, current in zip(case["shots"], case["shots"][1:]):
            inliers, where = pair_inliers(features[previous["shot_id"]], features[current["shot_id"]])
            pairs.append({
                "pair": [previous["shot_id"], current["shot_id"]],
                "same_location": previous["location_id"] == current["location_id"],
                "cut_type": current["cut_type"],
                "inliers": inliers,
                "best_keyframes": list(where),
            })
    return {"width": width, "height": height, "subject_ids": subject_ids, "phrases": phrases,
            "shots": shots, "pairs": pairs}


def _video_rows(path: Path) -> dict[str, str]:
    rows = _read(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a non-empty list of {{case_id, video_path}} rows")
    videos = {row["case_id"]: row["video_path"] for row in rows}
    if len(videos) != len(rows):
        raise ValueError(f"{path}: duplicate case_id")
    return videos


def run_measure(args: argparse.Namespace) -> int:
    from mspc.evaluation.renderer_qualification_official_backend import (
        DEFAULT_ROI_DETECTOR_CONFIG,
        FROZEN_PUBLIC_ROI_DETECTOR,
        _decode_rgb,
        _DinoRoiEngine,
        _model_batch_from_roi_config,
    )

    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    bench = _read(args.benchmark)
    cases = {case["case_id"]: case for case in bench["cases"]}
    videos = _video_rows(args.videos)
    if set(videos) != set(cases):
        raise ValueError(f"{args.videos} covers {sorted(set(videos) ^ set(cases))[:5]} differently from {args.benchmark}")
    batch = _model_batch_from_roi_config(DEFAULT_ROI_DETECTOR_CONFIG)
    engine = _DinoRoiEngine(args.resource_root, args.device)
    started = time.time()
    stories: dict[str, Any] = {}
    for index, story_id in enumerate(sorted(cases), start=1):
        case = cases[story_id]
        frames, fps = _decode_rgb(Path(videos[story_id]))
        planned = sum(shot["frame_count"] for shot in case["shots"])
        if len(frames) != planned:
            raise ValueError(f"{story_id}: decoded {len(frames)} frames, the benchmark plans {planned}")
        stories[story_id] = {
            "video_path": videos[story_id],
            "fps": fps,
            **measure_story(engine, frames, case, profile=FROZEN_PUBLIC_ROI_DETECTOR,
                            detector_frame_batch_size=batch.detector_frame_batch_size,
                            dino_crop_batch_size=batch.dino_crop_batch_size),
        }
        engine.clear_frame_caches()
        print(f"[{args.arm}] {index}/{len(cases)} {story_id}", file=sys.stderr, flush=True)
    _write_new(args.out, {
        "schema_version": MEASURE_SCHEMA,
        "arm": args.arm,
        "benchmark": str(args.benchmark),
        "videos": str(args.videos),
        "resource_root": str(args.resource_root),
        "roi_detector": FROZEN_PUBLIC_ROI_DETECTOR.payload(),
        "rules": {
            "side_fractions": rules.SIDE_FRACTIONS, "side_overlap_iou": rules.SIDE_OVERLAP_IOU,
            "side_min_gap_fraction": rules.SIDE_MIN_GAP_FRACTION, "sift_ratio": rules.SIFT_RATIO,
            "same_point_radius_px": rules.SAME_POINT_RADIUS_PX, "ransac_threshold_px": rules.RANSAC_THRESHOLD_PX,
            "ransac_confidence": rules.RANSAC_CONFIDENCE, "ransac_max_iterations": rules.RANSAC_MAX_ITERATIONS,
            "ransac_seed": rules.RANSAC_SEED, "mask_dilation_px": rules.MASK_DILATION_PX,
            "min_fundamental_matches": rules.MIN_FUNDAMENTAL_MATCHES,
        },
        "elapsed_seconds": round(time.time() - started, 1),
        "stories": stories,
    })
    print(json.dumps({"arm": args.arm, "stories": len(stories), "out": str(args.out)}))
    return 0


# ---------------------------------------------------------------------------
# make-arms (CPU)
# ---------------------------------------------------------------------------


def _shot_clips(video_path: str, case: Mapping[str, Any]) -> dict[str, Path]:
    """地板逐镜片段：``generate_story_by_single_shot_concat`` 写在 ``<成片目录>/shots/<故事>/<序号>_<镜>.mp4``。"""

    video = Path(video_path)
    clips = {}
    for index, shot in enumerate(case["shots"]):
        clip = video.parent / "shots" / video.stem / f"{index:02d}_{shot['shot_id']}.mp4"
        if not clip.is_file():
            raise FileNotFoundError(f"{case['case_id']}: per-shot clip is missing: {clip}")
        clips[shot["shot_id"]] = clip
    return clips


def run_make_arms(args: argparse.Namespace) -> int:
    import numpy as np

    from mspc.evaluation.continuity_dev_32_public_metrics import (
        public_metrics_cases_from_generated_manifest,
    )
    from mspc.evaluation.renderer_qualification_official_backend import _decode_rgb
    from mspc.inference.wan22 import save_video
    from mspc.core.video import concat_shot_videos

    if args.out_dir.exists():
        raise FileExistsError(f"refusing to reuse {args.out_dir}")
    cases = _read(args.slice)["cases"]
    by_id = {case["case_id"]: case for case in cases}
    manifest = _read(args.floor_manifest)
    floor_rows = {row["case_id"]: row for row in manifest}
    if set(floor_rows) != set(by_id):
        raise ValueError(f"{args.floor_manifest} does not cover exactly the slice stories")
    clips = {story: _shot_clips(floor_rows[story]["video_path"], by_id[story]) for story in by_id}

    def decode(path: Path) -> Any:
        frames, _ = _decode_rgb(path)
        return np.stack(frames)

    targets: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    arm_rows: dict[str, list[dict[str, str]]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        clip_dir = args.out_dir / arm / "target_clips"
        for case in cases:
            story = case["case_id"]
            target = TARGET_RULES[arm](case)
            if target is None:
                arm_rows[arm].append({"case_id": story, "video_path": floor_rows[story]["video_path"]})
                continue
            ids = [shot["shot_id"] for shot in case["shots"]]
            k = ids.index(target)
            count = case["shots"][k]["frame_count"]
            if arm == ARM_SWAP:
                donor_story, donor_shot = swap_donor(cases, story, target)
                replacement = clips[donor_story][donor_shot]
                source = f"{donor_story}/{donor_shot}"
            else:
                if arm == ARM_MIRROR:
                    frames, source = mirror_frames(decode(clips[story][target])), f"{story}/{target}"
                else:
                    frames, source = resample_frames(decode(clips[story][ids[k - 1]]), count), f"{story}/{ids[k - 1]}"
                replacement = clip_dir / f"{story}_{target}.mp4"
                save_video(list(frames), replacement, fps=args.fps, quality=args.video_quality)
            out = args.out_dir / arm / "videos" / f"{story}.mp4"
            concat_shot_videos([replacement if shot_id == target else clips[story][shot_id] for shot_id in ids],
                               output_path=out)
            decoded = len(_decode_rgb(out)[0])
            planned = sum(shot["frame_count"] for shot in case["shots"])
            if decoded != planned:
                raise ValueError(f"{arm}/{story}: arm video decodes to {decoded} frames, the slice plans {planned}")
            targets[arm][story] = {"target": target, "source": source}
            arm_rows[arm].append({"case_id": story, "video_path": str(out)})
            print(f"[{arm}] {story} target={target} source={source}", flush=True)
        _write_new(args.out_dir / arm / "manifest.json", arm_rows[arm])
    _write_new(args.out_dir / "targets.json", {
        "schema_version": TARGETS_SCHEMA,
        "slice": str(args.slice),
        "floor_manifest": str(args.floor_manifest),
        "rules": {
            ARM_MIRROR: "first same-location shot with >= 2 present subjects; whole shot flipped left-right",
            ARM_COPY: "first same-location shot whose shot_size differs from the previous shot; replaced by the "
                      "previous shot resampled to the target frame count",
            ARM_SWAP: "first same-location shot; replaced by the first equal-length shot of a later story "
                      "(cyclic) whose location text differs",
        },
        "targets": targets,
    })
    floor_cases = public_metrics_cases_from_generated_manifest(manifest, arm_id=args.floor_arm_id, generation_seed=args.seed)
    _write_new(args.out_dir / "judge_input_floor.json", {"arm_id": args.floor_arm_id, "cases": floor_cases})
    copy_cases = [
        {**case, "case_id": f"{case['story_id']}-{ARM_COPY}-seed-{args.seed}", "arm_id": ARM_COPY,
         "video_path": str(args.out_dir / ARM_COPY / "videos" / f"{case['story_id']}.mp4")}
        for case in floor_cases if case["story_id"] in targets[ARM_COPY]
    ]
    _write_new(args.out_dir / f"judge_input_{ARM_COPY}.json", {"arm_id": ARM_COPY, "cases": copy_cases})
    print(json.dumps({arm: len(rows) for arm, rows in targets.items()}))
    return 0


# ---------------------------------------------------------------------------
# summarize (CPU)
# ---------------------------------------------------------------------------


def _min_phrase_score() -> float:
    from mspc.evaluation.renderer_qualification_official_backend import (
        FROZEN_PUBLIC_ROI_DETECTOR,
    )

    return FROZEN_PUBLIC_ROI_DETECTOR.box_threshold


def _frame_boxes(story: Mapping[str, Any], shot_id: str, role: str) -> list[dict[str, Any]]:
    shot = story["shots"][shot_id]
    wanted = set(shot["roles"][role])
    threshold = _min_phrase_score()
    return [
        {"offset": frame["offset"],
         "boxes": subject_boxes(frame["regions"], story["subject_ids"], min_phrase_score=threshold)}
        for frame in shot["frames"] if frame["offset"] in wanted
    ]


def _side(story: Mapping[str, Any], shot_id: str, present: Sequence[str]) -> dict[str, str | None]:
    return shot_side([row["boxes"] for row in _frame_boxes(story, shot_id, "side")], present, width=story["width"])


def run_summarize_probe(args: argparse.Namespace) -> int:
    probe = _read(args.probe)
    measure = _read(args.measure)
    side_rows, size_rows = [], []
    for case in probe["cases"]:
        story = measure["stories"][case["case_id"]]
        if case["probe"] == PROBE_SIDE:
            shots = []
            for shot in case["shots"]:
                left = _side(story, shot["shot_id"], ["Subject 1", "Subject 2"])["Subject 1|Subject 2"]
                instructed = shot["instructed_left_to_right"][0]
                shots.append({"shot_id": shot["shot_id"], "instructed_left": instructed, "observed_left": left,
                              "obeyed": None if left is None else left == instructed})
            side_rows.append({"case_id": case["case_id"], "source_case_id": case["source_case_id"], "shots": shots})
        else:
            shots = []
            for shot in case["shots"]:
                (row,) = _frame_boxes(story, shot["shot_id"], "size")
                box = row["boxes"]["Subject 1"]
                shots.append({"shot_id": shot["shot_id"], "shot_size": shot["shot_size"],
                              "height_ratio": height_ratio(None if box is None else box["box"], height=story["height"])})
            close, wide = shots
            judgeable = close["height_ratio"] is not None and wide["height_ratio"] is not None
            size_rows.append({"case_id": case["case_id"], "source_case_id": case["source_case_id"], "shots": shots,
                              "judgeable": judgeable,
                              "close_larger": judgeable and close["height_ratio"] > wide["height_ratio"]})
    left_first = [row["shots"][0] for row in side_rows]
    right_first = [row["shots"][1] for row in side_rows]
    judged_shots = [shot for row in side_rows for shot in row["shots"] if shot["observed_left"] is not None]
    scenes = [row for row in side_rows if all(shot["observed_left"] is not None for shot in row["shots"])]
    readout = {
        "probe": str(args.probe),
        "measure": str(args.measure),
        "side": {
            "scenes": len(side_rows),
            "judgeable_scenes": _rate(len(scenes), len(side_rows)),
            "judgeable_shots": _rate(len(judged_shots), 2 * len(side_rows)),
            "strict_obedience": _rate(sum(all(shot["obeyed"] for shot in row["shots"]) for row in scenes), len(scenes)),
            "subject1_left_obeyed": _rate(sum(bool(s["obeyed"]) for s in left_first),
                                          sum(s["observed_left"] is not None for s in left_first)),
            "subject1_right_obeyed": _rate(sum(bool(s["obeyed"]) for s in right_first),
                                           sum(s["observed_left"] is not None for s in right_first)),
            "subject1_on_left_any_text": _rate(sum(s["observed_left"] == "Subject 1" for s in judged_shots),
                                               len(judged_shots)),
            "same_order_both_texts": _rate(
                sum(row["shots"][0]["observed_left"] == row["shots"][1]["observed_left"] for row in scenes), len(scenes)),
            "rows": side_rows,
        },
        "size": {
            "pairs": len(size_rows),
            "judgeable_pairs": _rate(sum(row["judgeable"] for row in size_rows), len(size_rows)),
            "close_larger": _rate(sum(row["close_larger"] for row in size_rows), sum(row["judgeable"] for row in size_rows)),
            "mean_height_ratio": {
                size: (sum(values) / len(values) if values else None)
                for size, values in (
                    (shot_size, [shot["height_ratio"] for row in size_rows for shot in row["shots"]
                                 if shot["shot_size"] == shot_size and shot["height_ratio"] is not None])
                    for shot_size in ("Close-up shot", "Wide shot")
                )
            },
            "rows": size_rows,
        },
    }
    _write_new(args.out, readout)
    print(json.dumps({key: {k: v for k, v in value.items() if k != "rows"} for key, value in readout.items()
                      if isinstance(value, dict)}, ensure_ascii=False))
    return 0


def _arm_specs(specs: Sequence[str]) -> dict[str, Path]:
    arms: dict[str, Path] = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if not name or not path or name in arms:
            raise ValueError(f"expected unique NAME=PATH, got {spec!r}")
        arms[name] = Path(path)
    return arms


def _side_readout(cases: Mapping[str, Any], measures: Mapping[str, Any], targets: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for story, entry in sorted(targets[ARM_MIRROR].items()):
        shot = entry["target"]
        present = next(s for s in cases[story]["shots"] if s["shot_id"] == shot)["present_subjects"]
        floor = _side(measures["floor"]["stories"][story], shot, present)
        mirror = _side(measures[ARM_MIRROR]["stories"][story], shot, present)
        both = [key for key in floor if floor[key] is not None and mirror[key] is not None]
        rows.append({"story_id": story, "shot_id": shot, "present": present, "floor": floor, "mirror": mirror,
                     "judgeable": bool(both), "flipped": bool(both) and all(floor[k] != mirror[k] for k in both)})
    judgeable = [row for row in rows if row["judgeable"]]
    return {
        "targets": len(rows),
        "coverage": _rate(len(judgeable), len(rows)),
        "flipped": _rate(sum(row["flipped"] for row in judgeable), len(judgeable)),
        "rows": rows,
    }


def _non_target_identity(cases: Mapping[str, Any], measures: Mapping[str, Any], targets: Mapping[str, Any]) -> dict[str, Any]:
    """坏例臂上非靶镜的检测结果与地板逐项相同的镜数（工具题是确定性的，预注册要求完全不变）。"""

    result = {}
    for arm in ARMS:
        same = differ = 0
        examples = []
        for story, case in cases.items():
            target = targets[arm].get(story, {}).get("target")
            for shot in case["shots"]:
                if shot["shot_id"] == target:
                    continue
                left = measures["floor"]["stories"][story]["shots"][shot["shot_id"]]["frames"]
                right = measures[arm]["stories"][story]["shots"][shot["shot_id"]]["frames"]
                if left == right:
                    same += 1
                else:
                    differ += 1
                    examples.append(f"{story}/{shot['shot_id']}")
        result[arm] = {"identical_shots": same, "differing_shots": differ, "differing_examples": examples[:10]}
    return result


def _layout_readout(cases: Mapping[str, Any], measures: Mapping[str, Any], targets: Mapping[str, Any]) -> dict[str, Any]:
    verdicts = {
        arm: layout_verdicts({story: measure["stories"][story]["pairs"] for story in sorted(cases)})
        for arm, measure in measures.items()
    }

    def keyed(arm: str, story: str) -> dict[str, dict[str, Any]]:
        return {"|".join(row["pair"]): row for row in verdicts[arm][story]}

    by_cut: dict[str, list[bool]] = {}
    for story in sorted(cases):
        for row in verdicts["floor"][story]:
            by_cut.setdefault(row["cut_type"], []).append(row["compatible"])
    arms = {}
    for arm in ARMS:
        target_rows, non_target_same_n, non_target_same_verdict, non_target_total = [], 0, 0, 0
        for story in sorted(cases):
            target = targets[arm].get(story, {}).get("target")
            floor_rows, arm_rows = keyed("floor", story), keyed(arm, story)
            for key, floor_row in floor_rows.items():
                arm_row = arm_rows[key]
                if target is not None and target in floor_row["pair"]:
                    target_rows.append({"story_id": story, "pair": floor_row["pair"], "cut_type": floor_row["cut_type"],
                                        "floor_inliers": floor_row["inliers"], "floor_control_median": floor_row["control_median"],
                                        "floor_compatible": floor_row["compatible"], "arm_inliers": arm_row["inliers"],
                                        "arm_control_median": arm_row["control_median"],
                                        "arm_compatible": arm_row["compatible"]})
                else:
                    non_target_total += 1
                    non_target_same_n += floor_row["inliers"] == arm_row["inliers"]
                    non_target_same_verdict += floor_row["compatible"] == arm_row["compatible"]
        floor_ok = [row for row in target_rows if row["floor_compatible"]]
        arms[arm] = {
            "target_pairs": len(target_rows),
            "floor_compatible_target_pairs": len(floor_ok),
            "became_incompatible": _rate(sum(not row["arm_compatible"] for row in floor_ok), len(floor_ok)),
            "non_target_pairs": non_target_total,
            "non_target_same_inliers": non_target_same_n,
            "non_target_same_verdict": non_target_same_verdict,
            "rows": target_rows,
        }
    controls = [row["control_median"] for story in sorted(cases) for row in verdicts["floor"][story]]
    return {
        "floor_compatible_by_cut_type": {cut: _rate(sum(values), len(values)) for cut, values in sorted(by_cut.items())},
        "floor_compatible_all": _rate(sum(sum(v) for v in by_cut.values()), sum(len(v) for v in by_cut.values())),
        "floor_control_median_range": [min(controls), max(controls)] if controls else None,
        "floor_control_sources": {
            source: sum(row["control_source"] == source for story in sorted(cases) for row in verdicts["floor"][story])
            for source in ("same_video", "other_stories")
        },
        "arms": arms,
        "floor_pairs": {story: verdicts["floor"][story] for story in sorted(cases)},
    }


def _head_answer(cases: Mapping[str, Any], measure: Mapping[str, Any], heads: Mapping[str, Any], story: str,
                 shot_id: str) -> dict[str, Any]:
    """相邻一档的工具题：本镜与上一镜里都在场、两镜都量到头框的第一个主体（名单顺序），头框高对数比判方向。"""

    shots = cases[story]["shots"]
    ids = [shot["shot_id"] for shot in shots]
    k = ids.index(shot_id)
    previous, current = shots[k - 1], shots[k]
    story_measure = measure["stories"][story]
    threshold = _min_phrase_score()

    def head(shot: Mapping[str, Any], subject: str) -> list[float] | None:
        row = heads["stories"][story][shot["shot_id"]]
        frame = next(f for f in story_measure["shots"][shot["shot_id"]]["frames"] if f["offset"] == row["offset"])
        owner = subject_boxes(frame["regions"], story_measure["subject_ids"], min_phrase_score=threshold)[subject]
        return subject_head(None if owner is None else owner["box"], row["heads"])

    for subject in story_measure["subject_ids"]:
        if subject in previous["present_subjects"] and subject in current["present_subjects"]:
            before, after = head(previous, subject), head(current, subject)
            if before is not None and after is not None:
                return {"subject": subject, "holds": size_step_holds(before, after, previous["shot_size"], current["shot_size"]),
                        "log_ratio": round(math.log((after[3] - after[1]) / (before[3] - before[1])), 4)}
    return {"subject": None, "holds": None, "log_ratio": None}


def _composition_readout(cases: Mapping[str, Any], targets: Mapping[str, Any], assertions: Path,
                         judges: Mapping[str, Path], measures: Mapping[str, Any], heads: Mapping[str, Any]) -> dict[str, Any]:
    """构图题按 10-05 改法分路：剧本景别与上一镜差 ≥ 2 档读判官，差 1 档读头框高工具，差 0 档不出题。

    判官答案取自同一批已判的行（断言产物里相邻档的题当时也问了，这里按分路丢掉），不重打。
    """

    from mspc.benchmark.script_assertions import (
        COMPOSITION_KIND,
        COMPOSITION_MIN_SIZE_GAP,
        load_script_assertions,
        load_script_judge_answers,
    )

    scripts = load_script_assertions(assertions)
    answers = {name: load_script_judge_answers([path]) for name, path in judges.items()}

    def gap(story: str, shot_id: str) -> int:
        shots = cases[story]["shots"]
        k = [shot["shot_id"] for shot in shots].index(shot_id)
        return shot_size_gap(shots[k]["shot_size"], shots[k - 1]["shot_size"])

    def judged(name: str, story: str, row: Mapping[str, Any]) -> bool | None:
        return answers[name].get(story, {}).get(row["assertion_id"], {}).get("holds")

    questions = [(story, row) for story in sorted(cases) for row in scripts[story]["assertions"]
                 if row["kind"] == COMPOSITION_KIND]
    judge_questions = [(s, r) for s, r in questions if gap(s, r["check_shot"]) >= COMPOSITION_MIN_SIZE_GAP]
    tool_questions = [(s, r) for s, r in questions if gap(s, r["check_shot"]) == 1]
    both = [(s, r) for s, r in judge_questions if judged("floor1", s, r) is not None and judged("floor2", s, r) is not None]
    copy_targets = targets[ARM_COPY]
    target_rows = []
    for story, row in questions:
        if story not in copy_targets or row["check_shot"] != copy_targets[story]["target"]:
            continue
        size_gap = gap(story, row["check_shot"])
        if size_gap >= COMPOSITION_MIN_SIZE_GAP:
            route, floor, copy = "judge", judged("floor1", story, row), judged(ARM_COPY, story, row)
        elif size_gap == 1:
            route = "head_tool"
            floor = _head_answer(cases, measures["floor"], heads["floor"], story, row["check_shot"])
            copy = _head_answer(cases, measures[ARM_COPY], heads[ARM_COPY], story, row["check_shot"])
        else:
            continue
        target_rows.append({"story_id": story, "shot_id": row["check_shot"], "asked": row["state"], "size_gap": size_gap,
                            "route": route, "floor": floor, "copy": copy})

    def copy_no(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        def answer(row: Mapping[str, Any]) -> bool | None:
            return row["copy"]["holds"] if row["route"] == "head_tool" else row["copy"]
        judgeable = [row for row in rows if answer(row) is not None]
        return {"no": _rate(sum(answer(row) is False for row in judgeable), len(judgeable)),
                "coverage": _rate(len(judgeable), len(rows))}

    tool_floor = [_head_answer(cases, measures["floor"], heads["floor"], s, r["check_shot"]) for s, r in tool_questions]
    return {
        "routing": {"judge_min_gap": COMPOSITION_MIN_SIZE_GAP, "head_step_min_log_ratio": rules.SIZE_STEP_MIN_LOG_RATIO,
                    "questions_asked_then": len(questions), "judge_questions": len(judge_questions),
                    "tool_questions": len(tool_questions),
                    "dropped_same_size": len(questions) - len(judge_questions) - len(tool_questions)},
        "judge_aa_disagreement": _rate(sum(judged("floor1", s, r) != judged("floor2", s, r) for s, r in both), len(both)),
        "judge_floor_yes": {name: _rate(sum(bool(judged(name, s, r)) for s, r in judge_questions),
                                        sum(judged(name, s, r) is not None for s, r in judge_questions))
                            for name in ("floor1", "floor2")},
        "tool_floor_yes": _rate(sum(row["holds"] is True for row in tool_floor), sum(row["holds"] is not None for row in tool_floor)),
        "tool_floor_coverage": _rate(sum(row["holds"] is not None for row in tool_floor), len(tool_floor)),
        "copy_target_no": copy_no(target_rows),
        "copy_target_no_by_route": {route: copy_no([row for row in target_rows if row["route"] == route])
                                    for route in ("judge", "head_tool")},
        "target_rows": target_rows,
    }


def run_summarize_ruler(args: argparse.Namespace) -> int:
    cases = {case["case_id"]: case for case in _read(args.slice)["cases"]}
    targets_payload = _read(args.targets)
    if targets_payload["schema_version"] != TARGETS_SCHEMA:
        raise ValueError(f"{args.targets}: not a {TARGETS_SCHEMA} file")
    targets = targets_payload["targets"]
    measures = {name: _read(path) for name, path in _arm_specs(args.measure).items()}
    if set(measures) != {"floor", *ARMS}:
        raise ValueError(f"--measure must name floor and {list(ARMS)}, got {sorted(measures)}")
    for name, measure in measures.items():
        if set(measure["stories"]) != set(cases):
            raise ValueError(f"{name}: measured stories differ from the slice")
    judges = _arm_specs(args.judge)
    if set(judges) != {"floor1", "floor2", ARM_COPY}:
        raise ValueError(f"--judge must name floor1, floor2 and {ARM_COPY}, got {sorted(judges)}")
    heads = {name: _read(path) for name, path in _arm_specs(args.heads).items()}
    if set(heads) != {"floor", ARM_COPY}:
        raise ValueError(f"--heads must name floor and {ARM_COPY}, got {sorted(heads)}")
    readout = {
        "slice": str(args.slice),
        "targets": str(args.targets),
        "measure": {name: str(path) for name, path in _arm_specs(args.measure).items()},
        "judge": {name: str(path) for name, path in judges.items()},
        "heads": {name: str(path) for name, path in _arm_specs(args.heads).items()},
        "target_counts": {arm: len(targets[arm]) for arm in ARMS},
        "side": _side_readout(cases, measures, targets),
        "layout": _layout_readout(cases, measures, targets),
        "non_target_detection_identity": _non_target_identity(cases, measures, targets),
        "composition": _composition_readout(cases, targets, args.assertions, judges, measures, heads),
    }
    _write_new(args.out, readout)
    summary = {
        "side": {k: v for k, v in readout["side"].items() if k != "rows"},
        "layout_swap": {k: v for k, v in readout["layout"]["arms"][ARM_SWAP].items() if k != "rows"},
        "layout_floor_by_cut_type": readout["layout"]["floor_compatible_by_cut_type"],
        "composition": {k: v for k, v in readout["composition"].items() if k != "target_rows"},
        "non_target_detection_identity": readout["non_target_detection_identity"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


# ---------------------------------------------------------------------------
# calibrate-real-layout (GPU)：地标布局尺子在真实镜界上的校准
# ---------------------------------------------------------------------------


def _real_shot_frames(sample: Mapping[str, Any], shot: str) -> list[Any]:
    """校准样本一镜的三帧（镜首、中、尾，与标注器同一组原片帧号），按原片解码后等比缩到长边 832。"""

    import cv2
    import numpy as np

    from mspc.data.msm_global_frames import default_msm_frame_reader

    images = default_msm_frame_reader(Path(sample["video"][shot]), sample["native_frame_indices"][shot])
    frames = []
    for image in images:
        array = np.asarray(image.convert("RGB"))
        scale = rules.REAL_LAYOUT_LONG_SIDE / max(array.shape[:2])
        size = (round(array.shape[1] * scale), round(array.shape[0] * scale))
        frames.append(np.ascontiguousarray(cv2.resize(array, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)))
    return frames


def run_measure_heads(args: argparse.Namespace) -> int:
    """一条臂每镜中间帧（布局关键帧里的中帧）上检测头框，给构图题相邻一档的工具读。"""

    from mspc.evaluation.renderer_qualification_official_backend import (
        DEFAULT_ROI_DETECTOR_CONFIG,
        FROZEN_PUBLIC_ROI_DETECTOR,
        _decode_rgb,
        _DinoRoiEngine,
        _model_batch_from_roi_config,
    )

    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    cases = {case["case_id"]: case for case in _read(args.benchmark)["cases"]}
    videos = _video_rows(args.videos)
    if set(videos) != set(cases):
        raise ValueError(f"{args.videos} covers different stories from {args.benchmark}")
    batch = _model_batch_from_roi_config(DEFAULT_ROI_DETECTOR_CONFIG)
    engine = _DinoRoiEngine(args.resource_root, args.device)
    stories: dict[str, Any] = {}
    for story_id in sorted(cases):
        frames, _ = _decode_rgb(Path(videos[story_id]))
        start, shots = 0, {}
        for shot in cases[story_id]["shots"]:
            offset = (shot["frame_count"] - 1) // 2
            (regions,) = engine.grounded_regions(
                [frames[start + offset]], [rules.HEAD_PHRASE], FROZEN_PUBLIC_ROI_DETECTOR,
                detector_frame_batch_size=batch.detector_frame_batch_size, dino_crop_batch_size=batch.dino_crop_batch_size,
            )
            shots[shot["shot_id"]] = {"offset": offset, "heads": [{"box": region["box"], "score": region["score"]}
                                                                  for region in regions]}
            start += shot["frame_count"]
        engine.clear_frame_caches()
        stories[story_id] = shots
    _write_new(args.out, {"arm": args.arm, "benchmark": str(args.benchmark), "videos": str(args.videos),
                          "phrase": rules.HEAD_PHRASE, "roi_detector": FROZEN_PUBLIC_ROI_DETECTOR.payload(),
                          "stories": stories})
    print(json.dumps({"arm": args.arm, "stories": len(stories), "out": str(args.out)}))
    return 0


def run_calibrate_real_layout(args: argparse.Namespace) -> int:
    from mspc.evaluation.renderer_qualification_official_backend import (
        DEFAULT_ROI_DETECTOR_CONFIG,
        FROZEN_PUBLIC_ROI_DETECTOR,
        _DinoRoiEngine,
        _model_batch_from_roi_config,
    )
    from tools.annotate.label_cut_types import load_reference

    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    samples = [json.loads(line) for line in args.samples.read_text(encoding="utf-8").splitlines() if line.strip()]
    reference = load_reference(args.reference)
    if set(reference) != {sample["sample_id"] for sample in samples}:
        raise ValueError(f"{args.reference} labels a different sample set from {args.samples}")
    by_id = {sample["sample_id"]: sample for sample in samples}
    pairs = rules.real_layout_pairs(samples, reference)
    shots = sorted({tuple(side) for pair in pairs for side in (pair["left"], pair["right"])})
    batch = _model_batch_from_roi_config(DEFAULT_ROI_DETECTOR_CONFIG)
    engine = _DinoRoiEngine(args.resource_root, args.device)
    started = time.time()
    features: dict[tuple[str, str], list[Any]] = {}
    shot_rows: dict[str, Any] = {}
    for index, (sample_id, shot) in enumerate(shots, start=1):
        frames = _real_shot_frames(by_id[sample_id], shot)
        regions = engine.grounded_regions(
            frames, list(rules.REAL_LAYOUT_PHRASES), FROZEN_PUBLIC_ROI_DETECTOR,
            detector_frame_batch_size=batch.detector_frame_batch_size, dino_crop_batch_size=batch.dino_crop_batch_size,
        )
        rows = []
        for frame, frame_regions in zip(frames, regions, strict=True):
            masks = engine._masks(frame, [region["box"] for region in frame_regions])
            mask = background_mask(masks, frame.shape[:2])
            feature = sift_features(frame, mask)
            features.setdefault((sample_id, shot), []).append(feature)
            rows.append({"boxes": len(frame_regions), "background_fraction": round(float((mask > 0).mean()), 4),
                         "sift_keypoints": len(feature[0])})
        shot_rows[f"{sample_id}:{shot}"] = {"size": [int(frames[0].shape[1]), int(frames[0].shape[0])], "frames": rows}
        engine.clear_frame_caches()
        if index % 50 == 0 or index == len(shots):
            print(f"[real-layout] shots {index}/{len(shots)} {time.time() - started:.0f}s", file=sys.stderr, flush=True)
    measured = []
    for pair in pairs:
        inliers, where = pair_inliers(features[tuple(pair["left"])], features[tuple(pair["right"])])
        measured.append({**pair, "inliers": inliers, "best_keyframes": list(where)})
    readout = rules.real_layout_readout(measured)
    _write_new(args.out, {
        "samples": str(args.samples),
        "reference": str(args.reference),
        "resource_root": str(args.resource_root),
        "roi_detector": FROZEN_PUBLIC_ROI_DETECTOR.payload(),
        "phrases": list(rules.REAL_LAYOUT_PHRASES),
        "long_side": rules.REAL_LAYOUT_LONG_SIDE,
        "elapsed_seconds": round(time.time() - started, 1),
        "shots": shot_rows,
        **readout,
    })
    print(json.dumps({key: readout[key] for key in ("control_median", "control_n", "control_inliers_p90", "by_stratum",
                                                     "by_stratum_sure")}, ensure_ascii=False, indent=1))
    return 0


# ---------------------------------------------------------------------------
# sheet (CPU)
# ---------------------------------------------------------------------------

_COLORS = ((230, 57, 70), (29, 120, 220), (40, 170, 90), (240, 160, 20))


def run_sheet(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    from mspc.evaluation.renderer_qualification_official_backend import _decode_rgb

    measure = _read(args.measure)
    threshold = _min_phrase_score()
    wanted = [line.strip() for line in args.select.split(",") if line.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for selector in wanted:
        story_id, _, shot_filter = selector.partition(":")
        story = measure["stories"][story_id]
        frames, _ = _decode_rgb(Path(story["video_path"]))
        ids = story["subject_ids"]
        rows, start = [], 0
        for shot_id, shot in story["shots"].items():
            if shot_filter and shot_id != shot_filter:
                start += shot["frame_count"]
                continue
            tiles = []
            for frame_row in shot["frames"]:
                image = np.ascontiguousarray(frames[start + frame_row["offset"]][:, :, ::-1])
                owners = subject_boxes(frame_row["regions"], ids, min_phrase_score=threshold)
                for region in frame_row["regions"]:
                    x0, y0, x1, y1 = (int(v) for v in region["box"])
                    cv2.rectangle(image, (x0, y0), (x1, y1), (160, 160, 160), 1)
                for index, subject in enumerate(ids):
                    if owners[subject] is None:
                        continue
                    x0, y0, x1, y1 = (int(v) for v in owners[subject]["box"])
                    color = _COLORS[index % len(_COLORS)][::-1]
                    cv2.rectangle(image, (x0, y0), (x1, y1), color, 3)
                    cv2.putText(image, f"S{subject.split()[-1]} {owners[subject]['phrase_score']:.2f}",
                                (x0 + 4, max(18, y0 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(image, f"{shot_id} f{frame_row['offset']}", (8, image.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                tiles.append(cv2.resize(image, (args.tile_width, int(image.shape[0] * args.tile_width / image.shape[1]))))
            row = np.zeros((tiles[0].shape[0], len(tiles) * args.tile_width, 3), dtype=np.uint8)
            for index, tile in enumerate(tiles):
                row[:, index * args.tile_width:(index + 1) * args.tile_width] = tile
            rows.append(row)
            start += shot["frame_count"]
        width = max(row.shape[1] for row in rows)
        sheet = np.zeros((sum(row.shape[0] for row in rows), width, 3), dtype=np.uint8)
        top = 0
        for row in rows:
            sheet[top:top + row.shape[0], :row.shape[1]] = row
            top += row.shape[0]
        name = selector.replace(":", "_")
        cv2.imwrite(str(args.out_dir / f"{measure['arm']}_{name}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(json.dumps({"sheets": len(wanted), "out_dir": str(args.out_dir)}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    steps = parser.add_subparsers(dest="step", required=True)

    build = steps.add_parser("build-items")
    build.add_argument("--bench", type=Path, required=True, help="configs/benchmark/script_state_bench_v2.json")
    build.add_argument("--probe-out", type=Path, required=True)
    build.add_argument("--slice-out", type=Path, required=True)
    build.add_argument("--slice-benchmark-id", required=True)
    build.add_argument("--side-size", type=int, required=True)
    build.add_argument("--side-seed", required=True)
    build.add_argument("--size-size", type=int, required=True)
    build.add_argument("--size-seed", required=True)
    build.add_argument("--slice-per-shot-count", type=int, required=True)
    build.add_argument("--slice-seed", required=True)
    build.set_defaults(handler=run_build_items)

    measure = steps.add_parser("measure")
    measure.add_argument("--benchmark", type=Path, required=True, help="探针题面或切片")
    measure.add_argument("--videos", type=Path, required=True, help="[{case_id, video_path}]：出片清单或坏例臂清单")
    measure.add_argument("--arm", required=True)
    measure.add_argument("--resource-root", type=Path, required=True, help="…/renderer_qualification/public_metrics_v1")
    measure.add_argument("--device", required=True)
    measure.add_argument("--out", type=Path, required=True)
    measure.set_defaults(handler=run_measure)

    arms = steps.add_parser("make-arms")
    arms.add_argument("--slice", type=Path, required=True)
    arms.add_argument("--floor-manifest", type=Path, required=True, help="地板的 generated_eval_manifest.json")
    arms.add_argument("--floor-arm-id", required=True)
    arms.add_argument("--seed", type=int, required=True)
    arms.add_argument("--fps", type=int, required=True, help="与出片同一帧率（载体配置 inference.fps）")
    arms.add_argument("--video-quality", type=int, required=True, help="与出片同一质量（载体配置 inference.video_quality）")
    arms.add_argument("--out-dir", type=Path, required=True)
    arms.set_defaults(handler=run_make_arms)

    probe = steps.add_parser("summarize-probe")
    probe.add_argument("--probe", type=Path, required=True)
    probe.add_argument("--measure", type=Path, required=True)
    probe.add_argument("--out", type=Path, required=True)
    probe.set_defaults(handler=run_summarize_probe)

    ruler = steps.add_parser("summarize-ruler")
    ruler.add_argument("--slice", type=Path, required=True)
    ruler.add_argument("--targets", type=Path, required=True)
    ruler.add_argument("--measure", nargs="+", required=True, help=f"floor=… {' '.join(f'{arm}=…' for arm in ARMS)}")
    ruler.add_argument("--assertions", type=Path, required=True, help="判官用的那份冻结剧本断言（含构图题）")
    ruler.add_argument("--judge", nargs="+", required=True, help=f"floor1=cases.jsonl floor2=… {ARM_COPY}=…")
    ruler.add_argument("--heads", nargs="+", required=True, help=f"floor=… {ARM_COPY}=…（measure-heads 的产物）")
    ruler.add_argument("--out", type=Path, required=True)
    ruler.set_defaults(handler=run_summarize_ruler)

    head = steps.add_parser("measure-heads")
    head.add_argument("--benchmark", type=Path, required=True, help="切片")
    head.add_argument("--videos", type=Path, required=True, help="[{case_id, video_path}]：出片清单或坏例臂清单")
    head.add_argument("--arm", required=True)
    head.add_argument("--resource-root", type=Path, required=True)
    head.add_argument("--device", required=True)
    head.add_argument("--out", type=Path, required=True)
    head.set_defaults(handler=run_measure_heads)

    real = steps.add_parser("calibrate-real-layout")
    real.add_argument("--samples", type=Path, required=True, help="镜界标注器校准的 samples.jsonl")
    real.add_argument("--reference", type=Path, required=True, help="同一批的参考标签 reference.tsv")
    real.add_argument("--resource-root", type=Path, required=True, help="…/renderer_qualification/public_metrics_v1")
    real.add_argument("--device", required=True)
    real.add_argument("--out", type=Path, required=True)
    real.set_defaults(handler=run_calibrate_real_layout)

    sheet = steps.add_parser("sheet")
    sheet.add_argument("--measure", type=Path, required=True)
    sheet.add_argument("--select", required=True, help="逗号分隔的 story 或 story:shot")
    sheet.add_argument("--tile-width", type=int, required=True)
    sheet.add_argument("--out-dir", type=Path, required=True)
    sheet.set_defaults(handler=run_sheet)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
