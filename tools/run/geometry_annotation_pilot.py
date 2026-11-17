#!/usr/bin/env python3
"""几何标注工具试点的薄入口。
几何与判据的纯函数在 :mod:`mspc.data.geometry_annotation`。

子命令按执行面分，各用自己的环境（命令行里给解释器）：

* ``frames``（CPU，wan22_fun 环境，decord）：v6 区间 16 fps 训练帧里每镜取 ≤48 帧跟踪片段（去首尾各 2 帧），
  自动裁黑边、长边缩到 640 存 JPEG；读 v8 标注（同地点、人物描述）。写 ``plan.json``。
* ``gpu``（GPU，sam3_sm80 环境，PYTHONPATH 加 ``.runtime/upstreams/{vggt,co-tracker}``）：每镜 SAM 3.1
  （``person`` 概念给实例并跟踪，人物描述概念给逐主体得分）、VGGT 单镜 8 帧、CoTracker3 片段跟踪与重投影核验；
  每对相邻镜 VGGT 16 帧联合前向、背景 SIFT 核验、人物核验、轴线侧。每窗写 ``gpu/<window>.json`` 与 ``.npz``。
* ``moge``（GPU，moge2_cu128 环境）：每镜中帧 MoGe-2 深度，写 ``moge/<window>.npz``。
* ``analyze``（CPU）：汇总问题 1–4 的读数，抽 20 对亲看，写 ``summary.json``、``pairs.jsonl``、``shots.jsonl``。
* ``sheets``（CPU）：亲看 20 对的拼图与失败镜头中帧总览。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
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

from mspc.data import geometry_annotation as geo  # noqa: E402
from mspc.data.spatial_facts import assign_one_to_one, short_phrase  # noqa: E402

PLAN_SCHEMA = "geometry_annotation_pilot_plan_v1"
GPU_SCHEMA = "geometry_annotation_pilot_gpu_v1"
LONG_SIDE = 640
VGGT_LONG_SIDE = 518
PERSON_CONCEPT = "person"
PERSON_MIN_SCORE = 0.5
DESCRIPTION_MATCH_IOU = 0.5
TRACK_GRID = 12
SIFT_RATIO = 0.75
FUNDAMENTAL_THRESHOLD_PX = 3.0
REVIEW_SEED = 20261010
REVIEW_COUNT = 20


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_new(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _round(value: Any, digits: int = 4) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_round(v, digits) for v in value]
    value = float(value)
    return None if not math.isfinite(value) else round(value, digits)


# ---------------------------------------------------------------------------
# frames (CPU, wan22_fun)
# ---------------------------------------------------------------------------


def _annotation(attempts: Path, v8_window: str, shot_count: int) -> dict[str, Any] | None:
    path = attempts / v8_window / "candidate.json"
    if not path.exists():
        return None
    cleaned = _read(path)["cleaned"]
    if cleaned is None or len(cleaned["shots"]) != shot_count:
        return None
    subjects = {row["id"]: {"kind": row["kind"], "appearance": row["appearance"]} for row in cleaned["subjects"]}
    shots = []
    for shot in cleaned["shots"]:
        visible = list(dict.fromkeys(list(shot.get("visible_first") or []) + list(shot.get("visible_last") or [])))
        shots.append({"place": shot.get("place"), "persons": [s for s in visible if subjects.get(s, {}).get("kind") == "person"],
                      "camera": shot.get("camera"),
                      "screen": {e["id"]: e.get("screen") for e in shot.get("entities") or [] if e.get("screen")}})
    return {"subjects": subjects, "shots": shots}


def _frames_window(job: tuple) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    from mspc.data.frames import default_frame_reader

    window, row, annotation, out = job
    started = time.perf_counter()
    plan = row["plan"]
    dataset = "ci_vid" if row["case_id"].startswith("ci_vid") else "cine250k"
    short = [len(indices) for indices in plan["frame_indices_by_shot"] if len(indices) < geo.VGGT_FRAMES]
    if short:
        return {"window_id": window, "case_id": row["case_id"], "dataset": dataset, "shots": [], "annotation": annotation,
                "skip": f"shot shorter than {geo.VGGT_FRAMES} frames: {short}", "decode_seconds": 0.0}
    shots = []
    for k, (indices, video) in enumerate(zip(plan["frame_indices_by_shot"], plan["source_video_paths"], strict=True)):
        positions = geo.clip_positions(len(indices))
        sources = [indices[p] for p in positions]
        images = default_frame_reader(Path(video), sources)
        arrays = np.stack([np.asarray(image) for image in images])
        vggt = geo.vggt_positions(len(sources))
        x0, y0, x1, y1 = geo.letterbox_box(arrays[vggt])
        width, height = x1 - x0, y1 - y0
        scale = LONG_SIDE / max(width, height)
        size = (int(round(width * scale / 2)) * 2, int(round(height * scale / 2)) * 2)
        paths = []
        for c, image in enumerate(images):
            path = out / "frames" / window / f"s{k}" / f"c{c:02d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.crop((x0, y0, x1, y1)).resize(size, Image.Resampling.LANCZOS).save(path, format="JPEG", quality=92)
            paths.append(str(path))
        shots.append({"shot": k, "train_frames": len(indices), "clip_source_indices": sources, "clip_paths": paths,
                      "vggt_positions": vggt, "crop_box": [x0, y0, x1, y1], "source_size": [int(arrays.shape[2]), int(arrays.shape[1])],
                      "size": list(size), "video": video})
    return {"window_id": window, "case_id": row["case_id"], "dataset": dataset, "shots": shots, "annotation": annotation,
            "skip": None, "decode_seconds": round(time.perf_counter() - started, 2)}


def run_frames(args: argparse.Namespace) -> int:
    from concurrent.futures import ProcessPoolExecutor

    out = args.out_dir
    if (out / "plan.json").exists():
        raise FileExistsError(out / "plan.json")
    rows = {}
    with open(args.rows, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["case_id"]] = row
    mapping = _read(args.window_map)["windows"]
    if args.limit:
        mapping = mapping[: args.limit]
    jobs = []
    for entry in mapping:
        row = rows[entry["case_id"]]
        shot_count = len(row["plan"]["frame_indices_by_shot"])
        jobs.append((entry["window_id"], row, _annotation(args.attempts, entry["v8_window_id"], shot_count), out))
    started = time.perf_counter()
    windows = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(_frames_window, jobs):
            windows[result["window_id"]] = result
            print(f"[frames] {result['window_id']} {len(result['shots'])} shots {result['decode_seconds']}s",
                  file=sys.stderr, flush=True)
    _write_new(out / "plan.json", {
        "schema_version": PLAN_SCHEMA, "created_at": _now(), "rows": str(args.rows), "window_map": str(args.window_map),
        "attempts": str(args.attempts), "workers": args.workers, "elapsed_seconds": round(time.perf_counter() - started, 1),
        "long_side": LONG_SIDE, "windows": windows,
    })
    print(json.dumps({"windows": len(windows), "shots": sum(len(w["shots"]) for w in windows.values()),
                      "annotated": sum(w["annotation"] is not None for w in windows.values()),
                      "skipped": {n: w["skip"] for n, w in windows.items() if w["skip"]}}))
    return 0


# ---------------------------------------------------------------------------
# gpu (sam3_sm80 + vggt + co-tracker)
# ---------------------------------------------------------------------------


def _unpack(packed: Any, shape: Any) -> Any:
    import numpy as np

    count = int(shape[0]) * int(shape[1])
    return np.unpackbits(packed)[:count].reshape(int(shape[0]), int(shape[1])).astype(bool)


def _iou(a: Any, b: Any) -> float:
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else 0.0


def _sam_session(model: Any, paths: Sequence[str], text: str, scratch: Path) -> list[list[dict[str, Any]]]:
    import numpy as np

    scratch.mkdir(parents=True, exist_ok=False)
    try:
        for index, path in enumerate(paths):
            os.symlink(path, scratch / f"{index:05d}.jpg")
        mask_path = scratch / "masks.npz"
        frames = model.track_frame_directory(frame_dir=str(scratch), text=text, mask_path=str(mask_path),
                                             hotstart_delay=0, hotstart_unmatch_thresh=0)
        if sorted(row["frame_index"] for row in frames) != list(range(len(paths))):
            raise ValueError(f"SAM 3.1 did not return every frame exactly once for {text!r}")
        packed = np.load(mask_path)
        out: list[list[dict[str, Any]]] = [[] for _ in paths]
        for row in frames:
            for det in row["detections"]:
                out[row["frame_index"]].append({"track_id": det["track_id"], "score": float(det["score"]),
                                                "mask": _unpack(packed[det["mask_key"]], packed[det["mask_key"] + "_shape"])})
        return out
    finally:
        for child in scratch.iterdir():
            child.unlink()
        scratch.rmdir()


def _shot_persons(model: Any, paths: Sequence[str], subjects: Mapping[str, Mapping[str, Any]], persons: Sequence[str],
                  scratch: Path) -> dict[str, Any]:
    """一镜 8 帧的人物实例（按 SAM 跟踪号）与身份绑定。"""

    generic = _sam_session(model, paths, PERSON_CONCEPT, scratch / "person")
    frames = [[d for d in dets if d["score"] >= PERSON_MIN_SCORE] for dets in generic]
    tracks = sorted({d["track_id"] for dets in frames for d in dets})
    scores: dict[int, dict[str, float]] = {t: {} for t in tracks}
    for subject in persons:
        phrase = short_phrase(subjects[subject]["appearance"], "person")
        described = _sam_session(model, paths, phrase, scratch / f"d{subject.split()[-1]}")
        for f, dets in enumerate(described):
            for det in dets:
                best = max(((_iou(det["mask"], inst["mask"]), inst["track_id"]) for inst in frames[f]), default=(0.0, None))
                if best[0] >= DESCRIPTION_MATCH_IOU:
                    scores[best[1]][subject] = max(scores[best[1]].get(subject, 0.0), det["score"])
    gain = {(subject, track): score for track, row in scores.items() for subject, score in row.items()}
    binding = assign_one_to_one(gain, min_gain=geo.BIND_MIN_SCORE)
    return {"frames": frames, "tracks": tracks, "scores": scores, "binding": binding}


def _resize_mask(mask: Any, size: tuple[int, int]) -> Any:
    import numpy as np
    from PIL import Image

    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)) > 127


def _vggt_images(paths: Sequence[str], size: tuple[int, int]) -> Any:
    import numpy as np
    import torch
    from PIL import Image

    arrays = [np.asarray(Image.open(p).convert("RGB").resize(size, Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
              for p in paths]
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)


def _vggt_size(size: Sequence[int]) -> tuple[int, int]:
    width, height = size
    if width >= height:
        return VGGT_LONG_SIDE, max(14, int(round(height * VGGT_LONG_SIDE / width / 14)) * 14)
    return max(14, int(round(width * VGGT_LONG_SIDE / height / 14)) * 14), VGGT_LONG_SIDE


def _vggt_run(model: Any, paths: Sequence[str], size: tuple[int, int]) -> dict[str, Any]:
    import torch
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    images = _vggt_images(paths, size).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictions = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    return {"extrinsic": extrinsic[0].float().cpu().numpy(), "intrinsic": intrinsic[0].float().cpu().numpy(),
            "depth": predictions["depth"][0, ..., 0].float().cpu().numpy(),
            "conf": predictions["depth_conf"][0].float().cpu().numpy()}


def _person_point(mask: Any, depth: Any, intrinsic: Any, extrinsic: Any) -> dict[str, Any] | None:
    """人物掩码（640 帧）→ VGGT 深度图上的中位深度与掩码中位像素 → 世界点。"""

    import numpy as np

    height, width = depth.shape
    scale = (width / mask.shape[1], height / mask.shape[0])
    small = _resize_mask(mask, (width, height))
    if small.sum() < 20:
        return None
    rows, cols = np.nonzero(small)
    value = float(np.median(depth[rows, cols]))
    if not value > 0:
        return None
    pixel = np.array([[float(np.median(cols)), float(np.median(rows))]])
    world = geo.backproject(pixel, np.array([value]), intrinsic, extrinsic)[0]
    return {"world": world, "depth": value, "pixel_640": [float(np.median(cols)) / scale[0], float(np.median(rows)) / scale[1]],
            "area": float(mask.mean())}


def _subject_frames(persons: Mapping[str, Any]) -> dict[str, list[int]]:
    """每个已绑定主体在 8 帧里出现的帧号。"""

    frames = {}
    for subject, track in persons["binding"].items():
        frames[subject] = [f for f, dets in enumerate(persons["frames"]) if any(d["track_id"] == track for d in dets)]
    return frames


def _pick_frame(candidates: Sequence[int]) -> int | None:
    if not candidates:
        return None
    return min(candidates, key=lambda f: (abs(f - geo.MID_POSITION), f))


def _subject_mask(persons: Mapping[str, Any], subject: str, frame: int) -> Any:
    track = persons["binding"][subject]
    return next(d["mask"] for d in persons["frames"][frame] if d["track_id"] == track)


def _track_queries(vggt: Sequence[int], size: Sequence[int]) -> Any:
    """CoTracker 查询点：前 7 个 VGGT 帧上各一张 ``TRACK_GRID``×``TRACK_GRID`` 网格（四周留 4% 边）。返回 (N, 3) 的 (t, x, y)。"""

    import numpy as np

    width, height = size
    xs = np.linspace(0.04 * width, 0.96 * width, TRACK_GRID)
    ys = np.linspace(0.04 * height, 0.96 * height, TRACK_GRID)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    rows = [np.concatenate([np.full((len(grid), 1), float(t)), grid], axis=1) for t in vggt[:-1]]
    return np.concatenate(rows).astype(np.float32)


def _track_check(tracks: Any, visible: Any, vggt: Sequence[int], run: Mapping[str, Any], persons: Mapping[str, Any],
                 size: Sequence[int], vsize: tuple[int, int]) -> dict[str, Any]:
    """CoTracker 背景轨迹核验 VGGT 单镜相机：相邻 VGGT 帧 j→j+1，取在 j 帧查询、j+1 帧可见、两帧都在人物掩码外的点，
    用 j 帧 VGGT 深度和两帧相机重投影。返回误差中位数 e、机位运动量 m（均占画宽）、背景点数 n（各帧对点数的中位数）。"""

    import numpy as np

    width, height = size
    sx, sy = vsize[0] / width, vsize[1] / height
    per_query = TRACK_GRID * TRACK_GRID
    union = [np.zeros((height, width), dtype=bool) for _ in vggt]
    for f, dets in enumerate(persons["frames"]):
        for det in dets:
            union[f] |= det["mask"]
    errors, moves, counts = [], [], []
    for j in range(len(vggt) - 1):
        sel = slice(j * per_query, (j + 1) * per_query)
        start = tracks[vggt[j], sel]
        now = tracks[vggt[j + 1], sel]
        keep = visible[vggt[j + 1], sel].copy()
        keep &= (now[:, 0] >= 0) & (now[:, 0] <= width - 1) & (now[:, 1] >= 0) & (now[:, 1] <= height - 1)
        at0 = np.clip(np.rint(start).astype(int), [0, 0], [width - 1, height - 1])
        at1 = np.clip(np.rint(now).astype(int), [0, 0], [width - 1, height - 1])
        keep &= ~union[j][at0[:, 1], at0[:, 0]] & ~union[j + 1][at1[:, 1], at1[:, 0]]
        counts.append(int(keep.sum()))
        if not keep.any():
            continue
        a = start[keep] * [sx, sy]
        b = now[keep] * [sx, sy]
        depth = geo.sample_map(run["depth"][j], np.clip(a, 0, [vsize[0] - 1, vsize[1] - 1]))
        err = geo.reprojection_errors(a, depth, (run["intrinsic"][j], run["extrinsic"][j]), b,
                                      (run["intrinsic"][j + 1], run["extrinsic"][j + 1])) / vsize[0]
        errors.append(err)
        moves.append(np.linalg.norm(now[keep] - start[keep], axis=1) / width)
    count = int(np.median(counts)) if counts else 0
    error = float(np.median(np.concatenate(errors))) if errors else None
    move = float(np.median(np.concatenate(moves))) if moves else None
    p90 = float(np.percentile(np.concatenate(errors), 90)) if errors else None
    if count < geo.TRACK_MIN_POINTS or error is None:
        verdict = "unverifiable"
    else:
        verdict = "pass" if error <= geo.TRACK_PASS_ERROR else "fail"
    return {"background_points": count, "pair_counts": counts, "reprojection_error": _round(error),
            "reprojection_error_p90": _round(p90), "camera_motion": _round(move), "verdict": verdict}


def _static_overlay(window: Mapping[str, Any]) -> dict[int, Any]:
    """窗内所有镜的 VGGT 帧逐像素灰度标准差 ≤3 视为静止叠加层（台标、字幕底），膨胀后按镜尺寸返回掩码。"""

    import cv2
    import numpy as np

    first = window["shots"][0]["size"]
    small = (first[0] // 2, first[1] // 2)
    stack = []
    for shot in window["shots"]:
        for p in shot["vggt_positions"]:
            grey = cv2.imread(shot["clip_paths"][p], cv2.IMREAD_GRAYSCALE)
            stack.append(cv2.resize(grey, small, interpolation=cv2.INTER_AREA).astype(np.float32))
    static = (np.stack(stack).std(axis=0) <= 3.0).astype(np.uint8)
    static = cv2.dilate(static, np.ones((5, 5), np.uint8))
    return {shot["shot"]: cv2.resize(static, tuple(shot["size"]), interpolation=cv2.INTER_NEAREST).astype(bool)
            for shot in window["shots"]}


def _sift_inliers(path_a: str, path_b: str, mask_a: Any, mask_b: Any) -> tuple[Any, Any]:
    import cv2
    import numpy as np

    sift = cv2.SIFT_create(nfeatures=4000)
    grey_a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
    grey_b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((9, 9), np.uint8)
    keep_a = (cv2.dilate(mask_a.astype(np.uint8), kernel) == 0).astype(np.uint8) * 255
    keep_b = (cv2.dilate(mask_b.astype(np.uint8), kernel) == 0).astype(np.uint8) * 255
    kp_a, des_a = sift.detectAndCompute(grey_a, keep_a)
    kp_b, des_b = sift.detectAndCompute(grey_b, keep_b)
    if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return np.zeros((0, 2)), np.zeros((0, 2))
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a, des_b, k=2)
    good = [m for m, n in (pair for pair in matches if len(pair) == 2) if m.distance < SIFT_RATIO * n.distance]
    if len(good) < 8:
        return np.zeros((0, 2)), np.zeros((0, 2))
    a = np.float32([kp_a[m.queryIdx].pt for m in good])
    b = np.float32([kp_b[m.trainIdx].pt for m in good])
    _, inlier = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, FUNDAMENTAL_THRESHOLD_PX, 0.999)
    if inlier is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    inlier = inlier.ravel().astype(bool)
    return a[inlier].astype(np.float64), b[inlier].astype(np.float64)


def _shot_geometry(run: Mapping[str, Any], persons: Mapping[str, Any], offset: int = 0) -> dict[str, dict[str, Any]]:
    """已绑定主体在一次 VGGT 前向里的世界点（取最近中帧的出现帧）。``offset`` 是该镜帧在前向里的起始位置。"""

    out = {}
    for subject, frames in _subject_frames(persons).items():
        f = _pick_frame(frames)
        if f is None:
            continue
        point = _person_point(_subject_mask(persons, subject, f), run["depth"][offset + f], run["intrinsic"][offset + f],
                              run["extrinsic"][offset + f])
        if point is not None:
            point["frame"] = f
            out[subject] = point
    return out


def _two_person_frame(persons: Mapping[str, Any], first: str, second: str) -> int | None:
    frames = _subject_frames(persons)
    both = sorted(set(frames.get(first, [])) & set(frames.get(second, [])))
    return _pick_frame(both)


def _order_2d(persons: Mapping[str, Any], first: str, second: str, width: int) -> dict[str, Any] | None:
    import numpy as np

    f = _two_person_frame(persons, first, second)
    if f is None:
        return None
    xs = []
    for subject in (first, second):
        cols = np.nonzero(_subject_mask(persons, subject, f))[1]
        xs.append(float(np.median(cols)))
    return {"frame": f, "x": xs, "order": geo.screen_order(xs[0], xs[1], width)}


def _axis_frame_shot(run: Mapping[str, Any], persons: Mapping[str, Any], first: str, second: str) -> dict[str, Any] | None:
    """退路：一镜单独的 VGGT 前向里，两人都在同一帧时，机位在人物连线坐标系里的位置。"""

    import numpy as np

    f = _two_person_frame(persons, first, second)
    if f is None:
        return None
    points = []
    for subject in (first, second):
        point = _person_point(_subject_mask(persons, subject, f), run["depth"][f], run["intrinsic"][f], run["extrinsic"][f])
        if point is None:
            return None
        points.append(point["world"])
    extrinsic = run["extrinsic"][f]
    try:
        frame = geo.axis_frame_camera(points[0], points[1], geo.camera_center(extrinsic), geo.camera_forward(extrinsic),
                                      geo.camera_up(extrinsic))
    except ValueError:
        return None
    frame["frame"] = f
    frame["side"] = geo.side_label(frame["side_sine"])
    return {k: (_round(v) if isinstance(v, float) else v) for k, v in frame.items()}


def _pair_geometry(window: Mapping[str, Any], k: int, runs: Mapping[str, Any], persons: Sequence[Mapping[str, Any]],
                   joint: Mapping[str, Any], vsize: tuple[int, int], overlay: Mapping[int, Any]) -> dict[str, Any]:
    import numpy as np

    shot_a, shot_b = window["shots"][k], window["shots"][k + 1]
    scale_a = (vsize[0] / shot_a["size"][0], vsize[1] / shot_a["size"][1])
    scale_b = (vsize[0] / shot_b["size"][0], vsize[1] / shot_b["size"][1])
    pa, pb = persons[k], persons[k + 1]
    mid_a = shot_a["clip_paths"][shot_a["vggt_positions"][geo.MID_POSITION]]
    mid_b = shot_b["clip_paths"][shot_b["vggt_positions"][geo.MID_POSITION]]
    union_a = np.zeros((shot_a["size"][1], shot_a["size"][0]), dtype=bool)
    union_b = np.zeros((shot_b["size"][1], shot_b["size"][0]), dtype=bool)
    for det in pa["frames"][geo.MID_POSITION]:
        union_a |= det["mask"]
    for det in pb["frames"][geo.MID_POSITION]:
        union_b |= det["mask"]
    union_a |= overlay[k]
    union_b |= overlay[k + 1]
    a, b = _sift_inliers(mid_a, mid_b, union_a, union_b)
    m = geo.MID_POSITION
    background = {"inliers": int(len(a)), "reprojection_error": None, "verdict": "unverifiable"}
    if len(a) >= geo.REGISTER_MIN_INLIERS:
        av = a * scale_a
        bv = b * scale_b
        depth = geo.sample_map(joint["depth"][m], np.clip(av, 0, [vsize[0] - 1, vsize[1] - 1]))
        err = geo.reprojection_errors(av, depth, (joint["intrinsic"][m], joint["extrinsic"][m]), bv,
                                      (joint["intrinsic"][8 + m], joint["extrinsic"][8 + m])) / vsize[0]
        background["reprojection_error"] = _round(float(np.median(err)))
        background["verdict"] = "pass" if float(np.median(err)) <= geo.REGISTER_PASS_ERROR else "contradicted"
    points_a = _shot_geometry(joint, pa, 0)
    points_b = _shot_geometry(joint, pb, 8)
    center_a = geo.camera_center(joint["extrinsic"][m])
    center_b = geo.camera_center(joint["extrinsic"][8 + m])
    person_rows = []
    for subject in sorted(set(points_a) & set(points_b)):
        gap = float(np.linalg.norm(points_a[subject]["world"] - points_b[subject]["world"]))
        reach = float(np.linalg.norm(points_a[subject]["world"] - center_a))
        person_rows.append({"subject": subject, "ratio": _round(gap / reach) if reach > 0 else None})
    ratios = [r["ratio"] for r in person_rows if r["ratio"] is not None]
    person = {"subjects": person_rows, "verdict": "unverifiable" if not ratios else
              ("pass" if float(np.median(ratios)) <= geo.PERSON_PASS_RATIO else "contradicted")}
    if background["verdict"] == "pass":
        registration = "background_pass"
    elif background["verdict"] == "contradicted":
        registration = "background_contradicted"
    elif person["verdict"] == "pass":
        registration = "person_pass"
    elif person["verdict"] == "contradicted":
        registration = "person_contradicted"
    else:
        registration = "unverifiable"
    up = geo.camera_up(joint["extrinsic"][m]) + geo.camera_up(joint["extrinsic"][8 + m])
    up = up / np.linalg.norm(up)
    bound = sorted(set(pa["binding"]) | set(pb["binding"]))
    world = {}
    for subject in bound:
        rows = [p[subject]["world"] for p in (points_a, points_b) if subject in p]
        if rows:
            world[subject] = np.mean(rows, axis=0)
    axes = []
    for i, first in enumerate(bound):
        for second in bound[i + 1:]:
            row: dict[str, Any] = {"first": first, "second": second}
            row["order_a"] = _order_2d(pa, first, second, shot_a["size"][0])
            row["order_b"] = _order_2d(pb, first, second, shot_b["size"][0])
            if first in world and second in world:
                sines = [geo.axis_side(world[first], world[second], c, up) for c in (center_a, center_b)]
                row["side_sine_3d"] = _round(sines)
                row["side_3d"] = [geo.side_label(s) for s in sines]
            else:
                row["side_sine_3d"] = None
                row["side_3d"] = None
            row["axis_frame_a"] = _axis_frame_shot(runs[k], pa, first, second)
            row["axis_frame_b"] = _axis_frame_shot(runs[k + 1], pb, first, second)
            axes.append(row)
    rotation = geo.rotation_angle_deg(joint["extrinsic"][m][:3, :3], joint["extrinsic"][8 + m][:3, :3])
    return {"pair": [k, k + 1], "background": background, "person": person, "registration": registration,
            "relative_rotation_deg": _round(rotation, 2),
            "cameras": {"a": {"center": _round(center_a.tolist()), "forward": _round(geo.camera_forward(joint["extrinsic"][m]).tolist())},
                        "b": {"center": _round(center_b.tolist()), "forward": _round(geo.camera_forward(joint["extrinsic"][8 + m]).tolist())},
                        "up": _round(up.tolist())},
            "persons_world": {s: _round(v.tolist()) for s, v in world.items()},
            "persons_a": {s: _round(p["world"].tolist()) for s, p in points_a.items()},
            "persons_b": {s: _round(p["world"].tolist()) for s, p in points_b.items()},
            "axes": axes}


def run_gpu(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    from cotracker.predictor import CoTrackerPredictor
    from vggt.models.vggt import VGGT

    from mspc.data.grounding.sam3_backend import Sam3Backend

    plan = _read(args.plan)
    out = args.out_dir
    (out / "gpu").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    started = time.perf_counter()
    sam = Sam3Backend(checkpoint_path=str(args.sam_checkpoint), bpe_path=str(args.sam_bpe), version="sam3.1",
                      objects_per_forward=16, enable_object_multiplexing=True, use_fa3=False, use_rope_real=True)
    vggt = VGGT()
    vggt.load_state_dict(torch.load(args.vggt_weights, map_location="cpu"))
    vggt = vggt.eval().cuda()
    tracker = CoTrackerPredictor(checkpoint=str(args.cotracker_weights), offline=True, window_len=60).cuda().eval()
    load_seconds = time.perf_counter() - started
    print(f"[gpu] models loaded {load_seconds:.0f}s", file=sys.stderr, flush=True)
    names = sorted(n for n, w in plan["windows"].items() if w["skip"] is None)
    if args.windows:
        names = [n for n in names if n in set(args.windows.split(","))]
    names = names[args.shard::args.num_shards]
    scratch_root = args.scratch / f"geom-{os.getpid()}"
    for name in names:
        target = out / "gpu" / f"{name}.json"
        if target.exists():
            continue
        window = plan["windows"][name]
        annotation = window["annotation"]
        timing = {"sam3": 0.0, "vggt_shot": 0.0, "cotracker": 0.0, "vggt_pair": 0.0, "check": 0.0}
        tick = time.perf_counter()
        record: dict[str, Any] = {"schema_version": GPU_SCHEMA, "window_id": name, "case_id": window["case_id"],
                                  "created_at": _now(), "gpu": torch.cuda.get_device_name(0)}
        try:
            persons, runs, shots_out, arrays = [], [], [], {}
            for shot in window["shots"]:
                k = shot["shot"]
                vggt_paths = [shot["clip_paths"][p] for p in shot["vggt_positions"]]
                subjects = annotation["subjects"] if annotation else {}
                visible = annotation["shots"][k]["persons"] if annotation else []
                t0 = time.perf_counter()
                found = _shot_persons(sam, vggt_paths, subjects, visible, scratch_root / name / f"s{k}")
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                vsize = _vggt_size(shot["size"])
                run = _vggt_run(vggt, vggt_paths, vsize)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                from PIL import Image

                clip = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in shot["clip_paths"]])
                video = torch.from_numpy(clip).permute(0, 3, 1, 2)[None].float().cuda()
                queries = torch.from_numpy(_track_queries(shot["vggt_positions"], shot["size"]))[None].cuda()
                with torch.no_grad():
                    tracks, visible_tracks = tracker(video, queries=queries)
                torch.cuda.synchronize()
                t3 = time.perf_counter()
                tracks = tracks[0].float().cpu().numpy()
                visible_tracks = visible_tracks[0].cpu().numpy().astype(bool)
                check = _track_check(tracks, visible_tracks, shot["vggt_positions"], run, found, shot["size"], vsize)
                own = _shot_geometry(run, found)
                rotation = geo.rotation_angle_deg(run["extrinsic"][0][:3, :3], run["extrinsic"][-1][:3, :3])
                centers = [geo.camera_center(e) for e in run["extrinsic"]]
                median_depth = float(np.median(run["depth"][0]))
                translation = float(np.linalg.norm(centers[-1] - centers[0])) / median_depth if median_depth > 0 else None
                areas = [float(sum(d["mask"].mean() for d in dets)) for dets in found["frames"]]
                timing["sam3"] += t1 - t0
                timing["vggt_shot"] += t2 - t1
                timing["cotracker"] += t3 - t2
                timing["check"] += time.perf_counter() - t3
                persons.append(found)
                runs.append(run)
                m = geo.MID_POSITION
                arrays[f"s{k}_depth"] = run["depth"][m].astype(np.float16)
                arrays[f"s{k}_conf"] = run["conf"][m].astype(np.float16)
                arrays[f"s{k}_extrinsic"] = run["extrinsic"]
                arrays[f"s{k}_intrinsic"] = run["intrinsic"]
                for det in found["frames"][m]:
                    arrays[f"s{k}_mid_t{det['track_id']}"] = np.packbits(det["mask"])
                shots_out.append({
                    "shot": k, "vggt_size": list(vsize), "track_check": check,
                    "rotation_first_last_deg": _round(rotation, 2), "translation_over_depth": _round(translation),
                    "rotation_jitter_deg": _round(geo.rotation_jitter_deg(run["extrinsic"]), 3),
                    "person_area_max": _round(max(areas)), "person_tracks": found["tracks"],
                    "mid_tracks": [d["track_id"] for d in found["frames"][m]],
                    "binding": found["binding"],
                    "binding_scores": {str(t): {k2: _round(v) for k2, v in s.items()} for t, s in found["scores"].items()},
                    "persons_visible_annotation": visible,
                    "subjects_camera": {s: {"depth": _round(p["depth"]), "pixel_640": _round(p["pixel_640"], 1), "frame": p["frame"],
                                            "area": _round(p["area"])} for s, p in own.items()},
                    "mid_luminance": _round(float(np.asarray(Image.open(vggt_paths[m]).convert("L")).mean()), 1),
                })
            pairs = []
            overlay = _static_overlay(window) if len(window["shots"]) > 1 else {}
            record["overlay_fraction"] = _round(float(np.mean([m.mean() for m in overlay.values()]))) if overlay else None
            for k in range(len(window["shots"]) - 1):
                shot_a, shot_b = window["shots"][k], window["shots"][k + 1]
                vsize = _vggt_size(shot_a["size"])
                paths = [shot_a["clip_paths"][p] for p in shot_a["vggt_positions"]] + \
                        [shot_b["clip_paths"][p] for p in shot_b["vggt_positions"]]
                t0 = time.perf_counter()
                joint = _vggt_run(vggt, paths, vsize)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                row = _pair_geometry(window, k, runs, persons, joint, vsize, overlay)
                row["vggt_size"] = list(vsize)
                row["same_place"] = None if not annotation else (
                    annotation["shots"][k]["place"] is not None and annotation["shots"][k]["place"] == annotation["shots"][k + 1]["place"])
                timing["vggt_pair"] += t1 - t0
                timing["check"] += time.perf_counter() - t1
                arrays[f"p{k}_extrinsic"] = joint["extrinsic"]
                arrays[f"p{k}_intrinsic"] = joint["intrinsic"]
                arrays[f"p{k}_depth_a"] = joint["depth"][geo.MID_POSITION].astype(np.float16)
                arrays[f"p{k}_depth_b"] = joint["depth"][8 + geo.MID_POSITION].astype(np.float16)
                pairs.append(row)
            np.savez_compressed(out / "gpu" / f"{name}.npz", **arrays)
            record.update({"status": "ok", "shots": shots_out, "pairs": pairs})
        except Exception as error:  # 记录到窗级产物里计数，不吞掉：analyze 把 error 窗单列
            record.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
            print(f"[gpu] {name} ERROR {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        record["timing_seconds"] = {k2: round(v, 3) for k2, v in timing.items()}
        record["window_seconds"] = round(time.perf_counter() - tick, 3)
        _write_new(target, record)
        print(f"[gpu] {name} {record['status']} {record['window_seconds']:.1f}s {record['timing_seconds']}", file=sys.stderr, flush=True)
    print(json.dumps({"windows": len(names), "load_seconds": round(load_seconds, 1),
                      "elapsed_seconds": round(time.perf_counter() - started, 1)}))
    return 0


# ---------------------------------------------------------------------------
# moge (moge2_cu128)
# ---------------------------------------------------------------------------


def run_moge(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    from moge.model.v2 import MoGeModel
    from PIL import Image

    plan = _read(args.plan)
    (args.out_dir / "moge").mkdir(parents=True, exist_ok=True)
    model = MoGeModel.from_pretrained(str(args.model)).cuda().eval()
    started = time.perf_counter()
    timing = {}
    for name, window in sorted(plan["windows"].items()):
        if window["skip"] is not None:
            continue
        target = args.out_dir / "moge" / f"{name}.npz"
        if target.exists():
            continue
        tick = time.perf_counter()
        arrays = {}
        for shot in window["shots"]:
            path = shot["clip_paths"][shot["vggt_positions"][geo.MID_POSITION]]
            image = np.asarray(Image.open(path).convert("RGB"))
            tensor = torch.tensor(image / 255.0, dtype=torch.float32, device="cuda").permute(2, 0, 1)
            with torch.no_grad():
                output = model.infer(tensor)
            depth = output["depth"].float().cpu().numpy()
            valid = output["mask"].cpu().numpy().astype(bool) & np.isfinite(depth)
            arrays[f"s{shot['shot']}_depth"] = np.where(valid, depth, 0).astype(np.float16)
        torch.cuda.synchronize()
        np.savez_compressed(target, **arrays)
        timing[name] = round(time.perf_counter() - tick, 3)
        print(f"[moge] {name} {timing[name]}s", file=sys.stderr, flush=True)
    stamp = args.out_dir / "moge" / f"timing-{int(time.time())}.json"
    _write_new(stamp, {"model": str(args.model), "created_at": _now(), "seconds": timing,
                       "elapsed_seconds": round(time.perf_counter() - started, 1)})
    return 0


# ---------------------------------------------------------------------------
# analyze (CPU)
# ---------------------------------------------------------------------------


def _depth_agreement(gpu_npz: Any, moge_npz: Any, k: int, size: Sequence[int], binding: Mapping[str, int],
                     subjects_camera: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    vdepth = gpu_npz[f"s{k}_depth"].astype(np.float32)
    vconf = gpu_npz[f"s{k}_conf"].astype(np.float32)
    mdepth = moge_npz[f"s{k}_depth"].astype(np.float32)
    width, height = size
    xs, ys = np.meshgrid(np.linspace(0, width - 1, 64), np.linspace(0, height - 1, 36))
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1)
    vh, vw = vdepth.shape
    vp = pixels * [vw / width, vh / height]
    vp = np.clip(vp, 0, [vw - 1, vh - 1])
    mh, mw = mdepth.shape
    mp = np.clip(pixels * [mw / width, mh / height], 0, [mw - 1, mh - 1])
    v = geo.sample_map(vdepth, vp)
    c = geo.sample_map(vconf, vp)
    mo = geo.sample_map(mdepth, mp)
    keep = (mo > 0) & (v > 0) & (c >= np.median(c))
    rho = geo.spearman(v[keep], mo[keep]) if keep.sum() >= 50 else None
    order = None
    bound = sorted(s for s in binding if s in subjects_camera)
    if len(bound) >= 2:
        first, second = bound[:2]
        vgg = geo.depth_order(subjects_camera[first]["depth"], subjects_camera[second]["depth"])
        medians: list[float] | None = []
        for subject in (first, second):
            key = f"s{k}_mid_t{binding[subject]}"
            if key not in gpu_npz or subjects_camera[subject]["frame"] != geo.MID_POSITION:
                medians = None
                break
            mask = np.unpackbits(gpu_npz[key])[: width * height].reshape(height, width).astype(bool)
            values = mdepth[_resize_mask(mask, (mw, mh)) & (mdepth > 0)]
            if values.size < 20:
                medians = None
                break
            medians.append(float(np.median(values)))
        mog = geo.depth_order(*medians) if medians else None
        if medians is None:
            vgg = None
        order = {"subjects": [first, second], "vggt": vgg, "moge": mog,
                 "agree": None if vgg is None or mog is None else vgg == mog}
    return {"spearman": _round(rho), "points": int(keep.sum()), "person_order": order}


def _pair_class(window: Mapping[str, Any], pair: Mapping[str, Any]) -> str:
    counts = [len(window["annotation"]["shots"][k]["persons"]) for k in pair["pair"]]
    two = ["two" if c >= 2 else ("one" if c == 1 else "none") for c in counts]
    return "/".join(two)


def _axis_verdict(axes: Sequence[Mapping[str, Any]], registration: str) -> dict[str, Any]:
    """一对镜的轴线侧：先看两镜都有两人的二维顺序，再看配准通过的三维侧别；取第一对主体。"""

    if not axes:
        return {"route": None, "same_side": None, "two_d": None, "three_d": None, "axis_frame": None, "subjects": None,
                "agree_2d_3d": None}
    row = axes[0]
    oa, ob = row["order_a"], row["order_b"]
    two_d = None
    if oa and ob and oa["order"] is not None and ob["order"] is not None:
        two_d = oa["order"] == ob["order"]
    three_d = None
    if registration in ("background_pass", "person_pass") and row["side_3d"] and None not in row["side_3d"]:
        three_d = row["side_3d"][0] == row["side_3d"][1]
    fa, fb = row["axis_frame_a"], row["axis_frame_b"]
    fallback = None
    if fa and fb and fa["side"] is not None and fb["side"] is not None:
        fallback = fa["side"] == fb["side"]
    if two_d is not None:
        route, same = "2d", two_d
    elif three_d is not None:
        route, same = "3d", three_d
    else:
        route, same = None, None
    return {"route": route, "same_side": same, "two_d": two_d, "three_d": three_d, "axis_frame": fallback,
            "subjects": [row["first"], row["second"]],
            "agree_2d_3d": None if two_d is None or three_d is None else two_d == three_d}


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {"n": numerator, "of": denominator, "rate": None if not denominator else round(numerator / denominator, 3)}


def run_analyze(args: argparse.Namespace) -> int:
    import numpy as np

    plan = _read(args.plan)
    out = args.out_dir
    shot_rows, pair_rows, errors, timings = [], [], [], []
    for name, window in sorted(plan["windows"].items()):
        path = out / "gpu" / f"{name}.json"
        if not path.exists():
            continue
        record = _read(path)
        timings.append({"window": name, "status": record["status"], "shots": len(window["shots"]),
                        "decode": window["decode_seconds"], **record["timing_seconds"], "window_seconds": record["window_seconds"]})
        if record["status"] != "ok":
            errors.append({"window": name, "error": record["error"]})
            continue
        gpu_npz = np.load(out / "gpu" / f"{name}.npz")
        moge_path = out / "moge" / f"{name}.npz"
        moge_npz = np.load(moge_path) if moge_path.exists() else None
        for shot, plan_shot in zip(record["shots"], window["shots"], strict=True):
            row = {"window": name, "dataset": window["dataset"], "shot": shot["shot"], "train_frames": plan_shot["train_frames"],
                   **{k: shot[k] for k in ("track_check", "rotation_first_last_deg", "translation_over_depth",
                                          "rotation_jitter_deg", "person_area_max", "mid_luminance", "binding",
                                          "persons_visible_annotation")}}
            row["depth"] = None if moge_npz is None else _depth_agreement(gpu_npz, moge_npz, shot["shot"], plan_shot["size"],
                                                                         shot["binding"], shot["subjects_camera"])
            row["mid_path"] = plan_shot["clip_paths"][plan_shot["vggt_positions"][geo.MID_POSITION]]
            shot_rows.append(row)
        if window["annotation"] is None:
            continue
        for pair in record["pairs"]:
            if not pair["same_place"]:
                continue
            k = pair["pair"][0]
            scene_persons = set(window["annotation"]["shots"][k]["persons"]) | set(window["annotation"]["shots"][k + 1]["persons"])
            row = {"window": name, "pair": pair["pair"], "class": _pair_class(window, pair),
                   "scene_persons": sorted(scene_persons), "registration": pair["registration"],
                   "background": pair["background"], "person": pair["person"],
                   "relative_rotation_deg": pair["relative_rotation_deg"],
                   "axis": _axis_verdict(pair["axes"], pair["registration"]), "axes": pair["axes"],
                   "mid_paths": [window["shots"][s]["clip_paths"][window["shots"][s]["vggt_positions"][geo.MID_POSITION]]
                                 for s in pair["pair"]]}
            pair_rows.append(row)
    # 问题 1
    moving = [r for r in shot_rows if r["track_check"]["camera_motion"] is not None
              and r["track_check"]["camera_motion"] > geo.MOVING_DISPLACEMENT]
    verdicts = [r["track_check"]["verdict"] for r in shot_rows]
    moving_verifiable = [r for r in moving if r["track_check"]["verdict"] != "unverifiable"]
    depth_rows = [r for r in shot_rows if r["depth"] and r["depth"]["spearman"] is not None]
    order_rows = [r for r in shot_rows if r["depth"] and r["depth"]["person_order"] and r["depth"]["person_order"]["agree"] is not None]
    q1 = {
        "shots": len(shot_rows), "verdicts": {v: verdicts.count(v) for v in sorted(set(verdicts))},
        "pass_among_verifiable": _rate(verdicts.count("pass"), verdicts.count("pass") + verdicts.count("fail")),
        "unverifiable": _rate(verdicts.count("unverifiable"), len(verdicts)),
        "moving_shots": len(moving),
        "moving_pass_among_verifiable": _rate(sum(r["track_check"]["verdict"] == "pass" for r in moving_verifiable), len(moving_verifiable)),
        "depth_agree": _rate(sum(r["depth"]["spearman"] >= geo.DEPTH_AGREE_RHO for r in depth_rows), len(depth_rows)),
        "depth_spearman_median": _round(float(np.median([r["depth"]["spearman"] for r in depth_rows]))) if depth_rows else None,
        "person_order_agree": _rate(sum(r["depth"]["person_order"]["agree"] for r in order_rows), len(order_rows)),
    }
    # 失败属性交叉表
    def attributes(r: Mapping[str, Any]) -> list[str]:
        tags = []
        if r["train_frames"] < 16:
            tags.append("fast_cut")
        if (r["rotation_jitter_deg"] or 0) > 0.5:
            tags.append("handheld")
        if (r["person_area_max"] or 0) > 0.5:
            tags.append("closeup")
        if (r["mid_luminance"] or 255) < 40:
            tags.append("dark")
        return tags or ["none"]
    table: dict[str, dict[str, int]] = {}
    for r in shot_rows:
        for tag in attributes(r):
            cell = table.setdefault(tag, {"pass": 0, "fail": 0, "unverifiable": 0})
            cell[r["track_check"]["verdict"]] += 1
        r["attributes"] = attributes(r)
    q1["attribute_table"] = table
    q1["by_dataset"] = {d: {v: sum(1 for r in shot_rows if r["dataset"] == d and r["track_check"]["verdict"] == v)
                            for v in ("pass", "fail", "unverifiable")} for d in sorted({r["dataset"] for r in shot_rows})}
    # 问题 2、3
    eligible = [r for r in pair_rows if len(r["scene_persons"]) >= 2]
    routes = [r["axis"]["route"] for r in eligible]
    registered = [r for r in eligible if r["registration"] in ("background_pass", "person_pass")]
    regs = [r["registration"] for r in eligible]
    fallback = [r for r in eligible if r["axis"].get("axis_frame") is not None]
    not_registered = [r for r in eligible if r["registration"] not in ("background_pass", "person_pass")]
    q2 = {"same_place_pairs": len(pair_rows), "eligible_pairs_two_persons": len(eligible),
          "by_class": {c: sum(1 for r in eligible if r["class"] == c) for c in sorted({r["class"] for r in eligible})},
          "determinable": _rate(sum(x is not None for x in routes), len(eligible)),
          "routes": {str(k): routes.count(k) for k in set(routes)},
          "same_side_among_determined": _rate(sum(r["axis"]["same_side"] is True for r in eligible if r["axis"]["route"]),
                                             sum(1 for x in routes if x)),
          "agree_2d_3d": _rate(sum(r["axis"]["agree_2d_3d"] is True for r in eligible if r["axis"]["agree_2d_3d"] is not None),
                               sum(1 for r in eligible if r["axis"]["agree_2d_3d"] is not None)),
          "binding_any_pair": _rate(sum(1 for r in eligible if r["axes"]), len(eligible))}
    q3 = {"eligible_pairs": len(eligible), "registration": {v: regs.count(v) for v in sorted(set(regs))},
          "registered": _rate(len(registered), len(eligible)),
          "unverifiable": _rate(regs.count("unverifiable"), len(eligible)),
          "axis_frame_coverage": _rate(len(fallback), len(eligible)),
          "axis_frame_coverage_unregistered": _rate(sum(1 for r in not_registered if r["axis"].get("axis_frame") is not None),
                                                    len(not_registered)),
          "all_same_place_registration": {v: [r["registration"] for r in pair_rows].count(v)
                                          for v in sorted({r["registration"] for r in pair_rows})}}
    # 问题 4
    ok = [t for t in timings if t["status"] == "ok"]
    gpu_seconds = [t["sam3"] + t["vggt_shot"] + t["cotracker"] + t["vggt_pair"] for t in ok]
    window_seconds = [t["window_seconds"] for t in ok]
    moge_files = sorted((out / "moge").glob("timing-*.json")) if (out / "moge").exists() else []
    moge_seconds = {}
    for f in moge_files:
        moge_seconds.update(_read(f)["seconds"])
    per_window_gpu = float(np.mean(window_seconds)) + (float(np.mean(list(moge_seconds.values()))) if moge_seconds else 0.0)
    q4 = {"windows": len(ok), "mean_window_seconds_gpu_process": _round(float(np.mean(window_seconds)), 2) if ok else None,
          "mean_model_seconds": _round(float(np.mean(gpu_seconds)), 2) if ok else None,
          "mean_stage_seconds": {k: _round(float(np.mean([t[k] for t in ok])), 2) for k in ("sam3", "vggt_shot", "cotracker", "vggt_pair", "check")} if ok else None,
          "mean_moge_seconds": _round(float(np.mean(list(moge_seconds.values()))), 2) if moge_seconds else None,
          "mean_decode_cpu_seconds": _round(float(np.mean([t["decode"] for t in ok])), 2) if ok else None,
          "gpu_hours_2000": _round(per_window_gpu * 2000 / 3600, 2), "gpu_hours_2500": _round(per_window_gpu * 2500 / 3600, 2)}
    # 抽 20 对亲看（按类分层）
    rng = random.Random(REVIEW_SEED)
    strata = {"two/two": "two/two", "one/two": "mixed", "two/one": "mixed", "one/one": "one/one"}
    by_class: dict[str, list] = {}
    for r in eligible:
        if r["class"] in strata:
            by_class.setdefault(strata[r["class"]], []).append(r)
    for rows in by_class.values():
        rng.shuffle(rows)
    review, classes = [], sorted(by_class)
    while len(review) < REVIEW_COUNT and any(by_class.values()):
        for c in classes:
            if by_class[c] and len(review) < REVIEW_COUNT:
                review.append(by_class[c].pop())
    summary = {"created_at": _now(), "plan": str(args.plan), "windows_with_gpu": len(timings), "errors": errors,
               "q1": q1, "q2": q2, "q3": q3, "q4": q4,
               "review_pairs": [{"window": r["window"], "pair": r["pair"], "class": r["class"]} for r in review]}
    for name, rows in (("shots.jsonl", shot_rows), ("pairs.jsonl", pair_rows), ("timings.jsonl", timings)):
        (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("q1", "q2", "q3", "q4")}, ensure_ascii=False, indent=1))
    return 0


# ---------------------------------------------------------------------------
# sheets (CPU)
# ---------------------------------------------------------------------------


def _colorize(depth: Any) -> Any:
    import numpy as np

    valid = depth > 0
    if not valid.any():
        return np.zeros(depth.shape + (3,), dtype=np.uint8)
    inverse = np.where(valid, 1.0 / np.maximum(depth, 1e-6), 0.0)
    lo, hi = np.percentile(inverse[valid], [2, 98])
    t = np.clip((inverse - lo) / max(hi - lo, 1e-6), 0, 1)
    anchors = np.array([[48, 18, 59], [65, 69, 171], [41, 150, 242], [41, 222, 156], [190, 240, 52], [251, 156, 35], [180, 30, 10]], float)
    position = t * (len(anchors) - 1)
    low = np.floor(position).astype(int).clip(0, len(anchors) - 2)
    frac = (position - low)[..., None]
    rgb = anchors[low] * (1 - frac) + anchors[low + 1] * frac
    rgb[~valid] = 0
    return rgb.astype(np.uint8)


def _font(size: int) -> Any:
    from PIL import ImageFont

    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


PALETTE = [(230, 57, 70), (69, 123, 157), (42, 157, 143), (233, 196, 106), (155, 93, 229), (244, 162, 97)]


def _subject_color(subject: str) -> tuple[int, int, int]:
    return PALETTE[(int(subject.split()[-1]) - 1) % len(PALETTE)]


def _overlay(path: str, masks: Mapping[str, Any], labels: Mapping[str, str]) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw

    image = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    for key, mask in masks.items():
        color = np.array(_subject_color(key), float)
        image[mask] = image[mask] * 0.55 + color * 0.45
    out = Image.fromarray(image.astype(np.uint8))
    draw = ImageDraw.Draw(out)
    for i, (key, mask) in enumerate(masks.items()):
        ys, xs = np.nonzero(mask)
        if xs.size:
            draw.text((float(np.median(xs)) - 20, float(ys.min()) + 4), labels.get(key, key), fill=(255, 255, 255),
                      font=_font(20), stroke_width=2, stroke_fill=(0, 0, 0))
    return out


def _birdseye(pair: Mapping[str, Any], size: tuple[int, int]) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, (250, 250, 250))
    draw = ImageDraw.Draw(image)
    cams = pair["cameras"]
    up = np.array(cams["up"])
    forward = np.array(cams["a"]["forward"])
    e1 = forward - np.dot(forward, up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)  # A 机位视角下的左方；横坐标取 -e2 即画面右方
    points = {"A": np.array(cams["a"]["center"]), "B": np.array(cams["b"]["center"])}
    for s, p in pair["persons_world"].items():
        points[s] = np.array(p)
    coords = {k: np.array([-np.dot(v, e2), np.dot(v, e1)]) for k, v in points.items()}
    allc = np.stack(list(coords.values()))
    lo, hi = allc.min(axis=0), allc.max(axis=0)
    span = max(float((hi - lo).max()), 1e-6) * 1.3
    centre = (lo + hi) / 2
    width, height = size

    def to_px(c: Any) -> tuple[float, float]:
        x = (c[0] - centre[0]) / span * (width - 40) + width / 2
        y = height / 2 - (c[1] - centre[1]) / span * (height - 40)
        return float(x), float(y)

    subjects = sorted(pair["persons_world"])
    if len(subjects) >= 2:
        a, b = to_px(coords[subjects[0]]), to_px(coords[subjects[1]])
        draw.line([a, b], fill=(120, 120, 120), width=2)
    for i, s in enumerate(subjects):
        x, y = to_px(coords[s])
        draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill=_subject_color(s))
        draw.text((x + 9, y - 9), s.replace("Subject ", "S"), fill=(0, 0, 0), font=_font(16))
    for key, color in (("a", (30, 90, 200)), ("b", (230, 120, 20))):
        c = np.array(cams[key]["center"])
        f = np.array(cams[key]["forward"])
        f2 = np.array([-np.dot(f, e2), np.dot(f, e1)])
        f2 = f2 / max(np.linalg.norm(f2), 1e-6) * span * 0.15
        x, y = to_px(coords[key.upper()])
        tx, ty = to_px(coords[key.upper()] + f2)
        draw.rectangle([x - 6, y - 6, x + 6, y + 6], fill=color)
        draw.line([(x, y), (tx, ty)], fill=color, width=3)
        draw.text((x + 8, y + 4), key.upper(), fill=color, font=_font(16))
    draw.text((6, 4), "top view (A looks up)", fill=(80, 80, 80), font=_font(14))
    return image


def run_sheets(args: argparse.Namespace) -> int:
    import numpy as np
    from PIL import Image, ImageDraw

    out = args.out_dir
    summary = _read(out / "summary.json")
    plan = _read(args.plan)
    pairs = {(r["window"], tuple(r["pair"])): r for r in map(json.loads, (out / "pairs.jsonl").read_text().splitlines())}
    sheet_dir = out / "sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(summary["review_pairs"]):
        row = pairs[(item["window"], tuple(item["pair"]))]
        window = plan["windows"][item["window"]]
        record = _read(out / "gpu" / f"{item['window']}.json")
        gpu_npz = np.load(out / "gpu" / f"{item['window']}.npz")
        pair = next(p for p in record["pairs"] if p["pair"] == item["pair"])
        tiles = []
        for side, k in zip("ab", item["pair"]):
            shot = record["shots"][k]
            plan_shot = window["shots"][k]
            width, height = plan_shot["size"]
            masks, labels = {}, {}
            for subject, track in shot["binding"].items():
                key = f"s{k}_mid_t{track}"
                if key in gpu_npz:
                    masks[subject] = np.unpackbits(gpu_npz[key])[: width * height].reshape(height, width).astype(bool)
                    labels[subject] = subject.replace("Subject ", "S")
            frame = _overlay(row["mid_paths"][0 if side == "a" else 1], masks, labels).resize((480, int(480 * height / width)))
            depth = Image.fromarray(_colorize(gpu_npz[f"s{k}_depth"].astype(np.float32))).resize(frame.size)
            tiles.append((frame, depth))
        th = max(t[0].size[1] for t in tiles)
        canvas = Image.new("RGB", (480 * 2 + 400, th * 2 + 150), (255, 255, 255))
        for i, (frame, depth) in enumerate(tiles):
            canvas.paste(frame, (480 * i, 0))
            canvas.paste(depth, (480 * i, th))
        canvas.paste(_birdseye(pair, (400, th * 2)), (960, 0))
        draw = ImageDraw.Draw(canvas)
        axis = row["axis"]
        first_axes = row["axes"][0] if row["axes"] else None
        lines = [f"#{index:02d} {item['window']} shots {item['pair']} class={row['class']} registration={row['registration']} "
                 f"(inliers={row['background']['inliers']}, bg_err={row['background']['reprojection_error']}, "
                 f"person={[(s['subject'][-1], s['ratio']) for s in row['person']['subjects']]}) rot={row['relative_rotation_deg']}deg",
                 f"axis: route={axis['route']} same_side={axis['same_side']} 2d={axis.get('two_d')} 3d={axis.get('three_d')} "
                 f"axis_frame={axis.get('axis_frame')} subjects={axis.get('subjects')}"]
        if first_axes:
            oa, ob = first_axes["order_a"], first_axes["order_b"]
            lines.append(f"order A={None if not oa else oa['order']} B={None if not ob else ob['order']} (+1: {first_axes['first']} left)  "
                         f"side3d={first_axes['side_sine_3d']}  axisframe A={first_axes['axis_frame_a'] and first_axes['axis_frame_a'].get('side_sine')} "
                         f"B={first_axes['axis_frame_b'] and first_axes['axis_frame_b'].get('side_sine')}")
        for i, text in enumerate(lines):
            draw.text((8, th * 2 + 8 + 30 * i), text, fill=(0, 0, 0), font=_font(17))
        canvas.save(sheet_dir / f"pair-{index:02d}-{item['window']}-s{item['pair'][0]}.jpg", quality=90)
    # 失败镜头中帧总览
    shots = [json.loads(line) for line in (out / "shots.jsonl").read_text().splitlines()]
    failed = [s for s in shots if s["track_check"]["verdict"] == "fail"]
    tiles = []
    for s in failed:
        image = Image.open(s["mid_path"]).convert("RGB")
        image.thumbnail((320, 320))
        tile = Image.new("RGB", (320, 220), (255, 255, 255))
        tile.paste(image, (0, 0))
        ImageDraw.Draw(tile).text((4, 184), f"{s['window']} s{s['shot']} e={s['track_check']['reprojection_error']} "
                                            f"m={s['track_check']['camera_motion']} {','.join(s['attributes'])}",
                                  fill=(0, 0, 0), font=_font(13))
        tiles.append(tile)
    for page in range(0, len(tiles), 24):
        chunk = tiles[page:page + 24]
        sheet = Image.new("RGB", (320 * 4, 220 * ((len(chunk) + 3) // 4)), (255, 255, 255))
        for i, tile in enumerate(chunk):
            sheet.paste(tile, (320 * (i % 4), 220 * (i // 4)))
        sheet.save(sheet_dir / f"failed-{page // 24:02d}.jpg", quality=88)
    print(json.dumps({"review_sheets": len(summary["review_pairs"]), "failed_shots": len(failed)}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("frames")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--window-map", type=Path, required=True)
    p.add_argument("--attempts", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=run_frames)
    p = sub.add_parser("gpu")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--sam-checkpoint", type=Path, required=True)
    p.add_argument("--sam-bpe", type=Path, required=True)
    p.add_argument("--vggt-weights", type=Path, required=True)
    p.add_argument("--cotracker-weights", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--windows", default="")
    p.set_defaults(func=run_gpu)
    p = sub.add_parser("moge")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.set_defaults(func=run_moge)
    p = sub.add_parser("analyze")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(func=run_analyze)
    p = sub.add_parser("sheets")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(func=run_sheets)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
