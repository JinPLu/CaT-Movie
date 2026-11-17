#!/usr/bin/env python3
"""多镜窗口逐镜空间事实抽取的薄入口（按镜切开）。规则在 :mod:`mspc.data.spatial_facts`。

所有产物写在 ``--work`` 目录下，逐镜文件已存在就跳过（可断点续跑、可多进程分窗）。子命令按执行面与环境分：

* ``frames``（CPU，项目环境）：长集 dev 真片按修订 3 镜界，每镜按 16 fps 网格取全部帧（首末帧入选），裁切缩到
  832×480 存 JPEG 帧目录（SAM 3 视频预测器的官方输入形式之一）。写 ``windows.json``。
* ``frames-render``（CPU，``env/renderer_public_metrics``）：生成片（``results.json`` 的硬切拼接视频）按声明镜界切开，
  每镜全部 16 fps 帧存帧目录；主体、在场名单与帧数取自同一批窗真片的 ``windows.json``。之后各步与真片相同。
* ``frames-pool``（CPU，``env/wan22_fun``）：训练池指定窗（v6 修订镜界 ``rows.jsonl``）按训练读帧路径取全部 16 fps 帧；
  ``attach-logs`` 在 v10 事件日志编译出来后接上主体、在场名单与衣着（标注没出来时通用概念、深度、姿态可以先跑）。
* ``sam3``（GPU，``env/sam3_sm80``）：通用概念 ``person``/``animal`` 每镜每概念一个会话逐帧跟踪；热启动用上游默认值；
  逐帧框与逐帧掩码（packbits＋压缩）都落盘。``sam3-phrase``：每个在场人物或动物的短属性词各一个会话，逐帧按掩码
  IoU ≥ 0.5 对齐到通用实例上计票。
* ``depth-dino``（GPU，``env/renderer_public_metrics``）：Depth Anything V2 Small 视差（模型卡用法，后处理回原尺寸），
  实例掩码内中位视差、两两相接处的视差差（遮挡次序）、本帧视差范围；DINOv2 掩码画布嵌入（与身份列同一栈）。
* ``people``（GPU，``env/person_tools_cu128``）：RTMW-x 全身 133 点（用 SAM 实例框作人体框）、InsightFace buffalo_l
  人脸（整帧检测，脸框中心落在实例掩码内归给该实例）、OSNet-AIN 人体重识别嵌入（实例框裁剪）；关键点另在水平翻转
  帧上跑一遍，供朝向镜像等变核对。
* ``bind``（CPU 客户端＋本地 Qwen3.8-27B vLLM）：短属性词唯一触发的直接指认，其余主体交 VLM 在中帧编号框上兜底。
* ``calibrate``（CPU）：在标定半（dev 窗按 id 排序取偶数位）上用金标定前后间隔、朝向侧向阈值与跨镜三路 logistic。
* ``facts``（CPU）：逐镜空间事实（布局三帧多数、首帧、末帧）与跨镜认人。
* ``screen-merge``（CPU）：训练池窗的 v10 事件日志里人物与动物的 screen 类事实换成本链首端事实：左右（5 帧投票）、
  前后、只写面朝镜头的朝向（依盲标金标准定）；其余写旁车；逐窗过事件日志校验，写新的 ``event_logs.jsonl``。
* ``evaluate``（CPU）：对 dev 金标逐字段读数，并与旧链（3 帧 SAM 3.1＋Qwen 指认＋Qwen 朝向＋MoGe-2 1.5 倍，即金标
  改错前的 ``oracle.json``）比。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for _root in (SOURCE_ROOT, REPOSITORY_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

WINDOWS_SCHEMA = "spatial_facts_windows_v1"
FRAME_WIDTH, FRAME_HEIGHT = 832, 480
GENERIC_CONCEPT = {"person": "person", "animal": "animal"}
ANIMATE_KINDS = ("person", "animal")
#: 人体框外扩：RTMW 前处理自带 1.25 倍 padding（rtmlib ``bbox_xyxy2cs(padding=1.25)``，与 SDK pipeline 同），不再另扩。
#: InsightFace：脸框中心落在实例掩码内才归给该实例。
CALIBRATION_GRID = {"depth_gap": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                    "face_visible": (2.0, 3.0, 4.0),
                    "face_back": (4.0, 5.0, 6.0, 7.0),
                    "lateral": (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)}
PATHS = ("face", "reid", "dino")


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


def _shots(work: Path, stories: str) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]:
    windows = _read(work / "windows.json")
    if windows["schema_version"] != WINDOWS_SCHEMA:
        raise ValueError(f"unexpected windows schema {windows['schema_version']!r}")
    wanted = None if not stories else set(stories.split(","))
    out = []
    for story_id, window in sorted(windows["windows"].items()):
        if wanted is not None and story_id not in wanted:
            continue
        for shot in window["shots"]:
            if shot["concepts"]:
                out.append((story_id, window, shot))
    return out


def _frame_path(work: Path, story_id: str, shot_id: str, position: int) -> Path:
    return work / "frames" / story_id / shot_id / f"{position:05d}.jpg"


def _unpack(packed: Any, shape: Any) -> Any:
    import numpy as np

    count = int(shape[0]) * int(shape[1])
    return np.unpackbits(packed)[:count].reshape(int(shape[0]), int(shape[1])).astype(bool)


def _pack(masks: dict[str, Any], key: str, mask: Any) -> None:
    import numpy as np

    masks[key] = np.packbits(mask)
    masks[key + "_shape"] = np.asarray(mask.shape)


def _sam3_dir(work: Path, story_id: str, shot_id: str) -> Path:
    return work / "sam3" / story_id / shot_id


def _sam3_record(work: Path, story_id: str, shot: Mapping[str, Any], *, phrases: bool) -> dict[str, Any]:
    """一镜的 SAM 3.1 结果：各通用概念文件合并成 ``tracks``；``phrases=True`` 时连同逐主体短属性词计票。

    目录 ``sam3/<窗>/<镜>/``：``<概念>.json``（逐帧框）＋``<概念>.npz``（逐帧掩码），``phrase-<主体>.json``（计票）。"""

    folder = _sam3_dir(work, story_id, shot["shot_id"])
    tracks: dict[str, Any] = {}
    hotstart: dict[str, Any] = {}
    for concept in shot["concepts"]:
        row = _read(folder / f"{concept}.json")
        tracks.update(row["tracks"])
        hotstart[concept] = row["hotstart"]
    record = {"frame_count": shot["frame_count"], "key_positions": shot["key_positions"],
              "layout_positions": shot["layout_positions"], "tracks": tracks, "hotstart": hotstart}
    if phrases:
        record["phrases"] = {s: _read(folder / f"phrase-{s}.json") for s in shot["animate"]}
    return record


def _shot_masks(work: Path, story_id: str, shot: Mapping[str, Any],
                positions: Sequence[int]) -> dict[tuple[str, int], Any]:
    """一镜通用概念实例在指定帧位置的掩码 ``{(实例, 位置): mask}``。"""

    import numpy as np

    folder = _sam3_dir(work, story_id, shot["shot_id"])
    masks = {}
    for concept in shot["concepts"]:
        packed = np.load(folder / f"{concept}.npz")
        for key in packed.files:
            if key.endswith("_shape"):
                continue
            track, position = key.rsplit("@", 1)
            if int(position) in positions:
                masks[(track, int(position))] = _unpack(packed[key], packed[key + "_shape"])
    return masks


def _key_masks(work: Path, story_id: str, shot: Mapping[str, Any]) -> tuple[dict[str, Any], dict[tuple[str, int], Any]]:
    """一镜的 SAM 3.1 结果与关键帧掩码。"""

    return _sam3_record(work, story_id, shot, phrases=False), _shot_masks(work, story_id, shot, shot["key_positions"])


# ---------------------------------------------------------------------------
# frames（CPU）
# ---------------------------------------------------------------------------


def run_frames(args: argparse.Namespace) -> int:
    from PIL import Image

    from mspc.data.frames import shot_grid
    from mspc.data.msm_global_frames import default_msm_frame_reader
    from mspc.data.spatial_facts import TARGET_FPS, key_positions, layout_positions

    work = args.work
    if (work / "windows.json").exists():
        raise FileExistsError(work / "windows.json")
    pairs = {pair["story_id"]: pair for pair in _read(args.pairs)["pairs"]}
    freeze = _read(args.freeze)
    dev = sorted(w["window_id"] for w in freeze["windows"] if w["split"] == "dev")
    if set(dev) != set(pairs):
        raise ValueError(f"{args.pairs} does not cover exactly the frozen dev windows")
    manifest = _read(args.candidate_r3)
    if manifest["render_revision"]["revision"] != 3:
        raise ValueError("candidate manifest is not render revision 3")
    candidate = {story["story_id"]: story for story in manifest["stories"]}
    windows: dict[str, Any] = {}
    frames_written = 0
    for story_id in dev:
        script = pairs[story_id]["context_b"]
        source = candidate[story_id]
        if len(source["shots"]) != len(script["shots"]):
            raise ValueError(f"{story_id}: {len(source['shots'])} r3 shots vs {len(script['shots'])} script shots")
        kinds = {row["subject_id"]: row["kind"] for row in script["subjects"]}
        shots = []
        grids = [shot_grid(s["start_frame"], s["end_frame"], s["render_frames"]) for s in source["shots"]]
        flat = [index + source["container_preroll_frames"] for grid in grids for index in grid]
        images = default_msm_frame_reader(Path(source["source_video_path"]), flat)
        x0, y0, x1, y1 = source["crop_box"]
        cursor = 0
        for script_shot, grid in zip(script["shots"], grids, strict=True):
            for position in range(len(grid)):
                path = _frame_path(work, story_id, script_shot["shot_id"], position)
                path.parent.mkdir(parents=True, exist_ok=True)
                image = images[cursor + position].crop((x0, y0, x1, y1))
                image.resize((FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.LANCZOS).save(path, format="JPEG", quality=95)
            cursor += len(grid)
            frames_written += len(grid)
            present = set(script_shot["present_subjects"])
            shots.append({"shot_id": script_shot["shot_id"], "location_id": script_shot["location_id"],
                          "present": list(script_shot["present_subjects"]),
                          "animate": [s["subject_id"] for s in script["subjects"]
                                      if s["subject_id"] in present and kinds[s["subject_id"]] in ANIMATE_KINDS],
                          "concepts": sorted({kinds[s] for s in present if kinds.get(s) in ANIMATE_KINDS}),
                          "attributes": {s["subject_id"]: s["attributes"] for s in script["subjects"]
                                         if s["subject_id"] in present and kinds[s["subject_id"]] in ANIMATE_KINDS},
                          "source_frames": grid, "frame_count": len(grid),
                          "layout_positions": list(layout_positions(len(grid))),
                          "key_positions": list(key_positions(len(grid)))})
        windows[story_id] = {"source_video_path": source["source_video_path"], "source_fps": source["fps"],
                             "container_preroll_frames": source["container_preroll_frames"], "crop_box": source["crop_box"],
                             "subjects": [{"subject_id": s["subject_id"], "kind": s["kind"], "attributes": s["attributes"]}
                                          for s in script["subjects"]],
                             "shots": shots}
        print(f"[frames] {story_id} {len(shots)} shots", file=sys.stderr, flush=True)
    _write_new(work / "windows.json", {"schema_version": WINDOWS_SCHEMA, "created_at": _now(), "fps": TARGET_FPS,
                                       "width": FRAME_WIDTH, "height": FRAME_HEIGHT, "pairs": str(args.pairs),
                                       "freeze": str(args.freeze), "candidate_r3": str(args.candidate_r3),
                                       "windows": windows})
    print(json.dumps({"windows": len(windows), "frames": frames_written}))
    return 0


def run_frames_render(args: argparse.Namespace) -> int:
    """历史空间诊断生成片（带 ``aligned_shots`` 的 ``results.json``，逐镜出片硬切拼接）按声明的镜界切开，每镜全部 16 fps 帧
    存成帧目录。主体、在场名单与每镜帧数取自同一批窗的真片 ``windows.json``（同一剧本、同一帧数），帧数不符即报错。"""

    from PIL import Image

    from mspc.data.spatial_facts import TARGET_FPS
    from mspc.evaluation.renderer_qualification_official_backend import _decode_rgb

    work = args.work
    if (work / "windows.json").exists():
        raise FileExistsError(work / "windows.json")
    template = _read(args.template)
    if template["schema_version"] != WINDOWS_SCHEMA:
        raise ValueError(f"unexpected template schema {template['schema_version']!r}")
    rows = [case for path in args.results for case in _read(path)["cases"]]
    cases = {case["story_id"]: case for case in rows}
    if len(cases) != len(rows) or set(cases) != set(template["windows"]):
        raise ValueError(f"{[str(p) for p in args.results]} do not cover the template windows exactly once")
    windows: dict[str, Any] = {}
    frames_written = 0
    for story_id, window in sorted(template["windows"].items()):
        case = cases[story_id]
        if case["fps"] != TARGET_FPS:
            raise ValueError(f"{story_id}: generated video is {case['fps']} fps, expected {TARGET_FPS}")
        frames, _ = _decode_rgb(Path(case["video_path"]))
        spans = case["aligned_shots"]
        if len(spans) != len(window["shots"]) or len(frames) != spans[-1]["end_frame"]:
            raise ValueError(f"{story_id}: {len(spans)} generated shots / {len(frames)} frames do not match the plan")
        shots = []
        for shot, span in zip(window["shots"], spans, strict=True):
            count = span["end_frame"] - span["start_frame"]
            if span["shot_id"] != shot["shot_id"] or count != shot["frame_count"]:
                raise ValueError(f"{story_id} {shot['shot_id']}: generated shot {span['shot_id']} has {count} frames, "
                                 f"the real shot has {shot['frame_count']}")
            for position in range(count):
                image = Image.fromarray(frames[span["start_frame"] + position])
                if image.size != (FRAME_WIDTH, FRAME_HEIGHT):
                    raise ValueError(f"{story_id}: generated frames are {image.size}, expected {FRAME_WIDTH}x{FRAME_HEIGHT}")
                path = _frame_path(work, story_id, shot["shot_id"], position)
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path, format="JPEG", quality=95)
            frames_written += count
            shots.append({key: shot[key] for key in ("shot_id", "location_id", "present", "animate", "concepts",
                                                     "attributes", "frame_count", "layout_positions", "key_positions")})
        windows[story_id] = {"video_path": case["video_path"], "seed": case["seed"], "subjects": window["subjects"],
                             "shots": shots}
        print(f"[frames-render] {story_id} {len(shots)} shots", file=sys.stderr, flush=True)
    _write_new(work / "windows.json", {"schema_version": WINDOWS_SCHEMA, "created_at": _now(), "fps": TARGET_FPS,
                                       "width": FRAME_WIDTH, "height": FRAME_HEIGHT,
                                       "results": [str(path) for path in args.results],
                                       "template": str(args.template), "windows": windows})
    print(json.dumps({"windows": len(windows), "frames": frames_written}))
    return 0


def _shot_clothing(event_log: Mapping[str, Any], index: int) -> dict[str, str]:
    """事件日志里这一镜首帧登记的衣着（state/costume）。"""

    return {f["subject"]: f["object_or_value"] for f in event_log["initial_state"]
            if f["family"] == "state" and f["predicate_or_axis"] == "costume" and f["shot_span"][0] == index}


def _pool_window(job: tuple[Path, Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    """训练池一窗：按训练读帧路径（全帧缩放覆盖＋CenterCrop 到 832×480，与 ``MultiShotDataset`` 同一读法）取 v6 计划的
    全部 16 fps 帧，逐镜存 JPEG。先只登记通用概念 person；主体在 ``attach-logs`` 时从 v10 事件日志接上，有动物在场的镜
    再补 animal 概念（``sam3`` 只跑缺的概念）。"""

    from PIL import Image
    from torchvision import transforms

    from mspc.data.multishot import VIDEO_OUTPUT_MODE_UINT8, read_video_frames
    from mspc.data.spatial_facts import key_positions, layout_positions

    work, row = job
    window_id = row["case_id"].replace(":", "__")
    plan = row["plan"]
    crop = transforms.CenterCrop((FRAME_HEIGHT, FRAME_WIDTH))
    shots = []
    for k, (indices, video) in enumerate(zip(plan["frame_indices_by_shot"], plan["source_video_paths"], strict=True)):
        frames = read_video_frames(path=video, frame_indices=list(indices), target_height=FRAME_HEIGHT,
                                   target_width=FRAME_WIDTH, transform_centercrop=crop, num_threads=1,
                                   output_mode=VIDEO_OUTPUT_MODE_UINT8)
        array = frames.permute(1, 2, 3, 0).numpy()
        shot_id = f"s{k + 1:02d}"
        for position, image in enumerate(array):
            path = _frame_path(work, window_id, shot_id, position)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(path, format="JPEG", quality=95)
        shots.append({"shot_id": shot_id, "present": [], "animate": [], "concepts": ["person"],
                      "attributes": {}, "source_video_path": video, "source_frames": list(indices),
                      "frame_count": len(indices), "layout_positions": list(layout_positions(len(indices))),
                      "key_positions": list(key_positions(len(indices)))})
    return window_id, {"case_id": row["case_id"], "window_id": None, "subjects": [], "event_log": None, "shots": shots}


def run_frames_pool(args: argparse.Namespace) -> int:
    """训练池（v6 修订镜界的 ``rows.jsonl``）指定窗：取帧、登记镜头；主体等标注编译出来后用 ``attach-logs`` 接上。"""

    from concurrent.futures import ProcessPoolExecutor

    from mspc.data.spatial_facts import TARGET_FPS

    work = args.work
    if (work / "windows.json").exists():
        raise FileExistsError(work / "windows.json")
    wanted = set(_read(args.case_ids))
    rows = []
    with open(args.rows, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["case_id"] in wanted:
                rows.append(row)
    if len(rows) != len(wanted):
        raise ValueError(f"{len(wanted) - len(rows)} requested case_ids are not in {args.rows}")
    windows: dict[str, Any] = {}
    frames = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (window_id, window) in enumerate(pool.map(_pool_window, [(work, row) for row in rows]), start=1):
            windows[window_id] = window
            frames += sum(shot["frame_count"] for shot in window["shots"])
            if done % 50 == 0 or done == len(rows):
                print(f"[frames-pool] {done}/{len(rows)} windows {frames} frames", file=sys.stderr, flush=True)
    _write_new(work / "windows.json", {"schema_version": WINDOWS_SCHEMA, "created_at": _now(), "fps": TARGET_FPS,
                                       "width": FRAME_WIDTH, "height": FRAME_HEIGHT, "rows": str(args.rows),
                                       "case_ids": str(args.case_ids), "windows": windows})
    print(json.dumps({"windows": len(windows), "frames": frames}))
    return 0


def run_attach_logs(args: argparse.Namespace) -> int:
    """把 v10 编译的事件日志接到训练池窗上：主体编号、逐镜在场名单、外观＋该镜衣着；有动物在场的镜补 animal 概念
    （之后再跑一遍 ``sam3`` 补缺的概念；这些动物实例没有深度与姿态，并入只用左右，不受影响）。已接过的窗不动。"""

    path = args.work / "windows.json"
    payload = _read(path)
    logs = {}
    with open(args.event_logs, encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            logs[entry["case_id"]] = entry
    attached = 0
    for window in payload["windows"].values():
        entry = logs.get(window["case_id"])
        if entry is None or window["event_log"] is not None:
            continue
        log = entry["event_log"]
        if len(log["shots"]) != len(window["shots"]):
            raise ValueError(f"{window['case_id']}: {len(log['shots'])} logged shots vs {len(window['shots'])} planned")
        kinds = {s["subject_id"]: s["kind"] for s in log["subjects"]}
        appearance = {s["subject_id"]: s["appearance"] for s in log["subjects"]}
        for index, (shot, logged) in enumerate(zip(window["shots"], log["shots"], strict=True)):
            if logged["shot_id"] != shot["shot_id"] or logged["frame_count"] != shot["frame_count"]:
                raise ValueError(f"{window['case_id']} {shot['shot_id']}: log shot {logged['shot_id']} with "
                                 f"{logged['frame_count']} frames does not match the plan")
            clothing = _shot_clothing(log, index)
            shot["present"] = list(logged["present_subjects"])
            shot["animate"] = [s for s in logged["present_subjects"] if kinds.get(s) in ANIMATE_KINDS]
            shot["attributes"] = {s: appearance[s] + (f", wearing {clothing[s]}" if s in clothing else "")
                                  for s in shot["animate"]}
            shot["concepts"] = sorted(set(shot["concepts"]) | {kinds[s] for s in shot["animate"]})
        window["window_id"] = entry["window_id"]
        window["subjects"] = [{"subject_id": s["subject_id"], "kind": s["kind"], "attributes": s["appearance"]}
                              for s in log["subjects"]]
        window["event_log"] = str(args.event_logs)
        attached += 1
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    print(json.dumps({"attached": attached, "windows": len(payload["windows"])}))
    return 0


# ---------------------------------------------------------------------------
# sam3（GPU，sam3_sm80）
# ---------------------------------------------------------------------------


#: 上游 SAM 3.1 已知缺陷（facebookresearch/sam3#572，未修）：热启动删掉一个桶里持有共享条件帧的实例时，会清空同桶其余实例的
#: 记忆，随后传播报这句错。只在官方热启动下才会删实例；遇到时这个会话改用热启动 0 整段重跑，并逐会话记下。
_UPSTREAM_572 = "No points are provided; please add points first"


def _track_session(model: Any, frame_dir: Path, text: str, scratch: Path,
                   count: int) -> tuple[list[list[dict[str, Any]]], list[int]]:
    """一个概念在一镜全部帧上的一次传播：逐帧 ``[{track_id, score, box(像素 xyxy), mask}]`` 与实际用的热启动。"""

    import numpy as np

    scratch.mkdir(parents=True, exist_ok=False)
    mask_path = scratch / "masks.npz"
    try:
        hotstart = list(model.official_hotstart)
        try:
            rows = model.track_frame_directory(frame_dir=str(frame_dir), text=text, mask_path=str(mask_path),
                                               hotstart_delay=hotstart[0], hotstart_unmatch_thresh=hotstart[1])
        except RuntimeError as error:
            if str(error) != _UPSTREAM_572:
                raise
            hotstart = [0, 0]
            rows = model.track_frame_directory(frame_dir=str(frame_dir), text=text, mask_path=str(mask_path),
                                               hotstart_delay=0, hotstart_unmatch_thresh=0)
        if sorted(row["frame_index"] for row in rows) != list(range(count)):
            raise ValueError(f"SAM 3.1 did not return every frame exactly once for {text!r} in {frame_dir}")
        packed = np.load(mask_path)
        out: list[list[dict[str, Any]]] = [[] for _ in range(count)]
        for row in rows:
            for det in row["detections"]:
                mask = _unpack(packed[det["mask_key"]], packed[det["mask_key"] + "_shape"])
                h, w = mask.shape
                x0, y0, x1, y1 = (float(v) / 1000.0 for v in det["box"])
                out[row["frame_index"]].append({"track_id": int(det["track_id"]), "score": float(det["score"]),
                                                "box": [x0 * w, y0 * h, x1 * w, y1 * h], "mask": mask})
        return out, hotstart
    finally:
        if mask_path.exists():
            mask_path.unlink()
        scratch.rmdir()


def _sam3_model(args: argparse.Namespace) -> Any:
    from mspc.data.grounding.sam3_backend import Sam3Backend

    # 官方默认：max_num_objects=16、multiplex_count=16、compile=False；use_fa3 只支持 Hopper，A100 关。
    return Sam3Backend(checkpoint_path=str(args.checkpoint), bpe_path=str(args.bpe), version="sam3.1",
                       objects_per_forward=16, enable_object_multiplexing=True, use_fa3=False, use_rope_real=True)


def run_sam3(args: argparse.Namespace) -> int:
    """通用概念（``person``/``animal``）逐帧跟踪：每镜每个概念一个会话，逐帧框与逐帧掩码都落盘。"""

    import numpy as np
    import torch

    torch.set_num_threads(4)
    started = time.perf_counter()
    model = _sam3_model(args)
    load_seconds = time.perf_counter() - started
    scratch_root = args.scratch / f"sam3-{os.getpid()}"
    done = 0
    for story_id, _window, shot in _shots(args.work, args.stories):
        folder = _sam3_dir(args.work, story_id, shot["shot_id"])
        frame_dir = args.work / "frames" / story_id / shot["shot_id"]
        for concept in shot["concepts"]:
            out_json = folder / f"{concept}.json"
            if out_json.exists():
                continue
            tick = time.perf_counter()
            frames, hotstart = _track_session(model, frame_dir, GENERIC_CONCEPT[concept], scratch_root / f"g-{concept}",
                                              shot["frame_count"])
            tracks: dict[str, dict[str, Any]] = {}
            masks: dict[str, Any] = {}
            for position, dets in enumerate(frames):
                for det in dets:
                    key = f"{concept}#{det['track_id']}"
                    row = tracks.setdefault(key, {"concept": concept, "track_id": det["track_id"], "frames": {}})
                    row["frames"][str(position)] = [round(v, 2) for v in det["box"]] + [round(det["score"], 4)]
                    _pack(masks, f"{key}@{position}", det["mask"])
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / f"{concept}.npz", **masks)
            _write_new(out_json, {"concept": concept, "frame_count": shot["frame_count"], "hotstart": hotstart,
                                  "tracks": tracks, "seconds": round(time.perf_counter() - tick, 2)})
            done += 1
            print(f"[sam3] {story_id}/{shot['shot_id']} {concept} {shot['frame_count']}f {len(tracks)} tracks "
                  f"{time.perf_counter() - tick:.1f}s", file=sys.stderr, flush=True)
    print(json.dumps({"sessions": done, "load_seconds": round(load_seconds, 1),
                      "elapsed_seconds": round(time.perf_counter() - started, 1)}))
    return 0


def run_sam3_phrase(args: argparse.Namespace) -> int:
    """逐主体短属性词：每镜每个在场人物或动物一个会话，逐帧按掩码 IoU ≥ 0.5 对齐到通用实例上计票。"""

    import torch

    from mspc.data.spatial_facts import phrase_frame_match, short_phrase

    torch.set_num_threads(4)
    started = time.perf_counter()
    model = _sam3_model(args)
    scratch_root = args.scratch / f"sam3p-{os.getpid()}"
    done = 0
    for story_id, window, shot in _shots(args.work, args.stories):
        folder = _sam3_dir(args.work, story_id, shot["shot_id"])
        todo = [s for s in shot["animate"] if not (folder / f"phrase-{s}.json").exists()]
        if not todo:
            continue
        kinds = {row["subject_id"]: row["kind"] for row in window["subjects"]}
        count = shot["frame_count"]
        masks = _shot_masks(args.work, story_id, shot, range(count))
        per_frame: list[dict[str, Any]] = [{} for _ in range(count)]
        for (track, position), mask in masks.items():
            per_frame[position][track] = mask
        frame_dir = args.work / "frames" / story_id / shot["shot_id"]
        for subject in todo:
            tick = time.perf_counter()
            phrase = short_phrase(shot["attributes"][subject], kinds[subject])
            frames, hotstart = _track_session(model, frame_dir, phrase, scratch_root / f"p-{subject.split()[-1]}", count)
            votes: dict[str, int] = {}
            detected = 0
            for position, dets in enumerate(frames):
                if dets:
                    detected += 1
                matched = {t for t in phrase_frame_match([d["mask"] for d in dets], per_frame[position]) if t is not None}
                for track in matched:
                    votes[track] = votes.get(track, 0) + 1
            _write_new(folder / f"phrase-{subject}.json", {"subject": subject, "phrase": phrase, "hotstart": hotstart,
                                                          "detected_frames": detected, "votes": votes,
                                                          "seconds": round(time.perf_counter() - tick, 2)})
            done += 1
        print(f"[sam3-phrase] {story_id}/{shot['shot_id']} {len(todo)} subjects", file=sys.stderr, flush=True)
    print(json.dumps({"sessions": done, "elapsed_seconds": round(time.perf_counter() - started, 1)}))
    return 0


# ---------------------------------------------------------------------------
# depth-dino（GPU，renderer_public_metrics）
# ---------------------------------------------------------------------------


def run_depth_dino(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    from mspc.data.spatial_facts import contact_difference, disparity_range
    from mspc.evaluation.renderer_qualification_official_backend import (
        DEFAULT_ROI_DETECTOR_CONFIG,
        _DinoRoiEngine,
        _model_batch_from_roi_config,
    )

    processor = AutoImageProcessor.from_pretrained(str(args.depth_model), local_files_only=True)
    depth_model = AutoModelForDepthEstimation.from_pretrained(str(args.depth_model), local_files_only=True)
    depth_model = depth_model.to(args.device).eval()
    engine = _DinoRoiEngine(args.resource_root, args.device)
    dino_batch = _model_batch_from_roi_config(DEFAULT_ROI_DETECTOR_CONFIG).dino_crop_batch_size
    started = time.perf_counter()
    frames_done = 0
    for story_id, _window, shot in _shots(args.work, args.stories):
        out_json = args.work / "depth_dino" / story_id / f"{shot['shot_id']}.json"
        if out_json.exists():
            continue
        record, masks = _key_masks(args.work, story_id, shot)
        positions = record["key_positions"]
        images = [Image.open(_frame_path(args.work, story_id, shot["shot_id"], p)).convert("RGB") for p in positions]
        # 模型卡用法：processor → model → post_process_depth_estimation 回原图尺寸；fp32，不改输出。
        inputs = processor(images=images, return_tensors="pt").to(args.device)
        with torch.no_grad():
            outputs = depth_model(**inputs)
        maps = processor.post_process_depth_estimation(outputs, target_sizes=[(im.height, im.width) for im in images])
        frames_done += len(images)
        rows: dict[str, Any] = {}
        crops, slots = [], []
        for position, image, result in zip(positions, images, maps, strict=True):
            disparity = result["predicted_depth"].float().cpu().numpy()
            here = {track: mask for (track, p), mask in masks.items() if p == position and mask.sum() > 0}
            medians = {track: float(np.median(disparity[mask])) for track, mask in here.items()}
            contact = {}
            for a, b in combinations(sorted(here), 2):
                diff = contact_difference(here[a], here[b], disparity)
                if diff is not None:
                    contact[f"{a}|{b}"] = round(diff, 5)
            rows[str(position)] = {"range": round(disparity_range(disparity), 5),
                                   "median": {t: round(v, 5) for t, v in medians.items()}, "contact": contact}
            frame = np.asarray(image)
            for track, mask in here.items():
                crops.append(Image.fromarray(engine._character_canvas(frame, mask)))
                slots.append(f"{track}@{position}")
        vectors = engine._embed_batch(crops, dino_crop_batch_size=dino_batch) if crops else []
        _write_new(out_json, {"depth": rows, "dino": {k: [round(x, 5) for x in v]
                                                      for k, v in zip(slots, vectors, strict=True)}})
    elapsed = time.perf_counter() - started
    print(json.dumps({"frames": frames_done, "seconds": round(elapsed, 1)}))
    return 0


# ---------------------------------------------------------------------------
# people（GPU，person_tools）
# ---------------------------------------------------------------------------


def run_people(args: argparse.Namespace) -> int:
    import numpy as np
    import onnxruntime
    import torch  # noqa: F401  先载入 torch 自带的 CUDA/cuDNN 库，再由 onnxruntime 预载（ORT 官方做法）
    from PIL import Image

    onnxruntime.preload_dlls()
    from insightface.app import FaceAnalysis
    from rtmlib import RTMPose
    from torchreid.utils import FeatureExtractor

    device = args.device
    providers = ["CUDAExecutionProvider"] if device.startswith("cuda") else ["CPUExecutionProvider"]
    # RTMW-x 384x288（rtmlib Wholebody performance 档的同一 ONNX）；输入按 SDK pipeline.json 的 to_rgb=true 给 RGB。
    pose = RTMPose(str(args.rtmw), model_input_size=(288, 384), backend="onnxruntime",
                   device="cuda" if device.startswith("cuda") else "cpu")
    # insightface 官方用法：FaceAnalysis(name="buffalo_l") + prepare(ctx_id=0)，检测尺寸用默认；只开检测与识别两个模块。
    faces = FaceAnalysis(name="buffalo_l", root=str(args.insightface_root), providers=providers,
                         allowed_modules=["detection", "recognition"])
    faces.prepare(ctx_id=0 if device.startswith("cuda") else -1)
    # torchreid 官方 FeatureExtractor：输入 RGB numpy，默认 256×128、ImageNet 均值方差。
    reid = FeatureExtractor(model_name="osnet_ain_x1_0", model_path=str(args.osnet), device=device, verbose=False)
    started = time.perf_counter()
    for story_id, _window, shot in _shots(args.work, args.stories):
        out_json = args.work / "people" / story_id / f"{shot['shot_id']}.json"
        if out_json.exists():
            continue
        record, masks = _key_masks(args.work, story_id, shot)
        rows: dict[str, Any] = {}
        for position in record["key_positions"]:
            rgb = np.asarray(Image.open(_frame_path(args.work, story_id, shot["shot_id"], position)).convert("RGB"))
            here = {track: row["frames"][str(position)][:4] for track, row in record["tracks"].items()
                    if str(position) in row["frames"] and record["tracks"][track]["concept"] == "person"}
            tracks = sorted(here)
            entry: dict[str, Any] = {"pose": {}, "pose_mirror": {}, "face": {}, "reid": {}}
            if tracks:
                boxes = [here[t] for t in tracks]
                kpts, scores = pose(rgb, bboxes=boxes)
                mirrored = rgb[:, ::-1].copy()
                width = rgb.shape[1]
                kpts_m, scores_m = pose(mirrored, bboxes=[[width - b[2], b[1], width - b[0], b[3]] for b in boxes])
                for i, track in enumerate(tracks):
                    entry["pose"][track] = {"xy": np.round(kpts[i], 1).tolist(), "score": np.round(scores[i], 3).tolist()}
                    entry["pose_mirror"][track] = {"xy": np.round(kpts_m[i], 1).tolist(),
                                                   "score": np.round(scores_m[i], 3).tolist()}
                crops = []
                for track in tracks:
                    x0, y0, x1, y1 = (int(round(v)) for v in here[track])
                    crops.append(rgb[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)])
                features = reid(crops).cpu().numpy()
                for track, vector in zip(tracks, features, strict=True):
                    entry["reid"][track] = np.round(vector / np.linalg.norm(vector), 5).tolist()
                found = faces.get(rgb[:, :, ::-1].copy())  # insightface 吃 BGR（cv2.imread）
                for face in found:
                    cx, cy = (face.bbox[0] + face.bbox[2]) / 2, (face.bbox[1] + face.bbox[3]) / 2
                    owners = [t for t in tracks if (t, position) in masks
                              and masks[(t, position)][min(int(cy), rgb.shape[0] - 1), min(int(cx), rgb.shape[1] - 1)]]
                    if len(owners) != 1:
                        continue
                    previous = entry["face"].get(owners[0])
                    if previous is None or float(face.det_score) > previous["det_score"]:
                        entry["face"][owners[0]] = {"det_score": round(float(face.det_score), 4),
                                                    "bbox": [round(float(v), 1) for v in face.bbox],
                                                    "embedding": np.round(face.normed_embedding, 5).tolist()}
            rows[str(position)] = entry
        _write_new(out_json, rows)
    print(json.dumps({"seconds": round(time.perf_counter() - started, 1)}))
    return 0


# ---------------------------------------------------------------------------
# bind（CPU 客户端 + 本地 Qwen3.8-27B）
# ---------------------------------------------------------------------------

_COLORS = ((230, 57, 70), (29, 120, 220), (40, 170, 90), (240, 160, 20), (160, 60, 200), (0, 190, 190))


def _jpeg_data_url(image: Any, *, quality: int = 92) -> str:
    import base64
    import io

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _local_chat(base_url: str, model: str, content: list[dict[str, Any]], response_format: Mapping[str, Any], *,
                max_tokens: int, timeout: float) -> dict[str, Any]:
    """本地 vLLM（OpenAI 兼容）一次请求；温度 0、非思考；返回原文与用量。"""

    import urllib.request

    body = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0,
            "max_tokens": max_tokens, "response_format": dict(response_format),
            "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    choice = payload["choices"][0]
    return {"text": choice["message"]["content"], "finish_reason": choice["finish_reason"], "usage": payload.get("usage")}


def binding_image(path: Path, boxes: Sequence[Sequence[float]]) -> Any:
    """中帧上画编号实例框（粗框、大号码），给 VLM 指认。"""

    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=28)
    for number, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        color = _COLORS[(number - 1) % len(_COLORS)]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
        label = str(number)
        draw.rectangle((x0, y0, x0 + 26 + 16 * (len(label) - 1), y0 + 34), fill=color)
        draw.text((x0 + 6, y0 + 2), label, fill=(255, 255, 255), font=font)
    return image


def run_bind(args: argparse.Namespace) -> int:
    from concurrent.futures import ThreadPoolExecutor

    from mspc.data.spatial_facts import phrase_binding
    from mspc.evaluation.cross_shot_layout import binding_prompt, binding_schema, parse_binding

    out = args.work / "bind.json"
    if out.exists():
        raise FileExistsError(out)
    jobs, results = [], {}
    for story_id, _window, shot in _shots(args.work, args.stories):
        record = _sam3_record(args.work, story_id, shot, phrases=True)
        phrases = {s: record["phrases"][s]["phrase"] for s in shot["animate"]}
        votes = {s: record["phrases"][s]["votes"] for s in shot["animate"]}
        by_phrase = phrase_binding(votes, phrases)
        mid = shot["layout_positions"][1]
        remaining_subjects = [s for s in shot["animate"] if s not in by_phrase]
        remaining_tracks = [t for t, row in sorted(record["tracks"].items())
                            if str(mid) in row["frames"] and t not in by_phrase.values()]
        results[(story_id, shot["shot_id"])] = {"phrase": by_phrase, "phrases": phrases, "vlm": None,
                                                "vlm_tracks": remaining_tracks, "vlm_subjects": remaining_subjects}
        if remaining_subjects and remaining_tracks:
            boxes = [record["tracks"][t]["frames"][str(mid)][:4] for t in remaining_tracks]
            jobs.append((story_id, shot["shot_id"], mid, remaining_subjects, remaining_tracks, boxes,
                         [(s, shot["attributes"][s]) for s in remaining_subjects]))

    def work(job: tuple) -> tuple[tuple[str, str], dict[str, Any]]:
        story_id, shot_id, mid, subject_ids, tracks, boxes, described = job
        image = binding_image(_frame_path(args.work, story_id, shot_id, mid), boxes)
        content = [{"type": "image_url", "image_url": {"url": _jpeg_data_url(image)}},
                   {"type": "text", "text": binding_prompt(described)}]
        answer = _local_chat(args.base_url, args.model, content, binding_schema(subject_ids, len(boxes)),
                             max_tokens=2048, timeout=600)
        try:
            picked = parse_binding(answer["text"], subject_ids, len(boxes))
        except json.JSONDecodeError as error:
            # 实测（判别第 1 步 S-LR 出片）：个别回答退化到写满 max_tokens，JSON 被截断。这一镜交兜底的主体都记未指认，
            # 原文与结束原因留档，不让整批指认作废。
            return (story_id, shot_id), {"raw": answer["text"], "usage": answer["usage"], "unparsed": str(error),
                                         "finish_reason": answer["finish_reason"],
                                         "binding": {s: None for s in subject_ids}}
        return (story_id, shot_id), {"raw": answer["text"], "usage": answer["usage"],
                                     "binding": {s: (None if b is None else tracks[b - 1]) for s, b in picked.items()}}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for done, (key, row) in enumerate(pool.map(work, jobs), start=1):
            results[key]["vlm"] = row
            if done % 25 == 0 or done == len(jobs):
                print(f"[bind] {done}/{len(jobs)} {time.perf_counter() - started:.0f}s", file=sys.stderr, flush=True)
    stories: dict[str, Any] = {}
    for (story_id, shot_id), row in sorted(results.items()):
        binding = {s: {"track": t, "source": "phrase"} for s, t in row["phrase"].items()}
        for s in row["vlm_subjects"]:
            track = None if row["vlm"] is None else row["vlm"]["binding"].get(s)
            source = "no_instance_left" if row["vlm"] is None else "vlm_unparsed" if "unparsed" in row["vlm"] else "vlm"
            binding[s] = {"track": track, "source": source}
        stories.setdefault(story_id, {})[shot_id] = {**row, "binding": binding}
    unparsed = sorted(f"{story_id}/{shot_id}" for (story_id, shot_id), row in results.items()
                      if row["vlm"] is not None and "unparsed" in row["vlm"])
    _write_new(out, {"created_at": _now(), "model": args.model, "vlm_calls": len(jobs), "temperature": 0,
                     "vlm_unparsed": unparsed, "stories": stories})
    print(json.dumps({"shots": len(results), "vlm_calls": len(jobs), "vlm_unparsed": unparsed}))
    return 0


# ---------------------------------------------------------------------------
# 读数共用：逐镜测量
# ---------------------------------------------------------------------------


def _shot_inputs(work: Path, story_id: str, shot: Mapping[str, Any]) -> dict[str, Any]:
    return {"sam3": _sam3_record(work, story_id, shot, phrases=False),
            "depth_dino": _read(work / "depth_dino" / story_id / f"{shot['shot_id']}.json"),
            "people": _read(work / "people" / story_id / f"{shot['shot_id']}.json")}


def _frame_orientation(inputs: Mapping[str, Any], track: str, position: int, *, orient: Mapping[str, float],
                       mirror: bool = False) -> str | None:
    import numpy as np

    from mspc.data.spatial_facts import orientation_from_keypoints

    row = inputs["people"][str(position)]["pose_mirror" if mirror else "pose"].get(track)
    if row is None:
        return None
    box = inputs["sam3"]["tracks"][track]["frames"][str(position)]
    return orientation_from_keypoints(np.asarray(row["xy"]), np.asarray(row["score"]), box_width=box[2] - box[0],
                                      face_visible=orient["face_visible"], face_back=orient["face_back"],
                                      lateral=orient["lateral"])


def _frame_depth(depth_row: Mapping[str, Any], binding: Mapping[str, str], *, gap: float) -> dict[str, str]:
    from mspc.data.spatial_facts import depth_labels, depth_relations

    medians = {s: depth_row["median"][t] for s, t in binding.items() if t in depth_row["median"]}
    if len(medians) < 2:
        return {}
    track_to_subject = {t: s for s, t in binding.items()}
    contact = {}
    for key, diff in depth_row["contact"].items():
        a, b = key.split("|")
        if a in track_to_subject and b in track_to_subject:
            sa, sb = track_to_subject[a], track_to_subject[b]
            contact[(sa, sb) if sa < sb else (sb, sa)] = diff if sa < sb else -diff
    relations = depth_relations(medians, contact, frame_range=depth_row["range"], gap=gap)
    return depth_labels(relations, sorted(medians))


def shot_facts(inputs: Mapping[str, Any], binding: Mapping[str, str], *, width: int, gap: float,
               orient: Mapping[str, float]) -> dict[str, Any]:
    """一镜的事实：布局三帧多数（horizontal 用三帧框中心中位数）、首帧与末帧各自的单帧事实（三轴），以及首末两端
    各 5 帧投票的 horizontal（并入事件日志用这一项）。"""

    import numpy as np

    from mspc.data.spatial_facts import (
        box_center_x,
        endpoint_horizontal,
        endpoint_positions,
        horizontal_labels,
        majority,
    )

    sam3 = inputs["sam3"]
    tracks = sam3["tracks"]

    def frame_facts(position: int) -> dict[str, dict[str, Any]]:
        visible = {s: t for s, t in binding.items() if str(position) in tracks[t]["frames"]}
        centers = {s: box_center_x(tracks[t]["frames"][str(position)]) for s, t in visible.items()}
        horizontal = horizontal_labels(centers, width=width)
        depth = _frame_depth(inputs["depth_dino"]["depth"][str(position)], visible, gap=gap)
        return {s: {"horizontal": horizontal.get(s), "depth": depth.get(s),
                    "orientation": _frame_orientation(inputs, t, position, orient=orient)}
                for s, t in visible.items()}

    layout_rows = [frame_facts(p) for p in sam3["layout_positions"]]
    centers = {}
    for s, t in binding.items():
        xs = [box_center_x(tracks[t]["frames"][str(p)]) for p in sam3["layout_positions"] if str(p) in tracks[t]["frames"]]
        if len(xs) >= 2:
            centers[s] = float(np.median(xs))
    horizontal = horizontal_labels(centers, width=width)
    layout = {}
    for s in binding:
        # depth 每帧一票：该帧看得见本人且至少两人可比时投 depth 或 "none"，否则不投；三帧过半才算。
        depth_votes = [(row[s]["depth"] or "none") if s in row and len(row) >= 2 else None for row in layout_rows]
        depth = majority(depth_votes)
        layout[s] = {"horizontal": horizontal.get(s), "depth": None if depth == "none" else depth,
                     "orientation": majority([row.get(s, {}).get("orientation") for row in layout_rows])}
    first, last = 0, sam3["frame_count"] - 1
    endpoints = {}
    for name, positions in endpoint_positions(sam3["frame_count"]).items():
        centers = [{s: box_center_x(tracks[t]["frames"][str(p)]) for s, t in binding.items() if str(p) in tracks[t]["frames"]}
                   for p in positions]
        endpoints[name] = {"positions": list(positions), "horizontal": endpoint_horizontal(centers, width=width)}
    return {"layout": layout, "first": frame_facts(first), "last": frame_facts(last), "endpoints": endpoints,
            "visible_first": {s: str(first) in tracks[t]["frames"] for s, t in binding.items()},
            "visible_last": {s: str(last) in tracks[t]["frames"] for s, t in binding.items()}}


def _instance_vectors(inputs: Mapping[str, Any], track: str) -> dict[str, list[float]]:
    """一个实例在关键帧上三路嵌入各自的单位化均值。"""

    from mspc.data.spatial_facts import unit_mean

    people, dino = inputs["people"], inputs["depth_dino"]["dino"]
    faces = [row["face"][track]["embedding"] for row in people.values() if track in row["face"]]
    reids = [row["reid"][track] for row in people.values() if track in row["reid"]]
    dinos = [vec for key, vec in dino.items() if key.split("@")[0] == track]
    out = {}
    for path, vectors in (("face", faces), ("reid", reids), ("dino", dinos)):
        mean = unit_mean(vectors)
        if mean is not None:
            out[path] = mean
    return out


# ---------------------------------------------------------------------------
# 金标对照（dev 30）
# ---------------------------------------------------------------------------


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def gold_tracks(sam3: Mapping[str, Any], oracle_shot: Mapping[str, Any], gold_shot: Mapping[str, Any]) -> dict[str, str]:
    """金标主体 -> 新链实例：旧链绑定（未被判错绑）的三帧框与新链布局三帧的框 IoU ≥ 0.5 至少两帧，一对一取最大。"""

    from mspc.data.spatial_facts import assign_one_to_one

    gain = {}
    for subject, row in oracle_shot.get("subjects", {}).items():
        if subject in gold_shot["binding_wrong"] or subject not in gold_shot["layout"]:
            continue
        for track, data in sam3["tracks"].items():
            ious = [_box_iou(old, data["frames"][str(p)][:4]) for old, p in zip(row["boxes"], sam3["layout_positions"])
                    if old is not None and str(p) in data["frames"]]
            hits = sum(iou >= 0.5 for iou in ious)
            if hits >= 2:
                gain[(subject, track)] = float(sum(ious))
    return {str(s): str(t) for s, t in assign_one_to_one(gain, min_gain=0.0).items()}


def _split(story_ids: Sequence[str]) -> tuple[set[str], set[str]]:
    ordered = sorted(story_ids)
    return set(ordered[0::2]), set(ordered[1::2])


def _identity_pairs(work: Path, shots: Sequence[tuple[str, Mapping, Mapping]], labels: Mapping) -> list[dict[str, Any]]:
    """同窗不同镜的实例对：三路余弦与金标同人与否（两端都有金标主体时）。"""

    from mspc.data.spatial_facts import cosine

    by_story: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for story_id, _window, shot in shots:
        inputs = _shot_inputs(work, story_id, shot)
        vectors = {t: _instance_vectors(inputs, t) for t in inputs["sam3"]["tracks"]}
        by_story.setdefault(story_id, []).append((shot["shot_id"], vectors))
    rows = []
    for story_id, items in by_story.items():
        for (shot_a, vec_a), (shot_b, vec_b) in combinations(items, 2):
            for ta, va in vec_a.items():
                for tb, vb in vec_b.items():
                    sims = {p: cosine(va[p], vb[p]) for p in PATHS if p in va and p in vb}
                    sa = labels.get((story_id, shot_a), {}).get(ta)
                    sb = labels.get((story_id, shot_b), {}).get(tb)
                    rows.append({"story": story_id, "a": (shot_a, ta), "b": (shot_b, tb), "sims": sims,
                                 "same": None if sa is None or sb is None else sa == sb})
    return rows


def _gold_labels(work: Path, shots: Sequence, oracle: Mapping, gold: Mapping) -> dict[tuple[str, str], dict[str, str]]:
    """``{(窗, 镜): {新链实例: 金标主体}}``。"""

    out = {}
    for story_id, _window, shot in shots:
        sam3 = _sam3_record(work, story_id, shot, phrases=False)
        mapping = gold_tracks(sam3, oracle["stories"][story_id][shot["shot_id"]], gold["stories"][story_id][shot["shot_id"]])
        out[(story_id, shot["shot_id"])] = {t: s for s, t in mapping.items()}
    return out


# ---------------------------------------------------------------------------
# calibrate（CPU）
# ---------------------------------------------------------------------------


def run_calibrate(args: argparse.Namespace) -> int:
    import math

    from mspc.data.spatial_facts import fit_logistic

    out = args.work / "calibration.json"
    if out.exists():
        raise FileExistsError(out)
    oracle, gold = _read(args.oracle), _read(args.gold)
    shots = _shots(args.work, "")
    calib, _test = _split({s for s, _, _ in shots})
    shots = [row for row in shots if row[0] in calib]
    labels = _gold_labels(args.work, shots, oracle, gold)
    # 前后与朝向：在标定半上用金标绑定（排除指认误差）逐主体比，网格取准确率最高的。
    scores: dict[str, dict[Any, list[int]]] = {"depth_gap": {}, "orientation": {}}
    orient_grid = [(f, fb, lat) for f in CALIBRATION_GRID["face_visible"] for fb in CALIBRATION_GRID["face_back"]
                   for lat in CALIBRATION_GRID["lateral"]]
    for story_id, _window, shot in shots:
        inputs = _shot_inputs(args.work, story_id, shot)
        binding = {s: t for t, s in labels[(story_id, shot["shot_id"])].items()}
        gold_layout = gold["stories"][story_id][shot["shot_id"]]["layout"]
        for gap in CALIBRATION_GRID["depth_gap"]:
            facts = shot_facts(inputs, binding, width=FRAME_WIDTH, gap=gap,
                               orient={"face_visible": 3.0, "face_back": 6.0, "lateral": 0.15})["layout"]
            if len(binding) >= 2:
                for s in binding:
                    scores["depth_gap"].setdefault(gap, []).append(int(facts[s]["depth"] == gold_layout[s].get("depth")))
        for face_visible, face_back, lateral in orient_grid:
            facts = shot_facts(inputs, binding, width=FRAME_WIDTH, gap=0.3,
                               orient={"face_visible": face_visible, "face_back": face_back, "lateral": lateral})["layout"]
            for s in binding:
                if gold_layout[s].get("orientation") is not None:
                    scores["orientation"].setdefault((face_visible, face_back, lateral), []).append(
                        int(facts[s]["orientation"] == gold_layout[s]["orientation"]))
    # 朝向按金标有朝向的全部主体计分（判不了算错），取准确率最高者；平局取更小的阈值。
    best = {name: max(grid, key=lambda v: (sum(grid[v]) / max(1, len(grid[v])), tuple(-x for x in (v if isinstance(v, tuple) else (v,)))))
            for name, grid in scores.items()}
    chosen = {"depth_gap": best["depth_gap"], "face_visible": best["orientation"][0], "face_back": best["orientation"][1],
              "lateral": best["orientation"][2]}
    pairs = [row for row in _identity_pairs(args.work, shots, labels) if row["same"] is not None]
    paths = {}
    for path in PATHS:
        xs = [row["sims"][path] for row in pairs if path in row["sims"]]
        ys = [int(row["same"]) for row in pairs if path in row["sims"]]
        paths[path] = {"fit": fit_logistic(xs, ys), "n": len(xs), "positives": sum(ys)}
    base = sum(int(row["same"]) for row in pairs) / len(pairs)
    _write_new(out, {"created_at": _now(), "calibration_windows": sorted(calib), "chosen": chosen,
                     "grid_accuracy": {name: {str(v): [sum(r), len(r)] for v, r in grid.items()}
                                       for name, grid in scores.items()},
                     "identity": {"prior": math.log(base / (1 - base)), "base_rate": base, "pairs": len(pairs),
                                  "paths": {p: v["fit"] for p, v in paths.items()},
                                  "path_counts": {p: [v["n"], v["positives"]] for p, v in paths.items()}}})
    print(json.dumps({"chosen": chosen, "pairs": len(pairs)}))
    return 0


# ---------------------------------------------------------------------------
# facts（CPU）
# ---------------------------------------------------------------------------


def run_facts(args: argparse.Namespace) -> int:
    from mspc.data.spatial_facts import cross_shot_matches, fused_log_odds

    out = args.work / "facts.json"
    if out.exists():
        raise FileExistsError(out)
    calibration = _read(args.calibration)
    gap = calibration["chosen"]["depth_gap"]
    orient = {k: calibration["chosen"][k] for k in ("face_visible", "face_back", "lateral")}
    bind = _read(args.work / "bind.json")["stories"]
    stories: dict[str, Any] = {}
    vectors: dict[str, dict[str, dict[str, Any]]] = {}
    present: dict[str, dict[str, set[str]]] = {}
    for story_id, _window, shot in _shots(args.work, args.stories):
        inputs = _shot_inputs(args.work, story_id, shot)
        binding = {s: row["track"] for s, row in bind[story_id][shot["shot_id"]]["binding"].items()
                   if row["track"] is not None}
        stories.setdefault(story_id, {})[shot["shot_id"]] = {
            "binding": bind[story_id][shot["shot_id"]]["binding"],
            **shot_facts(inputs, binding, width=FRAME_WIDTH, gap=gap, orient=orient)}
        vectors.setdefault(story_id, {})[shot["shot_id"]] = {t: _instance_vectors(inputs, t)
                                                             for t in inputs["sam3"]["tracks"]}
        present.setdefault(story_id, {})[shot["shot_id"]] = set(shot["animate"])
    identity = {}
    from mspc.data.spatial_facts import cosine

    for story_id, shots in vectors.items():
        rows = []
        for shot_a, shot_b in combinations(sorted(shots), 2):
            log_odds = {}
            for ta, va in shots[shot_a].items():
                for tb, vb in shots[shot_b].items():
                    sims = {p: cosine(va[p], vb[p]) for p in PATHS if p in va and p in vb}
                    value = fused_log_odds(sims, calibration["identity"])
                    if value is not None:
                        log_odds[(ta, tb)] = value
            limit = len(present[story_id][shot_a] & present[story_id][shot_b])
            matches = cross_shot_matches(log_odds, limit=limit)
            rows.append({"shots": [shot_a, shot_b], "matches": {a: b for a, b in matches.items()},
                         "log_odds": {f"{a}|{b}": round(log_odds[(a, b)], 3) for a, b in matches.items()}})
        identity[story_id] = rows
    _write_new(out, {"created_at": _now(), "calibration": str(args.calibration),
                     "depth_gap": gap, "orient": orient, "stories": stories, "identity": identity})
    print(json.dumps({"windows": len(stories)}))
    return 0


# ---------------------------------------------------------------------------
# evaluate（CPU）
# ---------------------------------------------------------------------------


def _rate(hit: int, total: int) -> dict[str, Any]:
    from mspc.evaluation.spatial_layout_probe import wilson_interval

    low, high = wilson_interval(hit, total) if total else (None, None)
    return {"hit": hit, "total": total, "rate": None if not total else round(hit / total, 4),
            "wilson95": None if low is None else [round(low, 4), round(high, 4)]}


def run_evaluate(args: argparse.Namespace) -> int:
    from mspc.data.spatial_facts import MIRROR_ORIENTATION

    out = args.work / "evaluation.json"
    if out.exists():
        raise FileExistsError(out)
    oracle, gold, facts = _read(args.oracle), _read(args.gold), _read(args.work / "facts.json")
    calibration = _read(args.work / "calibration.json")
    shots = _shots(args.work, "")
    calib, test = _split({s for s, _, _ in shots})
    labels = _gold_labels(args.work, shots, oracle, gold)
    report: dict[str, Any] = {}
    for half_name, half in (("test", test), ("all", calib | test)):
        tally: dict[str, list[int]] = {}

        def add(name: str, ok: bool) -> None:
            tally.setdefault(name, [0, 0])
            tally[name][0] += int(ok)
            tally[name][1] += 1

        for story_id, _window, shot in shots:
            if story_id not in half:
                continue
            shot_id = shot["shot_id"]
            g = gold["stories"][story_id][shot_id]
            new = facts["stories"][story_id][shot_id]
            old = oracle["stories"][story_id][shot_id]
            truth = {s: t for t, s in labels[(story_id, shot_id)].items()}
            # 镜内指认：金标可核的主体（旧链框未判错绑）。
            for s in shot["animate"]:
                bound = new["binding"].get(s, {}).get("track")
                source = new["binding"].get(s, {}).get("source")
                if s in truth:
                    add("bind_coverage", bound is not None)
                    if bound is not None:
                        add("bind_accuracy", bound == truth[s])
                        add(f"bind_accuracy_{source}", bound == truth[s])
                elif s not in g["layout"] and bound is not None:
                    add("bind_extra_when_gold_absent", True)
            # 逐字段：两边都给了该主体时比；depth 只在多人镜里比（金标不写＝无前后）。
            multi = len(g["layout"]) >= 2
            for s, gl in g["layout"].items():
                nl = new["layout"].get(s)
                ol = old["layout"].get(s)
                for field in ("horizontal", "orientation"):
                    if gl.get(field) is None:
                        continue
                    add(f"new_{field}_coverage", nl is not None and nl.get(field) is not None)
                    add(f"old_{field}_coverage", ol is not None and ol.get(field) is not None)
                    if nl is not None and nl.get(field) is not None:
                        add(f"new_{field}", nl[field] == gl[field])
                    if ol is not None and ol.get(field) is not None:
                        add(f"old_{field}", ol[field] == gl[field])
                if multi:
                    if nl is not None:
                        add("new_depth", nl.get("depth") == gl.get("depth"))
                    if ol is not None:
                        add("old_depth", ol.get("depth") == gl.get("depth"))
            # 朝向镜像等变（新链，布局帧逐帧）。
            inputs = _shot_inputs(args.work, story_id, shot)
            for s, row in new["binding"].items():
                track = row["track"]
                if track is None:
                    continue
                for p in inputs["sam3"]["layout_positions"]:
                    if str(p) not in inputs["people"] or track not in inputs["people"][str(p)]["pose"]:
                        continue
                    a = _frame_orientation(inputs, track, p, orient=facts["orient"])
                    b = _frame_orientation(inputs, track, p, orient=facts["orient"], mirror=True)
                    if a in ("screen-left", "screen-right") or b in ("screen-left", "screen-right"):
                        add("orientation_mirror_equivariance", b == MIRROR_ORIENTATION.get(a))
        # 跨镜认人：新链匹配对 vs 金标。
        for story_id, rows in facts["identity"].items():
            if story_id not in half:
                continue
            for row in rows:
                shot_a, shot_b = row["shots"]
                la, lb = labels[(story_id, shot_a)], labels[(story_id, shot_b)]
                for ta, tb in row["matches"].items():
                    if ta in la and tb in lb:
                        add("identity_precision", la[ta] == lb[tb])
                for ta, sa in la.items():
                    if sa in lb.values():
                        tb = next(t for t, s in lb.items() if s == sa)
                        add("identity_recall", row["matches"].get(ta) == tb)
        report[half_name] = {name: _rate(*v) for name, v in sorted(tally.items())}
    _write_new(out, {"created_at": _now(), "calibration_windows": calibration["calibration_windows"],
                     "test_windows": sorted(test), "depth_gap": facts["depth_gap"], "orient": facts["orient"],
                     "readouts": report})
    print(json.dumps(report["test"], ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# screen-merge（CPU）
# ---------------------------------------------------------------------------


def run_screen_merge(args: argparse.Namespace) -> int:
    """v10 事件日志 + 本链端点事实 -> 换掉人物与动物 screen 类事实的事件日志（逐窗过 ``validate_event_log``）。

    并入规则（依盲标金标准定，``merged_screen_layout``）：左右（首端 5 帧投票）照并；前后（首帧，只在同帧两个
    已指认主体之间才有）照并；朝向只并 toward camera。event_log_v3 的 screen 事实一镜一组、语义是首帧所见（v10 合同），
    所以首端进日志；没并入的全部写旁车 ``<out>.sidecar.json``：``{case_id: {shot_id: {"first": 首帧三轴原样,
    "last": 末端 horizontal 投票＋末帧三轴}}}``，等编译器有末帧槽位或朝向换工具后再接。"""

    from mspc.data.event_log import parse_event_log, validate_event_log
    from mspc.data.spatial_facts import merged_screen_layout, replace_screen_facts

    sidecar = args.out.with_name(args.out.name + ".sidecar.json")
    if args.out.exists() or sidecar.exists():
        raise FileExistsError(args.out)
    windows = _read(args.work / "windows.json")["windows"]
    facts = _read(args.work / "facts.json")["stories"]
    by_case = {window["case_id"]: window_id for window_id, window in windows.items()}
    rows, side = [], {}
    counts = {"windows": 0, "facts_before": 0, "facts_after": 0, "horizontal": 0, "depth": 0, "orientation": 0}
    with open(args.event_logs, encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            window_id = by_case[entry["case_id"]]
            layouts, kept = [], {}
            for shot in entry["event_log"]["shots"]:
                row = facts.get(window_id, {}).get(shot["shot_id"])
                if row is None:
                    layouts.append({})
                    kept[shot["shot_id"]] = {"first": {}, "last": {}}
                    continue
                layout = merged_screen_layout(row["endpoints"]["first"]["horizontal"], row["first"])
                layouts.append(layout)
                for values in layout.values():
                    for axis in values:
                        counts[axis] += 1
                kept[shot["shot_id"]] = {
                    "first": row["first"],
                    "last": {"horizontal_vote": row["endpoints"]["last"]["horizontal"], "frame": row["last"]}}
            merged = replace_screen_facts(entry["event_log"], layouts)
            validate_event_log(parse_event_log(merged))
            counts["windows"] += 1
            counts["facts_before"] += sum(len(s["screen_facts"]) for s in entry["event_log"]["shots"])
            counts["facts_after"] += sum(len(s["screen_facts"]) for s in merged["shots"])
            rows.append({"case_id": entry["case_id"], "window_id": entry["window_id"], "event_log": merged})
            side[entry["case_id"]] = kept
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".partial")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(tmp, args.out)
    _write_new(sidecar, side)
    print(json.dumps(counts))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)

    def add(name: str, handler: Any, *, stories: bool = True) -> argparse.ArgumentParser:
        p = steps.add_parser(name)
        p.add_argument("--work", type=Path, required=True)
        if stories:
            p.add_argument("--stories", default="", help="逗号分隔；空为全部")
        p.set_defaults(handler=handler)
        return p

    p = add("frames", run_frames, stories=False)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--candidate-r3", type=Path, required=True)
    p = add("frames-render", run_frames_render, stories=False)
    p.add_argument("--results", type=Path, nargs="+", required=True,
                   help="历史空间诊断的 results.json（含 aligned_shots；不是当前benchmark的requests.json）；分片出片时给全部分片，合起来须恰好覆盖全部窗")
    p.add_argument("--template", type=Path, required=True, help="同一批窗真片的 windows.json（主体、在场名单、每镜帧数）")
    p = add("frames-pool", run_frames_pool, stories=False)
    p.add_argument("--rows", type=Path, required=True, help="v6 修订镜界的 final2500/rows.jsonl")
    p.add_argument("--case-ids", type=Path, required=True, help="JSON 列表")
    p.add_argument("--workers", type=int, required=True)
    p = add("attach-logs", run_attach_logs, stories=False)
    p.add_argument("--event-logs", type=Path, required=True, help="v10 编译的 event_logs.jsonl（每行 case_id、window_id、event_log）")
    p = add("sam3", run_sam3)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bpe", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p = add("sam3-phrase", run_sam3_phrase)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bpe", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p = add("depth-dino", run_depth_dino)
    p.add_argument("--depth-model", type=Path, required=True)
    p.add_argument("--resource-root", type=Path, required=True)
    p.add_argument("--device", required=True)
    p = add("people", run_people)
    p.add_argument("--rtmw", type=Path, required=True, help="RTMW-x 384x288 end2end.onnx")
    p.add_argument("--insightface-root", type=Path, required=True)
    p.add_argument("--osnet", type=Path, required=True)
    p.add_argument("--device", required=True)
    p = add("bind", run_bind)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p = add("calibrate", run_calibrate, stories=False)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p = add("facts", run_facts)
    p.add_argument("--calibration", type=Path, required=True, help="dev 30 标定半给的 calibration.json")
    p = add("evaluate", run_evaluate, stories=False)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p = add("screen-merge", run_screen_merge, stories=False)
    p.add_argument("--event-logs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
