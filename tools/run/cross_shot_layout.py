#!/usr/bin/env python3
"""跨镜空间布局判别的薄入口。
规则（纯函数）在 :mod:`mspc.evaluation.cross_shot_layout`。

子命令按执行面分，GPU 步骤各用自己的环境（命令行里给解释器）：

* ``frames``（CPU，renderer_public_metrics 环境）：真片按修订 3 镜界解码；每镜取 30/50/70% 三帧，裁切后缩到 832×480
  存 JPEG，另存水平翻转的一份（Q1 镜像坏例）。写 ``plan.json``。生成片不走这里：按镜切帧与人物抽取都在
  ``tools/run/spatial_facts.py``（``frames-render`` 起）。
* ``sam3``（GPU，sam3_sm80 环境）：SAM 3.1 文本概念提示＋跟踪。每镜一个「person」（有动物时另加「animal」）概念
  给实例，每个在场主体的描述各一个概念给逐主体得分（掩码 IoU ≥ 0.5 归到实例上）。写 ``instances.json`` 与掩码。
* ``embed``（GPU，renderer_public_metrics 环境）：每个实例的 DINOv2 掩码画布嵌入（跨镜认人），中帧「a head」头框
  （Q4 景别，沿用 Z1 的 Grounding DINO 头框量法）。
* 逐镜 oracle 布局（左右、前后、朝向）的抽取已由 ``tools/run/spatial_facts.py``（规则在 :mod:`mspc.data.spatial_facts`）
  取代：原来的 ``depth``（MoGe-2）、``depth-da2``、``oracle`` 三步与 1.5 倍前后规则已删；判别实验第 0 步的
  ``oracle.json``、金标仍在主服务器原处，供本工具的读数步骤读。
* ``gdino``（GPU，renderer_public_metrics 环境）：旧方案（Grounding DINO 描述短语＋框阈值指派），只用于召回对照。
* ``compare-sheet``（CPU）：新旧两方案的框画在同一帧上，供看帧数召回。
* 第 1 步（生成片各臂）读数，输入是 ``tools/run/spatial_facts.py`` 的工作目录（每份出片一个，真片一个）：
  ``heads``（GPU，renderer_public_metrics）中帧头框；
  ``step1-readout``（CPU，wan22_fun）Q1 一致率、可判率、逐镜服从率、提及顺序、在场、前后与 Q4 的描述性读数
  （相邻镜对的背景几何 Q3 已撤：同一量法读真片也是 0/133），
  各臂配对比较（按窗聚类 bootstrap）与 §3.6 预写判定；``step1-sheets``（CPU）亲看用的镜对并排帧；
"""

from __future__ import annotations

import argparse
import json
import math
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

PLAN_SCHEMA = "cross_shot_layout_plan_v1"
INSTANCES_SCHEMA = "cross_shot_layout_instances_v1"
FRAME_WIDTH, FRAME_HEIGHT = 832, 480
VARIANTS = ("orig", "mirror")
#: SAM 3.1 概念提示：人物与动物的通用概念给实例；逐主体描述给得分。
GENERIC_CONCEPT = {"person": "person", "animal": "animal"}
#: 描述概念的检出归到实例上的掩码 IoU 下限（与 200 场试点的匹配操作点不同：这里是同一帧同一人两次分割的重合）。
DESCRIPTION_MATCH_IOU = 0.5


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


# ---------------------------------------------------------------------------
# frames (CPU)
# ---------------------------------------------------------------------------


def _story_subjects(script: Mapping[str, Any]) -> list[dict[str, Any]]:
    from mspc.evaluation.cross_shot_layout import detector_phrase

    return [{"subject_id": row["subject_id"], "kind": row["kind"], "attributes": row["attributes"],
             "phrase": detector_phrase(row["attributes"])} for row in script["subjects"]]


def _save_pair(image: Any, path: Path, mirror_path: Path) -> None:
    from PIL import ImageOps

    path.parent.mkdir(parents=True, exist_ok=True)
    mirror_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=95)
    ImageOps.mirror(image).save(mirror_path, format="JPEG", quality=95)


def run_frames(args: argparse.Namespace) -> int:
    from PIL import Image

    from mspc.data.msm_global_frames import default_msm_frame_reader
    from mspc.evaluation.cross_shot_layout import animate_subjects, source_offsets

    out = args.out_dir
    if (out / "plan.json").exists():
        raise FileExistsError(out / "plan.json")
    pairs = {pair["story_id"]: pair for pair in _read(args.pairs)["pairs"]}
    freeze = _read(args.freeze)
    dev = [w["window_id"] for w in freeze["windows"] if w["split"] == "dev"]
    if set(dev) != set(pairs):
        raise ValueError(f"{args.pairs} does not cover exactly the frozen dev windows")
    manifest = _read(args.candidate_r3)
    if manifest["render_revision"]["revision"] != 3:
        raise ValueError("candidate manifest is not render revision 3")
    candidate = {story["story_id"]: story for story in manifest["stories"]}
    stories: dict[str, Any] = {}
    for story_id in sorted(dev):
        script = pairs[story_id]["context_b"]
        windows = candidate[story_id]
        if len(windows["shots"]) != len(script["shots"]):
            raise ValueError(f"{story_id}: {len(windows['shots'])} r3 shots vs {len(script['shots'])} script shots")
        rows = []
        indices = [source_offsets(s["start_frame"], s["end_frame"]) for s in windows["shots"]]
        flat = [index + windows["container_preroll_frames"] for trio in indices for index in trio]
        images = default_msm_frame_reader(Path(windows["source_video_path"]), flat)
        x0, y0, x1, y1 = windows["crop_box"]
        images = [image.crop((x0, y0, x1, y1)).resize((FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.LANCZOS)
                  for image in images]
        for k, (shot, window_shot, trio) in enumerate(zip(script["shots"], windows["shots"], indices, strict=True)):
            paths, mirrors = [], []
            for j, image in enumerate(images[3 * k:3 * k + 3]):
                path = out / "frames" / "orig" / story_id / f"{shot['shot_id']}_{j}.jpg"
                mirror = out / "frames" / "mirror" / story_id / f"{shot['shot_id']}_{j}.jpg"
                _save_pair(image, path, mirror)
                paths.append(str(path))
                mirrors.append(str(mirror))
            rows.append({"shot_id": shot["shot_id"], "location_id": shot["location_id"],
                         "present": list(shot["present_subjects"]), "animate": animate_subjects(script, shot),
                         "render_frames": window_shot["render_frames"], "frame_indices": list(trio),
                         "frames": {"orig": paths, "mirror": mirrors}})
        stories[story_id] = {"subjects": _story_subjects(script), "shots": rows}
        print(f"[frames] {story_id} {len(rows)} shots", file=sys.stderr, flush=True)
    _write_new(out / "plan.json", {
        "schema_version": PLAN_SCHEMA, "source": "real", "created_at": _now(),
        "pairs": str(args.pairs), "freeze": str(args.freeze), "candidate_r3": str(args.candidate_r3),
        "width": FRAME_WIDTH, "height": FRAME_HEIGHT, "fractions": [0.3, 0.5, 0.7], "stories": stories,
    })
    print(json.dumps({"stories": len(stories), "shots": sum(len(s["shots"]) for s in stories.values())}))
    return 0


# ---------------------------------------------------------------------------
# sam3 (GPU, sam3_sm80)
# ---------------------------------------------------------------------------


def _unpack_mask(packed: Any, shape: Any) -> Any:
    import numpy as np

    count = int(shape[0]) * int(shape[1])
    return np.unpackbits(packed)[:count].reshape(int(shape[0]), int(shape[1])).astype(bool)


def _mask_iou(a: Any, b: Any) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else 0.0


def _session(model: Any, frame_paths: Sequence[str], text: str, scratch: Path) -> list[dict[str, Any]]:
    """一个概念在一镜三帧上的 SAM 3.1 会话：每帧的检出（像素框、概率、掩码）。"""

    import numpy as np

    scratch.mkdir(parents=True, exist_ok=False)
    try:
        for index, path in enumerate(frame_paths):
            os.symlink(path, scratch / f"{index:05d}.jpg")
        mask_path = scratch / "masks.npz"
        frames = model.track_frame_directory(frame_dir=str(scratch), text=text, mask_path=str(mask_path),
                                             hotstart_delay=0, hotstart_unmatch_thresh=0)
        if sorted(row["frame_index"] for row in frames) != list(range(len(frame_paths))):
            raise ValueError(f"SAM 3.1 did not return every frame exactly once for {text!r}")
        packed = np.load(mask_path)
        out: list[list[dict[str, Any]]] = [[] for _ in frame_paths]
        for row in frames:
            for det in row["detections"]:
                mask = _unpack_mask(packed[det["mask_key"]], packed[det["mask_key"] + "_shape"])
                x0, y0, x1, y1 = (float(v) / 1000.0 for v in det["box"])
                h, w = mask.shape
                out[row["frame_index"]].append({
                    "track_id": det["track_id"], "score": det["score"],
                    "box": [x0 * w, y0 * h, x1 * w, y1 * h], "mask": mask,
                })
        return out
    finally:
        for child in scratch.iterdir():
            child.unlink()
        scratch.rmdir()


def shot_instances(model: Any, shot: Mapping[str, Any], subjects: Mapping[str, Mapping[str, Any]],
                   variant: str, scratch: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """一镜（一种变体）的实例：通用概念给实例，逐主体描述给得分。返回 (三帧实例表, 掩码字典)。"""

    import numpy as np

    animate = shot["animate"]
    paths = shot["frames"][variant]
    kinds = sorted({subjects[s]["kind"] for s in animate})
    generic = {kind: _session(model, paths, GENERIC_CONCEPT[kind], scratch / f"g-{kind}") for kind in kinds}
    described = {s: _session(model, paths, subjects[s]["phrase"], scratch / f"d-{s.split()[-1]}") for s in animate}
    frames, masks = [], {}
    for f in range(len(paths)):
        instances: list[dict[str, Any]] = []
        for kind in kinds:
            for det in generic[kind][f]:
                instances.append({"concept": kind, "track_id": det["track_id"], "score": det["score"],
                                  "box": det["box"], "mask": det["mask"], "scores": {}})
        for subject in animate:
            for det in described[subject][f]:
                overlaps = [(_mask_iou(det["mask"], inst["mask"]), i) for i, inst in enumerate(instances)]
                best = max(overlaps, default=(0.0, -1))
                if best[0] >= DESCRIPTION_MATCH_IOU:
                    inst = instances[best[1]]
                else:
                    inst = {"concept": f"desc:{subject}", "track_id": det["track_id"], "score": det["score"],
                            "box": det["box"], "mask": det["mask"], "scores": {}}
                    instances.append(inst)
                inst["scores"][subject] = max(inst["scores"].get(subject, 0.0), float(det["score"]))
        rows = []
        for i, inst in enumerate(instances):
            key = f"{variant}_f{f}_i{i}"
            masks[key] = np.packbits(inst["mask"])
            masks[key + "_shape"] = np.asarray(inst["mask"].shape)
            rows.append({"iid": i, "concept": inst["concept"], "track_id": inst["track_id"],
                         "score": round(float(inst["score"]), 4), "box": [round(v, 2) for v in inst["box"]],
                         "area": round(float(inst["mask"].mean()), 5), "mask_key": key,
                         "phrase_scores": [round(inst["scores"].get(s, 0.0), 4) for s in animate]})
        frames.append({"instances": rows})
    return frames, masks


def run_sam3(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    from mspc.data.grounding.sam3_backend import Sam3Backend

    plan = _read(args.plan)
    if args.out.exists():
        raise FileExistsError(args.out)
    mask_dir = args.out.parent / (args.out.stem + "_masks")
    mask_dir.mkdir(parents=True, exist_ok=True)
    selected = None if not args.stories else set(args.stories.split(","))
    torch.set_num_threads(4)
    started = time.perf_counter()
    model = Sam3Backend(checkpoint_path=str(args.checkpoint), bpe_path=str(args.bpe), version="sam3.1",
                        objects_per_forward=16, enable_object_multiplexing=True, use_fa3=False, use_rope_real=True)
    load_seconds = time.perf_counter() - started
    stories: dict[str, Any] = {}
    scratch_root = args.scratch / f"sam3-{os.getpid()}"
    for story_id, story in sorted(plan["stories"].items()):
        if selected is not None and story_id not in selected:
            continue
        subjects = {row["subject_id"]: dict(row) for row in story["subjects"]}
        if args.phrase_mode == "short":
            from mspc.data.spatial_facts import short_phrase

            for row in subjects.values():
                row["phrase"] = short_phrase(row["attributes"], row["kind"])
        shots = {}
        for shot in story["shots"]:
            if not shot["animate"]:
                shots[shot["shot_id"]] = {"animate": [], "variants": {}}
                continue
            variants, masks = {}, {}
            tick = time.perf_counter()
            for variant in args.variants.split(","):
                frames, packed = shot_instances(model, shot, subjects, variant,
                                                scratch_root / story_id / shot["shot_id"] / variant)
                variants[variant] = frames
                masks.update(packed)
            mask_path = mask_dir / f"{story_id}_{shot['shot_id']}.npz"
            np.savez_compressed(mask_path, **masks)
            shots[shot["shot_id"]] = {"animate": shot["animate"], "variants": variants, "mask_path": str(mask_path),
                                      "seconds": round(time.perf_counter() - tick, 2)}
        stories[story_id] = shots
        print(f"[sam3] {story_id} {len(shots)} shots {time.perf_counter() - started:.0f}s", file=sys.stderr, flush=True)
    _write_new(args.out, {"schema_version": INSTANCES_SCHEMA, "plan": str(args.plan), "created_at": _now(),
                          "checkpoint": str(args.checkpoint), "load_seconds": round(load_seconds, 1),
                          "generic_concepts": GENERIC_CONCEPT, "description_match_iou": DESCRIPTION_MATCH_IOU,
                          "phrase_mode": args.phrase_mode,
                          "variants": args.variants.split(","), "elapsed_seconds": round(time.perf_counter() - started, 1),
                          "stories": stories})
    print(json.dumps({"stories": len(stories), "out": str(args.out)}))
    return 0


# ---------------------------------------------------------------------------
# embed (GPU, renderer_public_metrics)：DINOv2 掩码画布、中帧头框
# ---------------------------------------------------------------------------


def run_embed(args: argparse.Namespace) -> int:
    import numpy as np
    from PIL import Image

    from mspc.evaluation.renderer_qualification_official_backend import (
        DEFAULT_ROI_DETECTOR_CONFIG,
        FROZEN_PUBLIC_ROI_DETECTOR,
        _DinoRoiEngine,
        _model_batch_from_roi_config,
    )
    from mspc.evaluation.spatial_layout_probe import HEAD_PHRASE

    if args.out.exists():
        raise FileExistsError(args.out)
    plan = _read(args.plan)
    instances = _read(args.instances)
    batch = _model_batch_from_roi_config(DEFAULT_ROI_DETECTOR_CONFIG)
    engine = _DinoRoiEngine(args.resource_root, args.device)
    started = time.perf_counter()
    stories: dict[str, Any] = {}
    for story_id, shots in sorted(instances["stories"].items()):
        plan_shots = {shot["shot_id"]: shot for shot in plan["stories"][story_id]["shots"]}
        out_shots = {}
        for shot_id, shot in shots.items():
            if not shot["animate"]:
                continue
            packed = np.load(shot["mask_path"])
            out_variants = {}
            for variant, frames in shot["variants"].items():
                paths = plan_shots[shot_id]["frames"][variant]
                images = [np.asarray(Image.open(path).convert("RGB")) for path in paths]
                crops, slots = [], []
                for f, frame in enumerate(frames):
                    for inst in frame["instances"]:
                        mask = _unpack_mask(packed[inst["mask_key"]], packed[inst["mask_key"] + "_shape"])
                        if mask.sum() == 0:
                            continue
                        crops.append(Image.fromarray(engine._character_canvas(images[f], mask)))
                        slots.append(inst["mask_key"])
                vectors = engine._embed_batch(crops, dino_crop_batch_size=batch.dino_crop_batch_size) if crops else []
                (heads,) = engine.grounded_regions([images[1]], [HEAD_PHRASE], FROZEN_PUBLIC_ROI_DETECTOR,
                                                   detector_frame_batch_size=1,
                                                   dino_crop_batch_size=batch.dino_crop_batch_size)
                engine.clear_frame_caches()
                out_variants[variant] = {
                    "embeddings": {key: [round(v, 5) for v in vec] for key, vec in zip(slots, vectors, strict=True)},
                    "heads_mid": [{"box": [round(v, 2) for v in h["box"]], "score": round(h["score"], 4)} for h in heads],
                }
            out_shots[shot_id] = out_variants
        stories[story_id] = out_shots
        print(f"[embed] {story_id} {time.perf_counter() - started:.0f}s", file=sys.stderr, flush=True)
    _write_new(args.out, {"instances": str(args.instances), "created_at": _now(), "head_phrase": HEAD_PHRASE,
                          "roi_detector": FROZEN_PUBLIC_ROI_DETECTOR.payload(), "stories": stories})
    return 0


# ---------------------------------------------------------------------------
# gdino (GPU)：旧方案，只作召回对照
# ---------------------------------------------------------------------------


def run_gdino(args: argparse.Namespace) -> int:
    import numpy as np
    from PIL import Image

    from mspc.evaluation.cross_shot_layout import assign_subjects
    from mspc.evaluation.renderer_qualification_official_backend import (
        DEFAULT_ROI_DETECTOR_CONFIG,
        FROZEN_PUBLIC_ROI_DETECTOR,
        _DinoRoiEngine,
        _model_batch_from_roi_config,
    )

    plan = _read(args.plan)
    selected = set(args.stories.split(","))
    batch = _model_batch_from_roi_config(DEFAULT_ROI_DETECTOR_CONFIG)
    engine = _DinoRoiEngine(args.resource_root, args.device)
    stories = {}
    for story_id in sorted(selected):
        story = plan["stories"][story_id]
        phrases = {row["subject_id"]: row["phrase"] for row in story["subjects"]}
        shots = {}
        for shot in story["shots"]:
            if not shot["animate"]:
                continue
            images = [np.asarray(Image.open(p).convert("RGB")) for p in shot["frames"]["orig"]]
            regions = engine.grounded_regions(images, [phrases[s] for s in shot["animate"]], FROZEN_PUBLIC_ROI_DETECTOR,
                                              detector_frame_batch_size=batch.detector_frame_batch_size,
                                              dino_crop_batch_size=batch.dino_crop_batch_size)
            engine.clear_frame_caches()
            frames = []
            for frame_regions in regions:
                owner = assign_subjects(frame_regions, shot["animate"], min_phrase_score=FROZEN_PUBLIC_ROI_DETECTOR.box_threshold)
                frames.append({"boxes": [[round(v, 1) for v in r["box"]] for r in frame_regions],
                               "owner": {s: (None if i is None else [round(v, 1) for v in frame_regions[i]["box"]])
                                         for s, i in owner.items()}})
            shots[shot["shot_id"]] = frames
        stories[story_id] = shots
    _write_new(args.out, {"plan": str(args.plan), "stories": stories})
    return 0


# ---------------------------------------------------------------------------
# compare-sheet (CPU)
# ---------------------------------------------------------------------------

_COLORS = ((230, 57, 70), (29, 120, 220), (40, 170, 90), (240, 160, 20), (160, 60, 200), (0, 190, 190))


def run_compare_sheet(args: argparse.Namespace) -> int:
    from PIL import Image, ImageDraw

    plan = _read(args.plan)
    sam = _read(args.instances)
    gd = _read(args.gdino)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tile_w = 416
    tile_h = int(FRAME_HEIGHT * tile_w / FRAME_WIDTH)
    for story_id, shots in sorted(gd["stories"].items()):
        story = plan["stories"][story_id]
        plan_shots = {s["shot_id"]: s for s in story["shots"]}
        rows = []
        for shot_id, gframes in shots.items():
            shot = plan_shots[shot_id]
            sframes = sam["stories"][story_id][shot_id]["variants"]["orig"]
            for scheme, frames in (("GDINO", gframes), ("SAM3", sframes)):
                tiles = []
                for f, path in enumerate(shot["frames"]["orig"]):
                    image = Image.open(path).convert("RGB")
                    draw = ImageDraw.Draw(image)
                    if scheme == "GDINO":
                        for box in frames[f]["boxes"]:
                            draw.rectangle(box, outline=(150, 150, 150), width=1)
                        owner = frames[f]["owner"]
                    else:
                        # 召回对照只画实例：每个 SAM 3.1 实例一个框，标跟踪号与概率。
                        for inst in frames[f]["instances"]:
                            color = _COLORS[int(inst["track_id"]) % len(_COLORS)]
                            draw.rectangle(inst["box"], outline=color, width=4)
                            draw.text((inst["box"][0] + 6, inst["box"][1] + 6),
                                      f"t{inst['track_id']} {inst['concept'][:6]} {inst['score']:.2f}", fill=color)
                        owner = {}
                    for k, subject in enumerate(shot["animate"]):
                        box = owner.get(subject)
                        if box is None:
                            continue
                        color = _COLORS[int(subject.split()[-1]) % len(_COLORS)]
                        draw.rectangle(box, outline=color, width=4)
                        draw.text((box[0] + 6, box[1] + 6), f"S{subject.split()[-1]}", fill=color)
                    draw.text((6, FRAME_HEIGHT - 20), f"{story_id} {shot_id} {scheme} f{f}", fill=(255, 255, 255))
                    tiles.append(image.resize((tile_w, tile_h)))
                row = Image.new("RGB", (tile_w * 3, tile_h))
                for i, tile in enumerate(tiles):
                    row.paste(tile, (i * tile_w, 0))
                rows.append(row)
        for start in range(0, len(rows), 8):
            chunk = rows[start:start + 8]
            sheet = Image.new("RGB", (tile_w * 3, tile_h * len(chunk)))
            for i, row in enumerate(chunk):
                sheet.paste(row, (0, i * tile_h))
            sheet.save(args.out_dir / f"{story_id}_{start // 8:02d}.jpg", quality=85)
    return 0


# ---------------------------------------------------------------------------
# 公共：读图、画框、发本地 VLM 请求
# ---------------------------------------------------------------------------


def _font(size: int) -> Any:
    from PIL import ImageFont

    return ImageFont.load_default(size=size)


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


def _plan_shot(plan: Mapping[str, Any], story_id: str, shot_id: str) -> Mapping[str, Any]:
    return next(shot for shot in plan["stories"][story_id]["shots"] if shot["shot_id"] == shot_id)


# ---------------------------------------------------------------------------
# bind（CPU 客户端 + 本地 Qwen3.8-27B）：镜内绑定
# ---------------------------------------------------------------------------


def binding_image(path: str, instances: Sequence[Mapping[str, Any]]) -> Any:
    """中帧上画编号实例框（粗框、大号码），给 VLM 指认。"""

    from PIL import Image, ImageDraw

    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _font(28)
    for number, inst in enumerate(instances, start=1):
        color = _COLORS[(number - 1) % len(_COLORS)]
        x0, y0, x1, y1 = inst["box"]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
        label = str(number)
        draw.rectangle((x0, y0, x0 + 26 + 16 * (len(label) - 1), y0 + 34), fill=color)
        draw.text((x0 + 6, y0 + 2), label, fill=(255, 255, 255), font=font)
    return image


def run_bind(args: argparse.Namespace) -> int:
    from concurrent.futures import ThreadPoolExecutor

    from mspc.evaluation.cross_shot_layout import (
        MID_FRAME,
        binding_prompt,
        binding_schema,
        instance_key,
        parse_binding,
    )

    if args.out.exists():
        raise FileExistsError(args.out)
    plan = _read(args.plan)
    instances = _read(args.instances)
    jobs = []
    for story_id, shots in sorted(instances["stories"].items()):
        subjects = {row["subject_id"]: row for row in plan["stories"][story_id]["subjects"]}
        for shot_id, shot in shots.items():
            if not shot["animate"]:
                continue
            for variant in args.variants.split(","):
                jobs.append((story_id, shot_id, variant, shot["animate"], subjects,
                             shot["variants"][variant][MID_FRAME]["instances"]))

    def work(job: tuple) -> tuple[str, str, str, dict[str, Any]]:
        story_id, shot_id, variant, animate, subjects, mid = job
        keys = [instance_key(inst) for inst in mid]
        if not mid:
            return story_id, shot_id, variant, {"boxes": [], "binding": {s: None for s in animate}, "raw": None}
        image = binding_image(_plan_shot(plan, story_id, shot_id)["frames"][variant][MID_FRAME], mid)
        prompt = binding_prompt([(s, subjects[s]["attributes"]) for s in animate])
        content = [{"type": "image_url", "image_url": {"url": _jpeg_data_url(image)}}, {"type": "text", "text": prompt}]
        result = _local_chat(args.base_url, args.model, content, binding_schema(animate, len(mid)),
                             max_tokens=2048, timeout=600)
        picked = parse_binding(result["text"], animate, len(mid))
        return story_id, shot_id, variant, {"boxes": keys, "raw": result["text"], "usage": result["usage"],
                                            "binding": {s: (None if b is None else keys[b - 1]) for s, b in picked.items()}}

    started = time.perf_counter()
    stories: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for done, (story_id, shot_id, variant, row) in enumerate(pool.map(work, jobs), start=1):
            stories.setdefault(story_id, {}).setdefault(shot_id, {})[variant] = row
            if done % 25 == 0 or done == len(jobs):
                print(f"[bind] {done}/{len(jobs)} {time.perf_counter() - started:.0f}s", file=sys.stderr, flush=True)
    _write_new(args.out, {"plan": str(args.plan), "instances": str(args.instances), "model": args.model,
                          "created_at": _now(), "prompt_kind": "binding", "temperature": 0, "stories": stories})
    return 0


# ---------------------------------------------------------------------------
# oracle（CPU）：绑定、框、中心、深度、头框 -> 逐镜布局
# ---------------------------------------------------------------------------


def shot_measure(plan_shot: Mapping[str, Any], inst_shot: Mapping[str, Any], bind_row: Mapping[str, Any] | None,
                 embed_shot: Mapping[str, Any] | None, *, width: int) -> dict[str, Any]:
    """一镜一种变体的测量：每个在场主体的三帧框、中心、中帧头框、嵌入（供 Q1、Q4 共用）。"""

    from mspc.evaluation.cross_shot_layout import (
        MID_FRAME,
        bound_tracks,
        median_center,
    )
    from mspc.evaluation.spatial_layout_probe import subject_head

    animate = plan_shot["animate"]
    binding = {s: None for s in animate} if bind_row is None else dict(bind_row["binding"])
    embeddings = {} if embed_shot is None else embed_shot["embeddings"]
    tracks = bound_tracks(inst_shot, binding, embeddings)
    heads = [] if embed_shot is None else embed_shot["heads_mid"]
    subjects = {}
    for subject in animate:
        mid_box = tracks["boxes"][MID_FRAME][subject]
        subjects[subject] = {
            "instance": binding[subject],
            "boxes": [row[subject] for row in tracks["boxes"]],
            "center": median_center(tracks, subject),
            "head_mid": subject_head(mid_box, heads),
            "embedding": tracks["embeddings"][subject],
        }
    return {"tracks": tracks, "subjects": subjects}


# ---------------------------------------------------------------------------
# crops（CPU）：朝向裁剪（原图与翻转各一份）
# ---------------------------------------------------------------------------

CROP_LONG_SIDE = 448
ORIENTATION_SEED = 42


def subject_crop(frame_path: str, box: Sequence[float], *, width: int, height: int) -> Any:
    """中帧上框外扩 20% 的裁剪，原框画细红框，长边缩放到 448（小图放大）。"""

    from PIL import Image, ImageDraw

    from mspc.evaluation.cross_shot_layout import expand_box

    image = Image.open(frame_path).convert("RGB")
    x0, y0, x1, y1 = expand_box(box, width=width, height=height)
    crop = image.crop((x0, y0, x1, y1))
    ImageDraw.Draw(crop).rectangle((box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0), outline=(255, 0, 0), width=2)
    scale = CROP_LONG_SIDE / max(crop.size)
    return crop.resize((max(1, round(crop.size[0] * scale)), max(1, round(crop.size[1] * scale))), Image.Resampling.LANCZOS)


def run_crops(args: argparse.Namespace) -> int:
    from PIL import ImageOps

    from mspc.evaluation.cross_shot_layout import MID_FRAME, crop_flipped

    out = args.out_dir / "crops.json"
    if out.exists():
        raise FileExistsError(out)
    plan, oracle = _read(args.plan), _read(args.oracle)
    rows = []
    for story_id, shots in sorted(oracle["stories"].items()):
        for shot_id, shot in shots.items():
            for subject, row in shot["subjects"].items():
                box = row["boxes"][MID_FRAME]
                if box is None:
                    continue
                instance_id = f"{story_id}|{shot_id}|{subject}"
                crop = subject_crop(_plan_shot(plan, story_id, shot_id)["frames"]["orig"][MID_FRAME], box,
                                    width=plan["width"], height=plan["height"])
                stem = args.out_dir / "crops" / story_id / f"{shot_id}_{subject.split()[-1]}"
                stem.parent.mkdir(parents=True, exist_ok=True)
                crop.save(f"{stem}_orig.jpg", quality=92)
                ImageOps.mirror(crop).save(f"{stem}_flip.jpg", quality=92)
                rows.append({"instance_id": instance_id, "story_id": story_id, "shot_id": shot_id, "subject": subject,
                             "box": box, "paths": {"orig": f"{stem}_orig.jpg", "flip": f"{stem}_flip.jpg"},
                             "primary_flipped": crop_flipped(ORIENTATION_SEED, instance_id)})
    _write_new(out, {"oracle": str(args.oracle), "seed": ORIENTATION_SEED, "long_side": CROP_LONG_SIDE, "crops": rows})
    print(json.dumps({"crops": len(rows)}))
    return 0


def _orientation_items(crops: Sequence[Mapping[str, Any]], seed: int) -> list[tuple[str, str, str]]:
    """全部 (实例, 版本, 路径)，按种子打乱（同一实例的原图与翻转图不总在同一批）。"""

    import random

    items = [(row["instance_id"], version, row["paths"][version]) for row in crops for version in ("orig", "flip")]
    random.Random(seed).shuffle(items)
    return items


def run_orient_local(args: argparse.Namespace) -> int:
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    from mspc.evaluation.cross_shot_layout import ORIENTATION_PROMPT, orientation_schema, parse_orientation

    if args.out.exists():
        raise FileExistsError(args.out)
    crops = _read(args.crops)["crops"]
    items = _orientation_items(crops, args.shuffle_seed)
    batches = [items[i:i + args.batch] for i in range(0, len(items), args.batch)]

    def work(batch: Sequence[tuple[str, str, str]]) -> list[tuple[str, str, str, str]]:
        content: list[dict[str, Any]] = []
        for index, (_, _, path) in enumerate(batch, start=1):
            content.append({"type": "text", "text": f"image {index}/{len(batch)}"})
            content.append({"type": "image_url", "image_url": {"url": _jpeg_data_url(Image.open(path))}})
        content.append({"type": "text", "text": ORIENTATION_PROMPT})
        result = _local_chat(args.base_url, args.model, content, orientation_schema(len(batch)), max_tokens=4096, timeout=900)
        answers = parse_orientation(result["text"], len(batch))
        evidence = [row["evidence"] for row in json.loads(result["text"])["images"]]
        return [(iid, version, answer, ev) for (iid, version, _), answer, ev in zip(batch, answers, evidence, strict=True)]

    answers: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for rows in pool.map(work, batches):
            for iid, version, answer, evidence in rows:
                answers.setdefault(iid, {})[version] = answer
                answers[iid][version + "_evidence"] = evidence
    _write_new(args.out, {"crops": str(args.crops), "model": args.model, "batch": args.batch,
                          "shuffle_seed": args.shuffle_seed, "created_at": _now(), "answers": answers})
    print(json.dumps({"instances": len(answers), "batches": len(batches)}))
    return 0


# ---------------------------------------------------------------------------
# orient-judge（OpenRouter J2）：朝向判官，两遍，镜像随机化
# ---------------------------------------------------------------------------


def _judge_client(config: Mapping[str, Any], *, api_key_file: Path, proxy: str, already_spent: str) -> Any:
    from mspc.data.openrouter_client import build_openrouter_client, load_openrouter_api_key

    return build_openrouter_client(
        api_key=load_openrouter_api_key(api_key_file),
        pricing={"base_url": config["openrouter_base_url"], "model": config["model"],
                 "hard_cap_usd": config["budget_hard_cap_usd"], "input_usd_per_mtok": config["input_usd_per_mtok"],
                 "output_usd_per_mtok": config["output_usd_per_mtok"]},
        precall_prompt_tokens=config["precall_prompt_tokens"], precall_completion_tokens=config["precall_completion_tokens"],
        timeout_seconds=config["request_timeout_seconds"], proxy=proxy, already_spent_usd=already_spent,
    )


def _spent(rows: Sequence[Mapping[str, Any]]) -> str:
    from decimal import Decimal

    return str(sum((Decimal(row["cost_usd"]) for row in rows), Decimal("0")))


def run_orient_judge(args: argparse.Namespace) -> int:
    """J2 判朝向：一遍里每个实例的原图与翻转图都判；按 ``--pass-seed`` 打乱后每批 ``batch_size`` 张。

    逐批写进 ``<out>.calls.jsonl``（含费用），重跑同一命令只补没判的批；``--limit-calls`` 用于探针。
    """

    from mspc.evaluation.cross_shot_layout import ORIENTATION_PROMPT, orientation_schema, parse_orientation

    config = _read(args.config)
    crops = _read(args.crops)["crops"]
    items = _orientation_items(crops, args.pass_seed)
    batches = [items[i:i + config["batch_size"]] for i in range(0, len(items), config["batch_size"])]
    log_path = args.out.with_name(args.out.name + ".calls.jsonl")
    done = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
    finished = {row["batch"] for row in done if row["status"] == "ok"}
    ledger = [json.loads(line) for line in args.ledger.read_text().splitlines()] if args.ledger.exists() else []
    client = _judge_client(config, api_key_file=args.api_key_file, proxy=args.proxy, already_spent=_spent(ledger))
    calls = 0
    for number, batch in enumerate(batches):
        if number in finished:
            continue
        if args.limit_calls is not None and calls >= args.limit_calls:
            break
        from PIL import Image

        urls = [_jpeg_data_url(Image.open(path)) for _, _, path in batch]
        tick = time.perf_counter()
        response = client.complete_images(model=config["model"], image_data_urls=urls, prompt=ORIENTATION_PROMPT,
                                          temperature=config["temperature"], max_tokens=config["max_tokens"],
                                          request_options=config["request_options"],
                                          response_format=orientation_schema(len(batch)))
        calls += 1
        row = {"batch": number, "items": [[iid, version] for iid, version, _ in batch], "raw": response.text,
               "cost_usd": response.cost_usd, "usage": response.usage, "served": response.served,
               "seconds": round(time.perf_counter() - tick, 2), "at": _now()}
        try:
            row["answers"] = parse_orientation(response.text, len(batch))
            row["status"] = "ok"
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            row["status"], row["error"] = "format_failed", str(exc)
        with log_path.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with args.ledger.open("a") as handle:
            handle.write(json.dumps({"source": str(log_path), "batch": number, "cost_usd": response.cost_usd,
                                     "at": row["at"]}) + "\n")
        print(f"[judge] batch {number + 1}/{len(batches)} {row['status']} ${response.cost_usd} "
              f"spent ${client.cost_counter.snapshot()['spent_usd']}", file=sys.stderr, flush=True)
    rows = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
    ok = {row["batch"]: row for row in rows if row["status"] == "ok"}
    if len(ok) == len(batches):
        answers: dict[str, dict[str, Any]] = {}
        for row in ok.values():
            for (iid, version), answer in zip(row["items"], row["answers"], strict=True):
                answers.setdefault(iid, {})[version] = answer
        _write_new(args.out, {"crops": str(args.crops), "config": str(args.config), "pass_seed": args.pass_seed,
                              "batches": len(batches), "cost_usd": _spent(rows), "created_at": _now(),
                              "answers": answers})
    print(json.dumps({"batches": len(batches), "ok": len(ok), "calls_this_run": calls, "cost_usd_this_pass": _spent(rows)}))
    return 0


# ---------------------------------------------------------------------------
# sheets（CPU）：oracle 标注图，供人工逐镜定金标
# ---------------------------------------------------------------------------


def run_sheets(args: argparse.Namespace) -> int:
    from PIL import Image, ImageDraw

    from mspc.evaluation.cross_shot_layout import MID_FRAME, instance_key

    plan, oracle, inst = _read(args.plan), _read(args.oracle), _read(args.instances)
    crops = {row["instance_id"]: row for row in _read(args.crops)["crops"]} if args.crops else {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h = 448, 258
    font, small = _font(18), _font(15)
    rows = []
    for story_id, shots in sorted(oracle["stories"].items()):
        for shot in plan["stories"][story_id]["shots"]:
            entry = shots[shot["shot_id"]]
            if not entry["animate"]:
                continue
            frames = inst["stories"][story_id][shot["shot_id"]]["variants"]["orig"]
            tiles = []
            for f, path in enumerate(shot["frames"]["orig"]):
                image = Image.open(path).convert("RGB")
                draw = ImageDraw.Draw(image)
                bound = {row["instance"] for row in entry["subjects"].values()}
                for number, instance in enumerate(frames[f]["instances"], start=1):
                    if instance_key(instance) not in bound:
                        draw.rectangle(instance["box"], outline=(200, 200, 200), width=2)
                        draw.text((instance["box"][0] + 4, instance["box"][3] - 22), instance_key(instance), fill=(220, 220, 220), font=small)
                for subject, row in entry["subjects"].items():
                    box = row["boxes"][f]
                    if box is None:
                        continue
                    color = _COLORS[int(subject.split()[-1]) % len(_COLORS)]
                    draw.rectangle(box, outline=color, width=5)
                    draw.rectangle((box[0], box[1], box[0] + 52, box[1] + 26), fill=color)
                    draw.text((box[0] + 5, box[1] + 3), f"S{subject.split()[-1]}", fill=(255, 255, 255), font=font)
                if f == MID_FRAME:
                    for subject, row in entry["subjects"].items():
                        if row["head_mid"] is not None:
                            draw.rectangle(row["head_mid"], outline=(255, 255, 255), width=1)
                tiles.append(image.resize((tile_w, tile_h)))
            crop_tiles = []
            for subject in entry["animate"]:
                row = crops.get(f"{story_id}|{shot['shot_id']}|{subject}")
                if row is not None:
                    crop = Image.open(row["paths"]["orig"]).convert("RGB")
                    crop.thumbnail((180, tile_h - 24))
                    crop_tiles.append((subject, crop))
            text_lines = [f"{story_id} {shot['shot_id']}  {shot['location_id']}  present={','.join(shot['present'])}"]
            for subject in entry["animate"]:
                row, lay = entry["subjects"][subject], entry["layout"].get(subject, {})
                z = "-" if row["depth"] is None else f"{row['depth']:.2f}"
                text_lines.append(f"S{subject.split()[-1]} [{row['instance']}] h={lay.get('horizontal', '-')} "
                                  f"d={lay.get('depth', '-')}(z={z}) o={lay.get('orientation', '-')}")
            width = 3 * tile_w + 180 * max(1, len(crop_tiles))
            canvas = Image.new("RGB", (width, tile_h + 22 * len(text_lines) + 6), (20, 20, 20))
            for i, tile in enumerate(tiles):
                canvas.paste(tile, (i * tile_w, 0))
            draw = ImageDraw.Draw(canvas)
            for i, (subject, crop) in enumerate(crop_tiles):
                canvas.paste(crop, (3 * tile_w + 180 * i, 24))
                draw.text((3 * tile_w + 180 * i + 4, 2), f"S{subject.split()[-1]}", fill=(255, 255, 0), font=font)
            for i, line in enumerate(text_lines):
                draw.text((6, tile_h + 4 + 22 * i), line, fill=(255, 255, 255), font=font)
            rows.append(canvas)
    per = args.shots_per_sheet
    for start in range(0, len(rows), per):
        chunk = rows[start:start + per]
        sheet = Image.new("RGB", (max(r.size[0] for r in chunk), sum(r.size[1] for r in chunk) + 4 * len(chunk)), (60, 60, 60))
        y = 0
        for row in chunk:
            sheet.paste(row, (0, y))
            y += row.size[1] + 4
        sheet.save(args.out_dir / f"sheet_{start // per:03d}.jpg", quality=88)
    print(json.dumps({"shots": len(rows), "sheets": math.ceil(len(rows) / per)}))
    return 0


# ---------------------------------------------------------------------------
# gold（CPU）：看帧改错 -> 金标；改动数即抽取器精度
# ---------------------------------------------------------------------------

LAYOUT_AXES = ("horizontal", "depth", "orientation")


def apply_corrections(oracle: Mapping[str, Any], corrections: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """oracle 逐镜布局 + 改错表 -> (金标, 改动计数)。

    改错表键 ``"<story>/<shot>"``，值可含 ``horizontal`` / ``depth`` / ``orientation``（``{主体: 值或 null}``，null 即删去该轴）、
    ``binding_wrong``（工具框落在别人身上的主体）、``reviewed``（必须为 true）。没有出现的镜视为看过且全对；
    ``reviewed_shots`` 由调用方另给，必须覆盖全部有人物的镜。
    """

    gold: dict[str, Any] = {}
    changes = {axis: 0 for axis in LAYOUT_AXES}
    changes.update({"binding_wrong": 0, "shots_changed": 0, "subjects": 0})
    for story_id, shots in oracle["stories"].items():
        for shot_id, entry in shots.items():
            key = f"{story_id}/{shot_id}"
            fix = corrections.get(key, {})
            unknown = set(fix) - {*LAYOUT_AXES, "binding_wrong", "note"}
            if unknown:
                raise ValueError(f"{key}: unknown correction fields {sorted(unknown)}")
            layout = {subject: dict(values) for subject, values in entry["layout"].items()}
            changed = False
            for axis in LAYOUT_AXES:
                for subject, value in fix.get(axis, {}).items():
                    if subject not in entry["animate"]:
                        raise ValueError(f"{key}: {subject} is not an animate subject of this shot")
                    before = layout.get(subject, {}).get(axis)
                    if value == before:
                        raise ValueError(f"{key}: correction {axis}={value!r} for {subject} repeats the oracle value")
                    changes[axis] += 1
                    changed = True
                    if value is None:
                        layout.get(subject, {}).pop(axis, None)
                    else:
                        layout.setdefault(subject, {})[axis] = value
            wrong = list(fix.get("binding_wrong", []))
            changes["binding_wrong"] += len(wrong)
            changes["shots_changed"] += int(changed or bool(wrong))
            changes["subjects"] += len(entry["animate"])
            gold.setdefault(story_id, {})[shot_id] = {"animate": entry["animate"],
                                                      "layout": {s: v for s, v in layout.items() if v},
                                                      "binding_wrong": wrong, "note": fix.get("note")}
    return gold, changes


def run_gold(args: argparse.Namespace) -> int:
    if args.out.exists():
        raise FileExistsError(args.out)
    oracle = _read(args.oracle)
    corrections = _read(args.corrections)
    missing = [key for key in corrections["shots"] if key.split("/")[0] not in oracle["stories"]]
    if missing:
        raise ValueError(f"corrections name unknown stories: {missing[:5]}")
    reviewed = set(corrections["reviewed_shots"])
    animate_shots = {f"{st}/{sh}" for st, shots in oracle["stories"].items() for sh, e in shots.items() if e["animate"]}
    if reviewed != animate_shots:
        raise ValueError(f"reviewed_shots must equal the {len(animate_shots)} shots with animate subjects; "
                         f"missing {sorted(animate_shots - reviewed)[:5]}, extra {sorted(reviewed - animate_shots)[:5]}")
    gold, changes = apply_corrections(oracle, corrections["shots"])
    _write_new(args.out, {"oracle": str(args.oracle), "corrections": str(args.corrections), "created_at": _now(),
                          "reviewer": corrections["reviewer"], "changes": changes, "stories": gold})
    print(json.dumps(changes))
    return 0


# ---------------------------------------------------------------------------
# calibrate（CPU）：Q1、Q2、Q4 的真片阳性与坏例读数
# ---------------------------------------------------------------------------

_RANK = {"left": 0, "center": 1, "right": 2}


def gold_left(layout: Mapping[str, Mapping[str, str]], a: str, b: str) -> str | None:
    """金标里两人谁在左（按 horizontal 名次；同名次判不了）。"""

    ra, rb = (_RANK.get(layout.get(s, {}).get("horizontal", ""), None) for s in (a, b))
    if ra is None or rb is None or ra == rb:
        return None
    return a if ra < rb else b


def frozen_shot_exclusions(freeze_path: Path) -> dict[str, frozenset[str]]:
    """冻结清单（修订 4 起）的排除镜 ``{窗: {镜号}}``；由源片边界修订模块持有；不用于新生成视频五项评分。"""

    from mspc.benchmark.long_story_boundary_revision import freeze_shot_exclusions

    _, table = freeze_shot_exclusions(_read(freeze_path))
    return {story: frozenset(shots) for story, shots in table.items()}


def same_location_pairs(plan_story: Mapping[str, Any], *, min_common: int,
                        excluded: frozenset[str]) -> tuple[list[tuple[int, int, list[str]]], int]:
    """同地点、共有 ≥ ``min_common`` 个人或动物的镜对；涉及排除镜的镜对跳过。返回 (镜对, 跳过的镜对数)。"""

    shots = plan_story["shots"]
    out, skipped = [], 0
    for i in range(len(shots)):
        for j in range(i + 1, len(shots)):
            if shots[i]["location_id"] != shots[j]["location_id"]:
                continue
            common = [s for s in shots[i]["animate"] if s in shots[j]["animate"]]
            if len(common) < min_common:
                continue
            if {shots[i]["shot_id"], shots[j]["shot_id"]} & excluded:
                skipped += 1
                continue
            out.append((i, j, common))
    return out, skipped


def _measures(plan: Mapping[str, Any], inst: Mapping[str, Any], bind: Mapping[str, Any], embed: Mapping[str, Any],
              variant: str) -> dict[tuple[str, str], Any]:
    out = {}
    for story_id, story in plan["stories"].items():
        for shot in story["shots"]:
            if not shot["animate"]:
                continue
            sid = shot["shot_id"]
            out[(story_id, sid)] = shot_measure(shot, inst["stories"][story_id][sid]["variants"][variant],
                                                bind["stories"][story_id][sid][variant],
                                                embed["stories"][story_id][sid][variant], width=plan["width"])
    return out


def q1_readout(plan: Mapping[str, Any], gold: Mapping[str, Any], orig: Mapping, mirror: Mapping, *,
               binding_sample_seed: str, excluded: Mapping[str, frozenset[str]]) -> dict[str, Any]:
    from mspc.data.case_sampling import sample_case_ids
    from mspc.evaluation.cross_shot_layout import cross_shot_mapping, pair_left, pair_relation, rate, separable

    width = plan["width"]
    units, per_shot = [], []
    skipped_shots = skipped_pairs = 0
    for story_id, story in sorted(plan["stories"].items()):
        subjects = {row["subject_id"]: row for row in story["subjects"]}
        story_excluded = excluded.get(story_id, frozenset())
        for shot in story["shots"]:
            if len(shot["animate"]) < 2:
                continue
            if shot["shot_id"] in story_excluded:
                skipped_shots += 1
                continue
            layout = gold[story_id][shot["shot_id"]]["layout"]
            for a, b in combinations(shot["animate"], 2):
                truth = gold_left(layout, a, b)
                if truth is None or not separable(subjects[a], subjects[b]):
                    continue
                tool = pair_left(orig[(story_id, shot["shot_id"])]["tracks"], a, b, width=width)
                per_shot.append({"story": story_id, "shot": shot["shot_id"], "pair": [a, b], "gold_left": truth,
                                 "tool_left": tool})
        pairs, skipped = same_location_pairs(story, min_common=2, excluded=story_excluded)
        skipped_pairs += skipped
        for i, j, common in pairs:
            si, sj = story["shots"][i]["shot_id"], story["shots"][j]["shot_id"]
            for a, b in combinations(common, 2):
                def relation(first: Mapping, second: Mapping) -> dict[str, Any]:
                    ea, eb = first["subjects"][a]["embedding"], first["subjects"][b]["embedding"]
                    fa, fb = second["subjects"][a]["embedding"], second["subjects"][b]["embedding"]
                    mapping = None if None in (ea, eb, fa, fb) else cross_shot_mapping({a: ea, b: eb}, {a: fa, b: fb})
                    return pair_relation(left_first=pair_left(first["tracks"], a, b, width=width),
                                         left_second=pair_left(second["tracks"], a, b, width=width), a=a, b=b,
                                         separable_text=separable(subjects[a], subjects[b]), mapping=mapping)
                gi, gj = gold[story_id][si]["layout"], gold[story_id][sj]["layout"]
                li, lj = gold_left(gi, a, b), gold_left(gj, a, b)
                truth = None if li is None or lj is None else ("same" if li == lj else "reversed")
                tool = relation(orig[(story_id, si)], orig[(story_id, sj)])
                mirrored = relation(orig[(story_id, si)], mirror[(story_id, sj)])
                wrong = {(si, s) for s in gold[story_id][si]["binding_wrong"]} | {(sj, s) for s in gold[story_id][sj]["binding_wrong"]}
                units.append({"unit_id": f"{story_id}|{si}|{sj}|{a}|{b}", "story": story_id, "shots": [si, sj],
                              "pair": [a, b], "adjacent": j == i + 1, "separable": separable(subjects[a], subjects[b]),
                              "gold": truth, "tool": tool["relation"], "tool_reason": tool["reason"],
                              "mirror": mirrored["relation"], "mirror_reason": mirrored["reason"],
                              "binding_wrong": bool(wrong & {(si, a), (si, b), (sj, a), (sj, b)})})
    decided = [row for row in per_shot if row["tool_left"] is not None]
    judgeable = [u for u in units if u["tool"] is not None]
    both = [u for u in judgeable if u["mirror"] is not None]
    scored = [u for u in judgeable if u["gold"] is not None]
    # ④ 按声明「抽 30 对看绑定」：单位是镜对（一对镜里任一可判的主体对用到错绑的主体，就记这一对错绑）。
    shot_pairs: dict[str, list[dict[str, Any]]] = {}
    for u in judgeable:
        shot_pairs.setdefault("|".join([u["story"], *u["shots"]]), []).append(u)
    sample_ids = set(sample_case_ids(sorted(shot_pairs), size=min(30, len(shot_pairs)), seed=binding_sample_seed))
    sampled_errors = sum(any(u["binding_wrong"] for u in shot_pairs[k]) for k in sample_ids)
    population_errors = sum(any(u["binding_wrong"] for u in rows) for rows in shot_pairs.values())
    return {
        "units": len(units),
        "excluded_two_person_shots_skipped": skipped_shots,
        "excluded_shot_pairs_skipped": skipped_pairs,
        "c1_per_shot_accuracy": rate(sum(r["tool_left"] == r["gold_left"] for r in decided), len(decided)),
        "c1_per_shot_coverage": rate(len(decided), len(per_shot)),
        "c2_mirror_flip": rate(sum(u["tool"] != u["mirror"] for u in both), len(both)),
        "c3_judgeable": rate(len(judgeable), len(units)),
        "c4_binding_errors_in_sample": {"errors": sampled_errors, "sample": len(sample_ids), "unit": "shot_pair"},
        "c4_binding_errors_all_judgeable_shot_pairs": rate(population_errors, len(shot_pairs)),
        "c4_binding_errors_all_judgeable_units": rate(sum(u["binding_wrong"] for u in judgeable), len(judgeable)),
        "ceiling_R_tool_vs_gold": rate(sum(u["tool"] == u["gold"] for u in scored), len(scored)),
        "gold_keep_rate_descriptive": rate(sum(u["gold"] == "same" for u in units if u["gold"]), sum(1 for u in units if u["gold"])),
        "gold_defined": rate(sum(1 for u in units if u["gold"]), len(units)),
        "reasons": {reason: sum(u["tool_reason"] == reason for u in units) for reason in sorted({u["tool_reason"] for u in units})},
        "by_adjacent": {str(flag): rate(sum(u["tool"] == u["gold"] for u in scored if u["adjacent"] == flag),
                                        sum(1 for u in scored if u["adjacent"] == flag)) for flag in (True, False)},
        "per_shot_rows": per_shot,
        "unit_rows": units,
        "binding_sample": sorted(sample_ids),
    }


def q2_readout(gold: Mapping[str, Any], crops: Sequence[Mapping[str, Any]], passes: Sequence[Mapping[str, Any]],
               qwen: Mapping[str, Any] | None, *, excluded: Mapping[str, frozenset[str]]) -> dict[str, Any]:
    from mspc.evaluation.cross_shot_layout import INSUFFICIENT, LATERAL, MIRRORED_ORIENTATION, rate, unflip

    kept = [r for r in crops if r["shot_id"] not in excluded.get(r["story_id"], frozenset())]
    skipped_instances = len(crops) - len(kept)
    crops = kept

    def truth(row: Mapping[str, Any]) -> str:
        return gold[row["story_id"]][row["shot_id"]]["layout"].get(row["subject"], {}).get("orientation", INSUFFICIENT)

    def primary(answers: Mapping[str, Any], row: Mapping[str, Any]) -> str:
        version = "flip" if row["primary_flipped"] else "orig"
        return unflip(answers[row["instance_id"]][version], row["primary_flipped"])

    def judge_block(answers: Mapping[str, Any], *, use_primary: bool) -> dict[str, Any]:
        get = (lambda r: primary(answers, r)) if use_primary else (lambda r: answers[r["instance_id"]]["orig"])
        lateral = [r for r in crops if truth(r) in LATERAL]
        answered = [r for r in lateral if get(r) != INSUFFICIENT]
        equi = [r for r in crops if answers[r["instance_id"]]["orig"] in LATERAL and answers[r["instance_id"]]["flip"] in LATERAL]
        known = [r for r in crops if truth(r) != INSUFFICIENT]
        return {
            "c1_lateral_accuracy_answered": rate(sum(get(r) == truth(r) for r in answered), len(answered)),
            "c1_lateral_accuracy_strict": rate(sum(get(r) == truth(r) for r in lateral), len(lateral)),
            "c2_mirror_equivariance": rate(sum(answers[r["instance_id"]]["flip"] == MIRRORED_ORIENTATION[answers[r["instance_id"]]["orig"]]
                                               for r in equi), len(equi)),
            "c4_insufficient_rate": rate(sum(get(r) == INSUFFICIENT for r in crops), len(crops)),
            "four_way_accuracy_on_gold_known": rate(sum(get(r) == truth(r) for r in known), len(known)),
            "confusion": {f"{t}->{a}": n for (t, a), n in sorted(_count((truth(r), get(r)) for r in crops).items())},
        }

    out: dict[str, Any] = {"instances": len(crops), "excluded_instances_skipped": skipped_instances,
                           "gold_lateral": sum(truth(r) in LATERAL for r in crops),
                           "gold_known": sum(truth(r) != INSUFFICIENT for r in crops)}
    for index, answers in enumerate(passes, start=1):
        out[f"j2_pass{index}"] = judge_block(answers["answers"], use_primary=True)
    if len(passes) >= 2:
        first, second = passes[0]["answers"], passes[1]["answers"]
        out["c3_j2_two_pass_disagreement"] = rate(sum(primary(first, r) != primary(second, r) for r in crops), len(crops))
    if qwen is not None:
        out["qwen_orig"] = judge_block(qwen["answers"], use_primary=False)
    return out


def _count(items: Any) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def q4_items(plan: Mapping[str, Any], *, size: int, seed: str,
             excluded: Mapping[str, frozenset[str]]) -> list[dict[str, Any]]:
    """Q4 金标样本：同地点、两镜共有 ≥ 1 个人的镜对（不含排除镜），每对取名单上第一个共有主体，按种子抽 ``size`` 对。

    10-10 已标的 60 对是在冻结修订 4 之前按不含排除的池抽的；读数时 :func:`q4_readout` 跳过涉及排除镜的对。"""

    from mspc.data.case_sampling import sample_case_ids

    pool = {}
    for story_id, story in sorted(plan["stories"].items()):
        pairs, _ = same_location_pairs(story, min_common=1, excluded=excluded.get(story_id, frozenset()))
        for i, j, common in pairs:
            si, sj = story["shots"][i]["shot_id"], story["shots"][j]["shot_id"]
            item_id = f"{story_id}|{si}|{sj}|{common[0]}"
            pool[item_id] = {"item_id": item_id, "story": story_id, "shots": [si, sj], "subject": common[0]}
    return [pool[item_id] for item_id in sample_case_ids(sorted(pool), size=size, seed=seed)]


def q4_readout(items: Sequence[Mapping[str, Any]], labels: Mapping[str, str], oracle: Mapping[str, Any], *,
               excluded: Mapping[str, frozenset[str]]) -> dict[str, Any]:
    from mspc.evaluation.cross_shot_layout import composition_holds, composition_tool_direction, head_log_ratio, rate

    kept = [item for item in items if not set(item["shots"]) & excluded.get(item["story"], frozenset())]
    skipped_items = len(items) - len(kept)
    rows = []
    for item in kept:
        si, sj = item["shots"]
        head_i = oracle["stories"][item["story"]][si]["subjects"][item["subject"]]["head_mid"]
        head_j = oracle["stories"][item["story"]][sj]["subjects"][item["subject"]]["head_mid"]
        r = head_log_ratio(head_i, head_j)
        rows.append({**item, "gold": labels[item["item_id"]], "log_ratio": r, "tool": composition_tool_direction(r)})
    # 金标 none：真片两镜里有一镜看不到这人的头（只有手、背影出画或空镜），不出题。
    tested = [row for row in rows if row["log_ratio"] is not None and abs(row["log_ratio"]) >= 0.2 and row["gold"] != "none"]
    copy = [composition_holds(row["tool"], 0.0) for row in tested]
    return {
        "items": len(rows),
        "excluded_items_skipped": skipped_items,
        "judgeable": rate(sum(row["log_ratio"] is not None for row in rows), len(rows)),
        "tested_abs_r_ge_0_2": len(tested),
        "direction_accuracy_on_tested": rate(sum(row["tool"] == row["gold"] for row in tested), len(tested)),
        "copy_previous_judged_unchanged": rate(sum(value is False for value in copy), len(copy)),
        "gold_counts": _count(row["gold"] for row in rows),
        "rows": rows,
    }


def run_calibrate(args: argparse.Namespace) -> int:
    if args.out.exists():
        raise FileExistsError(args.out)
    plan, inst, bind, embed = (_read(path) for path in (args.plan, args.instances, args.binding, args.embed))
    gold = _read(args.gold)["stories"]
    orig = _measures(plan, inst, bind, embed, "orig")
    mirror = _measures(plan, inst, bind, embed, "mirror")
    excluded = frozen_shot_exclusions(args.freeze)
    in_plan = {story: sorted(shots) for story, shots in excluded.items() if story in plan["stories"]}
    readout: dict[str, Any] = {"created_at": _now(), "plan": str(args.plan), "gold": str(args.gold),
                               "freeze": str(args.freeze), "freeze_revision": _read(args.freeze)["revision"],
                               "excluded_shots": {"count": sum(len(v) for v in in_plan.values()), "by_story": in_plan}}
    readout["q1"] = q1_readout(plan, gold, orig, mirror, binding_sample_seed="csl-q1-binding-v1", excluded=excluded)
    crops = _read(args.crops)["crops"]
    passes = [_read(path) for path in args.judge]
    readout["q2"] = q2_readout(gold, crops, passes, None if args.qwen is None else _read(args.qwen), excluded=excluded)
    if args.q4_labels is not None:
        q4 = _read(args.q4_items)["items"]
        readout["q4"] = q4_readout(q4, _read(args.q4_labels)["labels"], _read(args.oracle), excluded=excluded)
    _write_new(args.out, readout)
    summary = {"excluded_shots": readout["excluded_shots"]["count"], "q1": {k: v for k, v in readout["q1"].items() if not k.endswith("rows") and k != "binding_sample"},
               "q2": readout["q2"], "q4": {k: v for k, v in readout.get("q4", {}).items() if k != "rows"}}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


def run_q4_items(args: argparse.Namespace) -> int:
    from PIL import Image, ImageDraw

    from mspc.evaluation.cross_shot_layout import MID_FRAME

    plan, oracle = _read(args.plan), _read(args.oracle)
    items = q4_items(plan, size=args.size, seed=args.seed, excluded=frozen_shot_exclusions(args.freeze))
    _write_new(args.out_dir / "q4_items.json", {"seed": args.seed, "size": args.size, "items": items})
    tile_w, tile_h = 400, 231
    font = _font(18)
    rows = []
    for number, item in enumerate(items, start=1):
        row = Image.new("RGB", (2 * tile_w + 8, tile_h + 26), (20, 20, 20))
        draw = ImageDraw.Draw(row)
        draw.text((4, 3), f"#{number} {item['item_id']}", fill=(255, 255, 0), font=font)
        for k, shot_id in enumerate(item["shots"]):
            image = Image.open(_plan_shot(plan, item["story"], shot_id)["frames"]["orig"][MID_FRAME]).convert("RGB")
            box = oracle["stories"][item["story"]][shot_id]["subjects"][item["subject"]]["boxes"][MID_FRAME]
            if box is not None:
                ImageDraw.Draw(image).rectangle(box, outline=(255, 60, 60), width=5)
            row.paste(image.resize((tile_w, tile_h)), (k * (tile_w + 8), 26))
        rows.append(row)
    for start in range(0, len(rows), 6):
        chunk = rows[start:start + 6]
        sheet = Image.new("RGB", (chunk[0].size[0], sum(r.size[1] + 4 for r in chunk)), (60, 60, 60))
        for i, row in enumerate(chunk):
            sheet.paste(row, (0, i * (row.size[1] + 4)))
        sheet.save(args.out_dir / f"q4_sheet_{start // 6:02d}.jpg", quality=88)
    print(json.dumps({"items": len(items)}))
    return 0


# ---------------------------------------------------------------------------
# q3-layout（GPU，renderer_public_metrics）：地标布局尺子加叠加层抑制，在 300 个真实镜界上重读
# ---------------------------------------------------------------------------

#: 一个源视频取多少个镜的中帧估叠加层（≥ 5 才估；超出的按顺序等距取）。
OVERLAY_SHOTS_MAX = 8


def overlay_identity(dataset: str, path: str) -> str:
    """叠加层按源视频估：Cine250K 是原片文件；CI-VID 每镜一个片段文件，同一源视频是片段的上两级目录。"""

    if dataset == "cine250k":
        return path
    if dataset == "ci_vid":
        return str(Path(path).parent.parent)
    raise ValueError(f"unknown dataset {dataset!r}")


def overlay_sources(metadata_csv: Path, identities: set[str]) -> dict[str, list[tuple[str, int]]]:
    """205k 池元数据里属于这些源视频的全部镜：(文件, 镜中帧号)，去重、排序。"""

    import csv

    csv.field_size_limit(10**9)
    shots: dict[str, set[tuple[str, int]]] = {identity: set() for identity in identities}
    with metadata_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] not in ("cine250k", "ci_vid"):
                continue
            paths = json.loads(row["source_video_paths"])
            segments = json.loads(row["source_shot_segments"])
            for path, (start, end) in zip(paths, segments, strict=True):
                identity = overlay_identity(row["dataset"], path)
                if identity in shots:
                    shots[identity].add((path, (int(start) + int(end) - 1) // 2))
    return {identity: sorted(rows) for identity, rows in shots.items()}


def _even_pick(rows: Sequence[Any], count: int) -> list[Any]:
    if len(rows) <= count:
        return list(rows)
    return [rows[round(k * (len(rows) - 1) / (count - 1))] for k in range(count)]


def run_q3_layout(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    from mspc.data.msm_global_frames import default_msm_frame_reader
    from mspc.evaluation import spatial_layout_probe as rules
    from mspc.evaluation.cross_shot_layout import OVERLAY_MIN_SHOTS, static_overlay_mask, suppress_overlay
    from mspc.evaluation.renderer_qualification_official_backend import (
        DEFAULT_ROI_DETECTOR_CONFIG,
        FROZEN_PUBLIC_ROI_DETECTOR,
        _DinoRoiEngine,
        _model_batch_from_roi_config,
    )
    from tools.annotate.label_cut_types import load_reference
    from tools.run.run_spatial_probe import _real_shot_frames

    if args.out.exists():
        raise FileExistsError(args.out)
    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    reference = load_reference(args.reference)
    by_id = {row["sample_id"]: row for row in samples}
    identity = {row["sample_id"]: overlay_identity(row["source"], row["video"]["a"]) for row in samples}
    started = time.time()
    sources = overlay_sources(args.metadata_csv, set(identity.values()))
    print(f"[q3] overlay sources read {time.time() - started:.0f}s", file=sys.stderr, flush=True)

    def resized(image: Any) -> Any:
        array = np.asarray(image.convert("RGB"))
        scale = rules.REAL_LAYOUT_LONG_SIDE / max(array.shape[:2])
        size = (round(array.shape[1] * scale), round(array.shape[0] * scale))
        return np.ascontiguousarray(cv2.resize(array, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC))

    overlays: dict[str, Any] = {}
    overlay_rows: dict[str, Any] = {}
    for key in sorted(set(identity.values())):
        picked = _even_pick(sources[key], OVERLAY_SHOTS_MAX)
        grays = []
        for path, frame in picked:
            try:
                (image,) = default_msm_frame_reader(Path(path), [frame])
            except (ValueError, RuntimeError, OSError):
                continue
            grays.append(cv2.cvtColor(resized(image), cv2.COLOR_RGB2GRAY))
        sizes = _count(g.shape for g in grays)
        shape = max(sizes, key=sizes.get) if sizes else None
        same = [g for g in grays if g.shape == shape]
        if len(same) >= OVERLAY_MIN_SHOTS:
            overlays[key] = static_overlay_mask(same)
        overlay_rows[key] = {"shots_available": len(sources[key]), "frames_used": len(same),
                             "overlay_fraction": None if key not in overlays else round(float(overlays[key].mean()), 4)}
    print(f"[q3] overlays {len(overlays)}/{len(overlay_rows)} {time.time() - started:.0f}s", file=sys.stderr, flush=True)

    pairs = rules.real_layout_pairs(samples, reference)
    shots = sorted({tuple(side) for pair in pairs for side in (pair["left"], pair["right"])})
    batch = _model_batch_from_roi_config(DEFAULT_ROI_DETECTOR_CONFIG)
    engine = _DinoRoiEngine(args.resource_root, args.device)
    plain: dict[tuple[str, str], list[Any]] = {}
    suppressed: dict[tuple[str, str], list[Any]] = {}
    for index, (sample_id, shot) in enumerate(shots, start=1):
        frames = _real_shot_frames(by_id[sample_id], shot)
        overlay = overlays.get(identity[sample_id])
        regions = engine.grounded_regions(frames, list(rules.REAL_LAYOUT_PHRASES), FROZEN_PUBLIC_ROI_DETECTOR,
                                          detector_frame_batch_size=batch.detector_frame_batch_size,
                                          dino_crop_batch_size=batch.dino_crop_batch_size)
        for frame, frame_regions in zip(frames, regions, strict=True):
            masks = engine._masks(frame, [region["box"] for region in frame_regions])
            background = rules.background_mask(masks, frame.shape[:2])
            plain.setdefault((sample_id, shot), []).append(rules.sift_features(frame, background))
            if overlay is not None and overlay.shape == background.shape:
                suppressed.setdefault((sample_id, shot), []).append(
                    rules.sift_features(frame, suppress_overlay(background, overlay)))
        engine.clear_frame_caches()
        if index % 50 == 0 or index == len(shots):
            print(f"[q3] shots {index}/{len(shots)} {time.time() - started:.0f}s", file=sys.stderr, flush=True)
    covered = [pair for pair in pairs if tuple(pair["left"]) in suppressed and tuple(pair["right"]) in suppressed
               and len(suppressed[tuple(pair["left"])]) == 3 and len(suppressed[tuple(pair["right"])]) == 3]
    uncovered = [pair["pair_id"] for pair in pairs if pair not in covered]

    def measure(features: Mapping, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for pair in rows:
            inliers, where = rules.pair_inliers(features[tuple(pair["left"])], features[tuple(pair["right"])])
            out.append({**pair, "inliers": inliers, "best_keyframes": list(where)})
        return out

    with_overlay = rules.real_layout_readout(measure(suppressed, covered))
    without_overlay = rules.real_layout_readout(measure(plain, covered))
    _write_new(args.out, {
        "samples": str(args.samples), "reference": str(args.reference), "metadata_csv": str(args.metadata_csv),
        "overlay_rule": {"min_shots": OVERLAY_MIN_SHOTS, "max_shots": OVERLAY_SHOTS_MAX, "std_max": 3.0, "dilation_px": 5},
        "overlays": overlay_rows, "pairs_total": len(pairs), "pairs_covered": len(covered), "pairs_uncovered": uncovered,
        "with_overlay_suppression": with_overlay, "without_overlay_same_pairs": without_overlay,
        "elapsed_seconds": round(time.time() - started, 1),
    })
    brief = {name: {k: v for k, v in block.items() if k != "rows"} for name, block in
             (("with_overlay_suppression", with_overlay), ("without_overlay_same_pairs", without_overlay))}
    print(json.dumps({"pairs_covered": len(covered), "pairs_total": len(pairs), **brief}, ensure_ascii=False, indent=1))
    return 0


# ---------------------------------------------------------------------------
# strip（OpenRouter 纯文本）：N 臂删画面方位短语，程序核对纯删除
# ---------------------------------------------------------------------------


def run_strip(args: argparse.Namespace) -> int:
    """逐镜动作散文删方位短语。gemini-3.8-flash 纯文本、温度 0（与三档反写同配置）；输出必须是输入的纯删除，
    不是就重问一次，仍不是记 ``not_pure_deletion`` 交人工。逐条写 ``<out>.calls.jsonl``，重跑只补没做的。"""

    from mspc.data.planner_baseline import build_shot_tier_client, load_shot_tier_rewrite_config
    from mspc.data.openrouter_client import load_openrouter_api_key
    from mspc.evaluation.cross_shot_layout import deleted_words, is_pure_deletion, strip_prompt

    config = load_shot_tier_rewrite_config(args.config)
    pairs = {pair["story_id"]: pair for pair in _read(args.pairs)["pairs"]}
    log_path = args.out.with_name(args.out.name + ".calls.jsonl")
    done = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            row = json.loads(line)
            done[row["key"]] = row
    client = build_shot_tier_client(config, api_key=load_openrouter_api_key(args.api_key_file), proxy=args.proxy)
    for story_id, pair in sorted(pairs.items()):
        for shot in pair["context_b"]["shots"]:
            key = f"{story_id}/{shot['shot_id']}"
            if key in done:
                continue
            original = shot["prompt"]
            attempts = []
            for _ in range(2):
                response = client.complete_messages(
                    model=config["model"], messages=[{"role": "user", "content": strip_prompt(original)}],
                    temperature=config["temperature"], max_tokens=config["max_tokens"],
                    request_options=config["request_options"], response_format={"type": "json_object"})
                try:
                    edited = " ".join(json.loads(response.text)["edited"].split())
                except (ValueError, KeyError, TypeError, AttributeError):
                    edited = None
                attempts.append({"raw": response.text, "cost_usd": response.cost_usd})
                if edited is not None and is_pure_deletion(original, edited):
                    break
            pure = edited is not None and is_pure_deletion(original, edited)
            row = {"key": key, "original": original, "edited": edited if pure else None,
                   "status": ("unchanged" if edited == " ".join(original.split()) else "deleted") if pure else "not_pure_deletion",
                   "deleted_words": deleted_words(original, edited) if pure else None, "attempts": attempts, "at": _now()}
            with log_path.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            done[key] = row
            print(f"[strip] {key} {row['status']}", file=sys.stderr, flush=True)
    rows = list(done.values())
    from decimal import Decimal

    cost = sum((Decimal(a["cost_usd"]) for row in rows for a in row["attempts"]), Decimal("0"))
    summary = {"shots": len(rows), "deleted": sum(r["status"] == "deleted" for r in rows),
               "unchanged": sum(r["status"] == "unchanged" for r in rows),
               "not_pure_deletion": sum(r["status"] == "not_pure_deletion" for r in rows), "cost_usd": str(cost)}
    if not args.out.exists():
        _write_new(args.out, {"pairs": str(args.pairs), "config": str(args.config), "created_at": _now(),
                              "summary": summary, "shots": {row["key"]: row for row in rows}})
    print(json.dumps(summary))
    return 0


# ---------------------------------------------------------------------------
# build-arms（CPU，wan22 环境：umT5 计 token）：旧空间诊断的 N / P / S / S-LR 逐镜文字备料；不接当前三档benchmark
# ---------------------------------------------------------------------------

#: 10-10 起编译器把 screen 事实写成目标行开头的空间句（取代末尾 ``Screen:`` 句），所以 S、S-LR 现在重建出的是新写法，
#: 不再复现原实验的末尾 ``Screen:`` 句；原实验的输入以当时提交（f93adaae、c2e40fd2）的代码与落盘产物为准。
ARM_AXES = {"N": (), "P": (), "S": ("horizontal", "depth", "orientation"), "S-LR": ("horizontal",), "S-first": (),
            "S-compiled": ("horizontal", "depth", "orientation")}
#: 验证臂 S-compiled（10-10）：N 臂散文＋oracle 屏幕事实，文字完全由改写法后的编译器出（开头空间句、短属性词指人，
#: 撞词时退回 ``Subject k, the <头名词>``），即首训将用的格式。
#: 补充臂 S-first：文字同 N 臂，另把 oracle 的左右与前后写成每镜文字的第一句，用外观短语指人、不用编号。
FIRST_SENTENCE_ARMS = frozenset({"S-first"})
#: 各臂逐镜散文取自哪份文字（读提及顺序用）：除 P 外都是删方位稿，S-first 的首句不算散文。
ARM_PROSE = {"N": "N", "P": "P", "S": "N", "S-LR": "N", "S-first": "N", "S-compiled": "N"}


def arm_event_logs(pairs: Mapping[str, Any], candidate_r3: Mapping[str, Any], gold: Mapping[str, Any],
                   strip: Mapping[str, Any], arm: str) -> dict[str, dict[str, Any]]:
    """一臂全部窗的事件日志：P 用冻结原文，其余用删方位稿（人工复核后的定稿）；S、S-LR 再把金标布局填进 screen 事实。"""

    from mspc.benchmark.long_story_event_log import event_log_from_script, r3_frame_counts
    from mspc.evaluation.cross_shot_layout import with_screen_facts

    if arm not in ARM_AXES:
        raise ValueError(f"unknown arm {arm!r}")
    logs = {}
    for story_id, pair in sorted(pairs.items()):
        raw = event_log_from_script(pair["context_b"], pair["change"], r3_frame_counts(candidate_r3, story_id))
        if arm != "P":
            shots = []
            for shot in raw["shots"]:
                row = strip[f"{story_id}/{shot['shot_id']}"]
                if row["final"] is None:
                    raise ValueError(f"{story_id}/{shot['shot_id']}: no reviewed N-arm text")
                if " ".join(shot["action"].split()) != " ".join(row["original"].split()):
                    raise ValueError(f"{story_id}/{shot['shot_id']}: event-log action differs from the stripped original")
                shots.append({**shot, "action": row["final"]})
            raw = {**raw, "shots": shots}
        if ARM_AXES[arm]:
            layouts = [gold[story_id][shot["shot_id"]]["layout"] for shot in raw["shots"]]
            raw = with_screen_facts(raw, layouts, axes=ARM_AXES[arm])
        logs[story_id] = raw
    return logs


def run_build_arms(args: argparse.Namespace) -> int:
    from mspc.data.event_log import compile_event_log, parse_event_log
    from mspc.evaluation.cross_shot_layout import layout_first_sentence
    from mspc.data.shot_text import UMT5_PROMPT_TOKEN_BUDGET, load_umt5_token_counter

    pairs = {pair["story_id"]: pair for pair in _read(args.pairs)["pairs"]}
    candidate = _read(args.candidate_r3)
    gold = _read(args.gold)["stories"]
    strip = _read(args.strip_final)["shots"]
    counter = load_umt5_token_counter(args.tokenizer_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for arm in args.arms.split(","):
        out = args.out_dir / f"arm_{arm}.json"
        if out.exists():
            raise FileExistsError(out)
        cases, tokens = [], []
        for story_id, raw in arm_event_logs(pairs, candidate, gold, strip, arm).items():
            log = parse_event_log(raw)
            compiled = compile_event_log(log, span=(0, len(log.shots)), references=[], token_counter=counter)
            prompts, counts = list(compiled.target_rows), list(compiled.token_counts)
            if arm in FIRST_SENTENCE_ARMS:
                attributes = {row["subject_id"]: row["attributes"] for row in pairs[story_id]["context_b"]["subjects"]}
                firsts = [layout_first_sentence(gold[story_id][shot.shot_id]["layout"], attributes) for shot in log.shots]
                prompts = [f"{first} {prompt}" if first else prompt for first, prompt in zip(firsts, prompts, strict=True)]
                counts = [counter(prompt) for prompt in prompts]
                over = [shot.shot_id for shot, count in zip(log.shots, counts, strict=True) if count > UMT5_PROMPT_TOKEN_BUDGET]
                if over:
                    raise ValueError(f"{story_id}: prompts over the umT5 budget of {UMT5_PROMPT_TOKEN_BUDGET} tokens: {over}")
            tokens.extend(counts)
            cases.append({"case_id": f"{story_id}-b", "story_id": story_id, "prompts": prompts,
                          "context": {"shots": [{"shot_id": shot.shot_id, "frame_count": shot.frame_count}
                                                for shot in log.shots]},
                          "event_log": raw, "token_counts": counts})
        _write_new(out, {"arm": arm, "axes": list(ARM_AXES[arm]),
                         "first_sentence_axes": ["horizontal", "depth"] if arm in FIRST_SENTENCE_ARMS else [],
                         "created_at": _now(), "pairs": str(args.pairs),
                         "gold": str(args.gold), "strip_final": str(args.strip_final), "cases": cases})
        summary[arm] = {"cases": len(cases), "shots": len(tokens), "max_tokens": max(tokens),
                        "frames": sum(s["frame_count"] for c in cases for s in c["context"]["shots"])}
    print(json.dumps(summary))
    return 0


# ---------------------------------------------------------------------------
# 第 1 步：生成片各臂读数。人物抽取走 tools/run/spatial_facts.py（frames-render → sam3 → sam3-phrase → depth-dino →
# people → bind → facts），本节只读它的工作目录；真片用同一版抽取链的工作目录读天花板 R。
# ---------------------------------------------------------------------------

#: 第 1 步亲看：每份出片抽多少个镜对。
STEP1_VISUAL_PAIRS = 15
Q1_VARIANTS = ("strict", "declared", "path1", "fused")


def chain_measures(work: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """空间事实抽取链的一个工作目录 -> 每个有人物的镜的测量：镜内指认、布局三帧框、DINOv2 掩码画布嵌入均值（布局三帧）、
    三路外观向量、布局事实、中帧框与中帧实例数。"""

    from mspc.evaluation.cross_shot_layout import MID_FRAME, mean_embedding
    from tools.run import spatial_facts as chain

    bind = _read(work / "bind.json")["stories"]
    facts = _read(work / "facts.json")["stories"]
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for story_id, _window, shot in chain._shots(work, ""):
        sid = shot["shot_id"]
        inputs = chain._shot_inputs(work, story_id, shot)
        tracks = inputs["sam3"]["tracks"]
        positions = inputs["sam3"]["layout_positions"]
        binding = {s: row["track"] for s, row in bind[story_id][sid]["binding"].items()}
        if set(binding) != set(shot["animate"]):
            raise ValueError(f"{work} {story_id} {sid}: binding covers {sorted(binding)}, script lists {shot['animate']}")
        boxes = [{s: (None if t is None or str(p) not in tracks[t]["frames"]
                      else [float(v) for v in tracks[t]["frames"][str(p)][:4]]) for s, t in binding.items()}
                 for p in positions]
        dino = inputs["depth_dino"]["dino"]
        embeddings = {s: (None if t is None else mean_embedding([dino[f"{t}@{p}"] for p in positions if f"{t}@{p}" in dino]))
                      for s, t in binding.items()}
        out[(story_id, sid)] = {
            "binding": binding,
            "source": {s: row["source"] for s, row in bind[story_id][sid]["binding"].items()},
            "tracks": {"boxes": boxes, "embeddings": embeddings},
            "vectors": {s: ({} if t is None else chain._instance_vectors(inputs, t)) for s, t in binding.items()},
            "layout": facts[story_id][sid]["layout"],
            "mid_boxes": boxes[MID_FRAME],
            "all_mid_boxes": {t: [float(v) for v in row["frames"][str(positions[MID_FRAME])][:4]]
                              for t, row in sorted(tracks.items()) if str(positions[MID_FRAME]) in row["frames"]},
            "mid_position": positions[MID_FRAME],
        }
    return out


def fused_pair_mapping(first: Mapping[str, Mapping[str, Any]], second: Mapping[str, Mapping[str, Any]], a: str, b: str,
                       identity: Mapping[str, Any]) -> dict[str, str] | None:
    """两镜里同一对主体的三路外观向量 -> 二选一的跨镜指派（恒等或交换），按合成对数几率之和取大者；缺任一项或相等为 None。"""

    from mspc.data.spatial_facts import cosine, fused_log_odds

    odds = {}
    for x in (a, b):
        for y in (a, b):
            sims = {path: cosine(first[x][path], second[y][path]) for path in first[x] if path in second[y]}
            value = fused_log_odds(sims, identity)
            if value is None:
                return None
            odds[(x, y)] = value
    keep, swap = odds[(a, a)] + odds[(b, b)], odds[(a, b)] + odds[(b, a)]
    if keep == swap:
        return None
    return {a: a, b: b} if keep > swap else {a: b, b: a}


def step1_rows(windows: Mapping[str, Any], measures: Mapping[tuple[str, str], Mapping[str, Any]],
               gold: Mapping[str, Any], excluded: Mapping[str, frozenset[str]],
               prompts: Mapping[str, Sequence[str]], identity: Mapping[str, Any]) -> dict[str, Any]:
    """一份出片（或真片）的逐单元行：Q1 镜对×主体对、逐镜主体对（服从与提及顺序）、在场、前后。

    Q1 的四种认人口径（工具关系，None 为判不了）：
    ``strict`` 全部主体对都要求两路一致（主读数）；``declared`` 声明原规则，文字分不开的只走第 2 路；
    ``path1`` 只用镜内指认；``fused`` 第 2 路换成三路外观合成的二选一。"""

    from mspc.evaluation.cross_shot_layout import (
        cross_shot_mapping,
        depth_front,
        first_mentioned,
        pair_left,
        pair_relation,
        separable,
    )

    width = windows["width"]
    pair_rows, shot_rows, presence_rows, depth_rows, shot_counts = [], [], [], [], []
    skipped_pairs = skipped_shots = 0
    for story_id, window in sorted(windows["windows"].items()):
        subjects = {row["subject_id"]: row for row in window["subjects"]}
        story_excluded = excluded.get(story_id, frozenset())
        shots = window["shots"]
        if len(prompts[story_id]) != len(shots):
            raise ValueError(f"{story_id}: {len(prompts[story_id])} prompts for {len(shots)} shots")
        for index, shot in enumerate(shots):
            sid = shot["shot_id"]
            if not shot["animate"]:
                continue
            if sid in story_excluded:
                skipped_shots += 1
                continue
            m = measures[(story_id, sid)]
            glayout = gold[story_id][sid]["layout"]
            for s in shot["animate"]:
                presence_rows.append({"unit": f"{story_id}|{sid}|{s}", "story": story_id, "bound": m["binding"][s] is not None,
                                      "source": m["source"][s], "gold_visible": s in glayout})
            shot_counts.append({"unit": f"{story_id}|{sid}", "story": story_id, "script": len(shot["animate"]),
                                "instances": len(m["all_mid_boxes"])})
            for a, b in combinations(shot["animate"], 2):
                truth = gold_left(glayout, a, b)
                if truth is not None:
                    shot_rows.append({"unit": f"{story_id}|{sid}|{a}|{b}", "story": story_id, "shot": sid, "pair": [a, b],
                                      "gold_left": truth, "tool_left": pair_left(m["tracks"], a, b, width=width),
                                      "first_mentioned": first_mentioned(prompts[story_id][index], a, b),
                                      "separable": separable(subjects[a], subjects[b])})
                front = depth_front(glayout, a, b)
                if front is not None:
                    both = m["binding"][a] is not None and m["binding"][b] is not None
                    depth_rows.append({"unit": f"{story_id}|{sid}|{a}|{b}", "story": story_id, "gold_front": front,
                                       "both_bound": both, "tool_front": depth_front(m["layout"], a, b) if both else None})
        pairs, skipped = same_location_pairs(window, min_common=2, excluded=story_excluded)
        skipped_pairs += skipped
        for i, j, common in pairs:
            si, sj = shots[i]["shot_id"], shots[j]["shot_id"]
            mi, mj = measures[(story_id, si)], measures[(story_id, sj)]
            between = {shots[k]["location_id"] for k in range(i + 1, j)}
            layer = "adjacent" if j == i + 1 else "revisit" if between - {shots[i]["location_id"]} else "same_run"
            for a, b in combinations(common, 2):
                li, lj = gold_left(gold[story_id][si]["layout"], a, b), gold_left(gold[story_id][sj]["layout"], a, b)
                truth = None if li is None or lj is None else ("same" if li == lj else "reversed")
                left_i = pair_left(mi["tracks"], a, b, width=width)
                left_j = pair_left(mj["tracks"], a, b, width=width)
                vectors = [mi["tracks"]["embeddings"][a], mi["tracks"]["embeddings"][b],
                           mj["tracks"]["embeddings"][a], mj["tracks"]["embeddings"][b]]
                mapping = None if any(v is None for v in vectors) else cross_shot_mapping(
                    {a: vectors[0], b: vectors[1]}, {a: vectors[2], b: vectors[3]})
                sep = separable(subjects[a], subjects[b])
                relation = {
                    "strict": pair_relation(left_first=left_i, left_second=left_j, a=a, b=b, separable_text=True, mapping=mapping),
                    "declared": pair_relation(left_first=left_i, left_second=left_j, a=a, b=b, separable_text=sep, mapping=mapping),
                    "fused": pair_relation(left_first=left_i, left_second=left_j, a=a, b=b, separable_text=True,
                                           mapping=fused_pair_mapping(mi["vectors"], mj["vectors"], a, b, identity)
                                           if all(mi["binding"][s] is not None and mj["binding"][s] is not None for s in (a, b)) else None),
                }
                path1 = None if left_i is None or left_j is None else ("same" if left_i == left_j else "reversed")
                pair_rows.append({"unit": f"{story_id}|{si}|{sj}|{a}|{b}", "story": story_id, "shots": [si, sj], "pair": [a, b],
                                  "layer": layer, "separable": sep, "gold": truth, "left": [left_i, left_j],
                                  "strict": relation["strict"]["relation"], "strict_reason": relation["strict"]["reason"],
                                  "declared": relation["declared"]["relation"], "path1": path1,
                                  "fused": relation["fused"]["relation"]})
    return {"pairs": pair_rows, "shot_pairs": shot_rows, "presence": presence_rows, "depth": depth_rows,
            "shot_counts": shot_counts, "excluded_shots_skipped": skipped_shots, "excluded_shot_pairs_skipped": skipped_pairs}


def q4_rows(windows: Mapping[str, Any], measures: Mapping[tuple[str, str], Mapping[str, Any]], heads: Mapping[str, Any],
            oracle: Mapping[str, Any], gold: Mapping[str, Any], excluded: Mapping[str, frozenset[str]]) -> list[dict[str, Any]]:
    """Q4 描述性读数的逐项行：同地点、共有同一人的镜对上，真片头框对数比 |r| ≥ 0.2 的出题，生成片按同一量法读是否同向变化。

    真片 r 取第 0 步旧链的中帧头框（``oracle.json``，金标判为错绑的不用）；生成片用新链指认的中帧框内得分最高的头框。"""

    from mspc.evaluation.cross_shot_layout import composition_holds, composition_item, head_log_ratio
    from mspc.evaluation.spatial_layout_probe import subject_head

    rows = []
    for story_id, window in sorted(windows["windows"].items()):
        shots = window["shots"]
        pairs, _ = same_location_pairs(window, min_common=1, excluded=excluded.get(story_id, frozenset()))
        for i, j, common in pairs:
            si, sj = shots[i]["shot_id"], shots[j]["shot_id"]
            for s in common:
                real = []
                for sid in (si, sj):
                    row = oracle["stories"][story_id][sid]["subjects"].get(s)
                    usable = row is not None and s in gold[story_id][sid]["layout"] and s not in gold[story_id][sid]["binding_wrong"]
                    real.append(row["head_mid"] if usable else None)
                planned = composition_item(head_log_ratio(real[0], real[1]))
                if planned is None:
                    continue
                generated = [subject_head(measures[(story_id, sid)]["mid_boxes"][s], heads[story_id][sid]) for sid in (si, sj)]
                ratio = head_log_ratio(generated[0], generated[1])
                rows.append({"unit": f"{story_id}|{si}|{sj}|{s}", "story": story_id, "planned": planned,
                             "real_log_ratio": round(head_log_ratio(real[0], real[1]), 4),
                             "log_ratio": None if ratio is None else round(ratio, 4),
                             "holds": composition_holds(planned, ratio)})
    return rows


def cluster_mean(rows: Sequence[Mapping[str, Any]], value: Any) -> dict[str, Any]:
    """逐单元 0/1 值的率：Wilson 95% 与按窗聚类 bootstrap 区间（10,000 次、seed 42，同带宽阶梯的读法）。"""

    from mspc.evaluation.cross_shot_layout import rate
    from mspc.evaluation.paired_bootstrap import cluster_bootstrap

    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["story"], []).append(float(value(row)))
    out = rate(int(sum(sum(v) for v in grouped.values())), sum(len(v) for v in grouped.values()))
    out["stories"] = len(grouped)
    out["cluster_ci95"] = [round(v, 4) for v in cluster_bootstrap(grouped)["ci95"]] if len(grouped) >= 2 else None
    return out


def paired_difference(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]], *, usable: Any,
                      value: Any) -> dict[str, Any]:
    """两臂在共有、且两边都可用的单元上的配对差（前者减后者），按窗聚类 bootstrap，读法按固定阈值。"""

    from mspc.evaluation.cross_shot_layout import READING_UNDECIDED, difference_reading
    from mspc.evaluation.paired_bootstrap import cluster_bootstrap

    left = {row["unit"]: row for row in first if usable(row)}
    right = {row["unit"]: row for row in second if usable(row)}
    common = sorted(set(left) & set(right))
    grouped: dict[str, list[float]] = {}
    for unit in common:
        grouped.setdefault(left[unit]["story"], []).append(float(value(left[unit])) - float(value(right[unit])))
    out: dict[str, Any] = {"units": len(common), "stories": len(grouped),
                           "first_rate": None if not common else round(sum(float(value(left[u])) for u in common) / len(common), 4),
                           "second_rate": None if not common else round(sum(float(value(right[u])) for u in common) / len(common), 4),
                           "difference": None, "ci95": None, "reading": READING_UNDECIDED}
    if len(grouped) >= 2:
        boot = cluster_bootstrap(grouped)
        out.update(difference=round(boot["mean"], 4), ci95=[round(v, 4) for v in boot["ci95"]],
                   reading=difference_reading(boot["mean"], boot["ci95"][0], boot["ci95"][1]))
    return out


def cluster_rate_gap(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]], *, value: Any) -> dict[str, Any]:
    """两组单元（不要求同一批单元）在同一批窗上的率差（前者减后者）：每轮有放回抽窗，两组各取被抽窗全部单元的均值再相减；
    轮数与种子同 ``paired_bootstrap.cluster_bootstrap``。用于生成臂对真片天花板 R。"""

    import numpy as np

    from mspc.evaluation.cross_shot_layout import difference_reading
    from mspc.evaluation.paired_bootstrap import BOOTSTRAP_ROUNDS, BOOTSTRAP_SEED

    stories = sorted({row["story"] for row in first} | {row["story"] for row in second})
    index = {story: k for k, story in enumerate(stories)}
    sums, counts = np.zeros((2, len(stories))), np.zeros((2, len(stories)))
    for group, rows in enumerate((first, second)):
        for row in rows:
            sums[group, index[row["story"]]] += float(value(row))
            counts[group, index[row["story"]]] += 1
    if counts[0].sum() == 0 or counts[1].sum() == 0:
        raise ValueError("both groups need at least one unit")
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(stories), size=(BOOTSTRAP_ROUNDS, len(stories)))
    den = counts[:, draws].sum(axis=2)
    valid = (den > 0).all(axis=0)
    num = sums[:, draws].sum(axis=2)
    gaps = num[0, valid] / den[0, valid] - num[1, valid] / den[1, valid]
    point = float(sums[0].sum() / counts[0].sum() - sums[1].sum() / counts[1].sum())
    low, high = float(np.quantile(gaps, 0.025)), float(np.quantile(gaps, 0.975))
    return {"first_rate": round(float(sums[0].sum() / counts[0].sum()), 4), "first_units": int(counts[0].sum()),
            "second_rate": round(float(sums[1].sum() / counts[1].sum()), 4), "second_units": int(counts[1].sum()),
            "stories": len(stories), "difference": round(point, 4), "ci95": [round(low, 4), round(high, 4)],
            "reading": difference_reading(point, low, high)}


def _count_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return {str(k): v for k, v in sorted(_count(row[key] for row in rows).items(), key=lambda item: str(item[0]))}


def step1_block(rows: Mapping[str, Any]) -> dict[str, Any]:
    """一份（或同臂两种子合并的）逐单元行 -> 各项率。Q1 只在金标关系有定义的单元上读。"""

    defined = [row for row in rows["pairs"] if row["gold"] is not None]
    q1: dict[str, Any] = {"units": len(rows["pairs"]), "gold_defined": len(defined),
                          "shot_pairs_gold_defined": len({(r["story"], *r["shots"], r.get("seed")) for r in defined}),
                          "strict_reasons": _count_by(defined, "strict_reason")}
    for variant in Q1_VARIANTS:
        judged = [row for row in defined if row[variant] is not None]
        q1[variant] = {"judgeable": cluster_mean(defined, lambda r, v=variant: r[v] is not None),
                       "consistent": cluster_mean(judged, lambda r, v=variant: r[v] == r["gold"]),
                       "consistent_unjudgeable_as_fail": cluster_mean(defined, lambda r, v=variant: r[v] == r["gold"]),
                       "tool_keeps_order": cluster_mean(judged, lambda r, v=variant: r[v] == "same")}
    q1["gold_keeps_order"] = cluster_mean(defined, lambda r: r["gold"] == "same")
    judged = [row for row in defined if row["strict"] is not None]
    q1["strict_by_layer"] = {layer: {"judgeable": cluster_mean([r for r in defined if r["layer"] == layer], lambda r: r["strict"] is not None),
                                     "consistent": cluster_mean([r for r in judged if r["layer"] == layer], lambda r: r["strict"] == r["gold"])}
                             for layer in ("adjacent", "same_run", "revisit")}
    q1["strict_by_separable"] = {str(flag): {"judgeable": cluster_mean([r for r in defined if r["separable"] == flag], lambda r: r["strict"] is not None),
                                             "consistent": cluster_mean([r for r in judged if r["separable"] == flag], lambda r: r["strict"] == r["gold"])}
                                 for flag in (True, False)}
    q1["strict_by_gold_relation"] = {relation: cluster_mean([r for r in judged if r["gold"] == relation], lambda r: r["strict"] == r["gold"])
                                     for relation in ("same", "reversed")}
    decided = [row for row in rows["shot_pairs"] if row["tool_left"] is not None]
    mentioned = [row for row in decided if row["first_mentioned"] is not None]
    per_shot = {"pairs": len(rows["shot_pairs"]),
                "decided": cluster_mean(rows["shot_pairs"], lambda r: r["tool_left"] is not None),
                "obeys_oracle": cluster_mean(decided, lambda r: r["tool_left"] == r["gold_left"]),
                "obeys_oracle_undecided_as_fail": cluster_mean(rows["shot_pairs"], lambda r: r["tool_left"] == r["gold_left"]),
                "first_mentioned_on_left": cluster_mean(mentioned, lambda r: r["tool_left"] == r["first_mentioned"]),
                # 提及顺序与 oracle 左右相关时，不写布局也能「服从」；按 oracle 里先提到的人在左还是在右拆开读。
                "obeys_oracle_when_first_mentioned_is_oracle_left": cluster_mean(
                    [r for r in mentioned if r["first_mentioned"] == r["gold_left"]], lambda r: r["tool_left"] == r["gold_left"]),
                "obeys_oracle_when_first_mentioned_is_oracle_right": cluster_mean(
                    [r for r in mentioned if r["first_mentioned"] != r["gold_left"]], lambda r: r["tool_left"] == r["gold_left"]),
                "oracle_first_mentioned_on_left": cluster_mean([r for r in rows["shot_pairs"] if r["first_mentioned"] is not None],
                                                               lambda r: r["gold_left"] == r["first_mentioned"])}
    visible = [row for row in rows["presence"] if row["gold_visible"]]
    presence = {"script_present_bound": cluster_mean(rows["presence"], lambda r: r["bound"]),
                "real_visible_bound": cluster_mean(visible, lambda r: r["bound"]),
                "shots_with_enough_instances": cluster_mean(rows["shot_counts"], lambda r: r["instances"] >= r["script"]),
                "binding_source": _count_by(rows["presence"], "source")}
    bound = [row for row in rows["depth"] if row["both_bound"]]
    asserted = [row for row in bound if row["tool_front"] is not None]
    depth = {"gold_front_pairs": len(rows["depth"]), "both_bound": len(bound),
             "tool_has_front": cluster_mean(bound, lambda r: r["tool_front"] is not None),
             "same_front_among_bound": cluster_mean(bound, lambda r: r["tool_front"] == r["gold_front"]),
             "same_front_among_tool_front": cluster_mean(asserted, lambda r: r["tool_front"] == r["gold_front"])}
    out = {"q1": q1, "per_shot": per_shot, "presence": presence, "depth": depth}
    if "q4" in rows:
        judged4 = [row for row in rows["q4"] if row["holds"] is not None]
        out["q4"] = {"tested": len(rows["q4"]), "judgeable": cluster_mean(rows["q4"], lambda r: r["holds"] is not None),
                     "changes_as_planned": cluster_mean(judged4, lambda r: r["holds"])}
    return out


def step1_comparison(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """两臂（同种子，或各自两种子合并）的配对比较，前者减后者。"""

    out: dict[str, Any] = {}
    for variant in Q1_VARIANTS:
        out[f"q1_{variant}"] = paired_difference(first["pairs"], second["pairs"],
                                                 usable=lambda r, v=variant: r["gold"] is not None and r[v] is not None,
                                                 value=lambda r, v=variant: r[v] == r["gold"])
        out[f"q1_{variant}_unjudgeable_as_fail"] = paired_difference(first["pairs"], second["pairs"], usable=lambda r: r["gold"] is not None,
                                                                     value=lambda r, v=variant: r[v] == r["gold"])
    out["q1_strict_judgeable_rate"] = paired_difference(first["pairs"], second["pairs"], usable=lambda r: r["gold"] is not None,
                                                        value=lambda r: r["strict"] is not None)
    out["per_shot_obeys_oracle"] = paired_difference(first["shot_pairs"], second["shot_pairs"], usable=lambda r: r["tool_left"] is not None,
                                                     value=lambda r: r["tool_left"] == r["gold_left"])
    for side, same in (("left", True), ("right", False)):
        out[f"per_shot_obeys_oracle_when_first_mentioned_is_oracle_{side}"] = paired_difference(
            first["shot_pairs"], second["shot_pairs"],
            usable=lambda r, same=same: r["tool_left"] is not None and r["first_mentioned"] is not None
            and (r["first_mentioned"] == r["gold_left"]) == same,
            value=lambda r: r["tool_left"] == r["gold_left"])
    out["per_shot_obeys_oracle_undecided_as_fail"] = paired_difference(first["shot_pairs"], second["shot_pairs"], usable=lambda r: True,
                                                                       value=lambda r: r["tool_left"] == r["gold_left"])
    out["presence_script_present_bound"] = paired_difference(first["presence"], second["presence"], usable=lambda r: True,
                                                             value=lambda r: r["bound"])
    out["presence_real_visible_bound"] = paired_difference(first["presence"], second["presence"], usable=lambda r: r["gold_visible"],
                                                           value=lambda r: r["bound"])
    out["depth_same_front_among_bound"] = paired_difference(first["depth"], second["depth"], usable=lambda r: r["both_bound"],
                                                            value=lambda r: r["tool_front"] == r["gold_front"])
    if "q4" in first and "q4" in second:
        out["q4_changes_as_planned"] = paired_difference(first["q4"], second["q4"], usable=lambda r: r["holds"] is not None,
                                                         value=lambda r: r["holds"])
    return out


def annotate_numbered_reference(rows: Mapping[str, Any], windows: Mapping[str, Any],
                                compiled_prompts: Mapping[str, Sequence[str]]) -> None:
    """给逐镜主体对的行标上：编译器空间句（S-compiled 臂的文字）里这两人有没有被退回成 ``Subject k`` 编号指代。
    ``True`` 有一人用了编号，``False`` 两人都用短属性词，``None`` 这一镜没有空间句。各臂的行都按同一份文字标，便于同一批镜上比。"""

    from mspc.evaluation.cross_shot_layout import numbered_space_references

    index = {(story_id, shot["shot_id"]): k for story_id, window in windows["windows"].items()
             for k, shot in enumerate(window["shots"])}
    for row in rows["shot_pairs"]:
        numbered = numbered_space_references(compiled_prompts[row["story"]][index[(row["story"], row["shot"])]])
        row["numbered_reference"] = None if numbered is None else bool(numbered & set(row["pair"]))


def numbered_reference_block(rows: Mapping[str, Any]) -> dict[str, Any]:
    """逐镜服从率按编译器空间句的指代方式分开：短属性词对不上、退回 ``Subject k`` 的镜，与两人都用短属性词的镜。"""

    out = {}
    for name, flag in (("numbered", True), ("phrase", False)):
        subset = [row for row in rows["shot_pairs"] if row["numbered_reference"] is flag]
        decided = [row for row in subset if row["tool_left"] is not None]
        out[name] = {"pairs": len(subset), "decided": cluster_mean(subset, lambda r: r["tool_left"] is not None),
                     "obeys_oracle": cluster_mean(decided, lambda r: r["tool_left"] == r["gold_left"])}
    return out


def numbered_reference_comparison(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    return {name: paired_difference(first["shot_pairs"], second["shot_pairs"],
                                    usable=lambda r, flag=flag: r["tool_left"] is not None and r["numbered_reference"] is flag,
                                    value=lambda r: r["tool_left"] == r["gold_left"])
            for name, flag in (("numbered", True), ("phrase", False))}


def _with_seed(rows: Mapping[str, Any], seed: str) -> dict[str, Any]:
    """逐单元行的单元名前加种子（两种子合并时同一单元各算一项，窗仍是聚类单位）。"""

    return {key: [{**row, "unit": f"{seed}|{row['unit']}", "seed": seed} for row in value]
            for key, value in rows.items() if isinstance(value, list)}


def _merge(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {key: [row for part in parts for row in part[key]] for key in parts[0]}


def _arm_prompts(path: Path) -> dict[str, list[str]]:
    return {case["story_id"]: list(case["prompts"]) for case in _read(path)["cases"]}


def state_judge_rows(folder: Path) -> dict[str, Any]:
    """长集状态判官一组的产物目录 -> 逐断言行（通过＝证据充分且与期望一致）；调用没解析出来的故事整条不出行、单列。"""

    prepared, results = _read(folder / "prepared.json"), _read(folder / "judge_results.json")
    rows, failed = [], []
    for call in prepared["calls"]:
        row = results.get(call["call_id"])
        if row is None or row["status"] != "parsed":
            failed.append(call["story_id"])
            continue
        for assertion in call["assertions"]:
            answer = row["answers"][assertion["assertion_id"]]
            sufficient = bool(answer["sufficient_evidence"])
            rows.append({"unit": f"{call['story_id']}|{assertion['assertion_id']}", "story": call["story_id"],
                         "kind": assertion["kind"], "sufficient": sufficient,
                         "passed": sufficient and answer["holds"] == (assertion["expected_answer"] == "yes")})
    return {"rows": rows, "failed_stories": sorted(failed), "calls": len(prepared["calls"]),
            "skipped_assertions": prepared["shot_exclusions"]["skipped_assertions"]}


def run_step1_readout(args: argparse.Namespace) -> int:
    from mspc.evaluation.cross_shot_layout import step1_verdict

    if args.out.exists():
        raise FileExistsError(args.out)
    gold = _read(args.gold)["stories"]
    oracle = _read(args.oracle)
    excluded = frozen_shot_exclusions(args.freeze)
    identity = _read(args.calibration)["identity"]
    compiled_prompts = _arm_prompts(args.arms_dir / "arm_S-compiled.json")
    renders: dict[str, dict[str, Any]] = {}
    for item in args.render:
        name, work = item.split("=", 1)
        arm, seed = name.rsplit("_seed", 1)
        if arm not in ARM_AXES:
            raise ValueError(f"unknown arm in render name {name!r}")
        work = Path(work)
        windows = _read(work / "windows.json")
        measures = chain_measures(work)
        # 提及顺序读逐镜散文（S-first 的首句按从左到右写，不算散文里的提及）。
        rows = step1_rows(windows, measures, gold, excluded, _arm_prompts(args.arms_dir / f"arm_{ARM_PROSE[arm]}.json"), identity)
        rows["q4"] = q4_rows(windows, measures, _read(work / "heads.json")["stories"], oracle, gold, excluded)
        annotate_numbered_reference(rows, windows, compiled_prompts)
        renders[name] = {"arm": arm, "seed": seed, "work": str(work), "rows": rows}
    real_windows = _read(args.real_work / "windows.json")
    in_plan = {story: sorted(shots) for story, shots in excluded.items() if story in real_windows["windows"]}
    # 真片：同一版抽取链、同一套口径读天花板 R；提及顺序按冻结原文（P 臂文字）。
    real_rows = step1_rows(real_windows, chain_measures(args.real_work), gold, excluded,
                           _arm_prompts(args.arms_dir / "arm_P.json"), identity)
    annotate_numbered_reference(real_rows, real_windows, compiled_prompts)
    arms: dict[str, dict[str, Any]] = {}
    for name, entry in renders.items():
        arms.setdefault(entry["arm"], {})[entry["seed"]] = entry["rows"]
    pooled = {arm: _merge([_with_seed(rows, seed) for seed, rows in sorted(seeds.items())]) for arm, seeds in arms.items()}
    real_judged = [row for row in real_rows["pairs"] if row["gold"] is not None and row["strict"] is not None]

    def against_ceiling(rows: Mapping[str, Any]) -> dict[str, Any]:
        judged = [row for row in rows["pairs"] if row["gold"] is not None and row["strict"] is not None]
        return cluster_rate_gap(judged, real_judged, value=lambda r: r["strict"] == r["gold"])

    comparisons: dict[str, Any] = {}
    for first, second in (("S", "N"), ("P", "N"), ("S-LR", "N"), ("S", "P"), ("S", "S-LR"), ("S-first", "N"), ("S-first", "S"),
                          ("S-compiled", "N"), ("S-compiled", "S-first"), ("S-compiled", "S")):
        if first not in arms or second not in arms:
            raise ValueError(f"comparison {first}-{second} needs both arms")
        shared = sorted(set(arms[first]) & set(arms[second]))
        block = {f"seed{seed}": step1_comparison(arms[first][seed], arms[second][seed]) for seed in shared}
        if len(shared) >= 2:
            block["pooled"] = step1_comparison(_merge([_with_seed(arms[first][s], s) for s in shared]),
                                               _merge([_with_seed(arms[second][s], s) for s in shared]))
        comparisons[f"{first}-{second}"] = block
    main = comparisons["S-N"]["pooled"]
    ceiling = against_ceiling(pooled["S"])
    guardrails = {"presence_script_present_bound": main["presence_script_present_bound"]["reading"]}
    state_judge: dict[str, Any] = {}
    judged = {name: state_judge_rows(Path(folder)) for name, folder in (item.split("=", 1) for item in args.state_judge)}
    for name, entry in judged.items():
        state_judge[name] = {"calls": entry["calls"], "failed_stories": entry["failed_stories"],
                             "skipped_assertions": entry["skipped_assertions"],
                             "passed": cluster_mean(entry["rows"], lambda r: r["passed"]),
                             "sufficient_evidence": cluster_mean(entry["rows"], lambda r: r["sufficient"]),
                             "passed_by_kind": {kind: cluster_mean([r for r in entry["rows"] if r["kind"] == kind], lambda r: r["passed"])
                                                for kind in sorted({r["kind"] for r in entry["rows"]})}}
    if judged:
        if set(judged) != {"S", "N"}:
            raise ValueError("--state-judge takes exactly S=<dir> and N=<dir>")
        # 任一臂调用失败的故事整条剔除（同带宽阶梯状态主判的配对口径）：失败故事在该臂没有行，取共有单元即可。
        state_judge["S-N"] = paired_difference(judged["S"]["rows"], judged["N"]["rows"], usable=lambda r: True,
                                               value=lambda r: r["passed"])
        guardrails["state_assertions_j2"] = state_judge["S-N"]["reading"]
    readout = {
        "created_at": _now(), "gold": str(args.gold), "oracle": str(args.oracle), "freeze": str(args.freeze),
        "freeze_revision": _read(args.freeze)["revision"],
        "excluded_shots": {"count": sum(len(v) for v in in_plan.values()), "by_story": in_plan},
        "identity_rule": "strict: every subject pair needs the in-shot binding and the cross-shot DINOv2 one-to-one mapping to agree",
        "real": {"work": str(args.real_work), **step1_block(real_rows),
                 "excluded_shots_skipped": real_rows["excluded_shots_skipped"],
                 "excluded_shot_pairs_skipped": real_rows["excluded_shot_pairs_skipped"]},
        "renders": {name: {"arm": entry["arm"], "seed": entry["seed"], "work": entry["work"], **step1_block(entry["rows"]),
                           "against_ceiling_R": against_ceiling(entry["rows"])}
                    for name, entry in sorted(renders.items())},
        "arms_pooled": {arm: {"seeds": sorted(arms[arm]), **step1_block(rows), "against_ceiling_R": against_ceiling(rows)}
                        for arm, rows in sorted(pooled.items())},
        "comparisons": comparisons,
        "state_judge_j2": state_judge,
        "verdict": {"main_S_minus_N_pooled": main["q1_strict"], "S_against_ceiling_R": ceiling, "guardrails": guardrails,
                    "verdict": step1_verdict(main["q1_strict"]["reading"], ceiling["reading"], list(guardrails.values())),
                    "main_reading_under_other_rules": {key: main[key]["reading"] for key in sorted(main) if key.startswith("q1_")}},
        # 补充臂 S-first（判定之后加的，不改上面的判定）：同一套读数对 N、对 S。
        "supplement_S_first": {"against_N_pooled": comparisons["S-first-N"]["pooled"],
                               "against_S_pooled": comparisons["S-first-S"]["pooled"],
                               "against_ceiling_R": against_ceiling(pooled["S-first"])},
        # 验证臂 S-compiled：文字由改写法后的编译器出。另按空间句的指代方式（退回 Subject k／短属性词）分开读各臂。
        "supplement_S_compiled": {
            "against_N_pooled": comparisons["S-compiled-N"]["pooled"],
            "against_S_first_pooled": comparisons["S-compiled-S-first"]["pooled"],
            "against_ceiling_R": against_ceiling(pooled["S-compiled"]),
            "by_reference": {"real": numbered_reference_block(real_rows),
                             **{arm: numbered_reference_block(rows) for arm, rows in sorted(pooled.items())}},
            "by_reference_S_compiled_minus_N": numbered_reference_comparison(pooled["S-compiled"], pooled["N"]),
            "by_reference_S_compiled_minus_S_first": numbered_reference_comparison(pooled["S-compiled"], pooled["S-first"])},
        "rows": {"real": real_rows, **{name: entry["rows"] for name, entry in sorted(renders.items())}},
    }
    _write_new(args.out, readout)
    print(json.dumps({key: readout[key] for key in ("verdict", "comparisons")}, ensure_ascii=False, indent=1))
    return 0


# ---------------------------------------------------------------------------
# heads（GPU，renderer_public_metrics）：中帧「a head」头框（Q4 描述性读数，量法同第 0 步）
# ---------------------------------------------------------------------------


def run_heads(args: argparse.Namespace) -> int:
    import numpy as np
    from PIL import Image

    from mspc.evaluation.renderer_qualification_official_backend import (
        DEFAULT_ROI_DETECTOR_CONFIG,
        FROZEN_PUBLIC_ROI_DETECTOR,
        _DinoRoiEngine,
        _model_batch_from_roi_config,
    )
    from mspc.evaluation.spatial_layout_probe import HEAD_PHRASE
    from tools.run import spatial_facts as chain

    out = args.work / "heads.json"
    if out.exists():
        raise FileExistsError(out)
    batch = _model_batch_from_roi_config(DEFAULT_ROI_DETECTOR_CONFIG)
    engine = _DinoRoiEngine(args.resource_root, args.device)
    stories: dict[str, Any] = {}
    for story_id, _window, shot in chain._shots(args.work, ""):
        image = np.asarray(Image.open(chain._frame_path(args.work, story_id, shot["shot_id"], shot["layout_positions"][1])).convert("RGB"))
        (heads,) = engine.grounded_regions([image], [HEAD_PHRASE], FROZEN_PUBLIC_ROI_DETECTOR, detector_frame_batch_size=1,
                                           dino_crop_batch_size=batch.dino_crop_batch_size)
        engine.clear_frame_caches()
        stories.setdefault(story_id, {})[shot["shot_id"]] = [{"box": [round(v, 2) for v in h["box"]], "score": round(h["score"], 4)}
                                                             for h in heads]
    _write_new(out, {"created_at": _now(), "head_phrase": HEAD_PHRASE, "roi_detector": FROZEN_PUBLIC_ROI_DETECTOR.payload(),
                     "stories": stories})
    print(json.dumps({"shots": sum(len(v) for v in stories.values())}))
    return 0


# ---------------------------------------------------------------------------
# step1-sheets（CPU）：亲看用的镜对并排帧
# ---------------------------------------------------------------------------


def run_step1_sheets(args: argparse.Namespace) -> int:
    """每份出片按种子抽 :data:`STEP1_VISUAL_PAIRS` 个金标关系有定义的镜对，两镜中帧并排，画上指认到的主体框（彩色）与
    未指认的实例框（灰），下方写主体描述、工具逐镜左边那位、认人口径与关系、金标关系，供亲看核对自动判定。"""

    from PIL import Image, ImageDraw

    from mspc.data.case_sampling import sample_case_ids
    from tools.run import spatial_facts as chain

    readout = _read(args.readout)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font, small = _font(20), _font(17)
    index = []
    for name in args.render:
        work = Path(readout["renders"][name]["work"])
        windows = _read(work / "windows.json")["windows"]
        measures = chain_measures(work)
        units: dict[str, list[dict[str, Any]]] = {}
        for row in readout["rows"][name]["pairs"]:
            if row["gold"] is not None:
                units.setdefault("|".join([row["story"], *row["shots"]]), []).append(row)
        sampled = sample_case_ids(sorted(units), size=STEP1_VISUAL_PAIRS, seed=args.seed)
        panels = []
        for number, key in enumerate(sampled, start=1):
            story_id, si, sj = key.split("|")
            subjects = {row["subject_id"]: row["attributes"] for row in windows[story_id]["subjects"]}
            tiles = []
            for sid in (si, sj):
                m = measures[(story_id, sid)]
                image = Image.open(chain._frame_path(work, story_id, sid, m["mid_position"])).convert("RGB")
                draw = ImageDraw.Draw(image)
                bound = {t for t in m["binding"].values() if t is not None}
                for track, box in m["all_mid_boxes"].items():
                    if track not in bound:
                        draw.rectangle(box, outline=(200, 200, 200), width=2)
                for subject, box in m["mid_boxes"].items():
                    if box is None:
                        continue
                    color = _COLORS[int(subject.split()[-1]) % len(_COLORS)]
                    draw.rectangle(box, outline=color, width=5)
                    draw.rectangle((box[0], box[1], box[0] + 52, box[1] + 28), fill=color)
                    draw.text((box[0] + 5, box[1] + 3), f"S{subject.split()[-1]}", fill=(255, 255, 255), font=font)
                draw.rectangle((0, 0, 70, 30), fill=(0, 0, 0))
                draw.text((6, 4), sid, fill=(255, 255, 0), font=font)
                tiles.append(image)
            involved = sorted({s for row in units[key] for s in row["pair"]})
            lines = [f"#{number} {name} {key} ({units[key][0]['layer']})"]
            lines += [f"S{s.split()[-1]}: {subjects[s][:150]}" for s in involved]
            for row in units[key]:
                a, b = (f"S{s.split()[-1]}" for s in row["pair"])
                left = ["?" if v is None else f"S{v.split()[-1]}" for v in row["left"]]
                verdict = "UNJUDGEABLE" if row["strict"] is None else "CONSISTENT" if row["strict"] == row["gold"] else "INCONSISTENT"
                lines.append(f"{a}/{b}: tool left {si}={left[0]} {sj}={left[1]} | strict={row['strict']} ({row['strict_reason']}) "
                             f"path1={row['path1']} | plan(real)={row['gold']} -> {verdict}")
                index.append({"render": name, "number": number, "shot_pair": key, "unit": row["unit"], "left": row["left"],
                              "strict": row["strict"], "strict_reason": row["strict_reason"], "path1": row["path1"],
                              "gold": row["gold"], "verdict": verdict})
            width = sum(tile.size[0] for tile in tiles) + 8
            panel = Image.new("RGB", (width, tiles[0].size[1] + 24 * len(lines) + 10), (20, 20, 20))
            panel.paste(tiles[0], (0, 0))
            panel.paste(tiles[1], (tiles[0].size[0] + 8, 0))
            draw = ImageDraw.Draw(panel)
            for k, line in enumerate(lines):
                draw.text((6, tiles[0].size[1] + 6 + 24 * k), line, fill=(255, 255, 255), font=small)
            panels.append(panel)
        per = args.pairs_per_sheet
        for start in range(0, len(panels), per):
            chunk = panels[start:start + per]
            sheet = Image.new("RGB", (max(p.size[0] for p in chunk), sum(p.size[1] + 6 for p in chunk)), (90, 90, 90))
            y = 0
            for panel in chunk:
                sheet.paste(panel, (0, y))
                y += panel.size[1] + 6
            sheet.save(args.out_dir / f"{name}_{start // per:02d}.jpg", quality=88)
    _write_new(args.out_dir / "index.json", {"created_at": _now(), "seed": args.seed, "readout": str(args.readout), "items": index})
    print(json.dumps({"renders": len(args.render), "units": len(index)}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)

    p = steps.add_parser("frames")
    p.add_argument("--pairs", type=Path, required=True, help="dev 30 A/B 配对清单 ab_pairs30.json（取 context_b）")
    p.add_argument("--freeze", type=Path, required=True, help="configs/benchmark/script_state_bench_evaluation_freeze_v1_r3.json")
    p.add_argument("--candidate-r3", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(handler=run_frames)

    p = steps.add_parser("sam3")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bpe", type=Path, required=True)
    p.add_argument("--variants", required=True, help="orig 或 orig,mirror")
    p.add_argument("--stories", default="", help="逗号分隔；空为全部")
    p.add_argument("--phrase-mode", choices=("description", "short"), required=True,
                   help="逐主体概念提示：剧本长描述，或规则抽取的短属性词")
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(handler=run_sam3)

    p = steps.add_parser("embed")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--instances", type=Path, required=True)
    p.add_argument("--resource-root", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(handler=run_embed)

    p = steps.add_parser("gdino")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--stories", required=True)
    p.add_argument("--resource-root", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(handler=run_gdino)

    p = steps.add_parser("compare-sheet")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--instances", type=Path, required=True)
    p.add_argument("--gdino", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(handler=run_compare_sheet)

    p = steps.add_parser("bind")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--instances", type=Path, required=True)
    p.add_argument("--variants", required=True)
    p.add_argument("--base-url", required=True, help="本地 vLLM，如 http://127.0.0.1:8189/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(handler=run_bind)

    p = steps.add_parser("crops")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(handler=run_crops)

    p = steps.add_parser("orient-local")
    p.add_argument("--crops", type=Path, required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--shuffle-seed", type=int, required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(handler=run_orient_local)

    p = steps.add_parser("orient-judge")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--crops", type=Path, required=True)
    p.add_argument("--pass-seed", type=int, required=True)
    p.add_argument("--api-key-file", type=Path, required=True)
    p.add_argument("--proxy", required=True)
    p.add_argument("--ledger", type=Path, required=True, help="本实验全部判官调用的费用账（跨遍累加，进预算闸）")
    p.add_argument("--limit-calls", type=int)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(handler=run_orient_judge)

    p = steps.add_parser("sheets")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--instances", type=Path, required=True)
    p.add_argument("--crops", type=Path)
    p.add_argument("--shots-per-sheet", type=int, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(handler=run_sheets)

    p = steps.add_parser("gold")
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--corrections", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(handler=run_gold)

    p = steps.add_parser("q4-items")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--seed", required=True)
    p.add_argument("--freeze", type=Path, required=True, help="现行冻结清单（修订 4 起带 excluded_shots）")
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(handler=run_q4_items)

    p = steps.add_parser("calibrate")
    for name in ("plan", "instances", "binding", "embed", "gold", "crops", "oracle", "freeze", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--judge", type=Path, nargs="+", required=True, help="J2 两遍的产物")
    p.add_argument("--qwen", type=Path)
    p.add_argument("--q4-items", type=Path)
    p.add_argument("--q4-labels", type=Path)
    p.set_defaults(handler=run_calibrate)

    p = steps.add_parser("q3-layout")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--metadata-csv", type=Path, required=True, help="205k 池 metadata.csv：找同一源视频的其他镜估叠加层")
    p.add_argument("--resource-root", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(handler=run_q3_layout)

    p = steps.add_parser("strip")
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True, help="configs/planner/shot_tier_rewrite.json（三档反写同配置）")
    p.add_argument("--api-key-file", type=Path, required=True)
    p.add_argument("--proxy", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(handler=run_strip)

    p = steps.add_parser("build-arms")
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--candidate-r3", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--strip-final", type=Path, required=True, help="人工复核后的删方位定稿（每镜 original/final）")
    p.add_argument("--tokenizer-path", type=Path, required=True)
    p.add_argument("--arms", required=True, help="如 N,S-first,S-compiled")
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(handler=run_build_arms)

    p = steps.add_parser("heads")
    p.add_argument("--work", type=Path, required=True, help="tools/run/spatial_facts.py 的一个工作目录")
    p.add_argument("--resource-root", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.set_defaults(handler=run_heads)

    p = steps.add_parser("step1-readout")
    p.add_argument("--render", action="append", required=True, help="<臂>_seed<种子>=<抽取链工作目录>，可重复")
    p.add_argument("--real-work", type=Path, required=True, help="真片 dev 30 用同一版抽取链跑出的工作目录（天花板 R）")
    p.add_argument("--arms-dir", type=Path, required=True, help="build-arms 的产物目录（arm_<臂>.json，取逐镜文字读提及顺序）")
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True, help="第 0 步 oracle.json（真片中帧头框，Q4 用）")
    p.add_argument("--calibration", type=Path, required=True, help="抽取链 dev 30 标定（三路外观合成，敏感性口径用）")
    p.add_argument("--freeze", type=Path, required=True, help="现行冻结清单（修订 4 起带 excluded_shots）")
    p.add_argument("--state-judge", action="append", default=[],
                   help="可选的 J2 状态护栏：S=<判官产物目录> 与 N=<判官产物目录>（各一个种子）；不给就不进护栏")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(handler=run_step1_readout)

    p = steps.add_parser("step1-sheets")
    p.add_argument("--readout", type=Path, required=True)
    p.add_argument("--render", action="append", required=True, help="readout 里的出片名，可重复")
    p.add_argument("--seed", required=True)
    p.add_argument("--pairs-per-sheet", type=int, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(handler=run_step1_sheets)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
