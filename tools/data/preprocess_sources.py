"""前置处理的主入口：prepare 列出场景；measure 每条源片解码一遍量齐并按镜判合格；collect 出分来源计数；
windows 在打分与定线之后装窗并拼代理片；blocks 从判过且过质量线的镜按连续段均分成块。

measure 进程只用 CPU 解码，切镜网络常驻在 CUDA_VISIBLE_DEVICES 给的那一张卡上；按源片路径分片，按源片落盘、可续跑。
打分、定线、装窗、代理片是第二遍，读这里产出的合格镜清单。
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import zlib

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools/data"))


def read(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path, rows):
    temporary = Path(path).with_suffix(".partial")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def dump(path, value):
    temporary = Path(path).with_suffix(".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")
    temporary.replace(path)


def probe_codec(ffprobe, video):
    return subprocess.check_output([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=codec_name", "-of", "csv=p=0", video], text=True).strip()


def prepare(cfg, root):
    """把影视场景清单与 CineDance 标注统一成场景行；烧录字幕的场景在这里排除。

    评测隔离不在这里做，隔离是片段级（片段与窗不重复即可），由装窗之后的查重一步按源片帧段
    与画面相似度给出要排除的窗。源片编号一经分配不再变：重跑时新进来的源片接在已有编号后面，已有的窗键、代理片、标注都还对得上。
    """
    from mspc.benchmark.long_story_holdout import SUBTITLE_FLAG
    from build_cinedance_windows import SourceTimelineError, converted_shots

    film = json.loads((REPO / cfg["film_config"]).read_text())
    scenes, skipped = [], Counter()
    for index, row in enumerate(read(film["manifest"])):
        if SUBTITLE_FLAG in row["video_flags"]:
            skipped["film_burned_in_subtitles"] += 1
        else:
            scenes.append({"family": "film", "scene_index": index, "seed": film["seed"], "row": row})
    for package in cfg["cinedance_packages"]:
        index = 0
        for probe in read(Path(cfg["data_root"]) / package["clips"]):
            annotation = json.loads((Path(cfg["data_root"]) / package["annotations"] / f"{probe['clip_id']}.json").read_text())
            stamps = probe["frame_timestamps_sec"]
            if stamps is not None and any(a >= b for a, b in zip(stamps, stamps[1:])):
                skipped["cinedance_timeline_not_increasing"] += 1
                continue
            try:
                shots = converted_shots(annotation, probe)
            except SourceTimelineError:
                skipped["cinedance_annotation_outside_timeline"] += 1
                continue
            blocks = []
            for shot in shots:  # 相同 scene 标签重新出现仍是不同连续场景
                if not blocks or blocks[-1][-1]["scene"] != shot["scene"]:
                    blocks.append([])
                blocks[-1].append({**shot, "flags": []})
            for number, block in enumerate(blocks):
                scenes.append({"family": "cinedance", "scene_index": index + number, "seed": cfg["cinedance_seed"],
                               "codec": probe["codec"],
                               "row": {"scene_id": f"cinedance:{annotation['video_id']}:{block[0]['scene']}:block{number:04d}",
                                       "source": "cinedance", "source_video_path": probe["source_video_path"],
                                       "fps": probe["fps"], "container_preroll_frames": None,
                                       "width": probe["width"], "height": probe["height"],
                                       "video_frame_count": probe["num_frames"],
                                       "active_box": [0, 0, probe["width"], probe["height"]], "shots": block}})
            index += len(blocks)
    videos = sorted({s["row"]["source_video_path"] for s in scenes if "codec" not in s})
    with ThreadPoolExecutor(max_workers=32) as pool:
        codecs = dict(zip(videos, pool.map(lambda v: probe_codec(cfg["ffprobe"], v), videos)))
    order = {}
    if (root / "scenes.jsonl").exists():
        for scene in read(root / "scenes.jsonl"):
            order[scene["row"]["source_video_path"]] = int(scene["sid"].rsplit("-", 1)[1])
        if sorted(order.values()) != list(range(len(order))):
            raise ValueError("existing source numbering is not contiguous")
    kept = len(order)
    for scene in scenes:
        video = scene["row"]["source_video_path"]
        scene.setdefault("codec", codecs.get(video))
        scene["sid"] = f"{scene['family']}-{order.setdefault(video, len(order)):06d}"
    root.mkdir(parents=True, exist_ok=True)
    write(root / "scenes.jsonl", scenes)
    per_source = {s["sid"]: s for s in scenes}.values()
    summary = {"scenes": len(scenes), "sources": len(order), "sources_numbered_before": kept, "skipped": dict(skipped),
               "scenes_by_family": dict(Counter(s["family"] for s in scenes)),
               "sources_by_codec": {f"{a}:{b}": n for (a, b), n in Counter((s["family"], s["codec"]) for s in per_source).items()},
               "frames": {f: sum(v) for f, v in _frames_by_family(scenes).items()}}
    dump(root / "prepare.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


def _frames_by_family(scenes):
    seen, totals = set(), defaultdict(list)
    for scene in scenes:
        if scene["sid"] not in seen:
            seen.add(scene["sid"])
            totals[scene["family"]].append(scene["row"]["video_frame_count"])
    return totals


def process_source(sid, scenes, cfg, root, engine, lock):
    import numpy as np
    from mspc.benchmark.long_story_holdout import render_frame_count
    from mspc.data.scene_windows import training_runs
    from mspc.data.source_pipeline import MEASURE_VERSION, decode_measure, judge_scene

    out = root / "sources" / sid
    if (out / "done.json").exists():
        return json.loads((out / "done.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    first = scenes[0]["row"]
    video = first["source_video_path"]
    measure = decode_measure(video, width=first["width"], height=first["height"], ffmpeg=cfg["ffmpeg"],
                             engine=engine, engine_lock=lock)
    measured = time.monotonic()
    with (out / "measure.partial").open("wb") as handle:
        np.savez_compressed(handle, single=measure["single"], content=measure["content"], luma=measure["luma"],
                            metadata=json.dumps({"video": video, "version": MEASURE_VERSION, "codec": scenes[0]["codec"],
                                                 "active_box": measure["active_box"], "frame_count": measure["frame_count"],
                                                 "fps": first["fps"]}))
    (out / "measure.partial").replace(out / "measure.npz")
    shot_rows, reasons = [], Counter()
    for scene in scenes:
        judged = judge_scene(scene["row"], measure, thresholds=cfg["thresholds"], luma_range=tuple(cfg["luma_range"]))
        _, rejected = training_runs(judged)
        why = {r["source_shot_index"]: r["reason"] for r in rejected}
        for shot in judged["shots"]:
            reason = why.get(shot["source_shot_index"])
            reason = "+".join(shot["flags"]) if reason == "source_flags" else reason
            row = {"sid": sid, "family": scene["family"], "source_video_path": video, "scene_id": judged["scene_id"],
                   "source_shot_index": shot["source_shot_index"], "fps": judged["fps"],
                   "width": judged["width"], "height": judged["height"], "active_box": judged["active_box"],
                   "start_frame": shot["start_frame"], "end_frame": shot["end_frame"], "flags": shot["flags"],
                   "mean_luma": shot["mean_luma"], "min_luma": shot["min_luma"], "max_luma": shot["max_luma"],
                   "gate": shot["gate"], "eligible": reason is None, "reason": reason}
            if reason is None:
                start, end = shot["gate"]["revised_segment"]
                row.update(segment=[start, end], render_frames=render_frame_count((end - start) / judged["fps"]))
            shot_rows.append(row)
            reasons[reason or "eligible"] += 1
    write(out / "shots.jsonl", shot_rows)
    done = {"sid": sid, "video": video, "family": scenes[0]["family"], "frames": measure["frame_count"],
            "manifest_frames": first["video_frame_count"], "active_box": measure["active_box"],
            "full_frame": measure["active_box"] == [0, 0, first["width"], first["height"]],
            "shots": len(shot_rows), "eligible": reasons["eligible"], "shot_reasons": dict(reasons),
            "seconds": {**measure["seconds"], "judge": time.monotonic() - measured}}
    dump(out / "done.json", done)
    return done


def measure(cfg, root, shard, workers, limit):
    import cv2
    from mspc.data.transnet_gpu_scoring import PersistentTransNet

    cv2.setNumThreads(1)  # 每条源片一个线程自己缩小画面；OpenCV 默认按整机核数再开线程会互相抢
    if not os.environ["CUDA_VISIBLE_DEVICES"].isdigit():
        raise ValueError("measure needs exactly one visible GPU")
    index, count = shard
    groups = defaultdict(list)
    for scene in read(root / "scenes.jsonl"):
        if zlib.crc32(scene["row"]["source_video_path"].encode()) % count == index:
            groups[scene["sid"]].append(scene)
    order = sorted(groups, key=lambda sid: -groups[sid][0]["row"]["video_frame_count"])
    if limit:
        order = order[::max(1, len(order) // limit)][:limit]
    tag = f"s{index}of{count}"
    (root / "logs").mkdir(exist_ok=True); (root / "status").mkdir(exist_ok=True)
    engine = PersistentTransNet(python=cfg["transnet_python"], root=cfg["resource_root"], device="cuda:0",
                                batch_windows=cfg["batch_windows"], log=root / "logs" / f"transnet-{tag}.log")
    lock, started, totals, failures = threading.Lock(), time.monotonic(), Counter(), []
    written = started

    def one(sid):
        try:
            return process_source(sid, groups[sid], cfg, root, engine, lock)
        except Exception as error:
            dump(root / "sources" / sid / "error.json", {"sid": sid, "video": groups[sid][0]["row"]["source_video_path"],
                                                           "error": f"{type(error).__name__}: {error}"[:2000]})
            return {"sid": sid, "error": True}

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for n, future in enumerate(as_completed([pool.submit(one, sid) for sid in order]), 1):
                done = future.result()
                if "error" in done:
                    failures.append(done["sid"])
                else:
                    totals.update(frames=done["frames"], shots=done["shots"], eligible=done["eligible"])
                if time.monotonic() - written >= 10 or n == len(order):
                    written = seconds = time.monotonic()
                    seconds -= started
                    dump(root / "status" / f"measure-{tag}.json",
                         {"shard": list(shard), "sources_done": n, "sources_total": len(order), "failed": len(failures),
                          **totals, "seconds": seconds, "frames_per_second": totals["frames"] / seconds,
                          "state": "completed" if n == len(order) else "running"})
    finally:
        engine.close()
    print(json.dumps({"shard": list(shard), "sources": len(order), "failed": failures[:20], **totals}, ensure_ascii=False))


def collect(cfg, root):
    """合出第二遍要读的合格镜清单；被筛的镜连同原因留在各源片的 shots.jsonl。"""
    eligible, summary = [], defaultdict(Counter)
    for folder in sorted((root / "sources").iterdir()):
        family = folder.name.split("-")[0]
        if not (folder / "done.json").exists():
            summary[family]["sources_failed" if (folder / "error.json").exists() else "sources_unfinished"] += 1
            continue
        done = json.loads((folder / "done.json").read_text())
        counts = summary[family]
        counts.update(sources=1, frames=done["frames"], shots=done["shots"], sources_full_frame=int(done["full_frame"]),
                      frame_count_mismatch=int(done["frames"] != done["manifest_frames"]))
        for reason, n in done["shot_reasons"].items():
            counts["shot:" + reason] += n
        eligible.extend(row for row in read(folder / "shots.jsonl") if row["eligible"])
    (root / "final").mkdir(exist_ok=True)
    write(root / "final" / "eligible_shots.jsonl", eligible)
    result = {family: dict(counts) for family, counts in summary.items()}
    dump(root / "final" / "measure-summary.json", result)
    print(json.dumps(result, ensure_ascii=False))


SCORE_READERS = {"dover_technical": lambda r: r["dover"]["technical_logit"],
                 "laion_aesthetic": lambda r: r["laion"]["mean"]}


def concatenate_proxy(parts, target, ffmpeg):
    """各镜代理片 stream copy；时间戳与逐帧源帧对照共用同一拼接顺序。"""
    video = parts[0]["video"]
    if len(parts) > 1:
        video, listing = str(target), target.with_suffix(".txt")
        listing.write_text("".join("file '" + part["video"].replace("'", "'\\''") + "'\n" for part in parts))
        subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y", "-threads", "1", "-f", "concat", "-safe", "0",
                        "-i", str(listing), "-c", "copy", "-threads", "1", "-f", "mp4", video + ".partial"],
                       check=True, capture_output=True)
        os.replace(video + ".partial", video)
        listing.unlink()
    stamps, frames, cursor = [], [], 0
    for number, part in enumerate(parts):
        stamps.append({"shot": number, "start": cursor / 2, "end": (cursor + part["frames"]) / 2})
        frames += [{"shot": number, "source_frame": frame} for frame in part["source_frames"]]
        cursor += part["frames"]
    return {"video": video, "fps": 2, "proxy_width": 832, "proxy_height": 480, "crop_box": parts[0]["crop_box"],
            "shot_timestamps": stamps, "proxy_to_source_frames": frames}


def source_windows(sid, scenes, cfg, root, scores):
    """一条源片：读第一遍的读数重判（纯算术），加上质量线，装窗，并把各镜的代理片拼成窗的代理片。"""
    import numpy as np
    from mspc.data.source_pipeline import judge_scene, scene_windows

    out = root / "sources" / sid
    with np.load(out / "measure.npz", allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        measure = {"single": data["single"], "content": data["content"], "luma": data["luma"],
                   "active_box": meta["active_box"], "frame_count": meta["frame_count"]}
    eligible = [r for r in read(out / "shots.jsonl") if r["eligible"]]
    eligible.sort(key=lambda r: r["segment"][0])  # 与打分时的镜键同一规则
    key_of = {(r["scene_id"], r["source_shot_index"]): f"{sid}-e{n:04d}" for n, r in enumerate(eligible)}
    rows, reasons = [], Counter()
    (out / "window-proxy").mkdir(exist_ok=True)
    for scene in scenes:
        judged = judge_scene(scene["row"], measure, thresholds=cfg["thresholds"], luma_range=tuple(cfg["luma_range"]))
        lines = cfg["quality_lines"].get(judged["source"], {})
        for shot in judged["shots"]:
            key = key_of.get((judged["scene_id"], shot["source_shot_index"]))
            if key is None:
                continue
            score = scores.get(key)
            if score is None:
                shot["flags"].append("score_missing")
                reasons["score_missing"] += 1
                continue
            low = [name for name, limit in lines.items() if SCORE_READERS[name](score) < limit]
            shot["flags"] += [f"below_{name}" for name in low]
            reasons["+".join(low) or "passed_quality"] += 1
        built, audit = scene_windows(judged, measure, seed=scene["seed"], scene_index=scene["scene_index"],
                                     thresholds=cfg["thresholds"], cut_version=cfg["cut_version"],
                                     min_window_seconds=cfg["min_window_seconds"])
        reasons["window_below_min_seconds"] += len(audit.get("short_windows", []))
        for window in built:
            kept = [scores[key_of[(judged["scene_id"], s["source_shot_index"])]] for s in window["shots"]]
            parts = [score["proxy"] for score in kept]
            # 分数跟着窗走：训练时可以再按更高的线选窗（例如每镜 LAION 都不低于 4.5），不用回头查表。
            for shot, score in zip(window["shots"], kept):
                shot.update({name: read(score) for name, read in SCORE_READERS.items()})
            window["quality"] = {f"min_{name}": min(shot[name] for shot in window["shots"]) for name in SCORE_READERS}
            key = f"{sid}-w{len(rows):04d}"
            proxy = concatenate_proxy(parts, out / "window-proxy" / f"{key}.mp4", cfg["encode_ffmpeg"])
            rows.append({"key": key, "case_id": window["window_id"], "family": scene["family"], "source": window["source"],
                         "scene": window, "proxy": proxy})
    write(out / "windows.jsonl", rows)
    return {"sid": sid, "family": scenes[0]["family"], "windows": len(rows), "by_shots": dict(Counter(r["scene"]["shot_count"] for r in rows)),
            "shots_in_windows": sum(r["scene"]["shot_count"] for r in rows), "quality": dict(reasons)}


def windows(cfg, root, workers, limit):
    """第 5 步：全部源片装窗并拼代理片，合出 final/windows.jsonl。质量线读配置的 quality_lines（按来源、按分数的下限）。"""
    scores = {}
    for path in sorted((root / "scores").glob("s*of*.jsonl")):
        for row in read(path):
            if row["status"] == "ok":
                scores[row["shot_key"]] = {"dover": {"technical_logit": row["dover"]["technical_logit"]},
                                           "laion": {"mean": row["laion"]["mean"]}, "proxy": row["proxy"]}
    groups = defaultdict(list)
    for scene in read(root / "scenes.jsonl"):
        if (root / "sources" / scene["sid"] / "done.json").exists():
            groups[scene["sid"]].append(scene)
    if limit:
        groups = {sid: groups[sid] for sid in sorted(groups)[:limit]}
    started, written, totals, failures = time.monotonic(), time.monotonic(), defaultdict(Counter), []

    def one(sid):
        try:
            return source_windows(sid, groups[sid], cfg, root, scores)
        except Exception as error:
            return {"sid": sid, "error": f"{type(error).__name__}: {error}"[:500]}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, done in enumerate(pool.map(one, sorted(groups)), 1):
            if "error" in done:
                failures.append(done)
            else:
                counts = totals[done["family"]]
                counts.update(sources=1, windows=done["windows"], shots_in_windows=done["shots_in_windows"])
                counts.update({f"windows_{k}shot": v for k, v in done["by_shots"].items()})
                counts.update({f"quality:{k}": v for k, v in done["quality"].items()})
            if time.monotonic() - written >= 10 or n == len(groups):
                written = time.monotonic()
                dump(root / "status" / "windows-s0of1.json",
                     {"shard": [0, 1], "sources_done": n, "sources_total": len(groups), "failed": len(failures),
                      "windows": sum(c["windows"] for c in totals.values()), "seconds": written - started,
                      "state": "completed" if n == len(groups) else "running"})
    (root / "final").mkdir(exist_ok=True)
    with (root / "final" / "windows.partial").open("w") as handle:
        for sid in sorted(groups):
            path = root / "sources" / sid / "windows.jsonl"
            if path.exists():
                handle.write(path.read_text())
    (root / "final" / "windows.partial").replace(root / "final" / "windows.jsonl")
    result = {"quality_lines": cfg["quality_lines"], "families": {k: dict(v) for k, v in totals.items()}, "failed": failures[:50]}
    dump(root / "final" / "windows-summary.json", result)
    print(json.dumps(result, ensure_ascii=False))


def scored_scenes(sid, scenes, saved, scores, cfg, excluded):
    """接回 judge_scene 已落盘的结果；质量分与打分键沿用 source_windows 装窗前的规则。"""
    eligible = sorted((r for r in saved if r["eligible"]), key=lambda r: r["segment"][0])
    key_of = {(r["scene_id"], r["source_shot_index"]): f"{sid}-e{n:04d}" for n, r in enumerate(eligible)}
    measured = {(r["scene_id"], r["source_shot_index"]): r for r in saved}
    result = []
    for scene in scenes:
        row = scene["row"]
        shots = []
        for original in row["shots"]:
            identity = (row["scene_id"], original["source_shot_index"])
            shot = {**original, **measured[identity]}
            shot["flags"] = list(shot["flags"])
            if shot["eligible"]:
                key = key_of[identity]
                shot["score_key"] = key
                score = scores.get(key)
                if score is None:
                    shot["flags"].append("score_missing")
                else:
                    shot.update({name: reader(score) for name, reader in SCORE_READERS.items()})
                    shot["flags"].extend(f"below_{name}" for name, limit in cfg["quality_lines"][row["source"]].items()
                                         if shot[name] < limit)
                if (sid, *identity) in excluded:
                    shot["flags"].append("eval_excluded")
            shots.append(shot)
        result.append({**row, "family": scene["family"], "active_box": shots[0]["active_box"], "shots": shots})
    return result


def source_blocks(sid, scenes, cfg, root, scores, excluded, windows):
    from mspc.data.scene_windows import story_blocks

    out = root / "sources" / sid
    judged = scored_scenes(sid, scenes, read(out / "shots.jsonl"), scores, cfg, excluded)
    built, audit = story_blocks(judged, sid=sid, min_window_seconds=cfg["min_window_seconds"],
                                thresholds=cfg["thresholds"], **cfg["blocks"])
    # 窗只作整块相等对账，不参与分段、切块或逐镜身份。
    signature = lambda row: (row["scene"]["scene_id"], tuple((s["source_shot_index"], s["start_frame"], s["end_frame"])
                                                          for s in row["scene"]["shots"]))
    same_windows = {signature(row): row for row in windows}
    media = out / "block-proxy" / "from-shots"
    media.mkdir(parents=True, exist_ok=True)
    for block in built:
        same = same_windows.get(signature(block))
        if same is not None:
            block.update(same_as_window=same["key"], proxy=same["proxy"])
        else:
            parts = [scores[shot["score_key"]]["proxy"] for shot in block["scene"]["shots"]]
            block["proxy"] = concatenate_proxy(parts, media / f"{block['key']}.mp4", cfg["encode_ffmpeg"])
    write(out / "blocks.jsonl", built)
    write(out / "block-rejections.jsonl", audit["rejected"])
    return {"sid": sid, "source": scenes[0]["row"]["source"], "runs": audit["runs"], "blocks": len(built),
            "eligible": sum(s["eligible"] for row in judged for s in row["shots"]),
            "rejected": len(audit["rejected"]),
            "rejection_reasons": dict(Counter("+".join(r["reasons"]) for r in audit["rejected"])),
            "shots": sum(b["scene"]["shot_count"] for b in built),
            "by_shots": dict(Counter(b["scene"]["shot_count"] for b in built)),
            "same_as_window": sum(b["same_as_window"] is not None for b in built),
            "durations": [sum(s["seconds"] for s in b["scene"]["shots"]) for b in built],
            "proxy_durations": [b["proxy"]["shot_timestamps"][-1]["end"] for b in built],
            "proxy_max_frames": max((len(b["proxy"]["proxy_to_source_frames"]) for b in built), default=0)}


def blocks(cfg, root, workers, limit):
    """第 5 步装块：复用已判镜与打分，剔评测后按源片并行；全成功才发布合并清单。"""
    from statistics import median
    from mspc.data.scene_windows import story_blocks

    story_blocks([], sid="", min_window_seconds=cfg["min_window_seconds"], thresholds=cfg["thresholds"], **cfg["blocks"])  # 在读取数据、写产物前检查必填配置
    if not 1 <= workers <= 32 or limit < 0:
        raise ValueError("blocks needs 1 <= workers <= 32 and limit >= 0")
    started = time.monotonic()
    status = {"shard": [0, 1], "state": "running", "phase": "loading", "seconds": 0,
              "sources_done": 0, "sources_total": 0, "failed": 0, "blocks": 0}
    status_path = root / "status" / "blocks-s0of1.json"
    dump(status_path, status)
    from eval_dedup import evaluation, shot_exclusions

    groups, windows_by_sid = defaultdict(list), defaultdict(list)
    for scene in read(root / "scenes.jsonl"):
        if (root / "sources" / scene["sid"] / "done.json").exists():
            groups[scene["sid"]].append(scene)
    all_windows = read(root / "final/windows.jsonl")
    excluded_windows = read(root / "final/eval-exclude.jsonl")
    refs = evaluation(cfg)
    if refs != read(root / "eval-dedup/inputs/evaluation.jsonl"):
        raise ValueError("evaluation differs from the window exclusion snapshot")
    exclusions = shot_exclusions(all_windows, excluded_windows, read(root / "final/eligible_shots.jsonl"), refs)
    write(root / "final/eval-exclude-shots.jsonl", exclusions)
    excluded = {(r["sid"], r["scene_id"], r["source_shot_index"]) for r in exclusions}
    for row in all_windows:
        windows_by_sid[row["key"].rsplit("-w", 1)[0]].append(row)
    if limit:
        groups = {sid: groups[sid] for sid in sorted(groups)[:limit]}
    scores = {}
    written = time.monotonic()
    status["sources_total"] = len(groups)
    for path in sorted((root / "scores").glob("s*of*.jsonl")):
        with path.open() as handle:
            for row in map(json.loads, handle):
                if row["status"] == "ok" and row["sid"] in groups:
                    scores[row["shot_key"]] = {"dover": {"technical_logit": row["dover"]["technical_logit"]},
                                               "laion": {"mean": row["laion"]["mean"]}, "proxy": row["proxy"]}
                if time.monotonic() - written >= 10:
                    written = time.monotonic()
                    status["seconds"] = written - started
                    dump(status_path, status)
    totals, failures, durations, proxy_durations = defaultdict(Counter), [], [], []
    max_frames = 0
    status["phase"] = "concatenating"
    dump(status_path, status)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(source_blocks, sid, groups[sid], cfg, root, scores, excluded, windows_by_sid[sid]): sid for sid in sorted(groups)}
        for future in as_completed(futures):
            sid = futures[future]
            try:
                done = future.result()
            except Exception as error:
                failures.append({"sid": sid, "error": f"{type(error).__name__}: {error}"[:500]})
            else:
                counts = totals[done["source"]]
                counts.update(sources=1, runs=done["runs"], blocks=done["blocks"], shots=done["shots"],
                              same_as_window=done["same_as_window"], seconds=sum(done["durations"]),
                              eligible=done["eligible"], rejected=done["rejected"])
                counts.update({f"rejected:{k}": v for k, v in done["rejection_reasons"].items()})
                counts.update({f"blocks_{k}shot": v for k, v in done["by_shots"].items()})
                durations.extend(done["durations"])
                proxy_durations.extend(done["proxy_durations"])
                max_frames = max(max_frames, done["proxy_max_frames"])
            status.update(sources_done=status["sources_done"] + 1, failed=len(failures),
                          blocks=sum(c["blocks"] for c in totals.values()), seconds=time.monotonic() - started)
            if time.monotonic() - written >= 10:
                written = time.monotonic()
                dump(status_path, status)
    counts = sum(totals.values(), Counter())
    result = {"config": cfg["blocks"], "limit": limit, "input_windows": len(all_windows),
              "excluded_windows": len(excluded_windows), "excluded_shots": len(exclusions),
              "sources": {k: dict(v) for k, v in totals.items()}, "totals": dict(counts),
              "by_shots": {str(k): counts[f"blocks_{k}shot"] for k in range(1, cfg["blocks"]["max_shots"] + 1)},
              "duration_seconds_median": median(durations) if durations else None,
              "duration_over_60_seconds": sum(d > 60 for d in durations),
              "proxy_seconds": sum(proxy_durations),
              "proxy_duration_seconds_median": median(proxy_durations) if proxy_durations else None,
              "proxy_duration_over_60_seconds": sum(d > 60 for d in proxy_durations),
              "proxy_max_frames": max_frames, "failed": failures, "seconds": time.monotonic() - started}
    if failures:
        status.update(state="failed", phase="concatenating")
        dump(status_path, status)
        dump(root / "final" / "blocks-summary.json", result)
        raise RuntimeError(f"blocks failed for {len(failures)} sources; final manifest was not replaced")
    status["phase"] = "merging"
    dump(status_path, status)
    for name in ("blocks.jsonl", "block-rejections.jsonl"):
        target = root / "final" / name
        temporary = target.with_suffix(".partial")
        with temporary.open("w") as handle:
            for sid in sorted(groups):
                handle.write((root / "sources" / sid / name).read_text())
        temporary.replace(target)
    dump(root / "final" / "blocks-summary.json", result)
    status.update(state="completed", phase="completed", seconds=time.monotonic() - started)
    dump(status_path, status)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "measure", "collect", "windows", "blocks"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--shard", type=int, nargs=2, metavar=("INDEX", "COUNT"))
    p.add_argument("--workers", type=int)
    p.add_argument("--limit", type=int, help="measure、windows、blocks：只跑这么多条源片，0 为全部")
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    out_root = Path(config["root"])
    if a.command == "prepare":
        prepare(config, out_root)
    elif a.command == "collect":
        collect(config, out_root)
    elif a.command == "blocks":
        if None in (a.workers, a.limit):
            p.error("blocks needs --workers and --limit")
        blocks(config, out_root, a.workers, a.limit)
    elif a.command == "windows":
        if None in (a.workers, a.limit):
            p.error("windows needs --workers and --limit")
        windows(config, out_root, a.workers, a.limit)
    else:
        if None in (a.shard, a.workers, a.limit):
            p.error("measure needs --shard, --workers and --limit")
        measure(config, out_root, tuple(a.shard), a.workers, a.limit)
