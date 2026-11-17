"""前置处理第二遍「取」：每条源片再顺序解码一遍，只取要用的帧，同时给合格镜打分并按镜导代理片。

画面质量要筛，所以全部合格镜都打分，定线、装窗在这之后；代理片按镜导出，装窗后直接拼接；
不跑文字检测；只用 CPU 解码。

一个打分进程 = 若干 CPU 解码进程（解码、DOVER／CLIP 的官方预处理、x264 编码）＋ 一张卡上常驻的 DOVER 与 LAION。
同一张卡可以起两个打分进程（不同分片）。
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time
import traceback
from typing import Any, Mapping

SCORE_VERSION = "shot_scores_v1"


def shot_plan(row: Mapping[str, Any], engine, proxy_stride: int) -> dict[str, Any]:
    """一个合格镜要取的帧：DOVER 清晰度视图的官方采样、LAION 的首中末、代理片的 2 fps 取帧。"""
    from mspc.data.frames import proxy_frames, shot_grid
    from mspc.data.quality_scorer.adapters.media import ShotLocator

    start, end = row["segment"]
    locator = ShotLocator(row["scene_id"], row["family"], "scene_window_v1", row["source_shot_index"],
                          row["source_video_path"], start, end, "half_open")
    # 只打清晰度头（美学用 LAION，DOVER 美学头不算）；清晰度的取帧种子按视图各自重置，与是否算美学头无关。
    dover = {"technical": engine.sample_local_indices(locator)["technical"]}
    # SVD §3.1 取首中末；中点向下取整是本实现的口径。
    laion = [0, (locator.frame_count - 1) // 2, locator.frame_count - 1]
    grid = shot_grid(start, end, row["render_frames"])
    picks = proxy_frames([[0, len(grid)]], proxy_stride)[0]
    native = {locator.absolute_index(i) for indices in dover.values() for i in indices}
    native |= {locator.absolute_index(i) for i in laion}
    return {"locator": locator, "dover": dover, "laion": laion, "picks": picks,
            "proxy_source": [grid[i] for i in picks], "native": native}


def decode_worker(cfg: Mapping[str, Any], tasks, out, finished, worker_id: str) -> None:
    """CPU 进程：一条源片顺序解一遍，凑齐一个镜就做预处理、编码代理片并交给 GPU 进程。"""
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from mspc.benchmark.long_story_holdout import crop_box
    from mspc.data.frames import center_crop, export_shot_span, iter_scene_frames
    from mspc.data.quality_scorer.adapters.dover import PreparedDoverShot
    from mspc.data.quality_scorer.scorers import dover_backend, laion_import

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    engine = dover_backend(cfg, "cpu")
    engine.load_preprocessor()
    preprocess = laion_import(cfg).clip._transform(224)
    root = Path(cfg["root"])
    while True:
        group = tasks.get()
        if group is None:
            break
        first, sent = group[0], set()
        folder = root / "sources" / first["sid"]
        (folder / "proxy").mkdir(parents=True, exist_ok=True)
        try:
            box = tuple(crop_box(first))
            plans, native_users, proxy_users = {}, defaultdict(list), defaultdict(list)
            for row in group:
                plan = shot_plan(row, engine, cfg["proxy_stride"])
                plan.update(row=row, native_frames={}, proxy_frames={}, last=max(plan["native"] | set(plan["proxy_source"])))
                plans[row["shot_key"]] = plan
                for frame in plan["native"]:
                    native_users[frame].append(row["shot_key"])
                for frame in set(plan["proxy_source"]):
                    proxy_users[frame].append(row["shot_key"])
            wanted = sorted(set(native_users) | set(proxy_users))
            scene = {"source_video_path": first["source_video_path"], "width": first["width"], "height": first["height"],
                     "video_frame_count": wanted[-1] + 1}
            for position, frame in iter_scene_frames(scene, wanted, ffmpeg=cfg["ffmpeg"],
                                                     error_path=folder / f"score-decode-{worker_id}.stderr",
                                                     video_decode_num_threads=1):
                if position in proxy_users:
                    small = center_crop(Image.fromarray(frame).crop(box))
                    for key in proxy_users[position]:
                        plans[key]["proxy_frames"][position] = small
                for key in native_users.get(position, ()):
                    plans[key]["native_frames"][position] = frame
                for key in {*native_users.get(position, ()), *proxy_users.get(position, ())}:
                    plan = plans[key]
                    if position != plan["last"]:
                        continue
                    row, locator = plan["row"], plan["locator"]
                    arrays = {name: np.stack([plan["native_frames"][locator.absolute_index(i)] for i in indices])
                              for name, indices in plan["dover"].items()}
                    views = engine.prepare_views(PreparedDoverShot(locator, plan["dover"], arrays, first["width"], first["height"]),
                                                 names=("technical",))
                    laion = [preprocess(Image.fromarray(plan["native_frames"][locator.absolute_index(i)])) for i in plan["laion"]]
                    source_of = dict(zip(plan["picks"], plan["proxy_source"]))
                    proxy = export_shot_span(wid=key, selected=[plan["picks"]], videos=folder / "proxy", ffmpeg=Path(cfg["encode_ffmpeg"]),
                                             frame=lambda target: plan["proxy_frames"][source_of[target]])
                    for tensor in [*views.values(), *laion]:
                        tensor.share_memory_()
                    out.put({"type": "shot", "row": row, "views": views, "laion": laion, "dover_local": plan["dover"],
                             "laion_frames": [locator.absolute_index(i) for i in plan["laion"]],
                             "proxy": {"video": proxy["video"], "frames": len(plan["picks"]), "source_frames": plan["proxy_source"],
                                       "crop_box": list(box)}})
                    sent.add(key)
                    del plans[key]
            if plans:
                raise ValueError("decoder ended before every shot was complete")
        except Exception as error:
            for row in group:
                if row["shot_key"] not in sent:
                    out.put({"type": "failed", "row": row, "error": f"{type(error).__name__}: {error}"[:1000],
                             "traceback": traceback.format_exc()[-2000:]})
        out.put({"type": "source_done", "sid": first["sid"]})
    out.put({"type": "worker_done"})
    finished.wait()  # 共享内存张量由生产者持有，消费者收完才能退出


def score_items(cfg: Mapping[str, Any], engine, laion, items: list[dict]) -> list[dict]:
    """GPU：一批镜的 DOVER 清晰度头（原始 logit）与 LAION 美学分。"""
    from mspc.data.quality_scorer.adapters.dover import PreparedDoverShot
    from mspc.data.quality_scorer.adapters.media import ShotLocator

    prepared = []
    for item in items:
        row = item["row"]
        locator = ShotLocator(row["scene_id"], row["family"], "scene_window_v1", row["source_shot_index"],
                              row["source_video_path"], *row["segment"], "half_open")
        prepared.append(PreparedDoverShot(locator, item["dover_local"], {}, row["width"], row["height"]))
    dover = engine.score_view_batch(prepared, [item["views"] for item in items], names=("technical",))
    frames = [tensor for item in items for tensor in item["laion"]]
    aesthetic = []
    for offset in range(0, len(frames), cfg["laion_batch_frames"]):
        aesthetic.extend(laion.score(frames[offset:offset + cfg["laion_batch_frames"]]))
    results = []
    for number, (item, payload) in enumerate(zip(items, dover, strict=True)):
        row, scores = item["row"], aesthetic[number * 3:number * 3 + 3]
        results.append({"shot_key": row["shot_key"], "sid": row["sid"], "family": row["family"], "scene_id": row["scene_id"],
                        "source_shot_index": row["source_shot_index"], "source_video_path": row["source_video_path"],
                        "segment": row["segment"], "render_frames": row["render_frames"], "status": "ok",
                        "dover": payload, "laion": {"frames": item["laion_frames"], "scores": scores, "mean": sum(scores) / 3},
                        "proxy": item["proxy"], "version": SCORE_VERSION})
    return results


def run_shard(cfg: Mapping[str, Any], shard: tuple[int, int], limit: int) -> dict[str, Any]:
    """一个打分进程：本分片的源片谁先量完就先打谁（读它的 shots.jsonl），跳过已有结果的镜，结果写进本分片的文件。"""
    import multiprocessing as mp
    import queue
    import threading
    import zlib

    import cv2
    import torch
    from mspc.data.quality_scorer.scorers import LaionBackend, dover_backend

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    root = Path(cfg["root"])
    index, count = shard
    sids = []
    with (root / "scenes.jsonl").open() as handle:
        for line in handle:
            scene = json.loads(line)
            if zlib.crc32(scene["row"]["source_video_path"].encode()) % count == index and scene["sid"] not in sids[-1:]:
                sids.append(scene["sid"])
    sids = sorted(set(sids))
    if limit:
        sids = sids[::max(1, len(sids) // limit)][:limit]
    (root / "scores").mkdir(exist_ok=True)
    (root / "status").mkdir(exist_ok=True)
    tag = f"s{index}of{count}"
    result_path = root / "scores" / f"{tag}.jsonl"
    done = set()
    if result_path.exists():
        data = result_path.read_bytes()
        if data and not data.endswith(b"\n"):  # 上次在写一行的中途被停掉：截掉这半行
            result_path.write_bytes(data[:data.rfind(b"\n") + 1])
        with result_path.open() as handle:
            done = {json.loads(line)["shot_key"] for line in handle if line.strip()}
    context = mp.get_context("spawn")
    tasks, out, finished = context.Queue(), context.Queue(maxsize=cfg["prefetch_shots"]), context.Event()
    fed = defaultdict(int)

    def feed() -> None:
        """流水：第一遍每量完一条源片（出现 done.json），就把它的合格镜交给解码进程；失败的源片跳过。"""
        waiting = list(sids)
        while waiting:
            later = []
            for sid in waiting:
                folder = root / "sources" / sid
                if (folder / "done.json").exists():
                    with (folder / "shots.jsonl").open() as handle:
                        rows = [row for row in map(json.loads, handle) if row["eligible"]]
                    rows.sort(key=lambda r: r["segment"][0])
                    for number, row in enumerate(rows):
                        row["shot_key"] = f"{sid}-e{number:04d}"
                    todo = [row for row in rows if row["shot_key"] not in done]
                    fed["sources"] += 1
                    fed["shots"] += len(rows)
                    fed["before"] += len(rows) - len(todo)
                    if todo:
                        tasks.put(todo)
                elif (folder / "error.json").exists():
                    fed["sources"] += 1
                else:
                    later.append(sid)
            waiting = later
            if waiting:
                time.sleep(30)
        for _ in range(cfg["decode_workers"]):
            tasks.put(None)

    threading.Thread(target=feed, daemon=True).start()
    workers = [context.Process(target=decode_worker, args=(cfg, tasks, out, finished, f"{tag}-{n}"))
               for n in range(cfg["decode_workers"])]
    for worker in workers:
        worker.start()
    engine = dover_backend(cfg, "cuda:0")
    engine.load()
    laion = LaionBackend(cfg, "cuda:0")
    started, counts, batch, alive, waited, inferred = time.monotonic(), defaultdict(int), [], len(workers), 0.0, 0.0

    def status(state: str) -> dict[str, Any]:
        seconds = time.monotonic() - started
        value = {"shard": list(shard), "state": state, "shots_fed": fed["shots"], "shots_ok": counts["ok"], "shots_failed": counts["failed"],
                 "shots_done": fed["before"] + counts["ok"] + counts["failed"],
                 "sources_fed": fed["sources"], "sources_total": len(sids), "seconds": seconds,
                 "shots_per_second": counts["ok"] / seconds if seconds else 0.0,
                 "wait_decode_share": waited / seconds if seconds else 0.0, "inference_share": inferred / seconds if seconds else 0.0}
        temporary = root / "status" / f"score-{tag}.partial"
        temporary.write_text(json.dumps(value))
        temporary.replace(root / "status" / f"score-{tag}.json")
        return value

    try:
        with result_path.open("a") as handle:
            while alive or batch:
                item = None
                if alive:
                    at = time.monotonic()
                    try:
                        item = out.get(timeout=0.2 if batch else 5)
                    except queue.Empty:
                        pass
                    waited += time.monotonic() - at
                if item is not None:
                    if item["type"] == "shot":
                        batch.append(item)
                    elif item["type"] == "failed":
                        row = item["row"]
                        handle.write(json.dumps({"shot_key": row["shot_key"], "sid": row["sid"], "family": row["family"],
                                                 "scene_id": row["scene_id"], "source_shot_index": row["source_shot_index"],
                                                 "status": "failed", "error": item["error"], "version": SCORE_VERSION},
                                                ensure_ascii=False) + "\n")
                        counts["failed"] += 1
                    elif item["type"] == "source_done":
                        counts["sources"] += 1
                    else:
                        alive -= 1
                if batch and (len(batch) >= cfg["batch_shots"] or item is None or not alive):
                    at = time.monotonic()
                    for result in score_items(cfg, engine, laion, batch):
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    torch.cuda.synchronize()
                    inferred += time.monotonic() - at
                    handle.flush()
                    counts["ok"] += len(batch)
                    batch = []
                    status("running")
    finally:
        finished.set()
        for worker in workers:
            worker.join(timeout=30)
            if worker.is_alive():
                worker.terminate()
        # 出错退出时队列里还有没取走的镜：不等队列的写线程，否则进程挂着不退（10-12 三个爆显存的进程就这样挂住）
        tasks.cancel_join_thread()
        out.cancel_join_thread()
    return status("completed")
