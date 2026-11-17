#!/usr/bin/env python3
"""镜界类型与逐镜景别／机位标注器的入口：抽样、联系图、标注、校准四步（领域代码归 ``mspc.data.cut_type_labels``）。

``sample``（主服务器，CPU）：从训练池按来源分层抽相邻镜界，抽帧落盘，写 ``samples.jsonl``。池 = 潜变量缓存的
205,200 个故事去掉 ``excluded_case_ids_with_screen_holdout`` 与质量权重 0 的，再要求源视频在盘上
（主服务器上 CI-VID 约 27%、Cine250K 约 45% 的行有源视频）。每个来源一半随机（每个镜界等概率），一半按
廉价的缩略图相似度补足稀有类（回机位、续拍候选、中等相似）。

``sheets``：把每个样本的帧拼成联系图（人工看帧打参考标签用）。

``label``：经 CloseAI 看图模型标注，并发、追加写、按 sample_id 续跑；总花费按 USD 硬上限熔断。

``calibrate``：和人工参考标签比，报每类准确率、混淆矩阵、景别／角度准确率，以及与 CI-VID 原生字段的一致率。

用法（主服务器，CPU；CloseAI 直连，不 source 代理）::

    export MSPC_SHARED_ROOT=${MSPC_SHARED_ROOT}
    PY=$MSPC_SHARED_ROOT/env/renderer_public_metrics/bin/python
    PYTHONPATH=src $PY tools/annotate/label_cut_types.py sample \\
        --config configs/annotation/cut_type_labeler_v1.json --out-dir <run dir>
    PYTHONPATH=src $PY tools/annotate/label_cut_types.py label \\
        --config configs/annotation/cut_type_labeler_v1.json --run-dir <run dir> \\
        --api-key-file $MSPC_SHARED_ROOT/OoVMetric/resource/closeai_api_key.txt --model qwen3.5-flash
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mspc.data import cut_type_labels as ctl  # noqa: E402
from mspc.data.closeai_client import (  # noqa: E402
    CloseAIBudgetExceeded,
    CloseAIClient,
    CloseAICostCounter,
    CloseAIError,
    load_closeai_api_key,
)

CONFIG_KEYS = {
    "schema_version", "closeai_base_url", "model", "temperature", "max_tokens", "enable_thinking",
    "max_retries", "retry_base_seconds", "request_timeout_seconds", "rpm_limit", "budget_hard_cap_usd",
    "cny_per_usd_budget_guard", "precall_prompt_tokens", "precall_completion_tokens", "price_source",
    "prices_usd_per_mtok", "frames", "pool", "sampling",
}
CONFIG_SCHEMA = "cut_type_labeler_config_v1"
SOURCES = ("ci_vid", "cine250k")
USD_QUANT = Decimal("0.00000001")

# 参考标签 TSV 的简码。
CUT_CODES = {"CO": "continuation", "NC": "same_scene_new_camera", "RT": "return_to_setup", "CC": "same_scene_cast_change", "SC": "scene_change"}
SIZE_CODES = dict(zip(("XCU", "CU", "MCU", "MS", "MW", "WS", "XWS"), ctl.SHOT_SIZES))
ANGLE_CODES = dict(zip(("EL", "LOW", "HIGH", "OVH", "AER", "DUTCH", "OTS"), ctl.CAMERA_ANGLES))


def load_config(path: Path) -> dict[str, Any]:
    from mspc.config.shared_root import expand_shared_root_token

    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != CONFIG_KEYS:
        raise ValueError(f"{path}: keys must be {sorted(CONFIG_KEYS)}; missing={sorted(CONFIG_KEYS - set(config))} extra={sorted(set(config) - CONFIG_KEYS)}")
    if config["schema_version"] != CONFIG_SCHEMA:
        raise ValueError(f"{path}: schema_version must be {CONFIG_SCHEMA!r}")
    config["pool"] = {key: expand_shared_root_token(value, f"{path}:pool.{key}") for key, value in config["pool"].items()}
    return config


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------


def _thumb(image: Any) -> Any:
    import numpy as np

    return np.asarray(image.convert("L").resize((32, 18)), dtype="float32") / 255.0


def _similarity(x: Any, y: Any) -> float:
    return float(1.0 - abs(x - y).mean())


def _pair_sims(thumbs: dict[str, list[Any]]) -> dict[str, float | None]:
    """缩略图相似度（0..1，越大越像）：A 尾 → B 头，A 与 B 任意两帧的最大值，Z 与 B 任意两帧的最大值。"""

    a, b = thumbs["a"], thumbs["b"]
    sims: dict[str, float | None] = {
        "end_start": _similarity(a[-1], b[0]),
        "max_ab": max(_similarity(x, y) for x in a for y in b),
        "max_zb": None,
    }
    if "z" in thumbs:
        sims["max_zb"] = max(_similarity(x, y) for x in thumbs["z"] for y in b)
    return sims


def _extract_candidate(task: dict[str, Any]) -> dict[str, Any]:
    """一个候选镜界：解码 Z（可无）、A、B 各三帧，缩放存 JPEG，算相似度。子进程里跑。"""

    from mspc.data.msm_global_frames import decode_mapped_frames, default_msm_frame_reader, map_concat_indices_to_sources, parse_msm_shot_groups

    row, j = task["row"], task["j"]
    shot_groups = parse_msm_shot_groups(row["shot_groups"])
    paths = ast.literal_eval(row["source_video_paths"])
    segments = ast.literal_eval(row["source_shot_segments"])
    by_shot = ast.literal_eval(row["source_frame_indices_by_shot"])
    shots = {"a": j - 1, "b": j}
    if j >= 2:
        shots = {"z": j - 2, **shots}
    concat: list[int] = []
    for shot in shots.values():
        concat += ctl.shot_frame_positions(*shot_groups[shot])
    mapped = map_concat_indices_to_sources(
        concat, shot_groups=shot_groups, source_video_paths=paths, source_shot_segments=segments, source_frame_indices_by_shot=by_shot
    )
    images = decode_mapped_frames(mapped, frame_reader=default_msm_frame_reader)
    frame_dir = Path(task["frames_root"]) / task["key"]
    frame_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"key": task["key"], "source": row["dataset"], "case_id": row["case_id"], "boundary": j, "shot_count": len(shot_groups), "frames": {}, "video": {}, "native_frame_indices": {}}
    thumbs: dict[str, list[Any]] = {}
    cursor = 0
    for name, shot in shots.items():
        chunk = images[cursor : cursor + ctl.FRAMES_PER_SHOT]
        native = mapped[cursor : cursor + ctl.FRAMES_PER_SHOT]
        cursor += ctl.FRAMES_PER_SHOT
        rels, thumbs[name] = [], []
        for index, image in enumerate(chunk):
            image = image.convert("RGB")
            image.thumbnail((task["max_side"], task["max_side"]))
            relative = f"frames/{task['key']}/{name}{index}.jpg"
            image.save(Path(task["out_dir"]) / relative, format="JPEG", quality=task["jpeg_quality"])
            rels.append(relative)
            thumbs[name].append(_thumb(image))
        record["frames"][name] = rels
        record["video"][name] = native[0][0]
        record["native_frame_indices"][name] = [index for _, index in native]
    record["sims"] = _pair_sims(thumbs)
    return record


def load_pool(config: dict[str, Any]) -> Any:
    """池：元数据行去掉排除名单与质量权重 0。返回 DataFrame。"""

    import pandas as pd

    pool = config["pool"]
    excluded = {json.loads(line)["case_id"] for line in Path(pool["excluded_case_ids"]).read_text(encoding="utf-8").splitlines() if line.strip()}
    positive: set[str] = set()
    with Path(pool["quality_index"]).open(encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            if entry["sample_weight"] > 0:
                positive.add(entry["case_id"])
    columns = ["case_id", "dataset", "shot_count", "shot_groups", "source_video_paths", "source_shot_segments", "source_frame_indices_by_shot"]
    frame = pd.read_csv(pool["metadata_csv"], usecols=columns, low_memory=False)
    keep = ~frame["case_id"].isin(excluded) & frame["case_id"].isin(positive) & (frame["shot_count"] >= 2)
    return frame[keep].reset_index(drop=True)


def draw_boundaries(pool: Any, source: str, n_draw: int, seed: int) -> list[tuple[int, int]]:
    """按镜界等概率（一个故事的权重 = 镜数 - 1）有放回抽，去掉重复故事；返回 (池行号, 后一镜序号 j)。"""

    import numpy as np

    rows = pool.index[pool["dataset"] == source].to_numpy()
    counts = pool.loc[rows, "shot_count"].to_numpy()
    weights = (counts - 1).astype("float64")
    rng = np.random.default_rng([seed, SOURCES.index(source)])
    picked = rng.choice(len(rows), size=n_draw, replace=True, p=weights / weights.sum())
    boundaries = rng.integers(1, counts[picked])
    seen: set[int] = set()
    drawn = []
    for position, j in zip(picked, boundaries):
        if int(position) in seen:
            continue
        seen.add(int(position))
        drawn.append((int(rows[position]), int(j)))
    return drawn


def select_boosted(
    candidates: Sequence[dict[str, Any]], *, n_return: int, n_continuation: int, total: int
) -> list[dict[str, Any]]:
    """从候选里补稀有类：回机位候选（有 Z，Z 像 B 且 B 不像 A）、续拍候选（A 尾像 B 头），余下取
    A 与 B 中等相似（缩略图相似度最接近 0.7）的——它们多是同景换机位／换人的边界情形。每个故事至多选一个。
    """

    chosen: list[dict[str, Any]] = []
    used_cases: set[str] = set()

    def take(pool: Sequence[dict[str, Any]], limit: int) -> None:
        taken = 0
        for candidate in pool:
            if taken >= limit:
                return
            if candidate["case_id"] in used_cases:
                continue
            chosen.append(candidate)
            used_cases.add(candidate["case_id"])
            taken += 1

    with_earlier = [c for c in candidates if c["sims"]["max_zb"] is not None]
    take(sorted(with_earlier, key=lambda c: c["sims"]["max_zb"] - c["sims"]["max_ab"], reverse=True), n_return)
    take(sorted(candidates, key=lambda c: c["sims"]["end_start"], reverse=True), n_continuation)
    take(sorted(candidates, key=lambda c: abs(c["sims"]["max_ab"] - 0.7)), total - len(chosen))
    return chosen


def find_civid_records(keys: set[str], paths: Sequence[str]) -> dict[str, dict[str, Any]]:
    """在 CI-VID 元数据 JSONL 里按 ``video_path`` 找记录；只对命中的行做 ``json.loads``。"""

    pattern = re.compile(r'^\{"video_path": "([^"]+)"')
    found: dict[str, dict[str, Any]] = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                match = pattern.match(line)
                if match and match.group(1) in keys and match.group(1) not in found:
                    found[match.group(1)] = json.loads(line)
    return found


def civid_key(video_path: str) -> tuple[str, int]:
    """``.../ci_vid/videos/chunk_0/<hash>/17/1.mp4`` -> (``chunk_0/<hash>/17``, 1)。"""

    path = Path(video_path)
    relative = path.parent.as_posix().split("/videos/", 1)[1]
    return relative, int(path.stem)


def native_for_sample(sample: dict[str, Any], records: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    key_a, clip_a = civid_key(sample["video"]["a"])
    key_b, clip_b = civid_key(sample["video"]["b"])
    record = records.get(key_a)
    if record is None or key_a != key_b or clip_b != clip_a + 1:
        return None
    individual, joint = record["individual_captions"], record["joint_captions"]
    if clip_b >= len(individual) or clip_a >= len(joint):
        return None
    features = ctl.native_features([individual[clip_a], individual[clip_b]], joint[clip_a])
    pair = ctl.parse_native_json(joint[clip_a])
    return {
        **features,
        "text": {
            "change_in_camera_angle": pair.get("change_in_camera_angle"),
            "change_in_video_background": pair.get("change_in_video_background"),
            "change_in_camera_movement": pair.get("change_in_camera_movement"),
            "camera_angle_a": ctl.parse_native_json(individual[clip_a]).get("camera_angle"),
            "camera_angle_b": ctl.parse_native_json(individual[clip_b]).get("camera_angle"),
        },
    }


def cmd_sample(args: argparse.Namespace) -> None:
    import numpy as np

    config = load_config(args.config)
    plan = config["sampling"]
    out_dir = args.out_dir
    if (out_dir / "samples.jsonl").exists():
        raise FileExistsError(f"{out_dir / 'samples.jsonl'} exists; refusing to overwrite a frozen sample")
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    pool = load_pool(config)
    print(f"pool rows after exclusion and quality: {len(pool)} ({time.monotonic() - started:.0f}s)", flush=True)

    tasks: list[dict[str, Any]] = []
    stratum_of: dict[str, str] = {}
    stats: dict[str, Any] = {"pool_rows": len(pool)}
    for source in SOURCES:
        drawn = draw_boundaries(pool, source, 8000, plan["seed"])
        need = plan["random_per_source"] + plan["candidates_per_source"]
        checked = with_media = 0
        source_tasks = []
        for row_index, j in drawn:
            row = pool.loc[row_index].to_dict()
            checked += 1
            if not all(Path(p).is_file() for p in set(ast.literal_eval(row["source_video_paths"]))):
                continue
            with_media += 1
            key = f"{source}-{len(source_tasks):04d}"
            source_tasks.append({"row": row, "j": j, "key": key, "frames_root": str(out_dir / "frames"), "out_dir": str(out_dir), "max_side": config["frames"]["max_side"], "jpeg_quality": config["frames"]["jpeg_quality"]})
            stratum_of[key] = "random" if len(source_tasks) <= plan["random_per_source"] else "candidate"
            if len(source_tasks) >= need:
                break
        stats[source] = {"boundaries_checked": checked, "with_media": with_media, "tasks": len(source_tasks)}
        tasks += source_tasks
    print(f"media checked, extracting {len(tasks)} candidates ({time.monotonic() - started:.0f}s)", flush=True)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ProcessPoolExecutor(plan["workers"]) as pool_executor:
        futures = {pool_executor.submit(_extract_candidate, task): task for task in tasks}
        for future, task in futures.items():
            try:
                records.append(future.result())
            except Exception as error:  # 单个候选解码失败只记账，不影响其余
                failures.append({"key": task["key"], "case_id": task["row"]["case_id"], "error": f"{type(error).__name__}: {error}"})
    stats["extraction_failures"] = failures
    print(f"extracted {len(records)} / {len(tasks)} ({time.monotonic() - started:.0f}s)", flush=True)

    selected: list[dict[str, Any]] = []
    for source in SOURCES:
        mine = sorted((r for r in records if r["source"] == source), key=lambda r: r["key"])
        random_rows = [r for r in mine if stratum_of[r["key"]] == "random"][: plan["random_per_source"]]
        for record in random_rows:
            record["stratum"] = "random"
        rest = [r for r in mine if stratum_of[r["key"]] == "candidate" and r["case_id"] not in {x["case_id"] for x in random_rows}]
        boosted = select_boosted(rest, n_return=plan["boosted_return_per_source"], n_continuation=plan["boosted_continuation_per_source"], total=plan["per_source"] - len(random_rows))
        for record in boosted:
            record["stratum"] = "boosted"
        selected += random_rows + boosted
    order = np.random.default_rng([plan["seed"], 99]).permutation(len(selected))
    selected = [selected[i] for i in order]
    for number, record in enumerate(selected, start=1):
        record["sample_id"] = f"s{number:03d}"
        record["split"] = "dev" if number <= plan["dev_size"] else "test"
        record["has_earlier"] = "z" in record["frames"]

    civid = [r for r in selected if r["source"] == "ci_vid"]
    keys = {civid_key(r["video"]["a"])[0] for r in civid}
    found = find_civid_records(keys, [config["pool"]["ci_vid_train_samples"], config["pool"]["ci_vid_test_samples"]])
    for record in selected:
        record["native"] = native_for_sample(record, found) if record["source"] == "ci_vid" else None
    stats["selected"] = Counter(f"{r['source']}/{r['stratum']}" for r in selected)
    stats["selected_with_earlier"] = sum(r["has_earlier"] for r in selected)
    stats["ci_vid_with_native"] = sum(1 for r in civid if r["native"] is not None)
    (out_dir / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(selected, key=lambda r: r["sample_id"])), encoding="utf-8")
    (out_dir / "sample_summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1, default=dict), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, default=dict), flush=True)


# ---------------------------------------------------------------------------
# sheets
# ---------------------------------------------------------------------------


def _load_samples(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cmd_sheets(args: argparse.Namespace) -> None:
    from PIL import Image, ImageDraw, ImageFont

    samples = _load_samples(args.run_dir / "samples.jsonl")
    if args.only_split:
        samples = [s for s in samples if s["split"] == args.only_split]
    cell_w, cell_h, pad, font = args.cell_width, args.cell_width * 9 // 16, 6, ImageFont.load_default(size=18)
    row_names = (("z", "Z (before A)"), ("a", "A"), ("b", "B"))

    def block(sample: dict[str, Any]) -> Any:
        rows = [(name, label) for name, label in row_names if name in sample["frames"]]
        image = Image.new("RGB", (ctl.FRAMES_PER_SHOT * (cell_w + pad) + 56, 26 + len(rows) * (cell_h + pad)), "white")
        draw = ImageDraw.Draw(image)
        draw.text((4, 2), f"{sample['sample_id']}  {'no Z' if not sample['has_earlier'] else 'with Z'}", fill="black", font=font)
        for r, (name, label) in enumerate(rows):
            top = 26 + r * (cell_h + pad)
            draw.text((4, top + cell_h // 2 - 9), name.upper(), fill=(200, 0, 0), font=font)
            for c, relative in enumerate(sample["frames"][name]):
                with Image.open(args.run_dir / relative) as frame:
                    frame = frame.convert("RGB")
                    frame.thumbnail((cell_w, cell_h))
                    image.paste(frame, (56 + c * (cell_w + pad), top))
        return image

    out = args.run_dir / "sheets"
    out.mkdir(exist_ok=True)
    for start in range(0, len(samples), args.per_sheet):
        blocks = [block(s) for s in samples[start : start + args.per_sheet]]
        columns = 2
        width = max(b.width for b in blocks)
        heights = [max(b.height for b in blocks[i : i + columns]) for i in range(0, len(blocks), columns)]
        sheet = Image.new("RGB", (columns * (width + 12), sum(heights) + 12 * len(heights)), (235, 235, 235))
        y = 0
        for row_index, i in enumerate(range(0, len(blocks), columns)):
            for column, b in enumerate(blocks[i : i + columns]):
                sheet.paste(b, (column * (width + 12), y))
            y += heights[row_index] + 12
        name = f"{samples[start]['sample_id']}-{samples[min(start + args.per_sheet, len(samples)) - 1]['sample_id']}.jpg"
        sheet.save(out / name, format="JPEG", quality=90)
    print(f"wrote sheets to {out}")


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------


def _usd(price: dict[str, Decimal], usage: dict[str, int]) -> Decimal:
    cost = (Decimal(usage["prompt_tokens"]) * price["input"] + Decimal(usage["completion_tokens"]) * price["output"]) / Decimal(1_000_000)
    return cost.quantize(USD_QUANT)


def build_client(config: dict[str, Any], *, model: str, api_key: str, already_spent_usd: Decimal) -> tuple[CloseAIClient, dict[str, Decimal]]:
    prices = config["prices_usd_per_mtok"]
    if model not in prices:
        raise ValueError(f"no USD price for model {model!r}; priced: {sorted(prices)}")
    price = {"input": Decimal(prices[model]["input_usd_per_mtok"]), "output": Decimal(prices[model]["output_usd_per_mtok"])}
    rate = Decimal(config["cny_per_usd_budget_guard"])
    counter = CloseAICostCounter(
        already_spent_cny=already_spent_usd * rate,
        hard_cap_cny=Decimal(config["budget_hard_cap_usd"]) * rate,
        prices={model: {"input_cny_per_mtok": price["input"] * rate, "output_cny_per_mtok": price["output"] * rate}},
        precall_prompt_tokens=config["precall_prompt_tokens"],
        precall_completion_tokens=config["precall_completion_tokens"],
    )
    client = CloseAIClient(
        api_key=api_key,
        base_url=config["closeai_base_url"],
        cost_counter=counter,
        timeout_seconds=config["request_timeout_seconds"],
        max_retries=config["max_retries"],
        retry_base_seconds=config["retry_base_seconds"],
        rpm_limit=config["rpm_limit"],
    )
    return client, price


def _data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def label_one(client: CloseAIClient, config: dict[str, Any], price: dict[str, Decimal], model: str, run_dir: Path, sample: dict[str, Any]) -> dict[str, Any]:
    urls = {name: [_data_url(run_dir / relative) for relative in relatives] for name, relatives in sample["frames"].items()}
    messages = ctl.labeler_messages(urls["a"], urls["b"], urls.get("z"))
    row: dict[str, Any] = {"sample_id": sample["sample_id"], "model": model, "schema_version": ctl.SCHEMA_VERSION, "prompt_id": ctl.PROMPT_ID, "prompt_tokens": None, "prompt_image_tokens": None, "completion_tokens": None, "cost_usd": None, "request_id": None, "raw_text": None, "parsed": None, "error": None}
    started = time.monotonic()
    try:
        response = client.complete_messages(model=model, messages=messages, temperature=config["temperature"], response_format={"type": "json_object"}, max_tokens=config["max_tokens"], enable_thinking=config["enable_thinking"])
        row.update(prompt_tokens=response.usage["prompt_tokens"], prompt_image_tokens=response.prompt_image_tokens, completion_tokens=response.usage["completion_tokens"], request_id=response.request_id, raw_text=response.text, cost_usd=str(_usd(price, response.usage)))
        row["parsed"] = ctl.parse_labeler_output(response.text, has_earlier=sample["has_earlier"])
    except CloseAIBudgetExceeded:
        raise
    except (ctl.LabelerOutputError, CloseAIError) as error:
        row["error"] = {"type": type(error).__name__, "message": str(error)}
    row["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return row


def spent_usd(run_dir: Path) -> Decimal:
    """run 目录下所有 ``labels.*.jsonl`` 已记的花费之和（预算是整个校准的总上限，不是单次运行的）。"""

    total = Decimal(0)
    for path in run_dir.glob("labels.*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            cost = json.loads(line)["cost_usd"]
            if cost is not None:
                total += Decimal(cost)
    return total


def cmd_label(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model = args.model or config["model"]
    run_dir = args.run_dir
    output = run_dir / f"labels.{args.tag or model}.jsonl"
    samples = _load_samples(run_dir / "samples.jsonl")
    if args.split:
        samples = [s for s in samples if s["split"] == args.split]
    if args.sample_ids:
        wanted = set(args.sample_ids.split(","))
        samples = [s for s in samples if s["sample_id"] in wanted]
    if args.limit:
        samples = samples[: args.limit]
    done = {json.loads(line)["sample_id"] for line in output.read_text(encoding="utf-8").splitlines()} if output.exists() else set()
    pending = [s for s in samples if s["sample_id"] not in done]
    client, price = build_client(config, model=model, api_key=load_closeai_api_key(args.api_key_file), already_spent_usd=spent_usd(run_dir))
    started, written = time.monotonic(), 0
    with output.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(args.concurrency) as executor:
        in_flight: set[Future] = set()
        iterator = iter(pending)
        try:
            while True:
                while len(in_flight) < 2 * args.concurrency:
                    sample = next(iterator, None)
                    if sample is None:
                        break
                    in_flight.add(executor.submit(label_one, client, config, price, model, run_dir, sample))
                if not in_flight:
                    break
                finished, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in finished:
                    handle.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
                    written += 1
                handle.flush()
        except BaseException:
            for future in in_flight:
                future.cancel()
            raise
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    costs = [Decimal(r["cost_usd"]) for r in rows if r["cost_usd"]]
    summary = {
        "output": str(output), "model": model, "written_this_run": written, "rows": len(rows),
        "parse_or_call_failures": sum(r["error"] is not None for r in rows),
        "seconds_this_run": round(time.monotonic() - started, 1),
        "cost_usd_this_file": str(sum(costs, Decimal(0))), "mean_cost_usd_per_call": str(sum(costs, Decimal(0)) / len(costs)) if costs else None,
        "mean_prompt_tokens": sum(r["prompt_tokens"] for r in rows if r["prompt_tokens"]) / max(1, sum(1 for r in rows if r["prompt_tokens"])),
        "mean_completion_tokens": sum(r["completion_tokens"] for r in rows if r["completion_tokens"]) / max(1, sum(1 for r in rows if r["completion_tokens"])),
        "total_spent_usd_all_files": str(spent_usd(run_dir)), "budget_cap_usd": config["budget_hard_cap_usd"],
    }
    print(json.dumps(summary, ensure_ascii=False))


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


def load_reference(path: Path) -> dict[str, dict[str, Any]]:
    """参考标签 TSV：``sample_id cut size_a angle_a size_b angle_b conf [note]``，简码见 ``*_CODES``。"""

    reference: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 7:
            raise ValueError(f"{path}:{number}: need 7 tab-separated fields, got {len(fields)}")
        sample_id, cut, size_a, angle_a, size_b, angle_b, conf = fields[:7]
        if sample_id in reference:
            raise ValueError(f"{path}:{number}: duplicate {sample_id}")
        if conf not in ("sure", "unsure"):
            raise ValueError(f"{path}:{number}: conf must be sure|unsure")
        reference[sample_id] = {
            "cut_type": CUT_CODES[cut], "conf": conf, "note": fields[7] if len(fields) > 7 else "",
            "shot_a": {"shot_size": SIZE_CODES[size_a], "camera_angle": ANGLE_CODES[angle_a]},
            "shot_b": {"shot_size": SIZE_CODES[size_b], "camera_angle": ANGLE_CODES[angle_b]},
        }
    return reference


def _rate(hits: int, total: int) -> dict[str, Any]:
    interval = ctl.wilson_interval(hits, total)
    return {"correct": hits, "total": total, "accuracy": hits / total if total else None, "wilson95": interval}


def calibrate(samples: list[dict[str, Any]], reference: dict[str, dict[str, Any]], labels: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {s["sample_id"]: s for s in samples}
    predicted = {row["sample_id"]: row for row in labels}
    ids = [sid for sid in by_id if sid in reference and sid in predicted]
    ok = [sid for sid in ids if predicted[sid]["parsed"] is not None]
    report: dict[str, Any] = {"samples_with_reference_and_label": len(ids), "parse_or_call_failures": len(ids) - len(ok)}

    def cut_pairs(subset: Sequence[str]) -> list[tuple[str, str]]:
        return [(reference[sid]["cut_type"], predicted[sid]["parsed"]["cut_type"]) for sid in subset]

    def cut_block(subset: Sequence[str]) -> dict[str, Any]:
        pairs = cut_pairs(subset)
        matrix = ctl.confusion_matrix(pairs, ctl.CUT_TYPES)
        return {
            "n": len(pairs), "overall": _rate(sum(a == b for a, b in pairs), len(pairs)),
            "per_class": ctl.per_class_scores(matrix), "confusion_reference_rows_predicted_columns": matrix,
            "reference_counts": ctl.class_counts([a for a, _ in pairs]),
        }

    report["coarse"] = {name: ctl.binary_scores(cut_pairs(ok), positive) for name, positive in ctl.COARSE_TASKS.items()}
    report["cut_type_all"] = cut_block(ok)
    report["cut_type_sure_only"] = cut_block([sid for sid in ok if reference[sid]["conf"] == "sure"])
    for source in SOURCES:
        report[f"cut_type_{source}"] = cut_block([sid for sid in ok if by_id[sid]["source"] == source])
    for stratum in ("random", "boosted"):
        report[f"cut_type_{stratum}"] = cut_block([sid for sid in ok if by_id[sid]["stratum"] == stratum])

    def shot_pairs(field: str, subset: Sequence[str]) -> list[tuple[str, str]]:
        return [(reference[sid][shot][field], predicted[sid]["parsed"][shot][field]) for sid in subset for shot in ("shot_a", "shot_b")]

    def shot_block(subset: Sequence[str]) -> dict[str, Any]:
        sizes = shot_pairs("shot_size", subset)
        angles = shot_pairs("camera_angle", subset)
        return {
            "shots": len(sizes),
            "shot_size_exact": _rate(sum(a == b for a, b in sizes), len(sizes)),
            "shot_size_3bin": _rate(sum(ctl.SHOT_SIZE_BINS[a] == ctl.SHOT_SIZE_BINS[b] for a, b in sizes), len(sizes)),
            "shot_size_within_1": _rate(sum(ctl.size_distance(a, b) <= 1 for a, b in sizes), len(sizes)),
            "shot_size_confusion": ctl.confusion_matrix(sizes, ctl.SHOT_SIZES),
            "camera_angle_exact": _rate(sum(a == b for a, b in angles), len(angles)),
            "camera_angle_confusion": ctl.confusion_matrix(angles, ctl.CAMERA_ANGLES),
            "reference_size_counts": dict(Counter(a for a, _ in sizes)),
            "reference_angle_counts": dict(Counter(a for a, _ in angles)),
        }

    report["shot_all"] = shot_block(ok)
    report["shot_sure_only"] = shot_block([sid for sid in ok if reference[sid]["conf"] == "sure"])
    for source in SOURCES:
        report[f"shot_{source}"] = shot_block([sid for sid in ok if by_id[sid]["source"] == source])

    native_ids = [sid for sid in ok if by_id[sid]["native"] is not None]
    cut_of = {"reference": lambda sid: reference[sid]["cut_type"], "labeler": lambda sid: predicted[sid]["parsed"]["cut_type"]}
    agree: dict[str, dict[str, list[bool]]] = {"labeler": defaultdict(list), "reference": defaultdict(list)}
    size_agree = {"labeler": [], "reference": []}
    for sid in native_ids:
        features = by_id[sid]["native"]
        for who, cut in (("labeler", predicted[sid]["parsed"]["cut_type"]), ("reference", reference[sid]["cut_type"])):
            for flag, hit in ctl.native_agreement(cut, features).items():
                agree[who][flag].append(hit)
        for who, source in (("labeler", predicted[sid]["parsed"]), ("reference", reference[sid])):
            for shot, native_key in (("shot_a", "native_size_a"), ("shot_b", "native_size_b")):
                native = features[native_key]
                if native is not None:
                    size_agree[who].append(ctl.SHOT_SIZE_BINS[source[shot]["shot_size"]] == ctl.SHOT_SIZE_BINS[native])
    report["native_cross_check"] = {
        "ci_vid_samples_with_native": len(native_ids),
        "cut_type_vs_native_flags": {who: {flag: _rate(sum(hits), len(hits)) for flag, hits in flags.items()} for who, flags in agree.items()},
        "shot_size_3bin_vs_native_size_word": {who: _rate(sum(hits), len(hits)) for who, hits in size_agree.items()},
        "native_flag_counts": {flag: dict(Counter(by_id[sid]["native"][flag] for sid in native_ids)) for flag in ("camera_angle_flag", "background_flag", "camera_movement_flag")},
        "native_flag_by_class": {
            who: {
                flag: {cut: dict(Counter(by_id[sid]["native"][flag] for sid in native_ids if cut_of[who](sid) == cut)) for cut in ctl.CUT_TYPES}
                for flag in ("camera_angle_flag", "background_flag")
            }
            for who in ("reference", "labeler")
        },
    }
    return report


def cmd_calibrate(args: argparse.Namespace) -> None:
    samples = _load_samples(args.run_dir / "samples.jsonl")
    if args.split:
        samples = [s for s in samples if s["split"] == args.split]
    reference = load_reference(args.reference)
    labels = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = calibrate(samples, reference, labels)
    report["labels_file"] = str(args.labels)
    report["split"] = args.split or "all"
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.report:
        args.report.write_text(text, encoding="utf-8")
    print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    sample = commands.add_parser("sample", help="抽样并抽帧（主服务器）")
    sample.add_argument("--config", type=Path, required=True)
    sample.add_argument("--out-dir", type=Path, required=True)
    sample.set_defaults(handler=cmd_sample)

    sheets = commands.add_parser("sheets", help="联系图（人工看帧）")
    sheets.add_argument("--run-dir", type=Path, required=True)
    sheets.add_argument("--only-split", choices=("dev", "test"))
    sheets.add_argument("--per-sheet", type=int, default=4)
    sheets.add_argument("--cell-width", type=int, default=300)
    sheets.set_defaults(handler=cmd_sheets)

    label = commands.add_parser("label", help="CloseAI 看图标注")
    label.add_argument("--config", type=Path, required=True)
    label.add_argument("--run-dir", type=Path, required=True)
    label.add_argument("--api-key-file", type=Path, required=True)
    label.add_argument("--model")
    label.add_argument("--tag", help="输出文件名里的标记，默认用模型名（同模型换提示词时区分版本）")
    label.add_argument("--split", choices=("dev", "test"))
    label.add_argument("--sample-ids", help="逗号分隔的 sample_id")
    label.add_argument("--limit", type=int)
    label.add_argument("--concurrency", type=int, default=16)
    label.set_defaults(handler=cmd_label)

    calibrate_parser = commands.add_parser("calibrate", help="与人工参考标签比")
    calibrate_parser.add_argument("--run-dir", type=Path, required=True)
    calibrate_parser.add_argument("--reference", type=Path, required=True)
    calibrate_parser.add_argument("--labels", type=Path, required=True)
    calibrate_parser.add_argument("--split", choices=("dev", "test"))
    calibrate_parser.add_argument("--report", type=Path)
    calibrate_parser.set_defaults(handler=cmd_calibrate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
