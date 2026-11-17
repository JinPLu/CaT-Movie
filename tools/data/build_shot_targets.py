"""块里每镜的目标潜变量：CPU 侧的清单、索引与对拍；GPU 编码原样走 ``run/precompute.sh``（一镜一行、一镜一个文件）。

  manifest            正式配置的训练／验证冻结目标并集 → 单镜清单与原VAE provenance
  index-first-batch   首批按窗打包的缓存 → 按镜的索引行，并与块清单逐镜对账
  crosscheck-prepare  从首批已有缓存的镜里分层取若干镜：出它们的单镜清单，与首批清单里同一镜逐项对拍
  crosscheck-pixels   这些镜按单镜行读出的像素，与按原窗行读出的同一段逐元素比（CPU，走 R6 的数据集读法）
  crosscheck-latents  GPU 编码之后：新建的单镜潜变量与首批窗文件里对应的一段逐元素比
  launch              在一张卡上跑一个分片：包一层 run/precompute.sh，写状态文件与失败记录；续跑即重跑同一条命令
  index               全部分片合并后：实际所需目标镜 → 唯一shot缓存索引
  merge-indexes       显式所需镜清单与各真实来源索引 → 仅重标来源的消费索引
  publish-source-collection  已有混合缓存索引与manifest → 原样保留每个来源的共同消费合同

``--limit`` 大于 0 时只出这么多镜，用于先验；0 为全量。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from itertools import chain
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
csv.field_size_limit(sys.maxsize)
SOURCES = ("cinedance", "cinepile_t2v", "moviechat_1k")
st = None  # mspc.data.shot_targets；launch 只用标准库，其余子命令在 main 里导入


def jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def write_jsonl(path, rows):
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_json(path, value):
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def first_batch_windows(mapping):
    """首批的镜到块的映射 → 每个窗按窗内镜序排好的镜键，窗按首次出现的次序。"""
    windows = defaultdict(list)
    for row in jsonl(mapping):
        windows[row["window_key"]].append((row["window_shot"], (row["sid"], row["scene_id"], row["source_shot_index"])))
    ordered = {}
    for window_key, shots in windows.items():
        shots.sort()
        if [number for number, _ in shots] != list(range(len(shots))):
            raise ValueError(f"{window_key}: window shots are not 0..{len(shots) - 1}")
        ordered[window_key] = [key for _, key in shots]
    keys = [key for shots in ordered.values() for key in shots]
    if len(set(keys)) != len(keys):
        raise ValueError("a shot is mapped from two first-batch windows")
    return ordered


def locate(blocks, wanted):
    """块清单里这些镜各在哪个块的第几镜；块清单里没有的镜直接报错。"""
    found = {}
    for block in jsonl(blocks):
        for position, shot in enumerate(block["scene"]["shots"]):
            key = st.shot_key(shot)
            if key in wanted:
                if key in found:
                    raise ValueError(f"shot appears in two blocks: {key}")
                found[key] = (block, position)
    if len(found) != len(wanted):
        raise ValueError(f"{len(wanted) - len(found)} first-batch shots are not in the block manifest")
    return found


def manifest(args):
    from mspc.config import load_experiment_config
    from mspc.data.cache_provenance import load_cache_contract, validate_provenance

    if any(type(value) is not int or value <= 0 for value in
           (args.latent_channels, args.latent_height, args.latent_width)):
        raise ValueError("latent channels and spatial dimensions must be positive integers")
    config = load_experiment_config(args.config).to_dict()
    dataset, validation = config["dataset"], config["training"]["validation"]
    from mspc.data.scene_training_manifest import media_from_task
    media = media_from_task(config["task"])
    stage = None
    if getattr(args, "cache_stage", None) is not None:
        from mspc.training.cache_readiness import read_cache_stage
        stage = read_cache_stage(args.cache_stage)
        validation_indices = list(range(0 if validation is None else validation["sample_budget"]))
        if validation is not None and validation["text_admission_path"] is not None:
            from mspc.training.text_admission import read_text_admission
            validation_indices = read_text_admission(validation["text_admission_path"],
                sample_budget=validation["sample_budget"])["eligible_sample_indices"]
        if (stage["media"] != media or stage["training_seed"] != config["training"]["seed"]
                or stage["latent_geometry"] != {"channels": args.latent_channels,
                    "spatial_shape": [args.latent_height, args.latent_width],
                    "vae_spatial_stride": media["height"] // args.latent_height}
                or stage["planned_optimizer_updates"] != config["training"]["max_train_steps"]
                or stage["sample_budget"] != (config["training"]["max_train_steps"]
                    * stage["loader"]["world_size"] * config["training"]["gradient_accumulation_steps"])
                or stage["validation_sample_indices"] != validation_indices):
            raise ValueError("cache stage does not match the source training/validation media and full sampling plan")
    if (media["height"] % args.latent_height or media["width"] % args.latent_width
            or media["height"] // args.latent_height != media["width"] // args.latent_width):
        raise ValueError("latent geometry must exactly divide configured target pixels by one VAE stride")
    if dataset["type"] != "block_latent_cache" or set(dataset["target_roots"]) != {st.SHOT}:
        raise ValueError("sampled shot production requires an explicit shot-only cached training config")
    existing_index = origin_paths(getattr(args, "existing_index", None) or (), "--existing-index")
    existing_root = origin_paths(getattr(args, "existing_root", None) or (), "--existing-root")
    if set(existing_index) != set(existing_root):
        raise ValueError("existing cache reuse requires one --existing-root per --existing-index origin")
    # 已有来源可以有几个（例如8秒全量镜与按窗加权镜各一个根），键在来源之间不得重复；复用行记下来源名。
    existing, records = {}, {}
    for origin, index_path in existing_index.items():
        files = set()
        for entry in jsonl(index_path):
            key = st.shot_key(entry)
            if key in existing:
                raise ValueError(f"duplicate existing cache target key: {key}")
            existing[key] = {**entry, "origin": origin}
            files.add(entry["file"])
        matched = set()
        for record in json.loads((existing_root[origin] / "manifest.json").read_text())["records"]:
            name = record["cache_file"]
            if name in files:
                if name in matched:
                    raise ValueError(f"duplicate existing cache record: {origin}/{name}")
                matched.add(name)
                records[origin, name] = record
        if matched != files:
            raise ValueError(f"{origin}: existing cache index files are missing from its manifest")
    contract = {"cache_version": Path(dataset["target_roots"][st.SHOT]).name,
                "models": {"vae": {"path": str(Path(config["model"]["paths"]["vae"]).resolve(strict=True)),
                                   "revision": args.vae_revision}},
                "prompt_template_revision": "unembedded_target_text",
                "implementation_revision": args.implementation_revision}
    inputs = [("train", dataset["members_path"], dataset["sampling_manifest_path"])]
    if validation is not None:
        inputs.append(("val", validation["members_path"], validation["sampling_manifest_path"]))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.out_dir / "cache_provenance.json", contract)
    loaded = load_cache_contract(args.out_dir / "cache_provenance.json", config["model"]["paths"], config["pipeline_inputs"])
    seen, names, required, reused, train_seen = set(), [], [], [], set()
    totals, by_split, by_source = Counter(shots=0, train_frames=0, latent_frames=0), {}, {}

    def rows():
        for split, members, sampling in inputs:
            selected = (None if stage is None else stage[
                "sample_indices" if split == "train" else "validation_sample_indices"])
            for member, position, row in st.sampled_shot_rows(Path(members), Path(sampling), media=media,
                                                           sample_indices=selected):
                shot = member["shots"][position]
                key = st.shot_key(shot)
                if key in seen:
                    raise ValueError(f"sampled shot appears twice or across splits: {key}")
                seen.add(key)
                if split == "train":
                    train_seen.add(key)
                frames = row["shot_groups"][0][1]
                named = {"sid": key[0], "scene_id": key[1], "source_shot_index": key[2], "case_id": row["case_id"],
                         "train_frames": frames, "source": member["scene"]["source"],
                         "block_key": member["key"], "member_case_id": member["case_id"],
                         "block_shot": position, "split": split}
                required.append(named)
                if key in existing:
                    entry = existing[key]
                    record = records[entry["origin"], entry["file"]]
                    provenance = validate_provenance(record["provenance"])
                    fields = ("dtype", "pixel_conversion", "tiled", "tile_size", "tile_stride")
                    old_vae, new_vae = provenance["models"]["vae"], contract["models"]["vae"]
                    if (set(provenance["models"]) != {"vae"} or provenance["context"] is not None
                            or old_vae["revision"] != new_vae["revision"]
                            or os.path.realpath(old_vae["path"]) != os.path.realpath(new_vae["path"])
                            or {key: provenance["encoding"][key] for key in fields}
                            != {key: loaded["encoding"][key] for key in fields}):
                        raise ValueError(f"{key}: existing cache VAE/encoding differs from new source")
                    _, groups = st.record_groups(record)
                    interval = [entry["latent_start"], entry["latent_end"]]
                    if interval not in groups:
                        raise ValueError(f"{key}: existing cache interval is not one complete shot")
                    problems = st.reconcile(entry, record, groups.index(interval),
                        {**member["scene"], "shots": member["shots"]}, position, media=media)
                    evidence = provenance["media"]
                    shape = record["input_latents_shape"]
                    if (problems or evidence["sampling"]["target_fps"] != media["target_fps"]
                            or evidence["preprocessing"]["spatial"][0]["output_size"] != [media["width"], media["height"]]
                            or shape[1] != args.latent_channels or shape[3:] != [args.latent_height, args.latent_width]):
                        raise ValueError(f"{key}: existing cache source/geometry mismatch {problems}")
                    if not (existing_root[entry["origin"]] / entry["file"]).is_file():
                        raise FileNotFoundError(f"{key}: existing cache file is missing")
                    reused.append({**named, "origin": entry["origin"]})
                    continue
                counts = dict(shots=1, train_frames=frames, latent_frames=st.latent_frames(frames))
                totals.update(counts)
                by_split.setdefault(split, Counter()).update(counts)
                by_source.setdefault(member["scene"]["source"], Counter()).update(counts)
                names.append(named)
                yield row
                if args.limit and len(names) == args.limit:
                    return

    pending = iter(rows())
    first = next(pending, None)
    if first is not None:
        st.write_manifest_csv(args.out_dir / "scene_manifest.csv", chain((first,), pending))
    if stage is not None and not args.limit and train_seen != {tuple(key) for key in stage["target_keys"]}:
        raise ValueError("actual stage target keys differ from the frozen complete sampler plan")
    if len({row["case_id"] for row in names}) != len(names):
        raise ValueError("two shots share a case_id")
    write_jsonl(args.out_dir / "shots.jsonl", names)
    write_jsonl(args.out_dir / "required-shots.jsonl", required)
    write_jsonl(args.out_dir / "reused-shots.jsonl", reused)
    summary = {"source_config": str(args.config), "media": media, "sampling_limit_shots": args.limit,
               "cache_stage": None if stage is None else {
                   "path": str(args.cache_stage), "target_step": stage["target_step"],
                   "ready_samples": len(stage["sample_indices"])},
               "to_build": dict(totals), "by_split": {key: dict(value) for key, value in by_split.items()},
               "to_build_by_source": {source: dict(count) for source, count in by_source.items()},
               "to_build_tensor_bytes": (totals["latent_frames"] * 2 * args.latent_channels
                                         * args.latent_height * args.latent_width),
               "latent_geometry": {"channels": args.latent_channels,
                                   "spatial_shape": [args.latent_height, args.latent_width],
                                   "dtype": "bfloat16"}, "written_rows": len(names),
               "required_shots": len(required), "reused_shots": len(reused),
               "existing_index": {origin: str(path) for origin, path in existing_index.items()},
               "existing_root": {origin: str(path) for origin, path in existing_root.items()},
               "inputs": [{"split": split, "members": members, "sampling": sampling}
                          for split, members, sampling in inputs]}
    summary["encoding"] = loaded["encoding"]
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


def origin_paths(items, flag):
    """``来源名=路径`` 的重复参数 → 来源名到路径；来源名不得重复。"""
    paths = {}
    for item in items:
        origin, separator, path = item.partition("=")
        if not separator or not origin or not path or origin in paths:
            raise ValueError(f"{flag} requires distinct explicit origin=path entries")
        paths[origin] = Path(path)
    return paths


def first_batch_records(cache_dir, scene_manifest):
    """首批缓存的清单记录，按窗键取：缓存清单按 case_id 记，窗键在首批的清单 CSV 里。"""
    records = {record["case_id"]: record for record in json.loads((cache_dir / "manifest.json").read_text())["records"]}
    with scene_manifest.open(newline="", encoding="utf-8") as handle:
        return {row["window_key"]: records[row["case_id"]] for row in csv.DictReader(handle)}


def index_first_batch(args):
    windows = first_batch_windows(args.first_batch_mapping)
    records = first_batch_records(args.first_batch_cache_dir, args.first_batch_scene_manifest)
    if set(windows) != set(records):
        raise ValueError("first-batch mapping and first-batch cache hold different windows")
    if args.limit:
        kept, count = {}, 0
        for window_key, shots in windows.items():
            if count >= args.limit:
                break
            kept[window_key] = shots
            count += len(shots)
        windows = kept
    found = locate(args.blocks, {key for shots in windows.values() for key in shots})
    rows, problems, examples = [], Counter(), defaultdict(list)
    for window_key, shots in windows.items():
        record = records[window_key]
        named = [dict(zip(("sid", "scene_id", "source_shot_index"), key)) for key in shots]
        for position, (row, key) in enumerate(zip(st.index_rows(record, named, st.FIRST_BATCH_WINDOW), shots, strict=True)):
            block, block_position = found[key]
            for problem in st.reconcile(row, record, position, block["scene"], block_position, media=st.ANNOTATION_MEDIA):
                problems[problem] += 1
                if len(examples[problem]) < 5:
                    shot = block["scene"]["shots"][block_position]
                    examples[problem].append({"shot": key, "window_key": window_key, "window_shot": position,
                                              "block_key": block["key"], "block_shot": block_position,
                                              "block_render_frames": shot["render_frames"],
                                              "block_segment": [shot["start_frame"], shot["end_frame"]],
                                              "cache_train_frames": row["train_frames"],
                                              "cache_latent_frames": row["latent_end"] - row["latent_start"],
                                              "cache_segment": record["provenance"]["media"]["source_shot_segments"][position]})
            rows.append(row)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "index.first-batch.jsonl", rows)
    summary = {"windows": len(windows), "shots": len(rows), "train_frames": sum(row["train_frames"] for row in rows),
               "latent_frames": sum(row["latent_end"] - row["latent_start"] for row in rows),
               "mismatches": dict(problems), "examples": examples, "cache_dir": str(args.first_batch_cache_dir)}
    write_json(args.out_dir / "index.first-batch.summary.json", summary)
    print(json.dumps({name: summary[name] for name in ("windows", "shots", "train_frames", "latent_frames", "mismatches")},
                     ensure_ascii=False))
    return int(bool(problems))


def crosscheck_prepare(args):
    windows = first_batch_windows(args.first_batch_mapping)
    where = {key: (window_key, position, len(shots)) for window_key, shots in windows.items()
             for position, key in enumerate(shots)}
    index = {st.shot_key(row): row for row in jsonl(args.first_batch_index)}
    if set(index) != set(where):
        raise ValueError("first-batch index and first-batch mapping hold different shots")
    found = locate(args.blocks, set(where))
    candidates = []
    for key, (window_key, position, size) in where.items():
        place = "alone" if size == 1 else "first" if position == 0 else "later"
        candidates.append({**index[key], "stratum": f"{found[key][0]['source']}/{place}", "window_key": window_key,
                           "window_shot": position})
    picked = st.pick_spread(candidates, args.count)
    with args.first_batch_scene_manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        needed = {row["window_key"] for row in picked}
        window_rows = {row["window_key"]: row for row in reader if row["window_key"] in needed}
    rows, shots, differences, examples = [], [], Counter(), []
    for row in picked:
        block, block_position = found[st.shot_key(row)]
        new = st.shot_row(block["scene"], block_position, media=st.ANNOTATION_MEDIA)
        old = st.manifest_cells(window_rows[row["window_key"]])
        differing = st.manifest_differences(new, old, row["window_shot"])
        differences.update(differing)
        if differing and len(examples) < 10:
            examples.append({"case_id": new["case_id"], "window_key": row["window_key"], "window_shot": row["window_shot"],
                             "differing": differing})
        rows.append(new)
        shots.append({**{name: row[name] for name in ("sid", "scene_id", "source_shot_index", "train_frames", "stratum",
                                                       "window_key", "window_shot", "latent_start", "latent_end")},
                      "case_id": new["case_id"], "first_batch_file": row["file"],
                      "window_case_id": window_rows[row["window_key"]]["case_id"],
                      "window_frames": old["shot_groups"][row["window_shot"]],
                      "container_preroll_frames": new["container_preroll_frames"]})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    st.write_manifest_csv(args.out_dir / "scene_manifest.csv", rows)
    with (args.out_dir / "window_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(window_rows[key] for key in sorted(window_rows))
    write_jsonl(args.out_dir / "shots.jsonl", shots)
    lengths = sorted(shot["train_frames"] for shot in shots)
    report = {"shots": len(shots), "windows": len(window_rows), "by_stratum": dict(sorted(Counter(s["stratum"] for s in shots).items())),
              "train_frames": {"min": lengths[0], "median": lengths[len(lengths) // 2], "max": lengths[-1]},
              "preroll_nonzero": sum(bool(shot["container_preroll_frames"]) for shot in shots),
              "compared_cells": [*st.WINDOW_CELLS, *st.SHOT_CELLS, "shot_groups"],
              "differing_cells": dict(differences), "examples": examples}
    write_json(args.out_dir / "manifest-check.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return int(bool(differences))


_DATASETS = {}


def _pixel_pair(number):
    import torch

    shot = _DATASETS["shots"][number]
    started = time.monotonic()
    single = _DATASETS["shot"]._load_sample(number)
    window = _DATASETS["window"]._load_sample(_DATASETS["window_index"][shot["window_case_id"]])
    start, end = shot["window_frames"]
    if single["case_id"] != shot["case_id"] or window["case_id"] != shot["window_case_id"]:
        raise ValueError(f"{shot['case_id']}: dataset rows are not the listed shot and window")
    part = window["video"][:, start:end]
    return {"case_id": shot["case_id"], "shape": list(single["video"].shape), "dtype": str(single["video"].dtype),
            "equal": part.shape == single["video"].shape and bool(torch.equal(part, single["video"])),
            "seconds": time.monotonic() - started}


def crosscheck_pixels(args):
    import multiprocessing
    import torch
    from mspc.config import load_experiment_config
    from mspc.data.multishot import MultishotCsvDataset
    from mspc.training.training_args import build_common_training_args

    torch.set_num_threads(1)
    # 与 latent_cache._build_source_dataset 同一份数据集配置（uint8、不重复），只换清单路径。
    base = {**build_common_training_args(load_experiment_config(args.config)).dataset_cfg,
            "repeat": 1, "video_decode_num_threads": 1, "video_output_mode": "uint8"}
    shots = list(jsonl(args.inputs / "shots.jsonl"))
    _DATASETS.update(shots=shots, shot=MultishotCsvDataset(**{**base, "metadata_path": str(args.inputs / "scene_manifest.csv")}),
                     window=MultishotCsvDataset(**{**base, "metadata_path": str(args.inputs / "window_manifest.csv")}))
    _DATASETS["window_index"] = {row["_case_id"]: number for number, row in enumerate(_DATASETS["window"].data)}
    if [row["_case_id"] for row in _DATASETS["shot"].data] != [shot["case_id"] for shot in shots]:
        raise ValueError("single-shot manifest and shots.jsonl are not in the same order")
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        results = list(pool.imap(_pixel_pair, range(len(shots))))
    failed = [row["case_id"] for row in results if not row["equal"]]
    write_jsonl(args.out_dir / "pixel-check.jsonl", results)
    report = {"shots": len(results), "equal": len(results) - len(failed), "different": failed,
              "frames": sum(row["shape"][1] for row in results), "decode_seconds": sum(row["seconds"] for row in results)}
    write_json(args.out_dir / "pixel-check.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return int(bool(failed))


def crosscheck_latents(args):
    from safetensors.torch import load_file

    results = []
    for shot in jsonl(args.inputs / "shots.jsonl"):
        new = load_file(str(args.shot_cache_dir / "latents" / f"{shot['case_id']}.safetensors"))["input_latents"]
        old = load_file(str(args.first_batch_cache_dir / shot["first_batch_file"]))["input_latents"]
        old = old[:, :, shot["latent_start"]:shot["latent_end"]]
        if new.shape != old.shape:
            results.append({"case_id": shot["case_id"], "stratum": shot["stratum"], "shape": list(new.shape),
                            "first_batch_shape": list(old.shape), "max_steps": None, "differing": None})
            continue
        steps = st.bf16_steps(new, old)
        results.append({"case_id": shot["case_id"], "stratum": shot["stratum"], "shape": list(new.shape),
                        "max_steps": int(steps.max()), "differing": int((steps > 0).sum()),
                        "max_abs": float((new.float() - old.float()).abs().max())})
    write_jsonl(args.out_dir / "latent-check.jsonl", results)
    # 判据：逐元素数值相同，或至多差一个 BF16 步长；形状不同、差两步以上都算不过。
    failed = [row["case_id"] for row in results if row["max_steps"] is None or row["max_steps"] > 1]
    report = {"shots": len(results), "identical": sum(row["max_steps"] == 0 for row in results),
              "within_one_step": sum(row["max_steps"] == 1 for row in results), "failed": failed,
              "max_steps": max((row["max_steps"] for row in results if row["max_steps"] is not None), default=None),
              "verdict": "pass" if not failed else "fail"}
    write_json(args.out_dir / "latent-check.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return int(bool(failed))


def index(args):
    from mspc.data.cache_provenance import validate_provenance

    shots = list(jsonl(args.inputs / "shots.jsonl"))
    source_summary = json.loads((args.inputs / "summary.json").read_text())
    contract = json.loads((args.inputs / "cache_provenance.json").read_text())
    records = {}
    for record in json.loads((args.shot_cache_dir / "manifest.json").read_text())["records"]:
        if record["case_id"] in records:
            raise ValueError(f"duplicate cache record: {record['case_id']}")
        records[record["case_id"]] = record
    rows, missing = [], []
    geometry = source_summary["latent_geometry"]
    for shot in shots:
        record = records.get(shot["case_id"])
        if record is None:
            missing.append(shot["case_id"])
            continue
        provenance = validate_provenance(record["provenance"], cache_version=contract["cache_version"])
        if provenance["models"] != contract["models"] or provenance["encoding"] != source_summary["encoding"]:
            raise ValueError(f"{shot['case_id']}: cache VAE/encoding differs from the source contract")
        encoded_media = provenance["media"]
        media = source_summary["media"]
        if (encoded_media["sampling"]["target_fps"] != media["target_fps"]
                or encoded_media["preprocessing"]["spatial"][0]["output_size"] != [media["width"], media["height"]]):
            raise ValueError(f"{shot['case_id']}: cache media differs from the sampled target manifest")
        shape = record["input_latents_shape"]
        if shape != [1, geometry["channels"], st.latent_frames(shot["train_frames"]), *geometry["spatial_shape"]]:
            raise ValueError(f"{shot['case_id']}: actual cache geometry differs from the source manifest")
        row, = st.index_rows(record, [shot], st.SHOT)
        if row["train_frames"] != shot["train_frames"]:
            raise ValueError(f"{record['case_id']}: cache frame count differs from the source manifest")
        rows.append(row)
    if missing:
        raise ValueError(f"{len(missing)} sampled target shots have no cache record (first: {missing[0]})")
    if set(records) != {shot["case_id"] for shot in shots}:
        raise ValueError("merged cache must contain exactly the sampled target source manifest")
    if len({st.shot_key(row) for row in rows}) != len(rows):
        raise ValueError("a shot is indexed twice")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "index.jsonl", rows)
    summary = {"shots": len(rows), "by_origin": dict(Counter(row["origin"] for row in rows)),
               "train_frames": sum(row["train_frames"] for row in rows),
               "roots": {st.SHOT: str(args.shot_cache_dir)}}
    write_json(args.out_dir / "index.summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


def publish_source_collection(args):
    from mspc.config import load_experiment_config
    from mspc.data.cache_provenance import build_cache_source_collection, load_cache_contract

    config = load_experiment_config(args.config).to_dict()
    dataset = config["dataset"]
    if dataset["type"] != "block_latent_cache":
        raise ValueError("source collection publication requires block_latent_cache")
    collection = build_cache_source_collection(Path(dataset["target_index_path"]), dataset["target_roots"])
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != collection:
            raise FileExistsError(f"published cache source collection differs: {output}")
        load_cache_contract(output, config["model"]["paths"], config["pipeline_inputs"])
    else:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=f".{output.name}.", delete=False) as handle:
            stage = Path(handle.name)
            json.dump(collection, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        try:
            load_cache_contract(stage, config["model"]["paths"], config["pipeline_inputs"])
            # Exclusive publication preserves any independently published source evidence.
            os.link(stage, output)
        finally:
            stage.unlink()
    print(json.dumps({"contract": str(output),
                      "sources": {origin: source["selected_files"]
                                  for origin, source in collection["sources"].items()}}, ensure_ascii=False))


def merge_indexes(args):
    from mspc.data.cache_provenance import merge_selected_cache_indexes

    rows = merge_selected_cache_indexes(origin_paths(args.source_index, "--source-index"), jsonl(args.selected_shots))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if list(jsonl(args.output)) != rows:
            raise FileExistsError(f"published merged cache index differs: {args.output}")
    else:
        write_jsonl(args.output, rows)
    print(json.dumps({"index": str(args.output), "shots": len(rows),
                      "by_origin": dict(Counter(row["origin"] for row in rows))}, ensure_ascii=False))


def launch(args):
    """一张卡一个分片。环境变量与首批的启动脚本（first-batch-members/launch-r6.sh）相同，分片号经 RANK 交给现有入口。"""
    if not 0 <= args.shard_index < args.num_shards or args.status_seconds <= 0:
        raise ValueError("invalid shard or status interval")
    root, repo = args.root.resolve(), args.root.resolve() / "Storyboard"
    free_gb = shutil.disk_usage(args.output_dir.parent).free / 1e9
    if free_gb < args.min_free_gb:
        raise SystemExit(f"{args.output_dir.parent} has {free_gb:.0f} GB free, {args.min_free_gb} GB required; nothing started")
    tag = f"s{args.shard_index}of{args.num_shards}"
    for name in ("logs", "status"):
        (root / name).mkdir(exist_ok=True)
    log_path, status_path = root / "logs" / f"shot-targets-{tag}.log", root / "status" / f"shot-targets-{tag}.json"
    shard_manifest = args.output_dir / "shards" / f"shard-{args.shard_index:05d}.json"
    env = {**os.environ, "MSPC_SHARED_ROOT": str(root), "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": args.gpu,
           "NPROC_PER_NODE": "1", "WORLD_SIZE": str(args.num_shards), "RANK": str(args.shard_index),
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PATH": f"{root}/env/bin:{os.environ['PATH']}",
           "PYTHONPATH": f"{repo}/src:{repo}:{repo}/third_party/wan22_runtime"}
    started = time.monotonic()
    state = {"shard": args.shard_index, "num_shards": args.num_shards, "host": socket.gethostname(), "gpu": args.gpu,
             "output_dir": str(args.output_dir), "log": str(log_path), "state": "running",
             "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "written": 0, "bytes": 0,
             "shard_rows": None, "skipped_existing": None, "progress": None, "exit_code": None}
    stop, tail = threading.Event(), []

    def status():
        seconds = time.monotonic() - started
        snapshot = {**state, "seconds": seconds, "shots_per_hour": state["written"] / seconds * 3600,
                    "shard_complete": shard_manifest.exists()}
        if snapshot["shard_complete"] and state["state"] == "running":
            snapshot["state"] = "completed"  # 分片 0 的入口进程还会等其余分片的结束标记，本片已完成
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1))
        tmp.replace(status_path)

    def heartbeat():
        while not stop.wait(args.status_seconds):
            status()

    status()
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        process = subprocess.Popen(["bash", "run/precompute.sh", str(args.config), "--output-dir", str(args.output_dir)],
                                   cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"=== {state['started_at']} {tag} gpu={args.gpu} host={state['host']}\n")
            for line in process.stdout:
                log.write(line)
                log.flush()
                tail = [*tail[-39:], line.rstrip("\n")]
                if line.startswith("[precompute-row] "):
                    state["written"] += 1
                    state["bytes"] += json.loads(line[len("[precompute-row] "):])["size_bytes"]
                elif line.startswith("[precompute-progress] "):
                    state["progress"] = json.loads(line[len("[precompute-progress] "):])
                elif match := re.match(r'\s*"(shard_rows|skipped_existing)": (\d+)', line):
                    state[match.group(1)] = int(match.group(2))
        state["exit_code"] = process.wait()
        state["state"] = "completed" if state["exit_code"] == 0 else "failed"
    except BaseException:
        state["state"] = "failed"
        raise
    finally:
        stop.set()
        thread.join()
        status()
        if state["state"] == "failed":
            with (root / "status" / "shot-targets-failures.jsonl").open("a", encoding="utf-8") as failures:
                failures.write(json.dumps({"shard": args.shard_index, "num_shards": args.num_shards, "host": state["host"],
                                           "gpu": args.gpu, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                           "exit_code": state["exit_code"], "written_before_failure": state["written"],
                                           "log": str(log_path), "tail": tail}, ensure_ascii=False) + "\n")
    print(status_path.read_text(), flush=True)
    return state["exit_code"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def command(name, paths, numbers=(), texts=()):
        cmd = sub.add_parser(name)
        for key in paths:
            cmd.add_argument(f"--{key}", type=Path, required=True)
        for key in numbers:
            cmd.add_argument(f"--{key}", type=int, required=True)
        for key in texts:
            cmd.add_argument(f"--{key}", required=True)
        return cmd

    source_manifest = command("manifest", ("config", "out-dir"),
            ("limit", "latent-channels", "latent-height", "latent-width"),
            ("vae-revision", "implementation-revision"))
    source_manifest.add_argument("--cache-stage", type=Path,
                                 help="同完整96k采样表上的运行阶段集合，验证集仍完整生成")
    source_manifest.add_argument("--existing-index", action="append",
                                 help="来源名=按实际来源与几何核验后复用的已有镜索引，可重复，每个来源须同传--existing-root")
    source_manifest.add_argument("--existing-root", action="append",
                                 help="来源名=该来源索引对应的实际缓存根，header与张量保持原样")
    command("index-first-batch", ("blocks", "first-batch-mapping", "first-batch-scene-manifest", "first-batch-cache-dir",
                                  "out-dir"), ("limit",))
    command("crosscheck-prepare", ("blocks", "first-batch-mapping", "first-batch-scene-manifest", "first-batch-index",
                                   "out-dir"), ("count",))
    command("crosscheck-pixels", ("config", "inputs", "out-dir"), ("workers",))
    command("crosscheck-latents", ("inputs", "first-batch-cache-dir", "shot-cache-dir", "out-dir"))
    command("index", ("inputs", "shot-cache-dir", "out-dir"))
    command("publish-source-collection", ("config", "output"))
    merged = command("merge-indexes", ("selected-shots", "output"))
    merged.add_argument("--source-index", action="append", required=True, help="显式来源名=实际索引路径，可重复")
    command("launch", ("root", "config", "output-dir"), ("shard-index", "num-shards", "status-seconds", "min-free-gb"), ("gpu",))
    args = p.parse_args()
    if getattr(args, "limit", 0) < 0 or getattr(args, "workers", 1) < 1:
        p.error("--limit must be >= 0 and --workers >= 1")
    if args.command not in {"launch", "publish-source-collection"}:
        global st
        from mspc.data import shot_targets as st
    return {"manifest": manifest, "index-first-batch": index_first_batch, "crosscheck-prepare": crosscheck_prepare,
            "crosscheck-pixels": crosscheck_pixels, "crosscheck-latents": crosscheck_latents, "index": index,
            "publish-source-collection": publish_source_collection, "merge-indexes": merge_indexes,
            "launch": launch}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
