"""训练实际文字的离线清单；原 sample_index、参考与文字增强由 BlockDataset 持有。"""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
import json
from itertools import islice
import multiprocessing
from pathlib import Path
import shutil
import time
from typing import Any


TEXT_CACHE_ARTIFACT_TYPE = "mspc_training_text_context_cache"
CONTEXT_SHAPE = (1, 512, 4096)
CONTEXT_DTYPE = "bfloat16"
CONTEXT_STORAGE = {"schema_version": "2", "context_storage": "valid_token_rows"}
CONTEXT_STORAGE_CONTRACT = {
    "stored_context_shape": [1, "token_count", 4096],
    "restored_context_shape": list(CONTEXT_SHAPE),
    "context_dtype": CONTEXT_DTYPE,
    "token_tensors": "original int64 ids and mask [1,512]",
    "padding": "restore positive zero bytes after the original valid token prefix",
    "reader": "mspc.training.text_cache.read_text_context",
}


def _token_count(ids, mask) -> int:
    import torch
    if (tuple(ids.shape) != (1, 512) or ids.dtype != torch.int64
            or tuple(mask.shape) != (1, 512) or mask.dtype != torch.int64):
        raise ValueError("text context requires original int64 ids and mask [1,512]")
    count = int(mask.sum())
    expected = torch.arange(512).unsqueeze(0) < count
    if not 1 <= count <= 512 or not torch.equal(mask, expected.to(torch.int64)):
        raise ValueError("text context mask must be a nonempty contiguous prefix of ones")
    return count


def read_text_context(path: Path) -> dict[str, Any]:
    """读取唯一生产格式v2；恢复CPU BF16 [1,512,4096]，原ids/mask不变。

    文件只存context[:, :mask.sum(), :]，尾部按原prompt unit恢复为正零。
    缺文件、旧schema、字段、dtype或几何错误均立即报错，不在线编码或兼容v1。
    """
    import torch
    values = _read_compact_text_context(path)
    count = values["context"].shape[1]
    context = torch.zeros(CONTEXT_SHAPE, dtype=torch.bfloat16)
    context[:, :count].copy_(values["context"])
    return {**values, "context": context}


def _read_compact_text_context(path: Path) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        if stream.metadata() != CONTEXT_STORAGE or set(stream.keys()) != {"context", "ids", "mask"}:
            raise ValueError(f"invalid compact text context schema: {path}")
        values = {name: stream.get_tensor(name) for name in ("context", "ids", "mask")}
    count = _token_count(values["ids"], values["mask"])
    compact = values["context"]
    if compact.dtype != torch.bfloat16 or tuple(compact.shape) != (1, count, 4096):
        raise ValueError(f"invalid compact text context geometry/dtype: {path}")
    return values


def _write_text_context(path: Path, *, context, ids, mask, token_count: int) -> int:
    """每条写后读回，并与本次原pipeline CPU输出逐字节比较，不重算T5。"""
    import torch
    from safetensors.torch import save_file
    context, ids, mask = (value.detach().cpu().contiguous() for value in (context, ids, mask))
    count = _token_count(ids, mask)
    if (count != token_count or tuple(context.shape) != CONTEXT_SHAPE
            or context.dtype != torch.bfloat16 or not torch.isfinite(context).all()):
        raise ValueError(f"unexpected original text context/token geometry or dtype: {path}")
    if torch.count_nonzero(context[:, count:].view(torch.uint8)):
        raise ValueError(f"original text context has nonzero padding bytes: {path}")
    pending = path.with_name(f".{path.name}.pending")
    save_file({"context": context[:, :count].contiguous(), "ids": ids, "mask": mask},
              str(pending), metadata=CONTEXT_STORAGE)
    restored = read_text_context(pending)
    if any(not torch.equal(restored[name].view(torch.uint8), original.view(torch.uint8))
           for name, original in (("context", context), ("ids", ids), ("mask", mask))):
        raise ValueError(f"compact text context write/read bytes differ: {path}")
    pending.replace(path)
    return count * 4096 * 2


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _publish_manifest(output: Path, value: dict[str, Any]) -> None:
    pending = output / ".manifest.json.pending"
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    pending.replace(output / "manifest.json")


def _text_runtime(config: Any, manifest: dict[str, Any]):
    """仅装原冻结文本组件；逐行调用现有生产 prompt unit。"""
    import torch
    from mspc.models.wan22_loader import (
        WAN22_COMPONENT_MODE_CACHE_ENCODE_TEXT, load_pure_t2v_pipeline,
    )
    model = config.resolved["model"]
    contract = manifest["encoding_contract"]
    if (Path(contract["t5_path"]).resolve() != Path(model["paths"]["t5"]).resolve()
            or Path(contract["tokenizer_path"]).resolve() != Path(model["paths"]["tokenizer"]).resolve()):
        raise ValueError("text encoding requires the exact manifest model and tokenizer paths")
    if torch.cuda.device_count() != 1:
        raise ValueError("text encoding requires exactly one explicitly leased visible GPU")
    pipe, _ = load_pure_t2v_pipeline(diffsynth_root=model["diffsynth_root"],
        model_paths=model["paths"], torch_dtype=torch.bfloat16, device=torch.device("cuda:0"),
        self_attention_policy=model["self_attention_policy"],
        component_mode=WAN22_COMPONENT_MODE_CACHE_ENCODE_TEXT)
    pipe.freeze_except([])
    options = config.resolved["runtime"]["deepspeed"]["torch_autocast"]
    if type(options["enabled"]) is not bool or options["dtype"] != "bfloat16":
        raise ValueError("text producer requires the actual explicit BF16 autocast contract")
    if (pipe.tokenizer.seq_len != 512 or pipe.tokenizer.clean != contract["clean"]
            or len(pipe.units) != 1 or pipe.units[0]._inference_cache is not None):
        raise ValueError("text producer tokenizer/unit contract differs from live training")
    return pipe, {"enabled": options["enabled"], "dtype": options["dtype"]}


def _selected_samples(manifest, samples, sample_limit):
    if sample_limit is None:
        return samples
    if type(sample_limit) is not int or sample_limit < 1:
        raise ValueError("text encoding sample limit must be positive or None")
    selected = set()
    for split, report in manifest["splits"].items():
        selected.update((split, row["sample_index"]) for row in
                        [row for row in samples if row["split"] == split][:sample_limit])
        selected.update((split, row["sample_index"]) for row in report["boundary_samples"].values())
    return [row for row in samples if (row["split"], row["sample_index"]) in selected]


def _selected_prompt_rows(prompts, *, row_ids: list[int] | None, shard_index: int,
                          shard_count: int, sample_limit: int | None):
    if row_ids is None:
        return [row for row in prompts if row["row_id"] % shard_count == shard_index]
    if (shard_index != 0 or shard_count != 1 or sample_limit is not None
            or type(row_ids) is not list or not row_ids
            or any(type(index) is not int or index < 0 for index in row_ids)
            or len(set(row_ids)) != len(row_ids)):
        raise ValueError("explicit text row IDs require unique integers, shard 0/count 1 and no sample limit")
    available = {row["row_id"] for row in prompts}
    if len(available) != len(prompts) or not set(row_ids) <= available:
        raise ValueError("explicit text row IDs must be a subset of the unique original prompt rows")
    chosen = set(row_ids)
    return [row for row in prompts if row["row_id"] in chosen]


def encode_text_cache(config: Any, *, inputs: Path, output: Path,
                      sample_limit: int | None, shard_index: int, shard_count: int,
                      row_ids: list[int] | None = None) -> dict[str, Any]:
    """顺序逐行编码；可先覆盖真实200切片加完整stage边界，不发布未编码项。"""
    import torch
    if (type(shard_count) is not int or shard_count < 1 or type(shard_index) is not int
            or not 0 <= shard_index < shard_count):
        raise ValueError("text encoding requires an explicit shard index within the positive shard count")
    if output.exists():
        raise FileExistsError(f"text context output already exists: {output}")
    source = json.loads((inputs / "manifest.json").read_text())
    samples = _selected_samples(source, _read_rows(inputs / source["sample_manifest"]), sample_limit)
    referenced = {index for sample in samples for index in sample["context_rows"]}
    prompts = [row for row in _read_rows(inputs / source["prompt_manifest"]) if row["row_id"] in referenced]
    if len(prompts) != len(referenced):
        raise ValueError("sample context row references missing actual prompt strings")
    prompts = _selected_prompt_rows(prompts, row_ids=row_ids, shard_index=shard_index,
                                    shard_count=shard_count, sample_limit=sample_limit)
    if not prompts:
        raise ValueError("text encoding shard has no actual prompt rows")
    partial = row_ids is not None
    pipe, autocast = _text_runtime(config, source)
    output.mkdir(parents=True, exist_ok=False)
    (output / "contexts").mkdir()
    started = time.monotonic()
    context_bytes = 0
    context_file_bytes = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast["enabled"]):
        for number, row in enumerate(prompts):
            # tokenize for an explicit producer/consumer token+mask receipt; encode_prompt
            # invokes that same tokenizer itself, including cleaning/special tokens/padding.
            ids, mask = pipe.tokenizer(row["prompt"], return_mask=True, add_special_tokens=True)
            context = pipe.units[0].encode_prompt(pipe, row["prompt"])
            context_bytes += _write_text_context(output / "contexts" / f"{row['row_id']}.safetensors",
                context=context, ids=ids, mask=mask, token_count=row["token_count"])
            context_file_bytes += (output / "contexts" / f"{row['row_id']}.safetensors").stat().st_size
            if (number + 1) % 100 == 0:
                print(json.dumps({"encoded_prompts": number + 1, "expected_prompts": len(prompts),
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
    torch.cuda.synchronize()
    manifests = [("prompts.jsonl", prompts)]
    if shard_count == 1 and not partial:
        manifests.append(("sample-context-rows.jsonl", samples))
    for filename, rows in manifests:
        with (output / filename).open("x") as stream:
            for row in rows:
                _write_jsonl(stream, row)
    result = {"artifact_type": TEXT_CACHE_ARTIFACT_TYPE if shard_count == 1 and not partial else
                  "mspc_training_text_context_cache_shard", "schema_version": 2,
              "shard": {"index": shard_index, "count": shard_count,
                        "assignment": "original row_id modulo shard_count"},
              "context_storage": dict(CONTEXT_STORAGE),
              "storage_contract": dict(CONTEXT_STORAGE_CONTRACT),
              "source_manifest": str((inputs / "manifest.json").absolute()),
              "encoding_contract": source["encoding_contract"], "outer_autocast": autocast,
              "complete_stage": not partial and shard_count == 1 and source["complete"] and len(samples) == sum(
                  report["expected_samples"] for report in source["splits"].values()),
              "encoded_prompts": len(prompts),
              "encoded_samples": len(samples) if shard_count == 1 and not partial else 0,
              "selected_samples": 0 if partial else len(samples),
              "sample_limit_per_split": sample_limit,
              "context_tensor_bytes": context_bytes,
              "context_file_bytes": context_file_bytes,
              "write_read_verified_prompts": len(prompts),
              "write_read_comparison": "all restored context bytes, original ids and mask against this encoding pass",
              "independent_live_checked_prompts": 0,
              "encoding_seconds": time.monotonic() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                          "device": torch.cuda.get_device_name(0),
                          "capability": list(torch.cuda.get_device_capability(0))},
              "t5_parameter_bytes": sum(p.numel() * p.element_size() for p in pipe.text_encoder.parameters()),
              "t5_parameter_tensors": len(list(pipe.text_encoder.parameters())),
              "created_at": datetime.now(timezone.utc).isoformat(), "verified_n0": False}
    if partial:
        result["selected_prompt_rows"] = [row["row_id"] for row in prompts]
    _publish_manifest(output, result)
    return result


def finalize_text_cache(*, inputs: Path, shards: list[Path], output: Path) -> dict[str, Any]:
    """核对原始 row 分片并发布完整缓存，或供同一阶段复用的缺失文字补编缓存。"""
    if output.exists():
        raise FileExistsError(f"text context output already exists: {output}")
    source_manifest = (inputs / "manifest.json").resolve()
    source = json.loads(source_manifest.read_text())
    prompts = _read_rows(inputs / source["prompt_manifest"])
    samples = _read_rows(inputs / source["sample_manifest"])
    by_id = {row["row_id"]: row for row in prompts}
    referenced = {index for sample in samples for index in sample["context_rows"]}
    complete_stage = source["complete"]
    if type(complete_stage) is not bool or len(by_id) != len(prompts) or set(by_id) != referenced:
        raise ValueError("text finalization requires the exact original prompt row references")
    target_manifest = source_manifest
    if complete_stage:
        if (len(samples) != sum(report["expected_samples"] for report in source["splits"].values())
                or len({(row["split"], row["sample_index"]) for row in samples}) != len(samples)):
            raise ValueError("text finalization requires the complete original prompt/sample mapping")
    else:
        target_manifest = Path(source["source_manifest"]).resolve()
        target = json.loads(target_manifest.read_text())
        target_rows = _read_rows(target_manifest.parent / target["prompt_manifest"])
        if (target["complete"] is not True or target["encoding_contract"] != source["encoding_contract"]
                or not prompts or prompts != [row for row in target_rows if row["row_id"] in by_id]):
            raise ValueError("partial text finalization requires an exact ordered subset of the original stage rows")
    if len(shards) < 2 or len(set(path.resolve() for path in shards)) != len(shards):
        raise ValueError("text finalization requires all distinct explicit worker shards")
    reports = []
    files = []
    indices = set()
    payload_bytes = file_bytes = 0
    started = time.monotonic()
    for path in shards:
        report = json.loads((path / "manifest.json").read_text())
        shard = report["shard"]
        index = shard["index"]
        if (report["artifact_type"] != "mspc_training_text_context_cache_shard"
                or report["schema_version"] != 2 or report["context_storage"] != CONTEXT_STORAGE
                or report["storage_contract"] != CONTEXT_STORAGE_CONTRACT
                or shard["count"] != len(shards) or type(index) is not int
                or not 0 <= index < len(shards) or index in indices
                or shard["assignment"] != "original row_id modulo shard_count"
                or report["complete_stage"] or report["sample_limit_per_split"] is not None
                or report["selected_samples"] != len(samples)
                or Path(report["source_manifest"]).resolve() != source_manifest
                or report["encoding_contract"] != source["encoding_contract"]):
            raise ValueError(f"incomplete or mismatched text context shard: {path}")
        if reports and report["outer_autocast"] != reports[0]["outer_autocast"]:
            raise ValueError("text worker shards have different outer autocast contracts")
        indices.add(index)
        rows = _read_rows(path / "prompts.jsonl")
        expected = [row for row in prompts if row["row_id"] % len(shards) == index]
        if (rows != expected or report["encoded_prompts"] != len(expected)
                or report["write_read_verified_prompts"] != len(expected)
                or {item.name for item in (path / "contexts").iterdir()}
                != {f"{row['row_id']}.safetensors" for row in expected}):
            raise ValueError(f"text context shard does not cover its exact original rows: {path}")
        shard_payload = shard_files = 0
        for row in expected:
            item = path / "contexts" / f"{row['row_id']}.safetensors"
            values = _read_compact_text_context(item)
            count = _token_count(values["ids"], values["mask"])
            if count != row["token_count"]:
                raise ValueError(f"text context token count differs from original row: {item}")
            shard_payload += count * 4096 * 2
            shard_files += item.stat().st_size
            files.append(item)
        if (shard_payload != report["context_tensor_bytes"]
                or shard_files != report["context_file_bytes"]):
            raise ValueError(f"text context shard physical bytes differ from producer: {path}")
        payload_bytes += shard_payload
        file_bytes += shard_files
        reports.append(report)
    output.mkdir(parents=True, exist_ok=False)
    (output / "contexts").mkdir()
    # All sources are verified before any mutation. On one filesystem these moves
    # keep one payload copy; no runnable manifest exists until every move finishes.
    for path in files:
        shutil.move(str(path), str(output / "contexts" / path.name))
    shutil.copyfile(inputs / source["prompt_manifest"], output / "prompts.jsonl")
    if complete_stage:
        shutil.copyfile(inputs / source["sample_manifest"], output / "sample-context-rows.jsonl")
    (output / "encoding-shards").mkdir()
    for report in reports:
        (output / "encoding-shards" / f"{report['shard']['index']}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    result = {
        "artifact_type": TEXT_CACHE_ARTIFACT_TYPE if complete_stage else "mspc_training_text_context_cache_shard",
        "schema_version": 2,
        "context_storage": dict(CONTEXT_STORAGE), "storage_contract": dict(CONTEXT_STORAGE_CONTRACT),
        "source_manifest": str(target_manifest),
        "encoding_contract": source["encoding_contract"], "outer_autocast": reports[0]["outer_autocast"],
        "complete_stage": complete_stage, "encoded_prompts": len(prompts),
        "encoded_samples": len(samples) if complete_stage else 0,
        "context_tensor_bytes": payload_bytes, "context_file_bytes": file_bytes,
        "write_read_verified_prompts": len(prompts),
        "write_read_comparison": reports[0]["write_read_comparison"],
        "independent_live_checked_prompts": 0, "verified_n0": False,
        "worker_shards": len(shards),
        "encoding_worker_seconds": [report["encoding_seconds"] for report in reports],
        "encoding_runtimes": [report["runtime"] for report in reports],
        "finalization_seconds": time.monotonic() - started,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if not complete_stage:
        result.update(
            shard={"index": 0, "count": 1, "assignment": "original row_id modulo shard_count"},
            selected_samples=0, sample_limit_per_split=None, selected_prompt_rows=list(by_id),
        )
    _publish_manifest(output, result)
    return result


def _require_reusable_text_contract(target: dict[str, Any], original: dict[str, Any]) -> None:
    paths = {"t5_path", "tokenizer_path"}
    if ({key: value for key, value in target.items() if key not in paths}
            != {key: value for key, value in original.items() if key not in paths}):
        raise ValueError("text context reuse requires the same encoding contract")
    if not Path(target["t5_path"]).samefile(original["t5_path"]):
        raise ValueError("text context reuse requires the same physical T5 weights")
    left, right = (Path(contract["tokenizer_path"]) for contract in (target, original))
    if left.samefile(right):
        return
    for name in ("spiece.model", "tokenizer.json", "tokenizer_config.json"):
        if (left / name).read_bytes() != (right / name).read_bytes():
            raise ValueError(f"text context reuse requires identical tokenizer bytes: {name}")
    # A separate special_tokens_map.json can alter otherwise identical core files.
    from transformers import AutoTokenizer
    tokenizers = [AutoTokenizer.from_pretrained(str(path), local_files_only=True) for path in (left, right)]
    first, second = tokenizers
    if (type(first) is not type(second) or first.get_vocab() != second.get_vocab()
            or first.special_tokens_map != second.special_tokens_map
            or first.all_special_ids != second.all_special_ids):
        raise ValueError("text context reuse requires identical resolved tokenizers")


def _reusable_text_context_rows(
    context_sources: tuple[Path, ...], *, encoding_contract: dict[str, Any], outer_autocast: dict[str, Any],
) -> dict[str, tuple[dict[str, Any], Path]]:
    if len(set(path.resolve() for path in context_sources)) != len(context_sources):
        raise ValueError("text context reuse requires distinct finalized sources")
    by_prompt = {}
    for root in context_sources:
        original = json.loads((root / "manifest.json").read_text())
        original_prompts = _read_rows(root / "prompts.jsonl")
        if (original["artifact_type"] != TEXT_CACHE_ARTIFACT_TYPE or original["schema_version"] != 2
                or not original["complete_stage"] or original["context_storage"] != CONTEXT_STORAGE
                or original["storage_contract"] != CONTEXT_STORAGE_CONTRACT
                or original["encoded_prompts"] != len(original_prompts)
                or original["write_read_verified_prompts"] != len(original_prompts)):
            raise ValueError(f"text context reuse requires a finalized, write/read verified source cache: {root}")
        _require_reusable_text_contract(encoding_contract, original["encoding_contract"])
        if original["outer_autocast"] != outer_autocast:
            raise ValueError(f"text context reuse requires the same actual outer autocast: {root}")
        if len({row["prompt"] for row in original_prompts}) != len(original_prompts):
            raise ValueError(f"text context reuse source has duplicate actual prompt strings: {root}")
        for row in original_prompts:
            matched, _ = by_prompt.setdefault(row["prompt"], (row, root))
            if matched["token_count"] != row["token_count"]:
                raise ValueError(f"text context reuse sources disagree on an actual prompt's token count: {root}")
    return by_prompt


def _ordered_context_file_operations(operation, items, workers):
    if workers == 1:
        yield from map(operation, items)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        source = iter(items)
        while batch := list(islice(source, 512)):
            yield from pool.map(operation, batch)


def reuse_text_cache(config: Any, *, inputs: Path, context_source: Path, output: Path,
                     additional_contexts: Path | None = None, context_sources: tuple[Path, ...] = (),
                     workers: int = 1) -> dict[str, Any]:
    """用原完整字符串复用已封存context；保留本地row与sample原文件，不复制payload。"""
    started = time.monotonic()
    if type(workers) is not int or workers < 1:
        raise ValueError("text context reuse workers must be a positive integer")
    if output.exists():
        raise FileExistsError(f"text context output already exists: {output}")
    source = json.loads((inputs / "manifest.json").read_text())
    prompts = _read_rows(inputs / source["prompt_manifest"])
    samples = _read_rows(inputs / source["sample_manifest"])
    row_ids = {row["row_id"] for row in prompts}
    if (not source["complete"] or len(row_ids) != len(prompts)
            or row_ids != {index for sample in samples for index in sample["context_rows"]}
            or len(samples) != sum(report["expected_samples"] for report in source["splits"].values())
            or len({(row["split"], row["sample_index"]) for row in samples}) != len(samples)):
        raise ValueError("text context reuse requires the complete local prompt/sample mapping")
    options = config.resolved["runtime"]["deepspeed"]["torch_autocast"]
    outer_autocast = {key: options[key] for key in ("enabled", "dtype")}
    source_roots = (context_source, *context_sources)
    by_prompt = _reusable_text_context_rows(source_roots, encoding_contract=source["encoding_contract"],
                                          outer_autocast=outer_autocast)
    source_manifests = {root: str((root / "manifest.json").absolute()) for root in source_roots}
    missing = [row for row in prompts if row["prompt"] not in by_prompt]
    additional_rows = {}
    additional_manifest = None
    if additional_contexts is not None:
        additional = json.loads((additional_contexts / "manifest.json").read_text())
        extra_prompts = _read_rows(additional_contexts / "prompts.jsonl")
        expected_ids = [row["row_id"] for row in missing]
        if (not missing or additional["artifact_type"] != "mspc_training_text_context_cache_shard"
                or additional["schema_version"] != 2 or additional["complete_stage"] is not False
                or additional["context_storage"] != CONTEXT_STORAGE
                or additional["storage_contract"] != CONTEXT_STORAGE_CONTRACT
                or Path(additional["source_manifest"]).resolve() != (inputs / "manifest.json").resolve()
                or additional["encoding_contract"] != source["encoding_contract"]
                or additional["outer_autocast"] != outer_autocast
                or additional["shard"] != {"index": 0, "count": 1,
                    "assignment": "original row_id modulo shard_count"}
                or additional["encoded_samples"] != 0 or additional["selected_samples"] != 0
                or additional["sample_limit_per_split"] is not None
                or additional["selected_prompt_rows"] != expected_ids
                or additional["encoded_prompts"] != len(missing)
                or additional["write_read_verified_prompts"] != len(missing)
                or extra_prompts != missing
                or (additional_contexts / "sample-context-rows.jsonl").exists()
                or {path.name for path in (additional_contexts / "contexts").iterdir()}
                    != {f"{index}.safetensors" for index in expected_ids}):
            raise ValueError("additional text contexts must exactly cover the missing original target rows")
        additional_rows = {row["row_id"]: row["token_count"] for row in missing}
        additional_manifest = str((additional_contexts / "manifest.json").absolute())
    links = []
    reused_by_source = Counter()
    for row in prompts:
        match = by_prompt.get(row["prompt"])
        if match is not None:
            matched, root = match
            if matched["token_count"] != row["token_count"]:
                raise ValueError(f"text context reuse has a missing or mismatched actual prompt: {row['row_id']}")
            source_row_id, context_manifest = matched["row_id"], source_manifests[root]
            reused_by_source[context_manifest] += 1
        elif row["row_id"] in additional_rows:
            root = additional_contexts
            source_row_id, context_manifest = row["row_id"], additional_manifest
        else:
            raise ValueError(f"text context reuse has a missing or mismatched actual prompt: {row['row_id']}")
        path = (root / "contexts" / f"{source_row_id}.safetensors").absolute()
        links.append((row["row_id"], source_row_id, path, context_manifest))

    def verify_file(link):
        row_id, _, path, context_manifest = link
        is_additional = context_manifest == additional_manifest
        if is_additional and _read_compact_text_context(path)["context"].shape[1] != additional_rows[row_id]:
            raise ValueError(f"additional text context token count differs from the original row: {row_id}")
        return path.stat().st_size, is_additional

    file_bytes = extra_bytes = 0
    for size, is_additional in _ordered_context_file_operations(verify_file, links, workers):
        file_bytes += size
        if is_additional:
            extra_bytes += size
    if additional_manifest is not None and (
            additional["context_tensor_bytes"] != sum(additional_rows.values()) * 4096 * 2
            or additional["context_file_bytes"] != extra_bytes):
        raise ValueError("additional text context physical bytes differ from its producer")

    output.mkdir(parents=True, exist_ok=False)
    (output / "contexts").mkdir()

    def link_file(link):
        row_id, source_row_id, path, context_manifest = link
        (output / "contexts" / f"{row_id}.safetensors").symlink_to(path)
        return {"row_id": row_id, "source_row_id": source_row_id, "source_manifest": context_manifest}

    with (output / "context-reuse.jsonl").open("x") as stream:
        for row in _ordered_context_file_operations(link_file, links, workers):
            _write_jsonl(stream, row)
    for source_name, target_name in ((source["prompt_manifest"], "prompts.jsonl"),
                                    (source["sample_manifest"], "sample-context-rows.jsonl")):
        shutil.copyfile(inputs / source_name, output / target_name)
    result = {
        "artifact_type": TEXT_CACHE_ARTIFACT_TYPE, "schema_version": 2,
        "context_storage": dict(CONTEXT_STORAGE), "storage_contract": dict(CONTEXT_STORAGE_CONTRACT),
        "source_manifest": str((inputs / "manifest.json").absolute()),
        "encoding_contract": source["encoding_contract"], "outer_autocast": outer_autocast,
        "complete_stage": True, "encoded_prompts": len(prompts), "encoded_samples": len(samples),
        "context_tensor_bytes": sum(row["token_count"] for row in prompts) * 4096 * 2,
        "context_file_bytes": file_bytes, "write_read_verified_prompts": len(prompts),
        "write_read_comparison": "inherited original source producer write/read verification by exact raw prompt",
        "reused_prompts": sum(reused_by_source.values()), "encoded_here_prompts": 0,
        "newly_encoded_source_prompts": len(additional_rows),
        "context_source_manifest": source_manifests[context_source],
        "context_reuse_manifest": "context-reuse.jsonl",
        "reuse_seconds": time.monotonic() - started,
        "independent_live_checked_prompts": 0, "verified_n0": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if context_sources:
        result["context_source_manifests"] = list(source_manifests.values())
        result["reused_prompts_by_source"] = dict(reused_by_source)
    if additional_manifest is not None:
        result["additional_contexts_manifest"] = additional_manifest
    _publish_manifest(output, result)
    return result


def check_text_cache(config: Any, *, inputs: Path, output: Path,
                     sample_limit: int | None = None) -> dict[str, Any]:
    """新进程重载原文本组件，逐值对拍原live方法的完整样本context及ids/mask。"""
    import torch
    from mspc.training.backends.wan22 import Wan22DiffSynthTrainingModule, _memory_prefix_options
    if output.exists():
        raise FileExistsError(f"text context check output already exists: {output}")
    encoded = json.loads((inputs / "manifest.json").read_text())
    if (encoded["artifact_type"] != TEXT_CACHE_ARTIFACT_TYPE or encoded["schema_version"] != 2
            or encoded["context_storage"] != CONTEXT_STORAGE):
        raise ValueError("text context check requires compact schema v2")
    source = json.loads(Path(encoded["source_manifest"]).read_text())
    pipe, autocast = _text_runtime(config, source)
    # Bind the unmodified production live method without constructing an unused DiT/VAE.
    live = Wan22DiffSynthTrainingModule.__new__(Wan22DiffSynthTrainingModule)
    torch.nn.Module.__init__(live)
    object.__setattr__(live, "_pipe", pipe)
    live.device, live.dtype = torch.device("cuda:0"), torch.bfloat16
    live.memory_prefix_options = _memory_prefix_options(config.resolved["model"])
    prompts = {row["row_id"]: row for row in _read_rows(inputs / "prompts.jsonl")}
    all_samples = _read_rows(inputs / "sample-context-rows.jsonl")
    samples = _selected_samples(source, all_samples, sample_limit)
    token_checked = set()
    started = time.monotonic()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast["enabled"]):
        for number, sample in enumerate(samples):
            rows = sample["context_rows"]
            tensors = [read_text_context(inputs / "contexts" / f"{index}.safetensors") for index in rows]
            data = {"prompt": [prompts[index]["prompt"] for index in rows],
                    "shot_groups": [[index, index + 1] for index in range(len(rows))],
                    "memory_frame_count": len(sample["metadata"]["references"])}
            expected = live._encode_live_t5_context(data).cpu()
            actual = torch.cat([value["context"] for value in tensors])
            if (actual.dtype != expected.dtype or actual.shape != expected.shape
                    or not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))):
                raise ValueError(f"offline/live context differs: {sample['split']}/{sample['sample_index']}")
            for index, value in zip(rows, tensors, strict=True):
                if index in token_checked:
                    continue
                ids, mask = pipe.tokenizer(prompts[index]["prompt"], return_mask=True, add_special_tokens=True)
                if not torch.equal(ids, value["ids"]) or not torch.equal(mask, value["mask"]):
                    raise ValueError(f"offline/live ids or mask differs: row {index}")
                token_checked.add(index)
            if (number + 1) % 50 == 0:
                print(json.dumps({"checked_samples": number + 1, "expected_samples": len(samples),
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
    torch.cuda.synchronize()
    output.mkdir(parents=True, exist_ok=False)
    result = {"verified_n0": True, "checked_samples": len(samples), "checked_unique_prompts": len(token_checked),
              "complete_stage": encoded["complete_stage"] and len(samples) == len(all_samples),
              "cache_complete_stage": encoded["complete_stage"],
              "sample_limit_per_split": sample_limit, "cache": str(inputs.resolve()),
              "owner": "Wan22DiffSynthTrainingModule._encode_live_t5_context",
              "comparison": "all context bytes, output dtype, prompt order, tokenizer ids and mask",
              "check_seconds": time.monotonic() - started, "outer_autocast": autocast,
              "created_at": datetime.now(timezone.utc).isoformat()}
    (output / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
_TEXT_CACHE_DATASET = None


def _write_jsonl(stream: Any, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def _init_text_cache_worker(dataset: Any) -> None:
    global _TEXT_CACHE_DATASET
    _TEXT_CACHE_DATASET = dataset


def _text_cache_sample(index: int) -> dict[str, Any]:
    sample = _TEXT_CACHE_DATASET.text(index)
    return {"sample_index": index, "prompts": sample.prompts, "metadata": sample.metadata,
            "reference_count": len(sample.draw.references),
            "target_count": sample.draw.span[1] - sample.draw.span[0]}


def _ordered_text_cache_samples(dataset: Any, indices: list[int], *, workers: int):
    """复用准入producer的有界fork生产方式，回写顺序始终是原stage顺序。"""
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork"),
            initializer=_init_text_cache_worker, initargs=(dataset,)) as pool:
        source = iter(indices)
        pending = deque(pool.submit(_text_cache_sample, index) for index in islice(source, 2 * workers))
        while pending:
            yield pending.popleft().result()
            index = next(source, None)
            if index is not None:
                pending.append(pool.submit(_text_cache_sample, index))


def _metadata_panels(config: Any, stage: dict[str, Any]):
    """仅投影目标存储几何；保留生产成员、真实参考标签、准入与全部抽签合同。"""
    from mspc.training.block_dataset import BlockDataset, build_validation_dataset
    from mspc.training.cache_readiness import cache_stage_plan
    from mspc.training.training_args import (
        build_common_training_args, configure_training_topology, replace_training_args,
    )

    world_size = stage["loader"]["world_size"]
    geometry = stage["latent_geometry"]
    args = configure_training_topology(build_common_training_args(config),
                                       gpus_per_node=world_size, nnodes=1)
    projected = {
        **args.dataset_cfg, "target_storage": "video", "target_index_path": None,
        "target_roots": None, "target_mask_weights": None,
        "target_latent_geometry": {
            "channels": geometry["channels"], "spatial_stride": geometry["vae_spatial_stride"],
        },
    }
    dataset = BlockDataset(**projected)
    if cache_stage_plan(dataset, args, world_size=world_size,
                        target_step=stage["target_step"]) != stage:
        raise ValueError("text cache requires the actual complete sampler, loader, media and stage plan")
    panels = [("train", dataset, stage["sample_indices"])]
    if args.validation is not None:
        validation = build_validation_dataset(replace_training_args(
            args, dataset_cfg=projected, cache_stage_plan=None))
        panels.append(("validation", validation, stage["validation_sample_indices"]))
    elif stage["validation_sample_indices"]:
        raise ValueError("text cache stage contains validation indices without a validation dataset")
    return panels


def prepare_text_cache(config: Any, *, stage: dict[str, Any], output: Path,
                       sample_limit: int | None, workers: int) -> dict[str, Any]:
    """按真实 stage+val 产出逐字清单，不编码 T5、不读取视频或目标 tensor。

    sample_limit 是每个 split 的确定性原 index 前缀，None 才覆盖完整 stage。
    同字符串仅在本次明确编码合同内共享连续 row_id；不规范化字符串或建立 hash 身份。
    """
    from mspc.config.experiment import _plain
    from mspc.data.shot_text import UMT5_PROMPT_TOKEN_BUDGET

    if sample_limit is not None and (type(sample_limit) is not int or sample_limit < 1):
        raise ValueError("text cache sample_limit must be an explicit positive integer or None")
    if type(workers) is not int or workers < 1:
        raise ValueError("text cache workers must be an explicit positive integer")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"text cache output already exists: {output}")
    started = time.monotonic()
    panels = _metadata_panels(config, stage)
    initialization_seconds = time.monotonic() - started
    output.mkdir(parents=True, exist_ok=False)
    prompt_ids: dict[str, int] = {}
    token_counts: dict[str, int] = {}
    split_reports = {}
    contracts = {}
    unique_bytes = 0
    total_prompt_rows = 0
    longest = None
    with (output / "prompts.jsonl").open("x", encoding="utf-8") as prompts_stream, \
            (output / "sample-context-rows.jsonl").open("x", encoding="utf-8") as samples_stream:
        for split, dataset, indices in panels:
            selected = indices if sample_limit is None else indices[:sample_limit]
            contracts[split] = _plain(dataset.text_admission_contract)
            counts = Counter()
            boundary_samples: dict[str, dict[str, Any]] = {}
            maximum_references = -1
            maximum_target_shots = -1
            maximum_prompt_tokens = -1
            for sample in _ordered_text_cache_samples(dataset, selected, workers=workers):
                index, metadata = sample["sample_index"], sample["metadata"]
                actual_counts = metadata["actual_token_counts"]
                if (len(actual_counts) != len(sample["prompts"])
                        or max(actual_counts) > UMT5_PROMPT_TOKEN_BUDGET):
                    raise ValueError(f"{split} sample {index}: invalid or over-limit actual prompt rows")
                context_rows = []
                for position, (prompt, tokens) in enumerate(zip(sample["prompts"], actual_counts, strict=True)):
                    if prompt not in prompt_ids:
                        row_id = len(prompt_ids)
                        prompt_ids[prompt], token_counts[prompt] = row_id, tokens
                        _write_jsonl(prompts_stream, {
                            "row_id": row_id, "prompt": prompt, "token_count": tokens,
                            "first_use": {"split": split, "sample_index": index, "prompt_position": position},
                        })
                        unique_bytes += len(prompt.encode("utf-8"))
                    elif token_counts[prompt] != tokens:
                        raise ValueError("identical actual prompt strings produced different token counts")
                    context_rows.append(prompt_ids[prompt])
                    if longest is None or tokens > longest["token_count"]:
                        longest = {"split": split, "sample_index": index, "prompt_position": position,
                                   "row_id": prompt_ids[prompt], "token_count": tokens}
                record = {"split": split, "sample_index": index, "context_rows": context_rows,
                          "metadata": metadata}
                _write_jsonl(samples_stream, record)
                reference_count, target_count = sample["reference_count"], sample["target_count"]
                high = max(actual_counts)
                counts["samples"] += 1
                counts["prompt_rows"] += len(context_rows)
                counts["reference_prompt_rows"] += reference_count
                counts["target_prompt_rows"] += target_count
                counts[f"target_shots_{target_count}"] += 1
                counts["with_reference" if reference_count else "without_reference"] += 1
                for name in ("layout_text_dropped", "picture_type_text_dropped",
                             "reference_group_dropped", "reference_text_dropped"):
                    counts[name] += int(metadata[name])
                # 第一条真实边界样本是后续N0核对的输入；只存指向完整样本行的轻量选择。
                chosen = {"sample_index": index, "context_rows": context_rows}
                for label, condition in (
                    ("with_reference", bool(reference_count)), ("without_reference", not reference_count),
                    ("single_shot", target_count == 1), ("two_shots", target_count == 2),
                    ("multiple_shots", target_count > 2),
                    ("layout_dropped", metadata["layout_text_dropped"]),
                    ("layout_retained", not metadata["layout_text_dropped"]),
                    ("picture_type_dropped", metadata["picture_type_text_dropped"]),
                    ("picture_type_retained", not metadata["picture_type_text_dropped"]),
                ):
                    if condition and label not in boundary_samples:
                        boundary_samples[label] = chosen
                combination = (f"layout_{'dropped' if metadata['layout_text_dropped'] else 'retained'}"
                               f"_picture_type_{'dropped' if metadata['picture_type_text_dropped'] else 'retained'}")
                counts[combination] += 1
                if combination not in boundary_samples:
                    boundary_samples[combination] = chosen
                for label, value, previous in (
                    ("most_references", reference_count, maximum_references),
                    ("most_target_shots", target_count, maximum_target_shots),
                    ("longest_prompt", high, maximum_prompt_tokens),
                ):
                    if value > previous:
                        boundary_samples[label] = {**chosen, "value": value}
                maximum_references = max(maximum_references, reference_count)
                maximum_target_shots = max(maximum_target_shots, target_count)
                maximum_prompt_tokens = max(maximum_prompt_tokens, high)
            total_prompt_rows += counts["prompt_rows"]
            split_reports[split] = {
                "expected_samples": len(indices), "produced_samples": len(selected),
                "complete": len(selected) == len(indices), "counts": dict(counts),
                "boundary_samples": boundary_samples,
            }
    complete = all(report["complete"] for report in split_reports.values())
    model = config.resolved["model"]
    manifest = {
        "artifact_type": TEXT_CACHE_ARTIFACT_TYPE, "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(), "complete": complete,
        "prompt_manifest": "prompts.jsonl", "sample_manifest": "sample-context-rows.jsonl",
        "sample_limit_per_split": sample_limit, "workers": workers,
        "stage": _plain(stage), "contracts": contracts,
        "encoding_contract": {
            "t5_path": str(Path(model["paths"]["t5"]).resolve()),
            "tokenizer_path": str(Path(model["paths"]["tokenizer"]).resolve()),
            "owner": "WanVideoUnit_PromptEmbedder.encode_prompt",
            "call_policy": "one original prompt string per tokenizer and T5 call",
            "add_special_tokens": True, "padding": "max_length", "max_length": 512,
            "clean": "whitespace", "mask": "original tokenizer attention_mask",
            "masked_output": "set all tokens at or beyond the actual mask length to zero",
            "context_shape_per_unique_prompt": list(CONTEXT_SHAPE), "context_dtype": CONTEXT_DTYPE,
        },
        "index_semantics": {
            "draw_key": ["training_seed", "original_sample_index"],
            "metadata_epoch": "original_sample_index // number_of_member_blocks",
            "loader_epoch": "changes order only; does not redraw prompt, references or dropout",
            "row_identity": "exact actual string within this encoding contract; first-use consecutive integer",
        },
        "splits": split_reports, "unique_prompts": len(prompt_ids), "total_prompt_rows": total_prompt_rows,
        "unique_prompt_utf8_bytes": unique_bytes, "longest_prompt": longest,
        "storage_budget": {
            "context_storage": dict(CONTEXT_STORAGE),
            "bytes_per_token": 4096 * 2,
            "context_tensor_bytes": sum(token_counts.values()) * 4096 * 2,
            "context_tensor_gib": sum(token_counts.values()) * 4096 * 2 / 2**30,
            "scope": "tensor payload only; safetensors and manifest metadata excluded",
        },
        "encoding_status": "not_started", "numerics_status": "not_checked",
        "initialization_seconds": initialization_seconds,
        "production_seconds": time.monotonic() - started,
    }
    _publish_manifest(output, manifest)
    return manifest
