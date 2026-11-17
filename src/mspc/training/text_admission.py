"""真实固定抽签文字的512-token准入；保留原data_id，不重抽或重排冻结窗口。"""
from __future__ import annotations

import json
import multiprocessing
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any, Mapping

from mspc.config.experiment import _plain
from mspc.data.shot_text import SHOT_TEXT_COMPILER_CONTRACT, UMT5_PROMPT_TOKEN_BUDGET


ARTIFACT_TYPE = "mspc_renderer_text_admission"
CONTRACT_FIELDS = (
    "members_path", "sampling_manifest_path", "sample_budget", "tokenizer_path", "media",
    "training_seed", "reference_policy", "reference_row_dropout_p", "memory_prefix_options",
    "reference_entities", "omit_target_text", "box_dropout_p", "picture_type_dropout_p",
)
_TEXT_DATASET = None


def _init_text_worker(dataset: Any) -> None:
    global _TEXT_DATASET
    _TEXT_DATASET = dataset


def _text_admission_row(index: int) -> dict[str, Any]:
    sample = _TEXT_DATASET.text(index)
    high = max(sample.metadata["actual_token_counts"])
    return {**sample.metadata, "eligible": high <= UMT5_PROMPT_TOKEN_BUDGET,
            "max_prompt_tokens": high, "compiled_targets": list(sample.compiled_targets)}


def _ordered_text_rows(dataset: Any, checked: int, *, workers: int):
    """fork共享已构造的只读数据；最多2×workers项在途，固定原data_id顺序写回。"""
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork"),
            initializer=_init_text_worker, initargs=(dataset,)) as pool:
        indices = iter(range(checked))
        pending = deque(pool.submit(_text_admission_row, index)
                        for index in islice(indices, 2 * workers))
        while pending:
            yield pending.popleft().result()
            index = next(indices, None)
            if index is not None:
                pending.append(pool.submit(_text_admission_row, index))


def text_admission_contract(values: Mapping[str, Any]) -> dict[str, Any]:
    """合同只含会改变实际文字/参考抽签的显式输入，不含缓存位置或身份哈希。"""
    contract = _plain({name: values[name] for name in CONTRACT_FIELDS})
    contract["tokenizer_path"] = str(Path(contract["tokenizer_path"]).resolve())
    contract["prompt_compiler_contract"] = SHOT_TEXT_COMPILER_CONTRACT
    return contract


def read_text_admission(path: str | Path, *, sample_budget: int,
                        contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if (value.get("artifact_type") != ARTIFACT_TYPE or value.get("schema_version") != 1
            or value.get("complete") is not True or value.get("sample_budget") != sample_budget
            or value.get("token_limit") != UMT5_PROMPT_TOKEN_BUDGET):
        raise ValueError("text admission must cover the complete original sample-index space at 512 tokens")
    stored_contract = value.get("contract")
    if (not isinstance(stored_contract, dict)
            or stored_contract.get("prompt_compiler_contract") != SHOT_TEXT_COMPILER_CONTRACT):
        raise ValueError("text admission must be produced with the current prompt compiler contract")
    if contract is not None and text_admission_contract(value["contract"]) != text_admission_contract(contract):
        raise ValueError("text admission differs from the actual members, labels, tokenizer or fixed text draws")
    eligible, excluded = value.get("eligible_sample_indices"), value.get("excluded_samples")
    if (not isinstance(eligible, list) or not eligible or any(type(index) is not int for index in eligible)
            or eligible != sorted(set(eligible)) or not isinstance(excluded, list)):
        raise ValueError("text admission requires distinct sorted eligible original indices and excluded records")
    removed = [row["sample_index"] for row in excluded]
    if (any(type(index) is not int for index in removed) or removed != sorted(set(removed))
            or set(eligible) & set(removed) or sorted([*eligible, *removed]) != list(range(sample_budget))):
        raise ValueError("text admission must partition every original sample index exactly once")
    if any(row.get("max_prompt_tokens", 0) <= UMT5_PROMPT_TOKEN_BUDGET for row in excluded):
        raise ValueError("excluded text samples require an observed prompt above the token limit")
    return value


def write_text_admission(config: Any, *, split: str, world_size: int, latent_channels: int, vae_spatial_stride: int,
                         workers: int, output: Path, rows_output: Path, limit: int | None = None) -> dict[str, Any]:
    """复用唯一BlockDataset.text，保真实身份行与dropout；只投影无需读取VAE的目标几何。"""
    from mspc.training.block_dataset import BlockDataset
    from mspc.training.training_args import build_common_training_args, configure_training_topology

    if split not in {"train", "validation"}:
        raise ValueError("text admission split must be train or validation")
    if type(workers) is not int or workers < 1:
        raise ValueError("text admission workers must be explicitly positive")
    if (type(latent_channels) is not int or latent_channels < 1
            or type(vae_spatial_stride) is not int or vae_spatial_stride < 1):
        raise ValueError("text admission requires explicit actual VAE channels and spatial stride")
    started = time.monotonic()
    args = build_common_training_args(config)
    args = configure_training_topology(args, gpus_per_node=world_size, nnodes=1)
    values = dict(args.dataset_cfg)
    if split == "validation":
        if args.validation is None:
            raise ValueError("validation text admission requires the configured validation dataset")
        for name in ("members_path", "sampling_manifest_path", "sample_budget", "reference_entities"):
            values[name] = args.validation[name]
        if values["planner_tokens"] is not None:
            values["planner_tokens"] = {**values["planner_tokens"], **args.validation["planner_tokens"]}
    # 这是待发布准入表的producer，不读取自己尚不存在的产物；实际文字/标签合同不投影。
    # 裁剪只改参考像素、负载上限只管组批，都不影响文字，准入不读它们。
    values.update(target_storage="video", target_index_path=None, target_roots=None,
                  target_mask_weights=None, text_admission_path=None, reference_crops=None, workload_cap=None,
                  target_latent_geometry={"channels": latent_channels, "spatial_stride": vae_spatial_stride})
    dataset = BlockDataset(**values)
    initialization_seconds = time.monotonic() - started
    budget = len(dataset)
    if limit is not None and (type(limit) is not int or not 1 <= limit <= budget):
        raise ValueError("text admission limit must be a positive prefix within the original sample budget")
    checked = budget if limit is None else limit
    if output.exists() or rows_output.exists():
        raise FileExistsError("text admission outputs must not overwrite an earlier result")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    eligible, excluded, maximum, prompt_count = [], [], 0, 0
    with rows_output.open("x", encoding="utf-8") as stream:
        for record in _ordered_text_rows(dataset, checked, workers=workers):
            index, high = record["sample_index"], record["max_prompt_tokens"]
            counts = record["actual_token_counts"]
            maximum, prompt_count = max(maximum, high), prompt_count + len(counts)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record["eligible"]:
                eligible.append(index)
            else:
                excluded.append({key: record[key] for key in
                    ("sample_index", "case_id", "span", "references", "actual_token_counts", "max_prompt_tokens")})
    result = {"artifact_type": ARTIFACT_TYPE, "schema_version": 1,
              "complete": checked == budget, "split": split,
              "sample_budget": budget, "checked_samples": checked,
              "token_limit": UMT5_PROMPT_TOKEN_BUDGET, "contract": dataset.text_admission_contract,
              "eligible_sample_indices": eligible, "excluded_samples": excluded,
              "eligible_count": len(eligible), "excluded_count": len(excluded),
              "compiled_prompt_count": prompt_count, "max_prompt_tokens": maximum,
              "rows_output": str(rows_output), "workers": workers,
              "initialization_seconds": initialization_seconds,
              "production_seconds": time.monotonic() - started}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result
