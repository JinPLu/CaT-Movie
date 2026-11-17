"""离线缓存阶段的运行控制：不改变完整训练配置、数据表或sampler随机流。"""
from __future__ import annotations

from itertools import islice
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
import json
import math
import multiprocessing
import os
from pathlib import Path
from typing import Any

import torch


ARTIFACT_TYPE = "mspc_training_cache_stage"
_VERIFICATION_PANELS = None


@contextmanager
def _serial_tokenizer_calls():
    """只关闭Rust线程池，不改原tokenizer、字符串或计数算法。"""
    previous = os.environ.get("TOKENIZERS_PARALLELISM")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = previous


def _init_verification_worker(panels, tokenizer_path):
    from mspc.data.shot_text import load_umt5_token_counter
    global _VERIFICATION_PANELS
    # fork只共享大只读元数据；每进程独立新建原tokenizer，不使用父进程Rust对象。
    counter = load_umt5_token_counter(Path(tokenizer_path))
    _VERIFICATION_PANELS = {split: panel for split, panel, _ in panels}
    for panel in _VERIFICATION_PANELS.values():
        panel.token_counter = counter


def _verify_text_sample(key):
    split, index = key
    sample = _VERIFICATION_PANELS[split].text(index)
    return {"split": split, "sample_index": index,
            "metadata": sample.metadata, "prompts": sample.prompts}


def _verified_text_rows(panels, *, verification_workers: int, tokenizer_path: str):
    """同一有界ProcessPool供生产与worker1基准使用；保原split/index/行序。"""
    from mspc.data.shot_text import UMT5_PROMPT_TOKEN_BUDGET

    if type(verification_workers) is not int or verification_workers < 1:
        raise ValueError("verification_workers must be an explicit positive integer")
    splits = [split for split, _, _ in panels]
    keys = [(split, index) for split, _, indices in panels for index in indices]
    if (len(splits) != len(set(splits)) or not keys
            or any(split not in {"train", "validation"} or type(index) is not int or index < 0
                   for split, index in keys) or len(keys) != len(set(keys))):
        raise ValueError("verification requires distinct original indices within each explicit split")
    expected, seen = set(keys), set()
    source = iter(enumerate(keys))
    # sampler/stage重放已完成后才限线程，不能影响原torch.randperm调用策略。
    threads = torch.get_num_threads()
    with _serial_tokenizer_calls():
        torch.set_num_threads(1)
        pool = ProcessPoolExecutor(max_workers=verification_workers,
            mp_context=multiprocessing.get_context("fork"),
            initializer=_init_verification_worker, initargs=(panels, tokenizer_path))
        pending, ready = {}, {}
        next_position = 0

        def submit_next():
            item = next(source, None)
            if item is not None:
                position, key = item
                pending[pool.submit(_verify_text_sample, key)] = (position, key)

        try:
            for _ in range(2 * verification_workers):
                submit_next()
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                # 先读所有已完成异常，不因前面较慢的原index延迟错误传播。
                for future in finished:
                    error = future.exception()
                    if error is not None:
                        raise error
                for future in finished:
                    position, key = pending.pop(future)
                    row = future.result()
                    actual = (row["split"], row["sample_index"])
                    if actual != key or actual in seen or row["metadata"]["sample_index"] != key[1]:
                        raise ValueError(f"verification worker changed or duplicated original sample identity: {key}")
                    counts, prompts = row["metadata"]["actual_token_counts"], row["prompts"]
                    if (not isinstance(counts, (list, tuple)) or not counts or len(counts) != len(prompts)
                            or any(type(count) is not int or count < 1 for count in counts)
                            or any(not isinstance(prompt, str) for prompt in prompts)):
                        raise ValueError(f"verification worker returned invalid actual prompt/token rows: {key}")
                    maximum = max(counts)
                    if maximum > UMT5_PROMPT_TOKEN_BUDGET:
                        raise ValueError(f"{key[0]} sample {key[1]}, case {row['metadata']['case_id']}: "
                            f"compiled text has {maximum} umT5 tokens, limit {UMT5_PROMPT_TOKEN_BUDGET}")
                    seen.add(actual)
                    ready[position] = row
                while next_position in ready:
                    yield ready.pop(next_position)
                    next_position += 1
                    submit_next()
            if ready or seen != expected or next_position != len(keys):
                raise ValueError("verification did not cover every original split/sample index exactly once")
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
        finally:
            torch.set_num_threads(threads)


def read_cache_stage(path: str | Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text())
    if (not isinstance(plan, dict) or plan.get("artifact_type") != ARTIFACT_TYPE
            or plan.get("schema_version") != 1):
        raise ValueError("cache stage requires an explicit training cache stage artifact")
    indices = plan.get("sample_indices")
    if (not isinstance(indices, list) or not indices
            or any(type(index) is not int or index < 0 for index in indices)
            or indices != sorted(set(indices))):
        raise ValueError("cache stage sample_indices must be nonempty distinct sorted dataset indices")
    return plan


def cache_stage_plan(dataset: Any, args: Any, *, world_size: int, target_step: int) -> dict[str, Any]:
    """用现行sampler与Accelerate批分片重放，包含有界在飞预取但不读取样本。

    独立CPU generator具有与正式sampler相同的种子；不会消费训练RNG或
    训练sampler的generator。完整96k表和每个样本的draw键保持原值。
    """
    from accelerate.data_loader import BatchSamplerShard
    from mspc.data.shot_targets import shot_key
    from mspc.training.loop import data_order_generator, training_sampler, training_epoch_count

    if (type(world_size) is not int or world_size < 1 or type(target_step) is not int
            or target_step < 1 or type(args.max_train_steps) is not int
            or target_step > args.max_train_steps):
        raise ValueError("cache stage needs a positive world size and a target within the full training plan")
    if args.num_epochs != 1 or args.dataloader_split_batches is not False or args.per_device_batch_size != 1:
        raise ValueError("cache stages support the current one-epoch, unsplit, one-complete-sample loader")
    if not hasattr(dataset, "sampling_manifest") or len(dataset.sampling_manifest) != len(dataset):
        raise ValueError("cache stage requires the complete frozen sampling manifest")
    accumulation = args.gradient_accumulation_steps
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError("cache stage requires positive gradient accumulation")
    workers = args.dataset_num_workers
    if type(workers) is not int or workers < 0:
        raise ValueError("cache stage requires nonnegative loader workers")
    prefetch = args.dataloader_prefetch_factor
    if workers and (type(prefetch) is not int or prefetch < 1):
        raise ValueError("cache stage requires positive worker prefetch")
    # DataLoaderShard fetches next before yielding current; PyTorch has at most
    # workers*prefetch outstanding index tasks, including after a stop request.
    ahead = workers * prefetch + 1 if workers else 1
    generator = data_order_generator(args)
    sampler = training_sampler(dataset, generator=generator,
        bucket_size=world_size * accumulation, experiment_config=args.experiment_config)
    batches = torch.utils.data.BatchSampler(sampler, batch_size=1, drop_last=args.dataloader_drop_last)
    per_rank = BatchSamplerShard(batches, num_processes=world_size, process_index=0,
                                split_batches=False, even_batches=True)
    epochs = training_epoch_count(args, dataset, updates_per_epoch=math.ceil(len(per_rank) / accumulation))
    ranks = [[] for _ in range(world_size)]
    remaining_updates = target_step
    for _ in range(epochs):
        # 同一epoch排列只生成一次，再按Accelerate原规则分rank；下一epoch继续同一generator。
        ordered = list(sampler)
        batches = torch.utils.data.BatchSampler(ordered, batch_size=1, drop_last=args.dataloader_drop_last)
        train_batches = min(len(per_rank), remaining_updates * accumulation)
        for rank in range(world_size):
            shard = BatchSamplerShard(batches, num_processes=world_size, process_index=rank,
                                      split_batches=False, even_batches=True)
            ranks[rank].extend(index for batch in islice(iter(shard), train_batches + ahead)
                               for index in batch)
        remaining_updates -= math.ceil(train_batches / accumulation)
        if remaining_updates == 0:
            break
    if remaining_updates:
        raise ValueError("cache stage target exceeds this complete loader's optimizer updates")
    indices = sorted({index for row in ranks for index in row})
    targets = set()
    for index in indices:
        sample = dataset.sampling_manifest[index]
        member = dataset.members[sample["case_id"]]
        for position in range(*sample["span"]):
            targets.add(shot_key(member["shots"][position]))
    validation_budget = 0 if args.validation is None else args.validation["sample_budget"]
    validation_indices = list(range(validation_budget))
    if args.validation is not None and args.validation["text_admission_path"] is not None:
        from mspc.training.text_admission import read_text_admission
        validation_indices = read_text_admission(args.validation["text_admission_path"],
            sample_budget=validation_budget)["eligible_sample_indices"]
    return {
        "artifact_type": ARTIFACT_TYPE, "schema_version": 1,
        "sample_budget": len(dataset), "planned_optimizer_updates": args.max_train_steps,
        "target_step": target_step, "training_seed": args.training_seed,
        "media": dict(dataset.media), "sampler": type(sampler).__name__,
        "latent_geometry": {"channels": dataset.latent_channels,
                            "spatial_shape": list(dataset.latent_spatial_shape),
                            "vae_spatial_stride": dataset.vae_spatial_stride},
        "workload_cost": args.experiment_config["training"].get("workload_cost"),
        "loader": {"world_size": world_size, "per_device_batch_size": 1,
                   "gradient_accumulation_steps": accumulation, "split_batches": False,
                   "drop_last": args.dataloader_drop_last, "even_batches": True,
                   "num_workers": workers, "prefetch_factor": prefetch,
                   "prefetch_batches_per_rank": ahead},
        "rank_sample_indices": ranks, "sample_indices": indices,
        "target_keys": [list(key) for key in sorted(targets)],
        "validation_sample_indices": validation_indices,
    }


def require_cache_stage(dataset: Any, args: Any, *, world_size: int) -> None:
    """模型构建前复核运行计划，不能用旧前缀或缩短数据集替代实际采样顺序。"""
    plan = args.cache_stage_plan
    if plan is None:
        return
    if dataset.target_storage != "latents":
        raise ValueError("operational cache stages require production latent targets, never online video fallback")
    expected = cache_stage_plan(dataset, args, world_size=world_size, target_step=plan.get("target_step"))
    if plan != expected:
        raise ValueError("cache stage differs from the complete actual sampler, loader, media or validation plan")
    if args.resume_state_path is not None:
        from mspc.training.training_state import inspect_training_state
        from mspc.training.resume_migration import resume_source_config
        state = inspect_training_state(args.resume_state_path, config=resume_source_config(args), world_size=world_size)
        if plan["target_step"] <= state["completed_steps"]:
            raise ValueError("cache stage target was already reached by the resumed training state")
    if plan["target_step"] < args.max_train_steps and args.state_checkpoint is None:
        raise ValueError("an intermediate cache stage requires complete training state checkpoints")
    if args.validation is not None:
        # Validation is a separate complete deterministic set, never the train
        # prefix. Validate its real cache records before constructing the model.
        from mspc.training.block_dataset import build_validation_dataset
        build_validation_dataset(args)


def metadata_cache_stage(config: Any, *, world_size: int, target_step: int,
                         latent_channels: int, vae_spatial_stride: int) -> dict[str, Any]:
    """仅为阶段规划复用block_video元数据能力；生产训练仍严格读取latent。

    不加载VAE、不取视频帧，VAE通道数与空间stride由调用方按真实缓存几何显式给出。
    """
    from mspc.core.wan22_contract import MEMORY_ROW_POLICY_FIXED_ROLE_V1
    from mspc.training.block_dataset import BlockDataset
    from mspc.training.training_args import build_common_training_args, configure_training_topology

    if config.resolved["dataset"]["type"] != "block_latent_cache":
        raise ValueError("cache stage planning requires a production block_latent_cache configuration")
    if (type(latent_channels) is not int or latent_channels < 1
            or type(vae_spatial_stride) is not int or vae_spatial_stride < 1):
        raise ValueError("cache stage planning requires explicit positive VAE channels and spatial stride")
    args = build_common_training_args(config)
    args = configure_training_topology(args, gpus_per_node=world_size, nnodes=1)
    projected = {**args.dataset_cfg, "target_storage": "video", "target_index_path": None,
                 "target_roots": None, "target_mask_weights": None, "target_latent_geometry": {
                     "channels": latent_channels,
                     "spatial_stride": vae_spatial_stride}}
    # 身份行在缓存规划之后生产；它们只影响文字，不参与参考选帧或组批。
    # 只投影文字合同，保留实际reference_policy、dropout和全部memory几何。
    if projected["text_admission_path"] is None:
        projected["reference_entities"] = None
        if projected["memory_prefix_options"] is not None:
            projected["memory_prefix_options"] = {**projected["memory_prefix_options"],
                                                   "row_policy": MEMORY_ROW_POLICY_FIXED_ROLE_V1}
    dataset = BlockDataset(**projected)
    return cache_stage_plan(dataset, args, world_size=world_size, target_step=target_step)


def verify_config_cache_stage(config: Any, path: str | Path, *, world_size: int,
                              verification_workers: int) -> dict[str, Any]:
    """申请训练GPU前核真实缓存和实际编译文字；不读取RGB或构造模型。"""
    from mspc.training.block_dataset import build_training_dataset, build_validation_dataset
    from mspc.training.training_args import build_common_training_args, configure_training_topology, replace_training_args

    if config.resolved["dataset"]["type"] != "block_latent_cache":
        raise ValueError("cache stage verification requires production latent targets")
    if type(verification_workers) is not int or verification_workers < 1:
        raise ValueError("verification_workers must be an explicit positive integer")
    args = configure_training_topology(build_common_training_args(config), gpus_per_node=world_size, nnodes=1)
    args = replace_training_args(args, cache_stage_plan=read_cache_stage(path))
    with _serial_tokenizer_calls():
        dataset = build_training_dataset(args)
        require_cache_stage(dataset, args, world_size=world_size)
        prompt_rows, max_tokens = 0, 0
        panels = [("train", dataset, args.cache_stage_plan["sample_indices"])]
        if args.validation is not None:
            panels.append(("validation", build_validation_dataset(args),
                           args.cache_stage_plan["validation_sample_indices"]))
        for row in _verified_text_rows(panels, verification_workers=verification_workers,
                                       tokenizer_path=args.dataset_cfg["tokenizer_path"]):
            counts = row["metadata"]["actual_token_counts"]
            prompt_rows += len(counts)
            max_tokens = max(max_tokens, max(counts))
    return {"target_step": args.cache_stage_plan["target_step"],
            "ready_samples": len(args.cache_stage_plan["sample_indices"]),
            "ready_target_shots": len(dataset.targets),
            "validation_samples": len(args.cache_stage_plan["validation_sample_indices"]),
            "compiled_prompt_rows": prompt_rows, "max_prompt_tokens": max_tokens,
            "verification_workers": verification_workers,
            "verified_samples_by_split": {split: len(indices) for split, _, indices in panels}}
