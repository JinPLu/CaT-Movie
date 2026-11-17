"""Build component-neutral runtime arguments from the experiment contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from mspc.config import ExperimentConfig
from mspc.config.experiment import (
    SHOT_FORWARD_MODE_JOINT,
    SHOT_FORWARD_MODE_PER_SHOT,
    WAN22_BLOCK_DATASET_TYPES,
    WAN22_SHOT_FORWARD_MODES,
)

__all__ = [
    "SHOT_FORWARD_MODE_JOINT",
    "SHOT_FORWARD_MODE_PER_SHOT",
    "WAN22_SHOT_FORWARD_MODES",
    "TrainingArgs",
    "build_common_training_args",
    "build_training_dataset_config",
    "replace_training_args",
    "require_shot_forward_mode",
]


def require_shot_forward_mode(value: object) -> str:
    """一步训练的镜前向形态，二选一，没有第三条路也没有默认。"""

    if value not in WAN22_SHOT_FORWARD_MODES:
        raise ValueError(
            "shot_forward_mode must be one of "
            f"{sorted(WAN22_SHOT_FORWARD_MODES)}; got {value!r}"
        )
    return str(value)


def build_training_dataset_config(config: ExperimentConfig) -> dict[str, object]:
    """Project the validated dataset section for the runtime dataset owner."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    dataset = config.resolved["dataset"]
    if dataset["type"] in WAN22_BLOCK_DATASET_TYPES:
        projected = {
            name: dataset[name]
            for name in (
                "members_path", "sampling_manifest_path", "text_admission_path",
                "segment_capacity", "reference_row_dropout_p",
                "target_mask_weights", "reference_policy", "minimum_layout_shot_fraction",
                "box_dropout_p", "box_jitter", "planner_tokens", "reference_entities", "picture_type_dropout_p",
                "reference_crops", "workload_cap",
            )
        }
        from mspc.data.scene_training_manifest import media_from_task
        projected["media"] = media_from_task(config.resolved["task"])
        projected["target_storage"] = "video" if dataset["type"] == "block_video" else "latents"
        projected["target_index_path"] = dataset["target_index_path"] if dataset["type"] == "block_latent_cache" else None
        projected["target_roots"] = dataset["target_roots"] if dataset["type"] == "block_latent_cache" else None
        projected["target_latent_geometry"] = dataset["target_latent_geometry"] if dataset["type"] == "block_video" else None
        projected["omit_target_text"] = tuple(config.resolved["model"]["omit_target_text"])
        projected["memory_prefix_options"] = config.resolved["model"]["memory_prefix"]
        # 数据集按 DiT 自注意力算每条样本的注意力帧对（负载上限、组批）；不影响抽签与文字。
        projected["self_attention_policy"] = config.resolved["model"]["self_attention_policy"]
        projected["tokenizer_path"] = config.resolved["model"]["paths"]["tokenizer"]
        projected["training_seed"] = config.resolved["training"]["seed"]
        projected["video_decode_num_threads"] = config.resolved["dataloader"]["video_decode_num_threads"]
        return projected
    task = config.resolved["task"]
    return {
        "schema": dataset["schema"],
        "prompt_format": dataset["prompt_format"],
        "metadata_path": dataset["metadata_path"],
        "shot2story_prompt_source": (
            dict(dataset["shot2story_prompt_source"])
            if dataset.get("shot2story_prompt_source") is not None
            else None
        ),
        "repeat": dataset["repeat"],
        "height": task["height"],
        "width": task["width"],
        "frame_policy": task["frame_policy"],
        "max_rgb_frames": task["max_rgb_frames"],
        "target_fps": task["target_fps"],
        "vae_temporal_stride": task["vae_temporal_stride"],
        "video_column": dataset["video_column"],
        "caption_column": dataset["caption_column"],
        "shot_groups_column": dataset["shot_groups_column"],
        "case_id_column": dataset["case_id_column"],
        "case_id_prefix": dataset["case_id_prefix"],
    }


@dataclass(frozen=True)
class TrainingArgs:
    """Frozen runtime arguments shared by all registered training components."""

    dataset_cfg: dict[str, object]
    task: str
    mixed_precision: str
    scheduler_train_timesteps: int | None
    flow_shift: float | int | None
    num_epochs: int
    save_steps: int
    max_train_steps: int | None
    gradient_accumulation_steps: int
    use_gradient_checkpointing: bool
    find_unused_parameters: bool
    learning_rate: float | int
    weight_decay: float | int
    lr_scheduler_type: str
    # ConstantLR keys; None when lr_scheduler_type is constant_with_warmup.
    lr_scheduler_factor: float | int | None
    lr_scheduler_total_iters: int | None
    per_device_batch_size: int
    dataloader_split_batches: bool
    dataloader_drop_last: bool
    dataset_num_workers: int
    dataloader_prefetch_factor: int
    dataloader_persistent_workers: bool
    dataloader_pin_memory: bool
    timing_log_steps: int
    timing_log_every: int
    timing_sync_cuda: bool
    training_seed: int
    output_root: str
    #: 整段联合去噪或逐镜前向；直接来自明确的训练配置。
    shot_forward_mode: str
    # Linear-warmup key; None when lr_scheduler_type is constant.
    warmup_steps: int | None = None
    gpus_per_node: int | None = None
    nnodes: int | None = None
    timing_prefix: str = ""
    remove_prefix_in_ckpt: str = ""
    run_id: str | None = None
    run_root: str | None = None
    #: reserve核实执行根Git身份后广播；保存时必须已赋值，不从环境重读。
    repository_sha: str | None = None
    output_path: str | None = None
    log_dir: str | None = None
    #: ``training.loss_tracking``（SwanLab 展示）与要记成超参的整份解析后配置；null 表示不接外部展示。
    loss_tracking: dict[str, Any] | None = None
    #: ``training.validation``（训练中的留出验证 loss）；只有段数据集的配置有这个键，null 表示不量。
    validation: dict[str, Any] | None = None
    experiment_config: dict[str, Any] | None = None
    progress_stream: Any = None
    model_cfg: dict[str, Any] | None = None
    training_cfg: dict[str, Any] | None = None
    pipeline_cfg: dict[str, Any] | None = None
    checkpoint_path: str | None = None
    #: 显式声明时保存完整训练状态；权重导出与热启动不改变含义。
    state_checkpoint: dict[str, Any] | None = None
    resume_state_path: str | None = None
    resume_migration: dict[str, Any] | None = None
    resume_migration_provenance: dict[str, Any] | None = None
    #: 运行阶段的缓存可用集合；不进入数学配置或完整状态匹配签名。
    cache_stage_plan: dict[str, Any] | None = None


def build_common_training_args(config: ExperimentConfig) -> TrainingArgs:
    """Return arguments shared by all registered training components."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    data = config.resolved
    training = data["training"]
    optimization = data["optimization"]
    dataloader = data["dataloader"]
    timing = data["timing"]
    flow_shift = training["flow_shift"]
    return TrainingArgs(
        dataset_cfg=build_training_dataset_config(config),
        task=training["task"],
        mixed_precision=training["mixed_precision"],
        scheduler_train_timesteps=training["scheduler_train_timesteps"],
        flow_shift=flow_shift,
        num_epochs=training["num_epochs"],
        save_steps=training["save_steps"],
        max_train_steps=training["max_train_steps"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        use_gradient_checkpointing=training["use_gradient_checkpointing"],
        find_unused_parameters=training["find_unused_parameters"],
        learning_rate=optimization["learning_rate"],
        weight_decay=optimization["weight_decay"],
        lr_scheduler_type=optimization["lr_scheduler_type"],
        lr_scheduler_factor=optimization.get("lr_scheduler_factor"),
        lr_scheduler_total_iters=optimization.get("lr_scheduler_total_iters"),
        warmup_steps=optimization.get("warmup_steps"),
        per_device_batch_size=dataloader["per_device_batch_size"],
        dataloader_split_batches=dataloader["split_batches"],
        dataloader_drop_last=dataloader["drop_last"],
        dataset_num_workers=dataloader["dataset_num_workers"],
        dataloader_prefetch_factor=dataloader["prefetch_factor"],
        dataloader_persistent_workers=dataloader["persistent_workers"],
        dataloader_pin_memory=dataloader["pin_memory"],
        loss_tracking=training["loss_tracking"],
        validation=training.get("validation"),
        experiment_config=config.to_dict(),
        state_checkpoint=(dict(training["state_checkpoint"]) if "state_checkpoint" in training else None),
        timing_log_steps=timing["log_steps"],
        timing_log_every=timing["log_every"],
        timing_sync_cuda=timing["sync_cuda"],
        training_seed=training["seed"],
        output_root=data["output"]["root"],
        shot_forward_mode=require_shot_forward_mode(training["shot_forward_mode"]),
    )


def replace_training_args(args: TrainingArgs, **updates: Any) -> TrainingArgs:
    """Return a copy of ``args`` with selected fields replaced."""

    return replace(args, **updates)


def configure_training_topology(args: TrainingArgs, *, gpus_per_node: int, nnodes: int) -> TrainingArgs:
    """确定拓扑后一次性推导样本预算，禁止单独配置出不一致的曝光数。"""
    if any(type(value) is not int or value < 1 for value in (gpus_per_node, nnodes)):
        raise ValueError("gpus_per_node and nnodes must be positive integers")
    dataset = dict(args.dataset_cfg)
    if args.experiment_config["dataset"]["type"] in WAN22_BLOCK_DATASET_TYPES:
        dataset["sample_budget"] = (args.max_train_steps * gpus_per_node * nnodes
                                    * args.per_device_batch_size * args.gradient_accumulation_steps)
    return replace(args, gpus_per_node=gpus_per_node, nnodes=nnodes, dataset_cfg=dataset)
