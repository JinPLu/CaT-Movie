"""Strict DiffSynth Wan2.2 pure-T2V adapter for unified MSPC SFT training.

This owner intentionally contains no local Wan scheduler, anchor, or Shot-RPE
implementation.  It loads the local materialized DiffSynth runtime, narrows it
to the P3 pure-T2V surface, and leaves every schedule quantity (sigmas,
``add_noise``, ``training_target``, ``training_weight``) with upstream
``pipe.scheduler``.

The SFT loss itself is local: upstream ``FlowMatchSFTLoss`` seals noise,
timestep, target and MSE inside one function that returns a single scalar, so the
drawn timestep and the reference-frame prefix have no seam to be read or appended
to and the whole function was replaced.  See :func:`flow_match_sft_loss`.
"""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Mapping

import torch
from torch import nn

from mspc.config import ExperimentConfig
from mspc.training.checkpoint import TrainingCheckpoint
from mspc.training.dit_parameter_freeze import (
    apply_named_dit_freeze,
    require_dit_trainable,
    load_dit_state_dict,
    require_dit_parameter_freeze,
)
from mspc.training.training_args import (
    TrainingArgs,
    replace_training_args,
)
from mspc.models.memory_prefix_options import MemoryPrefixModelOptions
from mspc.models.wan22_loader import (
    WAN22_COMPONENT_MODE_CACHE_ENCODE,
    WAN22_COMPONENT_MODE_CACHE_ENCODE_LATENTS,
    WAN22_COMPONENT_MODE_CACHED_TRAINING,
    WAN22_COMPONENT_MODE_CACHED_TRAINING_LIVE_T5,
    load_pure_t2v_pipeline,
    require_component_mode,
    require_path,
    require_shot_prompts,
    require_temporal_position_mode,
)
from mspc.core.wan22_contract import (
    ROUTED_MEMORY_ATTENTION_POLICIES,
    WAN22_CACHE_LAYOUT_LATENTS_ONLY,
    WAN22_CACHE_LAYOUTS,
    WAN22_FLOW_MATCH_SHIFT,
    require_memory_rows,
)
from mspc.data.reference_routing import require_reference_routing


class _CachedTextContextResolver:
    """完整原样本映射常驻；context仅在当前step读取，不保留tensor memo。"""

    def __init__(self, manifest_path: str, model_cfg: Mapping[str, Any]) -> None:
        from mspc.training.text_cache import (
            CONTEXT_DTYPE, CONTEXT_SHAPE, CONTEXT_STORAGE, CONTEXT_STORAGE_CONTRACT,
            TEXT_CACHE_ARTIFACT_TYPE,
        )

        self.path = Path(manifest_path)
        self.manifest = json.loads(self.path.read_text())
        manifest = self.manifest
        if (manifest.get("artifact_type") != TEXT_CACHE_ARTIFACT_TYPE
                or manifest.get("schema_version") != 2
                or manifest.get("context_storage") != CONTEXT_STORAGE
                or manifest.get("storage_contract") != CONTEXT_STORAGE_CONTRACT
                or manifest.get("complete_stage") is not True):
            raise ValueError("training requires a complete compact text context cache")
        source_path = Path(manifest["source_manifest"])
        self.source = json.loads(source_path.read_text())
        source = self.source
        if (source.get("artifact_type") != TEXT_CACHE_ARTIFACT_TYPE
                or source.get("schema_version") != 1 or source.get("complete") is not True
                or source.get("sample_limit_per_split") is not None):
            raise ValueError("text context cache requires the complete original prompt manifest")
        expected_encoding = {
            "t5_path": str(Path(model_cfg["paths"]["t5"]).resolve()),
            "tokenizer_path": str(Path(model_cfg["paths"]["tokenizer"]).resolve()),
            "owner": "WanVideoUnit_PromptEmbedder.encode_prompt",
            "call_policy": "one original prompt string per tokenizer and T5 call",
            "add_special_tokens": True, "padding": "max_length", "max_length": 512,
            "clean": "whitespace", "mask": "original tokenizer attention_mask",
            "masked_output": "set all tokens at or beyond the actual mask length to zero",
            "context_shape_per_unique_prompt": list(CONTEXT_SHAPE), "context_dtype": CONTEXT_DTYPE,
        }
        if (source["encoding_contract"] != expected_encoding
                or manifest["encoding_contract"] != expected_encoding):
            raise ValueError("text context cache differs from the original T5 encoding contract")
        # 缓存覆盖到本次停点阶段即可（与目标缓存同一口径）：阶段计划与本次起训的逐项相同由
        # require_training_contract 核；到点由 runner 发定点停止请求、存完整状态后停，续训前补编到新停点。
        stage = source["stage"]
        expected_indices = {"train": stage["sample_indices"],
                            "validation": stage["validation_sample_indices"]}
        if set(source["splits"]) != {name for name, indices in expected_indices.items() if indices}:
            raise ValueError("text context split coverage differs from the complete stage")
        for split, report in source["splits"].items():
            expected = len(expected_indices[split])
            if (report["complete"] is not True or report["expected_samples"] != expected
                    or report["produced_samples"] != expected):
                raise ValueError(f"text context cache has incomplete {split} coverage")

        # 配对流式读完整清单；保留字符串和轻量文件offset，不加载任何context。
        self.prompts: list[tuple[str, int]] = []
        seen_strings: set[str] = set()
        for _, row in self._matching_rows(self.path.parent / "prompts.jsonl",
                                          source_path.parent / source["prompt_manifest"]):
            text, count = row["prompt"], row["token_count"]
            if (type(row["row_id"]) is not int or row["row_id"] != len(self.prompts)
                    or not isinstance(text, str) or text in seen_strings
                    or type(count) is not int or not 1 <= count <= 512):
                raise ValueError("text context prompt rows require exact first-use consecutive identities")
            seen_strings.add(text)
            self.prompts.append((text, count))
        del seen_strings
        if (len(self.prompts) != source["unique_prompts"]
                or manifest["encoded_prompts"] != len(self.prompts)
                or manifest["write_read_verified_prompts"] != len(self.prompts)):
            raise ValueError("text context cache does not cover every original prompt row")
        self.sample_path = self.path.parent / "sample-context-rows.jsonl"
        self.samples: dict[tuple[str, int], tuple[int, tuple[int, ...]]] = {}
        observed = {split: [] for split in source["splits"]}
        used_rows: set[int] = set()
        total_rows = 0
        for offset, row in self._matching_rows(self.sample_path,
                                               source_path.parent / source["sample_manifest"]):
            split, index, rows = row["split"], row["sample_index"], row["context_rows"]
            if (split not in observed or type(index) is not int or (split, index) in self.samples
                    or not isinstance(rows, list) or not rows
                    or any(type(value) is not int or not 0 <= value < len(self.prompts) for value in rows)
                    or row["metadata"]["sample_index"] != index
                    or row["metadata"]["actual_token_counts"] != [self.prompts[value][1] for value in rows]):
                raise ValueError("text context sample mapping is invalid")
            self.samples[(split, index)] = (offset, tuple(rows))
            observed[split].append(index)
            used_rows.update(rows)
            total_rows += len(rows)
        if (any(observed[split] != expected_indices[split] for split in observed)
                or manifest["encoded_samples"] != len(self.samples)
                or used_rows != set(range(len(self.prompts)))
                or total_rows != source["total_prompt_rows"]):
            raise ValueError("text context samples differ from the complete original stage and row order")
        # 文件清单不触碰payload；单条shape/dtype/ids/mask由公共reader在首次消费核验。
        if {path.name for path in (self.path.parent / "contexts").iterdir()} != {
                f"{index}.safetensors" for index in range(len(self.prompts))}:
            raise ValueError("text context cache files do not cover the complete prompt manifest")

    @staticmethod
    def _matching_rows(path: Path, source: Path):
        with path.open("rb") as actual, source.open("rb") as original:
            while True:
                offset = actual.tell()
                line, expected = actual.readline(), original.readline()
                if line != expected:
                    raise ValueError(f"text context rows differ from the original manifest: {path}")
                if not line:
                    break
                yield offset, json.loads(line)

    def require_training_contract(self, args: TrainingArgs) -> None:
        from mspc.training.text_admission import text_admission_contract

        if self.source["stage"] != args.cache_stage_plan:
            raise ValueError("text context cache must match the actual complete sampler stage")
        actual_contracts = {"train": text_admission_contract(args.dataset_cfg)}
        if args.validation is not None:
            values = {**args.dataset_cfg, **{key: args.validation[key] for key in
                      ("members_path", "sampling_manifest_path", "sample_budget")}}
            if values["reference_entities"] is not None:
                values["reference_entities"] = args.validation["reference_entities"]
            actual_contracts["validation"] = text_admission_contract(values)
        if self.source["contracts"] != actual_contracts:
            raise ValueError("text context cache differs from the actual sample/text draw contracts")
        autocast = args.experiment_config["runtime"]["deepspeed"]["torch_autocast"]
        if self.manifest["outer_autocast"] != {"enabled": autocast["enabled"], "dtype": autocast["dtype"]}:
            raise ValueError("text context cache differs from the actual outer autocast contract")

    def sample_context_rows(self, stream, *, split: str, metadata: Mapping[str, Any],
                            prompts: Sequence[str]) -> tuple[int, ...]:
        key = (split, metadata["sample_index"])
        if key not in self.samples:
            raise ValueError(f"text context cache misses original sample {key}")
        offset, rows = self.samples[key]
        stream.seek(offset)
        sample = json.loads(stream.readline())
        if (sample["split"], sample["sample_index"], tuple(sample["context_rows"])) != (*key, rows):
            raise ValueError("text context sample mapping changed after initialization")
        if any(name not in metadata or metadata[name] != value for name, value in sample["metadata"].items()):
            raise ValueError(f"text context sample metadata differs from the original draw: {key}")
        if not isinstance(prompts, (list, tuple)) or list(prompts) != [self.prompts[index][0] for index in rows]:
            raise ValueError(f"text context actual prompt strings or row order differ: {key}")
        return rows

    def resolve(self, data: Mapping[str, Any], *, device: torch.device) -> torch.Tensor:
        from mspc.training.text_cache import read_text_context

        identity = data["_text_cache_sample"]
        split, metadata = identity["split"], identity["metadata"]
        with self.sample_path.open("rb") as stream:
            rows = self.sample_context_rows(stream, split=split, metadata=metadata, prompts=data["prompt"])
        shot_groups = data.get("shot_groups")
        group_count = 1 if shot_groups is None else len(shot_groups)
        if group_count != len(rows) or (split == "train" and data["flow_draw_seed"] != metadata["flow_draw_seed"]):
            raise ValueError(f"text context actual prompt strings or row order differ: {(split, metadata['sample_index'])}")
        tensors = []
        for index in rows:
            value = read_text_context(self.path.parent / "contexts" / f"{index}.safetensors")
            if int(value["mask"].sum()) != self.prompts[index][1]:
                raise ValueError(f"text context token count differs from original prompt row {index}")
            tensors.append(value["context"])
        return torch.cat(tensors, dim=0).to(device=device, dtype=torch.bfloat16)


def _tensor_to_pil_frames(video: torch.Tensor) -> list[Any]:
    """Map the dataset's ``[C,F,H,W]`` normalized sample to upstream inputs."""

    if not isinstance(video, torch.Tensor) or video.dim() != 4:
        raise ValueError("training data.video must be a [C,F,H,W] tensor")
    if video.shape[0] != 3 or video.shape[1] <= 0:
        raise ValueError(f"training data.video must have three non-empty RGB channels, got {tuple(video.shape)}")
    if not video.is_floating_point():
        raise ValueError("training data.video must be a floating-point tensor normalized to [-1, 1]")
    if not torch.isfinite(video).all():
        raise ValueError("training data.video contains non-finite values")
    if float(video.min()) < -1.00001 or float(video.max()) > 1.00001:
        raise ValueError("training data.video must be normalized to [-1, 1]")
    from PIL import Image

    rgb = ((video.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).round()
    rgb = rgb.to(dtype=torch.uint8).permute(1, 2, 3, 0).contiguous().numpy()
    return [Image.fromarray(frame) for frame in rgb]


def _uint8_tensor_to_pil_frames(video: torch.Tensor) -> list[Any]:
    """Map cache-only uint8 transport directly to upstream PIL inputs."""

    if not isinstance(video, torch.Tensor) or video.dim() != 4:
        raise ValueError("cache transport data.video must be a [C,F,H,W] tensor")
    if video.dtype != torch.uint8:
        raise ValueError("cache transport data.video must use torch.uint8")
    if video.shape[0] != 3 or video.shape[1] <= 0:
        raise ValueError(
            "cache transport data.video must have three non-empty RGB channels, "
            f"got {tuple(video.shape)}"
        )
    from PIL import Image

    rgb = video.detach().cpu().permute(1, 2, 3, 0).contiguous().numpy()
    return [Image.fromarray(frame) for frame in rgb]


def _require_group_sequence(value: object, label: str) -> list[Any] | tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty sequence of [start, end] groups")
    return value


def _require_wan_flow_match_shift(scheduler: Any, *, num_train_timesteps: int) -> None:
    """Assert the training schedule matches Wan FlowMatch with shift=5.0.

    Current DiffSynth no longer stores ``scheduler.shift`` as an instance
    attribute. ``set_timesteps`` injects ``WAN22_FLOW_MATCH_SHIFT`` when the
    caller omits ``shift``; ``set_timesteps_wan`` itself refuses a missing
    shift. Validate the realized sigma curve against that single constant.
    """
    if num_train_timesteps <= 0:
        raise ValueError("num_train_timesteps must be positive")
    attribute_shift = getattr(scheduler, "shift", None)
    if attribute_shift is not None:
        if float(attribute_shift) != WAN22_FLOW_MATCH_SHIFT:
            raise RuntimeError(
                "Wan2.2 requires upstream FlowMatchScheduler shift="
                f"{WAN22_FLOW_MATCH_SHIFT}, got {attribute_shift!r}"
            )
        return
    actual_sigmas = getattr(scheduler, "sigmas", None)
    if not isinstance(actual_sigmas, torch.Tensor) or actual_sigmas.numel() == 0:
        raise RuntimeError(
            "Wan2.2 FlowMatchScheduler did not materialize training sigmas after set_timesteps"
        )
    set_wan = getattr(type(scheduler), "set_timesteps_wan", None)
    if not callable(set_wan):
        raise RuntimeError(
            "Wan2.2 requires upstream FlowMatchScheduler shift="
            f"{WAN22_FLOW_MATCH_SHIFT}, got None"
        )
    expected_sigmas, _ = set_wan(
        num_inference_steps=num_train_timesteps,
        denoising_strength=1.0,
        shift=WAN22_FLOW_MATCH_SHIFT,
    )
    if not torch.allclose(
        actual_sigmas.detach().cpu().float(),
        expected_sigmas.detach().cpu().float(),
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError(
            "Wan2.2 training sigmas diverge from FlowMatchScheduler Wan "
            f"shift={WAN22_FLOW_MATCH_SHIFT} reference curve"
        )


def flow_match_sft_loss(
    pipe: Any,
    *,
    reference_noise_sigma_max: float,
    drawn: dict[str, Any] | None = None,
    profile_sync_cuda: bool = False,
    **inputs: Any,
) -> torch.Tensor:
    """本地 flow-matching SFT loss：上游 ``FlowMatchSFTLoss`` 的同形替换。

    ``drawn`` 给一个字典时，这一步抽到的时间步写进 ``drawn["timestep"]``（调度器的时间步取值，0–1000）：逐样本 loss
    要按时间步分开看才看得出收敛。不消耗随机数、不改求值顺序。

    时间与目标噪声由显式样本种子、独立CPU generator抽取，再转移到设备；新增模块初始化、
    参考张数和不同GPU架构均不消费这条随机流。有参考时只为目标段抽噪声。上游的 ``lora`` /
    ``noise_scale`` 两条分支在 P3 的输入面上取不到键，这里不复制。

    ``inputs["prefix_frames"]`` 是**前多少张潜帧是条件前缀**——上游那条
    ``first_frame_latents`` 分支（把第 0 帧粘回来、再把第 0 帧从预测与目标里切掉）的
    N 帧推广，也是 TI2V-5B 原生 I2V 口的一般形态。前 M 帧不进 loss，
    按独立样本随机域抽连续参考σ与ε，token时间同步为1000σ。
    σmax=0直接保留旧干净前缀，不抽参考ε；目标t/ε及调用顺序原样保留。
    这个数字由 ``_prepare_latent_inputs`` 显式写进 ``inputs``
    （非参考帧臂上是 0），不在这里给默认值——默认 0 会让一条真带参考帧的臂静默把参考帧当
    噪声去训，而 loss 曲线上看不出来。
    """

    # 只消费目标网格权重，不把监督信息交给模型条件通道。
    target_loss_weights = inputs.pop("target_loss_weights", None)
    draw_seed = inputs.pop("flow_draw_seed")
    if type(draw_seed) is not int or not 0 <= draw_seed < 2**63:
        raise ValueError("flow_draw_seed must be the explicit nonnegative sample seed below 2**63")
    time_generator = torch.Generator(device="cpu").manual_seed(draw_seed)
    noise_generator = torch.Generator(device="cpu").manual_seed(draw_seed ^ 0x5DEECE66D)
    scheduler = pipe.scheduler
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(scheduler.timesteps))
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,), generator=time_generator)
    timestep = scheduler.timesteps[timestep_id].to(dtype=torch.float32, device=pipe.device)
    if drawn is not None:
        drawn["timestep"] = float(timestep)
        drawn["flow_draw_seed"] = draw_seed

    prefix_frames = inputs["prefix_frames"]
    if type(prefix_frames) is not int:
        raise TypeError(
            "inputs.prefix_frames must be an int; got "
            f"{type(prefix_frames).__name__}"
        )
    input_latents = inputs["input_latents"]
    if not 0 <= prefix_frames < input_latents.shape[2]:
        raise ValueError(
            f"prefix_frames must be in [0, {input_latents.shape[2]}); "
            f"got {prefix_frames}"
        )
    memory_latents = input_latents[:, :, :prefix_frames]
    target_latents = input_latents[:, :, prefix_frames:]
    noise = torch.randn(target_latents.shape, dtype=torch.float32, device="cpu", generator=noise_generator).to(
        device=target_latents.device, dtype=target_latents.dtype)
    if drawn is not None:
        count = min(32, noise.numel())
        probes = torch.arange(count, dtype=torch.int64, device=noise.device) * (noise.numel() - 1) // max(1, count - 1)
        drawn["target_noise_probe"] = noise.detach().reshape(-1)[probes].float().cpu().tolist()
    noisy_target = scheduler.add_noise(target_latents, noise, timestep)
    from mspc.models.reference_noise import noise_reference_latents, reference_model_timestep, sample_reference_sigma
    reference_sigma = sample_reference_sigma(sigma_max=reference_noise_sigma_max, seed=draw_seed) if prefix_frames else 0.0
    memory_latents = noise_reference_latents(memory_latents, sigma=reference_sigma, seed=draw_seed)
    if prefix_frames:
        inputs["reference_timestep"] = reference_model_timestep(reference_sigma)
    inputs["latents"] = torch.cat([memory_latents, noisy_target], dim=2)
    training_target = scheduler.training_target(target_latents, noise, timestep)
    if drawn is not None:
        drawn["reference_noise_sigma"] = reference_sigma
        drawn["reference_timestep"] = reference_model_timestep(reference_sigma)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    if profile_sync_cuda:
        torch.cuda.synchronize()
    started = time.perf_counter()
    output = pipe.model_fn(**models, **inputs, timestep=timestep)
    prediction = output.noise
    if profile_sync_cuda:
        torch.cuda.synchronize()
    if drawn is not None:
        drawn["dit_forward_seconds"] = time.perf_counter() - started

    def _denoised(video_prediction: torch.Tensor) -> torch.Tensor:
        return video_prediction[:, :, prefix_frames:]

    from mspc.training.target_loss import target_flow_mse
    loss, uniform_loss = target_flow_mse(_denoised(prediction), training_target, target_loss_weights)
    weight = scheduler.training_weight(timestep)
    if drawn is not None:
        with torch.no_grad():
            frame_mse = (_denoised(prediction).detach().float() - training_target.float()).square().mean((0, 1, 3, 4))
            drawn["target_frame_mse"] = frame_mse.cpu().tolist()
            drawn["unweighted_mse"] = float(uniform_loss.detach())
            drawn["target_weighted_mse"] = float(loss.detach())
            drawn["uniform_loss"] = float((uniform_loss * weight).detach())
            drawn["loss_weight"] = float(weight)
    total = loss * weight
    if output.source_alignment_loss is not None:
        options = pipe.dit.source_identity_options
        if options is None:
            raise ValueError("source alignment loss requires the declared source identity model")
        total = total + options.loss_weight * output.source_alignment_loss
        if drawn is not None:
            drawn["source_alignment_loss"] = float(output.source_alignment_loss.detach())
            drawn["weighted_source_alignment_loss"] = float((options.loss_weight * output.source_alignment_loss).detach())
    return total


def _memory_prefix_options(
    model_cfg: Mapping[str, Any],
) -> MemoryPrefixModelOptions | None:
    raw = model_cfg.get("memory_prefix")
    if raw is None:
        return None
    options = MemoryPrefixModelOptions.from_mapping(raw)
    if not options.attention_window.full:
        raise ValueError("training reads references at every layer and timestep; "
                         "model.memory_prefix.attention_window must be the full window")
    return options


def configure_args(
    args: TrainingArgs,
    config: ExperimentConfig,
    checkpoint: TrainingCheckpoint | None,
) -> TrainingArgs:
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    if not isinstance(args, TrainingArgs):
        raise TypeError("args must be a TrainingArgs")
    data = config.resolved
    model = data["model"]
    training = data["training"]
    if model["type"] != "wan22_diffsynth":
        raise ValueError("config model.type must be 'wan22_diffsynth'")
    require_temporal_position_mode(model["temporal_position_mode"])
    if training["task"] != "sft":
        raise ValueError("Wan2.2 P3 supports only training.task='sft'")
    if training["scheduler_train_timesteps"] != 1000:
        raise ValueError("Wan2.2 P3 requires the upstream 1000-step FlowMatch scheduler")
    flow_shift = training["flow_shift"]
    if flow_shift != WAN22_FLOW_MATCH_SHIFT:
        raise ValueError(f"Wan2.2 requires FlowMatch shift={WAN22_FLOW_MATCH_SHIFT}")
    if training["find_unused_parameters"]:
        raise ValueError("Wan2.2 P3 requires find_unused_parameters=false")
    pipeline = data["pipeline_inputs"]
    if not isinstance(pipeline, Mapping):
        raise ValueError("Wan2.2 requires explicit pipeline_inputs")

    training_cfg = {
        key: training[key]
        for key in (
            "scheduler_train_timesteps",
            "use_gradient_checkpointing",
            "activation_checkpointing",
            "dit_parameter_freeze",
        )
    }
    training_cfg["flow_shift"] = flow_shift
    training_cfg["self_attention_cache_layers"] = training.get("self_attention_cache_layers")
    training_cfg["no_recompute_suffix_layers"] = training.get("no_recompute_suffix_layers")
    return replace_training_args(
        args,
        timing_prefix="[wan22_diffsynth_step]",
        model_cfg=dict(model),
        training_cfg=training_cfg,
        pipeline_cfg=dict(pipeline),
        remove_prefix_in_ckpt="dit.",
        checkpoint_path=None if checkpoint is None else str(checkpoint.path),
    )


class Wan22DiffSynthTrainingModule(nn.Module):
    """One registered full-DiT module backed by a non-registered upstream pipe."""

    #: ``model.memory_prefix: null`` 就是「这条臂不存在」，所以 ``None`` 是它的本义取值。
    #: ``__init__`` 总会覆盖它。
    memory_prefix_options: MemoryPrefixModelOptions | None = None
    spatial_layout_options = None
    planner_token_options = None
    first_frame_token_options = None
    source_identity_options = None
    _text_context_resolver: _CachedTextContextResolver | None = None

    def __init__(
        self,
        *,
        model_cfg: Mapping[str, Any],
        training_cfg: Mapping[str, Any],
        pipeline_cfg: Mapping[str, Any],
        device: torch.device,
        component_mode: str,
    ) -> None:
        super().__init__()
        self.model_cfg = dict(model_cfg)
        self.training_cfg = dict(training_cfg)
        self.pipeline_cfg = dict(pipeline_cfg)
        self.device = device
        self.component_mode = require_component_mode(component_mode)
        self.dtype = torch.bfloat16
        self._text_context_resolver = (
            _CachedTextContextResolver(self.model_cfg["text_context_cache"], self.model_cfg)
            if self.component_mode == WAN22_COMPONENT_MODE_CACHED_TRAINING else None
        )
        self.temporal_position_mode = require_temporal_position_mode(
            self.model_cfg["temporal_position_mode"]
        )
        # 上游 FlowMatchSFTLoss 不再被调用：改动点 I 把整个 loss 换成本地实现。
        pipe, shape_checker_type = load_pure_t2v_pipeline(
            diffsynth_root=self.model_cfg["diffsynth_root"],
            model_paths=self.model_cfg["paths"],
            self_attention_policy=self.model_cfg["self_attention_policy"],
            torch_dtype=self.dtype,
            device=device,
            component_mode=self.component_mode,
        )
        object.__setattr__(self, "_pipe", pipe)
        pipe.scheduler.set_timesteps(
            self.training_cfg["scheduler_train_timesteps"],
            training=True,
            shift=self.training_cfg["flow_shift"],
        )
        _require_wan_flow_match_shift(
            pipe.scheduler,
            num_train_timesteps=int(self.training_cfg["scheduler_train_timesteps"]),
        )
        # A14B 每个 run 只训一个去噪专家：时间步只从它那一段抽（``mspc.models.noise_expert``）；单模型底座抽全段。
        self.timestep_boundaries = (0.0, 1.0)
        if self.model_cfg.get("expert") is not None:
            from mspc.models.noise_expert import (NoiseExpertOptions, expert_timestep_index_range,
                                                  index_range_as_boundaries)
            count = int(self.training_cfg["scheduler_train_timesteps"])
            index_range = expert_timestep_index_range(
                pipe.scheduler.timesteps, NoiseExpertOptions.from_mapping(self.model_cfg["expert"]), count)
            self.timestep_boundaries = index_range_as_boundaries(index_range, len(pipe.scheduler.timesteps))
        self.memory_prefix_options = _memory_prefix_options(self.model_cfg)
        from mspc.models.spatial_layout_inputs import SpatialLayoutOptions
        layout_options = self.model_cfg.get("spatial_layout")
        self.spatial_layout_options = (None if layout_options is None else
                                       SpatialLayoutOptions.from_mapping(layout_options))
        from mspc.models.region_attention import attach_layout_modules
        from mspc.models.planner_tokens import PlannerTokenOptions, attach_planner_connector
        planner_options = self.model_cfg.get("planner_tokens")
        self.planner_token_options = (None if planner_options is None else
                                      PlannerTokenOptions.from_mapping(planner_options))
        if self.planner_token_options is not None and self.spatial_layout_options is not None:
            raise ValueError("planner tokens replace model.spatial_layout; set it to null")
        from mspc.models.first_frame_tokens import attach_first_frame_tokens, first_frame_token_options
        self.first_frame_token_options = first_frame_token_options(self.model_cfg)
        from mspc.models.source_identity import SourceIdentityOptions
        source_options = self.model_cfg.get("source_identity")
        self.source_identity_options = (None if source_options is None else
                                       SourceIdentityOptions.from_mapping(source_options))
        # Component modes are validated by the loader: cache encoding has no DiT.
        if pipe.dit is not None:
            attach_layout_modules(pipe.dit, self.spatial_layout_options)
            attach_planner_connector(pipe.dit, self.planner_token_options)
            attach_first_frame_tokens(pipe.dit, self.first_frame_token_options)
            pipe.dit.source_identity_options = self.source_identity_options
        if self.component_mode in (WAN22_COMPONENT_MODE_CACHED_TRAINING,
                                   WAN22_COMPONENT_MODE_CACHED_TRAINING_LIVE_T5):
            pipe.freeze_except(["dit"])
            freeze_spec = require_dit_parameter_freeze(self.training_cfg["dit_parameter_freeze"])
            apply_named_dit_freeze(pipe.dit, freeze_spec)
            require_dit_trainable(pipe.dit, freeze_spec)
            self.dit = pipe.dit
            policy = self.training_cfg["activation_checkpointing"]
            if policy == "save_self_attention":
                from diffsynth.core.gradient.gradient_checkpoint import install_self_attention_checkpointing
                install_self_attention_checkpointing(
                    self.dit,
                    cache_layers=self.training_cfg.get("self_attention_cache_layers"),
                    no_recompute_suffix_layers=self.training_cfg.get("no_recompute_suffix_layers"),
                )
            elif policy != "recompute_all":
                raise ValueError("unknown training.activation_checkpointing policy")
        else:
            for name in ("text_encoder", "vae"):
                module = getattr(pipe, name, None)
                if isinstance(module, nn.Module):
                    module.eval()
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
        self._shape_checker_type = shape_checker_type
        self.profile_sync_cuda = False
        # 逐步读数与计时值共用这一条通道，因为 ``loop.py`` 只有 ``last_timing`` 一个 per-step 出口。
        self.last_timing: dict[str, Any] = {}

    @property
    def memory_prefix_enabled(self) -> bool:
        """这条臂在不在。``model.memory_prefix: null`` 就是「不在」，没有第二个开关。

        5B无论参考是否dropout均使用原生per-token时间；M只决定哪些前缀token为0。
        """

        return self.memory_prefix_options is not None

    def _memory_frame_count(self, data: Mapping[str, Any]) -> int:
        """这一个样本前接了几张干净记忆潜帧。非记忆臂恒 0。

        记忆臂上这个键**必须**由数据集写出来（``mspc.training.block_dataset``），缺键
        即抛：此数量决定主进程编码并前接多少张参考，以及 loss 的目标起点。
        """

        if not self.memory_prefix_enabled:
            if "memory_frame_count" in data:
                raise RuntimeError(
                    "the sample carries memory_frame_count but model.memory_prefix is "
                    "null; this arm has no memory prefix"
                )
            return 0
        count = data.get("memory_frame_count")
        if type(count) is not int:
            raise RuntimeError(
                "model.memory_prefix is set, so every sample must carry an int "
                f"memory_frame_count; got {count!r}"
            )
        if not 0 <= count <= self.memory_prefix_options.frames_max:
            raise RuntimeError(
                f"memory_frame_count {count} is outside "
                f"[0, {self.memory_prefix_options.frames_max}]"
            )
        return count

    def _sft_loss(
        self,
        data: Mapping[str, Any],
        inputs_shared: Mapping[str, Any],
        inputs_posi: Mapping[str, Any],
    ) -> torch.Tensor:
        """参考按显式σ增强，目标段使用原生flow-matching监督。"""

        drawn: dict[str, Any] = {}
        loss = flow_match_sft_loss(
            self.pipe,
            reference_noise_sigma_max=(0.0 if self.memory_prefix_options is None
                                       else self.memory_prefix_options.noise.train_sigma_max),
            drawn=drawn,
            profile_sync_cuda=self.profile_sync_cuda,
            **inputs_shared,
            **inputs_posi,
        )
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
            # 报清楚坏在哪一层：输入、文字编码、还是权重（训练集群 1248398：第 1 次更新后两个 rank 同时非有限，原报错分不出来）。
            broken = [name for name, parameter in self.dit.named_parameters() if not torch.isfinite(parameter).all()]
            raise RuntimeError(
                "flow-matching SFT loss did not return a finite scalar: "
                f"case_id={data.get('case_id')!r} timestep={drawn.get('timestep')} "
                f"input_latents_finite={bool(torch.isfinite(inputs_shared['input_latents']).all())} "
                f"context_finite={bool(torch.isfinite(inputs_posi['context']).all())} "
                f"non_finite_dit_parameters={len(broken)} first={broken[:3]}")
        self.last_timing.update(drawn)
        return loss


    @property
    def pipe(self) -> Any:
        return self._pipe

    @property
    def checkpoint_input_offload(self) -> bool:
        """仅已验证的缓存文字训练移除输入卸载；其它既有配方保持原策略。"""
        return (self.component_mode != WAN22_COMPONENT_MODE_CACHED_TRAINING
                and self.training_cfg["use_gradient_checkpointing"]
                and self.training_cfg["activation_checkpointing"] == "recompute_all")

    def _prepare_inputs(
        self,
        data: Mapping[str, Any],
        *,
        allow_cache_uint8: bool = False,
        require_prompts: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if not isinstance(data, Mapping):
            raise TypeError("Wan2.2 P3 training data must be a mapping")
        video = data.get("video")
        frames = (
            _uint8_tensor_to_pil_frames(video)
            if allow_cache_uint8
            and isinstance(video, torch.Tensor)
            and video.dtype == torch.uint8
            else _tensor_to_pil_frames(video)
        )
        num_frames = len(frames)
        shot_groups = _require_group_sequence(
            data.get("shot_groups"), "training data.shot_groups"
        )
        latent_shot_groups = _require_group_sequence(
            data.get("latent_shot_groups"), "training data.latent_shot_groups"
        )
        if len(shot_groups) != len(latent_shot_groups):
            raise ValueError("training shot_groups and latent_shot_groups must have equal length")
        prompts = (
            require_shot_prompts(data.get("prompt"), len(shot_groups))
            if require_prompts
            else None
        )
        multi_shot = len(shot_groups) > 1

        height, width = int(video.shape[2]), int(video.shape[3])
        height_factor = getattr(self.pipe, "height_division_factor", None)
        width_factor = getattr(self.pipe, "width_division_factor", None)
        if (
            type(height_factor) is not int
            or type(width_factor) is not int
            or height_factor <= 0
            or width_factor <= 0
        ):
            raise RuntimeError("DiffSynth did not expose valid spatial division factors")
        if height % height_factor or width % width_factor:
            raise ValueError(
                "Wan2.2 P3 refuses implicit spatial resizing; training video dimensions "
                f"{height}x{width} must be divisible by {height_factor}x{width_factor}"
            )

        if multi_shot:
            validate_groups = getattr(
                importlib.import_module("diffsynth.pipelines.wan_video"),
                "validate_multishot_groups",
            )
            normalized_shots, normalized_latents = validate_groups(
                shot_groups,
                latent_shot_groups,
                num_frames,
            )
            prompt_input: str | list[str] | None = prompts
        else:
            if (
                not isinstance(shot_groups[0], (list, tuple))
                or not isinstance(latent_shot_groups[0], (list, tuple))
                or len(shot_groups[0]) != 2
                or len(latent_shot_groups[0]) != 2
            ):
                raise ValueError("K=1 training sample groups must be [start, end]")
            raw_start, raw_end = shot_groups[0]
            latent_start, latent_end = latent_shot_groups[0]
            expected_latent_frames = (num_frames - 1) // 4 + 1
            if (
                raw_start != 0
                or raw_end != num_frames
                or (num_frames - 1) % 4 != 0
                or latent_start != 0
                or latent_end != expected_latent_frames
            ):
                raise ValueError("K=1 training sample must be one native Wan 4n+1 segment")
            normalized_shots = None
            normalized_latents = None
            prompt_input = None if prompts is None else prompts[0]

        inputs_shared: dict[str, Any] = {
            "input_image": None,
            "end_image": None,
            "input_video": frames,
            "denoising_strength": 1.0,
            "control_video": None,
            "reference_image": None,
            "camera_control_direction": None,
            "camera_control_speed": 1 / 54,
            "camera_control_origin": (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
            "vace_video": None,
            "vace_video_mask": None,
            "vace_reference_image": None,
            "vace_scale": 1.0,
            "reference_frames": 0,
            "seed": None,
            "rand_device": self.pipe.device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": 1.0,
            "cfg_merge": False,
            "sigma_shift": self.training_cfg["flow_shift"],
            "motion_bucket_id": None,
            "longcat_video": None,
            "tiled": self.pipeline_cfg["tiled"],
            "tile_size": tuple(self.pipeline_cfg["tile_size"]),
            "tile_stride": tuple(self.pipeline_cfg["tile_stride"]),
            "sliding_window_size": None,
            "sliding_window_stride": None,
            "input_audio": None,
            "audio_sample_rate": 16000,
            "s2v_pose_video": None,
            "audio_embeds": None,
            "s2v_pose_latents": None,
            "motion_video": None,
            "animate_pose_video": None,
            "animate_face_video": None,
            "animate_inpaint_video": None,
            "animate_mask_video": None,
            "vap_video": None,
            "wantodance_music_path": None,
            "wantodance_reference_image": None,
            "wantodance_fps": 30,
            "wantodance_keyframes": None,
            "wantodance_keyframes_mask": None,
            "framewise_decoding": False,
            "use_gradient_checkpointing": self.training_cfg["use_gradient_checkpointing"],
            "use_gradient_checkpointing_offload": self.checkpoint_input_offload,
            "max_timestep_boundary": self.timestep_boundaries[1],
            "min_timestep_boundary": self.timestep_boundaries[0],
            "temporal_position_mode": self.temporal_position_mode,
            # 这条路径读原始视频、整段加噪，没有条件前缀可言。写出来而不是留给默认值：
            # loss 每一步都要知道前多少帧不进监督，那个数字只能有一个来源。
            "prefix_frames": 0,
        }
        if multi_shot:
            inputs_shared["shot_groups"] = normalized_shots
            inputs_shared["latent_shot_groups"] = normalized_latents
        inputs_posi = {} if prompt_input is None else {"prompt": prompt_input}
        return inputs_shared, inputs_posi, {}

    def forward(self, data: Mapping[str, Any]) -> torch.Tensor:
        """显式视频／潜变量输入共用同一参考、文字与flow-matching训练路径。"""
        if self.component_mode not in (WAN22_COMPONENT_MODE_CACHED_TRAINING,
                                       WAN22_COMPONENT_MODE_CACHED_TRAINING_LIVE_T5):
            raise RuntimeError("cache encoder cannot run DiT training loss")
        self.last_timing = {"target_vae_seconds": 0.0}
        if self.profile_sync_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        target_storage = data["target_storage"]
        if target_storage == "video":
            if "input_latents" in data:
                raise ValueError("video target storage must not carry cached input_latents")
            encoded = self._encode_online_target(data)
            data = {**data, "input_latents": encoded}
        elif target_storage == "latents":
            if "video" in data:
                raise ValueError("latent target storage must not carry raw video")
        else:
            raise ValueError("target_storage must explicitly be video or latents")
        inputs_shared = self._prepare_latent_inputs(data)
        if self.profile_sync_cuda:
            torch.cuda.synchronize()
        self.last_timing["input_prepare_seconds"] = time.perf_counter() - started
        if "context" in data:
            raise RuntimeError("training context must come from its configured text owner")
        started = time.perf_counter()
        context = (self._encode_live_t5_context(data) if self._text_context_resolver is None else
                   self._text_context_resolver.resolve(data, device=self.device))
        if self.spatial_layout_options is not None:
            from mspc.models.spatial_layout_inputs import encode_spatial_layout_inputs
            if "target_layouts" not in data:
                raise ValueError("spatial renderer training requires explicit target_layouts")
            prefix = self._memory_frame_count(data)
            groups = data.get("shot_groups", [[0, data["num_frames"]]])[prefix:]
            inputs_shared["text_attention_backend"] = self.spatial_layout_options.text_attention_backend
            inputs_shared["spatial_layout"] = encode_spatial_layout_inputs(
                self.pipe.tokenizer, data["target_layouts"], prompts=data["prompt"][prefix:],
                frame_counts=[end - start for start, end in groups], options=self.spatial_layout_options)
        if self.planner_token_options is not None:
            # token 要逐 token 的时间步调制。
            inputs_shared["planner_tokens"] = self._planner_token_condition(data)
            if inputs_shared["planner_tokens"] is not None:
                inputs_shared["fuse_vae_embedding_in_latents"] = True
        elif "planner_tokens" in data:
            raise ValueError("the sample carries planner tokens but model.planner_tokens is null")
        if "target_loss_weights" in data:
            inputs_shared["target_loss_weights"] = data["target_loss_weights"].to(self.device)
        inputs_shared["flow_draw_seed"] = data["flow_draw_seed"]
        if self.source_identity_options is not None:
            from mspc.models.source_identity import compile_source_identity_condition
            inputs_shared["source_identity_condition"] = compile_source_identity_condition(
                self.pipe.tokenizer, data["prompt"], data["reference_entities"],
                characters=data["reference_characters"], shot_objects=data["reference_shot_objects"],
                settings=data["reference_settings"],
                alignment_layer=self.source_identity_options.alignment_layer,
                max_alignment_cells_per_reference=self.source_identity_options.max_alignment_cells_per_reference)
        if self.profile_sync_cuda:
            torch.cuda.synchronize()
        if self._text_context_resolver is None:
            self.last_timing["t5_seconds"] = time.perf_counter() - started
        else:
            self.last_timing["t5_seconds"] = 0.0
            self.last_timing["text_context_read_seconds"] = time.perf_counter() - started
        return self._sft_loss(data, inputs_shared, {"context": context})

    def _planner_token_condition(self, data: Mapping[str, Any]):
        """数据集写出的 token（特征与本镜坐标）；整组丢弃的样本是 None。"""
        from mspc.models.planner_tokens import PlannerTokenCondition
        if "planner_tokens" not in data:
            raise ValueError("model.planner_tokens is set, so every sample must carry planner_tokens (or null)")
        value = data["planner_tokens"]
        if value is None:
            return None
        geometry = (value["geometry"].to(device=self.device)
                    if self.planner_token_options.content == "person_geometry" else None)
        return PlannerTokenCondition(
            features=value["features"].to(device=self.device),
            shot_tokens=tuple(tuple(int(v) for v in row) for row in value["shot_tokens"]),
            geometry=geometry)

    def _encode_live_t5_context(self, data: Mapping[str, Any]) -> torch.Tensor:
        """把这个样本的文字编成 ``context``，一行对应一个 shot 组。

        记忆样本的 prompt 由数据集直接写出 M+1 句（``wan22_contract.memory_prefix_case_layout``，
        推理侧调同一个函数）：前 M 句是记忆行（``model.memory_prefix.row_policy``：只有角色句，或
        角色句＋该记忆镜自己的文字），最后一句是目标镜。每句各自过 T5——
        与推理管线 ``WanVideoUnit_PromptEmbedder`` 逐句编码同形。
        """

        shot_groups = data.get("shot_groups")
        group_count = 1 if shot_groups is None else len(shot_groups)
        memory_frame_count = self._memory_frame_count(data)
        prompt_count = group_count - memory_frame_count
        if prompt_count < 1:
            raise RuntimeError(
                f"{group_count} shot group(s) with {memory_frame_count} memory frame(s) "
                "leaves no prompted shot"
            )
        raw_prompts = data.get("prompt")
        if isinstance(raw_prompts, (str, bytes)) or not isinstance(raw_prompts, (list, tuple)):
            raise ValueError("training data.prompt must be a list of prompts, one per shot group")
        if len(raw_prompts) != group_count:
            raise ValueError(
                f"training prompt count {len(raw_prompts)} does not match {group_count} "
                "shot group(s)"
            )
        if memory_frame_count:
            require_memory_rows(
                raw_prompts[:memory_frame_count],
                row_policy=self.memory_prefix_options.row_policy,
            )
        prompts = list(raw_prompts[:memory_frame_count]) + require_shot_prompts(
            list(raw_prompts[memory_frame_count:]), prompt_count
        )
        prompt_input: str | list[str] = prompts if len(prompts) > 1 else prompts[0]
        return self._encode_prompt_context(prompt_input)

    def _encode_prompt_context(self, prompt_input: str | list[str]) -> torch.Tensor:
        """跑管线的 prompt 单元，把 ``prompt_input`` 编成 ``context``（live T5 的唯一出口）。"""

        # DiffSynth unit_runner indexes cfg_scale; training live-T5 must not run CFG.
        inputs_shared: dict[str, Any] = {"cfg_scale": 1.0, "cfg_merge": False}
        inputs_posi: dict[str, Any] = {"prompt": prompt_input}
        inputs_nega: dict[str, Any] = {}
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit,
                self.pipe,
                inputs_shared,
                inputs_posi,
                inputs_nega,
            )
        context = inputs_posi.get("context")
        if not isinstance(context, torch.Tensor):
            raise RuntimeError("Wan2.2 live T5 did not produce context")
        return context.to(device=self.device, dtype=self.dtype)

    def load_frozen_vae(self, cache_provenance_path: Path | None) -> None:
        """主进程同一冻结VAE编码目标及参考；离线目标另核其明确缓存合同。"""
        from mspc.data.cache_provenance import load_cache_contract
        from mspc.models.wan22_loader import load_vae_only_runtime

        contract = (None if cache_provenance_path is None else
                    load_cache_contract(cache_provenance_path, self.model_cfg["paths"], self.pipeline_cfg))
        devices = ([self.device.index if self.device.index is not None else torch.cuda.current_device()]
                   if self.device.type == "cuda" else [])
        with torch.random.fork_rng(devices=devices):
            runtime = load_vae_only_runtime(self.model_cfg, self.pipeline_cfg, self.device)
        runtime.pipe.vae.requires_grad_(False).eval()
        object.__setattr__(self, "reference_runtime", runtime)
        self.reference_encoding = (None if contract is None else
            (contract["models"]["vae"]["revision"],
             {k: contract["encoding"][k]
              for k in ("dtype", "pixel_conversion", "tiled", "tile_size", "tile_stride")}))

    def _prepare_latent_inputs(self, data: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(data, Mapping):
            raise TypeError("latent inputs must be a mapping")
        input_latents = data.get("input_latents")
        if not isinstance(input_latents, torch.Tensor):
            raise TypeError("latent inputs.input_latents must be a tensor")
        memory_frame_count = self._memory_frame_count(data)
        self.last_timing["reference_encoded_frames"] = memory_frame_count
        self.last_timing["reference_vae_seconds"] = 0.0
        input_latents = input_latents.to(device=self.device, dtype=self.dtype)
        if self.memory_prefix_options is not None:
            images = data["reference_images"]
            if (not isinstance(images, torch.Tensor) or images.dtype != torch.uint8
                    or tuple(images.shape) != (memory_frame_count, 3, int(data["height"]), int(data["width"]))):
                raise ValueError("reference_images must contain exactly the selected uint8 CHW frames")
        if memory_frame_count:
            from mspc.inference.wan22 import encode_single_frame

            if self.profile_sync_cuda:
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            with torch.no_grad():
                references = [encode_single_frame(self.reference_runtime, image) for image in images]
            if self.profile_sync_cuda:
                torch.cuda.synchronize(self.device)
            self.last_timing["reference_vae_seconds"] = time.perf_counter() - started
            input_latents = torch.cat([*references, input_latents], dim=2)
        inputs_shared: dict[str, Any] = {
            "input_latents": input_latents.to(device=self.device, dtype=self.dtype),
            "height": int(data["height"]),
            "width": int(data["width"]),
            "num_frames": int(data["num_frames"]),
            "temporal_position_mode": self.temporal_position_mode,
            "use_gradient_checkpointing": self.training_cfg["use_gradient_checkpointing"],
            "use_gradient_checkpointing_offload": self.checkpoint_input_offload,
            "max_timestep_boundary": self.timestep_boundaries[1],
            "min_timestep_boundary": self.timestep_boundaries[0],
            "cfg_merge": False,
            # 5B沿用原生per-token时间，包括M==0/dropout；1.3B有参考时复用同一
            # time MLP新增逐token的参考t_ref／目标t展开，无参考时保留原生scalar时间。
            # fused声明条件前缀，prefix_frames给帧数；flow loss独立处理参考σ和目标t。
            "fuse_vae_embedding_in_latents": memory_frame_count > 0,
            "prefix_frames": memory_frame_count,
        }
        if memory_frame_count:
            # 同一样本内所有参考共享一个 RoPE 偏移；训练偏移由数据集按样本抽出（见
            # ``block_dataset.BlockDataset.draw``），这里只核形状。
            offset = data.get("reference_offset")
            if (not isinstance(offset, (list, tuple)) or len(offset) != 3
                    or any(type(value) is not int for value in offset)):
                raise RuntimeError(
                    "a sample with memory frames must carry reference_offset [dt, dh, dw]; "
                    f"got {offset!r}"
                )
            inputs_shared["reference_offset"] = tuple(offset)
            inputs_shared["memory_attention_policy"] = (
                self.memory_prefix_options.attention_policy
            )
            inputs_shared["memory_attention_window"] = self.memory_prefix_options.attention_window
        self.last_timing["memory_frame_count"] = memory_frame_count
        shot_groups = data.get("shot_groups")
        latent_shot_groups = data.get("latent_shot_groups")
        if shot_groups is not None or latent_shot_groups is not None:
            if shot_groups is None or latent_shot_groups is None:
                raise ValueError(
                    "latent inputs shot_groups and latent_shot_groups must be provided together"
                )
            validate_groups = getattr(
                importlib.import_module("diffsynth.pipelines.wan_video"),
                "validate_multishot_groups",
            )
            normalized_shots, normalized_latents = validate_groups(
                shot_groups,
                latent_shot_groups,
                inputs_shared["num_frames"],
            )
            inputs_shared["shot_groups"] = normalized_shots
            inputs_shared["latent_shot_groups"] = normalized_latents
        # 按镜路由的参考：数据集给带参考的样本写每个目标镜读哪些参考（``mspc.data.reference_routing``）；
        # 整组丢弃（M=0）的样本不带，走目标自己的注意力。
        routed = (self.memory_prefix_options is not None
                  and self.memory_prefix_options.attention_policy in ROUTED_MEMORY_ATTENTION_POLICIES)
        if (routed and memory_frame_count > 0) != ("reference_routing" in data):
            raise ValueError("reference_routing is carried exactly by samples with references under a routed "
                             f"model.memory_prefix.attention_policy; got {memory_frame_count} reference(s) and "
                             f"routing {data.get('reference_routing')!r}")
        if "reference_routing" in data:
            inputs_shared["reference_routing"] = require_reference_routing(
                data["reference_routing"], reference_count=memory_frame_count,
                target_shots=len(inputs_shared["latent_shot_groups"]) - memory_frame_count)
        return inputs_shared

    def _encode_online_target(self, data: Mapping[str, Any]) -> torch.Tensor:
        """原视频只含目标RGB；用原完整分组减掉参考前缀后逐镜独立编码。"""
        prefix = self._memory_frame_count(data)
        video = data["video"]
        groups = data.get("shot_groups")
        latent_groups = data.get("latent_shot_groups")
        if groups is None and latent_groups is None:
            if prefix or len(data["target_shot_ids"]) != 1:
                raise ValueError("only one target shot without references may omit shot groups")
            frames = int(video.shape[1])
            target_groups = [[0, frames]]
            target_latents = [[0, (frames - 1) // 4 + 1]]
        else:
            if groups is None or latent_groups is None:
                raise ValueError("online RGB and latent shot groups must be declared together")
            target_groups = [[start - prefix, end - prefix] for start, end in groups[prefix:]]
            target_latents = [[start - prefix, end - prefix] for start, end in latent_groups[prefix:]]
        if self.profile_sync_cuda:
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        latents, _, _ = self._encode_target_video(
            {"video": video, "shot_groups": target_groups, "latent_shot_groups": target_latents},
            self.reference_runtime.pipe,
        )
        if self.profile_sync_cuda:
            torch.cuda.synchronize(self.device)
        self.last_timing["target_vae_seconds"] = time.perf_counter() - started
        return latents

    def encode_for_cache(
        self, data: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        """Return ``(input_latents, context, embed_meta)`` for the offline cache.

        Dual-key encode keeps the VAE/T5 units used by video-read forward.
        Latents-only encode needs no prompt, so it drives the VAE directly and
        returns ``context=None``.
        """

        if self.component_mode in (WAN22_COMPONENT_MODE_CACHED_TRAINING,
                                   WAN22_COMPONENT_MODE_CACHED_TRAINING_LIVE_T5):
            raise RuntimeError("Wan2.2 cached_training_live_t5 cannot encode VAE cache tensors")
        if self.component_mode == WAN22_COMPONENT_MODE_CACHE_ENCODE_LATENTS:
            return self._encode_target_video(data, self.pipe)
        with torch.no_grad():
            inputs_shared, inputs_posi, inputs_nega = self._prepare_inputs(
                data,
                allow_cache_uint8=True,
                require_prompts=True,
            )
            multi_shot = "latent_shot_groups" in inputs_shared
            for unit in self.pipe.units:
                if multi_shot and isinstance(unit, self._shape_checker_type):
                    continue
                inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                    unit,
                    self.pipe,
                    inputs_shared,
                    inputs_posi,
                    inputs_nega,
                )
        input_latents = inputs_shared.get("input_latents")
        context = inputs_posi.get("context")
        if not isinstance(input_latents, torch.Tensor) or not isinstance(context, torch.Tensor):
            raise RuntimeError(
                "Wan2.2 P3 embedding units did not produce input_latents/context for caching"
            )
        embed_meta = {
            "height": inputs_shared["height"],
            "width": inputs_shared["width"],
            "num_frames": inputs_shared["num_frames"],
            "shot_groups": inputs_shared.get("shot_groups"),
            "latent_shot_groups": inputs_shared.get("latent_shot_groups"),
        }
        return input_latents, context, embed_meta

    def encode_batch_for_cache(
        self, samples: Sequence[Mapping[str, Any]]
    ) -> list[tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]]:
        """Encode one same-shape group of cache samples, one sample at a time.

        There is no cross-sample VAE batch here. The
        Wan VAE is causal in time, so a multi-shot sample must be encoded shot
        by shot (see ``_encode_target_video``) — and the producer
        groups a batch by video shape alone
        (``mspc.training.latent_cache._split_same_shape_groups``), so two
        samples of the same ``[C,T,H,W]`` routinely carry different shot
        partitions. Stacking them would force one shared temporal split on all
        of them, which is exactly the joint-encode bug this replaces.
        ``precompute.encode_batch_size`` therefore sizes this group; it does
        not concatenate samples or change any shot's temporal boundary.
        """

        if not samples:
            raise ValueError("encode_batch_for_cache requires at least one sample")
        return [self.encode_for_cache(sample) for sample in samples]

    def _encode_target_video(
        self, data: Mapping[str, Any], pipe: Any,
    ) -> tuple[torch.Tensor, None, dict[str, Any]]:
        """Online targets and offline cache use the same independent-shot VAE encoding.

        Mirrors ``WanVideoUnit_InputVideoEmbedder``
        (``diffsynth/pipelines/wan_video.py``): a multi-shot video is encoded
        one shot at a time and concatenated on the latent temporal axis. The
        Wan VAE is causal, so a joint call over K shots returns
        ``(sum(len) - 1) // 4 + 1`` latent frames while ``latent_shot_groups``
        — and ``validate_multishot_groups``, which this sample already passed —
        require ``sum((len - 1) // 4 + 1)``, one leading frame per shot.
        """

        video, embed_meta = self._target_video_and_meta(data, pipe)
        shot_groups = embed_meta["shot_groups"]
        latent_shot_groups = embed_meta["latent_shot_groups"]
        vae = getattr(pipe, "vae", None)
        if not callable(getattr(vae, "encode", None)):
            raise RuntimeError("DiffSynth VAE did not expose encode for cache encode")
        encode_kwargs = {
            "device": pipe.device,
            "tiled": self.pipeline_cfg["tiled"],
            "tile_size": tuple(self.pipeline_cfg["tile_size"]),
            "tile_stride": tuple(self.pipeline_cfg["tile_stride"]),
        }
        with torch.no_grad():
            pipe.load_models_to_device(("vae",))
            if shot_groups is None:
                input_latents = vae.encode(video[None], **encode_kwargs).to(
                    dtype=pipe.torch_dtype, device=pipe.device
                )
            else:
                shot_latents_list = []
                for (frame_start, frame_end), (latent_start, latent_end) in zip(
                    shot_groups, latent_shot_groups
                ):
                    shot_latents = vae.encode(
                        video[None, :, frame_start:frame_end], **encode_kwargs
                    ).to(dtype=pipe.torch_dtype, device=pipe.device)
                    if shot_latents.shape[2] != latent_end - latent_start:
                        raise ValueError("per-shot VAE output did not match latent_shot_groups")
                    shot_latents_list.append(shot_latents)
                input_latents = torch.cat(shot_latents_list, dim=2)
        return input_latents, None, embed_meta

    def _target_video_and_meta(
        self, data: Mapping[str, Any], pipe: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Validate one latents-only sample and return VAE input plus embed meta.

        Skips PIL round-trip used by ``_prepare_inputs``. DiffSynth
        ``preprocess_image`` (``diffsynth/diffusion/base_pipeline.py``) casts to
        ``torch_dtype`` *before* the affine map and then computes
        ``x * (2 / 255) - 1``; the cast order is load-bearing, because bf16
        quantizes ``2 / 255`` to ``1 / 128`` and a fp32 ``x / 255 * 2 - 1``
        rounded afterwards agrees on only half the uint8 domain.
        """

        if not isinstance(data, Mapping):
            raise TypeError("Wan2.2 P3 training data must be a mapping")
        video = data.get("video")
        if not isinstance(video, torch.Tensor) or video.dim() != 4:
            raise ValueError("cache encode data.video must be a [C,T,H,W] tensor")
        if video.shape[0] != 3 or video.shape[1] <= 0:
            raise ValueError(
                "cache encode data.video must have three non-empty RGB channels, "
                f"got {tuple(video.shape)}"
            )
        dtype = getattr(pipe, "torch_dtype", None)
        if not isinstance(dtype, torch.dtype):
            raise RuntimeError("DiffSynth did not expose torch_dtype for cache encode")
        if video.dtype == torch.uint8:
            # Transfer compact RGB before the established BF16 cast/mul/add.
            # The cache path shares the ordinary VAE owner; do not normalize on
            # CPU only to copy the expanded tensor back to the encoder device.
            prepared = video.to(device=pipe.device).to(dtype=dtype).mul(2 / 255).add(-1)
        elif video.is_floating_point():
            if not torch.isfinite(video).all():
                raise ValueError("cache encode data.video contains non-finite values")
            if float(video.min()) < -1.00001 or float(video.max()) > 1.00001:
                raise ValueError("cache encode data.video must be normalized to [-1, 1]")
            prepared = video.float()
        else:
            raise ValueError(
                f"cache encode data.video must be uint8 or floating-point, got {video.dtype}"
            )
        num_frames = int(video.shape[1])
        height, width = int(video.shape[2]), int(video.shape[3])
        shot_groups = _require_group_sequence(
            data.get("shot_groups"), "training data.shot_groups"
        )
        latent_shot_groups = _require_group_sequence(
            data.get("latent_shot_groups"), "training data.latent_shot_groups"
        )
        if len(shot_groups) != len(latent_shot_groups):
            raise ValueError("training shot_groups and latent_shot_groups must have equal length")
        height_factor = getattr(pipe, "height_division_factor", None)
        width_factor = getattr(pipe, "width_division_factor", None)
        if (
            type(height_factor) is not int
            or type(width_factor) is not int
            or height_factor <= 0
            or width_factor <= 0
        ):
            raise RuntimeError("DiffSynth did not expose valid spatial division factors")
        if height % height_factor or width % width_factor:
            raise ValueError(
                "Wan2.2 P3 refuses implicit spatial resizing; training video dimensions "
                f"{height}x{width} must be divisible by {height_factor}x{width_factor}"
            )
        multi_shot = len(shot_groups) > 1
        if multi_shot:
            validate_groups = getattr(
                importlib.import_module("diffsynth.pipelines.wan_video"),
                "validate_multishot_groups",
            )
            normalized_shots, normalized_latents = validate_groups(
                shot_groups,
                latent_shot_groups,
                num_frames,
            )
        else:
            if (
                not isinstance(shot_groups[0], (list, tuple))
                or not isinstance(latent_shot_groups[0], (list, tuple))
                or len(shot_groups[0]) != 2
                or len(latent_shot_groups[0]) != 2
            ):
                raise ValueError("K=1 training sample groups must be [start, end]")
            raw_start, raw_end = shot_groups[0]
            latent_start, latent_end = latent_shot_groups[0]
            expected_latent_frames = (num_frames - 1) // 4 + 1
            if (
                raw_start != 0
                or raw_end != num_frames
                or (num_frames - 1) % 4 != 0
                or latent_start != 0
                or latent_end != expected_latent_frames
            ):
                raise ValueError("K=1 training sample must be one native Wan 4n+1 segment")
            normalized_shots = None
            normalized_latents = None
        return prepared.to(dtype=dtype), {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "shot_groups": normalized_shots,
            "latent_shot_groups": normalized_latents,
        }

    def optimizer_param_groups(self, args: Any) -> list[dict[str, Any]]:
        """AdamW 的参数组：一个组，包含全部可训练的 DiT 参数。"""

        trainable = [
            (name, parameter)
            for name, parameter in self.dit.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError("Wan2.2 P3 has no trainable DiT parameters")
        from mspc.models.region_attention import region_parameter_names
        from mspc.models.planner_tokens import PLANNER_CONNECTOR
        from mspc.models.first_frame_tokens import FIRST_FRAME_TOKENS
        if self.first_frame_token_options is not None:
            # 首帧组附加 token 从头训，学习率由 model.first_frame_twin／shot_registers 的 learning_rate 给出；
            # DiT 其余参数用全局学习率。
            extra = {name for name, _ in trainable if name.startswith(FIRST_FRAME_TOKENS + ".")}
            if not extra:
                raise RuntimeError("first-frame tokens are configured but their table is not trainable")
            return [{"params": [parameter for name, parameter in trainable if name not in extra]},
                    {"params": [parameter for name, parameter in trainable if name in extra],
                     "lr": self.first_frame_token_options.learning_rate}]
        if self.planner_token_options is not None:
            # 连接器从零训，学习率由 model.planner_tokens.learning_rate 给出；DiT 其余参数用全局学习率。
            connector = {name for name, _ in trainable if name.startswith(PLANNER_CONNECTOR + ".")}
            if not connector:
                raise RuntimeError("model.planner_tokens is set but the connector has no trainable parameters")
            return [{"params": [parameter for name, parameter in trainable if name not in connector]},
                    {"params": [parameter for name, parameter in trainable if name in connector],
                     "lr": self.planner_token_options.learning_rate}]
        region = region_parameter_names(self.dit)
        if not region:
            return [{"params": [parameter for _, parameter in trainable]}]
        # 门控位置 token 通道从零训，学习率单独由 model.spatial_layout.region_module 给出。
        rate = self.spatial_layout_options.region_module["learning_rate"]
        return [{"params": [parameter for name, parameter in trainable if name not in region]},
                {"params": [parameter for name, parameter in trainable if name in region], "lr": rate}]

    def trainable_param_names(self) -> set[str]:
        if not hasattr(self, "dit"):
            return set()
        return {
            f"dit.{name}"
            for name, parameter in self.dit.named_parameters()
            if parameter.requires_grad
        }

    def export_trainable_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        remove_prefix: str,
    ) -> dict[str, torch.Tensor]:
        checkpoint_parameters = self.trainable_param_names()
        exported: dict[str, torch.Tensor] = {}
        for name, value in state_dict.items():
            if name not in checkpoint_parameters:
                continue
            if not name.startswith(remove_prefix):
                raise ValueError(
                    f"trainable parameter {name} does not start with {remove_prefix}"
                )
            exported[name[len(remove_prefix):]] = value
        if not exported:
            raise RuntimeError("Wan2.2 P3 refused to export an empty full-DiT checkpoint")
        return exported


def build_model(
    args: TrainingArgs,
    device: torch.device,
) -> Wan22DiffSynthTrainingModule:
    component_mode = (WAN22_COMPONENT_MODE_CACHED_TRAINING if args.model_cfg.get("text_context_cache") is not None
                      else WAN22_COMPONENT_MODE_CACHED_TRAINING_LIVE_T5)
    return _build_model(args, device, component_mode=component_mode, load_checkpoint=True)


def build_cache_encoder(
    args: TrainingArgs,
    device: torch.device,
    *,
    cache_layout: str,
) -> Wan22DiffSynthTrainingModule:
    """Build the cache encoder for the requested layout; no DiT is requested."""

    if cache_layout not in WAN22_CACHE_LAYOUTS:
        raise ValueError(
            f"cache_layout must be one of {sorted(WAN22_CACHE_LAYOUTS)}: {cache_layout!r}"
        )
    component_mode = (
        WAN22_COMPONENT_MODE_CACHE_ENCODE_LATENTS
        if cache_layout == WAN22_CACHE_LAYOUT_LATENTS_ONLY
        else WAN22_COMPONENT_MODE_CACHE_ENCODE
    )
    return _build_model(
        args,
        device,
        component_mode=component_mode,
        load_checkpoint=False,
    )


def _build_model(
    args: TrainingArgs,
    device: torch.device,
    *,
    component_mode: str,
    load_checkpoint: bool,
) -> Wan22DiffSynthTrainingModule:
    if args.model_cfg is None or args.training_cfg is None or args.pipeline_cfg is None:
        raise ValueError("Wan2.2 runtime args were not configured")
    model = Wan22DiffSynthTrainingModule(
        model_cfg=args.model_cfg,
        training_cfg=args.training_cfg,
        pipeline_cfg=args.pipeline_cfg,
        device=device,
        component_mode=component_mode,
    )
    if model._text_context_resolver is not None:
        model._text_context_resolver.require_training_contract(args)
    if args.checkpoint_path is not None and not load_checkpoint:
        raise ValueError("Wan2.2 cache encoder does not accept a DiT checkpoint")
    if args.checkpoint_path is not None:
        checkpoint_path = require_path(args.checkpoint_path, "checkpoint")
        from safetensors.torch import load_file
        from mspc.training.dit_parameter_freeze import LORA_V1

        state_dict = load_file(str(checkpoint_path), device=str(device))
        freeze_spec = require_dit_parameter_freeze(args.training_cfg["dit_parameter_freeze"])
        if freeze_spec["recipe"] == LORA_V1:
            from mspc.models.lora import load_lora_state_dict

            loaded = load_lora_state_dict(model.dit, state_dict)
            print(
                f"load DiffSynth DiT LoRA from checkpoint (tensors={loaded}): {checkpoint_path}"
            )
        else:
            mode = load_dit_state_dict(model.dit, state_dict)
            print(
                f"load DiffSynth DiT from checkpoint ({mode}): {checkpoint_path}"
            )
    return model
