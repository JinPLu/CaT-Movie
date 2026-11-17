#!/usr/bin/env python3
"""Strict DiffSynth Wan2.2 pure-T2V multi-shot inference entry."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Callable, Mapping, Sequence

import torch

from mspc.config import ExperimentConfig, load_experiment_config
from mspc.core.wan22_contract import (
    memory_prefix_case_layout,
    require_memory_rows,
)
from mspc.data.multishot import RENDERED_PROMPT_CONTRACT
from mspc.inference.continuity_dev_32 import crop_frames_to_shot_lengths
from mspc.inference.flow_unipc import (
    FLOW_UNIPC_OFFICIAL,
    FlowUniPCOfficial,
    require_sampler,
)
from mspc.models.wan22_loader import (
    load_pure_t2v_pipeline,
    require_model_paths as require_wan22_model_paths,
    require_path,
    require_shot_prompts,
    require_temporal_position_mode,
)
from mspc.models.memory_prefix_options import MemoryAttentionWindow, MemoryPrefixModelOptions
from mspc.models.reference_noise import noise_reference_latents, reference_model_timestep
from mspc.models.source_identity import SourceIdentityOptions
from mspc.training.dit_parameter_freeze import load_dit_state_dict


def save_video(frames: Sequence[Any], output_path: Path, *, fps: int, quality: int) -> None:
    if not frames:
        raise ValueError("refusing to save an empty generated video")
    import imageio
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        output_path,
        [np.asarray(frame) for frame in frames],
        fps=fps,
        quality=quality,
    )
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"generated video was not written: {output_path}")


class _CleanPrefixDenoiseState:
    """把 N 张干净潜帧钉在本 case 潜变量的 ``[0:N]``，走 TI2V-5B 原生的「干净前缀 ＋ per-token
    timestep 0」口。原生首帧（N=1，帧是本镜自己的第 0 潜帧）与记忆前缀（N=M，帧前接在目标镜
    之前）共用这一条钉住路径；两者只差交给 ``model_fn`` 的键（``forward_kwargs``）和前缀算不算
    成片——后者归调用方（``generate_case``）处理。

    ``pipe.model_fn``
    是可替换属性，装卸在 ``generate_case`` 的 try/finally 里——但这一条还要接到**去噪循环
    本身**：原生口的粘回写在 ``WanVideoPipeline.__call__`` 的 ``scheduler.step`` 之后，键是
    ``WanVideoUnit_ImageEmbedderFused`` 从 ``input_image`` 编出来的 ``first_frame_latents``；
    这里给的是潜变量本身（不经 RGB→VAE），所以同一件事接在同一个位置：``pipe.scheduler.step``
    换成本对象的 :meth:`scheduler_step`，每一步之后把 ``[0:N]`` 写回同一份干净潜帧。

    干净前缀不加噪；训练与推理都使用干净条件，时间步沿用原生首帧口恒为 0。两个落点合起来是
    一件事——「前缀在整个循环里恒等于那 N 张干净潜帧」：

    * :meth:`__call__`（每次 ``model_fn`` 之前）把前缀原地写进管线自己的 ``latents``。第一次
      调用就是「把前缀写到初始噪声上」（原生单元同样覆盖第 0 帧的噪声）；之后每次进来都已经
      是它，写进去是恒等操作。
    * :meth:`scheduler_step` 在每一步之后写回，**包括最后一步**。

    ``scheduler_step`` 还顺手留下最后一步的 ``prev_sample``（σ=0 的干净潜变量）。

    **正负两条分支拿同一份前缀。** 前缀是观测，不是引导信号：只出现在一支上，就会被管线的
    ``nega + cfg·(posi − nega)`` 按 ``cfg_scale`` 放大成一个没人声明过的引导项。``latents``
    是 ``inputs_shared`` 里那一个张量对象，两支读的就是同一份；``forward_kwargs`` 也每次原样给。
    :meth:`assert_finished` 核的正是「每一步的每一支都调到了」。

    ``clean_latents is None`` 是 M=0：无锚点时沿用标量时间步，有锚点时走 per-token timestep。
    """

    def __init__(
        self,
        *,
        pipe: Any,
        original_model_fn: Callable[..., Any],
        cfg_scale: float,
        clean_latents: torch.Tensor | None,
        forward_kwargs: Mapping[str, Any],
    ) -> None:
        self.pipe = pipe
        self.original_model_fn = original_model_fn
        #: 干净前缀 ``[1, C, N, H, W]``：写入与粘回的都是这一份。
        self.clean_latents = clean_latents
        self.clean_frame_count = 0 if clean_latents is None else int(clean_latents.shape[2])
        if int(forward_kwargs["prefix_frames"]) != self.clean_frame_count:
            raise ValueError(
                f"forward_kwargs announce prefix_frames={forward_kwargs['prefix_frames']} "
                f"but {self.clean_frame_count} clean latent frame(s) are pinned"
            )
        self.forward_kwargs: dict[str, Any] = dict(forward_kwargs)
        # 依据：cfg_scale==1.0 每步一次 model_fn，否则先 posi
        # 再 nega 两次（见 WanVideoPipeline.__call__ 的去噪循环）。
        self._calls_per_step = 1 if cfg_scale == 1.0 else 2
        self._call_count = 0
        self._step_count = 0
        #: 最后一次 ``scheduler.step`` 的返回值（σ=0 的干净潜变量）。
        self._final_sample: torch.Tensor | None = None
        self._scheduler: Any = None
        self._original_scheduler_step: Callable[..., Any] | None = None
        self._scheduler_had_instance_step = False

    def install(self) -> None:
        """接上 ``pipe.model_fn`` 与 ``pipe.scheduler.step``，N 是多少都接。

        N==0 那一支没有可粘回的东西，接上只为留下最后一步的干净潜变量；``step`` 本身原样
        返回，数值与不接时逐位相同。
        """

        self.pipe.model_fn = self
        scheduler = self.pipe.scheduler
        step = scheduler.step
        unbound = getattr(type(scheduler), "step", None)
        if unbound is None or getattr(step, "__func__", None) is not unbound:
            # 已经被别人包过（例如 mspc.inference.scheduler_paste_back 的重加噪回填）。
            # 两种回填语义不同，叠在一起出的是第三种东西，所以这里停。
            raise RuntimeError(
                "pipe.scheduler.step is already wrapped; the clean prefix paste-back "
                "cannot stack on another scheduler hook"
            )
        self._scheduler = scheduler
        self._original_scheduler_step = step
        self._scheduler_had_instance_step = "step" in getattr(scheduler, "__dict__", {})
        scheduler.step = self.scheduler_step

    def restore(self) -> None:
        self.pipe.model_fn = self.original_model_fn
        if self._scheduler is None:
            return
        if self._scheduler_had_instance_step:
            self._scheduler.step = self._original_scheduler_step
        elif "step" in self._scheduler.__dict__:
            del self._scheduler.__dict__["step"]
        self._scheduler = None

    def __call__(self, **kwargs: Any) -> Any:
        self._call_count += 1
        self._write_prefix(kwargs.get("latents"))
        return self.original_model_fn(**(kwargs | self.forward_kwargs))

    def _require_room(self, latents: Any, label: str) -> None:
        if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
            raise RuntimeError(
                f"the clean prefix needs this case's [1, C, F, H, W] {label}; got "
                f"{type(latents).__name__}"
            )
        if latents.shape[2] <= self.clean_frame_count:
            raise RuntimeError(
                f"{latents.shape[2]} latent frame(s) leave no generated frame after a "
                f"{self.clean_frame_count}-frame clean prefix"
            )
        if self.clean_latents is not None and (
            latents.shape[:2] != self.clean_latents.shape[:2]
            or latents.shape[3:] != self.clean_latents.shape[3:]
        ):
            raise RuntimeError(
                f"clean prefix {tuple(self.clean_latents.shape)} does not fit {label} "
                f"{tuple(latents.shape)}: batch, channels and the latent grid must match"
            )

    def _write_prefix(self, latents: Any) -> None:
        """把干净前缀原地写进 ``latents`` 的前 N 帧。

        原地是要点：``latents`` 就是 ``inputs_shared["latents"]`` 那个对象，管线下一步的
        ``scheduler.step`` 读的也是它。N=0 时无事可做。
        """

        if self.clean_latents is None:
            return
        self._require_room(latents, "latents")
        latents[:, :, : self.clean_frame_count] = self.clean_latents

    def scheduler_step(self, model_output, timestep, sample, to_final=False, **kwargs):
        """原样走一步 ``scheduler.step``，把 ``[0:N]`` 写回那份干净前缀，并留下这一步的结果。

        前缀帧不按调度往下推：它们的时间步整条循环恒为 0，粘回的就是同一份干净潜帧。回填写在
        ``step`` 返回的新张量上而不是原地改 ``sample``；N==0 时没有可回填的东西，原样放行那一个
        张量对象。
        """

        if self._original_scheduler_step is None:
            raise RuntimeError("the clean prefix scheduler hook was never installed")
        prev_sample = self._original_scheduler_step(
            model_output, timestep, sample, to_final=to_final, **kwargs
        )
        self._require_room(prev_sample, "scheduler.step sample")
        if self.clean_latents is not None:
            prev_sample = prev_sample.clone()
            prev_sample[:, :, : self.clean_frame_count] = self.clean_latents
        self._step_count += 1
        self._final_sample = prev_sample
        return prev_sample

    def assert_finished(self) -> None:
        """跑完之后核调用次数：每个 scheduler 步的每一支都必须经过本对象。"""

        expected_calls = len(self.pipe.scheduler.timesteps) * self._calls_per_step
        if self._call_count != expected_calls:
            raise RuntimeError(
                "clean-prefix denoise loop ended with a mismatched model_fn call "
                f"count: expected {expected_calls} "
                f"({len(self.pipe.scheduler.timesteps)} scheduler steps x "
                f"{self._calls_per_step} call(s)/step), got {self._call_count}"
            )
        expected_steps = len(self.pipe.scheduler.timesteps)
        if self._step_count != expected_steps:
            raise RuntimeError(
                "every scheduler step must go through the clean prefix hook (it pastes "
                "the prefix back and keeps this case's own clean latent); expected "
                f"{expected_steps} step(s), got {self._step_count}"
            )


class _MemoryPrefixDenoiseState(_CleanPrefixDenoiseState):
    """记忆前缀：M 张固定噪声级潜帧前接在目标镜之前（``memory_prefix_inference_layout``），
    出片时去掉。M 是多少都接 ``scheduler.step``：每条故事的第 0 镜 M==0，而后面几镜
    参考来源由调用方显式给出，生成历史参考使用末帧单帧编码。

    ``memory_latents is None`` 是这条臂的**记忆 dropout 形态**（M=0）——与训练侧
    ``Wan22DiffSynthTrainingModule._prepare_latent_inputs`` 同一条判断。
    """

    def __init__(
        self,
        *,
        pipe: Any,
        original_model_fn: Callable[..., Any],
        cfg_scale: float,
        memory_latents: torch.Tensor | None,
        reference_timestep: float,
        reference_offset: tuple[int, int, int],
        attention_policy: str,
        attention_window: MemoryAttentionWindow,
        planner_tokens: Any = None,
        reference_routing: tuple[tuple[int, ...], ...] | None = None,
    ) -> None:
        frames = 0 if memory_latents is None else int(memory_latents.shape[2])
        self.reference_timestep = reference_timestep
        #: 每次 ``model_fn`` 多给的键，与训练侧 ``_prepare_latent_inputs`` 逐字对齐。
        #: ``prefix_frames == 0`` 时参考偏移与注意力策略都必须**缺席**：
        #: ``shot_phase_index_sequence`` 对「没有前缀却给了它」当场抛，``model_fn`` 要求两者同进同出。
        #: 规划器 token 要逐 token 的时间步调制，所以有 token 时即便没有参考也走融合那条路（与训练侧一致）。
        forward_kwargs: dict[str, Any] = {
            "fuse_vae_embedding_in_latents": bool(frames) or planner_tokens is not None,
            "prefix_frames": frames,
        }
        if planner_tokens is not None:
            forward_kwargs["planner_tokens"] = planner_tokens
        if frames:
            forward_kwargs["reference_timestep"] = reference_timestep
            forward_kwargs["reference_offset"] = reference_offset
            forward_kwargs["memory_attention_policy"] = attention_policy
            forward_kwargs["memory_attention_window"] = attention_window
        # 按镜路由的参考策略：每次调用都带同一份路由，正负两支读同一组参考（校验在 model_fn）。
        if reference_routing is not None:
            forward_kwargs["reference_routing"] = reference_routing
        super().__init__(
            pipe=pipe,
            original_model_fn=original_model_fn,
            cfg_scale=cfg_scale,
            clean_latents=memory_latents,
            forward_kwargs=forward_kwargs,
        )

    @property
    def memory_frame_count(self) -> int:
        return self.clean_frame_count



@dataclass(frozen=True)
class MemoryPrefixInput:
    """参考干净潜帧[1,C,M,H,W]和同序的完整编译参考行。

    shot_texts是完整编译行：固定参考角色句，写可见身份的两种另含源镜分镜在场的登记人物及其登记外貌、源镜地点
    （:func:`mspc.inference.segment_story.reference_row`）。
    reference_routing只随按镜路由的参考策略给：每个目标镜一项，是路由给它的参考下标（``mspc.data.reference_routing``）。
    """

    latents: torch.Tensor
    shot_texts: tuple[str, ...]
    reference_routing: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.shot_texts, tuple) or any(
            not isinstance(text, str) for text in self.shot_texts
        ):
            raise TypeError("shot_texts must be a tuple of strings, one per memory frame")
        if int(self.latents.shape[2]) != len(self.shot_texts):
            raise ValueError(
                f"{self.latents.shape[2]} memory latent frame(s) but "
                f"{len(self.shot_texts)} memory shot text(s)"
            )


@dataclass(frozen=True)
class SDEditInput:
    """SDEdit 式重去噪的起点：这一镜每一帧的 RGB 图（张数等于这一镜的帧数），编码后按 DiffSynth
    ``denoising_strength`` 加噪，从调度的那一段往下去噪。``strength`` 是走完的调度比例
    （0 < s < 1），实际起点噪声是它经 ``sigma_shift`` 变换后的 σ。"""

    frames: tuple[Any, ...]
    strength: float

    def __post_init__(self) -> None:
        if not isinstance(self.frames, tuple) or not self.frames:
            raise TypeError("frames must be a non-empty tuple of RGB images")
        if type(self.strength) is not float or not 0.0 < self.strength < 1.0:
            raise ValueError(f"strength must be a float in (0, 1); got {self.strength!r}")


def build_memory_prefix_controller(
    *,
    pipe: Any,
    device: torch.device,
    dtype: torch.dtype,
    cfg_scale: float,
    memory_prefix_options: MemoryPrefixModelOptions | None,
    memory_latents: torch.Tensor | None,
    seed: int,
    planner_tokens: Any = None,
    reference_routing: tuple[tuple[int, ...], ...] | None = None,
) -> _MemoryPrefixDenoiseState | None:
    """这一条 case 走不走记忆前缀口，以及走的话前缀是哪 M 帧。

    臂在不在只由 ``model.memory_prefix`` 说——记忆前缀不加任何参数，checkpoint 的键里看不
    出它。``None`` 返回值就是「这份 checkpoint 上没有这口」，此时给了记忆帧即抛。

    参考按显式固定sigma和独立种子只加噪一次，整条去噪循环回填同一份。
    sigma=0保留旧干净参考tensor，不改变目标噪声或全局RNG。
    """

    if memory_prefix_options is None:
        if planner_tokens is not None:
            raise RuntimeError("planner tokens need model.memory_prefix")
        if memory_latents is not None:
            raise RuntimeError(
                "memory latents were passed but model.memory_prefix is null: this "
                "checkpoint's arm has no memory prefix"
            )
        return None
    clean_memory = None
    reference_sigma = 0.0
    if memory_latents is not None:
        if not isinstance(memory_latents, torch.Tensor) or memory_latents.ndim != 5:
            raise ValueError(
                "memory_latents must be a [1, C, M, H, W] latent tensor; got "
                f"{type(memory_latents).__name__}"
            )
        if memory_latents.shape[0] != 1:
            raise ValueError(
                "memory_latents carries one case at a time; got batch "
                f"{memory_latents.shape[0]}"
            )
        frames = int(memory_latents.shape[2])
        if frames == 0:
            raise ValueError(
                "an empty memory prefix is the dropout arm: pass memory_latents=None "
                "rather than a zero-frame tensor"
            )
        if frames > memory_prefix_options.frames_max:
            raise ValueError(
                f"{frames} memory frame(s) exceed model.memory_prefix.frames_max "
                f"{memory_prefix_options.frames_max}"
            )
        clean_memory = memory_latents.to(device=device, dtype=dtype)
        reference_sigma = memory_prefix_options.noise.inference_sigma
        clean_memory = noise_reference_latents(clean_memory, sigma=reference_sigma, seed=seed)
    return _MemoryPrefixDenoiseState(
        pipe=pipe,
        original_model_fn=pipe.model_fn,
        cfg_scale=cfg_scale,
        memory_latents=clean_memory,
        reference_timestep=reference_model_timestep(reference_sigma),
        reference_offset=memory_prefix_options.reference_offset.inference,
        attention_policy=memory_prefix_options.attention_policy,
        attention_window=memory_prefix_options.attention_window,
        planner_tokens=planner_tokens,
        reference_routing=reference_routing,
    )


class _FirstFrameDenoiseState(_CleanPrefixDenoiseState):
    """首帧在高噪声阶段保持干净，首次放开时接回当前噪声层。"""

    def __init__(self, *, pin_until_sigma: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pin_until_sigma = pin_until_sigma
        self._initial_noise: torch.Tensor | None = None
        self._released = False

    def __call__(self, **kwargs: Any) -> Any:
        latents = kwargs["latents"]
        self._require_room(latents, "latents")
        if self._initial_noise is None:
            self._initial_noise = latents[:, :, :1].clone()
        sigma = self.pipe.scheduler.sigmas[self._step_count]
        if not self._released and bool(sigma < self.pin_until_sigma):
            # 使用原始同种子噪声；正负 CFG 分支共享同一份接回后的首帧。
            latents[:, :, :1] = self.pipe.scheduler.add_noise(
                self.clean_latents, self._initial_noise,
                self.pipe.scheduler.timesteps[self._step_count],
            )
            self._released = True
        if not self._released:
            return super().__call__(**kwargs)
        self._call_count += 1
        # 不再给干净前缀时间步 0；与普通 T2V 帧走相同的模型入口。
        return self.original_model_fn(**kwargs)

    def scheduler_step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if not self._released:
            return super().scheduler_step(
                model_output, timestep, sample, to_final=to_final, **kwargs
            )
        result = self._original_scheduler_step(
            model_output, timestep, sample, to_final=to_final, **kwargs
        )
        self._step_count += 1
        self._final_sample = result
        return result


def build_first_frame_controller(
    *,
    pipe: Any,
    device: torch.device,
    dtype: torch.dtype,
    cfg_scale: float,
    first_frame: torch.Tensor,
    pin_until_sigma: float,
) -> _CleanPrefixDenoiseState | None:
    """原生首帧口，入参是潜变量：本镜第 0 潜帧钉成 ``first_frame``、时间步 0，其余帧照常去噪。

    与 ``input_image`` 走 ``WanVideoUnit_ImageEmbedderFused`` 是同一个口（``latents[:, :, 0:1]``
    写入并每步粘回、``fuse_vae_embedding_in_latents=True``、``prefix_frames=1``），差别只在帧从
    哪来：那边是 RGB 现编，这里是调用方给的潜帧（例如 latent 缓存里的真片第 0 潜帧，或上一镜
    末帧的单帧编码 :meth:`Wan22InferenceRuntime.encode_single_frame`）。帧是本镜成片的第 0 帧，
    不从输出里去掉；也不带记忆的位置／注意力策略——原生首帧口没有。
    """

    if not isinstance(first_frame, torch.Tensor) or first_frame.ndim != 5:
        raise ValueError(
            "first_frame must be a [1, C, 1, H, W] latent tensor; got "
            f"{type(first_frame).__name__}"
        )
    if first_frame.shape[0] != 1 or first_frame.shape[2] != 1:
        raise ValueError(
            "first_frame is one latent frame of one case: shape [1, C, 1, H, W]; got "
            f"{tuple(first_frame.shape)}"
        )
    if type(pin_until_sigma) not in (float, int) or not 0 <= pin_until_sigma <= 1:
        raise ValueError("first_frame_pin_until_sigma must be an explicit number in [0, 1]")
    # σ*=1 是声明规定的从不钉端点，包括调度器的首步 σ=1。
    if pin_until_sigma == 1:
        return None
    return _FirstFrameDenoiseState(
        pin_until_sigma=float(pin_until_sigma),
        pipe=pipe,
        original_model_fn=pipe.model_fn,
        cfg_scale=cfg_scale,
        clean_latents=first_frame.to(device=device, dtype=dtype),
        forward_kwargs={"fuse_vae_embedding_in_latents": True, "prefix_frames": 1},
    )


def memory_prefix_inference_layout(
    *,
    prompts: Sequence[str],
    shot_groups: Sequence[Sequence[int]],
    latent_shot_groups: Sequence[Sequence[int]],
    memory_shot_texts: Sequence[str],
    row_policy: str,
) -> tuple[list[str], tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """参考完整编译行＋联合目标段，直接调用训练侧同一个布局合同。"""

    require_memory_rows(memory_shot_texts, row_policy=row_policy)
    out_prompts, out_shots, out_latents = memory_prefix_case_layout(
        target_prompts=prompts,
        target_shot_groups=shot_groups,
        target_latent_shot_groups=latent_shot_groups,
        memory_rows=memory_shot_texts,
        row_policy=row_policy,
    )
    return list(out_prompts), out_shots, out_latents


@dataclass(frozen=True)
class LongShotRope:
    """单镜超过底座原生长度时，按 RIFLEx（ICML 2025，arXiv 2502.15894，``thu-ml/RIFLEx`` 的
    ``get_1d_rotary_pos_embed_riflex``）把时间 RoPE 的固有频率降到整镜只占 ``margin`` 个周期。

    ``index`` 是 1 起的频率序号（Wan2.2 头维 128、时间 44 维、22 个频率；原生 121 帧＝31 潜帧，周期最接近的
    是第 5 个，约 33.6 潜帧）；``native_latent_frames`` 以内的窗和多镜窗都不改。来源 renderer.json 的
    ``inference.long_shot_rope``，三个字段都必填。"""

    index: int
    native_latent_frames: int
    margin: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LongShotRope":
        if not isinstance(value, Mapping) or set(value) != {"index", "native_latent_frames", "margin"}:
            raise ValueError("inference.long_shot_rope requires exactly index, native_latent_frames, margin")
        index, native, margin = value["index"], value["native_latent_frames"], value["margin"]
        if type(index) is not int or index < 1 or type(native) is not int or native < 1:
            raise ValueError("long_shot_rope index and native_latent_frames must be positive ints")
        if type(margin) is not float or not 0.0 < margin <= 1.0:
            raise ValueError("long_shot_rope margin must be a float in (0, 1]")
        return cls(index=index, native_latent_frames=native, margin=margin)

    def applies(self, latent_shot_groups: Sequence[Sequence[int]]) -> bool:
        return len(latent_shot_groups) == 1 and latent_shot_groups[0][1] - latent_shot_groups[0][0] > self.native_latent_frames

    def temporal_table(self, table: torch.Tensor, latent_frames: int) -> torch.Tensor:
        """``table`` 是 ``precompute_freqs_cis`` 的时间表（位置×频率，复数）。只换第 ``index`` 列的角速度。"""
        if table.ndim != 2 or not torch.is_complex(table) or self.index > table.shape[1]:
            raise ValueError("long_shot_rope needs a complex (positions, frequencies) table that has the index column")
        positions = torch.arange(table.shape[0], dtype=torch.float64, device=table.device)
        theta = self.margin * 2.0 * torch.pi / latent_frames
        out = table.clone()
        out[:, self.index - 1] = torch.polar(torch.ones_like(positions), positions * theta).to(table.dtype)
        return out


def encode_single_frame(runtime: Any, image: Any) -> torch.Tensor:
    """一张 RGB 帧 → 一张潜帧 ``[1, C, 1, H, W]``，与原生首帧口（``WanVideoUnit_ImageEmbedderFused``）
    同一种编码：单独一帧过因果VAE，不复用视频latent中的压缩帧。

    ``runtime`` 只需 ``pipe`` / ``device`` / ``dtype`` / ``pipeline_cfg`` 四个属性：出片运行时
    （:class:`Wan22InferenceRuntime`）与训练的 ``wan22_loader.VaeOnlyRuntime`` 都调这一处。
    训练worker传来的uint8 CHW与推理PIL图使用同一归一化顺序。"""

    if isinstance(image, torch.Tensor):
        if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("reference image must be uint8 CHW RGB")
        tensor = (image.to(device=runtime.device, dtype=runtime.dtype) * (2 / 255) - 1).unsqueeze(1)
    else:
        tensor = runtime.pipe.preprocess_image(image).transpose(0, 1)
    latent = runtime.pipe.vae.encode(
        [tensor.to(device=runtime.device, dtype=runtime.dtype)],
        device=runtime.device,
        tiled=runtime.pipeline_cfg["tiled"],
        tile_size=tuple(runtime.pipeline_cfg["tile_size"]),
        tile_stride=tuple(runtime.pipeline_cfg["tile_stride"]),
    )
    if latent.ndim != 5 or int(latent.shape[2]) != 1:
        raise RuntimeError(f"one RGB frame encoded to latent shape {tuple(latent.shape)}")
    return latent.detach().clone()


class Wan22InferenceRuntime:
    """P3 inference wrapper around the materialized DiffSynth multi-shot patch."""

    def __init__(
        self,
        *,
        model_cfg: Mapping[str, Any],
        inference_cfg: Mapping[str, Any],
        pipeline_cfg: Mapping[str, Any],
        dit_ckpt: str | None,
    ) -> None:
        self.model_cfg = dict(model_cfg)
        self.inference_cfg = dict(inference_cfg)
        self.pipeline_cfg = dict(pipeline_cfg)
        # generate_case 把本次的帧存在 last_generated_* 上：同一进程几部故事共用本运行时时，调用方持此锁直到读完。
        self.lock = threading.Lock()
        self.device = torch.device(self.inference_cfg["device"])
        self.dtype = torch.bfloat16
        self.temporal_position_mode = require_temporal_position_mode(
            self.model_cfg["temporal_position_mode"]
        )

        self.weight_source = self.inference_cfg["weight_source"]
        #: Wan2.2-A14B 两个去噪专家：高噪 ``pipe.dit``、低噪 ``pipe.dit2``，各挂训练时的同一份 LoRA，
        #: 去噪到 ``t < boundary·1000`` 换低噪（``mspc.models.noise_expert``）；单模型底座为 None。
        self.experts = self.inference_cfg.get("experts")
        self.pipe, _ = load_pure_t2v_pipeline(
            diffsynth_root=self.model_cfg["diffsynth_root"],
            model_paths=self.inference_cfg["model_paths"],
            self_attention_policy=self.model_cfg["self_attention_policy"],
            torch_dtype=self.dtype,
            device=self.device,
            low_noise_dit=None if self.experts is None else list(self.experts["low_noise_dit"]),
        )
        # 长单镜的时间 RoPE 修正是候选入口（renderer 补差距方案 11-12）：配置里没有这一项就不改任何位置编码。
        raw_long_shot = self.inference_cfg.get("long_shot_rope")
        self.long_shot_rope = None if raw_long_shot is None else LongShotRope.from_mapping(raw_long_shot)
        self.dit_checkpoint_path: Path | None = None
        # 记忆前缀：``model.memory_prefix`` 是这条臂的唯一事实源。它不加任何参数，所以
        # checkpoint 的键里看不出它在不在；与训练侧
        # ``Wan22DiffSynthTrainingModule.memory_prefix_enabled`` 同一判据——``null`` 就是
        # 「这条臂不存在」，没有第二个开关。
        raw_memory_prefix = self.model_cfg.get("memory_prefix")
        self.memory_prefix_options = (
            None
            if raw_memory_prefix is None
            else (
                raw_memory_prefix
                if isinstance(raw_memory_prefix, MemoryPrefixModelOptions)
                else MemoryPrefixModelOptions.from_mapping(raw_memory_prefix)
            )
        )
        raw_source_identity = self.model_cfg.get("source_identity")
        self.source_identity_options = (None if raw_source_identity is None else
            SourceIdentityOptions.from_mapping(raw_source_identity))
        self.pipe.dit.source_identity_options = self.source_identity_options
        if self.experts is not None:
            self.pipe.dit2.source_identity_options = self.source_identity_options
        from mspc.models.spatial_layout_inputs import SpatialLayoutOptions
        layout_options = self.model_cfg.get("spatial_layout")
        self.spatial_layout_options = (None if layout_options is None else
                                       SpatialLayoutOptions.from_mapping(layout_options))
        from mspc.models.region_attention import attach_layout_modules
        attach_layout_modules(self.pipe.dit, self.spatial_layout_options)
        from mspc.models.planner_tokens import PlannerTokenOptions, attach_planner_connector
        planner_options = self.model_cfg.get("planner_tokens")
        attach_planner_connector(self.pipe.dit, None if planner_options is None else
                                 PlannerTokenOptions.from_mapping(planner_options))
        from mspc.models.first_frame_tokens import attach_first_frame_tokens, first_frame_token_options
        attach_first_frame_tokens(self.pipe.dit, first_frame_token_options(self.model_cfg))
        #: 同一镜渲完后**末帧**的单帧编码 ``[1, C, 1, H, W]``（:meth:`encode_single_frame`），尾帧探针
        #: ``generated_prev_tail`` 的记忆来源。末帧不在第 0 潜帧里，只能从像素重编码，所以这一张
        #: 经过一次 uint8 往返；读尾帧对第 0 帧的差时要记住这一处不同。
        self.last_generated_last_latent: torch.Tensor | None = None
        if self.weight_source == "full_dit_checkpoint":
            checkpoint_path = require_path(dit_ckpt, "inference.dit_checkpoint")
            self.dit_checkpoint_path = checkpoint_path
            from safetensors.torch import load_file

            state_dict = load_file(str(checkpoint_path), device=str(self.device))
            load_dit_state_dict(self.pipe.dit, state_dict)
        elif self.weight_source == "expert_lora_checkpoints":
            from safetensors.torch import load_file

            from mspc.models.lora import inject_lora, load_lora_state_dict
            spec = self.experts["lora"]
            for dit, key in ((self.pipe.dit, "high_noise_lora"), (self.pipe.dit2, "low_noise_lora")):
                inject_lora(dit, rank=spec["rank"], alpha=float(spec["alpha"]), targets=spec["targets"])
                load_lora_state_dict(dit, load_file(str(require_path(self.experts[key], f"inference.experts.{key}")),
                                                    device=str(self.device)))
        elif self.weight_source != "pretrained_model_paths":
            raise ValueError(
                "unsupported inference.weight_source=" f"{self.weight_source!r}"
            )
        self.pipe.freeze_except([])
        self.pipe.dit.eval().requires_grad_(False)
        if self.experts is not None:
            self.pipe.dit2.eval().requires_grad_(False)
        for unit in self.pipe.units:
            if type(unit).__name__ == "WanVideoUnit_PromptEmbedder":
                unit.cache_inference_prompts()

    @property
    def memory_prefix_enabled(self) -> bool:
        """这条臂在不在。``model.memory_prefix: null`` 就是「不在」，没有第二个开关。

        注意它是**跑次级**的事实而不是逐样本的：M=0 的 case（无记忆臂、每条故事的第 0 镜）
        与训练一致，无参考且无锚点时走标量时间步。
        """

        return self.memory_prefix_options is not None

    def _normalize_case(
        self,
        *,
        prompts: Sequence[str],
        shot_groups: Sequence[Sequence[int]],
        latent_shot_groups: Sequence[Sequence[int]],
        memory_frame_count: int,
    ) -> tuple[str | list[str], str | list[str], int, dict[str, Any]]:
        # 前 M 行是记忆行（``row_policy`` 的形态由契约核）；只有其后的镜头行按逐镜文字的契约核。
        if memory_frame_count:
            require_memory_rows(
                prompts[:memory_frame_count], row_policy=self.memory_prefix_options.row_policy
            )
        prompts = list(prompts[:memory_frame_count]) + require_shot_prompts(
            list(prompts[memory_frame_count:]), len(shot_groups) - memory_frame_count
        )
        if len(shot_groups) != len(latent_shot_groups) or not shot_groups:
            raise ValueError("shot_groups and latent_shot_groups must be non-empty and equally sized")
        num_frames = shot_groups[-1][1]
        if type(num_frames) is not int or num_frames <= 0:
            raise ValueError("shot_groups must end at a positive integer frame count")
        if len(shot_groups) == 1:
            raw_start, raw_end = shot_groups[0]
            latent_start, latent_end = latent_shot_groups[0]
            expected_latent_frames = (num_frames - 1) // 4 + 1
            if (
                type(raw_start) is not int
                or type(raw_end) is not int
                or type(latent_start) is not int
                or type(latent_end) is not int
                or raw_start != 0
                or raw_end != num_frames
                or (num_frames - 1) % 4 != 0
                or latent_start != 0
                or latent_end != expected_latent_frames
            ):
                raise ValueError("K=1 inference case must be one native Wan 4n+1 segment")
            return (
                prompts[0],
                self.inference_cfg["negative_prompt"],
                num_frames,
                {},
            )

        negative_prompt = self.inference_cfg["negative_prompt"]
        # 记忆行在负分支上与正分支逐字相同（记忆是观测，不是引导信号，CFG 只作用在目标行上）；
        # 其余镜头行给配置的负向 prompt。
        negative_prompts = list(prompts[:memory_frame_count]) + [negative_prompt] * (
            len(prompts) - memory_frame_count
        )
        return (
            prompts,
            negative_prompts,
            num_frames,
            {
                "shot_groups": shot_groups,
                "latent_shot_groups": latent_shot_groups,
            },
        )

    def encode_single_frame(self, image: Any) -> torch.Tensor:
        """见模块级 :func:`encode_single_frame`；出片运行时与只装 VAE 的运行时共用同一份编码。"""

        return encode_single_frame(self, image)

    def generate_case(
        self,
        *,
        prompts: Sequence[str],
        shot_groups: Sequence[Sequence[int]],
        latent_shot_groups: Sequence[Sequence[int]],
        shot_frame_counts: Sequence[int],
        output_path: Path,
        seed: int,
        memory: MemoryPrefixInput | None = None,
        sdedit: SDEditInput | None = None,
        first_frame: torch.Tensor | None = None,
        first_frame_pin_until_sigma: float | None = None,
        capture_history: bool = False,
        target_layouts: Sequence[Mapping[str, Any]] | None = None,
        planner_tokens: Any = None,
    ) -> dict[str, Any]:
        """``planner_tokens`` 是这一段的 :class:`mspc.models.planner_tokens.PlannerTokenCondition`（没有 token 时 None）。"""
        if first_frame is None and first_frame_pin_until_sigma is not None:
            raise ValueError("first_frame_pin_until_sigma requires first_frame")
        if sdedit is not None and memory is not None:
            raise ValueError("SDEdit re-denoises a shot from its own frames; it takes no memory prefix")
        if first_frame is not None:
            # 原生首帧口只在冻结原生底座的单镜 case 上有定义：第 0 潜帧就是本镜第 0 帧。记忆臂
            # 另有前接的前缀，SDEdit 从本镜自己的帧重去噪——各是另一件事，混进来出的东西不属于
            # 任何一条 Declaration。
            if memory is not None or sdedit is not None:
                raise ValueError(
                    "first_frame is the native first-frame port; it takes no memory prefix "
                    "and no SDEdit start"
                )
            if self.memory_prefix_options is not None:
                raise ValueError(
                    "first_frame runs on a plain checkpoint (no memory prefix)"
                )
            if len(shot_groups) != 1:
                raise ValueError(
                    "first_frame feeds the native first-frame port of a single-shot case; got "
                    f"{len(shot_groups)} shot groups"
                )
        # 成片的镜头几何：记忆前缀前接之前的那一份（补到 4n+1 的 RGB 区间）。存盘前按它裁回
        # ``shot_frame_counts``（每镜剧本镜长），见 ``continuity_dev_32.crop_frames_to_shot_lengths``。
        target_shot_groups = [list(group) for group in shot_groups]
        target_latent_groups = [list(group) for group in latent_shot_groups]
        # 记忆前缀先建：它要在 ``_normalize_case`` 之前把 M 张记忆组前接到这一镜的几何上。
        # ``memory is None`` 在这条臂上是 M=0（记忆 dropout 形态／每条故事的第 0 镜），在别的
        # 臂上必须为 None——``build_memory_prefix_controller`` 负责这条判据。
        memory_controller = build_memory_prefix_controller(
            pipe=self.pipe,
            device=self.device,
            dtype=self.dtype,
            cfg_scale=float(self.inference_cfg["cfg_scale"]),
            memory_prefix_options=self.memory_prefix_options,
            memory_latents=None if memory is None else memory.latents,
            seed=seed,
            planner_tokens=planner_tokens,
            reference_routing=None if memory is None else memory.reference_routing,
        )
        memory_frame_count = 0
        if memory_controller is not None:
            memory_frame_count = memory_controller.memory_frame_count
            prompts, shot_groups, latent_shot_groups = memory_prefix_inference_layout(
                prompts=prompts,
                shot_groups=shot_groups,
                latent_shot_groups=latent_shot_groups,
                memory_shot_texts=() if memory is None else memory.shot_texts,
                row_policy=self.memory_prefix_options.row_policy,
            )
        prompt, negative_prompt, num_frames, multi_shot = self._normalize_case(
            prompts=prompts,
            shot_groups=shot_groups,
            latent_shot_groups=latent_shot_groups,
            memory_frame_count=memory_frame_count,
        )
        sampler = require_sampler(self.inference_cfg["sampler"])
        if sampler == FLOW_UNIPC_OFFICIAL:
            if not isinstance(self.pipe.scheduler, FlowUniPCOfficial):
                self.pipe.scheduler = FlowUniPCOfficial()
            rand_device = str(self.device)
        else:
            from diffsynth.diffusion.flow_match import FlowMatchScheduler

            if not isinstance(self.pipe.scheduler, FlowMatchScheduler):
                self.pipe.scheduler = FlowMatchScheduler("Wan")
            rand_device = "cpu"
        generation_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "reference_frames": memory_frame_count,
            "rand_device": rand_device,
            "sampler": sampler,
            "height": self.inference_cfg["target_height"],
            "width": self.inference_cfg["target_width"],
            "num_frames": num_frames,
            "cfg_scale": self.inference_cfg["cfg_scale"],
            "cfg_merge": False,
            "num_inference_steps": self.inference_cfg["num_inference_steps"],
            "sigma_shift": self.inference_cfg["sigma_shift"],
            "tiled": self.pipeline_cfg["tiled"],
            "tile_size": tuple(self.pipeline_cfg["tile_size"]),
            "tile_stride": tuple(self.pipeline_cfg["tile_stride"]),
            "sliding_window_size": None,
            "sliding_window_stride": None,
            "framewise_decoding": False,
            "temporal_position_mode": self.temporal_position_mode,
        }
        experts = getattr(self, "experts", None)
        if experts is not None:
            generation_kwargs["switch_DiT_boundary"] = float(experts["boundary"])
        generation_kwargs.update(multi_shot)
        if getattr(self, "source_identity_options", None) is not None and memory_frame_count:
            # 源身份路由要每张参考里逐人的源点；推理的参考身份只由分镜写（不读图），这组权重给参考出不了片。
            raise ValueError("source identity routing needs per-person source points read from each reference frame; "
                             "inference writes reference identities from the storyboard and reads no frames")
        options = getattr(self, "spatial_layout_options", None)
        if options is not None:
            # 两组布局权重都走 sdpa。没有 target_layouts 就是这份计划没有框，不编 B。
            generation_kwargs["text_attention_backend"] = options.text_attention_backend
            if target_layouts is not None:
                from mspc.models.spatial_layout_inputs import encode_spatial_layout_inputs
                generation_kwargs["spatial_layout"] = encode_spatial_layout_inputs(
                    self.pipe.tokenizer, target_layouts, prompts=prompts[memory_frame_count:],
                    frame_counts=[end - start for start, end in target_shot_groups], options=options)
        if sdedit is not None:
            if len(sdedit.frames) != num_frames:
                raise ValueError(f"SDEdit got {len(sdedit.frames)} frame(s) for a {num_frames}-frame shot")
            size = (self.inference_cfg["target_width"], self.inference_cfg["target_height"])
            generation_kwargs["input_video"] = [frame.resize(size) for frame in sdedit.frames]
            generation_kwargs["denoising_strength"] = sdedit.strength

        first_frame_controller = (
            None
            if first_frame is None
            else build_first_frame_controller(
                pipe=self.pipe,
                device=self.device,
                dtype=self.dtype,
                cfg_scale=float(self.inference_cfg["cfg_scale"]),
                first_frame=first_frame,
                pin_until_sigma=first_frame_pin_until_sigma,
            )
        )
        # 两个落点（``model_fn`` ＋ ``scheduler.step``）由控制器自己装卸，见
        # ``_CleanPrefixDenoiseState.install``。与它共存的控制器在上面与 __init__ 已被拒。
        clean_controller = (
            memory_controller if memory_controller is not None else first_frame_controller
        )
        if clean_controller is not None:
            clean_controller.install()
        # 测试替身绕过 __init__ 构造运行时，没有这个属性就是没配候选入口。
        rope = getattr(self, "long_shot_rope", None)
        long_shot = rope is not None and rope.applies(target_latent_groups)
        if long_shot:
            native_freqs = self.pipe.dit.freqs
            frames = target_latent_groups[0][1] - target_latent_groups[0][0]
            self.pipe.dit.freqs = (rope.temporal_table(native_freqs[0], frames), *native_freqs[1:])
        try:
            video = self.pipe(**generation_kwargs)
        finally:
            if long_shot:
                self.pipe.dit.freqs = native_freqs
            if clean_controller is not None:
                clean_controller.restore()
        if clean_controller is not None:
            clean_controller.assert_finished()
        if len(video) != num_frames:
            raise RuntimeError(
                f"DiffSynth produced {len(video)} frames, expected requested {num_frames}"
            )
        if memory_controller is not None:
            # 记忆帧是观测，不是成片。管线按 shot 组逐组解码，前 M 组各是一张记忆帧
            # （一潜帧 → 一 RGB 帧），目标镜由 ``latents[:, :, M:]`` 独立解出，所以这里
            # 只是把那 M 帧去掉，目标镜的像素与它们无关。
            video = video[memory_controller.memory_frame_count :]
        video = crop_frames_to_shot_lengths(
            video,
            shot_groups=target_shot_groups,
            shot_frame_counts=shot_frame_counts,
            label=str(output_path),
        )
        if capture_history:
            # 每镜 first/middle/last 三候选帧（与训练参考帧库同一取帧规则），由调用方去重入库。
            from mspc.data.reference_frame_latents import shot_candidates

            # 整镜解码帧也留着：看历史视频选帧的规划器从中按秒取参考帧（不从压缩后的 mp4 回读）。
            self.last_generated_shot_frames = []
            self.last_generated_shot_videos = []
            cursor = 0
            for count in shot_frame_counts:
                self.last_generated_shot_frames.append(shot_candidates(video[cursor:cursor + count]))
                self.last_generated_shot_videos.append(list(video[cursor:cursor + count]))
                cursor += count
        if clean_controller is not None:
            # 末帧的单帧编码（与原生首帧口同一种编码）：续拍镜的首帧取它。
            self.last_generated_last_latent = self.encode_single_frame(video[-1])
        save_video(
            video,
            output_path,
            fps=self.inference_cfg["fps"],
            quality=self.inference_cfg["video_quality"],
        )
        readout: dict[str, Any] = {}
        if long_shot:
            readout["long_shot_rope"] = {"index": rope.index, "latent_frames": frames, "margin": rope.margin}
        if first_frame_controller is not None:
            readout["first_frame_fed"] = True
        if memory_controller is not None:
            # 三条臂里只有本镜的 M 是运行时才知道的（第 0 镜恒为 0），落在这里事后才分得清
            # 「无记忆臂」与「有记忆臂但漏传了记忆」。
            readout.update(
                {
                    "memory_frame_count": memory_controller.memory_frame_count,
                    "actual_reference_noise_sigma": (
                        self.memory_prefix_options.noise.inference_sigma if memory_frame_count else 0.0
                    ),
                    "reference_timestep": memory_controller.reference_timestep,
                    "reference_offset": memory_controller.forward_kwargs.get(
                        "reference_offset"
                    ),
                    "memory_attention_policy": memory_controller.forward_kwargs.get(
                        "memory_attention_policy"
                    ),
                    "reference_routing": memory_controller.forward_kwargs.get("reference_routing"),
                    "memory_attention_window": {
                        "layer_indices": list(self.memory_prefix_options.attention_window.layer_indices),
                        "timestep_range": list(self.memory_prefix_options.attention_window.timestep_range),
                    },
                }
            )
        return readout

def build_config(
    config: ExperimentConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    from mspc.inference.common import build_inference_dataset_config

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    data = config.resolved
    model = data["model"]
    inference = data["inference"]
    if data["model"]["type"] != "wan22_diffsynth":
        raise ValueError("config model.type must be 'wan22_diffsynth'")
    if not inference["enabled"]:
        raise ValueError("config inference.enabled must be true for Wan2.2 inference")
    model_paths = require_wan22_model_paths(model["paths"])
    model_cfg = dict(model)
    dataset_cfg = build_inference_dataset_config(config, require_case_id_column=True)
    inference_cfg = {
        key: inference[key]
        for key in (
            "device",
            "target_height",
            "target_width",
            "fps",
            "seed",
            "negative_prompt",
            "num_inference_steps",
            "sigma_shift",
            "cfg_scale",
            "video_quality",
            "weight_source",
            "sampler",
            "experts",
        )
    }
    require_sampler(inference_cfg["sampler"])
    inference_cfg["model_paths"] = model_paths
    pipeline_cfg = data["pipeline_inputs"]
    pipeline_cfg = dict(pipeline_cfg)
    return model_cfg, dataset_cfg, inference_cfg, pipeline_cfg


def parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument("--config", required=True, type=Path)
    return arg_parser


def run(config: ExperimentConfig) -> int:
    from mspc.inference.common import checkpoint_step_from_path
    from mspc.inference.sampler import MultishotInferenceSampler, default_manifest_row

    model_cfg, _, inference_cfg, pipeline_cfg = build_config(config)
    runtime = Wan22InferenceRuntime(
        model_cfg=model_cfg,
        inference_cfg=inference_cfg,
        pipeline_cfg=pipeline_cfg,
        dit_ckpt=config.resolved["inference"]["dit_checkpoint"],
    )
    sampler = MultishotInferenceSampler(config, require_case_id_column=True)

    def generate_case(
        *,
        case_id: str,
        case_inputs: dict[str, Any],
        output_path: Path,
        seed: int,
        source_row: Any,
    ) -> dict[str, Any]:
        provenance = runtime.generate_case(
            prompts=case_inputs["prompts"],
            shot_groups=case_inputs["shot_groups"],
            latent_shot_groups=case_inputs["latent_shot_groups"],
            # 训练 CSV 窗的镜长本来就是 4n+1（数据侧补齐），没有更短的剧本镜长可裁。
            shot_frame_counts=[end - start for start, end in case_inputs["shot_groups"]],
            output_path=output_path,
            seed=seed,
        )
        extra_fields: dict[str, Any] = {
            "conditioning": "pure_t2v",
            "prompt_contract": RENDERED_PROMPT_CONTRACT,
            **provenance,
        }
        if runtime.dit_checkpoint_path is not None:
            extra_fields["dit_checkpoint_path"] = str(runtime.dit_checkpoint_path)
            checkpoint_step = checkpoint_step_from_path(runtime.dit_checkpoint_path)
            if checkpoint_step is not None:
                extra_fields["dit_checkpoint_step"] = checkpoint_step
        return default_manifest_row(
            case_id=case_id,
            output_path=output_path,
            generation_paradigm=(
                "wan22_diffsynth_pure_t2v_multishot__"
                f"{model_cfg['temporal_position_mode']}"
            ),
            case_inputs=case_inputs,
            source_row=source_row,
            extra_fields=extra_fields,
        )

    sampler.run(generate_case)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return run(load_experiment_config(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
