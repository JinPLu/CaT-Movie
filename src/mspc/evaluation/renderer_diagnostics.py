"""现有渲染器的能力诊断：短面板、逐层读数及仅评估使用的跨镜 KV 切断。"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import MethodType
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

import torch


def own_shot_attention(self, x, freqs, latent_shot_groups=None, tokens_per_frame=None,
                       memory_attention_policy=None, prefix_frames=0):
    """同一 Q/K/V/O 与 RoPE，仅去掉目标镜外 KV；不接入生产模型策略。"""
    from diffsynth.models.wan_video_dit import rope_apply
    if self.self_attention_policy not in ("full", "holocine_first_frame"):
        # 首帧通信（及其孪生帧）的镜内可见关系不同：其余潜帧本就不读他镜，首帧组另有重复键与孪生帧。
        raise ValueError(f"own-shot diagnostic covers full and holocine_first_frame, not {self.self_attention_policy!r}")
    if prefix_frames or memory_attention_policy is not None:
        raise ValueError("own-shot diagnostic is only defined without references")
    if not latent_shot_groups or not tokens_per_frame:
        raise ValueError("own-shot diagnostic requires explicit shot geometry")
    q = rope_apply(self.norm_q(self.q(x)), freqs, self.num_heads)
    k = rope_apply(self.norm_k(self.k(x)), freqs, self.num_heads)
    v = self.v(x)
    ranges = [(a * tokens_per_frame, b * tokens_per_frame) for a, b in latent_shot_groups]
    if ranges[0][0] != 0 or ranges[-1][1] != x.shape[1] or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError("shot ranges do not partition attention tokens")
    outputs = []
    for a, b in ranges:
        own_k, own_v = k[:, a:b], v[:, a:b]
        if self.self_attention_policy == "holocine_first_frame":
            # 正常 Holo 路由重复本镜首帧；切断外镜时仍保留它，避免同时改变镜内权重。
            own_k = torch.cat([own_k, k[:, a:a + tokens_per_frame]], dim=1)
            own_v = torch.cat([own_v, v[:, a:a + tokens_per_frame]], dim=1)
        outputs.append(self.attn(q[:, a:b].contiguous(), own_k.contiguous(), own_v.contiguous()))
    return self.o(torch.cat(outputs, dim=1))


@contextmanager
def communication_intervention(dit, intervention):
    if intervention not in {"normal", "own_shot"}:
        raise ValueError(intervention)
    originals = []
    if intervention == "own_shot":
        for block in dit.blocks:
            module = block.self_attn
            originals.append((module, module.forward))
            module.forward = MethodType(own_shot_attention, module)
    try:
        yield
    finally:
        for module, forward in originals:
            module.forward = forward


class FeatureReadout:
    """在首/中/末去噪步的正负分支记录三层；hook只读，不替换输出。"""
    def __init__(self, dit, path: Path, latent_groups, steps: int):
        self.groups = latent_groups
        self.call = 0
        self.selected_calls = {1, 2, steps // 2 * 2 - 1, steps // 2 * 2, steps * 2 - 1, steps * 2}
        self.handle = path.open("w")
        self.hooks = [dit.blocks[0].register_forward_pre_hook(self._advance)]
        for index in sorted({0, len(dit.blocks) // 2, len(dit.blocks) - 1}):
            block = dit.blocks[index]
            self.hooks.append(block.self_attn.register_forward_pre_hook(self._attention_hook(index), with_kwargs=True))
            for name, module in (("hidden", block), ("self_attention", block.self_attn), ("cross_attention", block.cross_attn)):
                self.hooks.append(module.register_forward_hook(self._hook(index, name)))

    def _advance(self, module, inputs):
        self.call += 1

    def _attention_hook(self, layer):
        def record(module, inputs, kwargs):
            if self.call not in self.selected_calls:
                return
            rows = sampled_attention_mass(module, inputs[0], inputs[1], kwargs["latent_shot_groups"],
                kwargs["tokens_per_frame"], kwargs["memory_attention_policy"], kwargs["prefix_frames"],
                own_shot=getattr(module.forward, "__func__", None) is own_shot_attention)
            self.handle.write(json.dumps({"call": self.call, "layer": layer, "kind": "sampled_attention_mass",
                "queries_per_group": 8, "rows": rows}) + "\n")
        return record

    def _hook(self, layer, name):
        def record(module, inputs, output):
            if self.call not in self.selected_calls:
                return
            output = output if name == "self_attention" else output.hidden_states
            frames = self.groups[-1][1]
            if output.shape[1] % frames:
                raise ValueError("feature tokens do not match latent frames")
            stride = output.shape[1] // frames
            with torch.no_grad():
                values = []
                for start, end in self.groups:
                    value = output[:, start * stride:end * stride].detach().float()
                    values.append({"rms": float(value.square().mean().sqrt()),
                                   "mean": value.mean((0, 1)).cpu().tolist()})
            self.handle.write(json.dumps({"call": self.call, "cfg_branch": "positive" if self.call % 2 else "negative",
                                          "layer": layer, "kind": name, "shots": values}) + "\n")
            self.handle.flush()
        return record

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.handle.close()


@torch.no_grad()
def sampled_attention_mass(module, x, freqs, groups, tokens_per_frame, memory_policy, prefix_frames, *, own_shot):
    """各镜等距抽至多8个查询，用现有BF16 Q/K重建FP32 softmax的分组质量。

    这是抽样重建，非全查询注意力，也不是人物区域或因果贡献。保留Holo重复首帧计数。
    """
    if module.self_attention_policy not in ("full", "holocine_first_frame"):
        # 首帧通信下只有首潜帧读他镜；按全组或 HoloCine 读法重建会把可见关系算错。
        raise ValueError(f"attention-mass readout covers full and holocine_first_frame, "
                         f"not {module.self_attention_policy!r}")
    from diffsynth.models.wan_video_dit import rope_apply
    ranges = [(a * tokens_per_frame, b * tokens_per_frame) for a, b in groups]
    k = rope_apply(module.norm_k(module.k(x)), freqs, module.num_heads)
    k = k.reshape(1, x.shape[1], module.num_heads, module.head_dim).transpose(1, 2).float()
    rows = []
    for group, (a, b) in enumerate(ranges):
        selected = torch.linspace(a, b - 1, min(8, b - a), device=x.device).long()
        q = rope_apply(module.norm_q(module.q(x[:, selected])), freqs[selected], module.num_heads)
        q = q.reshape(1, len(selected), module.num_heads, module.head_dim).transpose(1, 2).float()
        if module.self_attention_policy == "holocine_first_frame":
            key_ranges = [(group, a, b), *[(i, c, c + tokens_per_frame) for i, (c, _) in enumerate(ranges)
                                         if not own_shot or i == group]]
        elif own_shot or (memory_policy == "memory_self_only_v1" and group < prefix_frames):
            key_ranges = [(group, a, b)]
        else:
            key_ranges = [(i, c, d) for i, (c, d) in enumerate(ranges)]
        keys = torch.cat([k[:, :, c:d] for _, c, d in key_ranges], dim=2)
        probabilities = ((q @ keys.transpose(-2, -1)) * (module.head_dim ** -0.5)).softmax(-1)
        masses = [0.0] * len(groups)
        cursor = 0
        for target, c, d in key_ranges:
            masses[target] += float(probabilities[..., cursor:cursor + d - c].sum(-1).mean())
            cursor += d - c
        rows.append(masses)
    return rows


def checkpoint_parameter_changes(dit, checkpoint: Path, output: Path):
    """逐张量比较当前原生参数与训练检查点；只将分层统计落盘。"""
    from safetensors import safe_open
    rows = []
    with safe_open(str(checkpoint), framework="pt", device="cpu") as saved:
        parameters = dict(dit.named_parameters())
        if set(saved.keys()) != set(parameters):
            raise ValueError("checkpoint keys differ from current native DiT parameters")
        for name, parameter in parameters.items():
            base = parameter.detach().float().cpu()
            trained = saved.get_tensor(name).float()
            delta = trained - base
            base_square = float(base.square().sum(dtype=torch.float64))
            delta_square = float(delta.square().sum(dtype=torch.float64))
            rows.append({"parameter": name, "elements": base.numel(), "base_square": base_square,
                         "delta_square": delta_square, "delta_max_abs": float(delta.abs().max())})
    output.write_text(json.dumps(rows, indent=2))


# 以下能力只由诊断入口安装；生产 forward 不增加开关。
def _indices(values, size, label):
    if not isinstance(values, (list, tuple)) or any(type(v) is not int for v in values):
        raise TypeError(f"{label} must be an explicit list/tuple of integers")
    if len(set(values)) != len(values) or any(v < 0 or v >= size for v in values):
        raise ValueError(f"{label} contains duplicate or out-of-range indices (size={size})")
    return tuple(sorted(values))


class DiagnosticStep:
    """零基步号；CFG 正负分支共享步号，激活按分支分开存。"""
    def __init__(self):
        self.step = None
        self.branch = None

    def active(self, steps):
        if self.step is None or self.branch is None:
            raise RuntimeError("diagnostic model_fn has not registered this call")
        return self.step in steps


@contextmanager
def diagnostic_steps(pipe, state, *, cfg_scale, start_step, step_count, positive_schedule=None):
    """包装 model_fn 登记调用；禁止合并 CFG/缓存，正常退出核调用数。"""
    _indices([start_step], len(pipe.scheduler.timesteps), "start_step")
    if type(step_count) is not int or step_count < 1 or start_step + step_count > len(pipe.scheduler.timesteps):
        raise ValueError("invalid step_count")
    original = pipe.model_fn
    calls = 0
    branches = 1 if cfg_scale == 1 else 2

    def forward(**kwargs):
        nonlocal calls
        step = start_step + calls // branches
        if step >= start_step + step_count:
            raise RuntimeError("too many diagnostic model calls")
        if kwargs.get("cfg_merge", False) or kwargs.get("tea_cache") is not None:
            raise ValueError("diagnostics require separate CFG calls and no TeaCache")
        expected = pipe.scheduler.timesteps[step].to(kwargs["timestep"])
        if not torch.equal(kwargs["timestep"].reshape(-1), expected.reshape(-1)):
            raise ValueError("model timestep differs from registered scheduler step")
        state.step = step
        state.branch = "positive" if calls % branches == 0 else "negative"
        calls += 1
        if positive_schedule is not None and state.branch == "positive":
            kwargs["context"] = positive_schedule.at(step)
        return original(**kwargs)

    pipe.model_fn = forward
    try:
        yield state
        if calls != step_count * branches:
            raise RuntimeError(f"expected {step_count * branches} model calls, got {calls}")
    finally:
        pipe.model_fn = original
        state.step = state.branch = None


class PositiveTextSchedule:
    def __init__(self, context_a, context_b, *, switch_step, total_steps):
        if type(total_steps) is not int or total_steps < 1:
            raise ValueError("total_steps must be positive")
        if type(switch_step) is not int or not 0 <= switch_step <= total_steps:
            raise ValueError("switch_step must be in [0, total_steps]")
        if context_a.shape != context_b.shape:
            raise ValueError("A/B contexts must have the same shot/token geometry")
        self.a, self.b = context_a, context_b
        self.switch_step, self.total_steps = switch_step, total_steps

    def at(self, step):
        _indices([step], self.total_steps, "step")
        return self.a if step < self.switch_step else self.b


@contextmanager
def _replace_forwards(replacements):
    originals = []
    try:
        for module, forward in replacements:
            # 删除临时实例属性，以恢复 nn.Module 原来的方法查找结构。
            had = "forward" in module.__dict__
            originals.append((module, module.forward, had))
            module.forward = MethodType(forward, module)
        yield
    finally:
        for module, forward, had in reversed(originals):
            if had:
                module.forward = forward
            else:
                del module.__dict__["forward"]


def _window(dit, layers, steps):
    return _indices(layers, len(dit.blocks), "layers"), _indices(steps, 50, "steps")


@contextmanager
def self_attention_key_mask(dit, state, *, layers, steps, query_key_sets):
    """每组 (query 下标, 被禁 key 下标)，query 组不重叠；只支持 full、无前缀。

    改被屏蔽 token 的输入仍可能改它自身的 query；不变性只针对该组 query 的 KV 路径。
    """
    from diffsynth.models.wan_video_dit import rope_apply
    layers, steps = _window(dit, layers, steps)
    if any(b.self_attn.self_attention_policy != "full" for b in dit.blocks):
        raise ValueError("key masking requires self_attention_policy=full")

    def wrap(original, layer):
        def forward(module, x, freqs, latent_shot_groups=None, tokens_per_frame=None,
                    memory_attention_policy=None, prefix_frames=0):
            if prefix_frames or memory_attention_policy is not None:
                raise ValueError("key masking requires no reference prefix")
            args = (x, freqs, latent_shot_groups, tokens_per_frame, memory_attention_policy, prefix_frames)
            active = state.active(steps) and layer in layers
            used, groups = set(), []
            for queries, blocked in query_key_sets:
                queries = _indices(queries, x.shape[1], "queries")
                blocked = _indices(blocked, x.shape[1], "keys")
                if used.intersection(queries):
                    raise ValueError("query groups overlap")
                used.update(queries)
                if queries and blocked:
                    allowed = sorted(set(range(x.shape[1])) - set(blocked))
                    if not allowed:
                        raise ValueError("cannot mask every key")
                    groups.append((queries, allowed))
            if not active or not groups:
                return original(*args)
            q = rope_apply(module.norm_q(module.q(x)), freqs, module.num_heads)
            k = rope_apply(module.norm_k(module.k(x)), freqs, module.num_heads)
            v = module.v(x)
            affected = {i for queries, _ in groups for i in queries}
            rest = sorted(set(range(x.shape[1])) - affected)
            if rest:
                groups.append((rest, list(range(x.shape[1]))))
            out = torch.empty_like(v)
            for queries, allowed in groups:
                out[:, queries] = module.attn(q[:, queries].contiguous(), k[:, allowed].contiguous(),
                                              v[:, allowed].contiguous())
            return module.o(out)
        return forward

    with _replace_forwards([(b.self_attn, wrap(b.self_attn.forward, i)) for i, b in enumerate(dit.blocks)]):
        yield


@contextmanager
def cross_attention_token_mask(dit, state, *, layers, steps, shot_tokens):
    """shot_tokens 为 {零基镜号: 正向文字的 token 下标}；只屏蔽 CFG 正向分支。

    负向提示是另一段文字，同一下标指向的不是同一个词，所以负向分支保持原样。
    """
    layers, steps = _window(dit, layers, steps)
    from mspc.models.source_identity import WanTextAttentionPrediction
    if any(b.cross_attn.has_image_input for b in dit.blocks):
        raise ValueError("text masking requires pure T2V cross attention")

    def wrap(original, layer):
        def forward(module, x, y, shot_token_ranges=None, spatial_layout_biases=None,
                    text_attention_backend="native", source_identity_condition=None):
            ranges = [(0, x.shape[1])] if shot_token_ranges is None else shot_token_ranges
            if x.shape[0] != 1 or y.shape[0] != len(ranges):
                raise ValueError("expected one video and one context per shot")
            if ranges[0][0] != 0 or ranges[-1][1] != x.shape[1] or any(a >= b for a, b in ranges) or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:])):
                raise ValueError("shot token ranges must partition the sequence")
            _indices(list(shot_tokens), len(ranges), "shots")
            blocked = {i: _indices(tokens, y.shape[1], "text tokens") for i, tokens in shot_tokens.items()}
            if any(len(tokens) == y.shape[1] for tokens in blocked.values()):
                raise ValueError("cannot mask every text token")
            if (not state.active(steps) or state.branch != "positive" or layer not in layers
                    or not any(blocked.values())):
                return original(x, y, shot_token_ranges, spatial_layout_biases, text_attention_backend,
                                source_identity_condition)
            if spatial_layout_biases is not None or text_attention_backend != "native" or source_identity_condition is not None:
                raise ValueError("text knockout requires ordinary native CA without layout or source routing")
            q, k, v = module.norm_q(module.q(x)), module.norm_k(module.k(y)), module.v(y)
            outputs = []
            for i, (a, b) in enumerate(ranges):
                allowed = sorted(set(range(y.shape[1])) - set(blocked.get(i, ())))
                outputs.append(module.attn(q[:, a:b].contiguous(), k[i:i+1, allowed].contiguous(),
                                           v[i:i+1, allowed].contiguous()))
            return WanTextAttentionPrediction(module.o(torch.cat(outputs, dim=1)), None)
        return forward

    with _replace_forwards([(b.cross_attn, wrap(b.cross_attn.forward, i)) for i, b in enumerate(dit.blocks)]):
        yield


@contextmanager
def activation_patch(dit, state, *, layers, steps, tokens, records, mode):
    """记录/替换整个 block 输出；键为 (步, 层, CFG 分支)，records 留原精度 CPU 张量。"""
    layers, steps = _window(dit, layers, steps)
    from mspc.models.source_identity import WanVideoBlockPrediction
    if mode not in {"record", "patch"}:
        raise ValueError("mode must be record or patch")

    def wrap(original, layer):
        def forward(module, *args, **kwargs):
            out = original(*args, **kwargs)
            values = out.hidden_states if isinstance(out, WanVideoBlockPrediction) else out
            if not isinstance(values, torch.Tensor):
                raise TypeError("activation patch requires tensor or Wan block prediction")
            indices = _indices(tokens, values.shape[1], "patch tokens")
            if not state.active(steps) or not indices:
                return out
            key = (state.step, layer, state.branch)
            if mode == "record":
                if key in records:
                    raise ValueError(f"duplicate activation: {key}")
                records[key] = (indices, values[:, indices].detach().cpu().clone())
                return out
            source_indices, value = records[key]
            if source_indices != indices or value.shape != values[:, indices].shape:
                raise ValueError("source/target activation geometry differs")
            patched = values.clone()
            patched[:, indices] = value.to(values)
            return out._replace(hidden_states=patched) if isinstance(out, WanVideoBlockPrediction) else patched
        return forward

    with _replace_forwards([(dit.blocks[i], wrap(dit.blocks[i].forward, i)) for i in layers]):
        yield


def pixel_masks_to_token_indices(masks, frame_indices, *, num_frames):
    """稀疏 bool [N,H,W] 像素掩码→展平 (潜帧,行,列) token 下标。

    4n+1 帧的因果 VAE：第 0 帧独占潜帧 0，帧 1..4→1、5..8→2。
    空间压缩 16、patch 1×2×2，即每 token 的像素格为 32×32；任意覆盖即入选。
    同潜帧多个掩码取并集；未提供的帧不插值、不延展。480×832→15×26。
    这是明确的离散分箱规则，不声称等于 VAE 卷积的真实感受野。
    """
    if type(num_frames) is not int or num_frames < 1 or (num_frames - 1) % 4:
        raise ValueError("num_frames must be 4n+1")
    if masks.ndim != 3 or masks.dtype != torch.bool:
        raise ValueError("masks must be bool [N,H,W]")
    _indices(frame_indices, num_frames, "frame_indices")
    n, h, w = masks.shape
    if n != len(frame_indices) or h < 32 or w < 32 or h % 32 or w % 32:
        raise ValueError("mask geometry must match frames and a 32-pixel grid")
    grid = torch.zeros(((num_frames - 1) // 4 + 1, h // 32, w // 32), dtype=torch.bool, device=masks.device)
    spatial = masks.reshape(n, h // 32, 32, w // 32, 32).any(4).any(2)
    for mask, frame in zip(spatial, frame_indices):
        grid[(frame + 3) // 4] |= mask
    return grid.flatten().nonzero().flatten().cpu().tolist()


def pool_block_features(output, token_grid, entity_tokens):
    f, h, w = token_grid
    if output.ndim != 3 or output.shape[1] != f * h * w or min(token_grid) < 1:
        raise ValueError("block output does not match token_grid")
    grid = output.reshape(output.shape[0], f, h, w, output.shape[-1]).float()
    values = {"center": grid[:, :, h // 2, w // 2].mean(1),
              "spatial_mean": grid.mean((2, 3)).mean(1)}
    for name, tokens in entity_tokens.items():
        indices = _indices(tokens, f * h * w, f"entity {name}")
        if not indices:
            raise ValueError(f"entity {name} has no observed token")
        values[f"entity.{name}"] = output[:, indices].float().mean(1)
    return values


def atlas_scheduler():
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    scheduler = FlowMatchScheduler("Wan")
    scheduler.set_timesteps(50, shift=5.0)
    return scheduler


def noisy_target(z0, *, seed, step):
    _indices([step], 50, "step")
    if type(seed) is not int:
        raise TypeError("seed must be int")
    scheduler = atlas_scheduler()
    # 同生产 generate_noise：CPU float32 随机数，再转运行精度。
    noise = torch.randn(z0.shape, generator=torch.Generator(device="cpu").manual_seed(seed),
                        dtype=torch.float32).to(z0)
    sigma = scheduler.sigmas[step]
    return (1 - sigma) * z0 + sigma * noise, sigma


@torch.no_grad()
def encode_atlas_text(pipe, prompts):
    from diffsynth.pipelines.wan_video import WanVideoUnit_PromptEmbedder
    for prompt in prompts if isinstance(prompts, (list, tuple)) else [prompts]:
        if not isinstance(prompt, str):
            raise TypeError("atlas prompt must be a string")
        tokens = pipe.tokenizer.tokenizer(pipe.tokenizer._clean(prompt),
                                         add_special_tokens=True, truncation=False)["input_ids"]
        if len(tokens) > pipe.tokenizer.seq_len:
            raise ValueError("prompt exceeds text capacity")
    return WanVideoUnit_PromptEmbedder().process(pipe, prompts, True)["context"]


@torch.no_grad()
def single_step_features(pipe, *, x_t, positive, negative, cfg_scale, step, state, entity_tokens,
                         temporal_position_mode):
    """一次日程步（CFG≠1 时两次 DiT forward），不跑采样轨迹；保存两分支特征。"""
    _indices([step], 50, "step")
    if x_t.ndim != 5 or x_t.shape[0] != 1 or x_t.shape[3] % 2 or x_t.shape[4] % 2:
        raise ValueError("x_t must be [1,C,F,even H,even W]")
    if positive.shape[0] != 1 or negative.shape != positive.shape:
        raise ValueError("single-step extraction requires one shot with matching CFG contexts")
    grid = (x_t.shape[2], x_t.shape[3] // 2, x_t.shape[4] // 2)
    scheduler = atlas_scheduler()
    original_scheduler = pipe.scheduler
    pipe.load_models_to_device(["dit"])
    tensors, handles = {}, []

    def hook(layer):
        def record(module, args, output):
            for name, value in pool_block_features(output.hidden_states, grid, entity_tokens).items():
                tensors[f"{state.branch}.layer.{layer}.{name}"] = value.detach().cpu()
        return record

    try:
        pipe.scheduler = scheduler
        for i, block in enumerate(pipe.dit.blocks):
            handles.append(block.register_forward_hook(hook(i)))
        timestep = pipe.scheduler.timesteps[step].reshape(1).to(x_t)
        with diagnostic_steps(pipe, state, cfg_scale=cfg_scale, start_step=step, step_count=1):
            kwargs = dict(dit=pipe.dit, latents=x_t, timestep=timestep,
                          temporal_position_mode=temporal_position_mode)
            positive_v = pipe.model_fn(**kwargs, context=positive).noise
            if cfg_scale == 1:
                v = positive_v
            else:
                negative_v = pipe.model_fn(**kwargs, context=negative).noise
                v = negative_v + cfg_scale * (positive_v - negative_v)
    finally:
        for handle in handles:
            handle.remove()
        pipe.scheduler = original_scheduler
    sigma = scheduler.sigmas[step]
    tensors.update(x_t=x_t.detach().cpu(), v=v.detach().cpu(), x0=(x_t - sigma * v).detach().cpu())
    return tensors, {"step": step, "sigma": float(sigma), "token_grid": list(grid),
                     "layers": list(range(len(pipe.dit.blocks))), "cfg_scale": cfg_scale,
                     "feature_branches": ["positive"] if cfg_scale == 1 else ["positive", "negative"],
                     "schedule_steps": 50, "schedule_shift": 5.0, "step_indexing": "zero_based"}


def save_atlas_features(path, tensors, metadata):
    from safetensors.torch import save
    # xb 原子拒绝覆盖，包括两个进程同时写同一路径的情况。
    encoded = save({k: v.to(dtype=torch.float16, device="cpu").contiguous() for k, v in tensors.items()},
                   metadata={k: json.dumps(v, ensure_ascii=False) for k, v in metadata.items()})
    with Path(path).open("xb") as stream:
        stream.write(encoded)


# 清单在加载模型/创建目录之前整体验证，禁止隐含参数和拼写错误。
class _AtlasSpec(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class AtlasLatent(_AtlasSpec):
    path: str
    key: str
    frame_start: int = Field(ge=0)
    frame_end: int = Field(gt=0)

    @model_validator(mode="after")
    def valid_range(self):
        if self.frame_end <= self.frame_start:
            raise ValueError("latent frame_end must exceed frame_start")
        return self

    def read(self):
        from safetensors import safe_open
        with safe_open(self.path, framework="pt", device="cpu") as f:
            shape = f.get_slice(self.key).get_shape()
            if len(shape) != 5 or shape[0] != 1 or self.frame_end > shape[2]:
                raise ValueError("latent slice is outside [1,C,F,H,W]")
            return f.get_slice(self.key)[:, :, self.frame_start:self.frame_end].clone()


class AtlasExtractUnit(_AtlasSpec):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    latent: AtlasLatent
    prompt: str = Field(min_length=1)
    seed: int = Field(ge=0)
    step: int = Field(ge=0, lt=50)
    entity_tokens: dict[str, list[int]]

    @model_validator(mode="after")
    def valid_entities(self):
        for name, indices in self.entity_tokens.items():
            if not name or not indices or any(i < 0 for i in indices) or len(indices) != len(set(indices)):
                raise ValueError("entities need a name and nonempty unique nonnegative token indices")
        return self


class AtlasSelfMask(_AtlasSpec):
    kind: Literal["self_mask"]
    layers: list[int]
    steps: list[int]
    query_key_sets: list[list[list[int]]]

    @model_validator(mode="after")
    def valid_pairs(self):
        if any(len(pair) != 2 for pair in self.query_key_sets):
            raise ValueError("each query_key_sets entry must be [queries, blocked_keys]")
        return self


class AtlasShotMask(_AtlasSpec):
    shot: int = Field(ge=0)
    tokens: list[int]


class AtlasCrossMask(_AtlasSpec):
    kind: Literal["cross_mask"]
    layers: list[int]
    steps: list[int]
    shot_tokens: list[AtlasShotMask]

    @model_validator(mode="after")
    def unique_shots(self):
        if len({s.shot for s in self.shot_tokens}) != len(self.shot_tokens):
            raise ValueError("duplicate shot mask")
        return self


class AtlasPatch(_AtlasSpec):
    kind: Literal["activation_patch"]
    layers: list[int]
    steps: list[int]
    tokens: list[int]
    source: Literal["horizontal_flip"]


class AtlasInterveneUnit(AtlasExtractUnit):
    intervention: Annotated[AtlasSelfMask | AtlasCrossMask | AtlasPatch, Field(discriminator="kind")]

    @model_validator(mode="after")
    def valid_window(self):
        _indices(self.intervention.layers, 30, "layers")
        _indices(self.intervention.steps, 50, "steps")
        return self


class AtlasScheduleUnit(_AtlasSpec):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    prompts_a: list[str]
    prompts_b: list[str]
    frame_counts: list[int]
    seed: int = Field(ge=0)
    switch_step: int = Field(ge=0, le=50)

    @model_validator(mode="after")
    def valid_shots(self):
        if not self.prompts_a or len(self.prompts_a) != len(self.prompts_b) or len(self.prompts_a) != len(self.frame_counts):
            raise ValueError("A/B/frame_counts must have equal nonzero shot counts")
        if any(n < 1 for n in self.frame_counts) or any(not p for p in self.prompts_a + self.prompts_b):
            raise ValueError("shots need positive frame counts and nonempty prompts")
        return self


class AtlasManifest(_AtlasSpec):
    model: Literal["Wan2.2-TI2V-5B"]
    units: list[dict]


def load_atlas_manifest(path, command):
    manifest = AtlasManifest.model_validate_json(Path(path).read_text())
    kind = {"extract": AtlasExtractUnit, "intervene": AtlasInterveneUnit, "schedule": AtlasScheduleUnit}[command]
    units = [kind.model_validate(unit) for unit in manifest.units]
    if not units or len({u.id for u in units}) != len(units):
        raise ValueError("manifest must contain unique nonempty units")
    return manifest.model, units


@torch.no_grad()
def run_feature_atlas(*, command, config_path, manifest_path, output):
    """三条命令的共享执行链；所有生成参数来自显式配置，原生渲染器单步特征诊断固定原生底座。"""
    from contextlib import nullcontext
    from mspc.config import load_experiment_config
    from mspc.inference.wan22 import Wan22InferenceRuntime
    from mspc.inference.continuity_dev_32 import wan22_shot_plan_from_frame_counts

    model_name, units = load_atlas_manifest(manifest_path, command)
    config = load_experiment_config(config_path).to_dict()
    model, inference = config["model"], config["inference"]
    if model["self_attention_policy"] != "full" or model["memory_prefix"] is not None or model["temporal_position_mode"] != "global_time_native_rope_v1":
        raise ValueError("原生渲染器单步特征诊断要求原生全注意力、原生位置和无参考输入")
    if inference["weight_source"] != "pretrained_model_paths" or inference["num_inference_steps"] != 50 or inference["sigma_shift"] != 5:
        raise ValueError("原生渲染器单步特征诊断要求预训练权重、50步采样和shift 5")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(Path(manifest_path).read_text())
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    runtime = Wan22InferenceRuntime(model_cfg=model, inference_cfg={**inference, "model_paths": model["paths"]},
                                   pipeline_cfg=config["pipeline_inputs"], dit_ckpt=None)
    pipe = runtime.pipe
    if len(pipe.dit.blocks) != 30 or tuple(pipe.dit.patch_size) != (1, 2, 2) or pipe.dit.in_dim != 48:
        raise ValueError("expected Wan2.2-TI2V-5B architecture (30 blocks, 48 latent channels)")
    for unit in units:
        state = DiagnosticStep()
        if command == "schedule":
            a = encode_atlas_text(pipe, unit.prompts_a)
            b = encode_atlas_text(pipe, unit.prompts_b)
            pipe.scheduler = atlas_scheduler()
            schedule = PositiveTextSchedule(a, b, switch_step=unit.switch_step, total_steps=50)
            plan = wan22_shot_plan_from_frame_counts(unit.frame_counts, label=unit.id)
            with diagnostic_steps(pipe, state, cfg_scale=inference["cfg_scale"], start_step=0, step_count=50,
                                  positive_schedule=schedule):
                runtime.generate_case(prompts=unit.prompts_a, shot_groups=plan["shot_groups"],
                    latent_shot_groups=plan["latent_shot_groups"], shot_frame_counts=unit.frame_counts,
                    output_path=output / f"{unit.id}.mp4", seed=unit.seed)
            continue
        z0 = unit.latent.read().to(device=runtime.device, dtype=runtime.dtype)
        if z0.shape[1] != pipe.dit.in_dim:
            raise ValueError("latent channel count differs from model")
        positive = encode_atlas_text(pipe, unit.prompt)
        negative = encode_atlas_text(pipe, inference["negative_prompt"])
        x_t, _ = noisy_target(z0, seed=unit.seed, step=unit.step)
        kwargs = dict(positive=positive, negative=negative, cfg_scale=inference["cfg_scale"], step=unit.step,
                      state=state, entity_tokens=unit.entity_tokens, temporal_position_mode=model["temporal_position_mode"])
        intervention = nullcontext()
        if command == "intervene":
            spec = unit.intervention
            window = dict(layers=spec.layers, steps=spec.steps)
            if spec.kind == "self_mask":
                intervention = self_attention_key_mask(pipe.dit, state, **window, query_key_sets=spec.query_key_sets)
            elif spec.kind == "cross_mask":
                intervention = cross_attention_token_mask(pipe.dit, state, **window,
                    shot_tokens={s.shot: s.tokens for s in spec.shot_tokens})
            else:
                records = {}
                source_xt, _ = noisy_target(z0.flip(-1), seed=unit.seed, step=unit.step)
                with activation_patch(pipe.dit, state, **window, tokens=spec.tokens, records=records, mode="record"):
                    single_step_features(pipe, x_t=source_xt, **kwargs)
                intervention = activation_patch(pipe.dit, state, **window, tokens=spec.tokens, records=records, mode="patch")
        with intervention:
            tensors, metadata = single_step_features(pipe, x_t=x_t, **kwargs)
        metadata.update(model=model_name, model_paths=model["paths"], seed=unit.seed, unit=unit.model_dump())
        save_atlas_features(output / f"{unit.id}.safetensors", tensors, metadata)
        print(json.dumps({"id": unit.id, "step": unit.step, "output": str(output)}, ensure_ascii=False), flush=True)
