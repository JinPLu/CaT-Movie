import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Sequence, Tuple
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel
from mspc.core.wan22_contract import (
    MEMORY_ATTENTION_POLICY_ROUTED_FIRST_LATENT_SELF_ONLY_V1,
    ROUTED_MEMORY_ATTENTION_POLICIES,
)
from mspc.models.spatial_layout import SpatialLayoutCondition, require_text_attention_backend
from mspc.models.source_identity import (
    SourceIdentityCondition, WanTextAttentionPrediction, WanVideoBlockPrediction, routed_source_attention,
)
from .wan_video_camera_controller import SimpleAdapter
from .wantodance import WanToDanceRotaryEmbedding, WanToDanceMusicEncoderLayer
from ..core.gradient.gradient_checkpoint import self_attention_checkpoint_scope

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    # Official Wan normalizes/rotates Q/K in FP32, runs the attention kernel in
    # V's AMP dtype, then restores Q's dtype before the output projection.
    out_dtype = q.dtype
    q, k = q.to(v.dtype), k.to(v.dtype)
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif q.is_cuda and torch.cuda.get_device_capability(q.device)[0] >= 10:
        # Blackwell (sm100): flash-attn 2 runs its old MMA path there; cuDNN attention is 3.2x faster at the
        # renderer's shapes (DiT 1.49x end to end, 11-14 measurement) with the same error against fp32.
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x.to(out_dtype)


def text_attention(q, k, v, num_heads, *, backend, bias=None):
    """Explicit text-attention backend; the reference path never selects Flash/Sage.

    The float mask broadcasts over heads. No head-expanded score or probability
    tensor is materialized by this code. SDPA math is a correctness reference;
    its cost at the actual video shape must be measured before a GPU run.
    """
    require_text_attention_backend(backend)
    if backend == "native":
        if bias is not None:
            raise ValueError("spatial layout bias requires text_attention_backend='sdpa'")
        return flash_attention(q, k, v, num_heads=num_heads)
    if bias is not None:
        if bias.ndim != 2 or bias.shape != (q.shape[1], k.shape[1]):
            raise ValueError("text layout bias must have shape [query tokens, text keys]")
        if not bias.is_floating_point():
            raise ValueError("text layout bias must be floating point")
        # The typed layout builder owns finiteness/nonnegativity. Keep float32
        # masks for half-precision q/k rather than overflow while casting B.
        mask_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        bias = bias.to(device=q.device, dtype=mask_dtype)[None, None]
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0)
    return rearrange(x, "b n s d -> b s (n d)", n=num_heads)


def frame_rows(x: torch.Tensor, rows: int, video_tokens: int) -> tuple[int, int]:
    """Per-frame timestep modulation (TI2V-5B): the first ``rows - extra`` rows each cover one latent frame of
    ``video_tokens // frames`` consecutive video tokens; each of the ``extra`` tokens after the video tokens
    (planner or first-frame tokens) has its own row. Returns (frames, tokens per frame)."""
    extra = x.shape[1] - video_tokens
    frames = rows - extra
    if extra < 0 or frames < 1 or video_tokens % frames:
        raise ValueError(f"{rows} modulation rows do not tile {video_tokens} video tokens plus {extra} extra tokens")
    return frames, video_tokens // frames


def by_frame(x: torch.Tensor, frames: int, per_frame: int) -> torch.Tensor:
    return x.reshape(x.shape[0], frames, per_frame, x.shape[2])


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, video_tokens: Optional[int] = None):
    """``x * (1 + scale) + shift``. With ``video_tokens`` the shift/scale rows are per latent frame (see
    ``frame_rows``) and broadcast over each frame's tokens: every element sees the same operands and the same two
    FP32 operations as with one row per token, so the result is bitwise the same, without materialising
    token-length modulation tensors."""
    if video_tokens is None:
        return (x * (1 + scale) + shift)
    frames, per_frame = frame_rows(x, shift.shape[1], video_tokens)
    video = (by_frame(x[:, :video_tokens], frames, per_frame) * (1 + scale[:, :frames, None])
             + shift[:, :frames, None]).reshape(x.shape[0], video_tokens, x.shape[2])
    if video_tokens == x.shape[1]:
        return video
    return torch.cat([video, x[:, video_tokens:] * (1 + scale[:, frames:]) + shift[:, frames:]], dim=1)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.float()


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def shot_phase_adjusted_freqs(
    freqs, latent_shot_groups, latent_frames, tokens_per_frame, *, shot_phase_index,
):
    """Fixed MultiShotMaster phase: target time + shot_index * 0.5.

    Reference groups have phase index zero. Their moved positions do not affect
    the native temporal step, which is read from the final two target frames.
    This position table is shared by every DiT block and has no learned weights.
    """
    if latent_frames < 2 or tokens_per_frame <= 0:
        raise ValueError("fixed shot phase requires at least two latent frames")
    if freqs.ndim != 3 or freqs.shape[0] != latent_frames * tokens_per_frame:
        raise ValueError("freqs token count must match the video token count")
    if len(shot_phase_index) != len(latent_shot_groups):
        raise ValueError("shot_phase_index must carry one index per latent shot group")
    temporal_dim = freqs.shape[-1] - 2 * (freqs.shape[-1] // 3)
    last_frame = (latent_frames - 1) * tokens_per_frame
    native_step = (
        freqs[last_frame, 0, :temporal_dim]
        * freqs[last_frame - tokens_per_frame, 0, :temporal_dim].conj()
    )
    phase_per_shot = torch.angle(native_step).to(torch.float64) * 0.5
    segments = []
    for shot_index, (start, end) in zip(shot_phase_index, latent_shot_groups):
        phase = torch.polar(
            torch.ones_like(phase_per_shot), shot_index * phase_per_shot,
        ).to(device=freqs.device, dtype=freqs.dtype)
        segment = freqs[start * tokens_per_frame:end * tokens_per_frame]
        segments.append(torch.cat((
            segment[:, :, :temporal_dim] * phase.view(1, 1, -1),
            segment[:, :, temporal_dim:],
        ), dim=-1))
    return torch.cat(segments, dim=0)


def set_to_torch_norm(models):
    for model in models:
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.use_torch_norm = True


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_torch_norm = False
        self.normalized_shape = (dim,)

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        if self.use_torch_norm:
            return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        else:        
            return self.norm(x.float()).to(dtype) * self.weight


class LayerNorm(nn.LayerNorm):
    """Official Wan LayerNorm: FP32 statistics, original activation dtype."""

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return super().forward(x.float()).type_as(x)


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


#: References and targets interact bidirectionally inside DiT. Clean reference latents
#: are restored at the denoising boundary, not by freezing their layer representations.
#: ``routed_self_only_v1`` lets each target shot read only the references routed to it and each
#: reference read only itself, see ``SelfAttention.routed_shot_attention``; it carries ``reference_routing``.
#: Under ``holocine_first_frame`` a shot reads its own tokens plus every shot's first latent frame;
#: under ``full`` it reads every target token (the MultiShotMaster-style window, ablation of HoloCine).
MEMORY_ATTENTION_BIDIRECTIONAL_V1 = "bidirectional_v1"
MEMORY_ATTENTION_ROUTED_SELF_ONLY_V1 = "routed_self_only_v1"
MEMORY_ATTENTION_POLICIES = (
    MEMORY_ATTENTION_BIDIRECTIONAL_V1,
    *ROUTED_MEMORY_ATTENTION_POLICIES,
)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.self_attention_policy = "full"
        
        self.attn = AttentionModule(self.num_heads)

    def forward(
        self,
        x,
        freqs,
        latent_shot_groups=None,
        tokens_per_frame=None,
        memory_attention_policy=None,
        prefix_frames=0,
        planner_layout=None,
        reference_routing=None,
        first_frame_tokens=0,
    ):
        if first_frame_tokens and self.self_attention_policy != "first_frame_communication":
            raise ValueError("first-frame tokens belong to the first-frame communication self-attention")
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        if self.self_attention_policy == "first_frame_interface":
            # 镜首帧接口的可见关系见 mspc.models.first_frame_interface；不带规划器 token。
            from mspc.models.first_frame_interface import first_frame_interface_attention
            if prefix_frames or memory_attention_policy is not None:
                raise ValueError("the first-frame interface does not implement bidirectional reference attention")
            if planner_layout is not None:
                raise ValueError("planner tokens go with the full self-attention policy, not the first-frame interface")
            x = first_frame_interface_attention(
                self.attn, q, k, v, groups=latent_shot_groups, prefix_frames=prefix_frames,
                tokens_per_frame=tokens_per_frame,
            )
        elif planner_layout is not None:
            # 规划器 token 的可见关系见 mspc.models.planner_tokens.planner_self_attention。
            from mspc.models.planner_tokens import planner_self_attention
            if self.self_attention_policy != "full":
                raise ValueError("planner tokens require the full self-attention policy")
            if memory_attention_policy not in (None, MEMORY_ATTENTION_BIDIRECTIONAL_V1):
                raise ValueError("planner tokens require bidirectional reference attention")
            x = planner_self_attention(
                self.attn, q, k, v, layout=planner_layout,
            )
        elif self.self_attention_policy == "holocine_first_frame":
            if memory_attention_policy is not None and memory_attention_policy not in ROUTED_MEMORY_ATTENTION_POLICIES:
                raise ValueError(f"HoloCine communication reads references only under one of {ROUTED_MEMORY_ATTENTION_POLICIES!r}")
            scope = ("first_latent"
                     if memory_attention_policy == MEMORY_ATTENTION_POLICY_ROUTED_FIRST_LATENT_SELF_ONLY_V1
                     else "all_latents")
            x = self.routed_shot_attention(q, k, v, latent_shot_groups, tokens_per_frame, prefix_frames=prefix_frames,
                                           routing=reference_routing, freqs=freqs, window="first_latents",
                                           reference_query_scope=scope)
        elif self.self_attention_policy == "first_frame_communication":
            # 首帧通信：只有每镜第一个潜帧读外部信息（本镜全部＋各镜首潜帧池＋路由参考），
            # 其余潜帧只读本镜；参考只读自己。语义见 ``routed_shot_attention``。
            if memory_attention_policy is not None and memory_attention_policy not in ROUTED_MEMORY_ATTENTION_POLICIES:
                raise ValueError("first-frame communication reads references only under one of "
                                 f"{ROUTED_MEMORY_ATTENTION_POLICIES!r}")
            x = self.routed_shot_attention(q, k, v, latent_shot_groups, tokens_per_frame, prefix_frames=prefix_frames,
                                           routing=reference_routing, freqs=freqs, window="first_frame_communication",
                                           reference_query_scope=None, first_frame_tokens=first_frame_tokens)
        elif self.self_attention_policy == "full" and memory_attention_policy == MEMORY_ATTENTION_ROUTED_SELF_ONLY_V1:
            x = self.routed_shot_attention(q, k, v, latent_shot_groups, tokens_per_frame, prefix_frames=prefix_frames,
                                           routing=reference_routing, freqs=freqs, window="all_targets",
                                           reference_query_scope="all_latents")
        elif self.self_attention_policy == "full":
            x = self.memory_prefix_attention(
                q, k, v, memory_attention_policy, prefix_frames, tokens_per_frame
            )
        else:
            raise ValueError(f"unknown self_attention_policy: {self.self_attention_policy}")
        return self.o(x)

    def routed_shot_attention(self, q, k, v, groups, tokens_per_frame, *, window, reference_query_scope=None,
                              prefix_frames=0, routing=None,
                              freqs=None, first_frame_tokens=0):
        """逐镜注意力，共用一个 softmax。``window="first_latents"``（HoloCine）：每镜完整 Q 读取本镜 K/V＋所有目标镜
        首潜帧 K/V；``window="all_targets"``（MultiShotMaster 式窗内全互读，HoloCine 的消融）：每镜读全部目标 K/V；
        ``window="first_frame_communication"``：只有每镜第一个潜帧读本镜全部＋各镜首潜帧 K/V＋路由参考，其余潜帧只读
        本镜 K/V（``reference_query_scope`` 必须为 None，参考只在首潜帧被读）。

        ``first_frame_tokens``（只配首帧通信）：视频 token 之后每个目标镜各有这么多附加 token（镜序排列），与本镜
        首潜帧合成**首帧组**：首帧组读本镜全部潜帧＋本镜附加 token＋各镜首帧组池＋路由参考；其余潜帧读本镜全部潜帧
        ＋本镜附加 token。输出按序列顺序拼回（附加 token 的输出排在全部视频输出之后）。语义见
        ``mspc.models.first_frame_tokens``。

        按 HoloCine 0a7630b 保留本镜首帧的重复条目；不 detach，梯度回到原视频。
        带参考（``routed_self_only_v1``）时前 ``prefix_frames`` 组各是一张参考：参考只读自己；镜 i 另读
        ``routing[i]`` 里的参考，这些参考的 K 再乘镜 i 首 token（h=w=0，空间角为零）在 ``freqs`` 里的那一行，
        时间从 −5(M−n) 移到 s_i − 5(M−n)（s_i 是镜 i 在目标潜帧里的起点），V 不动。镜 i 读到全部 M 张参考时，
        它到参考的距离与逐镜生成相同；只读到一部分时，参考仍在全局槽位上、中间有空槽。平移后的参考槽可能与更早
        某镜首潜帧的时间位置重合（MultiShotMaster 的背景参考同样如此），接受。
        """
        if not groups or type(tokens_per_frame) is not int or tokens_per_frame <= 0:
            raise ValueError("first-frame attention requires shot groups and tokens_per_frame")
        n = tokens_per_frame
        if len(groups) <= prefix_frames or any(tuple(groups[j]) != (j, j + 1) for j in range(prefix_frames)):
            raise ValueError("each reference must own one latent group before the target shots")
        ranges, end = [], prefix_frames
        for start, stop in groups[prefix_frames:]:
            if start != end or stop <= start:
                raise ValueError("shot groups must partition the video timeline")
            ranges.append((start * n, stop * n))
            end = stop
        if type(first_frame_tokens) is not int or first_frame_tokens < 0 or (
                first_frame_tokens and window != "first_frame_communication"):
            raise ValueError("first-frame tokens are a non-negative count used only by first_frame_communication")
        registers = first_frame_tokens
        video = end * n
        if video + registers * len(ranges) != q.shape[1]:
            raise ValueError("shot groups do not cover the video tokens")
        register_ranges = [(video + shot * registers, video + (shot + 1) * registers) for shot in range(len(ranges))]
        if (routing is None) != (prefix_frames == 0) or (routing is not None and len(routing) != len(ranges)):
            raise ValueError("references need reference_routing for every target shot, and only references do")
        if window not in ("first_latents", "all_targets", "first_frame_communication"):
            raise ValueError(f"unknown shot attention window {window!r}")
        if window == "first_frame_communication" and reference_query_scope is not None:
            raise ValueError("first_frame_communication reads references only at the first latent frame; "
                             "it takes no separate reference query scope")
        if window != "first_frame_communication" and reference_query_scope not in ("all_latents", "first_latent"):
            raise ValueError(f"unknown reference query scope {reference_query_scope!r}")
        if window == "first_frame_communication":
            # 首帧组池：各镜首潜帧（与本镜附加 token，若有）。
            global_k = torch.cat([part for (a, _), (r, s) in zip(ranges, register_ranges)
                                  for part in (k[:, a:a + n], k[:, r:s])], dim=1)
            global_v = torch.cat([part for (a, _), (r, s) in zip(ranges, register_ranges)
                                  for part in (v[:, a:a + n], v[:, r:s])], dim=1)
        else:
            global_k = torch.cat([k[:, a:a + n] for a, _ in ranges], dim=1)
            global_v = torch.cat([v[:, a:a + n] for a, _ in ranges], dim=1)
        target_k, target_v = k[:, prefix_frames * n:video], v[:, prefix_frames * n:video]
        register_pieces = []
        pieces = [self.attn(q[:, j * n:(j + 1) * n].contiguous(), k[:, j * n:(j + 1) * n].contiguous(),
                            v[:, j * n:(j + 1) * n].contiguous()) for j in range(prefix_frames)]
        for shot, (a, b) in enumerate(ranges):
            if window == "first_frame_communication":
                # 首帧组是唯一的跨镜/参考入口：首帧组读本镜全部（含本镜附加 token）＋各镜首帧组池（本镜按 HoloCine
                # 重复）＋路由参考；其余潜帧只读本镜全部（含本镜附加 token）。
                r, s = register_ranges[shot]
                local_k, local_v = torch.cat([k[:, a:b], k[:, r:s]], dim=1), torch.cat([v[:, a:b], v[:, r:s]], dim=1)
                first_keys, first_values = [local_k, global_k], [local_v, global_v]
                if prefix_frames and routing[shot]:
                    index = torch.cat([torch.arange(j * n, (j + 1) * n, device=q.device) for j in routing[shot]])
                    first_keys.append(rope_apply(k[:, index], freqs[a:a + 1], self.num_heads))
                    first_values.append(v[:, index])
                group = self.attn(torch.cat([q[:, a:a + n], q[:, r:s]], dim=1).contiguous(),
                                  torch.cat(first_keys, dim=1), torch.cat(first_values, dim=1))
                pieces.append(group[:, :n])
                register_pieces.append(group[:, n:])
                if b > a + n:
                    pieces.append(self.attn(q[:, a + n:b].contiguous(), local_k.contiguous(), local_v.contiguous()))
                continue
            keys, values = (([k[:, a:b], global_k], [v[:, a:b], global_v]) if window == "first_latents"
                            else ([target_k], [target_v]))
            if reference_query_scope == "first_latent" and prefix_frames and routing[shot] and b > a + n:
                # Only direct reference keys change. Later frames still read the same local
                # shot and dynamically updated first-latent pool, including duplicate keys.
                later = self.attn(q[:, a + n:b].contiguous(), torch.cat(keys, dim=1), torch.cat(values, dim=1))
            else:
                later = None
            if prefix_frames and routing[shot]:
                index = torch.cat([torch.arange(j * n, (j + 1) * n, device=q.device) for j in routing[shot]])
                keys.append(rope_apply(k[:, index], freqs[a:a + 1], self.num_heads))
                values.append(v[:, index])
            query_end = a + n if later is not None else b
            first = self.attn(q[:, a:query_end].contiguous(), torch.cat(keys, dim=1), torch.cat(values, dim=1))
            pieces.append(first if later is None else torch.cat((first, later), dim=1))
        return torch.cat(pieces + register_pieces, dim=1)

    def memory_prefix_attention(
        self, q, k, v, memory_attention_policy, prefix_frames, tokens_per_frame
    ):
        """Full attention for native video and the clean-reference prefix alike."""
        if memory_attention_policy not in (None, MEMORY_ATTENTION_BIDIRECTIONAL_V1):
            raise ValueError(
                f"full self-attention reads references under {MEMORY_ATTENTION_BIDIRECTIONAL_V1!r}; "
                f"got {memory_attention_policy!r}"
            )
        return self.attn(q, k, v)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        shot_token_ranges: Optional[Sequence[Sequence[int]]] = None,
        spatial_layout_biases: Optional[Sequence[torch.Tensor]] = None,
        text_attention_backend: str = "native",
        source_identity_condition: Optional[SourceIdentityCondition] = None,
    ):
        require_text_attention_backend(text_attention_backend)
        if spatial_layout_biases is not None and text_attention_backend != "sdpa":
            raise ValueError("spatial layout bias requires text_attention_backend='sdpa'")
        if shot_token_ranges is not None:
            if self.has_image_input:
                raise ValueError("multi-shot prompt routing requires a pure T2V DiT")
            if x.shape[0] != 1:
                raise ValueError("multi-shot prompt routing requires a single latent batch")
            if y.shape[0] != len(shot_token_ranges):
                raise ValueError("multi-shot prompt context count must match token ranges")
            if spatial_layout_biases is not None and len(spatial_layout_biases) != len(shot_token_ranges):
                raise ValueError("spatial layout bias count must match routed text rows")
            q = self.norm_q(self.q(x))
            k = self.norm_k(self.k(y))
            v = self.v(y)
            if source_identity_condition is not None and (
                    text_attention_backend != "native" or spatial_layout_biases is not None
                    or len(source_identity_condition.rows) >= len(shot_token_ranges)):
                raise ValueError("source identity requires native reference rows followed by at least one target row")
            pieces, losses = [], []
            for index, (start, end) in enumerate(shot_token_ranges):
                query, keys, values = q[:, start:end].contiguous(), k[index:index + 1], v[index:index + 1]
                if source_identity_condition is not None and index < len(source_identity_condition.rows):
                    prediction = routed_source_attention(self.attn, query, keys, values,
                        num_heads=self.num_heads, row=source_identity_condition.rows[index])
                    pieces.append(prediction.hidden_states)
                    if prediction.source_alignment_loss is not None:
                        losses.append(prediction.source_alignment_loss)
                else:
                    pieces.append(self.attn(query, keys, values) if text_attention_backend == "native" else text_attention(
                        query, keys, values, self.num_heads, backend=text_attention_backend,
                        bias=None if spatial_layout_biases is None else spatial_layout_biases[index]))
            routed = torch.cat(pieces, dim=1)
            return WanTextAttentionPrediction(self.o(routed), torch.stack(losses).mean() if losses else None)
        if source_identity_condition is not None:
            raise ValueError("source identity requires explicit reference/target text-row routing")
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        if spatial_layout_biases is not None and len(spatial_layout_biases) != 1:
            raise ValueError("single-row text attention requires one spatial layout bias")
        out = self.attn(q, k, v) if text_attention_backend == "native" else text_attention(
            q, k, v, self.num_heads, backend=text_attention_backend,
            bias=None if spatial_layout_biases is None else spatial_layout_biases[0],
        )
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            out = out + y
        return WanTextAttentionPrediction(self.o(out), None)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual, video_tokens=None):
        """``x + gate * residual`` in FP32; ``video_tokens`` as in ``modulate`` (per-frame gate rows, bitwise the
        same as one row per token)."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            x, gate, residual = x.float(), gate.float(), residual.float()
            if video_tokens is None:
                return x + gate * residual
            frames, per_frame = frame_rows(x, gate.shape[1], video_tokens)
            video = (by_frame(x[:, :video_tokens], frames, per_frame) + gate[:, :frames, None]
                     * by_frame(residual[:, :video_tokens], frames, per_frame)).reshape(
                         x.shape[0], video_tokens, x.shape[2])
            if video_tokens == x.shape[1]:
                return video
            return torch.cat([video, x[:, video_tokens:] + gate[:, frames:] * residual[:, video_tokens:]], dim=1)

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.activation_checkpoint_context_fn = None

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        x,
        context,
        t_mod,
        freqs,
        shot_token_ranges=None,
        latent_shot_groups=None,
        tokens_per_frame=None,
        memory_attention_policy=None,
        prefix_frames=0,
        spatial_layout_biases=None,
        text_attention_backend="native",
        region_inputs=None,
        planner_layout=None,
        source_identity_condition=None,
        reference_routing=None,
        first_frame_tokens=0,
    ):
        # DeepSpeed 的激活检查点按签名顺序位置传参（core/gradient/gradient_checkpoint.py），新参数只往末尾加。
        # 规划器 token、首帧组附加 token 排在视频 token 之后：交叉注意力与区域通道对前 n 个视频 token 照原路；规划器
        # token 不读文字，附加 token 读本镜那一行文字（见下）；FFN 对整段一致。没有这两类 token 时 n 就是整段，与原渲染器
        # 逐位相同。
        if planner_layout is not None and first_frame_tokens:
            raise ValueError("planner tokens and first-frame tokens are separate experiments")
        if planner_layout is not None:
            n = planner_layout.video_tokens
        elif first_frame_tokens:
            n = x.shape[1] - first_frame_tokens * (len(latent_shot_groups) - prefix_frames)
        else:
            n = x.shape[1]
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # 4 维 t_mod 每个潜帧一行（附加 token 各一行，见 ``frame_rows``），按帧广播到 token。
        video_tokens = n if has_seq else None
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x).float(), shift_msa, scale_msa, video_tokens)
        with self_attention_checkpoint_scope():
            attention = self.self_attn(
                input_x,
                freqs,
                latent_shot_groups=latent_shot_groups,
                tokens_per_frame=tokens_per_frame,
                memory_attention_policy=memory_attention_policy,
                prefix_frames=prefix_frames,
                **({} if planner_layout is None else {"planner_layout": planner_layout}),
                **({} if reference_routing is None else {"reference_routing": reference_routing}),
                **({} if not first_frame_tokens else {"first_frame_tokens": first_frame_tokens}),
            )
        x = self.gate(x, gate_msa, attention, video_tokens)
        del attention
        video = x[:, :n]
        cross_attention_kwargs = {}
        if spatial_layout_biases is not None or text_attention_backend != "native":
            cross_attention_kwargs = dict(
                spatial_layout_biases=spatial_layout_biases,
                text_attention_backend=text_attention_backend,
            )
        attention = self.cross_attn(self.norm3(video), context, shot_token_ranges,
            source_identity_condition=source_identity_condition, **cross_attention_kwargs)
        video = video + attention.hidden_states
        if first_frame_tokens:
            # 附加 token 与本镜潜帧读同一行文字：目标镜的文字行排在参考行之后，与镜一一对应。
            if (cross_attention_kwargs or source_identity_condition is not None or region_inputs is not None
                    or attention.source_alignment_loss is not None):
                raise ValueError("first-frame tokens read plain routed text rows only")
            twins = x[:, n:]
            if shot_token_ranges is None:
                twin_context, twin_ranges = context, None
            else:
                twin_context = context[prefix_frames:]
                twin_ranges = [(shot * first_frame_tokens, (shot + 1) * first_frame_tokens)
                               for shot in range(twin_context.shape[0])]
            twins = twins + self.cross_attn(self.norm3(twins), twin_context, twin_ranges).hidden_states
            x = torch.cat([video, twins], dim=1)
        if region_inputs is not None:
            region_attn = getattr(self, "region_attn", None)
            if region_attn is None:
                raise ValueError("region token inputs reached a block without region attention")
            video = video + region_attn(video, *region_inputs)
        if not first_frame_tokens:
            x = video if n == x.shape[1] else torch.cat([video, x[:, n:]], dim=1)
        input_x = modulate(self.norm2(x).float(), shift_mlp, scale_mlp, video_tokens)
        x = self.gate(x, gate_mlp, self.ffn(input_x), video_tokens)
        return WanVideoBlockPrediction(x, attention.source_alignment_loss)


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x, t_mod = x.float(), t_mod.float()
            if len(t_mod.shape) == 3:
                # 每个潜帧一行（见 ``frame_rows``），出 head 前附加 token 已去掉。
                shift, scale = (self.modulation.unsqueeze(0) + t_mod.unsqueeze(2)).chunk(2, dim=2)
                x = self.head(modulate(self.norm(x), shift.squeeze(2), scale.squeeze(2), x.shape[1]))
            else:
                shift, scale = (self.modulation + t_mod).chunk(2, dim=1)
                x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


def wantodance_torch_dfs(model: nn.Module, parent_name='root'):
    module_names, modules = [], []
    current_name = parent_name if parent_name else 'root'
    module_names.append(current_name)
    modules.append(model)
    for name, child in model.named_children():
        if parent_name:
            child_name = f'{parent_name}.{name}'
        else:
            child_name = name
        child_modules, child_names = wantodance_torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names


class WanToDanceInjector(nn.Module):
    def __init__(self, all_modules, all_modules_names, dim=2048, num_heads=32, inject_layer=[0, 27]):
        super().__init__()
        self.injected_block_id = {}
        injector_id = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, DiTBlock):
                for inject_id in inject_layer:
                    if f'root.transformer_blocks.{inject_id}' == mod_name:
                        self.injected_block_id[inject_id] = injector_id
                        injector_id += 1

        self.injector = nn.ModuleList(
            [
                CrossAttention(
                    dim=dim,
                    num_heads=num_heads,
                )
                for _ in range(injector_id)
            ]
        )
        self.injector_pre_norm_feat = nn.ModuleList(
            [
                nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6,)
                for _ in range(injector_id)
            ]
        )
        self.injector_pre_norm_vec = nn.ModuleList(
            [
                nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6,)
                for _ in range(injector_id)
            ]
        )


class WanModel(torch.nn.Module):

    _repeated_blocks = ["DiTBlock"]

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        wantodance_enable_music_inject: bool = False,
        wantodance_music_inject_layers = [0, 4, 8, 12, 16, 20, 24, 27],
        wantodance_enable_refimage: bool = False,
        wantodance_enable_refface: bool = False,
        wantodance_enable_global: bool = False,
        wantodance_enable_dynamicfps: bool = False,
        wantodance_enable_unimodel: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads

        if wantodance_enable_dynamicfps or wantodance_enable_unimodel:
            end = int(22350 / 8 + 0.5) # 149f * 30fps * 5s = 22350
            self.freqs = precompute_freqs_cis_3d(head_dim, end=end)
        else:
            self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

        self.prepare_wantodance(in_dim, dim, num_heads, has_image_pos_emb, out_dim, patch_size, eps,
                                wantodance_enable_music_inject, wantodance_music_inject_layers, wantodance_enable_refimage, wantodance_enable_refface,
                                wantodance_enable_global, wantodance_enable_dynamicfps, wantodance_enable_unimodel)

    def prepare_wantodance(
        self,
        in_dim, dim, num_heads, has_image_pos_emb, out_dim, patch_size, eps,
        wantodance_enable_music_inject: bool = False,
        wantodance_music_inject_layers = [0, 4, 8, 12, 16, 20, 24, 27],
        wantodance_enable_refimage: bool = False,
        wantodance_enable_refface: bool = False,
        wantodance_enable_global: bool = False,
        wantodance_enable_dynamicfps: bool = False,
        wantodance_enable_unimodel: bool = False,
    ):
        if wantodance_enable_music_inject:
            all_modules, all_modules_names = wantodance_torch_dfs(self.blocks, parent_name="root.transformer_blocks")
            self.music_injector = WanToDanceInjector(all_modules, all_modules_names, dim=dim, num_heads=num_heads, inject_layer=wantodance_music_inject_layers)
        if wantodance_enable_refimage:
            self.img_emb_refimage = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if wantodance_enable_refface:
            self.img_emb_refface = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if wantodance_enable_global or wantodance_enable_dynamicfps or wantodance_enable_unimodel:
            music_feature_dim = 35
            ff_size = 1024
            dropout = 0.1
            latent_dim = 256
            nhead = 4
            activation = F.gelu
            rotary = WanToDanceRotaryEmbedding(dim=latent_dim)
            self.music_projection = nn.Linear(music_feature_dim, latent_dim)
            self.music_encoder = nn.Sequential()
            for _ in range(2):
                self.music_encoder.append(
                    WanToDanceMusicEncoderLayer(
                        d_model=latent_dim,
                        nhead=nhead,
                        dim_feedforward=ff_size,
                        dropout=dropout,
                        activation=activation,
                        batch_first=True,
                        rotary=rotary,
                        device='cuda',
                    )
                )
        if wantodance_enable_unimodel:
            self.patch_embedding_global = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        if wantodance_enable_unimodel:
            self.head_global = Head(dim, out_dim, patch_size, eps)
        self.wantodance_enable_music_inject = wantodance_enable_music_inject
        self.wantodance_enable_refimage = wantodance_enable_refimage
        self.wantodance_enable_refface = wantodance_enable_refface
        self.wantodance_enable_global = wantodance_enable_global
        self.wantodance_enable_dynamicfps = wantodance_enable_dynamicfps
        self.wantodance_enable_unimodel = wantodance_enable_unimodel

    def wantodance_after_transformer_block(self, block_idx, hidden_states):
        if self.wantodance_enable_music_inject:
            if block_idx in self.music_injector.injected_block_id.keys():
                audio_attn_id = self.music_injector.injected_block_id[block_idx]
                audio_emb = self.merged_audio_emb  # b f n c
                num_frames = audio_emb.shape[1]
                input_hidden_states = hidden_states.clone()  # b (f h w) c
                input_hidden_states = rearrange(input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames)
                attn_hidden_states = self.music_injector.injector_pre_norm_feat[audio_attn_id](input_hidden_states)
                audio_emb = rearrange(audio_emb, "b t c -> (b t) 1 c", t=num_frames)
                attn_audio_emb = audio_emb
                residual_out = self.music_injector.injector[audio_attn_id](attn_hidden_states, attn_audio_emb)
                residual_out = rearrange(residual_out, "(b t) n c -> b (t n) c", t=num_frames)
                hidden_states = hidden_states + residual_out
        return hidden_states

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None, enable_wantodance_global=False):
        if enable_wantodance_global:
            x = self.patch_embedding_global(x)
        else:
            x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                spatial_layout: Optional[SpatialLayoutCondition] = None,
                text_attention_backend: str = "native",
                **kwargs,
                ):
        # The pipeline and direct entry share one execution owner, including
        # shot routing, prefix handling and checkpoint recomputation. The old
        # duplicated body expected patchify to return a (tensor, grid) pair,
        # whereas patchify actually returns a five-dimensional tensor.
        import inspect
        from ..pipelines.wan_video import model_fn_wan_video

        supported = set(inspect.signature(model_fn_wan_video).parameters) - {"kwargs", "dit", "latents"}
        unknown = set(kwargs) - supported
        if unknown:
            raise TypeError(f"unsupported WanModel.forward arguments: {sorted(unknown)}")
        return model_fn_wan_video(
            dit=self, latents=x, timestep=timestep, context=context,
            clip_feature=clip_feature, y=y,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            spatial_layout=spatial_layout, text_attention_backend=text_attention_backend,
            **kwargs,
        )
