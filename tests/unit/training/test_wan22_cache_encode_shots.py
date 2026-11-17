"""latents_only cache 编码必须按分镜切分，而不是整段联合编码。

真值来自已建成的 80k cache（真语料，非发明）：
``outputs/datasets/hq_pool/20260819_ci_cine_v1/coarse_subset_80000.csv`` 的
``shot_groups``，与
``outputs/datasets/latent_cache/hq_pool_ci_cine_20260819_coarse80k_15fps_latents_only_v2/manifest.json``
里同 case 的 ``input_latents_shape[2]`` / ``latent_shot_groups``。

Wan VAE 在时间轴上是 causal 的：整段联合编码给 ``(sum(len) - 1) // 4 + 1``，
按分镜编码给 ``sum((len - 1) // 4 + 1)``，每个分镜各留一个首帧。
``validate_multishot_groups`` 要求的是后者，所以联合编码的 cache 一定与自己的
``latent_shot_groups`` 对不上。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME = REPO_ROOT / "third_party" / "wan22_runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from mspc.models.wan22_loader import (  # noqa: E402
    WAN22_COMPONENT_MODE_CACHE_ENCODE_LATENTS,
)
from mspc.training.backends.wan22 import Wan22DiffSynthTrainingModule  # noqa: E402


# case_id, shot_groups, latent_shot_groups, manifest input_latents_shape[2]
_CI_VID_000000002 = ("ci_vid:test:000000002", [[0, 153], [153, 222]], [[0, 39], [39, 57]], 57)
_CI_VID_000000012 = ("ci_vid:test:000000012", [[0, 49], [49, 78]], [[0, 13], [13, 21]], 21)
_CI_VID_000000014 = (
    "ci_vid:test:000000014",
    [[0, 21], [21, 46], [46, 83]],
    [[0, 6], [6, 13], [13, 23]],
    23,
)
_CI_VID_000000029 = ("ci_vid:test:000000029", [[0, 57], [57, 78]], [[0, 15], [15, 21]], 21)


class _RecordingVAE:
    """站位真 Wan VAE，只保留它的 4n+1 时间算术和调用签名。

    ``diffsynth.models.wan_video_vae.WanVideoVAE.encode`` 遍历 batch 维、逐个
    ``single_encode`` 再 stack，返回 ``[B, z, (T - 1) // 4 + 1, H/f, W/f]``。
    分镜布局只取决于这条时间律，所以空间侧照搬 upsampling_factor 即可。
    """

    upsampling_factor = 16

    def __init__(self) -> None:
        self.frame_counts: list[int] = []

    def encode(self, videos, device, tiled, tile_size, tile_stride):
        if not isinstance(videos, torch.Tensor) or videos.dim() != 5:
            raise AssertionError(f"vae.encode wants [B,C,T,H,W], got {type(videos)}")
        batch, _channels, frames, height, width = videos.shape
        self.frame_counts.append(int(frames))
        return torch.zeros(
            batch,
            48,
            (frames - 1) // 4 + 1,
            height // self.upsampling_factor,
            width // self.upsampling_factor,
        )


class _CacheEncodePipe:
    height_division_factor = 16
    width_division_factor = 16
    device = torch.device("cpu")
    torch_dtype = torch.float32

    def __init__(self) -> None:
        self.vae = _RecordingVAE()
        self.onloaded: list[tuple[str, ...]] = []

    def load_models_to_device(self, names) -> None:
        self.onloaded.append(tuple(names))


def _latents_only_module() -> Wan22DiffSynthTrainingModule:
    module = object.__new__(Wan22DiffSynthTrainingModule)
    torch.nn.Module.__init__(module)
    object.__setattr__(module, "_pipe", _CacheEncodePipe())
    module.component_mode = WAN22_COMPONENT_MODE_CACHE_ENCODE_LATENTS
    module.pipeline_cfg = {"tiled": False, "tile_size": (30, 52), "tile_stride": (15, 26)}
    return module


def _sample(shot_groups, latent_shot_groups) -> dict[str, object]:
    num_frames = shot_groups[-1][1]
    return {
        "video": torch.zeros((3, num_frames, 16, 16), dtype=torch.uint8),
        "shot_groups": shot_groups,
        "latent_shot_groups": latent_shot_groups,
    }


def _shot_lengths(shot_groups) -> list[int]:
    return [end - start for start, end in shot_groups]


@pytest.mark.parametrize(
    ("case_id", "shot_groups", "latent_shot_groups", "manifest_latent_frames"),
    [_CI_VID_000000002, _CI_VID_000000012, _CI_VID_000000014, _CI_VID_000000029],
)
def test_multishot_cache_encode_matches_the_80k_manifest_latent_length(
    case_id: str,
    shot_groups: list[list[int]],
    latent_shot_groups: list[list[int]],
    manifest_latent_frames: int,
) -> None:
    module = _latents_only_module()

    input_latents, context, embed_meta = module.encode_for_cache(
        _sample(shot_groups, latent_shot_groups)
    )

    num_frames = shot_groups[-1][1]
    joint_latent_frames = (num_frames - 1) // 4 + 1
    # 联合编码与按分镜和不同，否则这一行验不出任何东西。
    assert joint_latent_frames != manifest_latent_frames, case_id
    assert int(input_latents.shape[2]) == manifest_latent_frames
    assert context is None
    assert embed_meta["latent_shot_groups"] == tuple(
        (start, end) for start, end in latent_shot_groups
    )
    # 每个分镜一次 VAE 调用，切片长度就是原始分镜长度。
    assert module.pipe.vae.frame_counts == _shot_lengths(shot_groups)
    assert module.pipe.onloaded == [("vae",)]


def test_encode_batch_keeps_each_sample_on_its_own_shot_partition() -> None:
    """同 shape、不同分镜切分的两行真语料，必须各自按自己的边界编码。

    生产端 ``latent_cache._split_same_shape_groups`` 只按 ``[C,T,H,W]`` 分桶，
    所以这两行会落进同一个 batch；把它们 stack 成一次 VAE 调用会强加同一条时间
    切分，两边都得到 20 而不是 21。
    """

    module = _latents_only_module()
    first_id, first_shots, first_latents, first_frames = _CI_VID_000000012
    second_id, second_shots, second_latents, second_frames = _CI_VID_000000029
    assert first_shots[-1][1] == second_shots[-1][1] == 78
    assert first_latents != second_latents

    encoded = module.encode_batch_for_cache(
        [
            _sample(first_shots, first_latents),
            _sample(second_shots, second_latents),
        ]
    )

    assert [int(latents.shape[2]) for latents, _context, _meta in encoded] == [
        first_frames,
        second_frames,
    ]
    assert module.pipe.vae.frame_counts == _shot_lengths(first_shots) + _shot_lengths(
        second_shots
    )
    # 78 帧联合编码只会是一次调用、20 帧 latent。
    assert 78 not in module.pipe.vae.frame_counts
    assert (78 - 1) // 4 + 1 == 20
    assert first_id != second_id


def test_cache_uint8_domain_matches_the_native_pil_preprocessor(monkeypatch) -> None:
    """All 256 pixel values retain the native BF16 preprocessing arithmetic."""
    from diffsynth.diffusion.base_pipeline import BasePipeline

    module = _latents_only_module()
    module.pipe.torch_dtype = torch.bfloat16
    captured = []
    encode = module.pipe.vae.encode

    def record(videos, **kwargs):
        captured.append(videos.clone())
        return encode(videos, **kwargs)

    monkeypatch.setattr(module.pipe.vae, "encode", record)
    pixels = np.repeat(np.arange(256, dtype=np.uint8).reshape(16, 16, 1), 3, axis=2)
    native = object.__new__(BasePipeline)
    native.torch_dtype, native.device = torch.bfloat16, torch.device("cpu")
    expected = native.preprocess_video([Image.fromarray(pixels)])
    sample = {"video": torch.from_numpy(pixels.copy()).permute(2, 0, 1).unsqueeze(1),
              "shot_groups": [[0, 1]], "latent_shot_groups": [[0, 1]]}
    module.encode_for_cache(sample)
    assert len(captured) == 1
    assert captured[0].dtype == torch.bfloat16
    assert torch.equal(captured[0], expected)
