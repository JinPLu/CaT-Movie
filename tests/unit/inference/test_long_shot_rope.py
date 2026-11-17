"""长单镜时间 RoPE 修正（RIFLEx 式）：只换固有频率那一列，只对超过原生长度的单镜窗生效。"""
import math

import pytest
import torch

from mspc.inference.wan22 import LongShotRope

pytest.importorskip("diffsynth", reason="Wan2.2 runtime is not installed in this environment")
from diffsynth.models.wan_video_dit import precompute_freqs_cis_3d  # noqa: E402

OPTIONS = {"index": 5, "native_latent_frames": 31, "margin": 0.9}


def test_wan22_temporal_period_of_index_five_is_closest_to_native_length():
    # Wan2.2-5B 头维 128：时间 44 维、22 个频率；原生 121 帧＝31 潜帧。
    temporal = precompute_freqs_cis_3d(128)[0]
    assert temporal.shape[1] == 22
    periods = [2 * math.pi / math.atan2(temporal[1, j].imag.item(), temporal[1, j].real.item()) for j in range(1, 8)]
    closest = min(range(len(periods)), key=lambda j: abs(periods[j] - 31)) + 2
    assert closest == OPTIONS["index"]


def test_only_the_intrinsic_column_changes_and_spans_one_margin_period():
    temporal = precompute_freqs_cis_3d(128)[0]
    rope = LongShotRope.from_mapping(OPTIONS)
    out = rope.temporal_table(temporal, 61)
    changed = [j for j in range(temporal.shape[1]) if not torch.equal(out[:, j], temporal[:, j])]
    assert changed == [4]
    angle = torch.angle(out[1, 4]).item()
    assert angle * 61 == pytest.approx(0.9 * 2 * math.pi, rel=1e-6)
    assert out.dtype == temporal.dtype


def test_applies_only_to_one_shot_longer_than_native():
    rope = LongShotRope.from_mapping(OPTIONS)
    assert rope.applies([[0, 32]])
    assert not rope.applies([[0, 31]])
    assert not rope.applies([[0, 20], [20, 60]])


def test_config_must_name_all_three_fields():
    with pytest.raises(ValueError):
        LongShotRope.from_mapping({"index": 5, "native_latent_frames": 31})
    with pytest.raises(ValueError):
        LongShotRope.from_mapping({"index": 5, "native_latent_frames": 31, "margin": 1})


def test_renderer_config_accepts_the_field_and_defaults_to_off():
    from mspc.config.experiment import _long_shot_rope, ExperimentConfigError

    assert _long_shot_rope(None, "inference.long_shot_rope") is None
    assert _long_shot_rope(OPTIONS, "inference.long_shot_rope") == OPTIONS
    with pytest.raises(ExperimentConfigError):
        _long_shot_rope({"index": 5}, "inference.long_shot_rope")
