"""vendored DiT 的两条真实边界：TI2V-5B 不带图像输入，带图像输入的 DiT 拒绝按镜路由。

按镜路由（``latent_shot_groups`` → 每镜一段 context）的前提是纯 T2V 的 DiT。这两条测试读
vendored 运行时自己的配置表和交叉注意力，不读抄下来的注释。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
# 跑的是 **vendored** 运行时自己的代码，不是替身，所以要按 ``configs/train/**`` 里
# ``train_pythonpath`` 的同一条路把它放上 sys.path。
_RUNTIME = REPO_ROOT / "third_party" / "wan22_runtime"
if str(_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_RUNTIME))

#: 迷你 DiT。除 dim / num_layers / 网格尺寸外，其余键与 vendored
#: ``diffsynth/configs/model_configs.py`` 里 TI2V-5B 那一条逐字相同——尤其是
#: ``has_image_input=False``，按镜路由依赖它。
TINY_DIT_KWARGS = dict(
    dim=24,
    num_heads=2,
    num_layers=1,
    in_dim=48,
    out_dim=48,
    ffn_dim=32,
    text_dim=4096,
    freq_dim=16,
    eps=1e-6,
    patch_size=(1, 2, 2),
    has_image_input=False,
    seperated_timestep=True,
    require_clip_embedding=False,
    require_vae_embedding=False,
)
TI2V_5B_MODEL_HASH = "1f5ab7703c6fc803fdded85ff040c316"


def _tiny_dit(**overrides: object):
    from diffsynth.models.wan_video_dit import WanModel

    dit = WanModel(**{**TINY_DIT_KWARGS, **overrides})
    return dit


def test_ti2v_5b_declares_has_image_input_false() -> None:
    """按镜路由的前提，读的是 vendored 配置表本身，不是抄下来的注释。"""

    from diffsynth.configs.model_configs import wan_series

    entries = [
        entry for entry in wan_series if entry.get("model_hash") == TI2V_5B_MODEL_HASH
    ]
    assert len(entries) == 1
    extra = entries[0]["extra_kwargs"]
    assert extra["has_image_input"] is False
    assert extra["dim"] == 3072
    assert extra["text_dim"] == 4096


def test_image_input_dit_refuses_multi_shot_routing() -> None:
    """``has_image_input=True`` 的 DiT 一旦拿到按镜 span 就抛，不会静默走单 context。"""

    dit = _tiny_dit(has_image_input=True, require_clip_embedding=True)
    cross_attn = dit.blocks[0].cross_attn
    assert hasattr(cross_attn, "k_img")
    with pytest.raises(ValueError, match="requires a pure T2V DiT"):
        cross_attn(
            torch.randn(1, 6, TINY_DIT_KWARGS["dim"]),
            torch.randn(2, 4, TINY_DIT_KWARGS["dim"]),
            [(0, 3), (3, 6)],
        )
    assert not hasattr(_tiny_dit().blocks[0].cross_attn, "k_img")
