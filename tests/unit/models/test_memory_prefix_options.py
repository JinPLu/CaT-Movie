"""参考主合同固定负时间间隔与零空间偏移；配置仍须显式给全。"""
from __future__ import annotations

from dataclasses import asdict

import pytest

from mspc.models.memory_prefix_options import MemoryPrefixModelOptions, ReferenceOffsetOptions
from tests.support.memory_prefix import FULL_WINDOW_CONFIG, NOISE_CONFIG, OFFSET_CONFIG, offset_options


def _raw():
    return {"frames_max": 10, "dropout_p": 0.2, "noise": NOISE_CONFIG, "reference_offset": OFFSET_CONFIG,
            "attention_policy": "bidirectional_v1", "row_policy": "fixed_role_v1",
            "attention_window": FULL_WINDOW_CONFIG, "placement": "ordered_slots"}


def test_fixed_reference_contract_roundtrips_and_matches_inference():
    options = MemoryPrefixModelOptions.from_mapping(_raw())
    assert options.reference_offset.train == {"dt": (-5, -5), "dh": (0, 0), "dw": (0, 0)}
    assert options.reference_offset.inference == (-5, 0, 0)
    assert {options.reference_offset.draw((u, v, w)) for u in (0.0, 0.5, 0.999)
            for v in (0.0, 0.999) for w in (0.0, 0.999)} == {(-5, 0, 0)}
    assert MemoryPrefixModelOptions.from_mapping(asdict(options)) == options


def test_invalid_offset_draw_is_rejected():
    with pytest.raises(ValueError):
        offset_options().draw((1.0, 0.0, 0.0))


@pytest.mark.parametrize("offset", [
    {"train": {"dt": [-5, -1], "dh": [0, 0], "dw": [0, 0]}, "inference": [-5, 0, 0]},
    {"train": {"dt": [-5, -5], "dh": [15, 30], "dw": [26, 52]}, "inference": [-5, 15, 26]},
    {"train": {"dt": [-3, -3], "dh": [15, 15], "dw": [26, 26]}, "inference": [-3, 15, 26]},
])
def test_non_contract_offsets_are_rejected(offset):
    with pytest.raises(ValueError, match="fixed offsets with dt=-5"):
        ReferenceOffsetOptions.from_mapping(offset, "test")


@pytest.mark.parametrize("dh,dw", [(0, 0), (15, 26)])
def test_in_grid_and_off_grid_fixed_offsets_parse(dh, dw):
    options = ReferenceOffsetOptions.from_mapping(
        {"train": {"dt": [-5, -5], "dh": [dh, dh], "dw": [dw, dw]}, "inference": [-5, dh, dw]}, "test")
    assert options.inference == (-5, dh, dw) and options.draw((0.3, 0.6, 0.9)) == (-5, dh, dw)


@pytest.mark.parametrize("train,inference,message", [
    ({"dt": [-5, 0], "dh": [0, 0], "dw": [0, 0]}, [-5, 0, 0], "dt < 0"),
    ({"dt": [-5, -5], "dh": [1, 0], "dw": [0, 0]}, [-5, 0, 0], "low <= high"),
    ({"dt": [-5, -5], "dh": [0, 0]}, [-5, 0, 0], "exactly"),
    ({"dt": [-5, -5], "dh": [0, 0], "dw": [0, 0]}, [-5, 1, 0], "inside the training ranges"),
])
def test_malformed_offset_config_raises(train, inference, message):
    raw = _raw()
    raw["reference_offset"] = {"train": train, "inference": inference}
    with pytest.raises(ValueError, match=message):
        MemoryPrefixModelOptions.from_mapping(raw)
