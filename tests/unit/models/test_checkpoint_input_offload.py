"""两个真实autograd块核checkpoint输入搬运、共享条件梯度与重算RNG。"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "wan_checkpoint_input_offload_under_test",
    ROOT / "third_party/wan22_runtime/diffsynth/core/gradient/gradient_checkpoint.py",
)
assert _SPEC is not None and _SPEC.loader is not None
checkpoint_owner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checkpoint_owner)


class _ConditionedBlock(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.x_projection = nn.Linear(4, 4)
        self.context_projection = nn.Linear(4, 4, bias=False)
        self.time_projection = nn.Linear(4, 4, bias=False)
        self.scale = scale
        self.calls = []

    def forward(self, x, context, t_mod, freqs, *, offset):
        self.calls.append({
            "x": x.data_ptr(),
            "shared": tuple(value.data_ptr() for value in (context, t_mod, freqs, offset)),
            "autocast": torch.is_autocast_enabled("cpu"),
        })
        # 两个块有不同参数和scale；共享条件各用多次，使重算绑定错块或
        # 将共享条件detach都会改变实际参数／输入梯度。
        hidden = (self.x_projection(x) + self.context_projection(context)
                  + self.time_projection(t_mod) + context * t_mod + freqs + offset)
        hidden = F.dropout(torch.sin(hidden), p=0.25, training=self.training)
        return x + self.scale * torch.tanh(hidden + 0.1 * context + 0.2 * t_mod)


@pytest.fixture(autouse=True)
def _ordinary_checkpoint_and_restored_rng(monkeypatch):
    monkeypatch.setattr(checkpoint_owner, "_HAS_DEEPSPEED", False)
    with torch.random.fork_rng(devices=[]):
        yield


def _inputs():
    generator = torch.Generator().manual_seed(210)
    return {
        name: torch.randn(3, 4, generator=generator).requires_grad_(name != "freqs")
        for name in ("x", "context", "t_mod", "freqs", "offset")
    }


def _run(blocks, source, *, offload: bool, autocast_enabled: bool):
    blocks = copy.deepcopy(blocks)
    inputs = {name: tensor.detach().clone().requires_grad_(tensor.requires_grad)
              for name, tensor in source.items()}
    shared_pointers = tuple(inputs[name].data_ptr()
                            for name in ("context", "t_mod", "freqs", "offset"))
    torch.manual_seed(824)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast_enabled):
        x = inputs["x"]
        for block in blocks:
            args = (x, inputs["context"], inputs["t_mod"], inputs["freqs"])
            if offload:
                x = checkpoint_owner.gradient_checkpoint_forward(
                    block, True, True, *args, offset=inputs["offset"])
            else:
                x = checkpoint(block, *args, offset=inputs["offset"], use_reentrant=False)
        loss = x.square().sum()
    after_forward = torch.get_rng_state().clone()
    loss.backward()
    after_backward = torch.get_rng_state().clone()
    gradients = {"parameter:" + name: parameter.grad.detach().clone()
                 for name, parameter in blocks.named_parameters()}
    gradients.update({"input:" + name: value.grad.detach().clone()
                      for name, value in inputs.items() if value.requires_grad})
    assert all(torch.isfinite(value).all() for value in gradients.values())
    assert all(torch.count_nonzero(value) for value in gradients.values())
    for block in blocks:
        assert len(block.calls) > 1  # 反向确实经过重算。
        assert all(call["shared"] == shared_pointers for call in block.calls)
        assert all(call["autocast"] is autocast_enabled for call in block.calls)
    for name, value in inputs.items():
        assert torch.equal(value, source[name])
    return {"output": x.detach().clone(), "gradients": gradients,
            "rng_forward": after_forward, "rng_backward": after_backward,
            "calls": [block.calls for block in blocks]}


@pytest.mark.parametrize("autocast_enabled", [False, True])
def test_two_blocks_shared_conditions_match_nonreentrant_gradients_and_rng(monkeypatch, autocast_enabled):
    torch.manual_seed(42)
    blocks = nn.ModuleList([_ConditionedBlock(0.7), _ConditionedBlock(1.3)])
    source = _inputs()
    baseline = _run(blocks, source, offload=False, autocast_enabled=autocast_enabled)
    actual_cpu_offload = _run(blocks, source, offload=True, autocast_enabled=autocast_enabled)
    assert torch.equal(actual_cpu_offload["output"], baseline["output"])
    assert actual_cpu_offload["gradients"].keys() == baseline["gradients"].keys()
    for name, gradient in actual_cpu_offload["gradients"].items():
        assert torch.equal(gradient, baseline["gradients"][name]), name
    assert torch.equal(actual_cpu_offload["rng_forward"], baseline["rng_forward"])
    assert torch.equal(actual_cpu_offload["rng_backward"], baseline["rng_backward"])
    assert torch.equal(actual_cpu_offload["rng_forward"], actual_cpu_offload["rng_backward"])
    packed_by_call = []

    @contextmanager
    def copy_saved_inputs_to_cpu():
        packed = []

        def pack(tensor):
            packed.append({"pointer": tensor.data_ptr(), "numel": tensor.numel(),
                           "shape": tuple(tensor.shape)})
            # CPU上模拟真正的设备搬运：新storage，不保留原tensor引用。
            return tensor.detach().clone()

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            yield
        packed_by_call.append(packed)

    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", copy_saved_inputs_to_cpu)
    replay = [_run(blocks, source, offload=True, autocast_enabled=autocast_enabled)
              for _ in range(2)]
    assert len(packed_by_call) == 4  # 两个block，各做两次独立replay。
    for run, pairs in zip(replay, (packed_by_call[:2], packed_by_call[2:]), strict=True):
        assert torch.equal(run["output"], baseline["output"])
        assert run["gradients"].keys() == baseline["gradients"].keys()
        for name, value in run["gradients"].items():
            assert torch.equal(value, baseline["gradients"][name]), name
        assert torch.equal(run["rng_forward"], baseline["rng_forward"])
        assert torch.equal(run["rng_backward"], baseline["rng_backward"])
        assert torch.equal(run["rng_forward"], run["rng_backward"])
        for calls, packed in zip(run["calls"], pairs, strict=True):
            assert [row["numel"] for row in packed] == [0, source["x"].numel()]
            assert packed[1]["shape"] == tuple(source["x"].shape)
            assert packed[1]["pointer"] == calls[0]["x"]
            assert calls[1]["x"] != calls[0]["x"]  # 重算实际使用搬回的新storage。


@pytest.mark.parametrize("offload", [True, False], ids=["input_offload", "self_attention_context"])
def test_no_grad_calls_each_original_block_without_offload_or_checkpoint(monkeypatch, offload):
    torch.manual_seed(42)
    blocks = nn.ModuleList([_ConditionedBlock(0.7), _ConditionedBlock(1.3)])
    source = _inputs()

    def unexpected(*args, **kwargs):
        pytest.fail("no_grad must execute the original module directly")

    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", unexpected)
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", unexpected)
    context_calls = []

    @contextmanager
    def selective_context():
        context_calls.append("entered")
        yield

    def context_fn():
        context_calls.append("created")
        return selective_context(), selective_context()

    def run(use_helper):
        copied = copy.deepcopy(blocks)
        torch.manual_seed(824)
        with torch.no_grad():
            x = source["x"]
            for block in copied:
                args = (x, source["context"], source["t_mod"], source["freqs"])
                x = (checkpoint_owner.gradient_checkpoint_forward(
                    block, True, offload, *args, offset=source["offset"],
                    context_fn=None if offload else context_fn)
                    if use_helper else block(*args, offset=source["offset"]))
        assert not x.requires_grad
        assert all(len(block.calls) == 1 for block in copied)
        assert all(parameter.grad is None for parameter in copied.parameters())
        return x, torch.get_rng_state().clone()

    expected, expected_rng = run(False)
    actual, actual_rng = run(True)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_rng, expected_rng)
    assert context_calls == []


def test_ordinary_deepspeed_branch_retains_argument_order_and_shared_gradients(monkeypatch):
    class PositionalOffsetBlock(_ConditionedBlock):
        def forward(self, x, context, t_mod, freqs, offset):
            return super().forward(x, context, t_mod, freqs, offset=offset)

    calls = []

    def deepspeed_checkpoint(function, *args):
        calls.append(tuple(value.data_ptr() for value in args))
        return checkpoint(function, *args, use_reentrant=True)

    monkeypatch.setattr(checkpoint_owner, "_HAS_DEEPSPEED", True)
    monkeypatch.setattr(checkpoint_owner, "deepspeed", SimpleNamespace(checkpointing=SimpleNamespace(
        is_configured=lambda: True, checkpoint=deepspeed_checkpoint)), raising=False)

    def unexpected(*args, **kwargs):
        pytest.fail("ordinary DeepSpeed checkpointing must not enter CPU offload")

    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", unexpected)
    torch.manual_seed(42)
    blocks = nn.ModuleList([PositionalOffsetBlock(0.7), PositionalOffsetBlock(1.3)])
    source = _inputs()

    def run(use_helper):
        copied = copy.deepcopy(blocks)
        inputs = {name: value.detach().clone().requires_grad_(value.requires_grad)
                  for name, value in source.items()}
        torch.manual_seed(824)
        x = inputs["x"]
        for block in copied:
            args = (x, inputs["context"], inputs["t_mod"], inputs["freqs"])
            x = (checkpoint_owner.gradient_checkpoint_forward(
                block, True, False, *args, offset=inputs["offset"])
                if use_helper else checkpoint(block, *args, inputs["offset"], use_reentrant=True))
        before_backward = torch.get_rng_state().clone()
        x.square().sum().backward()
        gradients = {"parameter:" + name: parameter.grad.detach().clone()
                     for name, parameter in copied.named_parameters()}
        gradients.update({"input:" + name: value.grad.detach().clone()
                          for name, value in inputs.items() if value.requires_grad})
        assert torch.equal(before_backward, torch.get_rng_state())
        for block in copied:
            assert len(block.calls) == 2
        return x.detach(), gradients, torch.get_rng_state().clone(), copied

    expected, expected_gradients, expected_rng, _ = run(False)
    actual, actual_gradients, actual_rng, copied = run(True)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_rng, expected_rng)
    assert actual_gradients.keys() == expected_gradients.keys()
    for name, gradient in actual_gradients.items():
        assert torch.equal(gradient, expected_gradients[name]), name
    assert len(calls) == 2
    for saved, block in zip(calls, copied, strict=True):
        assert saved == (block.calls[0]["x"], *block.calls[0]["shared"])
        assert block.calls[1]["shared"] == block.calls[0]["shared"]
