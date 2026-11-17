"""Frozen inference may reuse batch-one text encodings; training keeps real graphs."""
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

RUNTIME = Path(__file__).resolve().parents[3] / "third_party" / "wan22_runtime"
sys.path.insert(0, str(RUNTIME))
from diffsynth.pipelines.wan_video import WanVideoUnit_PromptEmbedder


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.calls = 0

    def forward(self, ids, mask):
        self.calls += 1
        return ids[..., None].float() * self.scale


class Tokenizer:
    def __call__(self, text, **kwargs):
        return torch.tensor([[ord(text[0]), len(text), 0, 0]]), torch.tensor([[1, 1, 0, 0]])


def pipe():
    encoder = Encoder().eval().requires_grad_(False)
    return SimpleNamespace(text_encoder=encoder, tokenizer=Tokenizer(), device=torch.device("cpu"))


def test_repeated_context_is_exact_rng_neutral_and_independent_storage():
    runtime = pipe()
    original, cached = WanVideoUnit_PromptEmbedder(), WanVideoUnit_PromptEmbedder()
    cached.cache_inference_prompts()
    with torch.no_grad():
        expected = original.encode_prompt(runtime, "low quality")
        rng = torch.random.get_rng_state()
        first = cached.encode_prompt(runtime, "low quality")
        first.fill_(123)  # A downstream consumer cannot overwrite cached data.
        second = cached.encode_prompt(runtime, "low quality")
    assert torch.equal(second, expected)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert runtime.text_encoder.calls == 2


def test_training_unit_stays_uncached_and_gradients_are_not_reused():
    runtime = pipe()
    runtime.text_encoder.train().requires_grad_(True)
    unit = WanVideoUnit_PromptEmbedder()
    for _ in range(2):
        unit.encode_prompt(runtime, "A person turns.").sum().backward()
    assert runtime.text_encoder.calls == 2
    assert runtime.text_encoder.scale.grad.item() == 2 * (ord("A") + len("A person turns."))
    assert unit._inference_cache is None


def test_enabled_cache_never_reuses_autograd_or_train_mode_results():
    runtime, unit = pipe(), WanVideoUnit_PromptEmbedder()
    unit.cache_inference_prompts()
    with torch.no_grad():
        unit.encode_prompt(runtime, "low quality")
    runtime.text_encoder.train().requires_grad_(True)
    for _ in range(2):
        unit.encode_prompt(runtime, "low quality").sum().backward()
    runtime.text_encoder.eval().requires_grad_(False)
    with torch.no_grad():
        unit.encode_prompt(runtime, "low quality")
    assert runtime.text_encoder.calls == 4


def test_memory_limit_evicts_oldest_context_without_changing_results():
    runtime, unit = pipe(), WanVideoUnit_PromptEmbedder()
    unit.cache_inference_prompts()
    unit._inference_cache_limit = 32  # two real float32 [1,4,1] contexts
    with torch.no_grad():
        expected = unit.encode_prompt(runtime, "first")
        unit.encode_prompt(runtime, "second")
        unit.encode_prompt(runtime, "second")
        unit.encode_prompt(runtime, "third")
        actual = unit.encode_prompt(runtime, "first")
    assert torch.equal(actual, expected)
    assert runtime.text_encoder.calls == 4
    assert unit._inference_cache_bytes <= unit._inference_cache_limit


def test_encoder_replacement_discards_previous_contexts():
    runtime, unit = pipe(), WanVideoUnit_PromptEmbedder()
    unit.cache_inference_prompts()
    with torch.no_grad():
        before = unit.encode_prompt(runtime, "low quality")
        runtime.text_encoder = Encoder().eval().requires_grad_(False)
        runtime.text_encoder.scale.fill_(3)
        after = unit.encode_prompt(runtime, "low quality")
    assert not torch.equal(before, after)
    assert runtime.text_encoder.calls == 1
