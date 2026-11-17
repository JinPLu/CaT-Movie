"""训练中的留出验证 loss：每一遍用同一组噪声与时间步、不扰动训练的随机数、各 rank 只量自己那份。"""
from __future__ import annotations

import json

import pytest
import torch

from mspc.training.validation_loss import ValidationLoss


class Windows:
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {"input_latents": torch.full((1, 1, 2, 1, 1), float(index)), "prompt": ["row"],
                "_sample_metadata": {"case_id": f"case-{index}", "span": [0, 1 + index % 3], "references": []},
                "_dataset_timing": {"shot_count": 1}}


class Module(torch.nn.Module):
    """像训练模块那样消耗随机数：先抽时间步（CPU），再抽噪声。loss 取决于权重、时间步与噪声。"""

    def __init__(self, weight):
        super().__init__()
        self.weight, self.last_timing, self.seen = weight, {}, []

    def __call__(self, data):
        assert not any(key.startswith("_") for key in data)
        timestep = float(torch.randint(0, 1000, (1,)))
        noise = torch.randn_like(data["input_latents"])
        self.last_timing = {"timestep": timestep}
        self.seen.append(float(data["input_latents"].flatten()[0]))
        return (self.weight * noise.abs().mean() + timestep / 1000).reshape(())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_every_pass_uses_the_same_noise_and_timesteps_and_leaves_training_randomness_alone(tmp_path):
    validation = ValidationLoss(Windows(5), every_steps=250, draws=2, seed=42, rank=0, world=1, log_dir=tmp_path, loader_kwargs={"batch_size": 1, "num_workers": 0})
    assert [validation.due(step) for step in (0, 1, 250, 251, 500)] == [True, False, True, False, True]
    torch.manual_seed(7)
    expected_next = torch.rand(3)
    torch.manual_seed(7)
    validation.run(Module(1.0), 0)
    assert torch.equal(torch.rand(3), expected_next)  # 训练接下来抽到的数与没量验证时一样
    validation.run(Module(1.0), 250)
    validation.run(Module(3.0), 500)
    lines = rows(tmp_path / "validation.rank0.jsonl")
    assert len(lines) == 3 * 5 * 2
    by_step = {step: [line for line in lines if line["step"] == step] for step in (0, 250, 500)}
    key = lambda line: (line["sample_index"], line["draw"], line["timestep"])  # noqa: E731
    assert [key(line) for line in by_step[0]] == [key(line) for line in by_step[250]] == [key(line) for line in by_step[500]]
    assert [line["loss"] for line in by_step[0]] == [line["loss"] for line in by_step[250]]  # 权重没变，读数逐条相同
    assert all(late["loss"] > early["loss"] for early, late in zip(by_step[0], by_step[500]))  # 差只来自权重
    assert {line["shots"] for line in lines} == {1, 2, 3}
    assert len({line["timestep"] for line in by_step[0]}) > 5  # 两次抽签、不同的窗，时间步各不相同


def test_each_rank_measures_its_own_share_without_talking_to_the_others(tmp_path):
    seen = set()
    for rank in range(3):
        module = Module(1.0)
        ValidationLoss(Windows(8), every_steps=1, draws=1, seed=1, rank=rank, world=3, log_dir=tmp_path, loader_kwargs={"batch_size": 1, "num_workers": 0}).run(module, 0)
        assert module.seen == [float(index) for index in range(8) if index % 3 == rank]
        seen |= {line["case_id"] for line in rows(tmp_path / f"validation.rank{rank}.jsonl")}
    assert seen == {f"case-{index}" for index in range(8)}


def test_cached_text_validation_preserves_original_sample_metadata_for_each_noise_draw(tmp_path):
    class CachedWindows(Windows):
        def __init__(self):
            super().__init__(3)
            self.original = []

        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample["_sample_metadata"].update(sample_index=index, flow_draw_seed=73,
                actual_token_counts=[2], reference_text_dropped=False)
            self.original.append(sample["_sample_metadata"])
            return sample

    class CachedModule(Module):
        def __init__(self):
            super().__init__(1.0)
            self.model_cfg = {"text_context_cache": "/explicit/manifest.json"}
            self.exposures = []

        def __call__(self, data):
            envelope = data.pop("_text_cache_sample")
            assert envelope["split"] == "validation"
            self.exposures.append((envelope["metadata"], data["flow_draw_seed"]))
            return super().__call__(data)

    dataset, model = CachedWindows(), CachedModule()
    validation = ValidationLoss(dataset, every_steps=1, draws=2, seed=42, rank=1, world=2,
        log_dir=tmp_path, loader_kwargs={"batch_size": 1, "num_workers": 0})
    validation.run(model, 0)

    assert len(dataset.original) == 1 and len(model.exposures) == 2
    assert all(metadata is dataset.original[0] for metadata, _ in model.exposures)
    assert len({seed for _, seed in model.exposures}) == 2
    assert dataset.original[0] == {"case_id": "case-1", "span": [0, 2], "references": [],
        "sample_index": 1, "flow_draw_seed": 73, "actual_token_counts": [2],
        "reference_text_dropped": False}
    assert all(row["sample_index"] == 1 for row in rows(validation.path))


def test_repeated_block_has_distinct_pairing_keys_noise_and_reference_groups(tmp_path):
    class RepeatedBlock(Windows):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample["_sample_metadata"] = {
                "case_id": "same-block", "span": [index, index + 1],
                "references": [] if index == 0 else [[1, 4]],
            }
            return sample

    validation = ValidationLoss(RepeatedBlock(2), every_steps=250, draws=2, seed=42,
                                rank=0, world=1, log_dir=tmp_path, loader_kwargs={"batch_size": 1, "num_workers": 0})
    torch.manual_seed(17)
    state = torch.get_rng_state().clone()
    for step in (0, 250):
        validation.run(Module(1.0), step)
        assert torch.equal(torch.get_rng_state(), state)
    lines = rows(tmp_path / "validation.rank0.jsonl")
    early, late = lines[:4], lines[4:]
    assert len({(row["sample_index"], row["draw"]) for row in early}) == 4
    assert len({(row["case_id"], row["draw"]) for row in early}) == 2
    assert {tuple(row["span"]) for row in early} == {(0, 1), (1, 2)}
    assert [row["references"] for row in early] == [[], [], [[1, 4]], [[1, 4]]]
    assert [row["reference_count"] for row in early] == [0, 0, 1, 1]
    measured = {"step", "data_seconds", "forward_seconds"}
    assert all({k: v for k, v in a.items() if k not in measured}
               == {k: v for k, v in b.items() if k not in measured} for a, b in zip(early, late))
    assert [(row["timestep"], row["loss"]) for row in early[:2]] != [
        (row["timestep"], row["loss"]) for row in early[2:]]


def test_validation_disables_dropout_and_restores_mixed_modes_even_on_failure(tmp_path):
    import pytest

    class StatefulModule(Module):
        def __init__(self):
            super().__init__(1.0)
            self.dropout = torch.nn.Dropout(0.75)
            self.frozen = torch.nn.Linear(1, 1).eval()
            self.fail = False

        def __call__(self, data):
            assert not self.training and not self.dropout.training and not self.frozen.training
            assert torch.equal(self.dropout(torch.ones(16)), torch.ones(16))
            if self.fail:
                raise RuntimeError('observed forward failure')
            return super().__call__(data)

    model = StatefulModule()
    validation = ValidationLoss(Windows(2), every_steps=1, draws=1, seed=1, rank=0, world=1,
        log_dir=tmp_path, loader_kwargs={'batch_size':1, 'num_workers':0})
    validation.run(model, 0)
    assert model.training and model.dropout.training and not model.frozen.training
    model.fail = True
    torch.manual_seed(22)
    state = torch.get_rng_state()
    with pytest.raises(RuntimeError, match='observed forward failure'):
        validation.run(model, 1)
    assert model.training and model.dropout.training and not model.frozen.training
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize("fail", [False, True])
def test_validation_preserves_python_numpy_and_torch_random_streams(tmp_path, fail):
    import random
    import numpy as np

    class MixedRandomModule(Module):
        def __call__(self, data):
            random.random()
            np.random.random()
            result = super().__call__(data)
            if fail:
                raise RuntimeError("observed forward failure")
            return result

    def draw():
        return random.random(), float(np.random.random()), float(torch.rand(()))

    random.seed(19); np.random.seed(19); torch.manual_seed(19)
    expected = draw()
    random.seed(19); np.random.seed(19); torch.manual_seed(19)
    validation = ValidationLoss(Windows(2), every_steps=1, draws=1, seed=1, rank=0, world=1,
        log_dir=tmp_path, loader_kwargs={"batch_size": 1, "num_workers": 0})
    if fail:
        with pytest.raises(RuntimeError, match="observed forward failure"):
            validation.run(MixedRandomModule(1.0), 1)
    else:
        validation.run(MixedRandomModule(1.0), 1)
    assert draw() == expected


@pytest.mark.parametrize("workers", [0, 1])
def test_intermediate_panel_is_a_fixed_subset_of_full_noise_draws_across_ranks(tmp_path, workers):
    torch.manual_seed(17)
    state = torch.get_rng_state().clone()
    for rank in range(3):
        validation = ValidationLoss(Windows(8), every_steps=500, draws=2, seed=42,
            rank=rank, world=3, log_dir=tmp_path,
            loader_kwargs={"batch_size": 1, "num_workers": workers, "persistent_workers": workers > 0},
            intermediate_panel={"sample_budget": 4, "draws": 1})
        for step in (0, 500, 1000):
            validation.run(Module(1.0), step)
            assert torch.equal(torch.get_rng_state(), state)
        validation.run(Module(1.0), 3000, full=True)
        assert torch.equal(torch.get_rng_state(), state)
    lines = [line for rank in range(3) for line in rows(tmp_path / f"validation.rank{rank}.jsonl")]
    by_step = {step: [row for row in lines if row["step"] == step] for step in (0, 500, 1000, 3000)}
    assert [len(by_step[step]) for step in by_step] == [16, 4, 4, 16]
    paired = lambda row: (row["sample_index"], row["draw"], row["timestep"], row["loss"])
    full_subset = {paired(row) for row in by_step[0] if row["sample_index"] < 4 and row["draw"] == 0}
    assert {paired(row) for row in by_step[500]} == {paired(row) for row in by_step[1000]} == full_subset
    assert {paired(row) for row in by_step[3000]} == {paired(row) for row in by_step[0]}
    assert {(row["validation_panel"], row["validation_forwards"]) for row in by_step[500]} == {("intermediate", 4)}
    assert {(row["validation_panel"], row["validation_forwards"]) for row in by_step[3000]} == {("full", 16)}


def test_intermediate_panel_rejects_a_denominator_larger_than_the_full_panel(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="sample_budget"):
        ValidationLoss(Windows(8), every_steps=500, draws=2, seed=42, rank=0, world=1,
            log_dir=tmp_path, loader_kwargs={"batch_size": 1, "num_workers": 0},
            intermediate_panel={"sample_budget": 9, "draws": 1})


def test_capacity_subset_preserves_original_indices_seeds_draws_and_rank_partition(tmp_path):
    for rank in range(2):
        full_dir, subset_dir = tmp_path / 'full', tmp_path / 'subset'
        full_dir.mkdir(exist_ok=True); subset_dir.mkdir(exist_ok=True)
        settings = dict(every_steps=1, draws=2, seed=42, rank=rank, world=2,
                        loader_kwargs={'batch_size':1, 'num_workers':0})
        full = ValidationLoss(Windows(8), log_dir=full_dir, **settings)
        subset = ValidationLoss(Windows(8), log_dir=subset_dir,
                                intermediate_panel={'sample_budget':2, 'draws':1}, **settings)
        full.run(Module(1.0), 3, full=True)
        model = Module(1.0)
        torch.manual_seed(55); state = torch.get_rng_state().clone()
        subset.run(model, 3, sample_indices=[6,3,5])
        assert model.training and torch.equal(torch.get_rng_state(), state)
        all_rows = rows(full_dir / f'validation.rank{rank}.jsonl')
        selected = rows(subset_dir / f'validation.rank{rank}.jsonl')
        expected = [index for index in [6,3,5] if index % 2 == rank]
        assert model.seen == [float(index) for index in expected for _ in range(2)]
        assert {row['validation_forwards'] for row in selected} == {6}
        assert {row['validation_panel'] for row in selected} == {'capacity'}
        original = {(row['sample_index'],row['draw']):row for row in all_rows}
        for row in selected:
            other = original[row['sample_index'],row['draw']]
            assert (row['timestep'],row['loss']) == (other['timestep'],other['loss'])
        subset.run(model, 4, full=True)
        assert len(rows(subset_dir / f'validation.rank{rank}.jsonl')) == len(selected) + 8


def test_capacity_subset_rejects_invalid_original_indices_before_forward(tmp_path):
    import pytest
    validation = ValidationLoss(Windows(8), every_steps=1, draws=2, seed=1, rank=0, world=1,
        log_dir=tmp_path, loader_kwargs={'batch_size':1, 'num_workers':0})
    for indices in ([],[1,1],[8],[-1],[True],[1.0]):
        model = Module(1.0)
        with pytest.raises(ValueError,match='original dataset indices'):
            validation.run(model, 3, sample_indices=indices)
        assert model.seen == []
