"""真实块fixture验证固定文字排除保原ID、参考抽签和完整采样随机流。"""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mspc.training.block_dataset import BlockDataset
from mspc.training.loop import CostBucketedSampler, training_epoch_count
from mspc.training.text_admission import read_text_admission, write_text_admission
from mspc.data.shot_text import SHOT_TEXT_COMPILER_CONTRACT
from mspc.training.validation_loss import ValidationLoss
from mspc.config import load_experiment_config
from mspc.training.training_args import build_common_training_args as original_training_args
from tests.unit.training.test_block_dataset import materialize
from tests.unit.training.test_validation_loss import Windows, Module, rows
from tests.support.paths import REPO_ROOT


def config_args(inputs):
    validation = {name: inputs[name] for name in ("members_path", "sampling_manifest_path", "sample_budget")}
    validation.update(reference_entities=None, planner_tokens=None, text_admission_path=None)
    source = original_training_args(load_experiment_config(REPO_ROOT / "tests/fixtures/block_renderer_config.json"))
    dataset = {key: value for key, value in inputs.items() if key != "sample_budget"}
    return replace(source, dataset_cfg={**dataset, "planner_tokens": None}, validation=validation,
        max_train_steps=inputs["sample_budget"], per_device_batch_size=1, gradient_accumulation_steps=1)


def test_fixed_text_admission_keeps_original_draws_and_rejects_stale_or_partial_receipts(tmp_path, monkeypatch):
    inputs = {**materialize(tmp_path), "target_mask_weights": None}
    monkeypatch.setattr("mspc.training.block_dataset.load_umt5_token_counter", lambda _: lambda text: len(text.split()))
    before = BlockDataset(**inputs)
    too_long = before.text(0).prompts[-1]
    monkeypatch.setattr("mspc.training.block_dataset.load_umt5_token_counter",
        lambda _: lambda text: 513 if text == too_long else min(512, len(text.split())))
    source = config_args(inputs)
    monkeypatch.setattr("mspc.training.training_args.build_common_training_args", lambda _: source)
    output, audit = tmp_path / "admission.json", tmp_path / "rows.jsonl"
    result = write_text_admission(None, split="train", world_size=1, workers=1, latent_channels=48, vae_spatial_stride=16,
                                 output=output, rows_output=audit)
    assert 0 < result["excluded_count"] < inputs["sample_budget"]
    after = BlockDataset(**{**inputs, "text_admission_path": str(output)})
    assert len(after) == len(before) == inputs["sample_budget"]
    assert after.sampling_manifest == before.sampling_manifest
    removed = {row["sample_index"] for row in result["excluded_samples"]}
    assert set(after.eligible_sample_indices) == set(range(len(before))) - removed
    for index in range(len(before)):
        assert after.draw(index) == before.draw(index)
        assert after.text(index).prompts == before.text(index).prompts
    with pytest.raises(ValueError, match="excluded by the 512-token text admission"):
        after[next(iter(removed))]
    assert all(row["max_prompt_tokens"] > 512 for row in result["excluded_samples"])
    with pytest.raises(ValueError, match="actual members, labels"):
        BlockDataset(**{**inputs, "text_admission_path": str(output), "training_seed": 43})
    incomplete = {**result, "complete": False}
    (tmp_path / "partial.json").write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match="complete original sample-index"):
        read_text_admission(tmp_path / "partial.json", sample_budget=len(before))
    with pytest.raises(FileNotFoundError):
        BlockDataset(**{**inputs, "text_admission_path": str(tmp_path / "missing.json")})


@pytest.mark.parametrize("frozen_config_options", [False, True])
def test_zero_overflow_publishes_explicit_complete_admission(tmp_path, monkeypatch, frozen_config_options):
    inputs = {**materialize(tmp_path), "target_mask_weights": None}
    if frozen_config_options:
        from mspc.config.experiment import _freeze
        inputs = dict(_freeze(inputs))
    monkeypatch.setattr("mspc.training.block_dataset.load_umt5_token_counter", lambda _: lambda text: len(text.split()))
    monkeypatch.setattr("mspc.training.training_args.build_common_training_args", lambda _: config_args(inputs))
    result = write_text_admission(None, split="validation", world_size=1, workers=1, latent_channels=48, vae_spatial_stride=16,
                                 output=tmp_path / "val.json", rows_output=tmp_path / "rows.jsonl")
    assert result["complete"] is True
    assert result["checked_samples"] == result["eligible_count"] == len(result["eligible_sample_indices"])
    assert result["excluded_count"] == 0 and result["excluded_samples"] == []
    assert result["contract"]["prompt_compiler_contract"] == SHOT_TEXT_COMPILER_CONTRACT


@pytest.mark.parametrize("old_contract", [None, "reference_id_only"])
def test_admission_rejects_missing_or_changed_prompt_compiler_contract(tmp_path, monkeypatch, old_contract):
    inputs = {**materialize(tmp_path), "target_mask_weights": None}
    monkeypatch.setattr("mspc.training.block_dataset.load_umt5_token_counter", lambda _: lambda text: len(text.split()))
    monkeypatch.setattr("mspc.training.training_args.build_common_training_args", lambda _: config_args(inputs))
    output = tmp_path / "admission.json"
    published = write_text_admission(None, split="train", world_size=1, workers=1,
        latent_channels=48, vae_spatial_stride=16, output=output, rows_output=tmp_path / "rows.jsonl")
    assert read_text_admission(output, sample_budget=inputs["sample_budget"]) == published
    if old_contract is None:
        published["contract"].pop("prompt_compiler_contract")
    else:
        published["contract"]["prompt_compiler_contract"] = old_contract
    output.write_text(json.dumps(published))
    with pytest.raises(ValueError, match="current prompt compiler contract"):
        read_text_admission(output, sample_budget=inputs["sample_budget"])
    with pytest.raises(ValueError, match="current prompt compiler contract"):
        BlockDataset(**{**inputs, "text_admission_path": str(output)})


def test_parallel_workers_keep_each_original_record_and_admission_exact(tmp_path, monkeypatch):
    from mspc.config.experiment import _freeze
    inputs = dict(_freeze({**materialize(tmp_path), "target_mask_weights": None}))
    monkeypatch.setattr("mspc.training.block_dataset.load_umt5_token_counter", lambda _: lambda text: len(text.split()))
    monkeypatch.setattr("mspc.training.training_args.build_common_training_args", lambda _: config_args(inputs))
    reports = []
    for workers in (1, 3):
        reports.append(write_text_admission(None, split="train", world_size=1, workers=workers,
            latent_channels=48, vae_spatial_stride=16, output=tmp_path / f"admission-{workers}.json",
            rows_output=tmp_path / f"rows-{workers}.jsonl"))
    assert (tmp_path / "rows-1.jsonl").read_bytes() == (tmp_path / "rows-3.jsonl").read_bytes()
    variable = {"rows_output", "workers", "initialization_seconds", "production_seconds"}
    assert {k: v for k, v in reports[0].items() if k not in variable} == {
        k: v for k, v in reports[1].items() if k not in variable}


def test_sampler_drops_excluded_samples_before_grouping_so_every_update_stays_one_cost_group():
    """不准入样本先去掉再分组：去掉几条之后，每次更新的 8 条仍来自同一个成本组（11-16 错位修正）。"""
    costs = [2 + (index * 7) % 13 for index in range(1300)]
    excluded = {2, 17, 32, 511, 1002}
    eligible = [index for index in range(len(costs)) if index not in excluded]
    sampler = CostBucketedSampler(costs, generator=torch.Generator().manual_seed(42), bucket_size=8,
                                  eligible_indices=eligible)
    order = list(sampler)
    assert sorted(order) == eligible and len(sampler) == len(eligible)
    # 同一次更新内成本差：分组后再丢会让后面的更新横跨两组（差可到 12），先丢则每组都是窗口内排序后的连续 8 条。
    full = [order[start:start + 8] for start in range(0, len(order) - len(order) % 8, 8)]
    assert max(max(costs[i] for i in step) - min(costs[i] for i in step) for step in full) <= 1
    # 排列仍对完整原始索引抽取、不重新编号：第一个窗口正是原排列去掉不准入项后的前 512 条。
    first = torch.randperm(len(costs), generator=torch.Generator().manual_seed(42)).tolist()
    assert set(order[:512]) == set([index for index in first if index not in excluded][:512])


def test_published_text_admission_accepts_same_tokenizer_through_execution_root_alias(tmp_path, monkeypatch):
    inputs = {**materialize(tmp_path), "target_mask_weights": None}
    tokenizer = tmp_path / "shared-tokenizer"
    tokenizer.mkdir()
    inputs["tokenizer_path"] = str(tokenizer)
    monkeypatch.setattr("mspc.training.block_dataset.load_umt5_token_counter", lambda _: lambda text: len(text.split()))
    monkeypatch.setattr("mspc.training.training_args.build_common_training_args", lambda _: config_args(inputs))
    output = tmp_path / "admission.json"
    write_text_admission(None, split="train", world_size=1, latent_channels=48, vae_spatial_stride=16,
                         output=output, rows_output=tmp_path / "rows.jsonl", workers=1)
    alias = tmp_path / "execution-root-tokenizer"
    alias.symlink_to(tokenizer, target_is_directory=True)
    published = json.loads(output.read_text())
    published["contract"]["tokenizer_path"] = str(alias)
    output.write_text(json.dumps(published))
    consumer = BlockDataset(**{**inputs, "text_admission_path": str(output)})
    assert len(consumer.eligible_sample_indices) == inputs["sample_budget"]
    different = tmp_path / "different-tokenizer"
    different.mkdir()
    with pytest.raises(ValueError, match="actual members, labels"):
        BlockDataset(**{**inputs, "tokenizer_path": str(different), "text_admission_path": str(output)})


def test_filtered_tail_keeps_full_optimizer_schedule_only_when_needed():
    args = SimpleNamespace(num_epochs=1, max_train_steps=12000)
    filtered = SimpleNamespace(excluded_sample_indices=frozenset({7}))
    assert training_epoch_count(args, filtered, updates_per_epoch=11999) == 2
    assert training_epoch_count(args, filtered, updates_per_epoch=12000) == 1
    assert training_epoch_count(args, SimpleNamespace(), updates_per_epoch=11999) == 1


def test_validation_uses_original_prefix_and_actual_filtered_denominator(tmp_path):
    data = Windows(6)
    data.eligible_sample_indices = (0, 2, 4, 5)
    validation = ValidationLoss(data, every_steps=1, draws=2, seed=42, rank=0, world=1,
        log_dir=tmp_path, loader_kwargs={"batch_size": 1, "num_workers": 0},
        intermediate_panel={"sample_budget": 4, "draws": 1})
    full, intermediate = Module(1.), Module(1.)
    validation.run(full, 0)
    validation.run(intermediate, 1)
    assert full.seen == [0., 0., 2., 2., 4., 4., 5., 5.]
    assert intermediate.seen == [0., 2.]
    logged = rows(tmp_path / "validation.rank0.jsonl")
    assert {row["validation_forwards"] for row in logged if row["step"] == 0} == {8}
    assert {row["validation_forwards"] for row in logged if row["step"] == 1} == {2}
    with pytest.raises(ValueError, match="original dataset indices"):
        validation.run(full, 2, sample_indices=[1])
