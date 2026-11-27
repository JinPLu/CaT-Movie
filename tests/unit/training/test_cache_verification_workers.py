"""CPU调度边界；数学counter仅测并行语义，真实tokenizer由200行实料核验。"""
from copy import copy
import json
import multiprocessing
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mspc.training import cache_readiness


class RecordedPanel:
    """真实BlockDataset.text行仅作调度传输fixture，不代替实际tokenizer验收。"""
    def __init__(self, rows):
        self.rows = {row["sample_index"]: row for row in rows}

    def text(self, index):
        row = self.rows[index]
        return SimpleNamespace(prompts=tuple(row["prompts"]), metadata=row["metadata"])


def _math_counter(text):
    assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"
    return len(text.split()) + 1


def _load_math_counter(path):
    assert Path(path) == Path("mathematical-tokenizer")
    assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"
    return _math_counter


def _unusable_parent_counter(text):
    raise AssertionError("worker must construct its own original counter")


class MathPanel:
    def __init__(self, name, *, broken=None):
        self.name, self.broken = name, broken
        self.token_counter = _unusable_parent_counter
        self.targets = {index: "physical target boundary fixture" for index in range(5)}

    def text(self, index):
        if self.broken == "text_error":
            raise RuntimeError(f"original text failed: {self.name}/{index}")
        prompts = (f"{self.name} original {index}", f"reference {index}")
        counts = [self.token_counter(prompt) for prompt in prompts]
        metadata = {
            "sample_index": index, "case_id": f"{self.name}-case-{index}",
            "actual_token_counts": counts, "token_counts": list(counts),
            "flow_draw_seed": 2**61 + index * 13,
            "references": [[index + 1, index * 4]], "span": [index, index + 1],
            "layout_text_dropped": index % 2 == 0,
            "reference_text_dropped": False,
        }
        if self.broken == "wrong_metadata_index":
            metadata["sample_index"] += 1
        if self.broken == "over_budget":
            metadata["actual_token_counts"][0] = 513
        return SimpleNamespace(prompts=prompts, metadata=metadata)


class EarlyErrorPanel(MathPanel):
    def __init__(self, started, release, finished):
        super().__init__("train")
        self.started, self.release, self.finished = started, release, finished

    def text(self, index):
        if index == 0:
            self.started.set()
            self.release.wait(timeout=3)
            self.finished.set()
        elif index == 1:
            assert self.started.wait(timeout=3)
            raise RuntimeError("later original index failed before first completed")
        return super().text(index)


@pytest.fixture(autouse=True)
def math_counter(monkeypatch):
    monkeypatch.setattr("mspc.data.shot_text.load_umt5_token_counter", _load_math_counter)


def rows(panels, workers):
    return list(cache_readiness._verified_text_rows(
        panels, verification_workers=workers, tokenizer_path="mathematical-tokenizer"))


def expected_rows(panels):
    values = []
    with cache_readiness._serial_tokenizer_calls():
        for split, panel, indices in panels:
            original = copy(panel)
            original.token_counter = _math_counter
            for index in indices:
                sample = original.text(index)
                values.append({"split": split, "sample_index": index,
                               "metadata": sample.metadata, "prompts": sample.prompts})
    return values


def test_one_and_eight_workers_preserve_original_order_all_fields_and_rng(monkeypatch):
    panels = [("train", MathPanel("train"), [9, 2, 7]),
              ("validation", MathPanel("validation"), [2, 7])]
    expected = expected_rows(panels)
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    original_threads = torch.get_num_threads()

    for workers in (1, 8):
        assert rows(panels, workers) == expected
        assert os.environ["TOKENIZERS_PARALLELISM"] == "true"
        assert torch.get_num_threads() == original_threads
        assert random.getstate() == python_rng
        current = np.random.get_state()
        assert current[0] == numpy_rng[0] and current[2:] == numpy_rng[2:]
        np.testing.assert_array_equal(current[1], numpy_rng[1])
        assert torch.equal(torch.get_rng_state(), torch_rng)
    assert [value["sample_index"] for value in expected] == [9, 2, 7, 2, 7]
    assert all(panel.token_counter is _unusable_parent_counter for _, panel, _ in panels)


def test_real_corpus_fixture_keeps_every_original_field_and_string():
    path = Path(__file__).resolve().parents[2] / "fixtures/cache_stage_verification_real_rows.json"
    original = json.loads(path.read_text())["rows"]
    panels = []
    for split in ("train", "validation"):
        selected = [row for row in original if row["split"] == split]
        panels.append((split, RecordedPanel(selected), [row["sample_index"] for row in selected]))
    for workers in (1, 8):
        actual = rows(panels, workers)
        assert json.loads(json.dumps(actual, ensure_ascii=False)) == original


@pytest.mark.parametrize("panels", [
    [("train", MathPanel("train"), [0]), ("train", MathPanel("other"), [1])],
    [("train", MathPanel("train"), [1, 1])],
    [("train", MathPanel("train"), [-1])],
    [("train", MathPanel("train"), [True])],
    [("wrong-split", MathPanel("train"), [1])],
    [("train", MathPanel("train"), [])],
])
def test_invalid_split_or_original_indices_fail_before_starting_workers(panels, monkeypatch):
    monkeypatch.setattr(cache_readiness, "ProcessPoolExecutor",
                        lambda **kwargs: pytest.fail("invalid original keys reached process creation"))
    with pytest.raises(ValueError, match="distinct original indices"):
        rows(panels, 8)


@pytest.mark.parametrize("workers", [None, 0, -1, False, 8.0])
def test_worker_count_is_explicit_positive_integer(workers, monkeypatch):
    monkeypatch.setattr(cache_readiness, "ProcessPoolExecutor",
                        lambda **kwargs: pytest.fail("invalid worker count reached process creation"))
    with pytest.raises(ValueError, match="verification_workers"):
        rows([("train", MathPanel("train"), [0])], workers)


@pytest.mark.parametrize("workers", [1, 8])
def test_original_panel_text_error_is_propagated(workers):
    with pytest.raises(RuntimeError, match="original text failed: validation/5"):
        rows([("validation", MathPanel("validation", broken="text_error"), [5])], workers)


def test_later_worker_error_does_not_wait_for_original_first_row():
    context = multiprocessing.get_context("fork")
    started, release, finished = context.Event(), context.Event(), context.Event()
    original_threads = torch.get_num_threads()
    try:
        with pytest.raises(RuntimeError, match="later original index failed"):
            rows([("train", EarlyErrorPanel(started, release, finished), [0, 1])], 8)
        assert not finished.is_set(), "parent waited for earlier row before propagating an available error"
        assert torch.get_num_threads() == original_threads
    finally:
        release.set()
        assert finished.wait(timeout=5)


@pytest.mark.parametrize("broken,match", [
    ("wrong_metadata_index", "changed or duplicated original sample identity"),
    ("over_budget", "compiled text has 513 umT5 tokens"),
])
def test_invalid_worker_metadata_or_token_budget_is_rejected(broken, match):
    with pytest.raises(ValueError, match=match):
        rows([("train", MathPanel("train", broken=broken), [4])], 8)


def _duplicated_transport_row(key):
    split, index = key
    sample = cache_readiness._VERIFICATION_PANELS[split].text(index)
    return {"split": split, "sample_index": 0,
            "metadata": {**sample.metadata, "sample_index": 0}, "prompts": sample.prompts}


def test_matching_aggregate_counts_cannot_hide_duplicate_or_missing_identity(monkeypatch):
    monkeypatch.setattr(cache_readiness, "_verify_text_sample", _duplicated_transport_row)
    with pytest.raises(ValueError, match="changed or duplicated original sample identity"):
        rows([("train", MathPanel("train"), [0, 1])], 8)


def test_text_cache_preflight_requires_an_identical_rank_prefix():
    published = {
        "target_step": 4, "rank_sample_indices": [[3, 1, 2], [6, 5, 4]],
        "sample_indices": [1, 2, 3, 4, 5, 6], "target_keys": [["a", 1], ["a", 2]],
        "validation_sample_indices": [0, 1], "training_seed": 42,
    }
    requested = {**published, "target_step": 2, "rank_sample_indices": [[3, 1], [6, 5]],
                 "sample_indices": [1, 3, 5, 6], "target_keys": [["a", 1]]}
    cache_readiness._require_text_stage_coverage(published, published)
    cache_readiness._require_text_stage_coverage(requested, published)
    for change in ({"rank_sample_indices": [[1, 3], [6, 5]]},
                   {"target_keys": [["missing", 1]]}, {"validation_sample_indices": [0]}):
        with pytest.raises(ValueError, match="identical rank-wise sampler prefix"):
            cache_readiness._require_text_stage_coverage({**requested, **change}, published)


@pytest.mark.parametrize("workers", [1, 8])
def test_public_verification_keeps_physical_stage_and_both_original_splits(tmp_path, monkeypatch, workers):
    from mspc.training import block_dataset, training_args

    plan = {"artifact_type": cache_readiness.ARTIFACT_TYPE, "schema_version": 1,
            "target_step": 12000, "sample_indices": [2, 5], "validation_sample_indices": [1]}
    stage = tmp_path / "actual-stage.json"
    stage.write_text(json.dumps(plan))
    args = SimpleNamespace(dataset_cfg={"tokenizer_path": "mathematical-tokenizer"},
                           validation={"original_validation": True})
    train, validation = MathPanel("train"), MathPanel("validation")
    calls = []
    monkeypatch.setattr(training_args, "build_common_training_args", lambda config: args)
    def topology(value, *, gpus_per_node, nnodes):
        assert value is args and (gpus_per_node, nnodes) == (8, 1)
        calls.append("topology")
        return value
    monkeypatch.setattr(training_args, "configure_training_topology", topology)
    def replace(value, **updates):
        return SimpleNamespace(**{**vars(value), **updates})
    monkeypatch.setattr(training_args, "replace_training_args", replace)
    def build_train(value):
        assert value.cache_stage_plan == plan
        calls.append("physical_train")
        return train
    def require(value, current_args, *, world_size):
        assert value is train and current_args.cache_stage_plan == plan and world_size == 8
        calls.append("actual_stage_and_physical_constraints")
    def build_validation(value):
        assert value.cache_stage_plan == plan and value.validation == args.validation
        calls.append("physical_validation")
        return validation
    monkeypatch.setattr(block_dataset, "build_training_dataset", build_train)
    monkeypatch.setattr(cache_readiness, "require_cache_stage", require)
    monkeypatch.setattr(block_dataset, "build_validation_dataset", build_validation)

    result = cache_readiness.verify_config_cache_stage(
        SimpleNamespace(resolved={"dataset": {"type": "block_latent_cache"}}),
        stage, world_size=8, verification_workers=workers)

    assert calls == ["topology", "physical_train", "actual_stage_and_physical_constraints", "physical_validation"]
    assert result == {"target_step": 12000, "ready_samples": 2, "ready_target_shots": 5,
                      "validation_samples": 1, "compiled_prompt_rows": 6, "max_prompt_tokens": 4,
                      "verification_workers": workers,
                      "verified_samples_by_split": {"train": 2, "validation": 1}}
