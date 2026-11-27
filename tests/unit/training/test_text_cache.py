"""纯序列化边界：只裁掉原context的正零尾部，读取后逐字节恢复。"""
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from mspc.training import text_cache


def _original(count: int):
    ids = torch.arange(512, dtype=torch.int64).unsqueeze(0) + 100
    mask = torch.zeros((1, 512), dtype=torch.int64)
    mask[:, :count] = 1
    context = torch.zeros((1, 512, 4096), dtype=torch.bfloat16)
    values = torch.arange(count * 4096, dtype=torch.int32).remainder(257) - 128
    context[:, :count] = values.reshape(1, count, 4096).to(torch.bfloat16) / 8
    context[0, 0, 0] = -0.0  # 有效位置中的负零也必须原字节保留。
    return {"context": context, "ids": ids, "mask": mask}


@pytest.mark.parametrize("count", [1, 451, 512])
def test_compact_row_restores_original_context_ids_and_mask_bytes(tmp_path: Path, count: int):
    original = _original(count)
    path = tmp_path / "row.safetensors"

    payload_bytes = text_cache._write_text_context(path, **original, token_count=count)

    assert payload_bytes == count * 4096 * 2
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        assert stream.metadata() == {"schema_version": "2", "context_storage": "valid_token_rows"}
        compact = stream.get_tensor("context")
        assert compact.shape == (1, count, 4096)
        assert compact.dtype == torch.bfloat16
        assert compact.numel() * compact.element_size() == payload_bytes
    restored = text_cache.read_text_context(path)
    assert restored["context"].shape == (1, 512, 4096)
    assert restored["context"].device.type == "cpu"
    for name in ("context", "ids", "mask"):
        assert torch.equal(restored[name].view(torch.uint8), original[name].view(torch.uint8))
    assert torch.count_nonzero(restored["context"][:, count:].view(torch.uint8)) == 0


@pytest.mark.parametrize("padding_value", [1.0, -0.0])
def test_writer_rejects_padding_that_cannot_be_recovered_byte_exactly(tmp_path: Path, padding_value: float):
    original = _original(3)
    original["context"][0, 3, 0] = padding_value
    path = tmp_path / "row.safetensors"

    with pytest.raises(ValueError, match="nonzero padding bytes"):
        text_cache._write_text_context(path, **original, token_count=3)

    assert not path.exists()


@pytest.mark.parametrize("metadata", [None, {"schema_version": "1", "context_storage": "valid_token_rows"}])
def test_reader_rejects_unmarked_and_old_schema_files(tmp_path: Path, metadata):
    path = tmp_path / "row.safetensors"
    save_file(_original(512), str(path), metadata=metadata)

    with pytest.raises(ValueError, match="compact text context schema"):
        text_cache.read_text_context(path)


@pytest.mark.parametrize("broken", ["mask_gap", "empty_mask", "ids_dtype", "compact_shape", "context_dtype", "missing_mask"])
def test_reader_rejects_malformed_compact_rows(tmp_path: Path, broken: str):
    tensors = _original(3)
    tensors["context"] = tensors["context"][:, :3].contiguous()
    expected_message = ""
    if broken == "mask_gap":
        tensors["mask"][0, 1] = 0
        tensors["mask"][0, 3] = 1
        expected_message = "contiguous prefix"
    elif broken == "empty_mask":
        tensors["mask"].zero_()
        expected_message = "nonempty contiguous prefix"
    elif broken == "ids_dtype":
        tensors["ids"] = tensors["ids"].to(torch.int32)
        expected_message = "original int64 ids and mask"
    elif broken == "compact_shape":
        tensors["context"] = tensors["context"][:, :2].contiguous()
        expected_message = "geometry/dtype"
    elif broken == "context_dtype":
        tensors["context"] = tensors["context"].float()
        expected_message = "geometry/dtype"
    elif broken == "missing_mask":
        del tensors["mask"]
        expected_message = "compact text context schema"
    path = tmp_path / "row.safetensors"
    save_file(tensors, str(path), metadata=text_cache.CONTEXT_STORAGE)

    with pytest.raises(ValueError, match=expected_message):
        text_cache.read_text_context(path)


def _jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def four_shards(tmp_path: Path):
    source = tmp_path / "original"
    source.mkdir()
    # 故意不用连续或递增row_id；样本中还有重复引用，合并不能重新编号或排序。
    prompts = [
        {"row_id": index, "prompt": f"original prompt {index}", "token_count": count}
        for index, count in ((12, 1), (5, 2), (10, 3), (7, 4), (16, 1))
    ]
    samples = [
        {"split": "train", "sample_index": 90000, "context_rows": [7, 12, 7, 5],
         "metadata": {"references": [{"shot": "original reference"}]}},
        {"split": "validation", "sample_index": 17, "context_rows": [16, 10],
         "metadata": {"references": []}},
    ]
    encoding_contract = {"owner": "original batch-one encoder", "max_length": 512}
    manifest = {
        "complete": True, "prompt_manifest": "prompts.jsonl",
        "sample_manifest": "sample-context-rows.jsonl", "encoding_contract": encoding_contract,
        "splits": {"train": {"expected_samples": 1}, "validation": {"expected_samples": 1}},
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    _jsonl(source / "prompts.jsonl", prompts)
    _jsonl(source / "sample-context-rows.jsonl", samples)
    shards = []
    tensors = {}
    for index in range(4):
        shard = tmp_path / f"worker-{index}"
        (shard / "contexts").mkdir(parents=True)
        rows = [row for row in prompts if row["row_id"] % 4 == index]
        for row in rows:
            count, row_id = row["token_count"], row["row_id"]
            mask = torch.zeros((1, 512), dtype=torch.int64)
            mask[:, :count] = 1
            values = {
                "context": torch.full((1, count, 4096), row_id, dtype=torch.bfloat16),
                "ids": torch.arange(512, dtype=torch.int64).unsqueeze(0) + row_id,
                "mask": mask,
            }
            tensors[row_id] = values
            save_file(values, str(shard / "contexts" / f"{row_id}.safetensors"),
                      metadata=text_cache.CONTEXT_STORAGE)
        _jsonl(shard / "prompts.jsonl", rows)
        report = {
            "artifact_type": "mspc_training_text_context_cache_shard", "schema_version": 2,
            "context_storage": text_cache.CONTEXT_STORAGE,
            "storage_contract": text_cache.CONTEXT_STORAGE_CONTRACT,
            "shard": {"index": index, "count": 4,
                      "assignment": "original row_id modulo shard_count"},
            "complete_stage": False, "sample_limit_per_split": None,
            "selected_samples": len(samples), "source_manifest": str(source / "manifest.json"),
            "encoding_contract": encoding_contract,
            "outer_autocast": {"enabled": True, "dtype": "bfloat16"},
            "encoded_prompts": len(rows), "write_read_verified_prompts": len(rows),
            "context_tensor_bytes": sum(row["token_count"] for row in rows) * 4096 * 2,
            "context_file_bytes": sum(path.stat().st_size for path in (shard / "contexts").iterdir()),
            "write_read_comparison": "all original context, ids and mask bytes",
            "encoding_seconds": index + 0.5, "runtime": {"device": f"worker device {index}"},
        }
        (shard / "manifest.json").write_text(json.dumps(report))
        shards.append(shard)
    return source, shards, prompts, samples, tensors


def test_finalize_four_shards_preserves_original_rows_and_recomputes_payload(four_shards, tmp_path: Path):
    source, shards, prompts, samples, tensors = four_shards
    output = tmp_path / "complete"
    expected_file_bytes = sum(path.stat().st_size for shard in shards
                              for path in (shard / "contexts").iterdir())

    result = text_cache.finalize_text_cache(inputs=source, shards=list(reversed(shards)), output=output)

    assert result["complete_stage"] is True
    assert result["worker_shards"] == 4
    assert result["encoded_prompts"] == 5
    assert result["encoded_samples"] == 2
    assert result["context_tensor_bytes"] == 11 * 4096 * 2
    assert result["context_file_bytes"] == expected_file_bytes
    assert result["verified_n0"] is False
    assert json.loads((output / "manifest.json").read_text()) == result
    for name in ("prompts.jsonl", "sample-context-rows.jsonl"):
        assert (output / name).read_bytes() == (source / name).read_bytes()
    assert text_cache._read_rows(output / "prompts.jsonl") == prompts
    assert text_cache._read_rows(output / "sample-context-rows.jsonl") == samples
    assert {path.name for path in (output / "contexts").iterdir()} == {
        f"{row['row_id']}.safetensors" for row in prompts
    }
    for row_id, expected in tensors.items():
        restored = text_cache.read_text_context(output / "contexts" / f"{row_id}.safetensors")
        count = expected["context"].shape[1]
        assert torch.equal(restored["context"][:, :count].view(torch.uint8),
                           expected["context"].view(torch.uint8))
        assert torch.count_nonzero(restored["context"][:, count:].view(torch.uint8)) == 0
        for name in ("ids", "mask"):
            assert torch.equal(restored[name].view(torch.uint8), expected[name].view(torch.uint8))


@pytest.mark.parametrize("modified", [False, True])
def test_finalize_missing_prompt_shards_binds_to_original_stage(four_shards, tmp_path: Path, modified: bool):
    source, shards, prompts, *_ = four_shards
    target = tmp_path / "original-stage"
    target.mkdir()
    source_manifest = source / "manifest.json"
    contract = json.loads(source_manifest.read_text())
    target_manifest = target / "manifest.json"
    target_manifest.write_text(json.dumps(contract))
    original_rows = [*prompts[:1], {"row_id": 99, "prompt": "already cached", "token_count": 1}, *prompts[1:]]
    if modified:
        original_rows[0] = {**original_rows[0], "prompt": "different original prompt"}
    _jsonl(target / "prompts.jsonl", original_rows)
    source_manifest.write_text(json.dumps({
        "complete": False, "source_manifest": str(target_manifest),
        "encoding_contract": contract["encoding_contract"],
        "prompt_manifest": "prompts.jsonl", "sample_manifest": "groups.jsonl",
    }))
    _jsonl(source / "groups.jsonl", [{"context_rows": [row["row_id"] for row in prompts]}])
    for shard in shards:
        manifest = shard / "manifest.json"
        report = json.loads(manifest.read_text())
        manifest.write_text(json.dumps({**report, "selected_samples": 1}))
    output = tmp_path / "additional-contexts"

    if modified:
        with pytest.raises(ValueError, match="exact ordered subset"):
            text_cache.finalize_text_cache(inputs=source, shards=shards, output=output)
        assert not output.exists()
        return

    result = text_cache.finalize_text_cache(inputs=source, shards=shards, output=output)

    assert result["artifact_type"] == "mspc_training_text_context_cache_shard"
    assert result["source_manifest"] == str(target_manifest)
    assert result["complete_stage"] is False
    assert result["encoded_samples"] == result["selected_samples"] == 0
    assert result["encoded_prompts"] == result["write_read_verified_prompts"] == len(prompts)
    assert result["selected_prompt_rows"] == [row["row_id"] for row in prompts]
    assert result["shard"] == {"index": 0, "count": 1, "assignment": "original row_id modulo shard_count"}
    assert result["worker_shards"] == 4
    assert (output / "prompts.jsonl").read_bytes() == (source / "prompts.jsonl").read_bytes()
    assert not (output / "sample-context-rows.jsonl").exists()


@pytest.mark.parametrize("broken", ["missing_shard", "duplicate_path", "duplicate_index"])
def test_finalize_rejects_incomplete_worker_set_before_publication(four_shards, tmp_path: Path, broken: str):
    source, shards, *_ = four_shards
    if broken == "missing_shard":
        shards = shards[:-1]
    elif broken == "duplicate_path":
        shards = [shards[0], shards[0], shards[2], shards[3]]
    else:
        path = shards[-1] / "manifest.json"
        report = json.loads(path.read_text())
        report["shard"]["index"] = 0
        path.write_text(json.dumps(report))
    original_files = [path for shard in shards for path in (shard / "contexts").iterdir()]
    output = tmp_path / "complete"

    with pytest.raises(ValueError, match="shard"):
        text_cache.finalize_text_cache(inputs=source, shards=shards, output=output)

    assert not output.exists()
    assert all(path.exists() for path in original_files)


@pytest.mark.parametrize("broken", [
    "missing_source_prompt", "changed_source_prompt", "dangling_sample_reference",
    "missing_context_file", "wrong_file_token_count", "wrong_payload_receipt",
])
def test_finalize_rejects_mapping_or_file_mismatch_without_moving_valid_rows(four_shards, tmp_path: Path, broken: str):
    source, shards, prompts, samples, tensors = four_shards
    if broken == "missing_source_prompt":
        _jsonl(source / "prompts.jsonl", prompts[1:])
    elif broken == "changed_source_prompt":
        prompts[0]["prompt"] = "different original prompt"
        _jsonl(source / "prompts.jsonl", prompts)
    elif broken == "dangling_sample_reference":
        samples[0]["context_rows"].append(999)
        _jsonl(source / "sample-context-rows.jsonl", samples)
    elif broken == "missing_context_file":
        (shards[3] / "contexts/7.safetensors").unlink()
    elif broken == "wrong_file_token_count":
        values = dict(tensors[7])
        values["context"] = values["context"][:, :2].contiguous()
        values["mask"] = values["mask"].clone()
        values["mask"][:, 2:] = 0
        save_file(values, str(shards[3] / "contexts/7.safetensors"), metadata=text_cache.CONTEXT_STORAGE)
    else:
        path = shards[3] / "manifest.json"
        report = json.loads(path.read_text())
        report["context_tensor_bytes"] += 8192
        path.write_text(json.dumps(report))
    original_files = [path for shard in shards for path in (shard / "contexts").iterdir()]
    output = tmp_path / "complete"

    with pytest.raises(ValueError):
        text_cache.finalize_text_cache(inputs=source, shards=shards, output=output)

    assert not output.exists()
    assert all(path.exists() for path in original_files)


@pytest.fixture
def reusable_finalized_cache(four_shards, tmp_path: Path):
    source, shards, prompts, *_ = four_shards
    t5 = tmp_path / "shared-t5.pth"
    t5.write_bytes(b"shared test encoder")
    target_t5 = tmp_path / "wan21-t5.pth"
    target_t5.hardlink_to(t5)
    tokenizer = tmp_path / "shared-tokenizer"
    tokenizer.mkdir()
    original = json.loads((source / "manifest.json").read_text())
    original["encoding_contract"].update(t5_path=str(t5), tokenizer_path=str(tokenizer))
    (source / "manifest.json").write_text(json.dumps(original))
    for shard in shards:
        report = json.loads((shard / "manifest.json").read_text())
        report["encoding_contract"] = original["encoding_contract"]
        (shard / "manifest.json").write_text(json.dumps(report))
    canonical = tmp_path / "wan22-final-canonical"
    text_cache.finalize_text_cache(inputs=source, shards=shards, output=canonical)

    target = tmp_path / "wan21-inputs"
    target.mkdir()
    by_id = {row["row_id"]: row for row in prompts}
    mapping = {0: 7, 1: 12, 2: 10}
    target_prompts = [{**by_id[original_id], "row_id": local_id}
                      for local_id, original_id in mapping.items()]
    target_samples = [
        {"split": "train", "sample_index": 314, "context_rows": [1, 0, 1],
         "metadata": {"sample_index": 314, "actual_token_counts": [1, 4, 1],
                      "references": [{"shot": "Wan21 own reference"}]}},
        {"split": "validation", "sample_index": 271, "context_rows": [2],
         "metadata": {"sample_index": 271, "actual_token_counts": [3], "references": []}},
    ]
    contract = {**original["encoding_contract"], "t5_path": str(target_t5)}
    (target / "manifest.json").write_text(json.dumps({
        "artifact_type": text_cache.TEXT_CACHE_ARTIFACT_TYPE, "schema_version": 1,
        "complete": True, "sample_limit_per_split": None,
        "prompt_manifest": "prompts.jsonl", "sample_manifest": "sample-context-rows.jsonl",
        "encoding_contract": contract, "unique_prompts": 3, "total_prompt_rows": 4,
        "splits": {"train": {"expected_samples": 1}, "validation": {"expected_samples": 1}},
    }))
    for name, values in (("prompts.jsonl", target_prompts),
                         ("sample-context-rows.jsonl", target_samples)):
        # 紧凑JSON与CRLF必须原样复制，不能读取后按默认格式重写。
        (target / name).write_bytes("".join(
            json.dumps(row, separators=(",", ":")) + "\r\n" for row in values).encode())
    config = SimpleNamespace(resolved={"runtime": {"deepspeed": {
        "torch_autocast": {"enabled": True, "dtype": "bfloat16"}}}})
    return canonical, target, config, mapping


def test_reuse_finalized_cache_remaps_local_rows_and_preserves_original_bytes(reusable_finalized_cache, tmp_path: Path):
    canonical, target, config, mapping = reusable_finalized_cache
    output = tmp_path / "wan21-context-view"

    result = text_cache.reuse_text_cache(config, inputs=target, context_source=canonical, output=output)

    assert result["complete_stage"] is True
    assert result["encoded_prompts"] == result["write_read_verified_prompts"] == result["reused_prompts"] == 3
    assert result["encoded_samples"] == 2
    assert result["encoded_here_prompts"] == result["independent_live_checked_prompts"] == 0
    assert result["verified_n0"] is False
    assert result["source_manifest"] == str(target / "manifest.json")
    assert result["context_source_manifest"] == str(canonical / "manifest.json")
    assert result["encoding_contract"] == json.loads((target / "manifest.json").read_text())["encoding_contract"]
    assert json.loads((output / "manifest.json").read_text()) == result
    for name in ("prompts.jsonl", "sample-context-rows.jsonl"):
        assert (output / name).read_bytes() == (target / name).read_bytes()
    assert {path.name for path in (output / "contexts").iterdir()} == {f"{index}.safetensors" for index in mapping}
    assert text_cache._read_rows(output / "context-reuse.jsonl") == [
        {"row_id": local_id, "source_row_id": original_id,
         "source_manifest": str(canonical / "manifest.json")}
        for local_id, original_id in mapping.items()
    ]
    for local_id, original_id in mapping.items():
        link = output / "contexts" / f"{local_id}.safetensors"
        original = canonical / "contexts" / f"{original_id}.safetensors"
        assert link.is_symlink()
        assert link.readlink() == original
        assert link.resolve() == original
        actual = text_cache.read_text_context(link)
        expected = text_cache.read_text_context(original)
        for name in ("context", "ids", "mask"):
            assert torch.equal(actual[name].view(torch.uint8), expected[name].view(torch.uint8))


def test_reuse_missing_raw_prompt_fails_before_creating_output(reusable_finalized_cache, tmp_path: Path):
    canonical, target, config, _ = reusable_finalized_cache
    prompts = text_cache._read_rows(target / "prompts.jsonl")
    prompts[0]["prompt"] += " "  # 近似相同仍不是同一个raw字符串。
    _jsonl(target / "prompts.jsonl", prompts)
    output = tmp_path / "missing-prompt-view"

    with pytest.raises(ValueError, match="missing or mismatched actual prompt"):
        text_cache.reuse_text_cache(config, inputs=target, context_source=canonical, output=output)

    assert not output.exists()


def test_reuse_rejects_different_outer_autocast(reusable_finalized_cache, tmp_path: Path):
    canonical, target, config, _ = reusable_finalized_cache
    config.resolved["runtime"]["deepspeed"]["torch_autocast"]["enabled"] = False
    output = tmp_path / "different-autocast-view"

    with pytest.raises(ValueError, match="same actual outer autocast"):
        text_cache.reuse_text_cache(config, inputs=target, context_source=canonical, output=output)

    assert not output.exists()


def test_text_row_selection_keeps_input_order_and_original_ids():
    prompts = [{"row_id": index, "prompt": f"original {index}"} for index in (12, 1, 7, 5)]

    selected = text_cache._selected_prompt_rows(
        prompts, row_ids=[5, 12], shard_index=0, shard_count=1, sample_limit=None)

    assert selected == [prompts[0], prompts[3]]
    assert text_cache._selected_prompt_rows(
        prompts, row_ids=None, shard_index=1, shard_count=2, sample_limit=None) == prompts[1:]


@pytest.mark.parametrize("row_ids", [[], [12, 12], [True], [1.0], [99]])
def test_text_row_selection_rejects_empty_duplicate_noninteger_or_missing_ids(row_ids):
    prompts = [{"row_id": index} for index in (12, 1, 7, 5)]

    with pytest.raises(ValueError):
        text_cache._selected_prompt_rows(
            prompts, row_ids=row_ids, shard_index=0, shard_count=1, sample_limit=None)


@pytest.mark.parametrize("options", [
    {"shard_index": 1, "shard_count": 2, "sample_limit": None},
    {"shard_index": 0, "shard_count": 1, "sample_limit": 1},
])
def test_text_row_selection_rejects_mixed_sample_or_worker_selection(options):
    with pytest.raises(ValueError):
        text_cache._selected_prompt_rows([{"row_id": 12}], row_ids=[12], **options)


def test_text_row_selection_encode_calls_only_selected_original_prompts(four_shards, tmp_path: Path, monkeypatch):
    inputs, _, prompts, *_ = four_shards
    by_prompt = {row["prompt"]: row for row in prompts}
    encoded, tokenized = [], []

    def tokenizer(prompt, **options):
        tokenized.append((prompt, options))
        original = _original(by_prompt[prompt]["token_count"])
        return original["ids"], original["mask"]

    def encode_prompt(pipe, prompt):
        encoded.append(prompt)
        return _original(by_prompt[prompt]["token_count"])["context"]

    pipe = SimpleNamespace(tokenizer=tokenizer,
        units=[SimpleNamespace(encode_prompt=encode_prompt)], text_encoder=torch.nn.Identity())
    monkeypatch.setattr(text_cache, "_text_runtime", lambda *_: (
        pipe, {"enabled": True, "dtype": "bfloat16"}))
    monkeypatch.setattr(torch, "autocast", lambda *_args, **_kwargs: nullcontext())
    for name in ("reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "fake CPU encoder")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (0, 0))
    output = tmp_path / "selected-contexts"

    result = text_cache.encode_text_cache(object(), inputs=inputs, output=output,
        sample_limit=None, shard_index=0, shard_count=1, row_ids=[16, 5])

    expected = [row for row in prompts if row["row_id"] in (16, 5)]
    assert encoded == [row["prompt"] for row in expected]
    assert tokenized == [(row["prompt"], {"return_mask": True, "add_special_tokens": True})
                         for row in expected]
    assert result["artifact_type"] == "mspc_training_text_context_cache_shard"
    assert result["complete_stage"] is False
    assert result["encoded_samples"] == result["selected_samples"] == 0
    assert result["encoded_prompts"] == result["write_read_verified_prompts"] == 2
    assert result["selected_prompt_rows"] == [5, 16]
    assert result["context_tensor_bytes"] == 3 * 4096 * 2
    assert text_cache._read_rows(output / "prompts.jsonl") == expected
    assert not (output / "sample-context-rows.jsonl").exists()
    assert {path.name for path in (output / "contexts").iterdir()} == {"5.safetensors", "16.safetensors"}


@pytest.fixture
def reusable_additional_contexts(reusable_finalized_cache, tmp_path: Path):
    canonical, target, config, mapping = reusable_finalized_cache
    prompts = text_cache._read_rows(target / "prompts.jsonl")
    added = {"row_id": 91, "prompt": "Wan21 new actual raw prompt", "token_count": 2}
    prompts.insert(1, added)
    _jsonl(target / "prompts.jsonl", prompts)
    samples = text_cache._read_rows(target / "sample-context-rows.jsonl")
    samples[0]["context_rows"].insert(1, 91)
    samples[0]["metadata"]["actual_token_counts"].insert(1, 2)
    _jsonl(target / "sample-context-rows.jsonl", samples)
    source = json.loads((target / "manifest.json").read_text())
    source.update(unique_prompts=4, total_prompt_rows=5)
    (target / "manifest.json").write_text(json.dumps(source))

    additional = tmp_path / "wan21-additional-canonical"
    (additional / "contexts").mkdir(parents=True)
    mask = torch.zeros((1, 512), dtype=torch.int64)
    mask[:, :2] = 1
    tensors = {
        "context": torch.full((1, 2, 4096), 91, dtype=torch.bfloat16),
        "ids": torch.arange(512, dtype=torch.int64).unsqueeze(0) + 91,
        "mask": mask,
    }
    context_file = additional / "contexts/91.safetensors"
    save_file(tensors, str(context_file), metadata=text_cache.CONTEXT_STORAGE)
    _jsonl(additional / "prompts.jsonl", [added])
    report = {
        "artifact_type": "mspc_training_text_context_cache_shard", "schema_version": 2,
        "context_storage": text_cache.CONTEXT_STORAGE,
        "storage_contract": text_cache.CONTEXT_STORAGE_CONTRACT,
        "source_manifest": str(target / "manifest.json"),
        "encoding_contract": source["encoding_contract"],
        "outer_autocast": {"enabled": True, "dtype": "bfloat16"},
        "shard": {"index": 0, "count": 1, "assignment": "original row_id modulo shard_count"},
        "complete_stage": False, "encoded_prompts": 1,
        "encoded_samples": 0, "selected_samples": 0, "sample_limit_per_split": None,
        "selected_prompt_rows": [91], "write_read_verified_prompts": 1,
        "context_tensor_bytes": 2 * 4096 * 2,
        "context_file_bytes": context_file.stat().st_size,
    }
    (additional / "manifest.json").write_text(json.dumps(report))
    return canonical, target, config, mapping, additional, prompts


def test_reuse_additional_links_exact_new_rows_and_counts_sources(reusable_additional_contexts, tmp_path: Path):
    canonical, target, config, mapping, additional, prompts = reusable_additional_contexts
    output = tmp_path / "wan21-merged-context-view"

    result = text_cache.reuse_text_cache(config, inputs=target, context_source=canonical,
        additional_contexts=additional, output=output)

    assert result["complete_stage"] is True
    assert result["encoded_prompts"] == result["write_read_verified_prompts"] == 4
    assert result["reused_prompts"] == 3
    assert result["newly_encoded_source_prompts"] == 1
    assert result["encoded_here_prompts"] == result["independent_live_checked_prompts"] == 0
    assert result["verified_n0"] is False
    assert result["context_tensor_bytes"] == 10 * 4096 * 2
    assert json.loads((output / "manifest.json").read_text()) == result
    for name in ("prompts.jsonl", "sample-context-rows.jsonl"):
        assert (output / name).read_bytes() == (target / name).read_bytes()
    expected_records = []
    for row in prompts:
        local_id = row["row_id"]
        origin, source_id = (additional, 91) if local_id == 91 else (canonical, mapping[local_id])
        original = origin / "contexts" / f"{source_id}.safetensors"
        link = output / "contexts" / f"{local_id}.safetensors"
        assert link.is_symlink() and link.readlink() == original
        for name, tensor in text_cache.read_text_context(original).items():
            assert torch.equal(text_cache.read_text_context(link)[name].view(torch.uint8), tensor.view(torch.uint8))
        expected_records.append({"row_id": local_id, "source_row_id": source_id,
                                 "source_manifest": str(origin / "manifest.json")})
    assert text_cache._read_rows(output / "context-reuse.jsonl") == expected_records
    assert result["context_file_bytes"] == sum(path.stat().st_size for path in (output / "contexts").iterdir())


def test_parallel_reuse_preserves_row_order_from_multiple_sources(reusable_additional_contexts, tmp_path):
    canonical, target, config, _, additional, prompts = reusable_additional_contexts
    original = json.loads((canonical / "manifest.json").read_text())
    original_rows = text_cache._read_rows(canonical / "prompts.jsonl")
    primary, secondary = tmp_path / "primary-source", tmp_path / "secondary-source"
    for root, ids in ((primary, {7, 10}), (secondary, {7, 12})):
        (root / "contexts").mkdir(parents=True)
        rows = [row for row in original_rows if row["row_id"] in ids]
        _jsonl(root / "prompts.jsonl", rows)
        file_bytes = 0
        for row in rows:
            filename = f"{row['row_id']}.safetensors"
            path = root / "contexts" / filename
            path.hardlink_to(canonical / "contexts" / filename)
            file_bytes += path.stat().st_size
        (root / "manifest.json").write_text(json.dumps({
            **original, "encoded_prompts": len(rows), "write_read_verified_prompts": len(rows),
            "context_file_bytes": file_bytes,
            "context_tensor_bytes": sum(row["token_count"] for row in rows) * 4096 * 2,
        }))

    output = tmp_path / "shared-context"
    result = text_cache.reuse_text_cache(config, inputs=target, context_source=primary,
        context_sources=(secondary,), additional_contexts=additional, output=output, workers=8)

    primary_manifest = str(primary / "manifest.json")
    secondary_manifest = str(secondary / "manifest.json")
    assert result["context_source_manifests"] == [primary_manifest, secondary_manifest]
    assert result["reused_prompts_by_source"] == {primary_manifest: 2, secondary_manifest: 1}
    assert result["newly_encoded_source_prompts"] == 1
    paths = {
        0: primary / "contexts/7.safetensors", 1: secondary / "contexts/12.safetensors",
        2: primary / "contexts/10.safetensors", 91: additional / "contexts/91.safetensors",
    }
    assert [row["row_id"] for row in text_cache._read_rows(output / "context-reuse.jsonl")] == [
        row["row_id"] for row in prompts]
    for row_id, source in paths.items():
        link = output / "contexts" / f"{row_id}.safetensors"
        assert link.readlink() == source
        assert link.read_bytes() == source.read_bytes()
    assert result["context_file_bytes"] == sum(path.stat().st_size for path in paths.values())


@pytest.mark.parametrize("broken", [
    "missing_source", "wrong_source", "wrong_contract", "missing_row", "wrong_row",
    "wrong_selection", "missing_file", "extra_file", "wrong_payload", "wrong_file_bytes", "unverified_write",
])
def test_reuse_additional_rejects_incomplete_or_mismatched_partial_before_publication(
        reusable_additional_contexts, tmp_path: Path, broken: str):
    canonical, target, config, _, additional, _ = reusable_additional_contexts
    path = additional / "manifest.json"
    report = json.loads(path.read_text())
    if broken == "missing_source":
        path.unlink()
    elif broken == "wrong_source":
        report["source_manifest"] = str(canonical / "manifest.json")
    elif broken == "wrong_contract":
        report["encoding_contract"]["max_length"] = 511
    elif broken == "missing_row":
        _jsonl(additional / "prompts.jsonl", [])
    elif broken == "wrong_row":
        rows = text_cache._read_rows(additional / "prompts.jsonl")
        rows[0]["prompt"] += " "
        _jsonl(additional / "prompts.jsonl", rows)
    elif broken == "wrong_selection":
        report["selected_prompt_rows"] = [0]
    elif broken == "missing_file":
        (additional / "contexts/91.safetensors").unlink()
    elif broken == "extra_file":
        original = canonical / "contexts/7.safetensors"
        (additional / "contexts/0.safetensors").write_bytes(original.read_bytes())
    elif broken == "wrong_payload":
        report["context_tensor_bytes"] += 8192
    elif broken == "wrong_file_bytes":
        report["context_file_bytes"] += 1
    elif broken == "unverified_write":
        report["write_read_verified_prompts"] = 0
    if broken != "missing_source":
        path.write_text(json.dumps(report))
    output = tmp_path / "invalid-additional-view"

    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        text_cache.reuse_text_cache(config, inputs=target, context_source=canonical,
            additional_contexts=additional, output=output)

    assert not (output / "manifest.json").exists()
