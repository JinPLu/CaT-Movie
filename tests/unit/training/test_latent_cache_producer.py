from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.support.cache_provenance import provenance, contract
import torch
from safetensors.torch import save_file

import mspc.training.latent_cache as latent_cache


def _record(index: int, case_id: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "index": index,
        "provenance": provenance(),
        "cache_file": f"latents/{case_id}.safetensors",
        "input_latents_shape": [1, 8, 4, 4, 4],
        "context_shape": [2, 7, 16],
        "shot_groups": [[0, 5], [5, 10]],
        "latent_shot_groups": [[0, 2], [2, 4]],
        "height": 32,
        "width": 32,
        "num_frames": 10,
        "shot_count": 2,
    }


def _write_tensor(root: Path, record: dict[str, object]) -> None:
    path = root / str(record["cache_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "input_latents": torch.zeros(*record["input_latents_shape"], dtype=torch.bfloat16),
            "context": torch.zeros(*record["context_shape"], dtype=torch.bfloat16),
        },
        str(path),
        metadata={
            "provenance": json.dumps(record["provenance"]),
            "case_id": str(record["case_id"]),
            "input_latents_shape": json.dumps(record["input_latents_shape"]),
            "context_shape": json.dumps(record["context_shape"]),
            "shot_groups": json.dumps(record["shot_groups"]),
            "latent_shot_groups": json.dumps(record["latent_shot_groups"]),
            "height": str(record["height"]),
            "width": str(record["width"]),
            "num_frames": str(record["num_frames"]),
        },
    )


class _Model:
    def __init__(self) -> None:
        vae = torch.nn.Module()
        vae.register_parameter("weight", torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16), requires_grad=False))
        self.pipe = SimpleNamespace(vae=vae.eval())

    def eval(self) -> "_Model":
        return self

    def encode_for_cache(self, sample: dict[str, object]):
        assert sample["prompt"] == ["first", "second"]
        return (
            torch.arange(1 * 8 * 4 * 4 * 4, dtype=torch.float32).reshape(1, 8, 4, 4, 4),
            torch.arange(2 * 7 * 16, dtype=torch.float32).reshape(2, 7, 16),
            {
                "shot_groups": [[0, 5], [5, 10]],
                "latent_shot_groups": [[0, 2], [2, 4]],
                "height": 32,
                "width": 32,
                "num_frames": 10,
            },
        )

    def encode_batch_for_cache(self, samples: list[dict[str, object]]):
        return [self.encode_for_cache(sample) for sample in samples]


class _Dataset:
    def __init__(self) -> None:
        self.data = [{"_case_id": "case_a"}, {"_case_id": "case_b"}]

    def describe_media(self, index):
        return {"provenance": provenance()["media"], "row": {"_prompts": ["first", "second"]}}

    def _load_sample(self, index: int) -> dict[str, object]:
        return {
            "case_id": self.data[index]["_case_id"],
            "prompt": ["first", "second"],
            "media_provenance": provenance()["media"],
            "shot_groups": [[0, 5], [5, 10]],
            "latent_shot_groups": [[0, 2], [2, 4]],
            "video": torch.zeros(3, 10, 32, 32, dtype=torch.uint8),
        }


def _source_config(layout: str, *, encode_batch_size: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        resolved={
            "dataset": {"cache_layout": layout},
            "model": {"paths": {}},
            "pipeline_inputs": {"tiled": False, "tile_size": [30,52], "tile_stride": [15,26]},
            "precompute": {
                "provenance_path": "test.json",
                "encode_batch_size": encode_batch_size,
                "dataset_num_workers": 0,
                "prefetch_factor": 1,
                "video_decode_num_threads": 1,
            },
        }
    )


def _precompute_kwargs(tmp_path: Path, output_dir: Path, **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "config_path": tmp_path / "config.json",
        "output_dir": output_dir,
        "shard_index": 0,
        "num_shards": 1,
    }
    kwargs.update(overrides)
    return kwargs


def test_encode_preserves_per_shot_vae_and_t5_layout(tmp_path: Path) -> None:
    encoded = latent_cache._encode_loaded_batch(
        _Model(),
        [{"sample": _Dataset()._load_sample(0), "index": 0, "case_id": "case_a", "provenance": provenance()}],
        latents_dir=tmp_path / "latents",
        cache_layout="latents_and_context",
    )[0]
    assert encoded.tensors["input_latents"].dtype == torch.bfloat16
    assert encoded.tensors["context"].dtype == torch.bfloat16
    assert encoded.record == _record(0, "case_a")
    assert encoded.record["context_shape"][0] == len(_Dataset()._load_sample(0)["prompt"])
    assert not any("hash" in key or "identity" in key for key in encoded.metadata)


@pytest.mark.parametrize("wrong_source_length", [False, True])
def test_real_scene_single_shot_native_metadata(tmp_path: Path, wrong_source_length: bool) -> None:
    rows = json.loads((Path(__file__).parents[2] / "fixtures/scene_cache/real_rows.json").read_text())
    row = next(r for r in rows if "061c63fa-3148-8d60-9442-d06351a44c3d" in r["case_id"])
    groups = json.loads(row["shot_groups"])
    frames = groups[0][1]
    latent_frames = (frames - 1) // 4 + 1
    source = provenance()
    source["media"]["sampling"].update(shot_groups=groups, latent_shot_groups=[[0, latent_frames]])
    if wrong_source_length:
        source["media"]["sampling"]["shot_groups"][0][1] -= 4
    kwargs = dict(input_latents=torch.zeros(1, 48, latent_frames, 30, 52), context=None,
        embed_meta={"shot_groups": None, "latent_shot_groups": None,
            "height": 480, "width": 832, "num_frames": frames},
        index=1, case_id=row["case_id"], cache_path=tmp_path / "single.safetensors",
        cache_layout="latents_only", dtoh_seconds=0.0, provenance=source)
    if wrong_source_length:
        with pytest.raises(ValueError, match="encoded shot layout differs"):
            latent_cache._finalize_encoded_sample(**kwargs)
    else:
        encoded = latent_cache._finalize_encoded_sample(**kwargs)
        saved = latent_cache._save_encoded_sample(encoded, cache_layout="latents_only")
        assert saved.record["shot_count"] == 1
        assert saved.record["shot_groups"] is None
        assert saved.record["latent_shot_groups"] is None


def _real_scene_encoding_kwargs(tmp_path: Path) -> dict:
    rows = json.loads((Path(__file__).parents[2] / "fixtures/scene_cache/real_rows.json").read_text())
    row = next(r for r in rows if "061c63fa-3148-8d60-9442-d06351a44c3d" in r["case_id"])
    groups = json.loads(row["shot_groups"])
    frames = groups[0][1]
    latent_frames = (frames - 1) // 4 + 1
    evidence = provenance()
    evidence["media"]["sampling"].update(shot_groups=groups, latent_shot_groups=[[0, latent_frames]])
    return dict(input_latents=torch.zeros(1, 16, latent_frames, 60, 104), context=None,
        embed_meta={"shot_groups": None, "latent_shot_groups": None,
                    "height": 480, "width": 832, "num_frames": frames},
        index=1, case_id=row["case_id"],
        cache_path=tmp_path / "latents" / f"{row['case_id']}.safetensors",
        cache_layout="latents_only", dtoh_seconds=0.0, provenance=evidence)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 3.4e38])
def test_real_scene_rejects_nonfinite_tensor_after_actual_cache_dtype_conversion(tmp_path: Path, value: float) -> None:
    kwargs = _real_scene_encoding_kwargs(tmp_path)
    kwargs["input_latents"].view(-1)[0] = value
    with pytest.raises(ValueError, match="non-finite input_latents.*case_id=.*index=1"):
        latent_cache._finalize_encoded_sample(**kwargs)
    assert not kwargs["cache_path"].exists()


def test_finalize_rejects_nonfinite_baked_context(tmp_path: Path) -> None:
    model = _Model()
    latents, context, metadata = model.encode_for_cache(_Dataset()._load_sample(0))
    context[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite context"):
        latent_cache._finalize_encoded_sample(input_latents=latents, context=context,
            embed_meta=metadata, index=0, case_id="case_a", cache_path=tmp_path / "latents/case_a.safetensors",
            cache_layout="latents_and_context", dtoh_seconds=0, provenance=provenance())


@pytest.mark.parametrize("temporary_bytes", [None, b"", b"\x08\x00\x00"])
def test_real_scene_unpublished_atomic_partial_is_quarantined_before_explicit_continuation(tmp_path: Path, temporary_bytes) -> None:
    kwargs = _real_scene_encoding_kwargs(tmp_path)
    encoded = latent_cache._finalize_encoded_sample(**kwargs)
    latent_cache._save_encoded_sample(encoded, cache_layout="latents_only")
    path = kwargs["cache_path"]
    temporary = path.with_name(f".{path.name}.12345.67890.tmp")
    path.rename(temporary)
    if temporary_bytes is not None:
        temporary.write_bytes(temporary_bytes)
    original = temporary.read_bytes()
    untouched = path.with_name(f".{path.name}.foreign.tmp")
    untouched.write_bytes(b"not this producer's atomic name")
    status, record = latent_cache._existing_cache_record(path, cache_root=tmp_path,
        case_id=kwargs["case_id"], index=kwargs["index"], cache_layout="latents_only",
        expected_provenance=kwargs["provenance"])
    assert status == "missing" and record is None
    assert not temporary.exists() and untouched.read_bytes() == b"not this producer's atomic name"
    events = list((tmp_path / "quarantine").glob("*/record.json"))
    assert len(events) == 1
    event = json.loads(events[0].read_text())
    assert event["action"] == "continue_after_quarantine_of_unpublished_write"
    assert event["files"][0]["original_path"] == str(temporary)
    assert Path(event["files"][0]["quarantine_path"]).read_bytes() == original
    latent_cache._save_encoded_sample(encoded, cache_layout="latents_only")
    assert path.is_file()


def test_real_scene_readable_temporary_source_conflict_is_retained_and_stops(tmp_path: Path) -> None:
    kwargs = _real_scene_encoding_kwargs(tmp_path)
    encoded = latent_cache._finalize_encoded_sample(**kwargs)
    latent_cache._save_encoded_sample(encoded, cache_layout="latents_only")
    path = kwargs["cache_path"]
    temporary = path.with_name(f".{path.name}.12345.67890.tmp")
    path.rename(temporary)
    original = temporary.read_bytes()
    expected = json.loads(json.dumps(kwargs["provenance"]))
    expected["models"]["vae"]["revision"] = "different-vae-source"
    with pytest.raises(ValueError, match="unpublished cache conflicts"):
        latent_cache._existing_cache_record(path, cache_root=tmp_path,
            case_id=kwargs["case_id"], index=kwargs["index"], cache_layout="latents_only",
            expected_provenance=expected)
    event = json.loads(next((tmp_path / "quarantine").glob("*/record.json")).read_text())
    assert event["action"] == "stop_after_quarantine"
    assert Path(event["files"][0]["quarantine_path"]).read_bytes() == original
    assert not path.exists()


@pytest.mark.parametrize("conflict", ["provenance", "shape"])
def test_readable_real_scene_final_conflict_stays_in_place(tmp_path: Path, conflict: str) -> None:
    kwargs = _real_scene_encoding_kwargs(tmp_path)
    encoded = latent_cache._finalize_encoded_sample(**kwargs)
    metadata = dict(encoded.metadata)
    if conflict == "provenance":
        stored = json.loads(metadata["provenance"])
        stored["models"]["vae"]["revision"] = "different-vae-source"
        metadata["provenance"] = json.dumps(stored)
    else:
        metadata["input_latents_shape"] = json.dumps([1, 16, encoded.record["input_latents_shape"][2], 30, 52])
    path = kwargs["cache_path"]
    path.parent.mkdir(parents=True)
    save_file(encoded.tensors, str(path), metadata=metadata)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="provenance differs" if conflict == "provenance" else "shape mismatch"):
        latent_cache._existing_cache_record(path, cache_root=tmp_path,
            case_id=kwargs["case_id"], index=kwargs["index"], cache_layout="latents_only",
            expected_provenance=kwargs["provenance"])
    assert path.read_bytes() == original
    assert not (tmp_path / "quarantine").exists()


def test_parameter_summary_reports_real_dtype_device_and_frozen_bytes() -> None:
    vae = torch.nn.Module()
    vae.register_parameter("bf16_weight", torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16), requires_grad=False))
    vae.register_parameter("fp32_weight", torch.nn.Parameter(torch.zeros(3, dtype=torch.float32), requires_grad=False))
    vae.eval()
    summary = latent_cache._vae_parameter_summary(vae)
    assert summary["tensor_count"] == 2 and summary["numel"] == 5 and summary["bytes"] == 16
    assert summary["by_device"] == {"cpu": {"tensor_count": 2, "numel": 5, "bytes": 16}}
    assert summary["by_dtype"] == {
        "torch.bfloat16": {"tensor_count": 1, "numel": 2, "bytes": 4},
        "torch.float32": {"tensor_count": 1, "numel": 3, "bytes": 12}}
    assert summary["all_frozen"] is True and summary["training"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("shot_groups", [[0, 4], [4, 10]], r"not 4n\+1"),
        ("latent_shot_groups", [[0, 2], [3, 4]], "not contiguous"),
        ("context_shape", [3, 7, 16], "context must have layout"),
    ],
)
def test_record_layout_keeps_only_semantic_invariants(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    record = _record(0, "case_a")
    record[field] = value
    with pytest.raises(ValueError, match=message):
        latent_cache._validate_record_layout(record, tmp_path / "manifest.json")


def test_merge_requires_complete_shards_and_writes_minimal_manifest(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    for shard_index, (index, case_id) in enumerate(((0, "case_a"), (1, "case_b"))):
        record = _record(index, case_id)
        record.update({"status": "written", "prompt_sha256": "legacy-extra"})
        _write_tensor(tmp_path, record)
        (shard_dir / f"shard-{shard_index:05d}.json").write_text(
            json.dumps(
                {
                    "schema": latent_cache.SHARD_MANIFEST_SCHEMA,
                    "shard_index": shard_index,
                    "num_shards": 2,
                    "total_rows": 2,
                    "records": [record],
                }
            ),
            encoding="utf-8",
        )
    result = latent_cache.merge_latent_cache_manifest(output_dir=tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert result["counts"] == {"written": 2}
    assert set(manifest) == {"schema", "shards", "total_rows", "records"}
    assert manifest["records"] == [_record(0, "case_a"), _record(1, "case_b")]


def test_merge_rejects_missing_shard(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    record = _record(0, "case_a")
    _write_tensor(tmp_path, record)
    (shard_dir / "shard-00000.json").write_text(
        json.dumps(
            {
                "schema": latent_cache.SHARD_MANIFEST_SCHEMA,
                "shard_index": 0,
                "num_shards": 2,
                "total_rows": 2,
                "records": [record],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="shard set is incomplete"):
        latent_cache.merge_latent_cache_manifest(output_dir=tmp_path)


def test_precompute_writes_once_and_skips_complete_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "cache"
    monkeypatch.setattr(
        latent_cache,
        "load_experiment_config",
        lambda _: _source_config("latents_and_context"),
    )
    monkeypatch.setattr(latent_cache, "_build_source_dataset", lambda *args, **kwargs: _Dataset())
    monkeypatch.setattr(latent_cache.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(latent_cache.wan22_backend, "configure_args", lambda *args: object())
    monkeypatch.setattr(
        latent_cache.wan22_backend,
        "build_cache_encoder",
        lambda *args, **kwargs: _Model(),
    )
    monkeypatch.setattr(latent_cache, "build_common_training_args", lambda _: SimpleNamespace())
    kwargs = _precompute_kwargs(tmp_path, output_dir)
    result = latent_cache.precompute_latent_cache(**kwargs)
    assert result["written"] == 2
    shard = json.loads((output_dir / "shards/shard-00000.json").read_text(encoding="utf-8"))
    assert shard["records"] == [_record(0, "case_a"), _record(1, "case_b")]
    assert not (output_dir / "provenance.json").exists()
    again = latent_cache.precompute_latent_cache(**kwargs)
    assert again["written"] == 0
    assert again["skipped_existing"] == 2
    assert again["timing"]["model_load_seconds"] is None
    assert again["vae_parameters"] is None and again["cuda_peak_memory"] is None


def test_precompute_skips_valid_existing_latents_only_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "cache"
    calls: list[str] = []

    class _CountingModel(_LatentsOnlyModel):
        def encode_for_cache(self, sample: dict[str, object]):
            calls.append(str(sample["case_id"]))
            return super().encode_for_cache(sample)

    existing = _latents_only_record(0, "case_a")
    _write_latents_only_tensor(output_dir, existing)
    monkeypatch.setattr(
        latent_cache,
        "load_experiment_config",
        lambda _: _source_config("latents_only"),
    )
    monkeypatch.setattr(latent_cache, "_build_source_dataset", lambda *args, **kwargs: _Dataset())
    monkeypatch.setattr(latent_cache.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(latent_cache.wan22_backend, "configure_args", lambda *args: object())
    monkeypatch.setattr(
        latent_cache.wan22_backend,
        "build_cache_encoder",
        lambda *args, **kwargs: _CountingModel(),
    )
    monkeypatch.setattr(latent_cache, "build_common_training_args", lambda _: SimpleNamespace())
    result = latent_cache.precompute_latent_cache(**_precompute_kwargs(tmp_path, output_dir))
    assert result["written"] == 1
    assert result["skipped_existing"] == 1
    assert calls == ["case_b"]
    shard = json.loads((output_dir / "shards/shard-00000.json").read_text(encoding="utf-8"))
    assert [record["case_id"] for record in shard["records"]] == ["case_a", "case_b"]


def test_all_valid_files_without_shard_publish_summary_needs_no_cuda(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "cache"
    for index, name in enumerate(("case_a", "case_b")):
        _write_latents_only_tensor(output_dir, _latents_only_record(index, name))
    monkeypatch.setattr(latent_cache, "load_experiment_config", lambda _: _source_config("latents_only"))
    monkeypatch.setattr(latent_cache, "_build_source_dataset", lambda *a, **kw: _Dataset())
    def unexpected_probe():
        pytest.fail("all valid-cache continuation should not need CUDA")
    monkeypatch.setattr(torch.cuda, "is_available", unexpected_probe)
    result = latent_cache.precompute_latent_cache(**_precompute_kwargs(tmp_path, output_dir))
    assert result["written"] == 0 and result["skipped_existing"] == 2
    assert result["timing"]["model_load_seconds"] is None
    assert result["vae_parameters"] is None and result["cuda_peak_memory"] is None
    assert (output_dir / "shards/shard-00000.json").is_file()


def test_precompute_fails_loud_on_corrupt_existing_latent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "cache"
    path = output_dir / "latents" / "case_a.safetensors"
    path.parent.mkdir(parents=True)
    save_file({"wrong": torch.zeros(1)}, str(path), metadata={"case_id": "case_a"})
    original = path.read_bytes()
    monkeypatch.setattr(
        latent_cache,
        "load_experiment_config",
        lambda _: _source_config("latents_only"),
    )
    monkeypatch.setattr(latent_cache, "_build_source_dataset", lambda *args, **kwargs: _Dataset())
    with pytest.raises(ValueError, match="tensor keys"):
        latent_cache.precompute_latent_cache(**_precompute_kwargs(tmp_path, output_dir))
    assert path.read_bytes() == original
    assert not (output_dir / "quarantine").exists()


def test_merge_refuses_automatic_remerge(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already merged"):
        latent_cache.merge_latent_cache_manifest(output_dir=tmp_path)


class _LatentsOnlyModel(_Model):
    def eval(self) -> "_LatentsOnlyModel":
        return self

    def encode_for_cache(self, sample: dict[str, object]):
        return (
            torch.arange(1 * 8 * 4 * 4 * 4, dtype=torch.float32).reshape(1, 8, 4, 4, 4),
            None,
            {
                "shot_groups": [[0, 5], [5, 10]],
                "latent_shot_groups": [[0, 2], [2, 4]],
                "height": 32,
                "width": 32,
                "num_frames": 10,
            },
        )

    def encode_batch_for_cache(self, samples: list[dict[str, object]]):
        return [self.encode_for_cache(sample) for sample in samples]


def _latents_only_record(index: int, case_id: str) -> dict[str, object]:
    evidence = provenance()
    evidence["context"] = None
    evidence["models"] = {"vae": evidence["models"]["vae"]}
    return {
        "case_id": case_id,
        "index": index,
        "provenance": evidence,
        "cache_file": f"latents/{case_id}.safetensors",
        "input_latents_shape": [1, 8, 4, 4, 4],
        "shot_groups": [[0, 5], [5, 10]],
        "latent_shot_groups": [[0, 2], [2, 4]],
        "height": 32,
        "width": 32,
        "num_frames": 10,
        "shot_count": 2,
    }


def _write_latents_only_tensor(root: Path, record: dict[str, object]) -> None:
    path = root / str(record["cache_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "input_latents": torch.zeros(*record["input_latents_shape"], dtype=torch.bfloat16),
        },
        str(path),
        metadata={
            "provenance": json.dumps(record["provenance"]),
            "case_id": str(record["case_id"]),
            "input_latents_shape": json.dumps(record["input_latents_shape"]),
            "shot_groups": json.dumps(record["shot_groups"]),
            "latent_shot_groups": json.dumps(record["latent_shot_groups"]),
            "height": str(record["height"]),
            "width": str(record["width"]),
            "num_frames": str(record["num_frames"]),
            "cache_layout": "latents_only",
        },
    )


def test_latents_only_encode_omits_context(tmp_path: Path) -> None:
    encoded = latent_cache._encode_loaded_batch(
        _LatentsOnlyModel(),
        [{"sample": _Dataset()._load_sample(0), "index": 0, "case_id": "case_a", "provenance": _latents_only_record(0, "case_a")["provenance"]}],
        latents_dir=tmp_path / "latents",
        cache_layout="latents_only",
    )[0]
    assert set(encoded.tensors) == {"input_latents"}
    assert "context_shape" not in encoded.record
    assert encoded.record == _latents_only_record(0, "case_a")
    assert encoded.metadata["cache_layout"] == "latents_only"


def test_latents_only_encode_rejects_baked_context(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="must not produce context"):
        latent_cache._encode_loaded_batch(
            _Model(),
            [{"sample": _Dataset()._load_sample(0), "index": 0, "case_id": "case_a", "provenance": provenance()}],
            latents_dir=tmp_path / "latents",
            cache_layout="latents_only",
        )


def test_merge_latents_only_v5_manifest(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    for shard_index, (index, case_id) in enumerate(((0, "case_a"), (1, "case_b"))):
        record = _latents_only_record(index, case_id)
        _write_latents_only_tensor(tmp_path, record)
        (shard_dir / f"shard-{shard_index:05d}.json").write_text(
            json.dumps(
                {
                    "schema": latent_cache.SHARD_MANIFEST_SCHEMA_V5,
                    "cache_layout": "latents_only",
                    "shard_index": shard_index,
                    "num_shards": 2,
                    "total_rows": 2,
                    "records": [record],
                }
            ),
            encoding="utf-8",
        )
    result = latent_cache.merge_latent_cache_manifest(output_dir=tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert result["counts"] == {"written": 2}
    assert manifest["schema"] == latent_cache.FINAL_MANIFEST_SCHEMA_V5
    assert manifest["cache_layout"] == "latents_only"
    assert manifest["records"] == [
        _latents_only_record(0, "case_a"),
        _latents_only_record(1, "case_b"),
    ]


def test_unknown_manifest_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="manifest schema is unsupported"):
        latent_cache.cache_layout_from_manifest_schema("mspc_latent_cache_manifest_v0")


def test_merge_rejects_mixed_v4_v5_shards(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    v4 = _record(0, "case_a")
    v5 = _latents_only_record(1, "case_b")
    _write_tensor(tmp_path, v4)
    _write_latents_only_tensor(tmp_path, v5)
    (shard_dir / "shard-00000.json").write_text(
        json.dumps(
            {
                "schema": latent_cache.SHARD_MANIFEST_SCHEMA_V4,
                "shard_index": 0,
                "num_shards": 2,
                "total_rows": 2,
                "records": [v4],
            }
        ),
        encoding="utf-8",
    )
    (shard_dir / "shard-00001.json").write_text(
        json.dumps(
            {
                "schema": latent_cache.SHARD_MANIFEST_SCHEMA_V5,
                "cache_layout": "latents_only",
                "shard_index": 1,
                "num_shards": 2,
                "total_rows": 2,
                "records": [v5],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schemas disagree"):
        latent_cache.merge_latent_cache_manifest(output_dir=tmp_path)


def test_seed_latents_only_hardlinks_preserves_inode(tmp_path: Path) -> None:
    source = tmp_path / "old"
    dest = tmp_path / "new"
    record = _latents_only_record(0, "case_a")
    _write_latents_only_tensor(source, record)
    result = latent_cache.seed_latents_only_hardlinks(
        source_dir=source,
        dest_dir=dest,
        case_ids=["case_a"],
        expected_provenance_by_case_id={"case_a": record["provenance"]},
    )
    assert result["hardlinked"] == 1
    source_file = source / "latents" / "case_a.safetensors"
    dest_file = dest / "latents" / "case_a.safetensors"
    assert source_file.stat().st_ino == dest_file.stat().st_ino
    with pytest.raises(FileExistsError, match="not empty"):
        latent_cache.seed_latents_only_hardlinks(
            source_dir=source,
            dest_dir=dest,
            case_ids=["case_a"],
        expected_provenance_by_case_id={"case_a": record["provenance"]},
        )


def test_encode_batch_size_and_decode_workers_fail_loud() -> None:
    with pytest.raises(ValueError, match="precompute.encode_batch_size"):
        latent_cache._require_int_at_least(0, name="precompute.encode_batch_size", minimum=1)
    with pytest.raises(ValueError, match="missing"):
        latent_cache._precompute_runtime(
            SimpleNamespace(resolved={"precompute": {"prefetch_factor": 1}})
        )
    config = _source_config("latents_only")
    config.resolved["precompute"]["video_decode_num_threads"] = 0
    with pytest.raises(ValueError, match="video_decode_num_threads"):
        latent_cache._precompute_runtime(config)


def test_same_shape_batches_respect_encode_batch_size() -> None:
    items = [
        {
            "index": 0,
            "case_id": "a",
            "sample": {"video": torch.zeros(3, 5, 8, 8, dtype=torch.uint8)},
        },
        {
            "index": 1,
            "case_id": "b",
            "sample": {"video": torch.zeros(3, 9, 8, 8, dtype=torch.uint8)},
        },
        {
            "index": 2,
            "case_id": "c",
            "sample": {"video": torch.zeros(3, 5, 8, 8, dtype=torch.uint8)},
        },
    ]
    batches = latent_cache._split_same_shape_groups(items, encode_batch_size=2)
    assert [len(batch) for batch in batches] == [2, 1]
    assert [item["case_id"] for item in batches[0]] == ["a", "c"]
    assert batches[1][0]["case_id"] == "b"
    with pytest.raises(ValueError, match="must be a \\[C,T,H,W\\] tensor"):
        latent_cache._split_same_shape_groups(
            [{"index": 0, "case_id": "bad", "sample": {"video": torch.zeros(3, 8, 8)}}],
            encode_batch_size=1,
        )


def test_ordered_prefetch_is_bounded_and_preserves_assignment(monkeypatch):
    import threading

    calls = []
    fourth = threading.Event()
    fifth = threading.Event()

    def decode(dataset, index, case_id, contract, cache_layout):
        calls.append(index)
        if index == 3:
            fourth.set()
        if index == 4:
            fifth.set()
        if index == 0:
            assert fourth.wait(timeout=5)
        return {"index": index, "case_id": case_id}

    monkeypatch.setattr(latent_cache, "_load_decoded_sample", decode)
    iterator = latent_cache._iter_decoded_samples(
        object(), list(range(10)), [f"case-{i}" for i in range(10)], {}, "latents_only",
        decode_num_workers=2, prefetch_factor=2,
    )
    first = next(iterator)
    assert first == {"index": 0, "case_id": "case-0"}
    assert fifth.wait(timeout=5)
    assert set(calls) == {0, 1, 2, 3, 4}
    assert [item["index"] for item in [first, *iterator]] == list(range(10))
    assert sorted(calls) == list(range(10))


def test_precompute_batches_same_shape_latents_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "cache"
    batch_sizes: list[int] = []

    class _BatchModel(_LatentsOnlyModel):
        def encode_batch_for_cache(self, samples: list[dict[str, object]]):
            batch_sizes.append(len(samples))
            return super().encode_batch_for_cache(samples)

    monkeypatch.setattr(
        latent_cache,
        "load_experiment_config",
        lambda _: _source_config("latents_only", encode_batch_size=2),
    )
    monkeypatch.setattr(latent_cache, "_build_source_dataset", lambda *args, **kwargs: _Dataset())
    monkeypatch.setattr(latent_cache.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(latent_cache.wan22_backend, "configure_args", lambda *args: object())
    monkeypatch.setattr(
        latent_cache.wan22_backend,
        "build_cache_encoder",
        lambda *args, **kwargs: _BatchModel(),
    )
    monkeypatch.setattr(latent_cache, "build_common_training_args", lambda _: SimpleNamespace())
    result = latent_cache.precompute_latent_cache(
        **_precompute_kwargs(tmp_path, output_dir)
    )
    assert result["written"] == 2
    assert batch_sizes == [2]
    shard = json.loads((output_dir / "shards/shard-00000.json").read_text(encoding="utf-8"))
    assert shard["schema"] == latent_cache.SHARD_MANIFEST_SCHEMA_V5
    for record in shard["records"]:
        path = output_dir / record["cache_file"]
        header = latent_cache._read_cache_file_header(path, cache_layout="latents_only")
        assert header.keys == frozenset({"input_latents"})


@pytest.mark.parametrize("broken", [b"", b"\x08\x00\x00"])
def test_damaged_published_latent_is_retained_and_stops_before_recompute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    broken: bytes,
) -> None:
    output_dir = tmp_path / "cache"
    path = output_dir / "latents" / "case_a.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(broken)
    temporary = path.with_name(f".{path.name}.12345.67890.tmp")
    temporary.write_bytes(b"retained unpublished partial")
    monkeypatch.setattr(
        latent_cache,
        "load_experiment_config",
        lambda _: _source_config("latents_only"),
    )
    monkeypatch.setattr(latent_cache, "_build_source_dataset", lambda *args, **kwargs: _Dataset())
    with pytest.raises(ValueError, match="published latent cache.*refusing automatic recompute"):
        latent_cache.precompute_latent_cache(**_precompute_kwargs(tmp_path, output_dir))
    event = json.loads(next((output_dir / "quarantine").glob("*/record.json")).read_text())
    assert event["action"] == "stop_after_quarantine"
    retained = {entry["original_path"]: Path(entry["quarantine_path"]).read_bytes() for entry in event["files"]}
    assert retained == {str(path): broken, str(temporary): b"retained unpublished partial"}
    assert not path.exists() and not temporary.exists()
    assert not (output_dir / "shards/shard-00000.json").exists()


def test_precompute_missing_section_fails_loud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        latent_cache,
        "load_experiment_config",
        lambda _: SimpleNamespace(resolved={"dataset": {"cache_layout": "latents_only"}}),
    )
    monkeypatch.setattr(latent_cache, "_build_source_dataset", lambda *args, **kwargs: _Dataset())
    with pytest.raises(ValueError, match="precompute"):
        latent_cache.precompute_latent_cache(**_precompute_kwargs(tmp_path, tmp_path / "cache"))


@pytest.fixture(autouse=True)
def cache_contract(monkeypatch):
    monkeypatch.setattr(latent_cache, "load_cache_contract", lambda *a: contract())
    # Existing encode-adapter unit tests simulate CUDA with CPU tensor models.
    # These stubs are not GPU memory measurements.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(name="unit-test CUDA stub", total_memory=0))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 0)


def test_scene_case_id_with_slash_can_reuse_cache(tmp_path: Path) -> None:
    rows = json.loads((Path(__file__).parents[2] / 'fixtures/scene_cache/real_rows.json').read_text())
    case_id = next(row['case_id'] for row in rows if '/' in row['case_id'])
    record = _latents_only_record(0, case_id)
    _write_latents_only_tensor(tmp_path, record)
    actual = latent_cache._record_from_existing_cache_file(
        tmp_path / record['cache_file'], case_id=case_id, index=0,
        cache_layout='latents_only', expected_provenance=record['provenance'])
    assert actual['case_id'] == case_id
    assert actual['cache_file'] == record['cache_file']
