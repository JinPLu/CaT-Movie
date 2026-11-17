"""Offline Wan2.2 cache with a small structural manifest.

Two layouts, never mixed in one directory:

* ``latents_and_context`` (schema v4): each row stores ``input_latents`` and
  per-shot ``context`` in the same safetensors file. Existing P4 caches.
* ``latents_only`` (schema v5): each row stores only ``input_latents``. T5
  context is encoded live at train time or loaded from a separate context
  cache (R1 requires live_t5). VAE latents bind only to video.

Decode and VAE encode are decoupled. A bounded ordered thread pool reads
explicit FFmpeg source frames and resizes them while the owning process encodes
and atomically publishes each shot. The queue keeps decoded RGB in this process,
without a DataLoader shared-memory copy. Each VAE call retains the shot boundary.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from mspc.config import ExperimentConfig, load_experiment_config
from mspc.data.multishot import MultishotCsvDataset
from mspc.data.cache_provenance import (
    PROVENANCE_SCHEMA, load_cache_contract, validate_provenance, require_matching_provenance,
)
from mspc.training.backends import wan22 as wan22_backend
from mspc.training.training_args import build_common_training_args
from mspc.core.wan22_contract import (
    WAN22_CACHE_LAYOUT_LATENTS_AND_CONTEXT,
    WAN22_CACHE_LAYOUT_LATENTS_ONLY,
    WAN22_CACHE_LAYOUTS,
)


SHARD_MANIFEST_SCHEMA_V4 = "mspc_latent_cache_shard_v4"
SHARD_MANIFEST_SCHEMA_V5 = "mspc_latent_cache_shard_v5"
FINAL_MANIFEST_SCHEMA_V4 = "mspc_latent_cache_manifest_v4"
FINAL_MANIFEST_SCHEMA_V5 = "mspc_latent_cache_manifest_v5"
SHARD_MANIFEST_SCHEMA = SHARD_MANIFEST_SCHEMA_V4
FINAL_MANIFEST_SCHEMA = FINAL_MANIFEST_SCHEMA_V4
_SHARD_SCHEMA_BY_LAYOUT = {
    WAN22_CACHE_LAYOUT_LATENTS_AND_CONTEXT: SHARD_MANIFEST_SCHEMA_V4,
    WAN22_CACHE_LAYOUT_LATENTS_ONLY: SHARD_MANIFEST_SCHEMA_V5,
}
_FINAL_SCHEMA_BY_LAYOUT = {
    WAN22_CACHE_LAYOUT_LATENTS_AND_CONTEXT: FINAL_MANIFEST_SCHEMA_V4,
    WAN22_CACHE_LAYOUT_LATENTS_ONLY: FINAL_MANIFEST_SCHEMA_V5,
}
_LAYOUT_BY_SHARD_SCHEMA = {
    SHARD_MANIFEST_SCHEMA_V4: WAN22_CACHE_LAYOUT_LATENTS_AND_CONTEXT,
    SHARD_MANIFEST_SCHEMA_V5: WAN22_CACHE_LAYOUT_LATENTS_ONLY,
}
_LAYOUT_BY_FINAL_SCHEMA = {
    FINAL_MANIFEST_SCHEMA_V4: WAN22_CACHE_LAYOUT_LATENTS_AND_CONTEXT,
    "mspc_latent_cache_manifest_v3": WAN22_CACHE_LAYOUT_LATENTS_AND_CONTEXT,
    FINAL_MANIFEST_SCHEMA_V5: WAN22_CACHE_LAYOUT_LATENTS_ONLY,
}
_SHARD_MANIFEST_RE = re.compile(r"^shard-(?P<index>[0-9]{5})\.json$")
_PROGRESS_LOG_INTERVAL = 50
_VAE_TEMPORAL_STRIDE = 4


def _atomic_publish(tmp: Path, path: Path) -> None:
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_cache_target(path: Path) -> Path | None:
    """Recognize the exact unpublished safetensors name emitted by _atomic_save."""
    match = re.fullmatch(r"\.(.+\.safetensors)\.[0-9]+\.[0-9]+\.tmp", path.name, flags=re.DOTALL)
    return None if match is None else path.with_name(match.group(1))


def _atomic_save(
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    save_file(tensors, str(tmp), metadata=metadata)
    _atomic_publish(tmp, path)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    _atomic_publish(tmp, path)


@dataclass(frozen=True)
class _EncodedSample:
    index: int
    case_id: str
    cache_path: Path
    tensors: dict[str, torch.Tensor]
    metadata: dict[str, str]
    record: dict[str, Any]
    dtoh_seconds: float


@dataclass(frozen=True)
class _SavedRecord:
    record: dict[str, Any]
    service_seconds: float


@dataclass(frozen=True)
class _CacheFileHeader:
    keys: frozenset[str]
    metadata: dict[str, str]
    input_latents_shape: list[int]
    context_shape: list[int] | None
    input_latents_dtype: str
    context_dtype: str | None


def require_cache_layout(value: object) -> str:
    if not isinstance(value, str) or value not in WAN22_CACHE_LAYOUTS:
        raise ValueError(
            f"cache_layout must be one of {sorted(WAN22_CACHE_LAYOUTS)}: {value!r}"
        )
    return value


def cache_layout_from_shard_schema(schema: object) -> str:
    if not isinstance(schema, str) or schema not in _LAYOUT_BY_SHARD_SCHEMA:
        raise ValueError(f"latent-cache shard schema is unsupported: {schema!r}")
    return _LAYOUT_BY_SHARD_SCHEMA[schema]


def cache_layout_from_manifest_schema(schema: object) -> str:
    if not isinstance(schema, str) or schema not in _LAYOUT_BY_FINAL_SCHEMA:
        raise ValueError(f"latent-cache manifest schema is unsupported: {schema!r}")
    return _LAYOUT_BY_FINAL_SCHEMA[schema]


def _cache_layout_from_source_config(config: ExperimentConfig) -> str:
    layout = config.resolved["dataset"].get("cache_layout")
    return require_cache_layout(layout)


def _require_int_at_least(value: int, *, name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}: {value!r}")


def _precompute_runtime(config: ExperimentConfig) -> dict[str, Any]:
    try:
        precompute = config.resolved["precompute"]
    except KeyError as exc:
        raise ValueError("latent precompute requires config.precompute") from exc
    if not isinstance(precompute, Mapping):
        raise ValueError("latent precompute config.precompute must be a mapping")
    required = (
        "provenance_path",
        "encode_batch_size",
        "dataset_num_workers",
        "prefetch_factor",
        "video_decode_num_threads",
    )
    missing = [key for key in required if key not in precompute]
    if missing:
        raise ValueError(f"latent precompute config.precompute is missing {missing}")
    encode_batch_size = precompute["encode_batch_size"]
    decode_num_workers = precompute["dataset_num_workers"]
    prefetch_factor = precompute["prefetch_factor"]
    video_decode_num_threads = precompute["video_decode_num_threads"]
    _require_int_at_least(encode_batch_size, name="precompute.encode_batch_size", minimum=1)
    _require_int_at_least(decode_num_workers, name="precompute.dataset_num_workers", minimum=0)
    _require_int_at_least(prefetch_factor, name="precompute.prefetch_factor", minimum=1)
    _require_int_at_least(video_decode_num_threads, name="precompute.video_decode_num_threads", minimum=1)
    return {
        "provenance_path": precompute["provenance_path"],
        "encode_batch_size": encode_batch_size,
        "decode_num_workers": decode_num_workers,
        "prefetch_factor": prefetch_factor,
        "video_decode_num_threads": video_decode_num_threads,
    }


def _canonical_shot_groups(groups: Any, *, label: str) -> list[list[int]] | None:
    if groups is None:
        return None
    if not isinstance(groups, (list, tuple)):
        raise ValueError(f"latent-cache {label} must be a list of integer pairs")
    canonical: list[list[int]] = []
    for group in groups:
        if not isinstance(group, (list, tuple)) or len(group) != 2:
            raise ValueError(f"latent-cache {label} must contain integer [start, end] pairs")
        start, end = group
        if type(start) is not int or type(end) is not int:
            raise ValueError(f"latent-cache {label} must contain integer [start, end] pairs")
        canonical.append([start, end])
    return canonical


def _sample_provenance(media: dict[str, Any], prompts: list[str], contract: dict[str, Any],
                       cache_layout: str) -> dict[str, Any]:
    if media["schema"] == "scene_window_v1" and cache_layout != WAN22_CACHE_LAYOUT_LATENTS_ONLY:
        raise ValueError("scene_window_v1 requires latents_only; text is compiled separately")
    evidence = {"schema": PROVENANCE_SCHEMA, "cache_version": contract["cache_version"],
                "media": media, "encoding": contract["encoding"], "models": {"vae": contract["models"]["vae"]}, "context": None}
    if cache_layout == WAN22_CACHE_LAYOUT_LATENTS_AND_CONTEXT:
        evidence["models"] = contract["models"]
        evidence["context"] = {"prompts": prompts, "max_tokens": 512,
            "prompt_template_revision": contract["prompt_template_revision"]}
    return validate_provenance(evidence)


def _expected_provenance(dataset: MultishotCsvDataset, index: int,
                         contract: dict[str, Any], cache_layout: str) -> dict[str, Any]:
    description = dataset.describe_media(index)
    return _sample_provenance(description["provenance"], list(description["row"]["_prompts"]),
                              contract, cache_layout)


def _build_source_dataset(
    config: ExperimentConfig,
    *,
    video_decode_num_threads: int,
) -> MultishotCsvDataset:
    cfg = dict(build_common_training_args(config).dataset_cfg)
    cfg["repeat"] = 1
    cfg["video_decode_num_threads"] = video_decode_num_threads
    cfg["video_output_mode"] = "uint8"
    return MultishotCsvDataset(**cfg)


def _validate_groups(groups: Any, *, limit: int, count: int, label: str) -> None:
    if groups is None:
        if count != 1:
            raise ValueError(f"latent-cache {label} is null but shot_count={count}")
        return
    if not isinstance(groups, list) or len(groups) != count:
        raise ValueError(f"latent-cache {label} does not match shot_count={count}")
    cursor = 0
    for group in groups:
        if (
            not isinstance(group, list)
            or len(group) != 2
            or any(type(value) is not int for value in group)
        ):
            raise ValueError(f"latent-cache {label} must contain integer [start, end] pairs")
        start, end = group
        if start != cursor or end <= start or end > limit:
            raise ValueError(f"latent-cache {label} is not contiguous over [0, {limit}]")
        cursor = end
    if cursor != limit:
        raise ValueError(f"latent-cache {label} does not cover [0, {limit}]")


_RECORD_FIELDS_SHARED = frozenset(
    {
        "case_id",
        "index",
        "cache_file",
        "input_latents_shape",
        "shot_groups",
        "latent_shot_groups",
        "height",
        "width",
        "num_frames",
        "shot_count",
        "provenance",
    }
)
_RECORD_FIELDS_V4 = _RECORD_FIELDS_SHARED | {"context_shape"}
_RECORD_FIELDS = _RECORD_FIELDS_V4


def _record_fields_for_layout(cache_layout: str) -> frozenset[str]:
    layout = require_cache_layout(cache_layout)
    if layout == WAN22_CACHE_LAYOUT_LATENTS_ONLY:
        return _RECORD_FIELDS_SHARED
    return _RECORD_FIELDS_V4


def _project_record(
    value: Any,
    *,
    path: Path,
    cache_layout: str = WAN22_CACHE_LAYOUT_LATENTS_AND_CONTEXT,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"latent-cache record must be an object: {path}")
    required = _record_fields_for_layout(cache_layout)
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"latent-cache record is missing fields {missing}: {path}")
    if cache_layout == WAN22_CACHE_LAYOUT_LATENTS_ONLY and "context_shape" in value:
        raise ValueError(f"latents-only cache record must not include context_shape: {path}")
    return {key: value[key] for key in required}


def _validate_record_layout(
    record: dict[str, Any],
    path: Path,
    *,
    cache_layout: str = WAN22_CACHE_LAYOUT_LATENTS_AND_CONTEXT,
) -> None:
    layout = require_cache_layout(cache_layout)
    provenance = validate_provenance(record.get("provenance"))
    if layout == WAN22_CACHE_LAYOUT_LATENTS_AND_CONTEXT and provenance.get("context") is None:
        raise ValueError(f"latent-cache embedded context provenance is missing: {path}")
    for field in ("index", "height", "width", "num_frames", "shot_count"):
        if type(record.get(field)) is not int or record[field] < (0 if field == "index" else 1):
            raise ValueError(f"latent-cache record {field} is invalid: {path}")
    case_id = record.get("case_id")
    if not isinstance(case_id, str) or not case_id or case_id != case_id.strip():
        raise ValueError(f"latent-cache record case_id is invalid: {path}")
    if record.get("cache_file") != f"latents/{case_id}.safetensors":
        raise ValueError(f"latent-cache record cache_file is invalid: {path}")
    input_shape = record.get("input_latents_shape")
    if (
        not isinstance(input_shape, list)
        or len(input_shape) != 5
        or any(type(value) is not int or value <= 0 for value in input_shape)
        or input_shape[0] != 1
    ):
        raise ValueError(f"latent-cache input_latents must have layout [1,C,T,H,W]: {path}")
    if layout == WAN22_CACHE_LAYOUT_LATENTS_AND_CONTEXT:
        context_shape = record.get("context_shape")
        if (
            not isinstance(context_shape, list)
            or len(context_shape) != 3
            or any(type(value) is not int or value <= 0 for value in context_shape)
            or context_shape[0] != record["shot_count"]
        ):
            raise ValueError(f"latent-cache context must have layout [shot_count,L,D]: {path}")
    elif "context_shape" in record:
        raise ValueError(f"latents-only cache record must not include context_shape: {path}")
    latent_height, latent_width = input_shape[-2:]
    if (
        record["height"] % latent_height != 0
        or record["width"] % latent_width != 0
        or record["height"] // latent_height != record["width"] // latent_width
    ):
        raise ValueError(f"latent-cache spatial latent layout is incoherent: {path}")
    if (record["shot_groups"] is None) != (record["latent_shot_groups"] is None):
        raise ValueError(f"latent-cache RGB and latent shot layouts disagree: {path}")
    _validate_groups(
        record["shot_groups"],
        limit=record["num_frames"],
        count=record["shot_count"],
        label="shot_groups",
    )
    _validate_groups(
        record["latent_shot_groups"],
        limit=input_shape[2],
        count=record["shot_count"],
        label="latent_shot_groups",
    )
    shot_groups = record["shot_groups"]
    latent_shot_groups = record["latent_shot_groups"]
    if shot_groups is None:
        rgb_lengths = [record["num_frames"]]
        latent_lengths = [input_shape[2]]
    else:
        rgb_lengths = [end - start for start, end in shot_groups]
        latent_lengths = [end - start for start, end in latent_shot_groups]
    for shot_index, (rgb_frames, latent_frames) in enumerate(
        zip(rgb_lengths, latent_lengths, strict=True)
    ):
        if (rgb_frames - 1) % _VAE_TEMPORAL_STRIDE != 0:
            raise ValueError(f"latent-cache shot {shot_index} RGB length is not 4n+1: {path}")
        expected = (rgb_frames - 1) // _VAE_TEMPORAL_STRIDE + 1
        if latent_frames != expected:
            raise ValueError(
                f"latent-cache shot {shot_index} temporal latent layout is incoherent: {path}"
            )


def _expected_tensor_keys(cache_layout: str) -> frozenset[str]:
    if cache_layout == WAN22_CACHE_LAYOUT_LATENTS_ONLY:
        return frozenset({"input_latents"})
    return frozenset({"input_latents", "context"})


def _read_cache_file_header(path: Path, *, cache_layout: str) -> _CacheFileHeader:
    layout = require_cache_layout(cache_layout)
    expected_keys = _expected_tensor_keys(layout)
    if not path.is_file():
        raise FileNotFoundError(f"latent-cache tensor is missing: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = frozenset(handle.keys())
        if keys != expected_keys:
            raise ValueError(
                f"latent-cache tensor keys for {path} are {sorted(keys)}, "
                f"expected {sorted(expected_keys)}"
            )
        input_slice = handle.get_slice("input_latents")
        context_slice = (
            None if layout == WAN22_CACHE_LAYOUT_LATENTS_ONLY else handle.get_slice("context")
        )
        return _CacheFileHeader(
            keys=keys,
            metadata=dict(handle.metadata() or {}),
            input_latents_shape=list(input_slice.get_shape()),
            context_shape=None if context_slice is None else list(context_slice.get_shape()),
            input_latents_dtype=str(input_slice.get_dtype()),
            context_dtype=None if context_slice is None else str(context_slice.get_dtype()),
        )


def _validate_cache_file(
    path: Path,
    record: dict[str, Any],
    *,
    cache_layout: str,
    _header: _CacheFileHeader | None = None,
) -> dict[str, str]:
    layout = require_cache_layout(cache_layout)
    _validate_record_layout(record, path, cache_layout=layout)
    header = _header or _read_cache_file_header(path, cache_layout=layout)
    if header.input_latents_dtype != "BF16":
        raise ValueError(f"latent-cache tensors must use BF16: {path}")
    if layout == WAN22_CACHE_LAYOUT_LATENTS_AND_CONTEXT:
        if header.context_dtype != "BF16":
            raise ValueError(f"latent-cache tensors must both use BF16: {path}")
    if header.input_latents_shape != record["input_latents_shape"]:
        raise ValueError(f"latent-cache input_latents shape mismatch for {path}")
    if layout == WAN22_CACHE_LAYOUT_LATENTS_AND_CONTEXT:
        if header.context_shape != record["context_shape"]:
            raise ValueError(f"latent-cache context shape mismatch for {path}")
    metadata = header.metadata
    required_metadata = {
        "case_id": str(record["case_id"]),
        "input_latents_shape": json.dumps(record["input_latents_shape"]),
        "shot_groups": json.dumps(record["shot_groups"]),
        "latent_shot_groups": json.dumps(record["latent_shot_groups"]),
        "height": str(record["height"]),
        "width": str(record["width"]),
        "num_frames": str(record["num_frames"]),
        "cache_layout": layout,
    }
    if layout == WAN22_CACHE_LAYOUT_LATENTS_AND_CONTEXT:
        required_metadata["context_shape"] = json.dumps(record["context_shape"])
    for key, expected in required_metadata.items():
        if key == "cache_layout" and layout == WAN22_CACHE_LAYOUT_LATENTS_AND_CONTEXT:
            stored = metadata.get(key)
            if stored is None:
                continue
            if stored != expected:
                raise ValueError(f"latent-cache metadata {key} mismatch for {path}")
            continue
        if metadata.get(key) != expected:
            raise ValueError(f"latent-cache metadata {key} mismatch for {path}")
    stored_provenance = json.loads(metadata["provenance"]) if "provenance" in metadata else None
    require_matching_provenance(stored_provenance, record["provenance"])
    return metadata


def _record_from_existing_cache_file(
    path: Path,
    *,
    case_id: str,
    index: int,
    cache_layout: str,
    expected_provenance: dict[str, Any],
) -> dict[str, Any]:
    layout = require_cache_layout(cache_layout)
    header = _read_cache_file_header(path, cache_layout=layout)
    metadata = header.metadata
    stored_case_id = metadata.get("case_id")
    if stored_case_id != case_id:
        raise ValueError(
            f"latent-cache case_id mismatch for {path}: stored={stored_case_id!r} expected={case_id!r}"
        )
    expected_parts = Path(f"{case_id}.safetensors").parts
    temporary_target = _atomic_cache_target(path)
    actual_parts = path.parts if temporary_target is None else temporary_target.parts
    if actual_parts[-len(expected_parts):] != expected_parts:
        raise ValueError(f"latent-cache filename does not match case_id={case_id!r}: {path}")
    required = (
        "input_latents_shape",
        "shot_groups",
        "latent_shot_groups",
        "height",
        "width",
        "num_frames",
        "provenance",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"latent-cache metadata is missing {missing}: {path}")
    shot_groups = json.loads(metadata["shot_groups"])
    record: dict[str, Any] = {
        "case_id": case_id,
        "index": index,
        "provenance": json.loads(metadata["provenance"]),
        "cache_file": f"latents/{case_id}.safetensors",
        "input_latents_shape": json.loads(metadata["input_latents_shape"]),
        "shot_groups": shot_groups,
        "latent_shot_groups": json.loads(metadata["latent_shot_groups"]),
        "height": int(metadata["height"]),
        "width": int(metadata["width"]),
        "num_frames": int(metadata["num_frames"]),
        "shot_count": 1 if shot_groups is None else len(shot_groups),
    }
    if layout == WAN22_CACHE_LAYOUT_LATENTS_AND_CONTEXT:
        if "context_shape" not in metadata:
            raise ValueError(f"latent-cache metadata is missing context_shape: {path}")
        record["context_shape"] = json.loads(metadata["context_shape"])
    _validate_cache_file(path, record, cache_layout=layout, _header=header)
    require_matching_provenance(record["provenance"], expected_provenance)
    return record


def _atomic_cache_temporaries(path: Path) -> list[Path]:
    """Only names emitted by this owner's _atomic_save, not arbitrary .tmp files."""
    if not path.parent.is_dir():
        return []
    return sorted(candidate for candidate in path.parent.iterdir()
                  if _atomic_cache_target(candidate) == path and candidate.is_file())


def _quarantine_cache_files(
    paths: Sequence[Path], *, cache_root: Path, reason: str, action: str,
    expected_provenance: dict[str, Any],
) -> Path:
    """Retain bytes and an intent before moving this run's rejected files."""
    directory = cache_root / "quarantine" / f"{time.time_ns()}-{os.getpid()}"
    directory.mkdir(parents=True, exist_ok=False)
    records = [{"original_path": str(path),
                "quarantine_path": str(directory / "files" / path.relative_to(cache_root)),
                "bytes": path.stat().st_size} for path in paths]
    event = {"action": action, "reason": reason, "files": records,
             "expected_provenance": expected_provenance}
    _atomic_write_json({**event, "status": "quarantine_started"}, directory / "intent.json")
    for path, record in zip(paths, records, strict=True):
        destination = Path(record["quarantine_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.rename(destination)
    _atomic_write_json({**event, "status": "quarantined"}, directory / "record.json")
    print("[precompute-quarantine] " + json.dumps({"record": str(directory / "record.json"),
          "reason": reason, "action": action}, ensure_ascii=False), flush=True)
    return directory


def _existing_cache_record(
    path: Path,
    *,
    cache_root: Path,
    case_id: str,
    index: int,
    cache_layout: str,
    expected_provenance: dict[str, Any],
) -> tuple[Literal["missing", "valid"], dict[str, Any] | None]:
    """Skip valid files; retain damaged published files and stop this invocation.

    The selected cache version/source is immutable and each shard has one writer.
    Explicit continuation may inspect old atomic temporaries only after the prior
    producer has exited; this function does not reclaim a live producer's files.
    A missing final name permits recomputing an unpublished partial in that same
    namespace. Readable metadata/source conflicts always stop, without replacing
    a potentially valid cache belonging to another source.
    """

    if not path.is_file():
        if path.exists():
            raise ValueError(f"latent-cache final path is not a file: {path}")
        for temporary in _atomic_cache_temporaries(path):
            reason = "unpublished same-source atomic cache before explicit continuation"
            conflict = None
            try:
                _record_from_existing_cache_file(temporary, case_id=case_id, index=index,
                    cache_layout=cache_layout, expected_provenance=expected_provenance)
            except (SafetensorError, EOFError) as exc:
                # No final was ever published. A truncated atomic write has no
                # readable header; the explicitly selected immutable cache
                # version/source supplies its continuation boundary.
                reason = f"unpublished partial atomic write in the selected cache namespace: {exc}"
            except (ValueError, OSError) as exc:
                reason, conflict = f"unpublished atomic cache rejected: {type(exc).__name__}: {exc}", exc
            directory = _quarantine_cache_files([temporary], cache_root=cache_root,
                reason=reason, expected_provenance=expected_provenance,
                action="stop_after_quarantine" if conflict else "continue_after_quarantine_of_unpublished_write")
            if conflict is not None:
                raise ValueError(f"unpublished cache conflicts with continuation; retained at {directory}") from conflict
        return "missing", None
    if path.stat().st_size == 0:
        directory = _quarantine_cache_files([path, *_atomic_cache_temporaries(path)],
            cache_root=cache_root, reason="published latent cache is empty",
            action="stop_after_quarantine", expected_provenance=expected_provenance)
        raise ValueError(f"published latent cache is empty; retained at {directory}; refusing automatic recompute")
    try:
        return "valid", _record_from_existing_cache_file(
            path,
            case_id=case_id,
            index=index,
            cache_layout=cache_layout,
            expected_provenance=expected_provenance,
        )
    except (SafetensorError, OSError, EOFError) as exc:
        directory = _quarantine_cache_files([path, *_atomic_cache_temporaries(path)],
            cache_root=cache_root, reason=f"published cache is unreadable: {type(exc).__name__}: {exc}",
            action="stop_after_quarantine", expected_provenance=expected_provenance)
        raise ValueError(f"published latent cache is unreadable; retained at {directory}; refusing automatic recompute") from exc


def seed_latents_only_hardlinks(
    *,
    source_dir: Path,
    dest_dir: Path,
    case_ids: Sequence[str],
    expected_provenance_by_case_id: Mapping[str, dict[str, Any]],
    cache_layout: str = WAN22_CACHE_LAYOUT_LATENTS_ONLY,
) -> dict[str, Any]:
    """Hardlink validated ``{input_latents}`` files into a fresh cache latents dir."""

    layout = require_cache_layout(cache_layout)
    if layout != WAN22_CACHE_LAYOUT_LATENTS_ONLY:
        raise ValueError("seed_latents_only_hardlinks requires cache_layout=latents_only")
    if not case_ids:
        raise ValueError("case_ids is empty")
    source_latents = source_dir / "latents"
    dest_latents = dest_dir / "latents"
    if dest_dir.exists() and any(dest_dir.iterdir()) and not dest_latents.exists():
        raise FileExistsError(f"destination cache dir already exists: {dest_dir}")
    if dest_latents.exists() and any(dest_latents.iterdir()):
        raise FileExistsError(f"destination latents dir is not empty: {dest_latents}")
    seen: set[str] = set()
    sources: list[tuple[str, Path]] = []
    for case_id in case_ids:
        if not isinstance(case_id, str) or not case_id.strip() or case_id != case_id.strip():
            raise ValueError(f"case_id must be a non-empty trimmed string: {case_id!r}")
        if case_id in seen:
            raise ValueError(f"duplicate case_id={case_id}")
        seen.add(case_id)
        path = source_latents / f"{case_id}.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"validated latent is missing: {path}")
        evidence = expected_provenance_by_case_id[case_id]
        validate_provenance(evidence)
        _record_from_existing_cache_file(path, case_id=case_id, index=0, cache_layout=layout,
                                         expected_provenance=evidence)
        sources.append((case_id, path))
    dest_latents.mkdir(parents=True, exist_ok=True)
    for case_id, source in sources:
        dest = dest_latents / f"{case_id}.safetensors"
        os.link(source, dest)
    return {
        "source_dir": str(source_dir),
        "dest_dir": str(dest_dir),
        "hardlinked": len(sources),
        "cache_layout": layout,
    }


def _load_decoded_sample(
    dataset: MultishotCsvDataset,
    index: int,
    case_id: str,
    contract: dict[str, Any],
    cache_layout: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    sample = dataset._load_sample(index)
    stored = sample.get("case_id")
    if stored != case_id:
        raise RuntimeError(
            f"latent-cache decode case_id mismatch: stored={stored!r} expected={case_id!r}"
        )
    return {
        "index": index,
        "case_id": case_id,
        "sample": sample,
        "provenance": _sample_provenance(sample["media_provenance"], list(sample["prompt"]), contract, cache_layout),
        "service_seconds": time.perf_counter() - started,
    }


def _video_shape(sample: dict[str, Any]) -> tuple[int, ...]:
    video = sample.get("video")
    if not isinstance(video, torch.Tensor) or video.dim() != 4:
        raise ValueError("decoded sample.video must be a [C,T,H,W] tensor")
    return tuple(int(dim) for dim in video.shape)


def _split_same_shape_groups(
    items: Sequence[dict[str, Any]],
    *,
    encode_batch_size: int,
) -> list[list[dict[str, Any]]]:
    _require_int_at_least(encode_batch_size, name="encode_batch_size", minimum=1)
    grouped: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    order: list[tuple[int, ...]] = []
    for item in items:
        key = _video_shape(item["sample"])
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(item)
    batches: list[list[dict[str, Any]]] = []
    for key in order:
        bucket = grouped[key]
        for start in range(0, len(bucket), encode_batch_size):
            batches.append(bucket[start : start + encode_batch_size])
    return batches


def _iter_decoded_samples(
    dataset: MultishotCsvDataset,
    indices: Sequence[int],
    case_ids: Sequence[str],
    contract: dict[str, Any],
    cache_layout: str,
    *,
    decode_num_workers: int,
    prefetch_factor: int,
) -> Iterator[dict[str, Any]]:
    """Read in assignment order with an exact upper bound on queued RGB shots."""
    if len(indices) != len(case_ids):
        raise ValueError("assigned indices and case_ids must have equal length")
    _require_int_at_least(decode_num_workers, name="decode_num_workers", minimum=0)
    _require_int_at_least(prefetch_factor, name="prefetch_factor", minimum=1)
    assigned = list(zip(indices, case_ids, strict=True))
    if decode_num_workers == 0:
        for index, case_id in assigned:
            yield _load_decoded_sample(dataset, index, case_id, contract, cache_layout)
        return
    limit = decode_num_workers * prefetch_factor
    with ThreadPoolExecutor(max_workers=decode_num_workers) as pool:
        futures = {}
        next_position = 0
        for position in range(min(limit, len(assigned))):
            index, case_id = assigned[position]
            futures[position] = pool.submit(_load_decoded_sample, dataset, index, case_id, contract, cache_layout)
            next_position += 1
        try:
            for position in range(len(assigned)):
                item = futures.pop(position).result()
                # Keep four queued shots in addition to the consumer's current shot.
                if next_position < len(assigned):
                    index, case_id = assigned[next_position]
                    futures[next_position] = pool.submit(_load_decoded_sample, dataset, index, case_id, contract, cache_layout)
                    next_position += 1
                yield item
        finally:
            for future in futures.values():
                future.cancel()


def _finalize_encoded_sample(
    *,
    input_latents: torch.Tensor,
    context: torch.Tensor | None,
    embed_meta: dict[str, Any],
    index: int,
    case_id: str,
    cache_path: Path,
    cache_layout: str,
    dtoh_seconds: float,
    provenance: dict[str, Any],
) -> _EncodedSample:
    layout = require_cache_layout(cache_layout)
    transfer_started = time.perf_counter()
    input_latents = input_latents.detach().to("cpu", torch.bfloat16).contiguous()
    if layout == WAN22_CACHE_LAYOUT_LATENTS_ONLY:
        if context is not None:
            raise RuntimeError("latents-only cache encode must not produce context")
    else:
        if not isinstance(context, torch.Tensor):
            raise RuntimeError("latents_and_context cache encode did not produce context")
        context = context.detach().to("cpu", torch.bfloat16).contiguous()
    dtoh_seconds += time.perf_counter() - transfer_started
    for name, tensor in (("input_latents", input_latents), ("context", context)):
        if tensor is not None and not torch.isfinite(tensor).all():
            raise ValueError(f"cache encode produced non-finite {name}: case_id={case_id} index={index} path={cache_path}")
    shot_groups = _canonical_shot_groups(embed_meta["shot_groups"], label="shot_groups")
    latent_shot_groups = _canonical_shot_groups(
        embed_meta["latent_shot_groups"],
        label="latent_shot_groups",
    )
    sampling = provenance["media"]["sampling"]
    # Native single-shot VAE metadata uses None; provenance retains explicit ranges.
    encoded_shots = [[0, int(embed_meta["num_frames"])]] if shot_groups is None else shot_groups
    encoded_latents = [[0, int(input_latents.shape[2])]] if latent_shot_groups is None else latent_shot_groups
    if sampling["shot_groups"] != encoded_shots or sampling["latent_shot_groups"] != encoded_latents:
        raise ValueError("encoded shot layout differs from provenance sampling")
    shot_count = 1 if shot_groups is None else len(shot_groups)
    height = int(embed_meta["height"])
    width = int(embed_meta["width"])
    num_frames = int(embed_meta["num_frames"])
    record: dict[str, Any] = {
        "case_id": case_id,
        "index": index,
        "provenance": provenance,
        "cache_file": f"latents/{case_id}.safetensors",
        "input_latents_shape": list(input_latents.shape),
        "shot_groups": shot_groups,
        "latent_shot_groups": latent_shot_groups,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "shot_count": shot_count,
    }
    tensors: dict[str, torch.Tensor] = {"input_latents": input_latents}
    metadata = {
        "provenance": json.dumps(provenance, ensure_ascii=False, sort_keys=True),
        "case_id": case_id,
        "shot_groups": json.dumps(shot_groups),
        "latent_shot_groups": json.dumps(latent_shot_groups),
        "height": str(height),
        "width": str(width),
        "num_frames": str(num_frames),
        "input_latents_shape": json.dumps(list(input_latents.shape)),
        "cache_layout": layout,
    }
    if layout == WAN22_CACHE_LAYOUT_LATENTS_AND_CONTEXT:
        assert context is not None
        record["context_shape"] = list(context.shape)
        tensors["context"] = context
        metadata["context_shape"] = json.dumps(list(context.shape))
    _validate_record_layout(record, cache_path, cache_layout=layout)
    return _EncodedSample(
        index=index,
        case_id=case_id,
        cache_path=cache_path,
        tensors=tensors,
        metadata=metadata,
        record=record,
        dtoh_seconds=dtoh_seconds,
    )


def _encode_loaded_batch(
    model: Any,
    items: Sequence[dict[str, Any]],
    *,
    latents_dir: Path,
    cache_layout: str,
) -> list[_EncodedSample]:
    if not items:
        raise ValueError("encode batch is empty")
    encode_batch = getattr(model, "encode_batch_for_cache", None)
    if not callable(encode_batch):
        raise RuntimeError("cache encoder must implement encode_batch_for_cache")
    encode_started = time.perf_counter()
    outputs = encode_batch([item["sample"] for item in items])
    dtoh_seconds = time.perf_counter() - encode_started
    if not isinstance(outputs, list) or len(outputs) != len(items):
        raise RuntimeError("encode_batch_for_cache must return one result per sample")
    encoded: list[_EncodedSample] = []
    per_item = dtoh_seconds / len(items)
    for item, output in zip(items, outputs, strict=True):
        input_latents, context, embed_meta = output
        encoded.append(
            _finalize_encoded_sample(
                input_latents=input_latents,
                context=context,
                embed_meta=embed_meta,
                index=item["index"],
                case_id=item["case_id"],
                cache_path=latents_dir / f"{item['case_id']}.safetensors",
                cache_layout=cache_layout,
                dtoh_seconds=per_item,
                provenance=item["provenance"],
            )
        )
    return encoded


def _save_encoded_sample(encoded: _EncodedSample, *, cache_layout: str) -> _SavedRecord:
    started = time.perf_counter()
    _atomic_save(encoded.tensors, encoded.metadata, encoded.cache_path)
    _validate_cache_file(encoded.cache_path, encoded.record, cache_layout=cache_layout)
    return _SavedRecord(
        record=dict(encoded.record),
        service_seconds=time.perf_counter() - started,
    )


def _round_timing(value: float) -> float:
    return round(value, 6)


def _vae_parameter_summary(vae: torch.nn.Module) -> dict[str, Any]:
    """Report actual parameter residency, independently of the chosen AMP policy."""
    summary: dict[str, Any] = {"tensor_count": 0, "numel": 0, "bytes": 0,
        "by_dtype": {}, "by_device": {}, "all_frozen": True, "training": vae.training}
    for parameter in vae.parameters():
        count, size = parameter.numel(), parameter.numel() * parameter.element_size()
        summary["tensor_count"] += 1
        summary["numel"] += count
        summary["bytes"] += size
        summary["all_frozen"] = summary["all_frozen"] and not parameter.requires_grad
        for category, key in (("by_dtype", str(parameter.dtype)), ("by_device", str(parameter.device))):
            group = summary[category].setdefault(key, {"tensor_count": 0, "numel": 0, "bytes": 0})
            group["tensor_count"] += 1
            group["numel"] += count
            group["bytes"] += size
    return summary


def _cuda_peak_memory(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {"device": str(device), "device_name": properties.name,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "total_bytes": properties.total_memory}


def _emit_progress(
    *,
    shard_index: int,
    completed: int,
    assigned: int,
    started: float,
    decode_service: float,
    decode_wait: float,
    encode_dtoh: float,
    write_service: float,
    write_wait: float,
) -> None:
    elapsed = max(time.perf_counter() - started, 1e-9)
    payload = {
        "shard_index": shard_index,
        "completed": completed,
        "assigned": assigned,
        "samples_per_second": _round_timing(completed / elapsed),
        "decode_service": _round_timing(decode_service),
        "decode_wait": _round_timing(decode_wait),
        "encode_dtoh": _round_timing(encode_dtoh),
        "write_service": _round_timing(write_service),
        "write_wait": _round_timing(write_wait),
    }
    print(
        "[precompute-progress] "
        + json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True),
        flush=True,
    )


def _precompute_assigned_records(
    *,
    model: Any,
    dataset: MultishotCsvDataset,
    assigned_indices: list[int],
    assigned_case_ids: list[str],
    latents_dir: Path,
    encode_batch_size: int,
    decode_num_workers: int,
    prefetch_factor: int,
    started: float,
    shard_index: int,
    cache_layout: str,
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    decoded = _iter_decoded_samples(
        dataset, assigned_indices, assigned_case_ids, contract, cache_layout,
        decode_num_workers=decode_num_workers, prefetch_factor=prefetch_factor,
    )
    records_by_index: dict[int, dict[str, Any]] = {}
    decode_service = decode_wait = encode_dtoh = write_service = write_wait = 0.0
    pending: dict[tuple[int, ...], list[dict[str, Any]]] = {}

    def flush(items: list[dict[str, Any]]) -> None:
        nonlocal encode_dtoh, write_service
        encoded_batch = _encode_loaded_batch(
            model,
            items,
            latents_dir=latents_dir,
            cache_layout=cache_layout,
        )
        for encoded in encoded_batch:
            encode_dtoh += encoded.dtoh_seconds
            saved = _save_encoded_sample(encoded, cache_layout=cache_layout)
            write_service += saved.service_seconds
            records_by_index[saved.record["index"]] = saved.record
            item = next(item for item in items if item["index"] == encoded.index)
            print("[precompute-row] " + json.dumps({
                "index": encoded.index, "case_id": encoded.case_id,
                "decode_seconds": item["service_seconds"],
                "encode_dtoh_seconds": encoded.dtoh_seconds,
                "write_seconds": saved.service_seconds,
                "size_bytes": encoded.cache_path.stat().st_size,
            }, ensure_ascii=False), flush=True)
        completed = len(records_by_index)
        if completed % _PROGRESS_LOG_INTERVAL == 0:
            _emit_progress(
                shard_index=shard_index,
                completed=completed,
                assigned=len(assigned_indices),
                started=started,
                decode_service=decode_service,
                decode_wait=decode_wait,
                encode_dtoh=encode_dtoh,
                write_service=write_service,
                write_wait=write_wait,
            )

    loader_iter = iter(decoded)
    while True:
        wait_started = time.perf_counter()
        try:
            loader_batch = [next(loader_iter)]
        except StopIteration:
            break
        decode_wait += time.perf_counter() - wait_started
        if not isinstance(loader_batch, list) or not loader_batch:
            raise RuntimeError("decode prefetch must yield a non-empty list of samples")
        for item in loader_batch:
            decode_service += float(item["service_seconds"])
            key = _video_shape(item["sample"])
            bucket = pending.setdefault(key, [])
            bucket.append(item)
            while len(bucket) >= encode_batch_size:
                flush(bucket[:encode_batch_size])
                del bucket[:encode_batch_size]
    for bucket in pending.values():
        if bucket:
            flush(bucket)

    records = []
    for index, case_id in zip(assigned_indices, assigned_case_ids, strict=True):
        record = records_by_index.get(index)
        if record is None or record["case_id"] != case_id:
            raise ValueError(f"latent-cache encode missed case_id={case_id} index={index}")
        records.append(record)
    timing = {
        "decode_service": _round_timing(decode_service),
        "decode_wait": _round_timing(decode_wait),
        "encode_dtoh": _round_timing(encode_dtoh),
        "write_service": _round_timing(write_service),
        "write_wait": _round_timing(write_wait),
        "elapsed": _round_timing(time.perf_counter() - started),
        "encode_batch_size": encode_batch_size,
        "decode_num_workers": decode_num_workers,
    }
    if assigned_indices and len(records) % _PROGRESS_LOG_INTERVAL:
        _emit_progress(
            shard_index=shard_index,
            completed=len(records),
            assigned=len(assigned_indices),
            started=started,
            decode_service=decode_service,
            decode_wait=decode_wait,
            encode_dtoh=encode_dtoh,
            write_service=write_service,
            write_wait=write_wait,
        )
    return records, timing


def _write_shard_manifest(
    *,
    shard_path: Path,
    cache_layout: str,
    shard_index: int,
    num_shards: int,
    total_rows: int,
    records: list[dict[str, Any]],
) -> None:
    shard_schema = _SHARD_SCHEMA_BY_LAYOUT[cache_layout]
    shard_manifest: dict[str, Any] = {
        "schema": shard_schema,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "total_rows": total_rows,
        "records": records,
    }
    if cache_layout == WAN22_CACHE_LAYOUT_LATENTS_ONLY:
        shard_manifest["cache_layout"] = cache_layout
    _atomic_write_json(shard_manifest, shard_path)


def _load_complete_shard_records(
    *,
    shard_path: Path,
    cache_layout: str,
    shard_index: int,
    num_shards: int,
    total_rows: int,
    assigned_indices: list[int],
    assigned_case_ids: list[str],
    output_dir: Path,
    dataset: MultishotCsvDataset,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    payload = json.loads(shard_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"latent-cache shard manifest must be an object: {shard_path}")
    shard_schema = _SHARD_SCHEMA_BY_LAYOUT[cache_layout]
    if payload.get("schema") != shard_schema:
        raise ValueError(f"latent-cache shard schema mismatch: {shard_path}")
    if payload.get("shard_index") != shard_index or payload.get("num_shards") != num_shards:
        raise ValueError(f"latent-cache shard identity mismatch: {shard_path}")
    if payload.get("total_rows") != total_rows:
        raise ValueError(f"latent-cache shard total_rows mismatch: {shard_path}")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != len(assigned_indices):
        raise ValueError(f"latent-cache shard record count is incomplete: {shard_path}")
    records: list[dict[str, Any]] = []
    for expected_index, expected_case_id, raw_record in zip(
        assigned_indices, assigned_case_ids, raw_records, strict=True
    ):
        record = _project_record(raw_record, path=shard_path, cache_layout=cache_layout)
        if record["index"] != expected_index or record["case_id"] != expected_case_id:
            raise ValueError(f"latent-cache shard record identity mismatch: {shard_path}")
        _validate_cache_file(
            output_dir / record["cache_file"],
            record,
            cache_layout=cache_layout,
        )
        require_matching_provenance(record["provenance"], _expected_provenance(dataset, expected_index, contract, cache_layout))
        records.append(record)
    return records


def precompute_latent_cache(
    *,
    config_path: Path,
    output_dir: Path,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError(f"invalid shard: index={shard_index} num_shards={num_shards}")

    config = load_experiment_config(config_path)
    runtime = _precompute_runtime(config)
    contract = load_cache_contract(Path(runtime["provenance_path"]), config.resolved["model"]["paths"], config.resolved["pipeline_inputs"])
    if output_dir.name != contract["cache_version"]:
        raise ValueError("cache_version must name the output directory; semantic changes require a new version directory")
    encode_batch_size = runtime["encode_batch_size"]
    decode_num_workers = runtime["decode_num_workers"]
    torch.set_num_threads(1)
    cache_layout = _cache_layout_from_source_config(config)
    dataset = _build_source_dataset(
        config,
        video_decode_num_threads=runtime["video_decode_num_threads"],
    )
    total_rows = len(dataset.data)
    assigned_indices = list(range(shard_index, total_rows, num_shards))
    assigned_case_ids = [dataset.data[index]["_case_id"] for index in assigned_indices]
    shard_path = output_dir / "shards" / f"shard-{shard_index:05d}.json"
    final_manifest_path = output_dir / "manifest.json"
    latents_dir = output_dir / "latents"
    if final_manifest_path.exists():
        raise FileExistsError(f"latent cache is already merged: {final_manifest_path}")
    if shard_path.exists():
        records = _load_complete_shard_records(
            shard_path=shard_path,
            cache_layout=cache_layout,
            shard_index=shard_index,
            num_shards=num_shards,
            total_rows=total_rows,
            assigned_indices=assigned_indices,
            assigned_case_ids=assigned_case_ids,
            output_dir=output_dir,
            dataset=dataset,
            contract=contract,
        )
        return {
            "shard_index": shard_index,
            "num_shards": num_shards,
            "total_rows": total_rows,
            "shard_rows": len(records),
            "written": 0,
            "skipped_existing": len(records),
            "elapsed_seconds": 0.0,
            "timing": {
                "decode_service": 0.0,
                "decode_wait": 0.0,
                "encode_dtoh": 0.0,
                "write_service": 0.0,
                "write_wait": 0.0,
                "elapsed": 0.0,
                "encode_batch_size": encode_batch_size,
                "decode_num_workers": decode_num_workers,
                "model_load_seconds": None,
            },
            "vae_parameters": None,
            "cuda_peak_memory": None,
            "output_dir": str(output_dir),
            "manifest": str(shard_path),
        }
    skipped: dict[int, dict[str, Any]] = {}
    encode_indices: list[int] = []
    encode_case_ids: list[str] = []
    for index, case_id in zip(assigned_indices, assigned_case_ids, strict=True):
        path = latents_dir / f"{case_id}.safetensors"
        status, record = _existing_cache_record(
            path,
            cache_root=output_dir,
            case_id=case_id,
            index=index,
            cache_layout=cache_layout,
            expected_provenance=_expected_provenance(dataset, index, contract, cache_layout),
        )
        if status == "valid":
            assert record is not None
            skipped[index] = record
            continue
        encode_indices.append(index)
        encode_case_ids.append(case_id)
    latents_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model_load_seconds = vae_parameters = cuda_peak_memory = None
    if encode_indices:
        if not torch.cuda.is_available():
            raise RuntimeError("latent precompute requires a visible CUDA device")
        device = torch.device("cuda", torch.cuda.current_device())
        args = wan22_backend.configure_args(build_common_training_args(config), config, None)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        load_started = time.perf_counter()
        model = wan22_backend.build_cache_encoder(args, device, cache_layout=cache_layout)
        model.eval()
        torch.cuda.synchronize(device)
        model_load_seconds = _round_timing(time.perf_counter() - load_started)
        vae_parameters = _vae_parameter_summary(model.pipe.vae)
        encoded_records, timing = _precompute_assigned_records(
            model=model,
            dataset=dataset,
            assigned_indices=encode_indices,
            assigned_case_ids=encode_case_ids,
            latents_dir=latents_dir,
            encode_batch_size=encode_batch_size,
            decode_num_workers=decode_num_workers,
            prefetch_factor=runtime["prefetch_factor"],
            started=started,
            shard_index=shard_index,
            cache_layout=cache_layout,
            contract=contract,
        )
        torch.cuda.synchronize(device)
        cuda_peak_memory = _cuda_peak_memory(device)
        encoded_by_index = {record["index"]: record for record in encoded_records}
    else:
        encoded_by_index = {}
        timing = {
            "decode_service": 0.0,
            "decode_wait": 0.0,
            "encode_dtoh": 0.0,
            "write_service": 0.0,
            "write_wait": 0.0,
            "elapsed": _round_timing(time.perf_counter() - started),
            "encode_batch_size": encode_batch_size,
            "decode_num_workers": decode_num_workers,
        }
    # elapsed already includes model loading; the separate field is a breakdown.
    timing["model_load_seconds"] = model_load_seconds
    records = []
    for index, case_id in zip(assigned_indices, assigned_case_ids, strict=True):
        record = skipped.get(index, encoded_by_index.get(index))
        if record is None or record["case_id"] != case_id:
            raise ValueError(f"latent-cache shard is missing case_id={case_id} index={index}")
        records.append(record)
    if [record["index"] for record in records] != assigned_indices:
        raise ValueError("latent-cache shard record indices are incomplete")
    if [record["case_id"] for record in records] != assigned_case_ids:
        raise ValueError("latent-cache shard case IDs are incomplete")
    _write_shard_manifest(
        shard_path=shard_path,
        cache_layout=cache_layout,
        shard_index=shard_index,
        num_shards=num_shards,
        total_rows=total_rows,
        records=records,
    )
    return {
        "shard_index": shard_index,
        "num_shards": num_shards,
        "total_rows": total_rows,
        "shard_rows": len(records),
        "written": len(encode_indices),
        "skipped_existing": len(skipped),
        "elapsed_seconds": round(timing["elapsed"], 2),
        "timing": timing,
        "vae_parameters": vae_parameters,
        "cuda_peak_memory": cuda_peak_memory,
        "output_dir": str(output_dir),
        "manifest": str(shard_path),
    }


def _load_shard_manifests(output_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    shard_dir = output_dir / "shards"
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(shard_dir.glob("shard-*.json")):
        match = _SHARD_MANIFEST_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid shard manifest name: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"latent-cache shard manifest must be an object: {path}")
        payload = dict(payload)
        payload["_filename_shard_index"] = int(match.group("index"))
        manifests.append((path, payload))
    if not manifests:
        raise FileNotFoundError(f"no shard manifests under {shard_dir}")
    return manifests


def merge_latent_cache_manifest(*, output_dir: Path) -> dict[str, Any]:
    """Validate complete shard coverage and publish one structural manifest."""

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"latent cache is already merged: {manifest_path}")
    shards = _load_shard_manifests(output_dir)
    first = shards[0][1]
    cache_layout = cache_layout_from_shard_schema(first.get("schema"))
    shard_schema = _SHARD_SCHEMA_BY_LAYOUT[cache_layout]
    num_shards = first.get("num_shards")
    total_rows = first.get("total_rows")
    if type(num_shards) is not int or num_shards <= 0:
        raise ValueError("latent-cache num_shards is malformed")
    if type(total_rows) is not int or total_rows <= 0:
        raise ValueError("latent-cache total_rows is malformed")
    actual_shards = {payload["_filename_shard_index"] for _, payload in shards}
    expected_shards = set(range(num_shards))
    if actual_shards != expected_shards:
        raise ValueError(
            f"latent-cache shard set is incomplete: found {sorted(actual_shards)}, "
            f"expected {sorted(expected_shards)}"
        )

    records: list[dict[str, Any]] = []
    for shard_path, payload in shards:
        shard_index = payload.get("shard_index")
        if shard_index != payload["_filename_shard_index"]:
            raise ValueError(f"latent-cache shard filename/index mismatch: {shard_path}")
        if payload.get("schema") != shard_schema:
            raise ValueError(
                f"latent-cache shard schemas disagree: {shard_path} has "
                f"{payload.get('schema')!r}, expected {shard_schema!r}"
            )
        if payload.get("num_shards") != num_shards or payload.get("total_rows") != total_rows:
            raise ValueError(f"latent-cache shard dimensions disagree: {shard_path}")
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise ValueError(f"latent-cache shard records are malformed: {shard_path}")
        assigned = list(range(shard_index, total_rows, num_shards))
        if len(raw_records) != len(assigned):
            raise ValueError(f"latent-cache shard record count is incomplete: {shard_path}")
        for expected_index, raw_record in zip(assigned, raw_records, strict=True):
            record = _project_record(
                raw_record,
                path=shard_path,
                cache_layout=cache_layout,
            )
            if record["index"] != expected_index:
                raise ValueError(f"latent-cache shard record index mismatch: {shard_path}")
            _validate_cache_file(
                output_dir / record["cache_file"],
                record,
                cache_layout=cache_layout,
            )
            records.append(record)

    records.sort(key=lambda record: record["index"])
    if [record["index"] for record in records] != list(range(total_rows)):
        raise ValueError("latent-cache shard records do not cover every row")
    case_ids = [record["case_id"] for record in records]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("latent-cache shard records contain duplicate case_ids")
    manifest: dict[str, Any] = {
        "schema": _FINAL_SCHEMA_BY_LAYOUT[cache_layout],
        "shards": num_shards,
        "total_rows": total_rows,
        "records": records,
    }
    if cache_layout == WAN22_CACHE_LAYOUT_LATENTS_ONLY:
        manifest["cache_layout"] = cache_layout
    _atomic_write_json(manifest, manifest_path)
    return {
        "output_dir": str(output_dir),
        "shards": num_shards,
        "counts": {"written": total_rows},
        "total_rows": total_rows,
    }
