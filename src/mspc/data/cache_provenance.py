"""缓存的显式语义证据：媒体、实际采样、预处理与模型版本。"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

PROVENANCE_SCHEMA = "wan22_cache_provenance_v1"
SOURCE_COLLECTION_SCHEMA = "latent_cache_source_collection_v1"
_REFERENCE_ENCODING_KEYS = ("dtype", "pixel_conversion", "tiled", "tile_size", "tile_stride")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"cache provenance {label} requires explicit non-empty text")
    return value


def validate_models(models: Any) -> dict[str, Any]:
    if not isinstance(models, dict) or set(models) not in ({"vae"}, {"vae", "text_encoder", "tokenizer"}):
        raise ValueError("cache provenance models requires vae, plus text_encoder/tokenizer for embedded text")
    for name, item in models.items():
        if not isinstance(item, dict) or set(item) != {"path", "revision"}:
            raise ValueError(f"cache provenance models.{name} requires path and revision")
        _text(item["path"], f"models.{name}.path")
        _text(item["revision"], f"models.{name}.revision")
    return models


def load_cache_contract(path: Path, model_paths: Mapping[str, Any],
                        pipeline_inputs: Mapping[str, Any]) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(contract, dict) and contract.get("schema") == SOURCE_COLLECTION_SCHEMA:
        loaded = _load_source_collection(contract)
        for origin, source in contract["sources"].items():
            if os.path.realpath(source["models"]["vae"]["path"]) != os.path.realpath(model_paths["vae"]):
                raise ValueError(f"cache source {origin} VAE path differs from loaded model: {path}")
        expected = {"dtype": "bfloat16", "pixel_conversion": "uint8_to_bfloat16_mul_2_over_255_add_minus1",
                    "tiled": pipeline_inputs["tiled"],
                    **{key: list(pipeline_inputs[key]) for key in ("tile_size", "tile_stride")}}
        if loaded["encoding"] != expected:
            raise ValueError(f"cache source encoding differs from loaded pipeline: {path}")
        return loaded
    if not isinstance(contract, dict) or set(contract) != {
        "cache_version", "models", "prompt_template_revision", "implementation_revision"
    }:
        raise ValueError(f"cache provenance contract is malformed: {path}")
    _text(contract["cache_version"], "cache_version")
    _text(contract["prompt_template_revision"], "prompt_template_revision")
    validate_models(contract["models"])
    _text(contract["implementation_revision"], "implementation_revision")
    for name, key in (("vae", "vae"), ("text_encoder", "t5"), ("tokenizer", "tokenizer")):
        # 比真实文件而不是字符串：训练集群按 SHA 钉住的根经软链指向同一份权重，路径字符串与产出时不同。
        if name in contract["models"] and (os.path.realpath(contract["models"][name]["path"])
                                           != os.path.realpath(model_paths[key])):
            raise ValueError(f"cache provenance models.{name}.path differs from loaded model: {path}")
    contract["encoding"] = {"dtype": "bfloat16",
        "pixel_conversion": "uint8_to_bfloat16_mul_2_over_255_add_minus1",
        "implementation_revision": contract["implementation_revision"],
        "tiled": pipeline_inputs["tiled"],
        # Resolved configs use tuples; provenance is a JSON record at this boundary.
        **{key: list(pipeline_inputs[key]) for key in ("tile_size", "tile_stride")}}
    validate_encoding(contract["encoding"])
    return contract


def _load_source_collection(contract: Any) -> dict[str, Any]:
    """校验混合来源；共同消费字段不伪装成某次VAE缓存生产的版本。"""
    if (not isinstance(contract, dict) or set(contract) != {"schema", "index_path", "sources"}
            or contract["schema"] != SOURCE_COLLECTION_SCHEMA):
        raise ValueError("cache source collection is malformed")
    _text(contract["index_path"], "index_path")
    sources = contract["sources"]
    if not isinstance(sources, dict) or not sources:
        raise ValueError("cache source collection requires non-empty sources")
    common = None
    for origin, source in sources.items():
        _text(origin, "source origin")
        if not isinstance(source, dict) or set(source) != {
            "manifest_path", "selected_files", "cache_version", "models", "encoding", "context"
        }:
            raise ValueError(f"cache source {origin} is malformed")
        _text(source["manifest_path"], "manifest_path")
        _text(source["cache_version"], "cache_version")
        if type(source["selected_files"]) is not int or source["selected_files"] <= 0:
            raise ValueError(f"cache source {origin} selected_files must be positive")
        validate_models(source["models"])
        if set(source["models"]) != {"vae"} or source["context"] is not None:
            raise ValueError("cache source collection requires VAE-only unembedded targets")
        validate_encoding(source["encoding"])
        value = (source["models"]["vae"]["revision"],
                 {key: source["encoding"][key] for key in _REFERENCE_ENCODING_KEYS})
        if common is not None and value != common:
            raise ValueError("cache source VAE revision/encoding differs across the collection")
        common = value
    first = next(iter(sources.values()))
    # Every producer's version and implementation stay in sources. Only the
    # VAE/encoding fields actually consumed by online reference encoding merge.
    return {**contract, "models": first["models"], "encoding": common[1]}


def merge_selected_cache_indexes(source_indexes: Mapping[str, Path],
                                 selected_shots: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """按显式物理镜清单合并来源索引；只改origin，不写张量或来源header。"""
    from mspc.data.shot_targets import shot_key

    if not isinstance(source_indexes, Mapping) or not source_indexes:
        raise ValueError("cache index merge requires explicit nonempty source indexes")

    def key(row):
        value = shot_key(row)
        _text(value[0], "shot sid")
        _text(value[1], "shot scene_id")
        if type(value[2]) is not int or value[2] < 0:
            raise ValueError("cache index source_shot_index must be a nonnegative integer")
        return value

    selected = [key(row) for row in selected_shots]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("cache index merge requires nonempty unique selected shot keys")
    required, found, seen = set(selected), {}, set()
    for origin, path in source_indexes.items():
        _text(origin, "source origin")
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                current = key(row)
                _text(row["origin"], "input source origin")
                if current in seen:
                    raise ValueError(f"duplicate cache target key across source indexes: {current}")
                seen.add(current)
                if current in required:
                    found[current] = {**row, "origin": origin}
    missing = required - found.keys()
    if missing:
        raise ValueError(f"{len(missing)} selected cache shots are absent from the explicit source indexes")
    return [found[current] for current in selected]


def build_cache_source_collection(index_path: Path, target_roots: Mapping[str, Path]) -> dict[str, Any]:
    """从实际索引选中的manifest记录发布来源集合；不打开张量、不补写生产身份。"""
    selected: dict[str, set[str]] = {}
    keys = set()
    with index_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["sid"], row["scene_id"], row["source_shot_index"])
            if key in keys:
                raise ValueError(f"duplicate cache target key: {key}")
            keys.add(key)
            selected.setdefault(row["origin"], set()).add(row["file"])
    if not selected or set(selected) != set(target_roots):
        raise ValueError("target_roots must exactly cover selected cache origins")
    sources = {}
    for origin, files in selected.items():
        manifest_path = Path(target_roots[origin]) / "manifest.json"
        matched, source = set(), None
        for record in json.loads(manifest_path.read_text(encoding="utf-8"))["records"]:
            name = record["cache_file"]
            if name not in files:
                continue
            if name in matched:
                raise ValueError(f"duplicate cache source record: {origin}/{name}")
            matched.add(name)
            provenance = validate_provenance(record["provenance"])
            value = {key: provenance[key] for key in ("cache_version", "models", "encoding", "context")}
            if source is not None and source != value:
                raise ValueError(f"cache source provenance differs within origin: {origin}")
            source = value
        if matched != files:
            raise ValueError(f"{origin}: {len(files - matched)} selected cache files are missing from manifest")
        sources[origin] = {"manifest_path": str(manifest_path.resolve(strict=True)),
                           "selected_files": len(files), **source}
    contract = {"schema": SOURCE_COLLECTION_SCHEMA, "index_path": str(index_path.resolve(strict=True)),
                "sources": sources}
    _load_source_collection(contract)
    return contract


def validate_encoding(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("cache provenance encoding is missing")
    for key in ("dtype", "pixel_conversion", "implementation_revision"):
        _text(value.get(key), f"encoding.{key}")
    if type(value.get("tiled")) is not bool:
        raise ValueError("cache provenance encoding.tiled must be explicit bool")
    for key in ("tile_size", "tile_stride"):
        if not isinstance(value.get(key), list) or len(value[key]) != 2 or any(type(n) is not int or n <= 0 for n in value[key]):
            raise ValueError(f"cache provenance encoding.{key} must be two positive integers")


def describe_media_provenance(*, paths: list[str], metadata: list[dict[str, Any]],
                              frame_plan: dict[str, Any], source_shot_segments: Any,
                              schema: str, target_height: int, target_width: int,
                              scene_row: dict[str, Any] | None = None) -> dict[str, Any]:
    from mspc.data.multishot import cover_crop_geometry

    if scene_row is not None:
        from mspc.data.scene_cache import describe_scene_media

        return describe_scene_media(scene_row, frame_plan, media={
            "height": target_height, "width": target_width,
            **{key: frame_plan[key] for key in ("target_fps", "vae_temporal_stride", "max_rgb_frames")}})
    files, spatial = [], []
    for path, meta in zip(paths, metadata, strict=True):
        stat = Path(path).stat()
        files.append({"path": path, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, **meta})
        geometry = cover_crop_geometry(video_width=meta["source_width"],
            video_height=meta["source_height"], target_width=target_width, target_height=target_height)
        spatial.append({**geometry, "source_to_target_matrix": geometry["matrix"]})
    return {"schema": schema, "files": files, "source_shot_segments": source_shot_segments,
        "sampling": frame_plan, "preprocessing": {"decoder": "eva-decord",
            "decoder_version": importlib.metadata.version("eva-decord"),
            "torchvision_version": importlib.metadata.version("torchvision"),
            "spatial": spatial, "pixel_conversion": "uint8_RGB_to_float_minus1_plus1"}}


def validate_provenance(value: Any, *, cache_version: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError("cache provenance is missing or unsupported; use a new version directory")
    version = _text(value.get("cache_version"), "cache_version")
    if cache_version is not None and version != cache_version:
        raise ValueError("cache provenance cache_version differs; use a new version directory")
    validate_models(value.get("models"))
    validate_encoding(value.get("encoding"))
    media = value.get("media")
    if not isinstance(media, dict) or not isinstance(media.get("files"), list) or not media["files"]:
        raise ValueError("cache provenance media files are missing")
    _text(media.get("schema"), "media.schema")
    for f in media["files"]:
        _text(f.get("path"), "media.path")
        for key in ("bytes", "mtime_ns", "source_num_frames", "source_height", "source_width"):
            if type(f.get(key)) is not int or f[key] <= 0:
                raise ValueError(f"cache provenance media.{key} is missing or invalid")
        if type(f.get("source_fps")) not in (int, float) or not math.isfinite(f["source_fps"]) or f["source_fps"] <= 0:
            raise ValueError("cache provenance source_fps is missing or invalid")
    if not media.get("source_shot_segments"):
        raise ValueError("cache provenance source_shot_segments is missing")
    sampling = media.get("sampling")
    if not isinstance(sampling, dict) or not {"target_fps", "shot_groups", "latent_shot_groups"} <= sampling.keys():
        raise ValueError("cache provenance sampling is missing")
    if not (sampling.get("frame_indices") or sampling.get("frame_indices_by_shot")):
        raise ValueError("cache provenance sampling frame mapping is missing")
    preprocessing = media.get("preprocessing")
    if not isinstance(preprocessing, dict) or not preprocessing.get("spatial"):
        raise ValueError("cache provenance preprocessing is missing")
    for key in ("decoder", "decoder_version", "torchvision_version", "pixel_conversion"):
        _text(preprocessing.get(key), f"preprocessing.{key}")
    for spatial in preprocessing["spatial"]:
        if not isinstance(spatial.get("source_to_target_matrix"), list) or len(spatial["source_to_target_matrix"]) != 3:
            raise ValueError("cache provenance crop/resize matrix is missing")
    context = value.get("context")
    if context is not None:
        if set(value["models"]) != {"vae", "text_encoder", "tokenizer"}:
            raise ValueError("embedded context requires text encoder and tokenizer provenance")
        if not isinstance(context, dict) or not isinstance(context.get("prompts"), list) or not context["prompts"]:
            raise ValueError("cache provenance context prompts are missing")
        _text(context.get("prompt_template_revision"), "context.prompt_template_revision")
        if context.get("max_tokens") != 512:
            raise ValueError("cache provenance context max_tokens must be 512")
    return value


def require_matching_provenance(stored: Any, expected: Any) -> None:
    validate_provenance(stored)
    validate_provenance(expected)
    if stored != expected:
        raise ValueError("cache provenance differs from current source/sampling/preprocessing/models; use a new version directory")
