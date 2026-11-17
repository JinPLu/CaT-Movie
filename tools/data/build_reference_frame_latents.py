"""每条训练故事的参考帧库（GPU）。模块：``mspc.data.reference_frame_latents``。

按 manifest 下标 ``idx % num_shards == shard_index`` 分片，逐条写 ``<out>/latents/<case_id>.safetensors``；
已存在的核对来源后跳过（续跑），每 500 条打一行进度。``--clip-model`` 是 StoryMem 关键帧去重用的同一份
CLIP 权重目录（transformers ``CLIPModel``）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mspc.data import reference_frame_latents as rfl  # noqa: E402


def _records(args: argparse.Namespace) -> list[dict]:
    from mspc.config import load_experiment_config
    from mspc.data.cache_provenance import load_cache_contract, validate_provenance

    config = load_experiment_config(args.source_config).resolved
    contract = load_cache_contract(Path(config["precompute"]["provenance_path"]),
        config["model"]["paths"], config["pipeline_inputs"])
    records = json.loads((args.latent_cache_dir / "manifest.json").read_text(encoding="utf-8"))["records"]
    for record in records:
        source = validate_provenance(record.get("provenance"), cache_version=contract["cache_version"])
        if source["models"]["vae"] != contract["models"]["vae"] or source["encoding"] != contract["encoding"]:
            raise ValueError(f"{record['case_id']}: reference-frame VAE/config differs from source cache")
    return records


def _runtime(args: argparse.Namespace):
    import torch

    from mspc.config import load_experiment_config
    from mspc.models.wan22_loader import load_vae_only_runtime

    # diffsynth 在 third_party/wan22_runtime，与出片同一来源：取这份 config 的 infer_pythonpath。
    sys.path[:0] = load_experiment_config(args.source_config).resolved["launch"]["infer_pythonpath"]
    config = load_experiment_config(args.source_config).resolved
    return load_vae_only_runtime(config["model"], config["pipeline_inputs"], torch.device(args.device))


def run_shard(args: argparse.Namespace) -> dict:
    from safetensors import safe_open

    records = [record for index, record in enumerate(_records(args)) if index % args.num_shards == args.shard_index]
    runtime = _runtime(args)
    embed = rfl.ClipImageEmbedder(args.clip_model, args.device)
    written = skipped = frames = 0
    started = time.monotonic()
    for record in records:
        target = args.out_dir / "latents" / f"{record['case_id']}.safetensors"
        if target.exists():
            with safe_open(str(target), framework="pt") as handle:
                rfl.validate_reference_source(handle.metadata() or {}, record)
            skipped += 1
            continue
        latents = rfl.read_input_latents(args.latent_cache_dir, record)
        library = rfl.case_reference_frames(runtime, embed, latents, record)
        rfl.write_case(args.out_dir, record, library, clip_model=str(args.clip_model))
        written += 1
        frames += len(library.frames)
        if written % 500 == 0:
            rate = written / (time.monotonic() - started)
            print(json.dumps({"shard": args.shard_index, "written": written, "skipped": skipped, "of": len(records),
                              "stories_per_second": round(rate, 3)}), flush=True)
    return {"shard": args.shard_index, "records": len(records), "written": written, "skipped": skipped,
            "library_frames_written": frames, "seconds": round(time.monotonic() - started, 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--latent-cache-dir", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True, help="生产这份缓存的 cache-source 配置（VAE 与分块取自它）")
    parser.add_argument("--clip-model", type=Path, required=True, help="StoryMem 关键帧去重用的 CLIP 权重目录")
    parser.add_argument("--device", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    print(json.dumps(run_shard(parser.parse_args()), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
