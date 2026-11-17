"""Data-domain command registration without owning the root parser."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


DATA_COMMANDS = frozenset(
    {
        "data-precompute-latents",
        "data-merge-latent-cache",
    }
)


def register_data_commands(commands: argparse._SubParsersAction[Any]) -> None:
    precompute_latents = commands.add_parser(
        "data-precompute-latents",
        help="write one fresh shard of VAE latents, optionally with T5 context",
    )
    precompute_latents.add_argument("--config", required=True, type=Path, metavar="EXACT_PATH")
    precompute_latents.add_argument("--output-dir", required=True, type=Path, metavar="EXACT_PATH")
    precompute_latents.add_argument("--shard-index", required=True, type=int)
    precompute_latents.add_argument("--num-shards", required=True, type=int)

    merge_latent_cache = commands.add_parser(
        "data-merge-latent-cache",
        help="merge complete latent-cache shard manifests into manifest.json",
    )
    merge_latent_cache.add_argument("--output-dir", required=True, type=Path, metavar="EXACT_PATH")


def run_data_command(args: argparse.Namespace) -> dict[str, Any]:
    command = args.command
    if command == "data-precompute-latents":
        from mspc.training.latent_cache import precompute_latent_cache

        return precompute_latent_cache(
            config_path=args.config,
            output_dir=args.output_dir,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
    if command == "data-merge-latent-cache":
        from mspc.training.latent_cache import merge_latent_cache_manifest

        return merge_latent_cache_manifest(output_dir=args.output_dir)
    raise ValueError(f"unsupported data command: {command}")
