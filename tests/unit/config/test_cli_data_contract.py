from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mspc.cli import build_parser, main


def test_precompute_latents_uses_one_internal_resource_policy(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    import mspc.training.latent_cache as latent_cache

    captured: dict[str, Any] = {}

    def fake_precompute_latent_cache(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(latent_cache, "precompute_latent_cache", fake_precompute_latent_cache)
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "cache"

    assert (
        main(
            [
                "data-precompute-latents",
                "--config",
                str(config),
                "--output-dir",
                str(output_dir),
                "--shard-index",
                "2",
                "--num-shards",
                "40",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"ok": True}
    assert captured == {
        "config_path": config,
        "output_dir": output_dir,
        "shard_index": 2,
        "num_shards": 40,
    }


def test_precompute_latents_rejects_cli_encode_batch_size(tmp_path: Path) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "data-precompute-latents",
                "--config",
                str(tmp_path / "config.json"),
                "--output-dir",
                str(tmp_path / "cache"),
                "--shard-index",
                "0",
                "--num-shards",
                "1",
                "--encode-batch-size",
                "2",
            ]
        )


def test_latent_cache_cli_has_no_collision_policy(monkeypatch, tmp_path: Path, capsys) -> None:
    import mspc.training.latent_cache as latent_cache

    captured: dict[str, Any] = {}

    def fake_merge_latent_cache_manifest(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"sealed": True}

    monkeypatch.setattr(
        latent_cache,
        "merge_latent_cache_manifest",
        fake_merge_latent_cache_manifest,
    )
    output_dir = tmp_path / "cache"

    assert main(["data-merge-latent-cache", "--output-dir", str(output_dir)]) == 0
    assert json.loads(capsys.readouterr().out) == {"sealed": True}
    assert captured == {"output_dir": output_dir}

    parser = build_parser()
    for argv in (
        [
            "data-precompute-latents",
            "--config",
            str(tmp_path / "config.json"),
            "--output-dir",
            str(output_dir),
            "--shard-index",
            "0",
            "--num-shards",
            "1",
            "--collision-policy",
            "error",
        ],
        [
            "data-merge-latent-cache",
            "--output-dir",
            str(output_dir),
            "--collision-policy",
            "overwrite",
        ],
    ):
        try:
            parser.parse_args(argv)
        except SystemExit as exc:
            assert exc.code == 2
        else:  # pragma: no cover - the retired policy switch must stay absent.
            raise AssertionError(f"retired collision policy was accepted: {argv!r}")
