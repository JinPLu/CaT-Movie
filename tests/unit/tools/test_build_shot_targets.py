"""按镜目标潜变量入口的清单、索引、对拍与分片启动；块与首批的行是主服务器上的真实行（见 data 夹具）。"""
from copy import deepcopy
import csv
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from mspc.data import shot_targets as st
from mspc.data.scene_training_manifest import MEDIA_COLUMNS
from tools.data import build_shot_targets as tool

tool.st = st
REAL = json.loads((Path(__file__).parents[1] / "data/fixtures/shot_targets_real.json").read_text())["cases"]
CACHED = [case for case in REAL if case["first_batch"]]
TO_BUILD = [case for case in REAL if not case["first_batch"]]


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def site(tmp_path):
    """真实的 5 个块、3 个首批窗的映射行、清单 CSV 行与缓存清单记录，摆成服务器上的文件形状。"""
    blocks, mapping, cache = tmp_path / "blocks.jsonl", tmp_path / "mapping.jsonl", tmp_path / "first"
    blocks.write_text("".join(json.dumps(case["block"], ensure_ascii=False) + "\n" for case in REAL))
    mapping.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                               for case in CACHED for row in case["first_batch"]["mapping_rows"]))
    with (tmp_path / "first_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CACHED[0]["first_batch"]["manifest_row"]))
        writer.writeheader()
        writer.writerows(case["first_batch"]["manifest_row"] for case in CACHED)
    cache.mkdir()
    vae = tmp_path / "OoVMetric/models_ckpt/wan22_5b_native/Wan2.2_VAE.pth"
    vae.parent.mkdir(parents=True)
    vae.write_bytes(b"metadata-only model path binding fixture")
    records = deepcopy([case["first_batch"]["record"] for case in CACHED])
    for record in records:
        record["provenance"]["models"]["vae"]["path"] = str(vae)
    (cache / "manifest.json").write_text(json.dumps({"records": records}))
    return SimpleNamespace(root=tmp_path, blocks=blocks, mapping=mapping, first_manifest=tmp_path / "first_manifest.csv",
                           first_cache=cache, vae=vae, vae_path=vae)


def build_manifest(site, limit=0, *, reuse_existing=False, stage_with_validation=False, only_existing=False):
    out = site.root / "inputs"
    cached = {(row["sid"], row["scene_id"], row["source_shot_index"])
              for row in lines(site.mapping)}
    members, samples = [], []
    for block in lines(site.blocks):
        shots = [{**shot, "frames": shot["render_frames"], "usable": True} for shot in block["scene"]["shots"]]
        members.append({**block, "case_id": block["key"], "shots": shots})
        for position, shot in enumerate(shots):
            if ((reuse_existing or st.shot_key(shot) not in cached)
                    and (not only_existing or st.shot_key(shot) in cached)):
                samples.append({"sample_index": len(samples), "case_id": block["key"], "span": [position, position + 1]})
    members_path, sampling_path = site.root / "members.jsonl", site.root / "sampling.jsonl"
    members_path.write_text("".join(json.dumps(row) + "\n" for row in members))
    sampling_path.write_text("".join(json.dumps(row) + "\n" for row in samples))
    config = json.loads((Path(__file__).parents[2] / "fixtures/block_renderer_config.json").read_text())
    extra = {}
    config["model"]["paths"]["vae"] = str(site.vae_path)
    if stage_with_validation:
        validation = samples.pop()
        sampling_path.write_text("".join(json.dumps(row) + "\n" for row in samples))
        val_sampling = site.root / "val-sampling.jsonl"
        val_sampling.write_text(json.dumps({**validation, "sample_index": 0}) + "\n")
        config["training"]["validation"] = {"members_path": str(members_path),
            "sampling_manifest_path": str(val_sampling), "sample_budget": 1, "every_steps": 1, "draws": 1,
            "planner_tokens": None, "reference_entities": None}
        stage_samples = [samples[0], samples[-1]]
        by_case = {member["case_id"]: member for member in members}
        stage_keys = [list(st.shot_key(by_case[sample["case_id"]]["shots"][sample["span"][0]])) for sample in stage_samples]
        stage = {"artifact_type": "mspc_training_cache_stage", "schema_version": 1,
                 "media": {key: config["task"][key] for key in
                           ("height", "width", "target_fps", "vae_temporal_stride", "max_rgb_frames")},
                 "training_seed": config["training"]["seed"],
                 "latent_geometry": {"channels": 48, "spatial_shape": [30, 52], "vae_spatial_stride": 16},
                 "planned_optimizer_updates": config["training"]["max_train_steps"],
                 "sample_budget": config["training"]["max_train_steps"] * config["training"]["gradient_accumulation_steps"],
                 "loader": {"world_size": 1}, "target_step": 1,
                 "sample_indices": [0, len(samples) - 1], "target_keys": stage_keys,
                 "validation_sample_indices": [0]}
        stage_path = site.root / "stage.json"
        stage_path.write_text(json.dumps(stage))
        extra["cache_stage"] = stage_path
    if reuse_existing:
        _, existing_index = build_first_index(site)
        for record in json.loads((site.first_cache / "manifest.json").read_text())["records"]:
            path = site.first_cache / record["cache_file"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"existing tensor is never rewritten")
        extra.update(existing_index=[f"first_batch={existing_index}"], existing_root=[f"first_batch={site.first_cache}"])
    config["dataset"].update(members_path=str(members_path), sampling_manifest_path=str(sampling_path),
                             target_roots={st.SHOT: str(site.root / "shot_targets_v1")})
    config_path = site.root / "training-config.json"
    config_path.write_text(json.dumps(config))
    tool.manifest(SimpleNamespace(config=config_path, latent_channels=48, latent_height=30, latent_width=52,
        vae_revision=CACHED[0]["first_batch"]["record"]["provenance"]["models"]["vae"]["revision"],
        implementation_revision="test_actual_sampled_manifest", limit=limit, out_dir=out, **extra))
    return out


def build_first_index(site, limit=0):
    code = tool.index_first_batch(SimpleNamespace(
        blocks=site.blocks, first_batch_mapping=site.mapping, first_batch_scene_manifest=site.first_manifest,
        first_batch_cache_dir=site.first_cache, limit=limit, out_dir=site.root))
    return code, site.root / "index.first-batch.jsonl"


def test_manifest_lists_every_block_shot_without_first_batch_latents(site):
    out = build_manifest(site)
    shots = lines(out / "shots.jsonl")
    block_shots = [shot for case in REAL for shot in case["block"]["scene"]["shots"]]
    cached = {(row["sid"], row["scene_id"], row["source_shot_index"])
              for case in CACHED for row in case["first_batch"]["mapping_rows"]}
    assert {st.shot_key(row) for row in shots} == {st.shot_key(shot) for shot in block_shots} - cached
    summary = json.loads((out / "summary.json").read_text())
    assert summary["by_split"]["train"]["shots"] == len(block_shots) - len(cached)
    assert summary["to_build"]["shots"] == summary["written_rows"] == len(shots) == len(block_shots) - len(cached)
    with (out / "scene_manifest.csv").open(newline="") as handle:
        rows = {row["case_id"]: row for row in csv.DictReader(handle)}
    assert list(rows) == [row["case_id"] for row in shots]
    for case in TO_BUILD:
        actual = {key: value for key, value in rows[case["case_id"]].items() if key not in MEDIA_COLUMNS.values()}
        assert actual == case["manifest_row"]
        assert {key: rows[case["case_id"]][key] for key in MEDIA_COLUMNS.values()} == {
            "target_height": "480", "target_width": "832", "target_fps": "16.0",
            "vae_temporal_stride": "4", "max_rgb_frames": "308"}


def test_manifest_limit_reports_the_actual_sampled_prefix_in_the_summary(site):
    out = build_manifest(site, limit=1)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["written_rows"] == 1 == len(lines(out / "shots.jsonl"))
    assert summary["sampling_limit_shots"] == summary["to_build"]["shots"] == 1
    assert summary["by_split"]["train"]["shots"] == 1


def test_manifest_reuses_actual_existing_sources_and_only_writes_missing_shots(site):
    old_manifest = (site.first_cache / "manifest.json").read_bytes()
    out = build_manifest(site, reuse_existing=True)
    required, reused, missing = [lines(out / name) for name in
                               ("required-shots.jsonl", "reused-shots.jsonl", "shots.jsonl")]
    assert {st.shot_key(row) for row in required} == {
        st.shot_key(shot) for case in REAL for shot in case["block"]["scene"]["shots"]}
    assert {st.shot_key(row) for row in reused} == {st.shot_key(row) for row in lines(site.mapping)}
    assert {row["origin"] for row in reused} == {"first_batch"}
    assert {st.shot_key(row) for row in missing} == {st.shot_key(row) for row in required} - {
        st.shot_key(row) for row in reused}
    with (out / "scene_manifest.csv").open(newline="") as handle:
        assert [row["case_id"] for row in csv.DictReader(handle)] == [row["case_id"] for row in missing]
    summary = json.loads((out / "summary.json").read_text())
    assert (summary["required_shots"], summary["reused_shots"], summary["written_rows"]) == (
        len(required), len(reused), len(missing))
    assert (site.first_cache / "manifest.json").read_bytes() == old_manifest
    for record in json.loads(old_manifest)["records"]:
        assert (site.first_cache / record["cache_file"]).read_bytes() == b"existing tensor is never rewritten"
    assert json.loads((out / "cache_provenance.json").read_text())["implementation_revision"] == "test_actual_sampled_manifest"



def test_manifest_canonicalizes_the_real_vae_behind_a_frozen_worktree_alias(site):
    # 11-08 的真实关系：执行根的 OoVMetric 软链与旧 header 的资产路径是同一文件。
    worktree = site.root / "Storyboard-runs/d5f02abf32da560724acb7cd1cff580a341b35d3-root"
    worktree.mkdir(parents=True)
    (worktree / "OoVMetric").symlink_to(site.root / "OoVMetric", target_is_directory=True)
    site.vae_path = worktree / "OoVMetric/models_ckpt/wan22_5b_native/Wan2.2_VAE.pth"
    assert site.vae_path.samefile(site.vae)
    original = (site.first_cache / "manifest.json").read_bytes()
    out = build_manifest(site, reuse_existing=True)
    contract = json.loads((out / "cache_provenance.json").read_text())
    old_models = json.loads(original)["records"][0]["provenance"]["models"]
    assert contract["models"] == old_models
    assert contract["models"]["vae"]["path"] == str(site.vae.resolve(strict=True))
    assert json.loads((site.root / "training-config.json").read_text())["model"]["paths"]["vae"] == str(site.vae_path)
    assert (site.first_cache / "manifest.json").read_bytes() == original
    for record in json.loads(original)["records"]:
        assert (site.first_cache / record["cache_file"]).read_bytes() == b"existing tensor is never rewritten"


def test_stage_reuse_checks_complete_train_keys_before_filter_and_keeps_full_validation(site):
    out = build_manifest(site, reuse_existing=True, stage_with_validation=True)
    required, reused, missing = [lines(out / name) for name in
                               ("required-shots.jsonl", "reused-shots.jsonl", "shots.jsonl")]
    stage = json.loads((site.root / "stage.json").read_text())
    assert {st.shot_key(row) for row in required if row["split"] == "train"} == {
        tuple(key) for key in stage["target_keys"]}
    assert len([row for row in required if row["split"] == "val"]) == 1
    assert reused and missing
    assert len(required) == len(reused) + len(missing)


def test_all_cached_targets_publish_zero_backfill_without_a_fake_csv(site):
    out = build_manifest(site, reuse_existing=True, only_existing=True)
    assert lines(out / "shots.jsonl") == []
    assert [{**row, "origin": "first_batch"} for row in lines(out / "required-shots.jsonl")] == lines(
        out / "reused-shots.jsonl")
    assert not (out / "scene_manifest.csv").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["written_rows"] == 0
    assert summary["to_build"] == {"shots": 0, "train_frames": 0, "latent_frames": 0}


@pytest.mark.parametrize("damage", ["frame_indices", "crop", "vae", "encoding"])
def test_manifest_refuses_existing_cache_with_a_different_source_or_encoding(site, damage):
    manifest_path = site.first_cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    provenance = manifest["records"][0]["provenance"]
    if damage == "frame_indices":
        provenance["media"]["sampling"]["frame_indices_by_shot"][0][0] += 1
    elif damage == "crop":
        provenance["media"]["preprocessing"]["spatial"][0]["train_crop_box"][0] += 2
    elif damage == "vae":
        provenance["models"]["vae"]["revision"] += "-different"
    else:
        provenance["encoding"]["pixel_conversion"] = "different-pixel-conversion"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source/geometry mismatch|VAE/encoding differs"):
        build_manifest(site, reuse_existing=True)


def test_merge_indexes_cli_uses_explicit_origins_and_preserves_original_rows(site):
    _, existing_index = build_first_index(site)
    rows = lines(existing_index)
    selected = site.root / "selected.jsonl"
    selected.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)))
    output = site.root / "merged-index.jsonl"
    original = existing_index.read_bytes()
    args = SimpleNamespace(source_index=[f"existing_shot={existing_index}"], selected_shots=selected, output=output)
    tool.merge_indexes(args)
    assert lines(output) == [{**row, "origin": "existing_shot"} for row in reversed(rows)]
    assert existing_index.read_bytes() == original
    tool.merge_indexes(args)
    selected.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(FileExistsError, match="published merged cache index differs"):
        tool.merge_indexes(args)


def test_first_batch_index_refuses_a_shot_missing_from_the_blocks(site):
    kept = [case["block"] for case in REAL if case["name"] != "cinedance_later_in_window"]
    site.blocks.write_text("".join(json.dumps(block, ensure_ascii=False) + "\n" for block in kept))
    with pytest.raises(ValueError, match="not in the block manifest"):
        build_first_index(site)


def test_first_batch_index_reconciles_with_the_blocks(site):
    code, path = build_first_index(site)
    assert code == 0
    rows = {st.shot_key(row): row for row in lines(path)}
    assert len(rows) == sum(len(case["first_batch"]["mapping_rows"]) for case in CACHED)
    for case in CACHED:
        assert rows[st.shot_key(case["block"]["scene"]["shots"][case["position"]])] == case["first_batch"]["index_row"]
    assert json.loads((site.root / "index.first-batch.summary.json").read_text())["mismatches"] == {}


def test_first_batch_index_counts_a_block_that_disagrees(site):
    case = next(case for case in CACHED if case["name"] == "cinepile_first_in_window_slash_scene_id")
    blocks = [json.loads(json.dumps(item["block"])) for item in REAL]
    for block in blocks:
        if block["key"] == case["block"]["key"]:
            block["scene"]["shots"][case["position"]]["render_frames"] += 4
    site.blocks.write_text("".join(json.dumps(block, ensure_ascii=False) + "\n" for block in blocks))
    code, _ = build_first_index(site)
    summary = json.loads((site.root / "index.first-batch.summary.json").read_text())
    assert code == 1 and summary["mismatches"] == {"train_frames": 1, "latent_frames": 1}
    assert summary["examples"]["train_frames"][0]["shot"] == list(st.shot_key(case["block"]["scene"]["shots"][case["position"]]))


def crosscheck_inputs(site, count):
    _, first_index = build_first_index(site)
    out = site.root / "crosscheck"
    code = tool.crosscheck_prepare(SimpleNamespace(
        blocks=site.blocks, first_batch_mapping=site.mapping, first_batch_scene_manifest=site.first_manifest,
        first_batch_index=first_index, count=count, out_dir=out))
    return code, out


def test_crosscheck_prepare_compares_the_picked_shots_with_the_first_batch_manifest(site):
    code, out = crosscheck_inputs(site, 5)
    report = json.loads((out / "manifest-check.json").read_text())
    assert code == 0 and report["differing_cells"] == {} and report["shots"] == 5
    assert set(report["by_stratum"]) == {"cinedance/first", "cinedance/later", "cinepile_t2v/first", "cinepile_t2v/later",
                                         "moviechat_1k/alone"}
    shots = lines(out / "shots.jsonl")
    with (out / "scene_manifest.csv").open(newline="") as handle:
        assert [row["case_id"] for row in csv.DictReader(handle)] == [shot["case_id"] for shot in shots]
    with (out / "window_manifest.csv").open(newline="") as handle:
        assert {row["case_id"] for row in csv.DictReader(handle)} == {shot["window_case_id"] for shot in shots}
    for shot in shots:
        start, end = shot["window_frames"]
        assert end - start == shot["train_frames"] and shot["latent_end"] - shot["latent_start"] == st.latent_frames(end - start)


def test_crosscheck_latents_passes_identical_and_one_step_and_fails_beyond(site):
    _, inputs = crosscheck_inputs(site, 5)
    shots = lines(inputs / "shots.jsonl")
    new_cache = site.root / "shot_targets_crosscheck_v1"
    torch.manual_seed(0)
    windows = {}
    for shot in shots:
        if shot["first_batch_file"] not in windows:
            record = next(case["first_batch"]["record"] for case in CACHED
                          if case["first_batch"]["record"]["cache_file"] == shot["first_batch_file"])
            windows[shot["first_batch_file"]] = torch.randn(1, 2, record["input_latents_shape"][2], 3, 4).to(torch.bfloat16)
    for name, tensor in windows.items():
        path = site.first_cache / name
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"input_latents": tensor}, str(path))

    def write(shot, bump):
        part = windows[shot["first_batch_file"]][:, :, shot["latent_start"]:shot["latent_end"]].clone()
        part.view(torch.int16)[0, 0, 0, 0, 0] += bump  # 位型加一即相邻的 BF16 数（同号内）
        path = new_cache / "latents" / f"{shot['case_id']}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"input_latents": part}, str(path))

    def check():
        return tool.crosscheck_latents(SimpleNamespace(inputs=inputs, first_batch_cache_dir=site.first_cache,
                                                        shot_cache_dir=new_cache, out_dir=site.root))

    for number, shot in enumerate(shots):
        write(shot, 1 if number == 0 else 0)
    assert check() == 0
    report = json.loads((site.root / "latent-check.json").read_text())
    assert (report["identical"], report["within_one_step"], report["failed"], report["verdict"]) == (4, 1, [], "pass")
    write(shots[1], 2)
    assert check() == 1
    report = json.loads((site.root / "latent-check.json").read_text())
    assert report["failed"] == [shots[1]["case_id"]] and report["max_steps"] == 2 and report["verdict"] == "fail"


def shard_record(row):
    frames = json.loads(row["shot_groups"])[0][1]
    return {"case_id": row["case_id"], "cache_file": f"latents/{row['case_id']}.safetensors", "shot_groups": None,
            "latent_shot_groups": None, "num_frames": frames, "input_latents_shape": [1, 48, st.latent_frames(frames), 30, 52]}


def test_index_covers_every_sampled_shot_or_refuses(site):
    inputs = build_manifest(site)
    with (inputs / "scene_manifest.csv").open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    source_contract = json.loads((inputs / "cache_provenance.json").read_text())
    source_summary = json.loads((inputs / "summary.json").read_text())
    records = []
    for row in source_rows:
        record = shard_record(row)
        provenance = json.loads(json.dumps(CACHED[0]["first_batch"]["record"]["provenance"]))
        provenance.update(cache_version=source_contract["cache_version"], models=source_contract["models"],
                          encoding=source_summary["encoding"])
        evidence = provenance["media"]
        evidence["files"][0].update(path=row["video"], source_fps=float(row["source_fps"]),
            **{key: int(row[key]) for key in ("source_num_frames", "source_height", "source_width")})
        evidence["source_shot_segments"] = json.loads(row["source_shot_segments"])
        evidence["sampling"].update(target_fps=float(row["target_fps"]),
            frame_indices_by_shot=json.loads(row["source_frame_indices_by_shot"]),
            shot_groups=json.loads(row["shot_groups"]),
            latent_shot_groups=[[0, record["input_latents_shape"][2]]])
        evidence["preprocessing"]["spatial"][0].update(
            train_crop_box=json.loads(row["train_crop_box"]), output_size=[832, 480])
        record["provenance"] = provenance
        records.append(record)
    cache = site.root / "shot_targets_v1"
    cache.mkdir()
    args = SimpleNamespace(inputs=inputs, shot_cache_dir=cache, out_dir=site.root)
    (cache / "manifest.json").write_text(json.dumps({"records": records[:-1]}))
    with pytest.raises(ValueError, match="1 sampled target shots have no cache record"):
        tool.index(args)
    (cache / "manifest.json").write_text(json.dumps({"records": records}))
    tool.index(args)
    rows = lines(site.root / "index.jsonl")
    expected = lines(inputs / "shots.jsonl")
    assert [st.shot_key(row) for row in rows] == [st.shot_key(shot) for shot in expected]
    assert [row["train_frames"] for row in rows] == [shot["train_frames"] for shot in expected]
    summary = json.loads((site.root / "index.summary.json").read_text())
    assert summary["by_origin"] == {st.SHOT: len(records)}
    assert summary["roots"] == {st.SHOT: str(cache)}
    for row in rows:
        assert (row["latent_start"], row["latent_end"]) == (0, st.latent_frames(row["train_frames"]))
        assert row["file"] == f"latents/{st.case_id(row)}.safetensors"


ENTRY = """#!/usr/bin/env bash
# 现有入口的替身：只回显启动器交给它的环境与参数，按第一个参数里的文件决定成功或失败。
echo "[precompute] rank=$RANK world_size=$WORLD_SIZE nproc_per_node=$NPROC_PER_NODE devices=$CUDA_VISIBLE_DEVICES root=$MSPC_SHARED_ROOT args=$*"
echo '[precompute-row] {"index": 0, "case_id": "a", "size_bytes": 100}'
echo '[precompute-row] {"index": 4, "case_id": "b", "size_bytes": 50}'
echo '[precompute-progress] {"assigned": 2, "completed": 2, "decode_wait": 1.0, "encode_dtoh": 3.0}'
if [[ -f "$MSPC_SHARED_ROOT/fail" ]]; then echo "mspc: error: boom" >&2; exit 3; fi
mkdir -p "$3/shards" && echo '{}' > "$3/shards/shard-0000$RANK.json"
echo '  "shard_rows": 5,'
echo '  "skipped_existing": 3,'
"""


@pytest.fixture
def run_root(tmp_path):
    entry = tmp_path / "Storyboard" / "run" / "precompute.sh"
    entry.parent.mkdir(parents=True)
    entry.write_text(ENTRY)
    entry.chmod(entry.stat().st_mode | stat.S_IEXEC)
    return tmp_path


def launch(root, **changes):
    args = {"root": root, "config": Path("configs/runtime/cache.json"), "gpu": "1", "shard_index": 2, "num_shards": 4,
            "output_dir": root / "shot_targets_v1", "status_seconds": 60, "min_free_gb": 0, **changes}
    return tool.launch(SimpleNamespace(**args))


def test_launch_passes_one_shard_on_one_card_to_the_entry_and_writes_status(run_root):
    assert launch(run_root) == 0
    status = json.loads((run_root / "status" / "shot-targets-s2of4.json").read_text())
    assert (status["state"], status["exit_code"], status["written"], status["bytes"]) == ("completed", 0, 2, 150)
    assert (status["shard_rows"], status["skipped_existing"], status["shard_complete"]) == (5, 3, True)
    assert status["progress"] == {"assigned": 2, "completed": 2, "decode_wait": 1.0, "encode_dtoh": 3.0}
    log = (run_root / "logs" / "shot-targets-s2of4.log").read_text()
    assert (f"rank=2 world_size=4 nproc_per_node=1 devices=1 root={run_root.resolve()} "
            f"args=configs/runtime/cache.json --output-dir {run_root / 'shot_targets_v1'}") in log
    assert not (run_root / "status" / "shot-targets-failures.jsonl").exists()


def test_launch_records_a_failed_shard(run_root):
    (run_root / "fail").write_text("")
    assert launch(run_root) == 3
    status = json.loads((run_root / "status" / "shot-targets-s2of4.json").read_text())
    assert (status["state"], status["exit_code"], status["shard_complete"]) == ("failed", 3, False)
    failure, = lines(run_root / "status" / "shot-targets-failures.jsonl")
    assert (failure["shard"], failure["num_shards"], failure["exit_code"], failure["written_before_failure"]) == (2, 4, 3, 2)
    assert failure["tail"][-1] == "mspc: error: boom"


def test_launch_refuses_low_disk_or_a_bad_shard(run_root):
    with pytest.raises(SystemExit, match="GB required; nothing started"):
        launch(run_root, min_free_gb=10 ** 9)
    with pytest.raises(ValueError, match="invalid shard"):
        launch(run_root, shard_index=4)
    assert not (run_root / "logs").exists()
