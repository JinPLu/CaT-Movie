"""混合目标库使用真实清单行与已观察到的两套生产版本；测试只读取元数据。"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mspc.data.cache_provenance import build_cache_source_collection, load_cache_contract, merge_selected_cache_indexes


REAL = json.loads((Path(__file__).parent / "fixtures/shot_targets_real.json").read_text())["cases"]
PIPELINE = {"tiled": False, "tile_size": (30, 52), "tile_stride": (15, 26)}


def write_manifest(site, origin):
    (site.roots[origin] / "manifest.json").write_text(json.dumps({"records": site.records[origin]}))


def write_index(site):
    site.index.write_text("".join(json.dumps(row) + "\n" for row in site.rows))


@pytest.fixture
def source_site(tmp_path):
    cases = [case for case in REAL if case["first_batch"]]
    weights = tmp_path / "Wan2.2_VAE.pth"
    weights.write_bytes(b"model path binding fixture")
    roots, records, rows = {}, {}, []
    for origin, case in zip(("first_batch_window", "shot"), cases):
        root = tmp_path / origin
        root.mkdir()
        alias = root / "Wan2.2_VAE.pth"
        alias.symlink_to(weights)
        record = deepcopy(case["first_batch"]["record"])
        record["provenance"]["models"]["vae"]["path"] = str(alias)
        if origin == "shot":
            # These are the actual supplemental-shot producer values measured
            # in the full-pool manifest, rather than a common replacement ID.
            record["provenance"]["cache_version"] = "shot_targets_v1"
            record["provenance"]["encoding"]["implementation_revision"] = (
                "86f59ef97bee57dd29006f2381fafe31db61d165")
        roots[origin], records[origin] = root, [record]
        row = {**case["first_batch"]["index_row"], "origin": origin}
        rows.append(row)
        if origin == "first_batch_window":
            # Several target shots may legitimately read one packed window.
            earlier = case["first_batch"]["mapping_rows"][0]
            extra = {**row, **{key: earlier[key] for key in ("sid", "scene_id", "source_shot_index")}}
            if extra["source_shot_index"] != row["source_shot_index"]:
                rows.append(extra)
    site = SimpleNamespace(index=tmp_path / "target-index.jsonl", roots=roots, records=records,
                           rows=rows, weights=weights, contract=tmp_path / "cache-source-collection.json")
    for origin in roots:
        write_manifest(site, origin)
    write_index(site)
    return site


def publish(site):
    value = build_cache_source_collection(site.index, site.roots)
    site.contract.write_text(json.dumps(value))
    return value


def test_collection_preserves_both_producers_and_counts_unique_selected_files(source_site):
    site = source_site
    value = publish(site)
    assert value["schema"] == "latent_cache_source_collection_v1"
    assert value["index_path"] == str(site.index.resolve())
    assert len(site.rows) == 3
    assert {source["selected_files"] for source in value["sources"].values()} == {1}
    for origin, source in value["sources"].items():
        assert source["manifest_path"] == str((site.roots[origin] / "manifest.json").resolve())
        original = site.records[origin][0]["provenance"]
        for key in ("cache_version", "models", "encoding", "context"):
            assert source[key] == original[key]
    loaded = load_cache_contract(site.contract, {"vae": str(site.weights)}, PIPELINE)
    assert loaded["sources"] == value["sources"]
    assert loaded["encoding"] == {key: site.records["shot"][0]["provenance"]["encoding"][key]
                                  for key in ("dtype", "pixel_conversion", "tiled", "tile_size", "tile_stride")}
    assert loaded["models"] == site.records["first_batch_window"][0]["provenance"]["models"]


def test_collection_refuses_a_selected_file_absent_from_manifest(source_site):
    site = source_site
    site.rows[-1]["file"] = "latents/missing.safetensors"
    write_index(site)
    with pytest.raises(ValueError, match="selected cache files are missing"):
        build_cache_source_collection(site.index, site.roots)


def test_collection_refuses_a_duplicate_target_shot_key(source_site):
    site = source_site
    site.rows.append(deepcopy(site.rows[0]))
    write_index(site)
    with pytest.raises(ValueError, match="duplicate cache target key"):
        build_cache_source_collection(site.index, site.roots)


def test_collection_refuses_duplicate_selected_manifest_records(source_site):
    site = source_site
    site.records["shot"].append(deepcopy(site.records["shot"][0]))
    write_manifest(site, "shot")
    with pytest.raises(ValueError, match="duplicate cache source record"):
        build_cache_source_collection(site.index, site.roots)


def test_collection_refuses_heterogeneous_provenance_within_one_origin(source_site):
    site = source_site
    second = deepcopy(site.records["shot"][0])
    second["cache_file"] = "latents/second-selected.safetensors"
    second["provenance"]["encoding"]["implementation_revision"] = "another-producer"
    site.records["shot"].append(second)
    site.rows.append({**site.rows[-1], "source_shot_index": 999, "file": second["cache_file"]})
    write_manifest(site, "shot")
    write_index(site)
    with pytest.raises(ValueError, match="provenance differs within origin"):
        build_cache_source_collection(site.index, site.roots)


@pytest.mark.parametrize("change", ["vae_revision", "dtype", "pixel_conversion", "tiled", "tile_size", "tile_stride"])
def test_collection_refuses_incompatible_vae_or_encoding_between_sources(source_site, change):
    site = source_site
    provenance = site.records["shot"][0]["provenance"]
    if change == "vae_revision":
        provenance["models"]["vae"]["revision"] = "different-vae"
    else:
        provenance["encoding"][change] = {"dtype": "float32", "pixel_conversion": "different-pixels",
            "tiled": True, "tile_size": [32, 52], "tile_stride": [16, 26]}[change]
    write_manifest(site, "shot")
    with pytest.raises(ValueError, match="VAE revision/encoding differs"):
        build_cache_source_collection(site.index, site.roots)


def test_collection_loader_binds_every_source_vae_path(source_site):
    site = source_site
    value = publish(site)
    value["sources"]["shot"]["models"]["vae"]["path"] = str(site.weights.with_name("another-VAE.pth"))
    site.contract.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="shot VAE path differs from loaded model"):
        load_cache_contract(site.contract, {"vae": str(site.weights)}, PIPELINE)


def test_collection_loader_refuses_current_pipeline_encoding_changes(source_site):
    site = source_site
    publish(site)
    with pytest.raises(ValueError, match="encoding differs from loaded pipeline"):
        load_cache_contract(site.contract, {"vae": str(site.weights)}, {**PIPELINE, "tiled": True})


def test_collection_requires_exact_origin_coverage(source_site):
    site = source_site
    with pytest.raises(ValueError, match="target_roots must exactly cover"):
        build_cache_source_collection(site.index, {"shot": site.roots["shot"]})


def test_single_producer_contract_keeps_its_original_encoding_revision(tmp_path):
    # The single-producer contract remains the Wan2.1 production format.
    declaration = {"cache_version": "shot", "models": {"vae": {"path": "Wan2.1_VAE.pth",
                   "revision": "Wan-AI/Wan2.1-T2V-1.3B@37ec512624d61f7aa208f7ea8140a131f93afc9a"}},
                   "prompt_template_revision": "unembedded_target_text",
                   "implementation_revision": "mspc@d2602edc8adbbd4c12ea47bbbfd402a8cdeb38d8"}
    path = tmp_path / "cache_provenance.json"
    path.write_text(json.dumps(declaration))
    loaded = load_cache_contract(path, {"vae": "Wan2.1_VAE.pth"}, PIPELINE)
    assert {key: loaded[key] for key in declaration} == declaration
    assert loaded["encoding"]["implementation_revision"] == declaration["implementation_revision"]


def source_indexes(site):
    paths, roots = {}, {}
    names = {"first_batch_window": "existing_shot", "shot": "backfilled_shot"}
    for old, name in names.items():
        path = site.index.with_name(f"{name}.jsonl")
        path.write_text("".join(json.dumps(row) + "\n" for row in site.rows if row["origin"] == old))
        paths[name], roots[name] = path, site.roots[old]
    return paths, roots


def test_merge_selected_indexes_only_relabels_origins_and_preserves_sources(source_site):
    site = source_site
    paths, roots = source_indexes(site)
    original = {path: path.read_bytes() for path in [*paths.values(),
                *(root / "manifest.json" for root in roots.values())]}
    selected = list(reversed(site.rows))
    merged = merge_selected_cache_indexes(paths, selected)
    names = {"first_batch_window": "existing_shot", "shot": "backfilled_shot"}
    assert merged == [{**row, "origin": names[row["origin"]]} for row in selected]
    output = site.index.with_name("merged-index.jsonl")
    output.write_text("".join(json.dumps(row) + "\n" for row in merged))
    collection = build_cache_source_collection(output, roots)
    for old, new in names.items():
        assert collection["sources"][new]["encoding"] == site.records[old][0]["provenance"]["encoding"]
    assert {path: path.read_bytes() for path in original} == original


@pytest.mark.parametrize("difference", ["duplicate_source", "missing_selected", "duplicate_selected", "empty_origin"])
def test_merge_selected_indexes_refuses_ambiguous_or_missing_sources(source_site, difference):
    paths, _ = source_indexes(source_site)
    selected = list(source_site.rows)
    if difference == "duplicate_source":
        with paths["backfilled_shot"].open("a") as handle:
            handle.write(json.dumps(selected[0]) + "\n")
    elif difference == "missing_selected":
        selected.append({**selected[0], "source_shot_index": 999999})
    elif difference == "duplicate_selected":
        selected.append(selected[0])
    else:
        paths[" "] = paths.pop("existing_shot")
    with pytest.raises(ValueError, match="duplicate cache target key|selected cache shots are absent|unique selected|source origin"):
        merge_selected_cache_indexes(paths, selected)
