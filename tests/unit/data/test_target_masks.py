"""从真实SAM缺帧记录派生的CPU测试；成员外壳仅测试来源连接，不作语义验收。"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import numpy as np
import torch.nn.functional as F

from mspc.data.scene_training_manifest import ANNOTATION_MEDIA
from mspc.data.target_masks import (TargetMaskWeights, build_target_mask_weights,
                                   compile_target_mask_weights, compile_member_target_mask_weights)
from tests.support.paths import REPO_ROOT

WAN22_MEDIA = {"height": 704, "width": 1280, "target_fps": 24, "vae_temporal_stride": 4, "max_rgb_frames": 360}


@pytest.fixture
def real():
    pytest.importorskip("pycocotools")
    return json.loads((REPO_ROOT / "tests/fixtures/target_masks/real_sparse_object.json").read_text())


def compile_fixture(real, stride=16, media=ANNOTATION_MEDIA):
    return compile_target_mask_weights(source=real["source"], frames=real["frames"], masks=real["masks"],
                                       shot_index=0, entity_names=real["selected_entities"], weighting="area_coverage",
                                       vae_spatial_stride=stride, media=media)


def member_for(real):
    source = real["source"]
    return {"schema_version": 1, "key": source["key"], "case_id": source["case_id"],
            "scene": {key: value for key, value in source["scene"].items() if key != "shots"},
            "shots": [{**shot, "shot": i + 1, "frames": shot["render_frames"], "usable": i == 0}
                      for i, shot in enumerate(source["scene"]["shots"])],
            "annotation": {"characters": [], "objects": [{"name": "Object 2"}],
                           "shots": [{"present": [{"name": "Object 2"}]}, {"present": []}]}}


def materialize(real, tmp_path):
    member = member_for(real)
    members = tmp_path / "members.jsonl"
    members.write_text(json.dumps(member, ensure_ascii=False) + "\n")
    root = tmp_path / "spatial"
    block = root / member["key"]
    block.mkdir(parents=True)
    for field in ("source", "frames", "masks"):
        (block / f"{field}.json").write_text(json.dumps(real[field]))
    return member, members, root, tmp_path / "weights-cache"


def test_real_sparse_mask_frames_remain_unknown_without_interpolation(real):
    weights, record = compile_fixture(real)
    assert weights.shape == (1, 1, 19, 30, 52)
    assert sum(record["valid_mask_frames"]) == 6
    assert len(record["unknown_mask_frames"]) == 13
    assert torch.all(weights[:, :, ~torch.tensor(record["valid_mask_frames"])] == 1)
    assert record["temporal_alignment"] == "causal_chunk_endpoint"
    assert record["vae_input_chunks"][:3] == [[0, 1], [1, 5], [5, 9]]
    assert record["mask_training_frames"] == list(range(0, 73, 4))
    assert weights.min() >= 1 and weights.max() <= 2
    # 面积平均必须保持真实RLE并集的总面积，而非patch 15×26格或bbox面积。
    area = sum(row["area"] for track in real["masks"]["shots"]["s01"]["tracks"].values()
               for row in track["frames"].values())
    torch.testing.assert_close((weights - 1).sum(), torch.tensor(area / 256), rtol=1e-6, atol=1e-5)


def test_real_overlapping_track_is_unioned_before_pooling(real):
    expected, _ = compile_fixture(real)
    tracks = real["masks"]["shots"]["s01"]["tracks"]
    tracks["duplicate-same-instance"] = deepcopy(next(iter(tracks.values())))
    actual, _ = compile_fixture(real)
    assert torch.equal(expected, actual)


def test_real_source_grid_and_damaged_rle_are_rejected(real):
    shifted = deepcopy(real)
    shifted["frames"]["shots"][0]["source_frames"][1] += 1
    with pytest.raises(ValueError, match="crop/time grid"):
        compile_fixture(shifted)
    track = next(iter(real["masks"]["shots"]["s01"]["tracks"].values()))
    next(iter(track["frames"].values()))["area"] += 1
    with pytest.raises(ValueError, match="RLE area"):
        compile_fixture(real)


def test_empty_real_selected_entity_is_unknown_uniform(real):
    real["masks"]["shots"]["s01"]["tracks"].clear()
    weights, record = compile_fixture(real)
    assert record["mask_status"] == "unknown_uniform"
    assert not any(record["valid_mask_frames"])
    assert torch.equal(weights, torch.ones_like(weights))


def test_producer_reader_roundtrip_and_missing_shot_error(real, tmp_path):
    member, members, root, output = materialize(real, tmp_path)
    summary = build_target_mask_weights(members_path=members, spatial_root=root, output_dir=output, weighting="area_coverage", vae_spatial_stride=16)
    assert (summary["shots"], summary["latent_frames"], summary["valid_mask_frames"], summary["unknown_mask_frames"]) == (1, 19, 6, 13)
    reader = TargetMaskWeights(index_path=str(output / "index.jsonl"), root=str(output), weighting="area_coverage")
    expected, _ = compile_fixture(real)
    weights, record = reader.read(member, 0, latent_spatial_shape=(30, 52))
    assert torch.equal(weights, expected)
    assert record["selected_entities"] == ["Object 2"]
    with pytest.raises(ValueError, match="missing real SAM"):
        reader.record_for(member, 1, latent_spatial_shape=(30, 52))
    wrong = deepcopy(member)
    wrong["scene"]["crop_box"][0] += 1
    with pytest.raises(ValueError, match="source differs"):
        reader.record_for(wrong, 0, latent_spatial_shape=(30, 52))


def test_online_original_rle_matches_the_same_offline_pool8_compiler(real, tmp_path):
    member, members, root, output = materialize(real, tmp_path)
    build_target_mask_weights(members_path=members, spatial_root=root, output_dir=output,
                              weighting='area_coverage', vae_spatial_stride=8)
    offline = TargetMaskWeights(index_path=str(output / 'index.jsonl'), root=str(output), weighting='area_coverage')
    expected, expected_record = offline.read(member, 0, latent_spatial_shape=(60, 104))
    [(actual, record)] = compile_member_target_mask_weights(member=member, positions=[0],
        spatial_root=str(root), weighting='area_coverage', vae_spatial_stride=8, media=ANNOTATION_MEDIA)
    assert torch.equal(actual, expected)
    assert all(record[key] == value for key, value in expected_record.items() if key in record)
    changed = deepcopy(member)
    changed['scene']['crop_box'][0] += 1
    with pytest.raises(ValueError, match='source media/crop'):
        compile_member_target_mask_weights(member=changed, positions=[0], spatial_root=str(root),
                                          weighting='area_coverage', vae_spatial_stride=8, media=ANNOTATION_MEDIA)
    shifted = deepcopy(member)
    shifted['shots'][0]['start_frame'] += 1
    shifted['shots'][0]['end_frame'] += 1
    shifted['scene']['boundary_revision']['shots'][0]['revised_segment'] = [
        shifted['shots'][0]['start_frame'], shifted['shots'][0]['end_frame']]
    with pytest.raises(ValueError, match='source time/grid'):
        compile_member_target_mask_weights(member=shifted, positions=[0], spatial_root=str(root),
                                          weighting='area_coverage', vae_spatial_stride=8, media=ANNOTATION_MEDIA)


def test_cached_validity_is_checked_against_real_unknown_frames(real, tmp_path):
    member, members, root, output = materialize(real, tmp_path)
    build_target_mask_weights(members_path=members, spatial_root=root, output_dir=output, weighting="area_coverage", vae_spatial_stride=16)
    reader = TargetMaskWeights(index_path=str(output / "index.jsonl"), root=str(output), weighting="area_coverage")
    weights, record = reader.read(member, 0, latent_spatial_shape=(30, 52))
    weights[:, :, record["valid_mask_frames"].index(False)] = 2
    save_file({"target_loss_weights": weights, "valid_mask_frames": torch.tensor(record["valid_mask_frames"])},
              str(output / record["file"]), metadata={"target_mask_record": json.dumps(record)})
    with pytest.raises(ValueError, match="invalid target SAM weight"):
        reader.read(member, 0, latent_spatial_shape=(30, 52))


def test_missing_source_file_is_a_technical_failure(real, tmp_path):
    _, members, root, output = materialize(real, tmp_path)
    (root / real["source"]["key"] / "masks.json").unlink()
    with pytest.raises(FileNotFoundError):
        build_target_mask_weights(members_path=members, spatial_root=root, output_dir=output, weighting="area_coverage", vae_spatial_stride=16)
    assert not output.exists()


def test_touched_cell_weighting_gives_every_cell_a_subject_pixel_touches_full_weight(real):
    area, record = compile_fixture(real)
    touched, touched_record = compile_target_mask_weights(
        source=real["source"], frames=real["frames"], masks=real["masks"], shot_index=0,
        entity_names=real["selected_entities"], weighting="touched_cell", vae_spatial_stride=16, media=ANNOTATION_MEDIA)
    assert touched_record["spatial_alignment"] == "union_then_any_pixel_16x16"
    assert touched_record["valid_mask_frames"] == record["valid_mask_frames"]
    assert torch.equal(touched > 1, area > 1)
    assert set(touched.unique().tolist()) <= {1.0, 2.0}
    assert (touched - 1).sum() >= (area - 1).sum()


def test_one_index_never_mixes_two_weightings(real, tmp_path):
    _, members, root, output = materialize(real, tmp_path)
    build_target_mask_weights(members_path=members, spatial_root=root, output_dir=output, weighting="touched_cell", vae_spatial_stride=16)
    rows = [json.loads(line) for line in (output / "index.jsonl").read_text().splitlines()]
    assert TargetMaskWeights(index_path=str(output / "index.jsonl"), root=str(output),
                             weighting="touched_cell").weighting == "touched_cell"
    with pytest.raises(ValueError, match="not the declared area_coverage"):
        TargetMaskWeights(index_path=str(output / "index.jsonl"), root=str(output), weighting="area_coverage")
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text("".join(json.dumps({**row, "shot": row["shot"] + 10 * i,
                                         "spatial_alignment": alignment}) + "\n"
                             for i, (row, alignment) in enumerate([(rows[0], "union_then_area_pool_16x16"),
                                                                    (rows[0], "union_then_any_pixel_16x16")])))
    with pytest.raises(ValueError, match="not the declared touched_cell"):
        TargetMaskWeights(index_path=str(mixed), root=str(output), weighting="touched_cell")


def test_wan21_pool_uses_original_rle_area_and_preserves_temporal_grid(real, tmp_path):
    native, record = compile_fixture(real, stride=8)
    compressed, compressed_record = compile_fixture(real, stride=16)
    assert native.shape == (1, 1, 19, 60, 104)
    assert record["spatial_alignment"] == "union_then_area_pool_8x8"
    for key in ("valid_mask_frames", "mask_source_frames", "mask_training_frames", "vae_input_chunks"):
        assert record[key] == compressed_record[key]
    # 同一真实RLE的主体面积，分别以64和256像素一个latent cell计算。
    torch.testing.assert_close((native - 1).sum(), (compressed - 1).sum() * 4,
                               rtol=1e-6, atol=1e-5)
    member, members, root, output = materialize(real, tmp_path)
    build_target_mask_weights(members_path=members, spatial_root=root, output_dir=output,
                              weighting="area_coverage", vae_spatial_stride=8)
    reader = TargetMaskWeights(index_path=str(output / "index.jsonl"), root=str(output), weighting="area_coverage")
    actual, _ = reader.read(member, 0, latent_spatial_shape=(60, 104))
    assert torch.equal(actual, native)
    with pytest.raises(ValueError, match="selected target latent grid"):
        reader.read(member, 0, latent_spatial_shape=(30, 52))


def test_annotation_media_is_the_identity_mapping(real):
    from pycocotools import mask as coco

    weights, record = compile_fixture(real)
    assert record["mask_frame_by_target"] == record["mask_training_frames"] == record["target_training_frames"]
    assert record["target_source_frames"] == record["mask_source_frames"]
    assert record["target_crop_box"] == record["crop_box"] and record["pixel_shape"] == [480, 832]
    planes = np.zeros((19, 480, 832), dtype=bool)
    for track in real["masks"]["shots"]["s01"]["tracks"].values():
        if track["name"] in real["selected_entities"]:
            for frame, row in track["frames"].items():
                planes[int(frame) // 4] |= coco.decode(row["rle"]).astype(bool)
    expected = 1 + F.avg_pool2d(torch.from_numpy(planes).float()[:, None], 16, 16).transpose(0, 1)[None]
    assert torch.equal(weights, expected)


def test_704p_24fps_maps_sam_through_source_crop_and_nearest_source_frame(real):
    from pycocotools import mask as coco

    tracks = real["masks"]["shots"]["s01"]["tracks"]
    name = real["selected_entities"][0]
    kept = next(key for key, track in tracks.items() if track["name"] == name)
    for key in [key for key in tracks if key != kept]:
        del tracks[key]
    # 旧480画面左半边为主体，只标在 SAM 训练帧4（源帧352）。
    left = np.zeros((480, 832), dtype=np.uint8)
    left[:, :416] = 1
    rle = coco.encode(np.asfortranarray(left))
    rle["counts"] = rle["counts"].decode()
    tracks[kept]["frames"] = {"4": {"rle": rle, "area": int(left.sum()), "bbox_xywh": coco.toBbox(rle).tolist()}}
    weights, record = compile_fixture(real, media=WAN22_MEDIA)
    # crop_inside(旧crop [357,0,1563,696], 20:11) = [360,18,1560,678]；源24fps，目标端点每4源帧，SAM端点每6源帧。
    assert record["target_crop_box"] == [360, 18, 1560, 678]
    assert record["target_source_frames"][:6] == [346, 350, 354, 358, 362, 366]
    assert record["mask_frame_by_target"][:6] == [0, 4, 4, 8, 12, 12]
    assert weights.shape == (1, 1, len(record["target_training_frames"]), 44, 80)
    assert record["latent_shape"] == list(weights.shape) and record["pixel_shape"] == [704, 1280]
    assert record["valid_mask_frames"][:4] == [False, True, True, False]
    assert sum(record["valid_mask_frames"]) == 2
    # 旧左半边经源坐标落到新画布第0–639列：前40个latent列全为主体，其余为背景。
    for frame in (1, 2):
        assert torch.equal(weights[0, 0, frame, :, :40], torch.full((44, 40), 2.0))
        assert torch.equal(weights[0, 0, frame, :, 40:], torch.ones(44, 40))
    assert torch.equal(weights[0, 0, 0], torch.ones(44, 80))
