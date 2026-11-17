"""参考放大裁剪：裁剪框包住参考帧里全部登记人物（不只被路由的人），物体不约束；包不住或 SAM 帧不可核就不裁。"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mspc.data.history_reading_labels import ShotMasks
from mspc.data.reference_crops import (ReferenceCrops, apply_crops, build_reference_crops,
                                       character_keep_box, draw_crop, entity_box, union_box)
from tests.support.paths import REPO_ROOT

#: 标注画幅与目标画幅相同时的几何（归一化框就是掩码外接框除以 832×480）。
SAME_GRID = {"source_to_target_matrix": np.eye(3).tolist(), "output_size": [832, 480]}


def _contains(outer, inner, tolerance=1e-9):
    return (outer[0] <= inner[0] + tolerance and outer[1] <= inner[1] + tolerance
            and outer[2] >= inner[2] - tolerance and outer[3] >= inner[3] - tolerance)


def test_crop_keeps_the_box_and_keeps_aspect():
    box = draw_crop([0.4, 0.3, 0.6, 0.7], area=0.0, x=0.5, y=0.5)
    assert box is not None and box[0] <= 0.4 and box[2] >= 0.6 and box[1] <= 0.3 and box[3] >= 0.7
    assert abs((box[2] - box[0]) - (box[3] - box[1])) < 1e-9 and (box[2] - box[0]) ** 2 >= 0.65 - 1e-9
    assert draw_crop([0.0, 0.0, 1.0, 0.5], area=0.0, x=0.0, y=0.0) is None  # 包不住就不裁
    assert union_box([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.5, 0.3]]) == [0.1, 0.1, 0.5, 0.4]
    images = torch.arange(2 * 3 * 8 * 16, dtype=torch.int64).reshape(2, 3, 8, 16).remainder(256).to(torch.uint8)
    out = apply_crops(images, [None, [0.25, 0.25, 0.75, 0.75]])
    assert out.shape == images.shape and torch.equal(out[0], images[0]) and not torch.equal(out[1], images[1])


def test_keep_box_covers_every_registered_character_not_only_the_routed_one():
    # 真实块 film-006492 镜 1 第 52 训练帧：Character 1 在右、Character 2 在左边缘，另有两件物体。
    case = json.loads((REPO_ROOT / "tests/fixtures/history_reading_labels/film-006492-b0000.json").read_text())
    reader = ShotMasks(case["masks"])
    characters = {entry["name"] for entry in case["annotation"]["characters"]}
    boxes = {name: entity_box(reader.mask(0, name, 52), SAME_GRID, SAME_GRID) for name in reader.names(0)}
    keep, kept = character_keep_box(reader, 0, 52, characters, SAME_GRID, SAME_GRID)
    assert kept == 2 and keep == union_box([boxes["Character 1"], boxes["Character 2"]])
    crop = draw_crop(keep, area=0.0, x=0.5, y=0.5)
    assert crop is not None and all(_contains(crop, boxes[name]) for name in ("Character 1", "Character 2"))
    # 旧写法只包被路由的人：只给 Character 1 的镜读时，裁剪会把左边的 Character 2 裁掉。
    routed_only = draw_crop(boxes["Character 1"], area=0.0, x=0.5, y=0.5)
    assert not _contains(routed_only, boxes["Character 2"])
    # 物体不约束；帧里没有登记人物时只守面积范围。
    assert character_keep_box(reader, 0, 52, set(), SAME_GRID, SAME_GRID) == (None, 0)
    area_only = draw_crop(None, area=0.0, x=0.0, y=0.0)
    assert abs((area_only[2] - area_only[0]) ** 2 - 0.65) < 1e-9


def _real_sparse_dataset(references):
    """真实块 cinedance-009332 的 SAM 只有一条物体轨迹；按 24fps 目标网格取参考时刻。"""
    from mspc.data.scene_training_manifest import ANNOTATION_MEDIA
    from mspc.data.shot_targets import planned_row

    real = json.loads((REPO_ROOT / "tests/fixtures/target_masks/real_sparse_object.json").read_text())
    source = real["source"]
    member = {"key": source["key"], "case_id": source["case_id"],
              "scene": {key: value for key, value in source["scene"].items() if key != "shots"},
              "shots": [{**shot, "frames": shot["render_frames"]} for shot in source["scene"]["shots"]],
              "annotation": {"characters": [], "shots": [{"present": []} for _ in source["scene"]["shots"]]}}
    media = {**ANNOTATION_MEDIA, "target_fps": 24}
    grid = planned_row({**member["scene"], "shots": member["shots"]}, 0, media=media)["source_frame_indices_by_shot"][0]
    old_sources = set(real["frames"]["shots"][0]["source_frames"])
    frames = {"known": next(frame for frame, source in enumerate(grid) if source in old_sources),
              "unknown": next(frame for frame, source in enumerate(grid) if frame % 4 and source not in old_sources)}
    dataset = SimpleNamespace(training_seed=42, media=media, members={member["case_id"]: member},
        metadata_summary={"sampling_manifest_path": "/frozen/train-sampling.jsonl"},
        draw=lambda index: SimpleNamespace(row=SimpleNamespace(case_id=member["case_id"]), span=(1, 2),
                                           references=((0, frames[references[index]]),)),
        reference_source_frames=lambda draw: [grid[frame] for _, frame in draw.references])
    return real, member, dataset


def _spatial_root(tmp_path, real, member):
    spatial = tmp_path / "spatial" / member["key"]
    spatial.mkdir(parents=True)
    for name in ("source", "frames", "masks"):
        (spatial / f"{name}.json").write_text(json.dumps(real[name]))
    return spatial.parent


def test_crop_table_counts_uncroppable_references_and_skips_frames_without_exact_sam(tmp_path, monkeypatch):
    references = ["known"] * 40 + ["unknown"]
    real, member, dataset = _real_sparse_dataset(references)
    root = _spatial_root(tmp_path, real, member)
    output = tmp_path / "crops.jsonl"
    summary = build_reference_crops(dataset, spatial_root=root, output_path=output,
                                    sample_indices=list(range(41)), workers=1)
    # 这块没有登记人物：每张都只守面积范围裁出；不可核的 SAM 帧不裁。
    assert summary["references"] == 41 and summary["references_without_characters"] == 40
    assert summary["reference_without_reliable_sam_frame"] == 1
    assert summary["cropped"] == 40
    assert summary["characters_too_large"] == 0 and summary["uncroppable_fraction_of_references"] == 0
    assert json.loads(output.with_suffix(".summary.json").read_text()) == summary
    rows = ReferenceCrops({"path": str(output)}, range(41)).records
    assert rows[40]["boxes"] == [None]

    # 人物框占满画面时包不住、不裁，计入摘要比例。
    monkeypatch.setattr("mspc.data.reference_crops.character_keep_box",
                        lambda *args: ([0.0, 0.0, 1.0, 1.0], 1))
    full = build_reference_crops(dataset, spatial_root=root, output_path=tmp_path / "full.jsonl",
                                 sample_indices=list(range(41)), workers=1)
    assert full["cropped"] == 0 and full["characters_too_large"] == 40
    assert full["uncroppable_fraction_of_references"] == pytest.approx(40 / 41)
