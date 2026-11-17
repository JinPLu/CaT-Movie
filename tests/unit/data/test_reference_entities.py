"""真实源帧group/缺mask边界：保图与已确认ID，不复活单体监督。"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mspc.data.reference_entities import ReferenceEntities
from mspc.data.shot_text import reference_text, reference_text_with_entity_spans

FIXTURE = Path(__file__).parents[2] / "fixtures/renderer_entities/source-group-and-missing-mask.jsonl"


def _fixture_characters(row):
    return [{"name": entity["name"], "appearance": entity["appearance"]} for entity in row["entities"]]


def test_real_group_keeps_name_but_cannot_supply_seed_or_aux(tmp_path):
    reader = ReferenceEntities({"labels_path": str(FIXTURE)}, [418, 14159])
    record = reader.records[418]
    row = next(row for row in record["references"] if row["shot"] == 3)
    group = next(entity for entity in row["entities"] if entity["name"] == "Character 5")
    assert group["kind"] == "group" and group["point_2d"] is None
    assert "Character 5" in reference_text_with_entity_spans(
        row["entities"], setting=None, characters=_fixture_characters(row))["text"]
    broken = copy.deepcopy(record)
    frame = next(row for row in broken["references"] if row["shot"] == 3)
    frame["alignment_cells"] = [{"name": "Character 5", "cell": [0, 0]}]
    path = tmp_path / "group-aux.jsonl"
    path.write_text(json.dumps(broken) + "\n")
    with pytest.raises(ValueError, match="seeded identity"):
        ReferenceEntities({"labels_path": str(path)}, [418])


def test_real_missing_selected_mask_does_not_become_source_caption_fact():
    reader = ReferenceEntities({"labels_path": str(FIXTURE)}, [418, 14159])
    row = next(row for row in reader.records[14159]["references"] if row["shot"] == 2)
    assert row["frame"] == 16
    assert "Character 1" not in reference_text_with_entity_spans(
        row["entities"], setting=None, characters=_fixture_characters(row))["text"]
    assert {"name": "Character 1", "reason": "no_positive_mask_at_selected_frame"} in row["unobserved_planned_identities"]


@pytest.mark.parametrize("name", ["Person 5", "Character 0", "Character 05"])
def test_real_source_labels_validate_character_ids_without_compiling_text(tmp_path, name):
    record = copy.deepcopy(ReferenceEntities({"labels_path": str(FIXTURE)}, [418, 14159]).records[418])
    row = next(row for row in record["references"] if row["entities"])
    row["entities"][0]["name"] = name
    path = tmp_path / "invalid-id.jsonl"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="distinct registered Character IDs"):
        ReferenceEntities({"labels_path": str(path)}, [418])


def test_real_source_labels_reject_duplicate_ids_without_compiling_text(tmp_path):
    record = copy.deepcopy(ReferenceEntities({"labels_path": str(FIXTURE)}, [418, 14159]).records[418])
    row = next(row for row in record["references"] if row["entities"])
    row["entities"].append(copy.deepcopy(row["entities"][0]))
    path = tmp_path / "duplicate-id.jsonl"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="distinct registered Character IDs"):
        ReferenceEntities({"labels_path": str(path)}, [418])


def test_real_source_reader_rejects_same_span_wrong_story():
    reader = ReferenceEntities({"labels_path": str(FIXTURE)}, [418, 14159])
    row = reader.records[418]
    draw = SimpleNamespace(key=(row["training_seed"],), row=SimpleNamespace(case_id="wrong-story"),
                           span=row["span"], references=[(shot - 1, frame) for shot, frame in row["reference_draw"]])
    with pytest.raises(ValueError, match="actual case/span/RGB draw"):
        reader.for_draw(418, draw, source_frames=[frame["source_frame"] for frame in row["references"]])


def test_real_source_reader_rejects_same_training_frame_from_a_different_media_grid():
    reader = ReferenceEntities({"labels_path": str(FIXTURE)}, [418, 14159])
    row = reader.records[418]
    draw = SimpleNamespace(key=(row["training_seed"],), row=SimpleNamespace(case_id=row["case_id"]),
                           span=row["span"], references=[(shot - 1, frame) for shot, frame in row["reference_draw"]])
    expected = [frame["source_frame"] for frame in row["references"]]
    assert reader.for_draw(418, draw, source_frames=expected) == row["references"]
    with pytest.raises(ValueError, match="actual training media grid"):
        reader.for_draw(418, draw, source_frames=[expected[0] + 1, *expected[1:]])


def test_unknown_non_four_grid_midpoint_keeps_generic_reference_text(tmp_path):
    original = ReferenceEntities({"labels_path": str(FIXTURE)}, [418, 14159]).records[418]
    record = copy.deepcopy(original)
    row = record["references"][0]
    row.update(frame=row["frame"] + 2, status="unknown_source_frame", entities=[], alignment_cells=[],
               errors=[{"reason": "no_exact_source_frame_mask"}])
    record["reference_draw"][0][1] = row["frame"]
    path = tmp_path / "unknown-midpoint.jsonl"
    path.write_text(json.dumps(record) + "\n")
    reader = ReferenceEntities({"labels_path": str(path)}, [418])
    compiled = reference_text_with_entity_spans(row["entities"], setting=None, characters=_fixture_characters(row))
    assert compiled["text"] == reference_text()
    assert compiled["entity_spans"] == []
    assert reader.records[418]["references"][0]["source_frame"] == row["source_frame"]


def _real_sparse_source():
    real = json.loads((Path(__file__).parents[2] / "fixtures/target_masks/real_sparse_object.json").read_text())
    source = real["source"]
    member = {"key": source["key"], "case_id": source["case_id"],
              "scene": {key: value for key, value in source["scene"].items() if key != "shots"},
              "shots": [{**shot, "frames": shot["render_frames"]} for shot in source["scene"]["shots"]],
              "annotation": {"characters": [], "shots": [{"present": []} for _ in source["scene"]["shots"]]}}
    return real, member


def test_new_24fps_frame_uses_exact_source_mask_instead_of_old_training_frame_number():
    from mspc.data.reference_entities import compile_reference_frame_entities, matched_mask_frame
    from mspc.data.scene_training_manifest import ANNOTATION_MEDIA
    from mspc.data.shot_targets import planned_row

    real, member = _real_sparse_source()
    media = {**ANNOTATION_MEDIA, "target_fps": 24}
    scene = {**member["scene"], "shots": member["shots"]}
    grid = planned_row(scene, 0, media=media)["source_frame_indices_by_shot"][0]
    old = dict(zip(real["frames"]["shots"][0]["source_frames"], real["frames"]["shots"][0]["train_frames"], strict=True))
    candidates = [frame for frame, source in enumerate(grid) if source in old and frame != old[source]]
    assert candidates
    frame = candidates[len(candidates) // 2]
    source, errors = matched_mask_frame(member=member, shot_index=0, source_frame=grid[frame], source=real["source"],
                                      frames=real["frames"], masks=real["masks"])
    assert source == grid[frame] and source in old and not errors
    row = compile_reference_frame_entities(member=member, shot_index=0, frame=frame, source_frame=grid[frame], source=real["source"],
        frames=real["frames"], masks=real["masks"], media=media, seed=42, sample_index=0,
        reference_index=0, max_alignment_cells=64, unresolved_names=frozenset())
    assert row["source_frame"] == source and row["status"] == "no_visible_character_mask"


def test_real_24fps_midpoint_without_exact_old_sam_stays_unknown():
    from mspc.data.reference_entities import compile_reference_frame_entities
    from mspc.data.scene_training_manifest import ANNOTATION_MEDIA
    from mspc.data.shot_targets import planned_row

    real, member = _real_sparse_source()
    media = {**ANNOTATION_MEDIA, "target_fps": 24}
    grid = planned_row({**member["scene"], "shots": member["shots"]}, 0,
                       media=media)["source_frame_indices_by_shot"][0]
    old_sources = set(real["frames"]["shots"][0]["source_frames"])
    candidates = [frame for frame in range(1, len(grid) - 1) if frame % 4 and grid[frame] not in old_sources]
    frame = min(candidates, key=lambda frame: abs(frame - (len(grid) - 1) / 2))
    row = compile_reference_frame_entities(member=member, shot_index=0, frame=frame, source_frame=grid[frame], source=real["source"],
        frames=real["frames"], masks=real["masks"], media=media, seed=42, sample_index=0,
        reference_index=0, max_alignment_cells=64, unresolved_names=frozenset())
    assert row["source_frame"] == grid[frame]
    assert row["status"] == "unknown_source_frame" and row["entities"] == row["alignment_cells"] == []
    assert row["errors"] == [{"reason": "no_exact_source_frame_mask"}]


def test_crop_visibility_checks_actual_foreground_pixels_and_not_only_box_overlap():
    import numpy as np
    from mspc.data.reference_entities import _mask_visible_after_crop

    old = {"source_to_target_matrix": np.eye(3), "output_size": [6, 4]}
    new = {"source_to_target_matrix": [[1, 0, -2], [0, 1, 0], [0, 0, 1]], "output_size": [2, 4]}
    mask = np.zeros((4, 6), dtype=bool)
    mask[1, 0] = mask[1, 5] = True
    assert not _mask_visible_after_crop(mask, old, new)
    mask[1, 3] = True
    assert _mask_visible_after_crop(mask, old, new)


def test_cpu_label_producer_publishes_actual_source_frames_and_unknown_without_resampling(tmp_path):
    from mspc.data.reference_entities import ReferenceIdentityReview, build_reference_entities
    from mspc.data.scene_training_manifest import ANNOTATION_MEDIA
    from mspc.data.shot_targets import planned_row

    real, member = _real_sparse_source()
    media = {**ANNOTATION_MEDIA, "target_fps": 24}
    grid = planned_row({**member["scene"], "shots": member["shots"]}, 0,
                       media=media)["source_frame_indices_by_shot"][0]
    old_sources = set(real["frames"]["shots"][0]["source_frames"])
    known = next(frame for frame, source in enumerate(grid) if source in old_sources)
    unknown = next(frame for frame, source in enumerate(grid) if frame % 4 and source not in old_sources)
    refs = [known, unknown]
    dataset = SimpleNamespace(training_seed=42, media=media, members={member["case_id"]: member},
        metadata_summary={"sampling_manifest_path": "/frozen/actual-96000.jsonl",
                          "reference_temporal_grid": {"target_fps": 24, "vae_temporal_stride": 4},
                          "target_spatial_media": {"height": 480, "width": 832}},
        draw=lambda index: SimpleNamespace(row=SimpleNamespace(case_id=member["case_id"]), span=(1, 2),
                                           references=((0, refs[index]),)),
        reference_source_frames=lambda draw: [grid[frame] for _, frame in draw.references])
    spatial = tmp_path / "spatial" / member["key"]
    spatial.mkdir(parents=True)
    for name in ("source", "frames", "masks"):
        (spatial / f"{name}.json").write_text(json.dumps(real[name]))
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"schema": "reference_identity_review_v1", "cases": []}))
    output = tmp_path / "references.jsonl"
    summary = build_reference_entities(dataset, spatial_root=spatial.parent, output_path=output,
        sample_indices=[0, 1], max_alignment_cells=64, identity_review=ReferenceIdentityReview(review),
        reuse_labels_path=None, workers=1)
    assert summary["samples"] == 2 and summary["unknown_source_frame"] == 1
    assert summary["media"] == media
    assert summary["reference_temporal_grid"] == {"target_fps": 24, "vae_temporal_stride": 4}
    assert summary["target_spatial_media"] == {"height": 480, "width": 832}
    rows = ReferenceEntities({"labels_path": str(output)}, [0, 1]).records
    assert [rows[index]["references"][0]["source_frame"] for index in (0, 1)] == [grid[known], grid[unknown]]
