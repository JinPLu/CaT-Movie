"""同一实际参考RGB的逐实体标签：CPU生产，训练worker只读，不运行SAM。"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from mspc.core.draws import unit_draw
from mspc.data.atomic_publish import publish_file
from mspc.data.history_reading_labels import ShotMasks
from mspc.data.scene_training_manifest import ANNOTATION_MEDIA, require_training_media, training_geometry
from mspc.data.shot_targets import planned_row
from mspc.data.target_masks import _match_member_source
from mspc.models.planner_tokens import GRID_COLUMNS, GRID_ROWS, point_cell


REFERENCE_ENTITY_SCHEMA = "reference_frame_entities_v2"


class ReferenceIdentityReview:
    """显式的已知身份疑例；未裁ID只撤可靠点/监督，不改名字或RGB。"""
    def __init__(self, path: Path):
        self.path = Path(path)
        value = _json(self.path)
        if set(value) != {"schema", "cases"} or value["schema"] != "reference_identity_review_v1":
            raise ValueError("identity review requires reference_identity_review_v1 and cases")
        self.cases = {}
        for row in value["cases"]:
            if (set(row) != {"case_id", "unresolved_names", "reason", "evidence"}
                    or row["case_id"] in self.cases or not row["reason"] or not row["evidence"]
                    or not isinstance(row["unresolved_names"], list)
                    or len(set(row["unresolved_names"])) != len(row["unresolved_names"])):
                raise ValueError("identity review cases must be explicit and unique")
            self.cases[row["case_id"]] = row

    def unresolved(self, member: Mapping[str, Any]) -> frozenset[str]:
        names = self.cases.get(member["case_id"], {}).get("unresolved_names", [])
        if not set(names) <= {entry["name"] for entry in member["annotation"]["characters"]}:
            raise ValueError("identity review names are absent from the actual registry")
        return frozenset(names)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _validate_training_record(record: Mapping[str, Any]) -> None:
    if set(record) != {"schema", "sample_index", "training_seed", "case_id", "span", "reference_draw", "references"}:
        raise ValueError("reference entity sample has unsupported or missing fields")
    if (record["schema"] != REFERENCE_ENTITY_SCHEMA or type(record["sample_index"]) is not int
            or record["sample_index"] < 0 or type(record["training_seed"]) is not int
            or not isinstance(record["case_id"], str) or not record["case_id"]
            or not isinstance(record["span"], list) or len(record["span"]) != 2
            or any(type(v) is not int for v in record["span"]) or not 0 <= record["span"][0] < record["span"][1]
            or not isinstance(record["references"], list) or len(record["references"]) > 10):
        raise ValueError("reference entity sample identity/span is invalid")
    moments = record["reference_draw"]
    if (not isinstance(moments, list) or any(not isinstance(m, list) or len(m) != 2
            or type(m[0]) is not int or m[0] < 1 or type(m[1]) is not int or m[1] < 0 for m in moments)
            or moments != sorted(moments) or len({m[0] for m in moments}) != len(moments)
            or [[row["shot"], row["frame"]] for row in record["references"]] != moments):
        raise ValueError("reference entity rows must match chronological actual RGB moments")
    for row in record["references"]:
        if (set(row) != {"shot", "frame", "source_frame", "status", "entities", "alignment_cells", "ignored_track_names", "errors", "unobserved_planned_identities"}
                or type(row["source_frame"]) is not int or row["source_frame"] < 0
                or row["status"] not in ("observed", "no_visible_character_mask", "annotation_error", "unknown_source_frame")
                or not isinstance(row["entities"], list) or not isinstance(row["alignment_cells"], list)
                or len(row["alignment_cells"]) > 64 or not isinstance(row["errors"], list)
                or not isinstance(row["ignored_track_names"], list)):
            raise ValueError("reference entity frame has invalid labels/status")
        names, seeded, point_cells = set(), set(), set()
        for entity in row["entities"]:
            if not isinstance(entity, Mapping) or set(entity) != {"name", "kind", "appearance", "identity_status", "point_2d", "point_cell", "point_status"}:
                raise ValueError("source entity requires its exact identity and point fields")
            name = entity["name"]
            if (not isinstance(name, str) or re.fullmatch(r"Character [1-9]\d*", name) is None
                    or name in names):
                raise ValueError("reference identities must be distinct registered Character IDs")
            names.add(name)
            if entity["kind"] not in {"person", "animal", "group"} or entity["identity_status"] not in {"registered", "unresolved"}:
                raise ValueError("source entity kind/identity status is invalid")
            point, cell = entity["point_2d"], entity["point_cell"]
            if point is None:
                reason = ("group_entity" if entity["kind"] == "group" else "identity_unresolved"
                          if entity["identity_status"] == "unresolved" else "no_unambiguous_cell")
                if cell is not None or entity["point_status"] not in ({reason, "not_produced_for_media"}
                        if reason == "no_unambiguous_cell" else {reason}):
                    raise ValueError("unknown source point must carry no routed cell")
            else:
                if (not isinstance(point, list) or len(point) != 2
                        or any(type(v) is not int or not 0 <= v <= 1000 for v in point)
                        or cell != list(point_cell(point)) or entity["point_status"] != "visible_internal"
                        or entity["kind"] == "group" or entity["identity_status"] == "unresolved"):
                    raise ValueError("source point and actual routed cell differ")
                seeded.add(entity["name"])
                point_cells.add(tuple(cell))
        cells = []
        for label in row["alignment_cells"]:
            if (set(label) != {"name", "cell"} or label["name"] not in seeded
                    or not isinstance(label["cell"], list) or len(label["cell"]) != 2
                    or any(type(v) is not int or not 0 <= v < limit for v, limit in zip(label["cell"], (15, 26)))):
                raise ValueError("source alignment cell requires an actual seeded identity and grid cell")
            cells.append(tuple(label["cell"]))
        if len(set(cells)) != len(cells) or point_cells.intersection(cells) or cells and len(seeded) < 2:
            raise ValueError("source alignment must use unambiguous nonpoint cells of at least two seeded identities")
        if row["status"] != "observed" and (row["entities"] or cells):
            raise ValueError("unknown source frames retain RGB but cannot supply identity labels")


def matched_mask_frame(*, member: Mapping[str, Any], shot_index: int, source_frame: int,
                                     source: Mapping[str, Any], frames: Mapping[str, Any], masks: Mapping[str, Any],
                                     ) -> tuple[int, list]:
    """核旧SAM完整来源合同及显式原片源帧；不存在同源mask也不借邻帧。"""
    _match_member_source(member, source)
    if (frames["key"] != member["key"] or frames["train_fps"] != 16 or frames["frame_stride"] != 4
            or masks["schema_version"] != 2 or (masks["height"], masks["width"]) != (480, 832)
            or masks["frame_numbers"] != "training frame within the shot"
            or masks["rle_format"] != "COCO compressed, Fortran order"):
        raise ValueError("source identity mask/frame producer contract differs")
    shot = member["shots"][shot_index]
    scene = {**member["scene"], "shots": member["shots"]}
    annotation_grid = planned_row(scene, shot_index, media=ANNOTATION_MEDIA)["source_frame_indices_by_shot"][0]
    if (type(source_frame) is not int
            or not shot["start_frame"] <= source_frame < shot["end_frame"]):
        raise ValueError("source identity frame is outside the actual source shot")
    sampled = frames["shots"][shot_index]
    endpoints = list(range(0, shot["frames"], 4))
    sid = f"s{shot_index + 1:02d}"
    ms = masks["shots"][sid]
    if (sampled["id"] != sid or sampled["shot"] != shot_index + 1
            or sampled["render_frames"] != shot["frames"] or sampled["train_frames"] != endpoints
            or sampled["source_frames"] != [annotation_grid[index] for index in endpoints]
            or ms["train_frames"] != sampled["train_frames"] or ms["source_frames"] != sampled["source_frames"]):
        raise ValueError("source identity SAM labels do not identify the actual selected RGB/crop")
    return source_frame, [*sampled["annotation_errors"], *ms["annotation_errors"]]


def _missing_planned_identities(member, shot_index, entities, *, reason="no_positive_mask_at_selected_frame"):
    registered = {entry["name"] for entry in member["annotation"]["characters"]}
    visible = {entry["name"] for entry in entities}
    return [{"name": entry["name"], "reason": reason}
            for entry in member["annotation"]["shots"][shot_index]["present"]
            if entry["name"] in registered and entry["name"] not in visible]


def compile_reference_frame_entities(*, member: Mapping[str, Any], shot_index: int, frame: int, source_frame: int,
                                     source: Mapping[str, Any], frames: Mapping[str, Any], masks: Mapping[str, Any],
                                     media: Mapping[str, Any],
                                     seed: int, sample_index: int, reference_index: int,
                                     max_alignment_cells: int, unresolved_names: frozenset[str]) -> dict:
    """源点取可见内部；群体和已知未裁ID保留文字/RGB但不供单实例路由与监督。"""
    if type(max_alignment_cells) is not int or not 1 <= max_alignment_cells <= 64:
        raise ValueError("source alignment sampling requires 1..64 cells per reference")
    source_frame, errors = matched_mask_frame(member=member, shot_index=shot_index, source_frame=source_frame,
                                             source=source, frames=frames, masks=masks)
    sampled = frames["shots"][shot_index]
    matching = [old_frame for old_frame, old_source in zip(sampled["train_frames"], sampled["source_frames"], strict=True)
                if old_source == source_frame]
    if len(matching) != 1:
        reason = "no_exact_source_frame_mask" if not matching else "ambiguous_source_frame_mask"
        return {"shot": shot_index + 1, "frame": frame, "source_frame": source_frame,
                "status": "unknown_source_frame", "entities": [], "alignment_cells": [],
                "ignored_track_names": [], "errors": [*errors, {"reason": reason}],
                "unobserved_planned_identities": _missing_planned_identities(member, shot_index, [], reason=reason)}
    if errors:
        # 文件存在但标签失败是明确的未知身份，不删除仍合法的RGB。
        return {"shot": shot_index + 1, "frame": frame, "source_frame": source_frame,
                "status": "annotation_error", "entities": [], "alignment_cells": [],
                "ignored_track_names": [], "errors": errors,
                "unobserved_planned_identities": _missing_planned_identities(member, shot_index, [])}
    registry = {entry["name"]: entry for entry in member["annotation"]["characters"]}
    reader = ShotMasks(masks)
    old_geometry = training_geometry(member["scene"], media=ANNOTATION_MEDIA)
    geometry = training_geometry(member["scene"], media=media)
    same_geometry = geometry == old_geometry
    visible = {}
    for name in reader.names(shot_index):
        if name not in registry:
            continue
        mask = reader.mask(shot_index, name, matching[0])
        if mask is not None and mask.any():
            if mask.shape != (480, 832):
                raise ValueError("source entity mask uses another crop/pixel grid")
            if same_geometry or _mask_visible_after_crop(mask, old_geometry, geometry):
                visible[name] = mask
    if not same_geometry:
        # 新基线只消费确认ID的文字。旧480网格的内部点/辅助格不冒充新画幅的点监督。
        entities = [{"name": name, "kind": registry[name]["kind"], "appearance": registry[name]["appearance"],
                     "identity_status": "unresolved" if name in unresolved_names else "registered",
                     "point_2d": None, "point_cell": None,
                     "point_status": ("group_entity" if registry[name]["kind"] == "group" else
                                      "identity_unresolved" if name in unresolved_names else "not_produced_for_media")}
                    for name in registry if name in visible]
        return {"shot": shot_index + 1, "frame": frame, "source_frame": source_frame,
                "status": "observed" if entities else "no_visible_character_mask", "entities": entities,
                "alignment_cells": [], "ignored_track_names": sorted(set(reader.names(shot_index)) - set(registry)),
                "errors": [], "unobserved_planned_identities": _missing_planned_identities(member, shot_index, entities)}
    majority = {name: mask.reshape(GRID_ROWS, 32, GRID_COLUMNS, 32).mean((1, 3)) > .5
                for name, mask in visible.items()}
    reliable, exclusive = {}, {}
    for name, covered in majority.items():
        others = np.zeros_like(covered)
        other_pixels = np.zeros_like(visible[name])
        for other, other_covered in majority.items():
            if other != name:
                others |= other_covered
                other_pixels |= visible[other]
        reliable[name] = covered & ~others
        exclusive[name] = visible[name] & ~other_pixels
    key = (seed, sample_index, reference_index, shot_index, frame)
    entities, point_cells = [], set()
    for name in registry:
        if name not in visible:
            continue
        point_grid = exclusive[name].reshape(GRID_ROWS, 32, GRID_COLUMNS, 32).any((1, 3))
        cells = list(zip(*np.nonzero(reliable[name] & point_grid)))
        point, cell = None, None
        kind = registry[name]["kind"]
        eligible = kind != "group" and name not in unresolved_names
        if cells and eligible:
            r, c = min(cells, key=lambda p: unit_draw("reference_identity_point", *key, name, *map(int, p)))
            r, c = int(r), int(c)
            ys, xs = np.nonzero(exclusive[name][r * 32:(r + 1) * 32, c * 32:(c + 1) * 32])
            nearest = np.argmin((ys - 15.5) ** 2 + (xs - 15.5) ** 2)
            y, x = r * 32 + int(ys[nearest]), c * 32 + int(xs[nearest])
            point = [round((x + .5) * 1000 / 832), round((y + .5) * 1000 / 480)]
            cell = [r, c]
            if tuple(point_cell(point)) != (r, c) or not exclusive[name][y, x]:
                raise ValueError("source internal point does not quantize to its own reliable mask cell")
            point_cells.add((r, c))
        reason = ("group_entity" if kind == "group" else "identity_unresolved" if name in unresolved_names
                  else "visible_internal" if point else "no_unambiguous_cell")
        entities.append({"name": name, "kind": kind, "appearance": registry[name]["appearance"],
                         "identity_status": "unresolved" if name in unresolved_names else "registered",
                         "point_2d": point, "point_cell": cell, "point_status": reason})
    candidates = {}
    for entity in entities:
        if entity["point_cell"] is None:
            continue
        name = entity["name"]
        cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(reliable[name])) if (int(r), int(c)) not in point_cells]
        candidates[name] = sorted(cells, key=lambda p: unit_draw("reference_identity_alignment", *key, name, *p))
    selected = []
    if len(candidates) >= 2:
        # 轮流从每个人物取格，最多64格；没有点种子的身份不提供监督。
        position = 0
        while len(selected) < max_alignment_cells:
            added = False
            for name, cells in candidates.items():
                if position < len(cells) and len(selected) < max_alignment_cells:
                    selected.append({"name": name, "cell": list(cells[position])})
                    added = True
            if not added:
                break
            position += 1
    return {"shot": shot_index + 1, "frame": frame, "source_frame": source_frame,
            "status": "observed" if entities else "no_visible_character_mask", "entities": entities,
            "alignment_cells": selected,
            "ignored_track_names": sorted(set(reader.names(shot_index)) - set(registry)), "errors": [],
            "unobserved_planned_identities": _missing_planned_identities(member, shot_index, entities)}


def _mask_visible_after_crop(mask, old_geometry, geometry):
    """只核同源SAM前景像素中心是否还在新画布内；不制造新分割或提示点。"""
    transform = np.asarray(geometry["source_to_target_matrix"], dtype=np.float64) @ np.linalg.inv(
        np.asarray(old_geometry["source_to_target_matrix"], dtype=np.float64))
    ys, xs = np.nonzero(mask)
    x = transform[0, 0] * (xs + .5) + transform[0, 1] * (ys + .5) + transform[0, 2]
    y = transform[1, 0] * (xs + .5) + transform[1, 1] * (ys + .5) + transform[1, 2]
    width, height = geometry["output_size"]
    return bool(((0 <= x) & (x < width) & (0 <= y) & (y < height)).any())


class ReferenceEntities:
    """完整样本标签读者；身份行和实际RGB抽样逐项匹配，不从整镜在场名单猜名字。"""

    def __init__(self, config: Mapping[str, Any], expected_sample_indices: Sequence[int]):
        if not isinstance(config, Mapping) or set(config) != {"labels_path"}:
            raise ValueError("reference_entities requires exactly labels_path")
        self.records = {}
        for record in _rows(Path(config["labels_path"])):
            _validate_training_record(record)
            index = record["sample_index"]
            if record["schema"] != REFERENCE_ENTITY_SCHEMA or index in self.records:
                raise ValueError("duplicate or unsupported reference entity row")
            self.records[index] = record
        if set(self.records) != set(expected_sample_indices):
            raise ValueError("reference entity labels must cover the exact sample indices")

    def for_draw(self, index: int, draw, *, source_frames: Sequence[int]) -> list[dict]:
        record = self.records[index]
        expected = [[shot + 1, frame] for shot, frame in draw.references]
        if (record["training_seed"] != draw.key[0] or record["case_id"] != draw.row.case_id
                or record["span"] != list(draw.span)
                or record["reference_draw"] != expected
                or [[row["shot"], row["frame"]] for row in record["references"]] != expected):
            raise ValueError("reference identity labels differ from the actual case/span/RGB draw")
        if (len(source_frames) != len(expected) or any(type(frame) is not int or frame < 0 for frame in source_frames)
                or [row["source_frame"] for row in record["references"]] != list(source_frames)):
            raise ValueError("reference identity source frames differ from the actual training media grid")
        return record["references"]


_REFERENCE_WORKER_STATE = None


def _init_reference_worker(dataset, spatial_root, identity_review, reused, max_alignment_cells):
    global _REFERENCE_WORKER_STATE
    _REFERENCE_WORKER_STATE = (dataset, spatial_root, identity_review, reused, max_alignment_cells)


@lru_cache(maxsize=16)
def _read_reference_block(spatial_root, block):
    directory = spatial_root / block
    return tuple(_json(directory / name) for name in ("source.json", "frames.json", "masks.json"))


def _reference_sample(index):
    if _REFERENCE_WORKER_STATE is None:
        raise RuntimeError("source label worker was not initialized")
    dataset, spatial_root, identity_review, reused, max_alignment_cells = _REFERENCE_WORKER_STATE
    counts = Counter()
    draw = dataset.draw(index)
    source_frames = dataset.reference_source_frames(draw)
    member = dataset.members[draw.row.case_id]
    block = member["key"]
    unresolved = identity_review.unresolved(member)
    registry = {entry["name"]: entry for entry in member["annotation"]["characters"]}
    references = []
    previous = reused.get(index)
    if previous is not None and (previous["schema"] != "reference_frame_entities_v1"
            or previous["case_id"] != draw.row.case_id or previous["span"] != list(draw.span)
            or previous["training_seed"] != dataset.training_seed
            or previous["reference_draw"] != [[shot + 1, frame] for shot, frame in draw.references]
            or [[row["shot"], row["frame"]] for row in previous["references"]] != previous["reference_draw"]):
        raise ValueError("reused source labels differ from this exact case/span/RGB draw")
    if draw.references:
        for slot, (shot, frame) in enumerate(draw.references):
            old = None if previous is None else previous["references"][slot]
            changed = old is None or require_training_media(dataset.media) != ANNOTATION_MEDIA or any(
                registry[e["name"]]["kind"] == "group" or e["name"] in unresolved for e in old["entities"])
            if changed:
                source, frames, masks = _read_reference_block(spatial_root, block)
                row = compile_reference_frame_entities(member=member, shot_index=shot, frame=frame,
                    source_frame=source_frames[slot], media=dataset.media,
                    source=source, frames=frames, masks=masks, seed=dataset.training_seed,
                    sample_index=index, reference_index=slot, max_alignment_cells=max_alignment_cells,
                    unresolved_names=unresolved)
                counts["reference_frames_recomputed"] += 1
            else:
                expected_source = source_frames[slot]
                if old["source_frame"] != expected_source:
                    raise ValueError("reused source labels differ from the actual training media source frame")
                row = {**old, "entities": [{**entity, "kind": registry[entity["name"]]["kind"],
                    "identity_status": "registered"} for entity in old["entities"]],
                    "unobserved_planned_identities": _missing_planned_identities(member, shot, old["entities"])}
                counts["reference_frames_reused"] += 1
            references.append(row)
    record = {"schema": REFERENCE_ENTITY_SCHEMA, "sample_index": index, "training_seed": dataset.training_seed,
        "case_id": draw.row.case_id,
        "span": list(draw.span), "reference_draw": [[shot + 1, frame] for shot, frame in draw.references],
        "references": references}
    _validate_training_record(record)
    counts["samples"] += 1
    counts["reference_frames"] += len(references)
    for row in references:
        counts[row["status"]] += 1
        counts["visible_entities"] += len(row["entities"])
        counts["source_points"] += sum(entity["point_cell"] is not None for entity in row["entities"])
        counts["alignment_cells"] += len(row["alignment_cells"])
        counts["supervised_reference_frames"] += bool(row["alignment_cells"])
    return record, counts


def build_reference_entities(dataset, *, spatial_root: Path, output_path: Path,
                             sample_indices: Sequence[int], max_alignment_cells: int,
                             identity_review: ReferenceIdentityReview,
                             reuse_labels_path: Path | None, workers: int) -> dict:
    """复用BlockDataset唯一draw，不另实现参考采样；仅CPU标注，不解码RGB或运行VLM。"""
    require_training_media(dataset.media)
    if output_path.exists():
        raise FileExistsError(output_path)
    if len(set(sample_indices)) != len(sample_indices):
        raise ValueError("reference entity production indices must be unique")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    reused = ({} if reuse_labels_path is None else {row["sample_index"]: row for row in _rows(reuse_labels_path)})
    if reuse_labels_path is not None and set(reused) != set(sample_indices):
        raise ValueError("reused source labels must cover the same exact sample indices")
    if type(workers) is not int or workers < 1:
        raise ValueError("source label workers must be explicitly positive")
    from contextlib import ExitStack
    import multiprocessing
    import time
    started = time.monotonic()
    staged = output_path.with_suffix(output_path.suffix + ".partial")
    worker_args = (dataset, spatial_root, identity_review, reused, max_alignment_cells)
    with ExitStack() as stack:
        if workers == 1:
            _init_reference_worker(*worker_args)
            records = map(_reference_sample, sample_indices)
        else:
            pool = stack.enter_context(multiprocessing.get_context("fork").Pool(
                workers, initializer=_init_reference_worker, initargs=worker_args))
            records = pool.imap(_reference_sample, sample_indices, chunksize=1)
        handle = stack.enter_context(staged.open("x", encoding="utf-8"))
        for record, measured in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            counts.update(measured)
    publish_file(staged, output_path)
    summary = {"schema": REFERENCE_ENTITY_SCHEMA, "output_path": str(output_path),
               "workers": workers, "production_seconds": time.monotonic() - started,
               "training_seed": dataset.training_seed,
               "media": dict(dataset.media),
               "reference_temporal_grid": dict(dataset.metadata_summary["reference_temporal_grid"]),
               "target_spatial_media": dict(dataset.metadata_summary["target_spatial_media"]),
               "sampling_manifest_path": dataset.metadata_summary["sampling_manifest_path"],
               "identity_review_path": str(identity_review.path),
               "reuse_labels_path": None if reuse_labels_path is None else str(reuse_labels_path),
               "max_alignment_cells_per_reference": max_alignment_cells, **counts}
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary
