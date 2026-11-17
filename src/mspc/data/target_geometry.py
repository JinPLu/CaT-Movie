"""同一目标RGB0的框尺寸与离散前后边；metric depth只写独立审计，不进入条件。"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mspc.data.atomic_publish import publish_file
from mspc.data.geometry_annotation import DEPTH_ORDER_MIN_LOG
from mspc.data.history_reading_labels import LABEL_SCHEMA, ShotMasks, box_center
from mspc.data.reference_entities import ReferenceIdentityReview, matched_mask_frame
from mspc.data.scene_training_manifest import ANNOTATION_MEDIA
from mspc.data.shot_targets import planned_row

TARGET_GEOMETRY_SCHEMA = "target_first_frame_geometry_v1"


def _rows(path: Path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def read_point_labels(path: Path, expected_indices: Sequence[int]) -> dict[int, dict]:
    labels = {}
    for row in _rows(path):
        index = row["sample_index"]
        if type(index) is not int or index in labels or row["schema"] != LABEL_SCHEMA:
            raise ValueError("point labels require unique sample_index and the actual history-reading schema")
        labels[index] = row
    if not set(expected_indices) <= set(labels):
        raise ValueError("point labels do not cover the requested actual sample indices")
    return labels


def _frame_zero_depths(depth: Mapping[str, Any], masks: Mapping[str, Any], shot_index: int,
                       names: Sequence[str]) -> tuple[dict[str, float], dict[str, str], float | None]:
    if (depth["unit"] != "relative_per_shot"
            or depth["sampling"] != "depth.mp4 decoded and dequantized; code map nearest-resized to the mask grid; median over the mask"):
        raise ValueError("target geometry requires the existing per-shot distance-depth producer")
    sid = f"s{shot_index + 1:02d}"
    error = depth["max_relative_error"].get(sid)
    if error is None:
        return {}, {name: "missing_depth_quantization_metadata" for name in names}, None
    if type(error) not in (int, float) or not math.isfinite(error) or error < 0:
        raise ValueError("depth quantization error must be finite and nonnegative")
    source = depth["shots"].get(sid, {})
    tracks = masks["shots"][sid]["tracks"]
    values, unknown = {}, {}
    for name in names:
        # 多个同名track不能随意取最近一个，也不能跨时间取中位数补RGB0。
        actual = [(key, track) for key, track in tracks.items() if track["name"] == name
                  and "0" in track["frames"]]
        if len(actual) != 1:
            unknown[name] = "missing_frame0_depth" if not actual else "multiple_frame0_tracks"
            continue
        key, track = actual[0]
        record = source.get(key)
        if record is None or record["name"] != name or "0" not in record["median_depth"]:
            unknown[name] = "missing_frame0_depth"
            continue
        value = record["median_depth"]["0"]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            unknown[name] = "invalid_frame0_depth"
            continue
        values[name] = float(value)
    return values, unknown, float(error)


def compile_target_geometry(*, member: Mapping[str, Any], span: Sequence[int], point_labels: Mapping[str, Any],
                            source: Mapping[str, Any], frames: Mapping[str, Any], masks: Mapping[str, Any],
                            depth: Mapping[str, Any], unresolved_names: frozenset[str]) -> tuple[list[dict], list[dict]]:
    """与既有point starts逐项核对后，只读同镜首的mask/depth；缺失显式unknown。"""
    if (point_labels["case_id"] != member["case_id"] or point_labels["key"] != member["key"]
            or point_labels["span"] != list(span)):
        raise ValueError("target point labels differ from the actual case/key/span")
    if [row["shot_index"] for row in point_labels["starts"]] != list(range(*span)):
        raise ValueError("target point labels must cover every actual target shot")
    minimum = point_labels["rules"]["start_min_area"]
    if type(minimum) not in (int, float) or not 0 < minimum < 1:
        raise ValueError("the reused point labels must declare their original start_min_area")
    registry = {entry["name"]: entry for entry in member["annotation"]["characters"]}
    reader = ShotMasks(masks)
    output, audit = [], []
    for old in point_labels["starts"]:
        index = old["shot_index"]
        first_source = planned_row({**member["scene"], "shots": member["shots"]}, index,
                                   media=ANNOTATION_MEDIA)["source_frame_indices_by_shot"][0][0]
        source_frame, errors = matched_mask_frame(member=member, shot_index=index, source_frame=first_source,
                                                  source=source, frames=frames, masks=masks)
        names = [entry["name"] for entry in member["annotation"]["shots"][index]["present"]
                 if entry["name"] in registry]
        if [entry["name"] for entry in old["characters"]] != names:
            raise ValueError("target point identity order differs from the actual shot's registered characters")
        entities = []
        blocked = {}
        for previous in old["characters"]:
            name = previous["name"]
            mask = reader.mask(index, name, 0)
            point = box_center(mask) if mask is not None and mask.any() and mask.mean() >= minimum else None
            if point != previous["point_2d"]:
                raise ValueError(f"target RGB0 point mismatch for {member['case_id']} shot {index + 1} {name}")
            box, size, valid = None, None, False
            reason = ("annotation_error" if errors else "identity_unresolved" if name in unresolved_names
                      else "missing_frame0_mask" if mask is None or not mask.any()
                      else "mask_below_start_min_area" if point is None else "known_frame0_mask")
            if reason == "known_frame0_mask":
                ys, xs = np.nonzero(mask)
                box = [int(xs.min()) / 832, int(ys.min()) / 480,
                       (int(xs.max()) + 1) / 832, (int(ys.max()) + 1) / 480]
                size = [box[2] - box[0], box[3] - box[1]]
                valid = True
            else:
                blocked[name] = reason
            if registry[name]["kind"] == "group":
                blocked[name] = "group_depth_unknown"
            entities.append({"name": name, "point_2d": point, "box_2d": box,
                             "size": size, "size_valid": valid, "reason": reason})
        depths, depth_unknown, relative_error = _frame_zero_depths(depth, masks, index, names)
        pairs, unknown_pairs = [], []
        # 两个量化距离的log误差相加；仍沿用现有0.1 log-gap，不新调判定常数。
        margin = None if relative_error is None else DEPTH_ORDER_MIN_LOG + 2 * math.log1p(relative_error)
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                reason = (blocked.get(first) or blocked.get(second) or depth_unknown.get(first)
                          or depth_unknown.get(second))
                if reason is None:
                    ratio = math.log(depths[second] / depths[first])
                    if abs(ratio) > margin:
                        near, far = (first, second) if ratio > 0 else (second, first)
                        pairs.append({"near": near, "far": far})
                        continue
                    reason = "frame0_depth_gap_below_margin"
                unknown_pairs.append({"first": first, "second": second, "reason": reason})
        output.append({"shot_index": index, "frame": 0, "characters": entities,
                       "front_of": pairs, "relation_unknown": unknown_pairs})
        audit.append({"shot_index": index, "frame": 0, "source_frame": source_frame,
                      "frame0_distance_depth": depths, "depth_unknown": depth_unknown,
                      "max_relative_quantization_error": relative_error,
                      "log_gap_margin": margin, "annotation_errors": errors})
    return output, audit


def _validate_record(record):
    if set(record) != {"schema", "sample_index", "training_seed", "case_id", "span", "starts"}:
        raise ValueError("target geometry sample has unsupported or missing fields")
    if record["schema"] != TARGET_GEOMETRY_SCHEMA or type(record["sample_index"]) is not int:
        raise ValueError("target geometry sample has invalid schema/index")
    if [row["shot_index"] for row in record["starts"]] != list(range(*record["span"])):
        raise ValueError("target geometry must cover exactly the target span")
    for row in record["starts"]:
        if set(row) != {"shot_index", "frame", "characters", "front_of", "relation_unknown"} or row["frame"] != 0:
            raise ValueError("target geometry must describe actual RGB0")
        names = [entry["name"] for entry in row["characters"]]
        if len(names) != len(set(names)):
            raise ValueError("target geometry has duplicate identities")
        for entity in row["characters"]:
            if set(entity) != {"name", "point_2d", "box_2d", "size", "size_valid", "reason"}:
                raise ValueError("target geometry entity requires exact size/valid/point fields")
            if type(entity["size_valid"]) is not bool:
                raise ValueError("target size validity must be boolean")
            if entity["size_valid"]:
                box, size = entity["box_2d"], entity["size"]
                if (entity["point_2d"] is None or not isinstance(box, list) or len(box) != 4
                        or not isinstance(size, list) or len(size) != 2
                        or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 1 for v in size)
                        or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1
                        or size != [box[2] - box[0], box[3] - box[1]]):
                    raise ValueError("valid target size must come from the same frame0 box and point")
            elif entity["size"] is not None or entity["box_2d"] is not None:
                raise ValueError("unknown target size cannot supply substituted dimensions")
        known = set()
        for pair in row["front_of"]:
            if set(pair) != {"near", "far"} or pair["near"] not in names or pair["far"] not in names or pair["near"] == pair["far"]:
                raise ValueError("front_of must be an actual directed pair of target identities")
            key = frozenset(pair.values())
            if key in known:
                raise ValueError("front_of repeats or reverses a known pair")
            known.add(key)
        unknown = set()
        for pair in row["relation_unknown"]:
            if set(pair) != {"first", "second", "reason"} or pair["first"] not in names or pair["second"] not in names:
                raise ValueError("unknown target relations require actual names and an explicit reason")
            key = frozenset((pair["first"], pair["second"]))
            if len(key) != 2 or key in known or key in unknown:
                raise ValueError("known/unknown target relations overlap or repeat")
            unknown.add(key)
        if len(known | unknown) != len(names) * (len(names) - 1) // 2:
            raise ValueError("target geometry must explicitly account for every known or unknown pair")


class TargetGeometry:
    def __init__(self, path: Path, expected_indices: Sequence[int]):
        self.records = {}
        for record in _rows(path):
            _validate_record(record)
            index = record["sample_index"]
            if index in self.records:
                raise ValueError("duplicate target geometry sample_index")
            self.records[index] = record
        if set(self.records) != set(expected_indices):
            raise ValueError("target geometry must cover the exact sample indices")

    def for_sample(self, index: int, *, case_id: str, span: Sequence[int], training_seed: int,
                   starts: Sequence[tuple[int, str, Any]]) -> list[dict]:
        record = self.records[index]
        if record["case_id"] != case_id or record["span"] != list(span) or record["training_seed"] != training_seed:
            raise ValueError("target geometry differs from the actual case/span/training seed")
        actual = [(row["shot_index"], entity["name"], entity["point_2d"])
                  for row in record["starts"] for entity in row["characters"]]
        if actual != list(starts):
            raise ValueError("target geometry and point token starts differ in identity/order/RGB0 point")
        return record["starts"]


_TARGET_WORKER_STATE = None


def _init_target_worker(dataset, spatial_root, points, identity_review):
    global _TARGET_WORKER_STATE
    _TARGET_WORKER_STATE = (dataset, spatial_root, points, identity_review)


@lru_cache(maxsize=16)
def _read_target_block(spatial_root, key):
    return tuple(json.loads((spatial_root / key / name).read_text())
                 for name in ("source.json", "frames.json", "masks.json", "instance_depth.json"))


def _target_sample(index):
    if _TARGET_WORKER_STATE is None:
        raise RuntimeError("target geometry worker was not initialized")
    dataset, spatial_root, points, identity_review = _TARGET_WORKER_STATE
    draw = dataset.draw(index)
    member = dataset.members[draw.row.case_id]
    source, frames, masks, depth = _read_target_block(spatial_root, member["key"])
    starts, audit = compile_target_geometry(member=member, span=draw.span, point_labels=points[index],
        source=source, frames=frames, masks=masks, depth=depth,
        unresolved_names=identity_review.unresolved(member))
    record = {"schema": TARGET_GEOMETRY_SCHEMA, "sample_index": index, "training_seed": dataset.training_seed,
              "case_id": draw.row.case_id, "span": list(draw.span), "starts": starts}
    _validate_record(record)
    audit_record = {"sample_index": index, "case_id": draw.row.case_id,
                    "key": member["key"], "starts": audit}
    counts = Counter(samples=1)
    for shot in starts:
        counts["target_shots"] += 1
        counts["characters"] += len(shot["characters"])
        counts["target_points"] += sum(entity["point_2d"] is not None for entity in shot["characters"])
        counts["known_sizes"] += sum(entity["size_valid"] for entity in shot["characters"])
        counts["known_front_of_pairs"] += len(shot["front_of"])
        counts["unknown_pairs"] += len(shot["relation_unknown"])
        for pair in shot["relation_unknown"]:
            counts["unknown_" + pair["reason"]] += 1
    return record, audit_record, counts


def build_target_geometry(dataset, *, spatial_root: Path, point_labels_path: Path, identity_review: ReferenceIdentityReview,
                          output_path: Path, sample_indices: Sequence[int], workers: int) -> dict:
    if output_path.exists():
        raise FileExistsError(output_path)
    if type(workers) is not int or workers < 1:
        raise ValueError("target geometry workers must be explicitly positive")
    from contextlib import ExitStack
    import multiprocessing
    import time
    started = time.monotonic()
    points = read_point_labels(point_labels_path, sample_indices)
    counts = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged = output_path.with_suffix(output_path.suffix + ".partial")
    audit_path = output_path.with_suffix(".audit.jsonl")
    worker_args = (dataset, spatial_root, points, identity_review)
    with ExitStack() as stack:
        if workers == 1:
            _init_target_worker(*worker_args)
            records = map(_target_sample, sample_indices)
        else:
            pool = stack.enter_context(multiprocessing.get_context("fork").Pool(
                workers, initializer=_init_target_worker, initargs=worker_args))
            records = pool.imap(_target_sample, sample_indices, chunksize=1)
        stream = stack.enter_context(staged.open("x"))
        audit_stream = stack.enter_context(audit_path.open("x"))
        for record, audit_record, measured in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            audit_stream.write(json.dumps(audit_record, ensure_ascii=False) + "\n")
            counts.update(measured)
    publish_file(staged, output_path)
    summary = {"schema": TARGET_GEOMETRY_SCHEMA, "output_path": str(output_path), "audit_path": str(audit_path),
               "workers": workers, "production_seconds": time.monotonic() - started,
               "point_labels_path": str(point_labels_path), "identity_review_path": str(identity_review.path),
               "depth_order_min_log": DEPTH_ORDER_MIN_LOG, **counts}
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary
