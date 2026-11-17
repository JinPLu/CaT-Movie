"""真实RGB0几何边界：缺mask不填尺寸，group不强排深度。"""
import copy
import json
from pathlib import Path

import pytest

from mspc.data.target_geometry import TargetGeometry
from mspc.models.planner_tokens import geometry_token_features

FIXTURE = Path(__file__).parents[2] / "fixtures/renderer_entities/target-mask-and-group-size.jsonl"


def _starts(record):
    return [(row["shot_index"], entity["name"], entity["point_2d"])
            for row in record["starts"] for entity in row["characters"]]


def test_real_missing_rgb0_mask_stays_unknown_in_actual_geometry_tensor():
    reader = TargetGeometry(FIXTURE, [650, 1210])
    record = reader.records[650]
    starts = _starts(record)
    rows = reader.for_sample(650, case_id=record["case_id"], span=record["span"],
                             training_seed=record["training_seed"], starts=starts)
    features = geometry_token_features(starts, rows, max_people=64)
    first = rows[0]
    missing = next(entity for entity in first["characters"] if entity["point_2d"] is None)
    position = next(i for i, (shot, name, _) in enumerate(starts) if shot == 0 and name == missing["name"])
    assert features[position].count_nonzero() == 0
    assert first["front_of"] == [] and first["relation_unknown"][0]["reason"] == "missing_frame0_mask"


def test_real_group_can_have_size_without_a_known_depth_relation():
    reader = TargetGeometry(FIXTURE, [650, 1210])
    record = reader.records[1210]
    starts = _starts(record)
    features = geometry_token_features(starts, record["starts"], max_people=64)
    position = next(i for i, (shot, name, _) in enumerate(starts) if shot == 5 and name == "Character 4")
    assert features[position, 2] == 1 and features[position, :2].min() > 0
    assert features[position, 3:].count_nonzero() == 0
    assert all("Character 4" not in edge.values() for edge in record["starts"][-1]["front_of"])


def test_real_geometry_rejects_a_point_from_another_frame():
    reader = TargetGeometry(FIXTURE, [650, 1210])
    record = reader.records[650]
    starts = _starts(record)
    shot, name, _ = starts[0]
    starts[0] = (shot, name, [500, 500])
    with pytest.raises(ValueError, match="identity/order/RGB0 point"):
        reader.for_sample(650, case_id=record["case_id"], span=record["span"],
                          training_seed=record["training_seed"], starts=starts)


def test_real_unknown_relation_requires_a_complete_unknown_pair_account(tmp_path):
    reader = TargetGeometry(FIXTURE, [650, 1210])
    broken = copy.deepcopy(reader.records[650])
    broken["starts"][0]["relation_unknown"] = []
    path = tmp_path / "missing-pair.jsonl"
    path.write_text(json.dumps(broken) + "\n")
    with pytest.raises(ValueError, match="every known or unknown pair"):
        TargetGeometry(path, [650])
