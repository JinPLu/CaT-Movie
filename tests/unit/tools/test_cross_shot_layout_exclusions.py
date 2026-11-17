"""跨镜布局读数读冻结清单的排除镜：排除镜不出题，涉及排除镜的镜对跳过并计数（冻结修订 4 真实清单）。"""
from __future__ import annotations

from tools.run.cross_shot_layout import frozen_shot_exclusions, q2_readout, q4_readout, same_location_pairs
from tests.support.paths import REPO_ROOT

FREEZE_R4 = REPO_ROOT / "configs/benchmark/script_state_bench_evaluation_freeze_v1_r4.json"


def test_real_freeze_dev_exclusions():
    excluded = frozen_shot_exclusions(FREEZE_R4)
    dev = {("lsh-002", "s05"), ("lsh-009", "s01"), ("lsh-021", "s04"), ("lsh-021", "s10"), ("lsh-025", "s04"),
           ("lsh-025", "s08")}
    found = {(story, shot) for story, shots in excluded.items() for shot in shots}
    assert dev <= found


def _story(shot_ids):
    return {"shots": [{"shot_id": sid, "location_id": "Location 1", "animate": ["Subject 1", "Subject 2"]}
                      for sid in shot_ids]}


def test_same_location_pairs_skip_pairs_touching_excluded_shot():
    excluded = frozen_shot_exclusions(FREEZE_R4)["lsh-002"]
    pairs, skipped = same_location_pairs(_story(["s04", "s05", "s06"]), min_common=2, excluded=excluded)
    assert [(i, j) for i, j, _ in pairs] == [(0, 2)]
    assert skipped == 2


def test_q2_and_q4_skip_excluded_shots():
    excluded = frozen_shot_exclusions(FREEZE_R4)
    gold = {"lsh-002": {"s04": {"layout": {"Subject 1": {"orientation": "screen-left"}}},
                        "s05": {"layout": {"Subject 1": {"orientation": "away"}}}}}
    crops = [{"instance_id": f"lsh-002|{sid}|Subject 1", "story_id": "lsh-002", "shot_id": sid, "subject": "Subject 1",
              "primary_flipped": False} for sid in ("s04", "s05")]
    answers = {"answers": {c["instance_id"]: {"orig": "screen-left", "flip": "screen-right"} for c in crops}}
    q2 = q2_readout(gold, crops, [answers], None, excluded=excluded)
    assert q2["instances"] == 1 and q2["excluded_instances_skipped"] == 1
    items = [{"item_id": "a", "story": "lsh-002", "shots": ["s04", "s05"], "subject": "Subject 1"},
             {"item_id": "b", "story": "lsh-002", "shots": ["s03", "s04"], "subject": "Subject 1"}]
    oracle = {"stories": {"lsh-002": {sid: {"subjects": {"Subject 1": {"head_mid": [0, 0, 10, 10]}}}
                                      for sid in ("s03", "s04", "s05")}}}
    q4 = q4_readout(items, {"a": "same", "b": "same"}, oracle, excluded=excluded)
    assert q4["items"] == 1 and q4["excluded_items_skipped"] == 1
