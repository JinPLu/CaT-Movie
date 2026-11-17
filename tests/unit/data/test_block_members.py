"""真实块切片的成员连接、断点及缓存未知语义；不造训练资产。"""
import copy
import json
from pathlib import Path

import pytest

from mspc.core.draws import unit_draw
from mspc.data import sampling_ledger as sl
from mspc.data.block_members import coverage, members, select_quality_members
from mspc.data.film_split import film_group, validation_group
from mspc.data.shot_text import UMT5_PROMPT_TOKEN_BUDGET, reference_text, shot_text

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests/fixtures/sampling_ledger"


def inputs():
    blocks = list(sl.read_blocks(FIXTURE / "blocks.jsonl"))
    annotations = sl.read_annotations(FIXTURE / "block-annotations.jsonl")
    exclusions = sl.read_eval_exclusions(FIXTURE / "eval-exclude-shots.jsonl")
    shots = [s for b in blocks for s in sl.shot_rows(
        b, annotations.get(b["key"]), eval_exclusions=exclusions,
        target_index=None, reference_frames=None, spatial=None)]
    return blocks, annotations, shots


def export(blocks, annotations, shots):
    return list(members(blocks, annotations, shots, validation_fraction=.02, split_seed=20261012))


def test_real_rows_preserve_bad_shots_and_source_identity():
    blocks, annotations, shots = inputs()
    rows = export(blocks, annotations, shots)
    by_key = {b["key"]: b for b in blocks}
    assert sum(s["usable"] for r in rows for s in r["shots"]) == sum(s["usable"] for s in shots)
    assert any(not s["usable"] for r in rows for s in r["shots"])
    for r in rows:
        original = by_key[r["key"]]
        assert r["annotation"] == annotations[r["key"]]["annotation"]
        assert len(r["shots"]) == original["scene"]["shot_count"]
        assert r["scene"]["source_video_path"] == original["scene"]["source_video_path"]
        for a, b in zip(r["shots"], original["scene"]["shots"], strict=True):
            assert sl.shot_key(a) == sl.shot_key(b)
            assert a["frames"] == b["render_frames"]


def test_real_unknown_cache_is_not_missing_or_ready():
    result = coverage(export(*inputs()))["counts"]
    assert result["target_unknown"] == result["reference_unknown"] == result["usable_shots"]
    assert "reference_missing" not in result
    assert "reference_manifest_present" not in result


def test_split_preserves_existing_window_policy_and_order_independence():
    blocks, annotations, shots = inputs()
    rows = export(blocks, annotations, shots)
    assert {r["key"]:r["split"] for r in rows} == {
        r["key"]:r["split"] for r in export(list(reversed(blocks)), annotations, list(reversed(shots)))}
    for r in rows:
        group = film_group(r["scene"]["film_key"], r["scene"]["source_video_path"])
        assert (r["split"] == "val") == (unit_draw("validation_group", 20261012, group) < .02)
        assert validation_group(group, fraction=.02, seed=20261012) == (r["split"] == "val")


def test_real_ledger_duplicate_missing_and_wrong_identity_rejected():
    blocks, annotations, shots = inputs()
    with pytest.raises(ValueError, match="duplicate ledger"):
        export(blocks, annotations, shots + [shots[0]])
    with pytest.raises(KeyError):
        export(blocks, annotations, shots[1:])
    changed = copy.deepcopy(shots)
    changed[0]["source_shot_index"] += 1
    with pytest.raises(ValueError, match="identity or frames"):
        export(blocks, annotations, changed)


def test_true_asset_absence_differs_from_unknown_and_partial_candidates_reject():
    blocks, annotations, shots = inputs()
    # Boundary mutation of real rows: an available manifest has no matching entries.
    for shot in shots:
        shot.update(target_latents=False, reference_frames=0)
    rows = export(blocks, annotations, shots)
    c = coverage(rows)["counts"]
    assert c["target_missing"] == c["reference_missing"] == c["usable_shots"]
    assert c["missing_candidate_frames"] == c["required_candidate_frames"]
    target = next(s for r in rows for s in r["shots"] if s["usable"])
    target["reference_frames"] = 10**6
    with pytest.raises(ValueError, match="candidate count differs"):
        coverage(rows)


def test_quality_selection_preserves_real_breakpoints_and_split():
    original = export(*inputs())
    rows = list(select_quality_members(original, min_laion_aesthetic=4.5, min_mean_luma=0, min_dover_technical=-.149,
                                      token_counter=lambda text: 1))
    before = {r["key"]: r for r in original}
    assert rows
    for row in rows:
        old = before[row["key"]]
        assert row["split"] == old["split"]
        assert len(row["shots"]) == len(old["shots"])
        for shot, source in zip(row["shots"], old["shots"], strict=True):
            assert shot["shot"] == source["shot"]
            assert shot["usable"] == (source["usable"] and source["laion_aesthetic"] >= 4.5
                                      and source["dover_technical"] >= -.149)


def _overflow_case(kind):
    case = next(row for row in json.loads((ROOT / "tests/fixtures/block_text_overflow.json").read_text())
                if row["kind"] == kind)
    # 此旧文字夹具不含亮度；受控非暗值仅隔离文字边界，不表示像素测量。
    for shot in case["member"]["shots"]: shot["mean_luma"] = 20
    return case


def _measured_counts(case):
    role = case['reference_role_measurement']
    assert role['text'] == reference_text()
    counts = {reference_text(): role['umt5_tokens']}
    for position, row in enumerate(case["measured_token_counts"]):
        text = shot_text(case["member"]["annotation"], position)
        assert text not in counts or counts[text] == row['target']
        counts[text] = row['target']
    return counts


@pytest.mark.parametrize("kind", ["long_target_prompt", "long_entity_registry"])
def test_real_text_overflow_is_a_breakpoint_and_keeps_annotation_and_neighbors(kind):
    case = _overflow_case(kind)
    member = case["member"]
    original = copy.deepcopy(member)
    counts = _measured_counts(case)
    selected, = select_quality_members([member], min_laion_aesthetic=4.5, min_mean_luma=0, min_dover_technical=-.149,
                                       token_counter=counts.__getitem__)
    assert member == original
    assert selected["annotation"] == original["annotation"]
    assert selected["split"] == original["split"]
    assert len(selected["shots"]) == len(original["shots"])
    for position, (shot, target_counts) in enumerate(zip(selected["shots"], case["measured_token_counts"], strict=True)):
        measured = {**target_counts, 'reference': case['reference_role_measurement']['umt5_tokens']}
        assert shot["shot"] == original["shots"][position]["shot"]
        assert shot["text_token_counts"] == measured
        assert shot["usable"] == all(n <= UMT5_PROMPT_TOKEN_BUDGET for n in measured.values())
        for role, count in measured.items():
            if count > UMT5_PROMPT_TOKEN_BUDGET:
                assert f"{role}_umt5_tokens={count}>{UMT5_PROMPT_TOKEN_BUDGET}" in shot["unusable_reasons"]
    assert not selected["shots"][case["position"]]["usable"]
    assert any(shot["usable"] for shot in selected["shots"])


@pytest.mark.parametrize("kind", ["long_target_prompt", "long_entity_registry"])
@pytest.mark.parametrize("token_count", [512, 513])
def test_controlled_target_token_boundary_on_real_text_with_fixed_reference_role(kind, token_count):
    case = _overflow_case(kind)
    counts = {text: min(count, UMT5_PROMPT_TOKEN_BUDGET) for text, count in _measured_counts(case).items()}
    # 只控制真实目标文字的边界计数；固定角色句沿用真实测得的短句计数，不造长常量。
    counts[shot_text(case["member"]["annotation"], case["position"])] = token_count
    selected, = select_quality_members([case["member"]], min_laion_aesthetic=4.5, min_mean_luma=0,
                                       min_dover_technical=-.149, token_counter=counts.__getitem__)
    assert selected["shots"][case["position"]]["usable"] == (token_count == 512)


def test_existing_bad_shot_is_not_compiled_or_restored():
    case = _overflow_case("long_target_prompt")
    member = case["member"]
    position = case["position"]
    member["shots"][position].update(usable=False, unusable_reasons=["annotation_invalid"])
    counts = _measured_counts(case)
    del counts[shot_text(member["annotation"], position)]
    selected, = select_quality_members([member], min_laion_aesthetic=4.5, min_mean_luma=0, min_dover_technical=-.149,
                                       token_counter=counts.__getitem__)
    shot = selected["shots"][position]
    assert not shot["usable"] and shot["unusable_reasons"] == ["annotation_invalid"]
    assert shot["text_token_counts"] is None


def test_whole_shot_brightness_threshold_keeps_neighbor_breakpoints():
    original=export(*inputs())
    member=next(m for m in original if sum(s['usable'] for s in m['shots'])>1)
    positions=[i for i,s in enumerate(member['shots']) if s['usable'] and s['laion_aesthetic']>=4.5 and s['dover_technical']>=-.149]
    assert len(positions)>1
    before=copy.deepcopy(member)
    member['shots'][positions[0]]['mean_luma']=19.99
    member['shots'][positions[1]]['mean_luma']=20
    selected,=select_quality_members([member],min_laion_aesthetic=4.5,min_dover_technical=-.149,
                                    min_mean_luma=20,token_counter=lambda _:1)
    assert not selected['shots'][positions[0]]['usable']
    assert selected['shots'][positions[1]]['usable']
    assert len(selected['shots'])==len(before['shots'])
    assert selected['annotation']==before['annotation']
