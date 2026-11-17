"""真实块回归：换装子段、坏镜断开、三档边界和编码的标签/EOS。"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mspc.data.planner_dataset import (build_planner_pools,
                                    planner_messages, production_script, validate_planner_example)
from mspc.data.storyboard import (Storyboard, from_annotation, shot_reference_labels,
                                slice_storyboard, validate_storyboard)
from mspc.evaluation.storyboard_planner import assess_answer, evaluate_structure

FIXTURE = Path(__file__).resolve().parents[2] / 'fixtures/planner_storyboard/members.json'


def members():
    return json.loads(FIXTURE.read_text())['rows']


def get_member(key):
    return next(m for m in members() if m['key'] == key)


def example():
    member = get_member('cinedance-009335-b0001')
    pools, rejected = build_planner_pools([member], [])
    assert rejected == []
    return pools['summary'][0]


def test_real_outfit_subsegment_rebases_registry_and_keeps_prose_references():
    member = get_member('cinedance-018630-b0000')
    original = deepcopy(member)
    board = from_annotation(member['annotation'], [s['frames'] for s in member['shots']], start=3, end=5)
    assert len(board['shots']) == 2
    assert [s['shot'] for s in board['shots']] == [1, 2]
    first = board['characters'][0]
    assert first['outfit'] == 'a red collar and a straw hat'
    assert all('outfit' not in e for s in board['shots'] for e in s['present'] if e['name'] == first['name'])
    assert member == original
    validate_storyboard(board)
    # 切到人物原编号不同的镜：动作文本和结构引用必须用同一张映射。
    garden = get_member('cinedance-010742-b0001')
    raw = garden['annotation']['shots'][1]
    cut = from_annotation(garden['annotation'], [s['frames'] for s in garden['shots']], start=1, end=2)
    assert raw['present'][0]['name'] == 'Character 2'
    assert cut['shots'][0]['present'][0]['name'] == 'Character 1'
    assert cut['shots'][0]['action'] == raw['action'].replace('Character 2', '__first__').replace('Character 1', 'Character 2').replace('__first__', 'Character 1')
    # 原答这一镜开镜已写戴着园艺手套，又记一条 before 为空、after 同为手套的换装；动作句里没有戴手套。10-16 起开镜对齐删掉这条。
    assert cut['shots'][0]['changes'] == []


def test_real_three_tiers_preserve_whole_block_pairing_and_explicit_spans():
    member = get_member('cinedance-018630-b0000')
    pools, rejected = build_planner_pools([member], [{'key': member['key'], 'start': 1, 'end': 5}])
    assert pools['summary'] == pools['paragraph'] == []
    assert len(pools['shot']) == 1
    assert {r['reason'] for r in rejected} == {'whole_block_has_unusable_shot'}
    assert pools['shot'][0]['shot_span'] == [1, 5]
    assert pools['shot'][0]['split'] == member['split']
    assert len(pools['shot'][0]['storyboard']['shots']) == 4
    with pytest.raises(ValueError, match='unusable'):
        build_planner_pools([member], [{'key': member['key'], 'start': 0, 'end': 5}])
    whole = get_member('cinedance-009335-b0001')
    pools, rejected = build_planner_pools([whole], [])
    assert rejected == [] and pools['shot'] == []
    for tier in ('summary', 'paragraph'):
        row = pools[tier][0]
        assert row['script'] == production_script(script=whole['annotation']['story'][tier],
            target_seconds=sum(s['frames'] for s in whole['shots']) / 16, fps=16)
        assert row['storyboard']['characters'] == whole['annotation']['characters']
        validate_planner_example(row)


def test_real_visibility_is_projected_to_the_existing_target_contract():
    member = get_member('cinedance-010141-b0002')
    original = json.dumps(member, sort_keys=True)
    pools, rejected = build_planner_pools([member], [{'key': member['key'], 'start': 0, 'end': len(member['shots'])}])
    assert rejected == []
    assert all(len(rows) == 1 for rows in pools.values())
    assert json.dumps(member, sort_keys=True) == original
    for rows in pools.values():
        validate_planner_example(rows[0])
        assert all('visibility' not in entry for shot in rows[0]['storyboard']['shots'] for entry in shot['present'])


def test_request_is_unchanged_when_target_changes_and_json_reload_preserves_contract():
    row = example()
    before = planner_messages(row)
    row['storyboard']['shots'][0]['frame_count'] = 5
    row['storyboard']['characters'] = []
    assert planner_messages(row) == before
    request = json.loads(before[-1]['content'])
    assert request == {'tier': row['tier'], 'script': row['script']}
    row = example()
    assert json.loads(json.dumps(row)) == row
    validate_planner_example(json.loads(json.dumps(row)))


def test_frozen_v5_candidate_is_not_silently_trained_as_current_request_contract():
    # 10-20 从冻结169,349条池实际抽出的首条summary；原件不带制作要求。
    row = json.loads((FIXTURE.parent / 'frozen_v5_candidate.json').read_text())
    with pytest.raises(ValueError, match='requires an exported production duration'):
        validate_planner_example(row)
    row['script'] = production_script(script=row['script'],
        target_seconds=sum(s['frame_count'] for s in row['storyboard']['shots']) / 16, fps=16)
    # 旧合同还要求参考镜号；现在参考镜号由在场名单决定、不进目标，补上制作要求后这条语义分镜就是合格目标。
    validate_planner_example(row)


@pytest.mark.parametrize('fps,maximum', [(16, 225), (24, 337)])
def test_real_request_media_derives_frame_rate_and_full_shot_limit(fps, maximum):
    row = example()
    body = row['script'].split('\n\nUser request:\n')[1]
    row['script'] = production_script(script=body, target_seconds=12, fps=fps)
    assert f'; {fps} fps; at most {maximum} frames per shot ' in row['script']
    validate_planner_example(row)
    messages = planner_messages(row)
    assert json.loads(messages[1]['content'])['script'] == row['script']
    assert 'Read the frame rate and maximum shot duration from Production requirements.' in messages[0]['content']


def test_real_export_rejects_a_mismatched_media_limit():
    row = example()
    row['script'] = row['script'].replace('at most 225 frames', 'at most 337 frames')
    with pytest.raises(ValueError, match='production requirements differ'):
        validate_planner_example(row)


def test_missing_camera_does_not_insert_facts_and_frame_count_uses_existing_positive_contract():
    board = example()['storyboard']
    board['shots'][0].pop('camera', None)
    board['shots'][0]['frame_count'] = 1
    validated = validate_storyboard(board)
    assert 'camera' not in validated['shots'][0]
    assert validated['shots'][0]['frame_count'] == 1
    camera_schema = Storyboard.model_json_schema()['$defs']['Camera']['properties']
    assert all(field.get('default') is None for field in camera_schema.values())
    board['shots'][0]['frame_count'] = 0
    with pytest.raises(ValueError):
        validate_storyboard(board)


def test_real_public_answer_cannot_keep_unregistered_subject_aliases_in_actions():
    board = json.loads((FIXTURE.parent / 'public_v3_lsh001_shots.json').read_text())
    assert board['characters'][0]['name'] == 'Character 1'
    assert "Subject 1's facial expression" in board['shots'][0]['action']
    for shot in board['shots']:
        shot.pop('references', None)
    with pytest.raises(ValueError, match='unregistered name in storyboard: Subject 1'):
        validate_storyboard(board)


def test_reference_shots_are_derived_from_presence_and_follow_a_slice():
    board = example()['storyboard']
    assert all('references' not in shot for shot in board['shots'])
    labels = shot_reference_labels(board['shots'])
    assert labels[0] == []
    cut = slice_storyboard(board, 1, 3)
    assert [shot['shot'] for shot in cut['shots']] == [1, 2]
    cut_labels = shot_reference_labels(cut['shots'])
    assert cut_labels[0] == [] and all(ref['shot'] < shot['shot'] for shot, refs in zip(cut['shots'], cut_labels)
                                        for ref in refs)


def test_layout_validation_references_existing_shots_entities_and_observed_visibility():
    board = example()['storyboard']
    name = board['characters'][0]['name']
    board['layouts'] = [{'shot': 1, 'entities': [{'name': name, 'anchors': [
        {'time': 0.2, 'visibility': 'visible', 'box': [0.1, 0.2, 0.3, 0.8]},
        {'time': 0.8, 'visibility': 'not_visible', 'box': None},
    ]}]}]
    assert list(validate_storyboard(board))[-1] == 'layouts'
    invalid = []
    for patch in ({'time': -0.1}, {'time': float('nan')}, {'visibility': 'unknown'},
                  {'box': None}, {'box': [0.5, 0.2, 0.3, 0.8]}, {'box': [0.1, 0.2, 1.2, 0.8]}):
        bad = deepcopy(board)
        bad['layouts'][0]['entities'][0]['anchors'][0].update(patch)
        invalid.append(bad)
    bad = deepcopy(board)
    bad['layouts'][0]['entities'][0]['anchors'][1]['box'] = [0.1, 0.2, 0.3, 0.8]
    invalid.append(bad)
    bad = deepcopy(board)
    bad['layouts'][0]['entities'][0]['anchors'].reverse()
    invalid.append(bad)
    bad = deepcopy(board)
    bad['layouts'][0]['shot'] = len(board['shots']) + 1
    invalid.append(bad)
    bad = deepcopy(board)
    bad['layouts'][0]['entities'][0]['name'] = 'Character 999'
    invalid.append(bad)
    bad = deepcopy(board)
    bad['layouts'].append(deepcopy(bad['layouts'][0]))
    invalid.append(bad)
    for bad in invalid:
        with pytest.raises(ValueError):
            validate_storyboard(bad)


def test_real_subsegment_preserves_layout_ids_and_unknown_versus_checked_empty():
    from mspc.data.storyboard import slice_storyboard
    member = get_member('cinedance-010742-b0001')
    board = from_annotation(member['annotation'], [s['frames'] for s in member['shots']],
                            start=0, end=len(member['shots']))
    name = board['shots'][1]['present'][0]['name']
    board['layouts'] = [{'shot': 2, 'entities': [{'name': name, 'anchors': [
        {'time': 0.5, 'visibility': 'visible', 'box': [0.2, 0.3, 0.4, 0.8]},
    ]}]}]
    cut = slice_storyboard(board, 1, 2)
    assert cut['layouts'][0]['shot'] == 1
    assert cut['layouts'][0]['entities'][0]['name'] == cut['shots'][0]['present'][0]['name'] == 'Character 1'
    board['layouts'] = []
    assert slice_storyboard(board, 1, 2)['layouts'] == []
    del board['layouts']
    assert 'layouts' not in slice_storyboard(board, 1, 2)


def test_structure_assessment_keeps_missing_truncated_and_semantic_unknown_separate():
    row = example()
    raw = json.dumps(row['storyboard'])
    good = {'case_id': row['sample_id'], 'raw_content': raw, 'status': 'completed', 'finish_reason': 'stop'}
    result = evaluate_structure([row], [good])
    assert result['summary']['structure_counts'] == {'valid': 1}
    assert result['rows'][0]['storyboard'] == row['storyboard']
    assert assess_answer({**good, 'finish_reason': 'length'}, timing=None, stated=None)['structure_status'] == 'incomplete'
    assert assess_answer({**good, 'raw_content': raw + '\nuser: invented'}, timing=None, stated=None)['structure_status'] == 'json_error'
    assert assess_answer({**good, 'raw_content': '{"events":[]}'}, timing=None, stated=None)['structure_status'] == 'schema_error'
    assert evaluate_structure([row], [])['summary']['structure_counts'] == {'no_answer': 1}


def test_prompt_field_guide_names_every_contract_field_and_allowed_value():
    """提示里不贴机器结构定义，改写逐字段说明；字段名与可选值要与合同一致（合同改了提示没跟上，曾让提示在合同改完后仍要求人物框与参考镜号）。"""
    from mspc.data.planner_dataset import PLANNER_SYSTEM, planner_messages
    from mspc.data.storyboard import StoryboardFields

    schema = StoryboardFields.model_json_schema()
    names, values = set(), set()

    def walk(node):
        if isinstance(node, dict):
            names.update(node.get('properties', {}))
            values.update(v for v in node.get('enum', []) if isinstance(v, str))
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(schema)
    assert not {n for n in names if n not in PLANNER_SYSTEM}
    assert not {v for v in values if v not in PLANNER_SYSTEM}
    system = planner_messages({'tier': 'shot', 'script': 's1: x'})[0]['content']
    assert system == PLANNER_SYSTEM and '"$defs"' not in system
