"""10-18 导出拒收中的真实源记录；只检验有证据的确定转换。"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from mspc.data.storyboard import entity_references, from_annotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools/data'))
from validate_annotations import block_record, normalise_by_shot  # noqa: E402


def case(key):
    return json.loads((ROOT / 'tests/fixtures/planner_source_v18' / f'{key}.json').read_text())


@pytest.mark.parametrize('key', ['cinedance-009622-b0000', 'cinedance-010303-b0001', 'cinedance-008887-b0002'])
def test_real_empty_values_compile_without_losing_empty_hands(key):
    row = case(key)
    answer = row['annotation']
    empty_hands = [(i, e['name']) for i, shot in enumerate(answer['shots']) for e in shot['present']
                   if e['name'].startswith('Character ') and e.get('holding') == []]
    actions = [s['action'] for s in answer['shots']]
    assert normalise_by_shot(answer, 17)
    from_annotation(answer, row['frame_counts'], start=0, end=len(answer['shots']))
    for i, name in empty_hands:
        assert next(e for e in answer['shots'][i]['present'] if e['name'] == name)['holding'] == []
    assert [s['action'] for s in answer['shots']] == actions
    assert normalise_by_shot(answer, 17) == []


def test_raw_cup_holding_is_preserved_for_semantic_review():
    row = case('cinedance-008887-b0002')
    contract = json.loads((ROOT / 'configs/data/block_annotation_v18.json').read_text())
    result = block_record(row['raw_answer'], contract, len(row['frame_counts']))
    entry = next(e for e in result['annotation']['shots'][1]['present'] if e['name'] == 'Object 2')
    raw_entry = next(e for e in json.loads(row['raw_answer'])['shots'][1]['present'] if e['name'] == 'Object 2')
    assert entry['holding'] == raw_entry['holding']
    assert result['status'] == 'invalid'
    assert result['raw_answer'] == row['raw_answer']


@pytest.mark.parametrize('key,number', [
    ('film-003415-b0000', 2), ('film-004844-b0000', 1), ('film-005782-b0000', 7),
])
def test_real_robot_or_claw_holding_is_preserved_and_rejected(key, number):
    row = case(key)
    contract = json.loads((ROOT / 'configs/data/block_annotation_v18.json').read_text())
    before = row['annotation']['shots'][number - 1]
    result = block_record(json.dumps(row['annotation']), contract, row['shot_count'])
    after = result['annotation']['shots'][number - 1]
    assert after == before
    assert result['status'] == 'invalid'
    assert any('holding of Object' in e for e in result['shot_errors'][number])
    assert number not in result['deletions']


@pytest.mark.parametrize('key,kind,old,new', [
    ('film-001479-b0006', 'objects', 'Object 4', 'Object 3'),
    ('film-006160-b0002', 'characters', 'Character 5', 'Character 4'),
])
def test_real_number_gaps_rename_story_and_every_reference(key, kind, old, new):
    row = case(key)
    before = deepcopy(row['annotation'])
    answer = row['annotation']
    assert ('登记编号连续化并同步所有引用', None) in normalise_by_shot(answer, 17)
    assert old not in entity_references(answer)
    assert new in entity_references(answer)
    assert answer['story'] == json.loads(json.dumps(before['story']).replace(old, new))
    assert [e['appearance'] for e in answer[kind]] == [e['appearance'].replace(old, new) for e in before[kind]]
    from_annotation(answer, row['frame_counts'], start=0, end=len(answer['shots']))


def test_real_subspan_keeps_indirect_registry_reference():
    row = case('film-004395-b0002')
    before = deepcopy(row['annotation'])
    board = from_annotation(row['annotation'], row['frame_counts'], start=6, end=10)
    stone = next(e for e in board['objects'] if e['appearance'].endswith(' stone'))
    referenced = entity_references(stone['appearance'])
    assert len(referenced) == 1
    assert any(e['name'] == referenced[0] for e in board['objects'])
    assert row['annotation'] == before


def test_real_visibility_projects_only_that_field_and_keeps_source():
    row = case('cinedance-008920-b0000')
    before = deepcopy(row['annotation'])
    board = from_annotation(row['annotation'], row['frame_counts'], start=0, end=len(row['frame_counts']))
    assert any('visibility' in e for s in before['shots'] for e in s['present'])
    for source, target in zip(before['shots'], board['shots'], strict=True):
        assert target['present'] == [{k: v for k, v in e.items() if k != 'visibility'} for e in source['present']]
    assert row['annotation'] == before


def test_real_final_expression_is_preserved_for_review():
    row = case('cinedance-009700-b0001')
    before = deepcopy(row['annotation'])
    normalise_by_shot(row['annotation'], 17)
    assert row['annotation']['shots'][5]['changes'] == before['shots'][5]['changes']
    assert any(c.get('end_state') == 'serious expression' for c in before['shots'][5]['changes'])
    with pytest.raises(ValueError, match='end_state'):
        from_annotation(row['annotation'], row['frame_counts'], start=0, end=len(row['frame_counts']))
