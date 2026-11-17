"""2026-10-16 全量复查：开镜记录写成镜内变化之后，排版函数曾整句丢掉这条变化。夹具是修补之前台账的真实行。"""
import copy
import json
from pathlib import Path

import pytest

from mspc.data.shot_text import shot_text, shot_text_with_entity_spans
from mspc.data.storyboard import (align_board_openings, align_opening_with_changes, compile_storyboard, from_annotation,
                                  require_visible_result_wording)
from tools.data.validate_annotations import block_record

ROOT = Path(__file__).parents[3]
ROWS = {row['case']: row for row in json.loads((ROOT/'tests/fixtures/block_annotation_v18/opening_after_change.json').read_text())['rows']}
CONTRACT = json.loads((ROOT/'configs/data/block_annotation_v18.json').read_text())


def _outfits(annotation):
    return {entry['name']: entry.get('outfit') for entry in annotation['characters']}


def _aligned(case):
    annotation = copy.deepcopy(ROWS[case]['annotation'])
    fixes = {}
    for fix in align_board_openings(annotation, _outfits(annotation)):
        fixes.setdefault(fix['shot'], []).append(fix)
    return annotation, fixes


@pytest.mark.parametrize('case', ['state', 'holding', 'posture', 'outfit'])
def test_unaligned_real_rows_report_conflicts_and_use_explicit_before_on_a_copy(case):
    row = ROWS[case]
    before = copy.deepcopy(row['annotation'])
    compiled = shot_text_with_entity_spans(row['annotation'], row['shot'] - 1)
    assert compiled['opening_conflicts']
    assert any(conflict['field'] == case for conflict in compiled['opening_conflicts'])
    assert row['annotation'] == before


@pytest.mark.parametrize('case', ['state', 'holding', 'posture', 'outfit'])
def test_opening_takes_the_before_value_and_the_change_is_written_at_the_end(case):
    annotation, fixes = _aligned(case)
    number = ROWS[case]['shot']
    shot = annotation['shots'][number - 1]
    restored = [fix for fix in fixes.get(number, []) if fix['rule'] == '开镜值取变化前']
    assert restored, fixes[number]
    for fix in restored:
        entry = next(e for e in shot['present'] if e['name'] == fix['name'])
        assert entry[fix['what']] == fix['to']
        change = next(c for c in shot['changes'] if c['name'] == fix['name'] and c['what'] == fix['what'])
        assert fix['to'] == change['before']
    opening, marker, ending = shot_text(annotation, number - 1).partition('By the end of the shot, ')
    assert marker
    expected_result = {
        'state': 'Object 1 is partially open',
        'holding': 'Character 1 holds Object 1',
        'posture': 'Character 1 is entering the frame from the top left',
        'outfit': 'Character 1 wears a peach dress',
    }[case]
    assert expected_result not in opening and expected_result in ending
    assert align_board_openings(annotation, _outfits(annotation)) == []


def test_real_door_is_closed_at_the_opening_and_partially_open_at_the_end():
    annotation, _ = _aligned('state')
    text = shot_text(annotation, 0)
    assert 'is closed' in text and text.endswith('is partially open.')


def test_real_outfit_change_back_to_the_registry_default_is_kept():
    # 登记表默认是 a peach dress，开镜没写穿着；镜内脱掉围裙。开镜按 before 写明穿着，镜末写回默认。
    annotation, fixes = _aligned('outfit')
    assert any(fix['rule'] == '开镜值取变化前' and fix['what'] == 'outfit' for fix in fixes[1])
    opening, marker, ending = shot_text(annotation, 0).partition('By the end of the shot, ')
    assert marker and 'a peach dress and a beige apron' in opening
    assert 'Character 1 wears a peach dress' in ending and 'beige apron' not in ending


def test_real_change_with_identical_ends_is_dropped():
    annotation, fixes = _aligned('noop')
    number = ROWS['noop']['shot']
    assert [fix['what'] for fix in fixes[number] if fix['rule'] == '删掉没有变化的变化'] == ['holding']
    assert all(change['what'] != 'holding' for change in annotation['shots'][number - 1]['changes'])
    shot_text(annotation, number - 1)


def test_real_change_without_a_before_value_is_dropped_when_the_opening_already_shows_the_result():
    annotation, fixes = _aligned('no_before')
    number = ROWS['no_before']['shot']
    dropped = [fix for fix in fixes[number] if fix['rule'] == '开镜已是变化后且没写变化前']
    assert dropped and all(fix['value']['before'] in (None, '') for fix in dropped)
    shot_text(annotation, number - 1)


def test_real_drop_then_pick_up_in_one_shot_is_written_in_order():
    annotation, fixes = _aligned('chain')
    assert fixes.get(ROWS['chain']['shot'], []) == []
    with pytest.raises(ValueError, match='holding becomes empty'):
        require_visible_result_wording(annotation)
    # 旧真实链保留供回源；不把中间释放自动写成镜末空手。
    assert annotation['shots'][ROWS['chain']['shot'] - 1]['changes'] == ROWS['chain']['annotation']['shots'][ROWS['chain']['shot'] - 1]['changes']


def test_block_validation_records_the_alignment():
    row = ROWS['state']
    record = block_record(json.dumps(row['annotation'], ensure_ascii=False), CONTRACT, len(row['annotation']['shots']))
    assert record['status'] == 'ok'
    assert any(fix['rule'] == '开镜值取变化前' and fix['shot'] == row['shot'] for fix in record['repairs'])
    shot_text(record['annotation'], row['shot'] - 1)


def test_planner_output_withholds_a_self_contradicting_opening():
    # 规划路径：在场条目写成变化后、before 另写一个值时，编译器不替规划器选边，在场值与 before 都不写进开镜。
    annotation = ROWS['posture']['annotation']
    board = from_annotation(annotation, [81] * len(annotation['shots']), start=0, end=len(annotation['shots']))
    compiled, report = compile_storyboard(board, target_seconds=None, fps=None, cut_times_seconds=None, stated=None)
    assert 'opening_aligned_with_changes' not in report
    fix = next(fix for fix in report['opening_withheld'] if fix['shot'] == ROWS['posture']['shot'])
    shot = compiled['shots'][fix['shot'] - 1]
    entry = next(entry for entry in shot['present'] if entry['name'] == fix['name'])
    change = next(change for change in shot['changes'] if change['name'] == fix['name'] and change['what'] == fix['what'])
    assert fix['what'] not in entry and change.get('before') is None and change['after'] == fix['after']
    shot_text(compiled, ROWS['posture']['shot'] - 1)


@pytest.mark.parametrize('key', ['film-004284-b0002', 'film-005434-b0001'])
def test_rebuild_cleans_invalid_endpoints_before_aligning_the_opening(key):
    rows = json.loads((ROOT/'tests/fixtures/block_annotation_v18/opening_rebuild_failures.json').read_text())['rows']
    source = next(row for row in rows if row['key'] == key)
    annotation = copy.deepcopy(source['annotation'])
    annotation['shots'][0]['shot'] = 1  # 保留真实镜内内容，只把独立测试切片编号为 1。
    record = block_record(json.dumps(annotation, ensure_ascii=False), CONTRACT, 1)
    assert record['status'] == 'ok'
    assert record['annotation']['shots'][0]['changes'] == []
    assert record['deletions'][1]
    if key == 'film-004284-b0002':
        assert any(fix['rule'] == '删掉没有变化的变化' for fix in record['repairs'])
    else:
        assert any(item['path'] == 'changes[0]' for item in record['deletions'][1])
    assert shot_text(record['annotation'], 0)
