"""真实冻结失败行：标注拒收保留原文，编译按显式释放关系补正面终态。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest
from mspc.data.storyboard import (visible_result_wording_errors,
                                require_visible_result_wording, compile_storyboard)
from mspc.data.planner_dataset import PLANNER_SYSTEM
ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / 'tests/fixtures/planner_visible_result'
sys.path.insert(0, str(ROOT / 'tools/data'))
from validate_annotations import block_record

@pytest.mark.parametrize('row', json.loads((FIXTURE / 'real-candidates.json').read_text()), ids=lambda r:r['sample_id'])
def test_actual_holding_release_without_producer_state_gets_a_visible_compiled_endpoint(row):
    # 标注生产边界仍要求原生产者给可见终态；本轮编译修复则保留明确holding释放并补人物空手结果。
    board = row['storyboard']
    before = deepcopy(board)
    errors = visible_result_wording_errors(board)
    assert errors and all('holding becomes empty' in e['reason'] for e in errors)
    with pytest.raises(ValueError, match='holding becomes empty'):
        require_visible_result_wording(board)
    trial = deepcopy(board)
    compiled, report = compile_storyboard(trial, target_seconds=None, fps=None, cut_times_seconds=None, stated=None)
    assert not any('holding becomes empty' in d['reason'] for d in report['dropped_fields'])
    assert any(f['rule'] == 'empty_hands_made_visible' for f in report['holding_carried'])
    for opening, ending in zip(board['shots'], compiled['shots'], strict=True):
        for change in opening['changes']:
            if change['what'] == 'holding' and change.get('before') and change['after'] == []:
                assert any(c['name'] == change['name'] and c['what'] == 'holding' and c['after'] == []
                           for c in ending['changes'])
                assert any(c['name'] == change['name'] and c['what'] == 'state' and 'empty-handed' in c['after']
                           for c in ending['changes'])
    require_visible_result_wording(compiled)
    assert board == trial == before

def test_actual_coat_result_is_rejected_before_normalization():
    board = json.loads((FIXTURE / 'coat-source.json').read_text())[0]['annotation']
    raw = json.dumps(board)
    result = block_record(raw, json.loads((ROOT / 'configs/data/block_annotation_v18.json').read_text()),len(board['shots']))
    assert result['status'] == 'invalid'
    assert result['raw_answer'] == raw
    assert any('positive visible result' in e['reason'] for e in result['wording_errors'])

def test_planner_worked_example_obeys_installed_contract():
    example = json.loads(PLANNER_SYSTEM.split('Output:\n')[1].split('\n')[0])
    require_visible_result_wording(example)
    compile_storyboard(example, target_seconds=12, fps=16, cut_times_seconds=None, stated=None)
