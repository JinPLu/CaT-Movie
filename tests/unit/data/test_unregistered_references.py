"""全池 token 预检暴露的真实悬空 Object 3：生产端、分镜表和训练资格同判。"""
import copy
import json
from pathlib import Path

import pytest

from mspc.data.block_members import select_quality_members
from mspc.data.storyboard import REGISTRIES, from_annotation, unregistered_entity_references
from tools.data.validate_annotations import block_record

ROOT = Path(__file__).parents[3]
CASE = json.loads((ROOT / 'tests/fixtures/block_annotation_v18/unregistered_reference.json').read_text())
CONTRACT = json.loads((ROOT / 'configs/data/block_annotation_v18.json').read_text())


def test_source_validation_keeps_the_real_phrase_and_marks_the_shot_invalid():
    annotation = CASE['member']['annotation']
    record = block_record(json.dumps(annotation), CONTRACT, len(annotation['shots']))
    assert record['status'] == 'invalid'
    assert any('unregistered entity reference: Object 3' in error for error in record['shot_errors'][1])
    assert record['annotation']['shots'][0] == annotation['shots'][0]
    assert record['annotation']['objects'] == annotation['objects']


@pytest.mark.parametrize('end', [1, 2])
def test_whole_and_sliced_storyboards_reject_the_same_unregistered_reference(end):
    member = CASE['member']
    with pytest.raises(ValueError, match='unregistered name.*Object 3'):
        from_annotation(member['annotation'], [shot['frames'] for shot in member['shots']], start=0, end=end)


def test_training_keeps_the_bad_shot_as_a_breakpoint_without_compiling_it():
    member = copy.deepcopy(CASE['member'])
    for shot in member["shots"]: shot["mean_luma"] = 20  # 受控值隔离编号边界
    original = copy.deepcopy(member)
    known = {entry['name'] for kind in REGISTRIES for entry in member['annotation'][kind]}
    assert unregistered_entity_references(member['annotation']['shots'][0], known) == ('Object 3',)
    compiled = []
    def count(text):
        compiled.append(text)
        assert 'Object 3' not in text
        return 1
    selected, = select_quality_members([member], min_laion_aesthetic=4.5, min_mean_luma=0, min_dover_technical=-.149,
                                       token_counter=count)
    assert member == original
    assert selected['annotation'] == original['annotation']
    assert len(selected['shots']) == 2
    assert not selected['shots'][0]['usable']
    assert selected['shots'][0]['unusable_reasons'] == ['unregistered_entity_reference:Object 3']
    assert selected['shots'][0]['text_token_counts'] is None
    assert selected['shots'][1]['usable']
    assert len(compiled) == 2
