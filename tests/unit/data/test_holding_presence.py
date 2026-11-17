"""2026-10-14 全量普查中的真实漏项和编号碰撞。"""
import copy
import json
from pathlib import Path

from mspc.data.storyboard import complete_holding_presence, holding_targets
from tools.data.validate_annotations import block_record, mend_block, structure_errors

ROOT = Path(__file__).parents[3]
ROWS = json.loads((ROOT/'tests/fixtures/block_annotation_v18/holding_presence.json').read_text())
CONTRACT = json.loads((ROOT/'configs/data/block_annotation_v18.json').read_text())


def test_registered_holding_targets_are_expanded_without_changing_identity():
    kinds = set()
    for row in ROWS:
        answer = copy.deepcopy(row['annotation'])
        registry = copy.deepcopy({k: answer[k] for k in ('characters', 'objects', 'places')})
        known = {e['name']: k for k, entries in registry.items() for e in entries}
        for shot in answer['shots']:
            before = copy.deepcopy(shot['present'])
            added = complete_holding_presence(shot, known)
            assert shot['present'] == before + [{'name': name} for name in added]
            assert set(holding_targets(shot)) <= {e['name'] for e in shot['present']}
            assert complete_holding_presence(shot, known) == []
            kinds.update(e.get('kind', 'object') for entries in registry.values() for e in entries if e['name'] in added)
        assert {k: answer[k] for k in registry} == registry
        assert structure_errors(answer, CONTRACT['labels'], len(answer['shots']), 17) == ([], {})
    assert {'person', 'animal', 'group', 'object'} <= kinds


def test_free_text_registration_cannot_capture_dangling_number():
    row = next(r for r in ROWS if r['key'] == 'film-008711-b0016')
    answer = json.loads(row['raw_answer'])
    record = block_record(row['raw_answer'], CONTRACT, len(answer['shots']))
    objects = {e['name']: e['appearance'] for e in record['annotation']['objects']}
    assert 'Object 3' not in objects
    assert objects['Object 4'] == 'vehicle controls'
    assert objects['Object 5'] == 'a small device'
    assert record['deletions'] == {}
    assert record['status'] == 'invalid'
    assert record['block_errors'] == ['object identity collision requires review of the raw answer']


def test_existing_valid_registry_does_not_hide_missing_presence():
    row = next(r for r in ROWS if r['key'] == 'film-007745-b0003')
    answer = copy.deepcopy(row['annotation'])
    block, errors = structure_errors(answer, CONTRACT['labels'], len(answer['shots']), 17)
    assert not block and any('holding targets not present' in e for e in errors[3])
    fixes = mend_block(answer, CONTRACT['labels'])
    assert {'rule': '持有实体补在场名单', 'shot': 3, 'to': 'Character 3'} in fixes
    assert structure_errors(answer, CONTRACT['labels'], len(answer['shots']), 17) == ([], {})
    assert mend_block(answer, CONTRACT['labels']) == []
