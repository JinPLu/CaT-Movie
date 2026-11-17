"""真实交杯、接布丢盆和空手状态的完整集合回归。"""
from copy import deepcopy
import json
import pytest

from mspc.data.shot_text import shot_text
from mspc.data.storyboard import holding_consistency_issues, reconcile_holding
from tests.support.paths import REPO_ROOT


CASES = {row['story_id']: row['raw_storyboard'] for row in json.loads(
    (REPO_ROOT / 'tests/fixtures/planner_holding/real_cases.json').read_text())['cases']}
PENDING_CASES = {row['block_key']: row['storyboard'] for row in json.loads(
    (REPO_ROOT / 'tests/fixtures/planner_holding/pending_review_cases.json').read_text())['cases']}


def compile_board(raw):
    """标注路径的持物修补（``from_annotation``／``slice_storyboard``／训练数据仍用它）；11-12 起规划原答不再走这里。"""
    board = deepcopy(raw)
    return board, {'holding_reconciled': reconcile_holding(board)}


def test_real_recipient_keeps_the_basin_when_receiving_the_cloth():
    raw = CASES['cinedance-012473-b0000-shot01']
    before = deepcopy(raw)
    board, report = compile_board(raw)
    change = next(c for c in board['shots'][2]['changes'] if c['name'] == 'Character 2' and c['what'] == 'holding')
    assert change['before'] == ['Object 1']
    assert change['after'] == ['Object 1', 'Object 3']
    assert any(f['rule'] == 'holding_complete_ending_set' for f in report['holding_reconciled'])
    assert raw == before
    assert reconcile_holding(board) == []
    assert holding_consistency_issues(board) == []


def test_real_missing_cup_endpoints_are_added_for_both_participants():
    raw = CASES['cinedance-012429-b0002-shot01']
    board, report = compile_board(raw)
    changes = {c['name']: c for c in board['shots'][0]['changes'] if c['what'] == 'holding'}
    assert changes['Character 2']['before'] == ['Object 1']
    assert changes['Character 2']['after'] == []
    assert changes['Character 3']['before'] == []
    assert changes['Character 3']['after'] == ['Object 1']
    assert sum(f['rule'] == 'holding_missing_action_endpoint' for f in report['holding_reconciled']) == 2
    assert any(c['what'] == 'state' and c['after'] == 'empty-handed' for c in board['shots'][0]['changes'])
    assert 'empty-handed' in shot_text(board, 0)


def test_real_empty_hands_preserve_the_other_visible_result():
    board = deepcopy(CASES['cinedance-012473-b0000-shot01-public'])
    before = next(c['after'] for c in board['shots'][2]['changes'] if c['name'] == 'Character 1' and c['what'] == 'state')
    assert before == 'with a wiped face'
    reconcile_holding(board)
    state = next(c['after'] for c in board['shots'][2]['changes'] if c['name'] == 'Character 1' and c['what'] == 'state')
    assert state == 'with a wiped face and empty-handed'
    assert reconcile_holding(board) == []


def test_real_collection_loss_without_a_bound_transfer_is_only_observed():
    board = deepcopy(CASES['cinedance-012473-b0000-shot01'])
    shot = board['shots'][2]
    shot['action'] = 'Character 2 looks down.'
    expected = deepcopy(shot['changes'])
    assert any(issue['rule'] == 'holding_ending_set_requires_review' for issue in holding_consistency_issues(board))
    reconcile_holding(board)
    assert shot['changes'] == expected


def test_real_basin_can_be_put_down_during_the_handoff():
    board = deepcopy(CASES['cinedance-012473-b0000-shot01'])
    shot = board['shots'][2]
    shot['action'] += ' Character 2 puts Object 1 on Object 2.'
    reconcile_holding(board)
    change = next(c for c in shot['changes'] if c['name'] == 'Character 2' and c['what'] == 'holding')
    assert change['after'] == ['Object 3']


def test_real_explicit_before_conflict_uses_the_consumer_time_contract_without_future_lookahead():
    board = deepcopy(CASES['cinedance-012473-b0000-shot01'])
    shot = board['shots'][2]
    shot['action'] = 'Character 2 looks down.'
    recipient = next(e for e in shot['present'] if e['name'] == 'Character 2')
    recipient['holding'] = ['Object 2']
    before = deepcopy(board)
    issues = holding_consistency_issues(board)
    assert any(issue['rule'] == 'holding_before_mismatch' for issue in issues)
    assert board == before
    fixes = reconcile_holding(board)
    assert recipient['holding'] == ['Object 1']
    assert any(f['name'] == 'Character 2' and f['rule'] == 'holding_opening_from_before'
               and f['evidence']['source_conflict_requires_review'] for f in fixes)


def test_real_missing_opening_uses_the_preceding_complete_holding_set():
    board = deepcopy(CASES['cinedance-012473-b0000-shot01'])
    recipient = next(e for e in board['shots'][3]['present'] if e['name'] == 'Character 2')
    recipient.pop('holding', None)
    fixes = reconcile_holding(board)
    assert recipient['holding'] == ['Object 1', 'Object 3']
    assert any(f['shot'] == 4 and f['rule'] == 'holding_opening_continuation' for f in fixes)


def test_explicit_empty_hands_end_the_previous_possession_continuation():
    board = deepcopy(CASES['cinedance-012473-b0000-shot01'])
    recipient = next(e for e in board['shots'][3]['present'] if e['name'] == 'Character 2')
    recipient.pop('holding', None)
    recipient['state'] = 'empty-handed'
    reconcile_holding(board)
    assert recipient['holding'] == []


def test_putting_the_real_cup_in_a_pocket_does_not_imply_releasing_it():
    board = deepcopy(CASES['cinedance-012429-b0002-shot01'])
    shot = board['shots'][0]
    shot['action'] = 'Character 2 puts Object 1 in his pocket.'
    shot['changes'] = []
    reconcile_holding(board)
    assert shot['changes'] == []


def test_real_offscreen_handoff_is_not_repaired_in_the_opposite_direction():
    board = deepcopy(CASES['film-003656-b0003/shot/0:10'])
    shot = board['shots'][7]
    assert shot['action'] == 'Character 2 hands Object 2 to someone off-screen.'
    before = deepcopy(shot)
    fixes = reconcile_holding(board)
    assert shot['changes'] == before['changes']
    assert not any(f['shot'] == 8 for f in fixes)
    assert any(issue['rule'] == 'holding_action_requires_review' and issue['shot'] == 8
               for issue in holding_consistency_issues(board))


def test_multiple_events_for_the_same_cup_do_not_use_unordered_set_deltas():
    for action in ('Character 2 drops Object 1 then picks up Object 1.',
                   'Character 2 picks up Object 1 then hands Object 1 to Character 3.',
                   'Character 2 hands Object 1 to Character 3, who then hands it back.'):
        board = deepcopy(CASES['cinedance-012429-b0002-shot01'])
        shot = board['shots'][0]
        shot['action'] = action
        shot['changes'] = []
        reconcile_holding(board)
        assert shot['changes'] == []
        assert any(issue['rule'] == 'holding_action_requires_review' and issue['shot'] == 1
                   for issue in holding_consistency_issues(board))


def test_negated_handoff_does_not_create_possession_changes():
    board = deepcopy(CASES['cinedance-012429-b0002-shot01'])
    shot = board['shots'][0]
    shot['action'] = 'Character 2 never hands Object 1 to Character 3.'
    shot['changes'] = []
    reconcile_holding(board)
    assert shot['changes'] == []


def test_real_late_added_held_character_consumes_its_history_in_the_same_call():
    board = deepcopy(CASES['cinedance-009433-b0000/paragraph/0:12'])
    fixes = reconcile_holding(board)
    child = next(e for e in board['shots'][2]['present'] if e['name'] == 'Character 5')
    assert child['holding'] == ['Object 2']
    assert any(e['name'] == 'Object 2' for e in board['shots'][2]['present'])
    assert any(f['shot'] == 3 and f['name'] == 'Character 5' and f['rule'] == 'holding_opening_continuation' for f in fixes)
    assert reconcile_holding(board) == []


def test_real_general_before_mismatch_is_normalized_and_the_source_conflict_is_kept_in_evidence():
    board = deepcopy(CASES['cinedance-009067-b0003/paragraph/0:7'])
    before = deepcopy(board)
    fixes = reconcile_holding(board)
    person = next(e for e in board['shots'][6]['present'] if e['name'] == 'Character 1')
    assert person['holding'] == ['Object 5']
    assert any(f['shot'] == 7 and f['rule'] == 'holding_opening_from_before'
               and f['evidence']['source_conflict_requires_review'] for f in fixes)
    assert before['shots'][6]['present'][0]['holding'] == []
    assert reconcile_holding(board) == []


def test_real_handbag_is_preserved_when_a_paper_bag_is_received():
    board = deepcopy(CASES['cinedance-015017-b0001/paragraph/0:7'])
    reconcile_holding(board)
    shot = board['shots'][1]
    recipient = next(e for e in shot['present'] if e['name'] == 'Character 2')
    change = next(c for c in shot['changes'] if c['name'] == 'Character 2' and c['what'] == 'holding')
    assert recipient['holding'] == ['Object 2']
    assert change['before'] == ['Object 2']
    assert set(change['after']) == {'Object 1', 'Object 2'}
    assert reconcile_holding(board) == []


@pytest.mark.parametrize('block', [
    'cinedance-012469-b0000', 'cinedance-013366-b0001', 'cinedance-015766-b0000',
    'film-001692-b0003', 'film-006916-b0001', 'film-007010-b0000', 'film-019378-b0014',
])
def test_real_non_holding_interactions_are_not_handoff_candidates(block):
    board = deepcopy(PENDING_CASES[block])
    original = deepcopy(board)
    assert holding_consistency_issues(board) == []
    assert reconcile_holding(board) == []
    assert board == original


@pytest.mark.parametrize('block', [
    'cinedance-012298-b0001', 'cinedance-010938-b0000', 'film-006455-b0004',
])
def test_real_drinking_and_phone_reposition_do_not_add_a_possession_event(block):
    board = deepcopy(PENDING_CASES[block])
    original = deepcopy(board)
    assert holding_consistency_issues(board) == []
    assert reconcile_holding(board) == []
    assert board == original


def test_real_receiver_takes_it_repeats_the_numbered_transfer():
    board = deepcopy(PENDING_CASES['cinedance-010926-b0000'])
    reconcile_holding(board)
    changes = {c['name']: c for c in board['shots'][0]['changes'] if c['what'] == 'holding'}
    assert changes['Character 1']['before'] == ['Object 5']
    assert changes['Character 1']['after'] == ['Object 5', 'Object 4']
    assert changes['Character 2']['before'] == ['Object 6', 'Object 4']
    assert changes['Character 2']['after'] == ['Object 6']
    assert holding_consistency_issues(board) == []
    assert reconcile_holding(board) == []


def test_real_receiver_sits_up_and_takes_it_has_both_transfer_endpoints():
    board = deepcopy(PENDING_CASES['cinedance-009581-b0001'])
    reconcile_holding(board)
    changes = {c['name']: c for c in board['shots'][0]['changes'] if c['what'] == 'holding'}
    assert changes['Character 1']['before'] == ['Object 1']
    assert changes['Character 1']['after'] == []
    assert changes['Character 2']['before'] == []
    assert changes['Character 2']['after'] == ['Object 1']
    assert holding_consistency_issues(board) == []
    assert reconcile_holding(board) == []


def test_real_already_complete_transfer_with_recipient_restatement_is_unchanged():
    board = deepcopy(PENDING_CASES['film-008511-b0001'])
    original = deepcopy(board)
    assert holding_consistency_issues(board) == []
    assert reconcile_holding(board) == []
    assert board == original


@pytest.mark.parametrize('action', [
    'Character 1 passes the basketball to the right.',
    'Character 1 passes the cup to the right, and exits the frame to the left.',
    'Character 1 picks up Object 1, then takes the photo from Character 2.',
    'Character 1 picks up Object 1, then takes a photo of Character 2 from Character 3.',
    'Character 1 picks up Object 1, then takes a drink from the tray.',
    'Character 1 picks up Object 1, then takes a hit to the face.',
])
def test_non_holding_phrase_filters_preserve_real_acquisition_and_handoff_ambiguity(action):
    board = deepcopy(PENDING_CASES['cinedance-009581-b0001'])
    board['shots'][0]['action'] = action
    board['shots'][0]['changes'] = []
    assert any(issue['category'] == 'semantic_coverage_candidate' for issue in holding_consistency_issues(board))
    assert reconcile_holding(board) == []
