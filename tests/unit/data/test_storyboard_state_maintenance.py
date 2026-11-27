"""分镜表维护：衣着沿当前值延续、没写的持物沿用上一镜（11-12 精简后不再报告冲突）。

夹具是 dev30 逐镜档 seed42 的真实原答：公共 Qwen v15 的换装回访例只在变化条目里写换装、在场条目从不写衣着；
修复监督 SFT 的交接例在交接镜开镜就让两人同持黑箱（计划写了的照写，不改）。"""
import json
from copy import deepcopy

from mspc.data.shot_text import shot_text
from mspc.data.storyboard import resolve_outfits
from mspc.evaluation.storyboard_planner import assess_answer
from mspc.data.reference_selection import reference_position, select_references
from tests.support.paths import REPO_ROOT

ROWS = {row['story_id']: row for row in json.loads(
    (REPO_ROOT / 'tests/fixtures/planner_storyboard/state_maintenance_real.json').read_text())['rows']}


def _compile(story_id):
    row = ROWS[story_id]
    timing = {'target_seconds': row['target_seconds'], 'fps': 16, 'expected_cut_times_seconds': None}
    result = assess_answer({'status': 'completed', 'raw_content': row['raw_content']}, timing=timing, stated=None)
    assert result['structure_status'] == 'valid'
    return result


def test_real_outfit_changes_carry_into_later_shots_and_reach_the_shot_text():
    result = _compile('film-000127-b0000-shot01')
    shots = result['storyboard']['shots']
    worn = {shot['shot']: entry.get('outfit') for shot in shots for entry in shot['present'] if entry['name'] == 'Character 1'}
    # 第3镜脱外套与领带、第5镜脱衬衫、第8镜换成内裤；之后没写衣着的镜都沿用当时衣着，不退回登记的灰西装。
    assert worn == {1: None, 2: None, 3: None, 4: 'a pink shirt and a pair of dark trousers',
                    5: 'a pink shirt and a pair of dark trousers', 6: 'a pair of dark trousers',
                    7: 'a pair of dark trousers', 8: 'a pair of dark trousers', 9: 'white underwear', 10: 'white underwear'}
    assert {fix['rule'] for fix in result['compilation']['outfit_resolved']} == {'outfit_carried'}
    assert not any(entry.get('outfit') for shot in shots for entry in shot['present'] if entry['name'] == 'Character 2')
    sixth = shot_text(result['storyboard'], 5)
    assert 'grey suit' not in sixth and 'dark trousers' in sixth
    # 原答不改。
    assert 'outfit' not in json.dumps([entry for shot in result['raw_storyboard']['shots'] for entry in shot['present']])


def test_written_back_old_outfit_is_replaced_and_a_new_outfit_without_event_is_kept():
    board = deepcopy(_compile('film-000127-b0000-shot01')['raw_storyboard'])
    board['shots'][5]['present'][0]['outfit'] = 'a grey suit, a pink shirt, and a purple tie'  # 第6镜写回登记衣着
    board['shots'][9]['present'][0]['outfit'] = 'a white bathrobe'  # 第10镜无事件换了新衣着
    fixes = resolve_outfits(board)
    assert board['shots'][5]['present'][0]['outfit'] == 'a pair of dark trousers'
    assert [fix['shot'] for fix in fixes if fix['rule'] == 'outfit_reverted_without_event'] == [6]
    assert board['shots'][9]['present'][0]['outfit'] == 'a white bathrobe'
    assert not any(fix['shot'] == 10 for fix in fixes)


def test_real_two_holders_after_a_one_sided_handoff_are_kept_as_written():
    result = _compile('cinedance-016627-b0003-shot01')
    raw = {entry['name']: entry.get('holding') for entry in result['raw_storyboard']['shots'][3]['present']}
    compiled = {entry['name']: entry.get('holding') for entry in result['storyboard']['shots'][3]['present']}
    assert raw['Character 1'] == compiled['Character 1'] == ['Object 1']
    assert raw['Character 2'] == compiled['Character 2'] == ['Object 1']
    assert 'state_conflicts' not in result['compilation']


def test_reference_selection_reads_the_same_carried_outfit_as_the_shot_text():
    board = _compile('film-000127-b0000-shot01')['storyboard']
    # 第10镜穿内裤：第8镜记了换装（长裤→内裤），第9镜开镜已是内裤。换装镜取末帧、算换装后，与第9镜同档，
    # 同档里同框人少、景别、最早依次比下来取第8镜的末帧；换装前的镜都排后。
    assert [index for index in range(9) if any(entry['name'] == 'Character 1' and entry.get('outfit') == 'white underwear'
                                               for entry in board['shots'][index]['present'])] == [8]
    assert [change['what'] for change in board['shots'][7]['changes'] if change['name'] == 'Character 1'] == ['outfit']
    picked = select_references(board, span=(9, 10), candidates=range(9), frames_max=10, distinct_camera=False)
    assert 7 in picked and 8 not in picked and reference_position(board['shots'][7]) == 'last'
