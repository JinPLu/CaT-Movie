"""逐镜档明示事实拷回、动作扩写保留：真实请求往返及失败案例的实际消费链回归。"""
from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from mspc.data.shot_text import DEFAULT_CAMERA, _outfit, _sentence, shot_script_text
from mspc.data.storyboard import compile_storyboard, copy_stated_values, parse_shot_script

ROOT = Path(__file__).resolve().parents[3]
STORIES = json.loads((ROOT / 'configs/benchmark/benchmark_storyboards_v4.json').read_text())['stories']
CASE = json.loads((ROOT / 'tests/fixtures/planner_information/cinedance-012429-b0002-shot01.json').read_text())


def _description(appearance, outfit):
    return ', '.join(part for part in (appearance, _outfit(outfit, 'wearing')) if part)


def test_every_benchmark_request_parses_back_to_the_given_storyboard():
    split_differs = 0
    for story in STORIES:
        board, request = story['storyboard'], story['inputs']['shots']['text']
        assert request == shot_script_text(board)
        stated = parse_shot_script(request)
        assert stated['objects'] == {e['name']: e['appearance'] for e in board['objects']}
        assert stated['places'] == {e['name']: e['appearance'] for e in board['places']}
        for entry in board['characters']:
            mine = stated['characters'][entry['name']]
            assert mine['kind'] == entry['kind']
            # 外貌与衣着的分界可能不同，但词与语序必须相同
            assert _description(mine['appearance'], mine['outfit']) == _description(entry['appearance'], entry.get('outfit'))
            split_differs += (mine['appearance'], mine['outfit']) != (entry['appearance'], _outfit(entry.get('outfit'), 'wearing')[8:] or None)
        assert len(stated['shots']) == len(board['shots'])
        for mine, shot in zip(stated['shots'], board['shots']):
            camera = {k: v for k, v in (shot.get('camera') or {}).items() if v and v not in DEFAULT_CAMERA}
            assert mine['camera'] == camera
            assert mine['action'] == _sentence(shot['action']).text
    assert split_differs == 3  # 基准 486 个人物里 3 个分界不同（见 parse_shot_script 说明）


def test_production_header_before_the_sections_is_skipped():
    story = STORIES[0]
    request = 'Production requirements: approximately 12 seconds total; 16 fps.\n\nUser request:\n' + story['inputs']['shots']['text']
    assert parse_shot_script(request) == parse_shot_script(story['inputs']['shots']['text'])


def test_real_zero_shot_plan_gets_stated_camera_and_registry_back_and_keeps_its_actions():
    # 真实原答：明示机位／登记拷回；11-08起不再用请求动作覆盖规划器的动作表达。
    stated = parse_shot_script(CASE['request'])
    plan = deepcopy(CASE['plan'])
    actions = [shot['action'] for shot in plan['shots']]
    plan['shots'][0]['camera'] = {**(plan['shots'][0].get('camera') or {}), 'shot_size': 'extreme wide shot'}
    copied = copy_stated_values(plan, stated)
    assert any(c['path'] == 'shots[0].camera.shot_size' for c in copied)
    for shot, want in zip(plan['shots'], stated['shots']):
        assert all(shot['camera'][k] == v for k, v in want['camera'].items())
    assert [shot['action'] for shot in plan['shots']] == actions
    for name, want in stated['characters'].items():
        entry = next(e for e in plan['characters'] if e['name'] == name)
        assert entry['appearance'] == want['appearance'] and (want['outfit'] is None or entry['outfit'] == want['outfit'])


def test_compile_records_the_copy_and_keeps_planner_completions():
    stated = parse_shot_script(CASE['request'])
    board, report = compile_storyboard(deepcopy(CASE['plan']), target_seconds=None, fps=None, cut_times_seconds=None, stated=stated)
    assert isinstance(report['stated_values_copied'], list)
    # 计划补全的构图、状态不被拷回动到
    assert [s.get('composition') for s in board['shots']] == [s.get('composition') for s in CASE['plan']['shots']]
    assert [s['changes'] for s in board['shots']] == [s['changes'] for s in CASE['plan']['shots']]
    assert [s['action'] for s in board['shots']] == [s['action'] for s in CASE['plan']['shots']]


@pytest.mark.parametrize('story_id,shot_index,expanded,prompt_excerpt', [
    ('film-001595-b0001-shot01', 3,
     'Character 4, the man in the yellow chicken costume, points at Object 5 floating in the water. '
     'The group looks at Object 5.',
     'Character 4, the man in the yellow chicken costume, points at Object 5'),
    ('cinedance-017367-b0000-shot01', 5,
     'Character 1, the man in the grey military uniform, adjusts his peaked cap. '
     'He turns toward Object 1, moves to its entrance, and begins getting into it.',
     'Character 1, the man in the grey military uniform, adjusts his peaked cap.'),
])
def test_real_failure_action_expansion_reaches_the_actual_renderer_text(story_id, shot_index, expanded, prompt_excerpt):
    # 真实失败故事及字段，扩写是显式的接口回归输入，不冒充模型答复或源片过程标签。
    from mspc.data.shot_text import shot_text
    from mspc.inference.story_planner import compile_record

    story = next(s for s in STORIES if s['story_id'] == story_id)
    board = deepcopy(story['storyboard'])
    original = board['shots'][shot_index]['action']
    board['shots'][shot_index]['action'] = expanded
    request = story['inputs']['shots']['text']
    record = {'raw_content': json.dumps(board), 'finish_reason': 'stop', 'original_story_request': request}
    task = {'input_tier': 'shots', 'story_request': request, 'expected_shot_count': len(board['shots']),
            'target_seconds': story['target_seconds'], 'fps': story['fps'], 'expected_cut_times_seconds': None}

    compile_record(record, task)

    assert record['status'] == 'completed'
    assert record['raw_storyboard']['shots'][shot_index]['action'] == expanded
    assert record['storyboard']['shots'][shot_index]['action'] == expanded
    assert parse_shot_script(record['original_story_request'])['shots'][shot_index]['action'] == original
    rendered = shot_text(record['storyboard'], shot_index)
    # Action句允许在登记ID后绑定短特征；去掉这类插入后，全部原动作字词仍完整消费。
    without_short_identities = re.sub(r'(Character \d+) \([^()]*\)', r'\1', rendered)
    assert prompt_excerpt in without_short_identities
    assert expanded in without_short_identities


def test_shot_count_mismatch_leaves_shots_alone():
    stated = parse_shot_script(CASE['request'])
    plan = deepcopy(CASE['plan'])
    plan['shots'] = plan['shots'][:-1]
    before = deepcopy(plan['shots'])
    copy_stated_values(plan, stated)
    assert plan['shots'] == before
