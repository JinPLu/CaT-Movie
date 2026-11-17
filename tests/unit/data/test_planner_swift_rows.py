"""ms-swift 标准流程的训练行：按段权重（照写 0、推得出的正例加权）与按故事长度取行，用重取池里的真实行核。"""
from copy import deepcopy
import json
import re
from pathlib import Path

import pytest

from mspc.data.planner_dataset import (EMPTY_THINK, field_supervision_tiers, prepare_planner_example, production_script,
                                    select_long_story_rows, swift_example, target_segments)
from mspc.data.storyboard import parse_shot_script

ROWS = json.loads((Path(__file__).resolve().parents[2] / 'fixtures/planner_retrain/rows.json').read_text())
EMPTY_HOLDING_ROWS = json.loads((Path(__file__).resolve().parents[2]
                               / 'fixtures/planner_retrain/empty_holding_commitments.json').read_text())['rows']
WEIGHTS = {'state': 2.0, 'outfit': 5.0, 'holding': 3.7, 'changes': 2.1}


def words(text):
    """测试用的切分：标点各成一段、字母数字连成一段（真实训练用规划器分词器的偏移）。"""
    return [m.span() for m in re.finditer(r'\w+|\s+|[^\w\s]', text)]


def test_segments_reproduce_the_compact_json_target():
    for row in (ROWS['shot'], ROWS['summary']):
        segments = target_segments(row, WEIGHTS, words)
        assert ''.join(t for t, _ in segments) == json.dumps(row['storyboard'], ensure_ascii=False, separators=(',', ':'))


def test_shot_tier_stated_values_carry_zero_weight():
    row = ROWS['shot']
    stated = parse_shot_script(row['script'])
    zero = ''.join(t for t, w in target_segments(row, WEIGHTS, words) if w == 0)
    for shot in stated['shots']:
        assert json.dumps(shot['action'], ensure_ascii=False) in zero
    for value in stated['objects'].values():
        assert json.dumps(value, ensure_ascii=False) in zero


def test_real_summary_without_request_binding_still_supervises_its_prior_commitments():
    segments = target_segments(ROWS['summary'], WEIGHTS, words)
    assert all(w > 0 for _, w in segments)
    # 真实粗档请求只有 a person / another person，首镜是补全；后镜必须对前文已写出的状态负责。
    tiers = field_supervision_tiers(ROWS['summary'])
    assert ('shots', 0, 'present', 0, 'posture') not in tiers
    assert tiers[('shots', 1, 'present', 0, 'posture')] == 'infer'
    assert tiers[('shots', 3, 'present', 0, 'holding')] == 'infer'
    assert any(w == WEIGHTS['state'] for _, w in segments)
    assert any(w == WEIGHTS['holding'] for _, w in segments)


def test_real_outfit_change_becomes_a_training_commitment_even_when_request_leaves_it_open():
    fixture = Path(__file__).resolve().parents[2] / 'fixtures/planner_information/outfit_continuity.json'
    case = json.loads(fixture.read_text())['cases'][1]
    row = {'tier': 'shot', 'script': case['request'], 'storyboard': case['reference']}
    tiers = field_supervision_tiers(row)
    # 请求只说走到椅子，源片的换装值不是请求可确认的本镜变化；后镜已在teacher前文中承诺。
    assert ('shots', 4, 'changes', 0) not in tiers
    index = next(j for j, e in enumerate(case['reference']['shots'][5]['present']) if e['name'] == 'Character 2')
    assert tiers[('shots', 5, 'present', index, 'outfit')] == 'infer'


def test_commitments_use_previous_shots_and_compare_full_holding_sets():
    row = deepcopy(ROWS['summary'])
    before = deepcopy(row)
    tiers = field_supervision_tiers(row)
    # 第2镜把原先的Object2改成Object3，不能把本镜新值当自己的前文依据；第4镜才是Object3延续。
    assert ('shots', 1, 'present', 0, 'holding') not in tiers
    assert tiers[('shots', 3, 'present', 0, 'holding')] == 'infer'
    assert row == before


def test_real_never_held_default_empty_sets_do_not_become_inferred_by_repetition():
    row = EMPTY_HOLDING_ROWS['cinedance-009067-b0003/paragraph/0:7']
    board = row['storyboard']
    # Character 2全故事没有非空持物或释放，只有s01/s06的默认空列表；请求也没说空手。
    assert all(not e.get('holding') for s in board['shots'] for e in s['present'] if e['name'] == 'Character 2')
    assert not any(c['name'] == 'Character 2' and c['what'] == 'holding' for s in board['shots'] for c in s['changes'])
    tiers = field_supervision_tiers(row)
    assert ('shots', 0, 'present', 1, 'holding') not in tiers
    assert ('shots', 5, 'present', 1, 'holding') not in tiers


def test_explicit_request_empty_handed_is_still_inferred_even_without_prior_possession():
    row = deepcopy(EMPTY_HOLDING_ROWS['cinedance-009067-b0003/paragraph/0:7'])
    # 在真实失败行上加一个最小请求对照：词面确实指定空手时不应被默认[]的排除规则遮掉。
    header = row['script'].split('\n\nUser request:\n', 1)[0]
    row['script'] = header + '\n\nUser request:\nThe scene is a reception area. Character 2 waits empty-handed.'
    tiers = field_supervision_tiers(row)
    assert tiers[('shots', 0, 'present', 1, 'holding')] == 'infer'
    assert tiers[('shots', 5, 'present', 1, 'holding')] == 'infer'


def test_real_release_establishes_a_later_empty_holding_commitment():
    row = EMPTY_HOLDING_ROWS['film-008506-b0005/paragraph/0:3']
    # C2第1镜把现金交出，before非空、after空；后镜的空集合是释放结果，不是默认空手。
    release = row['storyboard']['shots'][0]['changes'][0]
    assert release['name'] == 'Character 2' and release['before'] and release['after'] == []
    assert field_supervision_tiers(row)[('shots', 1, 'present', 0, 'holding')] == 'infer'


def test_real_nonempty_possession_then_an_explicit_clear_can_continue_without_repeated_release():
    row = EMPTY_HOLDING_ROWS['cinedance-016920-b0005/shot/0:10']
    # 实际目标第7/8镜连续空手，此前已有非空持物；第8镜继续承担第7镜明确的清空选择。
    assert any(c['name'] == 'Character 1' and c['what'] == 'holding' and c['after']
               for s in row['storyboard']['shots'][:6] for c in s['changes'])
    assert row['storyboard']['shots'][6]['present'][0]['holding'] == []
    assert row['storyboard']['shots'][7]['present'][0]['holding'] == []
    assert field_supervision_tiers(row)[('shots', 7, 'present', 0, 'holding')] == 'infer'


def test_boundaries_fall_on_token_boundaries():
    row = ROWS['shot']
    text = json.dumps(row['storyboard'], ensure_ascii=False, separators=(',', ':'))
    cuts = {b for _, b in words(text)}
    position = 0
    for segment, _ in target_segments(row, WEIGHTS, words):
        position += len(segment)
        assert position in cuts


def test_swift_row_starts_with_unsupervised_empty_think():
    row = deepcopy(ROWS['shot'])
    _, body = row['script'].split('\n\nUser request:\n', 1)
    frames = sum(shot['frame_count'] for shot in row['storyboard']['shots'])
    assert frames == 173
    row['script'] = production_script(script=body, target_seconds=frames / 16, fps=16)
    assert row['script'] == ('Production requirements: approximately 10.8125 seconds total; 16 fps; '
                             'at most 225 frames per shot (14.0625 seconds).\n\nUser request:\n' + body)
    example = swift_example(row, WEIGHTS, words)
    assistant = [m for m in example['messages'] if m['role'] == 'assistant']
    assert assistant[0] == {'role': 'assistant', 'content': EMPTY_THINK, 'loss_scale': 0.0}
    assert example['channel'] == 'shot' and [m['role'] for m in example['messages'][:2]] == ['system', 'user']
    assert json.loads(example['messages'][1]['content'])['script'] == row['script']
    assert row['storyboard'] == ROWS['shot']['storyboard']


def test_real_old_pool_preparation_is_an_explicit_nonmutating_producer_boundary():
    row = deepcopy(ROWS['shot'])
    _, body = row['script'].split('\n\nUser request:\n', 1)
    frames = sum(s['frame_count'] for s in row['storyboard']['shots'])
    row['script'] = f'Production requirements: approximately {frames / 16:g} seconds total; 16 fps.\n\nUser request:\n{body}'
    original = deepcopy(row)
    prepared, report = prepare_planner_example(row)
    assert row == original
    assert prepared['sample_id'] == original['sample_id']
    assert prepared['script'] == production_script(script=body, target_seconds=frames / 16, fps=16)
    assert report['production_requirements_updated']
    assert report['after'] == []
    # 训练请求不陈述介质：目标去掉逐镜画面类型（合同只在请求明说介质时写），原行不改。
    shots = original['storyboard']['shots']
    assert any('picture_type' in shot for shot in shots)
    assert report['picture_types_removed'] == sum('picture_type' in shot for shot in shots)
    assert all('picture_type' not in shot for shot in prepared['storyboard']['shots'])
    # 人物关系不在规划器合同里（11-13「同意，不让写」）：目标同样去掉。
    assert report['relations_removed'] == sum(len(shot.get('relations') or []) for shot in shots)
    assert all('relations' not in shot for shot in prepared['storyboard']['shots'])
    target = ''.join(m['content'] for m in swift_example(prepared, WEIGHTS, words)['messages'][3:])
    assert 'picture_type' not in target and '"relations"' not in target


def test_weights_must_be_complete_and_positive():
    with pytest.raises(ValueError):
        target_segments(ROWS['shot'], {**WEIGHTS, 'changes': 0}, words)
    with pytest.raises(ValueError):
        target_segments(ROWS['shot'], {'state': 2.0}, words)


def test_long_story_selection_keeps_long_rows_and_samples_short_shares():
    rows = []
    for tier in ('summary', 'paragraph', 'shot'):
        # 逐镜档的长行比整块多，要抽到与另两档同样多（三档 1:1:1）
        for n, length in enumerate([1] * 40 + [3] * 40 + [7] * (20 if tier == 'shot' else 12)):
            rows.append({'tier': tier, 'sample_id': f'{tier}-{n:03d}', 'shot_span': [0, length]})
    picked = select_long_story_rows(rows, min_all_shots=6, fraction_two_to_short=0.3, fraction_one=0.1, seed=42)
    for tier in ('summary', 'paragraph', 'shot'):
        lengths = [r['shot_span'][1] for r in picked if r['tier'] == tier]
        assert lengths.count(7) == 12 and lengths.count(3) == 6 and lengths.count(1) == 2
    assert picked == select_long_story_rows(rows, min_all_shots=6, fraction_two_to_short=0.3, fraction_one=0.1, seed=42)


def test_tokens_sharing_a_character_stay_in_one_segment():
    # 真实全池里一块含希腊字母，分词器把一个字符拆成两个 token、起止相同；段界不能落在它们中间。
    row = ROWS['shot']
    def split_char(text):
        spans = words(text)
        a, b = spans[3]
        return spans[:3] + [(a, b), (a, b)] + spans[4:]
    segments = target_segments(row, WEIGHTS, split_char)
    assert ''.join(t for t, _ in segments) == json.dumps(row['storyboard'], ensure_ascii=False, separators=(',', ':'))
