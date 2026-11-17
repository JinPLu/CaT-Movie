"""现行计分（第 9 版：第 8 版前提作废，难反例只作监测）的纯规则检查。

四镜。第 6 版题：人物 1↔2、物体 1↔3、地点 1↔4、事件、状态复查（变化在第 2 镜、复查在第 3 镜）、画风、同镜难反例、
跨镜人物难反例 1↔3、跨镜地点难反例 2↔4。切镜执行项：1↔2（相邻，人物题）、1↔3（物体题与人物难反例共用）、
2↔3（相邻，状态复查）、3↔4（相邻）。计划 1↔4 与 2↔4 同机位，地点题与地点难反例无前提。
"""
import json

import pytest

from mspc.evaluation.checklist_scoring import (
    COLUMNS, EVENT_DURATION_GROUPS, event_duration_scores, paired_intervals,
    planned_cut_hits, readout_version, storyboard_shot_seconds, story_scores, v8_premises,
)

S = 'story-a'


def row(rid, dimension, shot, reference_shot=None, question=''):
    return {'requirement_id': f'{S}/{rid}', 'dimension': dimension, 'shot': shot, 'reference_shot': reference_shot,
            'anchor': 'during', 'question': question, 'depends_on': []}


def item(rid, earlier, later, dependents):
    return {'item_id': f'{S}/{rid}', 'reference_shot': earlier, 'shot': later, 'planned_cut': later == earlier + 1,
            'dependents': [f'{S}/{q}' for q in dependents]}


ROWS = [row('q01', 'character', 2, 1), row('q02', 'object', 3, 1), row('q03', 'place', 4, 1), row('q04', 'event', 2),
        row('q05', 'state', 3), row('s01', 'style', 1), row('m01', 'monitor', 2, 2, 'Is A the same person as B?'),
        row('m02', 'monitor', 3, 1, 'Is A at {shot 1} the same person as B at {shot 3}?'),
        row('m03', 'monitor', 4, 2, 'Is Location 1 at {shot 2} the same place as Location 2 at {shot 4}?')]
ITEMS = [item('c01', 1, 2, ['q01']), item('c02', 1, 3, ['q02', 'm02']), item('c03', 2, 3, ['q05']),
         item('c04', 3, 4, [])]
ALL_RIGHT = {**{f'{S}/{q}': 'yes' for q in ('q01', 'q02', 'q03', 'q04', 'q05', 's01')},
             f'{S}/m01': 'no', f'{S}/m02': 'no', f'{S}/m03': 'no'}
ALL_NEW = {f'{S}/c0{k}': {'verdict': 'new'} for k in range(1, 5)}


def scores(answers, verdicts):
    found, holds, cut_passed = v8_premises(ITEMS, verdicts)
    return story_scores(ROWS, found, answers, holds, cut_passed)


def test_everything_right_passes_every_column_and_the_story():
    out = scores(ALL_RIGHT, ALL_NEW)
    assert out['scores'] == {'character': 1.0, 'object': 1.0, 'scene': 1.0, 'event': 1.0, 'state': 1.0, 'cut': 1.0}
    assert out['all_pass'] and out['voided'] == {'character': 0, 'object': 0, 'scene': 0, 'state': 0}


def test_same_setup_voids_only_the_questions_on_that_pair():
    # 第 3 镜与第 1 镜同机位（照抄）：物体题作废；1↔2 的人物题、事件、状态照常；切镜列不受 1↔3 影响。
    out = scores(ALL_RIGHT, {**ALL_NEW, f'{S}/c02': {'verdict': 'same'}})
    assert out['scores']['character'] == 1.0
    assert out['scores']['object'] == 0.0 and out['scores']['scene'] == 1.0
    assert out['scores']['event'] == 1.0 and out['scores']['cut'] == 1.0
    assert out['voided'] == {'character': 0, 'object': 1, 'scene': 0, 'state': 0}
    assert out['premise_holding']['character'] == {'score': 1.0, 'questions': 1, 'of': 1}
    assert not out['all_pass'] and out['premises_failed'] == 1


def test_hard_negatives_are_monitors_only():
    answers = {**ALL_RIGHT, f'{S}/m01': 'yes', f'{S}/m02': 'yes', f'{S}/m03': 'yes'}  # 判官把三对都判成同一个
    out = scores(answers, ALL_NEW)
    assert out['scores']['character'] == 1.0 and out['scores']['scene'] == 1.0 and out['all_pass']


def test_no_cut_fails_the_cut_and_its_dependents_and_state_rechecks_are_gated():
    out = scores(ALL_RIGHT, {**ALL_NEW, f'{S}/c03': {'verdict': 'no_cut'}, f'{S}/c04': {'verdict': 'same'}})
    assert out['scores']['cut'] == pytest.approx(1 / 3)
    assert out['scores']['state'] == 0.0 and out['voided']['state'] == 1


def test_undecided_items_are_an_error():
    with pytest.raises(ValueError):
        scores(ALL_RIGHT, {**ALL_NEW, f'{S}/c01': {'verdict': None}})


def test_premise_naming_a_question_outside_the_bank_is_an_error():
    found, holds, cut_passed = v8_premises(ITEMS, ALL_NEW)
    with pytest.raises(ValueError):
        story_scores(ROWS[:1], found, ALL_RIGHT, holds, cut_passed)


def test_cut_matching_is_one_to_one_nearest_first():
    assert planned_cut_hits([2.0, 2.4], [2.3]) == {1}
    assert planned_cut_hits([2.0, 4.0], [2.6, 4.4]) == {1}


def test_pairs_the_plan_keeps_alike_still_need_a_new_setup_and_stay_out_of_the_cut_column():
    """计划不要求不同的镜对（不进切镜执行列）：新机位前提成立，同机位或没切开都不成立。"""
    items = [{**i, 'planned_cut': False} if i['item_id'].endswith('c02') else i for i in ITEMS]
    found, holds, cut_passed = v8_premises(items, ALL_NEW)
    assert holds[f'{S}/c02'] and f'{S}/c02' not in cut_passed
    for verdict in ('same', 'no_cut'):
        found, holds, _ = v8_premises(items, {**ALL_NEW, f'{S}/c02': {'verdict': verdict}})
        assert not holds[f'{S}/c02']


def test_questions_whose_premise_failed_need_no_answer_and_others_must_have_one():
    """第 11 版起前提不成立的题不送判官：没有答案也记不过；前提成立却没答案要报错。"""
    verdicts = {**ALL_NEW, f'{S}/c02': {'verdict': 'same'}}
    found, holds, cut_passed = v8_premises(ITEMS, verdicts)
    answers = {k: v for k, v in ALL_RIGHT.items() if k not in (f'{S}/q02', f'{S}/m02')}  # c02 的依赖不送
    out = story_scores(ROWS, found, answers, holds, cut_passed)
    assert out['scores'] == scores(ALL_RIGHT, verdicts)['scores']
    with pytest.raises(ValueError):
        story_scores(ROWS, found, {k: v for k, v in answers.items() if k != f'{S}/q01'}, holds, cut_passed)


def test_event_duration_uses_each_story_fps_and_includes_five_seconds(tmp_path):
    path = tmp_path / 'storyboards.json'
    path.write_text(json.dumps({'stories': [
        {'story_id': S, 'fps': 24, 'storyboard': {'shots': [
            {'shot': 1, 'frame_count': 120}, {'shot': 2, 'frame_count': 121}]}},
        {'story_id': 'story-b', 'fps': 16, 'storyboard': {'shots': [{'shot': 1, 'frame_count': 120}]}},
    ]}))
    seconds = storyboard_shot_seconds(path)
    assert seconds[S] == {1: 5.0, 2: 121 / 24}
    assert seconds['story-b'] == {1: 7.5}
    rows = [row('e1', 'event', 1), row('e2', 'event', 2)]
    grouped = event_duration_scores(rows, [], {f'{S}/e1': 'yes', f'{S}/e2': 'no'}, {}, seconds[S])
    assert grouped == {'at_most_5_seconds': {'score': 1.0, 'questions': 1},
                       'over_5_seconds': {'score': 0.0, 'questions': 1}}


def test_duration_groups_preserve_failed_premise_without_fabricating_answer():
    rows = [row('e1', 'event', 2), row('e2', 'event', 3)]
    premises = [{'id': 'cut', 'planned_cut': True, 'dependents': [f'{S}/e1']}]
    result = event_duration_scores(rows, premises, {f'{S}/e2': 'yes'}, {'cut': False}, {2: 5, 3: 8})
    assert result['at_most_5_seconds']['score'] == 0.0
    assert result['over_5_seconds']['score'] == 1.0
    with pytest.raises(ValueError, match='no judge answer'):
        event_duration_scores(rows, premises, {}, {'cut': False}, {2: 5, 3: 8})


@pytest.mark.parametrize('fps', [None, 0, -1, True, float('nan')])
def test_storyboard_duration_rejects_missing_or_invalid_fps(tmp_path, fps):
    path = tmp_path / 'storyboards.json'
    path.write_text(json.dumps({'stories': [{'story_id': S, 'fps': fps, 'storyboard': {'shots': []}}]}))
    with pytest.raises(ValueError, match='fps'):
        storyboard_shot_seconds(path)


@pytest.mark.parametrize('version', ['v9', 'v11', 'v12'])
def test_pairing_averages_seeds_within_story_then_clusters_source_films(version):
    def story(sid, value):
        scored = {'scores': dict.fromkeys(COLUMNS, value), 'all_pass': bool(value),
                  'event_duration_groups': {g: {'score': value, 'questions': 1} for g in EVENT_DURATION_GROUPS}}
        return {'story_id': sid, **({'v8': scored} if version == 'v9' else scored)}
    rows = {
        'before': {'per_story': [story(s, 0.0) for s in ('a', 'b', 'c')]},
        'seed42': {'per_story': [story('a', 1.0), story('b', 0.0), story('c', 1.0)]},
        'seed43': {'per_story': [story('a', 0.0), story('b', 1.0), story('c', 1.0)]},
        'seed44': {'per_story': [story('a', 1.0), story('c', 1.0)]},
    }
    readout = {'rows': rows, **({'scoring': 'v9'} if version == 'v9' else {'version': version, 'judge_contract': 'v11'})}
    result = paired_intervals(readout, [('before', 'seed42,seed43,seed44')],
                              {'a': 'film-1', 'b': 'film-1', 'c': 'film-2'}, version != 'v9')[0]
    for metric in result['metrics'].values():
        assert metric['mean'] == pytest.approx((200 / 3 + 50 + 100) / 3)
        assert metric['shots'] == 3  # 故事是配对单位，三个种子没有变成9个故事。
        assert metric['stories'] == 2  # 同片两故事仍只是一簇。
        assert metric['ci95'] == pytest.approx([175 / 3, 100])
    assert ('event_over_5_seconds' in result['metrics']) == (version != 'v9')
    assert not {'state_establishment', 'state_revisit'} & result['metrics'].keys()


@pytest.mark.parametrize('version', ['v13', 'v14', 'v15', 'v16', 'v17'])
def test_pairing_state_roles_uses_produced_subscores_and_preserves_story_seed_averaging(version):
    def story(sid, changed, retained, *, establishment=True):
        rows = [{**row(key, dimension, 2), 'requirement_id': f'{sid}/{key}'}
                for key, dimension in [('character', 'character'), ('object', 'object'), ('place', 'place'),
                                       ('change', 'event'), ('retention', 'state')]]
        rows[-1]['state_role'] = 'revisit'
        answers = {r['requirement_id']: 'yes' for r in rows}
        answers[f'{sid}/change'] = 'yes' if changed else 'no'
        answers[f'{sid}/retention'] = 'yes' if retained else 'no'
        if establishment:
            rows.append({**row('establishment', 'state', 2), 'requirement_id': f'{sid}/establishment',
                         'state_role': 'establishment', 'change_event': f'{sid}/change'})
            answers[f'{sid}/establishment'] = 'yes'
        premise = {'id': f'{sid}/cut', 'planned_cut': True, 'dependents': []}
        return {'story_id': sid, **story_scores(rows, [premise], answers,
                                              {premise['id']: True}, {premise['id']: True})}

    readout = {'version': version, 'judge_contract': 'v11', 'rows': {
        'before': {'per_story': [story('a', False, True), story('b', False, True),
                                 story('c', False, True, establishment=False)]},
        'seed42': {'per_story': [story('a', True, False), story('b', True, True),
                                 story('c', False, False, establishment=False)]},
        'seed43': {'per_story': [story('a', False, True), story('b', True, True),
                                 story('c', False, False, establishment=False)]},
    }}
    result = paired_intervals(readout, [('before', 'seed42,seed43')],
                              {'a': 'film-1', 'b': 'film-2', 'c': 'film-2'})[0]['metrics']
    # 终态始终成立、变化只在部分种子执行；建立分不能拿总状态分代替。
    assert result['state_establishment']['mean'] == pytest.approx(75)
    assert result['state_establishment']['ci95'] == pytest.approx([50, 100])
    assert result['state_establishment']['story_ids'] == ['a', 'b']
    assert result['state_establishment']['shots'] == 2
    assert result['state_establishment']['stories'] == 2
    assert result['state_revisit']['mean'] == pytest.approx(-50)
    assert result['state_revisit']['story_ids'] == ['a', 'b', 'c']
    assert result['state']['mean'] == pytest.approx(-125 / 3)


def test_pairing_requires_explicit_scoring_identity():
    with pytest.raises(ValueError, match='must declare'):
        readout_version({'rows': {}})
    with pytest.raises(ValueError, match='must declare'):
        readout_version({'version': 'v12', 'judge_contract': 'v9', 'rows': {}})


def test_paired_cli_rejects_mixed_scoring_versions(tmp_path, monkeypatch):
    from mspc.evaluation.checklist_scoring import main
    paths = [tmp_path / 'v9.json', tmp_path / 'v12.json']
    paths[0].write_text(json.dumps({'scoring': 'v9', 'rows': {}}))
    paths[1].write_text(json.dumps({'version': 'v12', 'judge_contract': 'v11', 'rows': {}}))
    monkeypatch.setattr('sys.argv', ['checklist_scoring', 'paired', '--readout', *map(str, paths),
                                   '--requests', str(tmp_path / 'requests.json'), '--pair', 'a', 'b',
                                   '--output', str(tmp_path / 'paired.json')])
    with pytest.raises(ValueError, match='same scoring version'):
        main()
