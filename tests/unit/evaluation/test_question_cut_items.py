"""切镜执行项（question_writer 规则 4，第 8 版）的纯规则检查。

六镜：1 地点一远景（角色 1 拿物体 1），2 地点一特写（角色 1），3 地点一特写但换成角色 2（同景别、同角度，正反打），
4 地点二俯拍远景（角色 2），5 地点二俯拍、没写景别（角色 2），6 同 5 的机位但无人。
已有题：人物 1↔2、物体 1↔3、地点 1↔2、同镜两人难反例、跨镜难反例（镜号倒写 4↔1）、事件、旧布局 2↔4、
第 4 镜的状态复查（变化在第 2 镜）与第 2 镜镜中就问的结果。
"""
from mspc.evaluation.question_writer import cut_items

STORY = 'story-a'
CAMERA = {1: {'shot_size': 'wide shot', 'angle': 'eye-level shot'}, 2: {'shot_size': 'close-up shot', 'angle': 'eye-level shot'},
          3: {'shot_size': 'close-up shot'}, 4: {'shot_size': 'wide shot', 'angle': 'high angle shot'},
          5: {'angle': 'high angle shot'}, 6: {'angle': 'high angle shot'}}
PLACE = {1: 'Location 1', 2: 'Location 1', 3: 'Location 1', 4: 'Location 2', 5: 'Location 2', 6: 'Location 2'}
PRESENT = {1: ['Character 1', 'Object 1'], 2: ['Character 1'], 3: ['Character 2', 'Object 1'], 4: ['Character 2'],
           5: ['Character 2'], 6: []}
BOARD = {'characters': [{'name': 'Character 1', 'kind': 'person', 'appearance': 'a man'},
                        {'name': 'Character 2', 'kind': 'person', 'appearance': 'a woman'}],
         'objects': [{'name': 'Object 1', 'appearance': 'a cup'}],
         'places': [{'name': 'Location 1', 'appearance': 'a kitchen'}, {'name': 'Location 2', 'appearance': 'a roof'}],
         'shots': [{'shot': k, 'place': PLACE[k], 'camera': CAMERA[k], 'present': [{'name': n} for n in PRESENT[k]],
                    'action': ''} for k in CAMERA]}
OUTPUT = {'names': [{'id': i, 'name': i} for i in ('Character 1', 'Character 2', 'Object 1', 'Location 1', 'Location 2')],
          'shots': [{'shot': k, 'present': names, 'location': PLACE[k]} for k, names in PRESENT.items()],
          'actions': [{'shot': 2, 'question': 'Does Character 1 sit down?', 'participants': ['Character 1'],
                       'result': {'question': 'Is Character 1 sitting?', 'entity': 'Character 1', 'kind': 'appearance'}}]}


def row(rid, dimension, shot, reference_shot=None):
    return {'requirement_id': f'{STORY}/{rid}', 'dimension': dimension, 'shot': shot, 'reference_shot': reference_shot,
            'anchor': 'during', 'question': '', 'depends_on': []}


EXISTING = [row('q01', 'character', 2, 1), row('q02', 'object', 3, 1), row('q03', 'place', 2, 1),
            row('m01', 'monitor', 3, 3), row('m02', 'monitor', 1, 4), row('q04', 'event', 2), row('q05', 'layout', 4, 2),
            {**row('q06', 'state', 4), 'question': 'Is Character 1 sitting?'},
            {**row('q07', 'state', 2), 'question': 'Is Character 1 sitting?'}]


def test_pairs_come_from_compared_shots_rechecks_and_planned_cuts_that_the_plan_makes_different():
    rows = cut_items(STORY, OUTPUT, BOARD, EXISTING)
    # 4↔5 同地点同角度同一人、没写景别：不出项；同镜难反例、旧布局题与镜中就问的结果不设前提；1↔2 只出一项，两道题共用；
    # 第 4 镜的状态复查挂"变化镜 2 → 复查镜 4"。
    assert [(r['reference_shot'], r['shot'], r['planned_cut'], r['dependents']) for r in rows] == [
        (1, 2, True, [f'{STORY}/q01', f'{STORY}/q03']), (1, 3, False, [f'{STORY}/q02']),
        (1, 4, False, [f'{STORY}/m02']), (2, 3, True, []), (2, 4, False, [f'{STORY}/q06']), (3, 4, True, []),
        (5, 6, True, [])]
    assert [r['planned_difference'] for r in rows] == [
        {'place': False, 'size_gap': 4, 'angle': False, 'cast': False},
        {'place': False, 'size_gap': 4, 'angle': False, 'cast': True},
        {'place': True, 'size_gap': 0, 'angle': True, 'cast': True},
        {'place': False, 'size_gap': 0, 'angle': False, 'cast': True},
        {'place': True, 'size_gap': 4, 'angle': True, 'cast': True},
        {'place': True, 'size_gap': 4, 'angle': True, 'cast': False},
        {'place': False, 'size_gap': 0, 'angle': False, 'cast': True}]
    assert [r['item_id'] for r in rows] == [f'{STORY}/c0{n}' for n in range(1, 8)]


def test_compared_shots_the_plan_keeps_alike_get_an_item_outside_the_cut_column():
    """4↔5 计划不要求不同（同地点同角度同一人、没写景别）：没有题比较时不出项；有题比较时出项，前提同样是切开且换机位，
    但不进切镜执行列。"""
    rows = cut_items(STORY, OUTPUT, BOARD, EXISTING + [row('q08', 'character', 5, 4)])
    gate = [r for r in rows if (r['reference_shot'], r['shot']) == (4, 5)]
    assert [(r['planned_cut'], r['planned_difference'], r['dependents']) for r in gate] == [(False, None, [f'{STORY}/q08'])]
    assert all('requires' not in r for r in rows)
