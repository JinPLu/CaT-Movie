"""保持复查两个题族（question_writer 规则 2 的追加）。

从请求出的一族在合成故事上做纯规则检查：第 1 镜脱外套（外观），第 2 镜接过杯子（纯持有），第 6 镜穿回外套
（再次改变外观，且参与者含持有者）。从给定分镜标注出的一族用 11-03 原样摘出的真实故事（已有题＋标注）。
"""
import copy
import json
from pathlib import Path

from mspc.evaluation.question_writer import annotation_persistence_questions, persistence_questions, questions

ANNOTATED = json.loads((Path(__file__).resolve().parents[2] / 'fixtures/evaluation/question_persistence'
                        / 'annotation_stories.json').read_text())['stories']
STORY = 'story-a'
TAKE_OFF = 'Is Character 1 without Object 2?'
IN_HANDS = "Is Object 1 now in Character 1's hands?"
PRESENT = {1: ['Character 1', 'Object 2'], 2: ['Character 1', 'Character 2', 'Object 1'], 3: ['Character 1', 'Object 1'],
           4: ['Character 1'], 5: ['Character 1', 'Object 1'], 6: ['Character 1', 'Object 2'],
           7: ['Character 1', 'Object 1'], 8: ['Character 1']}


def story():
    board = {'characters': [{'name': 'Character 1', 'kind': 'person', 'appearance': 'a man'},
                            {'name': 'Character 2', 'kind': 'person', 'appearance': 'a woman'}],
             'objects': [{'name': 'Object 1', 'appearance': 'a cup'}, {'name': 'Object 2', 'appearance': 'a jacket'}],
             'places': [{'name': 'Location 1', 'appearance': 'a kitchen'}],
             'shots': [{'shot': k, 'action': '', 'present': [{'name': n} for n in names], 'place': 'Location 1'}
                       for k, names in PRESENT.items()]}
    output = {'names': [{'id': i, 'name': i} for i in ('Character 1', 'Character 2', 'Object 1', 'Object 2', 'Location 1')],
              'shots': [{'shot': k, 'present': names, 'location': 'Location 1'} for k, names in PRESENT.items()],
              'actions': [
                  {'shot': 1, 'question': 'Does Character 1 take off Object 2?', 'participants': ['Character 1', 'Object 2'],
                   'result': {'question': TAKE_OFF, 'entity': 'Character 1', 'kind': 'appearance'}},
                  {'shot': 2, 'question': 'Does Character 2 hand Object 1 to Character 1?',
                   'participants': ['Character 2', 'Character 1', 'Object 1'],
                   'result': {'question': IN_HANDS, 'entity': 'Object 1', 'kind': 'holding_or_place'}},
                  {'shot': 6, 'question': 'Does Character 1 put Object 2 back on?', 'participants': ['Character 1', 'Object 2'],
                   'result': {'question': 'Is Character 1 wearing Object 2?', 'entity': 'Character 1', 'kind': 'appearance'}}]}
    return output, board, questions(STORY, output, board)


def shots(rows, question):
    return [row['shot'] for row in rows if row['question'] == question]


def test_appearance_reasked_at_every_later_appearance_until_next_change():
    output, board, existing = story()
    rows = persistence_questions(STORY, output, board, existing)
    # 原规则已问第 2、5 镜（首末）；第 6 镜穿回外套，之后不再问脱外套。穿回后的首末（7、8）已有，不新增。
    assert shots(existing, TAKE_OFF) == [1, 2, 5]
    assert shots(rows, TAKE_OFF) == [3, 4]
    assert shots(rows, 'Is Character 1 wearing Object 2?') == []


def test_holding_reasked_where_both_appear_until_next_action_involving_them():
    output, board, existing = story()
    rows = persistence_questions(STORY, output, board, existing)
    # 第 4 镜没有杯子；第 6 镜的动作参与者含持有者，第 7 镜虽二者都在也不再问。
    assert shots(rows, IN_HANDS) == [3, 5]
    posed = copy.deepcopy(output)
    posed['actions'][1]['result']['question'] = 'Is Object 1 held up by Character 1?'
    assert shots(persistence_questions(STORY, posed, board, questions(STORY, posed, board)),
                 'Is Object 1 held up by Character 1?') == []


def test_identical_existing_state_rows_are_not_repeated():
    output, board, existing = story()
    existing = existing + [{'requirement_id': f'{STORY}/q18', 'dimension': 'state', 'shot': 3, 'reference_shot': None,
                            'anchor': 'during', 'question': TAKE_OFF, 'depends_on': ['Character 1']}]
    rows = persistence_questions(STORY, output, board, existing)
    assert shots(rows, TAKE_OFF) == [4]
    assert shots(rows, IN_HANDS) == [3, 5]


def test_ids_continue_after_highest_existing_q_number():
    output, board, existing = story()
    assert existing[-1]['requirement_id'] == f'{STORY}/q17'
    assert existing[4]['requirement_id'] == f'{STORY}/q05' and existing[4]['dimension'] == 'event'
    existing = [row for row in existing if row['requirement_id'] != f'{STORY}/q05'] + [
        {'requirement_id': f'{STORY}/s01', 'dimension': 'style', 'shot': 1, 'reference_shot': None, 'anchor': 'during',
         'question': 'Do all shots keep one style?', 'depends_on': []}]
    rows = persistence_questions(STORY, output, board, existing)
    assert [row['requirement_id'] for row in rows] == [f'{STORY}/q{n}' for n in (18, 19, 20, 21)]
    assert all(row['dimension'] == 'state' and row['anchor'] == 'during' and row['family'] == 'persistence'
               and row['reference_shot'] is None for row in rows)
    assert [(row['source_shot'], row['depends_on']) for row in rows] == [
        (1, ['Character 1']), (1, ['Character 1']), (2, ['Object 1']), (2, ['Object 1'])]


def test_review_dropped_reask_removes_its_result_and_that_shot():
    output, board, existing = story()
    existing = [row for row in existing if not (row['question'] == TAKE_OFF and row['shot'] == 5)]
    rows = persistence_questions(STORY, output, board, existing)
    assert shots(rows, TAKE_OFF) == []
    assert shots(rows, IN_HANDS) == [3]


def annotated(story_id):
    story = ANNOTATED[story_id]
    return annotation_persistence_questions(story_id, story['writer_output'], story['storyboard'], story['existing'])


def test_annotation_state_asked_at_each_later_shot_and_not_deduped_by_a_position_result():
    # 第 9 镜已有的「Has Character 3 moved away」是位置结果，不是同一人的外观题，所以第 9 镜照问哭。
    rows = annotated('cinedance-010942-b0000-shot03')
    assert [(r['requirement_id'], r['shot'], r['source_shot'], r['question']) for r in rows] == [
        ('cinedance-010942-b0000-shot03/q19', 9, 7, 'Is Character 3 crying?'),
        ('cinedance-010942-b0000-shot03/q20', 10, 7, 'Is Character 3 crying?'),
        ('cinedance-010942-b0000-shot03/q21', 11, 7, 'Is Character 3 crying?'),
        ('cinedance-010942-b0000-shot03/q22', 11, 10, 'Is Character 1 crying?')]
    assert all(r['dimension'] == 'state' and r['anchor'] == 'during' and r['family'] == 'persistence_annotation'
               and r['source_change'] == 'state' for r in rows)


def test_annotation_holding_skips_shot_already_asked_by_plain_holding_question():
    # 第 3 镜已有请求保持复查题「Is Object 1 held by Character 1?」；第 5 镜只有「拿离耳边」这种暂时姿态题，照问。
    rows = annotated('film-000193-b0001-shot01')
    assert [(r['shot'], r['question'], r['depends_on']) for r in rows] == [
        (5, 'Is Character 1 holding Object 1?', ['Character 1', 'Object 1'])]
    assert rows[0]['requirement_id'] == 'film-000193-b0001-shot01/q16'


def test_annotation_stops_where_the_annotation_shows_empty_hands():
    # 第 1 镜拿起电话，第 3 镜标注 holding 为空：此后不再问，即使后面又有空手的镜。
    assert annotated('film-019290-b0000-shot01') == []


def test_annotation_requires_request_presence_and_skips_existing_appearance_questions():
    # 第4镜已经抛出钓线；标注第5–7镜残留「拿着鱼饵」不能复查。第6镜请求没点名持有者；
    # 帽子在第 5、7 镜已有外观复查。整故事不出题。
    assert annotated('film-001594-b0006-shot04') == []


def test_holder_only_presence_still_allows_revisits_without_a_release():
    source = copy.deepcopy(ANNOTATED['film-001594-b0006-shot04'])
    source['writer_output']['actions'] = [a for a in source['writer_output']['actions']
                                          if 'cast his fishing line' not in a['question']]
    rows = annotation_persistence_questions(STORY, source['writer_output'], source['storyboard'], source['existing'])
    assert [(r['shot'], r['depends_on']) for r in rows] == [
        (5, ['Character 3', 'Object 3']), (7, ['Character 3', 'Object 3'])]


def test_casting_a_line_does_not_release_a_held_rod():
    source = copy.deepcopy(ANNOTATED['film-001594-b0006-shot04'])
    next(o for o in source['storyboard']['objects'] if o['name'] == 'Object 3')['appearance'] = 'a fishing rod'
    rows = annotation_persistence_questions(STORY, source['writer_output'], source['storyboard'], source['existing'])
    assert [r['shot'] for r in rows] == [5, 7]


def test_other_people_casting_does_not_end_the_holders_lure_chain():
    source = copy.deepcopy(ANNOTATED['film-001594-b0006-shot04'])
    next(a for a in source['writer_output']['actions'] if 'cast his fishing line' in a['question'])[
        'participants'] = ['Character 5']
    rows = annotation_persistence_questions(STORY, source['writer_output'], source['storyboard'], source['existing'])
    assert [r['shot'] for r in rows] == [5, 7]
