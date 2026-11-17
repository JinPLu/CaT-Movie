"""一致性镜对图片判官（consistency_pair_batch）：请求形状、按题号对回答案、前提不成立的题不送。"""
import json
from decimal import Decimal
from pathlib import Path

import pytest

from mspc.evaluation import consistency_pair_batch as c
from mspc.evaluation.cut_execution import shot_edges, shot_triplet_frames

REQUEST = '\n'.join(['SUBJECTS', '- Character 1 (person): a man in a red coat', '- Object 2 (object): a lamp',
                     'LOCATIONS', '- Location 1: a kitchen', 'SHOTS', 's01: A man stands.'])


def q(n, dimension, shot, ref, text, depends):
    return {'requirement_id': f's/q{n:02d}', 'dimension': dimension, 'shot': shot, 'reference_shot': ref,
            'anchor': 'during', 'question': text, 'depends_on': depends}


FILES = [{'uri': f'https://generativelanguage.googleapis.com/v1beta/files/f{n}', 'mimeType': 'image/jpeg'}
         for n in range(6)]


ROWS = [q(1, 'character', 3, 1, 'Is Character 1 at {shot 3} the same person as at {shot 1}?', ['Character 1']),
        q(2, 'place', 3, 1, 'Are {shot 3} and {shot 1} in the same place (Location 1)?', []),
        q(3, 'object', 5, 1, 'Is Object 2 at {shot 5} the same object as at {shot 1}?', ['Object 2']),
        q(4, 'event', 1, None, 'Does Character 1 stand?', ['Character 1'])]


def test_pairs_group_by_shot_pair_and_drop_failed_premises():
    groups = c.pair_questions(ROWS, failed={'s/q03'})
    assert {k: [r['requirement_id'] for r in v] for k, v in groups.items()} == {(1, 3): ['s/q01', 's/q02']}


def test_request_puts_rules_in_system_and_labels_each_frame():
    body = c.request_body(REQUEST, (1, 3), ROWS[:2], FILES, 'low')
    parts = body['contents'][0]['parts']
    assert body['systemInstruction'] == {'parts': [{'text': c.SYSTEM}]}
    assert parts[0]['text'] == '## Registry\n- Character 1 (person): a man in a red coat\n- Location 1: a kitchen'
    assert [p['text'] for p in parts[1:13:2]] == [
        'Shot 1 (earlier shot), first frame', 'Shot 1 (earlier shot), middle frame',
        'Shot 1 (earlier shot), last frame', 'Shot 3 (later shot), first frame',
        'Shot 3 (later shot), middle frame', 'Shot 3 (later shot), last frame']
    assert [p['fileData'] for p in parts[2:13:2]] == [{'fileUri': f['uri'], 'mimeType': 'image/jpeg'} for f in FILES]
    assert parts[-1]['text'].startswith('## Questions\n1. [character] Is Character 1 at Shot 3 the same person as at Shot 1?')
    assert parts[-1]['text'].endswith(c.CLOSING)
    assert body['generationConfig']['mediaResolution'] == 'MEDIA_RESOLUTION_MEDIUM'


def test_answers_map_by_the_given_numbers_and_wrong_numbers_are_format_failures():
    good = {'characters': [{'id': 1, 'face': 'same', 'hair': 'same', 'build': 'same', 'clothing': 'different',
                            'same_person': 'yes'}],
            'objects': [], 'locations': [{'id': 2, 'layout': 'same', 'landmarks': 'same', 'color_mood': 'same',
                                          'same_place': 'no'}]}
    line = {'key': 'r|s|1-3', 'response': {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
        {'text': json.dumps(good)}]}}], 'usageMetadata': {'promptTokenCount': 3000, 'candidatesTokenCount': 200}}}
    record = c.judge_record(line, ROWS[:2])
    assert record['status'] == 'answered' and record['answers'] == {'s/q01': 'yes', 's/q02': 'no'}
    renumbered = {**good, 'locations': [{**good['locations'][0], 'id': 1}]}
    line['response']['candidates'][0]['content']['parts'][0]['text'] = json.dumps(renumbered)
    assert c.judge_record(line, ROWS[:2])['status'] == 'format_failed'


def test_registry_must_find_every_named_entity():
    with pytest.raises(ValueError):
        c.registry(REQUEST, [q(9, 'character', 2, 1, 'Is Character 7 at {shot 2} the same person as at {shot 1}?', [])])


def test_pending_preserves_real_format_failure_and_skips_open_batch(tmp_path):
    row = 'test92-holocine'
    failed = f'{row}|cinedance-015067-b0001-shot01|1-3'
    queued, fresh, missing = f'{row}|story|1-2', f'{row}|story|2-3', f'{row}|story|3-4'
    c.g.write_json(tmp_path / 'pairs' / f'{row}.json', {failed: {}, queued: {}, fresh: {}, missing: {}})
    c.g.write_json(c.receipt_path(tmp_path, row, failed), {
        'status': 'format_failed', 'error': 'ValueError: objects: ids [1, 3] differ from [2, 3]'})
    c.g.write_json(tmp_path / 'batches/open.json', {'keys': [queued], 'closed': False})
    c.g.write_json(tmp_path / 'batches/closed.json', {'keys': [missing], 'closed': True})
    assert c.pending(tmp_path, [row]) == [(row, fresh)]


def test_prompt_blocked_pair_counts_its_questions_as_refused_and_is_not_resent(tmp_path):
    """实际回执（11-15 测试集）：输入阶段 PROHIBITED_CONTENT、没有 candidates；题记 refused，格式错仍是缺答。"""
    row = 'wan22-final-test92-window-7s-seed42'
    blocked, broken, ok = f'{row}|cinedance-014083-b0000-shot01|5-6', f'{row}|s|1-3', f'{row}|s|3-5'
    c.g.write_json(tmp_path / 'pairs' / f'{row}.json', {
        blocked: {'questions': ['cinedance-014083-b0000-shot01/q22', 'cinedance-014083-b0000-shot01/q24']},
        broken: {'questions': ['s/q01']}, ok: {'questions': ['s/q03']}})
    c.g.write_json(c.receipt_path(tmp_path, row, blocked), {
        'status': 'format_failed', 'error': "ValueError: no candidates; promptFeedback={'blockReason': 'PROHIBITED_CONTENT'}",
        'raw': {'promptFeedback': {'blockReason': 'PROHIBITED_CONTENT'}}})
    c.g.write_json(c.receipt_path(tmp_path, row, broken), {
        'status': 'format_failed', 'error': 'ValueError: objects: ids [1, 3] differ from [2, 3]', 'raw': {'candidates': [{}]}})
    c.g.write_json(c.receipt_path(tmp_path, row, ok), {'status': 'answered', 'answers': {'s/q03': 'yes'}})
    assert c.pending(tmp_path, [row]) == []
    assert c.answers(tmp_path, row) == {'cinedance-014083-b0000-shot01/q22': 'refused',
                                        'cinedance-014083-b0000-shot01/q24': 'refused', 's/q03': 'yes'}


@pytest.mark.parametrize('fixture_name', [
    'holocine-object-numbering.json',
    'multishotmaster-object-numbering.json',
    'multishotmaster-character-numbering.json',
])
def test_schema_rejects_real_entity_numbers_and_accepts_question_numbers(fixture_name):
    """三条真实失败原答：每组只允许该请求的题号；正例为独立构造，不把原答按顺序套回。"""
    path = Path(__file__).parents[2] / 'fixtures/gemini_native' / fixture_name
    saved = json.loads(path.read_text())
    questions, response = saved['questions'], saved['response']
    schema = c.response_schema(questions)
    raw = c.g.answer_json(response)
    assert c.judge_record({'key': saved['source_key'], 'response': response}, questions)['status'] == 'format_failed'
    rejected = []
    for dimension, (group, _) in c.GROUP_OF.items():
        numbers = [n for n, question in enumerate(questions, 1) if question['dimension'] == dimension]
        spec = schema['properties'][group]
        assert spec['minItems'] == spec['maxItems'] == len(numbers)
        if numbers:
            assert spec['items']['properties']['id']['enum'] == numbers
            rejected += [entry['id'] for entry in raw[group] if entry['id'] not in numbers]
        else:
            assert spec['maxItems'] == 0 and 'enum' not in spec['items']['properties']['id']
    assert rejected  # 原答中的实体编号不能通过本组题号约束。

    features = {
        'character': {'face': 'not visible', 'hair': 'not visible', 'build': 'not visible',
                      'clothing': 'not visible', 'same_person': 'no'},
        'object': {'shape': 'not visible', 'color_texture': 'not visible', 'details': 'not visible',
                   'same_object': 'no'},
        'place': {'layout': 'not visible', 'landmarks': 'not visible', 'color_mood': 'not visible',
                  'same_place': 'no'},
    }
    valid = {'characters': [], 'objects': [], 'locations': []}
    for n, question in enumerate(questions, 1):
        dimension = question['dimension']
        group, _ = c.GROUP_OF[dimension]
        valid[group].append({'id': n, **features[dimension]})
    assert c.map_answers(questions, valid) == {q['requirement_id']: 'no' for q in questions}
    for group in ('characters', 'objects', 'locations'):
        if len(valid[group]) > 1:
            valid[group][1]['id'] = valid[group][0]['id']  # 即使属于 enum，重复仍必须拒绝。
            with pytest.raises(ValueError):
                c.map_answers(questions, valid)
            break
    assert 'enum' not in c.SCHEMA['properties']['objects']['items']['properties']['id']


def test_request_embeds_question_specific_schema():
    body = c.request_body(REQUEST, (1, 3), ROWS[:2], FILES, 'low')
    schema = body['generationConfig']['responseJsonSchema']
    assert schema == c.response_schema(ROWS[:2])
    assert schema['properties']['characters']['items']['properties']['id']['enum'] == [1]
    assert schema['properties']['locations']['items']['properties']['id']['enum'] == [2]
    assert schema['properties']['objects']['maxItems'] == 0


def test_new_batch_records_schema_revision_before_creation(tmp_path, monkeypatch):
    row, key = 'row', 'row|s|1-3'
    bank = tmp_path / 'bank.json'
    c.g.write_json(bank, {'stories': {'s': {'request': REQUEST, 'requirements': ROWS[:2]}}})
    c.g.write_json(tmp_path / 'config.json', {'bank': str(bank)})
    c.g.write_json(tmp_path / 'pairs' / f'{row}.json', {
        key: {'story': 's', 'shots': [1, 3], 'questions': ['s/q01', 's/q02']}})
    uris = [f'gs://bucket/pairs/row/s/{n:06d}.jpg' for n in range(6)]
    c.g.write_json(tmp_path / 'images' / f'{row}.json', {key: uris})

    class Client:
        def register_all(self, given):
            assert given == uris
            return {u: {**f, 'name': f'files/f{n}'} for n, (u, f) in enumerate(zip(uris, FILES))}

    def start(client, path, requests, uploads):
        record = c.load(path)
        assert record['schema_revision'] == c.SCHEMA_REVISION
        assert Decimal(record['reserve_usd']) == c.g.reserve(6 * 560 + 800, c.MAX_OUTPUT_TOKENS, 'batch')
        assert requests[0][1]['generationConfig']['responseJsonSchema'] == c.response_schema(ROWS[:2])
        parts = requests[0][1]['contents'][0]['parts']
        assert [p['fileData']['fileUri'] for p in parts if 'fileData' in p] == [f['uri'] for f in FILES]
        assert uploads == {u: f'files/f{n}' for n, u in enumerate(uris)}  # 收取时撤销登记
        return record

    monkeypatch.setattr(c.g, 'start_batch', start)
    record = c.submit(tmp_path, [row], 1, Decimal('1'), 'low', Client())
    assert record['keys'] == [key]


@pytest.mark.parametrize(('bounds', 'shot', 'total', 'expected'), [
    ([0, 9, 21, 30], 1, 30, [2, 4, 6]),
    ([0, 9, 21, 30], 3, 30, [23, 25, 27]),
    ([0, 1], 1, 1, [0, 0, 0]),
])
def test_shot_triplet_preserves_edge_frames_and_adds_integer_midpoint(bounds, shot, total, expected):
    frames = shot_triplet_frames(bounds, shot, total)
    assert frames == expected
    assert frames[::2] == shot_edges(bounds, shot, total)


@pytest.mark.parametrize(('length', 'expected'), [(2, [0, 1, 1]), (3, [0, 1, 2]), (4, [0, 2, 3])])
def test_shot_triplet_short_shots_remain_in_chronological_order(length, expected):
    assert shot_triplet_frames([0, length], 1, length) == expected


def test_prepare_sends_six_aligned_frames_to_the_bucket_once(tmp_path, monkeypatch):
    row, sid, key = 'row', 's', 'row|s|1-3'
    root, media_root, cut_root = (tmp_path / name for name in ('judge', 'media', 'cuts'))
    bank, items, cuts = (tmp_path / name for name in ('bank.json', 'items.json', 'cuts.json'))
    c.g.write_json(bank, {'stories': {sid: {'request': REQUEST, 'requirements': ROWS[:2]}}})
    c.g.write_json(items, {'stories': {sid: {'items': []}}})
    c.g.write_json(cut_root / 'verdicts' / f'{row}.json', {})
    c.g.write_json(cuts, {'rows': {row: {sid: {'detected_cut_times_seconds': [0.9, 2.1]}}}})
    c.g.write_json(media_root / 'rows' / row / 'prepared.json', {'stories': [{'story_id': sid}]})
    c.g.write_json(media_root / 'rows' / row / 'media-map.json', {
        sid: {'shot_frames': [10, 10, 10], 'probe': {'fps': 10, 'frames': 30},
              'remote_video_path': '/video.mp4'}})
    calls = []
    expected = [2, 4, 6, 23, 25, 27]
    storage = object()

    def to_bucket(given, judge, given_row, jobs):
        calls.append(jobs)
        assert (given, judge, given_row) == (storage, 'pairs', row)
        assert jobs == [{'video': '/video.mp4', 'story': sid, 'pairs': [
            {'key': key, 'frames': expected, 'judge_frames': expected}]}]
        return {key: [f'gs://b/pairs/{row}/{sid}/{f:06d}.jpg' for f in expected]}

    monkeypatch.setattr(c, 'frames_to_bucket', to_bucket)
    summary = c.prepare(root, media_root, bank, items, cut_root, [cuts], [row], storage)
    assert summary == {row: {'pairs': 1, 'questions': 2, 'skipped_premise_failed': 0}}
    assert c.load(root / 'pairs' / f'{row}.json')[key]['frames'] == expected
    assert c.load(root / 'images' / f'{row}.json')[key][3] == f'gs://b/pairs/{row}/{sid}/000023.jpg'
    c.prepare(root, media_root, bank, items, cut_root, [cuts], [row], storage)
    assert len(calls) == 1  # 已写桶的镜对不再取帧
