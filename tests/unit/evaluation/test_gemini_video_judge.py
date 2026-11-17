"""整片事件／状态判官的登记表、实际镜界与费用预留。"""
import json
from itertools import accumulate
from pathlib import Path

import pytest

from mspc.evaluation import gemini_checklist_batch as b


@pytest.fixture
def story():
    return json.loads((Path(__file__).parents[2] / 'fixtures/gemini_native/story.json').read_text())


def test_registry_and_questions_keep_real_entities_without_shot_script(story):
    rows = [q for q in story['requirements'] if q['dimension'] in b.DIMENSIONS]
    text = b.video_judge_questions_text(story['request'], story['shot_seconds'], rows)
    registry, rest = text.split('\n\n## Shots in this video\n')
    assert registry == '## Registry\n' + story['request'].split('\nSHOTS\n')[0]
    assert '## Request' not in text and '\nSHOTS\n' not in text
    assert 's07:' not in text and 'raises his hand as if to throw something' not in text
    assert '1. (0:10.8–0:14.6) Does Character 1 hold Object 2?' in rest
    assert '3. (end of 0:19.7–0:21.5) Is Object 2 in the bullseye of Object 1?' in rest


def test_explicit_cut_bounds_replace_windows_and_placeholders(story):
    row = next(q for q in story['requirements'] if q['requirement_id'].endswith('/q05'))
    # 沿用真实状态题的实体与题面；时间格式独立于镜头长度来源。
    row = {**row, 'shot': 2, 'question': row['question'] + ' Compare {shot 1} and {shot 2}.'}
    text = b.video_judge_questions_text(story['request'], [2.5, 2.5], [row], [0.0, 2.4, 5.1])
    assert 'Shot 1: 0:00.0–0:02.4 · Shot 2: 0:02.4–0:05.1' in text
    assert '1. (end of 0:02.4–0:05.1)' in text
    assert 'Compare 0:00.0–0:02.4 and 0:02.4–0:05.1.' in text
    assert '{shot' not in text


def test_decimal_clock_carries_into_the_next_minute(story):
    text = b.video_judge_questions_text(story['request'], [59.96, 1.14], [])
    assert 'Shot 1: 0:00.0–1:00.0 · Shot 2: 1:00.0–1:01.1' in text


@pytest.mark.parametrize('bounds', [[0, 2.4], [0, 2.4, 2.0], [1, 2.4, 5.1], [0, float('nan'), 5.1]])
def test_invalid_actual_boundaries_fail_before_submission(story, bounds):
    with pytest.raises(ValueError, match='shot boundaries'):
        b.video_judge_questions_text(story['request'], [2.5, 2.5], [], bounds)


def test_missing_registry_or_unknown_shot_fails_before_submission(story):
    with pytest.raises(ValueError, match='registry blocks'):
        b.video_judge_questions_text('SHOTS\ns01: A man walks.', [2.0], [])
    row = story['requirements'][0]
    with pytest.raises(ValueError, match='unresolved shot placeholder'):
        b.video_judge_questions_text(story['request'], story['shot_seconds'],
                                     [{**row, 'question': 'What happens in {shot 99}?'}])


@pytest.mark.parametrize('alignment', ['cuts', 'cut_root', 'source', 'planned'])
def test_prepare_freezes_windows_filters_premises_and_builds_medium_request(tmp_path, story, alignment):
    source, cut_root, output = tmp_path / 'source', tmp_path / 'cut', tmp_path / 'judge'
    row, sid = 'dev30-row', 'cinedance-011901-b0000-shot01'
    story = {**story, 'story_id': sid, 'duration_seconds': sum(story['shot_seconds']),
             'method': 'text_renderer_planner_references', 'input_tier': 'shots', 'seed': 42}
    planned = [0.0, *accumulate(story['shot_seconds'])]
    aligned = list(planned)
    aligned[4] += 0.125
    aligned[5] -= 0.0625
    windows = [list(w) for w in zip(aligned, aligned[1:])]
    if alignment == 'source':
        story['shot_windows'] = windows
    b.write(source / 'rows' / row / 'prepared.json', {
        'stories': [story], 'source': {'questions': 'old'}, 'unavailable_stories': []})
    b.write(source / 'rows' / row / 'media-map.json', {sid: {
        'shot_frames': [round(t * 16) for t in story['shot_seconds']],
        'probe': {'fps': 16, 'frames': round(planned[-1] * 16)}, 'video_bytes': 1234}})
    bank, items = tmp_path / 'bank.json', tmp_path / 'items.json'
    b.write(bank, {'version': 12, 'stories': {sid: {'requirements': story['requirements']}}})
    skipped = story['requirements'][4]['requirement_id']
    b.write(items, {'stories': {sid: {'items': [{'item_id': sid + '/cut', 'dependents': [skipped]}]}}})
    b.write(cut_root / 'verdicts' / f'{row}.json', {sid + '/cut': {'verdict': 'same'}})
    paths = None
    if alignment == 'cut_root':
        b.write(cut_root / 'shot-windows' / f'{row}.json', {sid: windows})
    elif alignment == 'cuts':
        paths = [tmp_path / 'cuts.json']
        b.write(paths[0], {'rows': {row: {sid: {'detected_cut_times_seconds': aligned[1:-1]}}}})
        # 显式切点优先于切镜根里的旧时间窗。
        b.write(cut_root / 'shot-windows' / f'{row}.json', {sid: [list(w) for w in zip(planned, planned[1:])]})
    counts = b.prepare(source, bank, items, cut_root, [row], 2, 'low', 'reason_first', output, paths)
    prepared = b.load(output / 'rows' / row / 'prepared.json')['stories'][0]
    expected = planned if alignment == 'planned' else aligned
    assert prepared['shot_windows'] == [list(w) for w in zip(expected, expected[1:])]
    assert prepared['shot_seconds'] == [end - start for start, end in zip(expected, expected[1:])]
    assert sum(prepared['shot_seconds']) == story['duration_seconds']
    assert counts[row]['skipped_premise_failed'] == 1
    assert prepared['skipped_premise_failed'] == [skipped]
    assert all(q['dimension'] in b.DIMENSIONS and q['requirement_id'] != skipped for q in prepared['requirements'])
    judge = b.load(output / 'config.json')['judge']
    request = b.request_for(prepared, {'uri': 'u', 'mimeType': 'video/mp4'}, judge)
    text = request['contents'][0]['parts'][1]['text']
    assert ('1. (0:10.9–0:14.5)' if alignment != 'planned' else '1. (0:10.8–0:14.6)') in text
    assert request['generationConfig']['mediaResolution'] == 'MEDIA_RESOLUTION_MEDIUM'
    assert request['generationConfig']['responseJsonSchema'] == b.schema('reason_first')
    assert judge['prompt_version'] == 5
    assert b.load(source / 'rows' / row / 'prepared.json')['stories'][0] == story


def test_video_reserve_covers_medium_resolution_frame_tokens():
    # 5.1秒×2FPS，上取采样帧数11；文字和输出预留仍独立计入硬顶。
    expected = b.g.reserve(11 * 258 + 4000, 12288, 'batch')
    assert b.story_reserve({'duration_seconds': 5.1}, 2) == expected
    assert expected > b.g.reserve(11 * 70 + 4000, 12288, 'batch')


def test_submit_rejects_old_root_before_upload_or_batch_creation(tmp_path, monkeypatch):
    b.write(tmp_path / 'config.json', {'judge': {'prompt_version': 3}})

    def unexpected(*args, **kwargs):
        pytest.fail('old root must be rejected before reading pending stories or touching uploads')

    monkeypatch.setattr(b, 'pending', unexpected)
    with pytest.raises(ValueError, match='prepare a new root'):
        b.submit(tmp_path, ['row'], 1, b.D('1'), object(), object(), 1)
