"""Google 原生接口的请求、正文、费用与批任务账（gemini_native、gemini_checklist_batch、cut_execution 的判官）。
夹具是 11-06 视频接口核查的真实请求与回执（开发故事 cinedance-011901-b0000-shot01，第 10 版 23 题，2 FPS）：
同步 generateContent 一条、Batch 一条。"""
import json
from decimal import Decimal as D
from pathlib import Path

import pytest

from mspc.evaluation import gemini_native as g
from mspc.evaluation.gemini_checklist_batch import JUDGE, receipt, schema

FIXTURES = Path(__file__).parents[2] / 'fixtures/gemini_native'


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


def test_request_body_matches_the_verified_call():
    sent = fixture('standard-fps2.request.json')
    video, prompt = sent['contents'][0]['parts']
    file = {'uri': video['fileData']['fileUri'], 'mimeType': video['fileData']['mimeType']}
    # 试调用当时的设置（24,576 输出上限、无 system instruction），只核请求的拼法与实测请求逐字段相同。
    body = g.json_request([g.video_part(file, 2), prompt], sent['generationConfig']['responseJsonSchema'],
                          system=None, thinking='low', max_output_tokens=24576,
                          media_resolution=sent['generationConfig']['mediaResolution'])
    assert body == sent
    with pytest.raises(ValueError):
        g.video_part(file, 25)


def test_whole_video_request_uses_registry_and_decimal_windows_after_the_video():
    from mspc.evaluation.gemini_checklist_batch import CLOSING, SYSTEM, request_for
    story = {**fixture('story.json'), 'shot_seconds': [4.0, 3.0],
             'requirements': [{'requirement_id': 's/q01', 'dimension': 'event', 'shot': 1, 'reference_shot': None,
                               'anchor': 'during', 'question': 'Does Character 1 walk?', 'depends_on': []}]}
    judge = {**JUDGE, 'thinking': 'low', 'fps': 2, 'schema': schema('reason_first'), 'system': SYSTEM, 'closing': CLOSING}
    body = request_for(story, {'uri': 'u', 'mimeType': 'video/mp4'}, judge)
    video, text = body['contents'][0]['parts']
    assert 'fileData' in video and body['systemInstruction'] == {'parts': [{'text': SYSTEM}]}
    assert text['text'].endswith('1. (0:00.0\u20130:04.0) Does Character 1 walk?\n\n' + CLOSING)
    assert text['text'].startswith('## Registry\nSUBJECTS\n- Character 1')
    assert '\nSHOTS\n' not in text['text'] and 's01: Medium shot.' not in text['text']
    assert body['generationConfig']['mediaResolution'] == 'MEDIA_RESOLUTION_MEDIUM'
    assert 'Give a short reason' not in text['text']


def test_answers_usage_and_cost_from_real_responses():
    response = fixture('standard-fps2.response.json')
    assert len(g.answer_json(response)['answers']) == 23
    assert g.modality_tokens(response['usageMetadata'], 'VIDEO') == 3762
    assert g.cost(response['usageMetadata'], 'standard') == D('0.00770175')
    line = json.loads((FIXTURES / 'batch-results.jsonl').read_text().splitlines()[0])
    assert g.cost(line['response']['usageMetadata'], 'batch') == D('0.004379625')
    with pytest.raises(ValueError):
        g.answer_json({**response, 'candidates': [{**response['candidates'][0], 'finishReason': 'MAX_TOKENS'}]})


def test_batch_line_becomes_a_rated_receipt_with_requirement_ids():
    story = {**fixture('story.json'), 'method': 'text_renderer_planner_references', 'input_tier': 'shots', 'seed': 42,
             'story_id': 'cinedance-011901-b0000-shot01'}
    line = json.loads((FIXTURES / 'batch-results.jsonl').read_text().splitlines()[0])
    judge = {**JUDGE, 'thinking': 'low', 'fps': 2, 'schema': schema('answer_first')}
    out = receipt('dev30-row', story, line['key'], {'name': 'batches/x'}, line, judge)
    assert out['status'] == 'rated' and out['cost_usd'] == '0.004379625'
    assert [a['requirement_id'] for a in out['result']['answers']] == [q['requirement_id'] for q in story['requirements']]
    broken = {**line, 'response': {**line['response'], 'candidates': [{**line['response']['candidates'][0], 'finishReason': 'SAFETY'}]}}
    assert receipt('dev30-row', story, line['key'], {'name': 'batches/x'}, broken, judge)['status'] == 'format_failed'
    assert receipt('dev30-row', story, line['key'], {'name': 'batches/x'}, {'key': line['key'], 'error': {'code': 500}},
                   judge)['status'] == 'request_failed'


def test_setup_judge_body_and_batch_line():
    from mspc.evaluation import cut_execution as c
    files = [{'uri': f'https://generativelanguage.googleapis.com/v1beta/files/f{n}', 'mimeType': 'image/jpeg'} for n in range(4)]
    body = c.judge_body(files, 'low')
    parts = body['contents'][0]['parts']
    assert body['systemInstruction'] == {'parts': [{'text': c.SYSTEM}]} and parts[-1] == {'text': c.QUESTION}
    assert [p['text'] for p in parts[0:-1:2]] == [label for _, label in c.LABELS]
    assert [p['fileData']['fileUri'] for p in parts[1:-1:2]] == [f['uri'] for f in files]
    assert body['generationConfig']['thinkingConfig'] == {'thinkingLevel': 'low'}
    assert body['generationConfig']['mediaResolution'] == 'MEDIA_RESOLUTION_MEDIUM'
    text = json.dumps({'shot_a': 'x', 'shot_b': 'y', 'setup': 'same'})
    line = {'key': 'row|s/1', 'response': {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': text}]}}],
                                          'usageMetadata': {'promptTokenCount': 4600, 'candidatesTokenCount': 120},
                                          'modelVersion': 'gemini-3.8-flash'}}
    record = c.judge_record(line)
    assert record['status'] == 'answered' and record['answer']['setup'] == 'same'
    assert D(record['cost_usd']) == (4600 * D('0.375') + 120 * D('1.875')) / 1_000_000
    assert c.judge_record({'key': 'row|s/1', 'error': {'code': 500}})['status'] == 'request_failed'
    bad = json.dumps({'shot_a': 'x', 'shot_b': 'y', 'setup': 'maybe'})
    line['response']['candidates'][0]['content']['parts'][0]['text'] = bad
    assert c.judge_record(line)['status'] == 'format_failed'


def test_setup_verdicts_follow_the_chain_and_receipts(tmp_path):
    """两镜之间没切开判 no_cut；切开了、中帧不算几乎相同的都交判官比图，相似度低也不例外；判官 unclear 前提不成立。"""
    from mspc.evaluation import cut_execution as c
    pairs = {'s/no-cut': {'planned_cut': True, 'cut_between': False, 'corr_mid': 0.5, 'corr_cut': 0.5},
             's/low': {'planned_cut': True, 'cut_between': True, 'corr_mid': 0.1, 'corr_cut': 0.3},
             's/same-middle': {'planned_cut': True, 'cut_between': True, 'corr_mid': 0.95, 'corr_cut': 0.5},
             's/judge': {'planned_cut': False, 'cut_between': True, 'corr_mid': 0.6, 'corr_cut': 0.7},
             's/alike-no-cut': {'planned_cut': False, 'cut_between': False, 'corr_mid': 0.2, 'corr_cut': 0.2}}
    c.g.write_json(tmp_path / 'pairs/row.json', pairs)
    assert sorted(k for _, k in c.pending(tmp_path, ['row'])) == ['s/judge', 's/low']
    assert c.decide(tmp_path, 'row')['undecided'] == 2
    c.g.write_json(tmp_path / 'receipts/row/s_judge.json', {'status': 'answered', 'answer': {'setup': 'unclear'}})
    c.g.write_json(tmp_path / 'receipts/row/s_low.json', {'status': 'answered', 'answer': {'setup': 'same'}})
    c.decide(tmp_path, 'row')
    verdicts = c.load(tmp_path / 'verdicts/row.json')
    assert {k: v['verdict'] for k, v in verdicts.items()} == {
        's/no-cut': 'no_cut', 's/low': 'same', 's/same-middle': 'same', 's/judge': 'unclear', 's/alike-no-cut': 'no_cut'}
    assert c.pending(tmp_path, ['row']) == []


def test_every_item_needs_a_detected_cut_somewhere_between_its_two_shots():
    from mspc.evaluation import cut_execution as c
    media = {'shot_frames': [16, 16, 16, 16], 'probe': {'fps': 16.0, 'frames': 64}}
    items = [{'item_id': 'g', 'reference_shot': 1, 'shot': 4, 'planned_cut': False},
             {'item_id': 'n', 'reference_shot': 2, 'shot': 3, 'planned_cut': True}]
    frame_jobs, cut = c.jobs(media, items, [3.1])  # 只在第 3、4 镜之间（3.0 秒）检出
    assert [j['item_id'] for j in frame_jobs] == ['g', 'n'] and cut == {'g': True, 'n': False}
    # 第 2、3 镜各自的首末帧（镜界后、前第 2 帧）；3.1 秒检出的切点把第 3 镜的末界从第 48 帧挪到第 50 帧。
    assert frame_jobs[1]['judge_frames'] == [18, 29, 34, 47]
    assert c.jobs(media, items, [])[1] == {'g': False, 'n': False}


def test_setup_pending_does_not_resend_refused_or_missing_batch_result(tmp_path):
    from mspc.evaluation import cut_execution as c
    row = 'test92-ours-text-renderer-planner-references'
    refused = 'cinedance-014083-b0000-shot01/c09'
    pair = {'planned_cut': True, 'cut_between': True, 'corr_mid': 0.1, 'corr_cut': 0.3}
    c.g.write_json(tmp_path / 'pairs' / f'{row}.json', {refused: pair, 'missing': pair, 'fresh': pair})
    c.g.write_json(tmp_path / 'receipts' / row / f'{c.safe(refused)}.json',
                   {'status': 'format_failed', 'raw': {'candidates': [{'finishReason': 'PROHIBITED_CONTENT'}]}})
    c.g.write_json(tmp_path / 'batches/closed.json', {'keys': [f'{row}|missing'], 'closed': True})
    assert c.pending(tmp_path, [row]) == [(row, 'fresh')]
    c.decide(tmp_path, row)
    assert c.load(tmp_path / 'verdicts' / f'{row}.json')[refused]['judge_setup'] == 'refused'


def test_setup_prompt_block_uses_the_recorded_refusal_rule(tmp_path):
    """实际 Batch 在 promptFeedback 拒收、没有 candidates；保留回执且不重发。"""
    from mspc.evaluation import cut_execution as c
    row, item = 'test92-final', 'cinedance-015962-b0002-shot01/c06'
    pair = {'planned_cut': True, 'cut_between': True, 'corr_mid': 0.1, 'corr_cut': 0.3}
    c.g.write_json(tmp_path / 'pairs' / f'{row}.json', {item: pair})
    record = c.judge_record({'key': f'{row}|{item}', 'response': fixture('cut-prompt-prohibited.response.json')})
    assert record['status'] == 'format_failed' and c.refused(record)
    c.g.write_json(tmp_path / 'receipts' / row / f'{c.safe(item)}.json', record)
    assert c.pending(tmp_path, [row]) == []
    assert c.decide(tmp_path, row)['undecided'] == 0
    verdict = c.load(tmp_path / 'verdicts' / f'{row}.json')[item]
    assert verdict['verdict'] == 'unclear' and verdict['judge_setup'] == 'refused'
    assert not c.refused({'status': 'format_failed', 'raw': {'promptFeedback': {'blockReason': 'OTHER'}}})


def test_setup_prepare_sends_judge_frames_to_the_bucket_once(tmp_path, monkeypatch):
    """真实评分项已有四帧桶地址时不再取帧；缺地址的项才取帧写桶。"""
    from mspc.evaluation import cut_execution as c
    data = fixture('cut-frame-cache-real.json')
    sid, item = data['story_id'], data['item']
    row, root, source = 'row', tmp_path / 'cut', tmp_path / 'source'
    c.g.write_json(source / 'rows' / row / 'prepared.json', {'stories': [{'story_id': sid}]})
    c.g.write_json(source / 'rows' / row / 'media-map.json', {sid: {**data['media'], 'remote_video_path': '/real-source'}})
    c.g.write_json(tmp_path / 'items.json', {'stories': {sid: {'items': [item]}}})
    c.g.write_json(tmp_path / 'cuts.json', {'rows': {row: {sid: {'detected_cut_times_seconds': data['detected_cut_times_seconds']}}}})
    c.g.write_json(root / 'similarity' / f'{row}.json', {item['item_id']: data['similarity']})
    calls, storage = [], object()

    def to_bucket(given, judge, given_row, jobs):
        calls.append(jobs)
        assert (given, judge, given_row) == (storage, 'cut', row)
        return {p['key']: [f"gs://b/cut/{row}/{sid}/{f:06d}.jpg" for f in p['judge_frames']] for j in jobs for p in j['pairs']}

    monkeypatch.setattr(c, 'frames_to_bucket', to_bucket)
    c.prepare(root, source, tmp_path / 'items.json', [tmp_path / 'cuts.json'], [row], storage)
    assert len(calls) == 1 and calls[0][0]['story'] == sid and len(calls[0][0]['pairs']) == 1
    assert len(c.load(root / 'images' / f'{row}.json')[item['item_id']]) == 4
    c.prepare(root, source, tmp_path / 'items.json', [tmp_path / 'cuts.json'], [row], storage)
    assert len(calls) == 1


def test_collect_writes_receipts_records_batch_cost_and_deletes_uploads(tmp_path):
    from mspc.evaluation import gemini_checklist_batch as b
    story = {**fixture('story.json'), 'method': 'text_renderer_planner_references', 'input_tier': 'shots', 'seed': 42,
             'story_id': 'cinedance-011901-b0000-shot01'}
    key = f"dev30-row|{story['story_id']}"
    b.write(tmp_path / 'config.json', {'rows': ['dev30-row'], 'judge': {**JUDGE, 'thinking': 'low', 'fps': 2, 'schema': schema('answer_first')}})
    b.write(tmp_path / 'rows/dev30-row/prepared.json', {'stories': [story]})
    record = {'display_name': 'checklist-x', 'name': 'batches/x', 'state': 'BATCH_STATE_PENDING', 'keys': [key],
              'reserve_usd': '0.01', 'uploads': {key: 'files/video'}, 'upload_bytes': 7_000_000,
              'input_file': 'files/jsonl'}
    b.write(tmp_path / 'batches/checklist-x.json', record)
    line = json.loads((FIXTURES / 'batch-results.jsonl').read_text().splitlines()[0])
    deleted = []

    class Client:
        def batch(self, name):
            return {'metadata': {'state': 'BATCH_STATE_SUCCEEDED'}, 'response': {'responsesFile': 'files/out'}}

        def download(self, name):
            return (json.dumps({**line, 'key': key}) + '\n').encode()

        def delete(self, name):
            deleted.append(name)

    assert b.collect(tmp_path, Client())[0]['results'] == {'rated': 1}
    assert b.load(b.receipt_path(tmp_path, 'dev30-row', story))['status'] == 'rated'
    closed = b.load(tmp_path / 'batches/checklist-x.json')
    assert closed['closed'] and closed['cost_usd'] == '0.004379625' and closed['cost_basis'] == 'usage'
    assert deleted == ['files/video', 'files/jsonl']
    assert g.batch_spent(tmp_path) == D('0.004379625') and b.pending(tmp_path, ['dev30-row']) == []


def test_video_pending_preserves_real_failed_receipt(tmp_path):
    from mspc.evaluation import gemini_checklist_batch as b
    row = 'test92-storymem'
    base = {'method': 'storymem', 'input_tier': 'shots', 'seed': 42, 'requirements': [{'requirement_id': 's/q1'}]}
    failed = {**base, 'story_id': 'film-008001-b0014-shot01'}
    queued = {**base, 'story_id': 'queued'}
    fresh = {**base, 'story_id': 'fresh'}
    missing = {**base, 'story_id': 'missing'}
    empty = {**base, 'story_id': 'film-008001-b0014-shot01-empty', 'requirements': []}
    b.write(tmp_path / 'rows' / row / 'prepared.json', {'stories': [failed, queued, fresh, missing, empty]})
    b.write(b.receipt_path(tmp_path, row, failed), {'status': 'format_failed', 'error': 'AssertionError: '})
    b.write(tmp_path / 'batches/open.json', {'keys': [f'{row}|queued'], 'closed': False})
    b.write(tmp_path / 'batches/closed.json', {'keys': [f'{row}|missing'], 'closed': True})
    assert b.pending(tmp_path, [row]) == [(row, fresh)]


def test_open_batch_checks_cost_cap_and_files_storage(tmp_path):
    g.open_batch(tmp_path, 'x', ['a'], D('4'), D('5'), 15 * 10**9)
    with pytest.raises(RuntimeError, match='cap'):
        g.open_batch(tmp_path, 'y', ['b'], D('2'), D('5'), 1)
    with pytest.raises(RuntimeError, match='Files storage'):
        g.open_batch(tmp_path, 'y', ['b'], D('0.5'), D('5'), 6 * 10**9)


def test_reason_first_schema_only_reorders_fields():
    a, r = schema('answer_first'), schema('reason_first')
    item_a, item_r = a['properties']['answers']['items'], r['properties']['answers']['items']
    assert list(item_r['properties']) == item_r['required'] == ['question', 'time', 'reason', 'answer']
    assert {k: v for k, v in item_r['properties'].items()} == item_a['properties']


def _remote_frame_functions(monkeypatch, out_dir=None):
    """执行远端实际使用的函数体，保留输入／输出协议，不创建进程池或访问服务器。"""
    import io
    import sys
    from mspc.evaluation import cut_execution as c
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps({'ffmpeg': 'ffmpeg', 'out_dir': str(out_dir)})))
    namespace = {}
    exec(c.REMOTE.split("if P['mode'] == 'similarity':", 1)[0], namespace)
    return namespace


def test_remote_jpeg_decodes_unique_frames_once_and_preserves_pair_order(tmp_path, monkeypatch):
    """同一真实切镜项的首末帧被多个镜对复用时，只解码一次、每帧写一个文件，逐 key 按位置回帧号。"""
    import re
    from types import SimpleNamespace
    data = fixture('cut-frame-cache-real.json')
    from mspc.evaluation import cut_execution as c
    pair_jobs, _ = c.jobs(data['media'], [data['item']], data['detected_cut_times_seconds'])
    frames = pair_jobs[0]['judge_frames']
    pairs = [{'key': 'second', 'judge_frames': frames[::-1]},
             {'key': 'first', 'judge_frames': [frames[0], frames[0], frames[2], frames[3]]}]
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        selected = [int(n) for n in re.findall(r'eq\(n\\,(\d+)\)', command[command.index('-vf') + 1])]
        for index, frame in enumerate(selected, 1):
            Path(command[-1] % index).write_bytes(f'jpeg-{frame}'.encode())
        return SimpleNamespace(returncode=0)

    namespace = _remote_frame_functions(monkeypatch, tmp_path)
    monkeypatch.setattr(namespace['subprocess'], 'run', run)
    output = namespace['jpeg']({'video': '/real-source', 'story': 's', 'pairs': pairs})
    assert len(calls) == 1
    assert calls[0][calls[0].index('-frames:v') + 1] == str(len(set(frames)))
    assert calls[0][calls[0].index('-q:v') + 1] == '2'
    assert [(o['key'], o['story'], o['frames']) for o in output] == [(p['key'], 's', p['judge_frames']) for p in pairs]
    assert sorted(path.name for path in (tmp_path / 's').iterdir()) == [f'{f:06d}.jpg' for f in sorted(set(frames))]
    assert all((tmp_path / 's' / f'{f:06d}.jpg').read_bytes() == f'jpeg-{f}'.encode() for f in frames)
    assert not Path(calls[0][-1]).parent.exists()  # 解码临时目录用完即删。


def test_remote_jpeg_rejects_missing_frames_and_cleans_temporary_files(tmp_path, monkeypatch):
    from types import SimpleNamespace
    namespace = _remote_frame_functions(monkeypatch, tmp_path)
    directories = []

    def run(command, **kwargs):
        directories.append(Path(command[-1]).parent)
        Path(command[-1] % 1).write_bytes(b'jpeg')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(namespace['subprocess'], 'run', run)
    with pytest.raises(ValueError, match='decoded 1 of 2 judge frames'):
        namespace['jpeg']({'video': '/real-source', 'story': 's', 'pairs': [{'key': 'a', 'judge_frames': [2, 5, 2, 5]}]})
    assert not directories[0].exists()


def _storage(tmp_path, monkeypatch):
    account = tmp_path / 'sa.json'
    account.write_text(json.dumps({'type': 'service_account', 'project_id': 'proj', 'client_email': 'sa@proj',
                                   'private_key_id': 'k', 'private_key': 'unused'}))
    storage = g.Storage(account, 'bucket', 'http://127.0.0.1:9')
    monkeypatch.setattr(storage, 'token', lambda: 'secret-token')
    return storage


def test_put_from_host_sends_token_only_on_stdin_and_parses_the_host_reply(tmp_path, monkeypatch):
    """判分桶上传：令牌只走标准输入；远端回执变 gs:// 地址或 GeminiError。"""
    import sys
    fake = tmp_path / 'fake_host.py'
    fake.write_text(
        "import json,sys\n"
        "a=json.loads(sys.stdin.readline())\n"
        "assert 'secret-token' not in ' '.join(sys.argv) and a['token']=='secret-token' and 'def main' in a['script']\n"
        "if a['items'][0]['size']==7: print(json.dumps({'objects':[{'bucket':a['bucket'],'name':i['object'],'size':'7'} for i in a['items']]}))\n"
        "else: print(json.dumps({'status':403,'body':'denied'})); sys.exit(3)\n")
    storage = _storage(tmp_path, monkeypatch)
    python = f'{sys.executable} {fake}'
    items = [{'path': '/v.mp4', 'object': 'video/row/s.mp4', 'mime': 'video/mp4', 'size': 7},
             {'path': '/w.mp4', 'object': 'video/row/t.mp4', 'mime': 'video/mp4', 'size': 7}]
    assert storage.put_from_host(['sh', '-c'], python, items) == ['gs://bucket/video/row/s.mp4', 'gs://bucket/video/row/t.mp4']
    with pytest.raises(g.GeminiError) as error:
        storage.put_from_host(['sh', '-c'], python, [{**items[0], 'size': 8}])
    assert error.value.status == 403


def test_host_script_checks_bytes_before_any_request(tmp_path, monkeypatch):
    import sys
    video = tmp_path / 'v.mp4'
    video.write_bytes(b'12345')
    (tmp_path / 'p.txt').write_text('export https_proxy=http://127.0.0.1:9\n')
    monkeypatch.setattr(g, 'HOST_PROXY_FILE', str(tmp_path / 'p.txt'))
    storage = _storage(tmp_path, monkeypatch)
    with pytest.raises(OSError) as error:
        storage.put_from_host(['sh', '-c'], sys.executable, [{'path': str(video), 'object': 'r/s.mp4', 'mime': 'video/mp4', 'size': 6}])
    assert 'expected 6' in str(error.value) and 'secret-token' not in str(error.value)


def test_register_sends_the_gs_uris_with_the_api_key(monkeypatch):
    client = g.Gemini('api-key', 'http://127.0.0.1:9')
    sent = {}

    def json_call(url, payload=None, method=None):
        sent.update(url=url, payload=payload, method=method)
        return {'files': [{'name': 'files/a', 'uri': 'u', 'state': 'ACTIVE'}]}
    monkeypatch.setattr(client, 'json', json_call)
    assert client.register(['gs://bucket/r/s.mp4'])[0]['name'] == 'files/a'
    assert sent == {'url': f'{g.BASE}/v1beta/files:register', 'payload': {'uris': ['gs://bucket/r/s.mp4']}, 'method': 'POST'}
