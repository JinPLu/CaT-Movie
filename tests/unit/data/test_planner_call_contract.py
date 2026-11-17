"""真实200窗产物中的一窗与原响应；fake transport仅验证HTTP边界，不是模型成绩。"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from mspc.data.event_log import EventLog
from mspc.data.planner_baseline import baseline_messages, history_frames, request_payload, request_once

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / 'tests/fixtures/planner_call'
CASE = json.loads((FIXTURE / 'real_case.json').read_text())
from mspc.data.event_planner_contract import planner_output_contract
CONTRACT = planner_output_contract()


def words(text):
    return len(text.split())


def config(name='zero_shot.json'):
    return json.loads((ROOT / 'configs/planner' / name).read_text())


@pytest.mark.parametrize('name,t,p,presence,thinking,effort', [
    ('zero_shot.json', .7, .8, 1.5, False, None),
    ('zero_shot_qwen35_9b_thinking.json', 1., .95, 1.5, True, None),
    ('zero_shot_qwen38_27b_nonthinking.json', .7, .8, 1.5, False, None),
    ('zero_shot_qwen38_27b_thinking.json', 1., .95, 0., True, 'xhigh'),
])
def test_all_official_general_recipes_reach_request(name,t,p,presence,thinking,effort):
    c=config(name)
    r=request_payload(c['model'], [], c, schema=CONTRACT['schema'])
    assert (r['temperature'],r['top_p'],r['top_k'],r['min_p']) == (t,p,20,0)
    assert (r['presence_penalty'],r['repetition_penalty']) == (presence,1)
    assert r['chat_template_kwargs']['enable_thinking'] is thinking
    assert r['chat_template_kwargs'].get('reasoning_effort') == effort
    assert 'reasoning_effort' not in r
    assert r['seed'] == 42 and r['max_tokens'] == 32768
    assert r['response_format']['type'] == 'json_schema'
    assert r['response_format']['json_schema']['schema'] == CONTRACT['schema']


def test_real_history_only_adds_pixels_and_keeps_the_same_text():
    c=config(); c['schema_version']=CASE_SCHEMA=CONTRACT['schema']['properties']['schema']['const']
    case=copy.deepcopy(CASE); case['history']['true'][0]['path']=str(FIXTURE/'real_history.jpg')
    text=baseline_messages(case,'script_only',c,CONTRACT)
    image=baseline_messages(case,'script_true_history',c,CONTRACT)
    assert image[-1]['content'][0] == text[-1]['content'][0]
    assert len(text[-1]['content']) == 1 and len(image[-1]['content']) == 3
    assert image[-1]['content'][2]['image_url']['url'].startswith('data:image/png;base64,')
    assert CASE_SCHEMA == 'event_log_v3'


@pytest.mark.parametrize('change', ['shot','time','source','count','missing'])
def test_real_history_rejects_bad_boundary(change):
    c=config();case=copy.deepcopy(CASE); f=case['history']['true'][0]
    if change=='shot': f['shot_id']=case['target_start_shot_id']
    if change=='time': f['timestamp_seconds']=case['target_start_seconds']
    if change=='source': f['media_kind']='generated'
    if change=='count': c['image_quota']=0
    if change=='missing': del f['timestamp_seconds']
    with pytest.raises((ValueError,KeyError)):
        history_frames(case,'script_true_history',c)


def test_real_case_from_first_shot_has_no_history_and_one_request_for_all_arms():
    """新样本从 s01 起、不带历史：基线照常构造，三臂都是零张图、同一段请求文字。"""
    from mspc.data.planner_baseline import build_baseline_case
    c=config(); case=copy.deepcopy(CASE)
    example={'sample_id':case['case_id'],'case_id':'ci_vid:train:000000753','film_key':'ci_vid:000000753',
             'script':case['script'],'target_start_shot_id':'s01',
             'history_frames':[],'event_log':case['gold_event_log']}
    produced=build_baseline_case(example,history={'true':[],'generated':[]},target_start_seconds=0)
    assert all(history_frames(produced,mode,c)==[] for mode in ('script_only','script_true_history','script_generated_history'))
    messages=[baseline_messages(produced,mode,c,CONTRACT) for mode in ('script_only','script_true_history','script_generated_history')]
    assert messages[0]==messages[1]==messages[2] and len(messages[0][-1]['content'])==1
    request=json.loads(messages[0][-1]['content'][0]['text'].split('规划请求：',1)[1])
    assert request['target_start_shot_id']=='s01' and request['history_frames']==[]


def test_missing_recipe_value_fails_before_dispatch():
    c=config();del c['sampling']['min_p']
    with pytest.raises(ValueError,match='six'):
        request_payload(c['model'],[],c,schema=CONTRACT['schema'])
    c=config();c['enable_thinking']=0
    with pytest.raises(TypeError):
        request_payload(c['model'],[],c,schema=CONTRACT['schema'])


def test_thinking_controls_have_one_config_owner():
    c = config('zero_shot_qwen38_27b_thinking.json')
    c['chat_template_kwargs']['reasoning_effort'] = 'low'
    with pytest.raises(ValueError, match='one config owner'):
        request_payload(c['model'], [], c, schema=CONTRACT['schema'])


def test_real_multimodal_receiver_cache_failure_preserves_http_body(monkeypatch):
    import io
    from urllib.error import HTTPError
    failed = json.loads((FIXTURE / '27b_real_mm_cache_failure.json').read_text())
    c = config('zero_shot_qwen38_27b_thinking.json')
    case = copy.deepcopy(CASE)
    case['history']['true'][0]['path'] = str(FIXTURE / 'real_history.jpg')
    messages = baseline_messages(case, 'script_true_history', c, CONTRACT)
    def dispatch(request, **kwargs):
        assert kwargs['timeout'] == 3600
        raise HTTPError(request.full_url, failed['http_status'], 'Internal Server Error',
                        {}, io.BytesIO(failed['raw_response'].encode()))
    monkeypatch.setattr('urllib.request.urlopen', dispatch)
    row = request_once('http://localhost/v1', c['model'], messages, c,
                       schema=CONTRACT['schema'])
    assert row['status'] == 'transport_failed' and row['http_status'] == 500
    assert row['raw_response'] == failed['raw_response']
    assert row['raw_content'] == '' and row['usage'] is None


def test_sol_route_has_high_reasoning_schema_and_explicit_budget():
    c=config('zero_shot_gpt61_sol.json')
    r=request_payload('gpt-6.1-sol',[],c,schema=CONTRACT['schema'])
    assert r['reasoning_effort']=='high' and r['max_completion_tokens']==32768
    assert 'chat_template_kwargs' not in r and 'temperature' not in r
    assert c['budget_hard_cap_usd']=='12.00' and c['concurrency']==1
    assert c['modes']==['script_only','script_true_history']


@pytest.mark.parametrize('finish,content,status', [('stop','{}','completed'),('length','{}','incomplete'),('stop',None,'empty_final')])
def test_transport_preserves_reasoning_and_failure_denominators(monkeypatch,finish,content,status):
    body={'id':'fx2-unit','model':'served-qwen','choices':[{'finish_reason':finish,'message':{'content':content,'reasoning':'trace'}}], 'usage':{'prompt_tokens':10,'completion_tokens':20}}
    class Response:
        status=200
        def __enter__(self):return self
        def __exit__(self,*args):return None
        def read(self):return json.dumps(body).encode()
    monkeypatch.setattr('urllib.request.urlopen',lambda *args,**kwargs:Response())
    r=request_once('http://localhost/v1','Qwen3.5-9B',[],config(),schema=CONTRACT['schema'])
    assert r['status']==status and r['finish_reason']==finish
    assert r['raw_reasoning']=='trace' and r['served_model']=='served-qwen'
    assert json.loads(r['raw_response'])==body


def test_v2_producer_and_full_runner_keep_raw_identity_and_all_failures(tmp_path, monkeypatch):
    from mspc.data.planner_baseline import build_baseline_case, run_baseline
    case=copy.deepcopy(CASE)
    frame=case['history']['true'][0]; frame['path']=str(FIXTURE/'real_history.jpg')
    example={'sample_id':case['case_id'],'case_id':'ci_vid:train:000000753','film_key':'ci_vid:000000753',
             'script':case['script'],
             'target_start_shot_id':case['target_start_shot_id'],
             'history_frames':[{'shot_id':frame['shot_id'],'path':frame['path']}],
             'event_log':case['gold_event_log']}
    produced=build_baseline_case(example,history=case['history'],target_start_seconds=case['target_start_seconds'])
    assert produced==case
    benchmark=tmp_path/'benchmark.json';benchmark.write_text(json.dumps({'cases':[case],'output_contract':CONTRACT}))
    c=config();c.update(expected_benchmark_cases=1,modes=['script_only','script_true_history'])
    calls=[]
    class Response:
        status=200
        def __enter__(self):return self
        def __exit__(self,*args):return None
        def read(self):
            return json.dumps({'id':'actual-id','model':'actual-revision','provider':'test-backend',
              'choices':[{'finish_reason':'length' if len(calls)==2 else 'stop',
                          'message':{'content':json.dumps(case['gold_event_log']),'reasoning_content':'trace'}}],
              'usage':{'prompt_tokens':100,'completion_tokens':30}}).encode()
    def dispatch(req,**kwargs):
        calls.append(json.loads(req.data));return Response()
    monkeypatch.setattr('urllib.request.urlopen',dispatch)
    run_baseline(benchmark=benchmark,output=tmp_path/'out',base_url='http://localhost/v1',model='Qwen3.5-9B',config=c,token_counter=words)
    rows=[json.loads((tmp_path/'out'/f'{mode}.jsonl').read_text()) for mode in c['modes']]
    assert [r['status'] for r in rows]==['completed','incomplete']
    assert all(r['returned_model']=='actual-revision' and r['provider']=='test-backend' for r in rows)
    assert all(r['evaluation']['failed_layer'] is None for r in rows)
    assert rows[0]['history_ledger']==[] and rows[1]['history_ledger'][0]['sent_size']
    assert calls[0]['messages'][-1]['content'][0]==calls[1]['messages'][-1]['content'][0]
    assert 'gold_event_log' not in calls[0]['messages'][-1]['content'][0]['text']
    summary=json.loads((tmp_path/'out'/'script_true_history.summary.json').read_text())
    assert summary['cases']==1 and summary['compilability']['compile']['rate']==1


def test_shot_locks_reach_request_and_conflicts_are_visible():
    from mspc.data.planner_baseline import validate_baseline_case,evaluate_baseline_response
    case=copy.deepcopy(CASE)
    validate_baseline_case(case)
    messages=baseline_messages(case,'script_only',config(),CONTRACT)
    request=json.loads(messages[-1]['content'][0]['text'].split('规划请求：',1)[1])
    gold=case['gold_event_log']['shots']
    assert request['shots']==[{'shot_id':s['shot_id'],'frame_count':s['frame_count']} for s in gold]
    assert 'subjects' not in request and 'locations' not in request  # 金标主体与地点不进请求
    wrong=copy.deepcopy(case['gold_event_log']);wrong['shots'][1]['frame_count']+=4
    result=evaluate_baseline_response(json.dumps(wrong),case,token_counter=words)
    assert result['failed_layer']=='input_lock'


def test_closed_unknown_billing_stops_before_second_request(tmp_path,monkeypatch):
    from mspc.data.planner_baseline import run_baseline
    case=copy.deepcopy(CASE);case['history']['true'][0]['path']=str(FIXTURE/'real_history.jpg')
    benchmark=tmp_path/'benchmark.json';benchmark.write_text(json.dumps({'cases':[case],'output_contract':CONTRACT}))
    c=config('zero_shot_gpt61_sol.json');c['expected_benchmark_cases']=1
    calls=[]
    class Response:
        status=200
        def __enter__(self):return self
        def __exit__(self,*args):return None
        def read(self):return b'{"id":"charged-but-no-usage","choices":[]}'
    def dispatch(*args,**kwargs):calls.append(1);return Response()
    monkeypatch.setenv('OPENAI_API_KEY','test-only-secret')
    monkeypatch.setattr('urllib.request.urlopen',dispatch)
    with pytest.raises(RuntimeError,match='reconcile unknown billing'):
        run_baseline(benchmark=benchmark,output=tmp_path/'out',base_url='http://localhost/v1',model='gpt-6.1-sol',config=c,token_counter=words)
    assert len(calls)==1
    row=json.loads((tmp_path/'out'/'script_only.jsonl').read_text())
    assert row['status']=='billing_unknown' and 'charged-but-no-usage' in row['raw_response']


def test_missing_generated_history_rejected_before_any_dispatch(tmp_path,monkeypatch):
    from mspc.data.planner_baseline import run_baseline
    case=copy.deepcopy(CASE);case['history']['true'][0]['path']=str(FIXTURE/'real_history.jpg')
    benchmark=tmp_path/'benchmark.json';benchmark.write_text(json.dumps({'cases':[case],'output_contract':CONTRACT}))
    c=config();c['expected_benchmark_cases']=1
    calls=[];monkeypatch.setattr('urllib.request.urlopen',lambda *a,**kw:calls.append(1))
    with pytest.raises(KeyError,match='generated'):
        run_baseline(benchmark=benchmark,output=tmp_path/'out',base_url='http://localhost/v1',model='Qwen3.5-9B',config=c,token_counter=words)
    assert not calls and not (tmp_path/'out').exists()


@pytest.mark.parametrize("service_name,text_only", [("vllm_planner_qwen35_9b.yaml", True), ("vllm_planner_qwen38_27b.yaml", False)])
def test_service_snapshot_uses_service_interpreter_and_preserves_raw_template(tmp_path, monkeypatch, service_name, text_only):
    import yaml
    from mspc.inference.planner_serving import save_service_provenance, service_command
    c=yaml.safe_load((ROOT/'configs/serving'/service_name).read_text())
    model=tmp_path/'model';model.mkdir()
    (model/'chat_template.jinja').write_text('real template fixture {{ messages }}')
    metadata=model/'.cache/huggingface/download';metadata.mkdir(parents=True)
    (metadata/'chat_template.jinja.metadata').write_text('fixture-revision\netag\n0\n')
    binary=tmp_path/'env/bin/vllm';binary.parent.mkdir(parents=True)
    binary.write_text('#!/target/env/bin/python\n')
    c.update(model_path=str(model),vllm_executable=str(binary))
    source=tmp_path/'service.yaml';source.write_text(yaml.safe_dump(c))
    calls=[]
    def inspect(command,**kwargs):
        calls.append(command)
        return json.dumps({'python':str(binary.parent/'python'),'packages':{'transformers':{'version':'5.8.0'}}})
    monkeypatch.setattr('subprocess.check_output',inspect)
    result=save_service_provenance(source,c,tmp_path/'snapshot')
    assert calls[0][0]==str(binary.parent/'python')
    assert result['status']=='prelaunch_metadata_only'
    assert result['command']==service_command(c)
    assert result['command'][result['command'].index('--max-model-len')+1]=='131072'
    assert result['command'][result['command'].index('--chat-template-content-format')+1]=='openai'
    if text_only:
        assert '--language-model-only' in result['command'] and '--enforce-eager' not in result['command']
    else:
        assert result['command'][result['command'].index('--mm-processor-cache-gb')+1]=='0'
    template=next(r for r in result['model_files'] if r['file']=='chat_template.jinja')
    assert template['download_revision']=='fixture-revision'
    absent=next(r for r in result['model_files'] if r['file']=='processor_config.json')
    assert not absent['present'] and absent['download_revision'] is None
    assert (tmp_path/'snapshot/model_files/chat_template.jinja').read_text()=='real template fixture {{ messages }}'


@pytest.mark.parametrize("case_count", [1, 2])
def test_raw_response_survives_observed_validator_crash(tmp_path, monkeypatch, case_count):
    from mspc.data.planner_baseline import run_baseline
    case = copy.deepcopy(CASE)
    case["history"]["true"][0]["path"] = str(FIXTURE / "real_history.jpg")
    benchmark = tmp_path / "benchmark.json"
    cases = [dict(case, case_id=f"transport-{i}") for i in range(case_count)]
    benchmark.write_text(json.dumps({"cases": cases, "output_contract": CONTRACT}))
    c = config(); c.update(expected_benchmark_cases=case_count, modes=["script_only"])
    raw = json.dumps(case["gold_event_log"])
    body = {"model": c["model"], "choices": [{"finish_reason": "stop", "message": {"content": raw}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}}
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return json.dumps(body).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: Response())
    def fail(*args, **kwargs): raise IndexError("list index out of range")
    monkeypatch.setattr("mspc.data.planner_baseline.evaluate_baseline_response", fail)
    with pytest.raises(IndexError):
        run_baseline(benchmark=benchmark, output=tmp_path / "out", base_url="http://localhost/v1",
                     model=c["model"], config=c, token_counter=words)
    rows = [json.loads(line) for line in (tmp_path / "out/script_only.responses.jsonl").read_text().splitlines()]
    assert len(rows) == case_count
    row = rows[0]
    assert row["raw_content"] == raw
    assert json.loads(row["raw_response"]) == body
    assert "evaluation" not in row
    assert (tmp_path / "out/script_only.jsonl").read_text() == ""


def test_real_9b_malformed_spans_are_rejected_by_canonical_schema():
    from mspc.data.planner_baseline import evaluate_baseline_response
    row = json.loads((FIXTURE / "9b_real_span_failure.json").read_text())
    result = evaluate_baseline_response(row["raw_content"], CASE, token_counter=words)
    assert result["passed"] == ["json"] and result["failed_layer"] == "schema"
    spans = [issue for issue in result["issues"] if "shot_span" in issue["path"]]
    assert len(spans) == 4
    # 这条真实原答复还带着 10-09 已删的 style/segments/cut_type 键，现由 schema 一并拒收。
    assert all(issue["message"] == "Extra inputs are not permitted"
               and any(key in issue["path"] for key in ("style", "segments", "cut_type"))
               for issue in result["issues"] if issue not in spans)


def test_served_schema_exposes_exact_pair_for_real_span_failure():
    from mspc.data.event_log import EventLog
    from mspc.data.state_facts import FactWindow
    for owner in (EventLog, FactWindow):
        span = owner.model_json_schema()["$defs"]["AtomicFact"]["properties"]["shot_span"]
        assert span["minItems"] == span["maxItems"] == 2
        assert "[0,0]" in span["description"]
