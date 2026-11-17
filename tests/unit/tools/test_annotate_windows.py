"""块入口的选键、重发与费用预留；只用本地真实清单形状，不调接口。"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[3]/'tools/data'))
from annotate_windows import annotation_paths, selected_rows, request_line, SpendLedger, price


def test_paths_and_registry_only_resend(tmp_path):
    (tmp_path/'final').mkdir()
    records = [{'key': 'a', 'status': 'ok', 'block_errors': []},
               {'key': 'b', 'status': 'invalid', 'block_errors': [], 'shot_errors': {'1': ['bad holding']}},
               {'key': 'c', 'status': 'invalid', 'block_errors': ['story block']},
               {'key': 'd', 'status': 'no_answer', 'block_errors': ['no content']}]
    (tmp_path/'final/blocks.jsonl').write_text(''.join(json.dumps({'key':r['key']})+'\n' for r in records))
    (tmp_path/'final/block-annotations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    a=SimpleNamespace(unit='block', round=1, keys_file=None, limit=0)
    assert [r['key'] for r in selected_rows(a,tmp_path)] == ['c','d']
    keys=tmp_path/'keys.txt';keys.write_text('b\nd\n');a.keys_file=keys
    assert [r['key'] for r in selected_rows(a,tmp_path)] == ['d']
    assert annotation_paths(tmp_path,'window')['ledger'].name == 'annotation-batches.json'
    assert annotation_paths(tmp_path,'block',2)['ledger'].name == 'block-annotation-batches-r2.json'
    assert annotation_paths(tmp_path,'block',2)['status'].name == 'annotate-blocks-r2'


def test_batch_body_preserves_prompt_and_timestamp_contract():
    contract=json.loads((Path(__file__).parents[3]/'configs/data/block_annotation_v18.json').read_text())
    row={'key':'film-case','proxy':{'shot_timestamps':[{'shot':0,'start':0.,'end':4.5}]}}
    payload=json.loads(request_line(row,contract,'https://example.invalid/video'))
    body=payload['body']
    assert body['messages'][0]['content'][1]['text'] == contract['prompt']+'\nShot 1: 0.0–4.5 s'
    assert body['messages'][0]['content'][0]['fps'] == 2
    assert 'max_tokens' not in body


def test_shared_budget_never_reserves_twice_or_releases_unknown(tmp_path):
    one=SpendLedger(tmp_path/'spend.json',6)
    two=SpendLedger(tmp_path/'spend.json',6)
    assert one.reserve('rt',1.2)
    assert two.reserve('batch',4.8)
    assert not one.reserve('extra',.01)
    assert not two.reserve('rt',1.2)
    one.settle('rt',None)
    assert not two.reserve('extra',.01)
    two.settle('batch',.02)
    assert one.reserve('extra',1.2)
    assert abs(price({'prompt_tokens':40000,'completion_tokens':1000},True)-.0132)<1e-12


def test_block_batch_entry_submits_fetches_and_settles_without_network(monkeypatch,tmp_path):
    import annotate_windows as module
    root=Path(__file__).parents[3]
    fixture=json.loads((root/'tests/unit/tools/fixtures/blocks_v17/film-018940-b0001.json').read_text())
    contract=json.loads((root/'configs/data/block_annotation_v18.json').read_text())
    (tmp_path/'final').mkdir()
    row={'key':fixture['key'],'source':'cinepile_t2v','scene':{'shot_count':fixture['shots']},
         'proxy':{'shot_timestamps':[{'shot':i,'start':float(i),'end':float(i+1)} for i in range(fixture['shots'])]}}
    (tmp_path/'final/blocks.jsonl').write_text(json.dumps(row)+'\n')
    submitted=[]

    class FakeApi:
        def __init__(self,*args):pass
        def clean(self,text):return text
        def upload(self,name,payload):
            submitted.append(json.loads(payload))
            return {'id':'file-input'}
        def call(self,method,path,data=None,headers=None,raw=False):
            if path=='/batches':return {'id':'batch-test','status':'validating'}
            if path=='/batches/batch-test':
                return {'status':'completed','output_file_id':'output','request_counts':{'completed':1}}
            if path=='/files/output/content':
                return (json.dumps({'custom_id':fixture['key'],'response':{'body':{'choices':[{'message':{'content':json.dumps(fixture['answer'])}}],
                    'usage':{'prompt_tokens':1000,'completion_tokens':2000}}}})+'\n').encode()
            raise AssertionError(path)
    monkeypatch.setattr(module,'Api',FakeApi)
    a=SimpleNamespace(unit='block',round=0,keys_file=None,limit=0,lines_per_file=5,bucket='test',endpoint='example.invalid',
          prefix='blocks-v17-trial-test',spend_ledger=tmp_path/'spend.json',budget_cny=6,reserve_output_tokens=131072+262144,time_limit_seconds=30)
    module.run(a,tmp_path,contract,'fake-id','fake-secret','fake-key')
    assert len(submitted)==1
    assert submitted[0]['custom_id']==fixture['key']
    ledger=json.loads((tmp_path/'final/block-annotation-batches.json').read_text())
    assert ledger['batches'][0]['fetched']
    assert (tmp_path/'block-annotations/batch-00.output.jsonl').exists()
    assert (tmp_path/'status/annotate-blocks-s0of1.json').exists()
    spent=json.loads((tmp_path/'spend.json').read_text())
    assert abs(next(iter(spent.values()))['actual_cny']-.0009)<1e-12
    # 续跑不重复创建远端任务。
    module.run(a,tmp_path,contract,'fake-id','fake-secret','fake-key')
    assert len(submitted)==1
    assert 'OSSAccessKeyId' not in (tmp_path/'final/block-annotation-batches.json').read_text()


def test_row_level_errors_release_their_reservation_and_missing_rows_keep_it(monkeypatch,tmp_path):
    import annotate_windows as module
    root=Path(__file__).parents[3]
    fixture=json.loads((root/'tests/unit/tools/fixtures/blocks_v17/film-018940-b0001.json').read_text())
    contract=json.loads((root/'configs/data/block_annotation_v18.json').read_text())
    (tmp_path/'final').mkdir()
    keys=[fixture['key'],'cinedance-011957-b0000','film-000001-b0000']
    rows=[{'key':key,'source':'cinepile_t2v','scene':{'shot_count':fixture['shots']},
           'proxy':{'shot_timestamps':[{'shot':i,'start':float(i),'end':float(i+1)} for i in range(fixture['shots'])]}} for key in keys]
    (tmp_path/'final/blocks.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    # 全量第 0 轮真实返回的失败行（请求号已去掉）。
    failed={'custom_id':keys[1],'response':{'status_code':400,'body':{'error':{'code':'invalid_parameter_error','param':None,
            'message':'<400> InternalError.Algo.InvalidParameter: Download multimodal file timed out','type':'invalid_request_error'}}},
            'error':{'code':'invalid_parameter_error','param':None,
            'message':'<400> InternalError.Algo.InvalidParameter: Download multimodal file timed out','type':'invalid_request_error'}}

    class FakeApi:
        def __init__(self,*args):pass
        def clean(self,text):return text
        def upload(self,name,payload):return {'id':'file-input'}
        def call(self,method,path,data=None,headers=None,raw=False):
            if path=='/batches':return {'id':'batch-test','status':'validating'}
            if path=='/batches/batch-test':
                return {'status':'completed','output_file_id':'output','error_file_id':'error','request_counts':{'completed':1,'failed':1}}
            if path=='/files/output/content':
                return (json.dumps({'custom_id':keys[0],'response':{'body':{'choices':[{'message':{'content':json.dumps(fixture['answer'])}}],
                    'usage':{'prompt_tokens':1000,'completion_tokens':2000}}}})+'\n').encode()
            if path=='/files/error/content':return (json.dumps(failed)+'\n').encode()
            raise AssertionError(path)
    monkeypatch.setattr(module,'Api',FakeApi)
    a=SimpleNamespace(unit='block',round=0,keys_file=None,limit=0,lines_per_file=5,bucket='test',endpoint='example.invalid',
          prefix='blocks-test',spend_ledger=tmp_path/'spend.json',budget_cny=6,reserve_output_tokens=10000,time_limit_seconds=30)
    module.run(a,tmp_path,contract,'fake-id','fake-secret','fake-key')
    spent=next(iter(json.loads((tmp_path/'spend.json').read_text()).values()))
    kept=module.batch_reservation(rows[2],contract,10000)
    assert abs(spent['actual_cny']-.0009)<1e-12
    assert abs(spent['charged_or_reserved_cny']-(.0009+kept))<1e-9
    assert (spent['usage_accounted'],spent['failed_unbilled'])==(1,1)


def test_signed_url_is_removed_before_secret_redaction():
    from annotate_windows import Api
    api=Api('https://example.invalid','fake-key',('fake-id','fake-secret','fake-key'))
    value=api.clean('failed https://bucket.example/a.mp4?OSSAccessKeyId=fake-id&Expires=123&Signature=opaque')
    assert value=='failed <signed-url-redacted>'
