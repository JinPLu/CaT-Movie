"""给定计划逐项清单：一次请求、全英文，按题号回收答案；不改整片五项评分器。"""
from __future__ import annotations
import argparse, base64, copy, json, math, os, subprocess, tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from mspc.data.openrouter_client import OpenRouterClient as SdkClient, OpenRouterCostCounter as SdkCounter
from mspc.evaluation.question_writer import QUESTION_DIMENSIONS, judge_prompt

# 一次请求、全英文、只留判断说明。
# 只答yes／no，看不清即no（StoryEval、TC-Bench、VBench-2.0均为二值，模糊判未完成）。
# 视频在前、文字在后，时间写m:ss，上下文在前、问题与说明在最后（Gemini视频理解与提示策略官方文档）。
ANSWER = {'type':'object','additionalProperties':False,'required':['question','answer','reason','time'],'properties':{
 'question':{'type':'integer'},'answer':{'type':'string','enum':['yes','no']},
 'reason':{'type':'string','description':
  'For both yes and no, give brief visual evidence in this order: '
  '[Initial -> Final] the observed state at the start and end of the question\'s time range; '
  '[Transition] the specific observed action trajectory or state features, including any absence of a required transition. '
  'For an event action, answer no if the target pose or state is already present initially and stays static without the required transition. '
  'Identify people shown from behind or in profile using the registry\'s hairstyle, build and clothing. '
  'An interaction partly cropped at the frame edge can support yes when its action trajectory and interaction are clearly visible. '
  'If a state or transition cannot be seen, say so rather than infer it from the question or registry.'},
 'time':{'type':'string'}}}
RESPONSE = {'type':'json_schema','json_schema':{'name':'story_checklist','strict':True,'schema':{
 'type':'object','additionalProperties':False,'required':['answers'],'properties':{'answers':{'type':'array','items':ANSWER}}}}}

def now(): return datetime.now(timezone.utc).isoformat()
def write(path, data):
 path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def make_prompt(story, questions=None):
 """打分器文字由 question_writer.judge_prompt 生成：原请求、该视频各镜时间段、编号问题、判断说明。"""
 return judge_prompt(story['request'], story['shot_seconds'], story['requirements'] if questions is None else questions)

def story_key(story):
 return tuple(story[k] for k in ('method','input_tier','seed','story_id'))

def parts(story, limit):
 """题数超过单次上限时按原顺序均分成若干次请求，同视频、同题面。
 只含新增题的题库对某些故事没有题：这些故事不发请求，汇总记为不适用（checklist_rolling.summarize_row）。"""
 rows=story['requirements']
 if not rows:return []
 count=math.ceil(len(rows)/limit);size=math.ceil(len(rows)/count)
 return [rows[i:i+size] for i in range(0,len(rows),size)]

def requested_reads(config):
 lookup={story_key(s):s for s in config['stories']}
 keys=[story_key(s) for s in config['rereads']]
 assert len(keys)==len(set(keys))
 limit=config['judge']['max_questions_per_request']
 reads=[]
 for phase,rows in (('first-read',config['stories']),('reread',[lookup[k] for k in keys])):
  for s in rows:
   chunks=parts(s,limit)
   reads+=[(phase,s,index+1 if len(chunks)>1 else None,chunk) for index,chunk in enumerate(chunks)]
 return reads

def request_path(output, phase, story, part=None):
 name=story['story_id']+(f'.part-{part}' if part else '')+'.json'
 return output/'metrics'/phase/story['method']/story['input_tier']/f"seed-{story['seed']}"/name


def validate_config(config):
 rows=config['stories'];assert rows or config['unavailable_stories']
 assert len({story_key(s) for s in rows})==len(rows)
 for s in rows:
  assert all(isinstance(s[k],str) and s[k] and '/' not in s[k] for k in ('method','input_tier','story_id'))
  assert type(s['seed']) is int and s['request'].strip() and s['shot_seconds']
  assert abs(sum(s['shot_seconds'])-s['duration_seconds'])<0.1
 requested_reads(config)
 all_ids=[]
 for s in rows:
  for q in s['requirements']:
   assert q['dimension'] in QUESTION_DIMENSIONS and q['question'] and 1<=q['shot']<=len(s['shot_seconds'])
   all_ids.append((story_key(s),q['requirement_id']))
 assert len(all_ids)==len(set(all_ids))
 return {'stories':len(rows),'shots':sum(len(s['shot_seconds']) for s in rows),'requirements':len(all_ids)}

def validate_response(data, story, questions=None):
 """答案按题号对回内部题目；不按时间戳改答案，时间只作复核依据。"""
 rows=story['requirements'] if questions is None else questions
 answers=data['answers'];bynum={a['question']:a for a in answers}
 assert len(answers)==len(bynum) and set(bynum)==set(range(1,len(rows)+1))
 out=[]
 for number,q in enumerate(rows,1):
  a=bynum[number];assert a['answer'] in ('yes','no') and a['reason'].strip()
  out.append({'requirement_id':q['requirement_id'],'question':number,'answer':a['answer'],'reason':a['reason'],'time':a['time']})
 return {'answers':out}

def prepare(config, manifest):
 """仅绑定冻结题目与每种方法的实际镜头时间表；不读取key、不请求判官。

 时间表由生成产物或逐镜对齐记录显式给出，不能用共同分镜表的计划帧数兜底。
 v3题库目前只覆盖共同计划的逐镜档；其他输入档不能套用这份题库。
 """
 from mspc.config.shared_root import expand_shared_root_token
 requests=json.loads(Path(config['source']['requests']).read_text())['stories']
 catalog=json.loads(Path(expand_shared_root_token(config['source']['questions'], 'checklist questions')).read_text())['stories']
 selected={s['story_id']:s for s in requests if s['split']==manifest['split']}
 if manifest['split'] not in ('dev','test') or not selected:
  raise ValueError('split must be dev or test')
 # 这一行要判的故事：整集行为全集，只出受影响故事的子集行（11-15 起）为它实际出的那几个；分母就是这些故事。
 wanted=set(manifest['story_ids'])
 if not wanted or not wanted<=set(selected):
  raise ValueError('story_ids must be a non-empty part of the selected split')
 if manifest['evaluation_scope']!=config['evaluation_scope']:
  raise ValueError('evaluation scope differs from frozen configuration')
 rows=[];unavailable=[];groups={}
 for task in manifest['requests']:
  sid=task['story_id'];story=selected[sid];item=catalog[sid]
  if task['input_tier']!=config['input_tier'] or item['request']!=story['inputs']['shots']['text']:
   raise ValueError(f'question request differs: {sid}')
  groups.setdefault((task['method'],task['input_tier'],task['seed']),[]).append(sid)
  generation_path=task['generation_record_path']
  generation=json.loads(Path(generation_path).read_text()) if generation_path is not None else {'status':'pre_existing'}
  row={**task,'request':item['request'],'requirements':item['requirements'],'generation':generation}
  if generation['status'] not in ('completed','pre_existing'):
   if generation['status'] not in ('planning_failed','adaptation_failed','generation_failed'):
    raise ValueError(f'generation is not terminal: {sid}: {generation["status"]}')
   if any(task[k] is not None for k in ('shot_seconds','duration_seconds','timing_source')):
    raise ValueError(f'unavailable video must not have a timing table: {sid}')
   unavailable.append(row)
   continue
  seconds=task['shot_seconds']
  if len(seconds)!=story['inputs']['shots']['expected_shot_count'] or any(type(t) not in (float,int) or not math.isfinite(t) or t<=0 for t in seconds):
   raise ValueError(f'actual shot durations invalid: {sid}')
  if not isinstance(task['timing_source'],str) or not task['timing_source'].strip():
   raise ValueError(f'actual timing source is required: {sid}')
  rows.append(row)
 if not groups or any(len(ids)!=len(set(ids)) or set(ids)!=wanted for ids in groups.values()):
  raise ValueError('each method/tier/seed must cover exactly the declared stories')
 result={**config,'stories':rows,'unavailable_stories':unavailable,'rereads':[],
         'generation_coverage':{'expected_count':len(manifest['requests']),
          'available_count':len(rows),'unavailable_count':len(unavailable),
          'groups':[{'method':method,'input_tier':tier,'seed':seed,'story_ids':ids}
                    for (method,tier,seed),ids in groups.items()]}}
 validate_config(result)
 return result

def precall_input_tokens(story, frozen):
 # static 默认1 FPS；以1024 token/秒预留（含音频和时间戳），文本按UTF8字节上界，另留schema与传输封装。
 # 这是显式保守预留，不是服务端计费上限。
 return max(frozen['precall_prompt_tokens'],len(make_prompt(story).encode())+len(json.dumps(RESPONSE).encode())+4096+1024*(math.ceil(story['duration_seconds'])+2))


def quote(config, frozen, prices):
 frozen={**frozen,'max_tokens':config['judge']['max_tokens']}
 reads=requested_reads(config)
 details=[]
 for _,s,part,questions in reads:
  # 估算而非分词器实测：中英混排文本UTF8字节/3，视频沿旧实测66token/s；每条答案90token、medium思考3000。
  it=math.ceil(len(make_prompt(s,questions).encode())/3)+math.ceil(s['duration_seconds']*66)+600
  ot=3000+len(questions)*90
  if ot>frozen['max_tokens']:raise ValueError('预计单条输出超过冻结max_tokens')
  usd=(Decimal(it)*Decimal(prices['input_usd_per_mtok'])+Decimal(ot)*Decimal(prices['output_usd_per_mtok']))/1000000
  details.append({'method':s['method'],'story_id':s['story_id'],'part':part,'estimated_input_tokens':it,'estimated_output_tokens':ot,'estimated_usd':str(usd),'precall_input_tokens':precall_input_tokens(s,frozen)})
 it=sum(d['estimated_input_tokens'] for d in details);ot=sum(d['estimated_output_tokens'] for d in details)
 cost=Decimal(it)*Decimal(prices['input_usd_per_mtok'])/1000000+Decimal(ot)*Decimal(prices['output_usd_per_mtok'])/1000000
 return {'request_count':len(reads),'first_reads':sum(r[0]=='first-read' for r in reads),'rereads':sum(r[0]=='reread' for r in reads),'input_tokens':it,'output_tokens_including_reasoning':ot,'estimated_usd':str(cost),'prices':prices,'method':'文本UTF8字节/3＋video66token/s；输出每题90＋medium思考3000；估计值，非实测','precall_reserve_usd':str(Decimal(frozen['precall_prompt_tokens'])*Decimal(prices['input_usd_per_mtok'])/1000000+Decimal(frozen['max_tokens'])*Decimal(prices['output_usd_per_mtok'])/1000000),'details':details}

def request_paths(output):
 return sorted(path for phase in ('first-read','reread') for path in (output/'metrics'/phase).rglob('*.json'))

# Google AI Studio经OpenRouter实测拒收>20,000,000字节的完整请求体（10-26 HTTP413）；留出上游改写余量。
MAX_REQUEST_BODY_BYTES=19_900_000

def request_body_bytes(frozen, options, prompt, video_bytes):
 """用实际SDK把空视频请求序列化到离线MockTransport，再加base64长度；不联网、不计费。"""
 import httpx
 from openai import OpenAI
 seen=[]
 def handle(request):
  seen.append(len(request.content))
  return httpx.Response(400,json={'error':{'message':'offline body measurement'}})
 counter=SdkCounter(hard_cap_usd='1',prices={frozen['model']:{'input_usd_per_mtok':'1','output_usd_per_mtok':'1'}},precall_prompt_tokens=1,precall_completion_tokens=1)
 client=SdkClient(api_key='offline-body-measurement',base_url='https://openrouter.ai/api/v1',cost_counter=counter,timeout_seconds=1,proxy=None)
 client._sdk_client=OpenAI(base_url='https://openrouter.ai/api/v1',api_key='offline-body-measurement',max_retries=0,http_client=httpx.Client(transport=httpx.MockTransport(handle),trust_env=False))
 try:
  client.complete_video(model=frozen['model'],video_data_url='data:video/mp4;base64,',prompt=prompt,temperature=None,max_tokens=frozen['max_tokens'],request_options=options,response_format=RESPONSE)
 except Exception:
  pass
 finally:
  client.close()
 assert len(seen)==1
 return seen[0]+4*math.ceil(video_bytes/3)

def request_options(config, frozen, prices):
 options=copy.deepcopy(frozen['request_options']);options['reasoning']=config['judge']['reasoning']
 options['provider']['max_price']={'prompt':float(prices['input_usd_per_mtok']),'completion':float(prices['output_usd_per_mtok'])}
 return options

def initialize_trial(output, config, frozen, prices) -> None:
    """协调器在派发前冻结一次行级身份；续跑只校验，不改写同一份输入。"""
    frozen = {**frozen, 'max_tokens': config['judge']['max_tokens']}
    identity = {'checklist': config, 'frozen_judge': frozen, 'prices': prices, 'response_format': RESPONSE}
    path = output / 'inputs.json'
    if path.exists():
        assert json.loads(path.read_text()) == identity
    else:
        write(path, identity)


def write_trial_budget(output, cap, prior_unknown_reserves) -> dict:
    """由协调器或独立 trial 写行级快照；每笔回执仍是费用与未知预留的来源。"""
    ledger = {path: json.loads(path.read_text()) for path in request_paths(output)}
    known = sum((Decimal(r['cost_usd']) for r in ledger.values() if r.get('cost_usd') is not None), Decimal('0'))
    unknown = {path for path, record in ledger.items() if record.get('cost_usd') is None}
    reserves = {Path(path): Decimal(str(amount)) for path, amount in prior_unknown_reserves.items()}
    budget = {'hard_cap_usd': str(cap), 'conservative_spent_usd': str(known),
              'reserved_prior_unknown_usd': str(sum((reserves[path] for path in unknown if path in reserves), Decimal('0'))),
              'unreserved_unknown_count': len(unknown - set(reserves)),
              'record_count': len(ledger), 'unknown_cost_count': len(unknown)}
    write(output / 'budget.json', budget)
    return budget


def _start_request_record(path, record) -> None:
    """完整回执原子独占发布；并发账本不会读到半个 JSON，同一路径只准派发一次。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=f'.{path.name}.', suffix='.tmp') as temporary:
        temporary.write(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        temporary.flush()
        os.link(temporary.name, path)


def execute_read(args, config, frozen, prices, read, *, spent_usd: Decimal) -> dict:
    """只执行指定的一次读取，独占其回执；输入身份和共享预算由协调器持有。"""
    from mspc.data.openrouter_client import OpenRouterClient, OpenRouterCostCounter, OpenRouterError

    frozen = {**frozen, 'max_tokens': config['judge']['max_tokens']}
    phase, story, part, questions = read
    path = request_path(args.output, phase, story, part)
    if path.exists():
        raise FileExistsError(path)
    reserve_tokens = precall_input_tokens(story, frozen)
    counter = OpenRouterCostCounter(hard_cap_usd=args.budget_usd, prices={frozen['model']: prices},
        already_spent_usd=spent_usd, precall_prompt_tokens=reserve_tokens,
        precall_completion_tokens=frozen['max_tokens'])
    counter.reserve_precall(frozen['model'])
    probe = json.loads(subprocess.check_output([args.ffprobe, '-v', 'error', '-show_entries', 'format=duration',
                                              '-of', 'json', story['video_path']]))
    duration = float(probe['format']['duration'])
    assert abs(duration - story['duration_seconds']) < .1
    options = request_options(config, frozen, prices)
    prompt = make_prompt(story, questions)
    video_path = Path(story['video_path'])
    video_bytes = video_path.stat().st_size
    body = request_body_bytes(frozen, options, prompt, video_bytes)
    if body > MAX_REQUEST_BODY_BYTES:
        raise ValueError(f'完整请求体{body}字节超过{MAX_REQUEST_BODY_BYTES}，不发送: {story["story_id"]}')
    key = os.environ['OPENROUTER_API_KEY']
    record = {'method': story['method'], 'input_tier': story['input_tier'], 'seed': story['seed'],
              'story_id': story['story_id'], 'part': part, 'requirement_ids': [q['requirement_id'] for q in questions],
              'phase': phase, 'status': 'request_started', 'started_at': now(), 'cost_usd': None,
              'duration_seconds': duration, 'request_body_bytes': body, 'video_bytes': video_bytes}
    _start_request_record(path, record)

    def recorder(request, raw, error, elapsed):
        raw = raw.model_dump(mode='json') if hasattr(raw, 'model_dump') else raw
        record.update(raw_response=raw, error=error, seconds=elapsed,
                      status='request_failed' if error else 'response_received')
        write(path, record)

    client = None
    try:
        client = OpenRouterClient(api_key=key, base_url='https://openrouter.ai/api/v1',
                                  cost_counter=counter, timeout_seconds=300, proxy=args.proxy, recorder=recorder)
        video = 'data:video/mp4;base64,' + base64.b64encode(video_path.read_bytes()).decode()
        response = client.complete_video(model=frozen['model'], video_data_url=video, prompt=prompt, temperature=None,
            max_tokens=frozen['max_tokens'], request_options=options, response_format=RESPONSE)
        usage = response.usage
        actual = record['raw_response']['usage'].get('cost')
        # 回执按较高单价保留完整 Decimal 精度；provider 原始费用另存，不因显示舍入少计。
        charge = (Decimal(usage['prompt_tokens']) * Decimal(prices['input_usd_per_mtok'])
                  + Decimal(usage['completion_tokens']) * Decimal(prices['output_usd_per_mtok'])) / 1000000
        if actual is not None:
            charge = max(charge, Decimal(str(actual)))
        record.update(cost_usd=str(charge), reported_cost_usd=actual, usage=usage,
                      served=response.served, text=response.text, request_id=response.request_id)
        video_tokens = (response.served['prompt_tokens_details'] or {}).get('video_tokens')
        if type(video_tokens) is not int or video_tokens < frozen['min_video_tokens_per_second'] * duration:
            record['status'] = 'video_not_ingested'
            raise ValueError('视频未摄入')
        if response.served['provider'] != 'Google AI Studio':
            record['status'] = 'route_changed'
            raise ValueError('实际供应商改变，停止')
        record['result'] = validate_response(json.loads(response.text), story, questions)
        record['status'] = 'rated'
    except Exception as error:
        record['error'] = str(error)
        if isinstance(error, OpenRouterError):
            record.update(exception_chain=error.exception_chain, http_request_id=error.http_request_id,
                          status_code=error.status_code)
        if record['status'] == 'request_started':
            record['status'] = 'request_failed'
        if record['cost_usd'] is None:
            raise
        if record['status'] == 'response_received':
            record['status'] = 'format_failed'
        # 费用已知的格式错误留为测量失败；不在本函数重发或选择其它故事。
    finally:
        try:
            record['finished_at'] = now()
            write(path, record)
        finally:
            if client is not None:
                client.close()
    if record['status'] in ('video_not_ingested', 'route_changed'):
        raise ValueError(record['error'])
    return record


def trial(args, config, frozen, *, prior_unknown_reserves):
    """独立入口：校验已有费用，按 limit 顺序派发；复用同一条单请求执行路径。"""
    frozen = {**frozen, 'max_tokens': config['judge']['max_tokens']}
    cap = Decimal(args.budget_usd)
    quoted = json.loads(args.quote.read_text())
    prices = quoted['prices']
    initialize_trial(args.output, config, frozen, prices)
    known = Decimal('0')
    done, unknown = set(), set()
    for path in request_paths(args.output):
        record = json.loads(path.read_text())
        if record.get('cost_usd') is None:
            unknown.add(path)
        else:
            known += Decimal(record['cost_usd'])
        done.add((record['method'], record['story_id'], record.get('part')))
    reserves = {Path(path): Decimal(str(amount)) for path, amount in prior_unknown_reserves.items()}
    if set(reserves) != unknown:
        raise ValueError(f'费用未决记录与已计账预留不一致，停止: missing={sorted(unknown-set(reserves))} extra={sorted(set(reserves)-unknown)}')
    if any(not amount.is_finite() or amount <= 0 for amount in reserves.values()):
        raise ValueError('未知费用预留必须为正的有限金额')
    reserved = sum(reserves.values(), Decimal('0'))
    spent = known + reserved
    if spent + sum(Decimal(d['estimated_usd']) for d in quoted['details']
                   if (d['method'], d['story_id'], d['part']) not in done) > cap:
        raise ValueError('未读部分报价超过剩余硬顶，禁止启动')
    dispatched = 0
    for read in requested_reads(config):
        phase, story, part, _ = read
        if request_path(args.output, phase, story, part).exists():
            continue
        if dispatched >= args.limit:
            break
        reserve_tokens = precall_input_tokens(story, frozen)
        reserve = (Decimal(reserve_tokens) * Decimal(prices['input_usd_per_mtok'])
                   + Decimal(frozen['max_tokens']) * Decimal(prices['output_usd_per_mtok'])) / 1000000
        if spent + reserve > cap:
            write(args.output / 'budget-stop.json', {'spent_usd': str(spent), 'next_reserve_usd': str(reserve),
                'next': dict(zip(('method', 'input_tier', 'seed', 'story_id'), story_key(story))), 'part': part})
            break
        try:
            record = execute_read(args, config, frozen, prices, read, spent_usd=spent)
        finally:
            budget = write_trial_budget(args.output, cap, reserves)
        dispatched += 1
        print(json.dumps({key: record[key] for key in ('story_id', 'part', 'phase', 'status', 'cost_usd')},
                         ensure_ascii=False), flush=True)
        spent = Decimal(budget['conservative_spent_usd']) + reserved
        if spent > cap:
            raise ValueError('实际账单超过预算，停止')
    write_trial_budget(args.output, cap, reserves)

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','validate','quote','trial']);p.add_argument('--config',type=Path,required=True);p.add_argument('--manifest',type=Path);p.add_argument('--frozen',type=Path,default=Path('configs/eval/story_ratings.json'));p.add_argument('--prices',type=Path);p.add_argument('--quote',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--proxy');p.add_argument('--ffprobe');p.add_argument('--limit',type=int);p.add_argument('--budget-usd');a=p.parse_args()
 c=json.loads(a.config.read_text())
 if a.command=='prepare':
  if a.manifest is None:p.error('prepare requires --manifest with actual shot timing')
  write(a.output,prepare(c,json.loads(a.manifest.read_text())));return
 f=json.loads(a.frozen.read_text());audit=validate_config(c)
 if a.command=='validate':write(a.output,audit);print(audit)
 elif a.command=='quote':write(a.output,quote(c,f,json.loads(a.prices.read_text())));print(a.output.read_text())
 else:
  if a.limit is None or a.limit<1:p.error('trial requires positive --limit')
  if a.budget_usd is None:p.error('trial requires --budget-usd from the run authorization')
  trial(a,c,f,prior_unknown_reserves={})
if __name__=='__main__':main()
