"""实际失败原件约束回归；虚构示例只用于线协议，不含评测金标。"""
import copy
import json
from pathlib import Path

import pytest

from mspc.data.event_log import (
    EventLog, compile_event_log, dependency_segment_spans, frame_evidence, parse_event_log,
    validate_event_log,
)
from mspc.data.event_planner_contract import (
    planner_output_contract, planner_format_example, planner_contract_messages,
)
from mspc.data.planner_baseline import baseline_messages, evaluate_baseline_response

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'tests/fixtures/planner_call'
FAILURES = {r['name']: r for r in json.loads((FIXTURES/'contract_real_failures.json').read_text())['responses']}


def transmitted():
    case = json.loads((FIXTURES/'real_case.json').read_text())
    cfg = json.loads((ROOT/'configs/planner/zero_shot_qwen35_9b_thinking.json').read_text())
    return baseline_messages(case, 'script_only', cfg, planner_output_contract())


def test_example_in_transmitted_request_passes_every_offline_consumer():
    messages = transmitted()
    raw = messages[2]['content']
    example = json.loads(raw)
    assert example == planner_format_example()
    log = parse_event_log(example)
    validate_event_log(log)
    # 示例请求本身就是 planner_request 的输出（逐镜档、锁定镜号帧数、无历史）。
    request = json.loads(messages[1]['content'].split('规划请求：', 1)[1])
    assert request['shots'] == [{'shot_id': s['shot_id'], 'frame_count': s['frame_count']} for s in example['shots']]
    assert request['total_frames'] == 99 and request['history_frames'] == []
    case = {'case_id': 'fictional', 'script': request['script'], 'script_detail': 'shot', 'target_start_shot_id': 's01',
            'target_start_seconds': 0.0, 'history': {'true': []}, 'gold_event_log': example}
    result = evaluate_baseline_response(raw, case, token_counter=lambda text: len(text.split()))
    assert result['failed_layer'] is None and result['passed'][-1] == 'compile'
    # 与虚构故事一致的帧证据，只检验编译接口；不是实际像素验收。
    spans = dependency_segment_spans(log)
    assert spans == ((0, 3),)
    compiled = compile_event_log(log, span=spans[0], references=(), token_counter=lambda text: len(text.split()))
    assert compiled.reference_shot_ids == ()
    assert 'by the end of the shot' in compiled.target_rows[1]
    assert 'no longer holds' in compiled.target_rows[1]
    reference = frame_evidence(log, 's01', 'last', ['Subject 1', 'Subject 2', 'Subject 3'])
    later = compile_event_log(log, span=(2, 3), references=[reference], token_counter=lambda text: len(text.split()))
    assert 'Reference 1, the last frame of shot 1.' in later.reference_rows[0]
    assert 'Reference 1 is the last frame of shot 1; take Subject 1 appearance' in later.target_rows[0]


def test_example_covers_families_axes_endpoints_and_has_no_reference_keys():
    a = planner_format_example()
    facts = a['initial_state'] + [f for s in a['shots'] for f in s['screen_facts']]
    assert {f['family'] for f in facts} == {'state', 'relation', 'screen'}
    assert {f['predicate_or_axis'] for f in facts if f['family']=='state'} == {'costume','body','object_condition','environment'}
    assert {f['predicate_or_axis'] for f in facts if f['family']=='screen'} == {'horizontal','depth','orientation','gaze','bbox'}
    assert {f.get('evidence', 'unknown') for f in facts} == {'observed','carried','unknown'}
    assert all('evidence' not in f for f in facts if f['object_or_value'] == 'unknown')
    assert {(e['before'] is None, e['after'] is None) for e in a['events']} == {(False,False),(True,False),(False,True)}
    assert not {'segments', 'style'} & set(a)
    assert all('cut_type' not in s for s in a['shots'])
    assert all((s['frame_count'] - 1) % 4 == 0 and s['frame_count'] <= 228 for s in a['shots'])


def test_r2_actual_short_span_exposed_in_request_schema_and_rules():
    a=json.loads(FAILURES['r2_27b_plain']['raw_content'])
    assert a['events'][0]['after']['shot_span']==[0]
    span=planner_output_contract()['schema']['$defs']['AtomicFact']['properties']['shot_span']
    assert span['minItems']==span['maxItems']==2
    prompt=transmitted()[0]['content']
    assert '恰好两个整数' in prompt and '0<=start<=end<镜头数' in prompt
    assert '单镜必须[i,i]' in prompt


def test_r2_actual_unregistered_and_invisible_endpoints_explained_separately():
    plain=json.loads(FAILURES['r2_27b_plain']['raw_content'])
    assert plain['events'][1]['after']['object_or_value']=='red chair'
    assert 'red chair' not in plain['shots'][1]['visible_entities']
    history=json.loads(FAILURES['r2_9b_history']['raw_content'])
    assert history['events'][0]['before']['subject']=='Subject 1'
    assert 'Subject 1' not in history['shots'][1]['visible_entities']
    prompt=transmitted()[0]['content']
    assert '关系各端点、所有fact.subject都须已登记' in prompt
    assert '只有on_screen=true时' in prompt and 'before/after的全部端点ID' in prompt
    assert 'on_screen=false/unknown不要求所有端点可见' in prompt


def test_r2_actual_duplicate_same_key_and_unknown_event_explained():
    plain=json.loads(FAILURES['r2_27b_plain']['raw_content'])
    left,right=plain['events'][3:5]
    key=lambda e:(e['shot_id'],e['after']['subject'],e['after']['predicate_or_axis'])
    assert key(left)==key(right) and left['event_id']!=right['event_id']
    history=json.loads(FAILURES['r2_27b_history']['raw_content'])
    assert history['events'][1]['after']['evidence']=='unknown'
    prompt=transmitted()[0]['content']
    assert '同一shot_id、同一fact key最多一个事件' in prompt
    assert '非null端点必须是已知state/relation，不能screen或unknown' in prompt
    assert 'before必须匹配本镜所有事件之前的已知基线' in prompt


def test_r2_actual_anchor_and_cross_key_replacement_are_explained():
    a=json.loads(FAILURES['r2_9b_history']['raw_content'])
    assert a['locations'][0]['anchors'][0]['id']=='kitchen_sink'
    event=a['events'][0]
    assert event['before']['subject']!=event['after']['subject']
    prompt=transmitted()[0]['content']
    assert '锚点完整ID为Location N/Anchor M' in prompt
    assert '两个非null端点必须同fact key' in prompt
    assert '旧键删除和新键增加' in prompt


def test_transmitted_enums_are_owned_by_schema_and_registry():
    from mspc.data.event_log import registry
    contract=planner_output_contract()
    assert contract['schema']==EventLog.model_json_schema()
    marker='消费端原始登记表（轴/谓词/值全集）：\n'
    assert json.loads(contract['rules'].split(marker)[1])==registry().model_dump()
    prompt=transmitted()[0]['content']
    for values in [['state','relation','screen'],['observed','carried','unknown']]:
        assert all(value in prompt for value in values)
    assert '512个umT5 tokens' in prompt
    assert '各镜frame_count之和必须等于输入给定的total_frames' in prompt and '4n+1' in prompt and '228帧' in prompt
    assert not any(word in prompt for word in ('borrow_attributes', 'anchor_covered', 'reference_keys', 'SetupKey'))


def test_stale_frozen_rules_or_null_example_cannot_be_silently_reused():
    for field,value in [('rules','旧规则'),('example',None)]:
        old=planner_output_contract();old[field]=value
        before=copy.deepcopy(old)
        with pytest.raises(ValueError,match='explicitly freeze a new benchmark'):
            planner_contract_messages(old)
        assert old==before



def test_prompt_and_validator_agree_that_an_unknown_value_carries_no_evidence_field():
    """10-10 链路验证：提示曾要求 unknown 值的 evidence 写 unknown，校验器却禁止该字段，零样本 15/50、SFT 8/18 条答复
    因此在 schema 层被拒。失败事实取自这两次的真实答复；提示改成与校验器、金标序列化一致。"""
    from mspc.data.state_facts import AtomicFact
    rules = planner_output_contract()['rules']
    assert '值为unknown时不写evidence字段' in rules and 'evidence必须unknown' not in rules
    # 「每个事实严格包含…」那一句同样与校验器一致：值已知时带 evidence，值为 unknown 时不带。
    assert '值已知时再带evidence，值为unknown时不带evidence' in rules and 'object_or_value/scope/shot_span/evidence' not in rules
    real = json.loads((ROOT/'tests/fixtures/planner_generate/unknown_with_evidence_facts.json').read_text())
    assert {k: v['fact']['evidence'] for k, v in real.items()} == {
        'zeroshot_evidence_unknown': 'unknown', 'zeroshot_evidence_observed': 'observed', 'sft_evidence_observed': 'observed'}
    for entry in real.values():
        fact = entry['fact']
        with pytest.raises(ValueError, match='cannot carry an evidence field'):
            AtomicFact.model_validate(fact)  # 校验器没变：这样写仍被拒
        bare = {k: v for k, v in fact.items() if k != 'evidence'}
        assert 'evidence' not in AtomicFact.model_validate(bare).model_dump(by_alias=True, mode='json')  # 提示要求的写法
    # 合同自带的格式示例也是这样写的：值为 unknown 的事实没有 evidence 键。
    unknown = [f for s in planner_format_example()['shots'] for f in s['screen_facts'] if f['object_or_value'] == 'unknown']
    unknown += [f for f in planner_format_example()['initial_state'] if f['object_or_value'] == 'unknown']
    assert unknown and all('evidence' not in f for f in unknown)



def test_planner_answers_drop_evidence_of_unknown_values_before_validation():
    """规划器答复里值为 unknown 的事实带 evidence 不拒收。归一只在规划器答复的入口
    （normalize_planner_answer）；标注流水线的 AtomicFact 校验不动，仍然拒收。三条真实失败事实取自两次链路验证。"""
    from copy import deepcopy
    from mspc.data.event_planner_contract import normalize_planner_answer
    from mspc.data.state_facts import AtomicFact
    real = json.loads((ROOT/'tests/fixtures/planner_generate/unknown_with_evidence_facts.json').read_text())
    for entry in real.values():
        fact = entry['fact']
        bare = {k: v for k, v in fact.items() if k != 'evidence'}
        original = deepcopy(fact)
        assert normalize_planner_answer(fact) == bare and fact == original  # 丢字段，不改输入
        assert normalize_planner_answer({'shots': [{'screen_facts': [fact]}], 'events': []})['shots'][0]['screen_facts'] == [bare]
        with pytest.raises(ValueError, match='cannot carry an evidence field'):
            AtomicFact.model_validate(fact)  # 标注流水线的校验不变
    # 已知值的 evidence 不受影响；非事实内容原样保留。
    known = {'subject': 'Subject 1', 'family': 'state', 'predicate_or_axis': 'costume', 'object_or_value': 'a red coat',
             'scope': 'scene', 'shot_span': [0, 0], 'evidence': 'observed'}
    payload = {'initial_state': [known], 'note': 'unknown', 'shots': []}
    assert normalize_planner_answer(payload) == payload
    assert normalize_planner_answer([1, 'unknown', None]) == [1, 'unknown', None]
