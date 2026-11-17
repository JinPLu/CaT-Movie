"""P1a真实窗口与真实旧200窗转换；错误边界从真实行派生。"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mspc.data.event_log import (
    EventLogError, FrameEvidence, compile_event_log, event_key, fact_key, frame_evidence, parse_event_log,
    snapshots, state_text, validate_event_log,
)

# 固件里的 schema 字符串已按 event_log_v3 改写（2026-10-09），非逐字原文；其余内容未动。
ROOT = Path(__file__).parents[2] / 'fixtures/event_log_v3_reference'


def fixture(name='p1a-01'):
    return json.loads((ROOT / f'{name}.json').read_text())['event_log']


def compile_raw(raw, *, frames=(), count=lambda t: len(t.split())):
    """逐镜各编一段；frames 里早于该镜的帧都作参考（编译器只描述给定帧）。"""
    log = parse_event_log(raw)
    index = {s.shot_id: i for i, s in enumerate(log.shots)}
    return [compile_event_log(log, span=(i, i + 1), references=[f for f in frames if index[f.shot_id] < i][-4:],
                              token_counter=count) for i in range(len(log.shots))]


@pytest.mark.parametrize('name', ['p1a-01','p1a-03','p1a-06','p1a-10','p1a-12','p1a-25','p1a-35'])
def test_real_windows_compile_families_and_shot_local_space_sentence(name):
    raw=fixture(name);original=deepcopy(raw)
    result=compile_raw(raw)
    assert raw==original
    assert len(result)==len(raw['shots'])
    for s,c in zip(raw['shots'],result):
        assert ' is unknown' not in c.target_rows[0]
        assert 'bbox' not in c.target_rows[0]
        assert len(c.token_counts)==len(c.prompts)
        assert c.reference_rows==()
    # 空间句在每镜目标行最前面；末尾的 Screen 句已被取代。
    assert not any('Screen:' in c.target_rows[0] for c in result)
    spaced = [c.target_rows[0] for c in result if 'of the frame' in c.target_rows[0]]
    assert spaced and all(row.split(' ', 1)[0] in ('On', 'In') and 'of the frame' in row.split('.', 1)[0] for row in spaced)


def test_real_last_shot_event_has_opening_and_end_values_and_no_next_shot_pointer():
    raw=fixture();log=parse_event_log(raw);e=log.events[0]
    validate_event_log(log)
    start,end=snapshots(log),snapshots(log,end=True)
    k=event_key(e)
    assert start['s04'][k].object_or_value==e.before.object_or_value
    assert end['s04'][k].object_or_value==e.after.object_or_value
    text=compile_raw(raw)[-1].target_rows[0]
    assert 'at first' in text and 'by the end of the shot' in text
    assert 'effective_from' not in e.model_dump()


def test_real_uncertain_release_is_not_a_deletion_or_empty_hand_claim():
    raw=fixture('p1a-03')
    assert len(raw['events'])==1  # only independently witnessed fall; release is unknown
    rows=compile_raw(raw)
    assert 'holds nothing' not in ' '.join(c.target_rows[0] for c in rows)
    assert 'no longer holds' not in rows[-1].target_rows[0]
    assert 'Subject 3 is' not in rows[-1].target_rows[0]  # ball hidden at last keyframe


def test_hidden_relation_endpoint_is_omitted_while_fact_is_carried():
    raw=fixture();f=next(f for f in raw['initial_state'] if f['family']=='relation')
    f['shot_span']=[0,3]
    # remove repeated declarations so the carrier has one initial relation
    raw['initial_state']=[x for x in raw['initial_state'] if x is f or not(x['family']=='relation' and x['predicate_or_axis']=='holds')]
    log=parse_event_log(raw)
    carried=snapshots(log)['s04'][('Subject 1','holds','Subject 2')]
    assert carried.evidence=='carried'
    assert 'holds Subject 2' not in compile_raw(raw)[-1].target_rows[0]


def test_unknown_observation_cannot_erase_known_fact():
    raw=fixture();base=next(f for f in raw['initial_state'] if f['family']=='state' and f['predicate_or_axis']=='costume')
    base['shot_span']=[0,3]
    raw['initial_state']=[base,{**{k:v for k,v in base.items() if k!='evidence'},'object_or_value':'unknown','shot_span':[1,1]}]
    raw['events']=[]
    log=parse_event_log(raw)
    assert snapshots(log)['s02'][('Subject 1','costume')].object_or_value==base['object_or_value']


@pytest.mark.parametrize('mutation,rule', [
    ('undeclared',1),('self',3),('unregistered',3),('same_key_events',4),('wrong_before',2),('cycle',3),('screen_carried',3),('static_state',5)])
def test_real_rows_reject_invalid_rules_without_repair(mutation,rule):
    raw=fixture();f=next(f for f in raw['initial_state'] if f['family']=='relation')
    if mutation=='undeclared':f['object_or_value']='Subject 999'
    elif mutation=='self':f['object_or_value']=f['subject']
    elif mutation=='unregistered':f['predicate_or_axis']='owns'
    elif mutation=='same_key_events':raw['events'].append(dict(raw['events'][0],event_id='e2'))
    elif mutation=='wrong_before':raw['events'][0]['before']['object_or_value']='closed'
    elif mutation=='cycle':
        f.update(predicate_or_axis='in');other=deepcopy(f);other.update(subject=f['object_or_value'],object_or_value=f['subject']);raw['initial_state'].append(other)
    elif mutation=='screen_carried':raw['shots'][0]['screen_facts'][0]['evidence']='carried'
    elif mutation=='static_state':raw['subjects'][0]['appearance']='standing man'
    original=deepcopy(raw)
    with pytest.raises(EventLogError) as exc:compile_raw(raw)
    assert rule in {i.rule for i in exc.value.issues}
    assert raw==original


def test_location_change_expires_old_anchor_but_holds_follows_person():
    raw=fixture();raw['events']=[]
    raw['locations'].append({'location_id':'Location 2','appearance':'another room','anchors':[]})
    raw['shots'][1]['location_id']='Location 2'
    f=next(f for f in raw['initial_state'] if f['family']=='relation');f['shot_span']=[0,3]
    raw['initial_state']=[f,dict(f,subject='Subject 3',predicate_or_axis='on',object_or_value='Location 1/Anchor 1')]
    log=parse_event_log(raw);state=snapshots(log)
    assert ('Subject 1','holds','Subject 2') in state['s02']
    assert ('Subject 3','on','Location 1/Anchor 1') not in state['s02']
    assert ('Subject 3','on','Location 1/Anchor 1') not in state['s03']  # return does not resurrect


def test_token_budget_exact_boundary():
    assert compile_raw(fixture(),count=lambda _:512)
    with pytest.raises(EventLogError) as exc:compile_raw(fixture(),count=lambda _:513)
    assert {i.rule for i in exc.value.issues}=={7}


def test_more_than_four_references_or_later_frames_are_rejected():
    raw=fixture();log=parse_event_log(raw)
    frames=[FrameEvidence(s.shot_id,p,True,True,{},frozenset()) for s in log.shots[:3] for p in ('first','last')]
    with pytest.raises(EventLogError) as exc:
        compile_event_log(log,span=(3,4),references=frames[:5],token_counter=lambda t:len(t.split()))
    assert 6 in {i.rule for i in exc.value.issues}
    assert compile_event_log(log,span=(3,4),references=frames[:4],token_counter=lambda t:len(t.split())).reference_shot_ids==('s01','s01','s02','s02')
    with pytest.raises(EventLogError):
        compile_event_log(log,span=(2,4),references=[frames[4]],token_counter=lambda t:len(t.split()))
    with pytest.raises(EventLogError):
        compile_event_log(log,span=(3,4),references=[frames[0],frames[0]],token_counter=lambda t:len(t.split()))


def test_frame_evidence_positions_take_start_end_and_unchanged_facts():
    raw=fixture();log=parse_event_log(raw);e=log.events[0];k=event_key(e)
    shot=next(s for s in log.shots if s.shot_id==e.shot_id)
    visible=sorted({x for x in shot.visible_entities if x.startswith('Subject ')})
    first=frame_evidence(log,e.shot_id,'first',visible)
    last=frame_evidence(log,e.shot_id,'last',visible)
    middle=frame_evidence(log,e.shot_id,'middle',visible)
    assert first.result_states[k].object_or_value==e.before.object_or_value
    assert last.result_states[k].object_or_value==e.after.object_or_value
    assert k not in middle.result_states
    assert set(middle.result_states)<=set(first.result_states)&set(last.result_states)
    assert all(f.family!='screen' and f.object_or_value!='unknown' for f in (*first.result_states.values(),*last.result_states.values()))
    # 端点不可见的事实不收。
    hidden=frame_evidence(log,e.shot_id,'last',[])
    assert all(f.predicate_or_axis=='environment' for f in hidden.result_states.values())
    with pytest.raises(ValueError):
        frame_evidence(log,e.shot_id,'center',visible)


def test_reference_row_does_not_copy_planned_state_into_pixel_evidence():
    raw=fixture();raw['events']=[]
    frames=[FrameEvidence(s['shot_id'],'last',True,True,{},frozenset(s['visible_entities'])) for s in raw['shots']]
    compiled=compile_raw(raw,frames=frames)
    assert all('State:' not in row for c in compiled for row in c.reference_rows)
    assert any('current facts from text:' in row for c in compiled for row in c.target_rows)
    assert any('Reference 1 is the last frame of shot 1; take ' in row for c in compiled for row in c.target_rows)


def test_real_collection_retransfer_refuses_false_whole_set_removal():
    # Same collection is already on the plate when the next portion is moved; add(null→on) is false.
    with pytest.raises(EventLogError) as exc:
        compile_raw(fixture('p1a-07'))
    assert 2 in {i.rule for i in exc.value.issues}


@pytest.mark.parametrize("span", [[], [0], [0, 1, 2]])
@pytest.mark.parametrize("owner", ["initial", "event"])
def test_malformed_span_reports_validation_error_without_indexing(owner, span):
    # 真实9B首次调用暴露短span导致IndexError；原始回答在旧writer中丢失。
    # 用真实P1a源派生错误形状，明确不是保存下来的模型输出。
    raw = fixture()
    fact = raw["initial_state"][0] if owner == "initial" else raw["events"][0]["before"]
    fact["shot_span"] = span
    with pytest.raises(EventLogError, match="shot_span"):
        validate_event_log(parse_event_log(raw))


# ---------------------------------------------------------------------------
# 每镜目标行开头的空间句（依据跨镜布局判别 S-first 臂读数）：真实 dev 30 剧本与真实 p1a 窗
# ---------------------------------------------------------------------------

DEV30 = Path(__file__).parents[2] / 'fixtures/long_story_event_log/dev30_b_state_in_appearance.json'
CANDIDATE = Path(__file__).parents[3] / 'configs/benchmark/script_state_bench_long_holdout_v2_candidate_r3.json'


def dev30_log(story, layouts, *, appearance=None):
    """真实 dev 30 剧本转日志，再把给定的逐镜布局（{镜序: {主体: {轴: 值}}}）填进 screen 事实；
    appearance 是 {主体: 外观} 的显式改动（造出真实剧本里没有的同描述情形）。"""
    from mspc.benchmark.long_story_event_log import event_log_from_script, r3_frame_counts
    from mspc.evaluation.cross_shot_layout import with_screen_facts
    pair = next(p for p in json.loads(DEV30.read_text())['pairs'] if p['story_id'] == story)
    raw = event_log_from_script(pair['context_b'], pair['change'], r3_frame_counts(json.loads(CANDIDATE.read_text()), story))
    full = [layouts.get(i, {}) for i in range(len(raw['shots']))]
    filled = with_screen_facts(raw, full, axes=('horizontal', 'depth', 'orientation'))
    for subject in filled['subjects']:
        subject['appearance'] = (appearance or {}).get(subject['subject_id'], subject['appearance'])
    return parse_event_log(filled)


def test_real_space_sentence_is_first_orders_left_to_right_and_adds_depth_and_facing():
    from mspc.data.event_log import space_text
    log = dev30_log('lsh-020', {0: {'Subject 9': {'horizontal': 'right', 'depth': 'background'},
                                    'Subject 1': {'horizontal': 'left', 'depth': 'foreground', 'orientation': 'toward camera'}}})
    shot = log.shots[0]
    assert set(shot.present_subjects) >= {'Subject 1', 'Subject 9'}
    text = space_text(log, shot)
    assert text == ('On the left side of the frame, in the foreground, is an older woman with grey swept-back hair, '
                    'wearing a black dress, facing the camera; on the right side of the frame, in the background, '
                    'is a woman with short grey-brown hair and a white apron.')
    row = compile_event_log(log, span=(0, 1), references=(), token_counter=lambda t: len(t.split())).target_rows[0]
    assert row.startswith(text + ' ') and 'Screen:' not in row
    assert space_text(log, log.shots[1]) == ''  # 没有屏幕事实的镜没有空间句


def test_real_people_are_pointed_at_by_the_whole_first_sentence_of_their_description():
    from mspc.data.event_log import space_text
    # lsh-028 s02 的两个男子短属性词相同（都是穿浅灰 T 恤的白人男子），整句外观描述里一个有 “CREW” 字样，分得开。
    log = dev30_log('lsh-028', {1: {'Subject 2': {'horizontal': 'left'}, 'Subject 4': {'horizontal': 'right'}}})
    assert space_text(log, log.shots[1]) == (
        "On the left side of the frame is a white man with short brown hair, wearing a light grey t-shirt with the word 'CREW' on it; "
        "on the right side of the frame is a white man with short brown hair, wearing a light grey t-shirt.")
    # lsh-010 的 Subject 1 是一队士兵，整句描述照写。
    group = dev30_log('lsh-010', {0: {'Subject 1': {'horizontal': 'center'}}})
    assert space_text(group, group.shots[0]) == (
        'In the center of the frame is a group of soldiers wearing green camouflage uniforms and helmets.')


def test_real_people_with_the_same_description_or_none_fall_back_to_subject_id_and_head_noun():
    from mspc.data.event_log import space_text
    layout = {1: {'Subject 2': {'horizontal': 'left'}, 'Subject 4': {'horizontal': 'right'}}}
    base = dev30_log('lsh-028', layout)
    same = base.subjects[1].appearance
    log = dev30_log('lsh-028', layout, appearance={'Subject 4': same})  # 显式改动：两人描述逐字相同，读者分不出
    assert space_text(log, log.shots[1]) == ('On the left side of the frame is Subject 2, the man; '
                                             'on the right side of the frame is Subject 4, the man.')
    unknown = dev30_log('lsh-028', layout, appearance={'Subject 4': 'unknown'})
    assert space_text(unknown, unknown.shots[1]) == (
        "On the left side of the frame is a white man with short brown hair, wearing a light grey t-shirt with the word 'CREW' on it; "
        "on the right side of the frame is Subject 4.")


def test_real_space_sentence_writes_people_and_animals_only():
    from mspc.data.event_log import space_text
    log = parse_event_log(fixture())  # p1a-01：Subject 1 是人，Subject 2、3 是物体，三者都有真实屏幕事实
    shot = log.shots[0]
    assert {f.subject for f in shot.screen_facts if f.predicate_or_axis == 'horizontal'} == {'Subject 1', 'Subject 2', 'Subject 3'}
    assert space_text(log, shot) == 'In the center of the frame, in the foreground, is a man with short black hair.'
    pig = dev30_log('lsh-023', {0: {'Subject 1': {'horizontal': 'center'}}})
    assert space_text(pig, pig.shots[0]) == 'In the center of the frame is a young, white pig with pink ears and snout.'
