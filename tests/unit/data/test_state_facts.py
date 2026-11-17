from pathlib import Path
import pytest
from pydantic import ValidationError
from mspc.data.state_facts import AtomicFact, FactRecord, FactWindow, PredicateRegistry, carry_facts, validate_window

ROOT = Path(__file__).resolve().parents[3]

@pytest.fixture
def registry():
    return PredicateRegistry.load(ROOT / 'configs/data/state_predicates_v2.yaml')

@pytest.fixture
def window():
    return FactWindow.model_validate({
        'schema_version': 'state_facts_v2', 'window_id': 'math-contract',
        'entities': [{'id': f'Subject {i}', 'kind': 'object', 'static_appearance': 'rigid item'} for i in range(1, 4)],
        'locations': [{'id': f'Location {i}', 'description': 'room', 'anchors': [
            {'id': f'Location {i}/Anchor 1', 'description': 'wall', 'front_definition': 'room-facing side'},
            {'id': f'Location {i}/Anchor 2', 'description': 'round table', 'front_definition': None}]} for i in range(1,3)],
        'shots': [{'index': 0, 'location': 'Location 1', 'present': ['Subject 1','Subject 2','Subject 3']},
                  {'index': 1, 'location': 'Location 2', 'present': ['Subject 1','Subject 2','Subject 3']}],
        'sampled_frames': [{'id': f's{i}-after', 'shot': i, 'proxy_frame_index': 9,
                      'width': 640, 'height': 360, 'present': ['Subject 1','Subject 2','Subject 3']} for i in range(2)],
        'keyframes': [{'id': f's{i}-after', 'shot': i, 'phase': 'after', 'proxy_frame_index': 9,
                      'width': 640, 'height': 360, 'present': ['Subject 1','Subject 2','Subject 3']} for i in range(2)],
        'records': []})

def record(subject='Subject 1', predicate='in', value='Subject 2', *, family='relation', span=None, scope='scene', frame=None, evidence='observed', registration='registered', coverage='包含与位置'):
    return FactRecord(fact=AtomicFact(subject=subject, family=family, predicate_or_axis=predicate,
        object_or_value=value, scope=scope, shot_span=[0,0] if span is None else span, **({} if value=='unknown' else {'evidence':evidence})),
        registration=registration, frame_id=frame, evidence_frames=['s0-after'], coverage=coverage, source=None)

def check(w,r,registry):
    validate_window(w.model_copy(update={'records':r}),registry)


def test_open_state_and_relation_phrase(window,registry):
    facts=[record(predicate='costume',value='grey jacket with rolled sleeves',family='state',coverage='衣着外观'),
           record(predicate='leans beside',registration='unregistered')]
    check(window,facts,registry)
    assert facts[1].fact.predicate_or_axis == 'leans beside'
    assert registry.relation_registration('held_by') == 'unregistered'
    assert len(facts[0].fact.atomic_tuple()) == 7


def test_registration_no_synonym_repair(window,registry):
    with pytest.raises(ValueError,match='registration mismatch'):
        check(window,[record(predicate='inside',registration='registered')],registry)


@pytest.mark.parametrize('bad',[record(value='Subject 1'),record(value='Subject 99'),record(subject='Subject 99')])
def test_unregistered_and_self_relation(window,registry,bad):
    with pytest.raises(ValueError):check(window,[bad],registry)


def test_anchor_front_and_location_change(window,registry):
    check(window,[record(predicate='in front of',value='Location 1/Anchor 1')],registry)
    with pytest.raises(ValueError,match='no defined front'):
        check(window,[record(predicate='behind',value='Location 1/Anchor 2')],registry)
    with pytest.raises(ValueError,match='expired'):
        check(window,[record(predicate='at',value='Location 1/Anchor 1',span=[0,1])],registry)
    check(window,[record(predicate='holds',value='Subject 2',span=[0,1],coverage='占有与接触')],registry)


def test_containment_cycle_by_time(window,registry):
    with pytest.raises(ValueError,match='containment cycle'):
        check(window,[record(),record(subject='Subject 2',value='Subject 3'),record(subject='Subject 3',value='Subject 1')],registry)
    # Mutually inverse edges at different sampled phases do not coexist.
    data=window.model_dump();new=dict(data['keyframes'][0],id='s0-before',phase='before');data['keyframes'].append(new);data['sampled_frames'].append({k:v for k,v in new.items() if k!='phase'});w=FactWindow.model_validate(data)
    a=record(frame='s0-before').model_copy(update={'evidence_frames':['s0-before']})
    check(w,[a,record(subject='Subject 2',value='Subject 1',frame='s0-after')],registry)


def test_between_registered_endpoints(window,registry):
    check(window,[record(predicate='between',value=['Subject 2','Subject 3'])],registry)
    with pytest.raises(ValueError,match='two endpoints'):
        check(window,[record(predicate='between',value='Subject 2')],registry)


@pytest.mark.parametrize('bbox',[[0,0,1001,400],[400,0,100,400],[0,0,True,200], [0,0,300]])
def test_invalid_box(window,registry,bbox):
    with pytest.raises((ValueError,ValidationError)):
        check(window,[record(predicate='bbox',value=bbox,family='screen',scope='screen@0',frame='s0-after',registration='not_applicable',coverage=None)],registry)


def test_keyframe_box_and_screen_scope(window,registry):
    box=record(predicate='bbox',value=[100,200,700,800],family='screen',scope='screen@0',frame='s0-after',registration='not_applicable',coverage=None)
    check(window,[box],registry)
    with pytest.raises(ValueError,match='selected keyframe'):
        check(window,[box.model_copy(update={'frame_id':None})],registry)
    with pytest.raises(ValueError,match='never carried'):
        check(window,[box.model_copy(update={'fact':box.fact.model_copy(update={'evidence':'carried'})})],registry)


def test_invisible_does_not_rewrite_and_portable_relations_follow_person(registry):
    facts=[record(predicate='holds',coverage='占有与接触'),
           record(predicate='wears',value='Subject 3',coverage='占有与接触'),
           record(predicate='at',value='Location 1/Anchor 1'),
           record(predicate='touches',coverage='占有与接触'),
           record(predicate='costume',value='wet jacket',family='state',coverage='衣着外观'),
           record(predicate='costume',value='unknown',family='state',coverage='衣着外观',evidence='unknown')]
    same=carry_facts(facts,from_location='Location 1',to_location='Location 1',next_shot=1,registry=registry)
    moved=carry_facts(facts,from_location='Location 1',to_location='Location 2',next_shot=1,registry=registry)
    assert len(same)==4
    assert [r.fact.predicate_or_axis for r in moved]==['holds','wears','costume']
    assert moved[-1].fact.object_or_value=='wet jacket'
    assert all(r.fact.evidence=='carried' for r in moved)


def test_contact_does_not_automatically_persist(window,registry):
    with pytest.raises(ValueError,match='cannot automatically persist'):
        check(window,[record(predicate='touches',evidence='carried',coverage='占有与接触')],registry)


@pytest.mark.parametrize('count', [0, 1, 2, 3])
def test_visible_anchor_count(window, count):
    data = window.model_dump()
    anchors = data['locations'][0]['anchors']
    anchors.append({'id': 'Location 1/Anchor 3', 'description': 'visible door', 'front_definition': None})
    data['locations'][0]['anchors'] = anchors[:count]
    assert len(FactWindow.model_validate(data).locations[0].anchors) == count


def test_four_anchors_rejected(window):
    data = window.model_dump()
    data['locations'][0]['anchors'] *= 2
    with pytest.raises(ValidationError, match='0–3'):
        FactWindow.model_validate(data)


def test_state_environment_type_and_multiple_values(window,registry):
    check(window,[record(predicate='body',value='wet',family='state',coverage='身体状态'),
                  record(predicate='body',value='lying down',family='state',coverage='身体状态')],registry)
    with pytest.raises(ValueError,match='environment belongs'):
        check(window,[record(predicate='environment',value='dark',family='state',coverage='环境状态')],registry)


def test_contact_sample_evidence_is_not_box_keyframe(window,registry):
    data=window.model_dump();data['sampled_frames'].append(dict(data['sampled_frames'][0],id='s0-40',proxy_frame_index=4))
    w=FactWindow.model_validate(data)
    scene=record(frame='s0-40').model_copy(update={'evidence_frames':['s0-40']})
    check(w,[scene],registry)
    box=record(predicate='bbox',value=[0,0,100,100],family='screen',scope='screen@0',frame='s0-40',registration='not_applicable',coverage=None).model_copy(update={'evidence_frames':['s0-40']})
    with pytest.raises(ValueError,match='selected keyframe'):check(w,[box],registry)
    check(w,[record(predicate='attached to',coverage='占有与接触')],registry)
