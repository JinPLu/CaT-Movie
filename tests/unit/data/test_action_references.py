"""真实训练／开发镜：指称范围、集合人数、物化幂等与开镜状态语义。"""
import copy
import json
from pathlib import Path

import pytest

from mspc.data.shot_text import action_reference_candidates, resolve_action_references, shot_text_with_entity_spans, _posture_state_conflict

CASES = {row['key']: row for row in json.loads((Path(__file__).parents[2] /
    'fixtures/block_annotation_v18/action-reference-failures.json').read_text())['cases']}


@pytest.mark.parametrize('case', CASES.values(), ids=CASES.keys())
def test_real_actions_only_replace_nonoverlapping_noun_phrases_and_are_idempotent(case):
    annotation, index = case['annotation'], case['shot_index']
    before = copy.deepcopy(annotation)
    result = resolve_action_references(annotation, index)
    original, cursor, parts = annotation['shots'][index]['action'], 0, []
    for edit in result['replacements']:
        assert edit['start'] >= cursor
        assert original[edit['start']:edit['end']] == edit['phrase']
        parts.extend((original[cursor:edit['start']], edit['replacement']))
        cursor = edit['end']
    parts.append(original[cursor:])
    assert ''.join(parts) == result['action']
    materialized = copy.deepcopy(annotation)
    materialized['shots'][index]['action'] = result['action']
    assert resolve_action_references(materialized, index)['action'] == result['action']
    assert shot_text_with_entity_spans(materialized, index)['text'] == shot_text_with_entity_spans(annotation, index)['text']
    assert annotation == before


def test_real_other_actor_and_existing_octopus_apposition_stay_unmodified():
    for key in ['training-200-71', 'training-200-120']:
        case = CASES[key]
        assert resolve_action_references(case['annotation'], case['shot_index'])['action'] == case['annotation']['shots'][case['shot_index']]['action']


def test_real_octopus_action_expands_only_explicit_plural_entity_labels():
    from mspc.data.shot_text import _Row
    case = CASES['training-200-120']
    annotation = case['annotation']
    original = annotation['shots'][case['shot_index']]['action']
    compiled = shot_text_with_entity_spans(annotation, case['shot_index'])
    assert 'Characters ' not in compiled['text']
    assert annotation['shots'][case['shot_index']]['action'] == original
    for phrase in ('Characters 1 and 3', 'Characters 1, 2, and 3', 'Characters 1, 2 and 3'):
        value = _Row(annotation).prose(phrase + ' look at 24 objects, 3 times.', 'shots[0].action')
        numbers = ['1', '3'] if phrase == 'Characters 1 and 3' else ['1', '2', '3']
        assert [value.text[start:end] for _, start, end, _ in value.spans] == ['Character ' + number for number in numbers]
        assert value.text.endswith(' look at 24 objects, 3 times.')


def test_real_explicit_plural_counts_select_the_exact_present_identity_set():
    for key, expected in [('dev30-6', ['Character 3', 'Character 4']),
                          ('dev30-221', ['Character 1', 'Character 2', 'Character 3', 'Character 4'])]:
        case = CASES[key]
        result = resolve_action_references(case['annotation'], case['shot_index'])
        assert result['replacements'][0]['names'] == expected
        assert all(name in result['action'] for name in expected)
        assert not result['action'].startswith('The Character')


def test_real_resolution_rejects_identity_outside_original_candidate_set():
    case = CASES['dev30-269']
    candidate = next(row for row in action_reference_candidates(case['annotation'], case['shot_index']) if row['candidates'])
    with pytest.raises(ValueError, match='present candidate'):
        resolve_action_references(case['annotation'], case['shot_index'], resolutions=[{
            'start': candidate['start'], 'end': candidate['end'], 'names': ['Character 999']}])


@pytest.mark.parametrize('key', ['before-state-0', 'before-state-2'])
def test_real_before_state_owns_opening_and_driving_after_remains_at_ending(key):
    case = CASES[key]
    compiled = shot_text_with_entity_spans(case['annotation'], case['shot_index'])
    conflict = next(row for row in compiled['state_diagnostics'] if row['kind'] == 'posture_state_conflict')
    assert conflict['kept_field'] == 'state' and conflict['removed_field'] == 'posture'
    opening, ending = compiled['text'].split('By the end of the shot, ', 1)
    assert conflict['state'] in opening and 'driving' in ending


STATE_CASES = json.loads((Path(__file__).parents[2] /
    'fixtures/block_annotation_v18/state-predicate-failures.json').read_text())['cases']


@pytest.mark.parametrize('case', STATE_CASES, ids=lambda row: f"{row['member_key']}:{row['shot_index']}")
def test_real_local_motion_other_subject_and_temporal_complements_are_not_contradictions(case):
    assert _posture_state_conflict(case['posture'], case['state']) is case['expected_conflict']


@pytest.mark.parametrize('key', ['training-200-30', 'training-200-78'])
def test_real_indefinite_people_are_not_forced_into_an_exact_individual_set(key):
    case = CASES[key]
    candidate = next(row for row in action_reference_candidates(case['annotation'], case['shot_index'])
                     if row['plural'] and row['cardinality'] is None and len(row['candidates']) > 1)
    result = resolve_action_references(case['annotation'], case['shot_index'], resolutions=[{
        'start': candidate['start'], 'end': candidate['end'],
        'names': [name for name in candidate['candidates'] if name not in candidate['group_candidates']]}])
    assert any(row['start'] == candidate['start'] and row.get('reason') == 'identity_not_determined'
               for row in result['unresolved'])
    assert not any(row['start'] == candidate['start'] for row in result['replacements'])


def test_actual_weighted_stage_keeps_complete_unique_clothes_and_the_holding_relation_once():
    case = CASES['actual-stage-43460']
    compiled = shot_text_with_entity_spans(case['annotation'], 0, references=case['references'])
    for name, required in [('Character 3', 'purple short-sleeved top'),
                           ('Character 6', 'white top with red polka dots'),
                           ('Character 7', 'green and white striped top with a dark cardigan')]:
        registry = compiled['text'].split(name + ': ', 1)[1].split('. ', 1)[0]
        assert registry.count('wearing ') == 1 and required in registry
    assert 'stands, holding Object 1' in compiled['text']
    assert 'Object 1 is held by Character 1' not in compiled['text']
    assert any(row['kind'] == 'duplicate_holding_relation' for row in compiled['state_diagnostics'])


def test_actual_previously_unresolved_another_is_not_strengthened_by_materialized_ids():
    for key in ['actual-val-another', 'actual-test-another']:
        case = CASES[key]
        result = resolve_action_references(case['annotation'], case['shot_index'])
        assert result['action'] == case['annotation']['shots'][case['shot_index']]['action']
        assert any(row['phrase'].lower().startswith('another') and row['matched_names'] is None
                   for row in result['unresolved'])


def test_actual_stage_registers_every_present_object_and_keeps_original_fields_action_and_sources():
    case = CASES['actual-stage-43460']
    annotation = case['annotation']
    before = copy.deepcopy(annotation)
    compiled = shot_text_with_entity_spans(annotation, 0, references=case['references'])
    registry = compiled['text_sections']['registry']
    assert all(f'Object {i}:' in registry for i in range(1, 9))
    names = {span['source']: span['name'] for span in compiled['entity_spans']}
    assert all(names[f'objects[{i - 1}].name'] == f'Object {i}' for i in range(1, 9))
    assert 'Object 3 and Object 4 are hanging on the wall. Object 5 is closed.' in compiled['text']
    assert compiled['action_references']['action'] == annotation['shots'][0]['action']
    assert annotation == before


def test_layered_clothing_never_loses_the_direction_of_its_original_relation():
    from mspc.data.shot_text import _appearance_without_stable_clothing
    appearance = 'a man wearing a red coat over a blue shirt'
    assert _appearance_without_stable_clothing(appearance, 'a red coat under a blue shirt') == appearance
    assert _appearance_without_stable_clothing(appearance, 'a red coat over a blue shirt') == 'a man'


def test_real_eating_action_keeps_its_implicit_food_object_in_the_registry():
    case = CASES['dev-implicit-eating-object']
    compiled = shot_text_with_entity_spans(case['annotation'], case['shot_index'])
    assert 'Object 7: plates of food.' in compiled['text_sections']['registry']
