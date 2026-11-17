"""近邻真实失败句：局部手部绑定、多句外貌、状态终点与同貌双人。"""
from copy import deepcopy
import json
import re

import pytest

from mspc.comparison_methods.prompts import global_and_shots, segment_inputs
from tests.support.paths import REPO_ROOT


def board(story):
    rows = json.loads((REPO_ROOT / 'tests/fixtures/neighbor_prompt_examples.json').read_text())['stories']
    return next(row['storyboard'] for row in rows if row['story_id'] == story)


def test_hand_owner_and_multisentence_appearance_stay_with_dark_haired_man():
    source = board('lsh-002-b')
    original = deepcopy(source)
    prompt = segment_inputs(source, 'storymem', 0, 1)['prompts'][0]
    assert 'The hand of the man with short, dark hair enters the frame from the left' in prompt
    assert 'the head of the man with short, graying hair' in prompt
    assert 'For the man with short, dark hair, his face is glistening with sweat.' in prompt
    assert 'with a low ceiling and concrete walls' in prompt
    assert 'possibly a tunnel or underpass' in prompt
    assert 'is holding the handgun' in prompt
    assert 'sweat, is' not in prompt
    assert source == original


@pytest.mark.parametrize('family,first,second', [
    ('multishotmaster', 'Subject 1', 'Subject 4'),
    ('holocine', '[character1]', '[character2]'),
])
def test_official_tags_keep_whole_story_identity_across_segments(family, first, second):
    source = board('lsh-008-b')
    global_caption, shots = global_and_shots(source, 5, 6, family)
    assert first + ':' in global_caption and second + ':' in global_caption
    assert f'{second} is in the driver' in shots[0]
    assert first + ' is wearing dark sunglasses' in shots[0]
    assert not re.search(r'Location \d+|(?:Subject \d+|\[character\d+\])\.', shots[0])
    assert "driver's seat and window frame" in global_caption


def test_state_change_keeps_initial_action_and_final_result_without_duplicate_opening():
    source = board('lsh-002-b')
    segment = segment_inputs(source, 'multishotmaster', 0, 4)
    text = segment['shot_captions'][1]
    assert text.count('Subject 2 is holding the handgun') == 1
    assert 'opens his gun hand and drops the handgun onto the floor' in text
    assert 'Subject 2 is empty-handed, with the handgun released onto the floor' in text
    assert segment['frame_counts'] == [37, 37, 29, 53]
    assert segment['shot_groups'] == [[0, 37], [37, 74], [74, 103], [103, 156]]


def test_removed_clothing_is_a_state_not_an_added_action():
    text = segment_inputs(board('lsh-004-b'), 'multishotmaster', 6, 7)['shot_captions'][0]
    assert 'the dark vest is removed, with the white shirt exposed' in text
    assert 'is dark vest removed' not in text


def test_natural_references_distinguish_drivers_by_clothing_and_keep_anonymous_hand():
    source = board('lsh-008-b')
    text = segment_inputs(source, 'storymem', 5, 6)['prompts'][0]
    assert 'white t-shirt is in the driver' in text
    assert 'The back of the head of the male wearing a light-colored button-down shirt' in text
    assert "shirt's head" not in text
    anonymous = segment_inputs(board('lsh-003-b'), 'storymem', 3, 4)['prompts'][0]
    assert 'A hand holding the black semi-automatic handgun' in anonymous
    assert 'small, red cut on her right cheek' in anonymous


def test_identical_appearance_does_not_collapse_two_co_present_people():
    text = segment_inputs(board('lsh-020-b'), 'storymem', 2, 3)['prompts'][0]
    assert 'are seated in the center' in text
    assert 'Another man with short dark hair wearing a dark suit and tie is seated to their left' in text


def test_nested_reference_in_real_appearance_is_resolved():
    text = segment_inputs(board('lsh-028-b'), 'storymem', 2, 3)['prompts'][0]
    assert not re.search(r'\b(?:Character|Object|Location) \d+', text)


@pytest.mark.parametrize('story', ['lsh-002-b', 'lsh-003-b', 'lsh-008-b'])
def test_real_pilot_keeps_native_frame_quantization_and_deterministic_text(story):
    source = board(story)
    for family in ('multishotmaster', 'holocine', 'storymem'):
        width = {'multishotmaster': 5, 'holocine': 4, 'storymem': 1}[family]
        for start in range(0, len(source['shots']), width):
            end = min(start + width, len(source['shots']))
            row = segment_inputs(source, family, start, end)
            assert row == segment_inputs(source, family, start, end)
            assert row['frame_counts'] == [s['frame_count'] for s in source['shots'][start:end]]
            if family == 'holocine':
                assert row['prompt'].count('[shot cut]') == end - start - 1
                assert len(row['shot_cut_frames']) == end - start - 1
                assert row['num_frames'] % 4 == 1
            elif family == 'storymem':
                assert not re.search(r'\b(?:Character|Object|Location) \d+', row['prompts'][0])
