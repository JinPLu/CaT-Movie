import json

import pytest

from mspc.comparison_methods import st_bench
from mspc.evaluation.st_bench import load_story

LIMITS = {'wan_single': {'max_shots': 1, 'max_frames': 228, 'max_tokens': 512},
          'multishotmaster': {'max_shots': 5, 'max_frames': 308, 'max_tokens': 512},
          'holocine': {'max_shots': None, 'max_frames': 241, 'max_tokens': 512}}
words = lambda text: len(text.split())


@pytest.fixture
def story(tmp_path):
    # Same structure as StoryMem story/cat.json at 052c68d2 (scene 1 cut flags [true, false, false]); prompts shortened.
    data = {'story_name': 'A Cat', 'story_overview': 'A curious house cat spends a quiet day.',
            'scenes': [{'scene_num': 1, 'cut': [True, False, False],
                        'video_prompts': ['A fluffy orange-and-white cat stretches on the windowsill.',
                                          'The cat jumps down as a hand fills the bowl.',
                                          'Close-up of the cat eating quietly.'],
                        'first_frame_prompt': ['Cat mid-stretch on the windowsill.', 'Cat stepping toward its bowl.',
                                               'Close on the cat at the bowl.']},
                       {'scene_num': 2, 'cut': [True, False],
                        'video_prompts': ['The cat chases a patch of sunlight.', 'It bats at a dangling string toy.'],
                        'first_frame_prompt': ['Cat poised over a bright floor patch.', 'Cat on the couch, paws raised.']}]}
    path = tmp_path / 'cat.json'
    path.write_text(json.dumps(data))
    return load_story(path)


@pytest.fixture
def registry():
    mentions = [('A fluffy orange-and-white cat', 'Cat'), ('The cat', 'Cat'), ('the cat', 'the cat'),
                ('The cat', 'Cat'), (None, 'Cat')]
    return {'characters': [{'tag': 'Character 1', 'name': None, 'description': 'a fluffy orange-and-white cat'}],
            'shots': [{'shot': i, 'present': [{'tag': 'Character 1', 'video_mention': v, 'first_frame_mention': f}]}
                      for i, (v, f) in enumerate(mentions, 1)]}


def test_registry_accepts_verbatim_mentions_and_rejects_paraphrase(story, registry):
    assert st_bench.validate_registry(story, registry) is registry
    registry['shots'][0]['present'][0]['video_mention'] = 'A fluffy orange cat'
    with pytest.raises(ValueError, match='whole-word'):
        st_bench.validate_registry(story, registry)


def test_registry_tells_characters_apart_by_name_and_description(story, registry):
    registry['characters'].append({'tag': 'Character 2', 'name': None, 'description': 'A fluffy orange-and-white cat'})
    registry['shots'][1]['present'].append({'tag': 'Character 2', 'video_mention': None, 'first_frame_mention': None})
    with pytest.raises(ValueError, match='distinct'):
        st_bench.validate_registry(story, registry)
    registry['characters'][1]['name'] = 'house cat'
    assert st_bench.validate_registry(story, registry) is registry


def test_registry_name_must_be_a_script_span_and_reaches_every_tagged_method(story, registry):
    registry['characters'][0]['name'] = 'Garfield'
    with pytest.raises(ValueError, match='name'):
        st_bench.validate_registry(story, registry)
    registry['characters'][0]['name'] = 'cat'
    assert st_bench.validate_registry(story, registry) is registry
    assert st_bench.character_text(registry['characters'][0]) == 'a fluffy orange-and-white cat'
    named = {'tag': 'Character 1', 'name': 'Elon Musk', 'description': 'a man in black t-shirt'}
    assert st_bench.character_text(named) == 'Elon Musk, a man in black t-shirt'


def test_registry_requires_every_shot_and_present_characters(story, registry):
    registry['shots'].pop()
    with pytest.raises(ValueError, match='every shot'):
        st_bench.validate_registry(story, registry)


def test_tag_text_replaces_only_the_first_mention_of_each_character():
    text = 'The man hands the woman a cup; the man smiles.'
    tagged = st_bench.tag_text(text, {'Character 1': 'The man', 'Character 2': 'the woman', 'Character 3': None},
                               lambda k: f'[character{k}]')
    assert tagged == '[character1] hands [character2] a cup; the man smiles.'
    # Real registry for st-bench-explorer wrote 'the explorer' for a sentence-initial 'The explorer'.
    assert st_bench.tag_text('The explorer stops.', {'Character 1': 'the explorer'}, lambda k: f'[C{k}]') == '[C1] stops.'
    with pytest.raises(ValueError, match='whole-word'):
        st_bench.tag_text(text, {'Character 1': 'The man hands', 'Character 2': 'hands the'}, str)
    # Real registry for st-bench-santa: the mention ends with a dash that runs into the next word.
    assert st_bench.tag_text('Pole, Santa Claus\u2014an elderly man\u2014sits.', {'Character 1': 'Santa Claus\u2014an elderly man\u2014'},
                             lambda k: f'[character{k}]') == 'Pole, [character1] sits.'
    # Real registry for st-bench-princess: 'the prince' must not land inside 'The princess'.
    text = 'The princess places it in the prince\u2019s hand.'
    assert st_bench.tag_text(text, {'Character 1': 'the prince', 'Character 2': 'The princess'},
                             lambda k: f'[C{k}]') == '[C2] places it in [C1]\u2019s hand.'


def test_wan_reads_raw_prompts_one_shot_per_call(story):
    segments = st_bench.native_segments(story, None, {'family': 'wan_single', 'limits': LIMITS['wan_single']}, 81, words)
    assert [s['prompts'] for s in segments] == [[shot['prompt']] for shot in story['shots']]
    assert all(s['frame_counts'] == [81] for s in segments)


def test_multishotmaster_packs_three_five_second_shots_per_window(story, registry):
    segments = st_bench.native_segments(story, registry, {'family': 'multishotmaster', 'limits': LIMITS['multishotmaster']},
                                        81, words)
    assert [(s['start_shot'], s['end_shot']) for s in segments] == [(0, 3), (3, 5)]
    assert segments[0]['prompts'][0] == ('Story: A curious house cat spends a quiet day. '
                                         'Subject 1: a fluffy orange-and-white cat. Now: Subject 1 stretches on the windowsill.')
    assert segments[0]['shot_groups'] == [[0, 81], [81, 162], [162, 243]]


def test_holocine_rounds_three_five_second_shots_into_one_241_frame_window(story, registry):
    segments = st_bench.native_segments(story, registry, {'family': 'holocine', 'limits': LIMITS['holocine']}, 81, words)
    # Official nearest-4n+1 rounding turns 243 requested frames into 241, so a fourth shot no longer fits.
    assert [(s['start_shot'], s['end_shot']) for s in segments] == [(0, 3), (3, 5)]
    first = segments[0]
    assert first['global_caption'].startswith('A curious house cat spends a quiet day. [character1]: a fluffy')
    assert first['shot_captions'][:2] == ['[character1] stretches on the windowsill.',
                                          '[character1] jumps down as a hand fills the bowl.']
    assert (first['num_frames'], first['shot_cut_frames']) == (241, [81, 161])
    assert st_bench.segment_shot_frames(first) == [81, 80, 80]
    # The last shot names the cat only by pronoun, so its caption stays verbatim.
    assert segments[1]['shot_captions'][-1] == 'It bats at a dangling string toy.'


def test_keyframe_source_tags_first_frame_prompts_and_keeps_raw_video_prompts(story, registry):
    source = st_bench.keyframe_source(story, registry, 16, 81)
    assert source['characters'] == {'[C1]': 'a fluffy orange-and-white cat'}
    assert source['shots'][0]['tagged'] == '[C1] mid-stretch on the windowsill.'
    assert source['shots'][0]['expanded'] == 'a fluffy orange-and-white cat mid-stretch on the windowsill.'
    assert source['shots'][2]['tagged'] == 'Close on [C1] at the bowl.'
    assert [s['video_prompt'] for s in source['shots']] == [s['prompt'] for s in story['shots']]
    assert source['identity_sources'] == {'[C1]': {'name': 'Character 1', 'outfit': ''}}


def test_keyframe_source_feeds_storydiffusion_identity_bank(story, registry):
    from mspc.comparison_methods.storydiffusion import prepare_story

    prepared = prepare_story(st_bench.keyframe_source(story, registry, 16, 81))
    assert prepared['reference_indices'] == {'[C1]': [0]}
    assert prepared['identity_references'][0]['source'] == 'given_story_solo_shot'


def test_storymem_official_frames_and_files_follow_cut_flags(story, tmp_path):
    assert st_bench.storymem_shot_frames(story, 81) == [81, 80, 80, 81, 80]
    assert [p.name for p in st_bench.storymem_shot_files(story, tmp_path)] == [
        '01_01.mp4', '01_02.mp4', '01_03.mp4', '02_01.mp4', '02_02.mp4']


def test_registry_schema_is_strict_json_schema():
    from mspc.data.openrouter_client import require_response_format

    assert require_response_format(st_bench.registry_schema())['json_schema']['strict'] is True
