import json
from pathlib import Path

from mspc.comparison_methods.ic_lora import prepare_story as prepare_ic
from mspc.comparison_methods.keyframe_video import prepare_keyframe_story, wan_output_size
from mspc.comparison_methods.storydiffusion import prepare_story as prepare_sd


FIXTURES = Path(__file__).parents[2] / 'fixtures'


def test_real_wan_i2v_area_rounding_preserves_the_requested_frame():
    # 首次真实53帧采样返回(3,53,464,832)，原配置面积恰落在浮点向下取整边界。
    image_size = (832, 480)
    vae_stride = (4, 8, 8)
    patch_size = (1, 2, 2)
    assert wan_output_size(399360, image_size, vae_stride, patch_size) == (832, 464)
    assert wan_output_size(399361, image_size, vae_stride, patch_size) == image_size


def test_common_opening_producer_preserves_the_delivered_pilot_fields():
    story = json.loads((FIXTURES / 'keyframe_video/cinedance-014847-b0000-shot01.json').read_text())
    original = json.loads((FIXTURES / 'ic_lora/cinedance-014847-b0000-shot01.json').read_text())
    prepared = prepare_keyframe_story(story)
    # identity_sources 是之后补的身份图回退来源，试点交付里没有这一项。
    assert all(value == original[key] for key, value in prepared.items() if key != 'identity_sources')
    sd = prepare_sd(prepared)
    assert sd['reference_indices'] == original['reference_indices']
    assert all(sd[key] == value for key, value in prepared.items())


def test_real_thirteen_character_story_initializes_missing_solo_banks_from_text():
    story = json.loads((FIXTURES / 'keyframe_video/film-001591-b0001-shot01.json').read_text())
    source = prepare_keyframe_story(story)
    assert source['split'] == 'test'
    assert len(source['characters']) == 13
    sd = prepare_sd(source)
    assert sd['shots'] == source['shots']
    assert len(sd['identity_references']) == 13
    assert {r['character'] for r in sd['identity_references']} == set(source['characters'])
    for row in sd['identity_references']:
        if row['story_shot_index'] is not None:
            assert sd['prompt_characters'][row['story_shot_index']] == [row['character']]
    auxiliary = [r for r in sd['identity_references'] if r['story_shot_index'] is None]
    assert auxiliary
    for row in auxiliary:
        assert row['source'] == 'given_character_description'
        assert source['characters'][row['character']] in row['prompt']
    names = dict(zip(source['characters'], ['Alice', 'Ben', 'Carol', 'David', 'Emma',
                  'Frank', 'Grace', 'Henry', 'Iris', 'Jack', 'Kate', 'Leo', 'Mia']))
    prepared = prepare_ic(source, {'width': 832, 'panel_height': 480, 'character_names': names})
    assert prepared['shots'] == source['shots']
    assert prepared['characters'] == source['characters']
    for window in prepared['windows']:
        assert all(f'<{names[tag]}> is {appearance}.' in window['prompt']
                   for tag, appearance in source['characters'].items())


def test_real_three_character_story_uses_each_original_solo_reference():
    story = json.loads((FIXTURES / 'storydiffusion/cinedance-008935-b0001-shot01.json').read_text())
    prepared = prepare_sd(prepare_keyframe_story(story))
    assert len(prepared['characters']) == 3
    assert set(prepared['reference_indices']) == set(prepared['characters'])
    for tag, indices in prepared['reference_indices'].items():
        assert len(indices) == 1
        assert prepared['shots'][indices[0]]['characters'] == [tag]
