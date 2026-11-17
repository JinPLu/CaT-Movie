import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from mspc.comparison_methods import keyframe_story

PLAN = Path(__file__).resolve().parents[3] / 'configs/benchmark/benchmark_storyboards_v4.json'
STORY = 'film-000122-b0000-shot01'


def template(tmp_path, image_method, video_method):
    config = {'seed': 42, 'image_method': image_method, 'video_method': video_method,
              'given_plan_source': str(PLAN), 'output_root': str(tmp_path), 'python_path': ['/deps']}
    if image_method == 'ic_lora':
        config['ic_lora'] = {'steps': 20}
    (tmp_path / 'inputs').mkdir(parents=True)
    (tmp_path / 'inputs/keyframe-config.json').write_text(json.dumps(config))
    return config


def test_ic_lora_names_every_planned_character_by_its_number(tmp_path):
    config = template(tmp_path, 'ic_lora', 'ic_lora_wan_a14b')
    story = next(s for s in json.loads(PLAN.read_text())['stories'] if s['story_id'] == STORY)
    characters = keyframe_story.prepare_keyframe_story(story)['characters']
    result = keyframe_story.story_config(config, STORY, characters)
    assert result['ic_lora']['character_names'] == {'[C1]': 'Character 1', '[C2]': 'Character 2'}
    assert result['story_id'] == STORY and 'character_names' not in config['ic_lora']


def test_render_runs_four_stages_over_every_shot(tmp_path, monkeypatch):
    template(tmp_path, 'storydiffusion', 'storydiffusion_wan_a14b')
    calls = []

    monkeypatch.setenv('PYTHONPATH', '/code/src')

    def fake_run(argv, check, env):
        assert env['PYTHONPATH'] == '/deps:/code/src'
        calls.append(argv[3:])
        if argv[3] == 'prepare':
            prepared = tmp_path / 'inputs/prepared' / f'{STORY}.json'
            prepared.parent.mkdir(parents=True, exist_ok=True)
            prepared.write_text(json.dumps({'story_id': STORY, 'shots': [{}] * 8}))

    monkeypatch.setattr(keyframe_story.subprocess, 'run', fake_run)
    keyframe_story.render(tmp_path, 'storydiffusion_wan_a14b', [STORY])
    config = str(tmp_path / 'inputs/configs' / f'{STORY}.json')
    assert calls == [['prepare', '--config', config], ['images', '--config', config],
                     ['videos', '--shot-indices', *map(str, range(8)), '--config', config],
                     ['assemble', '--config', config]]
    assert json.loads(Path(config).read_text())['story_id'] == STORY


def completed_ic_lora_images(tmp_path):
    fixture = Path(__file__).resolve().parents[2] / 'fixtures/keyframe_video/ic_lora_completed_images_interrupted_story.json'
    source = json.loads(fixture.read_text())
    config = template(tmp_path, 'ic_lora', 'ic_lora_wan_a14b')
    prepared = source['prepared']
    directory = tmp_path / 'inference/raw/images/ic_lora' / prepared['story_id']
    directory.mkdir(parents=True)
    for i in range(len(prepared['shots'])):
        Image.new('RGB', tuple(prepared['panel_size'])).save(directory / f'{i+1:04d}.png')
    (directory / 'generation.json').write_text(json.dumps(source['image_generation']))
    return source, config, directory


def test_interrupted_story_reuses_all_completed_images_and_continues_video(tmp_path, monkeypatch):
    source, config, directory = completed_ic_lora_images(tmp_path)
    story_id = source['story_id']
    calls = []
    monkeypatch.setenv('PYTHONPATH', '/code/src')

    def fake_run(argv, check, env):
        calls.append(argv[3])
        if argv[3] == 'prepare':
            prepared = tmp_path / 'inputs/prepared' / f'{story_id}.json'
            prepared.parent.mkdir(parents=True)
            prepared.write_text(json.dumps(source['prepared']))

    monkeypatch.setattr(keyframe_story.subprocess, 'run', fake_run)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    keyframe_story.render(tmp_path, 'ic_lora_wan_a14b', [story_id])
    assert calls == ['prepare', 'videos', 'assemble']
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.parametrize('damage', ['missing_panel', 'incomplete_record', 'wrong_geometry'])
def test_existing_images_require_complete_matching_delivery(tmp_path, damage):
    source, config, directory = completed_ic_lora_images(tmp_path)
    if damage == 'missing_panel':
        (directory / '0001.png').unlink()
    elif damage == 'incomplete_record':
        (directory / 'generation.json').write_text(json.dumps({'status': 'generation_failed'}))
    else:
        Image.new('RGB', (16, 16)).save(directory / '0001.png')
    with pytest.raises(ValueError):
        keyframe_story.completed_images(config, source['prepared'])
