import json
from pathlib import Path

from PIL import Image

from mspc.comparison_methods.ic_lora import conditioning_canvas, prepare_story


def test_real_ten_shot_story_uses_three_panels_without_losing_input_or_regenerating_shots():
    source = json.loads((Path(__file__).parents[2] / 'fixtures/ic_lora/cinedance-014847-b0000-shot01.json').read_text())
    prepared = prepare_story(source, {'width': 832, 'panel_height': 480,
                                      'character_names': {'[C1]': 'Alice', '[C2]': 'Ben'}})
    assert prepared['shots'] == source['shots']
    assert prepared['canvas_size'] == [832, 1440]
    assert [w['shot_indices'] for w in prepared['windows']] == [
        [0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [7, 8, 9]]
    produced = []
    for window in prepared['windows']:
        assert window['prompt'] == window['prompt_2']
        assert set(window['preserved_indices']) <= set(produced)
        produced.extend(window['generated_indices'])
        for panel, index in enumerate(window['shot_indices']):
            text = source['shots'][index]['tagged'].replace('[C1]', '<Alice>').replace('[C2]', '<Ben>')
            assert f'[SCENE-{panel + 1}] {text}' in window['prompt_2']
    assert produced == list(range(10))
    assert prepared['windows'][-1]['preserved_indices'] == [7, 8]


def test_final_window_preserves_both_previous_panels_and_only_masks_the_last(tmp_path):
    Image.new('RGB', (832, 480), (20, 70, 120)).save(tmp_path / '0008.png')
    Image.new('RGB', (832, 480), (150, 40, 90)).save(tmp_path / '0009.png')
    canvas, mask = conditioning_canvas({'shot_indices': [7, 8, 9], 'preserved_indices': [7, 8]},
                                        tmp_path, (832, 480))
    assert canvas.size == mask.size == (832, 1440)
    assert canvas.crop((0, 0, 832, 480)).getextrema() == ((20, 20), (70, 70), (120, 120))
    assert canvas.crop((0, 480, 832, 960)).getextrema() == ((150, 150), (40, 40), (90, 90))
    assert mask.crop((0, 0, 832, 960)).getextrema() == (0, 0)
    assert mask.crop((0, 960, 832, 1440)).getextrema() == (255, 255)
