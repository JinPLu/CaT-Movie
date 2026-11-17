import json

import pytest

from mspc.benchmark.generation import load_requests
from mspc.evaluation import st_bench

CONFIG = {'shot_frames': 81, 'fps': 16, 'top_pairs': 2, 'viclip_batch_size': 8,
          'source': {'repository': 'https://github.com/Kevin-thu/StoryMem', 'commit': 'abc', 'stories': 1, 'shots': 3}}


def write_story(directory, name='cat'):
    # Structure of StoryMem story/cat.json (scene 1 cut flags [true, false, ...]), prompts shortened.
    story = {'story_name': 'A Cat', 'story_overview': 'A house cat spends a quiet day.',
             'scenes': [{'scene_num': 1, 'video_prompts': ['Cat stretches.', 'Cat jumps down.'],
                         'first_frame_prompt': ['a', 'b'], 'cut': [True, False]},
                        {'scene_num': 2, 'video_prompts': ['Cat chases light.'],
                         'first_frame_prompt': ['c'], 'cut': [True]}]}
    path = directory / f'{name}.json'
    path.write_text(json.dumps(story))
    return path


def test_load_story_flattens_scenes_and_keeps_cut_flags(tmp_path):
    story = st_bench.load_story(write_story(tmp_path))
    assert story['story_id'] == 'st-bench-cat'
    assert [(s['scene'], s['cut']) for s in story['shots']] == [(1, True), (1, False), (2, True)]


def test_load_story_rejects_misaligned_cut_flags(tmp_path):
    path = write_story(tmp_path)
    data = json.loads(path.read_text())
    data['scenes'][0]['cut'] = [True]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='disagree'):
        st_bench.load_story(path)


def test_request_catalog_is_accepted_by_generation_entry(tmp_path):
    config = {**CONFIG, 'source': {**CONFIG['source'], 'story_dir': str(tmp_path)}}
    write_story(tmp_path)
    catalog = st_bench.request_catalog(st_bench.load_stories(config), config)
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps(catalog))
    story, = load_requests(path)['stories']
    assert story['target_seconds'] == pytest.approx(3 * 81 / 16)
    assert story['inputs']['shots']['expected_shot_count'] == 3
    # every shot is fixed at 81 frames; shot 2's continuation stays in the text note.
    assert story['inputs']['shots']['expected_cut_times_seconds'] == pytest.approx([81 / 16, 2 * 81 / 16])
    assert story['inputs']['shots']['text'].splitlines()[3:] == [
        's01: Cat stretches.', 's02 (continues s01 without a cut): Cat jumps down.', 's03: Cat chases light.']


def test_load_stories_checks_published_counts(tmp_path):
    write_story(tmp_path)
    config = {**CONFIG, 'source': {**CONFIG['source'], 'story_dir': str(tmp_path), 'shots': 300}}
    with pytest.raises(ValueError, match='expected 1 / 300'):
        st_bench.load_stories(config)


def test_top_pairs_ranks_by_prompt_similarity_then_shot_order():
    features = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]
    assert st_bench.top_pairs(features, 3) == [(0, 1), (2, 3), (0, 3)]
    with pytest.raises(ValueError):
        st_bench.top_pairs(features[:2], 2)


class FakeViCLIP:
    """Each shot's video feature is a one-hot of its first sampled frame index; texts map to fixed vectors."""

    def embed_videos(self, shots, *, viclip_shot_batch_size):
        return [[1.0, 0.0] if frames[0] < 20 else [0.0, 1.0] for frames in shots]

    def score_shots(self, items, *, viclip_shot_batch_size):
        return [0.2 + 0.1 * i for i, _ in enumerate(items)]

    def score(self, frames, text):
        assert len(frames) == 8 and frames[0] == 0
        return 0.25

    def embed_texts(self, texts):
        return [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]


def test_viclip_metrics_uses_shot_boundaries_and_selected_pairs(tmp_path):
    story = st_bench.load_story(write_story(tmp_path))
    result = st_bench.viclip_metrics(frames=list(range(30)), shot_frame_counts=[10, 10, 10], story=story,
                                     viclip=FakeViCLIP(), config=CONFIG)
    assert result['prompt_following_global'] == 0.25
    assert result['prompt_following_single_shot'] == pytest.approx(0.3)
    # shots 0 and 1 start below frame 20 -> identical features; shot 2 is orthogonal.
    assert result['cross_shot_consistency_all_pairs'] == pytest.approx(1 / 3)
    assert result['top_pairs'] == [[0, 1], [1, 2]]
    assert result['cross_shot_consistency_top_pairs'] == pytest.approx(0.5)
    # shot 1 continues shot 0 (cut=false), so only the pairs across takes, (0,2) and (1,2), count here.
    assert st_bench.takes(story) == [0, 0, 1]
    assert result['cross_shot_consistency_cross_take_pairs'] == pytest.approx(0.0)


def test_viclip_metrics_rejects_shot_count_mismatch(tmp_path):
    story = st_bench.load_story(write_story(tmp_path))
    with pytest.raises(ValueError, match='2 generated shots for 3 prompts'):
        st_bench.viclip_metrics(frames=list(range(20)), shot_frame_counts=[10, 10], story=story,
                                viclip=FakeViCLIP(), config=CONFIG)
