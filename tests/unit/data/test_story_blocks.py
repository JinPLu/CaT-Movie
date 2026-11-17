"""D1：真实判镜记录、质量断点、块底线、均分与代理拼接。"""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
from mspc.data.scene_windows import balanced_block_sizes, training_runs, story_blocks

REPO = Path(__file__).resolve().parents[3]
REAL = json.loads((REPO / 'tests/fixtures/blocks/judged_real.json').read_text())['scenes']
SPEC = importlib.util.spec_from_file_location('preprocess_sources_blocks', REPO / 'tools/data/preprocess_sources.py')
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)
CFG = json.loads((REPO / 'configs/data/source_pipeline_v1.json').read_text())


def build(kind, excluded=(), scores=None):
    data = REAL[kind]
    sid = data['scene']['sid']
    judged = pipeline.scored_scenes(sid, [data['scene']], data['saved_shots'],
                                    data['scores'] if scores is None else scores, CFG, set(excluded))
    blocks, audit = story_blocks(judged, sid=sid, **CFG['blocks'],
                                 thresholds=CFG['thresholds'], min_window_seconds=CFG['min_window_seconds'])
    return judged, blocks, audit


@pytest.mark.parametrize('length, expected', [(1, [1]), (12, [12]), (13, [7, 6]), (24, [12, 12]),
                                              (25, [9, 8, 8]), (69, [12, 12, 12, 11, 11, 11])])
def test_balanced_lengths(length, expected):
    sizes = balanced_block_sizes(length, 12)
    assert sizes == expected
    cursor, actual = 0, []
    for size in sizes:
        actual.extend(range(cursor, cursor + size))
        cursor += size
    assert actual == list(range(length))
    assert max(sizes) <= 12 and max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize('length, cap', [(0, 12), (-1, 12), (1.5, 12), (1, 0), (1, 12.0), (1, True)])
def test_invalid_sizes_fail(length, cap):
    with pytest.raises(ValueError):
        balanced_block_sizes(length, cap)



def test_real_low_aesthetic_breaks_middle_of_run():
    judged, blocks, audit = build('low_middle')
    low = {s['source_shot_index'] for row in judged for s in row['shots'] if 'below_laion_aesthetic' in s['flags']}
    assert low
    assert any(min(s['source_shot_index'] for s in row['shots']) < i < max(s['source_shot_index'] for s in row['shots'])
               for row in judged for i in low)
    assert all(not low.intersection(s['source_shot_index'] for s in b['scene']['shots']) for b in blocks)
    for b in blocks:
        indices = [s['source_shot_index'] for s in b['scene']['shots']]
        assert indices == list(range(indices[0], indices[-1] + 1))
    assert any('below_laion_aesthetic' in r['reasons'] for r in audit['rejected'])


def test_real_short_single_is_rejected_at_block_level():
    judged, blocks, audit = build('short_single')
    short = {r['source_shot_index'] for r in audit['rejected'] if r['reason'] == 'block_below_min_seconds'}
    assert short
    assert all(sum(s['seconds'] for s in b['scene']['shots']) >= 2 for b in blocks)
    assert not short.intersection(s['source_shot_index'] for b in blocks for s in b['scene']['shots'])


def test_real_long_run_uses_raw_touching_intervals_and_preserves_revision():
    judged, blocks, audit = build('long')
    runs, _ = training_runs(judged[0])
    long = next(run for run in runs if len(run) > 12)
    expected = balanced_block_sizes(len(long), 12)
    members = {s['source_shot_index'] for s in long}
    actual = [b['scene']['shot_count'] for b in blocks if b['scene']['shots'][0]['source_shot_index'] in members]
    assert actual == expected
    saved = {s['source_shot_index']: s for s in judged[0]['shots']}
    identities = []
    old = {(r['scene']['scene_id'], s['source_shot_index']): (s, revision)
           for r in REAL['long']['windows'] for s, revision in zip(r['scene']['shots'], r['scene']['boundary_revision']['shots'])}
    for b in blocks:
        scene = b['scene']
        assert scene['render_frames'] == sum(s['render_frames'] for s in scene['shots'])
        for shot, revision in zip(scene['shots'], scene['boundary_revision']['shots']):
            original = saved[shot['source_shot_index']]
            assert [shot['start_frame'], shot['end_frame']] == original['gate']['revised_segment']
            assert shot['scene_id'] == scene['scene_id'] and shot['sid'] == REAL['long']['scene']['sid']
            assert shot['score_key'] in REAL['long']['scores'] and 'window' not in shot
            identities.append(shot['source_shot_index'])
            if (scene['scene_id'], shot['source_shot_index']) in old:
                old_shot, old_revision = old[scene['scene_id'], shot['source_shot_index']]
                assert {k:v for k,v in shot.items() if k not in ('sid','scene_id','score_key')} == old_shot
                assert revision == old_revision
        for a, b in zip(scene['shots'], scene['shots'][1:]):
            assert saved[a['source_shot_index']]['end_frame'] == saved[b['source_shot_index']]['start_frame']
            assert a['end_frame'] < b['start_frame']
    eligible = {s['source_shot_index'] for s in judged[0]['shots'] if s['eligible']}
    rejected = {r['source_shot_index'] for r in audit['rejected']}
    assert len(identities) == len(set(identities))
    assert set(identities).isdisjoint(rejected)
    assert set(identities) | rejected == eligible


def test_real_score_missing_and_eval_exclusion_break_runs():
    _, baseline, _ = build('long')
    shot = next(s for b in baseline for s in b['scene']['shots'])
    identity = (shot['sid'], shot['scene_id'], shot['source_shot_index'])
    scores = dict(REAL['long']['scores']); scores.pop(shot['score_key'])
    for kwargs, reason in [({'scores': scores}, 'score_missing'), ({'excluded': [identity]}, 'eval_excluded')]:
        _, blocks, audit = build('long', **kwargs)
        assert any(r['source_shot_index'] == identity[2] and reason in r['reasons'] for r in audit['rejected'])
        assert all(s['source_shot_index'] != identity[2] for b in blocks for s in b['scene']['shots'])


def test_source_blocks_reuses_real_windows_and_concat_mapping(tmp_path, monkeypatch):
    data = REAL['long']; sid = data['scene']['sid']
    folder = tmp_path/'sources'/sid; folder.mkdir(parents=True)
    pipeline.write(folder/'shots.jsonl', data['saved_shots'])
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b'copied proxy')
    monkeypatch.setattr(pipeline.subprocess, 'run', run)
    cfg = {**CFG, 'encode_ffmpeg': 'ffmpeg'}
    done = pipeline.source_blocks(sid, [data['scene']], cfg, tmp_path, data['scores'], set(), data['windows'])
    built = pipeline.read(folder/'blocks.jsonl')
    assert done['eligible'] == done['shots'] + done['rejected']
    by_key = {r['key']:r for r in data['windows']}
    assert any(b['same_as_window'] for b in built)
    for b in built:
        parts = [data['scores'][s['score_key']]['proxy'] for s in b['scene']['shots']]
        expected = [{'shot':i,'source_frame':f} for i,p in enumerate(parts) for f in p['source_frames']]
        assert b['proxy']['proxy_to_source_frames'] == expected
        assert b['proxy']['shot_timestamps'][-1]['end'] == sum(p['frames'] for p in parts)/2
        if b['same_as_window']:
            assert b['proxy'] == by_key[b['same_as_window']]['proxy']
    assert all(c[c.index('-c')+1] == 'copy' for c in commands)


@pytest.mark.parametrize('cfg', [{}, {'blocks': {}}, {'blocks': {'max_shots': 12}}])
def test_missing_block_config_fails_before_io(tmp_path, cfg):
    with pytest.raises((KeyError, TypeError)):
        pipeline.blocks(cfg, tmp_path, 1, 0)
    assert not list(tmp_path.iterdir())
