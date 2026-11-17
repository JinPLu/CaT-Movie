"""固定亲看量具的查表、首尾并界、任一帧规则回归。"""
import json
from pathlib import Path

import numpy as np
import pytest

from mspc.data.shot_gate import gate_shot, gate_window
from mspc.data.shot_boundary_revision import partition_boundary_cuts

TRUTH = [json.loads(line) for line in (Path(__file__).resolve().parents[2] / 'fixtures/intra_shot_cut_audit/reviewed_shots_20261011.jsonl').read_text().splitlines()]
RULE = dict(boundary_threshold=.5, single_threshold=.25, content_threshold=25, edge_seconds=.5, boundary_guard_frames=0)


def table(row):
    return dict(video=row['case_id'], coordinate='decord', fps=row['fps'], frame_count=row['frames'],
                single=np.asarray(row['single']), content=np.asarray([0., *row['content']]))


def shot(row):
    return dict(video=row['case_id'], segment=[0, row['frames']])


def test_all_real_old_scores_under_full_guarded_single():
    dirty = {(r['source'], r['window_id']) for r in TRUTH if r['cuts']}
    marked, flagged = set(), []
    for row in TRUTH:
        result = gate_shot(shot(row), table(row), coordinate='decord', **RULE)
        if result['status'] == 'rejected':
            marked.add((row['source'], row['window_id'])); flagged.append(row)
    assert len(marked & dirty) == 11
    assert len(marked - dirty) == 6
    assert sum(bool(r['cuts']) for r in flagged) == 13
    assert sum(not r['cuts'] for r in flagged) == 9


def test_real_peak_outside_edge_does_not_hide_interior_frame():
    row = next(r for r in TRUTH if r['window_id'] == 'floor-0285' and r['shot'] == 1)
    result = gate_shot(shot(row), table(row), coordinate='decord', **RULE)
    assert {'signal': 'content', 'frame': 51, 'score': 27.26} in result['reasons']


def test_offset_lookup_uses_source_coordinates():
    row = next(r for r in TRUTH if r['window_id'] == 'floor-0285' and r['shot'] == 1)
    scores = table(row)
    original = gate_shot(shot(row), scores, coordinate='decord', **RULE)
    offset = 130
    for name in ('single', 'content'):
        scores[name] = np.pad(scores[name], (offset, 17))
    scores['frame_count'] += offset + 17
    shifted = gate_shot({**shot(row), 'segment': [offset, offset + row['frames']]}, scores, coordinate='decord', **RULE)
    assert [{**r, 'frame': r['frame'] - offset} for r in shifted['reasons']] == original['reasons']


def test_real_window_family_mismatch_and_missing_scores_raise():
    row = TRUTH[0]
    window = dict(window_id=row['window_id'], family='scene_window_v1', shots=[shot(row)])
    with pytest.raises(ValueError, match='coordinate'):
        gate_window(window, {row['case_id']: table(row)}, **RULE)
    with pytest.raises(KeyError):
        gate_window(window, {}, **RULE)


def test_real_source_range_mismatch_raises():
    row = TRUTH[0]
    with pytest.raises(ValueError, match='frame count'):
        gate_shot({**shot(row), 'segment': [0, row['frames'] + 1]}, table(row), coordinate='decord', **RULE)


def test_threshold_and_coordinate_are_required():
    row = TRUTH[0]
    with pytest.raises(TypeError):
        gate_shot(shot(row), table(row), **RULE)
    for key in RULE:
        with pytest.raises(TypeError):
            gate_shot(shot(row), table(row), coordinate='decord', **{k: v for k, v in RULE.items() if k != key})


def test_boundary_partition_preserves_last_head_first_tail():
    # 纯区间数学：首尾恰好半秒属于边界带，head优先。
    revised, mid = partition_boundary_cuts(start=100, end=200, fps=20, cuts=[101, 110, 150, 190, 199], edge_seconds=.5)
    assert revised == [110, 190]
    assert mid == [150]


def test_exact_threshold_and_strict_interior():
    # 纯阈值数学：等号触发；镜内检查不把 single 的帧号加一。
    scores = dict(video='threshold', coordinate='decord', fps=10., frame_count=30,
                  single=np.zeros(30), content=np.zeros(30))
    scores['single'][5] = .5
    scores['single'][6] = .25
    scores['content'][25] = 25
    scores['content'][24] = 25
    result = gate_shot(dict(video='threshold', segment=[0, 30]), scores, coordinate='decord', **RULE)
    assert result['boundary_cuts'] == [6]
    assert [(r['signal'], r['frame']) for r in result['reasons']] == [('single', 5), ('single', 6), ('content', 24)]


def test_real_mixed_pool_csv_preserves_paths_segments_and_family(tmp_path):
    import csv
    from mspc.data.shot_gate import read_windows
    fixture = Path(__file__).resolve().parents[2] / 'fixtures/intra_shot_cut_audit/pool_input_rows.json'
    rows = json.loads(fixture.read_text())
    path = tmp_path / 'pool.csv'
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    windows = read_windows(path, family='pool205k')
    for source, window in zip(rows, windows, strict=True):
        assert window['family'] == source['dataset']
        assert [s['video'] for s in window['shots']] == json.loads(source['source_video_paths'])
        assert [s['segment'] for s in window['shots']] == json.loads(source['source_shot_segments'])


def test_window_result_exposes_the_family_coordinate_for_catalog():
    row = TRUTH[0]
    window = dict(window_id=row['window_id'], family='ci_vid', shots=[shot(row)])
    result = gate_window(window, {row['case_id']: table(row)}, **RULE)
    assert result['coordinate'] == result['shots'][0]['coordinate'] == 'decord'
