from __future__ import annotations

import copy

from mspc.benchmark.long_story_holdout import MAX_SEGMENT_FRAMES, film_key
from mspc.data import scene_windows as sw

# 场景清单第 1 行（CinePile「10,000 BC」片段）的前 9 镜，原样。
ROW = {"scene_id": "cinepile_t2v:video_batch_20250817_133943_001/10,000_BC_(7_10)_Movie_CLIP_-_We_Hunt_Together_(2008)_HD:scene000", "source": "cinepile_t2v", "video_id": "video_batch_20250817_133943_001/10,000_BC_(7_10)_Movie_CLIP_-_We_Hunt_Together_(2008)_HD", "source_video_path": "${MSPC_SHARED_ROOT}/OoVMetric/resource/cinepile_t2v/raw_videos/video_batch_20250817_133943_001/10,000_BC_(7_10)_Movie_CLIP_-_We_Hunt_Together_(2008)_HD.mp4", "fps": 23.976, "width": 1920, "height": 810, "container_preroll_frames": 0, "active_box": [0, 30, 1920, 810], "video_flags": [], "metadata": {"name_style": "movie_clip_hd", "film_title": "10,000 BC", "year": 2008, "clip_index": 7, "clip_total": 10}, "shots": [{"source_shot_index": 0, "start_frame": 0, "end_frame": 41, "frame_count": 41, "duration_s": 1.71, "flags": []}, {"source_shot_index": 1, "start_frame": 41, "end_frame": 97, "frame_count": 56, "duration_s": 2.336, "flags": []}, {"source_shot_index": 2, "start_frame": 97, "end_frame": 166, "frame_count": 69, "duration_s": 2.878, "flags": []}, {"source_shot_index": 3, "start_frame": 166, "end_frame": 237, "frame_count": 71, "duration_s": 2.961, "flags": []}, {"source_shot_index": 4, "start_frame": 237, "end_frame": 448, "frame_count": 211, "duration_s": 8.8, "flags": []}, {"source_shot_index": 5, "start_frame": 448, "end_frame": 647, "frame_count": 199, "duration_s": 8.3, "flags": []}, {"source_shot_index": 6, "start_frame": 647, "end_frame": 862, "frame_count": 215, "duration_s": 8.967, "flags": []}, {"source_shot_index": 7, "start_frame": 862, "end_frame": 941, "frame_count": 79, "duration_s": 3.295, "flags": []}, {"source_shot_index": 8, "start_frame": 941, "end_frame": 1045, "frame_count": 104, "duration_s": 4.338, "flags": []}]}


def test_training_windows_tile_runs_without_overlap_and_respect_segment_cap() -> None:
    windows = sw.training_windows(ROW, seed=20261005, scene_index=0)
    used = [shot["source_shot_index"] for window in windows for shot in window["shots"]]
    assert len(used) == len(set(used))
    for window in windows:
        assert 1 <= window["shot_count"] <= 5
        assert window["render_frames"] <= MAX_SEGMENT_FRAMES
        indices = [shot["source_shot_index"] for shot in window["shots"]]
        assert indices == list(range(indices[0], indices[0] + len(indices)))
        assert all(shot["render_frames"] % 4 == 1 for shot in window["shots"])
    assert windows == sw.training_windows(ROW, seed=20261005, scene_index=0)


def test_flagged_shot_breaks_the_run() -> None:
    row = copy.deepcopy(ROW)
    row["shots"][3]["flags"] = ["black"]
    windows = sw.training_windows(row, seed=1, scene_index=0)
    assert all(3 not in [s["source_shot_index"] for s in w["shots"]] for w in windows)
    assert all(
        max(s["source_shot_index"] for s in w["shots"]) < 3 or min(s["source_shot_index"] for s in w["shots"]) > 3
        for w in windows
    )


def test_build_excludes_held_out_films_and_subtitled_videos() -> None:
    subtitled = copy.deepcopy(ROW)
    subtitled["video_flags"] = ["burned_in_subtitles"]
    train, teacher, counts = sw.build([ROW, subtitled], held_out_films={film_key(ROW)}, seed=0)
    assert (train, teacher) == ([], [])
    assert counts == {"scenes": 2, "held_out_scenes": 1, "subtitle_scenes": 1}
    train, _, _ = sw.build([ROW], held_out_films=set(), seed=0)
    assert train and all(window["film_key"] == "cinepile_t2v:10 000 bc" for window in train)


def test_sample_frame_positions_stay_inside_the_shot() -> None:
    shot = ROW["shots"][0]
    assert sw.sample_frame_positions(shot, [0.05, 0.5, 0.95]) == [2, 20, 38]


def test_real_short_segments_restore_only_previous_floor_rejections() -> None:
    import json
    from pathlib import Path
    fixtures = json.loads((Path(__file__).parent / 'fixtures/scene_short_windows_real.json').read_text())
    for r in fixtures:
        audit = {}
        windows = sw.training_windows(r['scene'], seed=r['seed'], scene_index=r['scene_index'], audit=audit)
        shorts = {w['window_id']: {k:w[k] for k in ['window_id','shot_count','render_frames','shots']} for w in windows if w['shot_count']<=2}
        for old in r['expected_short_windows']:
            assert shorts[old['window_id']] == old
        old_keys = {w['window_id'] for w in r['expected_short_windows']}
        restored = [w for key,w in shorts.items() if key not in old_keys]
        old_drops = r['expected_rejected_short_windows']
        assert sorted(tuple(s['source_shot_index'] for s in w['shots']) for w in restored) == sorted(
            tuple(d['source_shot_indices']) for d in old_drops if d['reason']=='short_window_below_min_frames')
        assert audit['rejected_short_windows'] == [d for d in old_drops if d['reason']=='short_window_over_frame_cap']
        used = {s['source_shot_index'] for w in windows for s in w['shots']}
        dropped = {d['source_shot_index'] for d in audit['dropped']}
        assert not used & dropped
        assert used | dropped == {s['source_shot_index'] for s in r['scene']['shots']}


def test_real_isolated_one_second_shot_now_forms_a_window() -> None:
    import json
    from pathlib import Path
    r=json.loads((Path(__file__).parent/'fixtures/scene_one_second_run_real.json').read_text())
    assert 1 < r['seconds'] < 1.15
    runs,_=sw.training_runs(r['scene'])
    assert any([s['source_shot_index'] for s in run]==[r['isolated_shot_index']] for run in runs)
    windows=sw.training_windows(r['scene'],seed=r['seed'],scene_index=r['scene_index'])
    window=next(w for w in windows if [s['source_shot_index'] for s in w['shots']]==[r['isolated_shot_index']])
    assert window['shot_count']==1 and window['render_frames'] in (13,17)
