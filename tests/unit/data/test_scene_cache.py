"""场景清单来自 2026-10-12 final/scene_manifest.csv，三来源各含真实单镜/五镜窗。"""
import csv
import json
from pathlib import Path

import pytest

from mspc.data.multishot import MultishotCsvDataset
from mspc.data.frames import shot_grid

ROWS = json.loads((Path(__file__).parents[2] / 'fixtures/scene_cache/real_rows.json').read_text())


def dataset(tmp_path, row, **workpoint):
    manifest = tmp_path / 'rows.csv'
    with manifest.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    geometry = dict(height=480, width=832, max_rgb_frames=308, target_fps=16)
    geometry.update(workpoint)
    return MultishotCsvDataset(metadata_path=str(manifest), repeat=1, **geometry,
        frame_policy='fps_aware_target_fps',
        vae_temporal_stride=4, schema='scene_window_v1', prompt_format='caption_json_v1',
        video_column='video', caption_column=None, shot_groups_column='shot_groups',
        case_id_column='case_id', case_id_prefix='scene', video_output_mode='uint8')


def prepared_plan(ds):
    row = ds._prepare_row(0, ds.data[0].copy())
    meta = row['_scene_metadata']
    n = len(row['_shot_groups'])
    plan = ds.frame_planner.build_multisource_explicit_frame_plan(
        source_video_paths=[row['video']] * n, source_shot_segments=row['_source_shot_segments'],
        train_shot_groups=row['_shot_groups'], frame_indices_by_shot=row['_source_frame_indices_by_shot'],
        data_path=row['video'], source_fps_by_shot=[meta['source_fps']] * n,
        source_num_frames_by_shot=[meta['source_num_frames']] * n)
    return row, plan


@pytest.mark.parametrize('row', ROWS, ids=[r['window_key'] for r in ROWS])
def test_real_scene_mapping_and_lengths(tmp_path, row):
    parsed, plan = prepared_plan(dataset(tmp_path, row))
    assert plan['frame_indices_by_shot'] == json.loads(row['source_frame_indices_by_shot'])
    assert plan['shot_groups'] == json.loads(row['shot_groups'])
    assert len(plan['frame_indices']) == int(row['render_frames'])
    for segment, group, indices in zip(parsed['_source_shot_segments'], parsed['_shot_groups'], plan['frame_indices_by_shot'], strict=True):
        assert indices == shot_grid(*segment, group[1] - group[0])
        assert len(indices) == group[1] - group[0]
        assert (len(indices) - 1) % 4 == 0
    assert parsed['_prompts'] == ()
    assert parsed['_prompt_source_path'] is None


@pytest.mark.parametrize('value', ['', 'null', '[0,0,0,0]', '[1,0,1896,1080]'])
def test_missing_or_invalid_crop_rejected(tmp_path, value):
    row = dict(ROWS[0], train_crop_box=value)
    with pytest.raises(ValueError, match='train_crop_box'):
        prepared_plan(dataset(tmp_path, row))


def test_wrong_source_frame_rejected(tmp_path):
    row = dict(ROWS[0])
    indices = json.loads(row['source_frame_indices_by_shot'])
    indices[0][0] += 1
    row['source_frame_indices_by_shot'] = json.dumps(indices)
    with pytest.raises(ValueError, match='deterministic mapping'):
        prepared_plan(dataset(tmp_path, row))


def test_wrong_shot_length_rejected(tmp_path):
    row = dict(next(r for r in ROWS if r['shot_count'] == '1'))
    groups = json.loads(row['shot_groups'])
    groups[0][1] -= 4
    row['shot_groups'] = json.dumps(groups)
    with pytest.raises(ValueError, match='train shot lengths'):
        prepared_plan(dataset(tmp_path, row))


def test_scene_description_uses_manifest_not_decord(tmp_path, monkeypatch):
    import mspc.data.multishot as multishot
    import mspc.data.scene_cache as scene_cache
    video = tmp_path / 'source.mp4'
    video.write_bytes(b'only stat is used')
    def no_decord(*args, **kwargs):
        raise AssertionError('scene must not use decord')
    monkeypatch.setattr(multishot, 'read_video_metadata', no_decord)
    monkeypatch.setattr(scene_cache, 'ffmpeg_runtime', lambda: ('ffmpeg', 'test FFmpeg'))
    description = dataset(tmp_path, dict(ROWS[0], video=str(video))).describe_media(0)
    assert description['frame_plan']['source_fps'] == float(ROWS[0]['source_fps'])
    assert description['frame_plan']['source_num_frames'] == int(ROWS[0]['source_num_frames'])
    pre = description['provenance']['preprocessing']
    assert pre['decoder'] == 'ffmpeg-sequential'
    assert pre['preroll_added'] == 0
    assert pre['spatial'][0]['train_crop_box'] == json.loads(ROWS[0]['train_crop_box'])


def explicit_720_row(tmp_path):
    row = json.loads((Path(__file__).parents[2] / 'fixtures/scene_cache/explicit_720_row.json').read_text())
    video = tmp_path / 'native.mp4'
    video.write_bytes(b'only stat is used')
    return {key: json.dumps(value) if isinstance(value, list) else '' if value is None else value
            for key, value in {**row, 'video': str(video)}.items()}


def test_frozen_720_plan_keeps_long_source_grid_without_resampling(tmp_path, monkeypatch):
    from mspc.data.frame_planner import FpsAwareTargetFpsPlanner
    import mspc.data.scene_cache as scene_cache
    row = explicit_720_row(tmp_path)
    def no_resampling(*args, **kwargs):
        raise AssertionError('frozen source indices must not be regenerated')
    monkeypatch.setattr(FpsAwareTargetFpsPlanner, 'clean_indices_for_shot', no_resampling)
    monkeypatch.setattr(scene_cache, 'ffmpeg_runtime', lambda: ('ffmpeg', 'test FFmpeg'))
    description = dataset(tmp_path, row, height=704, width=1280, target_fps=24, max_rgb_frames=337).describe_media(0)
    plan = description['frame_plan']
    assert plan['frame_indices_by_shot'] == json.loads(row['source_frame_indices_by_shot'])
    assert plan['shot_groups'] == [[0, 337]]
    assert plan['latent_shot_groups'] == [[0, 85]]
    geometry = description['provenance']['preprocessing']['spatial'][0]
    assert geometry['train_crop_box'] == json.loads(row['train_crop_box'])
    assert geometry['output_size'] == [1280, 704]
    assert geometry['crop_box_resized'] == [0, 0, 1280, 704]
    x0, y0, x1, y1 = geometry['train_crop_box']
    sx, sy = 1280 / (x1 - x0), 704 / (y1 - y0)
    assert geometry['source_to_target_matrix'] == [[sx, 0, -sx*x0], [0, sy, -sy*y0], [0, 0, 1]]


def test_explicit_720_cells_must_match_config(tmp_path):
    row = explicit_720_row(tmp_path)
    row['height'] = 480
    with pytest.raises(ValueError, match='spacetime cells'):
        dataset(tmp_path, row, height=704, width=1280, target_fps=24, max_rgb_frames=337).describe_media(0)


def test_scene_decoder_uses_explicit_pixels_and_thread_count(tmp_path, monkeypatch):
    import numpy as np
    import mspc.data.scene_cache as scene_cache
    import mspc.data.frames as frames
    from PIL import Image
    raw = explicit_720_row(tmp_path)
    ds = dataset(tmp_path, raw, height=704, width=1280, target_fps=24, max_rgb_frames=337)
    row = ds._prepare_row(0, ds.data[0].copy())
    h, w = int(raw['source_height']), int(raw['source_width'])
    y, x = np.arange(h, dtype=np.uint16)[:, None], np.arange(w, dtype=np.uint16)[None, :]
    source = np.empty((h, w, 3), dtype=np.uint8)
    source[..., 0], source[..., 1], source[..., 2] = x % 256, y % 256, (x+y) % 256
    calls = []
    def decode(scene, indices, **kwargs):
        calls.append(kwargs['video_decode_num_threads'])
        for index in indices:
            yield index, source
    monkeypatch.setattr(frames, 'iter_scene_frames', decode)
    monkeypatch.setattr(scene_cache, 'ffmpeg_runtime', lambda: ('ffmpeg', 'test FFmpeg'))
    video = scene_cache.read_scene_video(row, [row['_source_frame_indices_by_shot'][0][0]],
        output_mode='uint8', video_decode_num_threads=2, media=ds.media)
    expected = np.asarray(frames.center_crop(Image.fromarray(source).crop(row['_scene_boxes']['train_crop_box']),
        width=1280, height=704))
    assert tuple(video.shape) == (3, 1, 704, 1280)
    assert video.is_contiguous()
    assert np.array_equal(video[:, 0].permute(1, 2, 0).numpy(), expected)
    assert calls == [2]


def test_derived_media_cells_are_checked_against_explicit_task(tmp_path):
    row = dict(ROWS[0], target_height=480, target_width=832, target_fps=16,
               vae_temporal_stride=4, max_rgb_frames=308)
    prepared_plan(dataset(tmp_path, row))
    row['target_width'] = 1280
    with pytest.raises(ValueError, match='spacetime cells.*target_width'):
        prepared_plan(dataset(tmp_path, row))
