"""真实200镜面板的三来源元数据：旧480回归及720生产媒体合同。"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mspc.data.frames import center_crop
from mspc.data.multishot import MultishotCsvDataset
from mspc.data.scene_cache import read_scene_video
from mspc.data.scene_training_manifest import ANNOTATION_MEDIA, training_geometry
from mspc.data.shot_targets import shot_row, write_manifest_csv, member_target_frame_counts

CASES = json.loads((Path(__file__).parent / 'fixtures/training_media_real.json').read_text())['cases']
MEDIA720 = dict(ANNOTATION_MEDIA, width=1280, height=704, target_fps=24)


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['key'])
@pytest.mark.parametrize('media,frame_key,crop_key', [
    (ANNOTATION_MEDIA, '16', '480x832'), (MEDIA720, '24', '704x1280')])
def test_production_sampling_matches_the_frozen_real_panel(case, media, frame_key, crop_key):
    original = deepcopy(case['scene'])
    row = shot_row(case['scene'], case['position'], media=media)
    indices = row['source_frame_indices_by_shot'][0]
    assert indices == case['expected_frames'][frame_key]
    assert row['train_crop_box'] == case['expected_crop_boxes'][crop_key]
    assert row['shot_groups'] == [[0, len(indices)]]
    assert len(set(indices)) == len(indices) and (len(indices) - 1) % 4 == 0
    shot = case['scene']['shots'][0]
    assert [indices[0], indices[-1]] == [shot['start_frame'], shot['end_frame'] - 1]
    member = {"scene": {k: v for k, v in case['scene'].items() if k != 'shots'},
              "shots": [{**shot, "usable": True, "frames": shot['render_frames']}]}
    assert member_target_frame_counts(member, media=media) == (len(indices),)
    assert case['scene'] == original  # 派生元数据不覆盖原16fps标注成员


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['key'])
def test_720_producer_csv_reader_and_geometry_share_the_same_contract(case, tmp_path, monkeypatch):
    import mspc.data.scene_cache as owner
    scene = case['scene']
    row = shot_row(scene, 0, media=MEDIA720)
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'only provenance stat; the decoder is controlled below')
    row['video'] = str(source)
    csv = tmp_path / 'scene.csv'
    write_manifest_csv(csv, [row])
    monkeypatch.setattr(owner, 'ffmpeg_runtime', lambda: ('ffmpeg', 'controlled-test'))
    dataset = MultishotCsvDataset(metadata_path=str(csv), repeat=1,
        **MEDIA720, frame_policy='fps_aware_target_fps', schema='scene_window_v1',
        prompt_format='caption_json_v1', video_column='video', caption_column=None,
        shot_groups_column='shot_groups', case_id_column='case_id', case_id_prefix='scene',
        video_output_mode='uint8')
    description = dataset.describe_media(0)
    assert description['frame_plan']['frame_indices_by_shot'] == [case['expected_frames']['24']]
    geometry = description['provenance']['preprocessing']['spatial'][0]
    expected_geometry = training_geometry(scene, media=MEDIA720)
    assert geometry == expected_geometry
    assert geometry['output_size'] == [1280, 704]
    # 实源尺寸与预先冻结crop，独立构造像素：验证reader确实按新crop读，而非只换sample的height/width标签。
    yy, xx = np.indices((scene['height'], scene['width']))
    native = np.stack((xx % 251, yy % 251, (xx + yy) % 251), axis=-1).astype(np.uint8)
    first = case['expected_frames']['24'][0]
    def decode(unused_scene, indices, **kwargs):
        assert indices == [first]
        yield first, native
    monkeypatch.setattr('mspc.data.frames.iter_scene_frames', decode)
    pixels = read_scene_video(description['row'], [first], output_mode='uint8',
                              video_decode_num_threads=1, media=MEDIA720)
    expected = np.asarray(center_crop(Image.fromarray(native).crop(case['expected_crop_boxes']['704x1280']),
                                      width=1280, height=704))
    assert pixels.dtype == torch.uint8 and tuple(pixels.shape) == (3, 1, 704, 1280)
    assert np.array_equal(pixels[:, 0].permute(1, 2, 0).numpy(), expected)


def test_720_reader_rejects_old_contract_csv_instead_of_resizing_it(tmp_path):
    row = shot_row(CASES[0]['scene'], 0, media=ANNOTATION_MEDIA)
    csv = tmp_path / 'old.csv'
    write_manifest_csv(csv, [row])
    dataset = MultishotCsvDataset(metadata_path=str(csv), repeat=1, **MEDIA720,
        frame_policy='fps_aware_target_fps', schema='scene_window_v1', prompt_format='caption_json_v1',
        video_column='video', caption_column=None, shot_groups_column='shot_groups',
        case_id_column='case_id', case_id_prefix='scene', video_output_mode='uint8')
    with pytest.raises(ValueError, match='configured task media'):
        dataset._prepare_row(0, dataset.data[0].copy())
