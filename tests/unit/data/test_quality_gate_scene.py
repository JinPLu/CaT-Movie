"""200 条真实窗读帧后固化的场景输入；不在单测中访问服务器媒体。"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mspc.data.quality_scorer.adapters.media import SceneFrameReader

ROOT = Path(__file__).resolve().parents[3]
ROWS = json.loads((ROOT / 'tests/fixtures/quality_gate_scene_rows.json').read_text())


def test_coordinate_and_metadata_are_required(tmp_path):
    with pytest.raises(ValueError, match='ffmpeg_sequential'):
        SceneFrameReader(ROWS[0], frame_coordinate='decord_zero_based',
                         ffmpeg='ffmpeg', error_path=tmp_path/'e')
    row = deepcopy(ROWS[0])
    del row['width']
    with pytest.raises(KeyError):
        SceneFrameReader(row, frame_coordinate='ffmpeg_sequential_zero_based',
                         ffmpeg='ffmpeg', error_path=tmp_path/'e')
