"""Regression from real FX3 scene015; full pixel replay lives on main-server."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from mspc.data.multishot import read_video_frames


def test_metadata_probe_does_not_require_unrequested_broken_first_frame(monkeypatch):
    row = json.loads((Path(__file__).parents[2] / 'fixtures/fx3_decoder/scene015.json').read_text())
    requested = row['requested_decord_frame']

    class Reader:
        def __init__(self, path, **kwargs):
            self.height = kwargs.get('height', row['height'])
            self.width = kwargs.get('width', row['width'])

        def __getitem__(self, index):
            if index == 0:
                raise RuntimeError(row['unrequested_frame_zero_error'])
            assert index == requested
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def get_batch(self, indices):
            assert indices == [requested]
            data = np.full((1, self.height, self.width, 3), 17, dtype=np.uint8)
            return SimpleNamespace(asnumpy=lambda: data)

    monkeypatch.setitem(sys.modules, 'decord', SimpleNamespace(VideoReader=Reader, cpu=lambda index: index))
    result = read_video_frames(path=row['source'], frame_indices=[requested], target_height=480,
                               target_width=832, transform_centercrop=lambda t: t[:, :, :, 10:842],
                               num_threads=1, output_mode='uint8')
    assert tuple(result.shape) == (3, 1, 480, 832)
    assert int(result.min()) == int(result.max()) == 17
