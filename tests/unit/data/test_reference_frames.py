"""回归正式1256301的真实重复PTS；完整像素对拍在训练集群真实视频上执行。"""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from mspc.data.scene_training_manifest import ANNOTATION_MEDIA
from mspc.data import reference_frames as reader


def test_duplicate_presentation_timestamps_keep_both_frame_ordinals(monkeypatch):
    evidence = json.loads((Path(__file__).parents[2] / 'fixtures/reference_frame_timestamps.json').read_text())
    timestamps = evidence['decoded_pts']
    assert timestamps == sorted(evidence['packet_pts'])
    assert timestamps[495] == timestamps[496]
    stream = SimpleNamespace(thread_count=0)
    seeks = []

    class Frame:
        colorspace = color_range = 0
        def __init__(self, index, pts):
            self.index, self.pts = index, pts
        def to_ndarray(self, **kwargs):
            return np.full((2, 2, 3), self.index % 256, dtype=np.uint8)

    class Container:
        streams = SimpleNamespace(video=[stream])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def demux(self, stream):
            return [SimpleNamespace(pts=pts, size=1, is_discard=False) for pts in evidence['packet_pts']]
        def seek(self, pts, **kwargs): seeks.append(pts)
        def decode(self, stream):
            return (Frame(i, pts) for i, pts in enumerate(timestamps))

    monkeypatch.setitem(sys.modules, 'av', SimpleNamespace(open=lambda p: Container(),
        video=SimpleNamespace(reformatter=SimpleNamespace(ColorRange=SimpleNamespace(JPEG=2)))))
    monkeypatch.setattr(reader, 'reroot_source_path', lambda path, **kw: path)
    monkeypatch.setattr(reader, 'center_crop', lambda image, **kwargs: image)
    scene = dict(source_video_path=evidence['source_video'], video_frame_count=503,
                 height=2, width=2, crop_box=[0, 0, 2, 2])
    pixels, timing = reader.read_reference_frames(scene, [496, 495, 496, 502, 0], video_decode_num_threads=1, media=ANNOTATION_MEDIA)
    assert pixels[:, 0, 0, 0].tolist() == [240, 239, 240, 246, 0]
    assert seeks == [timestamps[i] - 1 for i in [0, 495, 496, 502]]
    assert timing['reference_output_frames'] == 4
