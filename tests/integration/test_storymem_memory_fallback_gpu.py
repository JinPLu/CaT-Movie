"""真实 WanM2V 上的空记忆退法：前面一张关键帧都没选出时，前一镜末帧作唯一记忆帧走 M2V。

替身模型测不出 M2V 的调用约束（10-30 第一次退法在真模型上才暴露 ``size`` 参数不被接受），所以这里加载真实权重，
跑一镜 5 帧、2 步的最短片段。需要一张 CUDA 卡和 StoryMem 上游与权重：``MSPC_STORYMEM_CONFIG`` 指向
某次 StoryMem 运行的 ``inputs/config.json``（取其中 ``official_storymem_14b`` 一项）。
"""
import json
import os
from pathlib import Path

import pytest
import torch

from mspc.comparison_methods import renderer

CONFIG = os.environ.get('MSPC_STORYMEM_CONFIG')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not CONFIG,
                                reason='needs one CUDA device and MSPC_STORYMEM_CONFIG with real StoryMem weights')


def test_empty_memory_bank_uses_previous_last_frame_on_real_m2v(tmp_path, monkeypatch):
    from PIL import Image

    settings = json.loads(Path(CONFIG).read_text())['methods']['official_storymem_14b']
    settings = {**settings, 'sampling': {**settings['sampling'], 'num_inference_steps': 2}}
    renderer.select_runtime(settings)
    runtime = renderer.StoryMemRuntime(settings, 16, 'm2v')
    monkeypatch.setattr(runtime, '_save_keyframes', lambda target: None)  # 选帧不在本测试范围
    segments = tmp_path / 'story' / 'segments'
    segments.mkdir(parents=True)
    Image.new('RGB', (832, 480), (120, 90, 60)).save(segments / 'last_frame.jpg')
    previous = {'start_shot': 0, 'end_shot': 1, 'output': str(segments / '0001.mp4'), 'keyframes': []}
    segment = {'start_shot': 1, 'end_shot': 2, 'num_frames': 5, 'prompts': ['A man walks through a warehouse.']}
    unit = runtime.render_shot({'video_path': str(tmp_path / 'story' / 'video.mp4'), 'seed': 42}, segment, [previous])
    assert unit['memory_fallback'] == renderer.MEMORY_FALLBACK
    assert unit['memory_files'] == [str(segments / '0001_last_frame.jpg')]
    assert (segments / '0001_last_frame.jpg').is_file() and (segments / '0002.mp4').stat().st_size > 0
