"""真实 CineDance 源前103帧：滑窗分批不得改变上下文及帧索引。"""
from pathlib import Path

import numpy as np
import pytest

from mspc.data.transnet_gpu_scoring import window_batches

FIXTURE = Path(__file__).resolve().parents[2] / 'fixtures/transnet_gpu/real_frames.npz'


@pytest.mark.parametrize('length', [1, 49, 50, 51, 100, 103])
def test_real_segment_padding_and_batching(length):
    frames = np.load(FIXTURE)['frames'][:length]
    expected = np.concatenate((np.repeat(frames[:1],25,axis=0), frames,
                               np.repeat(frames[-1:],25+(50-length%50)%50,axis=0)))
    one = np.concatenate(list(window_batches(frames,1)))
    many = np.concatenate(list(window_batches(frames,16)))
    np.testing.assert_array_equal(one,many)
    for i,window in enumerate(many):
        np.testing.assert_array_equal(window,expected[i*50:i*50+100])
    restored = many[:,25:75].reshape(-1,27,48,3)[:length]
    np.testing.assert_array_equal(restored,frames)
