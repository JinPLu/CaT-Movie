"""真实配置的3/10记忆上限：浮点和bool必须在load_config拒绝。"""
import json
from pathlib import Path

import pytest

from mspc.benchmark.generation import load_config

ROOT=Path(__file__).resolve().parents[3]


@pytest.mark.parametrize('field,value', [('fixed_first_frames',3.0),('max_frames',10.0),
    ('fixed_first_frames',True),('max_frames',True),('fixed_first_frames',0),('max_frames',-1),
    ('fixed_first_frames',10)])
def test_storymem_memory_limits_rejected_before_any_render(tmp_path, field, value):
    raw=load_config(ROOT/'configs/infer/benchmark.json')
    settings=next(s for s in raw['methods'].values() if s['family']=='storymem')
    settings['memory'][field]=value
    path=tmp_path/'config.json';path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='memory'):load_config(path)
