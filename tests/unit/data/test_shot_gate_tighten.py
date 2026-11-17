"""G3七个真实漏窗：四个补抓、三个叠化仍漏；逐镜原生分数。"""
import json
from pathlib import Path
import numpy as np
import pytest
from mspc.data.shot_gate import RULE_VERSION, gate_shot

ROWS=[json.loads(x) for x in (Path(__file__).resolve().parents[2]/'fixtures/shot_gate_tighten/real_seven.jsonl').read_text().splitlines()]
RULE=dict(boundary_threshold=.5,single_threshold=.25,content_threshold=25,edge_seconds=.5,boundary_guard_frames=2)

@pytest.mark.parametrize('w',ROWS,ids=lambda w:w['name'])
def test_real_g3_union(w):
    got=[]
    for s in w['shots']:
        table=dict(s['metadata']);a=s['score_start']
        for signal in ('single','content'):
            table[signal]=np.zeros(table['frame_count'],dtype=np.float32)
            table[signal][a:a+len(s[signal])]=s[signal]
        result=gate_shot({'video':s['video'],'segment':s['segment']},table,coordinate=s['coordinate'],**RULE)
        assert result['rule_version']==RULE_VERSION
        assert result['revised_segment']==s['revised_segment']
        got.append(result['status']=='rejected')
    assert any(got)==w['expected_rejected']
