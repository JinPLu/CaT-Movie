"""来自 G2 已就绪实窗与四个已亲看残留窗的保护帧回归。"""
import json
from pathlib import Path
import numpy as np
import pytest
from mspc.data.shot_gate import gate_shot, gate_window

ROWS=[json.loads(line) for line in (Path(__file__).resolve().parents[2]/'fixtures/shot_gate_guard/real_shots.jsonl').read_text().splitlines()]
RULE=dict(boundary_threshold=.5,single_threshold=.25,content_threshold=25,edge_seconds=.5,boundary_guard_frames=2)


def table(row):
    t=dict(row['metadata'])
    for name in ('single','content'):
        t[name]=np.zeros(t['frame_count'],dtype=np.float32)
        a=row['score_start'];t[name][a:a+len(row[name])]=row[name]
    return t


@pytest.mark.parametrize('row',ROWS,ids=lambda r:f"w{r['window_number']}-s{r['shot_index']}-{r['shot']['segment'][0]}")
def test_real_shot_guard_matches_source_intervals(row):
    result=gate_shot(row['shot'],table(row),coordinate=row['metadata']['coordinate'],**RULE)
    for key in ('segment', 'boundary_revised_segment', 'boundary_guard_frames', 'revised_segment', 'boundary_cuts', 'coordinate'):
        assert result[key] == row['expected'][key]
    assert all(reason in result['reasons'] for reason in row['expected']['reasons'])
    a,b=result['boundary_revised_segment']
    assert result['revised_segment']==[a+2,b-2]
    assert result['boundary_guard_frames']==2


def test_four_known_residuals_are_outside_guarded_half_open_segments():
    for number,shot,dirty in [(5,3,[1697]),(6,1,[232,233]),(15,2,[664]),(18,1,[79])]:
        row=next(r for r in ROWS if r['window_number']==number and r['shot_index']==shot)
        a,b=gate_shot(row['shot'],table(row),coordinate=row['metadata']['coordinate'],**RULE)['revised_segment']
        assert all(not a<=f<b for f in dirty)


def test_real_short_shot_rejects_whole_window():
    row=next(r for r in ROWS if any(x['signal']=='duration' for x in r['expected']['reasons']))
    w=dict(window_id=row['window_id'],family='scene_window_v1',shots=[row['shot']])
    result=gate_window(w,{row['shot']['video']:table(row)},**RULE)
    assert result['status']=='rejected'
    assert result['reasons'][-1]['reason']=='guarded_shot_shorter_than_0.5_seconds'
    assert result['boundary_guard_frames']==2


@pytest.mark.parametrize('value',[-1,True,2.0,'2',None])
def test_invalid_guard_raises(value):
    row=ROWS[0]
    with pytest.raises(ValueError,match='boundary_guard_frames'):
        gate_shot(row['shot'],table(row),coordinate=row['metadata']['coordinate'],**{**RULE,'boundary_guard_frames':value})


def test_exact_half_second_is_kept_and_one_frame_less_rejected():
    scores=dict(video='duration',coordinate='decord',fps=24.,frame_count=17,single=np.zeros(17),content=np.zeros(17))
    assert gate_shot(dict(video='duration',segment=[0,16]),scores,coordinate='decord',**RULE)['status']=='kept'
    result=gate_shot(dict(video='duration',segment=[0,15]),scores,coordinate='decord',**RULE)
    assert result['status']=='rejected' and result['reasons'][0]['frames']==11
