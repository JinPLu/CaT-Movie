"""H1 首200真实行转换之后，从同次CineDance输入抽出的回归固件。"""
import copy
import json
from pathlib import Path
import pytest
from mspc.data.shot_gate import revised_scene_window

FIXTURE=json.loads((Path(__file__).parent/'fixtures/shot_gate_scene_window.json').read_text())


@pytest.mark.parametrize('key,value',[('window_id','wrong'),('status','rejected'),('coordinate','decord')])
def test_mismatched_gate_rejected(key,value):
    gate=copy.deepcopy(FIXTURE['gate']);gate[key]=value
    with pytest.raises(ValueError):revised_scene_window(FIXTURE['window'],gate)


def test_old_revised_input_cannot_be_revised_again():
    row=copy.deepcopy(FIXTURE['window']);row['shots'][0]['start_frame']+=1
    with pytest.raises(ValueError,match='original source interval'):
        revised_scene_window(row,FIXTURE['gate'])


@pytest.mark.parametrize('key',['width','height','video_frame_count'])
def test_missing_reader_field_fails(key):
    row=copy.deepcopy(FIXTURE['window']);del row[key]
    with pytest.raises(KeyError):revised_scene_window(row,FIXTURE['gate'])
