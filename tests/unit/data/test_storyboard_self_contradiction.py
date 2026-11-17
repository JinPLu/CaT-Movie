"""规划原答两处自相矛盾时编译器不替规划器选边：开镜值（在场条目对变化的 before）。相对位置规划器不再写（11-13），
原答里的人物关系整条丢掉，在场条目与动作句里写的位置照留。夹具是 11-12 判官在编译后文字里判出与请求矛盾的真实原答。"""
import json
from pathlib import Path

from mspc.data.shot_text import shot_text
from mspc.data.storyboard import PLANNER_RELATIONS_DROPPED, compile_storyboard

ROWS = json.loads((Path(__file__).resolve().parents[2] / 'fixtures/planner_storyboard/compiler_self_contradiction_real.json')
                  .read_text())['rows']


def compiled(story):
    board, report = compile_storyboard(ROWS[story]['raw_storyboard'], target_seconds=None, fps=None,
                                       cut_times_seconds=None, stated=None)
    return board, report, ROWS[story]['shot']


def test_invented_change_before_does_not_become_the_opening():
    board, report, shot = compiled('cinedance-016627-b0003-shot01')
    withheld = {(fix['name'], fix['before']) for fix in report['opening_withheld'] if fix['shot'] == shot}
    assert withheld == {('Character 1', 'standing'), ('Character 2', 'walking')}
    text = shot_text(board, shot - 1)
    assert 'Character 1 stands' not in text and 'Character 2 is walking' not in text
    assert 'By the end of the shot, Character 1 is sitting; Character 2 is driving.' in text


def test_a_written_relation_is_not_planned_and_the_written_position_stays():
    board, report, shot = compiled('cinedance-014847-b0000-shot01')
    dropped = [d for d in report['dropped_fields'] if d['path'] == f'shots[{shot - 1}].relations']
    assert [d['reason'] for d in dropped] == [PLANNER_RELATIONS_DROPPED] and 'in_front_of' in json.dumps(dropped)
    text = shot_text(board, shot - 1)
    assert 'behind Character 2' in text and 'in front of Character 2' not in text
