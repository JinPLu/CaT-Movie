"""当前分镜表规划器的答复结构验收；事实效果指标尚未定义，不沿用事件日志评分。"""
import json
from collections import Counter
from typing import Mapping, Sequence

from mspc.core.wan22_contract import STORYBOARD_FPS
from mspc.data.planner_dataset import validate_planner_example
from mspc.data.storyboard import SCHEMA, compile_storyboard


def assess_answer(answer: Mapping | None, *, timing: Mapping | None, stated: Mapping | None) -> dict:
    """``stated`` 是逐镜档请求写明的值（``storyboard.parse_shot_script``），编译时拷回；粗档与训练样本核对给 None。"""
    if answer is None or answer.get('status') in ('transport_failed', 'billing_unknown'):
        return {'output_schema': SCHEMA, 'structure_status': 'no_answer', 'storyboard': None}
    if answer.get('status') == 'incomplete' or answer.get('finish_reason') == 'length':
        return {'output_schema': SCHEMA, 'structure_status': 'incomplete', 'storyboard': None}
    try:
        payload = json.loads(answer['raw_content'])
    except (json.JSONDecodeError, TypeError) as exc:
        return {'output_schema': SCHEMA, 'structure_status': 'json_error', 'storyboard': None, 'error': str(exc)}
    try:
        if timing is not None and (type(timing['fps']) is not int or timing['fps'] not in (16, 24)):
            raise ValueError('planner production media requires 16 or 24 fps')
        board, compilation = compile_storyboard(payload,
            target_seconds=None if timing is None else timing['target_seconds'],
            # 公开Storyboard保持统一时间坐标；制作要求的媒体fps只在render_story派生执行帧数。
            # pacing建议按比例分配，无需把规划原答先取整换帧；总秒数与显式切点原样消费。
            fps=None if timing is None else STORYBOARD_FPS,
            cut_times_seconds=None if timing is None else timing['expected_cut_times_seconds'],
            stated=stated)
        if timing is not None:
            compilation['requested_fps'] = timing['fps']
    except (ValueError, TypeError) as exc:
        return {'output_schema': SCHEMA, 'structure_status': 'schema_error', 'raw_storyboard': payload,
                'storyboard': None, 'error': str(exc)}
    return {'output_schema': SCHEMA, 'structure_status': 'valid', 'raw_storyboard': payload,
            'compilation': compilation, 'storyboard': board}


def evaluate_structure(samples: Sequence[Mapping], answers: Sequence[Mapping]) -> dict:
    """保持提供的样本分母；缺答和撞长度单列，结构通过不等价于语义正确。"""
    by_id = {answer['case_id']: answer for answer in answers}
    ids = [sample['sample_id'] for sample in samples]
    if len(by_id) != len(answers) or len(set(ids)) != len(ids):
        raise ValueError('duplicate sample or answer id')
    if set(by_id) - set(ids):
        raise ValueError('answers outside supplied samples')
    rows = []
    for sample in samples:
        validate_planner_example(sample)
        rows.append({'sample_id': sample['sample_id'], 'tier': sample['tier'],
                     **assess_answer(by_id.get(sample['sample_id']), timing=None, stated=None)})
    return {'rows': rows, 'summary': {'schema': SCHEMA, 'samples': len(samples),
            'structure_counts': dict(Counter(row['structure_status'] for row in rows))}}
