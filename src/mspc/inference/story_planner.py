"""公共/SFT Qwen共用的文本规划调用；一次候选、原答先落盘。只有镜数、帧预算这类有真值的结构错误带具体错误回灌一次，
内容冲突不重问（11-10 规划器复查：无外部真值的冲突重问看图改对 9、改错 65）。"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time

from mspc.core.wan22_contract import STORYBOARD_FPS, STORYBOARD_MAX_SHOT_FRAMES
from mspc.data.planner_dataset import PLANNER_CONTRACT_VERSION, canonicalize_script, planner_messages
from mspc.data.storyboard import SCHEMA, StoryboardFields, parse_shot_script
from mspc.evaluation.storyboard_planner import assess_answer
from mspc.evaluation.benchmark import write_json

PLANNER_TIERS = {'synopsis': 'summary', 'paragraph': 'paragraph', 'shots': 'shot'}


#: Qwen3.8 聊天模板认的思考档位（模板里 low、xhigh 各注入一句思考长短的说明，medium 不注入，未给时默认 xhigh）；
#: 走 chat_template_kwargs 而非请求顶层：vLLM 0.19 顶层字段只收 none/low/medium/high，不收 xhigh。
REASONING_EFFORTS = ('low', 'medium', 'xhigh')


def completion_request(task: dict, planner: dict, sampling: dict) -> dict:
    """思考与否、思考档位都由 sampling 显式给出，非思考时档位为 None（模型卡两套官方采样：非思考
    0.7／0.8／20／0／presence 1.5；思考 1.0／0.95／20／0／presence 0）。"""
    if type(sampling['enable_thinking']) is not bool:
        raise TypeError('enable_thinking must be a boolean')
    effort = sampling['reasoning_effort']
    if (effort is None) == sampling['enable_thinking'] or (effort is not None and effort not in REASONING_EFFORTS):
        raise ValueError(f'reasoning_effort must be None without thinking and one of {REASONING_EFFORTS} with it')
    return {'model': planner['model'],
            'messages': planner_messages({'tier': PLANNER_TIERS[task['input_tier']], 'script': task['story_request']}),
            'temperature': sampling['temperature'], 'top_p': sampling['top_p'],
            'presence_penalty': sampling['presence_penalty'], 'max_tokens': sampling['max_tokens'],
            'seed': task['seed'], 'n': 1, 'stream': False,
            'response_format': {'type': 'json_schema', 'json_schema': {
                'name': SCHEMA, 'schema': StoryboardFields.model_json_schema()}},
            'extra_body': {'top_k': sampling['top_k'], 'min_p': sampling['min_p'],
                           'repetition_penalty': sampling['repetition_penalty'],
                           'chat_template_kwargs': {'enable_thinking': sampling['enable_thinking'],
                                                    **({'reasoning_effort': effort} if effort is not None else {})}}}


def stated_values(task: dict) -> dict | None:
    """解析逐镜档明示事实；登记与机位编译时拷回，动作保留规划器受约束扩写。粗档给 None。"""
    return parse_shot_script(canonicalize_script(task['story_request'])[0]) if task['input_tier'] == 'shots' else None


def compile_record(record: dict, task: dict) -> None:
    """规划器原答（``raw_content``、``finish_reason``）→ 编译与镜数核对，结果写回 ``record``。重新编译已有原答也走这里。
    逐镜档拷回请求写明的登记、地点与机位，动作保留规划器受约束扩写（``storyboard.copy_stated_values``）。"""
    record['status'] = 'completed' if record['finish_reason'] == 'stop' else 'incomplete'
    record.update(assess_answer(record, timing=task, stated=stated_values(task)))
    if record['structure_status'] != 'valid':
        record['status'] = 'planning_failed'
    else:
        shots = len(record['storyboard']['shots'])
        required = task['expected_shot_count']
        if required is not None and shots != required:
            record.update(status='planning_failed', error=f'planned {shots} shots; user required {required}')


#: 回灌前把第一次的原答与编译结果整体移进 ``first_attempt``，第二次重新写这些键。
ATTEMPT_FIELDS = ('status', 'raw_response', 'elapsed_seconds', 'raw_content', 'finish_reason', 'usage', 'output_schema',
                  'structure_status', 'raw_storyboard', 'compilation', 'storyboard', 'error')


def structural_correction(record: dict, task: dict) -> str | None:
    """第一次规划失败于有真值的结构约束时，返回给规划器的一条纠正消息；其余情况返回 None。

    两种：逐镜档请求的镜数与规划镜数不符；规划镜数装不下总帧预算（每镜至多 ``STORYBOARD_MAX_SHOT_FRAMES`` 帧，按
    ``STORYBOARD_FPS`` 计）。11-10 复查梗概档 62/180 份计划因镜数太少编译失败。"""
    if record['status'] != 'planning_failed':
        return None
    board = record.get('storyboard') or record.get('raw_storyboard')
    shots = board.get('shots') if isinstance(board, dict) else None
    if not isinstance(shots, list) or not shots:
        return None
    planned, required = len(shots), task['expected_shot_count']
    if record['structure_status'] == 'valid' and required is not None and planned != required:
        return (f'The request lists exactly {required} shots, but your storyboard has {planned}. Return the complete '
                f'storyboard again with exactly {required} shots, one per listed shot, in the order of the request.')
    frames = task['target_seconds'] * STORYBOARD_FPS
    minimum = math.ceil(frames / STORYBOARD_MAX_SHOT_FRAMES)
    if record['structure_status'] == 'schema_error' and planned < minimum:
        return (f'Your storyboard has {planned} shots, but {task["target_seconds"]} seconds at {STORYBOARD_FPS} fps is '
                f'{frames:g} frames and one shot holds at most {STORYBOARD_MAX_SHOT_FRAMES} frames, so it needs at least '
                f'{minimum} shots. Return the complete storyboard again with at least {minimum} shots, keeping '
                f'everything the request states.')
    return None


def plan_story(task: dict, planner: dict, sampling: dict, output: Path, complete) -> dict:
    """complete是服务自己的单次chat.completions.create；保存所有失败，续跑不重抽候选。有真值的结构错误
    （:func:`structural_correction`）在同一对话里带具体错误回灌一次，第一次原答留在 ``first_attempt``。"""
    request = completion_request(task, planner, sampling)
    provenance = json.loads(Path(planner['provenance']).read_text())
    record = {'status': 'request_started', 'contract_version': PLANNER_CONTRACT_VERSION,
              'request': request, 'service': provenance, 'original_story_request': task['story_request'],
              'input_name_mapping': canonicalize_script(task['story_request'])[1]}
    write_json(output, record)
    start = time.monotonic()
    try:
        raw = complete(**request)
        raw = raw.model_dump(mode='json') if hasattr(raw, 'model_dump') else raw
        record.update(raw_response=raw, elapsed_seconds=time.monotonic() - start, status='response_received')
        write_json(output, record)
        choice, = raw['choices']
        record.update(raw_content=choice['message']['content'], finish_reason=choice['finish_reason'], usage=raw['usage'])
        compile_record(record, task)
        correction = structural_correction(record, task)
        if correction is not None:
            followup = {**request, 'messages': [*request['messages'], {'role': 'assistant', 'content': record['raw_content']},
                                                {'role': 'user', 'content': correction}]}
            record['first_attempt'] = {key: record.pop(key) for key in ATTEMPT_FIELDS if key in record}
            record.update(corrective_request=followup, status='correction_requested')
            write_json(output, record)
            raw = complete(**followup)
            raw = raw.model_dump(mode='json') if hasattr(raw, 'model_dump') else raw
            record.update(raw_response=raw, elapsed_seconds=time.monotonic() - start, status='response_received')
            choice, = raw['choices']
            record.update(raw_content=choice['message']['content'], finish_reason=choice['finish_reason'], usage=raw['usage'])
            compile_record(record, task)
    except Exception as error:
        record.update(status='planning_failed' if 'raw_response' in record else 'transport_failed',
                      error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic() - start)
    write_json(output, record)
    return record


def planner_client(config: dict):
    from openai import OpenAI
    return OpenAI(base_url=config['base_url'], api_key=os.environ[config['api_key_env']],
                  timeout=config['timeout_seconds'], max_retries=0)
