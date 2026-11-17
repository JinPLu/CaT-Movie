"""只把逐镜用户请求编译为对比方法的分镜输入，不读取源标注状态。

共同计划只提供 ``frame_count``；登记、机位和动作来自 ``inputs.shots.text``，
在场与地点来自已完成的 request_presence。未知地点保留 null，不从标注补全。
指称对齐只插入已唯一判定的编号，保留原名词短语；其余在场编号用独立句声明。
这保证原生标签有入口，不把仍有歧义的动作归属伪装成已经解析。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from mspc.data.shot_text import action_reference_candidates, shot_script_text
from mspc.data.storyboard import FRAME_COUNTS, entity_references, parse_shot_script, validate_storyboard
from mspc.evaluation.request_presence import load_run, validate as validate_presence


def explicit_request_holdings(board: dict, index: int) -> tuple[dict[str, list[str]], list[dict]]:
    """只提取本句明确的 holds/holding/carries/carrying，不从取得或交接动作推开态。

    主体必须是原文或唯一指称对齐产生的 Character ID；宾语必须是明确 ID 或本镜
    唯一匹配的登记物体名词。who 仅绑定紧前的明确 Character；未解析指称留在原动作。
    """
    shot = board['shots'][index]
    text = shot['action'].split(' The shot includes ', 1)[0]
    present = {entry['name'] for entry in shot['present']}
    objects = {entry['name']: entry['appearance'] for entry in board['objects'] if entry['name'] in present}
    holdings: dict[str, list[str]] = {}
    evidence = []
    clauses = re.split(r'[.!?;]|\bwhile\s+(?=Character\s+\d+\b)|\band\s+(?=Character\s+\d+\b)', text)
    for clause in clauses:
        for verb in re.finditer(r'\b(?:holds?|holding|carries|carrying)\b(?:\s+up\b)?', clause, re.I):
            before = clause[:verb.start()]
            # Negated holding is not a positive relation; do not turn “not holding” into one.
            if re.search(r'\b(?:not|never|no longer)\b[^,;]*$', before, re.I):
                continue
            subjects = list(re.finditer(r'\bCharacter [1-9]\d*\b', before))
            if not subjects:
                continue
            relative = re.search(r'(Character [1-9]\d*),?\s+who\s+(?:is\s+)?$', before)
            immediate = re.search(r'(Character [1-9]\d*)\s+(?:is\s+)?$', before)
            if relative or immediate:
                subject = (relative or immediate).group(1)
            elif len(subjects) == 1 or re.search(r'\bwhile\s*$', before, re.I):
                subject = subjects[0].group()
            else:
                # Multiple mentions without an explicit grammatical attachment are left in action.
                continue
            if subject not in present:
                continue
            phrase = re.split(r'\b(?:while|as|but|to Character|in front of|in (?:his|her|their) hands|and (?=looks|talks|speaks|walks|turns|prepares|opens|stands|sits|points|reaches|raises|lowers))\b',
                              clause[verb.end():], maxsplit=1, flags=re.I)[0].strip(' ,')
            ids = re.findall(r'\bObject [1-9]\d*\b', phrase)
            if not ids:
                matches = []
                for name, appearance in objects.items():
                    nominal = re.split(r'\b(?:with|on|in|near|beside|behind|standing|sitting|lying|hanging)\b',
                                       appearance, maxsplit=1, flags=re.I)[0]
                    heads = [re.findall(r'[A-Za-z]+', part)[-1].lower()
                             for part in re.split(r'\bor\b', nominal) if re.findall(r'[A-Za-z]+', part)]
                    if any(re.search(r'\b' + re.escape(head) + r'\b', phrase, re.I) for head in heads):
                        matches.append(name)
                ids = matches if len(matches) == 1 else []
            if not ids or not set(ids) <= present:
                continue
            values = holdings.setdefault(subject, [])
            values.extend(name for name in ids if name not in values)
            evidence.append({'name': subject, 'holding': ids, 'verb': verb.group(), 'phrase': phrase,
                             'clause': clause.strip()})
    return holdings, evidence


def align_action_references(board: dict, index: int) -> dict[str, Any]:
    """唯一在场指称关联为ID同位语，逐字保留原动作和被关联短语。

    候选判断复用 shot_text 的语义 owner，不另建模糊匹配或按单人在场猜身份。
    所有格短语保持原文，避免插入同位语后破坏所有格语法。
    ``presence_sentence`` 只重复已授权的本镜在场事实，不增加姿态、状态或动作。
    """
    shot = board['shots'][index]
    original = shot['action']
    replacements, unresolved, pieces, cursor = [], [], [], 0
    for candidate in action_reference_candidates(board, index):
        names = candidate['matched_names']
        if names is None or original[candidate['end']:].startswith(("'s", '’s')):
            unresolved.append(candidate)
            continue
        if candidate['start'] < cursor:
            raise ValueError('overlapping action reference spans')
        label = ' and '.join(names)
        following = original[candidate['end']:]
        close = '' if not following.strip() or following.startswith((',', '.', '!', '?', ';', ':')) else ','
        replacement = f"{label}, {candidate['phrase']}{close}"
        pieces.extend((original[cursor:candidate['start']], replacement))
        cursor = candidate['end']
        replacements.append({**candidate, 'names': names, 'replacement': replacement})
    pieces.append(original[cursor:])
    aligned = ''.join(pieces)
    referenced = set(entity_references(aligned))
    missing = [entry['name'] for entry in shot['present'] if entry['name'] not in referenced]
    presence_sentence = ('The shot includes ' + ', '.join(missing) + '.') if missing else ''
    action = aligned + ((' ' if aligned.endswith(('.', '!', '?')) else '. ') + presence_sentence
                        if presence_sentence else '')
    return {'action': action, 'original_action': original, 'replacements': replacements,
            'unresolved': unresolved, 'presence_sentence': presence_sentence}


def build_request_storyboard(request: dict, presence: dict, frame_counts: list[int]) -> tuple[dict, list[dict]]:
    """单故事请求→分镜；不接受带状态标注的board作为参数。"""
    text = request['inputs']['shots']['text']
    stated = parse_shot_script(text)
    validate_presence(presence, text)
    if len(stated['shots']) != len(frame_counts) or len(presence['shots']) != len(frame_counts):
        raise ValueError(f"{request['story_id']}: request, presence and duration counts differ")
    board: dict[str, Any] = {'characters': [], 'objects': [], 'places': [], 'shots': []}
    for name, fields in stated['characters'].items():
        entry = {'name': name, 'kind': fields['kind'], 'appearance': fields['appearance']}
        if fields['outfit'] is not None:
            entry['outfit'] = fields['outfit']
        board['characters'].append(entry)
    for kind in ('objects', 'places'):
        board[kind] = [{'name': name, 'appearance': appearance}
                       for name, appearance in stated[kind].items()]
    unique_place = next(iter(stated['places'])) if len(stated['places']) == 1 else None
    for index, (shot, present, frames) in enumerate(zip(stated['shots'], presence['shots'], frame_counts)):
        if type(frames) is not int or frames not in FRAME_COUNTS:
            raise ValueError(f"{request['story_id']}: invalid common frame count {frames!r}")
        board['shots'].append({'shot': index + 1, 'camera': dict(shot['camera']),
                               'frame_count': frames, 'place': unique_place or present['location'],
                               'present': [{'name': name} for name in present['on_screen']],
                               'action': shot['action'], 'changes': []})
    # Parse→serialize verifies all registry/camera/action text before inserting any identity labels.
    if shot_script_text(board) != text:
        raise ValueError(f"{request['story_id']}: request text does not round-trip")
    alignments = []
    for index in range(len(board['shots'])):
        result = align_action_references(board, index)
        board['shots'][index]['action'] = result['action']
        holdings, holding_evidence = explicit_request_holdings(board, index)
        for entry in board['shots'][index]['present']:
            if entry['name'] in holdings:
                entry['holding'] = holdings[entry['name']]
        alignments.append({'story_id': request['story_id'], 'shot': index + 1, **result,
                           'explicit_holding_evidence': holding_evidence})
    # Do not invent a Location N to appease a schema that does not represent unknown places.
    if all(shot['place'] is not None for shot in board['shots']):
        board = validate_storyboard(board)
    return board, alignments


def build(requests: dict, common_plans: dict, presences: dict[str, dict]) -> tuple[dict, list[dict]]:
    """共同计划只投影出时长，在任何请求解析前丢弃它的其余字段。"""
    durations = {row['story_id']: [shot['frame_count'] for shot in row['storyboard']['shots']]
                 for row in common_plans['stories']}
    ids = [row['story_id'] for row in requests['stories']]
    if (len(ids) != len(set(ids)) or len(durations) != len(common_plans['stories'])
            or set(ids) != set(durations) or set(ids) != set(presences)):
        raise ValueError('request, common-duration and presence story membership differs')
    stories, alignments = [], []
    for request in requests['stories']:
        sid = request['story_id']
        board, rows = build_request_storyboard(request, presences[sid], durations[sid])
        stories.append({'story_id': sid, 'split': request['split'], 'fps': request['fps'],
                        'inputs': {'shots': request['inputs']['shots']},
                        'target_seconds': request['target_seconds'], 'storyboard': board})
        alignments.extend(rows)
    return {'evaluation_scope': 'renderer', 'storyboard_version': 'request_storyboards_v4',
            'authoring': {'information_condition': 'request_only',
                          'execution_route': 'renderer表示直接冻结本表，禁止再调用Planner；系统比较的信息条件仍为request_only',
                          'rule': '登记、动作与显式机位只读逐镜请求；在场／地点只读request_presence；共同计划只读帧数；无标注状态或末态',
                          'unknown_location': 'null；不得从源片标注补全',
                          'reference_alignment': '复用shot_text唯一指称候选；保留原短语；未解析动作不猜归属，在场编号独立声明'},
            'stories': stories}, alignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--common-plans', type=Path, required=True)
    parser.add_argument('--presence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--alignment-output', type=Path, required=True)
    args = parser.parse_args()
    result, alignments = build(json.loads(args.requests.read_text()), json.loads(args.common_plans.read_text()),
                               load_run(args.presence))
    result['authoring'].update(requests=str(args.requests), common_durations=str(args.common_plans),
                               request_presence=str(args.presence))
    for path in (args.output, args.alignment_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    args.alignment_output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in alignments))
    print(json.dumps({'stories': len(result['stories']), 'shots': len(alignments),
                      'unknown_location_shots': sum(s['place'] is None for r in result['stories'] for s in r['storyboard']['shots']),
                      'aligned_mentions': sum(len(row['replacements']) for row in alignments),
                      'unresolved_mentions': sum(len(row['unresolved']) for row in alignments)}))


if __name__ == '__main__':
    main()
