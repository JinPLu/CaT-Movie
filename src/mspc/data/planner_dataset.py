"""第 18 版分镜表监督：三档独立导出，抽样比例与逐镜子段由调用者显式给定。"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from mspc.core.wan22_contract import STORYBOARD_FPS, STORYBOARD_MAX_SHOT_FRAMES
from mspc.data.frame_planner import resample_frame_count
from mspc.data.storyboard import SCHEMA, NAME, from_annotation, require_visible_result_wording, validate_storyboard

PLANNER_CONTRACT_VERSION = 'storyboard_v18_action_expansion_request_v20'  # 11-13：构图只在请求写明或暗示时写；规划器不写人物关系，写明的方位留在照抄的动作句里（v19 只照抄写明方位仍另写出近乱猜的关系）
TIERS = ('summary', 'paragraph', 'shot')
EMPTY_THINK = '<think>\n\n</think>\n\n'
EXAMPLE_KEYS = {'sample_id', 'case_id', 'block_key', 'film_key', 'split', 'tier', 'script', 'shot_span', 'storyboard'}
# 规划器系统提示（按官方指南与近邻做法重写）：
# 角色与背景、步骤、三类值、逐字段说明、写法规则、一个示例。结构由约束解码保证，提示里不贴机器生成的结构定义（Qwen 结构化输出文档：提示里写清字段名、类型、可选值并给示例）。
PLANNER_SYSTEM = r'''You are the storyboard planner of a multi-shot video system. Turn the supplied tier and script into one JSON storyboard with characters, objects, places, and shots. Code then compiles the storyboard into one prompt per shot for a Wan video generator. Write concise English.

### What the code does after you
- Copies each registered appearance and outfit into the shot prompts, so write them once in the registries.
- Allocates the requested total frame budget across shots and applies the generator's frame grid.
- Adds a bare present entry for an object that appears only in a holding array.
- Carries each character's current outfit, and a holding you leave unwritten, into later shots.
- Chooses reference frames from earlier shots by who is present.
It does not infer missing actions or state changes.

### Steps
1. Register every character (including groups and animals), object, and place the request mentions as Character N, Object N, Location N. Keep supplied IDs; otherwise number each category from 1.
2. Lay out the shots. Read the frame rate and maximum shot duration from Production requirements. For tier=shot, keep every given shot in its order; do not merge, add, or remove shots. For tier=summary or paragraph, use at least the total seconds divided by the maximum shot duration, rounded up, and enough shots to show every requested event.
3. For each shot, write each entity's initial visible properties (present), the event with its visible progression (action), and the result at its end (changes).
4. Propose each shot's frame_count. Add the opening composition only when the request states or implies how the opening frame is laid out.

### Three kinds of values
- Stated by the request: copy registered appearance and outfit, places, camera instructions, and visual medium faithfully. Preserve every requested event, its performer, order, and required result. The action wording may change under the action rules below; do not weaken, omit, or replace a requested event.
- Implied by the request or by earlier shots (who is present, initial visible posture and state, possession, end-of-shot changes): write every one, consistent with earlier shots.
- Neither stated nor implied (details a coarse request leaves open, pacing): choose one concrete, plausible value that fits the request, the camera, and the action, and keep it the same across shots of the same setup. Never add plot events or subjects the request does not ask for, and do not write mood or atmosphere prose.

### Fields
characters: name "Character N"; kind person | animal | group; appearance and outfit: when the request describes the character, copy its words exactly, including any pose, accessory, or expression it mentions; otherwise write the stable look and the default clothing as lowercase noun phrases ("an adult courier", "a blue jacket"). Leave outfit out when the request gives none for a group or an animal. visibility clear | partial | distant, only when the request says how the character is seen.
objects, places: name "Object N" or "Location N"; appearance: copy the request's description exactly when it gives one; otherwise write a lowercase noun phrase ("a small brown parcel", "a sunlit workshop").
shots, one entry per shot in order:
- shot: 1, 2, 3, ...
- place: a registered location.
- picture_type: live_action | animation_2d | animation_3d_or_game | screen_or_slides, only when the request states the medium.
- camera, only the parts the request gives: shot_size extreme close-up shot | close-up shot | medium close-up shot | medium shot | medium wide shot | wide shot | extreme wide shot; angle eye-level shot | high angle shot | low angle shot | overhead shot | over-the-shoulder shot | dutch angle shot | aerial shot; movement static | pan | tilt up | tilt down | push in | pull out | tracking | handheld.
- composition: center composition | balanced composition | symmetrical composition | left-heavy composition | right-heavy composition, the words Wan's own prompt expander uses. Write it only when the request states or implies how the opening frame is laid out (a person framed alone on one side, two people placed symmetrically); otherwise leave it out.
- present: every participating character and object, each as it looks when it first becomes visible in this shot, with posture, state, holding (an array of registered entity names), and outfit only when it differs from the registered one. An entity that enters later need not be visible at the shot's start; preserve the entry or exit in action. Holding means physically holding or carrying the entity, including a person or animal, with hands, mouth, or body. Being near, pointing at, entering, sitting in, or riding an entity does not mean holding it; describe those interactions in action or posture.
- action: concise, ordered sentences describing this shot's event. Start with the performing registered subject (Character N or Object N as appropriate) and one short distinguishing feature from its registry or current outfit; do not repeat the full appearance. For an object-only or static shot, describe the requested subject or visible state without inventing a character or movement. Keep every requested action and its performer, and keep where the request places people (on the left, behind, in the background) in these sentences. When useful, infer a short, physically plausible progression that makes the requested action visible, without adding a new plot event, subject, or stronger ending. Inferred intermediate steps are a proposed execution, not observations of an unseen source video. Use then or finally only when the actions are sequential; preserve simultaneous actions as simultaneous.
- changes: the entity, what (posture | state | outfit | holding), before (equal to the opening), and after (the visible ending).
- ending_sides: characters who end the shot on another side of the frame, with left | center | right, only when the request says so.
- frame_count: relative pacing in frames, not seconds; multiply seconds by the frame rate in Production requirements. Longer actions need more time than reactions; keep requested timing; do not solve the total.

### Writing rules
1. Use only registered names, exactly as registered, in every field and sentence. A change belongs to an existing entity; never register it as a new one.
2. The opening never contains the shot's result: a person who stands up starts seated; a cake being cut starts uncut.
3. State and posture values are lowercase complements that can follow "is", with no subject, copula, or final period: "empty-handed", "with her hair loose", "in a plain white T-shirt", "with a fresh cut on her cheek".
4. Describe a result by what is visible, not by what is gone: never "without", "removed", "no longer", "not wearing", "not holding", or "no". An action may still name the item ("takes off the headband"), and an item that stays visible keeps its name.
5. A transfer releases the object from the giver and adds it to the recipient; later shots keep the possession until another release. When a holding array ends empty, also write the visible state ("empty-handed", or "with its mouth empty" for an animal).
6. Write an outfit change once, in changes, with the full new outfit as after; put uncovered hair or skin in state.
7. Do not replay a finished event to restate its result; carry the result into later openings.
8. Expand only what helps execute the requested action. Do not invent movements for a static shot or pad the action into a fixed number of steps. Match the detail to the available time; when the request fixes timing, keep it and avoid an overlong action chain.

### Example
Input:
{"tier":"shot","script":"Production requirements: approximately 12 seconds total; 16 fps; at most 225 frames per shot (14.0625 seconds).\n\nUser request:\nCreate a multi-shot video using the following 3 shots in order.\n\nSUBJECTS\n- Character 1 (person): an adult courier, wearing a blue jacket\n- Character 2 (person): an adult woman, wearing a grey coat\n- Object 1 (object): a small brown parcel\nLOCATIONS\n- Location 1: a sunlit workshop\nSHOTS\ns01: Character 1 holds Object 1 and greets Character 2.\ns02: Character 1 hands Object 1 to Character 2.\ns03: Character 2 carries Object 1 to the window while Character 1 waits by the door."}
Output:
{"characters":[{"name":"Character 1","kind":"person","appearance":"an adult courier","outfit":"a blue jacket"},{"name":"Character 2","kind":"person","appearance":"an adult woman","outfit":"a grey coat"}],"objects":[{"name":"Object 1","appearance":"a small brown parcel"}],"places":[{"name":"Location 1","appearance":"a sunlit workshop"}],"shots":[{"shot":1,"place":"Location 1","present":[{"name":"Character 1","posture":"standing","holding":["Object 1"]},{"name":"Character 2","posture":"standing","holding":[]}],"action":"Character 1, the courier in the blue jacket, greets Character 2, the woman in the grey coat, while holding Object 1.","changes":[],"frame_count":64},{"shot":2,"place":"Location 1","present":[{"name":"Character 1","posture":"standing","holding":["Object 1"]},{"name":"Character 2","posture":"standing","holding":[]}],"action":"Character 1, the courier in the blue jacket, extends Object 1 toward Character 2, the woman in the grey coat. Character 2 takes Object 1, and Character 1 lets go.","changes":[{"name":"Character 1","what":"holding","before":["Object 1"],"after":[]},{"name":"Character 1","what":"state","after":"empty-handed"},{"name":"Character 2","what":"holding","before":[],"after":["Object 1"]}],"frame_count":64},{"shot":3,"place":"Location 1","present":[{"name":"Character 1","posture":"standing by the doorway","state":"empty-handed","holding":[]},{"name":"Character 2","posture":"standing","holding":["Object 1"]}],"action":"Character 2, the woman in the grey coat, carries Object 1 to the window while Character 1, the courier in the blue jacket, waits by the door.","changes":[{"name":"Character 2","what":"posture","before":"standing","after":"standing at the window"}],"frame_count":64}]}
The parcel changes hands once and remains with Character 2. Extending, taking, and letting go are a plausible inferred execution of the requested handover, not observations of an unseen source video. The compiler keeps this expanded action, adds the referenced parcel's bare present entry, and resolves the total frame budget. Do not copy the example's characters or story into a different request.

Return only the JSON object.'''


def production_script(*, script: str, target_seconds: float, fps: int) -> str:
    """训练导出和评测请求共用的制作要求；只公开总预算，不公开逐镜时长。"""
    if not isinstance(script, str) or not script.strip():
        raise ValueError('script must be a nonempty string')
    if type(target_seconds) not in (int, float) or not math.isfinite(target_seconds) or target_seconds <= 0:
        raise ValueError('target_seconds must be positive and finite')
    if type(fps) is not int or fps not in (16, 24):
        raise ValueError('the storyboard production contract requires 16 or 24 fps')
    maximum = resample_frame_count(STORYBOARD_MAX_SHOT_FRAMES, source_fps=STORYBOARD_FPS,
                                  target_fps=fps, temporal_stride=4)
    return (f'Production requirements: approximately {target_seconds:g} seconds total; {fps} fps; '
            f'at most {maximum} frames per shot ({maximum / fps:g} seconds).'
            f'\n\nUser request:\n{script}')


def canonicalize_script(script: str) -> tuple[str, dict[str, str]]:
    """只读本档公开的带类别登记行；自然语言指代仍由模型理解。"""
    declaration = re.compile(r'(?m)^\s*- (Subject [1-9]\d*|Location [1-9]\d*)(?: \(([^)]+)\))?:')
    matches = list(declaration.finditer(script))
    starts = re.findall(r'(?m)^\s*- (?:Subject|Location)\b[^\n]*', script)
    if len(starts) != len(matches):
        raise ValueError('malformed public entity declaration')
    mapping, counts = {}, {'Character': 0, 'Object': 0, 'Location': 0}
    for match in matches:
        source, kind = match.groups()
        if source in mapping:
            raise ValueError(f'duplicate public entity declaration: {source}')
        if source.startswith('Location '):
            if kind is not None:
                raise ValueError(f'location declaration has an unexpected kind: {source}')
            prefix = 'Location'
        elif kind in ('person', 'animal', 'group'):
            prefix = 'Character'
        elif kind == 'object':
            prefix = 'Object'
        else:
            raise ValueError(f'public Subject declaration needs an explicit entity kind: {source}')
        counts[prefix] += 1
        mapping[source] = f'{prefix} {counts[prefix]}'
    references = {m.group() for m in NAME.finditer(script)}
    for source, target in mapping.items():
        if source != target and target in references and target not in mapping:
            raise ValueError(f'ambiguous public entity aliases: {source} and {target}')
    if mapping:
        unknown = {name for name in references if name.startswith('Subject ')} - mapping.keys()
        if unknown:
            raise ValueError(f'undeclared public Subject reference: {sorted(unknown)[0]}')
    return NAME.sub(lambda match: mapping.get(match.group(), match.group()), script), mapping


def planner_request(*, script: str, tier: str) -> dict:
    if not isinstance(script, str) or not script.strip():
        raise ValueError('script must be a nonempty string')
    if tier not in TIERS:
        raise ValueError(f'unknown planner tier: {tier!r}')
    return {'tier': tier, 'script': canonicalize_script(script)[0]}


def validate_planner_example(row: Mapping[str, Any]) -> None:
    if set(row) != EXAMPLE_KEYS:
        raise ValueError(f'planner example keys differ: {sorted(set(row) ^ EXAMPLE_KEYS)}')
    for key in ('sample_id', 'case_id', 'block_key', 'film_key'):
        if not isinstance(row[key], str) or not row[key]:
            raise ValueError(f'{key} must be a nonempty string')
    if row['split'] not in ('train', 'val'):
        raise ValueError('split must be train or val')
    planner_request(script=row['script'], tier=row['tier'])
    production = re.fullmatch(
        r'Production requirements: approximately (\S+) seconds total; (\d+) fps; '
        r'at most \d+ frames per shot \(\S+ seconds\)\.\n\nUser request:\n(.+)',
        row['script'], re.DOTALL,
    )
    if production is None:
        raise ValueError(f'{PLANNER_CONTRACT_VERSION} requires an exported production duration, frame rate and shot limit')
    if production_script(script=production[3], target_seconds=float(production[1]), fps=int(production[2])) != row['script']:
        raise ValueError('production requirements differ from the current export format')
    board = validate_storyboard(row['storyboard'])
    span = row['shot_span']
    if not isinstance(span, list) or len(span) != 2 or any(type(x) is not int for x in span) or not 0 <= span[0] < span[1]:
        raise ValueError('shot_span must be a zero-based half-open contiguous span')
    if span[1] - span[0] != len(board['shots']):
        raise ValueError('shot_span does not match storyboard length')


def prepare_planner_example(row: Mapping[str, Any]) -> tuple[dict, dict]:
    """冻结旧池到当前监督的显式迁移：制作预算只读原请求，持物只由分镜唯一 owner 修复，目标不写画面类型与人物关系。

    画面类型按合同只在请求明说介质时写（``Shot.picture_type`` 的字段说明与 ``PLANNER_SYSTEM``）。训练请求由标注同一遍
    写出、不陈述介质（请求里的 animated tomato、talks animatedly、slides 说的是画面内容），而标注逐镜都带画面类型，
    11-09 修复监督 SFT 因此 30/30 计划逐镜写画面类型、公共 Qwen 0/285。目标去掉这一字段；分镜行本身的其他用途不受影响，
    渲染器的画风句仍由标注给。人物关系不在规划器合同里，目标同样去掉。"""
    from mspc.data.storyboard import holding_consistency_issues, reconcile_holding
    prepared = deepcopy(row)
    production = re.fullmatch(
        r'Production requirements: approximately (\S+) seconds total; (\d+) fps\.\n\nUser request:\n(.+)',
        prepared['script'], re.DOTALL,
    )
    if production is None:
        raise ValueError('planner pool migration requires its frozen duration-and-fps request header')
    prepared['script'] = production_script(script=production[3], target_seconds=float(production[1]), fps=int(production[2]))
    board = prepared['storyboard']
    before = holding_consistency_issues(board)
    fixes = reconcile_holding(board)
    after = holding_consistency_issues(board)
    picture_types = sum(shot.pop('picture_type', None) is not None for shot in board['shots'])
    relations = sum(len(shot.pop('relations', None) or []) for shot in board['shots'])
    validate_planner_example(prepared)
    return prepared, {'before': before, 'fixes': fixes, 'after': after,
                      'production_requirements_updated': prepared['script'] != row['script'],
                      'picture_types_removed': picture_types, 'relations_removed': relations}


def planner_messages(row: Mapping[str, Any]) -> list[dict]:
    """输入是已带制作要求的脚本与档位；编码时不从目标表反推任何输入。"""
    request = planner_request(script=row['script'], tier=row['tier'])
    return [{'role': 'system', 'content': PLANNER_SYSTEM},
            {'role': 'user', 'content': json.dumps(request, ensure_ascii=False)}]


def build_planner_pools(members: Sequence[Mapping[str, Any]], shot_spans: Sequence[Mapping[str, Any]]) -> tuple[dict, list]:
    """整块梗概/段落只配全可用的块；逐镜档只配明确给出的连续子段，不跨坏镜。

    members 来自 block_members 已按片分好的 train/val。shot_spans 每行恰为 key/start/end，半开且从 0 开始。
    导出池不混合、不重采样、不替训练选择长度分布。
    """
    from mspc.data.shot_text import shot_script_text
    by_key = {member['key']: member for member in members}
    if len(by_key) != len(members):
        raise ValueError('duplicate block member')
    split_by_film = {}
    spans = {}
    for row in shot_spans:
        if set(row) != {'key', 'start', 'end'} or row['key'] not in by_key:
            raise ValueError('shot span must contain key/start/end and name an existing member')
        key, start, end = row['key'], row['start'], row['end']
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(by_key[key]['shots']):
            raise ValueError(f'{key}: invalid shot span')
        if (start, end) in spans.setdefault(key, []):
            raise ValueError(f'{key}: duplicate shot span')
        if not all(shot['usable'] for shot in by_key[key]['shots'][start:end]):
            raise ValueError(f'{key}: shot span crosses an unusable shot')
        spans[key].append((start, end))
    pools = {tier: [] for tier in TIERS}
    rejected = []
    for member in members:
        key, film, split = member['key'], member['film_key'], member['split']
        if split not in ('train', 'val') or not isinstance(film, str) or not film:
            raise ValueError(f'{key}: invalid producer split/film_key')
        if film in split_by_film and split_by_film[film] != split:
            raise ValueError(f'{film}: film appears in both train and val')
        split_by_film[film] = split
        count = len(member['shots'])
        if not count or any(type(s['usable']) is not bool for s in member['shots']):
            raise ValueError(f'{key}: missing shot usability')
        pairs = [('shot', a, b) for a, b in spans.get(key, [])]
        if all(shot['usable'] for shot in member['shots']):
            pairs += [(tier, 0, count) for tier in ('summary', 'paragraph')]
        else:
            rejected += [{'block_key': key, 'tier': tier, 'reason': 'whole_block_has_unusable_shot'} for tier in ('summary', 'paragraph')]
        for tier, start, end in pairs:
            try:
                board = from_annotation(member['annotation'], [s['frames'] for s in member['shots']], start=start, end=end)
                require_visible_result_wording(board)
                script = shot_script_text(board) if tier == 'shot' else member['annotation']['story'][tier]
                script = production_script(script=script, target_seconds=sum(s['frames'] for s in member['shots'][start:end]) / 16, fps=16)
                row = {'sample_id': f'{key}/{tier}/{start}:{end}', 'case_id': member['case_id'], 'block_key': key,
                       'film_key': film, 'split': split, 'tier': tier, 'script': script,
                       'shot_span': [start, end], 'storyboard': board}
                validate_planner_example(row)
            except (ValueError, TypeError, KeyError) as exc:
                rejected.append({'block_key': key, 'tier': tier, 'shot_span': [start, end], 'reason': 'source_contract_error', 'error': str(exc)})
            else:
                pools[tier].append(row)
    return pools, rejected


def select_long_story_rows(rows: Sequence[Mapping[str, Any]], *, min_all_shots: int, fraction_two_to_short: float,
                           fraction_one: float, seed: int) -> list[dict]:
    """训练故事长度对齐使用场景：三档行数相同（1:1:1），每档里
    镜数 ≥ ``min_all_shots`` 的长行占 ``1 - fraction_two_to_short - fraction_one``，2 至 ``min_all_shots - 1`` 镜的行与
    1 镜的行分别占 ``fraction_two_to_short`` 与 ``fraction_one``。每档总数由长行最少的那一档定：那一档长行全用，长行更多的
    档按固定种子抽到同样多（逐镜档的长子段比整块多）。行序按 sample_id 定。"""
    import random
    if not (type(min_all_shots) is int and min_all_shots >= 3 and 0 <= fraction_one and 0 <= fraction_two_to_short
            and fraction_one + fraction_two_to_short < 1):
        raise ValueError('invalid long-story selection shares')
    length = lambda r: r['shot_span'][1] - r['shot_span'][0]
    by_tier = {tier: sorted((r for r in rows if r['tier'] == tier), key=lambda r: r['sample_id']) for tier in TIERS}
    want_long = min(sum(length(r) >= min_all_shots for r in tier_rows) for tier_rows in by_tier.values())
    total = want_long / (1 - fraction_two_to_short - fraction_one)
    want_middle, want_single = round(total * fraction_two_to_short), round(total * fraction_one)
    selected = []
    for tier, tier_rows in by_tier.items():
        long = [r for r in tier_rows if length(r) >= min_all_shots]
        middle = [r for r in tier_rows if 2 <= length(r) < min_all_shots]
        single = [r for r in tier_rows if length(r) == 1]
        if want_middle > len(middle) or want_single > len(single):
            raise ValueError(f'{tier}: not enough short rows for the requested shares')
        rng = random.Random(f'{seed}:{tier}')
        selected += rng.sample(long, want_long) + rng.sample(middle, want_middle) + rng.sample(single, want_single)
    return sorted(selected, key=lambda r: r['sample_id'])


def field_supervision_tiers(row: Mapping[str, Any]) -> dict[tuple, str]:
    """训练目标的三层字段：照写、请求可确认或前文承诺的推断、其余补全。

    teacher forcing 的前文已经包含目标分镜的明确选择，因此不要求先前状态再次出现在请求中。
    前镜 present 或 changes.after 建立的状态在后镜同值延续时属于 infer；请求来源评测仍使用
    inferable_units 的保守定义，不能把自承诺延续改称请求唯一推出的事实。未列出的结构与字段属于 complete。
    """
    from mspc.data.inferable_units import inferable_units, request_lines, whole_request_lines
    from mspc.data.shot_text import DEFAULT_CAMERA
    from mspc.data.storyboard import CHANGE_FIELDS, _comparable, parse_shot_script
    board = row['storyboard']
    out: dict[tuple, str] = {}
    if row['tier'] == 'shot':
        stated = parse_shot_script(row['script'])
        lines = request_lines(row['script'])
        for i, entry in enumerate(board['characters']):
            if entry['name'] in stated['characters']:
                fields = ('kind', 'appearance') + (('outfit',) if stated['characters'][entry['name']]['outfit'] is not None else ())
                out.update({('characters', i, f): 'copy' for f in fields})
        for kind in ('objects', 'places'):
            for i, entry in enumerate(board[kind]):
                if entry['name'] in stated[kind]:
                    out[(kind, i, 'appearance')] = 'copy'
        for i, shot in enumerate(board['shots']):
            for field, value in (shot.get('camera') or {}).items():
                if value and value not in DEFAULT_CAMERA:
                    out[('shots', i, 'camera', field)] = 'copy'
            out[('shots', i, 'action')] = 'copy'
    else:
        lines = whole_request_lines(row['script'], len(board['shots']))
    opening, changes = inferable_units(board, lines)
    defaults = {entry['name']: entry.get('outfit') for entry in board['characters']}
    committed: dict[tuple[str, str], Any] = {}
    holding_history: set[str] = set()
    for i, shot in enumerate(board['shots']):
        for j, entry in enumerate(shot['present']):
            name = entry['name']
            for field in CHANGE_FIELDS:
                if field not in entry or entry[field] is None or entry[field] == '':
                    continue
                value = entry[field]
                if field == 'outfit' and _comparable(value) == _comparable(defaults.get(name)):
                    continue
                direct = opening.get((i, name, field), (None, False))[1]
                continued = ((name, field) in committed
                             and _comparable(committed[(name, field)]) == _comparable(value))
                if direct or continued:
                    out[('shots', i, 'present', j, field)] = 'infer'
        # 只在本镜分层完毕之后记下承诺，不能用本镜末态反证同镜开镜值。
        for entry in shot['present']:
            for field in CHANGE_FIELDS:
                if field in entry and entry[field] is not None and entry[field] != '':
                    if field == 'holding':
                        name = entry['name']
                        confirmed = opening.get((i, name, field), (None, False))[1]
                        # 从未持物的默认 [] 不是非默认状态；不能仅因重复就增加其监督权重。
                        # 明确空手请求或已持物后清空，才建立可在后镜负责的空集合承诺。
                        if not entry[field] and name not in holding_history and not confirmed:
                            committed.pop((name, field), None)
                            continue
                        holding_history.add(name)
                    committed[(entry['name'], field)] = entry[field]
            # 未写 outfit 的在场人物穿登记默认衣着，这是现有分镜合同的实际含义。
            if entry['name'] in defaults and 'outfit' not in entry:
                committed[(entry['name'], 'outfit')] = defaults[entry['name']]
        for change in shot['changes']:
            if change['what'] == 'holding':
                name = change['name']
                confirmed = changes.get((i, name, 'holding'), (None, False))[1]
                if not (change['after'] or change.get('before') or name in holding_history or confirmed):
                    committed.pop((name, 'holding'), None)
                    continue
                holding_history.add(name)
            committed[(change['name'], change['what'])] = change['after']
    for (i, name, what), (_, ok) in changes.items():
        if ok:
            for k, change in enumerate(board['shots'][i]['changes']):
                if change['name'] == name and change['what'] == what:
                    out[('shots', i, 'changes', k)] = 'infer'
    return out


def _weighted_paths(row: Mapping[str, Any], weights: Mapping[str, float]) -> dict[tuple, float]:
    """分层到实际损失权重的唯一映射；身体姿态是开镜状态，复用 state 倍数。"""
    out = {}
    for path, tier in field_supervision_tiers(row).items():
        if tier == 'copy':
            out[path] = 0.0
        else:
            category = 'changes' if len(path) == 4 and path[2] == 'changes' else path[-1]
            out[path] = float(weights['state' if category == 'posture' else category])
    return out


def target_segments(row: Mapping[str, Any], weights: Mapping[str, float],
                    token_offsets: Callable[[str], Sequence[tuple[int, int]]]) -> list[tuple[str, float]]:
    """紧凑 JSON 目标切成连续几段，每段一个损失权重；拼回与 ``json.dumps(..., separators=(',', ':'))`` 逐字相同。

    一个值有权重时，它的键也带同一权重（写不写这个键正是要学的决定）。ms-swift 对每段单独分词，所以段界只能落在整段分词
    的 token 边界上：``token_offsets`` 给整段文本每个 token 的字符起止（规划器分词器），每个 token 取它最后一个字符的权重
    （Qwen 把值的右引号、逗号、下一个键的左引号并成一个 ``","``，归下一个键），相邻同权重的 token 合成一段。11-06 在 504 条
    真实行上核过：按字符切段时 453 行的分段分词与整段不同。"""
    if set(weights) != {'state', 'outfit', 'holding', 'changes'} or any(type(v) not in (int, float) or v <= 0 for v in weights.values()):
        raise ValueError('weights must give positive state/outfit/holding/changes')
    board = validate_storyboard(row['storyboard'])
    paths = _weighted_paths({**row, 'storyboard': board}, weights)
    char_weights: list[float] = []
    pieces: list[str] = []

    def put(text: str, weight: float) -> None:
        pieces.append(text)
        char_weights.extend([weight] * len(text))

    def emit(value: Any, path: tuple, weight: float) -> None:
        weight = paths.get(path, weight)
        if isinstance(value, dict):
            put('{', weight)
            for n, (key, item) in enumerate(value.items()):
                child = path + (key,)
                put((',' if n else '') + json.dumps(key, ensure_ascii=False) + ':', paths.get(child, weight))
                emit(item, child, weight)
            put('}', weight)
        elif isinstance(value, list):
            put('[', weight)
            for n, item in enumerate(value):
                if n:
                    put(',', weight)
                emit(item, path + (n,), weight)
            put(']', weight)
        else:
            put(json.dumps(value, ensure_ascii=False, allow_nan=False), weight)

    emit(board, (), 1.0)
    text = ''.join(pieces)
    if text != json.dumps(board, ensure_ascii=False, separators=(',', ':'), allow_nan=False):
        raise ValueError('segmented target does not reproduce the compact JSON')
    # 一个多字节字符可能拆成几个 token、它们的字符起止相同（11-06 全池里一块含希腊字母 ΨΡΦ）：
    # 字符范围相交的相邻 token 并成一个单元，段界只落在单元之间。
    units: list[list[int]] = []
    for begin, finish in token_offsets(text):
        if units and begin < units[-1][1]:
            units[-1][1] = max(units[-1][1], finish)
        else:
            units.append([begin, finish])
    if not units or units[0][0] != 0 or units[-1][1] != len(text) or any(a[1] != b[0] for a, b in zip(units, units[1:])):
        raise ValueError('token offsets must tile the target text')
    segments: list[list] = []
    for begin, finish in units:
        weight = char_weights[finish - 1]
        if segments and segments[-1][1] == weight:
            segments[-1][0] += text[begin:finish]
        else:
            segments.append([text[begin:finish], weight])
    return [(t, w) for t, w in segments]


def swift_example(row: Mapping[str, Any], weights: Mapping[str, float],
                  token_offsets: Callable[[str], Sequence[tuple[int, int]]]) -> dict:
    """ms-swift 标准监督微调的一行：回答拆成相邻几条 assistant 消息，各带 ``loss_scale``
    （框架先合并相邻 assistant，再按段分词）；``channel`` 是档位，开分通道损失看三档曲线。

    第一条是非思考档的空思考段，权重 0：训练时要配 ``add_non_thinking_prefix=False``，否则 ms-swift 自己再拼一个空思考段、
    还继承第一段的权重（11-06 在主服务器用它的 ``qwen3_5`` 模板实测；它内置的 ``ignore_empty_think`` 在逐消息给权重时不生效）。
    这样训练序列到回答开头为止与推理时 HF 对话模板的提示逐位相同。"""
    validate_planner_example(row)
    messages = planner_messages(row)
    return {'messages': messages + [{'role': 'assistant', 'content': EMPTY_THINK, 'loss_scale': 0.0}]
                        + [{'role': 'assistant', 'content': text, 'loss_scale': weight}
                           for text, weight in target_segments(row, weights, token_offsets)],
            'channel': row['tier'], 'sample_id': row['sample_id']}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--members', type=Path, action='append', required=True)
    parser.add_argument('--shot-spans', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    def read(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    members = [row for path in args.members for row in read(path)]
    pools, rejected = build_planner_pools(members, read(args.shot_spans))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    counts = {}
    for tier, rows in pools.items():
        counts[tier] = {}
        for split in ('train', 'val'):
            selected = [row for row in rows if row['split'] == split]
            counts[tier][split] = len(selected)
            (args.output_dir / f'{tier}-{split}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected))
    manifest = {'contract_version': PLANNER_CONTRACT_VERSION, 'schema': SCHEMA, 'members': len(members),
                'counts': counts, 'rejected': rejected, 'paths': {'members': [str(p) for p in args.members], 'shot_spans': str(args.shot_spans)}}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
