"""同一分镜表到原生渲染条件；只排版、分段和换算帧数，不改写故事。"""
from __future__ import annotations

from itertools import accumulate
import re
from typing import Callable

from mspc.data.shot_text import shot_text
from mspc.data.storyboard import NAME, entity_references, validate_storyboard
from mspc.data.frame_planner import resample_frame_count
from mspc.core.wan22_contract import SEGMENT_MAX_FRAMES, STORYBOARD_FPS

FAMILIES = ('wan_single', 'wan_segment', 'wan_official', 'wan21_official', 'multishotmaster', 'holocine', 'storymem')
OFFICIAL_WAN = ('wan_official', 'wan21_official')
#: 本项目渲染器的 max_frames 是装窗预算：单镜超过预算时独占一窗（单镜帧数已由计划合同限在 228 内）。
#: 对比方法的 max_frames 是模型能生成的硬上限，单镜超过就报错。
WINDOW_BUDGET_FAMILIES = ('wan_single', 'wan_segment')


def official_ti2v_frames(frame_count: int) -> int:
    """把共同计划的 16fps 镜长换成官方 TI2V-5B 的 24fps、4n+1 帧数。

    官方 generate 要求帧数为 4n+1，示例长度 121。这里只做时长换算后的最近合法长度，不改镜头内容。
    """
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError('frame count must be a positive integer')
    frames = resample_frame_count(frame_count, source_fps=STORYBOARD_FPS,
                                  target_fps=24, temporal_stride=4)
    if frames < 5 or frames % 4 != 1:
        raise ValueError('official TI2V frame count must be 4n+1 and at least 5')
    return frames


def _sentence(text: str) -> str:
    text = text.strip()
    return text[0].upper() + text[1:].rstrip('.') + '.' if text else ''


def _definite(text: str) -> str:
    """把登记表的名词短语用于指称；不删除外貌、衣着或不确定性。"""
    text = re.sub(r'^an? ', 'the ', text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'^(the (?:male|female)), ([^,]+ hair),', r'\1 with \2,', text)
    return re.sub(r', (?=wearing\b)', ' ', text)


def _state_clause(who: str, state: str) -> str:
    """旧真实计划含“衣物 removed”等名词补语；只修句法，不补动作。"""
    clauses = []
    for index, part in enumerate(state.split(', ')):
        noun = re.fullmatch(
            r'(.+?) (removed|released|lowered|exposed|fastened|opened|zipped|pulled|tied|loose|down|open|empty)(.*)',
            part)
        if noun:
            item, predicate, tail = noun.groups()
            item = re.sub(r'^the ', '', item)
            clauses.append(f'with the {item} {predicate}{tail}' if index else
                           f'For {who}, the {item} is {predicate}{tail}')
        else:
            clauses.append(part if index else f'{who} is {part}')
    return ', '.join(clauses)


def _outfit_clause(who: str, outfit: str) -> str:
    outfit = re.sub(r'^(?:wearing|wears) ', '', outfit, flags=re.IGNORECASE)
    if re.match(r'^(?:shirtless|topless|naked|nude|bare)\b', outfit, re.IGNORECASE):
        return f'{who} is {outfit}'
    return f'{who} wears {outfit}'


class _OfficialCaption:
    """近邻提示的唯一文字适配：编号留给原生标签，StoryMem使用外貌指称。"""

    def __init__(self, board: dict, family: str, names: dict[str, str]):
        self.board, self.family, self.names = board, family, names
        self.registry = {entry['name']: entry for kind in ('characters', 'objects', 'places')
                         for entry in board[kind]}
        self.full = {name: _definite(re.split(r'(?<=[.!?])\s+', entry['appearance'])[0].rstrip('.'))
                     for name, entry in self.registry.items()}
        candidates = {name: re.split(r',?\s+wearing\b', text)[0] for name, text in self.full.items()}
        # 同发色而衣着不同的两人不能都缩成“the man with short dark hair”。
        self.short = {name: text if list(candidates.values()).count(text) == 1 else self.full[name]
                      for name, text in candidates.items()}
        for name, full in self.full.items():
            clothing = re.split(r'\s+wearing ', full, maxsplit=1)
            if self.short[name] == full and len(clothing) == 2:
                head = re.split(r' with |,', clothing[0])[0]
                dressed = f'{head} wearing {clothing[1]}'
                if sum(text.endswith('wearing ' + clothing[1]) for text in self.full.values()) == 1:
                    self.short[name] = dressed
            if name.startswith('Object '):
                head = full.split(' with ')[0]
                if sum(text.split(' with ')[0] == head for text in self.full.values()) == 1:
                    self.short[name] = head
        self.seen: set[str] = set()
        self.other_people: set[str] = set()

    def reference(self, name: str) -> str:
        if name in self.names:
            return self.names[name]
        text = self.short[name] if name in self.seen else self.full[name]
        if name in self.other_people:
            text = re.sub(r'^the ', 'the other ' if name in self.seen else 'another ', text)
        self.seen.add(name)
        return NAME.sub(lambda m: self.reference(m.group()), text)

    def prose(self, text: str) -> str:
        # 真实手部失败句：先建立手的所属关系，避免把人物外貌插入逗号从句。
        text = re.sub(r'\b[Aa] hand,? belonging to (Character \d+),?', r'The hand of \1', text)
        if self.family == 'storymem':
            # 真实语料中的身体部位所有格；长外貌短语放到of之后，不能变成“shirt's head”。
            pattern = rf"({NAME.pattern})(?:'s (facial expression|head and shoulder|head|face|hands|chest)\b)?"
            return re.sub(pattern, lambda m: (
                f'the {m[2]} of {self.reference(m[1])}' if m[2] else self.reference(m[1])), text)
        return NAME.sub(lambda m: self.reference(m.group()), text)

    def shot(self, index: int, *, stable_outfits: dict[str, str], single_place: bool) -> str:
        self.seen.clear()
        self.other_people.clear()
        shot = self.board['shots'][index]
        if self.family == 'storymem':
            descriptions: dict[str, list[str]] = {}
            for name in entity_references([shot]):
                if name.startswith('Character '):
                    descriptions.setdefault(self.full[name], []).append(name)
            for people in descriptions.values():
                if len(people) > 2:
                    raise ValueError('more than two co-present characters have identical appearance descriptions')
                self.other_people.update(people[1:])
        parts = []
        if shot.get('place') and (self.family == 'storymem' or not single_place):
            place = self.registry[shot['place']]['appearance']
            parts.append(_sentence(f'The setting is {place}'))
        camera = shot.get('camera') or {}
        camera_text = ', '.join(camera[key] for key in ('shot_size', 'angle', 'movement') if camera.get(key))
        if camera_text:
            parts.append(_sentence(camera_text))
        for entry in shot['present']:
            name = entry['name']
            for field in ('posture', 'outfit', 'state', 'holding'):
                value = entry.get(field, self.registry[name].get('outfit') if field == 'outfit' else None)
                if not value or (field == 'outfit' and stable_outfits.get(name) == value):
                    continue
                # 动作原文已明确写出的开镜状态不再重复；不做模糊语义删减。
                if field in ('state', 'posture') and f'{name} is {value}'.casefold() in shot['action'].casefold():
                    continue
                who = self.reference(name)
                if field == 'holding':
                    clause = f"{who} holds " + ' and '.join(self.reference(item) for item in value)
                elif field == 'outfit':
                    clause = _outfit_clause(who, self.prose(value))
                    clause = clause.replace(f'{who} wears ', f'{who} is wearing ', 1)
                else:
                    clause = _state_clause(who, self.prose(value))
                parts.append(_sentence(clause))
        parts.append(_sentence(self.prose(shot['action'])))
        for change in shot['changes']:
            # 变化前的值以分镜表 changes 自带的 before 为准；开镜名单可能没写持物，不能从它推。
            name, field, before, after = change['name'], change['what'], change['before'], change['after']
            if before == after:
                raise ValueError(f"shot {shot['shot']}: {name} {field} is already {after!r} before its change")
            who = self.reference(name)
            if field == 'holding':
                clause = (f'{who} holds ' + ' and '.join(self.reference(item) for item in after)
                          if after else f'{who} no longer holds ' + ' and '.join(
                              self.reference(item) for item in before or []))
                if not after and not before:
                    raise ValueError('empty holding change has no prior held entity')
            elif field == 'outfit':
                clause = _outfit_clause(who, self.prose(after))
            elif field in ('state', 'posture'):
                clause = _state_clause(who, self.prose(after))
            else:
                raise ValueError(f'unknown change kind {field!r}')
            if clause.startswith('For '):
                clause = 'for ' + clause[4:]
            parts.append(f'By the end of the shot, {clause}.')
        if self.family == 'storymem':
            # 多句appearance不能接进名词短语；补充句单独明确归属，保留原有内容。
            for name in entity_references([shot]):
                entry = self.registry[name]
                if name.startswith('Location '):
                    continue  # 场景完整描述已经写在镜头开头。
                sentences = re.split(r'(?<=[.!?])\s+', entry['appearance'])
                if name not in self.seen:
                    parts.append(_sentence(f"The shot includes {self.reference(name)}"))
                for extra in sentences[1:]:
                    who = self.short[name]
                    if re.match(r'^(?:He|She|It|They)\b', extra):
                        extra = re.sub(r'^(?:He|She|It|They)\b', who, extra)
                    elif re.match(r'^(?:His|Her|Its|Their)\b', extra):
                        extra = f'For {who}, {extra[0].lower() + extra[1:]}'
                    parts.append(_sentence(self.prose(extra)))
        return re.sub(r'(?<=[.!?])\s+([a-z])', lambda m: ' ' + m[1].upper(), ' '.join(parts))


def global_and_shots(board: dict, start: int, end: int, family: str) -> tuple[str, list[str]]:
    """全局登记外貌和场景，逐镜只编排给定动作、状态与镜头。"""
    referenced = set(entity_references(board['shots'][start:end]))
    order = [name for name in entity_references(board['shots']) if not name.startswith('Location ')]
    if family == 'multishotmaster':
        names = {name: f'Subject {i}' for i, name in enumerate(order, 1)}
    elif family == 'holocine':
        names = {entry['name']: f'[character{i}]' for i, entry in enumerate(board['characters'], 1)}
    else:
        raise ValueError(f'no global/shot format for {family}')

    writer = _OfficialCaption(board, family, names)
    descriptions, stable_outfits = [], {}
    for character in board['characters']:
        name = character['name']
        if name not in referenced:
            continue
        outfits = {entry.get('outfit', character.get('outfit', ''))
                   for shot in board['shots'][start:end] for entry in shot['present'] if entry['name'] == name}
        outfits.update(change['after'] for shot in board['shots'][start:end] for change in shot['changes']
                       if change['name'] == name and change['what'] == 'outfit')
        stable_outfit = next(iter(outfits)) if len(outfits) == 1 else ''
        stable_outfits[name] = stable_outfit
        description = f"{names[name]}: {writer.prose(character['appearance'])}"
        if stable_outfit:
            description += f'; outfit: {stable_outfit}'
        descriptions.append(description.rstrip('.') + '.')
    for kind in ('objects', 'places'):
        for entry in board[kind]:
            if entry['name'] in referenced:
                label = ('Setting' if kind == 'places' else
                         names[entry['name']] if family == 'multishotmaster' else 'Recurring object')
                descriptions.append(f"{label}: {writer.prose(entry['appearance']).rstrip('.')}.")
    single_place = len({shot.get('place') for shot in board['shots'][start:end]}) == 1
    return ' '.join(descriptions), [
        writer.shot(i, stable_outfits=stable_outfits, single_place=single_place) for i in range(start, end)]


def holocine_inputs(global_caption: str, shot_captions: list[str], frame_counts: list[int]) -> dict:
    """HoloCine官方prepare_multishot_inputs的纯函数部分。

    官方示例文件在module顶层加载权重，不能在备料时import；这里沿用其最近4n+1及标记规则。
    来源：yihao-meng/HoloCine/HoloCine_inference_full_attention.py。
    保留整个实际输出；不再通过裁一帧伪造逐镜等长。
    """
    nearest = lambda n: 4 * round((n - 1) / 4) + 1
    requested_cuts = list(accumulate(frame_counts))[:-1]
    frames = nearest(sum(frame_counts))
    cuts = sorted({nearest(cut) for cut in requested_cuts if 0 < nearest(cut) < frames})
    if len(cuts) != len(frame_counts) - 1:
        raise ValueError('HoloCine rounding collapses a required shot; do not drop that shot')
    if 'This scene contains' not in global_caption:
        global_caption = global_caption.strip() + f' This scene contains {len(shot_captions)} shots.'
    return {'global_caption': global_caption, 'shot_captions': shot_captions,
            'prompt': f"[global caption] {global_caption} [per shot caption] " + ' [shot cut] '.join(shot_captions),
            'requested_num_frames': sum(frame_counts), 'requested_shot_cut_frames': requested_cuts,
            'num_frames': frames, 'shot_cut_frames': cuts}


def segment_inputs(board: dict, family: str, start: int, end: int) -> dict:
    counts = [shot['frame_count'] for shot in board['shots'][start:end]]
    row = {'start_shot': start, 'end_shot': end, 'frame_counts': counts}
    if family in ('wan_single', 'wan_segment'):
        return {**row, 'prompts': [shot_text(board, i) for i in range(start, end)], 'num_frames': sum(counts)}
    if family == 'wan_official':
        output_counts = [official_ti2v_frames(count) for count in counts]
        return {**row, 'prompts': [shot_text(board, i) for i in range(start, end)],
                'source_frame_counts': counts, 'output_frame_counts': output_counts,
                'source_fps': 16, 'output_fps': 24, 'num_frames': sum(output_counts)}
    if family == 'wan21_official':
        # 官方 T2V-1.3B 的 sample_fps 就是 16，共同计划的 4n+1 帧数原样使用。
        return {**row, 'prompts': [shot_text(board, i) for i in range(start, end)],
                'output_fps': 16, 'num_frames': sum(counts)}
    if family == 'storymem':
        writer = _OfficialCaption(board, family, {})
        return {**row, 'prompts': [writer.shot(i, stable_outfits={}, single_place=False)
                                 for i in range(start, end)], 'num_frames': sum(counts)}
    global_caption, shots = global_and_shots(board, start, end, family)
    if family == 'holocine':
        return {**row, **holocine_inputs(global_caption, shots, counts)}
    # 官方MSM按每镜补4n+1；共享分镜表已经满足此条件，worker仍调用官方pad/recover。
    boundaries = [0, *accumulate(counts)]
    return {**row, 'global_caption': global_caption, 'shot_captions': shots,
            'prompts': [f'Story: {global_caption} Now: {shot}' for shot in shots],
            'shot_groups': [[a, b] for a, b in zip(boundaries, boundaries[1:])],
            'num_frames': sum(counts)}


def adapt_storyboard(board: dict, *, family: str, limits: dict, count_tokens: Callable[[str], int]) -> list[dict]:
    board = validate_storyboard(board)
    if family not in FAMILIES:
        raise ValueError(f'unknown renderer family {family}')
    # 本项目渲染器：从第一镜贪心装窗，整窗不超过 max_frames（逐镜为 228；联合窗口的帧数预算在 1..228 内由条件给，
    # 单镜超过预算时独占一窗，见 pack_segments）；镜数上限是渲染器训练时的窗口镜数（None 不限，1 为逐镜，
    # 旧的至多 4 镜渲染器给 4）。对比方法各自的镜数限制见下。
    if family in WINDOW_BUDGET_FAMILIES:
        shots_ok = (limits['max_shots'] == 1 and type(limits['max_shots']) is int if family == 'wan_single' else
                    limits['max_shots'] is None or (type(limits['max_shots']) is int and limits['max_shots'] >= 1))
        frames_ok = (limits['max_frames'] == SEGMENT_MAX_FRAMES if family == 'wan_single' else
                     type(limits['max_frames']) is int and 1 <= limits['max_frames'] <= SEGMENT_MAX_FRAMES)
        if not shots_ok or not frames_ok:
            raise ValueError(f'{family} uses the existing renderer interface: max_shots '
                             f'{"1" if family == "wan_single" else "None or a positive integer"}, max_frames '
                             f'{SEGMENT_MAX_FRAMES if family == "wan_single" else f"1..{SEGMENT_MAX_FRAMES}"}')
    if family in OFFICIAL_WAN and limits['max_shots'] != 1:
        raise ValueError('official Wan generates one shot per call')
    if family == 'storymem' and limits['max_shots'] != 1:
        raise ValueError('StoryMem generates one shot per call and updates memory after each shot')
    if family == 'storymem' and any(shot['frame_count'] % 4 != 1 for shot in board['shots']):
        raise ValueError('StoryMem requires every shot to have 4n+1 frames without rounding')
    return pack_segments(len(board['shots']), lambda start, end: segment_inputs(board, family, start, end),
                         family=family, limits=limits, count_tokens=count_tokens)


def pack_segments(shot_count: int, build: Callable[[int, int], dict], *, family: str, limits: dict,
                  count_tokens: Callable[[str], int]) -> list[dict]:
    """按原生窗口从前往后贪心装镜；build(start, end) 给出该段的原生输入。"""
    if (limits['max_shots'] is not None and (type(limits['max_shots']) is not int or limits['max_shots'] < 1)) or any(type(limits[k]) is not int or limits[k] < 1 for k in ('max_frames', 'max_tokens')):
        raise ValueError('renderer limits must be positive integers')
    segments, start = [], 0
    while start < shot_count:
        best = None
        maximum = shot_count if limits['max_shots'] is None else min(shot_count, start + limits['max_shots'])
        for end in range(start + 1, maximum + 1):
            try:
                candidate = build(start, end)
            except ValueError:
                break  # 合并导致切点消失时，在前一镜收段；单镜仍失败则在下方显式报错。
            texts = [candidate['prompt']] if family == 'holocine' else candidate['prompts']
            counts = [count_tokens(text) for text in texts]
            lone_shot = family in WINDOW_BUDGET_FAMILIES and end == start + 1
            if (candidate['num_frames'] > limits['max_frames'] and not lone_shot) or max(counts) > limits['max_tokens']:
                break
            best = {**candidate, 'token_counts': counts}
        if best is None:
            raise ValueError(f'shot {start + 1} cannot fit {family} without truncation or changing the story')
        segments.append(best)
        start = best['end_shot']
    return segments
