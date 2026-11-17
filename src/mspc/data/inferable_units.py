"""「推得出」的单元：监督加权与信息量评测共用这一处定义。

单元是（镜，实体，字段）开镜属性与（镜，实体，种类）镜内变化。这里只确认有明确实体绑定的词面证据，以及已确认值的延续。
False 表示词面无法确认，不表示不合理或语义错误。不能解析的同义词、指代、否定范围和时序不猜测。
逐镜档使用 ``sNN:`` 行；粗档若没有实体编号，不能靠共享的外貌词猜人物。该定义也用于未来监督数据的加权；已冻结的训练数据不变。
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Mapping

STOP = frozenset(
    'a an the of with and in on at to for from by is are be her his their its into onto over under near who that this these '
    'those while as it he she they them him then one two up down out off character object location shot'.split())
NAME = re.compile(r'\b(?:Character|Object) [1-9][0-9]*\b')
SHOT_LINE = re.compile(r'^s(\d+):\s*(.*)$')
ATTRIBUTES = ('posture', 'state', 'outfit', 'holding')


def stems(text: str) -> set[str]:
    out = set()
    for word in re.findall(r'[a-z]+', (text or '').lower()):
        if word in STOP or len(word) <= 2:
            continue
        for suffix in ('ing', 'ed', 'es', 's'):
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                word = word[: -len(suffix)]
                break
        out.add(word)
    return out


def request_lines(request: str) -> dict[int, str]:
    """逐镜档请求的 ``sNN: …`` 行，按零起镜号。"""
    lines = {}
    for line in request.splitlines():
        match = SHOT_LINE.match(line.strip())
        if match:
            lines[int(match[1]) - 1] = match[2]
    if not lines:
        raise ValueError('not a shot-tier request: no sNN lines')
    return lines


def entity_registry(board: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {e['name']: e for key in ('characters', 'objects') for e in board.get(key, [])}


def lexical_value(value: str) -> tuple[str, ...]:
    """仅统一大小写、标点与冠词；保留颜色、否定、方向、数字和词序，不猜同义词。"""
    return tuple(word for word in re.findall(r"[a-z]+|[0-9]+", value.lower()) if word not in {'a', 'an', 'the'})


def _evidence_words(value: str) -> set[str]:
    """全部内容词须有依据；只折叠常规词尾，不丢颜色、否定、方向或实体编号。"""
    words = set()
    for word in lexical_value(value):
        if word in {'of', 'with', 'and', 'in', 'on', 'at', 'to', 'for', 'from', 'by', 'is', 'are', 'be'}:
            continue
        for suffix in ('ing', 'ed', 'es', 's'):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[:-len(suffix)]
                if len(word) >= 2 and word[-1] == word[-2]:
                    word = word[:-1]
                break
        words.add(word)
    return words


SUBJECT = re.compile(r'(?:^|[.;]|\b(?:and|while|then|but|as)\b)\s*((?:Character|Object) [1-9][0-9]*)\b')
TRANSFER = re.compile(r'\b(Character [1-9][0-9]*)\s+(?:hands?|handed|passes?|passed|gives?|gave)\s+'
                      r'(Object [1-9][0-9]*)\s+to\s+(Character [1-9][0-9]*)\b')
NEGATION = re.compile(r"\b(?:not|no|never|without|neither|nor)\b|n't\b", re.I)
TEMPORAL = re.compile(r'\b(?:then|before|after|until)\b', re.I)
ENDPOINT = re.compile(r'\b(?:now|leaving|becomes?|changes? into)\b', re.I)


def _predicates(line: str, name: str) -> list[str]:
    matches = list(SUBJECT.finditer(line))
    return [line[m.end():matches[j + 1].start() if j + 1 < len(matches) else len(line)]
            for j, m in enumerate(matches) if m[1] == name]


def _transfer_holding(value: Any, line: str, name: str, phase: str, before: Any) -> bool:
    if not isinstance(value, list) or NEGATION.search(line) or TEMPORAL.search(line):
        return False
    transfers = list(TRANSFER.finditer(line))
    if len(transfers) != 1:
        return False  # 连续转交的开镜/镜末归属不能由任一局部交接推出。
    for match in transfers:
        source, obj, recipient = match.groups()
        subject = SUBJECT.search(line)
        if phase == 'opening' and name == source and value == [obj] and subject and subject.start(1) == match.start():
            return True
        if phase == 'change' and isinstance(before, list):
            if name == source and obj in before and sorted(value) == sorted(x for x in before if x != obj):
                return True
            if name == recipient and obj not in before and sorted(value) == sorted([*before, obj]):
                return True
    return False


def evidenced(value: Any, line: str, registry: Mapping[str, Mapping[str, Any]], *, name: str,
              field: str, phase: str, before: Any = None) -> bool:
    """保守词面确认：不能把别人的状态、镜末值或物体被提及当作本人的开镜值。"""
    # 必须在交接捷径及主语切分之前检查：切分会吞掉 then，局部交接不能证明后续事件后的归属。
    if TEMPORAL.search(line):
        return False
    if field == 'holding' and _transfer_holding(value, line, name, phase, before):
        return True
    for predicate in _predicates(line, name):
        # 带先后或否定的自然语言需要语义判读；此处明确留 unknown。
        if NEGATION.search(predicate):
            continue
        # 同样约束持物与空手分支：A watches B holding O 不能证明 A 持有 O。
        predicate = re.split(r'\bCharacter [1-9][0-9]*\b', predicate, maxsplit=1)[0]
        if phase == 'opening' and ENDPOINT.search(predicate):
            continue  # 模型即使漏写 changes，也不能把显式末态提前当开镜证据。
        if isinstance(value, list):
            if not value:
                if re.search(r'\bempty[- ]handed\b', predicate, re.I):
                    return True
                continue
            held = re.search(r'\b(?:holds?|holding|carries|carrying)\s+(.+)', predicate, re.I)
            if held:
                identifiers = set(NAME.findall(held[1]))
                if identifiers == set(value):
                    return True
                # 无编号时只能用登记表的完整描述，不能凭共同的 "black" 或 "small" 认物。
                words = _evidence_words(held[1])
                if not identifiers and all((_evidence_words(registry.get(obj, {}).get('appearance', '')) or {'__missing__'}) <= words
                                           for obj in value):
                    return True
            # 真实请求：Character 3 looks down at Object 1 in her hands.
            in_hands = re.search(r'\b(Object [1-9][0-9]*)\s+in\s+(?:her|his|their)\s+hands?\b', predicate)
            if in_hands and value == [in_hands[1]]:
                return True
            continue
        # 后续被提到的实体不是当前主语："A watches B stand" 不能证明 A 站着。
        scalar_predicate = NAME.split(predicate, maxsplit=1)[0]
        if phase == 'change':
            # 真实换装请求同时写了脱下的旧衣和 now 后的新衣，不能把 before 词当 after 的依据。
            endpoint = list(ENDPOINT.finditer(scalar_predicate))
            if endpoint:
                scalar_predicate = scalar_predicate[endpoint[-1].end():]
            elif re.search(r'\b(?:takes?|taking) off\b', scalar_predicate, re.I):
                continue
        if phase == 'opening' and field == 'posture' and re.search(r'\b(?:stands? up|sits? down)\b', scalar_predicate, re.I):
            continue
        if field == 'posture':
            # 身体姿态复用分镜唯一词形定义：stood/standing、seated/sitting 同词干，
            # understand 仍是独立单词，不能因包含 stand 而确认站立。
            from mspc.data.storyboard import posture_stem, posture_words
            want, got = set(posture_words(value)), set(posture_words(scalar_predicate))
            if posture_stem(value) not in got:
                continue
        else:
            want, got = _evidence_words(value), _evidence_words(scalar_predicate)
        if want and want <= got:
            return True
    return False


def inferable_units(board: Mapping[str, Any], lines: Mapping[int, str]) -> tuple[dict, dict]:
    """开镜属性与镜内变化单元，各带"推得出"标记：{(镜, 实体, 字段): (值, 推得出)}。"""
    registry = entity_registry(board)
    default_outfit = {name: e.get('outfit') for name, e in registry.items()}
    last: dict[str, dict[str, Any]] = defaultdict(dict)
    opening, changes = {}, {}
    for index, shot in enumerate(board['shots']):
        line = lines.get(index, '')
        for entry in shot.get('present', []):
            name = entry['name']
            for field in ATTRIBUTES:
                value = entry.get(field)
                if value is None or value == '' or (field == 'outfit' and value == default_outfit.get(name)):
                    continue
                carried = field in last[name] and last[name][field] == value
                changing = any(c['name'] == name and c['what'] == field for c in shot.get('changes', []))
                direct = evidenced(value, line, registry, name=name, field=field, phase='opening')
                # 同镜发生变化时，动作里的目标值不能反向当作开镜状态；显式交接的失去者例外。
                if changing:
                    direct = field == 'holding' and _transfer_holding(value, line, name, 'opening', None)
                opening[(index, name, field)] = (value, carried or direct)
        for change in shot.get('changes', []):
            after = change['after']
            changes[(index, change['name'], change['what'])] = (after, evidenced(
                after, line, registry, name=change['name'], field=change['what'], phase='change', before=change.get('before')))
        for entry in shot.get('present', []):
            for field in ATTRIBUTES:
                unit = opening.get((index, entry['name'], field))
                if unit and unit[1]:
                    last[entry['name']][field] = unit[0]
                elif field in entry:
                    last[entry['name']].pop(field, None)
        for change in shot.get('changes', []):
            after = change['after']
            if changes[(index, change['name'], change['what'])][1]:
                last[change['name']][change['what']] = after
            else:
                last[change['name']].pop(change['what'], None)
    return opening, changes




def whole_request_lines(request: str, shot_count: int) -> dict[int, str]:
    """粗档：整段请求都算每一镜的请求行。"""
    return {index: request for index in range(shot_count)}
