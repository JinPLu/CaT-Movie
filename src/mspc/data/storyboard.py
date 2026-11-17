"""v18 分镜表的运行合同：登记表、逐镜快照与帧数；子段只在这里重编号和重定默认穿着。"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import heapq
import math
import re
from typing import Annotated, Any, Literal, Mapping, Sequence, get_args

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, create_model, model_validator

from mspc.core.wan22_contract import SEGMENT_MAX_FRAMES
from mspc.data.shot_text import LAYOUT_FIELDS, with_layout_fields

SCHEMA = 'storyboard_v18'
NAME = re.compile(r'\b(?:Character|Object|Location|Subject) [1-9]\d*\b')
REGISTRIES = {'characters': 'Character', 'objects': 'Object', 'places': 'Location'}
SHOT_FIELDS = ('shot', 'picture_type', 'place', 'camera', 'present', 'action', 'changes')
FRAME_COUNTS = tuple(range(1, SEGMENT_MAX_FRAMES + 1, 4))
EntityName = Annotated[StrictStr, Field(pattern=r'^(Character|Object) [1-9][0-9]*$')]
UnitCoordinate = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


def entity_references(value):
    """任意结构中明确写出的实体编号，沿文本出现次序去重。"""
    return tuple(dict.fromkeys(match.group() for text in _strings(value) for match in NAME.finditer(text)))


def unregistered_entity_references(value, known):
    """只报原文里不在登记表的编号，不猜测它指向哪个实体。"""
    return tuple(name for name in entity_references(value) if name not in known)


def holding_targets(shot):
    """持有关系涉及的实体，按首次引用去重；类型与身份仍由登记表决定。"""
    values = [entry.get('holding') for entry in shot.get('present', []) if isinstance(entry, dict)]
    values += [change.get(side) for change in shot.get('changes', [])
               if isinstance(change, dict) and change.get('what') == 'holding' for side in ('before', 'after')]
    return list(dict.fromkeys(name for value in values if isinstance(value, list)
                              for name in value if isinstance(name, str)))


def complete_holding_presence(shot, known):
    """只补已登记的实体名；不推断外貌、类别、可见性或状态。返回新增名字。"""
    listed = {entry.get('name') for entry in shot['present'] if isinstance(entry, dict)}
    added = [name for name in holding_targets(shot)
             if known.get(name) in ('characters', 'objects') and name not in listed]
    shot['present'].extend({'name': name} for name in added)
    return added


CHANGE_FIELDS = ('outfit', 'posture', 'state', 'holding')


def _comparable(value):
    """比较两个状态用：持物按集合，文字去首尾空格、不分大小写。"""
    if isinstance(value, list):
        return tuple(sorted(item for item in value if isinstance(item, str)))
    if isinstance(value, str):
        return value.strip().casefold()
    return value


#: 姿态写法的不规则词形 → 词干。
_POSTURE_FORMS = {"lying": "lie", "lies": "lie", "lay": "lie", "seated": "sit", "sat": "sit", "knelt": "kneel",
                  "stood": "stand"}
#: 身体姿态的词干，以及动作句里「转入这个姿态」的写法（计划没写变化、动作暗含变化）。
BODY_POSTURE_TRANSITIONS = {
    "stand": re.compile(r"\b(?:stands?|stood|standing)\s+up\b|\bgets?\s+up\b|\bgot\s+up\b", re.IGNORECASE),
    "sit": re.compile(r"\b(?:sits?|sat|sitting)\s+down\b|\btakes?\s+a\s+seat\b", re.IGNORECASE),
    "kneel": re.compile(r"\b(?:kneels?|knelt|kneeling)\b", re.IGNORECASE),
    "lie": re.compile(r"\b(?:lies|lay|lie|lying)\s+down\b", re.IGNORECASE),
    "crouch": re.compile(r"\bcrouch(?:es|ed)?\b", re.IGNORECASE),
    "squat": re.compile(r"\bsquats?\b|\bsquatted\b", re.IGNORECASE),
}


#: 比姿态时不算实词的词：系动词、冠词、介词、代词、连词与 up／down 一类小品词。
_POSTURE_FILLER = frozenset("is are be being a an the on in at of to with and her his their its them him it "
                            "up down straight over onto into".split())


def _stem(word: str) -> str:
    if word in _POSTURE_FORMS:
        return _POSTURE_FORMS[word]
    if word.endswith("ing") and len(word) > 4:
        word = word[:-3]
        if len(word) > 2 and word[-1] == word[-2] and word[-1] not in "lsz":
            word = word[:-1]
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        word = word[:-1]
    return word


def posture_words(value) -> tuple:
    """姿态写法 → 各实词的词干（按出现顺序）：sitting down → (sit,)，kneeling on the floor → (kneel, floor)。"""
    if not isinstance(value, str):
        return ()
    return tuple(_stem(w) for w in re.findall(r"[a-z]+", value.casefold()) if w not in _POSTURE_FILLER)


def posture_stem(value):
    """姿态的动作中心词：第一个实词的词干（sitting down → sit，standing up straight → stand）；没有返回 None。"""
    words = posture_words(value)
    return words[0] if words else None


def _same(what, a, b) -> bool:
    """同一属性的两个值是否相同。姿态：中心词相同，且一边的实词是另一边的子集（sitting down 与 sitting、
    kneeling on the floor 与 kneeling 相同；standing with hands raised 与 standing and covering her chest 不同）；
    其余按 ``_comparable``。"""
    if what == "posture" and isinstance(a, str) and isinstance(b, str):
        x, y = posture_words(a), posture_words(b)
        return bool(x) and bool(y) and x[0] == y[0] and (set(x) <= set(y) or set(y) <= set(x))
    return _comparable(a) == _comparable(b)


def _unstated(value):
    """``before`` 没写：None 或空串。持物的空列表是写明了「空手」，不算没写。"""
    return value is None or (isinstance(value, str) and not value.strip())


def _transition_for(action: str, stem: str, name: str) -> bool:
    """动作句里有转入姿态 ``stem`` 的写法，且这个写法说的是 ``name``：同一句里写法之前最近的「Character N」就是此人；
    同一句里写法之前没有名字时（规划器计划多用外貌描述）不排除。"""
    for sentence in re.split(r"(?<=[.!?;])\s+", action):
        for match in BODY_POSTURE_TRANSITIONS[stem].finditer(sentence):
            names = re.findall(r"\b(?:Character|Object)\s+\d+\b", sentence[:match.start()])
            if not names or names[-1] == name:
                return True
    return False


def closing_postures(shot) -> dict:
    """这一镜各人的镜末姿态：开镜姿态，再按顺序走完姿态变化。没写姿态的人不给。"""
    out = {e["name"]: e["posture"] for e in shot.get("present") or []
           if isinstance(e, dict) and isinstance(e.get("posture"), str)}
    for change in shot.get("changes") or []:
        if isinstance(change, dict) and change.get("what") == "posture" and isinstance(change.get("after"), str):
            out[change["name"]] = change["after"]
    return out


def align_opening_with_changes(shot, outfits, previous_postures):
    """镜内变化是开镜值的唯一依据：返回逐项修补记录，直接改 ``shot``。

    在场名单写的是开镜时的样子；同一实体同一属性第一次变化时，若开镜值已经等于 ``after``，就改回 ``before``。
    ``before`` 没写时，这条变化与开镜值分不出先后，删掉这条变化。之后按顺序推进：结果与当时样子相同的变化删掉，
    包括 ``before`` 与 ``after`` 相同的空变化。穿着没写开镜值时按登记表默认穿着算，与排版函数一致。
    「开镜值是否已是变化后、该改回 before」这一判断，姿态按 ``_same`` 比（sitting down 与 sitting、kneeling on the floor
    与 kneeling 算同一姿态），开镜值本就等于 before 时不算；凡是要删掉一条变化的判断（没写 before、前后相同）仍逐字比
    （standing → standing at the window 是位置变化，不删；开镜 looking down、变化后 looking down at the phone 也不删）。
    2026-10-16 全量：25,623 条变化的开镜值写成了变化之后（逐字相同的那部分）。

    计划没写姿态变化、但动作句写了此人转入开镜姿态的过程（``BODY_POSTURE_TRANSITIONS``，归属见 ``_transition_for``），而此人在前面最近一次出场的
    镜末姿态（``previous_postures``，调用方按镜序给）是别的姿态时，开镜姿态其实是镜末终态：把它从开镜挪成一条
    变化，before 取上一次镜末姿态（是身体姿态时）或不写。之前没出场过的人不动。形状不合法的条目原样留给校验。
    """
    present, changes = shot.get('present'), shot.get('changes')
    if not isinstance(present, list) or not isinstance(changes, list):
        return []
    entries = {entry['name']: entry for entry in present if isinstance(entry, dict) and isinstance(entry.get('name'), str)}
    fixes, kept, current = [], [], {}
    for change in changes:
        if not (isinstance(change, dict) and change.get('name') in entries and change.get('what') in CHANGE_FIELDS
                and isinstance(change.get('after'), (str, list))):
            kept.append(change)
            continue
        name, what, before, after = change['name'], change['what'], change.get('before'), change['after']
        if not _unstated(before) and _comparable(before) == _comparable(after):
            fixes.append({'rule': '删掉没有变化的变化', 'name': name, 'what': what, 'value': deepcopy(change)})
            continue
        if (name, what) not in current:
            entry = entries[name]
            start = entry.get(what, outfits.get(name) if what == 'outfit' else None)
            if start is not None and _unstated(before) and _comparable(start) == _comparable(after):
                fixes.append({'rule': '开镜已是变化后且没写变化前', 'name': name, 'what': what, 'value': deepcopy(change)})
                continue
            if (start is not None and not _unstated(before) and _comparable(start) != _comparable(before)
                    and _same(what, start, after)):
                fixes.append({'rule': '开镜值取变化前', 'name': name, 'what': what, 'value': deepcopy(start), 'to': deepcopy(before)})
                entry[what] = deepcopy(before)
                start = before
            current[(name, what)] = start
        elif current[(name, what)] is not None and _comparable(current[(name, what)]) == _comparable(after):
            fixes.append({'rule': '删掉没有变化的变化', 'name': name, 'what': what, 'value': deepcopy(change)})
            continue
        current[(name, what)] = after
        kept.append(change)
    action = shot.get('action') if isinstance(shot.get('action'), str) else ''
    for name, entry in entries.items():
        posture = entry.get('posture')
        stem = posture_stem(posture)
        if (stem not in BODY_POSTURE_TRANSITIONS or (name, 'posture') in current or name not in previous_postures
                or not _transition_for(action, stem, name)
                or posture_stem(previous_postures[name]) == stem):
            continue
        earlier = previous_postures[name]
        before = earlier if posture_stem(earlier) in BODY_POSTURE_TRANSITIONS else None
        del entry['posture']
        kept.append({'name': name, 'what': 'posture', 'before': before, 'after': posture})
        fixes.append({'rule': '动作暗含的姿态变化', 'name': name, 'what': 'posture', 'value': posture, 'to': before})
    shot['changes'] = kept
    return fixes


def align_board_openings(board, outfits) -> list:
    """按镜序逐镜对齐开镜值，给每镜「各人在前面最近一次出场的镜末姿态」；返回带镜号的修补记录。"""
    previous, fixes = {}, []
    for shot in board['shots']:
        fixes += [{**fix, 'shot': shot['shot']} for fix in align_opening_with_changes(shot, outfits, dict(previous))]
        previous.update(closing_postures(shot))
    return fixes


def carry_holding(board) -> list[dict]:
    """规划路径的持物延续，两件事：①人物在场条目没写 ``holding`` 时，沿用他上一次出场的镜末持物（开镜持物走完那一镜的
    持物变化），写了的照写，空列表是写明了空手；②一条持物变化把人的手放空、本镜又没写他的状态变化时，补一条状态
    变化 ``empty-handed``。可见结果合同要求放空要有可见状态，否则整条交出被当错误丢掉，交接就从文字里没了。
    被持有的实体补进在场由 :func:`_complete_storyboard_holding_presence` 做。
    标注修复用的 :func:`reconcile_holding`（按动作句解析交接、由变化前补开镜等）不再用于
    规划原答，那些条目在公共 9B 159 份计划上合计约 40 处、没有证据说明有用。原地修改，返回带镜号的记录。"""
    characters = {entry['name'] for entry in board['characters']}
    closing, fixes = {}, []
    for shot in board['shots']:
        for entry in shot['present']:
            name = entry['name']
            if name in characters and 'holding' not in entry and closing.get(name):
                entry['holding'] = list(closing[name])
                fixes.append({'rule': 'holding_carried', 'shot': shot['shot'], 'name': name, 'after': list(closing[name])})
        for change in list(shot.get('changes') or []):
            name = change.get('name')
            if (name in characters and change.get('what') == 'holding' and change.get('before') and change.get('after') == []
                    and not any(c.get('name') == name and c.get('what') == 'state' for c in shot['changes'])):
                shot['changes'].append({'name': name, 'what': 'state', 'after': 'empty-handed'})
                fixes.append({'rule': 'empty_hands_made_visible', 'shot': shot['shot'], 'name': name})
        for entry in shot['present']:
            if entry['name'] in characters:
                held = list(entry.get('holding') or [])
                for change in shot.get('changes') or []:
                    if change.get('name') == entry['name'] and change.get('what') == 'holding' and isinstance(change.get('after'), list):
                        held = list(change['after'])
                closing[entry['name']] = held
    return fixes


def drop_unchanged_changes(board) -> list[dict]:
    """删掉不改变任何东西的变化：``before`` 与 ``after`` 相同；没写 ``before`` 而开镜值已等于 ``after``；同一实体同一属性
    在本镜里又变回当时的值，包括把同一条变化写两遍（11-14 实见 v19／v20 补充样本 film-008726 第 3 镜衣着变化重复，
    计划算完成、逐镜文字编译报错）。逐镜文字的镜末句只写真变化，留着这类条目编译会报错。开镜值没写衣着时按登记衣着算。
    规划路径只用这一条，标注路径的 :func:`align_board_openings` 不变。原地修改，返回带镜号的记录。"""
    outfits = {entry['name']: entry.get('outfit') for entry in board['characters']}
    fixes = []
    for shot in board['shots']:
        entries = {entry['name']: entry for entry in shot['present']}
        current, changed, kept = {}, set(), []
        for change in shot.get('changes') or []:
            name, what, before, after = change['name'], change['what'], change.get('before'), change['after']
            if name not in entries:
                kept.append(change)
                continue
            if (name, what) not in current:
                current[(name, what)] = entries[name].get(what, outfits.get(name) if what == 'outfit' else None)
            now = current[(name, what)]
            if ((not _unstated(before) and _comparable(before) == _comparable(after))
                    or ((_unstated(before) or (name, what) in changed) and now is not None
                        and _comparable(now) == _comparable(after))):
                fixes.append({'rule': 'unchanged_change_dropped', 'shot': shot['shot'], 'value': deepcopy(change)})
                continue
            current[(name, what)] = after
            changed.add((name, what))
            kept.append(change)
        shot['changes'] = kept
    return fixes


def withhold_contested_openings(board) -> list[dict]:
    """规划原答里开镜值写在两处：在场条目，与这一镜第一条同属性变化的 ``before``。在场条目已写成变化后的值（与 ``after``
    相同）而 ``before`` 另写一个值时，两处自相矛盾。标注的 ``before`` 来自看片，由 :func:`align_board_openings` 取作开镜值；
    规划器的 ``before`` 是它自己猜的，常是请求没有的变化（11-12 公共 Qwen 逐镜档 33 处取用里约 9 处与请求相反，如「坐在
    后座」被写成 standing → sitting，开镜成了「站着」）。编译器不替规划器选边：姿态与状态两项删掉在场条目里的值和
    这条变化的 ``before``（逐镜文字的起态取自首个明确 ``before``），开镜不写，变化的结果与动作句照写。衣着与持物各有代码维护（:func:`resolve_outfits`、:func:`reconcile_holding`），不在此列。
    只在规划路径调用，原地修改，返回带镜号的记录。"""
    fixes = []
    for shot in board['shots']:
        entries = {entry['name']: entry for entry in shot.get('present') or [] if isinstance(entry, dict)}
        seen = set()
        for change in shot.get('changes') or []:
            if not (isinstance(change, dict) and change.get('name') in entries and change.get('what') in ('posture', 'state')
                    and isinstance(change.get('after'), str)):
                continue
            name, what, before, after = change['name'], change['what'], change.get('before'), change['after']
            if (name, what) in seen:
                continue
            seen.add((name, what))
            entry = entries[name]
            start = entry.get(what)
            if (isinstance(start, str) and not _unstated(before) and _comparable(start) != _comparable(before)
                    and _same(what, start, after)):
                fixes.append({'rule': 'opening_withheld_planner_disagrees', 'shot': shot['shot'], 'name': name,
                              'what': what, 'present': entry.pop(what), 'before': change.pop('before'), 'after': after})
    return fixes


def resolve_outfits(board) -> list[dict]:
    """分镜表维护的衣着一项：衣着沿当前值由代码延续，提示词编译与选帧只读这里写进在场条目的结果。

    依据：11-10 规划器复查实测（换装后沿用当前衣着，看图改对 34、改错 6）。按镜序逐人：
    没写衣着而当前衣着已不是登记默认（前面有换装事件，或此前明写过新衣着），写进当前值；写回换装前的旧衣着、且本镜
    没有以它为 before 的换装事件，改成当前值；写了与当前值、旧值都不同的新衣着而没有换装事件，按写明的为准并往后
    延续。本镜的换装事件按顺序推进当前值。原地修改，返回带镜号的修补记录。"""
    registry = {entry['name']: entry.get('outfit') for entry in board['characters']}
    current, superseded, seen = dict(registry), {name: [] for name in registry}, set()
    fixes = []
    for i, shot in enumerate(board['shots']):
        number = shot['shot']
        changes = [change for change in shot.get('changes') or []
                   if change.get('what') == 'outfit' and change.get('name') in registry]
        for j, entry in enumerate(shot['present']):
            name = entry['name']
            if name not in registry:
                continue
            path, written, first = f'shots[{i}].present[{j}].outfit', entry.get('outfit'), name not in seen
            seen.add(name)
            if written is None:
                if _comparable(current[name]) != _comparable(registry[name]):
                    entry['outfit'] = deepcopy(current[name])
                    fixes.append({'rule': 'outfit_carried', 'shot': number, 'name': name, 'path': path,
                                  'after': deepcopy(current[name])})
                continue
            if _comparable(written) == _comparable(current[name]):
                continue
            opens_change = any(change['name'] == name and _comparable(change.get('before')) == _comparable(written)
                               for change in changes)
            if not first and not opens_change and any(_comparable(written) == _comparable(old) for old in superseded[name]):
                entry['outfit'] = deepcopy(current[name])
                fixes.append({'rule': 'outfit_reverted_without_event', 'shot': number, 'name': name, 'path': path,
                              'before': deepcopy(written), 'after': deepcopy(current[name])})
                continue
            superseded[name].append(current[name])
            current[name] = written
        for change in changes:
            if isinstance(change.get('after'), str):
                superseded[change['name']].append(current[change['name']])
                current[change['name']] = change['after']
    return fixes


def _complete_storyboard_holding_presence(board):
    known = {entry['name']: kind for kind in REGISTRIES for entry in board[kind]}
    added = []
    for shot in board['shots']:
        names = complete_holding_presence(shot, known)
        if names:
            added.append({'shot': shot['shot'], 'names': names})
    return added


_HOLDING_HANDOFF = re.compile(
    r'\b(?:hands?|handed|passes?|passed|gives?|gave|transfers?|transferred)\s+(?:over\s+)?'
    r'(?:it|them|Object [1-9]\d*|(?:the|a|an) [^.;!?]{1,80}?)\s+to\b'
    r'|\b(?:receives?|received)\s+(?:it|them|Object [1-9]\d*|(?:the|a|an) [^.;!?]{1,80}?)\s+from\b', re.I)
_HOLDING_RELEASE = re.compile(r'\b(?:puts?|put|sets?|set|places?|placed|drops?|dropped|releases?|released)\b', re.I)
_HOLDING_TRANSFER = re.compile(
    r'\b(Character [1-9]\d*)(?:,[^,.;!?]+,)?\s+(?:hands?|handed|passes?|passed|gives?|gave)\s+'
    r'(Object [1-9]\d*)\s+to\s+(Character [1-9]\d*)\b', re.I)
_HOLDING_ACQUIRE = re.compile(
    r'\b(Character [1-9]\d*)(?:,[^,.;!?]+,)?\s+(?:picks?\s+up|picked\s+up|takes?|took|grabs?|grabbed)\s+'
    r'(Object [1-9]\d*)\b(?!\s+from\b)', re.I)
_HOLDING_PUT_DOWN = re.compile(
    r'\b(Character [1-9]\d*)(?:,[^,.;!?]+,)?\s+(?:drops?|dropped|releases?|released)\s+'
    r'(Object [1-9]\d*)\b', re.I)
_HOLDING_PLACE = re.compile(
    r'\b(Character [1-9]\d*)(?:,[^,.;!?]+,)?\s+(?:puts?|put|sets?|set|places?|placed)\s+'
    r'(Object [1-9]\d*)\s+(?:down\b|on(?:to)?\s+(?:Object|Location) [1-9]\d*\b)', re.I)
_EMPTY_HANDED = re.compile(r'\bempty[- ]handed\b|\bhands? (?:are )?empty\b', re.I)
_HOLDING_UNBOUND_ACTION = re.compile(
    r'\b(?:hands?|handed|passes?|passed|gives?|gave|transfers?|transferred|receives?|received|picks?\s+up|picked\s+up|'
    r'takes?|took|grabs?|grabbed|puts?|put|sets?|set|places?|placed|drops?|dropped|releases?|released)\s+'
    r'(?:it\b|them\b|Object [1-9]\d*\b|(?:the|a|an) (?!seat\b|breath\b|look\b|step\b|walk\b|hands?\b)[a-z]+\b)', re.I)


def _holding_handoffs(action):
    """只排除明确的习语／名词误命中；一般自然语言交接仍交给来源复核。"""
    matches = []
    for match in _HOLDING_HANDOFF.finditer(action):
        phrase = action[match.start():]
        if re.match(r'(?:gives?|gave)\s+(?:a|the)\s+(?:thumbs?[- ]up|speech|command|massage)\b', phrase, re.I):
            continue
        if re.match(r'hands?\b', phrase, re.I) and re.search(
                r'\b(?:her|his|their|its|your|my|our)\s+$', action[:match.start()], re.I):
            continue
        if re.fullmatch(r'pass(?:es|ed)?\s+(?:the|a|an)\s+[^.;!?]+,\s+and\s+exits?\s+the\s+frame\s+to', match.group(), re.I) and re.match(
                r'\s+(?:the\s+)?(?:right|left|front|back)\b', action[match.end():], re.I):
            continue
        matches.append(match)
    return matches


def _holding_unbound_events(action, direct):
    """同一明确交接后的 who takes it 是收取方复述，不是第二次取得。"""
    receipts = []
    if len(direct) == 1:
        transfer = direct[0]
        receipt = re.match(
            r',\s*who\s+(?:then\s+)?(?:(?:sits?\s+up|reaches?\s+out)\s+(?:and|to)\s+)?'
            r'(?:takes?|took|receives?|received)\s+(?:it|them)\b', action[transfer.end():], re.I)
        if receipt:
            receipts.append((transfer.end(), transfer.end() + receipt.end()))
    matches = []
    for match in _HOLDING_UNBOUND_ACTION.finditer(action):
        phrase = action[match.start():]
        if re.match(r'(?:gives?|gave)\s+(?:a|the)\s+(?:thumbs?[- ]up|speech|command|massage)\b', phrase, re.I):
            continue
        if re.match(r'(?:takes?|took)\s+a\s+sip\b', phrase, re.I):
            continue
        if re.match(r'(?:takes?|took)\s+a\s+drink\b', phrase, re.I):
            container = r'(?:cup|glass|can|bottle)'
            drinking_from = re.match(
                rf'(?:takes?|took)\s+a\s+drink\s+from\s+(?:a|the|his|her)\s+'
                rf'[^,.;!?]{{0,40}}\b{container}\b', phrase, re.I)
            just_picked_up = re.search(
                rf'\b(?:picks?\s+up|picked\s+up)\s+(?:(?!\bCharacter\b)[^.;!?]){{1,100}}'
                rf'\b{container}\b(?:\s+of\s+[\w ]{{1,30}})?\s+and\s+$', action[:match.start()], re.I)
            if drinking_from or just_picked_up:
                continue
        if re.match(r'(?:puts?|put|places?|placed)\s+(?:it|Object [1-9]\d*)\s+to\s+(?:his|her|their|the)\s+ear\b', phrase, re.I):
            continue
        if any(start <= match.start() and match.end() <= end for start, end in receipts):
            continue
        matches.append(match)
    return matches


def _holding_action_review(action):
    """不能凭无序集合确定动作终点的情况；只拒绝自动补值，保留原有结构端点。"""
    bound = [match for pattern in (_HOLDING_TRANSFER, _HOLDING_ACQUIRE, _HOLDING_PUT_DOWN, _HOLDING_PLACE)
             for match in pattern.finditer(action)]
    spans = {match.span() for match in bound}
    direct = list(_HOLDING_TRANSFER.finditer(action))
    handoffs = _holding_handoffs(action)
    spans.update(match.span() for match in [*handoffs, *_holding_unbound_events(action, direct)])
    events = [span for span in spans if not any(other != span and other[0] <= span[0] and span[1] <= other[1] for other in spans)]
    if not events:
        return []
    reasons = []
    if re.search(r"\b(?:not|never|without|no)\b|n't\b", action, re.I):
        reasons.append('negated_holding_action')
    if handoffs and re.search(r'\boff[- ]screen\b|\bsomeone\b|\bsomebody\b', action, re.I):
        reasons.append('unbound_handoff_participant')
    if len(events) > 1 and re.search(r'\b(?:then|before|after|until|finally)\b', action, re.I):
        reasons.append('multiple_holding_events_with_temporal_order')
    if len(handoffs) > max(1, len(direct)):
        reasons.append('multiple_handoffs_not_fully_bound')
    if handoffs and not direct and NAME.search(action):
        reasons.append('numbered_handoff_not_fully_bound')
    parsed = [match[2] for match in direct]
    parsed += [match[2] for pattern in (_HOLDING_ACQUIRE, _HOLDING_PUT_DOWN, _HOLDING_PLACE)
               for match in pattern.finditer(action)]
    if any(parsed.count(obj) > 1 for obj in parsed):
        reasons.append('multiple_holding_events_for_one_entity')
    if direct and len(events) > len(parsed):
        reasons.append('additional_holding_event_not_bound')
    return reasons


def reconcile_holding(board) -> list[dict]:
    """补全可机械确认的持物集合，原地修改并返回逐项证据；不读取未来镜。

    before/after 都是完整集合。明确编号动作，或动作中的交接词加唯一的结构流向，才能补交接双方和保留旧持物。
    未绑定的放下动作、多次同物交接和一般 before 冲突不猜；原答与冻结数据由调用方保留。本函数可重复调用。
    """
    kinds = {entry['name']: entry.get('kind') for entry in board.get('characters', [])}
    known = {*kinds, *(entry['name'] for entry in board.get('objects', []))}
    previous, fixes, presence_added = {}, [], []

    def put(entry, key, value, *, shot, name, path, rule, evidence):
        if entry.get(key) == value:
            return
        fixes.append({'rule': rule, 'shot': shot, 'name': name, 'path': path,
                      'before': deepcopy(entry.get(key)), 'after': deepcopy(value), 'evidence': evidence})
        entry[key] = deepcopy(value)

    for i, shot in enumerate(board.get('shots', [])):
        number = shot.get('shot', i + 1)
        action = shot.get('action') or ''
        action_review = _holding_action_review(action)
        handoffs = _holding_handoffs(action)
        known_types = {name: 'characters' for name in kinds}
        known_types.update({entry['name']: 'objects' for entry in board.get('objects', [])})
        added = complete_holding_presence(shot, known_types)
        bound_transfers = list(_HOLDING_TRANSFER.finditer(action))
        direct = [] if action_review else bound_transfers
        entries, changes, paths = {}, {}, {}
        # present 可因持有的 Character 新增；列表增长时继续消费它的历史，而非下次调用才补。
        for j, entry in enumerate(shot.get('present', [])):
            name = entry['name']
            if name not in kinds:
                continue
            entries[name] = entry
            paths[name] = f'shots[{i}].present[{j}].holding'
            changes[name] = [change for change in shot.get('changes', [])
                             if change.get('name') == name and change.get('what') == 'holding'
                             and isinstance(change.get('before'), list) and isinstance(change.get('after'), list)]
            first = changes[name][0] if changes[name] else None
            start = entry.get('holding')
            if first and _comparable(start) != _comparable(first['before']):
                held = set(start) if isinstance(start, list) else set()
                concurrent = (len(bound_transfers) == 1 and bound_transfers[0][3] == name and bool(held)
                              and held <= set(first['after']) and all(re.search(
                                  rf'\b(?:while|still)\s+holding\s+{re.escape(obj)}\b', action, re.I) for obj in held))
                prior = bool(held) and _comparable(previous.get(name)) == _comparable(start)
                evidence = {'opening': deepcopy(start), 'change_before': deepcopy(first['before']),
                            'previous_closing': deepcopy(previous.get(name)),
                            'source_conflict_requires_review': isinstance(start, list) and not (concurrent or prior)}
                if concurrent or prior:
                    k = shot['changes'].index(first)
                    put(first, 'before', start, shot=number, name=name, path=f'shots[{i}].changes[{k}].before',
                        rule='holding_before_from_established_opening',
                        evidence={**evidence, 'basis': 'explicit_concurrent_holding' if concurrent else 'previous_closing'})
                else:
                    # 与 shot_text._opening_shot 同一时间合同；原冲突保留在证据，不称源语义已验证。
                    put(entry, 'holding', first['before'], shot=number, name=name, path=paths[name],
                        rule='holding_opening_from_before', evidence={**evidence, 'basis': 'declared_change_before'})
            elif 'holding' not in entry and kinds[name] == 'person' and _EMPTY_HANDED.search(entry.get('state') or ''):
                put(entry, 'holding', [], shot=number, name=name, path=paths[name],
                    rule='holding_opening_from_empty_state', evidence={'state': entry['state']})
            elif not action_review and 'holding' not in entry and previous.get(name):
                put(entry, 'holding', previous[name], shot=number, name=name, path=paths[name],
                    rule='holding_opening_continuation', evidence={'previous_closing': previous[name]})
            # 单一、完整绑定的交出动作要求给者在交接前持有该物；不能把原错误 after 当取得。
            if len(direct) == 1 and direct[0][1] == name and direct[0][2] in known:
                obj = direct[0][2]
                before = list(entry.get('holding') or [])
                if obj not in before:
                    before.append(obj)
                evidence = {'action': direct[0].group(), 'basis': 'single_bound_transfer'}
                put(entry, 'holding', before, shot=number, name=name, path=paths[name],
                    rule='holding_opening_from_transfer', evidence=evidence)
                if first:
                    k = shot['changes'].index(first)
                    put(first, 'before', before, shot=number, name=name, path=f'shots[{i}].changes[{k}].before',
                        rule='holding_before_from_transfer', evidence=evidence)
            if entry.get('holding') and isinstance(entry.get('state'), str) and _EMPTY_HANDED.search(entry['state']):
                state = entry['state']
                complement = re.sub(r'\b(?:empty[- ]handed|(?:with\s+)?(?:(?:her|his|their|its)\s+)?hands?\s+(?:are\s+)?empty)\b', '', state, flags=re.I)
                complement = re.sub(r'\band\s+and\b', 'and', complement)
                complement = re.sub(r',\s*,', ',', complement)
                complement = re.sub(r'^\s*(?:and\b|[,;])\s*|\s*(?:\band|[,;])\s*$', '', complement).strip()
                if complement:
                    put(entry, 'state', complement, shot=number, name=name, path=paths[name].removesuffix('holding') + 'state',
                        rule='holding_opening_empty_state_conflict', evidence={'holding': entry['holding']})
                else:
                    fixes.append({'rule': 'holding_opening_empty_state_conflict', 'shot': number, 'name': name,
                                  'path': paths[name].removesuffix('holding') + 'state', 'before': entry.pop('state'),
                                  'after': None, 'evidence': {'holding': entry['holding']}})
            added.extend(complete_holding_presence(shot, known_types))
        opening = {name: list(entry['holding']) for name, entry in entries.items() if isinstance(entry.get('holding'), list)}
        transfers = []
        for match in (() if action_review else _HOLDING_TRANSFER.finditer(action)):
            source, obj, recipient = match.groups()
            if source in entries and recipient in entries and obj in known and source != recipient:
                transfers.append((source, obj, recipient, match.group()))
        if not action_review and len(handoffs) == 1:
            gains = [(name, obj) for name, values in changes.items() if len(values) == 1
                     for obj in set(values[0]['after']) - set(values[0]['before'])]
            for recipient, obj in gains:
                if any(obj == transfer[1] for transfer in transfers):
                    continue
                sources = [name for name, values in opening.items() if name != recipient and obj in values
                           and (not changes[name] or len(changes[name]) == 1 and obj not in changes[name][0]['after'])]
                if len(sources) == 1 and sum(other == obj for _, other in gains) == 1:
                    transfers.append((sources[0], obj, recipient, 'handoff verb and unique structured holding flow'))
        # 连续同物转交需要逐事件时序，不能把多个局部片段混成一个镜末结果。
        transfers = [t for t in transfers if sum(other[1] == t[1] for other in transfers) == 1]
        add, remove, evidence = {}, {}, {}

        def delta(name, obj, gain, reason):
            (add if gain else remove).setdefault(name, set()).add(obj)
            evidence.setdefault(name, []).append(reason)

        for source, obj, recipient, reason in transfers:
            if obj not in opening.get(source, []):
                # 明示空集合与交接动作矛盾时不把持有时点倒灌到开镜。
                if source in opening:
                    continue
                put(entries[source], 'holding', [obj], shot=number, name=source, path=paths[source],
                    rule='holding_opening_from_transfer', evidence={'action': reason})
                opening[source] = [obj]
            if obj in opening.get(recipient, []):
                continue
            delta(source, obj, False, reason)
            delta(recipient, obj, True, reason)
        for pattern, gain in ((_HOLDING_ACQUIRE, True), (_HOLDING_PUT_DOWN, False), (_HOLDING_PLACE, False)):
            for match in (() if action_review else pattern.finditer(action)):
                name, obj = match.groups()
                if name in entries and obj in known and not any(obj == t[1] for t in transfers):
                    delta(name, obj, gain, match.group())
        for name in dict.fromkeys([*add, *remove]):
            values = changes[name]
            if len(values) > 1:
                continue
            before = opening.get(name, [])
            if values and set(values[0]['before']) != set(before):
                continue
            after = values[0]['after'] if values else before
            retained = set(before)
            # 自然语言还说了放下某物而没有编号绑定时，不能断言旧物仍在手中。
            if _HOLDING_RELEASE.search(action) and not remove.get(name):
                retained &= set(after)
            complete = (retained | set(after) | add.get(name, set())) - remove.get(name, set())
            complete = list(dict.fromkeys([obj for obj in [*before, *after, *sorted(add.get(name, set()))] if obj in complete]))
            if name not in opening:
                put(entries[name], 'holding', before, shot=number, name=name, path=paths[name],
                    rule='holding_opening_for_action', evidence={'action': evidence[name]})
                opening[name] = before
            if values:
                change = values[0]
                j = shot['changes'].index(change)
                put(change, 'after', complete, shot=number, name=name, path=f'shots[{i}].changes[{j}].after',
                    rule='holding_complete_ending_set', evidence={'action': evidence[name], 'opening': before})
            elif set(before) != set(complete):
                change = {'name': name, 'what': 'holding', 'before': list(before), 'after': complete}
                j = len(shot['changes'])
                shot['changes'].append(change)
                changes[name].append(change)
                fixes.append({'rule': 'holding_missing_action_endpoint', 'shot': number, 'name': name,
                              'path': f'shots[{i}].changes[{j}]', 'before': None, 'after': deepcopy(change),
                              'evidence': {'action': evidence[name]}})
        for name, values in changes.items():
            if not values:
                continue
            last = values[-1]
            if last['before'] and last['after'] == [] and kinds[name] == 'person':
                states = [(j, c) for j, c in enumerate(shot['changes']) if c.get('name') == name and c.get('what') == 'state']
                j, state = states[-1] if states else (len(shot['changes']), None)
                visible = state.get('after') if state else ''
                if not isinstance(visible, str):
                    continue  # 原有错误值仍由类型/字段边界处理，不能转成貌似合法的字符串。
                if not _EMPTY_HANDED.search(visible):
                    ending = f'{visible} and empty-handed' if visible else 'empty-handed'
                    if state:
                        put(state, 'after', ending, shot=number, name=name, path=f'shots[{i}].changes[{j}].after',
                            rule='holding_empty_visible_result', evidence={'holding_before': last['before'], 'holding_after': []})
                    else:
                        state = {'name': name, 'what': 'state', 'after': ending}
                        shot['changes'].append(state)
                        fixes.append({'rule': 'holding_empty_visible_result', 'shot': number, 'name': name,
                                      'path': f'shots[{i}].changes[{j}]', 'before': None, 'after': deepcopy(state),
                                      'evidence': {'holding_before': last['before'], 'holding_after': []}})
        if action_review:
            for name in entries:
                previous.pop(name, None)
        else:
            previous.update(opening)
        for name, values in changes.items():
            if values:
                previous[name] = list(values[-1]['after'])
        added.extend(complete_holding_presence(shot, known_types))
        if added:
            presence_added.append({'rule': 'holding_present_entity_added', 'shot': number, 'name': None,
                                   'path': f'shots[{i}].present', 'before': None,
                                   'after': [{'name': name} for name in added], 'evidence': {'holding_targets': added}})
    fixes.extend(presence_added)
    return [{**fix, 'category': 'deterministic_repair'} for fix in fixes]


def holding_consistency_issues(board) -> list[dict]:
    """只读检查显式开镜/变化链与可确定的集合修补；未绑定集合缩水单列为待复核。"""
    issues = []
    proposed = reconcile_holding(deepcopy(board))
    for i, shot in enumerate(board.get('shots', [])):
        if reasons := _holding_action_review(shot.get('action') or ''):
            issues.append({'rule': 'holding_action_requires_review', 'shot': shot.get('shot', i + 1),
                           'name': None, 'path': f'shots[{i}].action', 'action': shot.get('action'),
                           'reasons': reasons, 'repairable': False, 'category': 'semantic_coverage_candidate'})
        current = {entry['name']: entry['holding'] for entry in shot.get('present', []) if isinstance(entry.get('holding'), list)}
        for j, change in enumerate(shot.get('changes', [])):
            if change.get('what') != 'holding' or not isinstance(change.get('before'), list) or not isinstance(change.get('after'), list):
                continue
            name, before, after = change['name'], change['before'], change['after']
            if name in current and set(current[name]) != set(before):
                issues.append({'rule': 'holding_before_mismatch', 'shot': shot.get('shot', i + 1), 'name': name,
                               'path': f'shots[{i}].changes[{j}].before', 'before': deepcopy(before),
                               'expected': deepcopy(current[name]), 'category': 'explicit_structural_conflict'})
            removed, added = set(before) - set(after), set(after) - set(before)
            if (removed and added and not _HOLDING_RELEASE.search(shot.get('action') or '')
                    and not any(fix['shot'] == shot.get('shot', i + 1) and fix['name'] == name
                                and fix['rule'] == 'holding_complete_ending_set' for fix in proposed)):
                issues.append({'rule': 'holding_ending_set_requires_review', 'shot': shot.get('shot', i + 1),
                               'name': name, 'path': f'shots[{i}].changes[{j}].after',
                               'before': deepcopy(before), 'after': deepcopy(after),
                               'removed': sorted(removed), 'added': sorted(added), 'repairable': False,
                               'category': 'collection_completeness_candidate'})
            current[name] = after
    return issues + proposed



# 写法检查属于生产边界；不在既有分镜表读取、SFT encoder 或 shot_text 中改写旧文本。
# 这些是明确结构信号，不是自然语言语义判定器；命中拒收后由源帧审阅裁决。
NEGATIVE_RESULT_FORM = re.compile(
    r"\b(?:without|no\s+longer|not\s+(?:wearing|holding|carrying))\b"
    r"|(?<![/\w])no\s+\w|\b(?:removed|missing|lacking)\b", re.IGNORECASE)
RESULT_SENTENCE_START = re.compile(
    r"^(?:(?:Character|Object)\s+\d+\b|(?:he|she|it|they|is|are|was|were)\s"
    r"|(?:his|her|its|their)\b.+\b(?:is|are)\b)", re.IGNORECASE)


def visible_result_wording_errors(board):
    """返回原字段位置与拒收原因；不改标注，不判断画面事实。"""
    errors = []
    def entries(value):
        return value if isinstance(value, list) else []  # 类型错误由已有结构合同拒收。

    def check(value, path, shot, complement):
        if not isinstance(value, str) or not value.strip():
            return  # 缺项与类型归已有结构合同；这里不猜补。
        text = value.strip()
        # 实际界面文字是正面内容；只在匹配视图中排除，原字段逐字保留。
        view = text
        if re.search(r'\b(?:displaying|showing|text|sign|buttons)\b', text, re.IGNORECASE):
            view = re.sub(r'"[^"\n]*"|(?<!\w)\'[^\'\n]*\'', '', view)
        view = re.sub(r'\byes/no\s+buttons\b|\bno signal\s+screen\b|\bmissing person\s+poster\b', '', view, flags=re.IGNORECASE)
        if NEGATIVE_RESULT_FORM.search(view):
            errors.append({'shot': shot, 'path': path, 'reason': 'use a positive visible result; source review required'})
        if complement and (RESULT_SENTENCE_START.search(text) or text[0].isupper() or text.endswith(('.', '!', '?'))):
            errors.append({'shot': shot, 'path': path, 'reason': 'expected a lowercase complement after is, not a full sentence'})

    for i, entry in enumerate(entries(board.get('characters'))):
        if isinstance(entry, dict):
            check(entry.get('outfit'), f'characters[{i}].outfit', None, False)
    for i, shot in enumerate(entries(board.get('shots'))):
        if not isinstance(shot, dict):
            continue
        number = i + 1
        for j, entry in enumerate(entries(shot.get('present'))):
            if not isinstance(entry, dict):
                continue
            for field in ('state', 'posture', 'outfit'):
                check(entry.get(field), f'shots[{i}].present[{j}].{field}', number, field != 'outfit')
        for j, change in enumerate(entries(shot.get('changes'))):
            if not isinstance(change, dict) or change.get('what') not in ('state', 'posture', 'outfit'):
                continue
            for side in ('before', 'after'):
                check(change.get(side), f'shots[{i}].changes[{j}].{side}', number, change['what'] != 'outfit')

        # holding=[]是关系结果；人、动物、物体不能共用“empty-handed”的可见推断。
        # 显式可见终态由生产者给出；排版器不猜器官、不复述被放下物件。
        for j, change in enumerate(entries(shot.get('changes'))):
            if not isinstance(change, dict) or change.get('what') != 'holding':
                continue
            if not isinstance(change.get('before'), list) or not change['before'] or change.get('after') != []:
                continue
            ending = [c for c in entries(shot.get('changes')) if isinstance(c, dict)
                      and c.get('name') == change.get('name') and c.get('what') == 'state'
                      and isinstance(c.get('after'), str) and c['after'].strip()]
            if not ending:
                errors.append({'shot': number, 'path': f'shots[{i}].changes[{j}]',
                               'reason': 'holding becomes empty: provide an explicit visible state change for this entity'})
    return errors


def require_visible_result_wording(board):
    """标注目标导出／规划原答的生产拒收点；错误保留字段位置。"""
    errors = visible_result_wording_errors(board)
    if errors:
        raise ValueError(f"visible result wording requires source review: {errors}")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Character(StrictModel):
    name: StrictStr = Field(pattern=r'^Character [1-9][0-9]*$')
    kind: Literal['person', 'animal', 'group']
    visibility: Literal['clear', 'partial', 'distant'] | None = None
    appearance: StrictStr
    outfit: StrictStr = ''


class Object(StrictModel):
    name: StrictStr = Field(pattern=r'^Object [1-9][0-9]*$')
    appearance: StrictStr


class Place(StrictModel):
    name: StrictStr = Field(pattern=r'^Location [1-9][0-9]*$')
    appearance: StrictStr


class Camera(StrictModel):
    shot_size: Literal['extreme close-up shot', 'close-up shot', 'medium close-up shot', 'medium shot',
                       'medium wide shot', 'wide shot', 'extreme wide shot'] | None = None
    angle: Literal['eye-level shot', 'high angle shot', 'low angle shot', 'overhead shot',
                   'over-the-shoulder shot', 'dutch angle shot', 'aerial shot'] | None = None
    movement: Literal['static', 'pan', 'tilt up', 'tilt down', 'push in', 'pull out', 'tracking', 'handheld'] | None = None


CAMERA_WORDS = {value: field for field in ('shot_size', 'angle', 'movement')
                for value in get_args(get_args(Camera.model_fields[field].annotation)[0])}
_SUBJECT_LINE = re.compile(r'^- (Character [1-9][0-9]*) \((person|animal|group)\): (.+)$')
_OBJECT_LINE = re.compile(r'^- (Object [1-9][0-9]*) \(object\): (.+)$')
_PLACE_LINE = re.compile(r'^- (Location [1-9][0-9]*): (.+)$')
_SHOT_LINE = re.compile(r'^s([0-9]+): (.+)$')


def parse_shot_script(script: str) -> dict:
    """逐镜档请求（``shot_text.shot_script_text`` 排出的 SUBJECTS／LOCATIONS／SHOTS 三段，前面可以有制作要求）→ 请求写明的值。

    登记一行是"外貌, wearing 衣着"：按最后一个 ``, wearing `` 切开。基准 486 个人物里 483 个与给定分镜表逐字相同；其余 3 个
    （衣着里又有 wearing、赤膊不带 wearing、外貌里带 wearing 而没有衣着）词与语序不变，只是外貌与衣着的分界不同，重新排出的
    请求逐字相同。机位只有非缺省的写在请求里（``shot_text.DEFAULT_CAMERA`` 的平视、静止不写），缺省的不算请求写明。"""
    lines = script.splitlines()
    if 'SUBJECTS' not in lines:
        raise ValueError('not a shot-tier request: no SUBJECTS section')
    stated = {'characters': {}, 'objects': {}, 'places': {}, 'shots': []}
    section = None
    for line in lines[lines.index('SUBJECTS'):]:
        if line in ('SUBJECTS', 'LOCATIONS', 'SHOTS'):
            section = line
            continue
        if section == 'SUBJECTS' and (match := _SUBJECT_LINE.match(line)):
            appearance, sep, outfit = match[3].rpartition(', wearing ')
            stated['characters'][match[1]] = {'kind': match[2], 'appearance': appearance if sep else match[3],
                                              'outfit': outfit if sep else None}
        elif section == 'SUBJECTS' and (match := _OBJECT_LINE.match(line)):
            stated['objects'][match[1]] = match[2]
        elif section == 'LOCATIONS' and (match := _PLACE_LINE.match(line)):
            stated['places'][match[1]] = match[2]
        elif section == 'SHOTS' and (match := _SHOT_LINE.match(line)):
            if int(match[1]) != len(stated['shots']) + 1:
                raise ValueError(f'shot lines out of order at s{match[1]}')
            head, sep, rest = match[2].partition('. ')
            words = [w.strip() for w in head.split(',')] if sep else []
            fields = [CAMERA_WORDS.get(w[:1].lower() + w[1:]) for w in words]
            if words and all(fields) and len(set(fields)) == len(fields):
                camera = {field: w[:1].lower() + w[1:] for field, w in zip(fields, words)}
                action = rest
            else:
                camera, action = {}, match[2]
            stated['shots'].append({'camera': camera, 'action': action})
        else:
            raise ValueError(f'unexpected line in shot-tier request: {line!r}')
    if not stated['shots']:
        raise ValueError('shot-tier request has no shots')
    return stated


def copy_stated_values(board: dict, stated: Mapping[str, Any]) -> list[dict]:
    """逐镜档照写项由编译器拷回（三类规则：请求写明的不改不漏）：登记的类别、外貌与写了的衣着，
    物体与地点外貌，请求写明的机位。动作保留规划器的受约束扩写，不用请求原句覆写。
    请求有而计划没登记的实体、地点补进登记表；请求没写衣着的人物保留计划写的。
    镜数与请求不同时不动各镜（由镜数核对判失败）。就地改 ``board``，返回改了什么。"""
    copied = []

    def put(entry: dict, key: str, value: Any, path: str) -> None:
        if entry.get(key) != value:
            copied.append({'path': path, 'from': entry.get(key), 'to': value})
            entry[key] = value

    registries = {kind: {e['name']: e for e in board[kind]} for kind in ('characters', 'objects', 'places')}
    for name, value in stated['characters'].items():
        entry = registries['characters'].get(name)
        if entry is None:
            entry = {'name': name}
            board['characters'].append(entry)
        put(entry, 'kind', value['kind'], f'characters.{name}.kind')
        put(entry, 'appearance', value['appearance'], f'characters.{name}.appearance')
        if value['outfit'] is not None:
            put(entry, 'outfit', value['outfit'], f'characters.{name}.outfit')
    for kind in ('objects', 'places'):
        for name, appearance in stated[kind].items():
            entry = registries[kind].get(name)
            if entry is None:
                entry = {'name': name}
                board[kind].append(entry)
            put(entry, 'appearance', appearance, f'{kind}.{name}.appearance')
    if len(board['shots']) == len(stated['shots']):
        for index, (shot, value) in enumerate(zip(board['shots'], stated['shots'])):
            if value['camera']:
                camera = dict(shot.get('camera') or {})
                for field, word in value['camera'].items():
                    put(camera, field, word, f'shots[{index}].camera.{field}')
                shot['camera'] = camera
    return copied


class Present(StrictModel):
    name: EntityName
    outfit: StrictStr = ''
    posture: StrictStr = Field(default='', description='Lowercase positive visible complement after is; no subject, copula, or final period.')
    state: StrictStr = Field(default='', description='Positive visible result as a lowercase complement after is; do not name an absent item using without, removed, no longer, or not wearing.')
    holding: list[EntityName] = Field(default_factory=list, description=(
        "Registered entities physically held or carried, including people or animals; "
        "proximity, pointing, entering, sitting in, or riding an entity is not holding it."
    ))


class Change(StrictModel):
    name: EntityName = Field(description='The registered entity that changes, not a title for the change.')
    what: Literal['outfit', 'posture', 'state', 'holding']
    before: StrictStr | list[EntityName] | None = Field(default=None, description='Opening value: state/posture use positive complements; outfit uses clothing noun phrases; holding uses entity arrays.')
    after: StrictStr | list[EntityName] = Field(description='Ending value in the same property format; retain this result in later openings while true.')


COMPOSITIONS = ('center composition', 'balanced composition', 'symmetrical composition',
                'left-heavy composition', 'right-heavy composition')
FRAME_SIDES = ('left', 'center', 'right')
RELATION_ERROR = 'relation must name two different characters present in the shot'
PLANNER_BOXES_DROPPED = 'the planner writes composition, not person boxes'
PLANNER_RELATIONS_DROPPED = 'relative positions are not planned; positions the request states stay in the action'
PLANNER_REFERENCES_DROPPED = 'reference shots are derived from presence, not planned'
ENDING_SIDE_ERROR = 'ending side must name a character present in the shot, once'


class Relation(StrictModel):
    """镜首两个在场人物的相对位置（Wan 扩写写法里的自然短语，由编译器写出）。"""
    subject: EntityName
    relation: Literal['left_of', 'in_front_of'] = Field(
        description='left_of: subject is to the left of object in the frame; in_front_of: subject is closer to the camera.')
    object: EntityName


class EndingSide(StrictModel):
    """镜末换到画面另一侧的人物：镜末在左、中、右哪一侧。"""
    name: EntityName
    side: Literal['left', 'center', 'right']


class Shot(StrictModel):
    """标注分镜表的一镜。人物之间的相对位置 ``relations`` 由 SAM 人物框与 VGGT 深度推出，渲染器训练读它；规划器不写
    （:data:`PlannedShot`）。"""
    shot: StrictInt
    picture_type: Literal['live_action', 'animation_2d', 'animation_3d_or_game', 'screen_or_slides'] | None = Field(
        default=None, description='Only set when the user explicitly specifies the visual medium. Omit or null when unspecified; do not infer live action from silence.')
    place: StrictStr | None = Field(pattern=r'^Location [1-9][0-9]*$',
                                  description='Registered requested location, or null when the request does not establish one.')
    camera: Camera | None = None
    composition: Literal['center composition', 'balanced composition', 'symmetrical composition',
                         'left-heavy composition', 'right-heavy composition'] | None = Field(
        default=None, description='Opening-frame composition (Wan cinematic setting); only when the request states or implies the opening layout.')
    present: list[Present]
    relations: list[Relation] = Field(default_factory=list, description='Opening positions between present characters.')
    action: StrictStr = Field(description='Ordered visible action progression; preserve the requested performer, events, order and required result. Plausible intermediate steps may be inferred; use the performing registered subject (Character N or Object N as appropriate) and a short distinguishing feature. Do not invent a character or motion for an object-only or static shot.')
    changes: list[Change]
    ending_sides: list[EndingSide] = Field(default_factory=list, description='Characters who end the shot on another side of the frame.')
    frame_count: StrictInt = Field(gt=0,
        description='Suggested video frames for relative pacing, not seconds. Use the FPS in the production requirements. The compiler handles frame quantization and the total budget.')


# 规划器写的一镜＝标注的一镜去掉人物之间的相对位置，其余字段与顺序不变。11-13 逐镜档与 SAM＋VGGT 标注比，零样本写出的
# 方向近乱猜、前后系统性写反，只照抄写明方位的提示词也收不住；请求写明的方位已由照抄的动作句带着（238/242）。
PlannedShot = create_model('PlannedShot', __base__=StrictModel,
                           **{name: (info.annotation, info) for name, info in Shot.model_fields.items() if name != 'relations'})


class LayoutAnchor(StrictModel):
    time: UnitCoordinate
    visibility: Literal['visible', 'not_visible']
    box: list[UnitCoordinate] | None = Field(min_length=4, max_length=4)

    @model_validator(mode='after')
    def visible_box(self):
        if self.visibility == 'not_visible':
            if self.box is not None:
                raise ValueError('not_visible anchors require box=null')
        elif self.box is None or not (self.box[0] < self.box[2] and self.box[1] < self.box[3]):
            raise ValueError('visible anchors require a positive normalized xyxy box')
        return self


class EntityLayout(StrictModel):
    name: EntityName
    anchors: list[LayoutAnchor] = Field(min_length=1, max_length=2)

    @model_validator(mode='after')
    def ordered_times(self):
        times = [anchor.time for anchor in self.anchors]
        if any(left >= right for left, right in zip(times, times[1:])):
            raise ValueError('layout anchor times must be strictly increasing')
        return self


class FrontOf(StrictModel):
    near: EntityName
    far: EntityName

    @model_validator(mode='after')
    def distinct_pair(self):
        if self.near == self.far:
            raise ValueError('front_of near and far must be different entities')
        return self


class ShotLayout(StrictModel):
    shot: StrictInt = Field(gt=0)
    entities: list[EntityLayout]
    # 缺字段表示这对前后关系未知；空列表不是“没有前后”。exclude_unset 保留这一区别。
    front_of: list[FrontOf] = Field(default_factory=list)

    @model_validator(mode='after')
    def unique_entities(self):
        names = [entity.name for entity in self.entities]
        if len(names) != len(set(names)):
            raise ValueError('layout entities must be unique within a shot')
        present = set(names)
        pairs = [(pair.near, pair.far) for pair in self.front_of]
        if len(pairs) != len(set(pairs)):
            raise ValueError('front_of pairs must be unique within a shot')
        if any(pair.near not in present or pair.far not in present for pair in self.front_of):
            raise ValueError('front_of names must be entities in the same shot layout')
        return self


HOLDING_ERROR = 'holding must name registered others and belong to a character'
CHANGE_ERROR = 'change must refer to a present entity'
HOLDING_CHANGE_ERROR = 'holding change requires lists of registered others'
UNREGISTERED_ERROR = 'names an unregistered entity'


def _holds_others(values, holder, known) -> bool:
    return isinstance(values, list) and all(known.get(v) in ('characters', 'objects') and v != holder for v in values)


class StoryboardFields(StrictModel):
    """规划器的输出合同（提示里附的结构定义）；规划原答先过类型检查，再编译可推导量。

    规划器不写人物框与人物之间的相对位置：布局只有每镜 ``composition``／``ending_sides``。参考镜号也不在合同里：它总由
    在场名单决定（``shot_reference_labels``）。"""
    characters: list[Character]
    objects: list[Object]
    places: list[Place]
    shots: list[PlannedShot]


class Storyboard(StoryboardFields):
    """渲染器可执行合同；已有完整标注始终直接走本校验器。

    给定分镜表可以带标注测量的人物框 ``layouts``（渲染器的框读数要用），编译器从框推出与规划器字段相同的布局句；
    规划器的计划不带框。一张表里人物框与每镜布局字段只能有一种。"""
    shots: list[Shot]
    # 缺字段表示没有空间标注；显式[]表示已检查、没有可用约束。exclude_unset保留这一区别。
    layouts: list[ShotLayout] = Field(default_factory=list)
    @model_validator(mode='after')
    def references(self):
        raw = self.model_dump(exclude_unset=True)
        known = {}
        for kind, prefix in REGISTRIES.items():
            names = [entry['name'] for entry in raw[kind]]
            if names != [f'{prefix} {i}' for i in range(1, len(names)+1)]:
                raise ValueError(f'{kind}: names must be consecutive and ordered')
            known.update({name: kind for name in names})
        if not self.shots or [s.shot for s in self.shots] != list(range(1, len(self.shots)+1)):
            raise ValueError('shots must be nonempty and numbered 1..N')
        unknown = unregistered_entity_references(raw, known)
        if unknown:
            raise ValueError(f'unregistered name in storyboard: {unknown[0]}')
        layout_shots = [layout.shot for layout in self.layouts]
        if len(layout_shots) != len(set(layout_shots)) or any(number > len(self.shots) for number in layout_shots):
            raise ValueError('layouts must refer to distinct existing shots')
        if 'layouts' in raw and any(key in shot for shot in raw['shots'] for key in LAYOUT_FIELDS):
            raise ValueError('a storyboard carries either annotation person boxes or per-shot layout fields, not both')
        for shot in raw['shots']:
            count = shot['frame_count']
            if count not in FRAME_COUNTS:
                raise ValueError(f'frame_count must be 4n+1 within 1..{SEGMENT_MAX_FRAMES}')
            if shot['place'] is not None and known.get(shot['place']) != 'places':
                raise ValueError('shot place is not registered')
            names = [entry['name'] for entry in shot['present']]
            if len(set(names)) != len(names) or any(known.get(name) not in ('characters', 'objects') for name in names):
                raise ValueError('present names must be distinct registered characters or objects')
            if set(holding_targets(shot)) - set(names):
                raise ValueError('holding targets must be included in present')
            for entry in shot['present']:
                if 'holding' in entry and (known[entry['name']] != 'characters' or not _holds_others(entry['holding'], entry['name'], known)):
                    raise ValueError(HOLDING_ERROR)
            for change in shot['changes']:
                if change['name'] not in names:
                    raise ValueError(CHANGE_ERROR)
                if change['what'] == 'holding':
                    if not _holds_others(change.get('before'), change['name'], known) or not _holds_others(change['after'], change['name'], known):
                        raise ValueError(HOLDING_CHANGE_ERROR)
                elif not isinstance(change['after'], str) or (change.get('before') is not None and not isinstance(change['before'], str)):
                    raise ValueError('non-holding change endpoints must be strings')
            people = {name for name in names if known[name] == 'characters'}
            if any(_relation_error(relation, people, shot.get('relations', [])[:k])
                   for k, relation in enumerate(shot.get('relations', []))):
                raise ValueError(RELATION_ERROR)
            ending = [entry['name'] for entry in shot.get('ending_sides', [])]
            if len(set(ending)) != len(ending) or not set(ending) <= people:
                raise ValueError(ENDING_SIDE_ERROR)
        return self


def shot_reference_labels(shots: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    """每个在场实体指向它最近一次更早出现的镜；第一次出现没有条目。顺序跟 present。"""
    last: dict[str, int] = {}
    labels = []
    for shot in shots:
        if not isinstance(shot, Mapping) or not isinstance(shot.get('present'), list) or type(shot.get('shot')) is not int:
            raise ValueError('reference labels require numbered shots and present lists')
        refs = []
        for entry in shot['present']:
            if not isinstance(entry, Mapping) or not isinstance(entry.get('name'), str):
                raise ValueError('present entries need a name before reference labels')
            name = entry['name']
            if name in last:
                refs.append({'name': name, 'shot': last[name]})
        labels.append(refs)
        for entry in shot['present']:
            last[entry['name']] = shot['shot']
    return labels


def _relation_error(relation: Mapping[str, Any], people: set[str], earlier: Sequence[Mapping[str, Any]]) -> bool:
    """两人不同、都是这一镜在场的人物；同一对人同一种关系只写一次（反过来写也算同一对）。"""
    pair = frozenset((relation['subject'], relation['object']))
    return (len(pair) != 2 or not pair <= people
            or any(other['relation'] == relation['relation'] and frozenset((other['subject'], other['object'])) == pair
                   for other in earlier))


def validate_storyboard(board: Mapping[str, Any]) -> dict:
    """不填缺省事实，不改答复；返回只含原有字段的校验后对象。"""
    return Storyboard.model_validate(board).model_dump(exclude_unset=True)


def allocate_frame_counts(proposed: Sequence[int], target_frames: int | float | Fraction,
                          explicit_lengths: Sequence[int | float | Fraction] | None) -> list[int]:
    """最近合法总数＋有上下界的平方误差最小分配；等距总数取小、分配并列按镜序。"""
    if not proposed or any(type(value) is not int or value <= 0 for value in proposed):
        raise ValueError('pacing suggestions must be positive integer frame counts')
    if isinstance(target_frames, bool) or not isinstance(target_frames, (int, float, Fraction)) or not math.isfinite(target_frames):
        raise ValueError('target frame budget must be finite')
    requested = Fraction(str(target_frames))
    count, maximum = len(proposed), FRAME_COUNTS[-1]
    if not count <= requested <= maximum * count:
        raise ValueError(f'frame budget exceeds {count} shots capacity [{count}, {maximum * count}]')
    steps = (requested - count) / 4
    lower = steps.numerator // steps.denominator
    total = min((count + 4 * lower, min(count + 4 * (lower + 1), maximum * count)),
                key=lambda value: (abs(value - requested), value))
    if explicit_lengths is None:
        desired = [Fraction(total * value, sum(proposed)) for value in proposed]
    else:
        if len(explicit_lengths) != count:
            raise ValueError('explicit shot durations must match the planned shot count')
        desired = [Fraction(str(value)) for value in explicit_lengths]
        if any(not 1 <= value <= maximum for value in desired):
            raise ValueError(f'explicit shot duration exceeds capacity [1, {maximum}]')
        if sum(desired) != requested:
            raise ValueError('explicit shot durations do not sum to the requested total')
    frames = [1] * count
    # 平方损失是离散凸函数：每次选择成本最小的下一个4帧增量，容量满后移出。
    increments = [(8 * (1 - wanted) + 16, i) for i, wanted in enumerate(desired)]
    heapq.heapify(increments)
    for _ in range((total - count) // 4):
        _, index = heapq.heappop(increments)
        frames[index] += 4
        if frames[index] < maximum:
            heapq.heappush(increments, (8 * (frames[index] - desired[index]) + 16, index))
    return frames


def _drop_structural_errors(board: Mapping[str, Any]) -> tuple[Any, list[dict]]:
    """类型检查之前丢掉的字段级错误：计划合同里没有的人物框 ``layouts``、参考镜号 ``references`` 与人物关系
    ``relations``，词表外的构图词与镜末侧，同一镜在场名单里重名（留第一次）。"""
    if not isinstance(board, Mapping):
        return board, []
    board, dropped = deepcopy(dict(board)), []
    if 'layouts' in board:
        dropped.append({'path': 'layouts', 'value': board.pop('layouts'), 'reason': PLANNER_BOXES_DROPPED})
    for i, shot in enumerate(board.get('shots') or []):
        if not isinstance(shot, dict):
            continue
        if 'references' in shot:
            dropped.append({'path': f'shots[{i}].references', 'value': shot.pop('references'),
                            'reason': PLANNER_REFERENCES_DROPPED})
        if 'relations' in shot:
            dropped.append({'path': f'shots[{i}].relations', 'value': shot.pop('relations'),
                            'reason': PLANNER_RELATIONS_DROPPED})
        if 'composition' in shot and shot['composition'] not in COMPOSITIONS:
            dropped.append({'path': f'shots[{i}].composition', 'value': shot.pop('composition'),
                            'reason': f'composition must be one of {list(COMPOSITIONS)}'})
        if isinstance(shot.get('ending_sides'), list):
            kept = []
            for j, entry in enumerate(shot['ending_sides']):
                if isinstance(entry, dict) and entry.get('side') not in FRAME_SIDES:
                    dropped.append({'path': f'shots[{i}].ending_sides[{j}]', 'value': entry,
                                    'reason': f'side must be one of {list(FRAME_SIDES)}'})
                else:
                    kept.append(entry)
            shot['ending_sides'] = kept
    for i, shot in enumerate(board.get('shots') or []):
        if isinstance(shot, dict) and isinstance(shot.get('present'), list):
            seen, kept = set(), []
            for j, entry in enumerate(shot['present']):
                name = entry.get('name') if isinstance(entry, dict) else None
                if name is not None and name in seen:
                    dropped.append({'path': f'shots[{i}].present[{j}]', 'value': entry,
                                    'reason': 'name already present in this shot'})
                    continue
                seen.add(name)
                kept.append(entry)
            shot['present'] = kept
    return board, dropped


def _drop_wording_errors(board: dict) -> list[dict]:
    """``visible_result_wording_errors`` 指出的字段原地丢掉：登记表与在场条目里的那一项、变化的 before 删掉；
    变化的 after 出错或整条持物变化出错时，删掉这条变化。"""
    fields, changes, dropped = [], {}, []
    for error in visible_result_wording_errors(board):
        path = error['path']
        if match := re.fullmatch(r'characters\[(\d+)\]\.(\w+)', path):
            fields.append((board['characters'][int(match[1])], match[2], error))
        elif match := re.fullmatch(r'shots\[(\d+)\]\.present\[(\d+)\]\.(\w+)', path):
            fields.append((board['shots'][int(match[1])]['present'][int(match[2])], match[3], error))
        elif match := re.fullmatch(r'shots\[(\d+)\]\.changes\[(\d+)\]\.before', path):
            fields.append((board['shots'][int(match[1])]['changes'][int(match[2])], 'before', error))
        elif match := re.fullmatch(r'shots\[(\d+)\]\.changes\[(\d+)\](?:\.after)?', path):
            changes.setdefault((int(match[1]), int(match[2])), error)
        else:
            raise ValueError(f'unrecognized wording error path: {path}')
    for entry, key, error in fields:
        if key in entry:
            dropped.append({'path': error['path'], 'value': entry.pop(key), 'reason': error['reason']})
    for (shot, index), error in sorted(changes.items(), reverse=True):
        dropped.append({'path': error['path'], 'value': board['shots'][shot]['changes'].pop(index),
                        'reason': error['reason']})
    return dropped


def _drop_reference_errors(board: dict) -> list[dict]:
    """登记表与在场名单对不上的结构字段原地丢掉，判定与 ``Storyboard`` 校验相同：点了未登记实体的在场、持物、变化、
    人物关系与镜末侧（理由单列 ``UNREGISTERED_ERROR``，逐镜文字不动），不属于角色或点了自己的持物条目，名字不在这一镜
    在场名单里的变化，持物写法不合规的变化，不是两个在场人物或重复的关系，不在场或重复的镜末侧。"""
    known = {entry['name']: kind for kind in REGISTRIES for entry in board[kind]}
    registered = lambda name: known.get(name) in ('characters', 'objects')
    dropped = []
    for i, shot in enumerate(board['shots']):
        kept = []
        for j, entry in enumerate(shot['present']):
            if registered(entry['name']):
                kept.append(entry)
            else:
                dropped.append({'path': f'shots[{i}].present[{j}]', 'value': entry, 'reason': UNREGISTERED_ERROR})
        shot['present'] = kept
        names = [entry['name'] for entry in kept]
        for j, entry in enumerate(kept):
            if 'holding' in entry and (known[entry['name']] != 'characters'
                                       or not _holds_others(entry['holding'], entry['name'], known)):
                holding = entry['holding']
                reason = UNREGISTERED_ERROR if isinstance(holding, list) and not all(map(registered, holding)) else HOLDING_ERROR
                dropped.append({'path': f'shots[{i}].present[{j}].holding', 'value': entry.pop('holding'), 'reason': reason})
        kept = []
        for j, change in enumerate(shot['changes']):
            sides = [change.get('before'), change['after']] if change['what'] == 'holding' else []
            reason = (UNREGISTERED_ERROR if not registered(change['name']) or not all(
                          registered(name) for side in sides if isinstance(side, list) for name in side)
                      else CHANGE_ERROR if change['name'] not in names
                      else HOLDING_CHANGE_ERROR if sides and not all(_holds_others(side, change['name'], known) for side in sides)
                      else None)
            if reason:
                dropped.append({'path': f'shots[{i}].changes[{j}]', 'value': change, 'reason': reason})
            else:
                kept.append(change)
        shot['changes'] = kept
    for i, shot in enumerate(board['shots']):
        names = {entry['name'] for entry in shot['present']}
        people = {name for name in names if known[name] == 'characters'}
        kept, seen = [], set()
        for j, entry in enumerate(shot.get('ending_sides', [])):
            if not registered(entry['name']):
                dropped.append({'path': f'shots[{i}].ending_sides[{j}]', 'value': entry, 'reason': UNREGISTERED_ERROR})
            elif entry['name'] not in people or entry['name'] in seen:
                dropped.append({'path': f'shots[{i}].ending_sides[{j}]', 'value': entry, 'reason': ENDING_SIDE_ERROR})
            else:
                kept.append(entry)
                seen.add(entry['name'])
        if 'ending_sides' in shot:
            shot['ending_sides'] = kept
    return dropped


def compile_storyboard(board: Mapping[str, Any], *, target_seconds: int | float | None,
                       fps: int | None, cut_times_seconds: Sequence[int | float] | None,
                       stated: Mapping[str, Any] | None) -> tuple[dict, dict]:
    """规划原答的分镜表维护，依次：出错字段逐个丢弃 → 逐镜档拷回请求写明值 → 没写的持物沿用上一镜（:func:`carry_holding`）→ 被持有的实体补进在场 → 两处开镜值
    自相矛盾时不写（:func:`withhold_contested_openings`）→ 删掉不改变任何东西的变化（:func:`drop_unchanged_changes`）→
    衣着沿当前值延续（:func:`resolve_outfits`）→ 帧预算 → 唯一执行校验。不改身份或动作，不重问模型。

    方法内部出错也要退一步出片：镜末侧、构图词、在场名单、持物与变化的字段级错误、引用未登记实体的
    结构字段和否定式状态写法丢掉该字段并记进报告 ``dropped_fields``，不整份拒收；逐镜文字不动。旧合同的人物框、参考镜号与
    人物关系不在计划里，写了也
    丢掉、记进报告。
    ``stated`` 是逐镜档请求写明的值（:func:`parse_shot_script`），由 :func:`copy_stated_values` 拷回并记进报告；粗档给 None。"""
    board, dropped = _drop_structural_errors(board)
    compiled = StoryboardFields.model_validate(board).model_dump(exclude_unset=True)
    copied = [] if stated is None else copy_stated_values(compiled, stated)
    holding_carried = carry_holding(compiled)
    dropped += _drop_wording_errors(compiled)
    dropped += _drop_reference_errors(compiled)
    proposed = [shot['frame_count'] for shot in compiled['shots']]
    if target_seconds is None:
        if fps is not None or cut_times_seconds is not None:
            raise ValueError('FPS/cut times require an explicit total duration')
        requested, lengths = sum(proposed), None
    else:
        if type(target_seconds) not in (int, float) or not math.isfinite(target_seconds) or target_seconds <= 0:
            raise ValueError('target_seconds must be finite and positive')
        if type(fps) is not int or fps <= 0:
            raise ValueError('fps must be a positive integer')
        requested = Fraction(str(target_seconds)) * fps
        lengths = None
        if cut_times_seconds is not None:
            if not isinstance(cut_times_seconds, (list, tuple)) or any(
                    type(value) not in (int, float) or not math.isfinite(value) for value in cut_times_seconds):
                raise ValueError('cut times must be finite seconds')
            boundaries = [Fraction(0), *(Fraction(str(value)) for value in cut_times_seconds), Fraction(str(target_seconds))]
            if len(boundaries) != len(proposed) + 1 or any(right <= left for left, right in zip(boundaries, boundaries[1:])):
                raise ValueError('cut times must delimit every planned shot in increasing order')
            lengths = [(right - left) * fps for left, right in zip(boundaries, boundaries[1:])]
    frames = allocate_frame_counts(proposed, requested, lengths)
    added = _complete_storyboard_holding_presence(compiled)
    withheld = withhold_contested_openings(compiled)
    unchanged = drop_unchanged_changes(compiled)
    outfit_resolved = resolve_outfits(compiled)
    for shot, count in zip(compiled['shots'], frames, strict=True):
        shot['frame_count'] = count
    report = {'dropped_fields': dropped, 'stated_values_copied': copied,
              'holding_carried': holding_carried,
              'holding_presence_added': added, 'opening_withheld': withheld, 'unchanged_changes_dropped': unchanged,
              'outfit_resolved': outfit_resolved,
              'proposed_frame_counts': proposed,
              'compiled_frame_counts': frames, 'requested_frames': float(requested),
              'compiled_total_frames': sum(frames),
              'timing_source': 'explicit_cuts' if lengths is not None else 'model_pacing',
              'target_seconds': target_seconds, 'fps': fps}
    return validate_storyboard(compiled), report


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)


def slice_storyboard(board: Mapping[str, Any], start: int, end: int) -> dict:
    """取连续半开子段；按首次出现重编所有引用，默认穿着重定为子段第一次出现时所穿。"""
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(board['shots']):
        raise ValueError('invalid contiguous storyboard span')
    registries = {kind: {entry['name']: entry for entry in board[kind]} for kind in REGISTRIES}
    known = {name: kind for kind, entries in registries.items() for name in entries}
    if len(known) != sum(len(board[kind]) for kind in REGISTRIES):
        raise ValueError('duplicate registered names')
    shots = deepcopy(board['shots'][start:end])
    selected_shots = {shot['shot']: number for number, shot in enumerate(shots, 1)}
    layouts = [deepcopy(layout) for layout in board.get('layouts', []) if layout['shot'] in selected_shots]
    order = {kind: [] for kind in REGISTRIES}
    for shot in shots:
        unknown = unregistered_entity_references(shot, known)
        if unknown:
            raise ValueError(f'unregistered name in selected shots: {unknown[0]}')
        mentions = [entry['name'] for entry in shot['present']]
        if shot['place'] is not None:
            mentions.append(shot['place'])
        mentions += entity_references(shot)
        mentions += entity_references([layout for layout in layouts if layout['shot'] == shot['shot']])
        for name in mentions:
            if name not in known:
                raise ValueError(f'unregistered name in selected shots: {name}')
            bucket = order[known[name]]
            if name not in bucket:
                bucket.append(name)
    # 登记描述也会引用别的登记项（如 Object 6 的外貌是 "Object 3 stone"）。
    # 先闭合依赖，再统一重编号；依赖环按已收集的名字自然终止。
    pending = [name for names in order.values() for name in names]
    for name in pending:
        for dependency in entity_references(registries[known[name]][name]):
            if dependency not in known:
                raise ValueError(f'unregistered name in selected registry: {dependency}')
            bucket = order[known[dependency]]
            if dependency not in bucket:
                bucket.append(dependency)
                pending.append(dependency)
    selected = {kind: [deepcopy(registries[kind][name]) for name in names] for kind, names in order.items()}
    for character in selected['characters']:
        first = next((e for shot in shots for e in shot['present'] if e['name'] == character['name']), None)
        original = registries['characters'][character['name']]
        first_outfit = first.get('outfit', original.get('outfit')) if first is not None else original.get('outfit')
        if first_outfit is not None:
            character['outfit'] = first_outfit
        for shot in shots:
            for entry in shot['present']:
                if entry['name'] != character['name']:
                    continue
                current = entry.get('outfit', original.get('outfit'))
                if current is not None:
                    if current == first_outfit:
                        entry.pop('outfit', None)
                    else:
                        entry['outfit'] = current
    mapping = {name: f'{REGISTRIES[kind]} {i}' for kind, names in order.items() for i, name in enumerate(names, 1)}
    def rename(value):
        if isinstance(value, str):
            return NAME.sub(lambda m: mapping[m.group()] if m.group() in mapping else m.group(), value)
        if isinstance(value, list):
            return [rename(item) for item in value]
        if isinstance(value, dict):
            return {key: rename(item) for key, item in value.items()}
        return value
    for number, shot in enumerate(shots, 1):
        shot['shot'] = number
    selected_board = rename({**selected, 'shots': shots})
    if 'layouts' in board:
        for layout in layouts:
            layout['shot'] = selected_shots[layout['shot']]
        selected_board['layouts'] = rename(layouts)
    reconcile_holding(selected_board)
    _complete_storyboard_holding_presence(selected_board)
    return validate_storyboard(selected_board)


def from_annotation(annotation: Mapping[str, Any], frame_counts: Sequence[int], *, start: int, end: int) -> dict:
    """已校验v18标注→指定连续子段的分镜表。坏镜是否可取由取样台账决定，调用者不得跨断点。

    标注带人物框时，框经 ``with_layout_fields`` 换成每镜布局字段（与训练文字同一次推导）。"""
    if len(annotation['shots']) != len(frame_counts):
        raise ValueError('annotation/frame count mismatch')
    board = {kind: deepcopy(annotation[kind]) for kind in REGISTRIES}
    board['shots'] = [{**{field: deepcopy(shot[field]) for field in SHOT_FIELDS if field in shot}, 'frame_count': frames}
                      for shot, frames in zip(annotation['shots'], frame_counts, strict=True)]
    if 'layouts' in annotation:
        board = with_layout_fields({**board, 'layouts': deepcopy(annotation['layouts'])})
    # 当前目标只接收登记表可见性；逐镜 visibility 留在源标注，不扩充执行合同。
    # 其他额外键仍由严格校验报出，避免丢掉动作、持有者或镜末信息。
    for shot in board['shots']:
        for entry in shot['present']:
            entry.pop('visibility', None)
    reconcile_holding(board)
    # 整块梗概/段落沿用原登记编号，才能与原 story 中的编号保持对应。只有真子段重编号。
    if start == 0 and end == len(frame_counts):
        _complete_storyboard_holding_presence(board)
        return validate_storyboard(board)
    return slice_storyboard(board, start, end)
