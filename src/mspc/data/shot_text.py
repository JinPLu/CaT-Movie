"""v18标注／规划分镜表 → 同一渲染器文字与实体字符范围。

每镜使用统一登记的当前子表G_i加本镜S_i：稳定外貌、一致衣着与物体／地点登记在前，画风、机位／布局、
首次可见属性、原动作、可见末态按时间顺序跟进。present不保证RGB0就在场；明确before仍定义变化前状态。
易变或不明衣着／持物／姿态／state不固化到G_i。人物短识别保留原中心名词及同镜可区分特征，必要时用已有衣着分句区分。
动作中的自然称呼只按本镜在场实体作唯一匹配；歧义交离线受约束指称选择，不能确定的保留原句并报告。
物体／地点以原ID连接登记与逐镜内容。训练标注动作原样消费，不伪造中间动作监督。

实际prefix slot及同帧核实实体ID只作校验并进本镜登记；目标文字不描述参考映射。参考行带可见人物编号及登记外貌。
登记与实体状态直接写事实，不加Entities/Opening/Action/Ending字段标签或重复的参考使用指令；首次可见的含义由v18合同持有。
首帧图像消费者保留原完整自然语言开态，不提前绘制动作或末态。布局与画风的省略规则由
without_target_text统一持有。本镜在场的登记物体全部进登记，静态摆放句照写；真实umT5消费者检验512-token预算，不截断动作。
"""
from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Any, Callable, Iterable, Mapping, Sequence

from mspc.core.wan22_contract import (MEMORY_ROW_ROLE_TEXT, MEMORY_ROW_SETTING, MEMORY_ROW_SHOT_OBJECTS,
                                      MEMORY_ROW_VISIBLE_IDENTITIES)

PROSE_ENTITY = re.compile(r"(?:Character|Object|Location) [1-9]\d*")
_EXPLICIT_ENTITY_LIST = re.compile(
    r"\b(?P<category>Characters|Objects|Locations)\s+"
    r"(?P<numbers>[1-9]\d*(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+)[1-9]\d*)*)\b")
_ACTION_SUBJECT = re.compile(
    r"(?:^|[.!?;]\s*|\b(?:while|when|as|then|and|but)\s+)"
    r"(?P<names>Character [1-9]\d*(?:(?:\s*,\s*|\s*,?\s+and\s+)Character [1-9]\d*)*)"
    r"\s+(?P<predicate>(?:(?:[a-z]+ly)\s+)*[a-z]+\b)", re.IGNORECASE)
_FINITE_ACTION_VERB = re.compile(r"(?:is|are|was|were|has|have|had|does|do|did|can|could|will|would|should|must|may|might|[a-z]{3,}s)$")
ARTICLE = re.compile(r"^an? ")
DEFAULT_ANGLE = "eye-level shot"
DEFAULT_CAMERA = frozenset({DEFAULT_ANGLE, "static"})  # 平视与静止是缺省，不写
# 穿着一栏里不是衣物的两类真实返回（首批 115,077 条里 514 条，2026-10-12）：没看清的不写，没穿的不接在 wearing 后面。
UNSEEN_OUTFIT = re.compile(r"(?:unknown|unclear|none|not |no )", re.IGNORECASE)
UNDRESSED = re.compile(r"(?:shirtless|topless|naked|nude|bare)", re.IGNORECASE)
#: ``third_party/wan22_runtime/diffsynth/pipelines/wan_video.py`` 构造
#: ``HuggingfaceTokenizer(name=..., seq_len=512, clean='whitespace')``，超出部分静默截断。
UMT5_PROMPT_TOKEN_BUDGET = 512
#: 文字准入与文字缓存清单都记这个值，逐镜文字写法一变就换新值：同路径的旧准入、旧文字缓存不能冒充新文字被读到。
#: 现行写法：完整编号、参考行带登记外貌、本镜在场登记物体全部进登记。
SHOT_TEXT_COMPILER_CONTRACT = "registered_entity_labels_reference_appearance_setting_and_all_present_objects"
PICTURE_TYPE_TEXT = {
    "live_action": "Live-action footage.",
    "animation_2d": "2D animation.",
    "animation_3d_or_game": "The shot includes computer-generated imagery.",
    "screen_or_slides": "Screen recording or presentation slides.",
}


#: 目标文字里可整体省掉的两部分：标注／计划的方位布局句与画风句。训练与推理按 ``model.omit_target_text`` 同一规则省。
TARGET_TEXT_OMISSIONS = frozenset({"layouts", "picture_type"})


#: 计划里每镜的布局字段（规划器写，或由标注人物框经 ``derive_layout_fields`` 推出）。
LAYOUT_FIELDS = ("composition", "relations", "ending_sides")


def without_target_text(annotation: Mapping[str, Any], omitted: frozenset[str]) -> Mapping[str, Any]:
    """省掉 ``layouts`` 时，标注的人物框与计划的每镜布局字段一起去掉：两者写出的是同一组布局句。"""
    if not omitted <= TARGET_TEXT_OMISSIONS:
        raise ValueError(f"omitted target text must be items of {sorted(TARGET_TEXT_OMISSIONS)}")
    if "layouts" in omitted:
        annotation = {key: value for key, value in annotation.items() if key != "layouts"}
        annotation = {**annotation, "shots": [{key: value for key, value in shot.items() if key not in LAYOUT_FIELDS}
                                              for shot in annotation["shots"]]}
    if "picture_type" in omitted:
        annotation = {**annotation, "shots": [{key: value for key, value in shot.items() if key != "picture_type"}
                                              for shot in annotation["shots"]]}
    return annotation


def _picture_type_text(shot: Mapping[str, Any]) -> str:
    """只消费明确给出的画风；用户未指定时不默认补成实拍。"""
    value = shot.get("picture_type")
    if value is None:
        return ""
    if value not in PICTURE_TYPE_TEXT:
        raise ValueError(f"unsupported picture_type: {value!r}")
    return PICTURE_TYPE_TEXT[value]


@dataclass(frozen=True)
class _Text:
    text: str
    # name、半开字符范围与产生这次提及的源字段；不依赖外貌文字的唯一性。
    spans: tuple[tuple[str, int, int, str], ...] = ()

    @classmethod
    def entity(cls, name: str, text: str, source: str) -> "_Text":
        return cls(text, ((name, 0, len(text), source),) if text else ())


def _concat(*values: str | _Text) -> _Text:
    texts, spans, offset = [], [], 0
    for value in values:
        value = value if isinstance(value, _Text) else _Text(value)
        texts.append(value.text)
        spans.extend((name, start + offset, end + offset, source) for name, start, end, source in value.spans)
        offset += len(value.text)
    return _Text("".join(texts), tuple(spans))


def _separated(items: Iterable[str | _Text], separator: str) -> _Text:
    values = list(items)
    return _concat(*(part for i, value in enumerate(values) for part in ((separator, value) if i else (value,))))


def _sentence(text: str | _Text) -> _Text:
    value = text if isinstance(text, _Text) else _Text(text)
    left = len(value.text) - len(value.text.lstrip())
    right = len(value.text.rstrip())
    stripped = value.text[left:right]
    if not stripped:
        raise ValueError("a sentence cannot be empty")
    delta = len(stripped[0].upper()) - 1
    spans = []
    for name, start, end, source in value.spans:
        start, end = max(start, left) - left, min(end, right) - left
        if start < end:
            spans.append((name, start + (delta if start else 0), end + delta, source))
    return _Text(stripped[0].upper() + stripped[1:]
                 + ("" if stripped.endswith((".", "!", "?")) else "."), tuple(spans))


def _join(items: Iterable[_Text]) -> _Text:
    items = list(items)
    return items[0] if len(items) == 1 else _concat(_separated(items[:-1], ", "), " and ", items[-1])


def _identity_list(names: Sequence[tuple[str, Sequence[str]]]) -> _Text:
    """每次出现都写完整Character N／Object N，不共享类别前缀或留下裸编号。"""
    return _join([_Text(name, tuple((name, 0, len(name), source) for source in sources))
                  for name, sources in names])


def _explicit_action_subjects(text: str) -> dict[int, str]:
    """只识别明确ID开头的有限动词分句；不从自然称呼或宾语猜执行者。"""
    subjects = {}
    for clause in _ACTION_SUBJECT.finditer(text):
        names = list(PROSE_ENTITY.finditer(text, clause.start("names"), clause.end("names")))
        verb = clause.group("predicate").split()[-1].lower()
        if verb in {"as", "towards", "across", "besides", "unless", "whereas", "versus", "always", "sometimes", "perhaps"}:
            continue
        if not _FINITE_ACTION_VERB.fullmatch(verb) and not (
                len(names) > 1 and verb not in {"a", "an", "the", "to", "of", "on", "at", "from", "for", "with", "by", "in", "into", "onto", "near", "beside", "behind", "under", "above", "between", "among", "through", "around", "along", "against", "without", "within", "over", "out", "off", "up", "down", "and", "or", "but", "while", "when", "then"}):
            continue
        for name in names:
            subjects[name.start()] = text[clause.start("predicate"):]
    return subjects


class _Row:
    """一行文字里的称呼：人物始终是 ``Character N``；物件与场所第一次提到用外貌原文，再提到把不定冠词换成 the。"""

    def __init__(self, annotation: Mapping[str, Any]):
        self.characters = {entry["name"]: entry["appearance"] for entry in annotation["characters"]}
        self.outfits = {entry["name"]: entry.get("outfit") for entry in annotation["characters"]}
        self.things = {entry["name"]: entry["appearance"] for kind in ("objects", "places") for entry in annotation[kind]}
        if set(self.characters) & set(self.things) or len(self.characters) + len(self.things) != sum(
                len(annotation[kind]) for kind in ("characters", "objects", "places")):
            raise ValueError("duplicate name in the registry")
        self.mentioned: set[str] = set()

    def thing(self, name: str, source: str) -> _Text:
        phrase = self.things[name]
        if name in self.mentioned:
            phrase = ARTICLE.sub("the ", phrase)
        self.mentioned.add(name)
        return _Text.entity(name, phrase, source)

    def held(self, name: str, source: str) -> _Text:
        """拿着或抱着的可以是物件，也可以是人物（抱着婴儿、牵着另一个人）；人物只写编号。"""
        return _Text.entity(name, name, source) if name in self.characters else self.thing(name, source)

    def prose(self, text: str, source: str, *, identities: Mapping[str, _Text] | None = None) -> _Text:
        # 原动作也可能带“Characters 1 and 3”；只展开带明确类别的ID列表，不把人数/数值当ID。
        def explicit_names(match):
            category = match.group("category")[:-1]
            return _join([_Text(category + " " + number) for number in re.findall(r"[1-9]\d*", match.group("numbers"))]).text

        text = _EXPLICIT_ENTITY_LIST.sub(explicit_names, text)
        parts, position, identified = [], 0, set()
        subjects = _explicit_action_subjects(text) if identities is not None else {}
        for match in PROSE_ENTITY.finditer(text):
            name = match.group()
            identity = self.held(name, source)
            possessive = text[match.end():].startswith(("'s", "’s"))
            already_identified = text[match.end():].startswith(", " + identities[name].text) if identities is not None and name in identities else False
            if (identities is not None and match.start() in subjects and not possessive and not already_identified and name not in identified
                    and name in identities and identities[name].text):
                following = text[match.end():]
                close = "" if not following.strip() or following.startswith((",", ".", "!", "?", ";", ":")) else ","
                identity = _concat(identity, ", ", identities[name], close)
                identified.add(name)
            parts.extend((text[position:match.start()], identity))
            position = match.end()
        parts.append(text[position:])
        return _concat(*parts)


def _outfit(value: str | None, verb: str) -> str:
    """穿着短语：``wearing a grey coat``；``shirtless`` 之类原样写；标注没看清（``unknown``、``none``）时为空。"""
    if value:
        value = re.sub(r"^(?:wearing|wears)\s+", "", value, flags=re.IGNORECASE)
    if not value or UNSEEN_OUTFIT.match(value):
        return ""
    return value if UNDRESSED.match(value) else f"{verb} {value}"


def _camera(shot: Mapping[str, Any], composition: str | None = None) -> _Text:
    """景别、机位角度、运镜，布局推出构图词时接在最后（Wan 扩写提示的电影设定串）。"""
    camera = shot.get("camera") or {}
    words = [camera[key] for key in ("shot_size", "angle", "movement") if camera.get(key) and camera[key] not in DEFAULT_CAMERA]
    words += [composition] if composition else []
    return _sentence(", ".join(words)) if words else _Text("")


def _character(row: _Row, entry: Mapping[str, Any], source: str, positions: Sequence[_Text] = (),
               *, appearance: str | _Text, compact: bool, field_sources: Mapping[tuple[str, str], str],
               registered_outfits: AbstractSet[str], worn_objects: Sequence[_Text] = ()) -> _Text:
    """人物一句；``positions`` 是对先出现的人的相对位置，接在姿态后面（"is standing to the right of Character 1"）。"""
    name = entry["name"]
    origin = lambda field: field_sources.get((name, field), source + "." + field)
    absent = entry.get("posture") == "not present"
    parts = []
    where = _concat(" ", _join(positions)) if positions else _Text("")
    if entry.get("posture") and not absent:
        posture = entry["posture"]
        if posture.endswith(" behind") and any(position.text.startswith("behind ") for position in positions):
            posture = posture.removesuffix(" behind")
        parts.append(_concat("is ", row.prose(posture, origin("posture")), where))
    elif positions and not absent:
        parts.append(_concat("is", where))
    outfit = "" if name in registered_outfits else _outfit(entry.get("outfit", row.outfits[name]), "wearing")
    if outfit:
        parts.append(_Text(outfit if parts else "is " + outfit))
    if entry.get("state"):
        parts.append(_concat("" if parts else "is ", row.prose(entry["state"], origin("state"))))
    held = [row.held(item, origin("holding")) for item in entry.get("holding") or []]
    if held:
        parts.append(_concat("holding " if parts else "is holding ", _join(held)))
    if worn_objects:
        parts.append(_concat("wearing " if parts else "is wearing ", _join(worn_objects)))
    identity = _Text.entity(name, name, source + ".name")
    appearance = (appearance if isinstance(appearance, _Text) else
                  _Text.entity(name, appearance, f"characters[{list(row.characters).index(name)}].appearance"))
    if absent:
        absence = _concat(identity, " is not present at first.")
        if not parts:
            return absence
        # before的缺席先写；present的其余字段仍描述首次可见，不能接成“缺席且无意识”。
        visible = _concat(identity, " ", _separated(parts, ", "))
        return _concat(absence, " When first visible, ", visible)
    if compact and not parts and not appearance.text:
        return _Text("")
    if compact and appearance.text:
        label = _concat(identity, " (", appearance, ")")
        return _concat(label, " ", _separated(parts, ", ")) if parts else label
    if not appearance.text and parts and parts[0].text.startswith("is "):
        return _concat(identity, " ", _separated(parts, ", "))
    return _separated([part for part in [identity, appearance, *parts] if part.text], ", ")


def _scene(row: _Row, shot: Mapping[str, Any], entries: Iterable[Mapping[str, Any]], source: str,
           positions: Mapping[str, Sequence[_Text]] | None = None, *,
           appearances: Mapping[str, str | _Text], compact: bool,
           field_sources: Mapping[tuple[str, str], str], registered_outfits: AbstractSet[str]) -> list[_Text]:
    """场所加在场的人各一句，再加带状态的物件各一句；没有状态的物件只在被拿着或动作句提到时出现。"""
    entries = list(entries)
    positions = positions or {}
    worn = {}
    if compact:
        present_names = {entry["name"] for entry in entries if entry["name"] in row.characters}
        for i, entry in enumerate(entries):
            match = re.fullmatch(r"worn by (Character [1-9]\d*)\.?", entry.get("state") or "")
            if entry["name"] in row.things and match and match.group(1) in present_names:
                worn.setdefault(match.group(1), []).append(row.thing(entry["name"], f"{source}.present[{i}].name"))
    # 标注用 none／unknown 表示这一镜没有场所，不把这两个词写进句子。
    place_name = shot.get("place")
    place = row.thing(place_name, source + ".place") if place_name not in (None, "none", "unknown") else _Text("")
    people = [_character(row, entry, f"{source}.present[{i}]", positions.get(entry["name"], ()),
                         appearance=appearances[entry["name"]], compact=compact, field_sources=field_sources,
                         registered_outfits=registered_outfits, worn_objects=worn.get(entry["name"], ()))
              for i, entry in enumerate(entries) if entry["name"] in row.characters]
    objects = []
    for i, entry in enumerate(entries):
        if entry["name"] not in row.things or not (entry.get("state") or compact and entry.get("posture")):
            continue
        origin = field_sources.get((entry["name"], "state"), f"{source}.present[{i}].state")
        state = entry.get("state") or ""
        wearer = re.fullmatch(r"worn by (Character [1-9]\d*)\.?", state)
        if compact and wearer and wearer.group(1) in worn:
            state = ""
            if not entry.get("posture"):
                continue
        seated_on = re.fullmatch(r"(Character [1-9]\d* is sitting on) it\.?", state)
        worn_by = re.fullmatch(r"worn by (Character [1-9]\d*)\.?", state)
        if seated_on:
            # 真实对象state已是主谓句，it指本对象；不再前接“Object N is”制造双重主语。
            objects.append(_concat(row.prose(seated_on.group(1), origin), " ",
                                   row.thing(entry["name"], f"{source}.present[{i}].name")))
        elif compact and worn_by:
            # 完整明确的穿戴关系直接改主动语态；不合并不同穿戴者，也不改源state。
            objects.append(_concat(row.held(worn_by.group(1), origin), " wears ",
                                   row.thing(entry["name"], f"{source}.present[{i}].name")))
        else:
            predicates = []
            if compact and entry.get("posture"):
                predicates.append(row.prose(entry["posture"], field_sources.get(
                    (entry["name"], "posture"), f"{source}.present[{i}].posture")))
            if state:
                predicates.append(row.prose(state, origin))
            objects.append(_concat(row.thing(entry["name"], f"{source}.present[{i}].name"), " is ",
                                   _separated(predicates, ", ")))
    lines = [line for line in people + objects if line.text]
    if compact:
        # 同一谓语只写一次，每个原ID及其name来源仍逐一保留；不同位置/状态不会归为一组。
        groups: dict[tuple[str, str], list[int]] = {}
        for index, line in enumerate(lines):
            subject = PROSE_ENTITY.match(line.text)
            if subject and line.text[subject.end():].startswith(" is "):
                groups.setdefault((subject.group().split()[0], line.text[subject.end():]), []).append(index)
        combined, consumed = {}, set()
        for (category, predicate), indices in groups.items():
            if len(indices) < 2:
                continue
            subjects = []
            for index in indices:
                line = lines[index]
                end = len(line.text) - len(predicate)
                name = line.text[:end]
                subjects.append((name, (next(
                    origin for entity, start, stop, origin in line.spans if entity == name and start == 0 and stop == end),)))
            first, end = lines[indices[0]], len(lines[indices[0]].text) - len(predicate)
            remainder = _Text(predicate[4:], tuple((name, start - end - 4, stop - end - 4, origin)
                for name, start, stop, origin in first.spans if start >= end + 4))
            combined[indices[0]] = _concat(_identity_list(subjects), " are ", remainder)
            consumed.update(indices[1:])
        lines = [combined.get(index, line) for index, line in enumerate(lines) if index not in consumed]
        lines = [_finite_entity_state(line) for line in lines]
    if not lines:
        return [_sentence(place)] if place.text else []
    return [_sentence(_concat("In ", place, ", ", lines[0]) if place.text else lines[0]), *(_sentence(line) for line in lines[1:])]


def _finite_entity_state(line: _Text) -> _Text:
    """同谓语合并后，把单一静态姿态／持物补语写成标准现在时，不改原action。"""
    from mspc.data.storyboard import posture_stem

    match = re.search(r" (?P<be>is|are) (?P<pose>standing|sitting|kneeling|lying|leaning|bending|holding)\b", line.text)
    if (match is None or re.search(r"\b(?:is|are)\b", line.text[:match.start()])
            or re.search(r"\band\s+\w+ing\b", line.text[match.end():])):
        return line
    stem = posture_stem(match.group("pose"))
    replacement = " " + stem + ("s" if match.group("be") == "is" else "")
    delta = len(replacement) - (match.end() - match.start())
    spans = []
    for name, start, end, source in line.spans:
        if start < match.end() and end > match.start():
            raise ValueError("a finite state predicate overlaps an entity label")
        spans.append((name, start + (delta if start >= match.end() else 0),
                      end + (delta if end >= match.end() else 0), source))
    return _Text(line.text[:match.start()] + replacement + line.text[match.end():], tuple(spans))


def _object_posture(row: _Row, entry: Mapping[str, Any], source: str, *, posture_source: str) -> _Text:
    """沿用原开镜图的物体姿态写法，供缺失该字段的视频条件复用。"""
    return _sentence(_concat(row.thing(entry["name"], source + ".name"), ", ",
                             row.prose(entry["posture"], posture_source)))


def _ending(row: _Row, shot: Mapping[str, Any], source: str, layout: _ShotLayout | None = None) -> _Text:
    """镜末成立的变化；布局里镜末换了左中右的人物也写在这里（"Character 1 is on the right side of the frame"）。"""
    opening = {entry["name"]: entry for entry in shot["present"]}
    final_states = {change["name"]: change["after"] for change in shot.get("changes") or []
                    if change["what"] == "state"}
    visible_endings = {name for name, state in final_states.items() if isinstance(state, str) and state.strip()
                      and not re.search(r"\b(?:no longer|not|unknown|unclear)\b", state, re.IGNORECASE)}
    current = {}
    clauses = []
    for i, change in enumerate(shot.get("changes") or []):
        name, what, after = change["name"], change["what"], change["after"]
        key = (name, what)
        start = current[key] if key in current else opening[name].get(what, row.outfits.get(name) if what == "outfit" else None)
        if start == after:
            # 明确before已在副本中构造起态；仍相同表示端点无变化，或缺before时起态已是结果。
            # 不能补猜before或悄悄删除这条变化。
            raise ValueError(f"shot {shot.get('shot')}: {name} {what} is already {after!r} before its change")
        current[key] = after
        change_source = f"{source}.changes[{i}]"
        who = row.held(name, change_source + ".name")
        if what == "holding":
            if after:
                clauses.append(_concat(who, " holds ", _join(row.held(item, change_source + ".after") for item in after)))
            else:
                before = change.get("before")
                if not before:
                    raise ValueError("empty holding change has no prior held entity")
                # before/after明确给出释放关系；没有可见state时不补猜手或嘴的状态。
                if name not in visible_endings:
                    clauses.append(_concat(who, " has released ",
                                           _join(row.held(item, change_source + ".before") for item in before)))
        elif what == "outfit":
            outfit = _outfit(after, "wears")
            if outfit:
                clauses.append(_concat(who, " is " if UNDRESSED.match(outfit) else " ", outfit))
        elif what in ("posture", "state"):
            clauses.append(_concat(who, " is ", row.prose(after, change_source + ".after")))
        else:
            raise ValueError(f"unknown change kind {what!r}")
    for name, side, side_source in layout.ending_sides if layout is not None else ():
        clauses.append(_concat(row.held(name, side_source + ".name"), " is ", _FRAME_SIDE_PHRASE[side]))
    return _concat("By the end of the shot, ", _separated(clauses, "; "), ".") if clauses else _Text("")


# 布局按 Wan2.2 官方提示扩写的写法（``wan/utils/system_prompt.py``）：机位串里一个构图词，人物之间的位置用自然短语，
# 不写自造的 "Layout:" 模板与大小档。只写人物（含动物），不写物体。
# 下列阈值是约定值，没有官方出处；实现时在约 200 条真实样本上看图核对。
_OPENING_TIME_MAX = 0.5  # 第一个锚可见且不晚于镜内这一时刻，才算开镜就在画面里
# 构图的居中带：面积加权重心落在 0.4–0.6 之外才算偏重一侧。11-06 用 gpt-6-astra 看图核对 160 个随机真实镜：居中带取中间
# 三分之一时 center 判得太多（模型认为偏一侧、另一侧留空），认可率 136/160；收窄到 0.4–0.6 为 149/160（同一批样本内挑的整数）。
_HEAVY_SIDE = 0.4
_CENTER_SPREAD = 1 / 3  # 多人时最左最右两人中心相距不到这么多，仍算 Center
_SYMMETRY_TOLERANCE = 0.1  # 两人中心关于画面中线对称的容差
_SYMMETRY_AREA_RATIO = 0.5  # 对称还要求两人框面积之比不低于它
_LATERAL_GAP = 0.15  # 左右相邻两人中心横向差至少这么多才写谁在谁左边；镜末换左中右也要横移至少这么多
_RELATION_PHRASE = {"left_of": "to the left of", "right_of": "to the right of",
                    "in_front_of": "in front of", "behind": "behind"}
_INVERSE = {"left_of": "right_of", "right_of": "left_of", "in_front_of": "behind", "behind": "in_front_of"}
_FRAME_SIDE_PHRASE = {"left": "on the left side of the frame", "center": "in the center of the frame",
                      "right": "on the right side of the frame"}


def _third(center: float) -> str:
    """横向三等分：镜末位置按它分画面左中右。"""
    return "left" if center < 1 / 3 else "center" if center < 2 / 3 else "right"


@dataclass(frozen=True)
class _Relation:
    subject: str
    relation: str
    object: str
    source: str


@dataclass(frozen=True)
class _ShotLayout:
    """一镜的布局：构图词、人物两两的相对位置、镜末换了左中右的人物。由人物框推出，编译器只读这一结构。"""
    composition: str | None
    relations: tuple[_Relation, ...]
    ending_sides: tuple[tuple[str, str, str], ...]  # (人物, 镜末左中右, 来源)


def _composition(boxes: Sequence[Sequence[float]]) -> str | None:
    """开镜人物框 → Wan 构图词。按框面积加权的横向重心：在 0.4 以左为 Left-heavy、0.6 以右为 Right-heavy；居中时一人或
    挤在中间为 Center，两人关于中线对称且大小相近为 Symmetrical，其余为 Balanced。Short-side 要看朝向，框推不出，不写。"""
    if not boxes:
        return None
    centers = [(x1 + x2) / 2 for x1, _, x2, _ in boxes]
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    if min(areas) <= 0:
        raise ValueError("a visible layout box must have positive area")
    weight = sum(c * a for c, a in zip(centers, areas)) / sum(areas)
    if weight < _HEAVY_SIDE:
        return "left-heavy composition"
    if weight > 1 - _HEAVY_SIDE:
        return "right-heavy composition"
    if len(boxes) == 1 or max(centers) - min(centers) < _CENTER_SPREAD:
        return "center composition"
    if (len(boxes) == 2 and abs(sum(centers) - 1) <= _SYMMETRY_TOLERANCE
            and min(areas) / max(areas) >= _SYMMETRY_AREA_RATIO):
        return "symmetrical composition"
    return "balanced composition"


def _shot_layout(annotation: Mapping[str, Any], index: int) -> _ShotLayout | None:
    """这一镜的布局：标注带人物框（``layouts``）时由框推出，计划直接读每镜的布局字段；两样不能同时有。

    推导与写句各只有一处：训练文字从标注框推出，规划器的监督目标经 ``derive_layout_fields`` 由同一推导写成字段，
    推理时编译器读规划器写的字段，写句代码相同。"""
    shot = annotation["shots"][index]
    has_fields = any(key in shot for key in LAYOUT_FIELDS)
    if "layouts" in annotation:
        if has_fields:
            raise ValueError(f"shot {shot['shot']} carries plan layout fields beside annotation layouts")
        return _layout_from_boxes(annotation, shot)
    if not has_fields:
        return None
    source = f"shots[{index}]"
    relations = tuple(_Relation(entry["subject"], entry["relation"], entry["object"], f"{source}.relations[{k}]")
                      for k, entry in enumerate(shot.get("relations", [])))
    ending = tuple((entry["name"], entry["side"], f"{source}.ending_sides[{k}]")
                   for k, entry in enumerate(shot.get("ending_sides", [])))
    return _ShotLayout(shot.get("composition"), relations, ending)


def derive_layout_fields(annotation: Mapping[str, Any], index: int) -> dict[str, Any]:
    """标注人物框 → 计划里这一镜的布局字段；推不出的项不出现。规划器的监督目标、给定分镜表都经它。"""
    layout = _layout_from_boxes(annotation, annotation["shots"][index])
    if layout is None:
        return {}
    fields: dict[str, Any] = {}
    if layout.composition:
        fields["composition"] = layout.composition
    if layout.relations:
        fields["relations"] = [{"subject": r.subject, "relation": r.relation, "object": r.object} for r in layout.relations]
    if layout.ending_sides:
        fields["ending_sides"] = [{"name": name, "side": side} for name, side, _ in layout.ending_sides]
    return fields


def with_layout_fields(board: Mapping[str, Any]) -> dict:
    """标注的人物框（``layouts``）换成每镜布局字段，删掉 ``layouts``；没有 ``layouts`` 时只做浅拷贝。"""
    if "layouts" not in board:
        return dict(board)
    shots = [{**shot, **derive_layout_fields(board, index)} for index, shot in enumerate(board["shots"])]
    for shot in board["shots"]:
        if any(key in shot for key in LAYOUT_FIELDS):
            raise ValueError(f"shot {shot['shot']} carries plan layout fields beside annotation layouts")
    return {**{key: value for key, value in board.items() if key != "layouts"}, "shots": shots}


def _layout_from_boxes(annotation: Mapping[str, Any], shot: Mapping[str, Any]) -> _ShotLayout | None:
    """标注里这一镜的人物框 → ``_ShotLayout``；这一镜没有布局条目时为 None。"""
    entries = [(i, layout) for i, layout in enumerate(annotation.get("layouts", [])) if layout["shot"] == shot["shot"]]
    if not entries:
        return None
    if len(entries) > 1:
        raise ValueError(f"shot {shot['shot']} has more than one layout entry")
    i, layout = entries[0]
    characters = {entry["name"] for entry in annotation["characters"]}
    present = {entry["name"] for entry in shot["present"]}
    opening, ending_sides = [], []
    for j, entity in enumerate(layout["entities"]):
        if entity["name"] not in characters or entity["name"] not in present:
            continue
        source = f"layouts[{i}].entities[{j}]"
        first, last = entity["anchors"][0], entity["anchors"][-1]
        if first["visibility"] == "visible" and first["time"] <= _OPENING_TIME_MAX:
            opening.append((entity["name"], first["box"], source))
            if len(entity["anchors"]) > 1 and last["visibility"] == "visible":
                # 只跨三等分线不算：真实 200 条里站在线附近的轻微漂移让 23% 的镜写了镜末位置（11-06 看文字核对）。
                start, end = ((box[0] + box[2]) / 2 for box in (first["box"], last["box"]))
                if _third(start) != _third(end) and abs(end - start) >= _LATERAL_GAP:
                    ending_sides.append((entity["name"], _third(end), source))
    order = sorted(opening, key=lambda item: (item[1][0] + item[1][2]) / 2)
    relations = [_Relation(left[0], "left_of", right[0], right[2])
                 for left, right in zip(order, order[1:])
                 if (right[1][0] + right[1][2]) / 2 - (left[1][0] + left[1][2]) / 2 >= _LATERAL_GAP]
    relations += [_Relation(pair["near"], "in_front_of", pair["far"], f"layouts[{i}].front_of[{k}]")
                  for k, pair in enumerate(layout.get("front_of", []))
                  if {pair["near"], pair["far"]} <= characters & present]
    return _ShotLayout(_composition([box for _, box, _ in opening]), tuple(relations), tuple(ending_sides))


def _positions(row: _Row, shot: Mapping[str, Any], layout: _ShotLayout | None) -> dict[str, list[_Text]]:
    """每条相对位置挂在两人里在场名单靠后的那个人的描述上，说的是对先出现那人的位置。"""
    if layout is None:
        return {}
    order = {entry["name"]: k for k, entry in enumerate(shot["present"])}
    entries = {entry["name"]: entry for entry in shot["present"]}
    attached: dict[str, list[_Text]] = {}
    for relation in layout.relations:
        subject, kind, other = relation.subject, relation.relation, relation.object
        if order[subject] < order[other]:
            subject, kind, other = other, _INVERSE[kind], subject
        if any(entries[name].get("posture") == "not present" for name in (subject, other)):
            continue
        phrase = _RELATION_PHRASE[kind] + " " + other
        if any(re.search(re.escape(phrase) + r"\b", entries[subject].get(field) or "", re.IGNORECASE)
               for field in ("posture", "state")):
            continue
        attached.setdefault(subject, []).append(
            _concat(_RELATION_PHRASE[kind], " ", row.held(other, relation.source + ".name")))
    return attached


def _opening_shot(annotation: Mapping[str, Any], index: int):
    """首个明确before定义起态；仅在副本改值，并交付原字段冲突，不改源标注。"""
    shot = deepcopy(annotation["shots"][index])
    entries = {entry["name"]: (position, entry) for position, entry in enumerate(shot["present"])}
    outfits = {entry["name"]: (position, entry.get("outfit"))
               for position, entry in enumerate(annotation["characters"])}
    seen, sources, conflicts = set(), {}, []
    for position, change in enumerate(shot.get("changes") or []):
        name, field = change["name"], change["what"]
        key = (name, field)
        if key in seen:
            continue
        seen.add(key)
        if "before" not in change or change["before"] is None:
            continue
        present_position, entry = entries[name]
        original_source = f"shots[{index}].present[{present_position}].{field}"
        current = entry.get(field)
        if field == "outfit" and field not in entry and name in outfits:
            registry_position, current = outfits[name]
            original_source = f"characters[{registry_position}].outfit"
        before, before_source = change["before"], f"shots[{index}].changes[{position}].before"
        if current != before:
            conflicts.append({"name": name, "field": field, "source": original_source,
                              "value": current, "before_source": before_source, "before": deepcopy(before)})
        entry[field] = deepcopy(before)
        sources[key] = before_source
    return shot, sources, conflicts


def _stable_outfits(annotation: Mapping[str, Any]) -> dict[str, _Text]:
    """只登记整个已给故事内已知且一致的衣着；有变化或未知当前值时仍留本镜，不补猜。"""
    stable = {}
    for index, character in enumerate(annotation["characters"]):
        name = character["name"]
        if any(change["name"] == name and change["what"] == "outfit"
               for shot in annotation["shots"] for change in shot.get("changes") or []):
            continue
        values = []
        if character.get("outfit") is not None:
            values.append((_outfit(character["outfit"], "wearing"), f"characters[{index}].outfit"))
        for shot_index, shot in enumerate(annotation["shots"]):
            for position, entry in enumerate(shot["present"]):
                if entry["name"] != name:
                    continue
                value = entry.get("outfit", character.get("outfit"))
                source = (f"shots[{shot_index}].present[{position}].outfit" if "outfit" in entry
                          else f"characters[{index}].outfit")
                values.append((_outfit(value, "wearing"), source))
        if values and all(value for value, _ in values) and len({value.casefold() for value, _ in values}) == 1:
            stable[name] = _Text.entity(name, values[0][0], values[0][1])
    return stable


_CLOTHING_NOUNS = frozenset({
    "hat", "cap", "helmet", "headband", "shirt", "t-shirt", "top", "blouse", "coat", "jacket", "hoodie",
    "suit", "sweater", "cardigan", "vest", "dress", "tunic", "robe", "armor", "armour", "trousers",
    "pants", "shorts", "skirt", "jeans", "overalls", "boots", "shoes", "sneakers", "gloves", "scarf",
    "tie", "bowtie", "belt", "necklace", "earrings", "headscarf", "suspenders", "uniform", "outfit",
    "costume", "spacesuit", "tracksuit", "jumpsuit", "bodysuit", "gown", "apron", "blazer", "sari",
    "sweatshirt", "sweatpants", "turtleneck", "collar", "mask",
})


def _clothing_noun_fragment(text: str) -> str:
    """登记的衣物名词片段省模板冠词；引号、数字开头与内部文字保持原样。"""
    return re.sub(r"^(?:a|an)\s+(?=[A-Za-z])", "", text, flags=re.IGNORECASE)


def _appearance_without_stable_clothing(appearance: str, outfit: str) -> str:
    """只去被稳定衣着完整覆盖的尾部衣物短语；衣物、修饰及限定关系都须仍在。"""
    trailing = re.search(r"(?:,\s*|\s+)(?:wearing|dressed in|in)\s+(.+?)\.?$", appearance, re.IGNORECASE)
    if trailing and _garments_covered_by_outfit(trailing.group(1), outfit):
        return appearance[:trailing.start()].rstrip(", ")
    # 只比较顶层衣物，不能把shirt with a coat print里的coat当成另穿的外套。
    clothing = re.split(r"\s+(?:with|over|under|underneath|beneath|featuring)\s+", outfit, maxsplit=1,
                        flags=re.IGNORECASE)[0]
    clothing = re.sub(r"\b(?:a|an)\s+", "", clothing, flags=re.IGNORECASE).casefold()
    while appearance:
        features = _identity_parts(appearance)[1]
        if not features:
            break
        feature = features[-1]
        key = re.sub(r"\b(?:a|an)\s+", "", _identity_feature_key(feature), flags=re.IGNORECASE)
        words = re.findall(r"[\w-]+", key)
        if (not words or words[-1] not in _CLOTHING_NOUNS or re.search(r"\b(?:no|not|without)\b", key)
                or not re.search(r"(?<!\w)" + re.escape(key) + r"(?=$|,|\s+and\b)", clothing)):
            break
        tail = re.search(re.escape(feature) + r"\.?$", appearance, re.IGNORECASE)
        if tail is None:
            break
        prefix = appearance[:tail.start()]
        if feature.startswith(("wearing ", "dressed in ", "in ")):
            appearance = prefix.rstrip(", ")
        else:
            separator = re.search(r"(?:,\s*(?:and\s+)?|\s+and\s+|\s+with\s+)$", prefix)
            if separator is None:
                break
            appearance = prefix[:separator.start()].rstrip(", ")
    return appearance


def _garment_facts(text: str, *, require_shared_modifiers: bool) -> list[tuple[str, tuple[str, ...], str]]:
    """衣着字段的明确衣物中心词、前置修饰与后置限定；印花中的衣物不当另一件穿着。"""
    tokens = list(re.finditer(r"[\w-]+", text.casefold()))
    heads = []
    for i, token in enumerate(tokens):
        if token.group() not in _CLOTHING_NOUNS:
            continue
        tail = text[token.end():]
        if re.match(r"\s+(?:(?:small|large|embroidered|printed)\s+)?(?:print|pattern|design|emblem|image|drawing|picture|logo)\b", tail, re.IGNORECASE):
            continue
        if token.group() == "collar" and heads and re.search(r"\bwith\b", text[heads[-1].end():token.start()], re.IGNORECASE):
            continue
        heads.append(token)
    starts, ends = [0], []
    for left, right in zip(heads, heads[1:]):
        separators = list(re.finditer(r",|\b(?:and|with|over|under|underneath|beneath)\b", text[left.end():right.start()], re.IGNORECASE))
        if not separators:
            return []  # 无明确衣物边界，不通过词袋补猜关系。
        split = separators[-1]
        ends.append(left.end() + split.start())
        starts.append(left.end() + split.end())
    ends.append(len(text))
    result = []
    for head, start, end in zip(heads, starts, ends):
        prefix = _fact_words(text[start:head.start()])
        if set(prefix) & {"on", "of", "from", "inside", "featuring", "depicting", "showing"}:
            return []
        if require_shared_modifiers and not prefix and result:
            prefix = result[-1][1]  # 源“dark hat and coat”的两种读法都须被稳定衣着覆盖；目标不补猜此属性。
        result.append((head.group(), prefix, text[head.end():end].strip(" ,")))
    return result


def _garments_covered_by_outfit(phrase: str, outfit: str) -> bool:
    """仅删每项正面衣物事实都被同一稳定衣物保留的完整尾句；新增袖型/颜色细节不丢。"""
    if re.search(r"\b(?:no|not|without|unknown|unclear)\b", phrase + " " + outfit, re.IGNORECASE):
        return False
    # 层叠介词是关系而非衣物分隔符；无法逐项保留关系时只允许完整同值，不能把over当under。
    if re.search(r"\b(?:over|under|underneath|beneath)\b", phrase + " " + outfit, re.IGNORECASE):
        return _fact_words(phrase) == _fact_words(outfit)
    source = _garment_facts(phrase, require_shared_modifiers=True)
    target = _garment_facts(outfit, require_shared_modifiers=False)
    if not source or not target:
        return False
    for noun, attributes, qualifier in source:
        covered = False
        for other, other_attributes, other_qualifier in target:
            if noun != other:
                continue
            positions = iter(other_attributes)
            same_attributes = all(any(word == value for value in positions) for word in attributes)
            if same_attributes and (not qualifier or _contains_fact(other_qualifier, qualifier)):
                covered = True
                break
        if not covered:
            return False
    return True


def _compact_outfit_list(text: str) -> str:
    """只把已完整分出的独立衣物并列写成逗号列表；颜色内部and与共享修饰保持原文。"""
    if not text.startswith("wearing "):
        return text
    payload = text.removeprefix("wearing ")
    pieces = _identity_feature_phrases(payload)
    words = lambda value: [word for word in re.findall(r"[\w-]+", value.casefold()) if word != "and"]
    if (len(pieces) > 1 and all(re.findall(r"[\w-]+", piece.casefold())[-1] in _CLOTHING_NOUNS for piece in pieces)
            and words(payload) == [word for piece in pieces for word in words(piece)]):
        return "wearing " + ", ".join(pieces)
    return text


def _identity_registry(annotation: Mapping[str, Any], shot: Mapping[str, Any],
                       reference_entities: Iterable[str]) -> _Text:
    """统一登记的本镜子表G_i；只选本镜在场或用到的实体，不把未来人物外貌注入当前镜。"""
    # 在场的人物与登记物体全部写登记描述（按信息量写全，umT5 上限已全量核过）；
    # 不在场的物体仍只由持有/动作/变化/参考用途触发，G_i不是把整个源registry强塞进本镜。
    names = {entry["name"] for entry in shot["present"] if entry["name"].startswith(("Character ", "Object "))
             or entry.get("state") or entry.get("posture")}
    names.update(reference_entities)
    names.update(name for entry in shot["present"] for name in entry.get("holding") or [])
    # 物件原先也会在动作/变化中被展开；保留其识别信息，但未在场人物仍只写编号。
    names.update(name for name in PROSE_ENTITY.findall(shot["action"]) if name.startswith("Object "))
    for entry in shot["present"]:
        for field in ("state", "posture", "outfit"):
            names.update(PROSE_ENTITY.findall(entry.get(field) or ""))
    for change in shot.get("changes") or []:
        names.add(change["name"])
        if change["what"] == "holding":
            names.update(name for side in ("before", "after") for name in change.get(side) or [])
        else:
            for side in ("before", "after"):
                names.update(PROSE_ENTITY.findall(change.get(side) or ""))
    place = shot.get("place")
    if place not in (None, "none", "unknown"):
        names.add(place)
    present = {entry["name"] for entry in shot["present"]}
    sections, stable_outfits = [], _stable_outfits(annotation)
    for kind in ("characters", "objects", "places"):
        lines = []
        for i, entry in enumerate(annotation[kind]):
            name = entry["name"]
            if name not in names or (kind == "characters" and name not in present):
                continue
            source = f"{kind}[{i}]"
            single_place = kind == "places" and len(annotation["places"]) == 1
            parts = (["Setting: "] if single_place else ["Setting: ", _Text.entity(name, name, source + ".name")]) if kind == "places" else (
                     [_Text.entity(name, name, source + ".name")])
            if entry["appearance"]:
                # 登记是名词短语，去掉模板式不定冠词，不删除任何描述属性。
                appearance = ARTICLE.sub("", entry["appearance"])
                same_outfit = None
                if kind == "characters" and name in stable_outfits:
                    outfit = stable_outfits[name].text.removeprefix("wearing ")
                    same_outfit = re.search(r"(?<!\w)" + re.escape(outfit) + r"(?!\w)", appearance, re.IGNORECASE)
                    if same_outfit is None:
                        appearance = _appearance_without_stable_clothing(appearance, outfit)
                value = (_concat(_Text.entity(name, appearance[:same_outfit.start()], source + ".appearance"),
                                 _Text.entity(name, _clothing_noun_fragment(same_outfit.group()), stable_outfits[name].spans[0][3]),
                                 _Text.entity(name, appearance[same_outfit.end():], source + ".appearance"))
                         if same_outfit else _Text.entity(name, appearance, source + ".appearance"))
                parts += (["" if single_place else ", " if kind == "places" else ": ", value])
            else:
                same_outfit = None
            if kind == "characters" and name in stable_outfits and not same_outfit:
                outfit = stable_outfits[name]
                parts += [", " if entry["appearance"] else ": ", _Text.entity(name,
                    _compact_outfit_list("wearing " + _clothing_noun_fragment(outfit.text.removeprefix("wearing ")))
                    if outfit.text.startswith("wearing ") else outfit.text, outfit.spans[0][3])]
            lines.append(_sentence(_concat(*parts)))
        if lines:
            sections.append(_separated(lines, " "))
    # 只有全故事已知且一致的衣着进G；易变衣着、持物、姿态和state留在本镜。
    return _separated(sections, " ")


def _identity_parts(appearance: str) -> tuple[str, list[str]]:
    """保留中心名词短语，只在完整名词属性之间拆并列；不切颜色内部的and。"""
    if not appearance.strip():
        return "", []
    match = re.search(r"\s+(with|wearing|dressed in|in)\s+", appearance)
    head, marker, tail = ((appearance[:match.start()], match.group(1), appearance[match.end():])
                          if match else (appearance.split(",", 1)[0], "with", appearance.partition(",")[2]))
    head = "the " + re.sub(r"^(?:an?|the)\s+", "", head.strip(), flags=re.IGNORECASE)
    features = _identity_feature_phrases(tail)
    return head, [feature if marker == "with" else marker + " " + feature for feature in features]


_IDENTITY_FEATURE_NOUNS = frozenset({
    "hair", "beard", "mustache", "moustache", "goatee", "skin", "eyes", "glasses", "build", "complexion",
    "face", "jaw", "horns", "ears", "antennae", "body", "mane", "wings", "fur", "tail", "spots", "markings",
    "ponytail", "braids", "bangs", "bun", "freckles", "tattoos", "scar", "scars", "lipstick", "mask",
    "hat", "cap", "helmet", "headband", "clip", "shirt", "t-shirt", "top", "blouse", "coat", "jacket",
    "hoodie", "suit", "sweater", "cardigan", "vest", "dress", "tunic", "robe", "armor", "armour",
    "trousers", "pants", "shorts", "skirt", "jeans", "overalls", "boots", "shoes", "sneakers", "gloves",
    "scarf", "tie", "bowtie", "belt", "necklace", "earrings", "headscarf", "suspenders",
    "uniform", "badge", "outfit", "bag", "backpack", "costume", "spacesuit", "tracksuit", "jumpsuit",
    "bodysuit", "gown", "apron", "blazer", "sari", "sweatshirt", "sweatpants", "hairstyle",
})


def _identity_feature_phrases(text: str) -> list[str]:
    """完整名词属性才作为候选；颜色逗号和省略修饰语的衣物并列保持完整。"""
    def noun_ending(phrase):
        words = re.findall(r"[\w-]+", phrase.casefold())
        return bool(words) and words[-1] in _IDENTITY_FEATURE_NOUNS

    def conjunctions(phrase):
        for join in re.finditer(r"\s+and\s+", phrase):
            left, right = phrase[:join.start()].strip(), phrase[join.end():].strip()
            right_words = re.findall(r"[\w-]+", right.casefold())
            independent_right = len(right_words) > 1 or right_words[-1:] in (["glasses"], ["eyes"], ["ears"], ["horns"], ["antennae"], ["pants"], ["trousers"], ["shorts"], ["jeans"], ["boots"], ["shoes"], ["gloves"], ["suspenders"])
            if noun_ending(left) and noun_ending(right) and independent_right:
                return conjunctions(left) + conjunctions(right)
            if noun_ending(left) and noun_ending(right) and not independent_right:
                # dark hat and coat可能省略共享修饰；原事实不动，但不从这里造单个短属性。
                return []
        return [phrase]

    phrases, pending = [], []
    for clause in text.split(","):
        pending.append(clause.strip().rstrip("."))
        phrase = ", ".join(pending)
        if not noun_ending(phrase):
            continue
        phrase = re.sub(r"^and\s+", "", phrase)
        marker = re.match(r"(wearing|dressed in|in)\s+", phrase)
        payload = phrase[marker.end():] if marker else phrase
        phrases.extend((marker.group() if marker else "") + part for part in conjunctions(payload))
        pending = []
    return phrases


def _identity_feature_key(feature: str) -> str:
    value = re.sub(r"^(?:wearing|dressed in|in)\s+", "", feature, flags=re.IGNORECASE)
    return re.sub(r"^(?:an?|the)\s+", "", value, flags=re.IGNORECASE).casefold().strip()


def _identity_noun(head: str) -> str:
    words = re.findall(r"[\w-]+", head.casefold())
    return next((word for word in words if word in {"man", "woman", "boy", "girl", "person", "child", "baby", "toddler"}), words[-1] if words else "")


def _identity_feature_conflicts(feature: str, appearance: str, outfit: str) -> bool:
    """更具体的同一名词属性仍覆盖短属性；不能完整分出该属性时不声称排他。"""
    key = _identity_feature_key(feature)
    words = re.findall(r"[\w-]+", key)
    if not words:
        return True
    noun = words[-1]
    attributes = [*_identity_parts(appearance)[1],
                  *_identity_feature_phrases(outfit.removeprefix("wearing "))]
    comparisons = [re.findall(r"[\w-]+", _identity_feature_key(value)) for value in attributes]
    same_noun = [value for value in comparisons if value and value[-1] == noun]
    for value in same_noun:
        # 不把不同衣物或属性上的修饰词拼接成一个属性；无法拆清的范围只作不确定冲突。
        if any(word in _IDENTITY_FEATURE_NOUNS or word in {"with", "in", "on", "from", "over", "under", "behind"}
               for word in value[:-1]):
            return True
        position = 0
        for word in value:
            if position < len(words) and word == words[position]:
                position += 1
        if position == len(words):
            return True
    descriptions = (appearance.casefold(), outfit.casefold())
    if any(re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", value) for value in descriptions):
        return True
    # 原文有该名词、但属性拆分无法完整归属时，保留完整事实而不猜它是否可区分。
    return not same_noun and any(noun in re.findall(r"[\w-]+", value) for value in descriptions)


def _short_identity_choices(annotation: Mapping[str, Any], shot: Mapping[str, Any],
                            field_sources: Mapping[tuple[str, str], str], source: str) -> dict[str, tuple[_Text, str]]:
    """ID已区分实体；短称优先同镜排他属性，否则只用原中心词和一项原属性，完整事实留登记。"""
    characters = {entry["name"]: (i, entry) for i, entry in enumerate(annotation["characters"])}
    present = {entry["name"]: (i, entry) for i, entry in enumerate(shot["present"])
               if entry["name"] in characters}
    parts = {name: _identity_parts(characters[name][1]["appearance"]) for name in present}
    outfits = {name: _outfit(entry.get("outfit", characters[name][1].get("outfit")), "wearing")
               for name, (_, entry) in present.items()}
    descriptions = {name: " ".join((characters[name][1]["appearance"], outfits[name])).casefold() for name in present}
    choices = {}
    for name, (head, features) in parts.items():
        registry_position, character = characters[name]
        appearance_source = f"characters[{registry_position}].appearance"
        full = _Text("")
        if character["appearance"]:
            brief_head = _nominal_head(character["appearance"])
            if character["kind"] == "group":
                people = list(dict.fromkeys(match.group().casefold() for match in re.finditer(
                    r"\b(?:women|men|woman|man|girls|boys|girl|boy|children|people|soldiers|officers|rabbits|dogs|cats)\b",
                    character["appearance"], re.IGNORECASE)))
                if len(people) > 1:
                    brief_head = "the " + " and ".join(people)
            full = _Text.entity(name, brief_head, appearance_source)
            if features and character["kind"] != "group":
                feature = features[0]
                marker = re.match(r"(wearing|dressed in|in)\s+", feature)
                full = _concat(full, " " + marker.group() if marker else " with ",
                               _Text.entity(name, feature[marker.end():] if marker else feature, appearance_source))
        if character["kind"] == "group":
            choices[name] = (full, "group_preserved")
            continue
        if not head:
            choices[name] = (full, "missing_appearance")
            continue
        noun = _identity_noun(head)
        other_names = [other for other, (other_head, _) in parts.items() if other != name and (
            _identity_noun(other_head) == noun
            or noun == "person" or _identity_noun(other_head) == "person"
            or (characters[other][1]["kind"] == "group" and re.search(r"\b" + re.escape(noun) + r"\b", descriptions[other])))]
        others = [descriptions[other] for other in other_names]
        if re.search(r"\b" + re.escape(noun) + r"\s+\w+ing\b", head):
            choices[name] = (full, "short_shared_feature")
            continue
        if (not features and head.removeprefix("the ") == re.sub(r"^(?:an?|the)\s+", "", character["appearance"], flags=re.IGNORECASE)
                and noun in {"man", "woman", "boy", "girl", "person", "child", "baby", "toddler", "cow", "rabbit", "dog", "cat", "horse", "pony", "seapony", "alien", "creature", "robot"}
                and head != "the " + noun and not any(re.search(r"(?<!\w)" + re.escape(head.removeprefix("the ").casefold()) + r"(?!\w)", other) for other in others)):
            choices[name] = (full, "unique_noun_phrase")
            continue
        position, entry = present[name]
        outfit_source = field_sources.get((name, "outfit"), f"{source}.present[{position}].outfit" if "outfit" in entry else f"characters[{registry_position}].outfit")
        clothes = _identity_feature_phrases(outfits[name].removeprefix("wearing "))
        current_clothes = {_identity_feature_key(feature): feature for feature in clothes}
        candidates = []
        for feature in features:
            key = _identity_feature_key(feature)
            if key in current_clothes:
                candidates.append(("wearing " + current_clothes[key], outfit_source))
            elif (feature.startswith(("wearing ", "dressed in ", "in ")) and outfits[name]
                  and not re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", outfits[name].casefold())):
                continue
            else:
                candidates.append((feature, appearance_source))
        candidates += [("wearing " + feature, outfit_source) for feature in clothes]
        selected = next(((feature, origin) for feature, origin in candidates
                         if _identity_feature_key(feature) and not any(_identity_feature_conflicts(
                             feature, characters[other][1]["appearance"], outfits[other]) for other in other_names)), None)
        if selected:
            feature, origin = selected
            marker = re.match(r"(wearing|dressed in|in)\s+", feature)
            separator = " " + marker.group() if marker else " with "
            value = feature[marker.end():] if marker else feature
            choices[name] = (_concat(_Text.entity(name, head, appearance_source), separator,
                                     _Text.entity(name, value, origin)), "unique_feature")
        else:
            choices[name] = (full, "short_shared_feature")
    return choices


_PERSON_NOUN_ALIASES = {
    "men": "man", "women": "woman", "boys": "boy", "girls": "girl", "people": "person",
    "persons": "person", "children": "child", "babies": "baby", "toddlers": "toddler",
    "policeman": "police officer", "policemen": "police officer", "policewoman": "police officer",
    "policewomen": "police officer", "police officers": "police officer", "officers": "officer",
    "soldiers": "soldier", "guards": "guard", "students": "student", "doctors": "doctor",
    "nurses": "nurse", "workers": "worker", "skeletons": "skeleton", "characters": "character",
    "rabbits": "rabbit", "dogs": "dog", "cats": "cat", "horses": "horse", "ponies": "pony",
    "creatures": "creature", "aliens": "alien", "robots": "robot", "cows": "cow", "birds": "bird",
}
_PERSON_NOUNS = frozenset(_PERSON_NOUN_ALIASES) | frozenset(_PERSON_NOUN_ALIASES.values()) | frozenset({
    "man", "woman", "boy", "girl", "person", "child", "baby", "toddler", "police officer", "officer",
    "soldier", "guard", "student", "doctor", "nurse", "worker", "skeleton", "character", "rabbit",
    "dog", "cat", "horse", "pony", "alien", "creature", "robot", "cow", "seapony",
})
_PERSON_HEAD_ADJECTIVES = frozenset({
    "young", "old", "older", "elderly", "middle-aged", "younger", "little", "small", "tall", "short",
    "muscular", "slim", "thin", "heavy", "heavyset", "bearded", "bald", "blonde", "blond", "red-haired",
    "dark-haired", "black-haired", "white-haired", "grey-haired", "gray-haired", "first", "second", "alien",
})
def _natural_character_mention_pattern(nouns: Iterable[str], adjectives: Iterable[str]) -> re.Pattern[str]:
    return re.compile(
        r"(?<![\w'-])(?:(?:the|an?|another|one|two|three|four|five|six|several|some|both|all)\s+)?"
        r"(?:(?:one|two|three|four|five|six)\s+)?"
        r"(?:(?:group|pair)\s+of\s+)?"
        r"(?:(?:" + "|".join(sorted(set(adjectives), key=lambda value: (-len(value), value))) + r")\s+)*"
        r"(?P<noun>" + "|".join(re.escape(noun) for noun in sorted(set(nouns) - {""}, key=lambda value: (-len(value), value))) + r")(?![\w-])",
        re.IGNORECASE,
    )
_NOUN_MODIFIER_END = re.compile(
    r"[.!?;,]|\b(?:while|when|then|but|as|who|whose|that|which|is|are|was|were|has|have|had|"
    r"stands?|sits?|walks?|runs?|looks?|turns?|talks?|speaks?|smiles?|holds?|takes?|picks?|puts?|"
    r"raises?|lowers?|moves?|reaches?|points?|opens?|closes?|pulls?|pushes?|keeps?|continues?|"
    r"starts?|begins?|stops?|drives?|rides?|watches?|leans?|bends?|nods?|shakes?|gestures?|"
    r"jumps?|falls?|laughs?|cries|cry|shouts?|screams?|touches?|touch|hands?|gives?|faces?|"
    r"examines?|wipes?|carries|carry|brushes?|brush|covers?|removes?|enters?|leaves?|passes?|"
    r"stares?|glances?|follows?|approaches?|steps?|waits?|kisses?|hugs?|dances?|sings?)\b",
    re.IGNORECASE,
)


def _fact_words(text: str) -> tuple[str, ...]:
    return tuple(word for word in re.findall(r"[\w-]+", text.casefold())
                 if word not in {"a", "an", "the", "is", "are", "wearing", "wears", "dressed"})


def _contains_fact(text: str, fact: str) -> bool:
    haystack, needle = _fact_words(text), _fact_words(fact)
    negatives = {"no", "not", "without", "never", "neither", "nor"}
    if set(haystack) & negatives != set(needle) & negatives:
        return False
    return bool(needle) and any(haystack[i:i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1))


def _nominal_head(appearance: str) -> str:
    head = _identity_parts(appearance)[0]
    return re.split(r"\s+(?:at|on|near|beside|behind|next to|standing|sitting|lying|walking|looking|holding|seated)\b",
                    head, maxsplit=1, flags=re.IGNORECASE)[0]


def action_reference_candidates(annotation: Mapping[str, Any], index: int) -> list[dict[str, Any]]:
    """原action的名词短语→本镜候选实体；明确人数可对应多个在场ID。

    不把泛称person/character或未知修饰下的唯一present当身份依据。原ID同位短称、
    无法安全分界的并列修饰保留；输出范围不跨另一个人物短语。
    """
    shot = annotation["shots"][index]
    action = shot["action"]
    registered = {entry["name"]: entry for entry in annotation["characters"]}
    present = {entry["name"]: entry for entry in shot["present"] if entry["name"] in registered}
    heads = {name: _nominal_head(registered[name]["appearance"]) for name in present}
    nouns = set(_PERSON_NOUNS)
    adjectives = set(_PERSON_HEAD_ADJECTIVES)
    non_nouns = adjectives | {"the", "a", "an", "group", "pair", "of", "various", "several", "some", "large", "small",
                              "furry", "covered", "visible", "partial", "clear", "unknown"}
    for name, head in heads.items():
        words = re.findall(r"[\w-]+", head.casefold())
        known = [word for word in words if word in _PERSON_NOUNS]
        noun = _identity_noun(head) if known else (words[-1] if words and words[-1] not in non_nouns else "")
        if noun:
            nouns.add(noun)
        adjectives.update(word for word in words if word not in nouns | {"the", "a", "an", "group", "pair", "of", "and"}
                          and word not in {"at", "on", "near", "beside", "behind", "standing", "sitting", "lying", "walking"})
    noun_aliases = dict(_PERSON_NOUN_ALIASES)
    for head in heads.values():
        words = re.findall(r"[\w-]+", head.casefold())
        if len(words) > 1 and all(word in nouns for word in words[-2:]):
            compound = " ".join(words[-2:])
            nouns.add(compound)
            noun_aliases[compound] = noun_aliases.get(words[-1], words[-1])
        if words and words[-1] in nouns and words[-1] != "character":
            # 自然称呼可在已登记的中心词后补character；它属于同一NP，不能留到二次编译。
            compound = words[-1] + " character"
            nouns.add(compound)
            noun_aliases[compound] = noun_aliases.get(words[-1], words[-1])
    pattern = _natural_character_mention_pattern(nouns, adjectives)
    opening, sources, _ = _opening_shot(annotation, index)
    identities = _short_identity_choices(annotation, opening, sources, f"shots[{index}]")
    protected = []
    for explicit in PROSE_ENTITY.finditer(action):
        if explicit.group() not in identities:
            continue
        identity = identities[explicit.group()][0].text
        for phrase in (identity, re.sub(r"^the\s+", "a ", identity), re.sub(r"^the\s+", "an ", identity)):
            if phrase and action[explicit.end():].startswith(", " + phrase):
                protected.append((explicit.end(), explicit.end() + len(", " + phrase)))
    result = []
    for match in pattern.finditer(action):
        if any(start <= match.start() < end for start, end in protected):
            continue
        noun_text = match.group("noun").casefold()
        if noun_text == "character" and re.match(r"\s+[1-9]\d*\b", action[match.end():]):
            continue
        previous_ids = list(PROSE_ENTITY.finditer(action, 0, match.start()))
        if previous_ids:
            tail = action[previous_ids[-1].end():match.start()]
            if tail.startswith(",") and not any(hit.group() != "," for hit in _NOUN_MODIFIER_END.finditer(tail)):
                continue
        prefix = re.search(r"\b(?:the|an?|another)\s+(?:[\w-]+\s+){1,5}$", action[:match.start()], re.IGNORECASE)
        own_article = re.match(r"(?:the|an?|another|one|two|three|four|five|six|both|all|some|several)\b", match.group(), re.IGNORECASE)
        prefix_unknown = (not own_article and prefix is not None and not _NOUN_MODIFIER_END.search(prefix.group())
                          and not re.search(r"\b(?:and|but|while|when|as|then|to|of|on|in|at|for|with)\b", prefix.group(), re.IGNORECASE))
        noun = noun_aliases.get(noun_text, noun_text)
        plural = (noun_text.split()[-1] in set(_PERSON_NOUN_ALIASES) - {"policeman", "policewoman"}) or bool(
            re.search(r"\b(?:group|pair|two|three|four|five|six|several|some|both|all)\b", match.group(), re.IGNORECASE))
        number = next((count for word, count in {"both": 2, "pair": 2, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}.items()
                       if re.search(r"\b" + word + r"\b", match.group(), re.IGNORECASE)), None)
        exhaustive = bool(re.search(r"\ball\b", match.group(), re.IGNORECASE))
        end = match.end()
        conjoined = re.match(r"\s+and\s+", action[end:], re.IGNORECASE)
        if conjoined and pattern.match(action, end + conjoined.end()):
            prefix_unknown = True  # 未证明整个人物并列集合前，不将其中一个子集偷换为群体ID。
        modifier = re.match(r"\s+(?:with|wearing|dressed in|in)\s+", action[end:], re.IGNORECASE)
        descriptor = ""
        if modifier:
            begin = end + modifier.end()
            boundary = _NOUN_MODIFIER_END.search(action, begin)
            limit = boundary.start() if boundary else len(action)
            descriptor = action[begin:limit].rstrip()
            words = re.findall(r"[\w-]+", descriptor.casefold())
            next_person = pattern.search(action, begin)
            split_list = boundary is not None and boundary.group() == "," and bool(re.match(
                r"\s*(?:and\s+)?(?:[\w-]+\s+){0,3}(?:" + "|".join(_CLOTHING_NOUNS) + r")\b", action[boundary.end():], re.IGNORECASE))
            if split_list:
                following_start = boundary.end() + len(action[boundary.end():]) - len(action[boundary.end():].lstrip())
                if pattern.match(action, following_start):
                    split_list = False  # 逗号后是完整另一人物NP，不能因其衣着词误作当前衣着并列。
            if (words and words[-1] in _IDENTITY_FEATURE_NOUNS | _CLOTHING_NOUNS
                    and not split_list and not (next_person and next_person.start() < limit)):
                end = begin + len(descriptor)
        head_attributes = re.sub(r"^(?:(?:the|an?|another|one|two|three|four|five|six|several|some|both|all)\s+)+", "",
                                 match.group(), flags=re.IGNORECASE)
        head_attributes = re.sub(r"^(?:group|pair)\s+of\s+", "", head_attributes, flags=re.IGNORECASE)
        head_attributes = head_attributes[:len(head_attributes) - len(noun_text)].strip()
        names, matching, groups = [], [], []
        prior_mention = (any(name.startswith("Character ") for name in PROSE_ENTITY.findall(action[:match.start()]))
                         or pattern.search(action[:match.start()]) is not None)
        for name, entry in present.items():
            character = registered[name]
            description = " ".join(filter(None, (character["appearance"], entry.get("outfit", character.get("outfit")))))
            head_words = re.findall(r"[\w-]+", heads[name].casefold())
            aliases = {_PERSON_NOUN_ALIASES.get(word, word) for word in head_words}
            generic = noun in {"person", "character"}
            same_noun = (noun in aliases or generic and character["kind"] != "animal"
                         or noun == "police officer" and bool(re.search(r"\bpolice\b", description, re.IGNORECASE)))
            if not same_noun or not plural and character["kind"] == "group":
                continue
            if prefix_unknown:
                continue
            names.append(name)
            if character["kind"] == "group":
                groups.append(name)
            head_attribute_words = _fact_words(head_attributes)
            head_words_position = iter(_fact_words(heads[name]))
            head_matches = all(any(word == current for current in head_words_position) for word in head_attribute_words)
            verified = ((not head_attributes or head_matches)
                        and (not modifier or end > match.end() and _contains_fact(description, descriptor)))
            if (verified and not re.match(r"(?:another|other)\b", match.group(), re.IGNORECASE)
                    and (not generic or descriptor or plural and (number is not None or exhaustive))):
                matching.append(name)
        # 前文ID永不删候选以制造唯一。已有角色提及后的a/an泛称没有独占外貌时保守未定，
        # 前文是原自然称呼还是其物化ID结果相同；another本身不作确定性指派。
        if (re.match(r"an?\b", match.group(), re.IGNORECASE) and prior_mention
                and not ((head_attributes or descriptor and end > match.end()) and len(names) > 1 and len(matching) == 1)):
            matching = []
        individuals = [name for name in matching if name not in groups]
        matched = None
        if plural:
            group_matches = [name for name in matching if name in groups]
            if len(group_matches) == 1 and not individuals:
                matched = group_matches
            elif not group_matches and ((number is not None and len(individuals) == number) or exhaustive and len(individuals) > 1):
                matched = individuals
        elif len(matching) == 1:
            matched = matching
        result.append({"start": match.start(), "end": end, "phrase": action[match.start():end], "candidates": names,
                       "matched_names": matched, "cardinality": number if plural else 1,
                       "group_candidates": groups, "plural": plural, "exhaustive": exhaustive})
    return result


def resolve_action_references(annotation: Mapping[str, Any], index: int, *,
                              resolutions: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """唯一指称改写入口：只接受原名词短语→在场ID集合，输出编号和原事实短称。

    无自由改写文本；原动作与所有未选字符直接切片保留。训练标注和推理计划共用，
    物化后仍是原schema的action；既有显式ID和已编号同位短称不被二次膨胀。
    """
    action = annotation["shots"][index]["action"]
    candidates = action_reference_candidates(annotation, index)
    by_span = {(item["start"], item["end"]): item for item in candidates}
    accepted, unresolved_reasons = {}, {}
    for item in resolutions:
        if set(item) != {"start", "end", "names"} or type(item["start"]) is not int or type(item["end"]) is not int:
            raise ValueError("an action reference resolution requires exactly integer start/end and names")
        key = item["start"], item["end"]
        if key in accepted or key not in by_span:
            raise ValueError("an action reference resolution must select a distinct original noun phrase")
        candidate, names = by_span[key], item["names"]
        if not isinstance(names, (list, tuple)) or not names or any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
            raise ValueError("action reference names must be distinct nonempty identity strings")
        if not set(names) <= set(candidate["candidates"]):
            raise ValueError("an action reference resolution must select present candidate identities")
        if len(names) == 1 and candidate["plural"]:
            if names[0] not in candidate["group_candidates"]:
                raise ValueError("a plural noun phrase requires a registered group or multiple identities")
        elif candidate["cardinality"] is not None and len(names) != candidate["cardinality"]:
            raise ValueError("action reference identity count differs from the original noun phrase")
        elif len(names) > 1 and candidate["cardinality"] is None and not candidate["exhaustive"]:
            unresolved_reasons[key] = "identity_not_determined"
            continue
        elif any(name in candidate["group_candidates"] for name in names):
            raise ValueError("individual identities cannot be mixed with a registered group")
        if candidate["matched_names"] is not None and list(names) != candidate["matched_names"]:
            raise ValueError("an offline resolution cannot replace a deterministic unique identity set")
        accepted[key] = list(names)
    shot, sources, _ = _opening_shot(annotation, index)
    short = {name: text.text for name, (text, _) in _short_identity_choices(annotation, shot, sources, f"shots[{index}]").items()}
    replacements, unresolved = [], []
    for candidate in candidates:
        key = candidate["start"], candidate["end"]
        names = accepted.get(key, candidate["matched_names"])
        if names is None:
            unresolved.append({**candidate, **({"reason": unresolved_reasons[key]} if key in unresolved_reasons else {})})
            continue
        # 同位语的终止逗号属于替换短语，原动词/标点从未改动。
        following = action[candidate["end"]:]
        possessive = following.startswith(("'s", "’s"))
        labels = [name + (", " + short[name] if short.get(name) and not possessive and len(names) == 1 else "") for name in names]
        close = "" if not following.strip() or following.startswith((",", ".", "!", "?", ";", ":", "'s", "’s")) else ","
        if len(labels) == 1:
            replacement = labels[0] + close
        else:
            collective = candidate["phrase"]
            collective = collective[0].lower() + collective[1:]
            replacement = ", ".join(labels[:-1]) + " and " + labels[-1]
            if not possessive:
                replacement += ", " + collective
            replacement += close
        replacements.append({**candidate, "names": names, "replacement": replacement,
                             "method": "offline" if key in accepted else "deterministic"})
    pieces, cursor = [], 0
    for item in replacements:
        if item["start"] < cursor:
            raise ValueError("action reference noun phrases overlap")
        pieces.extend((action[cursor:item["start"]], item["replacement"]))
        cursor = item["end"]
    pieces.append(action[cursor:])
    return {"action": "".join(pieces), "replacements": replacements, "unresolved": unresolved}


def _physical_state(text: str) -> frozenset[str]:
    if re.search(r"\b(?:not|no longer|without|then|after|before|later|or|some|one of|ready|waking)\b", text, re.IGNORECASE):
        return frozenset()
    if re.match(r"^(?:moving|running|walking|flying)\s+(?:(?:his|her|their|its|the|a|both|one)\s+)*(?:arms?|legs?|hands?|fingers?|claws?|head|hair|lips?|mouth|eyes?|upper body)\b", text, re.IGNORECASE):
        return frozenset()
    from mspc.data.storyboard import posture_words

    words = posture_words(text)
    first = words[0] if words else None
    states = {name for name, stem in (("sitting", "sit"), ("standing", "stand"), ("lying", "lie")) if first == stem}
    if re.match(r"^(?:stationary|motionless|parked)(?:\b|$)|^still$|^(?:standing|sitting|lying) still\b", text, re.IGNORECASE):
        states.add("stationary")
    if re.match(r"^(?:driving|walking|running|travelling|traveling|flying)\b|^moving (?:forward|backward|towards?|away|along|across|through|down the|up the)\b", text, re.IGNORECASE):
        states.add("moving")
    return frozenset(states)


def _posture_state_conflict(posture: str, state: str) -> bool:
    left, right = _physical_state(posture), _physical_state(state)
    if not left or not right:
        return False
    return ("stationary" in left and "moving" in right or "moving" in left and "stationary" in right
            or bool(left & {"sitting", "standing", "lying"}) and bool(right & {"sitting", "standing", "lying"})
            and not left & right & {"sitting", "standing", "lying"})


def _normalized_opening_entries(annotation: Mapping[str, Any], shot: Mapping[str, Any],
                                field_sources: Mapping[tuple[str, str], str]):
    """开镜与镜末分别处理；posture持有物理姿态，state补充状态，不重复登记事实。"""
    registry = {entry["name"]: entry for kind in ("characters", "objects", "places") for entry in annotation[kind]}
    present = {entry["name"]: entry for entry in shot["present"]}
    from mspc.data.storyboard import _same, posture_words

    output, diagnostics = [], []
    for original in shot["present"]:
        entry = dict(original)
        name = entry["name"]
        registered = registry[name]
        known = " ".join(filter(None, (registered["appearance"], entry.get("outfit", registered.get("outfit")))))
        posture = entry.get("posture") or ""
        states, seen = [], set()
        for fragment in re.split(r"\s*[,;]\s*", entry.get("state") or ""):
            if not fragment:
                continue
            key = _fact_words(fragment)
            reason = None
            posture_duplicate = (_same("posture", posture, fragment)
                                 and set(posture_words(fragment)) <= set(posture_words(posture)))
            held = re.fullmatch(r"holding\s+(.+)", fragment, re.IGNORECASE)
            holding_duplicate = held is not None and any(
                _contains_fact(value, held.group(1)) for other in entry.get("holding") or []
                for value in (other, registry[other]["appearance"]))
            held_by = re.fullmatch(r"held by (Character [1-9]\d*)\.?", fragment)
            inverse_holding_duplicate = held_by is not None and name in present.get(held_by.group(1), {}).get("holding", [])
            if key in seen or _contains_fact(posture, fragment) or posture_duplicate or holding_duplicate:
                reason = "duplicate_posture_or_state"
            elif inverse_holding_duplicate:
                reason = "duplicate_holding_relation"
            elif _contains_fact(known, fragment):
                reason = "duplicate_registered_appearance_or_outfit"
            elif _posture_state_conflict(posture, fragment):
                # 显式changes.before先定义同一时刻；否则物理姿态归posture，state不再重写一份相反姿态。
                if (name, "state") in field_sources and (name, "posture") not in field_sources:
                    diagnostics.append({"name": name, "kind": "posture_state_conflict", "posture": posture,
                                        "state": fragment, "kept_field": "state", "removed_field": "posture"})
                    entry.pop("posture", None)
                    posture = ""
                else:
                    reason = "posture_state_conflict"
            if reason:
                diagnostics.append({"name": name, "kind": reason, "posture": posture,
                                    "state": fragment, "kept_field": "posture" if reason == "posture_state_conflict" else None,
                                    "removed_field": "state"})
                if reason == "posture_state_conflict":
                    remainder = re.sub(r"^(?:stationary|motionless|parked|standing|sitting|seated|lying)(?:\s+(?:up|down))?\b\s*", "", fragment, flags=re.IGNORECASE)
                    remainder = re.sub(r"^(?:and|while)\s+", "", remainder, flags=re.IGNORECASE)
                    if remainder and remainder != fragment:
                        states.append(remainder)
            else:
                seen.add(key)
                states.append(fragment)
        if states:
            entry["state"] = ", ".join(states)
        else:
            entry.pop("state", None)
        output.append(entry)
    return output, diagnostics


def _opening_without_repeated_postures(shot: Mapping[str, Any], field_sources: Mapping[tuple[str, str], str]):
    """同一明确ID的原动作已逐字包含同一静态姿态时只写一次；before与变化不压缩。"""
    predicates = _explicit_action_subjects(shot["action"])
    by_name = {}
    for match in PROSE_ENTITY.finditer(shot["action"]):
        if match.start() in predicates:
            by_name.setdefault(match.group(), predicates[match.start()])
    # 物化自然名词短语可在完整ID集合后有一份同位短称。只认有限的静态姿态动词，
    # 不用这个分支改写一般英语，也不把宾语分词当当前执行者。
    static_subject = re.compile(
        r"(?:^|[.!?;]\s*|\b(?:while|when|as|then|and|but)\s+)"
        r"(?P<names>Character [1-9]\d*(?:(?:\s*,\s*|\s*,?\s+and\s+)Character [1-9]\d*)*)"
        r"(?:,\s+(?:(?!Character [1-9]\d*|Object [1-9]\d*|[.!?;]).)+?,)?\s+"
        r"(?P<predicate>(?:sits?|stands?|kneels?|lies?|leans?|bends?|crouch(?:es)?|squats?)\b[^.!?;]*)",
        re.IGNORECASE)
    for match in static_subject.finditer(shot["action"]):
        for name in PROSE_ENTITY.findall(match.group("names")):
            by_name.setdefault(name, match.group("predicate"))
    from mspc.data.storyboard import posture_stem, posture_words
    changing = {change["name"] for change in shot.get("changes") or [] if change["what"] == "posture"}
    entries = []
    for entry in shot["present"]:
        posture = entry.get("posture") or ""
        predicate = by_name.get(entry["name"], "")
        stem = posture_stem(posture)
        if (stem in {"sit", "stand", "kneel", "lie", "lean", "bend", "crouch", "squat"}
                and entry["name"] not in changing and (entry["name"], "posture") not in field_sources
                and predicate and not re.match(r"^(?:sits? down|stands? up|kneels? down|lies? down|bends? down|crouches? down|squats? down)\b", predicate)):
            current_words, action_words = posture_words(posture), posture_words(predicate)
            if current_words and action_words[:len(current_words)] == current_words:
                entry = {key: value for key, value in entry.items() if key != "posture"}
        entries.append(entry)
    return entries


def _require_target_references(row: _Row, shot: Mapping[str, Any], references: Sequence[Mapping[str, Any]]) -> None:
    """只校验实际slot和可见ID供对象用途保护；不把映射或未知身份写进目标文字。"""
    if isinstance(references, (str, bytes)) or not isinstance(references, (list, tuple)):
        raise TypeError("references must be a sequence of actual reference slots and visible entities")
    known = set(row.characters) | set(row.things)
    relevant = {entry["name"] for entry in shot["present"]}
    relevant.update(name for entry in shot["present"] for name in entry.get("holding") or [])
    if shot.get("place") not in (None, "none", "unknown"):
        relevant.add(shot["place"])
    indices = set()
    for reference in references:
        if not isinstance(reference, Mapping) or set(reference) != {"index", "entities"}:
            raise ValueError("each reference requires exactly index and verified entities")
        number, entities = reference["index"], reference["entities"]
        if type(number) is not int or number < 1 or number in indices:
            raise ValueError("reference indices must be distinct positive prefix-slot numbers")
        indices.add(number)
        if (isinstance(entities, (str, bytes)) or not isinstance(entities, (list, tuple))
                or any(not isinstance(name, str) for name in entities)
                or len(entities) != len(set(entities)) or not set(entities) <= known & relevant):
            raise ValueError("reference entities must be distinct registered IDs relevant to this shot")


def shot_text_with_entity_spans(annotation: Mapping[str, Any], index: int, *,
                                references: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """唯一目标文字编译器；实体范围在排版时产生，范围是Python字符下标[start,end)。

    同外貌实体也分别携带原ID。opening_conflicts逐项报告源present/默认衣着与明确before的差异；
    只在副本中以before构造首次可见/变化前状态，不声称所有present都在RGB0出现，原annotation不变。
    token化和512-token验收由真实umT5消费者完成。
    """
    shot, field_sources, conflicts = _opening_shot(annotation, index)
    action_references = resolve_action_references(annotation, index)
    shot["action"] = action_references["action"]
    source = f"shots[{index}]"
    row = _Row(annotation)
    layout = _shot_layout(annotation, index)
    _require_target_references(row, shot, references)
    registry = _identity_registry(annotation, shot,
                                  (name for reference in references for name in reference["entities"]))
    entries, state_diagnostics = _normalized_opening_entries(annotation, shot, field_sources)
    shot["present"] = entries
    # G_i已经定义物体和地点，S_i按原ID引用；开镜图消费者仍用自己的完整自然语言描述。
    row.things = {name: name for name in row.things}
    short = {name: text for name, (text, _) in _short_identity_choices(annotation, shot, field_sources, source).items()}
    # G_i已用同一ID登记完整appearance，首次可见行也完整给出当前outfit；不再把这些同值短分句
    # 重复成括号。动作中每实体首次明确ID附短特征，后续ID复用已经建立的对应。
    first_appearance = {name: _Text("") for name in short}
    registered_outfits = {name for name, _, _, origin in registry.spans if origin.endswith(".outfit")}
    scene_shot = {**shot, "place": None} if len(annotation["places"]) == 1 else shot
    opening = _separated(_scene(row, scene_shot, _opening_without_repeated_postures(shot, field_sources), source, _positions(row, shot, layout),
                                appearances=first_appearance, compact=True, field_sources=field_sources,
                                registered_outfits=registered_outfits), " ")
    sections = dict(registry=registry, style=_Text(_picture_type_text(shot)),
                    camera=_camera(shot, layout.composition if layout else None), opening=opening,
                    action=_sentence(row.prose(shot["action"], source + ".action", identities=short)),
                    ending=_ending(row, shot, source, layout))
    parts = list(sections.values())
    compiled = _separated([part for part in parts if part.text], " ")
    return {"text": compiled.text, "entity_spans": [
        {"name": name, "start": start, "end": end, "source": source}
        for name, start, end, source in compiled.spans], "opening_conflicts": conflicts,
        "state_diagnostics": state_diagnostics, "action_references": action_references,
        "text_sections": {name: part.text for name, part in sections.items()}}


def shot_text(annotation: Mapping[str, Any], index: int, *,
              references: Sequence[Mapping[str, Any]] = ()) -> str:
    """目标镜那一行：镜头词与构图 → 首次可见的场所、人（含已知相对位置）、物 → 原动作 → 可见末态。"""
    return shot_text_with_entity_spans(annotation, index, references=references)["text"]


def opening_frame_text(annotation: Mapping[str, Any], index: int) -> str:
    """首帧只消费同一合同的开镜场所、在场外貌和起态，不提前绘制动作终点。"""
    shot = annotation["shots"][index]
    row = _Row(annotation)
    source = f"shots[{index}]"
    parts = [_Text(_picture_type_text(shot)), _camera(shot),
             *_scene(row, shot, shot["present"], source, appearances=row.characters, compact=False, field_sources={},
                     registered_outfits=frozenset())]
    # 首帧沿用已有的物体姿态与裸在场补写，不受视频行修复影响。
    for i, entry in enumerate(shot["present"]):
        name = entry["name"]
        if name in row.characters:
            continue
        if entry.get("posture"):
            parts.append(_object_posture(row, entry, f"{source}.present[{i}]",
                                         posture_source=f"{source}.present[{i}].posture"))
        elif name not in row.mentioned:
            parts.append(_sentence(_concat("Visible in the frame: ", row.thing(name, f"{source}.present[{i}].name"))))
    return _separated([part for part in parts if part.text], " ").text


def reference_text() -> str:
    """纯视觉参考只带固定角色；源镜文字不是所抽时刻的逐帧身份/服装标注。"""
    return MEMORY_ROW_ROLE_TEXT


def shot_characters(annotation: Mapping[str, Any], index: int) -> tuple[str, ...]:
    """该镜在场名单里的登记人物ID，按登记表顺序（与训练逐帧可见标签同序）；不看画面。

    推理时参考身份行的实体就是参考源镜的这一组：训练写的是「这一帧里看得到的登记人物」（离线逐帧标签），推理按分镜
    写源镜在场的登记人物，不读图。"""
    present = {entry["name"] for entry in annotation["shots"][index]["present"]}
    return tuple(entry["name"] for entry in annotation["characters"] if entry["name"] in present)


def shot_object_phrases(annotation: Mapping[str, Any], index: int) -> tuple[str, ...]:
    """该镜在场名单里的登记物体，按名单顺序取登记外貌（与目标文字首次提及同一短语）；不看画面。"""
    objects = {entry["name"]: entry["appearance"] for entry in annotation["objects"]}
    return tuple(objects[entry["name"]] for entry in annotation["shots"][index]["present"]
                 if entry["name"] in objects and objects[entry["name"]])


def reference_setting(annotation: Mapping[str, Any], index: int) -> str | None:
    """参考源镜的地点短语，与目标行登记段「Setting: …」逐字同一写法：多地点块写「Location N, 外貌」，单地点块只写外貌
    （目标行同样不写编号）；地点缺失（none/unknown）或单地点块无外貌时为 None。地点是整镜属性，不随所抽帧变。"""
    place = annotation["shots"][index].get("place")
    if place in (None, "none", "unknown"):
        return None
    registry = {entry["name"]: entry["appearance"] for entry in annotation["places"]}
    if place not in registry:
        raise ValueError(f"shot {index} place {place!r} is not a registered location")
    appearance = ARTICLE.sub("", registry[place]).strip() if registry[place] else ""
    if len(annotation["places"]) == 1:
        return appearance or None
    return f"{place}, {appearance}" if appearance else place


def reference_text_with_entity_spans(entities: Sequence[Mapping[str, Any]], *,
                                     characters: Sequence[Mapping[str, Any]],
                                     setting: str | None,
                                     shot_objects: Sequence[str] = ()) -> dict[str, Any]:
    """实际此帧确认的ID逐个附登记appearance，再写源镜地点（``setting``，见 :func:`reference_setting`）；
    不读取源镜present的姿态、动作或状态。

    ``shot_objects`` 是参考源镜在场物体的外貌短语（``shot_object_phrases``），接在人物ID之后，说的是"那一镜里有"，
    不声称此帧可见。"""
    names = [entity["name"] for entity in entities]
    if len(set(names)) != len(names) or any(
            not isinstance(name, str) or re.fullmatch(r"Character [1-9]\d*", name) is None for name in names):
        raise ValueError("reference identities must be distinct registered Character IDs")
    if isinstance(characters, (str, bytes)) or not isinstance(characters, (list, tuple)):
        raise TypeError("reference characters must be an explicit registry sequence")
    registered = {}
    for position, character in enumerate(characters):
        name, appearance = character["name"], character["appearance"]
        if (not isinstance(name, str) or re.fullmatch(r"Character [1-9]\d*", name) is None
                or name in registered or not isinstance(appearance, str) or not appearance.strip()):
            raise ValueError("reference character registry requires distinct IDs and nonempty appearance")
        registered[name] = (position, appearance)
    if not set(names) <= set(registered):
        raise ValueError("visible reference identities must exist in the character registry")
    if any(not isinstance(phrase, str) or not phrase.strip() for phrase in shot_objects):
        raise ValueError("shot object phrases must be nonempty strings")
    if setting is not None and (not isinstance(setting, str) or not setting.strip()):
        raise ValueError("reference setting must be a nonempty location phrase or None")
    objects = _concat(MEMORY_ROW_SHOT_OBJECTS, "; ".join(shot_objects), ".") if shot_objects else _Text("")
    place = _concat(MEMORY_ROW_SETTING, setting.rstrip("."), ".") if setting else _Text("")
    if not names:
        return {"text": MEMORY_ROW_ROLE_TEXT + place.text + objects.text, "entity_spans": []}
    descriptions = []
    for index, name in enumerate(names):
        position, appearance = registered[name]
        appearance = ARTICLE.sub("", appearance).strip().rstrip(".")
        descriptions.append(_concat(_Text.entity(name, name, f"reference_entities[{index}].name"), ": ",
                                    _Text.entity(name, appearance, f"characters[{position}].appearance")))
    compiled = _concat(MEMORY_ROW_ROLE_TEXT, MEMORY_ROW_VISIBLE_IDENTITIES, _separated(descriptions, "; "), ".", place,
                       objects)
    return {"text": compiled.text, "entity_spans": [
        {"name": name, "start": start, "end": end, "source": source}
        for name, start, end, source in compiled.spans]}


def shot_script_text(annotation: Mapping[str, Any]) -> str:
    """规划器逐镜输入：登记表加景别／角度／运镜与动作句，不泄露待补的状态、变化、帧数。"""
    lines = ["SUBJECTS"]
    for entry in annotation["characters"]:
        outfit = _outfit(entry.get("outfit"), "wearing")
        description = ", ".join(part for part in (entry["appearance"], outfit) if part)
        lines.append(f"- {entry['name']} ({entry['kind']}): {description}")
    for entry in annotation["objects"]:
        lines.append(f"- {entry['name']} (object): {entry['appearance']}")
    lines.append("LOCATIONS")
    lines.extend(f"- {entry['name']}: {entry['appearance']}" for entry in annotation["places"])
    lines.append("SHOTS")
    for shot in annotation["shots"]:
        sentence = _separated([part for part in (_camera(shot), _sentence(shot["action"])) if part.text], " ").text
        lines.append(f"s{shot['shot']:02d}: {sentence}")
    return "\n".join(lines)


def assert_shot_text_token_limit(annotation: Mapping[str, Any], counter: Callable[[str], int], *,
                                 limit: int = 512, shots: Sequence[int] | None = None) -> int:
    """抽中的镜超过 umT5 上限就报错，不截断。返回这些镜里的最大 token 数。"""
    if type(limit) is not int or limit < 1:
        raise ValueError("token limit must be a positive integer")
    indices = range(len(annotation["shots"])) if shots is None else shots
    peak = 0
    for index in indices:
        if type(index) is not int or not 0 <= index < len(annotation["shots"]):
            raise ValueError(f"shot index {index} is outside this block")
        count = counter(shot_text(annotation, index))
        peak = max(peak, count)
        if count > limit:
            raise ValueError(f"shot {index + 1} uses {count} umT5 tokens, above {limit}")
    return peak


def load_umt5_token_counter(tokenizer_path: Path) -> Callable[[str], int]:
    """Count prompt tokens with the renderer's own tokenizer class, without truncation.

    Needs ``third_party/wan22_runtime`` on ``PYTHONPATH``; the module must resolve
    inside that vendored tree, not a pip-installed ``diffsynth``.
    """

    import diffsynth.models.wan_video_text_encoder as text_encoder

    module_path = Path(text_encoder.__file__).resolve()
    if module_path.parents[2].parts[-2:] != ("third_party", "wan22_runtime"):
        raise RuntimeError(
            f"diffsynth must resolve inside <checkout>/third_party/wan22_runtime, got {module_path}"
        )
    source = tokenizer_path.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"umT5 tokenizer directory is missing: {source}")
    tokenizer = text_encoder.HuggingfaceTokenizer(name=str(source), seq_len=None, clean="whitespace")

    def count_tokens(text: str) -> int:
        _, mask = tokenizer(text, return_mask=True, add_special_tokens=True)
        return int(mask.sum())

    return count_tokens
