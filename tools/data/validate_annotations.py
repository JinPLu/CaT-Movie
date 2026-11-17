"""块标注批量答复的验收：解析、修补、核结构与取值，逐镜筛选并核花费。

读 ``block-annotations/*.output.jsonl`` 与 ``final/blocks.jsonl``，写块标注台账与汇总。
block_record 改回写法、补与改（mend_block）、判登记表与逐镜两层；登记表合法的块再逐项修镜。
结构合法不等于画面判得对，那要看图。旧窗验收入口已退役，历史运行使用当时的提交复现。

    python tools/data/validate_annotations.py --unit block --config configs/data/source_pipeline_v1.json \
        --contract configs/data/block_annotation_v18.json --input-cny-per-mtok 0.1 --output-cny-per-mtok 0.4
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from mspc.data.annotation_batches import block_batch_ledgers
from mspc.data.storyboard import (
    align_board_openings, complete_holding_presence, entity_references, holding_targets,
    unregistered_entity_references, visible_result_wording_errors,
)

NAMES = {"characters": re.compile(r"Character [1-9]\d*"), "objects": re.compile(r"Object [1-9]\d*"), "places": re.compile(r"Location [1-9]\d*")}


def _held_names(value, known, holder):
    """拿着的东西：登记过的物件，或登记过的另一个人物（抱着婴儿、牵着手）。"""
    return isinstance(value, list) and all(isinstance(item, str) and item != holder and known.get(item) in ("objects", "characters") for item in value)


INVENTED = re.compile(r"[A-Z][A-Za-z]*(?: [A-Za-z]+)* [1-9]\d*")  # Group 1、Animal 2、Crowd Person 3


def _renamed(node, mapping, pattern):
    if isinstance(node, str):
        return mapping.get(node) or pattern.sub(lambda match: mapping[match.group()], node)
    if isinstance(node, list):
        return [_renamed(item, mapping, pattern) for item in node]
    if isinstance(node, dict):
        return {key: _renamed(value, mapping, pattern) for key, value in node.items()}
    return node


def _rename_characters(answer):
    """人物表里不叫 ``Character N`` 的条目（人群、动物常被起成 Group 1、Animal 2）改成接着往下编的 ``Character N``，全文同步替换。

    只在不会认错时改：自创的名字带编号、不与任何登记名相同、不是物件或场所的名字格式。返回改名对照，不能改时返回空。
    """
    entries = [entry for kind in NAMES for entry in (answer.get(kind) if isinstance(answer.get(kind), list) else [])]
    names = [entry.get("name") if isinstance(entry, dict) else None for entry in entries]
    characters = answer.get("characters")
    if not isinstance(characters, list) or not all(isinstance(name, str) for name in names) or len(set(names)) != len(names):
        return {}
    odd = [entry["name"] for entry in characters if not NAMES["characters"].fullmatch(entry["name"])]
    if not odd or any(not INVENTED.fullmatch(name) or NAMES["objects"].fullmatch(name) or NAMES["places"].fullmatch(name) for name in odd):
        return {}
    used = {int(name.split()[1]) for name in names if NAMES["characters"].fullmatch(name)}
    mapping = {}
    for name in odd:
        used.add(max(used, default=0) + 1)
        mapping[name] = f"Character {max(used)}"
    pattern = re.compile("(?<![A-Za-z0-9])(?:" + "|".join(re.escape(name) for name in sorted(mapping, key=len, reverse=True)) + ")(?![A-Za-z0-9])")
    for key in list(answer):
        answer[key] = _renamed(answer[key], mapping, pattern)
    return mapping


def _renumber_registries(answer):
    """登记表跳号时同步改全文；悬空引用或重名交给原校验，不让改号把它们认成已有实体。"""
    mapping, names = {}, []
    for kind, pattern in NAMES.items():
        entries = answer.get(kind)
        if not isinstance(entries, list) or any(not isinstance(entry, dict)
                or not isinstance(entry.get("name"), str) or not pattern.fullmatch(entry["name"]) for entry in entries):
            return {}
        for number, entry in enumerate(entries, 1):
            name = entry["name"]
            names.append(name)
            mapping[name] = f"{name.split()[0]} {number}"
    if len(set(names)) != len(names) or unregistered_entity_references(answer, names):
        return {}
    mapping = {old: new for old, new in mapping.items() if old != new}
    if mapping:
        pattern = re.compile(r"\b(?:" + "|".join(re.escape(name) for name in mapping) + r")\b")
        for key in answer:
            answer[key] = _renamed(answer[key], mapping, pattern)
    return mapping


def _normalise_present_empty_values(shot, people):
    """空可选文字与未知持物省略；人物 holding: [] 明确表示空手，必须保留。"""
    repairs = []
    for entry in shot.get("present", []) if isinstance(shot.get("present"), list) else []:
        if not isinstance(entry, dict):
            continue
        for field in ("outfit", "posture", "state"):
            if field in entry and (entry[field] is None or entry[field] == [] or entry[field] == ""):
                del entry[field]
                repairs.append("空衣物值改为缺省" if field == "outfit" else "空姿态或状态改为缺省")
        if "holding" in entry and (entry["holding"] is None or
                (entry["holding"] == [] and entry.get("name") not in people)):
            del entry["holding"]
            repairs.append("空持物值改为缺省")
    return list(dict.fromkeys(repairs))


def normalise(answer, version=15):
    """不丢信息的写法偏差由程序改回合同的形状，返回改了什么；其余偏差一律判不合法、重发。

    真实返回里见到的（qwen3.7-flash 开思考，121 窗预演与 24 窗批量探针）：三张登记表和 shots 被写进了 ``video`` 里面；
    在场名单里多写了场所（场所已由 ``place`` 给出）；穿着、姿态、状态变化之后的值写成了只有一个字符串的列表；
    「被谁拿着」写在了物件那一条上（首个批量任务 5,000 窗里 31 处，如 cinedance-008887-w0008 第 3 镜）；
    人群与动物没叫 ``Character N``（同一任务里 Group N 41 处、Animal N 17 处）。
    """
    return list(dict.fromkeys(label for label, _ in normalise_by_shot(answer, version)))


def normalise_by_shot(answer, version):
    """``normalise`` 的本体：同样改 ``answer``，返回 [(改了什么, 第几镜)]；改在登记表或整份答复上的，镜号是 None。

    块的在场名单里写了场所不在这里改：那一条删掉的可能不是 ``place`` 那个场所，算删，由 ``repair_shot_entries`` 记账。
    """
    repairs = []

    def note(label, shot=None):
        if (label, shot) not in repairs:
            repairs.append((label, shot))

    if not isinstance(answer, dict):
        return repairs
    video = answer.get("video")
    if isinstance(video, dict):
        nested = [key for key in ("characters", "objects", "places", "shots") if key not in answer and key in video]
        for key in nested:
            answer[key] = video.pop(key)
        if nested:
            note("登记表或镜头写在 video 里")
    if _rename_characters(answer):
        note("人物名不是 Character N")
    if version == 17 and _renumber_registries(answer):
        note("登记编号连续化并同步所有引用")
    for entry in (answer.get("characters") if isinstance(answer.get("characters"), list) else []):
        # 第 16 版试标 50 块里 1 块：猫、狗两条写了 outfit: null（film-000690-b0000）。没有穿着就是缺省，删键不丢信息。
        if isinstance(entry, dict) and "outfit" in entry and (entry["outfit"] is None or (version == 17 and entry["outfit"] in ("", []))):
            del entry["outfit"]
            note("空默认穿着改为缺省")
    places, things, people = ({entry.get("name") for entry in (answer.get(kind) if isinstance(answer.get(kind), list) else []) if isinstance(entry, dict)} for kind in ("places", "objects", "characters"))
    for number, shot in enumerate(answer.get("shots") if isinstance(answer.get("shots"), list) else [], 1):
        present = shot.get("present") if isinstance(shot, dict) else None
        if isinstance(present, list) and version != 17:
            kept = [entry for entry in present if not (isinstance(entry, dict) and entry.get("name") in places)]
            if len(kept) != len(present):
                shot["present"] = kept
                note("在场名单里写了场所", number)
        if isinstance(shot, dict) and version == 17:
            for label in _normalise_present_empty_values(shot, people):
                note(label, number)
        if isinstance(present, list) and version != 17:
            for entry in shot["present"]:
                field = "clothing"
                if isinstance(entry, dict) and field in entry and not entry[field] and not isinstance(entry[field], str):
                    # 旧合法语料有84处 null/[]；旧排版把空值当缺省，删键不丢穿着信息。
                    del entry[field]
                    note("空衣物值改为缺省", number)
        if isinstance(present, list):  # 物件上写了「被谁拿着」：挪到那个人物的 holding 里，同一件事的合同写法
            by_name = {entry.get("name"): entry for entry in shot["present"] if isinstance(entry, dict)}
            for entry in shot["present"]:
                holders = entry.get("holding") if isinstance(entry, dict) else None
                if isinstance(entry, dict) and entry.get("name") in things and isinstance(holders, list) and holders and all(
                        isinstance(who, str) and who in people and who in by_name for who in holders):
                    for who in holders:
                        mine = by_name[who].setdefault("holding", [])
                        if isinstance(mine, list) and entry["name"] not in mine:
                            mine.append(entry["name"])
                    del entry["holding"]
                    note("物件上写了被谁拿着", number)
        for change in (shot.get("changes") if isinstance(shot, dict) else None) or []:
            after = change.get("after") if isinstance(change, dict) else None
            if isinstance(change, dict) and change.get("what") in (("outfit" if version == 17 else "clothing"), "posture", "state") and isinstance(after, list) and len(after) == 1 and isinstance(after[0], str):
                change["after"] = after[0]
                note("变化后的值写成了单元素列表", number)
    return repairs


def _blank(value):
    """没有可写的东西却写了个「没有」：None、空串、字符串 none（片尾卡那一镜的 place 与 camera，film-004095-b0008 各遍都有）。"""
    return value is None or (isinstance(value, str) and value.strip().casefold() in ("", "none"))


# 手、手臂这类身体部位不是物件：film-006561-b0006 第 9 镜 "the woman's hand"，film-004600-b0002 第 3 镜 "Character 3's hand"。
BODY_PART = re.compile(r"(?:.*['’]s? )?(?:(?:the|a|an|his|her|their|its|own|both|one|two|left|right|other) )*"
                       r"(?:hands?|arms?|fists?|fingers?|thumbs?|palms?|wrists?|elbows?|shoulders?|necks?|heads?|faces?|hair|waists?|hips?|legs?|knees?|ankles?|foot|feet|paws?|tails?)")


def _unstated_outfits(answer):
    """换装之后、回到登记表穿着之前，该人物在场却没写 ``outfit`` 的条目：逐个给出 (镜号, 在场名单里的位置, 条目, 此时的穿着)。

    换装指 ``changes`` 里 ``outfit`` 的 ``after`` 是字符串；此时的穿着取最近一次写明的——那次的 ``after``，或其后某镜在场条目
    自己写的 ``outfit``；写明的穿着与登记表相同（不分大小写与首尾空格）就算换回来了。同一镜里在场条目写的是开头，变化在后。
    """
    defaults = {e["name"]: e.get("outfit") for e in answer.get("characters", []) if isinstance(e, dict) and isinstance(e.get("name"), str)}
    wearing = {}

    def stated(name, outfit):
        if isinstance(defaults[name], str) and outfit.strip().casefold() == defaults[name].strip().casefold():
            wearing.pop(name, None)
        else:
            wearing[name] = outfit

    for shot in answer["shots"]:
        for index, entry in enumerate(shot["present"] if isinstance(shot.get("present"), list) else []):
            name = entry.get("name") if isinstance(entry, dict) else None
            if isinstance(name, str) and name in wearing:
                if "outfit" not in entry:
                    yield shot["shot"], index, entry, wearing[name]
                elif isinstance(entry["outfit"], str):
                    stated(name, entry["outfit"])
        for change in shot["changes"] if isinstance(shot.get("changes"), list) else []:
            if isinstance(change, dict) and change.get("what") == "outfit" and isinstance(change.get("name"), str) \
                    and change["name"] in defaults and isinstance(change.get("after"), str):
                stated(change["name"], change["after"])


def mend_block(answer, labels):
    """块独有的补与改，逐项记账；登记表层合法之后调用。信息能保住的就不要删。

    摇镜的左右并成合同词表里的 ``pan``（提示词写了不分方向，答复偶尔照旧）；人物的持物写了没登记的东西，补登记成新物体、
    同一块里同样的字符串用同一个（film-018940-b0001 两镜的 binoculars），身体部位、「没有」和没登记的编号不补，留给
    ``repair_shot_entries`` 删；换装之后漏写穿着的镜，顺着最近一次写明的穿着向后补（第 18 版快速检查 34 条里 2 条、5 镜）。
    开镜对齐由 ``block_record`` 在删除不合法的条目之后调用，避免删除持物后重新出现空变化。
    """
    fixes = []
    known = {entry["name"]: kind for kind in NAMES for entry in answer[kind]}
    # 原答复里悬空的编号不能被新登记条目占用，否则旧错误会悄悄绑定到另一个实体。
    reserved = set(entity_references(answer))
    added = {}
    for shot in answer["shots"]:
        camera = shot.get("camera")
        if isinstance(camera, dict) and camera.get("movement") in ("pan left", "pan right") and "pan" in labels["movement"]:
            fixes.append({"rule": "摇镜不分方向", "shot": shot["shot"], "path": "camera.movement", "value": camera["movement"], "to": "pan"})
            camera["movement"] = "pan"
        present = shot.get("present")
        if not isinstance(present, list):
            continue
        held = [(f"present[{index}].holding", entry["holding"]) for index, entry in enumerate(present)
                if isinstance(entry, dict) and isinstance(entry.get("name"), str) and known.get(entry["name"]) == "characters" and isinstance(entry.get("holding"), list)]
        for index, change in enumerate(shot["changes"] if isinstance(shot.get("changes"), list) else []):
            if isinstance(change, dict) and change.get("what") == "holding" and isinstance(change.get("name"), str) and known.get(change["name"]) == "characters":
                held += [(f"changes[{index}].{side}", change[side]) for side in ("before", "after") if isinstance(change.get(side), list)]
        for path, items in held:
            for position, item in enumerate(items):
                if not isinstance(item, str) or item in known or _blank(item) or BODY_PART.fullmatch(item.strip().casefold()) \
                        or any(pattern.fullmatch(item) for pattern in NAMES.values()):
                    continue
                name = added.get(item.strip().casefold())
                if name is None:
                    number = max((int(n.split()[1]) for n in known if known[n] == 'objects'), default=0) + 1
                    conflicted = []
                    while f"Object {number}" in reserved:
                        conflicted.append(f"Object {number}")
                        number += 1
                    if conflicted:
                        fixes.append({'rule': '新登记编号与悬空引用冲突', 'shot': shot['shot'], 'names': conflicted})
                    name = added[item.strip().casefold()] = f"Object {number}"
                    reserved.add(name)
                    answer["objects"].append({"name": name, "appearance": item.strip()})
                    known[name] = "objects"
                items[position] = name
                if not any(isinstance(entry, dict) and entry.get("name") == name for entry in present):
                    present.append({"name": name})
                fixes.append({"rule": "持物补登记", "shot": shot["shot"], "path": f"{path}[{position}]", "value": item, "to": name})
        for name in complete_holding_presence(shot, known):
            fixes.append({'rule': '持有实体补在场名单', 'shot': shot['shot'], 'to': name})
    for number, index, entry, outfit in _unstated_outfits(answer):
        entry["outfit"] = outfit
        fixes.append({"rule": "换装后补穿着", "shot": number, "path": f"present[{index}].outfit", "to": outfit})
    return sorted(fixes, key=lambda fix: fix["shot"])


def structure_errors(answer, labels, shot_count, version):
    """返回 (整块错误, {镜号: 错误列表})；两层都为空才整块合法。

    v15 保留原有判定；v17 读 story、逐镜筛选字段与 outfit。
    调用者先做 normalise；镜数或镜号不符属于整块错误。块的在场名单里出现没登记的 ``Character N``／``Object N``
    也是整块错误：登记表写在逐镜之前，写到后面才冒出来的人补不回去，整块重发（film-007241-b0000 最后一镜，三遍都有）。
    """
    if version not in (15, 17):
        raise ValueError(f"unsupported annotation version: {version}")
    errors = []
    if not isinstance(answer, dict):
        return ["answer is not an object"], {}
    video = answer.get("video")
    if version == 15 and (not isinstance(video, dict) or video.get("picture_type") not in labels["picture_type"] \
            or video.get("on_screen_text") not in labels["on_screen_text"] or not isinstance(video.get("summary"), str)):
        errors.append("video block")
    if version == 17:
        story = answer.get("story")
        if not isinstance(story, dict) or any(not isinstance(story.get(k), str) for k in ("paragraph", "summary")):
            errors.append("story block")
        if "video" in answer:
            errors.append("unexpected video block")
    known = {}
    for kind, pattern in NAMES.items():
        entries = answer.get(kind)
        if not isinstance(entries, list):
            errors.append(f"{kind} missing")
            continue
        for entry in entries:
            name = entry.get("name") if isinstance(entry, dict) else None
            if not isinstance(name, str) or not pattern.fullmatch(name) or name in known or not isinstance(entry.get("appearance"), str):
                errors.append(f"{kind} entry {name!r}")
                continue
            known[name] = kind
            if kind == "characters" and (entry.get("kind") not in labels["kind"] or entry.get("visibility") not in labels["visibility"]):
                errors.append(f"character labels {name}")
            if version == 17 and kind == "characters" and "outfit" in entry and not isinstance(entry["outfit"], str):
                errors.append(f"character outfit {name}")
    shots = answer.get("shots")
    if not isinstance(shots, list) or [s.get("shot") if isinstance(s, dict) else None for s in shots] != list(range(1, shot_count + 1)):
        return errors + [f"shots: expected 1..{shot_count}"], {}
    by_shot = {}
    for shot in shots:
        local = shot_errors(shot, known, labels, version)
        if local:
            by_shot[shot["shot"]] = local
    if version == 17:
        for number, _, entry, _ in _unstated_outfits(answer):
            by_shot.setdefault(number, []).append(f"shot {number} missing outfit after change of {entry['name']}")
        for shot in shots:
            for entry in shot["present"] if isinstance(shot.get("present"), list) else []:
                name = entry.get("name") if isinstance(entry, dict) else None
                if isinstance(name, str) and name not in known and (NAMES["characters"].fullmatch(name) or NAMES["objects"].fullmatch(name)):
                    errors.append(f"shot {shot['shot']} present {name!r} not registered")
    return errors, by_shot


def shot_errors(shot, known, labels, version):
    """一镜的校验语义，整块验收与删除修复共用。"""
    errors = []
    number = shot["shot"]
    if version == 17:
        errors.extend(f"shot {number} unregistered entity reference: {name}"
                      for name in unregistered_entity_references(shot, known))
    camera = shot.get("camera", {}) if version == 17 else (shot.get("camera") or {})
    # 第 18 版合同：不是连续画面的镜（片尾卡、一页文字）省掉 place 与 camera。
    placeless = version == 17 and shot.get("picture_type") == "screen_or_slides" and "place" not in shot
    if not (placeless or known.get(shot.get("place")) == "places") or shot.get("activity") not in labels["activity"] or not isinstance(shot.get("action"), str) \
            or not isinstance(camera, dict) or any(not isinstance(value, str) for value in camera.values()):
        errors.append(f"shot {number} place/camera/activity/action")
    if version == 17:
        for field in ("picture_type", "on_screen_text"):
            if shot.get(field) not in labels[field]:
                errors.append(f"shot {number} {field}")
        if isinstance(camera, dict):
            for field in ("shot_size", "angle", "movement"):
                if field in camera and camera[field] not in labels[field]:
                    errors.append(f"shot {number} camera.{field}")
    present = shot.get("present")
    if not isinstance(present, list) or not isinstance(shot.get("changes"), list):
        errors.append(f"shot {number} present/changes")
        return errors
    listed = set()
    for entry in present:
        name = entry.get("name") if isinstance(entry, dict) else None
        if known.get(name) not in ("characters", "objects") or name in listed:
            errors.append(f"shot {number} present {name!r}")
            continue
        listed.add(name)
        outfit_field = "outfit" if version == 17 else "clothing"
        if any(not isinstance(entry.get(key) or "", str) for key in (outfit_field, "posture", "state")) or (
                outfit_field in entry and not isinstance(entry[outfit_field], str)):
            errors.append(f"shot {number} text fields of {name}")
        if version == 17 and "clothing" in entry:
            errors.append(f"shot {number} unexpected clothing of {name}")
        if entry.get("holding") and (known[name] != "characters" or not _held_names(entry["holding"], known, name)):
            errors.append(f"shot {number} holding of {name}")
    for change in shot["changes"]:
        if not isinstance(change, dict) or change.get("name") not in listed or change.get("what") not in labels["change_what"]:
            errors.append(f"shot {number} change")
        elif change["what"] == "holding":
            if not (_held_names(change.get("before"), known, change["name"]) and _held_names(change.get("after"), known, change["name"])):
                errors.append(f"shot {number} holding change")
        elif not isinstance(change.get("after"), str) or (version == 17 and
                change.get("before") is not None and not isinstance(change["before"], str)):
            errors.append(f"shot {number} change value")
    if version == 17:
        missing = [name for name in holding_targets(shot) if known.get(name) in ('characters', 'objects') and name not in listed]
        if missing:
            errors.append(f"shot {number} holding targets not present: {missing}")
    return errors


def repair_shot_entries(shot, known, labels, version):
    """只删除本镜不合规的单个持物、变化或在场条目，逐项记账；不填值。

    不是连续画面的镜，place 与 camera 写成「没有」的删掉；握持变化里不合规的只删那一项，变化本身留着。
    """
    from copy import deepcopy

    removed = []
    if version == 17 and shot.get("picture_type") == "screen_or_slides":
        for field in ("place", "camera"):
            if field in shot and _blank(shot[field]):
                removed.append({"path": field, "value": shot.pop(field)})
    if version == 17 and isinstance(shot.get("camera"), dict):
        for field in ("shot_size", "angle", "movement"):
            if field in shot["camera"] and shot["camera"][field] not in labels[field]:
                removed.append({"path": f"camera.{field}", "value": shot["camera"].pop(field)})
    present = shot.get("present")
    if isinstance(present, list):
        seen, kept = set(), []
        for index, entry in enumerate(present):
            name = entry.get("name") if isinstance(entry, dict) else None
            if not isinstance(name, str) or known.get(name) not in ("characters", "objects") or name in seen:
                removed.append({"path": f"present[{index}]", "value": deepcopy(entry)})
                continue
            seen.add(name)
            kept.append(entry)
            if version == 17:
                for field in ("outfit", "posture", "state", "clothing"):
                    if field in entry and (field == "clothing" or not isinstance(entry[field], str)):
                        removed.append({"path": f"present[{index}].{field}", "value": entry.pop(field)})
            holding = entry.get("holding")
            if holding and not isinstance(holding, list):
                removed.append({"path": f"present[{index}].holding", "value": entry.pop("holding")})
            # 机械爪/机器人也可能被登记为 Object；删空其非空持物会抹掉真实关系并误放行。
            # 持有者类型不合规时保留内容，由校验拒收，等待语义复核。
            if isinstance(holding, list) and known[name] == "characters":
                valid = []
                for j, item in enumerate(holding):
                    if not _held_names([item], known, name):
                        removed.append({"path": f"present[{index}].holding[{j}]", "value": deepcopy(item)})
                    else:
                        valid.append(item)
                entry["holding"] = valid
        shot["present"] = kept
    if isinstance(shot.get("changes"), list) and isinstance(shot.get("present"), list):
        kept = []
        for index, change in enumerate(shot["changes"]):
            if isinstance(change, dict) and change.get("what") == "holding":
                for side in ("before", "after"):
                    if isinstance(change.get(side), list):
                        wrong = [position for position, item in enumerate(change[side]) if not _held_names([item], known, change.get("name"))]
                        removed += [{"path": f"changes[{index}].{side}[{position}]", "value": deepcopy(change[side][position])} for position in wrong]
                        change[side] = [item for position, item in enumerate(change[side]) if position not in wrong]
            probe = {**shot, "changes": [change]}
            if any(e in (f"shot {shot['shot']} change", f"shot {shot['shot']} holding change", f"shot {shot['shot']} change value")
                   for e in shot_errors(probe, known, labels, version)):
                removed.append({"path": f"changes[{index}]", "value": deepcopy(change)})
            else:
                kept.append(change)
        shot["changes"] = kept
    return removed


def classify(answer):
    """合同 ``derived`` 与 ``rules`` 的程序实现：留筛、类别、标记，1 到 5 镜同一套。"""
    video, shots = answer["video"], answer["shots"]
    people = [c for c in answer["characters"] if c["kind"] == "person"]
    clear = [c for c in people if c["visibility"] == "clear"]
    living = [c for c in answer["characters"] if c["kind"] in ("person", "animal")]
    appearances = Counter(entry["name"] for shot in shots for entry in shot["present"])
    recurring = sum(appearances[c["name"]] >= 2 for c in people)
    any_action = any(shot["activity"] == "action_or_state_change" for shot in shots)
    all_talking = all(shot["activity"] == "talking_only" for shot in shots)
    filters = []
    if video["picture_type"] == "screen_or_slides":
        filters.append("不是连续拍摄的画面")
    if video["on_screen_text"] == "burned_subtitles":
        filters.append("烧录字幕")
    if video["on_screen_text"] == "large_text_or_picture_in_picture":
        filters.append("大面积文字或画中画")
    if not living or all(c["visibility"] == "distant" for c in living):
        filters.append("没有可认的主体")
    if video["picture_type"] in ("animation_2d", "animation_3d_or_game"):
        name = "动画或游戏"
    elif video["picture_type"] == "live_action" and clear and any_action:
        name = "核心"
    elif not clear and any(c["visibility"] == "partial" for c in people):
        name = "只见手或局部"
    elif all_talking and len(people) >= 2:
        name = "对白"
    elif all_talking and len(people) == 1:
        name = "单人讲话"
    else:
        name = "其他留"
    marks = []
    if video["on_screen_text"] == "corner_logo_or_watermark":
        marks.append("有角标或水印")
    if len(shots) >= 2 and recurring == 0:
        marks.append("无跨镜同人")
    return {"hard_filter": filters, "class": name, "marks": marks, "people": len(people), "clear_people": len(clear), "recurring_people": recurring}


def classify_shots(answer, contract):
    """逐镜执行合同的简单相等规则，保留筛因与角标。"""
    def matching(shot, rules):
        result = []
        for rule in rules:
            field, value = rule["when"].split(" == ")
            if shot.get(field) == value:
                result.append(rule["name"])
        return result
    result = []
    for shot in answer.get("shots", []):
        reasons = matching(shot, contract["shot_filter"])
        marks = matching(shot, contract["shot_mark"])
        result.append({"shot": shot["shot"], "status": "被筛" if reasons else "可用", "reasons": reasons, "marks": marks})
    return result


def block_record(content, contract, shot_count):
    """块不编译每镜文字；原答复、补与改（``repairs``）、逐项删除（``deletions``）分别保存。

    ``repairs`` 每项带 ``rule``，改在某一镜上的带 ``shot``，补登记、摇镜、补穿着另带位置、原值与新值；不带 ``shot`` 的改在登记表或
    整份答复上（人物改名会连带改各镜里的名字）。``deletions`` 按镜号列删掉的位置与原值。一镜在两处都没有记录，内容就是答复原样。
    """
    from copy import deepcopy
    result = {"raw_answer": content, "annotation": None, "repairs": [], "deletions": {}, "reasks": {}, "shot_filter": []}
    try:
        answer = json.loads(content)
    except (ValueError, TypeError):
        result.update(status="invalid" if content else "no_answer", block_errors=["not json" if content else "no content"], shot_errors={})
        return result
    wording_errors = visible_result_wording_errors(answer) if isinstance(answer, dict) else []
    result["repairs"] = [{"rule": label} if shot is None else {"rule": label, "shot": shot} for label, shot in normalise_by_shot(answer, 17)]
    block, local = structure_errors(answer, contract["labels"], shot_count, 17)
    if not block:  # 补登记要往登记表里加条目，登记表层先得合法；初判记的是补与改之后、动手删之前
        result["repairs"] += mend_block(answer, contract["labels"])
        block, local = structure_errors(answer, contract["labels"], shot_count, 17)
        if any(fix['rule'] == '新登记编号与悬空引用冲突' for fix in result['repairs']):
            block.append('object identity collision requires review of the raw answer')
    result["initial_block_errors"], result["initial_shot_errors"] = block, deepcopy(local)
    if isinstance(answer, dict) and isinstance(answer.get("shots"), list) and all(isinstance(s, dict) and "shot" in s for s in answer["shots"]):
        result["shot_filter"] = classify_shots(answer, contract)
    if not block:
        known = {e["name"]: kind for kind in NAMES for e in answer[kind]}
        for number in local:
            removed = repair_shot_entries(answer["shots"][number-1], known, contract["labels"], 17)
            if removed:
                result["deletions"][number] = removed
        outfits = {entry["name"]: entry.get("outfit") for entry in answer["characters"]}
        for shot in answer["shots"]:
            result["repairs"] += [{"rule": label, "shot": shot["shot"]}
                                  for label in _normalise_present_empty_values(shot, outfits)]
        result["repairs"] += align_board_openings(answer, outfits)
        block, local = structure_errors(answer, contract["labels"], shot_count, 17)
    result['wording_errors'] = wording_errors
    for error in wording_errors:
        message = error['path'] + ': ' + error['reason']
        if error['shot'] is None:
            block.append(message)
        else:
            local.setdefault(error['shot'], []).append(message)
    result.update(annotation=answer, block_errors=block, shot_errors=local, status="invalid" if block or local else "ok")
    return result


def validate_blocks(root, contract, input_price, output_price):
    catalog = {r["key"]: r for r in map(json.loads, (root/"final/blocks.jsonl").open())}
    records, usage, totals = {}, Counter(), Counter()
    cost_cny = 0.0
    for kind in ("output", "error"):
        for path in sorted((root/"block-annotations").glob(f"*.{kind}.jsonl")):
            for item in map(json.loads, path.open()):
                key = item["custom_id"]
                row = catalog[key]
                body = (item.get("response") or {}).get("body") or {}
                spent = body.get("usage") or {}
                inp, out = spent.get("prompt_tokens", 0), spent.get("completion_tokens", 0)
                usage.update(input=inp, output=out, reasoning=(spent.get("completion_tokens_details") or {}).get("reasoning_tokens", 0))
                cost_cny += (inp*input_price+out*output_price)*(1 if inp <= 32000 else 3)/1e6
                totals[kind + "_lines"] += 1
                totals["higher_price_responses"] += inp > 32000
                if records.get(key, {}).get("status") == "ok" or (kind == "error" and records.get(key, {}).get("status") == "invalid"):
                    continue
                choice = (body.get("choices") or [{}])[0]
                record = block_record((choice.get("message") or {}).get("content"), contract, row["scene"]["shot_count"])
                record.update(key=key, source=row["source"], shot_count=row["scene"]["shot_count"], usage=spent,
                              **{"from": path.name.removesuffix(f".{kind}.jsonl")})
                if record["status"] == "no_answer":
                    error = item.get("error") or body.get("error") or {}
                    record.update(error_code=str(error.get("code")), error=str(error.get("message"))[:160])
                records[key] = record
    counts = Counter(r["status"] for r in records.values())
    selected, submitted, inflight = set(), set(), set()
    for _, path in block_batch_ledgers(root / "final"):
        ledger = json.loads(path.read_text())
        selected.update(ledger["selected_keys"])
        for batch in ledger["batches"]:
            if batch.get("batch_id"):
                submitted.update(batch["keys"])
                if not batch["fetched"]:
                    inflight.update(batch["keys"])
    initial_registry = sum(not r.get("initial_block_errors", r["block_errors"]) for r in records.values())
    final_shots = sum(r["shot_count"]-len(r["shot_errors"]) for r in records.values() if not r["block_errors"])
    summary = {"blocks_total": len(catalog), "in_ledger": len(records), "selected": len(selected),
               "submitted": len(submitted), "inflight": len(inflight), "submitted_without_response": len(submitted-records.keys()),
               "selected_not_submitted": len(selected-submitted), "never_submitted": len(catalog)-len(submitted), **totals, **counts,
               "registry_valid": initial_registry, "valid_shots": final_shots, "shots": sum(r["shot_count"] for r in records.values()),
               "usage_tokens": dict(usage), "cost_cny": cost_cny,
               "repairs": dict(Counter(fix["rule"] for r in records.values() for fix in r["repairs"])),
               "deleted_items": sum(len(items) for r in records.values() for items in r["deletions"].values()),
               "initial_all_valid": sum(not r.get("initial_block_errors", r["block_errors"]) and not r.get("initial_shot_errors", r["shot_errors"]) for r in records.values())}
    temporary = root/"final/block-annotations.partial"
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records.values()))
    temporary.replace(root/"final/block-annotations.jsonl")
    (root/"final/block-annotation-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1)+"\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unit", choices=("block",), required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--input-cny-per-mtok", type=float, required=True, help="批量接口的输入单价（元每百万 token）")
    p.add_argument("--output-cny-per-mtok", type=float, required=True, help="批量接口的输出单价，思考按输出计")
    a = p.parse_args()
    root = Path(json.loads(a.config.read_text())["root"])
    contract = json.loads(a.contract.read_text())
    # 块的第 18 版只改了提示词的写法，输出结构与第 17 版相同，下面按第 17 版的结构判。
    if contract["schema_version"] not in (17, 18):
        raise ValueError("block annotation requires contract version 17 or 18")
    print(json.dumps(validate_blocks(root, contract, a.input_cny_per_mtok, a.output_cny_per_mtok), ensure_ascii=False))
