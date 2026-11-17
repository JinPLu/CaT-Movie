"""地板标注：一窗一次主调用（思考开、自己解析，先剥最外层 Markdown 围栏），解析不出对象再做一次带输出上限的修复调用；请求原文与三层验收，转换器写出 event_log_v3。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import base64
import copy
import json
import fcntl
from datetime import datetime, timezone
from decimal import Decimal
import re
from threading import Condition, Event
import os
import time
from pathlib import Path

from mspc.data.event_log import EVENT_LOG_SCHEMA, canonical_shot_field
from mspc.data.closeai_client import (
    CloseAIClient, CloseAICostCounter, CloseAIResponseTruncated, _reassemble_stream, sanitize_log_value,
)


CONTRACT_VERSION = "state_pilot_annotation_v13_20261011"

SYSTEM = """You are a video annotator. You write down what is visible in a short video made of several shots, so that the notes can be used to train a video generation model. Answer with one JSON object and nothing else.

### Input
- The video, with its shots played in order. The time range of every shot is given.

### Step 1. List who and what is in the video
- characters: people and animals. Name them "Character 1", "Character 2", ... The same one keeps the same name in every shot. If you are not sure that two appearances are the same person, list them as two characters.
  List a person only if they appear in two or more shots, speak, do something with a listed character or object, or are the main person of a shot. Do not list passers-by or crowds that are only in the background.
- objects: things that a character touches, holds, picks up, puts on, takes off or uses. Name them "Object 1", "Object 2", ... Do not list things nobody interacts with. Clothes that stay on are not objects; describe them under "clothing".
- places: name them "Location 1", "Location 2", ... Two shots show the same place if they show the same room or site, even from opposite camera positions.
- landmarks: for each place, up to eight large fixed things that would still be recognisable from another camera position (bed, table, counter, sofa, door, window, staircase, car, tree). Name them "Landmark 1", "Landmark 2", ... across the whole video. Put the ones seen in more than one shot first.

Give each of them an "appearance": what it looks like in a way that stays true for the whole video.
- Characters (at most 20 words): hair, build, skin tone, age range, beard, face shape. Do not write clothing, accessories that can be taken off, pose, expression, held things, or blood, sweat, dirt, wounds or wetness.
  If the face is never seen, write the hair, build or height you can see. If nothing can be told, write "a person whose face is not seen".
- Objects and landmarks (at most 8 words): what kind of thing, colour, material, shape. Do not write whether it is open, closed, full, empty, broken, wet, lit, folded or worn at the moment.
- Good: "a middle-aged man with short black hair and a broad build"; "a red cloth"; "a newspaper"; "a white ceramic mug".
- Bad: "a man in a dark police vest" (clothing); "a pale, sweating face" (temporary state); "an open newspaper", "a blood-soaked cloth" (temporary state).

### Step 2. Describe every shot, in order
- shot: its number, starting at 1.
- place: a location name.
- camera (optional): shot_size ("extreme close-up shot", "close-up shot", "medium close-up shot", "medium shot", "medium wide shot", "wide shot", "extreme wide shot"); angle ("eye-level shot", "high angle shot", "low angle shot", "overhead shot", "over-the-shoulder shot", "dutch angle shot", "aerial shot"); movement ("static", "pan left", "pan right", "tilt up", "tilt down", "push in", "pull out", "tracking", "handheld").
- action: one sentence of at most 40 words saying what happens in the shot. Use only the names. Do not describe appearance, clothing or the camera.
- background (optional): at most 12 words about people or animals in the shot that you did not list, e.g. "several customers seated at tables". Leave it out if there are none.
- place_condition (optional): a visible state of the place, e.g. "raining", "lights off".
- landmarks: names of the landmarks visible at any moment of the shot.
- present[]: one entry for every listed character and object that is visible at any moment of the shot.
  - name
  - in_first_frame, in_last_frame: true or false.
  - The fields below describe it at the moment it is first visible in the shot. Leave out what you cannot see.
  - clothing (characters): what they wear, e.g. "a black cardigan over a white shirt".
  - posture (characters): body position and where they are, e.g. "sitting at the dining table", "lying on the floor", "standing in the doorway".
  - condition: a temporary visible state. Character: "sweating", "a bleeding cheek", "a bandaged arm". Object: "open", "broken", "full of water".
  - holding (characters): names of the objects in their hands.
- relations[] (optional): where an object is when nobody holds it, at the moment it is first visible: object; relation ("on", "in", "under", "attached to", "next to", "against"); target (a landmark, object or character name). Do not write relations for characters; their place belongs in "posture".
- changes[] (optional): what is different between the first frame and the last frame of this shot. One entry per change: name; aspect ("clothing", "posture", "condition" or "holding"); before (as in the first frame, or when first visible); after (as in the last frame); seen_happening (true if the change is seen taking place during the shot; false if it takes place out of view or hidden, and only the result is seen at the end). For "holding", before and after are lists of object names, [] for empty hands.

### Rules
- Write only what you can see. If you cannot tell something, leave that field out. Never write "unknown" and never guess.
- Do not write where things are in the frame (left, right, foreground, background) or which way people face; that is measured separately.
- Do not use captions, titles or knowledge of who the actors are.
- Write every value in short, plain English.
- Use only the names you listed.

### Example (two shots)
{"characters": [{"name": "Character 1", "kind": "person", "appearance": "an elderly woman with short grey hair and a slim build"}, {"name": "Character 2", "kind": "person", "appearance": "a tall person with long brown hair whose face is not seen"}], "objects": [{"name": "Object 1", "appearance": "a white ceramic mug"}, {"name": "Object 2", "appearance": "a brown leather suitcase"}], "places": [{"name": "Location 1", "appearance": "a small kitchen with wooden cabinets", "landmarks": [{"name": "Landmark 1", "appearance": "a wooden dining table"}, {"name": "Landmark 2", "appearance": "a glass back door"}]}], "shots": [{"shot": 1, "place": "Location 1", "camera": {"shot_size": "medium shot", "angle": "eye-level shot", "movement": "static"}, "action": "Character 1 lifts Object 1 from the table and drinks while Character 2 comes in through the back door carrying Object 2.", "landmarks": ["Landmark 1", "Landmark 2"], "present": [{"name": "Character 1", "in_first_frame": true, "in_last_frame": true, "clothing": "a navy sweater", "posture": "sitting at the dining table"}, {"name": "Object 1", "in_first_frame": true, "in_last_frame": true, "condition": "full of tea"}, {"name": "Character 2", "in_first_frame": false, "in_last_frame": true, "clothing": "a long grey coat", "posture": "walking in through the back door", "condition": "wet hair", "holding": ["Object 2"]}, {"name": "Object 2", "in_first_frame": false, "in_last_frame": true}], "relations": [{"object": "Object 1", "relation": "on", "target": "Landmark 1"}], "changes": [{"name": "Character 1", "aspect": "holding", "before": [], "after": ["Object 1"], "seen_happening": true}]}, {"shot": 2, "place": "Location 1", "camera": {"shot_size": "close-up shot", "angle": "eye-level shot", "movement": "static"}, "action": "Character 1 looks up and speaks to Character 2, who is out of view.", "background": "a cat sleeping on a chair", "landmarks": ["Landmark 1"], "present": [{"name": "Character 1", "in_first_frame": true, "in_last_frame": true, "clothing": "a navy sweater", "posture": "sitting at the dining table"}, {"name": "Object 1", "in_first_frame": true, "in_last_frame": true, "condition": "empty"}], "relations": [{"object": "Object 1", "relation": "on", "target": "Landmark 1"}], "changes": []}]}
"""

CHARACTER_ID = re.compile(r"Character [1-9]\d*")
OBJECT_ID = re.compile(r"Object [1-9]\d*")
LOCATION_ID = re.compile(r"Location [1-9]\d*")
LANDMARK_ID = re.compile(r"Landmark [1-9]\d*")
KINDS = {"person", "animal", "object"}
SHOT_SIZES = {"extreme close-up shot", "close-up shot", "medium close-up shot", "medium shot", "medium wide shot", "wide shot", "extreme wide shot"}
ANGLES = {"eye-level shot", "high angle shot", "low angle shot", "overhead shot", "over-the-shoulder shot", "dutch angle shot", "aerial shot"}
MOVEMENTS = {"static", "pan left", "pan right", "tilt up", "tilt down", "push in", "pull out", "tracking", "handheld"}
RELATIONS = {"on", "in", "under", "attached to", "next to", "against"}
ASPECTS = {"clothing", "posture", "condition", "holding"}
STATE_AXIS = {"clothing": "costume", "posture": "body", "condition": "object_condition"}


_TEXT = {"type": "string"}
_IDS = {"type": "array", "items": _TEXT}


def _nullable(schema: dict) -> dict:
    if "anyOf" in schema:
        return {"anyOf": [*schema["anyOf"], {"type": "null"}]}
    return {**schema, "type": [schema["type"], "null"]}


def _object(properties: dict, *, optional: tuple[str, ...] = ()) -> dict:
    """strict 对象：全部属性列入 required；SYSTEM 里可省略的字段改为可为 null。"""
    return {"type": "object", "additionalProperties": False, "required": list(properties),
            "properties": {key: _nullable(value) if key in optional else value for key, value in properties.items()}}


#: 修复调用的 json_schema：字段照 SYSTEM「### Fields」。出处：阿里云百炼「结构化输出」页对 strict json_schema 的
#: 要求——所有属性列入 required、additionalProperties 为 false，可省略的字段用可为 null 的类型表达。
#: 修复结果里的 null 即「没写这个字段」，进 accept_answer 前去掉（``_drop_nulls``）。取值不加枚举，原值照抄，
#: 合不合格仍由 accept_answer 逐字段判（与主调用直接得到对象时同一套账）。
#: 10-09 实测 CloseAI 网关收下 strict 但不执行（qwen3.5-flash 回了 {"error", "response": 散文摘要}），所以形状靠
#: ``REPAIR_SYSTEM`` 的明确指令给出，schema 仍随请求发作兜底。
ANSWER_SCHEMA = _object({
    "characters": {"type": "array", "items": _object({"name": _TEXT, "kind": _TEXT, "appearance": _TEXT}, optional=("appearance",))},
    "objects": {"type": "array", "items": _object({"name": _TEXT, "appearance": _TEXT}, optional=("appearance",))},
    "places": {"type": "array", "items": _object({
        "name": _TEXT, "appearance": _TEXT,
        "landmarks": {"type": "array", "items": _object({"name": _TEXT, "appearance": _TEXT}, optional=("appearance",))},
    }, optional=("appearance", "landmarks"))},
    "shots": {"type": "array", "items": _object({
        "shot": {"type": "integer"}, "place": _TEXT, "action": _TEXT,
        "camera": _object({"shot_size": _TEXT, "angle": _TEXT, "movement": _TEXT}, optional=("shot_size", "angle", "movement")),
        "landmarks": _IDS, "background": _TEXT, "place_condition": _TEXT,
        "present": {"type": "array", "items": _object({
            "name": _TEXT, "in_first_frame": {"type": "boolean"}, "in_last_frame": {"type": "boolean"},
            "clothing": _TEXT, "posture": _TEXT, "condition": _TEXT, "holding": _IDS,
        }, optional=("clothing", "posture", "condition", "holding"))},
        "relations": {"type": "array", "items": _object({"object": _TEXT, "relation": _TEXT, "target": _TEXT})},
        "changes": {"type": "array", "items": _object({
            "name": _TEXT, "aspect": _TEXT, "before": {"anyOf": [_TEXT, _IDS]},
            "after": {"anyOf": [_TEXT, _IDS]}, "seen_happening": {"type": "boolean"}})},
    }, optional=("camera", "background", "place_condition", "relations", "changes"))},
})


def _outline(schema: dict) -> str:
    """schema 的一行类型写法：对象 {key: 类型}、数组 [元素]、可选 A or B；可为 null 的字段后缀 "or null"。"""
    if "anyOf" in schema:
        return " or ".join(_outline(option) for option in schema["anyOf"])
    types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
    [kind] = [item for item in types if item != "null"]
    if kind == "object":
        text = "{" + ", ".join(f"{key}: {_outline(value)}" for key, value in schema["properties"].items()) + "}"
    elif kind == "array":
        text = "[" + _outline(schema["items"]) + "]"
    else:
        text = kind
    return text + " or null" if "null" in types else text


_ANSWER_FIELDS = ", ".join(f'"{key}"' for key in ANSWER_SCHEMA["properties"])

#: 修复调用的系统指令；字段说明逐个从 ANSWER_SCHEMA 生成，不另抄一份。
REPAIR_SYSTEM = f"""Convert the text the user sends into one JSON object that has exactly these top-level fields and no others: {_ANSWER_FIELDS}. The text is a video annotation; it may wrap the JSON in Markdown fences or put sentences around it.
- Copy every id, number and value exactly as the text gives it. Do not summarize, rewrite, translate, add or drop anything.
- Write null for a field the text does not give.
- Answer with the JSON object only: no Markdown fences and no other text.

Fields ("X or null" means: write null when the text does not give X):
""" + "\n".join(f"{key}: {_outline(value)}" for key, value in ANSWER_SCHEMA["properties"].items())

REPAIR_RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "floor_annotation", "strict": True, "schema": ANSWER_SCHEMA}}


def repair_messages(text: str) -> list[dict]:
    """修复调用的消息：系统指令＋主调用原文（原样，不剥围栏）。"""
    return [{"role": "system", "content": REPAIR_SYSTEM}, {"role": "user", "content": text}]


def _drop_nulls(value):
    """修复结果：null 是 strict schema 里「字段没写」的写法，去掉后与 SYSTEM 的「留空不写」一致。"""
    if isinstance(value, dict):
        return {key: _drop_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_nulls(item) for item in value]
    return value


def example_answer() -> dict:
    start = SYSTEM.index('{"characters":')
    answer, _ = json.JSONDecoder().raw_decode(SYSTEM[start:])
    return answer


def _han(value) -> bool:
    return isinstance(value, str) and any("\u4e00" <= ch <= "\u9fff" for ch in value)


def _note(drops: list, kind: str, path: str) -> None:
    drops.append({"kind": kind, "path": path})


def _drop_key(row: dict, key: str, drops: list, kind: str, path: str) -> None:
    row.pop(key, None)
    _note(drops, kind, path)


def _string_field(row: dict, key: str, drops: list, path: str) -> None:
    if key not in row:
        return
    value = row[key]
    if _han(value):
        _drop_key(row, key, drops, "han", path)
    elif not isinstance(value, str) or not value.strip():
        _drop_key(row, key, drops, "enum", path)
    else:
        row[key] = value.strip()


def accept_answer(answer: dict, shot_count: int) -> dict:
    """原答复不改；只清洗第13版的字段，所有删项留痕。"""
    if not isinstance(answer, dict):
        return {"status": "window_dropped", "reason": "not_json", "drops": [], "cleaned": None}
    shots = answer.get("shots")
    if (not isinstance(shots, list) or len(shots) != shot_count
            or any(not isinstance(s, dict) or s.get("shot") != i for i, s in enumerate(shots, 1))):
        return {"status": "window_dropped", "reason": "shot_count", "drops": [], "cleaned": None}
    if any(not isinstance(s.get("action"), str) or not s["action"].strip() for s in shots):
        return {"status": "window_dropped", "reason": "missing_action", "drops": [], "cleaned": None}
    drops = []
    cleaned = _drop_nulls(copy.deepcopy(answer))
    cleaned = {k: v for k, v in cleaned.items() if k in ANSWER_SCHEMA["properties"]}
    for group, pattern in (("characters", CHARACTER_ID), ("objects", OBJECT_ID), ("places", LOCATION_ID)):
        rows, kept, used = cleaned.get(group), [], set()
        if not isinstance(rows, list):
            return {"status": "window_dropped", "reason": f"missing_{group}", "drops": drops, "cleaned": None}
        for i, row in enumerate(rows):
            path = f"{group}[{i}]"
            if (not isinstance(row, dict) or not isinstance(row.get("name"), str)
                    or not pattern.fullmatch(row["name"]) or row["name"] in used):
                _note(drops, "unregistered_or_duplicate", path)
                continue
            if group == "characters" and row.get("kind") not in {"person", "animal"}:
                _note(drops, "enum", path + ".kind")
                continue
            _string_field(row, "appearance", drops, path + ".appearance")
            keys = ("name", "appearance", "kind") if group == "characters" else ("name", "appearance", "landmarks") if group == "places" else ("name", "appearance")
            kept.append({k: row[k] for k in keys if k in row})
            used.add(row["name"])
        cleaned[group] = kept
    marks = {}
    for i, place in enumerate(cleaned["places"]):
        kept = []
        for j, mark in enumerate(place.get("landmarks") or []):
            path = f"places[{i}].landmarks[{j}]"
            if (not isinstance(mark, dict) or not isinstance(mark.get("name"), str)
                    or not LANDMARK_ID.fullmatch(mark["name"]) or mark["name"] in marks):
                _note(drops, "unregistered_or_duplicate", path)
                continue
            if len(kept) >= 8:
                _note(drops, "landmark_limit", path)
                continue
            _string_field(mark, "appearance", drops, path + ".appearance")
            kept.append({k: mark[k] for k in ("name", "appearance") if k in mark})
            marks[mark["name"]] = place["name"]
        place["landmarks"] = kept
    characters = {r["name"] for r in cleaned["characters"]}
    objects = {r["name"] for r in cleaned["objects"]}
    places = {r["name"] for r in cleaned["places"]}
    entities = characters | objects
    keys = ANSWER_SCHEMA["properties"]["shots"]["items"]["properties"]
    cleaned["shots"] = [{k: v for k, v in s.items() if k in keys} for s in cleaned["shots"]]
    for i, shot in enumerate(cleaned["shots"]):
        path = f"shots[{i}]"
        for key in ("action", "place_condition", "background"):
            _string_field(shot, key, drops, path + "." + key)
        if "background" in shot and len(shot["background"].split()) > 12:
            _drop_key(shot, "background", drops, "word_limit", path + ".background")
        if shot.get("place") not in places:
            _drop_key(shot, "place", drops, "unregistered", path + ".place")
        camera = shot.get("camera")
        if camera is not None and not isinstance(camera, dict):
            _drop_key(shot, "camera", drops, "enum", path + ".camera")
        elif isinstance(camera, dict):
            for key, allowed in (("shot_size", SHOT_SIZES), ("angle", ANGLES), ("movement", MOVEMENTS)):
                if key in camera and camera[key] not in allowed:
                    _drop_key(camera, key, drops, "han" if _han(camera[key]) else "enum", path + ".camera." + key)
        kept = []
        for j, mark in enumerate(shot.get("landmarks") or []):
            if mark in marks and marks[mark] == shot.get("place"):
                if mark not in kept:
                    kept.append(mark)
            else:
                _note(drops, "unregistered_or_cross_location", f"{path}.landmarks[{j}]")
        shot["landmarks"] = kept
        kept, seen = [], set()
        for j, entity in enumerate(shot.get("present") or []):
            ep = f"{path}.present[{j}]"
            if not isinstance(entity, dict) or entity.get("name") not in entities or entity["name"] in seen:
                _note(drops, "unregistered_or_duplicate", ep)
                continue
            if any(type(entity.get(k)) is not bool for k in ("in_first_frame", "in_last_frame")):
                return {"status": "window_dropped", "reason": "invalid_visibility", "drops": drops, "cleaned": None}
            for key in ("clothing", "posture", "condition"):
                _string_field(entity, key, drops, ep + "." + key)
            if "holding" in entity and (not isinstance(entity["holding"], list) or any(x not in objects for x in entity["holding"])):
                _drop_key(entity, "holding", drops, "unregistered", ep + ".holding")
            if entity["name"] in objects:
                for key in ("clothing", "posture", "holding"):
                    if key in entity:
                        _drop_key(entity, key, drops, "character_only", ep + "." + key)
            kept.append({k: v for k, v in entity.items() if k in ANSWER_SCHEMA["properties"]["shots"]["items"]["properties"]["present"]["items"]["properties"]})
            seen.add(entity["name"])
        shot["present"] = kept
        relations = []
        for j, rel in enumerate(shot.get("relations") or []):
            rp = f"{path}.relations[{j}]"
            if (not isinstance(rel, dict) or rel.get("object") not in objects
                    or rel.get("target") not in entities | set(marks)):
                _note(drops, "unregistered_or_object_only", rp)
            elif rel.get("relation") not in RELATIONS:
                _note(drops, "enum", rp)
            else:
                relations.append({k: rel[k] for k in ("object", "relation", "target")})
        shot["relations"] = relations
        changes = []
        for j, change in enumerate(shot.get("changes") or []):
            cp = f"{path}.changes[{j}]"
            if not isinstance(change, dict) or change.get("name") not in seen:
                _note(drops, "unregistered_or_not_present", cp)
                continue
            if change.get("aspect") not in ASPECTS or type(change.get("seen_happening")) is not bool:
                _note(drops, "incomplete_change", cp)
                continue
            if change["name"] in objects and change["aspect"] != "condition":
                _note(drops, "character_only", cp)
                continue
            for side in ("before", "after"):
                if change["aspect"] == "holding":
                    if not isinstance(change.get(side), list) or any(x not in objects for x in change[side]):
                        _drop_key(change, side, drops, "unregistered", cp + "." + side)
                else:
                    _string_field(change, side, drops, cp + "." + side)
            changes.append({k: change[k] for k in ("name", "aspect", "before", "after", "seen_happening") if k in change})
        shot["changes"] = changes
    if any(not s.get("action") for s in cleaned["shots"]):
        return {"status": "window_dropped", "reason": "missing_action", "drops": drops, "cleaned": None}
    return {"status": "accepted", "reason": None, "drops": drops, "cleaned": cleaned,
            "action_words": [len(s["action"].split()) for s in cleaned["shots"]]}


def annotation_name_map(cleaned: dict) -> dict[str, str]:
    mapping = {r["name"]: f"Subject {i}" for i, r in enumerate(cleaned["characters"] + cleaned["objects"], 1)}
    for place in cleaned["places"]:
        mapping[place["name"]] = place["name"]
        mapping.update({r["name"]: f"{place['name']}/Anchor {i}" for i, r in enumerate(place["landmarks"], 1)})
    return mapping


def _fact(subject: str, family: str, axis: str, value, shot: int) -> dict:
    return {"subject": subject, "family": family, "predicate_or_axis": axis, "object_or_value": value,
            "scope": "scene", "shot_span": [shot, shot], "evidence": "observed"}


def _convert_shot(shot: dict, index: int, frame_count: int, drops: list, repairs: list, subject_kinds: dict[str, str], names: dict[str, str]) -> tuple[dict, list, list]:
    """一镜：返回 (镜头行, 开场事实, 事件)。语义错只丢单条并记账。"""
    from mspc.data.event_log import registry
    front_relations = set(registry().anchor_orientation_predicates)
    path = f"shots[{index}]"
    place = shot.get("place")
    camera = shot.get("camera") or {}
    present = {names[e["name"]]: e for e in shot["present"]}
    first = {sid for sid, e in present.items() if e["in_first_frame"]}
    visible = list(present)
    anchor_ids = list(dict.fromkeys(names[m] for m in shot["landmarks"]))
    relations = []
    for number, relation in enumerate(shot.get("relations") or []):
        ends = (names[relation["object"]], names[relation["target"]])
        relation = {"subject": ends[0], "relation": relation["relation"], "object": ends[1]}
        if any(e in present and e not in first for e in ends):
            continue
        anchors = [item for item in ends if "/Anchor " in item]
        if ends[0] == ends[1]:
            _note(drops, "self_relation", f"{path}.relations[{number}]")
        elif any(item.split("/")[0] != place for item in anchors):
            _note(drops, "cross_location_anchor", f"{path}.relations[{number}]")
        elif anchors and relation["relation"] in front_relations:
            _note(drops, "anchor_no_front", f"{path}.relations[{number}]")
        else:
            relations.append(relation)
            for anchor in anchors:
                if anchor not in anchor_ids:
                    anchor_ids.append(anchor)
    opening = {}
    for sid, entity in present.items():
        if sid not in first:
            continue
        for key, axis in STATE_AXIS.items():
            if key in entity and not (key == "condition" and subject_kinds[sid] != "object"):
                opening[(sid, axis)] = entity[key]
        for held in entity.get("holding") or []:
            opening[(sid, "holds", names[held])] = names[held]
    if place and "place_condition" in shot:
        opening[(place, "environment")] = shot["place_condition"]
    for relation in relations:
        opening[(relation["subject"], relation["relation"], relation["object"])] = relation["object"]
    events = []
    for number, change in enumerate(shot.get("changes") or []):
        where = f"{path}.changes[{number}]"
        if "aspect" not in change or type(change.get("seen_happening")) is not bool:
            _note(drops, "incomplete_change", where)
            continue
        sid, seen = names[change["name"]], change["seen_happening"]
        if sid not in first:
            continue
        # body 已承载 posture；人的临时状态留在编译 sidecar，避免覆盖姿态或误记为物体状态。
        if change["aspect"] == "condition" and subject_kinds[sid] != "object":
            continue
        if change["aspect"] == "holding":
            if not isinstance(change.get("before"), list) or not isinstance(change.get("after"), list):
                _note(drops, "incomplete_change", where)
                continue
            before, after = {names[v] for v in change["before"]}, {names[v] for v in change["after"]}
            if not (before | after) <= first:
                continue
            for held in sorted(before):
                opening.setdefault((sid, "holds", held), held)
                if seen and held not in after:
                    events.append({"before": _fact(sid, "relation", "holds", held, index), "after": None,
                                   "on_screen": seen, "shot_index": index})
            for held in sorted(after):
                if seen and held not in before:
                    events.append({"before": None, "after": _fact(sid, "relation", "holds", held, index),
                                   "on_screen": seen, "shot_index": index})
        elif isinstance(change.get("before"), str) and isinstance(change.get("after"), str):
            axis = STATE_AXIS[change["aspect"]]
            registered_before = opening.setdefault((sid, axis), change["before"])
            if registered_before != change["before"]:
                repairs.append({"kind": "before_reset", "path": where})
            if not seen:
                opening[(sid, axis)] = change["before"]
                continue
            if registered_before == change["after"]:
                _note(drops, "no_change", where)
                continue
            events.append({"before": _fact(sid, "state", axis, registered_before, index),
                           "after": _fact(sid, "state", axis, change["after"], index),
                           "on_screen": seen, "shot_index": index})
        else:
            _note(drops, "incomplete_change", where)
    initial = [_fact(key[0], "state", key[1], value, index) if len(key) == 2
               else _fact(key[0], "relation", key[1], value, index) for key, value in opening.items()]
    row = {"shot_id": f"s{index + 1:02d}", "location_id": place or "unknown",
           "present_subjects": visible, "visible_entities": [sid for sid in visible if sid in first] + anchor_ids, "screen_facts": [],
           "frame_count": frame_count, "shot_size": canonical_shot_field("shot_size", camera.get("shot_size", "unknown")),
           "camera_angle": canonical_shot_field("camera_angle", camera.get("angle", "unknown")), "movement": camera.get("movement", "unknown"),
           "action": re.sub(r"(?:Character|Object|Landmark|Location) [1-9]\d*", lambda m: names.get(m[0], m[0]), shot["action"])}
    return row, initial, events


def _assemble(cleaned: dict, shots: list, initial: list, events: list) -> dict:
    names = annotation_name_map(cleaned)
    subjects = [{"subject_id": names[row["name"]], "kind": row["kind"], "appearance": row.get("appearance", "unknown")}
                for row in cleaned["characters"]]
    subjects += [{"subject_id": names[row["name"]], "kind": "object", "appearance": row.get("appearance", "unknown")}
                 for row in cleaned["objects"]]
    locations = [{"location_id": row["name"], "appearance": row.get("appearance", "unknown"),
                  "anchors": [{"id": names[mark["name"]], "description": mark.get("appearance", "unknown"), "front_definition": None}
                              for mark in row["landmarks"]]} for row in cleaned["places"]]
    return {"schema": EVENT_LOG_SCHEMA, "subjects": subjects, "locations": locations,
            "initial_state": initial, "shots": shots,
            "events": [{"event_id": f"e{number}", "shot_id": f"s{event['shot_index'] + 1:02d}",
                        "before": event["before"], "after": event["after"], "on_screen": event["on_screen"]}
                       for number, event in enumerate(events, 1)]}


def to_event_log(cleaned: dict, frame_counts: list[int], drops: list, repairs: list) -> dict:
    """请求字段到 event_log_v3。缺的镜头字段写成 unknown；不给锚点编造正面。

    语义层面的单条错误只丢该条（记入 drops）：自指或跨地点或无正面的锚点关系、无变化或残缺的变化，
    以及校验器按事件指出的错事件。变化前状态与首帧登记不同时取首帧登记值（记入 repairs）。
    校验器指出的不是事件的错（结构性）仍抛 EventLogError，由调用方整窗拒收。
    """
    from mspc.data.event_log import EventLogError, parse_event_log, validate_event_log
    if len(frame_counts) != len(cleaned["shots"]):
        raise ValueError("frame counts must match the accepted shots")
    shots, initial, events = [], [], []
    names = annotation_name_map(cleaned)
    subject_kinds = {names[row["name"]]: row["kind"] for row in cleaned["characters"]}
    subject_kinds.update({names[row["name"]]: "object" for row in cleaned["objects"]})
    for index, shot in enumerate(cleaned["shots"]):
        row, opening, shot_events = _convert_shot(shot, index, frame_counts[index], drops, repairs, subject_kinds, names)
        shots.append(row)
        initial.extend(opening)
        events.extend(shot_events)
    while True:
        raw = _assemble(cleaned, shots, initial, events)
        try:
            log = parse_event_log(raw)
            validate_event_log(log)
        except EventLogError as exc:
            bad = {int(issue.path[1:]) - 1: issue for issue in exc.issues if re.fullmatch(r"e[1-9]\d*", issue.path)}
            if not bad:
                raise
            for number in sorted(bad):
                drops.append({"kind": "event_invalid", "path": f"events[{number}]", "detail": bad[number].message})
            events = [event for number, event in enumerate(events) if number not in bad]
            continue
        # 画外 holding 的空列表不能被当前时间线用 unknown 清除继承值。
        # 不能安全表达的窗拒收，而不是让 before=[] 被上一镜 holds 覆盖。
        from mspc.data.event_log import snapshots
        starts = snapshots(log)
        for shot in cleaned["shots"]:
            first = {e["name"] for e in shot["present"] if e["in_first_frame"]}
            for change in shot["changes"]:
                if change["seen_happening"] is False and change["aspect"] == "holding" and change["name"] in first and "before" in change:
                    sid = names[change["name"]]
                    actual = {f.object_or_value for f in starts[f"s{shot['shot']:02d}"].values()
                              if f.subject == sid and f.family == "relation" and f.predicate_or_axis == "holds"}
                    expected = {names[v] for v in change["before"]}
                    if actual != expected:
                        raise ValueError(f"s{shot['shot']:02d} {sid}: unrepresentable offscreen holding before; preserved in annotation_changes")
        return raw


def annotation_frame_evidence(cleaned: dict, log) -> dict:
    """首末可见名单照标注；不可表达的末态不冒充已观测的历史帧状态。"""
    from dataclasses import replace
    from mspc.data.event_log import endpoints, frame_evidence
    names = annotation_name_map(cleaned)
    result = {}
    for shot in cleaned["shots"]:
        sid = f"s{shot['shot']:02d}"
        last = [names[e["name"]] for e in shot["present"] if e["in_last_frame"]]
        late = {names[e["name"]] for e in shot["present"] if not e["in_first_frame"]}
        unresolved = {(names[c["name"]], "holds" if c["aspect"] == "holding" else STATE_AXIS[c["aspect"]])
                      for c in shot["changes"] if not c["seen_happening"]}
        frame = frame_evidence(log, sid, "last", last)
        states = {k: f for k, f in frame.result_states.items()
                  if not endpoints(f) & late and (f.subject, f.predicate_or_axis) not in unresolved}
        result[sid] = replace(frame, result_states=states)
    return result


def shot_annotation_sidecars(cleaned: dict) -> dict:
    """保留模型时间线不能表达的标注；名字映射同时带出，不伪造开头或末尾事实。"""
    names = annotation_name_map(cleaned)
    people = {r["name"] for r in cleaned["characters"] if r["kind"] == "person"}
    return {
        "annotation_name_map": names,
        "shot_visibility": [{"shot_id": f"s{s['shot']:02d}",
                             "first": [names[e["name"]] for e in s["present"] if e["in_first_frame"]],
                             "last": [names[e["name"]] for e in s["present"] if e["in_last_frame"]]}
                            for s in cleaned["shots"]],
        "shot_background": [{"shot_id": f"s{s['shot']:02d}", **({"background": s["background"]} if "background" in s else {})}
                            for s in cleaned["shots"]],
        "person_conditions": [{"shot_id": f"s{s['shot']:02d}",
                               "entities": [{"id": names[e["name"]], "condition": e["condition"]}
                                            for e in s["present"] if e["name"] in people and "condition" in e],
                               "changes": [c for c in s["changes"] if c["name"] in people and c["aspect"] == "condition"]}
                              for s in cleaned["shots"]],
        "first_visible_states": [{"shot_id": f"s{s['shot']:02d}",
                                  "present": [e for e in s["present"] if not e["in_first_frame"]]}
                                 for s in cleaned["shots"]],
        "annotation_changes": [{"shot_id": f"s{s['shot']:02d}", "changes": s["changes"]} for s in cleaned["shots"]],
        "annotation_relations": [{"shot_id": f"s{s['shot']:02d}", "relations": s["relations"]} for s in cleaned["shots"]],
    }


def messages(window: dict, assets: Path) -> list[dict]:
    stamps = window["shot_timestamps"]
    lines = [f"Annotate this video. It has {len(stamps)} shots:"]
    lines.extend(f"Shot {item['shot'] + 1}: {float(item['start']):.1f}\u2013{float(item['end']):.1f} s" for item in stamps)
    lines.append(f"shots must contain exactly {len(stamps)} items, numbered 1..{len(stamps)}; do not merge or split fast cuts.")
    content = [{"type": "text", "text": "\n".join(lines)}]
    video = assets / "inputs/videos" / Path(window["video"]).name
    content.append({"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(video.read_bytes()).decode()},
                    "fps": window["fps"]})
    content.append({"type": "text", "text": "Return the JSON object."})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def durable_json(path: Path, value: object) -> None:
    """原子替换并同步文件/目录；中断不能把已提交标记变成半个JSON。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _usd(usage: dict, input_usd_per_mtok, output_usd_per_mtok) -> Decimal:
    for key in ("prompt_tokens", "completion_tokens"):
        if type(usage[key]) is not int or usage[key] < 0:
            raise ValueError("usage token counts must be nonnegative integers")
    return (Decimal(usage["prompt_tokens"]) * Decimal(str(input_usd_per_mtok))
            + Decimal(usage["completion_tokens"]) * Decimal(str(output_usd_per_mtok))) / 1_000_000


def estimate_usage(usage: dict, cfg: dict) -> Decimal:
    """主调用（cfg["model"]）的估价。"""
    return _usd(usage, cfg["input_usd_per_mtok"], cfg["output_usd_per_mtok"])


def estimate_repair_usage(usage: dict, cfg: dict) -> Decimal:
    """修复调用（cfg["repair_model"]）的估价；单价只从配置读，缺键即 KeyError。"""
    return _usd(usage, cfg["repair_input_usd_per_mtok"], cfg["repair_output_usd_per_mtok"])


def main_reserve(cfg: dict) -> Decimal:
    return estimate_usage({"prompt_tokens": cfg["reserve_prompt_tokens"],
                           "completion_tokens": cfg["reserve_completion_tokens"]}, cfg)


def repair_reserve(cfg: dict) -> Decimal:
    return estimate_repair_usage({"prompt_tokens": cfg["repair_reserve_prompt_tokens"],
                                  "completion_tokens": cfg["repair_reserve_completion_tokens"]}, cfg)


def attempt_reserve(cfg: dict) -> Decimal:
    """一次尝试的预留：主调用＋可能发生的一次修复调用。"""
    return main_reserve(cfg) + repair_reserve(cfg)


def prior_occupancy(history: dict) -> Decimal:
    """显式历史attempt清单；不扫描相邻目录或把恢复回执算第二遍。"""
    if history["schema_version"] != "state_pilot_budget_v1" or history["currency"] != "USD":
        raise ValueError("explicit USD attempt history required")
    seen, total = set(), Decimal(0)
    for entry in history["attempts"]:
        if entry["attempt_id"] in seen:
            raise ValueError("duplicate historical attempt")
        seen.add(entry["attempt_id"])
        if not entry["evidence"] or entry["status"] not in {"usage_known", "charge_unknown"}:
            raise ValueError("historical attempt requires status and evidence")
        value = Decimal(str(entry["estimated_occupancy_usd"]))
        if not value.is_finite() or value < 0:
            raise ValueError("invalid historical occupancy")
        total += value
    return total


def sdk_stream(cfg: dict, request: dict):
    from openai import OpenAI
    import httpx
    with OpenAI(api_key=os.environ["CLOSEAI_API_KEY"], base_url=cfg["base_url"], max_retries=0,
                timeout=httpx.Timeout(connect=15, read=cfg["timeout_seconds"], write=60, pool=15),
                http_client=httpx.Client(trust_env=False)) as sdk:
        with sdk.chat.completions.create(**request, stream=True, stream_options={"include_usage": True}) as stream:
            yield from stream


RETRYABLE_REASONS = {"not_json", "request_failed", "shot_count", "missing_action", "truncated"}
#: 阿里云输入内容审查拒收的报错片段（DashScope ``InternalError.Algo.DataInspectionFailed``，输入侧）。
INPUT_INSPECTION_MARK = "DataInspectionFailed: Input"
FATAL_HTTP_STATUS = {401, 402, 403, 404}


def is_retryable(state: dict) -> bool:
    """窗失败是否值得重发：结构性失败可重采样；鉴权、欠费、找不到端点重发无益。"""
    if state["status"] == "succeeded" or state.get("reserve_exceeded"):
        return False
    return (state.get("reason") or state["status"]) in RETRYABLE_REASONS and state.get("http_status") not in FATAL_HTTP_STATUS


#: 在运行目录里放这个文件即请求安全停下：不再发新请求，在飞的照常结清后本阶段正常收尾。
STOP_REQUEST_FILE = "stop-requested"


def is_settled(attempts: Path, window_id: str, max_attempts: int) -> bool:
    """该窗不会再发请求：最后一次尝试已结清，且成功、不可重试或次数用完。"""
    found = attempt_dirs(attempts, window_id)
    if not found:
        return False
    last = json.loads((found[-1] / "attempt.json").read_text())
    if last["status"] in {"reserved", "submitted"}:
        return False
    return not is_retryable(last) or len(found) >= max_attempts


def attempt_dirs(attempts: Path, window_id: str) -> list[Path]:
    """该窗按尝试序的目录：初发是 window_id，第 k 次重发是 window_id-retryK。"""
    found = [attempts / window_id] if (attempts / window_id / "attempt.json").exists() else []
    number = 1
    while (attempts / f"{window_id}-retry{number}" / "attempt.json").exists():
        found.append(attempts / f"{window_id}-retry{number}")
        number += 1
    return found


def strip_outer_fence(text: str) -> str:
    """只剥包住整段回复的最外层 Markdown 代码围栏（开头一行 ``` 或 ```json，结尾 ```）；别的一律原样。

    主调用输出先剥围栏再解析，不做别的提取。地板 166 窗里解析不出的 10 次、S1 的
    floor-0005 都是这种整段包围栏的完整 JSON。围栏外另有文字的回复不算，原样返回、照旧走修复调用。
    """
    body = text.strip()
    newline = body.find("\n")
    if body.startswith("```") and body.endswith("```") and newline != -1 and len(body) >= newline + 4:
        if body[3:newline].strip() in ("", "json"):
            return body[newline + 1:-3]
    return text


def _json_object(text: str) -> dict | None:
    """整段文本 ``json.loads`` 得到对象才算；不做任何提取。"""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def repair_max_tokens(cfg: dict, main_usage: dict | None) -> int:
    """修复调用的输出上限：主调用正文 token 数（completion 减思考）× ``repair_max_tokens_ratio``，向上取整。

    10-09 S1 的 floor-0005 修复退化成重复输出、写满网关上限 65,536 token；有了上限，退化输出会被截断并按
    修复失败记这一窗。主调用没有 usage 时正文长度未知，取主调用自己的正文上限 ``max_tokens``。
    """
    ratio = Decimal(str(cfg["repair_max_tokens_ratio"]))
    if main_usage is None:
        content = cfg["max_tokens"]
    else:
        reasoning = (main_usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        content = main_usage["completion_tokens"] - reasoning
    if content <= 0:
        raise ValueError("main call content tokens must be positive to size the repair call")
    return int((Decimal(content) * ratio).to_integral_value(rounding="ROUND_CEILING"))


def error_chain(exc: BaseException) -> list[str]:
    """异常及其 ``__cause__`` 链上的类名，外层在前。SDK 把 httpx 的异常包成 APITimeoutError／APIConnectionError
    （``raise ... from err``），两者的文字都不带原因；原因只在这条链上。"""
    names = []
    while exc is not None:
        names.append(type(exc).__name__)
        exc = exc.__cause__
    return names


#: httpx 在连接没建起来时抛的三种异常：TCP／TLS 握手超时、握手出错、连接池等不到连接。发生时请求一个字节都没发出去。
#: 10-10 主服务器实测（openai 2.45.0／httpx 0.28.1）：连不上的地址报 "Request timed out."，链为 APITimeoutError ←
#: ConnectTimeout；拒绝连接报 "Connection error."，链为 APIConnectionError ← ConnectError。读超时与服务端断开
#: 报的是同样两句话，链上是 ReadTimeout／RemoteProtocolError，所以只认链、不认文字。
CONNECT_PHASE_ERRORS = {"ConnectTimeout", "ConnectError", "PoolTimeout"}


def _streamed_call(*, attempt_path: Path, prefix: str, state: dict, cfg: dict, stream_factory,
                   model: str, prices: dict, reserve_prompt_tokens: int, reserve_completion_tokens: int,
                   messages: list, temperature, response_format, max_tokens, enable_thinking: bool):
    """一次流式调用：请求原文、逐块流、进度与终态回复都先落盘；``prefix`` 区分主调用与修复调用的文件。"""
    start = time.monotonic()
    progress = {"received_chunks": 0, "reasoning_characters": 0, "content_characters": 0,
                "usage": None, "finish_reason": None, "returned_model": None, "request_id": None,
                "error_chain": None, "connect_failed": False}
    raw_pieces, reasoning_pieces = [], []
    def transport(**request):
        durable_json(attempt_path / f"{prefix}request.json", sanitize_log_value(
            {**request, "stream": True, "stream_options": {"include_usage": True}}, secrets=(os.environ["CLOSEAI_API_KEY"],)))
        state.update({"status": "submitted", f"{prefix}submitted_at": utc_now()})
        durable_json(attempt_path / "attempt.json", state)
        def chunks():
            with (attempt_path / f"{prefix}stream.jsonl").open("x") as handle:
                for chunk in stream_factory(cfg, request):
                    handle.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    progress["received_chunks"] += 1
                    progress["returned_model"] = chunk.model or progress["returned_model"]
                    progress["request_id"] = chunk.id or progress["request_id"]
                    if chunk.usage is not None:
                        progress["usage"] = chunk.usage.model_dump()
                    for choice in chunk.choices:
                        if choice.index != 0:
                            raise ValueError("unexpected multiple completion choices")
                        content = choice.delta.content or ""
                        reasoning = getattr(choice.delta, "reasoning_content", None) or ""
                        raw_pieces.append(content)
                        reasoning_pieces.append(reasoning)
                        progress["reasoning_characters"] += len(reasoning)
                        progress["content_characters"] += len(content)
                        if choice.finish_reason:
                            progress["finish_reason"] = choice.finish_reason
                    progress["elapsed_seconds"] = time.monotonic() - start
                    durable_json(attempt_path / f"{prefix}progress.json", progress)
                    yield chunk
        try:
            return _reassemble_stream(chunks())
        except Exception as exc:
            progress["http_status"] = getattr(exc, "status_code", None)
            progress["error_chain"] = error_chain(exc)
            # 连接阶段就失败、没收到任何响应：请求没发出去。
            progress["connect_failed"] = (progress["received_chunks"] == 0
                                          and not CONNECT_PHASE_ERRORS.isdisjoint(progress["error_chain"]))
            raise
    counter = CloseAICostCounter(hard_cap_cny=cfg["budget_usd"], prices={model: prices},
        precall_prompt_tokens=reserve_prompt_tokens, precall_completion_tokens=reserve_completion_tokens)
    client = CloseAIClient(api_key=os.environ["CLOSEAI_API_KEY"], base_url=cfg["base_url"], cost_counter=counter,
        timeout_seconds=cfg["timeout_seconds"], max_retries=1, retry_base_seconds=1, rpm_limit=cfg["rpm_limit"], transport=transport)
    response, error, truncated = None, None, False
    try:
        response = client.complete_messages(model=model, messages=messages, temperature=temperature,
                                            response_format=response_format, max_tokens=max_tokens,
                                            enable_thinking=enable_thinking)
    except CloseAIResponseTruncated as exc:
        error, truncated = str(exc), True
    except Exception as exc:
        error = str(exc)
    durable_json(attempt_path / f"{prefix}response.json", {**progress, "content": "".join(raw_pieces),
                                                           "reasoning_content": "".join(reasoning_pieces), "error": error})
    return response, error, truncated, progress


def annotate(*, window: dict, output: Path, assets: Path, cfg: dict, stop: Event,
             stream_factory=sdk_stream, stem: str) -> None:
    attempt_path = output / "attempts" / stem
    state = json.loads((attempt_path / "attempt.json").read_text())
    if state["status"] != "reserved" or stop.is_set():
        return
    start = time.monotonic()
    # 主调用：思考模型开思考、不设 response_format，max_tokens 照配置。
    response, error, truncated, progress = _streamed_call(
        attempt_path=attempt_path, prefix="", state=state, cfg=cfg, stream_factory=stream_factory,
        model=cfg["model"], prices={"input_cny_per_mtok": cfg["input_usd_per_mtok"], "output_cny_per_mtok": cfg["output_usd_per_mtok"]},
        reserve_prompt_tokens=cfg["reserve_prompt_tokens"], reserve_completion_tokens=cfg["reserve_completion_tokens"],
        messages=messages(window, assets), temperature=cfg["temperature"], response_format=cfg["response_format"],
        max_tokens=cfg["max_tokens"], enable_thinking=cfg["enable_thinking"])
    if progress["usage"] is not None:
        occupancy = estimate_usage(progress["usage"], cfg)
        usage = {"window_id": window["window_id"], "model": cfg["model"], **progress,
                 "provider_billed_cost": None, "estimated_usd": str(occupancy), "price_basis": cfg["price_basis"]}
        durable_json(attempt_path / "usage.json", usage)
        if (progress["usage"]["prompt_tokens"] > cfg["reserve_prompt_tokens"]
                or progress["usage"]["completion_tokens"] > cfg["reserve_completion_tokens"]):
            stop.set()
            state["reserve_exceeded"] = True
    elif progress["connect_failed"]:
        # 连接阶段就失败、一个字节都没收到的尝试按 $0 计（请求没发出去）。
        occupancy = Decimal(0)
    else:
        # 其余无 usage 的终态（请求已发出后读超时、流中断、被拒收）仍留下主调用预留，预算检查继续用它卡上限。
        # 单窗失败不中止其余窗。
        occupancy = main_reserve(cfg)
        state["charge_unknown"] = True
    parsed = None
    if truncated:
        state.update(status="window_dropped", reason="truncated", truncated=True, error=error)
    elif error is not None and INPUT_INSPECTION_MARK in error:
        # 输入内容审查拒收是确定性的（S1 floor-0273 连发三次同样被拒，各占一份整槽预留）：丢窗、不重发。
        state.update(status="window_dropped", reason="input_inspection", error=error, http_status=progress.get("http_status"))
    elif error is not None:
        state.update(status="request_failed", reason="request_failed", error=error, http_status=progress.get("http_status"),
                     error_chain=progress["error_chain"], connect_failed=progress["connect_failed"])
    else:
        durable_json(attempt_path / "raw.json", {"text": response.text})
        unfenced = strip_outer_fence(response.text)
        parsed = _json_object(unfenced)
        if parsed is not None and unfenced != response.text:
            state["fence_stripped"] = True
        if parsed is None and stop.is_set():
            # 预算已叫停：不再发修复调用。
            state.update(status="window_dropped", reason="not_json")
        elif parsed is None:
            # 依据：阿里云百炼「结构化输出」页 FAQ——思考模型先不设 response_format，解析失败再交非思考模型按
            # json_schema 修复。网关不执行 strict（10-09 实测），形状由 REPAIR_SYSTEM 给。输出上限按主调用正文
            # 长度给，被截断即本窗修复失败。
            repair, repair_error, repair_truncated, repair_progress = _streamed_call(
                attempt_path=attempt_path, prefix="repair_", state=state, cfg=cfg, stream_factory=stream_factory,
                model=cfg["repair_model"], prices={"input_cny_per_mtok": cfg["repair_input_usd_per_mtok"],
                                                   "output_cny_per_mtok": cfg["repair_output_usd_per_mtok"]},
                reserve_prompt_tokens=cfg["repair_reserve_prompt_tokens"],
                reserve_completion_tokens=cfg["repair_reserve_completion_tokens"],
                messages=repair_messages(response.text), temperature=cfg["repair_temperature"],
                response_format=REPAIR_RESPONSE_FORMAT, max_tokens=repair_max_tokens(cfg, progress["usage"]),
                enable_thinking=cfg["repair_enable_thinking"])
            if repair_progress["usage"] is not None:
                repair_usd = estimate_repair_usage(repair_progress["usage"], cfg)
                state.update(repair_usage=repair_progress["usage"], repair_estimated_usd=str(repair_usd))
                durable_json(attempt_path / "repair_usage.json", {
                    "window_id": window["window_id"], "model": cfg["repair_model"], **repair_progress,
                    "provider_billed_cost": None, "estimated_usd": str(repair_usd), "price_basis": cfg["price_basis"]})
                if (repair_progress["usage"]["prompt_tokens"] > cfg["repair_reserve_prompt_tokens"]
                        or repair_progress["usage"]["completion_tokens"] > cfg["repair_reserve_completion_tokens"]):
                    stop.set()
                    state["reserve_exceeded"] = True
            elif repair_progress["connect_failed"]:
                repair_usd = Decimal(0)
            else:
                repair_usd = repair_reserve(cfg)
                state["charge_unknown"] = True
            occupancy += repair_usd
            if repair_error is not None and not repair_truncated:
                # 请求层失败（超时、连接、空流、5xx 等）：修复没拿到回复，整窗按 request_failed 走重发。
                state.update(status="request_failed", reason="request_failed", error=repair_error,
                             repair_error=repair_error, http_status=repair_progress.get("http_status"),
                             error_chain=repair_progress["error_chain"], connect_failed=repair_progress["connect_failed"])
            else:
                parsed = None if repair is None else _json_object(repair.text)
                if repair is not None:
                    durable_json(attempt_path / "repair_raw.json", {"text": repair.text})
                if parsed is None or not set(ANSWER_SCHEMA["required"]) <= parsed.keys():
                    # 修复回了内容（含截断）仍不是带三个顶层字段的对象（10-09 实测会回 {"error", "response"}）：
                    # 记修复失败，不当成镜数错去重发。
                    state.update(status="window_dropped", reason="repair_failed", repair_error=repair_error)
                    parsed = None
                else:
                    parsed = _drop_nulls(parsed)
                    state["repaired"] = True
    if parsed is not None:
        acceptance = accept_answer(parsed, len(window["shot_timestamps"]))
        durable_json(attempt_path / "acceptance.json", {k: v for k, v in acceptance.items() if k != "cleaned"})
        if acceptance["status"] != "accepted":
            state.update(status="window_dropped", reason=acceptance["reason"], drops=acceptance["drops"])
        else:
            durable_json(attempt_path / "candidate.json", {"answer": parsed, "cleaned": acceptance["cleaned"],
                "drops": acceptance["drops"], "action_words": acceptance["action_words"],
                "repaired": state.get("repaired", False), "fence_stripped": state.get("fence_stripped", False)})
            state.update(status="succeeded", drops=len(acceptance["drops"]))
    state.update(estimated_occupancy_usd=str(occupancy), finished_at=utc_now(), elapsed_seconds=time.monotonic() - start,
                 finish_reason=progress["finish_reason"], usage_known=progress["usage"] is not None)
    durable_json(attempt_path / "attempt.json", state)
    print(json.dumps({k: v for k, v in state.items() if k != "drops"}, ensure_ascii=False), flush=True)


def reserve_attempt(attempts: Path, window_id: str, stem: str, reserve: Decimal) -> None:
    path = attempts / stem
    path.mkdir(exist_ok=True)
    durable_json(path / "attempt.json", {"attempt_id": stem, "window_id": window_id,
        "status": "reserved", "reserved_at": utc_now(), "reserve_usd": str(reserve),
        "estimated_occupancy_usd": str(reserve), "provider_billed_cost": None})


def run_phase(*, inputs: dict, cfg: dict, output: Path, assets: Path, history: dict,
              windows: int, workers: int, stream_factory=sdk_stream,
              window_ids: list[str] | None = None) -> dict:
    if inputs["ready"] is not True:
        raise ValueError("inputs not ready or invalid window count")
    if window_ids is None:
        if not 1 <= windows <= len(inputs["windows"]):
            raise ValueError("inputs not ready or invalid window count")
        jobs = inputs["windows"][:windows]
    else:
        by_id = {window["window_id"]: window for window in inputs["windows"]}
        missing = [item for item in window_ids if item not in by_id]
        if missing or len(window_ids) != len(set(window_ids)):
            raise ValueError("window_ids must be unique ids present in inputs")
        jobs = [by_id[item] for item in window_ids]
        windows = len(jobs)
    if type(cfg["window_max_attempts"]) is not int or cfg["window_max_attempts"] < 1:
        raise ValueError("window_max_attempts must be a positive int: total tries per window, first call included")
    if type(cfg["workers"]) is not int or cfg["workers"] < 1:
        raise ValueError("workers must be a positive int: concurrent attempts in the rolling phase")
    if type(workers) is not int or not 1 <= workers <= cfg["workers"]:
        raise ValueError("workers must be between 1 and the configured workers")
    if windows <= 5 and workers != 1:
        raise ValueError("first five windows must run sequentially")
    if cfg["contract_version"] != CONTRACT_VERSION:
        raise ValueError("config must name current contract version")
    if cfg["response_format"] is not None or cfg["enable_thinking"] is not True or cfg["temperature"] != 0.6:
        raise ValueError("main call shape must be no response_format, thinking on, temperature 0.6")
    if type(cfg["reserve_completion_tokens"]) is not int or cfg["reserve_completion_tokens"] < cfg["max_tokens"]:
        raise ValueError("reserve_completion_tokens must include answer and reasoning")
    if (not isinstance(cfg["repair_model"], str) or not cfg["repair_model"].strip()
            or cfg["repair_enable_thinking"] is not False or cfg["repair_temperature"] != 0):
        raise ValueError("repair call shape must be a named model, thinking off, temperature 0")
    for key in ("repair_reserve_prompt_tokens", "repair_reserve_completion_tokens"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive int")
    if cfg["repair_reserve_completion_tokens"] < repair_max_tokens(cfg, None):
        raise ValueError("repair_reserve_completion_tokens must cover the largest repair output cap "
                         "(max_tokens × repair_max_tokens_ratio)")
    historical = prior_occupancy(history)
    reserve = attempt_reserve(cfg)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = {"config": cfg, "history": history, "inputs": inputs, "system": SYSTEM, "answer_schema": ANSWER_SCHEMA,
                    "repair_system": REPAIR_SYSTEM}
        snapshot = output / "contract.json"
        if snapshot.exists() and json.loads(snapshot.read_text()) != contract:
            raise ValueError("contract changed: use a new output version")
        durable_json(snapshot, contract)
        attempts = output / "attempts"
        attempts.mkdir(exist_ok=True)
        states = {p.parent.name: json.loads(p.read_text()) for p in attempts.glob("*/attempt.json")}
        if any(s["status"] == "submitted" or s.get("reserve_exceeded") for s in states.values()):
            raise ValueError("unsettled submitted attempt/reserve breach: reconcile explicitly before new requests")
        # 预留按尝试滚动占用：同时在飞的尝试各占一份整槽预留，结清后换成实际估价；任何时刻
        # 历史＋已结清＋在飞预留 ≤ 硬顶。预算暂时不够而还有在飞尝试时等它们结清；没有在飞的可等就不再发。
        limit = Decimal(str(cfg["budget_usd"]))
        book = {stem: Decimal(str(state["estimated_occupancy_usd"])) for stem, state in states.items()}
        pending = [window for window in jobs if not is_settled(attempts, window["window_id"], cfg["window_max_attempts"])]
        if pending and historical + sum(book.values(), Decimal(0)) + reserve > limit:
            raise ValueError("reserved estimate for one more attempt exceeds USD budget")
        durable_json(output / "phase-reservation.json", {"windows": windows, "workers": workers,
            "jobs": [window["window_id"] for window in jobs], "pending": len(pending),
            "historical_occupancy_usd": str(historical),
            "occupied_usd": str(historical + sum(book.values(), Decimal(0))),
            "attempt_reserve_usd": str(reserve), "max_in_flight_reserve_usd": str(workers * reserve),
            "budget_usd": cfg["budget_usd"], "reserved_at": utc_now(), "price_basis": cfg["price_basis"]})
        stop, halt = Event(), Event()
        budget = Condition()
        running, budget_blocked, flags = set(), [], {"rate_limited": False, "stop_requested": False}
        stop_file = output / STOP_REQUEST_FILE

        def claim(wid: str, stem: str, path: Path) -> bool:
            """取得跑这次尝试的资格：已有预留的直接跑；否则在预算内新建预留，不够就等在飞的结清。"""
            with budget:
                while True:
                    if stop_file.exists():
                        flags["stop_requested"] = True
                    if stop.is_set() or halt.is_set() or flags["stop_requested"]:
                        return False
                    if path.exists():
                        break
                    if historical + sum(book.values(), Decimal(0)) + reserve <= limit:
                        reserve_attempt(attempts, wid, stem, reserve)
                        book[stem] = reserve
                        break
                    if not running:
                        budget_blocked.append(wid)
                        return False
                    budget.wait()
                running.add(stem)
                return True

        def settle(stem: str, state: dict) -> None:
            with budget:
                running.discard(stem)
                book[stem] = Decimal(str(state["estimated_occupancy_usd"]))
                if state.get("http_status") == 429:
                    # 网关限流：不再发新请求（在飞的照常结清），由调用方降并发后续跑。
                    flags["rate_limited"] = True
                    halt.set()
                budget.notify_all()

        def one(window):
            wid = window["window_id"]
            for number in range(cfg["window_max_attempts"]):
                stem = wid if number == 0 else f"{wid}-retry{number}"
                path = attempts / stem / "attempt.json"
                if path.exists() and json.loads(path.read_text())["status"] != "reserved":
                    state = json.loads(path.read_text())
                else:
                    if not claim(wid, stem, path):
                        return
                    try:
                        annotate(window=window, output=output, assets=assets, cfg=cfg, stop=stop,
                                 stream_factory=stream_factory, stem=stem)
                    finally:
                        state = json.loads(path.read_text())
                        settle(stem, state)
                if not is_retryable(state) or stop.is_set():
                    return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, jobs))
        states = [json.loads(p.read_text()) for p in sorted(attempts.glob("*/attempt.json"))]
        tries = {window["window_id"]: [json.loads((d / "attempt.json").read_text()) for d in attempt_dirs(attempts, window["window_id"])]
                 for window in jobs}
        summary = {"attempts": states,
                   "windows": [{"window_id": wid, "tries": len(rows), "final_status": rows[-1]["status"],
                                "final_reason": rows[-1].get("reason"), "retryable": is_retryable(rows[-1])}
                               if rows else
                               {"window_id": wid, "tries": 0, "final_status": "not_started", "final_reason": None, "retryable": False}
                               for wid, rows in tries.items()], "historical_occupancy_usd": str(historical),
                   "estimated_occupancy_usd": str(historical + sum((Decimal(str(s["estimated_occupancy_usd"])) for s in states), Decimal(0))),
                   "stopped_on_unknown_or_reserve_breach": stop.is_set(),
                   "stopped_on_rate_limit": flags["rate_limited"], "stop_requested": flags["stop_requested"],
                   "budget_blocked": sorted(set(budget_blocked)), "provider_billed_cost": None,
                   "updated_at": utc_now()}
        durable_json(output / "summary.json", summary)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("assets", "config", "output", "prior-budget"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--windows", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    run_phase(inputs=json.loads((args.assets / "inputs.json").read_text()),
              cfg=json.loads(args.config.read_text()), output=args.output, assets=args.assets,
              history=json.loads(args.prior_budget.read_text()), windows=args.windows, workers=args.workers)


if __name__ == "__main__":
    main()
