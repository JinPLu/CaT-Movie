"""读历史两步。第一步：看段前已生成的各镜片段，每个所需角色选一个时刻（作答只有 picks）。第二步：看选好的参考帧
（图片）指出所需角色，并给本段各目标镜各人的镜首点。

第一步给出参考帧；第二步只看这几张图：参考 i 标「Reference i (Shot k, t s)」，
再加最近一个历史镜的末帧（标「Last frame of Shot s」，用来接上左右方向），连同各目标镜的分镜文字、开镜在场的人
与所需状态。作答：每张参考图里各所需角色的点，各目标镜各在场角色的镜首点（看不见写 null）。

训练时参考帧取读历史标签选的帧，作答取标签（教师强制）；零样本时参考帧取第一步作答，作答由规划器生成。
两边用这里同一份提示。坐标 0–1000 相对宽高。

镜首人物token从计划（推理）或标签（训练）的真实第0帧框中心构造；晚入场不提前。
目标人物编号与几何组保留完整语义、方位、前后和画风文字。几何组沿同一人物token加入宽高和显式前后关系，
不把度量深度编码为时间。源图身份读取在最终选帧后进行；群体保留可见ID文字，但不提供单人内部点。
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from mspc.data.history_reading_labels import opening_state
from mspc.inference.visual_reading import character_phrase

SOURCE_FPS = 16
VIDEO_FPS = 2
COORDINATE_MAX = 1000
MAX_REFERENCES = 10

SELECTION_RULE = (
    "How to pick a moment for a person: first, the person must really be there and recognisable, and their outfit and "
    "held objects should match what the next shots require at their start; if no moment matches, pick a moment that "
    "still shows who they are. Among the moments that qualify, prefer the one that shows the most about who they are: "
    "face or full body visible, large, not occluded, not cut off by the frame edge, few other people, and consistent "
    "with how they have looked so far; if two are equally good, take the later one.")
STEP_ONE_INSTRUCTIONS = (
    "For each needed character, pick exactly one moment: a shot number from Shot 1 to Shot {last} and a time in "
    "seconds within that clip. Do not pick the very start or end of a clip, where the shot cuts.\n" + SELECTION_RULE + "\nReply with JSON only, in exactly this form:\n")

INSTRUCTIONS = (
    "1. In each reference image, point to each needed character it shows.{last_frame}\n"
    "2. For each next shot, point to where each present character starts; write null if the character is not "
    "visible at the start of that shot. Keep the screen side the last frame establishes unless the plan moves them.\n"
    "Coordinates are integers 0-1000 relative to width and height. Reply with JSON only, in exactly this form:\n"
)


def describe_state(state: Mapping[str, Any]) -> str:
    parts = [f"wearing {state['outfit']}"] if state["outfit"] else []
    parts.append("holding " + (", ".join(state["holding"]) if state["holding"] else "nothing"))
    return "; ".join(parts)


def target_people(annotation: Mapping[str, Any], span: Sequence[int]) -> dict[int, list[str]]:
    """各目标镜（镜号从 1 起）→ 开镜在场的登记角色，按标注顺序。"""
    registry = {entry["name"] for entry in annotation["characters"]}
    return {index + 1: [e["name"] for e in annotation["shots"][index]["present"] if e["name"] in registry]
            for index in range(*span)}


def sampled_indices(video_processor, frame_count: int) -> list[int]:
    """处理器在 fps 2 下会选中的帧号（16 fps 片段）。"""
    from transformers.video_utils import VideoMetadata

    metadata = VideoMetadata(total_num_frames=frame_count, fps=float(SOURCE_FPS), width=832, height=480,
                             duration=frame_count / SOURCE_FPS)
    return [int(i) for i in video_processor.sample_frames(metadata, fps=VIDEO_FPS)]


def video_inputs(video_processor, clips: Mapping[int, Any]):
    """各历史镜的全部帧（[帧数, H, W, 3] uint8，按镜号）→ fps 2 选出的帧与 ``video_metadata``，按镜号升序。"""
    import numpy as np
    from transformers.video_utils import VideoMetadata

    videos, metadata = [], []
    for shot in sorted(clips):
        frames = np.asarray(clips[shot])
        indices = sampled_indices(video_processor, len(frames))
        videos.append(frames[indices])
        metadata.append(VideoMetadata(total_num_frames=len(frames), fps=float(SOURCE_FPS), width=frames.shape[2],
                                      height=frames.shape[1], duration=len(frames) / SOURCE_FPS,
                                      frames_indices=indices))
    return videos, metadata


def step_one_content(annotation: Mapping[str, Any], *, span: Sequence[int], history_frames: Mapping[int, int],
                     target_prompts: Sequence[str], needed: Sequence[str]) -> list[dict]:
    """第一步请求内容：各历史镜片段（``history_frames``：镜下标 → 帧数）与镜号、时长交替，再给目标镜与所需角色。"""
    start, end = span
    if len(target_prompts) != end - start:
        raise ValueError("one target prompt per target shot")
    registry = {entry["name"]: entry for entry in annotation["characters"]}
    content: list[dict] = [{"type": "text", "text": "Story so far. Each clip is one finished shot, in order."}]
    for index in sorted(history_frames):
        content += [{"type": "text", "text": f"Shot {index + 1} ({history_frames[index] / SOURCE_FPS:.1f} s):"},
                    {"type": "video"}]
    targets = target_people(annotation, span)
    lines = ["Next shots to generate:"]
    for (shot, names), prompt in zip(sorted(targets.items()), target_prompts):
        who = "; ".join(f"{n} ({describe_state(opening_state(annotation, n, shot - 1))})" for n in names) or "none"
        lines += [f"Shot {shot}: {prompt}", f"Present at the start: {who}"]
    lines.append("Characters needed:")
    for name in needed:
        first = next(i for i in range(start, end) if name in targets[i + 1])
        lines.append(f"- {name}: {registry[name]['appearance']}; required at their first next shot: "
                     f"{describe_state(opening_state(annotation, name, first))}")
    last = max(history_frames) + 1
    example = {"picks": [{"name": name, "shot": last, "time": 1.5} for name in needed]}
    content.append({"type": "text", "text": "\n".join(lines) + "\n" + STEP_ONE_INSTRUCTIONS.format(last=last)
                    + json.dumps(example)})
    return content


def step_one_moments(answer: Any, *, needed: Sequence[str], history_frames: Mapping[int, int], cut_margin: int,
                     max_moments: int) -> tuple[list[list[int]], list[dict], list[dict]]:
    """核对第一步作答，返回参考时刻 (镜下标, 帧号)、逐人 picks 与丢弃记录。

    每个所需角色取它的第一条合格 pick（镜号在历史里、秒数在镜内、字段齐）；不合格的 pick、不在名单或重复的人、
    同镜不同帧按实体选择票数最多、同票按原答首次出现选一张，其他候选明确记录；再按原答顺序限制来源镜数。
    多于 ``max_moments`` 的来源镜丢掉并记录，那个人这一段就没有参考。作答不是带 picks 列表的对象时，所有人都没有参考。

    所选帧收进 [cut_margin, 帧数 − cut_margin)，取离所选时刻最近的合法帧——与训练标签同一条有效性约束（切点前后不选），
    ``cut_margin`` 与标签构造器的 ``--cut-margin`` 同值；镜太短没有合法帧时取中帧，picks 里记 ``short_shot``。
    原选帧记在 picks 的 ``chosen_frame``。同一时刻合并。
    baseline 不消费源人物点；不能把落选帧的人物点移到获选帧。后续若启用源绑定，必须对最终实际 RGB 重指。"""
    if type(cut_margin) is not int or cut_margin < 0:
        raise ValueError("cut_margin must be a nonnegative int")
    if type(max_moments) is not int or not 1 <= max_moments <= MAX_REFERENCES:
        raise ValueError(f"max_moments must be an int in 1..{MAX_REFERENCES}")
    dropped: list[dict] = []
    if not isinstance(answer, dict) or set(answer) != {"picks"} or not isinstance(answer["picks"], list):
        dropped.append({"value": answer, "reason": "not an object with exactly picks"})
        answer = {"picks": []}
    rows, moments, originals = [], [], {}
    for p in answer["picks"]:
        reason = None
        if not isinstance(p, dict) or set(p) != {"name", "shot", "time"}:
            reason = "bad pick"
        elif p["name"] not in needed:
            reason = "not needed"
        elif any(r["name"] == p["name"] for r in rows):
            reason = "second pick for the same character"
        else:
            index = p["shot"] - 1 if type(p["shot"]) is int else None
            if index not in history_frames:
                reason = "not a finished shot"
            elif isinstance(p["time"], bool) or not isinstance(p["time"], (int, float)) or not (
                    0 <= p["time"] <= history_frames[index] / SOURCE_FPS):
                reason = "time outside the shot"
        if reason is None:
            count = history_frames[index]
            chosen = min(max(round(p["time"] * SOURCE_FPS), 0), count - 1)
            low, high = cut_margin, count - cut_margin - 1
            frame = min(max(chosen, low), high) if low <= high else count // 2
        if reason is not None:
            dropped.append({"value": p, "reason": reason})
            continue
        if (index, frame) not in moments:
            moments.append((index, frame))
        rows.append({"name": p["name"], "shot_index": index, "frame": frame, "chosen_frame": chosen,
                     "short_shot": low > high, "source_binding_requires_repointing": True})
        originals[p["name"]] = p
    votes = {moment: sum((row["shot_index"], row["frame"]) == moment for row in rows) for moment in moments}
    best_by_shot = {}
    for moment in moments:
        previous = best_by_shot.get(moment[0])
        if previous is None or votes[moment] > votes[previous]:
            best_by_shot[moment[0]] = moment
    selected = [moment for moment in moments if best_by_shot[moment[0]] == moment][:max_moments]
    kept = []
    for row in rows:
        moment = (row["shot_index"], row["frame"])
        if moment != best_by_shot[moment[0]]:
            dropped.append({"value": originals[row["name"]], "reason": "one_reference_frame_per_shot",
                "candidate": row, "selected_frame": best_by_shot[moment[0]][1],
                "entity_votes": votes[moment], "selected_entity_votes": votes[best_by_shot[moment[0]]],
                "source_binding_requires_repointing": True})
        elif moment not in selected:
            dropped.append({"value": originals[row["name"]], "reason": f"more than {max_moments} moments"})
        else:
            kept.append(row)
    rows = kept
    for name in needed:
        if not any(r["name"] == name for r in rows):
            dropped.append({"value": name, "reason": "no usable pick; no reference this segment"})
    return [list(m) for m in sorted(selected)], rows, dropped


def parse_json_answer(raw: str) -> Any:
    """模型原样输出（可能带 ```json 围栏）→ JSON。"""
    body = raw.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(body)


def source_grounding_content(annotation: Mapping[str, Any], references: Sequence[Sequence[int]]) -> list[dict]:
    """第二步的源图读取：只看最终实际参考RGB，不读取选帧前的点或计划在场名单。

    每张参考图后给全部登记角色的辨认描述（外貌加该图所在镜的穿着，``visual_reading.character_phrase``），
    不只给该镜计划在场的人。格式示例不写坐标数值：数值示例会牵引作答。"""
    content = [{"type": "text", "text": "Identify registered characters in these final reference images. "
                "Use only what is visibly present in each actual image."}]
    for number, (shot, frame) in enumerate(references, 1):
        hints = [{"name": entry["name"], "kind": entry["kind"],
                  "description": character_phrase(annotation, entry["name"], shot)}
                 for entry in annotation["characters"]]
        content += [{"type": "text", "text": f"Reference {number} (Shot {shot + 1}, frame {frame}):"},
                    {"type": "image"},
                    {"type": "text", "text": f"Identification hints for Reference {number}: "
                     + json.dumps(hints, ensure_ascii=False)}]
    unknown_example = {"unknown_characters": [entry["name"] for entry in annotation["characters"][:1]]}
    example = {"references": [{"reference": number, "characters": [], "unknown_characters": []}
                              for number in range(1, len(references) + 1)]}
    content.append({"type": "text", "text": "The identification hints after each image list every registered "
        "character with their appearance and the outfit expected in that image's shot; they help tell characters "
        "apart and are not facts about presence, pose, or state. Use the exact name field as the Character ID, "
        "without appending kind, parentheses, descriptions, or any other text. kind is separate metadata. "
        "For each image, list only registered characters you can confidently identify as visible. Give an internal "
        "point inside each person's visible silhouette, not the center of a bounding box. Coordinates are integers "
        "0-1000 relative to the image width and height. If the identity is visible but an internal point is uncertain, "
        "write point_2d:null. A registered group may be listed as visibly present, but must always have "
        "point_2d:null: it is not an individual silhouette. Put a registered identity in unknown_characters when its visible identity cannot be "
        "confirmed. unknown_characters is an array of unique exact registered name strings, never objects. "
        "Do not give points or point_2d for unknown identities. A name cannot appear in both characters and "
        "unknown_characters for the same reference. A non-empty unknown list looks like " + json.dumps(unknown_example) +
        "; this is a schema example only and does not assert presence. Do not guess from the shot number, plan, "
        "expected cast, or another image. Unregistered people "
        "receive no Character ID. An image with no confirmed registered identity has characters:[]. "
        "Return exactly one entry per reference in this order, no target-shot predictions, JSON only. Each visible "
        "entry has exactly name and point_2d. A visible person entry looks like "
        '{"name": "<exact registered name>", "point_2d": [x, y]}, where x and y are the integer coordinates of that '
        "person's internal point, measured from the image's left and top edges on the 0-1000 scale; this illustrates "
        "the schema only and does not assert presence. Exact complete structure:\n" + json.dumps(example)})
    return content


def source_grounding_schema(annotation: Mapping[str, Any], references: Sequence[Sequence[int]]) -> dict:
    """源图读取的生成schema；供HF约束解码，最终仍由严格消费者核对。

    LM Format Enforcer 0.11.3不消费整数minimum/maximum，故坐标用有限整数enum。
    不依赖该版本未强制的uniqueItems或位置数组：编号顺序、身份唯一和两列表互斥仍由消费者核对。
    """
    names = [entry["name"] for entry in annotation["characters"]]
    if not names or not references:
        raise ValueError("source grounding schema requires registered identities and actual references")
    people = [entry["name"] for entry in annotation["characters"] if entry["kind"] != "group"]
    groups = [entry["name"] for entry in annotation["characters"] if entry["kind"] == "group"]
    point = {"anyOf": [{"type": "null"}, {"type": "array", "minItems": 2, "maxItems": 2,
               "items": {"enum": list(range(COORDINATE_MAX + 1))}}]}

    def visible_entry(allowed_names, point_schema):
        return {"type": "object", "properties": {"name": {"type": "string", "enum": allowed_names},
                "point_2d": point_schema}, "required": ["name", "point_2d"], "additionalProperties": False}

    entries = ([visible_entry(people, point)] if people else []) + (
        [visible_entry(groups, {"type": "null"})] if groups else [])
    item = {"anyOf": entries}
    row = {"type": "object", "properties": {
        "reference": {"enum": list(range(1, len(references) + 1))},
        "characters": {"type": "array", "items": item, "maxItems": len(names)},
        "unknown_characters": {"type": "array", "items": {"type": "string", "enum": names},
                               "maxItems": len(names)}},
        "required": ["reference", "characters", "unknown_characters"], "additionalProperties": False}
    return {"type": "object", "properties": {"references": {"type": "array", "items": row,
            "minItems": len(references), "maxItems": len(references)}},
            "required": ["references"], "additionalProperties": False}


def grounded_reference_entities(answer: Any, *, annotation: Mapping[str, Any],
                                references: Sequence[Sequence[int]]) -> list[dict]:
    """严格消费最终RGB作答；显式unknown保图无点，格式/编号/未登记名立即失败。

    同一张图里同一名字给了不止一个人，或同时列为可见与 unknown，说明这张图里认不准此人：记进
    ``ambiguous_characters`` 并并入 ``unknown_characters``，同样保图无点。"""
    from mspc.models.planner_tokens import point_cell
    moments = [tuple(moment) for moment in references]
    if (len(moments) > MAX_REFERENCES or moments != sorted(moments)
            or len({shot for shot, _ in moments}) != len(moments)
            or any(type(shot) is not int or shot < 0 or type(frame) is not int or frame < 0
                   for shot, frame in moments)):
        raise ValueError("source grounding requires final chronological one-frame-per-shot references")
    if not isinstance(answer, dict) or set(answer) != {"references"} or not isinstance(answer["references"], list):
        raise ValueError("source grounding requires exactly a references list")
    rows = answer["references"]
    if len(rows) != len(moments):
        raise ValueError("source grounding answer must identify every actual reference row")
    registry = {entry["name"]: entry for entry in annotation["characters"]}
    output = []
    for number, ((shot, frame), row) in enumerate(zip(moments, rows), 1):
        if (not isinstance(row, dict) or set(row) != {"reference", "characters", "unknown_characters"}
                or type(row["reference"]) is not int or row["reference"] != number
                or not isinstance(row["characters"], list) or not isinstance(row["unknown_characters"], list)):
            raise ValueError("source grounding row must match its exact final RGB reference number")
        unknown = row["unknown_characters"]
        if any(not isinstance(name, str) or name not in registry for name in unknown) or len(set(unknown)) != len(unknown):
            raise ValueError("unknown source identities must be unique registered names")
        listed = []
        for entity in row["characters"]:
            if not isinstance(entity, dict) or set(entity) != {"name", "point_2d"}:
                raise ValueError("visible source identity requires exactly name and point_2d")
            name, point = entity["name"], entity["point_2d"]
            if not isinstance(name, str) or name not in registry:
                raise ValueError("visible source identity must be a registered name")
            if point is not None and (not isinstance(point, list) or len(point) != 2
                                     or any(type(v) is not int or not 0 <= v <= 1000 for v in point)):
                raise ValueError("source internal point must be null or two integers in [0,1000]")
            listed.append((name, point))
        names = [name for name, _ in listed]
        ambiguous = [name for name in registry if names.count(name) > 1 or (name in names and name in unknown)]
        visible = {name: point for name, point in listed if name not in ambiguous}
        unknown = unknown + [name for name in ambiguous if name not in unknown]
        # 名字顺序与训练相同，字典仅用于辨认；最终renderer caption只消费这些已确认名字。
        entities = []
        for name, registered in registry.items():
            if name not in visible:
                continue
            kind = registered["kind"]
            point = None if kind == "group" else visible[name]
            entities.append({"name": name, "kind": kind, "appearance": registered["appearance"],
                "identity_status": "registered", "point_2d": point,
                "point_cell": None if point is None else list(point_cell(point)),
                "point_status": "group_entity" if kind == "group" else
                                "no_unambiguous_cell" if point is None else "visible_internal"})
        output.append({"shot": shot + 1, "frame": frame, "source_frame": frame,
                       "status": "observed" if entities else "no_confirmed_character", "entities": entities,
                       "alignment_cells": None, "unknown_characters": unknown,
                       "ambiguous_characters": ambiguous})
    return output


def step_two_content(annotation: Mapping[str, Any], *, span: Sequence[int], references: Sequence[Sequence[int]],
                     last_shot: int | None, target_prompts: Sequence[str], needed: Sequence[str]) -> list[dict]:
    """请求内容（文字与图片占位交替）。``references`` 是 (镜下标, 帧号)，按参考序号；``last_shot`` 是最近一个历史镜的
    镜下标（没有历史时为 None）。图片顺序：各参考，然后末帧。"""
    start, end = span
    if len(target_prompts) != end - start:
        raise ValueError("one target prompt per target shot")
    registry = {entry["name"]: entry for entry in annotation["characters"]}
    content: list[dict] = []
    if references:
        content.append({"type": "text", "text": "Reference images picked from the finished shots:"})
    for number, (shot, frame) in enumerate(references, start=1):
        content += [{"type": "text", "text": f"Reference {number} (Shot {shot + 1}, {frame / SOURCE_FPS:.1f} s):"},
                    {"type": "image"}]
    if last_shot is not None:
        content += [{"type": "text", "text": f"Last frame of Shot {last_shot + 1}:"}, {"type": "image"}]
    targets = target_people(annotation, span)
    lines = ["Next shots to generate:"]
    for (shot, names), prompt in zip(sorted(targets.items()), target_prompts):
        who = "; ".join(f"{n} ({describe_state(opening_state(annotation, n, shot - 1))})" for n in names) or "none"
        lines += [f"Shot {shot}: {prompt}", f"Present at the start: {who}"]
    lines.append("Characters needed:")
    lines += [f"- {name}: {registry[name]['appearance']}" for name in needed]
    if references:
        lines.append(f"There are {len(references)} reference images, Reference 1 to Reference {len(references)}.")
    else:
        lines.append("There are no reference images; give an empty references list.")
    last_frame = (" The last frame of the finished shots is not a reference: do not list it in references."
                  if last_shot is not None else "")
    example = {"references": [{"reference": number, "characters": [{"name": n, "point_2d": [412, 230]}
                                                                   for n in needed[:1]]}
                              for number in range(1, len(references) + 1)],
               "shots": [{"shot": shot, "characters": [{"name": n, "point_2d": [300, 520]} for n in names]}
                         for shot, names in sorted(targets.items())]}
    content.append({"type": "text", "text": "\n".join(lines) + "\n" + INSTRUCTIONS.format(last_frame=last_frame)
                    + json.dumps(example)})
    return content


def answer_from_labels(labels: Mapping[str, Any]) -> str:
    """读历史标签 → 第二步作答（参考按标签时刻顺序编号，镜号从 1 起）。"""
    moments = [tuple(m) for m in labels["moments"]]
    references = []
    for number, moment in enumerate(moments, start=1):
        people = [{"name": p["name"], "point_2d": list(p["point_2d"])} for p in labels["picks"]
                  if (p["shot_index"], p["frame"]) == moment]
        references.append({"reference": number, "characters": people})
    shots = [{"shot": start["shot_index"] + 1,
              "characters": [{"name": c["name"], "point_2d": c["point_2d"]} for c in start["characters"]]}
             for start in labels["starts"]]
    return json.dumps({"references": references, "shots": shots})


def validate_answer(answer: Any, *, references: int, needed: Sequence[str],
                    targets: Mapping[int, Sequence[str]]) -> tuple[dict, list[dict]]:
    """逐字段核对，不合格的字段丢掉并记录，返回 (可用作答, 丢弃记录)；出片照常走下去。

    - 参考：编号不是 1..references 按序时，指点对不到图上，整段参考指点丢掉（每张参考都没有人）；编号对时，每张参考里
      不在所需名单的名字、重复的名字（同名全部丢掉，分不出哪个对）、坏条目与坏点逐条丢掉。
    - 镜首：按目标镜与在场名单重排；某镜某人没有、重复或点不合格时，此人这一镜记 null（没有本镜 token）。
    整体不是带 references 与 shots 的对象时，两部分都按缺失处理（参考没有人、镜首全是 null）。"""
    dropped: list[dict] = []
    if not isinstance(answer, dict) or set(answer) != {"references", "shots"}:
        dropped.append({"where": "answer", "value": answer, "reason": "not an object with exactly references and shots"})
        answer = {"references": answer.get("references") if isinstance(answer, dict) else None,
                  "shots": answer.get("shots") if isinstance(answer, dict) else None}

    def good_point(value, allow_null):
        if value is None:
            return allow_null
        return (isinstance(value, list) and len(value) == 2
                and all(type(v) is int and 0 <= v <= COORDINATE_MAX for v in value))

    rows = answer["references"]
    if not isinstance(rows, list) or [r.get("reference") if isinstance(r, dict) else None for r in rows] != list(
            range(1, references + 1)):
        dropped.append({"where": "references", "value": rows, "reason": f"not numbered 1..{references} in order"})
        rows = [{"reference": number, "characters": []} for number in range(1, references + 1)]
    kept_references = []
    for row in rows:
        people = row.get("characters") if isinstance(row.get("characters"), list) else []
        if set(row) != {"reference", "characters"} or not isinstance(row["characters"], list):
            dropped.append({"where": f"reference {row['reference']}", "value": row, "reason": "bad reference entry"})
        names = [c.get("name") if isinstance(c, dict) else None for c in people]
        kept = []
        for c, name in zip(people, names):
            reason = ("bad character entry" if not isinstance(c, dict) or set(c) != {"name", "point_2d"}
                      else "not needed" if name not in needed
                      else "named more than once" if names.count(name) > 1
                      else "bad point" if not good_point(c["point_2d"], allow_null=False) else None)
            if reason:
                dropped.append({"where": f"reference {row['reference']}", "value": c, "reason": reason})
            else:
                kept.append({"name": name, "point_2d": c["point_2d"]})
        kept_references.append({"reference": row["reference"], "characters": kept})
    given = {}
    for row in answer["shots"] if isinstance(answer["shots"], list) else []:
        if isinstance(row, dict) and isinstance(row.get("characters"), list) and row.get("shot") in targets:
            given.setdefault(row["shot"], []).extend(row["characters"])
        else:
            dropped.append({"where": "shots", "value": row, "reason": "not a target shot entry"})
    kept_shots = []
    for shot, people in sorted(targets.items()):
        entries = [c for c in given.get(shot, []) if isinstance(c, dict)]
        characters = []
        for name in people:
            matches = [c for c in entries if c.get("name") == name]
            ok = len(matches) == 1 and set(matches[0]) == {"name", "point_2d"} and good_point(
                matches[0]["point_2d"], allow_null=True)
            if not ok:
                dropped.append({"where": f"shot {shot}", "value": matches, "name": name,
                                "reason": "missing, repeated or bad start entry; written as null"})
            characters.append({"name": name, "point_2d": matches[0]["point_2d"] if ok else None})
        for c in entries:
            if c.get("name") not in people:
                dropped.append({"where": f"shot {shot}", "value": c, "reason": "not present at the shot start"})
        kept_shots.append({"shot": shot, "characters": characters})
    return {"references": kept_references, "shots": kept_shots}, dropped


PLAN_TOKEN_INSTRUCTIONS = (
    "For each next shot, write where each present character starts, as the plan places them; write null if the "
    "character is not visible at the start of that shot. Coordinates are integers 0-1000 relative to width and "
    "height. Reply with JSON only, in exactly this form:\n"
)


def _opening_boxes(annotation: Mapping[str, Any]) -> dict[tuple[int, str], list[float]]:
    """目标点与宽高共用同一实际镜首锚，不借后帧补答案。"""
    boxes: dict[tuple[int, str], list[float]] = {}
    for layout in annotation.get("layouts", []):
        for entity in layout["entities"]:
            for anchor in entity["anchors"]:
                if anchor["time"] == 0 and anchor["visibility"] == "visible" and anchor["box"] is not None:
                    boxes[(layout["shot"] - 1, entity["name"])] = anchor["box"]
                    break
    return boxes


def plan_starts(annotation: Mapping[str, Any], span: Sequence[int]) -> list[tuple[int, str, list[int] | None]]:
    """计划 → 各目标镜登记角色的首帧点 (镜下标, 名字, 点或 None)。

    仅 time=0、visible 且有框的锚取中心（``planner_tokens.box_point``）；后续可见锚不提前到首帧。
    首帧无合格锚记 None，与训练只在 RGB0 取点的时间合同一致；完整事件与布局仍由文字编译器消费。
    """
    from mspc.models.planner_tokens import box_point

    boxes = _opening_boxes(annotation)
    return [(shot - 1, name, None if (shot - 1, name) not in boxes else box_point(boxes[(shot - 1, name)]))
            for shot, names in sorted(target_people(annotation, span).items()) for name in names]


def plan_geometry(annotation: Mapping[str, Any], span: Sequence[int]) -> list[dict]:
    """同一镜首框与离散 ``front_of`` 关系 → 与训练相同的几何行；未知保留为空。"""
    starts, boxes = plan_starts(annotation, span), _opening_boxes(annotation)
    kinds = {entity["name"]: entity["kind"] for entity in annotation["characters"]}
    layouts = {layout["shot"] - 1: layout for layout in annotation.get("layouts", [])}
    rows = []
    for shot in range(*span):
        characters = []
        for index, name, point in starts:
            if index != shot:
                continue
            box = boxes.get((shot, name))
            characters.append({"name": name, "point_2d": point,
                               "size": None if box is None else [box[2] - box[0], box[3] - box[1]],
                               "size_valid": box is not None})
        known = {entity["name"] for entity in characters
                 if entity["size_valid"] and kinds[entity["name"]] != "group"}
        pairs = [{"near": pair["near"], "far": pair["far"]}
                 for pair in layouts.get(shot, {}).get("front_of", [])
                 if pair["near"] in known and pair["far"] in known]
        rows.append({"shot_index": shot, "frame": 0, "characters": characters, "front_of": pairs})
    return rows


def answer_from_starts(starts: Sequence[tuple[int, str, Sequence[int] | None]]) -> str:
    """(镜下标, 名字, 点或 None) → 只有镜首段的作答（镜号从 1 起），紧凑写法，训练与推理同一种文字形式。"""
    shots: dict[int, list[dict]] = {}
    for index, name, point in starts:
        shots.setdefault(index + 1, []).append({"name": name, "point_2d": None if point is None else list(point)})
    return json.dumps({"shots": [{"shot": shot, "characters": people} for shot, people in sorted(shots.items())]})


def plan_token_content(annotation: Mapping[str, Any], *, span: Sequence[int], references: Sequence[Sequence[int]],
                       last_shot: int | None, target_prompts: Sequence[str]) -> list[dict]:
    """点控制加特征注入的预填提示（文字与图片占位交替）。``references`` 是这一段渲染器拿到的参考 (镜下标, 帧号)；
    ``last_shot`` 是段前最近一镜（没有历史时 None）。图片顺序：各参考，然后末帧。``target_prompts`` 不带方位句。"""
    start, end = span
    if len(target_prompts) != end - start:
        raise ValueError("one target prompt per target shot")
    content: list[dict] = []
    if references:
        content.append({"type": "text", "text": "Reference images from the finished shots:"})
    for number, (shot, frame) in enumerate(references, start=1):
        content += [{"type": "text", "text": f"Reference {number} (Shot {shot + 1}, {frame / SOURCE_FPS:.1f} s):"},
                    {"type": "image"}]
    if last_shot is not None:
        content += [{"type": "text", "text": f"Last frame of Shot {last_shot + 1}:"}, {"type": "image"}]
    targets = target_people(annotation, span)
    registry = {entry["name"]: entry for entry in annotation["characters"]}
    lines = ["Next shots to generate:"]
    for (shot, names), prompt in zip(sorted(targets.items()), target_prompts):
        who = "; ".join(f"{n} ({describe_state(opening_state(annotation, n, shot - 1))})" for n in names) or "none"
        lines += [f"Shot {shot}: {prompt}", f"Present at the start: {who}"]
    present = sorted({n for names in targets.values() for n in names})
    lines.append("Characters:")
    lines += [f"- {name}: {registry[name]['appearance']}" for name in present]
    example = {"shots": [{"shot": shot, "characters": [{"name": n, "point_2d": [300, 520]} for n in names]}
                         for shot, names in sorted(targets.items())]}
    content.append({"type": "text", "text": "\n".join(lines) + "\n" + PLAN_TOKEN_INSTRUCTIONS + json.dumps(example)})
    return content
