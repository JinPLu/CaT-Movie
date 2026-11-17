"""长集评测剧本 → event_log_v3：确定性转换，不调模型、不补写。

一条评测剧本（A/B 配对的 ``context_b``）加它的改写记录（``change``）转成一份事件日志：

- 主体、地点照抄（``attributes`` → ``appearance``，地点无锚点）；主体描述里的状态性短语（event_log 规则 5
  拒收的词，如 standing / seated / holding）由 :func:`split_appearance` 拆出外观，转成该主体首个出场镜的
  ``body`` 轴初始状态；外观只留外观；
- 每镜 ``action`` 用剧本逐镜原句（B 的改写句已在其中）；地点与在场主体照抄剧本镜级字段
  ``location_id`` / ``present_subjects``，可见实体同在场主体；机位三项 unknown，无 screen 事实；
- 帧数用修订 3 冻结候选清单（16 fps）对应镜的 ``render_frames``；
- 初始状态与唯一事件从改写记录转：``prior`` 是全片初值，``post`` 是 ``event_shot`` 镜内演出的结果。

状态取值规则（:func:`change_values`）：任一端以 ``holding`` 开头 → ``body`` 轴、两端原文；恰有一端是
``wearing X`` → ``costume`` 轴，``wearing X`` 取 ``X``，``not wearing Y`` 取 ``no Y``，其余写法（如 bareheaded、
X removed）取 ``no`` + X 的末词；其余 → ``costume`` 轴、两端原文。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from mspc.data.event_log import (
    EVENT_LOG_SCHEMA, MUTABLE_STATE_WORDS, UNKNOWN, parse_event_log, validate_event_log,
)


#: 只作定语的姿态词（“Standing woman …”“Older seated woman …”）：出现在描述前两个词、后面不是介词/副词。
_POSTURES = frozenset({"sitting", "seated", "standing", "walking", "running", "lying"})
_PREDICATE_FOLLOWERS = frozenset({"on", "in", "at", "behind", "beside", "near", "by", "slightly", "with", "a", "the"})


def split_appearance(attributes: str) -> tuple[str, list[str]]:
    """主体描述 → (只含外观的描述, 状态短语列表)。按逗号分句：

    - 定语姿态词（描述前两个词之一、后接名词）：删掉这个词，状态为该词本身；
    - 其余命中：从命中词到分句末尾是状态短语，分句前半留作外观（为空则整句去掉）。
    """
    tokens = attributes.split()
    clauses, states = [], []
    for clause in attributes.rstrip(".").split(", "):
        match = MUTABLE_STATE_WORDS.search(clause)
        if match is None:
            clauses.append(clause)
            continue
        word = match.group(0)
        position = next(i for i, token in enumerate(tokens) if token.strip(",.") == word)
        following = tokens[position + 1].strip(",.").lower() if position + 1 < len(tokens) else ""
        if word.lower() in _POSTURES and position <= 1 and following not in _PREDICATE_FOLLOWERS:
            states.append(word.lower())
            clauses.append(" ".join(t for i, t in enumerate(clause.split()) if t != word))
        else:
            states.append(clause[match.start():].strip())
            if clause[:match.start()].strip():
                clauses.append(clause[:match.start()].strip())
    appearance = ", ".join(clauses)
    if MUTABLE_STATE_WORDS.search(appearance):
        raise ValueError(f"appearance still holds a state word after splitting: {appearance!r}")
    return appearance[:1].upper() + appearance[1:] + ".", [state[:1].lower() + state[1:] for state in states]


def change_values(change: Mapping[str, Any]) -> tuple[str, str, str]:
    """改写记录的 (轴, 初值, 结果值)。"""
    prior, post = change["prior"], change["post"]
    if prior.startswith("holding ") or post.startswith("holding "):
        return "body", prior, post
    worn = [side.removeprefix("wearing ") for side in (prior, post) if side.startswith("wearing ")]
    if len(worn) != 1:
        return "costume", prior, post
    head = worn[0].split()[-1]

    def value(side: str) -> str:
        if side.startswith("wearing "):
            return side.removeprefix("wearing ")
        if side.startswith("not wearing "):
            return "no " + side.removeprefix("not wearing ")
        return "no " + head

    return "costume", value(prior), value(post)


def r3_frame_counts(candidate_r3: Mapping[str, Any], story_id: str) -> list[int]:
    """修订 3 冻结候选清单里某故事的逐镜渲染帧数（16 fps）。"""
    if candidate_r3.get("render_revision", {}).get("revision") != 3:
        raise ValueError("candidate manifest is not render revision 3")
    stories = [story for story in candidate_r3["stories"] if story["story_id"] == story_id]
    if len(stories) != 1:
        raise ValueError(f"{story_id}: expected one candidate story, got {len(stories)}")
    return [shot["render_frames"] for shot in stories[0]["shots"]]


def event_log_from_script(script: Mapping[str, Any], change: Mapping[str, Any],
                          frame_counts: Sequence[int]) -> dict[str, Any]:
    """评测剧本＋改写记录＋冻结帧数 → 校验过的 event_log_v3 原始映射。"""
    shots = script["shots"]
    if len(frame_counts) != len(shots):
        raise ValueError(f"{len(shots)} script shots but {len(frame_counts)} frozen frame counts")
    event_shot = change["event_shot"]
    if type(event_shot) is not int or not 1 <= event_shot <= len(shots):
        raise ValueError(f"event_shot {event_shot!r} outside 1..{len(shots)}")
    axis, prior, post = change_values(change)
    subjects, postures = [], []
    for subject in script["subjects"]:
        appearance, states = (split_appearance(subject["attributes"]) if MUTABLE_STATE_WORDS.search(subject["attributes"])
                              else (subject["attributes"], []))
        subjects.append({"subject_id": subject["subject_id"], "kind": subject["kind"], "appearance": appearance})
        if not states:
            continue
        state = ", ".join(states)
        if subject["subject_id"] == change["subject_id"] and axis == "body":
            # 同一主体同一 body 轴：姿态并进改写记录的两端，不另立同键事实。
            prior, post = f"{state}, {prior}", f"{state}, {post}"
            continue
        first = next(i for i, shot in enumerate(shots) if subject["subject_id"] in shot["present_subjects"])
        postures.append({"subject": subject["subject_id"], "family": "state", "predicate_or_axis": "body",
                         "object_or_value": state, "scope": "scene", "shot_span": [first, first], "evidence": "observed"})

    def fact(value: str) -> dict[str, Any]:
        return {"subject": change["subject_id"], "family": "state", "predicate_or_axis": axis,
                "object_or_value": value, "scope": "scene", "shot_span": [0, len(shots) - 1], "evidence": "observed"}

    raw = {
        "schema": EVENT_LOG_SCHEMA,
        "subjects": subjects,
        "locations": [{"location_id": l["location_id"], "appearance": l["attributes"], "anchors": []}
                      for l in script["locations"]],
        "initial_state": [fact(prior), *postures],
        "shots": [{"shot_id": s["shot_id"], "location_id": s["location_id"],
                   "present_subjects": list(s["present_subjects"]), "visible_entities": list(s["present_subjects"]),
                   "screen_facts": [], "frame_count": count, "shot_size": UNKNOWN, "camera_angle": UNKNOWN,
                   "movement": UNKNOWN, "action": s["prompt"]} for s, count in zip(shots, frame_counts)],
        "events": [{"event_id": "e1", "shot_id": shots[event_shot - 1]["shot_id"], "before": fact(prior),
                    "after": fact(post), "on_screen": True}],
    }
    validate_event_log(parse_event_log(raw))
    return raw


def convert_pairs(pairs: Mapping[str, Any], candidate_r3: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A/B 配对清单的每个 B 剧本 → ``{case_id: <story>-b, story_id, event_log}``。"""
    return [{"case_id": f"{pair['story_id']}-b", "story_id": pair["story_id"],
             "event_log": event_log_from_script(pair["context_b"], pair["change"],
                                                r3_frame_counts(candidate_r3, pair["story_id"]))}
            for pair in pairs["pairs"]]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, required=True, help="A/B 配对清单（dev 30 或 test 108）")
    parser.add_argument("--candidate-r3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cases = convert_pairs(json.loads(args.pairs.read_text()), json.loads(args.candidate_r3.read_text()))
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2))
    print(json.dumps({"cases": len(cases), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
