"""B事件干预的给定计划诊断：冻结剧本及状态 → 同一v18分镜表 → renderer请求目录。

只用于scope=renderer；三档system请求走generation的公共/SFT规划入口。
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from mspc.data.storyboard import validate_storyboard

SUBJECT = re.compile(r"Subject [1-9]\d*")
ARTICLE = re.compile(r"(?:an?|the|two|three|four|five|six|several|some|many|\d+) ", re.IGNORECASE)
CHARACTER_KINDS = ("person", "animal")


def noun_phrase(attributes: str) -> str:
    """剧本的一句描述 → 能接在句子里的名词短语：去句号、首字母小写、缺冠词补 a／an。"""
    phrase = attributes.strip().rstrip(".")
    if not phrase:
        raise ValueError("empty attributes")
    phrase = phrase[0].lower() + phrase[1:]
    if ARTICLE.match(phrase):
        return phrase
    return ("an " if phrase[0] in "aeiou" else "a ") + phrase


def subject_names(script: Mapping[str, Any]) -> dict[str, str]:
    """``Subject N`` → ``Character k``（人与动物）或 ``Object k``（物件），各按剧本里的出现顺序编号。"""
    names: dict[str, str] = {}
    counts = {"Character": 0, "Object": 0}
    for subject in script["subjects"]:
        family = "Character" if subject["kind"] in CHARACTER_KINDS else "Object"
        if subject["kind"] not in (*CHARACTER_KINDS, "object"):
            raise ValueError(f"{subject['subject_id']}: unknown kind {subject['kind']!r}")
        counts[family] += 1
        names[subject["subject_id"]] = f"{family} {counts[family]}"
    if len(names) != len(script["subjects"]):
        raise ValueError("duplicate subject_id in the script")
    return names


def annotation_from_script(script: Mapping[str, Any]) -> dict[str, Any]:
    """评测剧本 → 标注结构，不带状态：事件只出现在发生那一镜的原句里。"""
    names = subject_names(script)

    def rename(text: str) -> str:
        return SUBJECT.sub(lambda match: names[match.group()], text)

    registry: dict[str, list[dict[str, Any]]] = {"characters": [], "objects": []}
    for subject in script["subjects"]:
        entry = {"name": names[subject["subject_id"]], "appearance": noun_phrase(rename(subject["attributes"]))}
        if subject["kind"] in CHARACTER_KINDS:
            registry["characters"].append({**entry, "kind": subject["kind"]})
        else:
            registry["objects"].append(entry)
    places = [{"name": place["location_id"], "appearance": noun_phrase(place["attributes"])} for place in script["locations"]]
    known = {place["name"] for place in places}
    shots = []
    for index, shot in enumerate(script["shots"], 1):
        if shot["location_id"] not in known:
            raise ValueError(f"{shot['shot_id']}: location {shot['location_id']!r} is not in the script's locations")
        shots.append({"shot": index, "place": shot["location_id"], "camera": {},
                      "action": rename(shot["prompt"]),
                      "present": [{"name": names[subject]} for subject in shot["present_subjects"]], "changes": []})
    return {**registry, "places": places, "shots": shots}


def with_planned_state(annotation: Mapping[str, Any], script: Mapping[str, Any], change: Mapping[str, Any]) -> dict[str, Any]:
    """把改写记录的状态带进每一镜：事件镜之前 ``prior``，事件镜里由 ``prior`` 变到 ``post``，之后 ``post``。"""
    name = subject_names(script)[change["subject_id"]]
    event_shot = change["event_shot"]
    planned = copy.deepcopy(dict(annotation))
    if type(event_shot) is not int or not 1 <= event_shot <= len(planned["shots"]):
        raise ValueError(f"event_shot {event_shot!r} outside 1..{len(planned['shots'])}")
    for shot in planned["shots"]:
        entry = next((item for item in shot["present"] if item["name"] == name), None)
        if entry is None:
            if shot["shot"] == event_shot:
                raise ValueError(f"shot {event_shot}: the changing subject {name} is not present in the event shot")
            continue
        entry["state"] = change["prior"] if shot["shot"] <= event_shot else change["post"]
        if shot["shot"] == event_shot:
            shot["changes"] = [{"name": name, "what": "state", "before": change["prior"], "after": change["post"]}]
    return planned


def frame_counts(candidate: Mapping[str, Any], story_id: str) -> list[int]:
    """冻结候选清单里某故事的逐镜渲染帧数（16 fps）。"""
    if candidate.get("render_revision", {}).get("render_fps") != 16:
        raise ValueError("candidate manifest does not carry 16 fps render frames")
    stories = [story for story in candidate["stories"] if story["story_id"] == story_id]
    if len(stories) != 1:
        raise ValueError(f"{story_id}: expected one candidate story, got {len(stories)}")
    return [shot["render_frames"] for shot in stories[0]["shots"]]


def convert_pairs(pairs: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A/B 配对清单的每个 B 剧本 → 一条出片用例：``case_id``、镜号、帧数与标注结构。"""
    cases = []
    for pair in pairs["pairs"]:
        script = pair["context_b"]
        counts = frame_counts(candidate, pair["story_id"])
        if len(counts) != len(script["shots"]):
            raise ValueError(f"{pair['story_id']}: {len(script['shots'])} script shots but {len(counts)} frozen frame counts")
        annotation = annotation_from_script(script)
        annotation = with_planned_state(annotation, script, pair["change"])
        for shot, count in zip(annotation["shots"], counts, strict=True):
            shot["frame_count"] = count
        annotation = validate_storyboard(annotation)
        cases.append({"case_id": f"{pair['story_id']}-b", "story_id": pair["story_id"],
                      "shot_ids": [shot["shot_id"] for shot in script["shots"]],
                      "annotation": annotation})
    return cases


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, required=True, help="A/B 配对清单（dev 30 或 test 108）")
    parser.add_argument("--candidate", type=Path, required=True, help="冻结候选清单（逐镜 render_frames，16 fps）")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cases = convert_pairs(json.loads(args.pairs.read_text()), json.loads(args.candidate.read_text()))
    if args.output.exists():
        raise FileExistsError(args.output)
    from mspc.data.shot_text import shot_text
    from mspc.evaluation.benchmark import write_json
    from mspc.benchmark.generation import load_requests
    from itertools import accumulate
    stories = []
    for case in cases:
        board = case['annotation']
        counts = [shot['frame_count'] for shot in board['shots']]
        stories.append(dict(story_id=case['case_id'], split=args.split, fps=16, target_seconds=sum(counts)/16,
            storyboard=board, inputs={'shots': dict(text='\n\n'.join(
                f'Shot {i+1}: {shot_text(board, i)}' for i in range(len(counts))), expected_shot_count=len(counts),
                expected_cut_times_seconds=[n/16 for n in accumulate(counts[:-1])])}))
    write_json(args.output, {'evaluation_scope': 'renderer', 'stories': stories})
    load_requests(args.output)
    print(json.dumps({"cases": len(cases), "shots": sum(len(case["shot_ids"]) for case in cases), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
