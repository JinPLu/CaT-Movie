#!/usr/bin/env python3
"""规划器输出形态对照里的「直接写文字」一行：同一公共 Qwen3.5-9B、同一请求与制作要求、同一采样，模型不填分镜结构，
直接写渲染器读的文字——一段全局描述加逐镜描述，人物用固定称呼。这是 HoloCine（Gemini 写分层提示）、MultiShotMaster
（Gemini 写全局与逐镜字幕）、StoryMem（GPT-5 写逐镜提示）、Wan（推理前改写提示词）的共同做法。

动机：很多渲染器工作在仓库里放一个 LLM 提示词扩写器，规划器的一般输出就是渲染器直接读的文字、不输出 JSON；
我们的分镜表叙述由 JSON 加代码共同组成。内容要求与主线提示词
（``mspc.data.planner_dataset.PLANNER_SYSTEM``，v17）逐条对应，只把输出从分镜结构换成文字；有真值的结构错误
（逐镜档镜数不符、镜数装不下总时长）与主线一样回灌一次，内容不重问。

只调用已起好的 OpenAI 兼容服务（``--base-url``），不启服务、不选卡。

用法::

    PYTHONPATH=src .venv/bin/python tools/run/planner_prose_baseline.py --requests configs/benchmark/story_requests_v4.json \\
        --split dev --base-url http://127.0.0.1:8185/v1 --model Qwen3.5-9B --seeds 42 43 \\
        --output outputs/experiments/planner-storyboard-form-20261113/records/prose-9b --workers 64
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time
import urllib.request

from mspc.benchmark.generation import production_request
from mspc.core.wan22_contract import STORYBOARD_FPS, STORYBOARD_MAX_SHOT_FRAMES
from mspc.data.planner_dataset import planner_request

TIERS = {"synopsis": "summary", "paragraph": "paragraph", "shots": "shot"}
# 与 configs/infer/benchmark.json 的 sampling 相同（主线计划用的采样）。
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0, "presence_penalty": 1.5,
            "repetition_penalty": 1.0, "max_tokens": 32768, "enable_thinking": False}
REQUEST_SHOT = re.compile(r"(?m)^s(\d+): ")
GLOBAL_LINE = re.compile(r"^Global:\s*(.*)$")
SHOT_LINE = re.compile(r"^Shot (\d+)(?:\s*\(([\d.]+)\s*s(?:econds?)?\))?:\s*(.*)$")

PROSE_SYSTEM = r'''You are the story planner of a multi-shot video system. Turn the supplied tier and script into prompts for a Wan video generator: one global caption for the whole scene, then one caption per shot. Write concise English.

### Steps
1. Name every character (including groups and animals), object, and place the request mentions as Character N, Object N, Location N. Keep supplied IDs; otherwise number each category from 1.
2. Lay out the shots. Read the frame rate and maximum shot duration from Production requirements. For tier=shot, keep every given shot in its order; do not merge, add, or remove shots. For tier=summary or paragraph, use at least the total seconds divided by the maximum shot duration, rounded up, and enough shots to show every requested event.
3. For each shot, write the opening, the event with its visible progression, and the result at its end, so that the generator can render the shot from the global caption and that shot's caption alone.

### Three kinds of values
- Stated by the request: copy registered appearance and outfit, places, camera instructions, and visual medium faithfully. Preserve every requested event, its performer, order, and required result. The action wording may change, but do not weaken, omit, or replace a requested event.
- Implied by the request or by earlier shots (who is present, initial visible posture and state, possession, end-of-shot changes): write every one, consistent with earlier shots.
- Neither stated nor implied (details a coarse request leaves open, composition, positions, pacing): choose one concrete, plausible value that fits the request, the camera, and the action, and keep it the same across shots of the same setup. Never add plot events or subjects the request does not ask for, and do not write mood or atmosphere prose.

### Global caption
One paragraph naming every recurring character with its stable look and default outfit (copy the request's words when it describes them), every object, and every place, and the visual medium if the request states it.

### Shot captions
For each shot, in this order: the place; the camera (shot size, angle, movement) and the composition; every present character and object, each with a short distinguishing feature, its current outfit, posture, visible state, what it holds, and its position relative to the others; the action as concise ordered sentences that start with the performer; and what is visibly different by the end of the shot.

### Writing rules
1. Use only the registered names (Character N, Object N, Location N) for people, objects, and places.
2. The opening never contains the shot's result: a person who stands up starts seated; a cake being cut starts uncut.
3. Write each character's current outfit in every shot where it appears. After an outfit change, write the new outfit in every later shot; after a transfer, the giver no longer holds the object and the recipient keeps it until another release.
4. Describe a result by what is visible ("empty-handed", "bareheaded"), not by what is gone.
5. Do not replay a finished event to restate its result; carry the result into later shots.
6. Expand only what helps execute the requested action. Do not invent movements for a static shot. Match the detail to the shot's duration.

### Output format
Global: <global caption>
Shot 1 (<seconds> s): <caption>
Shot 2 (<seconds> s): <caption>
One line per shot, numbered from 1 in order; seconds is the proposed duration of that shot. Return only these lines.

### Example
Input:
{"tier":"shot","script":"Production requirements: approximately 12 seconds total; 16 fps; at most 225 frames per shot (14.0625 seconds).\n\nUser request:\nCreate a multi-shot video using the following 3 shots in order.\n\nSUBJECTS\n- Character 1 (person): an adult courier, wearing a blue jacket\n- Character 2 (person): an adult woman, wearing a grey coat\n- Object 1 (object): a small brown parcel\nLOCATIONS\n- Location 1: a sunlit workshop\nSHOTS\ns01: Character 1 holds Object 1 and greets Character 2.\ns02: Character 1 hands Object 1 to Character 2.\ns03: Character 2 carries Object 1 to the window while Character 1 waits by the door."}
Output:
Global: Character 1 is an adult courier wearing a blue jacket. Character 2 is an adult woman wearing a grey coat. Object 1 is a small brown parcel. Location 1 is a sunlit workshop.
Shot 1 (4 s): Location 1, symmetrical composition. Character 1, the courier in the blue jacket, stands on the left in front of Character 2, holding Object 1. Character 2, the woman in the grey coat, stands empty-handed. Character 1 greets Character 2 while holding Object 1.
Shot 2 (4 s): Location 1, balanced composition. Character 1, the courier in the blue jacket, stands on the left holding Object 1; Character 2, the woman in the grey coat, stands empty-handed on the right. Character 1 extends Object 1 toward Character 2. Character 2 takes Object 1, and Character 1 lets go. By the end, Character 2 holds Object 1 and Character 1 is empty-handed.
Shot 3 (4 s): Location 1, balanced composition. Character 1, the courier in the blue jacket, stands empty-handed by the doorway on the left; Character 2, the woman in the grey coat, stands holding Object 1. Character 2 carries Object 1 to the window while Character 1 waits by the door. By the end, Character 2 stands at the window on the right of the frame, still holding Object 1.
The parcel changes hands once and remains with Character 2. Extending, taking, and letting go are a plausible inferred execution of the requested handover. Do not copy the example's characters or story into a different request.'''


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse(text: str) -> tuple[str | None, list[str], list[float | None]]:
    """``Global:`` 一行，``Shot k (x s):`` 每镜一行；续行并入上一项。"""
    global_caption, shots, seconds, current = None, [], [], None
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        head, shot = GLOBAL_LINE.match(line), SHOT_LINE.match(line)
        if head:
            global_caption, current = head.group(1), "global"
        elif shot:
            if int(shot.group(1)) != len(shots) + 1:
                raise ValueError(f"shot {shot.group(1)} out of order")
            shots.append(shot.group(3))
            seconds.append(float(shot.group(2)) if shot.group(2) else None)
            current = "shot"
        elif current == "global":
            global_caption += " " + line
        elif current == "shot":
            shots[-1] += " " + line
        else:
            raise ValueError("text before the global caption")
    return global_caption, shots, seconds


def assess(text: str, finish_reason: str, task: dict) -> dict:
    """与主线同口径：完整返回、能解析、逐镜档镜数等于请求、镜数装得下总时长。"""
    if finish_reason != "stop":
        return {"status": "planning_failed", "error": f"finish_reason {finish_reason}"}
    try:
        global_caption, shots, seconds = parse(text)
    except ValueError as error:
        return {"status": "planning_failed", "error": str(error)}
    if not global_caption or not shots or any(not caption.strip() for caption in shots):
        return {"status": "planning_failed", "error": "missing global caption or empty shot caption",
                "global": global_caption, "shots": shots, "seconds": seconds}
    record = {"global": global_caption, "shots": shots, "seconds": seconds}
    required, minimum = task["expected_shot_count"], task["minimum_shots"]
    if required is not None and len(shots) != required:
        return {**record, "status": "planning_failed", "error": f"planned {len(shots)} shots; user required {required}"}
    if required is None and len(shots) < minimum:
        return {**record, "status": "planning_failed", "error": f"planned {len(shots)} shots; duration needs {minimum}"}
    return {**record, "status": "completed", "error": None}


def correction(record: dict, task: dict) -> str | None:
    """主线 ``story_planner.structural_correction`` 的两条，措辞换成文字输出。"""
    shots = record.get("shots")
    if record["status"] != "planning_failed" or not shots:
        return None
    if task["expected_shot_count"] is not None and len(shots) != task["expected_shot_count"]:
        required = task["expected_shot_count"]
        return (f"The request lists exactly {required} shots, but your answer has {len(shots)}. Return the complete answer "
                f"again with exactly {required} shot lines, one per listed shot, in the order of the request.")
    if task["expected_shot_count"] is None and len(shots) < task["minimum_shots"]:
        return (f"Your answer has {len(shots)} shots, but {task['target_seconds']} seconds at {STORYBOARD_FPS} fps is "
                f"{task['target_seconds'] * STORYBOARD_FPS:g} frames and one shot holds at most {STORYBOARD_MAX_SHOT_FRAMES} "
                f"frames, so it needs at least {task['minimum_shots']} shots. Return the complete answer again with at least "
                f"{task['minimum_shots']} shots, keeping everything the request states.")
    return None


def complete(base_url: str, model: str, messages: list[dict], seed: int, timeout: int) -> tuple[str, str, dict]:
    body = {"model": model, "messages": messages, "seed": seed, "n": 1, "stream": False,
            "temperature": SAMPLING["temperature"], "top_p": SAMPLING["top_p"],
            "presence_penalty": SAMPLING["presence_penalty"], "max_tokens": SAMPLING["max_tokens"],
            "top_k": SAMPLING["top_k"], "min_p": SAMPLING["min_p"], "repetition_penalty": SAMPLING["repetition_penalty"],
            "chat_template_kwargs": {"enable_thinking": SAMPLING["enable_thinking"]}}
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer local-unauthed"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    choice = data["choices"][0]
    return choice["message"]["content"], choice["finish_reason"], data.get("usage") or {}


def plan(task: dict, args: argparse.Namespace) -> dict:
    path = args.output / task["tier"] / f"seed-{task['seed']}" / f"{task['story_id']}.json"
    if path.is_file():
        return json.loads(path.read_text())
    user = json.dumps(planner_request(script=task["request"], tier=TIERS[task["tier"]]), ensure_ascii=False)
    messages = [{"role": "system", "content": PROSE_SYSTEM}, {"role": "user", "content": user}]
    started = time.monotonic()
    content, finish, usage = complete(args.base_url, args.model, messages, task["seed"], args.timeout)
    record = {"raw_content": content, "finish_reason": finish, "usage": usage, **assess(content, finish, task)}
    message = correction(record, task)
    if message is not None:
        first = dict(record)
        messages += [{"role": "assistant", "content": content}, {"role": "user", "content": message}]
        content, finish, usage = complete(args.base_url, args.model, messages, task["seed"], args.timeout)
        record = {"raw_content": content, "finish_reason": finish, "usage": usage, **assess(content, finish, task),
                  "first_attempt": first, "correction": message}
    record.update(story_id=task["story_id"], tier=task["tier"], seed=task["seed"],
                  elapsed_seconds=round(time.monotonic() - started, 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--tiers", nargs="+", choices=tuple(TIERS), default=list(TIERS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    catalog = json.loads(args.requests.read_text())
    stories = [story for story in catalog["stories"] if story["split"] == args.split]
    if not stories:
        raise ValueError(f"no stories in split {args.split}")
    tasks = []
    for story in stories:
        for tier in args.tiers:
            text = story["inputs"][tier]["text"]
            required = len(REQUEST_SHOT.findall(text)) if tier == "shots" else None
            if tier == "shots" and not required:
                raise ValueError(f"{story['story_id']}: shots request lists no shots")
            minimum = math.ceil(story["target_seconds"] * STORYBOARD_FPS / STORYBOARD_MAX_SHOT_FRAMES)
            for seed in args.seeds:
                tasks.append({"story_id": story["story_id"], "tier": tier, "seed": seed, "target_seconds": story["target_seconds"],
                              "request": production_request(story, tier, "system"), "expected_shot_count": required,
                              "minimum_shots": minimum})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run.json").write_text(json.dumps({
        "started_at": now(), "requests": str(args.requests), "split": args.split, "model": args.model,
        "base_url": args.base_url, "seeds": args.seeds, "tiers": args.tiers, "sampling": SAMPLING,
        "system_prompt": PROSE_SYSTEM, "tasks": len(tasks)}, ensure_ascii=False, indent=1) + "\n")
    with ThreadPoolExecutor(args.workers) as pool:
        records = list(pool.map(lambda task: plan(task, args), tasks))
    summary = {}
    for task, record in zip(tasks, records):
        key = f"{task['tier']}/seed-{task['seed']}"
        block = summary.setdefault(key, {"completed": 0, "planning_failed": 0, "corrected": 0})
        block[record["status"]] += 1
        block["corrected"] += "correction" in record
    (args.output / "summary.json").write_text(json.dumps({"finished_at": now(), "by_tier_seed": summary}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
