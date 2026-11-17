"""规划器跨镜状态检查（文字层）：同一请求、三种输出形态，每个人物的衣着、持物与开镜姿态和状态是否与给定分镜一致。

三种形态（规划器设计页「对照怎么设」）：
- ``prose``：语言模型直接写渲染器读的文字，一段全局描述加逐镜描述；
- ``structured``：分镜 JSON，模型自己逐镜写衣着与持物（v15 提示词原答 ``raw_storyboard``），不跑维护；
- ``ours``：分镜 JSON（v17）经分镜表维护后编译成的逐镜文字（``storyboard`` 经 ``shot_text``）。

检查点只从给定分镜（源标注）推出，判官之前冻结：逐镜往后走，记每个人物的当前衣着（登记值，被在场条目的
``outfit`` 覆盖，被 ``changes`` 里 what=outfit 的 after 更新）与持物（只在分镜写明过时才有：在场条目带 ``holding``
键，或由更早写明的持物、持物变化延续下来；不从动作句推）。第 2 镜起，给定分镜里在场的每个人物出一条开镜衣着
检查点，持物有定义时再出一条；该人物在更早的镜头记过同类变化的，标 ``after_change``。开镜姿态与状态是规划器逐镜自己写、代码不维护的推断值，每一镜（含第 1 镜）
在场人物在给定分镜里写了 ``posture`` 或 ``state`` 的各出一条；判官判计划写的与它相同或相容、矛盾，还是没写。
冻结时用 ``--kinds`` 选种类：衣着与持物的读数已按旧冻结件判过，开镜单独冻结、单独判，不重判前两类。

判官读每种形态给渲染器的那一镜视图（散文＝全局描述＋本镜描述；结构 JSON＝登记表＋本镜条目；我们＝本镜编译文字），
每个（形态，种子，故事）一次调用，逐条回答 matches／conflicts／unstated。判官是 DeepSeek-V4.1-Flash，调用机制、
密钥（环境变量 ``DEEPSEEK_API_KEY``）与回执沿用 ``planner_text_checks``；系统消息固定，故事内容都在用户消息里。
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mspc.data.shot_text import UNSEEN_OUTFIT, shot_text
from mspc.evaluation.planner_text_checks import EFFORT, MODEL, CallLedger, dev_requests, now, write

FORMS = ("prose", "structured", "ours")
KINDS = ("outfit", "holding", "opening")
POSITION_KINDS = {"posture", "state"}  # 变化里属于开镜姿态与状态的 what
VERDICTS = ("matches", "conflicts", "unstated")
OUTCOMES = (*VERDICTS, "failed", "unmeasured")
SUBSETS = ("all", "after_change", "request_present", "holding_stated", "held_items", "empty_handed")
HOLDING_SUBSETS = {"holding_stated", "held_items", "empty_handed"}
MAX_WORKERS = 48
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 10_000, 0


# ---------------------------------------------------------------- 检查点


def _stated_outfit(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and not UNSEEN_OUTFIT.match(value.strip())


def checkpoint_id(shot: int, name: str, kind: str) -> str:
    return f"s{shot}-{name.replace(' ', '').lower()}-{kind}"


def story_checkpoints(board: Mapping, request_board: Mapping, kinds: Sequence[str]) -> list[dict]:
    """给定分镜逐镜走一遍，第 2 镜起每个在场人物的开镜衣着与（有定义时）开镜持物各一条；每一镜在场人物写了姿态或
    状态的，开镜姿态与状态一条。只出 ``kinds`` 里的种类。

    ``request_board`` 是同一故事只读请求的分镜，只用来标 ``request_present``（请求也把这个人物放在这一镜）。"""
    if not kinds or set(kinds) - set(KINDS):
        raise ValueError(f"kinds must be a nonempty subset of {KINDS}")
    characters = {entry["name"]: entry for entry in board["characters"]}
    if len(request_board["shots"]) != len(board["shots"]):
        raise ValueError("request storyboard and given storyboard differ in shot count")
    request_present = [{entry["name"] for entry in shot["present"]} for shot in request_board["shots"]]
    outfit = {name: entry.get("outfit") for name, entry in characters.items()}
    holding: dict[str, list[str]] = {}
    changed = {kind: set() for kind in KINDS}
    rows = []
    for index, shot in enumerate(board["shots"]):
        number = index + 1
        if shot["shot"] != number:
            raise ValueError(f"given storyboard shot {shot['shot']} at position {number}")
        names = [entry["name"] for entry in shot["present"]]
        if len(names) != len(set(names)):
            raise ValueError(f"shot {number}: duplicate present entries")
        for entry in shot["present"]:
            name = entry["name"]
            if name not in characters:
                continue
            if "outfit" in entry:
                outfit[name] = entry["outfit"]
            stated_here = "holding" in entry
            if stated_here:
                if not isinstance(entry["holding"], list):
                    raise ValueError(f"shot {number} {name}: holding is not a list")
                holding[name] = list(entry["holding"])
            common = {"shot": number, "character": name, "request_present": name in request_present[index]}
            opening = "; ".join(f"{key} {entry[key]}" for key in ("posture", "state")
                                if isinstance(entry.get(key), str) and entry[key].strip())
            if opening:
                rows.append({"id": checkpoint_id(number, name, "opening"), "kind": "opening", "value": opening,
                             "after_change": name in changed["opening"], **common})
            if number == 1:
                continue
            if _stated_outfit(outfit[name]):
                rows.append({"id": checkpoint_id(number, name, "outfit"), "kind": "outfit", "value": outfit[name],
                             "after_change": name in changed["outfit"], **common})
            if name in holding:
                rows.append({"id": checkpoint_id(number, name, "holding"), "kind": "holding", "value": holding[name],
                             "after_change": name in changed["holding"], "holding_stated": stated_here, **common})
        for change in shot["changes"]:
            what, name = change["what"], change["name"]
            if name in characters and what in POSITION_KINDS:
                changed["opening"].add(name)
            if what not in KINDS or name not in characters:
                continue
            if what == "outfit":
                outfit[name] = change["after"]
            else:
                if not isinstance(change["after"], list):
                    raise ValueError(f"shot {number} {name}: holding change after is not a list")
                holding[name] = list(change["after"])
            changed[what].add(name)
    return [row for row in rows if row["kind"] in kinds]


def id_key(board: Mapping) -> dict[str, str]:
    """判官对照用的编号表：人物与物体只给外貌（不给衣着，免得把登记衣着当成答案），地点给描述。"""
    return {entry["name"]: entry["appearance"] for registry in ("characters", "objects", "places")
            for entry in board[registry]}


def frozen_kinds(stories: Mapping[str, Mapping]) -> list[str]:
    present = {row["kind"] for story in stories.values() for row in story["checkpoints"]}
    return [kind for kind in KINDS if kind in present]


def counts(stories: Mapping[str, Mapping]) -> dict:
    rows = [row for story in stories.values() for row in story["checkpoints"]]
    out = {}
    for kind in frozen_kinds(stories):
        kept = [row for row in rows if row["kind"] == kind]
        out[kind] = {subset: sum(_in_subset(row, subset) for row in kept) for subset in SUBSETS
                     if subset not in HOLDING_SUBSETS or kind == "holding"}
        out[kind]["stories"] = len({story for story, value in stories.items()
                                    if any(row["kind"] == kind for row in value["checkpoints"])})
        out[kind]["stories_after_change"] = len({story for story, value in stories.items()
                                                 if any(row["kind"] == kind and row["after_change"]
                                                        for row in value["checkpoints"])})
    return out


def _in_subset(row: Mapping, subset: str) -> bool:
    if subset == "all":
        return True
    if subset == "holding_stated":
        return row.get("holding_stated", False)
    if subset in ("held_items", "empty_handed"):
        return row["kind"] == "holding" and bool(row["value"]) == (subset == "held_items")
    return row[subset]


def freeze(requests: Path, split: str, storyboards: Path, request_storyboards: Path, kinds: Sequence[str]) -> dict:
    stories = dev_requests(requests, split)
    given = {story["story_id"]: story["storyboard"] for story in json.loads(storyboards.read_text())["stories"]}
    request_only = {story["story_id"]: story["storyboard"]
                    for story in json.loads(request_storyboards.read_text())["stories"]}
    # 只读请求的分镜（request_storyboards_v4）没有在场状态与变化，从它推不出换装、交接之后的检查点。
    character_changes = sum(change["name"] in {entry["name"] for entry in given[story_id]["characters"]}
                            for story_id in stories for shot in given[story_id]["shots"] for change in shot["changes"])
    if not character_changes:
        raise ValueError(f"{storyboards} records no character changes in split {split}; "
                         "the given storyboard must be the film annotation")
    frozen = {}
    for story_id in sorted(stories):
        board = given[story_id]
        frozen[story_id] = {"shot_count": len(board["shots"]), "id_key": id_key(board),
                            "checkpoints": story_checkpoints(board, request_only[story_id], kinds)}
    return {"written_at": now(), "requests": str(requests), "split": split, "storyboards": str(storyboards),
            "request_storyboards": str(request_storyboards), "kinds": list(kinds),
            "rule": "opening state of shot k = state after shot k-1 changes, overridden by shot k present entries; "
                    "outfit starts from the registry; holding exists only once a present holding key or a holding "
                    "change states it; outfit and holding checkpoints for shots >= 2, every present character; "
                    "opening = the present entry's posture and state, every shot, when written; after_change = the "
                    "character had a recorded change of the same kind (posture or state for opening) in an earlier shot",
            "counts": counts(frozen), "stories": frozen}


# ---------------------------------------------------------------- 渲染器视图


FORM_LABELS = {"prose": "prose (a global caption shared by every shot, plus one caption per shot)",
               "structured": "structured JSON (a registry shared by every shot, plus one JSON entry per shot)",
               "ours": "compiled text (one self-contained text per shot, nothing shared)"}


def plan_views(form: str, record: Mapping) -> tuple[str, list[str]] | None:
    """（各镜共享的部分，逐镜视图）；计划失败时为 None。镜数核对由调用方按给定分镜做。"""
    if record.get("status") != "completed":
        return None
    if form == "prose":
        return record["global"], list(record["shots"])
    if form == "structured":
        raw = record["raw_storyboard"]
        registry = {key: raw[key] for key in ("characters", "objects", "places")}
        compact = lambda value: json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
        return compact(registry), [compact(shot) for shot in raw["shots"]]
    if form == "ours":
        board = record["storyboard"]
        return "", [shot_text(board, index) for index in range(len(board["shots"]))]
    raise ValueError(f"unknown form {form!r}")


def _checkpoint_line(row: Mapping, key: Mapping[str, str]) -> str:
    if row["kind"] == "opening":
        return f"[{row['id']}] Shot {row['shot']}, {row['character']}, posture and visible state at the start of the shot: {row['value']}"
    if row["kind"] == "outfit":
        value = row["value"]
    elif row["value"]:
        value = ", ".join(f"{name} ({key[name]})" if name in key else name for name in row["value"])
    else:
        value = "empty-handed"
    return f"[{row['id']}] Shot {row['shot']}, {row['character']}, {row['kind']} at the start of the shot: {value}"


JUDGE_SCHEMA = {"type": "json_schema", "json_schema": {"name": "planner_state_checks", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["checkpoints"], "properties": {
        "checkpoints": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                   "required": ["id", "verdict", "evidence"], "properties": {
            "id": {"type": "string"}, "verdict": {"type": "string", "enum": list(VERDICTS)},
            "evidence": {"type": "string"}}}}}}}}


JUDGE_SYSTEM = """You check whether a multi-shot video plan gives each character the right outfit, held items, and opening posture and state. A video renderer draws every shot separately and reads only that shot's view of the plan. The user message gives the plan form, an ID key, the part of the plan shared by every shot (if any), the view of every shot, and checkpoints written from a reference storyboard before any plan was read. Each checkpoint names a shot, a character, and what that character wears, holds, or how it is posed and looks at the START of that shot.

How a shot's view is read, by plan form:
- prose: the global caption plus that shot's caption. A character the shot caption shows without its own outfit wording wears what the global caption gives it.
- structured JSON: the registry (characters with appearance and registered outfit, objects, places) plus that shot's JSON entry. A present entry without "outfit" wears the registered outfit. "holding" lists what the character holds at the start of the shot; [] means empty-handed; a missing "holding" key says nothing. In "changes", before is the start of the shot and after is the end.
- compiled text: that shot's text only. What follows "By the end of the shot" is the end of the shot, not its start.

For each checkpoint decide what a renderer reading only the view of the checkpoint's shot would show at the start of that shot:
- matches: the view gives the character that outfit (paraphrase, word order and articles do not matter), or exactly those held items (for "empty-handed", the view says the character holds nothing), or a posture and state that are the same or compatible (seated at the desk = sitting; tears on her face = crying; standing still is compatible with standing and looking down). For posture and state, judge the body position and visible condition only; gaze, gestures, and actions the checkpoint lists but the view leaves out do not make a conflict.
- conflicts: the view gives a different outfit or different held items: a garment the checkpoint no longer lists is still worn, a listed garment is missing or replaced, a held item is missing or extra, or the character holds something when the checkpoint says empty-handed. For posture and state: the view gives an incompatible body position or condition at the start of the shot (standing against sitting, lying against kneeling, calm against crying).
- unstated: the view says nothing about it, or the character does not appear in that shot's own part (its caption, its entry, or its text).
Use only the view of the checkpoint's shot. A change written in another shot does not count unless the checkpoint's shot itself shows the result, and the end of a shot does not count for its start. People and objects may be named by ID or described; use the ID key to match them.

Output one json object with the key "checkpoints": one entry per checkpoint, its id copied exactly as written inside the square brackets, verdict one of matches | conflicts | unstated, and evidence: the few words of the view that decided it ("" when unstated).

EXAMPLE JSON OUTPUT (for checkpoints [s2-character1-outfit] and [s2-character1-holding]):
{"checkpoints": [{"id": "s2-character1-outfit", "verdict": "matches", "evidence": "Character 1, the courier in the blue jacket"}, {"id": "s2-character1-holding", "verdict": "conflicts", "evidence": "stands holding Object 1"}]}"""


def judge_prompt(form: str, shared: str, views: Sequence[str], story: Mapping) -> tuple[str, str]:
    """（系统消息, 用户消息）：系统消息每次相同，便于前缀缓存；形态、视图与检查点都在用户消息。"""
    key = story["id_key"]
    shared_label = {"prose": "global caption", "structured": "registry", "ours": "nothing is shared"}[form]
    lines = [f"PLAN FORM: {FORM_LABELS[form]}", "",
             "ID KEY (appearance only, from the request)", *(f"- {name}: {text}" for name, text in key.items()), "",
             f"SHARED BY EVERY SHOT ({shared_label})", shared if shared else "(none)", "",
             "SHOT VIEWS", *(f"Shot {index}: {view}" for index, view in enumerate(views, 1)), "",
             "CHECKPOINTS", *(_checkpoint_line(row, key) for row in story["checkpoints"])]
    return JUDGE_SYSTEM, "\n".join(lines)


# ---------------------------------------------------------------- 判官


def load_form_records(form: str, specs: Sequence[str], stories: Sequence[str]) -> dict[tuple[int, str], dict]:
    """``SEED=DIR``，DIR 里每个故事一个 ``<story_id>.json``；缺一个就报错。"""
    if form not in FORMS:
        raise ValueError(f"unknown form {form!r}")
    records = {}
    for spec in specs:
        seed, directory = spec.split("=", 1)
        for story_id in stories:
            path = Path(directory) / f"{story_id}.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing plan record {path}")
            records[(int(seed), story_id)] = {**json.loads(path.read_text()), "_path": str(path)}
    return records


def grade(story: Mapping, answer: Mapping | None) -> dict[str, dict]:
    """判官回答折成每条检查点的结论；回答缺条、多条或重复时整份记 unmeasured。"""
    ids = [row["id"] for row in story["checkpoints"]]
    if answer is None:
        return {checkpoint: {"verdict": "unmeasured"} for checkpoint in ids}
    rows = answer["checkpoints"]
    by_id = {row["id"]: row for row in rows}
    if len(rows) != len(by_id) or set(by_id) != set(ids):
        return {checkpoint: {"verdict": "unmeasured", "error": "answer ids differ from the frozen checkpoints"}
                for checkpoint in ids}
    return {checkpoint: {"verdict": by_id[checkpoint]["verdict"], "evidence": by_id[checkpoint]["evidence"]}
            for checkpoint in ids}


def judge(form: str, records: Mapping[tuple[int, str], Mapping], frozen: Mapping, calls: Path, workers: int) -> dict:
    ledger = CallLedger(calls, EFFORT)
    plans, jobs = {}, []
    for (seed, story_id), record in sorted(records.items()):
        story = frozen["stories"][story_id]
        views = plan_views(form, record)
        base = {"form": form, "seed": seed, "story_id": story_id, "record": record["_path"]}
        if views is None or len(views[1]) != story["shot_count"]:
            reason = (f"status {record.get('status')}" if views is None
                      else f"{len(views[1])} shots, given storyboard has {story['shot_count']}")
            plans[f"{seed}/{story_id}"] = {**base, "plan": "failed", "reason": reason,
                                           "verdicts": {row["id"]: {"verdict": "failed"} for row in story["checkpoints"]}}
            continue
        if not story["checkpoints"]:
            plans[f"{seed}/{story_id}"] = {**base, "plan": "no_checkpoints", "verdicts": {}}
            continue
        jobs.append((f"{form}/seed-{seed}/{story_id}", seed, story_id, base, judge_prompt(form, *views, story)))
    print(json.dumps({"form": form, "judge_calls": len(jobs),
                      "failed_plans": sum(plan["plan"] == "failed" for plan in plans.values())}), flush=True)

    def run(job):
        call_id, seed, story_id, base, prompt = job
        record = ledger.call(call_id, *prompt, JUDGE_SCHEMA)
        print(json.dumps({"call": call_id, "status": record["status"]}), flush=True)
        answer = record.get("result") if record["status"] == "answered" else None
        return f"{seed}/{story_id}", {**base, "plan": "judged" if answer is not None else "unmeasured",
                                      "call": call_id, "call_status": record["status"],
                                      "verdicts": grade(frozen["stories"][story_id], answer)}

    with ThreadPoolExecutor(workers) as pool:
        plans.update(dict(pool.map(run, jobs)))
    return {"form": form, "observed_at": now(), "judge": {"model": MODEL, "effort": EFFORT},
            "checkpoints_written_at": frozen["written_at"], "plans": dict(sorted(plans.items()))}


# ---------------------------------------------------------------- 汇总


def _story_counts(frozen: Mapping, judged: Mapping, kind: str, subset: str) -> dict[str, Counter]:
    """每个故事两种子合起来的各结论计数（只含这一格有检查点的故事）。"""
    per_story: dict[str, Counter] = {}
    for plan in judged["plans"].values():
        rows = [row for row in frozen["stories"][plan["story_id"]]["checkpoints"]
                if row["kind"] == kind and _in_subset(row, subset)]
        if rows:
            bucket = per_story.setdefault(plan["story_id"], Counter())
            bucket.update(plan["verdicts"][row["id"]]["verdict"] for row in rows)
    return per_story


def _bootstrap_rate(per_story: Mapping[str, Counter], stories: Sequence[str]) -> np.ndarray:
    matches = np.array([per_story[story]["matches"] for story in stories], dtype=float)
    totals = np.array([sum(per_story[story].values()) for story in stories], dtype=float)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(stories), size=(BOOTSTRAP_RESAMPLES, len(stories)))
    return matches[draws].sum(axis=1) / totals[draws].sum(axis=1)


def summarize(frozen: Mapping, judged: Mapping[str, Mapping]) -> dict:
    """每形态 × 种类 × 子集：各结论计数与比率（两种子合计），匹配率的故事 bootstrap 95% 区间；形态两两的配对差。"""
    seeds = {form: sorted({plan["seed"] for plan in value["plans"].values()}) for form, value in judged.items()}
    cells, paired = {}, {}
    for kind in frozen_kinds(frozen["stories"]):
        for subset in SUBSETS:
            if subset in HOLDING_SUBSETS and kind != "holding":
                continue
            per_form = {}
            for form, value in judged.items():
                per_story = _story_counts(frozen, value, kind, subset)
                if not per_story:
                    continue
                stories = sorted(per_story)
                total = Counter()
                for bucket in per_story.values():
                    total.update(bucket)
                n = sum(total.values())
                low, high = np.percentile(_bootstrap_rate(per_story, stories), [2.5, 97.5])
                cells.setdefault(form, {}).setdefault(kind, {})[subset] = {
                    "checkpoints": n, "stories": len(stories), "seeds": seeds[form],
                    **{outcome: total[outcome] for outcome in OUTCOMES},
                    **{f"{outcome}_rate": total[outcome] / n for outcome in OUTCOMES},
                    "match_rate_interval": [float(low), float(high)]}
                per_form[form] = per_story
            for a, b in ((x, y) for i, x in enumerate(FORMS) for y in FORMS[i + 1:]):
                if a not in per_form or b not in per_form or seeds[a] != seeds[b]:
                    continue
                stories = sorted(set(per_form[a]) & set(per_form[b]))
                difference = _bootstrap_rate(per_form[b], stories) - _bootstrap_rate(per_form[a], stories)
                rate = lambda per: sum(per[s]["matches"] for s in stories) / sum(sum(per[s].values()) for s in stories)
                low, high = np.percentile(difference, [2.5, 97.5])
                paired.setdefault(kind, {}).setdefault(subset, {})[f"{b} - {a}"] = {
                    "match_rate_difference": rate(per_form[b]) - rate(per_form[a]),
                    "interval": [float(low), float(high)], "stories": len(stories)}
    return {"observed_at": now(), "checkpoints": frozen["storyboards"], "checkpoint_counts": counts(frozen["stories"]),
            "judge": {"model": MODEL, "effort": EFFORT},
            "bootstrap": {"unit": "story (both seeds of a story drawn together)", "resamples": BOOTSTRAP_RESAMPLES,
                          "seed": BOOTSTRAP_SEED, "statistic": "pooled matches / all checkpoints, 95% percentile",
                          "paired": "same story draws for both forms"},
            "definitions": {"matches": "the shot's renderer view gives that opening outfit / held items, or a same or compatible opening posture and state",
                            "conflicts": "the view gives a different outfit / held items, or an incompatible opening posture or state",
                            "unstated": "the view does not say, or the character is not in that shot's own part",
                            "failed": "plan status is not completed or its shot count differs from the given storyboard",
                            "unmeasured": "judge call without a usable answer",
                            "after_change": "the character had a recorded change of the same kind in an earlier shot",
                            "request_present": "the request-only storyboard also places the character in that shot",
                            "holding_stated": "the given storyboard writes the holding at that shot (not carried)",
                            "held_items": "holding checkpoints whose value lists at least one item",
                            "empty_handed": "holding checkpoints whose value is empty-handed"},
            "forms": cells, "paired": paired}


def print_table(summary: Mapping) -> None:
    print(f"{'form':<11}{'kind':<8}{'subset':<16}{'n':>5}{'match':>7}{'confl':>7}{'unst':>6}{'fail':>6}{'unmes':>6}"
          f"{'match%':>8}  interval")
    for form, kinds in summary["forms"].items():
        for kind, subsets in kinds.items():
            for subset, cell in subsets.items():
                low, high = cell["match_rate_interval"]
                print(f"{form:<11}{kind:<8}{subset:<16}{cell['checkpoints']:>5}{cell['matches']:>7}{cell['conflicts']:>7}"
                      f"{cell['unstated']:>6}{cell['failed']:>6}{cell['unmeasured']:>6}{100 * cell['matches_rate']:>7.1f}%"
                      f"  [{100 * low:.1f}, {100 * high:.1f}]")
    for kind, subsets in summary["paired"].items():
        for subset, pairs in subsets.items():
            for name, cell in pairs.items():
                low, high = cell["interval"]
                print(f"paired {kind:<8}{subset:<16}{name:<22}{100 * cell['match_rate_difference']:+.1f}  "
                      f"[{100 * low:+.1f}, {100 * high:+.1f}]")


# ---------------------------------------------------------------- 命令


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freezing = commands.add_parser("checkpoints", help="由给定分镜推出并冻结检查点")
    freezing.add_argument("--requests", type=Path, required=True)
    freezing.add_argument("--split", required=True, help="请求表里的划分名（dev、test，或规划器样本池扩样的 val、train-change）")
    freezing.add_argument("--storyboards", type=Path, required=True, help="给定分镜（带在场状态与变化的源标注）")
    freezing.add_argument("--request-storyboards", type=Path, required=True, help="只读请求的分镜，只用来标 request_present")
    freezing.add_argument("--kinds", nargs="+", choices=KINDS, required=True, help="冻结哪些种类的检查点")
    freezing.add_argument("--output", type=Path, required=True)
    judging = commands.add_parser("judge")
    judging.add_argument("--checkpoints", type=Path, required=True)
    judging.add_argument("--form", choices=FORMS, required=True)
    judging.add_argument("--records", nargs="+", required=True, help="SEED=DIR")
    judging.add_argument("--output", type=Path, required=True)
    judging.add_argument("--calls", type=Path, required=True)
    judging.add_argument("--workers", type=int, required=True)
    summarizing = commands.add_parser("summarize")
    summarizing.add_argument("--checkpoints", type=Path, required=True)
    summarizing.add_argument("--judged", nargs="+", required=True, help="FORM=judge output")
    summarizing.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "checkpoints":
        frozen = freeze(args.requests, args.split, args.storyboards, args.request_storyboards, args.kinds)
        write(args.output, frozen)
        print(json.dumps({"stories": len(frozen["stories"]), "counts": frozen["counts"]}, indent=1))
        return 0
    frozen = json.loads(args.checkpoints.read_text())
    if args.command == "judge":
        if not 1 <= args.workers <= MAX_WORKERS:
            parser.error(f"workers must be within 1..{MAX_WORKERS}")
        records = load_form_records(args.form, args.records, sorted(frozen["stories"]))
        write(args.output, judge(args.form, records, frozen, args.calls, args.workers))
        return 0
    judged = {}
    for spec in args.judged:
        form, path = spec.split("=", 1)
        if form not in FORMS:
            raise ValueError(f"unknown form {form!r}")
        content = json.loads(Path(path).read_text())
        if content["form"] != form or content["checkpoints_written_at"] != frozen["written_at"]:
            raise ValueError(f"{path}: judged {content['form']} against other checkpoints")
        judged[form] = content
    summary = summarize(frozen, {form: judged[form] for form in FORMS if form in judged})
    write(args.output, summary)
    print_table(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
