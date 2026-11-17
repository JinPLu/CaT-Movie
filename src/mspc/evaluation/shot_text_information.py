"""每镜可渲染信息量：渲染器读到的每一镜文字里，写明了几类可渲染的信息。

动机：引言要求一引
Context Scaling——生成保真随结构化的可渲染信息量增长，不随文字长度增长。方法图 3 (a) 只说明了请求里缺多少，
这里量分镜表给每一镜补回了多少。补的是不是对的，由照写判读（新编造、写反）另撑。

四种文字，都按「渲染器读到的那一镜」给判官：
- ``request_only``：逐镜请求不经语言模型、按请求直接编译（统一表 ``request-only`` 一行），只有逐镜档；
- ``prose``：语言模型直接写的全局描述＋这一镜的描述（``tools/run/planner_prose_baseline.py``）；
- ``ours``：维护后的分镜表编出的这一镜文字（与渲染器训练同一编译器 ``shot_text``）；
- ``film``：片子分镜标注编出的这一镜文字，渲染器训练时读的就是这种，作参照。

判官对每一镜按固定清单判每类信息「写了／没写／不适用」，清单照方法图 3 (a) 的分类，也覆盖 Context Scaling 字段消融里
贡献最大的场景与位置两类。每镜得分＝写了的类数。判官是 DeepSeek-V4.1-Flash，调用方式与回执沿用
``planner_text_checks.CallLedger``；它只看文字是否写明，不判对错。

用法::

    PYTHONPATH=src .venv/bin/python -m mspc.evaluation.shot_text_information judge --requests <请求表> --split dev \\
        --form ours --records shots:42=<目录> ... --storyboards <片子分镜表> --output <判读.json> --calls <回执目录> --workers 48
    PYTHONPATH=src .venv/bin/python -m mspc.evaluation.shot_text_information summarize --judged ours=<判读.json> ... --output <汇总.json>
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from mspc.evaluation.planner_text_checks import EFFORT, MODEL, CallLedger, dev_requests, now, write

FORMS = ("request_only", "prose", "ours", "film")
TIERS = ("synopsis", "paragraph", "shots")
#: 类别（键, 判官看到的说明）。人物类在这一镜没有人时为不适用；相对站位在少于两人时为不适用。
CATEGORIES = (
    ("place", "the location or setting of the shot"),
    ("shot_size", "the shot size (close-up, medium shot, wide shot, ...)"),
    ("camera", "the camera angle or camera movement (eye-level, high angle, over-the-shoulder, pan, push in, ...)"),
    ("composition", "the composition of the frame (centered, balanced, left-heavy, symmetrical, ...)"),
    ("identity", "who each person in the shot is, by appearance (hair, age, build, face)"),
    ("outfit", "what each person in the shot is wearing"),
    ("opening_state", "each person's posture or visible state when the shot opens (seated, standing, crying, ...)"),
    ("holding", "what each person holds, or that their hands are empty"),
    ("positions", "where the people stand relative to each other or in the frame (left of, behind, in the foreground, ...)"),
    ("action", "the action, as a visible process of who does what"),
    ("end_result", "what is visibly different by the end of the shot"),
)
PEOPLE = {"identity", "outfit", "opening_state", "holding"}
ANSWER = {"type": "string", "enum": ["stated", "not_stated", "not_applicable"]}
SCHEMA = {"type": "json_schema", "json_schema": {"name": "shot_text_information", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["shots"], "properties": {"shots": {
        "type": "array", "items": {"type": "object", "additionalProperties": False,
                                   "required": ["shot", *[key for key, _ in CATEGORIES]],
                                   "properties": {"shot": {"type": "integer"},
                                                  **{key: ANSWER for key, _ in CATEGORIES}}}}}}}}

SYSTEM = """You read the text prompts a video generator receives for a multi-shot video and record which kinds of renderable information each shot's prompt states. You do not judge whether the information is right or good; only whether it is written down.

For each shot, use only the text given for that shot (and the global caption when one is given, which the generator also reads for every shot). For each category answer:
- stated: the text for this shot explicitly gives it.
- not_stated: the text does not give it.
- not_applicable: only for the person categories when no person appears in this shot, and for positions when fewer than two people appear.

Categories:
""" + "\n".join(f"- {key}: {text}" for key, text in CATEGORIES) + """

A person category is stated when the text gives it for the people it describes in this shot. Output one json object with the key "shots": one entry per shot in order, "shot" being the shot number as given, and one answer per category.

EXAMPLE JSON OUTPUT (for one shot):
{"shots": [{"shot": 1, "place": "stated", "shot_size": "stated", "camera": "not_stated", "composition": "not_stated", "identity": "stated", "outfit": "stated", "opening_state": "stated", "holding": "not_stated", "positions": "not_applicable", "action": "stated", "end_result": "not_stated"}]}"""


def shot_views(form: str, record: Mapping | None, board: Mapping | None) -> list[str] | None:
    """这一行在每一镜给渲染器的文字；计划没编成（或结构不合格）返回 None。"""
    from mspc.data.shot_text import shot_text
    if form == "film":
        return [shot_text(board, index) for index in range(len(board["shots"]))]
    if record is None or record.get("status") != "completed":
        return None
    if form == "prose":
        return [f"GLOBAL CAPTION (read with every shot): {record['global']}\nSHOT: {caption}" for caption in record["shots"]]
    plan = record["storyboard"]
    return [shot_text(plan, index) for index in range(len(plan["shots"]))]


def user_message(views: Sequence[str]) -> str:
    return "\n\n".join(f"Shot {index + 1}:\n{text}" for index, text in enumerate(views))


def judge(args: argparse.Namespace) -> None:
    stories = dev_requests(args.requests, args.split)
    boards = {story["story_id"]: story["storyboard"] for story in json.loads(args.storyboards.read_text())["stories"]}
    jobs = []
    if args.form == "film":
        for story_id in sorted(stories):
            jobs.append(("shots", 0, story_id, None))
    else:
        for spec in args.records:
            tier_seed, directory = spec.split("=", 1)
            tier, seed = tier_seed.split(":")
            if tier not in TIERS:
                raise ValueError(f"unknown tier {tier!r}")
            for story_id in sorted(stories):
                path = Path(directory) / f"{story_id}.json"
                if not path.is_file():
                    raise FileNotFoundError(f"missing plan record {path}")
                jobs.append((tier, int(seed), story_id, json.loads(path.read_text())))
    ledger = CallLedger(args.calls, EFFORT)

    def run(job):
        tier, seed, story_id, record = job
        views = shot_views(args.form, record, boards.get(story_id))
        key = f"{tier}/seed-{seed}/{story_id}"
        if views is None:
            return key, {"tier": tier, "seed": seed, "story_id": story_id, "plan": "failed"}
        answer = ledger.call(f"{args.form}/{key}", SYSTEM, user_message(views), SCHEMA)
        out = {"tier": tier, "seed": seed, "story_id": story_id, "plan": "judged", "shots": len(views),
               "call_status": answer["status"]}
        if answer["status"] == "answered":
            rows = answer["result"]["shots"]
            if [row["shot"] for row in rows] != list(range(1, len(views) + 1)):
                out["call_status"] = "shot_numbers_differ"
            else:
                out["answers"] = [{key: row[key] for key, _ in CATEGORIES} for row in rows]
        return key, out

    with ThreadPoolExecutor(args.workers) as pool:
        plans = dict(pool.map(run, jobs))
    write(args.output, {"form": args.form, "observed_at": now(), "judge": {"model": MODEL, "effort": EFFORT},
                        "categories": [key for key, _ in CATEGORIES], "plans": plans})
    print(json.dumps({"form": args.form, "plans": len(plans),
                      "measured": sum("answers" in plan for plan in plans.values())}))


def story_counts(judged: Mapping) -> dict:
    """{档: {故事: {"shots": 镜数, "stated": 写了的类数之和, 类: [写了, 适用]}}}，两种子合并。"""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for plan in judged["plans"].values():
        if "answers" not in plan:
            continue
        block = out[plan["tier"]][plan["story_id"]]
        for answers in plan["answers"]:
            block["shots"][0] += 1
            block["stated"][0] += sum(value == "stated" for value in answers.values())
            for key, value in answers.items():
                if value != "not_applicable":
                    block[key][0] += value == "stated"
                    block[key][1] += 1
    return out


def bootstrap(stories: Sequence[str], numerator, denominator, resamples: int, rng) -> tuple[float, list[float]]:
    num = np.array([numerator(s) for s in stories], dtype=float)
    den = np.array([denominator(s) for s in stories], dtype=float)
    draws = rng.integers(0, len(stories), size=(resamples, len(stories)))
    ratios = num[draws].sum(axis=1) / np.maximum(den[draws].sum(axis=1), 1)
    return float(num.sum() / den.sum()), [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))]


def summarize(args: argparse.Namespace) -> None:
    counts = {spec.split("=", 1)[0]: story_counts(json.loads(Path(spec.split("=", 1)[1]).read_text())) for spec in args.judged}
    rng = np.random.default_rng(args.bootstrap_seed)
    summary = {"observed_at": now(), "bootstrap": {"resamples": args.resamples, "seed": args.bootstrap_seed,
                                                    "unit": "story; two seeds pooled; ratio of sums"}, "forms": {}, "paired": {}}
    for form, by_tier in counts.items():
        for tier, stories in by_tier.items():
            ids = sorted(stories)
            value, interval = bootstrap(ids, lambda s: stories[s]["stated"][0], lambda s: stories[s]["shots"][0], args.resamples, rng)
            block = {"stories": len(ids), "shots": int(sum(stories[s]["shots"][0] for s in ids)),
                     "stated_per_shot": {"value": value, "ci95": interval}, "categories": {}}
            for key, _ in CATEGORIES:
                den = sum(stories[s][key][1] for s in ids)
                if den:
                    block["categories"][key] = {"stated": int(sum(stories[s][key][0] for s in ids)), "applicable": int(den),
                                                "rate": sum(stories[s][key][0] for s in ids) / den}
            summary["forms"].setdefault(form, {})[tier] = block
    film = counts.get("film", {}).get("shots", {})
    for left, right in (("ours", "prose"), ("ours", "request_only"), ("prose", "request_only"), ("ours", "film")):
        for tier in TIERS:
            a = counts.get(left, {}).get(tier, {}) if left != "film" else film
            b = counts.get(right, {}).get(tier, {}) if right != "film" else film
            ids = sorted(set(a) & set(b))
            if not ids:
                continue
            draws = rng.integers(0, len(ids), size=(args.resamples, len(ids)))
            per = lambda c: np.array([[c[s]["stated"][0], c[s]["shots"][0]] for s in ids], dtype=float)
            pa, pb = per(a), per(b)
            diff = pa[draws, 0].sum(1) / pa[draws, 1].sum(1) - pb[draws, 0].sum(1) / pb[draws, 1].sum(1)
            summary["paired"][f"{tier}: {left} - {right}"] = {
                "stories": len(ids), "point": float(pa[:, 0].sum() / pa[:, 1].sum() - pb[:, 0].sum() / pb[:, 1].sum()),
                "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
    write(args.output, summary)
    for form, by_tier in summary["forms"].items():
        for tier, block in by_tier.items():
            v = block["stated_per_shot"]
            print(f"{form:13s} {tier:9s} stories {block['stories']:3d} shots {block['shots']:4d} stated/shot "
                  f"{v['value']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]")
    for name, block in summary["paired"].items():
        print(f"paired {name:30s} {block['point']:+.2f} [{block['ci95'][0]:+.2f}, {block['ci95'][1]:+.2f}]")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    judging = commands.add_parser("judge")
    judging.add_argument("--requests", type=Path, required=True)
    judging.add_argument("--split", required=True)
    judging.add_argument("--form", choices=FORMS, required=True)
    judging.add_argument("--records", nargs="*", default=[], help="TIER:SEED=DIR；film 不用")
    judging.add_argument("--storyboards", type=Path, required=True, help="片子分镜标注（film 一行用它编文字）")
    judging.add_argument("--output", type=Path, required=True)
    judging.add_argument("--calls", type=Path, required=True)
    judging.add_argument("--workers", type=int, required=True)
    summarizing = commands.add_parser("summarize")
    summarizing.add_argument("--judged", nargs="+", required=True, help="FORM=判读.json")
    summarizing.add_argument("--output", type=Path, required=True)
    summarizing.add_argument("--resamples", type=int, required=True)
    summarizing.add_argument("--bootstrap-seed", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "judge":
        if args.form != "film" and not args.records:
            parser.error("--records is required except for --form film")
        judge(args)
    else:
        summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
