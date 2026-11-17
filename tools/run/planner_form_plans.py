#!/usr/bin/env python3
"""规划器输出形态对照扩样：把规划器样本池的逐镜档行变成请求表与标注分镜表，再用主线同一个
出计划函数（``story_planner.plan_story``，同一约束解码、同一结构错误回灌、同一编译）出两种结构化计划：

- ``current``（我们）：现行提示词 ``PLANNER_SYSTEM``，代码维护分镜表；
- ``v15``（结构化不维护）：11-11 的 v15 提示词（提交 ``fb079f61c``，要求模型每镜自己写全衣着与持物），读它的原答
  ``raw_storyboard``；它的编译结果不用。

直接写文字那一行用 ``tools/run/planner_prose_baseline.py``，读同一份请求表。只调用已起好的 OpenAI 兼容服务，不启服务。

开发集只记了 4 条衣着变化、21 条持物变化，代码维护的作用量不出；验证集逐镜档 127 行也只有 2 条衣着变化，所以另从训练池
抽有衣着变化的逐镜行（每部片子一条，固定种子 0 抽 100 部）。主线规划器是零样本公共 Qwen，没在这个池上训练过。两份分开报。

用法::

    PYTHONPATH=src .venv/bin/python tools/run/planner_form_plans.py catalog --rows <池行.jsonl> --split val \\
        --requests-output <请求表.json> --storyboards-output <分镜表.json>
    PYTHONPATH=src .venv/bin/python tools/run/planner_form_plans.py plan --requests <请求表.json> --split val \\
        --prompt v17 --base-url http://127.0.0.1:18185/v1 --model Qwen3.5-9B --provenance <服务来源.json> \\
        --seeds 42 43 --output <记录根> --workers 64
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess

PRODUCTION = re.compile(r"Production requirements: approximately (\S+) seconds total; (\d+) fps; "
                        r"at most \d+ frames per shot \(\S+ seconds\)\.\n\nUser request:\n(.+)", re.DOTALL)
REQUEST_SHOT = re.compile(r"(?m)^s(\d+): ")
V15_COMMIT = "fb079f61c"
# 非思考档与 configs/infer/benchmark.json 相同；思考档照 Qwen3.8 模型卡（Best Practices）的思考采样，档位见 REASONING_EFFORTS
NONTHINKING = {"enable_thinking": False, "reasoning_effort": None, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0,
               "presence_penalty": 1.5, "repetition_penalty": 1.0, "max_tokens": 32768}
THINKING = {"enable_thinking": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0, "presence_penalty": 0.0,
            "repetition_penalty": 1.0, "max_tokens": 32768}


def sampling_preset(name: str) -> dict:
    return NONTHINKING if name == "nonthinking" else {**THINKING, "reasoning_effort": name.removeprefix("thinking-")}


def catalog(rows_path: Path, split: str) -> tuple[dict, dict]:
    """池行（``validate_planner_example`` 的格式）→ 请求表（``story_requests_v4`` 的形状，只有逐镜档）与标注分镜表。"""
    stories, boards = [], []
    for line in rows_path.read_text().splitlines():
        row = json.loads(line)
        if row["tier"] != "shot":
            continue
        match = PRODUCTION.fullmatch(row["script"])
        if match is None:
            raise ValueError(f"{row['sample_id']}: script lacks the production header")
        story_id = row["sample_id"].replace("/", "__")
        stories.append({"story_id": story_id, "split": split, "target_seconds": float(match[1]), "fps": int(match[2]),
                        "film_key": row["film_key"], "content": None, "inputs": {"shots": {"text": match[3]}}})
        boards.append({"story_id": story_id, "split": split, "storyboard": row["storyboard"]})
    if not stories:
        raise ValueError(f"{rows_path} has no shot-tier rows")
    return ({"evaluation_scope": "system", "request_version": f"planner-pool-{split}", "stories": stories},
            {"evaluation_scope": "renderer", "stories": boards})


def v15_prompt() -> str:
    source = subprocess.run(["git", "show", f"{V15_COMMIT}:src/mspc/data/planner_dataset.py"], check=True,
                            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2]).stdout
    if "PLANNER_CONTRACT_VERSION = 'storyboard_v18_action_expansion_request_v15'" not in source:
        raise ValueError(f"{V15_COMMIT} does not carry the v15 planner prompt")
    namespace: dict = {}
    start = source.index("PLANNER_SYSTEM = r'''")
    exec(source[start:source.index("'''", start + len("PLANNER_SYSTEM = r'''")) + 3], namespace)
    return namespace["PLANNER_SYSTEM"]


def plan(args: argparse.Namespace) -> None:
    from openai import OpenAI
    import mspc.data.planner_dataset as planner_dataset
    from mspc.benchmark.generation import production_request
    from mspc.inference.story_planner import plan_story

    if args.prompt == "v15":
        planner_dataset.PLANNER_SYSTEM = v15_prompt()  # planner_messages 调用时读模块里的提示词
    requests = json.loads(args.requests.read_text())
    stories = [story for story in requests["stories"] if story["split"] == args.split]
    if not stories:
        raise ValueError(f"no stories in split {args.split}")
    client = OpenAI(base_url=args.base_url, api_key="local-unauthed", timeout=1800, max_retries=0)
    sampling = sampling_preset(args.sampling)
    planner = {"model": args.model, "provenance": str(args.provenance)}
    tasks = []
    for story in stories:
        for tier in args.tiers:
            for seed in args.seeds:
                text = story["inputs"][tier]["text"]
                tasks.append({"story_id": story["story_id"], "input_tier": tier, "seed": seed,
                              "target_seconds": story["target_seconds"], "fps": story["fps"],
                              "story_request": production_request(story, tier, "system"),
                              "expected_shot_count": len(REQUEST_SHOT.findall(text)) if tier == "shots" else None,
                              "expected_cut_times_seconds": None})  # 与 story_requests_v4 相同：不给切点时间

    def run(task):
        output = args.output / task["input_tier"] / f"seed-{task['seed']}" / f"{task['story_id']}.json"
        if output.is_file() and json.loads(output.read_text()).get("status") in ("completed", "planning_failed"):
            return json.loads(output.read_text())["status"]
        output.parent.mkdir(parents=True, exist_ok=True)
        return plan_story(task, planner, sampling, output, client.chat.completions.create)["status"]

    with ThreadPoolExecutor(args.workers) as pool:
        statuses = list(pool.map(run, tasks))
    summary = {status: statuses.count(status) for status in sorted(set(statuses))}
    from mspc.data.planner_dataset import PLANNER_CONTRACT_VERSION
    (args.output / "summary.json").write_text(json.dumps({"prompt": args.prompt, "sampling": args.sampling, "contract": PLANNER_CONTRACT_VERSION if args.prompt == "current" else V15_COMMIT,
                                                          "model": args.model, "split": args.split, "seeds": args.seeds, "tiers": args.tiers,
                                                          "statuses": summary}, indent=1) + "\n")
    print(json.dumps(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    building = commands.add_parser("catalog")
    building.add_argument("--rows", type=Path, required=True)
    building.add_argument("--split", required=True)
    building.add_argument("--requests-output", type=Path, required=True)
    building.add_argument("--storyboards-output", type=Path, required=True)
    planning = commands.add_parser("plan")
    planning.add_argument("--requests", type=Path, required=True)
    planning.add_argument("--split", required=True)
    planning.add_argument("--prompt", choices=("current", "v15"), required=True,
                          help="current＝现行 PLANNER_SYSTEM；v15＝提交 fb079f61c 的提示词（11-11 规划器 SFT 训练时用的也是它）")
    planning.add_argument("--sampling", choices=("nonthinking", "thinking-low", "thinking-medium", "thinking-xhigh"), required=True,
                          help="nonthinking＝现行非思考官方采样；thinking-<档位>＝模型卡思考采样加 reasoning_effort")
    planning.add_argument("--base-url", required=True)
    planning.add_argument("--model", required=True)
    planning.add_argument("--provenance", type=Path, required=True)
    planning.add_argument("--seeds", type=int, nargs="+", required=True)
    planning.add_argument("--tiers", nargs="+", choices=("synopsis", "paragraph", "shots"), required=True)
    planning.add_argument("--output", type=Path, required=True)
    planning.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if args.command == "catalog":
        requests, boards = catalog(args.rows, args.split)
        for path, value in ((args.requests_output, requests), (args.storyboards_output, boards)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")
        print(json.dumps({"stories": len(requests["stories"])}))
    else:
        plan(args)


if __name__ == "__main__":
    main()
