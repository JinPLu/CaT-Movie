#!/usr/bin/env python3
"""论文规划器图的数据（写作蓝图 11-12：方法图 3 (b) 字段分工；实验规划器小节的扩写图与稳定性图）。

只读已有产物，不调用模型：
- 字段可推出比例：``outputs/experiments/planner-state-replay-audit-20261111/result.json`` 的 ``gt_self_consistency``；
- 文字层读数：``outputs/experiments/planner-unified-text-table-20261112`` 的计划记录与 DeepSeek-V4.1-Flash 判读
  （``judge-deepseek-v2/``）。编译通过的计划用编出文字的判读，没编译通过的计。

区间：同一行两个种子在每个故事上的计数先相加，再按故事有放回重采样（默认 10,000 次，种子 0），在合并计数上重算比例，
取 2.5%／97.5% 分位。判官同批重判差另在图注写明，不并进区间。

用法::

    PYTHONPATH=src python tools/analysis/planner_paper_readouts.py --root outputs/experiments/planner-unified-text-table-20261112 \\
        --replay outputs/experiments/planner-state-replay-audit-20261111/result.json --out-dir <论文>/figures/source/data/source
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

TIERS = ("synopsis", "paragraph", "shots")


def story_counts(root: Path, row: str, compiler: str, seeds, label: str | None = None) -> dict:
    """{档: {故事: {键: [通过, 适用]}}}；键为 events、states、contradicted（计数, 计划数）、compiled（通过, 计划数）。"""
    label = label or row
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for seed in seeds:
        raw_path = root / "judge-deepseek-v2" / f"{label}-s{seed}-raw.json"
        comp_path = root / "judge-deepseek-v2" / f"{label}-{compiler}-s{seed}-compiled.json"
        raw = json.loads(raw_path.read_text())["plans"] if raw_path.exists() else {}
        comp = json.loads(comp_path.read_text())["plans"] if comp_path.exists() else {}
        for tier in TIERS:
            directory = root / "records" / row / compiler / tier / f"seed-{seed}"
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.json")):
                story, record = path.stem, json.loads(path.read_text())
                ok = record["status"] == "completed"
                c = out[tier][story]
                c["compiled"][0] += ok
                c["compiled"][1] += 1
                judged = comp.get(f"{tier}/{story}/compiled") if ok else raw.get(f"{tier}/{story}/raw")
                if not judged or not judged.get("measured"):
                    continue
                for key, field in (("events", "event_kept"), ("states", "state_kept")):
                    c[key][0] += judged[field]["pass"]
                    c[key][1] += judged[field]["applicable"]
                c["contradicted"][0] += judged["contradicted"]
                c["contradicted"][1] += 1
                c["fabricated"][0] += judged["fabricated_conflicts"]
                c["fabricated"][1] += 1
    return out


def summarize(counts: dict, rng: np.random.Generator, resamples: int) -> dict:
    out = {}
    for tier, stories in counts.items():
        ids = sorted(stories)
        block = {"stories": len(ids)}
        for key in ("events", "states", "contradicted", "fabricated", "compiled"):
            arr = np.array([stories[s][key] for s in ids], dtype=float)
            if arr[:, 1].sum() == 0:
                continue
            scale = 100.0 if key in ("events", "states", "compiled") else 1.0
            value = scale * arr[:, 0].sum() / arr[:, 1].sum()
            draws = rng.integers(0, len(ids), size=(resamples, len(ids)))
            sums = arr[draws].sum(axis=1)
            ratios = scale * sums[:, 0] / np.maximum(sums[:, 1], 1)
            block[key] = {"value": round(float(value), 2), "ci95": [round(float(np.percentile(ratios, 2.5)), 2),
                                                                    round(float(np.percentile(ratios, 97.5)), 2)],
                          "pass": int(arr[:, 0].sum()), "applicable": int(arr[:, 1].sum())}
        out[tier] = block
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    source = str(args.root.resolve())

    replay = json.loads(args.replay.read_text())["gt_self_consistency"]
    fields = [("outfit", "Outfit", "code carries"), ("holding", "Holding", "planner writes, code fills gaps"),
              ("state", "Visible state", "planner writes"), ("posture", "Posture", "planner writes")]
    derivability = {"fields": [], "source": str(args.replay.resolve()),
                    "note": "share of later (shot, character) entries whose value follows from the first appearance plus "
                            "recorded changes in the annotated storyboards; dev30 356 and test92 1,024 entries"}
    for key, name, owner in fields:
        row = {"field": name, "owner": owner}
        for split in ("dev", "test"):
            c = replay[f"{split}:{key}:set"]
            row[split] = {"derivable": c["derivable"], "n": c["n"], "percent": round(100 * c["derivable"] / c["n"], 1)}
        derivability["fields"].append(row)

    expansion = {"source": source, "judge": "DeepSeek-V4.1-Flash reading the compiled per-shot text",
                 "rows": {"ours": summarize(story_counts(args.root, "public-9b-v17", "simplified", (42, 43)), rng, args.resamples),
                          "request_only": summarize(story_counts(args.root, "request-only", "fixed", (42,)), rng, args.resamples)}}

    stability = {"source": source, "judge": expansion["judge"], "models": {}, "prompts": {}, "compiler": {}, "correction": {}}
    for name, row, compiler in (("qwen35_9b_v17", "public-9b-v17", "simplified"), ("qwen35_9b_v15", "public-9b-current", "simplified"),
                                ("qwen38_27b", "qwen38-27b", "simplified"), ("sft_step4014", "sft-step4014", "simplified")):
        stability["models"][name] = summarize(story_counts(args.root, row, compiler, (42, 43)), rng, args.resamples)
    for name, row in (("v15", "public-9b-current"), ("v16", "public-9b-v16"), ("v16_rerun", "public-9b-v16-seqs64"),
                      ("v17", "public-9b-v17")):
        stability["prompts"][name] = summarize(story_counts(args.root, row, "simplified", (42,)), rng, args.resamples)
    for name, compiler in (("old", "old"), ("ours", "simplified")):
        stability["compiler"][name] = summarize(story_counts(args.root, "public-9b-current", compiler, (42, 43)), rng, args.resamples)
    stability["compiler"]["raw"] = {}
    for tier in TIERS:  # 原答层：同一批原答的判读（与编译器无关）
        judged = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for seed in (42, 43):
            raw = json.loads((args.root / "judge-deepseek-v2" / f"public-9b-current-s{seed}-raw.json").read_text())["plans"]
            for key, v in raw.items():
                t, story, _ = key.split("/")
                if t != tier or not v.get("measured"):
                    continue
                judged[story]["contradicted"][0] += v["contradicted"]; judged[story]["contradicted"][1] += 1
                judged[story]["fabricated"][0] += v["fabricated_conflicts"]; judged[story]["fabricated"][1] += 1
        stability["compiler"]["raw"].update(summarize({tier: judged}, rng, args.resamples))
    for attempt in ("first", "final"):
        row = "public-9b-current-first" if attempt == "first" else "public-9b-current"
        c = story_counts(args.root, row, "simplified", (42, 43))
        stability["correction"][attempt] = {t: {"compiled": int(sum(s["compiled"][0] for s in c[t].values())),
                                               "plans": int(sum(s["compiled"][1] for s in c[t].values()))} for t in TIERS}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    info = args.out_dir / "11_planner_information.json"
    data = json.loads(info.read_text())
    data.pop("b_planner_adherence", None)
    data["b_field_derivability"] = derivability
    info.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    (args.out_dir / "12_planner_expansion.json").write_text(json.dumps(expansion, ensure_ascii=False, indent=1))
    (args.out_dir / "13_planner_stability.json").write_text(json.dumps(stability, ensure_ascii=False, indent=1))
    print(json.dumps({"expansion": {t: {k: v["value"] for k, v in b.items() if isinstance(v, dict)}
                                    for t, b in expansion["rows"]["ours"].items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
