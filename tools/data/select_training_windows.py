"""第 9 步：从全部窗里选出一批进训练的窗，并把它们的清单行与标注取出来交给训练。

选窗条件全部写在选窗配置里（``configs/data/train_selection_*.json``），逐条按顺序判，每个窗记下第一条没过的原因；
没选上的窗同样进台账（``ledger.jsonl``），不从上游清单里删。条件的出处见配置里各键旁的说明。

读 ``final/windows.jsonl``、``final/annotations.jsonl``、``final/eval-exclude.jsonl``、``final/scene_manifest.csv``；
写 ``final/selections/<name>/``：``ledger.jsonl``（每窗一行：选上或第一条没过的原因）、``annotations.jsonl``（选上的窗的标注与每镜文字）、
``scene_manifest.csv``（选上的窗的清单行，给潜变量缓存）、``summary.json``（分来源的漏斗）。

    python tools/data/select_training_windows.py --config configs/data/source_pipeline_v1.json \
        --selection configs/data/train_selection_first_batch_v1.json
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random


def decide(window, record, excluded, rule):
    """第一条没过的原因；全过返回 None。顺序即漏斗的顺序。"""
    if record is None or record["status"] == "no_answer":
        return "标注没返回"
    if record["status"] != "ok":
        return "标注不合法"
    if record["hard_filter"]:
        return "内容筛：" + record["hard_filter"][0]
    if window["key"] in excluded:
        return "与评测重复"
    if set(record["marks"]) & set(rule["drop_marks"]):
        return "标记：" + sorted(set(record["marks"]) & set(rule["drop_marks"]))[0]
    if record["class"] not in rule["classes"]:
        return "类别不在本批：" + record["class"]
    if window["scene"]["quality"]["min_laion_aesthetic"] < rule["min_laion_aesthetic"]:
        return "有镜的美学分低于本批的线"
    return None


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    a = p.parse_args()
    root = Path(json.loads(a.config.read_text())["root"])
    rule = json.loads(a.selection.read_text())
    final = root / "final"
    with (final / "annotations.jsonl").open() as handle:
        records = {record["key"]: record for record in map(json.loads, handle)}
    with (final / "eval-exclude.jsonl").open() as handle:
        excluded = {json.loads(line)["key"] for line in handle if line.strip()}
    ledger, passed = [], defaultdict(list)
    with (final / "windows.jsonl").open() as handle:
        for window in map(json.loads, handle):
            record = records.get(window["key"])
            reason = decide(window, record, excluded, rule)
            ledger.append({"key": window["key"], "source": window["source"], "shot_count": window["scene"]["shot_count"],
                           "class": record.get("class") if record else None, "decision": reason or "过了全部条件"})
            if reason is None:
                passed[record["class"]].append(window["key"])
    # 有占比上限的类别（对白类）按种子随机取到上限；上限按不受限类别的入选数折算。
    rng = random.Random(rule["seed"])
    free = sum(len(keys) for name, keys in passed.items() if name not in rule["class_share_cap"])
    chosen = set()
    for name, keys in sorted(passed.items()):
        cap = rule["class_share_cap"].get(name)
        if cap is not None:
            limit = int(free * cap / (1 - sum(rule["class_share_cap"].values())))
            keys = rng.sample(keys, min(limit, len(keys)))
        chosen.update(keys)
    if rule["max_windows"] and len(chosen) > rule["max_windows"]:  # 超过本批要的量：按种子随机取，各来源的比例随之保持
        chosen = set(rng.sample(sorted(chosen), rule["max_windows"]))
    out = final / "selections" / rule["name"]
    out.mkdir(parents=True, exist_ok=True)
    funnel = defaultdict(Counter)
    with (out / "ledger.jsonl").open("w") as handle:
        for row in ledger:
            if row["decision"] == "过了全部条件":
                row["decision"] = "选上" if row["key"] in chosen else "过了全部条件，超出本批的量或类别占比"
            funnel[row["source"]][row["decision"]] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "annotations.jsonl").open("w") as handle:
        for key in sorted(chosen):
            record = records[key]
            handle.write(json.dumps({name: record[name] for name in ("key", "source", "shot_count", "class", "marks", "people", "clear_people",
                                                                      "recurring_people", "annotation", "shot_texts")}, ensure_ascii=False) + "\n")
    with (final / "scene_manifest.csv").open(newline="") as source, (out / "scene_manifest.csv").open("w", newline="") as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        rows = 0
        for row in reader:
            if row["window_key"] in chosen:
                writer.writerow(row)
                rows += 1
    if rows != len(chosen):
        raise ValueError(f"manifest holds {rows} of the {len(chosen)} selected windows; re-export the manifest first")
    picked = [row for row in ledger if row["key"] in chosen]
    summary = {"selection": rule["name"], "windows": len(ledger), "selected": len(chosen),
               "selected_by_source": dict(Counter(row["source"] for row in picked)),
               "selected_by_class": dict(Counter(row["class"] for row in picked)),
               "selected_by_shot_count": dict(sorted(Counter(row["shot_count"] for row in picked).items())),
               "funnel_by_source": {source: dict(counts.most_common()) for source, counts in sorted(funnel.items())}}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
