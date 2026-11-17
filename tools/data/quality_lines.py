"""定质量线：候选线先取自工具官方与近邻工作，再看每条候选线两侧的实物来选，线经确认后写进配置的 quality_lines 才生效。

定线的先后：一看工具官方怎么推荐、近邻工作怎么筛（后者很重要），二才是看图选线。所以这里的档位按
官方与近邻用过的线切，不按我们自己的分位数。

    python tools/data/quality_lines.py table  --config CFG                       # 每条候选线在各来源会筛掉多少
    python tools/data/quality_lines.py strata --config CFG --per-band 20 --workers 16   # 每档取若干镜导出原图
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import subprocess

# LAION 美学分的候选线（都是「低于则筛」）：
#   4.0 EchoShot（多镜头）；4.3、4.5 Goku（4.5 也是 LAION 官方最低一档子集）；4.75 Open-Sora-Plan（OpenHumanVid 4.8 相邻，并入）；
#   5.0 Open-Sora 示例、LAION 官方 5+ 子集。
LAION_LINES = [4.0, 4.3, 4.5, 4.75, 5.0]
# DOVER 清晰度头：近邻给的数值线（OpenHumanVid 0.09、Open-Sora-Plan 0）原文没写量纲，而原始分在官方自带的公开集预测里几乎全为负，
# 不能照搬。官方脚本 rescale_results 的做法是对公开集的预测分做标准化；这里用仓库自带的 LSVQ 测试集预测
# （dover_predictions/val-ltest.pkl 的 fragments，7,186 条：均值 −0.1048，标准差 0.0442）。Open-Sora-Plan 的清晰度线筛掉约 5%，可作幅度参照。
DOVER_TECHNICAL_MEAN, DOVER_TECHNICAL_STD = -0.1048, 0.0442
TECHNICAL_Z_LINES = [-2.0, -1.5, -1.0, -0.5, 0.0]
SCORES = {"dover_technical": lambda r: r["dover"]["technical_logit"],
          "laion_aesthetic": lambda r: r["laion"]["mean"]}
# 取样的档位：相邻两条候选线之间为一档，最后一档是线以上的对照。DOVER 美学头不算（近邻美学都用 LAION）。
LINES = {"laion_aesthetic": LAION_LINES,
         "dover_technical": [round(DOVER_TECHNICAL_MEAN + z * DOVER_TECHNICAL_STD, 4) for z in TECHNICAL_Z_LINES]}
REFERENCE_WIDTH = {"laion_aesthetic": 0.5, "dover_technical": 0.5 * DOVER_TECHNICAL_STD}
SEED = 20261012


def load(root):
    """打分成功的镜，补上来源名（scenes.jsonl 里的 source）。"""
    source_of = {}
    with (root / "scenes.jsonl").open() as handle:
        for line in handle:
            scene = json.loads(line)
            source_of[scene["sid"]] = scene["row"]["source"]
    by_source = defaultdict(list)
    for path in sorted((root / "scores").glob("s*of*.jsonl")):
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["status"] == "ok":
                    by_source[source_of[row["sid"]]].append(row)
    return by_source


def table(root):
    """每条候选线在各来源筛掉的比例（低于线的镜占比）。"""
    result = {}
    for source, rows in sorted(load(root).items()):
        result[source] = {"shots": len(rows)}
        for name, lines in LINES.items():
            values = [SCORES[name](r) for r in rows]
            result[source][name] = {str(line): f"{100 * sum(v < line for v in values) / len(values):.1f}%" for line in lines}
    result["dover_technical_lines_are_z_vs_lsvq_test"] = dict(zip(map(str, LINES["dover_technical"]), TECHNICAL_Z_LINES))
    print(json.dumps(result, ensure_ascii=False, indent=1))


def export_frame(task):
    row, image, ffmpeg = task
    start, end = row["segment"]
    middle = (start + end - 1) // 2
    # 顺序解码到这一帧，帧号与流程其余部分同一把尺；只看图用，JPEG 质量 2。
    result = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y", "-threads", "1", "-i", row["source_video_path"], "-map", "0:v:0",
                             "-vf", f"select=eq(n\\,{middle})", "-frames:v", "1", "-fps_mode", "passthrough", "-q:v", "2", str(image)],
                            capture_output=True, text=True)
    return None if result.returncode == 0 and image.exists() else f"{row['shot_key']}: {result.stderr[-300:]}"


def strata(cfg, root, per_band, workers):
    out = root / "quality-lines"
    rng = random.Random(SEED)
    picks, tasks = [], []
    for source, rows in sorted(load(root).items()):
        for name, lines in LINES.items():
            read = SCORES[name]
            edges = [float("-inf"), *lines, lines[-1] + REFERENCE_WIDTH[name]]
            for low, high in zip(edges, edges[1:]):
                band = [row for row in rows if low <= read(row) < high]
                label = f"below_{high}" if low == float("-inf") else f"{low}_to_{high}"
                folder = out / source / name / label
                folder.mkdir(parents=True, exist_ok=True)
                for row in rng.sample(band, min(per_band, len(band))):
                    image = folder / f"{row['shot_key']}.jpg"
                    picks.append({"source": source, "score": name, "band": label, "band_size": len(band), "source_shots": len(rows),
                                  "value": read(row), "shot_key": row["shot_key"],
                                  "scene_id": row["scene_id"], "source_video_path": row["source_video_path"], "segment": row["segment"],
                                  "dover_technical": row["dover"]["technical_logit"],
                                  "laion_aesthetic": row["laion"]["mean"], "image": str(image), "proxy": row["proxy"]["video"]})
                    if not image.exists():
                        tasks.append((row, image, cfg["ffmpeg"]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        failures = [message for message in pool.map(export_frame, tasks) if message]
    with (out / "strata.jsonl").open("w") as handle:
        for pick in picks:
            handle.write(json.dumps(pick, ensure_ascii=False) + "\n")
    print(json.dumps({"picks": len(picks), "exported": len(tasks) - len(failures), "failed": failures[:10], "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["strata", "table"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--per-band", type=int)
    p.add_argument("--workers", type=int)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    if a.command == "table":
        table(Path(config["root"]))
    else:
        if None in (a.per_band, a.workers):
            p.error("strata needs --per-band and --workers")
        strata(config, Path(config["root"]), a.per_band, a.workers)
