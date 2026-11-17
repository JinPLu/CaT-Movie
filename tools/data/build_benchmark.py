"""从按片留出的第 18 版验证集重建评测集（每片至多 3 段）。

候选取渲染器修订训练的验证成员（按片与训练集分开，规划器 SFT 与修订渲染器都没在这些片上训练）：同一场景里相邻的块拼回
连续镜头，按训练同一质量门的 `usable` 断开，贪心切出互不重叠、6–12 镜、≤60 秒的窗口（与旧长故事评测集同口径，
渲染器一段至多 4 镜，所以每个窗口至少两段）。每个窗口按第 18 版块标注合同重新标注（跨块窗口需要统一编号），
评测口径验收后按内容筛选（至少一处镜内变化、一个跨镜人物、一个动作镜），每片至多 3 段，按片分开发／测试。
三档请求与训练同法生成（逐镜档 `shot_script_text`，段落与梗概取标注 story，制作要求行由 `production_script` 加），
共同分镜表直接取标注。

    python tools/data/build_benchmark.py select   --out OUT --members .../renderer-revised-supervision-20261022/inputs/val.jsonl
    python tools/data/build_benchmark.py prepare  --out OUT
    <DashScope 密钥一行> | python tools/data/build_benchmark.py annotate --out OUT --budget-cny 3
    python tools/data/build_benchmark.py validate --out OUT          # 不合法的块：retire --round N 后再 annotate
    python tools/data/build_benchmark.py build    --out OUT --requests configs/benchmark/story_requests_v3.json \
        --storyboards configs/benchmark/benchmark_storyboards_v3.json
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import threading

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "data"))
sys.path.insert(0, str(REPO / "tools" / "analysis"))

CONTRACT = REPO / "configs/data/block_annotation_v18.json"
PIPELINE = REPO / "configs/data/source_pipeline_v1.json"
FPS = 16
MIN_SHOTS, MAX_SHOTS, MAX_SECONDS, PER_FILM, DEV_STORIES, SEED = 6, 12, 60, 3, 30, 42
# 实时价（北京，qwen3.7-flash，输入不超过 32k 一档），每百万 token；思考按输出计。
INPUT_CNY, OUTPUT_CNY = 0.2, 0.8


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")
    temp.replace(path)


def select(out, members_path):
    """相邻块拼回连续镜头 → 按 usable 断开 → 贪心切 6–12 镜、≤60 秒、互不重叠的窗口。"""
    from collections import defaultdict
    scenes = defaultdict(list)
    for member in map(json.loads, Path(members_path).open()):
        scenes[member["scene"]["scene_id"].split(":block")[0]].append(member)
    runs = []
    for blocks in scenes.values():
        blocks.sort(key=lambda b: b["shots"][0]["source_shot_index"])
        current = []
        for block in blocks:
            shots = [(block, i) for i in range(len(block["shots"]))]
            last = current[-1][0]["shots"][current[-1][1]]["source_shot_index"] if current else None
            if current and last + 1 == block["shots"][0]["source_shot_index"]:
                current += shots
            else:
                runs.append(current)
                current = shots
        runs.append(current)
    usable = []
    for run in runs:
        current = []
        for block, i in run:
            if block["shots"][i]["usable"]:
                current.append((block, i))
            else:
                usable.append(current)
                current = []
        usable.append(current)
    windows = []
    for run in usable:
        start = 0
        while len(run) - start >= MIN_SHOTS:
            size = next((n for n in range(min(MAX_SHOTS, len(run) - start), MIN_SHOTS - 1, -1)
                         if sum(b["shots"][i]["render_frames"] for b, i in run[start:start + n]) / FPS <= MAX_SECONDS), None)
            if size is None:
                start += 1
                continue
            part = run[start:start + size]
            first = part[0][0]
            scene = first["scene"]
            assert all(b["scene"]["source_video_path"] == scene["source_video_path"] and b["scene"]["crop_box"] == scene["crop_box"]
                       for b, _ in part), "窗口跨了源片或裁框"
            windows.append({"story_id": f"{first['key']}-shot{part[0][1] + 1:02d}", "film_key": first["film_key"],
                            "source": first["source"], "source_video_path": scene["source_video_path"], "fps": scene["fps"],
                            "crop_box": scene["crop_box"], "container_preroll_frames": scene["container_preroll_frames"] or 0,
                            "shots": [{"block_key": b["key"], "block_shot": i + 1, **{k: b["shots"][i][k] for k in
                                       ("start_frame", "end_frame", "render_frames", "source_shot_index")}} for b, i in part]})
            start += size
    write_json(out / "windows.json", windows)
    print(json.dumps({"windows": len(windows), "films": len({w["film_key"] for w in windows}),
                      "cross_block": sum(len({s["block_key"] for s in w["shots"]}) > 1 for w in windows)}))


def shot_proxy(window, shot, target, ffmpeg):
    """一镜：源帧区间按裁框裁切、缩到 832×480、取 2 fps。"""
    x0, y0, x1, y1 = window["crop_box"]
    start, end = shot["start_frame"] + window["container_preroll_frames"], shot["end_frame"] + window["container_preroll_frames"]
    # 先按时间粗定位到镜头前 2 秒（输入端 seek，免从片头顺序解码），再按帧号精确截取。
    lead = max(0, int(start - 2 * window["fps"]))
    filters = (f"select=between(n\\,{start - lead}\\,{end - 1 - lead}),setpts=N/({window['fps']}*TB),"
               f"crop={x1 - x0}:{y1 - y0}:{x0}:{y0},scale=832:480,fps=2")
    subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y", "-ss", f"{lead / window['fps']:.6f}", "-i", window["source_video_path"],
                    "-vf", filters, "-an",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(target)], check=True)


def frame_count(path, ffprobe):
    out = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(path)], check=True, capture_output=True, text=True)
    return int(out.stdout.strip())


def prepare(out):
    cfg = read_json(PIPELINE)
    ffmpeg, ffprobe = cfg["encode_ffmpeg"], cfg["ffprobe"]

    def one(window):
        key = window["story_id"]
        folder = out / "proxy" / key
        folder.mkdir(parents=True, exist_ok=True)
        parts, stamps, cursor = [], [], 0
        for number, shot in enumerate(window["shots"]):
            part = folder / f"shot-{number:02d}.mp4"
            if not part.exists():
                shot_proxy(window, shot, part, ffmpeg)
            frames = frame_count(part, ffprobe)
            if frames < 1:
                raise ValueError(f"{key} shot {number}: empty proxy")
            parts.append(part)
            stamps.append({"shot": number, "start": cursor / 2, "end": (cursor + frames) / 2})
            cursor += frames
        video = out / "proxy" / f"{key}.mp4"
        listing = folder / "parts.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", str(video)], check=True)
        return {"key": key, "film_key": window["film_key"],
                "frames": [s["render_frames"] for s in window["shots"]],
                "proxy": {"video": str(video), "fps": 2, "proxy_width": 832, "proxy_height": 480,
                          "crop_box": window["crop_box"], "shot_timestamps": stamps}}

    with ThreadPoolExecutor(24) as pool:
        rows = list(pool.map(one, read_json(out / "windows.json")))
    (out / "blocks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps({"blocks": len(rows), "shots": sum(len(r["frames"]) for r in rows),
                      "proxy_seconds": sum(r["proxy"]["shot_timestamps"][-1]["end"] for r in rows)}))


def annotate(out, budget, workers):
    from block_length_annotation_trial import ask
    key = sys.stdin.readline().strip()
    assert key, "DashScope 密钥应从标准输入给一行"
    contract = read_json(CONTRACT)
    rows = [json.loads(line) for line in (out / "blocks.jsonl").open()]
    lock, spent = threading.Lock(), [0.0]
    for path in [*(out / "answers").glob("*.json"), *out.glob("answers-round-*/*.json")]:  # 重问前移走的原答复也已计费
        spent[0] += read_json(path)["cost_cny"]
    # 预留：输入按代理片秒数 × 每秒上界 2,000 token，输出与思考按 40,000 token；超硬顶不再发。
    reserve = lambda row: (row["proxy"]["shot_timestamps"][-1]["end"] * 2000 + 4000) * INPUT_CNY / 1e6 + 40000 * OUTPUT_CNY / 1e6

    def one(row):
        target = out / "answers" / f"{row['key']}.json"
        if target.exists():
            return
        with lock:
            if spent[0] + reserve(row) > budget:
                print(json.dumps({"skipped_budget": row["key"]}), flush=True)
                return
            spent[0] += reserve(row)
        times = [(s["shot"] + 1, s["start"], s["end"]) for s in row["proxy"]["shot_timestamps"]]
        try:
            result = ask(contract, key, Path(row["proxy"]["video"]), times, {})
        except Exception as error:  # 连接与服务错误：不重发，记下来人工处理
            with lock:
                spent[0] -= reserve(row)
            write_json(out / "errors" / f"{row['key']}.json", {"key": row["key"], "error": str(error)[:500]})
            print(json.dumps({"error": row["key"]}), flush=True)
            return
        usage = result["usage"] or {}
        cost = usage.get("prompt_tokens", 0) * INPUT_CNY / 1e6 + usage.get("completion_tokens", 0) * OUTPUT_CNY / 1e6
        with lock:
            spent[0] += cost - reserve(row)
        write_json(target, {"key": row["key"], **result, "cost_cny": cost})
        print(json.dumps({"key": row["key"], "finish": result["finish_reason"], "cost_cny": round(cost, 5)}), flush=True)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, rows))
    print(json.dumps({"spent_cny": round(spent[0], 4)}))


def answer_rounds(out, key):
    """同一块各轮已计费的答复，按提问先后：answers-round-1 是第 0 轮（重问前移走），answers 是最后一轮。"""
    rounds = sorted(out.glob("answers-round-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
    return [p / f"{key}.json" for p in [*rounds, out / "answers"] if (p / f"{key}.json").exists()]


def judge_block(row, answer, contract):
    from validate_annotations import block_record
    from mspc.data.planner_dataset import production_script
    from mspc.data.shot_text import shot_script_text
    from mspc.data.storyboard import from_annotation
    record = {"key": row["key"], **block_record(answer["answer"], contract, len(row["frames"])),
              "finish_reason": answer["finish_reason"], "cost_cny": answer["cost_cny"]}
    # 评测集按评测实际消费的口径验收：结构、编号、镜数不合法才作废；写法规则（如持有变空须写
    # 可见终态）只服务规划器监督与渲染器镜末句，照样记录，不作废整块。
    wording = {e["path"] + ": " + e["reason"] for e in record.get("wording_errors") or []}
    structural = [m for m in record.get("block_errors") or [] if m not in wording]
    structural += [m for ms in (record.get("shot_errors") or {}).values() for m in ms if m not in wording]
    if record["annotation"] is not None and not structural:
        # 评测集真正消费的是分镜表与逐镜请求：转换不过（如 present 里多出合同没有的字段）同样作废
        try:
            board = from_annotation(record["annotation"], row["frames"], start=0, end=len(row["frames"]))
            production_script(script=shot_script_text(board), target_seconds=sum(row["frames"]) / FPS, fps=FPS)
        except (ValueError, AssertionError) as error:
            structural.append("storyboard: " + " ".join(str(error).split()[:16]))
    record["benchmark_status"] = "ok" if record["annotation"] is not None and not structural else "invalid"
    record["benchmark_errors"] = structural
    return record


def validate(out):
    """每块取按提问先后第一个通过评测口径的答复（早轮因旧的写法规则被重问的，不因后来重问反而丢掉）。"""
    contract = read_json(CONTRACT)
    records, counts = [], {}
    for row in map(json.loads, (out / "blocks.jsonl").open()):
        tried = [judge_block(row, read_json(path), contract) | {"round": n} for n, path in enumerate(answer_rounds(out, row["key"]))]
        passing = [r for r in tried if r["benchmark_status"] == "ok"]
        record = passing[0] if passing else (tried[-1] if tried else
                 {"key": row["key"], "status": "no_answer", "benchmark_status": "invalid", "benchmark_errors": ["no answer"]})
        record["rounds_tried"] = len(tried)
        counts[record["benchmark_status"]] = counts.get(record["benchmark_status"], 0) + 1
        records.append(record)
    (out / "annotations.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    paid = [*(out / "answers").glob("*.json"), *out.glob("answers-round-*/*.json")]
    summary = {"blocks": len(records), "status": counts, "cost_cny_total": round(sum(read_json(p)["cost_cny"] for p in paid), 4),
               "round_used": {str(n): sum(r.get("round") == n for r in records) for n in range(5)},
               "invalid": {r["key"]: r["benchmark_errors"] for r in records if r["benchmark_status"] != "ok"},
               "wording_only": sorted(r["key"] for r in records if r["benchmark_status"] == "ok" and r.get("status") != "ok")}
    write_json(out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("blocks", "status", "cost_cny_total", "round_used")}, ensure_ascii=False))


def retire_invalid(out, round_number):
    """整块重发前，把校验不合法的原答复移到 answers-round-N（保留不删），再跑 annotate 只补这些块。
    评测集不能像训练池那样删坏镜，所以局部不合法也整块重问。"""
    summary = read_json(out / "summary.json")
    keep = out / f"answers-round-{round_number}"
    keep.mkdir(exist_ok=False)
    for key in summary["invalid"]:  # 只重问评测口径下不合法的块
        source = out / "answers" / f"{key}.json"
        if source.exists():
            source.rename(keep / source.name)
    print(json.dumps({"retired": sorted(p.stem for p in keep.glob("*.json"))}))


def content(record, frames):
    """内容要求：至少一处镜内变化、至少一个跨镜人物、至少一个动作镜（标注的 activity）。"""
    from collections import Counter
    annotation = record["annotation"]
    shots = annotation["shots"]
    present = Counter(p["name"] for s in shots for p in s.get("present", []))
    return {"changes": sum(len(s.get("changes") or []) for s in shots),
            "change_kinds": sorted({c["what"] for s in shots for c in s.get("changes") or []}),
            "recurring_characters": sum(present[c["name"]] >= 2 for c in annotation["characters"]),
            "recurring_objects": sum(present[o["name"]] >= 2 for o in annotation["objects"]),
            "places": len({s.get("place") for s in shots}),
            "action_shots": sum(s.get("activity") == "action_or_state_change" for s in shots),
            "shots": len(frames), "seconds": sum(frames) / FPS}


def story_assets(key, film, frames, annotation, split, facts):
    """三档请求与共同计划的唯一转换；扩展只追加成员，不重写冻结故事。"""
    from mspc.data.planner_dataset import production_script
    from mspc.data.shot_text import shot_script_text
    from mspc.data.storyboard import SEGMENT_MAX_FRAMES, from_annotation
    shot_max = max(range(1, SEGMENT_MAX_FRAMES + 1, 4))
    board = from_annotation(annotation, frames, start=0, end=len(frames))
    inputs = {"synopsis": {"text": annotation["story"]["summary"], "expected_shot_count": None, "expected_cut_times_seconds": None},
              "paragraph": {"text": annotation["story"]["paragraph"], "expected_shot_count": None, "expected_cut_times_seconds": None},
              "shots": {"text": shot_script_text(board), "expected_shot_count": len(frames), "expected_cut_times_seconds": None}}
    production_script(script=inputs["shots"]["text"], target_seconds=sum(frames) / FPS, fps=FPS)
    story = {"story_id": key, "split": split, "target_seconds": sum(frames) / FPS, "fps": FPS, "film_key": film,
             "content": facts, "inputs": inputs}
    capped = [min(f, shot_max) for f in frames]
    common = {"story_id": key, "split": split, "fps": FPS, "target_seconds": sum(capped) / FPS,
              "storyboard": from_annotation(annotation, capped, start=0, end=len(capped)),
              "inputs": {"shots": inputs["shots"]}}
    return story, common


def append_test(out, members_path, base_requests_path, base_boards_path, requests_path, storyboards_path):
    """将已筛选合格、仅被影片上限排除的成员追加到 test，原成员与划分不动。"""
    from collections import Counter
    base = read_json(base_requests_path)
    common = read_json(base_boards_path)
    members = read_json(members_path)["additional_stories"]
    existing = {s["story_id"] for s in base["stories"]}
    added_ids = [m["story_id"] for m in members]
    if len(set(added_ids)) != len(added_ids) or existing.intersection(added_ids):
        raise ValueError("append members must be unique and absent from frozen requests")
    if {s["story_id"] for s in common["stories"]} != existing:
        raise ValueError("frozen requests and storyboards differ")
    blocks = {r["key"]: r for r in map(json.loads, (out / "blocks.jsonl").open())}
    records = {r["key"]: r for r in map(json.loads, (out / "annotations.jsonl").open())}
    added, boards = [], []
    for member in members:
        key = member["story_id"]
        block, record = blocks[key], records[key]
        if member["split"] != "test" or base["authoring"]["dropped"].get(key) != "per_film_cap":
            raise ValueError(f"member is not an approved test addition: {key}")
        if record["benchmark_status"] != "ok" or len(record["annotation"]["shots"]) != len(block["frames"]):
            raise ValueError(f"annotation is not valid for this block: {key}")
        facts = content(record, block["frames"])
        if not (facts["changes"] and facts["recurring_characters"] and facts["action_shots"]):
            raise ValueError(f"content filter failed: {key}")
        if (member["film_key"] != block["film_key"] or member["shot_count"] != facts["shots"]
                or member["target_seconds"] != facts["seconds"]):
            raise ValueError(f"approved member differs from source: {key}")
        story, board = story_assets(key, block["film_key"], block["frames"], record["annotation"], "test", facts)
        if story["target_seconds"] != board["target_seconds"]:
            raise ValueError(f"test addition would silently truncate a shot: {key}")
        added.append(story)
        boards.append(board)
    authoring = {"source": base["authoring"]["source"],
                 "rule": "保留冻结v3全部故事、请求、共同计划与划分；仅将原per_film_cap排除的合格故事追加到test；开发／测试允许同片但源帧不重叠",
                 "base_requests": str(base_requests_path), "base_storyboards": str(base_boards_path),
                 "base_authoring": base["authoring"], "test_additions": str(members_path),
                 "added_story_ids": added_ids}
    write_json(Path(requests_path), {**base, "request_version": "story_requests_v4", "authoring": authoring,
                                    "stories": base["stories"] + added})
    write_json(Path(storyboards_path), {**common, "storyboard_version": "benchmark_storyboards_v4", "authoring": authoring,
                                       "stories": common["stories"] + boards})
    print(json.dumps({"stories": len(base["stories"]) + len(added), "added": len(added),
                      "splits": dict(Counter(s["split"] for s in base["stories"] + added))}))


def build(out, requests_path, storyboards_path):
    import random
    from collections import defaultdict
    blocks = {r["key"]: r for r in map(json.loads, (out / "blocks.jsonl").open())}
    records = {r["key"]: r for r in map(json.loads, (out / "annotations.jsonl").open())}
    chosen, dropped = defaultdict(list), {}
    for key, block in blocks.items():
        record = records[key]
        if record["benchmark_status"] != "ok":
            dropped[key] = "annotation_invalid"
            continue
        facts = content(record, block["frames"])
        if not (facts["changes"] >= 1 and facts["recurring_characters"] >= 1 and facts["action_shots"] >= 1):
            dropped[key] = "content"
            continue
        chosen[block["film_key"]].append((key, facts))
    kept = []
    for film, items in chosen.items():
        items.sort(key=lambda item: (-item[1]["changes"], item[0]))  # 每片至多 3 段，先取变化多的
        kept += [(film, key, facts) for key, facts in items[:PER_FILM]]
        dropped.update({key: "per_film_cap" for key, _ in items[PER_FILM:]})
    films = sorted({film for film, _, _ in kept})
    random.Random(SEED).shuffle(films)
    dev, count = set(), 0
    for film in films:  # 按片分：随机顺序累加到开发集至少 DEV_STORIES 个故事
        if count >= DEV_STORIES:
            break
        dev.add(film)
        count += sum(f == film for f, _, _ in kept)
    stories, boards = [], []
    for film, key, facts in sorted(kept, key=lambda item: item[1]):
        frames = blocks[key]["frames"]
        annotation = records[key]["annotation"]
        split = "dev" if film in dev else "test"
        story, board = story_assets(key, film, frames, annotation, split, facts)
        stories.append(story)
        boards.append(board)
    authoring = {"source": "第 18 版质量池验证集（按片留出）重切窗口并按第 18 版合同重标（tools/data/build_benchmark.py）",
                 "rule": f"{MIN_SHOTS}–{MAX_SHOTS} 镜、≤{MAX_SECONDS} 秒、训练同一质量门；至少一处镜内变化、一个跨镜人物、一个动作镜；"
                         f"每片至多 {PER_FILM} 段；按片随机（seed {SEED}）分开发集至少 {DEV_STORIES} 个故事",
                 "dropped": dropped}
    Path(requests_path).write_text(json.dumps({"evaluation_scope": "system", "request_version": "story_requests_v3",
                                                "authoring": authoring, "stories": stories}, ensure_ascii=False, indent=2) + "\n")
    Path(storyboards_path).write_text(json.dumps({"evaluation_scope": "renderer", "authoring": {**authoring,
                                                  "storyboard": "分镜表即标注；单镜截到 225 帧"}, "stories": boards}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"stories": len(stories), "dev": sum(s["split"] == "dev" for s in stories),
                      "test": sum(s["split"] == "test" for s in stories), "films": len(films),
                      "dropped": dict(__import__("collections").Counter(dropped.values()))}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("select", "prepare", "annotate", "validate", "retire", "build", "append-test"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--members", type=Path)
    p.add_argument("--budget-cny", type=float)
    p.add_argument("--workers", type=int)
    p.add_argument("--round", type=int)
    p.add_argument("--requests", type=Path)
    p.add_argument("--storyboards", type=Path)
    p.add_argument("--base-requests", type=Path)
    p.add_argument("--base-storyboards", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "select":
        select(a.out, a.members)
    elif a.stage == "prepare":
        prepare(a.out)
    elif a.stage == "annotate":
        annotate(a.out, a.budget_cny, a.workers)
    elif a.stage == "validate":
        validate(a.out)
    elif a.stage == "retire":
        retire_invalid(a.out, a.round)
    elif a.stage == "append-test":
        append_test(a.out, a.members, a.base_requests, a.base_storyboards, a.requests, a.storyboards)
    else:
        build(a.out, a.requests, a.storyboards)


if __name__ == "__main__":
    main()
