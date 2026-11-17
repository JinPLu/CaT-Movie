import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT / "tools/data"))
from select_training_windows import decide  # noqa: E402
from validate_annotations import classify  # noqa: E402

ANNOTATION = json.loads((ROOT / "tests/fixtures/window_annotation_v15/trial-0132.json").read_text())["annotation"]
RULE = json.loads((ROOT / "configs/data/train_selection_first_batch_v1.json").read_text())


def record(**changes):
    return {"key": "w", "status": "ok", **classify(ANNOTATION), **changes}


def window(laion=5.0):
    # 与 final/windows.jsonl 的行同形：质量分在 scene.quality 里
    return {"key": "w", "source": "cinepile_t2v", "scene": {"shot_count": 4, "quality": {"min_laion_aesthetic": laion, "min_dover_technical": -0.1}}}


def test_conditions_are_checked_in_funnel_order():
    assert decide(window(), record(), set(), RULE) is None
    assert decide(window(), None, set(), RULE) == "标注没返回"
    assert decide(window(), {"key": "w", "status": "no_answer", "error_code": "data_inspection_failed"}, set(), RULE) == "标注没返回"
    assert decide(window(), record(status="invalid"), {"w"}, RULE) == "标注不合法"
    assert decide(window(), record(hard_filter=["烧录字幕"]), {"w"}, RULE) == "内容筛：烧录字幕"
    assert decide(window(), record(marks=["有角标或水印"]), {"w"}, RULE) == "与评测重复"
    assert decide(window(), record(marks=["有角标或水印", "无跨镜同人"]), set(), RULE) == "标记：有角标或水印"
    assert decide(window(), record(marks=["无跨镜同人"]), set(), RULE) is None
    assert decide(window(), record(**{"class": "单人讲话"}), set(), RULE) == "类别不在本批：单人讲话"
    assert decide(window(laion=4.2), record(), set(), RULE) == "有镜的美学分低于本批的线"


def test_run_writes_ledger_subset_and_capped_dialogue(tmp_path):
    final = tmp_path / "final"
    final.mkdir()
    keys = [f"film-{n:06d}-w0000" for n in range(12)]
    classes = ["核心"] * 6 + ["对白"] * 5 + ["其他留"]
    (final / "windows.jsonl").write_text("".join(json.dumps({**window(), "key": key}) + "\n" for key in keys))
    (final / "annotations.jsonl").write_text("".join(
        json.dumps({**record(**{"class": name}), "key": key, "source": "cinepile_t2v", "shot_count": 4, "annotation": ANNOTATION, "shot_texts": ["t"] * 4},
                   ensure_ascii=False) + "\n" for key, name in zip(keys, classes)))
    (final / "eval-exclude.jsonl").write_text(json.dumps({"key": keys[0]}) + "\n")
    (final / "scene_manifest.csv").write_text("case_id,window_key\n" + "".join(f"c{n},{key}\n" for n, key in enumerate(keys)))
    (tmp_path / "config.json").write_text(json.dumps({"root": str(tmp_path)}))
    subprocess.run([sys.executable, str(ROOT / "tools/data/select_training_windows.py"), "--config", str(tmp_path / "config.json"),
                    "--selection", str(ROOT / "configs/data/train_selection_first_batch_v1.json")], check=True, capture_output=True)
    out = final / "selections" / "first_batch_v1"
    summary = json.loads((out / "summary.json").read_text())
    # 核心 5 个（1 个与评测重复）；对白上限 = 5 × 0.15 / 0.85 取整 = 0
    assert summary["selected"] == 5 and summary["selected_by_class"] == {"核心": 5}
    ledger = [json.loads(line) for line in (out / "ledger.jsonl").read_text().splitlines()]
    assert len(ledger) == 12 and ledger[0]["decision"] == "与评测重复" and ledger[-1]["decision"] == "类别不在本批：其他留"
    assert sum(row["decision"] == "过了全部条件，超出本批的量或类别占比" for row in ledger) == 5
    assert len((out / "scene_manifest.csv").read_text().splitlines()) == 6
    assert len((out / "annotations.jsonl").read_text().splitlines()) == 5
