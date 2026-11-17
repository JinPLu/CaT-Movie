"""把窗清单导出成渲染器训练链读的场景族清单（一窗一行）。

帧号都是源文件的 FFmpeg 顺序解码帧号（0 起、半开区间）；框在解码后的源画面像素上。每镜取哪些帧与代理片同一个映射
（``shot_grid``）。清晰度分与美学分随行带出，训练时按它选窗。

    python tools/data/export_scene_manifest.py --config configs/data/source_pipeline_v1.json
"""
import argparse
import csv
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

COLUMNS = ["case_id", "mapping_schema", "video", "source_fps", "source_num_frames", "source_width", "source_height",
           "source_shot_segments", "source_frame_indices_by_shot", "shot_groups", "active_box", "train_crop_box",
           "film_key", "source", "scene_id", "container_preroll_frames", "window_key", "shot_count", "render_frames",
           "shot_laion_aesthetic", "shot_dover_technical", "min_laion_aesthetic", "min_dover_technical"]

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    a = p.parse_args()
    from mspc.data.frames import shot_grid

    root = Path(json.loads(a.config.read_text())["root"])
    rows = 0
    with (root / "final" / "windows.jsonl").open() as source, (root / "final" / "scene_manifest.partial").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=COLUMNS)
        writer.writeheader()
        for line in source:
            row = json.loads(line)
            window, shots = row["scene"], row["scene"]["shots"]
            groups, cursor = [], 0
            for shot in shots:
                groups.append([cursor, cursor + shot["render_frames"]])
                cursor += shot["render_frames"]
            if cursor != window["render_frames"]:
                raise ValueError(f"render frame total differs: {row['key']}")
            encode = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            writer.writerow({
                "case_id": window["window_id"], "mapping_schema": "scene_window_v1", "video": window["source_video_path"],
                "source_fps": window["fps"], "source_num_frames": window["video_frame_count"],
                "source_width": window["width"], "source_height": window["height"],
                "source_shot_segments": encode([[s["start_frame"], s["end_frame"]] for s in shots]),
                "source_frame_indices_by_shot": encode([shot_grid(s["start_frame"], s["end_frame"], s["render_frames"]) for s in shots]),
                "shot_groups": encode(groups), "active_box": encode(window["active_box"]), "train_crop_box": encode(window["crop_box"]),
                "film_key": window["film_key"] or "", "source": window["source"], "scene_id": window["scene_id"],
                "container_preroll_frames": "" if window["container_preroll_frames"] is None else window["container_preroll_frames"],
                "window_key": row["key"], "shot_count": window["shot_count"], "render_frames": window["render_frames"],
                "shot_laion_aesthetic": encode([round(s["laion_aesthetic"], 4) for s in shots]),
                "shot_dover_technical": encode([round(s["dover_technical"], 5) for s in shots]),
                "min_laion_aesthetic": round(window["quality"]["min_laion_aesthetic"], 4),
                "min_dover_technical": round(window["quality"]["min_dover_technical"], 5)})
            rows += 1
    (root / "final" / "scene_manifest.partial").replace(root / "final" / "scene_manifest.csv")
    print(json.dumps({"rows": rows, "out": str(root / "final" / "scene_manifest.csv")}, ensure_ascii=False))
