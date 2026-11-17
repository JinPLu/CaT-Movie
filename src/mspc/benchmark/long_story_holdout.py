"""主表长故事的留出集：从场景级清单按片名划出 6–12 镜真片故事窗，冻结 ID 清单与训练排除清单。

清单来源是 ``scene_manifests/v1/scenes.jsonl``。

**留出单位是片，不是场景**：CinePile 一部电影切成多段 MovieClips 片段，同一片的别的片段进训练，
测试时的人物、地点就见过了。所以先按片抽出留出片，每部留出片只出一条故事窗，这些片的**全部**场景都
写进训练排除清单。CinePile 的片是规范化片名。MovieChat 的视频 id 形如 ``AEC-8``、``AEC-22``：同一
前缀是同一部片的不同片段（【实测】78 个前缀分组，组内类别一致），所以片取 id 去掉 ``-N`` 后的前缀；
纯数字 id 没有分组信息，一个 id 算一部片（同片是否分在多个数字 id 未知）。

故事窗的规则（:func:`clean_runs`、:func:`story_windows`）：

- 场景合格：视频无硬字幕标记；CinePile 要有片名（``unnamed_tmp`` 的 17 条无从按片划分，不进留出也
  不进排除）；MovieChat 只取 ``movie``：纪录片不是剧情，``tvseries`` 的 ``SxxEyy-N`` 各季各集很可能是
  同一部剧，按集分会泄漏、按剧整体留出又要排除全部剧集。
- 镜头合格：无任何标记（片尾卡、黑场、冻结、黑底亮字），按渲染帧率（契约 ``target_fps``，16）换成 4n+1 帧后不超渲染链单段上限
  ``max_rgb_frames``（308 帧，约 20.5 秒），且 ``source_shot_index`` 与上一镜连续（中间没有被丢的
  <1 秒镜）。连续合格镜组成一段 run，故事窗是某段 run 里连续的 K 镜。
- 故事窗总长不超过 :data:`MAX_STORY_SECONDS`；按单段上限（≤308 帧、≤5 镜）至少要切的段数
  （:func:`min_segments`）不超过 :data:`MAX_SEGMENTS`。

抽样用 ``random.Random(seed)``，K 在 6..12 间轮转分配，保证各镜数都有。
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mspc.config.experiment import WAN22_CACHED_SPACETIME_CONTRACT
from mspc.data.frame_planner import align_rgb_length_to_vae, round_frame_count
from mspc.data.scene_shot_manifest import parse_cinepile_clip_name

SCHEMA_VERSION = "long_story_holdout_v1"
EXCLUSION_SCHEMA_VERSION = "long_story_training_exclusion_v1"
SOURCE_CINEPILE = "cinepile_t2v"
SOURCE_MOVIECHAT = "moviechat_1k"
#: 渲染帧率、4n+1 步长与渲染画幅（832×480）都取时空契约；真片参照帧按 :func:`crop_box` 从 ``active_box`` 中心裁到同一宽高比。
RENDER_FPS = int(WAN22_CACHED_SPACETIME_CONTRACT["target_fps"])
VAE_TEMPORAL_STRIDE = int(WAN22_CACHED_SPACETIME_CONTRACT["vae_temporal_stride"])
RENDER_SIZE = (832, 480)
#: 渲染链单段上限，也是单镜上限（帧）。
MAX_SEGMENT_FRAMES = int(WAN22_CACHED_SPACETIME_CONTRACT["max_rgb_frames"])
MAX_SEGMENT_SHOTS = 5
MIN_STORY_SHOTS, MAX_STORY_SHOTS = 6, 12
MAX_STORY_SECONDS = 60.0
MAX_SEGMENTS = 4
MOVIECHAT_GENRES = frozenset({"movie"})
SUBTITLE_FLAG = "burned_in_subtitles"
UNNAMED_STYLE = "unnamed_tmp"

_NON_WORD = re.compile(r"[^0-9a-z]+")
_CLIP_SUFFIX = re.compile(r"-\d+$")


def render_frame_count(seconds: float) -> int:
    """真片一镜的时长换成渲染帧数：与训练取帧同一规则（``FpsAwareTargetFpsPlanner.build_frame_plan`` 的单镜部分）：
    ``round(秒 × target_fps)`` 再向下对齐到 4n+1。"""

    if seconds <= 0:
        raise ValueError(f"shot duration must be positive, got {seconds}")
    return align_rgb_length_to_vae(round_frame_count(seconds * RENDER_FPS), VAE_TEMPORAL_STRIDE)


def shot_seconds(row: Mapping[str, Any], shot: Mapping[str, Any]) -> float:
    return (shot["end_frame"] - shot["start_frame"]) / float(row["fps"])


def min_segments(frame_counts: Sequence[int]) -> int:
    """按单段上限（≤:data:`MAX_SEGMENT_FRAMES` 帧、≤:data:`MAX_SEGMENT_SHOTS` 镜）连续切段的最少段数。

    两个上限都是可加、单调的，所以贪心装满就是最少段数。这只是下界读数：方法按镜间依赖切段，
    段数可以更多。
    """

    segments, frames, shots = 0, 0, 0
    for count in frame_counts:
        if count > MAX_SEGMENT_FRAMES:
            raise ValueError(f"a {count}-frame shot exceeds the {MAX_SEGMENT_FRAMES}-frame segment cap")
        if segments == 0 or frames + count > MAX_SEGMENT_FRAMES or shots == MAX_SEGMENT_SHOTS:
            segments, frames, shots = segments + 1, 0, 0
        frames, shots = frames + count, shots + 1
    return segments


def film_key(row: Mapping[str, Any]) -> str | None:
    """留出单位。CinePile：规范化片名（小写、非字母数字压成一个空格；同名不同年的片一起留出）；
    MovieChat：视频 id 去掉片段号 ``-N`` 的前缀。无片名的 CinePile 片段返回 ``None``。"""

    if row["source"] == SOURCE_MOVIECHAT:
        return f"{SOURCE_MOVIECHAT}:{_CLIP_SUFFIX.sub('', row['video_id'])}"
    if row["source"] != SOURCE_CINEPILE:
        raise ValueError(f"unknown source {row['source']!r}")
    # Source names own title semantics; v1 manifests contain stale scene-first
    # titles. Re-read them through the one parser, without rewriting the source.
    metadata = parse_cinepile_clip_name(Path(row["source_video_path"]).name)
    title = metadata.get("film_title")
    if metadata["name_style"] == UNNAMED_STYLE or not title:
        return None
    return f"{SOURCE_CINEPILE}:{_NON_WORD.sub(' ', title.lower()).strip()}"


def scene_eligible(row: Mapping[str, Any]) -> bool:
    if SUBTITLE_FLAG in row["video_flags"] or film_key(row) is None:
        return False
    if row["source"] == SOURCE_MOVIECHAT:
        return row["metadata"]["genre"] in MOVIECHAT_GENRES
    return True


def clean_runs(row: Mapping[str, Any]) -> list[list[dict[str, Any]]]:
    """场景里连续合格镜组成的 run（见模块说明的镜头合格规则）。"""

    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for shot in row["shots"]:
        ok = not shot["flags"] and render_frame_count(shot_seconds(row, shot)) <= MAX_SEGMENT_FRAMES
        if ok and current and shot["source_shot_index"] == current[-1]["source_shot_index"] + 1:
            current.append(shot)
            continue
        if current:
            runs.append(current)
        current = [shot] if ok else []
    if current:
        runs.append(current)
    return runs


def window_fits(row: Mapping[str, Any], shots: Sequence[Mapping[str, Any]]) -> bool:
    seconds = sum(shot_seconds(row, shot) for shot in shots)
    frames = [render_frame_count(shot_seconds(row, shot)) for shot in shots]
    return seconds <= MAX_STORY_SECONDS and min_segments(frames) <= MAX_SEGMENTS


def story_windows(row: Mapping[str, Any], shot_count: int) -> list[list[dict[str, Any]]]:
    """场景里所有合格的 K 镜故事窗（按起点顺序）。"""

    return [
        run[start : start + shot_count]
        for run in clean_runs(row)
        for start in range(len(run) - shot_count + 1)
        if window_fits(row, run[start : start + shot_count])
    ]


def crop_box(row: Mapping[str, Any]) -> list[int]:
    """真片参照的裁切框：``active_box`` 中心裁到渲染宽高比，偶数像素。"""

    left, top, right, bottom = row["active_box"]
    width, height = right - left, bottom - top
    target = RENDER_SIZE[0] / RENDER_SIZE[1]
    if width / height > target:
        crop_w, crop_h = int(round(height * target / 2)) * 2, height - height % 2
    else:
        crop_w, crop_h = width - width % 2, int(round(width / target / 2)) * 2
    x0 = left + (width - crop_w) // 2
    y0 = top + (height - crop_h) // 2
    return [x0, y0, x0 + crop_w, y0 + crop_h]


def story_record(row: Mapping[str, Any], shots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    frames = [render_frame_count(shot_seconds(row, shot)) for shot in shots]
    return {
        "film_key": film_key(row),
        "source": row["source"],
        "scene_id": row["scene_id"],
        "video_id": row["video_id"],
        "source_video_path": row["source_video_path"],
        "fps": row["fps"],
        "container_preroll_frames": row["container_preroll_frames"],
        "width": row["width"],
        "height": row["height"],
        "crop_box": crop_box(row),
        "shot_count": len(shots),
        "seconds": round(sum(shot_seconds(row, shot) for shot in shots), 3),
        "render_frames": sum(frames),
        "min_segments": min_segments(frames),
        "shots": [
            {
                "source_shot_index": shot["source_shot_index"],
                "start_frame": shot["start_frame"],
                "end_frame": shot["end_frame"],
                "seconds": round(shot_seconds(row, shot), 3),
                "render_frames": count,
            }
            for shot, count in zip(shots, frames)
        ],
        "metadata": (parse_cinepile_clip_name(Path(row["source_video_path"]).name)
                     if row["source"] == SOURCE_CINEPILE else row["metadata"]),
    }


def select_holdout(
    rows: Iterable[Mapping[str, Any]], *, counts: Mapping[str, int], seed: int
) -> list[dict[str, Any]]:
    """按片抽留出故事：每个来源抽 ``counts[source]`` 部片，每部片一条故事窗。

    片按 ``random.Random(seed)`` 打乱；K 按 6..12 轮转，轮到的 K 在这部片的合格场景里随机取一个
    故事窗；这部片没有该 K 的窗就顺延到下一个 K（仍在 6..12 内），都没有就跳过这部片。
    """

    by_film: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if scene_eligible(row):
            by_film.setdefault(film_key(row), []).append(row)
    rng = random.Random(seed)
    stories: list[dict[str, Any]] = []
    shot_counts = list(range(MIN_STORY_SHOTS, MAX_STORY_SHOTS + 1))
    for source, wanted in counts.items():
        films = sorted(key for key in by_film if key.startswith(f"{source}:"))
        rng.shuffle(films)
        turn = 0
        taken = 0
        for key in films:
            if taken == wanted:
                break
            for offset in range(len(shot_counts)):
                shot_count = shot_counts[(turn + offset) % len(shot_counts)]
                options = [
                    (row, window)
                    for row in sorted(by_film[key], key=lambda item: item["scene_id"])
                    for window in story_windows(row, shot_count)
                ]
                if options:
                    row, window = options[rng.randrange(len(options))]
                    stories.append(story_record(row, window))
                    turn, taken = turn + 1, taken + 1
                    break
        if taken < wanted:
            raise ValueError(f"{source}: only {taken} films have an eligible story window, wanted {wanted}")
    for number, story in enumerate(stories, start=1):
        story["story_id"] = f"lsh-{number:03d}"
    return stories


def exclusion_rows(rows: Iterable[Mapping[str, Any]], held_out_films: set[str]) -> list[dict[str, str]]:
    """留出片在清单里的全部场景（含不合格的场景），训练池须按 ``scene_id``／``video_id`` 排除。"""

    return [
        {"film_key": film_key(row), "scene_id": row["scene_id"], "video_id": row["video_id"]}
        for row in rows
        if film_key(row) in held_out_films
    ]


def summary(stories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def quantiles(values: Sequence[float]) -> list[float]:
        ordered = sorted(values)
        return [round(ordered[int(q * (len(ordered) - 1))], 2) for q in (0.0, 0.1, 0.5, 0.9, 1.0)]

    return {
        "stories": len(stories),
        "by_source": dict(Counter(story["source"] for story in stories)),
        "shot_count": dict(sorted(Counter(story["shot_count"] for story in stories).items())),
        "min_segments": dict(sorted(Counter(story["min_segments"] for story in stories).items())),
        "story_seconds_min_p10_p50_p90_max": quantiles([story["seconds"] for story in stories]),
        "shot_seconds_min_p10_p50_p90_max": quantiles(
            [shot["seconds"] for story in stories for shot in story["shots"]]
        ),
        "shots": sum(story["shot_count"] for story in stories),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True, help="scene_manifests/v1/scenes.jsonl")
    parser.add_argument("--limit", type=int, required=True, help="read only the first N rows (0 = all)")
    parser.add_argument("--cinepile-films", type=int, required=True)
    parser.add_argument("--moviechat-videos", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--holdout-out", type=Path, required=True)
    parser.add_argument("--exclusion-out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = []
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if args.limit and len(rows) == args.limit:
                break
    counts = {SOURCE_CINEPILE: args.cinepile_films, SOURCE_MOVIECHAT: args.moviechat_videos}
    stories = select_holdout(rows, counts={k: v for k, v in counts.items() if v}, seed=args.seed)
    films = {story["film_key"] for story in stories}
    excluded = exclusion_rows(rows, films)
    provenance = {"manifest": str(args.manifest), "rows_read": len(rows), "seed": args.seed}
    for path, payload in (
        (args.holdout_out, {"schema_version": SCHEMA_VERSION, **provenance, "summary": summary(stories),
                            "stories": stories}),
        (args.exclusion_out, {"schema_version": EXCLUSION_SCHEMA_VERSION, **provenance,
                              "film_keys": sorted(films),
                              "scene_count": len(excluded), "scenes": excluded}),
    ):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen output {path}")
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({**summary(stories), "held_out_films": len(films), "excluded_scenes": len(excluded)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
