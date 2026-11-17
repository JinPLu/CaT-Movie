"""社区基准 ST-Bench（StoryMem，arXiv 2512.19539）：读题集 → 我们方法的请求目录 → 四组指标评分 → 汇总。

指标照原文：美学（VBench 的 LAION 预测器）；ViCLIP 提示遵循，整片对故事概述、逐镜对本镜提示；
ViCLIP 跨镜一致，全部镜对与按提示特征最相近的十对。原文没写的抽帧、文本编码器、镜界口径
由本实现定。StoryMem 未公开评测代码，所有行在本实现上重算。
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import tempfile

from mspc.evaluation.benchmark import write_json

METRICS = ('aesthetic_quality', 'prompt_following_global', 'prompt_following_single_shot',
           'cross_shot_consistency_all_pairs', 'cross_shot_consistency_top_pairs',
           'cross_shot_consistency_cross_take_pairs')


def load_config(path: Path) -> dict:
    config = json.loads(os.path.expandvars(path.read_text()))
    if '${' in json.dumps(config):
        raise ValueError('unresolved environment variable in ST-Bench config')
    for key in ('shot_frames', 'fps', 'top_pairs', 'viclip_batch_size'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    return config


def load_story(path: Path) -> dict:
    data = json.loads(path.read_text())
    shots = []
    for scene in data['scenes']:
        prompts, first_frames, cuts = scene['video_prompts'], scene['first_frame_prompt'], scene['cut']
        if not prompts or not len(prompts) == len(first_frames) == len(cuts):
            raise ValueError(f'{path.name}: scene {scene["scene_num"]} prompts and cut flags disagree')
        for prompt, first_frame, cut in zip(prompts, first_frames, cuts):
            if any(not isinstance(t, str) or not t.strip() for t in (prompt, first_frame)) or type(cut) is not bool:
                raise ValueError(f'{path.name}: empty prompt or non-boolean cut flag')
            shots.append({'scene': scene['scene_num'], 'prompt': prompt.strip(),
                          'first_frame_prompt': first_frame.strip(), 'cut': cut})
    overview = data['story_overview'].strip()
    if not overview:
        raise ValueError(f'{path.name}: empty story overview')
    return {'story_id': f'st-bench-{path.stem}', 'source_file': str(path), 'overview': overview, 'shots': shots}


def load_stories(config: dict) -> list[dict]:
    source = config['source']
    stories = [load_story(path) for path in sorted(Path(source['story_dir']).glob('*.json'))]
    shots = sum(len(story['shots']) for story in stories)
    if (len(stories), shots) != (source['stories'], source['shots']):
        raise ValueError(f'ST-Bench has {len(stories)} stories / {shots} shots, '
                         f'expected {source["stories"]} / {source["shots"]}')
    return stories


def request_text(story: dict) -> str:
    """逐镜提示原文作 shots 档请求；不切镜的续接镜照原题集标明。"""
    lines = ['STORY', story['overview'], 'SHOTS']
    for index, shot in enumerate(story['shots'], 1):
        note = '' if shot['cut'] or index == 1 else f' (continues s{index - 1:02d} without a cut)'
        lines.append(f's{index:02d}{note}: {shot["prompt"]}')
    return '\n'.join(lines)


def shot_boundary_seconds(story: dict, config: dict) -> list[float]:
    """每镜固定 shot_frames 帧时第 2 镜起各镜的起点秒数（我们的方法逐镜固定 81 帧，不让规划器改镜长与
    镜数）。请求合同要求切点数＝镜数−1，续拍镜在我们的流水线里也是单独一镜，续拍只留在请求原文的标注里。"""
    seconds = config['shot_frames'] / config['fps']
    return [index * seconds for index in range(1, len(story['shots']))]


def request_catalog(stories: list[dict], config: dict) -> dict:
    seconds = config['shot_frames'] / config['fps']
    return {'evaluation_scope': 'system', 'request_version': 'st_bench_v1',
            'authoring': {'source': config['source']['repository'], 'commit': config['source']['commit']},
            'stories': [{'story_id': story['story_id'], 'split': 'test', 'fps': config['fps'],
                         'target_seconds': seconds * len(story['shots']),
                         'inputs': {'shots': {'text': request_text(story),
                                              'expected_shot_count': len(story['shots']),
                                              'expected_cut_times_seconds': shot_boundary_seconds(story, config)}}}
                        for story in stories]}


def cosine(left, right) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        sum(a * a for a in left) ** 0.5 * sum(b * b for b in right) ** 0.5)


def top_pairs(text_features: list, count: int) -> list[tuple[int, int]]:
    """按提示特征相似度取前 count 对镜头；同分按镜号先后。"""
    pairs = list(combinations(range(len(text_features)), 2))
    if len(pairs) < count:
        raise ValueError(f'{len(text_features)} shots give fewer than {count} pairs')
    return sorted(pairs, key=lambda p: (-cosine(text_features[p[0]], text_features[p[1]]), p))[:count]


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def takes(story: dict) -> list[int]:
    """每镜所属的镜次：`cut=false` 的镜与上一镜是同一次续拍，`cut=true`（含首镜）开新的一次。"""
    out = []
    for index, shot in enumerate(story['shots']):
        out.append(len(set(out)) if index == 0 or shot['cut'] else out[-1])
    return out


def cross_take_consistency(videos: list, story: dict) -> float | None:
    """只算不同镜次之间的镜对（公平性附加列）：续拍镜对在 StoryMem 里共用一帧、相似度天然高，
    其他方法在这些位置照常切镜；把它们去掉，各方法的跨镜一致按同一批硬切镜对比。只有一个镜次时为 None。"""
    owner = takes(story)
    pairs = [(a, b) for a, b in combinations(range(len(videos)), 2) if owner[a] != owner[b]]
    return mean(cosine(videos[a], videos[b]) for a, b in pairs) if pairs else None


def viclip_metrics(*, frames: list, shot_frame_counts: list[int], story: dict, viclip, config: dict) -> dict:
    from mspc.evaluation.renderer_qualification_official_backend import _eight_shot_indices

    if len(shot_frame_counts) != len(story['shots']):
        raise ValueError(f'{len(shot_frame_counts)} generated shots for {len(story["shots"])} prompts')
    if sum(shot_frame_counts) != len(frames):
        raise ValueError(f'shot frame counts sum to {sum(shot_frame_counts)}, video has {len(frames)} frames')
    starts = [sum(shot_frame_counts[:i]) for i in range(len(shot_frame_counts))]
    shots = [[frames[i] for i in _eight_shot_indices(start, count)]
             for start, count in zip(starts, shot_frame_counts)]
    whole = [frames[i] for i in _eight_shot_indices(0, len(frames))]
    prompts = [shot['prompt'] for shot in story['shots']]
    batch = config['viclip_batch_size']
    videos = viclip.embed_videos(shots, viclip_shot_batch_size=batch)
    single = viclip.score_shots(list(zip(shots, prompts)), viclip_shot_batch_size=batch)
    chosen = top_pairs(viclip.embed_texts(prompts), config['top_pairs'])
    return {'prompt_following_global': viclip.score(whole, story['overview']),
            'prompt_following_single_shot': mean(single),
            'cross_shot_consistency_all_pairs': mean(cosine(videos[a], videos[b])
                                                     for a, b in combinations(range(len(videos)), 2)),
            'cross_shot_consistency_top_pairs': mean(cosine(videos[a], videos[b]) for a, b in chosen),
            'cross_shot_consistency_cross_take_pairs': cross_take_consistency(videos, story),
            'single_shot_scores': single, 'top_pairs': [list(pair) for pair in chosen]}


def aesthetic(video: Path, shot_frame_counts: list[int], fps: float, non_vlm: dict) -> dict:
    """复用现行辅助指标的 VBench 子进程；美学取整段逐帧平均。"""
    from mspc.evaluation.renderer_qualification_official_backend import resolve_vbench_python

    starts = [sum(shot_frame_counts[:i]) for i in range(len(shot_frame_counts) + 1)]
    shots = [{'start_frame': a, 'end_frame': b} for a, b in zip(starts, starts[1:])]
    with tempfile.TemporaryDirectory(prefix='storyboard-st-bench-') as tmp:
        source = Path(tmp) / 'input.json'
        source.write_text(json.dumps({'video': str(video), 'fps': fps, 'shots': shots, 'config': non_vlm}))
        worker = Path(__file__).resolve().parents[3] / 'tools/run/vbench_worker.py'
        result = subprocess.run([str(resolve_vbench_python()), str(worker), '--input', str(source)],
                                check=True, stdout=subprocess.PIPE, text=True)
    return json.loads(result.stdout)


def score(args) -> None:
    from mspc.evaluation.renderer_qualification_official_backend import _decode_rgb, _ViCLIPEngine

    config = load_config(args.config)
    stories = {story['story_id']: story for story in load_stories(config)}
    non_vlm = json.loads(os.path.expandvars(Path(config['non_vlm_config']).read_text()))['non_vlm']
    manifest = json.loads(args.manifest.read_text())
    method, rows = manifest['method'], manifest['rows']
    if sorted(row['story_id'] for row in rows) != sorted(stories):
        raise ValueError('manifest must list every ST-Bench story exactly once')
    viclip = _ViCLIPEngine(Path(non_vlm['resource_root']), non_vlm['device'])
    for row in rows:
        target = args.output / method / f"{row['story_id']}.json"
        if target.exists():
            continue  # 已测与已记失败都不重算；换口径另开输出目录。
        record = {'method': method, **row}
        video = None if row['video_path'] is None else Path(row['video_path'])
        if video is None or not video.is_file():
            write_json(target, {**record, 'status': 'generation_missing'})
            continue
        try:
            counts = row['shot_frame_counts']
            frames, fps = _decode_rgb(video)
            record.update(viclip_metrics(frames=frames, shot_frame_counts=counts, story=stories[row['story_id']],
                                         viclip=viclip, config=config))
            del frames
            vbench = aesthetic(video, counts, fps, non_vlm)
            record.update(status='measured', aesthetic_quality=vbench['metrics']['aesthetic_quality']['value'],
                          vbench=vbench)
        except ValueError as error:
            record.update(status='measurement_failed', error=f'{type(error).__name__}: {error}')
        write_json(target, record)
        print(json.dumps({'method': method, 'story_id': row['story_id'], 'status': record['status']}), flush=True)


def summarize(args) -> None:
    summary = {}
    for directory in sorted(path for path in args.output.iterdir() if path.is_dir()):
        records = [json.loads(path.read_text()) for path in sorted(directory.glob('st-bench-*.json'))]
        measured = [record for record in records if record['status'] == 'measured']
        statuses = {}
        for record in records:
            statuses[record['status']] = statuses.get(record['status'], 0) + 1
        values = {name: [r[name] for r in measured if r[name] is not None] for name in METRICS}
        summary[directory.name] = {'stories': len(records), 'statuses': statuses,
                                   **{name: mean(v) if v else None for name, v in values.items()},
                                   'n': {name: len(v) for name, v in values.items()}}
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    requests = commands.add_parser('requests', help='write the shots-tier request catalog for our planner')
    requests.add_argument('--config', type=Path, required=True)
    requests.add_argument('--output', type=Path, required=True)
    scoring = commands.add_parser('score', help='score one method; manifest lists video and shot frame counts')
    scoring.add_argument('--config', type=Path, required=True)
    scoring.add_argument('--manifest', type=Path, required=True)
    scoring.add_argument('--output', type=Path, required=True)
    summary = commands.add_parser('summarize', help='story-weighted means per method')
    summary.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'requests':
        config = load_config(args.config)
        write_json(args.output, request_catalog(load_stories(config), config))
    elif args.command == 'score':
        score(args)
    else:
        summarize(args)


if __name__ == '__main__':
    main()
