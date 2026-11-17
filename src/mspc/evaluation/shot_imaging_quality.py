"""把已有八项辅助指标的逐帧 MUSIQ 按生成镜区间汇总；不运行模型或设阈值。

输入是 ``mspc.evaluation.benchmark`` 的单故事结果与同一视频的
``segments.json``。镜区间复用生成时间轴的唯一解析，不取检测到的切镜。
每个 ``--pair`` 明确配对两个文件，可重复指定；帧必须完整覆盖输入视频。
``--short-shot-max-seconds`` 必填；按实际输出帧数除以 ``non_vlm.fps``
分类（本轮用 17/16 = 1.0625 秒），不将不同输出帧率下的帧数直接相比，
也不把输出时长称为输入要求的帧数或时长。这不是成像质量的判定阈值。
可选噪块标记格式为 ``{"source": "原始判读记录", "shots": [
{"method": "...", "story_id": "...", "input_tier": "shots", "seed": 42,
"shot": 2}]}``，镜号从 1 开始。只描述标记的分布，不推导检测阈值。
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from itertools import accumulate
import json
import math
from pathlib import Path
from statistics import fmean

from mspc.evaluation.generation_timing import shot_frames


SAMPLE_FIELDS = ('method', 'story_id', 'input_tier', 'seed')


def sample_key(record: dict) -> tuple:
    values = []
    for name in SAMPLE_FIELDS[:-1]:
        value = record[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{name} must be a nonempty string')
        values.append(value)
    seed = record['seed']
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    return (*values, seed)


def positive_integer(value, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def positive_number(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite positive number')
    return float(value)


def frame_scores(record: dict) -> list[float]:
    quality = record['non_vlm']
    if quality['status'] != 'completed':
        raise ValueError('non_vlm metrics must be completed')
    imaging = quality['metrics']['imaging_quality']
    if imaging['status'] != 'measured':
        raise ValueError('imaging_quality must be measured')
    if imaging['frame_index_basis'] != 'zero_based_input_video':
        raise ValueError('frame_index_basis must be zero_based_input_video')
    if imaging['score_scale'] != 'MUSIQ/100':
        raise ValueError('score_scale must be MUSIQ/100')
    count = positive_integer(quality['frame_count'], 'frame_count')
    indices, scores = imaging['frame_indices'], imaging['frame_scores']
    if not isinstance(indices, list) or len(indices) != count or any(
            type(index) is not int or index != expected
            for expected, index in enumerate(indices)):
        raise ValueError('frame_indices must cover every frame in order, starting at zero')
    if not isinstance(scores, list) or len(scores) != count:
        raise ValueError('frame_scores length must equal frame_count')
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in scores):
        raise ValueError('frame_scores must be finite numbers')
    value = imaging['value']
    if type(value) not in (int, float) or not math.isfinite(value) or not math.isclose(
            value, fmean(scores), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError('imaging_quality.value must equal the mean of frame_scores')
    return scores


def shot_spans(segments: list[dict], frame_count: int, *, fps: float,
               short_shot_max_seconds: float) -> list[dict]:
    if not isinstance(segments, list) or not segments:
        raise ValueError('segments must be a nonempty list')
    fps = positive_number(fps, 'fps')
    threshold = positive_number(short_shot_max_seconds, 'short_shot_max_seconds')
    counts = shot_frames(segments)
    if sum(counts) != frame_count:
        raise ValueError(f'segment output frames total {sum(counts)} differs from scored frame_count {frame_count}')
    segment_ends = list(accumulate(segment['num_frames'] for segment in segments))
    shots, offset = [], 0
    for count in counts:
        duration = count / fps
        shots.append({'shot': len(shots) + 1, 'segment_index': bisect_right(segment_ends, offset),
                      'start_frame': offset, 'end_frame': offset + count, 'frames': count,
                      'duration_seconds': duration, 'short_shot': duration <= threshold})
        offset += count
    return shots


def distribution(shots: list[dict]) -> dict:
    """描述镜等权的分布；空组不产生完美分或零分。"""
    return {'shots': len(shots), 'frames': sum(shot['frames'] for shot in shots),
            'duration_seconds': sum(shot['duration_seconds'] for shot in shots),
            'mean_of_shot_means': fmean(shot['imaging_quality']['mean'] for shot in shots) if shots else None,
            'mean_of_shot_minima': fmean(shot['imaging_quality']['min'] for shot in shots) if shots else None,
            'minimum_frame_score': min(shot['imaging_quality']['min'] for shot in shots) if shots else None}


def duration_groups(shots: list[dict]) -> dict:
    return {'at_most_threshold_seconds': distribution([shot for shot in shots if shot['short_shot']]),
            'more_than_threshold_seconds': distribution([shot for shot in shots if not shot['short_shot']])}


def summarize_story(record: dict, segments: list[dict], *, short_shot_max_seconds: float) -> dict:
    sample_key(record)
    scores = frame_scores(record)
    fps = positive_number(record['non_vlm']['fps'], 'non_vlm.fps')
    threshold = positive_number(short_shot_max_seconds, 'short_shot_max_seconds')
    shots = shot_spans(segments, len(scores), fps=fps, short_shot_max_seconds=threshold)
    for shot in shots:
        values = scores[shot['start_frame']:shot['end_frame']]
        shot['imaging_quality'] = {'min': min(values), 'mean': fmean(values)}
    return {**{name: record[name] for name in SAMPLE_FIELDS}, 'frame_count': len(scores),
            'fps': fps, 'duration_seconds': len(scores) / fps, 'short_shot_max_seconds': threshold,
            'shots': shots, 'duration_groups': duration_groups(shots)}


def noise_labels(path: Path, samples: list[dict]) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data['source'], str) or not data['source'].strip():
        raise ValueError('noise labels require a nonempty source')
    if not isinstance(data['shots'], list):
        raise ValueError('noise labels shots must be a list')
    by_sample = {sample_key(sample): sample for sample in samples}
    seen, matched, outside = set(), [], []
    for row in data['shots']:
        key, number = sample_key(row), positive_integer(row['shot'], 'noise label shot')
        identity = (*key, number)
        if identity in seen:
            raise ValueError(f'duplicate noise label: {identity}')
        seen.add(identity)
        if key not in by_sample:
            outside.append(row)
            continue
        sample = by_sample[key]
        if number > len(sample['shots']):
            raise ValueError(f'noise label shot {number} is outside sample {key}')
        matched.append({**{name: row[name] for name in SAMPLE_FIELDS}, **sample['shots'][number - 1]})
    return {'path': str(path.resolve()), 'source': data['source'], 'labels_declared': len(seen),
            'labels_in_samples': len(matched), 'labels_outside_samples': outside,
            'duration_groups': duration_groups(matched), 'shots': matched}


def summarize_pairs(pairs: list[tuple[Path, Path]], *, short_shot_max_seconds: float,
                    labels: Path | None = None) -> dict:
    if not pairs:
        raise ValueError('at least one metrics/segments pair is required')
    samples, seen = [], set()
    for metrics, segments in pairs:
        record = json.loads(metrics.read_text())
        key = sample_key(record)
        if key in seen:
            raise ValueError(f'duplicate sample: {key}')
        seen.add(key)
        sample = summarize_story(record, json.loads(segments.read_text()),
                                 short_shot_max_seconds=short_shot_max_seconds)
        sample['sources'] = {'metrics': str(metrics.resolve()), 'segments': str(segments.resolve())}
        samples.append(sample)
    groups = defaultdict(list)
    for sample in samples:
        groups[(sample['method'], sample['input_tier'], sample['seed'])].append(sample)
    result = {'score_scale': 'MUSIQ/100', 'frame_intervals': 'zero_based_half_open',
              'shot_numbers': 'one_based', 'short_shot_max_seconds': short_shot_max_seconds,
              'short_shot_definition': 'actual_output_duration_seconds_lte_threshold',
              'duration_basis': 'generated_output_frames_divided_by_non_vlm_fps',
              'aggregation': 'shot_weighted_descriptive_statistics',
              'samples': samples, 'groups': []}
    for (method, tier, seed), members in sorted(groups.items()):
        shots = [shot for sample in members for shot in sample['shots']]
        result['groups'].append({'method': method, 'input_tier': tier, 'seed': seed,
                                 'samples': len(members), 'all_shots': distribution(shots),
                                 'duration_groups': duration_groups(shots)})
    if labels is not None:
        result['noise_labels'] = noise_labels(labels, samples)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', nargs=2, type=Path, action='append', required=True,
                        metavar=('METRICS_JSON', 'SEGMENTS_JSON'), help='同一成片的结果与生成镜区间；可重复')
    parser.add_argument('--noise-labels', type=Path, help='已确认噪块镜的来源与标记，不用于设阈值')
    parser.add_argument('--short-shot-max-seconds', type=float, required=True,
                        help='按实际输出时长分组的上限秒数；本轮为 17/16=1.0625，不是画质阈值')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize_pairs(args.pair, short_shot_max_seconds=args.short_shot_max_seconds,
                             labels=args.noise_labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
