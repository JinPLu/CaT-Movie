"""计划切点读数：期望切点取判分行的实际镜时（判官看的同一份时间段），检出切点取非 VLM 运行已存的
TransNetV2 记录；不重跑检测、不发判分请求。

- 切点准确率：直接调用 ``video_metrics.cut_metrics``（评测协议「切镜控制的精确定义」，沿用 HoloCine 的指数形式），故事等权。
- 切点 F1：同一调用的 ``cut_f1``，容差为 16 fps 网格上的 2 帧，故事等权；正文表 1 的切点列。
- ±0.5 秒检出率：计划切点前后 0.5 秒内有检出切点的比例（按时间差从小到大一一配对），即第 7 版切镜执行列的时间部分；
  报逐切点合并与故事等权两种。
- 计划外切点：没配上任何计划切点的检出切点，每部平均个数。

用法::

    PYTHONPATH=src python tools/analysis/planned_cut_accuracy.py --rows-root R/rows --metrics-root M/metrics \\
        --pair ROW METHOD [--pair ...] --output O.json
"""
from __future__ import annotations

import argparse
import json
from itertools import accumulate
from pathlib import Path
from statistics import mean

from mspc.evaluation.checklist_scoring import WINDOW_SECONDS, planned_cut_hits
from mspc.evaluation.video_metrics import CUT_F1_TOLERANCE_SECONDS, cut_metrics


def row_readout(prepared: dict, metrics_dir: Path) -> dict:
    stories = []
    for story in prepared['stories']:
        record = json.loads((metrics_dir / f"{story['story_id']}.json").read_text())
        duration = record['media']['duration_seconds']
        if abs(sum(story['shot_seconds']) - duration) > 0.1:
            raise ValueError(f"{story['story_id']}: judged timing {sum(story['shot_seconds'])} != video {duration}")
        planned = list(accumulate(story['shot_seconds']))[:-1]
        detected = record['non_vlm']['detected_cut_times_seconds']
        scores = cut_metrics({'expected_shot_count': len(story['shot_seconds']), 'expected_cut_times_seconds': planned},
                             detected, duration)
        sca = scores['shot_cut_accuracy']['value']
        hit = len(planned_cut_hits(planned, detected))
        stories.append({'story_id': story['story_id'], 'shot_cut_accuracy': sca,
                        'cut_f1': scores['cut_f1']['value'], 'cut_f1_matched': scores['cut_f1']['matched'],
                        'planned_cuts': len(planned),
                        'detected_within_window': hit, 'extra_cuts': len(detected) - hit})
    if prepared['unavailable_stories']:
        raise ValueError('rows with unavailable videos need an explicit failure rule before this readout')
    planned_total = sum(s['planned_cuts'] for s in stories)
    return {'stories': len(stories),
            'shot_cut_accuracy_mean': mean(s['shot_cut_accuracy'] for s in stories),
            'cut_f1_mean': mean(s['cut_f1'] for s in stories),
            'planned_cuts': planned_total,
            'detected_within_window': sum(s['detected_within_window'] for s in stories),
            'window_rate_pooled': sum(s['detected_within_window'] for s in stories) / planned_total,
            'window_rate_story_mean': mean(s['detected_within_window'] / s['planned_cuts']
                                           for s in stories if s['planned_cuts']),
            'extra_cuts_per_story': mean(s['extra_cuts'] for s in stories),
            'per_story': stories}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rows-root', type=Path, required=True)
    parser.add_argument('--metrics-root', type=Path, required=True)
    parser.add_argument('--pair', nargs=2, action='append', metavar=('ROW', 'METHOD'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'window_seconds': WINDOW_SECONDS, 'cut_f1_tolerance_seconds': CUT_F1_TOLERANCE_SECONDS,
              'definition': 'shot_cut_accuracy and cut_f1 = video_metrics.cut_metrics against cumulative judged shot_seconds; '
                            'window rate = planned cuts with a detected cut within window_seconds, one-to-one',
              'sources': {'judged_timing': str(args.rows_root), 'detected_cuts': str(args.metrics_root)},
              'tool': 'tools/analysis/planned_cut_accuracy.py', 'rows': {}}
    for row, method in args.pair:
        prepared = json.loads((args.rows_root / row / 'prepared.json').read_text())
        metrics_dir = args.metrics_root / method / method / prepared['input_tier'] / 'seed-42'
        result['rows'][row] = {'method': method, **row_readout(prepared, metrics_dir)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n')
    for row, value in result['rows'].items():
        print(row, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in value.items() if k != 'per_story'})


if __name__ == '__main__':
    main()
