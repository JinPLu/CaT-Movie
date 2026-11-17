"""冻结评测分镜表里的长程依赖结构：状态改变后首次再现、实体离场后再现，按依赖距离与是否跨渲染段计数。

只读共同分镜表，不读成片、不调模型。窗口与出片同一个装窗函数（``mspc.comparison_methods.prompts.adapt_storyboard``：
从第一镜贪心装满，镜头连续，整窗不超过 ``--max-frames`` 个 16fps 源帧，单镜超过时独占一窗，逐镜文字不超过 512 个 umT5
token）。测试集主表用 112（每窗 ≤7 秒），开发集消融用 128（≤8 秒）。

用法：.venv/bin/python tools/analysis/benchmark_long_range_dependencies.py \
    --storyboards configs/benchmark/benchmark_storyboards_v4.json --max-frames 112 --output <json>
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median

from mspc.comparison_methods.prompts import adapt_storyboard


def segment_of_shot(storyboard: dict, max_frames: int, count_tokens) -> list[int]:
    owner = [0] * len(storyboard['shots'])
    windows = adapt_storyboard(storyboard, family='wan_segment', count_tokens=count_tokens,
                               limits=dict(max_shots=None, max_frames=max_frames, max_tokens=512))
    for index, window in enumerate(windows):
        for shot in range(window['start_shot'], window['end_shot']):
            owner[shot] = index
    return owner


def story_dependencies(story: dict, max_frames: int, count_tokens) -> dict:
    shots = story['storyboard']['shots']
    present = [{entry['name'] for entry in shot['present']} for shot in shots]
    segment = segment_of_shot(story['storyboard'], max_frames, count_tokens)
    carryovers = []
    for index, shot in enumerate(shots):
        for change in shot['changes']:
            later = next((j for j in range(index + 1, len(shots)) if change['name'] in present[j]), None)
            if later is not None:
                carryovers.append({'entity': change['name'], 'what': change['what'], 'from_shot': index + 1,
                                   'to_shot': later + 1, 'distance': later - index,
                                   'cross_segment': segment[later] != segment[index]})
    returns = []
    for name in sorted(set().union(*present)):
        seen = [j for j, names in enumerate(present) if name in names]
        for before, after in zip(seen, seen[1:]):
            if after - before >= 2:
                returns.append({'entity': name, 'from_shot': before + 1, 'to_shot': after + 1,
                                'distance': after - before, 'cross_segment': segment[after] != segment[before]})
    return {'story_id': story['story_id'], 'split': story['split'], 'shots': len(shots),
            'segments': segment[-1] + 1, 'carryovers': carryovers, 'returns': returns}


def summarize(rows: list[dict]) -> dict:
    out ={'stories': len(rows), 'shots': sum(row['shots'] for row in rows),
           'segments_histogram': dict(sorted(Counter(row['segments'] for row in rows).items()))}
    for key in ('carryovers', 'returns'):
        items = [item for row in rows for item in row[key]]
        cross = [item for item in items if item['cross_segment']]
        out[key] = {'count': len(items), 'cross_segment': len(cross),
                    'stories_with_any': sum(1 for row in rows if row[key]),
                    'stories_with_cross_segment': sum(1 for row in rows if any(i['cross_segment'] for i in row[key])),
                    'distance_median': median(i['distance'] for i in items) if items else None,
                    'distance_histogram_capped_at_5': dict(sorted(Counter(min(i['distance'], 5) for i in items).items()))}
    out['stories_with_cross_segment_dependency'] = sum(
        1 for row in rows if any(i['cross_segment'] for key in ('carryovers', 'returns') for i in row[key]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--storyboards', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, required=True, help='每窗 16fps 源帧上限：112（7 秒）或 128（8 秒）')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import re
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('google/umt5-xxl', local_files_only=True)
    count_tokens = lambda text: len(tokenizer([re.sub(r'\s+', ' ', text).strip()], padding=False, truncation=False,
                                              add_special_tokens=True)['input_ids'][0])
    rows = [story_dependencies(story, args.max_frames, count_tokens)
            for story in json.loads(args.storyboards.read_text())['stories']]
    result = {'source': str(args.storyboards),
              'segment_rule': f'production greedy packing, windows of at most {args.max_frames} source frames at 16 fps',
              'by_split': {split: summarize([row for row in rows if row['split'] == split]) for split in ('dev', 'test')},
              'stories': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result['by_split'], ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
