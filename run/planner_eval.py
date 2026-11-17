#!/usr/bin/env python3
"""分镜表规划器 CPU 结构验收；保留全部提供样本，分别列缺答、截断、JSON/结构失败，不计算语义效果分。"""
import argparse
import json
from pathlib import Path

from mspc.evaluation.storyboard_planner import evaluate_structure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--answers', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    def read(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = evaluate_structure(read(args.samples), read(args.answers))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'rows.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in result['rows']))
    (args.output_dir / 'summary.json').write_text(json.dumps(result['summary'], ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
