"""Benchmark唯一生成入口：prepare固定任务 → plan共享规划 → render按方法出片。

评分使用prepare写出的同一requests.json，不扫描成功视频重建分母。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
from mspc.config.shared_root import expand_shared_root_token

from mspc.data.planner_dataset import production_script
from mspc.data.storyboard import validate_storyboard
from mspc.evaluation.benchmark import load_manifest, write_json
from mspc.evaluation.video_metrics import validate_cut_request
from mspc.comparison_methods.prompts import FAMILIES

TIERS = ('synopsis', 'paragraph', 'shots')


def load_config(path: Path) -> dict:
    config = json.loads(os.path.expandvars(expand_shared_root_token(path.read_text(), str(path))))
    for method, settings in config['methods'].items():
        # 配置引用只在准备时展开；运行快照始终保存完整值，不依赖之后的文件移动。
        if isinstance(settings, str):
            source = path.parent / settings
            settings = json.loads(os.path.expandvars(expand_shared_root_token(source.read_text(), str(source))))
            config['methods'][method] = settings
        if not isinstance(settings, dict):
            raise TypeError(f'{method}: settings must be an object or an explicit JSON file path')
        if settings['planner'] not in config['planners']:
            raise ValueError(f'{method}: planner is not configured')
        if settings['family'] not in FAMILIES:
            raise ValueError(f'{method}: unknown renderer family')
        if settings['family'] == 'holocine':
            output_fps = settings['sampling']['output_fps']
            if type(output_fps) is not int or output_fps <= 0:
                raise ValueError(f'{method}: sampling output_fps must be a positive integer')
        if settings['family'] == 'storymem':
            memory = settings['memory']
            for key in ('fixed_first_frames', 'max_frames'):
                if type(memory[key]) is not int or memory[key] <= 0:
                    raise ValueError(f'{method}: memory {key} must be a positive integer')
            if not 0 < memory['fixed_first_frames'] < memory['max_frames']:
                raise ValueError(f'{method}: memory requires 0 < fixed_first_frames < max_frames')
        if settings['family'] in ('wan_single', 'wan_segment'):
            from mspc.inference.segment_story import HISTORY_POLICIES
            if settings['history_policy'] not in HISTORY_POLICIES:
                raise ValueError(f'{method}: history_policy must be one of {list(HISTORY_POLICIES)}')
            # own_shot 是评估干预：同一权重推理时只读本镜 KV，只对无参考前缀的模型有定义。
            if settings['communication'] not in ('normal', 'own_shot'):
                raise ValueError(f'{method}: communication must be normal or own_shot')
            if settings['communication'] == 'own_shot' and settings['model']['memory_prefix'] is not None:
                raise ValueError(f'{method}: own_shot is only defined without reference frames')
    return config


def load_requests(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data['evaluation_scope'] not in ('system', 'renderer'):
        raise ValueError('request catalog scope must be system or renderer')
    seen = set()
    for story in data['stories']:
        name = story['story_id']
        if name in seen or not name or '/' in name or '\\' in name:
            raise ValueError(f'duplicate or invalid story ID {name!r}')
        seen.add(name)
        if story['split'] not in ('dev', 'test'):
            raise ValueError(f'{name}: split must be dev or test')
        if type(story['fps']) is not int or story['fps'] < 1:
            raise ValueError(f'{name}: fps must be a positive integer')
        if type(story['target_seconds']) not in (int, float) or not math.isfinite(story['target_seconds']) or story['target_seconds'] <= 0:
            raise ValueError(f'{name}: target_seconds must be positive')
        if not story['inputs'] or set(story['inputs']) - set(TIERS):
            raise ValueError(f'{name}: unknown or empty tiers')
        for tier, request in story['inputs'].items():
            if not isinstance(request['text'], str) or not request['text'].strip():
                raise ValueError(f'{name}/{tier}: empty user request')
            validate_cut_request(request)
        if data['evaluation_scope'] == 'renderer':
            validate_storyboard(story['storyboard'])
            if set(story['inputs']) != {'shots'}:
                raise ValueError('given-plan renderer diagnosis has one shots condition')
    if not seen:
        raise ValueError('empty request catalog')
    return data


def production_request(story: dict, tier: str, scope: str) -> str:
    text = story['inputs'][tier]['text']
    if scope == 'renderer':
        return text
    return production_script(script=text, target_seconds=story['target_seconds'], fps=story['fps'])


def plan_path(output: Path, planner: str, row: dict) -> Path:
    return output / 'inputs' / 'plans' / planner / row['input_tier'] / f"seed-{row['seed']}" / (row['story_id'] + '.json')


def prepare(*, catalog: dict, config: dict, methods: list[str], tiers: list[str], split: str,
            seeds: list[int], output: Path) -> dict:
    if len(set(methods)) != len(methods) or not methods or not set(methods) <= set(config['methods']):
        raise ValueError('methods must name distinct configured methods')
    if len(set(tiers)) != len(tiers) or not tiers or not set(tiers) <= set(TIERS):
        raise ValueError('tiers must be distinct known tiers')
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError('seeds must be distinct non-negative integers')
    stories = [story for story in catalog['stories'] if story['split'] == split]
    if not stories:
        raise ValueError(f'no stories for split {split}')
    for story in stories:
        if not set(tiers) <= set(story['inputs']):
            raise ValueError(f"{story['story_id']}: requested tiers are missing")
    for method in methods:
        selected = config['methods'][method]
        required = [selected, config['planners'][selected['planner']]] if catalog['evaluation_scope'] == 'system' else [selected]
        if '${' in json.dumps(required):
            raise ValueError(f'{method}: unresolved configuration environment variable')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows, comparisons = [], []
    for tier in tiers:
        for seed in seeds:
            comparisons.append({'input_tier': tier, 'seed': seed, 'methods': methods,
                                'story_ids': [story['story_id'] for story in stories]})
            for story in stories:
                request = story['inputs'][tier]
                for method in methods:
                    directory = output / 'inference' / 'raw' / 'videos' / method / tier / f'seed-{seed}' / story['story_id']
                    row = {'method': method, 'story_id': story['story_id'], 'input_tier': tier, 'seed': seed,
                           'story_request': production_request(story, tier, catalog['evaluation_scope']),
                           'target_seconds': story['target_seconds'], 'fps': story['fps'],
                           'expected_shot_count': request['expected_shot_count'],
                           'expected_cut_times_seconds': request['expected_cut_times_seconds'],
                           'video_path': str(directory / 'video.mp4'), 'generation_record_path': str(directory / 'generation.json')}
                    rows.append(row)
                    if catalog['evaluation_scope'] == 'renderer':
                        target = plan_path(output, config['methods'][method]['planner'], row)
                        write_json(target, {'status': 'completed', 'structure_status': 'valid',
                                           'origin': 'given_plan', 'storyboard': story['storyboard']})
    manifest = {'evaluation_scope': catalog['evaluation_scope'], 'comparisons': comparisons, 'requests': rows}
    write_json(output / 'inputs' / 'config.json', config)
    write_json(output / 'inputs' / 'stories.json', {**catalog, 'stories': stories})
    write_json(output / 'inputs' / 'requests.json', manifest)
    return load_manifest(output / 'inputs' / 'requests.json')


def planning_tasks(output: Path, manifest: dict, config: dict) -> dict[str, list[dict]]:
    tasks, seen = defaultdict(list), set()
    for row in manifest['requests']:
        planner = config['methods'][row['method']]['planner']
        path = plan_path(output, planner, row)
        if path not in seen:
            tasks[planner].append(row)
            seen.add(path)
    return tasks


def plan(output: Path, planners: list[str] | None, *, workers: int) -> None:
    """``workers`` 路并发调同一个规划服务：vLLM 连续批处理，并发低于服务的 ``--max-num-seqs`` 时卡在等请求
    （11-12 实测：每卡 4 路时 A100 大半空闲）。续跑跳过已有记录；每条记录仍由 :func:`plan_story` 先落盘再请求。"""
    from concurrent.futures import ThreadPoolExecutor
    from mspc.inference.story_planner import plan_story, planner_client
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    manifest = load_manifest(output / 'inputs' / 'requests.json')
    if manifest['evaluation_scope'] == 'renderer':
        raise ValueError('given-plan diagnosis does not call a planner')
    config = json.loads((output / 'inputs' / 'config.json').read_text())
    tasks = planning_tasks(output, manifest, config)
    if planners is not None and not set(planners) <= set(tasks):
        raise ValueError('requested planner is outside this run')
    for planner, rows in tasks.items():
        if planners is not None and planner not in planners:
            continue
        pending = [row for row in rows if not plan_path(output, planner, row).exists()]
        if not pending:
            continue
        with planner_client(config['planners'][planner]) as client:
            def run(row):
                return row, plan_story(row, config['planners'][planner], config['sampling'],
                                       plan_path(output, planner, row), client.chat.completions.create)
            with ThreadPoolExecutor(workers) as pool:
                for row, record in pool.map(run, pending):
                    print(json.dumps({'planner': planner, 'story_id': row['story_id'], 'tier': row['input_tier'],
                                      'seed': row['seed'], 'status': record['status'],
                                      'elapsed_seconds': record.get('elapsed_seconds')}), flush=True)


def recompile(output: Path, planners: list[str] | None) -> list[dict]:
    """有原答的记录按当前编译规则重新编译（不再调规划器）：编译规则改过之后（开镜对齐、字段级丢弃），同一份原答
    得到新的分镜。原状态与错误记进 ``previous_status``、``previous_error``。就地改写，要保留旧记录先拷一份目录。"""
    from mspc.inference.story_planner import compile_record
    manifest = load_manifest(output / 'inputs' / 'requests.json')
    config = json.loads((output / 'inputs' / 'config.json').read_text())
    changed = []
    for planner, rows in planning_tasks(output, manifest, config).items():
        if planners is not None and planner not in planners:
            continue
        for row in rows:
            path = plan_path(output, planner, row)
            record = json.loads(path.read_text())
            if 'raw_content' not in record:
                continue
            previous, previous_status = record.get('error'), record['status']
            for key in ('error', 'structure_status', 'raw_storyboard', 'storyboard', 'compilation'):
                record.pop(key, None)
            compile_record(record, row)
            record.update(previous_status=previous_status, previous_error=previous)
            write_json(path, record)
            changed.append({'story_id': row['story_id'], 'status': record['status'], 'previous_status': previous_status,
                            'previous_error': previous})
    return changed


def render(output: Path, method: str, story_ids: list[str] | None) -> None:
    from mspc.comparison_methods.renderer import render_tasks
    manifest = load_manifest(output / 'inputs' / 'requests.json')
    config = json.loads((output / 'inputs' / 'config.json').read_text())
    rows = [row for row in manifest['requests'] if row['method'] == method]
    if not rows:
        raise ValueError('method is outside the fixed comparison')
    if story_ids is not None:
        if not set(story_ids) <= {row['story_id'] for row in rows}:
            raise ValueError('story IDs are outside the fixed comparison')
        rows = [row for row in rows if row['story_id'] in story_ids]
    stories = {story['story_id']: story for story in json.loads((output / 'inputs' / 'stories.json').read_text())['stories']}
    render_tasks(output, config, method, rows, stories)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--requests', type=Path, required=True)
    prep.add_argument('--config', type=Path, required=True)
    prep.add_argument('--methods', nargs='+', required=True)
    prep.add_argument('--tiers', nargs='+', choices=TIERS, required=True)
    prep.add_argument('--split', choices=('dev', 'test'), required=True)
    prep.add_argument('--seeds', nargs='+', type=int, required=True)
    prep.add_argument('--output', type=Path, required=True)
    planning = commands.add_parser('plan')
    planning.add_argument('--output', type=Path, required=True)
    planning.add_argument('--planners', nargs='+')
    planning.add_argument('--workers', type=int, required=True, help='并发请求数，取服务的 --max-num-seqs')
    recompiling = commands.add_parser('recompile')
    recompiling.add_argument('--output', type=Path, required=True)
    recompiling.add_argument('--planners', nargs='+')
    rendering = commands.add_parser('render')
    rendering.add_argument('--output', type=Path, required=True)
    rendering.add_argument('--method', required=True)
    rendering.add_argument('--story-ids', nargs='+')
    args = parser.parse_args()
    if args.command == 'prepare':
        manifest = prepare(catalog=load_requests(args.requests), config=load_config(args.config), methods=args.methods,
                           tiers=args.tiers, split=args.split, seeds=args.seeds, output=args.output)
        print(json.dumps({'tasks': len(manifest['requests']), 'manifest': str(args.output / 'inputs' / 'requests.json')}))
    elif args.command == 'plan':
        plan(args.output.resolve(), args.planners, workers=args.workers)
    elif args.command == 'recompile':
        print(json.dumps(recompile(args.output.resolve(), args.planners), ensure_ascii=False))
    else:
        render(args.output.resolve(), args.method, args.story_ids)


if __name__ == '__main__':
    main()
