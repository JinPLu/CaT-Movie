"""统一benchmark入口：成片辅助指标与整片五项评分，共用请求、结果和汇总。

不依赖逐镜断言、源片排除清单、检测框或旧判官金标子集。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import subprocess
from itertools import product

from mspc.core.vlm_media import video_data_url
from mspc.data.openrouter_client import (
    OpenRouterClient, OpenRouterCostCounter, OpenRouterResponseError,
    load_openrouter_api_key,
)

from mspc.evaluation.video_metrics import METRICS, validate_cut_request, VideoMetricScorer

DIMENSIONS = ("character_consistency", "object_consistency", "location_consistency",
              "event_completion", "state_continuity_and_updates")
SAMPLE_FIELDS = ("method", "story_id", "input_tier", "seed")
RATING_FIELDS = {"question_number", "status", "score", "reason", "timestamps_seconds"}


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class VideoNotIngestedError(RuntimeError):
    """The provider billed the call but the video never reached the model; the answer is not a rating."""


def validate_ratings(data: dict, duration_seconds: float) -> list[dict]:
    if not isinstance(data, dict) or set(data) != {"ratings"}:
        raise ValueError("response must contain only ratings")
    ratings = data["ratings"]
    if not isinstance(ratings, list) or len(ratings) != 5:
        raise ValueError("exactly five ratings required")
    for number, row in enumerate(ratings, 1):
        if not isinstance(row, dict) or set(row) != RATING_FIELDS:
            raise ValueError("each rating must contain the five declared fields")
        if type(row["question_number"]) is not int or row["question_number"] != number:
            raise ValueError("ratings must be ordered from 1 to 5")
        if row["status"] == "rated":
            if type(row["score"]) is not int or not 1 <= row["score"] <= 4:
                raise ValueError("rated scores must be integers in 1..4")
        elif row["status"] == "not_applicable":
            if row["score"] is not None:
                raise ValueError("not_applicable requires null score")
        else:
            raise ValueError("status must be rated or not_applicable")
        if not isinstance(row["reason"], str) or not row["reason"].strip():
            raise ValueError("each rating requires evidence or an applicability reason")
        times = row["timestamps_seconds"]
        if not isinstance(times, list) or any(
            type(t) not in (int, float) or not math.isfinite(t) or not 0 <= t <= duration_seconds
            for t in times
        ):
            raise ValueError("evidence times must lie within the delivered video")
    return ratings


def sample_key(row: dict) -> tuple:
    return tuple(row[field] for field in SAMPLE_FIELDS)


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest["evaluation_scope"] not in ("renderer", "system"):
        raise ValueError("evaluation_scope must be renderer or system")
    rows = manifest["requests"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty request manifest")
    keys = set()
    for row in rows:
        for field in ("method", "story_id", "story_request", "video_path"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"missing {field}")
        if any('/' in row[field] or '\\' in row[field] or row[field] in ('.', '..')
               for field in ('method', 'story_id')):
            raise ValueError("method/story must be simple path components")
        if row["input_tier"] not in ("synopsis", "paragraph", "shots"):
            raise ValueError("input_tier must be synopsis, paragraph or shots")
        if type(row["seed"]) is not int or row["seed"] < 0:
            raise ValueError("seed must be a non-negative integer")
        validate_cut_request(row)
        key = sample_key(row)
        if key in keys:
            raise ValueError(f"duplicate method/story/tier/seed {key}")
        keys.add(key)
        generation = row['generation_record_path']
        if generation is not None and (not isinstance(generation, str) or not generation.strip()):
            raise ValueError('generation_record_path must be a path or null for pre-existing videos')
    expected = set()
    conditions = set()
    for comparison in manifest['comparisons']:
        condition = (comparison['input_tier'], comparison['seed'])
        if condition in conditions:
            raise ValueError(f'duplicate comparison {condition}')
        conditions.add(condition)
        methods, stories = comparison['methods'], comparison['story_ids']
        if not methods or not stories or len(set(methods)) != len(methods) or len(set(stories)) != len(stories):
            raise ValueError('comparison methods and story_ids must be nonempty and unique')
        expected.update((method, story, *condition) for method, story in product(methods, stories))
    if keys != expected:
        raise ValueError(f'requests differ from declared comparisons; missing={sorted(expected - keys)}, extra={sorted(keys - expected)}')
    # Same experimental condition must ask every method for the same story.
    requests = {}
    for row in rows:
        key = (row["story_id"], row["input_tier"], row["seed"])
        if key in requests and requests[key] != (row["story_request"], row["expected_shot_count"], row["expected_cut_times_seconds"]):
            raise ValueError(f"methods have different evaluation requests for {key}")
        requests[key] = (row["story_request"], row["expected_shot_count"], row["expected_cut_times_seconds"])
    return manifest


def result_path(output: Path, row: dict) -> Path:
    return output / row["method"] / row["input_tier"] / f"seed-{row['seed']}" / (row["story_id"] + ".json")


def inspect_video(path: Path, ffprobe: str) -> dict:
    """读成片时长；不拿计划帧数／16代替，也不要求与计划逐镜对齐。"""
    if not path.is_file():
        raise FileNotFoundError(path)
    result = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration,width,height,avg_frame_rate", "-of", "json", str(path)],
        check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError(f"{path}: one video stream required")
    media = streams[0]
    duration = float(media["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"{path}: invalid video duration")
    return {**media, "duration_seconds": duration}


def summarize(manifest: dict, output: Path) -> dict:
    grouped = defaultdict(list)
    for row in manifest["requests"]:
        grouped[(row["method"], row["input_tier"], row["seed"])].append(row)
    groups = []
    applicability = defaultdict(dict)
    for (method, tier, seed), cases in sorted(grouped.items()):
        columns = {name: [] for name in (*METRICS, *DIMENSIONS)}
        statuses = {name: defaultdict(int) for name in columns}
        generation_counts = defaultdict(int)
        for case in cases:
            path = result_path(output, case)
            record = json.loads(path.read_text()) if path.exists() else {}
            generation = read_generation(case)
            generation_counts[generation['status']] += 1
            quality = record.get('non_vlm', {})
            judge = record.get('judge', {})
            for name in METRICS:
                item = quality.get('metrics', {}).get(name, {'status': quality.get('status', record.get('media_status', missing_status(generation)))})
                statuses[name][item['status']] += 1
                if item['status'] == 'measured':
                    columns[name].append(item['value'])
            if judge.get('status') == 'rated':
                ratings = validate_ratings({'ratings': judge['ratings']}, record['media']['duration_seconds'])
                for name, item in zip(DIMENSIONS, ratings, strict=True):
                    applicability[(case['story_id'], tier, seed, name)][method] = item['status']
                    statuses[name][item['status']] += 1
                    if item['status'] == 'rated':
                        columns[name].append(item['score'])
            else:
                for name in DIMENSIONS:
                    statuses[name][judge.get('status', record.get('media_status', missing_status(generation)))] += 1
        groups.append({'method': method, 'input_tier': tier, 'seed': seed,
            'stories_expected': len(cases), 'story_ids': sorted(c['story_id'] for c in cases),
            'generation_status_counts': dict(generation_counts),
            'dimensions': {name: {'mean': sum(values) / len(values) if values else None,
                'n': len(values), 'status_counts': dict(statuses[name]),
                **({'score_counts': {str(k): values.count(k) for k in range(1, 5)}} if name in DIMENSIONS else {})}
                for name, values in columns.items()}})
    disagreements = [{'story_id': story, 'input_tier': tier, 'seed': seed, 'dimension': name,
                      'methods': methods} for (story, tier, seed, name), methods in sorted(applicability.items())
                     if len(set(methods.values())) > 1]
    return {'evaluation_scope': manifest['evaluation_scope'], 'groups': groups,
            'applicability_disagreements': disagreements}


def read_generation(row: dict) -> dict:
    path = row['generation_record_path']
    if path is None:
        return {'status': 'pre_existing'}
    source = Path(path)
    return json.loads(source.read_text()) if source.exists() else {'status': 'pending'}


def missing_status(generation: dict) -> str:
    return 'pending' if generation['status'] in ('completed', 'pre_existing') else generation['status']


def score(args):
    manifest = load_manifest(args.manifest)
    config = json.loads(os.path.expandvars(args.config.read_text()))
    template = Path(config['judge']['prompt']).read_text().strip()
    response_format = json.loads(Path(config['judge']['response_format']).read_text())
    judge_config = json.loads(Path(config['judge']['config']).read_text())
    if template.count('{story_request}') != 1:
        raise ValueError('prompt requires exactly one story_request placeholder')
    inputs = {'manifest': manifest, 'config': config, 'judge_config': judge_config,
              'prompt': template, 'response_format': response_format}
    args.output.mkdir(parents=True, exist_ok=True)
    identity_path = args.output / 'inputs.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != inputs:
        raise ValueError('output belongs to different inputs or evaluation configuration')
    write_json(identity_path, inputs)
    use_judge = args.components in ('judge', 'all')
    counter = None
    if use_judge:
        if args.budget_usd is None or args.api_key_file is None or args.proxy is None:
            raise ValueError('judge requires --budget-usd, --api-key-file and --proxy')
        spent = Decimal(0)
        for row in manifest['requests']:
            target = result_path(args.output, row)
            saved = json.loads(target.read_text()).get('judge') if target.exists() else None
            if saved:
                if saved['cost_usd'] is None:
                    raise ValueError(f'unresolved billed request at {target}; reconcile before continuing')
                spent += Decimal(saved['cost_usd'])
        counter = OpenRouterCostCounter(hard_cap_usd=args.budget_usd,
            prices={judge_config['model']: judge_config['price']}, already_spent_usd=spent,
            precall_prompt_tokens=judge_config['precall_prompt_tokens'],
            precall_completion_tokens=judge_config['max_tokens'])
        key = load_openrouter_api_key(args.api_key_file)
    with VideoMetricScorer(config['non_vlm']) as metric_scorer:
        processed = 0
        for row in manifest['requests']:
            target = result_path(args.output, row)
            record = json.loads(target.read_text()) if target.exists() else {field: row[field] for field in SAMPLE_FIELDS}
            record['generation'] = read_generation(row)
            if record['generation']['status'] not in ('completed', 'pre_existing'):
                write_json(target, record)
                continue
            need_metrics = args.components in ('metrics', 'all') and record.get('non_vlm', {}).get('status') != 'completed'
            need_judge = use_judge and 'judge' not in record
            if not need_metrics and not need_judge:
                continue
            if args.limit is not None and processed >= args.limit:
                break
            try:
                try:
                    media = inspect_video(Path(row['video_path']), config['ffprobe'])
                    record['media'] = media
                    record.pop('media_status', None)
                    record.pop('media_error', None)
                except Exception as error:
                    record.update(media_status='technical_failed', media_error=f'{type(error).__name__}: {error}')
                    raise
                if need_metrics:
                    try:
                        record['non_vlm'] = {'status': 'completed', **metric_scorer.score_video(row, media, config['ffprobe'])}
                    except Exception as error:
                        record['non_vlm'] = {'status': 'technical_failed', 'error': f'{type(error).__name__}: {error}'}
                        raise
                    write_json(target, record)
                if need_judge:
                    video = video_data_url(Path(row['video_path']))
                    counter.reserve_precall(judge_config['model'])
                    judge = {'status': 'request_started', 'cost_usd': None}
                    record['judge'] = judge
                    write_json(target, record)
                    def recorder(request, raw, error, elapsed):
                        raw = raw.model_dump(mode='json') if hasattr(raw, 'model_dump') else raw
                        judge.update(raw_response=raw, error=error, seconds=elapsed,
                                     status='request_failed' if error else 'response_received')
                        write_json(target, record)
                    client = OpenRouterClient(api_key=key, base_url='https://openrouter.ai/api/v1',
                        cost_counter=counter, timeout_seconds=300, proxy=args.proxy, recorder=recorder)
                    try:
                        response = client.complete_video(model=judge_config['model'], video_data_url=video,
                            prompt=template.replace('{story_request}', row['story_request']),
                            temperature=None, max_tokens=judge_config['max_tokens'],
                            request_options=judge_config['request_options'], response_format=response_format)
                        judge.update(cost_usd=response.cost_usd, usage=response.usage, served=response.served,
                                     request_id=response.request_id, text=response.text)
                        # 10-19 探针：写错 processing 时视频被静默丢掉，判官仍凭文字给出完整五项分。
                        video_tokens = (response.served['prompt_tokens_details'] or {}).get('video_tokens')
                        required = judge_config['min_video_tokens_per_second'] * media['duration_seconds']
                        if type(video_tokens) is not int or video_tokens < required:
                            judge.update(status='video_not_ingested',
                                         error=f'video_tokens={video_tokens} below {required:.0f} for {media["duration_seconds"]}s')
                            raise VideoNotIngestedError(judge['error'])
                        judge['ratings'] = validate_ratings(json.loads(response.text), media['duration_seconds'])
                        judge['status'] = 'rated'
                    except OpenRouterResponseError as error:
                        judge.update(status='response_failed', error=str(error), error_stage=error.stage,
                                     cost_usd=error.cost_usd, usage=error.usage, text=error.text)
                        raise
                    except (ValueError, TypeError, KeyError) as error:
                        judge.update(status='format_failed', error=str(error))
                        raise
            finally:
                write_json(target, record)
                write_json(args.output / 'summary.json', summarize(manifest, args.output))
            processed += 1
            print(json.dumps({**{field: row[field] for field in SAMPLE_FIELDS},
                              'metrics': record.get('non_vlm', {}).get('status'),
                              'judge': record.get('judge', {}).get('status')}), flush=True)
        write_json(args.output / 'summary.json', summarize(manifest, args.output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('score', 'summarize'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/eval/benchmark.json'))
    parser.add_argument('--components', choices=('all', 'metrics', 'judge'), default='metrics',
                        help='default: metrics; judge/all reproduce the historical OpenRouter contract')
    parser.add_argument('--api-key-file', type=Path)
    parser.add_argument('--budget-usd')
    parser.add_argument('--proxy')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.command == 'score':
        score(args)
    else:
        manifest = load_manifest(args.manifest)
        if json.loads((args.output / 'inputs.json').read_text())['manifest'] != manifest:
            raise ValueError('manifest differs from the scored inputs')
        write_json(args.output / 'summary.json', summarize(manifest, args.output))


if __name__ == '__main__':
    main()
