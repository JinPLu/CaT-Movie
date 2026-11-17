"""逐镜在场与地点只按请求判。

方法只拿到请求，所以「谁在这一镜」「这一镜在哪个地点」也只按请求读：gpt-6.1-sol 读请求原文（不给标注、不给源片），
逐镜给出请求要求出镜的已登记人、动物、群体与物体（只被看着、说到、提到的不算），以及读者能从请求看出的地点
（只登记一个地点的故事每镜都是它；看不出记 null）。规则文字在 ``configs/eval/request_presence/prompt.md``。

用法（本机，每故事一次 codex exec，回执与结果在 RUN/stories/<story>/）::

    python -m mspc.evaluation.request_presence run --bank questions.json --stories S1 S2 ... --output RUN --workers 15
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROMPT = ROOT / 'configs/eval/request_presence/prompt.md'
SCHEMA = ROOT / 'configs/eval/request_presence/schema.json'
MODEL, EFFORT = 'gpt-6.1-sol', 'high'
MAX_WORKERS = 15
SCRATCH = Path('/tmp/request-presence')
_REGISTERED = re.compile(r'^- ((?:Character|Object|Location) \d+)\b', re.M)
_SHOT = re.compile(r'^s(\d+): ', re.M)


def registered(request: str) -> tuple[set[str], set[str]]:
    """请求登记的人／物编号与地点编号。"""
    ids = set(_REGISTERED.findall(request))
    return {i for i in ids if not i.startswith('Location')}, {i for i in ids if i.startswith('Location')}


def validate(result: dict, request: str) -> None:
    entities, locations = registered(request)
    shots = [int(k) for k in _SHOT.findall(request)]
    if [s['shot'] for s in result['shots']] != shots:
        raise ValueError(f"shots {[s['shot'] for s in result['shots']]} do not match the request {shots}")
    for s in result['shots']:
        if not set(s['on_screen']) <= entities:
            raise ValueError(f"shot {s['shot']}: unregistered ids {sorted(set(s['on_screen']) - entities)}")
        if s['location'] is not None and s['location'] not in locations:
            raise ValueError(f"shot {s['shot']}: unregistered location {s['location']}")
    if len(locations) == 1 and any(s['location'] is None for s in result['shots']):
        raise ValueError('single-location request left a shot without its location')


def run_story(sid: str, request: str, output: Path) -> dict:
    """一个故事一次 codex exec；有成功回执的不重发，失败回执要人工处理。"""
    work = output / 'stories' / sid
    work.mkdir(parents=True, exist_ok=True)
    prompt = PROMPT.read_text(encoding='utf-8') + '\n\n## Request\n' + request + '\n'
    prompt_path, answer, receipt = work / 'input.md', work / 'presence.json', work / 'receipt.json'
    if prompt_path.exists() and prompt_path.read_text(encoding='utf-8') != prompt:
        raise ValueError(f'{sid}: input differs from the earlier call')
    if receipt.exists():
        if json.loads(receipt.read_text())['returncode'] != 0:
            raise ValueError(f'{sid}: earlier call failed, manual review required: {work}')
    else:
        if answer.exists():
            raise ValueError(f'{sid}: result without receipt')
        prompt_path.write_text(prompt, encoding='utf-8')
        scratch = SCRATCH / output.resolve().name / sid  # 工作目录放在仓库外：仓库里的文档不是这一步的输入
        scratch.mkdir(parents=True, exist_ok=True)
        command = ['codex', 'exec', '-m', MODEL, '-c', f'model_reasoning_effort={EFFORT}', '--disable', 'memories',
                   '--sandbox', 'read-only', '--skip-git-repo-check', '--json', '--output-schema', str(SCHEMA),
                   '--output-last-message', str(answer.resolve()), '-C', str(scratch)]
        started = datetime.now(timezone.utc)
        with (work / 'events.jsonl').open('w') as stdout, (work / 'stderr.log').open('w') as stderr:
            result = subprocess.run(command, input=prompt, text=True, stdout=stdout, stderr=stderr, check=False)
        finished = datetime.now(timezone.utc)
        receipt.write_text(json.dumps({'command': command, 'started_at': started.isoformat(),
                                       'finished_at': finished.isoformat(),
                                       'seconds': round((finished - started).total_seconds(), 1),
                                       'returncode': result.returncode}, indent=2) + '\n')
        if result.returncode:
            raise ValueError(f'{sid}: codex exec failed; see {work}')
    if not answer.is_file():
        raise ValueError(f'{sid}: codex returned without structured output; see {receipt}')
    presence = json.loads(answer.read_text())
    validate(presence, request)
    return presence


def run(bank: dict, stories: list[str], output: Path, workers: int) -> dict:
    """并发至多 15；一个故事失败就不再领新故事，收齐在途的回执后报错。"""
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f'workers must be within 1..{MAX_WORKERS}')
    output.mkdir(parents=True, exist_ok=True)
    for source, name in ((PROMPT, 'prompt.md'), (SCHEMA, 'schema.json')):
        target = output / name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f'frozen input differs: {target}')
        target.write_bytes(source.read_bytes())
    done, failures, remaining = {}, {}, iter(stories)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}

        def submit_next() -> None:
            sid = next(remaining, None)
            if sid is not None:
                pending[pool.submit(run_story, sid, bank['stories'][sid]['request'], output)] = sid
        for _ in range(workers):
            submit_next()
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                sid = pending.pop(future)
                try:
                    done[sid] = future.result()
                    print(json.dumps({'story_id': sid, 'shots': len(done[sid]['shots']),
                                      'unknown_location': sum(s['location'] is None for s in done[sid]['shots'])}), flush=True)
                except Exception as exc:
                    failures[sid] = f'{type(exc).__name__}: {exc}'
                    print(json.dumps({'story_id': sid, 'failure': failures[sid]}), flush=True)
            if not failures:
                for _ in finished:
                    submit_next()
    if failures:
        raise RuntimeError(f'request presence stopped after failures: {failures}; not started: {list(remaining)}')
    return done


def load_run(output: Path) -> dict[str, dict]:
    """运行根里每个故事的结果。"""
    return {p.parent.name: json.loads(p.read_text()) for p in sorted((output / 'stories').glob('*/presence.json'))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    r = commands.add_parser('run')
    r.add_argument('--bank', type=Path, required=True, help='含各故事请求原文（stories.<id>.request）的题库')
    r.add_argument('--stories', nargs='+', required=True)
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--workers', type=int, required=True)
    args = parser.parse_args()
    run(json.loads(args.bank.read_text()), args.stories, args.output, args.workers)


if __name__ == '__main__':
    main()
