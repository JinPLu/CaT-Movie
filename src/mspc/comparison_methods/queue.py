"""已授权固定任务的多进程领取；复用原生监督，按租约占用扣除共享预算。"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path

from mspc.comparison_methods.dispatch import ResidentRenderer, now, publish_new_json, run_task, write_json
from mspc.core.process_supervision import termination_handler


def read(path):
    return json.loads(Path(path).read_text())


def remaining_seconds(leases, method, cap_hours, prior_hours, current):
    rows = [row for row in leases if row['method'] == method]
    used = prior_hours * 3600
    active = 0
    for row in rows:
        end = datetime.fromisoformat(row['released_at']) if row['released_at'] else current
        used += (end - datetime.fromisoformat(row['held_since'])).total_seconds() * row['gpu_count']
        if not row['released_at']:
            active += row['gpu_count']
    return (cap_hours * 3600 - used - 120 * active) / max(1, active)


@termination_handler()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--lease-id', required=True)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--resume-story-id')
    parser.add_argument('--resume-source-slurm-job')
    parser.add_argument('--resume-code-root', type=Path)
    parser.add_argument('--resume-code-revision')
    args = parser.parse_args()
    resume_fields = (args.resume_story_id, args.resume_source_slurm_job, args.resume_code_root, args.resume_code_revision)
    if any(value is not None for value in resume_fields) and not all(value is not None for value in resume_fields):
        parser.error('resume requires a story ID, source Slurm job, code root and code revision together')
    resume = None
    if args.resume_story_id is not None:
        if args.method != 'official_storymem_14b':
            parser.error('completed-prefix resume is only defined for official StoryMem')
        resume = {'source_slurm_job': args.resume_source_slurm_job,
                  'code_root': str(args.resume_code_root), 'code_revision': args.resume_code_revision}
    root = args.run_root
    config = read(root / 'inputs/queue.json')
    method = config['methods'][args.method]
    if not method['authorized']:
        raise ValueError('method is not released for generation')
    tasks = read(method['queue'])
    if len({task['story_id'] for task in tasks}) != len(tasks):
        raise ValueError('duplicate story in frozen queue')
    if resume is not None:
        tasks = [task for task in tasks if task['story_id'] == args.resume_story_id]
        if len(tasks) != 1:
            raise ValueError('resume story is not uniquely present in the frozen queue')
    dispatch = read(root / 'inputs/dispatch.json')
    leases_path = root / 'manifests/leases.json'
    lease, = [row for row in read(leases_path)['leases']
              if row['lease_id'] == args.lease_id and not row['released_at'] and args.device in row['devices']]
    if lease['method'] != args.method or lease['released_at'] or args.device not in lease['devices']:
        raise ValueError('worker does not match its explicit lease')
    workers = root / 'logs/workers'
    workers.mkdir(exist_ok=True)
    worker_path = workers / f'{args.worker}.json'
    state = dict(worker=args.worker, method=args.method, device=args.device, lease_id=args.lease_id,
                 pid=os.getpid(), started_at=now(), status='running', completed=0, failed=0)
    with (workers / 'workers.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = [read(path) for path in workers.glob('*.json')]
        if any(row['status'] == 'running' and row['lease_id'] == args.lease_id and row['device'] == args.device for row in existing):
            raise ValueError('another worker owns this leased device')
        if sum(row['status'] == 'running' and row['method'] == args.method for row in existing) >= method['max_workers']:
            raise ValueError('method worker limit reached')
        publish_new_json(worker_path, state)
    pause = root / 'logs' / f'{args.method}-pause.json'

    def remaining():
        return remaining_seconds(read(leases_path)['leases'], args.method, method['cap_gpu_hours'],
                                 method['prior_gpu_hours'], datetime.fromisoformat(now()))

    renderer = ResidentRenderer(dispatch, args.method, args.device)
    try:
        for task in tasks:
            if pause.exists():
                state['status'] = 'paused_after_engineering_failure'
                break
            if remaining() <= 0:
                state['status'] = 'budget_exhausted'
                break
            if resume is None and Path(task['generation_record_path']).exists():
                continue
            try:
                result = run_task(root, dispatch, task, args.device, args.lease_id,
                                  method['story_limit_seconds'], remaining, resume=resume, renderer=renderer)
            except FileExistsError:
                if resume is not None:
                    raise
                continue  # 原生监督的独占记录已经被另一进程领取。
            state['completed' if result['status'] == 'completed' else 'failed'] += 1
            state.update(last_story_id=task['story_id'], last_finished_at=now())
            write_json(worker_path, state)
            if result['status'] != 'completed':
                try:
                    publish_new_json(pause, dict(created_at=now(), story_id=task['story_id'], result=result))
                except FileExistsError:
                    pass
                state['status'] = 'paused_after_engineering_failure'
                break
        else:
            state['status'] = 'finished'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        renderer.close()
        state['finished_at'] = now()
        state['release_owner'] = 'GPU scheduler after every process under this lease is terminal'
        write_json(worker_path, state)


if __name__ == '__main__':
    main()
