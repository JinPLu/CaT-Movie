"""在主服务器 GPU 机上常驻：本机各卡原 worker 结束后，按接手计划在同一租约、同一张卡上启动下一个方法的队列 worker。

每台 GPU 机只管自己端口下的卡，状态写 `manifests/handoff-state-<端口>.json`，日志写 `logs/handoff-<端口>.log`。
链上方法都领完后该卡不再接手，所有卡都结束时进程退出，空卡由 GPU scheduler 空闲回收。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    tmp = Path(f'{path}.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def unclaimed(run: dict) -> int:
    root = Path(run['run_root'])
    if (root / 'logs' / f"{run['method']}-pause.json").exists():
        return 0
    queue = read(read(root / 'inputs/queue.json')['methods'][run['method']]['queue'])
    return sum(not Path(t['generation_record_path']).exists()
               and not (root / 'logs' / f"{run['method']}-{t['story_id']}-process.json").exists() for t in queue)


def update_leases(root: Path, change) -> None:
    path = root / 'manifests/leases.json'
    with open(f'{path}.lock', 'a') as lock:
        fcntl.lockf(lock, fcntl.LOCK_EX)
        data = read(path)
        change(data['leases'])
        write(path, data)


def release(root: Path, key: str, at: str) -> None:
    def change(rows):
        for row in rows:
            if row['lease_id'] == key and row['released_at'] is None:
                row['released_at'] = at
    update_leases(root, change)


def step(plan: dict, control: Path, port: int, state: dict, log) -> bool:
    now = datetime.now(timezone.utc).isoformat()
    active = False
    for slot in [s for s in plan['slots'] if s['ssh_port'] == port]:
        sid = f"{slot['lease_id']}-{slot['device']}"
        history = state.setdefault(sid, [{'run': 'initial', 'root': str(control), 'worker': slot['after_worker'],
                                          'row': slot['lease_id']}])
        last = history[-1]
        if last.get('done'):
            continue
        active = True
        worker_path = Path(last['root']) / 'logs/workers' / f"{last['worker']}.json"
        if not worker_path.exists():
            log(f"{sid}: {last['worker']} 尚无 worker 记录（启动于 {last.get('launched_at')}）")
            continue
        status = read(worker_path)['status']
        if status == 'running':
            continue
        used = {h['run'] for h in history}
        choice = next((name for name in slot['chain'] if name not in used and unclaimed(plan['runs'][name])), None)
        if last['run'] == 'initial':
            siblings = [s for s in plan['slots'] if s['lease_id'] == slot['lease_id']]
            if all(s is slot or len(state.get(f"{s['lease_id']}-{s['device']}", [None])) > 1 for s in siblings):
                release(control, slot['lease_id'], now)
        else:
            release(Path(last['root']), last['row'], now)
        if choice is None:
            history.append({'done': True, 'at': now})
            log(f"{sid}: {last['worker']} 结束（{status}），链上方法都已领完，卡空出")
            continue
        run = plan['runs'][choice]
        root = Path(run['run_root'])
        update_leases(root, lambda rows: rows.append({
            'lease_id': sid, 'scheduler_lease': slot['lease_id'], 'method': run['method'],
            'server_id': slot['server_id'], 'devices': [slot['device']], 'gpu_count': 1,
            'held_since': now, 'released_at': None}))
        worker = f"{choice}-{port}-{slot['device']}"
        (root / 'logs/stdout').mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, MSPC_SHARED_ROOT=plan['shared_root'], PYTHONPATH=run['control_src'], PYTHONUNBUFFERED='1',
                   CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=slot['device'], LD_LIBRARY_PATH='')
        with open(root / 'logs/stdout' / f'{worker}.log', 'a') as out:
            subprocess.Popen([run['python'], '-m', 'mspc.comparison_methods.queue', '--run-root', str(root),
                              '--method', run['method'], '--device', slot['device'], '--lease-id', sid,
                              '--worker', worker], env=env, stdin=subprocess.DEVNULL, stdout=out,
                             stderr=subprocess.STDOUT, start_new_session=True)
        history.append({'run': choice, 'root': str(root), 'worker': worker, 'row': sid, 'launched_at': now})
        log(f"{sid}: {last['worker']} 结束（{status}），接手 {worker}（{run['method']}，余 {unclaimed(run)} 部待领）")
    return active


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-root', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--interval', type=int, required=True)
    args = parser.parse_args()
    plan = read(args.control_root / 'inputs/handoff.json')
    state_path = args.control_root / f'manifests/handoff-state-{args.port}.json'
    state = read(state_path) if state_path.exists() else {}
    log_path = args.control_root / f'logs/handoff-{args.port}.log'

    def log(message):
        with open(log_path, 'a') as stream:
            stream.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}\n")

    log(f'监督启动 pid={os.getpid()}')
    while True:
        active = step(plan, args.control_root, args.port, state, log)
        write(state_path, state)
        if not active:
            log('本机所有卡的方法链都已结束，监督退出')
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
