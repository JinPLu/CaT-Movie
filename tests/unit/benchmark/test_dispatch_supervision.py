"""本地 CPU 进程复现：组长先退出、SIGTERM、跨进程登记竞争。"""
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
from mspc.core.process_supervision import SupervisorTerminated

from mspc.comparison_methods import dispatch as dispatch
from mspc.core.process_supervision import live_group_members, stop_process_group


def await_file(path, process):
    deadline = time.monotonic() + 5
    while not path.exists():
        if process.poll() is not None or time.monotonic() > deadline:
            pytest.fail('CPU child did not become ready')
        time.sleep(.01)


def test_group_cleanup_waits_for_descendant_after_launcher_has_exited(tmp_path):
    ready = tmp_path / 'child'
    child_code = f'import os,signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)'
    launcher = subprocess.Popen([sys.executable, '-c',
        f'import subprocess,sys; subprocess.Popen([sys.executable,"-c",{child_code!r}])'], start_new_session=True)
    launcher.wait(timeout=5)
    deadline = time.monotonic() + 5
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    try:
        assert ready.exists() and live_group_members(launcher.pid)
        dispatch.stop_child(launcher)
        assert live_group_members(launcher.pid) == []
    finally:
        stop_process_group(launcher.pid, .2)
        launcher.wait()


def test_sigterm_supervisor_cleans_owned_group_and_restores_handler(tmp_path):
    # 在独立监督进程发真实SIGTERM，不对pytest或任何研究任务发信号。
    ready = tmp_path / 'ready'
    result = tmp_path / 'result'
    code = f'''
import json, signal, subprocess, sys, time
from pathlib import Path
from mspc.core.process_supervision import termination_handler, stop_process_group, SupervisorTerminated
previous = signal.getsignal(signal.SIGTERM)
child = None
try:
    with termination_handler():
        try:
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
            Path({str(ready)!r}).write_text(str(child.pid))
            time.sleep(60)
        finally:
            stop_process_group(child.pid)
            child.wait()
except SupervisorTerminated:
    Path({str(result)!r}).write_text(json.dumps({{'restored': signal.getsignal(signal.SIGTERM)==previous, 'returncode':child.returncode}}))
'''
    supervisor = subprocess.Popen([sys.executable, '-c', code])
    try:
        await_file(ready, supervisor)
        supervisor.send_signal(signal.SIGTERM)
        assert supervisor.wait(timeout=10) == 0
        assert json.loads(result.read_text())['restored']
        assert live_group_members(int(ready.read_text())) == []
    finally:
        if supervisor.poll() is None:
            supervisor.kill(); supervisor.wait()
        if ready.exists():
            stop_process_group(int(ready.read_text()), .2)


def register_competitor(root, name, barrier, results):
    from mspc.comparison_methods import dispatch as owner
    publish = owner.publish_new_json
    def slow_publish(*args):
        time.sleep(.15)  # 原实现两个进程会同时越过check，再分别发布成功。
        publish(*args)
    owner.publish_new_json = slow_publish
    root = Path(root)
    barrier.wait(timeout=5)
    state = {'status': 'running', 'bounds': {'gpu_hours': 1, 'story_limit_seconds': 10, 'max_workers': 1}}
    try:
        owner.register_worker(root, 'model', root / f'model-development-worker-{name}.json', state)
        results.put('registered')
    except ValueError as error:
        results.put(str(error))


def test_concurrent_registration_keeps_global_worker_limit(tmp_path):
    context = multiprocessing.get_context('spawn')
    barrier = context.Barrier(2); results = context.Queue()
    workers = [context.Process(target=register_competitor, args=(str(tmp_path), str(i), barrier, results)) for i in range(2)]
    try:
        for worker in workers: worker.start()
        for worker in workers:
            worker.join(10)
            assert worker.exitcode == 0
        outcomes = [results.get(timeout=2) for _ in workers]
        assert outcomes.count('registered') == 1
        assert outcomes.count('development worker parallelism limit reached') == 1
        assert len(list(tmp_path.glob('model-development-worker-*.json'))) == 1
    finally:
        for worker in workers:
            if worker.is_alive(): worker.terminate(); worker.join()
        results.close()


def test_dispatch_sigterm_propagates_after_cleanup_instead_of_starting_next_story(tmp_path, monkeypatch):
    for relative in ('env/bin/ffmpeg', 'env/ffmpeg7/bin/ffprobe'):
        executable=tmp_path/relative;executable.parent.mkdir(parents=True,exist_ok=True)
        executable.write_text('#!/bin/sh\nexit 0\n');executable.chmod(0o755)
    method='official_multishotmaster_1_3b'
    config=dict(shared_root=str(tmp_path),cpu_threads=1,methods={method:dict(devices=1,
        code_root=str(tmp_path),python=sys.executable,code_revision='cpu-sigterm-reproduction',
        render_module='mspc.benchmark.generation')})
    task=dict(method=method,first_story=True,story_id='lsh-002-b',
        generation_record_path=str(tmp_path/'video/generation.json'),execution_output_root=str(tmp_path))
    original=subprocess.Popen;children=[]
    def cpu_only(argv,**kwargs):
        child=original([sys.executable,'-c','import time; time.sleep(60)'],**kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(dispatch.subprocess,'Popen',cpu_only)
    previous=signal.getsignal(signal.SIGTERM)
    def interrupt():
        os.kill(os.getpid(),signal.SIGTERM)
    with pytest.raises(SupervisorTerminated,match='SIGTERM'):
        dispatch.run_task(tmp_path,config,task,'0','cpu-test',10,interrupt)
    assert len(children)==1 and children[0].poll() is not None
    assert live_group_members(children[0].pid)==[]
    assert signal.getsignal(signal.SIGTERM) is previous


def test_cleanup_before_spawn_setsid_stops_only_direct_child():
    parent_group=os.getpgrp()
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
    try:
        assert os.getpgid(child.pid)==parent_group
        stop_process_group(child.pid,.3)
        child.wait(timeout=2)
        assert os.getpgrp()==parent_group and live_group_members(child.pid)==[]
    finally:
        if child.poll() is None:child.kill();child.wait()
