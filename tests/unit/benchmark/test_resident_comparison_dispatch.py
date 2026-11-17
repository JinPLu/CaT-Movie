"""常驻协议的真实CPU进程验收：复用PID、逐故事记录与硬超时回收。"""
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from mspc.comparison_methods import dispatch
from mspc.core.process_supervision import live_group_members
from tests.support.paths import REPO_ROOT


@pytest.fixture
def resident_cpu(tmp_path, monkeypatch):
    for relative in ('env/bin/ffmpeg', 'env/ffmpeg7/bin/ffprobe'):
        executable = tmp_path / relative
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text('#!/bin/sh\nexit 0\n')
        executable.chmod(0o755)
    method = 'official_multishotmaster_1_3b'
    config = dict(shared_root=str(tmp_path), cpu_threads=1, methods={method: dict(
        devices=1, code_root=str(tmp_path), python=sys.executable, code_revision='cpu-resident-test',
        render_module='mspc.benchmark.generation')})
    # 只替换模型消费者；生产父进程真实创建socket、监督与回收同一进程。
    code = '''
import json, os, socket, sys, time
from pathlib import Path
with socket.socket(fileno=int(sys.argv[1])) as channel:
    def send(row):
        channel.sendall((json.dumps(row)+'\\n').encode())
    send({'status':'ready', 'pid':os.getpid()})
    with channel.makefile('r') as commands:
        for ordinal, line in enumerate(commands):
            task=json.loads(line)['task']
            if task['story_id']=='hang':
                time.sleep(60)
            path=Path(task['generation_record_path'])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({'status':'completed', 'model_load_seconds': 3 if ordinal==0 else 0,
                                        'renderer_pid':os.getpid()}))
            send({'status':'completed', 'pid':os.getpid()})
'''
    original = subprocess.Popen
    children = []

    def cpu_consumer(argv, **kwargs):
        channel_fd = argv[argv.index('--channel-fd') + 1]
        process = original([sys.executable, '-c', code, channel_fd], **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(dispatch.subprocess, 'Popen', cpu_consumer)

    def task(name):
        return dict(method=method, story_id=name, first_story=False, input_tier='shots', seed=42,
                    execution_output_root=str(tmp_path),
                    generation_record_path=str(tmp_path / name / 'generation.json'))

    renderer = dispatch.ResidentRenderer(config, method, '0')
    yield renderer, config, task, children
    renderer.close()


def test_two_stories_share_one_process_and_keep_separate_records(tmp_path, resident_cpu):
    renderer, config, task, children = resident_cpu
    rows = [task(name) for name in ('first', 'second')]
    results = [dispatch.run_task(tmp_path, config, row, '0', 'cpu-lease', 5, renderer=renderer)
               for row in rows]
    generations = [json.loads(Path(row['generation_record_path']).read_text()) for row in rows]
    assert len(children) == 1 and children[0].poll() is None
    assert [result['status'] for result in results] == ['completed', 'completed']
    assert [result['pid'] for result in results] == [children[0].pid] * 2
    assert [row['model_load_seconds'] for row in generations] == [3, 0]
    assert [row['renderer_pid'] for row in generations] == [children[0].pid] * 2
    assert all(result['resident_renderer'] and result['process_gpu_hours'] > 0 for result in results)
    assert len(list((tmp_path / 'logs').glob('*-process.json'))) == 2
    renderer.close()
    assert children[0].poll() is not None and live_group_members(children[0].pid) == []


def test_timeout_of_later_story_kills_reused_process_and_preserves_first(tmp_path, resident_cpu):
    renderer, config, task, children = resident_cpu
    first = task('first')
    dispatch.run_task(tmp_path, config, first, '0', 'cpu-lease', 5, renderer=renderer)
    original = Path(first['generation_record_path']).read_bytes()
    hanging = task('hang')
    result = dispatch.run_task(tmp_path, config, hanging, '0', 'cpu-lease', .05, renderer=renderer)
    assert len(children) == 1
    assert result['status'] == 'failed' and result['failure_kind'] == 'TimeoutError'
    assert children[0].poll() is not None and live_group_members(children[0].pid) == []
    assert Path(first['generation_record_path']).read_bytes() == original
    failed = json.loads(Path(hanging['generation_record_path']).read_text())
    assert failed['status'] == 'generation_failed' and 'TimeoutError' in failed['process_error']


def test_exclusive_claim_does_not_start_another_consumer(tmp_path, resident_cpu):
    renderer, config, task, children = resident_cpu
    row = task('first')
    dispatch.run_task(tmp_path, config, row, '0', 'cpu-lease', 5, renderer=renderer)
    with pytest.raises(FileExistsError):
        dispatch.run_task(tmp_path, config, row, '0', 'cpu-lease', 5, renderer=renderer)
    assert len(children) == 1 and children[0].poll() is None


def test_resident_child_routes_keyframes_to_existing_owner(tmp_path, monkeypatch):
    for relative in ('env/bin/ffmpeg', 'env/ffmpeg7/bin/ffprobe'):
        executable = tmp_path / relative
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text('#!/bin/sh\nexit 0\n')
        executable.chmod(0o755)
    method = 'ic_lora_wan_a14b'
    config = dict(shared_root=str(tmp_path), cpu_threads=1, methods={method: dict(
        devices=1, code_root=str(REPO_ROOT), python=sys.executable, code_revision='cpu-keyframe-route',
        render_module='mspc.comparison_methods.keyframe_story')})
    script = textwrap.dedent('''
        import contextlib, json, sys, types
        from pathlib import Path
        from mspc.comparison_methods.dispatch import resident_renderer
        class Tensor:
            def __matmul__(self, other): return self
        torch=types.SimpleNamespace(
            cuda=types.SimpleNamespace(device_count=lambda:1, synchronize=lambda:None, get_device_name=lambda i:'CPU fake'),
            random=types.SimpleNamespace(fork_rng=lambda **kwargs:contextlib.nullcontext()),
            randn=lambda *args, **kwargs:Tensor(), bfloat16='bf16',
            isfinite=lambda value:types.SimpleNamespace(all=lambda:True), __version__='cpu-fake')
        sys.modules['torch']=torch
        def render(output, method, ids):
            path=output/ids[0]/'generation.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'status':'completed', 'owner':'keyframe_story', 'method':method}))
        sys.modules['mspc.comparison_methods.keyframe_story']=types.SimpleNamespace(render=render)
        resident_renderer(int(sys.argv[1]), sys.argv[2], sys.argv[3])
    ''')
    original = subprocess.Popen

    def fake_models(argv, **kwargs):
        return original([sys.executable, '-c', script, argv[argv.index('--channel-fd') + 1],
                         method, config['methods'][method]['render_module']], **kwargs)

    monkeypatch.setattr(dispatch.subprocess, 'Popen', fake_models)
    task = dict(method=method, story_id='keyframe-story', first_story=False,
                execution_output_root=str(tmp_path),
                generation_record_path=str(tmp_path / 'keyframe-story/generation.json'))
    result = dispatch.run_task(tmp_path, config, task, '0', 'cpu-lease', 5)
    assert result['status'] == 'completed'
    generation = json.loads(Path(task['generation_record_path']).read_text())
    assert generation['owner'] == 'keyframe_story' and generation['method'] == method
