"""冻结近邻比较的任务账本，通过现有生成入口执行有时限、共享预算的故事。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from mspc.config.shared_root import expand_shared_root_token
import fcntl
import select
import socket
import subprocess
import sys
import tempfile
import time

from mspc.core.process_supervision import stop_process_group, termination_handler

from mspc.benchmark.generation import load_config, load_requests, prepare
from mspc.evaluation.benchmark import load_manifest, sample_key, write_json
from mspc.comparison_methods.prompts import segment_inputs


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def publish_new_json(path: Path, data: dict) -> None:
    """完整写入后再独占发布；观察者只能看见完整JSON，领取互斥仍由文件系统保证。"""
    descriptor, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False)
        os.link(temporary, path)
    finally:
        temporary.unlink()


def stop_child(child: subprocess.Popen | None) -> None:
    """异常和超时共用进程组清理，等待退出后才登记失败。"""
    if child is not None:
        stop_process_group(child.pid)
        child.wait()


def validate_preflight(board: dict, settings: dict, row: dict) -> None:
    """旧预检只能用于同一完整分镜；原生分段、帧量化和提示词逐项比对。"""
    next_shot = 0
    for segment in row['segments']:
        if segment['start_shot'] != next_shot or segment['end_shot'] <= next_shot:
            raise ValueError('preflight segments must cover each shot exactly once in order')
        expected = segment_inputs(board, settings['family'], next_shot, segment['end_shot'])
        if {key: value for key, value in segment.items() if key != 'token_counts'} != expected:
            raise ValueError('preflight differs from current native inputs')
        if max(segment['token_counts']) > settings['limits']['max_tokens']:
            raise ValueError('preflight exceeds native token capacity')
        next_shot = segment['end_shot']
    if next_shot != len(board['shots']) or row['frames'] != sum(s['num_frames'] for s in row['segments']):
        raise ValueError('preflight has incomplete shots or inconsistent frame total')


def reuse_first_story(row: dict, settings: dict, story: dict, old_output: Path) -> None:
    """沿用已准备的首故事输出地址；输入或方法设置变化即拒绝复用。"""
    old_config = json.loads((old_output / 'inputs/config.json').read_text())
    old_catalog = load_requests(old_output / 'inputs/stories.json')
    old_manifest = load_manifest(old_output / 'inputs/requests.json')
    previous = [r for r in old_manifest['requests'] if sample_key(r) == sample_key(row)]
    if len(previous) != 1 or old_config['methods'][row['method']] != settings:
        raise ValueError('first-story reuse requires identical method settings and one request')
    if next(s for s in old_catalog['stories'] if s['story_id'] == row['story_id']) != story:
        raise ValueError('first-story reuse requires the identical frozen story')
    paths = {'video_path', 'generation_record_path'}
    if {k: v for k, v in row.items() if k not in paths} != {k: v for k, v in previous[0].items() if k not in paths}:
        raise ValueError('first-story reuse requires identical request fields')
    for key in paths:
        row[key] = previous[0][key]


def prepare_dispatch(config_path: Path) -> dict:
    dispatch = json.loads(os.path.expandvars(expand_shared_root_token(config_path.read_text(), str(config_path))))
    if '${' in json.dumps(dispatch):
        raise ValueError('dispatch has unresolved environment variables')
    catalog = load_requests(Path(dispatch['frozen_stories']))
    if catalog['evaluation_scope'] != 'renderer':
        raise ValueError('neighbor dispatch requires frozen given-plan renderer inputs')
    config = load_config(Path(dispatch['generation_config']))
    methods = dispatch['methods']
    config['methods'] = {name: config['methods'][name] for name in methods}
    stories = {s['story_id']: s for s in catalog['stories'] if s['split'] == 'dev'}
    if len(stories) != dispatch['expected_story_count'] or dispatch['first_story'] not in stories:
        raise ValueError('frozen development membership differs from the declared dispatch')
    preflights = {}
    for method, launch in methods.items():
        evidence = json.loads(Path(launch['preflight']).read_text())
        rows = evidence[launch['preflight_rows_key']]
        preflights[method] = {r['story_id']: r for r in rows}
        if len(rows) != len(stories) or set(preflights[method]) != set(stories):
            raise ValueError(f'{method}: preflight membership differs from frozen development stories')
        for name, story in stories.items():
            validate_preflight(story['storyboard'], config['methods'][method], preflights[method][name])
    output = Path(dispatch['output_root'])
    manifest = prepare(catalog=catalog, config=config, methods=list(methods), tiers=['shots'],
                       split='dev', seeds=[dispatch['seed']], output=output)
    tasks = []
    for row in manifest['requests']:
        method, story_id = row['method'], row['story_id']
        launch, settings = methods[method], config['methods'][method]
        first = story_id == dispatch['first_story']
        run_output = Path(launch['first_story_output_root']) if first else output
        if first and run_output != output:
            reuse_first_story(row, settings, stories[story_id], run_output)
        evidence = preflights[method][story_id]
        segments = evidence['segments']
        frames = sum(s['num_frames'] for s in segments)
        tasks.append({**{k: row[k] for k in ('method', 'story_id', 'input_tier', 'seed', 'video_path', 'generation_record_path')},
                      'first_story': first, 'execution_output_root': str(run_output),
                      'status': 'prepared', 'requested_frames': sum(s['frame_count'] for s in stories[story_id]['storyboard']['shots']),
                      'native_output_frames': frames, 'generation_calls': len(segments),
                      'max_tokens': max(n for s in segments for n in s['token_counts']),
                      't2v_frames': segments[0]['num_frames'] if settings['family'] == 'storymem' else None,
                      'm2v_frames': frames - segments[0]['num_frames'] if settings['family'] == 'storymem' else None,
                      'actual_frames': None, 'actual_gpu_hours': None, 'lease_id': None,
                      'full_decode': None, 'visual_review': None})
    write_json(output / 'inputs/requests.json', manifest)
    load_manifest(output / 'inputs/requests.json')
    write_json(output / 'inputs/dispatch.json', dispatch)
    ledger = {'prepared_at': now(), 'evaluation_scope': 'renderer', 'seed': dispatch['seed'],
              'stories': len(stories), 'methods': list(methods), 'expected_requests': len(tasks),
              'first_story_requests': sum(r['first_story'] for r in tasks), 'tasks': tasks,
              'full_dispatch_gpu_hours_estimate': None, 'paid_api_calls': 0,
              'comparison_boundary': '同一故事与计划下的公开方案比较；底座、参数量、采样与计算量不同，不是架构因果消融。'}
    ledger['method_details'] = {}
    for method, launch in methods.items():
        rows = [r for r in tasks if r['method'] == method]
        settings = config['methods'][method]
        ledger['method_details'][method] = {
            'backbone': launch['backbone'], 'parameter_notes': launch['parameter_notes'],
            'sampling': settings['sampling'], 'seed_stride': settings['seed_stride'],
            'code_revision': launch['code_revision'], 'devices': launch['devices'],
            'requested_frames': sum(r['requested_frames'] for r in rows),
            'native_output_frames': sum(r['native_output_frames'] for r in rows),
            'generation_calls': sum(r['generation_calls'] for r in rows),
            't2v_frames': sum(r['t2v_frames'] for r in rows) if settings['family'] == 'storymem' else None,
            'm2v_frames': sum(r['m2v_frames'] for r in rows) if settings['family'] == 'storymem' else None,
            'first_story_hard_limit_seconds': launch['hard_limit_seconds']}
    write_json(output / 'manifests/task-ledger.json', ledger)
    return ledger


def validate_storymem_timeout(output: Path, task: dict, job_id: str) -> dict:
    """原监督进程可能被 Slurm 硬杀，必须以对应主作业和释放租约核终态。"""
    if not job_id.isdecimal() or int(job_id) <= 0 or task['method'] != 'official_storymem_14b':
        raise ValueError('resume requires the explicit original StoryMem Slurm job ID')
    label = 'first-story' if task['first_story'] else task['story_id']
    process = json.loads((output / 'logs' / f"{task['method']}-{label}-process.json").read_text())
    lease_id = f'slurm-{job_id}'
    if (process.get('method') != task['method'] or process.get('story_id') != task['story_id']
            or process.get('execution_output_root') != task['execution_output_root']
            or process.get('lease_id') != lease_id or process.get('stage') != 'render'
            or process.get('status') == 'completed'):
        raise ValueError('source process does not match this interrupted story and lease')
    leases = json.loads((output / 'manifests/leases.json').read_text())['leases']
    matching = [row for row in leases if row['lease_id'] == lease_id]
    if (len(matching) != 1 or matching[0]['method'] != task['method']
            or matching[0]['gpu_count'] != 1 or not matching[0]['released_at']):
        raise ValueError('source lease must be released and included in the shared budget')
    result = subprocess.run(['sacct', '-j', job_id, '-n', '-P', '-o', 'JobIDRaw,State,End'],
                            check=True, capture_output=True, text=True)
    main_job = [line.split('|') for line in result.stdout.splitlines()
                if line.split('|')[0].strip() == job_id]
    if len(main_job) != 1 or len(main_job[0]) < 3 or main_job[0][1].strip() != 'TIMEOUT':
        raise ValueError('source Slurm main job is not confirmed TIMEOUT')
    return {'job_id': job_id, 'state': 'TIMEOUT', 'end': main_job[0][2].strip(),
            'lease_id': lease_id, 'released_at': matching[0]['released_at']}


class ResidentRenderer:
    """一张租约卡对应一个模型进程；父进程仍能硬杀超时的 CUDA 调用。"""

    def __init__(self, dispatch: dict, method: str, devices: str):
        self.dispatch, self.method, self.devices = dispatch, method, devices
        self.child = self.channel = None
        self.buffer = b''

    def close(self):
        try:
            stop_child(self.child)
        finally:
            self.child = None
            if self.channel is not None:
                self.channel.close()
                self.channel = None

    def _receive(self, remaining):
        while b'\n' not in self.buffer:
            seconds = remaining()
            if seconds <= 0:
                raise TimeoutError('story or cumulative leased-GPU limit reached')
            readable, _, _ = select.select([self.channel], [], [], min(1, seconds))
            if readable:
                chunk = self.channel.recv(65536)
                if not chunk:
                    raise RuntimeError(f'resident renderer exited with {self.child.poll()}')
                self.buffer += chunk
        line, self.buffer = self.buffer.split(b'\n', 1)
        response = json.loads(line)
        if response['status'] == 'failed':
            raise RuntimeError(response['error'])
        return response

    def render(self, task, log_path, process_path, record, remaining, resume, destination, output):
        if task['method'] != self.method:
            raise ValueError('resident renderer method differs from claimed task')
        launch = self.dispatch['methods'][self.method]
        if self.child is None:
            shared = self.dispatch['shared_root']
            code = Path(resume['code_root'] if resume is not None else launch['code_root'])
            media = [Path(shared) / 'env/bin', Path(shared) / 'env/ffmpeg7/bin']
            env = dict(os.environ, MSPC_SHARED_ROOT=shared, PYTHONPATH=str(code / 'src'),
                       CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=self.devices,
                       OMP_NUM_THREADS=str(self.dispatch['cpu_threads']), MKL_NUM_THREADS=str(self.dispatch['cpu_threads']),
                       OPENBLAS_NUM_THREADS=str(self.dispatch['cpu_threads']), PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1',
                       TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       PATH=':'.join([str(path) for path in media] + [os.environ['PATH']]), LD_LIBRARY_PATH='')
            self.channel, child_channel = socket.socketpair()
            try:
                with log_path.open('a') as stream:
                    self.child = subprocess.Popen(
                        [launch['python'], '-m', 'mspc.comparison_methods.dispatch', 'resident-renderer',
                         '--channel-fd', str(child_channel.fileno()), '--method', self.method,
                         '--render-module', 'mspc.benchmark.generation' if resume is not None else launch['render_module']],
                        cwd=code, env=env, stdout=stream, stderr=subprocess.STDOUT,
                        pass_fds=(child_channel.fileno(),), start_new_session=True)
            finally:
                child_channel.close()
            record.update(stage='cuda_gate', pid=self.child.pid)
            write_json(process_path, record)
            ready = self._receive(remaining)
            if ready['status'] != 'ready':
                raise RuntimeError('resident renderer did not pass CUDA gate')
            record['cuda_gate'] = ready
        record.update(stage='render', pid=self.child.pid, resident_renderer=True)
        write_json(process_path, record)
        payload = dict(task=task, log_path=str(log_path), run_root=str(output), resume=resume,
                       destination=str(destination) if destination is not None else None)
        self.channel.sendall((json.dumps(payload) + '\n').encode())
        response = self._receive(remaining)
        if response['status'] != 'completed':
            raise RuntimeError('resident renderer returned an unexpected response')


def resident_renderer(channel_fd: int, method: str, render_module: str):
    """由生产监督器启动；控制通道独立于上游模型的 stdout/stderr。"""
    if render_module not in ('mspc.benchmark.generation', 'mspc.comparison_methods.keyframe_story'):
        raise ValueError('unknown configured comparison renderer module')
    with socket.socket(fileno=channel_fd) as channel:
        def respond(value):
            channel.sendall((json.dumps(value) + '\n').encode())

        try:
            import torch
            if torch.cuda.device_count() != 1:
                raise ValueError('resident renderer requires exactly one visible CUDA device')
            # gate过去在独立进程；在常驻进程中隔离其抽样，不消耗生成RNG。
            with torch.random.fork_rng(devices=[0]):
                x = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
                y = x @ x
                torch.cuda.synchronize()
                if not torch.isfinite(y).all():
                    raise ValueError('CUDA gate produced nonfinite values')
            del x, y
            ready = {'status': 'ready', 'pid': os.getpid(), 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}
            print(json.dumps({'cuda_gate': 'passed', **ready}), flush=True)
            respond(ready)
        except Exception as error:
            respond({'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
            raise
        session = None
        session_output = None
        with channel.makefile('r') as commands:
            for line in commands:
                command = json.loads(line)
                task = command['task']
                try:
                    # 每故事沿用独立日志；模型及控制连接在切换日志时保持常驻。
                    sys.stdout.flush(); sys.stderr.flush()
                    with open(command['log_path'], 'a') as log:
                        os.dup2(log.fileno(), 1)
                        os.dup2(log.fileno(), 2)
                    if task['method'] != method:
                        raise ValueError('claimed task differs from resident method')
                    output = Path(task['execution_output_root'])
                    if render_module == 'mspc.comparison_methods.keyframe_story':
                        if command['resume'] is not None:
                            raise ValueError('keyframe pipelines do not support StoryMem resume')
                        from mspc.comparison_methods.keyframe_story import render
                        render(output, method, [task['story_id']])
                        respond({'status': 'completed', 'pid': os.getpid()})
                        continue
                    from mspc.comparison_methods.renderer import RendererSession, resume_storymem
                    manifest = load_manifest(output / 'inputs/requests.json')
                    frozen, = [row for row in manifest['requests'] if
                               (row['method'], row['story_id'], row['input_tier'], row['seed']) ==
                               (method, task['story_id'], task['input_tier'], task['seed'])]
                    if any(task[key] != frozen[key] for key in
                           ('method', 'story_id', 'input_tier', 'seed', 'video_path', 'generation_record_path')):
                        raise ValueError('queue task differs from frozen generation request')
                    if command['resume'] is not None:
                        resume_storymem(output, method, task['story_id'], Path(command['destination']),
                                        Path(command['run_root']), command['resume']['source_slurm_job'])
                    else:
                        if session is None:
                            config = json.loads((output / 'inputs/config.json').read_text())
                            stories = {story['story_id']: story for story in
                                       json.loads((output / 'inputs/stories.json').read_text())['stories']}
                            session = RendererSession(output, config, method, stories)
                            session_output = output.resolve()
                        if output.resolve() != session_output:
                            raise ValueError('resident queue must share one frozen execution output root')
                        session.render_task(frozen)
                    respond({'status': 'completed', 'pid': os.getpid()})
                except Exception as error:
                    respond({'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
                    raise


@termination_handler()
def run_task(output: Path, dispatch: dict, task: dict, devices: str, lease_id: str,
             hard_limit_seconds: float, budget_remaining=None, *, resume: dict | None = None,
             renderer: ResidentRenderer | None = None) -> dict:
    method = task['method']
    launch = dispatch['methods'][method]
    if len(devices.split(',')) != launch['devices'] or launch['devices'] != 1 or not devices:
        raise ValueError('neighbor execution requires exactly one leased device selector')
    generation_path = Path(task['generation_record_path'])
    if resume is None and generation_path.exists():
        raise FileExistsError('generation record already exists; no automatic retry')
    shared = dispatch['shared_root']
    media_tools = [Path(shared) / 'env/bin/ffmpeg', Path(shared) / 'env/ffmpeg7/bin/ffprobe']
    for executable in media_tools:
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(f'required media executable is unavailable: {executable}')
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    label = 'first-story' if task['first_story'] else task['story_id']
    process_path = logs / f'{method}-{label}-process.json'
    log_path = logs / f'{method}-{label}.log'
    source_timeout = None
    code = Path(launch['code_root'])
    code_revision = launch['code_revision']
    if resume is not None:
        source_timeout = validate_storymem_timeout(output, task, resume['source_slurm_job'])
        from mspc.comparison_methods.renderer import load_storymem_resume_source
        source = load_storymem_resume_source(Path(task['execution_output_root']), method, task['story_id'], str(media_tools[1]))
        if any(source['task'].get(key) != task[key] for key in
               ('method', 'story_id', 'input_tier', 'seed', 'video_path', 'generation_record_path')):
            raise ValueError('queue task differs from the frozen source request')
        code = Path(resume['code_root']).resolve()
        code_revision = resume['code_revision']
        actual_revision = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
        if actual_revision != code_revision or Path(__file__).resolve() != code / 'src/mspc/comparison_methods/dispatch.py':
            raise ValueError('resume supervisor and explicit code revision must match the checked-out code')
        subprocess.run(['git', '-C', str(code), 'diff', '--quiet', 'HEAD', '--', 'src/mspc'], check=True)
        destination = output / 'resumed' / method / task['story_id']
        source_directory = Path(task['video_path']).parent.resolve()
        resolved = destination.resolve()
        if (resolved == source_directory or source_directory in resolved.parents or resolved in source_directory.parents):
            raise ValueError('resume output must be separate from the source story')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir(exist_ok=False)
        process_path, log_path = destination / 'process.json', destination / 'process.log'
        generation_path = destination / 'generation.json'
    # 独占创建防止两次分派同时起同一项；失败记录也保留，不自动覆盖重试。
    publish_new_json(process_path, {'status': 'starting', 'submitted_at': now(), 'lease_id': lease_id,
                                    'source_timeout': source_timeout})
    record = {'status': 'running', 'submitted_at': now(), 'lease_id': lease_id, 'devices': devices,
              'method': method, 'story_id': task['story_id'], 'code_revision': code_revision,
              'execution_output_root': task['execution_output_root'], 'hard_limit_seconds': hard_limit_seconds}
    if resume is not None:
        record.update(source_timeout=source_timeout, resume_directory=str(destination))
    started = time.monotonic()
    owned_renderer = renderer is None
    renderer = renderer if renderer is not None else ResidentRenderer(dispatch, method, devices)

    def remaining():
        seconds = hard_limit_seconds - (time.monotonic() - started)
        return min(seconds, budget_remaining()) if budget_remaining is not None else seconds

    try:
        with log_path.open('x'):
            pass
        renderer.render(task, log_path, process_path, record, remaining, resume,
                        destination if resume is not None else None, output)
        generation = json.loads(generation_path.read_text())
        if generation['status'] != 'completed':
            raise RuntimeError(f"renderer status is {generation['status']}")
        record['status'] = 'completed'
    except BaseException as error:
        renderer.close()
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        generation = json.loads(generation_path.read_text()) if generation_path.exists() else {}
        generation.update(status='generation_failed', process_error=record['error'])
        actual_error = generation.get('error', record['error'])
        record['failure_kind'] = 'CUDA out of memory' if 'out of memory' in actual_error else actual_error.split(':', 1)[0]
        write_json(generation_path, generation)
        if not isinstance(error, Exception):
            raise
    finally:
        if owned_renderer:
            renderer.close()
        record.update(finished_at=now(), wall_seconds=time.monotonic() - started)
        record['process_gpu_hours'] = record['wall_seconds'] * launch['devices'] / 3600
        record['lease_released_at'] = None
        write_json(process_path, record)
    return record


def run_first(output: Path, method: str, devices: str, lease_id: str) -> dict:
    dispatch = json.loads((output / 'inputs/dispatch.json').read_text())
    ledger = json.loads((output / 'manifests/task-ledger.json').read_text())
    task, = [r for r in ledger['tasks'] if r['method'] == method and r['first_story']]
    return run_task(output, dispatch, task, devices, lease_id, dispatch['methods'][method]['hard_limit_seconds'])


def development_budget(logs: Path, method: str, gpu_hours: float, current: datetime) -> float:
    """按租约起点累计占用；已退出但尚未释放的worker继续计费。"""
    workers = [json.loads(p.read_text()) for p in logs.glob(f'{method}-development-worker-*.json')]
    used = 0.0
    for worker in workers:
        end = datetime.fromisoformat(worker['lease_released_at']) if worker.get('lease_released_at') else current
        used += (end - datetime.fromisoformat(worker['lease_held_since'])).total_seconds()
    active = sum(w['status'] == 'running' for w in workers)
    # 每个活跃worker只使用剩余总卡秒的均分；轮询中持续扣减，避免各自享用完整预算。
    return (gpu_hours * 3600 - used - 30 * active) / max(1, active)


def repeated_failure(logs: Path, method: str) -> bool:
    results = [json.loads(p.read_text()) for p in logs.glob(f'{method}-lsh-*-process.json')]
    finished = sorted((r for r in results if 'finished_at' in r), key=lambda r: r['finished_at'])
    if len(finished) < 2 or any(r['status'] != 'failed' for r in finished[-2:]):
        return False
    return finished[-1]['failure_kind'] == finished[-2]['failure_kind']


def shared_budget_bounds(settings: dict) -> dict:
    """并发数属于当次调度；其余预算与停止条件须跨工作者保持一致。"""
    return {key: ({name: value for name, value in item.items() if name != 'max_workers'}
                  if key == 'long_segment_limits' else item)
            for key, item in settings.items() if key != 'max_workers'}


def check_long_segment_costs(tasks: list[dict], limits: dict, pause: Path) -> None:
    """已完成长段超出显式成本余量时，暂停领取新故事；正在生成的故事继续受原预算监督。"""
    if pause.exists():
        return
    for task in tasks:
        path = Path(task['generation_record_path']).parent / 'segments.json'
        if not path.exists():
            continue
        for segment in json.loads(path.read_text()):
            frames = segment['num_frames']
            if limits['method'] == 'official_storymem_14b':
                memory_count = len(segment['memory_files'])
                latent_length = (frames - 1) // 4 + 1 + memory_count
                first_shot = segment['start_shot'] == 0
                if first_shot:
                    if frames <= limits['measured_t2v_max_frames']:
                        continue
                elif latent_length <= limits['measured_m2v_max_latent_length']:
                    continue
                predicted = (limits['constant'] + limits['linear'] * latent_length
                             + limits['quadratic'] * latent_length ** 2)
                if first_shot:
                    predicted *= limits['t2v_scale']
                details = dict(sampling_branch='T2V' if first_shot else 'M2V',
                               shot_index=segment['start_shot'], memory_count=memory_count,
                               latent_length=latent_length)
            elif limits['method'] == 'official_holocine_14b_full':
                if frames <= limits['measured_max_frames']:
                    continue
                predicted = limits['linear'] * frames + limits['quadratic'] * frames ** 2
                details = dict(sampling_branch='joint_generation')
            else:
                raise ValueError('unknown long-segment cost method')
            if segment['seconds'] <= limits['max_cost_factor'] * predicted:
                continue
            evidence = dict(created_at=now(), reason='completed long segment exceeds cost estimate',
                            story_id=task['story_id'], num_frames=frames, actual_seconds=segment['seconds'],
                            predicted_seconds=predicted, max_cost_factor=limits['max_cost_factor'],
                            segments_path=str(path), **details)
            try:
                publish_new_json(pause, evidence)
            except FileExistsError:
                pass
            return


def register_worker(logs, method, path, state):
    """同一跨进程锁内完成共享预算校验、并发计数与登记。"""
    with (logs / f'{method}-development.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = [json.loads(p.read_text()) for p in logs.glob(f'{method}-development-worker-*.json')]
        if any(shared_budget_bounds(w['bounds']) != shared_budget_bounds(state['bounds']) for w in previous):
            raise ValueError('workers must share the same explicit development budget')
        if sum(w['status'] == 'running' for w in previous) >= state['bounds']['max_workers']:
            raise ValueError('development worker parallelism limit reached')
        publish_new_json(path, state)


@termination_handler()
def run_worker(output: Path, method: str, devices: str, lease_id: str, lease_held_since: str,
               worker: str, gpu_hours: float, story_limit_seconds: float, max_workers: int,
               long_segment_limits: Path | None = None) -> dict:
    if method not in ('official_multishotmaster_1_3b', 'official_holocine_14b_full', 'official_storymem_14b'):
        raise ValueError('unknown neighbor development method')
    if method in ('official_holocine_14b_full', 'official_storymem_14b') and long_segment_limits is None:
        raise ValueError('HoloCine and StoryMem expansion require explicit long-segment cost limits')
    if not worker.isalnum() or gpu_hours <= 0 or story_limit_seconds <= 0 or not 1 <= max_workers <= 6:
        raise ValueError('invalid explicit development execution bounds')
    datetime.fromisoformat(lease_held_since)
    dispatch = json.loads((output / 'inputs/dispatch.json').read_text())
    ledger = json.loads((output / 'manifests/task-ledger.json').read_text())
    tasks = [t for t in ledger['tasks'] if t['method'] == method and not t['first_story']]
    logs = output / 'logs'
    settings = dict(gpu_hours=gpu_hours, story_limit_seconds=story_limit_seconds, max_workers=max_workers)
    segment_limits = None
    if long_segment_limits is not None:
        segment_limits = json.loads(long_segment_limits.read_text())
        if (segment_limits['method'] != method
                or segment_limits['linear'] < 0 or segment_limits['quadratic'] < 0
                or segment_limits['max_cost_factor'] <= 1
                or segment_limits['story_limit_seconds'] != story_limit_seconds
                or segment_limits['max_workers'] != max_workers):
            raise ValueError('invalid long-segment cost limits')
        if method == 'official_storymem_14b':
            if (segment_limits['measured_t2v_max_frames'] <= 0
                    or segment_limits['measured_m2v_max_latent_length'] <= 0
                    or segment_limits['constant'] < 0 or segment_limits['t2v_scale'] <= 0):
                raise ValueError('invalid StoryMem T2V/M2V cost limits')
        elif segment_limits['measured_max_frames'] <= 0:
            raise ValueError('invalid measured frame limit')
        prior = segment_limits['prior_lease_gpu_hours']
        if prior < 0 or abs(gpu_hours + prior - segment_limits['all_inclusive_gpu_hours']) > 1e-8:
            raise ValueError('development budget must subtract already released first-story lease cost')
        settings['long_segment_limits'] = segment_limits
    path = logs / f'{method}-development-worker-{worker}.json'
    state = dict(status='running', worker=worker, method=method, devices=devices, lease_id=lease_id,
                 lease_held_since=lease_held_since, lease_released_at=None, submitted_at=now(),
                 bounds=settings, completed=0, failed=0)
    register_worker(logs, method, path, state)
    pause = logs / f'{method}-development-pause.json'

    def remaining():
        if segment_limits is not None:
            check_long_segment_costs(tasks, segment_limits, pause)
        return development_budget(logs, method, gpu_hours, datetime.now(timezone.utc))

    renderer = ResidentRenderer(dispatch, method, devices)
    try:
        for task in tasks:
            remaining_seconds = remaining()
            if not pause.exists() and repeated_failure(logs, method):
                try:
                    publish_new_json(pause, dict(created_at=now(), reason='two consecutive engineering failures of the same kind'))
                except FileExistsError:
                    pass
            if pause.exists():
                state.update(status='paused', reason=json.loads(pause.read_text())['reason'])
                break
            if remaining_seconds <= 0:
                state.update(status='budget_exhausted')
                break
            if Path(task['generation_record_path']).exists():
                continue
            try:
                result = run_task(output, dispatch, task, devices, lease_id, story_limit_seconds, remaining,
                                  renderer=renderer)
            except FileExistsError:
                continue  # 另一worker独占领取的同一项，不重试、不重抽。
            state['completed' if result['status'] == 'completed' else 'failed'] += 1
            state['last_story'] = task['story_id']
            write_json(path, state)
        else:
            state['status'] = 'finished'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        renderer.close()
        state['finished_at'] = now()
        write_json(path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--config', type=Path, required=True)
    first = commands.add_parser('run-first')
    first.add_argument('--output', type=Path, required=True)
    first.add_argument('--method', required=True)
    first.add_argument('--devices', required=True)
    first.add_argument('--lease-id', required=True)
    worker = commands.add_parser('run-worker')
    worker.add_argument('--output', type=Path, required=True)
    worker.add_argument('--method', required=True)
    worker.add_argument('--devices', required=True)
    worker.add_argument('--lease-id', required=True)
    worker.add_argument('--lease-held-since', required=True)
    worker.add_argument('--worker', required=True)
    worker.add_argument('--gpu-hours', type=float, required=True)
    worker.add_argument('--story-limit-seconds', type=float, required=True)
    worker.add_argument('--max-workers', type=int, required=True)
    worker.add_argument('--long-segment-limits', type=Path)
    resident = commands.add_parser('resident-renderer')
    resident.add_argument('--channel-fd', type=int, required=True)
    resident.add_argument('--method', required=True)
    resident.add_argument('--render-module', required=True)
    args = parser.parse_args()
    if args.command == 'resident-renderer':
        resident_renderer(args.channel_fd, args.method, args.render_module)
    elif args.command == 'prepare':
        result = prepare_dispatch(args.config)
        print(json.dumps({key: value for key, value in result.items() if key != 'tasks'}, ensure_ascii=False))
    elif args.command == 'run-first':
        result = run_first(args.output, args.method, args.devices, args.lease_id)
        print(json.dumps(result, ensure_ascii=False))
        if result['status'] != 'completed':
            raise SystemExit(1)
    else:
        result = run_worker(args.output, args.method, args.devices, args.lease_id, args.lease_held_since,
                            args.worker, args.gpu_hours, args.story_limit_seconds, args.max_workers,
                            args.long_segment_limits)
        print(json.dumps(result, ensure_ascii=False))
        if result['status'] not in ('finished', 'paused', 'budget_exhausted'):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
