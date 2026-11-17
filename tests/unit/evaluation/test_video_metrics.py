"""自由时长、显式切镜要求与真实时间轴的计算边界。"""
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from mspc.evaluation import video_metrics as metrics
from mspc.evaluation import benchmark

ROOT = Path(__file__).resolve().parents[3]


def request(count=None, times=None):
    return {'expected_shot_count': count, 'expected_cut_times_seconds': times}


def test_free_duration_and_unspecified_cuts():
    scores = metrics.cut_metrics(request(), [2.3, 8.1], 17.75)
    assert all(r['status'] == 'not_applicable' for r in scores.values())
    scores = metrics.cut_metrics(request(count=8), [2.3, 8.1], 17.75)
    assert scores['transition_count_error']['value'] == 5
    assert scores['shot_cut_accuracy']['status'] == 'not_applicable'


def test_exact_no_cut_is_distinct_from_unspecified():
    scores = metrics.cut_metrics(request(times=[]), [], 2.5)
    assert [scores[k]['value'] for k in ('transition_count_error', 'shot_cut_accuracy',
                                       'matched_transition_timing_mae_seconds')] == [0, 1, 0]
    assert metrics.cut_metrics(request(times=[]), [1], 2.5)['shot_cut_accuracy']['value'] == pytest.approx(math.exp(-1))


def test_missing_cuts_and_duration_drift_are_scored_in_seconds():
    scores = metrics.cut_metrics(request(times=[1, 4]), [3.75], 3.9)
    assert scores['shot_cut_accuracy']['matched_pairs_seconds'] == [[4, 3.75]]
    assert scores['shot_cut_accuracy']['value'] == pytest.approx(math.exp(-.25/3.9 - 1/3))
    assert scores['transition_count_error']['value'] == 1
    assert scores['matched_transition_timing_mae_seconds']['value'] == .25
    scores = metrics.cut_metrics(request(times=[1, 4]), [], .5)
    assert scores['shot_cut_accuracy']['value'] == pytest.approx(math.exp(-2/3))
    assert scores['matched_transition_timing_mae_seconds']['status'] == 'not_applicable'


def test_conflicting_input_requirements_fail_before_scoring():
    with pytest.raises(ValueError, match='disagree'):
        metrics.validate_cut_request(request(count=4, times=[2]))


def test_actual_frame_timestamps_not_average_fps(monkeypatch):
    payload = {'frames': [{'best_effort_timestamp_time': str(t)} for t in [5, 5.04, 5.2, 5.24]]}
    monkeypatch.setattr(metrics.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=json.dumps(payload)))
    assert metrics.frame_times(Path('video.mp4'), 'ffprobe') == pytest.approx([0, .04, .2, .24])
    assert metrics.cut_boundaries_from_peaks((1, 3), frame_count=4) == (2,)


def test_worker_uses_whole_video_for_quality_and_detected_static_shots_for_flicker(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('vbench_worker', ROOT / 'tools/run/vbench_worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    import sys
    monkeypatch.setitem(sys.modules, 'vbench', SimpleNamespace(__file__=str(tmp_path / '__init__.py')))
    weights = {}
    for name in ('clip', 'aesthetic', 'musiq', 'amt', 'raft'):
        path = tmp_path / ('sa_0_4_vit_l_14_linear.pth' if name == 'aesthetic' else name)
        path.touch()
        weights[name] = str(path)
    calls = []
    def compute(name, videos, *args):
        calls.append((name, [p.name for p in videos]))
        values = [0., 1.] if name == 'dynamic_degree' else [.8] * len(videos)
        return [{'value': v} for v in values]
    monkeypatch.setattr(worker, 'compute', compute)
    monkeypatch.setattr(worker.subprocess, 'run', lambda *args, **kwargs: None)
    # A one-frame detector fragment is unavailable for temporal metrics, not a perfect score.
    shots = [{'start_frame': a, 'end_frame': b} for a, b in [(0, 37), (37, 62), (62, 63)]]
    result = worker.score({'video': '/actual/full.mp4', 'fps': 16, 'shots': shots,
        'config': {'weights': weights, 'device': 'cuda', 'ffmpeg': 'ffmpeg'}}, tmp_path)
    assert calls == [('aesthetic_quality', ['full.mp4']), ('imaging_quality', ['full.mp4']),
        ('dynamic_degree', ['shot-0.mp4', 'shot-1.mp4']), ('motion_smoothness', ['shot-0.mp4', 'shot-1.mp4']),
        ('temporal_flickering', ['shot-0.mp4'])]
    assert result['metrics']['dynamic_degree']['value'] == .5
    assert result['metrics']['temporal_flickering']['shots_scored'] == 1
    assert result['metrics']['motion_smoothness']['shots_total'] == 3


def test_metrics_and_judge_share_case_and_resume_independently(tmp_path, monkeypatch):
    from tests.unit.evaluation.test_benchmark import args_for, install_transport, response
    from tests.support.closeai_fakes import fake_chat_completion
    args = args_for(tmp_path)
    args.components = 'metrics'
    called = []
    def score_video(*args):
        called.append(1)
        return {'metrics': {name: metrics.measured(.8) for name in metrics.METRICS}, 'shots': []}
    monkeypatch.setattr(benchmark.VideoMetricScorer, 'score_video', score_video)
    install_transport(monkeypatch, lambda **kw: json.loads(json.dumps(
        fake_chat_completion(json.dumps(response()), prompt_tokens=2894, completion_tokens=1291,
            prompt_tokens_details={'video_tokens': 1188}), default=vars)))
    benchmark.score(args)
    args.components = 'all'
    benchmark.score(args)
    benchmark.score(args)
    assert called == [1]
    group, = json.loads((args.output / 'summary.json').read_text())['groups']
    assert len(group['dimensions']) == 14
    assert all(item['n'] == 1 for item in group['dimensions'].values())


def test_media_and_metric_failures_are_saved_and_raised(tmp_path, monkeypatch):
    from tests.unit.evaluation.test_benchmark import args_for
    args = args_for(tmp_path)
    args.components = 'metrics'
    def broken_video(*args):
        raise ValueError('bad delivered video')
    monkeypatch.setattr(benchmark, 'inspect_video', broken_video)
    with pytest.raises(ValueError, match='bad delivered video'):
        benchmark.score(args)
    sample, = benchmark.load_manifest(args.manifest)['requests']
    target = benchmark.result_path(args.output, sample)
    assert json.loads(target.read_text())['media_status'] == 'technical_failed'
    monkeypatch.setattr(benchmark, 'inspect_video', lambda *args: {'duration_seconds': 17.75})
    def broken_metric(*args):
        raise RuntimeError('worker failed')
    monkeypatch.setattr(benchmark.VideoMetricScorer, 'score_video', broken_metric)
    with pytest.raises(RuntimeError, match='worker failed'):
        benchmark.score(args)
    saved = json.loads(target.read_text())
    assert saved['non_vlm']['status'] == 'technical_failed' and 'judge' not in saved
    group, = json.loads((args.output / 'summary.json').read_text())['groups']
    assert group['dimensions']['aesthetic_quality']['mean'] is None
    assert group['dimensions']['aesthetic_quality']['status_counts'] == {'technical_failed': 1}


def test_metric_worker_reuses_one_process_and_propagates_first_error():
    import sys
    script = '''import json, os, sys
for line in sys.stdin:
    value = json.loads(line)
    if value.get('fail'):
        print('original worker failure', file=sys.stderr, flush=True)
        print(json.dumps({'status': 'error', 'error': 'bad real input'}), flush=True)
        raise SystemExit(2)
    print(json.dumps({'status': 'ok', 'result': {'pid': os.getpid(), 'value': value}}), flush=True)
'''
    worker = metrics._MetricWorker([sys.executable, '-u', '-c', script])
    try:
        first = worker.request({'story': 'first'})
        second = worker.request({'story': 'second'})
        assert first['pid'] == second['pid']
        assert second['value'] == {'story': 'second'}
        with pytest.raises(RuntimeError, match='bad real input.*original worker failure'):
            worker.request({'fail': True})
        assert worker._process.poll() == 2
    finally:
        worker.close()
        worker._stderr.close()


def test_metric_scorer_closes_both_isolated_workers_on_error(monkeypatch):
    stopped = []
    class Worker:
        def __init__(self, label):
            self.label = label
            self._stderr = SimpleNamespace(close=lambda: None)
        def close(self):
            stopped.append(self.label)
    with pytest.raises(ValueError, match='decode failed'):
        with metrics.VideoMetricScorer({}) as scorer:
            scorer._transnet = Worker('tensorflow')
            scorer._vbench = Worker('torch')
            raise ValueError('decode failed')
    assert stopped == ['tensorflow', 'torch']


def test_vbench_reuses_official_model_but_runs_each_video(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('vbench_worker', ROOT / 'tools/run/vbench_worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    loaded, scored = [], []
    class Dynamic:
        def __init__(self, args, device):
            loaded.append((args, device))
    def official(model, paths):
        scored.append((model, paths))
        return .5, [{'video_path': p, 'video_results': .5} for p in paths]
    module = SimpleNamespace(edict=dict, DynamicDegree=Dynamic, dynamic_degree=official)
    monkeypatch.setattr(worker.importlib, 'import_module', lambda name: module)
    for path in (Path('first.mp4'), Path('second.mp4')):
        assert worker.compute('dynamic_degree', [path], {'model': 'same-real-weights'}, tmp_path, 'cuda') == [{'value': .5}]
    assert len(loaded) == 1
    assert scored[0][0] is scored[1][0]
    assert [paths for _, paths in scored] == [['first.mp4'], ['second.mp4']]


def test_cut_f1_counts_late_missing_and_extra_cuts_at_two_frames():
    # 晚 3 帧算错配：要求 2 个、检出 2 个、只配上 1 个
    assert metrics.cut_metrics(request(times=[2, 5]), [2.0625, 5.1875], 8)['cut_f1']['value'] == pytest.approx(.5)
    # 晚 2 帧在容差内；镜内多出的切点拉低准确率
    scores = metrics.cut_metrics(request(times=[2, 5]), [2.125, 3.5, 5], 8)['cut_f1']
    assert scores['value'] == pytest.approx(2 * 2 / 5) and scores['matched'] == 2
    assert metrics.cut_metrics(request(times=[]), [], 2.5)['cut_f1']['value'] == 1.0
    assert metrics.cut_metrics(request(times=[1]), [], 2.5)['cut_f1']['value'] == 0.0
    assert metrics.cut_metrics(request(count=3), [1, 2], 3)['cut_f1']['status'] == 'not_applicable'
