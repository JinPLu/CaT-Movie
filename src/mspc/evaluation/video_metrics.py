"""成片辅助指标：实际切镜分段；只用用户明确提出的切镜要求。"""
from __future__ import annotations

import json
import os
import selectors
import time
import math
from pathlib import Path
import subprocess
import tempfile

from mspc.evaluation.renderer_qualification_public_metrics import (
    transition_run_peaks, cut_boundaries_from_peaks, _monotonic_min_abs_pairs,
)

METRICS = ('aesthetic_quality', 'imaging_quality', 'temporal_flickering',
           'motion_smoothness', 'dynamic_degree', 'shot_cut_accuracy',
           'transition_count_error', 'matched_transition_timing_mae_seconds', 'cut_f1')
# 切点 F1 的容差：16 fps 评测帧网格上的 2 帧。切镜执行列的 ±0.5 秒只管“切点附近换没换机位”，
# 这里管切点落在第几帧、漏切和镜内多出的切点。
CUT_F1_TOLERANCE_SECONDS = 2 / 16


def measured(value, **details):
    if not math.isfinite(value):
        raise ValueError(f'non-finite metric: {value}')
    return {'status': 'measured', 'value': value, **details}


def unavailable(reason):
    return {'status': 'not_applicable', 'value': None, 'reason': reason}


def validate_cut_request(row):
    """null表示没有要求，[]表示明确要求不切镜；不从文字或档位猜要求。"""
    count, times = row['expected_shot_count'], row['expected_cut_times_seconds']
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError('expected_shot_count must be null or a positive integer')
    if times is not None:
        if not isinstance(times, list) or any(
            type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in times
        ) or any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError('expected_cut_times_seconds must be null or increasing positive seconds')
        if count is not None and count != len(times) + 1:
            raise ValueError('explicit shot count and cut times disagree')


def tolerance_pairs(expected, detected, tolerance):
    """容差内一一配对的 (要求下标, 检出下标)；时间差小的先配。"""
    candidates = sorted((abs(a - b), i, j) for i, a in enumerate(expected) for j, b in enumerate(detected)
                        if abs(a - b) <= tolerance)
    pairs, used_expected, used_detected = [], set(), set()
    for _, i, j in candidates:
        if i not in used_expected and j not in used_detected:
            pairs.append((i, j))
            used_expected.add(i)
            used_detected.add(j)
    return pairs


def cut_f1(expected, detected):
    """2·匹配数 /(要求切点数 + 检出切点数)，即容差内准确率与召回率的调和平均；两边都无切点为 1。"""
    if not expected and not detected:
        return measured(1.0, matched=0, expected=0, detected=0)
    matched = len(tolerance_pairs(expected, detected, CUT_F1_TOLERANCE_SECONDS))
    return measured(2 * matched / (len(expected) + len(detected)),
                    matched=matched, expected=len(expected), detected=len(detected))


def cut_metrics(row, detected, duration):
    validate_cut_request(row)
    count, expected = row['expected_shot_count'], row['expected_cut_times_seconds']
    if expected is not None:
        count = len(expected) + 1
    output = {'transition_count_error': unavailable('no explicit shot count or cut times'),
              'shot_cut_accuracy': unavailable('no explicit cut times'),
              'matched_transition_timing_mae_seconds': unavailable('no explicit cut times'),
              'cut_f1': unavailable('no explicit cut times')}
    if count is not None:
        output['transition_count_error'] = measured(abs(len(detected) + 1 - count))
    if expected is None:
        return output
    # Expected times beyond the delivered duration are neither rejected nor clipped.
    if len(expected) <= len(detected):
        pairs = _monotonic_min_abs_pairs(expected, detected)
    else:
        pairs = tuple((b, a) for a, b in _monotonic_min_abs_pairs(detected, expected))
    error = sum(abs(a - b) for a, b in pairs)
    unmatched = abs(len(expected) - len(detected))
    output['shot_cut_accuracy'] = measured(math.exp(-error / duration - unmatched / count),
        matched_pairs_seconds=[list(pair) for pair in pairs], unmatched_count=unmatched)
    output['matched_transition_timing_mae_seconds'] = (
        measured(error / len(pairs)) if pairs else
        measured(0.0) if not expected and not detected else unavailable('no matched cuts'))
    output['cut_f1'] = cut_f1(expected, detected)
    return output


def frame_times(video, ffprobe):
    result = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
        '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(video)],
        check=True, capture_output=True, text=True)
    times = [float(row['best_effort_timestamp_time']) for row in json.loads(result.stdout)['frames']]
    if not times or any(not math.isfinite(t) for t in times) or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError('video frame timestamps must be finite and strictly increasing')
    return [t - times[0] for t in times]


class _MetricWorker:
    """顺序 JSONL 子进程；stdout 只用于回执，stderr 留作首错诊断。"""

    def __init__(self, argv, env=None):
        self._stderr = tempfile.TemporaryFile()
        self._process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._stderr, env=env, bufsize=0)
        self._reader = selectors.DefaultSelector()
        self._reader.register(self._process.stdout, selectors.EVENT_READ)
        self._buffer = b''

    def request(self, payload):
        process = self._process
        try:
            process.stdin.write((json.dumps(payload, allow_nan=False) + '\n').encode())
            process.stdin.flush()
            deadline = time.monotonic() + 1800
            while b'\n' not in self._buffer:
                if not self._reader.select(max(0, deadline - time.monotonic())):
                    raise TimeoutError('metric worker did not respond within 1800 seconds')
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError(f'metric worker closed stdout (exit {process.poll()})')
                self._buffer += chunk
            line, self._buffer = self._buffer.split(b'\n', 1)
            result = json.loads(line)
            if result['status'] != 'ok':
                raise RuntimeError(result['error'])
            return result['result']
        except Exception as error:
            self.close()
            self._stderr.seek(0)
            detail = self._stderr.read().decode(errors='replace')[-8000:]
            raise RuntimeError(f'{error}; worker stderr: {detail}') from error

    def close(self):
        process = self._process
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self._reader.close()
        if process.stdout:
            process.stdout.close()


class VideoMetricScorer:
    """一次评分 CLI 的模型生命周期；TF 与 Torch 始终留在独立解释器。"""

    def __init__(self, config):
        self.config = config
        self._transnet = self._vbench = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for worker in (self._transnet, self._vbench):
            if worker is not None:
                worker.close()
                worker._stderr.close()

    def _predict(self, frames):
        import numpy as np
        from mspc.evaluation.renderer_qualification_official_backend import (
            REPO_ROOT, TRANSNET_WORKER, resolve_transnet_python, seed_tensorflow_ld_library_path,
        )
        if self._transnet is None:
            python = resolve_transnet_python()
            env = seed_tensorflow_ld_library_path(os.environ, python)
            src = str(REPO_ROOT / 'src')
            env['PYTHONPATH'] = src + os.pathsep + env.get('PYTHONPATH', '')
            self._transnet = _MetricWorker([str(python), str(TRANSNET_WORKER), '--mode', 'serve',
                '--device', self.config['device'], '--resource-root', self.config['resource_root']], env)
        with tempfile.TemporaryDirectory(prefix='mspc-transnet-') as tmp:
            path = Path(tmp) / 'frames.npy'
            np.save(path, np.stack(frames), allow_pickle=False)
            values = self._transnet.request({'frames_npy': str(path)})
        if len(values) != len(frames) or any(not math.isfinite(v) for v in values):
            raise RuntimeError('TransNet returned invalid frame probabilities')
        return values

    def _vbench_request(self, data):
        from mspc.evaluation.renderer_qualification_official_backend import resolve_vbench_python
        if self._vbench is None:
            worker = Path(__file__).resolve().parents[3] / 'tools/run/vbench_worker.py'
            self._vbench = _MetricWorker([str(resolve_vbench_python()), str(worker), '--serve'])
        return self._vbench.request(data)

    def score_video(self, row, media, ffprobe):
        from mspc.evaluation.renderer_qualification_official_backend import _decode_rgb
        config = self.config
        video = Path(row['video_path'])
        frames, fps = _decode_rgb(video)
        times = frame_times(video, ffprobe)
        if len(times) != len(frames):
            raise ValueError('ffprobe and decoder disagree on delivered frame count')
        probabilities = self._predict(frames)
        peaks = transition_run_peaks(probabilities, threshold=config['cut_threshold'])
        boundaries = list(cut_boundaries_from_peaks(peaks, frame_count=len(frames)))
        cuts = [times[i] for i in boundaries]
        del frames
        positions = [0, *boundaries, len(times)]
        shots = [{'start_frame': a, 'end_frame': b, 'start_seconds': times[a],
                  'end_seconds': times[b] if b < len(times) else media['duration_seconds']}
                 for a, b in zip(positions, positions[1:])]
        quality = self._vbench_request({'video': str(video), 'fps': fps, 'shots': shots, 'config': config})
        metrics = {**quality['metrics'], **cut_metrics(row, cuts, media['duration_seconds'])}
        return {'metrics': metrics, 'detected_cut_times_seconds': cuts,
                'shots': quality['shots'], 'frame_count': len(times), 'fps': fps}
