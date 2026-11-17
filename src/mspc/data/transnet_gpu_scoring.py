"""数据线 TransNet：CPU 顺序解码、TF-only 常驻 GPU、官方空间缩放。"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
import resource

import numpy as np

PREPROCESS_VERSION = "transnet_ffmpeg48_rgb24_sequential_v1"


def decode_source(source, config):
    """content 保留原路径；TransNet 直接缩放，禁止默认 fps 同步增删帧。"""
    import cv2
    from mspc.data.frames import iter_scene_frames
    from mspc.data.scene_shot_manifest import content_diff_scores, probe_video
    from mspc.data.shot_boundary_revision import DECODE_WIDTH, DECODE_HEIGHT, DETECT_WIDTH, DETECT_HEIGHT

    if source['coordinate'] != 'ffmpeg_sequential':
        raise ValueError('GPU source scoring requires ffmpeg_sequential coordinates')
    cv2.setNumThreads(1)
    started = time.perf_counter()
    cpu_started = time.process_time()
    child_started = resource.getrusage(resource.RUSAGE_CHILDREN)
    video = source['video']
    meta = probe_video(config['ffprobe'], video)
    path = Path(source['score_file'])
    path.parent.mkdir(parents=True, exist_ok=True)
    content_frames = np.stack([
        cv2.resize(cv2.resize(frame, (DECODE_WIDTH, DECODE_HEIGHT)),
                   (DETECT_WIDTH, DETECT_HEIGHT), interpolation=cv2.INTER_AREA)
        for _, frame in iter_scene_frames({**meta, 'source_video_path': video}, None,
            ffmpeg=config['ffmpeg'], error_path=path.with_suffix('.content.stderr'), video_decode_num_threads=1)
    ])
    native_decoded = time.perf_counter()
    content = np.zeros(len(content_frames), np.float32)
    for start in range(1, len(content), 256):
        end = min(len(content), start + 256)
        content[start:end] = content_diff_scores(np.ascontiguousarray(content_frames[start-1:end, ..., ::-1]))
    content_done = time.perf_counter()
    del content_frames
    command = [config['ffmpeg'], '-v', 'error', '-nostdin', '-threads', '1', '-filter_threads', '1',
               '-i', video, '-map', '0:v:0', '-vf', 'setpts=N', '-fps_mode', 'passthrough',
               '-enc_time_base', 'demux', '-s', '48x27', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-threads', '1', 'pipe:1']
    result = subprocess.run(command, capture_output=True, check=True)
    path.with_suffix('.transnet.stderr').write_bytes(result.stderr)
    if result.stderr:
        raise RuntimeError(result.stderr.decode(errors='replace')[-2000:])
    frames = np.frombuffer(result.stdout, np.uint8).reshape(-1, 27, 48, 3)
    if len(frames) != len(content) or len(frames) < 2:
        raise ValueError(f'sequential frame mismatch: TransNet {len(frames)}, content {len(content)}')
    finished = time.perf_counter()
    child_finished = resource.getrusage(resource.RUSAGE_CHILDREN)
    return frames, content, {
        'video': video, 'coordinate': source['coordinate'], 'frame_count': len(frames), 'fps': meta['fps'],
        'decode_seconds': finished-started, 'content_decode_seconds': content_done-started,
        'native_decode_resize_seconds': native_decoded-started, 'content_seconds': content_done-native_decoded,
        'transnet_decode_seconds': finished-content_done,
        'decode_cpu_seconds': time.process_time()-cpu_started + child_finished.ru_utime + child_finished.ru_stime
                              - child_started.ru_utime - child_started.ru_stime,
        'preprocess_version': PREPROCESS_VERSION,
        'grid': 'TransNet: ffmpeg direct 48x27 rgb24; content: 256x144 -> INTER_AREA 96x54',
        'frame_alignment': 'setpts=N; fps_mode=passthrough; enc_time_base=demux; no duplication/drop',
        'single_index': 'source frame', 'content_index': 'incoming source frame; frame0=0',
    }


def window_batches(frames, batch_windows):
    """官方 25 帧首填充、末填充、100/50 滑窗及取中间 50。"""
    if frames.dtype != np.uint8 or frames.shape[1:] != (27, 48, 3) or not len(frames):
        raise ValueError('expected nonempty uint8 Nx27x48x3 RGB')
    if batch_windows < 1:
        raise ValueError('batch_windows must be positive')
    end_pad = 25 + (50 - len(frames) % 50) % 50
    padded = np.concatenate((np.repeat(frames[:1], 25, axis=0), frames,
                             np.repeat(frames[-1:], end_pad, axis=0)))
    offsets = range(0, len(padded)-99, 50)
    for start in range(0, len(offsets), batch_windows):
        yield np.stack([padded[i:i+100] for i in offsets[start:start+batch_windows]])


class PersistentTransNet:
    """每分片一个 TF 进程；二进制管道传帧，无逐源模型加载或临时 NPY。"""
    def __init__(self, *, python, root, device, batch_windows, log):
        if not device.startswith('cuda:'):
            raise ValueError('data scoring requires an explicit cuda:N device')
        worker = Path(__file__).resolve().parents[3] / 'tools/run/transnet_gpu_worker.py'
        self.device, self.batch_windows = device, batch_windows
        self.log = Path(log).open('wb')
        self.process = subprocess.Popen([str(python), str(worker), '--resource-root', str(root),
            '--device', device, '--batch-windows', str(batch_windows)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
        ready = self.process.stdout.readline()
        if not ready:
            self.close()
            raise RuntimeError(f'TransNet startup failed; see {log}')
        self.identity = json.loads(ready)

    def predict(self, frames):
        p = self.process
        p.stdin.write(f'{len(frames)}\n'.encode())
        p.stdin.write(frames.tobytes()); p.stdin.flush()
        header = p.stdout.readline()
        if not header:
            raise RuntimeError('TransNet worker exited; see worker log')
        timing = json.loads(header)
        data = p.stdout.read(len(frames)*4)
        if len(data) != len(frames)*4:
            raise RuntimeError('truncated TransNet response')
        return np.frombuffer(data, np.float32), timing

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.terminate(); self.process.wait()
        self.process.stdout.close()
        self.log.close()


def save_scores(source, decoded, engine):
    frames, content, meta = decoded
    single, timing = engine.predict(frames)
    if not np.isfinite(single).all():
        raise ValueError('nonfinite TransNet output')
    meta.update(timing, device=engine.device, batch_windows=engine.batch_windows,
                model=engine.identity, total_seconds=meta['decode_seconds']+timing['score_seconds'])
    path = Path(source['score_file'])
    tmp = path.with_suffix('.partial')
    with tmp.open('wb') as handle:
        np.savez_compressed(handle, metadata=json.dumps(meta), single=single, content=content)
    tmp.replace(path)
    return meta
