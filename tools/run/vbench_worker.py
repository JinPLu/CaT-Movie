#!/usr/bin/env python3
"""隔离的官方VBench环境：整片画质，实际镜头内时序指标；直接读取已有权重。"""
from __future__ import annotations

import argparse
import contextlib
from functools import lru_cache
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


class MusiqScorer:
    """全帧与显式选帧共用的 MUSIQ owner；保留官方 FP32、longer 预处理。"""

    def __init__(self, model_path, device, batch_size):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('positive batch_size required')
        self.batch_size = batch_size
        self.module = importlib.import_module('vbench.imaging_quality')
        self.model = self.module.MUSIQ(pretrained_model_path=model_path)
        self.model.to(device)
        # 与官方 compute_imaging_quality 完全相同，不额外改变子模块模式。
        self.model.training = False
        self.device = device

    def score_frames(self, images, frame_indices):
        import torch
        indices = list(frame_indices)
        if not indices or indices != sorted(set(indices)) or indices[0] < 0:
            raise ValueError('frame indices must be non-empty, ordered, unique and nonnegative')
        if len(images) != len(indices):
            raise ValueError('frame count differs from frame indices')
        # 官方 load_video 先转 float32，再 Resize；uint8 Resize 会改变数值。
        images = self.module.transform(images.float(), 'longer')
        raw_scores = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = images[start:start + self.batch_size].to(self.device)
                values = self.model(batch).reshape(-1).tolist()
                if len(values) != len(batch) or any(not math.isfinite(v) for v in values):
                    raise ValueError('imaging_quality returned invalid frame scores')
                raw_scores.extend(values)
        return {'value': sum(raw_scores) / len(raw_scores) / 100,
                'frame_indices': indices, 'frame_scores': [v / 100 for v in raw_scores],
                'frame_index_basis': 'zero_based_input_video',
                'preprocessing': 'longer', 'precision': 'float32', 'score_scale': 'MUSIQ/100'}


@lru_cache(maxsize=5)
def _model(dimension, frozen_submodules, device):
    """只保留官方模型对象；前处理与逐帧算法继续调用 VBench 原函数。"""
    submodules = json.loads(frozen_submodules)
    if dimension == 'imaging_quality':
        scorer = MusiqScorer(submodules['model_path'], device, 1)
        return scorer.module, scorer
    module = importlib.import_module('vbench.' + dimension)
    if dimension == 'aesthetic_quality':
        aesthetic = module.get_aesthetic_model(submodules[1]).to(device)
        clip, _ = module.clip.load(submodules[0], device=device)
        return module, (aesthetic, clip)
    if dimension == 'dynamic_degree':
        args = module.edict({'model': submodules['model'], 'small': False,
                             'mixed_precision': False, 'alternate_corr': False})
        return module, module.DynamicDegree(args, device)
    if dimension == 'motion_smoothness':
        return module, module.MotionSmoothness(submodules['config'], submodules['ckpt'], device)
    if dimension == 'temporal_flickering':
        return module, None
    raise ValueError(f'unknown VBench dimension: {dimension}')


def compute(dimension, videos, submodules, work, device):
    module, model = _model(dimension, json.dumps(submodules, sort_keys=True), device)
    if dimension == 'imaging_quality':
        results = []
        for path in videos:
            images = model.module.load_video(str(path))
            results.append(model.score_frames(images, range(len(images))))
        return results
    paths = [str(p) for p in videos]
    if dimension == 'aesthetic_quality':
        _, rows = module.laion_aesthetic(*model, paths, device)
    elif dimension == 'dynamic_degree':
        _, rows = module.dynamic_degree(model, paths)
    elif dimension == 'motion_smoothness':
        _, rows = module.motion_smoothness(model, paths)
    else:
        _, rows = module.temporal_flickering(paths)
    by_path = {row['video_path']: row['video_results'] for row in rows}
    values = [float(by_path[str(path)]) for path in videos]
    if any(not math.isfinite(v) for v in values):
        raise ValueError(f'{dimension} returned non-finite values')
    return [{'value': v} for v in values]


def summarize(values, total):
    if not values:
        return {'status': 'not_applicable', 'value': None,
                'reason': 'no eligible detected shots', 'shots_scored': 0, 'shots_total': total}
    return {'status': 'measured', 'value': sum(values) / len(values),
            'shots_scored': len(values), 'shots_total': total}


def cut_clips(video, shots, work, ffmpeg):
    clips = []
    for i, shot in enumerate(shots):
        clip = work / f'shot-{i}.mp4'
        subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-i', str(video),
            '-vf', f"trim=start_frame={shot['start_frame']}:end_frame={shot['end_frame']},setpts=PTS-STARTPTS",
            '-an', '-fps_mode', 'passthrough', '-c:v', 'libx264', '-qp', '0', str(clip)], check=True)
        clips.append(clip)
    return clips


def score(data, work):
    import vbench
    cfg = data['config']
    weights = cfg['weights']
    for path in weights.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    # The official aesthetic loader expects this filename in an existing directory.
    aes = Path(weights['aesthetic'])
    if aes.name != 'sa_0_4_vit_l_14_linear.pth':
        raise ValueError('official aesthetic predictor requires its upstream filename')
    amt_config = Path(vbench.__file__).parent / 'third_party/amt/cfgs/AMT-S.yaml'
    submodules = {'aesthetic_quality': [weights['clip'], str(aes.parent)],
        'imaging_quality': {'model_path': weights['musiq']},
        'motion_smoothness': {'config': str(amt_config), 'ckpt': weights['amt']},
        'dynamic_degree': {'model': weights['raft']}, 'temporal_flickering': []}
    video = Path(data['video'])
    metrics = {}
    for name in ('aesthetic_quality', 'imaging_quality'):
        result, = compute(name, [video], submodules[name], work, cfg['device'])
        metrics[name] = {'status': 'measured', **result}
    shots = [dict(s) for s in data['shots']]
    clips = cut_clips(video, shots, work, cfg['ffmpeg'])
    for name in ('dynamic_degree', 'motion_smoothness', 'temporal_flickering'):
        # Upstream RAFT needs >=3 sampled frames (two rounds its count threshold to zero); AMT needs >=3 original frames.
        eligible = [i for i, s in enumerate(shots) if
            (len(range(0, s['end_frame'] - s['start_frame'], max(1, round(data['fps'] / 8)))) >= 3
             if name == 'dynamic_degree' else
             s['end_frame'] - s['start_frame'] >= 3 if name == 'motion_smoothness' else
             s.get('dynamic_degree') == 0)]
        results = compute(name, [clips[i] for i in eligible], submodules[name], work, cfg['device']) if eligible else []
        values = [r['value'] for r in results]
        for i, value in zip(eligible, values, strict=True):
            shots[i][name] = value
        metrics[name] = summarize(values, len(shots))
    return {'metrics': metrics, 'shots': shots}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--input', type=Path)
    mode.add_argument('--serve', action='store_true')
    args = parser.parse_args()
    if args.input:
        with contextlib.redirect_stdout(sys.stderr), tempfile.TemporaryDirectory(prefix='storyboard-vbench-') as tmp:
            result = score(json.loads(args.input.read_text()), Path(tmp))
        print(json.dumps(result, allow_nan=False))
        return
    config = None
    for line in sys.stdin:
        try:
            data = json.loads(line)
            if config is not None and data['config'] != config:
                raise ValueError('persistent VBench worker configuration changed')
            config = data['config']
            with contextlib.redirect_stdout(sys.stderr), tempfile.TemporaryDirectory(prefix='storyboard-vbench-') as tmp:
                result = score(data, Path(tmp))
            print(json.dumps({'status': 'ok', 'result': result}, allow_nan=False), flush=True)
        except Exception as error:
            print(json.dumps({'status': 'error', 'error': f'{type(error).__name__}: {error}'}), flush=True)
            raise


if __name__ == '__main__':
    main()
