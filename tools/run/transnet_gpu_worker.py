#!/usr/bin/env python3
"""数据切镜常驻 TF GPU worker；评测 worker 不变。"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resource-root', type=Path, required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--batch-windows', type=int, required=True)
    args = p.parse_args()
    from transnet_worker import load_backend_without_torch
    backend = load_backend_without_torch()
    import tensorflow as tf
    if not args.device.startswith('cuda:') or args.batch_windows < 1:
        raise ValueError('cuda:N and positive batch size required')
    load_started = time.perf_counter()
    engine = backend._TransNetEngine(args.resource_root, args.device)
    module_path = Path(__file__).resolve().parents[2] / 'src/mspc/data/transnet_gpu_scoring.py'
    spec = importlib.util.spec_from_file_location('transnet_gpu_scoring', module_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    identity = {'tensorflow': tf.__version__, 'weights': str(args.resource_root / 'transnetv2'),
                'tf_device': engine.tf_device, 'arithmetic': 'float32; TF32 disabled',
                'pid': os.getpid(), 'load_seconds': time.perf_counter()-load_started,
                'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']}
    tf.config.experimental.enable_tensor_float_32_execution(False)
    print(json.dumps(identity), flush=True)
    while line := sys.stdin.buffer.readline():
        count = int(line)
        data = sys.stdin.buffer.read(count*27*48*3)
        if len(data) != count*27*48*3:
            raise ValueError('truncated frame request')
        frames = np.frombuffer(data, np.uint8).reshape(count, 27, 48, 3)
        started = time.perf_counter()
        scores = []
        with tf.device(engine.tf_device):
            for batch in module.window_batches(frames, args.batch_windows):
                logits, _ = engine.model(tf.cast(batch, tf.float32))
                scores.append(tf.sigmoid(logits)[:, 25:75, 0].numpy().reshape(-1))
        single = np.concatenate(scores)[:count].astype(np.float32)
        header = {'score_seconds': time.perf_counter()-started}
        sys.stdout.buffer.write((json.dumps(header)+'\n').encode())
        sys.stdout.buffer.write(single.tobytes()); sys.stdout.buffer.flush()


if __name__ == '__main__':
    main()
