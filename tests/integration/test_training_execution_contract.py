"""真实两进程CPU更新：累积、学习率、全rank日志、保存与relay共用的数值合同。

没有Accelerate的基础CPU环境跳过；训练环境或 `uv run --locked --with accelerate==1.14.0 pytest ...` 执行。
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def worker(output: Path, *, short_dataset: bool) -> None:
    from types import SimpleNamespace
    from datetime import timedelta
    import torch
    from accelerate import Accelerator, DataLoaderConfiguration
    from accelerate.utils import InitProcessGroupKwargs
    from mspc.training import loop

    torch.set_num_threads(1)
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2,
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        kwargs_handlers=[InitProcessGroupKwargs(backend='gloo', timeout=timedelta(seconds=30))])
    output.mkdir(parents=True, exist_ok=True)
    points = []
    loop.open_loss_tracking = lambda *a, **kw: SimpleNamespace(
        log=lambda step, metrics: points.append({'step':step, **metrics}),
        finish=lambda: (output / 'direct-tracker.json').write_text(json.dumps(points)))

    class Samples(torch.utils.data.Dataset):
        def __len__(self):
            return 8 if short_dataset else 12

        def __getitem__(self, index):
            return {'value':torch.tensor([(index + 1) / 10.0]),
                    '_sample_metadata':{'sample_index':index, 'case_id':str(index), 'span':[0,1], 'references':[]}}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.last_timing = {'timestep':900.0}

        def forward(self, data):
            return (self.weight * data['value']).square().mean()

        def optimizer_param_groups(self, args):
            return [{'params':list(self.parameters())}]

        def export_trainable_state_dict(self, state, remove_prefix):
            return state

        def trainable_param_names(self):
            return {'weight'}

    class Logger(loop.TrainableStateDictLogger):
        def __init__(self):
            super().__init__(str(output / 'checkpoints'), 'model.')
            self.weights = []

        def on_step_end(self, accelerator, model, save_steps):
            super().on_step_end(accelerator, model, save_steps)
            self.weights.append(float(accelerator.unwrap_model(model).weight.detach()))

    args = SimpleNamespace(learning_rate=.1, weight_decay=0., lr_scheduler_type='constant_with_warmup',
        warmup_steps=2, training_seed=17, per_device_batch_size=1, dataloader_drop_last=True,
        dataset_num_workers=0, dataloader_pin_memory=False, dataloader_persistent_workers=False,
        gradient_accumulation_steps=2, max_train_steps=3, num_epochs=1, timing_log_steps=1,
        timing_log_every=0, timing_sync_cuda=False, timing_prefix='[test]', save_steps=2,
        progress_stream=None, log_dir=str(output), loss_tracking=None, experiment_config={})
    logger = Logger()
    try:
        loop.train_sft(accelerator, Samples(), Model(), logger, args)
        if accelerator.is_main_process:
            (output / 'weights.json').write_text(json.dumps(logger.weights))
    finally:
        accelerator.end_training()


def launch(output: Path, *, short_dataset: bool):
    return subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
        str(Path(__file__)), str(output), 'short' if short_dataset else 'full'],
        cwd=ROOT, env={**os.environ, 'PYTHONPATH':str(ROOT / 'src'), 'OMP_NUM_THREADS':'1'},
        text=True, capture_output=True, timeout=90)


@pytest.mark.skipif(sys.platform != 'linux' or importlib.util.find_spec('accelerate') is None,
                   reason='requires Linux Gloo and training extra accelerate; macOS barrier device IDs select MPS')
def test_actual_updates_share_global_loss_used_lr_and_final_checkpoint(tmp_path):
    import math
    from safetensors.torch import load_file
    from mspc.core.loss_curves import train_curves

    completed = launch(tmp_path / 'complete', short_dataset=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    root = tmp_path / 'complete'
    samples = [json.loads(line) for path in root.glob('samples.rank*.jsonl') for line in path.read_text().splitlines()]
    metrics = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
    direct = json.loads((root / 'direct-tracker.json').read_text())
    relay = train_curves(samples, metrics, samples_per_update=4, window=2)
    assert sorted(row['sample_index'] for row in samples) == list(range(12))
    assert [row['samples_seen'] for row in metrics] == [4, 8, 12]
    assert [row['lr_used'] for row in metrics] == pytest.approx([0., .05, .1])
    assert [row['lr_next'] for row in metrics] == pytest.approx([.05, .1, .1])
    for metric, online in zip(metrics, direct, strict=True):
        step = metric['step']
        actual = math.fsum(row['loss'] for row in samples if row['step'] == step) / 4
        assert metric['loss'] == pytest.approx(actual, rel=1e-12)
        assert online['train/loss'] == relay[step]['train/loss'] == metric['loss']
        assert online['train/lr_used'] == relay[step]['train/lr_used'] == metric['lr_used']
    weights = json.loads((root / 'weights.json').read_text())
    assert weights[0] == 1. and weights[1] < weights[0] and weights[2] < weights[1]
    assert sorted(p.name for p in (root / 'checkpoints').iterdir()) == ['step-2.safetensors','step-3.safetensors']
    assert float(load_file(str(root / 'checkpoints/step-3.safetensors'))['weight']) == weights[-1]


@pytest.mark.skipif(sys.platform != 'linux' or importlib.util.find_spec('accelerate') is None,
                   reason='requires Linux Gloo and training extra accelerate; macOS barrier device IDs select MPS')
def test_short_actual_dataloader_fails_before_recording_a_completed_update(tmp_path):
    root = tmp_path / 'short'
    completed = launch(root, short_dataset=True)
    assert completed.returncode != 0
    assert 'provides 2 updates, below max_train_steps=3' in completed.stderr
    assert not (root / 'checkpoints').exists()
    assert not (root / 'metrics.jsonl').exists()


if __name__ == '__main__':
    worker(Path(sys.argv[1]), short_dataset=sys.argv[2] == 'short')
