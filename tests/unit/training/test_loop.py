from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from pathlib import Path
import sys
import types

import pytest
import torch

from tests.support.paths import REPO_ROOT


def _load_loop_module():
    module_path = REPO_ROOT / "src/mspc/training/loop.py"
    fake_accelerate = types.ModuleType("accelerate")
    fake_accelerate.Accelerator = object
    fake_data_loader = types.ModuleType("accelerate.data_loader")
    fake_accelerate.data_loader = fake_data_loader
    previous = sys.modules.get("accelerate")
    previous_data_loader = sys.modules.get("accelerate.data_loader")
    sys.modules["accelerate"] = fake_accelerate
    sys.modules["accelerate.data_loader"] = fake_data_loader
    try:
        spec = importlib.util.spec_from_file_location("mspc_loop_unit_test", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("accelerate", None)
        else:
            sys.modules["accelerate"] = previous
        if previous_data_loader is None:
            sys.modules.pop("accelerate.data_loader", None)
        else:
            sys.modules["accelerate.data_loader"] = previous_data_loader


loop = _load_loop_module()


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def export_trainable_state_dict(self, state_dict, *, remove_prefix):
        assert remove_prefix == "dit."
        return dict(state_dict)

    def trainable_param_names(self):
        return {"weight"}



class _Accelerator:
    is_main_process = True
    sync_gradients = True
    optimizer_step_was_skipped = False

    def __init__(self, payload: bytes = b"checkpoint") -> None:
        self.payload = payload

    def wait_for_everyone(self) -> None:
        return None

    def unwrap_model(self, model):
        return model

    def save(self, state_dict, path, *, safe_serialization):
        assert state_dict
        assert safe_serialization is True
        Path(path).write_bytes(self.payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_finite_scalar_loss_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        loop.finite_scalar_loss_value(torch.tensor(value))


def test_optimizer_groups_keep_mixed_storage_and_per_module_learning_rates():
    bf16_first = torch.nn.Parameter(torch.tensor([1.25], dtype=torch.bfloat16))
    fp32 = torch.nn.Parameter(torch.tensor([1.1], dtype=torch.float32))
    bf16_last = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.bfloat16))
    connector = torch.nn.Parameter(torch.tensor([0.25], dtype=torch.float32))
    source = [{"params": [bf16_first, fp32, bf16_last], "weight_decay": 0.02},
              {"params": [connector], "lr": 0.003}]
    groups = loop.optimizer_groups_by_dtype(source)
    assert [[id(p) for p in g["params"]] for g in groups] == [
        [id(bf16_first), id(bf16_last)], [id(fp32)], [id(connector)]]
    assert groups[0]["weight_decay"] == groups[1]["weight_decay"] == 0.02
    assert groups[2]["lr"] == 0.003
    # ZeRO's per-group flatten must preserve BF16 instead of promoting a mixed list.
    assert [torch._utils._flatten_dense_tensors(g["params"]).dtype for g in groups] == [
        torch.bfloat16, torch.float32, torch.float32]
    optimizer = torch.optim.AdamW(groups, lr=0.001)
    for parameter in (bf16_first, fp32, bf16_last, connector):
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert [p.dtype for g in optimizer.param_groups for p in g["params"]] == [
        torch.bfloat16, torch.bfloat16, torch.float32, torch.float32]
    assert len(source[0]["params"]) == 3


@pytest.mark.parametrize("across_groups", [False, True])
def test_optimizer_groups_reject_repeated_parameters(across_groups):
    parameter = torch.nn.Parameter(torch.ones(()))
    groups = ([{"params": [parameter]}, {"params": [parameter]}] if across_groups
              else [{"params": [parameter, parameter]}])
    with pytest.raises(ValueError, match="repeats a parameter"):
        loop.optimizer_groups_by_dtype(groups)


@pytest.mark.parametrize("groups", [[], [{"params": []}]])
def test_optimizer_groups_reject_empty_groups(groups):
    with pytest.raises(ValueError, match="empty|no parameter groups"):
        loop.optimizer_groups_by_dtype(groups)


def test_finite_scalar_loss_rejects_non_scalar_tensor() -> None:
    with pytest.raises(ValueError, match="scalar torch.Tensor"):
        loop.finite_scalar_loss_value(torch.ones(1))


def test_checkpoint_save_writes_only_nonempty_checkpoint(tmp_path: Path) -> None:
    logger = loop.TrainableStateDictLogger(str(tmp_path), "dit.")

    logger.save_model(_Accelerator(), _Model(), "step-1.safetensors")

    assert (tmp_path / "step-1.safetensors").read_bytes() == b"checkpoint"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["step-1.safetensors"]


def test_checkpoint_save_rejects_and_removes_empty_output(tmp_path: Path) -> None:
    logger = loop.TrainableStateDictLogger(str(tmp_path), "dit.")

    with pytest.raises(RuntimeError, match="checkpoint was not written"):
        logger.save_model(_Accelerator(b""), _Model(), "step-1.safetensors")

    assert not (tmp_path / "step-1.safetensors").exists()


def test_a_warm_started_run_counts_its_own_steps_from_zero(tmp_path: Path) -> None:
    """热启动不是续跑：步数从 0 起，checkpoint 按本次运行的步号命名。

    CHECKPOINT 只载权重（v5.3 §12），来源步号记在启动行里。曾经把它当起点用，
    结果是从 A0 ``step-3000`` 热启动、``max_train_steps=3000`` 的论文主臂
    一步都跑不了；500 步快筛同理。训练不再保存续跑状态，
    所以现在每条臂、包括热启动，都只有这一种起点。
    """

    logger = loop.TrainableStateDictLogger(str(tmp_path), "dit.")

    assert logger.num_steps == 0
    assert logger.latest_checkpoint_step is None

    for _ in range(2):
        logger.on_step_end(_Accelerator(), _Model(), save_steps=1)

    assert logger.num_steps == 2
    assert logger.latest_checkpoint_step == 2
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "step-1.safetensors",
        "step-2.safetensors",
    ]


def test_training_accumulation_preserves_first_microbatch_gradient(monkeypatch) -> None:
    class FakeOptimizer:
        def __init__(self, accelerator) -> None:
            self.accelerator = accelerator
            self.parameters = []
            self.param_groups = [{"lr": 1.0}]
            self.zero_grad_syncs = []
            self.step_gradients = []

        def zero_grad(self) -> None:
            self.zero_grad_syncs.append(self.accelerator.sync_gradients)
            if self.accelerator.sync_gradients:
                for parameter in self.parameters:
                    parameter.grad = None

        def step(self) -> None:
            if self.accelerator.sync_gradients:
                self.step_gradients.append(float(self.parameters[0].grad.item()))

    class FakeScheduler:
        def __init__(self) -> None:
            self.step_calls = 0

        def step(self) -> None:
            self.step_calls += 1

    class FakeAccelerator:
        device = torch.device("cpu")
        process_index = 0
        num_processes = 1
        is_main_process = True
        is_local_main_process = True
        optimizer_step_was_skipped = False

        def __init__(self) -> None:
            self.microbatch_index = 0
            self.sync_gradients = True

        @contextmanager
        def accumulate(self, model):
            self.sync_gradients = self.microbatch_index == 1
            try:
                yield
            finally:
                self.microbatch_index += 1

        def backward(self, loss) -> None:
            loss.backward()

        def prepare(self, model, optimizer, dataloader, scheduler):
            optimizer.parameters = list(model.trainable_modules())
            return model, optimizer, dataloader, scheduler

        def reduce(self, value, reduction):
            return value

        def unwrap_model(self, model):
            return model

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.last_timing = {}

        def trainable_modules(self):
            return [self.weight]

        def optimizer_param_groups(self, args):
            return [{"params": list(self.trainable_modules())}]

        def forward(self, data):
            return self.weight * data["value"]

    class FakeLogger:
        num_steps = 0
        latest_checkpoint_step = None

        def on_step_end(self, accelerator, model, save_steps) -> None:
            self.num_steps += 1

        def on_training_end(self, accelerator, model, save_steps) -> None:
            return None

    class FakeProgress:
        def set_postfix(self, metrics, refresh) -> None:
            return None

        def write(self, line, *, file) -> None:
            return None

        def update(self, count) -> None:
            return None

        def close(self) -> None:
            return None

    accelerator = FakeAccelerator()
    optimizer = FakeOptimizer(accelerator)
    scheduler = FakeScheduler()
    monkeypatch.setattr(loop.torch.optim, "AdamW", lambda *args, **kwargs: optimizer)
    monkeypatch.setattr(
        loop.torch.optim.lr_scheduler,
        "ConstantLR",
        lambda *args, **kwargs: scheduler,
    )
    monkeypatch.setattr(loop, "disable_hf_datasets_probe_for_torch_dataset", lambda: None)
    monkeypatch.setattr(loop, "tqdm", lambda **kwargs: FakeProgress())

    args = types.SimpleNamespace(
        learning_rate=1.0,
        weight_decay=0.0,
        training_seed=7,
        per_device_batch_size=1,
        dataloader_drop_last=False,
        dataset_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        num_epochs=1,
        timing_log_steps=0,
        timing_log_every=0,
        timing_sync_cuda=False,
        timing_prefix="[train_step]",
        progress_stream=None,
        log_dir=None,
        save_steps=100,
        max_train_steps=None,
        lr_scheduler_type="constant",
        lr_scheduler_factor=1.0,
        lr_scheduler_total_iters=1,
        gradient_accumulation_steps=2,
        shot_forward_mode="joint",
    )

    loop.train_sft(
        accelerator,
        [{"value": 1.0}, {"value": 2.0}],
        FakeModel(),
        FakeLogger(),
        args,
    )

    assert optimizer.step_gradients == [pytest.approx(3.0)]
    assert optimizer.zero_grad_syncs == [False, True]
    assert scheduler.step_calls == 2


def _warmup_scheduler(
    warmup_steps: int,
    num_processes: int,
    learning_rate: float = 1e-5,
):
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=learning_rate)
    args = types.SimpleNamespace(
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=warmup_steps,
    )
    scheduler = loop.build_lr_scheduler(optimizer, args, num_processes=num_processes)
    return optimizer, scheduler


def _advance_optimizer_updates(
    optimizer,
    scheduler,
    updates: int,
    num_processes: int,
) -> None:
    # Mirrors accelerate's AcceleratedScheduler with split_batches=False: one
    # synchronized optimizer update steps the wrapped scheduler once per process.
    for _ in range(updates):
        optimizer.step()
        for _ in range(num_processes):
            scheduler.step()


def test_constant_scheduler_holds_the_configured_learning_rate() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=1e-5)
    args = types.SimpleNamespace(
        lr_scheduler_type="constant",
        lr_scheduler_factor=1.0,
        lr_scheduler_total_iters=1,
    )

    scheduler = loop.build_lr_scheduler(optimizer, args)

    assert isinstance(scheduler, torch.optim.lr_scheduler.ConstantLR)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)
    for _ in range(8):
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


def test_constant_scheduler_ignores_the_process_count() -> None:
    args = types.SimpleNamespace(
        lr_scheduler_type="constant",
        lr_scheduler_factor=1.0,
        lr_scheduler_total_iters=1,
    )
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=1e-5)

    scheduler = loop.build_lr_scheduler(optimizer, args, num_processes=8)

    assert isinstance(scheduler, torch.optim.lr_scheduler.ConstantLR)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


@pytest.mark.parametrize("num_processes", [1, 8])
def test_constant_with_warmup_ramps_over_optimizer_updates(num_processes: int) -> None:
    optimizer, scheduler = _warmup_scheduler(4, num_processes)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)
    _advance_optimizer_updates(optimizer, scheduler, 1, num_processes)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.25e-5)
    _advance_optimizer_updates(optimizer, scheduler, 1, num_processes)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5e-5)
    _advance_optimizer_updates(optimizer, scheduler, 2, num_processes)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)
    _advance_optimizer_updates(optimizer, scheduler, 6, num_processes)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


def test_zero_warmup_steps_never_scales_the_learning_rate() -> None:
    optimizer, scheduler = _warmup_scheduler(0, 8)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)
    _advance_optimizer_updates(optimizer, scheduler, 3, 8)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


def test_constant_with_warmup_requires_warmup_steps() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=1e-5)
    args = types.SimpleNamespace(lr_scheduler_type="constant_with_warmup")

    with pytest.raises(ValueError, match="warmup_steps must be a non-negative integer"):
        loop.build_lr_scheduler(optimizer, args, num_processes=8)


def test_constant_with_warmup_requires_the_process_count() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=1e-5)
    args = types.SimpleNamespace(
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=300,
    )

    with pytest.raises(ValueError, match="denominated in optimizer updates"):
        loop.build_lr_scheduler(optimizer, args)


def test_unsupported_scheduler_type_is_rejected() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=1e-5)
    args = types.SimpleNamespace(lr_scheduler_type="cosine")

    with pytest.raises(ValueError, match="unsupported lr_scheduler_type='cosine'"):
        loop.build_lr_scheduler(optimizer, args, num_processes=1)


def test_one_step_backs_off_exactly_once_whatever_the_sample_is() -> None:
    """每 rank 每步的同步边界数与样本内容无关——镜级平铺之后由构造保证。

    此前逐镜臂在这里把一条故事切成 K 个镜，K 逐样本不同；rank 0 拿到 K=2、rank 3 拿到
    K=5 时每步发出的集合通信次数就不一样，NCCL 会永久错位（训练集群 1158279 的死法）。
    这条测试守住的是：``forward_backward`` 只有一条路径，一次前向一次反向。
    """

    loop = _load_loop_module()

    class _FakeAccelerator:
        def __init__(self) -> None:
            self.backward_calls = 0

        def backward(self, loss) -> None:
            self.backward_calls += 1

    class _FakeModel:
        def __call__(self, data):
            return torch.tensor(float(data["value"]), requires_grad=True)

    for value, latent_frames in ((1.0, 2), (2.0, 17), (3.0, 308)):
        accelerator = _FakeAccelerator()
        raw = types.SimpleNamespace(last_timing={"train_latent_frames": latent_frames})
        timings: dict = {}
        loop.forward_backward(
            accelerator=accelerator,
            model=_FakeModel(),
            raw_model=raw,
            data={"value": value},
            timings=timings,
            timing_sync_cuda=False,
        )
        assert accelerator.backward_calls == 1
        assert "shot_forward_count" not in timings


def test_each_step_gets_similar_lengths() -> None:
    """长度分桶要在大窗口里排序再切：只在相邻 ``bucket_size`` 个位置里排序时，accum=1 下同一步的样本
    还是同一批随机样本，各 rank 的步时差不会变小。"""

    generator = torch.Generator().manual_seed(7)
    lengths = torch.randint(1, 78, (4096,), generator=torch.Generator().manual_seed(3)).tolist()
    bucket = 8
    order = list(loop.CostBucketedSampler(lengths, generator=generator, bucket_size=bucket))

    assert sorted(order) == list(range(len(lengths)))
    spreads = [
        max(lengths[i] for i in order[start : start + bucket])
        - min(lengths[i] for i in order[start : start + bucket])
        for start in range(0, len(order), bucket)
    ]
    # 一个窗口 64 步 × 8 个样本，长度 1–77 均匀：同一步内的长度差约 77/64，远小于随机分组的约 60。
    assert sum(spreads) / len(spreads) < 4
    assert loop.COST_BUCKET_WINDOW_STEPS == 64
    # 同一窗口内各步的次序被打乱：相邻步的最短样本长度不按大小单调递增。
    minima = [min(lengths[i] for i in order[start : start + bucket]) for start in range(0, 64 * bucket, bucket)]
    assert minima != sorted(minima)


def test_calibrated_float_costs_keep_every_sample_and_replay_from_generator_state():
    """成本小于1秒时仍按浮点排序；每轮曝光及generator恢复不能受新排序损坏。"""
    costs = [value / 1000 for value in range(1, 518)]
    generator = torch.Generator().manual_seed(42)
    sampler = loop.CostBucketedSampler(costs, generator=generator, bucket_size=8)
    first = list(sampler)
    saved = generator.get_state().clone()
    second = list(sampler)
    replay = loop.CostBucketedSampler(costs, generator=torch.Generator().set_state(saved), bucket_size=8)
    assert list(replay) == second
    assert first != second
    assert sorted(first) == sorted(second) == list(range(len(costs)))
    # The short epoch tail is retained, and sorting by integer-truncated costs would fail this bound.
    spreads = [max(costs[i] for i in first[j:j+8])-min(costs[i] for i in first[j:j+8])
               for j in range(0, len(first), 8)]
    assert max(spreads[:loop.COST_BUCKET_WINDOW_STEPS]) < 0.03
    assert sampler.costs == tuple(costs)


def test_training_sampler_consumes_declared_workload_metadata_without_changing_exposure():
    workloads = ({"latent_frames": 49, "attention_frame_pairs": 850},
                 {"latent_frames": 49, "attention_frame_pairs": 2303},
                 {"latent_frames": 35, "attention_frame_pairs": 1120})
    class Dataset:
        def __len__(self): return len(workloads)
        def latent_frame_upper_bounds(self):
            raise AssertionError("calibrated grouping ignored the actual attention workload")
        def training_workloads(self, *, self_attention_policy):
            assert self_attention_policy == "holocine_first_frame"
            return workloads
    coefficients = {"linear_seconds_per_latent": 0.211351,
                    "attention_seconds_per_frame_pair": 0.00987451}
    config = {"model": {"self_attention_policy": "holocine_first_frame"},
              "training": {"workload_cost": coefficients}}
    sampler = loop.training_sampler(Dataset(), generator=torch.Generator().manual_seed(42),
                                    bucket_size=2, experiment_config=config)
    assert isinstance(sampler, loop.CostBucketedSampler)
    assert sampler.costs == pytest.approx([18.7495325, 33.09719553, 18.4567362])
    assert sorted(sampler) == [0, 1, 2]


def test_calibrated_grouping_rejects_a_dataset_without_real_workload_metadata():
    class Dataset:
        def __len__(self): return 1
        def latent_frame_upper_bounds(self): return [49]
    config = {"model": {"self_attention_policy": "holocine_first_frame"},
              "training": {"workload_cost": {"linear_seconds_per_latent": 0.2,
                                             "attention_seconds_per_frame_pair": 0.01}}}
    with pytest.raises(ValueError, match="dataset.training_workloads"):
        loop.training_sampler(Dataset(), generator=torch.Generator().manual_seed(42),
                               bucket_size=1, experiment_config=config)


def test_null_workload_cost_uses_the_existing_latent_lengths():
    class Dataset:
        def __len__(self): return 3
        def latent_frame_upper_bounds(self): return [49, 49, 35]
        def training_workloads(self, **kwargs):
            raise AssertionError("disabled calibration read attention metadata")
    generator = torch.Generator().manual_seed(42)
    sampler = loop.training_sampler(Dataset(), generator=generator, bucket_size=2,
                                    experiment_config={"training": {"workload_cost": None}})
    assert sampler.costs == (49., 49., 35.)
    assert sorted(sampler) == [0, 1, 2]


@pytest.mark.parametrize(("accum", "rows", "epochs", "maximum", "expected_micro", "expected_updates", "filtered"),
    [(2, 10, 1, 2, 4, 2, False), (2, 3, 2, None, 6, 4, False), (3, 5, 1, None, 5, 2, False),
     (2, 10000, 1, 4000, 8000, 4000, False), (1, 5, 1, 3, 3, 3, False),
     (1, 5, 1, 6, 6, 6, True)])
def test_update_units_cover_accumulation_and_epoch_tails(
    monkeypatch, tmp_path, accum, rows, epochs, maximum, expected_micro, expected_updates, filtered
):
    """无模型/无反传计数；DataLoader 尾同步按 Accelerate 的公开边界模拟。"""
    accelerator = types.SimpleNamespace(num_processes=1, process_index=0,
        is_main_process=True, is_local_main_process=True, sync_gradients=False,
        optimizer_step_was_skipped=False, micro=0, position=0)
    optimizer = types.SimpleNamespace(param_groups=[{"lr": 1e-5}], updates=0)
    scheduler = types.SimpleNamespace(updates=0)

    class Loader:
        def __len__(self): return rows
        def __iter__(self):
            for i in range(rows):
                accelerator.position = i
                yield {}

    @contextmanager
    def accumulate(_):
        accelerator.micro += 1
        accelerator.sync_gradients = (accelerator.position+1) % accum == 0 or accelerator.position+1 == rows
        yield

    accelerator.reduce = lambda value, reduction: value * accelerator.num_processes
    accelerator.device = torch.device("cpu")
    accelerator.accumulate = accumulate
    accelerator.prepare = lambda *items: items
    accelerator.unwrap_model = lambda model: model

    def optimizer_step():
        optimizer.updates += int(accelerator.sync_gradients)
    def scheduler_step():
        scheduler.updates += int(accelerator.sync_gradients)
    optimizer.step, optimizer.zero_grad, scheduler.step = optimizer_step, lambda: None, scheduler_step
    logger = loop.TrainableStateDictLogger(str(tmp_path), "dit.")
    saves = []
    monkeypatch.setattr(logger, "save_model", lambda a, m, name: saves.append((name, optimizer.updates)))

    class Progress:
        def __init__(self, **kwargs):
            self.total, self.count, self.lines = kwargs["total"], 0, []
        def update(self, n): self.count += n
        def set_postfix(self, *a, **kw): pass
        def write(self, line, **kw): self.lines.append(line)
        def close(self): pass
    bars = []
    def progress(**kwargs):
        bar = Progress(**kwargs)
        bars.append(bar)
        return bar
    monkeypatch.setattr(loop, "tqdm", progress)
    monkeypatch.setattr(loop.torch.optim, "AdamW", lambda *a, **kw: optimizer)
    monkeypatch.setattr(loop.torch.utils.data, "DataLoader", lambda *a, **kw: Loader())
    monkeypatch.setattr(loop, "build_lr_scheduler", lambda *a, **kw: scheduler)
    monkeypatch.setattr(loop, "disable_hf_datasets_probe_for_torch_dataset", lambda: None)
    monkeypatch.setattr(loop, "training_sampler", lambda *a, **kw: None)
    monkeypatch.setattr(loop, "forward_backward", lambda **kw: 0.0)
    args = types.SimpleNamespace(learning_rate=1e-5, weight_decay=.01, training_seed=42,
        per_device_batch_size=1, dataloader_drop_last=False, dataset_num_workers=0,
        dataloader_pin_memory=False, dataloader_persistent_workers=False,
        gradient_accumulation_steps=accum, max_train_steps=maximum, num_epochs=epochs,
        timing_log_steps=1, timing_log_every=2, timing_sync_cuda=False,
        timing_prefix="[test]", save_steps=2, progress_stream=None, log_dir=None)
    model = types.SimpleNamespace(optimizer_param_groups=lambda _: [{"params": [torch.nn.Parameter(torch.ones(()))]}])
    class FilteredDataset(list):
        excluded_sample_indices = frozenset({rows})
    dataset = FilteredDataset([{}]*rows) if filtered else [{}]*rows
    loop.train_sft(accelerator, dataset, model, logger, args)
    assert accelerator.micro == expected_micro
    assert optimizer.updates == scheduler.updates == logger.num_steps == expected_updates
    assert bars[0].total == (maximum if maximum is not None else expected_updates)
    assert bars[0].count == len(bars[0].lines) == expected_updates
    assert f"microstep={expected_micro}" in bars[0].lines[-1]
    assert saves[-1] == (f"step-{expected_updates}.safetensors", expected_updates)
    assert all(int(name.split("-")[1].split(".")[0]) == updates for name, updates in saves)


@pytest.mark.parametrize("cached_text", [False, True])
def test_each_rank_logs_one_line_per_sample_with_its_loss(monkeypatch, tmp_path, cached_text):
    """C8：数据集给出 ``_sample_metadata`` 时，每个 microstep 写一行 samples.rank{r}.jsonl。"""
    import json

    accelerator = types.SimpleNamespace(num_processes=8, process_index=3,
        is_main_process=False, is_local_main_process=False, sync_gradients=True,
        optimizer_step_was_skipped=False)
    rows = [{"_sample_metadata": {"case_id": f"case-{i}", "k": i, "references": [["s01", "last"]] * i,
                                  "state": i % 2 == 0, "conflict": i == 2}} for i in range(3)]

    @contextmanager
    def accumulate(_):
        yield

    accelerator.reduce = lambda value, reduction: value * accelerator.num_processes
    accelerator.device = torch.device("cpu")
    accelerator.accumulate = accumulate
    accelerator.prepare = lambda *items: items
    accelerator.unwrap_model = lambda model: model
    optimizer = types.SimpleNamespace(param_groups=[{"lr": 1e-5}], step=lambda: None, zero_grad=lambda: None)
    monkeypatch.setattr(loop.torch.optim, "AdamW", lambda *a, **kw: optimizer)
    monkeypatch.setattr(loop.torch.utils.data, "DataLoader", lambda *a, **kw: [dict(r) for r in rows])
    monkeypatch.setattr(loop, "build_lr_scheduler", lambda *a, **kw: types.SimpleNamespace(step=lambda: None))
    monkeypatch.setattr(loop, "disable_hf_datasets_probe_for_torch_dataset", lambda: None)
    monkeypatch.setattr(loop, "training_sampler", lambda *a, **kw: None)
    losses = iter([0.5, 0.25, 0.125])
    forwarded = []
    def forward_backward(**kwargs):
        forwarded.append(kwargs["data"])
        return next(losses)
    monkeypatch.setattr(loop, "forward_backward", forward_backward)
    monkeypatch.setattr(loop, "cuda_peak_memory_line", lambda *a: "")
    logger = types.SimpleNamespace(num_steps=0, on_step_end=lambda *a: None, on_training_end=lambda *a: None)
    args = types.SimpleNamespace(learning_rate=1e-5, weight_decay=.01, training_seed=42,
        per_device_batch_size=1, dataloader_drop_last=False, dataset_num_workers=0,
        dataloader_pin_memory=False, dataloader_persistent_workers=False,
        gradient_accumulation_steps=1, max_train_steps=None, num_epochs=1,
        timing_log_steps=1, timing_log_every=0, timing_sync_cuda=False,
        timing_prefix="[test]", save_steps=2, progress_stream=None, log_dir=str(tmp_path),
        model_cfg={"text_context_cache": "/complete/manifest.json" if cached_text else None})
    model = types.SimpleNamespace(optimizer_param_groups=lambda _: [{"params": [torch.nn.Parameter(torch.ones(()))]}], profile_sync_cuda=False)
    calls = []
    validation = types.SimpleNamespace(due=lambda step: step % 2 == 0,
                                       run=lambda raw_model, step, **kw: calls.append((raw_model is model, step)) or 0.0)
    loop.train_sft(accelerator, rows, model, logger, args, validation=validation)
    # 留出验证：训练开始前量一遍（第 0 步），之后每到 due 的更新数量一遍；拿到的是解开包装的模块。
    assert calls == [(True, 0), (True, 2)]
    for sample, original in zip(forwarded, rows, strict=True):
        assert "_sample_metadata" not in sample
        if cached_text:
            assert sample["_text_cache_sample"]["split"] == "train"
            assert sample["_text_cache_sample"]["metadata"] is original["_sample_metadata"]
        else:
            assert "_text_cache_sample" not in sample
    lines = [json.loads(line) for line in (tmp_path / "samples.rank3.jsonl").read_text().splitlines()]
    assert all("timestep" in line for line in lines)  # 逐样本账带这一步抽到的时间步（这里的假前向不报，所以是 null）
    assert [(l["step"], l["case_id"], l["k"], l["conflict"], l["loss"]) for l in lines] == [
        (1, "case-0", 0, False, 0.5), (2, "case-1", 1, False, 0.25), (3, "case-2", 2, True, 0.125)]
    assert [len(l["references"]) for l in lines] == [0, 1, 2] and [l["state"] for l in lines] == [True, False, True]


def test_four_rank_accumulation_shares_optimizer_step_losses_and_timing_with_swanlab(monkeypatch, tmp_path):
    """CPU日志合同：模拟4rank归约与accum2边界，真实前反向；不代表多卡吞吐或稳定性。"""
    import json
    from mspc.core.loss_curves import update_scalar_curves

    clock = [0.]
    monkeypatch.setattr(loop, "time", types.SimpleNamespace(
        perf_counter=lambda: clock[0], time_ns=lambda: int(clock[0] * 1e9)))

    def advance(seconds):
        clock[0] += seconds

    accelerator = types.SimpleNamespace(num_processes=4, process_index=0, device=torch.device("cpu"),
        is_main_process=True, is_local_main_process=True, sync_gradients=False,
        optimizer_step_was_skipped=False, micro=0)

    @contextmanager
    def accumulate(_):
        accelerator.micro += 1
        accelerator.sync_gradients = accelerator.micro % 2 == 0
        yield

    reductions = []

    def reduce(value, reduction):
        assert accelerator.sync_gradients and reduction == "sum"
        reductions.append(value.tolist())
        # rank1/2/3的每条损失分别为rank0的2/3/4倍，分母都是2条。
        return value * torch.tensor([10.] * (value.numel() - 1) + [4.], dtype=value.dtype)

    def backward(loss):
        loss.backward()
        advance(4.)

    accelerator.accumulate, accelerator.reduce, accelerator.backward = accumulate, reduce, backward
    accelerator.prepare = lambda *items: items
    accelerator.unwrap_model = lambda model: model

    class Loader:
        def __len__(self):
            return 4

        def __iter__(self):
            for index, value in enumerate((1., 3., 5., 7.)):
                advance(2.)
                yield {"value": value, "_sample_metadata": {"sample_index": index}}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def optimizer_param_groups(self, _):
            return [{"params": list(self.parameters())}]

        def forward(self, data):
            value = data["value"]
            self.last_timing = {"uniform_loss": value * 2, "target_weighted_mse": value / 2,
                                "unweighted_mse": value, "loss_weight": 2.}
            advance(3.)
            return self.weight * value

    optimizer = types.SimpleNamespace(param_groups=[{"lr": 1e-5}],
                                     step=lambda: advance(1.), zero_grad=lambda: None)
    monkeypatch.setattr(loop.torch.optim, "AdamW", lambda *a, **kw: optimizer)
    monkeypatch.setattr(loop.torch.utils.data, "DataLoader", lambda *a, **kw: Loader())
    monkeypatch.setattr(loop, "build_lr_scheduler", lambda *a, **kw: types.SimpleNamespace(step=lambda: None))
    monkeypatch.setattr(loop, "disable_hf_datasets_probe_for_torch_dataset", lambda: None)
    monkeypatch.setattr(loop, "training_sampler", lambda *a, **kw: None)
    monkeypatch.setattr(loop, "cuda_peak_memory_line", lambda *a: "")
    points = []
    monkeypatch.setattr(loop, "open_loss_tracking", lambda *a, **kw: types.SimpleNamespace(
        log=lambda step, scalars: (points.append({"step": step, **scalars}), advance(1.)), finish=lambda: None))

    class Logger:
        num_steps = 0
        latest_checkpoint_step = None

        def on_step_end(self, *args):
            self.num_steps += 1
            advance(6.)

        def on_training_end(self, *args):
            advance(9.)

    validation_calls = []
    def validate(raw_model, step, *, full=False):
        validation_calls.append((step, full))
        advance(20.)
        return 20.
    validation = types.SimpleNamespace(due=lambda step: True, run=validate)
    args = types.SimpleNamespace(learning_rate=1e-5, weight_decay=0., training_seed=42,
        per_device_batch_size=1, dataloader_drop_last=True, dataset_num_workers=0,
        dataloader_pin_memory=False, dataloader_persistent_workers=False,
        gradient_accumulation_steps=2, max_train_steps=2, num_epochs=1,
        timing_log_steps=0, timing_log_every=0, timing_sync_cuda=False, timing_prefix="[test]",
        save_steps=2, progress_stream=None, log_dir=str(tmp_path), loss_tracking=None, experiment_config={})
    loop.train_sft(accelerator, [{}] * 4, Model(), Logger(), args, validation=validation)
    assert validation_calls == [(0, False), (1, False), (2, True)]
    metrics = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    samples = [json.loads(line) for line in (tmp_path / "samples.rank0.jsonl").read_text().splitlines()]
    assert [row["step"] for row in metrics] == [1, 2]
    assert [row["step"] for row in samples] == [1, 1, 2, 2]
    assert [row["samples_seen"] for row in metrics] == [8, 16]
    assert len(reductions) == 2 and all(row[-1] == 2 for row in reductions)
    assert [row["loss"] for row in metrics] == [5., 15.]
    assert [row["uniform_loss"] for row in metrics] == [10., 30.]
    assert [row["target_weighted_mse"] for row in metrics] == [2.5, 7.5]
    assert [row["unweighted_mse"] for row in metrics] == [5., 15.]
    assert [row["elapsed_seconds"] for row in metrics] == [46., 93.]
    assert all((row["training_step_seconds"], row["step_seconds"], row["data_seconds"],
                row["checkpoint_hook_seconds"]) == (16., 22., 4., 6.) for row in metrics)
    assert points == [{"step": row["step"], **update_scalar_curves(row)} for row in metrics]
    # 最后一条点之后的验证和最终补存不回填到该点；全程时长由运行汇总记录。
    assert clock[0] == 123.


def test_a_zero_partition_larger_than_int32_refuses_to_start():
    """训练集群 2026-10-12：5B 全参、2 卡时每个 rank 的那一份约 25 亿个元素，第 1 次更新就把权重写坏；4 卡每份 12.5 亿正常。"""
    class Parameter:
        def __init__(self, elements, requires_grad=True):
            self.elements, self.requires_grad = elements, requires_grad

        def numel(self):
            return self.elements

    full_dit = [{"params": [Parameter(4_999_774_276), Parameter(10**10, requires_grad=False)]}]
    with pytest.raises(RuntimeError, match="at least 3 processes"):
        loop.require_zero_partitions_within_int32(full_dit, 2)
    for world in (4, 8):
        loop.require_zero_partitions_within_int32(full_dit, world)
    loop.require_zero_partitions_within_int32([{"params": [Parameter(2**31 - 1)]}], 1)
    with pytest.raises(RuntimeError, match="group 1"):
        loop.require_zero_partitions_within_int32([{"params": [Parameter(8)]}, {"params": [Parameter(2**31)]}], 1)


def test_step_peaks_are_per_step_and_the_run_peak_keeps_earlier_steps(monkeypatch):
    """每步先清零再读回；全程峰值含两步之间（验证、存盘）的峰值；一次更新内取最大，不累加。"""
    loop = _load_loop_module()
    peaks = {"allocated": 7, "reserved": 9}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: peaks["allocated"])
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: peaks["reserved"])
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: peaks.update(allocated=0, reserved=0))
    recorder = loop.CudaStepPeaks()
    recorder.start()  # 起训前的 7/9（如起点验证）并入全程峰值
    peaks.update(allocated=5, reserved=6)
    assert recorder.finish() == {"cuda_peak_allocated_bytes": 5, "cuda_peak_reserved_bytes": 6}
    recorder.start()
    peaks.update(allocated=3, reserved=4)
    assert recorder.finish() == {"cuda_peak_allocated_bytes": 3, "cuda_peak_reserved_bytes": 4}
    assert recorder.run_peaks() == (7, 9)
    update = {}
    for microstep in ({"cuda_peak_reserved_bytes": 6, "forward": 1.0}, {"cuda_peak_reserved_bytes": 4, "forward": 2.0}):
        loop.accumulate_update_timings(update, microstep)
    assert update == {"cuda_peak_reserved_bytes": 6, "forward": 3.0}
