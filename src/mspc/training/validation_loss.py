"""训练中的留出验证 loss：固定的窗、固定的噪声与时间步，每隔若干次更新量一遍。

为什么要它：流匹配的训练 loss 随每条样本抽到的时间步起伏一个量级，单看它判断不了收敛。同一批留出窗在同一组噪声与
时间步上反复量，前后两遍的差只来自权重。第 0 步（训练开始前）也量一遍，是起点权重的读数。

没有任何跨 rank 通信：每个 rank 量下标 ``i % world == rank`` 的那一份窗，写自己的
``<log_dir>/validation.rank{r}.jsonl``（一行一次前向：step、sample_index、case_id、span、references、
reference_count、draw、镜数、时间步、loss）；汇总在 ``mspc.core.loss_curves``。训练循环
「每 rank 每步恰好一次集合通信」的约束因此不受影响。

随机数：每次前向之前按 (seed, sample_index, draw) 设种子；同一块可抽到不同目标段，面板下标区分每次曝光。
所以每一遍用的是同一组时间步与噪声；量完把训练自己的随机数状态原样放回，训练抽到的时间步与噪声同不量验证时一模一样。

显式配置 ``intermediate_panel`` 时，普通中间点只量完整面板前若干个sample_index和前若干draw；
第0步、完整更新停止点与训练终点仍量完整面板。日志每行写实际面板和全rank前向总分母，曲线分别汇总。
工程容量门可显式给原sample_indices，只量该子集；仍按原下标分rank、取噪声种子并保留原draw数。
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import torch

from mspc.core.draws import unit_draw
from mspc.training.training_state import _capture_rng, _restore_rng


class ValidationLoss:
    def __init__(self, dataset: Any, *, every_steps: int, draws: int, seed: int, rank: int, world: int,
                 log_dir: str | Path, loader_kwargs: dict[str, Any],
                 intermediate_panel: dict[str, int] | None = None) -> None:
        if type(every_steps) is not int or every_steps < 1 or type(draws) is not int or draws < 1:
            raise ValueError("validation every_steps and draws must be positive ints")
        self.dataset, self.every_steps, self.draws, self.seed, self.rank = dataset, every_steps, draws, seed, rank
        self.global_indices = tuple(getattr(dataset, "eligible_sample_indices", range(len(dataset))))
        self.indices = [index for index in self.global_indices if index % world == rank]
        self.intermediate_panel = intermediate_panel
        if intermediate_panel is not None:
            if set(intermediate_panel) != {"sample_budget", "draws"}:
                raise ValueError("validation intermediate_panel requires sample_budget and draws")
            for name, maximum in (("sample_budget", len(dataset)), ("draws", draws)):
                value = intermediate_panel[name]
                if type(value) is not int or not 1 <= value <= maximum:
                    raise ValueError(f"validation intermediate_panel.{name} must be between 1 and {maximum}")
            self.intermediate_indices = [index for index in self.indices
                                         if index < intermediate_panel["sample_budget"]]
            self.intermediate_sample_budget = sum(index < intermediate_panel["sample_budget"]
                                                  for index in self.global_indices)
        # sampler在主进程派发全局下标，切换面板不复制persistent worker池或其dataset。
        self._loader_indices = list(self.indices)
        from mspc.training.loop import collate_single_complete_sample
        self.loader = torch.utils.data.DataLoader(dataset, sampler=self._loader_indices,
            collate_fn=collate_single_complete_sample, **{**loader_kwargs, "drop_last": False},
            generator=torch.Generator().manual_seed(seed))
        self.path = Path(log_dir) / f"validation.rank{rank}.jsonl"

    def due(self, step: int) -> bool:
        return step % self.every_steps == 0

    def run(self, raw_model: Any, step: int, *, full: bool = False,
            sample_indices: list[int] | None = None) -> float:
        """量一遍，返回用掉的秒数。``raw_model`` 是解开包装的训练模块：直接调用模块（仍有DeepSpeed hooks），不动优化器状态。"""
        if sample_indices is not None:
            if (not sample_indices or len(set(sample_indices)) != len(sample_indices)
                    or any(type(index) is not int or index not in self.global_indices for index in sample_indices)):
                raise ValueError("validation sample_indices must be distinct original dataset indices")
            panel, draws, sample_budget = "capacity", self.draws, len(sample_indices)
            owned = set(self.indices)
            indices = [index for index in sample_indices if index in owned]
        elif self.intermediate_panel is not None and step != 0 and not full:
            panel = "intermediate"
            indices = self.intermediate_indices
            draws, sample_budget = self.intermediate_panel["draws"], self.intermediate_sample_budget
        else:
            panel = "full"
            indices, draws, sample_budget = self.indices, self.draws, len(self.global_indices)
        self._loader_indices[:] = indices
        started = time.perf_counter()
        rng_state = _capture_rng()
        module_modes = [(module, module.training) for module in raw_model.modules()]
        raw_model.eval()
        try:
            with torch.no_grad(), self.path.open("a", encoding="utf-8") as handle:
                previous_end = time.perf_counter()
                for position, sample in enumerate(self.loader):
                    data_seconds = time.perf_counter() - previous_end
                    index = indices[position]
                    metadata = sample["_sample_metadata"]
                    data = {key: value for key, value in sample.items() if not key.startswith("_")}
                    if getattr(raw_model, "model_cfg", {}).get("text_context_cache") is not None:
                        data["_text_cache_sample"] = {"split": "validation", "metadata": metadata}
                    for draw in range(draws):
                        draw_seed = int(unit_draw("validation_noise", self.seed, index, draw) * 2**31)
                        torch.manual_seed(draw_seed)
                        data["flow_draw_seed"] = draw_seed
                        forward_start = time.perf_counter()
                        loss = float(raw_model(dict(data)))
                        forward_seconds = time.perf_counter() - forward_start
                        handle.write(json.dumps({
                            **metadata, "step": step, "rank": self.rank, "sample_index": index,
                            "validation_panel": panel, "validation_forwards": sample_budget * draws,
                            "case_id": metadata["case_id"], "draw": draw,
                            "data_seconds": data_seconds if draw == 0 else 0.0,
                            "forward_seconds": forward_seconds,
                            "span": metadata["span"], "references": metadata["references"],
                            "reference_count": len(metadata["references"]),
                            "shots": metadata["span"][1] - metadata["span"][0],
                            **{k: raw_model.last_timing[k] for k in ("timestep", "flow_draw_seed", "target_noise_probe", "target_frame_mse", "unweighted_mse", "loss_weight", "source_alignment_loss", "weighted_source_alignment_loss") if k in raw_model.last_timing},
                            "loss": loss}) + "\n")
                    handle.flush()
                    previous_end = time.perf_counter()
        finally:
            for module, training in module_modes:
                module.training = training
            _restore_rng(rng_state)
        return time.perf_counter() - started
