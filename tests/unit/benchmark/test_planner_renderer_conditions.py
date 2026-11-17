"""论文表 1 我们这一行的出片条件（``configs/infer/planner_renderer_test92.json``）与定稿推理配置要对得上。"""
from pathlib import Path

from mspc.benchmark.layout_baseline import _checkpoint, load_run, repo_root, resolve_path
from mspc.config import load_experiment_config

CONDITIONS = repo_root() / "configs/infer/planner_renderer_test92.json"


def test_conditions_declare_the_checkpoint_the_final_inference_config_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("MSPC_SHARED_ROOT", str(tmp_path))
    condition = load_run(CONDITIONS)["conditions"][0]
    config_path = resolve_path(condition["renderer_config"], repo_root())
    assert _checkpoint(config_path) == str(Path(condition["checkpoint"]).resolve())
    model = load_experiment_config(config_path).to_dict()["model"]
    assert model["text_context_cache"] is None
    assert (condition["history_policy"], condition["segment_max_frames"], condition["seed"]) == ("plan_selection", 112, 42)
