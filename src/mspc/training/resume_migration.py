"""续训迁移：绑定原件，核代码根搬家、两种缓存只扩不换、接上参考前缀与实际DiT/优化器顺序。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mspc.runs.repository import require_clean_worktree

ARTIFACT_TYPE = "mspc_native_training_resume_migration"
_FIELDS = {"artifact_type", "schema_version", "from_repository_sha", "to_repository_sha",
           "run_id", "run_root", "state_path", "completed_steps", "world_size",
           "source_config", "target_config", "source_config_path", "target_config_path",
           "source_code_root", "target_code_root",
           "parameter_layout", "optimizer"}
_CACHE_FIELDS = {"target_index_path", "target_roots", "cache_provenance_path"}
#: 接上参考前缀时一起换的字段（配置解析要求它们与 ``model.memory_prefix`` 同时声明）：参考本身、参考标签与裁剪表、
#: 参考行丢弃，以及随参考行改变的文字准入。不给参考的一侧全部为 None。
_REFERENCE_FIELDS = (("model", "memory_prefix"), ("dataset", "reference_entities"), ("dataset", "reference_crops"),
                     ("dataset", "reference_row_dropout_p"), ("training", "validation", "reference_entities"))
_REFERENCE_TEXT_FIELDS = (("dataset", "text_admission_path"), ("training", "validation", "text_admission_path"))


def _sha(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 40
            and all(character in "0123456789abcdef" for character in value))


def _same_path(left: Any, right: Any) -> bool:
    return (isinstance(left, str) and isinstance(right, str)
            and Path(left).is_absolute() and Path(right).is_absolute()
            and os.path.realpath(left) == os.path.realpath(right))


def _code_path(value: Any, source_root: str, target_root: str) -> Any:
    if isinstance(value, str) and (value == source_root or value.startswith(source_root + "/")):
        return target_root + value[len(source_root):]
    if isinstance(value, list):
        return [_code_path(item, source_root, target_root) for item in value]
    return value


def _index(path: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    result = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["sid"], row["scene_id"], row["source_shot_index"])
            if key in result:
                raise ValueError(f"migration target index contains duplicate key: {key}")
            result[key] = row
    if not result:
        raise ValueError("migration target index must be nonempty")
    return result


def _target_extension(source: dict[str, Any], target: dict[str, Any]) -> None:
    """所有旧键继续读同一文件和同一区间；新增键只扩供给，不换已见张量。"""
    old, new = _index(source["target_index_path"]), _index(target["target_index_path"])
    for key, before in old.items():
        after = new.get(key)
        if after is None or {k: v for k, v in before.items() if k != "origin"} != {
                k: v for k, v in after.items() if k != "origin"}:
            raise ValueError(f"migration changed or removed an original target index row: {key}")
        old_file = Path(source["target_roots"][before["origin"]]) / before["file"]
        new_file = Path(target["target_roots"][after["origin"]]) / after["file"]
        if old_file.resolve(strict=True) != new_file.resolve(strict=True):
            raise ValueError(f"migration replaced the original target tensor source: {key}")
    # Build from actual selected manifests; equality also verifies the new source
    # collection instead of accepting an unrelated provenance path.
    from mspc.data.cache_provenance import build_cache_source_collection
    for dataset in (source, target):
        actual = build_cache_source_collection(Path(dataset["target_index_path"]),
            {name: Path(root) for name, root in dataset["target_roots"].items()})
        recorded = json.loads(Path(dataset["cache_provenance_path"]).read_text())
        if actual != recorded:
            raise ValueError("migration cache source collection differs from its actual index/manifests")


def _text_cache_continuity(before: Path, after: Path) -> None:
    """文字缓存不换已编码的串：T5 编码合同相同；两份缓存都有的串指向同一份张量文件。

    补编新停点时由 ``text_cache.reuse_text_cache`` 复用旧张量（链回原文件）、只为新串编码，所以按物理文件比对，
    与 :func:`_target_extension` 对目标张量的要求同一口径。旧缓存里有、新缓存里没有的串可以不在：接上参考会改写
    文字抽签，组批次序改变会换掉 epoch 尾被 ``drop_last`` 丢掉的几条样本（11-16）。新缓存是否覆盖目标配置的完整
    阶段与抽签合同，由训练起点的 ``_CachedTextContextResolver.require_training_contract`` 逐条核。
    """
    old, new = json.loads(before.read_text()), json.loads(after.read_text())
    if old["encoding_contract"] != new["encoding_contract"]:
        raise ValueError("resume migration text cache changes the T5 encoding contract")

    def contexts(manifest: Path) -> dict[str, Path]:
        rows = {}
        with (manifest.parent / "prompts.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                rows[row["prompt"]] = (manifest.parent / "contexts" / f"{row['row_id']}.safetensors").resolve(strict=True)
        return rows

    extended = contexts(after)
    for prompt, path in contexts(before).items():
        if prompt in extended and extended[prompt] != path:
            raise ValueError("resume migration text cache replaced an already-encoded prompt")


def _field(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    for name in path[:-1]:
        config = config[name]
    return config[path[-1]]


def _set_field(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    for name in path[:-1]:
        config = config[name]
    config[path[-1]] = value


def _connects_references(source: dict[str, Any], target: dict[str, Any]) -> bool:
    """不给参考训练的 run 续训时接上参考前缀（MultiShotMaster 式「先联合生成、再加参考」两阶段）。

    只认一个方向：原配置的参考字段全部为 None、目标配置给出参考前缀。目标侧这组字段整体取目标配置的值，
    由配置解析核它们彼此一致；其余字段仍由 :func:`validate_config_migration` 逐项核。参考前缀不加参数，
    DiT／优化器顺序照旧由原生状态 geometry 与 :func:`require_model_migration_layout` 核。
    """
    if target["model"]["memory_prefix"] is None or source["model"]["memory_prefix"] is not None:
        return False
    if any(_field(source, path) is not None for path in _REFERENCE_FIELDS):
        raise ValueError("resume migration connects references only from a run trained without any reference field")
    return True


def validate_config_migration(contract: dict[str, Any]) -> None:
    """续训只允许五类变化：代码入口搬到新代码根、目标缓存只扩不换、文字缓存不换已编码的串、存盘间隔取原值或
    1000、组批成本系数（只改 epoch 内次序）、不给参考训练的 run 接上参考前缀（:func:`_connects_references`）；
    其余数学与数据抽样字段逐项一致。
    任何 run 都按同一规则，没有按 run 名放行的特例。"""
    source, target = contract["source_config"], contract["target_config"]
    if not isinstance(source, dict) or not isinstance(target, dict):
        raise ValueError("resume migration requires complete source and target configs")
    # Work on JSON copies; the old state/config is never rewritten.
    normalized = json.loads(json.dumps(source))
    for section in ("dataset", "model", "training"):
        if not isinstance(normalized.get(section), dict) or not isinstance(target.get(section), dict):
            raise ValueError(f"resume migration requires complete {section} configs")
    for field in _CACHE_FIELDS:
        normalized["dataset"][field] = target["dataset"][field]
    connects = _connects_references(source, target)
    if connects:
        for path in _REFERENCE_FIELDS + _REFERENCE_TEXT_FIELDS:
            _set_field(normalized, path, json.loads(json.dumps(_field(target, path))))
    before, cache = source["model"].get("text_context_cache"), target["model"].get("text_context_cache")
    if cache is not None and (not isinstance(cache, str) or not Path(cache).is_absolute()):
        raise ValueError("cached-text migration requires an explicit artifact path")
    if before is not None and cache is None:
        raise ValueError("resume migration cannot drop the text cache of a cached-text run")
    # The new parser's explicit null is a representation change for live configs.
    if "text_context_cache" in target["model"]:
        normalized["model"]["text_context_cache"] = cache
    # 组批成本系数只决定一个 epoch 内的样本次序（CostBucketedSampler 窗口内排序），不改哪些样本、样本数与逐样本抽签；
    # 续训时可换成实测标定的系数（11-16 定稿续训）。完整状态里存的是 epoch 起点的随机状态，新次序由同一状态重放。
    if "workload_cost" in target["training"]:
        normalized["training"]["workload_cost"] = target["training"]["workload_cost"]
    if target["training"].get("save_steps") not in (source["training"].get("save_steps"), 1000):
        raise ValueError("resume migration only permits the existing or 1000-update save interval")
    normalized["training"]["save_steps"] = target["training"]["save_steps"]
    source_root, target_root = contract["source_code_root"], contract["target_code_root"]
    if any(not isinstance(root, str) or not Path(root).is_absolute() for root in (source_root, target_root)):
        raise ValueError("resume migration requires explicit absolute frozen code roots")
    if any(not isinstance(path, str) or not Path(path).is_absolute()
           for path in (contract["source_config_path"], contract["target_config_path"])):
        raise ValueError("resume migration requires the explicit original and target config paths")
    # Relocate only the code entrypoints. Assets must still resolve to the same
    # object, including tokenizer, VAE, source data, runtime and output directory.
    for section, field in (("model", "diffsynth_root"), ("launch", "train_entrypoint"),
                           ("launch", "train_pythonpath"), ("launch", "infer_entrypoint"),
                           ("launch", "infer_pythonpath")):
        normalized[section][field] = _code_path(normalized[section][field], source_root, target_root)

    def compare(left: Any, right: Any, path: str) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                raise ValueError(f"resume migration config keys differ at {path}")
            for key in left:
                compare(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                raise ValueError(f"resume migration config length differs at {path}")
            for index, (before, after) in enumerate(zip(left, right, strict=True)):
                compare(before, after, f"{path}[{index}]")
        elif left != right and not _same_path(left, right):
            raise ValueError(f"resume migration changes an unapproved config field: {path}")
    compare(normalized, target, "config")
    _target_extension(source["dataset"], target["dataset"])
    if before is not None and not _same_path(before, cache):
        _text_cache_continuity(Path(before), Path(cache))


def native_state_layout(state_path: str | Path, *, completed_steps: int, world_size: int) -> dict[str, Any]:
    """mmap只取旧DeepSpeed元数据与张量geometry，不复制70GB状态。"""
    path = Path(state_path)
    import torch
    from mspc.training.training_state import _validate_training_state_files
    _validate_training_state_files(path, world_size=world_size, deepspeed=True)
    saved = torch.load(path / "pytorch_model/mp_rank_00_model_states.pt",
                       map_location="cpu", mmap=True, weights_only=False)
    if saved["global_steps"] != completed_steps or saved["dp_world_size"] != world_size:
        raise ValueError("resume migration native DeepSpeed step/world differs from its receipt")
    module, groups = saved["module"], saved["param_shapes"]
    names = [name for group in groups for name in group]
    if len(names) != len(set(names)) or set(names) != set(module):
        raise ValueError("resume migration native parameter and optimizer names differ")
    for group in groups:
        for name, shape in group.items():
            if list(module[name].shape) != list(shape):
                raise ValueError("resume migration native optimizer parameter geometry differs")
    return {"parameter_layout": [{"name": name, "shape": list(value.shape),
                                   "dtype": str(value.dtype).removeprefix("torch.")}
                                  for name, value in module.items()],
            "optimizer": {"class": "torch.optim.AdamW", "parameter_groups": [list(group) for group in groups]}}


def source_snapshot_path(contract: dict[str, Any]) -> Path:
    return Path(contract["run_root"]) / "control" / f"resume-migration-source-step-{contract['completed_steps']}.json"


def migration_source_snapshot(contract: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    """首次load前失败可重试同一合同；旧身份只从未改写的封存snapshot读取。"""
    if snapshot.get("resume_migration") == contract:
        if (snapshot["repository_sha"] != contract["to_repository_sha"]
                or snapshot["config"] != contract["target_config"]
                or snapshot["source_config"] != contract["target_config_path"]):
            raise ValueError("resume migration retry target snapshot differs from the exact contract")
        return json.loads(source_snapshot_path(contract).read_text())
    return snapshot


def archive_migration_source_snapshot(contract: dict[str, Any], snapshot: dict[str, Any]) -> None:
    """完整写入并fsync后独占公布，失败重试不留下半份source truth。"""
    import tempfile
    path = source_snapshot_path(contract)
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != snapshot:
            raise ValueError("resume migration original source snapshot archive differs")
        return
    with tempfile.NamedTemporaryFile(mode="w", prefix=".migration-source-", dir=path.parent,
                                      delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(snapshot, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def read_resume_migration(path: str | Path, *, state_path: str | Path,
                          target_config: dict[str, Any], repository_sha: str | None,
                          world_size: int, target_config_path: str | Path) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text())
    if (not isinstance(contract, dict) or set(contract) != _FIELDS
            or contract["artifact_type"] != ARTIFACT_TYPE or contract["schema_version"] != 1):
        raise ValueError("invalid native training resume migration contract")
    if (not _sha(contract["from_repository_sha"]) or not _sha(contract["to_repository_sha"])
            or contract["from_repository_sha"] == contract["to_repository_sha"]
            or contract["to_repository_sha"] != repository_sha):
        raise ValueError("resume migration requires distinct full source/actual target Git SHAs")
    if type(world_size) is not int or world_size != 8 or contract["world_size"] != world_size:
        raise ValueError("resume migration requires the original world size 8")
    if contract["target_config"] != target_config:
        raise ValueError("resume migration target config differs from this launch")
    if contract["target_config_path"] != str(Path(target_config_path).resolve(strict=True)):
        raise ValueError("resume migration target config path differs from this launch")
    if Path(__file__).resolve().parents[3] != Path(contract["target_code_root"]).resolve(strict=True):
        raise ValueError("resume migration target code root differs from the executing source")
    require_clean_worktree(Path(contract["source_code_root"]),
                           expected_sha=contract["from_repository_sha"])
    require_clean_worktree(Path(contract["target_code_root"]),
                           expected_sha=contract["to_repository_sha"])
    original = Path(state_path).resolve(strict=True)
    if contract["state_path"] != str(original):
        raise ValueError("resume migration state path differs from the original artifact")
    from mspc.training.training_state import inspect_training_state
    state = inspect_training_state(original, config=contract["source_config"], world_size=world_size)
    if any(state[name] != contract[name] for name in ("run_id", "run_root", "completed_steps")):
        raise ValueError("resume migration run/step binding differs from the original state")
    root = Path(contract["run_root"]).resolve(strict=True)
    if original != root / "training-state" / f"step-{contract['completed_steps']}":
        raise ValueError("resume migration state is outside the original run/update")
    snapshot = json.loads((root / "run-snapshot.json").read_text())
    snapshot = migration_source_snapshot(contract, snapshot)
    if (snapshot["runtime"]["run_id"] != contract["run_id"]
            or Path(snapshot["runtime"]["run_root"]).resolve(strict=True) != root
            or snapshot["repository_sha"] != contract["from_repository_sha"]
            or snapshot["source_config"] != contract["source_config_path"]
            or snapshot["config"] != contract["source_config"]):
        raise ValueError("resume migration source snapshot/SHA/config differs")
    validate_config_migration(contract)
    actual = native_state_layout(original, completed_steps=contract["completed_steps"], world_size=world_size)
    if any(contract[name] != actual[name] for name in actual):
        raise ValueError("resume migration layout differs from the actual source native state")
    return contract


def prepare_resume_migration(*, state_path: str | Path, target_config_path: str | Path,
                              source_code_root: str | Path, target_code_root: str | Path,
                              repository_sha: str | None) -> dict[str, Any]:
    """已发布的目标源码中执行；从实际旧state/snapshot生成本次唯一合同。"""
    from mspc.config import load_experiment_config
    source_root = Path(source_code_root).resolve(strict=True)
    target_root = Path(target_code_root).resolve(strict=True)
    if Path(__file__).resolve().parents[3] != target_root:
        raise ValueError("resume migration target code root differs from the executing source")
    target_sha = require_clean_worktree(target_root, expected_sha=repository_sha)
    original = Path(state_path).resolve(strict=True)
    state = json.loads((original / "state.json").read_text())
    snapshot = json.loads((Path(state["run_root"]) / "run-snapshot.json").read_text())
    require_clean_worktree(source_root, expected_sha=snapshot["repository_sha"])
    target = load_experiment_config(target_config_path)
    contract = {"artifact_type": ARTIFACT_TYPE, "schema_version": 1,
        "from_repository_sha": snapshot["repository_sha"], "to_repository_sha": target_sha,
        **{name: state[name] for name in ("run_id", "run_root", "completed_steps", "world_size")},
        "state_path": str(original), "source_config": state["config"],
        "target_config": target.to_dict(), "source_config_path": snapshot["source_config"],
        "target_config_path": str(target.source_path.resolve(strict=True)),
        "source_code_root": str(source_root), "target_code_root": str(target_root),
        **native_state_layout(original, completed_steps=state["completed_steps"], world_size=state["world_size"])}
    # Reuse the exact launch validator before publication. No placeholder SHA,
    # synthetic native metadata or changes to the old receipt enter this file.
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
        json.dump(contract, stream)
        stream.flush()
        return read_resume_migration(stream.name, state_path=original, target_config=target.to_dict(),
            repository_sha=target_sha, world_size=state["world_size"], target_config_path=target.source_path)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-state", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--source-code-root", type=Path, required=True)
    parser.add_argument("--target-code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = prepare_resume_migration(state_path=args.resume_state, target_config_path=args.target_config,
        source_code_root=args.source_code_root, target_code_root=args.target_code_root,
        repository_sha=os.environ.get("MSPC_REPOSITORY_SHA"))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(contract, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()


def resume_source_config(args: Any) -> dict[str, Any]:
    contract = getattr(args, "resume_migration", None)
    return args.experiment_config if contract is None else contract["source_config"]


def require_model_migration_layout(model: Any, optimizer_groups: list[dict[str, Any]], args: Any) -> None:
    contract = getattr(args, "resume_migration", None)
    if contract is None:
        return
    parameters = dict(model.named_parameters())
    actual = [{"name": name, "shape": list(value.shape),
               "dtype": str(value.dtype).removeprefix("torch.")}
              for name, value in parameters.items()]
    names = {id(value): name for name, value in parameters.items()}
    groups = [[names[id(value)] for value in group["params"]] for group in optimizer_groups]
    if actual != contract["parameter_layout"] or groups != contract["optimizer"]["parameter_groups"]:
        raise ValueError("resume migration target parameter/optimizer layout differs from original native state")
    if contract["optimizer"]["class"] != "torch.optim.AdamW":
        raise ValueError("resume migration requires the original AdamW optimizer")
