#!/usr/bin/env python3
"""Plan or run renderer public metrics on an explicit input manifest.

v1 default stays the frozen five; declared diagnostic columns run only when
``--metric`` names them. v3 default scores the twelve main columns plus the
identity measurability diagnostics declared on the config.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OWNER_PATH = REPOSITORY_ROOT / "src/mspc/evaluation/renderer_qualification_public_metrics.py"


def _load_owner() -> Any:
    name = "mspc.evaluation.renderer_qualification_public_metrics"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, OWNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load public metric module: {OWNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_OWNER = _load_owner()
MetricMetadata = _OWNER.MetricMetadata
PublicMetricCase = _OWNER.PublicMetricCase
run_public_metric_case = _OWNER.run_public_metric_case
validate_metric_metadata = _OWNER.validate_metric_metadata
metric_metadata_from_declarations = _OWNER.metric_metadata_from_declarations
is_v3_main_table = _OWNER.is_v3_main_table
require_v3_identity_diagnostics = _OWNER.require_v3_identity_diagnostics


@dataclass
class Config:
    source_path: Path
    resource_root: Path
    pair_manifest: Path | None
    backend: str
    metrics: tuple[Any, ...]
    # Appendix columns declared by ``config.diagnostic_metrics``.  v1 default
    # scoring stays the five frozen public metrics; v3 default scoring adds
    # the identity measurability denominators.
    diagnostic_metrics: tuple[Any, ...] = ()


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {exc}") from exc


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _path(value: object, label: str, *, base: Path = REPOSITORY_ROOT) -> Path:
    path = Path(_text(value, label)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> Config:
    source = Path(path).expanduser().resolve()
    raw = _load_json(source, "public metrics config")
    if not isinstance(raw, dict):
        raise ValueError("public metrics config must be an object")
    if raw.get("schema_version") != "renderer_qualification_public_metrics_config_v1":
        raise ValueError("public metrics config has an unsupported schema_version")
    metric_rows = raw.get("metrics")
    if not isinstance(metric_rows, list):
        raise ValueError("config.metrics must be an array")
    metrics: list[Any] = []
    for index, row in enumerate(metric_rows):
        if not isinstance(row, dict):
            raise ValueError(f"config.metrics[{index}] must be an object")
        value_range = row.get("range")
        parsed_range = None
        if value_range is not None:
            if not isinstance(value_range, list) or len(value_range) != 2:
                raise ValueError(f"config.metrics[{index}].range must be null or two numbers")
            parsed_range = (float(value_range[0]), float(value_range[1]))
        allow_na = row.get("allow_not_applicable", False)
        if type(allow_na) is not bool:
            raise ValueError(f"config.metrics[{index}].allow_not_applicable must be boolean")
        metrics.append(MetricMetadata(
            metric_id=_text(row.get("metric_id"), f"config.metrics[{index}].metric_id"),
            direction=_text(row.get("direction"), f"config.metrics[{index}].direction"),
            unit=_text(row.get("unit"), f"config.metrics[{index}].unit"),
            value_range=parsed_range,
            allow_not_applicable=allow_na,
        ))
    parsed_metrics = validate_metric_metadata(metrics)
    diagnostic_metrics: tuple[Any, ...] = ()
    v3_profile = is_v3_main_table(parsed_metrics)
    if v3_profile and "diagnostic_metrics" not in raw:
        raise ValueError("v3 public metrics config must declare diagnostic_metrics")
    if "diagnostic_metrics" in raw:
        # Validating the main table plus the declarations as one column set
        # reuses the owner check that the frozen prefix stays first and in
        # order, that every extra column is a declared diagnostic ID, and that
        # no ID repeats.
        declared = validate_metric_metadata(
            parsed_metrics
            + metric_metadata_from_declarations(
                raw["diagnostic_metrics"], label="config.diagnostic_metrics"
            )
        )
        diagnostic_metrics = declared[len(parsed_metrics) :]
        if v3_profile:
            require_v3_identity_diagnostics(
                diagnostic_metrics, label="config.diagnostic_metrics"
            )
    pair_value = raw.get("pair_manifest")
    return Config(
        source_path=source,
        resource_root=_path(raw.get("resource_root"), "config.resource_root"),
        pair_manifest=(
            _path(pair_value, "config.pair_manifest") if pair_value is not None else None
        ),
        backend=_text(raw.get("backend"), "config.backend"),
        metrics=parsed_metrics,
        diagnostic_metrics=diagnostic_metrics,
    )


def build_plan(
    config: Config,
    *,
    metric_ids: Sequence[str] = (),
    backend_spec: str | None = None,
    resource_root: Path | None = None,
    pair_manifest: Path | None = None,
) -> dict[str, Any]:
    """预览 ``--execution-mode run`` 会跑什么。

    **三个 CLI 覆盖值必须传进来。** 它们在 ``run`` 路径上覆盖 config 的同名字段，
    而此前 plan 只读 config，于是 ``--pair-manifest`` 传了也预览成 ``null``、
    ``--backend`` 与 ``--resource-root`` 同理。``AGENTS.md`` 把干跑列为主服务器交付前置门，
    预览错的东西等于这道门失效——2026-08-31 的两次主表打分都因此绕开了 plan，
    改用只读探针顶替。

    回显的是**生效值**（CLI 覆盖 config），与 ``run`` 用的是同一条优先级。
    """

    metrics = _select_metrics(config, metric_ids)
    effective_backend = backend_spec if backend_spec is not None else config.backend
    effective_resource_root = (
        resource_root if resource_root is not None else config.resource_root
    )
    effective_pair_manifest = (
        pair_manifest if pair_manifest is not None else config.pair_manifest
    )
    plan: dict[str, Any] = {
        "status": "ready",
        "metric_columns": [
            {
                "metric_id": metric.metric_id,
                "direction": metric.direction,
                "unit": metric.unit,
            }
            for metric in metrics
        ],
        "input": "required for run via --input",
        "resource_root": str(effective_resource_root),
        "pair_manifest": str(effective_pair_manifest) if effective_pair_manifest else None,
        "backend": effective_backend,
    }
    if not metric_ids and is_v3_main_table(config.metrics):
        plan["public"] = [metric.metric_id for metric in config.metrics]
        plan["diagnostic"] = [metric.metric_id for metric in config.diagnostic_metrics]
    return plan


def _select_metrics(config: Config, metric_ids: Sequence[str]) -> tuple[Any, ...]:
    # v1 default stays the frozen five so historical main-table numbers and the
    # sidecar do not pick up diagnostic scoring cost by accident.  v3 default
    # scores the twelve columns plus the identity denominators declared on the
    # config.  Explicit ``--metric`` still names an optional subset.
    if not metric_ids:
        if is_v3_main_table(config.metrics):
            return validate_metric_metadata(
                tuple(config.metrics) + tuple(config.diagnostic_metrics)
            )
        return tuple(config.metrics)
    available = tuple(config.metrics) + tuple(config.diagnostic_metrics)
    requested = set(metric_ids)
    unknown = requested - {metric.metric_id for metric in available}
    if unknown:
        raise ValueError(f"unknown metric selectors: {sorted(unknown)}")
    selected = tuple(metric for metric in available if metric.metric_id in requested)
    return validate_metric_metadata(selected, allow_subset=True)


def _character_queries(row: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Open-vocabulary detector queries for this story, or None when undeclared.

    A row that never declares the field is not the same as a row declaring an
    empty list: the first is a scoring input the benchmark join never ran on,
    the second is a story the benchmark says has no character asset.  The
    backend refuses the first and reports the second as ineligible.
    """

    if "character_queries" not in row:
        return None
    raw = row["character_queries"]
    if not isinstance(raw, list):
        raise ValueError("input.character_queries must be an array")
    return tuple(
        _text(value, f"input.character_queries[{index}]")
        for index, value in enumerate(raw)
    )


def _load_input_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = _load_json(path, "public metrics input")
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("cases"), list):
            rows = payload["cases"]
        else:
            raise ValueError("input must be a JSON array, a {cases: [...]} object, or JSONL")
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("input must contain at least one case object")
    return rows


def _backend(spec: str) -> Callable[..., Any]:
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        raise ValueError("backend must use module:callable syntax")
    target = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(target):
        raise ValueError(f"backend target is not callable: {spec}")
    return target


def _case_from_row(
    row: Mapping[str, Any],
    *,
    input_dir: Path,
    resource_root: Path,
    pair_manifest: Path | None,
    metric_config_path: Path | None = None,
) -> Any:
    video_path = _path(row.get("video_path"), "input.video_path", base=input_dir)
    shots = row.get("shots")
    if not isinstance(shots, list):
        raise ValueError("input.shots must be an array")
    continuity = row.get("continuity", {})
    if not isinstance(continuity, dict):
        raise ValueError("input.continuity must be an object")
    arm_id = row.get("arm_id")
    if arm_id is not None:
        arm_id = _text(arm_id, "input.arm_id")
    result_metadata: dict[str, Any] = {}
    for field in ("story_id", "split"):
        if field in row:
            result_metadata[field] = _text(row[field], f"input.{field}")
    if "generation_seed" in row:
        generation_seed = row["generation_seed"]
        if type(generation_seed) is not int:
            raise ValueError("input.generation_seed must be an integer")
        result_metadata["generation_seed"] = generation_seed
    for field in ("shot_count", "K"):
        if field in row:
            value = row[field]
            if type(value) is not int or value <= 0:
                raise ValueError(f"input.{field} must be a positive integer")
            result_metadata[field] = value
    return PublicMetricCase(
        case_id=_text(row.get("case_id"), "input.case_id"),
        arm_id=arm_id,
        video_path=video_path,
        story_prompt=_text(row.get("story_prompt"), "input.story_prompt"),
        shots=tuple(shots),
        continuity=continuity,
        resource_root=resource_root,
        pair_manifest_path=pair_manifest,
        subject_class=_text(row.get("subject_class", "person"), "input.subject_class"),
        character_queries=_character_queries(row),
        result_metadata=result_metadata,
        metric_config_path=metric_config_path,
    )


def run(
    config: Config,
    *,
    input_path: Path,
    backend_spec: str | None = None,
    resource_root: Path | None = None,
    pair_manifest: Path | None = None,
    metric_ids: Sequence[str] = (),
) -> dict[str, Any]:
    input_path = input_path.expanduser().resolve()
    rows = _load_input_rows(input_path)
    metrics = _select_metrics(config, metric_ids)
    metric_backend = _backend(backend_spec or config.backend)
    # Engines are built lazily inside the loop below, so an environment problem
    # that only TensorFlow can see -- the LD_LIBRARY_PATH one that leaves it with
    # 0 GPUs -- would otherwise surface after every video has been decoded, and
    # after every video has been *generated* when the sidecar drives this. Ask
    # the backend to refuse now instead. Backends without the hook (test doubles)
    # simply skip it.
    resources = (resource_root or config.resource_root).expanduser().resolve()
    preflight = getattr(metric_backend, "preflight", None)
    if callable(preflight):
        preflight([metric.metric_id for metric in metrics], resource_root=resources)
    pairs = pair_manifest if pair_manifest is not None else config.pair_manifest
    if pairs is not None:
        pairs = pairs.expanduser().resolve()
    results = [
        run_public_metric_case(
            _case_from_row(
                row,
                input_dir=input_path.parent,
                resource_root=resources,
                pair_manifest=pairs,
                metric_config_path=config.source_path,
            ),
            metrics,
            backend=metric_backend,
        )
        for row in rows
    ]
    return {
        "status": "completed",
        "metric_columns": [metric.metric_id for metric in metrics],
        "case_count": len(results),
        "results": results,
    }


def execute(
    config_path: str | Path,
    *,
    execution_mode: str = "plan",
    input_path: Path | None = None,
    backend_spec: str | None = None,
    resource_root: Path | None = None,
    pair_manifest: Path | None = None,
    metric_ids: Sequence[str] = (),
) -> dict[str, Any]:
    config = load_config(config_path)
    if execution_mode == "plan":
        return build_plan(
            config,
            metric_ids=metric_ids,
            backend_spec=backend_spec,
            resource_root=resource_root,
            pair_manifest=pair_manifest,
        )
    if execution_mode != "run":
        raise ValueError("execution_mode must be plan or run")
    if input_path is None:
        raise ValueError("run requires --input")
    return run(
        config,
        input_path=input_path,
        backend_spec=backend_spec,
        resource_root=resource_root,
        pair_manifest=pair_manifest,
        metric_ids=metric_ids,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execution-mode", choices=("plan", "run"), default="plan")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--backend")
    parser.add_argument("--resource-root", type=Path)
    parser.add_argument("--pair-manifest", type=Path)
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = execute(
        args.config,
        execution_mode=args.execution_mode,
        input_path=args.input,
        backend_spec=args.backend,
        resource_root=args.resource_root,
        pair_manifest=args.pair_manifest,
        metric_ids=args.metric,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
