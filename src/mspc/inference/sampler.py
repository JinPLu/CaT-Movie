"""Shared multishot CSV inference orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from mspc.config import ExperimentConfig
from mspc.data.multishot import read_video_metadata
from mspc.inference.common import (
    build_inference_dataset_config,
    build_inference_frame_planner,
    build_multishot_case_inputs,
    generated_eval_manifest_row,
    infer_case_id,
)


class CaseGenerator(Protocol):
    def __call__(
        self,
        *,
        case_id: str,
        case_inputs: dict[str, Any],
        output_path: Path,
        seed: int,
        source_row: Any,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class MultishotInferenceSampler:
    """Iterate multishot CSV cases and write a generated eval manifest."""

    config: ExperimentConfig
    require_case_id_column: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.config, ExperimentConfig):
            raise TypeError("config must be an ExperimentConfig")
        inference = self.config.resolved["inference"]
        if not inference["enabled"]:
            raise ValueError("config inference.enabled must be true")

    @property
    def inference_cfg(self) -> dict[str, Any]:
        return dict(self.config.resolved["inference"])

    @property
    def dataset_cfg(self) -> dict[str, Any]:
        return build_inference_dataset_config(
            self.config,
            require_case_id_column=self.require_case_id_column,
        )

    @property
    def frame_planner(self) -> Any:
        return build_inference_frame_planner(
            self.dataset_cfg,
            inference_fps=self.inference_cfg["fps"],
        )

    @property
    def output_dir(self) -> Path:
        return Path(self.inference_cfg["output_dir"])

    @property
    def test_csv_path(self) -> str:
        return str(self.inference_cfg["test_csv_path"])

    def prepare_output_dir(self) -> None:
        # The train-config schema has no root-level collision_policy. Keep the
        # historical fail-if-present behavior unless a legacy caller supplies
        # that field explicitly.
        collision_policy = self.config.resolved.get("collision_policy", "error")
        if self.output_dir.exists() and collision_policy == "error":
            raise FileExistsError(
                f"inference output directory already exists: {self.output_dir}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=False)

    def iter_cases(self) -> list[tuple[str, dict[str, Any], Any]]:
        import pandas as pd

        cases: list[tuple[str, dict[str, Any], Any]] = []
        for index, row in pd.read_csv(self.test_csv_path).iterrows():
            case_id = infer_case_id(
                row,
                row_index=index,
                dataset_cfg=self.dataset_cfg,
                source_label=self.test_csv_path,
            )
            case_inputs = build_multishot_case_inputs(
                row,
                self.frame_planner,
                video_column=self.dataset_cfg["video_column"],
                caption_column=self.dataset_cfg["caption_column"],
                shot_groups_column=self.dataset_cfg["shot_groups_column"],
                schema=self.dataset_cfg["schema"],
                row_label=case_id,
                prompt_format=self.dataset_cfg["prompt_format"],
                shot2story_prompt_source=self.dataset_cfg.get(
                    "shot2story_prompt_source"
                ),
                read_video_metadata_fn=read_video_metadata,
            )
            cases.append((case_id, case_inputs, row))
        return cases

    def run(self, generate_case: CaseGenerator) -> list[dict[str, Any]]:
        self.prepare_output_dir()
        manifest: list[dict[str, Any]] = []
        base_seed = int(self.inference_cfg["seed"])
        for index, (case_id, case_inputs, row) in enumerate(self.iter_cases()):
            output_path = self.output_dir / f"{case_id}.mp4"
            manifest_row = generate_case(
                case_id=case_id,
                case_inputs=case_inputs,
                output_path=output_path,
                seed=base_seed + index,
                source_row=row,
            )
            manifest.append(manifest_row)
        self.write_manifest(manifest)
        return manifest

    def write_manifest(self, manifest: list[dict[str, Any]]) -> Path:
        manifest_path = self.output_dir / "generated_eval_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest_path


def default_manifest_row(
    *,
    case_id: str,
    output_path: Path,
    generation_paradigm: str,
    case_inputs: dict[str, Any],
    source_row: Any,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = generated_eval_manifest_row(
        case_id=case_id,
        output_path=output_path,
        generation_paradigm=generation_paradigm,
        case_inputs=case_inputs,
        source_row=source_row,
    )
    if extra_fields:
        row.update(extra_fields)
    return row
