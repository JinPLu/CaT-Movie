"""Close the Wan2.2 smoke gate after a successful training subprocess."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from mspc.config import ExperimentConfig, load_experiment_config
from mspc.inference.contact_sheet import shot_group_sample_indices
from mspc.runs import LaunchMode, resolve_launch


TRAINING_OUTPUT_CHECKPOINT = "training_output_checkpoint"
STEP_CHECKPOINT_PATTERN = re.compile(r"^step-(?P<step>[1-9][0-9]*)\.safetensors$")
SMOKE_GATE_STATUS_VISUAL_REVIEW_PENDING = "technical_passed_visual_review_pending"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_int(value: Any, label: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_shot_groups(value: Any, *, expected_frames: int) -> list[list[int]]:
    if not isinstance(value, list) or not value:
        raise ValueError("generated manifest shot_groups must be a non-empty list")
    normalized: list[list[int]] = []
    cursor = 0
    for index, group in enumerate(value):
        if (
            not isinstance(group, list)
            or len(group) != 2
            or any(type(item) is not int for item in group)
        ):
            raise ValueError(
                f"generated manifest shot_groups[{index}] must be [start, end] integers"
            )
        start, end = group
        if start != cursor or end <= start:
            raise ValueError(
                "generated manifest shot_groups must be contiguous, ordered, and non-empty"
            )
        normalized.append([start, end])
        cursor = end
    if cursor != expected_frames:
        raise ValueError(
            "generated manifest shot_groups do not cover train_rgb_frames exactly"
        )
    return normalized


def _load_generated_video_contract(
    inference_output: Path,
) -> tuple[Path, dict[str, Any], Path]:
    manifest_path = (inference_output / "generated_eval_manifest.json").resolve(
        strict=True
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or len(manifest) != 1:
        raise ValueError("Wan2.2 smoke generated manifest must contain exactly one row")
    row = manifest[0]
    if not isinstance(row, dict):
        raise ValueError("Wan2.2 smoke generated manifest row must be an object")
    video_path_value = row.get("video_path")
    if not isinstance(video_path_value, str) or not video_path_value:
        raise ValueError("Wan2.2 smoke generated manifest row has no video_path")
    video_path = Path(video_path_value).resolve(strict=True)
    return manifest_path, row, video_path


def _write_contact_sheet(
    frames: dict[int, Any],
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    if not frames:
        raise ValueError("cannot render a smoke contact sheet without decoded frames")
    images: list[tuple[int, Any]] = []
    for frame_index, frame in sorted(frames.items()):
        image = Image.fromarray(frame).convert("RGB")
        target_width = min(image.width, 320)
        target_height = max(1, round(image.height * target_width / image.width))
        if (target_width, target_height) != image.size:
            image = image.resize(
                (target_width, target_height),
                Image.Resampling.LANCZOS,
            )
        images.append((frame_index, image))
    columns = min(3, len(images))
    rows = (len(images) + columns - 1) // columns
    cell_width = max(image.width for _, image in images)
    cell_height = max(image.height for _, image in images)
    label_height = 24
    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * (cell_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for position, (frame_index, image) in enumerate(images):
        column = position % columns
        row = position // columns
        left = column * cell_width
        top = row * (cell_height + label_height)
        draw.text((left + 5, top + 4), f"frame {frame_index}", fill="black")
        sheet.paste(image, (left, top + label_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _inspect_generated_video(
    video_path: Path,
    *,
    expected_frames: int,
    expected_width: int,
    expected_height: int,
    expected_fps: float,
    shot_groups: Sequence[Sequence[int]],
    contact_sheet_path: Path,
) -> dict[str, Any]:
    import imageio
    import numpy as np

    sample_indices = set(shot_group_sample_indices(shot_groups))
    sampled_frames: dict[int, Any] = {}
    decoded_frames = 0
    reader: Any | None = None
    try:
        reader = imageio.get_reader(str(video_path), "ffmpeg")
        metadata = reader.get_meta_data()
        size = metadata["size"]
        observed_fps = float(metadata["fps"])
        observed_width, observed_height = int(size[0]), int(size[1])
        for frame_index, frame in enumerate(reader):
            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(
                    f"decoded frame {frame_index} must have RGB shape, got {array.shape}"
                )
            if tuple(array.shape[:2]) != (observed_height, observed_width):
                raise ValueError(
                    f"decoded frame {frame_index} has inconsistent shape {array.shape}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"decoded frame {frame_index} contains non-finite values")
            if frame_index in sample_indices:
                sampled_frames[frame_index] = array.copy()
            decoded_frames += 1
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"imageio-ffmpeg could not fully validate the generated video: {exc}"
        ) from exc
    finally:
        if reader is not None:
            reader.close()

    if decoded_frames != expected_frames:
        raise ValueError(
            "generated video frame count mismatch: "
            f"expected {expected_frames}, got {decoded_frames}"
        )
    if (observed_width, observed_height) != (expected_width, expected_height):
        raise ValueError(
            "generated video resolution mismatch: "
            f"expected {expected_width}x{expected_height}, "
            f"got {observed_width}x{observed_height}"
        )
    if abs(observed_fps - expected_fps) > 0.05:
        raise ValueError(
            f"generated video fps mismatch: expected {expected_fps}, got {observed_fps}"
        )
    if set(sampled_frames) != sample_indices:
        raise ValueError("generated video did not provide every contact-sheet frame")
    _write_contact_sheet(sampled_frames, contact_sheet_path)
    return {
        "status": "passed",
        "decoder": "imageio-ffmpeg/full-frame-decode",
        "video": {
            "path": str(video_path),
            "bytes": video_path.stat().st_size,
        },
        "decoded_frames": decoded_frames,
        "width": observed_width,
        "height": observed_height,
        "fps": observed_fps,
        "contact_sheet": {
            "path": str(contact_sheet_path),
            "sampled_frame_indices": sorted(sample_indices),
        },
    }


def require_completed_run_root(run_root: Path) -> Path:
    """确认这一条臂真的跑完了，再让 smoke 去读它的权重。"""

    resolved = run_root.resolve()
    snapshot_path = resolved / "run-snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot.get("status") != "completed":
        raise RuntimeError(f"training run did not complete before smoke inference: {resolved}")
    return resolved


def resolve_new_run_root(output_root: Path, existing: set[Path]) -> Path:
    """全新臂与热启动臂：run root 是这次新建出来的那一个，靠差集认。"""

    created = sorted(
        path.resolve()
        for path in output_root.iterdir()
        if path.is_dir() and path.resolve() not in existing
    )
    if len(created) != 1:
        raise RuntimeError(
            "Wan2.2 smoke gate requires exactly one newly completed run root, "
            f"found {len(created)}"
        )
    return require_completed_run_root(created[0])


def _smoke_source_dataset(config: ExperimentConfig) -> dict[str, Any]:
    dataset = dict(config.resolved["dataset"])
    if dataset["type"] != "multishot_csv":
        raise ValueError("Wan2.2 smoke case selection requires a multishot CSV source")
    return dataset


def _case_id_column(config: ExperimentConfig) -> str:
    column = _smoke_source_dataset(config).get("case_id_column") or "case_id"
    if not isinstance(column, str) or not column:
        raise ValueError("Wan2.2 smoke case selection requires a case ID column")
    return column


def _require_case_id(row: pd.Series, *, column: str, label: str) -> str:
    value = row.get(column)
    if pd.isna(value):
        raise ValueError(f"{label} has no selected case ID in column {column!r}")
    case_id = str(value).strip()
    if not case_id:
        raise ValueError(f"{label} has no selected case ID in column {column!r}")
    return case_id


def _read_single_csv_row(path: Path, *, column: str) -> tuple[pd.DataFrame, str, int]:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(
            "Wan2.2 smoke exact CSV selection requires exactly one row: "
            f"{path} has {len(frame)}"
        )
    if column not in frame.columns:
        raise ValueError(
            f"Wan2.2 smoke exact CSV selection requires case ID column {column!r}: {path}"
        )
    row_index = int(frame.index[0])
    case_id = _require_case_id(frame.iloc[0], column=column, label=str(path))
    return frame, case_id, row_index


def _select_csv_row_by_case_id(
    path: Path,
    *,
    column: str,
    case_id: str,
) -> tuple[pd.DataFrame, int]:
    frame = pd.read_csv(path)
    if column not in frame.columns:
        raise ValueError(
            f"Wan2.2 smoke case_id selection requires column {column!r}: {path}"
        )
    normalized = frame[column].map(
        lambda value: "" if pd.isna(value) else str(value).strip()
    )
    matches = frame.loc[normalized == case_id]
    if len(matches) != 1:
        raise ValueError(
            "Wan2.2 smoke case_id selection requires exactly one match for "
            f"{case_id!r} in {path}: found {len(matches)}"
        )
    return matches.reset_index(drop=True), int(matches.index[0])


def _build_one_case_csv(
    config: ExperimentConfig,
    run_root: Path,
) -> tuple[Path, dict[str, Any]]:
    inference = config.resolved["inference"]
    requested_case_id = inference.get("smoke_case_id")
    exact_csv_path_value = inference.get("test_csv_path")
    if requested_case_id is None and exact_csv_path_value is None:
        raise ValueError(
            "Wan2.2 smoke gate requires explicit inference.smoke_case_id or "
            "an exact one-row inference.test_csv_path; refusing implicit first-row fallback"
        )
    if requested_case_id is not None and exact_csv_path_value is not None:
        raise ValueError(
            "Wan2.2 smoke gate requires exactly one source selector: "
            "inference.smoke_case_id or inference.test_csv_path"
        )
    column = _case_id_column(config)
    if requested_case_id is None:
        metadata_path = Path(exact_csv_path_value)
        frame, selected_case_id, row_index = _read_single_csv_row(
            metadata_path,
            column=column,
        )
        mode = "exact_csv_path"
    else:
        selected_case_id = str(requested_case_id)
        metadata_path = Path(_smoke_source_dataset(config)["metadata_path"])
        frame, row_index = _select_csv_row_by_case_id(
            metadata_path,
            column=column,
            case_id=selected_case_id,
        )
        mode = "case_id"
    frame = frame.copy()
    frame["_mspc_smoke_selected_case_id"] = selected_case_id
    frame["_mspc_smoke_source_mode"] = mode
    frame["_mspc_smoke_source_csv_path"] = str(metadata_path)
    output = run_root / "smoke-inference-case.csv"
    frame.to_csv(output, index=False)
    return output, {
        "mode": mode,
        "selected_case_id": selected_case_id,
        "case_id_column": column,
        "source_row_index": row_index,
        "source_csv": {
            "path": str(metadata_path),
        },
        "single_case_csv": {
            "path": str(output),
        },
    }


def _resolve_training_checkpoint(
    config: ExperimentConfig,
    run_root: Path,
) -> tuple[int, Path]:
    checkpoint_dir = run_root / "checkpoints"
    candidates: list[tuple[int, Path]] = []
    if checkpoint_dir.is_dir():
        for path in checkpoint_dir.iterdir():
            match = STEP_CHECKPOINT_PATTERN.fullmatch(path.name)
            if match is not None and path.is_file():
                candidates.append((int(match.group("step")), path))
    if not candidates:
        raise FileNotFoundError(
            f"Wan2.2 training output contains no step checkpoint: {run_root}"
        )
    step, checkpoint = max(candidates, key=lambda item: item[0])
    configured_step = config.resolved["training"]["max_train_steps"]
    if configured_step is not None and step != configured_step:
        raise ValueError(
            "Wan2.2 final checkpoint step does not match training.max_train_steps: "
            f"checkpoint={step}, configured={configured_step}"
        )
    return step, checkpoint


def run_training_output_smoke_gate(config: ExperimentConfig, run_root: Path) -> dict[str, Any]:
    """Strict-reload one checkpoint and leave its generated video pending review."""

    inference = config.resolved["inference"]
    if not inference["enabled"] or inference["weight_source"] != TRAINING_OUTPUT_CHECKPOINT:
        raise ValueError("config does not request the Wan2.2 training-output smoke gate")
    step, checkpoint = _resolve_training_checkpoint(
        config,
        run_root,
    )

    one_case_csv, case_selection = _build_one_case_csv(config, run_root)
    inference_output = run_root / "smoke-inference"
    concrete = config.to_dict()
    concrete["inference"].update(
        {
            "enabled": True,
            "test_csv_path": str(one_case_csv),
            "output_dir": str(inference_output),
            "smoke_case_id": case_selection["selected_case_id"],
            "weight_source": "full_dit_checkpoint",
            "dit_checkpoint": str(checkpoint),
        }
    )
    concrete_config_path = run_root / "smoke-inference-config.json"
    _write_json(concrete_config_path, concrete)
    concrete_config = load_experiment_config(concrete_config_path)
    launch = resolve_launch(concrete_config, mode=LaunchMode.INFER)
    completed = subprocess.run(
        launch.argv,
        env={**os.environ, **launch.environment},
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Wan2.2 smoke inference failed with return code {completed.returncode}"
        )
    manifest_path, manifest_row, video_path = _load_generated_video_contract(
        inference_output,
    )
    expected_frames = _require_int(
        manifest_row.get("train_rgb_frames"),
        "generated manifest train_rgb_frames",
    )
    shot_groups = _require_shot_groups(
        manifest_row.get("shot_groups"),
        expected_frames=expected_frames,
    )
    video_integrity = _inspect_generated_video(
        video_path,
        expected_frames=expected_frames,
        expected_width=_require_int(
            inference["target_width"],
            "inference.target_width",
        ),
        expected_height=_require_int(
            inference["target_height"],
            "inference.target_height",
        ),
        expected_fps=float(inference["fps"]),
        shot_groups=shot_groups,
        contact_sheet_path=inference_output / "smoke-contact-sheet.png",
    )
    integrity_path = inference_output / "video-integrity.json"
    _write_json(integrity_path, video_integrity)
    visual_review_request = {
        "artifact_type": "wan22_training_smoke_visual_review",
        "schema_version": 1,
        "status": "pending",
        "required_decision": "accept_or_reject_after_human_contact_sheet_review",
        "video": video_integrity["video"],
        "contact_sheet": video_integrity["contact_sheet"],
        "technical_integrity": {
            "path": str(integrity_path),
        },
        "reviewer": None,
        "decision": None,
        "reviewed_at": None,
        "notes": None,
    }
    visual_review_request_path = inference_output / "visual-review.required.json"
    _write_json(visual_review_request_path, visual_review_request)
    result = {
        "status": SMOKE_GATE_STATUS_VISUAL_REVIEW_PENDING,
        "training_step": step,
        "conditioning": "pure_t2v",
        "temporal_position_mode": config.resolved["model"]["temporal_position_mode"],
        "checkpoint": {
            "path": str(checkpoint),
        },
        "case_selection": case_selection,
        "one_case_csv": {"path": str(one_case_csv)},
        "inference_config": {
            "path": str(concrete_config_path),
        },
        "generated_manifest": {
            "path": str(manifest_path),
        },
        "video_integrity": {
            "path": str(integrity_path),
            "status": video_integrity["status"],
        },
        "visual_review": {
            "required": True,
            "status": "pending",
            "request_path": str(visual_review_request_path),
            "contact_sheet": video_integrity["contact_sheet"],
        },
    }
    return result
