"""Adapters for the five renderer public metrics using an explicit resource root."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from mspc.evaluation.renderer_qualification_public_metrics import (
    PairManifestError,
    DIAGNOSTIC_METRIC_IDS,
    PUBLIC_METRIC_IDS,
    PUBLIC_METRIC_IDS_V2,
    PUBLIC_METRIC_IDS_V3,
    V2_TRANSITION_METRIC_IDS,
    V3_IDENTITY_DIAGNOSTIC_METRIC_IDS,
    V3_QUALITY_METRIC_IDS,
    MetricMetadata,
    MetricResult,
    ModelBatchSizes,
    PublicMetricCase,
    first_mid_last_frame_indices,
    matched_transition_timing_mae,
    parse_model_batch_sizes,
    planned_shot_boundaries,
    transition_count_error,
    transition_deviation,
    transition_run_peaks,
    cut_boundaries_from_peaks,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
# The current profile owns the default detector contract.  v1 and v2 configs
# still describe YOLO conf/iou fields that no longer exist, so they no longer
# load: their published numbers stay readable in their result files and are not
# reproducible in this checkout.
DEFAULT_ROI_DETECTOR_CONFIG = (
    REPO_ROOT / "configs/benchmark/renderer_qualification/public_metrics_v3.json"
)
TRANSNET_WORKER = REPO_ROOT / "tools/run/transnet_worker.py"
TRANSNET_PYTHON_ENV = "MSPC_TRANSNET_PYTHON"
TRANSNET_ROLE = "renderer_transnet"
QUALITY_WORKER = REPO_ROOT / "tools/run/quality_worker.py"
QUALITY_PYTHON_ENV = "MSPC_QUALITY_PYTHON"
QUALITY_ROLE = "renderer_quality"
VBENCH_PYTHON_ENV = "MSPC_VBENCH_PYTHON"
VBENCH_ROLE = "renderer_vbench"
_ENV_REGISTRY_SCHEMA = "mspc_env_by_arch_v1"

_IMAGENET_MEAN_RGB = (124, 116, 104)
_VIDEOSCORE_ASPECTS = (
    "visual_quality",
    "temporal_consistency",
    "dynamic_degree",
    "text_to_video_alignment",
    "factual_consistency",
)
_VIDEOSCORE_ASPECT_METRICS = {
    f"video_score_{aspect}": aspect for aspect in _VIDEOSCORE_ASPECTS
}
# ROI metrics whose detector settings are frozen in code, not in configuration:
# the main table must stay reproducible no matter how the config file changes.
# The v2 gated identity column: one subject class as the detector query, and a
# shot pair scores on whichever keyframe comparisons kept both ROIs.
_GATED_CHARACTER_METRIC_IDS = ("dinov2_character_consistency_gated",)
# The v3 identity column plus its two measurability denominators and the
# per-case scored indicator.  All four read one open-vocabulary ROI pass.
_IDENTITY_METRIC_IDS = (
    "dinov2_character_identity",
    *V3_IDENTITY_DIAGNOSTIC_METRIC_IDS,
)
_FROZEN_ROI_METRIC_IDS = (
    "dinov2_character_consistency",
    "dinov2_scene_consistency",
    *_IDENTITY_METRIC_IDS,
)
# Metric ids whose engine is TensorFlow rather than torch.  Only these can hit
# the LD_LIBRARY_PATH hazard, so only these are worth a preflight; the list is
# held to the dispatch by a test that asserts no other supported metric id
# constructs the TransNet engine.
TENSORFLOW_METRIC_IDS = (
    "transnet_transition_deviation",
    *V2_TRANSITION_METRIC_IDS,
)
_ROI_DIAGNOSTIC_METRIC_IDS = _GATED_CHARACTER_METRIC_IDS
# Columns this backend does not own.  The judge line computes them and writes
# them into the same result rows; naming them here keeps them out of "unknown
# metric" and out of a silent zero.
_JUDGE_METRIC_IDS = frozenset(PUBLIC_METRIC_IDS_V3) - frozenset(
    (*V3_QUALITY_METRIC_IDS, "viclip_text_alignment", "dinov2_character_identity",
     "dinov2_scene_consistency", *V2_TRANSITION_METRIC_IDS)
)
_SUPPORTED_METRIC_IDS = (
    (
        frozenset(PUBLIC_METRIC_IDS)
        | frozenset(DIAGNOSTIC_METRIC_IDS)
        | frozenset(PUBLIC_METRIC_IDS_V2)
        | frozenset(PUBLIC_METRIC_IDS_V3)
    )
    - _JUDGE_METRIC_IDS
)
_VIDEOSCORE_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the following frames of a given video and see the text prompt for generating the video,
then give scores from 5 different dimensions:
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) temporal consistency, both the consistency of objects or humans and the smoothness of motion or movements
(3) dynamic degree, the degree of dynamic changes
(4) text-to-video alignment, the alignment between the text prompt and the video content
(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge
for each dimension, output a float number from 1.0 to 4.0,
the higher the number is, the better the video performs in that sub-score,
the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the video is like a real video)
Here is an output example:
visual quality: 3.2
temporal consistency: 2.7
dynamic degree: 4.0
text-to-video alignment: 2.3
factual consistency: 1.8

For this video, the text prompt is "{text_prompt}",
all the frames of video are as follows:
"""


@dataclass(frozen=True)
class RoiDetectorProfile:
    """One explicit Grounding DINO / SAM setting bundle for the subject-ROI pipeline."""

    name: str
    box_threshold: float
    text_threshold: float
    min_mask_area_ratio: float | None

    def payload(self) -> dict[str, Any]:
        return {
            "profile": self.name,
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
            "min_mask_area_ratio": self.min_mask_area_ratio,
        }


# The ROI columns never read the config for their detector settings: they live
# in code so no configuration edit can move a published number.  The config
# restates the same numbers under ``frozen_public_metric_profile`` and the
# loader fails on any drift.
#
# From v3 this constant describes Grounding DINO, not YOLO11x.  The historical
# v1/v2 gated ROI columns were produced at YOLO conf 0.25 / iou 0.7, and that
# detector is gone from this checkout: those published numbers stay READABLE in
# their result files but are NOT REPRODUCIBLE here.  That is the real price of
# the detector swap; the drift check below therefore guards the v3 contract and
# any historical config that still names the YOLO fields will fail to load,
# loudly, instead of being re-scored under a different detector.
#
# 0.25 / 0.20 are this project's PRE-REGISTERED thresholds.  They are not
# copied from any external benchmark: the repository carries no source for a
# 0.25/0.20 pair, and the nearest anchor
# (the Grounded-SAM-2 demo at BOX 0.25 / TEXT 0.25) agrees on box and not on text.
FROZEN_PUBLIC_ROI_DETECTOR = RoiDetectorProfile(
    name="public_metrics_v3_gdino",
    box_threshold=0.25,
    text_threshold=0.20,
    min_mask_area_ratio=None,
)


@dataclass(frozen=True)
class RoiDetectorContract:
    """Detector profiles for the diagnostic ROI columns, read from the config."""

    source: Path
    profiles: Mapping[str, RoiDetectorProfile]
    metric_profiles: Mapping[str, str]

    def profile_for(self, metric_id: str) -> RoiDetectorProfile:
        if metric_id in _FROZEN_ROI_METRIC_IDS:
            return FROZEN_PUBLIC_ROI_DETECTOR
        name = self.metric_profiles.get(metric_id)
        if name is None:
            raise ValueError(
                f"roi_detector.metric_profiles is missing {metric_id!r} in {self.source}"
            )
        return self.profiles[name]


def _required(mapping: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{label}.{key} is required")
    return mapping[key]


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _ratio(value: object, label: str, *, allow_null: bool = False) -> float | None:
    if value is None:
        if not allow_null:
            raise ValueError(f"{label} must be a number")
        return None
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not 0.0 < number <= 1.0:
        raise ValueError(f"{label} must be inside (0, 1]")
    return number


def load_roi_detector_contract(config_path: Path) -> RoiDetectorContract:
    """Read the required ``roi_detector`` block; never fall back to defaults."""

    source = Path(config_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"ROI detector config does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"public metrics config must be an object: {source}")
    block = _required(payload, "roi_detector", "config")
    if not isinstance(block, dict):
        raise ValueError("config.roi_detector must be an object")
    raw_profiles = _required(block, "profiles", "config.roi_detector")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ValueError("config.roi_detector.profiles must be a non-empty object")
    profiles: dict[str, RoiDetectorProfile] = {}
    for name, row in raw_profiles.items():
        label = f"config.roi_detector.profiles[{name!r}]"
        if not isinstance(name, str) or not name.strip():
            raise ValueError("config.roi_detector.profiles keys must be names")
        if not isinstance(row, dict) or set(row) != {
            "box_threshold",
            "text_threshold",
            "min_mask_area_ratio",
        }:
            raise ValueError(
                f"{label} must declare box_threshold, text_threshold, and min_mask_area_ratio"
            )
        profiles[name] = RoiDetectorProfile(
            name=name,
            box_threshold=float(_ratio(row["box_threshold"], f"{label}.box_threshold")),
            text_threshold=float(_ratio(row["text_threshold"], f"{label}.text_threshold")),
            min_mask_area_ratio=_ratio(
                row["min_mask_area_ratio"], f"{label}.min_mask_area_ratio", allow_null=True
            ),
        )
    frozen_name = _required(
        block, "frozen_public_metric_profile", "config.roi_detector"
    )
    if frozen_name not in profiles:
        raise ValueError(
            "config.roi_detector.frozen_public_metric_profile must name a declared profile"
        )
    frozen = profiles[frozen_name]
    if (
        frozen.box_threshold,
        frozen.text_threshold,
        frozen.min_mask_area_ratio,
    ) != (
        FROZEN_PUBLIC_ROI_DETECTOR.box_threshold,
        FROZEN_PUBLIC_ROI_DETECTOR.text_threshold,
        FROZEN_PUBLIC_ROI_DETECTOR.min_mask_area_ratio,
    ):
        raise ValueError(
            "config.roi_detector.frozen_public_metric_profile differs from the frozen "
            "public-metric detector contract in code"
        )
    raw_metric_profiles = _required(block, "metric_profiles", "config.roi_detector")
    if not isinstance(raw_metric_profiles, dict):
        raise ValueError("config.roi_detector.metric_profiles must be an object")
    metric_profiles: dict[str, str] = {}
    for metric_id, name in raw_metric_profiles.items():
        if metric_id in _FROZEN_ROI_METRIC_IDS:
            raise ValueError(
                f"config.roi_detector.metric_profiles must not remap the frozen "
                f"public metric {metric_id!r}"
            )
        if metric_id not in _ROI_DIAGNOSTIC_METRIC_IDS:
            raise ValueError(
                f"config.roi_detector.metric_profiles has an unknown metric {metric_id!r}"
            )
        if name not in profiles:
            raise ValueError(
                f"config.roi_detector.metric_profiles[{metric_id!r}] names an "
                f"undeclared profile {name!r}"
            )
        metric_profiles[metric_id] = name
    missing = [
        metric_id
        for metric_id in _ROI_DIAGNOSTIC_METRIC_IDS
        if metric_id not in metric_profiles
    ]
    if missing:
        raise ValueError(f"config.roi_detector.metric_profiles is missing {missing}")
    return RoiDetectorContract(
        source=source,
        profiles=profiles,
        metric_profiles=metric_profiles,
    )


def _existing_file(root: Path, relative: str) -> Path:
    path = root.expanduser().resolve() / relative
    if not path.is_file():
        raise FileNotFoundError(f"metric file is missing: {path}")
    return path


def _existing_dir(root: Path, relative: str) -> Path:
    path = root.expanduser().resolve() / relative
    if not path.is_dir():
        raise FileNotFoundError(f"metric directory is missing: {path}")
    return path


def _uniform_indices(length: int, count: int) -> tuple[int, ...]:
    if length <= 0 or count <= 0:
        raise ValueError("length and count must be positive")
    if length <= count:
        return tuple(range(length))
    # Matches the official VideoScore np.arange(0, N, N/max_frames).astype(int).
    return tuple(int(index * length / count) for index in range(count))


def _eight_shot_indices(start: int, frame_count: int) -> tuple[int, ...]:
    if frame_count < 8:
        raise ValueError("ViCLIP requires at least eight frames in every shot")
    # Official ViCLIP demo: step = len(frames) // 8; frames[::step][:8].
    step = frame_count // 8
    return tuple(start + index * step for index in range(8))


def _decode_rgb(video_path: Path) -> tuple[list[Any], float]:
    import av

    frames: list[Any] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"unable to decode a non-empty native-fps video: {video_path}")
    return frames, fps


def _manifest_key(case: PublicMetricCase, *, arm_independent: object) -> str:
    story_id = case.result_metadata.get("story_id")
    if story_id is None:
        return case.case_id
    if not isinstance(story_id, str) or not story_id.strip():
        raise ValueError(f"story_id must be a non-empty string for {case.case_id}")
    if arm_independent is not True:
        raise ValueError(
            "pair manifest must declare arm_independent=true to be keyed by story_id"
        )
    return story_id


def _load_pair_case(case: PublicMetricCase) -> tuple[Mapping[str, Any], str]:
    if case.pair_manifest_path is None:
        shot_ids = [str(shot["shot_id"]) for shot in case.shots]
        if len(shot_ids) < 2:
            raise ValueError("DINOv2 consistency metrics require at least two shots")
        return {
            # 没有 manifest 时 character 的配对是**合成**的，产物里必须能分辨——
            # 两份出厂 config 的 ``pair_manifest`` 实测都是 null，所以这条合成分支
            # 就是默认在跑的那一条，而它与传 manifest 是两种口径
            # （adjacent_pairs 240 对 vs all_unordered_pairs 480 对）。
            # 此前输出行里没有任何标记，读数的人分不出手上这一列是哪一种。
            "pair_source": "synthesized_adjacent_pairs",
            "character": {
                "eligible": True,
                "eligible_shots": shot_ids,
                "pair_rule": "adjacent_pairs",
            },
            # Scene consistency is only defined on pairs the manifest names as
            # the same physical environment. Shot order is not that declaration.
            "scene": {
                "eligible": False,
                "same_environment_groups": [],
                "reason": (
                    "no explicit pair manifest: same-environment groups cannot "
                    "be synthesized from shot order"
                ),
            },
        }, case.subject_class
    payload = json.loads(case.pair_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pair manifest must be an object")
    contract = payload.get("subject_class_contract")
    if not isinstance(contract, dict):
        raise ValueError("pair manifest is missing subject_class_contract")
    default = contract.get("default_class_for_all_cases")
    overrides = contract.get("case_overrides")
    allowed = contract.get("allowed_detector_classes")
    if not isinstance(default, str) or not isinstance(overrides, dict) or not isinstance(allowed, list):
        raise ValueError("pair manifest subject_class_contract is malformed")
    if (
        not allowed
        or any(not isinstance(value, str) or not value for value in allowed)
        or len(set(allowed)) != len(allowed)
        or default not in allowed
        or any(not isinstance(key, str) or value not in allowed for key, value in overrides.items())
    ):
        raise ValueError("pair manifest subject_class_contract contains invalid classes")
    # The manifest is arm- and seed-independent, so it is keyed by story while a
    # scored case is one story x arm x seed. Resolve on the story when the input
    # carries one, otherwise the composite ID silently falls back to the default
    # detector class and never matches a row.
    manifest_key = _manifest_key(case, arm_independent=payload.get("arm_independent"))
    subject_class = overrides.get(manifest_key, default)
    if not isinstance(subject_class, str) or subject_class not in allowed:
        raise ValueError(f"invalid explicit detector subject class for {manifest_key}")
    rows = payload.get("cases")
    if not isinstance(rows, list):
        raise ValueError("pair manifest cases must be a list")
    matches = [row for row in rows if isinstance(row, dict) and row.get("case_id") == manifest_key]
    if len(matches) != 1:
        raise PairManifestError(
            f"pair manifest must contain exactly one row for {manifest_key} "
            f"(scored case {case.case_id})"
        )
    row = dict(matches[0])
    # manifest 自报来源时透传：sidecar 自动生成的那份会写 ``sidecar_autogen``，
    # 否则才是 owner 标注的 ``manifest``。一律标 ``manifest`` 会把程序按镜序凑的
    # 相邻对呈现成人工声明的镜对。
    row.setdefault("pair_source", "manifest")
    row["pair_manifest_path"] = str(case.pair_manifest_path)
    return row, subject_class


def _shot_pairs(row: Mapping[str, Any], kind: str) -> tuple[tuple[str, str], ...] | None:
    section = row.get(kind)
    if not isinstance(section, dict) or not isinstance(section.get("eligible"), bool):
        raise ValueError(f"pair manifest {kind} contract is malformed")
    if not section["eligible"]:
        if kind != "scene" or not isinstance(section.get("reason"), str):
            raise ValueError(f"invalid ineligible {kind} contract")
        return None
    if kind == "character":
        ids = section.get("eligible_shots")
        if not isinstance(ids, list):
            raise ValueError("character eligible_shots must be a list")
        if section.get("pair_rule") == "adjacent_pairs":
            return tuple(zip(ids, ids[1:], strict=False))
        if section.get("pair_rule") != "all_unordered_pairs":
            raise ValueError("character pair rule must be adjacent_pairs or all_unordered_pairs")
        return tuple(combinations(ids, 2))
    groups = section.get("same_environment_groups")
    if not isinstance(groups, list) or isinstance(groups, (str, bytes)):
        raise ValueError("scene same_environment_groups must be a list")
    parsed_groups: list[list[str]] = []
    for index, group in enumerate(groups):
        if not isinstance(group, list) or isinstance(group, (str, bytes)):
            raise ValueError(
                f"scene same_environment_groups[{index}] must be a list of shot IDs"
            )
        if any(not isinstance(shot_id, str) or not shot_id for shot_id in group):
            raise ValueError(
                f"scene same_environment_groups[{index}] must contain non-empty shot IDs"
            )
        parsed_groups.append(list(group))
    if section.get("pair_rule") == "adjacent_pairs":
        return tuple(
            pair
            for group in parsed_groups
            for pair in zip(group, group[1:], strict=False)
        )
    if section.get("pair_rule") != "within_group_all_unordered_pairs":
        raise ValueError(
            "scene pair rule must be adjacent_pairs or within_group_all_unordered_pairs"
        )
    return tuple(
        pair for group in parsed_groups for pair in combinations(group, 2)
    )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    ln = math.sqrt(sum(float(value) ** 2 for value in left))
    rn = math.sqrt(sum(float(value) ** 2 for value in right))
    if not left or ln == 0 or rn == 0:
        raise ValueError("embedding must be non-empty and non-zero")
    return dot / (ln * rn)


def _macro_with_missing(
    embeddings: Mapping[str, Sequence[Sequence[float] | None]],
    pairs: Sequence[tuple[str, str]],
) -> MetricResult:
    raw_pairs: list[dict[str, Any]] = []
    means: list[float] = []
    missing = 0
    for left_id, right_id in pairs:
        values: list[float] = []
        for left in embeddings[left_id]:
            for right in embeddings[right_id]:
                if left is None or right is None:
                    values.append(0.0)
                    missing += 1
                else:
                    values.append(_cosine(left, right))
        if len(values) != 9:
            raise ValueError("each shot pair must produce exactly nine keyframe comparisons")
        pair_mean = sum(values) / 9
        means.append(pair_mean)
        raw_pairs.append({"shot_pair": [left_id, right_id], "cosine_3x3": values, "pair_mean": pair_mean})
    if not means:
        raise ValueError("eligible pair list must not be empty")
    return MetricResult(
        value=sum(means) / len(means),
        raw_outputs={"pairs": raw_pairs},
        diagnostics={"pair_count": len(means), "cosines_per_pair": 9,
                     "missing_roi_keyframe_comparisons_zero_filled": missing},
    )


@dataclass(frozen=True)
class GatedCharacterReadout:
    """One gated pass over the eligible character pairs.

    ``_macro_with_missing`` keeps the frozen zero-filling contract, where a
    keyframe with no detected subject scores an explicit 0.0 cosine.  This
    readout instead averages only the comparisons whose two keyframes both
    produced an ROI, so a case that lost detection reports ``None`` rather than
    a low similarity, and the denominators stay visible.
    """

    macro: float | None
    pairs: tuple[dict[str, Any], ...]
    n_pairs: int
    n_pairs_detected: int
    comparisons: int
    comparisons_total: int


def _gated_character_readout(
    embeddings: Mapping[str, Sequence[Sequence[float] | None]],
    pairs: Sequence[tuple[str, str]],
) -> GatedCharacterReadout:
    pair_rows: list[dict[str, Any]] = []
    means: list[float] = []
    comparisons = 0
    for left_id, right_id in pairs:
        values: list[float] = []
        missing = 0
        for left in embeddings[left_id]:
            for right in embeddings[right_id]:
                if left is None or right is None:
                    missing += 1
                else:
                    values.append(_cosine(left, right))
        if len(values) + missing != 9:
            raise ValueError("each shot pair must produce exactly nine keyframe comparisons")
        pair_mean = sum(values) / len(values) if values else None
        if pair_mean is not None:
            means.append(pair_mean)
        comparisons += len(values)
        pair_rows.append({
            "shot_pair": [left_id, right_id],
            "gated_cosines": values,
            "pair_mean": pair_mean,
            "detected_comparisons": len(values),
            "missing_roi_comparisons": missing,
        })
    if not pair_rows:
        raise ValueError("eligible pair list must not be empty")
    return GatedCharacterReadout(
        macro=sum(means) / len(means) if means else None,
        pairs=tuple(pair_rows),
        n_pairs=len(pair_rows),
        n_pairs_detected=len(means),
        comparisons=comparisons,
        comparisons_total=9 * len(pair_rows),
    )


@dataclass(frozen=True)
class IdentityReadout:
    """The v3 identity column and the two denominators it must be read beside.

    A unit is one (shot pair x character query).  It scores only when all six
    keyframes -- three in each shot -- detected that character, so the nine
    cosines are always complete.  The looser rule, averaging whichever
    comparisons survived, would bias the column by arm: an arm whose subjects
    are harder to detect would be scored only on its easy keyframes (large,
    frontal, low motion blur) and read as better at identity for being worse at
    being seen.  The price is that ``pairs_scored`` differs between arms, which
    is why it is a reported column and not a footnote.
    """

    value: float | None
    pairs: tuple[dict[str, Any], ...]
    pairs_scored: int
    pairs_total: int
    reason: str | None = None

    @property
    def cases_scored(self) -> float:
        return 1.0 if self.value is not None else 0.0


def _identity_readout(
    embeddings: Mapping[str, Mapping[str, Sequence[Sequence[float] | None]]],
    pairs: Sequence[tuple[str, str]],
    queries: Sequence[str],
) -> IdentityReadout:
    pair_rows: list[dict[str, Any]] = []
    pair_means: list[float] = []
    scored = 0
    for left_id, right_id in pairs:
        unit_rows: list[dict[str, Any]] = []
        unit_means: list[float] = []
        for query in queries:
            left_side = embeddings[left_id][query]
            right_side = embeddings[right_id][query]
            if len(left_side) != 3 or len(right_side) != 3:
                raise ValueError("each shot must contribute exactly three keyframes")
            detected = sum(
                1 for value in (*left_side, *right_side) if value is not None
            )
            if detected != 6:
                unit_rows.append({
                    "query": query,
                    "scored": False,
                    "detected_keyframes": detected,
                })
                continue
            values = [
                _cosine(left, right) for left in left_side for right in right_side
            ]
            unit_mean = sum(values) / 9
            unit_means.append(unit_mean)
            scored += 1
            unit_rows.append({
                "query": query,
                "scored": True,
                "detected_keyframes": 6,
                "cosine_3x3": values,
                "unit_mean": unit_mean,
            })
        pair_mean = sum(unit_means) / len(unit_means) if unit_means else None
        if pair_mean is not None:
            pair_means.append(pair_mean)
        pair_rows.append({
            "shot_pair": [left_id, right_id],
            "units": unit_rows,
            "pair_mean": pair_mean,
        })
    return IdentityReadout(
        value=sum(pair_means) / len(pair_means) if pair_means else None,
        pairs=tuple(pair_rows),
        pairs_scored=scored,
        pairs_total=len(pairs) * len(queries),
    )


def _motion_magnitude(
    frames: Sequence[Any], counts: Sequence[int], *, native_fps: float
) -> MetricResult:
    """Mean absolute adjacent-frame change, cuts excluded, on decoded frames."""

    import numpy as np

    boundaries = set(planned_shot_boundaries(list(counts)))
    within: list[float] = []
    across: list[float] = []
    for index in range(1, len(frames)):
        current = np.asarray(frames[index], dtype=np.float32)
        previous = np.asarray(frames[index - 1], dtype=np.float32)
        delta = float(np.abs(current - previous).mean()) / 255.0
        (across if index in boundaries else within).append(delta)
    if not within:
        raise ValueError("motion_magnitude requires at least one within-shot frame pair")
    value = sum(within) / len(within)
    return MetricResult(value, {
        "within_shot_frame_pairs": len(within),
        "within_shot_min": min(within),
        "within_shot_max": max(within),
        "cross_boundary_deltas": across,
    }, {
        "definition": (
            "mean over within-shot adjacent frame pairs of the mean absolute RGB "
            "difference, divided by 255; shot cuts are reported separately"
        ),
        "within_shot_frame_pairs": len(within),
        "cross_boundary_frame_pairs": len(across),
        "cross_boundary_mean": sum(across) / len(across) if across else None,
        "all_adjacent_mean": (
            (sum(within) + sum(across)) / (len(within) + len(across))
        ),
        "native_fps": native_fps,
    })


class _VideoScoreEngine:
    def __init__(self, root: Path, device: str):
        import torch
        from transformers import AutoProcessor
        from mantis.models.idefics2 import Idefics2ForSequenceClassification

        model_dir = _existing_dir(root, "videoscore")
        self.processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
        self.model = Idefics2ForSequenceClassification.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16, local_files_only=True
        ).eval().to(device)
        self.device = device

    def score(self, frames: Sequence[Any], prompt: str) -> tuple[float, Mapping[str, float]]:
        import torch
        from PIL import Image

        selected = [Image.fromarray(frames[index]) for index in _uniform_indices(len(frames), 48)]
        text = _VIDEOSCORE_PROMPT.format(text_prompt=prompt) + "<image> " * len(selected)
        inputs = self.processor(text=text, images=selected, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits[0].float().cpu().tolist()
        if len(logits) != 5 or not all(math.isfinite(float(value)) for value in logits):
            raise ValueError("VideoScore-v1.1 must return exactly five finite aspect logits")
        aspects = dict(zip(_VIDEOSCORE_ASPECTS, map(float, logits), strict=True))
        return sum(aspects.values()) / 5, aspects


def _instantiate_local_viclip(
    module: Any, *, checkpoint: Path, bpe_path: Path
) -> tuple[Any, Any]:
    """Construct ViCLIP-L/14 from source files under the resource root."""

    cls = getattr(module, "ViCLIP", None)
    tokenizer_cls = getattr(module, "_Tokenizer", None)
    if cls is None or tokenizer_cls is None:
        raise ImportError("local ViCLIP source lacks ViCLIP or _Tokenizer")
    tokenizer = tokenizer_cls(str(bpe_path))
    model = cls(tokenizer=tokenizer, pretrain=str(checkpoint))
    return model, tokenizer


class _ViCLIPEngine:
    """Adapter for ViCLIP Python sources stored under the resource root."""

    def __init__(self, root: Path, device: str):
        import torch

        source = _existing_dir(root, "viclip/source")
        entry = _existing_file(source, "viclip.py")
        checkpoint = _existing_file(root, "viclip/ViClip-InternVid-10M-FLT.pth")
        for name in ("viclip_vision.py", "viclip_text.py", "simple_tokenizer.py", "bpe_simple_vocab_16e6.txt.gz"):
            _existing_file(source, name)
        package_name = "_mspc_local_viclip_source"
        package = types.ModuleType(package_name)
        package.__path__ = [str(source)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package
        spec = importlib.util.spec_from_file_location(f"{package_name}.viclip", entry)
        if spec is None or spec.loader is None:
            raise ImportError("unable to load local ViCLIP source")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        sys.path.insert(0, str(source))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(source))
        self.model, self.tokenizer = _instantiate_local_viclip(
            module,
            checkpoint=checkpoint,
            bpe_path=_existing_file(source, "bpe_simple_vocab_16e6.txt.gz"),
        )
        self.model = self.model.eval().to(device)
        if not callable(getattr(self.model, "get_vid_features", None)) or not callable(
            getattr(self.model, "get_text_features", None)
        ):
            raise TypeError("official ViCLIP adapter lacks normalized feature methods")
        self.device = torch.device(device)

    def _video_tensor(self, frames: Sequence[Any]) -> Any:
        import cv2
        import numpy as np
        import torch

        if len(frames) != 8:
            raise ValueError("ViCLIP video embedding requires exactly eight frames")
        resized = [cv2.resize(frame, (224, 224)) for frame in frames]
        array = np.stack(resized).astype(np.float32) / 255.0
        array = (array - np.asarray((0.485, 0.456, 0.406))) / np.asarray((0.229, 0.224, 0.225))
        return torch.from_numpy(array.transpose(0, 3, 1, 2)).float().to(self.device)

    def embed_video(self, frames: Sequence[Any]) -> list[float]:
        """Return the normalized ViCLIP video embedding for eight RGB frames."""

        import torch

        tensor = self._video_tensor(frames)[None]
        with torch.no_grad():
            video = self.model.get_vid_features(tensor).float()
            video = video / video.norm(dim=-1, keepdim=True)
        return [float(value) for value in video[0].cpu().tolist()]

    def embed_videos(
        self, shots: Sequence[Sequence[Any]], *, viclip_shot_batch_size: int
    ) -> list[list[float]]:
        import torch

        batch_size = _positive_int(viclip_shot_batch_size, "viclip_shot_batch_size")
        embeddings: list[list[float]] = []
        for start in range(0, len(shots), batch_size):
            chunk = shots[start : start + batch_size]
            tensor = torch.stack([self._video_tensor(frames) for frames in chunk], dim=0)
            with torch.no_grad():
                video = self.model.get_vid_features(tensor).float()
                video = video / video.norm(dim=-1, keepdim=True)
            embeddings.extend(
                [float(value) for value in row] for row in video.cpu().tolist()
            )
        return embeddings

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return normalized ViCLIP text embeddings, one per text."""

        import torch

        rows: list[list[float]] = []
        for text in texts:
            with torch.no_grad():
                feature = self.model.get_text_features(text, self.tokenizer, {}).float()
                feature = feature / feature.norm(dim=-1, keepdim=True)
            rows.append([float(value) for value in feature[0].cpu().tolist()])
        return rows

    def score(self, frames: Sequence[Any], text: str) -> float:
        import torch

        video = torch.tensor([self.embed_video(frames)], device=self.device)
        with torch.no_grad():
            text_feature = self.model.get_text_features(text, self.tokenizer, {}).float()
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        return float((video @ text_feature.T)[0, 0].cpu())

    def score_shots(
        self,
        items: Sequence[tuple[Sequence[Any], str]],
        *,
        viclip_shot_batch_size: int,
    ) -> list[float]:
        import torch

        videos = self.embed_videos(
            [frames for frames, _ in items],
            viclip_shot_batch_size=viclip_shot_batch_size,
        )
        if len(videos) != len(items):
            raise ValueError(
                f"ViCLIP returned {len(videos)} video embeddings for {len(items)} shots"
            )
        scores: list[float] = []
        for vector, (_, text) in zip(videos, items, strict=True):
            video = torch.tensor([vector], device=self.device)
            with torch.no_grad():
                text_feature = self.model.get_text_features(text, self.tokenizer, {}).float()
                text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            scores.append(float((video @ text_feature.T)[0, 0].cpu()))
        return scores


class _PyIQAEngine:
    """VBench's two per-frame quality dimensions, scored one shot at a time.

    Only ``aesthetic_quality`` and ``imaging_quality`` are reported, and they run
    on the two upstream models VBench itself uses: the LAION aesthetic predictor
    on CLIP ViT-L/14 embeddings, and MUSIQ trained on SPAQ.  Both come from
    ``pyiqa`` rather than the ``vbench`` distribution, because every published
    ``vbench`` wheel pins ``transformers==4.33.2``, which this project cannot
    take (Grounding DINO needs ``>=4.40``) -- measured: ``uv lock`` refuses to
    resolve.  This class runs inside the ``renderer_quality`` role's own
    interpreter (pyiqa 0.1.16, its own ``transformers>=5.0``); see
    ``_SubprocessVBenchEngine`` below for why that has to be a separate process
    rather than an in-process import, and ``tools/run/quality_worker.py`` for
    the process that actually constructs this class.

    The reported value is divided by the same constant VBench divides by (10 for
    the aesthetic predictor's ~1..10 range, 100 for MUSIQ's ~0..100), so a number
    here is on VBench's published scale.

    Both are frame models with a per-video reduction, so a whole concatenated
    story would charge every hard cut to picture quality; the score is therefore
    a mean over shots of a mean over that shot's sampled frames.  Frames are
    sampled with this repository's existing eight-per-shot rule (the one ViCLIP
    already uses), which keeps the column inside its GPU budget; that sampling is
    ours, not VBench's, and is recorded in the metric's ``definition``.
    """

    #: pyiqa registry names, read out of ``pyiqa.default_model_configs`` rather
    #: than assumed.  ``musiq`` alone is a different checkpoint (koniq); VBench's
    #: imaging quality is the SPAQ one.
    AESTHETIC_METRIC = "laion_aes"
    IMAGING_METRIC = "musiq-spaq"
    #: VBench's own normalisation, matching each model's native range.
    NORMALISERS = {"aesthetic_quality": 10.0, "imaging_quality": 100.0}
    DIMENSIONS = ("aesthetic_quality", "imaging_quality")
    #: The CLIP archive ``pyiqa``'s LAION predictor loads through ``clip.load``,
    #: which takes no download root, so the file has to be visible at the
    #: default cache path.
    CLIP_ARCHIVE = "ViT-L-14.pt"
    #: pyiqa 0.1.16 caches both weight files under ``<TORCH_HOME>/hub/pyiqa/``
    #: (``torch.hub.get_dir()/pyiqa``, read from
    #: ``pyiqa.utils.download_util.DEFAULT_CACHE_DIR``), not the
    #: ``<TORCH_HOME>/hub/checkpoints/`` layout pyiqa 0.1.10 used
    #: (``torch.hub``-style ``load_file_from_url`` with ``model_dir=None``) --
    #: measured by downloading and reading both wheels, not by running either.
    #: The two files are staged on the shared disk in the old layout; this
    #: class links them into the new one, the same shape as
    #: ``_link_clip_archive`` below.
    HUB_CHECKPOINTS_DIRNAME = "checkpoints"
    HUB_PYIQA_DIRNAME = "pyiqa"
    #: Basenames are unchanged between pyiqa 0.1.10 and 0.1.16, and both land
    #: on the percent-encoded form for the LAION file: ``load_file_from_url``
    #: (both versions) takes the cache filename from
    #: ``os.path.basename(urlparse(url).path)``, which never decodes percent
    #: escapes. 0.1.10 hardcodes the GitHub raw URL with a literal ``%2B`` in
    #: ``laion_aes_arch.py``; 0.1.16 builds the URL through
    #: ``huggingface_hub.hf_hub_url(..., filename=name)``, which runs the
    #: filename through ``urllib.parse.quote()`` (default ``safe='/'``, so
    #: ``+`` is escaped) -- read from both wheels and huggingface_hub's own
    #: ``file_download.py``, not observed by running either. The MUSIQ
    #: checkpoint's name has no character ``quote()`` touches, so it is
    #: identical either way.
    WEIGHT_BASENAMES = ("musiq_spaq_ckpt-358bb6af.pth", "sac%2Blogos%2Bava1-l14-linearMSE.pth")

    def __init__(self, root: Path, device: str):
        self.checkpoints = _existing_dir(root, "vbench")
        self.device = device
        # TORCH_HOME has to be set **before** ``import pyiqa``: pyiqa reads
        # ``torch.hub.get_dir()`` once, at import, into the module constant
        # ``pyiqa.utils.download_util.DEFAULT_CACHE_DIR``, so setting it
        # afterwards changes nothing.  Measured: with the assignment below the
        # import, predict reached out to
        # ``/root/.cache/torch/hub/pyiqa/sac%2Blogos%2Bava1-l14-linearMSE.pth``
        # and died on ``URLError: Network is unreachable`` -- the staged copy
        # was right there and never consulted.  pyiqa's release hosts (GitHub
        # release/raw for 0.1.10, Hugging Face for 0.1.16) are both unreachable
        # from the execution surface, so a late TORCH_HOME is not a slow path,
        # it is a dead one.
        os.environ["TORCH_HOME"] = str(self.checkpoints)
        self._link_clip_archive()
        self._link_hub_weights()

        import pyiqa
        available = set(pyiqa.list_models())
        missing = [
            name
            for name in (self.AESTHETIC_METRIC, self.IMAGING_METRIC)
            if name not in available
        ]
        if missing:
            raise RuntimeError(
                f"pyiqa does not register {missing}; the quality columns cannot run"
            )
        self._metrics = {
            "aesthetic_quality": pyiqa.create_metric(self.AESTHETIC_METRIC, device=device),
            "imaging_quality": pyiqa.create_metric(self.IMAGING_METRIC, device=device),
        }

    def _link_clip_archive(self) -> None:
        """Make the staged CLIP archive visible at ``clip.load``'s default path."""

        staged = self.checkpoints / "clip" / self.CLIP_ARCHIVE
        if not staged.is_file():
            raise FileNotFoundError(
                f"the LAION aesthetic predictor needs CLIP ViT-L/14 staged at {staged}"
            )
        cache = Path.home() / ".cache" / "clip"
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / self.CLIP_ARCHIVE
        if not target.exists():
            target.symlink_to(staged)

    def _link_hub_weights(self) -> None:
        """Bridge the staged ``hub/checkpoints`` weights into pyiqa 0.1.16's
        ``hub/pyiqa`` cache directory; see ``HUB_PYIQA_DIRNAME`` above."""

        source_dir = self.checkpoints / "hub" / self.HUB_CHECKPOINTS_DIRNAME
        target_dir = self.checkpoints / "hub" / self.HUB_PYIQA_DIRNAME
        target_dir.mkdir(parents=True, exist_ok=True)
        for basename in self.WEIGHT_BASENAMES:
            staged = source_dir / basename
            if not staged.is_file():
                raise FileNotFoundError(
                    f"the quality columns need {basename} staged at {staged}"
                )
            target = target_dir / basename
            if not target.exists():
                target.symlink_to(staged)

    def score(self, shots: Sequence[Sequence[Any]]) -> Mapping[str, float]:
        import numpy as np
        import torch

        if not shots:
            raise ValueError("the quality columns need at least one shot")
        per_dimension: dict[str, list[float]] = {name: [] for name in self.DIMENSIONS}
        for index, frames in enumerate(shots):
            if not frames:
                raise ValueError(f"shot {index} decoded to zero frames")
            sampled = [frames[i] for i in _eight_shot_indices(0, len(frames))]
            batch = torch.from_numpy(np.stack(sampled)).to(self.device)
            batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
            for name, metric in self._metrics.items():
                with torch.no_grad():
                    values = metric(batch)
                score = float(values.mean().cpu()) / self.NORMALISERS[name]
                if not math.isfinite(score):
                    raise ValueError(f"pyiqa returned a non-finite {name}")
                per_dimension[name].append(score)
        return {
            name: sum(values) / len(values) for name, values in per_dimension.items()
        }


def preflight_quality_gpu(device: str) -> None:
    """Refuse a GPU quality run when this process cannot see a CUDA device.

    Same lightweight strength as ``preflight_tensorflow_gpu``: checks device
    visibility only, never constructs ``_PyIQAEngine`` (which would load
    pyiqa's weights). The quality worker calls this; the parent scoring
    process must not import torch from this role's interpreter.
    """

    if device == "cpu":
        return
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            f"torch is not importable under {sys.executable}: {error}"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"pyiqa quality worker requested {device} but torch.cuda.is_available() "
            f"is False under {sys.executable}"
        )


def _arch_for_role(
    registry: Mapping[str, Any], shared_root: Path, *, role: str, python_env: str
) -> str:
    arch = os.environ.get("MSPC_GPU_ARCH") or os.environ.get("MSPC_GPU_ARCH_OVERRIDE")
    if arch and str(arch).strip():
        return str(arch).strip()
    inferred = _arch_from_parent_metrics_interpreter(registry, shared_root)
    if inferred:
        return inferred
    raise RuntimeError(
        f"cannot resolve {role} architecture: set {python_env} "
        "to that role's interpreter, or set MSPC_GPU_ARCH"
    )


def quality_role_device() -> str:
    """The device the ``renderer_quality`` role declares, from the registry.

    Not a default and not a fallback: the value is required, and a role without
    it is a configuration error rather than something to guess at.  See the
    role's ``device_reason`` for why it is what it is.
    """

    registry_path, registry = _load_env_registry()
    roles = registry.get("roles")
    if not isinstance(roles, dict) or QUALITY_ROLE not in roles:
        raise RuntimeError(f"role {QUALITY_ROLE} is not registered in {registry_path}")
    role = roles[QUALITY_ROLE]
    device = role.get("device") if isinstance(role, dict) else None
    if not isinstance(device, str) or not device:
        raise ValueError(
            f"the {QUALITY_ROLE} role must declare a non-empty string device "
            f"in {registry_path}"
        )
    return device


def _resolve_role_python(role: str, python_env: str) -> Path:
    """Return a worker role's interpreter. Fail loud; never fall back to this process."""

    explicit = os.environ.get(python_env, "").strip()
    if explicit:
        path = _venv_entry_python(Path(explicit))
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(
                f"{python_env} is set to {explicit} but the venv entry "
                f"{_interpreter_identity(path)} is not an executable file"
            )
        return path
    registry_path, registry = _load_env_registry()
    shared_root = _shared_root_from_registry(registry, registry_path)
    arch = _arch_for_role(registry, shared_root, role=role, python_env=python_env)
    roles = registry.get("roles")
    if not isinstance(roles, dict) or role not in roles:
        raise RuntimeError(f"role {role} is not registered in {registry_path}")
    spec = roles[role]
    envs = spec.get("envs") if isinstance(spec, dict) else None
    if not isinstance(envs, dict) or arch not in envs:
        note = ""
        if isinstance(spec, dict):
            notes = spec.get("unregistered_arch_notes", {})
            if isinstance(notes, dict) and isinstance(notes.get(arch), str):
                note = f"; registry note: {notes[arch]}"
        registered = ", ".join(sorted(envs)) if isinstance(envs, dict) else ""
        raise RuntimeError(
            f"role {role} has no interpreter registered for {arch} in "
            f"{registry_path}; registered for this role: {registered}{note}"
        )
    relative = envs[arch].get("python") if isinstance(envs[arch], dict) else None
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"{registry_path}: role {role} arch {arch} declares no python")
    if Path(relative).is_absolute():
        raise RuntimeError(
            f"{registry_path}: role {role} arch {arch} python must be "
            "relative to the shared root"
        )
    path = _venv_entry_python(shared_root / relative)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(
            f"the registry names {relative} for {role}/{arch}, but "
            f"{_interpreter_identity(path)} is not an executable file; the "
            "role's env may not be bootstrapped yet"
        )
    return path


def resolve_quality_python() -> Path:
    """Return the pyiqa interpreter."""

    return _resolve_role_python(QUALITY_ROLE, QUALITY_PYTHON_ENV)


def resolve_vbench_python() -> Path:
    """Return the official-``vbench`` interpreter (``renderer_vbench`` role)."""

    return _resolve_role_python(VBENCH_ROLE, VBENCH_PYTHON_ENV)


def _parse_worker_payload(
    stdout: str, *, label: str, python: Path, stderr: str
) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        extra = f"; stderr: {stderr.strip()}" if stderr.strip() else ""
        raise RuntimeError(
            f"{label} under "
            f"{_interpreter_identity(python)} wrote no JSON on stdout{extra}"
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{label} under "
            f"{_interpreter_identity(python)} wrote invalid JSON: {error}; "
            f"stdout={text!r}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"{label} under "
            f"{_interpreter_identity(python)} must write a JSON object, "
            f"got {type(payload).__name__}"
        )
    return payload


def _invoke_json_worker(
    worker: Path, *, label: str, python: Path, args: Sequence[str]
) -> dict[str, Any]:
    """Run one worker script under its role's interpreter and read its JSON line."""

    if not worker.is_file():
        raise FileNotFoundError(f"{label} is missing: {worker}")
    env = dict(os.environ)
    src = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else f"{src}{os.pathsep}{existing}"
    completed = subprocess.run(
        [str(python), str(worker), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    payload = _parse_worker_payload(
        completed.stdout, label=label, python=python, stderr=completed.stderr
    )
    if completed.returncode != 0 or payload.get("status") != "ok":
        message = payload.get("error")
        detail = (
            message
            if isinstance(message, str) and message.strip()
            else completed.stderr.strip() or completed.stdout.strip() or (
                f"exit {completed.returncode}"
            )
        )
        raise RuntimeError(
            f"{label} failed under {_interpreter_identity(python)}: {detail}"
        )
    return payload


def invoke_quality_worker(*, python: Path, args: Sequence[str]) -> dict[str, Any]:
    return _invoke_json_worker(
        QUALITY_WORKER, label="quality worker", python=python, args=args
    )


class _SubprocessVBenchEngine:
    """Run ``_PyIQAEngine`` in the ``renderer_quality`` role's own interpreter.

    The parent scoring process keeps the same ``vbench_factory(root, device)``
    contract and still calls ``.score(shots)``.  Frames cross the process
    boundary as one flat uint8 NHWC ``.npy`` (decoded RGB, not resampled) plus
    a ``--shot-counts`` boundary list, so the eight-per-shot sampling stays
    owned by ``_PyIQAEngine.score`` on the far side rather than being
    duplicated here.  Unlike ``_SubprocessTransNetEngine``, the isolation here
    is a version conflict (pyiqa 0.1.16 needs ``transformers>=5.0``; this
    project's own scoring env needs ``<5``), not a library-exclusivity
    conflict, so the child process is expected to import torch -- there is no
    ``_refuse_torch``-style contract to keep here. Child failures surface as
    ``RuntimeError`` with the worker's error text; they are never turned into
    N/A or 0.
    """

    def __init__(self, root: Path, device: str):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"resource root does not exist: {self.root}")
        # The device is **declared by the role**, not inherited from the parent
        # scoring process.  Measured on main-server A100 (torch 2.10.0+cu128, driver
        # 12.9, an environment rebuilt from scratch, an idle card, and batch
        # sizes from 8 down to 1): pyiqa's ``musiq-spaq`` always raises
        # ``CUDA error: CUBLAS_STATUS_INVALID_VALUE`` inside MUSIQ's forward,
        # while ``laion_aes`` runs on CUDA in the same process.  On CPU both
        # produce correct values that agree with the CUDA ``laion_aes`` result.
        # Reading the declaration rather than trying CUDA and catching the
        # failure is the difference between a declared device and a silent
        # fallback; the latter is what this project refuses.  ``device`` stays
        # in the signature because the factory contract is shared with the
        # other engines, and is checked against the declaration.
        declared = quality_role_device()
        if device != declared:
            raise ValueError(
                f"the {QUALITY_ROLE} role declares device={declared!r}; the scoring "
                f"process asked for {device!r}. Change the declaration in "
                "configs/runtime/env_by_arch.json, not the call site."
            )
        self.device = declared
        self.python = resolve_quality_python()

    @classmethod
    def preflight_device(cls, device: str) -> None:
        """Preflight the interpreter on the device **the role declares**.

        The caller passes the scoring process's own device, which is the card
        the other columns use.  This role does not run there -- see the
        ``device_reason`` in the registry -- so preflighting on the caller's
        device would ask pyiqa for a card it will never touch and fail with a
        message about CUDA that has nothing to do with the actual arrangement.
        The declaration is checked against the caller the same way ``__init__``
        checks it, so a disagreement is still an error rather than a shrug.
        """

        declared = quality_role_device()
        if device != declared:
            raise ValueError(
                f"the {QUALITY_ROLE} role declares device={declared!r}; the scoring "
                f"process asked for {device!r}. Change the declaration in "
                "configs/runtime/env_by_arch.json, not the call site."
            )
        invoke_quality_worker(
            python=resolve_quality_python(),
            args=["--mode", "preflight", "--device", declared],
        )

    def score(self, shots: Sequence[Sequence[Any]]) -> Mapping[str, float]:
        import numpy as np

        if not shots:
            raise ValueError("the quality columns need at least one shot")
        counts: list[int] = []
        flattened: list[Any] = []
        for index, frames in enumerate(shots):
            if not frames:
                raise ValueError(f"shot {index} decoded to zero frames")
            counts.append(len(frames))
            flattened.extend(np.asarray(frame) for frame in frames)
        stacked = np.stack(flattened)
        if stacked.ndim != 4 or stacked.shape[-1] != 3:
            raise ValueError(
                f"quality worker frames must be NHWC RGB, got shape {stacked.shape}"
            )
        with tempfile.TemporaryDirectory(prefix="mspc-quality-") as tmp:
            frames_path = Path(tmp) / "frames.npy"
            np.save(frames_path, stacked, allow_pickle=False)
            payload = invoke_quality_worker(
                python=self.python,
                args=[
                    "--mode",
                    "predict",
                    "--device",
                    self.device,
                    "--resource-root",
                    str(self.root),
                    "--frames-npy",
                    str(frames_path),
                    "--shot-counts",
                    ",".join(str(count) for count in counts),
                ],
            )
        scores = payload.get("scores")
        if not isinstance(scores, dict):
            raise RuntimeError(
                f"quality worker under {self.python} did not return a scores object"
            )
        missing = [name for name in _PyIQAEngine.DIMENSIONS if name not in scores]
        if missing:
            raise RuntimeError(f"quality worker under {self.python} did not report {missing}")
        values = {name: float(scores[name]) for name in _PyIQAEngine.DIMENSIONS}
        non_finite = [name for name, value in values.items() if not math.isfinite(value)]
        if non_finite:
            raise RuntimeError(
                f"quality worker under {self.python} returned a non-finite {non_finite}"
            )
        return values


class _DinoRoiEngine:
    """Open-vocabulary box -> SAM mask -> DINOv2 embedding.

    Grounding DINO replaces YOLO11x as the box source and nothing downstream
    moves: the box is still SAM's prompt, SAM still picks the mask with the
    highest predicted IoU, and the crop is still pasted on an ImageNet-mean
    square canvas before DINOv2 reads its CLS token.  Cropping the raw box
    instead would send the box's background pixels into a global CLS token and
    make cross-shot identity partly a measurement of whether the two shots
    share a background -- which is exactly the difference between the arms
    being compared.
    """

    def __init__(self, root: Path, device: str):
        from transformers import (
            AutoImageProcessor,
            AutoModel,
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            SamModel,
            SamProcessor,
        )

        dino_dir = _existing_dir(root, "dinov2")
        sam_dir = _existing_dir(root, "roi/sam-vit-huge")
        detector_dir = _existing_dir(root, "roi/grounding-dino-tiny")
        self.dino_processor = AutoImageProcessor.from_pretrained(str(dino_dir), local_files_only=True)
        self.dino = AutoModel.from_pretrained(str(dino_dir), local_files_only=True).eval().to(device)
        self.sam_processor = SamProcessor.from_pretrained(str(sam_dir), local_files_only=True)
        self.sam = SamModel.from_pretrained(str(sam_dir), local_files_only=True).eval().to(device)
        self.detector_processor = AutoProcessor.from_pretrained(
            str(detector_dir), local_files_only=True
        )
        self.detector = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                str(detector_dir), local_files_only=True
            )
            .eval()
            .to(device)
        )
        self.device = device
        self._detector_cache: dict[tuple[Any, ...], list[float] | None] = {}
        self._sam_cache: dict[tuple[Any, ...], list[Any]] = {}

    def clear_frame_caches(self) -> None:
        self._detector_cache.clear()
        self._sam_cache.clear()

    @staticmethod
    def _detector_cache_key(
        frame: Any, query: str, profile: RoiDetectorProfile
    ) -> tuple[Any, ...]:
        # The query string is part of the key.  An open-vocabulary detector
        # returns different boxes for different text on the same frame, so a
        # key without it would hand the second character of a story the first
        # character's box -- silently, and only on multi-character cases.
        return (id(frame), query, profile.name, profile.box_threshold, profile.text_threshold)

    @staticmethod
    def _grounding_text(query: str) -> str:
        # Grounding DINO's text encoder splits on ".", so one query per prompt
        # keeps every returned box attributable to that query without matching
        # the model's phrase output back to our wording.
        return f"{query.lower().strip('. ')}."

    @staticmethod
    def _top_box(result: Mapping[str, Any]) -> list[float] | None:
        """The highest-scoring box for this query, or None when nothing passed.

        ``post_process_grounded_object_detection`` renamed its label key from
        ``labels`` to ``text_labels``; the boxes and scores keys did not move.
        Reading the labels through both names keeps one detection path across
        the two transformers pins this repository resolves to (4.57.6 for
        ``gpu-renderer-public-metrics``, 4.51.3 for the sm120 variant).
        """

        labels = result.get("text_labels", result.get("labels"))
        if labels is None:
            raise ValueError(
                "grounded detection result carries neither text_labels nor labels"
            )
        scores = [float(value) for value in result["scores"]]
        if not scores:
            return None
        boxes = result["boxes"]
        best = max(range(len(scores)), key=lambda index: scores[index])
        return [float(value) for value in boxes[best].tolist()]

    def _predict_boxes(
        self,
        frames: Sequence[Any],
        query: str,
        profile: RoiDetectorProfile,
        *,
        detector_frame_batch_size: int,
    ) -> list[list[float] | None]:
        import torch

        batch_size = _positive_int(detector_frame_batch_size, "detector_frame_batch_size")
        keys = [self._detector_cache_key(frame, query, profile) for frame in frames]
        misses: list[Any] = []
        miss_keys: list[tuple[Any, ...]] = []
        for frame, key in zip(frames, keys, strict=True):
            if key not in self._detector_cache and key not in miss_keys:
                misses.append(frame)
                miss_keys.append(key)
        text = self._grounding_text(query)
        for start in range(0, len(misses), batch_size):
            chunk = misses[start : start + batch_size]
            chunk_keys = miss_keys[start : start + batch_size]
            from PIL import Image

            images = [Image.fromarray(frame) for frame in chunk]
            inputs = self.detector_processor(
                images=images, text=[text] * len(chunk), return_tensors="pt", padding=True
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.detector(**inputs)
            results = self.detector_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=profile.box_threshold,
                text_threshold=profile.text_threshold,
                target_sizes=[(int(frame.shape[0]), int(frame.shape[1])) for frame in chunk],
            )
            if len(results) != len(chunk):
                raise ValueError(
                    f"Grounding DINO returned {len(results)} results for a batch of {len(chunk)} frames"
                )
            for key, result in zip(chunk_keys, results, strict=True):
                self._detector_cache[key] = self._top_box(result)
        return [self._detector_cache[key] for key in keys]

    def _detections(
        self, frame: Any, queries: Sequence[str], profile: RoiDetectorProfile
    ) -> tuple[list[list[float]], list[str]]:
        """One box per query that detected, labelled with that query."""

        return self._detections_batch(
            [frame], queries, profile, detector_frame_batch_size=1
        )[0]

    def _detections_batch(
        self,
        frames: Sequence[Any],
        queries: Sequence[str],
        profile: RoiDetectorProfile,
        *,
        detector_frame_batch_size: int,
    ) -> list[tuple[list[list[float]], list[str]]]:
        ordered = tuple(queries)
        if len(set(ordered)) != len(ordered):
            raise ValueError("detector queries must be unique")
        per_query = {
            query: self._predict_boxes(
                frames,
                query,
                profile,
                detector_frame_batch_size=detector_frame_batch_size,
            )
            for query in ordered
        }
        rows: list[tuple[list[list[float]], list[str]]] = []
        for index in range(len(frames)):
            boxes: list[list[float]] = []
            labels: list[str] = []
            for query in ordered:
                box = per_query[query][index]
                if box is None:
                    continue
                boxes.append(box)
                labels.append(query)
            rows.append((boxes, labels))
        return rows

    def _run_sam(self, frame: Any, boxes: Sequence[Sequence[float]]) -> list[Any]:
        import torch

        inputs = self.sam_processor(images=frame, input_boxes=[list(boxes)], return_tensors="pt")
        original_sizes = inputs["original_sizes"]
        reshaped_sizes = inputs["reshaped_input_sizes"]
        inputs = {key: value.to(self.device) for key, value in inputs.items() if key not in {"original_sizes", "reshaped_input_sizes"}}
        with torch.no_grad():
            outputs = self.sam(**inputs, multimask_output=True)
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), original_sizes.cpu(), reshaped_sizes.cpu()
        )[0]
        scores = outputs.iou_scores.cpu()
        selected = []
        for index in range(len(boxes)):
            choice = int(scores[0, index].argmax())
            selected.append(masks[index, choice].numpy().astype(bool).squeeze())
        return selected

    def _masks(self, frame: Any, boxes: Sequence[Sequence[float]]) -> list[Any]:
        if not boxes:
            return []
        key = (id(frame), tuple(tuple(float(value) for value in box) for box in boxes))
        if key not in self._sam_cache:
            self._sam_cache[key] = self._run_sam(frame, boxes)
        return self._sam_cache[key]

    def _embed(self, image: Any) -> list[float]:
        import torch
        inputs = self.dino_processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            vector = self.dino(**inputs).last_hidden_state[:, 0, :].float()
            vector = vector / vector.norm(dim=-1, keepdim=True)
        return vector[0].cpu().tolist()

    def _embed_batch(
        self, images: Sequence[Any], *, dino_crop_batch_size: int
    ) -> list[list[float]]:
        import torch

        batch_size = _positive_int(dino_crop_batch_size, "dino_crop_batch_size")
        vectors: list[list[float]] = []
        for start in range(0, len(images), batch_size):
            chunk = list(images[start : start + batch_size])
            inputs = self.dino_processor(images=chunk, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                hidden = self.dino(**inputs).last_hidden_state[:, 0, :].float()
                hidden = hidden / hidden.norm(dim=-1, keepdim=True)
            if hidden.shape[0] != len(chunk):
                raise ValueError(
                    f"DINOv2 returned {int(hidden.shape[0])} embeddings for a batch of {len(chunk)} images"
                )
            vectors.extend(row for row in hidden.cpu().tolist())
        return vectors

    @staticmethod
    def _kept_mask_indices(
        masks: Sequence[Any], frame: Any, profile: RoiDetectorProfile
    ) -> list[int]:
        """Drop masks below the profile's minimum area share of the frame."""

        if profile.min_mask_area_ratio is None:
            return list(range(len(masks)))
        frame_area = int(frame.shape[0]) * int(frame.shape[1])
        if frame_area <= 0:
            raise ValueError("frame must have a positive pixel area")
        return [
            index
            for index in range(len(masks))
            if int(masks[index].sum()) / frame_area >= profile.min_mask_area_ratio
        ]

    @staticmethod
    def _character_canvas(frame: Any, mask: Any) -> Any:
        import numpy as np

        ys, xs = np.where(mask)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        crop = frame[y0:y1, x0:x1].copy()
        crop_mask = mask[y0:y1, x0:x1]
        crop[~crop_mask] = _IMAGENET_MEAN_RGB
        side = max(crop.shape[:2])
        canvas = np.empty((side, side, 3), dtype=np.uint8)
        canvas[:] = _IMAGENET_MEAN_RGB
        oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
        canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        return canvas

    def characters(
        self,
        frames: Sequence[Any],
        queries: Sequence[str],
        profile: RoiDetectorProfile,
        *,
        detector_frame_batch_size: int,
        dino_crop_batch_size: int,
    ) -> list[dict[str, tuple[list[float] | None, Mapping[str, Any]]]]:
        """Per frame, per query: the subject embedding or an explicit miss.

        One query yields at most one box, so the mask kept for that query is
        the only candidate and there is nothing to choose between.
        """

        from PIL import Image

        detections = self._detections_batch(
            frames,
            queries,
            profile,
            detector_frame_batch_size=detector_frame_batch_size,
        )
        results: list[dict[str, tuple[list[float] | None, Mapping[str, Any]]]] = []
        crops: list[Any] = []
        crop_slots: list[tuple[int, str]] = []
        for frame, (boxes, labels) in zip(frames, detections, strict=True):
            masks = self._masks(frame, boxes)
            kept = self._kept_mask_indices(masks, frame, profile)
            row: dict[str, tuple[list[float] | None, Mapping[str, Any]]] = {
                query: (None, {"detected": False, "query": query,
                               "roi_detector": profile.payload()})
                for query in queries
            }
            for index in kept:
                mask = masks[index]
                query = labels[index]
                crops.append(Image.fromarray(self._character_canvas(frame, mask)))
                crop_slots.append((len(results), query))
                row[query] = (None, {
                    "detected": True, "query": query, "box": boxes[index],
                    "mask_area": int(mask.sum()), "roi_detector": profile.payload(),
                })
            results.append(row)
        embeddings = (
            self._embed_batch(crops, dino_crop_batch_size=dino_crop_batch_size)
            if crops else []
        )
        if len(embeddings) != len(crops):
            raise ValueError(
                f"DINOv2 returned {len(embeddings)} embeddings for {len(crops)} character crops"
            )
        for (slot, query), embedding in zip(crop_slots, embeddings, strict=True):
            _, diagnostic = results[slot][query]
            results[slot][query] = (embedding, diagnostic)
        return results

    def grounded_regions(
        self,
        frames: Sequence[Any],
        phrases: Sequence[str],
        profile: RoiDetectorProfile,
        *,
        detector_frame_batch_size: int,
        dino_crop_batch_size: int,
    ) -> list[list[dict[str, Any]]]:
        """Per frame, **every** box the detector returns for a multi-phrase prompt.

        The prompt is the phrases, each formatted by :meth:`_grounding_text`,
        joined into one text.  Each region carries the detector score (the
        maximum over all text tokens, which is what selects the box -- the same
        rule ``post_process_grounded_object_detection`` applies), one score per
        phrase (the maximum over that phrase's tokens, used by the
        subject-gated column to decide which description a box belongs to), the
        SAM mask area, and the DINOv2 embedding of the same masked canvas
        :meth:`characters` embeds.  The frozen columns never call this method.
        """

        import torch
        from PIL import Image

        texts = [self._grounding_text(phrase) for phrase in phrases]
        if not texts or len(set(texts)) != len(texts):
            raise ValueError(f"phrases must be non-empty and distinct after formatting: {texts}")
        if any("." in text[:-1] for text in texts):
            raise ValueError(f"a phrase may not contain '.', the detector's phrase separator: {texts}")
        prompt = " ".join(texts)
        separator = self.detector_processor.tokenizer.convert_tokens_to_ids(".")
        batch_size = _positive_int(detector_frame_batch_size, "detector_frame_batch_size")
        detections: list[tuple[list[list[float]], list[float], list[list[float]]]] = []
        for start in range(0, len(frames), batch_size):
            chunk = list(frames[start : start + batch_size])
            inputs = self.detector_processor(
                images=[Image.fromarray(frame) for frame in chunk],
                text=[prompt] * len(chunk),
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.detector(**inputs)
            results = self.detector_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=profile.box_threshold,
                text_threshold=profile.text_threshold,
                target_sizes=[(int(frame.shape[0]), int(frame.shape[1])) for frame in chunk],
            )
            if len(results) != len(chunk):
                raise ValueError(
                    f"Grounding DINO returned {len(results)} results for a batch of {len(chunk)} frames"
                )
            probs = outputs.logits.sigmoid()
            for index, result in enumerate(results):
                ids = inputs["input_ids"][index].tolist()
                cuts = [position for position, token in enumerate(ids) if token == separator]
                if len(cuts) != len(texts):
                    raise ValueError(f"prompt tokenised into {len(cuts)} phrases, expected {len(texts)}")
                spans = [range(start_ + 1, cut) for start_, cut in zip([0, *cuts[:-1]], cuts)]
                kept = probs[index][probs[index].max(dim=-1).values > profile.box_threshold]
                if kept.shape[0] != len(result["scores"]):
                    raise ValueError("per-phrase scores do not align with the detector's kept boxes")
                phrase_scores = torch.stack(
                    [kept[:, list(span)].max(dim=-1).values for span in spans], dim=1
                ) if kept.shape[0] else torch.empty((0, len(texts)))
                detections.append((
                    [[float(value) for value in box.tolist()] for box in result["boxes"]],
                    [float(value) for value in result["scores"]],
                    [[float(value) for value in row] for row in phrase_scores.tolist()],
                ))
        regions: list[list[dict[str, Any]]] = []
        crops: list[Any] = []
        slots: list[tuple[int, int]] = []
        for frame_index, (frame, (boxes, scores, phrase_rows)) in enumerate(zip(frames, detections, strict=True)):
            frame_area = int(frame.shape[0]) * int(frame.shape[1])
            masks = self._masks(frame, boxes)
            rows: list[dict[str, Any]] = []
            for box, score, phrase_row, mask in zip(boxes, scores, phrase_rows, masks, strict=True):
                area = int(mask.sum())
                row = {"box": box, "score": score, "phrase_scores": phrase_row,
                       "mask_area_ratio": area / frame_area, "embedding": None}
                if area > 0:
                    crops.append(Image.fromarray(self._character_canvas(frame, mask)))
                    slots.append((frame_index, len(rows)))
                rows.append(row)
            regions.append(rows)
        embeddings = self._embed_batch(crops, dino_crop_batch_size=dino_crop_batch_size) if crops else []
        if len(embeddings) != len(crops):
            raise ValueError(f"DINOv2 returned {len(embeddings)} embeddings for {len(crops)} region crops")
        for (frame_index, row_index), embedding in zip(slots, embeddings, strict=True):
            regions[frame_index][row_index]["embedding"] = embedding
        return regions

    @staticmethod
    def _scene_background(frame: Any, masks: Sequence[Any]) -> tuple[Any, int]:
        import numpy as np

        background = frame.copy()
        if masks:
            union = np.logical_or.reduce(masks)
            background[union] = _IMAGENET_MEAN_RGB
            area = int(union.sum())
        else:
            area = 0
        return background, area

    def scenes(
        self,
        frames: Sequence[Any],
        queries: Sequence[str],
        profile: RoiDetectorProfile,
        *,
        detector_frame_batch_size: int,
        dino_crop_batch_size: int,
    ) -> list[tuple[list[float], Mapping[str, Any]]]:
        """Whole frame with every detected character masked out, then DINOv2.

        A frame where no query detected keeps all its pixels and still scores:
        the scene column does not need the subject found, it only subtracts the
        foreground it did find.
        """

        from PIL import Image

        detections = self._detections_batch(
            frames,
            queries,
            profile,
            detector_frame_batch_size=detector_frame_batch_size,
        )
        backgrounds: list[Any] = []
        diagnostics: list[dict[str, Any]] = []
        for frame, (boxes, labels) in zip(frames, detections, strict=True):
            detected = self._masks(frame, boxes)
            kept = self._kept_mask_indices(detected, frame, profile)
            masks = [detected[index] for index in kept]
            kept_labels = [labels[index] for index in kept]
            background, area = self._scene_background(frame, masks)
            backgrounds.append(Image.fromarray(background))
            diagnostics.append({
                "masked_instance_count": len(masks), "masked_queries": kept_labels,
                "mask_area": area, "roi_detector": profile.payload(),
            })
        embeddings = self._embed_batch(
            backgrounds, dino_crop_batch_size=dino_crop_batch_size
        )
        if len(embeddings) != len(backgrounds):
            raise ValueError(
                f"DINOv2 returned {len(embeddings)} embeddings for {len(backgrounds)} scene images"
            )
        return list(zip(embeddings, diagnostics, strict=True))


def _tensorflow_device_name(device: str) -> str:
    if device == "cpu":
        return "/CPU:0"
    if device == "cuda":
        return "/GPU:0"
    if device.startswith("cuda:"):
        index = device.split(":", 1)[1]
        if not index.isdigit():
            raise ValueError(f"unsupported TransNet device: {device!r}")
        return f"/GPU:{index}"
    raise ValueError(f"unsupported TransNet device: {device!r}")


def _tensorflow_remediation(device: str) -> str:
    """How to fix a TensorFlow-sees-no-GPU run, in the terms of this checkout.

    Names the actual interpreter in use rather than an env name: which
    public-metrics env this is depends on the leased card's architecture
    (e.g. sm_80, sm_120), so a hard-wired name is right on exactly
    one of them.
    """

    interpreter = Path(sys.executable)
    return (
        f"TransNetV2 requested {device} but TensorFlow sees 0 GPU(s). "
        "TensorFlow finds CUDA through this environment's own nvidia/*/lib "
        "directories, which are not on LD_LIBRARY_PATH. Seed it before starting "
        "python -- the dynamic loader reads LD_LIBRARY_PATH at process start, so "
        "setting it from inside this process would not take effect:\n"
        "  source run/lib/tensorflow_libs.sh\n"
        f'  mspc_seed_tensorflow_libs "{interpreter}"'
    )


def _bind_tensorflow_device(tf: Any, device: str) -> str:
    target = _tensorflow_device_name(device)
    gpus = list(tf.config.list_physical_devices("GPU"))
    if target == "/CPU:0":
        tf.config.set_visible_devices([], "GPU")
        return target
    gpu_index = int(target.rsplit(":", 1)[1])
    if gpu_index >= len(gpus):
        if not gpus:
            # The overwhelmingly common cause, and the one with a concrete fix.
            raise RuntimeError(_tensorflow_remediation(device))
        raise RuntimeError(
            f"TransNetV2 requested {device} but TensorFlow sees {len(gpus)} GPU(s)"
        )
    selected = gpus[gpu_index]
    tf.config.set_visible_devices([selected], "GPU")
    tf.config.experimental.set_memory_growth(selected, True)
    tf.config.set_soft_device_placement(False)
    return "/GPU:0"


def preflight_tensorflow_gpu(device: str) -> None:
    """Refuse a GPU TransNet run when this process cannot see a TensorFlow GPU.

    Same strength as the historical in-process OfficialMetricBackend.preflight:
    TensorFlow must be importable and ``list_physical_devices("GPU")`` must be
    non-empty.  It never repairs LD_LIBRARY_PATH and never falls back to CPU.
    The TF-only worker calls this; the parent scoring process must not.
    """

    if device == "cpu":
        return
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            f"TensorFlow is not importable under {sys.executable}: {error}"
        ) from error
    if not list(tf.config.list_physical_devices("GPU")):
        raise RuntimeError(_tensorflow_remediation(device))


def _load_env_registry() -> tuple[Path, dict[str, Any]]:
    raw = os.environ.get("MSPC_ENV_REGISTRY")
    path = (
        Path(raw).expanduser().resolve()
        if raw
        else (REPO_ROOT / "configs/runtime/env_by_arch.json")
    )
    if not path.is_file():
        raise RuntimeError(f"env registry is not readable: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != _ENV_REGISTRY_SCHEMA:
        raise RuntimeError(f"env registry {path} is not {_ENV_REGISTRY_SCHEMA}")
    return path, payload


def _shared_root_from_registry(registry: Mapping[str, Any], registry_path: Path) -> Path:
    name = registry.get("shared_root_env")
    if not isinstance(name, str) or not name:
        raise RuntimeError(f"{registry_path} declares no shared_root_env")
    raw = os.environ.get(name)
    if not raw or not str(raw).strip():
        raise RuntimeError(
            f"{name} is unset, so {TRANSNET_ROLE} cannot be resolved from {registry_path}; "
            f"export {TRANSNET_PYTHON_ENV} with the TF-only interpreter, or export {name}"
        )
    path = Path(str(raw).strip()).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{name} must be an absolute path, got {raw!r}")
    return path


def _arch_from_parent_metrics_interpreter(
    registry: Mapping[str, Any], shared_root: Path
) -> str | None:
    parent = _venv_entry_python(Path(sys.executable))
    role = registry.get("roles", {}).get("renderer_public_metrics")
    if not isinstance(role, dict):
        return None
    envs = role.get("envs")
    if not isinstance(envs, dict):
        return None
    for arch, entry in envs.items():
        if not isinstance(arch, str) or not isinstance(entry, dict):
            continue
        relative = entry.get("python")
        if not isinstance(relative, str) or not relative:
            continue
        if _venv_entry_python(shared_root / relative) == parent:
            return arch
    return None


def _arch_for_transnet(registry: Mapping[str, Any], shared_root: Path) -> str:
    arch = os.environ.get("MSPC_GPU_ARCH") or os.environ.get("MSPC_GPU_ARCH_OVERRIDE")
    if arch and str(arch).strip():
        return str(arch).strip()
    inferred = _arch_from_parent_metrics_interpreter(registry, shared_root)
    if inferred:
        return inferred
    raise RuntimeError(
        f"cannot resolve {TRANSNET_ROLE} architecture: set {TRANSNET_PYTHON_ENV} "
        "to the TF-only interpreter, or set MSPC_GPU_ARCH"
    )


def _venv_entry_python(path: Path) -> Path:
    """Keep the venv ``bin/python`` entry; do not follow that file's symlink.

    uv-managed venvs point ``bin/python`` at a shared CPython under
    ``env/uv_python_*``.  ``Path.resolve()`` follows that link, so the process
    starts as the base interpreter and drops the venv ``site-packages`` —
    including ``nvidia/*/lib``.  Canonicalize only the parent directory.
    """

    path = path.expanduser()
    parent = path.parent
    if not parent.is_absolute():
        parent = Path.cwd() / parent
    return parent.resolve() / path.name


def _interpreter_identity(python: Path) -> str:
    real = Path(os.path.realpath(python))
    if real == python:
        return str(python)
    return f"{python} (realpath={real})"


def resolve_transnet_python() -> Path:
    """Return the TF-only interpreter. Fail loud; never fall back to this process."""

    explicit = os.environ.get(TRANSNET_PYTHON_ENV, "").strip()
    if explicit:
        path = _venv_entry_python(Path(explicit))
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(
                f"{TRANSNET_PYTHON_ENV} is set to {explicit} but the venv entry "
                f"{_interpreter_identity(path)} is not an executable file"
            )
        return path
    registry_path, registry = _load_env_registry()
    shared_root = _shared_root_from_registry(registry, registry_path)
    arch = _arch_for_transnet(registry, shared_root)
    roles = registry.get("roles")
    if not isinstance(roles, dict) or TRANSNET_ROLE not in roles:
        raise RuntimeError(
            f"role {TRANSNET_ROLE} is not registered in {registry_path}"
        )
    role = roles[TRANSNET_ROLE]
    envs = role.get("envs") if isinstance(role, dict) else None
    if not isinstance(envs, dict) or arch not in envs:
        note = ""
        if isinstance(role, dict):
            notes = role.get("unregistered_arch_notes", {})
            if isinstance(notes, dict) and isinstance(notes.get(arch), str):
                note = f"; registry note: {notes[arch]}"
        registered = ", ".join(sorted(envs)) if isinstance(envs, dict) else ""
        raise RuntimeError(
            f"role {TRANSNET_ROLE} has no interpreter registered for {arch} in "
            f"{registry_path}; registered for this role: {registered}{note}"
        )
    relative = envs[arch].get("python") if isinstance(envs[arch], dict) else None
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(
            f"{registry_path}: role {TRANSNET_ROLE} arch {arch} declares no python"
        )
    if Path(relative).is_absolute():
        raise RuntimeError(
            f"{registry_path}: role {TRANSNET_ROLE} arch {arch} python must be "
            "relative to the shared root"
        )
    path = _venv_entry_python(shared_root / relative)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(
            f"the registry names {relative} for {TRANSNET_ROLE}/{arch}, but "
            f"{_interpreter_identity(path)} is not an executable file; the "
            "TF-only env may not be bootstrapped yet"
        )
    return path


def nvidia_lib_dirs_from_interpreter(python: Path) -> list[Path]:
    """Ask the interpreter which package dirs it can see, including chained .pth."""

    script = (
        "import site, sys\n"
        "paths = []\n"
        "for candidate in (*site.getsitepackages(), *sys.path):\n"
        "    if candidate and candidate not in paths:\n"
        "        paths.append(candidate)\n"
        "print('\\n'.join(paths))\n"
    )
    completed = subprocess.run(
        [str(python), "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or (
            f"exit {completed.returncode}"
        )
        raise RuntimeError(f"{python} could not report site-packages: {detail}")
    dirs: list[Path] = []
    for line in completed.stdout.splitlines():
        site_dir = line.strip()
        if not site_dir:
            continue
        dirs.extend(
            sorted(path for path in Path(site_dir).glob("nvidia/*/lib") if path.is_dir())
        )
    if not dirs:
        raise RuntimeError(
            "no nvidia/*/lib dirs in site-packages reported by "
            f"{_interpreter_identity(python)}; TransNetV2 would see 0 GPUs. "
            "The invoked path must be the venv entry (so site-packages stay "
            "the TF-only env), not a realpath into uv_python_*."
        )
    return dirs


def seed_tensorflow_ld_library_path(env: Mapping[str, str], python: Path) -> dict[str, str]:
    seeded = dict(env)
    nvlibs = ":".join(str(path) for path in nvidia_lib_dirs_from_interpreter(python))
    current = seeded.get("LD_LIBRARY_PATH") or "/usr/local/cuda/lib64"
    seeded["LD_LIBRARY_PATH"] = f"{nvlibs}:{current}"
    return seeded


def _parse_transnet_worker_payload(
    stdout: str, *, python: Path, stderr: str
) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        extra = f"; stderr: {stderr.strip()}" if stderr.strip() else ""
        raise RuntimeError(
            "TransNet TF-only worker under "
            f"{_interpreter_identity(python)} wrote no JSON on stdout{extra}"
        )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "TransNet TF-only worker under "
            f"{_interpreter_identity(python)} wrote invalid JSON: {error}; "
            f"stdout={text!r}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError(
            "TransNet TF-only worker under "
            f"{_interpreter_identity(python)} must write a JSON object, "
            f"got {type(payload).__name__}"
        )
    return payload


def invoke_transnet_worker(*, python: Path, args: Sequence[str]) -> dict[str, Any]:
    if not TRANSNET_WORKER.is_file():
        raise FileNotFoundError(f"TransNet worker is missing: {TRANSNET_WORKER}")
    env = seed_tensorflow_ld_library_path(os.environ, python)
    src = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else f"{src}{os.pathsep}{existing}"
    completed = subprocess.run(
        [str(python), str(TRANSNET_WORKER), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    payload = _parse_transnet_worker_payload(
        completed.stdout, python=python, stderr=completed.stderr
    )
    if completed.returncode != 0 or payload.get("status") != "ok":
        message = payload.get("error")
        detail = (
            message
            if isinstance(message, str) and message.strip()
            else completed.stderr.strip() or completed.stdout.strip() or (
                f"exit {completed.returncode}"
            )
        )
        raise RuntimeError(
            "TransNet TF-only worker failed under "
            f"{_interpreter_identity(python)}: {detail}"
        )
    return payload


class _TransNetEngine:
    def __init__(self, root: Path, device: str):
        import tensorflow as tf
        self.tf_device = _bind_tensorflow_device(tf, device)
        with tf.device(self.tf_device):
            self.model = tf.saved_model.load(str(_existing_dir(root, "transnetv2")))

    def predict(self, frames: Sequence[Any]) -> list[float]:
        import cv2
        import numpy as np
        import tensorflow as tf
        resized = np.stack([cv2.resize(frame, (48, 27)) for frame in frames])
        start_pad = 25
        end_pad = 25 + 50 - (len(resized) % 50 if len(resized) % 50 else 50)
        padded = np.concatenate(([resized[0]] * start_pad, resized, [resized[-1]] * end_pad))
        outputs = []
        with tf.device(self.tf_device):
            for pointer in range(0, len(padded) - 99, 50):
                logits, _ = self.model(tf.cast(padded[None, pointer:pointer + 100], tf.float32))
                outputs.extend(tf.sigmoid(logits)[0, 25:75, 0].numpy().tolist())
        return [float(value) for value in outputs[:len(resized)]]


class _SubprocessTransNetEngine:
    """Run ``_TransNetEngine`` in a TF-only interpreter that does not install torch.

    The parent scoring process keeps the same ``transnet_factory(root, device)``
    contract and still calls ``.predict(frames)``.  Frames cross the process
    boundary as a uint8 NHWC ``.npy`` (decoded RGB, not a video path) so the
    child's detection sees the same arrays the parent already validated against
    shot counts.  Child failures surface as ``RuntimeError`` with the worker's
    error text; they are never turned into N/A or 0.
    """

    def __init__(self, root: Path, device: str):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"resource root does not exist: {self.root}")
        self.device = device
        self.python = resolve_transnet_python()

    @classmethod
    def preflight_device(cls, device: str) -> None:
        invoke_transnet_worker(
            python=resolve_transnet_python(),
            args=["--mode", "preflight", "--device", device],
        )

    def predict(self, frames: Sequence[Any]) -> list[float]:
        import numpy as np

        stacked = np.stack([np.asarray(frame) for frame in frames])
        if stacked.ndim != 4 or stacked.shape[-1] != 3:
            raise ValueError(
                f"TransNet frames must be NHWC RGB, got shape {getattr(stacked, 'shape', None)}"
            )
        with tempfile.TemporaryDirectory(prefix="mspc-transnet-") as tmp:
            frames_path = Path(tmp) / "frames.npy"
            np.save(frames_path, stacked, allow_pickle=False)
            payload = invoke_transnet_worker(
                python=self.python,
                args=[
                    "--mode",
                    "predict",
                    "--device",
                    self.device,
                    "--resource-root",
                    str(self.root),
                    "--frames-npy",
                    str(frames_path),
                ],
            )
        probabilities = payload.get("probabilities")
        if not isinstance(probabilities, list) or not probabilities:
            raise RuntimeError(
                f"TransNet TF-only worker under {self.python} did not return a probability list"
            )
        values = [float(value) for value in probabilities]
        if len(values) != len(frames):
            raise RuntimeError(
                "TransNet TF-only worker returned a probability count different "
                f"from decoded frames: {len(values)} != {len(frames)}"
            )
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError("TransNet TF-only worker returned a non-finite probability")
        return values


def _model_batch_from_roi_config(config_path: Path) -> ModelBatchSizes:
    source = Path(config_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"ROI detector config does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"public metrics config must be an object: {source}")
    if "model_batch" not in payload:
        raise ValueError("config.model_batch is missing")
    return parse_model_batch_sizes(payload["model_batch"], label="config.model_batch")


@dataclass
class OfficialMetricBackend:
    """Lazy metric dispatcher; optional factories make CPU contract tests lightweight."""

    decoder: Callable[[Path], tuple[list[Any], float]] = _decode_rgb
    videoscore_factory: Callable[[Path, str], Any] = _VideoScoreEngine
    viclip_factory: Callable[[Path, str], Any] = _ViCLIPEngine
    dino_factory: Callable[[Path, str], Any] = _DinoRoiEngine
    vbench_factory: Callable[[Path, str], Any] = _SubprocessVBenchEngine
    transnet_factory: Callable[[Path, str], Any] = _SubprocessTransNetEngine
    device: str = "cuda"
    roi_detector_config: Path = DEFAULT_ROI_DETECTOR_CONFIG
    model_batch: ModelBatchSizes | None = None

    def __post_init__(self) -> None:
        if self.model_batch is None:
            self.model_batch = _model_batch_from_roi_config(self.roi_detector_config)
        elif type(self.model_batch) is not ModelBatchSizes:
            raise TypeError("model_batch must be a ModelBatchSizes")
        self._engines: dict[str, Any] = {}
        self._decoded_key: tuple[str, int, int] | None = None
        self._decoded_value: tuple[list[Any], float] | None = None
        self._roi_contract: RoiDetectorContract | None = None
        self._roi_cache: dict[tuple[Any, ...], tuple[list[float] | None, Mapping[str, Any]]] = {}
        self._videoscore_key: tuple[Any, ...] | None = None
        self._videoscore_value: tuple[float, Mapping[str, float]] | None = None
        self._vbench_key: tuple[Any, ...] | None = None
        self._vbench_value: Mapping[str, float] | None = None
        self._transnet_key: tuple[str, int, int] | None = None
        self._transnet_value: tuple[list[float], tuple[int, ...], tuple[int, ...]] | None = None

    def _frames(self, video_path: Path) -> tuple[list[Any], float]:
        resolved = video_path.expanduser().resolve()
        stat = resolved.stat()
        key = (str(resolved), stat.st_size, stat.st_mtime_ns)
        if key != self._decoded_key:
            self._decoded_value = self.decoder(resolved)
            self._decoded_key = key
            # Per-video caches follow the decoded frames so a long run keeps at
            # most one case in memory.
            self._roi_cache.clear()
            self._videoscore_key = None
            self._videoscore_value = None
            self._vbench_key = None
            self._vbench_value = None
            self._transnet_key = None
            self._transnet_value = None
            engine = self._engines.get("dino")
            if engine is not None:
                engine.clear_frame_caches()
        if self._decoded_value is None:
            raise RuntimeError("video decoder did not populate its one-case cache")
        return self._decoded_value

    def _engine(self, name: str, case: PublicMetricCase) -> Any:
        if name not in self._engines:
            root = case.resource_root.expanduser().resolve()
            if not root.is_dir():
                raise FileNotFoundError(f"resource root does not exist: {root}")
            factory = {"videoscore": self.videoscore_factory, "viclip": self.viclip_factory,
                       "dino": self.dino_factory, "vbench": self.vbench_factory,
                       "transnet": self.transnet_factory}[name]
            # Every engine but the quality one runs on this process's device.
            # That one runs on the device its role declares, which is not the
            # card the rest use -- see ``device_reason`` in the registry.  The
            # declaration is read here rather than inside the engine so that a
            # substituted factory in a test still receives a device it can act
            # on, and so the one place that knows "which engine goes where"
            # says it once.
            device = quality_role_device() if name == "vbench" else self.device
            self._engines[name] = factory(root, device)
        return self._engines[name]

    def preflight(
        self, metric_ids: Sequence[str], *, resource_root: Path | None = None
    ) -> None:
        """Refuse a run that would die at the scoring step, before any decoding.

        The TransNet and pyiqa engines are both built lazily, inside the
        per-case loop, so a broken subprocess env surfaces only after every
        video has been decoded -- and, when the sidecar drives this, after
        every video has been *generated*.  This turns that into a refusal
        measured in seconds.

        It only refuses; it never repairs.  For TransNet, LD_LIBRARY_PATH is
        read by the dynamic loader at process start, so assigning
        ``os.environ`` here would not reliably reach a later ``dlopen`` -- an
        auto-fix would be a silent fallback that works on some hosts and
        quietly does nothing on others.

        No-op unless a selected metric actually needs the corresponding GPU
        subprocess, so diagnostic-only runs and ``device="cpu"`` runs pay
        nothing.
        """

        if any(metric_id in TENSORFLOW_METRIC_IDS for metric_id in metric_ids) and self.device != "cpu":
            if self.transnet_factory is _SubprocessTransNetEngine:
                # Production path: the parent never imports TensorFlow. The
                # child runs preflight_tensorflow_gpu at the same strength as
                # the old in-process check.
                _SubprocessTransNetEngine.preflight_device(self.device)
            elif self.transnet_factory is _TransNetEngine:
                # In-process engine, used only by the TF-only worker.
                preflight_tensorflow_gpu(self.device)
            # A substituted factory owns its own device story; the CPU
            # contract tests rely on never importing TensorFlow at all.

        if any(metric_id in V3_QUALITY_METRIC_IDS for metric_id in metric_ids):
            # Unlike the TensorFlow block above, this one does **not** key off
            # the scoring process's own device.  These two columns run on the
            # device their role declares, which is not the card the other
            # columns use (see ``device_reason`` in the registry), so asking
            # "is the parent on a GPU?" answers a question about someone else.
            quality_device = quality_role_device()
            if self.vbench_factory is _SubprocessVBenchEngine:
                # Production path: the parent never imports pyiqa/torch from
                # the renderer_quality role's interpreter.
                _SubprocessVBenchEngine.preflight_device(quality_device)
            elif self.vbench_factory is _PyIQAEngine:
                # In-process engine, used only by the quality worker.
                preflight_quality_gpu(quality_device)
            # A substituted factory owns its own device story; the CPU
            # contract tests rely on never importing pyiqa at all.

    def _roi_profile(self, metric_id: str) -> RoiDetectorProfile:
        if metric_id in _FROZEN_ROI_METRIC_IDS:
            return FROZEN_PUBLIC_ROI_DETECTOR
        if self._roi_contract is None:
            self._roi_contract = load_roi_detector_contract(self.roi_detector_config)
        return self._roi_contract.profile_for(metric_id)

    @staticmethod
    def _character_queries(case: PublicMetricCase) -> tuple[str, ...]:
        """The declared open-vocabulary queries, or a refusal.

        ``None`` means the scoring input never carried the field.  Reading that
        as "this story has no characters" would turn a missing producer-side
        join into an arm-wide identity column of nulls that still looks like a
        finished run, so it raises here instead.
        """

        if case.character_queries is None:
            raise ValueError(
                f"{case.case_id}: character_queries is required for the open-vocabulary "
                "identity and scene columns; the scoring input declared none"
            )
        return tuple(case.character_queries)

    def _vbench(
        self, case: PublicMetricCase, frames: Sequence[Any], counts: Sequence[int],
    ) -> Mapping[str, float]:
        """Score both quality dimensions once per decoded video."""

        if self._vbench_key != self._decoded_key or self._vbench_value is None:
            shots: list[list[Any]] = []
            start = 0
            for count in counts:
                shots.append(list(frames[start : start + count]))
                start += count
            self._vbench_value = self._engine("vbench", case).score(shots)
            self._vbench_key = self._decoded_key
        return self._vbench_value


    def _video_score(
        self, case: PublicMetricCase, frames: Sequence[Any]
    ) -> tuple[float, Mapping[str, float]]:
        """Score once per decoded video so the aspect columns never re-run it."""

        key = (self._decoded_key, case.story_prompt)
        if key != self._videoscore_key or self._videoscore_value is None:
            self._videoscore_value = self._engine("videoscore", case).score(
                frames, case.story_prompt
            )
            self._videoscore_key = key
        value, aspects = self._videoscore_value
        return float(value), aspects

    def _transnet_detection(
        self, case: PublicMetricCase, frames: Sequence[Any]
    ) -> tuple[list[float], tuple[int, ...], tuple[int, ...]]:
        """Detect cuts once per decoded video so every TransNet column shares one pass.

        Returns the per-frame probabilities, the positive-run peaks, and the
        cut boundaries those peaks mean (see ``cut_boundaries_from_peaks``);
        every column matches the boundaries, never the raw peaks.
        """

        if self._transnet_key != self._decoded_key or self._transnet_value is None:
            probabilities = self._engine("transnet", case).predict(frames)
            if len(probabilities) != len(frames):
                raise ValueError(
                    "TransNetV2 returned a probability count different from decoded frames"
                )
            peaks = transition_run_peaks(probabilities, threshold=0.5)
            detected = cut_boundaries_from_peaks(peaks, frame_count=len(frames))
            self._transnet_value = (probabilities, peaks, detected)
            self._transnet_key = self._decoded_key
        return self._transnet_value

    def _keyframe_indices(
        self, case: PublicMetricCase, counts: Sequence[int]
    ) -> dict[str, tuple[int, ...]]:
        return dict(zip(
            [str(shot["shot_id"]) for shot in case.shots],
            first_mid_last_frame_indices(list(counts)),
            strict=True,
        ))

    def _character_rois(
        self,
        case: PublicMetricCase,
        frames: Sequence[Any],
        counts: Sequence[int],
        *,
        shot_ids: Sequence[str],
        queries: Sequence[str],
        profile: RoiDetectorProfile,
    ) -> dict[str, list[dict[str, tuple[list[float] | None, dict[str, Any]]]]]:
        """Subject ROIs per keyframe and query, memoized per video and profile."""

        ordered = tuple(queries)
        indices_by_shot = self._keyframe_indices(case, counts)
        engine = self._engine("dino", case)
        pending: list[int] = []
        seen: set[int] = set()
        for shot_id in shot_ids:
            if shot_id not in indices_by_shot:
                raise ValueError(f"pair manifest references unknown shot {shot_id!r}")
            for index in indices_by_shot[shot_id]:
                key = (self._decoded_key, "character", ordered, profile.name, index)
                if key not in self._roi_cache and index not in seen:
                    pending.append(index)
                    seen.add(index)
        if pending:
            results = engine.characters(
                [frames[index] for index in pending],
                ordered,
                profile,
                detector_frame_batch_size=self.model_batch.detector_frame_batch_size,
                dino_crop_batch_size=self.model_batch.dino_crop_batch_size,
            )
            if len(results) != len(pending):
                raise ValueError(
                    f"dino characters returned {len(results)} results for {len(pending)} frames"
                )
            for index, result in zip(pending, results, strict=True):
                missing = [query for query in ordered if query not in result]
                if missing:
                    raise ValueError(f"dino characters omitted queries {missing}")
                self._roi_cache[
                    (self._decoded_key, "character", ordered, profile.name, index)
                ] = result
        rois: dict[str, list[dict[str, tuple[list[float] | None, dict[str, Any]]]]] = {}
        for shot_id in shot_ids:
            rows: list[dict[str, tuple[list[float] | None, dict[str, Any]]]] = []
            for index in indices_by_shot[shot_id]:
                row = self._roi_cache[
                    (self._decoded_key, "character", ordered, profile.name, index)
                ]
                rows.append({
                    query: (embedding, {"frame_index": index, **dict(diagnostic)})
                    for query, (embedding, diagnostic) in row.items()
                })
            rois[shot_id] = rows
        return rois

    def _scene_rois(
        self,
        case: PublicMetricCase,
        frames: Sequence[Any],
        counts: Sequence[int],
        *,
        shot_ids: Sequence[str],
        queries: Sequence[str],
        profile: RoiDetectorProfile,
    ) -> dict[str, list[tuple[list[float], dict[str, Any]]]]:
        ordered = tuple(queries)
        indices_by_shot = self._keyframe_indices(case, counts)
        engine = self._engine("dino", case)
        pending: list[int] = []
        seen: set[int] = set()
        for shot_id in shot_ids:
            if shot_id not in indices_by_shot:
                raise ValueError(f"pair manifest references unknown shot {shot_id!r}")
            for index in indices_by_shot[shot_id]:
                key = (self._decoded_key, "scene", ordered, profile.name, index)
                if key not in self._roi_cache and index not in seen:
                    pending.append(index)
                    seen.add(index)
        if pending:
            results = engine.scenes(
                [frames[index] for index in pending],
                ordered,
                profile,
                detector_frame_batch_size=self.model_batch.detector_frame_batch_size,
                dino_crop_batch_size=self.model_batch.dino_crop_batch_size,
            )
            if len(results) != len(pending):
                raise ValueError(
                    f"dino scenes returned {len(results)} results for {len(pending)} frames"
                )
            for index, result in zip(pending, results, strict=True):
                self._roi_cache[
                    (self._decoded_key, "scene", ordered, profile.name, index)
                ] = result
        rois: dict[str, list[tuple[list[float], dict[str, Any]]]] = {}
        for shot_id in shot_ids:
            rows: list[tuple[list[float], dict[str, Any]]] = []
            for index in indices_by_shot[shot_id]:
                embedding, diagnostic = self._roi_cache[
                    (self._decoded_key, "scene", ordered, profile.name, index)
                ]
                rows.append((embedding, {"frame_index": index, **dict(diagnostic)}))
            rois[shot_id] = rows
        return rois

    def _identity(
        self,
        case: PublicMetricCase,
        frames: Sequence[Any],
        counts: Sequence[int],
        *,
        metric_id: str,
    ) -> tuple[IdentityReadout, dict[str, Any]]:
        row, _ = _load_pair_case(case)
        pairs = _shot_pairs(row, "character")
        if pairs is None:
            raise ValueError("character pairs must be eligible for the identity column")
        queries = self._character_queries(case)
        profile = self._roi_profile(metric_id)
        context: dict[str, Any] = {
            "roi_detector": profile.payload(),
            "character_queries": list(queries),
            "pair_source": row["pair_source"],
            "pair_count": len(pairs),
        }
        if not queries:
            # main_table_96 carries 21 stories whose only recurring subject is
            # an animal or an object, so the benchmark declares no character
            # asset for them.  Falling back to "person" would score a guinea
            # pig as a missing human and read protocol inapplicability as a bad
            # model, so the column is ineligible instead and the arm mean is
            # taken over the eligible stories only.
            return (
                IdentityReadout(
                    value=None, pairs=(), pairs_scored=0, pairs_total=0,
                    reason="ineligible_no_character_asset",
                ),
                context,
            )
        shot_ids = sorted({shot_id for pair in pairs for shot_id in pair})
        rois = self._character_rois(
            case, frames, counts, shot_ids=shot_ids, queries=queries, profile=profile,
        )
        embeddings = {
            shot_id: {
                query: [row_[query][0] for row_ in rows] for query in queries
            }
            for shot_id, rows in rois.items()
        }
        readout = _identity_readout(embeddings, pairs, queries)
        context["keyframes"] = {
            shot_id: [
                {query: diagnostic for query, (_, diagnostic) in row_.items()}
                for row_ in rows
            ]
            for shot_id, rows in rois.items()
        }
        return readout, context

    def _gated_character(
        self,
        case: PublicMetricCase,
        frames: Sequence[Any],
        counts: Sequence[int],
        *,
        metric_id: str,
    ) -> tuple[GatedCharacterReadout, dict[str, Any]]:
        row, subject_class = _load_pair_case(case)
        pairs = _shot_pairs(row, "character")
        if pairs is None:
            raise ValueError("character pairs must be eligible for the gated column")
        profile = self._roi_profile(metric_id)
        shot_ids = sorted({shot_id for pair in pairs for shot_id in pair})
        rois = self._character_rois(
            case, frames, counts,
            shot_ids=shot_ids, queries=(subject_class,), profile=profile,
        )
        embeddings = {
            shot_id: [row_[subject_class][0] for row_ in rows]
            for shot_id, rows in rois.items()
        }
        readout = _gated_character_readout(embeddings, pairs)
        keyframes = sum(len(rows) for rows in rois.values())
        detected = sum(
            1
            for rows in rois.values()
            for row_ in rows
            if row_[subject_class][0] is not None
        )
        context = {
            "subject_class": subject_class,
            "roi_detector": profile.payload(),
            "keyframes": {
                shot_id: [row_[subject_class][1] for row_ in rows]
                for shot_id, rows in rois.items()
            },
            "keyframe_count": keyframes,
            "detected_keyframe_count": detected,
            "n_pairs": readout.n_pairs,
            "n_pairs_detected": readout.n_pairs_detected,
            "detected_comparisons": readout.comparisons,
            "comparisons_per_pair": 9,
            "comparison_count": readout.comparisons_total,
        }
        return readout, context

    def __call__(self, metric: MetricMetadata, case: PublicMetricCase) -> MetricResult:
        if metric.metric_id not in _SUPPORTED_METRIC_IDS:
            raise ValueError(f"unsupported public metric: {metric.metric_id}")
        frames, fps = self._frames(case.video_path)
        counts = [int(shot["frame_count"]) for shot in case.shots]
        if len(frames) != sum(counts):
            raise ValueError(
                f"decoded frame count {len(frames)} differs from shot total {sum(counts)}"
            )
        if metric.metric_id == "video_score":
            value, aspects = self._video_score(case, frames)
            return MetricResult(value, {"aspect_scores": aspects}, {
                "sampled_frame_indices": list(_uniform_indices(len(frames), 48)),
                "aspect_order": list(_VIDEOSCORE_ASPECTS), "aggregation": "arithmetic_mean_5",
                "native_fps": fps,
            })
        if metric.metric_id in _VIDEOSCORE_ASPECT_METRICS:
            aspect = _VIDEOSCORE_ASPECT_METRICS[metric.metric_id]
            _, aspects = self._video_score(case, frames)
            if aspect not in aspects:
                raise ValueError(f"VideoScore-v1.1 did not report the {aspect} aspect")
            return MetricResult(float(aspects[aspect]), {"aspect_scores": aspects}, {
                "aspect": aspect, "source_metric_id": "video_score",
                "definition": "one VideoScore-v1.1 aspect head, reported without rescoring",
                "native_fps": fps,
            })
        if metric.metric_id in V3_QUALITY_METRIC_IDS:
            dimension = metric.metric_id.removeprefix("vbench_")
            scores = self._vbench(case, frames, counts)
            if dimension not in scores:
                raise ValueError(f"VBench did not report the {dimension} dimension")
            return MetricResult(float(scores[dimension]), {"dimension_scores": dict(scores)}, {
                "dimension": dimension,
                "granularity": "per_shot_clip_mean",
                "definition": (
                    "VBench custom_input score for this dimension over the case's "
                    "shot clips; a whole-story clip would charge every cut to "
                    "picture quality"
                ),
                "shot_count": len(counts),
                "native_fps": fps,
            })
        if metric.metric_id == "motion_magnitude":
            return _motion_magnitude(frames, counts, native_fps=fps)
        if metric.metric_id == "viclip_text_alignment":
            engine = self._engine("viclip", case)
            start, items, rows_meta = 0, [], []
            for shot in case.shots:
                indices = _eight_shot_indices(start, int(shot["frame_count"]))
                items.append(([frames[index] for index in indices], str(shot["prompt"])))
                rows_meta.append({"shot_id": shot["shot_id"], "frame_indices": list(indices)})
                start += int(shot["frame_count"])
            scores = engine.score_shots(
                items, viclip_shot_batch_size=self.model_batch.viclip_shot_batch_size
            )
            if len(scores) != len(items):
                raise ValueError(
                    f"ViCLIP returned {len(scores)} scores for {len(items)} shots"
                )
            rows = [
                {**meta, "cosine": float(score)}
                for meta, score in zip(rows_meta, scores, strict=True)
            ]
            return MetricResult(sum(row["cosine"] for row in rows) / len(rows), {"shots": rows}, {
                "shot_count": len(rows), "frames_per_shot": 8, "aggregation": "equal_shot_mean",
                "native_fps": fps,
            })
        if metric.metric_id in _IDENTITY_METRIC_IDS:
            readout, context = self._identity(
                case, frames, counts, metric_id=metric.metric_id
            )
            raw = {"pairs": [dict(row) for row in readout.pairs]}
            shared = {
                **context, "native_fps": fps,
                "identity_pairs_scored": readout.pairs_scored,
                "identity_pairs_total": readout.pairs_total,
            }
            if readout.reason is not None:
                shared["reason"] = readout.reason
            if metric.metric_id == "dinov2_character_identity":
                return MetricResult(readout.value, raw, {
                    **shared,
                    "definition": (
                        "mean over adjacent shot pairs of the mean over that pair's "
                        "characters of the 3x3 keyframe cosines; a (pair, character) "
                        "unit scores only when all six keyframes detected that "
                        "character; null when the story declares no character asset "
                        "or no unit scored"
                    ),
                })
            if metric.metric_id == "identity_pairs_scored":
                return MetricResult(readout.pairs_scored, raw, {
                    **shared,
                    "definition": (
                        "(shot pair x character) units that passed the "
                        "all-six-keyframes-detected rule; arm-dependent"
                    ),
                })
            if metric.metric_id == "identity_pairs_total":
                return MetricResult(readout.pairs_total, raw, {
                    **shared,
                    "definition": (
                        "adjacent shot pairs times declared characters; "
                        "arm-independent denominator of identity_pairs_scored"
                    ),
                })
            return MetricResult(readout.cases_scored, raw, {
                **shared,
                "definition": (
                    "1 when this story produced an identity score, 0 when it did "
                    "not; summing the column over an arm gives its scored case count"
                ),
            })
        if metric.metric_id in _FROZEN_ROI_METRIC_IDS:
            row, subject_class = _load_pair_case(case)
            kind = "character" if metric.metric_id.endswith("character_consistency") else "scene"
            pairs = _shot_pairs(row, kind)
            if pairs is None:
                reason = str(row[kind]["reason"])
                return MetricResult(None, {"pairs": []}, {
                    "reason": reason, "eligible_pair_count": 0,
                })
            # Frozen detector settings come from code, never from the config, so
            # a config edit can never move a published main-table number.
            profile = self._roi_profile(metric.metric_id)
            needed = sorted({shot_id for pair in pairs for shot_id in pair})
            if kind == "character":
                queries: tuple[str, ...] = (subject_class,)
                rois = self._character_rois(
                    case, frames, counts,
                    shot_ids=needed, queries=queries, profile=profile,
                )
                embeddings = {
                    shot_id: [row_[subject_class][0] for row_ in rows]
                    for shot_id, rows in rois.items()
                }
                frame_diagnostics = {
                    shot_id: [row_[subject_class][1] for row_ in rows]
                    for shot_id, rows in rois.items()
                }
            else:
                # The scene column subtracts every declared character of this
                # story, not one pre-registered detector class: the background
                # that remains is what "the same room" is measured on.
                queries = self._character_queries(case)
                rois = self._scene_rois(
                    case, frames, counts,
                    shot_ids=needed, queries=queries, profile=profile,
                )
                embeddings = {
                    shot_id: [embedding for embedding, _ in rows]
                    for shot_id, rows in rois.items()
                }
                frame_diagnostics = {
                    shot_id: [diagnostic for _, diagnostic in rows]
                    for shot_id, rows in rois.items()
                }
            result = _macro_with_missing(embeddings, pairs)
            keyframe_count = sum(len(values) for values in embeddings.values())
            detected_keyframes = sum(
                embedding is not None for values in embeddings.values() for embedding in values
            )
            return MetricResult(result.value, result.raw_outputs, {
                **dict(result.diagnostics), "subject_class": subject_class,
                "character_queries": list(queries),
                "keyframes": frame_diagnostics, "keyframe_count": keyframe_count,
                "detected_keyframe_count": detected_keyframes,
                "roi_detector": profile.payload(),
                "native_fps": fps,
                "pair_source": row["pair_source"],
            })
        if metric.metric_id in _GATED_CHARACTER_METRIC_IDS:
            readout, context = self._gated_character(
                case, frames, counts, metric_id=metric.metric_id
            )
            return MetricResult(readout.macro, {"pairs": [dict(row) for row in readout.pairs]}, {
                **context, "native_fps": fps,
                "definition": (
                    "macro mean over eligible shot pairs of the keyframe cosines "
                    "whose two ROIs were both detected; null when no pair kept a "
                    "single comparison"
                ),
                "zero_filling": False,
            })
        if metric.metric_id not in TENSORFLOW_METRIC_IDS:
            raise ValueError(f"unsupported public metric: {metric.metric_id}")
        probabilities, peaks, detected = self._transnet_detection(case, frames)
        reference = planned_shot_boundaries(counts)
        extras = {
            "single_frame_probabilities": probabilities,
            "positive_runs_reduced_to_peaks": list(peaks),
        }
        shared = {"threshold": 0.5, "strict_greater_than": True, "native_fps": fps}
        if metric.metric_id == "transnet_transition_deviation":
            result = transition_deviation(reference, detected, frame_count=len(frames))
            return MetricResult(
                result.value,
                {**dict(result.raw_outputs), **extras},
                {**dict(result.diagnostics), **shared},
            )
        if metric.metric_id == "transition_count_error":
            result = transition_count_error(reference, detected, frame_count=len(frames))
        else:
            result = matched_transition_timing_mae(
                reference, detected, frame_count=len(frames)
            )
        return MetricResult(
            result.value,
            {**dict(result.raw_outputs), **extras},
            {**dict(result.diagnostics), **shared},
        )


_DEFAULT_BACKEND = OfficialMetricBackend()
_BACKENDS_BY_ROI_CONFIG: dict[str, OfficialMetricBackend] = {}


def _backend_for_case(case: PublicMetricCase) -> OfficialMetricBackend:
    path = case.metric_config_path
    if path is None:
        return _DEFAULT_BACKEND
    resolved = Path(path).expanduser().resolve()
    default_resolved = Path(_DEFAULT_BACKEND.roi_detector_config).expanduser().resolve()
    if resolved == default_resolved:
        return _DEFAULT_BACKEND
    key = str(resolved)
    backend = _BACKENDS_BY_ROI_CONFIG.get(key)
    if backend is None:
        backend = OfficialMetricBackend(roi_detector_config=resolved)
        _BACKENDS_BY_ROI_CONFIG[key] = backend
    return backend


def score_metric(metric: MetricMetadata, case: PublicMetricCase) -> MetricResult:
    """CLI-compatible ``module:callable`` entry point."""

    return _backend_for_case(case)(metric, case)


# Callers that resolve a backend by ``module:callable`` get the preflight with
# it, without having to know which backend object stands behind the name.  The
# CLI looks this attribute up and skips the check when a backend does not carry
# one, so lightweight test doubles stay lightweight.
score_metric.preflight = _DEFAULT_BACKEND.preflight  # type: ignore[attr-defined]
