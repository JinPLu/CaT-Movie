"""Small model-agnostic helpers for the five renderer public metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PUBLIC_METRIC_IDS = (
    "video_score",
    "viclip_text_alignment",
    "dinov2_character_consistency",
    "dinov2_scene_consistency",
    "transnet_transition_deviation",
)

# v2 reports five effect families, not a VideoScore aggregate.  Cut execution
# is two raw metrics (count error + matched timing MAE), so the profile has
# six IDs.  ``video_score_visual_quality`` is also a v1 diagnostic; it is a
# v2 main column without leaving the v1 appendix.  Character identity uses the
# presence-gated DINOv2 readout so a missed detection is N/A, not a low score.
V2_TRANSITION_METRIC_IDS = (
    "transition_count_error",
    "matched_transition_timing_mae",
)
PUBLIC_METRIC_IDS_V2 = (
    "video_score_visual_quality",
    "viclip_text_alignment",
    "dinov2_character_consistency_gated",
    "dinov2_scene_consistency",
    *V2_TRANSITION_METRIC_IDS,
)

# v3 reports twelve columns in one fixed order.  Quality is VBench's two
# per-frame dimensions rather than VideoScore, which keeps writing its raw dump
# without holding a column.  Identity is the open-vocabulary column: one
# Grounding DINO text query per declared character, SAM mask, DINOv2 cosine, and
# a shot-pair unit only scores when all six keyframes detected that character.
# The last five columns are produced by the judge line (MSVBench state axis and
# the four MSM narrative-coherence axes); this module owns their order and
# metadata, not their computation.
V3_QUALITY_METRIC_IDS = (
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
)
V3_NARRATIVE_COHERENCE_METRIC_IDS = (
    "narrative_coherence_scene",
    "narrative_coherence_subject",
    "narrative_coherence_action",
    "narrative_coherence_spatial",
)
PUBLIC_METRIC_IDS_V3 = (
    *V3_QUALITY_METRIC_IDS,
    "viclip_text_alignment",
    "dinov2_character_identity",
    "dinov2_scene_consistency",
    *V2_TRANSITION_METRIC_IDS,
    "state_shift_persistence",
    *V3_NARRATIVE_COHERENCE_METRIC_IDS,
)
# Measurability columns for the v3 identity score.  They are read beside the
# identity mean and never enter it: the all-keyframes-detected rule makes the
# scored denominator arm-dependent, so an identity mean without them is not
# interpretable.  ``identity_pairs_total`` is arm-independent.
V3_IDENTITY_DIAGNOSTIC_METRIC_IDS = (
    "identity_pairs_scored",
    "identity_pairs_total",
    "identity_cases_scored",
)

# Diagnostic columns added by the 2026-08-23 evaluation-protocol repair.  They
# never join ``PUBLIC_METRIC_IDS``: the main table keeps reporting exactly the
# five frozen public metrics, and every column below is an appendix column with
# its own metric ID.  Reproducing a historical main-table number therefore only
# requires the five, whether or not any diagnostic column was also scored.
# v2-only TransNet IDs sit at the end so a v1 config can still name them as
# extras, and so TensorFlow preflight sees every ID that builds TransNetV2.
V1_DIAGNOSTIC_METRIC_IDS = (
    "dinov2_character_consistency_gated",
    "video_score_visual_quality",
    "video_score_temporal_consistency",
    "video_score_dynamic_degree",
    "video_score_text_to_video_alignment",
    "video_score_factual_consistency",
    "motion_magnitude",
)
DIAGNOSTIC_METRIC_IDS = (
    V1_DIAGNOSTIC_METRIC_IDS
    + V2_TRANSITION_METRIC_IDS
    + V3_IDENTITY_DIAGNOSTIC_METRIC_IDS
)
KNOWN_METRIC_IDS = tuple(
    dict.fromkeys(
        PUBLIC_METRIC_IDS
        + PUBLIC_METRIC_IDS_V3
        + DIAGNOSTIC_METRIC_IDS
    )
)
# ``neutral`` exists for denominators and raw counts, where neither direction is
# an improvement; the five public metrics stay higher/lower.
METRIC_DIRECTIONS = ("higher", "lower", "neutral")


class PairManifestError(ValueError):
    """pair manifest 与被打分的 case 对不上。

    定义在合同层而不是某个 backend 里：抛它的是 backend，捕它的是调用方，两边都已经
    依赖本模块，所以类型化不引入新的跨层依赖。

    调用方按类型捕获，不按 ``"pair manifest must contain exactly one row" in str(exc)``
    去匹配——那句话在**另一个模块**里，改一个词，归类就静默降级成通用打分失败。
    """


@dataclass
class MetricMetadata:
    metric_id: str
    direction: str
    unit: str
    value_range: tuple[float, float] | None = None
    allow_not_applicable: bool = False


@dataclass
class PublicMetricCase:
    """One scored story.

    ``character_queries`` carries the open-vocabulary text queries the v3
    identity and scene columns detect with, one per declared character of this
    story.  ``None`` and ``()`` are different facts and both are load-bearing:
    ``None`` means the scoring input never declared the field, and the identity
    and scene columns refuse to run rather than quietly reading zero characters;
    ``()`` means the benchmark declares this story has no character asset, and
    the identity column reports ``ineligible_no_character_asset`` while the
    scene column embeds the whole frame.
    """

    case_id: str
    video_path: Path
    story_prompt: str
    shots: Sequence[Mapping[str, Any]]
    resource_root: Path
    pair_manifest_path: Path | None = None
    subject_class: str = "person"
    character_queries: tuple[str, ...] | None = None
    arm_id: str | None = None
    continuity: Mapping[str, Any] = field(default_factory=dict)
    result_metadata: Mapping[str, Any] = field(default_factory=dict)
    metric_config_path: Path | None = None


@dataclass
class MetricResult:
    value: float | None
    raw_outputs: Any = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelBatchSizes:
    detector_frame_batch_size: int
    dino_crop_batch_size: int
    viclip_shot_batch_size: int

    def __post_init__(self) -> None:
        for name in (
            "detector_frame_batch_size",
            "dino_crop_batch_size",
            "viclip_shot_batch_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


RECOMMENDED_A100_MODEL_BATCH = ModelBatchSizes(
    detector_frame_batch_size=16,
    dino_crop_batch_size=32,
    viclip_shot_batch_size=8,
)

_MODEL_BATCH_KEYS = (
    "detector_frame_batch_size",
    "dino_crop_batch_size",
    "viclip_shot_batch_size",
)


def parse_model_batch_sizes(raw: object, *, label: str) -> ModelBatchSizes:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be an object")
    missing = [key for key in _MODEL_BATCH_KEYS if key not in raw]
    if missing:
        raise ValueError(f"{label} is missing {missing}")
    extra = [key for key in raw if key not in _MODEL_BATCH_KEYS]
    if extra:
        raise ValueError(f"{label} has unknown keys: {extra}")
    values: dict[str, int] = {}
    for key in _MODEL_BATCH_KEYS:
        value = raw[key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label}.{key} must be a positive integer")
        values[key] = value
    return ModelBatchSizes(**values)


MetricBackend = Callable[[MetricMetadata, PublicMetricCase], MetricResult]


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def validate_metric_metadata(
    metrics: Sequence[MetricMetadata], *, allow_subset: bool = False
) -> tuple[MetricMetadata, ...]:
    parsed = tuple(metrics)
    ids = tuple(metric.metric_id for metric in parsed)
    if allow_subset:
        if not ids or any(metric_id not in KNOWN_METRIC_IDS for metric_id in ids):
            raise ValueError(
                "metric selection must use the five configured public metrics "
                "or a declared diagnostic column"
            )
    else:
        # v1: the five frozen public metrics stay first and in order.
        # v2: the six effect-family IDs stay first and in order.
        # v3: the twelve columns stay first and in order.  Declared diagnostic
        # columns may follow any profile but never displace it.
        if ids[: len(PUBLIC_METRIC_IDS_V3)] == PUBLIC_METRIC_IDS_V3:
            prefix_len = len(PUBLIC_METRIC_IDS_V3)
        elif ids[: len(PUBLIC_METRIC_IDS)] == PUBLIC_METRIC_IDS:
            prefix_len = len(PUBLIC_METRIC_IDS)
        elif ids[: len(PUBLIC_METRIC_IDS_V2)] == PUBLIC_METRIC_IDS_V2:
            prefix_len = len(PUBLIC_METRIC_IDS_V2)
        else:
            raise ValueError(
                "metrics must be ordered as "
                f"{list(PUBLIC_METRIC_IDS)}, {list(PUBLIC_METRIC_IDS_V2)}, "
                f"{list(PUBLIC_METRIC_IDS_V3)} "
            )
        unknown = [
            metric_id
            for metric_id in ids[prefix_len:]
            if metric_id not in DIAGNOSTIC_METRIC_IDS
        ]
        if unknown:
            raise ValueError(f"unknown diagnostic metric IDs: {unknown}")
    if len(set(ids)) != len(ids):
        raise ValueError("metric IDs must be unique")
    for index, metric in enumerate(parsed):
        _text(metric.metric_id, f"metrics[{index}].metric_id")
        if metric.direction not in set(METRIC_DIRECTIONS):
            raise ValueError(
                f"metrics[{index}].direction must be one of {list(METRIC_DIRECTIONS)}"
            )
        _text(metric.unit, f"metrics[{index}].unit")
        if metric.value_range is not None:
            low, high = metric.value_range
            if not all(math.isfinite(float(value)) for value in (low, high)) or low >= high:
                raise ValueError(f"metrics[{index}].value_range must be increasing")
    return parsed


def is_v2_main_table(metrics: Sequence[MetricMetadata]) -> bool:
    return tuple(metric.metric_id for metric in metrics) == PUBLIC_METRIC_IDS_V2


def is_v3_main_table(metrics: Sequence[MetricMetadata]) -> bool:
    return tuple(metric.metric_id for metric in metrics) == PUBLIC_METRIC_IDS_V3


def require_v3_identity_diagnostics(
    diagnostics: Sequence[MetricMetadata], *, label: str
) -> None:
    """A v3 run must carry the identity measurability columns.

    The all-keyframes-detected rule makes the identity denominator differ
    between arms, so the mean cannot be published without the scored/total
    counts beside it.  Requiring them on the config is the only place that can
    keep the two from drifting apart.
    """

    ids = tuple(metric.metric_id for metric in diagnostics)
    missing = [
        metric_id
        for metric_id in V3_IDENTITY_DIAGNOSTIC_METRIC_IDS
        if metric_id not in ids
    ]
    if missing:
        raise ValueError(
            f"{label} must declare {list(V3_IDENTITY_DIAGNOSTIC_METRIC_IDS)}; "
            f"missing {missing}"
        )
    if "dinov2_character_identity" in ids:
        raise ValueError(
            f"{label} must not repeat the identity primary dinov2_character_identity"
        )


def metric_metadata_payload(metrics: Sequence[MetricMetadata]) -> list[dict[str, Any]]:
    return [
        {
            "metric_id": metric.metric_id,
            "direction": metric.direction,
            "unit": metric.unit,
            "range": list(metric.value_range) if metric.value_range is not None else None,
            "allow_not_applicable": metric.allow_not_applicable,
        }
        for metric in metrics
    ]


def metric_metadata_from_declarations(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> tuple[MetricMetadata, ...]:
    """Parse ``metrics``/``diagnostic_metrics`` config rows without defaulting."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError(f"{label} must be a non-empty array")
    parsed: list[MetricMetadata] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{label}[{index}] must be an object")
        for field_name in ("metric_id", "direction", "unit", "range", "allow_not_applicable"):
            if field_name not in row:
                raise ValueError(f"{label}[{index}].{field_name} is required")
        value_range = row["range"]
        parsed_range: tuple[float, float] | None = None
        if value_range is not None:
            if (
                not isinstance(value_range, Sequence)
                or isinstance(value_range, (str, bytes))
                or len(value_range) != 2
                or any(type(value) not in {int, float} for value in value_range)
            ):
                raise ValueError(f"{label}[{index}].range must be null or two numbers")
            parsed_range = (float(value_range[0]), float(value_range[1]))
        allow_na = row["allow_not_applicable"]
        if type(allow_na) is not bool:
            raise ValueError(f"{label}[{index}].allow_not_applicable must be boolean")
        parsed.append(
            MetricMetadata(
                metric_id=_text(row["metric_id"], f"{label}[{index}].metric_id"),
                direction=_text(row["direction"], f"{label}[{index}].direction"),
                unit=_text(row["unit"], f"{label}[{index}].unit"),
                value_range=parsed_range,
                allow_not_applicable=allow_na,
            )
        )
    return tuple(parsed)


def validate_case(case: PublicMetricCase) -> PublicMetricCase:
    _text(case.case_id, "case_id")
    _text(case.story_prompt, "story_prompt")
    _text(case.subject_class, "subject_class")
    if case.character_queries is not None:
        queries = tuple(case.character_queries)
        for index, query in enumerate(queries):
            _text(query, f"character_queries[{index}]")
            if query != query.strip():
                raise ValueError(f"character_queries[{index}] must be stripped")
        if len(set(queries)) != len(queries):
            raise ValueError("character_queries must be unique")
    if not isinstance(case.result_metadata, Mapping):
        raise ValueError("result_metadata must be an object")
    if not case.video_path.expanduser().is_file():
        raise FileNotFoundError(f"video does not exist: {case.video_path}")
    if not case.resource_root.expanduser().is_dir():
        raise FileNotFoundError(f"resource root does not exist: {case.resource_root}")
    if not isinstance(case.shots, Sequence) or not case.shots:
        raise ValueError("shots must be a non-empty array")
    seen: set[str] = set()
    for index, shot in enumerate(case.shots):
        if not isinstance(shot, Mapping):
            raise ValueError(f"shots[{index}] must be an object")
        shot_id = _text(shot.get("shot_id"), f"shots[{index}].shot_id")
        _text(shot.get("prompt"), f"shots[{index}].prompt")
        frame_count = shot.get("frame_count")
        if type(frame_count) is not int or frame_count <= 0:
            raise ValueError(f"shots[{index}].frame_count must be a positive integer")
        if shot_id in seen:
            raise ValueError(f"duplicate shot ID: {shot_id}")
        seen.add(shot_id)
    return case


def _validated_result(result: object, metric: MetricMetadata) -> MetricResult:
    if not isinstance(result, MetricResult):
        raise TypeError(f"backend must return MetricResult for {metric.metric_id}")
    if result.value is None:
        if not metric.allow_not_applicable:
            raise ValueError(f"{metric.metric_id} does not allow a null value")
        return result
    if type(result.value) not in {int, float} or not math.isfinite(float(result.value)):
        raise ValueError(f"{metric.metric_id} must return a finite number")
    value = float(result.value)
    if metric.value_range is not None and not metric.value_range[0] <= value <= metric.value_range[1]:
        raise ValueError(f"{metric.metric_id} is outside its configured range")
    return MetricResult(value, result.raw_outputs, dict(result.diagnostics))


def run_public_metric_case(
    case: PublicMetricCase,
    metrics: Sequence[MetricMetadata],
    *,
    backend: MetricBackend,
) -> dict[str, Any]:
    """Run selected metrics and return one ordinary JSON-ready result row."""

    validate_case(case)
    parsed = validate_metric_metadata(metrics, allow_subset=True)
    values: dict[str, float | None] = {}
    details: dict[str, dict[str, Any]] = {}
    for metric in parsed:
        result = _validated_result(backend(metric, case), metric)
        values[metric.metric_id] = result.value
        details[metric.metric_id] = {
            "value": result.value,
            "raw_outputs": result.raw_outputs,
            "diagnostics": dict(result.diagnostics),
        }
    return {
        **dict(case.result_metadata),
        "case_id": case.case_id,
        "arm_id": case.arm_id,
        "metrics": values,
        "metric_results": details,
    }


def aggregate_metric_rows(
    rows: Sequence[Mapping[str, Any]], metrics: Sequence[MetricMetadata]
) -> dict[str, dict[str, float | int | str | None]]:
    """Average each metric independently; null values are omitted per column."""

    parsed = validate_metric_metadata(metrics)
    columns: dict[str, dict[str, float | int | str | None]] = {}
    for metric in parsed:
        values: list[float] = []
        for index, row in enumerate(rows):
            metric_values = row.get("metrics")
            if not isinstance(metric_values, Mapping):
                raise ValueError(f"rows[{index}].metrics must be an object")
            value = metric_values.get(metric.metric_id)
            if value is None:
                continue
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                raise ValueError(f"rows[{index}].metrics.{metric.metric_id} must be finite or null")
            values.append(float(value))
        columns[metric.metric_id] = {
            "direction": metric.direction,
            "unit": metric.unit,
            "mean": sum(values) / len(values) if values else None,
            "count": len(values),
        }
    return columns


def planned_shot_boundaries(frame_counts: Sequence[int]) -> tuple[int, ...]:
    if not frame_counts:
        raise ValueError("frame_counts must not be empty")
    total = 0
    boundaries: list[int] = []
    for index, frame_count in enumerate(frame_counts):
        if type(frame_count) is not int or frame_count <= 0:
            raise ValueError(f"frame_counts[{index}] must be a positive integer")
        total += frame_count
        if index < len(frame_counts) - 1:
            boundaries.append(total)
    return tuple(boundaries)


def first_mid_last_frame_indices(frame_counts: Sequence[int]) -> tuple[tuple[int, int, int], ...]:
    planned_shot_boundaries(frame_counts)
    start = 0
    result: list[tuple[int, int, int]] = []
    for frame_count in frame_counts:
        last = start + frame_count - 1
        result.append((start, (start + last) // 2, last))
        start = last + 1
    return tuple(result)


def all_unordered_pairs(shot_ids: Sequence[str]) -> tuple[tuple[str, str], ...]:
    parsed = tuple(_text(value, "shot_id") for value in shot_ids)
    if len(set(parsed)) != len(parsed):
        raise ValueError("shot IDs must be unique")
    return tuple(combinations(parsed, 2))


def cosine_3x3_macro(
    embeddings_by_shot: Mapping[str, Sequence[Sequence[float]]],
    pairs: Sequence[Sequence[str]],
) -> MetricResult:
    pair_rows: list[dict[str, Any]] = []
    means: list[float] = []
    for pair in pairs:
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError("each pair must contain two distinct shot IDs")
        left_frames = embeddings_by_shot[pair[0]]
        right_frames = embeddings_by_shot[pair[1]]
        if len(left_frames) != 3 or len(right_frames) != 3:
            raise ValueError("each shot must contain three embeddings")
        values: list[float] = []
        for left in left_frames:
            for right in right_frames:
                if not left or len(left) != len(right):
                    raise ValueError("embedding dimensions must match and be non-zero")
                dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
                left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
                right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
                if left_norm == 0 or right_norm == 0:
                    raise ValueError("embedding norms must be non-zero")
                values.append(dot / (left_norm * right_norm))
        mean = sum(values) / len(values)
        means.append(mean)
        pair_rows.append({"shot_pair": list(pair), "cosines": values, "pair_mean": mean})
    if not means:
        raise ValueError("at least one pair is required")
    return MetricResult(sum(means) / len(means), {"pairs": pair_rows}, {"pair_count": len(means)})


def transition_run_peaks(
    probabilities: Sequence[float], *, threshold: float = 0.5
) -> tuple[int, ...]:
    values = [float(value) for value in probabilities]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("probabilities must be finite")
    peaks: list[int] = []
    index = 0
    while index < len(values):
        if values[index] <= threshold:
            index += 1
            continue
        start = index
        while index + 1 < len(values) and values[index + 1] > threshold:
            index += 1
        peaks.append(max(range(start, index + 1), key=lambda item: (values[item], -item)))
        index += 1
    return tuple(peaks)


def cut_boundaries_from_peaks(
    peaks: Sequence[int], *, frame_count: int
) -> tuple[int, ...]:
    """Turn TransNetV2 cut peaks into boundaries on the planned convention.

    TransNetV2's single-frame head peaks on the last frame of the outgoing
    shot; a planned boundary (``planned_shot_boundaries``) names the first frame
    of the incoming shot.  Measured on 16 real videos (2026-09-24): every
    matched peak sat exactly one frame before its planned boundary.  So each
    peak ``p`` is the boundary ``p + 1``; a peak on the final frame has no
    incoming frame and is not a cut inside the video.
    """

    if type(frame_count) is not int or frame_count <= 1:
        raise ValueError("frame_count must be greater than one")
    values = tuple(peaks)
    if any(type(value) is not int or not 0 <= value < frame_count for value in values):
        raise ValueError("peaks must be frame indices")
    if tuple(sorted(set(values))) != values:
        raise ValueError("peaks must be increasing")
    return tuple(value + 1 for value in values if value + 1 < frame_count)


def _validated_boundaries(
    reference_boundaries: Sequence[int],
    detected_boundaries: Sequence[int],
    *,
    frame_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if type(frame_count) is not int or frame_count <= 1:
        raise ValueError("frame_count must be greater than one")
    reference = tuple(reference_boundaries)
    detected = tuple(detected_boundaries)
    for label, values in (("reference", reference), ("detected", detected)):
        if any(type(value) is not int or not 0 <= value < frame_count for value in values):
            raise ValueError(f"{label} boundaries must be frame indices")
        if tuple(sorted(set(values))) != values:
            raise ValueError(f"{label} boundaries must be increasing")
    return reference, detected


def transition_deviation(
    reference_boundaries: Sequence[int],
    detected_boundaries: Sequence[int],
    *,
    frame_count: int,
) -> MetricResult:
    reference, detected = _validated_boundaries(
        reference_boundaries, detected_boundaries, frame_count=frame_count
    )
    penalty = frame_count - 1
    rows, columns = len(reference), len(detected)
    costs = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        costs[row][0] = row * penalty
    for column in range(1, columns + 1):
        costs[0][column] = column * penalty
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            costs[row][column] = min(
                costs[row - 1][column - 1] + abs(reference[row - 1] - detected[column - 1]),
                costs[row - 1][column] + penalty,
                costs[row][column - 1] + penalty,
            )
    count = max(rows, columns, 1)
    return MetricResult(
        costs[rows][columns] / count,
        {"reference_boundaries": list(reference), "detected_boundaries": list(detected)},
        {"matching_cost": costs[rows][columns], "normalization_count": count},
    )


def _monotonic_min_abs_pairs(
    shorter: Sequence[int], longer: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    """Match every item of ``shorter`` to a subsequence of ``longer``.

    The matching is order-preserving and minimises total absolute error, so a
    later short item cannot claim an earlier long item.  Equal-length inputs
    therefore zip; a longer side may skip unmatched items.
    """

    short = tuple(shorter)
    long = tuple(longer)
    rows, columns = len(short), len(long)
    infinity = float("inf")
    costs = [[infinity] * (columns + 1) for _ in range(rows + 1)]
    matched = [[False] * (columns + 1) for _ in range(rows + 1)]
    for column in range(columns + 1):
        costs[0][column] = 0.0
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            skip = costs[row][column - 1]
            pair = costs[row - 1][column - 1] + abs(short[row - 1] - long[column - 1])
            if pair <= skip:
                costs[row][column] = pair
                matched[row][column] = True
            else:
                costs[row][column] = skip
    row, column = rows, columns
    pairs: list[tuple[int, int]] = []
    while row > 0:
        if matched[row][column]:
            pairs.append((short[row - 1], long[column - 1]))
            row -= 1
            column -= 1
            continue
        column -= 1
    pairs.reverse()
    return tuple(pairs)


def _transition_match_audit(
    reference_boundaries: Sequence[int],
    detected_boundaries: Sequence[int],
    *,
    frame_count: int,
) -> dict[str, Any]:
    reference, detected = _validated_boundaries(
        reference_boundaries, detected_boundaries, frame_count=frame_count
    )
    if len(reference) <= len(detected):
        aligned = _monotonic_min_abs_pairs(reference, detected)
        matched = tuple(
            {"reference": left, "detected": right, "absolute_error": abs(left - right)}
            for left, right in aligned
        )
    else:
        aligned = _monotonic_min_abs_pairs(detected, reference)
        matched = tuple(
            {"reference": right, "detected": left, "absolute_error": abs(left - right)}
            for left, right in aligned
        )
    matched_reference = {row["reference"] for row in matched}
    matched_detected = {row["detected"] for row in matched}
    unmatched_reference = [value for value in reference if value not in matched_reference]
    unmatched_detected = [value for value in detected if value not in matched_detected]
    return {
        "reference_boundaries": list(reference),
        "detected_boundaries": list(detected),
        "matched": [dict(row) for row in matched],
        "unmatched_reference": unmatched_reference,
        "unmatched_detected": unmatched_detected,
    }


def transition_count_error(
    reference_boundaries: Sequence[int],
    detected_boundaries: Sequence[int],
    *,
    frame_count: int,
) -> MetricResult:
    raw = _transition_match_audit(
        reference_boundaries, detected_boundaries, frame_count=frame_count
    )
    value = abs(len(raw["reference_boundaries"]) - len(raw["detected_boundaries"]))
    return MetricResult(
        value,
        raw,
        {
            "reference_count": len(raw["reference_boundaries"]),
            "detected_count": len(raw["detected_boundaries"]),
            "matched_count": len(raw["matched"]),
        },
    )


def matched_transition_timing_mae(
    reference_boundaries: Sequence[int],
    detected_boundaries: Sequence[int],
    *,
    frame_count: int,
) -> MetricResult:
    raw = _transition_match_audit(
        reference_boundaries, detected_boundaries, frame_count=frame_count
    )
    reference = raw["reference_boundaries"]
    detected = raw["detected_boundaries"]
    matched = raw["matched"]
    if not reference and not detected:
        value: float | None = 0.0
    elif not reference or not detected:
        value = None
    else:
        value = sum(row["absolute_error"] for row in matched) / len(matched)
    return MetricResult(
        value,
        raw,
        {
            "matched_count": len(matched),
            "unmatched_reference_count": len(raw["unmatched_reference"]),
            "unmatched_detected_count": len(raw["unmatched_detected"]),
        },
    )
