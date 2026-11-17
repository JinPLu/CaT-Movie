"""冻结A/B数据的旧断言字段及专项历史结果读取。

只服务现有benchmark生产者和空间诊断读数；不生成判官提示词、不调用模型，
也不定义当前整片五项评分或分母。旧付费判官从Git历史复现。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

POST_EVENT_STALE_KIND = "post_event_stale"

IN_SHOT_EVENT_EXECUTION_KIND = "in_shot_event_execution"

SCRIPT_ASSERTIONS_SCHEMA_VERSION = "mspc_script_assertions_v1"

POST_EVENT_STATE_KIND = "post_event_state"

SUBJECT_PRESENCE_KIND = "subject_presence"

SUBJECT_IDENTITY_KIND = "subject_identity"

SAME_LOCATION_KIND = "same_location"

REVISIT_INHERITANCE_KIND = "revisit_inheritance"

COMPOSITION_KIND = "composition"

COMPOSITION_MIN_SIZE_GAP = 2

SCRIPT_ASSERTION_FIELDS: tuple[str, ...] = (
    "assertion_id",
    "kind",
    "subject_id",
    "check_shot",
    "reference_shot",
    "event_id",
    "state",
    "contrast_state",
    "expected_answer",
)

_SCRIPT_KIND_FIELDS: dict[str, frozenset[str]] = {
    SUBJECT_PRESENCE_KIND: frozenset({"subject_id", "check_shot"}),
    SUBJECT_IDENTITY_KIND: frozenset({"subject_id", "check_shot", "reference_shot"}),
    SAME_LOCATION_KIND: frozenset({"check_shot", "reference_shot"}),
    POST_EVENT_STATE_KIND: frozenset(
        {"subject_id", "check_shot", "event_id", "state", "contrast_state"}
    ),
    POST_EVENT_STALE_KIND: frozenset(
        {"subject_id", "check_shot", "event_id", "state", "contrast_state"}
    ),
    REVISIT_INHERITANCE_KIND: frozenset(
        {"subject_id", "check_shot", "event_id", "state", "contrast_state"}
    ),
    COMPOSITION_KIND: frozenset({"check_shot", "state"}),
    IN_SHOT_EVENT_EXECUTION_KIND: frozenset({"subject_id", "check_shot", "event_id", "state", "contrast_state"}),
}

SCRIPT_CONTEXT_FIELDS: tuple[str, ...] = ("story_prompt", "subjects", "locations", "shots")

ANSWER_YES = "yes"

ANSWER_NO = "no"

SCRIPT_JUDGE_FAILED_STATUSES: tuple[str, ...] = ("format_failed", "intake_failed")

def require_script_assertion(row: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """一条剧本断言的键集合与取值检查：键恰好 :data:`SCRIPT_ASSERTION_FIELDS`，kind 决定哪些键非空。"""

    if not isinstance(row, Mapping) or set(row) != set(SCRIPT_ASSERTION_FIELDS):
        keys = sorted(row) if isinstance(row, Mapping) else row
        raise ValueError(f"{label} must carry exactly {SCRIPT_ASSERTION_FIELDS}, got {keys}")
    kind = row["kind"]
    if kind not in _SCRIPT_KIND_FIELDS:
        raise ValueError(f"{label}: unknown kind {kind!r}")
    if row["expected_answer"] not in (ANSWER_YES, ANSWER_NO):
        raise ValueError(f"{label}: expected_answer must be 'yes' or 'no'")
    for key in SCRIPT_ASSERTION_FIELDS:
        if key in ("assertion_id", "kind", "expected_answer"):
            value = row[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label}.{key} must be a non-empty string")
            continue
        value = row[key]
        if key in _SCRIPT_KIND_FIELDS[kind]:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label}.{key} must be a non-empty string for kind {kind!r}")
        elif value is not None:
            raise ValueError(f"{label}.{key} must be null for kind {kind!r}, got {value!r}")
    return dict(row)

def load_script_assertions(path: Path) -> dict[str, dict[str, Any]]:
    """读冻结的剧本断言产物：``case_id -> {"context": ..., "assertions": (…)}``。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCRIPT_ASSERTIONS_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version must be {SCRIPT_ASSERTIONS_SCHEMA_VERSION!r}, "
            f"got {payload.get('schema_version')!r}"
        )
    cases = payload["cases"]
    if not isinstance(cases, dict) or not cases:
        raise ValueError(f"{path}: cases must be a non-empty object keyed by case_id")
    loaded: dict[str, dict[str, Any]] = {}
    total = 0
    for case_id, entry in cases.items():
        if set(entry) != {"context", "assertions"}:
            raise ValueError(f"{path}: {case_id} must carry exactly context and assertions")
        context = entry["context"]
        if set(context) != set(SCRIPT_CONTEXT_FIELDS):
            raise ValueError(f"{path}: {case_id}.context must carry exactly {SCRIPT_CONTEXT_FIELDS}")
        rows = entry["assertions"]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{path}: {case_id} must carry a non-empty assertions list")
        checked = tuple(
            require_script_assertion(row, label=f"{path}: {case_id} assertions[{index}]")
            for index, row in enumerate(rows)
        )
        ids = [row["assertion_id"] for row in checked]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path}: {case_id} has duplicate assertion ids")
        shot_ids = {shot["shot_id"] for shot in context["shots"]}
        for row in checked:
            for key in ("check_shot", "reference_shot"):
                if row[key] is not None and row[key] not in shot_ids:
                    raise ValueError(f"{path}: {case_id}/{row['assertion_id']}.{key} is not a shot of this case")
        loaded[case_id] = {"context": context, "assertions": checked}
        total += len(checked)
    if payload["case_count"] != len(loaded) or payload["assertion_count"] != total:
        raise ValueError(
            f"{path}: declares case_count={payload['case_count']} assertion_count="
            f"{payload['assertion_count']} but carries {len(loaded)} / {total}"
        )
    return loaded

def load_script_judge_answers(paths: Sequence[Path]) -> dict[str, dict[str, dict[str, Any]]]:
    """一条臂的剧本判官逐条结果（``*.cases.jsonl``，可分在几个续跑文件里）：``story_id -> 题号 -> 答案``。

    答案去掉 ``evidence``，只留判定字段（``holds`` 或 ``choice``）。同一条故事在两份文件里答案不同即抛：
    续跑只补没判过的故事，不会重判。``parse_status`` 在 ``SCRIPT_JUDGE_FAILED_STATUSES`` 里的行没有答案，
    这里记成空表供原专项读数识别；不参与新版五项评分。
    """

    merged: dict[str, dict[str, dict[str, Any]]] = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("parse_status") in SCRIPT_JUDGE_FAILED_STATUSES:
                answers = {}
            else:
                answers = {key: {k: v for k, v in value.items() if k != "evidence"} for key, value in row["answers"].items()}
            if row["story_id"] in merged and merged[row["story_id"]] != answers:
                raise ValueError(f"{path}: {row['story_id']} was judged twice with different answers")
            merged[row["story_id"]] = answers
    return merged
