"""Compile the identity column's text queries out of a benchmark's own fields.

Grounding DINO is open-vocabulary, so the identity and scene columns need one
text query per recurring character.  Both benchmarks already carry that text,
in two different places, so nothing is annotated here:

* ``bernini_continuous_planner_formal96_v1`` declares it per case in
  ``stable_asset_requirements`` entries whose ``asset_type`` is ``character``.
  Measured on ``formal96_candidate.json``: 135 character entries spread over
  75 of the 96 cases, so 21 cases legitimately have none.
* ``real_holdout_96_v1`` has no asset table; its ``story_prompt`` is the native
  ``global_caption``, which is a ``Subject N: ...; Subject N+1: ...`` table.
  Measured: all 96 cases carry one, 275 subjects in total.
* ``bernini_continuous_planner_dev32_v1`` has no asset table either, and its
  ``story_prompt`` is a plain sentence, not a Subject table. Its one recurring
  character is instead pinned by an authored ``continuity.preserve`` fact, the
  same fact the DEV32 fixed-reference producer already turns into an
  ``explicit_character_preserve_fact`` candidate for image generation
  (:func:`mspc.data.bernini_dev32_assets._character_fact`). This module calls
  that same extraction function on the benchmark case directly -- it does not
  read the generated ``candidates.json`` and does not re-implement the rule.
  Measured: all 32 cases have exactly one explicit character preserve fact, so
  none are empty; the wording (a short authored identity phrase such as
  "woman identity and yellow raincoat") is noticeably terser than the
  Subject-table entries of ``real_holdout_96_v1``.

A description is truncated to its first :data:`CHARACTER_QUERY_WORDS` words.
That truncation is **this project's pre-registered choice**, not a borrowed one:
the repository has no source for a 12-word rule, and the measured word counts
(formal96 median 5, max 18; dev-96 median 9, max 20) mean it only bites a handful
of the longest entries.

An empty tuple and a missing case are different facts and are reported
differently: an empty tuple says this story has no recurring character, which
makes the identity column ineligible for it; a missing case is a compilation
error and raises.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from mspc.data.bernini_dev32_assets import _character_fact

#: Truncation length for a query, in words.  See the module docstring.
CHARACTER_QUERY_WORDS = 12

_SUBJECT_PATTERN = re.compile(r"Subject\s+(\d+):\s*(.*?)(?=;\s*Subject\s+\d+:|$)")

_FORMAL96_SCHEMA = "bernini_continuous_planner_formal96_v1"
_DEV96_SCHEMA = "real_holdout_96_v1"
_DEV32_SCHEMA = "bernini_continuous_planner_dev32_v1"


def parse_subject_table(story_prompt: str) -> dict[str, str]:
    """Split dev-96's ``Subject N: ...; Subject N+1: ...`` story_prompt table."""

    matches = _SUBJECT_PATTERN.findall(story_prompt.strip())
    if not matches:
        raise ValueError(f"story_prompt has no 'Subject N:' entries: {story_prompt!r}")
    characters: dict[str, str] = {}
    for number, description in matches:
        cleaned = description.strip().rstrip(".").strip()
        if not cleaned:
            raise ValueError(f"Subject {number} has an empty description: {story_prompt!r}")
        characters[f"subject_{number}"] = cleaned
    return characters


def truncate_query(description: str, *, label: str) -> str:
    words = description.split()
    if not words:
        raise ValueError(f"{label} is empty")
    return " ".join(words[:CHARACTER_QUERY_WORDS])


def _formal96_queries(case: Mapping[str, Any], *, label: str) -> tuple[str, ...]:
    assets = case.get("stable_asset_requirements")
    if not isinstance(assets, list):
        raise ValueError(f"{label}.stable_asset_requirements must be a list")
    queries: list[str] = []
    for index, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            raise ValueError(f"{label}.stable_asset_requirements[{index}] must be an object")
        if asset.get("asset_type") != "character":
            continue
        description = asset.get("description")
        if not isinstance(description, str):
            raise ValueError(
                f"{label}.stable_asset_requirements[{index}].description must be a string"
            )
        queries.append(
            truncate_query(
                description, label=f"{label}.stable_asset_requirements[{index}].description"
            )
        )
    return tuple(queries)


def _dev96_queries(case: Mapping[str, Any], *, label: str) -> tuple[str, ...]:
    story_prompt = case.get("story_prompt")
    if not isinstance(story_prompt, str) or not story_prompt.strip():
        raise ValueError(f"{label}.story_prompt must be a non-empty string")
    subjects = parse_subject_table(story_prompt)
    return tuple(
        truncate_query(description, label=f"{label}.{key}")
        for key, description in subjects.items()
    )


def _dev32_queries(case: Mapping[str, Any], *, label: str) -> tuple[str, ...]:
    fact, source = _character_fact(case)
    return (truncate_query(fact, label=f"{label}.{source}"),)


def character_queries_by_case_id(benchmark_path: str | Path) -> dict[str, tuple[str, ...]]:
    """Read one benchmark file and return its per-case identity queries."""

    path = Path(benchmark_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"benchmark file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema_version")
    if schema == _FORMAL96_SCHEMA:
        extract = _formal96_queries
    elif schema == _DEV96_SCHEMA:
        extract = _dev96_queries
    elif schema == _DEV32_SCHEMA:
        extract = _dev32_queries
    else:
        raise ValueError(
            f"{path}: identity queries are defined for {_FORMAL96_SCHEMA!r}, "
            f"{_DEV96_SCHEMA!r}, and {_DEV32_SCHEMA!r}, not {schema!r}"
        )
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path}: cases must be a non-empty array")
    queries: dict[str, tuple[str, ...]] = {}
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError(f"{path}: cases[{index}] must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{path}: cases[{index}].case_id must be a non-empty string")
        if case_id in queries:
            raise ValueError(f"{path}: duplicate case_id {case_id!r}")
        extracted = extract(case, label=f"{path.name}:{case_id}")
        # 去重，保持原顺序。两个主体的描述截断到前 12 词后可能撞成同一句——实测
        # dev-96 有 2/96 条这样的 case（例如三个风筝冲浪者都被描述成
        # "person in black wetsuit holding control bar on a board"），main_table_96
        # 是 0/96。查询集本就该是**互异文本的集合**：开放词表检测器对同一段文本只会
        # 返回同一个框，留着重复等于把一次检出当成两个角色计进 case 均值，把身份列
        # 往上抬。所以这里去重，而不是给撞车的那几条放宽截断长度——后者会让同一份
        # benchmark 里的查询长度不一致。
        seen: set[str] = set()
        deduped = tuple(q for q in extracted if not (q in seen or seen.add(q)))
        queries[case_id] = deduped
    return queries
