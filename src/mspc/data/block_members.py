"""块清单＋已校验 v18 标注＋逐镜取样台账 → 按片分开的消费者成员。

成员资格取台账的 usable，再收紧质量和文字预算；目标潜变量/参考候选未知不会冒充覆盖。
坏镜原位保留为断点，整块没有可用镜才不导出。正式取样量由训练配置决定。
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from mspc.data.film_split import film_group, validation_group
from mspc.data import sampling_ledger as sl
from mspc.data.shot_targets import latent_frames
from mspc.data.storyboard import REGISTRIES, unregistered_entity_references
from mspc.data.shot_text import (
    UMT5_PROMPT_TOKEN_BUDGET, load_umt5_token_counter, reference_text, shot_text,
)


def members(blocks: Iterable[Mapping[str, Any]], annotations: Mapping[str, dict],
            ledger_shots: Iterable[Mapping[str, Any]], *, validation_fraction: float,
            split_seed: int) -> Iterable[dict]:
    ledger = {}
    for row in ledger_shots:
        key = (row["block_key"], row["shot"])
        if key in ledger:
            raise ValueError(f"duplicate ledger row {key}")
        ledger[key] = dict(row)
    seen_blocks, seen_shots = set(), set()
    for block in blocks:
        key, scene = block["key"], block["scene"]
        if key in seen_blocks:
            raise ValueError(f"duplicate block {key}")
        seen_blocks.add(key)
        joined = []
        for number, shot in enumerate(scene["shots"], 1):
            identity = sl.shot_key(shot)
            if identity in seen_shots:
                raise ValueError(f"shot occurs in multiple blocks: {identity}")
            seen_shots.add(identity)
            row = ledger.pop((key, number))
            if sl.shot_key(row) != identity or row["frames"] != shot["render_frames"]:
                raise ValueError(f"ledger identity or frames differ at {key}:{number}")
            if type(row["usable"]) is not bool or row["usable"] != (not row["unusable_reasons"]):
                raise ValueError(f"inconsistent usability at {key}:{number}")
            if row["usable"] and row["eval_excluded"]:
                raise ValueError(f"evaluation exclusion marked usable at {key}:{number}")
            latent_frames(row["frames"])
            joined.append({**shot, **row})
        if not any(row["usable"] for row in joined):
            continue
        answer = annotations[key]
        if not sl.registry_valid(answer):
            raise ValueError(f"usable block has invalid registry: {key}")
        annotation = answer["annotation"]
        if len(annotation["shots"]) != len(joined):
            raise ValueError(f"annotation shot count differs: {key}")
        for row, shot in zip(joined, annotation["shots"], strict=True):
            if row["usable"] and (shot["shot"] != row["shot"] or str(row["shot"]) in answer["shot_errors"]):
                raise ValueError(f"usable annotation shot invalid: {key}:{row['shot']}")
        group = film_group(scene["film_key"], scene["source_video_path"])
        split = "val" if validation_group(group, fraction=validation_fraction, seed=split_seed) else "train"
        yield {"schema_version": 1, "key": key, "case_id": block["case_id"],
               "source": block["source"], "family": block["family"], "film_key": group,
               "split": split, "annotation": annotation,
               "scene": {name: value for name, value in scene.items() if name != "shots"}, "shots": joined}
    if ledger:
        raise ValueError(f"ledger has {len(ledger)} rows outside block manifest")


def coverage(rows: Iterable[Mapping[str, Any]]) -> dict:
    """逐镜统计资格与台账已连接的缓存覆盖，缺失/未知分别计数。"""
    counts = Counter()
    groups: dict[str, set[str]] = {"train": set(), "val": set()}
    for member in rows:
        split = member["split"]
        groups[split].add(member["film_key"])
        counts[f"{split}_blocks"] += 1
        counts["whole_usable_blocks"] += all(s["usable"] for s in member["shots"])
        for shot in member["shots"]:
            if not shot["usable"]:
                counts["unusable_breakpoints"] += 1
                continue
            counts[f"{split}_usable_shots"] += 1
            counts["usable_shots"] += 1
            # 此列只盘点已有旧候选库，不是新训练的输入前置。
            n = latent_frames(shot["frames"]) - 1
            counts["required_candidate_frames"] += n
            target, ref = shot["target_latents"], shot["reference_frames"]
            counts["target_unknown" if target is None else "target_index_present" if target else "target_missing"] += 1
            if ref is None:
                counts["reference_unknown"] += 1
            elif ref == 0:
                counts["reference_missing"] += 1
                counts["missing_candidate_frames"] += n
            elif ref != n:
                raise ValueError(f"candidate count differs for {sl.shot_key(shot)}: {ref} != {n}")
            else:
                counts["reference_manifest_present"] += 1
                counts["present_candidate_frames"] += ref
    if groups["train"] & groups["val"]:
        raise ValueError("train and val share films")
    return {"counts": dict(sorted(counts.items())), "film_groups": {s: len(g) for s, g in groups.items()},
            "coverage_level": "index_and_manifest_only; tensor files require dataset preflight"}


def select_quality_members(rows: Iterable[Mapping[str, Any]], *,
                           min_laion_aesthetic: float, min_dover_technical: float, min_mean_luma: float,
                           token_counter: Callable[[str], int]) -> Iterable[dict]:
    """训练资格只收紧；被剔除镜原位作断点，目标与参考共用同一份资格。"""
    limits = {"laion_aesthetic": min_laion_aesthetic, "dover_technical": min_dover_technical, "mean_luma": min_mean_luma}
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in limits.values()):
        raise ValueError("quality thresholds must be finite numbers")
    for member in rows:
        shots = []
        known = {entry["name"] for kind in REGISTRIES for entry in member["annotation"][kind]}
        for position, original in enumerate(member["shots"]):
            reasons = list(original["unusable_reasons"])
            for name, limit in limits.items():
                value = original[name]
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(f"{member['key']}:{original['shot']} invalid {name}")
                if value < limit:
                    reasons.append(f"{name}<{limit}")
            counts = None
            if not reasons:
                reasons.extend(f"unregistered_entity_reference:{name}" for name in
                               unregistered_entity_references(member["annotation"]["shots"][position], known))
            if not reasons:
                counts = {
                    "target": token_counter(shot_text(member["annotation"], position)),
                    "reference": token_counter(reference_text()),
                }
                for role, count in counts.items():
                    if type(count) is not int or count <= 0:
                        raise ValueError(f"{member['key']}:{original['shot']} invalid {role} token count: {count}")
                    if count > UMT5_PROMPT_TOKEN_BUDGET:
                        reasons.append(f"{role}_umt5_tokens={count}>{UMT5_PROMPT_TOKEN_BUDGET}")
            shots.append({**original, "usable": not reasons, "unusable_reasons": reasons,
                          "text_token_counts": counts})
        if any(s["usable"] for s in shots):
            yield {**member, "shots": shots}


def build(*, blocks: Path, annotations: Path, ledger_shots: Path, output_dir: Path,
          validation_fraction: float, split_seed: int,
          min_laion_aesthetic: float, min_dover_technical: float, min_mean_luma: float, tokenizer_path: Path) -> dict:
    token_counter = load_umt5_token_counter(tokenizer_path)
    base = members(sl.read_blocks(blocks), sl.read_annotations(annotations), sl.read_jsonl(ledger_shots),
                   validation_fraction=validation_fraction, split_seed=split_seed)
    rows = list(select_quality_members(base, min_laion_aesthetic=min_laion_aesthetic,
                                      min_dover_technical=min_dover_technical, min_mean_luma=min_mean_luma, token_counter=token_counter))
    summary = coverage(rows)
    summary.update(schema_version=1, validation_fraction=validation_fraction, split_seed=split_seed,
                   quality={"min_laion_aesthetic": min_laion_aesthetic, "min_dover_technical": min_dover_technical,
                            "min_mean_luma": min_mean_luma},
                   text={"tokenizer_path": str(tokenizer_path), "token_budget": UMT5_PROMPT_TOKEN_BUDGET},
                   inputs={"blocks": str(blocks), "annotations": str(annotations), "ledger_shots": str(ledger_shots)})
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        path = output_dir / f"{split}.jsonl"
        tmp = path.with_suffix(".jsonl.partial")
        tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows if row["split"] == split))
        tmp.replace(path)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary
