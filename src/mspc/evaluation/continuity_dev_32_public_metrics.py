"""Build renderer public-metrics input from Wan continuity_dev_32 generated manifests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from mspc.evaluation.character_queries import character_queries_by_case_id


# schema_version string is a data-contract value and stays unchanged.
CONTINUITY_DEV_32_METRICS_INPUT_SCHEMA = "wan22_controlled32_public_metrics_input_v1"
#: The twelve-column protocol reads one more per-case field than the six-column
#: one did -- ``character_queries``, the open-vocabulary detector's text -- so it
#: writes its own schema string.  The two are mutually exclusive the same way
#: the v2 and v3 protocols (``public_metrics_v2`` / ``public_metrics_v3``) are.
V3_METRICS_INPUT_SCHEMA = "renderer_public_metrics_input_v3"
#: The non-VLM scoring input of ``script_state_bench_v1``: no detector queries,
#: planned per-shot frame counts joined from the benchmark.
FRAME_COUNT_SOURCE = "generated_vae_aligned"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _shots_from_manifest_row(row: Mapping[str, Any], *, label: str) -> list[dict[str, Any]]:
    explicit = row.get("shots")
    if isinstance(explicit, list) and explicit:
        shots: list[dict[str, Any]] = []
        for index, shot in enumerate(explicit):
            if not isinstance(shot, Mapping):
                raise ValueError(f"{label}.shots[{index}] must be an object")
            shots.append(
                {
                    "shot_id": _text(shot.get("shot_id"), f"{label}.shots[{index}].shot_id"),
                    "prompt": _text(shot.get("prompt"), f"{label}.shots[{index}].prompt"),
                    "frame_count": shot.get("frame_count"),
                }
            )
            if type(shots[-1]["frame_count"]) is not int or shots[-1]["frame_count"] <= 0:
                raise ValueError(
                    f"{label}.shots[{index}].frame_count must be a positive integer"
                )
        return shots
    shot_ids = row.get("shot_ids")
    shot_prompts = row.get("shot_prompts")
    shot_groups = row.get("shot_groups")
    if not isinstance(shot_ids, list) or not isinstance(shot_prompts, list) or not isinstance(shot_groups, list):
        raise ValueError(
            f"{label} must include shots or shot_ids+shot_prompts+shot_groups"
        )
    if not (len(shot_ids) == len(shot_prompts) == len(shot_groups)):
        raise ValueError(f"{label} shot_ids, shot_prompts, and shot_groups must align")
    shots = []
    for index, (shot_id, prompt, group) in enumerate(zip(shot_ids, shot_prompts, shot_groups)):
        if not isinstance(group, list) or len(group) != 2:
            raise ValueError(f"{label}.shot_groups[{index}] must be [start, end]")
        start, end = group
        if type(start) is not int or type(end) is not int or end <= start:
            raise ValueError(f"{label}.shot_groups[{index}] is not a valid half-open span")
        shots.append(
            {
                "shot_id": _text(shot_id, f"{label}.shot_ids[{index}]"),
                "prompt": _text(prompt, f"{label}.shot_prompts[{index}]"),
                "frame_count": end - start,
            }
        )
    return shots


def public_metrics_cases_from_generated_manifest(
    manifest: Sequence[Mapping[str, Any]],
    *,
    arm_id: str,
    generation_seed: int,
) -> list[dict[str, Any]]:
    if type(generation_seed) is not int:
        raise ValueError("generation_seed must be an integer")
    arm = _text(arm_id, "arm_id")
    if not manifest:
        raise ValueError("generated manifest is empty")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(manifest):
        if not isinstance(row, Mapping):
            raise ValueError(f"generated_eval_manifest[{index}] must be an object")
        story_id = _text(row.get("case_id"), f"generated_eval_manifest[{index}].case_id")
        if story_id in seen:
            raise ValueError(f"duplicate generated case_id={story_id!r}")
        seen.add(story_id)
        seed = row.get("generation_seed", generation_seed)
        if seed != generation_seed:
            raise ValueError(
                f"{story_id}: generation_seed {seed!r} does not match requested {generation_seed}"
            )
        video_path = _text(row.get("video_path"), f"{story_id}.video_path")
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"{story_id}: generated video is missing: {video_path}")
        shots = _shots_from_manifest_row(row, label=story_id)
        frame_count_source = row.get("frame_count_source", FRAME_COUNT_SOURCE)
        if frame_count_source != FRAME_COUNT_SOURCE:
            raise ValueError(
                f"{story_id}: frame_count_source must be {FRAME_COUNT_SOURCE!r} "
                f"(VAE-aligned generated lengths), got {frame_count_source!r}"
            )
        cases.append(
            {
                "case_id": f"{story_id}-{arm}-seed-{generation_seed}",
                "story_id": story_id,
                "arm_id": arm,
                "generation_seed": generation_seed,
                "video_path": video_path,
                "story_prompt": _text(row.get("story_prompt"), f"{story_id}.story_prompt"),
                "shots": shots,
                "shot_count": len(shots),
                "frame_count_source": FRAME_COUNT_SOURCE,
            }
        )
    return cases


def write_public_metrics_input(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    arm_id: str,
    generation_seed: int,
) -> dict[str, Any]:
    source = Path(manifest_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"generated eval manifest is missing: {source}")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("generated_eval_manifest.json must be a JSON array")
    cases = public_metrics_cases_from_generated_manifest(
        raw,
        arm_id=arm_id,
        generation_seed=generation_seed,
    )
    payload = {
        "schema_version": CONTINUITY_DEV_32_METRICS_INPUT_SCHEMA,
        "arm_id": arm_id,
        "generation_seed": generation_seed,
        "frame_count_source": FRAME_COUNT_SOURCE,
        "cases": cases,
    }
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"public metrics input already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _load_benchmark_cases(benchmark_path: str | Path) -> tuple[Path, list[Mapping[str, Any]]]:
    path = Path(benchmark_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"benchmark file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path}: cases must be a non-empty array")
    return path, cases


def benchmark_case_texts(benchmark_path: str | Path) -> dict[str, dict[str, Any]]:
    """Each case's own ``story_prompt`` and per-shot ``prompt``, keyed by ``case_id``.

    Every benchmark schema this project scores (``dev32``, ``real_holdout_96``,
    ``formal96_candidate``) stores these two fields the same way, so this reads
    them schema-agnostically -- unlike :func:`character_queries_by_case_id`,
    which parses a different source field per schema.  A story's text is a
    property of what was asked for, not of what a particular arm's render
    manifest carries: an arm may render from an augmented prompt (for example
    the text visual-setting control, which appends generated setting text to
    each shot prompt) while still being scored against the original
    benchmark's text, the same way character_queries already is.
    """

    path, cases = _load_benchmark_cases(benchmark_path)
    texts: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError(f"{path}: cases[{index}] must be an object")
        case_id = _text(case.get("case_id"), f"{path}:cases[{index}].case_id")
        if case_id in texts:
            raise ValueError(f"{path}: duplicate case_id {case_id!r}")
        story_prompt = _text(case.get("story_prompt"), f"{path}:{case_id}.story_prompt")
        shots = case.get("shots")
        if not isinstance(shots, list) or not shots:
            raise ValueError(f"{path}:{case_id}.shots must be a non-empty array")
        shot_prompts: dict[str, str] = {}
        for shot_index, shot in enumerate(shots):
            if not isinstance(shot, Mapping):
                raise ValueError(f"{path}:{case_id}.shots[{shot_index}] must be an object")
            shot_id = _text(shot.get("shot_id"), f"{path}:{case_id}.shots[{shot_index}].shot_id")
            if shot_id in shot_prompts:
                raise ValueError(f"{path}:{case_id}: duplicate shot_id {shot_id!r}")
            shot_prompts[shot_id] = _text(shot.get("prompt"), f"{path}:{case_id}.shots[{shot_index}].prompt")
        texts[case_id] = {"story_prompt": story_prompt, "shot_prompts": shot_prompts}
    return texts


def write_v3_public_metrics_input(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    arm_id: str,
    generation_seed: int,
    benchmark_path: str | Path,
) -> dict[str, Any]:
    """Write the twelve-column scoring input, carrying identity queries per case.

    The identity queries and every text-reading field (``story_prompt`` and
    each shot's ``prompt``) come from the benchmark file rather than the
    generated manifest: they are a property of the story being asked for, not
    of the arm that rendered it, so every arm scores the same text and the
    identity, ViCLIP, and judge-prompt columns stay arm-independent -- a
    render manifest whose own prompts were augmented for generation (for
    example the text visual-setting control) is still scored against the
    original benchmark text, so it is comparable to arms rendered straight
    from that benchmark.  A case the benchmark does not declare is a
    compilation error; a case it declares with no recurring character gets an
    empty list, which the backend reads as ``ineligible_no_character_asset``.
    """

    source = Path(manifest_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"generated eval manifest is missing: {source}")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("generated_eval_manifest.json must be a JSON array")
    cases = public_metrics_cases_from_generated_manifest(
        raw,
        arm_id=arm_id,
        generation_seed=generation_seed,
    )
    queries = character_queries_by_case_id(benchmark_path)
    texts = benchmark_case_texts(benchmark_path)
    for case in cases:
        story_id = case["story_id"]
        if story_id not in queries:
            raise ValueError(
                f"{story_id} is not declared by {benchmark_path}; the identity column "
                "cannot be compiled for it"
            )
        case["character_queries"] = list(queries[story_id])
        if story_id not in texts:
            raise ValueError(
                f"{story_id} is not declared by {benchmark_path}; its canonical text "
                "cannot be compiled for it"
            )
        case_text = texts[story_id]
        case["story_prompt"] = case_text["story_prompt"]
        for shot in case["shots"]:
            shot_id = shot["shot_id"]
            if shot_id not in case_text["shot_prompts"]:
                raise ValueError(
                    f"{story_id}.{shot_id} is not declared by {benchmark_path}; its canonical "
                    "shot prompt cannot be compiled for it"
                )
            shot["prompt"] = case_text["shot_prompts"][shot_id]
    payload = {
        "schema_version": V3_METRICS_INPUT_SCHEMA,
        "arm_id": arm_id,
        "generation_seed": generation_seed,
        "frame_count_source": FRAME_COUNT_SOURCE,
        "benchmark_path": str(Path(benchmark_path).expanduser().resolve()),
        "cases": cases,
    }
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"public metrics input already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def merge_generated_eval_manifests(
    shard_dirs: Sequence[str | Path],
    output_path: str | Path,
) -> list[dict[str, Any]]:
    """Concatenate finished shard manifests into the one file the v3 scorer reads.

    Each shard writes ``generated_eval_manifest.json`` only when it finishes, so a
    missing file means that shard is not done. Case ids must be unique across
    shards. The merged file is a JSON array of the same rows, with the paths the
    shards already wrote -- except the memory-arm feed record
    ``memory_shots[k].memory_frames[i].png``, which a shard writes relative to its own
    manifest directory (``mspc.evaluation.shot_boundary_copy`` resolves it against the
    directory of the manifest it reads). Those are rebased onto the merged manifest's
    directory, so the copy readout finds the fed frames from the merged file.
    """

    if not shard_dirs:
        raise ValueError("at least one shard directory is required")
    rows: list[dict[str, Any]] = []
    shard_rows: list[tuple[Path, dict[str, Any]]] = []
    seen: set[str] = set()
    for shard_dir in shard_dirs:
        manifest_path = Path(shard_dir).expanduser() / "generated_eval_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{manifest_path.parent}: no generated_eval_manifest.json; "
                "a shard writes it only when it finishes"
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"{manifest_path} must be a non-empty JSON array")
        for index, row in enumerate(payload):
            if not isinstance(row, dict):
                raise ValueError(f"{manifest_path}[{index}] must be an object")
            case_id = row.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"{manifest_path}[{index}].case_id must be a non-empty string")
            if case_id in seen:
                raise ValueError(f"duplicate case_id {case_id!r} across shard manifests")
            seen.add(case_id)
            rows.append(row)
            shard_rows.append((manifest_path.parent, row))
    destination = Path(output_path).expanduser()
    for shard_manifest_dir, row in shard_rows:
        for shot in row.get("memory_shots", ()):
            for frame in shot["memory_frames"]:
                frame["png"] = os.path.relpath(
                    shard_manifest_dir / frame["png"], destination.parent
                )
    if destination.exists():
        raise FileExistsError(f"merged manifest already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return rows


def parser():
    import argparse

    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument("--manifest", required=True, type=Path)
    arg_parser.add_argument("--output", required=True, type=Path)
    arg_parser.add_argument("--arm-id", required=True)
    arg_parser.add_argument("--generation-seed", required=True, type=int)
    return arg_parser


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    payload = write_public_metrics_input(
        args.manifest,
        args.output,
        arm_id=args.arm_id,
        generation_seed=args.generation_seed,
    )
    print(json.dumps({"status": "completed", "cases": len(payload["cases"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
