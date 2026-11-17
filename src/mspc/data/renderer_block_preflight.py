"""以训练同一块 reader 做 CPU 文字与张量验收；dataset JSON 显式给所有构造参数，不要求训练 recipe。"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from mspc.data.shot_text import (UMT5_PROMPT_TOKEN_BUDGET, reference_setting, reference_text_with_entity_spans,
                                shot_object_phrases)
from mspc.training.block_dataset import BlockDataset


def _span_errors(compiled, names):
    errors = []
    text = compiled["text"]
    for position, span in enumerate(compiled["entity_spans"]):
        if (not isinstance(span, Mapping) or set(span) != {"name", "start", "end", "source"}
                or not isinstance(span["name"], str) or span["name"] not in names
                or type(span["start"]) is not int or type(span["end"]) is not int
                or not 0 <= span["start"] < span["end"] <= len(text)
                or not isinstance(span["source"], str) or not span["source"]):
            errors.append({"span_index": position, "span": span, "error": "invalid entity span in actual text"})
    return errors


def inspect_dataset(*, dataset_cfg: Mapping[str, Any], output: Path, tensor_samples: int,
                    sample_indices: Sequence[int] | None = None) -> dict:
    """完整reader/标签合同不变，只限制审计遍历的原始索引；0个tensor样本只走文字与元数据。"""
    if type(tensor_samples) is not int or tensor_samples < 0:
        raise ValueError("tensor_samples must be a nonnegative int")
    output.mkdir(parents=True, exist_ok=True)
    dataset = BlockDataset(**dataset_cfg)
    indices = dataset.eligible_sample_indices if sample_indices is None else tuple(sample_indices)
    if (not indices or any(type(index) is not int or not 0 <= index < len(dataset) for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("sample_indices must be nonempty distinct original dataset indices")
    if set(indices) & dataset.excluded_sample_indices:
        raise ValueError("actual consumer audit cannot include text-admission excluded sample indices")
    references, starts, shots, tokens, blocks, overflow = Counter(), Counter(), Counter(), [], Counter(), []
    actual_targets, actual_references, spans, source_statuses, span_errors = [], [], Counter(), Counter(), []
    target_exposures, rgb_exposures, latent_exposures = Counter(), 0, 0
    for index in indices:
        text = dataset.text(index)
        meta = text.metadata
        draw = dataset.draw(index)
        reference_count = len(draw.references)
        actual = meta["actual_token_counts"]
        actual_references.extend(actual[:reference_count])
        actual_targets.extend(actual[reference_count:])
        annotation = dataset.annotations[draw.row.case_id]
        registered = {entry["name"] for kind in ("characters", "objects", "places") for entry in annotation[kind]}
        if tuple(row["text"] for row in text.compiled_targets) != text.prompts[reference_count:]:
            span_errors.append({"data_id": index, "kind": "target", "error": "compiled text differs from actual prompt"})
        for compiled in text.compiled_targets:
            spans["target_rows"] += 1
            spans["target_spans"] += len(compiled["entity_spans"])
            span_errors.extend({"data_id": index, "kind": "target", "shot_index": compiled["shot_index"], **error}
                               for error in _span_errors(compiled, registered))
        if dataset.reference_entities is not None:
            labels = dataset.reference_entities.for_draw(index, draw,
                source_frames=dataset.reference_source_frames(draw))
            source_statuses.update(row["status"] for row in labels)
            spans["labelled_reference_rows"] += len(labels)
            if not meta["reference_text_dropped"]:
                for slot, row in enumerate(labels):
                    compiled = reference_text_with_entity_spans(row["entities"], characters=annotation["characters"],
                        setting=reference_setting(annotation, row["shot"] - 1),
                        shot_objects=shot_object_phrases(annotation, row["shot"] - 1)
                        if dataset.shot_objects_in_rows else ())
                    spans["reference_rows"] += 1
                    spans["reference_spans"] += len(compiled["entity_spans"])
                    if compiled["text"] != text.prompts[slot]:
                        span_errors.append({"data_id": index, "kind": "reference", "slot": slot + 1,
                                            "error": "compiled text differs from actual prompt"})
                    span_errors.extend({"data_id": index, "kind": "reference", "slot": slot + 1, **error}
                        for error in _span_errors(compiled, {entry["name"] for entry in row["entities"]}))
        else:
            spans["generic_reference_rows"] += reference_count
        counts = draw.row.frame_counts[slice(*draw.span)]
        rgb_exposures += sum(counts)
        latent_exposures += sum((count - 1) // 4 + 1 for count in counts)
        target_exposures.update((draw.row.case_id, shot) for shot in range(*draw.span))
        references[len(meta["references"])] += 1
        starts[meta["span"][0]] += 1
        shots[meta["span"][1] - meta["span"][0]] += 1
        tokens.extend(meta["actual_token_counts"])
        blocks[meta["case_id"]] += 1
        if any(n > UMT5_PROMPT_TOKEN_BUDGET for n in meta["actual_token_counts"]):
            overflow.append({"data_id": index, **meta})
    (output / "text_overflow.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in overflow))
    (output / "text_span_errors.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in span_errors))
    blocked = {record["data_id"] for record in overflow}
    blocked.update(record["data_id"] for record in span_errors)
    readable = [i for i in indices if i not in blocked]
    picked = readable[::max(1, len(readable) // tensor_samples)][:tensor_samples] if tensor_samples else []
    shapes, actual_lengths = Counter(), []
    output_frames = 0
    if picked:
        import torch
        lengths = dataset.latent_frame_upper_bounds()
    for index in picked:
        sample = dataset[index]
        latents = sample["input_latents"]
        count = sample.get("memory_frame_count", 0)
        output_frames += sample["_dataset_timing"].get("reference_output_frames", 0)
        groups = sample.get("latent_shot_groups", [[0, int(latents.shape[2])]])
        if len(groups) != len(sample["prompt"]) or latents.shape[2] + count > lengths[index]:
            raise ValueError(f"sample {index}: prompt/visual groups or bucket length differs")
        if (list(latents.shape) != [1, dataset.latent_channels, latents.shape[2], *dataset.latent_spatial_shape]
                or not torch.isfinite(latents).all()):
            raise ValueError(f"sample {index}: invalid latent tensor")
        if set(sample["reference_shot_ids"]) & set(sample["target_shot_ids"]):
            raise ValueError(f"sample {index}: reference is a target shot")
        shapes[str(list(latents.shape))] += 1
        actual_lengths.append(int(latents.shape[2]) + count)
    summary = {"dataset": dataset.metadata_summary,
        "scope": {"dataset_samples": len(dataset), "sample_indices": None if sample_indices is None else list(indices)},
        "exposure": {"samples": len(indices), "target_shot_exposures": sum(target_exposures.values()),
            "distinct_target_shots": len(target_exposures), "target_rgb_frames": rgb_exposures,
            "target_latent_frames": latent_exposures,
            "reference_frame_exposures": sum(count * samples for count, samples in references.items())},
        "reference_io": {"output_frames": output_frames},
        "text_pass": {"samples": len(indices), "within_budget": len(indices) - len(overflow),
            "reference_count_distribution": dict(sorted(references.items())),
            "start_distribution": dict(sorted(starts.items())), "target_shot_distribution": dict(sorted(shots.items())),
            "distinct_blocks": len(blocks), "rows": len(tokens), "token_max": max(tokens), "token_median": median(tokens),
            "actual_target_token_max": max(actual_targets, default=0),
            "actual_reference_token_max": max(actual_references, default=0)},
        "text_spans": {**spans, "errors": len(span_errors), "source_label_statuses": dict(source_statuses)},
        "text_overflow": {"token_budget": UMT5_PROMPT_TOKEN_BUDGET, "samples": len(overflow)},
        "tensor_pass": {"samples": len(picked), "shapes": dict(shapes),
            "latent_frame_min": min(actual_lengths, default=None),
            "latent_frame_max": max(actual_lengths, default=None)}}
    (output / "construction.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True,
                        help="BlockDataset 的显式构造参数 JSON，路径相对当前目录；不加载模型")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-samples", type=int, required=True,
                        help="张量抽验数；0只验文字/元数据，不读目标缓存或解码RGB")
    parser.add_argument("--sample-indices-json", type=Path,
                        help="原dataset索引的显式JSON数组；从实际cache-stage的sample_indices投影，不重编号")
    args = parser.parse_args(argv)
    summary = inspect_dataset(dataset_cfg=json.loads(args.dataset_json.read_text()), output=args.output,
                              tensor_samples=args.tensor_samples,
                              sample_indices=None if args.sample_indices_json is None else
                              json.loads(args.sample_indices_json.read_text()))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["text_overflow"]["samples"] or summary["text_spans"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
