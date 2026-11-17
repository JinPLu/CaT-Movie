"""带规划器 token 的段数据集（点控制两组）：参考帧与其余各组相同（随机选帧）；token 只放本镜一处，落在该镜第一个潜帧
与（抖动后的）镜首点格，特征取该镜此人的镜首条目；镜首为 null 的人没有 token，另按人随机丢点；token 整组丢弃按概率；
标签与特征对不上、标签段与采样段对不上都报错；点控制与特征版放置一致、内容是人物编号。

标签行按 ``build_labels`` 的输出形状手写（生产端由 tests/unit/data/test_history_reading_labels.py 用真实块核过），
特征是序号张量，只核接线。"""
import numpy as np
import pytest
import torch

from mspc.data.shot_targets import latent_frames
from mspc.models.planner_tokens import point_cell
from mspc.training.block_dataset import BlockDataset
from tests.unit.training.test_block_dataset import controlled_masks, cpu_frames, inputs, lines, write  # noqa: F401

FEATURE_DIM = 8


def _labels(span):
    start, end = span
    starts = []
    for shot in range(start, end):
        people = [{"name": "Character 1", "point_2d": [500, 500]}]
        if shot == 1:
            people.append({"name": "Character 2", "point_2d": None})
        starts.append({"shot_index": shot, "characters": people})
    return {"schema": 1, "span": list(span), "starts": starts}


@pytest.fixture
def planner_inputs(inputs, tmp_path):
    samples = lines(tmp_path / "sampling.jsonl")
    features_root = tmp_path / "features"
    features_root.mkdir()
    rows = []
    for sample in samples:
        row = _labels(sample["span"])
        rows.append({"sample_index": sample["sample_index"], "case_id": sample["case_id"], **row})
        entries = [(s["shot_index"], c["name"]) for s in row["starts"] for c in s["characters"]]
        shot_value = np.arange(len(entries) * FEATURE_DIM, dtype=np.float16).reshape(len(entries), FEATURE_DIM)
        np.savez(features_root / f"{sample['sample_index']:06d}.npz", shot_name_layer19=shot_value,
                 shot_index=np.array([s for s, _ in entries], dtype=np.int64),
                 shot_name=np.array([n for _, n in entries], dtype=str))
    write(tmp_path / "labels.jsonl", rows)
    return {**inputs,
            "planner_tokens": {"content": "planner_features", "labels_path": str(tmp_path / "labels.jsonl"),
                               "features_root": str(features_root), "feature_layer": 19, "feature_pooling": "name",
                               "point_jitter": 0.1, "point_dropout_p": 0.1, "token_dropout_p": 0.2}}


def test_tokens_follow_the_labels_and_references_stay_random(planner_inputs):
    dataset = BlockDataset(**planner_inputs)
    plain = BlockDataset(**{key: value for key, value in planner_inputs.items() if key != "planner_tokens"})
    dropped, missing = 0, 0
    for i in range(len(dataset)):
        sample, draw = dataset[i], dataset.draw(i)
        assert draw.references == plain.draw(i).references  # 参考与不带 token 的同一套随机选帧
        assert sample["reference_images"].shape[0] == len(draw.references)
        assert "target_layouts" not in sample and "shot_references" not in sample
        tokens = sample["planner_tokens"]
        if tokens is None:
            dropped += 1
            continue
        start, end = draw.span
        lengths = [latent_frames(n) for n in draw.row.frame_counts[start:end]]
        # 每个目标镜至多一个 Character 1 的 token（按人丢点会少）；Character 2 镜首为 null，没有 token。
        placed = [(k, t) for _, k, t, _, _ in tokens["shot_tokens"]]
        assert set(placed) <= {(k, sum(lengths[:k])) for k in range(end - start)}
        missing += (end - start) - len(placed)
        for _, _, _, row, col in tokens["shot_tokens"]:
            low, high = point_cell([400, 400]), point_cell([600, 600])
            assert low[0] <= row <= high[0] and low[1] <= col <= high[1]  # 抖动不超过画面的 10%
        assert tokens["features"].shape[1] == FEATURE_DIM and set(tokens) == {"features", "shot_tokens"}
    assert 0 < dropped < len(dataset) // 2 and missing > 0
    again = BlockDataset(**planner_inputs)  # 同一种子同一下标，抖动、丢点、整组丢弃都复现
    for i in range(len(dataset)):
        a, b = dataset[i]["planner_tokens"], again[i]["planner_tokens"]
        assert (a is None) == (b is None) and (a is None or a["shot_tokens"] == b["shot_tokens"])


def test_labels_must_cover_every_sample(planner_inputs, tmp_path):
    rows = lines(tmp_path / "labels.jsonl")[:-1]
    write(tmp_path / "labels.jsonl", rows)
    with pytest.raises(ValueError, match="cover every sample"):
        BlockDataset(**planner_inputs)


def test_duplicate_label_index_is_rejected_before_it_can_overwrite_identity(planner_inputs, tmp_path):
    rows = lines(tmp_path / "labels.jsonl")
    duplicate = {**rows[0], "case_id": "a-different-story"}
    write(tmp_path / "labels.jsonl", [*rows, duplicate])
    with pytest.raises(ValueError, match="duplicate.*sample_index|duplicate.*index"):
        BlockDataset(**planner_inputs)


def test_label_case_must_match_even_when_index_and_span_match(planner_inputs, tmp_path):
    rows = lines(tmp_path / "labels.jsonl")
    rows[0]["case_id"] = "a-different-story"
    write(tmp_path / "labels.jsonl", rows)
    dataset = BlockDataset(**{**planner_inputs, "planner_tokens": {**planner_inputs["planner_tokens"],
                                                                  "token_dropout_p": 0.0}})
    with pytest.raises(ValueError, match="case"):
        dataset[0]


def test_label_span_must_match_the_sampled_span(planner_inputs, tmp_path):
    rows = lines(tmp_path / "labels.jsonl")
    rows[0]["span"] = [5, 6]
    write(tmp_path / "labels.jsonl", rows)
    dataset = BlockDataset(**{**planner_inputs, "planner_tokens": {**planner_inputs["planner_tokens"],
                                                                  "token_dropout_p": 0.0}})
    with pytest.raises(ValueError, match="label span"):
        dataset[0]


def test_a_label_entry_without_a_feature_is_an_error(planner_inputs, tmp_path):
    rows = lines(tmp_path / "labels.jsonl")
    index = rows[0]["sample_index"]
    with np.load(tmp_path / "features" / f"{index:06d}.npz") as data:
        arrays = {k: data[k] for k in data.files}
    arrays["shot_name"] = np.array(["Somebody else"] * len(arrays["shot_name"]), dtype=str)
    np.savez(tmp_path / "features" / f"{index:06d}.npz", **arrays)
    dataset = BlockDataset(**{**planner_inputs, "planner_tokens": {**planner_inputs["planner_tokens"],
                                                                  "token_dropout_p": 0.0}})
    with pytest.raises(ValueError, match="no feature"):
        dataset[index]


def test_point_control_uses_the_same_placement_with_person_numbers(planner_inputs):
    marker = {"content": "person_index", "labels_path": planner_inputs["planner_tokens"]["labels_path"],
              "point_jitter": 0.1, "point_dropout_p": 0.1, "token_dropout_p": 0.2}
    features_run = BlockDataset(**planner_inputs)
    marker_run = BlockDataset(**{**planner_inputs, "planner_tokens": marker})
    for i in range(len(marker_run)):
        a, b = features_run[i]["planner_tokens"], marker_run[i]["planner_tokens"]
        assert (a is None) == (b is None)
        if a is None:
            continue
        assert [r[1:] for r in a["shot_tokens"]] == [r[1:] for r in b["shot_tokens"]]
        assert b["features"].dtype == torch.long and set(b["features"].tolist()) <= {1, 2}
