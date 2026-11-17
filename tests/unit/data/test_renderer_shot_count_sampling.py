"""真实源成员验证镜数分层：共同源网格、完整镜、长镜过滤与冻结表前缀。"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mspc.data.renderer_training_members import draw_sampling, write_drawn_sampling
from mspc.data.scene_training_manifest import ANNOTATION_MEDIA
from mspc.data.shot_targets import member_target_frame_counts, shot_key
from mspc.training.block_dataset import BlockRow, BlockWindowSampler


MEMBERS = json.loads((Path(__file__).parent / "fixtures/non_overlapping_training_members_real.json").read_text())["members"]
for _member in MEMBERS:
    for _position, _shot in enumerate(_member["shots"]):
        _shot.update(sid=_member["case_id"], scene_id=_member["key"], source_shot_index=_position)
EXTRA_REAL = json.loads((Path(__file__).parent / "fixtures/shot_count_weighted_members_real.json").read_text())
MEMBERS += EXTRA_REAL["members"]
MEDIA720 = {**ANNOTATION_MEDIA, "height": 704, "width": 1280, "target_fps": 24, "max_rgb_frames": 360}


def options(*, media=ANNOTATION_MEDIA, sample_budget=24000):
    return {"media": media, "seed": 42, "sampling_scheme": "shot_count_weighted_windows",
            "partition_max_rgb_frames": 128, "max_source_shot_seconds": 10,
            "shot_count_weights": [15, 25, 30, 30], "frame_buckets": None, "bucket_weights": None,
            "max_shots": None, "sample_budget": sample_budget, "forced_samples": None}


def identity(sample):
    return sample["case_id"], tuple(sample["span"])


def test_source_grid_draws_are_shared_by_both_media_and_include_every_legal_continuous_window():
    before = deepcopy(MEMBERS)
    old, report = draw_sampling(MEMBERS, **options())
    new, new_report = draw_sampling(MEMBERS, **options(media=MEDIA720))
    assert list(map(identity, old)) == list(map(identity, new))
    assert report["source_sampling_media"] == ANNOTATION_MEDIA
    counts = Counter(min(s["span"][1] - s["span"][0], 4) for s in old)
    assert all(abs(counts[k] / len(old) - w) < .01 for k, w in enumerate([.15, .25, .30, .30], 1))
    expected, dropped = set(), 0
    for member in MEMBERS:
        for start, shot in enumerate(member["shots"]):
            if not shot["usable"] or shot["seconds"] > 10:
                dropped += int(shot["usable"] and shot["seconds"] > 10)
                continue
            total = 0
            for stop in range(start, len(member["shots"])):
                current = member["shots"][stop]
                if not current["usable"] or current["seconds"] > 10:
                    break
                total += current["frames"]
                if total > 128 and stop != start:
                    break
                expected.add((member["case_id"], (start, stop + 1)))
                if total > 128:
                    break
    assert set(map(identity, old)) == expected
    assert report["source_duration_dropped_usable_shots"] == dropped
    assert sum(report["legal_windows_by_shot_count_layer"]) == len(expected)
    assert new_report["source_window_target_rgb_frames_max"] > report["source_window_target_rgb_frames_max"]
    assert MEMBERS == before
    for example in EXTRA_REAL["examples"].values():
        assert (example["case_id"], tuple(example["span"])) in set(map(identity, old))
    assert max(s["span"][1] - s["span"][0] for s in old) >= 5


def test_target_capacity_rejects_complete_source_windows_instead_of_changing_the_table():
    _, report = draw_sampling(MEMBERS, **options(media=MEDIA720, sample_budget=1))
    capacity = report["source_window_target_rgb_frames_max"] - 4
    with pytest.raises(ValueError, match="complete shared window maps to.*do not split"):
        draw_sampling(MEMBERS, **options(media={**MEDIA720, "max_rgb_frames": capacity}, sample_budget=1))


def test_four_plus_layer_has_no_shot_count_cap_and_is_uniform_over_windows():
    # 纯数学窗口枚举可用合成输入；源parser边界由真实成员测试覆盖。
    row = BlockRow("short-shots", (17,) * 7, tuple(range(7)))
    sampler = BlockWindowSampler([row], frame_buckets=[128], bucket_weights=None,
        max_shots=None, seed=42, shot_count_weights=[0, 0, 0, 1])
    counts = Counter(span for _, span, _ in (sampler.draw(i) for i in range(24000)))
    assert len(counts) == 10
    assert {stop - start for start, stop in counts} == {4, 5, 6, 7}
    assert all(abs(n / 24000 - .1) < .01 for n in counts.values())


def test_long_complete_singleton_is_not_cropped_or_joined_to_neighbors():
    row = BlockRow("complete-long", (17, 157, 21), (0, 1, 2))
    sampler = BlockWindowSampler([row], frame_buckets=[128], bucket_weights=None,
        max_shots=None, seed=42, shot_count_weights=[1, 0, 0, 0])
    assert {span for _, span, _ in (sampler.draw(i) for i in range(100))} == {(0, 1), (1, 2), (2, 3)}


def test_deterministic_written_slice_is_a_byte_prefix_and_exposure_reports_physical_shots(tmp_path):
    path = tmp_path / "members.jsonl"
    path.write_text("".join(json.dumps(m) + "\n" for m in MEMBERS))
    kw = options(sample_budget=2000)
    full = write_drawn_sampling(members_path=path, output=tmp_path / "full.jsonl", limit=None, **kw)
    part = write_drawn_sampling(members_path=path, output=tmp_path / "part.jsonl", limit=200, **kw)
    assert (tmp_path / "full.jsonl").read_text().splitlines()[:200] == (tmp_path / "part.jsonl").read_text().splitlines()
    assert sum(full["shot_count_layer_counts"].values()) == 2000
    assert part["rows_written"] == 200
    by_case = {m["case_id"]: m for m in MEMBERS}
    sampled = [json.loads(line) for line in (tmp_path / "full.jsonl").read_text().splitlines()]
    assert full["unique_source_target_shots"] == len({shot_key(by_case[s["case_id"]]["shots"][p])
        for s in sampled for p in range(*s["span"])})
    assert full["exposure"]["target_shot_presentations"] == sum(s["span"][1] - s["span"][0] for s in sampled)
    assert draw_sampling(MEMBERS, **kw)[0] == sampled


@pytest.mark.parametrize("weights", [None, [], [15, 25, 30], [0, 0, 0, 0], [1, 2, 3, -1],
                                     [1, 2, 3, float("nan")], [1, 2, 3, True]])
def test_shot_count_weights_are_required_and_validated(weights):
    with pytest.raises(ValueError, match="shot_count_weights"):
        draw_sampling(MEMBERS, **{**options(sample_budget=1), "shot_count_weights": weights})


@pytest.mark.parametrize("limit", [None, 0, -1, True, float("inf"), float("nan")])
def test_source_duration_filter_requires_explicit_finite_positive_limit(limit):
    with pytest.raises(ValueError, match="max_source_shot_seconds"):
        draw_sampling(MEMBERS, **{**options(sample_budget=1), "max_source_shot_seconds": limit})


def test_missing_layer_or_mixed_sampling_controls_fail_explicitly():
    with pytest.raises(ValueError, match="layer 2"):
        BlockWindowSampler([BlockRow("single", (157,), (0,))], frame_buckets=[128], bucket_weights=None,
            max_shots=None, seed=42, shot_count_weights=[15, 25, 30, 30])
    with pytest.raises(ValueError, match="do not use duration buckets"):
        draw_sampling(MEMBERS, **{**options(sample_budget=1), "frame_buckets": [128]})
    with pytest.raises(ValueError, match="duplicate or empty member"):
        draw_sampling([MEMBERS[0], MEMBERS[0]], **options(sample_budget=1))


def _annotated_real_members():
    """真实块与块标注（sampling_ledger）组成带 ``annotation`` 的成员，与训练成员同形。"""
    ledger = Path(__file__).parents[2] / "fixtures/sampling_ledger"
    annotations = {row["key"]: row["annotation"] for row in map(json.loads, (ledger / "block-annotations.jsonl")
                                                                .read_text().splitlines()) if row["annotation"]}
    members = []
    for block in map(json.loads, (ledger / "blocks.jsonl").read_text().splitlines()):
        if block["key"] not in annotations or len(annotations[block["key"]]["shots"]) != len(block["scene"]["shots"]):
            continue  # 训练成员要求块与标注镜数一致（BlockDataset 同样核）
        scene = block["scene"]
        members.append({"case_id": block["case_id"], "key": block["key"], "film_key": block["key"].rsplit("-b", 1)[0],
                        "source": block["source"], "scene": {k: v for k, v in scene.items() if k != "shots"},
                        "shots": [{**shot, "frames": shot["render_frames"], "usable": True} for shot in scene["shots"]],
                        "annotation": annotations[block["key"]]})
    return members


def test_long_single_shots_are_long_shots_with_action_or_change_on_real_blocks(tmp_path):
    members = _annotated_real_members()
    by_key = {member["key"]: member for member in members}
    kw = {**options(sample_budget=4000), "long_single_shot": (0.5, 80)}
    samples, report = draw_sampling(members, **kw)
    # 真实失败行：cinedance-012911 镜 4（85 帧、无人活动、无变化）与 cinedance-017259 镜 9（105 帧、只说话）
    # 只按帧数会被当成长单镜；按定义只剩两条有动作的长镜。
    long_pool = {(by_key["cinedance-008884-b0000"]["case_id"], 1), (by_key["cinedance-012911-b0001"]["case_id"], 0)}
    assert report["long_single_shot"]["legal_windows"] == len(long_pool)
    singles = [(s["case_id"], s["span"][0]) for s in samples if s["span"][1] - s["span"][0] == 1]
    inactive = {(by_key["cinedance-012911-b0001"]["case_id"], 3), (by_key["cinedance-017259-b0000"]["case_id"], 8)}
    assert sum(single in long_pool for single in singles) / len(singles) > 0.5
    assert sum(single in inactive for single in singles) / len(singles) < 0.1
    assert report["long_single_shot"]["samples"] == sum(single in long_pool for single in singles)
    plain, _ = draw_sampling(members, **options(sample_budget=4000))
    assert [identity(a) for a in plain if a["span"][1] - a["span"][0] > 1] == [
        identity(b) for b in samples if b["span"][1] - b["span"][0] > 1]
    # 写表入口只带每镜 activity／changes，不常驻整块标注。
    path = tmp_path / "members.jsonl"
    path.write_text("".join(json.dumps(m) + "\n" for m in members))
    written = write_drawn_sampling(members_path=path, output=tmp_path / "sampling.jsonl", limit=None, **kw)
    assert written["long_single_shot"] == report["long_single_shot"]
    assert [json.loads(line) for line in (tmp_path / "sampling.jsonl").read_text().splitlines()] == samples
    without_annotation = [{k: v for k, v in m.items() if k != "annotation"} for m in members]
    with pytest.raises(KeyError):
        draw_sampling(without_annotation, **kw)


def test_unusable_shot_without_changes_does_not_block_long_single_draw(tmp_path):
    # 真实失败行（11-13 190 上 200 行切片）：训练成员 story_block_v1:cinedance:00a1f4fe…:block0000:0003-0011 镜 0
    # 不可用、块标注只有 activity 没有 changes，长单镜抽签读它时 KeyError。不可用镜本来不是候选，不该被读。
    members = _annotated_real_members()
    member = next(m for m in members if m["key"] == "cinedance-012911-b0001")
    member["shots"][3]["usable"] = False
    member["annotation"]["shots"][3] = {k: v for k, v in member["annotation"]["shots"][3].items() if k != "changes"}
    kw = {**options(sample_budget=400), "long_single_shot": (0.5, 80)}
    samples, report = draw_sampling(members, **kw)
    assert all(not (s["case_id"] == member["case_id"] and s["span"][0] <= 3 < s["span"][1]) for s in samples)
    path = tmp_path / "members.jsonl"
    path.write_text("".join(json.dumps(m) + "\n" for m in members))
    written = write_drawn_sampling(members_path=path, output=tmp_path / "sampling.jsonl", limit=None, **kw)
    assert written["long_single_shot"] == report["long_single_shot"]
