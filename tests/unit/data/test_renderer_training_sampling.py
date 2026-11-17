import json
from collections import Counter

import pytest

from mspc.data.renderer_training_members import draw_sampling, forced_span, write_drawn_sampling
from mspc.data.scene_training_manifest import ANNOTATION_MEDIA
from mspc.training.block_dataset import BlockRow, BlockWindowSampler


@pytest.fixture(autouse=True)
def isolated_frame_plans(monkeypatch):
    """本文件只测抽签策略；真实源时间到目标格的计划由test_training_media独立覆盖。"""
    monkeypatch.setattr('mspc.data.renderer_training_members.member_target_frame_counts',
                        lambda member, *, media: tuple(shot['frames'] for shot in member['shots']))


def member(case, *, outfit_change_at=None, shots=4):
    """抽样是成员元数据上的纯函数；一个角色在每镜在场，可选在某镜换装。"""
    annotation_shots = []
    for index in range(shots):
        changes = ([{"name": "Character 1", "what": "outfit", "after": "a red coat"}]
                   if index == outfit_change_at else [])
        annotation_shots.append({"shot": index + 1, "action": "Character 1 walks.", "changes": changes,
                                 "present": [{"name": "Character 1"}]})
    return {"case_id": case, "key": case, "film_key": f"film-{case}", "source": "test",
            "shots": [{"frames": 57, "seconds": 3.5, "usable": True} for _ in range(shots)],
            "annotation": {"characters": [{"name": "Character 1", "outfit": "a grey shirt"}],
                           "shots": annotation_shots}}


MEMBERS = [member(f"conflict-{i}", outfit_change_at=2) for i in range(6)] + [member("plain-0"), member("plain-1")]
LEGACY = {"sampling_scheme": "duration_bucket_windows", "partition_max_rgb_frames": None, "max_source_shot_seconds": None, "shot_count_weights": None}
ROWS = [BlockRow(m["case_id"], (57,) * 4, (0, 1, 2, 3)) for m in MEMBERS]


def test_single_shot_table_keeps_the_sampler_draws_and_forced_share():
    samples, report = draw_sampling(MEMBERS, **LEGACY, media=ANNOTATION_MEDIA, seed=42, frame_buckets=[228], bucket_weights=[1], max_shots=1,
                                    sample_budget=400, forced_samples=10)
    assert [s["sample_index"] for s in samples] == list(range(400))
    assert all(s["span"][1] - s["span"][0] == 1 for s in samples)
    forced = [s for s in samples if s["branch"] == "forced"]
    assert len(forced) == 10 and report["forced_bucket_quotas"] == [10]
    assert all(s["case_id"].startswith("conflict-") and s["candidate_reference_positions"] for s in forced)
    # 10 个定向样本落在 6 个候选块里，每块至多两次，所以最多的块正好两次。
    assert max(Counter(s["case_id"] for s in forced).values()) == 2 == report["forced_block_exposure_max"]
    sampler = BlockWindowSampler(ROWS, frame_buckets=[228], bucket_weights=[1], max_shots=1, seed=42, shot_count_weights=None)
    for sample in samples:
        if sample["branch"] == "natural":
            row, span, _ = sampler.draw(sample["sample_index"])
            assert (sample["case_id"], tuple(sample["span"])) == (row.case_id, span)
    assert draw_sampling(MEMBERS, **LEGACY, media=ANNOTATION_MEDIA, seed=42, frame_buckets=[228], bucket_weights=[1], max_shots=1,
                         sample_budget=400, forced_samples=10)[0] == samples


def test_joint_table_forces_conflicts_inside_each_duration_bucket():
    samples, report = draw_sampling(MEMBERS, **LEGACY, media=ANNOTATION_MEDIA, seed=42, frame_buckets=[114, 228], bucket_weights=[1, 1],
                                    max_shots=None, sample_budget=400, forced_samples=8)
    assert report["forced_bucket_quotas"] == [4, 4]
    frames = lambda s: 57 * (s["span"][1] - s["span"][0])
    forced = [s for s in samples if s["branch"] == "forced"]
    assert sorted(frames(s) <= 114 for s in forced) == [False] * 4 + [True] * 4
    assert max(s["span"][1] - s["span"][0] for s in samples) == 4


def test_forced_share_must_split_exactly_over_buckets():
    with pytest.raises(ValueError, match="do not split exactly"):
        draw_sampling(MEMBERS, **LEGACY, media=ANNOTATION_MEDIA, seed=42, frame_buckets=[80, 160, 228], bucket_weights=[1, 1, 1], max_shots=None,
                      sample_budget=400, forced_samples=5)


def test_forced_span_skips_blocks_already_drawn_twice():
    options = {0: {"a": [(0, 1)], "b": [(1, 2), (3, 4)]}}
    assert {forced_span(options, Counter(a=2), 0, seed=42, index=i)[0] for i in range(50)} == {"b"}
    with pytest.raises(ValueError, match="no outfit-conflict block"):
        forced_span(options, Counter(a=2, b=2), 0, seed=42, index=0)


def test_written_slice_is_a_prefix_of_the_full_table(tmp_path):
    members = tmp_path / "members.jsonl"
    members.write_text("".join(json.dumps(m) + "\n" for m in MEMBERS))
    kwargs = dict(**LEGACY, members_path=members, media=ANNOTATION_MEDIA, seed=42, frame_buckets=[228], bucket_weights=[1], max_shots=1,
                  sample_budget=400, forced_samples=10)
    full = write_drawn_sampling(output=tmp_path / "full.jsonl", limit=None, **kwargs)
    part = write_drawn_sampling(output=tmp_path / "part.jsonl", limit=50, **kwargs)
    assert (tmp_path / "full.jsonl").read_text().splitlines()[:50] == (tmp_path / "part.jsonl").read_text().splitlines()
    assert full["shot_counts"] == {1: 400} and part["rows_written"] == 50
    assert full["reference_candidate_shots"] == {3: 400}
    assert full["exposure"]["target_shot_presentations"] == 400
    with pytest.raises(FileExistsError):
        write_drawn_sampling(output=tmp_path / "full.jsonl", limit=None, **kwargs)


def test_conflict_candidates_do_not_depend_on_string_hash_order():
    # 两人在同一镜脱外套（动作句点名两人，``worn_item_changers`` 返回集合）：候选参考的人物与镜号顺序固定，
    # 冻结表跨进程逐字节相同（11-06 两次冻表的前缀核对因这里的顺序不同而失败）。
    import os, subprocess, sys
    two = member("two-people", shots=4)
    two["annotation"]["characters"].append({"name": "Character 2", "outfit": "a blue coat"})
    for shot in two["annotation"]["shots"]:
        shot["present"].append({"name": "Character 2"})
    two["annotation"]["shots"][2]["action"] = "Character 2 and Character 1 take off their jackets."
    script = ("import json,sys;from mspc.data.renderer_training_members import candidates;"
              "m=json.loads(sys.stdin.read());print(json.dumps({str(k):v for k,v in candidates(m,(0,1,2,3),frame_counts=[s['frames'] for s in m['shots']],max_frames=228).items()}))")
    outputs = {subprocess.run([sys.executable, "-c", script], input=json.dumps(two), capture_output=True, text=True, check=True,
                              env={**os.environ, "PYTHONHASHSEED": seed}).stdout for seed in ("1", "2", "3", "4")}
    assert len(outputs) == 1
    found = json.loads(outputs.pop())
    entries = next(iter(found.values()))
    assert list(entries) == sorted(entries, key=int)
    assert all([c["character"] for c in v] == sorted(c["character"] for c in v) for v in entries.values())



def test_draw_and_exposure_use_derived_target_counts_without_changing_raw_members(tmp_path, monkeypatch):
    from copy import deepcopy
    media = {**ANNOTATION_MEDIA, 'height': 704, 'width': 1280, 'target_fps': 24}
    before = deepcopy(MEMBERS)
    monkeypatch.setattr('mspc.data.renderer_training_members.member_target_frame_counts',
                        lambda member, *, media: (85,) * len(member['shots']))
    members = tmp_path / 'members.jsonl'
    members.write_text(''.join(json.dumps(member) + '\n' for member in MEMBERS))
    result = write_drawn_sampling(**LEGACY, members_path=members, media=media, seed=42, frame_buckets=[114, 228],
        bucket_weights=[1, 1], max_shots=None, sample_budget=100, forced_samples=0,
        output=tmp_path / 'target24.jsonl', limit=None)
    samples = [json.loads(line) for line in (tmp_path / 'target24.jsonl').read_text().splitlines()]
    assert max(sample['span'][1] - sample['span'][0] for sample in samples) == 2
    assert result['exposure']['target_frame_presentations'] == 85 * result['exposure']['target_shot_presentations']
    assert result['exposure']['fps'] == 24
    assert result['exposure']['target_grid_hours'] == pytest.approx(result['exposure']['target_frame_presentations'] / 24 / 3600)
    assert MEMBERS == before


def test_usable_single_shot_over_the_explicit_frame_limit_is_not_silently_dropped(monkeypatch):
    monkeypatch.setattr('mspc.data.renderer_training_members.member_target_frame_counts',
                        lambda member, *, media: (337,) * len(member['shots']))
    with pytest.raises(ValueError, match='complete usable shot.*capacity'):
        draw_sampling(MEMBERS, **LEGACY, media=ANNOTATION_MEDIA, seed=42, frame_buckets=[228], bucket_weights=[1],
                      max_shots=None, sample_budget=10, forced_samples=0)
