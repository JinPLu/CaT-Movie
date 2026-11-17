"""单镜清单行、首批索引与对拍判据。真实行取自主服务器的块清单、首批清单 CSV 与首批缓存清单（2026-10-13）。"""
import copy
import csv
import json
from pathlib import Path

import pytest
import torch

from mspc.data.scene_training_manifest import ANNOTATION_MEDIA
from mspc.data import shot_targets as st

REAL = json.loads((Path(__file__).parent / "fixtures/shot_targets_real.json").read_text())["cases"]
CACHED = [case for case in REAL if case["first_batch"]]
TO_BUILD = [case for case in REAL if not case["first_batch"]]
named = lambda cases: pytest.mark.parametrize("case", cases, ids=lambda case: case["name"])


def case_shot(case):
    return case["block"]["scene"]["shots"][case["position"]]


def window_shots(case):
    """这个窗在缓存文件里的各镜，按窗内镜序（映射行里的三个键）。"""
    rows = sorted(case["first_batch"]["mapping_rows"], key=lambda row: row["window_shot"])
    return [{name: row[name] for name in ("sid", "scene_id", "source_shot_index")} for row in rows]


def test_stage_indices_select_original_rows_and_still_validate_the_complete_manifest():
    samples = [{"sample_index": index, "case_id": case["block"]["key"],
                "span": [case["position"], case["position"] + 1]}
               for index, case in enumerate(REAL)]
    chosen = [0, len(samples) - 1]
    expected = {}
    for index in chosen:
        row = samples[index]
        expected.setdefault(row["case_id"], set()).update(range(*row["span"]))
    assert st.sampled_target_positions(samples, sample_indices=chosen) == expected
    with pytest.raises(ValueError, match="outside the complete"):
        st.sampled_target_positions(samples, sample_indices=[len(samples)])
    damaged = copy.deepcopy(samples)
    damaged[1]["sample_index"] = 999
    with pytest.raises(ValueError, match="consecutive and ordered"):
        st.sampled_target_positions(damaged, sample_indices=chosen)


@named(REAL)
def test_case_id_preserves_the_source_shot_name(case):
    shot = case_shot(case)
    name = st.case_id(shot)
    assert name == case["case_id"]
    assert name.count("/") == 1 and name.startswith(shot["sid"] + "/")
    assert ("%2F" in name) == ("/" in shot["scene_id"])


@named(CACHED)
def test_single_shot_row_equals_the_same_shot_of_the_first_batch_window(case):
    first = case["first_batch"]
    new = st.shot_row(case["block"]["scene"], case["position"], media=ANNOTATION_MEDIA)
    assert st.manifest_differences(new, st.manifest_cells(first["manifest_row"]), first["window_shot"]) == []
    # 缓存记录里写下的实际读帧，与当时的清单是两份独立的记录。
    media = first["record"]["provenance"]["media"]
    assert new["source_frame_indices_by_shot"] == [media["sampling"]["frame_indices_by_shot"][first["window_shot"]]]
    assert new["train_crop_box"] == media["preprocessing"]["spatial"][0]["train_crop_box"]
    assert (new["sid"], new["scene_id"], new["source_shot_index"]) == st.shot_key(case_shot(case))


def test_preroll_shot_keeps_ffmpeg_frame_numbers():
    case = next(case for case in CACHED if case["name"] == "moviechat_alone_window_preroll")
    row = st.shot_row(case["block"]["scene"], case["position"], media=ANNOTATION_MEDIA)
    shot = case_shot(case)
    assert row["container_preroll_frames"] > 0
    assert [row["source_frame_indices_by_shot"][0][0], row["source_frame_indices_by_shot"][0][-1]] == [
        shot["start_frame"], shot["end_frame"] - 1]


@named(CACHED)
def test_manifest_differences_name_the_cells(case):
    first = case["first_batch"]
    new = st.shot_row(case["block"]["scene"], case["position"], media=ANNOTATION_MEDIA)
    old = st.manifest_cells(first["manifest_row"])
    old["train_crop_box"] = [value + 2 for value in old["train_crop_box"]]
    old["source_frame_indices_by_shot"][first["window_shot"]][1] += 1
    start, end = old["shot_groups"][first["window_shot"]]
    old["shot_groups"][first["window_shot"]] = [start, end + 4]
    assert st.manifest_differences(new, old, first["window_shot"]) == [
        "train_crop_box", "source_frame_indices_by_shot", "shot_groups"]


@named(TO_BUILD)
def test_to_build_row_and_csv_cells(case, tmp_path):
    shot = case_shot(case)
    row = st.shot_row(case["block"]["scene"], case["position"], media=ANNOTATION_MEDIA)
    grid = row["source_frame_indices_by_shot"][0]
    assert len(grid) == shot["render_frames"] and len(set(grid)) == len(grid)
    assert [grid[0], grid[-1]] == [shot["start_frame"], shot["end_frame"] - 1]
    assert st.write_manifest_csv(tmp_path / "scene_manifest.csv", [row]) == 1
    with (tmp_path / "scene_manifest.csv").open(newline="") as handle:
        cells, = csv.DictReader(handle)
    assert {key: cells[key] for key in case["manifest_row"]} == case["manifest_row"]  # 旧源帧和裁框保持逐格相同
    assert {key: cells[key] for key in ("target_height", "target_width", "target_fps", "vae_temporal_stride", "max_rgb_frames")} == {
        "target_height": "480", "target_width": "832", "target_fps": "16.0", "vae_temporal_stride": "4", "max_rgb_frames": "308"}
    assert st.manifest_differences(row, st.manifest_cells(cells), 0) == []


def test_extreme_lengths_are_the_shortest_and_longest_planned():
    lengths = {case["name"]: case_shot(case)["render_frames"] for case in TO_BUILD}
    assert lengths == {"to_build_shortest": 5, "to_build_longest": 225}


def test_shot_row_refuses_a_block_that_disagrees_with_the_frame_planner_or_crop():
    case = CACHED[0]
    scene = copy.deepcopy(case["block"]["scene"])
    scene["shots"][case["position"]]["render_frames"] += 4
    with pytest.raises(ValueError, match="frame planner"):
        st.shot_row(scene, case["position"], media=ANNOTATION_MEDIA)
    scene = copy.deepcopy(case["block"]["scene"])
    scene["crop_box"] = [value + 2 for value in scene["crop_box"]]
    with pytest.raises(ValueError, match="crop"):
        st.shot_row(scene, case["position"], media=ANNOTATION_MEDIA)


def test_write_manifest_csv_refuses_an_empty_manifest(tmp_path):
    with pytest.raises(ValueError, match="no rows"):
        st.write_manifest_csv(tmp_path / "scene_manifest.csv", [])
    assert not (tmp_path / "scene_manifest.csv").exists()


def test_frozen_target_union_reuses_real_source_rows_and_rejects_an_unusable_target(tmp_path):
    block = copy.deepcopy(CACHED[0]['block'])
    scene = block['scene']
    member = {'case_id': block['key'], 'key': block['key'],
              'scene': {key: value for key, value in scene.items() if key != 'shots'},
              'shots': [{**shot, 'frames': shot['render_frames'], 'usable': True} for shot in scene['shots']]}
    position = CACHED[0]['position']
    samples = [{'sample_index': index, 'case_id': member['case_id'], 'span': [position, position + 1]} for index in range(3)]
    members, sampling = tmp_path / 'members.jsonl', tmp_path / 'sampling.jsonl'
    members.write_text(json.dumps(member) + '\n')
    sampling.write_text(''.join(json.dumps(row) + '\n' for row in samples))
    rows = list(st.sampled_shot_rows(members, sampling, media=ANNOTATION_MEDIA))
    assert len(rows) == 1
    assert rows[0][1:] == (position, st.shot_row(scene, position, media=ANNOTATION_MEDIA))
    member['shots'][position]['usable'] = False
    members.write_text(json.dumps(member) + '\n')
    with pytest.raises(ValueError, match='missing or unusable'):
        list(st.sampled_shot_rows(members, sampling, media=ANNOTATION_MEDIA))


@named(CACHED)
def test_first_batch_index_row_and_reconcile(case):
    first = case["first_batch"]
    rows = st.index_rows(first["record"], window_shots(case), st.FIRST_BATCH_WINDOW)
    row = rows[first["window_shot"]]
    assert row == first["index_row"]
    assert row["file"] == first["record"]["cache_file"]
    assert row["latent_end"] - row["latent_start"] == st.latent_frames(case_shot(case)["render_frames"])
    assert st.reconcile(row, first["record"], first["window_shot"], case["block"]["scene"], case["position"], media=ANNOTATION_MEDIA) == []


def test_single_shot_window_record_has_null_groups():
    case = next(case for case in CACHED if case["name"] == "moviechat_alone_window_preroll")
    record = case["first_batch"]["record"]
    assert record["shot_groups"] is None and record["latent_shot_groups"] is None
    frames, latents = st.record_groups(record)
    assert frames == [[0, record["num_frames"]]] and latents == [[0, record["input_latents_shape"][2]]]


def test_reconcile_reports_each_disagreement():
    case = next(case for case in CACHED if case["name"] == "cinedance_later_in_window")
    first, position = case["first_batch"], case["first_batch"]["window_shot"]
    row = st.index_rows(first["record"], window_shots(case), st.FIRST_BATCH_WINDOW)[position]
    scene = copy.deepcopy(case["block"]["scene"])
    scene["shots"][case["position"]]["render_frames"] += 4
    assert st.reconcile(row, first["record"], position, scene, case["position"], media=ANNOTATION_MEDIA) == ["train_frames", "latent_frames"]
    record = copy.deepcopy(first["record"])
    media = record["provenance"]["media"]
    media["source_shot_segments"][position][0] += 1
    media["sampling"]["frame_indices_by_shot"][position][1] += 1
    media["preprocessing"]["spatial"][0]["train_crop_box"][0] += 2
    media["files"][0]["path"] += ".other"
    assert st.reconcile(row, record, position, case["block"]["scene"], case["position"], media=ANNOTATION_MEDIA) == [
        "source_segment", "frame_indices", "crop_box", "video"]


def test_index_rows_refuse_wrong_shot_count_or_empty_origin():
    case = CACHED[0]
    shots = window_shots(case)
    with pytest.raises(ValueError, match="cache holds"):
        st.index_rows(case["first_batch"]["record"], shots[:-1], st.FIRST_BATCH_WINDOW)
    for origin in ("", " ", " existing_shot", None):
        with pytest.raises(ValueError, match="origin"):
            st.index_rows(case["first_batch"]["record"], shots, origin)
    rows = st.index_rows(case["first_batch"]["record"], shots, "existing_shot")
    assert all(row["origin"] == "existing_shot" for row in rows)
    assert st.reconcile(rows[case["first_batch"]["window_shot"]], case["first_batch"]["record"],
                        case["first_batch"]["window_shot"], case["block"]["scene"], case["position"],
                        media=ANNOTATION_MEDIA) == []


@pytest.mark.parametrize("frames,expected", [(5, 2), (41, 11), (225, 57)])
def test_latent_frames(frames, expected):
    assert st.latent_frames(frames) == expected


@pytest.mark.parametrize("frames", [1, 4, 40, 5.0, True, "5"])
def test_latent_frames_refuses_other_lengths(frames):
    with pytest.raises(ValueError):
        st.latent_frames(frames)


def test_pick_spread_takes_every_stratum_from_shortest_to_longest():
    rows = [{"sid": f"s{number:03d}", "scene_id": "scene", "source_shot_index": number, "train_frames": 5 + 4 * (number % 50),
             "stratum": ("a", "b", "c")[number % 3]} for number in range(300)]
    picked = st.pick_spread(rows, 20)
    assert picked == st.pick_spread(list(reversed(rows)), 20)
    assert len({st.shot_key(row) for row in picked}) == 20
    for name, quota in (("a", 7), ("b", 7), ("c", 6)):
        lengths = [row["train_frames"] for row in picked if row["stratum"] == name]
        every = [row["train_frames"] for row in rows if row["stratum"] == name]
        assert len(lengths) == quota and lengths == sorted(lengths)
        assert (lengths[0], lengths[-1]) == (min(every), max(every))
    with pytest.raises(ValueError, match="wanted"):
        st.pick_spread(rows[:6], 9)
    with pytest.raises(ValueError, match="strata"):
        st.pick_spread(rows, 2)


def bf16(bits):
    return torch.tensor(bits, dtype=torch.int16).view(torch.bfloat16)


def test_bf16_steps_counts_representable_values_between():
    one = torch.tensor([1.0], dtype=torch.bfloat16).view(torch.int16).item()
    a = bf16([one, one, one, 0, 1, -0x8000, -one])
    b = bf16([one, one + 1, one - 2, -0x8000, -0x7FFF, 0, -one - 1])
    # 相同、差一步、差两步、+0 对 −0、最小正数对最小负数（中间隔着 0）、−0 对 +0、负数一侧差一步
    assert st.bf16_steps(a, b).tolist() == [0, 1, 2, 0, 2, 0, 1]
    assert float(bf16([one + 1])) - 1.0 == 2 ** -7


def test_bf16_steps_refuses_other_inputs():
    a = torch.zeros(2, 3, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="bfloat16"):
        st.bf16_steps(a, a.float())
    with pytest.raises(ValueError, match="bfloat16"):
        st.bf16_steps(a, torch.zeros(3, 2, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="non-finite"):
        st.bf16_steps(a, torch.full((2, 3), float("inf"), dtype=torch.bfloat16))
