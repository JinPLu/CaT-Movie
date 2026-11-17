from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mspc.data import scene_shot_manifest as ssm


# Real upload names from the CinePile-T2V raw zips (batches 001-010).
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "The_Hunger_Games__Mockingjay_-_Part_1_(10_10)_Movie_CLIP_-_Reunited_with_Peeta_(2014)_HD.mp4",
            ("movie_clip_hd", "The Hunger Games: Mockingjay - Part 1", 2014, 10, 10),
        ),
        (
            "Mission__Impossible_-_Rogue_Nation_(2015)_-_The_Bomb_Around_Benji_Scene_(8_10)_|_Movieclips.mp4",
            ("movieclips_scene", "Mission: Impossible - Rogue Nation", 2015, 8, 10),
        ),
        (
            "A_Dog's_Purpose_(2017)_-_I_Had_a_Boy_Scene_(1_10)_|_Movieclips.f614.mp4",
            ("movieclips_scene", "A Dog's Purpose", 2017, 1, 10),
        ),
        (
            "Minions__The_Rise_of_Gru_(2022)_-_Gru's_Big_Day_Scene_|_Movieclips.mp4",
            ("movieclips_scene", "Minions: The Rise of Gru", 2022, None, None),
        ),
        (
            "Shrek_Forever_After_(2010_-_Puss_Let_Himself_Go_Scene_(6_10)_|_Movieclips.mp4",
            ("movieclips_scene", "Shrek Forever After", 2010, 6, 10),
        ),
        (
            "Hang_'Em_High_(12_12)_Movie_CLIP_-_I'm_the_Law_(1968)_HD.mp4",
            ("movie_clip_hd", "Hang 'Em High", 1968, 12, 12),
        ),
        (
            "Green_Lantern_#7_Movie_CLIP_-_Unprecedented_Danger_(2011)_HD.mp4",
            ("movie_clip_hd", "Green Lantern", 2011, 7, None),
        ),
        (
            "Harry_Potter_and_the_Deathly_Hallows__Part_1_-_Cafe_Attack_(2010)_HD.mp4",
            ("other", "Harry Potter and the Deathly Hallows: Part 1", 2010, None, None),
        ),
        (
            "Clearing_Sinuses_-_The_Odd_Couple_(6_8)_Movie_CLIP_(1968)_HD.mp4",
            ("movie_clip_hd", "The Odd Couple", 1968, 6, 8),
        ),
        (
            "Get_in_My_Belly!_-_Austin_Powers__The_Spy_Who_Shagged_Me_(4_7)_Movie_CLIP_(1999)_HD.mp4",
            ("movie_clip_hd", "Austin Powers: The Spy Who Shagged Me", 1999, 4, 7),
        ),
        (
            "I_Got_Nowhere_Else_to_Go!_-_An_Officer_and_a_Gentleman_(4_6)_Movie_CLIP_(1982)_HD.mp4",
            ("movie_clip_hd", "An Officer and a Gentleman", 1982, 4, 6),
        ),
        (
            "High_School_Girls_-_Dazed_and_Confused_(9_12)_Movie_CLIP_(1993)_HD.mp4",
            ("movie_clip_hd", "Dazed and Confused", 1993, 9, 12),
        ),
        (
            "Here's_Looking_At_You,_Kid_-_Casablanca_(5_6)_Movie_CLIP_(1942)_HD.mp4",
            ("movie_clip_hd", "Casablanca", 1942, 5, 6),
        ),
        (
            "A_Catch_With_Dad_-_Field_of_Dreams_(9_9)_Movie_CLIP_(1989)_HD.mp4",
            ("movie_clip_hd", "Field of Dreams", 1989, 9, 9),
        ),
        (
            "If_History_Has_Taught_Us_Anything_-_The_Godfather__Part_2_(6_8)_Movie_CLIP_(1974)_HD.mp4",
            ("movie_clip_hd", "The Godfather: Part 2", 1974, 6, 8),
        ),
        (
            "Warriors,_Come_Out_to_Play_-_The_Warriors_(7_8)_Movie_CLIP_(1979)_HD.mp4",
            ("movie_clip_hd", "The Warriors", 1979, 7, 8),
        ),
        (
            "Funny_Corpse_Questions_Scene_-_Dungeons_&_Dragons__Honor_Among_Thieves_(2023)_|_Movieclips.mp4",
            ("movieclips_scene", "Dungeons & Dragons: Honor Among Thieves", 2023, None, None),
        ),
        (
            "A_View_To_a_Kill_Movie_CLIP_-_Buying_or_Selling_(1985)_HD.mp4",
            ("movie_clip_hd", "A View To a Kill", 1985, None, None),
        ),
        (
            "2001__A_Space_Odyssey_Movie_CLIP_-_A_Conversation_with_HAL_(1968)_HD.mp4",
            ("movie_clip_hd", "2001: A Space Odyssey", 1968, None, None),
        ),
        (
            "The_Naked_Gun__From_the_Files_of_Police_Squad!_4_10)_Movie_CLIP_-_Enemies_of_America_(1988)_HD.mp4",
            ("movie_clip_hd", "The Naked Gun: From the Files of Police Squad!", 1988, 4, 10),
        ),
    ],
)
def test_parse_cinepile_clip_name_reads_real_upload_names(name, expected):
    parsed = ssm.parse_cinepile_clip_name(name)
    assert (parsed["name_style"], parsed["film_title"], parsed["year"],
            parsed["clip_index"], parsed["clip_total"]) == expected


def test_parse_cinepile_clip_name_marks_untitled_uploads():
    assert ssm.parse_cinepile_clip_name("tmp6ogmw2z1.mp4") == {"name_style": "unnamed_tmp"}


def _write_sseg(tmp_path: Path, shots: str, scenes: str) -> tuple[Path, Path]:
    shot_path = tmp_path / "shot.txt"
    scene_path = tmp_path / "scene.txt"
    shot_path.write_text(shots, encoding="ascii")
    scene_path.write_text(scenes, encoding="ascii")
    return shot_path, scene_path


def test_parse_sseg_video_turns_inclusive_indices_half_open(tmp_path):
    # First four shots of SSeg data/shot/10.txt; scenes regrouped over them.
    shot_path, scene_path = _write_sseg(tmp_path, "0 40\n41 180\n181 249\n250 277\n", "0 1\n2 3\n")
    parsed = ssm.parse_sseg_video(shot_path, scene_path)
    assert parsed["shots"] == [[0, 41], [41, 181], [181, 250], [250, 278]]
    assert parsed["scenes"] == [[0, 2], [2, 4]]


def test_parse_sseg_video_rejects_scenes_that_do_not_cover_the_shot_list(tmp_path):
    shot_path, scene_path = _write_sseg(tmp_path, "0 40\n41 180\n181 249\n", "0 1\n")
    with pytest.raises(ValueError, match="scenes end at shot 1"):
        ssm.parse_sseg_video(shot_path, scene_path)


def _measurements(frame_count: int, *, black=(), static=(), left_static=None) -> dict:
    dark = np.full(frame_count, 0.2)
    diff = np.full(frame_count, 4.0)
    dark[list(black)] = 1.0
    diff[list(static)] = 0.0
    near_black = np.where(dark >= 1.0, 1.0, 0.1)
    left = diff.copy() if left_static is None else np.full(frame_count, 4.0)
    if left_static is not None:
        left[list(left_static)] = 0.0
    zeros = np.zeros(frame_count)
    return {
        "black_fraction": ssm.encode_channel(ssm._quantize(near_black, 250.0)),
        "dark_fraction": ssm.encode_channel(ssm._quantize(dark, 250.0)),
        "bright_fraction": ssm.encode_channel(ssm._quantize(zeros, 1000.0)),
        "abs_diff": ssm.encode_channel(ssm._quantize(diff, 10.0)),
        "left_abs_diff": ssm.encode_channel(ssm._quantize(left, 10.0)),
        "row_luma": [0] * 10 + [80] * 88 + [0] * 10,
        "col_luma": [80] * ssm.MEASURE_WIDTH,
    }


def _detection(frame_count: int, boundaries: list[int], measurements: dict, **extra) -> dict:
    return {
        "status": "ok",
        "source": ssm.SOURCE_CINEPILE,
        "video_id": "video_batch_001/Clip",
        "video_path": "/data/video_batch_001/Hang_'Em_High_(12_12)_Movie_CLIP_-_I'm_the_Law_(1968)_HD.mp4",
        "fps": 24.0,
        "fps_fraction": "24/1",
        "width": 1920,
        "height": 1080,
        "container_frame_count": frame_count,
        "decoded_frame_count": frame_count,
        "transnet_cut_boundaries": boundaries,
        "measurements": measurements,
        "text_band_samples": [[0.0, 0.0]] * 12,
        **extra,
    }


def test_cinepile_scene_row_drops_short_shots_and_marks_black_and_end_card():
    # Shots: [0,48) film, [48,60) 0.5 s, [60,120) black, [120,200) film dissolving
    # at 130 into the end card (poster quarter still, thumbnails moving).
    frames = 200
    measurements = _measurements(frames, black=range(60, 120), left_static=range(130, 200))
    rows = ssm.cinepile_scene_rows(_detection(frames, [48, 60, 120], measurements))
    assert len(rows) == 1
    row = rows[0]
    assert row["scene_frame_span"] == [0, 200]
    assert row["dropped_short_shots"] == 1
    assert [shot["source_shot_index"] for shot in row["shots"]] == [0, 2, 3]
    assert [(shot["start_frame"], shot["end_frame"]) for shot in row["shots"]] == [(0, 48), (60, 120), (120, 200)]
    assert row["shots"][0]["flags"] == []
    assert "black" in row["shots"][1]["flags"]
    assert row["shots"][2]["flags"] == [ssm.END_CARD_FLAG]
    assert row["end_card_start_frame"] == 130
    assert row["metadata"]["film_title"] == "Hang 'Em High"
    assert row["active_box"] == [0, 100, 1920, 980]
    assert row["container_preroll_frames"] == 0
    assert row["video_flags"] == []


def test_static_flag_ignores_the_difference_across_the_cut():
    frames = 96
    # Every frame static except the first frame of each shot (the cut itself).
    still = [i for i in range(frames) if i not in (0, 48)]
    measurements = _measurements(frames, static=still, left_static=())
    row = ssm.cinepile_scene_rows(_detection(frames, [48], measurements))[0]
    assert [shot["flags"] for shot in row["shots"]] == [["frozen"], ["frozen"]]


def test_a_fade_to_black_ending_is_not_an_end_card():
    frames = 120
    measurements = _measurements(frames, black=range(72, 120), static=range(72, 120))
    row = ssm.cinepile_scene_rows(_detection(frames, [72], measurements))[0]
    assert row["end_card_start_frame"] is None
    assert row["shots"][1]["flags"] == ["black", "frozen"]


def test_boundary_alignment_distinguishes_exact_shifted_and_unrelated():
    annotated = [100, 250, 400, 610, 800]
    exact = ssm.boundary_alignment(annotated, [100, 251, 400, 610, 800, 900], tolerance=2, max_offset=12)
    assert exact["recall_at_zero"] == 1.0 and exact["best_offset"] == 0
    shifted = ssm.boundary_alignment(annotated, [b + 8 for b in annotated], tolerance=2, max_offset=12)
    assert shifted["recall_at_zero"] == 0.0 and shifted["best_offset"] in (6, 7, 8, 9, 10)
    assert shifted["recall_at_best"] == 1.0
    unrelated = ssm.boundary_alignment(annotated, [50, 330, 520], tolerance=2, max_offset=12)
    assert unrelated["recall_at_best"] <= 0.2


def test_moviechat_alignment_admits_only_videos_whose_annotation_sits_on_cuts():
    sseg = {"shots": [[0, 100], [100, 250], [250, 400], [400, 600]], "scenes": [[0, 2], [2, 4]]}
    base = _detection(600, [100, 250, 400], _measurements(600))
    assert ssm.moviechat_alignment(base, sseg)["status"] == "aligned"
    shifted = dict(base, transnet_cut_boundaries=[108, 258, 408])
    assert ssm.moviechat_alignment(shifted, sseg)["status"] == "shifted"
    sparse = dict(base, transnet_cut_boundaries=[250])
    assert ssm.moviechat_alignment(sparse, sseg)["status"] == "low_agreement"
    short = dict(base, decoded_frame_count=590)
    assert ssm.moviechat_alignment(short, sseg)["status"] == "beyond_video"


def test_moviechat_scene_rows_follow_sseg_scenes():
    sseg = {"shots": [[0, 100], [100, 110], [110, 400], [400, 600]], "scenes": [[0, 2], [2, 4]]}
    detection = dict(_detection(600, [100, 110, 400], _measurements(600)), source=ssm.SOURCE_MOVIECHAT,
                     video_id="10", container_frame_count=644)
    video = {"genre": "movie", "split": "train"}
    alignment = ssm.moviechat_alignment(detection, sseg)
    rows = ssm.moviechat_scene_rows(detection, video, sseg, alignment)
    assert [row["scene_id"] for row in rows] == ["moviechat_1k:10:scene000", "moviechat_1k:10:scene001"]
    assert rows[0]["shot_count"] == 1 and rows[0]["dropped_short_shots"] == 1
    assert rows[0]["scene_frame_span"] == [0, 110]
    assert rows[1]["shots"][0]["source_shot_index"] == 2
    assert rows[1]["container_preroll_frames"] == 44


def test_measure_frames_block_mean_is_transnet_input_and_flags_black_and_static():
    import cv2

    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(3, ssm.MEASURE_HEIGHT, ssm.MEASURE_WIDTH, 3), dtype=np.uint8)
    rgb[1] = rgb[0]
    rgb[2] = 0
    channels, small, luma = ssm.measure_frames(rgb, None)
    area = cv2.resize(rgb[0], (ssm.TRANSNET_WIDTH, ssm.TRANSNET_HEIGHT), interpolation=cv2.INTER_AREA)
    assert small.shape == (3, ssm.TRANSNET_HEIGHT, ssm.TRANSNET_WIDTH, 3)
    assert np.abs(small[0].astype(int) - area.astype(int)).max() <= 1
    assert np.isinf(channels["abs_diff"][0]) and channels["abs_diff"][1] == 0.0
    assert channels["dark_fraction"][2] == 1.0
    _, _, _ = ssm.measure_frames(rgb[2:], luma[-1])


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg")
def test_detect_video_counts_every_decoded_frame_and_places_the_cut(tmp_path):
    video = tmp_path / "two_shots.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=320x180:r=12:d=2",
         "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=12:d=2",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )

    class CutAtColorChange:
        def predict(self, frames):
            reds = [float(frame[..., 0].mean() > frame[..., 2].mean()) for frame in frames]
            return [1.0 if reds[i] and i + 1 < len(reds) and not reds[i + 1] else 0.0 for i in range(len(reds))]

    row = ssm.detect_video(
        {"source": ssm.SOURCE_CINEPILE, "video_id": "b/two_shots", "video_path": str(video)},
        engine=CutAtColorChange(), ffmpeg="ffmpeg", ffprobe="ffprobe", threads=1,
    )
    assert row["decoded_frame_count"] == 48
    assert row["fps_fraction"] == "12/1"
    assert row["transnet_cut_boundaries"] == [24]
    assert len(ssm.decode_channel(row["measurements"]["abs_diff"], "abs_diff")) == 48


def test_content_diff_scores_match_opencv_hsv_and_cuts_only_the_hard_switch():
    import cv2

    rng = np.random.default_rng(1)
    sample = rng.integers(0, 256, size=(4, 18, 24, 3), dtype=np.uint8)
    got = ssm.bgr_to_opencv_hsv(sample)
    expected = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) for frame in sample])
    hue_delta = np.abs(got[..., 0].astype(int) - expected[..., 0].astype(int))
    hue_delta = np.minimum(hue_delta, 180 - hue_delta)
    assert hue_delta.max() <= 1
    assert np.abs(got[..., 1].astype(int) - expected[..., 1].astype(int)).max() <= 1
    assert np.abs(got[..., 2].astype(int) - expected[..., 2].astype(int)).max() <= 1

    frames = np.zeros((6, 18, 24, 3), dtype=np.uint8)
    frames[:3] = (0, 0, 180)
    frames[3:] = (180, 40, 0)
    manual = []
    hsv = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) for frame in frames]).astype(np.int16)
    for index in range(len(frames) - 1):
        delta = np.abs(hsv[index + 1] - hsv[index])
        manual.append(float(delta.mean(axis=(0, 1)).mean()))
    scores = ssm.content_diff_scores(frames)
    assert np.allclose(scores, manual, atol=0.35)
    assert scores[2] > 27 and scores[0] == 0 and scores[1] == 0
    assert ssm.content_diff_scores(frames[:1]).shape == (0,)
    assert ssm.content_diff_cuts(scores, threshold=27.0, min_gap_frames=2) == [3]
    quiet = np.full(8, 1.0)
    quiet[4] = 30.0
    assert ssm.content_diff_cuts(quiet, threshold=27.0, min_gap_frames=3) == [5]
    # One dissolve-width run is one cut, at its strongest frame.
    assert ssm.content_diff_cuts([1, 1, 40, 50, 45, 1, 1], threshold=27, min_gap_frames=2) == [4]
    assert ssm.content_diff_cuts([1, 40, 1, 1, 1, 50, 1], threshold=27, min_gap_frames=2) == [2, 6]
    with pytest.raises(ValueError):
        ssm.content_diff_cuts(quiet, threshold=0, min_gap_frames=1)


def test_subtitle_probe_scores_outlined_white_text_in_the_bottom_band():
    gray = np.full((ssm.TEXT_PROBE["height"], ssm.TEXT_PROBE["width"]), 120, dtype=np.uint8)
    # A line of outlined white glyphs at 88% of the height, like a burned-in subtitle.
    for x in range(200, 440, 12):
        gray[312:326, x:x + 8] = 20
        gray[314:324, x + 2:x + 6] = 250
    bottom = ssm.text_band_score(gray, ssm.TEXT_PROBE["bottom_band"])
    top = ssm.text_band_score(gray, ssm.TEXT_PROBE["top_band"])
    assert bottom > ssm.SUBTITLE_RULE["band_score_min"] and top == 0.0
    assert ssm.subtitle_hits([[bottom, top]] * 5 + [[0.0, 0.0]] * 7) == 5
    # Marks at the bottom that the top band also carries are not counted.
    assert ssm.subtitle_hits([[0.01, 0.005]] * 12) == 0


def test_moviechat_scene_rows_carry_the_subtitle_video_flag():
    sseg = {"shots": [[0, 100], [100, 250]], "scenes": [[0, 2]]}
    samples = [[0.03, 0.0]] * 6 + [[0.0, 0.0]] * 6
    detection = dict(_detection(250, [100], _measurements(250)), source=ssm.SOURCE_MOVIECHAT, video_id="OMM-6",
                     text_band_samples=samples)
    rows = ssm.moviechat_scene_rows(detection, {"genre": "movie", "split": "train"}, sseg,
                                    ssm.moviechat_alignment(detection, sseg))
    assert rows[0]["video_flags"] == [ssm.SUBTITLE_FLAG] and rows[0]["subtitle_sample_hits"] == 6


def test_scene_statistics_counts_scenes_by_shot_count_and_unflagged_shots():
    def row(scene_id, durations, flags=(), video_flags=()):
        shots = [{"duration_s": d, "flags": list(flags) if i == 0 else []} for i, d in enumerate(durations)]
        return {"scene_id": scene_id, "video_id": scene_id, "source": ssm.SOURCE_CINEPILE,
                "shot_count": len(shots), "dropped_short_shots": 1, "shots": shots,
                "video_flags": list(video_flags)}

    stats = ssm.scene_statistics([
        row("a", [1.5, 3.0]), row("b", [2.0, 2.0, 9.0], flags=["black"]), row("c", [20.0]),
        row("d", [2.0, 2.0, 2.0], video_flags=[ssm.SUBTITLE_FLAG]),
    ])
    assert stats["scenes"] == 4
    assert stats["scenes_with_at_least"]["3"] == {"scenes": 2, "share": 0.5}
    assert stats["scenes_with_at_least_unflagged"]["3"]["scenes"] == 0
    assert stats["scenes_in_videos_flagged"][ssm.SUBTITLE_FLAG] == 1
    assert stats["shots_per_scene_histogram"]["2"]["scenes"] == 1
    assert stats["shot_duration_histogram"]["16+"]["shots"] == 1
    assert stats["flagged_shots"]["black"] == 1
    assert stats["dropped_short_shots"] == 4
