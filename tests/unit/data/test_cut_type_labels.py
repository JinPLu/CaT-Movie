from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from PIL import Image

from mspc.data import cut_type_labels as ctl
from mspc.data.closeai_client import CloseAIClient, CloseAICostCounter
from tests.support.closeai_fakes import fake_chat_completion
from tools.annotate import label_cut_types as tool

URL = "data:image/jpeg;base64,AAAA"
FRAMES = [URL, URL, URL]


def _shot(size: str = "Medium shot", angle: str = "Eye-level shot") -> dict[str, str]:
    return {"place": "kitchen", "main_subjects": "a woman", "shot_size": size, "camera_angle": angle}


def _output(**overrides: object) -> str:
    payload = {
        "shot_a": _shot(),
        "shot_b": _shot("Close-up shot"),
        "same_place": True,
        "same_cast": True,
        "same_camera_setup": False,
        "returns_to_earlier_setup": None,
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_frame_positions_cover_start_middle_end_and_short_shots() -> None:
    assert ctl.shot_frame_positions(37, 90) == [37 + 2, 37 + 26, 37 + 49]
    assert ctl.shot_frame_positions(5, 6) == [5, 5, 5]
    with pytest.raises(ValueError):
        ctl.shot_frame_positions(4, 4)


@pytest.mark.parametrize(
    ("place", "cast", "camera", "returns", "expected"),
    [
        (True, True, True, None, "continuation"),
        (True, True, False, None, "same_scene_new_camera"),
        (True, False, False, None, "same_scene_cast_change"),
        (True, False, True, None, "same_scene_cast_change"),
        (False, True, False, None, "scene_change"),
        (False, False, True, False, "scene_change"),
        (True, False, False, True, "return_to_setup"),
        (False, False, False, True, "return_to_setup"),
        (True, True, True, True, "continuation"),
    ],
)
def test_derive_cut_type_truth_table(place: bool, cast: bool, camera: bool, returns: bool | None, expected: str) -> None:
    assert (
        ctl.derive_cut_type(same_place=place, same_cast=cast, same_camera_setup=camera, returns_to_earlier_setup=returns)
        == expected
    )
    assert expected in ctl.CUT_TYPES


def test_parse_accepts_plain_and_fenced_json_and_derives_cut_type() -> None:
    parsed = ctl.parse_labeler_output(_output(), has_earlier=False)
    assert parsed["cut_type"] == "same_scene_new_camera"
    assert parsed["shot_b"]["shot_size"] == "Close-up shot"
    fenced = ctl.parse_labeler_output(f"```json\n{_output()}\n```", has_earlier=False)
    assert fenced == parsed
    returned = ctl.parse_labeler_output(_output(returns_to_earlier_setup=True), has_earlier=True)
    assert returned["cut_type"] == "return_to_setup"
    lowered = ctl.parse_labeler_output(_output(shot_a=_shot("medium wide shot", "eye-level shot")), has_earlier=False)
    assert lowered["shot_a"]["shot_size"] == "Medium wide shot" and lowered["shot_a"]["camera_angle"] == "Eye-level shot"


@pytest.mark.parametrize(
    ("raw", "has_earlier", "message"),
    [
        ("not json", False, "not JSON"),
        ("[1]", False, "JSON object"),
        (_output(same_place="yes"), False, "true or false"),
        (_output(returns_to_earlier_setup=True), False, "must be null"),
        (_output(returns_to_earlier_setup=None), True, "true or false"),
        (_output(shot_a=_shot("Huge shot")), False, "shot_size"),
        (_output(shot_b=_shot(angle="Bird shot")), False, "camera_angle"),
        (_output(shot_a={**_shot(), "extra": "x"}), False, "keys must be"),
        (_output(shot_a={**_shot(), "place": " "}), False, "non-empty"),
        (json.dumps({"shot_a": _shot()}), False, "keys must be"),
    ],
)
def test_parse_rejects_any_deviation(raw: str, has_earlier: bool, message: str) -> None:
    with pytest.raises(ctl.LabelerOutputError, match=message):
        ctl.parse_labeler_output(raw, has_earlier=has_earlier)


def test_messages_carry_labelled_frames_and_the_earlier_rule() -> None:
    with_z = ctl.labeler_messages(FRAMES, FRAMES, FRAMES)
    without_z = ctl.labeler_messages(FRAMES, FRAMES, None)
    images = lambda messages: [p for p in messages[1]["content"] if p["type"] == "image_url"]  # noqa: E731
    assert len(images(with_z)) == 9 and len(images(without_z)) == 6
    assert with_z[0]["role"] == "system" and with_z[1]["role"] == "user"
    assert "An earlier shot Z is shown, so returns_to_earlier_setup must be true or false" in with_z[1]["content"][-1]["text"]
    assert "must be null" in without_z[1]["content"][-1]["text"]
    assert "Earlier shot Z" in with_z[1]["content"][0]["text"]
    with pytest.raises(ValueError, match="expected 3 frames"):
        ctl.labeler_messages(FRAMES[:2], FRAMES, None)
    with pytest.raises(ValueError, match="data:image"):
        ctl.labeler_messages(["http://x", URL, URL], FRAMES, None)


def test_confusion_matrix_scores_and_intervals() -> None:
    matrix = ctl.confusion_matrix(
        [("scene_change", "scene_change"), ("scene_change", "continuation"), ("continuation", "continuation")],
        ctl.CUT_TYPES,
    )
    scores = ctl.per_class_scores(matrix)
    assert scores["scene_change"] == {"support": 2, "predicted": 1, "recall": 0.5, "precision": 1.0}
    assert scores["continuation"]["precision"] == 0.5
    assert scores["return_to_setup"]["recall"] is None
    with pytest.raises(ValueError, match="outside vocabulary"):
        ctl.confusion_matrix([("scene_change", "cut")], ctl.CUT_TYPES)
    low, high = ctl.wilson_interval(50, 100)
    assert 0.39 < low < 0.41 and 0.59 < high < 0.61
    assert ctl.wilson_interval(0, 0) is None
    assert ctl.size_distance("Close-up shot", "Medium close-up shot") == 1


def test_binary_scores_fold_the_five_classes() -> None:
    pairs = [
        ("scene_change", "scene_change"),
        ("scene_change", "continuation"),
        ("continuation", "return_to_setup"),
        ("same_scene_new_camera", "scene_change"),
        ("same_scene_new_camera", "same_scene_new_camera"),
    ]
    scene = ctl.binary_scores(pairs, ctl.COARSE_TASKS["scene_change"])
    assert (scene["tp"], scene["fp"], scene["fn"], scene["tn"]) == (1, 1, 1, 2)
    assert scene["precision"] == 0.5 and scene["recall"] == 0.5 and scene["accuracy"] == 0.6
    repeat = ctl.binary_scores(pairs, ctl.COARSE_TASKS["composition_repeat"])
    assert (repeat["tp"], repeat["fp"], repeat["fn"], repeat["tn"]) == (1, 1, 0, 3)
    with pytest.raises(ValueError, match="outside vocabulary"):
        ctl.binary_scores([("scene_change", "cut")], ctl.COARSE_TASKS["scene_change"])


# 真实行（CI-VID 训练集 chunk_0/0f85…/18 那一对）的原生字段摘录。
JOINT = json.dumps(
    {
        "change_in_camera_angle": "The angle shifts from a direct overhead view of the tray to a medium close-up shot centered on the animated man's upper body and face.",
        "change_in_video_background": "The background becomes more detailed with distinct decorations such as toys and a unicorn statue, diverging from the simpler illustrated kitchen.",
        "change_in_camera_movement": "The camera remains static in both clips but transitions from a steady tray focus to holding the man's face and upper body in view.",
    }
)
INDIVIDUAL = [
    json.dumps({"video_content": "cupcakes", "camera_angle": "A direct overhead view of the tray."}),
    json.dumps({"video_content": "a man", "camera_angle": "A medium close-up shot of the man's upper body."}),
]


def test_native_parsing_size_words_and_flags() -> None:
    fenced = '```json\n{"camera_angle": "close-up on the cat", "video_content": "x"}\n```'
    assert ctl.parse_native_json(fenced) == {"camera_angle": "close-up on the cat", "video_content": "x"}
    assert ctl.parse_native_json("{}") == {}
    assert ctl.native_shot_size("The camera is positioned at a close-up angle.") == "Close-up shot"
    assert ctl.native_shot_size("A medium close-up shot of the man.") == "Medium close-up shot"
    assert ctl.native_shot_size("Extreme close-up of an eye") == "Extreme close-up shot"
    assert ctl.native_shot_size("A direct overhead view.") is None
    assert ctl.native_shot_size(None) is None
    assert ctl.native_change_flag("The camera angle remains the same.") == "same"
    assert ctl.native_change_flag("The angle shifts from A to B.") == "changed"
    assert ctl.native_change_flag("The camera angle remains close-up but shifts focus.") == "unknown"
    features = ctl.native_features(INDIVIDUAL, JOINT)
    assert features["native_size_a"] is None and features["native_size_b"] == "Medium close-up shot"
    assert features["camera_angle_flag"] == "changed" and features["background_flag"] == "changed"
    assert ctl.native_agreement("scene_change", features) == {"background_flag": True}
    assert ctl.native_agreement("same_scene_new_camera", features) == {"camera_angle_flag": True, "background_flag": False}
    assert ctl.native_agreement("return_to_setup", features) == {}


def _sample(sample_id: str, source: str, *, earlier: bool = False) -> dict[str, object]:
    return {"sample_id": sample_id, "source": source, "stratum": "random", "split": "test", "has_earlier": earlier, "native": None}


def _row(sample_id: str, cut: str, size: str = "Medium shot") -> dict[str, object]:
    parsed = {"cut_type": cut, "shot_a": _shot(size), "shot_b": _shot(size)}
    return {"sample_id": sample_id, "parsed": parsed}


def _ref(cut: str, size: str = "Medium shot") -> dict[str, object]:
    return {"cut_type": cut, "conf": "sure", "shot_a": {"shot_size": size, "camera_angle": "Eye-level shot"}, "shot_b": {"shot_size": size, "camera_angle": "Eye-level shot"}}


def test_calibrate_reports_per_class_and_shot_accuracy() -> None:
    samples = [_sample("s1", "ci_vid"), _sample("s2", "ci_vid"), _sample("s3", "cine250k"), _sample("s4", "cine250k")]
    reference = {"s1": _ref("scene_change"), "s2": _ref("continuation", "Close-up shot"), "s3": _ref("scene_change"), "s4": _ref("same_scene_new_camera")}
    labels = [_row("s1", "scene_change"), _row("s2", "same_scene_new_camera", "Medium close-up shot"), _row("s3", "scene_change"), {"sample_id": "s4", "parsed": None}]
    report = tool.calibrate(samples, reference, labels)
    assert report["parse_or_call_failures"] == 1
    assert report["cut_type_all"]["overall"]["correct"] == 2 and report["cut_type_all"]["overall"]["total"] == 3
    assert report["cut_type_all"]["per_class"]["scene_change"]["recall"] == 1.0
    assert report["cut_type_all"]["confusion_reference_rows_predicted_columns"]["continuation"]["same_scene_new_camera"] == 1
    assert report["cut_type_ci_vid"]["n"] == 2 and report["cut_type_cine250k"]["n"] == 1
    shot = report["shot_all"]
    assert shot["shots"] == 6 and shot["shot_size_exact"]["correct"] == 4
    assert shot["shot_size_3bin"]["correct"] == 6 and shot["shot_size_within_1"]["correct"] == 6
    assert report["native_cross_check"]["ci_vid_samples_with_native"] == 0
    assert report["coarse"]["scene_change"]["tp"] == 2 and report["coarse"]["scene_change"]["fp"] == 0


def test_reference_tsv_codes_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "ref.tsv"
    path.write_text("# comment\ns001\tNC\tMS\tEL\tCU\tLOW\tsure\tnote here\ns002\tRT\tXCU\tOTS\tXWS\tAER\tunsure\n", encoding="utf-8")
    reference = tool.load_reference(path)
    assert reference["s001"]["cut_type"] == "same_scene_new_camera" and reference["s001"]["note"] == "note here"
    assert reference["s001"]["shot_b"] == {"shot_size": "Close-up shot", "camera_angle": "Low angle shot"}
    assert reference["s002"]["conf"] == "unsure" and reference["s002"]["shot_a"]["camera_angle"] == "Over-the-shoulder shot"
    path.write_text("s001\tNC\tMS\tEL\tCU\tLOW\tmaybe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sure|unsure"):
        tool.load_reference(path)
    path.write_text("s001\tNC\tMS\n", encoding="utf-8")
    with pytest.raises(ValueError, match="7 tab-separated"):
        tool.load_reference(path)


def test_select_boosted_takes_rare_candidates_first_and_one_per_story() -> None:
    def candidate(case: str, zb: float | None, ab: float, es: float) -> dict[str, object]:
        return {"case_id": case, "sims": {"max_zb": zb, "max_ab": ab, "end_start": es}}

    pool = [
        candidate("rt", 0.9, 0.2, 0.3),
        candidate("rt_dup", 0.85, 0.3, 0.3),
        candidate("co", None, 0.9, 0.99),
        candidate("mid", None, 0.7, 0.4),
        candidate("far", None, 0.1, 0.1),
    ]
    chosen = tool.select_boosted(pool, n_return=1, n_continuation=1, total=3)
    assert [c["case_id"] for c in chosen] == ["rt", "co", "mid"]
    assert len({c["case_id"] for c in chosen}) == 3


def test_civid_key_and_native_lookup() -> None:
    path = "/x/OoVMetric/resource/ci_vid/videos/chunk_0/0f85/18/0.mp4"
    assert tool.civid_key(path) == ("chunk_0/0f85/18", 0)
    sample = {"video": {"a": path, "b": path.replace("/0.mp4", "/1.mp4")}}
    record = {"individual_captions": INDIVIDUAL, "joint_captions": [JOINT]}
    native = tool.native_for_sample(sample, {"chunk_0/0f85/18": record})
    assert native["background_flag"] == "changed" and native["text"]["camera_angle_b"] == "A medium close-up shot of the man's upper body."
    assert tool.native_for_sample(sample, {}) is None
    skipped = {"video": {"a": path, "b": path.replace("/0.mp4", "/2.mp4")}}
    assert tool.native_for_sample(skipped, {"chunk_0/0f85/18": record}) is None


def test_label_one_uses_the_client_and_records_cost_and_failures(tmp_path: Path) -> None:
    for name in ("a0", "a1", "a2", "b0", "b1", "b2"):
        (tmp_path / "frames").mkdir(exist_ok=True)
        Image.new("RGB", (32, 18), "gray").save(tmp_path / "frames" / f"{name}.jpg")
    sample = {"sample_id": "s001", "has_earlier": False, "frames": {"a": [f"frames/a{i}.jpg" for i in range(3)], "b": [f"frames/b{i}.jpg" for i in range(3)]}}
    replies = iter([_output(), "not json"])
    seen: list[dict[str, object]] = []

    def transport(**request: object) -> object:
        seen.append(request)
        return fake_chat_completion(next(replies), prompt_tokens=2000, completion_tokens=100)

    config = json.loads(Path("configs/annotation/cut_type_labeler_v1.json").read_text(encoding="utf-8"))
    counter = CloseAICostCounter(
        hard_cap_cny="36.5",
        prices={"qwen3.5-flash": {"input_cny_per_mtok": "0.1752", "output_cny_per_mtok": "1.314"}},
        precall_prompt_tokens=8000,
        precall_completion_tokens=700,
    )
    client = CloseAIClient(api_key="sk-test", base_url=config["closeai_base_url"], cost_counter=counter, timeout_seconds=5, max_retries=1, retry_base_seconds=0.1, rpm_limit=6000, transport=transport)
    price = {"input": Decimal("0.024"), "output": Decimal("0.18")}
    first = tool.label_one(client, config, price, "qwen3.5-flash", tmp_path, sample)
    assert first["error"] is None and first["parsed"]["cut_type"] == "same_scene_new_camera"
    assert Decimal(first["cost_usd"]) == Decimal("0.000048") + Decimal("0.000018")
    assert seen[0]["temperature"] == 0 and seen[0]["extra_body"] == {"enable_thinking": False}
    assert sum(1 for p in seen[0]["messages"][1]["content"] if p["type"] == "image_url") == 6
    second = tool.label_one(client, config, price, "qwen3.5-flash", tmp_path, sample)
    assert second["parsed"] is None and second["error"]["type"] == "LabelerOutputError" and second["cost_usd"] is not None


def test_spent_usd_sums_every_label_file(tmp_path: Path) -> None:
    (tmp_path / "labels.a.jsonl").write_text(json.dumps({"cost_usd": "0.5"}) + "\n" + json.dumps({"cost_usd": None}) + "\n", encoding="utf-8")
    (tmp_path / "labels.b.jsonl").write_text(json.dumps({"cost_usd": "1.25"}) + "\n", encoding="utf-8")
    assert tool.spent_usd(tmp_path) == Decimal("1.75")


def test_config_is_exact_and_paths_expand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MSPC_SHARED_ROOT", "/shared")
    config = tool.load_config(Path("configs/annotation/cut_type_labeler_v1.json"))
    assert config["pool"]["metadata_csv"].startswith("/shared/Storyboard/outputs/datasets/msm_global_caption")
    assert config["model"] in config["prices_usd_per_mtok"]
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"schema_version": "cut_type_labeler_config_v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="keys must be"):
        tool.load_config(broken)
