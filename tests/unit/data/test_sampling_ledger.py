"""取样台账：块一行、镜一行。

夹具 ``tests/fixtures/sampling_ledger/`` 的行都是主服务器 2026-10-13 的真实产物原样剪出（块清单 11 块、块标注 10 行、评测排除 4 行、
装块被拒 4 行、首批目标潜变量索引前 50 行、三个做完全链的空间层目录的深度／组装阶段记录与 ``instances.json``）；
块标注行只去掉了台账不读的 ``raw_answer``。切片里没有的东西在用到的地方说明：没有任何一个真实块含评测排除镜
（装块时这些镜已被排出），没有参考帧库清单，空间层的三个目录对不上切片里的块。
"""
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from mspc.data.shot_targets import case_id
from mspc.data import sampling_ledger as sl

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests/fixtures/sampling_ledger"
CONTRACT = json.loads((REPO / "configs/data/block_annotation_v18.json").read_text())
KEYS = ["cinedance-008884-b0000", "cinedance-018630-b0000", "cinedance-016407-b0000", "cinedance-012911-b0001",
        "film-007824-b0010", "cinedance-013737-b0000", "cinedance-009548-b0000", "cinedance-009213-b0002",
        "cinedance-017259-b0000", "film-003626-b0001", "film-019369-b0011"]
SPATIAL = ["film-006483-b0011", "cinedance-012302-b0000", "cinedance-017336-b0002"]


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_lines(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """一个数据根：夹具文件摆在配置里的相对路径上，加上两处按用途补的输入（见各行说明）。"""
    root = tmp_path_factory.mktemp("ledger-root")
    (root / "final").mkdir()
    for name in ("blocks.jsonl", "block-annotations.jsonl", "block-rejections.jsonl"):
        shutil.copy(FIXTURE / name, root / "final" / name)
    blocks = {row["key"]: row for row in lines(FIXTURE / "blocks.jsonl")}
    # 评测排除：真实的前三行与一行没有 ``source`` 字段的真实行（按规则直接排除的），另加一行——真实第一行的形状，键指向夹具里 cinedance-009213-b0002 的第 2 镜。
    # 真实块里没有被排除的镜，这一行只为验证"对上了就记为不能用"。
    exclusions = lines(FIXTURE / "eval-exclude-shots.jsonl")
    target = blocks["cinedance-009213-b0002"]["scene"]["shots"][1]
    write_lines(root / "final/eval-exclude-shots.jsonl",
                exclusions + [{**exclusions[0], **{name: target[name] for name in ("sid", "scene_id", "source_shot_index")}}])
    # 参考帧库清单：用现行的 manifest_row 为 cinedance-008884-b0000 的两镜和 film-003626-b0001 的第 2 镜造行。
    reference = [blocks[key]["scene"]["shots"][position] for key, position in
                 (("cinedance-008884-b0000", 0), ("cinedance-008884-b0000", 1), ("film-003626-b0001", 1))]
    write_legacy_reference_manifest(root / "reference.jsonl", [{"source_id": s["sid"], "scene_id": s["scene_id"],
                                                  "source_shot_index": s["source_shot_index"],
                                                  "train_frames": s["render_frames"]} for s in reference])
    shutil.copy(FIXTURE / "target-index.jsonl", root / "target-index.jsonl")
    # 空间层：三个真实目录原样放着；另把 film-006483-b0011（2 镜）复制成 cinedance-008884-b0000（也是 2 镜）的目录名，
    # 只为验证按块键拼的读法——内容与那一块无关。
    spatial = root / "spatial-blocks"
    for name in SPATIAL:
        shutil.copytree(FIXTURE / "spatial-blocks" / name, spatial / name)
    shutil.copytree(FIXTURE / "spatial-blocks/film-006483-b0011", spatial / "cinedance-008884-b0000")
    return root


def config_inputs(root):
    return sl.resolve_inputs({"blocks": "final/blocks.jsonl", "block_annotations": "final/block-annotations.jsonl",
                              "eval_exclude_shots": "final/eval-exclude-shots.jsonl",
                              "block_rejections": "final/block-rejections.jsonl", "target_index": "target-index.jsonl",
                              "reference_frames": "reference.jsonl", "spatial_root": "spatial-blocks"}, root)


@pytest.fixture(scope="module")
def ledger(tree):
    out = tree / "ledger"
    summary = sl.build(config_inputs(tree), out, contract=CONTRACT)
    return {"blocks": {row["key"]: row for row in lines(out / "blocks.jsonl")}, "shots": lines(out / "shots.jsonl"),
            "summary": summary, "out": out}


def shot(ledger, key, number):
    rows = [row for row in ledger["shots"] if row["block_key"] == key and row["shot"] == number]
    assert len(rows) == 1
    return rows[0]


# ------------------------------------------------------------------ 每块每镜都有一行

def test_every_block_and_every_shot_has_a_row_in_manifest_order(ledger):
    manifest = lines(FIXTURE / "blocks.jsonl")
    assert list(ledger["blocks"]) == KEYS == [row["key"] for row in manifest]
    expected = [(row["key"], number) for row in manifest for number in range(1, row["scene"]["shot_count"] + 1)]
    assert [(row["block_key"], row["shot"]) for row in ledger["shots"]] == expected
    for row in manifest:
        block = ledger["blocks"][row["key"]]
        assert block["shot_count"] == row["scene"]["shot_count"] and block["frames"] == row["scene"]["render_frames"]
        assert block["sid"] == row["scene"]["shots"][0]["sid"] and block["film_key"] == row["scene"]["film_key"]


def test_block_status_and_registry_validity(ledger):
    got = {key: (row["annotation_status"], row["registry_valid"]) for key, row in ledger["blocks"].items()}
    assert got == {
        "cinedance-008884-b0000": ("ok", True), "cinedance-018630-b0000": ("invalid", True),
        "cinedance-016407-b0000": ("invalid", True), "cinedance-012911-b0001": ("invalid", False),
        "film-007824-b0010": ("invalid", False), "cinedance-013737-b0000": ("no_answer", None),
        "cinedance-009548-b0000": ("not_submitted", None), "cinedance-009213-b0002": ("ok", True),
        "cinedance-017259-b0000": ("ok", True), "film-003626-b0001": ("ok", True), "film-019369-b0011": ("ok", True)}
    assert ledger["blocks"]["cinedance-008884-b0000"]["contract"] == "block_annotation_v18_20261013"
    assert ledger["blocks"]["cinedance-009548-b0000"]["contract"] is None


def test_registry_counts_and_where_each_character_appears(ledger):
    first = ledger["blocks"]["cinedance-008884-b0000"]
    assert [first[name] for name in ("people", "clear_people", "animals", "groups", "objects", "places")] == [5, 4, 0, 0, 5, 1]
    assert [(c["name"], c["visibility"], c["shots"]) for c in first["characters"]] == [
        ("Character 1", "clear", [1, 2]), ("Character 2", "clear", [2]), ("Character 3", "clear", [2]),
        ("Character 4", "clear", [2]), ("Character 5", "partial", [2])]
    # 一只动物、两个人群条目，没有"人"；第 1 镜答复修不好，不算它的在场名单。
    second = ledger["blocks"]["cinedance-018630-b0000"]
    assert [second[name] for name in ("people", "clear_people", "animals", "groups", "objects", "places")] == [0, 0, 1, 2, 3, 2]
    assert [(c["kind"], c["shots"]) for c in second["characters"]] == [("animal", [2, 4, 5]), ("group", [2, 3]), ("group", [5])]
    assert second["valid_shots"] == 4 and second["usable_shots"] == 4


# ------------------------------------------------------------------ 每镜的栏

def test_untouched_and_program_edited_shots(ledger):
    one, two = shot(ledger, "cinedance-008884-b0000", 1), shot(ledger, "cinedance-008884-b0000", 2)
    assert (one["frames"], one["aesthetic"], one["sharpness"]) == (57, 4.915910243988037, -0.10866907984018326)
    assert (two["frames"], two["aesthetic"], two["sharpness"]) == (121, 5.0774641036987305, -0.10281173139810562)
    assert [(s["present_people"], s["present_clear_people"], s["present_objects"], s["changes"]) for s in (one, two)] == [
        (1, 1, 1, 1), (5, 4, 4, 1)]
    assert (one["shot_size"], one["angle"], one["movement"], one["place"]) == ("medium close-up shot", "eye-level shot", "static", "Location 1")
    assert one["place_appearance"].startswith("a classroom with wooden doors")
    # 两镜都有开镜写成变化之后的条目（门开镜已「半开」、第 2 镜姿态），10-16 起由开镜对齐改回 before。
    assert (one["answer_origin"], one["repair_rules"]) == ("program_edited", ["开镜值取变化前"])
    assert (two["movement"], two["answer_origin"], two["repair_rules"], two["deleted_paths"]) == (
        "pan", "program_edited", ["摇镜不分方向", "开镜值取变化前"], [])
    assert one["usable"] and two["usable"] and not one["unusable_reasons"]


def test_a_reasked_shot_is_recorded_as_reasked(tmp_path):
    # ``reasks`` 现在没有写入方，真实行里都是空字典；这里只验证有键就记"重问过"（键是镜号字符串，同 ``deletions``）。
    rows = lines(FIXTURE / "block-annotations.jsonl")
    rows[0]["reasks"] = {"2": {}}
    write_lines(tmp_path / "a.jsonl", rows)
    annotation = sl.read_annotations(tmp_path / "a.jsonl")["cinedance-008884-b0000"]
    block = lines(FIXTURE / "blocks.jsonl")[0]
    rows = sl.shot_rows(block, annotation, eval_exclusions={}, target_index=None, reference_frames=None, spatial=None)
    assert [row["answer_origin"] for row in rows] == ["program_edited", "reasked"]  # 第 1 镜另有开镜对齐
    assert sl.block_row(block, annotation, rows, contract="x")["reasked_shots"] == [2]


def test_outfit_columns(ledger):
    # 第 2 镜 Character 1 的 changes 里有 outfit；第 4、5 镜写了与登记表默认穿着不同的穿着。
    flags = [(shot(ledger, "cinedance-018630-b0000", n)["outfit_change"], shot(ledger, "cinedance-018630-b0000", n)["other_outfit_worn"])
             for n in (2, 3, 4, 5)]
    assert flags == [(True, False), (False, False), (False, True), (False, True)]
    assert shot(ledger, "cinedance-018630-b0000", 4)["repair_rules"] == ["换装后补穿着"]
    assert shot(ledger, "cinedance-018630-b0000", 4)["present_people"] == 0  # Character 1 是动物


def test_deleted_item_and_added_registration_are_recorded_on_the_shot(ledger):
    fourth = shot(ledger, "film-003626-b0001", 4)
    assert (fourth["answer_origin"], fourth["repair_rules"], fourth["deleted_paths"]) == (
        "program_edited", ["持物补登记"], ["camera.angle"])
    assert fourth["angle"] is None and fourth["usable"]
    assert ledger["blocks"]["film-003626-b0001"]["deleted_items"] == 1
    assert ledger["blocks"]["film-003626-b0001"]["repairs"] == {"持物补登记": 1, "开镜值取变化前": 1}


def test_unrepaired_shot_is_not_usable_and_has_no_answer_columns(ledger):
    broken = shot(ledger, "cinedance-018630-b0000", 1)
    assert not broken["usable"] and broken["unusable_reasons"] == ["这一镜修不好", "大面积文字或画中画"]
    assert all(broken[name] is None for name in sl.SHOT_FACTS)
    assert (broken["frames"], broken["answer_origin"]) == (41, "as_returned")
    # 同一块里别的镜照常。
    assert shot(ledger, "cinedance-018630-b0000", 2)["usable"]
    # 删过一项而仍然修不好。
    deleted = shot(ledger, "cinedance-016407-b0000", 1)
    assert (deleted["usable"], deleted["answer_origin"], deleted["deleted_paths"]) == (False, "program_edited", ["place"])
    assert deleted["unusable_reasons"] == ["这一镜修不好", "不是连续拍摄的画面"]


def test_shot_filter_reasons_and_marks_pass_through(ledger):
    slides = shot(ledger, "cinedance-009213-b0002", 1)
    assert (slides["usable"], slides["unusable_reasons"], slides["marks"], slides["picture_type"]) == (
        False, ["不是连续拍摄的画面"], ["有角标或水印"], "screen_or_slides")
    marked = shot(ledger, "film-019369-b0011", 1)
    assert marked["usable"] and marked["marks"] == ["有角标或水印"]
    # 非连续画面的镜可以没有场所：place 是 null，不是报错。
    placeless = shot(ledger, "cinedance-017259-b0000", 7)
    assert (placeless["place"], placeless["place_appearance"], placeless["unusable_reasons"]) == (None, None, ["不是连续拍摄的画面"])


def test_blocks_without_a_usable_answer_still_have_rows_with_a_reason(ledger):
    for key, reason in (("cinedance-013737-b0000", "整块没有答复"), ("cinedance-009548-b0000", "整块没提交标注"),
                        ("cinedance-012911-b0001", "整块答复不合法"), ("film-007824-b0010", "整块答复不合法")):
        block = ledger["blocks"][key]
        rows = [row for row in ledger["shots"] if row["block_key"] == key]
        assert len(rows) == block["shot_count"] and block["usable_shots"] == 0
        assert all(row["unusable_reasons"] == [reason] and row["frames"] > 0 for row in rows)
        assert all(block[name] is None for name in ("people", "clear_people", "animals", "groups", "objects", "places", "characters"))
    assert ledger["blocks"]["cinedance-013737-b0000"]["no_answer_error"] == {
        "code": "invalid_parameter_error", "message": "<400> InternalError.Algo.InvalidParameter: Download multimodal file timed out"}
    assert ledger["blocks"]["film-007824-b0010"]["block_errors"] == ["shots: expected 1..1"]
    assert ledger["blocks"]["cinedance-009548-b0000"]["answer_from"] is None


def test_eval_overlap_marks_the_shot_unusable_and_unmatched_rows_are_counted(ledger):
    hit = shot(ledger, "cinedance-009213-b0002", 2)
    assert (hit["eval_excluded"], hit["usable"], hit["unusable_reasons"]) == (True, False, ["与评测重复"])
    assert sum(row["eval_excluded"] for row in ledger["shots"]) == 1
    # 其余三行真实排除行对不上任何夹具里的镜：装块时它们就没进块。
    assert ledger["summary"]["joins"]["eval_exclusions"] == {
        "rows": 5, "in_blocks": 1, "in_block_rejections": 0, "in_neither": 4,
        "by_reason": {"deterministic_outside_windows": 1, "excluded_window": 4}}


# ------------------------------------------------------------------ 三个可以还没有产物的关

def test_target_latents_reference_frames_and_spatial_columns(ledger):
    first, second = shot(ledger, "cinedance-008884-b0000", 1), shot(ledger, "cinedance-008884-b0000", 2)
    assert (first["target_latents"], second["target_latents"]) == (True, True)  # 首批索引前 50 行里的两镜
    assert shot(ledger, "cinedance-018630-b0000", 1)["target_latents"] is False
    assert [shot(ledger, "cinedance-008884-b0000", n)["reference_frames"] for n in (1, 2)] == [
        14, 30]
    assert shot(ledger, "film-003626-b0001", 2)["reference_frames"] == (shot(ledger, "film-003626-b0001", 2)["frames"] - 1) // 4
    assert shot(ledger, "film-003626-b0001", 1)["reference_frames"] == 0
    assert (first["has_depth"], first["has_masks"], second["has_depth"], second["has_masks"]) == (True, True, True, True)
    assert (shot(ledger, "film-003626-b0001", 1)["has_depth"], shot(ledger, "film-003626-b0001", 1)["has_masks"]) == (False, False)
    assert ledger["summary"]["joins"]["target_index"] == {"rows": 50, "in_blocks": 2, "not_in_blocks": 48}
    assert ledger["summary"]["joins"]["reference_frames"] == {"rows": 3, "in_blocks": 3, "not_in_blocks": 0}


def test_stages_with_no_output_yet_give_null_not_false(tree):
    inputs = {**config_inputs(tree), "target_index": None, "reference_frames": None, "spatial_root": None}
    out = tree / "ledger-no-stages"
    summary = sl.build(inputs, out, contract=CONTRACT)
    for row in lines(out / "shots.jsonl"):
        assert (row["target_latents"], row["reference_frames"], row["has_depth"], row["has_masks"]) == (None, None, None, None)
    assert summary["joins"]["target_index"] is None and summary["joins"]["reference_frames"] is None
    assert summary["total"]["target_latents"] == {"null": summary["total"]["shots"]}


def test_target_index_with_a_different_frame_count_fails(tree, tmp_path):
    rows = lines(FIXTURE / "target-index.jsonl")
    block = lines(FIXTURE / "blocks.jsonl")[0]
    mine = next(row for row in rows if row["sid"] == block["scene"]["shots"][0]["sid"] and row["source_shot_index"] == 0)
    mine["train_frames"] += 4
    write_lines(tmp_path / "index.jsonl", rows)
    with pytest.raises(ValueError, match="target index holds"):
        sl.build({**config_inputs(tree), "target_index": tmp_path / "index.jsonl"}, tmp_path / "out", contract=CONTRACT)


def test_reference_library_with_a_different_frame_count_fails(tree, tmp_path):
    block = lines(FIXTURE / "blocks.jsonl")[0]["scene"]["shots"][0]
    write_legacy_reference_manifest(tmp_path / "reference.jsonl", [{"source_id": block["sid"], "scene_id": block["scene_id"],
                                                      "source_shot_index": block["source_shot_index"],
                                                      "train_frames": block["render_frames"] + 4}])
    with pytest.raises(ValueError, match="reference library holds"):
        sl.build({**config_inputs(tree), "reference_frames": tmp_path / "reference.jsonl"}, tmp_path / "out", contract=CONTRACT)


# ------------------------------------------------------------------ 空间层读法（真实目录）

def test_spatial_reader_on_the_real_directories():
    root = FIXTURE / "spatial-blocks"
    assert sl.read_spatial(root, "film-006483-b0011", 2) == [{"depth": True, "masks": True}] * 2
    assert sl.read_spatial(root, "cinedance-017336-b0002", 1) == [{"depth": True, "masks": True}]
    assert sl.read_spatial(root, "no-such-block", 3) == [{"depth": False, "masks": False}] * 3
    assert sl.read_spatial(None, "film-006483-b0011", 2) is None


def test_spatial_reader_stage_states(tmp_path):
    def copy(name="film-006483-b0011"):
        target = tmp_path / name
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(FIXTURE / "spatial-blocks/film-006483-b0011", target)
        return target

    only_depth = copy()
    (only_depth / "stages/assemble.json").unlink()
    assert sl.read_spatial(tmp_path, "film-006483-b0011", 2) == [{"depth": True, "masks": False}] * 2
    no_depth = copy()
    (no_depth / "stages/depth.json").unlink()
    assert sl.read_spatial(tmp_path, "film-006483-b0011", 2) == [{"depth": False, "masks": True}] * 2
    failed = copy()
    record = json.loads((failed / "stages/depth.json").read_text())
    (failed / "stages/depth.json").write_text(json.dumps({"state": "failed", "stage": "depth", "reason": "oom"}))
    assert sl.read_spatial(tmp_path, "film-006483-b0011", 2) == [{"depth": False, "masks": True}] * 2
    (failed / "stages/depth.json").write_text(json.dumps({**record, "state": "running"}))
    with pytest.raises(ValueError, match="unknown state"):
        sl.read_spatial(tmp_path, "film-006483-b0011", 2)
    # 问的镜数比阶段记录里的多：对不上，报错，不当作没做。
    copy()
    with pytest.raises(KeyError):
        sl.read_spatial(tmp_path, "film-006483-b0011", 3)


# ------------------------------------------------------------------ 缺字段、类型不对、键对不上都报错

def annotation_rows():
    return lines(FIXTURE / "block-annotations.jsonl")


@pytest.mark.parametrize("field", ["shot_filter", "deletions", "reasks", "repairs", "block_errors", "shot_errors", "from", "annotation"])
def test_annotation_row_missing_a_field_fails(tmp_path, field):
    rows = annotation_rows()
    del rows[0][field]
    write_lines(tmp_path / "a.jsonl", rows)
    with pytest.raises(KeyError):
        sl.read_annotations(tmp_path / "a.jsonl")


def test_answer_that_is_a_list_gives_an_invalid_block_with_null_columns(tmp_path):
    # 真实失败行（主服务器 2026-10-13，第 1 轮）：整份答复是 JSON 数组，校验记 ``answer is not an object``。
    real = json.loads((FIXTURE / "answer-is-a-list.json").read_text())
    assert isinstance(real["annotation"]["annotation"], list)
    write_lines(tmp_path / "a.jsonl", [real["annotation"]])
    annotation = sl.read_annotations(tmp_path / "a.jsonl")[real["block"]["key"]]
    rows = sl.shot_rows(real["block"], annotation, eval_exclusions={}, target_index=None, reference_frames=None,
                        spatial=None)
    block = sl.block_row(real["block"], annotation, rows, contract="x")
    assert (block["annotation_status"], block["registry_valid"], block["people"]) == ("invalid", False, None)
    assert [(row["usable"], row["unusable_reasons"], row["activity"]) for row in rows] == [
        (False, [sl.BLOCK_INVALID], None)] * real["block"]["scene"]["shot_count"]


def test_valid_registry_with_an_answer_that_is_not_an_object_fails(tmp_path):
    rows = annotation_rows()
    ok = next(row for row in rows if row["status"] == "ok")
    ok["annotation"] = [ok["annotation"]]
    write_lines(tmp_path / "a.jsonl", rows)
    with pytest.raises(TypeError, match="not an object"):
        sl.read_annotations(tmp_path / "a.jsonl")


def test_annotation_row_with_wrong_types_or_status_fails(tmp_path):
    rows = annotation_rows()
    rows[0]["deletions"] = []
    write_lines(tmp_path / "a.jsonl", rows)
    with pytest.raises(TypeError):
        sl.read_annotations(tmp_path / "a.jsonl")
    rows = annotation_rows()
    rows[0]["status"] = "fine"
    write_lines(tmp_path / "a.jsonl", rows)
    with pytest.raises(ValueError, match="unknown status"):
        sl.read_annotations(tmp_path / "a.jsonl")
    write_lines(tmp_path / "a.jsonl", annotation_rows() + annotation_rows()[:1])
    with pytest.raises(ValueError, match="duplicate"):
        sl.read_annotations(tmp_path / "a.jsonl")


def test_no_answer_row_without_its_error_fails(tmp_path):
    rows = annotation_rows()
    row = next(r for r in rows if r["status"] == "no_answer")
    del row["error_code"]
    write_lines(tmp_path / "a.jsonl", rows)
    with pytest.raises(KeyError):
        sl.read_annotations(tmp_path / "a.jsonl")


def test_annotation_for_a_block_the_manifest_lacks_fails(tree, tmp_path):
    write_lines(tmp_path / "a.jsonl", annotation_rows() + [{**annotation_rows()[0], "key": "film-999999-b0000"}])
    with pytest.raises(ValueError, match="not in the block manifest"):
        sl.build({**config_inputs(tree), "block_annotations": tmp_path / "a.jsonl"}, tmp_path / "out", contract=CONTRACT)


def test_annotation_with_another_shot_count_fails(tree, tmp_path):
    rows = annotation_rows()
    rows[0]["shot_count"] += 1
    write_lines(tmp_path / "a.jsonl", rows)
    with pytest.raises(ValueError, match="annotation holds"):
        sl.build({**config_inputs(tree), "block_annotations": tmp_path / "a.jsonl"}, tmp_path / "out", contract=CONTRACT)


def test_shot_filter_name_outside_the_contract_fails(tree, tmp_path):
    rows = annotation_rows()
    rows[0]["shot_filter"][0]["reasons"] = ["某个合同里没有的筛因"]
    write_lines(tmp_path / "a.jsonl", rows)
    with pytest.raises(ValueError, match="not in the contract"):
        sl.build({**config_inputs(tree), "block_annotations": tmp_path / "a.jsonl"}, tmp_path / "out", contract=CONTRACT)


def test_block_row_missing_a_field_or_repeating_a_key_fails(tmp_path):
    rows = lines(FIXTURE / "blocks.jsonl")
    del rows[0]["scene"]["shots"][0]["dover_technical"]
    write_lines(tmp_path / "b.jsonl", rows[:1])
    with pytest.raises(KeyError):
        sl.shot_rows(next(sl.read_blocks(tmp_path / "b.jsonl")), None, eval_exclusions={}, target_index=None,
                     reference_frames=None, spatial=None)
    write_lines(tmp_path / "b.jsonl", lines(FIXTURE / "blocks.jsonl") + lines(FIXTURE / "blocks.jsonl")[:1])
    with pytest.raises(ValueError, match="duplicate block key"):
        list(sl.read_blocks(tmp_path / "b.jsonl"))


def test_missing_input_path_fails(tree, tmp_path):
    with pytest.raises(FileNotFoundError):
        sl.build({**config_inputs(tree), "block_rejections": tmp_path / "missing.jsonl"}, tmp_path / "out", contract=CONTRACT)
    with pytest.raises(FileNotFoundError):
        sl.build({**config_inputs(tree), "spatial_root": tmp_path / "missing"}, tmp_path / "out", contract=CONTRACT)


# ------------------------------------------------------------------ 汇总

def test_summary_counts_agree_with_the_two_tables(ledger):
    summary, blocks, shots = ledger["summary"], ledger["blocks"], ledger["shots"]
    total = summary["total"]
    assert (total["blocks"], total["shots"]) == (len(blocks), len(shots)) == (11, 38)
    assert total["annotation_status"] == {"invalid": 4, "no_answer": 1, "not_submitted": 1, "ok": 5}
    assert total["registry_valid"] == {"false": 2, "null": 2, "true": 7}
    assert total["block_funnel"] == {"blocks": 11, "submitted": 10, "answered": 9, "registry_valid": 7, "has_usable_shot": 7}
    assert total["registry_valid_blocks_with_unrepaired_shots"] == 2
    assert total["shot_funnel"] == {"shots": 38, "block_submitted": 37, "block_answered": 35, "registry_valid": 30,
                                    "shot_answer_valid": 28, "passes_shot_filter": 26, "usable": 25}
    assert total["shot_funnel"]["usable"] == sum(row["usable"] for row in shots)  # 26 与 25 差的一镜只因与评测重复而不能用
    assert total["unusable_reasons"]["与评测重复"] == 1
    assert (total["unusable_reasons"]["整块答复不合法"], total["unusable_reasons"]["整块没有答复"],
            total["unusable_reasons"]["整块没提交标注"], total["unusable_reasons"]["这一镜修不好"]) == (5, 2, 1, 2)
    assert total["repairs"] == {"持物补登记": 1, "换装后补穿着": 2, "摇镜不分方向": 1, "空默认穿着改为缺省": 1, "开镜值取变化前": 4}
    assert total["deleted_items"] == sum(row["deleted_items"] or 0 for row in blocks.values()) == 3
    assert total["answer_origin"] == {"as_returned": 21, "null": 8, "program_edited": 9}  # 开镜对齐另改了 3 镜
    assert set(summary["by_source"]) == {"cinedance", "cinepile_t2v", "moviechat_1k"}
    for name in ("blocks", "shots", "deleted_items"):
        assert sum(group[name] for group in summary["by_source"].values()) == total[name]
    for name in ("annotation_status", "unusable_reasons", "shot_funnel", "block_funnel"):
        merged = {}
        for group in summary["by_source"].values():
            for key, count in group[name].items():
                merged[key] = merged.get(key, 0) + count
        assert merged == total[name]


def test_batch_rejections_are_counted_by_reason_and_family_not_listed(ledger):
    assert ledger["summary"]["block_rejections"] == {
        "rows": 4, "by_reason": {"below_laion_aesthetic": 2, "block_below_min_seconds": 2},
        "by_family": {"cinedance": {"below_laion_aesthetic": 2, "block_below_min_seconds": 2}}}
    assert sorted(path.name for path in ledger["out"].iterdir()) == ["blocks.jsonl", "shots.jsonl", "summary.json"]


def test_published_files_are_complete_and_no_partial_is_left(ledger):
    assert not list(ledger["out"].glob("*.partial"))
    assert json.loads((ledger["out"] / "summary.json").read_text()) == ledger["summary"]


# ------------------------------------------------------------------ 配置与入口

def test_committed_config_is_valid_and_points_at_the_pipeline_root():
    config = json.loads((REPO / "configs/data/sampling_ledger_v1.json").read_text())
    assert sl.check_config(config) is config
    assert json.loads((REPO / config["annotation_contract"]).read_text())["kind"] == "block_annotation"
    assert "root" in json.loads((REPO / config["pipeline_config"]).read_text())
    assert config["inputs"]["spatial_root"] and config["output_dir"] == "final/sampling-ledger"


def test_config_checks():
    config = json.loads((REPO / "configs/data/sampling_ledger_v1.json").read_text())
    sl.check_config({**config, "inputs": {**config["inputs"], "target_index": "x.jsonl", "reference_frames": None}})
    for bad in ({**config, "extra": 1}, {**config, "schema_version": 2}, {**config, "output_dir": ""},
                {**config, "inputs": {**config["inputs"], "blocks": None}},
                {**config, "inputs": {**config["inputs"], "target_index": 3}},
                {**config, "inputs": {k: v for k, v in config["inputs"].items() if k != "spatial_root"}}):
        with pytest.raises(ValueError):
            sl.check_config(bad)


def test_entry_builds_the_ledger_under_the_given_root(tree, capsys):
    spec = importlib.util.spec_from_file_location("build_sampling_ledger", REPO / "tools/data/build_sampling_ledger.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    root = tree / "entry-root"
    (root / "final").mkdir(parents=True)
    for name in ("blocks.jsonl", "block-annotations.jsonl", "block-rejections.jsonl", "eval-exclude-shots.jsonl"):
        shutil.copy(FIXTURE / name, root / "final" / name)
    (root / "spatial-blocks").mkdir()
    assert entry.main(["--config", str(REPO / "configs/data/sampling_ledger_v1.json"), "--root", str(root)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["total"]["blocks"] == 11 and printed["joins"]["target_index"] is None
    assert printed == json.loads((root / "final/sampling-ledger/summary.json").read_text())
    assert {row["has_depth"] for row in lines(root / "final/sampling-ledger/shots.jsonl")} == {False}


def write_legacy_reference_manifest(path, rows):
    write_lines(path, [{**row, 'file': case_id({**row, 'sid': row['source_id']}) + '.safetensors',
                        'N': (row['train_frames'] - 1) // 4} for row in rows])
