import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3] / "tools/data"))
from validate_annotations import classify, normalise, normalise_by_shot, structure_errors  # noqa: E402

ROOT = Path(__file__).parents[3]
ANNOTATION = json.loads((ROOT / "tests/fixtures/window_annotation_v15/trial-0132.json").read_text())["annotation"]
LABELS = json.loads((ROOT / "configs/data/window_annotation_v15.json").read_text())["labels"]


def test_real_answer_is_valid_and_core():
    assert structure_errors(ANNOTATION, LABELS, 4, 15) == ([], {})
    verdict = classify(ANNOTATION)
    assert (verdict["hard_filter"], verdict["class"], verdict["marks"]) == ([], "核心", [])
    assert (verdict["people"], verdict["clear_people"], verdict["recurring_people"]) == (3, 3, 3)


def test_wrong_shot_count_and_unknown_names_are_rejected():
    assert structure_errors(ANNOTATION, LABELS, 5, 15) == (["shots: expected 1..5"], {})
    broken = copy.deepcopy(ANNOTATION)
    broken["shots"][0]["present"][0]["name"] = "Character 9"
    broken["video"]["picture_type"] = "film"
    assert structure_errors(broken, LABELS, 4, 15) == (["video block"], {1: ["shot 1 present 'Character 9'"]})


def test_single_shot_window_uses_the_same_rules():
    single = copy.deepcopy(ANNOTATION)
    single["shots"] = single["shots"][:1]
    verdict = classify(single)
    assert verdict["class"] == "核心" and "无跨镜同人" not in verdict["marks"]


def test_lossless_deviations_are_repaired_and_the_rest_rejected():
    # 真实返回里的两种写法偏差（05_ci_vid_floor-0216：登记表写进 video；23_cine250k_floor-1730：在场名单里写了场所）
    nested = copy.deepcopy(ANNOTATION)
    for key in ("characters", "objects", "places", "shots"):
        nested["video"][key] = nested.pop(key)
    nested["video"]["shots"][0]["present"].append({"name": nested["video"]["shots"][0]["place"]})
    assert normalise_by_shot(copy.deepcopy(nested), 15) == [("登记表或镜头写在 video 里", None), ("在场名单里写了场所", 1)]
    assert normalise(nested) == ["登记表或镜头写在 video 里", "在场名单里写了场所"]
    assert nested == ANNOTATION and normalise(nested) == []
    # film-008813-w0007 第 5 镜：{"what": "posture", "after": ["standing behind Character 1"]}
    listed = copy.deepcopy(ANNOTATION)
    name = listed["shots"][0]["present"][0]["name"]
    listed["shots"][0]["changes"] = [{"name": name, "what": "posture", "before": [], "after": ["standing behind Character 1"]}]
    assert normalise(listed) == ["变化后的值写成了单元素列表"] and listed["shots"][0]["changes"][0]["after"] == "standing behind Character 1"
    assert structure_errors(listed, LABELS, 4, 15) == ([], {})
    # 握持变化写了没登记的物件（47_cine250k_285045216_21）、状态变化的 after 写成列表（57_ci_vid_000167956）：不修，判不合法
    broken = copy.deepcopy(ANNOTATION)
    name = broken["shots"][0]["present"][0]["name"]
    broken["shots"][0]["changes"] = [{"name": name, "what": "holding", "before": [], "after": ["Object 1 handlebars"]},
                                     {"name": name, "what": "state", "before": [], "after": ["overlaying the screen"]}]
    assert structure_errors(broken, LABELS, 4, 15) == ([], {1: ["shot 1 holding change", "shot 1 change value"]})


def test_holding_written_on_the_object_moves_to_the_holder_and_people_can_hold_people():
    # cinedance-008887-w0008 第 3 镜：{"name": "Object 3", "holding": ["Character 3"]}；cinedance-009198-w0003：抱着婴儿
    answer = copy.deepcopy(ANNOTATION)
    shot = answer["shots"][0]
    people = [entry for entry in shot["present"] if entry["name"].startswith("Character")]
    thing = answer["objects"][0]["name"]
    shot["present"] = [entry for entry in shot["present"] if entry["name"] != thing] + [{"name": thing, "holding": [people[0]["name"]]}]
    people[0]["holding"] = []
    assert normalise(answer) == ["物件上写了被谁拿着"]
    assert people[0]["holding"] == [thing] and "holding" not in shot["present"][-1]
    people[0]["holding"].append(people[1]["name"])
    assert structure_errors(answer, LABELS, 4, 15) == ([], {})
    people[0]["holding"].append(people[0]["name"])
    assert structure_errors(answer, LABELS, 4, 15) == ([], {1: [f"shot 1 holding of {people[0]['name']}"]})


def test_invented_character_names_are_renumbered_everywhere_or_left_alone():
    # 首个批量任务里人群、动物常被起成 Group 1、Animal 2；不带编号的自创名（Bride）不改，判不合法
    answer = copy.deepcopy(ANNOTATION)
    old = answer["characters"][-1]["name"]
    text = json.dumps(answer).replace(f'"{old}"', '"Group 1"').replace(f"{old} ", "Group 1 ").replace(f"{old}.", "Group 1.").replace(f"{old},", "Group 1,")
    renamed = json.loads(text)
    assert normalise(renamed) == ["人物名不是 Character N"]
    assert renamed == answer  # 原来的第 3 个人物被改叫 Group 1，改回来正好还是 Character 3
    bride = json.loads(json.dumps(answer).replace(f'"{old}"', '"Bride"'))
    assert normalise(bride) == [] and structure_errors(bride, LABELS, 4, 15)[0][0] == "characters entry 'Bride'"



V18 = json.loads((ROOT / "configs/data/block_annotation_v18.json").read_text())


def pans_merged(answer):
    """夹具里的真实答复多写于第 17 版，摇镜写 pan left / pan right；第 18 版词表只有 pan。
    别的断言要的不是摇镜那一条修补，先把这些答复照第 18 版的写法并好；并的规则本身在 test_pan_direction_is_merged_when_the_contract_only_has_pan 里测。"""
    for shot in answer['shots']:
        camera = shot.get('camera')
        if isinstance(camera, dict) and camera.get('movement') in ('pan left', 'pan right'):
            camera['movement'] = 'pan'
    return answer


# 夹具里的期望记在「后出现而没登记的人整块重发」之前：这一块现在另有一条整块层的错。
LATE = {'film-007241-b0000': ["shot 11 present 'Character 3' not registered"]}


def test_real_v17_failure_rows_have_two_layers():
    for path in sorted((Path(__file__).parent / 'fixtures/blocks_v17').glob('*.json')):
        row = json.loads(path.read_text())
        normalise(pans_merged(row['answer']), 17)
        block, shots = structure_errors(row['answer'], V18['labels'], row['shots'], 17)
        assert block == row['expected_block_errors'] + LATE.get(row['key'], []), row['key']
        legacy_errors = {k: [e for e in v if 'holding targets not present' not in e
                            and 'unregistered entity reference' not in e] for k, v in shots.items()}
        assert {k: v for k, v in legacy_errors.items() if v} == {int(k): v for k, v in row['expected_shot_errors'].items()}, row['key']
        if row['key'] == 'film-006177-b0000':
            assert len(shots) == 10 and all('holding targets not present' in v[0] for v in shots.values())


def v17_case(key='film-018940-b0001'):
    row = json.loads((Path(__file__).parent / f'fixtures/blocks_v17/{key}.json').read_text())
    normalise(pans_merged(row['answer']), 17)
    return row


# 开镜对齐在这些真实行上也会生效；它的语义由 tests/unit/data/test_opening_alignment.py 管，这里分开记账。
ALIGNMENT = {'开镜值取变化前', '删掉没有变化的变化', '开镜已是变化后且没写变化前'}


def other_repairs(record):
    return [fix for fix in record['repairs'] if fix['rule'] not in ALIGNMENT]


def aligned_shots(record):
    return {fix['shot'] for fix in record['repairs'] if fix['rule'] in ALIGNMENT}


def v18_case(name, keep_pan_direction=False):
    """fixtures/blocks_v18 里的一条真实答复，按现行合同判：返回 (夹具, 台账记录)。"""
    from validate_annotations import block_record
    row = json.loads((Path(__file__).parent / f'fixtures/blocks_v18/{name}.json').read_text())
    if not keep_pan_direction:
        pans_merged(row['answer'])
    return row, block_record(json.dumps(row['answer']), V18, row['shots'])


def test_real_v17_repair_deletes_only_wrong_items():
    from validate_annotations import block_record
    rows = [v17_case('film-001889-b0003')] + [v18_case(name)[0] for name in (
        'end-card-film-004095-b0008', 'place-in-present-film-001889-b0003', 'body-part-film-006561-b0006', 'body-part-change-film-004600-b0002')]
    for row in rows:
        normalise(row['answer'], 17)
        record = block_record(json.dumps(row['answer']), V18, row['shots'])
        assert record['status'] == ('invalid' if row['key'] == 'film-006561-b0006' else 'ok'), row['key']
        assert other_repairs(record) == ([{'rule': '持有实体补在场名单', 'shot': 3, 'to': 'Object 1'}] if row['key'] == 'film-004600-b0002' else [])
        assert record['deletions'] and set(record['deletions']) == set(record['initial_shot_errors']), row['key']
        for kind in ('characters', 'objects', 'places', 'story'):
            assert record['annotation'][kind] == row['answer'][kind]
        for shot in row['answer']['shots']:
            if shot['shot'] not in set(record['deletions']) | aligned_shots(record):
                assert record['annotation']['shots'][shot['shot']-1] == shot


def test_pan_direction_is_merged_when_the_contract_only_has_pan():
    from validate_annotations import block_record
    # 第 17 版正式试标 film-000726-b0001：4 镜里前 3 镜写 pan right。词表里有 pan left / pan right 的合同原样保留；第 18 版词表只有 pan，并过去。
    row, record = v18_case('pan-film-000726-b0001', keep_pan_direction=True)
    directions = [name for item in V18['labels']['movement'] for name in (('pan left', 'pan right') if item == 'pan' else (item,))]
    kept = block_record(json.dumps(row['answer']), {**V18, 'labels': {**V18['labels'], 'movement': directions}}, row['shots'])
    assert kept['status'] == 'ok' and kept['repairs'] == [] and kept['annotation'] == row['answer']
    assert record['status'] == 'ok' and [fix['value'] for fix in record['repairs']] == ['pan right'] * 3
    row['answer']['shots'][1]['camera']['movement'] = 'pan left'  # 这批答复里唯一的 pan left 在登记表不合法的块里，在真实行上改一个字段
    record = block_record(json.dumps(row['answer']), V18, row['shots'])
    assert record['status'] == 'ok' and record['deletions'] == {} and record['initial_shot_errors'] == {}
    assert record['repairs'] == [{'rule': '摇镜不分方向', 'shot': number, 'path': 'camera.movement', 'value': value, 'to': 'pan'}
                                 for number, value in ((1, 'pan right'), (2, 'pan left'), (3, 'pan right'))]
    assert [shot['camera']['movement'] for shot in record['annotation']['shots']] == ['pan', 'pan', 'pan', 'static']


def test_shot_that_is_not_footage_may_omit_place_and_camera():
    from validate_annotations import block_record
    # 第 18 版快速检查 film-004095-b0008 第 7 镜（片尾卡，screen_or_slides）：place 与镜头三栏都写成字符串 none，四遍第 17 版答复也一样
    row, record = v18_case('end-card-film-004095-b0008')
    assert record['initial_shot_errors'] == {7: ['shot 7 place/camera/activity/action', 'shot 7 camera.shot_size', 'shot 7 camera.angle', 'shot 7 camera.movement']}
    assert record['status'] == 'ok' and other_repairs(record) == [] and aligned_shots(record) == {2, 3}
    assert record['deletions'] == {7: [{'path': 'place', 'value': 'none'}] + [{'path': f'camera.{field}', 'value': 'none'} for field in ('shot_size', 'angle', 'movement')]}
    assert 'place' not in record['annotation']['shots'][6] and record['annotation']['shots'][6]['camera'] == {}
    # 整栏写成 None 或空串的也删
    blank = copy.deepcopy(row['answer'])
    blank['shots'][6].update(place='', camera=None)
    record = block_record(json.dumps(blank), V18, row['shots'])
    assert record['status'] == 'ok' and record['deletions'] == {7: [{'path': 'place', 'value': ''}, {'path': 'camera', 'value': None}]}
    # 其它镜照旧要登记过的场所：第 6 镜是实拍，写成 none 仍是这一镜的错，不删
    row['answer']['shots'][5]['place'] = 'none'
    record = block_record(json.dumps(row['answer']), V18, row['shots'])
    assert record['status'] == 'invalid' and record['block_errors'] == [] and record['shot_errors'] == {6: ['shot 6 place/camera/activity/action']}
    assert 6 not in record['deletions'] and record['annotation']['shots'][5]['place'] == 'none'


def test_unregistered_held_thing_is_registered_once_per_block():
    from validate_annotations import block_record
    # 快速检查 base 的 film-018940-b0001：第 7、8 镜 Character 3 拿着没登记的 binoculars
    row = v17_case('film-018940-b0001')
    record = block_record(json.dumps(row['answer']), V18, row['shots'])
    assert record['status'] == 'ok' and record['deletions'] == {} and record['initial_shot_errors'] == {}
    assert record['annotation']['objects'] == row['answer']['objects'] + [{'name': 'Object 7', 'appearance': 'binoculars'}]
    assert record['repairs'] == [{'rule': '持有实体补在场名单', 'shot': 3, 'to': 'Object 1'},
                                 {'rule': '持物补登记', 'shot': 7, 'path': 'present[1].holding[0]', 'value': 'binoculars', 'to': 'Object 7'},
                                 {'rule': '持物补登记', 'shot': 8, 'path': 'present[0].holding[0]', 'value': 'binoculars', 'to': 'Object 7'}]
    for number in (7, 8):
        present = record['annotation']['shots'][number-1]['present']
        assert present[-1] == {'name': 'Object 7'} and [e['holding'] for e in present if e['name'] == 'Character 3'] == [['Object 7']]
    row['answer']['shots'][7]['present'][0]['holding'] = [' Binoculars ']  # 大小写与首尾空格不同也是同一个
    record = block_record(json.dumps(row['answer']), V18, row['shots'])
    assert [fix['to'] for fix in record['repairs']] == ['Object 1', 'Object 7', 'Object 7'] and len(record['annotation']['objects']) == 7
    # 握持变化里的也补：快速检查 clarified-1 的同一块，第 8 镜 after 写了 goggles
    row, record = v18_case('held-change-film-018940-b0001')
    assert record['status'] == 'ok' and record['deletions'] == {}
    assert record['repairs'] == [{'rule': '持有实体补在场名单', 'shot': 1, 'to': 'Object 1'},
                                 {'rule': '持物补登记', 'shot': 8, 'path': 'changes[0].after[0]', 'value': 'goggles', 'to': 'Object 6'}]
    assert record['annotation']['objects'][-1] == {'name': 'Object 6', 'appearance': 'goggles'}
    shot = record['annotation']['shots'][7]
    assert shot['changes'][0]['after'] == ['Object 6'] and shot['present'][-1] == {'name': 'Object 6'}


def test_body_parts_are_dropped_and_a_held_person_is_left_alone():
    # 正式试标 film-006561-b0006 第 9 镜：两人各「拿着」对方的手
    row, record = v18_case('body-part-film-006561-b0006')
    assert record['status'] == 'invalid' and record['repairs'] == [] and record['annotation']['objects'] == row['answer']['objects']
    assert record['wording_errors'][0]['path'] == 'shots[3].changes[0].after'
    assert record['deletions'] == {9: [{'path': 'present[0].holding[0]', 'value': "the woman's hand"}, {'path': 'present[1].holding[0]', 'value': "the man's hand"}]}
    # 正式试标 film-004600-b0002 第 3 镜：先删身体部位，再删清理后 before=after 的空变化。
    row, record = v18_case('body-part-change-film-004600-b0002')
    assert record['status'] == 'ok' and other_repairs(record) == [{'rule': '人物名不是 Character N'}, {'rule': '持有实体补在场名单', 'shot': 3, 'to': 'Object 1'}] and len(record['annotation']['objects']) == len(row['answer']['objects'])
    assert record['deletions'] == {3: [{'path': 'changes[0].after[0]', 'value': "Character 3's hand"}, {'path': 'changes[1].after[0]', 'value': "Character 2's hand"}]}
    assert record['annotation']['shots'][2]['changes'] == []
    assert [(fix['name'], fix['rule']) for fix in record['repairs'] if fix['rule'] in ALIGNMENT] == [
        ('Character 2', '删掉没有变化的变化'), ('Character 3', '删掉没有变化的变化')]
    # 快速检查 clarified-2 的 film-001889-b0003 第 6、7 镜：持物项是登记过的另一个人物（抓着对方），照旧合法，不动
    row, record = v18_case('held-person-film-001889-b0003')
    assert row['answer']['shots'][5]['present'][0]['holding'] == ['Character 2']
    assert record['status'] == 'ok' and other_repairs(record) == [] and record['deletions'] == {}
    # 这块第 4、6、7 镜还有开镜写成变化之后的条目（真实返回）；第 8 镜开镜写「standing on the pyramid」、变化是
    # hanging → 「standing on pyramid」，只差一个 the，姿态按实词比之后（10-30）也认出开镜已是终态、改回 hanging。
    assert aligned_shots(record) == {4, 6, 7, 8}
    eighth = record['annotation']['shots'][7]
    assert [e.get('posture') for e in eighth['present'][:2]] == ['hanging', 'hanging']
    assert [s for s in record['annotation']['shots'] if s['shot'] not in {4, 6, 7, 8}] == [
        s for s in row['answer']['shots'] if s['shot'] not in {4, 6, 7, 8}]


def test_late_unregistered_name_voids_the_block_and_a_place_in_present_is_dropped():
    from validate_annotations import block_record
    # 快速检查 base 的 film-007241-b0000：最后一镜出来一个登记表里没有的 Character 3；整块层的错，这一项不删
    row = v17_case('film-007241-b0000')
    record = block_record(json.dumps(row['answer']), V18, row['shots'])
    assert record['status'] == 'invalid' and record['block_errors'] == LATE['film-007241-b0000']
    assert record['deletions'] == {} and record['annotation']['shots'][10]['present'] == row['answer']['shots'][10]['present']
    # 第 18 版快速检查 film-001889-b0003 第 8 镜：在场名单里写了 Location 1；删掉、记进 deletions
    row, record = v18_case('place-in-present-film-001889-b0003')
    assert record['status'] == 'ok' and other_repairs(record) == [] and aligned_shots(record) == {6}
    assert record['deletions'] == {8: [{'path': 'present[1]', 'value': {'name': 'Location 1', 'posture': 'a purple pyramid structure'}}]}
    assert record['annotation']['shots'][7]['present'] == [{'name': 'Object 3'}]


def test_outfit_is_carried_forward_after_a_change():
    from validate_annotations import block_record
    fills = lambda record: [(fix['shot'], fix['path']) for fix in record['repairs'] if fix['rule'] == '换装后补穿着']
    # 第 18 版快速检查 film-019237-b0008：第 7 镜脱掉红外套，第 8 镜他在场却没写穿着
    row, record = v18_case('outfit-film-019237-b0008')
    worn = 'a dark green jacket, black hat, and a backpack'
    assert row['answer']['shots'][6]['changes'][0]['after'] == worn and 'outfit' not in row['answer']['shots'][7]['present'][0]
    assert record['status'] == 'ok' and record['deletions'] == {} and record['initial_shot_errors'] == {}
    assert other_repairs(record) == [{'rule': '换装后补穿着', 'shot': 8, 'path': 'present[0].outfit', 'to': worn}]
    assert record['annotation']['shots'][7]['present'][0]['outfit'] == worn
    # 第 18 版快速检查 film-007241-b0000：第 5 镜换装，第 6 镜她不在场，第 7–10 镜在场都没写
    row, record = v18_case('outfit-film-007241-b0000')
    assert record['status'] == 'ok' and fills(record) == [(number, 'present[0].outfit') for number in (7, 8, 9, 10)]
    # 在真实行上加一条：第 8 镜里换回登记表的穿着（大小写与空格不同）。第 8 镜开头还是换过的，照补；之后不再补
    back = row['answer']['characters'][2]['outfit']
    row['answer']['shots'][7]['changes'].append({'name': 'Character 3', 'what': 'outfit', 'before': 'a coat', 'after': f' {back.upper()} '})
    record = block_record(json.dumps(row['answer']), V18, row['shots'])
    assert record['status'] == 'ok' and fills(record) == [(7, 'present[0].outfit'), (8, 'present[0].outfit')]
    assert 'outfit' not in record['annotation']['shots'][8]['present'][0]
    # 快速检查 clarified-2 的 film-004095-b0008：第 1 镜换装、第 2 镜漏写、第 3 镜在场条目自己写回了登记表的穿着
    row, record = v18_case('outfit-restated-film-004095-b0008')
    assert fills(record) == [(2, 'present[1].outfit')]
    assert row['answer']['shots'][2]['present'][0]['outfit'] == row['answer']['characters'][0]['outfit']
    del row['answer']['shots'][5]['present'][0]['outfit']  # 写回之后再漏写的，是登记表的穿着，不补
    record = block_record(json.dumps(row['answer']), V18, row['shots'])
    assert fills(record) == [(2, 'present[1].outfit')] and 'outfit' not in record['annotation']['shots'][5]['present'][0]


def test_block_record_tells_untouched_mended_and_trimmed_shots_apart():
    # 同一条答复里补过、对齐过、删过的镜：从 repairs 的 shot 与 deletions 的键读得出来，其余各镜与原答复相同。
    # 第 2 镜补穿着并对齐开镜姿态，第 3 镜对齐开镜姿态（真实返回），第 7 镜删片尾卡的 none。
    row, record = v18_case('outfit-restated-film-004095-b0008')
    mended, trimmed = {fix['shot'] for fix in record['repairs'] if 'shot' in fix}, set(record['deletions'])
    assert (mended, trimmed) == ({2, 3}, {7})
    for shot in row['answer']['shots']:
        assert (record['annotation']['shots'][shot['shot']-1] == shot) == (shot['shot'] not in mended | trimmed)


def test_outfit_required_after_change_until_explicit_return():
    # 在真实失败行上改一个字段，验证跨镜约束；没有补造另一条语料。
    row = v17_case('film-001889-b0003')
    answer = row['answer']
    name = answer['characters'][0]['name']
    default = answer['characters'][0]['outfit']
    shots = answer['shots']
    for shot in shots:
        shot['present'] = [{'name': name}]
        shot['changes'] = []
        shot['camera'] = {}
    shots[0]['changes'] = [{'name': name, 'what': 'outfit', 'before': default, 'after': 'a grey coat'}]
    shots[1]['present'][0]['outfit'] = 'a grey coat'
    shots[3]['present'][0]['outfit'] = 'a grey coat'
    shots[3]['changes'] = [{'name': name, 'what': 'outfit', 'before': 'a grey coat', 'after': default}]
    block, local = structure_errors(answer, V18['labels'], row['shots'], 17)
    assert not block and local == {3: [f'shot 3 missing outfit after change of {name}']}
    # 换回默认衣服的这一镜开始仍是灰衣，缺 outfit 也要报错。
    del shots[3]['present'][0]['outfit']
    assert set(structure_errors(answer, V18['labels'], row['shots'], 17)[1]) == {3, 4}


def test_shot_filter_uses_contract_and_null_outfit_is_removed():
    from validate_annotations import classify_shots
    row = v17_case('film-004095-b0008')
    answer = row['answer']
    filtered = classify_shots(answer, V18)
    assert filtered[-1]['status'] == '被筛'
    assert '不是连续拍摄的画面' in filtered[-1]['reasons']
    answer['shots'][0]['on_screen_text'] = 'corner_logo_or_watermark'
    assert classify_shots(answer, V18)[0]['marks'] == ['有角标或水印']
    answer['characters'][0]['outfit'] = None
    answer['shots'][0]['present'][0]['outfit'] = []
    assert normalise(answer, 17)
    assert 'outfit' not in answer['characters'][0]
    assert 'outfit' not in answer['shots'][0]['present'][0]


def test_registry_failure_does_not_hide_per_shot_filter_labels():
    from validate_annotations import block_record
    row = v17_case('film-001672-b0001')
    record = block_record(json.dumps(row['answer']), V18, row['shots'])
    assert record['status'] == 'invalid' and record['block_errors'] == ['shots: expected 1..10']
    assert len(record['shot_filter']) == 8
    assert record['deletions'] == {}


def test_block_ledger_and_summary_count_repairs_and_deletions(tmp_path):
    from validate_annotations import validate_blocks
    rows = [json.loads((Path(__file__).parent / f'fixtures/blocks_v18/{name}.json').read_text())
            for name in ('outfit-film-019237-b0008', 'end-card-film-004095-b0008')]
    (tmp_path/'final').mkdir()
    (tmp_path/'block-annotations').mkdir()
    (tmp_path/'final/blocks.jsonl').write_text(''.join(json.dumps({'key': r['key'], 'source': 'film', 'scene': {'shot_count': r['shots']}})+'\n' for r in rows))
    (tmp_path/'block-annotations/batch-00.output.jsonl').write_text(''.join(json.dumps({'custom_id': r['key'], 'response': {'body': {
        'choices': [{'message': {'content': json.dumps(r['answer'])}}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 10}}}})+'\n' for r in rows))
    summary = validate_blocks(tmp_path, V18, .1, .4)
    assert (summary['ok'], summary['registry_valid'], summary['initial_all_valid']) == (2, 2, 1)
    assert summary['repairs'] == {'换装后补穿着': 1, '删掉没有变化的变化': 1, '开镜值取变化前': 2} and summary['deleted_items'] == 4
    ledger = {r['key']: r for r in map(json.loads, (tmp_path/'final/block-annotations.jsonl').open())}
    assert [fix['shot'] for fix in ledger['film-019237-b0008']['repairs']] == [8, 8] and ledger['film-019237-b0008']['deletions'] == {}
    assert other_repairs(ledger['film-004095-b0008']) == [] and list(ledger['film-004095-b0008']['deletions']) == ['7']


def test_block_summary_distinguishes_pending_from_never_submitted(tmp_path):
    from validate_annotations import validate_blocks
    (tmp_path/'final').mkdir()
    (tmp_path/'block-annotations').mkdir()
    (tmp_path/'final/blocks.jsonl').write_text(''.join(json.dumps({'key': k})+'\n' for k in ['a','b','c']))
    (tmp_path/'final/block-annotation-batches.json').write_text(json.dumps({
        'selected_keys': ['a','b'], 'batches': [
            {'keys': ['a'], 'batch_id': 'batch-pending', 'fetched': False},
            {'keys': ['b'], 'fetched': False}]}))
    result = validate_blocks(tmp_path, V18, .1, .4)
    assert (result['in_ledger'], result['submitted'], result['inflight']) == (0,1,1)
    assert result['selected_not_submitted'] == 1 and result['never_submitted'] == 2
    assert result['submitted_without_response'] == 1
