from mspc.data.reference_routing import distinct_camera_character_coverage


def _shot(names, place="Location 1", size="medium shot", angle=None):
    camera = {"shot_size": size}
    if angle:
        camera["angle"] = angle
    return {"present": [{"name": n} for n in names], "place": place, "camera": camera}


# 目标镜 4：两个人在中景；候选里角色 1 有异机位镜，角色 2 只有同机位镜；另有同地点无人镜与异地无关镜。
SHOTS = [
    _shot(["Character 1"], size="close-up shot"),          # 0 角色 1 异机位
    _shot(["Character 2"]),                                 # 1 角色 2 同机位（中景、同地点）
    _shot([], size="wide shot"),                            # 2 同地点无人场景镜
    _shot(["Character 3"], place="Location 2"),             # 3 异地无关镜
    _shot(["Character 1", "Character 2"]),                  # 4 目标
    _shot(["Character 4"], size="close-up shot"),           # 5 同地点但有别人（无关）
]


def test_second_character_gets_a_same_camera_reference_and_only_one_same_place_scene_shot():
    chosen = distinct_camera_character_coverage(SHOTS, candidates=[3, 5, 2, 1, 0], target_shots=[4], count=1, limit=10)
    assert 0 in chosen and 1 in chosen          # 角色 1 异机位、角色 2 同机位兜底
    assert 3 not in chosen                      # 异地无关镜不选
    assert chosen.count(2) + chosen.count(5) == 1 and 2 in chosen   # 只补一张同地点场景镜，取在场名字最少的
    assert chosen == sorted(chosen)


def test_limit_caps_coverage_and_scene_fill():
    chosen = distinct_camera_character_coverage(SHOTS, candidates=[3, 5, 2, 1, 0], target_shots=[4], count=1, limit=1)
    assert chosen == [0]


def test_no_scene_fill_without_a_same_place_unrelated_shot():
    shots = [_shot(["Character 1"], size="close-up shot"), _shot(["Character 3"], place="Location 2"), _shot(["Character 1"])]
    assert distinct_camera_character_coverage(shots, candidates=[1, 0], target_shots=[2], count=2, limit=10) == [0]
