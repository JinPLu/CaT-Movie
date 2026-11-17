"""每镜可渲染信息量：渲染器视角的文字怎么取、判官答案怎么数（真实开发集故事，不发请求）。"""
import json

from mspc.evaluation.shot_text_information import CATEGORIES, shot_views, story_counts
from tests.support.paths import REPO_ROOT


def _film_board():
    catalog = json.loads((REPO_ROOT / "configs/benchmark/benchmark_storyboards_v4.json").read_text())
    return next(story["storyboard"] for story in catalog["stories"] if story["story_id"].startswith("film-000127"))


def test_film_and_prose_views_are_one_text_per_shot():
    board = _film_board()
    film = shot_views("film", None, board)
    assert len(film) == len(board["shots"]) and "white underwear" in film[8]
    prose = shot_views("prose", {"status": "completed", "global": "G", "shots": ["a", "b"]}, None)
    assert prose == ["GLOBAL CAPTION (read with every shot): G\nSHOT: a", "GLOBAL CAPTION (read with every shot): G\nSHOT: b"]
    assert shot_views("ours", {"status": "planning_failed"}, None) is None


def test_counts_skip_not_applicable_and_sum_stated_per_shot():
    keys = [key for key, _ in CATEGORIES]
    one = {key: "stated" for key in keys} | {"positions": "not_applicable", "composition": "not_stated"}
    judged = {"plans": {"shots/seed-42/s": {"tier": "shots", "story_id": "s", "answers": [one, one]}}}
    block = story_counts(judged)["shots"]["s"]
    assert block["shots"][0] == 2 and block["stated"][0] == 2 * (len(keys) - 2)
    assert block["positions"] == [0, 0] and block["composition"] == [0, 2]
