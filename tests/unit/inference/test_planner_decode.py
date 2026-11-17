"""规划器生成的结束判定：停止符取自 Qwen3.5-9B 的真实 config／分词器登记；失败行是 10-10 SFT 链路验证里三条越过
``<|im_end|>`` 继续编下一轮对话、跑满 12000 token 却记成 completed 的真实答复。"""
import json
from pathlib import Path

import pytest

from mspc.inference.planner_decode import answer_row, finish_reason, stop_token_ids

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/planner_generate"
STOP = json.loads((FIXTURES / "qwen35_9b_stop_tokens.json").read_text())
RUNAWAY = json.loads((FIXTURES / "sft_runaway_rows.json").read_text())
IM_END, ENDOFTEXT = STOP["tokenizer_eos_token_id"], STOP["config_text_eos_token_id"]


class Tokenizer:
    """只有 stop_token_ids 用到的两个成员，值取自上面那份真实登记。"""

    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id

    def convert_ids_to_tokens(self, token_id):
        return STOP["tokens"][str(token_id)]


class GenerationConfig:
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id


def test_stop_ids_are_the_models_eos_plus_the_chat_turn_end():
    assert stop_token_ids(Tokenizer(IM_END), GenerationConfig(ENDOFTEXT)) == [ENDOFTEXT, IM_END]
    assert stop_token_ids(Tokenizer(IM_END), GenerationConfig([ENDOFTEXT, IM_END])) == [ENDOFTEXT, IM_END]
    assert stop_token_ids(Tokenizer(IM_END), GenerationConfig(None)) == [IM_END]
    # 分词器的 eos 不是聊天回合结束符就报错，不猜。
    with pytest.raises(ValueError, match="<|im_end|>"):
        stop_token_ids(Tokenizer(ENDOFTEXT), GenerationConfig(ENDOFTEXT))


def test_the_real_runaway_rows_were_valid_json_followed_by_an_invented_next_turn():
    for row in RUNAWAY["extra_data_rows"]:
        decoder = json.JSONDecoder()
        answer, end = decoder.raw_decode(row["raw_content"])
        assert isinstance(answer, dict) and "events" in answer
        assert row["raw_content"][end:].startswith("\nuser\n规划请求：")  # <|im_end|> 被跳过后接着编的下一轮用户消息
        assert row["new_tokens"] == 12000 and row["status"] == "completed"  # 旧脚本把它们记成了正常结束
    assert RUNAWAY["normal_row_head"][0]["new_tokens"] < 12000


def test_passing_the_stop_token_mid_answer_is_not_a_stop_but_ending_on_it_is():
    stops = [ENDOFTEXT, IM_END]
    limit = RUNAWAY["extra_data_rows"][0]["new_tokens"]
    # 真实失败行的形状：中途出现过 <|im_end|>，之后又生成到上限，结尾不是停止符。
    runaway = [7] * 4000 + [IM_END] + [8] * (limit - 4001)
    assert len(runaway) == limit and finish_reason(runaway, stops, limit) == "length"
    # 修好后同一条答复在 <|im_end|> 处停下。
    assert finish_reason(runaway[:4001], stops, limit) == "stop"
    assert finish_reason([7, 8, ENDOFTEXT], stops, limit) == "stop"
    # 恰好在上限那一个 token 上停下也算停。
    assert finish_reason([7] * (limit - 1) + [IM_END], stops, limit) == "stop"
    # 没到上限又没有停止符不该出现。
    with pytest.raises(ValueError, match="without a stop token"):
        finish_reason([7, 8, 9], stops, limit)
    with pytest.raises(ValueError):
        finish_reason([], stops, limit)


def test_answer_row_records_incomplete_when_the_limit_is_hit():
    stops = [ENDOFTEXT, IM_END]
    limit = RUNAWAY["extra_data_rows"][0]["new_tokens"]
    stopped = answer_row("w/s01/shot", "{}", [7, IM_END], stops, max_new_tokens=limit, prompt_tokens=5500, elapsed_seconds=1.5)
    assert stopped == {"case_id": "w/s01/shot", "raw_content": "{}", "status": "completed", "finish_reason": "stop",
                       "new_tokens": 2, "prompt_tokens": 5500, "elapsed_seconds": 1.5}
    capped = answer_row("w/s01/shot", "{}", [7] * 4000 + [IM_END] + [8] * (limit - 4001), stops, max_new_tokens=limit,
                        prompt_tokens=5500, elapsed_seconds=1134.0)
    assert capped["status"] == "incomplete" and capped["finish_reason"] == "length" and capped["new_tokens"] == limit


# --- 定长整数数组的项数守卫（11-05：LMFE 0.11.3 在"刚写完一个数、后面跟空白"处少数一项） ---------------------------
POINTS = json.loads((Path(__file__).resolve().parents[2] / "fixtures/planner_reading/source_grounding_point_arrays.json")
                    .read_text())


def _schema(case):
    from mspc.inference.reference_pointing import source_grounding_schema

    annotation = {"characters": [{**entry, "appearance": "x"} for entry in case["characters"]]}
    return source_grounding_schema(annotation, case["references"])


def test_fixed_number_arrays_finds_the_point_array_of_the_source_grounding_schema():
    from mspc.inference.planner_decode import fixed_number_arrays

    assert fixed_number_arrays(_schema(POINTS["failed"])) == {"point_2d": 2}
    assert fixed_number_arrays({"type": "object", "properties": {"x": {"type": "array", "items": {"type": "string"}}}}) == {}


def _walk(parser, text):
    """逐字符喂；返回被拒的位置（或 None）与每个前缀上允许的字符。"""
    allowed = []
    for index, character in enumerate(text):
        here = parser.get_allowed_characters()
        allowed.append(here)
        if character not in here:
            return index, allowed
        parser = parser.add_character(character)
    return (None if parser.can_end() else len(text)), allowed


def test_the_real_three_coordinate_answer_is_stopped_at_the_whitespace_and_may_close():
    pytest.importorskip("lmformatenforcer")
    from lmformatenforcer import JsonSchemaParser
    from mspc.inference.planner_decode import schema_character_parser

    case = POINTS["failed"]
    raw = case["raw"]
    schema = _schema(case)
    assert _walk(JsonSchemaParser(dict(schema)), raw)[0] is None  # 库本身放行了这条三个数的答复
    second = raw.index('"point_2d": [\n           100,\n           100') + len('"point_2d": [\n           100,\n           100')
    rejected, allowed = _walk(schema_character_parser(schema), raw)
    assert rejected == second and raw[second] == "\n"
    assert "]" in allowed[second] and "," not in allowed[second]
    closed = raw[:second] + "]" + raw[raw.index("]", second) + 1:]
    assert _walk(schema_character_parser(schema), closed)[0] is None
    assert json.loads(closed)["references"][4]["characters"][0]["point_2d"] == [100, 100]


@pytest.mark.parametrize("index", range(len(POINTS["valid"])))
def test_recorded_valid_answers_pass_the_guard_and_it_only_removes_characters(index):
    pytest.importorskip("lmformatenforcer")
    from lmformatenforcer import JsonSchemaParser
    from mspc.inference.planner_decode import schema_character_parser

    case = POINTS["valid"][index]
    schema = _schema(case)
    plain_rejected, plain = _walk(JsonSchemaParser(dict(schema)), case["raw"])
    guarded_rejected, guarded = _walk(schema_character_parser(schema), case["raw"])
    assert plain_rejected is guarded_rejected is None
    # 每个前缀上守卫只删字符、删的只有空白，答复里实际写的那个字符一次都没被删
    assert len(plain) == len(guarded) == len(case["raw"])
    for g, p, written in zip(guarded, plain, case["raw"]):
        assert set(g) <= set(p) and all(c.isspace() for c in set(p) - set(g)) and written in g


@pytest.mark.parametrize("first", ["0", "5", "50", "100", "101", "260", "1000"])
@pytest.mark.parametrize("second", ["0", "7", "100", "999"])
@pytest.mark.parametrize("shape", ["[{a}, {b}]", "[\n {a},\n {b}\n ]", "[\n {a}, {b}\n ]"])
def test_every_recorded_valid_shape_is_still_accepted(first, second, shape):
    """库放行的写法守卫都放行；库卡住的写法（两个可再延长的数、后面跟空白再收尾）守卫在那个空白处改成直接收尾。"""
    pytest.importorskip("lmformatenforcer")
    from lmformatenforcer import JsonSchemaParser
    from mspc.inference.planner_decode import schema_character_parser

    case = POINTS["failed"]
    schema = _schema(case)

    def answer(point):
        return ('{"references": [' + ", ".join(
            '{"reference": %d, "characters": [{"name": "%s", "point_2d": %s}], "unknown_characters": []}'
            % (number, case["characters"][0]["name"], point) for number in range(1, len(case["references"]) + 1))
            + "]}")

    text = answer(shape.format(a=first, b=second))
    plain, guarded = _walk(JsonSchemaParser(dict(schema)), text)[0], _walk(schema_character_parser(schema), text)[0]
    if plain is None:
        assert guarded is None
    else:
        assert guarded is not None and guarded <= plain and text[guarded].isspace()
        assert _walk(schema_character_parser(schema), answer(shape.format(a=first, b=second).replace("\n ]", "]")))[0] is None
