"""规划器生成的结束判定：停止符怎么取、一条答复为什么结束、答复行怎么记。

Qwen3.5 的 ``config.json`` 只登记 ``<|endoftext|>`` 作 eos，聊天模板的回合结束符 ``<|im_end|>`` 是分词器的 eos
（``tokenizer_config.json``）。只传前者给 ``generate`` 时，模型写完答复后吐出 ``<|im_end|>`` 也不会停，会接着编造
下一轮对话直到撞 ``max_new_tokens``（10-10 SFT 链路验证里三条答复就是这样，答复文本后接 ``\\nuser\\n规划请求：…``）。
停止符的 id 从模型与分词器取，不写死数字。
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping, Sequence

CHAT_END = "<|im_end|>"


@lru_cache(maxsize=1)
def _schema_tokenizer_data(tokenizer: Any):
    """每个reader tokenizer只建一次前缀树，避免每请求重复解码完整词表。"""
    from lmformatenforcer import TokenEnforcerTokenizerData

    token_zero = tokenizer.encode("0", add_special_tokens=False)[-1]
    special = set(tokenizer.all_special_ids)
    regular = []
    for token in range(len(tokenizer)):
        if token in special:
            continue
        after_zero = tokenizer.decode([token_zero, token], clean_up_tokenization_spaces=False)[1:]
        direct = tokenizer.decode([token], clean_up_tokenization_spaces=False)
        regular.append((token, after_zero, len(after_zero) > len(direct)))

    def decode(ids):
        return tokenizer.decode(ids, clean_up_tokenization_spaces=False).rstrip("�")

    return TokenEnforcerTokenizerData(regular, decode, tokenizer.eos_token_id, False, len(tokenizer))


def fixed_number_arrays(schema: Mapping[str, Any]) -> dict[str, int]:
    """schema 里定长整数数组的属性名 → 长度（``minItems == maxItems``，``items`` 是整数 ``enum``），可在 ``anyOf`` 里。
    同名属性长度不一致就报错。源图读取 schema 里就是 ``{"point_2d": 2}``。"""
    found: dict[str, int] = {}

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for name, value in (node.get("properties") or {}).items():
                for option in [value, *value.get("anyOf", [])]:
                    items = option.get("items")
                    if (option.get("type") == "array" and option.get("minItems") is not None
                            and option.get("minItems") == option.get("maxItems") and isinstance(items, Mapping)
                            and items.get("enum") and all(type(v) is int for v in items["enum"])):
                        if found.setdefault(name, option["minItems"]) != option["minItems"]:
                            raise ValueError(f"fixed-length array {name!r} has two lengths in one schema")
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)
    return found


@lru_cache(maxsize=1)
def _fixed_array_guard_class():
    """LMFE 0.11.3 的 ``ListParsingState`` 在部分"某项刚写完、后面跟空白"的位置把已写项数少数一个（11-05 实测：
    ``[100, 100\\n ]`` 的 ``]`` 被拒、只许 ``,``，于是被逼写第三个数；``[5 , 260]`` 在 ``5 `` 之后只许空白）。
    这类位置往后不存在合法答案。本守卫只在定长整数数组里、刚写完一个数字时，删掉"走一步后库不许真实项数所需
    控制符"的空白字符；不加任何字符，其余位置原样转给库。所以库原本放行的合法答案每个字符都还被放行，贪心解码
    下逐字不变。用的是库公开的 ``CharacterLevelParser`` 扩展接口，不改库。"""
    from lmformatenforcer import CharacterLevelParser
    from lmformatenforcer.consts import WHITESPACE_CHARACTERS

    import re

    key_before_array = re.compile(r'"(\w+)":\[$')

    class FixedArrayCountGuard(CharacterLevelParser):
        def __init__(self, inner, lengths: Mapping[str, int], *, tail: str = "", array=None, last: str = ""):
            self.inner, self.lengths = inner, lengths
            self.tail, self.array, self.last = tail, array, last  # array: (定长, 已见逗号数) 或 None

        @property
        def config(self):
            return self.inner.config

        @config.setter
        def config(self, new_config):
            self.inner.config = new_config

        def add_character(self, new_character: str):
            tail, array = self.tail, self.array
            if new_character not in WHITESPACE_CHARACTERS:
                tail = (tail + new_character)[-64:]
            if array is None:
                if new_character == "[":
                    match = key_before_array.search(tail)
                    if match and match.group(1) in self.lengths:
                        array = (self.lengths[match.group(1)], 0)
            elif new_character == ",":
                array = (array[0], array[1] + 1)
            elif new_character == "]":
                array = None
            return FixedArrayCountGuard(self.inner.add_character(new_character), self.lengths, tail=tail,
                                        array=array, last=new_character)

        def get_allowed_characters(self) -> str:
            allowed = self.inner.get_allowed_characters()
            if self.array is None or not self.last.isdigit():
                return allowed
            length, commas = self.array
            needed = "]" if commas + 1 >= length else ","
            return "".join(character for character in allowed if character not in WHITESPACE_CHARACTERS
                           or needed in self.inner.add_character(character).get_allowed_characters())

        def can_end(self) -> bool:
            return self.inner.can_end()

        def shortcut_key(self):
            return self.inner.shortcut_key()

        def cache_key(self):
            key = self.inner.cache_key()
            return None if key is None else (key, self.array, self.last)

    return FixedArrayCountGuard


def schema_character_parser(schema: Mapping[str, Any]):
    """约束解码用的字符级解析器：LMFE 的 JSON schema 解析器，外加定长整数数组的项数守卫。"""
    from lmformatenforcer import JsonSchemaParser

    return _fixed_array_guard_class()(JsonSchemaParser(dict(schema)), fixed_number_arrays(schema))


def schema_prefix_allowed_tokens_fn(tokenizer: Any, schema: Mapping[str, Any]):
    """用LMFE公开扩展API接HF原生token回调，不导入不兼容HF5的旧integration模块。

    TokenEnforcer / TokenEnforcerTokenizerData是作者公开的新推理框架扩展入口；词表解码算法沿作者0.11.3实现。
    来源：https://github.com/noamgat/lm-format-enforcer/blob/v0.11.3/lmformatenforcer/tokenenforcer.py
    及同版本 integrations/transformers.py。这里不修改库、模型或tokenizer的类与属性；字符级解析器见
    ``schema_character_parser``。
    """
    from lmformatenforcer import TokenEnforcer

    enforcer = TokenEnforcer(_schema_tokenizer_data(tokenizer), schema_character_parser(schema))

    def allowed_tokens(batch_id, input_ids):
        return enforcer.get_allowed_tokens(input_ids.tolist()).allowed_tokens

    return allowed_tokens


def stop_token_ids(tokenizer: Any, generation_config: Any) -> list[int]:
    """模型登记的 eos（int 或 list）加分词器的 eos；分词器的 eos 必须就是聊天回合结束符 ``<|im_end|>``。"""
    registered = generation_config.eos_token_id
    ids = {registered} if isinstance(registered, int) else set(registered or ())
    chat_end = tokenizer.eos_token_id
    if chat_end is None or tokenizer.convert_ids_to_tokens(chat_end) != CHAT_END:
        raise ValueError(f"tokenizer eos must be {CHAT_END}, got id {chat_end!r}")
    return sorted(ids | {chat_end})


def finish_reason(new_ids: Sequence[int], stop_ids: Sequence[int], max_new_tokens: int) -> str:
    """``stop``：最后一个生成的 token 是停止符；``length``：用满 ``max_new_tokens`` 仍没有以停止符收尾。
    答复中间出现过停止符不算停（那正是越过 ``<|im_end|>`` 继续编的情形）。其余情形不该出现，报错。"""
    if new_ids and new_ids[-1] in stop_ids:
        return "stop"
    if len(new_ids) >= max_new_tokens:
        return "length"
    raise ValueError(f"generation ended after {len(new_ids)} tokens without a stop token (limit {max_new_tokens})")


def answer_row(sample_id: str, text: str, new_ids: Sequence[int], stop_ids: Sequence[int], *, max_new_tokens: int,
               prompt_tokens: int, elapsed_seconds: float) -> Mapping[str, Any]:
    """``run/planner_generate.py`` 写的一行。status 只在以停止符收尾时是 completed，撞上限记 incomplete。"""
    reason = finish_reason(new_ids, stop_ids, max_new_tokens)
    return {"case_id": sample_id, "raw_content": text, "status": "completed" if reason == "stop" else "incomplete",
            "finish_reason": reason, "new_tokens": len(new_ids), "prompt_tokens": prompt_tokens,
            "elapsed_seconds": elapsed_seconds}
