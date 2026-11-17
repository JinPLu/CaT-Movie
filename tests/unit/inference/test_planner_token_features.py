"""规划器 token 特征：标签→作答与提示里的作答形状一致；条目区间恰好盖住「名字＋镜首点」；取隐状态按这些区间平均，
前向只算到所需最高层（训练的教师强制与推理生成后的前向走同一函数）。"""
import json
from types import SimpleNamespace

import numpy as np
import torch

from mspc.inference.planner_token_features import POOLINGS, entry_spans, token_features
from mspc.inference.reference_pointing import answer_from_labels

LABELS = {"moments": [[0, 24]], "picks": [{"name": "Character 1", "shot_index": 0, "frame": 24, "point_2d": [400, 300]},
                    {"name": "Character 2", "shot_index": 0, "frame": 24, "point_2d": [700, 350]}],
          "starts": [{"shot_index": 2, "characters": [{"name": "Character 1", "point_2d": [300, 520]},
                                                     {"name": "Character 2", "point_2d": None}]}]}


def test_answer_has_the_prompt_form_and_one_based_shots():
    answer = json.loads(answer_from_labels(LABELS))
    assert answer == {"references": [{"reference": 1, "characters": [
        {"name": "Character 1", "point_2d": [400, 300]}, {"name": "Character 2", "point_2d": [700, 350]}]}],
        "shots": [{"shot": 3, "characters": [{"name": "Character 1", "point_2d": [300, 520]},
                                             {"name": "Character 2", "point_2d": None}]}]}


def test_entry_spans_cover_each_pooling_in_both_sections():
    answer = answer_from_labels(LABELS)
    spans = entry_spans(answer)
    assert [(s, k, n) for s, k, n, _ in spans] == [("shot", 3, "Character 1"), ("shot", 3, "Character 2"),
                                                   ("reference", 1, "Character 1"), ("reference", 1, "Character 2")]
    _, _, _, first = spans[0]
    assert answer[slice(*first["entry"])] == '{"name": "Character 1", "point_2d": [300, 520]}'
    assert answer[slice(*first["name"])] == "Character 1"
    assert answer[slice(*first["point"])] == "300, 520"
    assert answer[slice(*spans[1][3]["point"])] == "null"
    assert first["entry"][0] > answer.index('"shots"') and spans[2][3]["entry"][0] > answer.index('"references"')


class _Tokenizer:
    """一字符一 token，便于核对区间。"""

    def __call__(self, text, add_special_tokens, return_offsets_mapping):
        return {"input_ids": [ord(c) % 97 for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class _Layer(torch.nn.Module):
    """第 L 层把输入乘 (L+2)/(L+1)：嵌入是 x 时第 L 层输出 x×(L+2)。"""

    def __init__(self, index, last):
        super().__init__()
        self.index, self.last = index, last

    def forward(self, x):
        if self.index > self.last:
            raise AssertionError(f"layer {self.index} runs after the highest wanted layer {self.last}")
        return x * (self.index + 2) / (self.index + 1)


class _Model:
    """假的 Qwen3.5：嵌入后依次过各解码层，末尾不该被走到（取特征只算到所需最高层）。"""
    device = "cpu"

    def __init__(self, embed, wanted_last, depth=6):
        self.embed = embed
        self.model = SimpleNamespace(language_model=SimpleNamespace(
            layers=torch.nn.ModuleList(_Layer(i, wanted_last) for i in range(depth))))

    def __call__(self, input_ids, attention_mask):
        x = self.embed(input_ids)
        for layer in self.model.language_model.layers:
            x = layer(x)
        raise AssertionError("the forward reached the final norm and the output head")


def test_features_average_exactly_the_entry_tokens():
    answer = answer_from_labels(LABELS)
    prompt_ids = torch.zeros(1, 5, dtype=torch.long)

    def call(**kwargs):
        return {"input_ids": prompt_ids, "attention_mask": torch.ones_like(prompt_ids)}

    model = _Model(lambda ids: torch.arange(ids.shape[1], dtype=torch.float32)[None, :, None].expand(1, -1, 3),
                   wanted_last=2)
    # 第 L 层输出＝位置×(L+2)：第 1 层＝位置×3，第 2 层＝位置×4；第 3 层起不算。
    entries, features = token_features(model, _Processor(call), prompt="p", images=None,
                                       answer=answer, layers=[1, 2])
    spans = entry_spans(answer)
    assert entries == [(s, k, n) for s, k, n, _ in spans]
    for pooling in POOLINGS:
        begin, end = spans[0][3][pooling]
        expected = np.mean([5 + i for i in range(begin, end)])
        assert np.allclose(features[(pooling, 1)][0], expected * 3)
        assert np.allclose(features[(pooling, 2)][0], expected * 4)
    assert not any(layer._forward_hooks for layer in model.model.language_model.layers)


def test_an_indented_answer_gives_the_same_entries_and_features():
    answer = answer_from_labels(LABELS)
    prompt_ids = torch.zeros(1, 5, dtype=torch.long)

    def call(**kwargs):
        return {"input_ids": prompt_ids, "attention_mask": torch.ones_like(prompt_ids)}

    model = _Model(lambda ids: ids[0].float()[None, :, None].expand(1, -1, 3), wanted_last=1)
    compact = token_features(model, _Processor(call), prompt="p", images=None, answer=answer, layers=[1])
    indented = token_features(model, _Processor(call), prompt="p", images=None,
                              answer=json.dumps(json.loads(answer), indent=2), layers=[1])
    assert compact[0] == indented[0]
    assert all(np.array_equal(compact[1][k], indented[1][k]) for k in compact[1])


def test_no_people_in_the_targets_gives_no_entries_without_a_forward():
    labels = {**LABELS, "moments": [], "picks": [], "starts": [{"shot_index": 2, "characters": []}]}

    class Model:
        config = SimpleNamespace(get_text_config=lambda: SimpleNamespace(hidden_size=6))

        def __call__(self, **kwargs):
            raise AssertionError("no entries, no forward")

    entries, features = token_features(Model(), _Processor(None), prompt="p", images=None,
                                       answer=answer_from_labels(labels), layers=[17, 19])
    assert entries == [] and features[("name", 19)].shape == (0, 6)


class _Processor:
    def __init__(self, call):
        self.call, self.tokenizer = call, _Tokenizer()

    def __call__(self, **kwargs):
        return self.call(**kwargs)
