import copy
import html
import json
from pathlib import Path
import re

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from mspc.models.spatial_layout_inputs import (SpatialLayoutOptions,
    align_entity_token_indices, compile_spatial_layout_inputs, encode_spatial_layout_inputs)


class ActualFastTokenizer:
    """本地小词表检验offset/tokenizer接线，不冒充正式umT5词表验收。"""
    clean = "whitespace"
    seq_len = 512

    def __init__(self):
        backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2}, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        backend.post_processor = processors.TemplateProcessing(single="$A [EOS]", special_tokens=[("[EOS]", 2)])
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")

    def _clean(self, text):
        return re.sub(r"\s+", " ", html.unescape(html.unescape(text))).strip()

    def __call__(self, text, return_mask=False, **kwargs):
        result = self.tokenizer([self._clean(text)], padding="max_length", truncation=True,
                                max_length=self.seq_len, return_tensors="pt", **kwargs)
        return (result.input_ids, result.attention_mask) if return_mask else result.input_ids


def test_repeated_equal_appearance_remains_entity_specific_after_cleaning():
    tokenizer = ActualFastTokenizer()
    text = "  A &amp; B: a red cup.\nThen another a red cup. "
    first, last = text.index("a red cup"), text.rindex("a red cup")
    ids, mask = align_entity_token_indices(tokenizer, text, [
        {"name": "Object 1", "start": first, "end": first + 9},
        {"name": "Object 2", "start": last, "end": last + 9},
    ])
    assert set(ids["Object 1"]).isdisjoint(ids["Object 2"])
    assert all(mask[index] for indices in ids.values() for index in indices)
    assert mask.count(True) < len(mask)


def test_truncation_and_slow_tokenizer_do_not_guess_offsets():
    tokenizer = ActualFastTokenizer()
    tokenizer.seq_len = 4
    with pytest.raises(ValueError, match="truncated"):
        align_entity_token_indices(tokenizer, "one two three four five", [])
    tokenizer.tokenizer = object()
    with pytest.raises(ValueError, match="actual fast"):
        align_entity_token_indices(tokenizer, "one", [])


def test_real_storyboard_compiles_layout_and_consumed_text_same_no_reference_required():
    path = Path(__file__).parents[2] / "fixtures/block_annotation_v18/film-002869-b0001.json"
    annotation = json.loads(path.read_text())["annotation"]
    annotation = copy.deepcopy(annotation)
    entity = annotation["shots"][0]["present"][0]["name"]
    annotation["layouts"] = [{"shot": 1, "entities": [{"name": entity, "anchors": [
        {"time": 0.5, "visibility": "visible", "box": [0.1, 0.2, 0.4, 0.8]}]}]}]
    rows = compile_spatial_layout_inputs(annotation, [0])
    options = SpatialLayoutOptions.from_mapping({"text_attention_backend": "sdpa", "strength": 0.7,
                                                "layer_indices": [0, 2], "timestep_range": [0, 1000], "mode": "text_cross_attention_bias_v1"})
    condition = encode_spatial_layout_inputs(ActualFastTokenizer(), rows,
        prompts=[rows[0]["text"]], frame_counts=[9], options=options)
    assert condition.shots[0].entities[0].anchors[0].time == 0.5
    assert condition.shots[0].entities[0].token_indices
    with pytest.raises(ValueError, match="actual target prompt"):
        encode_spatial_layout_inputs(ActualFastTokenizer(), rows, prompts=["different"], frame_counts=[9], options=options)
    annotation.pop("layouts")
    with pytest.raises(ValueError, match="unannotated"):
        compile_spatial_layout_inputs(annotation, [0])
    unknown = compile_spatial_layout_inputs(annotation, [0], allow_unannotated=True)
    assert unknown[0]["annotation_status"] == "unannotated"
    assert unknown[0]["entities"] == [] and unknown[0]["text"]


def test_actual_compiled_reference_text_and_spans_are_reused_without_a_second_compilation(monkeypatch):
    from mspc.data.shot_text import shot_text_with_entity_spans
    path = Path(__file__).parents[2] / "fixtures/block_annotation_v18/film-002869-b0001.json"
    annotation = json.loads(path.read_text())["annotation"]
    annotation['layouts'] = []
    compiled = {'shot_index': 0, **shot_text_with_entity_spans(
        annotation, 0, references=[{'index': 2, 'entities': ['Character 1']}])}

    def unexpected(*args, **kwargs):
        raise AssertionError('actual compiled text must not be rebuilt without its reference use')

    monkeypatch.setattr('mspc.data.shot_text.shot_text_with_entity_spans', unexpected)
    rows = compile_spatial_layout_inputs(annotation, [0], compiled_rows=[compiled])
    assert rows[0]['text'] == compiled['text']
    assert rows[0]['entity_spans'] == compiled['entity_spans']
    with pytest.raises(ValueError, match='actual shot order'):
        compile_spatial_layout_inputs(annotation, [0], compiled_rows=[{**compiled, 'shot_index': 1}])
    with pytest.raises(ValueError, match='inside the actual text'):
        compile_spatial_layout_inputs(annotation, [0], compiled_rows=[{**compiled, 'entity_spans': [
            {**compiled['entity_spans'][0], 'end': len(compiled['text']) + 1}]}])


@pytest.mark.parametrize("change", [{"strength": float("nan")}, {"layer_indices": [30]},
    {"timestep_range": [1, 0]}, {"text_attention_backend": "native"}])
def test_invalid_options_are_rejected(change):
    raw = {"text_attention_backend": "sdpa", "strength": 0.7,
           "layer_indices": [0], "timestep_range": [0, 1000], "mode": "text_cross_attention_bias_v1"}
    with pytest.raises(ValueError):
        SpatialLayoutOptions.from_mapping(raw | change)
