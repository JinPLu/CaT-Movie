"""规划器 token 的特征：Qwen3.5-9B 底座做读历史第二步（看参考帧图片指人、给各目标镜镜首点，提示见
``mspc.inference.reference_pointing``），对「提示＋答案」做一次前向，取答案里每个人物条目（各目标镜的镜首条目、
各参考图里指出此人的条目）在指定各层的平均隐状态，名字、点、整条三种口径各存一份。训练时答案是读历史标签（教师强制），推理时答案是规划器自己生成的；两边都经这里的同一段代码。

训练时对规划器看到的图片做亮度、噪声、JPEG 增强，每段一个增强版本（按样本号定种子）。
"""
from __future__ import annotations

import io
import json
from typing import Any, Mapping, Sequence

import numpy as np


def augment(frames: np.ndarray, seed: int) -> np.ndarray:
    """亮度（×U(0.8,1.2)）、高斯噪声（σ~U(0,6)）、JPEG（质量 U(40,90)）；一段一组参数。"""
    from PIL import Image

    rng = np.random.default_rng(seed)
    gain, sigma, quality = rng.uniform(0.8, 1.2), rng.uniform(0.0, 6.0), int(rng.integers(40, 91))
    out = np.clip(frames.astype(np.float32) * gain + rng.normal(0.0, sigma, frames.shape), 0, 255).astype(np.uint8)
    encoded = []
    for frame in out:
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="JPEG", quality=quality)
        encoded.append(np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")))
    return np.stack(encoded)


SECTIONS = {"shots": "shot", "references": "reference"}
POOLINGS = ("name", "point", "entry")


def entry_spans(answer: str) -> list[tuple[str, int, str, dict[str, tuple[int, int]]]]:
    """作答里每个人物条目的字符区间：(段, 序号, 名字, {口径: (起, 止)})。

    段是 ``shot``（"shots" 里各目标镜的镜首条目，序号是镜号）或 ``reference``（"references" 里参考图 i 中的此人，
    序号是参考序号）。口径：``entry`` 整条 ``{"name": ..., "point_2d": ...}``；``name`` 名字的值（不含引号）；
    ``point`` 点的数字（``[x, y]`` 里从第一个数字到最后一个数字；null 时就是 ``null``）。"""
    data = json.loads(answer)
    spans = []
    for key, section in SECTIONS.items():
        if key not in data:  # 点控制加特征注入的作答只有镜首段
            continue
        cursor = answer.index(f'"{key}"')
        for row in data[key]:
            for entry in row["characters"]:
                text = json.dumps(entry)
                begin = answer.index(text, cursor)
                name_begin = begin + text.index('"name": "') + len('"name": "')
                point_text = json.dumps(entry["point_2d"])
                point_begin = begin + text.index('"point_2d": ') + len('"point_2d": ')
                point = ((point_begin + 1, point_begin + len(point_text) - 1) if entry["point_2d"] is not None
                         else (point_begin, point_begin + len(point_text)))
                spans.append((section, row["shot" if section == "shot" else "reference"], entry["name"],
                              {"entry": (begin, begin + len(text)),
                               "name": (name_begin, name_begin + len(entry["name"])), "point": point}))
                cursor = begin + len(text)
    return spans


def prompt_text(processor, content: Sequence[Mapping[str, Any]]) -> str:
    """请求内容（文字与图片占位交替）→ 聊天模板文字（非思考、带生成前缀）。"""
    return processor.apply_chat_template([{"role": "user", "content": list(content)}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


class _ReachedLastLayer(Exception):
    """所需最高层已算完：从该层的前向 hook 里抛出，后面的层与输出头不算。"""


def layer_outputs(model, inputs: Mapping[str, Any], layers: Sequence[int]) -> dict[int, Any]:
    """前向到 ``layers`` 里最高的一层为止，返回 {层: 该解码层输出 [1, token 数, 隐维]}（层号从 0 起）。

    取的是解码层自身的输出，与完整前向 ``output_hidden_states`` 的 ``hidden_states[L + 1]`` 是同一个张量
    （最后一层除外：那一项被换成过末层 RMSNorm 的输出）；之后的层、末层 RMSNorm、输出头都不算。"""
    import torch

    decoder_layers = model.model.language_model.layers
    if not 0 <= min(layers) <= max(layers) < len(decoder_layers):
        raise ValueError(f"layers {list(layers)} outside the {len(decoder_layers)} decoder layers")
    outputs = {}

    def keep(layer):
        def hook(module, args, output):
            outputs[layer] = output
            if layer == max(layers):
                raise _ReachedLastLayer
        return hook

    handles = [decoder_layers[layer].register_forward_hook(keep(layer)) for layer in set(layers)]
    try:
        with torch.no_grad():
            model(**inputs)
    except _ReachedLastLayer:
        pass
    finally:
        for handle in handles:
            handle.remove()
    if set(outputs) != set(layers):
        raise RuntimeError(f"forward stopped before layers {sorted(set(layers) - set(outputs))}")
    return outputs


def token_features(model, processor, *, prompt: str, images, answer: str, layers: Sequence[int]
                   ) -> tuple[list[tuple[str, int, str]], dict[tuple[str, int], np.ndarray]]:
    """对「提示＋答案」前向一次（只算到所需最高层，见 :func:`layer_outputs`），返回
    (各条目的 (段, 序号, 名字), {(口径, 层): [条目数, 隐维]} 的平均隐状态)。
    条目顺序同 :func:`entry_spans`（先 shots，后 references）。``images`` 是提示里图片占位依次对应的 PIL 图
    （没有图时为 None）。层号从 0 起。
    作答没有条目（目标镜与参考里都没有登记角色）时不做前向，各项返回 [0, 隐维]。
    作答先规范成与 ``reference_pointing.answer_from_labels`` 相同的紧凑写法（规划器原样输出常带缩进换行），
    训练与推理前向的是同一种文字形式。"""
    import torch

    answer = json.dumps(json.loads(answer))
    spans = entry_spans(answer)
    if not spans:
        width = model.config.get_text_config().hidden_size
        return [], {(pooling, layer): np.zeros((0, width), dtype=np.float32) for pooling in POOLINGS for layer in layers}
    tokenizer = processor.tokenizer
    inputs = processor(text=[prompt], images=images, return_tensors="pt")
    encoded = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    answer_ids = torch.tensor([encoded["input_ids"]], dtype=inputs["input_ids"].dtype)
    prompt_length = inputs["input_ids"].shape[1]
    full = dict(inputs)
    full["input_ids"] = torch.cat([inputs["input_ids"], answer_ids], dim=1)
    full["attention_mask"] = torch.ones_like(full["input_ids"])
    if "mm_token_type_ids" in full:
        full["mm_token_type_ids"] = torch.cat(
            [inputs["mm_token_type_ids"], torch.zeros_like(answer_ids, dtype=inputs["mm_token_type_ids"].dtype)], dim=1)
    full = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in full.items()}
    hidden = layer_outputs(model, full, layers)
    entries, rows = [], {pooling: [] for pooling in POOLINGS}
    for section, key, name, by_pooling in spans:
        entries.append((section, key, name))
        for pooling, (begin, end) in by_pooling.items():
            tokens = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < end and b > begin]
            if not tokens:
                raise ValueError(f"no answer tokens cover the {pooling} of {name} in {section} {key}")
            rows[pooling].append([prompt_length + i for i in tokens])
    features = {(pooling, layer): np.stack([hidden[layer][0, row].float().mean(0).cpu().numpy()
                                            for row in rows[pooling]])
                for pooling in POOLINGS for layer in layers}
    return entries, features
