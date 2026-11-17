"""源帧实体点到原文字CA的路由与有限格辅助监督；不新增视觉token或分割器。"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, NamedTuple, Sequence

import torch

from mspc.models.planner_tokens import GRID_COLUMNS, GRID_ROWS


class WanVideoPrediction(NamedTuple):
    noise: torch.Tensor
    source_alignment_loss: torch.Tensor | None


class WanVideoBlockPrediction(NamedTuple):
    hidden_states: torch.Tensor
    source_alignment_loss: torch.Tensor | None


class WanTextAttentionPrediction(NamedTuple):
    hidden_states: torch.Tensor
    source_alignment_loss: torch.Tensor | None


@dataclass(frozen=True)
class SourceIdentityOptions:
    loss_weight: float
    alignment_layer: int
    max_alignment_cells_per_reference: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceIdentityOptions":
        if not isinstance(value, Mapping) or set(value) != {
                "loss_weight", "alignment_layer", "max_alignment_cells_per_reference"}:
            raise ValueError("source_identity requires loss_weight, alignment_layer and max_alignment_cells_per_reference")
        if type(value["loss_weight"]) not in (int, float) or not math.isfinite(value["loss_weight"]) or value["loss_weight"] <= 0:
            raise ValueError("source_identity.loss_weight must be finite and positive")
        if type(value["alignment_layer"]) is not int or not 0 <= value["alignment_layer"] < 29:
            raise ValueError("source_identity.alignment_layer must precede the last Wan5B block")
        if type(value["max_alignment_cells_per_reference"]) is not int or not 1 <= value["max_alignment_cells_per_reference"] <= 64:
            raise ValueError("source_identity.max_alignment_cells_per_reference must be in [1,64]")
        return cls(float(value["loss_weight"]), value["alignment_layer"], value["max_alignment_cells_per_reference"])


@dataclass(frozen=True)
class SourceIdentityRow:
    """一张实际参考帧：实体文字列、点格及可选的训练/验证非点格答案。

    points/alignment_cells 的元组为 (entity_index,row,column)。辅助答案从不作为
    注意力前向mask；没有答案的推理记录用alignment_cells=None。
    """
    entity_names: tuple[str, ...]
    entity_tokens: tuple[tuple[int, ...], ...]
    common_tokens: tuple[int, ...]
    points: tuple[tuple[int, int, int], ...]
    alignment_cells: tuple[tuple[int, int, int], ...] | None

    def __post_init__(self):
        if len(self.entity_names) != len(self.entity_tokens) or len(set(self.entity_names)) != len(self.entity_names):
            raise ValueError("source entities and their unique text spans must match")
        if any(not tokens or len(set(tokens)) != len(tokens) or min(tokens) < 0 for tokens in self.entity_tokens):
            raise ValueError("each source entity requires nonempty unique text token indices")
        if any(type(token) is not int or token < 0 for token in self.common_tokens):
            raise ValueError("common source tokens must be nonnegative integers")
        for entries in (self.points, self.alignment_cells or ()):
            for entity, row, column in entries:
                if (type(entity) is not int or not 0 <= entity < len(self.entity_names)
                        or type(row) is not int or not 0 <= row < GRID_ROWS
                        or type(column) is not int or not 0 <= column < GRID_COLUMNS):
                    raise ValueError("source point/label is outside its entity or visual grid")
        seed_cells = {(row, column) for _, row, column in self.points}
        if self.alignment_cells is not None:
            if len(self.alignment_cells) > 64:
                raise ValueError("source alignment reads at most 64 cells per reference")
            if any((row, column) in seed_cells for _, row, column in self.alignment_cells):
                raise ValueError("source alignment must exclude every routed point cell")
            seeded = {entity for entity, _, _ in self.points}
            if any(entity not in seeded for entity, _, _ in self.alignment_cells):
                raise ValueError("unseeded source identities cannot supply alignment answers")
            cells = [(row, column) for _, row, column in self.alignment_cells]
            if len(set(cells)) != len(cells):
                raise ValueError("source alignment cells need one unambiguous identity")

    def without_alignment(self) -> "SourceIdentityRow":
        return replace(self, alignment_cells=None)


@dataclass(frozen=True)
class SourceIdentityCondition:
    rows: tuple[SourceIdentityRow, ...]
    alignment_layer: int

    def __post_init__(self):
        if not self.rows or len(self.rows) > 10:
            raise ValueError("source identity condition requires 1..10 actual reference rows")
        if type(self.alignment_layer) is not int or not 0 <= self.alignment_layer < 29:
            raise ValueError("source identity alignment layer must precede the last Wan5B block")

    def for_layer(self, index: int, depth: int) -> "SourceIdentityCondition | None":
        if not 0 <= index < depth or self.alignment_layer >= depth - 1:
            raise ValueError("source identity layer does not belong to this DiT")
        # 最后一层源CA之后没有目标SA消费，因此不做源点路由。
        if index == depth - 1:
            return None
        return self if index == self.alignment_layer else replace(
            self, rows=tuple(row.without_alignment() for row in self.rows))


def source_alignment_loss(q: torch.Tensor, k: torch.Tensor, *, value_dtype: torch.dtype,
                          num_heads: int, row: SourceIdentityRow) -> torch.Tensor | None:
    """在实际中层CA的Q/K上逐head比较实体词段，softmax的文字分母严格抵消。"""
    if not row.alignment_cells:
        return None
    active = sorted({entity for entity, _, _ in row.points})
    if len(active) < 2:
        return None
    if any(entity not in active for entity, _, _ in row.alignment_cells):
        raise ValueError("source alignment target has no point seed")
    cells = [r * GRID_COLUMNS + c for _, r, c in row.alignment_cells]
    text_keys = sorted({token for entity in active for token in row.entity_tokens[entity]})
    if not text_keys or text_keys[-1] >= k.shape[1]:
        raise ValueError("source entity tokens do not fit the actual text row")
    head_dim = q.shape[-1] // num_heads
    if q.shape[-1] % num_heads or q.shape[0] != 1 or k.shape[0] != 1:
        raise ValueError("source alignment requires one latent batch and complete attention heads")
    query = q[:, cells].to(value_dtype).float().reshape(1, len(cells), num_heads, head_dim).transpose(1, 2)
    keys = k[:, text_keys].to(value_dtype).float().reshape(1, len(text_keys), num_heads, head_dim).transpose(1, 2)
    logits = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(head_dim)
    lookup = {token: index for index, token in enumerate(text_keys)}
    scores = torch.stack([
        torch.logsumexp(logits[..., [lookup[token] for token in row.entity_tokens[entity]]], dim=-1)
        - math.log(len(row.entity_tokens[entity])) for entity in active
    ], dim=-1)
    target_values = [active.index(entity) for entity, _, _ in row.alignment_cells]
    targets = torch.tensor(target_values, device=q.device, dtype=torch.long)
    losses = torch.nn.functional.cross_entropy(
        scores.reshape(-1, len(active)), targets.repeat(num_heads), reduction="none").reshape(num_heads, len(cells))
    # 每人物等权，再由CA caller对有效参考帧等权；大mask不获得更多权重。
    groups = {entity: [index for index, target in enumerate(target_values) if target == entity]
              for entity in set(target_values)}
    return torch.stack([losses[:, indices].mean() for indices in groups.values()]).mean()


def routed_source_attention(attention, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
                            num_heads: int, row: SourceIdentityRow) -> WanTextAttentionPrediction:
    """只在源点格改变原CA的文字列，其余源格完整读取原行；输入监督不参与路由。"""
    if q.shape[0] != 1 or q.shape[1] != GRID_ROWS * GRID_COLUMNS:
        raise ValueError("source CA must receive one complete 15x26 reference query grid")
    if any(token >= k.shape[1] for tokens in row.entity_tokens for token in tokens) or any(
            token >= k.shape[1] for token in row.common_tokens):
        raise ValueError("source routes do not fit the actual encoded text")
    routes: dict[int, set[int]] = {}
    for entity, r, c in row.points:
        routes.setdefault(r * GRID_COLUMNS + c, set(row.common_tokens)).update(row.entity_tokens[entity])
    if not routes:
        values = attention(q, k, v)
    else:
        values = torch.zeros_like(q)
        ordinary = [index for index in range(q.shape[1]) if index not in routes]
        if ordinary:
            indices = torch.tensor(ordinary, dtype=torch.long, device=q.device)
            values = values.index_copy(1, indices, attention(q[:, indices].contiguous(), k, v))
        groups: dict[tuple[int, ...], list[int]] = {}
        for cell, tokens in routes.items():
            groups.setdefault(tuple(sorted(tokens)), []).append(cell)
        for tokens, cells in groups.items():
            indices = torch.tensor(cells, dtype=torch.long, device=q.device)
            selected = torch.tensor(tokens, dtype=torch.long, device=k.device)
            values = values.index_copy(1, indices, attention(
                q[:, indices].contiguous(), k[:, selected].contiguous(), v[:, selected].contiguous()))
    loss = source_alignment_loss(q, k, value_dtype=v.dtype, num_heads=num_heads, row=row)
    return WanTextAttentionPrediction(values, loss)


def compile_source_identity_condition(tokenizer, prompts: Sequence[str], references: Sequence[Mapping[str, Any]], *,
                                      characters: Sequence[Mapping[str, Any]],
                                      shot_objects: Sequence[Sequence[str]],
                                      settings: Sequence[str | None],
                                      alignment_layer: int,
                                      max_alignment_cells_per_reference: int) -> SourceIdentityCondition | None:
    """最终实际文字编译后再对齐T5 token，不解析名字字符串。只有训练调用：源点来自离线逐帧标签；推理的参考身份行由分镜写、
    不读图，没有源点，源身份路由的渲染器不出片（:func:`mspc.inference.segment_story.require_story_renderer`）。"""
    from mspc.data.shot_text import reference_text_with_entity_spans
    from mspc.models.spatial_layout_inputs import align_entity_token_indices
    if not references:
        return None
    if type(max_alignment_cells_per_reference) is not int or not 1 <= max_alignment_cells_per_reference <= 64:
        raise ValueError("source identity maximum alignment cells must be in [1,64]")
    if len(prompts) < len(references):
        raise ValueError("source identity rows outnumber the actual prompt rows")
    if len(shot_objects) != len(references) or len(settings) != len(references):
        raise ValueError("source object and location phrases must match every actual reference row")
    rows = []
    for prompt, record, objects, setting in zip(prompts, references, shot_objects, settings):
        compiled = reference_text_with_entity_spans(record["entities"], characters=characters, setting=setting,
                                                    shot_objects=objects)
        if compiled["text"] != prompt:
            raise ValueError("source entity spans do not describe the actual consumed caption")
        spans, valid = align_entity_token_indices(tokenizer, prompt, compiled["entity_spans"])
        names = tuple(entity["name"] for entity in record["entities"])
        if set(spans) != set(names):
            raise ValueError("source identity caption does not cover exactly its visible entities")
        indices = {name: index for index, name in enumerate(names)}
        owned = {token for tokens in spans.values() for token in tokens}
        points = tuple((indices[entity["name"]], *entity["point_cell"])
                       for entity in record["entities"] if entity["point_cell"] is not None)
        labels = record["alignment_cells"]
        if labels is not None and len(labels) > max_alignment_cells_per_reference:
            raise ValueError("source labels exceed the explicitly configured alignment-cell budget")
        rows.append(SourceIdentityRow(names, tuple(spans[name] for name in names),
            tuple(index for index, include in enumerate(valid) if include and index not in owned), points,
            None if labels is None else tuple((indices[label["name"]], *label["cell"]) for label in labels)))
    return SourceIdentityCondition(tuple(rows), alignment_layer)
