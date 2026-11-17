"""规划布局到原 umT5 文本行的精确对应；不替换 tokenizer，不猜 token 位移。"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SpatialLayoutOptions:
    """渲染器怎样用规划的框；三种模式见 ``mspc.models.spatial_layout``。

    加分模式（旧框条件组，复现已交结果用）：恰好 backend、strength、layer_indices、timestep_range、mode。
    压低模式另需 region_dilation、onset_latent_frames。门控模式再需 region_module
    （fourier_frequencies、learning_rate），文字注意力用原生内核、strength 为 0。
    """
    text_attention_backend: str
    strength: float
    layer_indices: tuple[int, ...]
    timestep_range: tuple[float, float]
    mode: str
    region_dilation: float | None
    onset_latent_frames: int | None
    region_module: Mapping[str, Any] | None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SpatialLayoutOptions":
        from mspc.models.spatial_layout import GATED_REGION_MODE, SPATIAL_LAYOUT_MODE, SUPPRESSION_MODE
        if not isinstance(raw, Mapping) or "mode" not in raw:
            raise ValueError("spatial_layout requires an explicit mode")
        mode = raw["mode"]
        base = {"text_attention_backend", "strength", "layer_indices", "timestep_range", "mode"}
        required = {SPATIAL_LAYOUT_MODE: base,
                    SUPPRESSION_MODE: base | {"region_dilation", "onset_latent_frames"},
                    GATED_REGION_MODE: base | {"region_dilation", "onset_latent_frames", "region_module"}}.get(mode)
        if required is None:
            raise ValueError(f"unsupported spatial layout mode: {mode!r}")
        if set(raw) != required:
            raise ValueError(f"spatial_layout mode {mode} requires exactly {sorted(required)}")
        backend = "native" if mode == GATED_REGION_MODE else "sdpa"
        if raw["text_attention_backend"] != backend:
            raise ValueError(f"spatial_layout mode {mode} requires text_attention_backend={backend!r}")
        strength = raw["strength"]
        layers, interval = raw["layer_indices"], raw["timestep_range"]
        if type(strength) not in (int, float) or not math.isfinite(strength) or strength < 0:
            raise ValueError("spatial strength must be finite and nonnegative")
        if mode == GATED_REGION_MODE and strength != 0:
            raise ValueError("the gated region mode reads no text bias; strength must be 0")
        if (not isinstance(layers, (list, tuple)) or not layers
                or any(type(i) is not int or not 0 <= i < 30 for i in layers)
                or len(set(layers)) != len(layers)):
            raise ValueError("layout layer_indices must be unique Wan5B layer indices in [0,30)")
        if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                or any(type(t) not in (int, float) or not math.isfinite(t) for t in interval)
                or not 0 <= interval[0] <= interval[1] <= 1000):
            raise ValueError("timestep_range uses original scheduler t in [0,1000]")
        dilation = raw.get("region_dilation")
        onset = raw.get("onset_latent_frames")
        if mode != SPATIAL_LAYOUT_MODE:
            if type(dilation) not in (int, float) or not 0 <= dilation < 1:
                raise ValueError("region_dilation must be a fraction in [0, 1)")
            if type(onset) is not int or onset < 1:
                raise ValueError("onset_latent_frames must be a positive int")
            dilation = float(dilation)
        module = raw.get("region_module")
        if mode == GATED_REGION_MODE:
            if not isinstance(module, Mapping) or set(module) != {"fourier_frequencies", "learning_rate"}:
                raise ValueError("region_module requires exactly fourier_frequencies and learning_rate")
            if type(module["fourier_frequencies"]) is not int or module["fourier_frequencies"] < 1:
                raise ValueError("region_module.fourier_frequencies must be a positive int")
            rate = module["learning_rate"]
            if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
                raise ValueError("region_module.learning_rate must be positive")
            module = {"fourier_frequencies": module["fourier_frequencies"], "learning_rate": float(rate)}
        return cls(raw["text_attention_backend"], float(strength), tuple(layers), tuple(map(float, interval)),
                   mode, dilation, onset, module)

    def to_mapping(self) -> dict[str, Any]:
        """与 ``from_mapping`` 互逆：只写该模式要求的键。"""
        mapping = {"text_attention_backend": self.text_attention_backend, "strength": self.strength,
                   "layer_indices": list(self.layer_indices), "timestep_range": list(self.timestep_range),
                   "mode": self.mode}
        if self.region_dilation is not None:
            mapping.update(region_dilation=self.region_dilation, onset_latent_frames=self.onset_latent_frames)
        if self.region_module is not None:
            mapping["region_module"] = dict(self.region_module)
        return mapping


def layout_shot_coverage(annotations: Mapping[str, Mapping[str, Any]],
                         samples: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """抽样目标镜里，带至少一条布局的镜数 / 目标镜数。缺 layouts 字段的块计为未覆盖。"""
    covered = total = 0
    for sample in samples:
        annotation = annotations[sample["case_id"]]
        present = {entry["shot"] for entry in annotation["layouts"]} if "layouts" in annotation else set()
        start, end = sample["span"]
        for shot in range(start, end):
            total += 1
            covered += int((shot + 1) in present)
    return covered, total


def compile_spatial_layout_inputs(annotation: Mapping[str, Any], shot_indices: Sequence[int], *,
                                  allow_unannotated: bool = False,
                                  compiled_rows: Sequence[Mapping[str, Any]] | None = None) -> tuple[dict, ...]:
    """仅编译目标镜；缺标注与确认空布局不能合并。

    compiled_rows 来自同一文字 owner 的实际目标输入，连同真实参考用途与 spans 一起消费；
    未提供时从计划或训练标注编译。
    """
    from mspc.data.shot_text import shot_text_with_entity_spans
    from mspc.data.storyboard import ShotLayout
    has_layouts = "layouts" in annotation
    if not has_layouts and not allow_unannotated:
        raise ValueError("spatial conditioning requires explicit layouts; absent is unannotated")
    raw_layouts = annotation["layouts"] if has_layouts else []
    if not isinstance(raw_layouts, list):
        raise ValueError("layouts must be a list; unknown annotations omit the field")
    # 真片块带有质量字段，规划器输出带frame_count；这里只校验两者共用的布局合同。
    checked = [ShotLayout.model_validate(entry).model_dump() for entry in raw_layouts]
    layouts = {entry["shot"]: entry["entities"] for entry in checked}
    names = {entry["name"] for registry in ("characters", "objects", "places") for entry in annotation[registry]}
    if (len(layouts) != len(checked) or any(shot > len(annotation["shots"]) for shot in layouts)
            or any(entity["name"] not in names for entities in layouts.values() for entity in entities)):
        raise ValueError("layouts must use distinct existing shots and registered entities")
    indices = tuple(shot_indices)
    if compiled_rows is not None and len(compiled_rows) != len(indices):
        raise ValueError("compiled layout text rows must cover the actual target shot indices")
    result = []
    for position, index in enumerate(indices):
        if type(index) is not int or not 0 <= index < len(annotation["shots"]):
            raise ValueError("shot index is outside the storyboard")
        if compiled_rows is None:
            compiled = shot_text_with_entity_spans(annotation, index)
        else:
            compiled = compiled_rows[position]
            if (not isinstance(compiled, Mapping) or type(compiled.get("shot_index")) is not int
                    or compiled["shot_index"] != index or not isinstance(compiled.get("text"), str)
                    or not compiled["text"] or not isinstance(compiled.get("entity_spans"), (list, tuple))):
                raise ValueError("compiled layout rows require actual shot order, text and entity spans")
            for span in compiled["entity_spans"]:
                if (not isinstance(span, Mapping) or set(span) != {"name", "start", "end", "source"}
                        or span["name"] not in names or type(span["start"]) is not int or type(span["end"]) is not int
                        or not 0 <= span["start"] < span["end"] <= len(compiled["text"])
                        or not isinstance(span["source"], str) or not span["source"]):
                    raise ValueError("compiled layout spans must identify registered entities inside the actual text")
        result.append({"text": compiled["text"], "entity_spans": compiled["entity_spans"],
                       "entities": layouts.get(index + 1, []),
                       "annotation_status": "provided" if has_layouts else "unannotated"})
    return tuple(result)


def _clean_span(text: str, start: int, end: int, cleaner: Any) -> tuple[str, int, int]:
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        raise ValueError("invalid entity character span")
    cleaned = cleaner(text)
    prefix, entity, suffix = cleaner(text[:start]), cleaner(text[start:end]), cleaner(text[end:])
    if not entity or not cleaned.startswith(prefix) or not cleaned.endswith(suffix):
        raise ValueError("text normalization does not preserve the declared entity span")
    left = len(prefix)
    while left < len(cleaned) and cleaned[left].isspace():
        left += 1
    right = left + len(entity)
    if cleaned[left:right] != entity or cleaned[right:].strip() != suffix:
        raise ValueError("entity span is not separable under the actual umT5 normalization")
    return cleaned, left, right


def align_entity_token_indices(tokenizer: Any, text: str, spans: Sequence[Mapping[str, Any]]) -> tuple[dict[str, tuple[int, ...]], tuple[bool, ...]]:
    """以实际wrapper的ids/mask逐项核对fast tokenizer offsets，拒绝截断和歧义。"""
    raw = tokenizer.tokenizer
    if not getattr(raw, "is_fast", False):
        raise ValueError("spatial alignment requires offsets from the actual fast umT5 tokenizer")
    cleaner = tokenizer._clean if tokenizer.clean else lambda value: value
    cleaned = cleaner(text)
    encoded = raw([cleaned], padding="max_length", truncation=True,
                  max_length=tokenizer.seq_len, add_special_tokens=True,
                  return_offsets_mapping=True, return_special_tokens_mask=True,
                  return_tensors="pt")
    actual_ids, actual_mask = tokenizer(text, return_mask=True, add_special_tokens=True)
    if (encoded["input_ids"].tolist() != actual_ids.tolist()
            or encoded["attention_mask"].tolist() != actual_mask.tolist()):
        raise ValueError("offset tokenization differs from the text encoder's actual input")
    full = raw([cleaned], truncation=False, padding=False, add_special_tokens=True)["input_ids"][0]
    if len(full) > tokenizer.seq_len:
        raise ValueError("spatial text was truncated by umT5")
    offsets = encoded["offset_mapping"][0].tolist()
    mask = tuple(bool(a) and not bool(s) and b > a0 for a, s, (a0, b) in zip(
        encoded["attention_mask"][0].tolist(), encoded["special_tokens_mask"][0].tolist(), offsets))
    selected: dict[str, set[int]] = {}
    owners: dict[int, str] = {}
    for span in spans:
        _, start, end = _clean_span(text, span["start"], span["end"], cleaner)
        indices = [i for i, ((a, b), valid) in enumerate(zip(offsets, mask))
                   if valid and a < end and b > start]
        if not indices:
            raise ValueError(f"entity {span['name']} has no consumed text token")
        for i in indices:
            if i in owners and owners[i] != span["name"]:
                raise ValueError("one text token overlaps two different entity spans")
            owners[i] = span["name"]
        selected.setdefault(span["name"], set()).update(indices)
    return {name: tuple(sorted(indices)) for name, indices in selected.items()}, mask


def encode_spatial_layout_inputs(tokenizer: Any, inputs: Sequence[Mapping[str, Any]], *,
                                prompts: Sequence[str], frame_counts: Sequence[int],
                                options: SpatialLayoutOptions):
    from mspc.models.spatial_layout import (SpatialLayoutAnchor, SpatialLayoutCondition,
                                          SpatialLayoutEntity, SpatialLayoutShot)
    if len(inputs) != len(prompts) or len(inputs) != len(frame_counts):
        raise ValueError("layout, target text and target shot geometry must have equal lengths")
    shots = []
    for row, prompt, count in zip(inputs, prompts, frame_counts):
        if row["text"] != prompt:
            raise ValueError("layout entity spans do not describe the actual target prompt")
        tokens, mask = align_entity_token_indices(tokenizer, prompt, row["entity_spans"])
        entities = []
        for entity in row["entities"]:
            anchors = tuple(SpatialLayoutAnchor(time=a["time"], box=tuple(a["box"]))
                            for a in entity["anchors"] if a["visibility"] == "visible")
            if not anchors:
                continue
            if entity["name"] not in tokens:
                raise ValueError(f"layout entity {entity['name']} is missing from the consumed text")
            entities.append(SpatialLayoutEntity(token_indices=tokens[entity["name"]], anchors=anchors))
        shots.append(SpatialLayoutShot(num_frames=count, entities=tuple(entities), text_token_mask=mask))
    return SpatialLayoutCondition(shots=tuple(shots), strength=options.strength,
                                  layer_indices=options.layer_indices, timestep_range=options.timestep_range,
                                  mode=options.mode, region_dilation=options.region_dilation,
                                  onset_latent_frames=options.onset_latent_frames)
