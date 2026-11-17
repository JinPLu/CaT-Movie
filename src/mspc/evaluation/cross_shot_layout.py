"""跨镜空间布局判别：oracle 逐镜布局抽取、四道布局题（Q1–Q4）的量具纯函数、N 臂删方位核对、S 臂 ``Screen`` 事实。

本模块只放纯函数；
GPU 检测、解码、判官与出片输入由 ``tools/run/cross_shot_layout.py`` 调。

**取帧**：每镜 30%、50%、70% 三帧（:func:`mspc.data.spatial_facts.layout_positions`），离开镜头两端，避开源片镜头头尾的邻镜残帧。
生成片与真片同一套位置。

**实例、绑定与深度**：

- 实例只用 SAM 3.1：每镜三帧一个会话，文本概念「person」（有动物时另开「animal」），会话内跟踪给出跨三帧一致的
  实例号与掩码。剧本主体描述作概念提示几乎不触发（真片 4 窗 32 镜试跑，多数描述得分为 0），所以描述不作实例来源。
- 两路一致才计：
  1. 镜内：VLM 看中帧上编号的实例框与本镜在场主体的剧本描述，一对一指认（:func:`binding_messages`，可答不在画面）；
  2. 跨镜：DINOv2 掩码画布余弦一对一指派（:func:`cross_shot_mapping`）。
  描述分不开的两人（:func:`separable`）只用第 2 路。
- 前后（oracle）：已由 :mod:`mspc.data.spatial_facts` 的空间事实抽取链取代（镜内中位视差排序＋遮挡定方向，删去 1.5 倍规则）。

**oracle**（§3.2）：``horizontal`` 单人按框中心三等分，两人以上按顺序（最左 left、最右 right、其余 center）；
``depth`` 见上；``orientation`` 由 VLM 判裁剪；不写 ``gaze``。

**四道题**：

- Q1 左右顺序（:func:`pair_relation`）：两人在两镜里的左右顺序「同序／反序」，与计划（真片）比。
- Q2 朝向（:func:`orientation_relation`）：同一人两镜都侧向时同向／反向；正反打单人镜两人朝向相反算视线相接。
- Q3 背景几何：沿用 :mod:`mspc.evaluation.spatial_layout_probe` 的 SIFT＋RANSAC 判据，加叠加层抑制
  （:func:`static_overlay_mask`）。
- Q4 构图（:func:`composition_item`）：同一人两镜中帧头框高对数比 r 与真片同号、真片 |r| ≥ 0.2 才出题。头框沿用
  Z1 构图工具路的量法（Grounding DINO「a head」，取主体框内得分最高的头框）；这是景别，不是前后。

**第 1 步（生成片各臂）**：配对差的固定读法 :func:`difference_reading`、§3.6 预写判定 :func:`step1_verdict`、
一镜文字里谁先被提到 :func:`first_mentioned`、两人谁在前 :func:`depth_front`。逐单元行与聚类 bootstrap 在工具侧。
补充臂 S-first 的首句（左右与前后、外观短语指人）由 :func:`layout_first_sentence` 出。
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np

from mspc.data.event_log import appearance_phrase
from mspc.data.spatial_facts import assign_one_to_one, layout_positions, screen_fact_rows
from mspc.evaluation.spatial_layout_probe import (
    HEAD_PHRASE,
    frame_left,
    majority,
    subject_head,
    wilson_interval,
)

#: 中帧（朝向裁剪、头框、构图）在三帧里的序号。
MID_FRAME = 1
ANIMATE_KINDS = frozenset({"person", "animal"})

#: 朝向值域：schema ``orientation`` 的四值，加判不了。
ORIENTATIONS = ("screen-left", "screen-right", "toward camera", "away")
LATERAL = frozenset({"screen-left", "screen-right"})
INSUFFICIENT = "insufficient_evidence"
ORIENTATION_ANSWERS = (*ORIENTATIONS, INSUFFICIENT)
MIRRORED_ORIENTATION = {"screen-left": "screen-right", "screen-right": "screen-left",
                        "toward camera": "toward camera", "away": "away", INSUFFICIENT: INSUFFICIENT}
#: 朝向裁剪：检测框四边各外扩 20%（§2 Q2 执行细节）。
CROP_EXPAND = 0.2

#: Q4：真片 |r| ≥ 0.2（约一档）出「按计划变」题；|r| < 0.1 不出题；介于两者之间也不出题（方向不够清楚）。
COMPOSITION_TEST_MIN_LOG_RATIO = 0.2
COMPOSITION_SKIP_BELOW_LOG_RATIO = 0.1
#: 生成片「变了」：|r| 至少半档（与 Z1 构图工具路同一阈值）。照抄前镜 |r| ≤ 0.01 必须判「未变」。
COMPOSITION_CHANGED_MIN_LOG_RATIO = 0.1

#: Q3 叠加层抑制（§2）：同一源视频 ≥ 5 个镜的关键帧，灰度逐像素标准差 ≤ 3 视为静止叠加层，膨胀 5 px 后不取特征。
OVERLAY_MIN_SHOTS = 5
OVERLAY_STD_MAX = 3.0
OVERLAY_DILATION_PX = 5

#: 两路一致：跨镜认人用的掩码画布 DINOv2 余弦。
_COLOR_WORDS = frozenset(
    {
        "red", "crimson", "scarlet", "maroon", "burgundy", "pink", "magenta", "orange", "rust",
        "yellow", "mustard", "gold", "golden", "green", "olive", "teal", "turquoise", "blue", "navy",
        "cobalt", "indigo", "purple", "violet", "lavender", "lilac", "brown", "tan", "beige", "camel",
        "khaki", "black", "white", "gray", "grey", "silver", "cream", "ivory", "charcoal", "denim",
    }
)
_WORD = re.compile(r"[a-z]+")


# ---------------------------------------------------------------------------
# 取帧与短语
# ---------------------------------------------------------------------------


def source_offsets(start: int, end: int) -> tuple[int, int, int]:
    """真片一镜（源片帧号半开区间 [start, end)）的 30/50/70% 源帧号。"""

    if type(start) is not int or type(end) is not int or end - start < 3:
        raise ValueError(f"source shot [{start}, {end}) is too short for 30/50/70% sampling")
    return tuple(start + offset for offset in layout_positions(end - start))  # type: ignore[return-value]


def detector_phrase(attributes: str) -> str:
    """剧本主体描述 -> 检测短语：去掉句末标点；句中的 ``.`` 是检测器的短语分隔符，换成逗号。"""

    text = attributes.strip().rstrip(".").strip()
    if not text:
        raise ValueError("subject attributes are empty")
    return re.sub(r"\.\s*", ", ", text)


def animate_subjects(script: Mapping[str, Any], shot: Mapping[str, Any]) -> list[str]:
    """本镜在场的人或动物，按剧本名单顺序。"""

    kinds = {row["subject_id"]: row["kind"] for row in script["subjects"]}
    present = set(shot["present_subjects"])
    return [row["subject_id"] for row in script["subjects"] if row["subject_id"] in present and kinds[row["subject_id"]] in ANIMATE_KINDS]


def _colors(text: str) -> frozenset[str]:
    return frozenset(word for word in _WORD.findall(text.lower()) if word in _COLOR_WORDS)


_MALE = frozenset({"man", "men", "male", "boy", "gentleman", "he", "his", "him", "father", "husband", "son"})
_FEMALE = frozenset({"woman", "women", "female", "girl", "lady", "she", "her", "mother", "wife", "daughter"})


def _gender(text: str) -> str | None:
    words = set(_WORD.findall(text.lower()))
    male, female = bool(words & _MALE), bool(words & _FEMALE)
    return "male" if male and not female else "female" if female and not male else None


def separable(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """两主体的文字描述分得开：种类不同；或男女不同；或同类且两段描述的颜色词都非空、互不包含。分不开的只走跨镜 DINOv2 一路。"""

    if a["kind"] != b["kind"]:
        return True
    ga, gb = _gender(a["attributes"]), _gender(b["attributes"])
    if ga is not None and gb is not None and ga != gb:
        return True
    left, right = _colors(a["attributes"]), _colors(b["attributes"])
    return bool(left) and bool(right) and not (left <= right or right <= left)


# ---------------------------------------------------------------------------
# 绑定：镜内按描述得分一对一指派，跨镜按 DINOv2 一对一指派
# ---------------------------------------------------------------------------


def _unit(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm == 0:
        raise ValueError("zero embedding")
    return array / norm


def mean_embedding(vectors: Sequence[Sequence[float]]) -> list[float] | None:
    """一镜内同一主体各帧掩码画布嵌入的单位化均值；没有为 None。"""

    if not vectors:
        return None
    return _unit(np.mean([_unit(v) for v in vectors], axis=0)).tolist()


def cross_shot_mapping(first: Mapping[str, Sequence[float]], second: Mapping[str, Sequence[float]]) -> dict[str, str]:
    """两镜各自的主体嵌入（键是镜内第 1 路给的身份）-> 跨镜一对一指派 ``{第一镜身份: 第二镜身份}``（第 2 路）。"""

    gain = {(a, b): float(_unit(first[a]) @ _unit(second[b])) for a in first for b in second}
    return {str(a): str(b) for a, b in assign_one_to_one(gain, min_gain=0.0).items()}


# ---------------------------------------------------------------------------
# 一镜的测量（检测结果 -> 每个主体的框、中心、头框、嵌入）
# ---------------------------------------------------------------------------


def instance_key(instance: Mapping[str, Any]) -> str:
    """SAM 3.1 实例在一镜内的身份：概念会话＋跟踪号（不同会话的跟踪号各自从 0 数）。"""

    return f"{instance['concept']}#{instance['track_id']}"


def bound_tracks(frames: Sequence[Mapping[str, Any]], binding: Mapping[str, str | None],
                 embeddings: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """一镜三帧的 SAM 3.1 实例 + 镜内绑定 ``{主体: 实例身份或 None}`` -> 每帧每个主体的框与掩码键、嵌入均值。

    ``frames[k]["instances"]`` 每项带 ``concept``、``track_id``、``box``、``mask_key``；``embeddings`` 以掩码键索引。
    """

    boxes: list[dict[str, list[float] | None]] = []
    keys: list[dict[str, str | None]] = []
    vectors: dict[str, list[Sequence[float]]] = {subject: [] for subject in binding}
    for frame in frames:
        by_key = {instance_key(inst): inst for inst in frame["instances"]}
        row_boxes: dict[str, list[float] | None] = {}
        row_keys: dict[str, str | None] = {}
        for subject, key in binding.items():
            inst = None if key is None else by_key.get(key)
            row_boxes[subject] = None if inst is None else [float(v) for v in inst["box"]]
            row_keys[subject] = None if inst is None else inst["mask_key"]
            if inst is not None and inst["mask_key"] in embeddings:
                vectors[subject].append(embeddings[inst["mask_key"]])
        boxes.append(row_boxes)
        keys.append(row_keys)
    return {"boxes": boxes, "mask_keys": keys,
            "embeddings": {subject: mean_embedding(vs) for subject, vs in vectors.items()}}


def pair_left(tracks: Mapping[str, Any], a: str, b: str, *, width: int) -> str | None:
    """一镜里两主体谁在左：逐帧 :func:`frame_left`，三帧多数；判不了为 None。"""

    votes = [frame_left(frame[a], frame[b], width=width) for frame in tracks["boxes"]]
    winner = majority(votes)
    return None if winner is None else (a, b)[winner]


def median_center(tracks: Mapping[str, Any], subject: str) -> float | None:
    """一个主体三帧框中心横坐标的中位数；至少两帧有框，否则 None。"""

    xs = [(frame[subject][0] + frame[subject][2]) / 2 for frame in tracks["boxes"] if frame[subject] is not None]
    return float(np.median(xs)) if len(xs) >= 2 else None


# ---------------------------------------------------------------------------
# oracle：horizontal / depth / orientation -> Screen 事实
# ---------------------------------------------------------------------------


def with_screen_facts(event_log: Mapping[str, Any], layouts: Sequence[Mapping[str, Mapping[str, str]]], *,
                      axes: Sequence[str]) -> dict[str, Any]:
    """事件日志原始映射 + 逐镜 oracle 布局 -> 填好 ``screen_facts`` 的新映射（只填 screen，机位字段不动）。"""

    shots = event_log["shots"]
    if len(layouts) != len(shots):
        raise ValueError(f"{len(layouts)} layouts for {len(shots)} shots")
    unknown_axes = set(axes) - {"horizontal", "depth", "orientation"}
    if unknown_axes:
        raise ValueError(f"unsupported screen axes {sorted(unknown_axes)}")
    out = dict(event_log)
    out["shots"] = []
    for index, (shot, layout) in enumerate(zip(shots, layouts, strict=True)):
        if shot["screen_facts"]:
            raise ValueError(f"{shot['shot_id']} already carries screen facts")
        missing = set(layout) - set(shot["visible_entities"])
        if missing:
            raise ValueError(f"{shot['shot_id']}: layout names subjects not visible in the shot: {sorted(missing)}")
        kept = {subject: {axis: value for axis, value in row.items() if axis in axes} for subject, row in layout.items()}
        out["shots"].append({**shot, "screen_facts": screen_fact_rows(index, kept)})
    return out


# ---------------------------------------------------------------------------
# Q1 左右顺序
# ---------------------------------------------------------------------------


def flip_relation(relation: str) -> str:
    return {"same": "reversed", "reversed": "same"}[relation]


def pair_relation(*, left_first: str | None, left_second: str | None, a: str, b: str, separable_text: bool,
                  mapping: Mapping[str, str] | None) -> dict[str, Any]:
    """两镜里同一对主体 (a, b) 的左右顺序关系。

    - ``left_first`` / ``left_second``：各镜第 1 路认出的左边那位；任一为 None 判不了。
    - ``mapping``：第 2 路跨镜指派（第一镜身份 -> 第二镜身份）；缺任一人为 None。
    - 文字分得开：两路一致（第 2 路是恒等）才计，关系按第 1 路；不一致判不了（``binding_disagree``）。
    - 文字分不开：只用第 2 路，第 2 路交换两人时关系取反。
    """

    if left_first is None or left_second is None:
        return {"relation": None, "reason": "order_undecided"}
    if mapping is None or a not in mapping or b not in mapping:
        return {"relation": None, "reason": "no_embedding"}
    by_text = "same" if left_first == left_second else "reversed"
    identity = mapping[a] == a and mapping[b] == b
    swapped = mapping[a] == b and mapping[b] == a
    if not (identity or swapped):
        return {"relation": None, "reason": "mapping_outside_pair"}
    if separable_text:
        if not identity:
            return {"relation": None, "reason": "binding_disagree"}
        return {"relation": by_text, "reason": "both_paths"}
    return {"relation": by_text if identity else flip_relation(by_text), "reason": "embedding_only"}


# ---------------------------------------------------------------------------
# Q2 朝向
# ---------------------------------------------------------------------------


def expand_box(box: Sequence[float], *, width: int, height: int, fraction: float = CROP_EXPAND) -> list[int]:
    """检测框四边各外扩 ``fraction`` 倍框宽／框高，裁到画面内（整数像素）。"""

    x0, y0, x1, y1 = (float(v) for v in box)
    dx, dy = fraction * (x1 - x0), fraction * (y1 - y0)
    out = [max(0, math.floor(x0 - dx)), max(0, math.floor(y0 - dy)),
           min(width, math.ceil(x1 + dx)), min(height, math.ceil(y1 + dy))]
    if out[0] >= out[2] or out[1] >= out[3]:
        raise ValueError(f"degenerate crop from box {box}")
    return out


def crop_flipped(seed: int, instance_id: str) -> bool:
    """镜像随机化（§2 Q2）：每个实例按种子与实例号确定是否先水平翻转再送判官（约 50%）。"""

    import hashlib

    digest = hashlib.sha256(f"{seed}:{instance_id}".encode()).digest()
    return digest[0] % 2 == 1


def unflip(answer: str, flipped: bool) -> str:
    """判官对（可能翻转过的）裁剪的回答 -> 原画面坐标下的朝向。"""

    if answer not in MIRRORED_ORIENTATION:
        raise ValueError(f"unknown orientation answer {answer!r}")
    return MIRRORED_ORIENTATION[answer] if flipped else answer


def orientation_relation(first: str | None, second: str | None) -> str | None:
    """同一人两镜（或正反打两人）的朝向关系：两边都侧向才判，同向 ``same``、反向 ``opposite``。"""

    if first not in LATERAL or second not in LATERAL:
        return None
    return "same" if first == second else "opposite"


# ---------------------------------------------------------------------------
# Q4 构图
# ---------------------------------------------------------------------------


def head_log_ratio(first: Sequence[float] | None, second: Sequence[float] | None) -> float | None:
    """同一人两镜中帧头框高的对数比 log(h2/h1)；任一镜没有头框为 None。"""

    if first is None or second is None:
        return None
    h1, h2 = first[3] - first[1], second[3] - second[1]
    if h1 <= 0 or h2 <= 0:
        return None
    return math.log(h2 / h1)


def composition_item(real_ratio: float | None) -> str | None:
    """真片对数比 -> 这一对出不出「按计划变」题：``tighter`` / ``looser`` 出题，``None`` 不出（|r| < 0.2 或判不了）。"""

    if real_ratio is None or abs(real_ratio) < COMPOSITION_TEST_MIN_LOG_RATIO:
        return None
    return "tighter" if real_ratio > 0 else "looser"


def composition_holds(planned: str, generated_ratio: float | None) -> bool | None:
    """生成片是否按计划变：与真片同号且 |r| ≥ 半档；照抄（|r| ≈ 0）判「未变」。没有头框为 None。"""

    if planned not in ("tighter", "looser"):
        raise ValueError(f"planned direction must be tighter or looser, got {planned!r}")
    if generated_ratio is None:
        return None
    if abs(generated_ratio) < COMPOSITION_CHANGED_MIN_LOG_RATIO:
        return False
    return (generated_ratio > 0) == (planned == "tighter")


def composition_tool_direction(real_ratio: float | None) -> str | None:
    """Q4 校准：工具在真片上读出的方向（|r| ≥ 0.2 才给方向；0.1–0.2 记 ``same``-以外的判不了；< 0.1 记 ``same``）。"""

    if real_ratio is None:
        return None
    if abs(real_ratio) >= COMPOSITION_TEST_MIN_LOG_RATIO:
        return "tighter" if real_ratio > 0 else "looser"
    if abs(real_ratio) < COMPOSITION_SKIP_BELOW_LOG_RATIO:
        return "same"
    return None


# ---------------------------------------------------------------------------
# Q3 叠加层抑制
# ---------------------------------------------------------------------------


def static_overlay_mask(gray_frames: Sequence[np.ndarray]) -> np.ndarray:
    """同一源视频 ≥ 5 个镜的关键帧（灰度、同尺寸）-> 静止叠加层掩码（True 为叠加层，已膨胀 5 px）。"""

    import cv2

    if len(gray_frames) < OVERLAY_MIN_SHOTS:
        raise ValueError(f"overlay estimation needs >= {OVERLAY_MIN_SHOTS} keyframes from distinct shots, got {len(gray_frames)}")
    shapes = {frame.shape for frame in gray_frames}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError(f"overlay keyframes must be same-size grayscale images, got {sorted(shapes)}")
    stack = np.stack([frame.astype(np.float32) for frame in gray_frames])
    static = (stack.std(axis=0) <= OVERLAY_STD_MAX).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * OVERLAY_DILATION_PX + 1, 2 * OVERLAY_DILATION_PX + 1))
    return cv2.dilate(static, kernel) > 0


def suppress_overlay(background: np.ndarray, overlay: np.ndarray | None) -> np.ndarray:
    """背景取特征掩码（255 可取）再去掉叠加层像素。"""

    if overlay is None:
        return background
    if overlay.shape != background.shape:
        raise ValueError(f"overlay mask {overlay.shape} does not match background mask {background.shape}")
    return np.where(overlay, 0, background).astype(np.uint8)


# ---------------------------------------------------------------------------
# N 臂：删画面方位短语的核对
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[A-Za-z0-9']+")


def is_pure_deletion(original: str, edited: str) -> bool:
    """``edited`` 是 ``original`` 的纯删除：字符层与词层都是子序列（删词不能拼出新词）。"""

    def subsequence(small: Sequence[Any], big: Sequence[Any]) -> bool:
        iterator = iter(big)
        return all(any(item == other for other in iterator) for item in small)

    return subsequence(edited, original) and subsequence(_TOKEN.findall(edited), _TOKEN.findall(original))


def deleted_words(original: str, edited: str) -> list[str]:
    """纯删除下被删掉的词（按原文顺序，贪心对齐）。调用方先保证 :func:`is_pure_deletion`。"""

    kept = _TOKEN.findall(edited)
    removed, index = [], 0
    for word in _TOKEN.findall(original):
        if index < len(kept) and kept[index] == word:
            index += 1
        else:
            removed.append(word)
    if index != len(kept):
        raise ValueError("edited text is not a word-level deletion of the original")
    return removed


# ---------------------------------------------------------------------------
# VLM 提示：镜内绑定、朝向、删方位短语
# ---------------------------------------------------------------------------

BINDING_PROMPT = """The image is one frame of a film shot. Every person or animal the segmenter found is outlined and numbered (box 1, box 2, ...). The script says these subjects are present in this shot:
{subjects}

For each listed subject, give the number of the box that shows that subject, or 0 if none of the boxes shows them (they may be off screen, hidden, or not segmented). Use each box for at most one subject. Some boxes may show people who are not in the list; leave them unmatched. Decide from appearance (face, hair, clothing, build, species), not from position. Write the evidence first, then the box number.

Return JSON {{"matches": [{{"subject": "Subject N", "evidence": "...", "box": <int>}}, ...]}} with one entry per listed subject, in the listed order."""


def binding_prompt(subjects: Sequence[tuple[str, str]]) -> str:
    """镜内绑定提示词：``subjects`` 为 (主体编号, 剧本描述)，按名单顺序。"""

    lines = "\n".join(f"- {subject_id}: {description}" for subject_id, description in subjects)
    return BINDING_PROMPT.format(subjects=lines)


def binding_schema(subject_ids: Sequence[str], box_count: int) -> dict[str, Any]:
    item = {"type": "object", "additionalProperties": False, "required": ["subject", "evidence", "box"],
            "properties": {"subject": {"type": "string", "enum": list(subject_ids)}, "evidence": {"type": "string"},
                           "box": {"type": "integer", "enum": list(range(box_count + 1))}}}
    schema = {"type": "object", "additionalProperties": False, "required": ["matches"],
              "properties": {"matches": {"type": "array", "items": item, "minItems": len(subject_ids),
                                         "maxItems": len(subject_ids)}}}
    return {"type": "json_schema", "json_schema": {"name": "subject_binding", "strict": True, "schema": schema}}


def parse_binding(text: str, subject_ids: Sequence[str], box_count: int) -> dict[str, int | None]:
    """绑定回答 -> ``{主体: 框号（1 起）或 None}``。主体须逐一、按序；同一框给了两人时，两人都记 None（判不了）。"""

    import json

    rows = json.loads(text)["matches"]
    if [row["subject"] for row in rows] != list(subject_ids):
        raise ValueError(f"binding answer lists {[row['subject'] for row in rows]}, expected {list(subject_ids)}")
    picked = {row["subject"]: (None if row["box"] == 0 else int(row["box"])) for row in rows}
    if any(box is not None and not 1 <= box <= box_count for box in picked.values()):
        raise ValueError(f"binding answer names a box outside 1..{box_count}: {picked}")
    used: dict[int, int] = {}
    for box in picked.values():
        if box is not None:
            used[box] = used.get(box, 0) + 1
    return {subject: (None if box is None or used[box] > 1 else box) for subject, box in picked.items()}


ORIENTATION_PROMPT = """Each image is a crop around one person or animal from a film frame; a thin red rectangle marks the subject. For every image, report which way the marked subject faces as seen on the screen. Use image directions (the viewer's left and right), not the subject's own left and right.

- "screen-left": the subject's face (or, if the face is hidden, the front of the body) points toward the left edge of the image: a profile or a three-quarter view turned to the image's left.
- "screen-right": the same toward the right edge of the image.
- "toward camera": the face points at the camera, roughly frontal; both sides of the face are about equally visible.
- "away": the back of the head or the back of the body is toward the camera.
- "insufficient_evidence": the direction cannot be seen (too small, blurred, cut off, occluded), or the rectangle does not mark one subject.

Judge each image on its own. Write the evidence first (which cheek, ear, nose direction or body side is visible), then the answer.

Return JSON {"images": [{"image": 1, "evidence": "...", "orientation": "..."}, ...]} with one entry per image, in order."""


def orientation_schema(count: int) -> dict[str, Any]:
    item = {"type": "object", "additionalProperties": False, "required": ["image", "evidence", "orientation"],
            "properties": {"image": {"type": "integer"}, "evidence": {"type": "string"},
                           "orientation": {"type": "string", "enum": list(ORIENTATION_ANSWERS)}}}
    schema = {"type": "object", "additionalProperties": False, "required": ["images"],
              "properties": {"images": {"type": "array", "items": item, "minItems": count, "maxItems": count}}}
    return {"type": "json_schema", "json_schema": {"name": "subject_orientation", "strict": True, "schema": schema}}


def parse_orientation(text: str, count: int) -> list[str]:
    """朝向回答 -> 按图序的答案；图号须是 1..count 且按序。"""

    import json

    rows = json.loads(text)["images"]
    if [row["image"] for row in rows] != list(range(1, count + 1)):
        raise ValueError(f"orientation answer numbers {[row['image'] for row in rows]}, expected 1..{count}")
    answers = [row["orientation"] for row in rows]
    if any(answer not in ORIENTATION_ANSWERS for answer in answers):
        raise ValueError(f"orientation answer outside the value set: {answers}")
    return answers


STRIP_PROMPT = """You edit one shot description from a film script. Delete every phrase that says where someone or something is in the frame, or which way they face or move on screen. Examples: "on the left", "on the right side of the frame", "in the center of the frame", "in the foreground", "in the background", "frame left", "facing right", "facing the camera", "with his back to the camera", "looking to her left", "turns to the right", "in profile", "over-the-shoulder", "exits frame right".

Keep everything else exactly as written: actions, expressions, objects, clothing, the camera's shot size, angle and movement, and relations between people, animals, objects and places (for example "facing Subject 1", "looks at Subject 2", "behind the counter", "next to Subject 3", "across the table"). Keep all IDs such as "Subject 2" and "Location 1".

Only delete words. Never add, replace or reorder a word. After a deletion, also delete a comma or "and" that would be left dangling, so the sentence still reads. If nothing needs deleting, return the text unchanged.

Return JSON {"edited": "<the text after deletion>", "deleted": ["<each deleted phrase>", ...]}.

Text:
"""


def strip_prompt(text: str) -> str:
    return STRIP_PROMPT + text


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------


def rate(num: int, den: int) -> dict[str, Any]:
    interval = wilson_interval(num, den)
    return {"num": num, "den": den, "rate": None if den == 0 else num / den,
            "wilson95": None if interval is None else [round(interval[0], 4), round(interval[1], 4)]}


def subject_pairs(subjects: Sequence[str]) -> list[tuple[str, str]]:
    return list(combinations(subjects, 2))


# ---------------------------------------------------------------------------
# 第 1 步（生成片各臂）：固定读法、预写判定、提及顺序、前后
# ---------------------------------------------------------------------------

#: 配对差的读法阈值（沿用带宽阶梯已冻结的读法）。
READING_MARGIN = 0.05
READING_MUCH_BETTER = "≫"
READING_ABOUT_EQUAL = "≈"
READING_CLEARLY_WORSE = "明显更差"
READING_UNDECIDED = "未决"


def difference_reading(point: float, low: float, high: float) -> str:
    """差（前者减后者）与其 95% 区间 -> 读法：``≫`` 差 ≥ 0.05 且区间不含 0；``≈`` 区间含 0 且 |差| < 0.05；
    ``明显更差`` 反向差 ≥ 0.05 且区间不含 0；其余 ``未决``。"""

    if point >= READING_MARGIN and low > 0:
        return READING_MUCH_BETTER
    if point <= -READING_MARGIN and high < 0:
        return READING_CLEARLY_WORSE
    if low <= 0 <= high and abs(point) < READING_MARGIN:
        return READING_ABOUT_EQUAL
    return READING_UNDECIDED


def step1_verdict(main: str, against_ceiling: str, guardrails: Sequence[str]) -> str:
    """预写判定。``main``：S−N 的读法；``against_ceiling``：S 对真片天花板 R 的读法；
    ``guardrails``：各护栏 S 对 N 的读法。"""

    readings = {READING_MUCH_BETTER, READING_ABOUT_EQUAL, READING_CLEARLY_WORSE, READING_UNDECIDED}
    if {main, against_ceiling, *guardrails} - readings:
        raise ValueError(f"unknown reading among {main!r}, {against_ceiling!r}, {list(guardrails)!r}")
    if main == READING_ABOUT_EQUAL:
        return "文字不够"
    if main == READING_MUCH_BETTER:
        if against_ceiling == READING_CLEARLY_WORSE:
            return "部分"
        if READING_CLEARLY_WORSE not in guardrails:
            return "文字够"
    return "未决"


_SUBJECT_MENTION = re.compile(r"\bSubject (\d+)\b")


def first_mentioned(prompt: str, a: str, b: str) -> str | None:
    """一镜文字里两主体谁先被提到（``Subject N`` 首次出现的位置）；有一个没出现为 None。"""

    seen: dict[str, int] = {}
    for match in _SUBJECT_MENTION.finditer(prompt):
        seen.setdefault(f"Subject {match.group(1)}", match.start())
    if a not in seen or b not in seen:
        return None
    return a if seen[a] < seen[b] else b


_HORIZONTAL_ORDER = {"left": 0, "center": 1, "right": 2}


def layout_first_sentence(layout: Mapping[str, Mapping[str, Any]], attributes: Mapping[str, str]) -> str:
    """S-first 臂的首句：把一镜 oracle 布局的左右（有前后时连同前后）写成一句，按从左到右的顺序、用外观短语指人，
    不用主体编号。方位措辞沿用编译器 ``fact_clause``（``on the left side of the frame`` 等）。没有可写的主体返回空串。"""

    rows = [(subject, row) for subject, row in layout.items() if row.get("horizontal") is not None]
    if not rows:
        return ""
    for subject, row in rows:
        if row["horizontal"] not in _HORIZONTAL_ORDER:
            raise ValueError(f"{subject}: unsupported horizontal value {row['horizontal']!r}")
        if row.get("depth") not in (None, "foreground", "background"):
            raise ValueError(f"{subject}: unsupported depth value {row['depth']!r}")
    order = {subject: index for index, subject in enumerate(layout)}
    rows.sort(key=lambda item: (_HORIZONTAL_ORDER[item[1]["horizontal"]], order[item[0]]))
    clauses = []
    for subject, row in rows:
        where = "in the center of the frame" if row["horizontal"] == "center" else f"on the {row['horizontal']} side of the frame"
        depth = f", in the {row['depth']}," if row.get("depth") is not None else ""
        clauses.append(f"{where}{depth} is {appearance_phrase(attributes[subject])}")
    text = "; ".join(clauses) + "."
    return text[0].upper() + text[1:]


def numbered_space_references(prompt: str) -> frozenset[str] | None:
    """编译器写在目标行开头的空间句里，用 ``Subject k`` 编号指代（短属性词对不上时的退路）的主体；没有空间句为 None。"""

    from mspc.data.event_log import SPACE_SENTENCE

    match = SPACE_SENTENCE.match(prompt)
    if match is None:
        return None
    return frozenset(f"Subject {number}" for number in _SUBJECT_MENTION.findall(match.group(0)))


def depth_front(layout: Mapping[str, Mapping[str, Any]], a: str, b: str) -> str | None:
    """布局里两主体谁在前：一方 ``foreground``、另一方 ``background`` 才算；同一平面或没写为 None。"""

    da, db = (layout.get(subject, {}).get("depth") for subject in (a, b))
    if da == "foreground" and db == "background":
        return a
    if da == "background" and db == "foreground":
        return b
    return None


__all__ = [
    "BINDING_PROMPT",
    "ORIENTATION_PROMPT",
    "STRIP_PROMPT",
    "binding_prompt",
    "binding_schema",
    "orientation_schema",
    "parse_binding",
    "parse_orientation",
    "strip_prompt",
    "ANIMATE_KINDS",
    "COMPOSITION_TEST_MIN_LOG_RATIO",
    "HEAD_PHRASE",
    "INSUFFICIENT",
    "LATERAL",
    "MIRRORED_ORIENTATION",
    "ORIENTATIONS",
    "ORIENTATION_ANSWERS",
    "animate_subjects",
    "appearance_phrase",
    "bound_tracks",
    "composition_holds",
    "composition_item",
    "composition_tool_direction",
    "crop_flipped",
    "cross_shot_mapping",
    "deleted_words",
    "depth_front",
    "detector_phrase",
    "difference_reading",
    "first_mentioned",
    "step1_verdict",
    "expand_box",
    "flip_relation",
    "head_log_ratio",
    "is_pure_deletion",
    "layout_first_sentence",
    "mean_embedding",
    "median_center",
    "numbered_space_references",
    "orientation_relation",
    "pair_left",
    "pair_relation",
    "rate",
    "separable",
    "instance_key",
    "source_offsets",
    "static_overlay_mask",
    "subject_pairs",
    "suppress_overlay",
    "unflip",
    "with_screen_facts",
]
