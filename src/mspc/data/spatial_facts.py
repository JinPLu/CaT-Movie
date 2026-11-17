"""多镜窗口逐镜空间事实抽取（按镜切开）的纯函数：取帧、短属性词、镜内指认、左右、前后、朝向、跨镜认人。

模型调用（SAM 3.1、
Depth Anything V2 Small、RTMW、InsightFace、OSNet-AIN、DINOv2、Qwen3.8-27B）在 ``tools/run/spatial_facts.py``。
本模块只依赖 numpy（指派处再导入 scipy）。

**取帧**：一镜按项目目标帧率（``WAN22_CACHED_SPACETIME_CONTRACT["target_fps"]``，16 fps）取全部帧，跟踪器逐帧跑
（SAM 3 的训练与评测视频是 6–24 fps 的连续帧：SA-Co/VIDEO 平均 84.1 帧 @ 6 fps，SA-Co/VEval 帧目录为 6 fps 与
24 fps）。首帧、末帧事实读这一网格的第 0 帧与最后一帧（与训练取帧「首末帧显式入选」同一网格）；布局事实读
30/50/70% 三帧多数（离开镜头两端，避开邻镜残帧）。

**镜内谁是谁**：通用概念 ``person``/``animal`` 给实例与跟踪号；每个剧本主体一个短属性词概念（:func:`short_phrase`），
逐帧与通用实例按掩码 IoU ≥ 0.5 对齐计票，唯一触发的才算（:func:`phrase_binding`）；其余交本地 VLM 兜底。

**前后**：只在镜内排序。单目深度（相对视差）在实例掩码内取中位数，两人掩码相接处用遮挡关系（:func:`contact_difference`），
中位视差差除以本帧视差范围过间隔阈值定有无，相接处的视差差定方向（:func:`depth_relations`）；不取倒数、不比倍数。

**朝向**：RTMW 全身 133 点（COCO-WholeBody 序），脸点平均置信度低于标定阈值记 ``away``，否则看鼻尖相对脸轮廓中点的横向偏移
（:func:`orientation_from_keypoints`）。

**并入事件日志**：v10 合同的屏幕字段是「首帧所见」（``entities[]`` 只列 ``visible_first``），所以并入用首帧一端的事实，
并入哪些轴由 :func:`merged_screen_layout` 定（左右、前后、只写面朝镜头）；
:func:`replace_screen_facts` 把一窗事件日志里人物与动物的 screen 类事实整组换成工具链的（工具链给不出的 gaze 与
「朝向某人」一并去掉），物体的 screen 事实与其余字段不动。

**跨镜认人**：人脸、人体重识别、DINOv2 三路余弦各自 logistic 标定后按朴素贝叶斯合成对数几率，两镜之间一对一指派，
接受的配对数不超过两镜剧本共同在场人数（:func:`cross_shot_matches`）。同一镜里两个实例必是两个人由一对一保证。
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Hashable

import numpy as np

from mspc.config.experiment import WAN22_CACHED_SPACETIME_CONTRACT

TARGET_FPS = int(WAN22_CACHED_SPACETIME_CONTRACT["target_fps"])
#: 布局帧：一镜 30/50/70%。
LAYOUT_FRACTIONS = (0.3, 0.5, 0.7)
#: 短属性词概念与通用实例的对齐：同一帧掩码 IoU ≥ 0.5。
PHRASE_MATCH_IOU = 0.5
#: 短属性词至少在 3 帧对齐上同一实例才算触发（SAM 3.1 自身确认一条轨迹也要连续 3 帧检出：
#: ``masklet_confirmation_consecutive_det_thresh=3``）。
PHRASE_MIN_FRAMES = 3
#: 唯一触发：指派到的实例上的对齐帧数至少是该短语在任何别的实例上帧数的 2 倍。SAM 3.1 的短属性词常在同性别的
#: 几个人身上一起触发（dev 30 实测 "man in dark suit" 在两人上逐帧都触发），这时交 VLM 兜底。
PHRASE_DOMINANCE = 2.0
#: 端点事实（首帧、末帧）各用该端最近 5 帧（16 fps 下约 0.3 秒）逐帧投票，过半才算。
ENDPOINT_FRAMES = 5
#: horizontal：单人按中心三等分。
THIRD = 1.0 / 3.0
#: 两人左右判不了：框 IoU ≥ 0.5 或中心间距 < 5% 画宽（沿用 Q1 量具）。
SIDE_OVERLAP_IOU = 0.5
SIDE_MIN_GAP_FRACTION = 0.05
#: 遮挡：两掩码膨胀 2 px 后相接的像素至少 10 个才算相接；相接带宽 5 px。
CONTACT_DILATION_PX = 2
CONTACT_MIN_PIXELS = 10
CONTACT_BAND_PX = 5
#: 视差范围取本帧 5–95 分位。
DISPARITY_RANGE_PERCENTILES = (5.0, 95.0)
ORIENTATIONS = ("screen-left", "screen-right", "toward camera", "away")
MIRROR_ORIENTATION = {"screen-left": "screen-right", "screen-right": "screen-left",
                      "toward camera": "toward camera", "away": "away"}

# COCO-WholeBody 133 点序：0 鼻、1 左眼、2 右眼、3 左耳、4 右耳、5 左肩、6 右肩、11 左髋、12 右髋；23–90 为 68 个脸点，
# 脸点 0 与 16 是下颌轮廓两端，30 是鼻尖。
_BODY = (5, 6, 11, 12)
_LEFT_SHOULDER, _RIGHT_SHOULDER = 5, 6
_FACE0 = 23
_JAW_LEFT, _JAW_RIGHT, _NOSE_TIP = _FACE0 + 0, _FACE0 + 16, _FACE0 + 30


# ---------------------------------------------------------------------------
# 取帧
# ---------------------------------------------------------------------------


def layout_positions(frame_count: int) -> tuple[int, int, int]:
    """一镜内 30%、50%、70% 处的帧位置（四舍五入，0 起）；三帧必须互不相同。"""

    if type(frame_count) is not int or frame_count < 3:
        raise ValueError(f"a shot needs at least 3 frames for 30/50/70% sampling, got {frame_count!r}")
    positions = tuple(math.floor(f * (frame_count - 1) + 0.5) for f in LAYOUT_FRACTIONS)
    if len(set(positions)) != len(positions):
        raise ValueError(f"{frame_count} frames do not give three distinct 30/50/70% frames: {positions}")
    return positions  # type: ignore[return-value]


def key_positions(frame_count: int) -> tuple[int, ...]:
    """需要掩码与逐帧测量的帧：首帧、三张布局帧、末帧。"""

    return tuple(sorted({0, *layout_positions(frame_count), frame_count - 1}))


# ---------------------------------------------------------------------------
# 短属性词（SAM 3.1 概念提示只收 32 token，论文：至多一到两个属性）
# ---------------------------------------------------------------------------

_COLOR_WORDS = frozenset({
    "red", "crimson", "scarlet", "maroon", "burgundy", "pink", "magenta", "orange", "rust",
    "yellow", "mustard", "gold", "golden", "green", "olive", "teal", "turquoise", "blue", "navy",
    "cobalt", "indigo", "purple", "violet", "lavender", "lilac", "brown", "tan", "beige", "camel",
    "khaki", "black", "white", "gray", "grey", "silver", "cream", "ivory", "charcoal", "denim",
})
_AGE = {"elderly": "old", "old": "old", "older": "old", "aging": "old", "young": "young", "younger": "young",
        "teenage": "young", "middle-aged": "middle-aged", "adult": None}
_HEADS = ("woman", "man", "girl", "boy", "child", "lady", "gentleman", "female", "male", "person", "soldier", "clown",
          "performer", "dog", "cat", "pig", "horse", "squirrel", "raccoon", "animal", "bird")
_GARMENTS = frozenset({"jacket", "suit", "shirt", "t-shirt", "top", "dress", "coat", "sweater", "hoodie", "blazer",
                       "uniform", "vest", "cardigan", "jersey", "costume", "blouse", "polo", "tank", "overalls", "robe",
                       "cap", "hat", "pants", "trousers", "jeans", "skirt", "sweatshirt", "apron"})
_TOKEN = re.compile(r"[a-z]+(?:-[a-z]+)*")


def short_phrase(attributes: str, kind: str) -> str:
    """剧本外观 -> SAM 3.1 概念提示的短属性词：年龄段＋头名词（没有就按种类 person/animal）＋第一处「颜色词＋衣物名」，
    bald 作发型保留；只用描述里出现的词。"""

    words = _TOKEN.findall(attributes.lower())
    head = next((w for w in words if w in _HEADS), "person" if kind == "person" else "animal")
    head = {"female": "woman", "male": "man", "lady": "woman", "gentleman": "man"}.get(head, head)
    age = next((_AGE[w] for w in words if w in _AGE and _AGE[w]), None)
    bald = "bald" if "bald" in words else None
    garment = None
    for i, w in enumerate(words):
        if w in _COLOR_WORDS or w in ("dark", "light"):
            nxt = next((x for x in words[i + 1:i + 4] if x in _GARMENTS), None)
            if nxt is not None:
                garment = f"{w} {nxt}"
                break
    parts = [x for x in (age, bald, head) if x]
    return " ".join(parts) + (f" in {garment}" if garment else "")


# ---------------------------------------------------------------------------
# 指派与镜内指认
# ---------------------------------------------------------------------------


def assign_one_to_one(gain: Mapping[tuple[Hashable, Hashable], float], *, min_gain: float) -> dict[Hashable, Hashable]:
    """``{(行, 列): 得分}`` 的一对一指派：只在得分 ≥ ``min_gain``（且 ≥ 0）的配对里取总分最大的一组，允许不配满。

    匈牙利算法；每行另配一个得分 0 的「不配」列，缺项与低于下限的配对不许配。"""

    from scipy.optimize import linear_sum_assignment  # 只在指派处用；SAM 3.1、渲染度量环境不带 scipy

    if min_gain < 0:
        raise ValueError("min_gain must be non-negative")
    rows = sorted({r for r, _ in gain}, key=repr)
    cols = sorted({c for _, c in gain}, key=repr)
    if not rows:
        return {}
    forbidden = -1e9
    matrix = np.full((len(rows), len(cols) + len(rows)), forbidden)
    matrix[:, len(cols):] = 0.0
    for (r, c), value in gain.items():
        if not math.isfinite(value):
            raise ValueError(f"non-finite gain for {(r, c)!r}")
        if value >= min_gain:
            matrix[rows.index(r), cols.index(c)] = float(value)
    picked_rows, picked_cols = linear_sum_assignment(matrix, maximize=True)
    return {rows[i]: cols[j] for i, j in zip(picked_rows, picked_cols, strict=True)
            if j < len(cols) and matrix[i, j] > forbidden}


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    return int(np.logical_and(a, b).sum()) / union if union else 0.0


def phrase_frame_match(phrase_masks: Sequence[np.ndarray], instance_masks: Mapping[int, np.ndarray]) -> list[int | None]:
    """一帧里短属性词概念的每个检出 -> 掩码 IoU 最大且 ≥ 0.5 的通用实例跟踪号（没有为 None）。"""

    out: list[int | None] = []
    for mask in phrase_masks:
        best = max(((mask_iou(mask, inst), track) for track, inst in instance_masks.items()), default=(0.0, None))
        out.append(best[1] if best[0] >= PHRASE_MATCH_IOU else None)
    return out


def phrase_binding(votes: Mapping[str, Mapping[str, int]], phrases: Mapping[str, str]) -> dict[str, str]:
    """逐主体短属性词在各通用实例上的对齐帧数 -> 镜内指认 ``{主体: 实例}``。

    - 两个主体短属性词完全相同（文字分不开）的都不用短语；
    - 唯一触发：帧数最多的实例 ≥ :data:`PHRASE_MIN_FRAMES` 帧，且 ≥ :data:`PHRASE_DOMINANCE` × 该短语在任何别的实例上的帧数；
    - 两个主体唯一触发到同一个实例时两人都不认。
    认不出的主体交 VLM 兜底。"""

    counts: dict[str, int] = {}
    for phrase in phrases.values():
        counts[phrase] = counts.get(phrase, 0) + 1
    dominant: dict[str, str] = {}
    for subject, row in votes.items():
        if counts[phrases[subject]] != 1 or not row:
            continue
        ranked = sorted(row.items(), key=lambda item: (-item[1], item[0]))
        best, top = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0
        if top >= PHRASE_MIN_FRAMES and top >= PHRASE_DOMINANCE * runner_up:
            dominant[subject] = best
    claimed: dict[str, int] = {}
    for track in dominant.values():
        claimed[track] = claimed.get(track, 0) + 1
    return {s: t for s, t in dominant.items() if claimed[t] == 1}


# ---------------------------------------------------------------------------
# 左右
# ---------------------------------------------------------------------------


def box_center_x(box: Sequence[float]) -> float:
    return (float(box[0]) + float(box[2])) / 2.0


def horizontal_labels(centers: Mapping[str, float], *, width: int) -> dict[str, str]:
    """主体中心横坐标 -> horizontal。单人三等分；两人以上最左 left、最右 right、其余 center。"""

    if not centers:
        return {}
    if len(centers) == 1:
        ((subject, x),) = centers.items()
        return {subject: "left" if x < THIRD * width else "right" if x > 2 * THIRD * width else "center"}
    order = sorted(centers, key=lambda s: (centers[s], s))
    return {s: "left" if i == 0 else "right" if i == len(order) - 1 else "center" for i, s in enumerate(order)}


def endpoint_positions(frame_count: int) -> dict[str, tuple[int, ...]]:
    """首端与末端各自投票用的帧位置（各 :data:`ENDPOINT_FRAMES` 帧，镜头短于两倍时两端重叠也照取）。"""

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    size = min(ENDPOINT_FRAMES, frame_count)
    return {"first": tuple(range(size)), "last": tuple(range(frame_count - size, frame_count))}


def endpoint_horizontal(frame_centers: Sequence[Mapping[str, float]], *, width: int) -> dict[str, str]:
    """端点几帧里各主体框中心横坐标 -> 逐帧 horizontal 后投票（该主体那帧不可见算一票空票），过半才写。"""

    votes = [horizontal_labels(centers, width=width) for centers in frame_centers]
    subjects = sorted({s for row in votes for s in row})
    out = {}
    for subject in subjects:
        value = majority([row.get(subject) for row in votes])
        if value is not None:
            out[subject] = value
    return out


def majority(values: Sequence[Any]) -> Any:
    """过半数的非 None 答案；没有为 None。"""

    tally: dict[Any, int] = {}
    for value in values:
        if value is not None:
            tally[value] = tally.get(value, 0) + 1
    winners = [v for v, n in tally.items() if n * 2 > len(values)]
    return winners[0] if winners else None


# ---------------------------------------------------------------------------
# 前后（镜内排序 + 遮挡）
# ---------------------------------------------------------------------------


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(radius):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def contact_difference(mask_a: np.ndarray, mask_b: np.ndarray, disparity: np.ndarray) -> float | None:
    """两实例掩码相接时，相接处两侧 5 px 带内中位视差之差（a − b，正＝a 更近）；不相接为 None。

    SAM 掩码是可见区域，遮挡边界上前者的掩码完整、后者在边界处截断，所以相接带的视差差就是遮挡次序，
    且只比较同一处的两侧，不受人物整体姿态与背景的影响。"""

    contact = _dilate(mask_a, CONTACT_DILATION_PX) & _dilate(mask_b, CONTACT_DILATION_PX)
    if int(contact.sum()) < CONTACT_MIN_PIXELS:
        return None
    band = _dilate(contact, CONTACT_BAND_PX)
    side_a, side_b = band & mask_a, band & mask_b
    if not side_a.any() or not side_b.any():
        return None
    return float(np.median(disparity[side_a])) - float(np.median(disparity[side_b]))


def disparity_range(disparity: np.ndarray) -> float:
    low, high = np.percentile(disparity, DISPARITY_RANGE_PERCENTILES)
    return float(high - low)


def depth_relations(medians: Mapping[str, float], contact: Mapping[tuple[str, str], float], *,
                    frame_range: float, gap: float) -> dict[tuple[str, str], str]:
    """一帧里主体两两的前后：``{(a, b): a 在前为 "front"，a 在后为 "behind"}``，判不了的对不出现。

    有没有前后看两人掩码内中位视差之差除以本帧视差范围（5–95 分位），绝对值 ≥ ``gap`` 才算；谁在前：两掩码相接时
    取相接处的视差差（遮挡次序，``contact[(a, b)]`` 是 a − b），不相接取中位视差差。相接带只有几像素，DA-V2 在边缘
    处平滑，相接差的大小偏小（dev 30 实测金标有前后的相接对 68/88 落在 0.3 以下），所以只用它定方向、不用它定有无。"""

    if frame_range <= 0:
        raise ValueError("disparity range must be positive")
    out: dict[tuple[str, str], str] = {}
    subjects = sorted(medians)
    for i, a in enumerate(subjects):
        for b in subjects[i + 1:]:
            diff = (medians[a] - medians[b]) / frame_range
            if abs(diff) < gap:
                continue
            raw = contact.get((a, b))
            sign = diff if raw is None or raw == 0 else raw
            out[(a, b)] = "front" if sign > 0 else "behind"
    return out


def depth_labels(relations: Mapping[tuple[str, str], str], subjects: Sequence[str]) -> dict[str, str]:
    """两两前后 -> depth：在别人前面、自己不在任何人后面的记 foreground；从 foreground 沿「在……后面」能走到的
    （传递）记 background；其余不写。"""

    ahead: dict[str, set[str]] = {s: set() for s in subjects}
    for (a, b), rel in relations.items():
        front, back = (a, b) if rel == "front" else (b, a)
        ahead[front].add(back)
    behind_anyone = {b for front in ahead for b in ahead[front]}
    fronts = [s for s in subjects if ahead[s] and s not in behind_anyone]
    out = {s: "foreground" for s in fronts}
    stack = [b for s in fronts for b in ahead[s]]
    while stack:
        b = stack.pop()
        if b in out:
            continue
        out[b] = "background"
        stack.extend(ahead[b])
    return out


# ---------------------------------------------------------------------------
# 朝向（RTMW 关键点）
# ---------------------------------------------------------------------------


#: 背面肩序：人体左肩在画面上位于右肩左侧、且间距 ≥ 实例框宽 2% 才算（正面时左肩在画面右侧）。
BACK_SHOULDER_MARGIN = 0.02


def orientation_from_keypoints(keypoints: np.ndarray, scores: np.ndarray, *, box_width: float, face_visible: float,
                               face_back: float, lateral: float) -> str | None:
    """一个实例的 133 点（像素坐标）与置信度 -> 屏幕朝向；判不了为 None。

    - ``away``：68 个脸点平均置信度 < ``face_visible``（肩或髋还有一点 ≥ 该阈值，否则判不了）；或者肩序是背面
      （:data:`BACK_SHOULDER_MARGIN`）且脸点平均置信度 < ``face_back``。RTMW 经 rtmlib 给的是 SimCC 原始峰值
      （dev 30 实测正侧脸的脸点均值中位数约 7，背影约 3），不是 0–1 概率；rtmlib 示例画图的 ``kpt_thr=0.43`` 在这个
      尺度上分不出背影，所以阈值在标定半上选。
    - 否则鼻尖相对下颌轮廓两端中点的横向偏移除以轮廓宽：绝对值 < ``lateral`` 为 ``toward camera``，
      正（鼻尖在右）为 ``screen-right``，负为 ``screen-left``。
    """

    keypoints = np.asarray(keypoints, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if keypoints.shape != (133, 2) or scores.shape != (133,):
        raise ValueError(f"expected 133 COCO-WholeBody keypoints, got {keypoints.shape} / {scores.shape}")
    if box_width <= 0:
        raise ValueError("box width must be positive")
    face = float(scores[_FACE0:_FACE0 + 68].mean())
    if face < face_visible:
        return "away" if (scores[list(_BODY)] >= face_visible).any() else None
    back = (keypoints[_LEFT_SHOULDER, 0] - keypoints[_RIGHT_SHOULDER, 0]) / box_width < -BACK_SHOULDER_MARGIN
    if back and face < face_back:
        return "away"
    left, right = keypoints[_JAW_LEFT, 0], keypoints[_JAW_RIGHT, 0]
    width = abs(right - left)
    if width <= 1.0:
        return None
    offset = (keypoints[_NOSE_TIP, 0] - (left + right) / 2.0) / width
    if abs(offset) < lateral:
        return "toward camera"
    return "screen-right" if offset > 0 else "screen-left"


# ---------------------------------------------------------------------------
# 跨镜认人
# ---------------------------------------------------------------------------


def unit_mean(vectors: Sequence[Sequence[float]]) -> list[float] | None:
    """单位化后取均值再单位化；没有向量为 None。"""

    if not vectors:
        return None
    rows = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("zero embedding")
    mean = (rows / norms).mean(axis=0)
    return (mean / np.linalg.norm(mean)).tolist()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


def fit_logistic(xs: Sequence[float], ys: Sequence[int], *, iterations: int = 200) -> tuple[float, float]:
    """一维 logistic 回归 ``P(y=1|x) = σ(a·x + b)``（牛顿法，带 1e-6 岭）。"""

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size == 0 or y.min() == y.max():
        raise ValueError("logistic fit needs both classes")
    design = np.stack([x, np.ones_like(x)], axis=1)
    w = np.zeros(2)
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(design @ w)))
        grad = design.T @ (p - y)
        hess = design.T @ (design * (p * (1 - p))[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        w -= step
        if float(np.abs(step).max()) < 1e-10:
            break
    return float(w[0]), float(w[1])


def fused_log_odds(similarities: Mapping[str, float], calibration: Mapping[str, Any]) -> float | None:
    """各路余弦 -> 合成对数几率（朴素贝叶斯：先验几率加各路似然比）。没有任何一路为 None。

    ``calibration = {"prior": 先验对数几率, "paths": {路: (a, b)}}``，每路 ``a·cos + b`` 是在标定集上单路拟合的后验
    对数几率，减去先验即这一路的似然比。"""

    present = [p for p in similarities if p in calibration["paths"]]
    if not present:
        return None
    prior = float(calibration["prior"])
    total = prior
    for path in present:
        a, b = calibration["paths"][path]
        total += a * similarities[path] + b - prior
    return total


def cross_shot_matches(log_odds: Mapping[tuple[Hashable, Hashable], float], *, limit: int) -> dict[Hashable, Hashable]:
    """两镜实例对的合成对数几率 -> 一对一指派，只接受对数几率 > 0 的配对，最多 ``limit`` 对（两镜剧本共同在场人数）。"""

    if limit < 0:
        raise ValueError("limit must be non-negative")
    positive = {pair: value for pair, value in log_odds.items() if value > 0}
    picked = assign_one_to_one(positive, min_gain=0.0)
    accepted = sorted(((positive[(a, b)], a, b) for a, b in picked.items()), key=lambda t: -t[0])
    return {a: b for _, a, b in accepted[:limit]}


# ---------------------------------------------------------------------------
# 并入事件日志（event_log_v3 的 screen_facts）
# ---------------------------------------------------------------------------

SCREEN_AXES = ("horizontal", "depth", "orientation")
_HORIZONTAL_RANK = {"left": 0, "center": 1, "right": 2}


def screen_fact_rows(shot_index: int, layout: Mapping[str, Mapping[str, str | None]]) -> list[dict[str, Any]]:
    """一镜的 ``{主体: {horizontal, depth, orientation}}`` -> event_log 的 screen 事实（值为 None 的不写）。

    顺序：先 horizontal（左到右），再 depth，再 orientation（同左到右）；措辞由编译器 ``fact_clause`` 决定。"""

    order = sorted(layout, key=lambda s: (_HORIZONTAL_RANK.get(layout[s].get("horizontal") or "", 3), s))
    facts = []
    for axis in SCREEN_AXES:
        for subject in order:
            value = layout[subject].get(axis)
            if value is None:
                continue
            facts.append({"subject": subject, "family": "screen", "predicate_or_axis": axis, "object_or_value": value,
                          "scope": f"screen@{shot_index}", "shot_span": [shot_index, shot_index], "evidence": "observed"})
    return facts


def merged_screen_layout(endpoint_horizontal: Mapping[str, str],
                         frame_facts: Mapping[str, Mapping[str, str | None]]) -> dict[str, dict[str, str]]:
    """并入事件日志的那部分事实（依盲标金标准定：左右、前后、只写面朝镜头）。

    - horizontal：该端 5 帧投票的结果，照并；
    - depth：该端单帧的前后，照并。它只在同帧至少两个已指认主体之间才会算出来，一人没指认上的镜不会有
      （盲金标：首帧「有前后」断言 12/12）；
    - orientation：只并 ``toward camera``（盲金标 17/17）；侧向（0.73）与 away 不并，留在旁车。
    返回 ``{主体: {轴: 值}}``，没有可并的轴的主体不出现。"""

    out: dict[str, dict[str, str]] = {}
    for subject in sorted(set(endpoint_horizontal) | set(frame_facts)):
        row: dict[str, str] = {}
        if subject in endpoint_horizontal:
            row["horizontal"] = endpoint_horizontal[subject]
        facts = frame_facts.get(subject, {})
        if facts.get("depth") is not None:
            row["depth"] = facts["depth"]  # type: ignore[assignment]
        if facts.get("orientation") == "toward camera":
            row["orientation"] = "toward camera"
        if row:
            out[subject] = row
    return out


def replace_screen_facts(event_log: Mapping[str, Any],
                         layouts: Sequence[Mapping[str, Mapping[str, str | None]]]) -> dict[str, Any]:
    """事件日志原始映射 + 逐镜专家工具链事实 -> 新映射：人物与动物的 screen 类事实整组换成工具链的，其余字段原样。

    - 工具链只覆盖人物与动物（SAM 3.1 ``person``/``animal``），只给 horizontal、depth、orientation 四值；这些主体在原日志
      里的 gaze、「toward Subject N」朝向一并去掉（没有专家模型能给，留着就是两套来源混写同一主体）。
    - 物体的 screen 事实原样保留，排在工具链事实之后。
    - 工具链事实里的主体必须是该镜 ``present_subjects`` 里的人物或动物。"""

    kinds = {row["subject_id"]: row["kind"] for row in event_log["subjects"]}
    covered = {s for s, kind in kinds.items() if kind in ("person", "animal")}
    shots = event_log["shots"]
    if len(layouts) != len(shots):
        raise ValueError(f"{len(layouts)} layouts for {len(shots)} shots")
    out = dict(event_log)
    out["shots"] = []
    for index, (shot, layout) in enumerate(zip(shots, layouts, strict=True)):
        outside = set(layout) - (set(shot["present_subjects"]) & covered)
        if outside:
            raise ValueError(f"{shot['shot_id']}: facts name subjects that are not present people or animals: "
                             f"{sorted(outside)}")
        bad = {(s, a, v) for s, row in layout.items() for a, v in row.items()
               if a not in SCREEN_AXES or (v is not None and v not in _SCREEN_VALUES[a])}
        if bad:
            raise ValueError(f"{shot['shot_id']}: unsupported screen values {sorted(bad)}")
        kept = [dict(f) for f in shot["screen_facts"] if f["subject"] not in covered]
        out["shots"].append({**shot, "screen_facts": screen_fact_rows(index, layout) + kept})
    return out


_SCREEN_VALUES = {"horizontal": frozenset(_HORIZONTAL_RANK), "depth": frozenset({"foreground", "background"}),
                  "orientation": frozenset(ORIENTATIONS)}
