"""空间尺子：原生 Wan 的文字方位服从探针（Z1），与三道空间题的合成坏例验收（Z2）。

本模块只放题面编法、计量与造臂的纯函数；GPU 检测、出片、判官都由 ``tools/run/run_spatial_probe.py`` 调。

**检测**复用身份列同一条栈：``renderer_qualification_official_backend._DinoRoiEngine.grounded_regions``
（Grounding DINO tiny，冻结阈值 box 0.25 / text 0.20，多短语一次检测、每个框带逐短语得分），SAM 掩码给
背景抠图。每个框归给逐短语得分最高的那条主体描述，得分还须高于框阈值（与 ``subject_gated_consistency``
选参考框同一条线）；一个主体取归给它的框里得分最高的那个。短语是 v2 镜头 prompt 里 ``Subject N (…)``
括号中的短描述（:func:`subject_phrases`）：渲染器看到的就是这句，衣着颜色也都在里面；身份列用的
12 词截断长描述常把颜色截掉，两个人分不开。

**画面方位**（:func:`shot_side`）：每镜 50%、75%、100% 处三帧；在场主体两两成对，逐帧按框中心横坐标定
谁在左；两框 IoU ≥ 0.5 或中心横向间距小于画宽 5% 记这一帧判不了；三帧里至少两帧给出同一答案才算这一对
可判。

**景别**：中间帧主体框高占画高的比例（:func:`height_ratio`）。第一项读数显示它分不开特写与远景，构图题相邻
一档改用头框高比例（:func:`size_step_holds`）。

**地标布局**（:func:`pair_inliers`、:func:`layout_verdicts`）：每镜首、中、末三帧（身份列同一关键帧规则），
抠掉全部检出主体（SAM 掩码并集再膨胀）后取 OpenCV SIFT，Lowe 比值检验 0.75，匹配按距离取一一对应
（:func:`one_to_one`），RANSAC 估基础矩阵（:func:`fundamental_inliers`：八点法最小样本、到极线距离两边取大
小于 3 px 记内点、置信 0.99 自适应停止、固定种子），记内点数；一对镜取 3×3 个关键帧对里的最大值。OpenCV 4.13
自带的两类估计器在本项目会遇到的输入上都会断言崩溃，见 :func:`fundamental_inliers`。判「相容」须 n ≥ 15 且
n ≥ 2 × 对照组中位数；对照组是同一条视频里地点不同的相邻镜对，这条视频没有就取同一臂里其他故事的这类镜对（合并）。

**合成坏例**（:func:`mirror_target` 等）：每条故事只动一个靶镜，靶镜按预注册取第一个合格的镜。
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from itertools import combinations
from statistics import median
from typing import Any

import numpy as np

from mspc.benchmark.script_state_bench import NON_EVENT_SHOT_FRAMES, roster_prompt
from mspc.data.case_sampling import sample_case_ids
from mspc.data.cut_type_labels import SHOT_SIZES
from mspc.inference.continuity_dev_32 import (
    SPATIAL_TEXT_OBEDIENCE_PROBE_SCHEMA as PROBE_SCHEMA,
)

PROBE_SIDE = "side"
PROBE_SIZE = "size"

#: 方位探针的镜头开头：两主体左右站开要看得到全身，取中远景；机位一律平视。
SIDE_SHOT_SIZE = "Medium wide shot"
PROBE_CAMERA_ANGLE = "eye-level shot"
#: 景别探针的两档：「特写」与「远景」，取 v2 词表原词。
SIZE_PAIR = ("Close-up shot", "Wide shot")
PROBE_FRAME_COUNT = NON_EVENT_SHOT_FRAMES
SIZE_ACTION = "stands still and looks around"
SIDES = ("left", "right")
ANIMATE_KINDS = frozenset({"person", "animal"})
#: 「衣着颜色不同」的粗判词表：同类两主体的描述里都有颜色词、且两组颜色词不相交，才算好区分。
COLOR_WORDS = frozenset(
    {
        "red", "crimson", "scarlet", "maroon", "burgundy", "pink", "magenta", "orange", "rust",
        "yellow", "mustard", "gold", "golden", "green", "olive", "teal", "turquoise", "blue", "navy",
        "cobalt", "indigo", "purple", "violet", "lavender", "lilac", "brown", "tan", "beige", "camel",
        "khaki", "black", "white", "gray", "grey", "silver", "cream", "ivory", "charcoal", "denim",
    }
)

#: 方位帧：每镜 50%、75%、100% 处（预注册）。
SIDE_FRACTIONS = (0.5, 0.75, 1.0)
SIDE_OVERLAP_IOU = 0.5
SIDE_MIN_GAP_FRACTION = 0.05
#: 一个框算某主体的，它在该主体短语上的得分须高于检出器的框阈值（冻结档 0.25）。
LAYOUT_MIN_INLIERS = 15
LAYOUT_CONTROL_FACTOR = 2.0
SIFT_RATIO = 0.75
RANSAC_THRESHOLD_PX = 3.0
RANSAC_CONFIDENCE = 0.99
#: RANSAC 的迭代上限与随机种子：同一组匹配永远得到同一个内点数。
RANSAC_MAX_ITERATIONS = 2000
RANSAC_SEED = 0
#: 基础矩阵至少要 8 对匹配；不到 8 对内点数记 0。
MIN_FUNDAMENTAL_MATCHES = 8
#: 一一对应时两个关键点相距不到这么多像素算同一个点（小于 RANSAC 的 3 px 内点阈值）。
SAME_POINT_RADIUS_PX = 2.0
#: 主体掩码并集向外膨胀的像素：SIFT 不在主体轮廓上取点（轮廓上的点跟着人走，会把照抄当成同景）。
MASK_DILATION_PX = 9

ARM_MIRROR = "mirror"
ARM_COPY = "copy_previous"
ARM_SWAP = "swap_background"
ARMS = (ARM_MIRROR, ARM_COPY, ARM_SWAP)

_PAREN = re.compile(r"\bSubject (\d+) \(([^()]*)\)")
_WORD = re.compile(r"[a-z]+")


# ---------------------------------------------------------------------------
# 题面
# ---------------------------------------------------------------------------


def subject_phrases(case: Mapping[str, Any]) -> dict[str, str]:
    """v2 一条故事 -> 每个主体在镜头 prompt 里的括号短描述（同一主体各镜必须一致，且主体之间互不相同）。"""

    found: dict[str, set[str]] = {}
    for shot in case["shots"]:
        for number, text in _PAREN.findall(shot["prompt"]):
            found.setdefault(f"Subject {number}", set()).add(text.strip())
    phrases: dict[str, str] = {}
    for subject in case["subjects"]:
        subject_id = subject["subject_id"]
        texts = found.get(subject_id, set())
        if len(texts) != 1:
            raise ValueError(f"{case['case_id']}: {subject_id} has {len(texts)} short descriptions {sorted(texts)}")
        phrases[subject_id] = texts.pop()
    if len(set(phrases.values())) != len(phrases):
        raise ValueError(f"{case['case_id']}: two subjects share a short description {phrases}")
    return phrases


def _colors(text: str) -> frozenset[str]:
    return frozenset(word for word in _WORD.findall(text.lower()) if word in COLOR_WORDS)


def distinguishable(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """两主体外观好区分：种类不同；或同类且两段描述的颜色词都非空、互不相交。"""

    if a["kind"] != b["kind"]:
        return True
    left, right = _colors(a["attributes"]), _colors(b["attributes"])
    return bool(left) and bool(right) and not (left & right)


def _side_pair(case: Mapping[str, Any]) -> tuple[str, str, str] | None:
    """故事里第一对同镜出现、都会动（人或动物）、外观好区分的主体，及它们第一次同镜的地点。"""

    subjects = {row["subject_id"]: row for row in case["subjects"]}
    order = [row["subject_id"] for row in case["subjects"]]
    for a, b in combinations(order, 2):
        if subjects[a]["kind"] not in ANIMATE_KINDS or subjects[b]["kind"] not in ANIMATE_KINDS:
            continue
        if not distinguishable(subjects[a], subjects[b]):
            continue
        together = [shot for shot in case["shots"] if a in shot["present_subjects"] and b in shot["present_subjects"]]
        if together:
            return a, b, together[0]["location_id"]
    return None


def _size_subject(case: Mapping[str, Any]) -> tuple[str, str] | None:
    """故事里第一个会动的主体，及它第一次出场的地点。"""

    for row in case["subjects"]:
        if row["kind"] in ANIMATE_KINDS:
            shot = next(shot for shot in case["shots"] if row["subject_id"] in shot["present_subjects"])
            return row["subject_id"], shot["location_id"]
    return None


def _location_text(case: Mapping[str, Any], location_id: str) -> str:
    return next(row["attributes"] for row in case["locations"] if row["location_id"] == location_id)


def side_prompt(location: str, phrases: Sequence[str], left_first: bool) -> str:
    """方位探针一镜：主体出场顺序固定（Subject 1 先写），只互换「画面左侧／右侧」。"""

    first, second = SIDES if left_first else SIDES[::-1]
    return (
        f"{SIDE_SHOT_SIZE}, {PROBE_CAMERA_ANGLE}, {location}. "
        f"Subject 1 ({phrases[0]}) stands on the {first} side of the frame, "
        f"and Subject 2 ({phrases[1]}) stands on the {second} side of the frame."
    )


def size_prompt(location: str, phrase: str, shot_size: str) -> str:
    """景别探针一镜：同一动作，只换景别。"""

    return f"{shot_size}, {PROBE_CAMERA_ANGLE}, {location}. Subject 1 ({phrase}) {SIZE_ACTION}."


def _probe_subjects(case: Mapping[str, Any], source_ids: Sequence[str], phrases: Mapping[str, str]) -> list[dict[str, Any]]:
    rows = {row["subject_id"]: row for row in case["subjects"]}
    return [
        {
            "subject_id": f"Subject {index}",
            "kind": rows[source]["kind"],
            "attributes": rows[source]["attributes"],
            "detector_phrase": phrases[source],
            "source_subject_id": source,
        }
        for index, source in enumerate(source_ids, start=1)
    ]


def _probe_case(case_id: str, probe: str, source: Mapping[str, Any], subjects: list[dict[str, Any]],
                location_id: str, shots: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "probe": probe,
        "source_case_id": source["case_id"],
        "source_location_id": location_id,
        "shot_count": len(shots),
        "story_prompt": roster_prompt(subjects),
        "subjects": subjects,
        "shots": shots,
    }


def side_probe_cases(bench: Mapping[str, Any], *, size: int, seed: str) -> list[dict[str, Any]]:
    """方位探针：按种子从「有一对会动、好区分、同镜出现过的主体」的故事里抽 ``size`` 条，每条一个场景两镜。"""

    by_id = {case["case_id"]: case for case in bench["cases"]}
    pool = [case_id for case_id, case in by_id.items() if _side_pair(case) is not None]
    rows = []
    for number, case_id in enumerate(sample_case_ids(pool, size=size, seed=seed), start=1):
        case = by_id[case_id]
        a, b, location_id = _side_pair(case)
        phrases = subject_phrases(case)
        subjects = _probe_subjects(case, (a, b), phrases)
        location = _location_text(case, location_id)
        pair = (phrases[a], phrases[b])
        shots = [
            {
                "shot_id": f"s{index:02d}",
                "prompt": side_prompt(location, pair, left_first),
                "frame_count": PROBE_FRAME_COUNT,
                "instructed_left_to_right": ["Subject 1", "Subject 2"] if left_first else ["Subject 2", "Subject 1"],
            }
            for index, left_first in enumerate((True, False), start=1)
        ]
        rows.append(_probe_case(f"sto-side-{number:02d}", PROBE_SIDE, case, subjects, location_id, shots))
    return rows


def size_probe_cases(bench: Mapping[str, Any], *, size: int, seed: str) -> list[dict[str, Any]]:
    """景别探针：按种子从「有会动的主体」的故事里抽 ``size`` 条，每条一个场景，特写与远景两镜。"""

    by_id = {case["case_id"]: case for case in bench["cases"]}
    pool = [case_id for case_id, case in by_id.items() if _size_subject(case) is not None]
    rows = []
    for number, case_id in enumerate(sample_case_ids(pool, size=size, seed=seed), start=1):
        case = by_id[case_id]
        subject_id, location_id = _size_subject(case)
        phrases = subject_phrases(case)
        subjects = _probe_subjects(case, (subject_id,), phrases)
        location = _location_text(case, location_id)
        shots = [
            {
                "shot_id": f"s{index:02d}",
                "prompt": size_prompt(location, phrases[subject_id], shot_size),
                "frame_count": PROBE_FRAME_COUNT,
                "shot_size": shot_size,
            }
            for index, shot_size in enumerate(SIZE_PAIR, start=1)
        ]
        rows.append(_probe_case(f"sto-size-{number:02d}", PROBE_SIZE, case, subjects, location_id, shots))
    return rows


def same_location_shots(case: Mapping[str, Any]) -> list[int]:
    """与上一镜同地点的镜（0 起下标，不含首镜）。"""

    shots = case["shots"]
    return [index for index in range(1, len(shots)) if shots[index]["location_id"] == shots[index - 1]["location_id"]]


def slice_case_ids(bench: Mapping[str, Any], *, per_shot_count: int, seed: str) -> list[str]:
    """按镜数分层，各抽 ``per_shot_count`` 条；只从至少有一处同地点镜界的故事里抽（三条坏例臂的靶镜都要求同地点镜）。"""

    pools: dict[int, list[str]] = {}
    for case in bench["cases"]:
        if same_location_shots(case):
            pools.setdefault(case["shot_count"], []).append(case["case_id"])
    picked: list[str] = []
    for count in sorted(pools):
        picked.extend(sample_case_ids(pools[count], size=per_shot_count, seed=f"{seed}:k{count}"))
    return picked


# ---------------------------------------------------------------------------
# 合成坏例：靶镜与变换
# ---------------------------------------------------------------------------


def mirror_target(case: Mapping[str, Any]) -> str | None:
    """左右镜像的靶镜：第一个含两个以上在场主体的同地点镜。"""

    for index in same_location_shots(case):
        if len(case["shots"][index]["present_subjects"]) >= 2:
            return case["shots"][index]["shot_id"]
    return None


def copy_target(case: Mapping[str, Any]) -> str | None:
    """照抄前镜的靶镜：第一个景别与上一镜不同的同地点镜。"""

    shots = case["shots"]
    for index in same_location_shots(case):
        if shots[index]["shot_size"] != shots[index - 1]["shot_size"]:
            return shots[index]["shot_id"]
    return None


def swap_target(case: Mapping[str, Any]) -> str | None:
    """换背景的靶镜：第一个同地点镜。"""

    found = same_location_shots(case)
    return case["shots"][found[0]]["shot_id"] if found else None


def swap_donor(cases: Sequence[Mapping[str, Any]], story_id: str, shot_id: str) -> tuple[str, str]:
    """换背景的供体：从本故事之后（循环）的故事里，取第一个帧数相同、地点描述不同的镜。"""

    order = [case["case_id"] for case in cases]
    by_id = {case["case_id"]: case for case in cases}
    target = next(shot for shot in by_id[story_id]["shots"] if shot["shot_id"] == shot_id)
    location = _location_text(by_id[story_id], target["location_id"])
    start = order.index(story_id)
    for step in range(1, len(order)):
        other = by_id[order[(start + step) % len(order)]]
        for shot in other["shots"]:
            if shot["frame_count"] == target["frame_count"] and _location_text(other, shot["location_id"]) != location:
                return other["case_id"], shot["shot_id"]
    raise ValueError(f"{story_id}/{shot_id}: no other story has a {target['frame_count']}-frame shot elsewhere")


def mirror_frames(frames: np.ndarray) -> np.ndarray:
    """整镜水平翻转。"""

    return np.ascontiguousarray(frames[:, :, ::-1])


def resample_frames(frames: np.ndarray, count: int) -> np.ndarray:
    """按帧数等距重采样（首末帧对齐）：照抄前镜时把上一镜拉成靶镜的帧数。"""

    if count < 1 or len(frames) < 1:
        raise ValueError(f"cannot resample {len(frames)} frames to {count}")
    index = np.round(np.linspace(0, len(frames) - 1, count)).astype(int)
    return frames[index]


# ---------------------------------------------------------------------------
# 检测框：归属、方位、景别
# ---------------------------------------------------------------------------


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def subject_boxes(
    regions: Sequence[Mapping[str, Any]], subject_ids: Sequence[str], *, min_phrase_score: float
) -> dict[str, dict[str, Any] | None]:
    """一帧的全部检出框 -> 每个主体的框。``regions[i]["phrase_scores"]`` 与 ``subject_ids`` 同序。

    每个框归给得分最高的那条描述，且这一得分须高于 ``min_phrase_score``；一个主体取归给它的框里得分最高的。
    """

    best: dict[str, dict[str, Any] | None] = {subject: None for subject in subject_ids}
    for region in regions:
        scores = region["phrase_scores"]
        if len(scores) != len(subject_ids):
            raise ValueError(f"region carries {len(scores)} phrase scores for {len(subject_ids)} subjects")
        owner = max(range(len(scores)), key=scores.__getitem__)
        score = float(scores[owner])
        if score <= min_phrase_score:
            continue
        subject = subject_ids[owner]
        if best[subject] is None or score > best[subject]["phrase_score"]:
            best[subject] = {"box": list(region["box"]), "phrase_score": score}
    return best


def frame_left(box_a: Sequence[float] | None, box_b: Sequence[float] | None, *, width: int) -> int | None:
    """一帧里两主体谁在左：0 是第一个、1 是第二个；缺框、IoU ≥ 0.5、中心横向间距小于画宽 5% 都记 None。"""

    if box_a is None or box_b is None:
        return None
    if box_iou(box_a, box_b) >= SIDE_OVERLAP_IOU:
        return None
    center_a, center_b = (box_a[0] + box_a[2]) / 2, (box_b[0] + box_b[2]) / 2
    if abs(center_a - center_b) < SIDE_MIN_GAP_FRACTION * width:
        return None
    return 0 if center_a < center_b else 1


def majority(values: Sequence[Any]) -> Any:
    """三帧取多数：至少两帧给出同一个非 None 答案才算数，否则 None。"""

    counts: dict[Any, int] = {}
    for value in values:
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    winners = [value for value, count in counts.items() if count * 2 > len(values)]
    return winners[0] if winners else None


def side_offsets(frame_count: int) -> tuple[int, ...]:
    """一镜内 50%、75%、100% 处的帧下标（四舍五入，0 起）。"""

    if frame_count < 1:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    return tuple(math.floor(fraction * (frame_count - 1) + 0.5) for fraction in SIDE_FRACTIONS)


def shot_side(
    frame_boxes: Sequence[Mapping[str, dict[str, Any] | None]], present: Sequence[str], *, width: int
) -> dict[str, str | None]:
    """一镜的画面方位：在场主体两两（按名单顺序）一对，键 ``"A|B"``，值是三帧多数认定的左边那位，判不了为 None。"""

    result: dict[str, str | None] = {}
    for a, b in combinations(present, 2):
        votes = [
            frame_left(None if boxes[a] is None else boxes[a]["box"], None if boxes[b] is None else boxes[b]["box"], width=width)
            for boxes in frame_boxes
        ]
        winner = majority(votes)
        result[f"{a}|{b}"] = None if winner is None else (a, b)[winner]
    return result


def height_ratio(box: Sequence[float] | None, *, height: int) -> float | None:
    """框高占画高的比例；没有框为 None。"""

    return None if box is None else (box[3] - box[1]) / height


# ---------------------------------------------------------------------------
# 构图：景别相邻一档的工具题（10-05 改法；判官只问相差 ≥ 2 档的镜）
# ---------------------------------------------------------------------------

#: 相邻一档改用工具：头框高占画高之比。第一项读数（10-05）里主体整框高比例分不开特写与远景
#: （看帧认定照做的 21 对 vs 没照做的 19 对 AUC 0.75），头框高比例 AUC 1.00；主体整框在近景里被画框截断、
#: 在远景里是站立全身，框高都接近画高。
HEAD_PHRASE = "a head"
#: 相邻一档要求头框高的对数比朝剧本方向至少这么多：第一项照做的 21 对特写对远景（相差 4 档）对数比中位 0.83，
#: 一档约 0.21，取半档。
SIZE_STEP_MIN_LOG_RATIO = 0.1


def subject_head(subject_box: Sequence[float] | None, heads: Sequence[Mapping[str, Any]]) -> list[float] | None:
    """主体框内（按头框中心）得分最高的头框；主体没框或框内没有头为 None。"""

    if subject_box is None:
        return None
    x0, y0, x1, y1 = subject_box
    inside = [head for head in heads
              if x0 <= (head["box"][0] + head["box"][2]) / 2 <= x1 and y0 <= (head["box"][1] + head["box"][3]) / 2 <= y1]
    return None if not inside else list(max(inside, key=lambda head: head["score"])["box"])


def size_step_holds(previous_head: Sequence[float] | None, current_head: Sequence[float] | None,
                    previous_size: str, current_size: str) -> bool | None:
    """剧本说本镜比上一镜紧（或松）一档：同一主体的头框高对数比须朝这个方向至少 :data:`SIZE_STEP_MIN_LOG_RATIO`。

    任一镜没有头框为 None（判不了）。只用于相差恰好一档的镜。
    """

    if shot_size_gap(previous_size, current_size) != 1:
        raise ValueError(f"the head-ratio tool answers one-step changes only: {previous_size} -> {current_size}")
    if previous_head is None or current_head is None:
        return None
    ratio = math.log((current_head[3] - current_head[1]) / (previous_head[3] - previous_head[1]))
    tighter = SHOT_SIZES.index(current_size) < SHOT_SIZES.index(previous_size)
    return ratio >= SIZE_STEP_MIN_LOG_RATIO if tighter else ratio <= -SIZE_STEP_MIN_LOG_RATIO


# ---------------------------------------------------------------------------
# 背景几何：抠主体、SIFT、比值检验、RANSAC 基础矩阵
# ---------------------------------------------------------------------------


def background_mask(masks: Sequence[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """全部主体掩码的并集膨胀 :data:`MASK_DILATION_PX` 像素后取反：255 是可取特征的背景。"""

    import cv2

    union = np.zeros(shape, dtype=np.uint8)
    for mask in masks:
        union[np.asarray(mask, dtype=bool)] = 255
    if MASK_DILATION_PX > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MASK_DILATION_PX + 1, 2 * MASK_DILATION_PX + 1))
        union = cv2.dilate(union, kernel)
    return np.where(union > 0, 0, 255).astype(np.uint8)


def sift_features(frame: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """RGB 帧在背景掩码内的 SIFT：(关键点坐标 N×2, 描述子 N×128 或 None)。"""

    import cv2

    gray = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = cv2.SIFT_create().detectAndCompute(gray, mask)
    points = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32).reshape(-1, 2)
    return points, descriptors


class _PointSet:
    """已用关键点的网格索引：查询一个点 :data:`SAME_POINT_RADIUS_PX` 以内有没有已用的点。"""

    def __init__(self) -> None:
        self._cells: dict[tuple[int, int], list[tuple[float, float]]] = {}

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return math.floor(x / SAME_POINT_RADIUS_PX), math.floor(y / SAME_POINT_RADIUS_PX)

    def near(self, x: float, y: float) -> bool:
        cx, cy = self._cell(x, y)
        return any(
            (x - px) ** 2 + (y - py) ** 2 < SAME_POINT_RADIUS_PX**2
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for px, py in self._cells.get((cx + dx, cy + dy), ())
        )

    def add(self, x: float, y: float) -> None:
        self._cells.setdefault(self._cell(x, y), []).append((x, y))


def one_to_one(matches: Sequence[Any], points_a: np.ndarray, points_b: np.ndarray) -> list[Any]:
    """比值检验后的匹配按描述子距离从小到大，两边各一个点只用一次（相距不到 2 px 算同一个点）。

    SIFT 在同一位置常给出几个方向或尺度不同的关键点，比值检验也放过多对一的匹配；一个点不能对应几个点。
    不去掉它们，基础矩阵会被几个重复点撑出假内点：10-04 实测 ssb2-k2-11 一对关键帧的 8 对匹配里右侧只有
    3 个不同的点，Linux 版 OpenCV 4.13 的 USAC 为此断言崩溃，macOS 版给出 8 个假内点；合成测试里右侧点只差
    0.01 px 的簇同样会崩，所以按半径而不是按坐标逐位相同去重。半径小于 RANSAC 的 3 px 内点阈值。
    """

    used_a, used_b = _PointSet(), _PointSet()
    kept = []
    for match in sorted(matches, key=lambda item: item.distance):
        ax, ay = (float(value) for value in points_a[match.queryIdx])
        bx, by = (float(value) for value in points_b[match.trainIdx])
        if used_a.near(ax, ay) or used_b.near(bx, by):
            continue
        used_a.add(ax, ay)
        used_b.add(bx, by)
        kept.append(match)
    return kept


def _hartley_normalize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """点坐标平移到质心、缩放到平均距离 √2（Hartley 归一化）：(归一化齐次坐标 N×3, 3×3 变换)。"""

    centre = points.mean(axis=0)
    spread = np.sqrt(((points - centre) ** 2).sum(axis=1)).mean()
    scale = math.sqrt(2.0) / spread
    transform = np.array([[scale, 0.0, -scale * centre[0]], [0.0, scale, -scale * centre[1]], [0.0, 0.0, 1.0]])
    return np.c_[points, np.ones(len(points))] @ transform.T, transform


def eight_point(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """归一化齐次坐标上的八点法：解 ``right^T F left = 0`` 的最小二乘，再强制秩 2。"""

    rows = np.einsum("ni,nj->nij", right, left).reshape(len(left), 9)
    _, _, vt = np.linalg.svd(rows)
    u, s, vt_f = np.linalg.svd(vt[-1].reshape(3, 3))
    return u @ np.diag([s[0], s[1], 0.0]) @ vt_f


def epipolar_distance(fundamental: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """每对点到对方极线的像素距离取两边的大者（OpenCV 经典 ``FM_RANSAC`` 的内点判据）。"""

    lines_right = left @ fundamental.T
    lines_left = right @ fundamental
    tiny = np.finfo(np.float64).tiny
    d_right = np.abs((right * lines_right).sum(axis=1)) / np.sqrt(np.maximum((lines_right[:, :2] ** 2).sum(axis=1), tiny))
    d_left = np.abs((left * lines_left).sum(axis=1)) / np.sqrt(np.maximum((lines_left[:, :2] ** 2).sum(axis=1), tiny))
    return np.maximum(d_left, d_right)


def fundamental_inliers(left: np.ndarray, right: np.ndarray) -> int:
    """一一对应的匹配点 -> RANSAC 估基础矩阵的最大内点数（八点最小样本、固定种子、置信度自适应停止）。

    不用 OpenCV 的估计器：4.13 的经典 ``FM_RANSAC`` 遇到整体平移的匹配会断言崩溃，``USAC_*`` 全部七种在
    Linux 版上遇到几乎不动的匹配（照抄臂靶镜对它的源镜）同样断言崩溃（10-04 实测 ssb2-k3-12，2664 对匹配）。
    几乎不动的两帧上任何反对称矩阵都满足极线约束，这里照常解出一个，全部点算内点，与几何上的「同一视角」一致。
    """

    count = len(left)
    if count < MIN_FUNDAMENTAL_MATCHES:
        return 0
    norm_left, t_left = _hartley_normalize(np.asarray(left, dtype=np.float64))
    norm_right, t_right = _hartley_normalize(np.asarray(right, dtype=np.float64))
    pixel_left = np.c_[np.asarray(left, dtype=np.float64), np.ones(count)]
    pixel_right = np.c_[np.asarray(right, dtype=np.float64), np.ones(count)]
    rng = np.random.default_rng(RANSAC_SEED)
    best, needed, iteration = 0, RANSAC_MAX_ITERATIONS, 0
    while iteration < needed:
        sample = rng.choice(count, MIN_FUNDAMENTAL_MATCHES, replace=False)
        fundamental = t_right.T @ eight_point(norm_left[sample], norm_right[sample]) @ t_left
        inliers = int((epipolar_distance(fundamental, pixel_left, pixel_right) < RANSAC_THRESHOLD_PX).sum())
        if inliers > best:
            best = inliers
            all_inlier_sample = (best / count) ** MIN_FUNDAMENTAL_MATCHES
            if all_inlier_sample >= 1.0:
                needed = 0
            elif all_inlier_sample > 0.0:
                needed = min(
                    RANSAC_MAX_ITERATIONS,
                    math.ceil(math.log(1.0 - RANSAC_CONFIDENCE) / math.log1p(-all_inlier_sample)),
                )
        iteration += 1
    return best


def ransac_inliers(
    a: tuple[np.ndarray, np.ndarray | None], b: tuple[np.ndarray, np.ndarray | None]
) -> int:
    """两帧背景特征：比值检验、一一对应后的匹配上估基础矩阵（RANSAC），返回内点数；匹配不到 8 对为 0。"""

    import cv2

    (points_a, desc_a), (points_b, desc_b) = a, b
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return 0
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_a, desc_b, k=2)
    ratio = [pair[0] for pair in knn if len(pair) == 2 and pair[0].distance < SIFT_RATIO * pair[1].distance]
    good = one_to_one(ratio, points_a, points_b)
    left = np.float64([points_a[match.queryIdx] for match in good]).reshape(-1, 2)
    right = np.float64([points_b[match.trainIdx] for match in good]).reshape(-1, 2)
    return fundamental_inliers(left, right)


def pair_inliers(
    shot_a: Sequence[tuple[np.ndarray, np.ndarray | None]], shot_b: Sequence[tuple[np.ndarray, np.ndarray | None]]
) -> tuple[int, tuple[int, int]]:
    """两镜各自的关键帧特征 -> 全部关键帧对里内点数的最大值，及取到最大值的 (前镜帧位, 后镜帧位)。"""

    best, where = -1, (0, 0)
    for i, left in enumerate(shot_a):
        for j, right in enumerate(shot_b):
            count = ransac_inliers(left, right)
            if count > best:
                best, where = count, (i, j)
    return best, where


def compatible(inliers: int, control_median: float) -> bool:
    """「相容」：n ≥ 15 且 n ≥ 2 × 对照组中位数（预注册）。"""

    return inliers >= LAYOUT_MIN_INLIERS and inliers >= LAYOUT_CONTROL_FACTOR * control_median


def layout_verdicts(stories: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """一条臂全部故事的相邻镜对 -> 同地点镜对的「相容」判定。

    ``stories[story]`` 是这条视频的相邻镜对，每项 ``{"pair": [前镜, 后镜], "same_location": bool, "inliers": n, ...}``。
    对照组：同一条视频里地点不同的相邻镜对的内点数；这条视频没有，就合并同一臂里其他故事的这类镜对。
    """

    own = {story: [row["inliers"] for row in rows if not row["same_location"]] for story, rows in stories.items()}
    verdicts: dict[str, list[dict[str, Any]]] = {}
    for story, rows in stories.items():
        if own[story]:
            control, source = own[story], "same_video"
        else:
            control = [value for other, values in own.items() if other != story for value in values]
            source = "other_stories"
        if not control:
            raise ValueError(f"{story}: no different-location adjacent pair in this arm to serve as the control")
        level = float(median(control))
        verdicts[story] = [
            {
                **row,
                "control_median": level,
                "control_source": source,
                "control_n": len(control),
                "compatible": compatible(row["inliers"], level),
            }
            for row in rows
            if row["same_location"]
        ]
    return verdicts


# ---------------------------------------------------------------------------
# 真片校准：地标布局尺子在真实镜界上的真阳性率／假阳性率
# ---------------------------------------------------------------------------

#: 真片没有剧本主体描述，检测短语固定为这两条，过阈值的框全部抠掉。
REAL_LAYOUT_PHRASES = ("a person", "an animal")
#: 真片按原片解码后等比缩到这个长边（生成片是 832×480）。
REAL_LAYOUT_LONG_SIDE = 832
#: 正例层 -> 取哪两镜成对：回机位取 Z→B（B 回到的就是 Z 的机位），其余取 A→B。
REAL_LAYOUT_POSITIVE_PAIRS = {
    "continuation": ("a", "b"),
    "same_scene_new_camera": ("a", "b"),
    "return_to_setup": ("z", "b"),
    "same_scene_cast_change": ("a", "b"),
}
REAL_LAYOUT_SCENE_CHANGE = "scene_change"
#: 回机位那一刀本身（A→B，换机位）作参照另报。
REAL_LAYOUT_RETURN_CUT = "return_to_setup_cut"
#: 跨视频镜对：C1 只给对照组中位数，C2 读假阳性率，两组不重叠。
REAL_LAYOUT_CONTROL = "cross_video_c1"
REAL_LAYOUT_NEGATIVE = "cross_video_c2"


def _cross_video_partner(ids: Sequence[str], case_of: Mapping[str, str], index: int, shift: int,
                         taken: set[tuple[str, str]]) -> str:
    """同一数据源按 sample_id 循环，从第 ``shift`` 个起往后找第一个 case 不同、且这对没被别的组用过的样本。"""

    sample = ids[index]
    for step in range(shift, shift + len(ids)):
        other = ids[(index + step) % len(ids)]
        if case_of[other] != case_of[sample] and (sample, other) not in taken:
            return other
    raise ValueError(f"{sample}: no cross-video partner left in its source")


def real_layout_pairs(samples: Sequence[Mapping[str, Any]], reference: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """校准样本＋参考标签 -> 全部要量的镜对 ``{pair_id, stratum, conf, left: [样本, 镜], right: [样本, 镜]}``。"""

    pairs: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = sample["sample_id"]
        label = reference[sample_id]
        cut, conf = label["cut_type"], label["conf"]
        left, right = REAL_LAYOUT_POSITIVE_PAIRS.get(cut, ("a", "b"))
        if left not in sample["frames"]:
            raise ValueError(f"{sample_id}: {cut} needs shot {left!r}, the sample has {sorted(sample['frames'])}")
        pairs.append({"pair_id": f"{sample_id}:{left}{right}", "stratum": cut, "conf": conf,
                      "left": [sample_id, left], "right": [sample_id, right]})
        if cut == "return_to_setup":
            pairs.append({"pair_id": f"{sample_id}:ab", "stratum": REAL_LAYOUT_RETURN_CUT, "conf": conf,
                          "left": [sample_id, "a"], "right": [sample_id, "b"]})
    case_of = {sample["sample_id"]: sample["case_id"] for sample in samples}
    by_source: dict[str, list[str]] = {}
    for sample in samples:
        by_source.setdefault(sample["source"], []).append(sample["sample_id"])
    taken: set[tuple[str, str]] = set()
    for shift, stratum in ((1, REAL_LAYOUT_CONTROL), (2, REAL_LAYOUT_NEGATIVE)):
        for source in sorted(by_source):
            ids = sorted(by_source[source])
            for index, sample_id in enumerate(ids):
                other = _cross_video_partner(ids, case_of, index, shift, taken)
                taken.add((sample_id, other))
                pairs.append({"pair_id": f"{sample_id}:a~{other}:b", "stratum": stratum, "conf": None,
                              "left": [sample_id, "a"], "right": [other, "b"]})
    return pairs


def real_layout_readout(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """量好内点数的镜对（每项多一个 ``inliers``）-> 对照组中位数、逐对判定、各层「相容」率。"""

    control = [row["inliers"] for row in pairs if row["stratum"] == REAL_LAYOUT_CONTROL]
    if not control:
        raise ValueError("no cross-video control pairs")
    level = float(median(control))
    judged = [{**row, "compatible": compatible(row["inliers"], level)} for row in pairs
              if row["stratum"] != REAL_LAYOUT_CONTROL]

    def rate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        hits, total = sum(row["compatible"] for row in rows), len(rows)
        interval = wilson_interval(hits, total)
        inliers = sorted(row["inliers"] for row in rows)
        return {"num": hits, "den": total, "rate": hits / total if total else None,
                "wilson95": None if interval is None else [round(interval[0], 4), round(interval[1], 4)],
                "inliers_median": float(median(inliers)) if inliers else None}

    strata = sorted({row["stratum"] for row in judged})
    return {
        "control_median": level,
        "control_n": len(control),
        "control_inliers_p90": float(np.percentile(control, 90)),
        "by_stratum": {stratum: rate([row for row in judged if row["stratum"] == stratum]) for stratum in strata},
        "by_stratum_sure": {
            stratum: rate([row for row in judged if row["stratum"] == stratum and row["conf"] == "sure"])
            for stratum in strata if stratum not in (REAL_LAYOUT_NEGATIVE,)
        },
        "rows": judged,
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    """二项比例的 Wilson 95% 区间；分母为 0 时为 None。"""

    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def shot_size_gap(a: str, b: str) -> int:
    """两档景别在七档词表上的距离。"""

    return abs(SHOT_SIZES.index(a) - SHOT_SIZES.index(b))


__all__ = [
    "ARMS",
    "ARM_COPY",
    "ARM_MIRROR",
    "ARM_SWAP",
    "PROBE_SCHEMA",
    "background_mask",
    "box_iou",
    "compatible",
    "copy_target",
    "distinguishable",
    "frame_left",
    "height_ratio",
    "layout_verdicts",
    "majority",
    "mirror_frames",
    "mirror_target",
    "one_to_one",
    "pair_inliers",
    "real_layout_pairs",
    "real_layout_readout",
    "ransac_inliers",
    "resample_frames",
    "same_location_shots",
    "shot_side",
    "shot_size_gap",
    "side_offsets",
    "side_probe_cases",
    "size_step_holds",
    "subject_head",
    "sift_features",
    "size_probe_cases",
    "slice_case_ids",
    "subject_boxes",
    "subject_phrases",
    "swap_donor",
    "swap_target",
    "wilson_interval",
]
