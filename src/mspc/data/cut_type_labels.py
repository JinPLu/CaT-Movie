"""镜界类型（cut_type）与逐镜景别／机位的自动标注器：词表、提示词、严格解析、判定规则、校准指标。

数据基建用途（给训练池分桶、给配对降权），不是方法设计。一个镜界 = 相邻两镜 A→B；标注器看两镜各
``FRAMES_PER_SHOT`` 帧（有更早一镜 Z 时再看 Z 的帧），输出：

- 两镜各自的 ``shot_size``、``camera_angle``（固定词表，见下）；
- 三个是非判断 ``same_place`` / ``same_cast`` / ``same_camera_setup``，以及有 Z 时的
  ``returns_to_earlier_setup``；
- ``cut_type`` 不由模型直接给，由 :func:`derive_cut_type` 按这四个判断机械算出（一条路径，模型不会
  给出与自己的判断相矛盾的类别）。

词表来源：景别与角度取 Wan2.2 文生视频提示词扩写器的系统提示词（``wan/utils/system_prompt.py``，
``T2V_A14B_EN_SYS_PROMPT`` 的「镜头尺寸」「拍摄角度」两行；主服务器上的副本在
``$MSPC_SHARED_ROOT/Github/StoryMem/wan/utils/system_prompt.py``），即
``renderer-memory-for-planner-ceiling`` 决策卡「景别和角度的词表直接用 Wan2.2 系统提示词里那一套」。
扩写器的角度表没有平视一项（平视是默认，不写），标注器必须给每个镜头一个角度，所以补一项
``Eye-level shot``——这是本仓库对该词表唯一的增补。

CI-VID 的原生字段（``change_in_camera_angle`` 等）只做交叉核对特征（:func:`native_features`），
不当标签，也不直接映射成景别（``DATASET_ROLES.md`` 禁止）。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = "cut_type_labeler_v1"
PROMPT_ID = "cut_type_prompt_v2"

CUT_TYPES = (
    "continuation",
    "same_scene_new_camera",
    "return_to_setup",
    "same_scene_cast_change",
    "scene_change",
)

#: Wan2.2 扩写器「镜头尺寸」七档，按由近到远排列（序数距离依赖这个顺序）。
SHOT_SIZES = (
    "Extreme close-up shot",
    "Close-up shot",
    "Medium close-up shot",
    "Medium shot",
    "Medium wide shot",
    "Wide shot",
    "Extreme wide shot",
)
#: 三档粗分：近（含中近景）、中、远。
SHOT_SIZE_BINS = {
    "Extreme close-up shot": "close",
    "Close-up shot": "close",
    "Medium close-up shot": "close",
    "Medium shot": "medium",
    "Medium wide shot": "medium",
    "Wide shot": "wide",
    "Extreme wide shot": "wide",
}
#: Wan2.2 扩写器「拍摄角度」六档 + 平视（增补）。
CAMERA_ANGLES = (
    "Eye-level shot",
    "Low angle shot",
    "High angle shot",
    "Overhead shot",
    "Aerial shot",
    "Dutch angle shot",
    "Over-the-shoulder shot",
)

FRAMES_PER_SHOT = 3
#: 一镜取 3 帧：镜首、镜中、镜尾（镜内位置比例，落到该镜编码时间轴上取整）。
FRAME_FRACTIONS = (0.05, 0.5, 0.95)
FRAME_NAMES = ("start", "middle", "end")

SHOT_KEYS = {"place", "main_subjects", "shot_size", "camera_angle"}
OUTPUT_KEYS = {
    "shot_a",
    "shot_b",
    "same_place",
    "same_cast",
    "same_camera_setup",
    "returns_to_earlier_setup",
}


class LabelerOutputError(ValueError):
    """模型输出不是合法的标注 JSON；记为该样本的标注失败，不重试、不兜底。"""


# ---------------------------------------------------------------------------
# 抽帧位置：纯函数
# ---------------------------------------------------------------------------


def shot_frame_positions(start: int, end: int) -> list[int]:
    """一镜 ``[start, end)`` 的三个采样位置（编码时间轴帧号）：镜首、镜中、镜尾附近。"""

    if type(start) is not int or type(end) is not int or start < 0 or end <= start:
        raise ValueError(f"shot range must be a non-empty half-open range; got [{start}, {end})")
    span = end - start - 1
    return [start + int(fraction * span) for fraction in FRAME_FRACTIONS]


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a film editor labelling the cut between two consecutive shots of a video. "
    "You are given still frames sampled from each shot, in time order. Answer only with the JSON object asked for."
)

_RULES = """Definitions.
shot_size (for A and for B; judge it on the middle frame) is exactly one of:
- "Extreme close-up shot": only a detail fills the frame (an eye, a mouth, a hand, a small object).
- "Close-up shot": a face, or one object of similar size, fills the frame.
- "Medium close-up shot": head and shoulders down to the chest.
- "Medium shot": a person from the waist up.
- "Medium wide shot": a person from the knees up, or a small group at conversation distance.
- "Wide shot": whole bodies are visible with a good deal of their surroundings; the place is clearly shown.
- "Extreme wide shot": people are tiny or absent; a landscape, cityscape or aerial view.
For a subject that is not a person use the same scale by analogy (an object filling the frame is a close-up).
camera_angle (for A and for B) is exactly one of: "Eye-level shot" (camera about at the subject's eye height; the default), "Low angle shot" (camera below the subject, looking up), "High angle shot" (camera above the subject, looking down at an angle), "Overhead shot" (camera straight above, looking straight down), "Aerial shot" (from a drone, helicopter or great height over the ground), "Dutch angle shot" (camera rolled so the horizon is slanted), "Over-the-shoulder shot" (camera just behind one person's shoulder, looking at another person or thing).
place: name the specific spot and its distinctive background details (walls, decor, furniture, landscape, signs) in a few words. main_subjects: name who or what the shot is about (people by clothing or hair, animals, objects) in a few words.

Questions about the cut from shot A to shot B. Compare the end of A with the start of B.
- same_place: true only if the backgrounds show the same specific spot (the same walls, decor, furniture, layout or landscape), even when seen from another side. A different room, studio, court, street, venue, stage set or landscape is false even when it is the same kind of place (two different podcast studios: false; two different tennis courts: false; a title card or a new scene: false).
- same_cast: true if a main person or character of A is still a main subject of B, seen closer, wider or from another side, while B is not mainly about someone else; a newcomer who joins them does not make it false. False if B is mainly about a different person or character than A (the reverse side of a conversation, another person taking over the frame), or if A's main subjects are gone. A close-up of an object or a hand in the same scene is true. When neither shot has people or characters, compare the main objects the same way.
- same_camera_setup: true if the composition of B is essentially the composition of A's end: the same shot size class, camera angle and camera position, with only the action changing (a static camera that goes on, a slow pan, zoom or dolly that continues, subjects moving inside the same frame, footage from the same body or dash camera). False if the camera jumps to a new position, viewpoint or shot size.
- returns_to_earlier_setup: only when an earlier shot Z is shown. true only if B looks like Z (the same specific place, the same main subject, framed the same way from the same viewpoint) AND A is clearly different from B. If A also looks like Z and B, or if B only shares the kind of place or the kind of subject with Z, answer false. When no earlier shot is shown this must be null.

Examples of how the answers combine.
- A static wide shot of a marina, then the same view again: same_place true, same_cast true, same_camera_setup true.
- A wide shot of two singers on a stage, then a closer shot of the same two from another position: true, true, false.
- A close-up of a woman, then a close-up of the man she talks to in the same room: true, false, and same_camera_setup false.
- A shot in a kitchen, then a shot on a street: same_place false.
- Interview close-up (Z), then a wide shot of the room (A), then the same close-up framing again (B): returns_to_earlier_setup true."""

_OUTPUT_SPEC = """Output only this JSON object:
{"shot_a": {"place": "...", "main_subjects": "...", "shot_size": "...", "camera_angle": "..."}, "shot_b": {"place": "...", "main_subjects": "...", "shot_size": "...", "camera_angle": "..."}, "same_place": true|false, "same_cast": true|false, "same_camera_setup": true|false, "returns_to_earlier_setup": true|false|null}"""


def _image_parts(label: str, frames: Sequence[str]) -> list[dict[str, Any]]:
    if len(frames) != FRAMES_PER_SHOT:
        raise ValueError(f"{label}: expected {FRAMES_PER_SHOT} frames, got {len(frames)}")
    parts: list[dict[str, Any]] = []
    for index, url in enumerate(frames):
        if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
            raise ValueError(f"{label}: frame {index} must be a data:image/...;base64,... URL")
        parts.append({"type": "text", "text": f"{label}, frame {index + 1}/{FRAMES_PER_SHOT} ({FRAME_NAMES[index]} of the shot)"})
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def labeler_messages(
    frames_a: Sequence[str],
    frames_b: Sequence[str],
    frames_earlier: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """标注一个镜界的整条对话：Z（可无）、A、B 三组帧 + 定义 + 输出格式。帧是 data URL。"""

    content: list[dict[str, Any]] = []
    if frames_earlier is not None:
        content += _image_parts("Earlier shot Z (the shot right before A)", frames_earlier)
    content += _image_parts("Shot A", frames_a)
    content += _image_parts("Shot B (the shot right after A)", frames_b)
    earlier_line = (
        "An earlier shot Z is shown, so returns_to_earlier_setup must be true or false, never null."
        if frames_earlier is not None
        else "No earlier shot is shown, so returns_to_earlier_setup must be null."
    )
    content.append({"type": "text", "text": f"{earlier_line}\n\n{_RULES}\n\n{_OUTPUT_SPEC}"})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


# ---------------------------------------------------------------------------
# 严格解析与判定规则
# ---------------------------------------------------------------------------

_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _loads_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    fenced = _FENCED_JSON.fullmatch(text)
    if fenced is not None:
        text = fenced.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise LabelerOutputError(f"output is not JSON: {error}") from error
    if not isinstance(value, dict):
        raise LabelerOutputError("output must be a JSON object")
    return value


def _exact_keys(value: object, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise LabelerOutputError(f"{label} keys must be {sorted(keys)}, got {got}")
    return value


def _enum(value: object, allowed: Sequence[str], label: str) -> str:
    """词表取值；只放宽大小写（模型常把 "Medium wide shot" 写成小写），其余一字不差。"""

    by_lower = {item.lower(): item for item in allowed}
    if not isinstance(value, str) or value.strip().lower() not in by_lower:
        raise LabelerOutputError(f"{label} must be one of {list(allowed)}, got {value!r}")
    return by_lower[value.strip().lower()]


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LabelerOutputError(f"{label} must be a non-empty string")
    return " ".join(value.split())


def _flag(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise LabelerOutputError(f"{label} must be true or false, got {value!r}")
    return value


def derive_cut_type(
    *,
    same_place: bool,
    same_cast: bool,
    same_camera_setup: bool,
    returns_to_earlier_setup: bool | None,
) -> str:
    """四个判断 -> 五类。回机位优先（B 回到更早一镜 Z 的机位而 A 不是）；其余按地点、人、机位依次分。"""

    if returns_to_earlier_setup and not same_camera_setup:
        return "return_to_setup"
    if not same_place:
        return "scene_change"
    if not same_cast:
        return "same_scene_cast_change"
    return "continuation" if same_camera_setup else "same_scene_new_camera"


def parse_labeler_output(raw_text: str, *, has_earlier: bool) -> dict[str, Any]:
    """严格解析一次标注输出；任何偏差抛 :class:`LabelerOutputError`。"""

    root = _exact_keys(_loads_object(raw_text), OUTPUT_KEYS, "output")
    shots: dict[str, dict[str, str]] = {}
    for name in ("shot_a", "shot_b"):
        raw = _exact_keys(root[name], SHOT_KEYS, name)
        shots[name] = {
            "place": _text(raw["place"], f"{name}.place"),
            "main_subjects": _text(raw["main_subjects"], f"{name}.main_subjects"),
            "shot_size": _enum(raw["shot_size"], SHOT_SIZES, f"{name}.shot_size"),
            "camera_angle": _enum(raw["camera_angle"], CAMERA_ANGLES, f"{name}.camera_angle"),
        }
    returns = root["returns_to_earlier_setup"]
    if has_earlier:
        returns = _flag(returns, "returns_to_earlier_setup")
    elif returns is not None:
        raise LabelerOutputError("returns_to_earlier_setup must be null when no earlier shot is shown")
    parsed = {
        "shot_a": shots["shot_a"],
        "shot_b": shots["shot_b"],
        "same_place": _flag(root["same_place"], "same_place"),
        "same_cast": _flag(root["same_cast"], "same_cast"),
        "same_camera_setup": _flag(root["same_camera_setup"], "same_camera_setup"),
        "returns_to_earlier_setup": returns,
    }
    parsed["cut_type"] = derive_cut_type(
        same_place=parsed["same_place"],
        same_cast=parsed["same_cast"],
        same_camera_setup=parsed["same_camera_setup"],
        returns_to_earlier_setup=returns,
    )
    return parsed


# ---------------------------------------------------------------------------
# 校准指标：纯函数
# ---------------------------------------------------------------------------


def confusion_matrix(
    pairs: Sequence[tuple[str, str]], labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    """``pairs`` 是 (参考, 预测)；返回 ``{参考: {预测: 计数}}``，词表外的值直接报错。"""

    matrix = {row: {column: 0 for column in labels} for row in labels}
    for reference, predicted in pairs:
        if reference not in matrix or predicted not in matrix[reference]:
            raise ValueError(f"label outside vocabulary: reference={reference!r} predicted={predicted!r}")
        matrix[reference][predicted] += 1
    return matrix


def per_class_scores(matrix: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, Any]]:
    """每类的支持数、召回（该类参考被判对的比例）、精确率（判成该类的里对的比例）。"""

    labels = list(matrix)
    scores: dict[str, dict[str, Any]] = {}
    for label in labels:
        support = sum(matrix[label].values())
        predicted = sum(matrix[row][label] for row in labels)
        correct = matrix[label][label]
        scores[label] = {
            "support": support,
            "predicted": predicted,
            "recall": correct / support if support else None,
            "precision": correct / predicted if predicted else None,
        }
    return scores


#: 下游真正用到的粗粒度二分：换景（分桶）、构图重复 = 续拍或回机位（配对降权）。
COARSE_TASKS = {
    "scene_change": frozenset({"scene_change"}),
    "composition_repeat": frozenset({"continuation", "return_to_setup"}),
}


def binary_scores(pairs: Sequence[tuple[str, str]], positive: frozenset[str]) -> dict[str, Any]:
    """把五类折成「属于 positive / 不属于」的二分，报混淆计数、精确率、召回、准确率。"""

    tp = fp = fn = tn = 0
    for reference, predicted in pairs:
        if reference not in CUT_TYPES or predicted not in CUT_TYPES:
            raise ValueError(f"label outside vocabulary: reference={reference!r} predicted={predicted!r}")
        ref_hit, pred_hit = reference in positive, predicted in positive
        tp += ref_hit and pred_hit
        fp += pred_hit and not ref_hit
        fn += ref_hit and not pred_hit
        tn += not ref_hit and not pred_hit
    total = tp + fp + fn + tn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "accuracy": (tp + tn) / total if total else None,
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """比例的 Wilson 95% 区间；``total == 0`` 返回 ``None``。"""

    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return (max(0.0, center - half), min(1.0, center + half))


def size_distance(reference: str, predicted: str) -> int:
    """景别序数距离（七档由近到远）。"""

    return abs(SHOT_SIZES.index(reference) - SHOT_SIZES.index(predicted))


# ---------------------------------------------------------------------------
# CI-VID 原生字段：只做交叉核对特征
# ---------------------------------------------------------------------------

_NATIVE_KEY = re.compile(r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)


def parse_native_json(text: str) -> dict[str, str]:
    """CI-VID 的 ``individual_captions`` / ``joint_captions`` 是 JSON 字符串，常带尾部的代码围栏残渣。

    先按整段 JSON 解析；失败就逐个 ``"键": "值"`` 抠出来（值内的转义还原）。一个键都抠不到才返回空。
    """

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return {key: item for key, item in value.items() if isinstance(item, str)}
    except json.JSONDecodeError:
        pass
    found = {}
    for key, raw in _NATIVE_KEY.findall(text):
        try:
            found[key] = json.loads(f'"{raw}"')
        except json.JSONDecodeError:
            found[key] = raw
    return found


# 原生字幕里的景别词，长词在前（"extreme close-up" 先于 "close-up"）。
_NATIVE_SIZE_PATTERNS = (
    ("Extreme close-up shot", r"extreme close[- ]?up"),
    ("Medium close-up shot", r"medium close[- ]?up"),
    ("Close-up shot", r"close[- ]?up"),
    ("Extreme wide shot", r"extreme wide|extreme long|aerial"),
    ("Medium wide shot", r"medium wide|medium long|full shot|full[- ]body"),
    ("Wide shot", r"wide[- ]angle|wide shot|wide view|long shot|wide"),
    ("Medium shot", r"medium"),
)


def native_shot_size(camera_angle_text: str | None) -> str | None:
    """从一个 clip 的原生 ``camera_angle`` 描述里抠景别词；抠不到返回 ``None``。只是交叉核对特征。"""

    if not camera_angle_text:
        return None
    lowered = camera_angle_text.lower()
    hits = [(match.start(), name) for name, pattern in _NATIVE_SIZE_PATTERNS for match in [re.search(pattern, lowered)] if match]
    if not hits:
        return None
    return min(hits)[1]


_SAME_WORDS = re.compile(r"\b(remains?|remained|same|unchanged|consistent|maintains?|maintained|continues?|continuing|persists?|static|no (?:noticeable |observable |significant |visible )?change)\b")
_CHANGE_WORDS = re.compile(r"\b(shifts?|shifted|changes?|changed|changing|transitions?|transitioned|switch(?:es|ed)?|alters?|altered|differs?|different|new|becomes?|diverg\w*|contrasts?|moves? from|from an?\b)")


def native_change_flag(text: str | None) -> str:
    """一条原生 ``change_in_*`` 描述的粗读：``same`` / ``changed`` / ``unknown``（两类词都有或都没有）。"""

    if not text:
        return "unknown"
    lowered = text.lower()
    same = bool(_SAME_WORDS.search(lowered))
    changed = bool(_CHANGE_WORDS.search(lowered))
    if same and not changed:
        return "same"
    if changed and not same:
        return "changed"
    return "unknown"


def native_features(individual: Sequence[str], joint: str) -> dict[str, Any]:
    """一个 CI-VID 镜界的原生交叉核对特征。

    ``individual`` 是 A、B 两个 clip 的 ``individual_captions`` 字符串，``joint`` 是这一对的
    ``joint_captions`` 字符串。返回两 clip 的原生景别词、三个粗读旗标（相机角度、背景、内容）。
    """

    if len(individual) != 2:
        raise ValueError("individual must hold the captions of shot A and shot B")
    a, b = (parse_native_json(item) for item in individual)
    pair = parse_native_json(joint)
    return {
        "native_size_a": native_shot_size(a.get("camera_angle")),
        "native_size_b": native_shot_size(b.get("camera_angle")),
        "camera_angle_flag": native_change_flag(pair.get("change_in_camera_angle")),
        "background_flag": native_change_flag(pair.get("change_in_video_background")),
        "camera_movement_flag": native_change_flag(pair.get("change_in_camera_movement")),
    }


#: 一个 cut_type 对原生旗标的预期（缺项 = 没有预期）。``camera_angle_flag`` 与 ``background_flag`` 取值
#: ``same`` / ``changed``；一致率只在旗标不是 ``unknown`` 的样本上算。
NATIVE_EXPECTATION: dict[str, dict[str, str]] = {
    "continuation": {"camera_angle_flag": "same", "background_flag": "same"},
    "same_scene_new_camera": {"camera_angle_flag": "changed", "background_flag": "same"},
    "same_scene_cast_change": {"background_flag": "same"},
    "scene_change": {"background_flag": "changed"},
    "return_to_setup": {},
}


def native_agreement(cut_type: str, features: Mapping[str, Any]) -> dict[str, bool]:
    """``cut_type`` 与原生旗标逐项是否一致；旗标为 ``unknown`` 或该类无预期的项不出现在结果里。"""

    expected = NATIVE_EXPECTATION[cut_type]
    return {
        flag: features[flag] == want
        for flag, want in expected.items()
        if features[flag] in ("same", "changed")
    }


def class_counts(labels: Sequence[str]) -> dict[str, int]:
    counts = Counter(labels)
    return {label: counts.get(label, 0) for label in CUT_TYPES}
