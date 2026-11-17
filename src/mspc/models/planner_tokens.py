"""目标点 token：每个「目标镜 × 在场人物」一个干净 token 接进 DiT 的自注意力，
不进文字交叉注意力、不加门控与偏置。内容是可学的人物标记（``person_index``）或规划器隐状态经连接器（``planner_features``）。
完整目标空间接口（``person_geometry``）在同一人物嵌入上叠加零初始化的宽高／有向前后关系投影，不追加 token。

token 只放本镜一处：三维 RoPE 放在该镜第一个潜帧（全互读带镜相位时含 +0.5·镜序）、镜首点所在格
（行 ⌊15y/1000⌋、列 ⌊26x/1000⌋）；只读自己，全部视频 token 读它。镜首没有点（null）的人没有 token。镜首点训练时取标注
掩码外接框中心，推理时取计划镜首框中心（``box_point``），同一定义。参考帧不追加目标点 token；源实体的文字 CA
路由由 ``source_identity`` 持有。

token 的时间步调制取 0（与干净参考帧同）。序列排在视频 token 之后：[参考帧][目标][本镜 token]，
过完全部块后、进 head 前去掉。连接器：RMSNorm → Linear(4096→3072) → SiLU → Linear(3072→3072)，末层零初始化，
所有 token 共用一个。刚接上时连接器输出为 0，但这不等于「没有 token」：进块后 norm1(0) 经时间步 0 的调制成为
shift，再经 q/k/v 的偏置，每个 token 成为同一个非零 key/value，目标与参考从第 0 步起就读到它；差多少见运行记录。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

import torch
from torch import nn

PLANNER_CONNECTOR = "planner_connector"
#: token 内容从哪来 → 这种内容的 ``model.planner_tokens`` 必填键。``planner_features``：规划器隐状态经连接器；
#: ``person_index``：位置标记对照，同一格放一个可学的「人物 k」嵌入（k 是名字 Character k 的编号），不用规划器特征。
#: ``person_geometry``：同一人物嵌入额外消费同镜首宽高与前后关系有效位，参数共用显式 learning_rate。
PLANNER_TOKEN_OPTION_KEYS = {"planner_features": ("content", "feature_dim", "feature_layer", "learning_rate"),
                             "person_index": ("content", "max_people", "learning_rate"),
                             "person_geometry": ("content", "max_people", "learning_rate")}
GRID_ROWS, GRID_COLUMNS = 15, 26


@dataclass(frozen=True)
class PlannerTokenOptions:
    content: str
    learning_rate: float
    feature_dim: int | None = None
    feature_layer: int | None = None
    max_people: int | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], location: str = "model.planner_tokens") -> "PlannerTokenOptions":
        if not isinstance(value, Mapping) or value.get("content") not in PLANNER_TOKEN_OPTION_KEYS:
            raise ValueError(f"{location}.content must be one of {sorted(PLANNER_TOKEN_OPTION_KEYS)}")
        keys = PLANNER_TOKEN_OPTION_KEYS[value["content"]]
        if set(value) != set(keys):
            raise ValueError(f"{location} with content {value['content']} requires exactly {sorted(keys)}")
        for name in set(keys) - {"content", "learning_rate"}:
            if type(value[name]) is not int or value[name] < (0 if name == "feature_layer" else 1):
                raise ValueError(f"{location}.{name} must be a {'nonnegative' if name == 'feature_layer' else 'positive'} int")
        if type(value["learning_rate"]) is not float or not value["learning_rate"] > 0:
            raise ValueError(f"{location}.learning_rate must be a positive float")
        return cls(**{key: value[key] for key in keys})


class PlannerConnector(nn.Module):
    def __init__(self, feature_dim: int, dim: int):
        super().__init__()
        self.norm = nn.RMSNorm(feature_dim, eps=1e-6)
        self.inner = nn.Linear(feature_dim, dim)
        self.outer = nn.Linear(dim, dim)
        nn.init.zeros_(self.outer.weight)
        nn.init.zeros_(self.outer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.outer(nn.functional.silu(self.inner(self.norm(features))))


class PersonEmbedding(nn.Module):
    """位置标记对照：「人物 k」的可学嵌入，零初始化（与连接器末层零初始化同一起点）。输入是编号 [token 数]。"""

    def __init__(self, max_people: int, dim: int):
        super().__init__()
        # Embedding 的临时随机初始化随后被清零，不能改变对照组的目标噪声 RNG。
        with torch.random.fork_rng(devices=[]):
            self.table = nn.Embedding(max_people + 1, dim)
        nn.init.zeros_(self.table.weight)

    def forward(self, people: torch.Tensor) -> torch.Tensor:
        if people.dtype != torch.long or people.ndim != 1:
            raise ValueError("person embedding takes a 1-D long tensor of person numbers")
        if int(people.max()) >= self.table.num_embeddings or int(people.min()) < 1:
            raise ValueError(f"person numbers must be in 1..{self.table.num_embeddings - 1}")
        return self.table(people)


class PersonGeometryEmbedding(PersonEmbedding):
    """同一人物ID嵌入加宽高／有向前后关系；几何投影零初始化，不改变初始点接口。"""

    def __init__(self, max_people: int, dim: int):
        super().__init__(max_people, dim)
        with torch.random.fork_rng(devices=[]):
            self.geometry_projection = nn.Linear(3 + 2 * max_people, dim)
        nn.init.zeros_(self.geometry_projection.weight)
        nn.init.zeros_(self.geometry_projection.bias)

    def forward(self, people: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        max_people = self.table.num_embeddings - 1
        if (not isinstance(geometry, torch.Tensor) or not geometry.is_floating_point()
                or geometry.shape != (people.shape[0], 3 + 2 * max_people)
                or not bool(torch.isfinite(geometry).all())):
            raise ValueError("person geometry must be finite [people, 3 + 2 * max_people] floating values")
        size, size_valid = geometry[:, :2], geometry[:, 2:3]
        relation, relation_valid = geometry[:, 3:3 + max_people], geometry[:, 3 + max_people:]
        if (bool(((size_valid != 0) & (size_valid != 1)).any())
                or bool(((relation_valid != 0) & (relation_valid != 1)).any())
                or bool(((size < 0) | (size > 1)).any())
                or bool(((size_valid == 0) & (size != 0)).any())
                or bool(((size_valid == 1) & (size <= 0)).any())
                or bool(((relation_valid == 0) & (relation != 0)).any())
                or bool(((relation_valid == 1) & (relation.abs() != 1)).any())):
            raise ValueError("geometry requires normalized valid sizes, signed known relations and explicit unknown masks")
        identity = super().forward(people)
        return identity + self.geometry_projection(geometry)


def attach_planner_connector(dit: nn.Module, options: PlannerTokenOptions | None) -> None:
    """配置了规划器 token 才挂连接器（或位置标记对照的人物嵌入，挂在同一个属性名下）；没配置时 DiT 结构不变。
    建在 DiT 所在的设备上（DiT 常已先上卡）。"""
    if options is None:
        return
    device = next(dit.parameters()).device
    if options.content == "planner_features":
        module = PlannerConnector(options.feature_dim, dit.dim)
    elif options.content == "person_geometry":
        module = PersonGeometryEmbedding(options.max_people, dit.dim)
    else:
        module = PersonEmbedding(options.max_people, dit.dim)
    setattr(dit, PLANNER_CONNECTOR, module.to(device))


def box_point(box: Sequence[float]) -> list[int]:
    """计划里的框（0–1 相对坐标 x0, y0, x1, y1）→ 中心点（0–1000），与标签构造器的 ``box_center`` 同一定义。"""
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f"box {box!r} is not an ordered box inside 0..1")
    return [min(1000, int(round((x0 + x1) / 2 * 1000))), min(1000, int(round((y0 + y1) / 2 * 1000)))]


def point_cell(point: Sequence[int]) -> tuple[int, int]:
    """0–1000 相对坐标的点 → (行, 列)。"""
    x, y = point
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        raise ValueError(f"point {point!r} is outside 0..1000")
    return min(GRID_ROWS - 1, GRID_ROWS * y // 1000), min(GRID_COLUMNS - 1, GRID_COLUMNS * x // 1000)


@dataclass(frozen=True)
class PlannerTokenCondition:
    """一条样本的规划器 token。

    ``features``：[token 数, feature_dim]，规划器隐状态（还没过连接器）；点控制是 [token 数] 的人物编号（long）。
    ``shot_tokens``：每个 token (特征下标, 目标镜序号 k, 该镜第一个潜帧在目标里的下标, 行, 列)。
    ``geometry``：完整空间接口的 [条目数, 3+2*max_people] 数值／有效位，与 features 同行序，按相同下标取 token。
    """
    features: torch.Tensor
    shot_tokens: tuple[tuple[int, int, int, int, int], ...]
    geometry: torch.Tensor | None = None

    def __post_init__(self):
        if not isinstance(self.features, torch.Tensor) or not (
                self.features.ndim == 2 or (self.features.ndim == 1 and self.features.dtype == torch.long)):
            raise ValueError("planner token features must be [tokens, feature_dim] or [tokens] person numbers")
        for index, *_ in self.shot_tokens:
            if not 0 <= index < self.features.shape[0]:
                raise ValueError("planner token feature index out of range")
        if not self.shot_tokens:
            raise ValueError("a planner token condition needs at least one token; pass None instead")
        if self.geometry is not None and (
                not isinstance(self.geometry, torch.Tensor) or not self.geometry.is_floating_point()
                or self.geometry.ndim != 2 or self.geometry.shape[0] != self.features.shape[0]
                or self.features.dtype != torch.long or self.features.ndim != 1):
            raise ValueError("planner geometry must accompany one person ID per geometry row")

    @property
    def count(self) -> int:
        return len(self.shot_tokens)


def token_placement(*, starts: Sequence[tuple[int, str, Sequence[int] | None]], span_start: int,
                    latent_starts: Sequence[int], shot_lookup: Mapping[tuple[int, str], int],
                    move_point=None) -> dict | None:
    """镜首点 → token 的放置（训练数据集与推理共用）。

    ``starts``：(块内镜下标, 名字, 镜首点或 None)；``shot_lookup``：(镜下标, 名字) → 特征行号；
    ``latent_starts``：目标各镜第一个潜帧下标；``move_point(镜下标, 名字, 点)``：训练时的抖动与丢点（推理为 None），
    返回 None 表示丢掉这个人的点。镜首为 null 的人没有 token；有点、特征里没有的条目报错。返回 None 表示这一段没有 token。"""
    shot_tokens = []
    for shot_index, name, point in starts:
        if (shot_index, name) not in shot_lookup:
            raise ValueError(f"no feature for {name} at shot {shot_index}")
        if point is None:
            continue
        if move_point is not None:
            point = move_point(shot_index, name, point)
            if point is None:
                continue
        k = shot_index - span_start
        row, col = point_cell(point)
        shot_tokens.append((shot_lookup[(shot_index, name)], k, latent_starts[k], row, col))
    if not shot_tokens:
        return None
    return {"shot_tokens": shot_tokens}


def planner_token_states(connector: nn.Module, condition: PlannerTokenCondition, dtype: torch.dtype) -> torch.Tensor:
    """连接器输出，按序列顺序。返回 [1, token 数, dim]。"""
    order = [i for i, *_ in condition.shot_tokens]
    features = condition.features.to(device=next(connector.parameters()).device)[order]
    if features.dtype != torch.long:
        features = features.to(next(connector.parameters()).dtype)
    if isinstance(connector, PersonGeometryEmbedding):
        if condition.geometry is None:
            raise ValueError("person_geometry requires explicit geometry for every person ID row")
        geometry = condition.geometry.to(device=features.device, dtype=next(connector.parameters()).dtype)[order]
        states = connector(features, geometry)
    else:
        if condition.geometry is not None:
            raise ValueError("geometry reached a planner connector without person_geometry content")
        states = connector(features)
    return states.to(dtype).unsqueeze(0)


def geometry_token_features(starts: Sequence[tuple[int, str, Sequence[int] | None]],
                            geometry_rows: Sequence[Mapping[str, Any]], max_people: int) -> torch.Tensor:
    """训推共用：同一镜首尺寸及已知 ``front_of`` 边 → 人物ID槽的数值／有效位。

    列顺序为 width,height,size_valid、M个关系值、M个关系有效位；近端+1，远端−1。
    缺边为0且无效，不是同深度；不补传递闭包，不给人物强排顺序。
    """
    if type(max_people) is not int or max_people < 1:
        raise ValueError("geometry max_people must be a positive int")
    rows = {}
    for row in geometry_rows:
        if (type(row["shot_index"]) is not int or type(row["frame"]) is not int
                or row["frame"] != 0 or row["shot_index"] in rows):
            raise ValueError("geometry needs one explicit frame-zero row per target shot")
        rows[row["shot_index"]] = row
    shots = {shot for shot, _, _ in starts}
    if {shot for shot, row in rows.items() if row["characters"]} != shots:
        raise ValueError("geometry rows must cover exactly the target point shots")
    features = torch.zeros((len(starts), 3 + 2 * max_people), dtype=torch.float32)
    positions = {}
    for index, (shot, name, _) in enumerate(starts):
        match = re.fullmatch(r"Character (\d+)", name)
        if match is None or not 1 <= int(match.group(1)) <= max_people or (shot, name) in positions:
            raise ValueError("geometry needs unique registered Character IDs within max_people for each shot")
        positions[(shot, name)] = (index, int(match.group(1)) - 1)
    for shot, row in rows.items():
        expected = [(name, point) for target, name, point in starts if target == shot]
        if [entity["name"] for entity in row["characters"]] != [name for name, _ in expected]:
            raise ValueError("geometry characters must preserve target point identity and row order")
        for entity, (name, point) in zip(row["characters"], expected):
            if entity["point_2d"] != point:
                raise ValueError("geometry and target points must describe the same frame-zero location")
            index, _ = positions[(shot, name)]
            size, valid = entity["size"], entity["size_valid"]
            if type(valid) is not bool or (valid and size is None) or (not valid and size is not None):
                raise ValueError("geometry size must be explicitly known or null with size_valid=false")
            if valid:
                if (not isinstance(size, (list, tuple)) or len(size) != 2
                        or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1
                               for value in size)):
                    raise ValueError("known geometry size must contain normalized positive width and height")
                features[index, :3] = torch.tensor([*size, 1.0])
        pairs = set()
        for pair in row["front_of"]:
            near, far = pair["near"], pair["far"]
            if (near == far or (shot, near) not in positions or (shot, far) not in positions
                    or (near, far) in pairs or (far, near) in pairs):
                raise ValueError("front_of must contain unique consistent pairs from this target shot")
            pairs.add((near, far))
            for source, other, value in ((near, far, 1.0), (far, near, -1.0)):
                index, _ = positions[(shot, source)]
                _, slot = positions[(shot, other)]
                features[index, 3 + slot] = value
                features[index, 3 + max_people + slot] = 1.0
    return features


def planner_token_freqs(dit: nn.Module, condition: PlannerTokenCondition, *, video_freqs: torch.Tensor,
                        latent_frames: int, tokens_per_frame: int, prefix_frames: int,
                        reference_offset: tuple[int, int, int] | None, shot_phase: bool) -> torch.Tensor:
    """token 的 RoPE：取目标第 t 个潜帧、(行, 列)，镜相位时时间上加镜序 k 的固定相位（与视频 token 同一套）。
    返回 [token 数, 1, 维]。"""
    temporal_dim = video_freqs.shape[-1] - 2 * (video_freqs.shape[-1] // 3)
    freqs = []
    phase_step = None
    if shot_phase and condition.shot_tokens:
        target_frames = latent_frames - (prefix_frames if reference_offset is not None else 0)
        if target_frames < 2:
            raise ValueError("fixed shot phase needs at least two target latent frames")
        step = dit.freqs[0][1, :temporal_dim]
        phase_step = torch.angle(step).to(torch.float64) * 0.5
    for _, k, t, row, col in condition.shot_tokens:
        temporal = dit.freqs[0][t]
        if phase_step is not None:
            temporal = temporal * torch.polar(torch.ones_like(phase_step), k * phase_step).to(temporal.dtype)
        freqs.append(torch.cat([temporal, dit.freqs[1][row], dit.freqs[2][col]]))
    return torch.stack(freqs).unsqueeze(1).to(video_freqs.device)


@dataclass(frozen=True)
class PlannerTokenLayout:
    """块内自注意力需要知道的序列分段：视频 token 数、每个 token 属于哪个目标镜。"""
    video_tokens: int
    shot_token_shots: tuple[int, ...]

    @classmethod
    def of(cls, condition: "PlannerTokenCondition", video_tokens: int) -> "PlannerTokenLayout":
        return cls(video_tokens=video_tokens, shot_token_shots=tuple(k for _, k, *_ in condition.shot_tokens))


def planner_self_attention(attn, q, k, v, *, layout: PlannerTokenLayout):
    """带规划器 token 的全互读自注意力（q/k/v 已加 RoPE）：

    - 全部视频 query（参考和目标）读全部 key（参考、目标、点 token）；
    - 点 token 只读自己，即输出等于自己的 value。
    """
    n = layout.video_tokens
    total = n + len(layout.shot_token_shots)
    if q.shape[1] != total:
        raise ValueError("planner token layout does not match the sequence length")
    video = attn(q[:, :n].contiguous(), k, v)
    return torch.cat([video, v[:, n:total]], dim=1)
