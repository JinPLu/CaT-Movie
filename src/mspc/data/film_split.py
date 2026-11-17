"""按片留出：保持既有场景训练的分组和确定性抽样语义。"""
from mspc.core.draws import unit_draw


def film_group(film_key: str | None, source_video: str) -> str:
    group = film_key or source_video
    if not isinstance(group, str) or not group:
        raise ValueError("film_key or source_video must identify the film group")
    return group


def validation_group(group: str, *, fraction: float, seed: int) -> bool:
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    if type(seed) is not int:
        raise TypeError("split seed must be an int")
    return unit_draw("validation_group", seed, group) < fraction
