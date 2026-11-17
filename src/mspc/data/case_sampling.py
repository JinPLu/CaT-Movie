"""定额、确定性的 case_id 抽样。

生产者问的是同一件事——「同一个 seed 与同一个候选池，永远给同一批案例，重跑不换池」：
``tools/annotate/build_state_assertions.py`` 抽穿帮断言的案例池。若有第二处各写一份 sha256
排序，seed 相同却抽出两批案例的那一天不会有人发现，所以排序函数只留这一处。

``sha256`` 在这里**只是排序函数**，不是收据、不是身份层：它把 case_id 打散成一个与语料
顺序无关、与 seed 绑定的次序，取前 ``size`` 条之后再按 case_id 排回去。
"""

from __future__ import annotations

import hashlib
from typing import Sequence


def sample_case_ids(case_ids: Sequence[str], *, size: int, seed: str) -> list[str]:
    """定额抽样，确定性的。``sha256`` 只当排序函数用，不是任何收据或身份层。

    同一个 ``seed`` 与同一个候选池永远给同一批案例，所以重跑不会换池；换 ``seed`` 才换。
    候选不足 ``size`` 时直接抛——悄悄少给几条会让下游的分母变成一个没人知道的数。
    """

    if size < 1:
        raise ValueError(f"--sample-size must be >= 1, got {size}")
    if len(case_ids) < size:
        raise ValueError(
            f"only {len(case_ids)} eligible case(s) but --sample-size={size}; "
            "widen the pool or lower the size rather than shipping a smaller denominator"
        )
    ordered = sorted(
        case_ids,
        key=lambda case_id: hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest(),
    )
    return sorted(ordered[:size])


__all__ = ["sample_case_ids"]
