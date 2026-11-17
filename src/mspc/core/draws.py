"""确定性抽签：同一 (tag, key) 永远给同一个 [0,1) 数。训练采样、参考裁剪、评测子集抽取共用这一处。"""
from __future__ import annotations

import hashlib


def unit_draw(tag: str, *key: object) -> float:
    """sha256 只作为确定性抽签函数；不作为身份或凭据。"""
    digest = hashlib.sha256("|".join([tag, *map(str, key)]).encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)
