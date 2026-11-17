"""读图提示里人物的辨认描述（外貌加该镜穿着），以及一段里需要从历史读取的回场角色。"""
from __future__ import annotations

from typing import Any, Mapping


def character_phrase(annotation: Mapping[str, Any], name: str, shot_index: int) -> str:
    person = next(entry for entry in annotation["characters"] if entry["name"] == name)
    present = next((entry for entry in annotation["shots"][shot_index]["present"] if entry["name"] == name), {})
    outfit = present.get("outfit") or person.get("outfit")
    return person["appearance"] if not outfit else f"{person['appearance']}, wearing {outfit}"


def needed_characters(annotation: Mapping[str, Any], start: int, end: int) -> list[tuple[str, str]]:
    """本段在场、且在更早镜里出现过的登记角色 → (名字, 外貌加本段首次在场那一镜的穿着)，按登记顺序。"""
    registry = [entry["name"] for entry in annotation["characters"]]
    earlier = {entry["name"] for shot in annotation["shots"][:start] for entry in shot["present"]}
    first_here: dict[str, int] = {}
    for index in range(start, end):
        for entry in annotation["shots"][index]["present"]:
            first_here.setdefault(entry["name"], index)
    return [(name, character_phrase(annotation, name, first_here[name]))
            for name in registry if name in first_here and name in earlier]
