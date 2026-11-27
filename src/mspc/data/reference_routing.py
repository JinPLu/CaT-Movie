"""参考按镜路由：每张参考给哪些目标镜读。训练（``mspc.training.block_dataset``）与推理
（``mspc.inference.segment_story``）调同一个函数 :func:`reference_readers`；注意力上的用法见 vendored
``SelfAttention.routed_shot_attention``（``routed_self_only_v1``）。

参考 j 来自块／故事里的镜 k：与 k 共有任一在场名字（``present``）的目标镜读它（``entity``）；一镜都不共有时，
全部目标镜读它（``all``）。训练的随机参考会补与目标无关的镜，推理的地点参考（同地点、不共有实体）也会，它们都走
``all``。不按场所路由：块里多半只有一个场所，按场所与「给全部镜」几乎相同。推理不在这之上另加过滤
（11-14 删掉外人路由，见 :mod:`mspc.data.reference_selection`）。
"""
from __future__ import annotations

from collections import Counter
from typing import AbstractSet, Any, Hashable, Mapping, Sequence

ReferenceRouting = tuple[tuple[int, ...], ...]


def reference_slot_metadata(shots: Sequence[Mapping[str, Any]], *, target_shots: Sequence[int],
                            reference_count: int, reference_entities: Sequence[Mapping[str, Any]] | None,
                            routing: ReferenceRouting | None) -> tuple[tuple[dict[str, Any], ...], ...]:
    """实际prefix slot到目标文字的接线；visible ID只取同帧核实结果，present只过滤相关性。"""
    targets = tuple(target_shots)
    if (type(reference_count) is not int or reference_count < 0 or not targets
            or any(type(index) is not int or not 0 <= index < len(shots) for index in targets)):
        raise ValueError("reference slot metadata requires actual prefix count and target shot indices")
    if reference_entities is None:
        visible = [() for _ in range(reference_count)]
    else:
        if len(reference_entities) != reference_count:
            raise ValueError("verified source rows must match the actual reference prefix slots")
        visible = []
        for row in reference_entities:
            names = tuple(entity["name"] for entity in row["entities"])
            if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
                raise ValueError("verified reference entity IDs must be distinct nonempty names")
            if any(entity["identity_status"] not in {"registered", "unresolved"} for entity in row["entities"]):
                raise ValueError("source identity status must explicitly distinguish registered and unresolved")
            visible.append(tuple(entity["name"] for entity in row["entities"]
                                 if entity["identity_status"] == "registered"))
    if routing is not None:
        readers = require_reference_routing(routing, reference_count=reference_count, target_shots=len(targets))
    else:
        readers = tuple(tuple(range(reference_count)) for _ in targets)
    result = []
    for target, slots in zip(targets, readers, strict=True):
        relevant = {entry["name"] for entry in shots[target]["present"]}
        result.append(tuple({"index": slot + 1, "entities": [name for name in visible[slot] if name in relevant]}
                            for slot in slots))
    return tuple(result)


def rank_reference_views(entities: Mapping[int, AbstractSet[Hashable]], needed: AbstractSet[Hashable],
                         frames_max: int) -> list[int]:
    """按视角最少的所需实体贪心补视角，全平取更近的镜；返回选取顺序。"""
    views = dict.fromkeys(needed, 0)
    remaining = {index for index, names in entities.items() if names}
    picked: list[int] = []
    while remaining and len(picked) < frames_max:
        def gain(index: int) -> tuple[tuple[int, ...], int]:
            levels = Counter(views[name] for name in entities[index])
            return tuple(levels[level] for level in range(len(picked) + 1)), index
        index = max(remaining, key=gain)
        picked.append(index)
        remaining.remove(index)
        for name in entities[index]:
            views[name] += 1
    return picked


def select_past_reference_shots(annotation: Mapping[str, Any], *, span: Sequence[int],
                                eligible_past_shots: Sequence[int], frames_max: int) -> list[int]:
    """训推共用的逐目标镜补视角：只读可用过去镜，按深度轮流合并，最终按镜序返回。

    每个目标镜独立按自己的在场人物／物体排序，第一轮各取第一张、第二轮各取第二张；
    同一源镜去重、最多frames_max，不抽随机张数，不补无关镜。RGB中帧由共同frame_index选择。
    """
    if type(frames_max) is not int or not 1 <= frames_max <= 10:
        raise ValueError("frames_max must be an int in 1..10")
    shots = annotation["shots"]
    if (len(span) != 2 or any(type(index) is not int for index in span)
            or not 0 <= span[0] < span[1] <= len(shots)):
        raise ValueError("per-shot coverage fill requires the segment shot span")
    start, end = span
    past = list(eligible_past_shots)
    if (len(set(past)) != len(past) or any(type(index) is not int or not 0 <= index < start for index in past)):
        raise ValueError("eligible reference shots must be unique usable shots before the target span")
    present = [{entry["name"] for entry in shot["present"]} for shot in shots]
    rankings = [rank_reference_views({index: present[index] & present[target] for index in past},
                                    present[target], frames_max) for target in range(start, end)]
    picked: list[int] = []
    for depth in range(frames_max):
        for ranking in rankings:
            if depth < len(ranking) and ranking[depth] not in picked and len(picked) < frames_max:
                picked.append(ranking[depth])
    return sorted(picked)


def route_references(shots: Sequence[Mapping[str, Any]], *, reference_shots: Sequence[int],
                     target_shots: Sequence[int]) -> tuple[ReferenceRouting, tuple[str, ...]]:
    """``shots`` 是标注的 ``shots``；``reference_shots`` 是各参考（按序列顺序）的源镜下标，``target_shots`` 是目标镜下标。

    返回 ``(routing, reasons)``：``routing`` 每个目标镜一项，是路由给它的参考下标（升序）；``reasons`` 每张参考一项。"""
    targets = list(target_shots)
    if not targets or any(type(index) is not int or not 0 <= index < len(shots) for index in [*targets, *reference_shots]):
        raise ValueError("reference routing needs at least one target shot and shot indices inside the annotation")
    names = [{entry["name"] for entry in shot["present"]} for shot in shots]
    routing: list[list[int]] = [[] for _ in targets]
    reasons = []
    for reference, source in enumerate(reference_shots):
        readers = [i for i, target in enumerate(targets) if names[source] & names[target]]
        reasons.append("entity" if readers else "all")
        for i in readers or range(len(targets)):
            routing[i].append(reference)
    return tuple(tuple(entry) for entry in routing), tuple(reasons)


def camera_setup(shot: Mapping[str, Any]) -> tuple[Any, Any, Any] | None:
    """机位＝地点、景别、角度三项（标注 ``place``、``camera.shot_size``、``camera.angle``）。

    没写角度按平视：文字编译器不写平视（``DEFAULT_CAMERA``），规划器计划因此约九成镜不写角度，判官出切镜题也按平视读
    （``question_writer.planned_difference``）。缺地点或景别为 None（未知）。"""
    from mspc.data.shot_text import DEFAULT_ANGLE
    camera = shot.get("camera") or {}
    setup = (shot.get("place"), camera.get("shot_size"), camera.get("angle") or DEFAULT_ANGLE)
    return None if None in setup else setup


def distinct_camera_first(shots: Sequence[Mapping[str, Any]], *, candidates: Sequence[int],
                          target_shots: Sequence[int], count: int, limit: int,
                          same_camera_fallback: bool, fill_unrelated: bool) -> list[int]:
    """异机位优先选参考；是否用同机位兜底（``same_camera_fallback``）与是否用无关镜补满张数（``fill_unrelated``）均显式声明。

    ``candidates`` 已按随机次序排好。每张候选按 :func:`route_references` 的同一规则找它的读者（共有任一在场名字的目标镜）：
    有读者且与每个读者机位都不同 → 第一档；有读者但与某个读者同机位（或任一方机位未知）→ 第二档；没有读者 → 无关镜。
    先取第一档到 ``count`` 张；``same_camera_fallback`` 为真时，某个有在场名字的目标镜仍没有共有名字的参考就从第二档补一张给它；
    ``fill_unrelated`` 为真时剩下名额用无关镜补，为假时不再补无关镜（无异机位共有实体候选的窗口返回空表，自然退化为纯窗内首潜帧互读）。
    总数不超过 ``limit``。
    """
    if type(same_camera_fallback) is not bool or type(fill_unrelated) is not bool:
        raise TypeError("same_camera_fallback and fill_unrelated must be explicit booleans")
    names = [{entry["name"] for entry in shot["present"]} for shot in shots]
    targets = list(target_shots)
    distinct, same, unrelated = [], [], []
    for position in candidates:
        readers = [t for t in targets if names[position] & names[t]]
        if not readers:
            unrelated.append(position)
            continue
        setup = camera_setup(shots[position])
        clash = setup is None or any(camera_setup(shots[t]) in (None, setup) for t in readers)
        (same if clash else distinct).append(position)
    chosen = distinct[:min(count, limit)]
    for target in targets if same_camera_fallback else ():
        if len(chosen) >= limit:
            break
        if names[target] and not any(names[p] & names[target] for p in chosen):
            fallback = [p for p in same if names[p] & names[target] and p not in chosen]
            if fallback:
                chosen.append(fallback[0])
    if fill_unrelated:
        chosen += unrelated[:max(0, min(count, limit) - len(chosen))]
    return sorted(chosen)


def distinct_camera_character_coverage(shots: Sequence[Mapping[str, Any]], *, candidates: Sequence[int],
                                       target_shots: Sequence[int], count: int, limit: int) -> list[int]:
    """异机位优先、按在场实体逐个补齐、只补一张同地点场景镜；不补与本窗无关的镜。

    ``candidates`` 已按随机次序排好，分档同 :func:`distinct_camera_first`（第一档异机位共有实体、第二档同机位或机位未知的
    共有实体、无读者的为无关镜）。先取第一档到 ``count`` 张；再对每个目标镜的每个在场名字（按 ``present`` 顺序）检查已选参考
    里有没有它，没有就先从第一档、再从第二档补一张含它的镜（只用异机位兜底按「整镜」判的旧做法会漏掉双人镜里的第二个人）；
    最后若目标镜有地点，从与某个目标镜同地点的无关镜里补一张在场名字最少的作场景参考（同数按随机次序），对应推理时
    :func:`mspc.data.reference_selection.select_references` 每个目标镜加的一张地点参考。与地点无关、与本窗无共有实体的镜
    一律不选，所以全窗广播的参考只剩这一张同地点场景镜。总数不超过 ``limit``。
    """
    names = [{entry["name"] for entry in shot["present"]} for shot in shots]
    targets = list(target_shots)
    distinct, same, unrelated = [], [], []
    for position in candidates:
        readers = [t for t in targets if names[position] & names[t]]
        if not readers:
            unrelated.append(position)
            continue
        setup = camera_setup(shots[position])
        clash = setup is None or any(camera_setup(shots[t]) in (None, setup) for t in readers)
        (same if clash else distinct).append(position)
    chosen = distinct[:min(count, limit)]
    for target in targets:
        for name in (entry["name"] for entry in shots[target]["present"]):
            if len(chosen) >= limit:
                break
            if any(name in names[p] for p in chosen):
                continue
            pick = next((p for tier in (distinct, same) for p in tier if name in names[p] and p not in chosen), None)
            if pick is not None:
                chosen.append(pick)
    places = {shots[t].get("place") for t in targets} - {None}
    scene = [p for p in unrelated if shots[p].get("place") in places]
    if scene and len(chosen) < limit:
        chosen.append(min(scene, key=lambda p: len(names[p])))
    return sorted(chosen)


def reference_readers(policy: str, shots: Sequence[Mapping[str, Any]], *, reference_shots: Sequence[int],
                      target_shots: Sequence[int]) -> tuple[ReferenceRouting, tuple[str, ...]]:
    """按 ``model.memory_prefix.attention_policy`` 决定每张参考给哪些目标镜读；训练与推理的唯一分派。

    ``window_shared_self_only_v1``：每个目标镜读全部参考（依据记为 ``window``）；两种按镜路由策略：
    :func:`route_references`（共有在场实体）。"""
    from mspc.core.wan22_contract import (MEMORY_ATTENTION_POLICY_ROUTED_FIRST_LATENT_SELF_ONLY_V1,
                                          MEMORY_ATTENTION_POLICY_ROUTED_SELF_ONLY_V1,
                                          MEMORY_ATTENTION_POLICY_WINDOW_SHARED_SELF_ONLY_V1)
    if policy == MEMORY_ATTENTION_POLICY_WINDOW_SHARED_SELF_ONLY_V1:
        targets = list(target_shots)
        if not targets or not reference_shots or any(type(index) is not int or not 0 <= index < len(shots)
                                                     for index in [*targets, *reference_shots]):
            raise ValueError("window-shared references need targets, references and shot indices inside the annotation")
        everyone = tuple(range(len(reference_shots)))
        return tuple(everyone for _ in targets), tuple("window" for _ in reference_shots)
    if policy in (MEMORY_ATTENTION_POLICY_ROUTED_SELF_ONLY_V1, MEMORY_ATTENTION_POLICY_ROUTED_FIRST_LATENT_SELF_ONLY_V1):
        return route_references(shots, reference_shots=reference_shots, target_shots=target_shots)
    raise ValueError(f"{policy!r} is not a routed reference attention policy")


def require_reference_routing(routing: Any, *, reference_count: int, target_shots: int) -> ReferenceRouting:
    """训练后端与 ``model_fn_wan_video`` 两处入口的完整校验：每个目标镜一项、下标严格升序、并集是全部参考（某镜可以
    为空）；返回元组的元组。"""
    if type(reference_count) is not int or reference_count < 1 or type(target_shots) is not int or target_shots < 1:
        raise ValueError("reference routing needs at least one reference and one target shot")
    if isinstance(routing, (str, bytes)) or not isinstance(routing, (list, tuple)) or len(routing) != target_shots:
        raise ValueError(f"reference_routing must list the references of each of the {target_shots} target shot(s); "
                         f"got {routing!r}")
    parsed = []
    for shot, entry in enumerate(routing):
        if (isinstance(entry, (str, bytes)) or not isinstance(entry, (list, tuple))
                or any(type(index) is not int or not 0 <= index < reference_count for index in entry)
                or any(a >= b for a, b in zip(entry, entry[1:]))):
            raise ValueError(f"reference_routing[{shot}] must be strictly increasing reference indices in "
                             f"[0, {reference_count}); got {entry!r}")
        parsed.append(tuple(entry))
    if set().union(*parsed) != set(range(reference_count)):
        raise ValueError(f"reference_routing must route every one of the {reference_count} reference(s) to some "
                         f"target shot; got {routing!r}")
    return tuple(parsed)
