"""推理端按分镜选参考帧（锚视角）：每个在场实体在外观不变期间读同一张最干净的早期视角，地点读一张人少的远景。

规则（只读分镜表，确定性，没有阈值）：

1. 选哪几镜（:func:`select_references`）：对窗内每个目标镜，按 ``present`` 顺序取每个在场的登记实体（名字属于登记的
   ``characters`` 或 ``objects``），在候选镜里它出现过的镜中按 :func:`anchor_key` 取最小：
   早于该实体最近一次外观变化（目标镜之前最后一个记了它 ``outfit`` 或 ``state`` 变化的镜；变化镜本身取末帧，
   算变化后）的排后 → 同框其他登记角色少的优先 →
   角色看景别（特写、中近景、中景优先，大特写、中远景其次，远景最后；物体不看景别）→ 同分取最早。
   目标镜有地点（``place``）时，再取同地点候选镜里同框登记角色最少、景别最远、最早的一镜（:func:`place_key`）。
   按「目标镜顺序；每个目标镜内先实体（``present`` 顺序）、最后地点」收集，同一源镜只取一次，保留前 ``frames_max`` 张，
   按镜序返回。
2. 每张参考取哪一帧（:func:`reference_position`）：源镜有衣着或状态变化取末帧，否则中帧（只有姿态或手持物变化取中帧）。
3. 谁读哪张：与训练同一个 :func:`mspc.data.reference_routing.reference_readers`，推理不另加过滤。

训练不调用本模块的选镜：训练参考是同块随机抽（``mspc.training.block_dataset.REFERENCE_POLICY``），锚视角选出的参考
在这个分布之内。

为什么从「每个实体取最近一次出现」换成锚视角：

- 看片【实测，定稿权重开发集种子 42】：最近一次出现常落在群像、远景或背影上（cinedance-015642 镜 11 读洞穴远景、
  film-004063 镜 4 读体育馆远景）；最近一次出现是上一窗刚生成的镜，走样顺着参考链传（cinedance-014617 角色 2 在镜 4 已
  走样，镜 5 照着镜 4 画）。
- CPU 演练【实测，开发集分镜表 30 个故事、种子 42 的窗划分、读者 ``routed_self_only_v1``】：角色在参考里独占画面
  32.0% → 48.9%，参考是三人以上群像 33.2% → 24.1%，读取对含外人 48.6% → 34.8%，读到的参考离第一窗隔的参考层数平均
  1.31 → 0.58，每个角色一路用过的不同视角 2.51 → 1.60 张；每窗 1.85 → 2.26 张，缺视角两边都是 0。分镜随机标错
  （景别错一档、漏标或多标一人）时独占画面仍有 46.4%。
- 近邻【读知识库卡片】：StoryMem 把早期关键帧固定为长期锚（memory sink），去掉后跨镜一致下降；SubjectAnchor 每个主体
  回溯到最早出现的那一镜；CANVAS 去掉标准角色锚后脸、衣服、身体掉得最多。
- 只有姿态变化不取末帧：film-000123 镜 3 记了「转身看身后」，末帧是背影。
- 选帧消融「变化镜之后的镜比变化镜本身优先」：
  起因是 film-000127 定稿种子 42 第 10 镜读了第 8 镜（换装镜）末帧，又穿回长裤。改法在 ``9eb9965ff`` 里，开发集只改变
  5 个故事的选帧，这 5 个故事×三种子另出（``configs/render_batches/wan22-final-dev30-settled-appearance-20261114.json``）。
  种子 42、44 读数比本规则差【实测，11-15】：判官平均 −4.1［−5.3, −2.2］，人物 −10.9［−16.7, −2.3］，故不采用。

选帧消融「首次出场一起给」：
每个实体除锚视角外再给它外观不变期间的首次出场，地点除地点视角外再给它的首次出场，代码留在 ``6dfa2ffa0``（只含人物与
物体的前一版 ``563408ada`` 未出片），开发集受影响的 20 个故事×三种子另出
（``configs/render_batches/wan22-final-dev30-first-appearance-20261115.json``）。依据：测试集定稿行跨窗人物题里，
目标镜读的参考「源镜自己就和被比镜不一致」的 67 道只过 13.4%，其中 62 道是被比的首次出场同框人更多、锚视角改取了后面
人少却已画走样的一镜。读数出来前主线仍是锚视角。

不处理的【推断】：同机位照抄（锚视角下 38.2%，略高于最近一次出现），按原决定在判分里看照抄次数；窗前从没出现过的
实体给不了视角。
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

#: 景别由近到远的序数；分镜没写景别（``camera`` 字段可省）按中景算。
SHOT_DISTANCE = {"extreme close-up shot": 0, "close-up shot": 1, "medium close-up shot": 2, "medium shot": 3,
                 "medium wide shot": 4, "wide shot": 5, "extreme wide shot": 6, None: 3}
#: 角色看得清脸和衣着的档：特写、中近景、中景 0，大特写、中远景 1，远景 2。
FACE_LEGIBILITY = {0: 1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 2}
#: 改变实体外观、之后要继承的变化（衣着、状态）：选镜时早于它的视角排后，源镜有这类变化取末帧。姿态与手持物不算：
#: 拿起东西不该把锚换到更挤的镜（cinedance-012473 镜 2 拿起物体时角色在走廊远处），转身的末帧可能是背影。
APPEARANCE_CHANGES = ("outfit", "state")


def _distance(shot: Mapping[str, Any]) -> int:
    return SHOT_DISTANCE[(shot.get("camera") or {}).get("shot_size")]


def anchor_key(annotation: Mapping[str, Any], *, candidate: int, target: int, name: str) -> tuple[int, int, int, int]:
    """实体 ``name`` 在候选镜 ``candidate`` 的视角作目标镜 ``target`` 参考的排序键，越小越好。"""
    shots, characters = annotation["shots"], {entry["name"] for entry in annotation["characters"]}
    changed = [index for index in range(target) if any(
        change["name"] == name and change["what"] in APPEARANCE_CHANGES for change in shots[index].get("changes", ()))]
    stale = int(bool(changed) and candidate < changed[-1])
    others = len({entry["name"] for entry in shots[candidate]["present"]} & characters - {name})
    legibility = FACE_LEGIBILITY[_distance(shots[candidate])] if name in characters else 0
    return stale, others, legibility, candidate


def place_key(annotation: Mapping[str, Any], *, candidate: int) -> tuple[int, int, int]:
    """同地点候选镜作地点参考的排序键：同框登记角色少、景别远、早的优先。"""
    shot, characters = annotation["shots"][candidate], {entry["name"] for entry in annotation["characters"]}
    return len({entry["name"] for entry in shot["present"]} & characters), -_distance(shot), candidate


def select_references(annotation: Mapping[str, Any], *, span: Sequence[int], candidates: Sequence[int],
                      frames_max: int) -> list[int]:
    """返回选中的源镜下标（升序）。``span`` 是目标镜 ``(start, end)``；``candidates`` 是可作参考的镜（不能与目标段相交）。"""
    if type(frames_max) is not int or not 1 <= frames_max <= 10:
        raise ValueError("frames_max must be an int in 1..10")
    shots = annotation["shots"]
    if len(span) != 2 or any(type(index) is not int for index in span) or not 0 <= span[0] < span[1] <= len(shots):
        raise ValueError("reference selection requires the target shot span")
    start, end = span
    pool = list(candidates)
    if len(set(pool)) != len(pool) or any(type(i) is not int or not 0 <= i < len(shots) or start <= i < end for i in pool):
        raise ValueError("reference candidates must be unique shots outside the target span")
    registered = {entry["name"] for entry in [*annotation["characters"], *annotation.get("objects", ())]}
    names = [{entry["name"] for entry in shot["present"]} for shot in shots]

    picked: list[int] = []
    for target in range(start, end):
        found = []
        for entry in shots[target]["present"]:
            appearances = [index for index in pool if entry["name"] in names[index]]
            if entry["name"] in registered and appearances:
                found.append(min(appearances, key=lambda index: anchor_key(
                    annotation, candidate=index, target=target, name=entry["name"])))
        place = shots[target].get("place")
        same_place = [index for index in pool if place and shots[index].get("place") == place]
        if same_place:
            found.append(min(same_place, key=lambda index: place_key(annotation, candidate=index)))
        for index in found:
            if index not in picked:
                picked.append(index)
    return sorted(picked[:frames_max])


def reference_position(shot: Mapping[str, Any]) -> str:
    """有衣着或状态变化的镜取末帧，否则中帧（只有姿态或手持物变化取中帧）。"""
    return "last" if any(change["what"] in APPEARANCE_CHANGES for change in shot.get("changes", ())) else "middle"
