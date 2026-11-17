"""取样台账：块一行、镜一行，把各关已落盘的产物按键拼在一起，并数漏斗。

台账是派生视图：每一栏只有一个生产者（某一关已经落盘的产物），这里只读、按键拼、数数，不做任何判断；
信号都由那些产物里的标注答复和测量算出。全部块都有一行（没提交的、没返回的、答复不合法的也有，带状态），
全部镜都有一行（不能用的也有，带原因）。

各关产物与它供的栏：

- 块清单 ``final/blocks.jsonl``（``tools/data/preprocess_sources.py blocks``）：块与镜的身份、帧数、两个质量分。
- 块标注 ``final/block-annotations.jsonl``（``tools/data/validate_annotations.py --unit block``）：状态、登记表、
  每镜的画面类型与镜头与在场与变化、程序改过什么（``repairs``）、删了哪一项（``deletions``）、重问过哪一镜（``reasks``）、
  每镜的筛选结果（``shot_filter``）。``reasks`` 现在没有写入方（切片与代码里都是空字典）；读它只取键，按 ``deletions`` 的写法
  当镜号字符串，有键就记"重问过"。
- 评测排除 ``final/eval-exclude-shots.jsonl``：与评测重复的镜。装块时这些镜已经被排出块清单，
  所以正常情况下没有一行对得上块里的镜；对得上就说明排除清单比块清单新，那一镜记为不能用。
- 装块被拒的候选 ``final/block-rejections.jsonl``：这些镜不在任何块里，没有标注、潜变量可拼，
  所以不进两张表，只在汇总里按原因计数。
- 目标潜变量索引、参考帧库清单、空间层根：这三关可以还没有产物（配置里写 ``null``），对应的栏就是 ``null``（不知道），
  不是 ``false``。有产物而这一镜不在里面，才是 ``false`` 或 0。

镜的键是 ``(sid, scene_id, source_shot_index)``，块清单每镜自带这三项；目标潜变量索引同名，参考帧库清单里 ``sid`` 叫 ``source_id``。
块的键是块清单的 ``key``，块标注、空间层目录都用它。
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping

from mspc.data.atomic_publish import publish_file

LEDGER = "sampling_ledger_v1"
ANSWERED = ("ok", "invalid")
ANSWER_STATUSES = (*ANSWERED, "no_answer")
STAGE_STATES = ("succeeded", "failed")

# 配置里每个输入一项路径，相对数据根；值为 True 的可以写 null，表示这一关还没有产物。
INPUTS = {"blocks": False, "block_annotations": False, "eval_exclude_shots": False, "block_rejections": False,
          "target_index": True, "reference_frames": True, "spatial_root": True}
CONFIG_KEYS = {"schema_version", "pipeline_config", "annotation_contract", "inputs", "output_dir"}

# 台账自己写的不能用的原因；筛选规则的原因（画面类型、叠加文字）原样取自标注行的 ``shot_filter``。
NOT_SUBMITTED = "整块没提交标注"
NO_ANSWER = "整块没有答复"
BLOCK_INVALID = "整块答复不合法"
SHOT_UNREPAIRED = "这一镜修不好"
EVAL_OVERLAP = "与评测重复"


def check_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError(f"config keys must be exactly {sorted(CONFIG_KEYS)}")
    if config["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    for name in ("pipeline_config", "annotation_contract", "output_dir"):
        if not isinstance(config[name], str) or not config[name]:
            raise ValueError(f"{name} must be a nonempty string")
    inputs = config["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != set(INPUTS):
        raise ValueError(f"inputs must be exactly {sorted(INPUTS)}")
    for name, nullable in INPUTS.items():
        value = inputs[name]
        if not (isinstance(value, str) and value) and not (nullable and value is None):
            raise ValueError(f"inputs.{name} must be a nonempty path" + (" or null" if nullable else ""))
    return config


def resolve_inputs(inputs: Mapping[str, str | None], root: Path) -> dict[str, Path | None]:
    return {name: None if value is None else root / value for name, value in inputs.items()}


def read_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def shot_key(shot: Mapping[str, Any]) -> tuple[str, str, int]:
    return shot["sid"], shot["scene_id"], shot["source_shot_index"]


# ---------------------------------------------------------------- 每关一个读函数

def read_blocks(path: Path) -> Iterator[dict]:
    """块清单，按文件顺序一块一块给；键重复、镜数对不上即报错。"""
    seen = set()
    for row in read_jsonl(path):
        key = row["key"]
        if key in seen:
            raise ValueError(f"{path}: duplicate block key {key}")
        seen.add(key)
        scene = row["scene"]
        if len(scene["shots"]) != scene["shot_count"] or not scene["shot_count"]:
            raise ValueError(f"{key}: shot_count differs from the shot list")
        for name in ("case_id", "family", "source"):
            if not isinstance(row[name], str):
                raise TypeError(f"{key}: {name} must be a string")
        yield row


def read_annotations(path: Path) -> dict[str, dict]:
    """块标注（校验后的行）。``raw_answer`` 与用量台账不读，这里就不留。"""
    rows: dict[str, dict] = {}
    for row in read_jsonl(path):
        key = row["key"]
        if key in rows:
            raise ValueError(f"{path}: duplicate annotation key {key}")
        if row["status"] not in ANSWER_STATUSES:
            raise ValueError(f"{key}: unknown status {row['status']!r}")
        keep = {name: row[name] for name in ("key", "status", "shot_count", "from", "block_errors", "shot_errors",
                                             "shot_filter", "repairs", "deletions", "reasks", "annotation")}
        for name, kind in (("block_errors", list), ("shot_errors", dict), ("shot_filter", list),
                           ("repairs", list), ("deletions", dict), ("reasks", dict)):
            if not isinstance(keep[name], kind):
                raise TypeError(f"{key}: {name} must be a {kind.__name__}")
        if keep["block_errors"] or row["status"] == "no_answer":
            # 登记表不合法时答复可以是任何形状（真实行里有整份答复是 JSON 数组的），台账不读它。
            keep["annotation"] = None
        elif not isinstance(keep["annotation"], dict):
            raise TypeError(f"{key}: the registry is valid but the answer is not an object")
        if row["status"] == "no_answer":
            keep.update(error_code=row["error_code"], error=row["error"])
        rows[key] = keep
    return rows


def read_eval_exclusions(path: Path) -> dict[tuple[str, str, int], dict]:
    rows = {}
    for row in read_jsonl(path):
        key = shot_key(row)
        if key in rows:
            raise ValueError(f"{path}: duplicate shot {key}")
        rows[key] = {"reason": row["reason"]}  # ``source`` 不是每行都有（真实行里按规则直接排除的没有），不读
    return rows


def read_rejections(path: Path) -> list[dict]:
    return [{"key": shot_key(row), "reason": row["reason"], "reasons": row["reasons"]} for row in read_jsonl(path)]


def read_target_index(path: Path | None) -> dict[tuple[str, str, int], int] | None:
    """目标潜变量索引：镜的键 → 训练帧数。``None`` 表示这一关还没有产物。"""
    if path is None:
        return None
    rows = {}
    for row in read_jsonl(path):
        key = shot_key(row)
        if key in rows:
            raise ValueError(f"{path}: duplicate shot {key}")
        rows[key] = row["train_frames"]
    return rows


def read_reference_frames(path: Path | None) -> dict[tuple[str, str, int], tuple[int, int]] | None:
    """参考帧库清单（``manifest.jsonl``）：镜的键 → (训练帧数, 候选帧张数)。``None`` 表示这一关还没有产物。"""
    if path is None:
        return None
    rows = {}
    for row in read_jsonl(path):
        key = (row["source_id"], row["scene_id"], row["source_shot_index"])
        if key in rows:
            raise ValueError(f"{path}: duplicate shot {key}")
        rows[key] = (row["train_frames"], row["N"])
    return rows


def _stage(directory: Path, name: str) -> dict | None:
    path = directory / "stages" / f"{name}.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["state"] not in STAGE_STATES:
        raise ValueError(f"{path}: unknown state {record['state']!r}")
    return record


def read_spatial(root: Path | None, key: str, shot_count: int) -> list[dict[str, bool]] | None:
    """空间层：这个块每镜有没有深度、有没有掩码。``None`` 表示这一关还没有产物；块的目录不在或阶段没做完，是 ``False``。

    读各阶段自己落的记录：深度阶段成功且产物里有这一镜的 ``depth.mp4``；组装阶段成功，且这一镜
    ``instances.json`` 里至少一个登记名有掩码帧。不再逐个文件核对在位。
    """
    if root is None:
        return None
    directory = root / key
    none = [{"depth": False, "masks": False} for _ in range(shot_count)]
    if not directory.is_dir():
        return none
    depth, assemble = _stage(directory, "depth"), _stage(directory, "assemble")
    products = set(depth["products"]) if depth and depth["state"] == "succeeded" else set()
    instances = None
    if assemble and assemble["state"] == "succeeded":
        if "instances.json" not in assemble["products"]:
            raise ValueError(f"{directory}: assemble succeeded without instances.json")
        instances = json.loads((directory / "instances.json").read_text(encoding="utf-8"))["shots"]
    result = []
    for number in range(1, shot_count + 1):
        shot_id = f"s{number:02d}"
        masks = instances is not None and any(entry["frames_with_mask"] > 0 for entry in instances[shot_id].values())
        result.append({"depth": f"depth/{shot_id}/depth.mp4" in products, "masks": masks})
    return result


# ---------------------------------------------------------------- 拼行

def registry_valid(annotation: Mapping[str, Any] | None) -> bool | None:
    """登记表那一层（整块错误为空）合不合法。没有答复就没有可判的，是 ``None``。"""
    if annotation is None or annotation["status"] == "no_answer":
        return None
    return not annotation["block_errors"]


def _answer(annotation: Mapping[str, Any], shot_count: int) -> dict:
    answer = annotation["annotation"]
    if not isinstance(answer, dict) or len(answer["shots"]) != shot_count:
        raise ValueError(f"{annotation['key']}: registry is valid but the answer does not hold {shot_count} shots")
    return answer


def _registry(answer: Mapping[str, Any]) -> dict[str, tuple[str, dict]]:
    return {entry["name"]: (kind, entry) for kind in ("characters", "objects", "places") for entry in answer[kind]}


def _bad_shots(annotation: Mapping[str, Any], shot_count: int) -> set[int]:
    numbers = {int(number) for number in annotation["shot_errors"]}
    if any(not 1 <= number <= shot_count for number in numbers):
        raise ValueError(f"{annotation['key']}: shot_errors do not match the shot count")
    return numbers


def _other_outfit(entry: Mapping[str, Any], registered: Mapping[str, Any]) -> bool:
    """在场条目写了穿着，且与登记表里的默认穿着不同（不分大小写与首尾空格）。"""
    if "outfit" not in entry:
        return False
    default = registered.get("outfit")
    return not (isinstance(default, str) and default.strip().casefold() == entry["outfit"].strip().casefold())


def _shot_facts(shot: Mapping[str, Any], registry: Mapping[str, tuple[str, dict]]) -> dict[str, Any]:
    """一镜的答复里算得出的栏；调用者保证这一镜没有剩下的错误。镜头与场所三栏在非连续画面的镜上可以没有。"""
    camera, place = shot.get("camera", {}), shot.get("place")
    characters = [(registry[entry["name"]][1], entry) for entry in shot["present"]
                  if registry[entry["name"]][0] == "characters"]
    people = [registered for registered, _ in characters if registered["kind"] == "person"]
    return {
        "picture_type": shot["picture_type"], "on_screen_text": shot["on_screen_text"], "activity": shot["activity"],
        "shot_size": camera.get("shot_size"), "angle": camera.get("angle"), "movement": camera.get("movement"),
        "place": place, "place_appearance": registry[place][1]["appearance"] if place else None,
        "present_people": len(people),
        "present_clear_people": sum(person["visibility"] == "clear" for person in people),
        "present_objects": sum(registry[entry["name"]][0] == "objects" for entry in shot["present"]),
        "other_outfit_worn": any(_other_outfit(entry, registered) for registered, entry in characters),
        "outfit_change": any(change["what"] == "outfit" for change in shot["changes"]),
        "changes": len(shot["changes"]),
    }


def _answer_origin(annotation: Mapping[str, Any], number: int) -> dict[str, Any]:
    """这一镜的答复：原样返回／程序改过（改了什么规则、删了哪些位置）／重问过。重问优先，其次程序改过。"""
    rules = [fix["rule"] for fix in annotation["repairs"] if fix.get("shot") == number]
    deleted = [item["path"] for item in annotation["deletions"].get(str(number), [])]
    reasked = str(number) in annotation["reasks"]
    origin = "reasked" if reasked else "program_edited" if rules or deleted else "as_returned"
    return {"answer_origin": origin, "repair_rules": rules, "deleted_paths": deleted}


SHOT_FACTS = ("picture_type", "on_screen_text", "activity", "shot_size", "angle", "movement", "place", "place_appearance",
              "present_people", "present_clear_people", "present_objects", "other_outfit_worn", "outfit_change", "changes")


def shot_rows(block: Mapping[str, Any], annotation: Mapping[str, Any] | None, *,
              eval_exclusions: Mapping[tuple, Any], target_index: Mapping[tuple, int] | None,
              reference_frames: Mapping[tuple, tuple[int, int]] | None,
              spatial: list[dict[str, bool]] | None) -> list[dict]:
    """块里每镜一行。

    不能用的原因不互斥，全列：整块层（没提交、没答复、登记表不合法）、这一镜自己（修不好）、筛选规则（原样取自 ``shot_filter``）、
    与评测重复。有答复但登记表不合法的块，各镜的标注栏都是 ``null``；登记表合法而这一镜修不好的，只有标注栏是 ``null``。
    """
    scene, status = block["scene"], annotation["status"] if annotation else "not_submitted"
    valid = registry_valid(annotation)
    shot_count = scene["shot_count"]
    answer = registry = bad = None
    if valid:
        answer = _answer(annotation, shot_count)
        registry, bad = _registry(answer), _bad_shots(annotation, shot_count)
        if len(annotation["shot_filter"]) != shot_count:
            raise ValueError(f"{block['key']}: shot_filter does not hold {shot_count} shots")
    rows = []
    for number, shot in enumerate(scene["shots"], 1):
        key = shot_key(shot)
        reasons, marks = [], []
        if annotation is None:
            reasons.append(NOT_SUBMITTED)
        elif status == "no_answer":
            reasons.append(NO_ANSWER)
        elif not valid:
            reasons.append(BLOCK_INVALID)
        else:
            judged = annotation["shot_filter"][number - 1]
            if judged["shot"] != number:
                raise ValueError(f"{block['key']}: shot_filter is out of order at shot {number}")
            reasons += ([SHOT_UNREPAIRED] if number in bad else []) + judged["reasons"]
            marks = judged["marks"]
        excluded = key in eval_exclusions
        if excluded:
            reasons.append(EVAL_OVERLAP)
        row = {"block_key": block["key"], "shot": number, "sid": key[0], "scene_id": key[1], "source_shot_index": key[2],
               "source": block["source"], "family": block["family"],
               "frames": shot["render_frames"], "seconds": shot["seconds"],
               "aesthetic": shot["laion_aesthetic"], "sharpness": shot["dover_technical"],
               "usable": not reasons, "unusable_reasons": reasons, "marks": marks, "eval_excluded": excluded}
        if valid and number not in bad:
            if answer["shots"][number - 1]["shot"] != number:
                raise ValueError(f"{block['key']}: answer shots are out of order at shot {number}")
            row.update(_shot_facts(answer["shots"][number - 1], registry))
        else:
            row.update(dict.fromkeys(SHOT_FACTS))
        row.update(_answer_origin(annotation, number) if valid else
                   {"answer_origin": None, "repair_rules": [], "deleted_paths": []})
        if target_index is None:
            row["target_latents"] = None
        else:
            row["target_latents"] = key in target_index
            if row["target_latents"] and target_index[key] != shot["render_frames"]:
                raise ValueError(f"{key}: target index holds {target_index[key]} frames, block manifest {shot['render_frames']}")
        if reference_frames is None:
            row["reference_frames"] = None
        else:
            row["reference_frames"] = 0
            if key in reference_frames:
                frames, count = reference_frames[key]
                if frames != shot["render_frames"]:
                    raise ValueError(f"{key}: reference library holds {frames} frames, block manifest {shot['render_frames']}")
                row["reference_frames"] = count
        row["has_depth"] = None if spatial is None else spatial[number - 1]["depth"]
        row["has_masks"] = None if spatial is None else spatial[number - 1]["masks"]
        rows.append(row)
    return rows


def block_row(block: Mapping[str, Any], annotation: Mapping[str, Any] | None, shots: list[dict], *,
              contract: str) -> dict:
    """一块一行。登记表不合法或没有答复的块，登记表那几栏是 ``null``（数出来也不可靠）。"""
    scene = block["scene"]
    status = annotation["status"] if annotation else "not_submitted"
    valid = registry_valid(annotation)
    row = {"key": block["key"], "case_id": block["case_id"], "family": block["family"], "source": block["source"],
           "sid": scene["shots"][0]["sid"], "film_key": scene["film_key"], "scene_id": scene["scene_id"],
           "shot_count": scene["shot_count"], "frames": scene["render_frames"],
           "seconds": sum(shot["seconds"] for shot in scene["shots"]),
           "annotation_status": status, "contract": contract if annotation else None,
           "answer_from": annotation["from"] if annotation else None,
           "no_answer_error": {"code": annotation["error_code"], "message": annotation["error"]}
           if status == "no_answer" else None,
           "registry_valid": valid, "block_errors": annotation["block_errors"] if annotation else []}
    row.update(people=None, clear_people=None, animals=None, groups=None, objects=None, places=None,
               valid_shots=None, characters=None)
    if valid:
        answer = _answer(annotation, scene["shot_count"])
        bad = _bad_shots(annotation, scene["shot_count"])
        people = [entry for entry in answer["characters"] if entry["kind"] == "person"]
        # 出现在哪几镜：只看答复没有剩下错误的镜（有错误的镜里在场名单不可信）。
        characters = [{"name": entry["name"], "kind": entry["kind"], "visibility": entry["visibility"],
                       "shots": [number for number, shot in enumerate(answer["shots"], 1) if number not in bad
                                 and any(item["name"] == entry["name"] for item in shot["present"])]}
                      for entry in answer["characters"]]
        row.update(people=len(people), clear_people=sum(entry["visibility"] == "clear" for entry in people),
                   animals=sum(entry["kind"] == "animal" for entry in answer["characters"]),
                   groups=sum(entry["kind"] == "group" for entry in answer["characters"]),
                   objects=len(answer["objects"]), places=len(answer["places"]),
                   valid_shots=scene["shot_count"] - len(bad), characters=characters)
    row["usable_shots"] = sum(shot["usable"] for shot in shots)
    answered = annotation is not None and status != "no_answer"
    row.update(repairs=dict(Counter(fix["rule"] for fix in annotation["repairs"])) if answered else None,
               deleted_items=sum(len(items) for items in annotation["deletions"].values()) if answered else None,
               reasked_shots=sorted(int(number) for number in annotation["reasks"]) if answered else None)
    return row


# ---------------------------------------------------------------- 汇总

def _tri(value: Any) -> str:
    return "null" if value is None else "true" if value else "false"


def _paired(blocks_path: Path, shots_path: Path) -> Iterator[tuple[dict, list[dict]]]:
    groups = itertools.groupby(read_jsonl(shots_path), key=lambda row: row["block_key"])
    for block in read_jsonl(blocks_path):
        key, group = next(groups)
        if key != block["key"]:
            raise ValueError(f"shots.jsonl and blocks.jsonl disagree at {block['key']}")
        yield block, list(group)
    if next(groups, None) is not None:
        raise ValueError("shots.jsonl has blocks that blocks.jsonl lacks")


def _group() -> dict[str, Any]:
    names = ("annotation_status", "registry_valid", "block_funnel", "shot_funnel", "unusable_reasons", "answer_origin",
             "target_latents", "reference_frames", "has_depth", "has_masks", "repairs", "no_answer_errors")
    return {"blocks": 0, "shots": 0, **{name: Counter() for name in names}, "deleted_items": 0, "reasked_shots": 0,
            "registry_valid_blocks_with_unrepaired_shots": 0}


def _funnel(names: tuple[str, ...], steps: tuple[bool, ...]) -> list[str]:
    """依次累计：一步没过，后面各步都不算。"""
    passed = []
    for name, ok in zip(names, steps, strict=True):
        if not ok:
            break
        passed.append(name)
    return passed


def _tally(group: dict, block: Mapping[str, Any], shots: list[dict]) -> None:
    group["blocks"] += 1
    group["shots"] += len(shots)
    group["annotation_status"][block["annotation_status"]] += 1
    group["registry_valid"][_tri(block["registry_valid"])] += 1
    steps = (True, block["annotation_status"] != "not_submitted", block["annotation_status"] in ANSWERED,
             block["registry_valid"] is True, block["usable_shots"] > 0)
    group["block_funnel"].update(_funnel(BLOCK_STEPS, steps))
    group["registry_valid_blocks_with_unrepaired_shots"] += block["registry_valid"] is True and block["valid_shots"] < block["shot_count"]
    if block["no_answer_error"]:
        group["no_answer_errors"][block["no_answer_error"]["message"]] += 1
    group["repairs"].update(block["repairs"] or {})
    group["deleted_items"] += block["deleted_items"] or 0
    group["reasked_shots"] += len(block["reasked_shots"] or [])
    for shot in shots:
        reasons = set(shot["unusable_reasons"])
        steps = (True, block["annotation_status"] != "not_submitted", block["annotation_status"] in ANSWERED,
                 block["registry_valid"] is True, SHOT_UNREPAIRED not in reasons, not reasons - {EVAL_OVERLAP}, shot["usable"])
        group["shot_funnel"].update(_funnel(SHOT_STEPS, steps))
        group["unusable_reasons"].update(reasons)
        group["answer_origin"][shot["answer_origin"] or "null"] += 1
        for name in ("target_latents", "has_depth", "has_masks"):
            group[name][_tri(shot[name])] += 1
        group["reference_frames"][_tri(None if shot["reference_frames"] is None else shot["reference_frames"] > 0)] += 1


BLOCK_STEPS = ("blocks", "submitted", "answered", "registry_valid", "has_usable_shot")
SHOT_STEPS = ("shots", "block_submitted", "block_answered", "registry_valid", "shot_answer_valid", "passes_shot_filter", "usable")


def _plain(group: Mapping[str, Any]) -> dict:
    return {name: dict(sorted(value.items())) if isinstance(value, Counter) else value for name, value in group.items()}


def summarize(blocks_path: Path, shots_path: Path, *, contract: str, joins: Mapping[str, Any],
              rejections: list[dict]) -> dict:
    """数两张表：总的与按来源的，各状态、各原因、漏斗各步的块数与镜数；再加各关对账（``joins``）与装块被拒的候选。

    漏斗各步依次累计（前一步没过，后面不再算）；``unusable_reasons`` 不互斥，一镜有几条原因就各计一次。
    装块被拒的候选没有 ``source``，按 ``sid`` 的前缀（``film`` 是 cinepile_t2v 与 moviechat_1k，``cinedance``）分。
    """
    total, by_source = _group(), {}
    for block, shots in _paired(blocks_path, shots_path):
        _tally(total, block, shots)
        _tally(by_source.setdefault(block["source"], _group()), block, shots)
    by_reason, by_family = Counter(), {}
    for row in rejections:
        reason = "+".join(row["reasons"])
        by_reason[reason] += 1
        by_family.setdefault(row["key"][0].rsplit("-", 1)[0], Counter())[reason] += 1
    return {"ledger": LEDGER, "contract": contract, "joins": joins,
            "block_rejections": {"rows": len(rejections), "by_reason": dict(sorted(by_reason.items())),
                                 "by_family": {family: dict(sorted(counts.items())) for family, counts in sorted(by_family.items())}},
            "total": _plain(total), "by_source": {source: _plain(group) for source, group in sorted(by_source.items())}}


# ---------------------------------------------------------------- 总装

def build(inputs: Mapping[str, Path | None], out_dir: Path, *, contract: Mapping[str, Any]) -> dict:
    """读各关产物，写 ``blocks.jsonl``、``shots.jsonl``、``summary.json``。三份都写成临时文件、汇总成功后才改名。"""
    if contract["kind"] != "block_annotation":
        raise ValueError("annotation_contract is not a block annotation contract")
    annotations = read_annotations(inputs["block_annotations"])
    names_in_contract = {"reasons": {rule["name"] for rule in contract["shot_filter"]},
                         "marks": {rule["name"] for rule in contract["shot_mark"]}}
    for annotation in annotations.values():
        for judged in annotation["shot_filter"]:
            for field, allowed in names_in_contract.items():
                if not set(judged[field]) <= allowed:
                    raise ValueError(f"{annotation['key']}: shot_filter {field} {judged[field]} are not in the contract")
    exclusions = read_eval_exclusions(inputs["eval_exclude_shots"])
    rejections = read_rejections(inputs["block_rejections"])
    target, references = read_target_index(inputs["target_index"]), read_reference_frames(inputs["reference_frames"])
    if inputs["spatial_root"] is not None and not inputs["spatial_root"].is_dir():
        raise FileNotFoundError(inputs["spatial_root"])
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ("blocks.jsonl", "shots.jsonl", "summary.json")
    staged = {name: out_dir / (name + ".partial") for name in names}
    answered, matched, block_total = set(), {"eval": set(), "target": set(), "reference": set()}, 0
    with staged["blocks.jsonl"].open("w", encoding="utf-8") as blocks_out, staged["shots.jsonl"].open("w", encoding="utf-8") as shots_out:
        for block in read_blocks(inputs["blocks"]):
            key = block["key"]
            block_total += 1
            annotation = annotations.get(key)
            if annotation is not None:
                answered.add(key)
                if annotation["shot_count"] != block["scene"]["shot_count"]:
                    raise ValueError(f"{key}: annotation holds {annotation['shot_count']} shots, block manifest {block['scene']['shot_count']}")
            spatial = read_spatial(inputs["spatial_root"], key, block["scene"]["shot_count"])
            shots = shot_rows(block, annotation, eval_exclusions=exclusions, target_index=target,
                              reference_frames=references, spatial=spatial)
            for shot in shots:
                identity = (shot["sid"], shot["scene_id"], shot["source_shot_index"])
                for name, hit in (("eval", shot["eval_excluded"]), ("target", shot["target_latents"]),
                                  ("reference", shot["reference_frames"])):
                    if hit:
                        matched[name].add(identity)
                shots_out.write(json.dumps(shot, ensure_ascii=False) + "\n")
            blocks_out.write(json.dumps(block_row(block, annotation, shots, contract=contract["contract"]), ensure_ascii=False) + "\n")
    if set(annotations) - answered:
        raise ValueError(f"{len(set(annotations) - answered)} annotation rows are not in the block manifest")
    rejected = {row["key"] for row in rejections}
    joins = {"block_annotations": {"rows": len(annotations), "blocks_without_row": block_total - len(answered)},
             "eval_exclusions": {"rows": len(exclusions), "in_blocks": len(matched["eval"]),
                                 "in_block_rejections": len(exclusions.keys() & rejected),
                                 "in_neither": len(exclusions.keys() - matched["eval"] - rejected),
                                 "by_reason": dict(sorted(Counter(row["reason"] for row in exclusions.values()).items()))},
             "target_index": None if target is None else {"rows": len(target), "in_blocks": len(matched["target"]),
                                                          "not_in_blocks": len(target) - len(matched["target"])},
             "reference_frames": None if references is None else {"rows": len(references), "in_blocks": len(matched["reference"]),
                                                                  "not_in_blocks": len(references) - len(matched["reference"])}}
    summary = summarize(staged["blocks.jsonl"], staged["shots.jsonl"], contract=contract["contract"], joins=joins,
                        rejections=rejections)
    staged["summary.json"].write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name in names:
        publish_file(staged[name], out_dir / name)
    return summary
