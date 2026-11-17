"""旧 event_log_v3 benchmark 合同与样本固件构造；不再承接当前规划器训练或生成。

仅冻结的旧 benchmark、其零样本答复和历史评分使用这个合同。当前入口位于 planner_dataset.py。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from mspc.core.wan22_contract import SEGMENT_MAX_FRAMES
from mspc.data.event_log import (
    EVENT_LOG_SCHEMA, dependency_segment_spans, parse_event_log, validate_event_log,
)

MAX_HISTORY_FRAMES = 8
# 请求合同唯一owner；schema和枚举直接来自原消费端，规则不另造校验器。
PLANNER_CONTRACT_VERSION = "event_log_v3_explicit_request_v5"
PLANNER_RULES = {
    "output": "根据规划请求输出全故事唯一完整event_log_v3 JSON，包括历史镜与目标镜之后。只输出紧凑JSON，不要Markdown或额外键；所有schema.required字段必填，类型严格，不用字符串代替整数、布尔或数组。虚构示例仅解释格式，不是当前故事事实。",
    "input_lock": "请求给出shots（逐镜剧本），镜号集合、顺序与每镜frame_count锁定，逐字保留。剧本里的Character N就是subjects里的Subject N（同一个N，人物和动物写Character，物体写Subject）；剧本点名的Subject N/Location N编号原样沿用，其余主体、地点由你登记。target_start_shot_id之前的镜是历史，history_frames列出带历史关键帧的镜；历史像素是观测证据，不得改写剧本计划，不假造生成历史。",
    "ids": "各列表按顺序连续编号：subjects的subject_id为Subject 1、Subject 2…；locations的location_id为Location 1、Location 2…；shots的shot_id为s01、s02…；events为e1、e2…。锚点完整ID为Location N/Anchor M（N/M为无前导零的正整数），归属所在Location，锚点ID全局唯一；不可只写Anchor 1或自然名。",
    "presence": "shot.location_id只能是已登记Location ID或unknown。present_subjects只含已登记Subject，visible_entities可含已登记Subject/Location/完整Anchor ID，两个列表分别不重复；可见Subject必须在present_subjects，可见Anchor必须属于本镜location_id。关系各端点、所有fact.subject都须已登记；不可凭未登记的自由名称引用物体。",
    "appearance": "subjects.appearance只写静态身份，持物、姿态、湿/开合等写事实。现行静态检查会拒绝这些整词/片段：holds, holding, carrying, wet, opened, closed, sitting, seated, standing, walking, running, lying, operating, handling, launching，以及拿着、持有、穿上、湿了、站着、坐着、躺着。",
    "atomic": "每个AtomicFact严格包含subject/family/predicate_or_axis/object_or_value/scope/shot_span；值已知时再带evidence，值为unknown时不带evidence。family只取state/relation/screen，不写轴名或relation_groups分组名。shot_span恰好两个整数，为从0开始的闭区间[start,end]，0<=start<=end<镜头数；单镜必须[i,i]，不是[i]，也不是s01字符串。",
    "evidence": "evidence只取observed/carried/unknown：observed为有证据的声明，carried为按保持规则延续，unknown为不可判。值为unknown时不写evidence字段（连键一起省略；校验器拒收带evidence的unknown值，序列化也不输出它）。已知事实必须写evidence，且值和evidence都不是unknown；unknown不覆盖已有已知值，不等于否定/不存在。遮挡、出画或换机位不自动成为状态事件。",
    "state": "state只在initial_state或事件端点，scope=scene；轴全集见登记表。environment的subject必须是Location；costume/body/object_condition必须是Subject，不能是Anchor。值为非空、无首尾空白的自然语言字符串。锚点可变状态另登记Subject。",
    "relation": "relation只在initial_state或事件端点，scope=scene；谓词须精确匹配登记表（非空、无首尾空白），不能用开放未注册谓词。object_or_value为登记ID或ID数组；between恰好两个不同目标，其他谓词恰好一个。禁止自指、重复目标及开镜/镜末in包含环。touches只单镜且不能carried；in front of/behind涉及的锚点必须有非null的front_definition。",
    "screen": "screen只在对应shots[i].screen_facts，不在initial_state/events；subject必须是本镜visible_entities中的Subject；scope=screen@i，shot_span=[i,i]，evidence不得carried。轴和值全集见登记表，未知写unknown。bbox为[ x1,y1,x2,y2 ]四个0–1000整数、正面积x1<x2/y1<y2，不是字符串。gaze为不同于自身的登记ID或unknown；orientation为登记表值、toward Subject N（另一已登记Subject）或unknown。screen按镜重新给，不继承；bbox不进入编译文字。",
    "initial": "initial_state允许中途首次声明，以shot_span起点进入时间线；同键在重叠闭区间的已知声明不能互相冲突。后续不同的已知观测更新同键，不必虚构变化发生时刻。相同事实比较subject/family/predicate_or_axis/object_or_value，不靠改evidence或span制造变化。",
    "carry": "每镜先延续已知持久事实并标carried，丢弃touches；换地点清除environment及任一端点绑定旧Location/Anchor的事实，holds/wears/carries可随人走。再应用本镜起点initial_state已知声明，得到事件前基线；unknown不擦除已知值。",
    "event": "events.before/after是完整AtomicFact或null，不是FactRecord包装；至少一端非null，非null端点必须是已知state/relation，不能screen或unknown。null是已确认的增删端点；before=null表示该键在事件前基线中不存在，不是看不清。before必须匹配本镜所有事件之前的已知基线；after相同不构成事件。两个非null端点必须同fact key，转交/换关系目标须拆成旧键删除和新键增加。事件shot_id须登记，各非null端点的shot_span须包含事件镜。",
    "event_visibility": "on_screen只取true/false/unknown。只有on_screen=true时，before/after的全部端点ID（fact.subject及关系全部目标）都必须在事件镜visible_entities、并已登记，缺一不可；可见Subject还须present。不要为通过检查把未知或不可见端点强写可见。on_screen=false/unknown不要求所有端点可见，但端点仍须已知且登记。",
    "event_uniqueness": "同一shot_id、同一fact key最多一个事件，不能用不同event_id重复该键。on_screen=true在本镜开头保留before、镜末应用after；false/unknown从本镜开头应用after。不使用旧版effective_from，也不把未知变化写成确认删除。",
    "duration": "各镜frame_count之和必须等于输入给定的total_frames；每镜frame_count为4n+1帧（n为非负整数），单镜不超过228帧。",
    "token_budget": "编译后每条完整参考行/目标镜行（含Design/State/Screen/参考说明）最多512个umT5 tokens，超限整故事拒绝、不截断。写简明准确内容；输入未提供真实分词器，不能自行声称已通过这项编译门。",
}
PLANNER_SYSTEM = "你是故事视觉规划器。\n" + "\n".join(
    f"[{key}] {rule}" for key, rule in PLANNER_RULES.items()
)


def planner_format_example() -> dict:
    """本次手写的虚构天文台灯笼故事；不读取语料、长集或评测金标。"""
    def fact(subject, family, axis, value, shot, *, evidence="observed"):
        # unknown 值不带 evidence 字段（AtomicFact 校验）。
        return dict(subject=subject, family=family, predicate_or_axis=axis,
                    object_or_value=value, scope=f"screen@{shot}" if family == "screen" else "scene",
                    shot_span=[shot, shot], **({} if value == "unknown" else {"evidence": evidence}))
    subject, animal, lantern = "Subject 1", "Subject 2", "Subject 3"
    place, bench, post = "Location 1", "Location 1/Anchor 1", "Location 1/Anchor 2"
    closed = fact(lantern, "state", "object_condition", "sealed", 0)
    hold = fact(subject, "relation", "holds", lantern, 0)
    shots = []
    for i, action in enumerate(("A caretaker displays a triangular lantern beside a white hare.",
            "The caretaker opens the lantern and places it on the stone bench.",
            "The open lantern rests on the bench beside the caretaker and hare.")):
        screen = [fact(subject, "screen", axis, value, i) for axis, value in (
            ("horizontal", "center"), ("depth", "foreground"),
            ("orientation", "toward Subject 2"), ("gaze", lantern), ("bbox", [300,100,650,950]))]
        screen.append(fact(animal, "screen", "gaze", "unknown", i, evidence="unknown"))
        shots.append(dict(shot_id=f"s{i+1:02d}", location_id=place,
            present_subjects=[subject,animal,lantern], visible_entities=[subject,animal,lantern,place,bench,post],
            screen_facts=screen, frame_count=33, shot_size="medium", camera_angle="eye level", movement="unknown",
            action=action))
    return {"schema":EVENT_LOG_SCHEMA, "subjects":[
        {"subject_id":subject,"kind":"person","appearance":"caretaker with silver hair"},
        {"subject_id":animal,"kind":"animal","appearance":"white hare with long ears"},
        {"subject_id":lantern,"kind":"object","appearance":"teal triangular lantern"}],
        "locations":[{"location_id":place,"appearance":"stone observatory","anchors":[
            {"id":bench,"description":"stone bench","front_definition":"toward the central aisle"},
            {"id":post,"description":"stone post","front_definition":None}]}],
        "initial_state":[fact(subject,"state","costume","amber tunic",0),
            fact(subject,"state","body","uninjured",0),closed,
            fact(place,"state","environment","dimly lit",0),hold,
            fact(animal,"relation","between",[bench,post],0),
            fact(subject,"state","costume","amber tunic",1,evidence="carried"),
            fact(animal,"state","body","unknown",0,evidence="unknown")],
        "shots":shots, "events":[
            {"event_id":"e1","shot_id":"s02","before":dict(closed,shot_span=[1,1]),
             "after":fact(lantern,"state","object_condition","open",1),"on_screen":True},
            {"event_id":"e2","shot_id":"s02","before":dict(hold,shot_span=[1,1]),"after":None,"on_screen":True},
            {"event_id":"e3","shot_id":"s02","before":None,
             "after":fact(lantern,"relation","on",bench,1),"on_screen":True}]}


FORMAT_EXAMPLE_SCRIPT = """SUBJECTS
- Character 1 (person): caretaker with silver hair
- Character 2 (animal): white hare with long ears
- Subject 3 (object): teal triangular lantern
LOCATIONS
- Location 1: stone observatory
SHOTS
s01: Medium shot in the dimly lit stone observatory. In the center foreground, Character 1, in an amber tunic, holds the sealed Subject 3 and looks at it, turned toward Character 2, who sits between the stone bench and the stone post.
s02: Medium shot. Character 1 opens Subject 3 and sets it on the stone bench.
s03: Medium shot. The open Subject 3 rests on the stone bench beside Character 1 and Character 2."""


def planner_output_contract() -> dict:
    """供新benchmark冻结；不得修改旧benchmark或历史评分来换合同。"""
    from mspc.data.event_log import EventLog, registry
    example = planner_format_example()
    request = planner_request(script=FORMAT_EXAMPLE_SCRIPT, event_log=example,
                              target_start_shot_id="s01", history_shot_ids=())
    return {"contract_version":PLANNER_CONTRACT_VERSION,
        "schema":EventLog.model_json_schema(),
        "rules":PLANNER_SYSTEM + "\n消费端原始登记表（轴/谓词/值全集）：\n" + registry().model_dump_json(),
        "example":[{"role":"user","content":"以下是虚构格式示例，与实际输入无关；展示三类事实、四种state轴、screen轴与事件增删替换。\n"
                    + planner_request_text(request)},
                   {"role":"assistant","content":json.dumps(example,ensure_ascii=False,separators=(",",":"))}]}


def planner_contract_messages(output_contract: Mapping) -> list[dict]:
    """冻结合同必须来自唯一owner；拒绝旧规则/空示例，禁止静默改写冻结输入。"""
    if output_contract != planner_output_contract():
        raise ValueError("frozen planner output_contract differs from current owner; explicitly freeze a new benchmark")
    return [{"role":"system","content":output_contract["rules"] + "\nJSON schema：" +
             json.dumps(output_contract["schema"],ensure_ascii=False)}, *output_contract["example"]]


def planner_request(*, script: str, event_log: Mapping[str, Any],
                    target_start_shot_id: str, history_shot_ids: Sequence[str]) -> dict[str, Any]:
    """规划请求的唯一构造处。event_log 只用来取总帧数和锁定的镜号与帧数，不进请求的其余部分。

    字段：script（逐镜剧本）、total_frames、shots（镜号与帧数，锁定）、target_start_shot_id、history_frames
    （带历史关键帧的镜号；像素由消费端按同一顺序附上）。
    """
    if not isinstance(script, str) or not script.strip():
        raise ValueError("script must be a nonempty string")
    shots = [{"shot_id": shot["shot_id"], "frame_count": shot["frame_count"]} for shot in event_log["shots"]]
    order = [shot["shot_id"] for shot in shots]
    if target_start_shot_id not in order:
        raise ValueError(f"target_start_shot_id {target_start_shot_id!r} is not a shot of the log")
    history = list(history_shot_ids)
    if (len(set(history)) != len(history) or len(history) > MAX_HISTORY_FRAMES
            or any(shot not in order or order.index(shot) >= order.index(target_start_shot_id) for shot in history)):
        raise ValueError(f"history frames must be distinct earlier shots, at most {MAX_HISTORY_FRAMES}")
    return {"script": script, "total_frames": sum(shot["frame_count"] for shot in shots), "shots": shots,
            "target_start_shot_id": target_start_shot_id, "history_frames": history}


def planner_request_text(request: Mapping[str, Any]) -> str:
    return "规划请求：" + json.dumps(request, ensure_ascii=False)


def planner_user_content(request: Mapping[str, Any], images: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """用户消息：请求文字，再按 history_frames 顺序逐张附历史关键帧。images 为空表示只给文字（无图臂）。

    图像部件由消费端按各自格式给出（训练用 processor 的 {"type":"image"}，服务端用 image_url），文字逐字相同。
    """
    history = request["history_frames"]
    if images and len(images) != len(history):
        raise ValueError("history images must be absent or match history_frames one to one")
    content: list[dict[str, Any]] = [{"type": "text", "text": planner_request_text(request)}]
    for shot_id, image in zip(history, images):
        content.extend([{"type": "text", "text": f"历史关键帧 {shot_id}："}, dict(image)])
    return content


def normalize_planner_answer(payload: Any) -> Any:
    """规划器答复进入校验前的唯一归一（范围只限这一种）：值为 unknown 的事实若带了 evidence，丢掉该字段。
    下游把值为 unknown 的事实当作不存在，序列化本来也不输出 evidence，所以写了也不影响结果；提示仍要求不写
    （``PLANNER_RULES["evidence"]``）。标注流水线的 ``AtomicFact`` 校验不动，仍然拒收：那边的合同是另一回事。"""
    if isinstance(payload, dict):
        if payload.get("object_or_value") == "unknown" and "evidence" in payload:
            payload = {key: value for key, value in payload.items() if key != "evidence"}
        return {key: normalize_planner_answer(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [normalize_planner_answer(item) for item in payload]
    return payload


def planner_lock_issues(request: Mapping[str, Any], log: Mapping[str, Any]) -> list[str]:
    """输入锁与时长规则：总帧数相等；每镜 4n+1 帧且不超过段帧上限；镜号、顺序、帧数与请求锁定的逐一相等。"""
    issues = []
    counts = [shot["frame_count"] for shot in log["shots"]]
    if sum(counts) != request["total_frames"]:
        issues.append(f"frame_count sum {sum(counts)} differs from total_frames {request['total_frames']}")
    for shot, count in zip(log["shots"], counts):
        if (count - 1) % 4 or count > SEGMENT_MAX_FRAMES:
            issues.append(f"{shot['shot_id']}: frame_count {count} is not 4n+1 within {SEGMENT_MAX_FRAMES}")
    locked = [(shot["shot_id"], shot["frame_count"]) for shot in request["shots"]]
    if [(shot["shot_id"], shot["frame_count"]) for shot in log["shots"]] != locked:
        issues.append("locked shot ids, order or frame counts changed")
    return issues


EXAMPLE_KEYS = {"sample_id", "case_id", "film_key", "script", "target_start_shot_id",
                "history_frames", "event_log"}


def validate_planner_example(row: Mapping[str, Any], *, require_files: bool) -> None:
    if set(row) != EXAMPLE_KEYS:
        raise ValueError(f"planner example keys differ: missing={EXAMPLE_KEYS-set(row)}, extra={set(row)-EXAMPLE_KEYS}")
    for key in ("sample_id", "case_id", "film_key", "script", "target_start_shot_id"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    validate_event_log(parse_event_log(row["event_log"]))
    if not isinstance(row["history_frames"], list):
        raise TypeError("history_frames must be a list")
    for frame in row["history_frames"]:
        if set(frame) != {"shot_id", "path"}:
            raise ValueError("history frame requires exactly shot_id and path")
        if not isinstance(frame["path"], str) or not frame["path"]:
            raise ValueError("history frame path must be a nonempty string")
        if require_files and not Path(frame["path"]).is_file():
            raise FileNotFoundError(frame["path"])
    try:
        example_request(row)
    except ValueError as exc:
        raise ValueError(f"{row['sample_id']}: {exc}; history frame must precede the target segment") from exc


def example_request(row: Mapping[str, Any]) -> dict[str, Any]:
    return planner_request(script=row["script"], event_log=row["event_log"],
                           target_start_shot_id=row["target_start_shot_id"],
                           history_shot_ids=[frame["shot_id"] for frame in row["history_frames"]])


def build_planner_examples(
    results: Sequence[Mapping[str, Any]], writer_rows: Sequence[Mapping[str, Any]],
    frame_ledger: Sequence[Mapping[str, Any]], film_keys: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """每条已接受日志：写手的逐镜剧本保留的出 1 条从 s01 起、不带历史的样本；依赖段第 2 段起每段再出 1 条，
    历史取段首之前各镜的末帧（最近的至多 8 张），缺任一张就拒收该样本。"""
    writer_index = {row["case_id"]: row for row in writer_rows}
    if len(writer_index) != len(writer_rows):
        raise ValueError("duplicate writer case_id")
    frames: dict[str, dict[str, str]] = {}
    for item in frame_ledger:
        if item["status"] == "ok" and item["position"] == "last":
            shot_id = f"s{item['shot_id']:02d}"
            case_frames = frames.setdefault(item["case_id"], {})
            if shot_id in case_frames:
                raise ValueError("duplicate last-frame evidence")
            case_frames[shot_id] = item["path"]
    examples, rejected = [], []
    for result in results:
        case_id = result["case_id"]
        if result["status"] != "accepted":
            rejected.append({"case_id": case_id, "reason": "source_log_not_accepted", "status": result["status"]})
            continue
        if case_id not in writer_index:
            rejected.append({"case_id": case_id, "reason": "missing_writer_row"})
            continue
        writer = writer_index[case_id]
        if writer["status"] != "kept":
            rejected.append({"case_id": case_id, "reason": "script_discarded"})
            continue
        log = parse_event_log(result["event_log"])
        validate_event_log(log)

        def example(target: str, history: list[dict[str, str]]) -> dict[str, Any]:
            row = {"sample_id": f"{case_id}/{target}", "case_id": case_id, "film_key": film_keys[case_id],
                   "script": writer["text"], "target_start_shot_id": target,
                   "history_frames": history, "event_log": result["event_log"]}
            validate_planner_example(row, require_files=False)
            return row

        examples.append(example(log.shots[0].shot_id, []))
        for start, _ in dependency_segment_spans(log)[1:]:
            target = log.shots[start].shot_id
            before = [shot.shot_id for shot in log.shots[:start]][-MAX_HISTORY_FRAMES:]
            missing = [shot for shot in before if shot not in frames.get(case_id, {})]
            if missing:
                rejected.append({"case_id": case_id, "target_start_shot_id": target,
                                 "reason": "missing_history_frame", "shots": missing})
                continue
            examples.append(example(target, [{"shot_id": shot, "path": frames[case_id][shot]} for shot in before]))
    return examples, rejected
