"""规划器文本检查：检查点只从请求定义，逐条看规划器原答（以及编译修补过的文字）保住了什么。

确定性规则能判的由程序判：合同、请求登记的实体、镜数、各镜在场、跨镜身份延续、布局合法、回看合法。
要语义判断的（事件、主体与受体、事件先后、末态、与请求矛盾的新编造）交文字判官；检查点先冻结，
判官只回答这些检查点，不从答案里另立检查点。

判官是 DeepSeek 官方接口的 DeepSeek-V4.1-Flash（模型名 ``deepseek-flash``；与被评的 Qwen 规划器不同家）。密钥从环境变量 ``DEEPSEEK_API_KEY`` 读，不进命令行、日志与回执。
冻结检查点（梗概与段落由 gpt-6.1-sol 只读请求写出）是题库，不随判官换。
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import http.client
import os
import urllib.request
from typing import Any, Iterable, Mapping, Sequence

TIERS = ("synopsis", "paragraph", "shots")
MODEL, EFFORT = "deepseek-flash", "high"
EFFORTS = ("disabled", "low", "high", "max")  # 官方文档：思考默认开、档位默认 high；关思考用 thinking: {type: disabled}
ENDPOINT = "https://api.deepseek.com/chat/completions"
MAX_TOKENS = 65536  # 思考模式默认上限；含推理在内，回答本身只有几百 token
MAX_WORKERS = 64  # DeepSeek 官方接口不限并发档位
REGISTRY_LINE = re.compile(r"^- (Character \d+|Object \d+) \(", re.M)
LOCATION_LINE = re.compile(r"^- (Location \d+):", re.M)
SHOT_LINE = re.compile(r"^s(\d+): ", re.M)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def dev_requests(requests: Path, split: str) -> dict[str, dict]:
    catalog = json.loads(Path(requests).read_text())
    stories = {story["story_id"]: story for story in catalog["stories"] if story["split"] == split}
    if not stories:
        raise ValueError(f"no stories in split {split}")
    return stories


# ---------------------------------------------------------------- 检查点


def shots_checkpoints(questions: Mapping, stories: Mapping[str, Mapping]) -> dict[str, dict]:
    """逐镜档：冻结题库里写题模型只读请求写下的题（事件、末态、跨镜身份）与各镜在场名单。"""
    result = {}
    for story_id, story in stories.items():
        entry = questions["stories"][story_id]
        if entry["request"] != story["inputs"]["shots"]["text"]:
            raise ValueError(f"{story_id}: question bank was written for a different shots request")
        result[story_id] = {"source": "question_bank", "requirements": entry["requirements"],
                            "presence": entry["writer_output"]["shots"]}
    return result


WRITER_SCHEMA = {"type": "json_schema", "json_schema": {"name": "request_checkpoints", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["events"], "properties": {"events": {
        "type": "array", "items": {"type": "object", "additionalProperties": False,
                                   "required": ["question", "doer", "receiver", "result"], "properties": {
            "question": {"type": "string"}, "doer": {"type": "string"},
            "receiver": {"type": ["string", "null"]}, "result": {"type": ["string", "null"]}}}}}}}}


WRITER_SYSTEM = """You prepare checkpoints for judging whether a written storyboard keeps the user request. The request alone decides what is required; a storyboard may add details the request leaves open.

List, in the order the request tells them, every event the request describes that acts on an object or on another person, or moves the whole body (standing up, sitting down, walking away, falling, riding). Steps that serve one goal are one event. Leave out facial expressions, gaze, speaking, listening, camera and background.

For each event give:
- question: a short yes/no question about that one event, in the request's own words ("Does the man hand the black case to the other man?").
- doer: who does it, in the request's words.
- receiver: the person or object it acts on, in the request's words; null if none.
- result: if the event leaves something changed afterwards (the case is now in his hands, the door is open, she is seated), a yes/no question about that new state; otherwise null.

Output one json object with the key "events".

EXAMPLE JSON OUTPUT:
{"events": [{"question": "Does the man hand the black case to the other man?", "doer": "the man", "receiver": "the black case", "result": "Is the black case now in the other man's hands?"}]}"""


def writer_prompt(request: str) -> tuple[str, str]:
    return WRITER_SYSTEM, f"USER REQUEST\n{request}"


def semantic_checkpoints(events: Sequence[Mapping]) -> list[dict]:
    rows = []
    for index, event in enumerate(events, 1):
        rows.append({"requirement_id": f"e{index:02d}", "dimension": "event", "order": index,
                     "question": event["question"], "doer": event["doer"], "receiver": event["receiver"]})
        if event["result"] is not None:
            rows.append({"requirement_id": f"e{index:02d}r", "dimension": "state", "order": index,
                         "question": event["result"], "doer": event["doer"], "receiver": None})
    return rows


# ---------------------------------------------------------------- 程序检查


def _names(board: Mapping, registry: str) -> list[str]:
    return [entry.get("name") for entry in board.get(registry) or [] if isinstance(entry, Mapping)]


def _present(shot: Mapping) -> set[str]:
    """在场名单加在场人物手里的东西：原答常只把持物写进 holding，编译器才补进 present。"""
    entries = [entry for entry in shot.get("present") or [] if isinstance(entry, Mapping)]
    held = {name for entry in entries for name in (entry.get("holding") or []) if isinstance(name, str)}
    return {entry.get("name") for entry in entries} | held


def _count(units: Iterable[bool]) -> dict:
    units = list(units)
    return {"pass": sum(units), "applicable": len(units)}


def layout_units(board: Mapping) -> list[dict]:
    """每镜每实体一条：框在画面内且有正面积、每实体每镜至多 2 个时刻且递增、实体已登记且在该镜在场。"""
    if "layouts" not in board:
        return []
    shots = {shot.get("shot"): shot for shot in board.get("shots") or [] if isinstance(shot, Mapping)}
    registered = set(_names(board, "characters")) | set(_names(board, "objects"))
    units = []
    for layout in board["layouts"] if isinstance(board["layouts"], list) else []:
        shot = shots.get(layout.get("shot")) if isinstance(layout, Mapping) else None
        for entity in (layout.get("entities") or []) if isinstance(layout, Mapping) else []:
            anchors = entity.get("anchors") if isinstance(entity, Mapping) else None
            reasons = []
            if shot is None:
                reasons.append("shot_missing")
            if entity.get("name") not in registered:
                reasons.append("unregistered")
            elif shot is not None and entity.get("name") not in _present(shot):
                reasons.append("not_present")
            if not isinstance(anchors, list) or not 1 <= len(anchors) <= 2:
                reasons.append("anchor_count")
                anchors = anchors if isinstance(anchors, list) else []
            times = [anchor.get("time") for anchor in anchors if isinstance(anchor, Mapping)]
            if len(times) != len(anchors) or any(type(t) not in (int, float) or not 0 <= t <= 1 for t in times) \
                    or any(a >= b for a, b in zip(times, times[1:])):
                reasons.append("anchor_time")
            for anchor in anchors:
                box = anchor.get("box") if isinstance(anchor, Mapping) else None
                visible = isinstance(anchor, Mapping) and anchor.get("visibility") == "visible"
                if visible and not (isinstance(box, list) and len(box) == 4
                                    and all(type(v) in (int, float) and 0 <= v <= 1 for v in box)
                                    and box[0] < box[2] and box[1] < box[3]):
                    reasons.append("box")
                elif not visible and (not isinstance(anchor, Mapping) or anchor.get("visibility") != "not_visible"
                                      or box is not None):
                    reasons.append("visibility")
            units.append({"shot": layout.get("shot"), "entity": entity.get("name"), "pass": not reasons,
                          "reasons": sorted(set(reasons))})
    return units


def reference_units(board: Mapping) -> list[dict]:
    """每条回看一条：指向更早一镜，该实体在那一镜和当前镜都在场；另记是否是最近一次出场。"""
    shots = [shot for shot in board.get("shots") or [] if isinstance(shot, Mapping)]
    by_number = {shot.get("shot"): shot for shot in shots}
    units = []
    for shot in shots:
        if "references" not in shot:
            continue
        for reference in shot["references"] if isinstance(shot["references"], list) else []:
            name, target = reference.get("name"), reference.get("shot")
            earlier = type(target) is int and type(shot.get("shot")) is int and target < shot["shot"]
            appeared = earlier and target in by_number and name in _present(by_number[target])
            here = name in _present(shot)
            latest = appeared and not any(name in _present(by_number[n]) for n in range(target + 1, shot["shot"])
                                          if n in by_number)
            units.append({"shot": shot.get("shot"), "entity": name, "reference_shot": target,
                          "pass": bool(earlier and appeared and here), "most_recent": bool(latest)})
    return units


def program_checks(record: Mapping, tier: str, request: str, checkpoints: Mapping | None) -> dict:
    raw = record.get("raw_storyboard")
    result = {"status": record["status"], "contract": {
        "json_parsed": isinstance(raw, Mapping),
        "compiled_valid": record.get("structure_status") == "valid" and isinstance(record.get("storyboard"), Mapping),
        "contract_accepted": record["status"] == "completed",
        "truncated": record.get("finish_reason") == "length"}}
    compilation = record.get("compilation") or {}
    result["compile_repairs"] = {key: len(compilation.get(key) or [])
                                 for key in ("holding_presence_added", "opening_aligned_with_changes")}
    if not isinstance(raw, Mapping):
        return result
    shots = [shot for shot in raw.get("shots") or [] if isinstance(shot, Mapping)]
    by_number = {shot.get("shot"): shot for shot in shots}
    checks = {}
    layouts, references = layout_units(raw), reference_units(raw)
    if "layouts" in raw:
        checks["layout_legal"] = {**_count(unit["pass"] for unit in layouts), "units": layouts}
    if any("references" in shot for shot in shots):
        checks["reference_legal"] = {**_count(unit["pass"] for unit in references),
                                     "most_recent": sum(unit["most_recent"] for unit in references), "units": references}
    if tier == "shots":
        request_shots = [int(n) for n in SHOT_LINE.findall(request)]
        if request_shots != list(range(1, len(request_shots) + 1)):
            raise ValueError("shots request must number its shots s01..sNN")
        checks["shot_count_kept"] = _count([len(shots) == len(request_shots)])
        registered = set(_names(raw, "characters")) | set(_names(raw, "objects")) | set(_names(raw, "places"))
        ids = REGISTRY_LINE.findall(request) + LOCATION_LINE.findall(request)
        checks["request_entities_registered"] = _count(name in registered for name in ids)
        presence = []
        for row in checkpoints["presence"]:
            shot = by_number.get(row["shot"])
            presence += [shot is not None and name in _present(shot) for name in row["present"]]
        checks["request_presence_kept"] = _count(presence)
        identity = []
        for requirement in checkpoints["requirements"]:
            if requirement["dimension"] not in ("character", "object", "place"):
                continue
            a, b = by_number.get(requirement["shot"]), by_number.get(requirement["reference_shot"])
            if requirement["dimension"] == "place":
                place = re.search(r"\((Location \d+)\)", requirement["question"]).group(1)
                identity.append(a is not None and b is not None and a.get("place") == place == b.get("place"))
            else:
                name, = requirement["depends_on"]
                identity.append(a is not None and b is not None and name in _present(a) and name in _present(b))
        checks["identity_continuity"] = _count(identity)
    result["checks"] = checks
    return result


# ---------------------------------------------------------------- 文字判官


JUDGE_SCHEMA = {"type": "json_schema", "json_schema": {"name": "planner_text_checks", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["checkpoints", "contradictions"], "properties": {
        "checkpoints": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                   "required": ["id", "status", "shot", "roles"], "properties": {
            "id": {"type": "string"}, "status": {"type": "string", "enum": ["stated", "contradicted", "absent"]},
            "shot": {"type": ["integer", "null"]},
            "roles": {"type": "string", "enum": ["correct", "wrong", "not_applicable"]}}}},
        "contradictions": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                      "required": ["shot", "plan", "request"], "properties": {
            "shot": {"type": ["integer", "null"]}, "plan": {"type": "string"}, "request": {"type": "string"}}}}}}}}


def semantic_requirements(tier: str, checkpoints: Mapping) -> list[dict]:
    if tier == "shots":
        return [row for row in checkpoints["requirements"] if row["dimension"] in ("event", "state")]
    return list(checkpoints["requirements"])


JUDGE_SYSTEM = """You check a written storyboard plan against the user request it was made from. The request alone decides what is required; the plan may add details the request leaves open. The user message gives the request, the plan, and checkpoints that were written from the request before any plan was read.

For each checkpoint decide what the plan writes:
- stated: the plan writes this event or result (paraphrase is fine; the plan uses ids such as Character 1 for people the request describes).
- contradicted: the plan writes something incompatible with it (another doer or receiver, the opposite direction, the result undone, the event replaced by a different one).
- absent: the plan does not write it.
Give the plan shot number where it is stated or contradicted, or null when absent. For event checkpoints give roles: correct when the doer and the receiver match the request, wrong when either is another person or object; use not_applicable for results and for absent checkpoints.

Then list contradictions: anything the plan writes that conflicts with the request (an identity, appearance, place, action, order or outcome the request fixes). Do not list details the request leaves open, and do not repeat checkpoints judged contradicted.

Output one json object with exactly the keys "checkpoints" and "contradictions": one checkpoints entry per checkpoint, its id copied exactly as written inside the square brackets, status one of stated | contradicted | absent, roles one of correct | wrong | not_applicable.

EXAMPLE JSON OUTPUT (for checkpoints [e01] and [e01r]):
{"checkpoints": [{"id": "e01", "status": "stated", "shot": 2, "roles": "correct"}, {"id": "e01r", "status": "absent", "shot": null, "roles": "not_applicable"}], "contradictions": [{"shot": 3, "plan": "Character 2 sits at the desk.", "request": "The woman stands by the window."}]}"""


def judge_prompt(request: str, plan_text: str, plan_form: str, requirements: Sequence[Mapping], tier: str) -> tuple[str, str]:
    """（系统消息, 用户消息）。系统消息每次相同、放在前面，便于 DeepSeek 按前缀命中上下文缓存；变化的内容都在用户消息。"""
    lines = []
    for row in requirements:
        # 基准逐镜档的题库检查点带镜号与时刻；写题调用只读请求写的检查点（任何档）不带，按整段请求判
        where = f" (request shot {row['shot']}, {'by the end of the shot' if row['anchor'] == 'end' else 'during the shot'})" \
            if "shot" in row else ""
        lines.append(f"[{row['requirement_id']}] {row['dimension']}{where}: {row['question']}")
    form = {"raw": "a JSON storyboard: registered characters, objects and places; per shot the place, who is present "
                   "(posture, state, holding), the action, and changes (what is different by the end of the shot)",
            "compiled": "the per-shot text a video renderer reads, compiled from the storyboard",
            "prose": "prompts for a video renderer written directly by the planner: a global caption, then one caption per shot"
            }[plan_form]
    return JUDGE_SYSTEM, f"""USER REQUEST
{request}

PLAN ({form})
{plan_text}

CHECKPOINTS
{chr(10).join(lines)}"""


def plan_forms(record: Mapping) -> dict[str, str]:
    """原答是规划器的 JSON；编译通过的计划另判编译后文字（渲染器读的就是它）。编译器丢字段、对齐或不写开镜、
    延续衣着都会让文字与原答不同，所以不按修补记录挑，编译通过的都判。

    直接写文字的规划器（``tools/run/planner_prose_baseline.py``，输出形态对照）没有 JSON，
    记录里是 ``global`` 与逐镜 ``shots``，只有 ``prose`` 一种形式。"""
    from mspc.data.shot_text import shot_text
    if "raw_storyboard" not in record:
        if not isinstance(record.get("global"), str) or not isinstance(record.get("shots"), list):
            raise ValueError("a record without raw_storyboard must be a direct-text plan with global and shots")
        return {"prose": "\n".join([f"Global: {record['global']}"] +
                                    [f"Shot {index + 1}: {caption}" for index, caption in enumerate(record["shots"])])}
    forms = {"raw": json.dumps(record["raw_storyboard"], ensure_ascii=False, separators=(",", ":"))}
    if record["status"] == "completed":
        board = record["storyboard"]
        forms["compiled"] = "\n".join(f"Shot {index + 1}: {shot_text(board, index)}" for index in range(len(board["shots"])))
    return forms


def conforms(value: Any, schema: Mapping) -> bool:
    """判官回答是否合本模块用到的那部分 JSON Schema（object／array／string 枚举／integer／null）。"""
    kinds = schema.get("type")
    kinds = kinds if isinstance(kinds, list) else [kinds]
    checks = {"object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list),
              "string": lambda v: isinstance(v, str), "integer": lambda v: type(v) is int, "null": lambda v: v is None}
    kind = next((k for k in kinds if checks[k](value)), None)
    if kind is None or ("enum" in schema and value not in schema["enum"]):
        return False
    if kind == "object":
        props = schema.get("properties", {})
        return (all(key in value for key in schema.get("required", []))
                and (schema.get("additionalProperties", True) or set(value) <= set(props))
                and all(conforms(value[key], sub) for key, sub in props.items() if key in value))
    if kind == "array":
        return all(conforms(item, schema["items"]) for item in value)
    return True


class CallLedger:
    """每次调用一个回执文件；发出前先落 request_started。已有回答（含不合结构的回答）的调用不重发；连接失败、限流、
    空回答或中途停掉的调用没有回答，下次运行时重发。

    按 DeepSeek 官方指南调用 chat/completions（11-12 读过 JSON Output、Thinking Mode、Context Caching、Rate Limit 四页）：
    系统消息放每次相同的说明与示例 JSON（提示里写到「json」并给示例，前缀相同便于命中上下文缓存），用户消息放变化的
    内容；``response_format`` 为 ``json_object``（接口不收 json_schema），回答再按结构定义核对，不合记 format_failed；
    思考模式默认开，``reasoning_effort`` 用 high，思考模式下不起作用的采样参数不发；``max_tokens`` 显式给足；
    指南说非 JSON 输出会被换成空格，空回答按未答处理、下次重发。走本机代理环境变量（11-12 实测直连 TLS 握手失败）。"""

    def __init__(self, root: Path, effort: str):
        """effort：low／high／max 走 ``reasoning_effort``；disabled 关掉思考（``thinking: {type: disabled}``）。"""
        if effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS}")
        self.effort = effort
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.opener = urllib.request.build_opener()

    def records(self) -> list[dict]:
        return [json.loads(path.read_text()) for path in sorted(self.root.rglob("*.json"))]

    def call(self, call_id: str, system: str, user: str, schema: Mapping) -> dict:
        path = self.root / f"{call_id}.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if (previous["system"], previous["user"], previous["effort"]) != (system, user, self.effort):
                # 11-13 实见：同一回执目录换了计划重判，按调用名命中旧回答，新判读悄悄沿用了旧计划的判词
                raise ValueError(f"{path} answers different messages; judge changed inputs into a new calls directory")
            if previous["status"] in ("answered", "format_failed"):
                return previous
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        body = {"model": MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_object"}, "max_tokens": MAX_TOKENS,
                **({"thinking": {"type": "disabled"}} if self.effort == "disabled" else {"reasoning_effort": self.effort})}
        record = {"call_id": call_id, "model": MODEL, "effort": self.effort, "endpoint": ENDPOINT, "status": "request_started",
                  "started_at": now(), "system": system, "user": user}
        write(path, record)
        request = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        try:
            with self.opener.open(request, timeout=900) as response:
                raw = json.loads(response.read())
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as error:
            record.update(finished_at=now(), status="request_failed", error=f"{type(error).__name__}: {error}"[:2000])
            write(path, record)
            return record
        message = raw["choices"][0]["message"]
        record.update(finished_at=now(), usage=raw.get("usage"), finish_reason=raw["choices"][0].get("finish_reason"),
                      text=message.get("content"), reasoning=message.get("reasoning_content"))
        if not (record["text"] or "").strip():
            record["status"] = "blank_answer"
        else:
            try:
                result = json.loads(record["text"])
            except json.JSONDecodeError as error:
                record.update(status="format_failed", error=str(error))
            else:
                if conforms(result, schema["json_schema"]["schema"]):
                    record.update(result=result, status="answered")
                else:
                    record.update(status="format_failed", error="answer does not match the schema")
        write(path, record)
        return record


# ---------------------------------------------------------------- 记录读取与汇总


def load_records(specs: Sequence[str], stories: Mapping[str, Mapping]) -> dict[tuple[str, str], dict]:
    """``TIER=DIR``，DIR 里每个故事一个计划记录（generation.plan 写的 <story_id>.json）；缺一个就报错。"""
    records = {}
    for spec in specs:
        tier, directory = spec.split("=", 1)
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}")
        for story_id in stories:
            path = Path(directory) / f"{story_id}.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing plan record {path}")
            records[(tier, story_id)] = {**json.loads(path.read_text()), "_path": str(path)}
    return records


def grade(tier: str, requirements: Sequence[Mapping], answer: Mapping | None) -> dict:
    """判官回答折成要求保留、在请求那一镜或按请求先后、主体与受体、末态。"""
    if answer is None:
        return {"measured": False}
    by_id = {row["id"]: row for row in answer["checkpoints"]}
    if set(by_id) != {row["requirement_id"] for row in requirements}:
        return {"measured": False, "error": "checkpoint ids differ from the frozen checkpoints"}
    events = [row for row in requirements if row["dimension"] == "event"]
    states = [row for row in requirements if row["dimension"] == "state"]
    stated = lambda row: by_id[row["requirement_id"]]["status"] == "stated"
    out = {"measured": True,
           "event_kept": _count(stated(row) for row in events),
           "state_kept": _count(stated(row) for row in states),
           "contradicted": sum(by_id[row["requirement_id"]]["status"] == "contradicted" for row in requirements),
           "roles_correct": _count(by_id[row["requirement_id"]]["roles"] == "correct" for row in events if stated(row)),
           "fabricated_conflicts": len(answer["contradictions"])}
    if tier == "shots":
        for name, rows in (("event_order", events), ("state_order", states)):
            anchored = [row for row in rows if "shot" in row]
            missing = [row["requirement_id"] for row in rows if "shot" not in row]
            if anchored:
                out[name] = _count(stated(row) and by_id[row["requirement_id"]]["shot"] == row["shot"] for row in anchored)
            elif not missing:
                out[name] = _count([])
            if missing:
                out.setdefault("unmeasured", {})[name] = {
                    "checkpoints": len(missing), "requirement_ids": missing,
                    "reason": "frozen_checkpoint_missing_request_shot"}
    else:
        shots = [by_id[row["requirement_id"]]["shot"] for row in sorted(events, key=lambda r: r["order"]) if stated(row)]
        out["event_order"] = _count(type(a) is int and type(b) is int and a <= b for a, b in zip(shots, shots[1:]))
    return out


def _story_mean(rows: Sequence[Mapping]) -> float | None:
    rates = [row["pass"] / row["applicable"] for row in rows if row["applicable"]]
    return sum(rates) / len(rates) if rates else None


def summarize(stories: Mapping[str, Mapping], records: Mapping, program: Mapping, judged: Mapping) -> dict:
    tiers = sorted({tier for tier, _ in records}, key=TIERS.index)
    summary = {}
    for tier in tiers:
        ids = sorted(story for t, story in records if t == tier)
        contract = {key: sum(program[(tier, story)]["contract"][key] for story in ids)
                    for key in ("json_parsed", "compiled_valid", "contract_accepted", "truncated")}
        block = {"expected": len(ids), **contract, "status_counts": dict(Counter(program[(tier, s)]["status"] for s in ids)),
                 "compile_repairs": {key: sum(program[(tier, s)]["compile_repairs"][key] for s in ids)
                                     for key in ("holding_presence_added", "opening_aligned_with_changes")},
                 "checks": {}}
        names = sorted({name for s in ids for name in program[(tier, s)].get("checks", {})})
        for name in names:
            rows = [program[(tier, s)]["checks"][name] for s in ids if name in program[(tier, s)].get("checks", {})]
            block["checks"][name] = {"pass": sum(r["pass"] for r in rows), "applicable": sum(r["applicable"] for r in rows),
                                     "plans": len(rows), "story_mean": _story_mean(rows), "checker": "program"}
        def judge_section(graded):
            measured = [row for row in graded if row["measured"]]
            section = {"plans_sent": len(graded), "plans_measured": len(measured), "checker": "judge"}
            for name in ("event_kept", "state_kept", "roles_correct", "event_order", "state_order"):
                rows = [row[name] for row in measured if name in row]
                if rows:
                    section[name] = {"pass": sum(r["pass"] for r in rows), "applicable": sum(r["applicable"] for r in rows),
                                     "story_mean": _story_mean(rows)}
            section["contradicted_checkpoints"] = sum(row["contradicted"] for row in measured)
            section["fabricated_conflicts"] = sum(row["fabricated_conflicts"] for row in measured)
            unmeasured_names = sorted({name for row in measured for name in row.get("unmeasured", {})})
            for name in unmeasured_names:
                missing = [row["unmeasured"][name] for row in measured if name in row.get("unmeasured", {})]
                reasons = Counter()
                for row in missing:
                    reasons[row["reason"]] += row["checkpoints"]
                section.setdefault("unmeasured", {})[name] = {
                    "checkpoints": sum(row["checkpoints"] for row in missing), "plans": len(missing),
                    "reason_counts": dict(reasons)}
            return section

        raw = [judged[(tier, s, "raw")] for s in ids if (tier, s, "raw") in judged]
        if raw:
            block["judge_raw"] = judge_section(raw)
        repaired = [s for s in ids if (tier, s, "compiled") in judged]
        if repaired:
            # 编译后文字只在被修补过的计划上判；同一批计划的原答另列，两列才可比。
            block["judge_compiled"] = judge_section([judged[(tier, s, "compiled")] for s in repaired])
            block["judge_compiled"]["raw_on_same_plans"] = judge_section([judged[(tier, s, "raw")] for s in repaired])
        summary[tier] = block
    return summary


DEFINITIONS = {
    "shot_count_kept": "shots tier: the answer has exactly the number of shots the request lists",
    "request_entities_registered": "shots tier: every Character/Object/Location id the request declares is registered in the answer",
    "request_presence_kept": "shots tier: each (shot, id) the question writer read from the request as in that shot is in the answer's present list or held by someone present",
    "identity_continuity": "shots tier: for each frozen identity question (same person/object/place across two shots) the answer keeps the same id present in both shots (same Location id for place)",
    "layout_legal": "per answer layout entry (shot, entity): box inside [0,1] with positive area, at most 2 strictly increasing anchor times, registered entity present in that shot, not_visible anchors carry box=null",
    "reference_legal": "per answer reference: points to an earlier shot where the entity is present, and the entity is present in the current shot",
    "event_kept": "frozen event checkpoints the judge reads as stated in the plan",
    "state_kept": "frozen end-state (result) checkpoints the judge reads as stated in the plan",
    "roles_correct": "among stated events, doer and receiver match the request",
    "event_order": "shots tier: stated in the requested shot, only where the frozen checkpoint supplies a shot number; missing anchors are unmeasured; synopsis/paragraph: consecutive stated events appear in non-decreasing plan shots",
    "state_order": "shots tier: end-state stated in the requested shot, only where the frozen checkpoint supplies a shot number; missing anchors are unmeasured",
}


def export(spec: Mapping, *, stories: Mapping[str, Mapping], checkpoints: Mapping) -> dict:
    """论文 evidence：每个规划器每档的合同、程序检查与判官检查，故事等权另给计数。"""
    planners, calls = [], Counter()
    for planner in spec["planners"]:
        program = json.loads(Path(planner["program"]).read_text())
        records = {tuple(key.split("/")): path for key, path in program["records"].items()}
        plans = {tuple(key.split("/")): value for key, value in program["plans"].items()}
        judged = {}
        for path in planner["judge"]:
            content = json.loads(Path(path).read_text())
            for key, value in content["plans"].items():
                tier, story, form = key.split("/")
                judged[(tier, story, form)] = value
        summary = summarize(stories, records, plans, judged)
        for block in summary.values():
            for name, check in block["checks"].items():
                check["definition"] = DEFINITIONS[name]
            for form in ("raw", "compiled"):
                for name, check in block.get(f"judge_{form}", {}).items():
                    if isinstance(check, dict) and name in DEFINITIONS:
                        check["definition"] = DEFINITIONS[name]
        planners.append({key: planner[key] for key in ("name", "label", "adapter", "base_model", "contract_version", "records_root")}
                        | {"tiers": summary})
    for root in spec["calls"]:
        for path in Path(root).rglob("*.json"):
            row = json.loads(path.read_text())
            if row.get("model") != MODEL:
                raise ValueError(f"{path}: judged by {row.get('model')}, not {MODEL}; export one judge at a time")
            calls[row["status"]] += 1
    return {"observed_at": now(), "exported_by": spec["exported_by"],
            "export_tool": "src/mspc/evaluation/planner_text_checks.py export",
            "export_code_revision": spec["export_code_revision"],
            "generation_code_revision": spec["generation_code_revision"],
            "scoring_code_revision": spec["scoring_code_revision"],
            "benchmark": {"version": "v4", "split": spec["split"], "stories_expected": len(stories),
                          "story_ids": sorted(stories), "requests": "configs/benchmark/story_requests_v4.json",
                          "storyboards": "configs/benchmark/benchmark_storyboards_v4.json",
                          "question_bank": spec["question_bank"], "checkpoints": spec["checkpoints"],
                          "checkpoint_counts": {tier: sum(len(semantic_requirements(tier, checkpoints[tier][story]))
                                                          for story in stories) for tier in TIERS if tier in checkpoints}},
            "evaluation_scope": "planner_text", "input_tiers": list(TIERS), "seed": 42, "candidates_per_request": 1,
            "judge": {"model": MODEL, "effort": EFFORT, "interface": "DeepSeek official chat/completions, JSON output mode",
                      "checkpoint_writer": "frozen checkpoints; synopsis and paragraph written once from the tier request only",
                      "shots_checkpoints": "frozen v4 question bank written by gpt-6.1-sol from the shots request"},
            "run_roots": spec["run_roots"], "planners": planners,
            "fixed_renderer_videos": spec["fixed_renderer_videos"],
            "judge_calls": dict(calls),
            "boundary": spec["boundary"]}


# ---------------------------------------------------------------- 命令


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("checkpoints", help="冻结检查点：基准逐镜档来自题库，其余档由写题调用只读请求写出")
    freeze.add_argument("--requests", type=Path, required=True)
    freeze.add_argument("--split", required=True, help="请求表里的划分名（基准 dev／test，或规划器样本池扩样的划分）")
    freeze.add_argument("--writer-tiers", nargs="+", choices=TIERS, required=True,
                        help="由写题调用写检查点的档；基准逐镜档用题库（--questions），没看过的样本池三档都用写题调用")
    freeze.add_argument("--questions", type=Path, help="基准题库；只在逐镜档不由写题调用写时给")
    freeze.add_argument("--calls", type=Path, required=True)
    freeze.add_argument("--workers", type=int, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    for name in ("program", "judge"):
        command = commands.add_parser(name)
        command.add_argument("--requests", type=Path, required=True)
        command.add_argument("--split", required=True)
        command.add_argument("--checkpoints", type=Path, required=True)
        command.add_argument("--records", nargs="+", required=True, help="TIER=DIR")
        command.add_argument("--planner", required=True)
        command.add_argument("--output", type=Path, required=True)
    exporting = commands.add_parser("export")
    exporting.add_argument("--requests", type=Path, required=True)
    exporting.add_argument("--split", required=True)
    exporting.add_argument("--checkpoints", type=Path, required=True)
    exporting.add_argument("--spec", type=Path, required=True)
    exporting.add_argument("--output", type=Path, required=True)
    judge = commands.choices["judge"]
    judge.add_argument("--calls", type=Path, required=True)
    judge.add_argument("--workers", type=int, required=True)
    judge.add_argument("--forms", nargs="+", choices=("raw", "compiled", "prose"), required=True,
                       help="原答与编译后文字分开判：同一份原答换编译器重编时只判编译后文字")
    args = parser.parse_args(argv)
    if getattr(args, "workers", 1) is not None and not 1 <= getattr(args, "workers", 1) <= MAX_WORKERS:
        parser.error(f"workers must be within 1..{MAX_WORKERS}")
    stories = dev_requests(args.requests, args.split)

    if args.command == "checkpoints":
        if "shots" in args.writer_tiers and args.questions is not None:
            parser.error("shot-list checkpoints come from either --questions or the writer, not both")
        frozen = {"requests": str(args.requests), "split": args.split, "written_at": now(),
                  "writer": {"model": MODEL, "effort": EFFORT, "prompt": list(writer_prompt("{request}"))}}
        if args.questions is not None:
            frozen["shots"] = shots_checkpoints(json.loads(args.questions.read_text()), stories)
        jobs = [(tier, story) for tier in args.writer_tiers for story in stories]
        ledger = CallLedger(args.calls, EFFORT)
        with ThreadPoolExecutor(args.workers) as pool:
            answers = list(pool.map(lambda job: ledger.call(f"checkpoints/{job[0]}/{job[1]}",
                                                            *writer_prompt(stories[job[1]]["inputs"][job[0]]["text"]),
                                                            WRITER_SCHEMA), jobs))
        for (tier, story_id), record in zip(jobs, answers):
            if record["status"] != "answered":
                raise ValueError(f"checkpoint writer failed for {tier}/{story_id}: {record['status']}")
            frozen.setdefault(tier, {})[story_id] = {"source": "writer", "call": record["call_id"],
                                                     "requirements": semantic_checkpoints(record["result"]["events"])}
        write(args.output, frozen)
        return 0

    checkpoints = json.loads(args.checkpoints.read_text())
    if args.command == "export":
        write(args.output, export(json.loads(args.spec.read_text()), stories=stories, checkpoints=checkpoints))
        return 0
    records = load_records(args.records, stories)
    if args.command == "program":
        program = {f"{tier}/{story}": program_checks(record, tier, stories[story]["inputs"][tier]["text"],
                                                     checkpoints["shots"][story] if tier == "shots" else None)
                   for (tier, story), record in records.items()}
        write(args.output, {"planner": args.planner, "observed_at": now(), "records": {
            f"{tier}/{story}": record["_path"] for (tier, story), record in records.items()}, "plans": program})
        return 0

    jobs = []
    for (tier, story), record in records.items():
        prose = "raw_storyboard" not in record
        if (prose and record["status"] != "completed") or (not prose and not isinstance(record.get("raw_storyboard"), Mapping)):
            continue
        if tier not in checkpoints:
            raise ValueError(f"checkpoints for {tier} have not been written")
        requirements = semantic_requirements(tier, checkpoints[tier][story])
        if not requirements:
            continue
        for form, text in plan_forms(record).items():
            if form not in args.forms:
                continue
            prompt = judge_prompt(stories[story]["inputs"][tier]["text"], text, form, requirements, tier)
            jobs.append((f"{args.planner}/{tier}/{form}/{story}", tier, story, form, prompt, requirements))
    print(json.dumps({"judge_calls": len(jobs), "by_form": dict(Counter(job[3] for job in jobs))}), flush=True)
    ledger = CallLedger(args.calls, EFFORT)

    def run(job):
        call_id, tier, story, form, prompt, requirements = job
        record = ledger.call(call_id, *prompt, JUDGE_SCHEMA)
        print(json.dumps({"call": call_id, "status": record["status"]}), flush=True)
        return f"{tier}/{story}/{form}", {"call": call_id, **grade(
            tier, requirements, record.get("result") if record["status"] == "answered" else None)}

    with ThreadPoolExecutor(args.workers) as pool:
        graded = dict(pool.map(run, jobs))
    write(args.output, {"planner": args.planner, "observed_at": now(), "judge": {"model": MODEL, "effort": EFFORT},
                        "plans": graded})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
