"""旧 event_log_v3 冻结基线与评测逐镜档写手；当前分镜表规划器入口是 planner_dataset。"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Mapping, Sequence

from mspc.core.draws import unit_draw
from mspc.data.event_log import (
    EVENT_LOG_SCHEMA, EventLog, FactEntry, entry_labels, fact_roles, parse_event_log, script_name, table_entries, validate_event_log,
)

MODES = ("script_only", "script_true_history", "script_generated_history")
def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected nonempty JSON object rows")
    return rows


BASELINE_CASE_KEYS = {"case_id", "script", "target_start_shot_id", "target_start_seconds",
                      "history", "gold_event_log"}


def validate_baseline_case(case: Mapping) -> None:
    """正式入口只消费 v3 金标；请求由 planner_request 唯一构造，付费前先构造一遍。"""
    if set(case) != BASELINE_CASE_KEYS:
        raise ValueError(f"baseline case keys differ: {sorted(set(case) ^ BASELINE_CASE_KEYS)}")
    validate_event_log(parse_event_log(case["gold_event_log"]))
    baseline_request(case)


def baseline_request(case: Mapping) -> dict:
    """无图臂与有图臂同一请求文字；历史目录取真历史的镜号。"""
    from mspc.data.event_planner_contract import planner_request
    return planner_request(script=case["script"], event_log=case["gold_event_log"],
                           target_start_shot_id=case["target_start_shot_id"],
                           history_shot_ids=[frame["shot_id"] for frame in case["history"]["true"]])


def build_baseline_case(example: Mapping, *, history: Mapping, target_start_seconds: float) -> dict:
    """从规划器样本 producer 接入；历史时间来自外部实测ledger，不猜测源PTS。"""
    from mspc.data.event_planner_contract import validate_planner_example
    validate_planner_example(example, require_files=False)
    case = {"case_id": example["sample_id"], "script": example["script"],
            "target_start_shot_id": example["target_start_shot_id"], "target_start_seconds": target_start_seconds,
            "history": dict(history), "gold_event_log": example["event_log"]}
    expected = [(f["shot_id"], f["path"]) for f in example["history_frames"]]
    if [(f["shot_id"], f["path"]) for f in history["true"]] != expected:
        raise ValueError("history ledger differs from the planner sample producer")
    validate_baseline_case(case)
    return case


def evaluate_baseline_response(raw: str, case: Mapping, *, token_counter: Callable[[str], int]) -> dict:
    """逐层可编译性；事实 F1 与保住率要标签和裁决表，离线由 event_planner_metrics 另算。"""
    from mspc.evaluation.event_planner_metrics import compile_layers
    result = compile_layers(raw, baseline_request(case), token_counter=token_counter)
    result["unknown_counts"] = dict(scalar_counts(result["event_log"])) if result["event_log"] is not None else {}
    return result


def history_frames(case: Mapping, mode: str, config: Mapping) -> list[dict]:
    indices = {shot["shot_id"]: i for i, shot in enumerate(case["gold_event_log"]["shots"])}
    target = indices[case["target_start_shot_id"]]
    selected = {"script_only": "true", "script_true_history": "true",
                "script_generated_history": "generated"}[mode]
    if type(config["image_quota"]) is not int or config["image_quota"] <= 0:
        raise ValueError("image_quota must be a positive int")
    if type(case["target_start_seconds"]) not in (int, float) or not math.isfinite(case["target_start_seconds"]):
        raise ValueError("target_start_seconds must be finite")
    frames = case["history"][selected]
    # 从 s01 起的样本没有更早的镜，历史为空；空历史时各臂都是同一段无图请求。
    if not isinstance(frames, list) or len(frames) > config["image_quota"]:
        raise ValueError("history count must be within explicit image_quota")
    if len({f["shot_id"] for f in frames}) != len(frames):
        raise ValueError("duplicate history shot")
    directory = [(f["shot_id"], f["timestamp_seconds"]) for f in case["history"]["true"]]
    if [(f["shot_id"], f["timestamp_seconds"]) for f in frames] != directory:
        raise ValueError("history arms must use the same frozen frame selection")
    for frame in frames:
        if set(frame) != {"shot_id", "timestamp_seconds", "path", "media_kind"}:
            raise ValueError("history requires shot_id/timestamp_seconds/path/media_kind only")
        if type(frame["timestamp_seconds"]) not in (int, float) or not math.isfinite(frame["timestamp_seconds"]):
            raise ValueError("history timestamp must be finite")
        if indices[frame["shot_id"]] >= target or not 0 <= frame["timestamp_seconds"] < case["target_start_seconds"]:
            raise ValueError("history pixels must precede target shot and target timestamp")
        if frame["media_kind"] != selected:
            raise ValueError("history source differs from arm")
    return [] if mode == "script_only" else frames


def history_image_data(frame: Mapping, *, image_max_pixels: int) -> tuple[str, dict]:
    from PIL import Image
    if type(image_max_pixels) is not int or image_max_pixels <= 0:
        raise ValueError("image_max_pixels must be a positive int")
    with Image.open(frame["path"]) as source:
        image = source.convert("RGB")
    original_size = list(image.size)
    scale = min(1.0, (image_max_pixels / (image.width * image.height)) ** 0.5)
    if scale < 1:
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(), {
        **frame, "original_size": original_size, "sent_size": list(image.size)}


def baseline_messages(case: Mapping, mode: str, config: Mapping, output_contract: Mapping) -> list[dict]:
    schema = output_contract["schema"]
    if schema != EventLog.model_json_schema() or config["schema_version"] != EVENT_LOG_SCHEMA:
        raise ValueError("frozen schema version differs from config")
    from mspc.data.event_planner_contract import planner_contract_messages, planner_user_content
    messages = planner_contract_messages(output_contract)
    images = [{"type": "image_url", "image_url": {"url": history_image_data(frame, image_max_pixels=config["image_max_pixels"])[0]}}
              for frame in history_frames(case, mode, config)]
    messages.append({"role": "user", "content": planner_user_content(baseline_request(case), images)})
    return messages


def scalar_counts(value: Any, prefix: str = "") -> Counter:
    counts = Counter()
    if isinstance(value, dict):
        for key, child in value.items():
            counts.update(scalar_counts(child, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for child in value:
            counts.update(scalar_counts(child, prefix + "[]"))
    else:
        counts["total"] += 1
        counts[f"total:{prefix}"] += 1
        if value == "unknown":
            counts["unknown"] += 1
            counts[f"unknown:{prefix}"] += 1
    return counts


def request_payload(model: str, messages: list[dict], config: Mapping, *, schema: Mapping | None) -> dict:
    """请求协议 owner；所有后端必填值来自显式配置，没有旧配方兼容分支。"""
    if model != config["model"]:
        raise ValueError("requested model differs from the explicit recipe")
    if type(config["seed"]) is not int:
        raise TypeError("seed must be int")
    payload = {"model": model, "messages": messages}
    if type(config["max_output_tokens"]) is not int or config["max_output_tokens"] <= 0:
        raise ValueError("max_output_tokens must be a positive int")
    backend = config["backend"]
    if backend == "vllm":
        if type(config["enable_thinking"]) is not bool:
            raise TypeError("enable_thinking must be bool")
        sampling = config["sampling"]
        required = {"temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"}
        if set(sampling) != required or any(type(v) not in (int, float) or not math.isfinite(v) for v in sampling.values()):
            raise ValueError("sampling must contain all six numeric parameters")
        if type(sampling["top_k"]) is not int or sampling["top_k"] < 1:
            raise ValueError("top_k must be a positive int")
        if not 0 < sampling["top_p"] <= 1 or not 0 <= sampling["min_p"] <= 1 or sampling["temperature"] < 0 or sampling["repetition_penalty"] <= 0:
            raise ValueError("invalid sampling range")
        if {"enable_thinking", "reasoning_effort"} & config["chat_template_kwargs"].keys():
            raise ValueError("thinking controls have exactly one config owner")
        payload["seed"] = config["seed"]
        payload.update(sampling)
        payload["max_tokens"] = config["max_output_tokens"]
        payload["chat_template_kwargs"] = {"enable_thinking": config["enable_thinking"],
                                           **config["chat_template_kwargs"]}
        if config["reasoning_effort"] is not None:
            payload["chat_template_kwargs"]["reasoning_effort"] = config["reasoning_effort"]
    elif backend == "openai":
        # 官方 Chat Completions：无 tools；不向 reasoning 模型透传 Qwen 采样参数。
        payload["max_completion_tokens"] = config["max_output_tokens"]
        payload["reasoning_effort"] = config["reasoning_effort"]
    else:
        raise ValueError("backend must be vllm or openai")
    fmt = config["response_format"]
    if fmt == "json_schema":
        if schema is None:
            raise ValueError("json_schema requires the frozen output schema")
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": config["schema_version"], "strict": True, "schema": dict(schema)}}
    elif fmt is not None:
        raise ValueError("response_format must be json_schema or explicit null")
    return payload


def request_once(base_url: str, model: str, messages: list[dict], config: Mapping, *, schema: Mapping | None = None, cost_counter=None) -> dict:
    payload = request_payload(model, messages, config, schema=schema)
    headers = {"Content-Type": "application/json"}
    secret = None
    if config["backend"] == "openai":
        if cost_counter is None:
            raise ValueError("closed planner requires a run cost counter")
        cost_counter.reserve_precall(model)
        secret = os.environ[config["api_key_env"]]
        if not secret.strip():
            raise ValueError("empty API key environment variable")
        headers["Authorization"] = "Bearer " + secret
    started = time.monotonic()
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(), headers=headers)
    result = {"request": payload, "raw_response": None, "raw_content": "", "raw_reasoning": None,
              "reasoning_details": None, "usage": None, "served_model": None,
              "status": "transport_failed", "finish_reason": None, "http_status": None,
              "requested_model": model, "returned_model": None, "provider": None,
              "model_version": None, "system_fingerprint": None, "request_id": None,
              "endpoint": base_url.rstrip("/") + "/chat/completions"}
    try:
        with urllib.request.urlopen(request, timeout=config["timeout_seconds"]) as response:
            body = response.read().decode()
            result["raw_response"] = body
            if cost_counter is not None:
                result["status"] = "billing_unknown"
            result["http_status"] = response.status
        parsed = json.loads(body)
        result.update(usage=parsed.get("usage"), returned_model=parsed.get("model"),
                      served_model=parsed.get("model"), provider=parsed.get("provider"),
                      model_version=parsed.get("model_version"),
                      system_fingerprint=parsed.get("system_fingerprint"), request_id=parsed.get("id"))
        if cost_counter is not None:
            actual = cost_counter.record_usage(model, parsed["usage"])
            result["estimated_cost_usd"] = str(actual)
        choice = parsed["choices"][0]
        message = choice["message"]
        content = message["content"]
        if content is not None and not isinstance(content, str):
            raise TypeError("final content must be string or null")
        finish = choice["finish_reason"]
        status = "incomplete" if finish != "stop" else ("completed" if content and content.strip() else "empty_final")
        result.update(raw_content=content or "", raw_reasoning=message.get("reasoning", message.get("reasoning_content")),
                      reasoning_details=message.get("reasoning_details"), usage=parsed.get("usage"),
                      served_model=parsed.get("model"), request_id=parsed.get("id"),
                      finish_reason=finish, status=status)
    except urllib.error.HTTPError as exc:
        result.update(http_status=exc.code, raw_response=exc.read().decode(), error=str(exc))
    except (urllib.error.URLError, TimeoutError, ValueError, TypeError, KeyError, IndexError, RuntimeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    if secret:
        for key in ("raw_response", "error"):
            if isinstance(result.get(key), str):
                result[key] = result[key].replace(secret, "<redacted>")
    result["latency_seconds"] = time.monotonic() - started
    return result


def wilson(success: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = success / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [center-radius, center+radius]


def rate(success: int, total: int) -> dict:
    return {"numerator": success, "denominator": total, "rate": success/total if total else None,
            "wilson_95": wilson(success, total)}


def baseline_summary(rows: Sequence[Mapping]) -> dict:
    from mspc.evaluation.event_planner_metrics import compilability
    scalar = Counter()
    for row in rows:
        scalar.update(row["evaluation"]["unknown_counts"])
    return {"cases": len(rows), "transport_status": dict(Counter(row["status"] for row in rows)),
        "compilability": compilability([row["evaluation"] for row in rows]),
        "failed_layer_counts": dict(Counter(str(row["evaluation"]["failed_layer"]) for row in rows)),
        "unknown_scalar": rate(scalar["unknown"], scalar["total"]), "unknown_counts_by_field": dict(scalar),
        "unknown_scope": "仅schema可解析输出；所有JSON标量值计分母，包括标识符、布尔和整数，不计键名",
        "manual_review": "pending"}


def run_baseline(*, benchmark: Path, output: Path, base_url: str, model: str, config: Mapping,
                 token_counter: Callable[[str], int]) -> None:
    cost_counter = None
    if config["backend"] == "openai":
        from mspc.data.openrouter_client import OpenRouterCostCounter
        if config["concurrency"] != 1:
            raise ValueError("closed planner uses sequential requests for the shared budget")
        cost_counter = OpenRouterCostCounter(hard_cap_usd=config["budget_hard_cap_usd"],
            prices={model: config["pricing"]}, precall_prompt_tokens=config["precall_prompt_tokens"],
            precall_completion_tokens=config["max_output_tokens"])
    document = json.loads(benchmark.read_text())
    cases = document["cases"]
    contract = document["output_contract"]
    if not config["modes"] or len(set(config["modes"])) != len(config["modes"]):
        raise ValueError("modes must be nonempty and unique")
    for case in cases:
        validate_baseline_case(case)
    if len(cases) != config["expected_benchmark_cases"] or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("benchmark count/unique case IDs differ from explicit config")
    messages_by_mode = {mode: [baseline_messages(case, mode, config, contract) for case in cases]
                        for mode in config["modes"]}
    for messages in messages_by_mode.values():
        for message in messages:
            request_payload(model, message, config, schema=contract["schema"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "run_config.json").write_text(json.dumps({"benchmark": str(benchmark), "model": model, "base_url": base_url, **config}, indent=2))
    for mode in config["modes"]:
        # 先构造/校验整臂，防止缺字段/未来帧在已发请求后才暴露；不选择性重试。
        messages = messages_by_mode[mode]
        response_lock = Lock()
        def execute(item):
            case, message = item
            from PIL import Image
            images = [part["image_url"]["url"] for part in message[-1]["content"] if part["type"] == "image_url"]
            ledger = []
            for frame, encoded in zip(history_frames(case, mode, config), images, strict=True):
                with Image.open(io.BytesIO(base64.b64decode(encoded.split(",", 1)[1]))) as image:
                    ledger.append({**frame, "sent_size": list(image.size)})
            row = request_once(base_url, model, message, config, schema=contract["schema"], cost_counter=cost_counter)
            row["history_ledger"] = ledger
            row.update(case_id=case["case_id"], model=model, mode=mode)
            # 独立于离线评估保存每个已返回请求；并发中其它回执也不会因单例校验崩溃丢失。
            with response_lock:
                responses.write(json.dumps(row, ensure_ascii=False) + "\n")
                responses.flush()
            return row
        rows = []
        cases_by_id = {case["case_id"]: case for case in cases}
        with (output / f"{mode}.responses.jsonl").open("w") as responses, (output / f"{mode}.jsonl").open("w") as file, ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
            for row in (map(execute, zip(cases, messages, strict=True)) if cost_counter is not None
                        else pool.map(execute, zip(cases, messages, strict=True))):
                if cost_counter is not None:
                    (output / "cost.json").write_text(json.dumps(cost_counter.snapshot(), indent=2))
                row["evaluation"] = evaluate_baseline_response(row["raw_content"], cases_by_id[row["case_id"]],
                                                               token_counter=token_counter)
                rows.append(row)
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()
                if cost_counter is not None and row["status"] in {"billing_unknown", "transport_failed"}:
                    raise RuntimeError("closed response saved; reconcile unknown billing before further requests")
        summary = baseline_summary(rows)
        summary["semantic_metrics"] = "fact F1 / retention need writer labels and free-text verdicts; scored offline"
        (output / f"{mode}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text.casefold())


def native_leakage(script: str, captions: Sequence[str]) -> dict:
    # Full sentences use exact characters. Eight-word spans use normalized word tokens.
    sentences = [s.strip() for cap in captions for s in re.split(r"(?<=[.!?])\s+", cap.strip()) if s.strip()]
    copied = [sentence for sentence in sentences if sentence in script]
    script_words = words(script)
    positions = set()
    matches = []
    for caption in captions:
        native = words(caption)
        grams = {tuple(native[i:i+8]) for i in range(max(0, len(native)-7))}
        for i in range(max(0, len(script_words)-7)):
            if tuple(script_words[i:i+8]) in grams:
                positions.update(range(i, i+8))
                matches.append({"start_word": i, "text": " ".join(script_words[i:i+8])})
    return {"exact_copied_sentences": copied, "native_sentence_count": len(sentences),
            "exact_sentence_ratio": len(copied)/len(sentences) if sentences else None,
            "script_word_count": len(script_words), "copied_8word_positions": len(positions),
            "copied_8word_ratio": len(positions)/len(script_words) if script_words else None,
            "eight_word_matches": matches, "tokenization": "English word-token normalized; exact full-sentence case-sensitive"}


def quoted_annotation_leakage(script: str, captions: Sequence[str]) -> dict:
    """额外检出短字幕/对白引文；输入是caption标注，不冒充像素OCR字幕金标。"""
    quotes = []
    for index, caption in enumerate(captions):
        for match in re.finditer(r"(?<![A-Za-z])(['\"])(.+?)\1(?![A-Za-z])", caption):
            text = match.group(2)
            context = caption[max(0, match.start()-120):match.end()+120].casefold()
            kind = "subtitle_or_dialogue_annotation" if re.search(r"subtitles?|dialogue", context) else "other_quoted_annotation"
            quotes.append({"caption_index": index, "text": text, "context_class": kind,
                           "exact_copied": text in script})
    candidates = [q for q in quotes if q["context_class"] == "subtitle_or_dialogue_annotation"]
    return {"scope": "case-sensitive literal quoted spans in native captions; annotation context, not pixel/OCR truth; translation not covered",
            "subtitle_dialogue_candidate_count": len(candidates),
            "exact_copied_candidate_count": sum(q["exact_copied"] for q in candidates),
            "exact_candidate_ratio": sum(q["exact_copied"] for q in candidates)/len(candidates) if candidates else None,
            "quotes": quotes}


def format_shot_tier(script: Mapping) -> str:
    """逐镜档交给反写的正文：主体、地点、各镜 prompt。不含 story_prompt 与 observed_states。"""
    lines = ["SUBJECTS"]
    for subject in script["subjects"]:
        lines.append(f"- {subject['subject_id']} ({subject.get('kind') or ''}): {subject.get('attributes') or ''}")
    lines.append("LOCATIONS")
    for location in script["locations"]:
        lines.append(f"- {location['location_id']}: {location.get('attributes') or ''}")
    lines.append("SHOTS")
    for shot in script["shots"]:
        lines.append(f"{shot['shot_id']}: {shot['prompt']}")
    return "\n".join(lines)


SHOT_TIER_REWRITE_CONFIG_KEYS = (
    "model", "base_url", "temperature", "max_tokens", "timeout_seconds", "request_options",
    "input_usd_per_mtok", "output_usd_per_mtok", "budget_hard_cap_usd",
    "precall_prompt_tokens", "precall_completion_tokens",
)


def load_shot_tier_rewrite_config(path: Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or set(payload) != set(SHOT_TIER_REWRITE_CONFIG_KEYS):
        raise ValueError(f"shot-tier config must contain exactly {SHOT_TIER_REWRITE_CONFIG_KEYS}")
    # 温度由配置显式给：写手按官方推荐 1.0，冻结的评测反写配方与纯删除任务仍是 0；这里只验类型与范围。
    if (type(payload["temperature"]) not in (int, float) or not 0 <= payload["temperature"] <= 2
            or type(payload["max_tokens"]) is not int or payload["max_tokens"] <= 0):
        raise ValueError("shot-tier config requires a numeric temperature in [0, 2] and a positive int max_tokens")
    if type(payload["timeout_seconds"]) is not int or payload["timeout_seconds"] <= 0:
        raise ValueError("timeout_seconds must be a positive int")
    if type(payload["precall_prompt_tokens"]) is not int or type(payload["precall_completion_tokens"]) is not int:
        raise ValueError("precall token estimates must be ints")
    return payload


def build_shot_tier_client(config: Mapping, *, api_key: str, proxy: str | None, recorder=None):
    """规划器训练与评测反写共用的 OpenRouter 客户端。proxy 必须显式传入，None 表示直连。
    recorder 见 :class:`OpenRouterClient`；要把每次请求与原始答复落盘的调用方（写手入口）传 :class:`WriterCallRecorder`。"""
    from mspc.data.openrouter_client import build_openrouter_client

    return build_openrouter_client(
        api_key=api_key,
        pricing={
            "base_url": config["base_url"],
            "model": config["model"],
            "hard_cap_usd": config["budget_hard_cap_usd"],
            "input_usd_per_mtok": config["input_usd_per_mtok"],
            "output_usd_per_mtok": config["output_usd_per_mtok"],
        },
        precall_prompt_tokens=config["precall_prompt_tokens"],
        precall_completion_tokens=config["precall_completion_tokens"],
        timeout_seconds=config["timeout_seconds"],
        proxy=proxy,
        recorder=recorder,
    )


def complete_shot_tier_call(client, messages, *, model: str, temperature: float, max_tokens: int, request_options: Mapping) -> str:
    """把反写消息交给共用客户端。返回模型原文，不在这里改写。"""
    response = client.complete_messages(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        request_options=request_options,
        response_format={"type": "json_object"},
    )
    if not isinstance(response.text, str) or not response.text.strip():
        raise ValueError("shot-tier call returned empty text")
    return response.text




# ---------------------------------------------------------------------------
# 逐镜档写手：从事件日志的编号事实表写逐镜剧本，只写起因；另起一次调用核对写到了哪些条目
# ---------------------------------------------------------------------------

STORYBOARD_WRITER_INSTRUCTIONS = """You write a shot-by-shot storyboard script from a numbered fact table of one video story.

Rules:
1. Write every entry marked WRITE and nothing else. Do not write an entry marked DO-NOT-WRITE, and do not add any person, animal, object, action, place, state, position in the frame, or camera detail that is not a WRITE entry.
2. Write an event only in the shot where it happens. Write a state only in the shot where it first appears or changes; the table already marks restated states DO-NOT-WRITE.
3. Match the level of detail of the two examples below. They are invented and unrelated to the input story.
4. Use your own words. Do not copy the wording of an action entry when a shorter paraphrase states the same fact.
5. The table marks an eye-level angle, a static camera and a place unchanged from the previous shot DO-NOT-WRITE; leave them out. Mention shot size, camera angle, camera movement, the place and positions in the frame only when they are WRITE entries.
6. Write shot size, camera angle and camera movement in exactly the words the table gives (for example "Medium shot", "High angle shot", "camera pans left"), at the start of the shot's text. Write a position entry (left, center, right, foreground, background) in front of the person: "On the left, Subject 1, ...".

Refer to each person and animal by its ID as the table writes it (Character N) and to each object by its ID (Subject N). Name a place by its description, never by its ID; a fixed place feature may also be named by its description. Put appearance entries in SUBJECTS/LOCATIONS attributes, not in the shot prompts.

Return only a JSON object {"subjects":[{"subject_id":"Character N or Subject N, as in the table","kind":"...","attributes":"..."}],"locations":[{"location_id":"Location N","attributes":"..."}],"shots":[{"shot_id":"s01","prompt":"..."}]}. List every subject, location and shot of the table in the table's order; attributes is "" when the table has no appearance entry for it.
"""
# 规则 1 出处：WanPE 改写的硬约束（不增删用户事实）；规则 2 与评测主报写法一致；规则 3 的两条示例为本处手写虚构故事，
# 不取任何评测窗；规则 4 另由 native_leakage 对原生字幕做事后检查。
# 规则 5 与地点用描述指代：首跑逐镜档每镜写运镜 97%、Location 编号 100%，
# 冻结评测逐镜剧本只有 1–13%、0–2%，训练输入要贴近评测时规划器拿到的输入。

STORYBOARD_LABEL_INSTRUCTIONS = """You check a script against a numbered fact table of the same story. Entries marked DO-NOT-WRITE were not meant to be written.

written: every entry, whatever its mark, that the script states or directly implies. A paraphrase or a shorter or more general wording counts. The script may name a subject or place by its description instead of its ID; match it to that ID. An entry of a shot is stated only by that shot's text, and an appearance entry by the SUBJECTS/LOCATIONS attributes.

restated: the DO-NOT-WRITE entries in written that the script states beyond what the WRITE entries of the same shot already state or imply. For example, a WRITE action "he takes off his hat" already implies "he is not wearing a hat", so that entry is written but not restated.

added_facts: each person, animal, object, action, place, state, or camera detail the script states that no table entry states or implies. A summary of several entries is not an added fact.

Return only a JSON object {"written":[entry numbers],"restated":[entry numbers],"added_facts":["short English clause", ...]}. Use [] when nothing applies.
"""
# 10-10 首跑亲看 20 窗的误报：核对器把粗档的外观指代对不回编号、要求粗档“同一镜”，
# 段落／梗概被成批判新增；把后续镜未写的服装记为写到；把 WRITE 动作句已蕴含的状态记为复述。三处各对应上面一句。


def _writer_example_logs() -> list[dict]:
    """两条手写虚构故事（码头钓者、红伞女孩），只用于示范写法的具体度；不读取语料或评测金标。"""
    def fact(subject, family, axis, value, shot):
        return {"subject": subject, "family": family, "predicate_or_axis": axis, "object_or_value": value,
                "scope": f"screen@{shot}" if family == "screen" else "scene", "shot_span": [shot, shot], "evidence": "observed"}
    pier = {"schema": EVENT_LOG_SCHEMA,
        "subjects": [{"subject_id": "Subject 1", "kind": "person", "appearance": "old man with a grey beard"},
                     {"subject_id": "Subject 2", "kind": "object", "appearance": "wooden fishing rod"}],
        "locations": [{"location_id": "Location 1", "appearance": "weathered wooden pier", "anchors": []}],
        "initial_state": [fact("Subject 1", "state", "costume", "a yellow raincoat", 0),
                          fact("Subject 1", "relation", "holds", "Subject 2", 0),
                          fact("Location 1", "state", "environment", "foggy", 0),
                          fact("Subject 1", "state", "costume", "a yellow raincoat", 1)],
        "shots": [{"shot_id": "s01", "location_id": "Location 1", "present_subjects": ["Subject 1", "Subject 2"],
                   "visible_entities": ["Subject 1", "Subject 2", "Location 1"],
                   "screen_facts": [fact("Subject 1", "screen", "horizontal", "left", 0),
                                    fact("Subject 1", "screen", "orientation", "away", 0)],
                   "frame_count": 49, "shot_size": "wide", "camera_angle": "eye level", "movement": "static",
                   "action": "The man casts the fishing line out over the water."},
                  {"shot_id": "s02", "location_id": "Location 1", "present_subjects": ["Subject 1", "Subject 2"],
                   "visible_entities": ["Subject 1", "Subject 2", "Location 1"],
                   "screen_facts": [fact("Subject 1", "screen", "horizontal", "center", 1)],
                   "frame_count": 33, "shot_size": "close-up", "camera_angle": "unknown", "movement": "push in",
                   "action": "The man lays the rod down on the pier boards."}],
        "events": [{"event_id": "e1", "shot_id": "s02", "before": fact("Subject 1", "relation", "holds", "Subject 2", 1),
                    "after": None, "on_screen": True}]}
    street = {"schema": EVENT_LOG_SCHEMA,
        "subjects": [{"subject_id": "Subject 1", "kind": "person", "appearance": "girl with two braids"},
                     {"subject_id": "Subject 2", "kind": "object", "appearance": "red umbrella"}],
        "locations": [{"location_id": "Location 1", "appearance": "narrow city street", "anchors": []}],
        "initial_state": [fact("Subject 1", "relation", "holds", "Subject 2", 0),
                          fact("Subject 2", "state", "object_condition", "closed", 0),
                          fact("Location 1", "state", "environment", "raining", 0),
                          fact("Subject 1", "relation", "holds", "Subject 2", 1)],
        "shots": [{"shot_id": "s01", "location_id": "Location 1", "present_subjects": ["Subject 1", "Subject 2"],
                   "visible_entities": ["Subject 1", "Subject 2", "Location 1"],
                   "screen_facts": [fact("Subject 1", "screen", "depth", "foreground", 0)],
                   "frame_count": 41, "shot_size": "medium", "camera_angle": "unknown", "movement": "unknown",
                   "action": "The girl walks along the street carrying the umbrella."},
                  {"shot_id": "s02", "location_id": "Location 1", "present_subjects": ["Subject 1", "Subject 2"],
                   "visible_entities": ["Subject 1", "Subject 2", "Location 1"], "screen_facts": [],
                   "frame_count": 45, "shot_size": "medium close-up", "camera_angle": "low", "movement": "unknown",
                   "action": "The girl opens the umbrella above her head."}],
        "events": [{"event_id": "e1", "shot_id": "s02", "before": fact("Subject 2", "state", "object_condition", "closed", 1),
                    "after": fact("Subject 2", "state", "object_condition", "open", 1), "on_screen": True}]}
    return [pier, street]


_WRITER_EXAMPLE_OUTPUTS = [
    {"subjects": [{"subject_id": "Character 1", "kind": "person", "attributes": "old man with a grey beard"},
                  {"subject_id": "Subject 2", "kind": "object", "attributes": "wooden fishing rod"}],
     "locations": [{"location_id": "Location 1", "attributes": "weathered wooden pier"}],
     "shots": [{"shot_id": "s01", "prompt": "Wide shot on the foggy wooden pier. On the left, Character 1, in a yellow raincoat, holds Subject 2 and casts the line out over the water."},
               {"shot_id": "s02", "prompt": "Close-up shot, camera pushes in. Character 1 lays Subject 2 down on the boards and no longer holds it."}]},
    {"subjects": [{"subject_id": "Character 1", "kind": "person", "attributes": "girl with two braids"},
                  {"subject_id": "Subject 2", "kind": "object", "attributes": "red umbrella"}],
     "locations": [{"location_id": "Location 1", "attributes": "narrow city street"}],
     "shots": [{"shot_id": "s01", "prompt": "Medium shot on a narrow city street in the rain. Character 1 walks along holding Subject 2, which is closed."},
               {"shot_id": "s02", "prompt": "Medium close-up shot, low angle shot. Character 1, with Subject 2, opens it above her head."}]},
]
# 示例一的首镜示范“抽中写方位”的镜（只有人物左右一条空间条目标 WRITE，朝向留给规划器）；其余三镜示范不写方位、
# 缺省机位与未换地点不写。


def _writer_example_spatial(index: int, entries: Sequence[FactEntry]) -> frozenset[int]:
    return frozenset(e.number for e in entries if index == 0 and e.role == "spatial" and e.shot_id == "s01"
                     and e.fact.predicate_or_axis == "horizontal")


def drawn_spatial_entries(log: EventLog, entries: Sequence[FactEntry], *, case_id: str, fraction: float,
                         seed: int) -> frozenset[int]:
    """抽中要把方位写进剧本的镜：在有人物或动物左右（horizontal）条目的镜里按 unit_draw 取 fraction，
    返回这些镜里人物与动物的左右条目编号。

    比例取冻结评测逐镜剧本里写了画面方位的镜的占比（长集 23–24%）。只写人物（含动物）的
    左右：前后、朝向、物体方位不写，留给规划器从剧本推出。
    空间条目写没写都留在空间类单列计分；核对确认写到的记在各档的 spatial_written。
    """
    if not 0 <= fraction <= 1:
        raise ValueError("spatial fraction must lie in [0, 1]")
    kinds = {s.subject_id: s.kind for s in log.subjects}
    # 空间类里还有只说画面方位的声明与事件（kind 为 fact、event；事件条目没有 fact），左右只来自屏幕事实（kind=screen）。
    lateral = [e for e in entries if e.role == "spatial" and e.kind == "screen" and e.fact.predicate_or_axis == "horizontal"
               and kinds[e.fact.subject] in ("person", "animal")]
    drawn = {shot for shot in {e.shot_id for e in lateral}
             if unit_draw("storyboard_writer_spatial", seed, case_id, shot) < fraction}
    return frozenset(e.number for e in lateral if e.shot_id in drawn)


def fact_table(entries: Sequence[FactEntry], *, marks: bool, spatial: frozenset[int]) -> str:
    """编号事实表：按镜分组，每条一行；marks=True 时标 WRITE / DO-NOT-WRITE（起因写，可推出不写）。

    spatial 是抽中要写的空间条目编号：列在所属镜下、标 WRITE；其余空间条目不进表。
    """
    if not spatial <= {e.number for e in entries if e.role == "spatial"}:
        raise ValueError("spatial must be spatial entry numbers of this table")
    by_shot: dict[str, list[FactEntry]] = {}
    for e in entries:
        if e.role != "spatial" or e.number in spatial:
            by_shot.setdefault(e.shot_id, []).append(e)
    lines = []
    for shot, rows in by_shot.items():
        lines.append(f"SHOT {shot}")
        for e in rows:
            mark = ("DO-NOT-WRITE " if e.role == "derivable" else "WRITE ") if marks else ""
            lines.append(f"[{e.number}] {mark}{e.text}")
    return "\n".join(lines)


def _table_header(log: EventLog) -> str:
    subjects = ", ".join(f"{script_name(s.subject_id, s.kind)} ({s.kind})" for s in log.subjects) or "none"
    locations = ", ".join(l.location_id for l in log.locations) or "none"
    return f"SUBJECTS: {subjects}\nLOCATIONS: {locations}\nSHOTS: {', '.join(s.shot_id for s in log.shots)}"


def storyboard_writer_messages(log: EventLog, *, spatial: frozenset[int]) -> list[dict]:
    """写手调用：系统提示＋两条虚构示例（表与期望输出）＋本故事的编号事实表。输出可交给 format_shot_tier。"""
    examples = []
    for index, (raw, output) in enumerate(zip(_writer_example_logs(), _WRITER_EXAMPLE_OUTPUTS, strict=True)):
        example = parse_event_log(raw)
        entries = fact_roles(example)
        table = fact_table(entries, marks=True, spatial=_writer_example_spatial(index, entries))
        examples.append(f"EXAMPLE INPUT\n{_table_header(example)}\n{table}\nEXAMPLE OUTPUT\n{json.dumps(output, ensure_ascii=False)}")
    system = STORYBOARD_WRITER_INSTRUCTIONS + "\n" + "\n\n".join(examples)
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"{_table_header(log)}\n{fact_table(fact_roles(log), marks=True, spatial=spatial)}"}]


def storyboard_writer_retry_messages(log: EventLog, added_facts: Sequence[str], *, spatial: frozenset[int]) -> list[dict]:
    """新增事实后的那一次重写：系统提示不变，只把多出来的事实列进用户消息。"""
    if not added_facts:
        raise ValueError("retry requires the facts the checker found outside the table")
    messages = storyboard_writer_messages(log, spatial=spatial)
    messages[-1] = {"role": "user", "content": messages[-1]["content"] + (
        "\n\nThe previous script added facts that are not WRITE entries. Write it again without them:\n"
        + "\n".join(f"- {fact}" for fact in added_facts))}
    return messages


def parse_storyboard_writer(raw: str, log: EventLog) -> dict:
    """写手输出 → format_shot_tier 的输入；主体、地点、镜号必须与日志逐一相同。"""
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != {"subjects", "locations", "shots"}:
        raise ValueError("writer JSON must contain exactly subjects, locations and shots")
    subjects = [(row.get("subject_id"), row.get("kind")) for row in payload["subjects"]]
    if subjects != [(script_name(s.subject_id, s.kind), s.kind) for s in log.subjects]:
        raise ValueError("writer subjects must list the log subjects in order with their kinds")
    if [row.get("location_id") for row in payload["locations"]] != [l.location_id for l in log.locations]:
        raise ValueError("writer locations must list the log locations in order")
    if [row.get("shot_id") for row in payload["shots"]] != [s.shot_id for s in log.shots]:
        raise ValueError("writer shots must list the log shots in order")
    for group, keys, text in (("subjects", {"subject_id", "kind", "attributes"}, "attributes"),
                              ("locations", {"location_id", "attributes"}, "attributes"),
                              ("shots", {"shot_id", "prompt"}, "prompt")):
        for row in payload[group]:
            if set(row) != keys or not isinstance(row[text], str):
                raise ValueError(f"writer {group} rows must contain exactly {sorted(keys)} with string {text}")
    if any(not row["prompt"].strip() for row in payload["shots"]):
        raise ValueError("writer shot prompts must be nonempty")
    return payload


def storyboard_label_messages(log: EventLog, text: str, *, spatial: frozenset[int]) -> list[dict]:
    """核对调用：给带 WRITE / DO-NOT-WRITE 标记的同一张表和逐镜剧本，问写到了哪些编号、哪些 DO-NOT-WRITE 是另行复述、
    有哪些表外事实。标记只用来判复述：动作句或事件已蕴含的状态算写到、不算复述。"""
    if not text.strip():
        raise ValueError("script text must be nonempty")
    body = f"{_table_header(log)}\n{fact_table(fact_roles(log), marks=True, spatial=spatial)}\n\nSCRIPT\n{text}"
    return [{"role": "system", "content": STORYBOARD_LABEL_INSTRUCTIONS}, {"role": "user", "content": body}]


def parse_storyboard_labels(raw: str, entries: Sequence[FactEntry], *, spatial: frozenset[int]) -> dict:
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != {"written", "restated", "added_facts"}:
        raise ValueError("label JSON must contain exactly written, restated and added_facts")
    table = table_entries(entries)
    lists = {}
    for key, allowed in (("written", {e.number for e in table} | spatial),
                         ("restated", {e.number for e in table if e.role == "derivable"})):
        value = payload[key]
        if (not isinstance(value, list) or any(type(n) is not int or n not in allowed for n in value)
                or len(set(value)) != len(value)):
            raise ValueError(f"{key} must list distinct {'DO-NOT-WRITE ' if key == 'restated' else ''}entry numbers of the table")
        lists[key] = sorted(value)
    if not set(lists["restated"]) <= set(lists["written"]):
        raise ValueError("restated entries must also be written")
    added = payload["added_facts"]
    if not isinstance(added, list) or any(not isinstance(f, str) or not f.strip() for f in added):
        raise ValueError("added_facts must be a list of nonempty strings")
    return {**lists, "added_facts": [f.strip() for f in added]}


def write_storyboard_script(log: EventLog, complete: Callable[[list[dict]], str], *, spatial: frozenset[int]) -> dict:
    """一条日志的逐镜剧本与标签。spatial 是抽中要写进剧本的空间条目（drawn_spatial_entries）。

    写手写 → 核对；有表外新增就重写一次再核对，仍有就丢弃（status=discarded）。保留时的标签：写明＝核对确认的写手表条目；
    需推出、空间与自由由 entry_labels 按日志算出。空间条目写没写都在空间类，核对确认写到的另记在 spatial_written
    （写了的算写明、没写的算需推出，供空间类分开读）。剧本只有逐镜这一档。
    """
    validate_event_log(log)
    entries = fact_roles(log)

    def check_script(script: Mapping) -> dict:
        return parse_storyboard_labels(complete(storyboard_label_messages(log, format_shot_tier(script), spatial=spatial)),
                                       entries, spatial=spatial)

    script = parse_storyboard_writer(complete(storyboard_writer_messages(log, spatial=spatial)), log)
    check = check_script(script)
    attempts = 1
    if check["added_facts"]:
        script = parse_storyboard_writer(complete(storyboard_writer_retry_messages(log, check["added_facts"], spatial=spatial)), log)
        check = check_script(script)
        attempts = 2
    row = {"script": script, "spatial_write": sorted(spatial), "text": format_shot_tier(script), "attempts": attempts, **check,
           "spatial_written": sorted(set(check["written"]) & spatial)}
    if check["added_facts"]:
        return {**row, "status": "discarded", "reason": "added_facts_after_retry"}
    return {**row, "status": "kept", "labels": entry_labels(log, [n for n in check["written"] if n not in spatial])}


def storyboard_writer_summary(rows: Sequence[Mapping]) -> dict:
    """三个率，各自分子分母同口径；丢弃按原因另报。

    新增事实率：分母＝进过核对的窗，分子＝其中最后一次核对仍有表外事实的窗。
    复述率：分母＝经过核对的窗的 DO-NOT-WRITE 条目数，分子＝这些窗写到的 DO-NOT-WRITE 条目数。
    方位写到率：分母＝经过核对的窗抽中要写的空间条目数，分子＝核对确认写到的。
    """
    done = [row for row in rows if "error" not in row]
    return {
        "windows": len(rows), "errors": len(rows) - len(done),
        "kept": sum(row["status"] == "kept" for row in done),
        "discarded": dict(sorted(Counter(row["reason"] for row in done if row["status"] == "discarded").items())),
        "added_fact_rate": rate(sum(bool(row["added_facts"]) for row in done), len(done)),
        "restated_entry_rate": rate(sum(len(row["restated"]) for row in done), sum(row["derivable_entries"] for row in done)),
        "spatial_written_rate": rate(sum(len(row["spatial_written"]) for row in done), sum(len(row["spatial_write"]) for row in done)),
        "exact_caption_sentence_leak_windows": rate(sum(bool(row["leakage"]["exact_copied_sentences"]) for row in done), len(done)),
    }


_WRITER_CALL_KINDS = (("write", STORYBOARD_WRITER_INSTRUCTIONS), ("label", STORYBOARD_LABEL_INSTRUCTIONS))


class WriterCallRecorder:
    """写手每次实际发出的请求与原始答复，落在 ``<输出目录>/attempts/<窗>/``。

    沿用标注流水线（``state_pilot_annotation``）的 attempts 约定：每窗一个目录，``attempt.json`` 记状态与各次调用摘要，
    每次调用的 ``request.json``（发给 SDK 的参数）、``response.json``（响应体原样，外加 error 与耗时）、``raw.json``
    （``{"text": 模型原文}``）；标注里同一窗多次调用用 ``repair_`` 前缀区分，这里一窗两到四次调用，前缀是
    ``<序号>-<种类>_``（种类 write／label，由系统提示认出）。窗目录已存在（上次中断留下的）就
    接 ``-retryN``，与标注的重发目录同名约定。进程中断时 ``attempt.json`` 停在 running，最后一次调用的 error 记在里面。
    """

    def __init__(self, attempts: Path) -> None:
        self.attempts = attempts
        self._directory: Path | None = None
        self._state: dict[str, Any] | None = None

    def begin(self, case_id: str) -> Path:
        from mspc.data.state_pilot_annotation import durable_json

        stem = case_id.replace(":", "-")
        if not stem or "/" in stem:
            raise ValueError(f"case id {case_id!r} cannot name an attempt directory")
        self.attempts.mkdir(parents=True, exist_ok=True)
        name, number = stem, 0
        while (self.attempts / name).exists():
            number += 1
            name = f"{stem}-retry{number}"
        self._directory = self.attempts / name
        self._directory.mkdir()
        self._state = {"attempt_id": name, "window_id": case_id, "status": "running", "calls": []}
        durable_json(self._directory / "attempt.json", self._state)
        return self._directory

    def __call__(self, request: Mapping[str, Any], raw: Any, error: str | None, elapsed_seconds: float) -> None:
        from mspc.data.closeai_client import sanitize_log_value
        from mspc.data.state_pilot_annotation import durable_json

        if self._directory is None or self._state is None:
            raise RuntimeError("a model call arrived outside begin()/finish() of a window")
        system = request["messages"][0]["content"]
        kind = next((name for name, text in _WRITER_CALL_KINDS if system.startswith(text)), None)
        if kind is None:
            raise ValueError(f"unrecognized writer call: {system[:60]!r}")
        call = f"{len(self._state['calls']) + 1:02d}-{kind}"
        durable_json(self._directory / f"{call}_request.json", sanitize_log_value(dict(request)))
        body = None if raw is None else (dict(raw) if isinstance(raw, Mapping) else raw.model_dump())
        durable_json(self._directory / f"{call}_response.json", {"error": error, "elapsed_seconds": elapsed_seconds, "response": body})
        choice = ((body or {}).get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content")
        if isinstance(text, str):
            durable_json(self._directory / f"{call}_raw.json", {"text": text})
        self._state["calls"].append({
            "call": call, "request_id": (body or {}).get("id"), "returned_model": (body or {}).get("model"),
            "provider": (body or {}).get("provider"), "finish_reason": choice.get("finish_reason"),
            "usage": (body or {}).get("usage"), "error": error, "elapsed_seconds": elapsed_seconds})
        durable_json(self._directory / "attempt.json", self._state)

    def finish(self, *, error: str | None) -> None:
        from mspc.data.state_pilot_annotation import durable_json

        if self._directory is None or self._state is None:
            raise RuntimeError("finish() without begin()")
        self._state.update(status="window_failed" if error is not None else "succeeded", error=error)
        durable_json(self._directory / "attempt.json", self._state)
        self._directory = self._state = None


STORYBOARD_WRITER_ROWS_KEYS = {"case_id", "status", "event_log", "native_captions"}


def run_storyboard_writer(*, windows_path: Path, output: Path, config: Mapping, expected_windows: int,
                          spatial_fraction: float, spatial_seed: int,
                          complete: Callable[[list[dict]], str], spent: Callable[[], str],
                          recorder: WriterCallRecorder) -> int:
    """对已接受的窗逐个写三档；每窗一行落盘，费用由客户端的硬顶控制。只写不跑：开跑前须人工批准花费。

    spatial_fraction／spatial_seed：有人物左右条目的镜里抽多大比例把方位写进剧本，及抽签种子（见 drawn_spatial_entries）。

    recorder：每次实际发出的请求与原始答复落在 ``output/attempts/<窗>/``（见 :class:`WriterCallRecorder`）；它必须同时
    交给 ``complete`` 背后的客户端，否则目录里只有 attempt.json、没有调用文件。

    续跑：输出目录已存在时，要求其 run_config.json 与本次参数逐项相同，已写进 writer.jsonl 的窗（含记了 error 的）
    跳过，接着往下写；返回跳过的窗数。费用硬顶和 spent 都是每个进程自己的，续跑后的总花费是各次之和。
    """
    windows = read_jsonl(windows_path)
    if any(set(row) != STORYBOARD_WRITER_ROWS_KEYS for row in windows):
        raise ValueError(f"writer input rows must contain exactly {sorted(STORYBOARD_WRITER_ROWS_KEYS)}")
    windows = [row for row in windows if row["status"] == "accepted"]
    if len(windows) != expected_windows or len({row["case_id"] for row in windows}) != len(windows):
        raise ValueError("accepted window count/unique case IDs differ from the explicit expectation")
    if recorder.attempts != output / "attempts":
        raise ValueError(f"recorder must write to {output / 'attempts'}; got {recorder.attempts}")
    if len({row["case_id"].replace(":", "-") for row in windows}) != len(windows):
        raise ValueError("case IDs collide once ':' is replaced for attempt directory names")
    logs = {row["case_id"]: parse_event_log(row["event_log"]) for row in windows}
    for log in logs.values():
        validate_event_log(log)
    run_config = json.dumps({"windows": str(windows_path), "spatial_fraction": spatial_fraction,
                             "spatial_seed": spatial_seed, **config}, indent=2)
    rows = []
    if output.exists():
        if not (output / "run_config.json").is_file() or not (output / "writer.jsonl").is_file():
            raise ValueError(f"{output} exists but holds no writer run to resume")
        if json.loads((output / "run_config.json").read_text()) != json.loads(run_config):
            raise ValueError(f"{output} holds a run with a different config or windows; resume needs identical arguments")
        rows = [json.loads(line) for line in (output / "writer.jsonl").read_text().splitlines() if line.strip()]
        if len({row["case_id"] for row in rows}) != len(rows) or not {row["case_id"] for row in rows} <= set(logs):
            raise ValueError(f"{output}/writer.jsonl has repeated windows or windows outside the accepted set")
    else:
        output.mkdir(parents=True)
        (output / "run_config.json").write_text(run_config)
    done = {row["case_id"] for row in rows}
    with (output / "writer.jsonl").open("a") as handle:
        for window in windows:
            if window["case_id"] in done:
                continue
            log = logs[window["case_id"]]
            entries = fact_roles(log)
            row = {"case_id": window["case_id"], "derivable_entries": sum(e.role == "derivable" for e in entries)}
            spatial = drawn_spatial_entries(log, entries, case_id=window["case_id"], fraction=spatial_fraction, seed=spatial_seed)
            recorder.begin(window["case_id"])
            try:
                row.update(write_storyboard_script(log, complete, spatial=spatial))
                row["leakage"] = {**native_leakage(row["text"], window["native_captions"]),
                                  "quoted": quoted_annotation_leakage(row["text"], window["native_captions"])}
            except (ValueError, KeyError, TypeError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            recorder.finish(error=row.get("error"))
            row["spent_usd"] = spent()
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
    (output / "summary.json").write_text(json.dumps(storyboard_writer_summary(rows), ensure_ascii=False, indent=2))
    return len(done)
