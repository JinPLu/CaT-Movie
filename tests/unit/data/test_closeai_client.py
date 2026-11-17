from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from mspc.data.closeai_client import (
    CLOSEAI_API_KEY_FILE_ENV,
    CloseAIBudgetExceeded,
    CloseAIClient,
    CloseAICostCounter,
    CloseAIEmptyStream,
    CloseAIError,
    DEFAULT_CLOSEAI_KEY_PATH,
    RMB_BUDGET_HARD_CAP,
    _reassemble_stream,
    _reject_truncated_response,
    display_cny,
    load_closeai_api_key,
    redact_secrets,
    sanitize_invocation,
    sanitize_log_value,
)
from tests.support.closeai_fakes import fake_chat_completion


SECRET = "sk-super-secret-test-key-do-not-leak"


def _prices() -> dict[str, dict[str, str]]:
    return {"qwen3.7-plus": {"input_cny_per_mtok": "1.728", "output_cny_per_mtok": "6.912"}}


def _counter(*, cap: str = "1.00") -> CloseAICostCounter:
    return CloseAICostCounter(
        hard_cap_cny=cap,
        prices=_prices(),
        precall_prompt_tokens=6000,
        precall_completion_tokens=500,
    )


def test_cost_counter_uses_rmb_and_stops_at_cap() -> None:
    counter = _counter(cap="0.02")
    first = counter.estimate_cny("qwen3.7-plus")
    assert first > 0
    counter.record_usage(
        "qwen3.7-plus",
        {"prompt_tokens": 6000, "completion_tokens": 500, "total_tokens": 6500},
    )
    snapshot = counter.snapshot()
    assert snapshot["currency"] == "CNY"
    assert "api_key" not in snapshot
    assert SECRET not in str(snapshot)
    with pytest.raises(CloseAIBudgetExceeded, match="budget"):
        counter.reserve_precall("qwen3.7-plus")
    assert RMB_BUDGET_HARD_CAP == Decimal("100.00")
    resumed = CloseAICostCounter(
        hard_cap_cny="0.02",
        prices=_prices(),
        precall_prompt_tokens=6000,
        precall_completion_tokens=500,
        already_spent_cny="0.02",
    )
    assert resumed.snapshot()["spent_cny"] == "0.0200"
    with pytest.raises(CloseAIBudgetExceeded, match="budget"):
        resumed.reserve_precall("qwen3.7-plus")


def test_repr_logs_and_errors_never_contain_the_key() -> None:
    def transport(**_kwargs):
        raise RuntimeError(f"upstream rejected {SECRET}")

    client = CloseAIClient(
        api_key=SECRET,
        base_url="https://api.openai-proxy.org/v1",
        cost_counter=_counter(),
        timeout_seconds=5,
        max_retries=1,
        retry_base_seconds=0.01,
        rpm_limit=1000,
        transport=transport,
    )
    assert SECRET not in repr(client)
    assert SECRET not in str(client)
    logged = client.request_log(
        {
            "authorization": f"Bearer {SECRET}",
            "api_key": SECRET,
            "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}],
        }
    )
    assert SECRET not in str(logged)
    assert "authorization" not in logged
    assert "api_key" not in logged
    assert logged["content"][0]["image_url"]["url"] == "data:image/jpeg;base64,<omitted>"
    with pytest.raises(CloseAIError) as exc_info:
        client.complete_multi_image(
            model="qwen3.7-plus",
            image_data_urls=["data:image/jpeg;base64,AAAA"],
            prompt="Return JSON.",
            temperature=0.0,
            max_tokens=32,
        )
    assert SECRET not in str(exc_info.value)
    assert SECRET not in redact_secrets(f"key={SECRET}", (SECRET,))
    assert sanitize_invocation(["--api-key-file", "/secret/path", "--live"]) == [
        "--api-key-file",
        "<redacted>",
        "--live",
    ]


def _stream_chunk(*, content=None, finish=None, usage=None, chunk_id=None):
    delta = SimpleNamespace(content=content)
    choice = SimpleNamespace(delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=usage, id=chunk_id)


def test_reassemble_stream_collects_content_and_usage() -> None:
    usage = SimpleNamespace(prompt_tokens=9000, completion_tokens=40, total_tokens=9040)
    chunks = [
        _stream_chunk(content='{"sub', chunk_id="req-1"),
        _stream_chunk(content='questions": {}}'),
        _stream_chunk(finish="stop"),
        _stream_chunk(usage=usage),
    ]
    response = _reassemble_stream(iter(chunks))
    assert response.choices[0].message.content == '{"subquestions": {}}'
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.prompt_tokens == 9000
    assert response.id == "req-1"


def test_reassemble_stream_empty_raises_retryable() -> None:
    chunks = [_stream_chunk(content=""), _stream_chunk(finish="stop")]
    with pytest.raises(CloseAIEmptyStream):
        _reassemble_stream(iter(chunks))


def test_truncated_response_is_not_retried() -> None:
    """finish_reason=length and HTTP 400 are not retryable (CloseAI quota)."""

    calls = {"n": 0}

    def transport(**kwargs):
        calls["n"] += 1
        return fake_chat_completion(
            '{"',
            finish_reason="length",
            prompt_tokens=9000,
            completion_tokens=60,
            total_tokens=9060,
            request_id="chatcmpl-test",
        )

    client = CloseAIClient(
        api_key=SECRET,
        base_url="https://api.openai-proxy.org/v1",
        cost_counter=_counter(),
        timeout_seconds=5,
        max_retries=3,
        retry_base_seconds=0.01,
        rpm_limit=1000,
        transport=transport,
    )
    with pytest.raises(CloseAIError, match="max_tokens"):
        client.complete_video_frames(
            model="qwen3.7-plus",
            frame_data_urls=["data:image/jpeg;base64,AAAA"],
            prompt="Return JSON.",
            temperature=0.0,
            max_tokens=32,
        )
    assert calls["n"] == 1


def test_http_400_is_not_retried() -> None:
    calls = {"n": 0}

    def transport(**kwargs):
        del kwargs
        calls["n"] += 1
        error = RuntimeError("max tokens")
        error.status_code = 400
        raise error

    client = CloseAIClient(
        api_key=SECRET,
        base_url="https://api.openai-proxy.org/v1",
        cost_counter=_counter(),
        timeout_seconds=5,
        max_retries=3,
        retry_base_seconds=0.01,
        rpm_limit=1000,
        transport=transport,
    )
    with pytest.raises(CloseAIError, match="max tokens"):
        client.complete_text(
            model="qwen3.7-plus",
            prompt="Return JSON.",
            temperature=0.0,
            max_tokens=32,
        )
    assert calls["n"] == 1


def test_empty_proxy_body_is_retried_not_lost() -> None:
    """HTTP 200 with an empty body is a proxy flake, not a completion."""

    calls = {"n": 0}

    def transport(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise json.JSONDecodeError("Expecting value", "", 0)
        return fake_chat_completion(
            '{"ok": true}',
            prompt_tokens=9000,
            completion_tokens=20,
            total_tokens=9020,
            request_id="chatcmpl-test",
        )

    client = CloseAIClient(
        api_key=SECRET,
        base_url="https://api.openai-proxy.org/v1",
        cost_counter=_counter(),
        timeout_seconds=5,
        max_retries=3,
        retry_base_seconds=0.01,
        rpm_limit=1000,
        transport=transport,
    )
    response = client.complete_video_frames(
        model="qwen3.7-plus",
        frame_data_urls=["data:image/jpeg;base64,AAAA"],
        prompt="Return JSON.",
        temperature=0.0,
        max_tokens=32,
        fps=2.0,
    )
    assert calls["n"] == 2
    assert response.usage["prompt_tokens"] == 9000


def test_video_frames_payload_sends_ordered_image_parts() -> None:
    seen: dict[str, object] = {}

    def transport(**kwargs):
        seen["content"] = kwargs["messages"][0]["content"]
        return fake_chat_completion(
            '{"ok": true}',
            prompt_tokens=9000,
            completion_tokens=20,
            total_tokens=9020,
            request_id="chatcmpl-test",
        )

    client = CloseAIClient(
        api_key=SECRET,
        base_url="https://api.openai-proxy.org/v1",
        cost_counter=_counter(),
        timeout_seconds=5,
        max_retries=1,
        retry_base_seconds=0.01,
        rpm_limit=1000,
        transport=transport,
    )
    client.complete_video_frames(
        model="qwen3.7-plus",
        frame_data_urls=[
            "data:image/jpeg;base64,AAAA",
            "data:image/jpeg;base64,BBBB",
        ],
        prompt="Return JSON.",
        temperature=0.0,
        max_tokens=32,
        fps=2.0,
    )
    content = seen["content"]
    images = [part for part in content if part.get("type") == "image_url"]
    assert len(images) == 2
    labels = [part["text"] for part in content if part.get("type") == "text"]
    assert "frame 1/2" in labels
    assert "frame 2/2" in labels
    assert content[-1]["text"] == "Return JSON."

    with pytest.raises(ValueError, match="frame_data_urls"):
        client.complete_video_frames(
            model="qwen3.7-plus",
            frame_data_urls=[],
            prompt="Return JSON.",
            temperature=0.0,
            max_tokens=32,
        )
    with pytest.raises(ValueError, match="data:image"):
        client.complete_video_frames(
            model="qwen3.7-plus",
            frame_data_urls=["data:video/mp4;base64,AAAA"],
            prompt="Return JSON.",
            temperature=0.0,
            max_tokens=32,
        )


def test_retry_then_success_records_usage_without_leaking_images() -> None:
    calls = {"n": 0}

    def transport(**kwargs):
        calls["n"] += 1
        assert kwargs["extra_body"] == {"enable_thinking": False}
        if calls["n"] == 1:
            error = RuntimeError("temporary")
            error.status_code = 429
            raise error
        return fake_chat_completion(
            '{"ok": true}',
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            request_id="chatcmpl-test",
        )

    client = CloseAIClient(
        api_key=SECRET,
        base_url="https://api.openai-proxy.org/v1",
        cost_counter=_counter(),
        timeout_seconds=5,
        max_retries=3,
        retry_base_seconds=0.01,
        rpm_limit=1000,
        transport=transport,
    )
    response = client.complete_multi_image(
        model="qwen3.7-plus",
        image_data_urls=["data:image/jpeg;base64,AAAA"],
        prompt="Return JSON.",
        temperature=0.0,
        max_tokens=32,
    )
    assert calls["n"] == 2
    assert response.usage["prompt_tokens"] == 100
    assert Decimal(response.cost_cny) > 0
    assert SECRET not in response.text
    assert display_cny(Decimal(response.cost_cny)) == response.cost_cny


def test_sdk_connection_error_is_not_reconstructed() -> None:
    class APIConnectionError(Exception):
        def __init__(self) -> None:
            Exception.__init__(self, "Connection error.")

    class _SDK:
        class chat:
            class completions:
                @staticmethod
                def create(**_kwargs):
                    raise APIConnectionError()

    client = CloseAIClient(
        api_key=SECRET,
        base_url="https://api.openai-proxy.org/v1",
        cost_counter=_counter(),
        timeout_seconds=5,
        max_retries=1,
        retry_base_seconds=0.01,
        rpm_limit=1000,
    )
    client._sdk_client = _SDK()
    with pytest.raises(CloseAIError, match="Connection error") as exc_info:
        client.complete_multi_image(
            model="qwen3.7-plus",
            image_data_urls=["data:image/jpeg;base64,AAAA"],
            prompt="Return JSON.",
            temperature=0.0,
            max_tokens=32,
        )
    assert "takes 1 positional argument" not in str(exc_info.value)
    assert SECRET not in str(exc_info.value)


def test_complete_text_is_json_only_and_disables_thinking() -> None:
    seen: dict[str, object] = {}

    def transport(**kwargs):
        seen.update(kwargs)
        return fake_chat_completion(
            '{"ok": true}',
            prompt_tokens=40,
            completion_tokens=8,
            total_tokens=48,
            request_id="chatcmpl-text",
        )

    client = CloseAIClient(
        api_key=SECRET,
        base_url="https://api.openai-proxy.org/v1",
        cost_counter=_counter(),
        timeout_seconds=5,
        max_retries=1,
        retry_base_seconds=0.01,
        rpm_limit=1000,
        transport=transport,
    )
    response = client.complete_text(
        model="qwen3.7-plus",
        prompt="Return JSON.",
        temperature=0.0,
        max_tokens=32,
        enable_thinking=False,
    )
    assert seen["model"] == "qwen3.7-plus"
    assert seen["response_format"] == {"type": "json_object"}
    assert seen["extra_body"] == {"enable_thinking": False}
    assert seen["messages"] == [{"role": "user", "content": "Return JSON."}]
    assert SECRET not in str(seen)
    assert response.usage["prompt_tokens"] == 40
    assert Decimal(response.cost_cny) > 0


def test_image_generation_model_names_are_rejected() -> None:
    counter = _counter()
    with pytest.raises(ValueError, match="image generation"):
        counter.price_for("gemini-3.7-flash-image")


def test_load_closeai_api_key_requires_explicit_path(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(CLOSEAI_API_KEY_FILE_ENV, raising=False)
    assert DEFAULT_CLOSEAI_KEY_PATH is None
    with pytest.raises(CloseAIError, match="must be provided"):
        load_closeai_api_key()
    key_file = tmp_path / "close_api_key.txt"
    key_file.write_text("sk-test-key-from-file", encoding="utf-8")
    assert load_closeai_api_key(key_file) == "sk-test-key-from-file"
    monkeypatch.setenv(CLOSEAI_API_KEY_FILE_ENV, str(key_file))
    assert load_closeai_api_key() == "sk-test-key-from-file"


def test_missing_finish_reason_is_rejected() -> None:
    with pytest.raises(CloseAIError, match="finish_reason"):
        _reject_truncated_response(SimpleNamespace())
    with pytest.raises(CloseAIError, match="finish_reason"):
        _reject_truncated_response(
            SimpleNamespace(choices=[SimpleNamespace(finish_reason=None)])
        )
    kept = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")])
    assert _reject_truncated_response(kept) is kept


def test_complete_messages_sends_caller_messages_unchanged() -> None:
    seen: dict[str, object] = {}

    def transport(**kwargs):
        seen.update(kwargs)
        return fake_chat_completion('{"ok": true}', prompt_tokens=30, completion_tokens=5)

    client = CloseAIClient(
        api_key=SECRET,
        base_url="https://api.openai-proxy.org/v1",
        cost_counter=_counter(),
        timeout_seconds=5,
        max_retries=1,
        retry_base_seconds=0.01,
        rpm_limit=1000,
        transport=transport,
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You design."}]},
        {"role": "user", "content": [{"type": "text", "text": "Story. Output JSON."}]},
    ]
    response = client.complete_messages(
        model="qwen3.7-plus", messages=messages, temperature=0.0,
        response_format={"type": "json_object"}, max_tokens=64,
    )
    assert seen["messages"] == messages
    assert seen["response_format"] == {"type": "json_object"}
    assert seen["extra_body"] == {"enable_thinking": False}
    assert response.usage["completion_tokens"] == 5
    assert response.provider_billed_cost is None

    def billed_transport(**kwargs):
        del kwargs
        response = fake_chat_completion('{"ok": true}', prompt_tokens=30, completion_tokens=5)
        response.usage.cost = "0.0123"
        return response

    billed_client = CloseAIClient(
        api_key=SECRET,
        base_url="https://api.openai-proxy.org/v1",
        cost_counter=_counter(),
        timeout_seconds=5,
        max_retries=1,
        retry_base_seconds=0.01,
        rpm_limit=1000,
        transport=billed_transport,
    )
    billed = billed_client.complete_text(
        model="qwen3.7-plus", prompt="Return JSON.", temperature=0.0, max_tokens=32,
    )
    assert billed.provider_billed_cost == "0.0123"
    with pytest.raises(ValueError, match="neither non-empty text"):
        client.complete_messages(
            model="qwen3.7-plus",
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x"}}]}],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=64,
        )


def test_complete_messages_leaves_out_none_response_format_and_max_tokens() -> None:
    seen: dict[str, object] = {}

    def transport(**kwargs):
        seen.update(kwargs)
        return fake_chat_completion('{"ok": true}', prompt_tokens=30, completion_tokens=5)

    client = CloseAIClient(api_key=SECRET, base_url="https://api.openai-proxy.org/v1",
        cost_counter=_counter(), timeout_seconds=5, max_retries=1,
        retry_base_seconds=0.01, rpm_limit=1000, transport=transport)
    messages = [{"role": "user", "content": "Return JSON."}]
    client.complete_messages(model="qwen3.7-plus", messages=messages, temperature=0.6,
        response_format=None, max_tokens=None, enable_thinking=True)
    assert "response_format" not in seen and "max_tokens" not in seen
    schema = {"type": "json_schema", "json_schema": {"name": "answer", "strict": True,
                                                     "schema": {"type": "object"}}}
    client.complete_messages(model="qwen3.7-plus", messages=messages, temperature=0,
        response_format=schema, max_tokens=None)
    assert seen["response_format"] == schema and "max_tokens" not in seen
    with pytest.raises(TypeError):
        client.complete_messages(model="qwen3.7-plus", messages=messages, temperature=0, max_tokens=8)
    with pytest.raises(ValueError, match="response_format"):
        client.complete_messages(model="qwen3.7-plus", messages=messages, temperature=0,
            response_format={"type": "text"}, max_tokens=8)


def test_mixed_video_keyframe_and_reasoning_audit():
    seen = {}
    def transport(**kwargs):
        seen.update(kwargs)
        raw = fake_chat_completion('{"ok":true}', prompt_tokens=30, completion_tokens=5)
        raw.choices[0].message.reasoning_content = "visible reasoning"
        raw.usage.prompt_tokens_details = {"video_tokens": 20, "image_tokens": 5}
        return raw
    client = CloseAIClient(api_key=SECRET, base_url="https://api.openai-proxy.org/v1",
        cost_counter=_counter(), timeout_seconds=5, max_retries=1,
        retry_base_seconds=0.01, rpm_limit=1000, transport=transport)
    messages = [{"role":"user", "content":[
        {"type":"video_url", "video_url":{"url":"data:video/mp4;base64,YQ=="}, "fps":2},
        {"type":"image_url", "image_url":{"url":"data:image/jpeg;base64,YQ=="}},
        {"type":"text", "text":"Return JSON"}]}]
    result = client.complete_messages(model="qwen3.7-plus", messages=messages,
        temperature=0.6, response_format={"type": "json_object"}, max_tokens=32768, enable_thinking=True)
    assert seen["messages"] == messages
    assert result.raw_usage["prompt_tokens_details"]["video_tokens"] == 20
    assert result.reasoning_characters == len("visible reasoning")
    assert result.finish_reason == "stop"
    assert "YQ==" not in str(client.request_log(seen))


def test_stream_keeps_reasoning_for_audit():
    chunk = _stream_chunk(content='{"ok":true}', finish="stop")
    chunk.choices[0].delta.reasoning_content = "checked the video"
    chunk.model = "qwen3.8-flash-2026-09-30"
    raw = _reassemble_stream(iter([chunk, _stream_chunk(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))]))
    assert raw.choices[0].message.reasoning_content == "checked the video"
    assert raw.model == "qwen3.8-flash-2026-09-30"
