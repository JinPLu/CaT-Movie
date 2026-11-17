from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from mspc.data.closeai_client import RMB_BUDGET_HARD_CAP
from mspc.data.openrouter_client import (
    DEFAULT_OPENROUTER_BASE_URL,
    OPENROUTER_API_KEY_FILE_ENV,
    OPENROUTER_KEY_FILE_RELATIVE,
    PRICING_CONFIG_RELATIVE,
    PRICING_SCHEMA_VERSION,
    USD_BUDGET_HARD_CAP,
    OpenRouterBudgetExceeded,
    OpenRouterClient,
    OpenRouterCostCounter,
    OpenRouterError,
    OpenRouterResponseError,
    build_openrouter_client,
    build_openrouter_cost_counter,
    load_openrouter_api_key,
    load_openrouter_pricing_config,
    reject_openrouter_batch_entry_point,
)
from tests.support.closeai_fakes import fake_chat_completion


FIXTURE_KEY = "test-openrouter-fixture-key"
REPO_PRICING = Path(PRICING_CONFIG_RELATIVE)
LIST_INPUT_USD = "0.7500"
LIST_OUTPUT_USD = "3.7500"
MODEL = "google/gemini-3.7-flash"


def test_whole_video_static_without_sampling_parameters():
    sent = []

    def transport(**request):
        sent.append(request)
        return fake_chat_completion('{"ratings": []}', prompt_tokens=2894, completion_tokens=1291)

    client = OpenRouterClient(api_key=FIXTURE_KEY, base_url=DEFAULT_OPENROUTER_BASE_URL,
        cost_counter=OpenRouterCostCounter(hard_cap_usd="5", prices=_prices(),
            precall_prompt_tokens=32768, precall_completion_tokens=8192),
        timeout_seconds=5, proxy=None, transport=transport)
    response = client.complete_video(model=MODEL, video_data_url="data:video/mp4;base64,AAAA",
        prompt="Rate the complete video.", temperature=None, max_tokens=8192,
        request_options={"reasoning": {"effort": "medium"}}, response_format={"type": "json_object"})
    request = sent[0]
    assert "temperature" not in request
    assert request["messages"][0]["content"][0]["video_url"]["processing"] == "static"
    assert response.usage["completion_tokens"] == 1291


def _pricing_payload(tmp_path: Path, **overrides: object) -> Path:
    payload = {
        "schema_version": PRICING_SCHEMA_VERSION,
        "platform": "openrouter",
        "currency": "USD",
        "hard_cap_usd": "70.00",
        "api_key_file": str(tmp_path / "openrouter_api_key.txt"),
        "base_url": DEFAULT_OPENROUTER_BASE_URL,
        "model": MODEL,
        "input_usd_per_mtok": LIST_INPUT_USD,
        "output_usd_per_mtok": LIST_OUTPUT_USD,
    }
    payload.update(overrides)
    path = tmp_path / "openrouter_pricing_v1.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_key(tmp_path: Path) -> Path:
    path = tmp_path / "openrouter_api_key.txt"
    path.write_text(FIXTURE_KEY, encoding="utf-8")
    return path


def _prices() -> dict[str, dict[str, str]]:
    return {
        MODEL: {
            "input_usd_per_mtok": LIST_INPUT_USD,
            "output_usd_per_mtok": LIST_OUTPUT_USD,
        }
    }


def test_key_loads_from_fixture_path(tmp_path: Path) -> None:
    key_path = _write_key(tmp_path)
    loaded = load_openrouter_api_key(key_path)
    assert loaded == FIXTURE_KEY


def test_key_loads_from_env_fixture_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_path = _write_key(tmp_path)
    monkeypatch.setenv(OPENROUTER_API_KEY_FILE_ENV, str(key_path))
    loaded = load_openrouter_api_key()
    assert loaded == FIXTURE_KEY


def test_key_path_missing_raises() -> None:
    with pytest.raises(OpenRouterError, match="OPENROUTER_API_KEY_FILE"):
        load_openrouter_api_key(None)


def test_usd_cap_is_independent_of_closeai_rmb_cap() -> None:
    assert USD_BUDGET_HARD_CAP == Decimal("70.00")
    assert RMB_BUDGET_HARD_CAP == Decimal("100.00")
    counter = OpenRouterCostCounter(
        hard_cap_usd=USD_BUDGET_HARD_CAP,
        prices=_prices(),
        precall_prompt_tokens=1_000_000,
        precall_completion_tokens=1_000_000,
    )
    assert counter.snapshot()["hard_cap_usd"] == "70.0000"
    assert counter.snapshot()["currency"] == "USD"
    with pytest.raises(ValueError, match="must not exceed the account hard cap 70.0000"):
        OpenRouterCostCounter(
            hard_cap_usd="70.01",
            prices=_prices(),
            precall_prompt_tokens=100,
            precall_completion_tokens=50,
        )
    # 一次运行可以声明比账户硬顶更小的顶，并在它上面拦截。
    small = OpenRouterCostCounter(
        hard_cap_usd="1.00",
        prices=_prices(),
        precall_prompt_tokens=1_000_000,
        precall_completion_tokens=1_000_000,
    )
    with pytest.raises(OpenRouterBudgetExceeded, match="would exceed 1.0000"):
        small.reserve_precall(MODEL)


def test_missing_price_in_config_raises(tmp_path: Path) -> None:
    path = _pricing_payload(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    del raw["input_usd_per_mtok"]
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input_usd_per_mtok"):
        load_openrouter_pricing_config(path)
    with pytest.raises(ValueError, match="input_usd_per_mtok"):
        build_openrouter_cost_counter(
            pricing={"model": MODEL, "hard_cap_usd": "70.00", "output_usd_per_mtok": LIST_OUTPUT_USD},
            precall_prompt_tokens=100,
            precall_completion_tokens=50,
        )


def test_repo_pricing_config_demands_live_prices() -> None:
    with pytest.raises(ValueError, match="input_usd_per_mtok"):
        load_openrouter_pricing_config(REPO_PRICING)
    assert OPENROUTER_KEY_FILE_RELATIVE == "OoVMetric/resource/openrouter_api_key.txt"


def test_batch_entry_point_is_rejected() -> None:
    with pytest.raises(ValueError, match=":batch"):
        reject_openrouter_batch_entry_point(
            base_url=DEFAULT_OPENROUTER_BASE_URL,
            model=f"{MODEL}:batch",
        )
    with pytest.raises(ValueError, match=":batch"):
        reject_openrouter_batch_entry_point(
            base_url="https://openrouter.ai/api/v1/batch",
            model=MODEL,
        )
    with pytest.raises(ValueError, match=":batch"):
        OpenRouterClient(
            api_key=FIXTURE_KEY,
            base_url="https://openrouter.ai/api/v1/batch",
            cost_counter=OpenRouterCostCounter(
                hard_cap_usd=USD_BUDGET_HARD_CAP,
                prices=_prices(),
                precall_prompt_tokens=100,
                precall_completion_tokens=50,
            ),
            timeout_seconds=5,
            proxy=None,
        )
    with pytest.raises(ValueError, match=":batch"):
        build_openrouter_client(
            api_key=FIXTURE_KEY,
            pricing={
                "base_url": DEFAULT_OPENROUTER_BASE_URL,
                "model": f"{MODEL}:batch",
                "hard_cap_usd": "70.00",
                "input_usd_per_mtok": LIST_INPUT_USD,
                "output_usd_per_mtok": LIST_OUTPUT_USD,
            },
            precall_prompt_tokens=100,
            precall_completion_tokens=50,
            timeout_seconds=5,
        )


def test_cost_accounting_from_known_token_counts() -> None:
    counter = OpenRouterCostCounter(
        hard_cap_usd=USD_BUDGET_HARD_CAP,
        prices=_prices(),
        precall_prompt_tokens=2_000_000,
        precall_completion_tokens=1_000_000,
    )
    actual = counter.record_usage(
        MODEL,
        {"prompt_tokens": 2_000_000, "completion_tokens": 1_000_000, "total_tokens": 3_000_000},
    )
    assert actual == Decimal("5.2500")
    snapshot = counter.snapshot()
    assert snapshot["spent_usd"] == "5.2500"
    assert snapshot["remaining_usd"] == "64.7500"
    assert counter.remaining_usd() == Decimal("64.7500")


def test_cap_refuses_the_call_that_would_cross() -> None:
    counter = OpenRouterCostCounter(
        hard_cap_usd=USD_BUDGET_HARD_CAP,
        prices=_prices(),
        precall_prompt_tokens=1_000_000,
        precall_completion_tokens=1_000_000,
        already_spent_usd="69.0000",
    )
    assert counter.remaining_usd() == Decimal("1.0000")
    with pytest.raises(OpenRouterBudgetExceeded, match="would exceed"):
        counter.reserve_precall(MODEL)
    assert counter.snapshot()["spent_usd"] == "69.0000"
    assert counter.remaining_usd() == Decimal("1.0000")


def test_client_refuses_over_cap_before_transport() -> None:
    calls = {"n": 0}

    def transport(**_kwargs):
        calls["n"] += 1
        return fake_chat_completion('{"score": 1}', prompt_tokens=100, completion_tokens=50)

    client = OpenRouterClient(
        api_key=FIXTURE_KEY,
        base_url=DEFAULT_OPENROUTER_BASE_URL,
        cost_counter=OpenRouterCostCounter(
            hard_cap_usd=USD_BUDGET_HARD_CAP,
            prices=_prices(),
            precall_prompt_tokens=1_000_000,
            precall_completion_tokens=1_000_000,
            already_spent_usd="69.0000",
        ),
        timeout_seconds=5,
        transport=transport,
        proxy=None,
    )
    with pytest.raises(OpenRouterBudgetExceeded, match="would exceed"):
        client.complete_text(
            model=MODEL,
            prompt="Return JSON.",
            temperature=0.0,
            max_tokens=32,
            response_format={"type": "json_object"},
        )
    assert calls["n"] == 0
    assert FIXTURE_KEY not in repr(client)


def test_client_records_usage_on_standard_endpoint(tmp_path: Path) -> None:
    pricing = load_openrouter_pricing_config(_pricing_payload(tmp_path))

    def transport(**kwargs):
        assert kwargs["model"] == MODEL
        return fake_chat_completion(
            '{"score": 1}',
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            total_tokens=2_000_000,
        )

    client = build_openrouter_client(
        api_key=FIXTURE_KEY,
        pricing=pricing,
        precall_prompt_tokens=1_000_000,
        precall_completion_tokens=1_000_000,
        timeout_seconds=5,
        transport=transport,
    )
    response = client.complete_text(
        model=MODEL,
        prompt="Return JSON.",
        temperature=0.0,
        max_tokens=32,
        response_format={"type": "json_object"},
    )
    assert response.cost_usd == "4.5000"
    assert client.cost_counter.remaining_usd() == Decimal("65.5000")
    assert FIXTURE_KEY not in repr(client)
    assert FIXTURE_KEY not in str(client)


def test_video_frames_payload_matches_the_closeai_message_shape(tmp_path: Path) -> None:
    """两个 provider 的判官消息必须逐块相同，否则两边的判官读数不可比。"""

    pricing = load_openrouter_pricing_config(_pricing_payload(tmp_path))
    seen: dict[str, object] = {}

    def transport(**kwargs):
        seen["content"] = kwargs["messages"][0]["content"]
        return fake_chat_completion(
            '{"score": 1}', prompt_tokens=10, completion_tokens=10, total_tokens=20
        )

    client = build_openrouter_client(
        api_key=FIXTURE_KEY,
        pricing=pricing,
        precall_prompt_tokens=10,
        precall_completion_tokens=10,
        timeout_seconds=5,
        transport=transport,
    )
    client.complete_video_frames(
        model=MODEL,
        frame_data_urls=["data:image/jpeg;base64,AAAA", "data:image/jpeg;base64,BBBB"],
        prompt="Return JSON.",
        temperature=0.0,
        max_tokens=32,
        request_options={},
        response_format={"type": "json_object"},
        fps=2.0,
    )
    content = seen["content"]
    assert content[0] == {
        "type": "text",
        "text": "VIDEO FRAMES: 2 JPEG frames in chronological order, sampled at 2.0 FPS.",
    }
    assert content[1] == {"type": "text", "text": "frame 1/2"}
    assert content[2] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}
    assert content[3] == {"type": "text", "text": "frame 2/2"}
    assert content[-1] == {"type": "text", "text": "Return JSON."}


def test_video_frames_rejects_a_non_image_data_url(tmp_path: Path) -> None:
    pricing = load_openrouter_pricing_config(_pricing_payload(tmp_path))
    client = build_openrouter_client(
        api_key=FIXTURE_KEY,
        pricing=pricing,
        precall_prompt_tokens=10,
        precall_completion_tokens=10,
        timeout_seconds=5,
        transport=lambda **_kwargs: None,
    )
    with pytest.raises(ValueError, match="data:image/"):
        client.complete_video_frames(
            model=MODEL,
            frame_data_urls=["data:video/mp4;base64,AAAA"],
            prompt="Return JSON.",
            temperature=0.0,
            max_tokens=32,
            request_options={},
            response_format={"type": "json_object"},
        )


def _strict_format() -> dict:
    return {"type": "json_schema", "json_schema": {
        "name": "answers", "strict": True,
        "schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}}


def test_recorder_sees_the_exact_request_and_the_raw_response_or_the_redacted_error(tmp_path: Path) -> None:
    """落盘钩子拿到的是交给传输层的那份请求和原始响应对象；请求层失败时拿到脱敏后的报错，异常照常往外抛。"""

    pricing = load_openrouter_pricing_config(_pricing_payload(tmp_path))
    seen: list[tuple] = []
    sent: list[dict] = []
    outcome: list = [fake_chat_completion('{"score": 1}', prompt_tokens=10, completion_tokens=10, total_tokens=20),
                     RuntimeError(f"upstream refused {FIXTURE_KEY}")]

    def transport(**kwargs):
        sent.append(kwargs)
        result = outcome.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    client = build_openrouter_client(
        api_key=FIXTURE_KEY, pricing=pricing, precall_prompt_tokens=10, precall_completion_tokens=10,
        timeout_seconds=5, transport=transport, recorder=lambda *args: seen.append(args),
    )
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Return JSON."}]
    call = dict(model=MODEL, messages=messages, temperature=0.0, max_tokens=32, request_options={}, response_format={"type": "json_object"})
    first = client.complete_messages(**call)
    with pytest.raises(RuntimeError, match="upstream refused"):
        client.complete_messages(**call)
    (request, raw, error, elapsed), (failed_request, no_raw, failed, failed_elapsed) = seen
    assert request == sent[0] and request["messages"] == messages and request["response_format"] == {"type": "json_object"}
    assert raw.choices[0].message.content == '{"score": 1}' == first.text and error is None and elapsed >= 0
    assert failed_request == sent[1] and no_raw is None and failed == "upstream refused <redacted>" and failed_elapsed >= 0


@pytest.mark.parametrize("proxy", [None, "http://127.0.0.1:7890"])
@pytest.mark.parametrize("timeout_seconds", [5, 300])
def test_sdk_video_upload_uses_configured_request_timeout(monkeypatch, proxy, timeout_seconds):
    """Inspect the actual SDK request sent to an offline HTTP transport."""
    import httpx
    import openai

    sent = []
    proxy_timeouts = []

    def handle(request):
        sent.append(request)
        return httpx.Response(200, json={
            "id": "chatcmpl-timeout-test", "object": "chat.completion", "created": 0,
            "model": MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        })

    real_openai = openai.OpenAI

    class OfflineOpenAI(real_openai):
        def __init__(self, *, http_client=None, **kwargs):
            if http_client is not None:
                proxy_timeouts.append(http_client.timeout.as_dict())
                http_client.close()
            super().__init__(http_client=httpx.Client(transport=httpx.MockTransport(handle), trust_env=False),
                             **kwargs)

    monkeypatch.setattr(openai, "OpenAI", OfflineOpenAI)
    client = OpenRouterClient(api_key=FIXTURE_KEY, base_url=DEFAULT_OPENROUTER_BASE_URL,
        cost_counter=OpenRouterCostCounter(hard_cap_usd="1", prices=_prices(),
            precall_prompt_tokens=10, precall_completion_tokens=10),
        timeout_seconds=timeout_seconds, proxy=proxy)
    try:
        client.complete_video(model=MODEL, video_data_url="data:video/mp4;base64,AAAA",
            prompt="Rate the complete video.", temperature=None, max_tokens=32,
            request_options={"reasoning": {"effort": "low"}}, response_format={"type": "json_object"})
        expected = {"connect": 15.0, "read": float(timeout_seconds),
                    "write": float(timeout_seconds), "pool": 15.0}
        assert len(sent) == 1
        assert sent[0].extensions["timeout"] == expected
        assert proxy_timeouts == ([] if proxy is None else [expected])
        assert client._sdk_client.max_retries == 0
        body = json.loads(sent[0].content)
        assert body["messages"][0]["content"][0]["video_url"] == {
            "url": "data:video/mp4;base64,AAAA", "processing": "static"}
    finally:
        client.close()


@pytest.mark.parametrize("failure", ["ssl_eof", "http_502"])
def test_sdk_failure_keeps_redacted_cause_and_http_request_id_without_retry(monkeypatch, failure):
    """Exercise the real SDK error wrapping without opening a socket."""
    import ssl

    import httpx
    import openai

    sent = []

    def handle(request):
        sent.append(request)
        if failure == "ssl_eof":
            try:
                raise ssl.SSLError(f"[SSL: UNEXPECTED_EOF_WHILE_READING] EOF {FIXTURE_KEY}")
            except ssl.SSLError as cause:
                raise httpx.ReadError(f"TLS read failed {FIXTURE_KEY}", request=request) from cause
        return httpx.Response(502, headers={"x-request-id": f"http-error-{FIXTURE_KEY}"},
            json={"error": {"message": f"upstream failed {FIXTURE_KEY}", "type": "server_error"}})

    real_openai = openai.OpenAI

    class OfflineOpenAI(real_openai):
        def __init__(self, *, http_client=None, **kwargs):
            if http_client is not None:
                http_client.close()
            super().__init__(http_client=httpx.Client(transport=httpx.MockTransport(handle), trust_env=False),
                             **kwargs)

    monkeypatch.setattr(openai, "OpenAI", OfflineOpenAI)
    recorded = []
    client = OpenRouterClient(api_key=FIXTURE_KEY, base_url=DEFAULT_OPENROUTER_BASE_URL,
        cost_counter=OpenRouterCostCounter(hard_cap_usd="1", prices=_prices(),
            precall_prompt_tokens=10, precall_completion_tokens=10),
        timeout_seconds=5, proxy=None, recorder=lambda *args: recorded.append(args))
    try:
        with pytest.raises(OpenRouterError) as raised:
            client.complete_text(model=MODEL, prompt="Return JSON.", temperature=0,
                max_tokens=32, response_format={"type": "json_object"})
        error = raised.value
        assert len(sent) == 1 and len(recorded) == 1
        assert client._sdk_client.max_retries == 0
        assert recorded[0][1] is None and recorded[0][2] == str(error)
        assert FIXTURE_KEY not in str(error)
        assert all(FIXTURE_KEY not in item["message"] for item in error.exception_chain)
        names = [item["type"].rsplit(".", 1)[-1] for item in error.exception_chain]
        if failure == "ssl_eof":
            assert names == ["APIConnectionError", "ReadError", "SSLError"]
            assert error.http_request_id is None and error.status_code is None
            assert "UNEXPECTED_EOF_WHILE_READING" in error.exception_chain[-1]["message"]
        else:
            assert names == ["InternalServerError", "HTTPStatusError"]
            assert error.http_request_id == "http-error-<redacted>" and error.status_code == 502
    finally:
        sdk_client = client._sdk_client
        client.close()
    assert sdk_client.is_closed()
    assert client._sdk_client is None


def test_close_is_idempotent_and_does_not_initialize_a_transport_client(monkeypatch):
    client = OpenRouterClient(api_key=FIXTURE_KEY, base_url=DEFAULT_OPENROUTER_BASE_URL,
        cost_counter=OpenRouterCostCounter(hard_cap_usd="1", prices=_prices(),
            precall_prompt_tokens=10, precall_completion_tokens=10),
        timeout_seconds=5, proxy=None, transport=lambda **request: None)
    monkeypatch.setattr(client, "_sdk", lambda: pytest.fail("close must not initialize the SDK"))
    client.close()
    client.close()
    assert client._sdk_client is None


def test_response_format_is_forwarded_verbatim_per_call(tmp_path: Path) -> None:
    """客户端不替调用方选格式：json_object 与 json_schema 都原样进请求体，且每次调用各带各的。"""

    pricing = load_openrouter_pricing_config(_pricing_payload(tmp_path))
    sent: list[dict] = []

    def transport(**kwargs):
        sent.append(kwargs)
        return fake_chat_completion('{"score": 1}', prompt_tokens=10, completion_tokens=10, total_tokens=20)

    client = build_openrouter_client(
        api_key=FIXTURE_KEY, pricing=pricing, precall_prompt_tokens=10, precall_completion_tokens=10,
        timeout_seconds=5, transport=transport,
    )
    for response_format in ({"type": "json_object"}, _strict_format()):
        client.complete_video_frames(
            model=MODEL, frame_data_urls=["data:image/jpeg;base64,AAAA"], prompt="Return JSON.",
            temperature=1.0, max_tokens=32, request_options={}, response_format=response_format,
        )
    assert [request["response_format"] for request in sent] == [{"type": "json_object"}, _strict_format()]


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"type": "text"}, "json_object or json_schema"),
        ({"type": "json_object", "extra": 1}, "no other keys"),
        ({"type": "json_schema", "json_schema": {"name": "x", "strict": False, "schema": {}}}, "strict=true"),
        ({"type": "json_schema"}, "exactly"),
        ("json_object", "mapping"),
    ],
)
def test_response_format_is_validated_before_any_spend(tmp_path: Path, bad, message) -> None:
    pricing = load_openrouter_pricing_config(_pricing_payload(tmp_path))
    client = build_openrouter_client(
        api_key=FIXTURE_KEY, pricing=pricing, precall_prompt_tokens=10, precall_completion_tokens=10,
        timeout_seconds=5, transport=lambda **_kwargs: pytest.fail("must not reach the transport"),
    )
    with pytest.raises(ValueError, match=message):
        client.complete_text(model=MODEL, prompt="Return JSON.", temperature=0.0, max_tokens=32, response_format=bad)


def test_served_records_the_returned_model_finish_reason_and_reasoning_tokens(tmp_path: Path) -> None:
    """响应体回报的模型名、结束原因与思考 token 进 served；固件是 2026-10-11 写手探针的真实响应体。"""

    fixture = json.loads((Path(__file__).resolve().parents[2] / "fixtures/openrouter/gemini38_flash_reasoning_medium_response_20261011.json").read_text(encoding="utf-8"))
    pricing = load_openrouter_pricing_config(_pricing_payload(tmp_path))
    client = build_openrouter_client(
        api_key=FIXTURE_KEY, pricing=pricing, precall_prompt_tokens=10, precall_completion_tokens=10,
        timeout_seconds=5, transport=lambda **_kwargs: fixture["response"],
    )
    response = client.complete_messages(
        model=MODEL, messages=[{"role": "user", "content": "Return JSON."}], temperature=1.0, max_tokens=65536,
        request_options={"reasoning": {"effort": "medium"}}, response_format={"type": "json_object"},
    )
    assert response.model == MODEL  # 请求的模型名；响应回报的在 served 里
    assert response.served["model"] == "google/gemini-3.8-flash"
    assert response.served["finish_reason"] == "stop"
    assert response.served["provider"] == "Google AI Studio" and response.served["service_tier"] == "default"
    assert response.served["reasoning_tokens"] == 2769 and response.usage["completion_tokens"] == 2802
    assert response.served["reported_cost_usd"] == "0.01202175"


@pytest.mark.parametrize("failure", ["text", "usage", "budget"])
def test_received_response_failure_retains_usage_and_incurred_cost(failure: str) -> None:
    """Replay real usage, perturb only the failing field, and never touch the network."""
    raw = json.loads((Path(__file__).resolve().parents[2] /
        "fixtures/openrouter/gemini38_flash_reasoning_medium_response_20261011.json").read_text())["response"]
    if failure == "text":
        raw["choices"][0]["message"]["content"] = ""
        raw["choices"][0]["finish_reason"] = "length"
    elif failure == "usage":
        del raw["usage"]["prompt_tokens"]
    counter = OpenRouterCostCounter(hard_cap_usd="0.01" if failure == "budget" else "1.00",
        prices=_prices(), precall_prompt_tokens=1, precall_completion_tokens=1)
    client = OpenRouterClient(api_key=FIXTURE_KEY, base_url=DEFAULT_OPENROUTER_BASE_URL,
        cost_counter=counter, timeout_seconds=5, proxy=None, transport=lambda **kw: raw)
    with pytest.raises(OpenRouterResponseError) as raised:
        client.complete_text(model=MODEL, prompt="Return JSON.", temperature=1.0,
                             max_tokens=32768, response_format={"type": "json_object"})
    error = raised.value
    assert error.stage == failure and error.raw_response == raw
    assert error.served["reasoning_tokens"] == 2769
    assert error.served["reported_cost_usd"] == "0.01202175"
    assert error.cost_usd == (None if failure == "usage" else "0.0120")
    assert counter.snapshot()["spent_usd"] == ("0.0000" if failure == "usage" else "0.0120")
    if failure == "budget":
        with pytest.raises(OpenRouterBudgetExceeded):
            counter.reserve_precall(MODEL)
