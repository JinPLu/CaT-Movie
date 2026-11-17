"""CloseAI OpenAI-compatible client with RMB cost control and no-secret logs."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from types import SimpleNamespace
import os
import threading
import time
from typing import Any, Callable, Mapping, Sequence


DEFAULT_CLOSEAI_BASE_URL = "https://api.openai-proxy.org/v1"
CLOSEAI_API_KEY_FILE_ENV = "CLOSEAI_API_KEY_FILE"
# No machine-local default. Callers must pass a path or set CLOSEAI_API_KEY_FILE.
# Kept as None so existing `args.api_key_file or DEFAULT_CLOSEAI_KEY_PATH` imports
# still resolve and then fail loud inside load_closeai_api_key.
DEFAULT_CLOSEAI_KEY_PATH: Path | None = None
RMB_BUDGET_HARD_CAP = Decimal("100.00")
_CNY_QUANT = Decimal("0.0001")
_CNY_DISPLAY = Decimal("0.0001")
# CloseAI: HTTP 400 (including max-token / length failures) is not retryable;
# a 400 burns several times the rate-limit quota.
# https://doc.closeai-asia.com/tutorial/api/rate-limit.html
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class CloseAIError(RuntimeError):
    """Raised for CloseAI request or credential failures. Messages are redacted."""


class CloseAIBudgetExceeded(CloseAIError):
    """Raised before a call that would cross the RMB hard cap."""


class CloseAIResponseTruncated(CloseAIError):
    """Raised when the model stopped at ``max_tokens`` before finishing its JSON.

    Gemini bills hidden reasoning against ``max_tokens``, so a budget that looks
    generous for the visible answer can still cut the JSON mid-token. Surfacing
    this separately keeps a truncated answer from being reported as a parse bug.
    """


class CloseAIEmptyStream(CloseAIError):
    """Raised when a streamed completion yielded no assistant content.

    The proxy gateway returns an empty HTTP 200 on non-streamed multimodal calls
    whose upstream generation outlasts its response-buffer budget. Streaming
    avoids that idle cut; an empty stream is still never a valid completion, so
    it is retried rather than scored.
    """


def cny_amount(value: Decimal | int | float | str) -> Decimal:
    return Decimal(str(value)).quantize(_CNY_QUANT, rounding=ROUND_HALF_UP)


def display_cny(value: Decimal) -> str:
    return str(value.quantize(_CNY_DISPLAY, rounding=ROUND_HALF_UP))


def redact_secrets(text: str, secrets: Sequence[str]) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    redacted = text
    for secret in secrets:
        if not isinstance(secret, str) or not secret:
            continue
        redacted = redacted.replace(secret, "<redacted>")
    return redacted


def load_closeai_api_key(path: Path | str | None = None) -> str:
    """Read the owner key file. The return value must stay in memory only."""

    provided = path if path is not None else os.environ.get(CLOSEAI_API_KEY_FILE_ENV)
    if provided is None or not str(provided).strip():
        raise CloseAIError(
            "CloseAI key file path must be provided via load_closeai_api_key(path) "
            f"or the {CLOSEAI_API_KEY_FILE_ENV} environment variable"
        )
    key_path = Path(provided)
    resolved = key_path.expanduser().resolve(strict=True)
    text = resolved.read_text(encoding="utf-8")
    if not text or text != text.strip() or len(text.splitlines()) != 1:
        raise ValueError("CloseAI key file must contain exactly one non-empty trimmed line")
    key = text.strip()
    if any(character.isspace() for character in key):
        raise ValueError("CloseAI key file must not contain internal whitespace")
    return key


def sanitize_log_value(value: Any, *, secrets: Sequence[str] = ()) -> Any:
    if isinstance(value, str):
        text = redact_secrets(value, secrets)
        if text.startswith(("data:image/", "data:video/")) and ";base64," in text:
            prefix, _, _payload = text.partition(";base64,")
            return f"{prefix};base64,<omitted>"
        if "authorization" in text.lower() and "bearer " in text.lower():
            return "<redacted>"
        return text
    if isinstance(value, Mapping):
        return {
            str(key): sanitize_log_value(item, secrets=secrets)
            for key, item in value.items()
            if str(key).lower() not in {"authorization", "api_key", "api-key"}
        }
    if isinstance(value, list):
        return [sanitize_log_value(item, secrets=secrets) for item in value]
    if isinstance(value, tuple):
        return [sanitize_log_value(item, secrets=secrets) for item in value]
    return value


_KEY_FILE_FLAGS = (
    "--api-key-file",
    "--closeai-key-file",
    "--dashscope-key-file",
)


def sanitize_invocation(argv: Sequence[str]) -> list[str]:
    sanitized: list[str] = []
    redact_next = False
    for value in argv:
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        if value in _KEY_FILE_FLAGS:
            sanitized.append(value)
            redact_next = True
            continue
        redacted_flag = next(
            (flag for flag in _KEY_FILE_FLAGS if value.startswith(f"{flag}=")),
            None,
        )
        if redacted_flag is not None:
            sanitized.append(f"{redacted_flag}=<redacted>")
            continue
        sanitized.append(value)
    return sanitized


@dataclass(frozen=True)
class ModelTokenPrice:
    input_cny_per_mtok: Decimal
    output_cny_per_mtok: Decimal

    def estimate(self, prompt_tokens: int, completion_tokens: int) -> Decimal:
        if type(prompt_tokens) is not int or type(completion_tokens) is not int:
            raise ValueError("token counts must be ints")
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        input_cost = (Decimal(prompt_tokens) / Decimal(1_000_000)) * self.input_cny_per_mtok
        output_cost = (Decimal(completion_tokens) / Decimal(1_000_000)) * self.output_cny_per_mtok
        return cny_amount(input_cost + output_cost)


class CloseAICostCounter:
    """Running RMB spend against an explicit hard cap."""

    def __init__(
        self,
        *,
        hard_cap_cny: Decimal | int | float | str,
        prices: Mapping[str, Mapping[str, Any]],
        precall_prompt_tokens: int,
        precall_completion_tokens: int,
        already_spent_cny: Decimal | int | float | str = 0,
    ) -> None:
        # The cap is the caller's explicit, user-approved spend for that run; family caps such as
        # ``RMB_BUDGET_HARD_CAP`` are enforced by the modules that own them.
        self.hard_cap_cny = cny_amount(hard_cap_cny)
        if self.hard_cap_cny <= 0:
            raise ValueError("hard_cap_cny must be positive")
        if type(precall_prompt_tokens) is not int or type(precall_completion_tokens) is not int:
            raise ValueError("precall token estimates must be ints")
        if precall_prompt_tokens <= 0 or precall_completion_tokens <= 0:
            raise ValueError("precall token estimates must be positive")
        self.precall_prompt_tokens = precall_prompt_tokens
        self.precall_completion_tokens = precall_completion_tokens
        self._prices = {
            _require_model_name(model): ModelTokenPrice(
                input_cny_per_mtok=cny_amount(price["input_cny_per_mtok"]),
                output_cny_per_mtok=cny_amount(price["output_cny_per_mtok"]),
            )
            for model, price in prices.items()
        }
        if not self._prices:
            raise ValueError("prices must contain at least one model")
        self._spent_cny = cny_amount(already_spent_cny)
        if self._spent_cny < 0:
            raise ValueError("already_spent_cny must be non-negative")
        if self._spent_cny > self.hard_cap_cny:
            raise CloseAIBudgetExceeded(
                "CloseAI RMB budget already exceeds "
                f"{display_cny(self.hard_cap_cny)}; "
                f"spent={display_cny(self._spent_cny)}"
            )
        self._calls = 0
        self._by_model: dict[str, Decimal] = {model: cny_amount(0) for model in self._prices}

    def price_for(self, model: str) -> ModelTokenPrice:
        name = _require_model_name(model)
        if name not in self._prices:
            raise ValueError(f"no RMB price schedule for model {name!r}")
        return self._prices[name]

    def estimate_cny(
        self,
        model: str,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> Decimal:
        return self.price_for(model).estimate(
            self.precall_prompt_tokens if prompt_tokens is None else prompt_tokens,
            self.precall_completion_tokens if completion_tokens is None else completion_tokens,
        )

    def remaining_cny(self) -> Decimal:
        remaining = self.hard_cap_cny - self._spent_cny
        return remaining if remaining > 0 else cny_amount(0)

    def assert_can_spend(self, estimated_cny: Decimal, *, model: str) -> None:
        cost = cny_amount(estimated_cny)
        if self._spent_cny + cost > self.hard_cap_cny:
            raise CloseAIBudgetExceeded(
                "CloseAI RMB budget would exceed "
                f"{display_cny(self.hard_cap_cny)} if {model} spent {display_cny(cost)}; "
                f"spent={display_cny(self._spent_cny)} remaining={display_cny(self.remaining_cny())}"
            )

    def reserve_precall(self, model: str) -> Decimal:
        estimated = self.estimate_cny(model)
        self.assert_can_spend(estimated, model=model)
        return estimated

    def record_usage(self, model: str, usage: Mapping[str, Any]) -> Decimal:
        prompt_tokens = _require_token(usage.get("prompt_tokens"), "usage.prompt_tokens")
        completion_tokens = _require_token(
            usage.get("completion_tokens"), "usage.completion_tokens"
        )
        actual = self.estimate_cny(
            model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        name = _require_model_name(model)
        self._spent_cny = cny_amount(self._spent_cny + actual)
        self._calls += 1
        self._by_model[name] = cny_amount(self._by_model.get(name, cny_amount(0)) + actual)
        if self._spent_cny > self.hard_cap_cny:
            raise CloseAIBudgetExceeded(
                "CloseAI RMB budget exceeded after usage accounting: "
                f"spent={display_cny(self._spent_cny)} add={display_cny(actual)} "
                f"cap={display_cny(self.hard_cap_cny)}"
            )
        return actual

    def snapshot(self) -> dict[str, Any]:
        return {
            "currency": "CNY",
            "hard_cap_cny": display_cny(self.hard_cap_cny),
            "spent_cny": display_cny(self._spent_cny),
            "remaining_cny": display_cny(self.remaining_cny()),
            "calls": self._calls,
            "spent_cny_by_model": {
                model: display_cny(amount) for model, amount in sorted(self._by_model.items())
            },
            "near_cap": self.remaining_cny() <= cny_amount("5.00"),
        }


@dataclass(frozen=True)
class CloseAIResponse:
    text: str
    model: str
    usage: dict[str, int]
    cost_cny: str
    request_id: str | None
    #: ``usage.prompt_tokens_details.image_tokens`` as the provider reported it (part of
    #: ``usage["prompt_tokens"]``); ``None`` when the response carries no such field.
    prompt_image_tokens: int | None = None
    #: Provider-reported billed amount when the response carries one. ``None`` when absent.
    #: Not converted; the caller decides the currency.
    provider_billed_cost: str | None = None
    raw_usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    reasoning_characters: int = 0
    returned_model: str | None = None


class CloseAIClient:
    """OpenAI-compatible chat client. The API key is never part of repr or logs."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        cost_counter: CloseAICostCounter,
        timeout_seconds: float,
        max_retries: int,
        retry_base_seconds: float,
        rpm_limit: int,
        transport: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip() or api_key != api_key.strip():
            raise ValueError("api_key must be a non-empty trimmed string")
        if not isinstance(base_url, str) or not base_url.strip() or base_url != base_url.strip():
            raise ValueError("base_url must be a non-empty trimmed string")
        if not base_url.endswith("/v1"):
            raise ValueError("CloseAI OpenAI-compatible base_url must end with /v1")
        if not isinstance(cost_counter, CloseAICostCounter):
            raise TypeError("cost_counter must be a CloseAICostCounter")
        if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool):
            raise ValueError("timeout_seconds must be a positive number")
        if float(timeout_seconds) <= 0:
            raise ValueError("timeout_seconds must be a positive number")
        if type(max_retries) is not int or max_retries < 1:
            raise ValueError("max_retries must be a positive int")
        if not isinstance(retry_base_seconds, (int, float)) or float(retry_base_seconds) <= 0:
            raise ValueError("retry_base_seconds must be positive")
        if type(rpm_limit) is not int or rpm_limit < 1:
            raise ValueError("rpm_limit must be a positive int")
        self._api_key = api_key
        self._base_url = base_url
        self._cost_counter = cost_counter
        self._timeout_seconds = float(timeout_seconds)
        self._max_retries = max_retries
        self._retry_base_seconds = float(retry_base_seconds)
        self._min_interval = 60.0 / float(rpm_limit)
        self._transport = transport
        self._last_request_at = 0.0
        self._sdk_client: Any | None = None
        # One client serves concurrent worker threads: request pacing, the RMB counter and the
        # lazily built SDK client are shared state.
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"CloseAIClient(base_url={self._base_url!r}, api_key='<redacted>')"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def cost_counter(self) -> CloseAICostCounter:
        return self._cost_counter

    def complete_text(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        enable_thinking: bool = False,
    ) -> CloseAIResponse:
        name = _require_chat_args(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
        return self._submit_json_chat(
            model=name,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(temperature),
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )

    def complete_messages(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        temperature: float,
        response_format: Mapping[str, Any] | None,
        max_tokens: int | None,
        enable_thinking: bool = False,
    ) -> CloseAIResponse:
        """Send a caller-built chat (e.g. a system turn plus a user turn) unchanged.

        ``complete_text`` wraps one prompt into a single user turn; callers that own their
        own message template (``story_visual_setting.writer_messages``) pass it here instead.
        A content item is a text part or an inline ``data:image/...;base64,`` image part.
        Streamed for the same reason as ``complete_video_frames``: the proxy returns an empty
        HTTP 200 on long non-streamed multimodal generations.

        ``response_format`` and ``max_tokens`` are required and may be ``None``: ``None`` leaves
        the key out of the request. A thinking model answers without ``response_format`` and a
        structured-output repair call sets ``json_schema`` without ``max_tokens``.
        """

        if not isinstance(messages, Sequence) or isinstance(messages, str) or not messages:
            raise ValueError("messages must be a non-empty list")
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
                raise ValueError(f"messages[{index}] must have exactly role and content")
            if message["role"] not in {"system", "user", "assistant"}:
                raise ValueError(f"messages[{index}].role is unsupported: {message['role']!r}")
            content = message["content"]
            parts = [content] if isinstance(content, str) else content
            if not isinstance(parts, list) or not parts:
                raise ValueError(f"messages[{index}].content must be a string or a list of parts")
            for part in parts:
                if isinstance(part, str) and part.strip():
                    continue
                if not isinstance(part, Mapping):
                    raise ValueError(f"messages[{index}] carries a malformed content part")
                if part.get("type") == "text" and isinstance(part.get("text"), str) and part["text"].strip():
                    continue
                url = part.get("image_url", {}).get("url") if part.get("type") == "image_url" else None
                if isinstance(url, str) and url.startswith("data:image/") and ";base64," in url:
                    continue
                video = part.get("video_url", {}).get("url") if part.get("type") == "video_url" else None
                if isinstance(video, str) and video.startswith("data:video/") and ";base64," in video:
                    if "fps" in part and (not isinstance(part["fps"], (int, float)) or isinstance(part["fps"], bool) or part["fps"] <= 0):
                        raise ValueError("video fps must be positive")
                    if "max_frames" in part and (type(part["max_frames"]) is not int or part["max_frames"] <= 0):
                        raise ValueError("video max_frames must be a positive int")
                    continue
                raise ValueError(
                    f"messages[{index}] carries a content part that is neither non-empty text "
                    "nor an inline data:image/data:video URL"
                )
        name = _require_decode_args(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
        return self._submit_json_chat(
            model=name,
            messages=[dict(message) for message in messages],
            temperature=float(temperature),
            response_format=_require_response_format(response_format),
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            stream=True,
        )

    def complete_video(
        self,
        *,
        model: str,
        video_data_url: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        enable_thinking: bool = False,
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> CloseAIResponse:
        name = _require_chat_args(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
        if (
            not isinstance(video_data_url, str)
            or not video_data_url.startswith("data:video/")
            or ";base64," not in video_data_url
        ):
            raise ValueError("video_data_url must be a data:video/...;base64,... URL")
        video_entry: dict[str, Any] = {"type": "video_url", "video_url": {"url": video_data_url}}
        if fps is not None:
            video_entry["fps"] = float(fps)
        if max_frames is not None:
            video_entry["max_frames"] = int(max_frames)
        content: list[dict[str, Any]] = [
            video_entry,
            {"type": "text", "text": prompt},
        ]
        return self._submit_json_chat(
            model=name,
            messages=[{"role": "user", "content": content}],
            temperature=float(temperature),
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )

    def complete_video_frames(
        self,
        *,
        model: str,
        frame_data_urls: Sequence[str],
        prompt: str,
        temperature: float,
        max_tokens: int,
        enable_thinking: bool = False,
        fps: float | None = None,
    ) -> CloseAIResponse:
        """Send an ordered frame sequence as labelled image parts.

        ``complete_video`` relies on a ``video_url`` part, which OpenAI-compatible
        proxies drop without error when the backend has no video ingestion path.
        Image parts are the format those proxies actually forward, and the
        chronological labels keep multi-shot ordering judgeable.
        """

        name = _require_chat_args(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
        urls = list(frame_data_urls)
        if not urls:
            raise ValueError("frame_data_urls must be non-empty")
        for index, url in enumerate(urls):
            if (
                not isinstance(url, str)
                or not url.startswith("data:image/")
                or ";base64," not in url
            ):
                raise ValueError(
                    f"frame_data_urls[{index}] must be a data:image/...;base64,... URL"
                )
        header = f"VIDEO FRAMES: {len(urls)} JPEG frames in chronological order"
        if fps is not None:
            header = f"{header}, sampled at {float(fps)} FPS"
        content: list[dict[str, Any]] = [{"type": "text", "text": f"{header}."}]
        for index, url in enumerate(urls, start=1):
            content.append({"type": "text", "text": f"frame {index}/{len(urls)}"})
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": prompt})
        return self._submit_json_chat(
            model=name,
            messages=[{"role": "user", "content": content}],
            temperature=float(temperature),
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            # Non-streamed multimodal calls hit the proxy gateway's empty-200
            # response-buffer timeout; streaming keeps the connection alive so
            # long-reasoning verdicts are not silently dropped (survivorship
            # would otherwise bias the sample toward short reasoning).
            stream=True,
        )

    def complete_multi_image(
        self,
        *,
        model: str,
        image_data_urls: Sequence[str],
        prompt: str,
        temperature: float,
        max_tokens: int,
        enable_thinking: bool = False,
    ) -> CloseAIResponse:
        name = _require_chat_args(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
        if not image_data_urls:
            raise ValueError("image_data_urls must be non-empty")
        if any(
            not isinstance(url, str) or not url.startswith("data:image/")
            for url in image_data_urls
        ):
            raise ValueError("image_data_urls must be data:image/... URLs")
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": url}} for url in image_data_urls
        ]
        content.append({"type": "text", "text": prompt})
        return self._submit_json_chat(
            model=name,
            messages=[{"role": "user", "content": content}],
            temperature=float(temperature),
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )

    def _submit_json_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        response_format: Mapping[str, Any] | None,
        max_tokens: int | None,
        enable_thinking: bool,
        stream: bool = False,
    ) -> CloseAIResponse:
        with self._lock:
            estimated = self._cost_counter.reserve_precall(model)
        request: dict[str, Any] = {"model": model, "temperature": temperature}
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if response_format is not None:
            request["response_format"] = dict(response_format)
        request.update(
            messages=messages,
            # Qwen hybrid thinking defaults ON. Classification JSON must not
            # wait out a hidden reasoning trace.
            extra_body={"enable_thinking": enable_thinking},
        )
        raw = self._complete_with_retry(request, stream=stream)
        usage = _usage_from_response(raw)
        text = _text_from_response(raw)
        with self._lock:
            actual = self._cost_counter.record_usage(model, usage)
        del estimated
        return CloseAIResponse(
            text=text,
            model=model,
            returned_model=getattr(raw, "model", None),
            usage=usage,
            cost_cny=display_cny(actual),
            request_id=_request_id(raw),
            prompt_image_tokens=_prompt_image_tokens(raw),
            provider_billed_cost=_provider_billed_cost(raw),
            raw_usage=(dict(raw.usage) if isinstance(raw.usage, Mapping) else raw.usage.model_dump() if hasattr(raw.usage, "model_dump") else vars(raw.usage)),
            finish_reason=raw.choices[0].finish_reason,
            reasoning_characters=len(getattr(raw.choices[0].message, "reasoning_content", None) or ""),
        )

    def request_log(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return sanitize_log_value(request, secrets=(self._api_key,))

    def _complete_with_retry(self, request: dict[str, Any], *, stream: bool = False) -> Any:
        last_error = "request failed"
        for attempt in range(self._max_retries):
            self._respect_rpm()
            try:
                return self._dispatch(request, stream=stream)
            except (CloseAIBudgetExceeded, CloseAIResponseTruncated):
                # Neither is retryable; keep the type so callers can tell a truncated
                # answer from any other request failure.
                raise
            except Exception as exc:
                last_error = redact_secrets(str(exc), (self._api_key,))
                status = getattr(exc, "status_code", None)
                retryable = status in _RETRYABLE_STATUS or _is_retryable_error(exc)
                if not retryable or attempt + 1 >= self._max_retries:
                    raise CloseAIError(last_error) from None
                time.sleep(self._retry_base_seconds * (2**attempt))
        raise CloseAIError(last_error)

    def _dispatch(self, request: dict[str, Any], *, stream: bool = False) -> Any:
        if self._transport is not None:
            # Injected transports return a complete response object; the streaming
            # reassembly path is exercised only against the real SDK.
            return _reject_truncated_response(self._transport(**request))
        client = self._sdk()
        try:
            if stream:
                streamed = client.chat.completions.create(
                    **request,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                return _reject_truncated_response(_reassemble_stream(streamed))
            return _reject_truncated_response(client.chat.completions.create(**request))
        except Exception as exc:
            # Do not reconstruct SDK exceptions: OpenAI APIConnectionError
            # rejects a message argument and would hide the real failure.
            message = redact_secrets(str(exc), (self._api_key,))
            if self._api_key and self._api_key in str(exc):
                raise CloseAIError(message) from None
            raise

    def _sdk(self) -> Any:
        with self._lock:
            return self._sdk_locked()

    def _sdk_locked(self) -> Any:
        if self._sdk_client is not None:
            return self._sdk_client
        try:
            import httpx
            from openai import OpenAI
        except ImportError as exc:
            raise CloseAIError("CloseAI client requires the 'api' extra (openai)") from exc
        self._sdk_client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=httpx.Timeout(
                connect=15.0,
                read=self._timeout_seconds,
                write=60.0,
                pool=15.0,
            ),
            max_retries=0,
            http_client=httpx.Client(trust_env=False),
        )
        return self._sdk_client

    def _respect_rpm(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._last_request_at + self._min_interval)
            self._last_request_at = slot
        if slot > now:
            time.sleep(slot - now)


def _require_chat_args(
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
) -> str:
    if not isinstance(prompt, str) or not prompt.strip() or prompt != prompt.strip():
        raise ValueError("prompt must be a non-empty trimmed string")
    return _require_decode_args(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )


def _require_decode_args(
    *,
    model: str,
    temperature: float,
    max_tokens: int | None,
    enable_thinking: bool,
) -> str:
    name = _require_model_name(model)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise ValueError("temperature must be a number")
    if not 0 <= float(temperature) <= 1:
        raise ValueError("temperature must be in [0, 1]")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
        raise ValueError("max_tokens must be None or a positive int")
    if type(enable_thinking) is not bool:
        raise ValueError("enable_thinking must be a bool")
    return name


def _require_response_format(value: Any) -> dict[str, Any] | None:
    """``None`` (key left out), ``{"type": "json_object"}`` or a ``json_schema`` object."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("response_format must be None or an object")
    kind = value.get("type")
    if kind == "json_object" and set(value) == {"type"}:
        return dict(value)
    if kind == "json_schema" and set(value) == {"type", "json_schema"}:
        spec = value["json_schema"]
        if (isinstance(spec, Mapping) and isinstance(spec.get("name"), str) and spec["name"].strip()
                and isinstance(spec.get("schema"), Mapping) and set(spec) <= {"name", "strict", "schema", "description"}):
            return {"type": "json_schema", "json_schema": dict(spec)}
    raise ValueError(
        "response_format must be None, {'type': 'json_object'} or "
        "{'type': 'json_schema', 'json_schema': {'name', 'schema', ...}}"
    )


def _require_model_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("model must be a non-empty trimmed string")
    if value.endswith("-image"):
        raise ValueError(
            f"model {value!r} looks like image generation, not vision-in; refuse"
        )
    return value


def _require_token(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative int")
    return value


def _usage_from_response(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, Mapping):
        usage = response.get("usage")
    if isinstance(usage, Mapping):
        payload = usage
    else:
        payload = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    prompt = _require_token(payload.get("prompt_tokens"), "usage.prompt_tokens")
    completion = _require_token(payload.get("completion_tokens"), "usage.completion_tokens")
    total = payload.get("total_tokens")
    if total is None:
        total = prompt + completion
    if type(total) is not int or total < max(prompt, completion):
        raise ValueError("usage.total_tokens is inconsistent")
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _provider_billed_cost(response: Any) -> str | None:
    """Return a provider cost string when the payload carries one, else None."""

    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, Mapping):
        usage = response.get("usage")
    found: list[Any] = []

    def take(container: Any, key: str) -> None:
        if isinstance(container, Mapping):
            if key in container and container[key] is not None:
                found.append(container[key])
            return
        value = getattr(container, key, None)
        if value is not None:
            found.append(value)
        extra = getattr(container, "model_extra", None)
        if isinstance(extra, Mapping) and extra.get(key) is not None:
            found.append(extra[key])

    if usage is not None:
        for key in ("cost", "total_cost", "billed_cost"):
            take(usage, key)
    for key in ("cost", "total_cost", "billed_cost"):
        take(response, key)
    if not found or isinstance(found[0], bool):
        return None
    try:
        amount = Decimal(str(found[0]))
    except Exception:
        return None
    if not amount.is_finite() or amount < 0:
        return None
    return format(amount, "f")


def _prompt_image_tokens(response: Any) -> int | None:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, Mapping):
        usage = response.get("usage")
    details = usage.get("prompt_tokens_details") if isinstance(usage, Mapping) else getattr(
        usage, "prompt_tokens_details", None
    )
    value = details.get("image_tokens") if isinstance(details, Mapping) else getattr(
        details, "image_tokens", None
    )
    if value is None:
        return None
    return _require_token(value, "usage.prompt_tokens_details.image_tokens")


def _text_from_response(response: Any) -> str:
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
            content = message.get("content") if isinstance(message, Mapping) else None
            if isinstance(content, str) and content.strip():
                return content
        text = response.get("text")
        if isinstance(text, str) and text.strip():
            return text
        raise CloseAIError("CloseAI response is missing text")
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    try:
        content = response.choices[0].message.content
    except Exception as exc:
        raise CloseAIError("CloseAI response is missing text") from exc
    if isinstance(content, str) and content.strip():
        return content
    raise CloseAIError("CloseAI response is missing text")


def _request_id(response: Any) -> str | None:
    if isinstance(response, Mapping):
        value = response.get("id")
        return value if isinstance(value, str) and value.strip() else None
    value = getattr(response, "id", None)
    return value if isinstance(value, str) and value.strip() else None


def _reassemble_stream(stream: Any) -> Any:
    """Collect SSE chunks into a non-streamed response shape.

    Returns an object matching ``response.choices[0].message.content``,
    ``.finish_reason``, ``.usage`` and ``.id`` so the existing extractors work
    unchanged. Raises :class:`CloseAIEmptyStream` (retryable) if no assistant
    content arrived, so a stalled stream is retried instead of scored empty.
    """

    pieces: list[str] = []
    finish_reason: str | None = None
    usage: Any = None
    request_id: str | None = None
    reasoning: list[str] = []
    returned_model: str | None = None
    for chunk in stream:
        chunk_model = getattr(chunk, "model", None)
        if isinstance(chunk_model, str) and chunk_model.strip():
            returned_model = chunk_model
        chunk_id = getattr(chunk, "id", None)
        if isinstance(chunk_id, str) and chunk_id.strip():
            request_id = chunk_id
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        piece = getattr(delta, "content", None) if delta is not None else None
        if isinstance(piece, str) and piece:
            pieces.append(piece)
        thought = getattr(delta, "reasoning_content", None) if delta is not None else None
        if isinstance(thought, str):
            reasoning.append(thought)
        chunk_finish = getattr(choices[0], "finish_reason", None)
        if chunk_finish:
            finish_reason = chunk_finish
    text = "".join(pieces)
    if not text.strip():
        raise CloseAIEmptyStream(
            "CloseAI streamed completion produced no assistant content"
        )
    message = SimpleNamespace(content=text, reasoning_content="".join(reasoning))
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage, id=request_id, model=returned_model)


def _reject_truncated_response(response: Any) -> Any:
    """Raise when the response stopped at the token ceiling instead of finishing."""

    try:
        finish_reason = response.choices[0].finish_reason
    except Exception as exc:
        raise CloseAIError(
            "CloseAI response is missing finish_reason; cannot verify the JSON was not truncated"
        ) from exc
    if not isinstance(finish_reason, str) or not finish_reason.strip():
        raise CloseAIError(
            "CloseAI response is missing finish_reason; cannot verify the JSON was not truncated"
        )
    if finish_reason == "length":
        raise CloseAIResponseTruncated(
            "CloseAI response hit max_tokens before finishing; raise max_tokens so "
            "hidden reasoning cannot truncate the JSON answer"
        )
    return response


def _is_retryable_error(exc: Exception) -> bool:
    name = type(exc).__name__
    return name in {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        # The proxy intermittently answers large multimodal requests with HTTP 200
        # and an empty body, which the SDK surfaces as a decode failure. An empty
        # body is never a valid completion, so retry instead of losing the sample.
        "JSONDecodeError",
        "APIResponseValidationError",
        # A stalled or empty stream is a transient route failure, not a verdict.
        "CloseAIEmptyStream",
    }
