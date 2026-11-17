"""OpenRouter OpenAI-compatible client with a USD hard cap independent of CloseAI."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Sequence, Callable, Mapping

import os
import time

from mspc.config.shared_root import expand_shared_root_token
from mspc.data.strict_json import load_strict_json_file


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_FILE_ENV = "OPENROUTER_API_KEY_FILE"
OPENROUTER_API_KEY_FILENAME = "openrouter_api_key.txt"
OPENROUTER_KEY_FILE_RELATIVE = f"OoVMetric/resource/{OPENROUTER_API_KEY_FILENAME}"
USD_BUDGET_HARD_CAP = Decimal("70.00")
PRICING_SCHEMA_VERSION = "mspc_openrouter_pricing_v1"
PRICING_CONFIG_RELATIVE = "configs/runtime/openrouter_pricing_v1.json"
_USD_QUANT = Decimal("0.0001")
_REQUIRED_PRICING_KEYS = (
    "schema_version",
    "platform",
    "currency",
    "hard_cap_usd",
    "api_key_file",
    "base_url",
    "model",
    "input_usd_per_mtok",
    "output_usd_per_mtok",
)


class OpenRouterError(RuntimeError):
    """Raised for OpenRouter request or credential failures. Messages are redacted."""

    def __init__(self, message: str, *, exception_chain: Sequence[Mapping[str, str]] = (),
                 http_request_id: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.exception_chain = [dict(item) for item in exception_chain]
        self.http_request_id = http_request_id
        self.status_code = status_code


class OpenRouterBudgetExceeded(OpenRouterError):
    """Raised when the USD hard cap blocks a call or received usage exceeds it."""


class OpenRouterResponseError(OpenRouterError):
    """The server returned a response; retain it and never retry as a transport failure."""

    def __init__(self, message: str, *, stage: str, raw_response: Any,
                 usage: Mapping[str, Any], cost_usd: str | None, text: str | None):
        super().__init__(message)
        self.stage = stage
        self.raw_response = _response_json(raw_response)
        self.usage = dict(usage)
        self.cost_usd = cost_usd
        self.text = text
        self.served = _served_from_response(raw_response)


def usd_amount(value: Decimal | int | float | str) -> Decimal:
    if type(value) is bool:
        raise ValueError("USD amount must not be a bool")
    return Decimal(str(value)).quantize(_USD_QUANT, rounding=ROUND_HALF_UP)


def display_usd(value: Decimal) -> str:
    return str(value.quantize(_USD_QUANT, rounding=ROUND_HALF_UP))


def _trimmed(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _require_positive_usd(value: Any, label: str) -> Decimal:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{label} is required; no default unit price")
    amount = usd_amount(value)
    if amount <= 0:
        raise ValueError(f"{label} must be a positive USD amount")
    return amount


def _redact(text: str, secret: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if secret:
        return text.replace(secret, "<redacted>")
    return text


def _transport_error_details(error: BaseException, secret: str) -> dict[str, Any]:
    """Keep SDK diagnostics without retaining requests, credentials, or raw exceptions."""
    chain = []
    http_request_id = None
    status_code = None
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append({"type": f"{type(error).__module__}.{type(error).__name__}",
                      "message": _redact(str(error), secret)})
        request_id = getattr(error, "request_id", None)
        if http_request_id is None and isinstance(request_id, str) and request_id:
            http_request_id = _redact(request_id, secret)
        code = getattr(error, "status_code", None)
        if status_code is None and type(code) is int:
            status_code = code
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return {"exception_chain": chain, "http_request_id": http_request_id,
            "status_code": status_code}


def _looks_like_batch_entry(value: str) -> bool:
    lowered = value.lower()
    if ":batch" in lowered:
        return True
    if lowered == "batch":
        return True
    parts = [part for part in lowered.replace("\\", "/").split("/") if part]
    return "batch" in parts


def reject_openrouter_batch_entry_point(
    *,
    base_url: str,
    model: str,
    endpoint: Any = None,
) -> None:
    """Refuse OpenRouter :batch. Gemini 3.7 Flash does not accept video there."""

    checks = (("base_url", base_url), ("model", model))
    if endpoint is not None:
        checks = (*checks, ("endpoint", endpoint))
    for label, raw in checks:
        text = _trimmed(raw, label)
        if _looks_like_batch_entry(text):
            raise ValueError(
                f"{label} {text!r} is an OpenRouter :batch entry point; "
                "OpenRouter Gemini 3.7 Flash does not accept video input on :batch; "
                "use the standard chat completions endpoint"
            )


def load_openrouter_api_key(path: Path | str | None = None) -> str:
    """Read the OpenRouter key file. The return value must stay in memory only."""

    provided = path if path is not None else os.environ.get(OPENROUTER_API_KEY_FILE_ENV)
    if provided is None or not str(provided).strip():
        raise OpenRouterError(
            "OpenRouter key file path must be provided via load_openrouter_api_key(path) "
            f"or the {OPENROUTER_API_KEY_FILE_ENV} environment variable"
        )
    key_path = Path(provided)
    resolved = key_path.expanduser().resolve(strict=True)
    text = resolved.read_text(encoding="utf-8")
    if not text or text != text.strip() or len(text.splitlines()) != 1:
        raise ValueError("OpenRouter key file must contain exactly one non-empty trimmed line")
    key = text.strip()
    if any(character.isspace() for character in key):
        raise ValueError("OpenRouter key file must not contain internal whitespace")
    return key


def load_openrouter_pricing_config(
    path: Path | str,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    source = Path(path)
    payload = load_strict_json_file(source, label="openrouter pricing")
    if not isinstance(payload, dict):
        raise ValueError("openrouter pricing must be an object")
    for key in _REQUIRED_PRICING_KEYS:
        if key not in payload:
            raise ValueError(f"openrouter pricing missing required {key}")
    if payload["schema_version"] != PRICING_SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {PRICING_SCHEMA_VERSION}")
    if payload["platform"] != "openrouter":
        raise ValueError("platform must equal openrouter")
    if payload["currency"] != "USD":
        raise ValueError("currency must equal USD")
    cap = usd_amount(payload["hard_cap_usd"])
    if cap != usd_amount(USD_BUDGET_HARD_CAP):
        raise ValueError(f"hard_cap_usd must equal the hard cap {display_usd(USD_BUDGET_HARD_CAP)}")
    base_url = _trimmed(payload["base_url"], "base_url")
    if not base_url.endswith("/v1"):
        raise ValueError("base_url must end with /v1")
    model = _trimmed(payload["model"], "model")
    endpoint = payload.get("endpoint")
    reject_openrouter_batch_entry_point(base_url=base_url, model=model, endpoint=endpoint)
    input_price = _require_positive_usd(payload["input_usd_per_mtok"], "input_usd_per_mtok")
    output_price = _require_positive_usd(payload["output_usd_per_mtok"], "output_usd_per_mtok")
    raw_key_file = _trimmed(payload["api_key_file"], "api_key_file")
    key_file = expand_shared_root_token(raw_key_file, "api_key_file", environ=environ)
    return {
        "schema_version": PRICING_SCHEMA_VERSION,
        "platform": "openrouter",
        "currency": "USD",
        "hard_cap_usd": display_usd(cap),
        "api_key_file": key_file,
        "base_url": base_url,
        "model": model,
        "input_usd_per_mtok": display_usd(input_price),
        "output_usd_per_mtok": display_usd(output_price),
    }


@dataclass(frozen=True)
class OpenRouterTokenPrice:
    input_usd_per_mtok: Decimal
    output_usd_per_mtok: Decimal

    def estimate(self, prompt_tokens: int, completion_tokens: int) -> Decimal:
        if type(prompt_tokens) is not int or type(completion_tokens) is not int:
            raise ValueError("token counts must be ints")
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        input_cost = (Decimal(prompt_tokens) / Decimal(1_000_000)) * self.input_usd_per_mtok
        output_cost = (Decimal(completion_tokens) / Decimal(1_000_000)) * self.output_usd_per_mtok
        return usd_amount(input_cost + output_cost)


class OpenRouterCostCounter:
    """Running USD spend against the OpenRouter-only $70 hard cap."""

    def __init__(
        self,
        *,
        hard_cap_usd: Decimal | int | float | str,
        prices: Mapping[str, Mapping[str, Any]],
        precall_prompt_tokens: int,
        precall_completion_tokens: int,
        already_spent_usd: Decimal | int | float | str = 0,
    ) -> None:
        self.hard_cap_usd = usd_amount(hard_cap_usd)
        if self.hard_cap_usd <= 0:
            raise ValueError("hard_cap_usd must be positive")
        # 账户硬顶是上限，一次运行可以声明更小的顶，但不能高于账户硬顶。
        if self.hard_cap_usd > usd_amount(USD_BUDGET_HARD_CAP):
            raise ValueError(
                f"hard_cap_usd must not exceed the account hard cap "
                f"{display_usd(USD_BUDGET_HARD_CAP)}"
            )
        if type(precall_prompt_tokens) is not int or type(precall_completion_tokens) is not int:
            raise ValueError("precall token estimates must be ints")
        if precall_prompt_tokens <= 0 or precall_completion_tokens <= 0:
            raise ValueError("precall token estimates must be positive")
        self.precall_prompt_tokens = precall_prompt_tokens
        self.precall_completion_tokens = precall_completion_tokens
        self._prices = {
            _require_model_name(model): OpenRouterTokenPrice(
                input_usd_per_mtok=_require_positive_usd(
                    price["input_usd_per_mtok"], "input_usd_per_mtok"
                ),
                output_usd_per_mtok=_require_positive_usd(
                    price["output_usd_per_mtok"], "output_usd_per_mtok"
                ),
            )
            for model, price in prices.items()
        }
        if not self._prices:
            raise ValueError("prices must contain at least one model")
        self._spent_usd = usd_amount(already_spent_usd)
        if self._spent_usd < 0:
            raise ValueError("already_spent_usd must be non-negative")
        if self._spent_usd > self.hard_cap_usd:
            raise OpenRouterBudgetExceeded(
                "OpenRouter USD budget already exceeds "
                f"{display_usd(self.hard_cap_usd)}; "
                f"spent={display_usd(self._spent_usd)}"
            )
        self._calls = 0
        self._by_model: dict[str, Decimal] = {model: usd_amount(0) for model in self._prices}

    def price_for(self, model: str) -> OpenRouterTokenPrice:
        name = _require_model_name(model)
        if name not in self._prices:
            raise ValueError(f"no USD price schedule for model {name!r}")
        return self._prices[name]

    def estimate_usd(
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

    def remaining_usd(self) -> Decimal:
        remaining = self.hard_cap_usd - self._spent_usd
        return remaining if remaining > 0 else usd_amount(0)

    def assert_can_spend(self, estimated_usd: Decimal, *, model: str) -> None:
        cost = usd_amount(estimated_usd)
        if self._spent_usd + cost > self.hard_cap_usd:
            raise OpenRouterBudgetExceeded(
                "OpenRouter USD budget would exceed "
                f"{display_usd(self.hard_cap_usd)} if {model} spent {display_usd(cost)}; "
                f"spent={display_usd(self._spent_usd)} remaining={display_usd(self.remaining_usd())}"
            )

    def reserve_precall(self, model: str) -> Decimal:
        estimated = self.estimate_usd(model)
        self.assert_can_spend(estimated, model=model)
        return estimated

    def record_usage(self, model: str, usage: Mapping[str, Any]) -> Decimal:
        prompt_tokens = _require_token(usage.get("prompt_tokens"), "usage.prompt_tokens")
        completion_tokens = _require_token(
            usage.get("completion_tokens"), "usage.completion_tokens"
        )
        actual = self.estimate_usd(
            model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        name = _require_model_name(model)
        self._spent_usd = usd_amount(self._spent_usd + actual)
        self._calls += 1
        self._by_model[name] = usd_amount(self._by_model.get(name, usd_amount(0)) + actual)
        # This usage has already been incurred, even when it exceeds the estimate.
        if self._spent_usd > self.hard_cap_usd:
            raise OpenRouterBudgetExceeded(
                "OpenRouter USD budget exceeds "
                f"{display_usd(self.hard_cap_usd)} after usage accounting: "
                f"spent={display_usd(self._spent_usd)} last_call={display_usd(actual)} "
                f"remaining={display_usd(self.remaining_usd())}"
            )
        return actual

    def snapshot(self) -> dict[str, Any]:
        return {
            "currency": "USD",
            "hard_cap_usd": display_usd(self.hard_cap_usd),
            "spent_usd": display_usd(self._spent_usd),
            "remaining_usd": display_usd(self.remaining_usd()),
            "calls": self._calls,
            "spent_usd_by_model": {
                model: display_usd(amount) for model, amount in sorted(self._by_model.items())
            },
        }


#: 判官请求体里除消息外、由判官配置显式给出的 OpenRouter 字段。只收这三个键：
#: ``reasoning``（gemini-3.x 不能关思考，只能给 effort）、``service_tier``（``flex`` 半价档，
#: 不回落到默认档）、``provider``（把路由钉在一个端点上：不同上游是不同量具）。
REQUEST_OPTION_KEYS = ("reasoning", "service_tier", "provider")


def require_request_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """校验判官配置给的请求选项。键只能是 :data:`REQUEST_OPTION_KEYS` 的子集，值原样进请求体。

    空映射是合法的显式选择（2.5 系判官不传这三项，历史读数就是这么跑的）；``None`` 不是。
    """

    if not isinstance(options, Mapping):
        raise ValueError("request_options must be a mapping (use {} to send none)")
    unknown = sorted(set(options) - set(REQUEST_OPTION_KEYS))
    if unknown:
        raise ValueError(f"request_options has unknown keys {unknown}; allowed {REQUEST_OPTION_KEYS}")
    if "reasoning" in options:
        reasoning = options["reasoning"]
        if not isinstance(reasoning, Mapping) or set(reasoning) != {"effort"}:
            raise ValueError("request_options.reasoning must be exactly {'effort': <level>}")
        _trimmed(reasoning["effort"], "request_options.reasoning.effort")
    if "service_tier" in options:
        _trimmed(options["service_tier"], "request_options.service_tier")
    if "provider" in options:
        provider = options["provider"]
        if not isinstance(provider, Mapping) or not provider:
            raise ValueError("request_options.provider must be a non-empty mapping")
    return {key: options[key] for key in REQUEST_OPTION_KEYS if key in options}


def require_response_format(response_format: Mapping[str, Any]) -> dict[str, Any]:
    """校验调用方显式给的 ``response_format``：``{"type": "json_object"}`` 或 json_schema strict。

    客户端不替调用方选格式：旧判官显式传 ``json_object``，J2 判官传按断言编号生成的 strict schema。
    """

    if not isinstance(response_format, Mapping):
        raise ValueError("response_format must be a mapping")
    kind = response_format.get("type")
    if kind == "json_object":
        if set(response_format) != {"type"}:
            raise ValueError("response_format json_object takes no other keys")
    elif kind == "json_schema":
        spec = response_format.get("json_schema")
        if set(response_format) != {"type", "json_schema"} or not isinstance(spec, Mapping):
            raise ValueError("response_format json_schema must be exactly {type, json_schema}")
        if spec.get("strict") is not True or not isinstance(spec.get("schema"), Mapping):
            raise ValueError("response_format.json_schema must set strict=true and carry a schema mapping")
        _trimmed(spec.get("name"), "response_format.json_schema.name")
    else:
        raise ValueError(f"response_format.type must be json_object or json_schema, got {kind!r}")
    return dict(response_format)


@dataclass(frozen=True)
class OpenRouterResponse:
    text: str
    model: str
    usage: dict[str, int]
    cost_usd: str
    request_id: str | None
    #: OpenRouter 回报的实际落点与计费：``provider``（哪个上游）、``service_tier``（哪一档）、
    #: ``model``（响应体回报的模型名，与请求里的 ``model`` 并列）、``finish_reason``（``length`` 即被 ``max_tokens``
    #: 截断：思考 token 也占这个上限）、``reported_cost_usd``（OpenRouter 自己算的钱，与按配置单价算的 ``cost_usd``
    #: 并列）、``reasoning_tokens``（含在 completion 里的思考 token）、``prompt_tokens_details``。
    #: 响应里没有的键记 ``None``：这是回报，不是计价输入。
    served: dict[str, Any]


class OpenRouterClient:
    """OpenAI-compatible chat client on the standard OpenRouter endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        cost_counter: OpenRouterCostCounter,
        timeout_seconds: float,
        proxy: str | None,
        transport: Callable[..., Any] | None = None,
        recorder: Callable[[dict[str, Any], Any, str | None, float], None] | None = None,
    ) -> None:
        """``recorder(request, raw, error, elapsed_seconds)``：每次实际发出的请求（SDK 收到的参数）与原始响应对象；
        请求层失败时 ``raw`` 为 None、``error`` 为已脱敏的报错。只在调用方要落盘时传，落盘失败不吞。"""
        if not isinstance(api_key, str) or not api_key.strip() or api_key != api_key.strip():
            raise ValueError("api_key must be a non-empty trimmed string")
        # 代理必须显式传：openai SDK 会给自己的 base_url 挂一个独立 transport，环境变量
        # ``ALL_PROXY`` 对它不生效（2026-09-16 实测：裸 httpx 走隧道、SDK 的请求直连出去被
        # 地区闸挡）。``None`` 表示直连，不是「按环境猜」。
        if proxy is not None:
            proxy = _trimmed(proxy, "proxy")
        self._proxy = proxy
        self._base_url = _trimmed(base_url, "base_url")
        reject_openrouter_batch_entry_point(base_url=self._base_url, model="standard")
        if not self._base_url.endswith("/v1"):
            raise ValueError("OpenRouter OpenAI-compatible base_url must end with /v1")
        if not isinstance(cost_counter, OpenRouterCostCounter):
            raise TypeError("cost_counter must be an OpenRouterCostCounter")
        if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool):
            raise ValueError("timeout_seconds must be a positive number")
        if float(timeout_seconds) <= 0:
            raise ValueError("timeout_seconds must be a positive number")
        self._api_key = api_key
        self._cost_counter = cost_counter
        self._timeout_seconds = float(timeout_seconds)
        self._transport = transport
        self._recorder = recorder
        self._sdk_client: Any | None = None

    def __repr__(self) -> str:
        return f"OpenRouterClient(base_url={self._base_url!r}, api_key='<redacted>')"

    def __str__(self) -> str:
        return self.__repr__()

    def close(self) -> None:
        client = self._sdk_client
        self._sdk_client = None
        if client is not None:
            client.close()

    @property
    def cost_counter(self) -> OpenRouterCostCounter:
        return self._cost_counter

    def complete_text(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        response_format: Mapping[str, Any],
    ) -> OpenRouterResponse:
        name = _require_chat_args(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        reject_openrouter_batch_entry_point(base_url=self._base_url, model=name)
        return self._submit_json_chat(
            model=name,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(temperature),
            max_tokens=max_tokens,
            request_options={},
            response_format=response_format,
        )

    def complete_messages(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        temperature: float,
        max_tokens: int,
        request_options: Mapping[str, Any],
        response_format: Mapping[str, Any],
    ) -> OpenRouterResponse:
        """按给定的 system/user 消息调用。规划器反写与评测反写共用这一入口。"""

        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise ValueError("messages must be a sequence of role/content objects")
        payload: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
                raise ValueError(f"messages[{index}] must contain only role and content")
            role = message["role"]
            content = message["content"]
            if role not in {"system", "user"} or not isinstance(content, str) or not content.strip():
                raise ValueError(f"messages[{index}] must be a non-empty system or user string")
            payload.append({"role": role, "content": content})
        if not payload or payload[-1]["role"] != "user":
            raise ValueError("messages must end with a user message")
        name = _require_chat_args(
            model=model,
            prompt=payload[-1]["content"],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        reject_openrouter_batch_entry_point(base_url=self._base_url, model=name)
        return self._submit_json_chat(
            model=name,
            messages=payload,
            temperature=float(temperature),
            max_tokens=max_tokens,
            request_options=request_options,
            response_format=response_format,
        )

    def complete_video(
        self,
        *,
        model: str,
        video_data_url: str,
        prompt: str,
        temperature: float | None,
        max_tokens: int,
        request_options: Mapping[str, Any],
        response_format: Mapping[str, Any],
    ) -> OpenRouterResponse:
        """整段成片作为一个 ``video_url`` 块（base64 data URL）发出，后接提示词。

        OpenRouter 文档：Gemini 在 Vertex 上不收视频 URL、在 AI Studio 上只收 YouTube 链接，
        本地成片只能走 data URL。显式使用 ``processing=static``；不把文件帧率当成模型采样帧率的保证。
        ``temperature=None`` 表示不发送采样参数（Gemini 3.8 的原生要求）。
        """

        name = _require_chat_args(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        reject_openrouter_batch_entry_point(base_url=self._base_url, model=name)
        if (
            not isinstance(video_data_url, str)
            or not video_data_url.startswith("data:video/")
            or ";base64," not in video_data_url
        ):
            raise ValueError("video_data_url must be a data:video/...;base64,... URL")
        content: list[dict[str, Any]] = [
            {"type": "video_url", "video_url": {"url": video_data_url, "processing": "static"}},
            {"type": "text", "text": prompt},
        ]
        return self._submit_json_chat(
            model=name,
            messages=[{"role": "user", "content": content}],
            temperature=None if temperature is None else float(temperature),
            max_tokens=max_tokens,
            request_options=request_options,
            response_format=response_format,
        )

    def complete_video_frames(
        self,
        *,
        model: str,
        frame_data_urls: Sequence[str],
        prompt: str,
        temperature: float,
        max_tokens: int,
        request_options: Mapping[str, Any],
        response_format: Mapping[str, Any],
        fps: float | None = None,
    ) -> OpenRouterResponse:
        """按时序把整组帧作为带序号的图像块发出去，消息形状与
        :meth:`mspc.data.closeai_client.CloseAIClient.complete_video_frames` 逐字段相同。

        为什么不共用 ``complete_video``：OpenAI 兼容代理会静默丢掉 ``video_url`` 块，而图像块
        是它们真正转发的格式；判官的多镜次序判断靠的正是每帧前面那行 ``frame i/N`` 标签。
        两个 provider 必须发同一形状的消息，否则两边的判官读数不可比。
        """

        name = _require_chat_args(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        reject_openrouter_batch_entry_point(base_url=self._base_url, model=name)
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
            max_tokens=max_tokens,
            request_options=request_options,
            response_format=response_format,
        )

    def complete_images(
        self,
        *,
        model: str,
        image_data_urls: Sequence[str],
        prompt: str,
        temperature: float,
        max_tokens: int,
        request_options: Mapping[str, Any],
        response_format: Mapping[str, Any],
    ) -> OpenRouterResponse:
        """一组互相独立的图（不是视频帧）按编号发出：每张前一行 ``image i/N``，最后是提示词。

        与 :meth:`complete_video_frames` 的区别只在开头一行：这里不声称时序，裁剪、对照图之类的
        独立样本走这一条，免得判官把它们当成同一段视频去推理。
        """

        name = _require_chat_args(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        reject_openrouter_batch_entry_point(base_url=self._base_url, model=name)
        urls = list(image_data_urls)
        if not urls:
            raise ValueError("image_data_urls must be non-empty")
        for index, url in enumerate(urls):
            if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
                raise ValueError(f"image_data_urls[{index}] must be a data:image/...;base64,... URL")
        content: list[dict[str, Any]] = [
            {"type": "text", "text": f"IMAGES: {len(urls)} independent JPEG images (not a video)."}
        ]
        for index, url in enumerate(urls, start=1):
            content.append({"type": "text", "text": f"image {index}/{len(urls)}"})
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": prompt})
        return self._submit_json_chat(
            model=name,
            messages=[{"role": "user", "content": content}],
            temperature=float(temperature),
            max_tokens=max_tokens,
            request_options=request_options,
            response_format=response_format,
        )

    def _submit_json_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int,
        request_options: Mapping[str, Any],
        response_format: Mapping[str, Any],
    ) -> OpenRouterResponse:
        options = require_request_options(request_options)
        response_format = require_response_format(response_format)
        self._cost_counter.reserve_precall(model)
        request = {
            "model": model,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "messages": messages,
            **options,
        }
        if temperature is not None:
            request["temperature"] = temperature
        raw = self._dispatch(request)
        try:
            usage = _usage_from_response(raw)
        except (ValueError, TypeError) as exc:
            raw_usage = _response_json(_field(raw, "usage"))
            raise OpenRouterResponseError(str(exc), stage="usage", raw_response=raw,
                usage=raw_usage if isinstance(raw_usage, Mapping) else {}, cost_usd=None,
                text=_response_text(raw)) from exc
        actual = self._cost_counter.estimate_usd(model,
            prompt_tokens=usage["prompt_tokens"], completion_tokens=usage["completion_tokens"])
        try:
            self._cost_counter.record_usage(model, usage)
        except OpenRouterBudgetExceeded as exc:
            raise OpenRouterResponseError(str(exc), stage="budget", raw_response=raw,
                usage=usage, cost_usd=display_usd(actual), text=_response_text(raw)) from exc
        try:
            text = _text_from_response(raw)
        except OpenRouterError as exc:
            raise OpenRouterResponseError(str(exc), stage="text", raw_response=raw,
                usage=usage, cost_usd=display_usd(actual), text=_response_text(raw)) from exc
        return OpenRouterResponse(
            text=text,
            model=model,
            usage=usage,
            cost_usd=display_usd(actual),
            request_id=_request_id(raw),
            served=_served_from_response(raw),
        )

    def _dispatch(self, request: dict[str, Any]) -> Any:
        if self._recorder is None:
            return self._send(request)
        started = time.monotonic()
        try:
            raw = self._send(request)
        except Exception as exc:
            self._recorder(request, None, _redact(str(exc), self._api_key), time.monotonic() - started)
            raise
        self._recorder(request, raw, None, time.monotonic() - started)
        return raw

    def _send(self, request: dict[str, Any]) -> Any:
        if self._transport is not None:
            return self._transport(**request)
        client = self._sdk()
        # OpenRouter 自有字段不在 openai SDK 的参数表里，经 extra_body 合进请求体。
        standard = {key: value for key, value in request.items() if key not in REQUEST_OPTION_KEYS}
        extra = {key: value for key, value in request.items() if key in REQUEST_OPTION_KEYS}
        try:
            return client.chat.completions.create(**standard, extra_body=extra or None)
        except Exception as exc:
            message = _redact(str(exc), self._api_key)
            raise OpenRouterError(message, **_transport_error_details(exc, self._api_key)) from None

    def _sdk(self) -> Any:
        if self._sdk_client is not None:
            return self._sdk_client
        try:
            import httpx
            from openai import OpenAI
        except ImportError as exc:
            raise OpenRouterError("OpenRouter client requires the 'api' extra (openai)") from exc
        timeout = httpx.Timeout(
            connect=15.0,
            read=self._timeout_seconds,
            write=self._timeout_seconds,
            pool=15.0,
        )
        http_client = (
            None if self._proxy is None else httpx.Client(proxy=self._proxy, timeout=timeout)
        )
        self._sdk_client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            http_client=http_client,
            timeout=timeout,
            max_retries=0,
        )
        return self._sdk_client


def build_openrouter_cost_counter(
    *,
    pricing: Mapping[str, Any],
    precall_prompt_tokens: int,
    precall_completion_tokens: int,
    already_spent_usd: Decimal | int | float | str = 0,
) -> OpenRouterCostCounter:
    model = _trimmed(pricing.get("model"), "model")
    if "hard_cap_usd" not in pricing:
        raise ValueError("hard_cap_usd is required")
    if "input_usd_per_mtok" not in pricing:
        raise ValueError("input_usd_per_mtok is required; no default unit price")
    if "output_usd_per_mtok" not in pricing:
        raise ValueError("output_usd_per_mtok is required; no default unit price")
    return OpenRouterCostCounter(
        hard_cap_usd=pricing["hard_cap_usd"],
        prices={
            model: {
                "input_usd_per_mtok": pricing["input_usd_per_mtok"],
                "output_usd_per_mtok": pricing["output_usd_per_mtok"],
            }
        },
        precall_prompt_tokens=precall_prompt_tokens,
        precall_completion_tokens=precall_completion_tokens,
        already_spent_usd=already_spent_usd,
    )


def build_openrouter_client(
    *,
    api_key: str,
    pricing: Mapping[str, Any],
    precall_prompt_tokens: int,
    precall_completion_tokens: int,
    timeout_seconds: float,
    proxy: str | None = None,
    transport: Callable[..., Any] | None = None,
    recorder: Callable[[dict[str, Any], Any, str | None, float], None] | None = None,
    already_spent_usd: Decimal | int | float | str = 0,
    cost_counter: OpenRouterCostCounter | None = None,
) -> OpenRouterClient:
    base_url = _trimmed(pricing.get("base_url"), "base_url")
    model = _trimmed(pricing.get("model"), "model")
    reject_openrouter_batch_entry_point(base_url=base_url, model=model)
    return OpenRouterClient(
        api_key=api_key,
        base_url=base_url,
        cost_counter=cost_counter
        or build_openrouter_cost_counter(
            pricing=pricing,
            precall_prompt_tokens=precall_prompt_tokens,
            precall_completion_tokens=precall_completion_tokens,
            already_spent_usd=already_spent_usd,
        ),
        timeout_seconds=timeout_seconds,
        proxy=proxy,
        transport=transport,
        recorder=recorder,
    )


def _require_chat_args(
    *,
    model: str,
    prompt: str,
    temperature: float | None,
    max_tokens: int,
) -> str:
    name = _require_model_name(model)
    if not isinstance(prompt, str) or not prompt.strip() or prompt != prompt.strip():
        raise ValueError("prompt must be a non-empty trimmed string")
    if temperature is not None:
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise ValueError("temperature must be a number or None")
        if not 0 <= float(temperature) <= 1:
            raise ValueError("temperature must be in [0, 1]")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive int")
    return name


def _require_model_name(value: Any) -> str:
    name = _trimmed(value, "model")
    reject_openrouter_batch_entry_point(base_url=DEFAULT_OPENROUTER_BASE_URL, model=name)
    return name


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


def _text_from_response(response: Any) -> str:
    text = _response_text(response)
    if isinstance(text, str) and text.strip():
        return text
    raise OpenRouterError("OpenRouter response is missing text")


def _response_text(response: Any) -> str | None:
    """Keep empty content too: a returned empty answer is not a missing response."""
    choices = _field(response, "choices")
    if isinstance(choices, (list, tuple)) and choices:
        content = _field(_field(choices[0], "message"), "content")
        if isinstance(content, str):
            return content
    text = _field(response, "text")
    return text if isinstance(text, str) else None


def _response_json(value: Any) -> Any:
    """Serialize the received API envelope before reporting a validation failure."""
    if isinstance(value, Mapping):
        return {key: _response_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_response_json(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return {key: _response_json(item) for key, item in vars(value).items()}
    return value


def _field(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    value = getattr(obj, key, None)
    if value is None:
        extra = getattr(obj, "model_extra", None)
        if isinstance(extra, Mapping):
            value = extra.get(key)
    return value


def _served_from_response(response: Any) -> dict[str, Any]:
    usage = _field(response, "usage")
    details = _field(usage, "completion_tokens_details")
    prompt_details = _field(usage, "prompt_tokens_details")
    if prompt_details is not None and not isinstance(prompt_details, Mapping):
        dump = getattr(prompt_details, "model_dump", None)
        prompt_details = dump() if callable(dump) else None
    cost = _field(usage, "cost")
    choices = _field(response, "choices")
    return {
        "provider": _field(response, "provider"),
        "service_tier": _field(response, "service_tier"),
        "model": _field(response, "model"),
        "finish_reason": _field(choices[0], "finish_reason") if isinstance(choices, (list, tuple)) and choices else None,
        "reported_cost_usd": None if cost is None else str(cost),
        "reasoning_tokens": _field(details, "reasoning_tokens"),
        "prompt_tokens_details": prompt_details,
    }


def _request_id(response: Any) -> str | None:
    if isinstance(response, Mapping):
        value = response.get("id")
        return value if isinstance(value, str) and value.strip() else None
    value = getattr(response, "id", None)
    return value if isinstance(value, str) and value.strip() else None
