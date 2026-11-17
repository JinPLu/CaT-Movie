"""OpenAI/CloseAI-shaped chat completion objects for tests.

Production CloseAI and video-VLM clients refuse a response without
``choices[0].finish_reason``. Tests that assert that gate, or that want a
truncated answer, must pass ``finish_reason`` explicitly rather than omitting it.
"""

from __future__ import annotations

from types import SimpleNamespace


def fake_chat_completion(
    content: str,
    *,
    finish_reason: str = "stop",
    prompt_tokens: int = 80,
    completion_tokens: int = 40,
    total_tokens: int | None = None,
    request_id: str = "chatcmpl-test",
    prompt_tokens_details: dict | None = None,
) -> SimpleNamespace:
    """Return a chat.completions object the truncation gate will accept.

    Default ``finish_reason="stop"`` is a finished, non-truncated answer.
    Pass ``finish_reason="length"`` when the test is specifically about
    hitting ``max_tokens``.
    """

    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        text=content,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=prompt_tokens_details,
        ),
        id=request_id,
    )
