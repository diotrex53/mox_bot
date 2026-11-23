"""Минимальный клиент DeepSeek через OpenAI-совместимый API."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx


class DeepSeekError(Exception):
    def __init__(self, message: str, status_code: int | None = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@dataclass
class ChatResult:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    credit: float = 0.0
    model: str = ""
    finish_reason: str = ""

    def assistant_message(self, include_reasoning: bool = False) -> dict:
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": self.content or "",
        }
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if include_reasoning and self.reasoning_content:
            msg["reasoning_content"] = self.reasoning_content
        return msg


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        http_client: httpx.AsyncClient,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.http = http_client
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    async def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int = 4000,
        effort: str = "low",
    ) -> ChatResult:
        if not self.configured:
            raise DeepSeekError("AIKEYS_API не задан")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "thinking": {"type": "enabled"},
            "reasoning_effort": effort,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        try:
            response = await self.http.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except Exception as e:
            raise DeepSeekError(f"Сетевая ошибка DeepSeek: {e}") from e

        if response.status_code >= 400:
            body = response.text[:1000]
            logging.warning(
                f"DeepSeek HTTP {response.status_code}: {body[:500]}"
            )
            raise DeepSeekError(
                f"DeepSeek HTTP {response.status_code}: {body[:300]}",
                status_code=response.status_code,
                body=body,
            )

        try:
            data = response.json()
        except Exception as e:
            raise DeepSeekError(f"DeepSeek вернул не JSON: {e}") from e

        choices = data.get("choices") or []
        if not choices:
            raise DeepSeekError(f"DeepSeek вернул пустой ответ: {str(data)[:300]}")
        choice = choices[0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}

        raw_calls = msg.get("tool_calls") or []
        tool_calls: list[dict] = []
        for tc in raw_calls:
            function = tc.get("function") or {}
            tool_calls.append(
                {
                    "id": tc.get("id") or f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": function.get("name") or "",
                        "arguments": function.get("arguments") or "{}",
                    },
                }
            )

        result = ChatResult(
            content=(msg.get("content") or "").strip(),
            reasoning_content=(msg.get("reasoning_content") or "").strip(),
            tool_calls=tool_calls,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            reasoning_tokens=int(
                details.get("reasoning_tokens")
                or usage.get("completion_thinking_tokens")
                or 0
            ),
            cached_tokens=int(
                usage.get("cached_tokens")
                or usage.get("prompt_cache_hit_tokens")
                or details.get("cached_tokens")
                or 0
            ),
            total_tokens=int(usage.get("total_tokens") or 0),
            credit=float(usage.get("credit") or 0),
            model=data.get("model") or self.model,
            finish_reason=choice.get("finish_reason") or "",
        )
        if not result.total_tokens:
            result.total_tokens = result.prompt_tokens + result.completion_tokens
        return result
