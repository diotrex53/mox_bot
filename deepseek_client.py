"""Клиент для OpenAI-совместимых API (DeepSeek через aikeysforyou, Qwen через voidluv).

Умеет обычные ответы и SSE-стриминг с накоплением текста, мыслей и tool-call'ов.
"""

from __future__ import annotations

import inspect
import json
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


def _usage_numbers(usage: dict) -> dict:
    usage = usage or {}
    details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": int(
            details.get("reasoning_tokens")
            or usage.get("completion_thinking_tokens")
            or 0
        ),
        "cached_tokens": int(
            usage.get("cached_tokens")
            or usage.get("prompt_cache_hit_tokens")
            or prompt_details.get("cached_tokens")
            or details.get("cached_tokens")
            or 0
        ),
        "total_tokens": total_tokens or prompt_tokens + completion_tokens,
        "credit": float(usage.get("credit") or 0),
    }


def _normalize_tool_calls(raw_calls: list[dict]) -> list[dict]:
    tool_calls: list[dict] = []
    for tc in raw_calls or []:
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
    return tool_calls


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        http_client: httpx.AsyncClient,
        timeout: float = 120.0,
        label: str = "API",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.http = http_client
        self.timeout = timeout
        self.label = label

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _payload(
        self,
        messages: list[dict],
        *,
        model: str | None,
        tools: list[dict] | None,
        tool_choice: str | dict | None,
        max_tokens: int,
        effort: str,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
        }
        if effort and effort != "off":
            payload["thinking"] = {"type": "enabled"}
            payload["reasoning_effort"] = effort
        else:
            payload["thinking"] = {"type": "disabled"}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int = 4000,
        effort: str = "low",
        model: str | None = None,
        on_delta=None,
    ) -> ChatResult:
        if not self.configured:
            raise DeepSeekError(f"{self.label}: API-ключ не задан")
        stream = on_delta is not None
        payload = self._payload(
            messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            effort=effort,
            stream=stream,
        )
        if stream:
            return await self._chat_stream(payload, on_delta)
        return await self._chat_plain(payload)

    async def _chat_plain(self, payload: dict) -> ChatResult:
        try:
            response = await self.http.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json=payload,
                timeout=self.timeout,
            )
        except Exception as e:
            raise DeepSeekError(f"{self.label}: сетевая ошибка: {e}") from e

        if response.status_code >= 400:
            body = response.text[:1000]
            logging.warning(f"{self.label} HTTP {response.status_code}: {body[:500]}")
            raise DeepSeekError(
                f"{self.label} HTTP {response.status_code}: {body[:300]}",
                status_code=response.status_code,
                body=body,
            )

        try:
            data = response.json()
        except Exception as e:
            raise DeepSeekError(f"{self.label}: ответ не JSON: {e}") from e

        choices = data.get("choices") or []
        if not choices:
            raise DeepSeekError(f"{self.label}: пустой ответ: {str(data)[:300]}")
        choice = choices[0]
        msg = choice.get("message") or {}
        numbers = _usage_numbers(data.get("usage") or {})
        return ChatResult(
            content=(msg.get("content") or "").strip(),
            reasoning_content=(msg.get("reasoning_content") or "").strip(),
            tool_calls=_normalize_tool_calls(msg.get("tool_calls") or []),
            model=data.get("model") or payload.get("model") or self.model,
            finish_reason=choice.get("finish_reason") or "",
            **numbers,
        )

    async def _chat_stream(self, payload: dict, on_delta) -> ChatResult:
        content: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict] = {}
        finish = ""
        usage: dict = {}
        model_name = payload.get("model") or self.model

        async def emit(kind: str, piece: str) -> None:
            if not piece:
                return
            try:
                result = on_delta(kind, piece)
                if inspect.isawaitable(result):
                    await result
            except Exception as e:
                logging.warning(f"{self.label}: on_delta failed: {e}")

        try:
            async with self.http.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json=payload,
                timeout=self.timeout,
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")[:1000]
                    logging.warning(
                        f"{self.label} stream HTTP {response.status_code}: {body[:500]}"
                    )
                    raise DeepSeekError(
                        f"{self.label} HTTP {response.status_code}: {body[:300]}",
                        status_code=response.status_code,
                        body=body,
                    )

                async for line in response.aiter_lines():
                    line = (line or "").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                    except Exception:
                        continue
                    if data.get("usage"):
                        usage = data["usage"]
                    if data.get("model"):
                        model_name = data["model"]
                    for choice in data.get("choices") or []:
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
                        delta = choice.get("delta") or {}
                        piece = delta.get("content")
                        if piece:
                            content.append(piece)
                            await emit("content", piece)
                        rpiece = delta.get("reasoning_content") or delta.get("reasoning")
                        if rpiece:
                            reasoning.append(rpiece)
                            await emit("reasoning", rpiece)
                        for tc in delta.get("tool_calls") or []:
                            idx = int(tc.get("index") or 0)
                            slot = calls.setdefault(
                                idx, {"id": "", "name": "", "arguments": ""}
                            )
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            function = tc.get("function") or {}
                            if function.get("name"):
                                slot["name"] = function["name"]
                            if function.get("arguments"):
                                slot["arguments"] += function["arguments"]
        except DeepSeekError:
            raise
        except Exception as e:
            raise DeepSeekError(f"{self.label}: ошибка стрима: {e}") from e

        tool_calls = [
            {
                "id": slot["id"] or f"call_{idx}",
                "type": "function",
                "function": {
                    "name": slot["name"],
                    "arguments": slot["arguments"] or "{}",
                },
            }
            for idx, slot in sorted(calls.items())
            if slot["name"]
        ]
        numbers = _usage_numbers(usage)
        return ChatResult(
            content="".join(content).strip(),
            reasoning_content="".join(reasoning).strip(),
            tool_calls=tool_calls,
            model=model_name,
            finish_reason=finish,
            **numbers,
        )
