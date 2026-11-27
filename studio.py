"""Студия нейросетей «Мудрого Мха».

Здесь живут:
• каталоги бесплатных (Pollinations anonymous/VIP, Groq, Gemini) и платных
  (DeepSeek через шлюз) моделей;
• клиенты провайдеров;
• карточки-справки о моделях;
• дневные пулы токенов free/paid.

Ключи берутся только из переменных окружения Render. В репозитории их нет.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
from openai import AsyncOpenAI

from deepseek_client import ChatResult, DeepSeekClient, DeepSeekError

MSK = timezone(timedelta(hours=3))

FREE = "free"
PAID = "paid"
SECTION_LABELS = {FREE: "🆓 Free", PAID: "💎 Paid"}
SECTION_TITLES = {FREE: "Бесплатные модели", PAID: "Платные модели"}
POOL_LIMITS = {FREE: 250_000, PAID: 100_000}
POOL_NAMES = {FREE: "free", PAID: "paid"}

DEFAULT_FREE_MODEL = "groq@openai/gpt-oss-120b"
DEFAULT_PAID_MODEL = "deepseek@deepseek-v4.1-flash-1017"

# ---------------------------------------------------------------------------
# Настройки из окружения
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API", "")
GROQ_API_PAID = os.environ.get("GROQ_API_PAID", "") or os.environ.get(
    "GROQ_API_NEW2", ""
)
GROQ_API_FREE = os.environ.get("GROQ_API_FREE", "") or os.environ.get(
    "GROQ_API_NEW3", ""
)
GEMINI_API_KEY = os.environ.get("GEMINI_API", "")
POLLINATIONS_API_KEY = (
    os.environ.get("POLLINATIONS_API", "")
    or os.environ.get("POLLINATIONS_VIP_API", "")
    or os.environ.get("POLLINATIONS_KEY", "")
)
TAVILY_API_KEY = os.environ.get("TAVIL_API", "") or os.environ.get(
    "TAVILY_API", ""
)
TAVILY_API_KEY_PAID = os.environ.get("TAVILY_API_PAID", "") or os.environ.get(
    "TAVILY_API_NEW", ""
)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_VISION_MODEL = "qwen/qwen3.8-27b"
GROQ_FALLBACK_VOICE_MODEL = "whisper-large-v3-turbo"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
POLLINATIONS_BASE_URL = "https://text.pollinations.ai/openai"
POLLINATIONS_MODELS_URL = "https://text.pollinations.ai/models"

GATEWAY_API_KEY = os.environ.get("GATEWAY_API", "")
GATEWAY_BASE_URL = os.environ.get(
    "GATEWAY_BASE_URL", "https://api.aikeysforyou.com/v1"
).rstrip("/")
DEFAULT_GATEWAY_MODEL = "deepseek-v4.1-flash-1017"

PROVIDER_LABELS = {
    "gateway": "DeepSeek · шлюз aikeysforyou",
    "pollinations_free": "Pollinations · анонимно",
    "pollinations_vip": "Pollinations · VIP",
    "groq": "Groq Cloud",
    "gemini": "Google Gemini",
}

FAMILY_LABELS = {
    "deepseek": "DeepSeek",
    "pollinations_free": "Pollinations Free",
    "pollinations_vip": "Pollinations VIP",
    "groq": "Groq",
    "gemini": "Gemini",
}

FAMILY_ORDER = {
    PAID: ["deepseek"],
    FREE: ["pollinations_free", "pollinations_vip", "groq", "gemini"],
}

FAMILY_SECTIONS = {
    "deepseek": PAID,
    "pollinations_free": FREE,
    "pollinations_vip": FREE,
    "groq": FREE,
    "gemini": FREE,
}

# ---------------------------------------------------------------------------
# Метаданные моделей
# ---------------------------------------------------------------------------

# Значения, которые провайдеры не раскрывают, показываем как «н/д».
# Ключ — id модели (без семейства).
KNOWN_INFO: dict[str, dict] = {
    # DeepSeek через шлюз
    "deepseek-v4.1-flash-1017": {
        "params": "н/д",
        "context": 128_000,
        "max_output": 32_000,
        "license": "проприетарная",
        "reasoning": True,
        "price": "по тарифу шлюза",
        "link": "https://api.aikeysforyou.com",
    },
    "deepseek-v4.1-flash": {
        "params": "н/д",
        "context": 128_000,
        "max_output": 32_000,
        "license": "проприетарная",
        "reasoning": True,
        "price": "по тарифу шлюза",
        "link": "https://api.aikeysforyou.com",
    },
    # Groq
    "openai/gpt-oss-120b": {
        "params": "120B (MoE, ~5B активных)",
        "context": 131_072,
        "max_output": 32_768,
        "license": "Apache 2.0",
        "reasoning": True,
        "link": "https://console.groq.com/docs/models",
    },
    "openai/gpt-oss-20b": {
        "params": "20B (MoE, ~3.6B активных)",
        "context": 131_072,
        "max_output": 32_768,
        "license": "Apache 2.0",
        "reasoning": True,
        "link": "https://console.groq.com/docs/models",
    },
    "llama-3.3-70b-versatile": {
        "params": "70B",
        "context": 131_072,
        "max_output": 32_768,
        "license": "Llama 3.3 Community",
        "reasoning": False,
        "link": "https://console.groq.com/docs/models",
    },
    "llama-3.1-8b-instant": {
        "params": "8B",
        "context": 131_072,
        "max_output": 8_192,
        "license": "Llama 3.1 Community",
        "reasoning": False,
        "link": "https://console.groq.com/docs/models",
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "params": "17B (16 экспертов)",
        "context": 131_072,
        "max_output": 8_192,
        "license": "Llama 4 Community",
        "reasoning": False,
        "vision": True,
        "link": "https://console.groq.com/docs/models",
    },
    "meta-llama/llama-4-maverick-17b-128e-instruct": {
        "params": "17B (128 экспертов)",
        "context": 131_072,
        "max_output": 8_192,
        "license": "Llama 4 Community",
        "reasoning": False,
        "vision": True,
        "link": "https://console.groq.com/docs/models",
    },
    "qwen/qwen3-32b": {
        "params": "32B",
        "context": 131_072,
        "max_output": 16_384,
        "license": "Apache 2.0",
        "reasoning": True,
        "link": "https://console.groq.com/docs/models",
    },
    "qwen/qwen3.8-27b": {
        "params": "27B",
        "context": 131_072,
        "max_output": 16_384,
        "license": "Apache 2.0",
        "reasoning": True,
        "vision": True,
        "link": "https://console.groq.com/docs/models",
    },
    "moonshotai/kimi-k2-instruct": {
        "params": "1T (MoE)",
        "context": 131_072,
        "max_output": 16_384,
        "license": "Modified MIT",
        "reasoning": True,
        "link": "https://console.groq.com/docs/models",
    },
    "deepseek-r1-distill-llama-70b": {
        "params": "70B (distill)",
        "context": 131_072,
        "max_output": 16_384,
        "license": "MIT / Llama",
        "reasoning": True,
        "link": "https://console.groq.com/docs/models",
    },
    # Gemini
    "gemini-3.5-flash-lite": {
        "params": "н/д",
        "context": 1_000_000,
        "max_output": 65_536,
        "license": "проприетарная",
        "reasoning": True,
        "vision": True,
        "link": "https://ai.google.dev/gemini-api/docs/models",
    },
    "gemini-3.5-flash": {
        "params": "н/д",
        "context": 1_000_000,
        "max_output": 65_536,
        "license": "проприетарная",
        "reasoning": True,
        "vision": True,
        "link": "https://ai.google.dev/gemini-api/docs/models",
    },
    "gemini-3.5-pro": {
        "params": "н/д",
        "context": 2_000_000,
        "max_output": 65_536,
        "license": "проприетарная",
        "reasoning": True,
        "vision": True,
        "link": "https://ai.google.dev/gemini-api/docs/models",
    },
    "gemini-2.5-flash": {
        "params": "н/д",
        "context": 1_048_576,
        "max_output": 65_536,
        "license": "проприетарная",
        "reasoning": True,
        "vision": True,
        "link": "https://ai.google.dev/gemini-api/docs/models",
    },
    "gemini-2.5-pro": {
        "params": "н/д",
        "context": 1_048_576,
        "max_output": 65_536,
        "license": "проприетарная",
        "reasoning": True,
        "vision": True,
        "link": "https://ai.google.dev/gemini-api/docs/models",
    },
}

# Известные id Pollinations, которые точно отдают текст.
POLLINATIONS_FALLBACK = [
    "openai",
    "openai-fast",
    "openai-large",
    "openai-reasoning",
    "qwen-coder",
    "mistral",
    "llama",
    "llamascout",
    "deepseek",
    "deepseek-reasoning",
    "gemini",
    "gemini-flash",
    "gemini-thinking",
    "nova-fast",
    "grok",
    "grok-large",
    "amazon-nova",
    "mistral-small",
    "phi",
]

GROQ_FALLBACK = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "qwen/qwen3-32b",
    "qwen/qwen3.8-27b",
    "moonshotai/kimi-k2-instruct",
    "deepseek-r1-distill-llama-70b",
]

GEMINI_FALLBACK = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.5-pro",
    "gemini-2.5-flash",
]


# ---------------------------------------------------------------------------
# Клиенты
# ---------------------------------------------------------------------------

http_client: httpx.AsyncClient | None = None
gateway_client: DeepSeekClient | None = None
pollinations_free_client: DeepSeekClient | None = None
pollinations_vip_client: DeepSeekClient | None = None
groq_client: AsyncOpenAI | None = None
groq_paid_client: AsyncOpenAI | None = None
groq_free_client: AsyncOpenAI | None = None

AI_TIMEOUT = 180.0
PREMIUM_TIMEOUT: float | httpx.Timeout = 90.0


def configure(
    client: httpx.AsyncClient,
    *,
    ai_timeout: float = 180.0,
    premium_timeout: float | httpx.Timeout = 90.0,
) -> None:
    """Создаёт клиентов. Вызывается один раз на старте бота."""
    global http_client, gateway_client, pollinations_free_client
    global pollinations_vip_client, groq_client, groq_paid_client
    global groq_free_client, AI_TIMEOUT, PREMIUM_TIMEOUT

    http_client = client
    AI_TIMEOUT = ai_timeout
    PREMIUM_TIMEOUT = premium_timeout

    gateway_client = DeepSeekClient(
        api_key=GATEWAY_API_KEY,
        base_url=GATEWAY_BASE_URL,
        model=DEFAULT_GATEWAY_MODEL,
        http_client=client,
        timeout=premium_timeout,
        label="Шлюз",
    )
    pollinations_free_client = DeepSeekClient(
        api_key="",
        base_url=POLLINATIONS_BASE_URL,
        model="openai",
        http_client=client,
        timeout=ai_timeout,
        label="Pollinations Free",
        anonymous=True,
    )
    pollinations_vip_client = DeepSeekClient(
        api_key=POLLINATIONS_API_KEY,
        base_url=POLLINATIONS_BASE_URL,
        model="openai",
        http_client=client,
        timeout=ai_timeout,
        label="Pollinations VIP",
    )

    def _oai(key: str) -> AsyncOpenAI | None:
        if not key:
            return None
        return AsyncOpenAI(
            api_key=key,
            base_url=GROQ_BASE_URL,
            http_client=client,
            max_retries=0,
            timeout=ai_timeout,
        )

    groq_client = _oai(GROQ_API_KEY)
    groq_paid_client = _oai(GROQ_API_PAID or GROQ_API_KEY)
    groq_free_client = _oai(GROQ_API_FREE or GROQ_API_KEY)


# ---------------------------------------------------------------------------
# Каталог
# ---------------------------------------------------------------------------

CATALOG: dict[str, list[dict]] = {}
CATALOG_LOADED_AT = 0.0
CATALOG_SOURCE = "fallback"


def _now_iso() -> str:
    return datetime.now(MSK).strftime("%d.%m.%Y")


def _alias(model: str) -> str:
    return str(model or "").strip()


def _info_for(model: str, *, vision: bool = False, reasoning: bool = False) -> dict:
    base = dict(KNOWN_INFO.get(model) or {})
    base.setdefault("vision", bool(vision))
    base.setdefault("reasoning", bool(reasoning))
    return base


def make_spec(
    section: str,
    family: str,
    model: str,
    *,
    label: str | None = None,
    provider: str | None = None,
    vision: bool = False,
    reasoning: bool = False,
    tools: bool = True,
    description: str = "",
) -> dict:
    provider = provider or family
    info = _info_for(model, vision=vision, reasoning=reasoning)
    spec = {
        "key": f"{family}@{model}",
        "section": section,
        "family": family,
        "family_label": FAMILY_LABELS.get(family, family),
        "provider": provider,
        "provider_label": PROVIDER_LABELS.get(provider, provider),
        "model": model,
        "label": label or model,
        "vision": bool(info.get("vision") or vision),
        "reasoning": bool(info.get("reasoning") or reasoning),
        "tools": tools,
        "description": description,
        "params": info.get("params"),
        "context": info.get("context"),
        "max_output": info.get("max_output"),
        "license": info.get("license"),
        "price": info.get("price"),
        "speed": info.get("speed"),
        "link": info.get("link") or "",
        "release": info.get("release") or "",
        "status": "доступна",
    }
    return spec


def _fallback_catalog() -> dict[str, list[dict]]:
    catalog: dict[str, list[dict]] = {}

    catalog["deepseek"] = [
        make_spec(PAID, "deepseek", model, provider="gateway") for model in (
            "deepseek-v4.1-flash-1017",
            "deepseek-v4.1-flash",
        )
    ]
    catalog["pollinations_free"] = [
        make_spec(FREE, "pollinations_free", model) for model in POLLINATIONS_FALLBACK
    ]
    catalog["pollinations_vip"] = [
        make_spec(FREE, "pollinations_vip", model) for model in POLLINATIONS_FALLBACK
    ]
    catalog["groq"] = [
        make_spec(FREE, "groq", model, vision=(model == GROQ_VISION_MODEL))
        for model in GROQ_FALLBACK
    ]
    catalog["gemini"] = [
        make_spec(FREE, "gemini", model, vision=True) for model in GEMINI_FALLBACK
    ]
    return catalog


def _dedupe(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for item in items:
        key = item["key"]
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _parse_pollinations(data: Any, family: str) -> list[dict]:
    specs: list[dict] = []
    if isinstance(data, dict):
        data = data.get("data") or data.get("models") or []
    if not isinstance(data, list):
        return specs
    for item in data:
        if isinstance(item, str):
            model = item
            meta: dict = {}
        elif isinstance(item, dict):
            model = str(
                item.get("name")
                or item.get("id")
                or item.get("model")
                or ""
            ).strip()
            meta = item
        else:
            continue
        if not model:
            continue
        output = meta.get("output_modalities") or meta.get("output") or []
        input_mods = meta.get("input_modalities") or meta.get("input") or []
        if isinstance(output, str):
            output = [output]
        if isinstance(input_mods, str):
            input_mods = [input_mods]
        if output and "text" not in [str(x).lower() for x in output]:
            continue
        if input_mods and not (
            {"text", "image"} & {str(x).lower() for x in input_mods}
        ):
            continue
        lowered = model.lower()
        if any(
            bad in lowered
            for bad in ("audio", "embedding", "embed", "tts", "whisper", "sdxl", "flux")
        ):
            continue
        vision = bool(meta.get("vision")) or bool(
            {"image"} & {str(x).lower() for x in input_mods}
        )
        reasoning = bool(meta.get("reasoning")) or "reason" in lowered or "think" in lowered
        description = str(meta.get("description") or "").strip()
        specs.append(
            make_spec(
                FREE,
                family,
                model,
                vision=vision,
                reasoning=reasoning,
                description=description,
            )
        )
    return specs


def _parse_groq(data: Any) -> list[dict]:
    specs: list[dict] = []
    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return specs
    for item in items:
        model = str(
            (item.get("id") if isinstance(item, dict) else item) or ""
        ).strip()
        if not model:
            continue
        lowered = model.lower()
        if any(
            bad in lowered
            for bad in (
                "whisper",
                "tts",
                "embed",
                "guard",
                "moderation",
                "distil-whisper",
            )
        ):
            continue
        vision = any(
            marker in lowered
            for marker in (
                "vision",
                "scout",
                "maverick",
                "llama-4",
                "llava",
                "qwen3.8",
                "qwen2.5-vl",
            )
        )
        reasoning = any(
            marker in lowered
            for marker in ("reason", "r1", "thinking", "qwen3", "qwen3.8", "gpt-oss")
        )
        specs.append(
            make_spec(
                FREE,
                "groq",
                model,
                vision=vision,
                reasoning=reasoning,
            )
        )
    return specs


def _parse_gemini(data: Any) -> list[dict]:
    specs: list[dict] = []
    items = data.get("models") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return specs
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").removeprefix("models/")
        if not name.startswith("gemini"):
            continue
        methods = item.get("supportedGenerationMethods") or []
        if methods and "generateContent" not in methods:
            continue
        lowered = name.lower()
        if any(
            bad in lowered
            for bad in (
                "embedding", "embed", "aqa", "imagen", "image",
                "veo", "tts", "audio",
            )
        ):
            continue
        specs.append(make_spec(FREE, "gemini", name, vision=True, reasoning=True))
    return specs


async def _get_json(url: str, *, headers: dict | None = None) -> Any:
    if http_client is None:
        return None
    try:
        response = await http_client.get(url, headers=headers or {}, timeout=25.0)
        if response.status_code >= 400:
            logging.warning(f"studio: {url} -> HTTP {response.status_code}")
            return None
        return response.json()
    except Exception as e:
        logging.warning(f"studio: {url} failed: {e}")
        return None


async def refresh_catalog() -> dict[str, list[dict]]:
    """Тянет живые каталоги. При сбое оставляет рабочий fallback."""
    global CATALOG, CATALOG_LOADED_AT, CATALOG_SOURCE

    fallback = _fallback_catalog()
    catalog = {family: list(items) for family, items in fallback.items()}

    poll_headers = {}
    if POLLINATIONS_API_KEY:
        poll_headers["Authorization"] = f"Bearer {POLLINATIONS_API_KEY}"

    free_data, vip_data, groq_data, gemini_data = await asyncio.gather(
        _get_json(POLLINATIONS_MODELS_URL),
        _get_json(POLLINATIONS_MODELS_URL, headers=poll_headers),
        _get_json(
            f"{GROQ_BASE_URL}/models",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"} if GROQ_API_KEY else {},
        ),
        _get_json(
            f"{GEMINI_BASE_URL}/models?key={GEMINI_API_KEY}"
            if GEMINI_API_KEY
            else f"{GEMINI_BASE_URL}/models"
        ),
        return_exceptions=True,
    )

    sources: list[str] = []
    if isinstance(free_data, (list, dict)):
        parsed = _dedupe(_parse_pollinations(free_data, "pollinations_free"))
        if parsed:
            catalog["pollinations_free"] = parsed
            sources.append("pollinations-free")
    if isinstance(vip_data, (list, dict)):
        parsed = _dedupe(_parse_pollinations(vip_data, "pollinations_vip"))
        if parsed:
            catalog["pollinations_vip"] = parsed
            sources.append("pollinations-vip")
    if isinstance(groq_data, (list, dict)):
        parsed = _dedupe(_parse_groq(groq_data))
        if parsed:
            catalog["groq"] = parsed
            sources.append("groq")
    if isinstance(gemini_data, (list, dict)):
        parsed = _dedupe(_parse_gemini(gemini_data))
        if parsed:
            catalog["gemini"] = parsed
            sources.append("gemini")

    CATALOG = catalog
    CATALOG_LOADED_AT = time.time()
    CATALOG_SOURCE = ", ".join(sources) if sources else "fallback"
    logging.info(
        "studio catalog: "
        + ", ".join(f"{k}={len(v)}" for k, v in catalog.items())
        + f" | live={CATALOG_SOURCE}"
    )
    return catalog


def get_catalog() -> dict[str, list[dict]]:
    if not CATALOG:
        return _fallback_catalog()
    return CATALOG


def all_specs(section: str | None = None) -> list[dict]:
    catalog = get_catalog()
    out: list[dict] = []
    for family in FAMILY_ORDER.get(section or PAID, FAMILY_ORDER[PAID]) + (
        FAMILY_ORDER[FREE] if section is None else []
    ):
        out.extend(catalog.get(family) or [])
    if section is None:
        # Без дублей: порядок paid → free.
        seen: set[str] = set()
        unique: list[dict] = []
        for spec in out:
            if spec["key"] in seen:
                continue
            seen.add(spec["key"])
            unique.append(spec)
        return unique
    return out


def families(section: str) -> list[str]:
    catalog = get_catalog()
    return [f for f in FAMILY_ORDER[section] if catalog.get(f)]


def model_by_key(key: str) -> dict | None:
    for spec in all_specs():
        if spec["key"] == key:
            return spec
    return None


# ---------------------------------------------------------------------------
# Отправка запроса
# ---------------------------------------------------------------------------


async def _retry(coro_factory, retries: int = 2):
    last: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            return await coro_factory()
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt + 1 < max(1, retries):
                await asyncio.sleep(0.7 * (attempt + 1))
    raise last or RuntimeError("неизвестная ошибка")


async def pollinations_chat(
    spec: dict,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = "auto",
    max_tokens: int = 2000,
    on_delta=None,
) -> ChatResult:
    client = (
        pollinations_vip_client
        if spec.get("provider") == "pollinations_vip"
        else pollinations_free_client
    )
    if client is None or not client.configured:
        raise DeepSeekError("Pollinations не настроен")

    async def _ask(with_tools: bool):
        return await client.chat(
            messages,
            tools=tools if with_tools else None,
            tool_choice=tool_choice if with_tools else None,
            max_tokens=max_tokens,
            effort="off",
            model=spec["model"],
            on_delta=on_delta,
        )

    try:
        return await _retry(lambda: _ask(bool(tools)), retries=2)
    except Exception as e:
        if not tools:
            raise
        logging.warning(
            f"Pollinations {spec['model']}: tools не приняты ({e}); повтор без tools"
        )
        return await _retry(lambda: _ask(False), retries=2)


def _gemini_parts(content: Any) -> list[dict]:
    if isinstance(content, str):
        return [{"text": content}]
    if isinstance(content, list):
        parts: list[dict] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append({"text": str(item.get("text") or "")})
            elif item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url") or ""
                if url.startswith("data:") and ";base64," in url:
                    header, b64 = url.split(";base64,", 1)
                    mime = header.replace("data:", "") or "image/jpeg"
                    parts.append(
                        {"inline_data": {"mime_type": mime, "data": b64}}
                    )
        return parts or [{"text": ""}]
    return [{"text": str(content or "")}]


def _gemini_usage(usage: dict) -> dict:
    prompt = int(usage.get("promptTokenCount") or 0)
    completion = int(usage.get("candidatesTokenCount") or 0)
    reasoning = int(usage.get("thoughtsTokenCount") or 0)
    cached = int(usage.get("cachedContentTokenCount") or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "cached_tokens": cached,
        "total_tokens": prompt + completion + reasoning,
    }


async def gemini_chat(
    spec: dict,
    messages: list[dict],
    *,
    max_tokens: int = 2000,
    **_kwargs,
) -> ChatResult:
    if not GEMINI_API_KEY:
        raise DeepSeekError("Gemini: ключ GEMINI_API не задан")
    if http_client is None:
        raise DeepSeekError("Gemini: HTTP-клиент не настроен")

    system_parts: list[str] = []
    contents: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            system_parts.append(str(content or ""))
            continue
        if role == "tool":
            role = "user"
            content = f"Результат инструмента:\n{content}"
        gem_role = "model" if role == "assistant" else "user"
        parts = _gemini_parts(content)
        if not parts:
            continue
        contents.append({"role": gem_role, "parts": parts})

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": int(max_tokens or 2000),
            "temperature": 0.7,
        },
    }
    if system_parts:
        payload["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_parts)}]
        }

    async def _do():
        response = await http_client.post(
            f"{GEMINI_BASE_URL}/models/{spec['model']}:generateContent",
            headers={"x-goog-api-key": GEMINI_API_KEY},
            json=payload,
            timeout=AI_TIMEOUT,
        )
        if response.status_code >= 400:
            raise DeepSeekError(
                f"Gemini HTTP {response.status_code}: {response.text[:300]}",
                status_code=response.status_code,
                body=response.text[:1000],
            )
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise DeepSeekError(f"Gemini: пустой ответ: {str(data)[:300]}")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()
        if not text:
            raise DeepSeekError(
                "Gemini не вернул текст "
                f"(finishReason={candidates[0].get('finishReason')})"
            )
        return ChatResult(
            content=text,
            model=spec["model"],
            finish_reason=candidates[0].get("finishReason") or "",
            **_gemini_usage(data.get("usageMetadata") or {}),
        )

    return await _retry(_do, retries=2)


# ---------------------------------------------------------------------------
# Фото и голос
# ---------------------------------------------------------------------------

VISION_PROMPT = (
    "Опиши это изображение подробно для памяти ассистента: объекты, люди, "
    "цвета, расположение, весь видимый текст, важные детали. 80–150 слов. "
    "Пиши только описание, без вступлений вроде «на фото»."
)


async def describe_image(
    image_bytes: bytes,
    mime: str,
    *,
    section: str = FREE,
) -> tuple[str, dict]:
    """Распознавание фото только через Groq new3 (бесплатный ключ)."""
    import base64

    client = groq_free_client
    if client is None:
        raise RuntimeError("Groq vision: бесплатный ключ GROQ_API_FREE не задан")
    b64 = base64.b64encode(image_bytes).decode()
    content = [
        {"type": "text", "text": VISION_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ]

    async def _do():
        response = await client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=800,
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("Groq vision вернул пустой ответ")
        return text, response

    return await _retry(_do, retries=2)


async def transcribe_voice(
    file_path: str,
    *,
    section: str = FREE,
    language: str = "ru",
) -> tuple[str, int]:
    """Whisper: free → new3, paid → new2. Возвращает текст и примерный расход."""
    import os as _os

    client = groq_paid_client if section == PAID else groq_free_client
    if client is None:
        client = groq_client
    if client is None:
        raise RuntimeError("Groq Whisper: ключи не заданы")

    ext = _os.path.splitext(file_path)[1].lower()
    if ext not in {
        ".flac", ".mp3", ".mp4", ".mpeg", ".m4a",
        ".ogg", ".opus", ".wav", ".webm",
    }:
        ext = ".ogg"
    content_type = {
        ".flac": "audio/flac",
        ".mp3": "audio/mpeg",
        ".mp4": "audio/mp4",
        ".mpeg": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
        ".opus": "audio/opus",
        ".wav": "audio/wav",
        ".webm": "audio/webm",
    }[ext]

    async def _do():
        with open(file_path, "rb") as f:
            result = await asyncio.wait_for(
                client.audio.transcriptions.create(
                    file=(f"voice{ext}", f, content_type),
                    model=GROQ_FALLBACK_VOICE_MODEL,
                    language=language,
                ),
                timeout=AI_TIMEOUT,
            )
        return (result.text or "").strip()

    text = await _retry(_do, retries=2)
    estimate = max(40, len(text) // 3)
    return text, estimate


# ---------------------------------------------------------------------------
# Поиск
# ---------------------------------------------------------------------------


def tavily_key(section: str) -> str:
    if section == PAID:
        return TAVILY_API_KEY_PAID or TAVILY_API_KEY
    return TAVILY_API_KEY


# ---------------------------------------------------------------------------
# Справка о модели
# ---------------------------------------------------------------------------


def _fmt(value: Any, *, tokens: bool = False) -> str:
    if value is None or value == "":
        return "н/д"
    if tokens:
        try:
            return f"{int(value):,}".replace(",", " ") + " ток."
        except Exception:
            return str(value)
    return str(value)


def _modalities(spec: dict) -> str:
    parts = ["текст"]
    if spec.get("vision"):
        parts.append("фото")
    parts.append("голос (Whisper)")
    return ", ".join(parts)


def info_text(spec: dict) -> str:
    family = spec.get("family_label") or spec.get("family") or ""
    section = "Платные" if spec.get("section") == PAID else "Бесплатные"
    rows = [
        ("Название", spec.get("label") or spec.get("model")),
        ("ID", spec.get("model")),
        ("Провайдер", spec.get("provider_label") or spec.get("provider")),
        ("Семейство", family),
        ("Раздел", section),
        ("Параметры", _fmt(spec.get("params"))),
        ("Контекст", _fmt(spec.get("context"), tokens=True)),
        ("Модальности", _modalities(spec)),
        (
            "Инструменты",
            "нативные tool-calls" if spec.get("tools") else "текстовая эмуляция",
        ),
        ("Reasoning", "да" if spec.get("reasoning") else "нет"),
        ("Макс. вывод", _fmt(spec.get("max_output"), tokens=True)),
        (
            "Цена",
            spec.get("price")
            or ("бесплатно" if spec.get("section") == FREE else "н/д"),
        ),
        ("Скорость", _fmt(spec.get("speed"))),
        ("Лицензия", _fmt(spec.get("license"))),
        ("Статус", _fmt(spec.get("status"))),
        ("Дата релиза", _fmt(spec.get("release"))),
        ("Ссылка", spec.get("link") or "н/д"),
        (
            "Доступ",
            "выдаётся всем из белого списка"
            if spec.get("section") == PAID
            else "бесплатно, в пределах пула 250к/день",
        ),
    ]
    width = max(len(name) for name, _ in rows)
    lines = [f"{name.ljust(width)} │ {value}" for name, value in rows]
    table = "\n".join(lines)
    description = (spec.get("description") or "").strip()
    text = (
        f"ℹ️ <b>{_escape(spec.get('label') or spec.get('model'))}</b>\n\n"
        f"<pre>{_escape(table)}</pre>"
    )
    if description:
        text += f"\n\n<i>{_escape(description[:600])}</i>"
    text += (
        "\n\n💡 Совет: выбери модель кнопкой ниже — контекст чата сохранится."
    )
    return text


def _escape(text: Any) -> str:
    import html as _html

    return _html.escape(str(text or ""), quote=False)


# ---------------------------------------------------------------------------
# Пулы токенов
# ---------------------------------------------------------------------------


def pool_limit(section: str) -> int:
    return POOL_LIMITS.get(section, POOL_LIMITS[FREE])


def pool_label(section: str) -> str:
    return SECTION_LABELS.get(section, section)


def today_msk() -> date:
    return datetime.now(MSK).date()


def reset_text() -> str:
    return "Сброс в 00:00 МСК."


def status_line(section: str, used: int, *, owner: bool = False) -> str:
    if owner:
        return f"{SECTION_LABELS.get(section, section)}: безлимит (владелец)"
    limit = pool_limit(section)
    left = max(0, limit - int(used))
    return (
        f"{SECTION_LABELS.get(section, section)}: "
        f"{_fmt(left, tokens=True)} из {_fmt(limit, tokens=True)}"
    )
