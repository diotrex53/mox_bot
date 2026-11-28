"""Студия нейросетей «Мудрого Мха».

Здесь живут:
• каталоги бесплатных (Pollinations anonymous/VIP, Groq, Gemini,
  Cloudflare Workers AI, OpenRouter) и платных (DeepSeek через шлюз) моделей;
• клиенты провайдеров;
• карточки-справки о моделях;
• дневные пулы токенов free/paid.

Ключи берутся только из переменных окружения Render. В репозитории их нет.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

import httpx
from openai import AsyncOpenAI

from deepseek_client import ChatResult, DeepSeekClient, DeepSeekError
from prompts import VISION_PROMPT

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
CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID", "") or os.environ.get(
    "CLOUDFLARE_ACCOUNT_ID", ""
)
CF_API_TOKEN = os.environ.get("CF_API_TOKEN", "") or os.environ.get(
    "CLOUDFLARE_API_TOKEN", ""
)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API", "") or os.environ.get(
    "OPENROUTER_API_KEY", ""
)
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
POLLINATIONS_GEN_BASE_URL = "https://gen.pollinations.ai/v1"
POLLINATIONS_VIP_MODELS_URL = "https://gen.pollinations.ai/v1/models"
CF_BASE_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}"
    if CF_ACCOUNT_ID
    else "https://api.cloudflare.com/client/v4/accounts"
)
CF_CHAT_BASE_URL = f"{CF_BASE_URL}/ai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

GATEWAY_API_KEY = os.environ.get("GATEWAY_API", "")
GATEWAY_BASE_URL = os.environ.get(
    "GATEWAY_BASE_URL", "https://api.aikeysforyou.com/v1"
).rstrip("/")
DEFAULT_GATEWAY_MODEL = "deepseek-v4.1-flash-1017"

PROVIDER_LABELS = {
    "gateway": "DeepSeek · шлюз aikeysforyou",
    "pollinations_free": "Pollinations · Text Free (анонимно)",
    "pollinations_vip": "Pollinations · Text VIP",
    "pollinations_image_free": "Pollinations · Image Free (анонимно)",
    "pollinations_image_vip": "Pollinations · Image VIP",
    "groq": "Groq Cloud",
    "gemini": "Google Gemini",
    "cloudflare": "Cloudflare Workers AI",
    "cloudflare_image": "Cloudflare Workers AI",
    "openrouter": "OpenRouter",
}

COMPANY_BY_PROVIDER = {
    "gateway": "DeepSeek",
    "pollinations_free": "Pollinations",
    "pollinations_vip": "Pollinations",
    "pollinations_image_free": "Pollinations",
    "pollinations_image_vip": "Pollinations",
    "groq": "Groq",
    "gemini": "Google",
    "cloudflare": "Cloudflare",
    "cloudflare_image": "Cloudflare",
    "openrouter": "OpenRouter",
}

FAMILY_LABELS = {
    "deepseek": "DeepSeek",
    "pollinations_free": "Pollinations Text Free",
    "pollinations_vip": "Pollinations Text VIP",
    "pollinations_image_free": "Pollinations Image Free",
    "pollinations_image_vip": "Pollinations Image VIP",
    "groq": "Groq",
    "gemini": "Gemini",
    "cloudflare": "Cloudflare Text",
    "cloudflare_image": "Cloudflare Images",
    "openrouter": "OpenRouter Free",
}

FAMILY_ORDER = {
    PAID: ["deepseek"],
    FREE: [
        "pollinations_free",
        "pollinations_vip",
        "pollinations_image_free",
        "pollinations_image_vip",
        "groq",
        "gemini",
        "cloudflare",
        "cloudflare_image",
        "openrouter",
    ],
}

FAMILY_SECTIONS = {
    "deepseek": PAID,
    "pollinations_free": FREE,
    "pollinations_vip": FREE,
    "pollinations_image_free": FREE,
    "pollinations_image_vip": FREE,
    "groq": FREE,
    "gemini": FREE,
    "cloudflare": FREE,
    "cloudflare_image": FREE,
    "openrouter": FREE,
}

IMAGE_CHARGE = 2_500
FREE_IMAGE_EDIT_MAX_SIDE = 512

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
    # Cloudflare Workers AI — текстовые модели Free-плана
    "@cf/openai/gpt-oss-120b": {
        "params": "120B (MoE, ~5B активных)",
        "context": 128_000,
        "max_output": 32_768,
        "license": "Apache 2.0",
        "reasoning": True,
        "price": "бесплатно (квота Workers AI)",
        "link": "https://developers.cloudflare.com/workers-ai/models/",
    },
    "@cf/openai/gpt-oss-20b": {
        "params": "20B (MoE, ~3.6B активных)",
        "context": 128_000,
        "max_output": 32_768,
        "license": "Apache 2.0",
        "reasoning": True,
        "price": "бесплатно (квота Workers AI)",
        "link": "https://developers.cloudflare.com/workers-ai/models/",
    },
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast": {
        "params": "70B (FP8)",
        "context": 131_072,
        "max_output": 8_192,
        "license": "Llama 3.3 Community",
        "price": "бесплатно (квота Workers AI)",
        "link": "https://developers.cloudflare.com/workers-ai/models/",
    },
    "@cf/meta/llama-4-scout-17b-16e-instruct": {
        "params": "17B (16 экспертов)",
        "context": 131_072,
        "max_output": 8_192,
        "license": "Llama 4 Community",
        "vision": True,
        "price": "бесплатно (квота Workers AI)",
        "link": "https://developers.cloudflare.com/workers-ai/models/",
    },
    "@cf/qwen/qwen3.8-27b": {
        "params": "27B",
        "context": 131_072,
        "max_output": 16_384,
        "license": "Apache 2.0",
        "reasoning": True,
        "price": "бесплатно (квота Workers AI)",
        "link": "https://developers.cloudflare.com/workers-ai/models/",
    },
    "@cf/qwen/qwen2.5-coder-32b-instruct": {
        "params": "32B",
        "context": 32_768,
        "max_output": 8_192,
        "license": "Apache 2.0",
        "price": "бесплатно (квота Workers AI)",
        "link": "https://developers.cloudflare.com/workers-ai/models/",
    },
    "@cf/google/gemma-4-26b-a4b-it": {
        "params": "26B (MoE, ~4B активных)",
        "context": 128_000,
        "max_output": 16_384,
        "license": "Gemma Terms of Use",
        "reasoning": True,
        "price": "бесплатно (квота Workers AI)",
        "link": "https://developers.cloudflare.com/workers-ai/models/",
    },
    "@cf/mistralai/mistral-small-3.1-24b-instruct": {
        "params": "24B",
        "context": 128_000,
        "max_output": 8_192,
        "license": "Apache 2.0",
        "price": "бесплатно (квота Workers AI)",
        "link": "https://developers.cloudflare.com/workers-ai/models/",
    },
    "@cf/ibm-granite/granite-4.0-h-micro": {
        "params": "micro",
        "context": 128_000,
        "max_output": 8_192,
        "license": "Apache 2.0",
        "price": "бесплатно (квота Workers AI)",
        "link": "https://developers.cloudflare.com/workers-ai/models/",
    },
    "@cf/aisingapore/gemma-sea-lion-v4-27b-it": {
        "params": "27B",
        "context": 128_000,
        "max_output": 8_192,
        "license": "Gemma Terms of Use",
        "price": "бесплатно (квота Workers AI)",
        "link": "https://developers.cloudflare.com/workers-ai/models/",
    },
    "@cf/mistral/mistral-7b-instruct-v0.2-lora": {
        "params": "7B (LoRA)",
        "context": 32_768,
        "max_output": 8_192,
        "license": "Apache 2.0",
        "price": "бесплатно (квота Workers AI)",
        "link": "https://developers.cloudflare.com/workers-ai/models/",
    },
    "@cf/daily-co/phonellm-mini-alpha-2": {
        "params": "mini",
        "context": 8_192,
        "max_output": 4_096,
        "license": "см. условия провайдера",
        "price": "бесплатно (квота Workers AI)",
        "link": "https://developers.cloudflare.com/workers-ai/models/",
    },
    # Cloudflare Workers AI — изображения
    "@cf/black-forest-labs/flux-1-schnell": {
        "params": "12B",
        "license": "Apache 2.0",
        "price": "бесплатно (квота Workers AI)",
        "link": "https://developers.cloudflare.com/workers-ai/models/flux-1-schnell/",
    },
    "@cf/black-forest-labs/flux-2-klein-4b": {
        "params": "4B",
        "license": "FLUX.2 [klein] (BFL)",
        "price": "бесплатно (квота Workers AI)",
        "link": "https://developers.cloudflare.com/workers-ai/models/flux-2-klein-4b/",
    },
    "@cf/black-forest-labs/flux-2-klein-9b": {
        "params": "9B",
        "license": "FLUX.2 [klein] (BFL)",
        "price": "бесплатно (квота Workers AI)",
        "link": "https://developers.cloudflare.com/workers-ai/models/flux-2-klein-9b/",
    },
    "@cf/black-forest-labs/flux-2-dev": {
        "params": "FLUX.2 [dev]",
        "license": "FLUX.2 [dev] Non-Commercial",
        "price": "бесплатно (квота Workers AI)",
        "link": "https://developers.cloudflare.com/workers-ai/models/flux-2-dev/",
    },
}

# Анонимный Pollinations теперь отдаёт только одну модель.
POLLINATIONS_FREE_FALLBACK = ["openai-fast"]

# Запасной VIP-каталог на случай, если gen.pollinations.ai не ответил.
POLLINATIONS_VIP_FALLBACK = [
    "openai/gpt-5.4-nano",
    "openai/gpt-6.1-terra",
    "minimax/minimax-m3.1-flash",
    "moonshotai/kimi-k3.1-pro",
    "deepseek/deepseek-v4.1-flash-1017",
    "deepseek/deepseek-v4.1-flash",
    "meta/muse-spark-open",
    "openai/gpt-6.1-sol",
    "openai/gpt-6-luna",
    "openai/gpt-6-sol",
    "z-ai/glm-5.3-flash",
    "nvidia/nemotron-3.5-lightning",
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

# Cloudflare Workers AI: только модели, проверенные на Free-плане.
# (id, label, vision, reasoning, tools)
CLOUDFLARE_TEXT_FALLBACK: list[tuple[str, str, bool, bool, bool]] = [
    ("@cf/openai/gpt-oss-120b", "GPT-OSS 120B", False, True, True),
    ("@cf/openai/gpt-oss-20b", "GPT-OSS 20B", False, True, True),
    (
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "Llama 3.3 70B",
        False,
        False,
        True,
    ),
    (
        "@cf/meta/llama-4-scout-17b-16e-instruct",
        "Llama 4 Scout 17B",
        True,
        False,
        True,
    ),
    ("@cf/qwen/qwen3.8-27b", "Qwen3.8 27B", False, True, True),
    (
        "@cf/qwen/qwen2.5-coder-32b-instruct",
        "Qwen2.5 Coder 32B",
        False,
        False,
        True,
    ),
    ("@cf/google/gemma-4-26b-a4b-it", "Gemma 4 26B A4B", False, True, True),
    (
        "@cf/mistralai/mistral-small-3.1-24b-instruct",
        "Mistral Small 3.1 24B",
        False,
        False,
        True,
    ),
    (
        "@cf/ibm-granite/granite-4.0-h-micro",
        "Granite 4.0 H Micro",
        False,
        False,
        False,
    ),
    (
        "@cf/aisingapore/gemma-sea-lion-v4-27b-it",
        "Gemma Sea Lion v4 27B",
        False,
        False,
        False,
    ),
    (
        "@cf/mistral/mistral-7b-instruct-v0.2-lora",
        "Mistral 7B v0.2 LoRA",
        False,
        False,
        False,
    ),
    (
        "@cf/daily-co/phonellm-mini-alpha-2",
        "PhoneLLM Mini Alpha 2",
        False,
        False,
        False,
    ),
]

# Cloudflare Workers AI: изображения. (id, label, edits)
CLOUDFLARE_IMAGE_FALLBACK: list[tuple[str, str, bool]] = [
    ("@cf/black-forest-labs/flux-1-schnell", "FLUX.1 Schnell", False),
    ("@cf/black-forest-labs/flux-2-klein-4b", "FLUX.2 Klein 4B", True),
    ("@cf/black-forest-labs/flux-2-klein-9b", "FLUX.2 Klein 9B", True),
    ("@cf/black-forest-labs/flux-2-dev", "FLUX.2 Dev", True),
]

# OpenRouter: живой каталог подтягивается с фильтром :free. Запасной список
# намеренно пуст — id бесплатных моделей часто меняются, а неверный id даёт
# ошибку. Если каталог недоступен, семейство просто не показывается.
OPENROUTER_FALLBACK: list[str] = []

# Анонимный image.pollinations.ai отдаёт ровно одну модель.
POLLINATIONS_IMAGE_FREE_FALLBACK = ["sana"]

# Запасной VIP-каталог изображений: (id, title, edits).
POLLINATIONS_IMAGE_VIP_FALLBACK: list[tuple[str, str, bool]] = [
    ("tongyi-mai/z-image-turbo", "Z-Image Turbo", False),
    ("microsoft/mai-image-2.7", "MAI Image 2.7", True),
    ("microsoft/mai-image-2.7-flash", "MAI Image 2.7 Flash", True),
    ("black-forest-labs/flux.1.1-pro", "FLUX 1.1 Pro", False),
    ("microsoft/mai-image-2.6", "MAI Image 2.6", True),
    ("microsoft/mai-image-2.6-flash", "MAI Image 2.6 Flash", True),
    ("black-forest-labs/flux.2-flex", "FLUX.2 Flex", True),
    ("black-forest-labs/flux.2-klein-4b", "FLUX.2 Klein 4B", True),
    ("black-forest-labs/flux.1-kontext-pro", "FLUX.1 Kontext Pro", True),
    ("black-forest-labs/flux.1-schnell", "FLUX.1 Schnell", False),
    (
        "community/tomdacatto/grok-imagine-2-quest",
        "Grok Imagine 2 (Quest access)",
        False,
    ),
    (
        "community/sharktide/inferenceport-ai-lightning-image-plus",
        "InferencePort AI Lightning Image Plus",
        True,
    ),
    (
        "community/vendouple/animagine",
        "Animagine (read description)",
        False,
    ),
    (
        "community/sharktide/inferenceport-ai-lightning-image-turbo",
        "Lightning Image Turbo",
        True,
    ),
    (
        "community/tomdacatto/nano-banana-pro",
        "Nano banana pro",
        False,
    ),
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
cloudflare_client: DeepSeekClient | None = None
openrouter_client: DeepSeekClient | None = None

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
    global cloudflare_client, openrouter_client

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
        supports_reasoning_off=False,
    )
    pollinations_vip_client = DeepSeekClient(
        api_key=POLLINATIONS_API_KEY,
        base_url=POLLINATIONS_GEN_BASE_URL,
        model="openai/gpt-5.4-nano",
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

    # Cloudflare Workers AI: OpenAI-совместимый эндпоинт /ai/v1.
    # reasoning_effort: none не поддерживается (400), поэтому off не отправляем.
    cloudflare_client = DeepSeekClient(
        api_key=CF_API_TOKEN,
        base_url=CF_CHAT_BASE_URL,
        model="@cf/openai/gpt-oss-120b",
        http_client=client,
        timeout=ai_timeout,
        label="Cloudflare",
        supports_reasoning_off=False,
    )
    # OpenRouter: бесплатные :free-модели. Схема размышлений — объект reasoning.
    openrouter_client = DeepSeekClient(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model="openai/gpt-oss-20b:free",
        http_client=client,
        timeout=ai_timeout,
        label="OpenRouter",
        supports_reasoning_off=False,
        reasoning_style="openrouter",
    )


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
    kind: str = "text",
    edits: bool = False,
    can_disable_reasoning: bool | None = None,
    supported_endpoints: list[str] | None = None,
) -> dict:
    provider = provider or family
    info = _info_for(model, vision=vision, reasoning=reasoning)
    if can_disable_reasoning is None:
        # reasoning_effort="none" реально поддерживают шлюз DeepSeek и
        # Pollinations VIP. У Groq и Gemini минимум — low, у Free — нет.
        can_disable_reasoning = bool(reasoning) and provider in (
            "gateway",
            "pollinations_vip",
        )
    spec = {
        "key": f"{family}@{str(model).lstrip('@')}",
        "section": section,
        "family": family,
        "family_label": FAMILY_LABELS.get(family, family),
        "provider": provider,
        "provider_label": PROVIDER_LABELS.get(provider, provider),
        "company": COMPANY_BY_PROVIDER.get(provider, provider),
        "model": model,
        "label": label or model,
        "kind": kind,
        "vision": bool(info.get("vision") or vision),
        "reasoning": bool(info.get("reasoning") or reasoning),
        "can_disable_reasoning": bool(can_disable_reasoning),
        "edits": bool(edits),
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
    if supported_endpoints:
        spec["supported_endpoints"] = list(supported_endpoints)
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
        make_spec(FREE, "pollinations_free", model)
        for model in POLLINATIONS_FREE_FALLBACK
    ]
    catalog["pollinations_vip"] = [
        make_spec(FREE, "pollinations_vip", model)
        for model in POLLINATIONS_VIP_FALLBACK
    ]
    catalog["pollinations_image_free"] = [
        make_spec(
            FREE,
            "pollinations_image_free",
            model,
            provider="pollinations_image_free",
            kind="image",
            edits=True,
            tools=False,
        )
        for model in POLLINATIONS_IMAGE_FREE_FALLBACK
    ]
    catalog["pollinations_image_vip"] = [
        make_spec(
            FREE,
            "pollinations_image_vip",
            model,
            label=title,
            provider="pollinations_image_vip",
            kind="image",
            edits=edits,
            tools=False,
        )
        for model, title, edits in POLLINATIONS_IMAGE_VIP_FALLBACK
    ]
    catalog["groq"] = [
        make_spec(FREE, "groq", model, vision=(model == GROQ_VISION_MODEL))
        for model in GROQ_FALLBACK
    ]
    catalog["gemini"] = [
        make_spec(FREE, "gemini", model, vision=True) for model in GEMINI_FALLBACK
    ]
    if CF_ACCOUNT_ID and CF_API_TOKEN:
        catalog["cloudflare"] = [
            make_spec(
                FREE,
                "cloudflare",
                model,
                label=label,
                vision=vision,
                reasoning=reasoning,
                tools=tools,
                can_disable_reasoning=False,
            )
            for model, label, vision, reasoning, tools in CLOUDFLARE_TEXT_FALLBACK
        ]
        catalog["cloudflare_image"] = [
            make_spec(
                FREE,
                "cloudflare_image",
                model,
                label=label,
                provider="cloudflare_image",
                kind="image",
                edits=edits,
                tools=False,
                can_disable_reasoning=False,
            )
            for model, label, edits in CLOUDFLARE_IMAGE_FALLBACK
        ]
    if OPENROUTER_API_KEY and OPENROUTER_FALLBACK:
        catalog["openrouter"] = [
            make_spec(
                FREE,
                "openrouter",
                model,
                tools=False,
                can_disable_reasoning=False,
            )
            for model in OPENROUTER_FALLBACK
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


def _parse_pollinations_gen(data: Any, family: str) -> list[dict]:
    """Новый каталог gen.pollinations.ai: id, title, modalities, capabilities."""
    specs: list[dict] = []
    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return specs
    for item in items:
        if not isinstance(item, dict):
            continue
        model = str(item.get("id") or "").strip()
        if not model:
            continue
        if item.get("category") not in (None, "text"):
            continue
        endpoints = item.get("supported_endpoints") or []
        if endpoints and "/v1/chat/completions" not in endpoints:
            continue
        output = item.get("output_modalities") or []
        if output and "text" not in [str(x).lower() for x in output]:
            continue
        if item.get("paid_only"):
            continue
        lowered = model.lower()
        if any(
            bad in lowered
            for bad in ("audio", "embedding", "embed", "tts", "whisper", "sdxl", "flux")
        ):
            continue
        inputs = [str(x).lower() for x in (item.get("input_modalities") or [])]
        caps = [str(x).lower() for x in (item.get("capabilities") or [])]
        supported = [str(x).lower() for x in (item.get("supported_parameters") or [])]
        vision = "image" in inputs
        reasoning = bool(item.get("reasoning")) or "reasoning" in caps
        tools = "tool_calling" in caps or "tools" in caps
        can_disable = "reasoning_effort" in supported
        spec = make_spec(
            FREE,
            family,
            model,
            label=str(item.get("title") or model),
            vision=vision,
            reasoning=reasoning,
            tools=tools,
            description=str(item.get("description") or ""),
            can_disable_reasoning=can_disable if reasoning else False,
            supported_endpoints=[str(x) for x in endpoints],
        )
        context = item.get("context_length")
        if context:
            spec["context"] = int(context)
        pricing = item.get("pricing") or {}
        if pricing:
            spec["price"] = "бесплатно (VIP)"
        specs.append(spec)
    return specs


def _parse_pollinations_image(data: Any, family: str) -> list[dict]:
    """Каталог изображений gen.pollinations.ai: category=image и edits."""
    specs: list[dict] = []
    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return specs
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("category") != "image":
            continue
        if item.get("paid_only"):
            continue
        model = str(item.get("id") or "").strip()
        if not model:
            continue
        endpoints = [str(x) for x in (item.get("supported_endpoints") or [])]
        specs.append(
            make_spec(
                FREE,
                family,
                model,
                label=str(item.get("title") or model),
                provider=family,
                kind="image",
                edits="/v1/images/edits" in endpoints,
                tools=False,
                description=str(item.get("description") or ""),
                supported_endpoints=endpoints,
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


def _price_is_zero(value: Any) -> bool:
    try:
        return float(str(value or 0)) == 0.0
    except Exception:
        return False


def _parse_openrouter(data: Any) -> list[dict]:
    """Каталог OpenRouter: только бесплатные (:free / нулевые цены) модели."""
    specs: list[dict] = []
    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return specs
    for item in items:
        if not isinstance(item, dict):
            continue
        model = str(item.get("id") or "").strip()
        if not model:
            continue
        pricing = item.get("pricing") or {}
        free_id = model.lower().endswith(":free")
        free_price = _price_is_zero(pricing.get("prompt")) and _price_is_zero(
            pricing.get("completion")
        )
        if not (free_id or free_price):
            continue
        if free_id and pricing and not free_price:
            # Бывает, что :free помечен, но цены уже не нулевые. Верим ценам.
            continue
        architecture = item.get("architecture") or {}
        inputs = [str(x).lower() for x in (architecture.get("input_modalities") or [])]
        outputs = [str(x).lower() for x in (architecture.get("output_modalities") or [])]
        if outputs and "text" not in outputs:
            continue
        if "image" in outputs:
            continue
        params = [
            str(x).lower() for x in (item.get("supported_parameters") or [])
        ]
        vision = "image" in inputs
        reasoning = bool(
            {"reasoning", "include_reasoning", "reasoning_effort"} & set(params)
        )
        tools = bool({"tools", "tool_choice"} & set(params))
        spec = make_spec(
            FREE,
            "openrouter",
            model,
            label=str(item.get("name") or model),
            vision=vision,
            reasoning=reasoning,
            tools=tools,
            description=str(item.get("description") or "").strip(),
            can_disable_reasoning=False,
        )
        context = item.get("context_length") or (item.get("top_provider") or {}).get(
            "context_length"
        )
        if context:
            spec["context"] = context
        max_out = (item.get("top_provider") or {}).get("max_completion_tokens")
        if max_out:
            spec["max_output"] = max_out
        spec["price"] = "бесплатно (:free)"
        spec["license"] = "см. условия модели"
        spec["link"] = f"https://openrouter.ai/{model}"
        specs.append(spec)
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
    or_headers = {}
    if OPENROUTER_API_KEY:
        or_headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"
        or_headers["X-Title"] = "Mox Studio"

    (
        free_data,
        vip_data,
        groq_data,
        gemini_data,
        image_free_data,
        openrouter_data,
    ) = await asyncio.gather(
        _get_json(POLLINATIONS_MODELS_URL),
        _get_json(POLLINATIONS_VIP_MODELS_URL, headers=poll_headers),
        _get_json(
            f"{GROQ_BASE_URL}/models",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"} if GROQ_API_KEY else {},
        ),
        _get_json(
            f"{GEMINI_BASE_URL}/models?key={GEMINI_API_KEY}"
            if GEMINI_API_KEY
            else f"{GEMINI_BASE_URL}/models"
        ),
        _get_json("https://image.pollinations.ai/models"),
        _get_json(f"{OPENROUTER_BASE_URL}/models", headers=or_headers),
        return_exceptions=True,
    )

    sources: list[str] = []
    if isinstance(free_data, (list, dict)):
        parsed = _dedupe(_parse_pollinations(free_data, "pollinations_free"))
        if parsed:
            catalog["pollinations_free"] = parsed
            sources.append("pollinations-free")
    if isinstance(vip_data, (list, dict)):
        parsed = _dedupe(_parse_pollinations_gen(vip_data, "pollinations_vip"))
        if parsed:
            catalog["pollinations_vip"] = parsed
            sources.append("pollinations-vip")
        images = _dedupe(_parse_pollinations_image(vip_data, "pollinations_image_vip"))
        if images:
            catalog["pollinations_image_vip"] = images
            sources.append("pollinations-image-vip")
    if isinstance(image_free_data, list):
        free_images = [
            make_spec(
                FREE,
                "pollinations_image_free",
                str(model),
                provider="pollinations_image_free",
                kind="image",
                edits=True,
                tools=False,
            )
            for model in image_free_data
            if isinstance(model, str) and model.strip()
        ]
        if free_images:
            catalog["pollinations_image_free"] = free_images
            sources.append("pollinations-image-free")
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
    if OPENROUTER_API_KEY and isinstance(openrouter_data, (list, dict)):
        parsed = _dedupe(_parse_openrouter(openrouter_data))
        if parsed:
            catalog["openrouter"] = parsed
            sources.append("openrouter")

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
    effort: str = "off",
    on_delta=None,
) -> ChatResult:
    client = (
        pollinations_vip_client
        if spec.get("provider") == "pollinations_vip"
        else pollinations_free_client
    )
    if client is None or not client.configured:
        raise DeepSeekError("Pollinations не настроен")

    # Без автоподмены: reasoning-модель получает выбранный пользователем
    # уровень. Если модель не умеет отключать размышления, «Не думать»
    # превращается в минимальное размышление, а не в тишину.
    effective = effort
    if not spec.get("reasoning"):
        effective = "off"
    elif not spec.get("can_disable_reasoning") and effective == "off":
        effective = "medium"

    async def _ask(with_tools: bool):
        return await client.chat(
            messages,
            tools=tools if with_tools else None,
            tool_choice=tool_choice if with_tools else None,
            max_tokens=max_tokens,
            effort=effective,
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
    effort: str = "off",
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

    # Gemini 3 по умолчанию думает очень долго. В обычном режиме просим
    # минимальное размышление, иначе ответ легко уходит за 60-80 секунд.
    thinking_variants: list[dict] = []
    model_name = str(spec.get("model") or "")
    if model_name.startswith("gemini-2.5"):
        if effort == "off":
            thinking_variants = [{"thinkingBudget": 0}]
        elif effort == "medium":
            thinking_variants = [{"thinkingBudget": 1024}]
        else:
            thinking_variants = [{"thinkingBudget": -1}]
    elif model_name.startswith("gemini-3"):
        if effort in ("off", "medium"):
            thinking_variants = [{"thinkingLevel": "low"}, {"thinkingBudget": 0}]
        else:
            thinking_variants = [{"thinkingLevel": "high"}]

    async def _post(thinking: dict | None):
        body = dict(payload)
        config = dict(payload["generationConfig"])
        if thinking:
            config["thinkingConfig"] = thinking
        body["generationConfig"] = config
        response = await http_client.post(
            f"{GEMINI_BASE_URL}/models/{model_name}:generateContent",
            headers={"x-goog-api-key": GEMINI_API_KEY},
            json=body,
            timeout=AI_TIMEOUT,
        )
        return response

    async def _do():
        variants = list(thinking_variants) + [None]
        response = None
        for thinking in variants:
            response = await _post(thinking)
            if response.status_code < 400:
                break
            text = (response.text or "").lower()
            if response.status_code != 400 or "thinking" not in text:
                break
        assert response is not None
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
            model=model_name,
            finish_reason=candidates[0].get("finishReason") or "",
            **_gemini_usage(data.get("usageMetadata") or {}),
        )

    return await _retry(_do, retries=2)


# ---------------------------------------------------------------------------
# Фото и голос
# ---------------------------------------------------------------------------

POLLINATIONS_IMAGE_URL = "https://image.pollinations.ai/prompt/"


def _image_ext(mime: str) -> str:
    return {
        "image/png": "png",
        "image/webp": "webp",
        "image/gif": "gif",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
    }.get((mime or "").lower(), "jpg")


def _detect_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"GIF8"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def shrink_image(
    image_bytes: bytes, *, max_side: int = FREE_IMAGE_EDIT_MAX_SIDE
) -> tuple[bytes, str]:
    """Сжимает исходник для Free-редактирования, чтобы он влез в URL."""
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise DeepSeekError(f"Не удалось открыть изображение: {e}") from e
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side), Image.LANCZOS)
    for side, quality in (
        (max_side, 70),
        (max_side, 55),
        (max(256, max_side * 3 // 4), 55),
        (max(192, max_side // 2), 50),
    ):
        copy = image.copy()
        copy.thumbnail((side, side), Image.LANCZOS)
        buffer = io.BytesIO()
        copy.save(buffer, format="JPEG", quality=quality, optimize=True)
        data = buffer.getvalue()
        if len(data) <= 24_000:
            return data, "image/jpeg"
    return data, "image/jpeg"


async def _download_image_url(url: str) -> tuple[bytes, str]:
    if http_client is None:
        raise DeepSeekError("HTTP-клиент не настроен")
    response = await http_client.get(url, timeout=AI_TIMEOUT)
    if response.status_code >= 400:
        raise DeepSeekError(
            f"Картинка HTTP {response.status_code}: {response.text[:200]}",
            status_code=response.status_code,
        )
    mime = (response.headers.get("content-type") or "").split(";")[0].strip()
    data = response.content
    if not data:
        raise DeepSeekError("Пустой ответ с изображением")
    return data, mime or _detect_mime(data)


def _vip_image_payload(spec: dict) -> tuple[str, dict[str, str]]:
    return (
        f"{POLLINATIONS_GEN_BASE_URL}/images/generations",
        {
            "Authorization": f"Bearer {POLLINATIONS_API_KEY}",
            "Content-Type": "application/json",
        },
    )


def _image_from_vip_json(data: dict) -> tuple[str, str]:
    item = ((data or {}).get("data") or [{}])[0]
    b64 = item.get("b64_json") or ""
    if b64:
        raw = base64.b64decode(b64)
        mime = item.get("media_type") or _detect_mime(raw)
        return raw, mime
    url = item.get("url") or ""
    if url:
        return url, ""
    raise DeepSeekError("VIP-картинка: пустой ответ")


def _ensure_cloudflare() -> None:
    if not CF_ACCOUNT_ID:
        raise DeepSeekError("Cloudflare Workers AI: не задан CF_ACCOUNT_ID")
    if not CF_API_TOKEN:
        raise DeepSeekError("Cloudflare Workers AI: не задан CF_API_TOKEN")


def _cloudflare_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {CF_API_TOKEN}"}


def _image_from_cloudflare(response, model: str) -> tuple[bytes, str]:
    if response.status_code >= 400:
        raise DeepSeekError(
            f"Cloudflare-картинка HTTP {response.status_code}: "
            f"{response.text[:300]}",
            status_code=response.status_code,
        )
    try:
        data = response.json()
    except Exception as e:
        raise DeepSeekError(f"Cloudflare-картинка: ответ не JSON: {e}") from e
    result = data.get("result") if isinstance(data, dict) else None
    b64 = ""
    if isinstance(result, dict):
        b64 = result.get("image") or ""
    if not b64 and isinstance(data, dict):
        b64 = data.get("image") or ""
    if not b64:
        raise DeepSeekError(f"Cloudflare-картинка: пустой ответ ({str(data)[:200]})")
    try:
        raw = base64.b64decode(b64)
    except Exception as e:
        raise DeepSeekError(f"Cloudflare-картинка: битый base64: {e}") from e
    if not raw:
        raise DeepSeekError("Cloudflare-картинка: пустое изображение")
    logging.info(f"Cloudflare image {model}: {len(raw)} байт")
    return raw, _detect_mime(raw)


async def generate_image(spec: dict, prompt: str) -> tuple[bytes, str]:
    """Генерация изображения выбранной моделью. Без автоподмены."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise DeepSeekError("Пустой промпт")
    provider = spec.get("provider")
    model = spec.get("model") or ""

    if provider == "pollinations_image_free":
        if http_client is None:
            raise DeepSeekError("HTTP-клиент не настроен")
        params = {
            "model": model,
            "width": 1024,
            "height": 1024,
            "nologo": "true",
            "seed": str(int(time.time()) % 1_000_000),
        }
        response = await http_client.get(
            POLLINATIONS_IMAGE_URL + quote(prompt, safe=""),
            params=params,
            timeout=AI_TIMEOUT,
        )
        if response.status_code >= 400:
            raise DeepSeekError(
                f"Free-картинка HTTP {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )
        data = response.content
        if not data:
            raise DeepSeekError("Free-картинка: пустой ответ")
        return data, (response.headers.get("content-type") or "").split(";")[0].strip() or _detect_mime(data)

    if provider == "cloudflare_image":
        _ensure_cloudflare()
        if http_client is None:
            raise DeepSeekError("HTTP-клиент не настроен")
        url = f"{CF_BASE_URL}/ai/run/{model}"
        if model == "@cf/black-forest-labs/flux-1-schnell":
            response = await http_client.post(
                url,
                headers=_cloudflare_headers(),
                json={"prompt": prompt, "steps": 4},
                timeout=AI_TIMEOUT,
            )
        else:
            response = await http_client.post(
                url,
                headers=_cloudflare_headers(),
                data={"prompt": prompt, "steps": "4"},
                timeout=AI_TIMEOUT,
            )
        return _image_from_cloudflare(response, model)

    if provider != "pollinations_image_vip":
        raise DeepSeekError("Эта модель не умеет генерировать изображения")
    if not POLLINATIONS_API_KEY:
        raise DeepSeekError("Pollinations VIP: ключ POLLINATIONS_API не задан")
    if http_client is None:
        raise DeepSeekError("HTTP-клиент не настроен")

    url, headers = _vip_image_payload(spec)
    response = await http_client.post(
        url,
        headers=headers,
        json={"model": model, "prompt": prompt, "size": "1024x1024", "n": 1},
        timeout=AI_TIMEOUT,
    )
    if response.status_code >= 400:
        raise DeepSeekError(
            f"VIP-картинка HTTP {response.status_code}: {response.text[:300]}",
            status_code=response.status_code,
        )
    raw_or_url, mime = _image_from_vip_json(response.json())
    if mime:
        return raw_or_url, mime
    return await _download_image_url(raw_or_url)


async def edit_image(
    spec: dict, image_bytes: bytes, mime: str, prompt: str
) -> tuple[bytes, str]:
    """Редактирование фото выбранной моделью. Без автоподмены."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise DeepSeekError("Пустой промпт для редактирования")
    provider = spec.get("provider")
    model = spec.get("model") or ""
    if not spec.get("edits"):
        raise DeepSeekError(
            f"Модель {spec.get('label') or model} не умеет редактировать фото"
        )
    if http_client is None:
        raise DeepSeekError("HTTP-клиент не настроен")

    if provider == "pollinations_image_free":
        small, small_mime = shrink_image(image_bytes)
        b64 = base64.b64encode(small).decode()
        params = {
            "model": model,
            "image": f"data:{small_mime};base64,{b64}",
            "nologo": "true",
            "seed": str(int(time.time()) % 1_000_000),
        }
        encoded_prompt = quote(prompt, safe="")
        response = await http_client.get(
            POLLINATIONS_IMAGE_URL + encoded_prompt,
            params=params,
            timeout=AI_TIMEOUT,
        )
        if response.status_code >= 400:
            raise DeepSeekError(
                f"Free-редактирование HTTP {response.status_code}: "
                f"{response.text[:200]}",
                status_code=response.status_code,
            )
        data = response.content
        if not data:
            raise DeepSeekError("Free-редактирование: пустой ответ")
        return data, (response.headers.get("content-type") or "").split(";")[0].strip() or _detect_mime(data)

    if provider == "cloudflare_image":
        _ensure_cloudflare()
        # Free-редактирование: исходник ужимаем до 512 px. Если Pillow
        # недоступен или картинка не открылась, отправляем оригинал.
        small, small_mime = image_bytes, mime
        try:
            small, small_mime = shrink_image(image_bytes)
        except Exception as e:
            logging.warning(f"Cloudflare edit: shrink skipped ({e})")
        ext = _image_ext(small_mime)
        response = await http_client.post(
            f"{CF_BASE_URL}/ai/run/{model}",
            headers=_cloudflare_headers(),
            data={"prompt": prompt, "steps": "4"},
            files={
                "input_image_0": (
                    f"photo.{ext}",
                    small,
                    small_mime or "image/jpeg",
                )
            },
            timeout=AI_TIMEOUT,
        )
        return _image_from_cloudflare(response, model)

    if not POLLINATIONS_API_KEY:
        raise DeepSeekError("Pollinations VIP: ключ POLLINATIONS_API не задан")
    ext = _image_ext(mime)
    response = await http_client.post(
        f"{POLLINATIONS_GEN_BASE_URL}/images/edits",
        headers={"Authorization": f"Bearer {POLLINATIONS_API_KEY}"},
        data={"model": model, "prompt": prompt},
        files={"image": (f"photo.{ext}", image_bytes, mime or "image/jpeg")},
        timeout=AI_TIMEOUT,
    )
    if response.status_code >= 400:
        raise DeepSeekError(
            f"VIP-редактирование HTTP {response.status_code}: {response.text[:300]}",
            status_code=response.status_code,
        )
    raw_or_url, out_mime = _image_from_vip_json(response.json())
    if out_mime:
        return raw_or_url, out_mime
    return await _download_image_url(raw_or_url)


VISION_DEFAULT_PROMPT = VISION_PROMPT


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
    if spec.get("kind") == "image":
        parts = ["изображения"]
        if spec.get("edits"):
            parts.append("редактирование фото")
        return ", ".join(parts)
    parts = ["текст"]
    if spec.get("vision"):
        parts.append("фото")
    parts.append("голос (Whisper)")
    return ", ".join(parts)


def _yes_no(value: Any) -> str:
    return "да" if value else "нет"


def info_text(spec: dict) -> str:
    is_image = spec.get("kind") == "image"
    family = spec.get("family_label") or spec.get("family") or ""
    section = "Платные" if spec.get("section") == PAID else "Бесплатные"
    title = _escape(spec.get("label") or spec.get("model"))

    main_rows = [
        ("ID", f"<code>{_escape(spec.get('model'))}</code>"),
        ("Провайдер", _escape(spec.get("provider_label") or spec.get("provider"))),
        ("Компания", _escape(spec.get("company") or "н/д")),
        ("Семейство", _escape(family)),
        ("Раздел", section),
    ]
    capability_rows = [
        ("Модальности", _escape(_modalities(spec))),
        (
            "Инструменты",
            "нативные tool-calls" if spec.get("tools") else "нет",
        ),
        ("Reasoning", _yes_no(spec.get("reasoning") and not is_image)),
        (
            "Может не думать",
            _yes_no(spec.get("can_disable_reasoning") and not is_image),
        ),
        (
            "Редактирование фото",
            _yes_no(spec.get("edits")),
        ),
    ]
    if is_image:
        capability_rows.append(("Генерация изображений", "да"))

    parameter_rows = [
        ("Параметры", _escape(_fmt(spec.get("params")))),
        ("Контекст", _escape(_fmt(spec.get("context"), tokens=True))),
        ("Макс. вывод", _escape(_fmt(spec.get("max_output"), tokens=True))),
        (
            "Цена",
            _escape(
                spec.get("price")
                or ("бесплатно" if spec.get("section") == FREE else "н/д")
            ),
        ),
        ("Лицензия", _escape(_fmt(spec.get("license")))),
        ("Скорость", _escape(_fmt(spec.get("speed")))),
        ("Статус", _escape(_fmt(spec.get("status")))),
        ("Дата релиза", _escape(_fmt(spec.get("release")))),
        ("Ссылка", _escape(spec.get("link") or "н/д")),
    ]

    def block(title: str, rows: list[tuple[str, str]]) -> str:
        body = "\n".join(f"• {name}: {value}" for name, value in rows)
        return f"<b>{title}</b>\n{body}"

    parts = [
        f"ℹ️ <b>{title}</b>",
        block("Основное", main_rows),
        block("Возможности", capability_rows),
        block("Параметры", parameter_rows),
    ]
    description = (spec.get("description") or "").strip()
    if description:
        parts.append(f"<i>{_escape(description[:600])}</i>")
    parts.append("Выбери модель кнопкой ниже — контекст чата сохранится.")
    return "\n\n".join(parts)


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
