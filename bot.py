import asyncio
import base64
import html
import inspect
import json
import logging
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote

import httpx
from aiogram import BaseMiddleware, Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction, ChatType
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BotCommand,
    BotCommandScopeAllPrivateChats,
    BotCommandScopeChat,
    BotCommandScopeDefault,
    BufferedInputFile,
    CallbackQuery,
    ChatMemberUpdated,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQuery,
    InlineQueryResultArticle,
    InputRichMessage,
    InputTextMessageContent,
    Message,
)
from openai import AsyncOpenAI

from deepseek_client import ChatResult, DeepSeekClient
from storage import DB_LIMIT_BYTES, MSK, storage

# ============================================================
# НАСТРОЙКИ
# ============================================================

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")

GROQ_API_KEY = os.environ.get("GROQ_API", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CHAT_MODEL = "openai/gpt-oss-120b"
FAST_MODEL = "llama-3.1-8b-instant"
WHISPER_MODEL = "whisper-large-v3-turbo"

GEMINI_API_KEY = os.environ.get("GEMINI_API", "")
# ВАЖНО: для AQ.-ключей в Google Cloud Console у ключа должно стоять
# ограничение "Generative Language API" (не "Gemini API"), иначе будет 401.
# Мы используем нативный REST-эндпоинт, т.к. AQ.-ключи не работают
# на OpenAI-совместимом маршруте /v1beta/openai.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
VISION_MODEL = "gemini-3.5-flash-lite"
# Запасная модель на случай, если Gemini недоступен (например, ключ AQ. отклонён).
GROQ_VISION_MODEL = "qwen/qwen3.8-27b"

TAVILY_API_KEY = os.environ.get("TAVIL_API", "")

# DeepSeek (aikeysforyou) и Qwen (voidluv) — только владелец и выданные пользователи.
AIKEYS_API_KEY = os.environ.get("AIKEYS_API", "")
DEEPSEEK_BASE_URL = "https://api.aikeysforyou.com/v1"
DEEPSEEK_MODEL = "deepseek-v4.1-flash"

QWEN_API_KEY = os.environ.get("QWEN_API", "")
QWEN_BASE_URL = "https://api.voidluv.com/v1"
# Публичная страница баланса ключа voidluv (её можно переопределить через env).
VOIDLUV_BALANCE_URL = os.environ.get(
    "VOIDLUV_BALANCE_URL",
    "https://voidluv.com/u/05dadbd0-082b-4405-bf35-95e5e0785a62",
)

# Реестр моделей: ключ -> подпись, провайдер, реальный model id.
MODEL_REGISTRY: dict[str, dict] = {
    "groq": {"label": "Groq", "provider": "groq", "model": CHAT_MODEL},
    "deepseek": {
        "label": "deepseek-v4.1-flash",
        "provider": "deepseek",
        "model": DEEPSEEK_MODEL,
    },
    "deepseek-pro": {
        "label": "deepseek-v4-pro",
        "provider": "deepseek",
        "model": "deepseek-v4-pro-0813",
    },
    "qwen4-max": {"label": "qwen4-max", "provider": "qwen", "model": "qwen4-max"},
    "qwen3.8-max": {"label": "qwen3.8-max", "provider": "qwen", "model": "qwen3.8-max"},
    "qwen3.8-flash": {
        "label": "qwen3.8-flash",
        "provider": "qwen",
        "model": "qwen3.8-flash",
    },
}
PREMIUM_MODELS = [key for key in MODEL_REGISTRY if key != "groq"]
PROVIDER_LIMITS = {"deepseek": 50, "qwen": 75}
PROVIDER_NAMES = {"deepseek": "DeepSeek", "qwen": "Qwen"}
# Для фото используем модель провайдера, которая умеет смотреть картинки.
PROVIDER_VISION_MODEL = {"deepseek": "deepseek-v4.1-flash", "qwen": "qwen4-max"}

PREMIUM_HISTORY_CHARS_LIMIT = 300000
PREMIUM_MAX_TOKENS = 16000
PREMIUM_MAX_TOKENS_CAP = 32000
DEEPSEEK_PHOTO_LIMIT = 10

OWNER_ID = 6101597477
OWNER_USERNAME = "Diotrex"
ALLOWED_GROUPS: set[int] = {-1002495961136}

HISTORY_LIMIT = 50
HISTORY_STORE_CHARS_LIMIT = 300000
HISTORY_CHARS_LIMIT = 14000
MAX_REPLY_LENGTH = 4000
MAX_RICH_REPLY_LENGTH = 30000
CHAT_MAX_TOKENS = 1200
COOLDOWN_SECONDS = 2
AI_TIMEOUT = 180.0
TYPING_INTERVAL = 4.0
FALLBACK_COOLDOWN = 600
PHOTO_SESSION_DELAY = 2.0
MAX_ALBUM_PHOTOS = DEEPSEEK_PHOTO_LIMIT
ADMIN_PAGE_SIZE = 6
BLOCK_REPLY_COOLDOWN = 5.0
RICH_DRAFT_INTERVAL = 1.1
BALANCE_REFRESH_SECONDS = 3600
BALANCE_ALERT_PERCENT = 20

# Если запускаешь локально и нужен SOCKS-прокси — впиши сюда,
# например "socks5://127.0.0.1:10808". На Render оставь пустым.
PROXY_URL = os.environ.get("PROXY_URL", "")

TRIGGER_ROOTS = ("мох", "mox", "moh")
FALLBACK_TEXT = "Чтобы добавить меня в чат, напишите владельцу @Diotrex."

START_TIME = time.time()

# ============================================================
# ЛОГИ
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("duckduckgo_search").setLevel(logging.WARNING)

# ============================================================
# КЛИЕНТЫ
# ============================================================

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp = Dispatcher()

http_client = httpx.AsyncClient(timeout=AI_TIMEOUT, proxy=PROXY_URL or None)

ai_client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL,
    http_client=http_client,
    max_retries=0,
    timeout=AI_TIMEOUT,
)

deepseek_client = DeepSeekClient(
    api_key=AIKEYS_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model=DEEPSEEK_MODEL,
    http_client=http_client,
    timeout=AI_TIMEOUT,
    label="DeepSeek",
)

qwen_client = DeepSeekClient(
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    model="qwen4-max",
    http_client=http_client,
    timeout=AI_TIMEOUT,
    label="Qwen",
)

PROVIDER_CLIENTS: dict[str, DeepSeekClient] = {
    "deepseek": deepseek_client,
    "qwen": qwen_client,
}

BOT_ID: int | None = None
BOT_USERNAME = ""

_history: dict[int, list[dict]] = {}
_stats: dict[int, int] = {}
_last_request: dict[int, float] = {}
_typing_last: dict[int, float] = {}
_fallback_last: dict[int, float] = {}
_media_cache: dict[tuple[int, int], str] = {}
_locks: dict[int, asyncio.Lock] = {}
_search_cache: dict[str, tuple[float, list[dict]]] = {}
_image_lock = asyncio.Lock()

# Фото-сессии: все подряд идущие фото одного пользователя склеиваются в одну пачку.
_photo_sessions: dict[tuple[int, int], list[Message]] = {}
_photo_tasks: dict[tuple[int, int], asyncio.Task] = {}
_photo_generation: dict[tuple[int, int], int] = {}

# Настройки пользователей и доступы живут в Postgres, здесь — кэш.
_user_settings: dict[int, dict] = {}
_grants: dict[tuple[int, str], dict] = {}
_dm_blacklist: dict[int, dict] = {}
_groups_cache: dict[int, dict] = {}
_limit_notified: set[tuple[int, str, str]] = set()
_block_reply_last: dict[int, float] = {}
_balance: dict[str, dict] = {}
_balance_alerted: set[str] = set()
_model_blocked: dict[str, str] = {}

# Состояние админ-панели
_admin_input: dict[int, dict] = {}
_admin_filter: dict[int, dict] = {}
_db_size_cache = 0

GROUP_TYPES = (ChatType.GROUP, ChatType.SUPERGROUP)

# ============================================================
# РЕТРАЙ
# ============================================================


def _parse_duration(value: str) -> float | None:
    """Парсит '9.7875s', '665ms', '1m30s' в секунды."""
    if not value:
        return None
    value = str(value).strip().lower()
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)?", value)
    if not matches:
        return None
    total = 0.0
    for number, unit in matches:
        amount = float(number)
        if unit == "ms":
            total += amount / 1000.0
        elif unit == "m":
            total += amount * 60.0
        elif unit == "h":
            total += amount * 3600.0
        else:
            total += amount
    return total or None


def retry_after_seconds(error: Exception) -> float | None:
    """Достаёт паузу из Retry-After или текста ошибки Groq."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        for key in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
            raw = headers.get(key)
            if raw:
                wait = _parse_duration(raw)
                if wait:
                    return wait

    match = re.search(
        r"try again in\s+([\d.]+\s*(?:ms|s|m|h)?)",
        str(error),
        flags=re.IGNORECASE,
    )
    if match:
        return _parse_duration(match.group(1))
    return None


def is_retryable_error(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status in (408, 409, 425, 429, 500, 502, 503, 504)

    err = str(error).lower()
    markers = (
        "rate limit",
        "too many requests",
        "temporarily",
        "connect",
        "timed out",
        "timeout",
        "unavailable",
        "high demand",
        "overloaded",
        "service unavailable",
        "internal server error",
    )
    return any(marker in err for marker in markers)


async def with_retry(coro_fn, *args, retries=3, base_wait=2.0, **kwargs):
    last_err = None
    for attempt in range(retries):
        try:
            return await coro_fn(*args, **kwargs)
        except asyncio.TimeoutError as e:
            last_err = e
            wait = base_wait * (attempt + 1)
            logging.warning(f"Timeout, retry {attempt + 1}/{retries} in {wait}s")
            await asyncio.sleep(wait)
        except Exception as e:
            last_err = e
            if is_retryable_error(e):
                wait = base_wait * (attempt + 1)
                # Groq в 429 обычно сам говорит, сколько ждать. Игнорировать
                # это нельзя: 2-4 секунды недостаточно, и все ретраи сгорят.
                retry_after = retry_after_seconds(e)
                if retry_after:
                    wait = max(wait, retry_after + 0.5)
                wait = min(wait, 60.0)
                logging.warning(f"Retry {attempt + 1}/{retries} in {wait:.1f}s: {e}")
                await asyncio.sleep(wait)
                continue
            raise
    raise last_err


def friendly_ai_error(error: Exception) -> str:
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    if status == 429:
        wait = retry_after_seconds(error)
        seconds = int(wait) + 1 if wait else 10
        return f"Слишком много запросов к AI. Подожди примерно {seconds} с и повтори."
    return str(error)


# ============================================================
# HTML
# ============================================================

FORBIDDEN_TAGS = [
    "ul", "ol", "li", "p", "br", "div", "span", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "tr", "td", "th", "thead", "tbody",
    "img", "video", "audio", "iframe", "section", "article",
]


def placeholder_key(i: int) -> str:
    return f"\x00TAG{i}\x00"


def sanitize_html(text: str) -> str:
    if not text:
        return ""
    for tag in FORBIDDEN_TAGS:
        text = re.sub(rf"</?{tag}\b[^>]*>", "", text, flags=re.IGNORECASE)

    allowed_pattern = re.compile(
        r"</?(?:b|i|u|s|code|pre|blockquote|tg-spoiler|a)(?:\s+[^>]*)?>",
        re.IGNORECASE,
    )
    placeholders: dict[str, str] = {}

    def _hide(match):
        key = placeholder_key(len(placeholders))
        placeholders[key] = match.group(0)
        return key

    text = allowed_pattern.sub(_hide, text)
    text = re.sub(r"&(?!(?:amp|lt|gt|quot|#\d+|#x[0-9a-f]+);)", "&amp;", text, flags=re.I)
    text = text.replace("<", "&lt;").replace(">", "&gt;")

    for key, tag in placeholders.items():
        text = text.replace(key, tag)

    return text


def esc(text) -> str:
    return html.escape(str(text), quote=False)


def plain_text(text) -> str:
    """Убирает HTML-теги: callback-ответы Telegram показываются без разметки."""
    return html.unescape(re.sub(r"<[^>]+>", "", str(text or ""))).strip()


# ============================================================
# КОНТЕКСТ
# ============================================================


def _trim_history(history: list[dict], chars_limit: int = HISTORY_STORE_CHARS_LIMIT) -> None:
    while len(history) > HISTORY_LIMIT:
        history.pop(0)
    total = sum(len(str(m.get("content") or "")) for m in history)
    while len(history) > 1 and total > chars_limit:
        total -= len(str(history[0].get("content") or ""))
        history.pop(0)


async def add_to_history(chat_id: int, role: str, content: str) -> None:
    if not content:
        return
    history = _history.setdefault(chat_id, [])
    history.append({"role": role, "content": content})
    _trim_history(history)
    await storage.save_context(chat_id, history)


def get_history(chat_id: int) -> list[dict]:
    return _history.get(chat_id, [])


def history_for_model(chat_id: int, model: str) -> list[dict]:
    limit = HISTORY_CHARS_LIMIT if model == "groq" else PREMIUM_HISTORY_CHARS_LIMIT
    source = list(_history.get(chat_id, []))
    total = sum(len(str(m.get("content") or "")) for m in source)
    while len(source) > 1 and total > limit:
        total -= len(str(source[0].get("content") or ""))
        source.pop(0)
    return source


async def reset_history(chat_id: int) -> None:
    _history.pop(chat_id, None)
    await storage.delete_context(chat_id)


def get_history_size(chat_id: int) -> int:
    return len(_history.get(chat_id, []))


def chat_lock(chat_id: int) -> asyncio.Lock:
    return _locks.setdefault(chat_id, asyncio.Lock())


def clear_media_cache(chat_id: int) -> None:
    for key in list(_media_cache.keys()):
        if key[0] == chat_id:
            _media_cache.pop(key, None)


def cancel_photo_sessions(chat_id: int) -> None:
    for key in list(_photo_sessions.keys()):
        if key[0] == chat_id:
            _photo_sessions.pop(key, None)
            _photo_generation[key] = _photo_generation.get(key, 0) + 1
            task = _photo_tasks.pop(key, None)
            if task and not task.done():
                task.cancel()


async def clear_chat_context(chat_id: int) -> None:
    await reset_history(chat_id)
    clear_media_cache(chat_id)
    cancel_photo_sessions(chat_id)


# ============================================================
# ПРОВЕРКИ
# ============================================================


def is_whitelisted(chat_id: int) -> bool:
    return chat_id in ALLOWED_GROUPS


def is_group(message: Message) -> bool:
    return message.chat.type in GROUP_TYPES


def is_private(message: Message) -> bool:
    return message.chat.type == ChatType.PRIVATE


def is_reply_to_bot(message: Message) -> bool:
    reply = message.reply_to_message
    if not reply or not reply.from_user:
        return False
    return BOT_ID is not None and reply.from_user.id == BOT_ID


def is_on_cooldown(message: Message) -> bool:
    user = message.from_user
    if not user:
        return False
    now = time.time()
    last = _last_request.get(user.id, 0)
    if now - last < COOLDOWN_SECONDS:
        return True
    _last_request[user.id] = now
    return False


def can_send_fallback(chat_id: int) -> bool:
    now = time.time()
    if now - _fallback_last.get(chat_id, 0) < FALLBACK_COOLDOWN:
        return False
    _fallback_last[chat_id] = now
    return True


# ============================================================
# МОДЕЛИ, ДОСТУП И НАСТРОЙКИ
# ============================================================


DEFAULT_SETTINGS = {"model": "groq", "rich": True, "deepthink": "off", "search": "auto"}


def user_settings(user_id: int | None) -> dict:
    if user_id is None:
        return dict(DEFAULT_SETTINGS)
    return _user_settings.setdefault(int(user_id), dict(DEFAULT_SETTINGS))


def provider_of(model_key: str) -> str:
    return (MODEL_REGISTRY.get(model_key) or {}).get("provider", "groq")


def provider_eligible(user_id: int | None, provider: str) -> bool:
    if provider == "groq":
        return True
    if user_id is None:
        return False
    if int(user_id) == OWNER_ID:
        return True
    return (int(user_id), provider) in _grants


def model_available(user_id: int | None, model_key: str) -> bool:
    """Доступна ли модель пользователю (без учёта дневного лимита)."""
    info = MODEL_REGISTRY.get(model_key)
    if not info:
        return False
    provider = info["provider"]
    if provider == "groq":
        return True
    if not provider_eligible(user_id, provider):
        return False
    if model_key in _model_blocked:
        return False
    client = PROVIDER_CLIENTS.get(provider)
    return bool(client and client.configured)


def selected_model(user_id: int | None) -> str:
    """Выбранная модель. Если доступ пропал — Groq."""
    if user_id is None:
        return "groq"
    key = user_settings(user_id).get("model") or "groq"
    if not model_available(user_id, key):
        return "groq"
    return key


def model_label(model_key: str) -> str:
    info = MODEL_REGISTRY.get(model_key)
    return info["label"] if info else str(model_key)


def is_blacklisted(user_id: int | None) -> bool:
    return user_id is not None and int(user_id) in _dm_blacklist


def msk_now() -> datetime:
    return datetime.now(MSK)


def msk_today() -> date:
    return msk_now().date()


def msk_day_start(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=MSK).astimezone(timezone.utc)


def is_complex_task(text: str) -> bool:
    text = text or ""
    lowered = text.lower()
    if len(text) > 220:
        return True
    markers = (
        "реши", "задач", "уравнен", "докажи", "вычисли", "найди", "объясни",
        "почему", "как работает", "код", "напиши программу", "проанализируй",
        "сравни", "план", "эссе", "сочинени", "переведи", "формул",
    )
    if any(m in lowered for m in markers):
        return True
    if re.search(r"[=<>√∑∫]|\d+\s*[+\-*/^]\s*\d+", text):
        return True
    return False


async def notify_limit(user_id: int, provider: str, limit: int) -> None:
    key = (int(user_id), provider, msk_today().isoformat())
    if key in _limit_notified:
        return
    _limit_notified.add(key)
    name = PROVIDER_NAMES.get(provider, provider)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🌿 Перейти на Groq", callback_data="model:set:groq")]
        ]
    )
    try:
        await bot.send_message(
            user_id,
            f"⚠️ Лимит {name} на сегодня исчерпан "
            f"({limit} сообщений). Отвечаю через Groq. "
            "Лимит сбросится в 00:00 МСК.",
            reply_markup=keyboard,
        )
    except Exception:
        pass


async def resolve_model(user_id: int | None, private: bool) -> str:
    """Возвращает модель для запроса и списывает дневной лимит провайдера."""
    if not private or user_id is None:
        return "groq"
    key = selected_model(user_id)
    provider = provider_of(key)
    if provider == "groq":
        return "groq"
    if int(user_id) == OWNER_ID:
        return key
    limit = PROVIDER_LIMITS.get(provider, 0)
    today = msk_today()
    used = await storage.get_usage(int(user_id), provider, today)
    if used >= limit:
        await notify_limit(int(user_id), provider, limit)
        return "groq"
    await storage.inc_usage(int(user_id), provider, today)
    return key


async def set_model_for_user(user_id: int, model_key: str) -> tuple[bool, str]:
    """Переключает модель. Контекст, кэш фото и фотосессии сохраняются."""
    if model_key not in MODEL_REGISTRY:
        return False, "Неизвестная модель."
    if model_key in _model_blocked:
        return False, f"{model_label(model_key)}: {_model_blocked[model_key]}."
    if not model_available(user_id, model_key):
        provider = provider_of(model_key)
        name = PROVIDER_NAMES.get(provider, provider)
        return False, f"{name} тебе не выдан. Напиши владельцу @{OWNER_USERNAME}."
    info = MODEL_REGISTRY[model_key]
    client = PROVIDER_CLIENTS.get(info["provider"])
    if info["provider"] != "groq" and (client is None or not client.configured):
        name = PROVIDER_NAMES.get(info["provider"], info["provider"])
        return False, f"{name} не настроен на сервере."
    settings = user_settings(user_id)
    if settings.get("model") == model_key:
        return True, f"Уже выбрана модель {model_label(model_key)}."
    settings["model"] = model_key
    await storage.save_user_settings(int(user_id), model=model_key)
    note = ""
    provider = info["provider"]
    if provider != "groq" and int(user_id) != OWNER_ID:
        used = await storage.get_usage(int(user_id), provider, msk_today())
        limit = PROVIDER_LIMITS.get(provider, 0)
        note = f"\nОсталось сегодня: {max(0, limit - used)} из {limit}."
    return (
        True,
        f"Модель переключена на {model_label(model_key)}. Контекст сохранён.{note}",
    )


async def save_toggle(user_id: int, field: str, value) -> None:
    settings = user_settings(user_id)
    settings[field] = value
    await storage.save_user_settings(int(user_id), **{field: value})


def user_rich(user_id: int | None) -> bool:
    return bool(user_settings(user_id).get("rich", True))


def user_effort(user_id: int | None) -> str:
    level = user_settings(user_id).get("deepthink", "off")
    return level if level in ("low", "medium", "high") else "off"


def user_search_mode(user_id: int | None) -> str:
    mode = user_settings(user_id).get("search", "auto")
    return mode if mode in ("auto", "always", "off") else "auto"


async def safe_typing(chat_id: int) -> None:
    now = time.time()
    if now - _typing_last.get(chat_id, 0) < TYPING_INTERVAL:
        return
    _typing_last[chat_id] = now
    try:
        await bot.send_chat_action(chat_id, ChatAction.TYPING)
    except Exception:
        pass


# ============================================================
# ИМЕНА И ТРИГГЕРЫ
# ============================================================


def display_user(user) -> str:
    if user is None:
        return "Аноним"
    name = (user.first_name or user.username or f"id{user.id}").strip()
    return f"{name} ({user.id})"


def username_of(user) -> str:
    if user is None:
        return ""
    return user.username or user.full_name or str(user.id)


@dataclass
class Track:
    request_id: int | None
    user_id: int | None
    chat_id: int | None


async def begin_request(
    message: Message,
    *,
    kind: str,
    prompt: str,
    model: str | None,
    is_request: bool = True,
) -> Track:
    user = message.from_user
    request_id = await storage.create_request(
        user_id=user.id if user else None,
        username=username_of(user),
        chat_id=message.chat.id,
        chat_type=str(message.chat.type),
        kind=kind,
        is_request=is_request,
        model=model,
        prompt_preview=prompt or "",
    )
    return Track(
        request_id=request_id,
        user_id=user.id if user else None,
        chat_id=message.chat.id,
    )


async def finish_track(
    track: Track | None,
    *,
    status: str = "ok",
    error: str = "",
    latency_ms: int | None = None,
    model: str | None = None,
    tools: str = "",
) -> None:
    if track is None:
        return
    await storage.finish_request(
        track.request_id,
        status=status,
        error=error,
        latency_ms=latency_ms,
        model=model,
        tools=tools,
    )


async def log_group_event(message: Message, text: str, kind: str = "group_message") -> None:
    user = message.from_user
    await storage.create_request(
        user_id=user.id if user else None,
        username=username_of(user),
        chat_id=message.chat.id,
        chat_type=str(message.chat.type),
        kind=kind,
        is_request=False,
        model=None,
        prompt_preview=text or media_placeholder(message),
        status="logged",
    )


async def record_blocked(message: Message) -> None:
    await log_group_event(message, message.text or message.caption or "", kind="blocked_dm")


def first_word(text: str) -> str:
    m = re.match(r"[\w@]+", (text or "").lstrip())
    return m.group(0) if m else ""


def is_trigger_word(word: str) -> bool:
    w = (word or "").lstrip("@").lower()
    if not w:
        return False
    return any(w.startswith(root) for root in TRIGGER_ROOTS)


def starts_with_trigger(text: str) -> bool:
    return bool(text) and is_trigger_word(first_word(text))


def mentions_bot(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if "@m0xaibot" in lowered:
        return True
    if BOT_USERNAME and f"@{BOT_USERNAME.lower()}" in lowered:
        return True
    return False


def strip_triggers(text: str) -> str:
    t = (text or "").strip()
    word = first_word(t)
    if word and is_trigger_word(word):
        t = t[t.find(word) + len(word):]
    if BOT_USERNAME:
        t = re.sub(rf"@{re.escape(BOT_USERNAME)}\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"@m0xaibot\b", " ", t, flags=re.IGNORECASE)
    return t.strip(" \t\n,.:;!?—-")


def is_addressed(message: Message, text: str | None) -> bool:
    if is_private(message):
        return True
    if is_reply_to_bot(message):
        return True
    if text and (starts_with_trigger(text) or mentions_bot(text)):
        return True
    return False


# ============================================================
# РИСОВАНИЕ: ЯВНЫЕ ТРИГГЕРЫ
# ============================================================

DRAW_VERBS = r"(нарисуй(?:-ка)?|нарисуйте|draw|sketch)"
IMAGE_NOUNS = (
    r"(картинк\w*|изображени\w*|арт\w*|иллюстраци\w*|постер\w*|логотип\w*|"
    r"аватарк\w*|обложк\w*|фотк\w*|фото|мем\w*|персонаж\w*)"
)
DRAW_NEGATIVE = (
    r"(схем\w*|график\w*|таблиц\w*|код\w*|план\w*|диаграмм\w*|"
    r"текст\w*|письм\w*|презентаци\w*|список\w*|инструкци\w*)"
)
DRAW_LEADING_WORDS = re.compile(
    r"^(мне|пожалуйста|плиз|картинку|картинка|изображение|фото|фотку|"
    r"picture|image|a|an|the)\b[\s,:—-]+",
    re.IGNORECASE,
)


def clean_draw_prompt(tail: str) -> str:
    result = (tail or "").strip(" \t\n,.:;!?—-")
    for _ in range(4):
        new = DRAW_LEADING_WORDS.sub("", result, count=1)
        if new == result:
            break
        result = new.strip(" \t\n,.:;!?—-")
    return result


def detect_explicit_draw(text: str) -> str | None:
    if not text:
        return None

    # «нарисуй кота», «draw a cat»
    m = re.search(rf"\b{DRAW_VERBS}\b", text, flags=re.IGNORECASE)
    if m:
        prompt = clean_draw_prompt(text[m.end():])
        if not prompt:
            return None
        # «нарисуй схему/таблицу/код» — это не картинка, пусть решает Groq.
        if re.match(DRAW_NEGATIVE, prompt, flags=re.IGNORECASE):
            return None
        return prompt

    # «сгенерируй картинку кота», «сделай постер с котом»
    m = re.search(
        r"\b(сгенерируй|сгенерируйте|сделай|сделайте|создай|создайте|generate)\b(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        tail = m.group(2)
        noun = re.search(IMAGE_NOUNS, tail, flags=re.IGNORECASE)
        if noun:
            prompt = clean_draw_prompt(tail[noun.end():]) or clean_draw_prompt(tail)
            if not prompt:
                return None
            if re.match(DRAW_NEGATIVE, prompt, flags=re.IGNORECASE):
                return None
            return prompt
    return None


SEARCH_VERBS = (
    "поищи", "поищи-ка", "найди", "загугли", "погугли", "search",
    "проверь в интернете", "посмотри в интернете", "глянь в интернете",
)


def has_search_verb(text: str) -> bool:
    lowered = (text or "").lower()
    return any(v in lowered for v in SEARCH_VERBS)


# ============================================================
# РАЗНОЕ
# ============================================================


def format_uptime() -> str:
    delta = int(time.time() - START_TIME)
    hours = delta // 3600
    minutes = (delta % 3600) // 60
    seconds = delta % 60
    if hours > 0:
        return f"{hours}ч {minutes}м {seconds}с"
    if minutes > 0:
        return f"{minutes}м {seconds}с"
    return f"{seconds}с"


def media_placeholder(message: Message) -> str:
    if message.photo:
        return "[фото]"
    if message.voice:
        return "[голосовое]"
    if message.audio:
        return "[аудио]"
    if message.sticker:
        return "[стикер]"
    if message.animation:
        return "[гиф]"
    if message.video:
        return "[видео]"
    if message.video_note:
        return "[кружок]"
    if message.document:
        return "[документ]"
    if message.location:
        return "[гео]"
    if message.contact:
        return "[контакт]"
    if message.poll:
        return "[опрос]"
    if message.dice:
        return "[кубик]"
    return ""

# ============================================================
# СИСТЕМНЫЙ ПРОМПТ
# ============================================================

SYSTEM_PROMPT_BASE = """Ты — Мудрый Мох, дружелюбный ассистент в Telegram. Отвечай по-русски, по делу, с лёгким юмором. Если пользователь просит написать текст на другом языке (письмо, код, задание) — напиши на нём.

КОНТЕКСТ ЧАТА:
- Ты видишь историю разговора. В группе каждое сообщение пользователя выглядит как «Имя (id): текст». Твои прошлые ответы — обычные сообщения ассистента. Не путай авторов и отвечай тому, кто обратился последним.
- Запись вида «[в ответ на сообщение Имя (id): «...»]» — это сообщение, на которое ответил пользователь. Учитывай его смысл, но НЕ цитируй и не пересказывай в ответе.
- Сообщения с пометкой «[фото]» или «[голосовое]» без описания означают, что в чате было медиа, но его содержимое неизвестно.
- Пометки «(расшифровка голосового, возможны ошибки)» и описания фото сделаны другими моделями — доверяй им, но допускай неточности.

ИНСТРУМЕНТЫ:
- web_search — вызывай, когда для ответа нужны свежие данные из интернета: новости, погода, курсы, цены, версии, события, актуальные факты о людях и компаниях. НЕ вызывай для общих знаний, математики, объяснений, написания текста и кода.
- generate_image — вызывай, когда просят нарисовать или сгенерировать визуальный объект. Код, текст, таблица, схема, график, план — это НЕ картинки, для них инструмент не используй. Если просят и картинку, и объяснение — сначала вызови generate_image, затем ответь на остальное.
- В generate_image передай prompt_en (подробный промпт на английском) и prompt_ru (тот же промпт на русском для подписи).
- После вызова generate_image картинка уже отправлена пользователю. Не описывай её повторно. Если добавить нечего — верни ровно OK.

ПРАВИЛА:
- Не здоровайся в каждом сообщении: приветствие уместно только в начале диалога или если пользователь сам поздоровался.
- После web_search опирайся на найденные данные, а не на свои знания. Если данных нет — честно скажи об этом. В конце ответа добавь строку «Источники:» с 1–3 самыми полезными ссылками из результатов.
- Не выводи служебные пометки, названия инструментов и технические маркеры."""

CLASSIC_FORMAT = """ФОРМАТИРОВАНИЕ — СТРОГО:
- Разрешены только теги Telegram: <b>, <i>, <u>, <s>, <code>, <pre>, <a href="...">, <blockquote>, <tg-spoiler>.
- Запрещены: <ul>, <ol>, <li>, <p>, <br>, <div>, <span>, <h1>-<h6>, <table>, <img>.
- Списки — символами •, — или «1.», «2.» без тегов. Переносы строк — реальные.
- Никакого Markdown (** и _). Максимум 1–2 <b> на ответ.
- Символы <, >, & в обычном тексте экранируй как &lt;, &gt;, &amp;."""

RICH_FORMAT = """ФОРМАТИРОВАНИЕ — Telegram Rich Message (HTML), как статья:
- Структурируй ответ: <h1>, <h2>, <h3>, <p>, <ul><li>, <ol><li>, <blockquote>.
- Таблицы: <table><tr><th>…</th><th>…</th></tr><tr><td>…</td></tr></table>.
- Код: <pre><code>…</code></pre>. Акценты: <b>, <i>, <s>, <a href="...">, <mark>.
- Раскрывающийся блок: <details><summary>Заголовок</summary><p>текст</p></details>.
- Между блоками оставляй пустую строку. Не используй <br>, <div>, <span>, <img>.
- Заголовки и таблицы уместны для ответов от 3–4 абзацев. Короткий ответ — просто <p> без заголовка.
- Списки — когда перечисляешь 3+ пункта; не превращай каждый ответ в список.
- Символы <, >, & в обычном тексте экранируй как &lt;, &gt;, &amp;. Никакого Markdown."""

# Совместимость со старым именем.
SYSTEM_PROMPT = SYSTEM_PROMPT_BASE


def system_prompt(rich: bool = False) -> str:
    fmt = RICH_FORMAT if rich else CLASSIC_FORMAT
    return (
        SYSTEM_PROMPT_BASE
        + "\n\n"
        + fmt
        + f"\n\nСегодня: {date.today().isoformat()}."
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Поиск в интернете. Используй для свежих данных: новости, погода, курсы, "
                "цены, версии, события, актуальные факты. Не используй для общих знаний."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "Сгенерировать и отправить изображение. Только для визуальных объектов: "
                "кот, пейзаж, постер, логотип. Не для кода, текста, таблиц, схем и графиков."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt_en": {"type": "string", "description": "Промпт на английском"},
                    "prompt_ru": {"type": "string", "description": "Тот же промпт на русском"},
                },
                "required": ["prompt_en", "prompt_ru"],
            },
        },
    },
]

SEARCH_TOOLS = [TOOLS[0]]


# ============================================================
# GROQ
# ============================================================


async def groq_chat(messages, tools=None, tool_choice="auto", max_tokens: int | None = None):
    async def _do():
        kwargs = {
            "model": CHAT_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": int(max_tokens or CHAT_MAX_TOKENS),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        return await asyncio.wait_for(
            ai_client.chat.completions.create(**kwargs),
            timeout=AI_TIMEOUT,
        )

    return await with_retry(_do, retries=3)


def openai_to_chat_result(response, provider: str, model: str) -> ChatResult:
    message = response.choices[0].message
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
    reasoning_tokens = 0
    cached_tokens = 0
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    if details is not None:
        reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
    prompt_details = getattr(usage, "prompt_tokens_details", None) if usage else None
    if prompt_details is not None:
        cached_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)

    calls = []
    for tc in (getattr(message, "tool_calls", None) or []):
        calls.append(
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
        )

    return ChatResult(
        content=(message.content or "").strip(),
        reasoning_content=(getattr(message, "reasoning_content", "") or "").strip(),
        tool_calls=calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
        total_tokens=total_tokens or prompt_tokens + completion_tokens,
        model=model,
        finish_reason=response.choices[0].finish_reason or "",
    )


async def groq_stream(
    messages,
    tools=None,
    tool_choice="auto",
    max_tokens: int | None = None,
    on_delta=None,
) -> ChatResult:
    """Стриминг Groq с накоплением текста, мыслей и tool-call'ов."""
    base = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": int(max_tokens or CHAT_MAX_TOKENS),
        "stream": True,
    }
    if tools:
        base["tools"] = tools
        base["tool_choice"] = tool_choice or "auto"

    async def _open(with_usage: bool):
        kwargs = dict(base)
        if with_usage:
            kwargs["stream_options"] = {"include_usage": True}
        return await asyncio.wait_for(
            ai_client.chat.completions.create(**kwargs), timeout=AI_TIMEOUT
        )

    try:
        stream = await _open(True)
    except Exception:
        stream = await _open(False)

    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict] = {}
    finish = ""
    usage = None

    async def emit(kind: str, piece: str) -> None:
        if not piece or on_delta is None:
            return
        try:
            result = on_delta(kind, piece)
            if inspect.isawaitable(result):
                await result
        except Exception as e:
            logging.warning(f"Groq on_delta failed: {e}")

    async for chunk in stream:
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = chunk_usage
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        choice = choices[0]
        if getattr(choice, "finish_reason", None):
            finish = choice.finish_reason
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        piece = getattr(delta, "content", None)
        if piece:
            content.append(piece)
            await emit("content", piece)
        rpiece = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if rpiece:
            reasoning.append(rpiece)
            await emit("reasoning", rpiece)
        for tc in (getattr(delta, "tool_calls", None) or []):
            idx = getattr(tc, "index", 0) or 0
            slot = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if getattr(tc, "id", None):
                slot["id"] = tc.id
            function = getattr(tc, "function", None)
            if function is not None:
                if getattr(function, "name", None):
                    slot["name"] = function.name
                if getattr(function, "arguments", None):
                    slot["arguments"] += function.arguments

    tool_calls = [
        {
            "id": slot["id"] or f"call_{idx}",
            "type": "function",
            "function": {"name": slot["name"], "arguments": slot["arguments"] or "{}"},
        }
        for idx, slot in sorted(calls.items())
        if slot["name"]
    ]

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
    reasoning_tokens = 0
    cached_tokens = 0
    if usage is not None:
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if prompt_details is not None:
            cached_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)

    return ChatResult(
        content="".join(content).strip(),
        reasoning_content="".join(reasoning).strip(),
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
        total_tokens=total_tokens or prompt_tokens + completion_tokens,
        model=CHAT_MODEL,
        finish_reason=finish or "",
    )


async def call_model(
    provider: str,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = "auto",
    track: Track | None = None,
    effort: str = "low",
    max_tokens: int | None = None,
    model: str | None = None,
    on_delta=None,
) -> ChatResult:
    started = time.perf_counter()
    status = "ok"
    error_text = ""
    result: ChatResult | None = None
    client = PROVIDER_CLIENTS.get(provider)
    model_name = model or (
        CHAT_MODEL if provider == "groq" else (client.model if client else provider)
    )
    default_max = CHAT_MAX_TOKENS if provider == "groq" else PREMIUM_MAX_TOKENS
    try:
        if provider == "groq":
            if on_delta is not None:
                result = await groq_stream(
                    messages,
                    tools,
                    tool_choice or "auto",
                    max_tokens or default_max,
                    on_delta,
                )
            else:
                response = await groq_chat(
                    messages,
                    tools,
                    tool_choice or "auto",
                    max_tokens or default_max,
                )
                result = openai_to_chat_result(response, provider, CHAT_MODEL)
        else:
            if client is None or not client.configured:
                raise RuntimeError(f"{PROVIDER_NAMES.get(provider, provider)} не настроен")
            result = await with_retry(
                lambda: client.chat(
                    messages,
                    model=model_name,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=max_tokens or default_max,
                    effort=effort,
                    on_delta=on_delta,
                ),
                retries=1 if on_delta is not None else 2,
            )
        return result
    except Exception as e:
        status = "error"
        error_text = str(e)
        raise
    finally:
        latency = int((time.perf_counter() - started) * 1000)
        if track is not None:
            await storage.add_call(
                request_id=track.request_id,
                user_id=track.user_id,
                chat_id=track.chat_id,
                provider=provider,
                model=(result.model if result else model_name),
                prompt_tokens=result.prompt_tokens if result else 0,
                completion_tokens=result.completion_tokens if result else 0,
                reasoning_tokens=result.reasoning_tokens if result else 0,
                cached_tokens=result.cached_tokens if result else 0,
                total_tokens=result.total_tokens if result else 0,
                credit=result.credit if result else 0,
                latency_ms=latency,
                status=status,
                error=error_text,
            )


def assistant_tool_message(msg) -> dict:
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in (msg.tool_calls or [])
        ],
    }


# ============================================================
# GEMINI — ОПИСАНИЕ ФОТО
# ============================================================

VISION_PROMPT = (
    "Опиши это изображение подробно для памяти ассистента: объекты, люди, "
    "цвета, расположение, весь видимый текст, важные детали. 80–150 слов. "
    "Пиши только описание, без вступлений вроде «на фото»."
)


def image_mime(image_bytes: bytes, fallback: str | None = None) -> str:
    if fallback:
        return fallback
    if image_bytes.startswith(b"\x89PNG"):
        return "image/png"
    if image_bytes.startswith(b"GIF8"):
        return "image/gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


async def describe_image(
    image_bytes: bytes,
    mime: str | None = None,
    *,
    track: Track | None = None,
) -> str:
    mime = image_mime(image_bytes, mime)
    b64 = base64.b64encode(image_bytes).decode()

    async def gemini_describe() -> tuple[str, dict]:
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": VISION_PROMPT},
                        {"inline_data": {"mime_type": mime, "data": b64}},
                    ],
                }
            ],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 700},
        }

        response = await http_client.post(
            f"{GEMINI_BASE_URL}/models/{VISION_MODEL}:generateContent",
            headers={"x-goog-api-key": GEMINI_API_KEY},
            json=payload,
            timeout=AI_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini вернул пустой ответ: {str(data)[:200]}")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()
        if not text:
            raise RuntimeError(
                f"Gemini не вернул описание (finishReason={candidates[0].get('finishReason')})"
            )
        return text, data.get("usageMetadata") or {}

    async def groq_describe() -> tuple[str, dict]:
        content = [
            {"type": "text", "text": VISION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
        response = await ai_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=800,
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("Groq vision вернул пустой ответ")
        return text, response

    started = time.perf_counter()
    try:
        text, usage = await with_retry(gemini_describe, retries=2)
        if track is not None:
            prompt_tokens = int(usage.get("promptTokenCount") or 0)
            completion_tokens = int(usage.get("candidatesTokenCount") or 0)
            reasoning_tokens = int(usage.get("thoughtsTokenCount") or 0)
            cached_tokens = int(usage.get("cachedContentTokenCount") or 0)
            await storage.add_call(
                request_id=track.request_id,
                user_id=track.user_id,
                chat_id=track.chat_id,
                provider="gemini",
                model=VISION_MODEL,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
                cached_tokens=cached_tokens,
                total_tokens=prompt_tokens + completion_tokens + reasoning_tokens,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        return text
    except Exception as e:
        logging.warning(
            f"Gemini vision failed ({type(e).__name__}: {e}); "
            f"пробую {GROQ_VISION_MODEL}"
        )
        if track is not None:
            await storage.add_call(
                request_id=track.request_id,
                user_id=track.user_id,
                chat_id=track.chat_id,
                provider="gemini",
                model=VISION_MODEL,
                latency_ms=int((time.perf_counter() - started) * 1000),
                status="error",
                error=str(e),
            )

    started = time.perf_counter()
    try:
        text, response = await with_retry(groq_describe, retries=2)
    except Exception as e:
        if track is not None:
            await storage.add_call(
                request_id=track.request_id,
                user_id=track.user_id,
                chat_id=track.chat_id,
                provider="groq",
                model=GROQ_VISION_MODEL,
                latency_ms=int((time.perf_counter() - started) * 1000),
                status="error",
                error=str(e),
            )
        raise

    if track is not None:
        usage = response.usage
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        await storage.add_call(
            request_id=track.request_id,
            user_id=track.user_id,
            chat_id=track.chat_id,
            provider="groq",
            model=GROQ_VISION_MODEL,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    return text


async def describe_image_premium(
    image_bytes: bytes,
    mime: str | None = None,
    *,
    provider: str = "deepseek",
    track: Track | None = None,
) -> str:
    """Описание фото моделью провайдера (DeepSeek или Qwen)."""
    mime = image_mime(image_bytes, mime)
    b64 = base64.b64encode(image_bytes).decode()
    vision_model = PROVIDER_VISION_MODEL.get(provider)
    result = await call_model(
        provider,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }
        ],
        track=track,
        effort="off",
        max_tokens=1500,
        model=vision_model,
    )
    return (result.content or "").strip()


# ============================================================
# WHISPER
# ============================================================

# Groq принимает только эти расширения. Важно: .oga (как у голосовых
# Telegram) в списке нет — его нужно отправлять как .ogg.
GROQ_AUDIO_EXTENSIONS = {
    ".flac", ".mp3", ".mp4", ".mpeg", ".m4a",
    ".ogg", ".opus", ".wav", ".webm",
}
AUDIO_MIME_BY_EXT = {
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".mpeg": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}
AUDIO_EXT_BY_MIME = {
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/ogg": ".ogg",
    "audio/opus": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
}


def audio_extension(media, file_path: str | None) -> str:
    """Подбирает расширение, которое примет Groq."""
    suffix = os.path.splitext(file_path or "")[1].lower()
    if suffix == ".oga":  # Telegram voice
        suffix = ".ogg"
    if suffix in GROQ_AUDIO_EXTENSIONS:
        return suffix
    mime = (getattr(media, "mime_type", "") or "").lower()
    return AUDIO_EXT_BY_MIME.get(mime, ".ogg")


async def transcribe_voice(file_path: str, *, track: Track | None = None) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in GROQ_AUDIO_EXTENSIONS:
        ext = ".ogg"
    content_type = AUDIO_MIME_BY_EXT[ext]

    async def _do():
        with open(file_path, "rb") as f:
            result = await asyncio.wait_for(
                ai_client.audio.transcriptions.create(
                    # Явное имя обязательно: иначе SDK возьмёт имя временного
                    # файла (например, .oga), и Groq ответит 400.
                    file=(f"voice{ext}", f, content_type),
                    model=WHISPER_MODEL,
                    language="ru",
                ),
                timeout=AI_TIMEOUT,
            )
        return (result.text or "").strip()

    started = time.perf_counter()
    try:
        text = await with_retry(_do, retries=2)
        if track is not None:
            await storage.add_call(
                request_id=track.request_id,
                user_id=track.user_id,
                chat_id=track.chat_id,
                provider="groq",
                model=WHISPER_MODEL,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        return text
    except Exception as e:
        if track is not None:
            await storage.add_call(
                request_id=track.request_id,
                user_id=track.user_id,
                chat_id=track.chat_id,
                provider="groq",
                model=WHISPER_MODEL,
                latency_ms=int((time.perf_counter() - started) * 1000),
                status="error",
                error=str(e),
            )
        raise


# ============================================================
# ПОИСК — TAVILY + запасной DuckDuckGo
# ============================================================


async def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    async def _do():
        response = await http_client.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json().get("results", []) or []

    return await with_retry(_do, retries=2)


async def ddg_search(query: str, max_results: int = 5) -> list[dict]:
    def _sync():
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    try:
        return await asyncio.wait_for(asyncio.to_thread(_sync), timeout=25.0)
    except Exception as e:
        logging.warning(f"DDG fallback failed: {e}")
        return []


def normalize_results(results: list[dict]) -> list[dict]:
    normalized = []
    for item in results:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or item.get("href") or "").strip()
        snippet = (item.get("content") or item.get("body") or "").strip()
        if not (title or snippet):
            continue
        normalized.append({"title": title, "url": url, "snippet": snippet})
    return normalized


async def search_web(query: str, max_results: int = 5) -> list[dict]:
    query = (query or "").strip()
    if not query:
        return []

    key = query.lower()
    now = time.time()
    cached = _search_cache.get(key)
    if cached and now - cached[0] < 600:
        return cached[1]

    results: list[dict] = []
    try:
        results = normalize_results(await tavily_search(query, max_results))
        logging.info(f"TAVILY '{query}' -> {len(results)} результатов")
    except Exception as e:
        logging.warning(f"Tavily failed: {e}")

    if not results:
        results = normalize_results(await ddg_search(query, max_results))
        logging.info(f"DDG '{query}' -> {len(results)} результатов")

    _search_cache[key] = (now, results)
    if len(_search_cache) > 300:
        for old_key in list(_search_cache.keys())[:100]:
            _search_cache.pop(old_key, None)
    return results


def render_results(results: list[dict]) -> str:
    if not results:
        return "Поиск не дал результатов."
    lines = []
    for i, r in enumerate(results[:4], 1):
        lines.append(
            f"{i}. {r['title']}\n{r['url']}\n{r['snippet'][:250]}"
        )
    return "\n\n".join(lines)


# ============================================================
# КАРТИНКИ — POLLINATIONS
# ============================================================


def clamp_size(width: int, height: int) -> tuple[int, int]:
    width = max(256, min(2048, int(width)))
    height = max(256, min(2048, int(height)))
    max_area = 768 * 768
    if width * height > max_area:
        k = (max_area / (width * height)) ** 0.5
        width = int(width * k)
        height = int(height * k)
    return max(256, width // 8 * 8), max(256, height // 8 * 8)


def parse_image_params(text: str) -> tuple[int, int]:
    width, height = 768, 768
    lowered = (text or "").lower()

    m = re.search(r"(\d{2,4})\s*[xх]\s*(\d{2,4})", lowered)
    if m:
        width, height = int(m.group(1)), int(m.group(2))
    elif "16:9" in lowered or "горизонталь" in lowered or "широк" in lowered:
        width, height = 768, 432
    elif "9:16" in lowered or "вертикаль" in lowered:
        width, height = 432, 768
    elif "4:3" in lowered:
        width, height = 768, 576

    return clamp_size(width, height)


async def generate_image_bytes(prompt_en: str, width: int, height: int) -> bytes:
    seed = random.randint(1, 1_000_000)
    url = (
        f"https://image.pollinations.ai/prompt/{quote(prompt_en, safe='')}"
        f"?width={width}&height={height}&nologo=true&seed={seed}"
    )

    async def _do():
        response = await http_client.get(url, timeout=90.0)
        response.raise_for_status()
        data = response.content
        ctype = (response.headers.get("content-type") or "").lower()
        looks_like_image = (
            data.startswith(b"\x89PNG")
            or data.startswith(b"\xff\xd8")
            or data.startswith(b"GIF8")
            or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")
        )
        if not looks_like_image:
            raise RuntimeError(
                f"Генератор вернул не картинку (content-type: {ctype or 'unknown'})"
            )
        if len(data) < 1000:
            raise RuntimeError("Пустой ответ от генератора картинок")
        return data

    async with _image_lock:
        return await with_retry(_do, retries=2, base_wait=3.0)


# ============================================================
# ПЕРЕВОД ПРОМПТА
# ============================================================


async def translate_to_english(text: str, *, track: Track | None = None) -> str:
    async def _do():
        response = await asyncio.wait_for(
            ai_client.chat.completions.create(
                model=FAST_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Переведи текст на английский для генератора изображений. "
                            "Сохрани объекты, стиль, атмосферу и детали. Убери служебные "
                            "фразы вроде «нарисуй». Верни только перевод без кавычек."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.2,
                max_tokens=300,
            ),
            timeout=30.0,
        )
        return response

    started = time.perf_counter()
    try:
        response = await with_retry(_do, retries=2)
        result = (response.choices[0].message.content or "").strip().strip('"').strip("'")
        if track is not None:
            usage = response.usage
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            await storage.add_call(
                request_id=track.request_id,
                user_id=track.user_id,
                chat_id=track.chat_id,
                provider="groq",
                model=FAST_MODEL,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        return result or text
    except Exception as e:
        logging.warning(f"translate_to_english failed: {e}")
        if track is not None:
            await storage.add_call(
                request_id=track.request_id,
                user_id=track.user_id,
                chat_id=track.chat_id,
                provider="groq",
                model=FAST_MODEL,
                latency_ms=int((time.perf_counter() - started) * 1000),
                status="error",
                error=str(e),
            )
        return text

# ============================================================
# ОТПРАВКА И СТАТУСЫ
# ============================================================


@dataclass
class Status:
    message: Message
    msg: Message | None = None

    async def show(self, text: str) -> None:
        if self.msg is not None:
            try:
                if self.msg.text != text:
                    await self.msg.edit_text(text)
            except Exception:
                pass
            return
        try:
            self.msg = await self.message.answer(text)
        except Exception as e:
            logging.warning(f"Status send failed: {e}")

    async def clear(self) -> None:
        if self.msg is None:
            return
        try:
            await self.msg.delete()
        except Exception:
            pass
        self.msg = None


@dataclass
class Turn:
    message: Message
    chat_id: int
    is_private: bool
    user_text: str = ""
    explicit_draw: str | None = None
    image_sent: bool = False
    image_prompt_ru: str = ""
    image_error: str = ""
    model: str = "groq"
    track: Track | None = None
    tools_used: list[str] = field(default_factory=list)
    rich_allowed: bool = True
    status: Status = field(init=False)

    def __post_init__(self):
        self.status = Status(self.message)


def utf16_units(text: str) -> int:
    return sum(2 if ord(ch) > 0xFFFF else 1 for ch in text)


def truncate_utf16(text: str, limit: int) -> str:
    units = 0
    out = []
    for ch in text:
        u = 2 if ord(ch) > 0xFFFF else 1
        if units + u > limit:
            break
        out.append(ch)
        units += u
    return "".join(out)


def split_text(text: str) -> list[str]:
    if utf16_units(text) <= MAX_REPLY_LENGTH:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    units = 0
    for ch in text:
        u = 2 if ord(ch) > 0xFFFF else 1
        if units + u > MAX_REPLY_LENGTH and current:
            chunks.append("".join(current))
            current = []
            units = 0
        current.append(ch)
        units += u
    if current:
        chunks.append("".join(current))
    return chunks


async def send_text(turn: Turn, text: str) -> None:
    text = sanitize_html(text)
    if not text:
        return
    for chunk in split_text(text):
        try:
            if turn.is_private:
                await turn.message.answer(chunk)
            else:
                await turn.message.reply(chunk)
            continue
        except Exception as e:
            logging.error(f"send_text error: {e}")

        # Если Telegram не принял HTML (битый тег и т.п.) — отправляем как обычный текст.
        plain = html.unescape(re.sub(r"<[^>]+>", "", chunk)).strip()
        if not plain:
            continue
        try:
            if turn.is_private:
                await turn.message.answer(plain, parse_mode=None)
            else:
                await turn.message.reply(plain, parse_mode=None)
        except Exception as e2:
            logging.error(f"send_text plain fallback error: {e2}")


RICH_ALLOWED_TAGS = (
    "h1|h2|h3|h4|h5|h6|p|ul|ol|li|blockquote|pre|code|b|i|u|s|a|mark|"
    "details|summary|table|thead|tbody|tr|td|th|tg-spoiler"
)
RICH_DROP_TAGS = ("script", "style", "iframe", "svg", "object", "embed", "form", "video", "audio")


def sanitize_rich(text: str) -> str:
    """Чистит HTML от опасных тегов, оставляя разметку Rich Message."""
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    for tag in RICH_DROP_TAGS:
        text = re.sub(
            rf"<{tag}\b[^>]*>.*?</{tag}>", "", text, flags=re.IGNORECASE | re.DOTALL
        )
        text = re.sub(rf"</?{tag}\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<img\b[^>]*>", "", text, flags=re.IGNORECASE)

    allowed_pattern = re.compile(
        rf"</?(?:{RICH_ALLOWED_TAGS})(?:\s+[^>]*)?>", re.IGNORECASE
    )
    placeholders: dict[str, str] = {}

    def _hide(match):
        key = placeholder_key(len(placeholders))
        placeholders[key] = match.group(0)
        return key

    text = allowed_pattern.sub(_hide, text)
    text = re.sub(
        r"&(?!(?:amp|lt|gt|quot|#\d+|#x[0-9a-f]+);)", "&amp;", text, flags=re.IGNORECASE
    )
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    for key, tag in placeholders.items():
        text = text.replace(key, tag)
    return text


def split_rich(text: str, limit: int = MAX_RICH_REPLY_LENGTH) -> list[str]:
    if len(text) <= limit:
        return [text] if text.strip() else []
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n\n", 0, limit)
        if cut < limit * 0.4:
            cut = remaining.rfind("\n", 0, limit)
        if cut < limit * 0.4:
            cut = remaining.rfind(". ", 0, limit)
        if cut < limit * 0.4:
            cut = limit
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return [c for c in chunks if c]


def plain_from_rich(text: str) -> str:
    text = re.sub(
        r"<details\b[^>]*>.*?</details>", "", text, flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(
        r"</(?:p|div|h[1-6]|li|tr|blockquote)>", "\n", text, flags=re.IGNORECASE
    )
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


class RichStreamer:
    """Draft-стриминг Rich Message: мысли в <details>, текст — как статья."""

    def __init__(self, turn: Turn) -> None:
        self.turn = turn
        self.draft_id = random.randint(1, 2**31 - 1)
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.draft_ok = True
        self.finished = False
        self.last_update = 0.0

    def reset(self) -> None:
        self.content.clear()
        self.reasoning.clear()

    def _thoughts_html(self) -> str:
        raw = "".join(self.reasoning).strip()
        if not raw:
            return ""
        body = sanitize_rich(raw[:20000])
        return f"<details><summary>🧠 Размышления</summary>\n<p>{body}</p>\n</details>"

    def build_html(self) -> str:
        parts: list[str] = []
        thoughts = self._thoughts_html()
        if thoughts:
            parts.append(thoughts)
        body = sanitize_rich("".join(self.content))
        if body:
            parts.append(body)
        return "\n\n".join(parts)

    async def on_delta(self, kind: str, piece: str) -> None:
        if kind == "content":
            self.content.append(piece)
        else:
            self.reasoning.append(piece)
        now = time.monotonic()
        if now - self.last_update < RICH_DRAFT_INTERVAL:
            return
        self.last_update = now
        await self.push_draft()

    async def push_draft(self) -> None:
        if not self.draft_ok:
            return
        html_text = self.build_html()
        if not html_text:
            return
        if len(html_text) > MAX_RICH_REPLY_LENGTH:
            pieces = split_rich(html_text)
            html_text = pieces[0] if pieces else html_text[:MAX_RICH_REPLY_LENGTH]
        try:
            await bot.send_rich_message_draft(
                chat_id=self.turn.chat_id,
                draft_id=self.draft_id,
                rich_message=InputRichMessage(html=html_text),
            )
        except Exception as e:
            logging.info(f"rich draft disabled: {e}")
            self.draft_ok = False

    async def finish(self, answer: str = "") -> None:
        if self.finished:
            return
        self.finished = True
        raw = "".join(self.content).strip()
        if not raw:
            raw = (answer or "").strip()
        if not raw:
            return
        chunks = split_rich(raw)
        for index, chunk in enumerate(chunks):
            html_text = sanitize_rich(chunk)
            if index == 0:
                thoughts = self._thoughts_html()
                if thoughts:
                    html_text = thoughts + "\n\n" + html_text
            if not html_text.strip():
                continue
            try:
                await bot.send_rich_message(
                    chat_id=self.turn.chat_id,
                    rich_message=InputRichMessage(html=html_text),
                )
            except Exception as e:
                logging.warning(f"send_rich_message failed: {e}; fallback to text")
                plain = plain_from_rich(html_text)
                if plain:
                    await send_text(self.turn, plain)


async def send_photo(turn: Turn, data: bytes, caption: str) -> bool:
    ext = "png" if data.startswith(b"\x89PNG") else "jpg"
    caption = caption[:1000]
    file = BufferedInputFile(data, filename=f"image.{ext}")
    try:
        if turn.is_private:
            sent = await turn.message.answer_photo(photo=file, caption=caption)
        else:
            sent = await turn.message.reply_photo(photo=file, caption=caption)
        _media_cache[(turn.chat_id, sent.message_id)] = turn.image_prompt_ru or "сгенерированная картинка"
        return True
    except Exception as e:
        logging.error(f"send_photo error: {e}")
        await send_text(turn, f"⚠️ Не удалось отправить картинку: <code>{esc(e)}</code>")
        return False


# ============================================================
# СКАЧИВАНИЕ МЕДИА
# ============================================================


async def download_image_bytes(message: Message) -> bytes:
    if message.photo:
        file_id = message.photo[-1].file_id
    elif message.document and (message.document.mime_type or "").startswith("image/"):
        file_id = message.document.file_id
    else:
        raise ValueError("В сообщении нет изображения")

    file = await bot.get_file(file_id)
    buf = await bot.download_file(file.file_path)
    buf.seek(0)
    data = buf.read()
    if len(data) > 8 * 1024 * 1024:
        raise ValueError("изображение больше 8 МБ")
    return data


async def download_media_to_temp(message: Message) -> str:
    media = message.voice or message.audio
    if media is None:
        raise ValueError("В сообщении нет голосового")
    file = await bot.get_file(media.file_id)
    suffix = audio_extension(media, file.file_path)
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    await bot.download_file(file.file_path, path)
    return path


async def get_photo_description(
    message: Message, *, track: Track | None = None, provider: str = "groq"
) -> str:
    key = (message.chat.id, message.message_id)
    cached = _media_cache.get(key)
    if cached:
        return cached
    try:
        data = await download_image_bytes(message)
        if provider in ("deepseek", "qwen"):
            desc = await describe_image_premium(data, track=track, provider=provider)
        else:
            desc = await describe_image(data, track=track)
    except Exception as e:
        logging.warning(f"describe photo failed: {e}")
        desc = "[фото] (не удалось описать)"

    _media_cache[key] = desc
    if len(_media_cache) > 500:
        for old_key in list(_media_cache.keys())[:150]:
            _media_cache.pop(old_key, None)
    return desc


async def get_voice_text(message: Message, *, track: Track | None = None) -> str:
    key = (message.chat.id, message.message_id)
    cached = _media_cache.get(key)
    if cached:
        return cached

    path = None
    try:
        path = await download_media_to_temp(message)
        text = await transcribe_voice(path, track=track)
    except Exception as e:
        logging.warning(f"transcribe failed: {e}")
        text = ""
    finally:
        if path:
            try:
                os.unlink(path)
            except Exception:
                pass

    if not text:
        text = "[голосовое] (не удалось расшифровать)"

    _media_cache[key] = text
    return text


# ============================================================
# РЕПЛАИ
# ============================================================


async def build_reply_block(
    message: Message,
    need_media: bool = True,
    *,
    track: Track | None = None,
    provider: str = "groq",
) -> tuple[str, str]:
    """Возвращает (цитата для контекста, описание медиа из реплая)."""
    reply = message.reply_to_message
    if not reply:
        return "", ""

    author = display_user(reply.from_user) if reply.from_user else "канал"
    body = ""
    media_desc = ""

    if reply.photo:
        if need_media:
            media_desc = await get_photo_description(reply, track=track, provider=provider)
            body = f"фото: {media_desc}"
        else:
            body = "[фото]"
    elif reply.document and (reply.document.mime_type or "").startswith("image/"):
        if need_media:
            media_desc = await get_photo_description(reply, track=track, provider=provider)
            body = f"фото: {media_desc}"
        else:
            body = "[фото]"
    elif reply.voice or reply.audio:
        if need_media:
            media_desc = await get_voice_text(reply, track=track)
            body = f"голосовое: {media_desc}"
        else:
            body = "[голосовое]"
    elif reply.text:
        body = reply.text[:500]
    elif reply.caption:
        body = reply.caption[:500]
    else:
        body = media_placeholder(reply) or "[сообщение]"

    if reply.caption and reply.photo:
        body += f" (подпись: {reply.caption[:200]})"

    return f"[в ответ на сообщение {author}: «{body}»] ", media_desc


# ============================================================
# ИНСТРУМЕНТЫ И ОТВЕТ
# ============================================================


def is_trivial(text: str) -> bool:
    cleaned = (text or "").strip().strip(".!,").lower()
    return cleaned in {"", "ok", "ок", "готово", "done"}


async def execute_tool(turn: Turn, name: str, args: dict) -> str:
    started = time.perf_counter()
    status = "ok"
    error_text = ""
    turn.tools_used.append(name)
    try:
        if name == "web_search":
            query = (args.get("query") or "").strip()
            if not query:
                return "Ошибка: пустой поисковый запрос."
            await turn.status.show("🔍 Ищу в интернете…")
            logging.info(f"[{turn.chat_id}] TOOL web_search: {query}")
            results = await search_web(query)
            return render_results(results)

        if name == "generate_image":
            if turn.image_sent:
                return "Изображение уже отправлено, повторно не генерируй."
            prompt_en = (args.get("prompt_en") or "").strip()
            prompt_ru = (args.get("prompt_ru") or "").strip() or prompt_en
            if not prompt_en:
                return "Ошибка: пустой промпт изображения."

            await turn.status.show("🎨 Рисую…")
            width, height = parse_image_params(turn.user_text)
            logging.info(f"[{turn.chat_id}] TOOL generate_image: {prompt_en[:100]} ({width}x{height})")
            try:
                data = await generate_image_bytes(prompt_en, width, height)
                turn.image_prompt_ru = prompt_ru
                ok = await send_photo(turn, data, f"🎨 Нарисовал: <i>{esc(prompt_ru[:300])}</i>")
                if not ok:
                    turn.image_error = "не удалось отправить изображение в Telegram"
                    return "Ошибка: изображение сгенерировано, но не отправлено."
                turn.image_sent = True
                turn.image_error = ""
                return "Изображение сгенерировано и отправлено пользователю."
            except Exception as e:
                turn.image_error = str(e)
                logging.exception("Image generation error")
                return f"Ошибка генерации изображения: {e}"

        return "Неизвестный инструмент."
    except Exception as e:
        status = "error"
        error_text = str(e)
        raise
    finally:
        if turn.track is not None:
            await storage.add_tool(
                request_id=turn.track.request_id,
                user_id=turn.track.user_id,
                chat_id=turn.track.chat_id,
                tool=name,
                args_preview=json.dumps(args, ensure_ascii=False)[:300],
                status=status,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error=error_text,
            )


async def run_tools(
    messages,
    tools,
    executor,
    forced=None,
    *,
    provider: str = "groq",
    track: Track | None = None,
    effort: str = "off",
    model: str | None = None,
    on_delta=None,
    reset_cb=None,
) -> str:
    choice = forced or "auto"
    max_tokens = CHAT_MAX_TOKENS if provider == "groq" else PREMIUM_MAX_TOKENS
    for _ in range(4):
        try:
            result = await call_model(
                provider,
                messages,
                tools=tools,
                tool_choice=choice,
                track=track,
                effort=effort,
                max_tokens=max_tokens,
                model=model,
                on_delta=on_delta,
            )
        except Exception:
            if choice != "auto":
                choice = "auto"
                continue
            raise

        if not result.tool_calls:
            return (result.content or "").strip()

        if reset_cb is not None:
            reset_cb()
        messages.append(result.assistant_message(include_reasoning=provider == "deepseek"))
        for tc in result.tool_calls:
            function = tc.get("function") or {}
            try:
                args = json.loads(function.get("arguments") or "{}")
            except Exception:
                args = {}
            try:
                output = await executor(function.get("name") or "", args)
            except Exception as e:
                output = f"Ошибка инструмента: {e}"
            messages.append(
                {"role": "tool", "tool_call_id": tc.get("id"), "content": str(output)[:6000]}
            )
        choice = "auto"
    return ""


async def fallback_draw(turn: Turn) -> None:
    prompt_ru = (turn.explicit_draw or turn.user_text or "").strip()
    if not prompt_ru:
        return
    await turn.status.show("🎨 Рисую…")
    prompt_en = await translate_to_english(prompt_ru, track=turn.track)
    width, height = parse_image_params(turn.user_text)
    try:
        data = await generate_image_bytes(prompt_en, width, height)
        turn.image_prompt_ru = prompt_ru
        ok = await send_photo(turn, data, f"🎨 Нарисовал: <i>{esc(prompt_ru[:300])}</i>")
        if not ok:
            turn.image_error = "не удалось отправить изображение в Telegram"
        else:
            turn.image_sent = True
            turn.image_error = ""
    except Exception as e:
        turn.image_error = str(e)
        logging.exception("Fallback image generation error")
        await send_text(turn, f"⚠️ Ошибка генерации: <code>{esc(e)}</code>")


async def send_switch_button(turn: Turn, error_text: str) -> None:
    provider = provider_of(turn.model)
    name = PROVIDER_NAMES.get(provider, "Модель")
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🌿 Перейти на Groq", callback_data="model:set:groq")]
        ]
    )
    text = (
        f"⚠️ {name} не ответил: <code>{esc(error_text[:500])}</code>\n\n"
        f"Модель осталась {esc(model_label(turn.model))}. "
        "Можно переключиться на Groq кнопкой ниже."
    )
    try:
        if turn.is_private:
            await turn.message.answer(text, reply_markup=keyboard)
        else:
            await turn.message.reply(text, reply_markup=keyboard)
    except Exception:
        await send_text(turn, text)


async def run_ai(turn: Turn) -> None:
    started = time.perf_counter()
    model_key = turn.model if turn.model in MODEL_REGISTRY else "groq"
    info = MODEL_REGISTRY[model_key]
    provider = info["provider"]
    model_id = info["model"]
    user = turn.message.from_user
    user_id = user.id if user else None
    settings_active = turn.is_private and user_id is not None
    rich = bool(turn.rich_allowed and settings_active and user_rich(user_id))
    streamer = RichStreamer(turn) if rich else None
    status = "ok"
    error_text = ""
    try:
        await safe_typing(turn.chat_id)
        messages = [{"role": "system", "content": system_prompt(rich)}]
        messages += [dict(m) for m in history_for_model(turn.chat_id, model_key)]

        search_mode = user_search_mode(user_id) if settings_active else "auto"
        tools = TOOLS
        if search_mode == "off":
            tools = [
                t
                for t in TOOLS
                if (t.get("function") or {}).get("name") != "web_search"
            ]

        if turn.explicit_draw:
            forced = {"type": "function", "function": {"name": "generate_image"}}
        elif search_mode == "always":
            forced = {"type": "function", "function": {"name": "web_search"}}
        elif has_search_verb(turn.user_text):
            forced = {"type": "function", "function": {"name": "web_search"}}
        else:
            forced = None

        effort = user_effort(user_id) if settings_active else "off"

        async def executor(name, args):
            return await execute_tool(turn, name, args)

        answer = await run_tools(
            messages,
            tools,
            executor,
            forced,
            provider=provider,
            track=turn.track,
            effort=effort,
            model=model_id,
            on_delta=streamer.on_delta if streamer else None,
            reset_cb=streamer.reset if streamer else None,
        )

        if turn.explicit_draw and not turn.image_sent and not turn.image_error:
            await fallback_draw(turn)

        if turn.image_sent:
            await add_to_history(
                turn.chat_id,
                "assistant",
                f"Отправил сгенерированную картинку: {turn.image_prompt_ru}",
            )
        elif turn.image_error:
            await add_to_history(
                turn.chat_id,
                "assistant",
                f"Не удалось сгенерировать изображение: {turn.image_error}",
            )

        if answer:
            if not (turn.image_sent and is_trivial(answer)):
                if streamer is not None:
                    await streamer.finish(answer)
                else:
                    await send_text(turn, answer)
                await add_to_history(turn.chat_id, "assistant", answer)
        elif not turn.image_sent and not turn.image_error:
            fallback_text = "Не понял. Повтори, пожалуйста."
            if streamer is not None:
                await streamer.finish(fallback_text)
            else:
                await send_text(turn, fallback_text)
            await add_to_history(turn.chat_id, "assistant", fallback_text)

        _stats[turn.chat_id] = _stats.get(turn.chat_id, 0) + 1
        logging.info(
            f"[{turn.chat_id}] AI готов | model={model_key} | rich={rich} | "
            f"draw={turn.image_sent} | поиск={search_mode}"
        )
    except asyncio.CancelledError:
        status = "cancelled"
        raise
    except Exception as e:
        status = "error"
        error_text = str(e)
        logging.exception("AI handling error")
        if provider in ("deepseek", "qwen"):
            await send_switch_button(turn, friendly_ai_error(e))
        else:
            await send_text(turn, f"⚠️ Ошибка: <code>{esc(friendly_ai_error(e))}</code>")
    finally:
        await finish_track(
            turn.track,
            status=status,
            error=error_text,
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=model_key,
            tools=",".join(turn.tools_used),
        )
        await turn.status.clear()

# ============================================================
# КОМАНДЫ
# ============================================================


def command_allowed(message: Message) -> bool:
    if is_private(message):
        return True
    return is_whitelisted(message.chat.id)


async def reply_any(message: Message, text: str) -> None:
    try:
        if is_private(message):
            await message.answer(text)
        else:
            await message.reply(text)
    except Exception as e:
        logging.error(f"reply_any error: {e}")


@dp.message(CommandStart())
async def cmd_start(message: Message):
    if not command_allowed(message):
        return
    await reply_any(
        message,
        "🌿 Привет! Я — <b>Мудрый Мох</b>.\n\n"
        "• В <b>личке</b> — отвечаю на всё: текст, голосовые, фото, реплаи.\n"
        "• В <b>группах</b> — когда обращаются словом <b>мох</b>, через "
        f"<b>@{BOT_USERNAME or 'M0XAIBOT'}</b> или реплаем на моё сообщение.\n"
        "• <b>Рисование</b>: <code>мох нарисуй кота</code>\n"
        "• <b>Поиск</b> — включается сам, когда нужны свежие данные.\n"
        "• <b>Inline</b>: <code>@M0XAIBOT запрос</code>\n\n"
        "🧩 Модели и режимы: /models\n"
        "🧹 Сбросить контекст: /clear\n"
        "ℹ️ Обо мне: /about",
    )


@dp.message(Command("help"))
async def cmd_help(message: Message):
    if not command_allowed(message):
        return
    extra = ""
    if is_private(message) and message.from_user:
        extra = "/models — выбрать модель и режимы\n"
        if message.from_user.id == OWNER_ID:
            extra += "/admin — админ-панель\n"
    await reply_any(
        message,
        "🌿 <b>Что я умею:</b>\n\n"
        "• Отвечаю на вопросы и помню контекст\n"
        "• Понимаю фото и голосовые\n"
        "• Рисую: <code>нарисуй кота</code>\n"
        "• Ищу в интернете, когда нужно\n"
        "• Отвечаю в inline-режиме\n\n"
        "<b>Команды:</b>\n"
        "/start — приветствие\n"
        "/help — это меню\n"
        "/about — инфо обо мне\n"
        "/clear — очистить контекст\n"
        "/context — размер памяти\n"
        "/whoami — твой ID\n"
        "/stats — статистика\n"
        f"{extra}",
    )


@dp.message(Command("about"))
async def cmd_about(message: Message):
    if not command_allowed(message):
        return
    chat_id = message.chat.id
    current = "groq"
    if is_private(message) and message.from_user:
        current = selected_model(message.from_user.id)
    current_line = (
        f"\nСейчас: <b>{esc(model_label(current))}</b> (только в личке)"
        if is_private(message)
        else "\nВ группах: 🌿 Groq"
    )
    model_lines = "\n".join(
        f"• {esc(info['label'])}"
        for info in MODEL_REGISTRY.values()
    )
    await reply_any(
        message,
        "🌿 <b>Мудрый Мох — информация</b>\n\n"
        "<b>🧩 Модели:</b>\n"
        f"{model_lines}{current_line}\n\n"
        "<b>👁 Фото:</b>\n"
        f"• <code>{VISION_MODEL}</code> (Gemini) для Groq\n"
        "• Своя vision-модель для DeepSeek и Qwen\n"
        f"• Резерв: <code>{GROQ_VISION_MODEL}</code> (Groq)\n\n"
        "<b>🎙 Голос:</b>\n"
        f"• <code>{WHISPER_MODEL}</code> (Groq)\n\n"
        "<b>🎨 Рисование:</b>\n"
        "• Pollinations.ai\n\n"
        "<b>🔍 Поиск:</b>\n"
        "• Tavily (+ запасной DuckDuckGo)\n\n"
        f"<b>📊 Статистика чата:</b>\n"
        f"• Время работы: <b>{format_uptime()}</b>\n"
        f"• Запросов в этом чате: <b>{_stats.get(chat_id, 0)}</b>\n"
        f"• В контексте: <b>{get_history_size(chat_id)}</b> из {HISTORY_LIMIT}\n\n"
        "<b>🛠 Служебные команды:</b>\n"
        "/models — модели и режимы\n"
        "/clear — очистить контекст\n"
        "/context — размер памяти\n"
        "/whoami — ID и username\n"
        "/stats — статистика чата",
    )


@dp.message(Command("clear"))
async def cmd_clear(message: Message):
    if not command_allowed(message):
        return
    chat_id = message.chat.id
    await clear_chat_context(chat_id)
    await reply_any(message, "🧹 Контекст этого чата очищен. Начинаем с чистого листа!")
    logging.info(f"[{chat_id}] /clear")


MODEL_ICONS = {"groq": "🌿", "deepseek": "🧠", "qwen": "🐳"}
DEEPTHINK_LABELS = {"off": "выкл", "low": "низкий", "medium": "средний", "high": "высокий"}
DEEPTHINK_ORDER = ["off", "low", "medium", "high"]
SEARCH_LABELS = {"auto": "авто", "always": "всегда", "off": "выкл"}
SEARCH_ORDER = ["auto", "always", "off"]


def _tab_row(active: str) -> list[InlineKeyboardButton]:
    row = []
    for tab, label in (
        ("models", "🧩 Модели"),
        ("modes", "⚙️ Режимы"),
        ("access", "🔑 Доступ"),
    ):
        row.append(
            InlineKeyboardButton(
                text=("• " if tab == active else "") + label,
                callback_data=f"models:tab:{tab}",
            )
        )
    return row


async def models_panel_view(user_id: int, tab: str = "models") -> tuple[str, InlineKeyboardMarkup]:
    current = selected_model(user_id)
    settings = user_settings(user_id)
    rows: list[list[InlineKeyboardButton]] = [_tab_row(tab)]

    if tab == "modes":
        rich_on = bool(settings.get("rich", True))
        level = settings.get("deepthink", "off")
        search = settings.get("search", "auto")
        text = (
            "⚙️ <b>Режимы</b>\n\n"
            "🧩 <b>Rich-статьи</b> — оформленные ответы со стримингом "
            "(заголовки, таблицы, раскрывающиеся мысли).\n"
            "🧠 <b>Deepthink</b> — сколько модель размышляет перед ответом. "
            "Мысли показываются в свёрнутом блоке и не попадают в контекст.\n"
            "🔍 <b>Поиск</b> — когда обращаться в интернет: сам по необходимости, "
            "всегда или никогда.\n\n"
            f"Rich: <b>{'включён' if rich_on else 'выключен'}</b>\n"
            f"Deepthink: <b>{DEEPTHINK_LABELS.get(level, 'выкл')}</b>\n"
            f"Поиск: <b>{SEARCH_LABELS.get(search, 'авто')}</b>"
        )
        rows += [
            [
                InlineKeyboardButton(
                    text=f"{'✅' if rich_on else '⬜️'} Rich-статьи",
                    callback_data="models:toggle:rich:modes",
                )
            ],
            [
                InlineKeyboardButton(
                    text=f"🧠 Deepthink: {DEEPTHINK_LABELS.get(level, 'выкл')}",
                    callback_data="models:cycle:deepthink:modes",
                )
            ],
            [
                InlineKeyboardButton(
                    text=f"🔍 Поиск: {SEARCH_LABELS.get(search, 'авто')}",
                    callback_data="models:cycle:search:modes",
                )
            ],
        ]
        return text, InlineKeyboardMarkup(inline_keyboard=rows)

    if tab == "access":
        lines = [
            "🔑 <b>Доступ</b>",
            "",
            f"👑 Владелец: <a href=\"https://t.me/{OWNER_USERNAME}\">@{OWNER_USERNAME}</a>",
            "",
        ]
        today = msk_today()
        for provider, name in PROVIDER_NAMES.items():
            allowed = (int(user_id) == OWNER_ID) or ((int(user_id), provider) in _grants)
            limit = PROVIDER_LIMITS.get(provider, 0)
            client = PROVIDER_CLIENTS.get(provider)
            configured = bool(client and client.configured)
            if not configured:
                lines.append(f"⚠️ {name}: не настроен на сервере")
            elif not allowed:
                lines.append(f"🔒 {name}: нет доступа")
            elif int(user_id) == OWNER_ID:
                lines.append(f"✅ {name}: безлимит")
            else:
                used = await storage.get_usage(int(user_id), provider, today)
                lines.append(f"✅ {name}: {used}/{limit} сегодня")
        lines.append("")
        if int(user_id) == OWNER_ID:
            lines.append("У тебя есть доступ ко всем моделям.")
        else:
            lines.append(
                f"Доступ выдаёт владелец "
                f"<a href=\"https://t.me/{OWNER_USERNAME}\">@{OWNER_USERNAME}</a>."
            )
        rows.append(
            [
                InlineKeyboardButton(
                    text="🌿 Перейти на Groq", callback_data="model:set:groq"
                )
            ]
        )
        return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)

    lines = [
        "🧩 <b>Модели</b>",
        "",
        "Смена модели сохраняет контекст, кэш фото и фотосессии.",
        "",
    ]
    for key, info in MODEL_REGISTRY.items():
        allowed = model_available(user_id, key)
        provider = info["provider"]
        client = PROVIDER_CLIENTS.get(provider)
        configured = provider == "groq" or bool(client and client.configured)
        blocked = key in _model_blocked
        if int(user_id) == OWNER_ID:
            mark = "⚠️ " if (blocked or not configured) else ("✅ " if key == current else "")
        else:
            if not provider_eligible(user_id, provider):
                mark = "🔒 "
            elif blocked or not configured:
                mark = "⚠️ "
            elif key == current:
                mark = "✅ "
            else:
                mark = ""
        lines.append(f"{mark}{info['label']}")
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{mark}{info['label']}",
                    callback_data=(
                        f"model:set:{key}" if allowed else f"model:locked:{key}"
                    ),
                )
            ]
        )
    owner_line = (
        "У тебя доступ ко всем моделям."
        if int(user_id) == OWNER_ID
        else "🔒 — доступ выдаёт владелец "
        f"<a href=\"https://t.me/{OWNER_USERNAME}\">@{OWNER_USERNAME}</a>."
    )
    blocked_notes = []
    if int(user_id) == OWNER_ID:
        for key, reason in _model_blocked.items():
            blocked_notes.append(
                f"⚠️ {esc(MODEL_REGISTRY[key]['label'])}: {esc(reason)}"
            )
        for key, info in MODEL_REGISTRY.items():
            provider = info["provider"]
            client = PROVIDER_CLIENTS.get(provider)
            if provider != "groq" and (client is None or not client.configured):
                blocked_notes.append(
                    f"⚠️ {esc(info['label'])}: нет API-ключа на сервере"
                )
    lines.append("")
    lines.append(owner_line)
    if blocked_notes:
        lines.append("")
        lines.extend(blocked_notes)
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


@dp.message(Command("models"))
async def cmd_models(message: Message):
    if not is_private(message) or not message.from_user:
        return
    text, kb = await models_panel_view(message.from_user.id, "models")
    await message.answer(text, reply_markup=kb)


@dp.callback_query(F.data.startswith("model:locked"))
async def on_model_locked(callback: CallbackQuery):
    parts = (callback.data or "").split(":")
    key = parts[2] if len(parts) > 2 else ""
    reason = _model_blocked.get(key)
    if reason:
        await callback.answer(f"Модель недоступна: {reason}.", show_alert=True)
        return
    info = MODEL_REGISTRY.get(key)
    if info and info["provider"] != "groq":
        provider = info["provider"]
        client = PROVIDER_CLIENTS.get(provider)
        if client is None or not client.configured:
            name = PROVIDER_NAMES.get(provider, provider)
            await callback.answer(
                f"{name} не настроен на сервере: нет API-ключа.", show_alert=True
            )
            return
    if callback.from_user and int(callback.from_user.id) == OWNER_ID:
        await callback.answer("Эта модель сейчас недоступна.", show_alert=True)
        return
    await callback.answer(
        f"Доступ выдаёт владелец @{OWNER_USERNAME} (t.me/{OWNER_USERNAME}).",
        show_alert=True,
    )


@dp.callback_query(F.data.startswith("models:"))
async def on_models_callback(callback: CallbackQuery):
    if not callback.from_user or callback.message is None:
        return
    if callback.message.chat.type != ChatType.PRIVATE:
        await callback.answer("Только в личке.", show_alert=True)
        return
    user_id = callback.from_user.id
    parts = (callback.data or "").split(":")
    action = parts[1] if len(parts) > 1 else ""
    tab = parts[-1] if action in ("toggle", "cycle") else "models"
    if action == "tab":
        tab = parts[2] if len(parts) > 2 else "models"
    elif action == "toggle":
        field = parts[2] if len(parts) > 2 else "rich"
        if field == "rich":
            await save_toggle(user_id, "rich", not user_rich(user_id))
    elif action == "cycle":
        field = parts[2] if len(parts) > 2 else ""
        if field == "deepthink":
            level = user_settings(user_id).get("deepthink", "off")
            index = (DEEPTHINK_ORDER.index(level) + 1) % len(DEEPTHINK_ORDER) if level in DEEPTHINK_ORDER else 1
            await save_toggle(user_id, "deepthink", DEEPTHINK_ORDER[index])
        elif field == "search":
            mode = user_search_mode(user_id)
            index = (SEARCH_ORDER.index(mode) + 1) % len(SEARCH_ORDER) if mode in SEARCH_ORDER else 0
            await save_toggle(user_id, "search", SEARCH_ORDER[index])
    text, kb = await models_panel_view(user_id, tab)
    try:
        await callback.message.edit_text(text, reply_markup=kb)
    except Exception as e:
        if "not modified" not in str(e).lower():
            try:
                await callback.message.answer(text, reply_markup=kb)
            except Exception:
                pass
    await callback.answer()


@dp.callback_query(F.data.startswith("model:set:"))
async def on_model_callback(callback: CallbackQuery):
    if not callback.from_user or not callback.message:
        return
    user_id = callback.from_user.id
    if callback.message.chat.type != ChatType.PRIVATE:
        await callback.answer("Модели доступны только в личке.", show_alert=True)
        return
    model = (callback.data or "").rsplit(":", 1)[-1]
    ok, text = await set_model_for_user(user_id, model)
    await callback.answer(("✅ " if ok else "⚠️ ") + plain_text(text)[:180], show_alert=not ok)
    if not ok:
        return
    panel, kb = await models_panel_view(user_id, "models")
    try:
        await callback.message.edit_text(panel, reply_markup=kb)
    except Exception:
        pass


@dp.message(Command("context"))
async def cmd_context(message: Message):
    if not command_allowed(message):
        return
    size = get_history_size(message.chat.id)
    await reply_any(message, f"📊 В контексте: <b>{size}</b> сообщений из {HISTORY_LIMIT}.")


@dp.message(Command("whoami"))
async def cmd_whoami(message: Message):
    if not command_allowed(message):
        return
    user = message.from_user
    if not user:
        return
    await reply_any(
        message,
        f"👤 <b>{esc(user.full_name)}</b>\n"
        f"ID: <code>{user.id}</code>\n"
        f"Username: @{user.username or '—'}",
    )


@dp.message(Command("stats"))
async def cmd_stats(message: Message):
    if not command_allowed(message):
        return
    chat_id = message.chat.id
    await reply_any(
        message,
        f"📊 <b>Статистика чата:</b>\n"
        f"• Обработано запросов: <b>{_stats.get(chat_id, 0)}</b>\n"
        f"• Сейчас в контексте: <b>{get_history_size(chat_id)}</b> сообщений",
    )


# ============================================================
# АДМИН-ПАНЕЛЬ
# ============================================================


def fmt_int(value) -> str:
    try:
        return f"{int(value):,}".replace(",", " ")
    except Exception:
        return str(value)


def admin_kb(rows: list[list[InlineKeyboardButton]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=rows)


def admin_filters(user_id: int) -> dict:
    return _admin_filter.setdefault(
        user_id,
        {"period": "7", "user_id": None, "model": None, "date_from": None},
    )


def admin_period_label(flt: dict) -> str:
    if flt.get("date_from"):
        return f"дата {flt['date_from']}"
    return {"t": "сегодня", "7": "7 дней", "a": "всё время"}.get(
        flt.get("period", "7"), "7 дней"
    )


def admin_period_bounds(flt: dict) -> tuple[datetime | None, datetime | None]:
    if flt.get("date_from"):
        try:
            day = date.fromisoformat(flt["date_from"])
            return msk_day_start(day), msk_day_start(day + timedelta(days=1))
        except Exception:
            pass
    period = flt.get("period", "7")
    today = msk_today()
    if period == "t":
        return msk_day_start(today), None
    if period == "a":
        return None, None
    return msk_day_start(today - timedelta(days=6)), None


def db_usage_line(size: int) -> str:
    mb = size / (1024 * 1024)
    percent = min(100.0, size / DB_LIMIT_BYTES * 100) if size else 0.0
    line = f"🗄 База: <b>{mb:.1f} МБ</b> из 1024 МБ ({percent:.1f}%)"
    if percent >= 80:
        line += "\n⚠️ <b>База заполнена более чем на 80%.</b> Почисти старые логи в «Настройках»."
    return line


def fmt_number(value) -> str:
    try:
        return f"{float(value):,.0f}".replace(",", " ")
    except Exception:
        return "—"


def balance_short() -> str:
    parts = []
    for key, name in (("deepseek", "DeepSeek"), ("qwen", "Qwen")):
        item = _balance.get(key) or {}
        if item.get("error"):
            parts.append(f"{name}: ошибка")
        elif item.get("amount") is None:
            parts.append(f"{name}: —")
        else:
            parts.append(f"{name}: {fmt_number(item['amount'])}")
    return " · ".join(parts)


def balance_block() -> str:
    lines = ["💳 <b>Балансы</b>"]
    for key, name in (("deepseek", "aikeysforyou (DeepSeek)"), ("qwen", "voidluv (Qwen)")):
        item = _balance.get(key) or {}
        if item.get("error"):
            lines.append(f"• {name}: ⚠️ {esc(str(item['error'])[:120])}")
            continue
        if item.get("amount") is None:
            lines.append(f"• {name}: нет данных")
            continue
        line = f"• {name}: <b>{fmt_number(item['amount'])}</b>"
        total = item.get("total")
        if total:
            percent = item["amount"] / total * 100
            line += f" из {fmt_number(total)} ({percent:.1f}%)"
        lines.append(line)
    updated = max(
        (
            float((_balance.get(k) or {}).get("updated") or 0)
            for k in ("deepseek", "qwen")
        ),
        default=0,
    )
    if updated:
        stamp = datetime.fromtimestamp(updated, MSK).strftime("%d.%m %H:%M")
        lines.append(f"Обновлено: {stamp} МСК")
    return "\n".join(lines)


def _page_stat(text: str, label: str) -> int | None:
    match = re.search(
        rf'<div class="k">{label}</div>\s*<div class="v">[^<]*</div>\s*'
        rf'<div class="n">([\d\s\u00a0]+)</div>',
        text,
    )
    if not match:
        return None
    digits = re.sub(r"[^\d]", "", match.group(1))
    return int(digits) if digits else None


async def refresh_balance(provider: str) -> dict:
    item: dict = {
        "amount": None,
        "total": None,
        "updated": time.time(),
        "error": "",
    }
    prev = _balance.get(provider) or {}
    try:
        if provider == "deepseek":
            if not deepseek_client.configured:
                raise RuntimeError("нет AIKEYS_API")
            response = await http_client.get(
                f"{DEEPSEEK_BASE_URL}/balance",
                headers={"Authorization": f"Bearer {AIKEYS_API_KEY}"},
                timeout=30,
            )
            response.raise_for_status()
            amount = float((response.json() or {}).get("token_balance") or 0)
            item["amount"] = amount
            item["total"] = max(float(prev.get("total") or 0), amount)
        else:
            response = await http_client.get(
                VOIDLUV_BALANCE_URL, timeout=30, follow_redirects=True
            )
            response.raise_for_status()
            total = _page_stat(response.text, "Лимит")
            left = _page_stat(response.text, "Осталось")
            if left is None:
                raise RuntimeError("не удалось разобрать страницу баланса")
            item["amount"] = float(left)
            item["total"] = float(total) if total else float(prev.get("total") or 0) or None
    except Exception as e:
        item["error"] = str(e)
        if prev.get("amount") is not None:
            item["amount"] = prev.get("amount")
            item["total"] = prev.get("total")
    _balance[provider] = item
    return item


async def check_balance_alert(provider: str) -> None:
    item = _balance.get(provider) or {}
    amount = item.get("amount")
    total = item.get("total")
    if amount is None:
        return
    name = PROVIDER_NAMES.get(provider, provider)
    if not total:
        return
    percent = amount / total * 100
    if percent >= BALANCE_ALERT_PERCENT:
        _balance_alerted.discard(provider)
        return
    if provider in _balance_alerted:
        return
    _balance_alerted.add(provider)
    try:
        await bot.send_message(
            OWNER_ID,
            f"⚠️ Баланс <b>{name}</b> ниже {BALANCE_ALERT_PERCENT}%: "
            f"{fmt_number(amount)} из {fmt_number(total)} ({percent:.1f}%).",
        )
    except Exception:
        pass


async def refresh_all_balances(notify: bool = False) -> None:
    for provider in ("deepseek", "qwen"):
        await refresh_balance(provider)
        await check_balance_alert(provider)
    if notify:
        try:
            await bot.send_message(OWNER_ID, balance_block())
        except Exception:
            pass


async def balances_loop() -> None:
    while True:
        try:
            await refresh_all_balances()
        except Exception:
            logging.exception("balance refresh failed")
        await asyncio.sleep(BALANCE_REFRESH_SECONDS)


async def probe_premium_models() -> None:
    """Проверяет, какие премиум-модели реально доступны на ключах."""
    for key, info in MODEL_REGISTRY.items():
        provider = info["provider"]
        if provider == "groq":
            continue
        client = PROVIDER_CLIENTS.get(provider)
        if client is None or not client.configured:
            continue
        old_timeout = client.timeout
        try:
            client.timeout = 12.0
            await client.chat(
                [{"role": "user", "content": "ping"}],
                model=info["model"],
                effort="off",
                max_tokens=8,
            )
            _model_blocked.pop(key, None)
        except Exception as e:
            text = str(e).lower()
            if "model_not_allowed" in text or "not enabled" in text or "permission" in text:
                _model_blocked[key] = "модель не включена для API-ключа"
                logging.warning(f"Модель {key} недоступна на ключе: {e}")
            else:
                logging.warning(f"Проверка модели {key} не удалась: {e}")
        finally:
            client.timeout = old_timeout


def admin_main_text() -> str:
    current = selected_model(OWNER_ID)
    ds_count = sum(1 for (_, p) in _grants if p == "deepseek")
    qwen_count = sum(1 for (_, p) in _grants if p == "qwen")
    return (
        "🛠 <b>Админ-панель Мудрого Мха</b>\n\n"
        f"Модель в твоей ЛС: <b>{esc(model_label(current))}</b>\n"
        f"Групп в белом списке: <b>{len(_groups_cache)}</b>\n"
        f"Заблокировано ЛС: <b>{len(_dm_blacklist)}</b>\n"
        f"Доступ DeepSeek: <b>{ds_count}</b> · Qwen: <b>{qwen_count}</b>\n"
        f"💰 {balance_short()}\n\n"
        f"{db_usage_line(_db_size_cache)}\n\n"
        "Выбери раздел:"
    )


def admin_main_kb() -> InlineKeyboardMarkup:
    return admin_kb(
        [
            [
                InlineKeyboardButton(text="📊 Статистика", callback_data="adm:stats:t"),
                InlineKeyboardButton(text="🧠 Модель", callback_data="adm:model"),
            ],
            [
                InlineKeyboardButton(text="👥 Группы", callback_data="adm:groups:0"),
                InlineKeyboardButton(text="🚫 ЛС", callback_data="adm:bl:0"),
            ],
            [
                InlineKeyboardButton(text="🔑 DeepSeek", callback_data="adm:acc:deepseek:0"),
                InlineKeyboardButton(text="🐳 Qwen", callback_data="adm:acc:qwen:0"),
            ],
            [
                InlineKeyboardButton(text="📝 Запросы", callback_data="adm:req:0"),
                InlineKeyboardButton(text="💳 Балансы", callback_data="adm:bal"),
            ],
            [InlineKeyboardButton(text="⚙️ Настройки", callback_data="adm:set")],
        ]
    )


def stats_text(stats: dict, period: str, flt: dict) -> str:
    lines = [f"📊 <b>Статистика</b> · {admin_period_label(flt)}", ""]
    lines.append(
        f"<b>A.</b> Запросов: <b>{fmt_int(stats['requests'])}</b> · "
        f"вызовов моделей: <b>{fmt_int(stats['calls'])}</b>"
    )
    lines.append("<b>B.</b> Токены по моделям:")
    models = stats.get("models") or []
    if not models:
        lines.append("• пока нет данных")
    for m in models[:8]:
        provider = m.get("provider") or "?"
        name = m.get("model") or "?"
        lines.append(
            f"• <code>{esc(provider)}/{esc(name)}</code> — "
            f"{fmt_int(m['calls'])} выз., {fmt_int(m['total_tokens'])} ток.\n"
            f"  prompt {fmt_int(m['prompt_tokens'])} · completion {fmt_int(m['completion_tokens'])} · "
            f"reasoning {fmt_int(m['reasoning_tokens'])} · cached {fmt_int(m['cached_tokens'])}"
        )
    lines.append("")
    lines.append("<b>C.</b> Топ пользователей:")
    if stats.get("top_users"):
        for u in stats["top_users"]:
            uname = f" @{esc(u['username'])}" if u.get("username") else ""
            lines.append(f"• <code>{u['user_id']}</code>{uname} — {fmt_int(u['requests'])}")
    else:
        lines.append("• пока нет")
    lines.append("Топ чатов:")
    if stats.get("top_chats"):
        for c in stats["top_chats"]:
            lines.append(f"• <code>{c['chat_id']}</code> ({esc(c['chat_type'])}) — {fmt_int(c['requests'])}")
    else:
        lines.append("• пока нет")
    ctx = stats.get("contexts") or {}
    lines.append("")
    lines.append(
        f"<b>D.</b> Контекст: <b>{fmt_int(ctx.get('chats', 0))}</b> чатов, "
        f"{fmt_int(ctx.get('messages', 0))} сообщений, "
        f"{fmt_int(ctx.get('chars', 0))} символов (~{fmt_int(ctx.get('tokens_est', 0))} ток.)"
    )
    lines.append("<b>E.</b> Инструменты:")
    if stats.get("tools"):
        lines.append(
            " • ".join(f"{esc(t['tool'])}: {fmt_int(t['count'])}" for t in stats["tools"])
        )
    else:
        lines.append("• пока нет")
    lines.append(
        f"<b>F.</b> Ошибки запросов: <b>{fmt_int(stats['request_errors'])}</b>, "
        f"ошибки моделей: <b>{fmt_int(stats['call_errors'])}</b>, "
        f"средняя задержка: <b>{fmt_int(stats['avg_latency'])} мс</b>"
    )
    lines.append(f"<b>G.</b> Кредиты/стоимость: <b>{stats['credits']:.4f}</b>")
    lines.append(
        f"\nСообщений в белых группах: <b>{fmt_int(stats['group_messages'])}</b>"
    )
    return "\n".join(lines)[:4000]


def stats_kb(period: str) -> InlineKeyboardMarkup:
    def label(code: str, text: str) -> str:
        return ("✅ " if period == code else "") + text

    return admin_kb(
        [
            [
                InlineKeyboardButton(text=label("t", "Сегодня"), callback_data="adm:stats:t"),
                InlineKeyboardButton(text=label("7", "7 дней"), callback_data="adm:stats:7"),
                InlineKeyboardButton(text=label("a", "Всё"), callback_data="adm:stats:a"),
            ],
            [
                InlineKeyboardButton(text="🔄 Обновить", callback_data=f"adm:stats:{period}"),
                InlineKeyboardButton(text="🏠 Меню", callback_data="adm:main"),
            ],
        ]
    )


def model_text() -> str:
    current = selected_model(OWNER_ID)
    lines = [
        "🧠 <b>Модель для твоей ЛС</b>",
        "",
        f"Сейчас: <b>{esc(model_label(current))}</b>",
        "",
    ]
    for key, info in MODEL_REGISTRY.items():
        note = ""
        if key in _model_blocked:
            note = f" · ⚠️ {esc(_model_blocked[key])}"
        lines.append(f"• <b>{esc(info['label'])}</b> · {info['provider']}{note}")
    lines += [
        "",
        "При переключении контекст, кэш фото и фотосессии сохраняются. "
        "Выбор запоминается после рестарта.",
    ]
    return "\n".join(lines)


def model_kb() -> InlineKeyboardMarkup:
    current = selected_model(OWNER_ID)
    rows = []
    for key, info in MODEL_REGISTRY.items():
        prefix = "⚠️ " if key in _model_blocked else ("✅ " if key == current else "")
        rows.append(
            [
                InlineKeyboardButton(
                    text=prefix + info["label"],
                    callback_data=f"adm:mset:{key}",
                )
            ]
        )
    rows.append([InlineKeyboardButton(text="🏠 Меню", callback_data="adm:main")])
    return admin_kb(rows)


def paginate(items: list, page: int) -> tuple[list, int, int]:
    total = max(1, (len(items) + ADMIN_PAGE_SIZE - 1) // ADMIN_PAGE_SIZE)
    page = max(0, min(page, total - 1))
    start = page * ADMIN_PAGE_SIZE
    return items[start: start + ADMIN_PAGE_SIZE], page, total


def paginator_kb(prefix: str, page: int, total: int, back: str = "adm:main") -> list[InlineKeyboardButton]:
    row = []
    if page > 0:
        row.append(InlineKeyboardButton(text="◀️", callback_data=f"{prefix}:{page - 1}"))
    row.append(InlineKeyboardButton(text=f"{page + 1}/{total}", callback_data="adm:noop"))
    if page + 1 < total:
        row.append(InlineKeyboardButton(text="▶️", callback_data=f"{prefix}:{page + 1}"))
    row.append(InlineKeyboardButton(text="🔄", callback_data=f"{prefix}:{page}"))
    row.append(InlineKeyboardButton(text="🏠", callback_data=back))
    return row


def groups_view(page: int) -> tuple[str, InlineKeyboardMarkup]:
    groups = sorted(_groups_cache.values(), key=lambda g: (g.get("title") or "").lower())
    page_items, page, total = paginate(groups, page)
    lines = [f"👥 <b>Белый список групп</b> · всего {len(groups)}", ""]
    if not groups:
        lines.append("Список пуст.")
    for g in page_items:
        lines.append(f"• <b>{esc(g.get('title') or 'без названия')}</b>\n  <code>{g['chat_id']}</code>")
    rows = [
        [InlineKeyboardButton(text="➕ Добавить группу", callback_data="adm:gadd")]
    ]
    for g in page_items:
        title = (g.get("title") or str(g["chat_id"]))[:28]
        rows.append(
            [InlineKeyboardButton(text=f"🗑 {title}", callback_data=f"adm:gdel:{g['chat_id']}")]
        )
    rows.append(paginator_kb("adm:groups", page, total))
    return "\n".join(lines), admin_kb(rows)


def blacklist_view(page: int) -> tuple[str, InlineKeyboardMarkup]:
    items = sorted(
        _dm_blacklist.values(),
        key=lambda x: x.get("added_at") or datetime.min.replace(tzinfo=timezone.utc),
    )
    page_items, page, total = paginate(items, page)
    lines = [f"🚫 <b>Чёрный список ЛС</b> · всего {len(items)}", ""]
    if not items:
        lines.append("Список пуст.")
    for u in page_items:
        name = u.get("full_name") or u.get("username") or "без имени"
        lines.append(f"• <b>{esc(name)}</b>\n  <code>{u['user_id']}</code>")
    rows = [[InlineKeyboardButton(text="➕ Заблокировать", callback_data="adm:bladd")]]
    for u in page_items:
        name = (u.get("full_name") or u.get("username") or str(u["user_id"]))[:25]
        rows.append(
            [InlineKeyboardButton(text=f"🗑 {name}", callback_data=f"adm:bldel:{u['user_id']}")]
        )
    rows.append(paginator_kb("adm:bl", page, total))
    return "\n".join(lines), admin_kb(rows)


async def access_view(provider: str, page: int) -> tuple[str, InlineKeyboardMarkup]:
    if provider not in PROVIDER_NAMES:
        provider = "deepseek"
    name = PROVIDER_NAMES[provider]
    limit = PROVIDER_LIMITS.get(provider, 0)
    today = msk_today()
    usage = await storage.usage_for_day(today, provider)
    items = sorted(
        (g for (uid, prov), g in _grants.items() if prov == provider),
        key=lambda x: x.get("granted_at") or datetime.min.replace(tzinfo=timezone.utc),
    )
    page_items, page, total = paginate(items, page)
    lines = [
        f"🔑 <b>Доступ к {name}</b>",
        "",
        f"👑 Владелец: <code>{OWNER_ID}</code> — безлимит",
        f"Другим: <b>{len(items)}</b> чел. · лимит {limit} сообщений/день",
        "",
    ]
    if not items:
        lines.append("Пока никому не выдан.")
    for u in page_items:
        item_name = u.get("full_name") or u.get("username") or "без имени"
        used = usage.get((int(u["user_id"]), provider), 0)
        lines.append(
            f"• <b>{esc(item_name)}</b> · <code>{u['user_id']}</code>\n"
            f"  сегодня: {used}/{limit}"
        )
    rows = [
        [
            InlineKeyboardButton(
                text=("• " if provider == "deepseek" else "") + "🔑 DeepSeek",
                callback_data="adm:acc:deepseek:0",
            ),
            InlineKeyboardButton(
                text=("• " if provider == "qwen" else "") + "🐳 Qwen",
                callback_data="adm:acc:qwen:0",
            ),
        ],
        [
            InlineKeyboardButton(
                text=f"➕ Выдать {name}", callback_data=f"adm:accadd:{provider}"
            )
        ],
    ]
    for u in page_items:
        item_name = (u.get("full_name") or u.get("username") or str(u["user_id"]))[:25]
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"🗑 {item_name}",
                    callback_data=f"adm:accdel:{provider}:{u['user_id']}",
                )
            ]
        )
    rows.append(paginator_kb(f"adm:acc:{provider}", page, total))
    return "\n".join(lines), admin_kb(rows)


async def requests_view_async(page: int, flt: dict) -> tuple[str, InlineKeyboardMarkup]:
    start, end = admin_period_bounds(flt)
    rows, total = await storage.list_requests(
        limit=ADMIN_PAGE_SIZE,
        offset=max(0, page) * ADMIN_PAGE_SIZE,
        user_id=flt.get("user_id"),
        model=flt.get("model"),
        start=start,
        end=end,
    )
    total_pages = max(1, (total + ADMIN_PAGE_SIZE - 1) // ADMIN_PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    if page * ADMIN_PAGE_SIZE >= total and total:
        rows, total = await storage.list_requests(
            limit=ADMIN_PAGE_SIZE,
            offset=page * ADMIN_PAGE_SIZE,
            user_id=flt.get("user_id"),
            model=flt.get("model"),
            start=start,
            end=end,
        )

    lines = [
        f"📝 <b>Последние запросы</b> · {admin_period_label(flt)}",
        f"Фильтры: пользователь <code>{flt.get('user_id') or 'все'}</code>, "
        f"модель <code>{flt.get('model') or 'все'}</code>",
        f"Всего: <b>{fmt_int(total)}</b>",
        "",
    ]
    if not rows:
        lines.append("Ничего не найдено.")
    for r in rows:
        created = r.get("created_at")
        if isinstance(created, datetime):
            created = created.astimezone(MSK).strftime("%d.%m %H:%M")
        status = r.get("status") or "?"
        if status == "logged":
            icon = "📝"
        elif status == "ok":
            icon = "✅"
        elif status == "running":
            icon = "⏳"
        else:
            icon = "❌"
        uname = f" @{r.get('username')}" if r.get("username") else ""
        prompt = (r.get("prompt_preview") or "").replace("\n", " ")[:100]
        lines.append(
            f"{icon} <b>#{r.get('id')}</b> {created} · {esc(r.get('kind') or '?')} · "
            f"{esc(r.get('model') or '—')}\n"
            f"  <code>{r.get('user_id') or '—'}</code>{esc(uname)} · "
            f"чат <code>{r.get('chat_id') or '—'}</code> · "
            f"{fmt_int(r.get('tokens') or 0)} ток."
            + (f" · {fmt_int(r.get('latency_ms') or 0)} мс" if r.get("latency_ms") else "")
            + f"\n  {esc(prompt)}"
        )
        if r.get("tools"):
            lines.append(f"  🛠 {esc(r['tools'])}")
        if r.get("error"):
            lines.append(f"  ⚠️ {esc((r.get('error') or '')[:120])}")
    rows_kb = [
        [
            InlineKeyboardButton(text="🔎 Фильтры", callback_data="adm:filter"),
            InlineKeyboardButton(text="⬇️ CSV", callback_data="adm:reqcsv"),
        ],
        [
            InlineKeyboardButton(text="🔄 Обновить", callback_data=f"adm:req:{page}"),
            InlineKeyboardButton(text="🏠 Меню", callback_data="adm:main"),
        ],
        paginator_kb("adm:req", page, total_pages),
    ]
    return "\n".join(lines)[:4000], admin_kb(rows_kb)


def filter_view(flt: dict) -> tuple[str, InlineKeyboardMarkup]:
    text = (
        "🔎 <b>Фильтры запросов</b>\n\n"
        f"Период: <b>{admin_period_label(flt)}</b>\n"
        f"Пользователь: <code>{flt.get('user_id') or 'все'}</code>\n"
        f"Модель: <code>{flt.get('model') or 'все'}</code>"
    )

    def mark(value, current, label):
        return ("✅ " if value == current else "") + label

    rows = [
        [
            InlineKeyboardButton(text="Сегодня", callback_data="adm:fperiod:t"),
            InlineKeyboardButton(text="7 дней", callback_data="adm:fperiod:7"),
            InlineKeyboardButton(text="Всё", callback_data="adm:fperiod:a"),
        ],
        [
            InlineKeyboardButton(text="👤 Пользователь", callback_data="adm:fuser"),
            InlineKeyboardButton(text="📅 Дата", callback_data="adm:fdate"),
        ],
        [
            InlineKeyboardButton(text=mark(None, flt.get("model"), "Все модели"), callback_data="adm:fmodel:all"),
            InlineKeyboardButton(text=mark("groq", flt.get("model"), "Groq"), callback_data="adm:fmodel:groq"),
            InlineKeyboardButton(text=mark("deepseek", flt.get("model"), "deepseek-v4.1-flash"), callback_data="adm:fmodel:deepseek"),
        ],
        [
            InlineKeyboardButton(text=mark("deepseek-pro", flt.get("model"), "deepseek-v4-pro"), callback_data="adm:fmodel:deepseek-pro"),
            InlineKeyboardButton(text=mark("qwen4-max", flt.get("model"), "qwen4-max"), callback_data="adm:fmodel:qwen4-max"),
        ],
        [
            InlineKeyboardButton(text=mark("qwen3.8-max", flt.get("model"), "qwen3.8-max"), callback_data="adm:fmodel:qwen3.8-max"),
            InlineKeyboardButton(text=mark("qwen3.8-flash", flt.get("model"), "qwen3.8-flash"), callback_data="adm:fmodel:qwen3.8-flash"),
        ],
        [
            InlineKeyboardButton(text="🗑 Сбросить", callback_data="adm:fclear"),
            InlineKeyboardButton(text="📝 К запросам", callback_data="adm:req:0"),
        ],
    ]
    return text, admin_kb(rows)


async def settings_view() -> tuple[str, InlineKeyboardMarkup]:
    size = await storage.db_size()
    ctx = await storage.context_summary()
    mode = "Postgres (Neon)" if storage.available else "память (DATABASE_URL не задан)"
    text = (
        "⚙️ <b>Настройки</b>\n\n"
        f"Хранилище: <b>{mode}</b>\n"
        f"{db_usage_line(size)}\n\n"
        f"Контекст: {fmt_int(ctx['chats'])} чатов, {fmt_int(ctx['messages'])} сообщений\n"
        f"Логи: <b>хранятся вечно</b> (ручная очистка ниже)\n"
        f"Сводка: <b>каждый день в 04:00 МСК</b>\n"
        f"Лимиты: DeepSeek <b>{PROVIDER_LIMITS['deepseek']}</b>, "
        f"Qwen <b>{PROVIDER_LIMITS['qwen']}</b> сообщений/день\n"
        f"Контекст премиум-моделей: {fmt_int(PREMIUM_HISTORY_CHARS_LIMIT)} символов, "
        f"ответ до {fmt_int(PREMIUM_MAX_TOKENS)} токенов\n\n"
        f"{balance_block()}\n\n"
        "Очистка удаляет запросы, вызовы моделей и инструментов старше указанного срока. "
        "Контексты и списки не трогает."
    )
    rows = [
        [InlineKeyboardButton(text="💳 Обновить балансы", callback_data="adm:balrefresh")],
        [InlineKeyboardButton(text="⬇️ CSV всех запросов", callback_data="adm:csvall")],
        [InlineKeyboardButton(text="🧹 Очистить старые логи", callback_data="adm:cleanup")],
        [
            InlineKeyboardButton(text="🔄 Обновить", callback_data="adm:set"),
            InlineKeyboardButton(text="🏠 Меню", callback_data="adm:main"),
        ],
    ]
    return text, admin_kb(rows)


async def admin_section(
    section: str, page: int = 0, period: str | None = None, provider: str | None = None
) -> tuple[str, InlineKeyboardMarkup]:
    global _db_size_cache
    if section == "main":
        _db_size_cache = await storage.db_size()
        return admin_main_text(), admin_main_kb()
    if section == "model":
        return model_text(), model_kb()
    if section == "groups":
        return groups_view(page)
    if section == "bl":
        return blacklist_view(page)
    if section == "acc":
        return await access_view(provider or "deepseek", page)
    if section == "set":
        return await settings_view()
    if section == "filter":
        return filter_view(admin_filters(OWNER_ID))
    if section == "req":
        return await requests_view_async(page, admin_filters(OWNER_ID))
    if section == "stats":
        flt = admin_filters(OWNER_ID)
        if period:
            flt["period"] = period
            flt["date_from"] = None
        start, end = admin_period_bounds(flt)
        data = await storage.stats(start, end)
        return stats_text(data, flt.get("period", "7"), flt), stats_kb(flt.get("period", "7"))
    return admin_main_text(), admin_main_kb()


async def edit_admin(callback: CallbackQuery, text: str, kb: InlineKeyboardMarkup) -> None:
    if callback.message is None:
        return
    try:
        await callback.message.edit_text(text, reply_markup=kb)
    except Exception:
        try:
            await callback.message.answer(text, reply_markup=kb)
        except Exception:
            pass


def confirm_kb(ok_callback: str, back_callback: str) -> InlineKeyboardMarkup:
    return admin_kb(
        [
            [
                InlineKeyboardButton(text="✅ Да, удалить", callback_data=ok_callback),
                InlineKeyboardButton(text="↩️ Отмена", callback_data=back_callback),
            ]
        ]
    )


@dp.message(Command("admin"))
async def cmd_admin(message: Message):
    if not is_private(message) or not message.from_user:
        return
    if message.from_user.id != OWNER_ID:
        return
    _admin_input.pop(OWNER_ID, None)
    admin_filters(OWNER_ID)
    text, kb = await admin_section("main")
    await message.answer(text, reply_markup=kb)


@dp.message(Command("cancel"))
async def cmd_cancel(message: Message):
    if not is_private(message) or not message.from_user or message.from_user.id != OWNER_ID:
        return
    if _admin_input.pop(OWNER_ID, None):
        await message.answer("Отменено.", reply_markup=admin_main_kb())
    else:
        await message.answer("Нечего отменять.")


def _forwarded_chat(message: Message) -> tuple[int, str] | None:
    chat = getattr(message, "forward_from_chat", None)
    if chat is not None:
        return int(chat.id), getattr(chat, "title", "") or ""
    origin = getattr(message, "forward_origin", None)
    if origin is not None:
        origin_chat = getattr(origin, "chat", None)
        if origin_chat is not None:
            return int(origin_chat.id), getattr(origin_chat, "title", "") or ""
        sender_chat = getattr(origin, "sender_chat", None)
        if sender_chat is not None:
            return int(sender_chat.id), getattr(sender_chat, "title", "") or ""
    return None


def _forwarded_user(message: Message) -> tuple[int, str, str] | None:
    user = getattr(message, "forward_from", None)
    if user is not None:
        return int(user.id), user.username or "", user.full_name or ""
    origin = getattr(message, "forward_origin", None)
    if origin is not None:
        sender = getattr(origin, "sender_user", None)
        if sender is not None:
            return int(sender.id), sender.username or "", sender.full_name or ""
    return None


def _parse_int(text: str) -> int | None:
    match = re.search(r"-?\d+", text or "")
    if not match:
        return None
    try:
        return int(match.group(0))
    except Exception:
        return None


async def maybe_admin_input(message: Message) -> bool:
    """Перехватывает ввод для админ-панели. Возвращает True, если сообщение съедено."""
    if not is_private(message) or not message.from_user:
        return False
    if message.from_user.id != OWNER_ID:
        return False
    state = _admin_input.get(OWNER_ID)
    if not state:
        return False

    text = (message.text or "").strip()
    if text.lower() in ("отмена", "cancel") or text.startswith("/"):
        _admin_input.pop(OWNER_ID, None)
        await message.answer("Отменено.", reply_markup=admin_main_kb())
        return True

    action = state.get("action")
    try:
        if action == "group_id":
            forwarded = _forwarded_chat(message)
            if forwarded:
                chat_id, title = forwarded
            else:
                chat_id = _parse_int(text)
                title = ""
            if chat_id is None or chat_id >= 0:
                await message.answer(
                    "Нужен ID группы (отрицательное число, например <code>-100...</code>) "
                    "или пересланное сообщение из группы. Попробуй ещё раз.",
                )
                return True
            await storage.add_group(chat_id, title or f"Группа {chat_id}", OWNER_ID)
            ALLOWED_GROUPS.add(chat_id)
            _groups_cache[chat_id] = {
                "chat_id": chat_id,
                "title": title or f"Группа {chat_id}",
                "added_at": msk_now(),
            }
            await storage.add_admin_action(OWNER_ID, "group_add", f"{chat_id} {title}")
            _admin_input.pop(OWNER_ID, None)
            await message.answer(
                f"✅ Группа <b>{esc(title or chat_id)}</b> добавлена в белый список.\n"
                "Бот начнёт отвечать, когда его позовут.",
                reply_markup=admin_main_kb(),
            )
            return True

        if action == "blacklist_id":
            forwarded = _forwarded_user(message)
            if forwarded:
                user_id, uname, full_name = forwarded
            else:
                user_id = _parse_int(text)
                uname, full_name = "", ""
            if user_id is None or user_id <= 0:
                await message.answer(
                    "Нужен ID пользователя (положительное число) или пересланное сообщение от него."
                )
                return True
            await storage.add_blacklist(user_id, uname, full_name, OWNER_ID)
            _dm_blacklist[user_id] = {
                "user_id": user_id,
                "username": uname,
                "full_name": full_name,
                "added_at": msk_now(),
            }
            await clear_chat_context(user_id)
            await storage.add_admin_action(OWNER_ID, "blacklist_add", str(user_id))
            _admin_input.pop(OWNER_ID, None)
            await message.answer(
                f"✅ Пользователь <code>{user_id}</code> заблокирован в ЛС. "
                "Его контекст удалён. В группах он не блокируется.",
                reply_markup=admin_main_kb(),
            )
            return True

        if action == "grant_id":
            forwarded = _forwarded_user(message)
            if forwarded:
                user_id, uname, full_name = forwarded
            else:
                user_id = _parse_int(text)
                uname, full_name = "", ""
            if user_id is None or user_id <= 0:
                await message.answer(
                    "Нужен ID пользователя (положительное число) или пересланное сообщение от него."
                )
                return True
            if user_id == OWNER_ID:
                await message.answer("У владельца доступ уже есть.")
                return True
            provider = state.get("provider") or "deepseek"
            if provider not in PROVIDER_NAMES:
                provider = "deepseek"
            provider_name = PROVIDER_NAMES[provider]
            await storage.add_grant(
                user_id, provider, uname, full_name, OWNER_ID
            )
            _grants[(user_id, provider)] = {
                "user_id": user_id,
                "provider": provider,
                "username": uname,
                "full_name": full_name,
                "granted_at": msk_now(),
            }
            await refresh_user_commands(user_id)
            await storage.add_admin_action(
                OWNER_ID, "grant_add", f"{provider}:{user_id}"
            )
            _admin_input.pop(OWNER_ID, None)
            limit = PROVIDER_LIMITS.get(provider, 0)
            try:
                await bot.send_message(
                    user_id,
                    f"🔑 Тебе выдан доступ к {provider_name}.\n"
                    "Открой /models в личке и выбери модель. "
                    f"Лимит: {limit} сообщений в день.",
                )
            except Exception:
                pass
            await message.answer(
                f"✅ Доступ к {provider_name} выдан <code>{user_id}</code>.",
                reply_markup=admin_main_kb(),
            )
            return True

        if action == "cleanup_days":
            days = _parse_int(text)
            if days is None or days < 1 or days > 3650:
                await message.answer("Введи число дней от 1 до 3650.")
                return True
            _admin_input.pop(OWNER_ID, None)
            await message.answer(
                f"Удалить логи старше <b>{days}</b> дней? Контексты и списки останутся.",
                reply_markup=confirm_kb(f"adm:cleanok:{days}", "adm:set"),
            )
            return True

        if action == "req_user":
            user_id = _parse_int(text)
            if user_id is None:
                await message.answer("Нужен числовой ID пользователя или 0, чтобы сбросить.")
                return True
            flt = admin_filters(OWNER_ID)
            flt["user_id"] = None if user_id == 0 else user_id
            _admin_input.pop(OWNER_ID, None)
            txt, kb = await requests_view_async(0, flt)
            await message.answer(txt, reply_markup=kb)
            return True

        if action == "req_date":
            raw = text.strip()
            day = None
            for fmt in ("%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d"):
                try:
                    day = datetime.strptime(raw, fmt).date()
                    break
                except Exception:
                    continue
            if day is None:
                await message.answer("Введи дату как <code>22.11.2026</code> или <code>2026-11-22</code>.")
                return True
            flt = admin_filters(OWNER_ID)
            flt["date_from"] = day.isoformat()
            _admin_input.pop(OWNER_ID, None)
            txt, kb = await requests_view_async(0, flt)
            await message.answer(txt, reply_markup=kb)
            return True
    except Exception as e:
        logging.exception("Admin input error")
        _admin_input.pop(OWNER_ID, None)
        await message.answer(
            f"⚠️ Ошибка: <code>{esc(e)}</code>", reply_markup=admin_main_kb()
        )
        return True
    return False


@dp.callback_query(F.data.startswith("adm:"))
async def on_admin_callback(callback: CallbackQuery):
    if not callback.from_user or callback.message is None:
        return
    if callback.from_user.id != OWNER_ID:
        await callback.answer("Нет доступа.", show_alert=True)
        return
    if callback.message.chat.type != ChatType.PRIVATE:
        await callback.answer("Админ-панель работает только в ЛС.", show_alert=True)
        return

    data = callback.data or ""
    parts = data.split(":")
    action = parts[1] if len(parts) > 1 else ""

    try:
        # Любая кнопка панели отменяет незавершённый текстовый ввод.
        _admin_input.pop(OWNER_ID, None)
        if action == "noop":
            await callback.answer()
            return
        if action == "main":
            _admin_input.pop(OWNER_ID, None)
            text, kb = await admin_section("main")
            await edit_admin(callback, text, kb)
            await callback.answer()
            return
        if action == "stats":
            period = parts[2] if len(parts) > 2 else "t"
            text, kb = await admin_section("stats", period=period)
            await edit_admin(callback, text, kb)
            await callback.answer()
            return
        if action == "model":
            text, kb = await admin_section("model")
            await edit_admin(callback, text, kb)
            await callback.answer()
            return
        if action == "mset":
            model = parts[2] if len(parts) > 2 else "groq"
            ok, msg = await set_model_for_user(OWNER_ID, model)
            await callback.answer(plain_text(msg)[:180], show_alert=not ok)
            if ok:
                await storage.add_admin_action(OWNER_ID, "model_set", model)
            text, kb = await admin_section("model")
            await edit_admin(callback, text, kb)
            return
        if action == "groups":
            page = int(parts[2]) if len(parts) > 2 else 0
            text, kb = await admin_section("groups", page)
            await edit_admin(callback, text, kb)
            await callback.answer()
            return
        if action == "bl":
            page = int(parts[2]) if len(parts) > 2 else 0
            text, kb = await admin_section("bl", page)
            await edit_admin(callback, text, kb)
            await callback.answer()
            return
        if action == "acc":
            page = int(parts[3]) if len(parts) > 3 else 0
            provider = parts[2] if len(parts) > 2 else "deepseek"
            text, kb = await admin_section("acc", page, provider=provider)
            await edit_admin(callback, text, kb)
            await callback.answer()
            return
        if action == "bal":
            await refresh_all_balances()
            text = balance_block()
            kb = admin_kb(
                [
                    [InlineKeyboardButton(text="🔄 Обновить", callback_data="adm:bal")],
                    [InlineKeyboardButton(text="🏠 Меню", callback_data="adm:main")],
                ]
            )
            await edit_admin(callback, text, kb)
            await callback.answer("Балансы обновлены.")
            return
        if action == "balrefresh":
            await refresh_all_balances()
            text, kb = await admin_section("set")
            await edit_admin(callback, text, kb)
            await callback.answer("Балансы обновлены.")
            return
        if action == "req":
            page = int(parts[2]) if len(parts) > 2 else 0
            text, kb = await admin_section("req", page)
            await edit_admin(callback, text, kb)
            await callback.answer()
            return
        if action == "set":
            text, kb = await admin_section("set")
            await edit_admin(callback, text, kb)
            await callback.answer()
            return
        if action == "filter":
            text, kb = filter_view(admin_filters(OWNER_ID))
            await edit_admin(callback, text, kb)
            await callback.answer()
            return

        if action == "gadd":
            _admin_input[OWNER_ID] = {"action": "group_id"}
            await callback.answer()
            await callback.message.answer(
                "Отправь ID группы (например <code>-1001234567890</code>) "
                "или перешли любое сообщение из этой группы.\n\n/cancel — отмена."
            )
            return
        if action == "gdel":
            chat_id = int(parts[2])
            title = (_groups_cache.get(chat_id) or {}).get("title") or str(chat_id)
            await edit_admin(
                callback,
                f"Удалить группу <b>{esc(title)}</b> из белого списка?\n\n"
                "Бот останется в группе, но перестанет отвечать. Контекст группы будет удалён.",
                confirm_kb(f"adm:gdelok:{chat_id}", "adm:groups:0"),
            )
            await callback.answer()
            return
        if action == "gdelok":
            chat_id = int(parts[2])
            await storage.remove_group(chat_id)
            ALLOWED_GROUPS.discard(chat_id)
            _groups_cache.pop(chat_id, None)
            await clear_chat_context(chat_id)
            await storage.add_admin_action(OWNER_ID, "group_remove", str(chat_id))
            text, kb = await admin_section("groups")
            await edit_admin(callback, text, kb)
            await callback.answer("Группа убрана из белого списка.")
            return

        if action == "bladd":
            _admin_input[OWNER_ID] = {"action": "blacklist_id"}
            await callback.answer()
            await callback.message.answer(
                "Отправь ID пользователя или перешли его сообщение.\n\n/cancel — отмена."
            )
            return
        if action == "bldel":
            user_id = int(parts[2])
            item = _dm_blacklist.get(user_id) or {}
            name = item.get("full_name") or item.get("username") or str(user_id)
            await edit_admin(
                callback,
                f"Разблокировать <b>{esc(name)}</b> в ЛС?",
                confirm_kb(f"adm:bldelok:{user_id}", "adm:bl:0"),
            )
            await callback.answer()
            return
        if action == "bldelok":
            user_id = int(parts[2])
            await storage.remove_blacklist(user_id)
            _dm_blacklist.pop(user_id, None)
            await storage.add_admin_action(OWNER_ID, "blacklist_remove", str(user_id))
            text, kb = await admin_section("bl")
            await edit_admin(callback, text, kb)
            await callback.answer("Разблокирован.")
            return

        if action == "accadd":
            provider = parts[2] if len(parts) > 2 else "deepseek"
            if provider not in PROVIDER_NAMES:
                provider = "deepseek"
            provider_name = PROVIDER_NAMES[provider]
            _admin_input[OWNER_ID] = {"action": "grant_id", "provider": provider}
            await callback.answer()
            await callback.message.answer(
                f"Отправь ID пользователя или перешли его сообщение, "
                f"чтобы выдать {provider_name}.\n\n/cancel — отмена."
            )
            return
        if action == "accdel":
            provider = parts[2] if len(parts) > 2 else "deepseek"
            user_id = int(parts[3]) if len(parts) > 3 else 0
            item = _grants.get((user_id, provider)) or {}
            item_name = item.get("full_name") or item.get("username") or str(user_id)
            provider_name = PROVIDER_NAMES.get(provider, provider)
            await edit_admin(
                callback,
                f"Отозвать доступ к {provider_name} у <b>{esc(item_name)}</b>?\n\n"
                "Модель переключится на Groq, контекст сохранится.",
                confirm_kb(
                    f"adm:accdelok:{provider}:{user_id}",
                    f"adm:acc:{provider}:0",
                ),
            )
            await callback.answer()
            return
        if action == "accdelok":
            provider = parts[2] if len(parts) > 2 else "deepseek"
            user_id = int(parts[3]) if len(parts) > 3 else 0
            provider_name = PROVIDER_NAMES.get(provider, provider)
            await storage.remove_grant(user_id, provider)
            _grants.pop((user_id, provider), None)
            if provider_of(selected_model(user_id)) == provider:
                await storage.save_user_settings(user_id, model="groq")
            await storage.add_admin_action(
                OWNER_ID, "grant_remove", f"{provider}:{user_id}"
            )
            await refresh_user_commands(user_id)
            try:
                await bot.send_message(
                    user_id,
                    f"🔒 Доступ к {provider_name} отозван. "
                    "Модель переключена на Groq, контекст сохранён.",
                )
            except Exception:
                pass
            text, kb = await admin_section("acc", 0, provider=provider)
            await edit_admin(callback, text, kb)
            await callback.answer("Доступ отозван.")
            return

        if action == "fperiod":
            period = parts[2] if len(parts) > 2 else "7"
            flt = admin_filters(OWNER_ID)
            flt["period"] = period if period in ("t", "7", "a") else "7"
            flt["date_from"] = None
            text, kb = filter_view(flt)
            await edit_admin(callback, text, kb)
            await callback.answer()
            return
        if action == "fmodel":
            model = parts[2] if len(parts) > 2 else "all"
            flt = admin_filters(OWNER_ID)
            flt["model"] = None if model == "all" else model
            text, kb = filter_view(flt)
            await edit_admin(callback, text, kb)
            await callback.answer()
            return
        if action == "fuser":
            _admin_input[OWNER_ID] = {"action": "req_user"}
            await callback.answer()
            await callback.message.answer("Введи ID пользователя или 0, чтобы сбросить.")
            return
        if action == "fdate":
            _admin_input[OWNER_ID] = {"action": "req_date"}
            await callback.answer()
            await callback.message.answer(
                "Введи дату: <code>22.11.2026</code> или <code>2026-11-22</code>."
            )
            return
        if action == "fclear":
            _admin_filter[OWNER_ID] = {
                "period": "7",
                "user_id": None,
                "model": None,
                "date_from": None,
            }
            text, kb = filter_view(admin_filters(OWNER_ID))
            await edit_admin(callback, text, kb)
            await callback.answer("Фильтры сброшены.")
            return

        if action == "reqcsv":
            flt = admin_filters(OWNER_ID)
            start, end = admin_period_bounds(flt)
            csv_text = await storage.export_requests(
                user_id=flt.get("user_id"),
                model=flt.get("model"),
                start=start,
                end=end,
            )
            filename = f"mox_requests_{msk_now().strftime('%Y%m%d_%H%M')}.csv"
            await bot.send_document(
                callback.message.chat.id,
                BufferedInputFile(csv_text.encode("utf-8"), filename=filename),
                caption="📄 Экспорт запросов по текущим фильтрам.",
            )
            await callback.answer("CSV отправлен.")
            return
        if action == "csvall":
            csv_text = await storage.export_requests(limit=50000)
            filename = f"mox_all_requests_{msk_now().strftime('%Y%m%d_%H%M')}.csv"
            await bot.send_document(
                callback.message.chat.id,
                BufferedInputFile(csv_text.encode("utf-8"), filename=filename),
                caption="📄 Все запросы (до 50 000 строк).",
            )
            await callback.answer("CSV отправлен.")
            return
        if action == "cleanup":
            _admin_input[OWNER_ID] = {"action": "cleanup_days"}
            await callback.answer()
            await callback.message.answer("За сколько дней удалить логи? Введи число, например 30.")
            return
        if action == "cleanok":
            days = int(parts[2])
            deleted = await storage.cleanup_logs(days)
            await storage.add_admin_action(OWNER_ID, "logs_cleanup", f"{days} дней")
            text, kb = await admin_section("set")
            await edit_admin(
                callback,
                f"✅ Удалено записей: <b>{fmt_int(deleted)}</b> (старше {days} дней).\n\n" + text,
                kb,
            )
            await callback.answer("Готово.")
            return

        await callback.answer()
    except Exception as e:
        logging.exception("Admin callback error")
        await callback.answer(plain_text(f"Ошибка: {e}")[:180], show_alert=True)


async def send_daily_summary() -> None:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=24)
    data = await storage.stats(start, end)
    lines = [
        "📊 <b>Сводка за 24 часа</b>",
        "",
        f"Запросов: <b>{fmt_int(data['requests'])}</b>",
        f"Вызовов моделей: <b>{fmt_int(data['calls'])}</b>",
        f"Ошибок: <b>{fmt_int(data['request_errors'] + data['call_errors'])}</b>",
    ]
    for m in (data.get("models") or [])[:6]:
        lines.append(
            f"• <code>{esc(m['model'])}</code>: {fmt_int(m['total_tokens'])} ток."
        )
    if data.get("top_users"):
        lines.append("\nТоп пользователей:")
        for u in data["top_users"][:5]:
            lines.append(f"• <code>{u['user_id']}</code> — {fmt_int(u['requests'])}")
    lines.append(f"\nКредиты: <b>{data['credits']:.4f}</b>")
    try:
        await bot.send_message(OWNER_ID, "\n".join(lines))
    except Exception as e:
        logging.warning(f"Daily summary send failed: {e}")
    await storage.set_setting("last_daily_summary", msk_now().isoformat())


async def daily_summary_loop() -> None:
    # Догоняем сводку, если бот был перезапущен после 04:00 и она не ушла.
    try:
        now = msk_now()
        today_four = now.replace(hour=4, minute=0, second=0, microsecond=0)
        raw = await storage.get_setting("last_daily_summary")
        last: datetime | None = None
        if raw:
            try:
                last = datetime.fromisoformat(raw)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=MSK)
                last = last.astimezone(MSK)
            except ValueError:
                last = None
        if last is None:
            await storage.set_setting("last_daily_summary", now.isoformat())
        elif now >= today_four and last < today_four:
            logging.info("Отправляю пропущенную ежедневную сводку")
            await send_daily_summary()
    except Exception:
        logging.exception("Daily summary catch-up failed")

    while True:
        try:
            now = msk_now()
            target = now.replace(hour=4, minute=0, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            wait = (target - now).total_seconds()
            logging.info(f"Ежедневная сводка через {wait / 3600:.1f} ч")
            await asyncio.sleep(wait)
            await send_daily_summary()
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception("Daily summary error")
            await asyncio.sleep(300)


class DMBlacklistMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        if (
            isinstance(event, Message)
            and event.chat.type == ChatType.PRIVATE
            and event.from_user
            and is_blacklisted(event.from_user.id)
        ):
            await record_blocked(event)
            now = time.time()
            if now - _block_reply_last.get(event.from_user.id, 0) >= BLOCK_REPLY_COOLDOWN:
                _block_reply_last[event.from_user.id] = now
                try:
                    await event.answer("доступ закрыт")
                except Exception:
                    pass
            return
        return await handler(event, data)


# ============================================================
# СЛУЖЕБНЫЕ СОБЫТИЯ
# ============================================================


@dp.my_chat_member()
async def on_my_chat_member(update: ChatMemberUpdated):
    chat = update.chat
    if chat.type not in GROUP_TYPES:
        return
    old_status = update.old_chat_member.status
    new_status = update.new_chat_member.status
    if old_status in ("left", "kicked") and new_status in ("member", "administrator"):
        if chat.id in ALLOWED_GROUPS:
            logging.info(f"Добавлен в разрешённую группу {chat.title} ({chat.id})")
            return
        logging.info(f"Добавлен в неизвестную группу {chat.title} ({chat.id})")
        try:
            await bot.send_message(
                OWNER_ID,
                "Меня добавили в новую группу.\n"
                f"Название: <b>{esc(chat.title)}</b>\n"
                f"ID: <code>{chat.id}</code>\n"
                "Чтобы включить бота, добавь группу через /admin.",
            )
        except Exception as e:
            logging.warning(f"Owner notify failed: {e}")


@dp.message(F.new_chat_members)
async def on_users_join(message: Message):
    if is_private(message) or not is_whitelisted(message.chat.id):
        return
    for user in message.new_chat_members or []:
        if BOT_ID is not None and user.id == BOT_ID:
            continue
        await add_to_history(message.chat.id, "user", f"[в чат вошёл {display_user(user)}]")


@dp.message(F.left_chat_member)
async def on_user_leave(message: Message):
    if is_private(message) or not is_whitelisted(message.chat.id):
        return
    user = message.left_chat_member
    if user is None:
        return
    if BOT_ID is not None and user.id == BOT_ID:
        return
    await add_to_history(message.chat.id, "user", f"[из чата вышел {display_user(user)}]")


# ============================================================
# INLINE
# ============================================================


async def simple_answer(prompt: str, *, track: Track | None = None) -> str:
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": prompt},
    ]

    async def executor(name: str, args: dict) -> str:
        if name == "web_search":
            query = (args.get("query") or "").strip()
            return render_results(await search_web(query))
        return "Инструмент недоступен."

    forced = None
    if has_search_verb(prompt):
        forced = {"type": "function", "function": {"name": "web_search"}}

    return await run_tools(
        messages, SEARCH_TOOLS, executor, forced, provider="groq", track=track, effort="low"
    )


@dp.inline_query()
async def on_inline(query: InlineQuery):
    prompt = (query.query or "").strip()

    if not prompt:
        await query.answer(
            results=[
                InlineQueryResultArticle(
                    id="hint",
                    title="Напиши запрос после @M0XAIBOT",
                    description="Например: @M0XAIBOT расскажи про Python",
                    input_message_content=InputTextMessageContent(
                        message_text="Пример: @M0XAIBOT расскажи про Python"
                    ),
                )
            ],
            cache_time=0,
            is_personal=True,
        )
        return

    user = query.from_user
    request_id = await storage.create_request(
        user_id=user.id if user else None,
        username=username_of(user),
        chat_id=None,
        chat_type="inline",
        kind="inline",
        is_request=True,
        model="groq",
        prompt_preview=prompt,
    )
    track = Track(request_id=request_id, user_id=user.id if user else None, chat_id=None)
    started = time.perf_counter()
    status = "ok"
    error_text = ""
    try:
        answer = await simple_answer(prompt, track=track)
    except Exception as e:
        logging.exception("Inline error")
        status = "error"
        error_text = str(e)
        answer = f"⚠️ Ошибка: {e}"
    finally:
        await finish_track(
            track,
            status=status,
            error=error_text,
            latency_ms=int((time.perf_counter() - started) * 1000),
            model="groq",
        )

    answer = sanitize_html(answer)[:3800] or "Пустой ответ."
    # Для inline надёжнее отправить чистый текст: обрезанный HTML может
    # оказаться битым и Telegram отклонит сообщение.
    plain_answer = html.unescape(re.sub(r"<[^>]+>", "", answer)).strip() or "Пустой ответ."
    try:
        await query.answer(
            results=[
                InlineQueryResultArticle(
                    id=f"ai_{int(time.time() * 1000)}",
                    title=f"Мох: {prompt[:55]}",
                    description=plain_answer[:120].replace("\n", " "),
                    input_message_content=InputTextMessageContent(
                        message_text=truncate_utf16(plain_answer, 4000)
                    ),
                )
            ],
            cache_time=0,
            is_personal=True,
        )
        logging.info(f"[INLINE] {query.from_user.id if query.from_user else '?'} | {prompt[:80]}")
    except Exception:
        logging.exception("Inline answer error")


# ============================================================
# ФОТО
# ============================================================

DEEPSEEK_PHOTO_DESCRIBE_PROMPT = (
    "Ты — внимательный ассистент. На фото — учебное задание или несколько заданий. "
    "Сначала полностью перепиши ВЕСЬ видимый текст с каждого фото, включая условия, "
    "варианты ответов, подписи, формулы и числа. Формулы запиши понятным текстом. "
    "Опиши схемы, таблицы, графики и рисунки. Обязательно отметь, что видно плохо, "
    "размыто, обрезано или не помещается в кадр. Не решай задания на этом шаге. "
    "Сделай разбор по каждому фото, затем короткий общий итог."
)

DEEPSEEK_PHOTO_SOLVE_PROMPT = (
    "Ниже полное описание всех фото и вопрос пользователя. Реши задания подробно "
    "и по порядку. Если заданий несколько — реши каждое отдельно, не пропускай. "
    "Если в описании есть сомнительные или плохо видимые места — скажи об этом. "
    "Не пересказывай всё описание заново, сразу переходи к решению."
)


async def photo_session_worker(
    key: tuple[int, int], generation: int, delay: float = PHOTO_SESSION_DELAY
) -> None:
    try:
        await asyncio.sleep(delay)
        if _photo_generation.get(key) != generation:
            return
        messages = _photo_sessions.pop(key, [])
    finally:
        if _photo_generation.get(key) == generation:
            _photo_tasks.pop(key, None)
    if messages:
        try:
            await process_photos(messages)
        except Exception:
            logging.exception("Photo session processing error")


def enqueue_photo(message: Message, delay: float = PHOTO_SESSION_DELAY) -> None:
    if not message.from_user:
        return
    key = (message.chat.id, message.from_user.id)
    _photo_sessions.setdefault(key, []).append(message)
    _photo_generation[key] = _photo_generation.get(key, 0) + 1
    generation = _photo_generation[key]
    old_task = _photo_tasks.pop(key, None)
    if old_task and not old_task.done():
        old_task.cancel()
    if len(_photo_sessions[key]) >= DEEPSEEK_PHOTO_LIMIT:
        delay = 0.15
    _photo_tasks[key] = asyncio.create_task(photo_session_worker(key, generation, delay))


@dp.message(F.photo)
async def on_photo(message: Message):
    user = message.from_user
    if not user or user.is_bot:
        return
    if await maybe_admin_input(message):
        return

    if not is_private(message) and not is_whitelisted(message.chat.id):
        if is_addressed(message, message.caption) and can_send_fallback(message.chat.id):
            await message.reply(FALLBACK_TEXT)
        else:
            await log_group_event(message, message.caption or "")
        return

    enqueue_photo(message)


async def process_photos(messages: list[Message]) -> None:
    messages = messages[:DEEPSEEK_PHOTO_LIMIT]
    first = messages[0]
    chat_id = first.chat.id
    private = is_private(first)
    user = first.from_user

    if not private and not is_whitelisted(chat_id):
        if any(is_addressed(m, m.caption) for m in messages) and can_send_fallback(chat_id):
            await first.reply(FALLBACK_TEXT)
        else:
            for m in messages:
                await log_group_event(m, m.caption or "")
        return

    addressed = private or any(is_addressed(m, m.caption) for m in messages)
    if not addressed:
        for m in messages:
            await log_group_event(m, m.caption or "")
        return

    if not private:
        for m in messages:
            await log_group_event(m, m.caption or "")

    if is_on_cooldown(first):
        return

    model = await resolve_model(user.id if user else None, private)
    provider = provider_of(model)

    caption = next((m.caption.strip() for m in messages if m.caption), "")
    question = strip_triggers(caption) if caption else ""
    if not question:
        question = "реши подробно" if provider in ("deepseek", "qwen") else "Что на фото?"

    track = await begin_request(
        first,
        kind="photo",
        prompt=question,
        model=model,
    )
    started = time.perf_counter()

    if provider in ("deepseek", "qwen") and private:
        async with chat_lock(chat_id):
            await process_premium_photos(messages, question, track, started, model)
        return

    await process_groq_photos(messages, question, track, started)


async def process_groq_photos(messages: list[Message], question: str, track: Track, started: float) -> None:
    first = messages[0]
    chat_id = first.chat.id
    private = is_private(first)
    turn = Turn(message=first, chat_id=chat_id, is_private=private, model="groq", track=track)
    await turn.status.show("👁 Смотрю фото…")

    descriptions: list[str] = []
    failed: list[int] = []
    for index, m in enumerate(messages, 1):
        try:
            data = await download_image_bytes(m)
            desc = await describe_image(data, track=track)
            if desc:
                descriptions.append(desc)
                _media_cache[(chat_id, m.message_id)] = desc
            else:
                failed.append(index)
        except Exception as e:
            failed.append(index)
            logging.warning(f"Photo describe failed: {e}")

    if not descriptions:
        await turn.status.clear()
        await send_text(
            turn,
            "⚠️ Не удалось посмотреть фото. Попробуй отправить ещё раз.",
        )
        await finish_track(
            track,
            status="error",
            error="all photos failed",
            latency_ms=int((time.perf_counter() - started) * 1000),
            model="groq",
        )
        return

    if len(descriptions) == 1:
        photo_block = descriptions[0]
    else:
        photo_block = "\n".join(f"{i}) {d}" for i, d in enumerate(descriptions, 1))
    if failed:
        photo_block += "\n\nНе удалось прочитать фото: " + ", ".join(str(i) for i in failed)

    quote = ""
    for m in messages:
        if m.reply_to_message:
            quote, _ = await build_reply_block(m, need_media=True, track=track)
            break

    if private:
        entry = (
            f"{quote}Пользователь прислал фото:\n{photo_block}\n\n"
            f"Вопрос/подпись: {question}"
        )
    else:
        entry = (
            f"{display_user(first.from_user)}: {quote}прислал фото:\n{photo_block}\n\n"
            f"Вопрос/подпись: {question}"
        )
    await add_to_history(chat_id, "user", entry)

    turn.user_text = question
    draw = detect_explicit_draw(question)
    if draw and re.search(r"(такое же|похожее|это же|это)", draw, flags=re.IGNORECASE):
        draw = f"{draw}. Референс: {'; '.join(descriptions)}"
    turn.explicit_draw = draw

    if failed:
        await send_text(
            turn,
            "⚠️ Не удалось прочитать фото: <b>"
            + ", ".join(str(i) for i in failed)
            + "</b>. Остальные обработал.",
        )

    logging.info(f"[{chat_id}] ФОТО Groq ({len(descriptions)}) | {question[:80]}")
    async with chat_lock(chat_id):
        await run_ai(turn)


def _photo_content_messages(images: list[tuple[str, bytes]], text: str) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": text}]
    for mime, data in images:
        b64 = base64.b64encode(data).decode()
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
        )
    return [{"role": "user", "content": content}]


async def process_premium_photos(
    messages: list[Message],
    question: str,
    track: Track,
    started: float,
    model_key: str,
) -> None:
    first = messages[0]
    chat_id = first.chat.id
    private = is_private(first)
    info = MODEL_REGISTRY.get(model_key) or MODEL_REGISTRY["deepseek"]
    provider = info["provider"]
    provider_name = PROVIDER_NAMES.get(provider, provider)
    # Фото всегда обрабатывает vision-модель провайдера:
    # для DeepSeek — deepseek-v4.1-flash (в т.ч. при выбранной pro), для Qwen — qwen4-max.
    vision_model = PROVIDER_VISION_MODEL.get(provider)
    solve_model = vision_model or info["model"]
    user = first.from_user
    user_id = user.id if user else None
    rich = bool(private and user_rich(user_id))
    turn = Turn(
        message=first,
        chat_id=chat_id,
        is_private=private,
        user_text=question,
        model=model_key,
        track=track,
    )
    streamer = RichStreamer(turn) if rich else None
    status = "ok"
    error_text = ""
    try:
        await turn.status.show(f"👁 {provider_name} читает фото…")
        images: list[tuple[str, bytes]] = []
        failed: list[int] = []
        for index, m in enumerate(messages, 1):
            try:
                data = await download_image_bytes(m)
                mime = image_mime(
                    data,
                    (m.document.mime_type if m.document else None),
                )
                images.append((mime, data))
            except Exception as e:
                failed.append(index)
                logging.warning(f"Premium photo download failed: {e}")

        if not images:
            await send_text(turn, "⚠️ Не удалось скачать ни одного фото.")
            status = "error"
            error_text = "all photos failed"
            return

        failed_note = ""
        if failed:
            failed_note = (
                "\n\nВнимание: фото " + ", ".join(str(i) for i in failed)
                + " не удалось прочитать. Реши то, что видно на остальных."
            )

        describe_text = DEEPSEEK_PHOTO_DESCRIBE_PROMPT + failed_note
        describe_result = await call_model(
            provider,
            _photo_content_messages(images, describe_text),
            track=track,
            effort="off",
            max_tokens=4000,
            model=vision_model,
        )
        description = (describe_result.content or "").strip()
        if not description:
            raise RuntimeError(f"{provider_name} не вернул описание фото")

        await turn.status.show("🧠 Решаю задание…")
        user_question = question or "реши подробно"
        solve_text = (
            DEEPSEEK_PHOTO_SOLVE_PROMPT
            + "\n\nОПИСАНИЕ ФОТО:\n"
            + description
            + "\n\nВОПРОС ПОЛЬЗОВАТЕЛЯ: "
            + user_question
            + failed_note
        )
        search_mode = user_search_mode(user_id) if private else "auto"
        tools = TOOLS
        if search_mode == "off":
            tools = [
                t for t in TOOLS
                if (t.get("function") or {}).get("name") != "web_search"
            ]
        solve_messages = [
            {"role": "system", "content": system_prompt(rich)},
            {"role": "user", "content": solve_text},
        ]

        turn.explicit_draw = draw_with_reference(detect_explicit_draw(question), description)
        if turn.explicit_draw:
            forced = {"type": "function", "function": {"name": "generate_image"}}
        elif search_mode == "always" or has_search_verb(user_question):
            forced = {"type": "function", "function": {"name": "web_search"}}
        else:
            forced = None

        effort = user_effort(user_id) if private else "off"

        async def executor(name, args):
            return await execute_tool(turn, name, args)

        answer = await run_tools(
            solve_messages,
            tools,
            executor,
            forced,
            provider=provider,
            track=track,
            effort=effort,
            model=solve_model,
            on_delta=streamer.on_delta if streamer else None,
            reset_cb=streamer.reset if streamer else None,
        )

        photo_label = "фото" if len(images) == 1 else f"фото ({len(images)})"
        if private:
            entry = (
                f"Пользователь прислал {photo_label}:\n{description}\n\n"
                f"Вопрос/подпись: {user_question}{failed_note}"
            )
        else:
            entry = (
                f"{display_user(first.from_user)}: прислал {photo_label}:\n{description}\n\n"
                f"Вопрос/подпись: {user_question}{failed_note}"
            )
        await add_to_history(chat_id, "user", entry)

        if failed:
            await send_text(
                turn,
                "⚠️ Не удалось прочитать фото: <b>"
                + ", ".join(str(i) for i in failed)
                + "</b>. Остальные обработал.",
            )

        if turn.explicit_draw and not turn.image_sent and not turn.image_error:
            await fallback_draw(turn)

        if turn.image_sent:
            await add_to_history(
                chat_id,
                "assistant",
                f"Отправил сгенерированную картинку: {turn.image_prompt_ru}",
            )
        elif turn.image_error:
            await add_to_history(
                chat_id,
                "assistant",
                f"Не удалось сгенерировать изображение: {turn.image_error}",
            )

        if answer:
            if not (turn.image_sent and is_trivial(answer)):
                if streamer is not None:
                    await streamer.finish(answer)
                else:
                    await send_text(turn, answer)
                await add_to_history(chat_id, "assistant", answer)
        elif not turn.image_sent and not turn.image_error:
            fallback_text = "Не смог разобрать задание. Попробуй сфотографировать чётче."
            if streamer is not None:
                await streamer.finish(fallback_text)
            else:
                await send_text(turn, fallback_text)
            await add_to_history(chat_id, "assistant", fallback_text)

        _stats[chat_id] = _stats.get(chat_id, 0) + 1
        logging.info(
            f"[{chat_id}] ФОТО {provider_name} ({len(images)}) | {question[:80]}"
        )
    except asyncio.CancelledError:
        status = "cancelled"
        raise
    except Exception as e:
        status = "error"
        error_text = str(e)
        logging.exception("Premium photo error")
        await send_switch_button(turn, friendly_ai_error(e))
    finally:
        await finish_track(
            track,
            status=status,
            error=error_text,
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=model_key,
            tools=",".join(turn.tools_used),
        )
        await turn.status.clear()


# ============================================================
# ДОКУМЕНТЫ-КАРТИНКИ
# ============================================================


@dp.message(F.document)
async def on_document(message: Message):
    user = message.from_user
    if not user or user.is_bot:
        return
    if await maybe_admin_input(message):
        return
    mime = (message.document.mime_type or "") if message.document else ""
    if mime.startswith("image/"):
        if not is_private(message) and not is_whitelisted(message.chat.id):
            if is_addressed(message, message.caption) and can_send_fallback(message.chat.id):
                await message.reply(FALLBACK_TEXT)
            else:
                await log_group_event(message, message.caption or "")
            return
        enqueue_photo(message)
        return
    await log_other(message)


# ============================================================
# ГОЛОС
# ============================================================


@dp.message(F.voice | F.audio)
async def on_voice(message: Message):
    user = message.from_user
    if not user or user.is_bot:
        return
    if await maybe_admin_input(message):
        return

    chat_id = message.chat.id
    private = is_private(message)

    if not private and not is_whitelisted(chat_id):
        if is_reply_to_bot(message) and can_send_fallback(chat_id):
            await message.reply(FALLBACK_TEXT)
        return

    if private and is_on_cooldown(message):
        return

    await safe_typing(chat_id)

    model = await resolve_model(user.id, private) if private else "groq"
    track: Track | None = None
    if private:
        # Запрос создаём до расшифровки, чтобы вызов Whisper попал в лог,
        # а превью обновим распознанным текстом сразу после транскрипции.
        track = await begin_request(
            message,
            kind="voice",
            prompt="(голосовое)",
            model=model,
        )

    path = None
    whisper_started = time.perf_counter()
    try:
        path = await download_media_to_temp(message)
        transcript = await transcribe_voice(path, track=track)
    except Exception as e:
        logging.warning(f"Voice transcription failed: {e}")
        if track is not None:
            await finish_track(
                track,
                status="error",
                error=str(e),
                latency_ms=int((time.perf_counter() - whisper_started) * 1000),
                model=model,
            )
        if not private:
            await log_group_event(message, "(голосовое, не удалось расшифровать)")
        if private or is_reply_to_bot(message):
            await reply_any(message, f"⚠️ Не удалось расшифровать голосовое: <code>{esc(friendly_ai_error(e))}</code>")
        return
    finally:
        if path:
            try:
                os.unlink(path)
            except Exception:
                pass

    whisper_ms = int((time.perf_counter() - whisper_started) * 1000)

    if not transcript:
        if track is not None:
            await finish_track(
                track,
                status="error",
                error="empty transcript",
                latency_ms=whisper_ms,
                model=model,
            )
        if not private:
            await log_group_event(message, "(голосовое, не удалось распознать)")
        if private or is_reply_to_bot(message):
            await reply_any(message, "🤔 Не удалось распознать голосовое.")
        return

    addressed = private or is_reply_to_bot(message) or starts_with_trigger(transcript)
    if not addressed:
        await log_group_event(message, f"(голосовое, расшифровка): {transcript}")
        # В неадресованных сообщениях запроса нет, но Whisper уже вызван —
        # фиксируем его как отдельный вызов модели.
        await storage.add_call(
            request_id=None,
            user_id=user.id,
            chat_id=chat_id,
            provider="groq",
            model=WHISPER_MODEL,
            latency_ms=whisper_ms,
        )
        return

    cleaned = strip_triggers(transcript) if starts_with_trigger(transcript) else transcript
    if not cleaned:
        cleaned = "реши подробно" if private else "(обратился без вопроса)"
    _media_cache[(chat_id, message.message_id)] = cleaned

    if not private:
        await log_group_event(message, f"(голосовое, расшифровка): {cleaned}")

    if not private and is_on_cooldown(message):
        # Whisper уже отработал, поэтому фиксируем вызов даже без ответа.
        await storage.add_call(
            request_id=None,
            user_id=user.id,
            chat_id=chat_id,
            provider="groq",
            model=WHISPER_MODEL,
            latency_ms=whisper_ms,
        )
        return

    if not private:
        track = await begin_request(
            message,
            kind="voice",
            prompt=cleaned,
            model=model,
        )
        await storage.add_call(
            request_id=track.request_id,
            user_id=user.id,
            chat_id=chat_id,
            provider="groq",
            model=WHISPER_MODEL,
            latency_ms=whisper_ms,
        )
    else:
        await storage.update_request_preview(track.request_id, cleaned)
    quote, media_desc = await build_reply_block(
        message, need_media=True, track=track, provider=provider_of(model)
    )
    label = "" if private else f"{display_user(user)}: "
    entry = f"{label}{quote}(голосовое, расшифровка, возможны ошибки): {cleaned}"
    await add_to_history(chat_id, "user", entry)

    draw = detect_explicit_draw(cleaned)
    if draw and media_desc and re.search(r"(такое же|похожее|это же|это)", draw, flags=re.IGNORECASE):
        draw = f"{draw}. Референс: {media_desc}"
    turn = Turn(
        message=message,
        chat_id=chat_id,
        is_private=private,
        user_text=cleaned,
        explicit_draw=draw,
        model=model,
        track=track,
        rich_allowed=False,
    )

    logging.info(f"[{chat_id}] ГОЛОС | model={model} | {cleaned[:80]}")
    async with chat_lock(chat_id):
        await run_ai(turn)


# ============================================================
# ТЕКСТ
# ============================================================


def draw_with_reference(draw: str | None, media_desc: str) -> str | None:
    if draw and media_desc and re.search(r"(такое же|похожее|это же|это)", draw, flags=re.IGNORECASE):
        return f"{draw}. Референс: {media_desc}"
    return draw


async def handle_private_text(message: Message) -> None:
    if is_on_cooldown(message):
        return

    text = (message.text or "").strip()
    if not text:
        return

    chat_id = message.chat.id
    user = message.from_user
    model = await resolve_model(user.id if user else None, True)
    track = await begin_request(message, kind="text", prompt=text, model=model)
    quote, media_desc = await build_reply_block(
        message, need_media=True, track=track, provider=provider_of(model)
    )
    await add_to_history(chat_id, "user", f"{quote}{text}")

    turn = Turn(
        message=message,
        chat_id=chat_id,
        is_private=True,
        user_text=text,
        explicit_draw=draw_with_reference(detect_explicit_draw(text), media_desc),
        model=model,
        track=track,
    )
    logging.info(f"[{chat_id}] ЛС | model={model} | {text[:100]}")
    async with chat_lock(chat_id):
        await run_ai(turn)


async def handle_group_text(message: Message) -> None:
    chat_id = message.chat.id
    text = message.text or ""
    user = message.from_user

    if not is_whitelisted(chat_id):
        if is_addressed(message, text) and can_send_fallback(chat_id):
            await message.reply(FALLBACK_TEXT)
        return

    # В белых группах логируем каждое сообщение отдельно.
    await log_group_event(message, text)

    addressed = is_addressed(message, text)

    if not addressed:
        quote, _ = await build_reply_block(message, need_media=False)
        await add_to_history(chat_id, "user", f"{display_user(user)}: {quote}{text}")
        return

    if is_on_cooldown(message):
        return

    quote, media_desc = await build_reply_block(message, need_media=True)
    cleaned = strip_triggers(text)
    entry_text = cleaned or "(обратился без вопроса)"
    if not cleaned:
        cleaned = "реши подробно"
    await add_to_history(chat_id, "user", f"{display_user(user)}: {quote}{entry_text}")

    track = await begin_request(
        message,
        kind="group_request",
        prompt=text,
        model="groq",
    )

    logging.info(f"{display_user(user)} | {text[:100]}")

    turn = Turn(
        message=message,
        chat_id=chat_id,
        is_private=False,
        user_text=cleaned,
        explicit_draw=draw_with_reference(detect_explicit_draw(cleaned), media_desc),
        model="groq",
        track=track,
    )
    async with chat_lock(chat_id):
        await run_ai(turn)


@dp.message(F.text)
async def on_text(message: Message):
    user = message.from_user
    if not user or user.is_bot:
        return
    if await maybe_admin_input(message):
        return
    if is_private(message):
        await handle_private_text(message)
    else:
        await handle_group_text(message)


# ============================================================
# ПРОЧИЕ СООБЩЕНИЯ
# ============================================================


async def log_other(message: Message) -> None:
    placeholder = media_placeholder(message)
    if not placeholder:
        return
    caption = f" (подпись: {message.caption[:200]})" if message.caption else ""
    if is_private(message):
        await add_to_history(message.chat.id, "user", f"{placeholder}{caption}")
    elif is_whitelisted(message.chat.id):
        await add_to_history(
            message.chat.id,
            "user",
            f"{display_user(message.from_user)}: {placeholder}{caption}",
        )
        await log_group_event(message, f"{placeholder}{caption}")


@dp.message()
async def on_other(message: Message):
    user = message.from_user
    if not user or user.is_bot:
        return
    if await maybe_admin_input(message):
        return
    if (
        not is_private(message)
        and not is_whitelisted(message.chat.id)
        and is_reply_to_bot(message)
        and can_send_fallback(message.chat.id)
    ):
        await message.reply(FALLBACK_TEXT)
        return
    await log_other(message)


# ============================================================
# ЗАПУСК
# ============================================================


GROUP_COMMANDS = [
    BotCommand(command="start", description="Приветствие"),
    BotCommand(command="help", description="Что я умею"),
    BotCommand(command="about", description="Инфо обо мне и моделях"),
    BotCommand(command="clear", description="Очистить контекст чата"),
    BotCommand(command="context", description="Сколько сообщений в памяти"),
    BotCommand(command="whoami", description="Мой ID и username"),
    BotCommand(command="stats", description="Статистика чата"),
]
PRIVATE_COMMANDS = [
    BotCommand(command="start", description="Приветствие"),
    BotCommand(command="help", description="Что я умею"),
    BotCommand(command="about", description="Инфо обо мне и моделях"),
    BotCommand(command="models", description="Модели и режимы"),
    BotCommand(command="clear", description="Очистить контекст чата"),
    BotCommand(command="context", description="Сколько сообщений в памяти"),
    BotCommand(command="whoami", description="Мой ID и username"),
    BotCommand(command="stats", description="Статистика чата"),
]
BASE_COMMANDS = GROUP_COMMANDS


async def refresh_user_commands(user_id: int) -> None:
    commands = list(PRIVATE_COMMANDS)
    if int(user_id) == OWNER_ID:
        commands.append(BotCommand(command="admin", description="Админ-панель"))
    try:
        await bot.set_my_commands(
            commands, scope=BotCommandScopeChat(chat_id=int(user_id))
        )
    except Exception as e:
        logging.info(f"set commands for {user_id} skipped: {e}")


async def set_commands() -> None:
    await bot.set_my_commands(GROUP_COMMANDS, scope=BotCommandScopeDefault())
    await bot.set_my_commands(PRIVATE_COMMANDS, scope=BotCommandScopeAllPrivateChats())
    await refresh_user_commands(OWNER_ID)


async def load_runtime_state() -> None:
    await storage.connect()
    groups = await storage.load_groups()
    _groups_cache.clear()
    _groups_cache.update(groups)
    ALLOWED_GROUPS.clear()
    ALLOWED_GROUPS.update(groups.keys())

    blacklist = await storage.load_blacklist()
    _dm_blacklist.clear()
    _dm_blacklist.update(blacklist)

    grants = await storage.load_grants()
    _grants.clear()
    _grants.update(grants)

    settings = await storage.load_user_settings()
    _user_settings.clear()
    _user_settings.update(settings)

    contexts = await storage.load_contexts()
    _history.clear()
    for chat_id, messages in contexts.items():
        if isinstance(messages, list):
            _trim_history(messages)
            _history[int(chat_id)] = messages

    logging.info(
        f"Загружено: групп={len(_groups_cache)}, ЛС в блоке={len(_dm_blacklist)}, "
        f"доступов={len(_grants)}, настроек={len(_user_settings)}, "
        f"контекстов={len(_history)}"
    )


async def health_server() -> None:
    try:
        from aiohttp import web
    except Exception as e:
        logging.warning(f"aiohttp недоступен, health server выключен: {e}")
        return

    try:
        app = web.Application()
        app.router.add_get("/", lambda request: web.Response(text="ok"))
        app.router.add_get("/health", lambda request: web.Response(text="ok"))
        runner = web.AppRunner(app)
        await runner.setup()
        port = int(os.environ.get("PORT", "8080"))
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()
        logging.info(f"Health server запущен на порту {port}")
    except Exception as e:
        logging.error(f"Health server error: {e}")


async def main() -> None:
    global BOT_ID, BOT_USERNAME

    # Health server поднимаем сразу: Render ждёт открытый порт, не дожидаясь Telegram.
    tasks = []
    if os.getenv("PORT"):
        tasks.append(asyncio.create_task(health_server()))

    me = await bot.get_me()
    BOT_ID = me.id
    BOT_USERNAME = me.username or ""
    logging.info(f"Бот запущен: @{BOT_USERNAME} (id={BOT_ID})")
    logging.info(f"Chat: {CHAT_MODEL} (Groq), fast: {FAST_MODEL}")
    logging.info(
        f"DeepSeek: {DEEPSEEK_MODEL} "
        f"({'настроен' if deepseek_client.configured else 'нет AIKEYS_API'})"
    )
    logging.info(
        f"Qwen: qwen4-max "
        f"({'настроен' if qwen_client.configured else 'нет QWEN_API'})"
    )
    logging.info(f"Vision: {VISION_MODEL} (Gemini), fallback: {GROQ_VISION_MODEL} (Groq)")
    logging.info(f"Voice: {WHISPER_MODEL} (Groq)")
    logging.info("Search: Tavily + DDG fallback")

    await load_runtime_state()
    await probe_premium_models()
    dp.message.outer_middleware(DMBlacklistMiddleware())

    logging.info(f"Allowed groups: {sorted(ALLOWED_GROUPS)}")

    try:
        await set_commands()
    except Exception:
        logging.exception("set_commands failed")
    try:
        await bot.delete_webhook(drop_pending_updates=True)
    except Exception:
        logging.exception("delete_webhook failed")

    tasks.append(asyncio.create_task(daily_summary_loop()))
    tasks.append(asyncio.create_task(balances_loop()))
    tasks.append(asyncio.create_task(dp.start_polling(bot)))
    try:
        await asyncio.gather(*tasks)
    finally:
        await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
