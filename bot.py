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
    BusinessConnection,
    CallbackQuery,
    ChatMemberUpdated,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQuery,
    InlineQueryResultArticle,
    InputChecklist,
    InputChecklistTask,
    InputRichMessage,
    InputTextMessageContent,
    LinkPreviewOptions,
    MenuButtonCommands,
    Message,
)
from openai import AsyncOpenAI

from deepseek_client import ChatResult, DeepSeekClient
from prompts import (
    AGENT_INSTRUCTIONS,
    CLASSIC_FORMAT,
    LENGTH_INSTRUCTIONS,
    RICH_FORMAT,
    SYSTEM_PROMPT_BASE,
)
from storage import DB_LIMIT_BYTES, MSK, storage
import studio

# widgets.py — новый файл. Если его забыли загрузить, бот продолжит работать,
# а виджеты просто будут отдавать текстовый фолбэк вместо падения на старте.
WIDGETS_MODULE = True
try:
    from widgets import (
        WidgetError,
        dashboard_text,
        render_checklist,
        render_dashboard,
        render_widget,
    )
    from widgets import text_version as widget_text_version
except Exception:  # pragma: no cover
    WIDGETS_MODULE = False
    logging.exception("widgets.py не загружен: виджеты отключены")

    class WidgetError(ValueError):
        pass

    def _widgets_missing(*_args, **_kwargs):
        raise WidgetError("модуль widgets.py не загружен на сервер")

    render_checklist = render_dashboard = render_widget = _widgets_missing
    dashboard_text = _widgets_missing

    def widget_text_version(_kind, _spec):
        return "Виджеты временно недоступны."

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

# ---- Универсальный шлюз ------------------------------------------------
# Ключи и клиенты студии живут в studio.py; здесь остаются алиасы для
# совместимости со старым кодом статистики и баланса.
GATEWAY_API_KEY = studio.GATEWAY_API_KEY
GATEWAY_BASE_URL = studio.GATEWAY_BASE_URL
GATEWAY_BALANCE_URL = os.environ.get(
    "GATEWAY_BALANCE_URL", "https://api.aikeysforyou.com/v1/balance"
)
DEFAULT_GATEWAY_MODEL = studio.DEFAULT_GATEWAY_MODEL

# Дневные пулы токенов: free 250к и paid 100к. Владелец — безлимит.
DAILY_TOKEN_LIMIT = studio.POOL_LIMITS[studio.FREE]
PAID_TOKEN_LIMIT = studio.POOL_LIMITS[studio.PAID]

# Каталог заполняется при первом импорте fallback-списком, затем
# refresh_catalog() подтягивает живые списки провайдеров.
MODEL_REGISTRY: dict[str, dict] = {}
MODEL_SHORT: dict[str, str] = {}
MODEL_SHORT_BY_KEY: dict[str, str] = {}
MODEL_ALIASES: dict[str, str] = {}


def rebuild_model_registry(catalog: dict[str, list[dict]] | None = None) -> None:
    """Пересобирает реестр моделей и короткие callback-коды."""
    global MODEL_REGISTRY, MODEL_SHORT, MODEL_SHORT_BY_KEY, MODEL_ALIASES
    global GATEWAY_MODELS, GROQ_MODELS, ALL_MODELS

    data = catalog if catalog is not None else studio.get_catalog()
    registry: dict[str, dict] = {}
    for family in studio.FAMILY_ORDER[studio.PAID] + studio.FAMILY_ORDER[studio.FREE]:
        for spec in data.get(family) or []:
            registry[spec["key"]] = dict(spec)

    MODEL_REGISTRY = registry
    MODEL_SHORT = {}
    MODEL_SHORT_BY_KEY = {}
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    for index, key in enumerate(registry.keys()):
        code = ""
        value = index
        while True:
            code = alphabet[value % len(alphabet)] + code
            value //= len(alphabet)
            if value <= 0:
                break
        code = f"m{code}"
        MODEL_SHORT[code] = key
        MODEL_SHORT_BY_KEY[key] = code

    aliases: dict[str, str] = {}
    for key, spec in registry.items():
        model = spec.get("model") or ""
        aliases.setdefault(key, key)
        aliases.setdefault(model, key)
        aliases.setdefault(spec.get("label") or model, key)
    aliases.setdefault("deepseek", studio.DEFAULT_PAID_MODEL)
    aliases.setdefault("groq", studio.DEFAULT_FREE_MODEL)
    aliases.setdefault("gemini", "gemini@gemini-3.5-flash-lite")
    aliases.setdefault("openai/gpt-oss-120b", "groq@openai/gpt-oss-120b")
    MODEL_ALIASES = aliases
    GATEWAY_MODELS = [
        spec["model"] for spec in registry.values() if spec["family"] == "deepseek"
    ]
    GROQ_MODELS = [
        spec["model"] for spec in registry.values() if spec["family"] == "groq"
    ]
    ALL_MODELS = [spec["model"] for spec in registry.values()]


rebuild_model_registry()

PREMIUM_MODELS = list(GATEWAY_MODELS)
PROVIDER_NAMES = {
    "gateway": "DeepSeek",
    "pollinations_free": "Pollinations Free",
    "pollinations_vip": "Pollinations VIP",
    "groq": "Groq",
    "gemini": "Gemini",
}

# Думают, но не показывают мысли текстом — честно предупреждаем.
# Среди двух рабочих моделей таких нет, механизм оставлен на будущее.
SILENT_REASONING_MODELS: set[str] = set()
# Модели, которые думают всегда: переключатель размышлений им не нужен.
# Сейчас таких в каталоге нет, но механизм оставлен на будущее.
ALWAYS_REASONING_MODELS: set[str] = set()

PREMIUM_HISTORY_CHARS_LIMIT = 60000
PREMIUM_MAX_TOKENS = 32000
PREMIUM_MAX_TOKENS_CAP = 32000
PHOTO_LIMIT = 10

LENGTH_LABELS = {
    "short": "Кратко",
    "normal": "Обычно",
    "long": "Подробно",
    "max": "Максимально",
}
LENGTH_ORDER = ["short", "normal", "long", "max"]
LENGTH_TOKENS = {"short": 600, "normal": 2000, "long": 6000, "max": 32000}
EFFORT_LABELS = {"off": "Не думать", "medium": "Думать", "high": "Думать долго"}
EFFORT_ORDER = ["off", "medium", "high"]


def plural_ru(count: int, one: str, few: str, many: str) -> str:
    """Русское склонение по числу: 1 модель, 2 модели, 5 моделей."""
    count = abs(int(count))
    if count % 10 == 1 and count % 100 != 11:
        return one
    if 2 <= count % 10 <= 4 and count % 100 not in (12, 13, 14):
        return few
    return many

OWNER_ID = 6101597477
OWNER_USERNAME = "Diotrex"
ALLOWED_GROUPS: set[int] = {-1002495961136}
# Белый список ЛС. Пустое множество = только владелец.
ALLOWED_USERS: set[int] = set()

HISTORY_LIMIT = 50
HISTORY_STORE_CHARS_LIMIT = 60000
HISTORY_CHARS_LIMIT = 14000
MAX_REPLY_LENGTH = 4000
# Telegram сворачивает сообщения длиннее ~8000 символов под «Показать ещё».
# Держим куски заметно ниже этого порога, чтобы ответ читался прямо в чате.
MAX_RICH_REPLY_LENGTH = 7000
# Сколько символов размышлений показываем в раскрытом блоке. Мысли идут в
# одном сообщении с ответом, поэтому их надо держать заметно короче лимита.
REASONING_PREVIEW_CHARS = 1200
CHAT_MAX_TOKENS = 4000
COOLDOWN_SECONDS = 2
AI_TIMEOUT = 180.0
# Для DeepSeek/Qwen: быстро падаем, если хост недоступен, и не держим
# пользователя в тишине минутами при обрыве соединения.
PREMIUM_TIMEOUT = httpx.Timeout(90.0, connect=12.0, write=30.0, pool=12.0)
TYPING_INTERVAL = 4.0
FALLBACK_COOLDOWN = 600
PHOTO_SESSION_DELAY = 2.0
MAX_ALBUM_PHOTOS = PHOTO_LIMIT
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

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(
        parse_mode="HTML",
        # Ссылки не должны разворачиваться в предпросмотр нигде.
        link_preview=LinkPreviewOptions(is_disabled=True),
    ),
)
dp = Dispatcher()

# Видно в логах Render: по этой строке понятно, что задеплоена свежая версия.
APP_BUILD = "2026-11-27 studio-free-paid-models"

# Rich Message API появился в свежих aiogram. Если на сервере старая версия,
# честно отключаем «статьи» вместо тихих ошибок на каждом ответе.
RICH_SUPPORTED = hasattr(bot, "send_rich_message") and hasattr(
    bot, "send_rich_message_draft"
)
WIDGET_SUPPORTED = WIDGETS_MODULE and hasattr(bot, "send_widget")
CHECKLIST_SUPPORTED = hasattr(bot, "send_checklist")
if not RICH_SUPPORTED:
    logging.warning(
        "Rich messages недоступны: нужен aiogram>=3.33. "
        "Обновите зависимости и перезапустите сервис."
    )
if not WIDGET_SUPPORTED:
    logging.warning(
        "HTML-виджеты недоступны: нужен aiogram>=3.33 и файл widgets.py."
    )

http_client = httpx.AsyncClient(timeout=AI_TIMEOUT, proxy=PROXY_URL or None)

ai_client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL,
    http_client=http_client,
    max_retries=0,
    timeout=AI_TIMEOUT,
)

studio.configure(
    http_client,
    ai_timeout=AI_TIMEOUT,
    premium_timeout=PREMIUM_TIMEOUT,
)
gateway_client = studio.gateway_client

PROVIDER_CLIENTS: dict[str, DeepSeekClient] = {
    "gateway": gateway_client,
    "pollinations_free": studio.pollinations_free_client,
    "pollinations_vip": studio.pollinations_vip_client,
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

# Фото-сессии: все подряд идущие фото одного пользователя склеиваются в одну пачку.
_photo_sessions: dict[tuple[int, int], list[Message]] = {}
_photo_tasks: dict[tuple[int, int], asyncio.Task] = {}
_photo_generation: dict[tuple[int, int], int] = {}

# Настройки пользователей и доступы живут в Postgres, здесь — кэш.
_user_settings: dict[int, dict] = {}
_grants: dict[tuple[int, str], dict] = {}
_dm_blacklist: dict[int, dict] = {}
_allowed_users_cache: dict[int, dict] = {}
_groups_cache: dict[int, dict] = {}
_limit_notified: set[tuple[int, str, str]] = set()
_block_reply_last: dict[int, float] = {}
_balance: dict[str, dict] = {}
_balance_alerted: set[str] = set()
_model_blocked: dict[str, str] = {}
# Чек-листы Telegram можно слать только через business-подключение.
_business_connection_id: str | None = None

# Состояние админ-панели
_admin_input: dict[int, dict] = {}
_admin_filter: dict[int, dict] = {}
_db_size_cache = 0
_today_usage_cache: dict[tuple[int, str], int] = {}

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
            if attempt + 1 >= retries:
                break
            wait = base_wait * (attempt + 1)
            logging.warning(f"Timeout, retry {attempt + 1}/{retries} in {wait}s")
            await asyncio.sleep(wait)
        except Exception as e:
            last_err = e
            if is_retryable_error(e):
                if attempt + 1 >= retries:
                    raise
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
# ИМЕНА ПОЛЬЗОВАТЕЛЕЙ
# ============================================================

# В статистике мало одного ID: владельцу нужно видеть, кто это.
# Кэш заполняется из входящих сообщений, а для старых записей — через get_chat.
_user_names: dict[int, dict] = {}


def remember_user(user) -> None:
    """Запоминает имя и ник из входящего сообщения."""
    if user is None:
        return
    uid = getattr(user, "id", None)
    if not uid:
        return
    full_name = " ".join(
        part
        for part in (
            getattr(user, "first_name", "") or "",
            getattr(user, "last_name", "") or "",
        )
        if part
    ).strip()
    _user_names[int(uid)] = {
        "username": getattr(user, "username", "") or "",
        "full_name": full_name or getattr(user, "full_name", "") or "",
    }


def cached_user_name(user_id) -> tuple[str, str]:
    item = _user_names.get(int(user_id)) or {}
    return item.get("username") or "", item.get("full_name") or ""


async def resolve_user_name(user_id) -> tuple[str, str]:
    """Достаёт ник и имя через Telegram, если их ещё нет в кэше."""
    uid = int(user_id)
    if uid in _user_names:
        item = _user_names[uid]
        return item.get("username") or "", item.get("full_name") or ""
    username = full_name = ""
    try:
        chat = await bot.get_chat(uid)
        username = getattr(chat, "username", "") or ""
        full_name = " ".join(
            part
            for part in (
                getattr(chat, "first_name", "") or "",
                getattr(chat, "last_name", "") or "",
            )
            if part
        ).strip()
        if not full_name:
            full_name = getattr(chat, "title", "") or ""
    except Exception as e:
        logging.info(f"get_chat({uid}) skipped: {e}")
    _user_names[uid] = {"username": username, "full_name": full_name}
    return username, full_name


async def resolve_user_names(user_ids) -> None:
    """Подтягивает имена для списка ID, не больше 30 за раз."""
    missing = [
        int(uid)
        for uid in dict.fromkeys(user_ids)
        if uid and int(uid) not in _user_names
    ]
    if not missing:
        return
    await asyncio.gather(
        *(resolve_user_name(uid) for uid in missing[:30]),
        return_exceptions=True,
    )


def user_tag(user_id, username: str = "", full_name: str = "") -> str:
    """HTML-подпись «Имя @ник · id» — чтобы в статистике было понятно, кто это."""
    cached_username, cached_full_name = cached_user_name(user_id)
    username = (username or cached_username or "").strip()
    full_name = (full_name or cached_full_name or "").strip()
    parts: list[str] = []
    if full_name:
        parts.append(f"<b>{esc(full_name)}</b>")
    if username:
        parts.append(f"@{esc(username)}")
    label = " ".join(parts)
    if label:
        return f"{label} · <code>{user_id}</code>"
    return f"<code>{user_id}</code>"


def user_plain(user_id, username: str = "", full_name: str = "") -> str:
    """Текстовая подпись для виджетов и списков."""
    cached_username, cached_full_name = cached_user_name(user_id)
    username = (username or cached_username or "").strip()
    full_name = (full_name or cached_full_name or "").strip()
    parts: list[str] = []
    if full_name:
        parts.append(full_name)
    if username:
        parts.append(f"@{username}")
    label = " ".join(parts)
    return f"{label} · {user_id}" if label else str(user_id)


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


DEFAULT_SETTINGS = {
    "model": "groq",
    "rich": True,
    "deepthink": "off",
    "search": "on",
    "length": "normal",
    "agent": False,
    "model_free": "",
    "model_paid": "",
    "model_section": studio.FREE,
}


def normalize_model_key(key: str | None) -> str:
    """Приводит старые ключи (deepseek, groq, qwen) к ключам каталога."""
    raw = str(key or "").strip()
    return MODEL_ALIASES.get(raw, raw)


def is_owner(user_id: int | None) -> bool:
    return user_id is not None and int(user_id) == OWNER_ID


def default_model_for_section(section: str) -> str:
    """Рабочая модель по умолчанию для раздела free/paid."""
    preferred = (
        studio.DEFAULT_PAID_MODEL if section == studio.PAID else studio.DEFAULT_FREE_MODEL
    )
    if preferred and preferred in MODEL_REGISTRY:
        return preferred
    for spec in studio.all_specs(section):
        if spec.get("key") in MODEL_REGISTRY:
            return spec["key"]
    for key, info in MODEL_REGISTRY.items():
        if info.get("section") == section:
            return key
    return next(iter(MODEL_REGISTRY), preferred)


def user_settings(user_id: int | None) -> dict:
    if user_id is None:
        return dict(DEFAULT_SETTINGS)
    settings = _user_settings.setdefault(int(user_id), dict(DEFAULT_SETTINGS))
    # Мягко переносим настройки из старых версий.
    if settings.get("deepthink") == "low":
        settings["deepthink"] = "medium"
    if settings.get("search") in ("auto", "always"):
        settings["search"] = "on"
    if settings.get("length") not in LENGTH_ORDER:
        settings["length"] = "normal"
    if settings.get("model_section") not in (studio.FREE, studio.PAID):
        settings["model_section"] = studio.FREE
    # Проверяем сохранённые ключи моделей по живому каталогу.
    for section, field_name in ((studio.FREE, "model_free"), (studio.PAID, "model_paid")):
        key = normalize_model_key(settings.get(field_name) or "")
        if key not in MODEL_REGISTRY:
            key = default_model_for_section(section)
        settings[field_name] = key
    return settings


def provider_of(model_key: str) -> str:
    info = MODEL_REGISTRY.get(normalize_model_key(model_key)) or {}
    return info.get("provider", "groq")


def provider_eligible(user_id: int | None, provider: str) -> bool:
    # Все разделы открыты белому списку; в группах доступен только free.
    return True


def model_available(user_id: int | None, model_key: str) -> bool:
    """Доступна ли модель (без учёта дневного пула токенов)."""
    key = normalize_model_key(model_key)
    if key not in MODEL_REGISTRY:
        return False
    if key in _model_blocked:
        return False
    return True


def user_section(user_id: int | None) -> str:
    if user_id is None:
        return studio.FREE
    section = user_settings(user_id).get("model_section", studio.FREE)
    return section if section in (studio.FREE, studio.PAID) else studio.FREE


def section_model(user_id: int | None, section: str) -> str:
    if user_id is None:
        return default_model_for_section(section)
    settings = user_settings(user_id)
    field = "model_paid" if section == studio.PAID else "model_free"
    key = normalize_model_key(settings.get(field) or "")
    if key not in MODEL_REGISTRY:
        key = default_model_for_section(section)
    return key


def selected_model(user_id: int | None) -> str:
    """Выбранная модель текущего раздела пользователя."""
    return section_model(user_id, user_section(user_id))


def model_label(model_key: str) -> str:
    info = MODEL_REGISTRY.get(normalize_model_key(model_key))
    return info["label"] if info else str(model_key)


# ------------------------------------------------------------------
# Дневные пулы токенов: free 250к, paid 100к. Владелец — безлимит.
# ------------------------------------------------------------------


async def pool_used(user_id: int | None, section: str) -> int:
    if user_id is None:
        return 0
    return await storage.get_usage(int(user_id), f"pool_{section}", msk_today())


async def pool_remaining(user_id: int | None, section: str) -> int:
    limit = studio.pool_limit(section)
    if is_owner(user_id):
        return limit
    used = await pool_used(user_id, section)
    return max(0, limit - used)


async def pool_add(user_id: int | None, section: str, tokens: int) -> None:
    if user_id is None or tokens is None or int(tokens) <= 0:
        return
    if is_owner(user_id):
        return
    await storage.add_tokens(int(user_id), section, msk_today(), int(tokens))


# Раньше один апстрим приходил под разными id и статистика их склеивала.
# Теперь id в каталоге различаются, поэтому склейка пустая: в статистике
# видно ровно ту модель, что выбрана. Префикс «provider/» по-прежнему
# отрезается в canonical_model_name().
MODEL_DISPLAY_ALIASES: dict[str, str] = {}


def canonical_model_name(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        return ""
    if raw in MODEL_DISPLAY_ALIASES:
        return MODEL_DISPLAY_ALIASES[raw]
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    return MODEL_DISPLAY_ALIASES.get(raw, raw)


def merge_stats_models(rows, limit: int = 8) -> list[dict]:
    """Склеивает строки статистики по каноническому имени модели."""
    merged: dict[str, dict] = {}
    for row in rows or []:
        name = canonical_model_name(row.get("model")) or "?"
        item = merged.setdefault(
            name,
            {
                "model": name,
                "provider": row.get("provider") or "",
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "cached_tokens": 0,
                "total_tokens": 0,
                "credit": 0.0,
            },
        )
        provider = row.get("provider") or ""
        if provider and provider not in item["provider"].split("/"):
            item["provider"] = (item["provider"] + "/" + provider).strip("/")
        item["calls"] += int(row.get("calls") or 0)
        for token_field in (
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "cached_tokens",
            "total_tokens",
        ):
            item[token_field] += int(row.get(token_field) or 0)
        item["credit"] += float(row.get("credit") or 0)
    return sorted(
        merged.values(),
        key=lambda r: (r["total_tokens"], r["calls"]),
        reverse=True,
    )[:limit]


def is_blacklisted(user_id: int | None) -> bool:
    return user_id is not None and int(user_id) in _dm_blacklist


def is_allowed_user(user_id: int | None) -> bool:
    """Белый список ЛС: владелец и добавленные пользователи."""
    if user_id is None:
        return False
    if int(user_id) == OWNER_ID:
        return True
    return int(user_id) in ALLOWED_USERS


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


async def notify_pool_exhausted(message: Message, section: str | None = None) -> None:
    """Дневной пул токенов исчерпан: показываем ошибку и кнопку другого раздела."""
    section = section or studio.FREE
    other = studio.PAID if section == studio.FREE else studio.FREE
    limit = studio.pool_limit(section)
    text = (
        f"⚠️ Дневной пул {studio.SECTION_LABELS.get(section, section)} исчерпан "
        f"({fmt_int(limit)} токенов). Сброс в 00:00 МСК.\n\n"
        f"Можно перейти в {studio.SECTION_LABELS.get(other, other)}."
    )
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"Перейти в {studio.SECTION_LABELS.get(other, other)}",
                    callback_data=f"models:sec:{other}",
                )
            ]
        ]
    )
    try:
        await message.reply(text, reply_markup=kb)
    except Exception:
        try:
            await message.answer(text, reply_markup=kb)
        except Exception:
            pass


async def resolve_model(user_id: int | None, private: bool) -> str:
    """Возвращает модель для запроса.

    В группах — только бесплатный раздел. Пустая строка означает, что
    дневной пул раздела исчерпан: вызывающий код показывает ошибку.
    """
    if not private:
        key = section_model(user_id, studio.FREE)
        if key not in MODEL_REGISTRY:
            key = default_model_for_section(studio.FREE)
        if not is_owner(user_id) and await pool_remaining(user_id, studio.FREE) <= 0:
            return ""
        return key
    if user_id is None:
        return default_model_for_section(studio.FREE)
    section = user_section(user_id)
    key = section_model(user_id, section)
    if not is_owner(user_id) and await pool_remaining(user_id, section) <= 0:
        return ""
    return key


async def set_model_for_user(user_id: int, model_key: str) -> tuple[bool, str]:
    """Переключает модель. Контекст и кэш фото сохраняются."""
    model_key = normalize_model_key(model_key)
    if model_key not in MODEL_REGISTRY:
        return False, "Неизвестная модель."
    if model_key in _model_blocked:
        return False, f"Модель {model_label(model_key)} сейчас недоступна."
    info = MODEL_REGISTRY[model_key]
    section = info.get("section", studio.FREE)
    field = "model_paid" if section == studio.PAID else "model_free"
    settings = user_settings(user_id)
    if settings.get(field) == model_key and settings.get("model_section") == section:
        return True, f"Уже выбрана модель {model_label(model_key)}."
    settings[field] = model_key
    settings["model_section"] = section
    settings["model"] = model_key
    await storage.save_user_settings(
        int(user_id),
        **{field: model_key, "model_section": section, "model": model_key},
    )
    note = ""
    if not is_owner(user_id):
        left = await pool_remaining(user_id, section)
        note = (
            f"\nОсталось сегодня: {fmt_int(left)} ток."
            if left > 0
            else "\nПул этого раздела на сегодня исчерпан."
        )
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
    if level == "low":
        level = "medium"
    return level if level in EFFORT_ORDER else "off"


def user_search_on(user_id: int | None) -> bool:
    return user_settings(user_id).get("search", "on") != "off"


def user_length(user_id: int | None) -> str:
    length = user_settings(user_id).get("length", "normal")
    return length if length in LENGTH_ORDER else "normal"


def user_agent(user_id: int | None) -> bool:
    return bool(user_settings(user_id).get("agent", False))


async def safe_typing(chat_id: int) -> None:
    now = time.time()
    if now - _typing_last.get(chat_id, 0) < TYPING_INTERVAL:
        return
    _typing_last[chat_id] = now
    try:
        await bot.send_chat_action(chat_id, ChatAction.TYPING)
    except Exception:
        pass


async def keep_typing(chat_id: int) -> None:
    """Держит «печатает…», пока модель отвечает, чтобы бот не выглядел зависшим."""
    try:
        while True:
            await safe_typing(chat_id)
            await asyncio.sleep(TYPING_INTERVAL)
    except asyncio.CancelledError:
        raise


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
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


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
# РАЗНОЕ
# ============================================================

SEARCH_VERBS = (
    "поищи", "поищи-ка", "найди", "загугли", "погугли", "search",
    "проверь в интернете", "посмотри в интернете", "глянь в интернете",
)


def has_search_verb(text: str) -> bool:
    lowered = (text or "").lower()
    return any(v in lowered for v in SEARCH_VERBS)


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

# Промпты вынесены в prompts.py — их можно редактировать отдельно.


def system_prompt(
    rich: bool = False, length: str = "normal", agent: bool = False
) -> str:
    fmt = RICH_FORMAT if rich else CLASSIC_FORMAT
    length_block = LENGTH_INSTRUCTIONS.get(length) or LENGTH_INSTRUCTIONS["normal"]
    parts = [SYSTEM_PROMPT_BASE]
    if agent:
        parts.append(AGENT_INSTRUCTIONS)
    parts.append(fmt)
    parts.append(length_block)
    parts.append(f"Сегодня: {date.today().isoformat()}.")
    return "\n\n".join(parts)


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
            "name": "render_widget",
            "description": (
                "Показать интерактивную карточку-виджет в личном чате. Используй, когда "
                "просят график, диаграмму, сравнительную таблицу, тест/квиз или трекер "
                "прогресса. Не используй для обычного текста и простых ответов."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["chart", "table", "quiz", "progress"],
                        "description": "Тип виджета",
                    },
                    "title": {"type": "string", "description": "Заголовок карточки"},
                    "spec": {
                        "type": "object",
                        "description": (
                            "chart: {labels:[...], series:[{name, values:[...]}], "
                            "chart_type:'line'|'bar', unit}. "
                            "table: {columns:[...], rows:[[...]]}. "
                            "quiz: {question, options:[...], correct: индекс с нуля, "
                            "explanation}. progress: {items:[{label, value, max}]}"
                        ),
                    },
                },
                "required": ["kind", "spec"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_checklist",
            "description": (
                "Отправить настоящий чек-лист Telegram с галочками. Используй для планов, "
                "списков покупок, подготовки к экзамену, чек-листа деплоя и любых задач "
                "из 3+ пунктов, которые пользователь будет отмечать."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Название чек-листа"},
                    "tasks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Пункты чек-листа, максимум 30",
                    },
                },
                "required": ["title", "tasks"],
            },
        },
    },
]

SEARCH_TOOLS = [TOOLS[0]]


# ============================================================
# GROQ
# ============================================================


async def groq_chat(
    messages,
    tools=None,
    tool_choice="auto",
    max_tokens: int | None = None,
    model: str | None = None,
):
    async def _do():
        kwargs = {
            "model": model or CHAT_MODEL,
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
    model: str | None = None,
) -> ChatResult:
    """Стриминг Groq с накоплением текста, мыслей и tool-call'ов."""
    base = {
        "model": model or CHAT_MODEL,
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
        model=model or CHAT_MODEL,
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
    model_key: str | None = None,
    on_delta=None,
    reset_cb=None,
) -> ChatResult:
    started = time.perf_counter()
    status = "ok"
    error_text = ""
    result: ChatResult | None = None
    spec = MODEL_REGISTRY.get(model_key) if model_key else None
    if spec is None and model:
        for candidate in MODEL_REGISTRY.values():
            if candidate.get("model") == model and candidate.get("provider") == provider:
                spec = candidate
                break
    client = PROVIDER_CLIENTS.get(provider)
    model_name = model or (spec.get("model") if spec else None) or (
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
                    model=model_name,
                )
            else:
                response = await groq_chat(
                    messages,
                    tools,
                    tool_choice or "auto",
                    max_tokens or default_max,
                    model=model_name,
                )
                result = openai_to_chat_result(response, provider, model_name)
        elif provider in ("pollinations_free", "pollinations_vip"):
            if spec is None:
                raise RuntimeError("Модель не найдена в каталоге студии.")
            result = await studio.pollinations_chat(
                spec,
                messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens or default_max,
                on_delta=on_delta,
            )
        elif provider == "gemini":
            if spec is None:
                raise RuntimeError("Модель не найдена в каталоге студии.")
            result = await studio.gemini_chat(
                spec,
                messages,
                max_tokens=max_tokens or default_max,
                effort=effort,
            )
        else:
            if client is None or not client.configured:
                raise RuntimeError(f"{PROVIDER_NAMES.get(provider, provider)} не настроен")

            def ask(stream: bool):
                return client.chat(
                    messages,
                    model=model_name,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=max_tokens or default_max,
                    effort=effort,
                    on_delta=on_delta if stream else None,
                )

            try:
                result = await with_retry(
                    lambda: ask(True),
                    retries=1 if on_delta is not None else 2,
                )
            except Exception as e:
                # Стрим мог оборваться на полуслове. Повторяем без стрима,
                # предварительно очистив недописанный черновик.
                if on_delta is None:
                    raise
                logging.warning(
                    f"{PROVIDER_NAMES.get(provider, provider)}: стрим упал ({e}), "
                    "повторяю без стриминга"
                )
                if reset_cb is not None:
                    reset_cb()
                result = await with_retry(lambda: ask(False), retries=1)
        return result
    except Exception as e:
        status = "error"
        error_text = str(e)
        raise
    finally:
        latency = int((time.perf_counter() - started) * 1000)
        if track is not None and result is not None:
            track.prompt_tokens += int(result.prompt_tokens or 0)
            track.completion_tokens += int(result.completion_tokens or 0)
            track.reasoning_tokens += int(result.reasoning_tokens or 0)
            track.total_tokens += int(
                result.total_tokens
                or (result.prompt_tokens + result.completion_tokens + result.reasoning_tokens)
            )
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
# РАСПОЗНАВАНИЕ ФОТО — Groq new3
# ============================================================


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
    section: str = studio.FREE,
) -> str:
    """Распознавание фото только через Groq new3 (бесплатный ключ)."""
    mime = image_mime(image_bytes, mime)
    started = time.perf_counter()
    try:
        text, response = await studio.describe_image(image_bytes, mime, section=section)
    except Exception as e:
        if track is not None:
            await storage.add_call(
                request_id=track.request_id,
                user_id=track.user_id,
                chat_id=track.chat_id,
                provider="groq",
                model=studio.GROQ_VISION_MODEL,
                latency_ms=int((time.perf_counter() - started) * 1000),
                status="error",
                error=str(e),
            )
        raise
    if track is not None:
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total = prompt_tokens + completion_tokens
        track.total_tokens += total
        await storage.add_call(
            request_id=track.request_id,
            user_id=track.user_id,
            chat_id=track.chat_id,
            provider="groq",
            model=studio.GROQ_VISION_MODEL,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    return text


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


async def transcribe_voice(
    file_path: str,
    *,
    track: Track | None = None,
    section: str = studio.FREE,
    user_id: int | None = None,
) -> str:
    started = time.perf_counter()
    try:
        text, estimate = await studio.transcribe_voice(file_path, section=section)
    except Exception as e:
        if track is not None:
            await storage.add_call(
                request_id=track.request_id,
                user_id=track.user_id,
                chat_id=track.chat_id,
                provider="groq",
                model=studio.GROQ_FALLBACK_VOICE_MODEL,
                latency_ms=int((time.perf_counter() - started) * 1000),
                status="error",
                error=str(e),
            )
        raise
    estimate = int(estimate or 0)
    if track is not None:
        track.total_tokens += estimate
        await storage.add_call(
            request_id=track.request_id,
            user_id=track.user_id,
            chat_id=track.chat_id,
            provider="groq",
            model=studio.GROQ_FALLBACK_VOICE_MODEL,
            total_tokens=estimate,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    elif user_id is not None:
        await pool_add(user_id, section, estimate)
    return text


# ============================================================
# ПОИСК — TAVILY + запасной DuckDuckGo
# ============================================================


async def tavily_search(
    query: str, max_results: int = 5, *, section: str = studio.FREE
) -> list[dict]:
    api_key = studio.tavily_key(section)
    if not api_key:
        raise RuntimeError("Tavily: ключ не задан")

    async def _do():
        response = await http_client.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "api_key": api_key,
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


async def search_web(
    query: str, max_results: int = 5, *, section: str = studio.FREE
) -> list[dict]:
    query = (query or "").strip()
    if not query:
        return []

    key = f"{section}:{query.lower()}"
    now = time.time()
    cached = _search_cache.get(key)
    if cached and now - cached[0] < 600:
        return cached[1]

    results: list[dict] = []
    try:
        results = normalize_results(
            await tavily_search(query, max_results, section=section)
        )
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
    model: str = "groq"
    section: str = studio.FREE
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
    "details|summary|table|thead|tbody|tr|td|th|tg-spoiler|footer|"
    "tg-math-block|tg-math"
)
RICH_DROP_TAGS = ("script", "style", "iframe", "svg", "object", "embed", "form", "video", "audio")

# Участки, внутри которых нельзя трогать LaTeX.
_MATH_PROTECTED_RE = re.compile(
    r"<tg-math-block\b[^>]*>.*?</tg-math-block>"
    r"|<tg-math\b[^>]*>.*?</tg-math>"
    r"|<pre\b[^>]*>.*?</pre>"
    r"|<code\b[^>]*>.*?</code>"
    r"|\$\$.*?\$\$"
    r"|\$[^$\n]{1,200}\$"
    r"|\\\(.*?\\\)"
    r"|\\\[.*?\\\]",
    re.IGNORECASE | re.DOTALL,
)
_FRAC_RE = re.compile(r"\\(?:d)?frac\s*")


def _balanced_group(text: str, start: int) -> int:
    """Возвращает индекс сразу после парной закрывающей скобки или -1."""
    if start >= len(text) or text[start] != "{":
        return -1
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def _wrap_fracs(segment: str) -> str:
    if "frac" not in segment:
        return segment
    out: list[str] = []
    i = 0
    while True:
        match = _FRAC_RE.search(segment, i)
        if not match:
            out.append(segment[i:])
            break
        out.append(segment[i : match.start()])
        pos = match.end()
        while pos < len(segment) and segment[pos].isspace():
            pos += 1
        first_end = _balanced_group(segment, pos)
        if first_end == -1:
            out.append(segment[match.start(): match.end()])
            i = match.end()
            continue
        pos = first_end
        while pos < len(segment) and segment[pos].isspace():
            pos += 1
        second_end = _balanced_group(segment, pos)
        if second_end == -1:
            out.append(segment[match.start(): first_end])
            i = first_end
            continue
        out.append(f"<tg-math>{segment[match.start():second_end]}</tg-math>")
        i = second_end
    return "".join(out)


def autowrap_frac(text: str) -> str:
    """Оборачивает голый \\frac{}{} в <tg-math>, чтобы дробь рисовалась чертой."""
    if not text or "frac" not in text:
        return text
    parts: list[str] = []
    last = 0
    for match in _MATH_PROTECTED_RE.finditer(text):
        parts.append(_wrap_fracs(text[last : match.start()]))
        parts.append(match.group(0))
        last = match.end()
    parts.append(_wrap_fracs(text[last:]))
    return "".join(parts)


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
    text = autowrap_frac(text)

    # LaTeX внутри <tg-math> трогать нельзя: экранирование сломает формулу.
    math_elements: dict[str, str] = {}

    def _hide_math(match):
        key = f"\x00MATH{len(math_elements)}\x00"
        math_elements[key] = match.group(0)
        return key

    text = re.sub(
        r"<tg-math-block\b[^>]*>.*?</tg-math-block>|<tg-math\b[^>]*>.*?</tg-math>",
        _hide_math,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

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
    for key, tag in math_elements.items():
        text = text.replace(key, tag)
    return text


TEX_SIMPLE = (
    (r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)"),
    (r"\\dfrac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)"),
    (r"\\sqrt\s*\{([^{}]*)\}", r"√(\1)"),
    (r"\\cdot", "·"),
    (r"\\times", "×"),
    (r"\\div", "÷"),
    (r"\\pm", "±"),
    (r"\\le", "≤"),
    (r"\\ge", "≥"),
    (r"\\neq", "≠"),
    (r"\\approx", "≈"),
    (r"\\infty", "∞"),
    (r"\\alpha", "α"),
    (r"\\beta", "β"),
    (r"\\gamma", "γ"),
    (r"\\pi", "π"),
)


def tex_to_plain(text: str) -> str:
    """Упрощает LaTeX для режимов без поддержки формул."""
    if not text:
        return text
    for pattern, repl in TEX_SIMPLE:
        text = re.sub(pattern, repl, text)
    text = text.replace("{", "").replace("}", "")
    return text


RICH_BLOCK_RE = re.compile(
    r"(<details\b[^>]*>.*?</details>|<h[1-6]\b[^>]*>.*?</h[1-6]>|"
    r"<p\b[^>]*>.*?</p>|<ul\b[^>]*>.*?</ul>|<ol\b[^>]*>.*?</ol>|"
    r"<blockquote\b[^>]*>.*?</blockquote>|<pre\b[^>]*>.*?</pre>|"
    r"<footer\b[^>]*>.*?</footer>|"
    r"<table\b[^>]*>.*?</table>|<tg-math-block\b[^>]*>.*?</tg-math-block>)",
    re.IGNORECASE | re.DOTALL,
)
RICH_TAG_RE = re.compile(
    rf"<(/?)({RICH_ALLOWED_TAGS})(?:\s[^>]*)?>", re.IGNORECASE
)


def _open_rich_tags(text: str) -> list[tuple[str, str]]:
    stack: list[tuple[str, str]] = []
    for match in RICH_TAG_RE.finditer(text):
        closing, tag = match.group(1), match.group(2).lower()
        if closing:
            for index in range(len(stack) - 1, -1, -1):
                if stack[index][0] == tag:
                    del stack[index:]
                    break
        else:
            stack.append((tag, match.group(0)))
    return stack


def _hard_split_rich(text: str, limit: int) -> list[str]:
    """Режет слишком длинный блок, закрывая и заново открывая теги."""
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = max(
            remaining.rfind("\n\n", 0, limit),
            remaining.rfind("\n", 0, limit),
            remaining.rfind(". ", 0, limit),
        )
        if cut < limit * 0.4:
            cut = limit
        # Не разрезаем HTML-тег посередине.
        if remaining.rfind("<", 0, cut) > remaining.rfind(">", 0, cut):
            cut = remaining.rfind("<", 0, cut)
            if cut <= 0:
                cut = remaining.find(">", limit) + 1 or limit
        chunk = remaining[:cut].rstrip()
        opened = _open_rich_tags(chunk)
        if opened:
            chunk += "".join(f"</{tag}>" for tag, _ in reversed(opened))
        chunks.append(chunk)
        reopen = "".join(raw for _, raw in opened)
        remaining = (reopen + remaining[cut:]).strip()
    if remaining.strip():
        chunks.append(remaining.strip())
    return chunks


def split_rich(text: str, limit: int = MAX_RICH_REPLY_LENGTH) -> list[str]:
    if len(text) <= limit:
        return [text] if text.strip() else []
    # Сначала режем по целым блокам (<p>, <h1>, <ul>, <details>...),
    # чтобы не разрывать теги. Внутри слишком длинного блока — по предложениям.
    blocks: list[str] = []
    for part in RICH_BLOCK_RE.split(text):
        if not part or not part.strip():
            continue
        stripped = part.strip()
        if RICH_BLOCK_RE.fullmatch(stripped):
            blocks.append(stripped)
        else:
            blocks.extend(p.strip() for p in re.split(r"\n{2,}", part) if p.strip())
    chunks: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split_rich(block, limit))
            continue
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) > limit:
            if current:
                chunks.append(current)
            current = block
        else:
            current = candidate
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]


def plain_from_rich(text: str) -> str:
    text = re.sub(
        r"<details\b[^>]*>.*?</details>", "", text, flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(
        r"<footer\b[^>]*>(.*?)</footer>",
        lambda m: "\n\n<i>" + m.group(1) + "</i>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"</t[dh]>\s*", " · ", text, flags=re.IGNORECASE)
    text = re.sub(r"</tr>\s*", "\n", text, flags=re.IGNORECASE)
    text = re.sub(
        r"</(?:p|div|h[1-6]|li|blockquote)>", "\n", text, flags=re.IGNORECASE
    )
    text = re.sub(r"<[^>]+>", "", text)
    return tex_to_plain(html.unescape(text).strip())


def classic_from_rich(text: str) -> str:
    """Конвертирует Rich HTML в обычный Telegram HTML: заголовки → жирный."""
    text = re.sub(
        r"<details\b[^>]*>.*?</details>", "", text, flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(
        r"<footer\b[^>]*>(.*?)</footer>",
        lambda m: "\n\n<i>" + m.group(1) + "</i>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"<h([1-6])\b[^>]*>(.*?)</h\1>",
        r"\n<b>\2</b>\n",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"<p\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<li\b[^>]*>", "• ", text, flags=re.IGNORECASE)
    text = re.sub(r"</li>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</t[dh]>\s*", " · ", text, flags=re.IGNORECASE)
    text = re.sub(r"</tr>\s*", "\n", text, flags=re.IGNORECASE)
    text = re.sub(
        r"</?(?:ul|ol|div|span|table|thead|tbody|tr|td|th|mark|tg-math|tg-math-block)\b[^>]*>",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return tex_to_plain(text).strip()


def usage_footer(turn: Turn, rich: bool) -> str:
    """Мелкая подпись в конце ответа: модель и потраченные токены."""
    track = turn.track
    if track is None or track.total_tokens <= 0:
        return ""
    label = esc(model_label(turn.model))
    tokens = fmt_int(track.total_tokens)
    if rich:
        return f"<footer>{label} · {tokens} ток.</footer>"
    return f"\n\n<i>{label} · {tokens} ток.</i>"


def silent_reasoning_note(streamer: "RichStreamer | None", effort: str) -> str:
    """Честная пометка: модель думала, но текст мыслей не прислала."""
    if streamer is None or effort == "off":
        return ""
    if streamer._thoughts_html():
        return ""
    return (
        "<p><i>🧠 Модель оставила рассуждения внутренними — "
        "текст мыслей она не прислала.</i></p>"
    )


def reasoning_to_html(raw: str) -> str:
    """Размышления модели → аккуратные абзацы Rich Message.

    Модели шлют мысли либо обычным текстом, либо слипшимися жирными
    тезисами вида `**Сравниваю таблицы****Проверяю исключения**`.
    Приводим это к <p> и <b>, чтобы не показывать сырой Markdown.
    """
    text = (raw or "").strip()
    if not text:
        return ""
    # Слипшиеся тезисы: **A****B** → **A** **B** (пары разбираются по очереди).
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = text.replace("**", "")
    text = text.replace("</b><b>", "</b> <b>")
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-•]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    body = sanitize_rich(text)
    paragraphs = [chunk.strip() for chunk in re.split(r"\n{2,}", body) if chunk.strip()]
    split: list[str] = []
    for chunk in paragraphs:
        lines = [line.strip() for line in chunk.split("\n") if line.strip()]
        if len(lines) > 1 and all(len(line) <= 300 for line in lines):
            split.extend(lines)
        else:
            split.append(chunk)
    return "\n".join(f"<p>{chunk}</p>" for chunk in split[:40])


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
        # Между раундами инструментов сбрасываем только текст ответа:
        # размышления уже показаны пользователю и не должны исчезать.
        self.content.clear()

    def restart(self) -> None:
        # Повтор после сбоя модели: мысли неудачной попытки тоже убираем.
        self.content.clear()
        self.reasoning.clear()

    def _thoughts_html(self) -> str:
        raw = strip_text_tool_calls("".join(self.reasoning)).strip()
        if not raw:
            return ""
        # Мысли идут в том же сообщении, что и начало ответа. Если модель
        # вывалила простыню, показываем только начало: иначе <details>
        # вытеснит сам ответ в следующее сообщение.
        cut = raw[:REASONING_PREVIEW_CHARS]
        if len(raw) > REASONING_PREVIEW_CHARS:
            cut = cut.rstrip()
            stop = max(
                cut.rfind(". "), cut.rfind("! "), cut.rfind("? "), cut.rfind("\n")
            )
            if stop > 300:
                cut = cut[: stop + 1]
            else:
                cut = cut[: REASONING_PREVIEW_CHARS - 1].rstrip() + "…"
        body = reasoning_to_html(cut)
        if not body:
            return ""
        return (
            "<details open><summary>🧠 Размышления</summary>\n"
            f"{body}\n</details>"
        )

    def build_html(self) -> str:
        # Сначала размышления, потом ответ — одним rich-сообщением.
        parts: list[str] = []
        thoughts = self._thoughts_html()
        if thoughts:
            parts.append(thoughts)
        body = sanitize_rich(strip_text_tool_calls("".join(self.content)))
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

    async def _deliver(self, html_text: str) -> None:
        if not html_text.strip():
            return
        try:
            await bot.send_rich_message(
                chat_id=self.turn.chat_id,
                rich_message=InputRichMessage(html=html_text),
            )
        except Exception as e:
            logging.warning(f"send_rich_message failed: {e}; fallback to text")
            classic = classic_from_rich(html_text)
            if classic:
                await send_text(self.turn, classic)
                return
            plain = plain_from_rich(html_text)
            if plain:
                await send_text(self.turn, plain)

    async def finish(self, answer: str = "", footer: str = "") -> None:
        if self.finished:
            return
        self.finished = True
        raw = strip_text_tool_calls("".join(self.content)).strip()
        if not raw:
            raw = strip_text_tool_calls(answer or "").strip()
        body = sanitize_rich(raw)
        parts: list[str] = []
        thoughts = self._thoughts_html()
        if thoughts:
            parts.append(thoughts)
        if body:
            parts.append(body)
        full = "\n\n".join(parts)
        if footer:
            full = (full + "\n\n" + footer) if full else footer
        if not full.strip():
            return
        # Мысли и ответ — в одном rich-сообщении; длинный текст режем по блокам,
        # чтобы ни один кусок не превышал лимит Telegram.
        for chunk in split_rich(full):
            await self._deliver(chunk)


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


async def send_widget_card(turn: Turn, html_text: str, caption: str = "") -> bool:
    """Отправляет HTML-виджет (только ЛС: ограничение Bot API)."""
    if not WIDGET_SUPPORTED or not turn.is_private:
        return False
    kwargs = {"chat_id": turn.chat_id, "html": html_text}
    if caption:
        kwargs["caption"] = caption
    try:
        await bot.send_widget(**kwargs)
        return True
    except Exception as e:
        text = str(e)
        if "WIDGET_NOT_ALLOWED" in text:
            # Telegram ограничил sendWidget обычным ботам (с 13.11.2026).
            # Больше не тратим запросы: работаем текстовыми версиями.
            globals()["WIDGET_SUPPORTED"] = False
            logging.warning(
                "sendWidget запрещён Telegram для этого бота "
                "(INPUT_MEDIA_WIDGET_NOT_ALLOWED). Виджеты отключены до "
                "перезапуска, включён текстовый фолбэк."
            )
        else:
            logging.warning(f"send_widget failed: {e}")
        return False


async def enrich_dashboard_users(data: dict) -> dict:
    """Добавляет к топ-пользователям читаемую подпись вместо одного ID."""
    users = data.get("top_users") or []
    if not users:
        return data
    await resolve_user_names([u.get("user_id") for u in users])
    enriched = []
    for user in users:
        item = dict(user)
        item["label"] = user_plain(
            user.get("user_id"), user.get("username", ""), user.get("full_name", "")
        )
        enriched.append(item)
    data = dict(data)
    data["top_users"] = enriched
    return data


async def send_dashboard_message(
    chat_id: int, data: dict, period_label: str = "24 часа"
) -> str:
    """Дашборд виджетом, а если Telegram запрещает виджеты — обычным текстом."""
    data = await enrich_dashboard_users(data)
    if WIDGET_SUPPORTED:
        try:
            html_text, caption = render_dashboard(data, period_label)
            await bot.send_widget(chat_id=chat_id, html=html_text, caption=caption)
            return "widget"
        except Exception as e:
            if "WIDGET_NOT_ALLOWED" in str(e):
                globals()["WIDGET_SUPPORTED"] = False
                logging.warning(
                    "sendWidget запрещён Telegram; дашборд отправляю текстом."
                )
            else:
                logging.warning(f"dashboard widget failed: {e}")
    await bot.send_message(chat_id, dashboard_text(data, period_label))
    return "text"


async def send_checklist_card(turn: Turn, title: str, tasks: list[str]) -> bool:
    """Чек-лист: нативный (через business), иначе интерактивный виджет, иначе текст.

    Bot API разрешает sendChecklist только для подключённого business-аккаунта,
    поэтому в обычной личке рисуем интерактивный HTML-чек-лист: галочки
    переключаются на клиенте.
    """
    title = title[:120]
    clean = [str(task).strip()[:100] for task in tasks if str(task or "").strip()][:30]
    if not clean:
        return False

    if CHECKLIST_SUPPORTED and _business_connection_id:
        try:
            checklist = InputChecklist(
                title=title,
                tasks=[
                    InputChecklistTask(id=index + 1, text=text)
                    for index, text in enumerate(clean)
                ],
                others_can_add_tasks=False,
                others_can_mark_tasks_as_done=True,
            )
            await bot.send_checklist(
                business_connection_id=_business_connection_id,
                chat_id=turn.chat_id,
                checklist=checklist,
            )
            return True
        except Exception as e:
            logging.warning(f"send_checklist failed: {e}; using widget")

    if WIDGET_SUPPORTED and turn.is_private:
        try:
            html_text, caption = render_checklist(title, clean)
            if await send_widget_card(turn, html_text, caption):
                return True
        except Exception as e:
            logging.warning(f"checklist widget failed: {e}")

    lines = [f"<b>{esc(title)}</b>"] + [f"☐ {esc(text)}" for text in clean]
    await send_text(turn, "\n".join(lines))
    return False


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
            results = await search_web(query, section=turn.section)
            return render_results(results)

        if name == "render_widget":
            kind = (args.get("kind") or "").strip().lower()
            spec = args.get("spec") if isinstance(args.get("spec"), dict) else {}
            title = (args.get("title") or "").strip()
            if title and "title" not in spec:
                spec = dict(spec, title=title)
            try:
                html_text, caption = render_widget(kind, spec)
            except WidgetError as e:
                return f"Виджет не отправлен: {e}"
            except Exception as e:
                logging.exception("Widget render error")
                return f"Ошибка виджета: {e}"

            await turn.status.show("🧩 Собираю виджет…")
            logging.info(f"[{turn.chat_id}] TOOL render_widget: {kind}")
            if not turn.is_private:
                await send_text(turn, widget_text_version(kind, spec))
                return "В группах виджеты недоступны. Отправил текстовую версию."
            if await send_widget_card(turn, html_text, caption):
                return "Виджет отправлен пользователю."
            await send_text(turn, widget_text_version(kind, spec))
            return "Клиент не поддержал виджет. Отправил текстовую версию."

        if name == "make_checklist":
            title = (args.get("title") or "Чек-лист").strip()[:120]
            raw_tasks = args.get("tasks") or []
            tasks = [str(task).strip() for task in raw_tasks if str(task).strip()][:30]
            if not tasks:
                return "Ошибка: пустой список пунктов."
            await turn.status.show("📋 Собираю чек-лист…")
            logging.info(f"[{turn.chat_id}] TOOL make_checklist: {len(tasks)} пунктов")
            ok = await send_checklist_card(turn, title, tasks)
            if ok:
                return "Чек-лист отправлен пользователю."
            return "Клиент не поддержал чек-лист. Отправил текстом."

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


TEXT_TOOL_NAMES = ("make_checklist", "render_widget", "web_search")
_TEXT_TOOL_BLOCK_RE = re.compile(
    r"<(make_checklist|render_widget|web_search)\b([^>]*?)"
    r"(?:/>|>(.*?)</\1\s*>)",
    re.IGNORECASE | re.DOTALL,
)
_TEXT_TOOL_START_RE = re.compile(
    rf"<({'|'.join(TEXT_TOOL_NAMES)})\b[^>]*>", re.IGNORECASE
)
_TEXT_TOOL_CLOSE_RE = re.compile(
    rf"</({'|'.join(TEXT_TOOL_NAMES)})\s*>", re.IGNORECASE
)
_TEXT_TOOL_TAIL_RE = re.compile(
    rf"\s*<({'|'.join(TEXT_TOOL_NAMES)})\b[^>]*$",
    re.IGNORECASE | re.DOTALL,
)
_ATTR_RE = re.compile(
    r"([\w-]+)\s*=\s*(\"([^\"]*)\"|'([^']*)'|([^\s\"'>]+))",
    re.DOTALL,
)


def _autoclose_text_tools(text: str) -> str:
    """Достраивает закрывающие теги текстовых вызовов.

    Модели часто закрывают вложенный тег вместо внешнего (`<tasks>` вместо
    `<make_checklist>`): вызов без `</make_checklist>` протекал бы в ответ
    как сырой текст. Собираем структуру по порядку открытия и закрываем сами.
    """
    out: list[str] = []
    stack: list[str] = []
    pos = 0
    while True:
        start = _TEXT_TOOL_START_RE.search(text, pos)
        close = _TEXT_TOOL_CLOSE_RE.search(text, pos)
        if start and (close is None or start.start() < close.start()):
            name = start.group(1).lower()
            out.append(text[pos:start.start()])
            out.append(start.group(0))
            pos = start.end()
            if start.group(0).endswith("/>"):
                continue
            stack.append(name)
            continue
        if close:
            name = close.group(1).lower()
            out.append(text[pos:close.start()])
            pos = close.end()
            if name in stack:
                while stack:
                    top = stack.pop()
                    out.append(f"</{top}>")
                    if top == name:
                        break
            continue
        break
    out.append(text[pos:])
    if stack:
        out.append("".join(f"</{name}>" for name in reversed(stack)))
    return "".join(out)


def _parse_text_tool_attrs(raw: str) -> dict:
    attrs: dict[str, str] = {}
    for match in _ATTR_RE.finditer(raw or ""):
        key = match.group(1).lower()
        value = match.group(3) or match.group(4) or match.group(5) or ""
        attrs[key] = html.unescape(value).strip()
    return attrs


def _json_or_none(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def _split_tasks(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    parsed = _json_or_none(text)
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    parts = re.split(r"\s*(?:\||;|\n|•|·)\s*", text)
    return [part.strip(" -—\t") for part in parts if part.strip(" -—\t")]


def _text_tool_args(name: str, attrs: dict, body: str) -> dict:
    body = (body or "").strip()
    # Модели часто оборачивают данные в свои теги: <tasks>[...]</tasks>.
    wrapped = re.match(
        r"^<([a-z_]+)\b[^>]*>(.*)</\1\s*>$", body, re.IGNORECASE | re.DOTALL
    )
    if wrapped:
        body = wrapped.group(2).strip()
    else:
        # Незакрытая обёртка: `<tasks>[...]` без `</tasks>`.
        opening = re.match(r"^<([a-z_]+)\b[^>]*>", body, re.IGNORECASE)
        if opening and not opening.group(0).endswith("/>"):
            body = body[opening.end():].strip()
    args: dict = dict(attrs)
    data = _json_or_none(body) if body else None

    if name == "make_checklist":
        if isinstance(data, dict):
            args.update(data)
        elif isinstance(data, list):
            args["tasks"] = data
        tasks = args.get("tasks")
        if isinstance(tasks, str):
            args["tasks"] = _split_tasks(tasks)
        elif isinstance(tasks, list):
            args["tasks"] = _split_tasks(tasks)
        elif body and not isinstance(data, (dict, list)):
            args["tasks"] = _split_tasks(body)
        args.setdefault("title", "Чек-лист")
        return args

    if name == "render_widget":
        if isinstance(data, dict):
            args.update(data)
        elif body:
            args["spec"] = body
        kind = str(args.get("kind") or args.get("type") or "").strip().lower()
        if kind:
            args["kind"] = kind
        spec = args.get("spec")
        if isinstance(spec, str):
            parsed_spec = _json_or_none(spec)
            if parsed_spec is not None:
                args["spec"] = parsed_spec
        return args

    # web_search
    if body and not args.get("query"):
        args["query"] = body
    return args


def extract_text_tool_calls(text: str) -> tuple[str, list[tuple[str, dict]]]:
    """Находит текстовые вызовы вида <make_checklist ...> и вырезает их из ответа."""
    raw = _autoclose_text_tools(text or "")
    calls: list[tuple[str, dict]] = []

    def _take(match: re.Match) -> str:
        name = match.group(1).lower()
        attrs = _parse_text_tool_attrs(match.group(2) or "")
        args = _text_tool_args(name, attrs, match.group(3) or "")
        calls.append((name, args))
        return ""

    cleaned = _TEXT_TOOL_BLOCK_RE.sub(_take, raw)
    cleaned = _TEXT_TOOL_TAIL_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, calls


def strip_text_tool_calls(text: str) -> str:
    """Только чистит текст, ничего не выполняя (для стрима и финального ответа)."""
    return extract_text_tool_calls(text)[0]


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
    model_key: str | None = None,
    on_delta=None,
    reset_cb=None,
    max_tokens: int | None = None,
    max_steps: int = 4,
) -> str:
    choice = forced or "auto"
    limit = max_tokens or (
        CHAT_MAX_TOKENS if provider == "groq" else PREMIUM_MAX_TOKENS
    )
    for _ in range(max(1, max_steps)):
        try:
            result = await call_model(
                provider,
                messages,
                tools=tools,
                tool_choice=choice,
                track=track,
                effort=effort,
                max_tokens=limit,
                model=model,
                model_key=model_key,
                on_delta=on_delta,
                reset_cb=reset_cb,
            )
        except Exception:
            if choice != "auto":
                choice = "auto"
                continue
            raise

        if not result.tool_calls:
            content = (result.content or "").strip()
            clean, text_calls = extract_text_tool_calls(content)
            if text_calls:
                if reset_cb is not None:
                    reset_cb()
                for name, args in text_calls:
                    try:
                        await executor(name, args)
                    except Exception as e:
                        logging.warning(f"text tool {name} failed: {e}")
                return clean
            return content

        if reset_cb is not None:
            reset_cb()
        messages.append(result.assistant_message(include_reasoning=provider != "groq"))
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

    # Шаги кончились: просим итог без новых вызовов инструментов.
    if reset_cb is not None:
        reset_cb()
    try:
        final = await call_model(
            provider,
            messages,
            tools=tools,
            tool_choice="auto" if provider == "groq" else "none",
            track=track,
            effort=effort,
            max_tokens=limit,
            model=model,
            model_key=model_key,
            on_delta=on_delta,
            reset_cb=reset_cb,
        )
    except Exception:
        return ""
    content = (final.content or "").strip()
    clean, _text_calls = extract_text_tool_calls(content)
    return clean or content


async def send_switch_button(turn: Turn, error_text: str) -> None:
    current = normalize_model_key(turn.model)
    info = MODEL_REGISTRY.get(current) or {}
    section = info.get("section", studio.FREE)
    rows: list[list[InlineKeyboardButton]] = []
    if turn.is_private:
        # Сначала альтернативы из этого же раздела, затем модель из соседнего.
        candidates: list[str] = []
        for key, spec in MODEL_REGISTRY.items():
            if key == current or spec.get("section") != section:
                continue
            candidates.append(key)
        other = studio.PAID if section == studio.FREE else studio.FREE
        cross: list[str] = []
        for key, spec in MODEL_REGISTRY.items():
            if key == current or spec.get("section") != other:
                continue
            if spec.get("provider") == "gateway":
                client = PROVIDER_CLIENTS.get("gateway")
                if client is None or not client.configured:
                    continue
            cross.append(key)
        defaults = [
            default_model_for_section(section),
            default_model_for_section(other),
        ]
        for bucket in (candidates, cross):
            for default in defaults:
                if default in bucket:
                    bucket.remove(default)
                    bucket.insert(0, default)
        chosen = (candidates[:2] + cross[:1])[:3]
        for key in chosen:
            code = MODEL_SHORT_BY_KEY.get(key)
            if not code:
                continue
            rows.append(
                [
                    InlineKeyboardButton(
                        text=f"Переключиться на {model_label(key)}",
                        callback_data=f"m:s:{code}",
                    )
                ]
            )
    keyboard = InlineKeyboardMarkup(inline_keyboard=rows) if rows else None
    text = (
        f"⚠️ Модель <b>{esc(model_label(current))}</b> сейчас недоступна.\n\n"
        f"<code>{esc(error_text[:300])}</code>"
    )
    if rows:
        text += "\n\nМожно переключиться кнопкой ниже."
    try:
        if turn.is_private:
            await turn.message.answer(text, reply_markup=keyboard)
        else:
            await turn.message.reply(text)
    except Exception:
        await send_text(turn, text)


def _is_transient(error: Exception) -> bool:
    """Временный сбой шлюза (сеть/перегрузка), а не отказ модели."""
    status = getattr(error, "status_code", None)
    if status is not None:
        try:
            code = int(status)
        except (TypeError, ValueError):
            code = 0
        if code in (408, 425, 429, 500, 502, 503, 504):
            return True
        if 400 <= code < 500:
            return False
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "сетевая ошибка",
            "connection",
            "timeout",
            "timed out",
            "temporarily",
            "unavailable",
            "try again",
        )
    )


def _is_model_not_found(error: Exception) -> bool:
    text = str(error).lower()
    return (
        "model_not_found" in text
        or "model not found" in text
        or "unknown model" in text
        or "model_not_allowed" in text
        or "not enabled for the api key" in text
        or getattr(error, "status_code", None) == 404
    )


def _is_image_rejected(error: Exception) -> bool:
    """Шлюз отверг именно вложение-картинку, а не модель и не запрос."""
    text = (str(error) + " " + str(getattr(error, "body", "") or "")).lower()
    if "model_not_allowed" in text or "unknown model" in text:
        return False
    if "image" in text or "photo" in text or "vision" in text:
        return True
    status = getattr(error, "status_code", None)
    try:
        code = int(status) if status is not None else 0
    except (TypeError, ValueError):
        code = 0
    return code in (400, 413, 415, 422) and "invalid_request" in text


async def run_ai(turn: Turn) -> None:
    started = time.perf_counter()
    model_key = normalize_model_key(turn.model)
    if model_key not in MODEL_REGISTRY:
        model_key = default_model_for_section(turn.section)
    turn.model = model_key
    info = MODEL_REGISTRY[model_key]
    provider = info["provider"]
    model_id = info["model"]
    section = info.get("section", turn.section)
    turn.section = section
    user = turn.message.from_user
    user_id = user.id if user else None
    settings_active = turn.is_private and user_id is not None
    rich = bool(
        RICH_SUPPORTED and turn.rich_allowed and settings_active and user_rich(user_id)
    )
    streamer = RichStreamer(turn) if rich else None
    status = "ok"
    error_text = ""
    typing_task: asyncio.Task | None = None
    try:
        await safe_typing(turn.chat_id)
        typing_task = asyncio.create_task(keep_typing(turn.chat_id))
        length = user_length(user_id) if settings_active else "normal"
        agent = settings_active and user_agent(user_id)
        messages = [
            {
                "role": "system",
                "content": system_prompt(rich, length, agent=agent),
            }
        ]
        messages += [dict(m) for m in history_for_model(turn.chat_id, model_key)]

        search_on = user_search_on(user_id) if settings_active else True
        tools = TOOLS
        if agent:
            # Агент сам решает, когда искать: поиск не должен быть выключен.
            search_on = True
        elif not search_on:
            tools = [
                t
                for t in TOOLS
                if (t.get("function") or {}).get("name") != "web_search"
            ]

        if agent:
            # В агентском режиме модель сама выбирает инструменты на каждом шаге.
            forced = None
        elif search_on and has_search_verb(turn.user_text):
            forced = {"type": "function", "function": {"name": "web_search"}}
        else:
            forced = None

        effort = (
            "high" if agent else (user_effort(user_id) if settings_active else "off")
        )
        max_steps = 8 if agent else 4
        max_tokens = (
            CHAT_MAX_TOKENS
            if provider == "groq"
            else LENGTH_TOKENS.get(length, LENGTH_TOKENS["normal"])
        )
        if agent:
            max_tokens = max(int(max_tokens or 0), 2000)

        async def executor(name, args):
            return await execute_tool(turn, name, args)

        attempt_args = dict(
            provider=provider,
            track=turn.track,
            effort=effort,
            model=model_id,
            model_key=model_key,
            on_delta=streamer.on_delta if streamer else None,
            reset_cb=streamer.reset if streamer else None,
            max_tokens=max_tokens,
            max_steps=max_steps,
        )

        try:
            answer = await run_tools(messages, tools, executor, forced, **attempt_args)
        except Exception as e:
            if _is_model_not_found(e):
                _model_blocked[model_key] = "модель недоступна в шлюзе"
            if streamer is not None:
                streamer.restart()
            if _is_transient(e) and not _is_model_not_found(e):
                # Сетевые сбои шлюза обычно кратковременные: два повтора с
                # паузой. Иначе пользователь зря видит «модель недоступна».
                for wait in (2.0, 5.0):
                    logging.warning(
                        f"model {model_key}: временный сбой ({e}); "
                        f"повтор через {wait:.0f}с"
                    )
                    await asyncio.sleep(wait)
                    try:
                        answer = await run_tools(
                            messages, tools, executor, forced, **attempt_args
                        )
                        break
                    except Exception as again:
                        e = again
                        if streamer is not None:
                            streamer.restart()
                else:
                    raise
            else:
                # Отказ модели: ровно один повтор, потом честный переход.
                logging.warning(f"model {model_key}: сбой ({e}); один повтор")
                answer = await run_tools(
                    messages, tools, executor, forced, **attempt_args
                )

        footer = usage_footer(turn, rich)
        if answer:
            answer = strip_text_tool_calls(answer)
        note = silent_reasoning_note(streamer, effort)
        if answer and note:
            footer = note + ("\n\n" + footer if footer else "")
        if answer:
            if streamer is not None:
                await streamer.finish(answer, footer=footer)
            else:
                await send_text(turn, answer + footer)
            await add_to_history(turn.chat_id, "assistant", answer)
        elif turn.tools_used:
            # Инструмент уже сработал — не пишем «Не понял».
            if footer and streamer is not None:
                await streamer.finish("", footer=footer)
        else:
            fallback_text = "Не понял. Повтори, пожалуйста."
            if streamer is not None:
                await streamer.finish(fallback_text, footer=footer)
            else:
                await send_text(turn, fallback_text + footer)
            await add_to_history(turn.chat_id, "assistant", fallback_text)

        _stats[turn.chat_id] = _stats.get(turn.chat_id, 0) + 1
        logging.info(
            f"[{turn.chat_id}] AI готов | model={model_key} | section={section} | "
            f"rich={rich} | поиск={search_on} | длина={length} | "
            f"agent={agent} | effort={effort} | "
            f"токенов={turn.track.total_tokens if turn.track else 0}"
        )
    except asyncio.CancelledError:
        status = "cancelled"
        raise
    except Exception as e:
        status = "error"
        error_text = str(e)
        logging.exception("AI handling error")
        if provider == "groq":
            await send_text(turn, f"⚠️ Ошибка: <code>{esc(friendly_ai_error(e))}</code>")
        else:
            await send_switch_button(turn, friendly_ai_error(e))
    finally:
        if typing_task is not None:
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass
        await finish_track(
            turn.track,
            status=status,
            error=error_text,
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=model_key,
            tools=",".join(turn.tools_used),
        )
        if turn.track is not None:
            await pool_add(user_id, section, turn.track.total_tokens)
        await turn.status.clear()

# ============================================================
# КОМАНДЫ
# ============================================================


def command_allowed(message: Message) -> bool:
    if is_private(message):
        user = message.from_user
        return user is not None and is_allowed_user(user.id)
    return is_whitelisted(message.chat.id)


async def reply_any(message: Message, text: str) -> None:
    try:
        if is_private(message):
            await message.answer(text)
        else:
            await message.reply(text)
    except Exception as e:
        logging.error(f"reply_any error: {e}")


@dp.business_connection()
async def on_business_connection(connection: BusinessConnection):
    """Запоминаем business-подключение: оно нужно для нативных чек-листов."""
    global _business_connection_id
    if connection.is_enabled:
        _business_connection_id = connection.id
        logging.info(f"Business connection enabled: {connection.id}")
    elif _business_connection_id == connection.id:
        _business_connection_id = None
        logging.info(f"Business connection disabled: {connection.id}")


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
        "• <b>Модели</b> — бесплатные и платные, в /models.\n"
        "• <b>Поиск</b> — включается сам, когда нужны свежие данные.\n"
        "• <b>Inline</b>: <code>@M0XAIBOT запрос</code>\n\n"
        "🧩 Модели и режимы: /models\n"
        "👤 Профиль и лимиты: /profile\n"
        "🧹 Сбросить контекст: /clear\n"
        "ℹ️ Обо мне: /about",
    )


@dp.message(Command("help"))
async def cmd_help(message: Message):
    # /help теперь ведёт в /about: единая справка.
    await cmd_about(message)


@dp.message(Command("about"))
async def cmd_about(message: Message):
    if not command_allowed(message):
        return
    chat_id = message.chat.id
    current = default_model_for_section(studio.FREE)
    if is_private(message) and message.from_user:
        current = selected_model(message.from_user.id)
    section = (MODEL_REGISTRY.get(normalize_model_key(current)) or {}).get(
        "section", studio.FREE
    )
    current_line = (
        f"Сейчас: <b>{esc(model_label(current))}</b> · "
        f"{studio.SECTION_LABELS.get(section, section)}"
        if is_private(message)
        else f"В группах: <b>{esc(model_label(current))}</b> (только free)"
    )
    agent_on = bool(
        is_private(message)
        and message.from_user
        and user_agent(message.from_user.id)
    )
    agent_line = (
        "\n🤖 Режим агента: <b>"
        + ("включён" if agent_on else "выключен")
        + "</b> (до 8 шагов)"
        if is_private(message)
        else ""
    )
    free_n = len(_section_keys(studio.FREE))
    paid_n = len(_section_keys(studio.PAID))
    await reply_any(
        message,
        "🌿 <b>Мудрый Мох — информация</b>\n\n"
        f"<b>🧩 Модели:</b> {free_n + paid_n} "
        f"{plural_ru(free_n + paid_n, 'модель', 'модели', 'моделей')}\n"
        f"• 🆓 Free: {free_n} — Pollinations, Groq, Gemini\n"
        f"• 💎 Paid: {paid_n} — DeepSeek\n"
        f"{current_line}{agent_line}\n\n"
        "<b>🤖 Агент:</b>\n"
        "• Включается в /models → «Режимы»\n"
        "• Сам планирует, ищет и вызывает инструменты — до 8 шагов\n"
        "• Всегда думает максимально глубоко\n"
        "• Работает только в личке\n\n"
        "<b>👁 Фото:</b>\n"
        "• 💎 Paid — DeepSeek видит фото сам\n"
        "• 🆓 Free — нативно, если у модели есть зрение, иначе Groq new3\n\n"
        "<b>🎙 Голос:</b>\n"
        "• 🆓 Free — Groq new3, 💎 Paid — Groq new2\n\n"
        "<b>🔍 Поиск:</b>\n"
        "• 🆓 Free — Tavily (старый ключ) + DuckDuckGo\n"
        "• 💎 Paid — Tavily (новый ключ) + DuckDuckGo\n"
        "• Включается в «Режимах», в агенте доступен всегда\n\n"
        "<b>🧠 Размышления и длина:</b>\n"
        "• Не думать / Думать / Думать долго\n"
        "• Кратко / Обычно / Подробно / Максимально\n\n"
        "<b>🔑 Лимиты в сутки:</b>\n"
        f"• 🆓 Free — {fmt_int(studio.POOL_LIMITS[studio.FREE])} токенов\n"
        f"• 💎 Paid — {fmt_int(studio.POOL_LIMITS[studio.PAID])} токенов\n"
        "• Раздельные счётчики, сброс в 00:00 МСК\n"
        "• В группах — по автору сообщения, только free\n"
        "• Владельцу — безлимит\n\n"
        "<b>📊 Статистика чата:</b>\n"
        f"• Время работы: <b>{format_uptime()}</b>\n"
        f"• Запросов в этом чате: <b>{_stats.get(chat_id, 0)}</b>\n"
        f"• В контексте: <b>{get_history_size(chat_id)}</b> из {HISTORY_LIMIT}\n\n"
        "<b>🛠 Команды:</b>\n"
        "/models — модели и режимы\n"
        "/profile — профиль, лимиты и статистика\n"
        "/clear — очистить контекст\n"
        "/about — эта справка",
    )


@dp.message(Command("clear"))
async def cmd_clear(message: Message):
    if not command_allowed(message):
        return
    chat_id = message.chat.id
    await clear_chat_context(chat_id)
    await reply_any(message, "🧹 Контекст этого чата очищен. Начинаем с чистого листа!")
    logging.info(f"[{chat_id}] /clear")


MODELS_PAGE = 8


def _section_keys(section: str) -> list[str]:
    return [spec["key"] for spec in studio.all_specs(section) if spec["key"] in MODEL_REGISTRY]


def _family_keys(section: str, family: str) -> list[str]:
    return [
        spec["key"]
        for spec in studio.all_specs(section)
        if spec.get("family") == family and spec["key"] in MODEL_REGISTRY
    ]


def _tab_row(active: str) -> list[InlineKeyboardButton]:
    row = []
    for tab, label in (
        ("models", "🧩 Модели"),
        ("modes", "⚙️ Режимы"),
    ):
        row.append(
            InlineKeyboardButton(
                text=("• " if tab == active else "") + label,
                callback_data=f"models:tab:{tab}",
            )
        )
    return row


def _section_row(active: str) -> list[InlineKeyboardButton]:
    row = []
    for section in (studio.FREE, studio.PAID):
        row.append(
            InlineKeyboardButton(
                text=("✅ " if section == active else "") + studio.SECTION_LABELS[section],
                callback_data=f"models:sec:{section}",
            )
        )
    return row


async def _models_sections_view(user_id: int) -> tuple[str, InlineKeyboardMarkup]:
    section = user_section(user_id)
    current = selected_model(user_id)
    lines = [
        "🧩 <b>Модели</b>",
        "",
        f"Сейчас: <b>{esc(model_label(current))}</b>",
        f"Раздел: <b>{studio.SECTION_LABELS.get(section, section)}</b>",
        "",
        "Выбери раздел:",
    ]
    for sec in (studio.FREE, studio.PAID):
        count = len(_section_keys(sec))
        used = await pool_used(user_id, sec)
        limit = studio.pool_limit(sec)
        lines.append(
            f"• {studio.SECTION_LABELS[sec]}: <b>{count}</b> "
            f"{plural_ru(count, 'модель', 'модели', 'моделей')} · "
            f"{fmt_int(used)}/{fmt_int(limit)} ток."
        )
    rows = [
        _tab_row("models"),
        _section_row(section),
        [InlineKeyboardButton(text="🔄 Обновить", callback_data="models:tab:models")],
    ]
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


async def _models_families_view(
    user_id: int, section: str
) -> tuple[str, InlineKeyboardMarkup]:
    current = section_model(user_id, section)
    section = MODEL_REGISTRY.get(current, {}).get("section", section)
    families = studio.families(section)
    lines = [
        f"🧩 <b>{esc(studio.SECTION_TITLES.get(section, section))}</b>",
        "",
        f"Сейчас: <b>{esc(model_label(current))}</b>",
        "",
        "Выбери семейство:",
    ]
    rows: list[list[InlineKeyboardButton]] = [_tab_row("models"), _section_row(section)]
    for family in families:
        keys = _family_keys(section, family)
        mark = "✅ " if current in keys else ""
        label = studio.FAMILY_LABELS.get(family, family)
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{mark}{label} · {len(keys)}",
                    callback_data=f"models:fam:{section}:{family}:0",
                )
            ]
        )
    rows.append(
        [InlineKeyboardButton(text="◀️ К разделам", callback_data="models:tab:models")]
    )
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


async def _models_list_view(
    user_id: int, section: str, family: str, page: int = 0
) -> tuple[str, InlineKeyboardMarkup]:
    keys = _family_keys(section, family)
    if not keys:
        return await _models_families_view(user_id, section)
    current = section_model(user_id, section)
    total = max(1, (len(keys) + MODELS_PAGE - 1) // MODELS_PAGE)
    page = max(0, min(int(page or 0), total - 1))
    start = page * MODELS_PAGE
    page_keys = keys[start: start + MODELS_PAGE]
    label = studio.FAMILY_LABELS.get(family, family)
    lines = [
        f"🧩 <b>{esc(label)}</b>",
        "",
        f"Раздел: {studio.SECTION_LABELS.get(section, section)}",
        f"Сейчас: <b>{esc(model_label(current))}</b>",
        "",
        "Выбери модель, а по ℹ️ — подробности:",
    ]
    rows: list[list[InlineKeyboardButton]] = [_tab_row("models")]
    for key in page_keys:
        code = MODEL_SHORT_BY_KEY.get(key)
        if not code:
            continue
        mark = "✅ " if key == current else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{mark}{MODEL_REGISTRY[key]['label']}",
                    callback_data=f"m:s:{code}",
                ),
                InlineKeyboardButton(text="ℹ️", callback_data=f"m:i:{code}"),
            ]
        )
    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(
            InlineKeyboardButton(
                text="◀️", callback_data=f"models:fam:{section}:{family}:{page - 1}"
            )
        )
    nav.append(
        InlineKeyboardButton(
            text=f"{page + 1}/{total}",
            callback_data=f"models:fam:{section}:{family}:{page}",
        )
    )
    if page + 1 < total:
        nav.append(
            InlineKeyboardButton(
                text="▶️", callback_data=f"models:fam:{section}:{family}:{page + 1}"
            )
        )
    rows.append(nav)
    rows.append(
        [
            InlineKeyboardButton(
                text="◀️ К семействам", callback_data=f"models:sec:{section}"
            ),
            InlineKeyboardButton(text="🔄 Обновить", callback_data=f"models:fam:{section}:{family}:{page}"),
        ]
    )
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


def _model_info_view(
    code: str, section: str, family: str, page: int
) -> tuple[str, InlineKeyboardMarkup]:
    key = MODEL_SHORT.get(code) or ""
    spec = studio.model_by_key(key) or MODEL_REGISTRY.get(key)
    if not spec:
        return "Модель не найдена.", InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="◀️ Назад", callback_data="models:tab:models")]]
        )
    text = studio.info_text(spec)
    rows = [
        [
            InlineKeyboardButton(
                text="✅ Выбрать модель", callback_data=f"m:s:{code}"
            )
        ],
        [
            InlineKeyboardButton(
                text="◀️ Назад",
                callback_data=f"models:fam:{section}:{family}:{page}",
            )
        ],
    ]
    return text, InlineKeyboardMarkup(inline_keyboard=rows)


def _models_modes_view(user_id: int) -> tuple[str, InlineKeyboardMarkup]:
    settings = user_settings(user_id)
    model_key = selected_model(user_id)
    rich_on = bool(settings.get("rich", True)) and RICH_SUPPORTED
    level = user_effort(user_id)
    length = user_length(user_id)
    search_on = user_search_on(user_id)
    agent_on = user_agent(user_id)

    history_len = get_history_size(int(user_id))
    lines = [
        "⚙️ <b>Режимы</b>",
        "",
        "🧩 <b>Rich-статьи</b> — оформленные ответы со стримингом: "
        "заголовки, таблицы, раскрывающиеся размышления.",
        "🧠 <b>Размышления</b> — сколько модель думает перед ответом. "
        "Мысли показываются над ответом в раскрытом блоке.",
        "📏 <b>Длина ответа</b> — от короткой выжимки до максимума.",
        "🔍 <b>Поиск</b> — разрешить модели искать в интернете.",
        "🤖 <b>Агент</b> — сам планирует, ищет и вызывает инструменты "
        "до 8 шагов. Думает долго, отвечает по факту.",
        "",
        f"Rich: <b>{'включён' if rich_on else 'выключен'}</b>",
        f"Размышления: <b>{EFFORT_LABELS.get(level, 'Не думать')}</b>",
        f"Длина: <b>{LENGTH_LABELS.get(length, 'Обычно')}</b>",
        f"Поиск: <b>{'включён' if search_on else 'выключен'}</b>",
        f"Агент: <b>{'включён' if agent_on else 'выключен'}</b>",
        f"Контекст: <b>{history_len} сообщений</b> из {HISTORY_LIMIT}",
    ]
    if not RICH_SUPPORTED:
        lines.append("\n⚠️ Rich-статьи недоступны на этой версии aiogram.")
    if model_key in ALWAYS_REASONING_MODELS:
        lines.append(
            f"\n🧠 <b>{esc(model_label(model_key))}</b> думает всегда — "
            "переключатель размышлений скрыт."
        )
    elif model_key in SILENT_REASONING_MODELS:
        lines.append(
            f"\n🧠 <b>{esc(model_label(model_key))}</b> размышляет, "
            "но не показывает мысли текстом."
        )
    if agent_on:
        lines.append(
            "\n🤖 В агентском режиме размышления всегда максимальные, "
            "а поиск доступен всегда."
        )

    rows: list[list[InlineKeyboardButton]] = [_tab_row("modes")]
    rows.append(
        [
            InlineKeyboardButton(
                text=f"{'✅' if rich_on else '⬜️'} Rich-статьи",
                callback_data="models:toggle:rich:modes",
            )
        ]
    )
    rows.append(
        [
            InlineKeyboardButton(
                text=f"{'✅' if agent_on else '⬜️'} 🤖 Режим агента",
                callback_data="models:toggle:agent:modes",
            )
        ]
    )
    if model_key not in ALWAYS_REASONING_MODELS and not agent_on:
        rows.append(
            [
                InlineKeyboardButton(
                    text=("✅ " if level == code else "") + label,
                    callback_data=f"models:effort:{code}:modes",
                )
                for code, label in (
                    ("off", "Не думать"),
                    ("medium", "Думать"),
                    ("high", "Думать долго"),
                )
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(
                text=("✅ " if length == code else "") + label,
                callback_data=f"models:length:{code}:modes",
            )
            for code, label in (
                ("short", "Кратко"),
                ("normal", "Обычно"),
            )
        ]
    )
    rows.append(
        [
            InlineKeyboardButton(
                text=("✅ " if length == code else "") + label,
                callback_data=f"models:length:{code}:modes",
            )
            for code, label in (
                ("long", "Подробно"),
                ("max", "Максимально"),
            )
        ]
    )
    if not agent_on:
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{'✅' if search_on else '⬜️'} Поиск в интернете",
                    callback_data=f"models:search:{'off' if search_on else 'on'}:modes",
                )
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(
                text="🧹 Очистить контекст", callback_data="models:clear"
            )
        ]
    )
    rows.append(
        [InlineKeyboardButton(text="🔄 Обновить", callback_data="models:tab:modes")]
    )
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=rows)


async def models_panel_view(
    user_id: int,
    tab: str = "models",
    family: str | None = None,
    section: str | None = None,
    page: int = 0,
) -> tuple[str, InlineKeyboardMarkup]:
    if tab == "modes":
        return _models_modes_view(user_id)
    if tab == "family" and family:
        return await _models_list_view(
            user_id, section or user_section(user_id), family, page
        )
    if tab == "section" and section:
        return await _models_families_view(user_id, section)
    return await _models_sections_view(user_id)


@dp.message(Command("models"))
async def cmd_models(message: Message):
    if not command_allowed(message) or not is_private(message) or not message.from_user:
        return
    text, kb = await models_panel_view(message.from_user.id, "models")
    await message.answer(text, reply_markup=kb)


@dp.callback_query(F.data.startswith("model:locked"))
async def on_model_locked(callback: CallbackQuery):
    # Старые клавиатуры могли сохранить прежний id модели — приводим к каталогу.
    key = normalize_model_key((callback.data or "").split(":")[-1])
    reason = _model_blocked.get(key)
    if reason:
        await callback.answer(f"Модель недоступна: {reason}.", show_alert=True)
        return
    info = MODEL_REGISTRY.get(key)
    if info and info["provider"] != "groq":
        client = PROVIDER_CLIENTS.get(info["provider"])
        if client is None or not client.configured:
            await callback.answer(
                "Шлюз не настроен на сервере: нет API-ключа GATEWAY_API.", show_alert=True
            )
            return
    await callback.answer("Эта модель сейчас недоступна.", show_alert=True)


@dp.callback_query(F.data.startswith("m:s:"))
async def on_pick_model(callback: CallbackQuery):
    if not callback.from_user or callback.message is None:
        return
    if callback.message.chat.type != ChatType.PRIVATE:
        await callback.answer("Модели доступны только в личке.", show_alert=True)
        return
    user_id = callback.from_user.id
    code = (callback.data or "").split(":", 2)[-1]
    key = MODEL_SHORT.get(code)
    if not key:
        await callback.answer("Модель не найдена, обнови список.", show_alert=True)
        return
    ok, text = await set_model_for_user(user_id, key)
    await callback.answer(("✅ " if ok else "⚠️ ") + plain_text(text)[:180], show_alert=not ok)
    if not ok:
        return
    info = MODEL_REGISTRY.get(normalize_model_key(key)) or {}
    panel, kb = await models_panel_view(
        user_id, "family", info.get("family"), info.get("section"), 0
    )
    try:
        await callback.message.edit_text(panel, reply_markup=kb)
    except Exception:
        pass


@dp.callback_query(F.data.startswith("m:i:"))
async def on_model_info(callback: CallbackQuery):
    if not callback.from_user or callback.message is None:
        return
    if callback.message.chat.type != ChatType.PRIVATE:
        await callback.answer("Только в личке.", show_alert=True)
        return
    code = (callback.data or "").split(":", 2)[-1]
    key = MODEL_SHORT.get(code)
    info = MODEL_REGISTRY.get(normalize_model_key(key or "")) or {}
    if not info:
        await callback.answer("Модель не найдена, обнови список.", show_alert=True)
        return
    text, kb = _model_info_view(
        code, info.get("section", studio.FREE), info.get("family", ""), 0
    )
    await callback.answer()
    try:
        await callback.message.edit_text(text, reply_markup=kb)
    except Exception as e:
        if "not modified" not in str(e).lower():
            try:
                await callback.message.answer(text, reply_markup=kb)
            except Exception:
                pass


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
    tab = "models"
    family: str | None = None
    section: str | None = None
    page = 0
    if action == "tab":
        tab = parts[2] if len(parts) > 2 else "models"
    elif action == "sec":
        section = parts[2] if len(parts) > 2 else studio.FREE
        if section not in (studio.FREE, studio.PAID):
            section = studio.FREE
        tab = "section"
    elif action == "fam":
        section = parts[2] if len(parts) > 2 else studio.FREE
        family = parts[3] if len(parts) > 3 else None
        try:
            page = int(parts[4]) if len(parts) > 4 else 0
        except (TypeError, ValueError):
            page = 0
        tab = "family"
    elif action == "toggle":
        field = parts[2] if len(parts) > 2 else "rich"
        if field == "rich":
            await save_toggle(user_id, "rich", not user_rich(user_id))
        elif field == "agent":
            await save_toggle(user_id, "agent", not user_agent(user_id))
            if user_agent(user_id):
                await callback.answer(
                    "🤖 Агент включён: до 8 шагов, максимальная глубина.",
                    show_alert=True,
                )
            else:
                await callback.answer("🤖 Агент выключен")
        tab = "modes"
    elif action == "effort":
        level = parts[2] if len(parts) > 2 else "off"
        if level in EFFORT_ORDER:
            await save_toggle(user_id, "deepthink", level)
        tab = "modes"
    elif action == "length":
        code = parts[2] if len(parts) > 2 else "normal"
        if code in LENGTH_ORDER:
            await save_toggle(user_id, "length", code)
        tab = "modes"
    elif action == "search":
        state = parts[2] if len(parts) > 2 else "on"
        await save_toggle(user_id, "search", "off" if state == "off" else "on")
        tab = "modes"
    elif action == "clear":
        await clear_chat_context(user_id)
        tab = "modes"
    text, kb = await models_panel_view(user_id, tab, family, section, page)
    try:
        await callback.message.edit_text(text, reply_markup=kb)
    except Exception as e:
        if "not modified" not in str(e).lower():
            try:
                await callback.message.answer(text, reply_markup=kb)
            except Exception:
                pass
    if action == "clear":
        await callback.answer("🧹 Контекст очищен")
    elif action in ("toggle", "effort", "length", "search"):
        pass
    else:
        await callback.answer()


@dp.message(Command("profile", "context", "whoami", "stats"))
async def cmd_profile(message: Message):
    """/profile — профиль, лимиты и статистика чата.

    Старые /context, /whoami, /stats оставлены как алиасы.
    """
    if not command_allowed(message):
        return
    chat_id = message.chat.id
    user = message.from_user
    lines = []
    if user:
        lines.append(f"👤 <b>{esc(user.full_name)}</b>")
        lines.append(f"ID: <code>{user.id}</code>")
        lines.append(f"Username: @{user.username or '—'}")
        if is_owner(user.id):
            lines.append("Роль: <b>владелец</b> (безлимит)")
        elif is_allowed_user(user.id):
            lines.append("Роль: <b>гость</b> из белого списка")
        lines.append("")
    if is_private(message) and user:
        section = user_section(user.id)
        current = selected_model(user.id)
        lines.append("<b>🧩 Модели</b>")
        lines.append(
            f"• Сейчас: <b>{esc(model_label(current))}</b> · "
            f"{studio.SECTION_LABELS.get(section, section)}"
        )
        free_used = await pool_used(user.id, studio.FREE)
        paid_used = await pool_used(user.id, studio.PAID)
        lines.append(
            f"• 🆓 Free: {fmt_int(free_used)} / "
            f"{fmt_int(studio.POOL_LIMITS[studio.FREE])} ток."
        )
        lines.append(
            f"• 💎 Paid: {fmt_int(paid_used)} / "
            f"{fmt_int(studio.POOL_LIMITS[studio.PAID])} ток."
        )
        lines.append(f"• Сброс пулов: {studio.reset_text()}")
        lines.append("")
    lines.append("<b>📊 Статистика чата</b>")
    lines.append(f"• Обработано запросов: <b>{_stats.get(chat_id, 0)}</b>")
    lines.append(
        f"• В контексте: <b>{get_history_size(chat_id)}</b> из {HISTORY_LIMIT}"
    )
    lines.append(f"• Время работы: <b>{format_uptime()}</b>")
    await reply_any(message, "\n".join(lines))


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
    item = _balance.get("gateway") or {}
    if item.get("error"):
        return "Премиум: ошибка"
    if item.get("amount") is None:
        return "Премиум: —"
    line = f"Премиум: {fmt_number(item['amount'])}"
    if item.get("total"):
        line += f" из {fmt_number(item['total'])}"
    return line


def balance_block() -> str:
    item = _balance.get("gateway") or {}
    lines = ["💳 <b>Баланс шлюза</b>"]
    if item.get("error"):
        lines.append(f"⚠️ {esc(str(item['error'])[:160])}")
        if item.get("amount") is None:
            return "\n".join(lines)
    amount = item.get("amount")
    total = item.get("total")
    spent = item.get("spent")
    if amount is None:
        lines.append("Нет данных.")
        return "\n".join(lines)
    line = f"• Осталось: <b>{fmt_number(amount)}</b>"
    if total:
        percent = amount / total * 100 if total else 0
        line += f" из {fmt_number(total)} ({percent:.1f}%)"
    lines.append(line)
    if spent is not None:
        lines.append(f"• Потрачено: {fmt_number(spent)}")
    updated = float(item.get("updated") or 0)
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


async def refresh_balance(provider: str = "gateway") -> dict:
    item: dict = {
        "amount": None,
        "total": None,
        "spent": None,
        "updated": time.time(),
        "error": "",
    }
    prev = _balance.get("gateway") or {}
    try:
        headers: dict[str, str] = {}
        client = PROVIDER_CLIENTS.get("gateway")
        api_key = getattr(client, "api_key", "") if client else ""
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = await http_client.get(
            GATEWAY_BALANCE_URL,
            timeout=30,
            follow_redirects=True,
            headers=headers,
        )
        response.raise_for_status()
        left = None
        total = None
        spent = None
        # Новый шлюз отдаёт JSON: {"object":"balance","token_balance":N}.
        try:
            data = response.json()
        except Exception:
            data = None
        if isinstance(data, dict):
            for field_name in ("token_balance", "balance", "credits"):
                value = data.get(field_name)
                if value is not None:
                    left = float(value)
                    break
        if left is None:
            # Запасной путь: старая HTML-страница «Потрачено/Лимит/Осталось».
            spent = _page_stat(response.text, "Потрачено")
            total = _page_stat(response.text, "Лимит")
            left = _page_stat(response.text, "Осталось")
            if left is None and total is not None and spent is not None:
                left = max(0, total - spent)
        if left is None:
            raise RuntimeError("не удалось разобрать ответ баланса")
        item["amount"] = float(left)
        item["total"] = float(total) if total else None
        item["spent"] = float(spent) if spent is not None else None
    except Exception as e:
        item["error"] = str(e)
        if prev.get("amount") is not None:
            item["amount"] = prev.get("amount")
            item["total"] = prev.get("total")
            item["spent"] = prev.get("spent")
    _balance["gateway"] = item
    return item


async def check_balance_alert(provider: str = "gateway") -> None:
    item = _balance.get("gateway") or {}
    amount = item.get("amount")
    total = item.get("total")
    if amount is None or not total:
        return
    percent = amount / total * 100
    if percent >= BALANCE_ALERT_PERCENT:
        _balance_alerted.discard("gateway")
        return
    if "gateway" in _balance_alerted:
        return
    _balance_alerted.add("gateway")
    try:
        await bot.send_message(
            OWNER_ID,
            f"⚠️ Баланс <b>{PROVIDER_NAMES['gateway']}</b> ниже "
            f"{BALANCE_ALERT_PERCENT}%: {fmt_number(amount)} из "
            f"{fmt_number(total)} ({percent:.1f}%).",
        )
    except Exception:
        pass


async def refresh_all_balances(notify: bool = False) -> None:
    await refresh_balance("gateway")
    await check_balance_alert("gateway")
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


def admin_main_text() -> str:
    gateway_client_info = PROVIDER_CLIENTS.get("gateway")
    configured = bool(gateway_client_info and gateway_client_info.configured)
    usage = _today_usage_cache
    today_total = sum(usage.values())
    today_users = len({uid for (uid, _prov) in usage})
    return (
        "🛠 <b>Админ-панель Мудрого Мха</b>\n\n"
        "<table><thead><tr>"
        "<th>Шлюз</th><th>Модели</th><th>Пул paid</th>"
        "</tr></thead><tbody><tr>"
        f"<td>{'настроен' if configured else 'нет ключа'}</td>"
        f"<td>{len(GATEWAY_MODELS)}</td>"
        f"<td>{fmt_int(studio.POOL_LIMITS[studio.PAID])} ток./день</td>"
        "</tr></tbody></table>\n\n"
        "<table><thead><tr>"
        "<th>Сегодня</th><th>Люди</th><th>Групп</th><th>Блок</th>"
        "</tr></thead><tbody><tr>"
        f"<td>{fmt_int(today_total)}</td>"
        f"<td>{fmt_int(today_users)}</td>"
        f"<td>{len(_groups_cache)}</td>"
        f"<td>{len(_dm_blacklist)}</td>"
        "</tr></tbody></table>\n\n"
        f"💰 {balance_short()}\n\n"
        f"{db_usage_line(_db_size_cache)}\n\n"
        "Выбери раздел:"
    )


def admin_main_kb() -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(text="📊 Статистика", callback_data="adm:stats:t"),
            InlineKeyboardButton(text="📝 Запросы", callback_data="adm:req:0"),
        ],
        [
            InlineKeyboardButton(text="👥 Группы", callback_data="adm:groups:0"),
            InlineKeyboardButton(text="✅ ЛС", callback_data="adm:wl:0"),
        ],
        [
            InlineKeyboardButton(text="🔑 Доступ", callback_data="adm:acc:gateway:0"),
            InlineKeyboardButton(text="📨 Сводка", callback_data="adm:summary"),
        ],
        [
            InlineKeyboardButton(text="⚙️ Настройки", callback_data="adm:set"),
        ],
    ]
    return admin_kb(rows)


async def stats_text(stats: dict, period: str, flt: dict) -> str:
    today = msk_today()
    today_usage = await storage.usage_for_day(today, "gateway")
    today_total = sum(today_usage.values())
    today_users = len({uid for (uid, _prov) in today_usage})
    models = merge_stats_models(stats.get("models"), limit=8)
    total_tokens = sum(int(m.get("total_tokens") or 0) for m in models)
    errors = int(stats["request_errors"]) + int(stats["call_errors"])

    lines = [f"📊 <b>Статистика</b> · {admin_period_label(flt)}", ""]
    lines.append(
        "<table><thead><tr>"
        "<th>Запросы</th><th>Вызовы</th><th>Ошибки</th>"
        "</tr></thead><tbody><tr>"
        f"<td>{fmt_int(stats['requests'])}</td>"
        f"<td>{fmt_int(stats['calls'])}</td>"
        f"<td>{fmt_int(errors)}</td>"
        "</tr></tbody></table>"
    )
    lines.append(
        "<table><thead><tr>"
        "<th>Токены</th><th>Задержка</th><th>Сегодня</th>"
        "</tr></thead><tbody><tr>"
        f"<td>{fmt_int(total_tokens)}</td>"
        f"<td>{fmt_int(stats['avg_latency'])} мс</td>"
        f"<td>{fmt_int(today_total)}</td>"
        "</tr></tbody></table>"
    )
    lines.append(
        "<table><thead><tr>"
        "<th>Люди</th><th>Чаты</th><th>Пулы</th>"
        "</tr></thead><tbody><tr>"
        f"<td>{fmt_int(today_users)}</td>"
        f"<td>{fmt_int(stats.get('chats_total') or 0)}</td>"
        "<td>250к / 100к</td>"
        "</tr></tbody></table>"
    )
    lines.append("")

    lines.append("<b>Модели</b>")
    if models:
        rows = []
        for m in models:
            rows.append(
                "<tr>"
                f"<td>{esc(m['model'])}</td>"
                f"<td>{fmt_int(m['calls'])}</td>"
                f"<td>{fmt_int(m['total_tokens'])}</td>"
                "</tr>"
            )
        lines.append(
            "<table><thead><tr>"
            "<th>Модель</th><th>Вызовы</th><th>Токены</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )
    else:
        lines.append("<p>Пока нет данных.</p>")
    lines.append("")

    lines.append("<b>Люди и чаты</b>")
    top_users = (stats.get("top_users") or [])[:5]
    await resolve_user_names([u.get("user_id") for u in top_users])
    merged: list[tuple[str, str, int]] = []
    for u in top_users:
        chats_n = int(u.get("chats") or 0)
        where = "👤 личка" if chats_n <= 1 else f"👤 {chats_n} чата"
        merged.append(
            (user_tag(u["user_id"], u.get("username", "")), where, int(u["requests"]))
        )
    for c in stats.get("top_chats") or []:
        if c.get("chat_type") == "private":
            continue
        chat_id = c["chat_id"]
        title = (_groups_cache.get(int(chat_id)) or {}).get("title") or ""
        name = esc(title) if title else f"<code>{chat_id}</code>"
        merged.append(
            (name, f"💬 {esc(c.get('chat_type') or 'чат')}", int(c["requests"]))
        )
    merged = merged[:8]
    if merged:
        rows = [
            "<tr>"
            f"<td>{name}</td>"
            f"<td>{where}</td>"
            f"<td>{fmt_int(requests)}</td>"
            "</tr>"
            for name, where, requests in merged
        ]
        lines.append(
            "<table><thead><tr>"
            "<th>Кто</th><th>Где</th><th>Запросы</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )
    else:
        lines.append("<p>Пока нет данных.</p>")
    lines.append("")

    ctx = stats.get("contexts") or {}
    lines.append("<b>Контекст</b>")
    lines.append(
        "<table><thead><tr>"
        "<th>Чатов</th><th>Сообщений</th><th>В группах</th>"
        "</tr></thead><tbody><tr>"
        f"<td>{fmt_int(ctx.get('chats', 0))}</td>"
        f"<td>{fmt_int(ctx.get('messages', 0))}</td>"
        f"<td>{fmt_int(stats['group_messages'])}</td>"
        "</tr></tbody></table>"
    )
    lines.append("")
    lines.append(balance_block())
    lines.append(
        f"<b>Сервис:</b> {format_uptime()} · "
        f"{'Postgres' if storage.available else 'память'} · "
        f"rich {'✓' if RICH_SUPPORTED else '—'}"
    )
    return "\n".join(lines)


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


async def groups_view(page: int) -> tuple[str, InlineKeyboardMarkup]:
    groups = sorted(_groups_cache.values(), key=lambda g: (g.get("title") or "").lower())
    page_items, page, total = paginate(groups, page)
    # В кэше могли остаться старые названия: подтягиваем актуальные из Telegram.
    for g in page_items:
        chat_id = int(g["chat_id"])
        try:
            chat = await bot.get_chat(chat_id)
        except Exception:
            continue
        title = getattr(chat, "title", None) or getattr(chat, "full_name", None) or ""
        if title and title != (g.get("title") or ""):
            g["title"] = title
            _groups_cache[chat_id] = dict(g)
            await storage.add_group(chat_id, title)

    groups = sorted(_groups_cache.values(), key=lambda g: (g.get("title") or "").lower())
    page_items, page, total = paginate(groups, page)
    lines = [
        f"👥 <b>Белый список групп</b> · всего {len(groups)}",
        "",
    ]
    if page_items:
        table_rows = "".join(
            "<tr>"
            f"<td>{esc(g.get('title') or 'без названия')}</td>"
            f"<td><code>{g['chat_id']}</code></td>"
            "</tr>"
            for g in page_items
        )
        lines.append(
            "<table><thead><tr><th>Группа</th><th>ID</th></tr></thead><tbody>"
            + table_rows
            + "</tbody></table>"
        )
    else:
        lines.append("<p>Список пуст.</p>")
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
    lines = [
        f"🚫 <b>Чёрный список ЛС</b> · всего {len(items)}",
        "",
    ]
    if page_items:
        table_rows = "".join(
            "<tr>"
            f"<td>{esc(u.get('full_name') or u.get('username') or 'без имени')}</td>"
            f"<td><code>{u['user_id']}</code></td>"
            "</tr>"
            for u in page_items
        )
        lines.append(
            "<table><thead><tr><th>Кто</th><th>ID</th></tr></thead><tbody>"
            + table_rows
            + "</tbody></table>"
        )
    else:
        lines.append("<p>Список пуст.</p>")
    rows = [[InlineKeyboardButton(text="➕ Заблокировать", callback_data="adm:bladd")]]
    for u in page_items:
        name = (u.get("full_name") or u.get("username") or str(u["user_id"]))[:25]
        rows.append(
            [InlineKeyboardButton(text=f"🗑 {name}", callback_data=f"adm:bldel:{u['user_id']}")]
        )
    rows.append(paginator_kb("adm:bl", page, total))
    return "\n".join(lines), admin_kb(rows)


def wl_view(page: int) -> tuple[str, InlineKeyboardMarkup]:
    items = sorted(
        _allowed_users_cache.values(),
        key=lambda x: (x.get("added_at") or datetime.min.replace(tzinfo=timezone.utc)),
    )
    page_items, page, total = paginate(items, page)
    lines = [
        f"✅ <b>Белый список ЛС</b> · всего {len(items)}",
        "",
        "Кто в списке — тому бот отвечает в личке. Остальным доступ закрыт.",
        "",
    ]
    if page_items:
        table_rows = "".join(
            "<tr>"
            f"<td>{esc(u.get('full_name') or u.get('username') or 'без имени')}</td>"
            f"<td><code>{u['user_id']}</code></td>"
            "</tr>"
            for u in page_items
        )
        lines.append(
            "<table><thead><tr><th>Кто</th><th>ID</th></tr></thead><tbody>"
            + table_rows
            + "</tbody></table>"
        )
    else:
        lines.append("<p>Список пуст — в личку пускаем только владельца.</p>")
    rows = [
        [InlineKeyboardButton(text="➕ Добавить пользователя", callback_data="adm:wladd")]
    ]
    for u in page_items:
        name = (u.get("full_name") or u.get("username") or str(u["user_id"]))[:25]
        rows.append(
            [InlineKeyboardButton(text=f"🗑 {name}", callback_data=f"adm:wldel:{u['user_id']}")]
        )
    rows.append(paginator_kb("adm:wl", page, total))
    return "\n".join(lines), admin_kb(rows)


async def access_view(provider: str = "gateway", page: int = 0) -> tuple[str, InlineKeyboardMarkup]:
    today = msk_today()
    usage = await storage.usage_for_day(today, "gateway")
    items = sorted(
        (
            {"user_id": int(uid), "requests": int(count)}
            for (uid, _prov), count in usage.items()
        ),
        key=lambda x: x["requests"],
        reverse=True,
    )
    total_today = sum(u["requests"] for u in items)
    page_items, page, total = paginate(items, page)
    await resolve_user_names([u["user_id"] for u in page_items])
    lines = [
        "🔑 <b>Доступ к шлюзу</b>",
        "",
        "Модели открыты белому списку. Пулы в сутки: "
        f"free {fmt_int(studio.POOL_LIMITS[studio.FREE])} ток., "
        f"paid {fmt_int(studio.POOL_LIMITS[studio.PAID])} ток.",
        f"👑 Владелец: <code>{OWNER_ID}</code> — безлимит",
        "",
        "<table><thead><tr>"
        "<th>Сегодня</th><th>Люди</th><th>Пулы</th>"
        "</tr></thead><tbody><tr>"
        f"<td>{fmt_int(total_today)}</td>"
        f"<td>{fmt_int(len(items))}</td>"
        f"<td>250к / 100к</td>"
        "</tr></tbody></table>",
        "",
    ]
    if page_items:
        table_rows = "".join(
            "<tr>"
            f"<td>{user_tag(u['user_id'])}</td>"
            f"<td>{u['requests']} запросов</td>"
            "</tr>"
            for u in page_items
        )
        lines.append(
            "<table><thead><tr><th>Кто</th><th>Израсходовано</th></tr></thead><tbody>"
            + table_rows
            + "</tbody></table>"
        )
    else:
        lines.append("<p>Сегодня ещё никто не пользовался.</p>")
    rows: list[list[InlineKeyboardButton]] = []
    if items:
        rows.append(paginator_kb("adm:acc:gateway", page, total))
    else:
        rows.append(
            [InlineKeyboardButton(text="🔄 Обновить", callback_data="adm:acc:gateway:0")]
        )
    rows.append([InlineKeyboardButton(text="🏠 Меню", callback_data="adm:main")])
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
        "",
        "<table><thead><tr>"
        "<th>Всего</th><th>Страница</th><th>Модель</th>"
        "</tr></thead><tbody><tr>"
        f"<td>{fmt_int(total)}</td>"
        f"<td>{page + 1} из {total_pages}</td>"
        f"<td>{esc(flt.get('model') or 'все')}</td>"
        "</tr></tbody></table>",
        "",
    ]
    if not rows:
        lines.append("<p>Ничего не найдено.</p>")
    await resolve_user_names([r.get("user_id") for r in rows if r.get("user_id")])
    if rows:
        table_rows: list[str] = []
        for r in rows:
            created = r.get("created_at")
            if isinstance(created, datetime):
                created = created.astimezone(MSK).strftime("%d.%m %H:%M")
            elif not created:
                created = "—"
            status = r.get("status") or "?"
            if status == "logged":
                icon = "📝"
            elif status == "ok":
                icon = "✅"
            elif status == "running":
                icon = "⏳"
            else:
                icon = "❌"
            who = (
                user_tag(r.get("user_id"), r.get("username") or "")
                if r.get("user_id")
                else "—"
            )
            model_name = (r.get("model") or "—")[:28]
            prompt = (r.get("prompt_preview") or "").replace("\n", " ").strip()
            if len(prompt) > 90:
                prompt = prompt[:90] + "…"
            meta: list[str] = []
            if r.get("latency_ms"):
                meta.append(f"{fmt_int(r.get('latency_ms'))} мс")
            if r.get("tools"):
                meta.append("🛠 " + esc(r["tools"])[:40])
            tokens = fmt_int(r.get("tokens") or 0)
            if meta:
                tokens += " · " + " · ".join(meta)
            if r.get("error"):
                prompt = "⚠️ " + (prompt or "ошибка")
            table_rows.append(
                "<tr>"
                f"<td>{icon} {created}</td>"
                f"<td>{who}</td>"
                f"<td>{esc(model_name)}</td>"
                f"<td>{esc(prompt) or '—'}</td>"
                f"<td>{tokens}</td>"
                "</tr>"
            )
        lines.append(
            "<table><thead><tr>"
            "<th>Когда</th><th>Кто</th><th>Модель</th><th>Запрос</th><th>Токены</th>"
            "</tr></thead><tbody>" + "".join(table_rows) + "</tbody></table>"
        )

        calls = await storage.tool_calls_for_requests([r.get("id") for r in rows])
        if calls:
            call_rows: list[str] = []
            for c in calls[:12]:
                when = c.get("created_at")
                if isinstance(when, datetime):
                    when = when.astimezone(MSK).strftime("%d.%m %H:%M")
                else:
                    when = "—"
                cstatus = c.get("status") or "?"
                cicon = "✅" if cstatus == "ok" else "❌"
                args = (c.get("args_preview") or "").replace("\n", " ").strip()
                if len(args) > 60:
                    args = args[:60] + "…"
                latency = (
                    f"{fmt_int(c.get('latency_ms'))} мс"
                    if c.get("latency_ms")
                    else "—"
                )
                call_rows.append(
                    "<tr>"
                    f"<td>{cicon} {when}</td>"
                    f"<td><code>{esc(c.get('tool') or '?')}</code></td>"
                    f"<td>{esc(args) or '—'}</td>"
                    f"<td>{latency}</td>"
                    "</tr>"
                )
            lines.append("")
            lines.append("<b>🛠 Вызовы инструментов</b>")
            lines.append(
                "<table><thead><tr>"
                "<th>Когда</th><th>Инструмент</th><th>Аргументы</th><th>Время</th>"
                "</tr></thead><tbody>" + "".join(call_rows) + "</tbody></table>"
            )
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
    return "\n".join(lines), admin_kb(rows_kb)


def filter_view(flt: dict) -> tuple[str, InlineKeyboardMarkup]:
    rows_html = [
        "<table><thead><tr><th>Фильтр</th><th>Значение</th></tr></thead><tbody>",
        f"<tr><td>Период</td><td>{admin_period_label(flt)}</td></tr>",
        f"<tr><td>Пользователь</td><td>{esc(flt.get('user_id') or 'все')}</td></tr>",
        f"<tr><td>Модель</td><td>{esc(flt.get('model') or 'все')}</td></tr>",
    ]
    if flt.get("date_from"):
        rows_html.append(
            f"<tr><td>Дата</td><td>{esc(flt['date_from'])}</td></tr>"
        )
    rows_html.append("</tbody></table>")
    text = "🔎 <b>Фильтры запросов</b>\n\n" + "".join(rows_html)

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
        ],
        *[
            [
                InlineKeyboardButton(
                    text=mark(key, flt.get("model"), model_label(key)),
                    callback_data=f"adm:fmodel:{key}",
                )
                for key in pair
            ]
            for pair in (
                GATEWAY_MODELS[i : i + 2]
                for i in range(0, len(GATEWAY_MODELS), 2)
            )
        ],
        [
            InlineKeyboardButton(text="🗑 Сбросить", callback_data="adm:fclear"),
            InlineKeyboardButton(text="📝 К запросам", callback_data="adm:req:0"),
        ],
    ]
    return text, admin_kb(rows)


async def settings_view(refresh: bool = False) -> tuple[str, InlineKeyboardMarkup]:
    if refresh:
        await refresh_all_balances()
    size = await storage.db_size()
    ctx = await storage.context_summary()
    mode = "Postgres (Neon)" if storage.available else "память (DATABASE_URL не задан)"
    client = PROVIDER_CLIENTS.get("gateway")
    gateway_ok = bool(client and client.configured)
    text = (
        "⚙️ <b>Настройки</b>\n\n"
        "<table><thead><tr><th>Параметр</th><th>Значение</th></tr></thead><tbody>"
        f"<tr><td>Хранилище</td><td>{mode}</td></tr>"
        f"<tr><td>Контекст</td><td>{fmt_int(ctx['chats'])} чатов · "
        f"{fmt_int(ctx['messages'])} сообщений</td></tr>"
        "<tr><td>Логи</td><td>хранятся вечно</td></tr>"
        "<tr><td>Сводка</td><td>каждый день в 04:00 МСК</td></tr>"
        f"<tr><td>Шлюз</td><td>{'настроен' if gateway_ok else 'нет GATEWAY_API'} · "
        f"{len(GATEWAY_MODELS)} "
        f"{plural_ru(len(GATEWAY_MODELS), 'модель', 'модели', 'моделей')}</td></tr>"
        f"<tr><td>Лимит</td><td>free {fmt_int(studio.POOL_LIMITS[studio.FREE])} · "
        f"paid {fmt_int(studio.POOL_LIMITS[studio.PAID])} токенов/день, "
        "владелец — безлимит</td></tr>"
        f"<tr><td>История</td><td>{fmt_int(PREMIUM_HISTORY_CHARS_LIMIT)} символов · "
        f"ответ до {fmt_int(PREMIUM_MAX_TOKENS)} токенов</td></tr>"
        "</tbody></table>\n\n"
        f"{db_usage_line(size)}\n\n"
        f"{balance_block()}\n\n"
        "Очистка удаляет запросы, вызовы моделей и инструментов старше указанного срока. "
        "Контексты и списки не трогает."
    )
    rows = [
        [
            InlineKeyboardButton(text="🔄 Обновить", callback_data="adm:set"),
            InlineKeyboardButton(text="🏠 Меню", callback_data="adm:main"),
        ],
        [InlineKeyboardButton(text="⬇️ CSV всех запросов", callback_data="adm:csvall")],
        [InlineKeyboardButton(text="🧹 Очистить старые логи", callback_data="adm:cleanup")],
    ]
    return text, admin_kb(rows)


async def admin_section(
    section: str, page: int = 0, period: str | None = None, provider: str | None = None
) -> tuple[str, InlineKeyboardMarkup]:
    global _db_size_cache, _today_usage_cache
    if section == "main":
        _db_size_cache = await storage.db_size()
        _today_usage_cache = await storage.usage_for_day(msk_today(), "gateway")
        return admin_main_text(), admin_main_kb()
    if section == "groups":
        return await groups_view(page)
    if section == "bl":
        return blacklist_view(page)
    if section == "wl":
        return wl_view(page)
    if section == "acc":
        return await access_view(provider or "gateway", page)
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
        return await stats_text(data, flt.get("period", "7"), flt), stats_kb(flt.get("period", "7"))
    return admin_main_text(), admin_main_kb()


_ADMIN_SCREENS: dict[int, int] = {}


def admin_article(text: str) -> str:
    """Превращает обычный HTML админки в rich-статью из абзацев."""
    blocks = [b.strip() for b in re.split(r"\n{2,}", text or "") if b.strip()]
    parts: list[str] = []
    for block in blocks:
        if re.match(
            r"^<(?:h[1-6]|p|ul|ol|blockquote|pre|details|table|footer)\b",
            block,
            re.IGNORECASE,
        ):
            parts.append(block)
            continue
        for line in block.split("\n"):
            line = line.strip()
            if line:
                parts.append(f"<p>{line}</p>")
    return "\n".join(parts)


async def send_admin_article(
    chat_id: int,
    text: str,
    kb: InlineKeyboardMarkup | None,
    old_id: int | None = None,
    *,
    track: bool = True,
) -> Message | None:
    sent: Message | None = None
    if RICH_SUPPORTED:
        try:
            sent = await bot.send_rich_message(
                chat_id=chat_id,
                rich_message=InputRichMessage(html=sanitize_rich(admin_article(text))),
                reply_markup=kb,
            )
        except Exception as e:
            logging.warning(f"admin rich send failed: {e}; fallback to text")
    if sent is None:
        try:
            # В обычном сообщении <table> недопустим: переводим статью в текст.
            plain = classic_from_rich(text) or plain_from_rich(text) or text
            sent = await bot.send_message(
                chat_id, plain, reply_markup=kb, disable_web_page_preview=True
            )
        except Exception as e:
            logging.warning(f"admin text send failed: {e}")
            return None
    if old_id and sent and old_id != sent.message_id:
        try:
            await bot.delete_message(chat_id, old_id)
        except Exception:
            pass
    if track and sent is not None:
        _ADMIN_SCREENS[chat_id] = sent.message_id
    return sent


async def edit_admin(callback: CallbackQuery, text: str, kb: InlineKeyboardMarkup) -> None:
    if callback.message is None:
        return
    chat_id = callback.message.chat.id
    old_id = _ADMIN_SCREENS.get(chat_id) or callback.message.message_id
    await send_admin_article(chat_id, text, kb, old_id)


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
    await send_admin_article(message.chat.id, text, kb)


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


async def _cleanup_admin_input(message: Message, state: dict | None) -> None:
    """Убирает подсказку и сообщение с ID, чтобы не засорять чат."""
    prompt_id = (state or {}).get("prompt_id")
    if prompt_id:
        try:
            await bot.delete_message(message.chat.id, prompt_id)
        except Exception:
            pass
    try:
        await message.delete()
    except Exception:
        pass


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
            await _cleanup_admin_input(message, state)
            text, kb = await admin_section("groups")
            await send_admin_article(
                message.chat.id, text, kb, old_id=_ADMIN_SCREENS.get(OWNER_ID)
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

        if action == "whitelist_id":
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
            if int(user_id) == OWNER_ID:
                _admin_input.pop(OWNER_ID, None)
                await _cleanup_admin_input(message, state)
                await message.answer("Владелец и так имеет доступ.")
                return True
            await storage.add_allowed_user(user_id, uname, full_name, OWNER_ID)
            ALLOWED_USERS.add(int(user_id))
            _allowed_users_cache[int(user_id)] = {
                "user_id": int(user_id),
                "username": uname,
                "full_name": full_name,
                "added_at": msk_now(),
            }
            await storage.add_admin_action(OWNER_ID, "whitelist_add", str(user_id))
            _admin_input.pop(OWNER_ID, None)
            await _cleanup_admin_input(message, state)
            text, kb = await admin_section("wl")
            await send_admin_article(
                message.chat.id, text, kb, old_id=_ADMIN_SCREENS.get(OWNER_ID)
            )
            return True

        if action == "grant_note":
            await message.answer(
                "Доступ к шлюзу открыт всем: 25 запросов в день, "
                "у владельца без лимита. Выдавать ничего не надо.",
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
            await send_admin_article(message.chat.id, txt, kb)
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
            await send_admin_article(message.chat.id, txt, kb)
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
        if action == "dash":
            end = datetime.now(timezone.utc)
            start = end - timedelta(hours=24)
            data = await storage.stats(start, end)
            data = dict(data)
            data["models"] = merge_stats_models(data.get("models"), limit=6)
            try:
                mode = await send_dashboard_message(OWNER_ID, data, "24 часа")
                await callback.answer(
                    "Дашборд отправлен."
                    if mode == "widget"
                    else "Виджеты недоступны — отправил текстом."
                )
            except Exception as e:
                logging.warning(f"admin dashboard failed: {e}")
                await callback.answer(
                    plain_text(f"Не удалось собрать дашборд: {e}")[:180],
                    show_alert=True,
                )
            return
        if action == "summary":
            await send_daily_summary(mark=False)
            await callback.answer("Сводка отправлена.")
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
        if action == "wl":
            page = int(parts[2]) if len(parts) > 2 else 0
            text, kb = await admin_section("wl", page)
            await edit_admin(callback, text, kb)
            await callback.answer()
            return
        if action == "acc":
            page = int(parts[3]) if len(parts) > 3 else 0
            provider = parts[2] if len(parts) > 2 else "gateway"
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
        if action == "req":
            page = int(parts[2]) if len(parts) > 2 else 0
            text, kb = await admin_section("req", page)
            await edit_admin(callback, text, kb)
            await callback.answer()
            return
        if action == "set":
            text, kb = await settings_view(refresh=True)
            await edit_admin(callback, text, kb)
            await callback.answer("Баланс обновлён.")
            return
        if action == "filter":
            text, kb = filter_view(admin_filters(OWNER_ID))
            await edit_admin(callback, text, kb)
            await callback.answer()
            return

        if action == "gadd":
            sent = await callback.message.answer(
                "Отправь ID группы (например <code>-1001234567890</code>) "
                "или перешли любое сообщение из этой группы.\n\n/cancel — отмена."
            )
            _admin_input[OWNER_ID] = {"action": "group_id", "prompt_id": sent.message_id}
            await callback.answer()
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

        if action == "wladd":
            sent = await callback.message.answer(
                "Отправь ID пользователя или перешли его сообщение.\n\n/cancel — отмена."
            )
            _admin_input[OWNER_ID] = {"action": "whitelist_id", "prompt_id": sent.message_id}
            await callback.answer()
            return
        if action == "wldel":
            user_id = int(parts[2])
            item = _allowed_users_cache.get(user_id) or {}
            name = item.get("full_name") or item.get("username") or str(user_id)
            await edit_admin(
                callback,
                f"Убрать <b>{esc(name)}</b> из белого списка ЛС?",
                confirm_kb(f"adm:wldelok:{user_id}", "adm:wl:0"),
            )
            await callback.answer()
            return
        if action == "wldelok":
            user_id = int(parts[2])
            await storage.remove_allowed_user(user_id)
            ALLOWED_USERS.discard(user_id)
            _allowed_users_cache.pop(user_id, None)
            await storage.add_admin_action(OWNER_ID, "whitelist_remove", str(user_id))
            text, kb = await admin_section("wl")
            await edit_admin(callback, text, kb)
            await callback.answer("Убран из белого списка.")
            return

        if action == "accadd":
            await callback.answer()
            await callback.message.answer(
                "Доступ к шлюзу открыт всем: 25 запросов в день, "
                "у владельца без лимита. Выдавать доступ вручную "
                "больше не нужно — список выше показывает расход "
                "за сегодня."
            )
            return
        if action == "accdel":
            provider = parts[2] if len(parts) > 2 else "gateway"
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
            provider = parts[2] if len(parts) > 2 else "gateway"
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


SUMMARY_MESSAGE_KEY = "daily_summary_message_id"


async def send_daily_summary(mark: bool = True) -> None:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=24)
    data = await storage.stats(start, end)
    errors = int(data["request_errors"]) + int(data["call_errors"])
    lines = [
        "📊 <b>Сводка за 24 часа</b>",
        "",
        "<table><thead><tr>"
        "<th>Запросы</th><th>Вызовы</th><th>Ошибки</th>"
        "</tr></thead><tbody><tr>"
        f"<td>{fmt_int(data['requests'])}</td>"
        f"<td>{fmt_int(data['calls'])}</td>"
        f"<td>{fmt_int(errors)}</td>"
        "</tr></tbody></table>",
    ]

    models = merge_stats_models(data.get("models"), limit=6)
    if models:
        rows = "".join(
            "<tr>"
            f"<td>{esc(m['model'])}</td>"
            f"<td>{fmt_int(m['calls'])}</td>"
            f"<td>{fmt_int(m['total_tokens'])}</td>"
            "</tr>"
            for m in models
        )
        lines.append("")
        lines.append("<b>Модели</b>")
        lines.append(
            "<table><thead><tr>"
            "<th>Модель</th><th>Вызовы</th><th>Токены</th>"
            "</tr></thead><tbody>" + rows + "</tbody></table>"
        )

    if data.get("top_users"):
        top_users = data["top_users"][:5]
        await resolve_user_names([u.get("user_id") for u in top_users])
        rows = "".join(
            "<tr>"
            f"<td>{user_tag(u['user_id'], u.get('username', ''))}</td>"
            f"<td>{fmt_int(u.get('chats') or 0)}</td>"
            f"<td>{fmt_int(u['requests'])}</td>"
            "</tr>"
            for u in top_users
        )
        lines.append("")
        lines.append("<b>Люди</b>")
        lines.append(
            "<table><thead><tr>"
            "<th>Кто</th><th>Чаты</th><th>Запросы</th>"
            "</tr></thead><tbody>" + rows + "</tbody></table>"
        )

    old_raw = await storage.get_setting(SUMMARY_MESSAGE_KEY)
    try:
        old_id = int(old_raw) if old_raw else None
    except (TypeError, ValueError):
        old_id = None

    kb = admin_kb(
        [
            [
                InlineKeyboardButton(text="🔄 Обновить", callback_data="adm:summary"),
                InlineKeyboardButton(text="🏠 Меню", callback_data="adm:main"),
            ]
        ]
    )
    sent = await send_admin_article(OWNER_ID, "\n".join(lines), kb, old_id=old_id)
    if sent is not None:
        await storage.set_setting(SUMMARY_MESSAGE_KEY, str(sent.message_id))
    if mark:
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
        if isinstance(event, Message) and event.from_user:
            # Запоминаем имя, чтобы в статистике были не только ID.
            remember_user(event.from_user)
        if (
            isinstance(event, Message)
            and event.chat.type == ChatType.PRIVATE
            and event.from_user
        ):
            blocked = is_blacklisted(event.from_user.id) or not is_allowed_user(
                event.from_user.id
            )
            if blocked:
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
                        message_text="Пример: @M0XAIBOT расскажи про Python",
                        link_preview_options=LinkPreviewOptions(is_disabled=True),
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
                        message_text=truncate_utf16(plain_answer, 4000),
                        link_preview_options=LinkPreviewOptions(is_disabled=True),
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
    if len(_photo_sessions[key]) >= PHOTO_LIMIT:
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
    messages = messages[:PHOTO_LIMIT]
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
    if not model:
        await notify_pool_exhausted(
            first,
            user_section(user.id if user else None) if private else studio.FREE,
        )
        return
    provider = provider_of(model)

    caption = next((m.caption.strip() for m in messages if m.caption), "")
    question = strip_triggers(caption) if caption else ""
    if not question:
        question = "реши подробно" if provider == "gateway" else "Что на фото?"

    track = await begin_request(
        first,
        kind="photo",
        prompt=question,
        model=model,
    )
    started = time.perf_counter()

    info = MODEL_REGISTRY.get(normalize_model_key(model)) or {}
    section = info.get("section", studio.FREE)

    if provider == "gateway" and private:
        async with chat_lock(chat_id):
            await process_gateway_photos(
                messages, question, track, started, model,
                provider=provider, section=section,
            )
        return

    if info.get("vision"):
        async with chat_lock(chat_id):
            await process_gateway_photos(
                messages, question, track, started, model,
                provider=provider, section=section,
            )
        return

    await process_groq_photos(messages, question, track, started, model, section)


async def process_groq_photos(
    messages: list[Message],
    question: str,
    track: Track,
    started: float,
    model_key: str = "groq",
    section: str = studio.FREE,
) -> None:
    first = messages[0]
    chat_id = first.chat.id
    private = is_private(first)
    turn = Turn(
        message=first,
        chat_id=chat_id,
        is_private=private,
        model=model_key,
        section=section,
        track=track,
    )
    await turn.status.show("👁 Смотрю фото…")

    descriptions: list[str] = []
    failed: list[int] = []
    for index, m in enumerate(messages, 1):
        try:
            data = await download_image_bytes(m)
            desc = await describe_image(data, track=track, section=studio.FREE)
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
            model=model_key,
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


async def process_gateway_photos(
    messages: list[Message],
    question: str,
    track: Track,
    started: float,
    model_key: str,
    *,
    provider: str | None = None,
    section: str = studio.PAID,
) -> None:
    """Модели со зрением сами видят фото.

    Paid-фото распознаёт DeepSeek нативно. Free-фото идут этим путём, только
    если у выбранной модели есть vision. Если модель не приняла картинки,
    для free-раздела срабатывает запасной путь через Groq new3.
    """
    info = MODEL_REGISTRY[model_key]
    provider = provider or info["provider"]
    section = info.get("section", section)
    first = messages[0]
    chat_id = first.chat.id
    private = is_private(first)
    user = first.from_user
    user_id = user.id if user else None
    rich = bool(RICH_SUPPORTED and private and user_rich(user_id))
    turn = Turn(
        message=first,
        chat_id=chat_id,
        is_private=private,
        user_text=question,
        model=model_key,
        section=section,
        track=track,
    )
    streamer = RichStreamer(turn) if rich else None
    status = "ok"
    error_text = ""
    try:
        await turn.status.show("👁 Смотрю фото…")
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
                logging.warning(f"Gateway photo download failed: {e}")

        if not images:
            await send_text(turn, "⚠️ Не удалось скачать ни одного фото.")
            status = "error"
            error_text = "all photos failed"
            return

        length = user_length(user_id) if private else "normal"
        user_question = question or "Что на фото? Реши подробно."
        prompt_text = f"Пользователь прислал фото. Вопрос/подпись: {user_question}"
        if failed:
            prompt_text += (
                "\n\nВнимание: фото " + ", ".join(str(i) for i in failed)
                + " не удалось прочитать. Оцени остальные."
            )

        vision_messages = _photo_content_messages(images, prompt_text)
        model_messages = [
            {"role": "system", "content": system_prompt(rich, length)}
        ] + [dict(m) for m in history_for_model(chat_id, model_key)] + vision_messages

        search_on = user_search_on(user_id) if private else True
        tools = TOOLS
        if not search_on:
            tools = [
                t for t in TOOLS
                if (t.get("function") or {}).get("name") != "web_search"
            ]

        if search_on and has_search_verb(user_question):
            forced = {"type": "function", "function": {"name": "web_search"}}
        else:
            forced = None

        effort = user_effort(user_id) if private else "off"
        max_tokens = LENGTH_TOKENS.get(length, LENGTH_TOKENS["normal"])

        async def executor(name, args):
            return await execute_tool(turn, name, args)

        async def ask_with(payload_messages: list[dict]) -> str:
            return await run_tools(
                payload_messages,
                tools,
                executor,
                forced,
                provider=provider,
                track=track,
                effort=effort,
                model=MODEL_REGISTRY[model_key]["model"],
                model_key=model_key,
                on_delta=streamer.on_delta if streamer else None,
                reset_cb=streamer.reset if streamer else None,
                max_tokens=max_tokens,
            )

        try:
            answer = await ask_with(model_messages)
        except Exception as e:
            if section != studio.FREE or not _is_image_rejected(e):
                raise
            # Free-модель не приняла картинки: описываем их через Groq new3
            # и повторяем запрос уже текстом.
            logging.warning(
                f"Free vision rejected photos ({type(e).__name__}: {e}); "
                f"перехожу на Groq new3"
            )
            if streamer is not None:
                streamer.reset()
            await turn.status.show("🧩 Описываю фото…")
            descriptions: list[tuple[int, str]] = []
            for index, (mime, data) in enumerate(images, 1):
                try:
                    desc = await describe_image(
                        data, mime, track=track, section=studio.FREE
                    )
                except Exception as exc:
                    logging.warning(f"Photo {index} describe failed: {exc}")
                    desc = ""
                if desc:
                    descriptions.append((index, desc.strip()))
            if not descriptions:
                raise
            text_parts = [prompt_text]
            for index, desc in descriptions:
                text_parts.append(f"Описание фото {index}:\n{desc}")
            text_messages = [
                {"role": "system", "content": system_prompt(rich, length)}
            ] + [
                dict(m) for m in history_for_model(chat_id, model_key)
            ] + [{"role": "user", "content": "\n\n".join(text_parts)}]
            answer = await ask_with(text_messages)

        photo_label = "фото" if len(images) == 1 else f"фото ({len(images)})"
        history_entry = (
            f"{display_user(first.from_user)}: прислал {photo_label}. "
            f"Вопрос/подпись: {user_question}"
        )
        await add_to_history(chat_id, "user", history_entry)

        if failed:
            await send_text(
                turn,
                "⚠️ Не удалось прочитать фото: <b>"
                + ", ".join(str(i) for i in failed)
                + "</b>. Остальные обработал.",
            )

        footer = usage_footer(turn, rich)
        if answer:
            answer = strip_text_tool_calls(answer)
        note = silent_reasoning_note(streamer, effort)
        if answer and note:
            footer = note + ("\n\n" + footer if footer else "")
        if answer:
            if streamer is not None:
                await streamer.finish(answer, footer=footer)
            else:
                await send_text(turn, answer + footer)
            await add_to_history(chat_id, "assistant", answer)
        elif not turn.tools_used:
            fallback_text = "Не смог разобрать задание. Попробуй сфотографировать чётче."
            if streamer is not None:
                await streamer.finish(fallback_text, footer=footer)
            else:
                await send_text(turn, fallback_text + footer)
            await add_to_history(chat_id, "assistant", fallback_text)

        _stats[chat_id] = _stats.get(chat_id, 0) + 1
        logging.info(f"[{chat_id}] ФОТО {model_key} ({len(images)}) | {question[:80]}")
    except asyncio.CancelledError:
        status = "cancelled"
        raise
    except Exception as e:
        status = "error"
        error_text = str(e)
        logging.exception("Gateway photo error")
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
        await pool_add(user_id, section, track.total_tokens)
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

    model = await resolve_model(user.id, private)
    if not model:
        await notify_pool_exhausted(
            message, user_section(user.id) if private else studio.FREE
        )
        return
    voice_section = (MODEL_REGISTRY.get(normalize_model_key(model)) or {}).get(
        "section", studio.FREE
    )
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
        transcript = await transcribe_voice(
            path,
            track=track,
            section=voice_section,
            user_id=user.id if user else None,
        )
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

    turn = Turn(
        message=message,
        chat_id=chat_id,
        is_private=private,
        user_text=cleaned,
        model=model,
        section=voice_section,
        track=track,
        rich_allowed=False,
    )

    logging.info(f"[{chat_id}] ГОЛОС | model={model} | {cleaned[:80]}")
    async with chat_lock(chat_id):
        await run_ai(turn)


# ============================================================
# ТЕКСТ
# ============================================================


async def handle_private_text(message: Message) -> None:
    if is_on_cooldown(message):
        return

    text = (message.text or "").strip()
    if not text:
        return

    chat_id = message.chat.id
    user = message.from_user
    if user is None or not is_allowed_user(user.id):
        return
    model = await resolve_model(user.id, True)
    if not model:
        await notify_pool_exhausted(message, user_section(user.id))
        return
    section = (MODEL_REGISTRY.get(normalize_model_key(model)) or {}).get(
        "section", studio.FREE
    )
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
        model=model,
        section=section,
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

    group_model = await resolve_model(user.id if user else None, False)
    if not group_model:
        await notify_pool_exhausted(message, studio.FREE)
        return
    track = await begin_request(
        message,
        kind="group_request",
        prompt=text,
        model=group_model,
    )

    logging.info(f"{display_user(user)} | {text[:100]}")

    turn = Turn(
        message=message,
        chat_id=chat_id,
        is_private=False,
        user_text=cleaned,
        model=group_model,
        section=studio.FREE,
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
    BotCommand(command="profile", description="Профиль и статистика"),
]
PRIVATE_COMMANDS = [
    BotCommand(command="start", description="Приветствие"),
    BotCommand(command="help", description="Что я умею"),
    BotCommand(command="about", description="Инфо обо мне и моделях"),
    BotCommand(command="models", description="Модели и режимы"),
    BotCommand(command="profile", description="Профиль и лимиты"),
    BotCommand(command="clear", description="Очистить контекст чата"),
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

    allowed_users = await storage.load_allowed_users()
    ALLOWED_USERS.clear()
    ALLOWED_USERS.update(int(uid) for uid in allowed_users)
    _allowed_users_cache.clear()
    _allowed_users_cache.update({int(uid): info for uid, info in allowed_users.items()})

    settings = await storage.load_user_settings()
    _user_settings.clear()
    _user_settings.update(settings)

    contexts = await storage.load_contexts()
    _history.clear()
    for chat_id, messages in contexts.items():
        if isinstance(messages, list):
            _trim_history(messages)
            _history[int(chat_id)] = messages

    # Живой каталог моделей: Pollinations / Groq / Gemini.
    catalog = await studio.refresh_catalog()
    rebuild_model_registry(catalog)

    logging.info(
        f"Загружено: групп={len(_groups_cache)}, ЛС в блоке={len(_dm_blacklist)}, "
        f"доступов={len(_grants)}, whitelist={len(ALLOWED_USERS)}, "
        f"настроек={len(_user_settings)}, контекстов={len(_history)}, "
        f"моделей={len(MODEL_REGISTRY)} (каталог: {studio.CATALOG_SOURCE})"
    )


async def health_server() -> None:
    try:
        from aiohttp import web
    except Exception as e:
        logging.warning(f"aiohttp недоступен, health server выключен: {e}")
        return

    async def health(_request):
        return web.json_response(
            {
                "status": "ok",
                "build": APP_BUILD,
                "uptime": format_uptime(),
                "models": len(GATEWAY_MODELS),
                "gateway": bool(gateway_client.configured),
                "storage": "postgres" if storage.available else "memory",
                "rich": RICH_SUPPORTED,
            }
        )

    try:
        app = web.Application()
        app.router.add_get("/health", health)
        app.router.add_get("/", health)
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
    logging.info(f"Build: {APP_BUILD}")
    logging.info(f"Chat: {CHAT_MODEL} (Groq), fast: {FAST_MODEL}")
    logging.info(
        f"Шлюз: {len(GATEWAY_MODELS)} "
        f"{plural_ru(len(GATEWAY_MODELS), 'модель', 'модели', 'моделей')}, "
        f"{'настроен' if gateway_client.configured else 'нет GATEWAY_API'}"
    )
    logging.info(f"Vision: {VISION_MODEL} (Gemini), fallback: {GROQ_VISION_MODEL} (Groq)")
    logging.info(f"Voice: {WHISPER_MODEL} (Groq)")
    logging.info("Search: Tavily + DDG fallback")

    await load_runtime_state()
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
    try:
        # Раньше тут висела кнопка «Админка» (Mini App). Возвращаем обычное
        # меню команд — иначе Telegram продолжает показывать кнопку.
        await bot.set_chat_menu_button(menu_button=MenuButtonCommands())
        logging.info("Menu button reset to commands")
    except Exception:
        logging.exception("set_chat_menu_button failed")

    tasks.append(asyncio.create_task(daily_summary_loop()))
    tasks.append(asyncio.create_task(balances_loop()))
    tasks.append(asyncio.create_task(dp.start_polling(bot)))
    try:
        await asyncio.gather(*tasks)
    finally:
        try:
            await http_client.aclose()
        except Exception:
            pass
        await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
