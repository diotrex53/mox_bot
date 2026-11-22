import asyncio
import base64
import html
import json
import logging
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date
from urllib.parse import quote

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction, ChatType
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BotCommand,
    BotCommandScopeDefault,
    BufferedInputFile,
    ChatMemberUpdated,
    InlineQuery,
    InlineQueryResultArticle,
    InputTextMessageContent,
    Message,
)
from openai import AsyncOpenAI

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

# Если запускаешь локально и нужен SOCKS-прокси — впиши сюда,
# например "socks5://127.0.0.1:10808". На Render оставь пустым.
PROXY_URL = os.environ.get("PROXY_URL", "")

OWNER_ID = 6101597477
ALLOWED_GROUPS = {-1002495961136}

HISTORY_LIMIT = 50
HISTORY_CHARS_LIMIT = 60000
MAX_REPLY_LENGTH = 4000
COOLDOWN_SECONDS = 2
AI_TIMEOUT = 120.0
TYPING_INTERVAL = 4.0
FALLBACK_COOLDOWN = 600
ALBUM_DELAY = 1.3
MAX_ALBUM_PHOTOS = 8

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

BOT_ID: int | None = None
BOT_USERNAME = ""

_history: dict[int, list[dict]] = {}
_stats: dict[int, int] = {}
_last_request: dict[int, float] = {}
_typing_last: dict[int, float] = {}
_fallback_last: dict[int, float] = {}
_media_cache: dict[tuple[int, int], str] = {}
_locks: dict[int, asyncio.Lock] = {}
_album_buffers: dict[tuple[int, str], list[Message]] = {}
_album_tasks: dict[tuple[int, str], asyncio.Task] = {}
_search_cache: dict[str, tuple[float, list[dict]]] = {}
_image_lock = asyncio.Lock()

GROUP_TYPES = (ChatType.GROUP, ChatType.SUPERGROUP)

# ============================================================
# РЕТРАЙ
# ============================================================


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
                logging.warning(f"Retry {attempt + 1}/{retries} in {wait}s: {e}")
                await asyncio.sleep(wait)
                continue
            raise
    raise last_err


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


# ============================================================
# КОНТЕКСТ
# ============================================================


def add_to_history(chat_id: int, role: str, content: str) -> None:
    if not content:
        return
    history = _history.setdefault(chat_id, [])
    history.append({"role": role, "content": content})
    while len(history) > HISTORY_LIMIT:
        history.pop(0)
    # Страховка от переполнения контекста очень длинными сообщениями.
    total = sum(len(m.get("content") or "") for m in history)
    while len(history) > 1 and total > HISTORY_CHARS_LIMIT:
        total -= len(history[0].get("content") or "")
        history.pop(0)


def get_history(chat_id: int) -> list[dict]:
    return _history.get(chat_id, [])


def reset_history(chat_id: int) -> None:
    _history.pop(chat_id, None)


def get_history_size(chat_id: int) -> int:
    return len(_history.get(chat_id, []))


def chat_lock(chat_id: int) -> asyncio.Lock:
    return _locks.setdefault(chat_id, asyncio.Lock())


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

SYSTEM_PROMPT = """Ты — Мудрый Мох, дружелюбный ассистент в Telegram. Отвечай по-русски, по делу, с лёгким юмором. Если пользователь просит написать текст на другом языке (письмо, код, задание) — напиши на нём.

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
- Не выводи служебные пометки, названия инструментов и технические маркеры.

ФОРМАТИРОВАНИЕ — СТРОГО:
- Разрешены только теги Telegram: <b>, <i>, <u>, <s>, <code>, <pre>, <a href="...">, <blockquote>, <tg-spoiler>.
- Запрещены: <ul>, <ol>, <li>, <p>, <br>, <div>, <span>, <h1>-<h6>, <table>, <img>.
- Списки — символами •, — или «1.», «2.» без тегов. Переносы строк — реальные.
- Никакого Markdown (** и _). Максимум 1–2 <b> на ответ.
- Символы <, >, & в обычном тексте экранируй как &lt;, &gt;, &amp;."""


def system_prompt() -> str:
    return SYSTEM_PROMPT + f"\n\nСегодня: {date.today().isoformat()}."


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


async def groq_chat(messages, tools=None, tool_choice="auto"):
    async def _do():
        kwargs = {
            "model": CHAT_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        return await asyncio.wait_for(
            ai_client.chat.completions.create(**kwargs),
            timeout=AI_TIMEOUT,
        )

    return await with_retry(_do, retries=2)


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


async def describe_image(image_bytes: bytes, mime: str | None = None) -> str:
    if not mime:
        if image_bytes.startswith(b"\x89PNG"):
            mime = "image/png"
        elif image_bytes.startswith(b"GIF8"):
            mime = "image/gif"
        elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            mime = "image/webp"
        else:
            mime = "image/jpeg"

    b64 = base64.b64encode(image_bytes).decode()

    async def gemini_describe() -> str:
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
        return text

    async def groq_describe() -> str:
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
        return text

    try:
        return await with_retry(gemini_describe, retries=2)
    except Exception as e:
        logging.warning(
            f"Gemini vision failed ({type(e).__name__}: {e}); "
            f"пробую {GROQ_VISION_MODEL}"
        )
        return await with_retry(groq_describe, retries=2)


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


async def transcribe_voice(file_path: str) -> str:
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

    return await with_retry(_do, retries=2)


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
    for i, r in enumerate(results[:5], 1):
        lines.append(
            f"{i}. {r['title']}\n{r['url']}\n{r['snippet'][:350]}"
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


async def translate_to_english(text: str) -> str:
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
        return (response.choices[0].message.content or "").strip().strip('"').strip("'")

    try:
        result = await with_retry(_do, retries=2)
        return result or text
    except Exception as e:
        logging.warning(f"translate_to_english failed: {e}")
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


async def get_photo_description(message: Message) -> str:
    key = (message.chat.id, message.message_id)
    cached = _media_cache.get(key)
    if cached:
        return cached
    try:
        data = await download_image_bytes(message)
        desc = await describe_image(data)
    except Exception as e:
        logging.warning(f"describe photo failed: {e}")
        desc = "[фото] (не удалось описать)"

    _media_cache[key] = desc
    if len(_media_cache) > 500:
        for old_key in list(_media_cache.keys())[:150]:
            _media_cache.pop(old_key, None)
    return desc


async def get_voice_text(message: Message) -> str:
    key = (message.chat.id, message.message_id)
    cached = _media_cache.get(key)
    if cached:
        return cached

    path = None
    try:
        path = await download_media_to_temp(message)
        text = await transcribe_voice(path)
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


async def build_reply_block(message: Message, need_media: bool = True) -> tuple[str, str]:
    """Возвращает (цитата для контекста, описание медиа из реплая)."""
    reply = message.reply_to_message
    if not reply:
        return "", ""

    author = display_user(reply.from_user) if reply.from_user else "канал"
    body = ""
    media_desc = ""

    if reply.photo:
        if need_media:
            media_desc = await get_photo_description(reply)
            body = f"фото: {media_desc}"
        else:
            body = "[фото]"
    elif reply.document and (reply.document.mime_type or "").startswith("image/"):
        if need_media:
            media_desc = await get_photo_description(reply)
            body = f"фото: {media_desc}"
        else:
            body = "[фото]"
    elif reply.voice or reply.audio:
        if need_media:
            media_desc = await get_voice_text(reply)
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


async def run_tools(messages, tools, executor, forced=None) -> str:
    choice = forced or "auto"
    for _ in range(4):
        try:
            response = await groq_chat(messages, tools, choice)
        except Exception:
            if choice != "auto":
                choice = "auto"
                continue
            raise

        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            return (msg.content or "").strip()

        messages.append(assistant_tool_message(msg))
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            try:
                output = await executor(tc.function.name, args)
            except Exception as e:
                output = f"Ошибка инструмента: {e}"
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": str(output)[:6000]}
            )
        choice = "auto"
    return ""


async def fallback_draw(turn: Turn) -> None:
    prompt_ru = (turn.explicit_draw or turn.user_text or "").strip()
    if not prompt_ru:
        return
    await turn.status.show("🎨 Рисую…")
    prompt_en = await translate_to_english(prompt_ru)
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


async def run_ai(turn: Turn) -> None:
    try:
        await safe_typing(turn.chat_id)
        history = _history.setdefault(turn.chat_id, [])
        messages = [{"role": "system", "content": system_prompt()}]
        messages += [dict(m) for m in history]

        if turn.explicit_draw:
            forced = {"type": "function", "function": {"name": "generate_image"}}
        elif has_search_verb(turn.user_text):
            forced = {"type": "function", "function": {"name": "web_search"}}
        else:
            forced = None

        async def executor(name, args):
            return await execute_tool(turn, name, args)

        answer = await run_tools(messages, TOOLS, executor, forced)

        if turn.explicit_draw and not turn.image_sent and not turn.image_error:
            await fallback_draw(turn)

        if turn.image_sent:
            history.append(
                {"role": "assistant", "content": f"Отправил сгенерированную картинку: {turn.image_prompt_ru}"}
            )
        elif turn.image_error:
            history.append(
                {"role": "assistant", "content": f"Не удалось сгенерировать изображение: {turn.image_error}"}
            )

        if answer:
            if not (turn.image_sent and is_trivial(answer)):
                await send_text(turn, answer)
                history.append({"role": "assistant", "content": answer})
        elif not turn.image_sent and not turn.image_error:
            fallback_text = "Не понял. Повтори, пожалуйста."
            await send_text(turn, fallback_text)
            history.append({"role": "assistant", "content": fallback_text})

        _stats[turn.chat_id] = _stats.get(turn.chat_id, 0) + 1
        logging.info(
            f"[{turn.chat_id}] AI готов | draw={turn.image_sent} | поиск={has_search_verb(turn.user_text)}"
        )
    except Exception as e:
        logging.exception("AI handling error")
        await send_text(turn, f"⚠️ Ошибка: <code>{esc(e)}</code>")
    finally:
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
        "🧹 Сбросить контекст: <code>/clear</code>\n"
        "ℹ️ Обо мне: <code>/about</code>",
    )


@dp.message(Command("help"))
async def cmd_help(message: Message):
    if not command_allowed(message):
        return
    await reply_any(
        message,
        "🌿 <b>Что я умею:</b>\n\n"
        "• Отвечаю на вопросы и помню контекст\n"
        "• Понимаю фото и голосовые\n"
        "• Рисую: <code>нарисуй кота</code>\n"
        "• Ищу в интернете, когда нужно\n"
        "• Отвечаю в inline-режиме\n\n"
        "<b>Команды:</b>\n"
        "<code>/start</code> — приветствие\n"
        "<code>/help</code> — это меню\n"
        "<code>/about</code> — инфо обо мне\n"
        "<code>/clear</code> — очистить контекст\n"
        "<code>/context</code> — размер памяти\n"
        "<code>/whoami</code> — твой ID\n"
        "<code>/stats</code> — статистика",
    )


@dp.message(Command("about"))
async def cmd_about(message: Message):
    if not command_allowed(message):
        return
    chat_id = message.chat.id
    await reply_any(
        message,
        "🌿 <b>Мудрый Мох — информация</b>\n\n"
        "<b>📝 Текст:</b>\n"
        f"• <code>{CHAT_MODEL}</code> (Groq)\n"
        f"• Быстрая модель: <code>{FAST_MODEL}</code>\n\n"
        "<b>👁 Vision (фото):</b>\n"
        f"• <code>{VISION_MODEL}</code> (Google Gemini)\n"
        f"• Резерв: <code>{GROQ_VISION_MODEL}</code> (Groq)\n\n"
        "<b>🎙 Голос:</b>\n"
        f"• <code>{WHISPER_MODEL}</code> (Groq)\n\n"
        "<b>🎨 Рисование:</b>\n"
        "• Pollinations.ai\n\n"
        "<b>🔍 Поиск:</b>\n"
        "• Tavily (+ запасной DuckDuckGo)\n\n"
        f"<b>📊 Статистика:</b>\n"
        f"• Время работы: <b>{format_uptime()}</b>\n"
        f"• Запросов в этом чате: <b>{_stats.get(chat_id, 0)}</b>\n"
        f"• В контексте: <b>{get_history_size(chat_id)}</b> из {HISTORY_LIMIT}",
    )


@dp.message(Command("clear"))
async def cmd_clear(message: Message):
    if not command_allowed(message):
        return
    chat_id = message.chat.id
    reset_history(chat_id)
    for key in list(_media_cache.keys()):
        if key[0] == chat_id:
            _media_cache.pop(key, None)
    await reply_any(message, "🧹 Контекст этого чата очищен. Начинаем с чистого листа!")
    logging.info(f"[{chat_id}] /clear")


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
                "Чтобы включить бота, добавь этот ID в <code>ALLOWED_GROUPS</code>.",
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
        add_to_history(message.chat.id, "user", f"[в чат вошёл {display_user(user)}]")


@dp.message(F.left_chat_member)
async def on_user_leave(message: Message):
    if is_private(message) or not is_whitelisted(message.chat.id):
        return
    user = message.left_chat_member
    if user is None:
        return
    if BOT_ID is not None and user.id == BOT_ID:
        return
    add_to_history(message.chat.id, "user", f"[из чата вышел {display_user(user)}]")


# ============================================================
# INLINE
# ============================================================


async def simple_answer(prompt: str) -> str:
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

    return await run_tools(messages, SEARCH_TOOLS, executor, forced)


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

    try:
        answer = await simple_answer(prompt)
    except Exception as e:
        logging.exception("Inline error")
        answer = f"⚠️ Ошибка: {e}"

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


async def album_worker(key):
    try:
        await asyncio.sleep(ALBUM_DELAY)
        messages = _album_buffers.pop(key, [])
    finally:
        _album_tasks.pop(key, None)
    if messages:
        try:
            await process_photos(messages)
        except Exception:
            logging.exception("Album processing error")


@dp.message(F.photo)
async def on_photo(message: Message):
    user = message.from_user
    if not user or user.is_bot:
        return

    if message.media_group_id:
        key = (message.chat.id, message.media_group_id)
        _album_buffers.setdefault(key, []).append(message)
        if key not in _album_tasks:
            _album_tasks[key] = asyncio.create_task(album_worker(key))
        return

    await process_photos([message])


async def process_photos(messages: list[Message]) -> None:
    first = messages[0]
    chat_id = first.chat.id
    private = is_private(first)

    if not private and not is_whitelisted(chat_id):
        addressed = any(is_addressed(m, m.caption) for m in messages)
        if addressed and can_send_fallback(chat_id):
            await first.reply(FALLBACK_TEXT)
        return

    addressed = private or any(is_addressed(m, m.caption) for m in messages)
    if not addressed:
        # В группах фото без обращения не попадает в контекст (только текст,
        # стикеры и прочие медиа логируются заглушками).
        return

    if is_on_cooldown(first):
        return

    turn = Turn(message=first, chat_id=chat_id, is_private=private)
    await turn.status.show("👁 Смотрю фото…")

    descriptions: list[str] = []
    last_error = ""
    for m in messages[:MAX_ALBUM_PHOTOS]:
        try:
            data = await download_image_bytes(m)
            desc = await describe_image(data)
            if desc:
                descriptions.append(desc)
                _media_cache[(chat_id, m.message_id)] = desc
        except Exception as e:
            last_error = str(e)
            logging.warning(f"Photo describe failed: {e}")

    if not descriptions:
        await turn.status.clear()
        await send_text(turn, f"⚠️ Не удалось посмотреть фото: <code>{esc(last_error or 'неизвестная ошибка')}</code>")
        return

    caption = next((m.caption.strip() for m in messages if m.caption), "")
    question = strip_triggers(caption) if caption else ""
    question = question or "Что на фото?"

    if len(descriptions) == 1:
        photo_block = descriptions[0]
    else:
        photo_block = "\n".join(f"{i}) {d}" for i, d in enumerate(descriptions, 1))

    quote = ""
    for m in messages:
        if m.reply_to_message:
            # Если отвечают на чужое фото/голосовое — описываем его через Gemini/Whisper.
            quote, _ = await build_reply_block(m, need_media=True)
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
    add_to_history(chat_id, "user", entry)

    turn.user_text = question
    draw = detect_explicit_draw(question)
    if draw and re.search(r"(такое же|похожее|это же|это)", draw, flags=re.IGNORECASE):
        draw = f"{draw}. Референс: {'; '.join(descriptions)}"
    turn.explicit_draw = draw

    logging.info(f"[{chat_id}] ФОТО ({len(descriptions)}) | {question[:80]}")
    async with chat_lock(chat_id):
        await run_ai(turn)


# ============================================================
# ДОКУМЕНТЫ-КАРТИНКИ
# ============================================================


@dp.message(F.document)
async def on_document(message: Message):
    user = message.from_user
    if not user or user.is_bot:
        return
    mime = (message.document.mime_type or "") if message.document else ""
    if mime.startswith("image/"):
        await process_photos([message])
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

    chat_id = message.chat.id
    private = is_private(message)

    if not private and not is_whitelisted(chat_id):
        if is_reply_to_bot(message) and can_send_fallback(chat_id):
            await message.reply(FALLBACK_TEXT)
        return

    await safe_typing(chat_id)

    path = None
    try:
        path = await download_media_to_temp(message)
        transcript = await transcribe_voice(path)
    except Exception as e:
        logging.warning(f"Voice transcription failed: {e}")
        if private or is_reply_to_bot(message):
            await reply_any(message, f"⚠️ Не удалось расшифровать голосовое: <code>{esc(e)}</code>")
        return
    finally:
        if path:
            try:
                os.unlink(path)
            except Exception:
                pass

    if not transcript:
        if private or is_reply_to_bot(message):
            await reply_any(message, "🤔 Не удалось распознать голосовое.")
        return

    addressed = private or is_reply_to_bot(message) or starts_with_trigger(transcript)
    if not addressed:
        # В группах голосовые без обращения в контекст не попадают.
        return

    if is_on_cooldown(message):
        return

    cleaned = strip_triggers(transcript) if starts_with_trigger(transcript) else transcript
    _media_cache[(chat_id, message.message_id)] = cleaned

    quote, media_desc = await build_reply_block(message, need_media=True)
    label = "" if private else f"{display_user(user)}: "
    entry_text = cleaned or "(обратился без вопроса)"
    entry = f"{label}{quote}(голосовое, расшифровка, возможны ошибки): {entry_text}"
    add_to_history(chat_id, "user", entry)

    draw = detect_explicit_draw(cleaned)
    if draw and media_desc and re.search(r"(такое же|похожее|это же|это)", draw, flags=re.IGNORECASE):
        draw = f"{draw}. Референс: {media_desc}"
    turn = Turn(
        message=message,
        chat_id=chat_id,
        is_private=private,
        user_text=cleaned,
        explicit_draw=draw,
    )

    logging.info(f"[{chat_id}] ГОЛОС | {cleaned[:80]}")
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
    quote, media_desc = await build_reply_block(message, need_media=True)
    add_to_history(chat_id, "user", f"{quote}{text}")

    turn = Turn(
        message=message,
        chat_id=chat_id,
        is_private=True,
        user_text=text,
        explicit_draw=draw_with_reference(detect_explicit_draw(text), media_desc),
    )
    logging.info(f"[{chat_id}] ЛС | {text[:100]}")
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

    addressed = is_addressed(message, text)

    if not addressed:
        quote, _ = await build_reply_block(message, need_media=False)
        add_to_history(chat_id, "user", f"{display_user(user)}: {quote}{text}")
        return

    quote, media_desc = await build_reply_block(message, need_media=True)
    cleaned = strip_triggers(text)
    entry_text = cleaned or "(обратился без вопроса)"
    add_to_history(chat_id, "user", f"{display_user(user)}: {quote}{entry_text}")

    logging.info(f"{display_user(user)} | {text[:100]}")

    if is_on_cooldown(message):
        return

    turn = Turn(
        message=message,
        chat_id=chat_id,
        is_private=False,
        user_text=cleaned or text,
        explicit_draw=draw_with_reference(detect_explicit_draw(cleaned), media_desc),
    )
    async with chat_lock(chat_id):
        await run_ai(turn)


@dp.message(F.text)
async def on_text(message: Message):
    user = message.from_user
    if not user or user.is_bot:
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
        add_to_history(message.chat.id, "user", f"{placeholder}{caption}")
    elif is_whitelisted(message.chat.id):
        add_to_history(
            message.chat.id,
            "user",
            f"{display_user(message.from_user)}: {placeholder}{caption}",
        )


@dp.message()
async def on_other(message: Message):
    user = message.from_user
    if not user or user.is_bot:
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


async def set_commands() -> None:
    commands = [
        BotCommand(command="start", description="Приветствие"),
        BotCommand(command="help", description="Что я умею"),
        BotCommand(command="about", description="Инфо обо мне и моделях"),
        BotCommand(command="clear", description="Очистить контекст чата"),
        BotCommand(command="context", description="Сколько сообщений в памяти"),
        BotCommand(command="whoami", description="Мой ID и username"),
        BotCommand(command="stats", description="Статистика чата"),
    ]
    await bot.set_my_commands(commands, scope=BotCommandScopeDefault())


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
    logging.info(f"Vision: {VISION_MODEL} (Gemini), fallback: {GROQ_VISION_MODEL} (Groq)")
    logging.info(f"Voice: {WHISPER_MODEL} (Groq)")
    logging.info("Search: Tavily + DDG fallback")
    logging.info(f"Allowed groups: {sorted(ALLOWED_GROUPS)}")

    try:
        await set_commands()
    except Exception:
        logging.exception("set_commands failed")
    try:
        await bot.delete_webhook(drop_pending_updates=True)
    except Exception:
        logging.exception("delete_webhook failed")

    tasks.append(asyncio.create_task(dp.start_polling(bot)))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
