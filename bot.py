import asyncio
import logging
import os
import re
import time
import tempfile
import base64
import httpx

from aiohttp import web

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatType, ChatAction
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    Message,
    BotCommand,
    BotCommandScopeDefault,
    InlineQuery,
    InlineQueryResultArticle,
    InputTextMessageContent,
    BufferedInputFile,
)

from openai import AsyncOpenAI

# ============================================================
# НАСТРОЙКИ (из переменных окружения Render)
# ============================================================

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

CHAT_MODEL = "openai/gpt-oss-120b"
WHISPER_MODEL = "whisper-large-v3-turbo"
VISION_MODEL = "gemini-3.5-flash-lite"

TRIGGER_WORD = "мох"
HISTORY_LIMIT = 30
MAX_REPLY_LENGTH = 4000

BLOCKED_USERS = set()
COOLDOWN_SECONDS = 2
AI_REQUEST_TIMEOUT = 120.0
TYPING_INTERVAL = 3.0

START_TIME = time.time()

# ============================================================
# СИСТЕМНЫЙ ПРОМПТ
# ============================================================
SYSTEM_PROMPT = """Ты — Мудрый Мох, дружелюбный и полезный ассистент.
Ты помогаешь людям, отвечаешь по делу и с лёгким чувством юмора.
Если тебя спрашивают, кто ты — представляйся как «Мудрый Мох».
Отвечай на русском языке. Со всеми общаешься вежливо и нейтрально.

ВАЖНО ПРО ПРИВЕТСТВИЯ:
- НЕ здоровайся в каждом сообщении. Приветствие уместно ТОЛЬКО в самом начале диалога или когда пользователь сам поздоровался.
- Если пользователь задаёт вопрос — сразу отвечай по делу, без «Привет, я Мудрый Мох».
- Если в истории уже было приветствие от тебя — больше не повторяй его.
- Ты не начинаешь диалог заново при каждом сообщении — ты продолжаешь беседу.

КОНТЕКСТ С ФОТО:
- Если в истории есть строка вида "[ФОТО] <описание>" — значит пользователь ранее прислал фото, и вот его описание. Используй это описание для ответов.
- НИКОГДА не выводи сам текст "[ФОТО]" или "[ФОТО из реплая]" в ответе пользователю.
- Только если в истории ВООБЩЕ нет ничего про фото — вежливо попроси прислать.

КОНТЕКСТ С ГОЛОСОВЫМИ:
- Если в истории есть строка вида "[ГОЛОС] <расшифровка>" — используй её как обычный текст от пользователя.
- НИКОГДА не выводи "[ГОЛОС]" в ответе пользователю.

КОНТЕКСТ С РИСУНКАМИ:
- Если в истории есть строка вида "[РИСУНОК] <промпт>" — значит бот ранее сгенерировал картинку по этому промпту.
- НИКОГДА не выводи "[РИСУНОК]" в ответе пользователю.

КОНТЕКСТ С ПОИСКОМ:
- Если в истории есть строка вида "[РЕЗУЛЬТАТЫ ПОИСКА]" — это свежие данные из интернета. Используй их для ответа.
- Если в результатах есть противоречия — доверяй источникам, а не своим знаниям.

ФОРМАТИРОВАНИЕ — СТРОГО:
- Разрешены ТОЛЬКО эти HTML-теги Telegram: <b>, <i>, <u>, <s>, <code>, <pre>, <a href="...">, <blockquote>, <tg-spoiler>.
- КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНЫ: <ul>, <ol>, <li>, <p>, <br>, <div>, <span>, <h1>-<h6>, <table>, <tr>, <td>, <hr>.
- Для списков используй символы: • или — или цифры "1." "2." — БЕЗ тегов.
- Для переносов строк используй РЕАЛЬНЫЕ переносы (жми Enter).
- НЕ используй Markdown (** и _), только разрешённые HTML-теги.
- НЕ ЗЛОУПОТРЕБЛЯЙ форматированием! Максимум 1-2 <b> на ответ.
- Символы <, >, & в обычном тексте пиши словами или экранируй как &lt;, &gt;, &amp;.

Будь информативным."""


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
logging.getLogger("aiohttp.access").setLevel(logging.WARNING)

# ============================================================
# КЛИЕНТЫ
# ============================================================
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp = Dispatcher()

# Без прокси — Render за границей
http_client = httpx.AsyncClient(timeout=AI_REQUEST_TIMEOUT)

ai_client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL,
    http_client=http_client,
    max_retries=0,
    timeout=AI_REQUEST_TIMEOUT,
)

vision_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL,
    http_client=http_client,
    max_retries=0,
    timeout=AI_REQUEST_TIMEOUT,
)

BOT_ID: int | None = None
_history: dict[int, list[dict]] = {}
_stats: dict[int, int] = {}
_last_request: dict[int, float] = {}
_typing_last: dict[int, float] = {}


# ============================================================
# HTTP-СЕРВЕР ДЛЯ RENDER (health-check + cron-job.org пинги)
# ============================================================
async def health_handler(request):
    return web.Response(text="OK")

async def run_http_server():
    app = web.Application()
    app.router.add_get("/", health_handler)
    app.router.add_get("/ping", health_handler)
    app.router.add_get("/health", health_handler)

    port = int(os.environ.get("PORT", 8080))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logging.info(f"✅ HTTP-сервер запущен на порту {port}")


# ============================================================
# ОЧИСТКА ПАМЯТИ
# ============================================================
async def cleanup_old_records():
    while True:
        await asyncio.sleep(3600)  # раз в час
        now = time.time()
        for d in (_last_request, _typing_last):
            stale = [k for k, v in d.items() if now - v > 3600]
            for k in stale:
                d.pop(k, None)
        logging.info(f"🧹 Cleanup: удалено {len(stale)} устаревших записей")


# ============================================================
# РЕТРАЙ
# ============================================================
async def with_retry(coro_fn, *args, retries=3, base_wait=2.0, **kwargs):
    last_err = None
    for attempt in range(retries):
        try:
            return await coro_fn(*args, **kwargs)
        except asyncio.TimeoutError as e:
            last_err = e
            wait = base_wait * (attempt + 1)
            logging.warning(f"Timeout, retry {attempt+1}/{retries} in {wait}s")
            await asyncio.sleep(wait)
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if (
                "503" in str(e)
                or "unavailable" in err_str
                or "429" in str(e)
                or "rate" in err_str
                or "too many" in err_str
                or "connection" in err_str
                or "timeout" in err_str
                or "high demand" in err_str
            ):
                wait = base_wait * (attempt + 1)
                logging.warning(f"Retry {attempt+1}/{retries} in {wait}s: {e}")
                await asyncio.sleep(wait)
                continue
            raise
    raise last_err


# ============================================================
# САНИТАЙЗЕР HTML
# ============================================================
FORBIDDEN_TAGS = [
    "ul", "ol", "li", "p", "br", "div", "span", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "tr", "td", "th", "thead", "tbody",
    "img", "video", "audio", "iframe", "section", "article",
]


def sanitize_html(text: str) -> str:
    for tag in FORBIDDEN_TAGS:
        text = re.sub(rf"</?{tag}\b[^>]*>", "", text, flags=re.IGNORECASE)

    allowed_pattern = re.compile(
        r"</?(?:b|i|u|s|code|pre|blockquote|tg-spoiler|a)(?:\s+[^>]*)?>",
        re.IGNORECASE,
    )
    placeholders = {}

    def _hide(match):
        key = f"\x00TAG{len(placeholders)}\x00"
        placeholders[key] = match.group(0)
        return key

    text = allowed_pattern.sub(_hide, text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")

    for key, tag in placeholders.items():
        text = text.replace(key, tag)

    return text


# ============================================================
# ПРОВЕРКИ
# ============================================================
def is_blocked(message: Message) -> bool:
    user = message.from_user
    return user is not None and user.id in BLOCKED_USERS


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


# ============================================================
# "ПЕЧАТАЕТ…"
# ============================================================
async def safe_typing(chat_id: int):
    now = time.time()
    last = _typing_last.get(chat_id, 0)
    if now - last < TYPING_INTERVAL:
        return
    _typing_last[chat_id] = now
    try:
        await bot.send_chat_action(chat_id, ChatAction.TYPING)
    except Exception:
        pass


# ============================================================
# КОНТЕКСТ
# ============================================================
def add_to_history(chat_id: int, role: str, content: str) -> None:
    history = _history.setdefault(chat_id, [])
    history.append({"role": role, "content": content})
    if len(history) > HISTORY_LIMIT:
        history[:] = history[-HISTORY_LIMIT:]


def reset_history(chat_id: int) -> None:
    _history.pop(chat_id, None)


def get_history_size(chat_id: int) -> int:
    return len(_history.get(chat_id, []))


# ============================================================
# AI — Groq (текст)
# ============================================================
async def _ask_ai_raw(chat_id: int, user_content) -> str:
    history = _history.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_content})
    if len(history) > HISTORY_LIMIT:
        history[:] = history[-HISTORY_LIMIT:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]

    try:
        response = await asyncio.wait_for(
            ai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.8,
                max_tokens=2048,
            ),
            timeout=AI_REQUEST_TIMEOUT,
        )
        answer = response.choices[0].message.content.strip()
    except Exception:
        if history:
            history.pop()
        raise

    history.append({"role": "assistant", "content": answer})
    return answer


async def ask_ai(chat_id: int, user_content) -> str:
    try:
        return await with_retry(_ask_ai_raw, chat_id, user_content, retries=3)
    except Exception as e:
        logging.exception("Groq API error after retries")
        err = str(e).lower()
        if "429" in str(e) or "rate" in err or "too many" in err:
            return "⚠️ Слишком много запросов. Подожди минуту."
        return f"⚠️ Ошибка при обращении к нейросети: <code>{e}</code>"


async def transcribe_voice(file_path: str) -> str:
    async def _do():
        with open(file_path, "rb") as f:
            result = await asyncio.wait_for(
                ai_client.audio.transcriptions.create(
                    file=f,
                    model=WHISPER_MODEL,
                    language="ru",
                ),
                timeout=AI_REQUEST_TIMEOUT,
            )
        return result.text.strip()

    return await with_retry(_do, retries=2)


# ============================================================
# AI — Gemini (vision)
# ============================================================
async def ask_gemini_vision(prompt: str, image_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode()

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ]

    response = await asyncio.wait_for(
        vision_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=2048,
        ),
        timeout=AI_REQUEST_TIMEOUT,
    )
    return (response.choices[0].message.content or "").strip()


async def describe_photo_for_memory(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode()

    content = [
        {
            "type": "text",
            "text": (
                "Опиши это изображение в 80-120 слов для памяти ассистента. "
                "Укажи: объекты, людей, цвета, расположение, весь видимый текст, "
                "важные детали. Пиши только описание, без вступлений типа «на фото»."
            ),
        },
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ]

    response = await asyncio.wait_for(
        vision_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=400,
        ),
        timeout=AI_REQUEST_TIMEOUT,
    )
    return (response.choices[0].message.content or "").strip()


# ============================================================
# ПОИСК — DuckDuckGo
# ============================================================
SEARCH_TRIGGERS = (
    "поищи", "найди", "загугли", "погугли", "search",
    "что нового", "последние новости", "новости о",
    "курс ", "котировк", "цена ",
    "погода", "прогноз погоды",
    "сегодня", "вчера", "завтра", "в этом году",
    "что там с", "что случилось", "что происходит",
)


def has_search_trigger(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(t in lowered for t in SEARCH_TRIGGERS)


async def needs_search(text: str) -> bool:
    if has_search_trigger(text):
        return True

    short = text[:300]
    try:
        response = await asyncio.wait_for(
            ai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты — классификатор. Отвечай ТОЛЬКО одним словом: YES или NO. "
                            "YES — если для ответа нужны свежие данные из интернета: "
                            "новости, погода, курсы, цены, события после 2024, "
                            "актуальные факты про людей/компании/технологии. "
                            "NO — если это общий вопрос (математика, философия, "
                            "как что-то сделать, определение термина, объяснение, "
                            "сгенерируй текст, придумай, переведи)."
                        ),
                    },
                    {"role": "user", "content": short},
                ],
                temperature=0,
                max_tokens=3,
            ),
            timeout=15.0,
        )
        decision = (response.choices[0].message.content or "").strip().upper()
        return decision.startswith("YES")
    except Exception as e:
        logging.warning(f"needs_search failed: {e}")
        return has_search_trigger(text)


async def web_search(query: str, max_results: int = 5) -> list[dict]:
    def _sync_search():
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    try:
        results = await asyncio.to_thread(_sync_search)
        return results or []
    except Exception as e:
        logging.warning(f"DDG search failed: {e}")
        return []


def format_search_results(results: list[dict]) -> str:
    if not results:
        return "[РЕЗУЛЬТАТЫ ПОИСКА] Не удалось получить результаты из интернета."
    lines = ["[РЕЗУЛЬТАТЫ ПОИСКА]"]
    for i, r in enumerate(results[:5], 1):
        title = r.get("title", "").strip()
        body = r.get("body", "").strip()
        href = r.get("href", "").strip()
        lines.append(f"{i}. {title}\n   {body}\n   {href}")
    return "\n".join(lines)


# ============================================================
# ГЕНЕРАЦИЯ КАРТИНОК
# ============================================================
DRAW_TRIGGERS = (
    "нарисуй", "сгенерируй", "сделай картинку", "покажи картинку",
    "нарисуй картинку", "сгенерируй картинку", "draw", "generate image",
)


def extract_draw_prompt(text: str) -> str | None:
    if not text:
        return None
    lowered = text.lower()
    for trigger in DRAW_TRIGGERS:
        if trigger in lowered:
            idx = lowered.find(trigger) + len(trigger)
            prompt = text[idx:].strip(" ,.!?:;-—\n\t")
            return prompt if prompt else None
    return None


def parse_image_params(text: str) -> dict:
    params = {"width": 1024, "height": 1024, "model": "flux", "fmt": "png"}
    lowered = text.lower()

    m = re.search(r"(\d{2,4})\s*[xх]\s*(\d{2,4})", lowered)
    if m:
        params["width"] = min(int(m.group(1)), 2048)
        params["height"] = min(int(m.group(2)), 2048)
    elif "16:9" in lowered or "горизонталь" in lowered or "широк" in lowered:
        params["width"], params["height"] = 1344, 768
    elif "9:16" in lowered or "вертикаль" in lowered:
        params["width"], params["height"] = 768, 1344
    elif "квадрат" in lowered:
        params["width"], params["height"] = 1024, 1024

    if " turbo" in lowered:
        params["model"] = "turbo"
    elif "kontext" in lowered:
        params["model"] = "kontext"

    if " jpg" in lowered or " jpeg" in lowered:
        params["fmt"] = "jpg"

    return params


async def translate_to_english(text: str) -> str:
    try:
        response = await asyncio.wait_for(
            ai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты — переводчик для генератора изображений. "
                            "Переведи текст на английский. Сохрани суть образа: "
                            "объекты, стиль, атмосферу, детали. Убери служебные фразы. "
                            "Верни ТОЛЬКО перевод, без объяснений и кавычек. "
                            "Если текст уже на английском — верни как есть."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
                max_tokens=200,
            ),
            timeout=20.0,
        )
        result = (response.choices[0].message.content or "").strip().strip('"').strip("'")
        return result or text
    except Exception as e:
        logging.warning(f"translate_to_english failed: {e}")
        return text


async def generate_image(prompt_en: str, width: int = 1024, height: int = 1024,
                         model: str = "flux", fmt: str = "png") -> bytes:
    import urllib.parse
    import random

    encoded = urllib.parse.quote(prompt_en)
    seed = random.randint(1, 1_000_000)
    url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width={width}&height={height}&nologo=true&seed={seed}"
        f"&model={model}"
    )

    async def _do():
        response = await http_client.get(url, timeout=120.0)
        response.raise_for_status()
        return response.content

    return await with_retry(_do, retries=2, base_wait=3.0)


# ============================================================
# УТИЛИТЫ
# ============================================================
def has_trigger(text: str | None) -> bool:
    if not text:
        return False
    return TRIGGER_WORD.lower() in text.lower()


def extract_prompt(text: str | None) -> str:
    if not text:
        return ""
    lowered = text.lower()
    trigger = TRIGGER_WORD.lower()
    result = []
    i = 0
    while i < len(text):
        if lowered.startswith(trigger, i):
            i += len(trigger)
            continue
        result.append(text[i])
        i += 1
    return "".join(result).strip(" ,.!?:;-—\n\t")


def is_reply_to_bot(message: Message) -> bool:
    if not message.reply_to_message:
        return False
    from_user = message.reply_to_message.from_user
    return from_user is not None and from_user.id == BOT_ID


def should_reply(message: Message, text: str | None) -> bool:
    if message.chat.type == ChatType.PRIVATE:
        return True
    if is_reply_to_bot(message):
        return True
    if has_trigger(text):
        return True
    return False


def format_log_prefix(message: Message) -> str:
    user = message.from_user
    if not user:
        return "[unknown]"
    nickname = user.username or user.first_name or "unknown"
    chat_type = "личка" if message.chat.type == ChatType.PRIVATE else f"группа '{message.chat.title}'"
    return f"[{nickname} | {user.id} | {chat_type}]"


def make_user_label(message: Message) -> str:
    user = message.from_user
    if not user:
        return "Аноним"
    nickname = user.first_name or user.username or f"id{user.id}"
    return nickname


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


async def send_long_message(message: Message, text: str):
    text = sanitize_html(text)
    if not text:
        return
    if len(text) <= MAX_REPLY_LENGTH:
        try:
            await message.reply(text)
        except Exception as e:
            logging.error(f"send_long_message error: {e}")
        return
    chunks = [text[i:i + MAX_REPLY_LENGTH] for i in range(0, len(text), MAX_REPLY_LENGTH)]
    for idx, chunk in enumerate(chunks):
        try:
            if idx == 0:
                await message.reply(chunk)
            else:
                await message.answer(chunk)
        except Exception as e:
            logging.error(f"send_long_message chunk error: {e}")


async def handle_ai_text(message: Message, prompt: str):
    try:
        await safe_typing(message.chat.id)
        label = make_user_label(message)
        labeled_prompt = f"{label}: {prompt}" if message.chat.type != ChatType.PRIVATE else prompt

        search_used = False
        if await needs_search(prompt):
            logging.info(f"{format_log_prefix(message)} ПОИСК DDG | {prompt[:80]}")
            results = await web_search(prompt, max_results=5)
            if results:
                search_block = format_search_results(results)
                labeled_prompt = f"{search_block}\n\nВопрос: {labeled_prompt}"
                search_used = True

        answer = await ask_ai(message.chat.id, labeled_prompt)
        _stats[message.chat.id] = _stats.get(message.chat.id, 0) + 1
        logging.info(
            f"{format_log_prefix(message)} ОТВЕТ{' (с поиском)' if search_used else ''} | {prompt[:80]}"
        )
        await send_long_message(message, answer)
    except Exception as e:
        logging.exception("Text handling error")


# ============================================================
# КОМАНДЫ
# ============================================================
@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "🌿 Привет! Я — <b>Мудрый Мох</b>.\n\n"
        "• В <b>личке</b> — отвечаю на всё: текст, фото, голосовые.\n"
        "• В <b>группах</b> — только если обратиться словом <b>мох</b> "
        "или ответить реплаем на моё сообщение.\n"
        "• <b>Рисование</b>: <code>мох нарисуй кота</code>\n"
        "• <b>Поиск в интернете</b> — включается автоматически.\n"
        "• <b>Inline-режим</b>: <code>@M0XAIBOT запрос</code>\n\n"
        "🧹 Сбросить контекст: <code>/clear</code>\n"
        "ℹ️ Обо мне: <code>/about</code>"
    )


@dp.message(Command("about"))
async def cmd_about(message: Message):
    uptime = format_uptime()
    await message.answer(
        "🌿 <b>Мудрый Мох — информация</b>\n\n"

        "<b>📝 Текстовая модель:</b>\n"
        f"• <code>{CHAT_MODEL}</code>\n"
        "• Провайдер: <b>Groq</b>\n"
        "• ~120B параметров, MoE\n"
        "• Контекст: 128K токенов\n\n"

        "<b>👁 Vision (фото):</b>\n"
        f"• <code>{VISION_MODEL}</code>\n"
        "• Провайдер: <b>Google Gemini</b>\n\n"

        "<b>🎙 Голос:</b>\n"
        f"• <code>{WHISPER_MODEL}</code>\n"
        "• Провайдер: <b>Groq</b>\n\n"

        "<b>🎨 Рисование:</b>\n"
        "• <b>Pollinations.ai</b>\n"
        "• Бесплатно, без ключа\n\n"

        "<b>🔍 Поиск:</b>\n"
        "• <b>DuckDuckGo</b> — без ключа\n"
        "• Автоматически, когда нужен\n\n"

        "<b>⚙️ Возможности:</b>\n"
        "• Текст, фото, голосовые\n"
        "• Генерация картинок\n"
        "• Поиск в интернете\n"
        "• Фото и голос из реплая\n"
        "• Inline-режим\n\n"

        f"<b>📊 Статистика:</b>\n"
        f"• Время работы: <b>{uptime}</b>\n"
        f"• Запросов в этом чате: <b>{_stats.get(message.chat.id, 0)}</b>\n"
        f"• В контексте: <b>{get_history_size(message.chat.id)}</b> из {HISTORY_LIMIT}\n\n"

        "💡 <i>Все сервисы на бесплатных тарифах</i>"
    )


@dp.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "🌿 <b>Что я умею:</b>\n\n"
        "• Отвечаю на вопросы\n"
        "• Понимаю фото и голосовые\n"
        "• Рисую картинки: <code>мох нарисуй кота</code>\n"
        "• Ищу в интернете, когда нужно\n"
        "• Помню контекст разговора\n"
        "• Работаю в inline-режиме\n\n"
        "<b>Команды:</b>\n"
        "<code>/start</code> — приветствие\n"
        "<code>/help</code> — это меню\n"
        "<code>/about</code> — инфо обо мне\n"
        "<code>/clear</code> — очистить контекст\n"
        "<code>/context</code> — размер памяти\n"
        "<code>/whoami</code> — твой ID\n"
        "<code>/stats</code> — статистика"
    )


@dp.message(Command("clear"))
async def cmd_clear(message: Message):
    reset_history(message.chat.id)
    await message.reply("🧹 Контекст этого чата очищен. Начинаем с чистого листа!")


@dp.message(Command("context"))
async def cmd_context(message: Message):
    size = get_history_size(message.chat.id)
    await message.reply(f"📊 В контексте: <b>{size}</b> сообщений из {HISTORY_LIMIT}.")


@dp.message(Command("whoami"))
async def cmd_whoami(message: Message):
    user = message.from_user
    if not user:
        return
    await message.reply(
        f"👤 <b>{user.full_name}</b>\n"
        f"ID: <code>{user.id}</code>\n"
        f"Username: @{user.username or '—'}"
    )


@dp.message(Command("stats"))
async def cmd_stats(message: Message):
    count = _stats.get(message.chat.id, 0)
    size = get_history_size(message.chat.id)
    await message.reply(
        f"📊 <b>Статистика чата:</b>\n"
        f"• Обработано запросов: <b>{count}</b>\n"
        f"• Сейчас в контексте: <b>{size}</b> сообщений"
    )


# ============================================================
# INLINE
# ============================================================
@dp.inline_query()
async def handle_inline(query: InlineQuery):
    user = query.from_user
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

    if user and user.id in BLOCKED_USERS:
        await query.answer(results=[], cache_time=0, is_personal=True)
        return

    try:
        inline_chat_id = user.id if user else 0
        answer = await ask_ai(inline_chat_id, prompt)
        answer = sanitize_html(answer)[:3800]

        await query.answer(
            results=[
                InlineQueryResultArticle(
                    id=f"ai_{int(time.time())}",
                    title=f"Мох: {prompt[:60]}",
                    description=answer[:120].replace("\n", " "),
                    input_message_content=InputTextMessageContent(
                        message_text=answer,
                        parse_mode="HTML",
                    ),
                )
            ],
            cache_time=0,
            is_personal=True,
        )
    except Exception as e:
        logging.exception("Inline error")
        await query.answer(
            results=[
                InlineQueryResultArticle(
                    id="error",
                    title="Ошибка",
                    description=str(e)[:120],
                    input_message_content=InputTextMessageContent(
                        message_text=f"⚠️ Ошибка: {e}"
                    ),
                )
            ],
            cache_time=0,
            is_personal=True,
        )


# ============================================================
# ТЕКСТ
# ============================================================
@dp.message(F.text)
async def handle_text(message: Message):
    if is_blocked(message):
        return
    if is_on_cooldown(message):
        return

    user = message.from_user
    if not user:
        return

    if message.chat.type != ChatType.PRIVATE:
        logging.info(f"{format_log_prefix(message)} {message.text[:100]}")

    if not should_reply(message, message.text):
        return

    is_private = message.chat.type == ChatType.PRIVATE
    bare_mention = False

    if is_private:
        prompt = message.text.strip()
    else:
        prompt = extract_prompt(message.text)
        if not prompt:
            if is_reply_to_bot(message) and not has_trigger(message.text):
                prompt = message.text.strip() or "продолжи"
            if not prompt:
                bare_mention = True

    reply = message.reply_to_message

    # === ГЕНЕРАЦИЯ КАРТИНОК ===
    draw_prompt = extract_draw_prompt(prompt)
    if draw_prompt:
        try:
            await safe_typing(message.chat.id)
            logging.info(f"{format_log_prefix(message)} РИСУЮ | {draw_prompt[:80]}")

            prompt_en = await translate_to_english(draw_prompt)
            logging.info(f"{format_log_prefix(message)} ПРОМПТ EN | {prompt_en[:80]}")

            params = parse_image_params(draw_prompt)

            image_bytes = await generate_image(
                prompt_en,
                width=params["width"],
                height=params["height"],
                model=params["model"],
            )

            photo_file = BufferedInputFile(image_bytes, filename=f"image.{params['fmt']}")
            await message.reply_photo(
                photo=photo_file,
                caption=f"🎨 Нарисовал: <i>{draw_prompt[:200]}</i>",
            )
            _stats[message.chat.id] = _stats.get(message.chat.id, 0) + 1

            label = make_user_label(message)
            add_to_history(
                message.chat.id, "user",
                f"{label}: <b>[РИСУНОК]</b> {draw_prompt}"
            )
            add_to_history(
                message.chat.id, "assistant",
                f"Нарисовал картинку по запросу: {draw_prompt[:200]}"
            )
            return
        except Exception as e:
            logging.exception("Image generation error")
            await message.reply(f"⚠️ Ошибка генерации: <code>{e}</code>")
            return

    # === ФОТО из реплая ===
    if reply and reply.photo and not is_reply_to_bot(message):
        try:
            photo = reply.photo[-1]
            file = await bot.get_file(photo.file_id)
            file_bytes = await bot.download_file(file.file_path)
            image_data = file_bytes.read()

            await safe_typing(message.chat.id)
            base_prompt = prompt or "Что на этом изображении? Опиши и ответь, если есть вопрос."

            reply_context = ""
            if reply.caption:
                reply_context = f"\n\nПодпись к фото из реплая: {reply.caption}"
            if reply.from_user:
                reply_author = reply.from_user.first_name or reply.from_user.username or "кто-то"
                reply_context += f"\n\nАвтор фото в реплае: {reply_author}"

            full_prompt = f"{base_prompt}{reply_context}"

            answer_task = with_retry(ask_gemini_vision, full_prompt, image_data, retries=3)
            desc_task = with_retry(describe_photo_for_memory, image_data, retries=3)

            answer, description = await asyncio.gather(
                answer_task, desc_task, return_exceptions=True
            )

            if isinstance(answer, Exception):
                logging.exception("Vision answer error (reply)", exc_info=answer)
                answer = f"⚠️ Ошибка с фото: <code>{answer}</code>"

            if isinstance(description, Exception) or not description:
                description = "(описание не удалось)"

            label = make_user_label(message)
            add_to_history(
                message.chat.id,
                "user",
                f"{label}: <b>[ФОТО из реплая]</b> {description}"
            )

            _stats[message.chat.id] = _stats.get(message.chat.id, 0) + 1
            await send_long_message(message, answer)
            return
        except Exception as e:
            logging.exception("Reply photo handling error")
            await message.reply(f"⚠️ Ошибка с фото из реплая: <code>{e}</code>")
            return

    # === ГОЛОС из реплая ===
    if reply and (reply.voice or reply.audio) and not is_reply_to_bot(message):
        try:
            media = reply.voice or reply.audio
            file = await bot.get_file(media.file_id)

            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                await bot.download_file(file.file_path, tmp.name)
                tmp_path = tmp.name

            try:
                await safe_typing(message.chat.id)
                transcript = await transcribe_voice(tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            if transcript:
                reply_author = ""
                if reply.from_user:
                    reply_author = reply.from_user.first_name or reply.from_user.username or "кто-то"

                label = make_user_label(message)
                add_to_history(
                    message.chat.id,
                    "user",
                    f"{label}: <b>[ГОЛОС из реплая от {reply_author}]</b> {transcript}"
                )

                if prompt:
                    full_prompt = (
                        f"[Пользователь ответил на голосовое от {reply_author}. "
                        f"Расшифровка: \"{transcript}\"]\n\n{prompt}"
                    )
                else:
                    full_prompt = (
                        f"[Пользователь ответил на голосовое от {reply_author}. "
                        f"Расшифровка: \"{transcript}\"]\n\n"
                        f"Отреагируй коротко и по делу на суть голосового."
                    )
            else:
                if prompt:
                    full_prompt = f"[Голосовое в реплае, не удалось расшифровать]\n\n{prompt}"
                else:
                    full_prompt = "Скажи что не расслышал голосовое, попроси повторить."

            await handle_ai_text(message, full_prompt)
            return
        except Exception as e:
            logging.exception("Reply voice handling error")
            await message.reply(f"⚠️ Ошибка с голосовым из реплая: <code>{e}</code>")
            return

    # текст из реплая — контекст
    if reply and reply.text and not is_reply_to_bot(message):
        reply_author = ""
        if reply.from_user:
            reply_author = reply.from_user.first_name or reply.from_user.username or "кто-то"
        if prompt:
            prompt = f"[В реплае на сообщение от {reply_author}: \"{reply.text[:500]}\"]\n\n{prompt}"
        else:
            prompt = f"[В реплае на сообщение от {reply_author}: \"{reply.text[:500]}\"]"

    # === ГОЛОЕ "МОХ" ===
    if bare_mention and not prompt:
        prompt = (
            "[СИСТЕМА: пользователь просто окликнул тебя без вопроса. "
            "Ответь ОДНОЙ короткой нейтральной фразой — спроси чего хочет. "
            "БЕЗ эмоций, БЕЗ восклицаний, БЕЗ шуток, БЕЗ обращения по имени. "
            "Примеры: «Чего?», «Слушаю», «Ну?», «Чем помочь?». "
            "Максимум 3 слова.]"
        )

    await handle_ai_text(message, prompt)


# ============================================================
# ФОТО
# ============================================================
@dp.message(F.photo)
async def handle_photo(message: Message):
    if is_blocked(message):
        return
    if is_on_cooldown(message):
        return

    caption = message.caption or ""
    if not should_reply(message, caption):
        return

    is_private = message.chat.type == ChatType.PRIVATE
    if is_private:
        prompt = caption.strip() or "Что на этом изображении? Опиши и ответь, если есть вопрос."
    else:
        prompt = extract_prompt(caption) or "Что на этом изображении?"

    try:
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        file_bytes = await bot.download_file(file.file_path)
        image_data = file_bytes.read()

        await safe_typing(message.chat.id)
        logging.info(f"{format_log_prefix(message)} ФОТО | {prompt[:80]}")

        answer_task = with_retry(ask_gemini_vision, prompt, image_data, retries=3)
        desc_task = with_retry(describe_photo_for_memory, image_data, retries=3)

        answer, description = await asyncio.gather(
            answer_task, desc_task, return_exceptions=True
        )

        if isinstance(answer, Exception):
            logging.exception("Vision answer error", exc_info=answer)
            answer = f"⚠️ Ошибка с фото: <code>{answer}</code>"

        if isinstance(description, Exception) or not description:
            description = "(описание не удалось)"

        label = make_user_label(message)
        add_to_history(
            message.chat.id,
            "user",
            f"{label}: <b>[ФОТО]</b> {description}"
        )

        _stats[message.chat.id] = _stats.get(message.chat.id, 0) + 1
        await send_long_message(message, answer)
    except asyncio.TimeoutError:
        logging.error("Vision timeout")
        await message.reply("⚠️ Нейросеть затупила на фото. Попробуй ещё раз.")
    except Exception as e:
        logging.exception("Photo handling error")
        await message.reply(f"⚠️ Ошибка с фото: <code>{e}</code>")


# ============================================================
# ГОЛОСОВЫЕ
# ============================================================
@dp.message(F.voice | F.audio)
async def handle_voice(message: Message):
    if is_blocked(message):
        return
    if is_on_cooldown(message):
        return

    is_private = message.chat.type == ChatType.PRIVATE
    is_reply = is_reply_to_bot(message)

    if not is_private and not is_reply:
        return

    media = message.voice or message.audio
    file = await bot.get_file(media.file_id)

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        await bot.download_file(file.file_path, tmp.name)
        tmp_path = tmp.name

    try:
        await safe_typing(message.chat.id)
        transcript = await transcribe_voice(tmp_path)
        if not transcript:
            await message.reply("🤔 Не удалось распознать голосовое.")
            return

        if not is_private and not has_trigger(transcript) and not is_reply:
            return

        prompt = transcript
        if not is_private and has_trigger(transcript):
            prompt = extract_prompt(transcript) or transcript

        label = make_user_label(message)

        if message.chat.type != ChatType.PRIVATE:
            add_to_history(
                message.chat.id,
                "user",
                f"{label}: <b>[ГОЛОС]</b> {prompt}"
            )
            full_prompt = f"[Голосовое от {label}, расшифровка: \"{prompt}\"]\n\n{prompt}"
        else:
            full_prompt = f"[Голосовое, расшифровка: \"{prompt}\"]\n\n{prompt}"

        answer = await ask_ai(message.chat.id, full_prompt)
        _stats[message.chat.id] = _stats.get(message.chat.id, 0) + 1
        await send_long_message(message, answer)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ============================================================
# ЗАПУСК
# ============================================================
async def set_commands():
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


async def main():
    global BOT_ID
    me = await bot.get_me()
    BOT_ID = me.id
    logging.info(f"🌿 Бот запущен: @{me.username} (id={BOT_ID})")
    logging.info(f"📝 Chat: {CHAT_MODEL} (Groq)")
    logging.info(f"👁 Vision: {VISION_MODEL} (Gemini)")
    logging.info(f"🔍 Search: DuckDuckGo")
    logging.info(f"🎨 Images: Pollinations")
    logging.info(f"🌐 Режим: Render (без прокси)")

    # Запуск HTTP-сервера и очистки параллельно
    asyncio.create_task(run_http_server())
    asyncio.create_task(cleanup_old_records())

    await set_commands()
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
