"""Хранилище бота: Neon Postgres (asyncpg) с fallback в память.

Все методы намеренно не бросают исключения наружу: бот должен продолжать
работать, даже если база временно недоступна. Ошибки только логируются.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

try:
    import asyncpg
except Exception:  # pragma: no cover - локальный запуск без asyncpg
    asyncpg = None

MSK = timezone(timedelta(hours=3))
DEFAULT_GROUP_ID = -1002495961136
DB_LIMIT_BYTES = 1024 ** 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS bot_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS allowed_groups (
    chat_id BIGINT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    added_by BIGINT,
    added_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dm_blacklist (
    user_id BIGINT PRIMARY KEY,
    username TEXT,
    full_name TEXT,
    added_by BIGINT,
    added_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Старые таблицы DeepSeek оставлены для миграции данных.
CREATE TABLE IF NOT EXISTS deepseek_grants (
    user_id BIGINT PRIMARY KEY,
    username TEXT,
    full_name TEXT,
    granted_by BIGINT,
    granted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS deepseek_usage (
    user_id BIGINT NOT NULL,
    day DATE NOT NULL,
    requests INT NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, day)
);

CREATE TABLE IF NOT EXISTS provider_grants (
    user_id BIGINT NOT NULL,
    provider TEXT NOT NULL,
    username TEXT,
    full_name TEXT,
    granted_by BIGINT,
    granted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, provider)
);

CREATE TABLE IF NOT EXISTS usage_daily (
    user_id BIGINT NOT NULL,
    provider TEXT NOT NULL,
    day DATE NOT NULL,
    requests INT NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, provider, day)
);

CREATE TABLE IF NOT EXISTS user_settings (
    user_id BIGINT PRIMARY KEY,
    model TEXT NOT NULL DEFAULT 'groq',
    rich BOOLEAN NOT NULL DEFAULT TRUE,
    deepthink TEXT NOT NULL DEFAULT 'off',
    search TEXT NOT NULL DEFAULT 'auto',
    length TEXT NOT NULL DEFAULT 'normal',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS contexts (
    chat_id BIGINT PRIMARY KEY,
    messages JSONB NOT NULL DEFAULT '[]'::jsonb,
    message_count INT NOT NULL DEFAULT 0,
    char_count INT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS requests (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id BIGINT,
    username TEXT,
    chat_id BIGINT,
    chat_type TEXT,
    kind TEXT NOT NULL,
    is_request BOOLEAN NOT NULL DEFAULT TRUE,
    model TEXT,
    prompt_preview TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    error TEXT,
    latency_ms INT,
    tools TEXT,
    request_ref TEXT
);

CREATE INDEX IF NOT EXISTS idx_requests_created ON requests (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_requests_user ON requests (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests (model, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_requests_kind ON requests (kind, created_at DESC);

CREATE TABLE IF NOT EXISTS model_calls (
    id BIGSERIAL PRIMARY KEY,
    request_id BIGINT REFERENCES requests(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id BIGINT,
    chat_id BIGINT,
    provider TEXT,
    model TEXT,
    prompt_tokens INT NOT NULL DEFAULT 0,
    completion_tokens INT NOT NULL DEFAULT 0,
    reasoning_tokens INT NOT NULL DEFAULT 0,
    cached_tokens INT NOT NULL DEFAULT 0,
    total_tokens INT NOT NULL DEFAULT 0,
    credit DOUBLE PRECISION NOT NULL DEFAULT 0,
    latency_ms INT,
    status TEXT NOT NULL DEFAULT 'ok',
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_calls_created ON model_calls (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_calls_model ON model_calls (model, created_at DESC);

CREATE TABLE IF NOT EXISTS tool_calls (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    request_id BIGINT REFERENCES requests(id) ON DELETE SET NULL,
    user_id BIGINT,
    chat_id BIGINT,
    tool TEXT NOT NULL,
    args_preview TEXT,
    status TEXT NOT NULL DEFAULT 'ok',
    latency_ms INT,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_tools_created ON tool_calls (created_at DESC);

CREATE TABLE IF NOT EXISTS admin_actions (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    admin_id BIGINT,
    action TEXT NOT NULL,
    details TEXT
);
"""

# Миграция старых данных DeepSeek в общие таблицы. После успешного копирования
# старые строки удаляются, чтобы отозванные гранты не возвращались при рестарте.
MIGRATIONS = """
ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS length TEXT NOT NULL DEFAULT 'normal';

INSERT INTO provider_grants (user_id, provider, username, full_name, granted_by, granted_at)
SELECT user_id, 'deepseek', username, full_name, granted_by, granted_at
FROM deepseek_grants
WHERE TRUE
ON CONFLICT (user_id, provider) DO NOTHING;

INSERT INTO usage_daily (user_id, provider, day, requests)
SELECT user_id, 'deepseek', day, requests
FROM deepseek_usage
WHERE TRUE
ON CONFLICT (user_id, provider, day) DO NOTHING;

DELETE FROM deepseek_grants;
DELETE FROM deepseek_usage;
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _char_count(messages: list[dict]) -> int:
    return sum(len(str(m.get("content") or "")) for m in messages)


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except Exception:
        return default


class Storage:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = (database_url or "").strip()
        self.pool: Any = None
        self.memory = False

        # Fallback-хранилище
        self._groups: dict[int, dict] = {
            DEFAULT_GROUP_ID: {
                "chat_id": DEFAULT_GROUP_ID,
                "title": "Мудрый Мох",
                "added_at": _now(),
            }
        }
        self._blacklist: dict[int, dict] = {}
        self._grants: dict[tuple[int, str], dict] = {}
        self._usage: dict[tuple[int, str, str], int] = {}
        self._user_settings: dict[int, dict] = {}
        self._contexts: dict[int, list[dict]] = {}
        self._requests: list[dict] = []
        self._calls: list[dict] = []
        self._tools: list[dict] = []
        self._actions: list[dict] = []
        self._settings: dict[str, str] = {}
        self._request_seq = 0

    # ------------------------------------------------------------------
    # Подключение
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        if not self.database_url:
            logging.warning(
                "DATABASE_URL не задан — статистика и контекст будут только в памяти. "
                "Для Neon добавь DATABASE_URL в Render."
            )
            self.memory = True
            return
        if asyncpg is None:
            logging.error("asyncpg не установлен — работаю без Postgres")
            self.memory = True
            return

        try:
            dsn = self.database_url
            # Neon иногда даёт строку с ?sslmode=require&channel_binding=require.
            # asyncpg не понимает channel_binding, а SSL мы задаём отдельно.
            if "?" in dsn:
                dsn = dsn.split("?", 1)[0]
            self.pool = await asyncpg.create_pool(
                dsn,
                min_size=1,
                max_size=5,
                command_timeout=30,
                ssl="require",
                statement_cache_size=0,
            )
            async with self.pool.acquire() as conn:
                await conn.execute(SCHEMA)
                try:
                    await conn.execute(MIGRATIONS)
                except Exception as e:
                    logging.warning(f"Миграция старых данных не удалась: {e}")
                count = await conn.fetchval("SELECT count(*) FROM allowed_groups")
                if not count:
                    await conn.execute(
                        "INSERT INTO allowed_groups (chat_id, title) VALUES ($1, $2) "
                        "ON CONFLICT (chat_id) DO NOTHING",
                        DEFAULT_GROUP_ID,
                        "Мудрый Мох",
                    )
            self.memory = False
            logging.info("Postgres подключён, схема готова")
        except Exception as e:
            logging.exception(f"Postgres connection failed: {e}")
            self.pool = None
            self.memory = True

    async def close(self) -> None:
        if self.pool is not None:
            try:
                await self.pool.close()
            except Exception:
                pass

    @property
    def available(self) -> bool:
        return not self.memory and self.pool is not None

    # ------------------------------------------------------------------
    # Настройки
    # ------------------------------------------------------------------

    async def get_setting(self, key: str, default: str | None = None) -> str | None:
        try:
            if self.available:
                value = await self.pool.fetchval(
                    "SELECT value FROM bot_settings WHERE key = $1", key
                )
                return value if value is not None else default
            return self._settings.get(key, default)
        except Exception as e:
            logging.warning(f"get_setting failed: {e}")
            return default

    async def set_setting(self, key: str, value: str) -> None:
        try:
            if self.available:
                await self.pool.execute(
                    "INSERT INTO bot_settings (key, value) VALUES ($1, $2) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                    key,
                    value,
                )
            else:
                self._settings[key] = value
        except Exception as e:
            logging.warning(f"set_setting failed: {e}")

    # ------------------------------------------------------------------
    # Группы
    # ------------------------------------------------------------------

    async def load_groups(self) -> dict[int, dict]:
        try:
            if self.available:
                rows = await self.pool.fetch(
                    "SELECT chat_id, title, added_at FROM allowed_groups ORDER BY added_at"
                )
                self._groups = {
                    int(r["chat_id"]): {
                        "chat_id": int(r["chat_id"]),
                        "title": r["title"] or "",
                        "added_at": r["added_at"],
                    }
                    for r in rows
                }
            return dict(self._groups)
        except Exception as e:
            logging.warning(f"load_groups failed: {e}")
            return dict(self._groups)

    async def add_group(self, chat_id: int, title: str, added_by: int | None = None) -> None:
        self._groups[int(chat_id)] = {
            "chat_id": int(chat_id),
            "title": title or "",
            "added_at": _now(),
        }
        try:
            if self.available:
                await self.pool.execute(
                    "INSERT INTO allowed_groups (chat_id, title, added_by) VALUES ($1, $2, $3) "
                    "ON CONFLICT (chat_id) DO UPDATE SET title = EXCLUDED.title",
                    int(chat_id),
                    title or "",
                    added_by,
                )
        except Exception as e:
            logging.warning(f"add_group failed: {e}")

    async def remove_group(self, chat_id: int) -> None:
        self._groups.pop(int(chat_id), None)
        try:
            if self.available:
                await self.pool.execute(
                    "DELETE FROM allowed_groups WHERE chat_id = $1", int(chat_id)
                )
        except Exception as e:
            logging.warning(f"remove_group failed: {e}")

    # ------------------------------------------------------------------
    # Чёрный список ЛС
    # ------------------------------------------------------------------

    async def load_blacklist(self) -> dict[int, dict]:
        try:
            if self.available:
                rows = await self.pool.fetch(
                    "SELECT user_id, username, full_name, added_at FROM dm_blacklist ORDER BY added_at"
                )
                self._blacklist = {
                    int(r["user_id"]): {
                        "user_id": int(r["user_id"]),
                        "username": r["username"] or "",
                        "full_name": r["full_name"] or "",
                        "added_at": r["added_at"],
                    }
                    for r in rows
                }
            return dict(self._blacklist)
        except Exception as e:
            logging.warning(f"load_blacklist failed: {e}")
            return dict(self._blacklist)

    async def add_blacklist(
        self,
        user_id: int,
        username: str = "",
        full_name: str = "",
        added_by: int | None = None,
    ) -> None:
        self._blacklist[int(user_id)] = {
            "user_id": int(user_id),
            "username": username or "",
            "full_name": full_name or "",
            "added_at": _now(),
        }
        try:
            if self.available:
                await self.pool.execute(
                    "INSERT INTO dm_blacklist (user_id, username, full_name, added_by) "
                    "VALUES ($1, $2, $3, $4) "
                    "ON CONFLICT (user_id) DO UPDATE SET username = EXCLUDED.username, "
                    "full_name = EXCLUDED.full_name",
                    int(user_id),
                    username or "",
                    full_name or "",
                    added_by,
                )
        except Exception as e:
            logging.warning(f"add_blacklist failed: {e}")

    async def remove_blacklist(self, user_id: int) -> None:
        self._blacklist.pop(int(user_id), None)
        try:
            if self.available:
                await self.pool.execute(
                    "DELETE FROM dm_blacklist WHERE user_id = $1", int(user_id)
                )
        except Exception as e:
            logging.warning(f"remove_blacklist failed: {e}")

    # ------------------------------------------------------------------
    # Доступ к провайдерам (DeepSeek / Qwen)
    # ------------------------------------------------------------------

    async def load_grants(self) -> dict[tuple[int, str], dict]:
        try:
            if self.available:
                rows = await self.pool.fetch(
                    "SELECT user_id, provider, username, full_name, granted_at "
                    "FROM provider_grants ORDER BY granted_at"
                )
                self._grants = {
                    (int(r["user_id"]), str(r["provider"])): {
                        "user_id": int(r["user_id"]),
                        "provider": str(r["provider"]),
                        "username": r["username"] or "",
                        "full_name": r["full_name"] or "",
                        "granted_at": r["granted_at"],
                    }
                    for r in rows
                }
            return dict(self._grants)
        except Exception as e:
            logging.warning(f"load_grants failed: {e}")
            return dict(self._grants)

    async def add_grant(
        self,
        user_id: int,
        provider: str = "deepseek",
        username: str = "",
        full_name: str = "",
        granted_by: int | None = None,
    ) -> None:
        self._grants[(int(user_id), provider)] = {
            "user_id": int(user_id),
            "provider": provider,
            "username": username or "",
            "full_name": full_name or "",
            "granted_at": _now(),
        }
        try:
            if self.available:
                await self.pool.execute(
                    "INSERT INTO provider_grants "
                    "(user_id, provider, username, full_name, granted_by) "
                    "VALUES ($1, $2, $3, $4, $5) "
                    "ON CONFLICT (user_id, provider) DO UPDATE SET "
                    "username = EXCLUDED.username, full_name = EXCLUDED.full_name",
                    int(user_id),
                    provider,
                    username or "",
                    full_name or "",
                    granted_by,
                )
        except Exception as e:
            logging.warning(f"add_grant failed: {e}")

    async def remove_grant(self, user_id: int, provider: str = "deepseek") -> None:
        self._grants.pop((int(user_id), provider), None)
        try:
            if self.available:
                await self.pool.execute(
                    "DELETE FROM provider_grants WHERE user_id = $1 AND provider = $2",
                    int(user_id),
                    provider,
                )
        except Exception as e:
            logging.warning(f"remove_grant failed: {e}")

    async def get_usage(self, user_id: int, provider: str, day: date) -> int:
        key = (int(user_id), provider, day.isoformat())
        if key in self._usage:
            return self._usage[key]
        try:
            if self.available:
                value = await self.pool.fetchval(
                    "SELECT requests FROM usage_daily "
                    "WHERE user_id = $1 AND provider = $2 AND day = $3",
                    int(user_id),
                    provider,
                    day,
                )
                self._usage[key] = int(value or 0)
                return self._usage[key]
        except Exception as e:
            logging.warning(f"get_usage failed: {e}")
        return 0

    async def inc_usage(self, user_id: int, provider: str, day: date) -> int:
        key = (int(user_id), provider, day.isoformat())
        current = self._usage.get(key)
        if current is None:
            current = await self.get_usage(user_id, provider, day)
        current += 1
        self._usage[key] = current
        try:
            if self.available:
                await self.pool.execute(
                    "INSERT INTO usage_daily (user_id, provider, day, requests) "
                    "VALUES ($1, $2, $3, 1) "
                    "ON CONFLICT (user_id, provider, day) DO UPDATE SET "
                    "requests = usage_daily.requests + 1",
                    int(user_id),
                    provider,
                    day,
                )
        except Exception as e:
            logging.warning(f"inc_usage failed: {e}")
        return current

    async def usage_for_day(
        self, day: date, provider: str | None = None
    ) -> dict[tuple[int, str], int]:
        try:
            if self.available:
                if provider:
                    rows = await self.pool.fetch(
                        "SELECT user_id, provider, requests FROM usage_daily "
                        "WHERE day = $1 AND provider = $2",
                        day,
                        provider,
                    )
                else:
                    rows = await self.pool.fetch(
                        "SELECT user_id, provider, requests FROM usage_daily "
                        "WHERE day = $1",
                        day,
                    )
                return {
                    (int(r["user_id"]), str(r["provider"])): int(r["requests"])
                    for r in rows
                }
        except Exception as e:
            logging.warning(f"usage_for_day failed: {e}")
        prefix = day.isoformat()
        return {
            (user_id, prov): count
            for (user_id, prov, d), count in self._usage.items()
            if d == prefix and (provider is None or prov == provider)
        }

    # ------------------------------------------------------------------
    # Настройки пользователей
    # ------------------------------------------------------------------

    async def load_user_settings(self) -> dict[int, dict]:
        try:
            if self.available:
                rows = await self.pool.fetch(
                    "SELECT user_id, model, rich, deepthink, search, length "
                    "FROM user_settings"
                )
                self._user_settings = {
                    int(r["user_id"]): {
                        "model": r["model"] or "groq",
                        "rich": bool(r["rich"]),
                        "deepthink": r["deepthink"] or "off",
                        "search": r["search"] or "on",
                        "length": r["length"] or "normal",
                    }
                    for r in rows
                }
            return dict(self._user_settings)
        except Exception as e:
            logging.warning(f"load_user_settings failed: {e}")
            return dict(self._user_settings)

    async def save_user_settings(self, user_id: int, **fields) -> None:
        allowed = {"model", "rich", "deepthink", "search", "length"}
        values = {k: v for k, v in fields.items() if k in allowed}
        if not values:
            return
        current = self._user_settings.setdefault(
            int(user_id),
            {
                "model": "groq",
                "rich": True,
                "deepthink": "off",
                "search": "on",
                "length": "normal",
            },
        )
        current.update(values)
        try:
            if not self.available:
                return
            columns = list(values.keys())
            placeholders = ", ".join(f"${i + 2}" for i in range(len(columns)))
            updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns)
            sql = (
                f"INSERT INTO user_settings (user_id, {', '.join(columns)}) "
                f"VALUES ($1, {placeholders}) "
                f"ON CONFLICT (user_id) DO UPDATE SET {updates}, updated_at = now()"
            )
            await self.pool.execute(sql, int(user_id), *[values[c] for c in columns])
        except Exception as e:
            logging.warning(f"save_user_settings failed: {e}")

    # ------------------------------------------------------------------
    # Контекст
    # ------------------------------------------------------------------

    async def load_contexts(self) -> dict[int, list[dict]]:
        try:
            if self.available:
                rows = await self.pool.fetch("SELECT chat_id, messages FROM contexts")
                loaded: dict[int, list[dict]] = {}
                for r in rows:
                    raw = r["messages"]
                    if isinstance(raw, str):
                        try:
                            raw = json.loads(raw)
                        except Exception:
                            raw = []
                    if isinstance(raw, list):
                        loaded[int(r["chat_id"])] = raw
                self._contexts = loaded
            return dict(self._contexts)
        except Exception as e:
            logging.warning(f"load_contexts failed: {e}")
            return dict(self._contexts)

    async def save_context(self, chat_id: int, messages: list[dict]) -> None:
        messages = list(messages)
        self._contexts[int(chat_id)] = messages
        try:
            if self.available:
                payload = json.dumps(messages, ensure_ascii=False)
                await self.pool.execute(
                    "INSERT INTO contexts (chat_id, messages, message_count, char_count, updated_at) "
                    "VALUES ($1, $2::jsonb, $3, $4, now()) "
                    "ON CONFLICT (chat_id) DO UPDATE SET messages = EXCLUDED.messages, "
                    "message_count = EXCLUDED.message_count, char_count = EXCLUDED.char_count, "
                    "updated_at = now()",
                    int(chat_id),
                    payload,
                    len(messages),
                    _char_count(messages),
                )
        except Exception as e:
            logging.warning(f"save_context failed: {e}")

    async def delete_context(self, chat_id: int) -> None:
        self._contexts.pop(int(chat_id), None)
        try:
            if self.available:
                await self.pool.execute("DELETE FROM contexts WHERE chat_id = $1", int(chat_id))
        except Exception as e:
            logging.warning(f"delete_context failed: {e}")

    # ------------------------------------------------------------------
    # Запросы и вызовы моделей
    # ------------------------------------------------------------------

    async def create_request(
        self,
        *,
        user_id: int | None,
        username: str,
        chat_id: int | None,
        chat_type: str,
        kind: str,
        is_request: bool,
        model: str | None,
        prompt_preview: str,
        status: str = "running",
    ) -> int | None:
        now = _now()
        if not self.available:
            self._request_seq += 1
            rid = self._request_seq
            self._requests.append(
                {
                    "id": rid,
                    "created_at": now,
                    "user_id": user_id,
                    "username": username,
                    "chat_id": chat_id,
                    "chat_type": chat_type,
                    "kind": kind,
                    "is_request": is_request,
                    "model": model,
                    "prompt_preview": (prompt_preview or "")[:300],
                    "status": status,
                    "error": "",
                    "latency_ms": None,
                    "tools": "",
                }
            )
            return rid
        try:
            rid = await self.pool.fetchval(
                "INSERT INTO requests (user_id, username, chat_id, chat_type, kind, "
                "is_request, model, prompt_preview, status) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING id",
                user_id,
                username,
                chat_id,
                chat_type,
                kind,
                is_request,
                model,
                (prompt_preview or "")[:300],
                status,
            )
            return int(rid) if rid is not None else None
        except Exception as e:
            logging.warning(f"create_request failed: {e}")
            return None

    async def finish_request(
        self,
        request_id: int | None,
        *,
        status: str = "ok",
        error: str = "",
        latency_ms: int | None = None,
        model: str | None = None,
        tools: str = "",
    ) -> None:
        if request_id is None:
            return
        try:
            if not self.available:
                for row in self._requests:
                    if row["id"] == request_id:
                        row["status"] = status
                        row["error"] = (error or "")[:1000]
                        row["latency_ms"] = latency_ms
                        if model:
                            row["model"] = model
                        row["tools"] = tools or ""
                        break
                return
            await self.pool.execute(
                "UPDATE requests SET status = $2, error = $3, latency_ms = $4, "
                "model = COALESCE($5, model), tools = $6 WHERE id = $1",
                int(request_id),
                status,
                (error or "")[:1000],
                latency_ms,
                model,
                tools or "",
            )
        except Exception as e:
            logging.warning(f"finish_request failed: {e}")

    async def update_request_preview(
        self, request_id: int | None, prompt_preview: str
    ) -> None:
        if request_id is None:
            return
        preview = (prompt_preview or "")[:300]
        try:
            if not self.available:
                for row in self._requests:
                    if row["id"] == request_id:
                        row["prompt_preview"] = preview
                        break
                return
            await self.pool.execute(
                "UPDATE requests SET prompt_preview = $2 WHERE id = $1",
                int(request_id),
                preview,
            )
        except Exception as e:
            logging.warning(f"update_request_preview failed: {e}")

    async def add_call(
        self,
        *,
        request_id: int | None,
        user_id: int | None,
        chat_id: int | None,
        provider: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        reasoning_tokens: int = 0,
        cached_tokens: int = 0,
        total_tokens: int = 0,
        credit: float = 0,
        latency_ms: int | None = None,
        status: str = "ok",
        error: str = "",
    ) -> None:
        row = {
            "request_id": request_id,
            "created_at": _now(),
            "user_id": user_id,
            "chat_id": chat_id,
            "provider": provider,
            "model": model,
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "reasoning_tokens": int(reasoning_tokens or 0),
            "cached_tokens": int(cached_tokens or 0),
            "total_tokens": int(total_tokens or 0),
            "credit": float(credit or 0),
            "latency_ms": latency_ms,
            "status": status,
            "error": (error or "")[:1000],
        }
        if not self.available:
            self._calls.append(row)
            return
        try:
            await self.pool.execute(
                "INSERT INTO model_calls (request_id, user_id, chat_id, provider, model, "
                "prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens, "
                "total_tokens, credit, latency_ms, status, error) "
                "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)",
                row["request_id"],
                row["user_id"],
                row["chat_id"],
                row["provider"],
                row["model"],
                row["prompt_tokens"],
                row["completion_tokens"],
                row["reasoning_tokens"],
                row["cached_tokens"],
                row["total_tokens"],
                row["credit"],
                row["latency_ms"],
                row["status"],
                row["error"],
            )
        except Exception as e:
            logging.warning(f"add_call failed: {e}")

    async def add_tool(
        self,
        *,
        request_id: int | None,
        user_id: int | None,
        chat_id: int | None,
        tool: str,
        args_preview: str = "",
        status: str = "ok",
        latency_ms: int | None = None,
        error: str = "",
    ) -> None:
        row = {
            "request_id": request_id,
            "created_at": _now(),
            "user_id": user_id,
            "chat_id": chat_id,
            "tool": tool,
            "args_preview": (args_preview or "")[:300],
            "status": status,
            "latency_ms": latency_ms,
            "error": (error or "")[:1000],
        }
        if not self.available:
            self._tools.append(row)
            return
        try:
            await self.pool.execute(
                "INSERT INTO tool_calls (request_id, user_id, chat_id, tool, args_preview, "
                "status, latency_ms, error) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
                row["request_id"],
                row["user_id"],
                row["chat_id"],
                row["tool"],
                row["args_preview"],
                row["status"],
                row["latency_ms"],
                row["error"],
            )
        except Exception as e:
            logging.warning(f"add_tool failed: {e}")

    async def add_admin_action(self, admin_id: int, action: str, details: str = "") -> None:
        row = {
            "created_at": _now(),
            "admin_id": admin_id,
            "action": action,
            "details": (details or "")[:1000],
        }
        if not self.available:
            self._actions.append(row)
            return
        try:
            await self.pool.execute(
                "INSERT INTO admin_actions (admin_id, action, details) VALUES ($1,$2,$3)",
                admin_id,
                action,
                (details or "")[:1000],
            )
        except Exception as e:
            logging.warning(f"add_admin_action failed: {e}")

    # ------------------------------------------------------------------
    # Аналитика
    # ------------------------------------------------------------------

    def _period_filter(self, rows: list[dict], start: datetime | None, end: datetime | None):
        for row in rows:
            created = row.get("created_at")
            if created is None:
                continue
            if start is not None and created < start:
                continue
            if end is not None and created >= end:
                continue
            yield row

    async def stats(self, start: datetime | None, end: datetime | None = None) -> dict:
        contexts = await self.context_summary()
        result: dict[str, Any] = {
            "requests": 0,
            "group_messages": 0,
            "request_errors": 0,
            "running": 0,
            "avg_latency": 0,
            "calls": 0,
            "call_errors": 0,
            "models": [],
            "top_users": [],
            "top_chats": [],
            "tools": [],
            "credits": 0.0,
            "contexts": contexts,
        }

        if not self.available:
            reqs = list(self._period_filter(self._requests, start, end))
            calls = list(self._period_filter(self._calls, start, end))
            tools = list(self._period_filter(self._tools, start, end))

            result["requests"] = sum(1 for r in reqs if r.get("is_request"))
            result["group_messages"] = sum(1 for r in reqs if r.get("kind") == "group_message")
            result["request_errors"] = sum(
                1 for r in reqs if r.get("is_request") and r.get("status") == "error"
            )
            result["running"] = sum(1 for r in reqs if r.get("status") == "running")
            lat = [r["latency_ms"] for r in reqs if r.get("is_request") and r.get("latency_ms")]
            result["avg_latency"] = int(sum(lat) / len(lat)) if lat else 0
            result["calls"] = len(calls)
            result["call_errors"] = sum(1 for c in calls if c.get("status") == "error")
            result["credits"] = round(sum(float(c.get("credit") or 0) for c in calls), 4)

            by_model: dict[tuple[str, str], dict] = {}
            for c in calls:
                key = (c.get("provider") or "", c.get("model") or "")
                item = by_model.setdefault(
                    key,
                    {
                        "provider": key[0],
                        "model": key[1],
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "reasoning_tokens": 0,
                        "cached_tokens": 0,
                        "total_tokens": 0,
                        "credit": 0.0,
                    },
                )
                item["calls"] += 1
                for field in (
                    "prompt_tokens",
                    "completion_tokens",
                    "reasoning_tokens",
                    "cached_tokens",
                    "total_tokens",
                ):
                    item[field] += int(c.get(field) or 0)
                item["credit"] += float(c.get("credit") or 0)
            result["models"] = sorted(
                by_model.values(), key=lambda r: (r["total_tokens"], r["calls"]), reverse=True
            )

            users: dict[int, dict] = {}
            chats: dict[int, dict] = {}
            for r in reqs:
                if not r.get("is_request"):
                    continue
                uid = r.get("user_id")
                if uid is not None:
                    item = users.setdefault(
                        int(uid),
                        {"user_id": int(uid), "username": r.get("username") or "", "requests": 0},
                    )
                    item["requests"] += 1
                cid = r.get("chat_id")
                if cid is not None:
                    item = chats.setdefault(
                        int(cid),
                        {"chat_id": int(cid), "chat_type": r.get("chat_type") or "", "requests": 0},
                    )
                    item["requests"] += 1
            result["top_users"] = sorted(
                users.values(), key=lambda x: x["requests"], reverse=True
            )[:7]
            result["top_chats"] = sorted(
                chats.values(), key=lambda x: x["requests"], reverse=True
            )[:7]

            tool_counts: dict[str, int] = {}
            for t in tools:
                tool_counts[t.get("tool") or "?"] = tool_counts.get(t.get("tool") or "?", 0) + 1
            result["tools"] = sorted(
                [{"tool": k, "count": v} for k, v in tool_counts.items()],
                key=lambda x: x["count"],
                reverse=True,
            )
            return result

        try:
            req_row = await self.pool.fetchrow(
                "SELECT "
                "count(*) FILTER (WHERE is_request) AS requests, "
                "count(*) FILTER (WHERE kind = 'group_message') AS group_messages, "
                "count(*) FILTER (WHERE is_request AND status = 'error') AS request_errors, "
                "count(*) FILTER (WHERE status = 'running') AS running, "
                "avg(latency_ms) FILTER (WHERE is_request) AS avg_latency "
                "FROM requests WHERE ($1::timestamptz IS NULL OR created_at >= $1) "
                "AND ($2::timestamptz IS NULL OR created_at < $2)",
                start,
                end,
            )
            if req_row:
                result["requests"] = int(req_row["requests"] or 0)
                result["group_messages"] = int(req_row["group_messages"] or 0)
                result["request_errors"] = int(req_row["request_errors"] or 0)
                result["running"] = int(req_row["running"] or 0)
                result["avg_latency"] = int(req_row["avg_latency"] or 0)

            call_row = await self.pool.fetchrow(
                "SELECT count(*) AS calls, "
                "count(*) FILTER (WHERE status = 'error') AS call_errors, "
                "coalesce(sum(credit), 0) AS credits "
                "FROM model_calls WHERE ($1::timestamptz IS NULL OR created_at >= $1) "
                "AND ($2::timestamptz IS NULL OR created_at < $2)",
                start,
                end,
            )
            if call_row:
                result["calls"] = int(call_row["calls"] or 0)
                result["call_errors"] = int(call_row["call_errors"] or 0)
                result["credits"] = round(float(call_row["credits"] or 0), 4)

            model_rows = await self.pool.fetch(
                "SELECT provider, model, count(*) AS calls, "
                "coalesce(sum(prompt_tokens),0) AS prompt_tokens, "
                "coalesce(sum(completion_tokens),0) AS completion_tokens, "
                "coalesce(sum(reasoning_tokens),0) AS reasoning_tokens, "
                "coalesce(sum(cached_tokens),0) AS cached_tokens, "
                "coalesce(sum(total_tokens),0) AS total_tokens, "
                "coalesce(sum(credit),0) AS credit "
                "FROM model_calls WHERE ($1::timestamptz IS NULL OR created_at >= $1) "
                "AND ($2::timestamptz IS NULL OR created_at < $2) "
                "GROUP BY provider, model ORDER BY total_tokens DESC, calls DESC",
                start,
                end,
            )
            result["models"] = [
                {
                    "provider": r["provider"] or "",
                    "model": r["model"] or "",
                    "calls": int(r["calls"] or 0),
                    "prompt_tokens": int(r["prompt_tokens"] or 0),
                    "completion_tokens": int(r["completion_tokens"] or 0),
                    "reasoning_tokens": int(r["reasoning_tokens"] or 0),
                    "cached_tokens": int(r["cached_tokens"] or 0),
                    "total_tokens": int(r["total_tokens"] or 0),
                    "credit": float(r["credit"] or 0),
                }
                for r in model_rows
            ]

            user_rows = await self.pool.fetch(
                "SELECT user_id, max(username) AS username, count(*) AS requests "
                "FROM requests WHERE is_request AND user_id IS NOT NULL "
                "AND ($1::timestamptz IS NULL OR created_at >= $1) "
                "AND ($2::timestamptz IS NULL OR created_at < $2) "
                "GROUP BY user_id ORDER BY requests DESC LIMIT 7",
                start,
                end,
            )
            result["top_users"] = [
                {
                    "user_id": int(r["user_id"]),
                    "username": r["username"] or "",
                    "requests": int(r["requests"] or 0),
                }
                for r in user_rows
            ]

            chat_rows = await self.pool.fetch(
                "SELECT chat_id, max(chat_type) AS chat_type, count(*) AS requests "
                "FROM requests WHERE is_request AND chat_id IS NOT NULL "
                "AND ($1::timestamptz IS NULL OR created_at >= $1) "
                "AND ($2::timestamptz IS NULL OR created_at < $2) "
                "GROUP BY chat_id ORDER BY requests DESC LIMIT 7",
                start,
                end,
            )
            result["top_chats"] = [
                {
                    "chat_id": int(r["chat_id"]),
                    "chat_type": r["chat_type"] or "",
                    "requests": int(r["requests"] or 0),
                }
                for r in chat_rows
            ]

            tool_rows = await self.pool.fetch(
                "SELECT tool, count(*) AS count FROM tool_calls "
                "WHERE ($1::timestamptz IS NULL OR created_at >= $1) "
                "AND ($2::timestamptz IS NULL OR created_at < $2) "
                "GROUP BY tool ORDER BY count DESC",
                start,
                end,
            )
            result["tools"] = [
                {"tool": r["tool"], "count": int(r["count"] or 0)} for r in tool_rows
            ]
        except Exception as e:
            logging.exception(f"stats failed: {e}")
        return result

    async def context_summary(self) -> dict:
        if not self.available:
            messages = sum(len(v) for v in self._contexts.values())
            chars = sum(_char_count(v) for v in self._contexts.values())
            return {
                "chats": len(self._contexts),
                "messages": messages,
                "chars": chars,
                "tokens_est": chars // 4,
            }
        try:
            row = await self.pool.fetchrow(
                "SELECT count(*) AS chats, coalesce(sum(message_count),0) AS messages, "
                "coalesce(sum(char_count),0) AS chars FROM contexts"
            )
            chars = int(row["chars"] or 0) if row else 0
            return {
                "chats": int(row["chats"] or 0) if row else 0,
                "messages": int(row["messages"] or 0) if row else 0,
                "chars": chars,
                "tokens_est": chars // 4,
            }
        except Exception as e:
            logging.warning(f"context_summary failed: {e}")
            return {"chats": 0, "messages": 0, "chars": 0, "tokens_est": 0}

    async def db_size(self) -> int:
        if not self.available:
            return 0
        try:
            return int(
                await self.pool.fetchval("SELECT pg_database_size(current_database())") or 0
            )
        except Exception as e:
            logging.warning(f"db_size failed: {e}")
            return 0

    async def list_requests(
        self,
        *,
        limit: int = 8,
        offset: int = 0,
        user_id: int | None = None,
        model: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> tuple[list[dict], int]:
        if not self.available:
            rows = [
                r
                for r in self._requests
                if (user_id is None or r.get("user_id") == user_id)
                and (model is None or r.get("model") == model)
                and (start is None or (r.get("created_at") and r["created_at"] >= start))
                and (end is None or (r.get("created_at") and r["created_at"] < end))
            ]
            rows = sorted(rows, key=lambda x: x["id"], reverse=True)
            total = len(rows)
            page = rows[offset: offset + limit]
            for r in page:
                r["tokens"] = sum(
                    c.get("total_tokens", 0) for c in self._calls if c.get("request_id") == r["id"]
                )
            return page, total
        try:
            where = [
                "($1::bigint IS NULL OR r.user_id = $1)",
                "($2::text IS NULL OR r.model = $2)",
                "($3::timestamptz IS NULL OR r.created_at >= $3)",
                "($4::timestamptz IS NULL OR r.created_at < $4)",
            ]
            args = [user_id, model, start, end]
            where_sql = " AND ".join(where)
            total = await self.pool.fetchval(
                f"SELECT count(*) FROM requests r WHERE {where_sql}", *args
            )
            rows = await self.pool.fetch(
                f"SELECT r.*, coalesce(sum(c.total_tokens),0) AS tokens "
                f"FROM requests r LEFT JOIN model_calls c ON c.request_id = r.id "
                f"WHERE {where_sql} GROUP BY r.id ORDER BY r.id DESC "
                f"LIMIT $5 OFFSET $6",
                *args,
                int(limit),
                int(offset),
            )
            return [dict(r) for r in rows], int(total or 0)
        except Exception as e:
            logging.warning(f"list_requests failed: {e}")
            return [], 0

    async def export_requests(
        self,
        *,
        user_id: int | None = None,
        model: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 20000,
    ) -> str:
        rows, _ = await self.list_requests(
            limit=min(limit, 50000),
            offset=0,
            user_id=user_id,
            model=model,
            start=start,
            end=end,
        )
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "id",
                "created_at_msk",
                "user_id",
                "username",
                "chat_id",
                "chat_type",
                "kind",
                "is_request",
                "model",
                "status",
                "latency_ms",
                "tokens",
                "tools",
                "prompt_preview",
                "error",
            ]
        )
        for r in rows:
            created = r.get("created_at")
            if isinstance(created, datetime):
                created = created.astimezone(MSK).strftime("%d.%m.%Y %H:%M:%S")
            writer.writerow(
                [
                    r.get("id"),
                    created,
                    r.get("user_id"),
                    r.get("username"),
                    r.get("chat_id"),
                    r.get("chat_type"),
                    r.get("kind"),
                    "1" if r.get("is_request") else "0",
                    r.get("model"),
                    r.get("status"),
                    r.get("latency_ms"),
                    r.get("tokens") or 0,
                    r.get("tools"),
                    (r.get("prompt_preview") or "").replace("\n", " "),
                    (r.get("error") or "").replace("\n", " "),
                ]
            )
        return "\ufeff" + buf.getvalue()

    async def cleanup_logs(self, days: int) -> int:
        cutoff = _now() - timedelta(days=max(1, int(days)))
        if not self.available:
            before = len(self._requests)
            self._requests = [
                r for r in self._requests if r.get("created_at") and r["created_at"] >= cutoff
            ]
            self._calls = [
                c for c in self._calls if c.get("created_at") and c["created_at"] >= cutoff
            ]
            self._tools = [
                t for t in self._tools if t.get("created_at") and t["created_at"] >= cutoff
            ]
            return before - len(self._requests)
        try:
            rows = await self.pool.execute(
                "DELETE FROM requests WHERE created_at < $1", cutoff
            )
            await self.pool.execute("DELETE FROM model_calls WHERE created_at < $1", cutoff)
            await self.pool.execute("DELETE FROM tool_calls WHERE created_at < $1", cutoff)
            try:
                return int(str(rows).split()[-1])
            except Exception:
                return 0
        except Exception as e:
            logging.warning(f"cleanup_logs failed: {e}")
            return 0


storage = Storage(os.environ.get("DATABASE_URL", ""))
