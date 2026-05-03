"""
memory.py
=========
Долговременная память пользователя на SQLite.

Каждое сообщение (как от пользователя, так и от ассистента)
сохраняется в одну таблицу `messages` с полями:
    id, user_id, role, content, created_at

При каждом новом запросе из БД достаются последние N сообщений
и подмешиваются в промпт как "история диалога".

Почему SQLite, а не JSON:
  * атомарные транзакции => данные не теряются при сбое;
  * индексы по user_id => быстрая выборка даже при росте истории;
  * стандартная библиотека Python => нет лишних зависимостей;
  * легко расширить на нескольких пользователей.

ВАЖНО про роли:
  Внутри БД мы храним роли в нейтральном формате "user" / "assistant",
  а конвертация в формат конкретного LLM (для Gemini это "user"/"model")
  происходит в gemini_client.py. Так модуль памяти остаётся независимым
  от выбора LLM.
"""

import sqlite3
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

import config


class MemoryStore:
    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _conn(self):
        """Контекстный менеджер: гарантирует commit и close."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Создание схемы при первом запуске. Идемпотентно."""
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    TEXT    NOT NULL,
                    role       TEXT    NOT NULL CHECK (role IN ('user','assistant')),
                    content    TEXT    NOT NULL,
                    created_at TEXT    NOT NULL
                )
            """)
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_time "
                "ON messages(user_id, created_at)"
            )

    def add_message(self, user_id: str, role: str, content: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO messages (user_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (user_id, role, content, datetime.utcnow().isoformat()),
            )

    def get_recent_messages(self, user_id: str,
                            limit: int = config.RECENT_MESSAGES_LIMIT) -> list[dict]:
        """
        Возвращает последние `limit` сообщений в ХРОНОЛОГИЧЕСКОМ порядке
        в формате [{"role": "user"|"assistant", "content": "..."}].
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT role, content FROM messages "
                "WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        # ORDER BY DESC + reversed: достаём свежие, но возвращаем по порядку.
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def get_all_messages(self, user_id: str) -> list[dict]:
        """Полная история (для отладки / экспорта)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT role, content, created_at FROM messages "
                "WHERE user_id = ? ORDER BY id ASC",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_user(self, user_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
