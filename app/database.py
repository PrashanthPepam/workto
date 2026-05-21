"""
Raw aiosqlite database layer.

Design notes
------------
- Single module-level connection opened at app startup (via lifespan) and closed at
  shutdown.  SQLite is embedded and single-process, so a shared connection is the
  simplest correct approach.
- WAL journal mode enables concurrent readers alongside the single writer, which is
  important because FastAPI serves multiple requests concurrently on one event loop.
- PRAGMA foreign_keys=ON is required at connect-time; SQLite disables FK enforcement
  by default for backwards-compatibility reasons.
- PRAGMA synchronous=NORMAL (instead of the default FULL) halves fsync overhead while
  still being safe with WAL mode — a crash can lose at most the last committed
  transaction, which is acceptable for this use-case.
- Query helpers return plain dicts so callers stay decoupled from sqlite3.Row details.
- Auto-commit after every write is intentional: all mutations in this service are
  single-statement; there is no need for multi-statement transactions.
"""

import sqlite3
from pathlib import Path
from typing import Any

import aiosqlite

_db: aiosqlite.Connection | None = None


# ── Lifecycle ──────────────────────────────────────────────────────────────────


async def connect(db_path: str) -> None:
    """Open the database, configure pragmas, and initialise the schema."""
    global _db
    # Close any stale connection (e.g. during tests that call connect() repeatedly)
    if _db is not None:
        await _db.close()

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    _db = await aiosqlite.connect(db_path)
    _db.row_factory = sqlite3.Row  # enables dict(row) conversion

    await _db.execute("PRAGMA journal_mode=WAL")
    await _db.execute("PRAGMA foreign_keys=ON")
    await _db.execute("PRAGMA synchronous=NORMAL")
    await _init_schema()


async def disconnect() -> None:
    """Close the database connection gracefully."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None


def get_db() -> aiosqlite.Connection:
    """Return the active connection; raise if called before connect()."""
    if _db is None:
        raise RuntimeError("Database is not connected. Was lifespan called?")
    return _db


# ── Schema initialisation ──────────────────────────────────────────────────────


async def _init_schema() -> None:
    db = get_db()

    await db.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id         TEXT PRIMARY KEY,
            title      TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # tool_calls stores the raw JSON list from an assistant message so the agent
    # loop can reconstruct the full conversation history on follow-up requests.
    # tool_call_id links a role=tool result back to its originating tool_call.
    await db.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id           TEXT PRIMARY KEY,
            chat_id      TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            role         TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'tool')),
            content      TEXT,
            tool_calls   TEXT,
            tool_call_id TEXT,
            created_at   TEXT NOT NULL
        )
    """)

    # Covers the common query: fetch all messages for a chat ordered by time.
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id)"
    )

    await db.commit()


# ── Query helpers ──────────────────────────────────────────────────────────────


async def fetch_one(
    sql: str, params: tuple[Any, ...] = ()
) -> dict[str, Any] | None:
    """Return the first matching row as a dict, or None."""
    async with get_db().execute(sql, params) as cursor:
        row = await cursor.fetchone()
        return dict(row) if row else None


async def fetch_all(
    sql: str, params: tuple[Any, ...] = ()
) -> list[dict[str, Any]]:
    """Return all matching rows as a list of dicts."""
    async with get_db().execute(sql, params) as cursor:
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def execute(sql: str, params: tuple[Any, ...] = ()) -> None:
    """Execute a write statement and commit immediately."""
    db = get_db()
    await db.execute(sql, params)
    await db.commit()
