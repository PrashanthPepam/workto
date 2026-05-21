"""Unit tests for database helpers and schema initialisation."""

from app import database


async def test_schema_creates_expected_tables(db) -> None:
    rows = await database.fetch_all(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    names = {r["name"] for r in rows}
    assert "chats" in names
    assert "messages" in names


async def test_schema_creates_messages_index(db) -> None:
    rows = await database.fetch_all(
        "SELECT name FROM sqlite_master WHERE type='index'"
    )
    names = {r["name"] for r in rows}
    assert "idx_messages_chat_id" in names


async def test_fetch_one_returns_none_on_miss(db) -> None:
    row = await database.fetch_one(
        "SELECT * FROM chats WHERE id = ?", ("nonexistent",)
    )
    assert row is None


async def test_execute_insert_and_fetch_one(db) -> None:
    await database.execute(
        "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        ("chat-1", "Hello", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    )
    row = await database.fetch_one("SELECT * FROM chats WHERE id = ?", ("chat-1",))

    assert row is not None
    assert row["id"] == "chat-1"
    assert row["title"] == "Hello"


async def test_fetch_all_returns_all_rows(db) -> None:
    for i in range(3):
        await database.execute(
            "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (f"id-{i}", f"Chat {i}", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
        )

    rows = await database.fetch_all("SELECT * FROM chats ORDER BY id")
    assert len(rows) == 3
    assert rows[0]["id"] == "id-0"
    assert rows[2]["id"] == "id-2"


async def test_fetch_all_returns_empty_list_when_no_rows(db) -> None:
    rows = await database.fetch_all("SELECT * FROM chats")
    assert rows == []


async def test_foreign_key_cascade_delete(db) -> None:
    """ON DELETE CASCADE must remove child messages when the parent chat is deleted."""
    await database.execute(
        "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        ("chat-1", "Test", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    )
    await database.execute(
        "INSERT INTO messages (id, chat_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
        ("msg-1", "chat-1", "user", "hi", "2024-01-01T00:00:00Z"),
    )

    await database.execute("DELETE FROM chats WHERE id = ?", ("chat-1",))

    orphan = await database.fetch_one(
        "SELECT * FROM messages WHERE id = ?", ("msg-1",)
    )
    assert orphan is None


async def test_row_check_constraint_rejects_invalid_role(db) -> None:
    """The CHECK constraint on messages.role must reject unknown values."""
    import aiosqlite

    await database.execute(
        "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        ("chat-1", "Test", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    )
    try:
        await database.execute(
            "INSERT INTO messages (id, chat_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            ("msg-1", "chat-1", "invalid_role", "hi", "2024-01-01T00:00:00Z"),
        )
        assert False, "Expected an IntegrityError"
    except aiosqlite.IntegrityError:
        pass
