"""Tests for message endpoints."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient

from app import database


@pytest.fixture(autouse=True)
def _mock_run_agent(monkeypatch):
    """Stub the agent loop so message-router tests don't need a real API key."""
    from app.routers import messages

    monkeypatch.setattr(messages, "run_agent", AsyncMock(return_value="Mocked answer."))


# ── GET /chats/{chat_id}/messages ──────────────────────────────────────────────


async def test_get_messages_empty(client: AsyncClient, chat_id: str) -> None:
    r = await client.get(f"/chats/{chat_id}/messages")
    assert r.status_code == 200
    assert r.json()["messages"] == []


async def test_get_messages_nonexistent_chat_returns_404(client: AsyncClient) -> None:
    r = await client.get("/chats/no-such-chat/messages")
    assert r.status_code == 404
    assert r.json()["detail"] == "Chat not found"


async def test_tool_messages_excluded_from_get(
    client: AsyncClient, chat_id: str
) -> None:
    """role='tool' rows are internal agent scaffolding; never surfaced in GET."""
    await database.execute(
        "INSERT INTO messages (id, chat_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), chat_id, "tool", '{"kb": "content"}',
         datetime.now(timezone.utc).isoformat()),
    )
    r = await client.get(f"/chats/{chat_id}/messages")
    roles = [m["role"] for m in r.json()["messages"]]
    assert "tool" not in roles


# ── POST /chats/{chat_id}/messages ─────────────────────────────────────────────


async def test_post_message_returns_201_assistant(
    client: AsyncClient, chat_id: str
) -> None:
    r = await client.post(
        f"/chats/{chat_id}/messages",
        json={"content": "What is FastAPI?"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["role"] == "assistant"
    assert body["chat_id"] == chat_id
    assert body["content"] is not None
    assert "id" in body
    assert "created_at" in body


async def test_post_message_empty_content_returns_422(
    client: AsyncClient, chat_id: str
) -> None:
    r = await client.post(f"/chats/{chat_id}/messages", json={"content": ""})
    assert r.status_code == 422


async def test_post_message_missing_content_returns_422(
    client: AsyncClient, chat_id: str
) -> None:
    r = await client.post(f"/chats/{chat_id}/messages", json={})
    assert r.status_code == 422


async def test_post_message_nonexistent_chat_returns_404(
    client: AsyncClient,
) -> None:
    r = await client.post(
        "/chats/no-such-chat/messages",
        json={"content": "Hello"},
    )
    assert r.status_code == 404


# ── Ordering & persistence ─────────────────────────────────────────────────────


async def test_messages_stored_in_chronological_order(
    client: AsyncClient, chat_id: str
) -> None:
    """GET /messages returns user+assistant pairs in ascending created_at order."""
    await client.post(f"/chats/{chat_id}/messages", json={"content": "First"})
    await client.post(f"/chats/{chat_id}/messages", json={"content": "Second"})

    messages = (await client.get(f"/chats/{chat_id}/messages")).json()["messages"]

    # Two user+assistant pairs = 4 messages total
    assert len(messages) == 4
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "First"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == "Second"
    assert messages[3]["role"] == "assistant"


async def test_post_message_bumps_chat_updated_at(
    client: AsyncClient, chat_id: str
) -> None:
    """Posting a message must refresh chat.updated_at for list ordering."""
    before = (await client.get(f"/chats/{chat_id}")).json()["updated_at"]
    await client.post(f"/chats/{chat_id}/messages", json={"content": "Hello"})
    after = (await client.get(f"/chats/{chat_id}")).json()["updated_at"]
    # ISO-8601 strings are lexicographically comparable
    assert after >= before


async def test_delete_chat_also_removes_messages(
    client: AsyncClient, chat_id: str
) -> None:
    """ON DELETE CASCADE: deleting the chat must wipe its messages."""
    await client.post(f"/chats/{chat_id}/messages", json={"content": "Orphan?"})
    await client.delete(f"/chats/{chat_id}")

    # Chat is gone
    assert (await client.get(f"/chats/{chat_id}")).status_code == 404

    # Messages are gone too (direct DB check)
    rows = await database.fetch_all(
        "SELECT * FROM messages WHERE chat_id = ?", (chat_id,)
    )
    assert rows == []
