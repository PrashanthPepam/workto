"""
Chat resource router  —  /chats

Handles the full lifecycle of a chat session.  Messages live in a child router
(messages.py) so each file owns exactly one resource type.

HTTP conventions used:
  201 Created   — successful POST
  204 No Content — successful DELETE (no body)
  404 Not Found  — chat_id doesn't exist
  422 Unprocessable — Pydantic validation failure (handled automatically)
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from app import database
from app.models.chat import Chat, ChatCreate, ChatList

router = APIRouter(prefix="/chats", tags=["chats"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _require_chat(chat_id: str) -> dict:
    """Fetch a chat row or raise 404.  Shared by GET, DELETE, and messages router."""
    row = await database.fetch_one("SELECT * FROM chats WHERE id = ?", (chat_id,))
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    return row


# ── Collection endpoints ───────────────────────────────────────────────────────


@router.post("", response_model=Chat, status_code=status.HTTP_201_CREATED)
async def create_chat(body: ChatCreate) -> Chat:
    chat_id = str(uuid.uuid4())
    now = _now()
    await database.execute(
        "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (chat_id, body.title, now, now),
    )
    return Chat(id=chat_id, title=body.title, created_at=now, updated_at=now)


@router.get("", response_model=ChatList)
async def list_chats() -> ChatList:
    # Most-recently-updated first — matches a typical chat-app inbox ordering.
    rows = await database.fetch_all(
        "SELECT * FROM chats ORDER BY updated_at DESC"
    )
    chats = [Chat(**row) for row in rows]
    return ChatList(chats=chats, total=len(chats))


# ── Item endpoints ─────────────────────────────────────────────────────────────


@router.get("/{chat_id}", response_model=Chat)
async def get_chat(chat_id: str) -> Chat:
    row = await _require_chat(chat_id)
    return Chat(**row)


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(chat_id: str) -> None:
    await _require_chat(chat_id)
    # ON DELETE CASCADE in the schema removes child messages automatically.
    await database.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
