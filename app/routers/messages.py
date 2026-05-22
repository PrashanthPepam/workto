"""
Message resource router  —  /chats/{chat_id}/messages

POST flow
---------
1. Verify the parent chat exists (404 otherwise).
2. Persist the incoming user message.
3. Delegate to the agent runner which manages the LLM + tool-calling loop
   and returns the final assistant text.
4. Persist the assistant response.
5. Bump chat.updated_at so list ordering stays fresh.
6. Return the assistant Message to the caller.

The agent runner is imported from app.agent.runner.  Phase 3 ships a minimal
stub; Phase 4 replaces the stub with the real agentic loop.

GET /messages filters out role='tool' rows — those are internal scaffolding
for the agent loop and have no meaning in the public API.
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status
from openai import APIError

from app import database
from app.agent.runner import run_agent
from app.models.message import Message, MessageCreate, MessageList

router = APIRouter(prefix="/chats", tags=["messages"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _require_chat(chat_id: str) -> None:
    row = await database.fetch_one("SELECT id FROM chats WHERE id = ?", (chat_id,))
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")


# ── Endpoints ──────────────────────────────────────────────────────────────────


@router.get("/{chat_id}/messages", response_model=MessageList)
async def get_messages(chat_id: str) -> MessageList:
    await _require_chat(chat_id)
    rows = await database.fetch_all(
        """
        SELECT id, chat_id, role, content, created_at
        FROM   messages
        WHERE  chat_id = ?
          AND  role IN ('user', 'assistant')
        ORDER  BY created_at ASC
        """,
        (chat_id,),
    )
    return MessageList(messages=[Message(**row) for row in rows])


@router.post(
    "/{chat_id}/messages",
    response_model=Message,
    status_code=status.HTTP_201_CREATED,
)
async def post_message(chat_id: str, body: MessageCreate) -> Message:
    await _require_chat(chat_id)

    # 1. Persist user turn
    user_id = str(uuid.uuid4())
    user_ts = _now()
    await database.execute(
        "INSERT INTO messages (id, chat_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, chat_id, "user", body.content, user_ts),
    )

    # 2. Run agent (Phase 4 replaces the stub with the real LLM loop)
    try:
        assistant_content = await run_agent(chat_id=chat_id)
    except APIError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM service error",
        ) from exc

    # 3. Persist assistant turn
    asst_id = str(uuid.uuid4())
    asst_ts = _now()
    await database.execute(
        "INSERT INTO messages (id, chat_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
        (asst_id, chat_id, "assistant", assistant_content, asst_ts),
    )

    # 4. Keep chat.updated_at current so list ordering reflects last activity
    await database.execute(
        "UPDATE chats SET updated_at = ? WHERE id = ?",
        (asst_ts, chat_id),
    )

    return Message(
        id=asst_id,
        chat_id=chat_id,
        role="assistant",
        content=assistant_content,
        created_at=asst_ts,
    )
