"""
Pydantic v2 request / response schemas.

Only the public API surface is modelled here.  Internal storage details
(tool_calls JSON blob, tool_call_id) are handled at the database layer and
never exposed in responses — clients have no need for them.
"""

from pydantic import BaseModel, Field


# ── Chats ──────────────────────────────────────────────────────────────────────


class ChatCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class Chat(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class ChatList(BaseModel):
    chats: list[Chat]
    total: int


# ── Messages ───────────────────────────────────────────────────────────────────


class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1)


class Message(BaseModel):
    id: str
    chat_id: str
    # Exposed roles: 'user' | 'assistant'.  'tool' messages are internal
    # scaffolding stored in the DB but filtered out before API responses.
    role: str
    content: str | None
    created_at: str


class MessageList(BaseModel):
    messages: list[Message]


# ── Ops endpoints ──────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str    # "ok" | "degraded"
    database: str  # "ok" | "error"
