from pydantic import BaseModel, Field


class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1)


class Message(BaseModel):
    id: str
    chat_id: str
    # Exposed roles: 'user' | 'assistant'.
    # 'tool' messages are internal scaffolding; filtered out before API responses.
    role: str
    content: str | None
    created_at: str


class MessageList(BaseModel):
    messages: list[Message]
