from pydantic import BaseModel, Field


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
