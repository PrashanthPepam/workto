"""Tests for chat CRUD endpoints."""

from httpx import AsyncClient


# ── POST /chats ────────────────────────────────────────────────────────────────


async def test_create_chat_returns_201_and_fields(client: AsyncClient) -> None:
    r = await client.post("/chats", json={"title": "My Chat"})
    assert r.status_code == 201
    body = r.json()
    assert body["title"] == "My Chat"
    assert "id" in body
    assert "created_at" in body
    assert "updated_at" in body


async def test_create_chat_empty_title_returns_422(client: AsyncClient) -> None:
    r = await client.post("/chats", json={"title": ""})
    assert r.status_code == 422


async def test_create_chat_missing_title_returns_422(client: AsyncClient) -> None:
    r = await client.post("/chats", json={})
    assert r.status_code == 422


async def test_create_chat_title_too_long_returns_422(client: AsyncClient) -> None:
    r = await client.post("/chats", json={"title": "x" * 201})
    assert r.status_code == 422


# ── GET /chats ─────────────────────────────────────────────────────────────────


async def test_list_chats_empty_db(client: AsyncClient) -> None:
    r = await client.get("/chats")
    assert r.status_code == 200
    body = r.json()
    assert body["chats"] == []
    assert body["total"] == 0


async def test_list_chats_returns_all(client: AsyncClient) -> None:
    await client.post("/chats", json={"title": "Alpha"})
    await client.post("/chats", json={"title": "Beta"})
    r = await client.get("/chats")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    titles = {c["title"] for c in body["chats"]}
    assert titles == {"Alpha", "Beta"}


async def test_list_chats_schema(client: AsyncClient) -> None:
    await client.post("/chats", json={"title": "Schema Test"})
    r = await client.get("/chats")
    chat = r.json()["chats"][0]
    assert set(chat.keys()) == {"id", "title", "created_at", "updated_at"}


# ── GET /chats/{chat_id} ───────────────────────────────────────────────────────


async def test_get_chat_success(client: AsyncClient) -> None:
    created = (await client.post("/chats", json={"title": "Fetch Me"})).json()
    r = await client.get(f"/chats/{created['id']}")
    assert r.status_code == 200
    assert r.json()["id"] == created["id"]
    assert r.json()["title"] == "Fetch Me"


async def test_get_chat_not_found(client: AsyncClient) -> None:
    r = await client.get("/chats/nonexistent-id")
    assert r.status_code == 404
    assert r.json()["detail"] == "Chat not found"


# ── DELETE /chats/{chat_id} ────────────────────────────────────────────────────


async def test_delete_chat_returns_204(client: AsyncClient) -> None:
    created = (await client.post("/chats", json={"title": "Delete Me"})).json()
    r = await client.delete(f"/chats/{created['id']}")
    assert r.status_code == 204
    assert r.content == b""  # 204 must have no body


async def test_delete_chat_removes_it(client: AsyncClient) -> None:
    created = (await client.post("/chats", json={"title": "Gone"})).json()
    await client.delete(f"/chats/{created['id']}")
    r = await client.get(f"/chats/{created['id']}")
    assert r.status_code == 404


async def test_delete_chat_not_found(client: AsyncClient) -> None:
    r = await client.delete("/chats/nonexistent-id")
    assert r.status_code == 404
    assert r.json()["detail"] == "Chat not found"


async def test_delete_chat_decrements_list(client: AsyncClient) -> None:
    c1 = (await client.post("/chats", json={"title": "Keep"})).json()
    c2 = (await client.post("/chats", json={"title": "Remove"})).json()
    await client.delete(f"/chats/{c2['id']}")
    r = await client.get("/chats")
    assert r.json()["total"] == 1
    assert r.json()["chats"][0]["id"] == c1["id"]
