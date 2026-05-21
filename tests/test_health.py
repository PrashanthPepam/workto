"""Tests for /health and /ready endpoints."""

from httpx import AsyncClient


async def test_health_returns_200(client: AsyncClient) -> None:
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_ready_returns_200_when_db_connected(client: AsyncClient) -> None:
    r = await client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"


async def test_ready_schema_shape(client: AsyncClient) -> None:
    r = await client.get("/ready")
    assert set(r.json().keys()) == {"status", "database"}
