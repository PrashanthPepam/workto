"""
Shared pytest fixtures.

Database isolation
------------------
Each test function gets its own temporary SQLite file via pytest's `tmp_path`
fixture.  We monkeypatch `config.settings.db_path` before triggering the
lifespan so the app connects to the test DB, not the real one.

Lifespan management
-------------------
httpx's ASGITransport does NOT trigger the ASGI lifespan automatically.
We call `lifespan(app)` as an async context manager ourselves so that
startup (DB connect + schema init) and teardown (DB close) happen correctly
around each test.

OPENAI_API_KEY
--------------
Settings requires the key at import time.  We set a placeholder so that tests
that do not call the LLM can import the app without a real key in the env.
For integration tests that DO call the LLM, set the real key in your .env or
pass it via the environment before running pytest.
"""

import os

# Must be set before any app import triggers Settings()
os.environ.setdefault("OPENAI_API_KEY", "test-placeholder")

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
async def client(tmp_path, monkeypatch):
    """AsyncClient connected to the app with an isolated temp database."""
    from app import config

    monkeypatch.setattr(config.settings, "db_path", str(tmp_path / "test.db"))

    # Import after monkeypatching so lifespan picks up the new db_path
    from app.main import app, lifespan

    async with lifespan(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c


@pytest.fixture
async def db(tmp_path):
    """Bare database connection for unit-testing DB helpers directly."""
    from app import database

    await database.connect(str(tmp_path / "test.db"))
    yield
    await database.disconnect()
