"""
Agent loop tests.

All OpenAI calls are mocked — tests run without a real API key.
The `knowledge_dir` fixture (from conftest) creates sample KB files and
monkeypatches settings.knowledge_dir to point at them.

Mock helpers
------------
_resp(content, tool_calls=None)  — build a mock completions response.
_tc(id, name, args)              — build a mock ToolCall object.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from app import database


# ── Mock helpers ───────────────────────────────────────────────────────────────


def _resp(content: str | None, tool_calls=None) -> MagicMock:
    """Build a minimal mock response from chat.completions.create()."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop" if not tool_calls else "tool_calls"

    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _tc(call_id: str, name: str, arguments: str) -> MagicMock:
    """Build a mock ToolCall object."""
    fn = MagicMock()
    fn.name = name
    fn.arguments = arguments

    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function = fn
    return tc


# ── Tool dispatch tests (no HTTP, no DB) ───────────────────────────────────────


def test_dispatch_list_files(knowledge_dir) -> None:
    from app.agent.tools import dispatch
    from app.kb.store import KBStore
    from app import config

    kb = KBStore(config.settings.knowledge_dir)
    result, files_read = dispatch("list_knowledge_files", "{}", kb, 0, 2)

    assert "python_async_basics.txt" in result
    assert "fastapi_routing_guide.txt" in result
    assert files_read == 0  # listing does not count against quota


def test_dispatch_read_file(knowledge_dir) -> None:
    from app.agent.tools import dispatch
    from app.kb.store import KBStore
    from app import config

    kb = KBStore(config.settings.knowledge_dir)
    result, files_read = dispatch(
        "read_knowledge_file", '{"filename": "python_async_basics.txt"}', kb, 0, 2
    )

    assert "asyncio" in result.lower()
    assert files_read == 1


def test_dispatch_read_file_limit_enforced(knowledge_dir) -> None:
    from app.agent.tools import dispatch
    from app.kb.store import KBStore
    from app import config

    kb = KBStore(config.settings.knowledge_dir)
    result, files_read = dispatch(
        "read_knowledge_file", '{"filename": "python_async_basics.txt"}', kb, 2, 2
    )

    assert "maximum" in result.lower() or "already read" in result.lower()
    assert files_read == 2  # unchanged


def test_dispatch_unknown_tool(knowledge_dir) -> None:
    from app.agent.tools import dispatch
    from app.kb.store import KBStore
    from app import config

    kb = KBStore(config.settings.knowledge_dir)
    result, _ = dispatch("nonexistent_tool", "{}", kb, 0, 2)
    assert "unknown tool" in result.lower()


def test_dispatch_bad_json_arguments(knowledge_dir) -> None:
    from app.agent.tools import dispatch
    from app.kb.store import KBStore
    from app import config

    kb = KBStore(config.settings.knowledge_dir)
    result, _ = dispatch("read_knowledge_file", "{not valid json}", kb, 0, 2)
    assert "error" in result.lower()


# ── Agent loop via HTTP (mocked OpenAI) ────────────────────────────────────────


async def test_agent_direct_answer(
    client: AsyncClient, chat_id: str, knowledge_dir
) -> None:
    """Model answers immediately without calling any tools."""
    with patch("app.agent.runner._make_client") as mock_make:
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(
            return_value=_resp("Paris is the capital of France.")
        )
        mock_make.return_value = mock_openai

        r = await client.post(
            f"/chats/{chat_id}/messages",
            json={"content": "What is the capital of France?"},
        )

    assert r.status_code == 201
    assert r.json()["content"] == "Paris is the capital of France."
    assert mock_openai.chat.completions.create.call_count == 1


async def test_agent_tool_call_then_answer(
    client: AsyncClient, chat_id: str, knowledge_dir
) -> None:
    """Full loop: list_files → read_file → final answer."""
    tc_list = _tc("c1", "list_knowledge_files", "{}")
    tc_read = _tc("c2", "read_knowledge_file", '{"filename": "python_async_basics.txt"}')

    with patch("app.agent.runner._make_client") as mock_make:
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(
            side_effect=[
                _resp(None, tool_calls=[tc_list]),
                _resp(None, tool_calls=[tc_read]),
                _resp("asyncio enables concurrent I/O."),
            ]
        )
        mock_make.return_value = mock_openai

        r = await client.post(
            f"/chats/{chat_id}/messages",
            json={"content": "Explain async programming."},
        )

    assert r.status_code == 201
    body = r.json()
    assert body["role"] == "assistant"
    assert "asyncio" in body["content"]
    assert mock_openai.chat.completions.create.call_count == 3


async def test_agent_tool_scaffolding_persisted_to_db(
    client: AsyncClient, chat_id: str, knowledge_dir
) -> None:
    """Intermediate tool messages must be stored in the DB."""
    tc = _tc("c1", "list_knowledge_files", "{}")

    with patch("app.agent.runner._make_client") as mock_make:
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(
            side_effect=[
                _resp(None, tool_calls=[tc]),
                _resp("Here is the list of files."),
            ]
        )
        mock_make.return_value = mock_openai

        await client.post(
            f"/chats/{chat_id}/messages", json={"content": "List KB files"}
        )

    # The DB must contain user, assistant(tool_calls), tool, assistant(final)
    all_rows = await database.fetch_all(
        "SELECT role FROM messages WHERE chat_id = ? ORDER BY created_at", (chat_id,)
    )
    roles = [r["role"] for r in all_rows]
    assert roles == ["user", "assistant", "tool", "assistant"]


async def test_agent_tool_messages_not_in_public_api(
    client: AsyncClient, chat_id: str, knowledge_dir
) -> None:
    """GET /messages must filter out the internal role='tool' rows."""
    tc = _tc("c1", "list_knowledge_files", "{}")

    with patch("app.agent.runner._make_client") as mock_make:
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(
            side_effect=[
                _resp(None, tool_calls=[tc]),
                _resp("Done."),
            ]
        )
        mock_make.return_value = mock_openai

        await client.post(f"/chats/{chat_id}/messages", json={"content": "Hello"})

    r = await client.get(f"/chats/{chat_id}/messages")
    roles = {m["role"] for m in r.json()["messages"]}
    assert "tool" not in roles
    assert roles == {"user", "assistant"}


async def test_agent_max_iterations_guard(
    client: AsyncClient, chat_id: str, knowledge_dir, monkeypatch
) -> None:
    """Loop must terminate after agent_max_iterations and return a fallback message."""
    from app import config

    monkeypatch.setattr(config.settings, "agent_max_iterations", 2)

    # Model always returns a tool call — never a final answer
    tc = _tc("c1", "list_knowledge_files", "{}")
    runaway = _resp(None, tool_calls=[tc])

    with patch("app.agent.runner._make_client") as mock_make:
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=runaway)
        mock_make.return_value = mock_openai

        r = await client.post(
            f"/chats/{chat_id}/messages", json={"content": "Run forever"}
        )

    assert r.status_code == 201
    assert mock_openai.chat.completions.create.call_count == 2  # exactly max_iterations
    content = r.json()["content"].lower()
    assert "unable" in content or "steps" in content


async def test_agent_history_included_in_second_turn(
    client: AsyncClient, chat_id: str, knowledge_dir
) -> None:
    """On the second message, the first exchange must be in the context sent to the LLM."""
    captured_messages: list = []

    async def capture_and_respond(**kwargs):
        captured_messages.append(kwargs["messages"])
        return _resp("Response.")

    with patch("app.agent.runner._make_client") as mock_make:
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(side_effect=capture_and_respond)
        mock_make.return_value = mock_openai

        await client.post(f"/chats/{chat_id}/messages", json={"content": "First question"})
        await client.post(f"/chats/{chat_id}/messages", json={"content": "Second question"})

    # The second call's messages list must include context from the first exchange
    second_call_msgs = captured_messages[1]
    roles = [m["role"] for m in second_call_msgs]
    assert roles.count("user") >= 2
    assert roles.count("assistant") >= 1


async def test_agent_api_error_propagates(
    client: AsyncClient, chat_id: str, knowledge_dir
) -> None:
    """An OpenAI APIError must propagate as a 500 response."""
    from openai import APIStatusError

    with patch("app.agent.runner._make_client") as mock_make:
        mock_openai = MagicMock()
        # APIStatusError needs a response object; use a plain APIError instead
        from openai import APIConnectionError
        mock_openai.chat.completions.create = AsyncMock(
            side_effect=APIConnectionError(request=MagicMock())
        )
        mock_make.return_value = mock_openai

        r = await client.post(
            f"/chats/{chat_id}/messages", json={"content": "Trigger API error"}
        )

    assert r.status_code == 500
