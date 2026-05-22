"""
Agentic QnA loop.

Flow per user turn
------------------
1. Load the full conversation history for this chat from the database
   (includes all previous user/assistant/tool messages).
2. Prepend a system prompt.
3. Call the OpenAI-compatible chat completions API with both tool definitions.
4. If the model issues tool_calls:
     a. Persist the assistant message (content + tool_calls JSON) to the DB.
     b. Execute each tool call via dispatch() — never raises.
     c. Persist each tool result (role='tool', tool_call_id=...) to the DB.
     d. Append all new messages to the in-memory list.
     e. Loop back to step 3.
5. When the model returns finish_reason='stop' (no tool_calls):
     return the final text to the router, which persists it as the
     public assistant message.

Persistence strategy
--------------------
Every turn of the loop — including intermediate tool scaffolding — is
written to the DB immediately.  This means if the process crashes mid-loop,
the partial history is visible on the next request.  The model will see the
partial context and may recover gracefully.  A more robust implementation
would wrap each full turn in a transaction and roll back on failure.

Guard rails
-----------
- agent_max_iterations caps the loop so a misbehaving model cannot spin forever.
- files_read across all loop iterations enforces agent_max_kb_files.

Testability
-----------
_make_client() is a thin factory that tests can patch to inject a mock
without touching the public run_agent() signature.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from openai import APIError, AsyncOpenAI

from app import database
from app.agent.tools import TOOLS, dispatch
from app.config import settings
from app.kb.store import KBStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a knowledge base of plain-text files. "
    "When the user asks a question, first call list_knowledge_files to see what is "
    "available, then call read_knowledge_file on the most relevant file(s) before "
    "composing your answer.  Ground your response in the knowledge base where possible. "
    "If no relevant file exists, say so clearly and answer from general knowledge."
)


# ── Client factory ─────────────────────────────────────────────────────────────


def _make_client() -> AsyncOpenAI:
    """Return a configured OpenAI-compatible async client.

    Kept as a standalone function so tests can monkeypatch it without
    modifying run_agent's signature.
    """
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_url,
    )


# ── History loader ─────────────────────────────────────────────────────────────


async def _load_history(chat_id: str) -> list[dict[str, Any]]:
    """Reconstruct the OpenAI message list from the DB for this chat.

    Row mapping
    -----------
    role='user'      → {"role": "user", "content": ...}
    role='assistant' with tool_calls → include parsed tool_calls list
    role='assistant' without         → {"role": "assistant", "content": ...}
    role='tool'      → {"role": "tool", "tool_call_id": ..., "content": ...}
    """
    rows = await database.fetch_all(
        """
        SELECT role, content, tool_calls, tool_call_id
        FROM   messages
        WHERE  chat_id = ?
        ORDER  BY created_at ASC
        """,
        (chat_id,),
    )

    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for row in rows:
        if row["role"] == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": row["tool_call_id"],
                    "content": row["content"] or "",
                }
            )
        elif row["tool_calls"]:
            # Assistant turn that triggered tool calls
            messages.append(
                {
                    "role": "assistant",
                    "content": row["content"],  # may be None — that is valid
                    "tool_calls": json.loads(row["tool_calls"]),
                }
            )
        else:
            messages.append({"role": row["role"], "content": row["content"] or ""})

    return messages


# ── Timestamp helper ───────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Agent loop ─────────────────────────────────────────────────────────────────


async def run_agent(chat_id: str) -> str:
    """Run the agentic loop for one user turn; return the final assistant text.

    The router persists the returned string as the public assistant message.
    All intermediate tool scaffolding is persisted here.
    """
    client = _make_client()
    kb = KBStore(settings.knowledge_dir)
    messages = await _load_history(chat_id)
    files_read = 0  # accumulates across all iterations for this turn

    for iteration in range(settings.agent_max_iterations):
        logger.debug(
            "agent iter=%d/%d  chat=%s  msgs=%d  files_read=%d",
            iteration + 1,
            settings.agent_max_iterations,
            chat_id,
            len(messages),
            files_read,
        )

        try:
            response = await client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except APIError:
            logger.exception("OpenAI API error on iteration %d for chat %s", iteration, chat_id)
            raise  # let the router return a 500; caller can retry

        choice = response.choices[0]
        msg = choice.message

        # ── Terminal: model produced its final answer ──────────────────────────
        if not msg.tool_calls:
            logger.debug("agent done after %d iteration(s) for chat %s", iteration + 1, chat_id)
            return msg.content or ""

        # ── Tool calls: persist assistant turn, execute tools, loop ───────────

        # Build the tool_calls payload in the exact format the API expects back.
        tool_calls_payload = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]

        # Persist assistant message (with tool_calls) — must be stored BEFORE
        # the tool results so history reconstruction stays in the right order.
        await database.execute(
            """
            INSERT INTO messages (id, chat_id, role, content, tool_calls, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                chat_id,
                "assistant",
                msg.content,
                json.dumps(tool_calls_payload),
                _now(),
            ),
        )

        # Mirror to in-memory list for the next API call
        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": tool_calls_payload,
            }
        )

        # Execute every tool call in this batch and persist results
        for tc in msg.tool_calls:
            result, files_read = dispatch(
                name=tc.function.name,
                arguments_json=tc.function.arguments,
                kb=kb,
                files_read=files_read,
                max_files=settings.agent_max_kb_files,
            )
            logger.debug("tool %r → %d chars", tc.function.name, len(result))

            await database.execute(
                """
                INSERT INTO messages (id, chat_id, role, content, tool_call_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), chat_id, "tool", result, tc.id, _now()),
            )

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Exhausted all iterations without a clean stop
    logger.warning(
        "agent exhausted %d iterations for chat %s",
        settings.agent_max_iterations,
        chat_id,
    )
    return (
        "I was unable to produce a final answer within the allowed number of steps. "
        "Please try rephrasing your question."
    )
