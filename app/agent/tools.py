"""
OpenAI tool schemas and dispatch.

TOOLS       — the list passed to every chat.completions.create() call.
dispatch()  — executes one tool call and returns (result_str, new_files_read).

Design rule: dispatch() never raises.  Errors are returned as descriptive
strings so the LLM can read them and decide how to proceed (e.g. try a
different file, or tell the user no relevant information was found).
"""

import json
import logging
from typing import Any

from app.kb.store import KBAccessError, KBFileNotFoundError, KBStore

logger = logging.getLogger(__name__)

# ── OpenAI tool schemas ────────────────────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_knowledge_files",
            "description": (
                "List all available knowledge base files with a one-line summary "
                "of each. Always call this first to discover which files exist "
                "before deciding which ones to read."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_knowledge_file",
            "description": (
                "Read the full contents of a specific knowledge base file. "
                "Use the exact filename returned by list_knowledge_files. "
                "You may only read a limited number of files per response — "
                "choose the most relevant ones based on their summaries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": (
                            "Exact filename to read, e.g. 'python_async_basics.txt'. "
                            "Must be one of the filenames returned by list_knowledge_files."
                        ),
                    }
                },
                "required": ["filename"],
            },
        },
    },
]


# ── Dispatch ───────────────────────────────────────────────────────────────────


def dispatch(
    name: str,
    arguments_json: str,
    kb: KBStore,
    files_read: int,
    max_files: int,
) -> tuple[str, int]:
    """Execute one tool call; return (result_string, updated_files_read).

    Parameters
    ----------
    name            Tool function name.
    arguments_json  Raw JSON string from the model's tool_call.function.arguments.
    kb              KBStore instance for this request.
    files_read      How many files have already been read this turn.
    max_files       Hard cap on files per turn (from settings).

    Returns
    -------
    (result, new_files_read) — result is always a non-empty string.
    """
    try:
        args: dict[str, Any] = json.loads(arguments_json) if arguments_json.strip() else {}
    except json.JSONDecodeError as exc:
        return f"Error: malformed tool arguments — {exc}", files_read

    if name == "list_knowledge_files":
        return _list_files(kb), files_read

    if name == "read_knowledge_file":
        return _read_file(args, kb, files_read, max_files)

    return f"Error: unknown tool {name!r}. Available: list_knowledge_files, read_knowledge_file.", files_read


def _list_files(kb: KBStore) -> str:
    entries = kb.list_files()
    if not entries:
        return "The knowledge base is empty — no files are available."
    lines = [f"- {e['filename']}: {e['summary']}" for e in entries]
    return "\n".join(lines)


def _read_file(
    args: dict[str, Any], kb: KBStore, files_read: int, max_files: int
) -> tuple[str, int]:
    filename = args.get("filename", "").strip()
    if not filename:
        return "Error: 'filename' argument is required.", files_read

    if files_read >= max_files:
        return (
            f"Error: you have already read {files_read} file(s) this turn "
            f"(maximum is {max_files}). "
            "Use list_knowledge_files to identify the single most relevant file.",
            files_read,
        )

    try:
        content = kb.read_file(filename)
        logger.debug("KB read: %r  (%d chars)", filename, len(content))
        return content, files_read + 1
    except KBFileNotFoundError as exc:
        return f"Error: {exc}", files_read
    except KBAccessError as exc:
        return f"Error: {exc}", files_read
