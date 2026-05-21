# QnA Agent API

A chat-based QnA service that answers questions using a local knowledge base.
The agent uses OpenAI tool/function calling to navigate and retrieve relevant files — no
vector databases, no LangChain, no ORM. Pure OpenAI SDK + raw SQLite.

## Architecture

```
User → POST /chats/{id}/messages
            │
            ▼
       Agent Loop
            ├─► list_knowledge_files()   ← scans ./knowledge, returns filenames + first lines
            │         model picks relevant files by name
            ├─► read_knowledge_file(name) ← returns full content (max 2 files per response)
            └─► Final answer with grounded context
```

**Stack**: FastAPI · aiosqlite (no ORM) · OpenAI SDK · SSE for real-time updates · uv

---

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Install dependencies

```bash
cd qna-agent
uv sync
```

### Configure environment

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and (optionally) OPENAI_API_URL / OPENAI_MODEL
```

### Run

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Swagger UI: http://localhost:8000/docs

### Docker

```bash
docker build -t qna-agent .
docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data qna-agent
```

### Tests

```bash
uv run pytest -v
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | **required** | API key (OpenAI / OpenRouter / Ollama) |
| `OPENAI_API_URL` | `https://api.openai.com/v1` | Base URL — override for OpenRouter or Ollama |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name |
| `KNOWLEDGE_DIR` | `./knowledge` | Directory of `.txt` knowledge base files |
| `DB_PATH` | `./data/qna.db` | SQLite database file path |
| `AGENT_MAX_ITERATIONS` | `10` | Max tool-call iterations per agent run |
| `AGENT_MAX_KB_FILES` | `2` | Max KB files readable per response (context budget) |

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/chats` | Create a chat session |
| `GET` | `/chats` | List all chats |
| `GET` | `/chats/{id}` | Get chat details |
| `DELETE` | `/chats/{id}` | Delete chat and its messages |
| `GET` | `/chats/{id}/messages` | Get full message history |
| `POST` | `/chats/{id}/messages` | Post a user message; returns AI response |
| `GET` | `/chats/{id}/stream` | SSE stream — receive new messages as they arrive |
| `GET` | `/health` | Liveness check |
| `GET` | `/ready` | Readiness check |

### Quick example

```bash
# 1. Create a session
curl -X POST http://localhost:8000/chats \
  -H "Content-Type: application/json" \
  -d '{"title": "My first chat"}'

# 2. Ask a question
curl -X POST http://localhost:8000/chats/{chat_id}/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "How does async/await work in Python?"}'

# 3. Stream updates (SSE)
curl -N http://localhost:8000/chats/{chat_id}/stream
```

---

## Knowledge Base

Drop `.txt` files into `./knowledge/`. Each file should cover a single topic.
**Filename quality matters** — the agent reads filenames (plus the first line of each file)
to decide which files are relevant before fetching content.

Good: `python_async_basics.txt`, `company_refund_policy.txt`
Bad:  `file001.txt`, `doc_v3_final.txt`

---

## Design Decisions

### RAG without a vector database
Total KB size can exceed the model's context window, and the guidelines cap in-context files
at 2 simultaneously. Instead of embeddings, the agent receives a compact index
(filename + first line per file) via `list_knowledge_files`, then fetches up to 2 files with
`read_knowledge_file`. The model uses filenames as a semantic index.

### SSE over WebSockets
Server-Sent Events are HTTP-native, require no extra library, and suit the unidirectional
(server → client) push pattern here. WebSockets would add complexity without benefit.

### Raw aiosqlite over SQLAlchemy
Required by the assessment. Keeps the dependency footprint small and makes the SQL
visible and auditable. A thin `database.py` helper wraps connection management.

### uv over Poetry
Faster resolution, cleaner lockfiles, first-class support for `dependency-groups` (dev deps).

---

## Production TODOs

- **Secrets**: Use K8s Secrets / external secret manager (Vault, AWS SSM) — never ConfigMaps for API keys.
- **Health probes**: `/health` (liveness) and `/ready` (readiness, checks DB connection) for K8s.
- **Observability**: Structured JSON logging + Prometheus metrics endpoint (`/metrics`).
- **Persistent storage**: Mount a PersistentVolumeClaim for `./data` in K8s; SQLite write concurrency is limited — migrate to Postgres for multi-replica deployments.
- **TLS**: Terminate at ingress (nginx/Traefik); app listens on plain HTTP internally.
- **Performance**: Add keyword-based pre-filtering (grep index) on KB filenames to reduce token usage on large file sets; consider async file I/O for the KB scan.
