"""
FastAPI application entry point.

Lifespan
--------
FastAPI's lifespan context manager (introduced in Starlette 0.20) is the
recommended replacement for the deprecated @app.on_event("startup/shutdown")
hooks.  It guarantees symmetric teardown even if startup raises, and it
integrates cleanly with pytest fixtures that manage the lifespan manually.

Router registration
-------------------
Chat and message routers are imported and mounted here.  They are in separate
files to keep each module focused on one resource type.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app import database
from app.config import settings
from app.models import HealthResponse, ReadyResponse
from app.routers import chats, messages


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    await database.connect(settings.db_path)
    yield
    await database.disconnect()


app = FastAPI(
    title="QnA Agent API",
    description=(
        "Chat-based QnA system using OpenAI tool/function calling "
        "and a local plain-text knowledge base (RAG without specialised libraries)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(chats.router)
app.include_router(messages.router)


# ── Ops endpoints ──────────────────────────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["ops"],
    summary="Liveness check",
    description="Returns 200 as long as the process is running. Used by K8s liveness probes.",
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["ops"],
    summary="Readiness check",
    description=(
        "Verifies that the service can handle requests (DB reachable). "
        "Returns 503 when degraded. Used by K8s readiness probes."
    ),
)
async def ready() -> ReadyResponse | JSONResponse:
    try:
        await database.fetch_one("SELECT 1 AS ping")
        return ReadyResponse(status="ok", database="ok")
    except Exception:
        # 503 tells the load balancer to stop routing traffic here until the
        # DB recovers — the process stays alive (unlike a liveness failure).
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "database": "error"},
        )
