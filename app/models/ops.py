from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str    # "ok" | "degraded"
    database: str  # "ok" | "error"
