from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor to this file's location so .env is always found regardless of CWD.
# config.py lives at <project_root>/app/config.py, so .parent.parent == project root.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    openai_api_key: str
    openai_api_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # Storage
    db_path: str = "./data/qna.db"

    # Knowledge base
    knowledge_dir: str = "./knowledge"

    # Agent behaviour
    agent_max_iterations: int = 10
    agent_max_kb_files: int = 2
    agent_timeout_seconds: float = 60.0  # per-LLM-call HTTP timeout


settings = Settings()
