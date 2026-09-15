"""Application configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import List, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application Settings
    app_name: str = "Multi-Agent Orchestration Platform"
    app_env: str = "development"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    # Security & CORS Settings
    cors_origins: Union[List[str], str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse comma-separated string into a list of allowed origins."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    # Planned Infrastructure Settings (Reserved for Future Phases)
    postgres_server: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "orchestrator_db"
    postgres_user: str = "postgres"
    postgres_password: str = "changeme_in_production"
    database_url: str = (
        "postgresql+asyncpg://postgres:changeme_in_production@localhost:5432/orchestrator_db"
    )

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_url: str = "redis://localhost:6379/0"

    # LLM API Keys (Reserved for Future Phases)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of application settings."""
    return Settings()
