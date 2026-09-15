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

    # Multi-Provider LLM Settings (Phase 2.5)
    llm_provider: str = "openrouter"
    openrouter_api_key: str = ""
    openrouter_model: str = "openrouter/free"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    google_api_key: str = ""

    @field_validator("llm_provider", mode="before")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        """Ensure provider is one of the supported providers."""
        allowed = {"openrouter", "gemini", "groq", "mock", "openai"}
        normalized = v.strip().lower()
        if normalized not in allowed:
            raise ValueError(
                f"Unsupported LLM_PROVIDER '{v}'. Allowed providers: {sorted(allowed)}"
            )
        return normalized

    # Agent Settings (Phase 2)
    agent_max_iterations: int = 5

    # Tool Settings (Phase 2)
    tool_http_timeout: float = 10.0
    tool_http_max_size_bytes: int = 100_000
    allowed_http_domains: Union[List[str], str] = [
        "httpbin.org",
        "api.github.com",
    ]

    @field_validator("allowed_http_domains", mode="before")
    @classmethod
    def assemble_allowed_http_domains(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse comma-separated string into a list of allowed HTTP domains."""
        if isinstance(v, str):
            return [domain.strip().lower() for domain in v.split(",") if domain.strip()]
        return [d.lower() for d in v]


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of application settings."""
    return Settings()
