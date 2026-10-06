"""Application configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import Any, List, Optional, Union

from pydantic import field_validator, model_validator
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

    # Tool Settings (Phase 2 & Phase 7 Milestone 3)
    tool_http_timeout: float = 10.0
    tool_http_max_size_bytes: int = 100_000
    allowed_http_domains: Union[List[str], str] = [
        "httpbin.org",
        "api.github.com",
    ]
    max_tool_calls_per_run: int = 10
    tool_failure_disable_threshold: int = 3
    tool_max_input_size_bytes: int = 65_536
    tool_max_output_size_bytes: int = 1_000_000

    @field_validator("max_tool_calls_per_run", mode="before")
    @classmethod
    def validate_max_tool_calls_per_run(cls, v: Any) -> int:
        val = int(v)
        if val <= 0:
            raise ValueError("MAX_TOOL_CALLS_PER_RUN must be positive.")
        if val > 1000:
            raise ValueError("MAX_TOOL_CALLS_PER_RUN exceeds maximum allowed ceiling (1000).")
        return val

    @field_validator("tool_failure_disable_threshold", mode="before")
    @classmethod
    def validate_tool_failure_disable_threshold(cls, v: Any) -> int:
        val = int(v)
        if val <= 0:
            raise ValueError("TOOL_FAILURE_DISABLE_THRESHOLD must be positive.")
        if val > 100:
            raise ValueError(
                "TOOL_FAILURE_DISABLE_THRESHOLD exceeds maximum allowed ceiling (100)."
            )
        return val

    @field_validator("tool_max_input_size_bytes", "tool_max_output_size_bytes", mode="before")
    @classmethod
    def validate_tool_size_limits(cls, v: Any) -> int:
        val = int(v)
        if val <= 0:
            raise ValueError("Tool size limits must be positive.")
        return val

    @field_validator("allowed_http_domains", mode="before")
    @classmethod
    def assemble_allowed_http_domains(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse comma-separated string into a list of allowed HTTP domains."""
        if isinstance(v, str):
            return [domain.strip().lower() for domain in v.split(",") if domain.strip()]
        return [d.lower() for d in v]

    # Model Context Protocol (MCP) Settings (Phase 4)
    mcp_enabled: bool = False
    mcp_local_server_enabled: bool = True
    mcp_local_server_name: str = "local"
    mcp_allowed_tools: Union[List[str], str] = [
        "mcp.local.calculator",
        "mcp.local.text_stats",
    ]

    @field_validator("mcp_allowed_tools", mode="before")
    @classmethod
    def assemble_mcp_allowed_tools(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse comma-separated string into a list of allowed MCP tools."""
        if isinstance(v, str):
            return [tool.strip().lower() for tool in v.split(",") if tool.strip()]
        return [t.lower() for t in v]

    # RAG & Knowledge System Settings (Phase 5)
    rag_enabled: bool = True
    rag_persist_directory: Optional[str] = "./data/chroma"
    rag_collection_name: str = "knowledge_base"
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 150
    rag_default_top_k: int = 5
    rag_hybrid_alpha: float = 0.6
    rag_embedding_provider: str = "local"
    rag_max_file_size_bytes: int = 10_000_000

    @field_validator("rag_embedding_provider", mode="before")
    @classmethod
    def validate_rag_embedding_provider(cls, v: str) -> str:
        """Ensure RAG embedding provider is supported."""
        allowed = {"local", "mock", "openai"}
        normalized = v.strip().lower()
        if normalized not in allowed:
            raise ValueError(
                f"Unsupported RAG_EMBEDDING_PROVIDER '{v}'. Allowed providers: {sorted(allowed)}"
            )
        return normalized

    # Memory Settings (Phase 6)
    memory_enabled: bool = True
    memory_persist_directory: Optional[str] = "./data/chroma_memory"
    memory_collection_name: str = "agent_memory"
    memory_default_top_k: int = 3
    memory_similarity_threshold: float = 0.0
    memory_consolidation_threshold: float = 0.90
    memory_default_ttl_seconds: Optional[int] = None

    # Human-in-the-Loop & Checkpointing Settings (Phase 6)
    hitl_enabled: bool = True
    hitl_confidence_threshold: float = 0.65
    hitl_max_specialist_retries: int = 2
    checkpoint_backend: str = "postgres"  # "postgres" (durable runtime) or "memory" (tests)

    # Observability & Tracing Settings (Phase 7 Milestone 1)
    telemetry_enabled: bool = True
    telemetry_service_name: str = "multi-agent-orchestrator"
    telemetry_exporter: str = "memory"  # memory | console | none
    telemetry_record_payloads: bool = False
    telemetry_max_in_memory_spans: int = 500

    @field_validator("telemetry_exporter", mode="before")
    @classmethod
    def validate_telemetry_exporter(cls, v: str) -> str:
        """Ensure telemetry exporter is one of the supported modes."""
        allowed = {"memory", "console", "none"}
        normalized = v.strip().lower()
        if normalized not in allowed:
            raise ValueError(
                f"Unsupported TELEMETRY_EXPORTER '{v}'. Allowed exporters: {sorted(allowed)}"
            )
        return normalized

    @model_validator(mode="after")
    def validate_production_security(self) -> "Settings":
        """Enforce strict security boundaries and password safety in production environments."""
        env = self.app_env.strip().lower()
        if env == "production":
            # Force telemetry payload recording off to prevent data leakage
            if self.telemetry_record_payloads:
                self.telemetry_record_payloads = False

            # Disallow insecure default or placeholder passwords in production
            insecure_defaults = {
                "changeme_in_production",
                "postgres_dev_password",
                "password",
                "",
            }
            if (
                self.checkpoint_backend == "postgres"
                and self.postgres_password.strip().lower() in insecure_defaults
            ):
                raise ValueError(
                    "Insecure default POSTGRES_PASSWORD detected in production environment. "
                    "A secure, non-default password must be configured."
                )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of application settings."""
    return Settings()
