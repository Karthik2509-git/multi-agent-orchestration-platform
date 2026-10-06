"""Domain schemas and models for Developer Execution Fork / Replay."""

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

ALLOWED_SUPERVISOR_ROUTES = {"research", "data", "code", "final"}
ALLOWED_LLM_PROVIDERS = {"mock", "openrouter", "gemini", "groq", "openai"}
MAX_TASK_CHARS = 5_000
MAX_MOCK_PAYLOAD_BYTES = 65_536


class ModelOverride(BaseModel):
    """Explicit validated LLM model and provider override for execution replay."""

    provider: str = Field(description="Approved LLM provider name")
    model: Optional[str] = Field(default=None, description="Approved model identifier")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        prov = v.strip().lower()
        if prov not in ALLOWED_LLM_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM provider '{v}'. "
                f"Allowed providers: {sorted(ALLOWED_LLM_PROVIDERS)}"
            )
        return prov

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        trimmed = v.strip()
        if not trimmed:
            return None
        if len(trimmed) > 64:
            raise ValueError("Model identifier exceeds maximum length (64 characters).")
        if not re.match(r"^[a-zA-Z0-9_\-\.\/:]+$", trimmed):
            raise ValueError("Model identifier contains invalid characters.")
        return trimmed


class MockToolResult(BaseModel):
    """Explicit structured mock outcome for a specified tool call during replay."""

    success: bool = True
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_category: Optional[str] = None

    @field_validator("data")
    @classmethod
    def validate_data_payload(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if v is None:
            return None
        try:
            serialized = json.dumps(v)
        except (TypeError, ValueError) as err:
            raise ValueError(f"Mock tool data must be JSON serializable: {err}") from err

        if len(serialized.encode("utf-8")) > MAX_MOCK_PAYLOAD_BYTES:
            raise ValueError(
                f"Mock tool data exceeds maximum size limit ({MAX_MOCK_PAYLOAD_BYTES} bytes)."
            )
        return v


class ReplayModification(BaseModel):
    """Explicitly typed, validated modifications permitted for an execution fork."""

    task_override: Optional[str] = Field(
        default=None,
        description="Replacement task description for the execution fork",
    )
    model_override: Optional[ModelOverride] = Field(
        default=None,
        description="Validated provider and model configuration override",
    )
    supervisor_route_override: Optional[str] = Field(
        default=None,
        description="Explicit initial specialist route from the supervisor",
    )
    mock_tool_results: Optional[Dict[str, MockToolResult]] = Field(
        default=None,
        description="Deterministic mock tool results mapped by tool name",
    )
    scope_id_override: Optional[str] = Field(
        default=None,
        description="Memory scope identifier override",
    )

    @field_validator("task_override")
    @classmethod
    def validate_task_override(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("task_override cannot be empty when provided.")
        if len(trimmed) > MAX_TASK_CHARS:
            raise ValueError(
                f"task_override exceeds maximum allowed length ({MAX_TASK_CHARS} characters)."
            )
        # Prevent control characters or null bytes
        if any(ord(c) < 32 and c not in ("\n", "\r", "\t") for c in trimmed):
            raise ValueError("task_override contains illegal control characters.")
        return trimmed

    @field_validator("supervisor_route_override")
    @classmethod
    def validate_route_override(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        route = v.strip().lower()
        if route not in ALLOWED_SUPERVISOR_ROUTES:
            raise ValueError(
                f"Invalid supervisor route override '{v}'. "
                f"Allowed routes: {sorted(ALLOWED_SUPERVISOR_ROUTES)}"
            )
        return route

    @field_validator("scope_id_override")
    @classmethod
    def validate_scope_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        scope = v.strip()
        if not scope:
            raise ValueError("scope_id_override cannot be empty when provided.")
        if len(scope) > 64:
            raise ValueError("scope_id_override exceeds maximum length of 64 characters.")
        if not re.match(r"^[a-zA-Z0-9_\-]+$", scope):
            raise ValueError(
                "scope_id_override contains invalid characters. "
                "Must be alphanumeric with underscores and hyphens."
            )
        return scope

    @field_validator("mock_tool_results")
    @classmethod
    def validate_mock_tool_names(
        cls, v: Optional[Dict[str, MockToolResult]]
    ) -> Optional[Dict[str, MockToolResult]]:
        if v is None:
            return None
        cleaned: Dict[str, MockToolResult] = {}
        for tool_name, result in v.items():
            name = tool_name.strip()
            if not re.match(r"^[a-zA-Z0-9_\.\-]+$", name):
                raise ValueError(f"Invalid tool name format for mock tool: '{tool_name}'")
            cleaned[name] = result
        return cleaned


class ReplayRequest(BaseModel):
    """Developer request payload initiating an execution fork from an existing thread."""

    source_thread_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Thread ID of the source execution to fork",
    )
    modifications: Optional[ReplayModification] = Field(
        default=None,
        description="Optional explicitly validated execution modifications",
    )

    @field_validator("source_thread_id")
    @classmethod
    def validate_source_thread_id(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("source_thread_id cannot be empty.")
        if len(trimmed) > 128:
            raise ValueError("source_thread_id exceeds maximum length (128 characters).")
        return trimmed


class ReplayResult(BaseModel):
    """Structured response describing the newly created execution fork."""

    success: bool = Field(description="True if the replay execution succeeded or completed")
    source_thread_id: str = Field(description="Identifier of the immutable parent execution")
    replay_thread_id: Optional[str] = Field(
        default=None,
        description="Unique thread identifier of the newly created execution fork",
    )
    status: str = Field(
        default="completed",
        description="Status of the replay fork ('completed', 'interrupted', or 'error')",
    )
    applied_modifications: Dict[str, Any] = Field(
        default_factory=dict,
        description="Record of explicit modifications applied to this fork",
    )
    answer: Optional[str] = Field(
        default=None,
        description="Synthesized answer produced by the replayed workflow",
    )
    agents_used: List[str] = Field(
        default_factory=list,
        description="Specialized agents utilized during the replayed execution",
    )
    execution_time_seconds: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Total elapsed execution time for the replay",
    )
    is_interrupted: bool = Field(
        default=False,
        description="True if the execution paused awaiting human review",
    )
    pending_approval: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Approval details if the replay paused on an interrupt",
    )
    error: Optional[str] = Field(
        default=None,
        description="Structured error message if replay creation or execution failed",
    )
    error_category: Optional[str] = Field(
        default=None,
        description="Categorized error identifier (e.g. source_not_found, validation_error)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Non-sensitive operational metadata describing the execution fork",
    )
