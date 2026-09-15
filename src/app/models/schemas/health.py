"""Health check response schema."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema representing health check status."""

    status: str = Field(default="healthy", description="Current application status")
    app_name: str = Field(description="Name of the application")
    version: str = Field(description="Current release version")
    environment: str = Field(description="Runtime environment (e.g. development, production)")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the health check invocation",
    )
