from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings


@dataclass(frozen=True)
class AgentConfig:
    low_confidence_threshold: float
    max_iterations: int
    timeout_seconds: float

    @classmethod
    def from_settings(cls, settings: Settings) -> "AgentConfig":
        return cls(
            low_confidence_threshold=settings.low_confidence_threshold,
            max_iterations=settings.full_agent_max_iterations,
            timeout_seconds=settings.full_agent_timeout_seconds,
        )
