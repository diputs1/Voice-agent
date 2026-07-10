from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.agents import state as agent_state

AgentState = agent_state.AgentState
SafariState = agent_state.SafariState


class SupervisorDecision(BaseModel):
    primary_intent: Literal[
        "ticket_price",
        "opening_hours",
        "animal_info",
        "show_schedule",
        "time_sensitive",
        "general",
        "small_talk",
        "out_of_scope",
    ]
    is_time_sensitive: bool
    entities: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    suggested_route: Literal["safari_knowledge", "direct_response", "direct_handoff"]

    @property
    def legacy_intent(self) -> str:
        if self.primary_intent in {"small_talk", "out_of_scope"}:
            return self.primary_intent
        if self.is_time_sensitive:
            return "time_sensitive"
        return "stable"
