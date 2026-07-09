from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class SafariState(TypedDict, total=False):
    messages: list[dict[str, str]]
    transcript: str
    thread_history: list[dict[str, Any]]
    search_query: str
    rewrite_source: str
    query_retry_count: int
    retry_reason: str | None
    intent: str
    primary_intent: str
    intent_entities: list[str]
    supervisor_confidence: float
    suggested_route: str
    retrieved_context: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]
    confidence: float
    classifier_confidence: float
    classifier_source: str
    handoff_required: bool
    handoff_reason: str | None
    recommended_action: str | None
    route: str


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
