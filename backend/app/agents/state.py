from __future__ import annotations

from typing import Any, TypedDict


class RetrievalState(TypedDict, total=False):
    search_query: str
    rewrite_source: str
    retrieved_context: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    confidence: float
    retrieval_debug: dict[str, Any]
    freshness_status: str | None


class CrawlState(TypedDict, total=False):
    site_id: str
    crawl_job_id: str
    crawl_version: str | None


class VoiceState(TypedDict, total=False):
    answer: str
    grounded: bool
    evaluator_score: float | None
    handoff_required: bool
    handoff_reason: str | None
    recommended_action: str | None


class AgentState(RetrievalState, CrawlState, VoiceState, total=False):
    messages: list[dict[str, str]]
    transcript: str
    thread_history: list[dict[str, Any]]
    query_retry_count: int
    retry_reason: str | None
    intent: str
    primary_intent: str
    intent_entities: list[str]
    supervisor_confidence: float
    suggested_route: str
    classifier_confidence: float
    classifier_source: str
    route: str
    agent_runtime_mode: str
    agent_iteration_count: int
    tool_call_count: int


AgentUpdate = dict[str, Any]
SafariState = AgentState
