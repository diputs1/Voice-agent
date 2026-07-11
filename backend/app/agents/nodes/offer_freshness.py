from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from langsmith import traceable

from app.agents.logic.answer_grounding import HANDOFF_ACTIONS, parse_datetime
from app.agents.logic.intent import TIME_SENSITIVE_CATEGORIES
from app.agents.nodes.retry_query import MAX_RETRIEVAL_RETRIES
from app.agents.schemas import SafariState
from app.agents.state import AgentUpdate


@traceable(name="graph.offer_freshness")
async def offer_freshness_node(
    state: SafariState,
    low_confidence_threshold: float,
) -> AgentUpdate:
    now = datetime.now(UTC)
    handoff_required = False
    handoff_reason: str | None = None
    if state.get("intent") == "time_sensitive":
        handoff_required = True
        handoff_reason = "stale_or_missing_time_sensitive_data"
        for item in state.get("retrieved_context", []):
            if item.get("category") not in TIME_SENSITIVE_CATEGORIES:
                continue
            valid_until = parse_datetime(item.get("valid_until"))
            if valid_until and valid_until > now:
                handoff_required = False
                handoff_reason = None
                break
    if state.get("confidence", 0.0) < low_confidence_threshold:
        handoff_required = True
        handoff_reason = "low_confidence"
    return {
        "handoff_required": handoff_required,
        "handoff_reason": handoff_reason,
        "recommended_action": HANDOFF_ACTIONS.get(handoff_reason),
    }


def route_after_freshness(
    state: SafariState,
    low_confidence_threshold: float,
) -> Literal["retry", "escalate", "answer"]:
    if should_retry_retrieval(state, low_confidence_threshold):
        return "retry"
    return "escalate" if state.get("handoff_required") else "answer"


def should_retry_retrieval(state: SafariState, low_confidence_threshold: float) -> bool:
    if state.get("route") != "safari_knowledge":
        return False
    if state.get("intent") == "time_sensitive":
        return False
    if state.get("handoff_reason") != "low_confidence":
        return False
    if float(state.get("confidence") or 0.0) >= low_confidence_threshold:
        return False
    return int(state.get("query_retry_count", 0)) < MAX_RETRIEVAL_RETRIES
