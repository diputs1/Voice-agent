from __future__ import annotations

from langsmith import traceable

from app.agents.logic.answer_grounding import handoff_state
from app.agents.schemas import SafariState


@traceable(name="graph.escalation")
async def escalation_node(state: SafariState) -> SafariState:
    return handoff_state(state, state.get("handoff_reason") or "stale_or_missing_time_sensitive_data")
