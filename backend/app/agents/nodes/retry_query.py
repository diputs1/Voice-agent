from __future__ import annotations

import re

from langsmith import traceable

from app.agents.schemas import SafariState
from app.agents.state import AgentUpdate

MAX_RETRIEVAL_RETRIES = 1


@traceable(name="graph.retry_query")
async def retry_query_node(state: SafariState) -> AgentUpdate:
    return {
        "search_query": expanded_search_query(state),
        "rewrite_source": "retry_expanded",
        "query_retry_count": int(state.get("query_retry_count", 0)) + 1,
        "retry_reason": state.get("handoff_reason") or "low_confidence",
        "handoff_required": False,
        "handoff_reason": None,
        "recommended_action": None,
    }


def expanded_search_query(state: SafariState) -> str:
    parts = [
        state.get("search_query") or state.get("transcript", ""),
        state.get("transcript", ""),
        "Vinpearl Safari Phú Quốc",
    ]
    parts.extend(str(entity) for entity in state.get("intent_entities", []) if str(entity).strip())

    primary_intent = state.get("primary_intent")
    if primary_intent == "opening_hours":
        parts.append("giờ mở cửa lịch hoạt động thời gian vận hành")
    elif primary_intent == "animal_info":
        parts.append("động vật loài thú khu tham quan")
    elif primary_intent == "show_schedule":
        parts.append("show biểu diễn lịch diễn hoạt động")
    elif primary_intent == "ticket_price":
        parts.append("giá vé đặt vé booking")
    else:
        parts.append("trải nghiệm dịch vụ tham quan")

    deduped = []
    seen = set()
    for part in parts:
        cleaned = re.sub(r"\s+", " ", str(part).strip())
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return " ".join(deduped)[:500]
