from __future__ import annotations

from langsmith import traceable

from app.agents.logic.answer_grounding import citation, hit_payload
from app.agents.schemas import SafariState
from app.agents.state import AgentUpdate
from app.knowledge.kb import Retriever


@traceable(name="graph.safari_knowledge")
async def safari_knowledge_node(state: SafariState, kb: Retriever) -> AgentUpdate:
    query = state.get("search_query") or state.get("transcript", "")
    try:
        hits = await kb.search(query, limit=5, site_id=state.get("site_id"))
    except TypeError:
        hits = await kb.search(query, limit=5)
    citations = [citation(hit) for hit in hits]
    confidence = max((hit.score for hit in hits), default=0.0)
    return {
        "retrieved_context": [hit_payload(hit) for hit in hits],
        "citations": citations,
        "confidence": float(confidence),
    }
