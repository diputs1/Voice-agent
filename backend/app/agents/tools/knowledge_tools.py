from __future__ import annotations

from app.agents.logic.answer_grounding import citation, hit_payload
from app.knowledge.kb import Retriever

TOOL_WHITELIST = frozenset(
    {
        "rewrite_query",
        "search_kb",
        "verify_evidence",
        "finish_answer",
        "handoff",
    }
)


async def search_kb(
    *,
    kb: Retriever,
    query: str,
    site_id: str | None = None,
    limit: int = 5,
) -> dict:
    hits = await kb.search(query, limit=limit, site_id=site_id)
    return {
        "retrieved_context": [hit_payload(hit) for hit in hits],
        "citations": [citation(hit) for hit in hits],
        "confidence": max((hit.score for hit in hits), default=0.0),
        "retrieval_debug": {
            "site_id": site_id,
            "candidate_count": len(hits),
            "retriever": "hybrid_pgvector_fts_rrf",
        },
    }
