from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from langsmith import traceable

from app.api.schemas import (
    VoiceAgentKnowledgeHit,
    VoiceAgentKnowledgeRequest,
    VoiceAgentKnowledgeResponse,
    VoiceAgentTraceMetadata,
)
from app.knowledge.kb import KnowledgeHit, Retriever

logger = logging.getLogger(__name__)


async def search_voice_agent_knowledge(
    *,
    kb: Retriever,
    payload: VoiceAgentKnowledgeRequest,
    raw_payload: dict[str, Any] | None = None,
    log_raw_payload: bool = False,
) -> VoiceAgentKnowledgeResponse:
    received_at = datetime.now(UTC)
    started = time.perf_counter()
    correlation_id = _correlation_id(payload)
    turn_id = _turn_id(payload, correlation_id)
    raw_payload_keys = sorted(raw_payload.keys()) if raw_payload else []

    logger.info(
        "elevenlabs_knowledge_tool_received",
        extra={
            "correlation_id": correlation_id,
            "turn_id": turn_id,
            "conversation_id": payload.conversation_id,
            "site_id": payload.site_id,
            "limit": payload.limit,
            "query_length": len(payload.query),
            "raw_payload_keys": raw_payload_keys,
            "raw_payload": raw_payload if log_raw_payload else None,
        },
    )

    hits = await _trace_voice_agent_retrieval(
        kb=kb,
        payload=payload,
        correlation_id=correlation_id,
        turn_id=turn_id,
        raw_payload_keys=raw_payload_keys,
    )

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    completed_at = datetime.now(UTC)
    top_score = max((hit.score for hit in hits), default=None)
    logger.info(
        "elevenlabs_knowledge_tool_completed",
        extra={
            "correlation_id": correlation_id,
            "turn_id": turn_id,
            "conversation_id": payload.conversation_id,
            "site_id": payload.site_id,
            "hit_count": len(hits),
            "top_score": top_score,
            "latency_ms": latency_ms,
        },
    )

    trace = VoiceAgentTraceMetadata(
        correlation_id=correlation_id,
        turn_id=turn_id,
        received_at=received_at.isoformat(),
        completed_at=completed_at.isoformat(),
        latency_ms=latency_ms,
        hit_count=len(hits),
        top_score=top_score,
    )
    return VoiceAgentKnowledgeResponse(
        query=payload.query,
        site_id=payload.site_id,
        conversation_id=payload.conversation_id,
        correlation_id=correlation_id,
        turn_id=turn_id,
        context=[_hit_response(hit) for hit in hits],
        trace=trace,
    )


@traceable(
    name="elevenlabs.search_knowledge_tool",
    run_type="tool",
    process_inputs=lambda inputs: {
        "query": inputs["payload"].query,
        "site_id": inputs["payload"].site_id,
        "limit": inputs["payload"].limit,
        "conversation_id": inputs["payload"].conversation_id,
        "correlation_id": inputs["correlation_id"],
        "turn_id": inputs["turn_id"],
        "raw_payload_keys": inputs["raw_payload_keys"],
    },
    process_outputs=lambda hits: {
        "hit_count": len(hits),
        "top_score": max((hit.score for hit in hits), default=None),
        "top_source_url": hits[0].source_url if hits else None,
    },
)
async def _trace_voice_agent_retrieval(
    *,
    kb: Retriever,
    payload: VoiceAgentKnowledgeRequest,
    correlation_id: str,
    turn_id: str,
    raw_payload_keys: list[str],
) -> list[KnowledgeHit]:
    del correlation_id, turn_id, raw_payload_keys
    return await kb.search(payload.query, limit=payload.limit, site_id=payload.site_id)


def _hit_response(hit: KnowledgeHit) -> VoiceAgentKnowledgeHit:
    return VoiceAgentKnowledgeHit(
        content=hit.content,
        source_url=hit.source_url,
        title=hit.title,
        section=hit.section,
        category=hit.category,
        score=hit.score,
        language=hit.language,
        crawled_at=hit.crawled_at,
        valid_until=hit.valid_until,
        metadata=hit.metadata,
    )


def _correlation_id(payload: VoiceAgentKnowledgeRequest) -> str:
    return (
        payload.correlation_id
        or payload.request_id
        or payload.tool_call_id
        or payload.conversation_id
        or uuid.uuid4().hex
    )


def _turn_id(payload: VoiceAgentKnowledgeRequest, correlation_id: str) -> str:
    return payload.turn_id or payload.tool_call_id or payload.request_id or correlation_id
