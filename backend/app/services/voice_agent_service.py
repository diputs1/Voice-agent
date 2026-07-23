from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from langsmith import traceable

from app.agents.logic.query_rewrite import heuristic_rewrite, looks_like_follow_up
from app.api.schemas import (
    VoiceAgentKnowledgeHit,
    VoiceAgentKnowledgeRequest,
    VoiceAgentKnowledgeResponse,
    VoiceAgentTraceMetadata,
)
from app.core.cache import QACacheKey, TTLQACache, normalize_cache_query
from app.core.voice_memory import VoiceConversationMemory
from app.knowledge.kb import KnowledgeHit, Retriever

logger = logging.getLogger(__name__)
VOICE_TOOL_CACHE_VERSION = "voice-tool-context-v2"
VOICE_TOOL_MAX_CONTEXT_HITS = 3
VOICE_TOOL_MAX_CONTENT_CHARS = 900
VOICE_TOOL_MAX_TITLE_CHARS = 120
VOICE_TOOL_MAX_SECTION_CHARS = 160


async def search_voice_agent_knowledge(
    *,
    kb: Retriever,
    payload: VoiceAgentKnowledgeRequest,
    raw_payload: dict[str, Any] | None = None,
    log_raw_payload: bool = False,
    retrieval_cache: TTLQACache | None = None,
    conversation_memory: VoiceConversationMemory | None = None,
) -> VoiceAgentKnowledgeResponse:
    received_at = datetime.now(UTC)
    started = time.perf_counter()
    correlation_id = _correlation_id(payload)
    turn_id = _turn_id(payload, correlation_id)
    raw_payload_keys = sorted(raw_payload.keys()) if raw_payload else []
    resolved_query, rewrite_source = await _resolve_voice_query(payload, conversation_memory)
    effective_payload = _with_query_and_limit(payload, resolved_query)

    logger.info(
        "elevenlabs_knowledge_tool_received",
        extra={
            "correlation_id": correlation_id,
            "turn_id": turn_id,
            "conversation_id": payload.conversation_id,
            "site_id": payload.site_id,
            "limit": payload.limit,
            "query_length": len(payload.query),
            "resolved_query_length": len(resolved_query),
            "rewrite_source": rewrite_source,
            "raw_payload_keys": raw_payload_keys,
            "raw_payload": raw_payload if log_raw_payload else None,
        },
    )

    cache_key = _voice_tool_cache_key(effective_payload)
    cached = await retrieval_cache.get(cache_key) if retrieval_cache else None
    cache_hit = cached is not None
    if cached:
        context = _cached_context(cached)
    else:
        hits = await _trace_voice_agent_retrieval(
            kb=kb,
            payload=effective_payload,
            correlation_id=correlation_id,
            turn_id=turn_id,
            raw_payload_keys=raw_payload_keys,
        )
        context = [_hit_response(hit) for hit in hits]
        if retrieval_cache:
            await retrieval_cache.set(cache_key, _voice_tool_cache_payload(context))

    await _remember_voice_query(
        original_payload=payload,
        effective_payload=effective_payload,
        conversation_memory=conversation_memory,
    )

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    completed_at = datetime.now(UTC)
    top_score = max((hit.score for hit in context), default=None)
    context_char_count = sum(len(hit.content) for hit in context)
    context_byte_count = sum(len(hit.content.encode("utf-8")) for hit in context)
    logger.info(
        "elevenlabs_knowledge_tool_completed",
        extra={
            "correlation_id": correlation_id,
            "turn_id": turn_id,
            "conversation_id": payload.conversation_id,
            "site_id": payload.site_id,
            "hit_count": len(context),
            "top_score": top_score,
            "latency_ms": latency_ms,
            "context_byte_count": context_byte_count,
            "context_char_count": context_char_count,
            "cache_hit": cache_hit,
            "rewrite_source": rewrite_source,
        },
    )

    trace = VoiceAgentTraceMetadata(
        correlation_id=correlation_id,
        turn_id=turn_id,
        received_at=received_at.isoformat(),
        completed_at=completed_at.isoformat(),
        latency_ms=latency_ms,
        hit_count=len(context),
        top_score=top_score,
        context_byte_count=context_byte_count,
        context_char_count=context_char_count,
        cache_hit=cache_hit,
        rewrite_source=rewrite_source,
        resolved_query=resolved_query,
    )
    return VoiceAgentKnowledgeResponse(
        query=payload.query,
        site_id=payload.site_id,
        conversation_id=payload.conversation_id,
        correlation_id=correlation_id,
        turn_id=turn_id,
        context=context,
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
        "context_byte_count": sum(len(hit.content.encode("utf-8")) for hit in hits),
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
        content=_compact_text(hit.content, VOICE_TOOL_MAX_CONTENT_CHARS),
        source_url=hit.source_url,
        title=_compact_text(hit.title, VOICE_TOOL_MAX_TITLE_CHARS),
        section=_compact_text(hit.section, VOICE_TOOL_MAX_SECTION_CHARS),
        category=hit.category,
        score=hit.score,
        language=hit.language,
        crawled_at=hit.crawled_at,
        valid_until=hit.valid_until,
        metadata=None,
    )


async def _resolve_voice_query(
    payload: VoiceAgentKnowledgeRequest,
    conversation_memory: VoiceConversationMemory | None,
) -> tuple[str, str]:
    if not conversation_memory or not payload.conversation_id:
        return payload.query, "original"
    if not looks_like_follow_up(payload.query):
        return payload.query, "original"
    history = await conversation_memory.history(payload.conversation_id)
    if not history:
        return payload.query, "original"
    rewritten = heuristic_rewrite(payload.query, history).strip()[:1000]
    if rewritten and rewritten != payload.query:
        return rewritten, "voice_memory_heuristic"
    return payload.query, "original"


async def _remember_voice_query(
    *,
    original_payload: VoiceAgentKnowledgeRequest,
    effective_payload: VoiceAgentKnowledgeRequest,
    conversation_memory: VoiceConversationMemory | None,
) -> None:
    if not conversation_memory:
        return
    await conversation_memory.append(
        conversation_id=original_payload.conversation_id,
        transcript=original_payload.query,
        search_query=effective_payload.query,
        site_id=original_payload.site_id,
    )


def _with_query_and_limit(payload: VoiceAgentKnowledgeRequest, query: str) -> VoiceAgentKnowledgeRequest:
    updates: dict[str, object] = {}
    if query != payload.query:
        updates["query"] = query
    if payload.limit > VOICE_TOOL_MAX_CONTEXT_HITS:
        updates["limit"] = VOICE_TOOL_MAX_CONTEXT_HITS
    return payload.model_copy(update=updates) if updates else payload


def _compact_text(value: str, max_chars: int) -> str:
    normalized = " ".join(str(value).split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "..."


def _voice_tool_cache_key(payload: VoiceAgentKnowledgeRequest) -> QACacheKey:
    site_id = payload.site_id or "default"
    query = normalize_cache_query(payload.query)
    return QACacheKey(query=f"{site_id}:{query}", doc_set_hash=VOICE_TOOL_CACHE_VERSION)


def _voice_tool_cache_payload(context: list[VoiceAgentKnowledgeHit]) -> dict[str, object]:
    return {"context": [hit.model_dump(mode="json") for hit in context]}


def _cached_context(cached: dict[str, object]) -> list[VoiceAgentKnowledgeHit]:
    items = cached.get("context") or []
    if not isinstance(items, list):
        return []
    return [VoiceAgentKnowledgeHit.model_validate(item) for item in items if isinstance(item, dict)]


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
