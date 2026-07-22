from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from app.agents.graphs import WebsiteAgentGraph
from app.core.cache import QACacheKey, TTLQACache, normalize_cache_query
from app.knowledge.kb import ChatKnowledgeStore
from app.api.schemas import ChatRequest


class ChatService:
    def __init__(
        self,
        *,
        agent_graph: WebsiteAgentGraph,
        kb: ChatKnowledgeStore,
        agent_semaphore,
        qa_cache: TTLQACache,
    ) -> None:
        self.agent_graph = agent_graph
        self.kb = kb
        self.agent_semaphore = agent_semaphore
        self.qa_cache = qa_cache

    async def stream_chat(self, payload: ChatRequest) -> AsyncIterator[str]:
        thread_id = payload.thread_id or str(uuid.uuid4())
        final_state: dict[str, Any] = {"transcript": payload.transcript, "site_id": payload.site_id}
        try:
            doc_set_hash = await self.kb.doc_set_hash(payload.site_id)
        except TypeError:
            doc_set_hash = await self.kb.doc_set_hash()
        cache_key: QACacheKey | None = None
        cached: dict[str, object] | None = None

        async with self.agent_semaphore:
            async for node_name, _, current_state in self.agent_graph.astream_pre_answer(
                {"transcript": payload.transcript, "site_id": payload.site_id},
                thread_id,
            ):
                final_state = current_state
                yield sse("status", {"thread_id": thread_id, "node": node_name})
                if node_name in {"contextualize_query", "direct_response", "direct_handoff"}:
                    cache_key = qa_cache_key(final_state, doc_set_hash)
                    cached = await self.qa_cache.get(cache_key)
                    if cached:
                        break

            if cached:
                final_state = {**final_state, **cached}
                answer = str(final_state.get("answer", ""))
                yield sse("status", {"thread_id": thread_id, "node": "cache_hit"})
                yield sse("token", {"text": answer})
            else:
                answer_parts: list[str] = []
                yield sse("status", {"thread_id": thread_id, "node": "voice_answer"})
                async for token in self.agent_graph.astream_voice_answer_tokens(final_state):
                    answer_parts.append(token)
                    yield sse("token", {"text": token})

                answer = "".join(answer_parts)
                final_update = self.agent_graph.finalize_streamed_answer(final_state, answer)
                final_state = {**final_state, **final_update}
                final_answer = final_state.get("answer", answer)
                if final_answer != answer:
                    answer = str(final_answer)
                    yield sse("replace", {"text": answer})

                cache_key = cache_key or qa_cache_key(final_state, doc_set_hash)
                await self.qa_cache.set(cache_key, qa_cache_payload(final_state))

            citations = final_state.get("citations", [])
            try:
                await self.kb.save_thread_turn(thread_id, payload.transcript, answer, payload.site_id)
            except TypeError:
                await self.kb.save_thread_turn(thread_id, payload.transcript, answer)
            yield sse(
                "done",
                {
                    "thread_id": thread_id,
                    "site_id": payload.site_id,
                    "graph_version": getattr(self.agent_graph, "graph_version", "website-agent-v1"),
                    "agent_runtime_mode": "website",
                    "answer": answer,
                    "citations": citations,
                    "confidence": final_state.get("confidence", 0.0),
                    "handoff_required": final_state.get("handoff_required", False),
                    "handoff_reason": final_state.get("handoff_reason"),
                    "recommended_action": final_state.get("recommended_action"),
                    "retrieval_debug": final_state.get("retrieval_debug", {}),
                    "grounded": final_state.get("grounded", not final_state.get("handoff_required", False)),
                    "evaluator_score": final_state.get("evaluator_score"),
                    "crawl_version": final_state.get("crawl_version"),
                    "freshness_status": final_state.get("freshness_status"),
                    "agent_iteration_count": final_state.get("agent_iteration_count", 0),
                    "tool_call_count": final_state.get("tool_call_count", 0),
                    "cache_hit": bool(cached),
                },
            )


def qa_cache_key(state: dict[str, object], doc_set_hash: str) -> QACacheKey:
    query = str(state.get("search_query") or state.get("transcript") or "")
    return QACacheKey(query=normalize_cache_query(query), doc_set_hash=doc_set_hash)


def qa_cache_payload(state: dict[str, object]) -> dict[str, object]:
    return {
        "answer": state.get("answer", ""),
        "citations": state.get("citations", []),
        "confidence": state.get("confidence", 0.0),
        "handoff_required": state.get("handoff_required", False),
        "handoff_reason": state.get("handoff_reason"),
        "recommended_action": state.get("recommended_action"),
    }


def sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
