import asyncio
import json

import pytest

from app.agents.graphs import WebsiteAgentGraph
from app.api.schemas import ChatRequest
from app.core.cache import TTLQACache
from app.core.config import Settings
from app.knowledge.embeddings import EmbeddingProvider
from app.knowledge.kb import InMemoryKnowledgeBase
from app.services.chat_service import ChatService


@pytest.mark.asyncio
async def test_chat_service_streams_end_to_end_graph_answer():
    settings = Settings(openai_api_key=None)
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, settings.openai_embedding_model))
    await kb.ensure_ready()
    service = ChatService(
        agent_graph=WebsiteAgentGraph(kb, settings),
        kb=kb,
        agent_semaphore=asyncio.Semaphore(1),
        qa_cache=TTLQACache(ttl_seconds=60, max_entries=8),
    )

    events = [
        _parse_sse_event(event)
        async for event in service.stream_chat(
            ChatRequest(transcript="Vinpearl Safari mở cửa mấy giờ?")
        )
    ]

    assert [event["event"] for event in events].count("done") == 1
    assert any(event["event"] == "token" for event in events)
    done = events[-1]["data"]
    assert done["handoff_required"] is False
    assert done["citations"]
    assert "09:00" in done["answer"]
    assert "16:00" in done["answer"]
    assert await kb.get_thread(done["thread_id"])


def _parse_sse_event(raw: str) -> dict:
    lines = raw.strip().splitlines()
    event = lines[0].removeprefix("event: ")
    data = json.loads(lines[1].removeprefix("data: "))
    return {"event": event, "data": data}
