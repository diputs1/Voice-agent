from collections.abc import Callable

import pytest

from app.api.schemas import ChatRequest
from app.knowledge.kb import InMemoryKnowledgeBase
from app.services.chat_service import ChatService


@pytest.mark.asyncio
async def test_chat_service_streams_end_to_end_graph_answer(
    integration_chat_service: ChatService,
    integration_kb: InMemoryKnowledgeBase,
    parse_sse_event: Callable[[str], dict],
):
    events = [
        parse_sse_event(event)
        async for event in integration_chat_service.stream_chat(
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
    assert await integration_kb.get_thread(done["thread_id"])
