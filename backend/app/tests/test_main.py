import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks
from fastapi import HTTPException

from app import main
from app.core.cache import TTLQACache
from app.routers import admin, chat, health
from app.api.schemas import CrawlRequest, TTSRequest
from app.services.chat_service import ChatService
from app.services.ingestion_service import IngestionService


@pytest.mark.asyncio
async def test_health_reports_ok_for_primary_kb():
    main.app.state.kb_status = {"provider": "postgres", "fallback": False, "fallback_reason": None}

    response = await health.health(_request())

    assert response["status"] == "ok"
    assert response["kb"]["provider"] == "postgres"
    assert response["kb"]["fallback"] is False


@pytest.mark.asyncio
async def test_health_reports_degraded_for_kb_fallback():
    main.app.state.kb_status = {
        "provider": "memory",
        "fallback": True,
        "fallback_reason": "connection failed",
    }

    response = await health.health(_request())

    assert response["status"] == "degraded"
    assert response["kb"]["provider"] == "memory"
    assert response["kb"]["fallback"] is True


@pytest.mark.asyncio
async def test_admin_auth_allows_dev_when_key_is_unset(monkeypatch):
    monkeypatch.setattr(main.settings, "app_env", "development")
    monkeypatch.setattr(main.settings, "admin_api_key", None)
    main.app.state.settings = main.settings

    await admin.require_admin_api_key(_request())


@pytest.mark.asyncio
async def test_admin_auth_rejects_missing_or_wrong_key_when_configured(monkeypatch):
    monkeypatch.setattr(main.settings, "app_env", "development")
    monkeypatch.setattr(main.settings, "admin_api_key", "secret-admin-key")
    main.app.state.settings = main.settings

    with pytest.raises(HTTPException) as missing:
        await admin.require_admin_api_key(_request())
    assert missing.value.status_code == 401

    with pytest.raises(HTTPException) as wrong:
        await admin.require_admin_api_key(_request(), "wrong")
    assert wrong.value.status_code == 401

    await admin.require_admin_api_key(_request(), "secret-admin-key")


@pytest.mark.asyncio
async def test_admin_auth_fails_closed_outside_dev_when_key_is_unset(monkeypatch):
    monkeypatch.setattr(main.settings, "app_env", "production")
    monkeypatch.setattr(main.settings, "admin_api_key", None)
    main.app.state.settings = main.settings

    with pytest.raises(HTTPException) as exc:
        await admin.require_admin_api_key(_request())

    assert exc.value.status_code == 500
    assert "ADMIN_API_KEY" in exc.value.detail


@pytest.mark.asyncio
async def test_crawl_creates_queued_job_in_store(monkeypatch):
    store = FakeCrawlJobStore()
    service = IngestionService(
        settings=main.settings,
        kb=FakeKnowledgeBase(),
        crawl_job_store=store,
        firecrawl_client=object(),
    )

    response = await service.crawl(
        CrawlRequest(url="https://vinwonders.com/vi/vinpearl-safari-phu-quoc/"),
        BackgroundTasks(),
    )

    assert response.status == "queued"
    assert response.job_id in store.jobs
    assert store.jobs[response.job_id]["status"] == "queued"
    assert store.jobs[response.job_id]["url"] == "https://vinwonders.com/vi/vinpearl-safari-phu-quoc/"


@pytest.mark.asyncio
async def test_get_crawl_job_returns_404_for_missing_store_job():
    service = IngestionService(
        settings=main.settings,
        kb=FakeKnowledgeBase(),
        crawl_job_store=FakeCrawlJobStore(),
        firecrawl_client=object(),
    )

    with pytest.raises(HTTPException) as exc:
        await service.get_crawl_job("missing")

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_chat_stream_uses_cached_answer_after_contextualized_query():
    graph = FakeAgentGraph()
    kb = FakeKnowledgeBase()
    main.app.state.chat_service = ChatService(
        agent_graph=graph,
        kb=kb,
        agent_semaphore=asyncio.Semaphore(1),
        qa_cache=TTLQACache(ttl_seconds=60, max_entries=8),
    )

    first = await _collect_sse_events(
        await chat.chat_stream(_request(), _chat_request("Mở cửa mấy giờ?"))
    )
    second = await _collect_sse_events(
        await chat.chat_stream(_request(), _chat_request("Mở cửa mấy giờ?"))
    )

    assert first[-1]["payload"]["cache_hit"] is False
    assert second[-1]["payload"]["cache_hit"] is True
    assert graph.voice_calls == 1
    assert graph.completed_pre_answer_calls == 1
    assert kb.saved_turns == [
        ("test-thread", "Mở cửa mấy giờ?", "Câu trả lời từ graph."),
        ("test-thread", "Mở cửa mấy giờ?", "Câu trả lời từ graph."),
    ]


@pytest.mark.asyncio
async def test_text_to_speech_enforces_vietnamese_model_and_language(monkeypatch):
    sent_requests = []
    main.app.state.settings = SimpleNamespace(
        elevenlabs_api_key="test-key",
        elevenlabs_voice_id="test-voice",
        elevenlabs_tts_model="eleven_flash_v2_5",
        elevenlabs_tts_language_code="vi",
    )

    class FakeTTSResponse:
        status_code = 200

        async def aiter_bytes(self):
            yield b"audio"

    class FakeStream:
        async def __aenter__(self):
            return FakeTTSResponse()

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        def stream(self, method, url, *, headers, json):
            sent_requests.append(
                {
                    "method": method,
                    "url": url,
                    "headers": headers,
                    "json": json,
                }
            )
            return FakeStream()

    monkeypatch.setattr(chat.httpx, "AsyncClient", FakeAsyncClient)

    response = await chat.text_to_speech(_request(), TTSRequest(text="Xin chào quý khách."))
    audio = b""
    async for chunk in response.body_iterator:
        audio += chunk

    assert audio == b"audio"
    assert sent_requests == [
        {
            "method": "POST",
            "url": (
                "https://api.elevenlabs.io/v1/text-to-speech/test-voice/stream"
                "?output_format=mp3_44100_128"
            ),
            "headers": {
                "xi-api-key": "test-key",
                "Content-Type": "application/json",
            },
            "json": {
                "text": "Xin chào quý khách.",
                "model_id": "eleven_flash_v2_5",
                "language_code": "vi",
            },
        }
    ]


class FakeCrawlJobStore:
    def __init__(self) -> None:
        self.jobs = {}

    async def create(self, *, job_id, url, payload, initial):
        del payload
        self.jobs[job_id] = {**initial, "job_id": job_id, "url": url}

    async def update(self, job_id, updates):
        self.jobs[job_id].update(updates)

    async def get(self, job_id):
        return self.jobs.get(job_id)


class FakeKnowledgeBase:
    def __init__(self) -> None:
        self.saved_turns = []

    async def doc_set_hash(self):
        return "docs-v1"

    async def save_thread_turn(self, thread_id, transcript, answer):
        self.saved_turns.append((thread_id, transcript, answer))


class FakeAgentGraph:
    def __init__(self) -> None:
        self.voice_calls = 0
        self.completed_pre_answer_calls = 0

    async def astream_pre_answer(self, state, thread_id):
        del thread_id
        current = {
            **state,
            "search_query": "Vinpearl Safari mở cửa mấy giờ?",
            "confidence": 0.9,
            "citations": [],
            "handoff_required": False,
        }
        yield "contextualize_query", {"search_query": current["search_query"]}, current
        self.completed_pre_answer_calls += 1

    async def astream_voice_answer_tokens(self, state):
        del state
        self.voice_calls += 1
        yield "Câu trả lời từ graph."

    def finalize_streamed_answer(self, state, answer):
        return {
            **state,
            "answer": answer,
            "citations": [],
            "confidence": 0.9,
            "handoff_required": False,
            "handoff_reason": None,
            "recommended_action": None,
        }


def _chat_request(transcript):
    from app.api.schemas import ChatRequest

    return ChatRequest(transcript=transcript, thread_id="test-thread")


async def _collect_sse_events(response):
    body = ""
    async for chunk in response.body_iterator:
        body += chunk.decode() if isinstance(chunk, bytes) else chunk

    events = []
    for raw_event in body.strip().split("\n\n"):
        lines = raw_event.splitlines()
        event = lines[0].removeprefix("event: ")
        payload = json.loads(lines[1].removeprefix("data: "))
        events.append({"event": event, "payload": payload})
    return events


def _request():
    return SimpleNamespace(app=main.app)
