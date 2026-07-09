from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.agents.graph import SafariAgentGraph
from app.config import Settings, get_settings
from app.embeddings import EmbeddingProvider
from app.firecrawl import FirecrawlClient, firecrawl_result_to_chunks, scrape_result_to_chunks
from app.ingestion import fetch_vinwonders_chunks
from app.kb import InMemoryKnowledgeBase, KnowledgeBase, PostgresKnowledgeBase
from app.link_discovery import LinkDiscoveryAgent, discovery_metadata_for, normalize_url
from app.schemas import ChatRequest, CrawlJobResponse, CrawlRequest, IngestRequest, TTSRequest

logger = logging.getLogger(__name__)

app = FastAPI(title="Vin Agent API", version="0.1.0")
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    app.state.settings = settings
    app.state.kb_status = {"provider": "unknown", "fallback": False, "fallback_reason": None}
    app.state.kb = await _build_knowledge_base(settings)
    app.state.agent_graph = SafariAgentGraph(app.state.kb, settings)
    app.state.firecrawl_client = (
        FirecrawlClient(settings.firecrawl_api_key, settings.firecrawl_base_url)
        if settings.firecrawl_api_key
        else None
    )
    app.state.crawl_jobs = {}


@app.get("/health")
async def health() -> dict[str, object]:
    kb_status = getattr(
        app.state,
        "kb_status",
        {"provider": "unknown", "fallback": False, "fallback_reason": None},
    )
    return {"status": "degraded" if kb_status.get("fallback") else "ok", "kb": kb_status}


@app.post("/voice/stt-token")
async def create_stt_token() -> dict[str, str]:
    if not settings.elevenlabs_api_key:
        raise HTTPException(status_code=501, detail="ELEVENLABS_API_KEY is not configured")

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            "https://api.elevenlabs.io/v1/single-use-token/realtime_scribe",
            headers={"xi-api-key": settings.elevenlabs_api_key},
        )
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


@app.post("/voice/tts")
async def text_to_speech(payload: TTSRequest) -> StreamingResponse:
    if not settings.elevenlabs_api_key:
        raise HTTPException(status_code=501, detail="ELEVENLABS_API_KEY is not configured")

    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{settings.elevenlabs_voice_id}/stream"
        "?output_format=mp3_44100_128"
    )

    async def audio_stream() -> AsyncIterator[bytes]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                url,
                headers={
                    "xi-api-key": settings.elevenlabs_api_key or "",
                    "Content-Type": "application/json",
                },
                json={"text": payload.text, "model_id": settings.elevenlabs_tts_model},
            ) as response:
                if response.status_code >= 400:
                    detail = await response.aread()
                    raise HTTPException(status_code=response.status_code, detail=detail.decode())
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(audio_stream(), media_type="audio/mpeg")


@app.post("/chat/stream")
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    thread_id = payload.thread_id or str(uuid.uuid4())

    async def sse() -> AsyncIterator[str]:
        final_state = {"transcript": payload.transcript}
        async for node_name, node_state, current_state in app.state.agent_graph.astream_pre_answer(
            {"transcript": payload.transcript},
            thread_id,
        ):
            final_state = current_state
            yield _sse("status", {"thread_id": thread_id, "node": node_name})

        answer = ""
        yield _sse("status", {"thread_id": thread_id, "node": "voice_answer"})
        async for token in app.state.agent_graph.astream_voice_answer_tokens(final_state):
            answer += token
            yield _sse("token", {"text": token})

        final_update = app.state.agent_graph.finalize_streamed_answer(final_state, answer)
        final_state = {**final_state, **final_update}
        final_answer = final_state.get("answer", answer)
        if final_answer != answer:
            answer = final_answer
            yield _sse("replace", {"text": answer})

        citations = final_state.get("citations", [])
        await app.state.kb.save_thread_turn(thread_id, payload.transcript, answer)
        yield _sse(
            "done",
            {
                "thread_id": thread_id,
                "answer": answer,
                "citations": citations,
                "confidence": final_state.get("confidence", 0.0),
                "handoff_required": final_state.get("handoff_required", False),
                "handoff_reason": final_state.get("handoff_reason"),
                "recommended_action": final_state.get("recommended_action"),
            },
        )

    return StreamingResponse(sse(), media_type="text/event-stream")


async def require_admin_api_key(x_admin_api_key: str | None = Header(default=None)) -> None:
    configured_key = settings.admin_api_key
    if configured_key:
        if x_admin_api_key != configured_key:
            raise HTTPException(status_code=401, detail="Invalid admin API key")
        return

    if settings.app_env.lower() not in {"development", "dev", "test", "testing", "local"}:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY is not configured")


@app.post("/admin/ingest-vinwonders", dependencies=[Depends(require_admin_api_key)])
async def ingest_vinwonders(payload: IngestRequest | None = None) -> dict[str, int | str]:
    url = payload.url if payload and payload.url else settings.vinwonders_source_url
    if app.state.firecrawl_client:
        chunks, summary = await _collect_firecrawl_chunks(
            url=url,
            max_depth=2,
            max_pages=40,
            crawl_job_id=None,
        )
        inserted = await app.state.kb.upsert_chunks(chunks)
        return {
            "source_url": url,
            "provider": "firecrawl",
            "pages_crawled": summary["pages_crawled"],
            "chunks_seen": len(chunks),
            "chunks_inserted": inserted,
        }

    chunks = await fetch_vinwonders_chunks(url)
    inserted = await app.state.kb.upsert_chunks(chunks)
    return {
        "source_url": url,
        "provider": "fallback_http",
        "warning": "FIRECRAWL_API_KEY is not configured; used one-page fallback ingestion",
        "chunks_seen": len(chunks),
        "chunks_inserted": inserted,
    }


@app.post("/admin/crawl", response_model=CrawlJobResponse, dependencies=[Depends(require_admin_api_key)])
async def crawl(payload: CrawlRequest, background_tasks: BackgroundTasks) -> CrawlJobResponse:
    if not app.state.firecrawl_client:
        raise HTTPException(status_code=501, detail="FIRECRAWL_API_KEY is not configured")
    job_id = str(uuid.uuid4())
    app.state.crawl_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "url": str(payload.url),
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "pages_crawled": 0,
        "pages_failed": 0,
        "pages_skipped": 0,
        "chunks_seen": 0,
        "chunks_inserted": 0,
        "errors": [],
        "discovered_urls": [],
        "candidate_urls": [],
        "selected_urls": [],
        "skipped_urls": [],
        "skip_reasons": {},
    }
    background_tasks.add_task(_run_crawl_job, job_id, payload)
    return CrawlJobResponse(job_id=job_id, status="queued", url=str(payload.url))


@app.get("/admin/crawl-jobs/{job_id}", dependencies=[Depends(require_admin_api_key)])
async def get_crawl_job(job_id: str) -> dict[str, object]:
    job = app.state.crawl_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Crawl job not found")
    return job


@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str) -> dict[str, object]:
    return {"thread_id": thread_id, "turns": await app.state.kb.get_thread(thread_id)}


async def _build_knowledge_base(settings: Settings) -> KnowledgeBase:
    embeddings = EmbeddingProvider(settings.openai_api_key, settings.openai_embedding_model)
    postgres = PostgresKnowledgeBase(settings.database_url, embeddings)
    try:
        await postgres.ensure_ready()
        app.state.kb_status = {"provider": "postgres", "fallback": False, "fallback_reason": None}
        return postgres
    except Exception as exc:
        if not settings.allow_kb_fallback:
            app.state.kb_status = {
                "provider": "postgres",
                "fallback": False,
                "fallback_reason": str(exc),
            }
            raise
        logger.warning("Falling back to in-memory knowledge base: %s", exc)
        memory = InMemoryKnowledgeBase(embeddings)
        await memory.ensure_ready()
        app.state.kb_status = {
            "provider": "memory",
            "fallback": True,
            "fallback_reason": str(exc),
        }
        return memory


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _word_chunks(text: str) -> list[str]:
    words = text.split(" ")
    return [word + (" " if idx < len(words) - 1 else "") for idx, word in enumerate(words)]


async def _run_crawl_job(job_id: str, payload: CrawlRequest) -> None:
    job = app.state.crawl_jobs[job_id]
    job.update({"status": "running", "updated_at": _utc_now()})
    try:
        chunks, summary = await _collect_firecrawl_chunks(
            url=str(payload.url),
            max_depth=payload.max_depth,
            crawl_job_id=job_id,
            max_pages=payload.max_pages,
        )
        inserted = await app.state.kb.upsert_chunks(chunks)
        job.update(
            {
                "status": "completed",
                "updated_at": _utc_now(),
                "provider": "firecrawl",
                "firecrawl_id": summary["firecrawl_id"],
                "started_at": summary["started_at"],
                "finished_at": summary["finished_at"],
                "pages_seen": summary["pages_seen"],
                "pages_crawled": summary["pages_crawled"],
                "pages_failed": summary["pages_failed"],
                "pages_skipped": 0,
                "credits_used": summary["credits_used"],
                "chunks_seen": len(chunks),
                "chunks_inserted": inserted,
                "errors": summary["errors"],
                "discovered_urls": summary["discovered_urls"],
                "candidate_urls": summary["candidate_urls"],
                "selected_urls": summary["selected_urls"],
                "skipped_urls": summary["skipped_urls"],
                "skip_reasons": summary["skip_reasons"],
            }
        )
    except Exception as exc:
        logger.exception("Crawl job failed: %s", job_id)
        job.update({"status": "failed", "updated_at": _utc_now(), "error": str(exc)})


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


async def _collect_firecrawl_chunks(
    *,
    url: str,
    max_depth: int,
    max_pages: int,
    crawl_job_id: str | None,
) -> tuple[list, dict[str, object]]:
    firecrawl_job = await app.state.firecrawl_client.start_crawl(
        url=url,
        max_depth=max_depth,
        limit=max_pages,
        include_paths=_default_include_paths(url),
        exclude_paths=_default_exclude_paths(),
    )
    if crawl_job_id and crawl_job_id in app.state.crawl_jobs:
        app.state.crawl_jobs[crawl_job_id].update(
            {
                "status": "waiting_firecrawl",
                "updated_at": _utc_now(),
                "firecrawl_id": str(firecrawl_job["id"]),
            }
        )
    firecrawl_result = await app.state.firecrawl_client.wait_for_crawl(
        str(firecrawl_job["id"]),
        timeout_seconds=240.0,
    )
    if crawl_job_id and crawl_job_id in app.state.crawl_jobs:
        app.state.crawl_jobs[crawl_job_id].update(
            {
                "status": "processing_firecrawl_result",
                "updated_at": _utc_now(),
                "pages_seen": firecrawl_result.get("total", 0),
                "pages_crawled": int(firecrawl_result.get("completed") or 0),
                "credits_used": int(firecrawl_result.get("creditsUsed") or 0),
            }
        )
    discovered_urls = [
        item.get("metadata", {}).get("sourceURL") or item.get("metadata", {}).get("url")
        for item in firecrawl_result.get("data", [])
        if item.get("metadata")
    ]
    discovered_set = {str(item) for item in discovered_urls if item}
    normalized_discovered_set = {normalized for item in discovered_set if (normalized := normalize_url(item))}
    errors: list[dict[str, str]] = []
    scraped_count = 0
    discovery = LinkDiscoveryAgent(
        seed_url=url,
        fallback_urls=_strategic_firecrawl_urls(url),
    ).discover_from_firecrawl_result(firecrawl_result)
    selected_urls = [item.url for item in discovery.selected_urls]
    selected_metadata = {
        discovered.url: discovery_metadata_for(discovered) for discovered in discovery.selected_urls
    }
    chunks = firecrawl_result_to_chunks(
        result=firecrawl_result,
        seed_url=url,
        crawl_job_id=crawl_job_id,
        url_metadata=selected_metadata,
    )

    for discovered in discovery.selected_urls:
        scrape_url = discovered.url
        if scrape_url in normalized_discovered_set:
            continue
        try:
            if crawl_job_id and crawl_job_id in app.state.crawl_jobs:
                app.state.crawl_jobs[crawl_job_id].update(
                    {
                        "status": "scraping_discovered_urls",
                        "updated_at": _utc_now(),
                        "current_scrape_url": scrape_url,
                        "selected_urls": selected_urls,
                        "candidate_urls": discovery.candidate_urls,
                        "skipped_urls": discovery.skipped_urls,
                        "skip_reasons": discovery.skip_reasons,
                        "scraped_count": scraped_count,
                    }
                )
            scrape_result = await app.state.firecrawl_client.scrape(scrape_url)
            scrape_chunks = scrape_result_to_chunks(
                result=scrape_result,
                seed_url=url,
                crawl_job_id=crawl_job_id,
                url_metadata={scrape_url: selected_metadata[scrape_url]},
            )
            if scrape_chunks:
                chunks.extend(scrape_chunks)
                discovered_urls.append(scrape_url)
                scraped_count += 1
                if crawl_job_id and crawl_job_id in app.state.crawl_jobs:
                    app.state.crawl_jobs[crawl_job_id].update(
                        {
                            "updated_at": _utc_now(),
                            "pages_crawled": int(firecrawl_result.get("completed") or 0) + scraped_count,
                            "chunks_seen": len(chunks),
                            "scraped_count": scraped_count,
                        }
                    )
        except Exception as exc:
            errors.append({"url": scrape_url, "error": str(exc)})
            if crawl_job_id and crawl_job_id in app.state.crawl_jobs:
                app.state.crawl_jobs[crawl_job_id].update(
                    {
                        "updated_at": _utc_now(),
                        "pages_failed": len(errors),
                        "errors": errors,
                    }
                )

    summary: dict[str, object] = {
        "firecrawl_id": str(firecrawl_job["id"]),
        "started_at": firecrawl_result.get("createdAt"),
        "finished_at": firecrawl_result.get("completedAt"),
        "pages_seen": firecrawl_result.get("total", 0),
        "pages_crawled": int(firecrawl_result.get("completed") or 0) + scraped_count,
        "pages_failed": len(errors),
        "credits_used": int(firecrawl_result.get("creditsUsed") or 0) + scraped_count,
        "errors": errors,
        "discovered_urls": discovered_urls,
        "candidate_urls": discovery.candidate_urls,
        "selected_urls": selected_urls,
        "skipped_urls": discovery.skipped_urls,
        "skip_reasons": discovery.skip_reasons,
    }
    return chunks, summary


def _strategic_firecrawl_urls(url: str) -> list[str]:
    if "vinwonders.com" not in url:
        return []
    return [
        "https://vinwonders.com/vi/vinpearl-safari-phu-quoc/",
        "https://vinwonders.com/vi/vinpearl-safari-phu-quoc-gia-ve-va-quy-dinh/",
        "https://vinwonders.com/vi/promotions/",
        "https://vinwonders.com/vi/uu-dai/vinwonders-uu-dai-15-hoi-vien-vinclub/",
        "https://vinwonders.com/vi/wonderpedia/news/ra-mat-chuong-trinh-vinwonders-affiliate/",
        "https://vinwonders.com/vi/wonderpedia/news/gia-ve-vinpearl-safari-phu-quoc/",
        "https://vinwonders.com/vi/wonderpedia/news/voucher-vinpearl-safari-phu-quoc/",
        "https://vinwonders.com/en/vinpearl-safari-phu-quoc/",
        "https://vinwonders.com/en/vinpearl-safari-phu-quoc-price-and-regulations/",
        "https://vinwonders.com/en/promotions/",
        "https://vinwonders.com/en/terms-and-conditions/",
    ]


def _default_include_paths(url: str) -> list[str]:
    if "/vi/" not in url and "/en/" not in url:
        return []
    return [
        "vi/.*vinpearl-safari-phu-quoc.*",
        "en/.*vinpearl-safari-phu-quoc.*",
        "vi/promotions.*",
        "en/promotions.*",
        "vi/uu-dai/.*",
        "en/uu-dai/.*",
        "vi/wonderpedia/news/.*(vinpearl-safari-phu-quoc|voucher|combo|affiliate|vinclub|gia-ve).*",
        "en/wonderpedia/news/.*(vinpearl-safari-phu-quoc|voucher|combo|affiliate|vinclub|ticket|price).*",
    ]


def _default_exclude_paths() -> list[str]:
    return [
        ".*\\.(jpg|jpeg|png|gif|webp|svg|ico|css|js|mp4|mov|zip)$",
        ".*/(ko|zh|ru)/.*",
        ".*login.*",
    ]
