from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI

from app.agents.graph import SafariAgentGraph
from app.cache import TTLQACache
from app.config import Settings
from app.crawl_jobs import InMemoryCrawlJobStore, PostgresCrawlJobStore
from app.embeddings import EmbeddingProvider
from app.firecrawl import FirecrawlClient
from app.kb import InMemoryKnowledgeBase, KnowledgeBase, PostgresKnowledgeBase
from app.services.chat_service import ChatService
from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)


async def initialize_app_state(app: FastAPI, settings: Settings) -> None:
    app.state.settings = settings
    app.state.kb_status = {"provider": "unknown", "fallback": False, "fallback_reason": None}
    app.state.kb = await build_knowledge_base(app, settings)
    app.state.crawl_job_store = await build_crawl_job_store(app, settings)
    app.state.agent_graph = SafariAgentGraph(app.state.kb, settings)
    app.state.agent_semaphore = asyncio.Semaphore(settings.agent_max_concurrency)
    app.state.qa_cache = TTLQACache(
        ttl_seconds=settings.qa_cache_ttl_seconds,
        max_entries=settings.qa_cache_max_entries,
    )
    app.state.firecrawl_client = (
        FirecrawlClient(settings.firecrawl_api_key, settings.firecrawl_base_url)
        if settings.firecrawl_api_key
        else None
    )
    app.state.chat_service = ChatService(
        agent_graph=app.state.agent_graph,
        kb=app.state.kb,
        agent_semaphore=app.state.agent_semaphore,
        qa_cache=app.state.qa_cache,
    )
    app.state.ingestion_service = IngestionService(
        settings=settings,
        kb=app.state.kb,
        crawl_job_store=app.state.crawl_job_store,
        firecrawl_client=app.state.firecrawl_client,
    )


async def build_knowledge_base(app: FastAPI, settings: Settings) -> KnowledgeBase:
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


async def build_crawl_job_store(app: FastAPI, settings: Settings):
    if getattr(app.state, "kb_status", {}).get("provider") == "postgres":
        store = PostgresCrawlJobStore(settings.database_url)
        await store.ensure_ready()
        return store
    store = InMemoryCrawlJobStore()
    await store.ensure_ready()
    return store
