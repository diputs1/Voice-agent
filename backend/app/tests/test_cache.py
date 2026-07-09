import asyncio
from datetime import UTC, datetime

import pytest

from app.cache import QACacheKey, TTLQACache, normalize_cache_query
from app.embeddings import EmbeddingProvider
from app.ingestion import IngestedChunk
from app.kb import InMemoryKnowledgeBase


@pytest.mark.asyncio
async def test_qa_cache_returns_recent_value_and_evicts_oldest():
    cache = TTLQACache(ttl_seconds=60, max_entries=1)
    first = QACacheKey(query="a", doc_set_hash="docs")
    second = QACacheKey(query="b", doc_set_hash="docs")

    await cache.set(first, {"answer": "first"})
    await cache.set(second, {"answer": "second"})

    assert await cache.get(first) is None
    assert await cache.get(second) == {"answer": "second"}


@pytest.mark.asyncio
async def test_qa_cache_honors_ttl():
    cache = TTLQACache(ttl_seconds=1, max_entries=8)
    key = QACacheKey(query="a", doc_set_hash="docs")

    await cache.set(key, {"answer": "cached"})
    assert await cache.get(key) == {"answer": "cached"}
    await asyncio.sleep(1.01)

    assert await cache.get(key) is None


def test_normalize_cache_query_is_case_and_whitespace_insensitive():
    assert normalize_cache_query("  Giá   Vé  ") == "giá vé"


@pytest.mark.asyncio
async def test_in_memory_doc_set_hash_changes_when_documents_change():
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    await kb.ensure_ready()
    before = await kb.doc_set_hash()
    await kb.upsert_chunks(
        [
            IngestedChunk(
                source_url="https://example.test/new",
                title="New",
                section="New",
                category="general",
                content="Nội dung mới cho cache hash.",
                content_hash="new-cache-hash",
                language="vi",
                crawled_at=datetime.now(UTC),
                valid_until=None,
                metadata={},
            )
        ]
    )

    assert await kb.doc_set_hash() != before
