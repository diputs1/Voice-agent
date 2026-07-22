from __future__ import annotations

import json
import re
import unicodedata
from contextlib import asynccontextmanager
from hashlib import sha256
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from collections.abc import AsyncIterator
from typing import Protocol
from urllib.parse import urlparse

import psycopg
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langsmith import traceable

from app.knowledge.embeddings import EmbeddingProvider, pgvector_literal
from app.crawling.ingestion import IngestedChunk, seed_chunks
from app.core.sites import canonical_root_url, relevance_profile_for_text, site_id_for_url


CATEGORY_KEYWORDS = {
    "wonderpedia": ("wonderpedia", "wonderculture", "wonderland", "wondermoment", "wondercreature"),
    "affiliate": ("affiliate", "hoa hồng", "đối tác"),
    "vinclub": ("vinclub", "hội viên", "gold", "platinum", "diamond"),
    "offer": ("ưu đãi", "voucher", "khuyến mãi", "giảm", "hạn áp dụng"),
    "price": ("giá", "giá vé", "bảng giá", "vnđ", "vnd"),
    "booking": (
        "booking",
        "đặt vé",
        "đặt online",
        "qr",
        "sản phẩm",
        "san pham",
        "product",
        "products",
        "gói",
        "combo",
        "tour",
        "onsite service",
    ),
    "schedule": ("giờ", "mở cửa", "lịch", "lúc nào", "thời gian"),
    "experience": (
        "show",
        "biểu diễn",
        "động vật",
        "animal",
        "night safari",
        "kid zoo",
        "trải nghiệm",
        "tham quan",
        "sản phẩm",
        "san pham",
        "tour",
    ),
}


@dataclass
class KnowledgeHit:
    id: str
    title: str
    section: str
    category: str
    content: str
    source_url: str
    language: str
    crawled_at: str | None
    valid_until: str | None
    metadata: dict | None
    score: float
    site_id: str | None = None


@dataclass(frozen=True)
class SearchContext:
    raw_query: str
    normalized_query: str
    query_tokens: frozenset[str]
    query_lower: str
    wanted_categories: tuple[str, ...]


class KnowledgeBase(Protocol):
    async def ensure_ready(self) -> None: ...

    async def upsert_chunks(self, chunks: list[IngestedChunk]) -> int: ...

    async def search(self, query: str, limit: int = 5, site_id: str | None = None) -> list[KnowledgeHit]: ...

    async def save_thread_turn(
        self, thread_id: str, transcript: str, answer: str, site_id: str | None = None
    ) -> None: ...

    async def get_thread(self, thread_id: str) -> list[dict]: ...

    async def doc_set_hash(self, site_id: str | None = None) -> str: ...


class Retriever(Protocol):
    async def search(self, query: str, limit: int = 5, site_id: str | None = None) -> list[KnowledgeHit]: ...


class IngestionSink(Protocol):
    async def upsert_chunks(self, chunks: list[IngestedChunk]) -> int: ...


class ThreadStore(Protocol):
    async def save_thread_turn(
        self, thread_id: str, transcript: str, answer: str, site_id: str | None = None
    ) -> None: ...

    async def get_thread(self, thread_id: str) -> list[dict]: ...


class DocSetHasher(Protocol):
    async def doc_set_hash(self, site_id: str | None = None) -> str: ...


class AgentKnowledgeBase(Retriever, ThreadStore, Protocol):
    pass


class ChatKnowledgeStore(ThreadStore, DocSetHasher, Protocol):
    pass


class InMemoryKnowledgeBase:
    def __init__(self, embeddings: EmbeddingProvider) -> None:
        self.embeddings = embeddings
        self.rows: list[dict] = []
        self.threads: dict[str, list[dict]] = {}
        self.sites: dict[str, dict] = {}

    async def ensure_ready(self) -> None:
        if not self.rows:
            await self.upsert_chunks(seed_chunks())

    async def upsert_chunks(self, chunks: list[IngestedChunk]) -> int:
        vectors = await self.embeddings.embed_documents([chunk.content for chunk in chunks])
        existing_hashes = {(row.get("site_id") or "default", row["content_hash"]) for row in self.rows}
        inserted = 0
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk_site_id = chunk.site_id or site_id_for_url(chunk.source_url)
            if (chunk_site_id, chunk.content_hash) in existing_hashes:
                continue
            row = {**asdict(chunk), "embedding": vector, "id": chunk.content_hash}
            row["site_id"] = chunk_site_id
            row["search_text"] = _search_text(row)
            self.rows.append(row)
            existing_hashes.add((chunk_site_id, chunk.content_hash))
            inserted += 1
        return inserted

    @traceable(name="kb.in_memory_search")
    async def search(self, query: str, limit: int = 5, site_id: str | None = None) -> list[KnowledgeHit]:
        context = _search_context(query)
        query_vector = await self.embeddings.embed_query(query)
        vector_ranked = []
        lexical_ranked = []
        for row in self.rows:
            if site_id and row.get("site_id") != site_id:
                continue
            confidence = _adjusted_score(context, row, _cosine(query_vector, row["embedding"]))
            lexical = _lexical_score(context, row)
            vector_ranked.append((confidence, row, confidence))
            if lexical > 0:
                lexical_confidence = _confidence_with_lexical_score(confidence, lexical)
                lexical_ranked.append((lexical_confidence, row, lexical_confidence))
        fused = _rrf_fuse(vector_ranked, lexical_ranked, limit=max(limit * 6, 20))
        scored = _diversify_categories(context, fused, limit)
        return [
            KnowledgeHit(
                id=row["id"],
                site_id=row.get("site_id"),
                title=row["title"],
                section=row["section"],
                category=row["category"],
                content=row["content"],
                source_url=row["source_url"],
                language=row.get("language", "vi"),
                crawled_at=_iso(row["crawled_at"]),
                valid_until=_iso(row["valid_until"]),
                metadata=row.get("metadata") or {},
                score=float(row.get("score") or 0.0),
            )
            for _, row in scored[:limit]
        ]

    async def save_thread_turn(
        self, thread_id: str, transcript: str, answer: str, site_id: str | None = None
    ) -> None:
        self.threads.setdefault(thread_id, []).append(
            {
                "site_id": site_id,
                "transcript": transcript,
                "answer": answer,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )

    async def get_thread(self, thread_id: str) -> list[dict]:
        return self.threads.get(thread_id, [])

    async def doc_set_hash(self, site_id: str | None = None) -> str:
        content_hashes = sorted(
            str(row.get("content_hash") or row.get("id") or "")
            for row in self.rows
            if not site_id or row.get("site_id") == site_id
        )
        return sha256("|".join(content_hashes).encode("utf-8")).hexdigest()

    async def upsert_site(
        self,
        *,
        site_id: str,
        root_url: str,
        allowed_domains: list[str],
        site_aliases: list[str] | None = None,
        crawl_policy: dict | None = None,
    ) -> None:
        self.sites[site_id] = {
            "site_id": site_id,
            "root_url": root_url,
            "allowed_domains": allowed_domains,
            "site_aliases": site_aliases or [],
            "crawl_policy": crawl_policy or {},
            "created_at": self.sites.get(site_id, {}).get("created_at") or datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }

    async def site_status(self, site_id: str | None = None) -> dict:
        rows = [row for row in self.rows if not site_id or row.get("site_id") == site_id]
        category_counts: dict[str, int] = {}
        for row in rows:
            category_counts[str(row.get("category") or "unknown")] = (
                category_counts.get(str(row.get("category") or "unknown"), 0) + 1
            )
        return {
            "site_id": site_id,
            "site_count": len(self.sites),
            "document_count": len(rows),
            "category_counts": category_counts,
        }


class PostgresKnowledgeBase:
    def __init__(
        self,
        database_url: str,
        embeddings: EmbeddingProvider,
        *,
        pool: AsyncConnectionPool | None = None,
    ) -> None:
        self.database_url = database_url
        self.embeddings = embeddings
        self.pool = pool

    @asynccontextmanager
    async def _connection(
        self,
        *,
        autocommit: bool = False,
    ) -> AsyncIterator[AsyncConnection]:
        if self.pool:
            async with self.pool.connection() as conn:
                yield conn
            return
        async with await psycopg.AsyncConnection.connect(
            self.database_url,
            autocommit=autocommit,
        ) as conn:
            yield conn

    async def ensure_ready(self) -> None:
        async with self._connection(autocommit=True) as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sites (
                  site_id TEXT PRIMARY KEY,
                  root_url TEXT NOT NULL,
                  allowed_domains TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
                  site_aliases TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
                  crawl_policy JSONB NOT NULL DEFAULT '{}'::jsonb,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                  site_id TEXT NOT NULL DEFAULT 'default',
                  source_url TEXT NOT NULL,
                  canonical_url TEXT,
                  title TEXT NOT NULL,
                  section TEXT NOT NULL,
                  category TEXT NOT NULL,
                  content TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  language TEXT NOT NULL DEFAULT 'vi',
                  crawled_at TIMESTAMPTZ NOT NULL,
                  valid_until TIMESTAMPTZ,
                  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                  embedding vector(1536) NOT NULL
                )
                """
            )
            await conn.execute(
                "ALTER TABLE sites ADD COLUMN IF NOT EXISTS site_aliases TEXT[] NOT NULL DEFAULT ARRAY[]::text[]"
            )
            await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS site_id TEXT NOT NULL DEFAULT 'default'")
            await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS canonical_url TEXT")
            await conn.execute("ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_content_hash_key")
            await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS search_text TEXT NOT NULL DEFAULT ''")
            await conn.execute(
                """
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS search_vector tsvector
                GENERATED ALWAYS AS (to_tsvector('simple', search_text)) STORED
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS thread_turns (
                  id BIGSERIAL PRIMARY KEY,
                  thread_id TEXT NOT NULL,
                  site_id TEXT,
                  transcript TEXT NOT NULL,
                  answer TEXT NOT NULL,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute("ALTER TABLE thread_turns ADD COLUMN IF NOT EXISTS site_id TEXT")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                  run_id TEXT PRIMARY KEY,
                  graph_version TEXT NOT NULL,
                  dataset TEXT NOT NULL,
                  metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS documents_site_content_hash_idx ON documents (site_id, content_hash)"
            )
            await conn.execute("CREATE INDEX IF NOT EXISTS documents_site_idx ON documents (site_id)")
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS thread_turns_thread_created_idx
                ON thread_turns (thread_id, created_at ASC, id ASC)
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS documents_metadata_idx ON documents USING gin (metadata)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS documents_search_vector_idx ON documents USING gin (search_vector)"
            )
            await self._backfill_search_text(conn)
            await self._ensure_vector_index(conn)

    async def upsert_chunks(self, chunks: list[IngestedChunk]) -> int:
        vectors = await self.embeddings.embed_documents([chunk.content for chunk in chunks])
        inserted = 0
        site_sources: dict[str, str] = {}
        prepared_rows = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk_site_id = chunk.site_id or site_id_for_url(chunk.source_url)
            site_sources.setdefault(chunk_site_id, chunk.source_url)
            prepared_rows.append((chunk_site_id, chunk, vector))

        async with self._connection(autocommit=True) as conn:
            for chunk_site_id, source_url in site_sources.items():
                await self._upsert_site_for_chunk(conn, chunk_site_id, source_url)

            for chunk_site_id, chunk, vector in prepared_rows:
                canonical_url = (chunk.metadata or {}).get("canonical_url") or chunk.source_url
                cursor = await conn.execute(
                    """
                    INSERT INTO documents (
                      site_id, source_url, canonical_url, title, section, category, content, content_hash,
                      language, crawled_at, valid_until, metadata, embedding, search_text
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector, %s)
                    ON CONFLICT (site_id, content_hash) DO NOTHING
                    RETURNING id
                    """,
                    (
                        chunk_site_id,
                        chunk.source_url,
                        str(canonical_url),
                        chunk.title,
                        chunk.section,
                        chunk.category,
                        chunk.content,
                        chunk.content_hash,
                        chunk.language,
                        chunk.crawled_at,
                        chunk.valid_until,
                        json.dumps(_metadata_for(chunk), ensure_ascii=False),
                        pgvector_literal(vector),
                        _search_text(asdict(chunk)),
                    ),
                )
                result = await cursor.fetchone()
                if result:
                    inserted += 1
            if inserted:
                await self._ensure_vector_index(conn)
        return inserted

    @traceable(name="kb.postgres_search")
    async def search(self, query: str, limit: int = 5, site_id: str | None = None) -> list[KnowledgeHit]:
        context = _search_context(query)
        vector = await _trace_embed_query(self.embeddings, query)
        candidate_limit = max(limit * 6, 20)
        async with self._connection() as conn:
            rows = await _trace_hybrid_sql(
                conn=conn,
                context=context,
                vector=vector,
                candidate_limit=candidate_limit,
                site_id=site_id,
            )
        return await _trace_rerank_results(
            context=context,
            rows=rows,
            limit=limit,
            candidate_limit=candidate_limit,
        )

    async def save_thread_turn(
        self, thread_id: str, transcript: str, answer: str, site_id: str | None = None
    ) -> None:
        async with self._connection(autocommit=True) as conn:
            await conn.execute(
                "INSERT INTO thread_turns (thread_id, site_id, transcript, answer) VALUES (%s, %s, %s, %s)",
                (thread_id, site_id, transcript, answer),
            )

    async def get_thread(self, thread_id: str) -> list[dict]:
        async with self._connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT site_id, transcript, answer, created_at::text
                    FROM thread_turns
                    WHERE thread_id = %s
                    ORDER BY created_at ASC, id ASC
                    """,
                    (thread_id,),
                )
                rows = await cur.fetchall()
                return list(rows)

    async def doc_set_hash(self, site_id: str | None = None) -> str:
        async with self._connection() as conn:
            cursor = await conn.execute(
                f"""
                SELECT md5(coalesce(string_agg(content_hash, ',' ORDER BY content_hash), ''))
                FROM documents
                {'WHERE site_id = %s' if site_id else ''}
                """,
                (site_id,) if site_id else (),
            )
            row = await cursor.fetchone()
            value = row[0]
        return str(value)

    async def upsert_site(
        self,
        *,
        site_id: str,
        root_url: str,
        allowed_domains: list[str],
        site_aliases: list[str] | None = None,
        crawl_policy: dict | None = None,
    ) -> None:
        async with self._connection(autocommit=True) as conn:
            await conn.execute(
                """
                INSERT INTO sites (site_id, root_url, allowed_domains, site_aliases, crawl_policy)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (site_id) DO UPDATE SET
                  root_url = EXCLUDED.root_url,
                  allowed_domains = EXCLUDED.allowed_domains,
                  site_aliases = EXCLUDED.site_aliases,
                  crawl_policy = EXCLUDED.crawl_policy,
                  updated_at = now()
                """,
                (
                    site_id,
                    root_url,
                    allowed_domains,
                    site_aliases or [],
                    json.dumps(_json_safe(crawl_policy or {}), ensure_ascii=False),
                ),
            )

    async def site_status(self, site_id: str | None = None) -> dict:
        async with self._connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT site_id, root_url, allowed_domains, site_aliases, crawl_policy,
                           created_at::text, updated_at::text
                    FROM sites
                    WHERE (%s::text IS NULL OR site_id = %s)
                    ORDER BY updated_at DESC
                    """,
                    (site_id, site_id),
                )
                site_rows = await cur.fetchall()
                await cur.execute(
                    """
                    SELECT category, count(*) AS count
                    FROM documents
                    WHERE (%s::text IS NULL OR site_id = %s)
                    GROUP BY category
                    ORDER BY count DESC, category ASC
                    """,
                    (site_id, site_id),
                )
                category_rows = await cur.fetchall()
                await cur.execute(
                    """
                    SELECT count(*) AS count, max(crawled_at)::text AS latest_crawl
                    FROM documents
                    WHERE (%s::text IS NULL OR site_id = %s)
                    """,
                    (site_id, site_id),
                )
                doc_count = await cur.fetchone()
                await cur.execute(
                    """
                    SELECT count(*) AS count
                    FROM documents
                    WHERE (%s::text IS NULL OR site_id = %s)
                      AND valid_until IS NOT NULL
                      AND valid_until < now()
                    """,
                    (site_id, site_id),
                )
                stale_count = await cur.fetchone()
        return {
            "site_id": site_id,
            "sites": [dict(row) for row in site_rows],
            "document_count": int(doc_count["count"] or 0),
            "latest_crawl": doc_count["latest_crawl"],
            "category_counts": {row["category"]: int(row["count"]) for row in category_rows},
            "stale_count": int(stale_count["count"] or 0),
        }

    async def _backfill_search_text(self, conn) -> None:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT id::text, title, section, category, content, source_url, language, metadata
                FROM documents
                WHERE search_text = ''
                """
            )
            rows = await cur.fetchall()
        for row in rows:
            await conn.execute(
                "UPDATE documents SET search_text = %s WHERE id = %s::uuid",
                (_search_text(dict(row)), row["id"]),
            )

    async def _upsert_site_for_chunk(self, conn, site_id: str, source_url: str) -> None:
        root_url = canonical_root_url(source_url)
        await conn.execute(
            """
            INSERT INTO sites (site_id, root_url, allowed_domains, site_aliases, crawl_policy)
            VALUES (%s, %s, %s, ARRAY[]::text[], '{}'::jsonb)
            ON CONFLICT (site_id) DO NOTHING
            """,
            (site_id, root_url, [urlparse_root_host(source_url)]),
        )

    async def _ensure_vector_index(self, conn) -> None:
        cursor = await conn.execute("SELECT count(*) AS count FROM documents")
        row = await cursor.fetchone()
        document_count = int(row["count"] if isinstance(row, dict) else row[0])
        if document_count <= 0:
            return
        lists = _ivfflat_lists(document_count)
        await conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS documents_embedding_idx
            ON documents USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {lists})
            """
        )


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _ivfflat_lists(document_count: int) -> int:
    if document_count < 1_000:
        return 1
    return max(1, min(100, document_count // 1_000))


def _hybrid_search_sql(*, site_id: str | None, include_lexical: bool) -> str:
    site_filter = "WHERE d.site_id = %s" if site_id else ""
    lexical_site_filter = "d.site_id = %s AND " if site_id else ""
    lexical_cte = (
        f"""
        lexical_matches AS (
          SELECT d.id::text, d.site_id, d.title, d.section, d.category, d.content,
                 d.source_url, d.language, d.crawled_at::text, d.valid_until::text,
                 d.metadata, COALESCE(s.site_aliases, ARRAY[]::text[]) AS site_aliases,
                 ts_rank_cd(d.search_vector, websearch_to_tsquery('simple', %s)) AS lexical_score,
                 1 - (d.embedding <=> %s::vector) AS score,
                 'lexical' AS match_source
          FROM documents d
          LEFT JOIN sites s ON s.site_id = d.site_id
          WHERE {lexical_site_filter}d.search_vector @@ websearch_to_tsquery('simple', %s)
          ORDER BY lexical_score DESC
          LIMIT %s
        )
        """
        if include_lexical
        else """
        lexical_matches AS (
          SELECT * FROM vector_matches WHERE false
        )
        """
    )
    return f"""
        WITH vector_matches AS (
          SELECT d.id::text, d.site_id, d.title, d.section, d.category, d.content,
                 d.source_url, d.language, d.crawled_at::text, d.valid_until::text,
                 d.metadata, COALESCE(s.site_aliases, ARRAY[]::text[]) AS site_aliases,
                 NULL::double precision AS lexical_score,
                 1 - (d.embedding <=> %s::vector) AS score,
                 'vector' AS match_source
          FROM documents d
          LEFT JOIN sites s ON s.site_id = d.site_id
          {site_filter}
          ORDER BY d.embedding <=> %s::vector
          LIMIT %s
        ),
        {lexical_cte}
        SELECT * FROM vector_matches
        UNION ALL
        SELECT * FROM lexical_matches
        """


def _hybrid_search_params(
    *,
    vector: list[float],
    query_text: str,
    candidate_limit: int,
    site_id: str | None,
) -> tuple[object, ...]:
    vector_literal = pgvector_literal(vector)
    params: list[object] = [vector_literal]
    if site_id:
        params.append(site_id)
    params.extend([vector_literal, candidate_limit])
    if query_text:
        params.append(query_text)
        params.append(vector_literal)
        if site_id:
            params.append(site_id)
        params.extend([query_text, candidate_limit])
    return tuple(params)


@traceable(
    name="kb.embed_query",
    run_type="retriever",
    process_inputs=lambda inputs: {
        "query": inputs.get("query", ""),
        "query_length": len(str(inputs.get("query", ""))),
    },
    process_outputs=lambda output: {"dimensions": len(output)},
)
async def _trace_embed_query(embeddings: EmbeddingProvider, query: str) -> list[float]:
    return await embeddings.embed_query(query)


@traceable(
    name="kb.hybrid_sql",
    run_type="retriever",
    process_inputs=lambda inputs: {
        "query_text": inputs["context"].normalized_query,
        "candidate_limit": inputs["candidate_limit"],
        "site_id": inputs.get("site_id"),
        "vector_dimensions": len(inputs["vector"]),
        "include_lexical": bool(inputs["context"].normalized_query),
    },
    process_outputs=lambda rows: {
        "row_count": len(rows),
        "vector_row_count": sum(1 for row in rows if row.get("match_source") == "vector"),
        "lexical_row_count": sum(1 for row in rows if row.get("match_source") == "lexical"),
    },
)
async def _trace_hybrid_sql(
    *,
    conn: AsyncConnection,
    context: SearchContext,
    vector: list[float],
    candidate_limit: int,
    site_id: str | None,
) -> list[dict]:
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            _hybrid_search_sql(site_id=site_id, include_lexical=bool(context.normalized_query)),
            _hybrid_search_params(
                vector=vector,
                query_text=context.normalized_query,
                candidate_limit=candidate_limit,
                site_id=site_id,
            ),
        )
        rows = await cur.fetchall()
    return [dict(row) for row in rows]


@traceable(
    name="kb.rerank_results",
    run_type="chain",
    process_inputs=lambda inputs: {
        "query_text": inputs["context"].normalized_query,
        "row_count": len(inputs["rows"]),
        "limit": inputs["limit"],
        "candidate_limit": inputs["candidate_limit"],
    },
    process_outputs=lambda hits: {
        "hit_count": len(hits),
        "top_score": hits[0].score if hits else None,
        "top_category": hits[0].category if hits else None,
        "top_source_url": hits[0].source_url if hits else None,
    },
)
async def _trace_rerank_results(
    *,
    context: SearchContext,
    rows: list[dict],
    limit: int,
    candidate_limit: int,
) -> list[KnowledgeHit]:
    vector_rows = []
    lexical_rows = []
    for row in rows:
        item = dict(row)
        match_source = item.pop("match_source", "")
        if match_source == "lexical":
            lexical_rows.append(item)
        else:
            vector_rows.append(item)

    vector_ranked = []
    for row in vector_rows:
        item = dict(row)
        confidence = _adjusted_score(context, item, float(item["score"]))
        vector_ranked.append((confidence, item, confidence))

    lexical_ranked = []
    for row in lexical_rows:
        item = dict(row)
        confidence = _adjusted_score(context, item, float(item["score"]))
        lexical_confidence = _confidence_with_lexical_score(
            confidence,
            float(item["lexical_score"]),
        )
        lexical_ranked.append((lexical_confidence, item, lexical_confidence))

    fused = _rrf_fuse(vector_ranked, lexical_ranked, limit=candidate_limit)
    reranked = [row for _, row in _diversify_categories(context, fused, limit)]
    return [_knowledge_hit_from_row(row) for row in reranked[:limit]]


def _knowledge_hit_from_row(row: dict) -> KnowledgeHit:
    return KnowledgeHit(
        id=str(row["id"]),
        site_id=row.get("site_id"),
        title=row["title"],
        section=row["section"],
        category=row["category"],
        content=row["content"],
        source_url=row["source_url"],
        language=row["language"],
        crawled_at=row["crawled_at"],
        valid_until=row["valid_until"],
        metadata=row.get("metadata") or {},
        score=float(row.get("score") or 0.0),
    )


def _rrf_fuse(
    vector_ranked: list[tuple[float, dict, float]],
    lexical_ranked: list[tuple[float, dict, float]],
    *,
    limit: int,
    k: int = 60,
) -> list[tuple[float, dict]]:
    fused: dict[str, tuple[float, dict, float | None]] = {}
    for ranked in (vector_ranked, lexical_ranked):
        ordered = sorted(ranked, key=lambda item: item[0], reverse=True)
        for rank, (_, row, confidence) in enumerate(ordered, start=1):
            row_id = _row_key(row)
            current_score, current_row, current_confidence = fused.get(row_id, (0.0, dict(row), None))
            fused[row_id] = (
                current_score + 1.0 / (k + rank),
                current_row,
                confidence if current_confidence is None else max(current_confidence, confidence),
            )

    results = []
    for fusion_score, row, confidence in fused.values():
        confidence = float(confidence or 0.0)
        row["score"] = confidence
        row["fusion_score"] = fusion_score
        results.append((confidence, row))
    results.sort(key=lambda item: (item[0], float(item[1].get("fusion_score") or 0.0)), reverse=True)
    return results[:limit]


def _row_key(row: dict) -> str:
    return str(row.get("id") or row.get("content_hash") or row.get("source_url") or row.get("content"))


def _search_text(row: dict) -> str:
    text = " ".join(
        str(row.get(key, ""))
        for key in ("title", "section", "category", "content", "source_url", "language")
    )
    metadata = row.get("metadata") or {}
    if metadata:
        text = f"{text} {json.dumps(_json_safe(metadata), ensure_ascii=False)}"
    return _normalize_text(text)


def _search_context(query: str) -> SearchContext:
    normalized_query = _normalize_text(query)
    return SearchContext(
        raw_query=query,
        normalized_query=normalized_query,
        query_tokens=frozenset(_lexical_tokens(normalized_query)),
        query_lower=query.lower(),
        wanted_categories=tuple(_wanted_categories(query)),
    )


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.lower())
    without_marks = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    asciiish = without_marks.replace("đ", "d")
    return " ".join(re.findall(r"[a-z0-9]+", asciiish))


def _lexical_score(query: str | SearchContext, row: dict) -> float:
    context = _coerce_search_context(query)
    if not context.normalized_query:
        return 0.0
    search_text = str(row.get("search_text") or _search_text(row))
    query_tokens = context.query_tokens
    search_tokens = _lexical_tokens(search_text)
    if not query_tokens or not search_tokens:
        return 0.0
    overlap = len(query_tokens & search_tokens)
    phrase_bonus = 2.0 if context.normalized_query in search_text else 0.0
    return float(overlap) + phrase_bonus


def _lexical_tokens(value: str) -> set[str]:
    return {token for token in value.split() if len(token) >= 2}


def _keyword_score(query: str | SearchContext, row: dict) -> float:
    context = _coerce_search_context(query)
    haystack = " ".join(
        str(row.get(key, "")) for key in ("title", "section", "category", "content", "source_url")
    ).lower()
    score = 0.0
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in context.query_lower for keyword in keywords):
            if row.get("category") == category:
                score += 0.45
            if any(keyword in haystack for keyword in keywords):
                score += 0.25
    for token in set(context.query_lower.split()):
        if len(token) >= 4 and token in haystack:
            score += 0.03
    return score


def _adjusted_score(query: str | SearchContext, row: dict, base_score: float) -> float:
    return base_score + _keyword_score(query, row) + _site_relevance_score(query, row)


def _confidence_with_lexical_score(confidence: float, lexical_score: float) -> float:
    lexical_boost = min(max(lexical_score, 0.0) * 0.08, 0.30)
    return confidence + lexical_boost


def _site_relevance_score(query: str | SearchContext, row: dict) -> float:
    context = _coerce_search_context(query)
    query_lower = context.query_lower
    haystack = " ".join(
        str(row.get(key, "")) for key in ("title", "section", "category", "content", "source_url")
    ).lower()
    source_url = str(row.get("source_url", "")).lower()
    profile = relevance_profile_for_text(str(row.get("site_id") or ""), f"{haystack} {source_url}")
    configured_aliases = tuple(str(alias).lower() for alias in (row.get("site_aliases") or ()))
    profile_aliases = profile.aliases if profile else ()
    aliases = configured_aliases + tuple(
        alias for alias in profile_aliases if alias not in configured_aliases
    )
    if not aliases:
        return 0.0

    score = 0.0
    off_topic_aliases = profile.off_topic_aliases if profile else ()
    url_markers = profile.url_markers if profile else ()

    site_domain_query = not _mentions_off_topic_alias(query_lower, off_topic_aliases)
    if site_domain_query:
        if any(marker in haystack for marker in aliases):
            score += 0.85
        if any(marker in source_url for marker in url_markers):
            score += 0.45
        if row.get("language") == "vi" and _looks_vietnamese(query_lower):
            score += 0.12
        if (
            profile
            and profile.offer_path_penalty
            and profile.offer_path_penalty in source_url
            and not any(marker in haystack for marker in aliases)
        ):
            score -= 0.35

        for marker in off_topic_aliases:
            if marker in haystack and not any(alias in haystack for alias in aliases):
                score -= 0.7
                break

    return score


def _mentions_off_topic_alias(query_lower: str, aliases: tuple[str, ...]) -> bool:
    return any(marker in query_lower for marker in aliases)


def _looks_vietnamese(value: str) -> bool:
    vietnamese_chars = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    return any(char in value for char in vietnamese_chars) or any(
        token in value for token in ("giá", "vé", "mở", "cửa", "lúc", "nào")
    )


def _diversify_categories(
    query: str | SearchContext,
    scored: list[tuple[float, dict]],
    limit: int,
) -> list[tuple[float, dict]]:
    context = _coerce_search_context(query)
    wanted = list(context.wanted_categories)
    if not wanted:
        return scored

    picked: list[tuple[float, dict]] = []
    picked_ids: set[str] = set()
    for category in wanted:
        match = next((item for item in scored if item[1].get("category") == category), None)
        if match:
            picked.append(match)
            picked_ids.add(str(match[1].get("id") or match[1].get("content_hash")))

    for item in scored:
        row_id = str(item[1].get("id") or item[1].get("content_hash"))
        if row_id in picked_ids:
            continue
        picked.append(item)
        picked_ids.add(row_id)
        if len(picked) >= max(limit, len(wanted)):
            break
    return picked + [item for item in scored if str(item[1].get("id") or item[1].get("content_hash")) not in picked_ids]


def _wanted_categories(query: str) -> list[str]:
    query_lower = query.lower()
    wanted = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            wanted.append(category)
    return wanted


def _coerce_search_context(query: str | SearchContext) -> SearchContext:
    if isinstance(query, SearchContext):
        return query
    return _search_context(query)


def _iso(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _metadata_for(chunk: IngestedChunk) -> dict:
    metadata = chunk.metadata or {}
    metadata.setdefault("ingestion", "vinwonders")
    return _json_safe(metadata)


def _json_safe(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def urlparse_root_host(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.lower() or "default"
