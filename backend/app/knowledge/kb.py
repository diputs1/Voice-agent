from __future__ import annotations

import json
import re
import unicodedata
from hashlib import sha256
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import urlparse

import psycopg
from psycopg.rows import dict_row
from langsmith import traceable

from app.knowledge.embeddings import EmbeddingProvider, pgvector_literal
from app.crawling.ingestion import IngestedChunk, seed_chunks
from app.core.sites import canonical_root_url, site_id_for_url


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
        query_vector = await self.embeddings.embed_query(query)
        vector_ranked = []
        lexical_ranked = []
        for row in self.rows:
            if site_id and row.get("site_id") != site_id:
                continue
            confidence = _adjusted_score(query, row, _cosine(query_vector, row["embedding"]))
            lexical = _lexical_score(query, row)
            vector_ranked.append((confidence, row, confidence))
            if lexical > 0:
                lexical_ranked.append((lexical, row, confidence))
        fused = _rrf_fuse(vector_ranked, lexical_ranked, limit=max(limit * 6, 20))
        scored = _diversify_categories(query, fused, limit)
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
        crawl_policy: dict | None = None,
    ) -> None:
        self.sites[site_id] = {
            "site_id": site_id,
            "root_url": root_url,
            "allowed_domains": allowed_domains,
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
    def __init__(self, database_url: str, embeddings: EmbeddingProvider) -> None:
        self.database_url = database_url
        self.embeddings = embeddings

    async def ensure_ready(self) -> None:
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sites (
                  site_id TEXT PRIMARY KEY,
                  root_url TEXT NOT NULL,
                  allowed_domains TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
                  crawl_policy JSONB NOT NULL DEFAULT '{}'::jsonb,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
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
            conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS site_id TEXT NOT NULL DEFAULT 'default'")
            conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS canonical_url TEXT")
            conn.execute("ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_content_hash_key")
            conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS search_text TEXT NOT NULL DEFAULT ''")
            conn.execute(
                """
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS search_vector tsvector
                GENERATED ALWAYS AS (to_tsvector('simple', search_text)) STORED
                """
            )
            conn.execute(
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
            conn.execute("ALTER TABLE thread_turns ADD COLUMN IF NOT EXISTS site_id TEXT")
            conn.execute(
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
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS documents_site_content_hash_idx ON documents (site_id, content_hash)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS documents_site_idx ON documents (site_id)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING ivfflat (embedding vector_cosine_ops)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS documents_metadata_idx ON documents USING gin (metadata)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS documents_search_vector_idx ON documents USING gin (search_vector)"
            )
            self._backfill_search_text(conn)

    async def upsert_chunks(self, chunks: list[IngestedChunk]) -> int:
        vectors = await self.embeddings.embed_documents([chunk.content for chunk in chunks])
        inserted = 0
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            for chunk, vector in zip(chunks, vectors, strict=True):
                chunk_site_id = chunk.site_id or site_id_for_url(chunk.source_url)
                canonical_url = (chunk.metadata or {}).get("canonical_url") or chunk.source_url
                self._upsert_site_for_chunk(conn, chunk_site_id, chunk.source_url)
                result = conn.execute(
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
                ).fetchone()
                if result:
                    inserted += 1
        return inserted

    @traceable(name="kb.postgres_search")
    async def search(self, query: str, limit: int = 5, site_id: str | None = None) -> list[KnowledgeHit]:
        vector = await self.embeddings.embed_query(query)
        query_text = _normalize_text(query)
        site_filter = "WHERE site_id = %s" if site_id else ""
        with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
            vector_rows = conn.execute(
                f"""
                SELECT id::text, site_id, title, section, category, content, source_url, language,
                       crawled_at::text, valid_until::text, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM documents
                {site_filter}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (pgvector_literal(vector), site_id, pgvector_literal(vector), max(limit * 6, 20))
                if site_id
                else (pgvector_literal(vector), pgvector_literal(vector), max(limit * 6, 20)),
            ).fetchall()
            lexical_rows = []
            if query_text:
                lexical_rows = conn.execute(
                    f"""
                    SELECT id::text, site_id, title, section, category, content, source_url, language,
                           crawled_at::text, valid_until::text, metadata,
                           ts_rank_cd(search_vector, websearch_to_tsquery('simple', %s)) AS lexical_score,
                           1 - (embedding <=> %s::vector) AS score
                    FROM documents
                    WHERE {'site_id = %s AND ' if site_id else ''}search_vector @@ websearch_to_tsquery('simple', %s)
                    ORDER BY lexical_score DESC
                    LIMIT %s
                    """,
                    (query_text, pgvector_literal(vector), site_id, query_text, max(limit * 6, 20))
                    if site_id
                    else (query_text, pgvector_literal(vector), query_text, max(limit * 6, 20)),
                ).fetchall()
        vector_ranked = []
        for row in vector_rows:
            item = dict(row)
            confidence = _adjusted_score(query, item, float(item["score"]))
            vector_ranked.append((confidence, item, confidence))
        lexical_ranked = []
        for row in lexical_rows:
            item = dict(row)
            confidence = _adjusted_score(query, item, float(item["score"]))
            lexical_ranked.append((float(item["lexical_score"]), item, confidence))
        fused = _rrf_fuse(vector_ranked, lexical_ranked, limit=max(limit * 6, 20))
        reranked = [row for _, row in _diversify_categories(query, fused, limit)]
        return [_knowledge_hit_from_row(row) for row in reranked[:limit]]

    async def save_thread_turn(
        self, thread_id: str, transcript: str, answer: str, site_id: str | None = None
    ) -> None:
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            conn.execute(
                "INSERT INTO thread_turns (thread_id, site_id, transcript, answer) VALUES (%s, %s, %s, %s)",
                (thread_id, site_id, transcript, answer),
            )

    async def get_thread(self, thread_id: str) -> list[dict]:
        with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
            return list(
                conn.execute(
                    """
                    SELECT site_id, transcript, answer, created_at::text
                    FROM thread_turns
                    WHERE thread_id = %s
                    ORDER BY created_at ASC, id ASC
                    """,
                    (thread_id,),
                ).fetchall()
            )

    async def doc_set_hash(self, site_id: str | None = None) -> str:
        with psycopg.connect(self.database_url) as conn:
            value = conn.execute(
                f"""
                SELECT md5(coalesce(string_agg(content_hash, ',' ORDER BY content_hash), ''))
                FROM documents
                {'WHERE site_id = %s' if site_id else ''}
                """
                ,
                (site_id,) if site_id else (),
            ).fetchone()[0]
        return str(value)

    async def upsert_site(
        self,
        *,
        site_id: str,
        root_url: str,
        allowed_domains: list[str],
        crawl_policy: dict | None = None,
    ) -> None:
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            conn.execute(
                """
                INSERT INTO sites (site_id, root_url, allowed_domains, crawl_policy)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (site_id) DO UPDATE SET
                  root_url = EXCLUDED.root_url,
                  allowed_domains = EXCLUDED.allowed_domains,
                  crawl_policy = EXCLUDED.crawl_policy,
                  updated_at = now()
                """,
                (
                    site_id,
                    root_url,
                    allowed_domains,
                    json.dumps(_json_safe(crawl_policy or {}), ensure_ascii=False),
                ),
            )

    async def site_status(self, site_id: str | None = None) -> dict:
        with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
            site_rows = conn.execute(
                """
                SELECT site_id, root_url, allowed_domains, crawl_policy, created_at::text, updated_at::text
                FROM sites
                WHERE (%s::text IS NULL OR site_id = %s)
                ORDER BY updated_at DESC
                """,
                (site_id, site_id),
            ).fetchall()
            category_rows = conn.execute(
                """
                SELECT category, count(*) AS count
                FROM documents
                WHERE (%s::text IS NULL OR site_id = %s)
                GROUP BY category
                ORDER BY count DESC, category ASC
                """,
                (site_id, site_id),
            ).fetchall()
            doc_count = conn.execute(
                """
                SELECT count(*) AS count, max(crawled_at)::text AS latest_crawl
                FROM documents
                WHERE (%s::text IS NULL OR site_id = %s)
                """,
                (site_id, site_id),
            ).fetchone()
            stale_count = conn.execute(
                """
                SELECT count(*) AS count
                FROM documents
                WHERE (%s::text IS NULL OR site_id = %s)
                  AND valid_until IS NOT NULL
                  AND valid_until < now()
                """,
                (site_id, site_id),
            ).fetchone()
        return {
            "site_id": site_id,
            "sites": [dict(row) for row in site_rows],
            "document_count": int(doc_count["count"] or 0),
            "latest_crawl": doc_count["latest_crawl"],
            "category_counts": {row["category"]: int(row["count"]) for row in category_rows},
            "stale_count": int(stale_count["count"] or 0),
        }

    def _backfill_search_text(self, conn) -> None:
        with conn.cursor(row_factory=dict_row) as cur:
            rows = cur.execute(
                """
                SELECT id::text, title, section, category, content, source_url, language, metadata
                FROM documents
                WHERE search_text = ''
                """
            ).fetchall()
        for row in rows:
            conn.execute(
                "UPDATE documents SET search_text = %s WHERE id = %s::uuid",
                (_search_text(dict(row)), row["id"]),
            )

    def _upsert_site_for_chunk(self, conn, site_id: str, source_url: str) -> None:
        root_url = canonical_root_url(source_url)
        conn.execute(
            """
            INSERT INTO sites (site_id, root_url, allowed_domains, crawl_policy)
            VALUES (%s, %s, %s, '{}'::jsonb)
            ON CONFLICT (site_id) DO NOTHING
            """,
            (site_id, root_url, [urlparse_root_host(source_url)]),
        )


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


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
    fused: dict[str, tuple[float, dict, float]] = {}
    for ranked in (vector_ranked, lexical_ranked):
        ordered = sorted(ranked, key=lambda item: item[0], reverse=True)
        for rank, (_, row, confidence) in enumerate(ordered, start=1):
            row_id = _row_key(row)
            current_score, current_row, current_confidence = fused.get(row_id, (0.0, dict(row), 0.0))
            fused[row_id] = (
                current_score + 1.0 / (k + rank),
                current_row,
                max(current_confidence, confidence),
            )

    results = []
    for fusion_score, row, confidence in fused.values():
        row["score"] = confidence
        row["fusion_score"] = fusion_score
        results.append((fusion_score, row))
    results.sort(key=lambda item: item[0], reverse=True)
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


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.lower())
    without_marks = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    asciiish = without_marks.replace("đ", "d")
    return " ".join(re.findall(r"[a-z0-9]+", asciiish))


def _lexical_score(query: str, row: dict) -> float:
    query_text = _normalize_text(query)
    if not query_text:
        return 0.0
    search_text = str(row.get("search_text") or _search_text(row))
    query_tokens = _lexical_tokens(query_text)
    search_tokens = _lexical_tokens(search_text)
    if not query_tokens or not search_tokens:
        return 0.0
    overlap = len(query_tokens & search_tokens)
    phrase_bonus = 2.0 if query_text in search_text else 0.0
    return float(overlap) + phrase_bonus


def _lexical_tokens(value: str) -> set[str]:
    return {token for token in value.split() if len(token) >= 2}


def _keyword_score(query: str, row: dict) -> float:
    haystack = " ".join(
        str(row.get(key, "")) for key in ("title", "section", "category", "content", "source_url")
    ).lower()
    query_lower = query.lower()
    score = 0.0
    category_boosts = {
        "affiliate": ("affiliate", "hoa hồng", "đối tác"),
        "vinclub": ("vinclub", "hội viên", "gold", "platinum", "diamond"),
        "offer": ("ưu đãi", "voucher", "khuyến mãi", "giảm", "hạn áp dụng"),
        "price": ("giá", "giá vé", "bảng giá", "vnđ", "vnd"),
        "booking": ("booking", "đặt vé", "đặt online", "qr"),
        "schedule": ("giờ", "mở cửa", "lịch", "lúc nào", "thời gian"),
        "experience": ("show", "biểu diễn", "động vật", "animal", "night safari", "kid zoo"),
    }
    for category, keywords in category_boosts.items():
        if any(keyword in query_lower for keyword in keywords):
            if row.get("category") == category:
                score += 0.45
            if any(keyword in haystack for keyword in keywords):
                score += 0.25
    for token in set(query_lower.split()):
        if len(token) >= 4 and token in haystack:
            score += 0.03
    return score


def _adjusted_score(query: str, row: dict, base_score: float) -> float:
    return base_score + _keyword_score(query, row) + _safari_relevance_score(query, row)


def _safari_relevance_score(query: str, row: dict) -> float:
    query_lower = query.lower()
    haystack = " ".join(
        str(row.get(key, "")) for key in ("title", "section", "category", "content", "source_url")
    ).lower()
    source_url = str(row.get("source_url", "")).lower()
    score = 0.0

    safari_domain_query = not _mentions_other_destination(query_lower)
    if safari_domain_query:
        safari_markers = (
            "vinpearl safari phú quốc",
            "vinpearl safari phu quoc",
            "vinpearl-safari-phu-quoc",
        )
        if any(marker in haystack for marker in safari_markers):
            score += 0.85
        if "/vinpearl-safari-phu-quoc/" in source_url:
            score += 0.45
        if row.get("language") == "vi" and _looks_vietnamese(query_lower):
            score += 0.12
        if "/vi/uu-dai/" in source_url and not any(marker in haystack for marker in safari_markers):
            score -= 0.35

        off_topic_markers = (
            "nha trang",
            "vũ yên",
            "vu yen",
            "hà nội",
            "ha noi",
            "ocean city",
            "horse academy",
            "đất nước thiên hùng ca",
            "grand world",
            "vinwonders phú quốc",
            "vinwonders phu quoc",
        )
        for marker in off_topic_markers:
            if marker in haystack and not any(safari in haystack for safari in safari_markers):
                score -= 0.7
                break

    return score


def _mentions_other_destination(query_lower: str) -> bool:
    return any(
        marker in query_lower
        for marker in (
            "nha trang",
            "vũ yên",
            "vu yen",
            "hà nội",
            "ha noi",
            "grand world",
            "vinwonders phú quốc",
            "vinwonders phu quoc",
        )
    )


def _looks_vietnamese(value: str) -> bool:
    vietnamese_chars = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    return any(char in value for char in vietnamese_chars) or any(
        token in value for token in ("giá", "vé", "mở", "cửa", "lúc", "nào")
    )


def _diversify_categories(query: str, scored: list[tuple[float, dict]], limit: int) -> list[tuple[float, dict]]:
    wanted = _wanted_categories(query)
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
    checks = [
        ("affiliate", ("affiliate", "hoa hồng", "đối tác")),
        ("vinclub", ("vinclub", "hội viên", "gold", "platinum", "diamond")),
        ("offer", ("ưu đãi", "voucher", "khuyến mãi", "giảm")),
        ("price", ("giá", "giá vé", "bảng giá")),
        ("booking", ("booking", "đặt vé", "đặt online")),
    ]
    for category, keywords in checks:
        if any(keyword in query_lower for keyword in keywords):
            wanted.append(category)
    return wanted


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
