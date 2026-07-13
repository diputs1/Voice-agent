import pytest

from app.knowledge.embeddings import EmbeddingProvider
from app.crawling.ingestion import IngestedChunk
from app.knowledge.kb import (
    InMemoryKnowledgeBase,
    PostgresKnowledgeBase,
    _adjusted_score,
    _diversify_categories,
    _normalize_text,
    _rrf_fuse,
    _wanted_categories,
)

from datetime import UTC, datetime


def test_safari_price_chunk_beats_off_topic_price_chunk():
    query = "Giá vé hôm nay bao nhiêu tiền?"
    safari_row = {
        "title": "[MỚI] Bảng giá vé Vinpearl Safari Phú Quốc",
        "section": "Bảng giá vé Vinpearl Safari Phú Quốc mới nhất",
        "category": "price",
        "content": "Giá vé Vinpearl Safari Phú Quốc từ 710.000 VND đến 950.000 VND.",
        "source_url": "https://vinwonders.com/vi/wonderpedia/news/gia-ve-vinpearl-safari-phu-quoc/",
        "language": "vi",
    }
    nha_trang_row = {
        "title": "Ưu đãi ẩm thực VinWonders Nha Trang",
        "section": "Combo Vé Vào Cửa VinWonders Nha Trang",
        "category": "price",
        "content": "Combo buffet và vé vào cửa VinWonders Nha Trang giá 620.000 VNĐ.",
        "source_url": "https://vinwonders.com/vi/uu-dai/combo-am-thuc-vinwonders-nha-trang/",
        "language": "vi",
    }

    assert _adjusted_score(query, safari_row, 0.1) > _adjusted_score(query, nha_trang_row, 0.4)


def test_safari_show_chunk_beats_other_show_chunk():
    query = "Show biểu diễn động vật diễn ra lúc nào?"
    safari_row = {
        "title": "Vinpearl Safari Phú Quốc | Official Website",
        "section": "Must-see events",
        "category": "schedule",
        "content": "Animal Show 10:00 - 10:30 | 14:00 - 14:30.",
        "source_url": "https://vinwonders.com/vi/vinpearl-safari-phu-quoc/",
        "language": "vi",
    }
    other_show_row = {
        "title": "Ưu đãi mở bán show Đất Nước Thiên Hùng Ca",
        "section": "Show diễn Đất Nước Thiên Hùng Ca",
        "category": "booking",
        "content": "Show diễn ra tại Vinpearl Theatre Ocean City.",
        "source_url": "https://vinwonders.com/vi/uu-dai/mo-ban-show-dat-nuoc-thien-hung-ca/",
        "language": "vi",
    }

    assert _adjusted_score(query, safari_row, 0.1) > _adjusted_score(query, other_show_row, 0.4)


def test_site_aliases_boost_matching_site_content():
    row = {
        "site_id": "custom-site",
        "site_aliases": ["custom park"],
        "title": "Custom Park",
        "section": "Bảng giá",
        "category": "price",
        "content": "Giá vé Custom Park áp dụng trong ngày.",
        "source_url": "https://example.test/custom-park/",
        "language": "vi",
    }

    assert _adjusted_score("giá vé custom park", row, 0.1) > 1.0


def test_diversify_categories_does_not_duplicate_extra_picks():
    rows = [
        (1.0, {"id": "price-1", "category": "price"}),
        (0.9, {"id": "offer-1", "category": "offer"}),
        (0.8, {"id": "price-2", "category": "price"}),
    ]

    picked = _diversify_categories("giá vé", rows, 2)
    ids = [row["id"] for _, row in picked]

    assert ids == ["price-1", "offer-1", "price-2"]


def test_product_query_prioritizes_booking_and_experience_categories():
    rows = [
        (0.95, {"id": "overview-1", "category": "overview"}),
        (0.40, {"id": "booking-1", "category": "booking"}),
        (0.35, {"id": "experience-1", "category": "experience"}),
    ]

    picked = _diversify_categories("Các sản phẩm của VinSafari Phú Quốc", rows, 3)
    ids = [row["id"] for _, row in picked[:3]]

    assert _wanted_categories("Các sản phẩm của VinSafari Phú Quốc") == ["booking", "experience"]
    assert ids == ["booking-1", "experience-1", "overview-1"]


def test_normalize_text_strips_vietnamese_diacritics():
    assert _normalize_text("Cầu Hôn Phú Quốc") == "cau hon phu quoc"


def test_rrf_fuse_deduplicates_and_preserves_confidence_score():
    row = {"id": "same", "title": "Cầu Hôn", "score": 0.2}
    fused = _rrf_fuse(
        [(0.9, {"id": "vector-only", "score": 0.9}, 0.9), (0.2, row, 0.2)],
        [(2.0, row, 0.2)],
        limit=10,
    )

    ids = [item[1]["id"] for item in fused]

    assert ids.count("same") == 1
    assert next(item[1] for item in fused if item[1]["id"] == "same")["score"] == 0.2


def test_rrf_fuse_orders_by_adjusted_confidence_before_fusion_score():
    high_confidence = {"id": "safari-price", "category": "price", "score": 0.85}
    low_confidence = {"id": "off-topic-price", "category": "price", "score": 0.1}

    fused = _rrf_fuse(
        [(0.85, high_confidence, 0.85), (0.1, low_confidence, 0.1)],
        [(0.1, low_confidence, 0.1)],
        limit=10,
    )

    assert [row["id"] for _, row in fused[:2]] == ["safari-price", "off-topic-price"]
    assert fused[0][0] == fused[0][1]["score"] == 0.85
    assert fused[1][1]["fusion_score"] > fused[0][1]["fusion_score"]


@pytest.mark.asyncio
async def test_in_memory_hybrid_search_matches_unaccented_exact_keyword():
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    now = datetime.now(UTC)
    await kb.upsert_chunks(
        [
            IngestedChunk(
                title="Cầu Hôn Phú Quốc",
                section="Điểm tham quan",
                category="experience",
                content="Thông tin về Cầu Hôn gần khu vực Phú Quốc.",
                source_url="https://vinwonders.com/vi/wonderpedia/news/cau-hon-phu-quoc/",
                content_hash="cau-hon",
                crawled_at=now,
                valid_until=None,
            ),
            IngestedChunk(
                title="Ẩm thực Nha Trang",
                section="Combo",
                category="offer",
                content="Ưu đãi ẩm thực tại Nha Trang.",
                source_url="https://vinwonders.com/vi/uu-dai/nha-trang/",
                content_hash="nha-trang",
                crawled_at=now,
                valid_until=None,
            ),
        ]
    )

    hits = await kb.search("cau hon", limit=2)

    assert hits[0].id == "cau-hon"


@pytest.mark.asyncio
async def test_postgres_ensure_ready_creates_thread_lookup_index():
    pool = FakePool()
    kb = PostgresKnowledgeBase(
        "postgresql://example",
        EmbeddingProvider(None, "text-embedding-3-small"),
        pool=pool,
    )

    await kb.ensure_ready()

    assert any("thread_turns_thread_created_idx" in statement for statement in pool.statements)


class FakePool:
    def __init__(self) -> None:
        self.connection_obj = FakeConnection()
        self.statements = self.connection_obj.statements

    def connection(self):
        return FakeConnectionContext(self.connection_obj)


class FakeConnectionContext:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class FakeConnection:
    def __init__(self) -> None:
        self.statements = []

    async def execute(self, statement, params=None):
        del params
        self.statements.append(str(statement))
        return FakeCursor()

    def cursor(self, row_factory=None):
        del row_factory
        return FakeCursorContext()


class FakeCursorContext:
    async def __aenter__(self):
        return FakeCursor()

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class FakeCursor:
    async def execute(self, statement, params=None):
        del statement, params

    async def fetchall(self):
        return []

    async def fetchone(self):
        return {"count": 0}
